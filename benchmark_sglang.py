"""
SGLang serving benchmark -- a third engine alongside vLLM and the naive HF baseline.

Mirrors benchmark_vllm.py: same prompts.py, same batch sizes, same length
buckets, same max_new_tokens, same temperature/top_p, fixed seed, and the same
output JSON schema (written to results/sglang_results.json). Uses SGLang's
offline Engine API (`sglang.Engine`), not the HTTP server, for the same reason
the vLLM script uses `vllm.LLM`: no HTTP/JSON overhead in the measured path.

Version and GPU notes (see requirements-sglang.txt for the full reasoning):
  * Pinned to sglang 0.4.6.post5. Newer SGLang dropped Turing (sm75, i.e. the
    Colab T4): sgl-kernel stopped building sm75 in Aug 2025 and the docs now
    list sm80 (Ampere) as the minimum. 0.4.7+ also moved to torch 2.7.1 /
    triton 3.3.x, which has a known fp16 codegen bug on compute capability < 8.0.
  * On a GPU below sm80 this script defaults to the Triton attention backend
    and the PyTorch sampling backend, the SGLang-documented fallback for
    FlashInfer problems on T4-class cards. Override with the CLI flags.

Measurement differences vs. benchmark_vllm.py, stated up front:
  * Peak memory: SGLang runs the model in a separate scheduler process, so
    torch.cuda.max_memory_allocated() in this process would read ~0. Peak
    memory is instead sampled from NVML (device-level used memory, polled
    every 20 ms during the run). It includes the CUDA context (a few hundred
    MB) that torch's allocator counter does not, so compare shapes, not the
    last decimal.
  * Seed: the pinned SGLang has no per-request sampling seed, only an
    engine-wide `random_seed`. Runs are reproducible run-to-run for the same
    script, but the RNG stream is shared across batches. Token counts are
    recorded per request so throughput is only compared on equal work.
  * TTFB: the offline Engine's non-streaming result carries no first-token
    timestamp. Throughput is measured on a non-streaming batched call (same
    shape as vLLM's llm.generate). TTFB comes from a separate streaming pass
    over the same batch, timed client-side, so streaming overhead never
    touches the throughput number. Disable with --no-ttfb-pass.

Usage:
    python benchmark_sglang.py --model Qwen/Qwen2.5-3B-Instruct --output results/sglang_results.json
"""

import argparse
import asyncio
import json
import threading
import time
from dataclasses import dataclass, asdict, field

import numpy as np

from prompts import get_prompt_batch

BATCH_SIZES = [1, 4, 8, 16, 32]
LENGTH_BUCKETS = ["short", "medium", "long"]
MAX_NEW_TOKENS = 256
TEMPERATURE = 0.7
# vLLM's SamplingParams default top_p is 1.0, and benchmark_vllm.py does not
# override it, so 1.0 here keeps the sampling distribution identical.
TOP_P = 1.0
# Same fraction benchmark_vllm.py hands vLLM (gpu_memory_utilization=0.85).
# Semantics differ slightly: SGLang's mem_fraction_static covers weights +
# KV pool, vLLM's covers everything it may allocate.
MEM_FRACTION_STATIC = 0.85
NVML_POLL_SEC = 0.02


@dataclass
class RunResult:
    batch_size: int
    length_bucket: str
    total_prompts: int
    wall_clock_sec: float
    total_tokens_generated: int
    throughput_tokens_per_sec: float
    ttfb_p50_ms: float
    ttfb_p90_ms: float
    ttfb_p99_ms: float
    per_token_latency_p50_ms: float
    peak_gpu_mem_gb: float
    tokens_per_request: list[int] = field(default_factory=list)


def sampling_params(ignore_eos: bool = False) -> dict:
    """SGLang sampling params matching benchmark_vllm.py's SamplingParams."""
    return {
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "max_new_tokens": MAX_NEW_TOKENS,
        "ignore_eos": ignore_eos,
    }


def percentiles_ms(values_ms: list[float]) -> tuple[float, float, float]:
    vals = values_ms or [0.0]
    return tuple(round(float(np.percentile(vals, p)), 2) for p in (50, 90, 99))


def build_run_result(
    batch_size: int,
    length_bucket: str,
    tokens_per_request: list[int],
    elapsed: float,
    ttfbs_ms: list[float],
    peak_mem_gb: float,
) -> RunResult:
    """Pure function: turns raw measurements into a RunResult (same math as vLLM)."""
    total_tokens = int(sum(tokens_per_request))
    p50, p90, p99 = percentiles_ms(ttfbs_ms)
    per_token_ms = (elapsed * 1000) / max(total_tokens, 1) * batch_size
    return RunResult(
        batch_size=batch_size,
        length_bucket=length_bucket,
        total_prompts=len(tokens_per_request),
        wall_clock_sec=round(elapsed, 4),
        total_tokens_generated=total_tokens,
        throughput_tokens_per_sec=round(total_tokens / elapsed, 2) if elapsed > 0 else 0.0,
        ttfb_p50_ms=p50,
        ttfb_p90_ms=p90,
        ttfb_p99_ms=p99,
        per_token_latency_p50_ms=round(per_token_ms, 3),
        peak_gpu_mem_gb=round(peak_mem_gb, 2),
        tokens_per_request=[int(t) for t in tokens_per_request],
    )


def build_output(model: str, seed: int, results: list[RunResult], meta: dict | None = None) -> dict:
    """Top-level JSON, same keys as vllm_results.json plus SGLang run metadata."""
    out = {
        "engine": "sglang",
        "model": model,
        "max_new_tokens": MAX_NEW_TOKENS,
        "seed": seed,
        "runs": [asdict(r) for r in results],
    }
    if meta:
        out.update(meta)
    return out


class NvmlPeakSampler:
    """Polls NVML device memory in a background thread and keeps the max."""

    def __init__(self, device_index: int = 0):
        import pynvml

        self._nvml = pynvml
        pynvml.nvmlInit()
        self._handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        self._peak = 0
        self._stop = threading.Event()
        self._thread = None

    def compute_capability(self) -> tuple[int, int]:
        return tuple(self._nvml.nvmlDeviceGetCudaComputeCapability(self._handle))

    def device_name(self) -> str:
        name = self._nvml.nvmlDeviceGetName(self._handle)
        return name.decode() if isinstance(name, bytes) else name

    def _used(self) -> int:
        return self._nvml.nvmlDeviceGetMemoryInfo(self._handle).used

    def _loop(self):
        while not self._stop.is_set():
            self._peak = max(self._peak, self._used())
            time.sleep(NVML_POLL_SEC)

    def __enter__(self):
        self._peak = self._used()
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()
        self._peak = max(self._peak, self._used())
        return False

    @property
    def peak_gb(self) -> float:
        return self._peak / (1024**3)


async def _stream_ttfb_ms(engine, prompt: str, sp: dict) -> float:
    start = time.perf_counter()
    first = None
    gen = await engine.async_generate(prompt=prompt, sampling_params=sp, stream=True)
    async for _chunk in gen:
        if first is None:
            first = time.perf_counter()
    return ((first or time.perf_counter()) - start) * 1000


def measure_ttfb(engine, prompts: list[str], sp: dict) -> list[float]:
    """All requests submitted concurrently (as one batch would be), first chunk timed per request."""
    loop = asyncio.get_event_loop()

    async def _all():
        return await asyncio.gather(*[_stream_ttfb_ms(engine, p, sp) for p in prompts])

    return list(loop.run_until_complete(_all()))


def run_sglang_benchmark(
    model_name: str,
    seed: int = 42,
    attention_backend: str | None = None,
    sampling_backend: str | None = None,
    ttfb_pass: bool = True,
    ignore_eos: bool = False,
) -> tuple[list[RunResult], dict]:
    # Imported here so the rest of the script (schema, analysis, tests) can be
    # inspected on machines without sglang or a GPU, same as benchmark_vllm.py.
    import sglang as sgl

    sampler = NvmlPeakSampler()
    cc = sampler.compute_capability()
    if cc < (8, 0):
        # SGLang-documented workaround for FlashInfer issues on T4-class GPUs.
        attention_backend = attention_backend or "triton"
        sampling_backend = sampling_backend or "pytorch"

    engine_kwargs = dict(
        model_path=model_name,
        dtype="float16",
        mem_fraction_static=MEM_FRACTION_STATIC,
        random_seed=seed,
    )
    if attention_backend:
        engine_kwargs["attention_backend"] = attention_backend
    if sampling_backend:
        engine_kwargs["sampling_backend"] = sampling_backend

    print(
        f"Loading {model_name} with SGLang {sgl.__version__} (seed={seed}, "
        f"gpu={sampler.device_name()} sm{cc[0]}{cc[1]}, "
        f"attention={attention_backend or 'default'}, sampling={sampling_backend or 'default'})..."
    )
    engine = sgl.Engine(**engine_kwargs)
    sp = sampling_params(ignore_eos)
    results = []

    try:
        # One untimed warmup so CUDA graph capture / Triton JIT does not land
        # inside the first measured run.
        engine.generate(prompt=get_prompt_batch(1, "short"), sampling_params=sp)

        for length_bucket in LENGTH_BUCKETS:
            for batch_size in BATCH_SIZES:
                prompts = get_prompt_batch(batch_size, length_bucket)

                with sampler:
                    start = time.perf_counter()
                    outputs = engine.generate(prompt=prompts, sampling_params=sp)
                    elapsed = time.perf_counter() - start

                if isinstance(outputs, dict):
                    outputs = [outputs]
                tokens = [int(o["meta_info"]["completion_tokens"]) for o in outputs]

                ttfbs_ms = measure_ttfb(engine, prompts, sp) if ttfb_pass else []

                result = build_run_result(
                    batch_size, length_bucket, tokens, elapsed, ttfbs_ms, sampler.peak_gb
                )
                results.append(result)
                print(
                    f"[SGLang] batch={batch_size:2d} bucket={length_bucket:6s} "
                    f"throughput={result.throughput_tokens_per_sec:>8.1f} tok/s "
                    f"tokens={result.total_tokens_generated:5d} "
                    f"peak_mem={result.peak_gpu_mem_gb:.2f}GB"
                )
    finally:
        engine.shutdown()

    meta = {
        "sglang_version": sgl.__version__,
        "gpu": sampler.device_name(),
        "compute_capability": f"{cc[0]}.{cc[1]}",
        "attention_backend": attention_backend or "default",
        "sampling_backend": sampling_backend or "default",
        "mem_fraction_static": MEM_FRACTION_STATIC,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "ignore_eos": ignore_eos,
        "peak_mem_method": "nvml_device_used",
        "ttfb_method": "separate_streaming_pass_client_side" if ttfb_pass else "not_measured",
    }
    return results, meta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--output", default="results/sglang_results.json")
    parser.add_argument("--seed", type=int, default=42,
                        help="Fixed seed, same default as the vLLM and baseline scripts.")
    parser.add_argument("--attention-backend", default=None,
                        help="Override SGLang attention backend (default: triton below sm80, else SGLang's default).")
    parser.add_argument("--sampling-backend", default=None,
                        help="Override SGLang sampling backend (default: pytorch below sm80, else SGLang's default).")
    parser.add_argument("--no-ttfb-pass", action="store_true",
                        help="Skip the separate streaming pass used to measure TTFB.")
    parser.add_argument("--ignore-eos", action="store_true",
                        help="Force every request to max_new_tokens (strict equal work). Off by default to mirror vLLM.")
    args = parser.parse_args()

    results, meta = run_sglang_benchmark(
        args.model,
        args.seed,
        attention_backend=args.attention_backend,
        sampling_backend=args.sampling_backend,
        ttfb_pass=not args.no_ttfb_pass,
        ignore_eos=args.ignore_eos,
    )
    output = build_output(args.model, args.seed, results, meta)

    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved {len(results)} runs to {args.output}")


if __name__ == "__main__":
    main()
