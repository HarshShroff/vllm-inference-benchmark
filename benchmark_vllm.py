"""
vLLM serving benchmark.

Measures throughput (tokens/sec), latency (TTFB + per-token P50/P90/P99),
and peak GPU memory across a sweep of batch sizes, using vLLM's engine
directly (not the OpenAI-compatible server) for precise control over batching.

Usage:
    python benchmark_vllm.py --model Qwen/Qwen2.5-7B-Instruct --output results/vllm_results.json
"""

import argparse
import json
import time
from dataclasses import dataclass, asdict, field

import numpy as np
import torch

from prompts import get_prompt_batch

BATCH_SIZES = [1, 4, 8, 16, 32]
LENGTH_BUCKETS = ["short", "medium", "long"]
MAX_NEW_TOKENS = 256


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


def run_vllm_benchmark(model_name: str, seed: int = 42) -> list[RunResult]:
    # Imported here so the rest of the script (prompts, analysis) can be
    # inspected/tested on machines without vllm installed.
    from vllm import LLM, SamplingParams

    print(f"Loading {model_name} with vLLM (seed={seed})...")
    llm = LLM(model=model_name, gpu_memory_utilization=0.85, dtype="float16", seed=seed)
    results = []

    for length_bucket in LENGTH_BUCKETS:
        for batch_size in BATCH_SIZES:
            prompts = get_prompt_batch(batch_size, length_bucket)
            sampling_params = SamplingParams(
                temperature=0.7, max_tokens=MAX_NEW_TOKENS, seed=seed
            )

            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

            # vLLM's engine reports per-request timing via RequestOutput metrics
            # when available; we also wall-clock the whole batch for throughput.
            start = time.perf_counter()
            outputs = llm.generate(prompts, sampling_params)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start

            tokens_per_request = [len(o.outputs[0].token_ids) for o in outputs]
            total_tokens = sum(tokens_per_request)
            peak_mem_gb = torch.cuda.max_memory_allocated() / (1024**3)

            # vLLM RequestOutput exposes .metrics with first_token_time /
            # arrival_time on recent versions; fall back gracefully if absent.
            ttfbs_ms = []
            for o in outputs:
                m = getattr(o, "metrics", None)
                if m and getattr(m, "first_token_time", None) and getattr(m, "arrival_time", None):
                    ttfbs_ms.append((m.first_token_time - m.arrival_time) * 1000)
            if not ttfbs_ms:
                ttfbs_ms = [0.0]

            per_token_ms = (elapsed * 1000) / max(total_tokens, 1) * batch_size

            result = RunResult(
                batch_size=batch_size,
                length_bucket=length_bucket,
                total_prompts=len(prompts),
                wall_clock_sec=round(elapsed, 4),
                total_tokens_generated=total_tokens,
                throughput_tokens_per_sec=round(total_tokens / elapsed, 2),
                ttfb_p50_ms=round(float(np.percentile(ttfbs_ms, 50)), 2),
                ttfb_p90_ms=round(float(np.percentile(ttfbs_ms, 90)), 2),
                ttfb_p99_ms=round(float(np.percentile(ttfbs_ms, 99)), 2),
                per_token_latency_p50_ms=round(per_token_ms, 3),
                peak_gpu_mem_gb=round(peak_mem_gb, 2),
                tokens_per_request=tokens_per_request,
            )
            results.append(result)
            print(
                f"[vLLM] batch={batch_size:2d} bucket={length_bucket:6s} "
                f"throughput={result.throughput_tokens_per_sec:>8.1f} tok/s "
                f"peak_mem={result.peak_gpu_mem_gb:.2f}GB"
            )

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--output", default="results/vllm_results.json")
    parser.add_argument("--seed", type=int, default=42,
                         help="Fixed seed for reproducible, matched-workload comparison "
                              "against the baseline script (same default seed).")
    args = parser.parse_args()

    results = run_vllm_benchmark(args.model, args.seed)

    output = {
        "engine": "vllm",
        "model": args.model,
        "max_new_tokens": MAX_NEW_TOKENS,
        "runs": [asdict(r) for r in results],
    }

    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved {len(results)} runs to {args.output}")


if __name__ == "__main__":
    main()
