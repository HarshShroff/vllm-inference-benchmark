"""
Naive HuggingFace `transformers.generate()` baseline.

This intentionally does NOT use continuous batching, PagedAttention, or any
serving-layer optimization -- it's the "just call .generate() in a loop"
approach many people start with, used here as the comparison point that
makes vLLM's throughput/memory gains legible and quantifiable.

Usage:
    python benchmark_baseline.py --model Qwen/Qwen2.5-7B-Instruct --output results/baseline_results.json
"""

import argparse
import json
import time
from dataclasses import dataclass, asdict

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

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
    peak_gpu_mem_gb: float


def run_baseline_benchmark(model_name: str, seed: int = 42) -> list[RunResult]:
    print(f"Loading {model_name} with naive HF generate() (seed={seed})...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Decoder-only models require left-padding for correct batched generation --
    # right-padding (the HF default) silently corrupts results for any prompt
    # shorter than the longest in the batch. Caught via CPU dry-run before
    # this ever touched real GPU results.
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.float16, device_map="cuda"
    )
    model.eval()
    results = []

    for length_bucket in LENGTH_BUCKETS:
        for batch_size in BATCH_SIZES:
            prompts = get_prompt_batch(batch_size, length_bucket)

            inputs = tokenizer(
                prompts, return_tensors="pt", padding=True, truncation=True
            ).to("cuda")

            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

            # Reset the seed before every batch (not just once at script start)
            # so batch-size and length-bucket ordering can't drift the RNG
            # state between runs -- each run gets an identical sampling seed.
            torch.manual_seed(seed)

            # Naive baseline: single batched .generate() call, no continuous
            # batching or request-level streaming -- this is the point of
            # comparison, not a limitation of the harness.
            ttfb_start = time.perf_counter()
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=tokenizer.eos_token_id,
                )
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - ttfb_start

            new_tokens = output_ids[:, inputs["input_ids"].shape[1]:]
            total_tokens = int((new_tokens != tokenizer.pad_token_id).sum().item())
            peak_mem_gb = torch.cuda.max_memory_allocated() / (1024**3)

            # No true per-request TTFB without streaming; approximate as
            # time-to-first-decode-step scaled by batch (documented limitation).
            approx_ttfb_ms = (elapsed / MAX_NEW_TOKENS) * 1000

            result = RunResult(
                batch_size=batch_size,
                length_bucket=length_bucket,
                total_prompts=len(prompts),
                wall_clock_sec=round(elapsed, 4),
                total_tokens_generated=total_tokens,
                throughput_tokens_per_sec=round(total_tokens / elapsed, 2),
                ttfb_p50_ms=round(approx_ttfb_ms, 2),
                peak_gpu_mem_gb=round(peak_mem_gb, 2),
            )
            results.append(result)
            print(
                f"[Baseline] batch={batch_size:2d} bucket={length_bucket:6s} "
                f"throughput={result.throughput_tokens_per_sec:>8.1f} tok/s "
                f"peak_mem={result.peak_gpu_mem_gb:.2f}GB"
            )

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--output", default="results/baseline_results.json")
    parser.add_argument("--seed", type=int, default=42,
                         help="Fixed seed for reproducible, matched-workload comparison "
                              "against the vLLM script (same default seed).")
    args = parser.parse_args()

    results = run_baseline_benchmark(args.model, args.seed)

    output = {
        "engine": "huggingface_naive",
        "model": args.model,
        "max_new_tokens": MAX_NEW_TOKENS,
        "runs": [asdict(r) for r in results],
    }

    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved {len(results)} runs to {args.output}")


if __name__ == "__main__":
    main()
