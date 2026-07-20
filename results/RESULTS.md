# Results Write-Up

## Setup

- **GPU**: Google Colab T4, 16GB VRAM
- **Model**: Qwen2.5-3B-Instruct, fp16
- **vLLM**: `gpu_memory_utilization=0.85`
- **Sampling**: temperature=0.7, max_new_tokens=256, seed=42 (fixed, both engines)

## Finding 1 — Headline: ~1.6–2.2x throughput improvement on matched workloads

This benchmark isolates one specific comparison: **vLLM's continuous-batching engine with
PagedAttention** vs. **naive HuggingFace `.generate()`** called in a single batched loop
with no serving-layer optimization. It does not vary GPU count, tensor parallelism, or
pipeline parallelism — both engines run on the same single T4 GPU. The throughput gain
below is attributable to vLLM's scheduling and memory-management design, not hardware.

For medium- and long-length prompts, every run (both engines, all batch sizes 1–32, with
a fixed seed) generated **exactly matching token counts** — verified directly against the
raw JSON (`total_tokens_generated` identical between engines at every batch size). These
two buckets are a genuinely fair, matched-workload comparison.

| Bucket | Batch=1 | Batch=4 | Batch=8 | Batch=16 | Batch=32 |
|---|---|---|---|---|---|
| Medium | 1.35x | 1.74x | 1.61x | 2.20x | 1.67x |
| Long   | 1.15x | 1.75x | 1.71x | 1.73x | 1.91x |

vLLM's continuous batching and PagedAttention deliver a consistent **~1.6–2.2x throughput
gain** over naive HF `.generate()` at batch sizes ≥4, once generation length is controlled
for. Batch=1 shows a smaller gain (1.15–1.35x) — expected, since vLLM's advantage comes
from request scheduling across concurrent sequences, which has nothing to exploit at
batch size 1.

## Finding 2 — Seeding fixed the confound for MEDIUM/LONG but exposed a deeper one in SHORT

The original (unseeded) run showed noisy, non-monotonic speedups on SHORT prompts because
both engines sampled independently at `temperature=0.7` and produced different amounts of
text on prompts that hit early EOS. **Fix applied**: `SamplingParams(seed=42)` for vLLM,
`torch.manual_seed(42)` + `torch.cuda.manual_seed_all(42)` reset before every batch for
the HF baseline.

The fix worked completely for MEDIUM/LONG — token counts now match exactly, every batch,
every size. It did **not** fully resolve SHORT:

| Bucket | Batch=1 | Batch=4 | Batch=8 | Batch=16 | Batch=32 |
|---|---|---|---|---|---|
| vLLM tokens | 32 | 336 | 913 | 1,808 | 3,744 |
| Baseline tokens | 50 | 1,021 | 2,042 | 4,083 | 8,170 |

Even with an identical seed, the HF baseline generates roughly **2–2.5x more tokens** than
vLLM on the same short prompts. This is not a seeding bug — a shared seed guarantees
*within-engine* reproducibility (rerun the same script twice, get identical output) but
does not guarantee *between-engine* agreement, because vLLM and HF `transformers` use
different sampling kernels, different numerical paths, and — most likely here — different
EOS/early-stopping logic in their respective scheduling loops. The two engines are making
genuinely different decisions about when to stop generating, independent of the RNG seed.

**Conclusion**: SHORT-bucket throughput numbers (2.30x, 1.01x, 0.99x, 1.60x, 1.00x) are
not reported as a headline result — they're measuring a mix of engine speed and each
engine's independent stopping behavior, which is itself an interesting but separate
finding from raw throughput. Reporting the MEDIUM/LONG numbers as the headline, and this
as a documented, understood limitation, is the honest framing.

## Finding 3 — Memory: vLLM trades a large fixed footprint for flat, predictable scaling

Read naively, the memory chart looks backwards: vLLM sits at 9.75–10.13GB peak
regardless of batch size, while naive HF grows from 5.77GB to 6.52GB with batch size —
naive HF appears to "use less memory." This is a measurement artifact, not a real
advantage for the baseline: `gpu_memory_utilization=0.85` tells vLLM to reserve a large
KV-cache memory pool **upfront**, at engine initialization, before any request is served.
`torch.cuda.reset_peak_memory_stats()` resets the *peak tracking counter* per run, not
the already-allocated pool, so what's measured per-batch is "pre-reserved pool + a small
increment" — not incremental memory consumed by that batch.

The honest, correctly-framed finding: **vLLM trades a large, fixed, upfront memory
reservation for flat and predictable scaling as load increases, while naive HF allocates
on-demand and scales roughly linearly with batch size.** In a production serving context
handling variable, unpredictable concurrent load, vLLM's flat footprint is the intended
tradeoff — you pay the memory cost once and get consistent headroom, rather than risking
OOM under unexpected load spikes the way an on-demand allocator would.

## Limitations

- Single-GPU (T4) only; no tensor parallelism or multi-GPU serving tested.
- Fixed prompt set (5 short / 3 medium / 2 long prompts, cycled) rather than a
  production traffic distribution.
- No quantization (AWQ/GPTQ) comparison — fp16 baseline only.
- SHORT-bucket throughput numbers reflect a mix of engine speed and diverging
  early-stopping behavior between engines, even with a matched seed (see Finding 2) —
  not usable as a clean throughput comparison without forcing `min_new_tokens=max_new_tokens`.
- Memory numbers reflect `gpu_memory_utilization=0.85`'s upfront pool, not per-request
  incremental usage; a lower utilization setting would shrink but not eliminate this effect.

## Next steps (if extending this project)

- Force `min_new_tokens=max_new_tokens` on SHORT bucket to disable early stopping
  entirely and get a true matched-workload comparison, isolating pure throughput from
  stopping-behavior differences.
- Add a lower `gpu_memory_utilization` (e.g. 0.3) run to show vLLM's KV-cache pool
  scaling more directly against baseline.
- Add a quantized (AWQ) vLLM run to measure the memory/throughput/quality tradeoff.
- Test with a realistic concurrent-request simulator instead of fixed batches.
