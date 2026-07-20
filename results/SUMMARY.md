# vLLM vs. Naive Baseline: Results Summary

Seeded run (seed=42, both engines). See `RESULTS.md` for full analysis and caveats —
**MEDIUM/LONG numbers are the reportable headline; SHORT is a documented limitation,
not a clean comparison** (see Finding 2 in RESULTS.md).

## Short prompts — NOT a clean throughput comparison, see RESULTS.md Finding 2

| Batch | vLLM tok/s | Baseline tok/s | Speedup | vLLM peak mem (GB) | Baseline peak mem (GB) |
|---|---|---|---|---|---|
| 1 | 28.5 | 12.4 | 2.30x | 9.75 | 5.77 |
| 4 | 69.9 | 69.4 | 1.01x | 9.76 | 5.83 |
| 8 | 114.8 | 116.5 | 0.99x | 9.77 | 5.92 |
| 16 | 217.3 | 135.8 | 1.60x | 9.80 | 6.06 |
| 32 | 392.1 | 392.2 | 1.00x | 9.85 | 6.34 |

## Medium prompts — matched token counts, valid comparison

| Batch | vLLM tok/s | Baseline tok/s | Speedup | vLLM peak mem (GB) | Baseline peak mem (GB) |
|---|---|---|---|---|---|
| 1 | 24.1 | 17.9 | 1.35x | 9.75 | 5.78 |
| 4 | 125.5 | 72.0 | 1.74x | 9.76 | 5.84 |
| 8 | 229.2 | 142.6 | 1.61x | 9.78 | 5.92 |
| 16 | 390.4 | 177.2 | 2.20x | 9.81 | 6.08 |
| 32 | 624.1 | 372.8 | 1.67x | 9.86 | 6.39 |

## Long prompts — matched token counts, valid comparison

| Batch | vLLM tok/s | Baseline tok/s | Speedup | vLLM peak mem (GB) | Baseline peak mem (GB) |
|---|---|---|---|---|---|
| 1 | 20.4 | 17.8 | 1.15x | 9.76 | 5.78 |
| 4 | 122.9 | 70.4 | 1.75x | 9.79 | 5.86 |
| 8 | 227.6 | 133.1 | 1.71x | 9.84 | 5.95 |
| 16 | 395.2 | 228.5 | 1.73x | 9.94 | 6.14 |
| 32 | 575.1 | 300.4 | 1.91x | 10.13 | 6.52 |

## Notes

- Speedup = vLLM throughput / baseline throughput at matched batch size and prompt length.
- Sampling seed fixed at 42 for both engines (`SamplingParams(seed=...)` for vLLM,
  `torch.manual_seed()` + `torch.cuda.manual_seed_all()` reset before every batch for HF).
  This made MEDIUM/LONG token counts match exactly between engines, but SHORT prompts
  still diverge in total tokens generated even with a matched seed — see `RESULTS.md`.
- Peak memory captured via `torch.cuda.max_memory_allocated()`, reset per run. vLLM's
  peak reflects its upfront KV-cache pool (`gpu_memory_utilization=0.85`), not
  per-request incremental usage — see `RESULTS.md` Finding 3.
- Baseline TTFB is approximated (no true streaming); vLLM TTFB uses per-request
  first-token timestamps where available.
- See `plots/` for throughput and memory scaling charts.
