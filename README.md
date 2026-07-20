# vLLM Inference Benchmark: Throughput & Memory vs. Naive HF Serving

A benchmark that answers a specific, practical question: how much do you actually gain by
serving an LLM with [vLLM](https://github.com/vllm-project/vllm)'s continuous batching and
PagedAttention, instead of the naive `model.generate()` loop most people reach for first?

The short answer: **a consistent 1.6-2.2x throughput improvement**, on prompts where both
engines do a genuinely equal amount of work. The longer answer — including a sampling bug
this project found in its own methodology, fixed, and then found a second, deeper issue
underneath it — is in [`results/RESULTS.md`](results/RESULTS.md).

## Why this exists

Most of my production LLM work (VITG, CARDS, Silicon Oracle) has run on managed inference —
AWS Bedrock, the Gemini API, local Ollama. Someone else's infrastructure team already made
the serving-engine decision. This project exists to close that gap: to actually run a
self-hosted, high-throughput serving engine, measure it properly, and understand *why*
PagedAttention and continuous batching outperform the naive approach, rather than taking
that claim on faith.

**Scope, stated precisely:** this compares vLLM's continuous-batching engine against naive,
unbatched HuggingFace `.generate()` calls, on a single GPU, with a single model instance, no
quantization. It says nothing about vLLM versus other serving engines (TGI, TensorRT-LLM,
SGLang), and nothing about multi-GPU or tensor/pipeline parallelism. The claim under test is
narrow and defensible: *vLLM's scheduling and memory management beat the naive unbatched
approach under these conditions* — not "vLLM is the fastest engine that exists."

## Three findings, not one

Most benchmark write-ups report a single number and move on. This one surfaced three things
worth knowing, and the second one is arguably more interesting than the headline:

**1. The throughput gain is real and consistent.** On prompts where both engines are
verified to generate the identical number of tokens, vLLM wins by 1.6-2.2x at batch sizes
4 and above. The gain shows up even at batch size 1, though more modestly (1.15-1.35x) —
expected, since vLLM's advantage comes from scheduling across concurrent requests, and
there's nothing to schedule with only one request in flight.

![Throughput vs. batch size](plots/throughput_vs_batch.png)

**2. Fixing a sampling bug revealed a second, more interesting one.** The first run left
both engines' random seeds unfixed. Since both sample at `temperature=0.7`, they generated
different amounts of text on prompts that stop early — which meant part of the "speedup"
being measured was actually just one engine writing more words than the other. Seeding both
engines identically fixed this completely for medium- and long-length prompts. It did *not*
fix it for short prompts: even with a matched seed, HF's baseline still generates roughly
2-2.5x more tokens than vLLM on the same short input. That's not a seeding bug — it's vLLM
and HF `transformers` reaching different EOS/early-stopping decisions independently of the
seed, using different sampling kernels. Rather than paper over that with a caveat, this
project reports it as a distinct, unresolved finding and excludes short-prompt numbers from
the headline claim.

**3. The memory chart looks backwards until you know why.** vLLM's peak GPU memory sits flat
around 9.75-10.13GB no matter the batch size, while the naive baseline grows from 5.77GB to
6.52GB as batch size increases. Read naively, that says "vLLM uses more memory" — which is
the wrong conclusion. vLLM reserves a large KV-cache pool *upfront*, at startup
(`gpu_memory_utilization=0.85`), so what's being measured is a large fixed reservation, not
per-request growth. The naive baseline allocates on demand, so its memory use tracks load
directly — which also means it has no headroom against a load spike. The honest framing:
vLLM trades a bigger reserved footprint for flat, predictable scaling.

![Peak GPU memory vs. batch size](plots/memory_vs_batch.png)

## What's measured

- **Throughput** — tokens/sec at batch sizes 1, 4, 8, 16, 32, across short/medium/long prompts
- **Latency** — time-to-first-byte and P50/P90/P99 per-token latency
- **Memory** — peak GPU memory, PagedAttention's upfront pool vs. naive on-demand allocation
- **A fixed sampling seed (42)** across both engines, so the comparison is reproducible and
  the confound in Finding 2 above doesn't recur silently

## Setup used for the reported results

| | |
|---|---|
| GPU | Google Colab T4 (16GB VRAM) |
| Model | `Qwen/Qwen2.5-3B-Instruct`, fp16 |
| Sampling | temperature=0.7, max_new_tokens=256, seed=42 (both engines) |

## Running it yourself

**Colab (recommended — no local GPU required).** Open
[`vLLM_vs_Naive_HF_Benchmark.ipynb`](vLLM_vs_Naive_HF_Benchmark.ipynb), select a T4-or-better
GPU runtime, and run all cells in order. It installs dependencies, writes out `config.py` /
`run_vllm.py` / `run_baseline.py`, runs both benchmarks, generates the comparison plots, and
zips `results/` + `plots/` for download.

**Local GPU.** The same benchmark logic as standalone scripts:

```bash
pip install -r requirements.txt

python benchmark_vllm.py --model Qwen/Qwen2.5-3B-Instruct --output results/vllm_results.json
python benchmark_baseline.py --model Qwen/Qwen2.5-3B-Instruct --output results/baseline_results.json
python analyze_results.py --vllm results/vllm_results.json --baseline results/baseline_results.json
```

## Project structure

```
vllm-inference-benchmark/
├── vLLM_vs_Naive_HF_Benchmark.ipynb   Colab notebook — the primary path for these results
├── benchmark_vllm.py                  standalone vLLM engine benchmark harness
├── benchmark_baseline.py              standalone naive HF transformers.generate() baseline
├── prompts.py                         fixed short/medium/long prompt set, for reproducibility
├── analyze_results.py                 comparison plots + markdown summary report
├── requirements.txt
├── results/
│   ├── vllm_results.json / baseline_results.json   raw benchmark output
│   ├── SUMMARY.md                     per-batch throughput and memory tables
│   └── RESULTS.md                     the full write-up: all three findings, in detail
└── plots/                             throughput_vs_batch.png, memory_vs_batch.png
```

## Status

Complete. Results generated on a Google Colab T4 running `Qwen/Qwen2.5-3B-Instruct`. Read
[`results/RESULTS.md`](results/RESULTS.md) for the full analysis, or
[`results/SUMMARY.md`](results/SUMMARY.md) for the raw per-batch numbers.
