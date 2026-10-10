"""
Compare vLLM vs. naive baseline (and SGLang, when present) results and
generate plots + a markdown summary.

Usage:
    python analyze_results.py --vllm results/vllm_results.json --baseline results/baseline_results.json
    # SGLang is picked up automatically from results/sglang_results.json if that
    # file exists; point elsewhere with --sglang, and it is skipped if missing.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

# Fixed plot order/labels so colours stay stable whether 2 or 3 engines load.
ENGINE_LABELS = {
    "vllm": "vLLM",
    "huggingface_naive": "Naive HF",
    "sglang": "SGLang",
}
ENGINE_ORDER = list(ENGINE_LABELS)


def load_results(path: str) -> pd.DataFrame:
    with open(path) as f:
        data = json.load(f)
    df = pd.DataFrame(data["runs"])
    df["engine"] = data["engine"]
    df["model"] = data["model"]
    return df


def load_optional_results(path: str | None) -> pd.DataFrame | None:
    """Load a results file if it exists, else None (lets SGLang be optional)."""
    if not path or not Path(path).is_file():
        if path:
            print(f"Skipping {path}: not found")
        return None
    return load_results(path)


def _ordered_engines(df: pd.DataFrame) -> list[str]:
    present = list(df["engine"].unique())
    return [e for e in ENGINE_ORDER if e in present] + [e for e in present if e not in ENGINE_ORDER]


def _plot_metric_vs_batch(df: pd.DataFrame, metric: str, ylabel: str, title: str, out_path: str):
    fig, ax = plt.subplots(figsize=(8, 5))
    for engine in _ordered_engines(df):
        subset = df[(df["engine"] == engine) & (df["length_bucket"] == "medium")]
        subset = subset.sort_values("batch_size")
        ax.plot(
            subset["batch_size"],
            subset[metric],
            marker="o",
            label=ENGINE_LABELS.get(engine, engine),
        )
    ax.set_xlabel("Batch Size")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_throughput_vs_batch(df: pd.DataFrame, out_path: str):
    _plot_metric_vs_batch(
        df,
        "throughput_tokens_per_sec",
        "Throughput (tokens/sec)",
        "Throughput vs. Batch Size (medium-length prompts)",
        out_path,
    )


def plot_memory_vs_batch(df: pd.DataFrame, out_path: str):
    _plot_metric_vs_batch(
        df,
        "peak_gpu_mem_gb",
        "Peak GPU Memory (GB)",
        "Peak GPU Memory vs. Batch Size (medium-length prompts)",
        out_path,
    )


def _row(df: pd.DataFrame, batch: int, bucket: str):
    r = df[(df["batch_size"] == batch) & (df["length_bucket"] == bucket)]
    return None if r.empty else r.iloc[0]


def _ratio(a: float, b: float) -> str:
    return f"{a / b:.2f}x" if b else "n/a"


def generate_summary(
    vllm_df: pd.DataFrame,
    baseline_df: pd.DataFrame,
    out_path: str,
    sglang_df: pd.DataFrame | None = None,
):
    has_sglang = sglang_df is not None and not sglang_df.empty
    title = "vLLM vs. SGLang vs. Naive Baseline" if has_sglang else "vLLM vs. Naive Baseline"
    lines = [f"# {title}: Results Summary\n"]

    if has_sglang:
        header = (
            "| Batch | vLLM tok/s | SGLang tok/s | Baseline tok/s | vLLM speedup | SGLang speedup "
            "| SGLang/vLLM | Equal tokens | vLLM peak mem (GB) | SGLang peak mem (GB) | Baseline peak mem (GB) |"
        )
        sep = "|---|---|---|---|---|---|---|---|---|---|---|"
    else:
        header = "| Batch | vLLM tok/s | Baseline tok/s | Speedup | vLLM peak mem (GB) | Baseline peak mem (GB) |"
        sep = "|---|---|---|---|---|---|"

    for bucket in ["short", "medium", "long"]:
        lines.append(f"## {bucket.capitalize()} prompts\n")
        lines.append(header)
        lines.append(sep)
        for batch in sorted(vllm_df["batch_size"].unique()):
            v = _row(vllm_df, batch, bucket)
            b = _row(baseline_df, batch, bucket)
            if v is None or b is None:
                continue
            v_tps = v["throughput_tokens_per_sec"]
            b_tps = b["throughput_tokens_per_sec"]
            if not has_sglang:
                lines.append(
                    f"| {batch} | {v_tps:.1f} | {b_tps:.1f} | {_ratio(v_tps, b_tps)} "
                    f"| {v['peak_gpu_mem_gb']:.2f} | {b['peak_gpu_mem_gb']:.2f} |"
                )
                continue
            s = _row(sglang_df, batch, bucket)
            if s is None:
                s_tps_str = s_speed = s_vs_v = s_mem = "n/a"
                equal = "n/a"
            else:
                s_tps = s["throughput_tokens_per_sec"]
                s_tps_str = f"{s_tps:.1f}"
                s_speed = _ratio(s_tps, b_tps)
                s_vs_v = _ratio(s_tps, v_tps)
                s_mem = f"{s['peak_gpu_mem_gb']:.2f}"
                counts = {
                    int(v["total_tokens_generated"]),
                    int(b["total_tokens_generated"]),
                    int(s["total_tokens_generated"]),
                }
                equal = "yes" if len(counts) == 1 else "no"
            lines.append(
                f"| {batch} | {v_tps:.1f} | {s_tps_str} | {b_tps:.1f} | {_ratio(v_tps, b_tps)} "
                f"| {s_speed} | {s_vs_v} | {equal} | {v['peak_gpu_mem_gb']:.2f} | {s_mem} "
                f"| {b['peak_gpu_mem_gb']:.2f} |"
            )
        lines.append("")

    lines.append("## Notes\n")
    speedup_def = "engine throughput" if has_sglang else "vLLM throughput"
    notes = (
        f"- Speedup = {speedup_def} / baseline throughput at matched batch size and prompt length.\n"
        "- Peak memory captured via `torch.cuda.max_memory_allocated()` reset per run.\n"
        "- Baseline TTFB is approximated (no true streaming); vLLM TTFB uses per-request "
        "  first-token timestamps where available.\n"
        "- See `plots/` for throughput and memory scaling charts.\n"
    )
    if has_sglang:
        notes += (
            "- SGLang peak memory is NVML device-used memory (its model runs in a separate "
            "  scheduler process), so it includes the CUDA context; compare shape, not decimals.\n"
            "- SGLang TTFB comes from a separate client-side streaming pass and does not affect "
            "  its throughput number.\n"
            "- `Equal tokens` = all three engines generated the same total tokens for that cell. "
            "  Only `yes` rows are an equal-work throughput comparison.\n"
        )
    lines.append(notes)

    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Saved {out_path}")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--vllm", default="results/vllm_results.json")
    parser.add_argument("--baseline", default="results/baseline_results.json")
    parser.add_argument("--sglang", default="results/sglang_results.json",
                        help="Optional; skipped gracefully if the file does not exist.")
    parser.add_argument("--plots-dir", default="plots")
    parser.add_argument("--summary-out", default="results/SUMMARY.md")
    args = parser.parse_args(argv)

    Path(args.plots_dir).mkdir(exist_ok=True)

    vllm_df = load_results(args.vllm)
    baseline_df = load_results(args.baseline)
    sglang_df = load_optional_results(args.sglang)
    frames = [vllm_df, baseline_df] + ([sglang_df] if sglang_df is not None else [])
    combined = pd.concat(frames, ignore_index=True)

    plot_throughput_vs_batch(combined, f"{args.plots_dir}/throughput_vs_batch.png")
    plot_memory_vs_batch(combined, f"{args.plots_dir}/memory_vs_batch.png")
    generate_summary(vllm_df, baseline_df, args.summary_out, sglang_df)


if __name__ == "__main__":
    main()
