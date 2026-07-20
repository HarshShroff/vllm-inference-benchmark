"""
Compare vLLM vs. naive baseline results and generate plots + a markdown summary.

Usage:
    python analyze_results.py --vllm results/vllm_results.json --baseline results/baseline_results.json
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def load_results(path: str) -> pd.DataFrame:
    with open(path) as f:
        data = json.load(f)
    df = pd.DataFrame(data["runs"])
    df["engine"] = data["engine"]
    df["model"] = data["model"]
    return df


def plot_throughput_vs_batch(df: pd.DataFrame, out_path: str):
    fig, ax = plt.subplots(figsize=(8, 5))
    for engine in df["engine"].unique():
        subset = df[(df["engine"] == engine) & (df["length_bucket"] == "medium")]
        subset = subset.sort_values("batch_size")
        ax.plot(
            subset["batch_size"],
            subset["throughput_tokens_per_sec"],
            marker="o",
            label=engine,
        )
    ax.set_xlabel("Batch Size")
    ax.set_ylabel("Throughput (tokens/sec)")
    ax.set_title("Throughput vs. Batch Size (medium-length prompts)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")


def plot_memory_vs_batch(df: pd.DataFrame, out_path: str):
    fig, ax = plt.subplots(figsize=(8, 5))
    for engine in df["engine"].unique():
        subset = df[(df["engine"] == engine) & (df["length_bucket"] == "medium")]
        subset = subset.sort_values("batch_size")
        ax.plot(
            subset["batch_size"], subset["peak_gpu_mem_gb"], marker="o", label=engine
        )
    ax.set_xlabel("Batch Size")
    ax.set_ylabel("Peak GPU Memory (GB)")
    ax.set_title("Peak GPU Memory vs. Batch Size (medium-length prompts)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")


def generate_summary(vllm_df: pd.DataFrame, baseline_df: pd.DataFrame, out_path: str):
    lines = ["# vLLM vs. Naive Baseline: Results Summary\n"]

    for bucket in ["short", "medium", "long"]:
        lines.append(f"## {bucket.capitalize()} prompts\n")
        lines.append("| Batch | vLLM tok/s | Baseline tok/s | Speedup | vLLM peak mem (GB) | Baseline peak mem (GB) |")
        lines.append("|---|---|---|---|---|---|")
        for batch in sorted(vllm_df["batch_size"].unique()):
            v = vllm_df[(vllm_df["batch_size"] == batch) & (vllm_df["length_bucket"] == bucket)]
            b = baseline_df[(baseline_df["batch_size"] == batch) & (baseline_df["length_bucket"] == bucket)]
            if v.empty or b.empty:
                continue
            v_tps = v["throughput_tokens_per_sec"].iloc[0]
            b_tps = b["throughput_tokens_per_sec"].iloc[0]
            speedup = v_tps / b_tps if b_tps else float("nan")
            v_mem = v["peak_gpu_mem_gb"].iloc[0]
            b_mem = b["peak_gpu_mem_gb"].iloc[0]
            lines.append(
                f"| {batch} | {v_tps:.1f} | {b_tps:.1f} | {speedup:.2f}x | {v_mem:.2f} | {b_mem:.2f} |"
            )
        lines.append("")

    lines.append("## Notes\n")
    lines.append(
        "- Speedup = vLLM throughput / baseline throughput at matched batch size and prompt length.\n"
        "- Peak memory captured via `torch.cuda.max_memory_allocated()` reset per run.\n"
        "- Baseline TTFB is approximated (no true streaming); vLLM TTFB uses per-request "
        "  first-token timestamps where available.\n"
        "- See `plots/` for throughput and memory scaling charts.\n"
    )

    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Saved {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vllm", default="results/vllm_results.json")
    parser.add_argument("--baseline", default="results/baseline_results.json")
    parser.add_argument("--plots-dir", default="plots")
    parser.add_argument("--summary-out", default="results/SUMMARY.md")
    args = parser.parse_args()

    Path(args.plots_dir).mkdir(exist_ok=True)

    vllm_df = load_results(args.vllm)
    baseline_df = load_results(args.baseline)
    combined = pd.concat([vllm_df, baseline_df], ignore_index=True)

    plot_throughput_vs_batch(combined, f"{args.plots_dir}/throughput_vs_batch.png")
    plot_memory_vs_batch(combined, f"{args.plots_dir}/memory_vs_batch.png")
    generate_summary(vllm_df, baseline_df, args.summary_out)


if __name__ == "__main__":
    main()
