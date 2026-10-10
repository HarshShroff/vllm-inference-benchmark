"""
GPU-free checks for the SGLang baseline: results-JSON schema, parity of the
benchmark config with benchmark_vllm.py, and analyze_results.py with 2 or 3
engines. Run with:  python -m pytest tests/
"""

import ast
import json
import sys
from dataclasses import fields
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import analyze_results  # noqa: E402
import benchmark_sglang  # noqa: E402

REQUIRED_RUN_KEYS = {
    "batch_size", "length_bucket", "total_prompts", "wall_clock_sec",
    "total_tokens_generated", "throughput_tokens_per_sec", "ttfb_p50_ms",
    "peak_gpu_mem_gb",
}
REQUIRED_TOP_KEYS = {"engine", "model", "max_new_tokens", "runs"}


def _module_constants(path: Path) -> dict:
    """Top-level constant assignments, read via AST so torch/vllm need not be installed."""
    tree = ast.parse(path.read_text())
    out = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try:
                out[node.targets[0].id] = ast.literal_eval(node.value)
            except ValueError:
                pass
    return out


def _dataclass_fields(path: Path, name: str = "RunResult") -> list[str]:
    tree = ast.parse(path.read_text())
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == name:
            return [s.target.id for s in node.body if isinstance(s, ast.AnnAssign)]
    raise AssertionError(f"{name} not found in {path}")


def _vllm_sampling_kwargs() -> dict:
    tree = ast.parse((ROOT / "benchmark_vllm.py").read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "SamplingParams":
            return {k.arg: k.value for k in node.keywords}
    raise AssertionError("SamplingParams call not found in benchmark_vllm.py")


def validate_results(data: dict) -> None:
    assert REQUIRED_TOP_KEYS <= set(data), REQUIRED_TOP_KEYS - set(data)
    assert isinstance(data["runs"], list) and data["runs"]
    for run in data["runs"]:
        missing = REQUIRED_RUN_KEYS - set(run)
        assert not missing, missing
        assert run["length_bucket"] in {"short", "medium", "long"}
        assert run["total_tokens_generated"] >= 0
        if "tokens_per_request" in run:
            assert len(run["tokens_per_request"]) == run["total_prompts"]
            assert sum(run["tokens_per_request"]) == run["total_tokens_generated"]


def _fake_sglang_output() -> dict:
    """Synthetic data for exercising the code path only -- not benchmark numbers."""
    results = []
    for bucket in benchmark_sglang.LENGTH_BUCKETS:
        for bs in benchmark_sglang.BATCH_SIZES:
            tokens = [benchmark_sglang.MAX_NEW_TOKENS] * bs
            results.append(benchmark_sglang.build_run_result(
                bs, bucket, tokens, elapsed=1.0 + bs / 10, ttfbs_ms=[10.0] * bs, peak_mem_gb=9.9,
            ))
    return benchmark_sglang.build_output("Qwen/Qwen2.5-3B-Instruct", 42, results, {"sglang_version": "test"})


# --- config parity with benchmark_vllm.py ----------------------------------

def test_sweep_matches_vllm():
    v = _module_constants(ROOT / "benchmark_vllm.py")
    assert benchmark_sglang.BATCH_SIZES == v["BATCH_SIZES"]
    assert benchmark_sglang.LENGTH_BUCKETS == v["LENGTH_BUCKETS"]
    assert benchmark_sglang.MAX_NEW_TOKENS == v["MAX_NEW_TOKENS"]


def test_sampling_matches_vllm():
    kw = _vllm_sampling_kwargs()
    assert ast.literal_eval(kw["temperature"]) == benchmark_sglang.TEMPERATURE
    # vLLM leaves top_p at its default of 1.0; SGLang must match it.
    top_p = ast.literal_eval(kw["top_p"]) if "top_p" in kw else 1.0
    assert top_p == benchmark_sglang.TOP_P
    sp = benchmark_sglang.sampling_params()
    assert sp["temperature"] == benchmark_sglang.TEMPERATURE
    assert sp["top_p"] == benchmark_sglang.TOP_P
    assert sp["max_new_tokens"] == benchmark_sglang.MAX_NEW_TOKENS
    assert sp["ignore_eos"] is False


def test_runresult_schema_matches_vllm():
    vllm_fields = _dataclass_fields(ROOT / "benchmark_vllm.py")
    sglang_fields = [f.name for f in fields(benchmark_sglang.RunResult)]
    assert sglang_fields == vllm_fields


def test_same_prompts_module():
    from prompts import get_prompt_batch
    assert benchmark_sglang.get_prompt_batch is get_prompt_batch


# --- results JSON schema ---------------------------------------------------

@pytest.mark.parametrize("name", ["vllm_results.json", "baseline_results.json"])
def test_committed_results_schema(name):
    validate_results(json.loads((ROOT / "results" / name).read_text()))


def test_sglang_output_schema():
    data = _fake_sglang_output()
    validate_results(data)
    assert data["engine"] == "sglang"
    assert data["seed"] == 42
    assert data["max_new_tokens"] == benchmark_sglang.MAX_NEW_TOKENS
    assert len(data["runs"]) == len(benchmark_sglang.BATCH_SIZES) * len(benchmark_sglang.LENGTH_BUCKETS)
    json.dumps(data)  # serialisable


def test_build_run_result_math():
    r = benchmark_sglang.build_run_result(4, "medium", [256, 256, 100, 200], 2.0, [10, 20, 30, 40], 9.876)
    assert r.total_tokens_generated == 812
    assert r.throughput_tokens_per_sec == 406.0
    assert r.ttfb_p50_ms == 25.0
    assert r.peak_gpu_mem_gb == 9.88
    assert r.per_token_latency_p50_ms == round(2000 / 812 * 4, 3)


def test_build_run_result_without_ttfb():
    r = benchmark_sglang.build_run_result(1, "short", [5], 0.5, [], 1.0)
    assert (r.ttfb_p50_ms, r.ttfb_p90_ms, r.ttfb_p99_ms) == (0.0, 0.0, 0.0)


# --- run loop against stubbed sglang / pynvml (no GPU) ---------------------

class _FakeEngine:
    """Stands in for sglang.Engine: echoes a fixed token count per prompt."""

    last_kwargs = None

    def __init__(self, **kwargs):
        _FakeEngine.last_kwargs = kwargs
        self.calls = 0

    def generate(self, prompt, sampling_params):
        self.calls += 1
        n = min(17, sampling_params["max_new_tokens"])
        return [{"text": "x", "meta_info": {"completion_tokens": n}} for _ in prompt]

    async def async_generate(self, prompt, sampling_params, stream=False):
        async def gen():
            for _ in range(3):
                yield {"text": "x"}
        return gen()

    def shutdown(self):
        pass


def _install_stubs(monkeypatch, cc=(7, 5)):
    import types

    sgl = types.ModuleType("sglang")
    sgl.__version__ = "0.4.6.post5-stub"
    sgl.Engine = _FakeEngine
    nvml = types.ModuleType("pynvml")
    nvml.nvmlInit = lambda: None
    nvml.nvmlDeviceGetHandleByIndex = lambda i: object()
    nvml.nvmlDeviceGetCudaComputeCapability = lambda h: cc
    nvml.nvmlDeviceGetName = lambda h: b"Tesla T4"
    nvml.nvmlDeviceGetMemoryInfo = lambda h: types.SimpleNamespace(used=5 * 1024**3)
    monkeypatch.setitem(sys.modules, "sglang", sgl)
    monkeypatch.setitem(sys.modules, "pynvml", nvml)


def test_run_loop_with_stubs_t4(monkeypatch):
    import asyncio
    asyncio.set_event_loop(asyncio.new_event_loop())
    _install_stubs(monkeypatch, cc=(7, 5))
    results, meta = benchmark_sglang.run_sglang_benchmark("Qwen/Qwen2.5-3B-Instruct", 42)
    assert len(results) == 15
    assert _FakeEngine.last_kwargs["attention_backend"] == "triton"
    assert _FakeEngine.last_kwargs["sampling_backend"] == "pytorch"
    assert _FakeEngine.last_kwargs["random_seed"] == 42
    assert _FakeEngine.last_kwargs["mem_fraction_static"] == 0.85
    assert meta["compute_capability"] == "7.5" and meta["gpu"] == "Tesla T4"
    r = results[-1]
    assert r.tokens_per_request == [17] * 32 and r.total_tokens_generated == 17 * 32
    assert r.peak_gpu_mem_gb == 5.0
    assert r.ttfb_p50_ms >= 0
    validate_results(benchmark_sglang.build_output("m", 42, results, meta))


def test_run_loop_keeps_defaults_on_ampere(monkeypatch):
    import asyncio
    asyncio.set_event_loop(asyncio.new_event_loop())
    _install_stubs(monkeypatch, cc=(8, 9))
    _, meta = benchmark_sglang.run_sglang_benchmark("m", 42, ttfb_pass=False)
    assert "attention_backend" not in _FakeEngine.last_kwargs
    assert meta["ttfb_method"] == "not_measured"


# --- analyze_results.py with 2 and 3 engines -------------------------------

def _run_analyze(tmp_path, sglang_path):
    plots = tmp_path / "plots"
    summary = tmp_path / "SUMMARY.md"
    analyze_results.main([
        "--vllm", str(ROOT / "results" / "vllm_results.json"),
        "--baseline", str(ROOT / "results" / "baseline_results.json"),
        "--sglang", str(sglang_path),
        "--plots-dir", str(plots),
        "--summary-out", str(summary),
    ])
    assert (plots / "throughput_vs_batch.png").stat().st_size > 0
    assert (plots / "memory_vs_batch.png").stat().st_size > 0
    return summary.read_text()


def test_analyze_two_engines_when_sglang_missing(tmp_path):
    text = _run_analyze(tmp_path, tmp_path / "does_not_exist.json")
    assert "SGLang" not in text
    assert "| Batch | vLLM tok/s | Baseline tok/s | Speedup |" in text
    assert text.count("\n| 32 |") == 3


def test_analyze_three_engines(tmp_path):
    p = tmp_path / "sglang_results.json"
    p.write_text(json.dumps(_fake_sglang_output()))
    text = _run_analyze(tmp_path, p)
    assert "SGLang tok/s" in text and "SGLang/vLLM" in text and "Equal tokens" in text
    assert text.count("\n| 32 |") == 3


def test_three_engine_series_in_plot_data(tmp_path):
    p = tmp_path / "sglang_results.json"
    p.write_text(json.dumps(_fake_sglang_output()))
    combined = analyze_results.pd.concat([
        analyze_results.load_results(str(ROOT / "results" / "vllm_results.json")),
        analyze_results.load_results(str(ROOT / "results" / "baseline_results.json")),
        analyze_results.load_optional_results(str(p)),
    ])
    assert analyze_results._ordered_engines(combined) == ["vllm", "huggingface_naive", "sglang"]


def test_load_optional_missing_returns_none(tmp_path):
    assert analyze_results.load_optional_results(str(tmp_path / "nope.json")) is None
    assert analyze_results.load_optional_results(None) is None
