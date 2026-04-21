from __future__ import annotations

from pathlib import Path

from experiments.local_resources import select_local_model_entries
from experiments.chapter2_prestudy import (
    _beneficiary_short_fraction,
    _collect_output_records,
    _collect_workload_records,
    _load_json,
    _make_lora_requests,
    _phase1_absolute_metrics,
    _resolve_model,
    _phase12_absolute_metrics,
)


def test_make_lora_requests_long_first_is_beneficiary_rich() -> None:
    reqs = _make_lora_requests(
        pattern="long_first",
        short_count=4,
        long_count=1,
        short_prompt_repeat=1,
        long_prompt_repeat=8,
        short_start_s=0.2,
        short_gap_s=0.05,
        long_gap_s=0.0,
        sequential_delay_s=6.0,
        mixed_adapters=True,
    )
    beneficiary_frac = _beneficiary_short_fraction(reqs)
    assert beneficiary_frac is not None
    assert beneficiary_frac > 0.9


def test_existing_v1_result_exposes_phase_metrics() -> None:
    result_path = Path(
        "results/openworkload_v1_local_realworld/"
        "20260410_v1_local_realworld_main_w1r2/raw/low/gemma-7b-it_dataset_eval.json"
    )
    data = _load_json(result_path)
    phase1 = _phase1_absolute_metrics(data)
    phase12 = _phase12_absolute_metrics(data)

    assert phase1["fixed_chunking"]["short_ttft_p99_ms"] is not None
    assert phase1["online_control"]["short_ttft_p99_ms"] is not None
    assert phase12["baseline"]["short_ttft_p99_ms"] is not None
    assert phase12["controlled"]["short_completion_p99_ms"] is not None
    assert len(phase12["rows"]) >= 1


def test_paper_e3_source_has_expected_ordering() -> None:
    result_path = Path(
        "results/openworkload_execescape_tradeoff/"
        "reuse_exec6_20260408_v2/raw/gemma-7b-it_dataset_eval.json"
    )
    data = _load_json(result_path)
    phase1 = _phase1_absolute_metrics(data)

    no_chunk = phase1["no_chunk"]["short_ttft_p99_ms"]
    chunked = phase1["fixed_chunking"]["short_ttft_p99_ms"]
    online = phase1["online_control"]["short_ttft_p99_ms"]

    assert no_chunk is not None
    assert chunked is not None
    assert online is not None
    assert no_chunk > chunked > online


def test_collectors_attach_model_keys_for_realworld_run() -> None:
    run_root = Path(
        "results/openworkload_v1_local_realworld/20260410_v1_local_realworld_main_w1r2"
    )
    workload_records = _collect_workload_records(run_root)
    output_records = _collect_output_records(run_root)

    assert workload_records
    assert output_records
    assert all(record["model_key"] for record in workload_records[:10])
    assert all(record["model_key"] for record in output_records[:10])



def test_resolve_model_accepts_explicit_entry() -> None:
    model = _resolve_model(
        {
            "model": {
                "key": "qwen2.5-7b-instruct",
                "model_id": "Qwen/Qwen2.5-7B-Instruct",
                "lut_name": "Qwen--Qwen2.5-7B-Instruct",
                "label": "Qwen2.5-7B-Instruct"
            },
            "eval": {"max_model_len": 3072},
        }
    )
    assert model["key"] == "qwen2.5-7b-instruct"
    assert model["model_name"] == "Qwen--Qwen2.5-7B-Instruct"
    assert model["model_id"] == "Qwen/Qwen2.5-7B-Instruct"


def test_local_model_selection_can_require_lora_and_exclude_llama_names() -> None:
    selected, diagnostics = select_local_model_entries(
        [
            "mistral-7b-v0.1",
            {
                "key": "qwen2.5-7b-instruct",
                "model_id": "Qwen/Qwen2.5-7B-Instruct",
                "lut_name": "Qwen--Qwen2.5-7B-Instruct",
                "label": "Qwen2.5-7B-Instruct",
            },
            {
                "key": "gpt-j-6b",
                "model_id": "EleutherAI/gpt-j-6b",
                "lut_name": "EleutherAI--gpt-j-6b",
                "label": "GPT-J-6B",
            },
            {
                "key": "sparse-llama",
                "model_id": "SparseLLM/ReluLLaMA-7B",
                "lut_name": "SparseLLM--ReluLLaMA-7B",
                "label": "SparseLLaMA-7B",
            },
        ],
        require_runtime_sanity=False,
        require_lora_support=True,
        exclude_name_substrings=["llama"],
    )
    selected_keys = {model.key for model in selected}
    assert "mistral-7b-v0.1" in selected_keys
    assert "qwen2.5-7b-instruct" in selected_keys
    assert "gpt-j-6b" not in selected_keys
    assert "sparse-llama" not in selected_keys
    by_key = {item.get("key"): item for item in diagnostics if item.get("key")}
    assert by_key["gpt-j-6b"]["lora_supported"] is False
    assert by_key["sparse-llama"]["excluded_by_name"] is True
