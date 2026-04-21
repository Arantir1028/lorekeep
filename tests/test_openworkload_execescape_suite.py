from __future__ import annotations

from experiments.run_openworkload_execescape_suite import (
    _load_config,
    _resolve_selected_densities,
    _resolve_selected_datasets,
    _resolve_selected_models,
)


def test_chapter5_lora8_config_selects_expected_models_and_datasets() -> None:
    config = _load_config("experiments/configs/openworkload_v1_local_realworld_lora8.json")

    models, model_diagnostics = _resolve_selected_models(config, "")
    datasets, dataset_diagnostics = _resolve_selected_datasets(config, "")

    assert [model.key for model in models] == [
        "mistral-7b-v0.1",
        "mistral-7b-instruct-v0.2",
        "zephyr-7b-beta",
        "openchat-3.5-0106",
        "gemma-7b-it",
        "baichuan2-7b-chat",
        "qwen2.5-7b-instruct",
        "gemma-2-9b-it",
    ]
    assert [dataset["key"] for dataset in datasets] == [
        "ultrachat200k",
        "longbench",
    ]

    selected_model_keys = {
        item.get("key") for item in model_diagnostics if item.get("selected")
    }
    selected_dataset_keys = {
        item.get("key") for item in dataset_diagnostics if item.get("selected")
    }
    assert selected_model_keys == {model.key for model in models}
    assert selected_dataset_keys == {dataset["key"] for dataset in datasets}

    by_key = {item.get("key"): item for item in model_diagnostics if item.get("key")}
    assert by_key["mistral-7b-v0.1"]["lora_supported"] is True
    assert by_key["qwen2.5-7b-instruct"]["runtime_sanity_ok"] is True
    assert by_key["gpt-j-6b"]["selected"] is False
    assert by_key["gpt-j-6b"]["lora_supported"] is False


def test_chapter5_density_filter_supports_subset_selection() -> None:
    config = _load_config("experiments/configs/openworkload_v1_local_realworld_lora8.json")
    densities = _resolve_selected_densities(config, "mid,peak")

    assert [item["name"] for item in densities] == ["mid", "peak"]
