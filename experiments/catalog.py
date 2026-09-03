"""Shared model and dataset catalog for the experiment drivers."""

from __future__ import annotations

import re
from dataclasses import dataclass

from waveslice.lut.config import lut_name_from_model_ref


@dataclass(frozen=True)
class ExperimentModelSpec:
    key: str
    model_id: str
    trust_remote_code: bool = False
    max_model_len_override: int | None = None

    @property
    def lut_name(self) -> str:
        return lut_name_from_model_ref(self.model_id)


@dataclass(frozen=True)
class SyntheticAdapterPreset:
    name: str
    rank: int
    alpha: int
    seed: int
    init_std: float


@dataclass(frozen=True)
class DatasetSourceSpec:
    key: str
    dataset_id: str
    split: str
    extractor: str
    streaming: bool = False


_MODEL_CATALOG: tuple[ExperimentModelSpec, ...] = (
    ExperimentModelSpec("mistral-7b-v0.1", "mistralai/Mistral-7B-v0.1"),
    ExperimentModelSpec("mistral-7b-instruct-v0.2", "mistralai/Mistral-7B-Instruct-v0.2"),
    ExperimentModelSpec("zephyr-7b-beta", "HuggingFaceH4/zephyr-7b-beta"),
    ExperimentModelSpec("openchat-3.5-0106", "openchat/openchat-3.5-0106"),
    ExperimentModelSpec("gemma-7b-it", "google/gemma-7b-it"),
    ExperimentModelSpec(
        "decilm-7b",
        "Deci/DeciLM-7B",
        trust_remote_code=True,
    ),
    ExperimentModelSpec(
        "phi-2",
        "microsoft/phi-2",
        max_model_len_override=2048,
    ),
    ExperimentModelSpec(
        "baichuan2-7b-chat",
        "baichuan-inc/Baichuan2-7B-Chat",
        trust_remote_code=True,
    ),
    ExperimentModelSpec(
        "gpt-j-6b",
        "EleutherAI/gpt-j-6b",
        max_model_len_override=2048,
    ),
    ExperimentModelSpec(
        "pythia-6.9b",
        "EleutherAI/pythia-6.9b",
        max_model_len_override=2048,
    ),
    ExperimentModelSpec("qwen2.5-7b-instruct", "Qwen/Qwen2.5-7B-Instruct"),
    ExperimentModelSpec("gemma-2-9b-it", "google/gemma-2-9b-it"),
)
_MODEL_BY_KEY = {model.key: model for model in _MODEL_CATALOG}
_DEFAULT_MODEL_KEYS = (
    "mistral-7b-v0.1",
    "mistral-7b-instruct-v0.2",
    "zephyr-7b-beta",
    "openchat-3.5-0106",
    "gemma-7b-it",
    "decilm-7b",
    "phi-2",
    "baichuan2-7b-chat",
)
DEFAULT_EXPERIMENT_MODELS: list[ExperimentModelSpec] = [
    _MODEL_BY_KEY[key] for key in _DEFAULT_MODEL_KEYS
]
DEFAULT_SYNTHETIC_ADAPTER_PRESETS: list[SyntheticAdapterPreset] = [
    SyntheticAdapterPreset(name="adapter_rank8_seed7", rank=8, alpha=16, seed=7, init_std=0.02),
    SyntheticAdapterPreset(name="adapter_rank16_seed11", rank=16, alpha=32, seed=11, init_std=0.04),
]
DEFAULT_DATASET_SOURCES: dict[str, DatasetSourceSpec] = {
    "ultrachat200k": DatasetSourceSpec(
        key="ultrachat200k",
        dataset_id="HuggingFaceH4/ultrachat_200k",
        split="train_sft",
        extractor="ultrachat",
        streaming=True,
    ),
    "longbench": DatasetSourceSpec(
        key="longbench",
        dataset_id="Xnhyacinth/LongBench",
        split="test",
        extractor="longbench",
        streaming=False,
    ),
}
DEFAULT_LONG_BENCH_CONFIGS = ["qmsum", "gov_report", "multifieldqa_en", "hotpotqa"]


def safe_key(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", value)


def find_model_spec(key: str) -> ExperimentModelSpec | None:
    return _MODEL_BY_KEY.get(key)


def get_model_spec(key: str) -> ExperimentModelSpec:
    spec = find_model_spec(key)
    if spec is None:
        raise ValueError(f"Unknown model key: {key}")
    return spec


def get_model_specs(keys: str | None = None) -> list[ExperimentModelSpec]:
    if not keys:
        return list(DEFAULT_EXPERIMENT_MODELS)
    key_set = {k.strip() for k in keys.split(",") if k.strip()}
    selected = [model for model in _MODEL_CATALOG if model.key in key_set]
    missing = key_set - {m.key for m in selected}
    if missing:
        raise ValueError(f"Unknown model keys: {sorted(missing)}")
    return selected
