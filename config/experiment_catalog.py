from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


def _checkpoint_lut_name(model_id: str) -> str:
    return model_id.replace("/", "--")


@dataclass(frozen=True)
class ExperimentModelSpec:
    key: str
    model_id: str
    lut_name: str
    family_hint: str
    raw_profile_name: Optional[str] = None
    trust_remote_code: bool = False
    max_model_len_override: Optional[int] = None


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


DEFAULT_EXPERIMENT_MODELS: list[ExperimentModelSpec] = [
    ExperimentModelSpec("mistral-7b-v0.1", "mistralai/Mistral-7B-v0.1", _checkpoint_lut_name("mistralai/Mistral-7B-v0.1"), "mistral", raw_profile_name=_checkpoint_lut_name("mistralai/Mistral-7B-v0.1")),
    ExperimentModelSpec("mistral-7b-instruct-v0.2", "mistralai/Mistral-7B-Instruct-v0.2", _checkpoint_lut_name("mistralai/Mistral-7B-Instruct-v0.2"), "mistral", raw_profile_name=_checkpoint_lut_name("mistralai/Mistral-7B-Instruct-v0.2")),
    ExperimentModelSpec("zephyr-7b-beta", "HuggingFaceH4/zephyr-7b-beta", _checkpoint_lut_name("HuggingFaceH4/zephyr-7b-beta"), "mistral", raw_profile_name=_checkpoint_lut_name("HuggingFaceH4/zephyr-7b-beta")),
    ExperimentModelSpec("openchat-3.5-0106", "openchat/openchat-3.5-0106", _checkpoint_lut_name("openchat/openchat-3.5-0106"), "mistral", raw_profile_name=_checkpoint_lut_name("openchat/openchat-3.5-0106")),
    ExperimentModelSpec("gemma-7b-it", "google/gemma-7b-it", _checkpoint_lut_name("google/gemma-7b-it"), "gemma", raw_profile_name=_checkpoint_lut_name("google/gemma-7b-it")),
    ExperimentModelSpec("decilm-7b", "Deci/DeciLM-7B", _checkpoint_lut_name("Deci/DeciLM-7B"), "deci", raw_profile_name=_checkpoint_lut_name("Deci/DeciLM-7B"), trust_remote_code=True),
    ExperimentModelSpec("phi-2", "microsoft/phi-2", _checkpoint_lut_name("microsoft/phi-2"), "phi", raw_profile_name=_checkpoint_lut_name("microsoft/phi-2"), max_model_len_override=2048),
    ExperimentModelSpec("baichuan2-7b-chat", "baichuan-inc/Baichuan2-7B-Chat", _checkpoint_lut_name("baichuan-inc/Baichuan2-7B-Chat"), "baichuan", raw_profile_name=_checkpoint_lut_name("baichuan-inc/Baichuan2-7B-Chat"), trust_remote_code=True),
]

DEFAULT_SYNTHETIC_SUITE_KEYS = [spec.key for spec in DEFAULT_EXPERIMENT_MODELS]
DEFAULT_DATASET_SUITE_KEYS = [
    "mistral-7b-v0.1",
    "mistral-7b-instruct-v0.2",
    "zephyr-7b-beta",
    "openchat-3.5-0106",
    "gemma-7b-it",
    "baichuan2-7b-chat",
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
    import re

    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", value)


def get_model_specs(keys: Optional[str] = None) -> list[ExperimentModelSpec]:
    if not keys:
        return list(DEFAULT_EXPERIMENT_MODELS)
    key_set = {k.strip() for k in keys.split(",") if k.strip()}
    selected = [m for m in DEFAULT_EXPERIMENT_MODELS if m.key in key_set]
    missing = key_set - {m.key for m in selected}
    if missing:
        raise ValueError(f"Unknown model keys: {sorted(missing)}")
    return selected
