from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from experiments.model_assets import resolve_local_snapshot
from experiments.openworkload_models import ResolvedModel, resolve_model_entry, runtime_lut_is_valid

_HF_HUB_DIR = Path.home() / ".cache" / "huggingface" / "hub"
_SUPPORTED_DATASET_EXTRACTORS = {"ultrachat", "longbench"}
_FALLBACK_LORA_SUPPORTED_ARCHITECTURES = {
    "BaichuanForCausalLM",
    "BaiChuanForCausalLM",
    "DeciLMForCausalLM",
    "GemmaForCausalLM",
    "Gemma2ForCausalLM",
    "MistralForCausalLM",
    "MixtralForCausalLM",
    "PhiForCausalLM",
    "QWenLMHeadModel",
    "Qwen2ForCausalLM",
}


def _cached_repo_ids(prefix: str) -> set[str]:
    if not _HF_HUB_DIR.exists():
        return set()
    ids: set[str] = set()
    for path in _HF_HUB_DIR.glob(f"{prefix}--*"):
        if path.is_dir():
            ids.add(path.name[len(prefix) + 2 :].replace("--", "/"))
    return ids


def list_local_model_repo_ids() -> list[str]:
    return sorted(_cached_repo_ids("models"))


def list_local_dataset_repo_ids() -> list[str]:
    return sorted(_cached_repo_ids("datasets"))


def _read_local_model_architectures(model_id: str) -> list[str]:
    local_snapshot = resolve_local_snapshot(model_id)
    if not local_snapshot:
        return []
    config_path = Path(local_snapshot) / "config.json"
    if not config_path.exists():
        return []
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception:
        return []
    architectures = payload.get("architectures") or []
    return [str(item).strip() for item in architectures if str(item).strip()]


def _supports_lora_via_vllm(architectures: list[str]) -> bool | None:
    if not architectures:
        return None
    try:
        from vllm.model_executor.models import ModelRegistry, supports_lora
    except Exception:
        return None
    for arch in architectures:
        try:
            model_cls = ModelRegistry._try_load_model_cls(arch)
        except Exception:
            continue
        if model_cls is not None:
            return bool(supports_lora(model_cls))
    return None


def _supports_lora_architecture(architectures: list[str]) -> bool:
    detected = _supports_lora_via_vllm(architectures)
    if detected is not None:
        return detected
    return any(arch in _FALLBACK_LORA_SUPPORTED_ARCHITECTURES for arch in architectures)


def _matches_name_filters(model: ResolvedModel, patterns: list[str]) -> bool:
    if not patterns:
        return False
    haystacks = [model.key, model.model_id, model.lut_name, model.label]
    text = " ".join(part for part in haystacks if part).lower()
    return any(pattern.lower() in text for pattern in patterns if pattern)


def select_local_model_entries(
    entries: list[Any],
    *,
    require_runtime_sanity: bool = True,
    require_lora_support: bool = False,
    exclude_name_substrings: list[str] | None = None,
) -> tuple[list[ResolvedModel], list[dict[str, Any]]]:
    local_ids = set(list_local_model_repo_ids())
    selected: list[ResolvedModel] = []
    diagnostics: list[dict[str, Any]] = []
    seen_keys: set[str] = set()
    deny_patterns = [str(item).strip() for item in (exclude_name_substrings or []) if str(item).strip()]

    for entry in entries:
        try:
            model = resolve_model_entry(entry)
        except Exception as exc:
            diagnostics.append(
                {
                    "entry": entry,
                    "selected": False,
                    "reason": f"resolve_failed:{exc}",
                }
            )
            continue

        local_snapshot = resolve_local_snapshot(model.model_id)
        local_cached = bool(local_snapshot) or model.model_id in local_ids
        runtime_ok, runtime_reason = runtime_lut_is_valid(model.lut_name)
        architectures = _read_local_model_architectures(model.model_id)
        lora_ok = _supports_lora_architecture(architectures)
        excluded_by_name = _matches_name_filters(model, deny_patterns)
        selected_flag = (
            bool(local_cached)
            and (runtime_ok or not require_runtime_sanity)
            and (lora_ok or not require_lora_support)
            and not excluded_by_name
        )
        diagnostics.append(
            {
                "key": model.key,
                "model_id": model.model_id,
                "lut_name": model.lut_name,
                "label": model.label,
                "model_path_mode": model.model_path_mode,
                "local_cached": local_cached,
                "local_snapshot": local_snapshot,
                "architectures": architectures,
                "runtime_sanity_ok": runtime_ok,
                "runtime_sanity_reason": runtime_reason,
                "lora_supported": lora_ok,
                "excluded_by_name": excluded_by_name,
                "exclude_name_substrings": deny_patterns,
                "selected": selected_flag,
            }
        )
        if selected_flag and model.key not in seen_keys:
            selected.append(model)
            seen_keys.add(model.key)
    return selected, diagnostics


def select_local_dataset_entries(
    entries: list[dict[str, Any]],
    *,
    require_supported_extractors: bool = True,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    local_ids = set(list_local_dataset_repo_ids())
    selected: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    seen_keys: set[str] = set()

    for raw in entries:
        if not isinstance(raw, dict):
            diagnostics.append({"entry": raw, "selected": False, "reason": "invalid_dataset_entry"})
            continue
        key = str(raw.get("key") or raw.get("dataset_id") or "").strip()
        dataset_id = str(raw.get("dataset_id") or "").strip()
        extractor = str(raw.get("extractor") or "").strip().lower()
        local_cached = bool(dataset_id) and dataset_id in local_ids
        extractor_ok = (not require_supported_extractors) or extractor in _SUPPORTED_DATASET_EXTRACTORS
        selected_flag = bool(key and dataset_id and local_cached and extractor_ok)
        diagnostics.append(
            {
                "key": key,
                "dataset_id": dataset_id,
                "extractor": extractor,
                "local_cached": local_cached,
                "supported_extractor": extractor_ok,
                "selected": selected_flag,
            }
        )
        if selected_flag and key not in seen_keys:
            selected.append(dict(raw))
            seen_keys.add(key)
    return selected, diagnostics
