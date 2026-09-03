from __future__ import annotations

from typing import Any

from experiments.model_assets import _hf_hub_dir, resolve_local_snapshot
from experiments.openworkload_models import ResolvedModel, resolve_model_entry, runtime_lut_is_valid

_SUPPORTED_DATASET_EXTRACTORS = {"ultrachat", "longbench"}


def _cached_repo_ids(prefix: str) -> set[str]:
    _HF_HUB_DIR = _hf_hub_dir()
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
    auto_download: bool = False,
) -> tuple[list[ResolvedModel], list[dict[str, Any]]]:
    local_ids = set(list_local_model_repo_ids())
    selected: list[ResolvedModel] = []
    diagnostics: list[dict[str, Any]] = []
    seen_keys: set[str] = set()
    deny_patterns = [
        str(item).strip() for item in exclude_name_substrings or [] if str(item).strip()
    ]
    for entry in entries:
        try:
            model = resolve_model_entry(entry)
        except (TypeError, ValueError) as exc:
            diagnostics.append(
                {"entry": entry, "selected": False, "reason": f"resolve_failed:{exc}"}
            )
            continue
        local_snapshot = resolve_local_snapshot(model.model_id)
        local_cached = bool(local_snapshot) or model.model_id in local_ids
        (runtime_ok, runtime_reason) = runtime_lut_is_valid(model.lut_name)
        explicit_lora = entry.get("lora_supported") if isinstance(entry, dict) else None
        lora_ok = bool(explicit_lora)
        excluded_by_name = _matches_name_filters(model, deny_patterns)
        downloadable = bool(auto_download and model.model_id and (not excluded_by_name))
        selected_flag = (
            bool(local_cached or downloadable)
            and (runtime_ok or not require_runtime_sanity)
            and (lora_ok or not require_lora_support)
            and (not excluded_by_name)
        )
        diagnostics.append(
            {
                "key": model.key,
                "model_id": model.model_id,
                "lut_name": model.lut_name,
                "local_cached": local_cached,
                "local_snapshot": local_snapshot,
                "downloadable": downloadable,
                "runtime_sanity_ok": runtime_ok,
                "runtime_sanity_reason": runtime_reason,
                "lora_supported": lora_ok,
                "excluded_by_name": excluded_by_name,
                "selected": selected_flag,
            }
        )
        if selected_flag and model.key not in seen_keys:
            selected.append(model)
            seen_keys.add(model.key)
    return (selected, diagnostics)


def select_local_dataset_entries(
    entries: list[dict[str, Any]],
    *,
    require_supported_extractors: bool = True,
    auto_download: bool = False,
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
        extractor_ok = (
            not require_supported_extractors or extractor in _SUPPORTED_DATASET_EXTRACTORS
        )
        downloadable = bool(auto_download and dataset_id and extractor_ok)
        selected_flag = bool(key and dataset_id and (local_cached or downloadable) and extractor_ok)
        diagnostics.append(
            {
                "key": key,
                "dataset_id": dataset_id,
                "extractor": extractor,
                "local_cached": local_cached,
                "downloadable": downloadable,
                "supported_extractor": extractor_ok,
                "selected": selected_flag,
            }
        )
        if selected_flag and key not in seen_keys:
            selected.append(dict(raw))
            seen_keys.add(key)
    return (selected, diagnostics)
