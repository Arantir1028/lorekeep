"""Resource selection and per-density configuration for open-workload runs."""

from __future__ import annotations

import sys
from collections.abc import Callable
from copy import deepcopy
from typing import Any

from experiments.local_resources import select_local_dataset_entries, select_local_model_entries
from experiments.openworkload_models import ResolvedModel, resolve_model_entry
from experiments.openworkload_support import load_config, project_path


def _load_resource_catalog(config: dict[str, Any]) -> dict[str, Any]:
    value = str(config.get("resource_catalog_config") or "").strip()
    if not value:
        return {}
    path = project_path(value)
    if not path.exists():
        raise FileNotFoundError(f"resource catalog config not found: {path}")
    return load_config(str(path))


def _candidate_entries(config: dict[str, Any], kind: str) -> list[Any]:
    return list(config.get(kind) or _load_resource_catalog(config).get(kind) or [])


def _requested(
    entries: list[Any], override: str, key_fn: Callable[[Any], str], kind: str
) -> list[Any]:
    keys = {item.strip() for item in override.split(",") if item.strip()}
    if not keys:
        return entries
    selected = [entry for entry in entries if key_fn(entry) in keys]
    missing = sorted(keys - {key_fn(entry) for entry in selected})
    if missing:
        raise ValueError(f"unknown requested {kind} keys: {missing}")
    return selected


def _resolve_selected_models(
    config: dict[str, Any], model_keys_override: str
) -> tuple[list[ResolvedModel], list[dict[str, Any]]]:
    selection = dict(config.get("resource_selection") or {})
    entries = _requested(
        _candidate_entries(config, "models"),
        model_keys_override,
        lambda entry: resolve_model_entry(entry).key,
        "model",
    )
    mode = (
        str(
            selection.get("model_mode")
            or ("local_all_runnable" if config.get("resource_catalog_config") else "configured")
        )
        .strip()
        .lower()
    )
    if mode == "local_all_runnable":
        return select_local_model_entries(
            entries,
            require_runtime_sanity=bool(selection.get("require_runtime_sanity", True)),
            require_lora_support=bool(selection.get("require_lora_support", False)),
            exclude_name_substrings=list(selection.get("exclude_name_substrings") or []),
            auto_download=bool(
                selection.get(
                    "auto_download", (config.get("resources") or {}).get("auto_download", True)
                )
            ),
        )
    if mode != "configured":
        raise ValueError(f"unknown model selection mode: {mode}")
    models = [resolve_model_entry(entry) for entry in entries]
    return models, [
        {
            "key": model.key,
            "model_id": model.model_id,
            "lut_name": model.lut_name,
            "label": model.label,
            "selected": True,
            "selection_mode": "configured",
        }
        for model in models
    ]


def _resolve_selected_datasets(
    config: dict[str, Any], dataset_keys_override: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    selection = dict(config.get("resource_selection") or {})
    entries = [entry for entry in _candidate_entries(config, "datasets") if isinstance(entry, dict)]
    entries = _requested(
        entries, dataset_keys_override, lambda entry: str(entry.get("key") or "").strip(), "dataset"
    )
    mode = (
        str(
            selection.get("dataset_mode")
            or (
                "local_supported_from_catalog"
                if config.get("resource_catalog_config")
                else "configured"
            )
        )
        .strip()
        .lower()
    )
    if mode == "local_supported_from_catalog":
        return select_local_dataset_entries(
            entries,
            require_supported_extractors=bool(selection.get("require_supported_extractors", True)),
            auto_download=bool(
                selection.get(
                    "auto_download", (config.get("resources") or {}).get("auto_download", True)
                )
            ),
        )
    if mode != "configured":
        raise ValueError(f"unknown dataset selection mode: {mode}")
    return entries, [
        {
            "key": str(entry.get("key") or ""),
            "dataset_id": str(entry.get("dataset_id") or ""),
            "extractor": str(entry.get("extractor") or ""),
            "selected": True,
            "selection_mode": "configured",
        }
        for entry in entries
    ]


def _resolve_selected_densities(
    config: dict[str, Any], densities_override: str
) -> list[dict[str, Any]]:
    entries = [
        dict(item)
        for item in (config.get("workload") or {}).get("densities") or []
        if isinstance(item, dict)
    ]
    return _requested(
        entries, densities_override, lambda entry: str(entry.get("name") or "").strip(), "density"
    )


def _density_arrival_pressure_score(
    density: dict[str, Any], reference_densities: list[dict[str, Any]]
) -> tuple[float, dict[str, float]]:
    def rate(item: dict[str, Any]) -> float:
        return float(item.get("phase2_arrival_rate", item.get("phase1_arrival_rate")) or 0)

    rates = [value for item in reference_densities if (value := rate(item)) > 0]
    current = max(0.0, rate(density))
    low, high = (min(rates), max(rates)) if rates else (0.0, 0.0)
    score = 0.5 if not rates or high <= low else max(0.0, min(1.0, (current - low) / (high - low)))
    return score, {"current_rate": current, "min_rate": low, "max_rate": high}


def _adapt_runtime_policy(
    adapted: dict[str, Any],
    policy: dict[str, Any],
    density: dict[str, Any],
    score: float,
    pressure: dict[str, float],
) -> dict[str, Any]:
    phase1, phase2 = dict(adapted.get("phase1") or {}), dict(adapted.get("phase2") or {})
    phase1["runtime_adaptive_enabled"] = True
    phase1_fields = (
        "runtime_aggressive_long_fraction",
        "runtime_conservative_long_fraction",
        "runtime_aggressive_ingress_target_chunk",
        "runtime_conservative_ingress_target_chunk",
        "runtime_queue_high_watermark",
        "runtime_waiting_short_high_watermark",
        "runtime_wait_us_high_watermark",
        "runtime_long_high_watermark",
        "runtime_urgency_discount",
        "runtime_ema_alpha",
    )
    phase1.update({key: policy[key] for key in phase1_fields})
    phase2["runtime_adaptive_enabled"] = True
    phase2_fields = (
        "runtime_low_pressure_min_hetero_ratio",
        "runtime_high_pressure_min_hetero_ratio",
        "runtime_low_pressure_min_pressure_ratio",
        "runtime_high_pressure_min_pressure_ratio",
        "runtime_low_pressure_min_long_prefill",
        "runtime_high_pressure_min_long_prefill",
    )
    phase2.update({key: policy[key] for key in phase2_fields})
    adapted.update(phase1=phase1, phase2=phase2)
    adapted["_adaptive_density_runtime"] = {
        "enabled": True,
        "density": str(density.get("name") or ""),
        "scope": "runtime_queue_pressure",
        "pressure_source": "scheduler_queue_waiting_short_wait_us_long_remaining_cap_hit_ema",
        "pressure_formula": "per-schedule weighted wall pressure with short-urgency discount",
        "workload_pressure_score": score,
        **{f"pressure_{key}": value for key, value in pressure.items()},
        "runtime_queue_pressure": True,
        "phase1_runtime_aggressive_long_fraction": phase1.get("runtime_aggressive_long_fraction"),
        "phase1_runtime_conservative_long_fraction": phase1.get(
            "runtime_conservative_long_fraction"
        ),
        "phase1_runtime_aggressive_ingress_target_chunk": phase1.get(
            "runtime_aggressive_ingress_target_chunk"
        ),
        "phase1_runtime_conservative_ingress_target_chunk": phase1.get(
            "runtime_conservative_ingress_target_chunk"
        ),
        "phase2_runtime_adaptive_enabled": phase2.get("runtime_adaptive_enabled"),
    }
    return adapted


def _adapt_config_for_density(
    config: dict[str, Any], density: dict[str, Any], reference_densities: list[dict[str, Any]]
) -> dict[str, Any]:
    policy = dict(config.get("adaptive_density_policy") or {})
    if not policy.get("enabled"):
        return config
    adapted = deepcopy(config)
    score, pressure = _density_arrival_pressure_score(density, reference_densities)
    return _adapt_runtime_policy(adapted, policy, density, score, pressure)


def _case_eval_config(
    *,
    model: ResolvedModel,
    model_path: str,
    req_json: str,
    lora_req_json: str,
    adapter_a: str,
    adapter_b: str,
    config: dict[str, Any],
    eval_cfg: dict[str, Any],
) -> dict[str, Any]:
    defaults = {
        "python_bin": sys.executable,
        "warmup_iters": 2,
        "repeats": 3,
        "timeout_sec": 240,
        "max_new_tokens": 64,
        "max_model_len": 3072,
        "max_num_batched_tokens": 1536,
        "gpu_memory_utilization": 0.60,
        "queue_reorder_mode": "sjf",
        "queue_reorder_aging_quantum_us": 20000,
    }
    casts = {
        "python_bin": str,
        "warmup_iters": int,
        "repeats": int,
        "timeout_sec": int,
        "max_new_tokens": int,
        "max_model_len": int,
        "max_num_batched_tokens": int,
        "gpu_memory_utilization": float,
        "queue_reorder_mode": str,
        "queue_reorder_aging_quantum_us": int,
    }
    runtime = {
        key: casts[key](eval_cfg.get(key, value) or value) for key, value in defaults.items()
    }
    runtime.update(
        trust_remote_code=bool(model.trust_remote_code),
        max_model_len=int(model.max_model_len_override or eval_cfg.get("max_model_len", 3072)),
    )
    return {
        "evaluator": "tests/evaluate_waveslice_claims.py",
        "model": {"name": model.lut_name, "path": model_path},
        "workload": {"requests_json": req_json, "lora_requests_json": lora_req_json},
        "adapters": {"adapter_a": adapter_a, "adapter_b": adapter_b},
        "runtime": runtime,
        "phase1": dict(config.get("phase1") or {}),
        "phase12_soft_gate": dict(config.get("phase12_soft_gate") or {}),
        "phase2": dict(config.get("phase2") or {}),
    }
