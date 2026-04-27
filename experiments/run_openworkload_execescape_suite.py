# Run the full configurable supplement suite:
#   python3 experiments/run_openworkload_execescape_suite.py \
#     --config experiments/configs/openworkload_execescape_default.json
#
# Rebuild metadata and English figures from an existing CSV/JSON run without rerunning GPUs:
#   python3 experiments/run_openworkload_execescape_suite.py \
#     --config experiments/configs/openworkload_execescape_default.json \
#     --reuse-csv <existing-main-suite>/metadata/suite_results.csv \
#     --reuse-results-dir <existing-main-suite>
#
# Output layout:
#   <out_root>/<timestamp>/
#     metadata/   # resolved config, rationale, CSV/JSON summaries, copied raw result JSONs
#     figures/    # English PNG charts
#     raw/        # per-model/per-density evaluation JSONs (when available)
#     workloads/  # generated request JSONs (when this script runs workloads itself)

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

os.environ.setdefault("VLLM_NO_USAGE_STATS", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig_wave_slice")

from config.experiment_catalog import safe_key
from experiments.local_resources import select_local_dataset_entries, select_local_model_entries
from experiments.model_assets import ensure_adapters as _ensure_adapters
from experiments.model_assets import resolve_local_snapshot as _resolve_local_snapshot
from experiments.openworkload_models import (
    ResolvedModel,
    resolve_model_entry as _resolve_model_entry,
    runtime_lut_is_valid as _runtime_lut_is_valid,
)
from experiments.openworkload_results import (
    aggregate_rows as _aggregate_rows,
    copy_existing_artifacts as _copy_existing_artifacts,
    enrich_rows_with_config as _enrich_rows_with_config,
    plot_density_summary as _plot_density_summary,
    plot_metric_by_model as _plot_metric_by_model,
    plot_tradeoff_scatter as _plot_tradeoff_scatter,
    rows_from_existing_csv as _rows_from_existing_csv,
    write_rationale_markdown as _write_rationale_markdown,
    write_result_summary_markdown as _write_result_summary_markdown,
)
from experiments.run_frozen_eval_config import build_eval_invocation as _build_eval_invocation
from experiments.openworkload_support import (
    build_dataset_source_payload as _build_dataset_source_payload,
    clear_gpu_lock as _clear_gpu_lock,
    completed_case_keys as _completed_case_keys,
    ensure_dir as _ensure_dir,
    extract_summary_from_result_json as _extract_summary_from_result_json,
    float_or_none as _float_or_none,
    kill_process_group as _kill_process_group,
    load_config as _load_config,
    load_existing_rows as _load_existing_rows,
    maybe_rel_to as _maybe_rel_to,
    pid_is_alive as _pid_is_alive,
    purge_experiment_processes as _purge_experiment_processes,
    resolve_existing_meta_json as _resolve_existing_meta_json,
    resolve_existing_result_json as _resolve_existing_result_json,
    wait_for_clean_gpu as _wait_for_clean_gpu,
    write_csv as _write_csv,
    write_json as _write_json,
    workload_meta_matches_model as _workload_meta_matches_model,
)


_GPU_LOCK_PATH = Path(os.environ.get("WAVESLICE_GPU_LOCK_PATH", "/tmp/waveslice_gpu_experiment.lock"))
_EXPERIMENT_PROC_PATTERNS = (
    "run_openworkload_execescape_suite.py",
    "evaluate_waveslice_claims.py",
    "VLLM::EngineCore",
)


def _load_resource_catalog(config: dict[str, Any]) -> dict[str, Any]:
    catalog_path = str(config.get("resource_catalog_config") or "").strip()
    if not catalog_path:
        return {}
    path = Path(catalog_path)
    if not path.exists():
        raise FileNotFoundError(f"resource catalog config not found: {path}")
    return _load_config(str(path))


def _candidate_model_entries(config: dict[str, Any]) -> list[Any]:
    explicit = list(config.get("models") or [])
    if explicit:
        return explicit
    catalog = _load_resource_catalog(config)
    from_catalog = list(catalog.get("models") or [])
    if from_catalog:
        return from_catalog
    return []


def _candidate_dataset_entries(config: dict[str, Any]) -> list[dict[str, Any]]:
    explicit = list(config.get("datasets") or [])
    if explicit:
        return [dict(item) for item in explicit if isinstance(item, dict)]
    catalog = _load_resource_catalog(config)
    return [dict(item) for item in (catalog.get("datasets") or []) if isinstance(item, dict)]


def _resolve_selected_models(
    config: dict[str, Any],
    model_keys_override: str,
) -> tuple[list[ResolvedModel], list[dict[str, Any]]]:
    selection_cfg = dict(config.get("resource_selection") or {})
    requested_keys = {item.strip() for item in model_keys_override.split(",") if item.strip()}
    candidate_entries = _candidate_model_entries(config)
    if requested_keys:
        filtered: list[Any] = []
        matched: set[str] = set()
        for entry in candidate_entries:
            resolved = _resolve_model_entry(entry)
            if resolved.key in requested_keys:
                filtered.append(entry)
                matched.add(resolved.key)
        missing = sorted(requested_keys - matched)
        if missing:
            raise ValueError(f"unknown requested model keys: {missing}")
        candidate_entries = filtered

    mode = str(
        selection_cfg.get("model_mode")
        or ("local_all_runnable" if config.get("resource_catalog_config") else "configured")
    ).strip().lower()
    if mode == "configured":
        models = [_resolve_model_entry(entry) for entry in candidate_entries]
        diagnostics = [
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
        return models, diagnostics
    if mode == "local_all_runnable":
        return select_local_model_entries(
            candidate_entries,
            require_runtime_sanity=bool(selection_cfg.get("require_runtime_sanity", True)),
            require_lora_support=bool(selection_cfg.get("require_lora_support", False)),
            exclude_name_substrings=list(selection_cfg.get("exclude_name_substrings") or []),
        )
    raise ValueError(f"unknown model selection mode: {mode}")


def _resolve_selected_datasets(
    config: dict[str, Any],
    dataset_keys_override: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    selection_cfg = dict(config.get("resource_selection") or {})
    requested_keys = {item.strip() for item in dataset_keys_override.split(",") if item.strip()}
    candidate_entries = _candidate_dataset_entries(config)
    if requested_keys:
        filtered: list[dict[str, Any]] = []
        matched: set[str] = set()
        for entry in candidate_entries:
            key = str(entry.get("key") or "").strip()
            if key in requested_keys:
                filtered.append(entry)
                matched.add(key)
        missing = sorted(requested_keys - matched)
        if missing:
            raise ValueError(f"unknown requested dataset keys: {missing}")
        candidate_entries = filtered

    mode = str(
        selection_cfg.get("dataset_mode")
        or ("local_supported_from_catalog" if config.get("resource_catalog_config") else "configured")
    ).strip().lower()
    if mode == "configured":
        diagnostics = [
            {
                "key": str(entry.get("key") or ""),
                "dataset_id": str(entry.get("dataset_id") or ""),
                "extractor": str(entry.get("extractor") or ""),
                "selected": True,
                "selection_mode": "configured",
            }
            for entry in candidate_entries
        ]
        return candidate_entries, diagnostics
    if mode == "local_supported_from_catalog":
        return select_local_dataset_entries(
            candidate_entries,
            require_supported_extractors=bool(selection_cfg.get("require_supported_extractors", True)),
        )
    raise ValueError(f"unknown dataset selection mode: {mode}")


def _resolve_selected_densities(
    config: dict[str, Any],
    densities_override: str,
) -> list[dict[str, Any]]:
    densities = [dict(item) for item in list(config.get("workload", {}).get("densities") or []) if isinstance(item, dict)]
    requested = {item.strip() for item in densities_override.split(",") if item.strip()}
    if not requested:
        return densities

    selected: list[dict[str, Any]] = []
    matched: set[str] = set()
    for entry in densities:
        name = str(entry.get("name") or "").strip()
        if name in requested:
            selected.append(entry)
            matched.add(name)
    missing = sorted(requested - matched)
    if missing:
        raise ValueError(f"unknown requested density names: {missing}")
    return selected


def _density_arrival_pressure_score(
    density: dict[str, Any],
    reference_densities: list[dict[str, Any]],
) -> tuple[float, dict[str, float]]:
    def _rate(item: dict[str, Any]) -> float:
        phase2 = item.get("phase2_arrival_rate")
        phase1 = item.get("phase1_arrival_rate")
        try:
            return float(phase2 if phase2 is not None else phase1)
        except Exception:
            return 0.0

    rates = [_rate(item) for item in reference_densities if isinstance(item, dict)]
    rates = [rate for rate in rates if rate > 0.0]
    current = max(0.0, _rate(density))
    meta = {
        "current_rate": current,
        "min_rate": min(rates) if rates else 0.0,
        "max_rate": max(rates) if rates else 0.0,
    }
    if not rates:
        return 0.5, meta
    lo = min(rates)
    hi = max(rates)
    if hi <= lo:
        return 0.5, meta
    return max(0.0, min(1.0, (current - lo) / (hi - lo))), meta


def _lerp(low: float, high: float, pressure_score: float) -> float:
    score = max(0.0, min(1.0, float(pressure_score)))
    return float(low) + (float(high) - float(low)) * score


def _adapt_config_for_density(
    config: dict[str, Any],
    density: dict[str, Any],
    reference_densities: list[dict[str, Any]],
) -> dict[str, Any]:
    adaptive_cfg = dict(config.get("adaptive_density_policy") or {})
    if not bool(adaptive_cfg.get("enabled", False)):
        return config

    adapted = deepcopy(config)
    pressure_score, pressure_meta = _density_arrival_pressure_score(density, reference_densities)
    phase1_cfg = dict(adapted.get("phase1") or {})
    phase2_cfg = dict(adapted.get("phase2") or {})
    runtime_queue_pressure = bool(adaptive_cfg.get("runtime_queue_pressure", False)) or str(
        adaptive_cfg.get("scope") or ""
    ).strip().lower() in {"runtime_queue_pressure", "runtime_pressure"}

    if runtime_queue_pressure:
        phase1_cfg["runtime_adaptive_enabled"] = True
        phase1_cfg["runtime_aggressive_long_fraction"] = float(
            adaptive_cfg.get(
                "runtime_aggressive_long_fraction",
                adaptive_cfg.get("high_pressure_target_long_fraction", phase1_cfg.get("target_long_fraction", 0.33)),
            )
        )
        phase1_cfg["runtime_conservative_long_fraction"] = float(
            adaptive_cfg.get(
                "runtime_conservative_long_fraction",
                adaptive_cfg.get("low_pressure_target_long_fraction", phase1_cfg.get("target_long_fraction", 0.50)),
            )
        )
        phase1_cfg["runtime_aggressive_ingress_target_chunk"] = int(
            adaptive_cfg.get(
                "runtime_aggressive_ingress_target_chunk",
                adaptive_cfg.get("high_pressure_ingress_target_chunk", phase1_cfg.get("ingress_target_chunk", 768)),
            )
        )
        phase1_cfg["runtime_conservative_ingress_target_chunk"] = int(
            adaptive_cfg.get(
                "runtime_conservative_ingress_target_chunk",
                adaptive_cfg.get("low_pressure_ingress_target_chunk", phase1_cfg.get("ingress_target_chunk", 1536)),
            )
        )
        for cfg_key, default in {
            "runtime_queue_high_watermark": 8,
            "runtime_waiting_short_high_watermark": 4,
            "runtime_wait_us_high_watermark": 1_000_000.0,
            "runtime_long_high_watermark": 3072,
            "runtime_urgency_discount": 0.55,
            "runtime_ema_alpha": 0.35,
        }.items():
            if cfg_key in adaptive_cfg:
                phase1_cfg[cfg_key] = adaptive_cfg[cfg_key]
            elif cfg_key not in phase1_cfg:
                phase1_cfg[cfg_key] = default

        phase2_cfg["runtime_adaptive_enabled"] = True
        phase2_cfg["runtime_low_pressure_min_hetero_ratio"] = float(
            adaptive_cfg.get("runtime_low_pressure_phase2_min_hetero_ratio", adaptive_cfg.get("low_pressure_phase2_min_hetero_ratio", phase2_cfg.get("min_hetero_ratio", 6.0)))
        )
        phase2_cfg["runtime_high_pressure_min_hetero_ratio"] = float(
            adaptive_cfg.get("runtime_high_pressure_phase2_min_hetero_ratio", adaptive_cfg.get("high_pressure_phase2_min_hetero_ratio", phase2_cfg.get("min_hetero_ratio", 4.0)))
        )
        phase2_cfg["runtime_low_pressure_min_pressure_ratio"] = float(
            adaptive_cfg.get("runtime_low_pressure_phase2_min_pressure_ratio", adaptive_cfg.get("low_pressure_phase2_min_pressure_ratio", phase2_cfg.get("min_pressure_ratio", 6.0)))
        )
        phase2_cfg["runtime_high_pressure_min_pressure_ratio"] = float(
            adaptive_cfg.get("runtime_high_pressure_phase2_min_pressure_ratio", adaptive_cfg.get("high_pressure_phase2_min_pressure_ratio", phase2_cfg.get("min_pressure_ratio", 4.0)))
        )
        phase2_cfg["runtime_low_pressure_min_long_prefill"] = int(
            adaptive_cfg.get("runtime_low_pressure_phase2_min_long_prefill", adaptive_cfg.get("low_pressure_phase2_min_long_prefill", phase2_cfg.get("min_long_prefill", 1024)))
        )
        phase2_cfg["runtime_high_pressure_min_long_prefill"] = int(
            adaptive_cfg.get("runtime_high_pressure_phase2_min_long_prefill", adaptive_cfg.get("high_pressure_phase2_min_long_prefill", phase2_cfg.get("min_long_prefill", 768)))
        )
        phase2_cfg["runtime_low_pressure_escape_spillover_cap"] = int(
            adaptive_cfg.get("runtime_low_pressure_escape_spillover_cap", adaptive_cfg.get("low_pressure_escape_spillover_cap", phase2_cfg.get("execution_escape_spillover_cap", 1)))
        )
        phase2_cfg["runtime_high_pressure_escape_spillover_cap"] = int(
            adaptive_cfg.get("runtime_high_pressure_escape_spillover_cap", adaptive_cfg.get("high_pressure_escape_spillover_cap", phase2_cfg.get("execution_escape_spillover_cap", 3)))
        )
        phase2_cfg["runtime_low_pressure_escape_max_active"] = int(
            adaptive_cfg.get("runtime_low_pressure_escape_max_active", adaptive_cfg.get("low_pressure_escape_max_active", phase2_cfg.get("execution_escape_max_active", 2)))
        )
        phase2_cfg["runtime_high_pressure_escape_max_active"] = int(
            adaptive_cfg.get("runtime_high_pressure_escape_max_active", adaptive_cfg.get("high_pressure_escape_max_active", phase2_cfg.get("execution_escape_max_active", 5)))
        )
        phase2_cfg["runtime_disable_execution_escape_below_pressure"] = float(
            adaptive_cfg.get("runtime_disable_execution_escape_below_pressure", adaptive_cfg.get("disable_execution_escape_below_pressure", -1.0))
        )

        adapted["phase1"] = phase1_cfg
        adapted["phase2"] = phase2_cfg
        adapted["_adaptive_density_runtime"] = {
            "enabled": True,
            "density": str(density.get("name") or ""),
            "scope": "runtime_queue_pressure",
            "pressure_source": "scheduler_queue_waiting_short_wait_us_long_remaining_cap_hit_ema",
            "pressure_formula": "per-schedule weighted wall pressure with short-urgency discount",
            "workload_pressure_score": pressure_score,
            "pressure_current_rate": pressure_meta["current_rate"],
            "pressure_min_rate": pressure_meta["min_rate"],
            "pressure_max_rate": pressure_meta["max_rate"],
            "runtime_queue_pressure": True,
            "phase1_runtime_aggressive_long_fraction": phase1_cfg.get("runtime_aggressive_long_fraction"),
            "phase1_runtime_conservative_long_fraction": phase1_cfg.get("runtime_conservative_long_fraction"),
            "phase1_runtime_aggressive_ingress_target_chunk": phase1_cfg.get("runtime_aggressive_ingress_target_chunk"),
            "phase1_runtime_conservative_ingress_target_chunk": phase1_cfg.get("runtime_conservative_ingress_target_chunk"),
            "phase2_runtime_adaptive_enabled": phase2_cfg.get("runtime_adaptive_enabled"),
        }
        return adapted

    phase1_cfg["target_long_fraction"] = _lerp(
        float(adaptive_cfg.get("low_pressure_target_long_fraction", phase1_cfg.get("target_long_fraction", 0.33))),
        float(adaptive_cfg.get("high_pressure_target_long_fraction", phase1_cfg.get("target_long_fraction", 0.33))),
        pressure_score,
    )
    phase1_cfg["ingress_target_chunk"] = int(round(_lerp(
        float(adaptive_cfg.get("low_pressure_ingress_target_chunk", phase1_cfg.get("ingress_target_chunk", 768))),
        float(adaptive_cfg.get("high_pressure_ingress_target_chunk", phase1_cfg.get("ingress_target_chunk", 768))),
        pressure_score,
    )))

    phase2_cfg["min_hetero_ratio"] = _lerp(
        float(adaptive_cfg.get("low_pressure_phase2_min_hetero_ratio", phase2_cfg.get("min_hetero_ratio", 4.0))),
        float(adaptive_cfg.get("high_pressure_phase2_min_hetero_ratio", phase2_cfg.get("min_hetero_ratio", 4.0))),
        pressure_score,
    )
    phase2_cfg["min_pressure_ratio"] = _lerp(
        float(adaptive_cfg.get("low_pressure_phase2_min_pressure_ratio", phase2_cfg.get("min_pressure_ratio", 4.0))),
        float(adaptive_cfg.get("high_pressure_phase2_min_pressure_ratio", phase2_cfg.get("min_pressure_ratio", 4.0))),
        pressure_score,
    )
    phase2_cfg["min_long_prefill"] = int(round(_lerp(
        float(adaptive_cfg.get("low_pressure_phase2_min_long_prefill", phase2_cfg.get("min_long_prefill", 768))),
        float(adaptive_cfg.get("high_pressure_phase2_min_long_prefill", phase2_cfg.get("min_long_prefill", 768))),
        pressure_score,
    )))
    phase2_cfg["execution_escape_spillover_cap"] = int(round(_lerp(
        float(adaptive_cfg.get("low_pressure_escape_spillover_cap", phase2_cfg.get("execution_escape_spillover_cap", 3))),
        float(adaptive_cfg.get("high_pressure_escape_spillover_cap", phase2_cfg.get("execution_escape_spillover_cap", 3))),
        pressure_score,
    )))
    phase2_cfg["execution_escape_max_active"] = int(round(_lerp(
        float(adaptive_cfg.get("low_pressure_escape_max_active", phase2_cfg.get("execution_escape_max_active", 5))),
        float(adaptive_cfg.get("high_pressure_escape_max_active", phase2_cfg.get("execution_escape_max_active", 5))),
        pressure_score,
    )))

    disable_below = adaptive_cfg.get("disable_execution_escape_below_pressure")
    if disable_below is not None and pressure_score < float(disable_below):
        phase2_cfg["enable_execution_escape"] = False

    adapted["phase1"] = phase1_cfg
    adapted["phase2"] = phase2_cfg
    adapted["_adaptive_density_runtime"] = {
        "enabled": True,
        "density": str(density.get("name") or ""),
        "scope": str(adaptive_cfg.get("scope") or "workload_pressure"),
        "pressure_source": str(adaptive_cfg.get("pressure_source") or "configured_phase2_arrival_rate"),
        "pressure_formula": str(adaptive_cfg.get("pressure_formula") or "(current_rate - min_rate) / (max_rate - min_rate)"),
        "pressure_score": pressure_score,
        "pressure_current_rate": pressure_meta["current_rate"],
        "pressure_min_rate": pressure_meta["min_rate"],
        "pressure_max_rate": pressure_meta["max_rate"],
        "runtime_queue_pressure": bool(adaptive_cfg.get("runtime_queue_pressure", False)),
        "phase1_target_long_fraction": phase1_cfg.get("target_long_fraction"),
        "phase1_ingress_target_chunk": phase1_cfg.get("ingress_target_chunk"),
        "phase2_min_hetero_ratio": phase2_cfg.get("min_hetero_ratio"),
        "phase2_min_pressure_ratio": phase2_cfg.get("min_pressure_ratio"),
        "phase2_min_long_prefill": phase2_cfg.get("min_long_prefill"),
        "phase2_enable_execution_escape": phase2_cfg.get("enable_execution_escape"),
        "phase2_execution_escape_spillover_cap": phase2_cfg.get("execution_escape_spillover_cap"),
        "phase2_execution_escape_max_active": phase2_cfg.get("execution_escape_max_active"),
    }
    return adapted


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
    runtime_cfg = {
        "python_bin": str(eval_cfg.get("python_bin") or sys.executable),
        "vllm_mode": str(eval_cfg.get("vllm_mode") or "v1"),
        "trust_remote_code": bool(model.trust_remote_code),
        "warmup_iters": int(eval_cfg.get("warmup_iters", 2)),
        "repeats": int(eval_cfg.get("repeats", 3)),
        "timeout_sec": int(eval_cfg.get("timeout_sec", 240)),
        "max_new_tokens": int(eval_cfg.get("max_new_tokens", 64)),
        "max_model_len": int(model.max_model_len_override or eval_cfg.get("max_model_len", 3072)),
        "max_num_batched_tokens": int(eval_cfg.get("max_num_batched_tokens", 1536)),
        "gpu_memory_utilization": float(eval_cfg.get("gpu_memory_utilization", 0.60)),
        "queue_reorder_mode": str(eval_cfg.get("queue_reorder_mode", "sjf")),
        "queue_reorder_aging_quantum_us": int(eval_cfg.get("queue_reorder_aging_quantum_us", 20000)),
    }
    phase2_cfg = dict(config.get("phase2") or {})
    legacy_phase2_map = {
        "dispatch_mode": "phase2_dispatch_mode",
        "enable_execution_escape": "phase2_enable_execution_escape",
        "execution_escape_mode": "phase2_execution_escape_mode",
        "execution_escape_spillover_cap": "phase2_execution_escape_spillover_cap",
        "execution_escape_max_active": "phase2_execution_escape_max_active",
    }
    for target_key, legacy_key in legacy_phase2_map.items():
        if target_key not in phase2_cfg and legacy_key in eval_cfg:
            phase2_cfg[target_key] = eval_cfg.get(legacy_key)
    return {
        "evaluator": "tests/evaluate_waveslice_claims.py",
        "model": {
            "name": model.lut_name,
            "path": model_path,
        },
        "workload": {
            "requests_json": req_json,
            "lora_requests_json": lora_req_json,
        },
        "adapters": {
            "adapter_a": adapter_a,
            "adapter_b": adapter_b,
        },
        "runtime": runtime_cfg,
        "phase1": dict(config.get("phase1") or {}),
        "phase12_soft_gate": dict(config.get("phase12_soft_gate") or {}),
        "phase2": phase2_cfg,
    }


def _per_model_progress_rows(
    *,
    rows: list[dict[str, Any]],
    models: list[ResolvedModel],
    densities: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    density_names = [str(item.get("name") or "") for item in densities if isinstance(item, dict)]
    progress: list[dict[str, Any]] = []
    for model in models:
        model_rows = [row for row in rows if str(row.get("model_key") or "") == model.key]
        by_density = {str(row.get("density") or ""): row for row in model_rows}
        ok_rows = [row for row in model_rows if str(row.get("status") or "") == "ok"]

        def _mean(metric: str) -> Optional[float]:
            vals = [
                float(row[metric])
                for row in ok_rows
                if _float_or_none(row.get(metric)) is not None
            ]
            return (sum(vals) / len(vals)) if vals else None

        progress.append(
            {
                "model_key": model.key,
                "model_label": model.label,
                "model": model.model_id,
                "completed_density_count": len(model_rows),
                "ok_density_count": len(ok_rows),
                "expected_density_count": len(density_names),
                "density_status": {
                    density: str(by_density.get(density, {}).get("status") or "pending")
                    for density in density_names
                },
                "phase12_ttft_improve_mean": _mean("phase12_ttft_improve_mean"),
                "phase12_wall_improve_mean": _mean("phase12_wall_improve_mean"),
                "phase12_slowdown_improve_mean": _mean("phase12_slowdown_improve_mean"),
                "phase1_runtime_pressure_mean": _mean("phase1_runtime_pressure_mean"),
                "phase1_runtime_target_fraction_mean": _mean("phase1_runtime_target_fraction_mean"),
                "phase1_runtime_target_chunk_mean": _mean("phase1_runtime_target_chunk_mean"),
            }
        )
    return progress


def _run_single_case(
    *,
    model: ResolvedModel,
    density: dict[str, Any],
    config: dict[str, Any],
    dataset_source_path: Path,
    run_root: Path,
) -> dict[str, Any]:
    eval_cfg = config.get("eval", {})
    workload_cfg = config.get("workload", {})
    gpu_guard_timeout_sec = int(eval_cfg.get("gpu_guard_timeout_sec", 1800))
    gpu_guard_poll_interval_s = float(eval_cfg.get("gpu_guard_poll_interval_s", 5.0))

    local_snapshot = _resolve_local_snapshot(model.model_id)
    if model.model_path_mode == "model_id":
        model_path = model.model_id
    elif model.model_path_mode == "local_snapshot_required":
        model_path = local_snapshot or model.model_id
    else:
        model_path = local_snapshot or model.model_id
    effective_max_model_len = int(model.max_model_len_override or eval_cfg.get("max_model_len", 3072))

    adapters_root = str(run_root / "adapters")
    adapter_dir = os.path.join(adapters_root, safe_key(model.key))
    adapter_a, adapter_b = _ensure_adapters(
        base_model_path=model_path,
        out_dir=adapter_dir,
        trust_remote_code=model.trust_remote_code,
    )

    workloads_dir = _ensure_dir(run_root / "workloads" / density["name"])
    raw_dir = _ensure_dir(run_root / "raw" / density["name"])
    out_prefix = workloads_dir / safe_key(model.key)
    existing_req_json = Path(f"{out_prefix}_requests.json")
    existing_lora_req_json = Path(f"{out_prefix}_lora_requests.json")
    existing_meta_json = Path(f"{out_prefix}_meta.json")
    reuse_pool_root = str(workload_cfg.get("reuse_workload_pool_root") or "").strip()
    reuse_pool_prefix = None
    compatible_local_workload = (
        existing_req_json.exists()
        and existing_lora_req_json.exists()
        and _workload_meta_matches_model(
            meta_path=existing_meta_json,
            model=model,
            model_path=model_path,
            local_snapshot=local_snapshot,
            density=density,
            workload_cfg=workload_cfg,
            require_density_match=True,
        )
    )
    if reuse_pool_root:
        candidate = Path(reuse_pool_root) / safe_key(model.key)
        candidate_req = Path(f"{candidate}_requests.json")
        candidate_lora_req = Path(f"{candidate}_lora_requests.json")
        candidate_meta = Path(f"{candidate}_meta.json")
        if (
            candidate_req.exists()
            and candidate_lora_req.exists()
            and _workload_meta_matches_model(
                meta_path=candidate_meta,
                model=model,
                model_path=model_path,
                local_snapshot=local_snapshot,
                density=density,
                workload_cfg=workload_cfg,
                require_density_match=False,
            )
        ):
            reuse_pool_prefix = candidate
    runtime_ok, runtime_reason = _runtime_lut_is_valid(model.lut_name)
    if not runtime_ok:
        return {
            "density": density["name"],
            "density_phase1_arrival_rate": float(density["phase1_arrival_rate"]),
            "density_phase2_arrival_rate": float(density["phase2_arrival_rate"]),
            "density_scenario": str(density.get("scenario", "")),
            "density_reason": str(density.get("reason", "")),
            "model_key": model.key,
            "model_label": model.label,
            "model": model.model_id,
            "lut_name": model.lut_name,
            "model_reason": model.reason,
            "model_path": model_path,
            "adapter_a": adapter_a,
            "adapter_b": adapter_b,
            "status": "failed",
            "error": runtime_reason,
        }

    if compatible_local_workload:
        workload_proc = None
    elif reuse_pool_prefix is not None:
        workload_cmd = [
            sys.executable,
            "experiments/remix_dataset_workload.py",
            "--src-prefix",
            str(reuse_pool_prefix),
            "--out-prefix",
            str(out_prefix),
            "--arrival-mode",
            str(workload_cfg.get("arrival_mode", "poisson")),
            "--phase1-arrival-rate",
            str(float(density["phase1_arrival_rate"])),
            "--phase2-arrival-rate",
            str(float(density["phase2_arrival_rate"])),
            "--phase1-arrival-layout",
            str(workload_cfg.get("phase1_arrival_layout", "beneficiary_rich")),
            "--phase2-arrival-layout",
            str(workload_cfg.get("phase2_arrival_layout", "beneficiary_rich")),
            "--phase1-early-short-frac",
            str(float(workload_cfg.get("phase1_early_short_frac", 0.25))),
            "--phase2-early-short-frac",
            str(float(workload_cfg.get("phase2_early_short_frac", 0.20))),
            "--phase1-post-long-short-bias",
            str(float(workload_cfg.get("phase1_post_long_short_bias", 0.70))),
            "--phase2-post-long-short-bias",
            str(float(workload_cfg.get("phase2_post_long_short_bias", 0.60))),
        ]
        workload_env = os.environ.copy()
        workload_env.setdefault("HF_ENDPOINT", "https://huggingface.co")
        workload_env.setdefault("HF_DATASETS_OFFLINE", "1")
        workload_env.setdefault("HF_HUB_OFFLINE", "1")
        workload_env.setdefault("TRANSFORMERS_OFFLINE", "1")
        workload_proc = subprocess.run(workload_cmd, capture_output=True, text=True, check=False, env=workload_env)
    else:
        longbench_cfgs: list[str] = []
        for ds in config.get("datasets", []):
            if isinstance(ds, dict) and str(ds.get("key", "")).lower() == "longbench":
                longbench_cfgs.extend(ds.get("configs") or [])

        workload_cmd = [
            sys.executable,
            "experiments/build_dataset_workload.py",
            "--model-path",
            model_path,
            "--out-prefix",
            str(out_prefix),
            "--dataset-source-config",
            str(dataset_source_path),
            "--datasets",
            ",".join(str(ds["key"]) for ds in config.get("datasets", []) if isinstance(ds, dict)),
            "--longbench-configs",
            ",".join(longbench_cfgs),
            "--max-prompt-tokens",
            str(max(16, effective_max_model_len - int(eval_cfg.get("max_new_tokens", 64)) - 16)),
            "--sample-count",
            str(int(workload_cfg.get("sample_count", 256))),
            "--arrival-mode",
            str(workload_cfg.get("arrival_mode", "poisson")),
            "--phase1-arrival-layout",
            str(workload_cfg.get("phase1_arrival_layout", "beneficiary_rich")),
            "--phase2-arrival-layout",
            str(workload_cfg.get("phase2_arrival_layout", "beneficiary_rich")),
            "--phase1-early-short-frac",
            str(float(workload_cfg.get("phase1_early_short_frac", 0.25))),
            "--phase2-early-short-frac",
            str(float(workload_cfg.get("phase2_early_short_frac", 0.20))),
            "--phase1-post-long-short-bias",
            str(float(workload_cfg.get("phase1_post_long_short_bias", 0.70))),
            "--phase2-post-long-short-bias",
            str(float(workload_cfg.get("phase2_post_long_short_bias", 0.60))),
            "--phase1-arrival-rate",
            str(float(density["phase1_arrival_rate"])),
            "--phase2-arrival-rate",
            str(float(density["phase2_arrival_rate"])),
            "--phase1-short-count",
            str(int(density["phase1_short_count"])),
            "--phase1-long-count",
            str(int(density["phase1_long_count"])),
            "--phase2-short-count",
            str(int(density["phase2_short_count"])),
            "--phase2-long-count",
            str(int(density["phase2_long_count"])),
        ]
        if model.trust_remote_code:
            workload_cmd.append("--trust-remote-code")
        workload_env = os.environ.copy()
        workload_env.setdefault("HF_ENDPOINT", "https://huggingface.co")
        workload_env.setdefault("HF_DATASETS_OFFLINE", "1")
        workload_env.setdefault("HF_HUB_OFFLINE", "1")
        workload_env.setdefault("TRANSFORMERS_OFFLINE", "1")
        workload_proc = subprocess.run(workload_cmd, capture_output=True, text=True, check=False, env=workload_env)

    row: dict[str, Any] = {
        "density": density["name"],
        "density_phase1_arrival_rate": float(density["phase1_arrival_rate"]),
        "density_phase2_arrival_rate": float(density["phase2_arrival_rate"]),
        "density_scenario": str(density.get("scenario", "")),
        "density_reason": str(density.get("reason", "")),
        "model_key": model.key,
        "model_label": model.label,
        "model": model.model_id,
        "lut_name": model.lut_name,
        "model_reason": model.reason,
        "model_path": model_path,
        "adapter_a": adapter_a,
        "adapter_b": adapter_b,
        "status": "failed",
        "workload_stdout_tail": (
            "[reuse existing workload files]"
            if workload_proc is None
            else "\n".join((workload_proc.stdout or "").splitlines()[-20:])
        ),
        "workload_stderr_tail": (
            ""
            if workload_proc is None
            else "\n".join((workload_proc.stderr or "").splitlines()[-20:])
        ),
    }
    adaptive_runtime = dict(config.get("_adaptive_density_runtime") or {})
    if adaptive_runtime:
        row.update({f"adaptive_{key}": value for key, value in adaptive_runtime.items()})
    if workload_proc is not None and workload_proc.returncode != 0:
        row["error"] = f"build_dataset_workload exited with code {workload_proc.returncode}"
        return row

    req_json = str(existing_req_json if existing_req_json.exists() else Path(f"{out_prefix}_requests.json"))
    lora_req_json = str(existing_lora_req_json if existing_lora_req_json.exists() else Path(f"{out_prefix}_lora_requests.json"))
    meta_json = str(existing_meta_json if existing_meta_json.exists() else Path(f"{out_prefix}_meta.json"))
    out_json = raw_dir / f"{safe_key(model.key)}_dataset_eval.json"
    if out_json.exists() and Path(meta_json).exists():
        row["status"] = "ok"
        row["result_json"] = str(out_json)
        row["workload_meta_json"] = meta_json
        row["stdout_tail"] = "[reuse existing result json]"
        row["stderr_tail"] = ""
        row.update(_extract_summary_from_result_json(out_json))
        meta = json.loads(Path(meta_json).read_text(encoding="utf-8"))
        row["phase1_request_count"] = meta.get("phase1_request_count")
        row["phase2_request_count"] = meta.get("phase2_request_count")
        row["dataset_short_a_tokens"] = meta.get("short_a_tokens")
        row["dataset_short_b_tokens"] = meta.get("short_b_tokens")
        row["dataset_long_a_tokens"] = meta.get("long_a_tokens")
        row["dataset_long_b_tokens"] = meta.get("long_b_tokens")
        return row
    _purge_experiment_processes(
        reason=f"before-case:{density['name']}:{model.key}",
        gpu_lock_path=_GPU_LOCK_PATH,
        experiment_proc_patterns=_EXPERIMENT_PROC_PATTERNS,
        preserve_pid=os.getpid(),
    )
    _wait_for_clean_gpu(
        label=f"{density['name']}:{model.label}",
        gpu_lock_path=_GPU_LOCK_PATH,
        timeout_sec=gpu_guard_timeout_sec,
        poll_interval_s=gpu_guard_poll_interval_s,
    )
    case_eval_config = _case_eval_config(
        model=model,
        model_path=model_path,
        req_json=req_json,
        lora_req_json=lora_req_json,
        adapter_a=adapter_a,
        adapter_b=adapter_b,
        config=config,
        eval_cfg=eval_cfg,
    )
    eval_cmd, eval_env = _build_eval_invocation(
        case_eval_config,
        out_json_override=str(out_json),
    )
    eval_cmd.append("--no-serialize-gpu-tests")
    eval_env.setdefault("HF_DATASETS_OFFLINE", "1")
    eval_env.setdefault("HF_HUB_OFFLINE", "1")
    eval_env.setdefault("TRANSFORMERS_OFFLINE", "1")
    logs_dir = _ensure_dir(run_root / "logs" / density["name"])
    eval_stdout_path = logs_dir / f"{safe_key(model.key)}_eval.stdout.log"
    eval_stderr_path = logs_dir / f"{safe_key(model.key)}_eval.stderr.log"
    eval_returncode = 1
    with eval_stdout_path.open("w", encoding="utf-8") as stdout_f, eval_stderr_path.open(
        "w", encoding="utf-8"
    ) as stderr_f:
        eval_proc = subprocess.Popen(
            eval_cmd,
            stdout=stdout_f,
            stderr=stderr_f,
            text=True,
            env=eval_env,
            start_new_session=True,
        )
        try:
            eval_returncode = eval_proc.wait()
        finally:
            _kill_process_group(eval_proc.pid)
            _clear_gpu_lock(_GPU_LOCK_PATH)
            time.sleep(1.0)
            _purge_experiment_processes(
                reason=f"after-case:{density['name']}:{model.key}",
                gpu_lock_path=_GPU_LOCK_PATH,
                experiment_proc_patterns=_EXPERIMENT_PROC_PATTERNS,
                preserve_pid=os.getpid(),
            )
            _wait_for_clean_gpu(
                label=f"cleanup:{density['name']}:{model.label}",
                gpu_lock_path=_GPU_LOCK_PATH,
                timeout_sec=min(gpu_guard_timeout_sec, 120),
                poll_interval_s=1.0,
            )
    stdout_tail = eval_stdout_path.read_text(encoding="utf-8", errors="replace").splitlines()[-20:]
    stderr_tail = eval_stderr_path.read_text(encoding="utf-8", errors="replace").splitlines()[-20:]
    row["stdout_tail"] = "\n".join(stdout_tail)
    row["stderr_tail"] = "\n".join(stderr_tail)
    row["result_json"] = str(out_json)
    row["workload_meta_json"] = meta_json
    row["stdout_log"] = str(eval_stdout_path)
    row["stderr_log"] = str(eval_stderr_path)
    if eval_returncode != 0:
        row["error"] = f"evaluate_waveslice_claims exited with code {eval_returncode}"
        return row
    row["status"] = "ok"
    row.update(_extract_summary_from_result_json(out_json))
    meta = json.loads(Path(meta_json).read_text(encoding="utf-8"))
    row["phase1_request_count"] = meta.get("phase1_request_count")
    row["phase2_request_count"] = meta.get("phase2_request_count")
    row["dataset_short_a_tokens"] = meta.get("short_a_tokens")
    row["dataset_short_b_tokens"] = meta.get("short_b_tokens")
    row["dataset_long_a_tokens"] = meta.get("long_a_tokens")
    row["dataset_long_b_tokens"] = meta.get("long_b_tokens")
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description="Configurable open-workload execution-escape supplement suite.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-name", default="")
    parser.add_argument("--model-keys", default="", help="Optional comma-separated model keys to restrict the selected model set.")
    parser.add_argument("--dataset-keys", default="", help="Optional comma-separated dataset keys to restrict the selected dataset set.")
    parser.add_argument("--densities", default="", help="Optional comma-separated density names to restrict the selected density set.")
    parser.add_argument("--reuse-csv", default="", help="Reuse an existing CSV instead of running new GPU jobs.")
    parser.add_argument("--reuse-results-dir", default="", help="Directory containing existing result JSONs to copy into metadata/raw.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--write-rationale",
        action="store_true",
        help="Write experiment_rationale.md. Disabled by default so later reruns do not auto-generate it.",
    )
    args = parser.parse_args()

    repo_root = Path.cwd()
    config = _load_config(args.config)
    resolved_models, model_diagnostics = _resolve_selected_models(config, args.model_keys)
    selected_datasets, dataset_diagnostics = _resolve_selected_datasets(config, args.dataset_keys)
    selected_densities = _resolve_selected_densities(config, args.densities)
    if not resolved_models:
        raise RuntimeError("no models selected for open-workload suite")
    if not selected_datasets:
        raise RuntimeError("no datasets selected for open-workload suite")
    if not selected_densities:
        raise RuntimeError("no densities selected for open-workload suite")
    effective_config = deepcopy(config)
    effective_config["models"] = [asdict(model) for model in resolved_models]
    effective_config["datasets"] = selected_datasets
    effective_config.setdefault("workload", {})
    effective_config["workload"] = dict(effective_config.get("workload") or {})
    effective_config["workload"]["densities"] = selected_densities
    effective_config["resource_selection_diagnostics"] = {
        "models": model_diagnostics,
        "datasets": dataset_diagnostics,
    }
    reference_densities = [
        dict(item)
        for item in list(config.get("workload", {}).get("densities") or selected_densities)
        if isinstance(item, dict)
    ]
    run_name = args.run_name or time.strftime("%Y%m%d_%H%M%S")
    out_root = Path(str(effective_config.get("out_root", "results/openworkload_execescape_tradeoff")))
    run_root = out_root / run_name
    metadata_dir = _ensure_dir(run_root / "metadata")
    figures_dir = _ensure_dir(run_root / "figures")
    raw_dir = _ensure_dir(run_root / "raw")
    _ensure_dir(run_root / "workloads")

    dataset_payload = _build_dataset_source_payload(effective_config)
    _write_json(metadata_dir / "resolved_config.json", effective_config)
    _write_json(metadata_dir / "models.json", [asdict(m) for m in resolved_models])
    _write_json(metadata_dir / "model_selection_diagnostics.json", model_diagnostics)
    _write_json(metadata_dir / "optional_models.json", effective_config.get("optional_model_extensions", []))
    _write_json(metadata_dir / "datasets.json", selected_datasets)
    _write_json(metadata_dir / "dataset_selection_diagnostics.json", dataset_diagnostics)
    _write_json(metadata_dir / "optional_datasets.json", effective_config.get("optional_dataset_extensions", []))
    _write_json(metadata_dir / "densities.json", selected_densities)
    _write_json(metadata_dir / "dataset_sources_resolved.json", dataset_payload)

    rows: list[dict[str, Any]] = _load_existing_rows(metadata_dir / "suite_results.json")
    reuse_results_dir = Path(args.reuse_results_dir).resolve() if args.reuse_results_dir else None
    if args.reuse_csv:
        rows = _rows_from_existing_csv(Path(args.reuse_csv), safe_key_fn=safe_key)
        _enrich_rows_with_config(rows, resolved_models, selected_densities)
        _copy_existing_artifacts(
            rows,
            raw_dir,
            repo_root,
            reuse_results_dir,
            resolve_existing_result_json=_resolve_existing_result_json,
            resolve_existing_meta_json=_resolve_existing_meta_json,
            safe_key_fn=safe_key,
        )
        for row in rows:
            result_json = _resolve_existing_result_json(row, reuse_results_dir, safe_key_fn=safe_key)
            if result_json is not None:
                row["result_json"] = str(result_json)
                row.update({k: v for k, v in _extract_summary_from_result_json(result_json).items() if v is not None})
            meta_json = _resolve_existing_meta_json(row, repo_root)
            if meta_json is not None:
                meta = json.loads(meta_json.read_text(encoding="utf-8"))
                row["workload_meta_json"] = str(meta_json)
                row["phase1_request_count"] = meta.get("phase1_request_count")
                row["phase2_request_count"] = meta.get("phase2_request_count")
                row["dataset_short_a_tokens"] = meta.get("short_a_tokens")
                row["dataset_short_b_tokens"] = meta.get("short_b_tokens")
                row["dataset_long_a_tokens"] = meta.get("long_a_tokens")
                row["dataset_long_b_tokens"] = meta.get("long_b_tokens")
    else:
        dataset_source_path = metadata_dir / "dataset_sources_resolved.json"
        done_keys = _completed_case_keys(rows)
        dry_run_done = False
        for model in resolved_models:
            for density in selected_densities:
                if not isinstance(density, dict):
                    continue
                case_key = (str(density.get("name") or "").strip(), model.key)
                if case_key in done_keys:
                    print(
                        f"[SupplementSuite] skip density={density.get('name')} "
                        f"model={model.label} reason=already_completed",
                        flush=True,
                    )
                    continue
                print(
                    f"[SupplementSuite] start density={density.get('name')} "
                    f"model={model.label}",
                    flush=True,
                )
                case_config = _adapt_config_for_density(effective_config, density, reference_densities)
                adaptive_runtime = dict(case_config.get("_adaptive_density_runtime") or {})
                if adaptive_runtime:
                    print(
                        "[SupplementSuite] adaptive-policy "
                        f"density={density.get('name')} "
                        f"scope={adaptive_runtime.get('scope')} "
                        f"source={adaptive_runtime.get('pressure_source')} "
                        f"workload_pressure={adaptive_runtime.get('pressure_score', adaptive_runtime.get('workload_pressure_score'))} "
                        f"chunk={adaptive_runtime.get('phase1_ingress_target_chunk', adaptive_runtime.get('phase1_runtime_aggressive_ingress_target_chunk'))} "
                        f"fraction={adaptive_runtime.get('phase1_target_long_fraction', adaptive_runtime.get('phase1_runtime_aggressive_long_fraction'))} "
                        f"phase2_runtime={adaptive_runtime.get('phase2_runtime_adaptive_enabled', adaptive_runtime.get('phase2_enable_execution_escape'))}",
                        flush=True,
                    )
                if args.dry_run:
                    row = {
                        "density": str(density.get("name") or ""),
                        "density_phase1_arrival_rate": float(density.get("phase1_arrival_rate") or 0.0),
                        "density_phase2_arrival_rate": float(density.get("phase2_arrival_rate") or 0.0),
                        "density_scenario": str(density.get("scenario") or ""),
                        "density_reason": str(density.get("reason") or ""),
                        "model_key": model.key,
                        "model_label": model.label,
                        "model": model.model_id,
                        "lut_name": model.lut_name,
                        "model_reason": model.reason,
                        "status": "dry_run",
                    }
                    if adaptive_runtime:
                        row.update({f"adaptive_{key}": value for key, value in adaptive_runtime.items()})
                else:
                    row = _run_single_case(
                        model=model,
                        density=density,
                        config=case_config,
                        dataset_source_path=dataset_source_path,
                        run_root=run_root,
                    )
                rows = [
                    existing
                    for existing in rows
                    if (
                        str(existing.get("density") or "").strip(),
                        str(existing.get("model_key") or "").strip(),
                    ) != case_key
                ]
                rows.append(row)
                if str(row.get("status", "")).strip().lower() == "ok":
                    done_keys.add(case_key)
                _write_csv(metadata_dir / "suite_results.csv", rows)
                _write_json(metadata_dir / "suite_results.json", rows)
                print(
                    "[SupplementSuite] done "
                    f"density={density.get('name')} model={model.label} "
                    f"status={row.get('status')} "
                    f"ttft={row.get('phase12_ttft_improve_mean')} "
                    f"wall={row.get('phase12_wall_improve_mean')} "
                        f"slow={row.get('phase12_slowdown_improve_mean')}",
                    flush=True,
                )
                if args.dry_run:
                    dry_run_done = True
                    break
            model_progress = _per_model_progress_rows(
                rows=rows,
                models=resolved_models,
                densities=selected_densities,
            )
            _write_json(metadata_dir / "per_model_progress.json", model_progress)
            current_progress = next(
                (item for item in model_progress if item.get("model_key") == model.key),
                None,
            )
            if current_progress is not None:
                print(
                    "[SupplementSuite] model_complete "
                    f"model={model.label} "
                    f"ok={current_progress.get('ok_density_count')}/{current_progress.get('expected_density_count')} "
                    f"ttft={current_progress.get('phase12_ttft_improve_mean')} "
                    f"wall={current_progress.get('phase12_wall_improve_mean')} "
                    f"slow={current_progress.get('phase12_slowdown_improve_mean')} "
                    f"statuses={current_progress.get('density_status')}",
                    flush=True,
                )
            if args.dry_run:
                dry_run_done = True
                break
        if dry_run_done:
            pass

    _write_csv(metadata_dir / "suite_results.csv", rows)
    _write_json(metadata_dir / "suite_results.json", rows)
    _write_json(
        metadata_dir / "per_model_progress.json",
        _per_model_progress_rows(
            rows=rows,
            models=resolved_models,
            densities=selected_densities,
        ),
    )

    aggregate_by_model = _aggregate_rows(
        rows,
        ["model_label", "model"],
        [
            "phase1_ttft_improve_mean",
            "phase1_wall_improve_mean",
            "phase2_ttft_improve_mean",
            "phase2_wall_improve_mean",
            "phase12_ttft_improve_mean",
            "phase12_wall_improve_mean",
            "phase12_slowdown_improve_mean",
        ],
    )
    aggregate_by_density = _aggregate_rows(
        rows,
        ["density"],
        [
            "phase1_ttft_improve_mean",
            "phase1_wall_improve_mean",
            "phase2_ttft_improve_mean",
            "phase2_wall_improve_mean",
            "phase12_ttft_improve_mean",
            "phase12_wall_improve_mean",
            "phase12_slowdown_improve_mean",
        ],
    )
    _write_json(metadata_dir / "aggregate_by_model.json", aggregate_by_model)
    _write_json(metadata_dir / "aggregate_by_density.json", aggregate_by_density)

    _plot_metric_by_model(
        rows,
        metric="phase1_ttft_improve_mean",
        ylabel="TTFT Improvement (baseline / Wave-Slice)",
        title="Phase I TTFT Improvement by Model",
        out_path=figures_dir / "phase1_ttft_by_model.png",
    )
    _plot_metric_by_model(
        rows,
        metric="phase1_wall_improve_mean",
        ylabel="Round Wall-Time Improvement (baseline / Wave-Slice)",
        title="Phase I Wall-Time Tradeoff by Model",
        out_path=figures_dir / "phase1_wall_by_model.png",
    )
    _plot_metric_by_model(
        rows,
        metric="phase2_ttft_improve_mean",
        ylabel="TTFT Improvement (baseline / Wave-Slice)",
        title="Phase II TTFT Improvement by Model",
        out_path=figures_dir / "phase2_ttft_by_model.png",
    )
    _plot_metric_by_model(
        rows,
        metric="phase2_wall_improve_mean",
        ylabel="Round Wall-Time Improvement (baseline / Wave-Slice)",
        title="Phase II Wall-Time Tradeoff by Model",
        out_path=figures_dir / "phase2_wall_by_model.png",
    )
    _plot_metric_by_model(
        rows,
        metric="phase12_ttft_improve_mean",
        ylabel="TTFT Improvement (baseline / Wave-Slice)",
        title="Phase I+II TTFT Improvement by Model",
        out_path=figures_dir / "phase12_ttft_by_model.png",
    )
    _plot_metric_by_model(
        rows,
        metric="phase12_wall_improve_mean",
        ylabel="Round Wall-Time Improvement (baseline / Wave-Slice)",
        title="Phase I+II Wall-Time Tradeoff by Model",
        out_path=figures_dir / "phase12_wall_by_model.png",
    )
    _plot_metric_by_model(
        rows,
        metric="phase12_ttft_improve_mean",
        ylabel="TTFT Improvement (baseline / Wave-Slice)",
        title="TTFT Improvement by Model",
        out_path=figures_dir / "ttft_by_model.png",
    )
    _plot_metric_by_model(
        rows,
        metric="phase12_wall_improve_mean",
        ylabel="Round Wall-Time Improvement (baseline / Wave-Slice)",
        title="Wall-Time Tradeoff by Model",
        out_path=figures_dir / "wall_by_model.png",
    )
    _plot_tradeoff_scatter(rows, figures_dir / "ttft_vs_wall_scatter.png")
    _plot_density_summary(rows, figures_dir / "density_summary.png")
    _write_result_summary_markdown(run_root, rows)
    if args.write_rationale:
        _write_rationale_markdown(run_root, config, rows)

    print(f"[SupplementSuite] run_root={run_root}")
    print(f"[SupplementSuite] metadata={metadata_dir}")
    print(f"[SupplementSuite] figures={figures_dir}")
    print(f"[SupplementSuite] rows={len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
