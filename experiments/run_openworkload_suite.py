from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections.abc import Callable
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from typing import Any

os.environ.setdefault("VLLM_NO_USAGE_STATS", "1")
from experiments.catalog import safe_key
from experiments.local_resources import select_local_dataset_entries, select_local_model_entries
from experiments.model_assets import (
    ensure_adapters as _ensure_adapters,
    ensure_model_available as _ensure_model_available,
)
from experiments.openworkload_models import ResolvedModel, resolve_model_entry, runtime_lut_is_valid
from experiments.openworkload_results import (
    aggregate_rows,
    write_result_summary_markdown,
)
from experiments.openworkload_support import (
    apply_hf_resource_env,
    build_dataset_source_payload,
    completed_case_keys,
    ensure_dir,
    extract_summary_from_result_json,
    load_config,
    load_existing_rows,
    project_path,
    repo_root,
    resource_policy,
    workload_meta_matches_model,
    write_csv,
    write_json,
)
from experiments.run_frozen_eval_config import build_eval_invocation

_load_config = load_config
_SUMMARY_METRICS = (
    "phase1_ttft_improve_mean",
    "phase1_wall_improve_mean",
    "phase2_ttft_improve_mean",
    "phase2_wall_improve_mean",
    "phase12_ttft_improve_mean",
    "phase12_wall_improve_mean",
    "phase12_slowdown_improve_mean",
)
_WORKLOAD_DEFAULTS = {
    "arrival_mode": "poisson",
    "phase1_arrival_layout": "beneficiary_rich",
    "phase2_arrival_layout": "beneficiary_rich",
    "phase1_early_short_frac": 0.25,
    "phase2_early_short_frac": 0.20,
    "phase1_post_long_short_bias": 0.70,
    "phase2_post_long_short_bias": 0.60,
}


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


def _base_row(
    model: ResolvedModel, density: dict[str, Any], model_path: str, **extra: Any
) -> dict[str, Any]:
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
        **extra,
    }


def _attach_workload_meta(row: dict[str, Any], meta_path: Path) -> None:
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    row.update(
        {
            target: meta.get(source)
            for target, source in (
                ("phase1_request_count", "phase1_request_count"),
                ("phase2_request_count", "phase2_request_count"),
                ("dataset_short_a_tokens", "short_a_tokens"),
                ("dataset_short_b_tokens", "short_b_tokens"),
                ("dataset_long_a_tokens", "long_a_tokens"),
                ("dataset_long_b_tokens", "long_b_tokens"),
            )
        }
    )


def _workload_common_args(workload: dict[str, Any], density: dict[str, Any]) -> list[str]:
    args = []
    for key, default in _WORKLOAD_DEFAULTS.items():
        args.extend((f"--{key.replace('_', '-')}", str(workload.get(key, default))))
    for phase in ("phase1", "phase2"):
        args.extend((f"--{phase}-arrival-rate", str(float(density[f"{phase}_arrival_rate"]))))
        for kind in ("short", "long"):
            args.extend((f"--{phase}-{kind}-count", str(int(density[f"{phase}_{kind}_count"]))))
    return args


def _workload_prefix_is_valid(
    prefix: Path,
    *,
    model: ResolvedModel,
    model_path: str,
    snapshot: str | None,
    density: dict[str, Any],
    workload: dict[str, Any],
    density_match: bool,
) -> bool:
    return (
        Path(f"{prefix}_requests.json").exists()
        and Path(f"{prefix}_lora_requests.json").exists()
        and workload_meta_matches_model(
            meta_path=Path(f"{prefix}_meta.json"),
            model=model,
            model_path=model_path,
            local_snapshot=snapshot,
            density=density,
            workload_cfg=workload,
            require_density_match=density_match,
        )
    )


def _prepare_workload(
    *,
    model: ResolvedModel,
    model_path: str,
    snapshot: str | None,
    density: dict[str, Any],
    config: dict[str, Any],
    source_path: Path,
    prefix: Path,
) -> subprocess.CompletedProcess[str] | None:
    workload, evaluation = config.get("workload", {}), config.get("eval", {})
    if _workload_prefix_is_valid(
        prefix,
        model=model,
        model_path=model_path,
        snapshot=snapshot,
        density=density,
        workload=workload,
        density_match=True,
    ):
        return None
    pool = str(workload.get("reuse_workload_pool_root") or "").strip()
    pool_prefix = project_path(pool) / safe_key(model.key) if pool else None
    if pool_prefix and _workload_prefix_is_valid(
        pool_prefix,
        model=model,
        model_path=model_path,
        snapshot=snapshot,
        density=density,
        workload=workload,
        density_match=False,
    ):
        cmd = [
            sys.executable,
            "experiments/remix_dataset_workload.py",
            "--src-prefix",
            str(pool_prefix),
            "--out-prefix",
            str(prefix),
        ]
    else:
        longbench = [
            name
            for dataset in config.get("datasets", [])
            if isinstance(dataset, dict) and str(dataset.get("key", "")).lower() == "longbench"
            for name in dataset.get("configs") or []
        ]
        max_len = int(model.max_model_len_override or evaluation.get("max_model_len", 3072))
        cmd = [
            sys.executable,
            "experiments/build_dataset_workload.py",
            "--model-path",
            model_path,
            "--out-prefix",
            str(prefix),
            "--dataset-source-config",
            str(source_path),
            "--datasets",
            ",".join(
                str(item["key"]) for item in config.get("datasets", []) if isinstance(item, dict)
            ),
            "--longbench-configs",
            ",".join(longbench),
            "--max-prompt-tokens",
            str(max(16, max_len - int(evaluation.get("max_new_tokens", 64)) - 16)),
            "--sample-count",
            str(int(workload.get("sample_count", 256))),
        ] + (["--trust-remote-code"] if model.trust_remote_code else [])
    cmd += _workload_common_args(workload, density)
    env = apply_hf_resource_env(
        os.environ.copy()
        | {"HF_ENDPOINT": os.environ.get("HF_ENDPOINT", "https://huggingface.co")},
        config,
    )
    return subprocess.run(
        cmd, capture_output=True, text=True, check=False, env=env, cwd=str(repo_root())
    )


def _run_single_case(
    *,
    model: ResolvedModel,
    density: dict[str, Any],
    config: dict[str, Any],
    dataset_source_path: Path,
    run_root: Path,
) -> dict[str, Any]:
    evaluation = config.get("eval", {})
    resources = resource_policy(config)
    snapshot = _ensure_model_available(
        model.model_id,
        auto_download=bool(resources["auto_download"]),
        local_files_only=bool(resources["offline"]),
    )
    if model.model_path_mode == "model_id":
        model_path = model.model_id
    elif (
        model.model_path_mode == "local_snapshot_required" and not snapshot and resources["offline"]
    ):
        return _base_row(
            model,
            density,
            model.model_id,
            status="failed",
            error=f"local snapshot required but not found for {model.model_id}",
        )
    else:
        model_path = snapshot or model.model_id
    adapter_dir = run_root / "adapters" / safe_key(model.key)
    adapter_a, adapter_b = _ensure_adapters(
        base_model_path=model_path,
        out_dir=str(adapter_dir),
        trust_remote_code=model.trust_remote_code,
    )
    runtime_ok, runtime_reason = runtime_lut_is_valid(model.lut_name)
    row = _base_row(
        model,
        density,
        model_path,
        adapter_a=adapter_a,
        adapter_b=adapter_b,
        status="failed",
    )
    if not runtime_ok:
        row["error"] = runtime_reason
        return row
    prefix = ensure_dir(run_root / "workloads" / density["name"]) / safe_key(model.key)
    raw_dir = ensure_dir(run_root / "raw" / density["name"])
    workload_proc = _prepare_workload(
        model=model,
        model_path=model_path,
        snapshot=snapshot,
        density=density,
        config=config,
        source_path=dataset_source_path,
        prefix=prefix,
    )
    row.update(
        {
            f"adaptive_{key}": value
            for key, value in (config.get("_adaptive_density_runtime") or {}).items()
        }
    )
    if workload_proc is not None and workload_proc.returncode != 0:
        row["error"] = f"build_dataset_workload exited with code {workload_proc.returncode}"
        return row
    request_json, lora_json, meta_json = (
        Path(f"{prefix}_{suffix}.json") for suffix in ("requests", "lora_requests", "meta")
    )
    out_json = raw_dir / f"{safe_key(model.key)}_dataset_eval.json"
    row.update(result_json=str(out_json), workload_meta_json=str(meta_json))
    if out_json.exists() and meta_json.exists():
        row["status"] = "ok"
        row.update(extract_summary_from_result_json(out_json))
        _attach_workload_meta(row, meta_json)
        return row
    case_config = _case_eval_config(
        model=model,
        model_path=model_path,
        req_json=str(request_json),
        lora_req_json=str(lora_json),
        adapter_a=adapter_a,
        adapter_b=adapter_b,
        config=config,
        eval_cfg=evaluation,
    )
    cmd, env = build_eval_invocation(case_config, out_json_override=str(out_json))
    logs = ensure_dir(run_root / "logs" / density["name"])
    stdout_path, stderr_path = (
        logs / f"{safe_key(model.key)}_eval.{stream}.log" for stream in ("stdout", "stderr")
    )
    with (
        stdout_path.open("w", encoding="utf-8") as stdout,
        stderr_path.open("w", encoding="utf-8") as stderr,
    ):
        returncode = subprocess.run(
            cmd,
            stdout=stdout,
            stderr=stderr,
            text=True,
            env=apply_hf_resource_env(env, config),
            cwd=repo_root(),
            check=False,
        ).returncode
    row.update(stdout_log=str(stdout_path), stderr_log=str(stderr_path))
    if returncode:
        row["error"] = f"evaluate_waveslice_claims exited with code {returncode}"
        return row
    row["status"] = "ok"
    row.update(extract_summary_from_result_json(out_json))
    _attach_workload_meta(row, meta_json)
    return row


def _write_initial_metadata(
    directory: Path,
    config: dict[str, Any],
    models: list[ResolvedModel],
    model_diagnostics: list[dict[str, Any]],
    datasets: list[dict[str, Any]],
    dataset_diagnostics: list[dict[str, Any]],
    densities: list[dict[str, Any]],
) -> Path:
    payloads = {
        "resolved_config": config,
        "models": [asdict(model) for model in models],
        "model_selection_diagnostics": model_diagnostics,
        "optional_models": config.get("optional_model_extensions", []),
        "datasets": datasets,
        "dataset_selection_diagnostics": dataset_diagnostics,
        "optional_datasets": config.get("optional_dataset_extensions", []),
        "densities": densities,
        "dataset_sources_resolved": build_dataset_source_payload(config),
    }
    for name, payload in payloads.items():
        write_json(directory / f"{name}.json", payload)
    return directory / "dataset_sources_resolved.json"


def _dry_run_row(
    model: ResolvedModel, density: dict[str, Any], adaptive: dict[str, Any]
) -> dict[str, Any]:
    row = _base_row(model, density, model.model_id, status="dry_run")
    row.pop("model_path")
    row.update({f"adaptive_{key}": value for key, value in adaptive.items()})
    return row


def _run_cases(
    *,
    rows: list[dict[str, Any]],
    models: list[ResolvedModel],
    densities: list[dict[str, Any]],
    config: dict[str, Any],
    references: list[dict[str, Any]],
    source_path: Path,
    run_root: Path,
    metadata: Path,
    dry_run: bool,
) -> list[dict[str, Any]]:
    done = completed_case_keys(rows)
    for model in models:
        for density in densities:
            key = str(density.get("name") or "").strip(), model.key
            if key in done:
                print(
                    f"[SupplementSuite] skip density={key[0]} model={model.label} reason=already_completed",
                    flush=True,
                )
                continue
            print(f"[SupplementSuite] start density={key[0]} model={model.label}", flush=True)
            case_config = _adapt_config_for_density(config, density, references)
            adaptive = dict(case_config.get("_adaptive_density_runtime") or {})
            if adaptive:
                print(
                    f"[SupplementSuite] adaptive-policy density={key[0]} scope={adaptive.get('scope')} "
                    f"source={adaptive.get('pressure_source')} workload_pressure={adaptive.get('pressure_score', adaptive.get('workload_pressure_score'))}",
                    flush=True,
                )
            row = (
                _dry_run_row(model, density, adaptive)
                if dry_run
                else _run_single_case(
                    model=model,
                    density=density,
                    config=case_config,
                    dataset_source_path=source_path,
                    run_root=run_root,
                )
            )
            rows = [
                existing
                for existing in rows
                if (
                    str(existing.get("density") or "").strip(),
                    str(existing.get("model_key") or "").strip(),
                )
                != key
            ]
            rows.append(row)
            if row.get("status") == "ok":
                done.add(key)
            write_csv(metadata / "suite_results.csv", rows)
            write_json(metadata / "suite_results.json", rows)
            print(
                f"[SupplementSuite] done density={key[0]} model={model.label} status={row.get('status')} "
                f"ttft={row.get('phase12_ttft_improve_mean')} wall={row.get('phase12_wall_improve_mean')} "
                f"slow={row.get('phase12_slowdown_improve_mean')}",
                flush=True,
            )
            if dry_run:
                break
        if dry_run:
            break
    return rows


def _finish_outputs(run_root: Path, metadata: Path, rows: list[dict[str, Any]]) -> None:
    write_csv(metadata / "suite_results.csv", rows)
    write_json(metadata / "suite_results.json", rows)
    write_json(
        metadata / "aggregate_by_model.json",
        aggregate_rows(rows, ["model_label", "model"], list(_SUMMARY_METRICS)),
    )
    write_json(
        metadata / "aggregate_by_density.json",
        aggregate_rows(rows, ["density"], list(_SUMMARY_METRICS)),
    )
    write_result_summary_markdown(run_root, rows)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the maintained open-workload scheduler-priority suite."
    )
    parser.add_argument("--config", required=True)
    for name in ("run-name", "model-keys", "dataset-keys", "densities"):
        parser.add_argument(f"--{name}", default="")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    config = load_config(args.config)
    models, model_diagnostics = _resolve_selected_models(config, args.model_keys)
    datasets, dataset_diagnostics = _resolve_selected_datasets(config, args.dataset_keys)
    densities = _resolve_selected_densities(config, args.densities)
    if not models or not datasets or not densities:
        missing = "models" if not models else "datasets" if not datasets else "densities"
        raise RuntimeError(f"no {missing} selected for open-workload suite")
    effective = deepcopy(config)
    effective.update(
        models=[asdict(model) for model in models],
        datasets=datasets,
        resource_selection_diagnostics={
            "models": model_diagnostics,
            "datasets": dataset_diagnostics,
        },
    )
    effective["workload"] = dict(effective.get("workload") or {}) | {"densities": densities}
    references = [
        dict(item)
        for item in (config.get("workload") or {}).get("densities") or densities
        if isinstance(item, dict)
    ]
    run_root = project_path(
        str(effective.get("out_root", "results/openworkload_priority_cashout"))
    ) / (args.run_name or time.strftime("%Y%m%d_%H%M%S"))
    metadata, raw = (ensure_dir(run_root / name) for name in ("metadata", "raw"))
    ensure_dir(run_root / "workloads")
    source_path = _write_initial_metadata(
        metadata, effective, models, model_diagnostics, datasets, dataset_diagnostics, densities
    )
    rows = _run_cases(
        rows=load_existing_rows(metadata / "suite_results.json"),
        models=models,
        densities=densities,
        config=effective,
        references=references,
        source_path=source_path,
        run_root=run_root,
        metadata=metadata,
        dry_run=args.dry_run,
    )
    _finish_outputs(run_root, metadata, rows)
    print(f"[SupplementSuite] run_root={run_root}")
    print(f"[SupplementSuite] metadata={metadata}")
    print(f"[SupplementSuite] rows={len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
