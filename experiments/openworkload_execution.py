"""Workload preparation and case execution for the open-workload suite."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from experiments.catalog import safe_key
from experiments.model_assets import (
    ensure_adapters as _ensure_adapters,
    ensure_model_available as _ensure_model_available,
)
from experiments.openworkload_config import _adapt_config_for_density, _case_eval_config
from experiments.openworkload_models import ResolvedModel, runtime_lut_is_valid
from experiments.openworkload_support import (
    apply_hf_resource_env,
    completed_case_keys,
    ensure_dir,
    extract_summary_from_result_json,
    project_path,
    repo_root,
    resource_policy,
    workload_meta_matches_model,
    write_csv,
    write_json,
)
from experiments.run_frozen_eval_config import build_eval_invocation

_WORKLOAD_DEFAULTS = {
    "arrival_mode": "poisson",
    "phase1_arrival_layout": "beneficiary_rich",
    "phase2_arrival_layout": "beneficiary_rich",
    "phase1_early_short_frac": 0.25,
    "phase2_early_short_frac": 0.20,
    "phase1_post_long_short_bias": 0.70,
    "phase2_post_long_short_bias": 0.60,
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
