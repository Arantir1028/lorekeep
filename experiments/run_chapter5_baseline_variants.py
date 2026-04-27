from __future__ import annotations

import argparse
import json
import subprocess
import time
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

from experiments.openworkload_models import ResolvedModel, resolve_model_entry
from experiments.openworkload_support import (
    ensure_dir as _ensure_dir,
    load_config as _load_config,
    load_existing_rows as _load_existing_rows,
    write_csv as _write_csv,
    write_json as _write_json,
)
from experiments.run_frozen_eval_config import build_eval_invocation
from experiments.run_openworkload_execescape_suite import _case_eval_config


def _mean(values: list[Optional[float]]) -> Optional[float]:
    data = [float(v) for v in values if v is not None]
    if not data:
        return None
    return float(sum(data) / len(data))


def _percentile(values: list[float], p: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(float(v) for v in values)
    if p <= 0:
        return ordered[0]
    if p >= 100:
        return ordered[-1]
    k = (len(ordered) - 1) * (p / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(ordered) - 1)
    if lo == hi:
        return ordered[lo]
    frac = k - lo
    return ordered[lo] + (ordered[hi] - ordered[lo]) * frac


def _safe_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except Exception:
        return None


def _timing_summary(request_timings: dict[str, Any]) -> dict[str, Optional[float]]:
    if not isinstance(request_timings, dict):
        return {
            "request_count": None,
            "short_request_count": None,
            "long_request_count": None,
            "all_ttft_p99_ms": None,
            "short_ttft_p99_ms": None,
            "long_ttft_p99_ms": None,
            "all_completion_p99_ms": None,
            "short_completion_p99_ms": None,
            "long_completion_p99_ms": None,
        }

    all_ttft: list[float] = []
    short_ttft: list[float] = []
    long_ttft: list[float] = []
    all_finish: list[float] = []
    short_finish: list[float] = []
    long_finish: list[float] = []
    short_count = 0
    long_count = 0

    for item in request_timings.values():
        if not isinstance(item, dict):
            continue
        first = _safe_float(item.get("first_latency_ms"))
        finish = _safe_float(item.get("finish_latency_ms"))
        is_short = bool(item.get("is_short"))
        if first is not None:
            all_ttft.append(first)
            if is_short:
                short_ttft.append(first)
            else:
                long_ttft.append(first)
        if finish is not None:
            all_finish.append(finish)
            if is_short:
                short_finish.append(finish)
            else:
                long_finish.append(finish)
        if is_short:
            short_count += 1
        else:
            long_count += 1

    return {
        "request_count": short_count + long_count,
        "short_request_count": short_count,
        "long_request_count": long_count,
        "all_ttft_p99_ms": _percentile(all_ttft, 99.0),
        "short_ttft_p99_ms": _percentile(short_ttft, 99.0),
        "long_ttft_p99_ms": _percentile(long_ttft, 99.0),
        "all_completion_p99_ms": _percentile(all_finish, 99.0),
        "short_completion_p99_ms": _percentile(short_finish, 99.0),
        "long_completion_p99_ms": _percentile(long_finish, 99.0),
    }


def _aggregate_side_rows(
    rows: list[dict[str, Any]],
    *,
    ttft_key: str,
    slowdown_key: str,
    wall_key: str,
    timing_key: str,
) -> dict[str, Optional[float]]:
    ttft_vals = [_safe_float(row.get(ttft_key)) for row in rows]
    slow_vals = [_safe_float(row.get(slowdown_key)) for row in rows]
    wall_vals = [_safe_float(row.get(wall_key)) for row in rows]
    timing_stats = [_timing_summary(row.get(timing_key) or {}) for row in rows]

    request_counts = [_safe_float(item.get("request_count")) for item in timing_stats]
    short_request_counts = [_safe_float(item.get("short_request_count")) for item in timing_stats]
    long_request_counts = [_safe_float(item.get("long_request_count")) for item in timing_stats]
    throughput_vals = [
        (req_count * 1000.0 / wall_ms)
        for req_count, wall_ms in zip(request_counts, wall_vals)
        if req_count is not None and wall_ms is not None and wall_ms > 0
    ]

    return {
        "request_count_mean": _mean(request_counts),
        "short_request_count_mean": _mean(short_request_counts),
        "long_request_count_mean": _mean(long_request_counts),
        "all_ttft_p99_ms": _mean([item.get("all_ttft_p99_ms") for item in timing_stats]),
        "short_ttft_p99_ms": _mean([item.get("short_ttft_p99_ms") for item in timing_stats]) or _mean(ttft_vals),
        "long_ttft_p99_ms": _mean([item.get("long_ttft_p99_ms") for item in timing_stats]),
        "all_completion_p99_ms": _mean([item.get("all_completion_p99_ms") for item in timing_stats]),
        "short_completion_p99_ms": _mean([item.get("short_completion_p99_ms") for item in timing_stats]),
        "long_completion_p99_ms": _mean([item.get("long_completion_p99_ms") for item in timing_stats]),
        "short_slowdown_p99": _mean(slow_vals),
        "round_wall_ms": _mean(wall_vals),
        "throughput_rps": _mean(throughput_vals),
    }


def _extract_variant_methods(
    *,
    summary_path: Path,
    variant: dict[str, Any],
    density: str,
    model: ResolvedModel,
) -> list[dict[str, Any]]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    per_repeat = dict(summary.get("per_repeat") or {})
    method_labels = dict(variant.get("method_labels") or {})
    method_rows: list[dict[str, Any]] = []

    phase2_rows = list(per_repeat.get("phase2") or [])
    if phase2_rows:
        base_metrics = _aggregate_side_rows(
            phase2_rows,
            ttft_key="base_ttft_short_p99_ms",
            slowdown_key="base_slowdown_short_p99",
            wall_key="base_round_wall_ms",
            timing_key="base_request_timings",
        )
        method_rows.append(
            {
                "variant_key": str(variant.get("key") or ""),
                "method_key": "baseline",
                "method_label": str(method_labels.get("baseline") or "Baseline"),
                "density": density,
                "model_key": model.key,
                "model_label": model.label,
                **base_metrics,
                "result_json": str(summary_path),
            }
        )
        wave_metrics = _aggregate_side_rows(
            phase2_rows,
            ttft_key="wave_ttft_short_p99_ms",
            slowdown_key="wave_slowdown_short_p99",
            wall_key="wave_round_wall_ms",
            timing_key="wave_request_timings",
        )
        method_rows.append(
            {
                "variant_key": str(variant.get("key") or ""),
                "method_key": "wave",
                "method_label": str(method_labels.get("wave") or "Wave"),
                "density": density,
                "model_key": model.key,
                "model_label": model.label,
                **wave_metrics,
                "result_json": str(summary_path),
            }
        )

    strict_rows = list(per_repeat.get("phase2_strict") or [])
    if strict_rows:
        strict_metrics = _aggregate_side_rows(
            strict_rows,
            ttft_key="strict_ttft_short_p99_ms",
            slowdown_key="strict_slowdown_short_p99",
            wall_key="strict_round_wall_ms",
            timing_key="strict_request_timings",
        )
        method_rows.append(
            {
                "variant_key": str(variant.get("key") or ""),
                "method_key": "strict",
                "method_label": str(method_labels.get("strict") or "Strict"),
                "density": density,
                "model_key": model.key,
                "model_label": model.label,
                **strict_metrics,
                "result_json": str(summary_path),
            }
        )
    return method_rows


def _load_source_context(config: dict[str, Any]) -> tuple[Path, dict[str, Any], list[dict[str, Any]], dict[str, ResolvedModel]]:
    source_root = Path(str(config.get("source_run_root") or "")).expanduser()
    if not str(config.get("source_run_root") or "").strip():
        raise ValueError("missing source_run_root; pass --source-run-root or set it in the config")
    if not source_root.exists():
        raise FileNotFoundError(f"source run root not found: {source_root}")
    source_resolved = _load_config(str(source_root / "metadata" / "resolved_config.json"))
    source_rows = _load_existing_rows(source_root / "metadata" / "suite_results.json")
    source_models = {
        resolve_model_entry(item).key: resolve_model_entry(item)
        for item in list(source_resolved.get("models") or [])
    }
    return source_root, source_resolved, source_rows, source_models


def _selected_cases(
    *,
    config: dict[str, Any],
    source_root: Path,
    source_rows: list[dict[str, Any]],
    model_keys_override: str = "",
    densities_override: str = "",
) -> list[dict[str, Any]]:
    selection = dict(config.get("selection") or {})
    required_status = str(selection.get("source_status") or "ok").strip().lower()
    requested_models = {item.strip() for item in model_keys_override.split(",") if item.strip()}
    requested_densities = {item.strip() for item in densities_override.split(",") if item.strip()}
    selected: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    matched_models: set[str] = set()
    matched_densities: set[str] = set()
    for row in source_rows:
        density = str(row.get("density") or "").strip()
        model_key = str(row.get("model_key") or "").strip()
        if not density or not model_key:
            continue
        if requested_models and model_key not in requested_models:
            continue
        if requested_densities and density not in requested_densities:
            continue
        if str(row.get("status") or "").strip().lower() != required_status:
            continue
        req_json = source_root / "workloads" / density / f"{model_key}_requests.json"
        lora_req_json = source_root / "workloads" / density / f"{model_key}_lora_requests.json"
        if not (req_json.exists() and lora_req_json.exists()):
            continue
        case_key = (density, model_key)
        if case_key in seen:
            continue
        seen.add(case_key)
        matched_models.add(model_key)
        matched_densities.add(density)
        selected.append(dict(row))
    missing_models = sorted(requested_models - matched_models)
    missing_densities = sorted(requested_densities - matched_densities)
    if missing_models:
        raise ValueError(f"unknown or unavailable requested model keys: {missing_models}")
    if missing_densities:
        raise ValueError(f"unknown or unavailable requested density names: {missing_densities}")
    return selected


def _merged_phase2(source_resolved: dict[str, Any], variant: dict[str, Any]) -> dict[str, Any]:
    phase2_cfg = deepcopy(dict(source_resolved.get("phase2") or {}))
    overrides = dict(variant.get("phase2") or {})
    if "baseline_enable_chunked_prefill" in overrides:
        phase2_cfg["baseline_enable_chunked_prefill"] = bool(overrides["baseline_enable_chunked_prefill"])
    if "enable_scheduler_cashout" in overrides:
        phase2_cfg["enable_scheduler_cashout"] = bool(overrides["enable_scheduler_cashout"])
    if "enable_execution_escape" in overrides:
        phase2_cfg["enable_execution_escape"] = bool(overrides["enable_execution_escape"])
    if "enable_mixed_prefill_decode" in overrides:
        phase2_cfg["enable_mixed_prefill_decode"] = bool(overrides["enable_mixed_prefill_decode"])
    return phase2_cfg


def _build_variant_case_config(
    *,
    source_root: Path,
    source_resolved: dict[str, Any],
    source_row: dict[str, Any],
    model: ResolvedModel,
    variant: dict[str, Any],
) -> dict[str, Any]:
    density = str(source_row["density"])
    model_key = str(model.key)
    req_json = str(source_root / "workloads" / density / f"{model_key}_requests.json")
    lora_req_json = str(source_root / "workloads" / density / f"{model_key}_lora_requests.json")
    case_config = _case_eval_config(
        model=model,
        model_path=str(source_row.get("model_path") or model.model_id),
        req_json=req_json,
        lora_req_json=lora_req_json,
        adapter_a=str(source_row.get("adapter_a") or ""),
        adapter_b=str(source_row.get("adapter_b") or ""),
        config={
            "phase1": deepcopy(dict(source_resolved.get("phase1") or {})),
            "phase12_soft_gate": deepcopy(dict(source_resolved.get("phase12_soft_gate") or {})),
            "phase2": _merged_phase2(source_resolved, variant),
        },
        eval_cfg=deepcopy(dict(source_resolved.get("eval") or {})),
    )
    case_config["include_phase12"] = bool(variant.get("include_phase12", False))
    case_config["include_strict"] = bool(variant.get("include_strict", False))
    return case_config


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Chapter 5 baseline variants on an existing open-workload suite.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-name", default="")
    parser.add_argument("--source-run-root", default="", help="Override the source main-suite run root.")
    parser.add_argument("--out-root", default="", help="Override the baseline output root.")
    parser.add_argument("--variants", default="", help="Optional comma-separated variant keys to run.")
    parser.add_argument("--model-keys", default="", help="Optional comma-separated model keys to restrict the selected source cases.")
    parser.add_argument("--densities", default="", help="Optional comma-separated density names to restrict the selected source cases.")
    parser.add_argument("--limit-cases", type=int, default=0, help="Optional cap on the number of case executions.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = _load_config(args.config)
    if args.source_run_root:
        config["source_run_root"] = args.source_run_root
    if args.out_root:
        config["out_root"] = args.out_root
    source_root, source_resolved, source_rows, source_models = _load_source_context(config)
    selected_cases = _selected_cases(
        config=config,
        source_root=source_root,
        source_rows=source_rows,
        model_keys_override=args.model_keys,
        densities_override=args.densities,
    )
    if not selected_cases:
        raise RuntimeError("no source cases selected for chapter5 baseline variants")

    requested_variants = {item.strip() for item in args.variants.split(",") if item.strip()}
    variants = [
        dict(item)
        for item in list(config.get("variants") or [])
        if isinstance(item, dict) and bool(item.get("enabled", True))
    ]
    if requested_variants:
        variants = [item for item in variants if str(item.get("key") or "") in requested_variants]
    if not variants:
        raise RuntimeError("no baseline variants selected")

    run_name = args.run_name or time.strftime("%Y%m%d_%H%M%S")
    run_root = _ensure_dir(Path(str(config.get("out_root") or "results/chapter5_baseline_variants")) / run_name)
    metadata_dir = _ensure_dir(run_root / "metadata")
    configs_dir = _ensure_dir(run_root / "configs")
    raw_dir = _ensure_dir(run_root / "raw")
    logs_dir = _ensure_dir(run_root / "logs")

    _write_json(metadata_dir / "suite_config.json", config)
    _write_json(metadata_dir / "selected_cases.json", selected_cases)
    _write_json(metadata_dir / "source_context.json", {
        "source_run_root": str(source_root),
        "source_suite_name": source_resolved.get("suite_name"),
        "selected_case_count": len(selected_cases),
        "variant_keys": [str(item.get("key") or "") for item in variants],
    })

    variant_rows = _load_existing_rows(metadata_dir / "variant_suite_results.json")
    method_rows = _load_existing_rows(metadata_dir / "method_metrics.json")
    done_variant_keys = {
        (
            str(row.get("variant_key") or "").strip(),
            str(row.get("density") or "").strip(),
            str(row.get("model_key") or "").strip(),
        )
        for row in variant_rows
        if str(row.get("status") or "").strip().lower() == "ok"
    }

    execution_count = 0
    for variant in variants:
        variant_key = str(variant.get("key") or "").strip()
        for source_row in selected_cases:
            density = str(source_row.get("density") or "").strip()
            model_key = str(source_row.get("model_key") or "").strip()
            case_key = (variant_key, density, model_key)
            if case_key in done_variant_keys:
                print(
                    f"[Chapter5Baseline] skip variant={variant_key} density={density} model={model_key} reason=already_completed",
                    flush=True,
                )
                continue

            model = source_models.get(model_key)
            if model is None:
                continue

            density_cfg_dir = _ensure_dir(configs_dir / density)
            density_raw_dir = _ensure_dir(raw_dir / density)
            density_log_dir = _ensure_dir(logs_dir / density)
            config_path = density_cfg_dir / f"{model_key}_{variant_key}.json"
            out_json = density_raw_dir / f"{model_key}_{variant_key}.json"
            stdout_path = density_log_dir / f"{model_key}_{variant_key}.stdout.log"
            stderr_path = density_log_dir / f"{model_key}_{variant_key}.stderr.log"

            case_config = _build_variant_case_config(
                source_root=source_root,
                source_resolved=source_resolved,
                source_row=source_row,
                model=model,
                variant=variant,
            )
            case_config["result_json"] = str(out_json)
            _write_json(config_path, case_config)

            row = {
                "variant_key": variant_key,
                "variant_label": str(variant.get("label") or variant_key),
                "density": density,
                "model_key": model.key,
                "model_label": model.label,
                "status": "failed",
                "config_json": str(config_path),
                "result_json": str(out_json),
                "stdout_log": str(stdout_path),
                "stderr_log": str(stderr_path),
            }

            if args.dry_run:
                row["status"] = "dry_run"
                variant_rows = [
                    existing
                    for existing in variant_rows
                    if (
                        str(existing.get("variant_key") or "").strip(),
                        str(existing.get("density") or "").strip(),
                        str(existing.get("model_key") or "").strip(),
                    ) != case_key
                ]
                variant_rows.append(row)
                continue

            cmd, env = build_eval_invocation(
                case_config,
                out_json_override=str(out_json),
            )
            print(
                f"[Chapter5Baseline] start variant={variant_key} density={density} model={model.label}",
                flush=True,
            )
            with stdout_path.open("w", encoding="utf-8") as stdout_f, stderr_path.open("w", encoding="utf-8") as stderr_f:
                completed = subprocess.run(
                    cmd,
                    stdout=stdout_f,
                    stderr=stderr_f,
                    text=True,
                    check=False,
                    env=env,
                )
            row["returncode"] = int(completed.returncode)
            if stdout_path.exists():
                row["stdout_tail"] = "\n".join(stdout_path.read_text(encoding="utf-8", errors="replace").splitlines()[-20:])
            if stderr_path.exists():
                row["stderr_tail"] = "\n".join(stderr_path.read_text(encoding="utf-8", errors="replace").splitlines()[-20:])
            if completed.returncode == 0 and out_json.exists():
                row["status"] = "ok"
                done_variant_keys.add(case_key)
                extracted = _extract_variant_methods(
                    summary_path=out_json,
                    variant=variant,
                    density=density,
                    model=model,
                )
                method_rows = [
                    existing
                    for existing in method_rows
                    if not (
                        str(existing.get("variant_key") or "").strip() == variant_key
                        and str(existing.get("density") or "").strip() == density
                        and str(existing.get("model_key") or "").strip() == model.key
                    )
                ]
                method_rows.extend(extracted)
            else:
                row["error"] = f"evaluate_waveslice_claims exited with code {completed.returncode}"

            variant_rows = [
                existing
                for existing in variant_rows
                if (
                    str(existing.get("variant_key") or "").strip(),
                    str(existing.get("density") or "").strip(),
                    str(existing.get("model_key") or "").strip(),
                ) != case_key
            ]
            variant_rows.append(row)
            _write_json(metadata_dir / "variant_suite_results.json", variant_rows)
            _write_csv(metadata_dir / "variant_suite_results.csv", variant_rows)
            _write_json(metadata_dir / "method_metrics.json", method_rows)
            _write_csv(metadata_dir / "method_metrics.csv", method_rows)
            print(
                f"[Chapter5Baseline] done variant={variant_key} density={density} model={model.label} "
                f"status={row.get('status')}",
                flush=True,
            )
            execution_count += 1
            if args.limit_cases and execution_count >= int(args.limit_cases):
                return 0

    _write_json(metadata_dir / "variant_suite_results.json", variant_rows)
    _write_csv(metadata_dir / "variant_suite_results.csv", variant_rows)
    _write_json(metadata_dir / "method_metrics.json", method_rows)
    _write_csv(metadata_dir / "method_metrics.csv", method_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
