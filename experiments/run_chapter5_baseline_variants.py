from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

os.environ.setdefault("VLLM_NO_USAGE_STATS", "1")

from experiments.openworkload_config import _case_eval_config
from experiments.openworkload_models import ResolvedModel, resolve_model_entry
from experiments.openworkload_support import (
    ensure_dir,
    load_config,
    load_existing_rows,
    project_path,
    repo_root,
    write_csv,
    write_json,
)
from experiments.result_io import mean_values, safe_float, timing_summary
from experiments.run_frozen_eval_config import build_eval_invocation


def _aggregate_side_rows(
    rows: list[dict[str, Any]], *, ttft_key: str, slowdown_key: str, wall_key: str, timing_key: str
) -> dict[str, float | None]:
    timings = [timing_summary(row.get(timing_key) or {}) for row in rows]
    walls = [safe_float(row.get(wall_key)) for row in rows]
    counts = {
        name: [safe_float(item.get(name)) for item in timings]
        for name in ("request_count", "short_request_count", "long_request_count")
    }
    throughput = [
        count * 1000 / wall
        for count, wall in zip(counts["request_count"], walls, strict=False)
        if count is not None and wall is not None and wall > 0
    ]
    output = {f"{name}_mean": mean_values(values) for name, values in counts.items()}
    for metric in ("ttft", "completion"):
        for percent in ("p50", "p90", "p99"):
            for scope in ("all", "short", "long"):
                key = f"{scope}_{metric}_{percent}_ms"
                output[key] = mean_values([item.get(key) for item in timings])
    output["short_ttft_p99_ms"] = output["short_ttft_p99_ms"] or mean_values(
        [safe_float(row.get(ttft_key)) for row in rows]
    )
    output.update(
        short_slowdown_p99=mean_values([safe_float(row.get(slowdown_key)) for row in rows]),
        round_wall_ms=mean_values(walls),
        throughput_rps=mean_values(throughput),
    )
    return output


def _extract_variant_methods(
    *, summary_path: Path, variant: dict[str, Any], density: str, model: ResolvedModel
) -> list[dict[str, Any]]:
    repeats = dict(json.loads(summary_path.read_text(encoding="utf-8")).get("per_repeat") or {})
    labels = dict(variant.get("method_labels") or {})
    specs = (
        ("phase2", "baseline", "base", "Baseline"),
        ("phase2", "wave", "wave", "Wave"),
    )
    output = []
    for repeat_key, method_key, prefix, default_label in specs:
        rows = list(repeats.get(repeat_key) or [])
        if not rows:
            continue
        metrics = _aggregate_side_rows(
            rows,
            ttft_key=f"{prefix}_ttft_short_p99_ms",
            slowdown_key=f"{prefix}_slowdown_short_p99",
            wall_key=f"{prefix}_round_wall_ms",
            timing_key=f"{prefix}_request_timings",
        )
        output.append(
            {
                "variant_key": str(variant.get("key") or ""),
                "method_key": method_key,
                "method_label": str(labels.get(method_key) or default_label),
                "density": density,
                "model_key": model.key,
                "model_label": model.label,
                **metrics,
                "result_json": str(summary_path),
            }
        )
    return output


def _load_source_context(
    config: dict[str, Any],
) -> tuple[Path, dict[str, Any], list[dict[str, Any]], dict[str, ResolvedModel]]:
    value = str(config.get("source_run_root") or "").strip()
    if not value:
        raise ValueError("missing source_run_root; pass --source-run-root or set it in the config")
    source = project_path(value)
    if not source.exists():
        raise FileNotFoundError(f"source run root not found: {source}")
    resolved = load_config(str(source / "metadata/resolved_config.json"))
    rows = load_existing_rows(source / "metadata/suite_results.json")
    models = [resolve_model_entry(item) for item in resolved.get("models") or []]
    return source, resolved, rows, {model.key: model for model in models}


def _selected_cases(
    *,
    config: dict[str, Any],
    source_root: Path,
    source_rows: list[dict[str, Any]],
    model_keys_override: str = "",
    densities_override: str = "",
) -> list[dict[str, Any]]:
    status = str((config.get("selection") or {}).get("source_status") or "ok").strip().lower()
    requested_models = {item.strip() for item in model_keys_override.split(",") if item.strip()}
    requested_densities = {item.strip() for item in densities_override.split(",") if item.strip()}
    selected, seen, matched_models, matched_densities = [], set(), set(), set()
    for row in source_rows:
        density, model = (
            str(row.get("density") or "").strip(),
            str(row.get("model_key") or "").strip(),
        )
        key = density, model
        if (
            not all(key)
            or key in seen
            or requested_models
            and model not in requested_models
            or requested_densities
            and density not in requested_densities
        ):
            continue
        if str(row.get("status") or "").strip().lower() != status:
            continue
        if not all(
            (source_root / "workloads" / density / f"{model}_{suffix}.json").exists()
            for suffix in ("requests", "lora_requests")
        ):
            continue
        selected.append(dict(row))
        seen.add(key)
        matched_models.add(model)
        matched_densities.add(density)
    if missing := sorted(requested_models - matched_models):
        raise ValueError(f"unknown or unavailable requested model keys: {missing}")
    if missing := sorted(requested_densities - matched_densities):
        raise ValueError(f"unknown or unavailable requested density names: {missing}")
    return selected


def _merged_phase2(source_resolved: dict[str, Any], variant: dict[str, Any]) -> dict[str, Any]:
    phase2 = deepcopy(dict(source_resolved.get("phase2") or {}))
    overrides = dict(variant.get("phase2") or {})
    phase2.update(
        {
            key: bool(overrides[key])
            for key in ("baseline_enable_chunked_prefill", "enable_scheduler_cashout")
            if key in overrides
        }
    )
    return phase2


def _build_variant_case_config(
    *,
    source_root: Path,
    source_resolved: dict[str, Any],
    source_row: dict[str, Any],
    model: ResolvedModel,
    variant: dict[str, Any],
) -> dict[str, Any]:
    density = str(source_row["density"])
    prefix = source_root / "workloads" / density / model.key
    config = _case_eval_config(
        model=model,
        model_path=str(source_row.get("model_path") or model.model_id),
        req_json=f"{prefix}_requests.json",
        lora_req_json=f"{prefix}_lora_requests.json",
        adapter_a=str(source_row.get("adapter_a") or ""),
        adapter_b=str(source_row.get("adapter_b") or ""),
        config={
            "phase1": deepcopy(dict(source_resolved.get("phase1") or {})),
            "phase12_soft_gate": deepcopy(dict(source_resolved.get("phase12_soft_gate") or {})),
            "phase2": _merged_phase2(source_resolved, variant),
        },
        eval_cfg=deepcopy(dict(source_resolved.get("eval") or {})),
    )
    config.update(include_phase12=bool(variant.get("include_phase12", False)))
    return config


def _save(
    metadata: Path, variant_rows: list[dict[str, Any]], method_rows: list[dict[str, Any]]
) -> None:
    for name, rows in (("variant_suite_results", variant_rows), ("method_metrics", method_rows)):
        write_json(metadata / f"{name}.json", rows)
        write_csv(metadata / f"{name}.csv", rows)


def _replace(
    rows: list[dict[str, Any]], key: tuple[str, str, str], row: dict[str, Any]
) -> list[dict[str, Any]]:
    return [
        item
        for item in rows
        if tuple(
            str(item.get(name) or "").strip() for name in ("variant_key", "density", "model_key")
        )
        != key
    ] + [row]


def _run_variant(
    *,
    variant: dict[str, Any],
    source_row: dict[str, Any],
    model: ResolvedModel,
    source_root: Path,
    source_resolved: dict[str, Any],
    directories: dict[str, Path],
    dry_run: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    variant_key, density = (
        str(variant.get("key") or "").strip(),
        str(source_row.get("density") or "").strip(),
    )
    stem = f"{model.key}_{variant_key}"
    paths = {
        "config_json": ensure_dir(directories["configs"] / density) / f"{stem}.json",
        "result_json": ensure_dir(directories["raw"] / density) / f"{stem}.json",
        "stdout_log": ensure_dir(directories["logs"] / density) / f"{stem}.stdout.log",
        "stderr_log": ensure_dir(directories["logs"] / density) / f"{stem}.stderr.log",
    }
    case_config = _build_variant_case_config(
        source_root=source_root,
        source_resolved=source_resolved,
        source_row=source_row,
        model=model,
        variant=variant,
    )
    case_config["result_json"] = str(paths["result_json"])
    write_json(paths["config_json"], case_config)
    row = {
        "variant_key": variant_key,
        "variant_label": str(variant.get("label") or variant_key),
        "density": density,
        "model_key": model.key,
        "model_label": model.label,
        "status": "dry_run" if dry_run else "failed",
        **{key: str(value) for key, value in paths.items()},
    }
    if dry_run:
        return row, []
    cmd, env = build_eval_invocation(case_config, out_json_override=str(paths["result_json"]))
    print(
        f"[Chapter5Baseline] start variant={variant_key} density={density} model={model.label}",
        flush=True,
    )
    with (
        paths["stdout_log"].open("w", encoding="utf-8") as stdout,
        paths["stderr_log"].open("w", encoding="utf-8") as stderr,
    ):
        code = subprocess.run(
            cmd, stdout=stdout, stderr=stderr, text=True, check=False, env=env, cwd=str(repo_root())
        ).returncode
    row["returncode"] = int(code)
    if code == 0 and paths["result_json"].exists():
        row["status"] = "ok"
        methods = _extract_variant_methods(
            summary_path=paths["result_json"], variant=variant, density=density, model=model
        )
    else:
        row["error"] = f"evaluate_waveslice_claims exited with code {code}"
        methods = []
    return row, methods


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Chapter 5 baseline variants on an existing open-workload suite."
    )
    parser.add_argument("--config", required=True)
    for name in ("run-name", "source-run-root", "out-root", "variants", "model-keys", "densities"):
        parser.add_argument(f"--{name}", default="")
    parser.add_argument("--limit-cases", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    config = load_config(args.config)
    for key, value in (("source_run_root", args.source_run_root), ("out_root", args.out_root)):
        if value:
            config[key] = value
    source, resolved, source_rows, models = _load_source_context(config)
    cases = _selected_cases(
        config=config,
        source_root=source,
        source_rows=source_rows,
        model_keys_override=args.model_keys,
        densities_override=args.densities,
    )
    requested = {item.strip() for item in args.variants.split(",") if item.strip()}
    variants = [
        dict(item)
        for item in config.get("variants") or []
        if isinstance(item, dict)
        and item.get("enabled", True)
        and (not requested or str(item.get("key") or "") in requested)
    ]
    if not cases or not variants:
        raise RuntimeError(f"no {'source cases' if not cases else 'baseline variants'} selected")
    run_root = ensure_dir(
        project_path(str(config.get("out_root") or "results/chapter5_baseline_variants"))
        / (args.run_name or time.strftime("%Y%m%d_%H%M%S"))
    )
    directories = {
        name: ensure_dir(run_root / name) for name in ("metadata", "configs", "raw", "logs")
    }
    metadata = directories["metadata"]
    write_json(metadata / "suite_config.json", config)
    write_json(metadata / "selected_cases.json", cases)
    write_json(
        metadata / "source_context.json",
        {
            "source_run_root": str(source),
            "source_suite_name": resolved.get("suite_name"),
            "selected_case_count": len(cases),
            "variant_keys": [str(item.get("key") or "") for item in variants],
        },
    )
    variant_rows = load_existing_rows(metadata / "variant_suite_results.json")
    method_rows = load_existing_rows(metadata / "method_metrics.json")
    done = {
        tuple(str(row.get(name) or "").strip() for name in ("variant_key", "density", "model_key"))
        for row in variant_rows
        if row.get("status") == "ok"
    }
    executions = 0
    for variant in variants:
        for source_row in cases:
            key = (
                str(variant.get("key") or "").strip(),
                str(source_row.get("density") or "").strip(),
                str(source_row.get("model_key") or "").strip(),
            )
            if key in done or (model := models.get(key[2])) is None:
                continue
            row, methods = _run_variant(
                variant=variant,
                source_row=source_row,
                model=model,
                source_root=source,
                source_resolved=resolved,
                directories=directories,
                dry_run=args.dry_run,
            )
            variant_rows = _replace(variant_rows, key, row)
            if methods:
                method_rows = [
                    item
                    for item in method_rows
                    if tuple(
                        str(item.get(name) or "").strip()
                        for name in ("variant_key", "density", "model_key")
                    )
                    != key
                ] + methods
                done.add(key)
            _save(metadata, variant_rows, method_rows)
            print(
                f"[Chapter5Baseline] done variant={key[0]} density={key[1]} model={model.label} status={row['status']}",
                flush=True,
            )
            executions += 1
            if args.limit_cases and executions >= args.limit_cases:
                return 0
    _save(metadata, variant_rows, method_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
