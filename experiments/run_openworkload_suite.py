"""Command-line entry point for the open-workload evaluation suite."""

from __future__ import annotations

import argparse
import os
import time
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from typing import Any

os.environ.setdefault("VLLM_NO_USAGE_STATS", "1")

from experiments.openworkload_config import (
    _resolve_selected_datasets,
    _resolve_selected_densities,
    _resolve_selected_models,
)
from experiments.openworkload_execution import _run_cases
from experiments.openworkload_models import ResolvedModel
from experiments.openworkload_results import aggregate_rows, write_result_summary_markdown
from experiments.openworkload_support import (
    build_dataset_source_payload,
    ensure_dir,
    load_config,
    load_existing_rows,
    project_path,
    write_csv,
    write_json,
)

_SUMMARY_METRICS = (
    "phase1_ttft_improve_mean",
    "phase1_wall_improve_mean",
    "phase2_ttft_improve_mean",
    "phase2_wall_improve_mean",
    "phase12_ttft_improve_mean",
    "phase12_wall_improve_mean",
    "phase12_slowdown_improve_mean",
)


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
