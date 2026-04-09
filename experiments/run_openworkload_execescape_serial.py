from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

from experiments.run_openworkload_execescape_suite import (
    _build_dataset_source_payload,
    _ensure_dir,
    _load_config,
    _purge_experiment_processes,
    _resolve_model_entry,
    _run_single_case,
    _write_csv,
    _write_json,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Serial, per-case open-workload execution-escape runner.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-name", default="")
    args = parser.parse_args()

    config = _load_config(args.config)
    run_name = args.run_name or time.strftime("%Y%m%d_%H%M%S")
    out_root = Path(str(config.get("out_root", "results/openworkload_execescape_tradeoff")))
    run_root = out_root / run_name
    metadata_dir = _ensure_dir(run_root / "metadata")
    _ensure_dir(run_root / "figures")
    _ensure_dir(run_root / "raw")
    _ensure_dir(run_root / "workloads")

    resolved_models = [_resolve_model_entry(entry) for entry in config.get("models", [])]
    dataset_payload = _build_dataset_source_payload(config)
    _write_json(metadata_dir / "resolved_config.json", config)
    _write_json(metadata_dir / "models.json", [asdict(m) for m in resolved_models])
    _write_json(metadata_dir / "datasets.json", config.get("datasets", []))
    _write_json(metadata_dir / "densities.json", config.get("workload", {}).get("densities", []))
    dataset_source_path = metadata_dir / "dataset_sources_resolved.json"
    _write_json(dataset_source_path, dataset_payload)
    _purge_experiment_processes(reason=f"serial-run-start:{run_name}")

    rows: list[dict] = []
    suite_json = metadata_dir / "suite_results.json"
    suite_csv = metadata_dir / "suite_results.csv"
    if suite_json.exists():
        try:
            loaded = json.loads(suite_json.read_text(encoding="utf-8"))
            if isinstance(loaded, list):
                rows = loaded
        except Exception:
            rows = []
    completed_ok = {
        (str(r.get("density")), str(r.get("model_key")))
        for r in rows
        if str(r.get("status")) == "ok"
    }

    for density in config.get("workload", {}).get("densities", []):
        if not isinstance(density, dict):
            continue
        for model in resolved_models:
            if (str(density["name"]), str(model.key)) in completed_ok:
                print(
                    f"[SerialSuite] skip density={density['name']} model={model.label} status=ok",
                    flush=True,
                )
                continue
            print(f"[SerialSuite] start density={density['name']} model={model.label}", flush=True)
            row = _run_single_case(
                model=model,
                density=density,
                config=config,
                dataset_source_path=dataset_source_path,
                run_root=run_root,
            )
            rows.append(row)
            _write_json(suite_json, rows)
            _write_csv(suite_csv, rows)
            if str(row.get("status")) == "ok":
                completed_ok.add((str(density["name"]), str(model.key)))
            print(
                f"[SerialSuite] done density={density['name']} model={model.label} "
                f"status={row.get('status')} "
                f"ttft={row.get('phase12_ttft_improve_mean')} "
                f"wall={row.get('phase12_wall_improve_mean')} "
                f"slow={row.get('phase12_slowdown_improve_mean')}",
                flush=True,
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
