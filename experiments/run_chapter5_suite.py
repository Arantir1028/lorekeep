from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from experiments.openworkload_support import (
    ensure_dir,
    load_config,
    project_path,
    relative_to_repo,
    repo_root,
    write_json,
)

STAGES = ("preflight", "main", "baseline")


def _stages(value: str) -> list[str]:
    selected = [item.strip().lower() for item in value.split(",") if item.strip()]
    selected = list(STAGES) if not selected or selected == ["all"] else selected
    invalid = set(selected) - set(STAGES)
    if invalid:
        raise ValueError(f"unknown stage names: {sorted(invalid)}")
    return selected


def _run(command: list[str], dry_run: bool) -> int:
    print("[Chapter5Suite] " + shlex.join(command), flush=True)
    return 0 if dry_run else subprocess.run(command, cwd=repo_root(), check=False).returncode


def _root(config_path: Path) -> Path:
    return project_path(str(load_config(str(config_path)).get("out_root") or "")).resolve()


def _add_filter(command: list[str], name: str, value: Any) -> None:
    if value:
        command.extend([f"--{name}", str(value)])


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the maintained Chapter 5 preflight, main, and baseline stages."
    )
    parser.add_argument("--config", default="experiments/configs/chapter5_pipeline_default.json")
    parser.add_argument("--stages", default="all")
    parser.add_argument("--run-tag", default="")
    parser.add_argument("--preflight-run-root", default="")
    parser.add_argument("--main-run-root", default="")
    parser.add_argument("--baseline-run-root", default="")
    parser.add_argument("--main-config", default="")
    parser.add_argument("--baseline-config", default="")
    parser.add_argument("--baseline-out-root", default="")
    parser.add_argument("--model-keys", default="")
    parser.add_argument("--dataset-keys", default="")
    parser.add_argument("--densities", default="")
    parser.add_argument("--variants", default="")
    parser.add_argument("--limit-baseline-cases", type=int, default=0)
    parser.add_argument("--skip-preflight-engine-smoke", action="store_true")
    parser.add_argument("--skip-preflight-lut-rebuild", action="store_true")
    parser.add_argument("--force-preflight-lut-rebuild", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    pipeline_path = project_path(args.config).resolve()
    pipeline = load_config(str(pipeline_path))
    selected = _stages(args.stages)
    main_config = project_path(args.main_config or pipeline.get("main_config") or "").resolve()
    baseline_config = project_path(
        args.baseline_config or pipeline.get("baseline_config") or ""
    ).resolve()
    if not main_config.exists() or not baseline_config.exists():
        raise FileNotFoundError("Chapter 5 main or baseline config is missing")
    tag = args.run_tag.strip() or time.strftime("%Y%m%d_%H%M%S_chapter5")
    preflight_root = project_path(args.preflight_run_root) if args.preflight_run_root else None
    main_root = project_path(args.main_run_root) if args.main_run_root else None
    baseline_root = project_path(args.baseline_run_root) if args.baseline_run_root else None
    effective_main = main_config
    if "preflight" in selected and preflight_root is None:
        preflight_root = _root(main_config) / f"{tag}_preflight"
        command = [
            sys.executable,
            "experiments/run_environment_preflight.py",
            "--config",
            relative_to_repo(main_config),
            "--run-name",
            f"{tag}_preflight",
        ]
        for name in ("model-keys", "dataset-keys", "densities"):
            _add_filter(command, name, getattr(args, name.replace("-", "_")))
        for enabled, flag in (
            (args.skip_preflight_engine_smoke, "--skip-engine-smoke"),
            (args.skip_preflight_lut_rebuild, "--skip-lut-rebuild"),
            (args.force_preflight_lut_rebuild, "--force-lut-rebuild"),
            (args.dry_run, "--dry-run"),
        ):
            if enabled:
                command.append(flag)
        if code := _run(command, args.dry_run):
            return code
    if preflight_root is not None:
        resolved = preflight_root / "metadata/resolved_config.json"
        if resolved.exists() or not args.dry_run:
            effective_main = resolved
    if "main" in selected and main_root is None:
        main_root = _root(main_config) / f"{tag}_main"
        command = [
            sys.executable,
            "experiments/run_openworkload_suite.py",
            "--config",
            relative_to_repo(effective_main),
            "--run-name",
            f"{tag}_main",
        ]
        for name in ("model-keys", "dataset-keys", "densities"):
            _add_filter(command, name, getattr(args, name.replace("-", "_")))
        if args.dry_run:
            command.append("--dry-run")
        if code := _run(command, args.dry_run):
            return code
    if "baseline" in selected and baseline_root is None:
        if main_root is None:
            raise RuntimeError("baseline requires --main-run-root or the main stage")
        baseline_out = (
            project_path(args.baseline_out_root).resolve()
            if args.baseline_out_root
            else _root(baseline_config)
        )
        baseline_root = baseline_out / f"{tag}_baseline"
        command = [
            sys.executable,
            "experiments/run_chapter5_baseline_variants.py",
            "--config",
            relative_to_repo(baseline_config),
            "--run-name",
            f"{tag}_baseline",
            "--source-run-root",
            relative_to_repo(main_root),
        ]
        for name, value in (
            ("out-root", args.baseline_out_root),
            ("variants", args.variants),
            ("model-keys", args.model_keys),
            ("densities", args.densities),
            ("limit-cases", args.limit_baseline_cases),
        ):
            _add_filter(command, name, value)
        if args.dry_run:
            command.append("--dry-run")
        if code := _run(command, args.dry_run):
            return code
    export_root = (
        project_path(str(pipeline.get("figures_out_root") or "results/chapter5_exports")) / tag
    )
    if not args.dry_run:
        ensure_dir(export_root)
        write_json(
            export_root / "chapter5_pipeline_manifest.json",
            {
                "pipeline_config": relative_to_repo(pipeline_path),
                "stages": selected,
                "main_config": relative_to_repo(main_config),
                "baseline_config": relative_to_repo(baseline_config),
                "preflight_run_root": relative_to_repo(preflight_root) if preflight_root else "",
                "main_run_root": relative_to_repo(main_root) if main_root else "",
                "baseline_run_root": relative_to_repo(baseline_root) if baseline_root else "",
            },
        )
    print(
        f"[Chapter5Suite] preflight_run_root={(relative_to_repo(preflight_root) if preflight_root else None)}"
    )
    print(f"[Chapter5Suite] main_run_root={(relative_to_repo(main_root) if main_root else None)}")
    print(
        f"[Chapter5Suite] baseline_run_root={(relative_to_repo(baseline_root) if baseline_root else None)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
