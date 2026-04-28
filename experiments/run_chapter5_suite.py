from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
import time
from pathlib import Path

from experiments.openworkload_support import ensure_dir, load_config, project_path, relative_to_repo, repo_root, write_json


_VALID_STAGES = ("preflight", "main", "baseline", "figures", "partial-figures")


def _resolve_stages(spec: str) -> list[str]:
    raw = [item.strip().lower() for item in spec.split(",") if item.strip()]
    if not raw or raw == ["all"]:
        return list(_VALID_STAGES)
    invalid = sorted({item for item in raw if item not in _VALID_STAGES})
    if invalid:
        raise ValueError(f"unknown stage names: {invalid}")
    return raw


def _repo_path(value: str) -> Path:
    return project_path(value)


def _resolved_output_root(config_path: Path) -> Path:
    config = load_config(str(config_path))
    return _repo_path(str(config.get("out_root") or "")).resolve()


def _default_run_tag() -> str:
    return time.strftime("%Y%m%d_%H%M%S_chapter5")


def _run_command(cmd: list[str], *, dry_run: bool) -> int:
    print("[Chapter5Suite] command:")
    print("  " + shlex.join(cmd))
    if dry_run:
        return 0
    completed = subprocess.run(cmd, check=False, cwd=str(repo_root()))
    return int(completed.returncode)


def _write_manifest(
    out_dir: Path,
    *,
    pipeline_config: Path,
    stages: list[str],
    main_config: Path,
    baseline_config: Path,
    preflight_run_root: Path | None,
    main_run_root: Path | None,
    baseline_run_root: Path | None,
    e5_summary: Path | None,
) -> None:
    ensure_dir(out_dir)
    write_json(
        out_dir / "chapter5_pipeline_manifest.json",
        {
            "pipeline_config": relative_to_repo(pipeline_config),
            "stages": stages,
            "main_config": relative_to_repo(main_config),
            "baseline_config": relative_to_repo(baseline_config),
            "preflight_run_root": relative_to_repo(preflight_run_root) if preflight_run_root is not None else "",
            "main_run_root": relative_to_repo(main_run_root) if main_run_root is not None else "",
            "baseline_run_root": relative_to_repo(baseline_run_root) if baseline_run_root is not None else "",
            "e5_summary": relative_to_repo(e5_summary) if e5_summary is not None else "",
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Chapter 5 experiments stage-by-stage or end-to-end.")
    parser.add_argument(
        "--config",
        default="experiments/configs/chapter5_pipeline_default.json",
        help="Pipeline config pointing to the main config, baseline config, and figure output root.",
    )
    parser.add_argument(
        "--stages",
        default="all",
        help="Comma-separated stages: preflight,main,baseline,figures,partial-figures or all.",
    )
    parser.add_argument("--run-tag", default="", help="Shared tag used to name the main run, baseline run, and export directory.")
    parser.add_argument("--main-run-root", default="", help="Reuse an existing main-suite run root instead of running the main stage.")
    parser.add_argument("--baseline-run-root", default="", help="Reuse an existing baseline-suite run root instead of running the baseline stage.")
    parser.add_argument("--preflight-run-root", default="", help="Reuse an existing preflight run root and its metadata/resolved_config.json.")
    parser.add_argument("--main-config", default="", help="Override the main-suite config path.")
    parser.add_argument("--baseline-config", default="", help="Override the baseline-suite config path.")
    parser.add_argument("--baseline-out-root", default="", help="Override the baseline-suite output root.")
    parser.add_argument("--figures-out-root", default="", help="Override the root directory where Chapter 5 exports are written.")
    parser.add_argument("--export-name", default="", help="Override the export directory name under figures-out-root.")
    parser.add_argument("--e5-summary", default="", help="Optional Chapter 2 E5 summary JSON used by partial-figure regeneration.")
    parser.add_argument("--model-keys", default="", help="Optional comma-separated model keys passed to the main stage.")
    parser.add_argument("--dataset-keys", default="", help="Optional comma-separated dataset keys passed to the main stage.")
    parser.add_argument("--densities", default="", help="Optional comma-separated density names passed to the main and baseline stages.")
    parser.add_argument("--variants", default="", help="Optional comma-separated baseline variant keys.")
    parser.add_argument("--limit-baseline-cases", type=int, default=0, help="Optional cap on baseline case executions.")
    parser.add_argument("--skip-preflight-engine-smoke", action="store_true", help="Run preflight metadata/config resolution without vLLM engine smoke tests.")
    parser.add_argument("--skip-preflight-lut-rebuild", action="store_true", help="Do not rebuild stale/missing hardware-dependent LUTs during preflight.")
    parser.add_argument("--force-preflight-lut-rebuild", action="store_true", help="Force preflight to rebuild hardware-dependent LUTs for selected models.")
    parser.add_argument("--write-rationale", action="store_true", help="Forwarded to the main stage.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    pipeline_config_path = _repo_path(args.config).resolve()
    pipeline_config = load_config(str(pipeline_config_path))
    stages = _resolve_stages(args.stages)

    main_config_path = _repo_path(str(args.main_config or pipeline_config.get("main_config") or "")).resolve()
    baseline_config_path = _repo_path(str(args.baseline_config or pipeline_config.get("baseline_config") or "")).resolve()
    if not main_config_path.exists():
        raise FileNotFoundError(f"main config not found: {main_config_path}")
    if not baseline_config_path.exists():
        raise FileNotFoundError(f"baseline config not found: {baseline_config_path}")

    run_tag = args.run_tag.strip() or _default_run_tag()
    main_run_name = f"{run_tag}_main"
    preflight_run_name = f"{run_tag}_preflight"
    baseline_run_name = f"{run_tag}_baseline"
    figures_out_root = _repo_path(
        str(args.figures_out_root or pipeline_config.get("figures_out_root") or "results/chapter5_exports")
    ).resolve()
    export_name = args.export_name.strip() or run_tag
    export_dir = figures_out_root / export_name

    main_run_root = project_path(args.main_run_root) if args.main_run_root else None
    preflight_run_root = project_path(args.preflight_run_root) if args.preflight_run_root else None
    baseline_run_root = project_path(args.baseline_run_root) if args.baseline_run_root else None
    e5_summary = project_path(args.e5_summary) if args.e5_summary else None
    if e5_summary is None:
        partial_cfg = dict(pipeline_config.get("partial_figures") or {})
        if partial_cfg.get("e5_summary"):
            e5_summary = _repo_path(str(partial_cfg["e5_summary"])).resolve()

    effective_main_config_path = main_config_path
    if "preflight" in stages and preflight_run_root is None:
        main_out_root = _resolved_output_root(main_config_path)
        preflight_run_root = main_out_root / preflight_run_name
        cmd = [
            sys.executable,
            "experiments/run_environment_preflight.py",
            "--config",
            relative_to_repo(main_config_path),
            "--run-name",
            preflight_run_name,
        ]
        if args.model_keys:
            cmd.extend(["--model-keys", args.model_keys])
        if args.dataset_keys:
            cmd.extend(["--dataset-keys", args.dataset_keys])
        if args.densities:
            cmd.extend(["--densities", args.densities])
        if args.skip_preflight_engine_smoke:
            cmd.append("--skip-engine-smoke")
        if args.skip_preflight_lut_rebuild:
            cmd.append("--skip-lut-rebuild")
        if args.force_preflight_lut_rebuild:
            cmd.append("--force-lut-rebuild")
        if args.dry_run:
            cmd.append("--dry-run")
        rc = _run_command(cmd, dry_run=args.dry_run)
        if rc != 0:
            return rc

    if preflight_run_root is not None:
        resolved_config = preflight_run_root / "metadata" / "resolved_config.json"
        if resolved_config.exists() or not args.dry_run:
            effective_main_config_path = resolved_config

    if "main" in stages and main_run_root is None:
        main_out_root = _resolved_output_root(main_config_path)
        main_run_root = main_out_root / main_run_name
        cmd = [
            sys.executable,
            "experiments/run_openworkload_execescape_suite.py",
            "--config",
            relative_to_repo(effective_main_config_path),
            "--run-name",
            main_run_name,
        ]
        if args.model_keys:
            cmd.extend(["--model-keys", args.model_keys])
        if args.dataset_keys:
            cmd.extend(["--dataset-keys", args.dataset_keys])
        if args.densities:
            cmd.extend(["--densities", args.densities])
        if args.write_rationale:
            cmd.append("--write-rationale")
        if args.dry_run:
            cmd.append("--dry-run")
        rc = _run_command(cmd, dry_run=args.dry_run)
        if rc != 0:
            return rc

    if "baseline" in stages and baseline_run_root is None:
        if main_run_root is None:
            raise RuntimeError("baseline stage requires --main-run-root or a preceding main stage")
        baseline_out_root = (
            _repo_path(args.baseline_out_root).resolve()
            if args.baseline_out_root
            else _resolved_output_root(baseline_config_path)
        )
        baseline_run_root = baseline_out_root / baseline_run_name
        cmd = [
            sys.executable,
            "experiments/run_chapter5_baseline_variants.py",
            "--config",
            relative_to_repo(baseline_config_path),
            "--run-name",
            baseline_run_name,
            "--source-run-root",
            relative_to_repo(main_run_root),
        ]
        if args.baseline_out_root:
            cmd.extend(["--out-root", relative_to_repo(baseline_out_root)])
        if args.variants:
            cmd.extend(["--variants", args.variants])
        if args.model_keys:
            cmd.extend(["--model-keys", args.model_keys])
        if args.densities:
            cmd.extend(["--densities", args.densities])
        if args.limit_baseline_cases:
            cmd.extend(["--limit-cases", str(args.limit_baseline_cases)])
        if args.dry_run:
            cmd.append("--dry-run")
        rc = _run_command(cmd, dry_run=args.dry_run)
        if rc != 0:
            return rc

    if "figures" in stages:
        if main_run_root is None:
            raise RuntimeError("figures stage requires --main-run-root or a preceding main stage")
        if baseline_run_root is None:
            raise RuntimeError("figures stage requires --baseline-run-root or a preceding baseline stage")
        cmd = [
            sys.executable,
            "scripts/regenerate_chapter5_main_outputs.py",
            "--main-run",
            relative_to_repo(main_run_root),
            "--baseline-run",
            relative_to_repo(baseline_run_root),
            "--out-dir",
            relative_to_repo(export_dir),
        ]
        rc = _run_command(cmd, dry_run=args.dry_run)
        if rc != 0:
            return rc

    if "partial-figures" in stages:
        if main_run_root is None:
            raise RuntimeError("partial-figures stage requires --main-run-root or a preceding main stage")
        if e5_summary is None:
            print("[Chapter5Suite] skip partial-figures: no E5 summary configured", flush=True)
        else:
            cmd = [
                sys.executable,
                "scripts/regenerate_chapter5_partial_figures.py",
                "--main-run",
                relative_to_repo(main_run_root),
                "--out-dir",
                relative_to_repo(export_dir),
                "--e5-summary",
                relative_to_repo(e5_summary),
            ]
            rc = _run_command(cmd, dry_run=args.dry_run)
            if rc != 0:
                return rc

    if not args.dry_run:
        _write_manifest(
            export_dir,
            pipeline_config=pipeline_config_path,
            stages=stages,
            main_config=main_config_path,
            baseline_config=baseline_config_path,
            preflight_run_root=preflight_run_root,
            main_run_root=main_run_root,
            baseline_run_root=baseline_run_root,
            e5_summary=e5_summary,
        )

    print(f"[Chapter5Suite] preflight_run_root={relative_to_repo(preflight_run_root) if preflight_run_root else None}")
    print(f"[Chapter5Suite] main_run_root={relative_to_repo(main_run_root) if main_run_root else None}")
    print(f"[Chapter5Suite] baseline_run_root={relative_to_repo(baseline_run_root) if baseline_run_root else None}")
    print(f"[Chapter5Suite] export_dir={relative_to_repo(export_dir)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
