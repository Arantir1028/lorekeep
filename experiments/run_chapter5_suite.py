from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
import time
from pathlib import Path

from experiments.openworkload_support import ensure_dir, load_config, write_json


_VALID_STAGES = ("main", "baseline", "figures", "partial-figures")


def _resolve_stages(spec: str) -> list[str]:
    raw = [item.strip().lower() for item in spec.split(",") if item.strip()]
    if not raw or raw == ["all"]:
        return list(_VALID_STAGES)
    invalid = sorted({item for item in raw if item not in _VALID_STAGES})
    if invalid:
        raise ValueError(f"unknown stage names: {invalid}")
    return raw


def _repo_path(value: str) -> Path:
    return Path(value).expanduser()


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
    completed = subprocess.run(cmd, check=False)
    return int(completed.returncode)


def _write_manifest(
    out_dir: Path,
    *,
    pipeline_config: Path,
    stages: list[str],
    main_config: Path,
    baseline_config: Path,
    main_run_root: Path | None,
    baseline_run_root: Path | None,
    e5_summary: Path | None,
) -> None:
    ensure_dir(out_dir)
    write_json(
        out_dir / "chapter5_pipeline_manifest.json",
        {
            "pipeline_config": str(pipeline_config),
            "stages": stages,
            "main_config": str(main_config),
            "baseline_config": str(baseline_config),
            "main_run_root": str(main_run_root) if main_run_root is not None else "",
            "baseline_run_root": str(baseline_run_root) if baseline_run_root is not None else "",
            "e5_summary": str(e5_summary) if e5_summary is not None else "",
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
        help="Comma-separated stages: main,baseline,figures,partial-figures or all.",
    )
    parser.add_argument("--run-tag", default="", help="Shared tag used to name the main run, baseline run, and export directory.")
    parser.add_argument("--main-run-root", default="", help="Reuse an existing main-suite run root instead of running the main stage.")
    parser.add_argument("--baseline-run-root", default="", help="Reuse an existing baseline-suite run root instead of running the baseline stage.")
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
    baseline_run_name = f"{run_tag}_baseline"
    figures_out_root = _repo_path(
        str(args.figures_out_root or pipeline_config.get("figures_out_root") or "results/chapter5_exports")
    ).resolve()
    export_name = args.export_name.strip() or run_tag
    export_dir = figures_out_root / export_name

    main_run_root = Path(args.main_run_root).expanduser().resolve() if args.main_run_root else None
    baseline_run_root = Path(args.baseline_run_root).expanduser().resolve() if args.baseline_run_root else None
    e5_summary = Path(args.e5_summary).expanduser().resolve() if args.e5_summary else None
    if e5_summary is None:
        partial_cfg = dict(pipeline_config.get("partial_figures") or {})
        if partial_cfg.get("e5_summary"):
            e5_summary = _repo_path(str(partial_cfg["e5_summary"])).resolve()

    if "main" in stages and main_run_root is None:
        main_out_root = _resolved_output_root(main_config_path)
        main_run_root = main_out_root / main_run_name
        cmd = [
            sys.executable,
            "experiments/run_openworkload_execescape_suite.py",
            "--config",
            str(main_config_path),
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
            str(baseline_config_path),
            "--run-name",
            baseline_run_name,
            "--source-run-root",
            str(main_run_root),
        ]
        if args.baseline_out_root:
            cmd.extend(["--out-root", str(baseline_out_root)])
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
            str(main_run_root),
            "--baseline-run",
            str(baseline_run_root),
            "--out-dir",
            str(export_dir),
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
                str(main_run_root),
                "--out-dir",
                str(export_dir),
                "--e5-summary",
                str(e5_summary),
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
            main_run_root=main_run_root,
            baseline_run_root=baseline_run_root,
            e5_summary=e5_summary,
        )

    print(f"[Chapter5Suite] main_run_root={main_run_root}")
    print(f"[Chapter5Suite] baseline_run_root={baseline_run_root}")
    print(f"[Chapter5Suite] export_dir={export_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
