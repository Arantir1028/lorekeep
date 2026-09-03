from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments.result_io import read_csv, resolve, write_csv, write_json
from scripts.run_cucumis_2a100_dispatch_sweep import build_equal_resource_comparison

DEFAULT_SOURCE = (
    ROOT / "results/openworkload_ratio_sweep_lora8/ratio_sweep_20step_5models_a100_overnight_main"
)
DEFAULT_DISTSERVE_CONFIG = ROOT / "experiments/configs/distserve_functional_repro_ratio_sweep.json"
DEFAULT_OUT = (
    ROOT
    / "results/openworkload_ratio_sweep_lora8/unified_distserve_continuous_cucumis_2a100_formal"
)
DEFAULT_MODELS = (
    "baichuan2-7b-chat,gemma-2-9b-it,gemma-7b-it,mistral-7b-instruct-v0.2,qwen2.5-7b-instruct"
)
DEFAULT_DENSITIES = (
    "mid_l10,mid_l30,mid_l50,mid_l70,mid_l90,high_l10,high_l30,high_l50,high_l70,high_l90"
)
DEFAULT_DISPATCHERS = "round_robin,least_backlog"


def _resolve(value: str) -> Path:
    return resolve(ROOT, value)


def _run_step(name: str, cmd: list[str], *, progress_path: Path) -> None:
    started = time.time()
    write_json(
        progress_path,
        {"active_step": name, "status": "running", "started_at_unix": started, "command": cmd},
    )
    print(f"[Unified] starting {name}")
    print("[Unified] command: " + " ".join(cmd))
    code = subprocess.run(cmd, cwd=str(ROOT), check=False).returncode
    finished = time.time()
    write_json(
        progress_path,
        {
            "active_step": name,
            "status": "ok" if code == 0 else "failed",
            "returncode": code,
            "started_at_unix": started,
            "finished_at_unix": finished,
            "elapsed_sec": finished - started,
            "command": cmd,
        },
    )
    if code:
        raise RuntimeError(f"{name} failed with return code {code}")
    print(f"[Unified] finished {name} elapsed_sec={finished - started:.1f}")


def _copy_existing_cucumis_outputs(source: Path, target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    for name in (
        "cucumis_2a100_real_method_metrics.csv",
        "cucumis_2a100_real_method_metrics.json",
        "cucumis_2a100_real_request_metrics.csv",
        "cucumis_2a100_real_request_metrics.json",
    ):
        if (path := source / name).exists():
            shutil.copy2(path, target / name)
    for name in ("progress.csv", "progress.json"):
        if (path := source / "metadata" / name).exists():
            (target / "metadata").mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target / "metadata" / name)


def _merge_outputs(out_dir: Path) -> None:
    dist_dir, cuc_dir = out_dir / "distserve_serial", out_dir / "cucumis_2a100"
    dist_methods = read_csv(dist_dir / "distserve_serial_method_metrics.csv")
    cuc_methods = read_csv(cuc_dir / "cucumis_2a100_real_method_metrics.csv")
    dist_requests = read_csv(dist_dir / "distserve_serial_request_metrics.csv")
    cuc_requests = read_csv(cuc_dir / "cucumis_2a100_real_request_metrics.csv")
    comparison = build_equal_resource_comparison(
        dist_dir / "distserve_serial_method_metrics.csv", cuc_methods
    )
    write_csv(out_dir / "unified_method_metrics.csv", [*dist_methods, *cuc_methods])
    write_csv(out_dir / "unified_request_metrics.csv", [*dist_requests, *cuc_requests])
    for path in (
        out_dir / "unified_equal_resource_comparison.csv",
        cuc_dir / "distserve_equal_resource_real_comparison.csv",
    ):
        write_csv(path, comparison)
        write_json(path.with_suffix(".json"), comparison)
    write_json(
        out_dir / "metadata/manifest.json",
        {
            "distserve_serial_dir": str(dist_dir),
            "cucumis_2a100_dir": str(cuc_dir),
            "distserve_method_rows": len(dist_methods),
            "cucumis_method_rows": len(cuc_methods),
            "distserve_decode_replay_formula_versions": sorted(
                {
                    row.get("decode_replay_formula_version", "legacy_unversioned")
                    for row in dist_methods
                }
            ),
            "comparison_rows": len(comparison),
            "method_metrics": str(out_dir / "unified_method_metrics.csv"),
            "request_metrics": str(out_dir / "unified_request_metrics.csv"),
            "equal_resource_comparison": str(out_dir / "unified_equal_resource_comparison.csv"),
        },
    )


def _add(cmd: list[str], name: str, value: Any, condition: bool = True) -> None:
    if condition:
        cmd.extend((f"--{name}", str(value)))


def _distserve_command(args: argparse.Namespace, source: Path, out_dir: Path) -> list[str]:
    cmd = [str(args.python_bin), "scripts/run_distserve_serial_stage_sweep.py"]
    for name, value in (
        ("source-main-run", source),
        ("distserve-config", _resolve(args.distserve_config)),
        ("out-dir", out_dir / "distserve_serial"),
        ("models", args.models),
        ("densities", args.densities),
        ("output-tokens", args.output_tokens),
        ("decode-replay-mode", args.distserve_decode_replay_mode),
        ("decode-batch-size", args.distserve_decode_batch_size),
        ("decode-batch-alpha", args.distserve_decode_batch_alpha),
        ("kv-profile", args.distserve_kv_profile),
    ):
        _add(cmd, name, value)
    optional = (
        (
            "replay-source-dir",
            _resolve(args.distserve_replay_source_dir)
            if args.distserve_replay_source_dir.strip()
            else "",
            bool(args.distserve_replay_source_dir.strip()),
        ),
        (
            "stage-warmup-iters",
            args.distserve_stage_warmup_iters,
            args.distserve_stage_warmup_iters >= 0,
        ),
        ("stage-repeats", args.distserve_stage_repeats, args.distserve_stage_repeats >= 0),
        ("limit-cases", args.limit_distserve_cases, bool(args.limit_distserve_cases)),
        (
            "gpu-memory-utilization-override",
            args.gpu_memory_utilization_override,
            args.gpu_memory_utilization_override > 0,
        ),
        (
            "gpu-memory-utilization-by-model",
            args.gpu_memory_utilization_by_model,
            bool(args.gpu_memory_utilization_by_model.strip()),
        ),
        (
            "max-num-batched-tokens-by-model",
            args.max_num_batched_tokens_by_model,
            bool(args.max_num_batched_tokens_by_model.strip()),
        ),
        ("hardware-profile", args.hardware_profile, bool(args.hardware_profile.strip())),
        ("hardware-label", args.hardware_label, bool(args.hardware_label.strip())),
    )
    for name, value, condition in optional:
        _add(cmd, name, value, condition)
    return cmd


def _cucumis_command(args: argparse.Namespace, source: Path, out_dir: Path) -> list[str]:
    distserve_metrics = out_dir / "distserve_serial/distserve_serial_method_metrics.csv"
    if not distserve_metrics.exists():
        raise FileNotFoundError(f"DistServe metrics not found: {distserve_metrics}")
    cmd = [str(args.python_bin), "scripts/run_cucumis_2a100_dispatch_sweep.py"]
    for name, value in (
        ("source-main-run", source),
        ("distserve-method-metrics", distserve_metrics),
        ("out-dir", out_dir / "cucumis_2a100"),
        ("models", args.models),
        ("densities", args.densities),
        ("dispatchers", args.dispatchers),
        ("output-tokens", args.output_tokens),
    ):
        _add(cmd, name, value)
    for name, value, condition in (
        ("limit-cases", args.limit_cucumis_cases, bool(args.limit_cucumis_cases)),
        (
            "gpu-memory-utilization-override",
            args.gpu_memory_utilization_override,
            args.gpu_memory_utilization_override > 0,
        ),
        (
            "gpu-memory-utilization-by-model",
            args.gpu_memory_utilization_by_model,
            bool(args.gpu_memory_utilization_by_model.strip()),
        ),
        (
            "max-num-batched-tokens-by-model",
            args.max_num_batched_tokens_by_model,
            bool(args.max_num_batched_tokens_by_model.strip()),
        ),
        ("hardware-profile", args.hardware_profile, bool(args.hardware_profile.strip())),
        ("hardware-label", args.hardware_label, bool(args.hardware_label.strip())),
        (
            "replica-cuda-devices",
            args.cucumis_replica_cuda_devices,
            bool(args.cucumis_replica_cuda_devices.strip()),
        ),
    ):
        _add(cmd, name, value, condition)
    if args.cucumis_parallel_replicas:
        cmd.append("--parallel-replicas")
    return cmd


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the unified DistServe P/D + CUCUMIS-2A100 experiment."
    )
    defaults = {
        "source-main-run": str(DEFAULT_SOURCE),
        "distserve-config": str(DEFAULT_DISTSERVE_CONFIG),
        "out-dir": str(DEFAULT_OUT),
        "python-bin": sys.executable,
        "models": DEFAULT_MODELS,
        "densities": DEFAULT_DENSITIES,
        "dispatchers": DEFAULT_DISPATCHERS,
        "distserve-decode-replay-mode": "continuous_batching",
        "distserve-kv-profile": "realistic_pcie",
        "distserve-replay-source-dir": "",
        "gpu-memory-utilization-by-model": "",
        "max-num-batched-tokens-by-model": "",
        "hardware-profile": "",
        "hardware-label": "",
        "cucumis-source-dir": "",
        "cucumis-replica-cuda-devices": "",
    }
    for name, default in defaults.items():
        parser.add_argument(f"--{name}", default=default)
    for name, default in (
        ("output-tokens", 64),
        ("distserve-decode-batch-size", 16),
        ("distserve-stage-warmup-iters", -1),
        ("distserve-stage-repeats", -1),
        ("limit-distserve-cases", 0),
        ("limit-cucumis-cases", 0),
    ):
        parser.add_argument(f"--{name}", type=int, default=default)
    parser.add_argument("--distserve-decode-batch-alpha", type=float, default=0.08)
    parser.add_argument("--gpu-memory-utilization-override", type=float, default=0.0)
    for name in ("merge-only", "skip-distserve", "skip-cucumis", "cucumis-parallel-replicas"):
        parser.add_argument(f"--{name}", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    source, out_dir = _resolve(args.source_main_run), _resolve(args.out_dir)
    progress = out_dir / "metadata/active_step.json"
    out_dir.mkdir(parents=True, exist_ok=True)
    if not args.merge_only and not args.skip_distserve:
        _run_step(
            "distserve_serial_stage_sweep",
            _distserve_command(args, source, out_dir),
            progress_path=progress,
        )
    if not args.merge_only and not args.skip_cucumis:
        _run_step(
            "cucumis_2a100_dispatch_sweep",
            _cucumis_command(args, source, out_dir),
            progress_path=progress,
        )
    elif args.cucumis_source_dir.strip():
        _copy_existing_cucumis_outputs(_resolve(args.cucumis_source_dir), out_dir / "cucumis_2a100")
    _merge_outputs(out_dir)
    write_json(progress, {"active_step": "complete", "status": "ok", "out_dir": str(out_dir)})
    print(f"[Unified] out_dir={out_dir}")
    print(f"[Unified] method_metrics={out_dir / 'unified_method_metrics.csv'}")
    print(f"[Unified] comparison={out_dir / 'unified_equal_resource_comparison.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
