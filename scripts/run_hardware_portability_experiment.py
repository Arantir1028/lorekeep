from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from experiments.result_io import comma_list, read_json, resolve, write_json

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = (
    ROOT / "results/openworkload_ratio_sweep_lora8/ratio_sweep_20step_5models_a100_overnight_main"
)
DEFAULT_OUT_ROOT = ROOT / "results/hardware_portability_a100_4090_5090"
DEFAULT_MODELS = (
    "baichuan2-7b-chat,gemma-2-9b-it,gemma-7b-it,mistral-7b-instruct-v0.2,qwen2.5-7b-instruct"
)
DEFAULT_DENSITIES = (
    "mid_l10,mid_l30,mid_l50,mid_l70,mid_l90,high_l10,high_l30,high_l50,high_l70,high_l90"
)
MODEL_IDS = {
    "baichuan2-7b-chat": "baichuan-inc/Baichuan2-7B-Chat",
    "gemma-2-9b-it": "google/gemma-2-9b-it",
    "gemma-7b-it": "google/gemma-7b-it",
    "mistral-7b-instruct-v0.2": "mistralai/Mistral-7B-Instruct-v0.2",
    "qwen2.5-7b-instruct": "Qwen/Qwen2.5-7B-Instruct",
}
GPU_PROFILES = {
    "a100": ("NVIDIA A100", "A100", "datacenter HBM reference", "0.7"),
    "rtx4090": ("NVIDIA GeForce RTX 4090", "RTX 4090", "high-end consumer GPU", "0.85"),
    "rtx5090": (
        "NVIDIA GeForce RTX 5090",
        "RTX 5090",
        "new-generation consumer GPU with larger VRAM than RTX 4090",
        "0.9",
    ),
}


def _path(value: str | Path) -> Path:
    return resolve(ROOT, value)


def _case_config(source: Path, density: str, model: str) -> dict[str, Any]:
    path = source / "raw" / density / f"{model}_dataset_eval.json"
    payload = read_json(path)
    if not isinstance(payload, dict) or not isinstance(payload.get("config"), dict):
        raise ValueError(f"missing source config in {path}")
    return dict(payload["config"])


def _preflight_model_paths(args: argparse.Namespace) -> None:
    densities = comma_list(args.densities)
    if not densities:
        raise ValueError("--densities must not be empty")
    missing = []
    for model in comma_list(args.models):
        cfg = _case_config(_path(args.source_main_run), densities[0], model)
        model_path = Path(str(cfg.get("model_path") or ""))
        weights = any(
            model_path.glob(pattern)
            for pattern in ("*.safetensors", "model*.bin", "pytorch_model*.bin", "*.gguf")
        )
        if not model_path.exists():
            missing.append(f"{model}: model_path missing: {model_path}")
        elif not (model_path / "config.json").exists():
            missing.append(f"{model}: config.json missing in {model_path}")
        elif not weights:
            missing.append(f"{model}: no local weight file in {model_path}")
        for key in ("adapter_a", "adapter_b"):
            adapter = Path(str(cfg.get(key) or ""))
            if not adapter.exists():
                missing.append(f"{model}: {key} missing: {adapter}")
            elif not (adapter / "adapter_config.json").exists():
                missing.append(f"{model}: {key} has no adapter_config.json: {adapter}")
    if missing:
        raise FileNotFoundError(
            "hardware portability preflight failed; complete model/adapters before starting GPU runs:\n  - "
            + "\n  - ".join(missing)
        )


def _lut_models(args: argparse.Namespace) -> str:
    if str(args.lut_models).strip().lower() not in {"", "selected"}:
        return str(args.lut_models).strip()
    unknown = [model for model in comma_list(args.models) if model not in MODEL_IDS]
    if unknown:
        raise ValueError(f"unknown model keys for LUT rebuild: {unknown}")
    return ",".join(MODEL_IDS[model] for model in comma_list(args.models))


def _lut_device_list(args: argparse.Namespace) -> str:
    devices = comma_list(args.cuda_visible_devices)
    return str(args.lut_cuda_visible_devices).strip() or (devices[0] if devices else "0")


def _lut_command(args: argparse.Namespace, model: str) -> list[str]:
    lock = (
        str(args.lut_gpu_lock_path).strip() or f"/tmp/waveslice_lut_rebuild_{args.gpu_profile}.lock"
    )
    options = (
        ("models", model),
        ("device", args.lut_device),
        ("dtype", args.lut_dtype),
        ("base-warmup-iters", args.lut_base_warmup_iters),
        ("base-active-iters", args.lut_base_active_iters),
        ("runtime-repeats", args.lut_runtime_repeats),
        ("budget-frac", args.lut_budget_frac),
        ("batch-size-cap", args.lut_batch_size_cap),
        ("gpu-memory-utilization", args.lut_gpu_memory_utilization),
        ("max-num-batched-tokens", args.lut_max_num_batched_tokens),
        ("gpu-lock-path", lock),
    )
    command = [str(args.python_bin), "-m", "experiments.build_hybrid_checkpoint_runtime_luts"]
    for name, value in options:
        command.extend([f"--{name}", str(value)])
    if args.force_lut_rebuild:
        command.append("--force")
    return command


def _run_lut_rebuild(args: argparse.Namespace, out_dir: Path) -> None:
    models = comma_list(_lut_models(args))
    metadata_path = out_dir / "metadata/lut_rebuild.json"
    started = time.time()
    payload: dict[str, Any] = {
        "status": "running",
        "gpu_profile": args.gpu_profile,
        "gpu_label": GPU_PROFILES[args.gpu_profile][0],
        "hardware_label": GPU_PROFILES[args.gpu_profile][1],
        "lut_models": _lut_models(args),
        "cuda_visible_devices": _lut_device_list(args),
        "lut_gpu_memory_utilization": args.lut_gpu_memory_utilization,
        "commands": [],
        "completed_models": [],
        "started_at_unix": started,
        "force_lut_rebuild": bool(args.force_lut_rebuild),
    }
    write_json(metadata_path, payload)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=_lut_device_list(args))
    env.setdefault("VLLM_USE_V1", "1")
    print(f"[HardwarePortability] rebuilding LUTs for gpu_profile={args.gpu_profile}")
    for model in models:
        command, model_started = _lut_command(args, model), time.time()
        print("[HardwarePortability] lut command: " + " ".join(command))
        payload.update(
            current_model=model,
            commands=payload["commands"]
            + [{"model": model, "command": command, "status": "running"}],
        )
        write_json(metadata_path, payload)
        proc = subprocess.run(command, cwd=str(ROOT), env=env, check=False)
        finished = time.time()
        payload["commands"].append(
            {
                "model": model,
                "command": command,
                "status": "ok" if proc.returncode == 0 else "failed",
                "returncode": proc.returncode,
                "started_at_unix": model_started,
                "finished_at_unix": finished,
                "elapsed_sec": finished - model_started,
            }
        )
        if proc.returncode:
            payload.update(
                status="failed",
                finished_at_unix=finished,
                elapsed_sec=finished - started,
                returncode=proc.returncode,
            )
            write_json(metadata_path, payload)
            raise RuntimeError(f"LUT rebuild failed for {model} with return code {proc.returncode}")
        payload["completed_models"].append(model)
    finished = time.time()
    payload.update(
        status="ok",
        current_model=None,
        finished_at_unix=finished,
        elapsed_sec=finished - started,
        returncode=0,
    )
    write_json(metadata_path, payload)


def _metadata(
    args: argparse.Namespace,
    out_dir: Path,
    command: list[str],
    status: str,
    started: float | None = None,
    finished: float | None = None,
    returncode: int | None = None,
) -> dict[str, Any]:
    label, paper_label, role, _ = GPU_PROFILES[args.gpu_profile]
    return {
        "experiment": "hardware_portability_a100_4090_5090",
        "status": status,
        "gpu_profile": args.gpu_profile,
        "gpu_label": label,
        "paper_label": paper_label,
        "hardware_label": paper_label,
        "hardware_role": role,
        "run_location": args.run_location,
        "host_label": args.host_label,
        "cuda_visible_devices": args.cuda_visible_devices,
        "gpu_count": len(comma_list(args.cuda_visible_devices)),
        "methods": ["DistServe-2GPU", "CUCUMIS-2GPU-LB"],
        "method_order": "serial_on_each_gpu_type",
        "cross_gpu_type_schedule": "run a100, rtx4090, and rtx5090 servers concurrently",
        "distserve_backend": "functional_reproduction_continuous_batching_replay",
        "cucumis_replica_execution": "two replicas run concurrently on the two local CUDA devices",
        "source_main_run": str(_path(args.source_main_run)),
        "models": args.models,
        "densities": args.densities,
        "dispatchers": args.dispatchers,
        "output_tokens": args.output_tokens,
        "gpu_memory_utilization_override": args.gpu_memory_utilization_override,
        "gpu_memory_utilization_by_model": args.gpu_memory_utilization_by_model,
        "max_num_batched_tokens_by_model": args.max_num_batched_tokens_by_model,
        "distserve_kv_profile": args.distserve_kv_profile,
        "lut_rebuild": "skipped" if args.skip_lut_rebuild or args.merge_only else "required",
        "lut_models": _lut_models(args),
        "lut_cuda_visible_devices": _lut_device_list(args),
        "model_path_preflight": "skipped" if args.skip_model_path_preflight else "enabled",
        "out_dir": str(out_dir),
        "command": command,
        "started_at_unix": started,
        "finished_at_unix": finished,
        "elapsed_sec": finished - started if started and finished else None,
        "returncode": returncode,
        "local_vs_server_note": "Full GPU execution for this profile is server-side. Local runs should use --dry-run, --limit-* smoke checks, or merge/regeneration over copied artifacts unless the local host has the requested two-GPU profile.",
    }


def _unified_command(args: argparse.Namespace, out_dir: Path) -> list[str]:
    command = [str(args.python_bin), "scripts/run_unified_distserve_cucumis_experiment.py"]
    options = (
        ("source-main-run", _path(args.source_main_run)),
        ("out-dir", out_dir),
        ("models", args.models),
        ("densities", args.densities),
        ("dispatchers", args.dispatchers),
        ("output-tokens", args.output_tokens),
        ("distserve-decode-replay-mode", "continuous_batching"),
        ("cucumis-replica-cuda-devices", args.cuda_visible_devices),
        ("hardware-profile", args.gpu_profile),
        ("hardware-label", GPU_PROFILES[args.gpu_profile][1]),
        ("distserve-kv-profile", args.distserve_kv_profile),
    )
    for name, value in options:
        command.extend([f"--{name}", str(value)])
    command.append("--cucumis-parallel-replicas")
    for flag in ("merge_only", "skip_distserve", "skip_cucumis"):
        if getattr(args, flag):
            command.append("--" + flag.replace("_", "-"))
    for name in (
        "gpu_memory_utilization_override",
        "gpu_memory_utilization_by_model",
        "max_num_batched_tokens_by_model",
        "limit_distserve_cases",
        "limit_cucumis_cases",
    ):
        value = getattr(args, name)
        if value and (not isinstance(value, str) or value.strip()):
            command.extend(["--" + name.replace("_", "-"), str(value)])
    return command


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the A100/4090/5090 hardware portability experiment."
    )
    parser.add_argument("--gpu-profile", choices=sorted(GPU_PROFILES), required=True)
    for name, default in (
        ("run-location", "server"),
        ("host-label", ""),
        ("cuda-visible-devices", "0,1"),
        ("source-main-run", str(DEFAULT_SOURCE)),
        ("out-root", str(DEFAULT_OUT_ROOT)),
        ("models", DEFAULT_MODELS),
        ("densities", DEFAULT_DENSITIES),
        ("dispatchers", "least_backlog"),
        ("distserve-kv-profile", ""),
        ("gpu-memory-utilization-by-model", ""),
        ("max-num-batched-tokens-by-model", ""),
        ("lut-models", "selected"),
        ("lut-cuda-visible-devices", ""),
        ("lut-device", "cuda:0"),
        ("lut-dtype", "fp16"),
        ("lut-gpu-lock-path", ""),
    ):
        kwargs = {"default": default}
        if name == "run-location":
            kwargs["choices"] = ["local", "server"]
        if name == "lut-dtype":
            kwargs["choices"] = ["fp16", "bf16"]
        parser.add_argument("--" + name, **kwargs)
    parser.add_argument("--python-bin", default=sys.executable)
    for name, default in (
        ("output-tokens", 64),
        ("lut-base-warmup-iters", 3),
        ("lut-base-active-iters", 8),
        ("lut-runtime-repeats", 3),
        ("lut-batch-size-cap", 128),
        ("lut-max-num-batched-tokens", 1536),
        ("limit-distserve-cases", 0),
        ("limit-cucumis-cases", 0),
    ):
        parser.add_argument("--" + name, type=int, default=default)
    for name, default in (
        ("gpu-memory-utilization-override", 0.0),
        ("lut-budget-frac", 0.18),
        ("lut-gpu-memory-utilization", 0.0),
    ):
        parser.add_argument("--" + name, type=float, default=default)
    for flag in (
        "skip-lut-rebuild",
        "merge-only",
        "skip-distserve",
        "skip-cucumis",
        "skip-model-path-preflight",
        "dry-run",
    ):
        parser.add_argument("--" + flag, action="store_true")
    parser.add_argument("--no-force-lut-rebuild", dest="force_lut_rebuild", action="store_false")
    parser.set_defaults(force_lut_rebuild=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if not args.merge_only and len(comma_list(args.cuda_visible_devices)) < 2:
        raise ValueError(
            "full hardware portability runs require two CUDA devices, for example --cuda-visible-devices 0,1"
        )
    if args.gpu_memory_utilization_override <= 0:
        args.gpu_memory_utilization_override = float(GPU_PROFILES[args.gpu_profile][3])
    if args.lut_gpu_memory_utilization <= 0:
        args.lut_gpu_memory_utilization = args.gpu_memory_utilization_override
    args.distserve_kv_profile = str(args.distserve_kv_profile).strip() or "realistic_pcie"
    if not args.dry_run and not args.merge_only and not args.skip_model_path_preflight:
        _preflight_model_paths(args)
    out_dir = _path(args.out_root) / args.gpu_profile / "unified_distserve_cucumis_2gpu_lb"
    command, metadata_path = (
        _unified_command(args, out_dir),
        out_dir / "metadata/hardware_profile.json",
    )
    write_json(metadata_path, _metadata(args, out_dir, command, "planned"))
    print(
        f"[HardwarePortability] gpu_profile={args.gpu_profile}\n[HardwarePortability] out_dir={out_dir}\n[HardwarePortability] command: {' '.join(command)}"
    )
    if args.dry_run:
        print("[HardwarePortability] dry_run=true")
        return 0
    if not args.merge_only and not args.skip_lut_rebuild:
        try:
            _run_lut_rebuild(args, out_dir)
        except Exception:
            write_json(metadata_path, _metadata(args, out_dir, command, "failed_lut_rebuild"))
            raise
    started = time.time()
    write_json(metadata_path, _metadata(args, out_dir, command, "running", started))
    proc = subprocess.run(
        command,
        cwd=str(ROOT),
        env=dict(os.environ, CUDA_VISIBLE_DEVICES=args.cuda_visible_devices),
        check=False,
    )
    finished = time.time()
    status = "ok" if proc.returncode == 0 else "failed"
    write_json(
        metadata_path, _metadata(args, out_dir, command, status, started, finished, proc.returncode)
    )
    print(f"[HardwarePortability] status={status} elapsed_sec={finished - started:.1f}")
    return int(proc.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
