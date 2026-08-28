from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from typing import Any

from experiments.model_assets import ensure_model_available
from experiments.openworkload_models import ResolvedModel, runtime_lut_is_valid
from experiments.openworkload_support import (
    apply_hf_resource_env,
    load_config,
    project_path,
    relative_to_repo,
    repo_root,
    resource_policy,
    write_json,
)
from experiments.run_openworkload_suite import (
    _resolve_selected_datasets,
    _resolve_selected_densities,
    _resolve_selected_models,
)
from waveslice.lut.config import BUCKETS


def _detect_environment() -> dict[str, Any]:
    import torch

    gpus = []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            gpus.append(
                {
                    "index": index,
                    "name": props.name,
                    "total_memory_bytes": int(props.total_memory),
                    "total_memory_gb": round(props.total_memory / 1024**3, 2),
                    "capability": list(torch.cuda.get_device_capability(index)),
                }
            )

    def version(name: str) -> str:
        return str(getattr(__import__(name), "__version__", "") or "")

    return {
        "python_executable": sys.executable,
        "python_version": sys.version.split()[0],
        "torch_version": version("torch"),
        "torch_cuda": str(torch.version.cuda or ""),
        "cuda_available": bool(torch.cuda.is_available()),
        "vllm_version": version("vllm"),
        "transformers_version": version("transformers"),
        "datasets_version": version("datasets"),
        "gpus": gpus,
    }


def _gpu_memory_gb(environment: dict[str, Any]) -> float:
    values = [
        float(gpu.get("total_memory_gb") or 0)
        for gpu in environment.get("gpus", [])
        if isinstance(gpu, dict)
    ]
    return min((value for value in values if value > 0), default=0.0)


def _runtime_config(evaluation: dict[str, Any], memory_gb: float) -> dict[str, Any]:
    length = int(evaluation.get("max_model_len", 3072))
    batch = int(evaluation.get("max_num_batched_tokens", 1536))
    memory = float(evaluation.get("gpu_memory_utilization", 0.6))
    source = "config_no_gpu_detected" if memory_gb <= 0 else "memory_heuristic"
    if 0 < memory_gb < 24:
        length, batch, memory = min(length, 2048), min(batch, 1024), max(memory, 0.75)
    elif memory_gb >= 24:
        memory = max(memory, 0.7)
    return {
        "max_model_len": length,
        "max_num_batched_tokens": min(batch, length),
        "gpu_memory_utilization": round(memory, 3),
        "source": source,
    }


def _vllm_env(config: dict[str, Any]) -> dict[str, str]:
    env = apply_hf_resource_env(os.environ.copy(), config)
    env["VLLM_USE_V1"] = "1"
    return env


def _engine_smoke_child(args: argparse.Namespace) -> int:
    from waveslice.vllm.bootstrap import bootstrap_vllm_runtime

    bootstrap_vllm_runtime()
    from vllm.engine.arg_utils import EngineArgs
    from vllm.engine.llm_engine import LLMEngine
    from vllm.sampling_params import SamplingParams

    started = time.perf_counter()
    try:
        engine = LLMEngine.from_engine_args(
            EngineArgs(
                model=args.model_path,
                trust_remote_code=args.trust_remote_code,
                seed=0,
                enable_lora=False,
                max_num_batched_tokens=args.max_num_batched_tokens,
                max_num_partial_prefills=1,
                max_long_partial_prefills=1,
                enable_chunked_prefill=True,
                disable_sliding_window=True,
                enforce_eager=True,
                max_model_len=args.max_model_len,
                gpu_memory_utilization=args.gpu_memory_utilization,
            )
        )
        engine.add_request(
            "preflight",
            "Briefly define online serving.",
            SamplingParams(max_tokens=1, temperature=0),
        )
        deadline = time.time() + args.timeout_sec
        while time.time() < deadline and engine.has_unfinished_requests():
            engine.step()
        status = "failed" if engine.has_unfinished_requests() else "ok"
        payload = {
            "status": status,
            "reason": "timeout" if status == "failed" else "",
            "elapsed_s": time.perf_counter() - started,
        }
    except Exception as exc:
        payload = {
            "status": "failed",
            "reason": "oom" if "out of memory" in str(exc).lower() else "error",
            "error": f"{type(exc).__name__}: {exc}",
            "elapsed_s": time.perf_counter() - started,
        }
    print(json.dumps(payload, ensure_ascii=False), flush=True)
    return 0 if payload["status"] == "ok" else 1


def _resolve_model_path(model: ResolvedModel, config: dict[str, Any], download: bool) -> str:
    policy = resource_policy(config)
    snapshot = ensure_model_available(
        model.model_id,
        auto_download=bool(download and policy["auto_download"]),
        local_files_only=bool(policy["offline"]),
    )
    if model.model_path_mode == "model_id":
        return model.model_id
    if model.model_path_mode == "local_snapshot_required" and not snapshot:
        raise FileNotFoundError(f"local snapshot required for {model.model_id}")
    return snapshot or model.model_id


def _run_engine_smoke(
    model: ResolvedModel,
    model_path: str,
    runtime: dict[str, Any],
    timeout: int,
    config: dict[str, Any],
) -> dict[str, Any]:
    command = [
        sys.executable,
        __file__,
        "--engine-smoke-child",
        "--model-path",
        model_path,
        "--max-model-len",
        str(
            min(runtime["max_model_len"], model.max_model_len_override or runtime["max_model_len"])
        ),
        "--max-num-batched-tokens",
        str(runtime["max_num_batched_tokens"]),
        "--gpu-memory-utilization",
        str(runtime["gpu_memory_utilization"]),
        "--timeout-sec",
        str(timeout),
    ]
    if model.trust_remote_code:
        command.append("--trust-remote-code")
    proc = subprocess.run(
        command, cwd=repo_root(), env=_vllm_env(config), capture_output=True, text=True, check=False
    )
    line = next((line for line in reversed(proc.stdout.splitlines()) if line.startswith("{")), "")
    return (
        json.loads(line)
        if line
        else {"status": "failed", "reason": "child_failed", "stderr_tail": proc.stderr[-2000:]}
    )


def _candidate_models(config: dict[str, Any], keys: str) -> list[ResolvedModel]:
    candidate = deepcopy(config)
    candidate["resource_selection"] = dict(candidate.get("resource_selection") or {}) | {
        "require_runtime_sanity": False
    }
    return _resolve_selected_models(candidate, keys)[0]


def _ensure_luts(
    models: list[ResolvedModel],
    config: dict[str, Any],
    runtime: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    stale = [
        model
        for model in models
        if args.force_lut_rebuild or not runtime_lut_is_valid(model.lut_name)[0]
    ]
    report: dict[str, Any] = {
        "candidate_model_count": len(models),
        "rebuild_needed_count": len(stale),
        "ran": False,
    }
    if not stale or args.skip_lut_rebuild or args.dry_run:
        report["reason"] = "current" if not stale else "skipped"
        return report
    for model in stale:
        _resolve_model_path(model, config, True)
    command = [
        sys.executable,
        "experiments/build_hybrid_checkpoint_runtime_luts.py",
        "--models",
        ",".join(model.model_id for model in stale),
        "--max-num-batched-tokens",
        str(runtime["max_num_batched_tokens"]),
        "--gpu-memory-utilization",
        str(runtime["gpu_memory_utilization"]),
        "--force",
    ]
    subprocess.run(command, cwd=repo_root(), env=_vllm_env(config), check=True)
    report.update(ran=True, command=command, returncode=0)
    return report


def _align_bucket(value: int) -> int:
    ordered = sorted(int(item) for item in BUCKETS if int(item) > 0)
    return next((bucket for bucket in ordered if bucket >= value), ordered[-1])


def _derive_policy_overrides(config: dict[str, Any], runtime: dict[str, Any]) -> dict[str, Any]:
    length, batch = int(runtime["max_model_len"]), int(runtime["max_num_batched_tokens"])
    aggressive = _align_bucket(max(128, min(batch, int(length * 0.25))))
    conservative = _align_bucket(max(aggressive, min(batch, int(length * 0.5))))
    waiting, queue, long = (
        max(2, min(4, batch // aggressive)),
        0,
        max(conservative, int(length * 0.75)),
    )
    queue = max(4, waiting * 2)
    phase1 = dict(config.get("phase1") or {}) | {
        "ingress_target_chunk": aggressive,
        "force_min_chunk": min(
            int((config.get("phase1") or {}).get("force_min_chunk", 128)), aggressive
        ),
    }
    phase2 = dict(config.get("phase2") or {}) | {"min_long_prefill": aggressive}
    adaptive = dict(config.get("adaptive_density_policy") or {})
    adaptive.update(
        runtime_aggressive_ingress_target_chunk=aggressive,
        runtime_conservative_ingress_target_chunk=conservative,
        runtime_queue_high_watermark=queue,
        runtime_waiting_short_high_watermark=waiting,
        runtime_long_high_watermark=long,
        runtime_high_pressure_min_long_prefill=aggressive,
        runtime_low_pressure_min_long_prefill=conservative,
    )
    return {
        "phase1": phase1,
        "phase2": phase2,
        "adaptive_density_policy": adaptive,
        "derived": {
            "aggressive_chunk": aggressive,
            "conservative_chunk": conservative,
            "queue_high_watermark": queue,
            "waiting_short_high_watermark": waiting,
            "long_high_watermark": long,
        },
    }


def _memory_workload_scale(memory_gb: float) -> float:
    return 1.0 if memory_gb >= 24 else 0.75 if memory_gb >= 18 else 0.5 if memory_gb >= 12 else 0.35


def _scale_count(value: Any, scale: float, minimum: int) -> int:
    return max(minimum, round(int(value) * scale))


def _derive_workload_overrides(
    *, config: dict[str, Any], runtime_cfg: dict[str, Any], memory_gb: float
) -> tuple[dict[str, Any], dict[str, Any]]:
    evaluation, workload = dict(config.get("eval") or {}), dict(config.get("workload") or {})
    base_batch = max(1, int(evaluation.get("max_num_batched_tokens", 1536)))
    batch = max(1, int(runtime_cfg.get("max_num_batched_tokens", base_batch)))
    scale = max(0.25, min(1.0, batch / base_batch, _memory_workload_scale(memory_gb)))
    base_new = int(evaluation.get("max_new_tokens", 64))
    new = (
        base_new
        if memory_gb >= 24 and scale >= 0.95
        else min(base_new, 48 if memory_gb >= 18 and scale >= 0.7 else 32)
    )
    new = max(16, next((value for value in (16, 32, 48, 64, 96, 128) if value >= new), new))
    repeats = (
        min(int(evaluation.get("repeats", 2)), 1)
        if scale < 0.5
        else int(evaluation.get("repeats", 2))
    )
    samples = _scale_count(workload.get("sample_count", 256), scale, 64)
    preserve = bool(workload.get("preserve_density_request_counts", False))
    overrides = {
        "eval": {
            "max_new_tokens": new,
            "repeats": repeats,
            "warmup_iters": int(evaluation.get("warmup_iters", 1)),
        },
        "workload": {"sample_count": samples},
    }
    meta = {
        "scale": scale,
        "memory_scale": _memory_workload_scale(memory_gb),
        "batch_scale": min(1.0, batch / base_batch),
        "memory_gb": memory_gb,
        "preserve_density_request_counts": preserve,
        "base_max_num_batched_tokens": base_batch,
        "resolved_max_num_batched_tokens": batch,
        "base_max_new_tokens": base_new,
        "resolved_max_new_tokens": new,
        "base_repeats": int(evaluation.get("repeats", 2)),
        "resolved_repeats": repeats,
        "base_sample_count": int(workload.get("sample_count", 256)),
        "resolved_sample_count": samples,
    }
    return overrides, meta


def _derive_densities(
    densities: list[dict[str, Any]],
    runtime: dict[str, Any],
    evaluation: dict[str, Any],
    meta: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    base = max(1, int(evaluation.get("max_num_batched_tokens", 1536)))
    batch = max(1, int(runtime.get("max_num_batched_tokens", base)))
    rate_scale, count_scale = (
        max(0.35, min(1.5, batch / base)),
        max(0.25, min(1.0, float(meta.get("scale") or 1))),
    )
    resolved, dropped = [], []
    for item in densities:
        if item.get("name") == "peak" and (
            float(meta.get("memory_gb") or 0) < 18 or count_scale < 0.5
        ):
            dropped.append(str(item.get("name") or ""))
            continue
        density = dict(item)
        for key in ("phase1_arrival_rate", "phase2_arrival_rate"):
            if key in density:
                density[key] = round(float(density[key]) * rate_scale, 3)
        if not meta.get("preserve_density_request_counts"):
            for key, minimum in (
                ("phase1_short_count", 8),
                ("phase1_long_count", 3),
                ("phase2_short_count", 8),
                ("phase2_long_count", 4),
            ):
                if key in density:
                    density[key] = _scale_count(density[key], count_scale, minimum)
        resolved.append(density)
    return resolved or densities[:1], {
        "density_scale": rate_scale,
        "request_count_scale": count_scale,
        "base_max_num_batched_tokens": base,
        "resolved_max_num_batched_tokens": batch,
        "dropped_densities": dropped,
        "workload_capacity": meta,
    }


def _resolved_config(
    config: dict[str, Any],
    models: list[ResolvedModel],
    datasets: list[dict[str, Any]],
    densities: list[dict[str, Any]],
    runtime: dict[str, Any],
    policy: dict[str, Any],
    workload: dict[str, Any],
    density_meta: dict[str, Any],
) -> dict[str, Any]:
    output = deepcopy(config)
    output.update(
        models=[asdict(model) for model in models],
        datasets=datasets,
        phase1=policy["phase1"],
        phase2=policy["phase2"],
        adaptive_density_policy=policy["adaptive_density_policy"],
    )
    output["workload"] = (
        dict(output.get("workload") or {}) | workload["workload"] | {"densities": densities}
    )
    output["eval"] = (
        dict(output.get("eval") or {})
        | {
            key: runtime[key]
            for key in ("max_model_len", "max_num_batched_tokens", "gpu_memory_utilization")
        }
        | workload["eval"]
    )
    output["preflight"] = {
        "resolved_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "runtime_source": runtime["source"],
        "density_source": density_meta,
    }
    return output


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Resolve a Chapter 5 config for the current GPU.")
    for name, default in (
        ("config", "experiments/configs/openworkload_v1_local_realworld_lora8.json"),
        ("run-name", ""),
        ("out-root", ""),
        ("model-keys", ""),
        ("dataset-keys", ""),
        ("densities", ""),
        ("model-path", ""),
    ):
        parser.add_argument(f"--{name}", default=default)
    for name, default in (
        ("smoke-timeout-sec", 90),
        ("max-model-len", 3072),
        ("max-num-batched-tokens", 1536),
        ("timeout-sec", 90),
    ):
        parser.add_argument(f"--{name}", type=int, default=default)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.6)
    for name in (
        "skip-engine-smoke",
        "skip-lut-rebuild",
        "force-lut-rebuild",
        "dry-run",
        "engine-smoke-child",
        "trust-remote-code",
    ):
        parser.add_argument(f"--{name}", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.engine_smoke_child:
        return _engine_smoke_child(args)
    config = load_config(args.config)
    environment, run_root = (
        _detect_environment(),
        project_path(args.out_root or config.get("out_root") or "results/openworkload")
        / (args.run_name or time.strftime("%Y%m%d_%H%M%S_preflight")),
    )
    metadata, memory = run_root / "metadata", _gpu_memory_gb(environment)
    runtime = _runtime_config(dict(config.get("eval") or {}), memory)
    candidates = _candidate_models(config, args.model_keys)
    lut_report = _ensure_luts(candidates, config, runtime, args)
    models, model_selection = _resolve_selected_models(config, args.model_keys)
    datasets, dataset_selection = _resolve_selected_datasets(config, args.dataset_keys)
    densities = _resolve_selected_densities(config, args.densities)
    if args.dry_run and not models:
        models = candidates
    if not models or not datasets or not densities:
        raise RuntimeError("preflight selected no models, datasets, or densities")
    smoke = bool(environment["cuda_available"] and not args.skip_engine_smoke and not args.dry_run)
    diagnostics = []
    for model in models:
        path = _resolve_model_path(model, config, smoke)
        result = (
            _run_engine_smoke(model, path, runtime, args.smoke_timeout_sec, config)
            if smoke
            else {"status": "ok", "reason": "metadata_only"}
        )
        diagnostics.append(
            {
                "key": model.key,
                "model_id": model.model_id,
                "model_path": relative_to_repo(path) if Path(path).is_absolute() else path,
                "smoke": result,
                "status": result["status"],
            }
        )
    runnable = {row["key"] for row in diagnostics if row["status"] == "ok"}
    models = [model for model in models if model.key in runnable]
    if not models:
        raise RuntimeError("preflight found no runnable models")
    workload, workload_meta = _derive_workload_overrides(
        config=config, runtime_cfg=runtime, memory_gb=memory
    )
    densities, density_meta = _derive_densities(
        densities, runtime, dict(config.get("eval") or {}), workload_meta
    )
    policy = _derive_policy_overrides(config, runtime)
    resolved = _resolved_config(
        config, models, datasets, densities, runtime, policy, workload, density_meta
    )
    summary = {
        "run_root": relative_to_repo(run_root),
        "resolved_config": relative_to_repo(metadata / "resolved_config.json"),
        "selected_model_count": len(models),
        "engine_smoke_ran": smoke,
        "runtime": runtime,
        "workload": workload_meta,
        "derived_policy": policy["derived"],
        "density": density_meta,
    }
    if args.dry_run:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    for name, payload in {
        "resolved_environment": environment,
        "lut_preflight": lut_report,
        "model_selection_diagnostics": model_selection,
        "dataset_selection_diagnostics": dataset_selection,
        "model_preflight": diagnostics,
        "workload_capacity": {"overrides": workload, "meta": workload_meta},
        "resolved_config": resolved,
        "preflight_summary": summary,
    }.items():
        write_json(metadata / f"{name}.json", payload)
    print(
        f"[Preflight] run_root={summary['run_root']}\n[Preflight] resolved_config={summary['resolved_config']}\n[Preflight] selected_models={len(models)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
