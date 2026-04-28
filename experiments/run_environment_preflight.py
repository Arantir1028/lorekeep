from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
import traceback
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

from config import hw_config as hw_cfg
from experiments.lut_fingerprint import current_lut_fingerprint, lut_fingerprint_status
from experiments.model_assets import ensure_model_available
from experiments.openworkload_models import ResolvedModel, runtime_lut_is_valid
from experiments.openworkload_support import (
    apply_hf_resource_env,
    ensure_dir,
    load_config,
    project_path,
    relative_to_repo,
    repo_root,
    resource_policy,
    write_json,
)
from experiments.run_openworkload_execescape_suite import (
    _resolve_selected_datasets,
    _resolve_selected_densities,
    _resolve_selected_models,
)


def _import_version(module_name: str) -> Optional[str]:
    try:
        mod = __import__(module_name)
    except Exception:
        return None
    return str(getattr(mod, "__version__", "") or "")


def _detect_gpus() -> list[dict[str, Any]]:
    gpus: list[dict[str, Any]] = []
    try:
        import torch

        if torch.cuda.is_available():
            for idx in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(idx)
                gpus.append(
                    {
                        "index": idx,
                        "name": str(props.name),
                        "total_memory_bytes": int(props.total_memory),
                        "total_memory_gb": round(float(props.total_memory) / (1024**3), 2),
                        "capability": list(torch.cuda.get_device_capability(idx)),
                    }
                )
    except Exception as exc:
        gpus.append({"error": repr(exc)})
    if gpus:
        return gpus

    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
    except Exception:
        return gpus
    if proc.returncode != 0:
        return gpus
    for line in (proc.stdout or "").splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 3:
            continue
        try:
            mem_mib = float(parts[2])
        except Exception:
            mem_mib = 0.0
        gpus.append(
            {
                "index": int(parts[0]) if parts[0].isdigit() else parts[0],
                "name": parts[1],
                "total_memory_bytes": int(mem_mib * 1024 * 1024),
                "total_memory_gb": round(mem_mib / 1024.0, 2),
            }
        )
    return gpus


def _detect_nvidia_smi_versions() -> dict[str, str]:
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=driver_version,cuda_version",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
    except Exception:
        return {}
    if proc.returncode != 0:
        return {}
    for line in (proc.stdout or "").splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) >= 2 and parts[0] and parts[1]:
            return {"driver_version": parts[0], "nvidia_smi_cuda_version": parts[1]}
    return {}


def _detect_environment() -> dict[str, Any]:
    try:
        import torch

        torch_cuda = str(getattr(torch.version, "cuda", "") or "")
        cuda_available = bool(torch.cuda.is_available())
    except Exception:
        torch_cuda = ""
        cuda_available = False
    environment = {
        "python_executable": sys.executable,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "torch_version": _import_version("torch"),
        "torch_cuda": torch_cuda,
        "cuda_available": cuda_available,
        "vllm_version": _import_version("vllm"),
        "transformers_version": _import_version("transformers"),
        "datasets_version": _import_version("datasets"),
        "gpus": _detect_gpus(),
    }
    environment.update(_detect_nvidia_smi_versions())
    environment["lut_hardware_fingerprint"] = current_lut_fingerprint(environment)
    return environment


def _gpu_memory_gb(environment: dict[str, Any]) -> float:
    vals = [
        float(gpu.get("total_memory_gb") or 0.0)
        for gpu in list(environment.get("gpus") or [])
        if isinstance(gpu, dict)
    ]
    vals = [v for v in vals if v > 0.0]
    return min(vals) if vals else 0.0


def _candidate_runtime_configs(base_eval: dict[str, Any], memory_gb: float) -> list[dict[str, Any]]:
    base_model_len = int(base_eval.get("max_model_len", 3072))
    base_batched = int(base_eval.get("max_num_batched_tokens", 1536))
    base_gpu_mem = float(base_eval.get("gpu_memory_utilization", 0.60))

    if memory_gb <= 0:
        return [
            {
                "max_model_len": base_model_len,
                "max_num_batched_tokens": base_batched,
                "gpu_memory_utilization": base_gpu_mem,
                "source": "config_no_gpu_detected",
            }
        ]

    candidates: list[dict[str, Any]] = []
    if memory_gb >= 40:
        len_candidates = [base_model_len, max(base_model_len, 4096)]
        batch_candidates = [base_batched, max(base_batched, 2048)]
        mem_candidates = [max(base_gpu_mem, 0.70), base_gpu_mem]
    elif memory_gb >= 24:
        len_candidates = [base_model_len, min(base_model_len, 2048)]
        batch_candidates = [base_batched, min(base_batched, 1024)]
        mem_candidates = [max(base_gpu_mem, 0.70), base_gpu_mem]
    else:
        len_candidates = [min(base_model_len, 2048), min(base_model_len, 1536)]
        batch_candidates = [min(base_batched, 1024), min(base_batched, 768)]
        mem_candidates = [max(base_gpu_mem, 0.75), base_gpu_mem]

    seen: set[tuple[int, int, float]] = set()
    for model_len in len_candidates:
        for batched in batch_candidates:
            batched = min(int(batched), int(model_len))
            for gpu_mem in mem_candidates:
                key = (int(model_len), int(batched), round(float(gpu_mem), 3))
                if key in seen:
                    continue
                seen.add(key)
                candidates.append(
                    {
                        "max_model_len": key[0],
                        "max_num_batched_tokens": key[1],
                        "gpu_memory_utilization": key[2],
                        "source": "memory_heuristic",
                    }
                )
    return candidates


def _model_size_score(model: ResolvedModel) -> int:
    text = " ".join([model.key, model.model_id, model.lut_name]).lower()
    score = 0
    for marker, weight in (("9b", 90), ("7b", 70), ("6.9b", 69), ("6b", 60), ("2", 20)):
        if marker in text:
            score = max(score, weight)
    if model.max_model_len_override:
        score -= 5
    return score


def _representative_models(models: list[ResolvedModel], limit: int) -> list[ResolvedModel]:
    if not models:
        return []
    ordered = sorted(models, key=lambda item: (_model_size_score(item), item.key), reverse=True)
    chosen: list[ResolvedModel] = []
    for model in ordered:
        if model not in chosen:
            chosen.append(model)
        if len(chosen) >= max(1, int(limit)):
            break
    return chosen


def _write_smoke_child_result(payload: dict[str, Any]) -> int:
    print(json.dumps(payload, ensure_ascii=False), flush=True)
    return 0 if payload.get("status") == "ok" else 1


def _engine_smoke_child(args: argparse.Namespace) -> int:
    start = time.perf_counter()
    try:
        from engine.runtime_bootstrap import bootstrap_vllm_runtime

        bootstrap_vllm_runtime()
        from vllm.engine.arg_utils import EngineArgs
        from vllm.engine.llm_engine import LLMEngine
        from vllm.sampling_params import SamplingParams

        engine_args = EngineArgs(
            model=args.model_path,
            trust_remote_code=bool(args.trust_remote_code),
            seed=0,
            enable_lora=False,
            max_num_batched_tokens=int(args.max_num_batched_tokens),
            max_num_partial_prefills=1,
            max_long_partial_prefills=1,
            enable_chunked_prefill=True,
            disable_sliding_window=True,
            enforce_eager=True,
            max_model_len=int(args.max_model_len),
            gpu_memory_utilization=float(args.gpu_memory_utilization),
        )
        engine = LLMEngine.from_engine_args(engine_args)
        prompt = "Briefly answer: what is an online serving workload?"
        sampling = SamplingParams(max_tokens=1, temperature=0.0)
        engine.add_request("preflight_smoke", prompt, sampling)
        deadline = time.time() + int(args.timeout_sec)
        while time.time() < deadline and engine.has_unfinished_requests():
            engine.step()
        timed_out = bool(engine.has_unfinished_requests())
        elapsed = time.perf_counter() - start
        del engine
        if timed_out:
            return _write_smoke_child_result(
                {
                    "status": "failed",
                    "reason": "timeout",
                    "elapsed_s": elapsed,
                    "timeout_sec": int(args.timeout_sec),
                }
            )
        return _write_smoke_child_result({"status": "ok", "elapsed_s": elapsed})
    except Exception as exc:
        text = "".join(traceback.format_exception_only(type(exc), exc)).strip()
        reason = "oom" if "out of memory" in text.lower() or "cuda oom" in text.lower() else "error"
        return _write_smoke_child_result(
            {
                "status": "failed",
                "reason": reason,
                "error": text,
                "elapsed_s": time.perf_counter() - start,
            }
        )


def _run_engine_smoke(
    *,
    model: ResolvedModel,
    model_path: str,
    runtime_cfg: dict[str, Any],
    timeout_sec: int,
    config: dict[str, Any],
) -> dict[str, Any]:
    cmd = [
        sys.executable,
        "experiments/run_environment_preflight.py",
        "--engine-smoke-child",
        "--model-path",
        model_path,
        "--max-model-len",
        str(int(runtime_cfg["max_model_len"])),
        "--max-num-batched-tokens",
        str(int(runtime_cfg["max_num_batched_tokens"])),
        "--gpu-memory-utilization",
        str(float(runtime_cfg["gpu_memory_utilization"])),
        "--timeout-sec",
        str(int(timeout_sec)),
    ]
    if model.trust_remote_code:
        cmd.append("--trust-remote-code")
    env = apply_hf_resource_env(os.environ.copy(), config)
    env["WAVESLICE_VLLM_MODE"] = str((config.get("eval") or {}).get("vllm_mode") or "v1")
    env["VLLM_USE_V1"] = "1" if env["WAVESLICE_VLLM_MODE"] == "v1" else "0"
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False,
        env=env,
        cwd=str(repo_root()),
    )
    payload: dict[str, Any] = {}
    for line in reversed((proc.stdout or "").splitlines()):
        try:
            payload = json.loads(line)
            break
        except Exception:
            continue
    if not payload:
        payload = {
            "status": "failed",
            "reason": "child_failed",
            "stdout_tail": "\n".join((proc.stdout or "").splitlines()[-20:]),
            "stderr_tail": "\n".join((proc.stderr or "").splitlines()[-20:]),
        }
    payload["returncode"] = int(proc.returncode)
    return payload


def _resolve_model_path(
    model: ResolvedModel,
    config: dict[str, Any],
    *,
    allow_download: bool = True,
) -> tuple[str, Optional[str]]:
    policy = resource_policy(config)
    local_snapshot = ensure_model_available(
        model.model_id,
        auto_download=bool(policy["auto_download"]) and bool(allow_download),
        local_files_only=bool(policy["offline"]),
    )
    if model.model_path_mode == "model_id":
        return model.model_id, local_snapshot
    if model.model_path_mode == "local_snapshot_required" and local_snapshot:
        return local_snapshot, local_snapshot
    return local_snapshot or model.model_id, local_snapshot


def _lut_artifact_status(model: ResolvedModel, environment: dict[str, Any]) -> dict[str, Any]:
    paths = hw_cfg.get_lut_paths(model.lut_name)
    sanity_path = Path(hw_cfg.DATA_DIR) / f"runtime_sanity_{model.lut_name}.json"
    calibration_path = Path(hw_cfg.DATA_DIR) / f"runtime_calibration_{model.lut_name}.json"
    runtime_ok, runtime_reason = runtime_lut_is_valid(model.lut_name)
    fingerprint = lut_fingerprint_status(model.lut_name, environment=environment)
    artifacts = {
        "raw": Path(paths["raw"]).exists(),
        "gain": Path(paths["gain"]).exists(),
        "penalty": Path(paths["penalty"]).exists(),
        "runtime_sanity": sanity_path.exists(),
        "runtime_calibration": calibration_path.exists(),
    }
    missing = [name for name, exists in artifacts.items() if not exists]
    current = bool(runtime_ok and fingerprint.get("ok") and not missing)
    reason = ""
    if missing:
        reason = f"missing_artifacts:{','.join(missing)}"
    elif not runtime_ok:
        reason = runtime_reason
    elif not fingerprint.get("ok"):
        reason = str(fingerprint.get("reason") or "lut_hardware_fingerprint_mismatch")
    return {
        "key": model.key,
        "model_id": model.model_id,
        "lut_name": model.lut_name,
        "artifacts": artifacts,
        "runtime_sanity_ok": runtime_ok,
        "runtime_sanity_reason": runtime_reason,
        "lut_fingerprint_ok": bool(fingerprint.get("ok")),
        "lut_fingerprint_reason": str(fingerprint.get("reason") or ""),
        "current_fingerprint": fingerprint.get("current"),
        "stored_fingerprint": fingerprint.get("stored"),
        "status": "current" if current else "stale_or_missing",
        "reason": reason,
    }


def _resolve_lut_candidate_models(config: dict[str, Any], model_keys: str) -> tuple[list[ResolvedModel], list[dict[str, Any]]]:
    candidate_config = deepcopy(config)
    selection_cfg = dict(candidate_config.get("resource_selection") or {})
    selection_cfg["require_runtime_sanity"] = False
    candidate_config["resource_selection"] = selection_cfg
    return _resolve_selected_models(candidate_config, model_keys)


def _ensure_current_luts(
    *,
    models: list[ResolvedModel],
    config: dict[str, Any],
    environment: dict[str, Any],
    runtime_hint: dict[str, Any],
    skip_rebuild: bool,
    force_rebuild: bool,
    dry_run: bool,
) -> dict[str, Any]:
    before = [_lut_artifact_status(model, environment) for model in models]
    needs_rebuild = [
        model
        for model, status in zip(models, before)
        if force_rebuild or str(status.get("status")) != "current"
    ]
    report: dict[str, Any] = {
        "hardware_fingerprint": environment.get("lut_hardware_fingerprint"),
        "candidate_model_count": len(models),
        "rebuild_needed_count": len(needs_rebuild),
        "before": before,
        "rebuild": {
            "ran": False,
            "skipped": bool(skip_rebuild or dry_run or not needs_rebuild),
            "reason": (
                "dry_run"
                if dry_run and needs_rebuild
                else "skip_lut_rebuild"
                if skip_rebuild and needs_rebuild
                else "all_luts_current"
                if not needs_rebuild
                else ""
            ),
        },
        "after": before,
    }
    if not needs_rebuild or skip_rebuild or dry_run:
        return report

    resolved_model_ids: list[str] = []
    for model in needs_rebuild:
        _resolve_model_path(model, config, allow_download=True)
        resolved_model_ids.append(model.model_id)

    cmd = [
        sys.executable,
        "experiments/build_hybrid_checkpoint_runtime_luts.py",
        "--models",
        ",".join(resolved_model_ids),
        "--max-num-batched-tokens",
        str(int(runtime_hint.get("max_num_batched_tokens", 1536))),
        "--gpu-memory-utilization",
        str(float(runtime_hint.get("gpu_memory_utilization", 0.80))),
        "--force",
    ]
    env = apply_hf_resource_env(os.environ.copy(), config)
    env["WAVESLICE_VLLM_MODE"] = str((config.get("eval") or {}).get("vllm_mode") or "v1")
    env["VLLM_USE_V1"] = "1" if env["WAVESLICE_VLLM_MODE"] == "v1" else "0"
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False,
        env=env,
        cwd=str(repo_root()),
    )
    report["rebuild"] = {
        "ran": True,
        "skipped": False,
        "returncode": int(proc.returncode),
        "cmd": cmd,
        "stdout_tail": "\n".join((proc.stdout or "").splitlines()[-40:]),
        "stderr_tail": "\n".join((proc.stderr or "").splitlines()[-40:]),
    }
    report["after"] = [_lut_artifact_status(model, environment) for model in models]
    if proc.returncode != 0:
        report["rebuild"]["status"] = "failed"
    else:
        report["rebuild"]["status"] = "ok"
    return report


def _model_diagnostics(
    *,
    models: list[ResolvedModel],
    config: dict[str, Any],
    runtime_cfg: dict[str, Any],
    smoke_models: set[str],
    run_smoke: bool,
    smoke_timeout_sec: int,
    environment: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model in models:
        runtime_ok, runtime_reason = runtime_lut_is_valid(model.lut_name)
        fingerprint = lut_fingerprint_status(model.lut_name, environment=environment)
        row: dict[str, Any] = {
            "key": model.key,
            "model_id": model.model_id,
            "lut_name": model.lut_name,
            "label": model.label,
            "trust_remote_code": bool(model.trust_remote_code),
            "max_model_len_override": model.max_model_len_override,
            "runtime_sanity_ok": runtime_ok,
            "runtime_sanity_reason": runtime_reason,
            "lut_fingerprint_ok": bool(fingerprint.get("ok")),
            "lut_fingerprint_reason": str(fingerprint.get("reason") or ""),
            "lut_fingerprint_current": fingerprint.get("current"),
            "lut_fingerprint_stored": fingerprint.get("stored"),
            "status": "pending",
        }
        try:
            model_path, local_snapshot = _resolve_model_path(model, config, allow_download=run_smoke)
            row["model_path"] = relative_to_repo(model_path) if Path(str(model_path)).is_absolute() else model_path
            row["local_snapshot"] = relative_to_repo(local_snapshot) if local_snapshot else ""
            if not runtime_ok:
                row["status"] = "skipped"
                row["reason"] = "missing_or_failed_lut"
            elif run_smoke and model.key in smoke_models:
                smoke_cfg = dict(runtime_cfg)
                if model.max_model_len_override:
                    smoke_cfg["max_model_len"] = min(
                        int(smoke_cfg["max_model_len"]),
                        int(model.max_model_len_override),
                    )
                    smoke_cfg["max_num_batched_tokens"] = min(
                        int(smoke_cfg["max_num_batched_tokens"]),
                        int(smoke_cfg["max_model_len"]),
                    )
                smoke = _run_engine_smoke(
                    model=model,
                    model_path=model_path,
                    runtime_cfg=smoke_cfg,
                    timeout_sec=smoke_timeout_sec,
                    config=config,
                )
                row["smoke"] = smoke
                row["status"] = "ok" if smoke.get("status") == "ok" else "failed"
                row["reason"] = smoke.get("reason", "")
            else:
                row["status"] = "ok"
                row["reason"] = "metadata_only"
        except Exception as exc:
            row["status"] = "failed"
            row["reason"] = "resolve_or_download_failed"
            row["error"] = "".join(traceback.format_exception_only(type(exc), exc)).strip()
        rows.append(row)
    return rows


def _choose_runtime_config(
    *,
    candidates: list[dict[str, Any]],
    representatives: list[ResolvedModel],
    config: dict[str, Any],
    run_smoke: bool,
    smoke_timeout_sec: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    attempts: list[dict[str, Any]] = []
    if not run_smoke or not representatives:
        return dict(candidates[0]), [{"runtime": dict(candidates[0]), "status": "selected_without_smoke"}]
    for candidate in candidates:
        model_results: list[dict[str, Any]] = []
        all_ok = True
        for model in representatives:
            try:
                model_path, _ = _resolve_model_path(model, config, allow_download=True)
                effective = dict(candidate)
                if model.max_model_len_override:
                    effective["max_model_len"] = min(int(effective["max_model_len"]), int(model.max_model_len_override))
                    effective["max_num_batched_tokens"] = min(
                        int(effective["max_num_batched_tokens"]),
                        int(effective["max_model_len"]),
                    )
                smoke = _run_engine_smoke(
                    model=model,
                    model_path=model_path,
                    runtime_cfg=effective,
                    timeout_sec=smoke_timeout_sec,
                    config=config,
                )
                model_results.append({"model_key": model.key, "smoke": smoke})
                if smoke.get("status") != "ok":
                    all_ok = False
                    break
            except Exception as exc:
                model_results.append(
                    {
                        "model_key": model.key,
                        "smoke": {
                            "status": "failed",
                            "reason": "resolve_or_download_failed",
                            "error": "".join(traceback.format_exception_only(type(exc), exc)).strip(),
                        },
                    }
                )
                all_ok = False
                break
        attempts.append({"runtime": dict(candidate), "status": "ok" if all_ok else "failed", "models": model_results})
        if all_ok:
            return dict(candidate), attempts
    fallback = dict(candidates[-1])
    fallback["source"] = f"{fallback.get('source', 'unknown')}_fallback_after_failed_smoke"
    return fallback, attempts


def _align_bucket(value: int, buckets: list[int]) -> int:
    ordered = sorted(int(item) for item in buckets if int(item) > 0)
    if not ordered:
        return max(1, int(value))
    for bucket in ordered:
        if bucket >= int(value):
            return int(bucket)
    return int(ordered[-1])


def _derive_policy_overrides(config: dict[str, Any], runtime_cfg: dict[str, Any]) -> dict[str, Any]:
    from config.hw_config import BUCKETS

    max_model_len = int(runtime_cfg.get("max_model_len", 3072))
    max_batched = int(runtime_cfg.get("max_num_batched_tokens", 1536))
    aggressive = _align_bucket(max(128, min(max_batched, int(max_model_len * 0.25))), list(BUCKETS))
    conservative = _align_bucket(max(aggressive, min(max_batched, int(max_model_len * 0.50))), list(BUCKETS))
    waiting_short = max(2, min(4, max_batched // max(1, aggressive)))
    queue_high = max(4, waiting_short * 2)
    long_high = max(conservative, int(max_model_len * 0.75))

    adaptive = dict(config.get("adaptive_density_policy") or {})
    adaptive.update(
        {
            "runtime_aggressive_ingress_target_chunk": int(aggressive),
            "runtime_conservative_ingress_target_chunk": int(conservative),
            "high_pressure_ingress_target_chunk": int(aggressive),
            "low_pressure_ingress_target_chunk": int(conservative),
            "runtime_queue_high_watermark": int(queue_high),
            "runtime_waiting_short_high_watermark": int(waiting_short),
            "runtime_long_high_watermark": int(long_high),
            "high_pressure_phase2_min_long_prefill": int(aggressive),
            "low_pressure_phase2_min_long_prefill": int(conservative),
        }
    )

    phase1 = dict(config.get("phase1") or {})
    phase1["ingress_target_chunk"] = int(aggressive)
    phase1["force_min_chunk"] = min(int(phase1.get("force_min_chunk", 128)), int(aggressive))

    phase2 = dict(config.get("phase2") or {})
    phase2["min_long_prefill"] = int(aggressive)
    if max_batched <= 1024:
        phase2["execution_escape_spillover_cap"] = min(int(phase2.get("execution_escape_spillover_cap", 3)), 2)
        phase2["execution_escape_max_active"] = min(int(phase2.get("execution_escape_max_active", 5)), 3)

    return {
        "phase1": phase1,
        "adaptive_density_policy": adaptive,
        "phase2": phase2,
        "derived": {
            "aggressive_chunk": int(aggressive),
            "conservative_chunk": int(conservative),
            "queue_high_watermark": int(queue_high),
            "waiting_short_high_watermark": int(waiting_short),
            "long_high_watermark": int(long_high),
        },
    }


def _derive_timeout_sec(config: dict[str, Any], model_preflight: list[dict[str, Any]]) -> int:
    current = int((config.get("eval") or {}).get("timeout_sec", 240))
    elapsed_vals: list[float] = []
    for row in model_preflight:
        smoke = row.get("smoke")
        if not isinstance(smoke, dict) or smoke.get("status") != "ok":
            continue
        try:
            elapsed = float(smoke.get("elapsed_s") or 0.0)
        except Exception:
            elapsed = 0.0
        if elapsed > 0:
            elapsed_vals.append(elapsed)
    if not elapsed_vals:
        return current
    observed = max(elapsed_vals)
    return int(max(current, 60, round(observed * 20.0)))


def _memory_workload_scale(memory_gb: float) -> float:
    if memory_gb >= 24.0:
        return 1.0
    if memory_gb >= 18.0:
        return 0.75
    if memory_gb >= 12.0:
        return 0.50
    return 0.35


def _bucketed_max_new_tokens(base: int, scale: float, memory_gb: float) -> int:
    if memory_gb >= 24.0 and scale >= 0.95:
        limit = int(base)
    elif memory_gb >= 18.0 and scale >= 0.70:
        limit = min(int(base), 48)
    else:
        limit = min(int(base), 32)
    choices = [16, 32, 48, 64, 96, 128]
    for value in choices:
        if value >= limit:
            return max(16, min(int(base), int(value)))
    return max(16, min(int(base), int(limit)))


def _scale_count(value: Any, scale: float, *, minimum: int) -> int:
    try:
        base = int(value)
    except Exception:
        base = int(minimum)
    return max(int(minimum), int(round(float(base) * float(scale))))


def _derive_workload_overrides(
    *,
    config: dict[str, Any],
    runtime_cfg: dict[str, Any],
    memory_gb: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    base_eval = dict(config.get("eval") or {})
    base_workload = dict(config.get("workload") or {})
    base_batched = max(1, int(base_eval.get("max_num_batched_tokens", 1536)))
    current_batched = max(1, int(runtime_cfg.get("max_num_batched_tokens", base_batched)))
    batch_scale = max(0.25, min(1.0, float(current_batched) / float(base_batched)))
    memory_scale = _memory_workload_scale(float(memory_gb))
    scale = max(0.25, min(1.0, batch_scale, memory_scale))

    base_max_new = int(base_eval.get("max_new_tokens", 64))
    max_new_tokens = _bucketed_max_new_tokens(base_max_new, scale, float(memory_gb))
    repeats = int(base_eval.get("repeats", 2))
    if scale < 0.50:
        repeats = min(repeats, 1)
    warmup_iters = int(base_eval.get("warmup_iters", 1))
    sample_count = _scale_count(base_workload.get("sample_count", 256), scale, minimum=64)

    eval_overrides = {
        "max_new_tokens": int(max_new_tokens),
        "repeats": int(repeats),
        "warmup_iters": int(warmup_iters),
    }
    workload_overrides = {"sample_count": int(sample_count)}
    meta = {
        "scale": scale,
        "memory_scale": memory_scale,
        "batch_scale": batch_scale,
        "memory_gb": memory_gb,
        "base_max_num_batched_tokens": base_batched,
        "resolved_max_num_batched_tokens": current_batched,
        "base_max_new_tokens": base_max_new,
        "resolved_max_new_tokens": max_new_tokens,
        "base_repeats": int(base_eval.get("repeats", 2)),
        "resolved_repeats": repeats,
        "base_sample_count": int(base_workload.get("sample_count", 256)),
        "resolved_sample_count": sample_count,
        "density_policy": "scale_arrival_and_counts_drop_peak_when_capacity_is_low",
    }
    return {"eval": eval_overrides, "workload": workload_overrides}, meta


def _derive_densities(
    densities: list[dict[str, Any]],
    runtime_cfg: dict[str, Any],
    base_eval: dict[str, Any],
    workload_meta: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    base_batched = max(1, int(base_eval.get("max_num_batched_tokens", 1536)))
    current_batched = max(1, int(runtime_cfg.get("max_num_batched_tokens", base_batched)))
    rate_scale = max(0.35, min(1.50, float(current_batched) / float(base_batched)))
    count_scale = max(0.25, min(1.0, float(workload_meta.get("scale") or 1.0)))
    memory_gb = float(workload_meta.get("memory_gb") or 0.0)
    dropped: list[str] = []
    resolved: list[dict[str, Any]] = []
    for density in densities:
        name = str(density.get("name") or "")
        if name == "peak" and (memory_gb < 18.0 or count_scale < 0.50):
            dropped.append(name)
            continue
        copied = dict(density)
        for key in ("phase1_arrival_rate", "phase2_arrival_rate"):
            if key in copied:
                copied[key] = round(float(copied[key]) * rate_scale, 3)
        for key, minimum in (
            ("phase1_short_count", 8),
            ("phase1_long_count", 3),
            ("phase2_short_count", 8),
            ("phase2_long_count", 4),
        ):
            if key in copied:
                copied[key] = _scale_count(copied[key], count_scale, minimum=minimum)
        resolved.append(copied)
    if not resolved and densities:
        density = dict(densities[0])
        for key in ("phase1_arrival_rate", "phase2_arrival_rate"):
            if key in density:
                density[key] = round(float(density[key]) * rate_scale, 3)
        resolved.append(density)
    return resolved, {
        "density_scale": rate_scale,
        "request_count_scale": count_scale,
        "source": "global_batch_token_capacity_and_memory_workload_capacity",
        "base_max_num_batched_tokens": base_batched,
        "resolved_max_num_batched_tokens": current_batched,
        "dropped_densities": dropped,
        "workload_capacity": workload_meta,
    }


def _build_resolved_config(
    *,
    config: dict[str, Any],
    models: list[ResolvedModel],
    datasets: list[dict[str, Any]],
    densities: list[dict[str, Any]],
    runtime_cfg: dict[str, Any],
    policy_overrides: dict[str, Any],
    workload_overrides: dict[str, Any],
    density_meta: dict[str, Any],
) -> dict[str, Any]:
    resolved = deepcopy(config)
    resolved["models"] = [asdict(model) for model in models]
    resolved["datasets"] = datasets
    resolved.setdefault("workload", {})
    resolved["workload"] = dict(resolved.get("workload") or {})
    resolved["workload"].update(workload_overrides.get("workload", {}))
    resolved["workload"]["densities"] = densities
    resolved["eval"] = dict(resolved.get("eval") or {})
    resolved["eval"].pop("python_bin", None)
    resolved["eval"].update(
        {
            "max_model_len": int(runtime_cfg["max_model_len"]),
            "max_num_batched_tokens": int(runtime_cfg["max_num_batched_tokens"]),
            "gpu_memory_utilization": float(runtime_cfg["gpu_memory_utilization"]),
        }
    )
    resolved["eval"].update(workload_overrides.get("eval", {}))
    if runtime_cfg.get("timeout_sec") is not None:
        resolved["eval"]["timeout_sec"] = int(runtime_cfg["timeout_sec"])
    resolved["phase1"] = policy_overrides["phase1"]
    resolved["adaptive_density_policy"] = policy_overrides["adaptive_density_policy"]
    resolved["phase2"] = policy_overrides["phase2"]
    resolved["preflight"] = {
        "resolved_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "runtime_source": runtime_cfg.get("source"),
        "policy_source": "global_common_parameters",
        "density_source": density_meta,
        "model_parameter_policy": "common_runtime_with_model_skip_on_failure",
    }
    return resolved


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare a portable resolved experiment config for the current GPU environment.")
    parser.add_argument("--config", default="experiments/configs/openworkload_v1_local_realworld_lora8.json")
    parser.add_argument("--run-name", default="")
    parser.add_argument("--out-root", default="")
    parser.add_argument("--model-keys", default="")
    parser.add_argument("--dataset-keys", default="")
    parser.add_argument("--densities", default="")
    parser.add_argument("--representative-count", type=int, default=1)
    parser.add_argument("--smoke-timeout-sec", type=int, default=90)
    parser.add_argument("--skip-engine-smoke", action="store_true")
    parser.add_argument("--skip-lut-rebuild", action="store_true")
    parser.add_argument("--force-lut-rebuild", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--engine-smoke-child", action="store_true")
    parser.add_argument("--model-path", default="")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--max-model-len", type=int, default=3072)
    parser.add_argument("--max-num-batched-tokens", type=int, default=1536)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.6)
    parser.add_argument("--timeout-sec", type=int, default=90)
    args = parser.parse_args()

    if args.engine_smoke_child:
        return _engine_smoke_child(args)

    config = load_config(args.config)
    environment = _detect_environment()
    runtime_candidates = _candidate_runtime_configs(dict(config.get("eval") or {}), _gpu_memory_gb(environment))
    run_name = args.run_name or time.strftime("%Y%m%d_%H%M%S_preflight")
    base_out_root = project_path(
        args.out_root
        or str(config.get("out_root") or "results/openworkload_v1_local_realworld_lora8")
    )
    run_root = base_out_root / run_name
    metadata_dir = run_root / "metadata"
    lut_candidate_models, lut_candidate_selection = _resolve_lut_candidate_models(config, args.model_keys)
    lut_preflight = _ensure_current_luts(
        models=lut_candidate_models,
        config=config,
        environment=environment,
        runtime_hint=dict(runtime_candidates[0]),
        skip_rebuild=bool(args.skip_lut_rebuild),
        force_rebuild=bool(args.force_lut_rebuild),
        dry_run=bool(args.dry_run),
    )
    models, model_selection = _resolve_selected_models(config, args.model_keys)
    datasets, dataset_selection = _resolve_selected_datasets(config, args.dataset_keys)
    densities = _resolve_selected_densities(config, args.densities)
    if args.dry_run and not models:
        models = lut_candidate_models
        model_selection = lut_candidate_selection
    if not models:
        if not args.dry_run:
            ensure_dir(metadata_dir)
            write_json(metadata_dir / "resolved_environment.json", environment)
            write_json(metadata_dir / "lut_candidate_selection_diagnostics.json", lut_candidate_selection)
            write_json(metadata_dir / "lut_preflight.json", lut_preflight)
            write_json(metadata_dir / "model_selection_diagnostics.json", model_selection)
        raise RuntimeError(
            "preflight selected no models after LUT fingerprint checks; "
            "inspect metadata/lut_preflight.json or rerun without --skip-lut-rebuild"
        )
    if not datasets:
        raise RuntimeError("preflight selected no datasets")
    if not densities:
        raise RuntimeError("preflight selected no densities")

    representatives = _representative_models(models, args.representative_count)
    run_smoke = bool(environment.get("cuda_available")) and not bool(args.skip_engine_smoke) and not bool(args.dry_run)
    selected_runtime, runtime_attempts = _choose_runtime_config(
        candidates=runtime_candidates,
        representatives=representatives,
        config=config,
        run_smoke=run_smoke,
        smoke_timeout_sec=int(args.smoke_timeout_sec),
    )
    smoke_model_keys = {model.key for model in models}
    model_preflight = _model_diagnostics(
        models=models,
        config=config,
        runtime_cfg=selected_runtime,
        smoke_models=smoke_model_keys,
        run_smoke=run_smoke,
        smoke_timeout_sec=int(args.smoke_timeout_sec),
        environment=environment,
    )
    ok_model_keys = {str(row.get("key")) for row in model_preflight if str(row.get("status")) == "ok"}
    resolved_models = [model for model in models if model.key in ok_model_keys]
    if not resolved_models and args.dry_run:
        resolved_models = models
    elif not resolved_models:
        ensure_dir(metadata_dir)
        write_json(metadata_dir / "resolved_environment.json", environment)
        write_json(metadata_dir / "lut_candidate_selection_diagnostics.json", lut_candidate_selection)
        write_json(metadata_dir / "lut_preflight.json", lut_preflight)
        write_json(metadata_dir / "model_selection_diagnostics.json", model_selection)
        write_json(metadata_dir / "dataset_selection_diagnostics.json", dataset_selection)
        write_json(metadata_dir / "model_preflight.json", model_preflight)
        write_json(metadata_dir / "runtime_capacity.json", {"selected": selected_runtime, "attempts": runtime_attempts})
        raise RuntimeError(
            "preflight found no runnable models after LUT fingerprint and smoke checks; "
            "inspect metadata/model_preflight.json"
        )
    selected_runtime["timeout_sec"] = _derive_timeout_sec(config, model_preflight)
    workload_overrides, workload_meta = _derive_workload_overrides(
        config=config,
        runtime_cfg=selected_runtime,
        memory_gb=_gpu_memory_gb(environment),
    )
    resolved_densities, density_meta = _derive_densities(
        densities,
        selected_runtime,
        dict(config.get("eval") or {}),
        workload_meta,
    )
    policy_overrides = _derive_policy_overrides(config, selected_runtime)
    resolved_config = _build_resolved_config(
        config=config,
        models=resolved_models,
        datasets=datasets,
        densities=resolved_densities,
        runtime_cfg=selected_runtime,
        policy_overrides=policy_overrides,
        workload_overrides=workload_overrides,
        density_meta=density_meta,
    )

    summary = {
        "run_root": relative_to_repo(run_root),
        "resolved_config": relative_to_repo(metadata_dir / "resolved_config.json"),
        "selected_model_count": len(resolved_models),
        "candidate_model_count": len(models),
        "representative_models": [model.key for model in representatives],
        "lut_rebuild_ran": bool((lut_preflight.get("rebuild") or {}).get("ran")),
        "lut_rebuild_needed_count": int(lut_preflight.get("rebuild_needed_count") or 0),
        "engine_smoke_ran": run_smoke,
        "runtime": selected_runtime,
        "workload": workload_meta,
        "derived_policy": policy_overrides["derived"],
        "density": density_meta,
    }
    if args.dry_run:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    ensure_dir(metadata_dir)
    write_json(metadata_dir / "resolved_environment.json", environment)
    write_json(metadata_dir / "lut_candidate_selection_diagnostics.json", lut_candidate_selection)
    write_json(metadata_dir / "lut_preflight.json", lut_preflight)
    write_json(metadata_dir / "model_selection_diagnostics.json", model_selection)
    write_json(metadata_dir / "dataset_selection_diagnostics.json", dataset_selection)
    write_json(metadata_dir / "model_preflight.json", model_preflight)
    write_json(metadata_dir / "runtime_capacity.json", {"selected": selected_runtime, "attempts": runtime_attempts})
    write_json(metadata_dir / "workload_capacity.json", {"overrides": workload_overrides, "meta": workload_meta})
    write_json(metadata_dir / "resolved_config.json", resolved_config)
    write_json(metadata_dir / "preflight_summary.json", summary)
    print(f"[Preflight] run_root={summary['run_root']}")
    print(f"[Preflight] resolved_config={summary['resolved_config']}")
    print(f"[Preflight] selected_models={len(resolved_models)}/{len(models)}")
    print(f"[Preflight] lut_rebuild_ran={summary['lut_rebuild_ran']}")
    print(f"[Preflight] engine_smoke_ran={run_smoke}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
