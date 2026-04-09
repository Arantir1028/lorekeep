from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import signal
import shutil
import subprocess
import sys
import time
from pathlib import Path
from statistics import median
from typing import Any

from engine.runtime_bootstrap import bootstrap_vllm_runtime

bootstrap_vllm_runtime()

import torch
from config import hw_config as cfg
from profiler import offline_profiler as offline_profiler_mod
from profiler.lut_generator import generate_lut_for_model
from profiler.offline_profiler import ModelProfiler
from tools.experiment_lock import gpu_experiment_lock


def _discover_local_snapshots() -> list[tuple[str, Path]]:
    hub = Path.home() / ".cache" / "huggingface" / "hub"
    out: list[tuple[str, Path]] = []
    for repo in sorted(hub.glob("models--*")):
        snaps = repo / "snapshots"
        if not snaps.exists():
            continue
        for snap in sorted(snaps.iterdir()):
            if snap.is_dir() and (snap / "config.json").exists():
                model_id = repo.name[len("models--") :].replace("--", "/")
                out.append((model_id, snap))
                break
    return out


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _infer_params(config_json: dict[str, Any]) -> dict[str, int]:
    q_heads = config_json.get("num_attention_heads") or config_json.get("n_head") or config_json.get("num_heads")
    d_model = (
        config_json.get("hidden_size")
        or config_json.get("n_embd")
        or config_json.get("d_model")
        or config_json.get("model_dim")
        or config_json.get("n_embed")
    )
    kv_heads = (
        config_json.get("num_key_value_heads")
        or config_json.get("n_head_kv")
        or config_json.get("multi_query_group_num")
    )
    if kv_heads is None:
        kv_heads = 1 if bool(config_json.get("multi_query")) else q_heads
    if not all(v is not None for v in (q_heads, kv_heads, d_model)):
        raise ValueError("cannot infer q_heads/kv_heads/d_model from config")
    q_heads_i = int(q_heads)
    kv_heads_i = int(kv_heads)
    d_model_i = int(d_model)
    if d_model_i % q_heads_i != 0:
        raise ValueError(f"d_model={d_model_i} not divisible by q_heads={q_heads_i}")
    return {
        "q_heads": q_heads_i,
        "kv_heads": kv_heads_i,
        "d_model": d_model_i,
        "head_dim": d_model_i // q_heads_i,
    }


def _estimate_batch_size(
    *,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
    d_model: int,
    dtype_bytes: int,
    max_bucket: int,
    batch_cap: int,
    budget_bytes: int,
) -> int:
    per_sample_elems = (
        q_heads * max_bucket * head_dim
        + 2 * kv_heads * max_bucket * head_dim
        + 2 * max_bucket * d_model
        + 2 * d_model * cfg.LORA_RANK
    )
    per_sample_bytes = int(per_sample_elems * dtype_bytes * 1.35)
    return max(1, min(batch_cap, max(1, budget_bytes // max(per_sample_bytes, 1))))


def _selected_models(local: list[tuple[str, Path]], models_arg: str) -> list[tuple[str, Path]]:
    if models_arg.strip().lower() in {"all", "all-local"}:
        return local
    chosen = {m.strip() for m in models_arg.split(",") if m.strip()}
    selected = [(mid, snap) for mid, snap in local if mid in chosen or cfg.checkpoint_lut_name(mid) in chosen]
    missing = chosen - {mid for mid, _ in selected} - {cfg.checkpoint_lut_name(mid) for mid, _ in selected}
    if missing:
        raise ValueError(f"unknown local models: {sorted(missing)}")
    return selected


def _maybe_trust_remote_code(model_id: str) -> bool:
    lowered = model_id.lower()
    return any(k in lowered for k in ("baichuan", "deci", "falcon", "qwen", "mixtral"))


def _safe_max_pos(config_json: dict[str, Any], default: int = 2048) -> int:
    cand = (
        config_json.get("max_position_embeddings")
        or config_json.get("n_positions")
        or config_json.get("n_ctx")
        or config_json.get("seq_length")
        or config_json.get("max_sequence_length")
        or default
    )
    try:
        return int(cand)
    except Exception:
        return default


def _pick_calibration_buckets(max_model_len: int, *, reserve_new_tokens: int = 8) -> list[int]:
    candidates = [256, 512, 1024, 1536, 2048, 3072]
    usable_limit = max(1, int(max_model_len) - max(1, int(reserve_new_tokens)))
    out = [b for b in candidates if b <= usable_limit]
    if not out:
        out = [min(256, usable_limit)]
    if len(out) > 4:
        out = [out[0], out[1], out[-2], out[-1]]
    return sorted(set(out))


def _gpu_compute_pids() -> list[int]:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=pid",
                "--format=csv,noheader",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return []
    pids: list[int] = []
    for line in out.splitlines():
        line = line.strip()
        if not line or not line[0].isdigit():
            continue
        try:
            pids.append(int(line.split(",")[0].strip()))
        except Exception:
            continue
    return sorted(set(pids))


def _ensure_clean_gpu(*, grace_s: float = 2.0) -> None:
    self_pid = os.getpid()
    for _ in range(3):
        pids = [pid for pid in _gpu_compute_pids() if pid != self_pid]
        if not pids:
            return
        for pid in pids:
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.kill(pid, signal.SIGKILL)
        time.sleep(grace_s)


def _runtime_calibration(
    *,
    model_id: str,
    snapshot: Path,
    trust_remote_code: bool,
    buckets: list[int],
    max_model_len: int,
    max_num_batched_tokens: int,
    gpu_memory_utilization: float,
    repeats: int,
) -> dict[str, Any]:
    out_path = Path(cfg.DATA_DIR) / f"runtime_calibration_{cfg.checkpoint_lut_name(model_id)}.json"
    cmd = [
        sys.executable,
        str(Path(__file__).with_name("run_runtime_calibration.py")),
        "--model-ref",
        str(snapshot),
        "--snapshot",
        str(snapshot),
        "--max-model-len",
        str(int(max_model_len)),
        "--max-num-batched-tokens",
        str(int(max_num_batched_tokens)),
        "--gpu-memory-utilization",
        str(float(gpu_memory_utilization)),
        "--repeats",
        str(int(repeats)),
        "--buckets",
        ",".join(str(int(b)) for b in buckets),
        "--out",
        str(out_path),
    ]
    if trust_remote_code:
        cmd.append("--trust-remote-code")
    env = os.environ.copy()
    env.setdefault("WAVESLICE_VLLM_MODE", "v1")
    env.setdefault("VLLM_USE_V1", "1")
    proc = subprocess.Popen(cmd, env=env, start_new_session=True)
    rc = proc.wait()
    with contextlib.suppress(ProcessLookupError):
        os.killpg(proc.pid, signal.SIGKILL)
    if rc != 0:
        raise subprocess.CalledProcessError(rc, cmd)
    return _load_json(out_path)


def _runtime_sanity_path(lut_name: str) -> Path:
    return Path(cfg.DATA_DIR) / f"runtime_sanity_{lut_name}.json"


def _runtime_sanity_check(runtime: dict[str, Any]) -> dict[str, Any]:
    buckets = [int(b) for b in runtime.get("buckets") or []]
    solo_raw = runtime.get("solo_us") or {}
    conc_raw = runtime.get("concurrent_short_us") or {}
    reasons: list[str] = []
    ratios: list[float] = []
    solo_vals: list[tuple[int, float]] = []

    for bucket in buckets:
        val = solo_raw.get(str(bucket), solo_raw.get(bucket))
        if val is None:
            reasons.append(f"missing_solo:{bucket}")
            continue
        try:
            fval = float(val)
        except Exception:
            reasons.append(f"bad_solo:{bucket}")
            continue
        if not math.isfinite(fval) or fval <= 0:
            reasons.append(f"nonpositive_solo:{bucket}")
            continue
        solo_vals.append((bucket, fval))

    for idx in range(1, len(solo_vals)):
        prev_bucket, prev_val = solo_vals[idx - 1]
        bucket, val = solo_vals[idx]
        if val < prev_val * 0.90:
            reasons.append(f"solo_non_monotonic:{prev_bucket}->{bucket}")

    for short_bucket in buckets:
        row = conc_raw.get(str(short_bucket))
        if row is None:
            row = conc_raw.get(short_bucket, {})
        if not isinstance(row, dict):
            reasons.append(f"bad_concurrent_row:{short_bucket}")
            continue
        solo = None
        for bucket, value in solo_vals:
            if bucket == short_bucket:
                solo = value
                break
        if solo is None:
            continue
        for chunk_bucket, val in row.items():
            try:
                ratio = float(val) / max(float(solo), 1e-9)
            except Exception:
                reasons.append(f"bad_ratio:{short_bucket}:{chunk_bucket}")
                continue
            ratios.append(ratio)
            if ratio < 0.20 or ratio > 2.0:
                reasons.append(f"ratio_out_of_range:{short_bucket}:{chunk_bucket}:{ratio:.3f}")

    passed = bool(buckets) and not reasons
    return {
        "passed": passed,
        "buckets": buckets,
        "solo_bucket_count": len(solo_vals),
        "ratio_count": len(ratios),
        "ratio_min": min(ratios) if ratios else None,
        "ratio_median": median(ratios) if ratios else None,
        "ratio_max": max(ratios) if ratios else None,
        "reasons": reasons,
    }


def _ensure_base_profile(
    *,
    lut_name: str,
    params: dict[str, int],
    device: torch.device,
    dtype: torch.dtype,
    batch_size: int,
    warmup_iters: int,
    active_iters: int,
) -> None:
    paths = cfg.get_lut_paths(lut_name)
    if all(os.path.exists(paths[key]) for key in ("raw", "gain", "penalty")):
        return
    cfg.register_checkpoint_model(lut_name, aliases=[lut_name], **params)
    base_batch_size = int(cfg.BATCH_SIZE)
    offline_profiler_mod.WARMUP_ITERS = max(1, int(warmup_iters))
    offline_profiler_mod.ACTIVE_ITERS = max(1, int(active_iters))
    try:
        cfg.BATCH_SIZE = int(batch_size)
        profiler = ModelProfiler(lut_name, device, dtype)
        profiler.run()
        generate_lut_for_model(lut_name)
    finally:
        cfg.BATCH_SIZE = base_batch_size


def _backup_if_missing(src: str, dst: str) -> None:
    if os.path.exists(src) and not os.path.exists(dst):
        shutil.copy2(src, dst)


def _build_hybrid_raw(
    *,
    base_raw: dict[str, Any],
    runtime: dict[str, Any],
) -> dict[str, Any]:
    out = json.loads(json.dumps(base_raw))
    solo_runtime = {int(k): float(v) for k, v in runtime["solo_us"].items()}
    conc_runtime = {
        int(k): {int(kk): float(vv) for kk, vv in row.items()}
        for k, row in runtime["concurrent_short_us"].items()
    }

    base_solo = {int(k): float(v) for k, v in base_raw["T_solo"].items()}
    ratios = []
    for bucket, rt in solo_runtime.items():
        base = base_solo.get(bucket)
        if base and base > 0:
            ratios.append(rt / base)
    runtime_scale = float(median(ratios)) if ratios else 1.0

    for bucket, rt in solo_runtime.items():
        out["T_solo"][str(bucket)] = rt
    for short_bucket, row in conc_runtime.items():
        out.setdefault("T_conc", {}).setdefault(str(short_bucket), {})
        for chunk_bucket, rt in row.items():
            out["T_conc"][str(short_bucket)][str(chunk_bucket)] = rt
    for long_bucket, row in list(out.get("T_read_amp", {}).items()):
        for chunk_bucket, val in list(row.items()):
            out["T_read_amp"][long_bucket][chunk_bucket] = float(val) * runtime_scale
    return out


def _write_registry(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(json.dumps({"models": rows}, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build hybrid checkpoint-runtime LUTs for all local models.")
    parser.add_argument("--models", default="all-local")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=["fp16", "bf16"], default="fp16")
    parser.add_argument("--base-warmup-iters", type=int, default=3)
    parser.add_argument("--base-active-iters", type=int, default=8)
    parser.add_argument("--runtime-repeats", type=int, default=3)
    parser.add_argument("--budget-frac", type=float, default=0.18)
    parser.add_argument("--batch-size-cap", type=int, default=128)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.80)
    parser.add_argument("--max-num-batched-tokens", type=int, default=1536)
    parser.add_argument("--gpu-lock-path", default="")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to build hybrid checkpoint-runtime LUTs.")

    local = _discover_local_snapshots()
    targets = _selected_models(local, args.models)
    if not targets:
        raise RuntimeError("No local models discovered.")

    device = torch.device(args.device)
    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    dtype_bytes = 2
    total_mem = int(torch.cuda.get_device_properties(device).total_memory)
    budget_bytes = max(int(total_mem * float(args.budget_frac)), 1 << 30)
    max_bucket = max(cfg.BUCKETS)

    registry_path = Path(cfg.DATA_DIR) / "hybrid_checkpoint_registry.json"
    rows: list[dict[str, Any]] = []

    with gpu_experiment_lock(label="hybrid_checkpoint_runtime_lut", enabled=True, lock_path=args.gpu_lock_path or None):
        for model_id, snap in targets:
            _ensure_clean_gpu()
            cfg_json = _load_json(snap / "config.json")
            params = _infer_params(cfg_json)
            lut_name = cfg.checkpoint_lut_name(model_id)
            aliases = [model_id, model_id.split("/")[-1], lut_name]
            cfg.register_checkpoint_model(lut_name, aliases=aliases, **params)
            batch_size = _estimate_batch_size(
                q_heads=params["q_heads"],
                kv_heads=params["kv_heads"],
                head_dim=params["head_dim"],
                d_model=params["d_model"],
                dtype_bytes=dtype_bytes,
                max_bucket=max_bucket,
                batch_cap=int(args.batch_size_cap),
                budget_bytes=budget_bytes,
            )
            safe_max_len = _safe_max_pos(cfg_json, default=2048)
            requested_len = max(1024, int(args.max_num_batched_tokens) + 32)
            max_model_len = min(safe_max_len, requested_len)
            calib_buckets = _pick_calibration_buckets(max_model_len, reserve_new_tokens=8)
            trust_remote_code = _maybe_trust_remote_code(model_id)
            row: dict[str, Any] = {
                "model_id": model_id,
                "lut_name": lut_name,
                "snapshot": str(snap),
                "aliases": aliases,
                "q_heads": params["q_heads"],
                "kv_heads": params["kv_heads"],
                "head_dim": params["head_dim"],
                "d_model": params["d_model"],
                "base_batch_size": batch_size,
                "calibration_buckets": calib_buckets,
                "status": "pending",
            }
            print(f"[HybridLUT] start model={model_id} lut={lut_name}")
            try:
                _ensure_base_profile(
                    lut_name=lut_name,
                    params=params,
                    device=device,
                    dtype=dtype,
                    batch_size=batch_size,
                    warmup_iters=int(args.base_warmup_iters),
                    active_iters=int(args.base_active_iters),
                )
                paths = cfg.get_lut_paths(lut_name)
                base_raw_path = str(Path(cfg.DATA_DIR) / f"raw_profile_base_{lut_name}.json")
                base_gain_path = str(Path(cfg.DATA_DIR) / f"lut_gain_base_{lut_name}.json")
                base_penalty_path = str(Path(cfg.DATA_DIR) / f"lut_penalty_base_{lut_name}.json")
                _backup_if_missing(paths["raw"], base_raw_path)
                _backup_if_missing(paths["gain"], base_gain_path)
                _backup_if_missing(paths["penalty"], base_penalty_path)

                runtime = _runtime_calibration(
                    model_id=model_id,
                    snapshot=snap,
                    trust_remote_code=trust_remote_code,
                    buckets=calib_buckets,
                    max_model_len=max_model_len,
                    max_num_batched_tokens=int(args.max_num_batched_tokens),
                    gpu_memory_utilization=float(args.gpu_memory_utilization),
                    repeats=int(args.runtime_repeats),
                )
                sanity = _runtime_sanity_check(runtime)
                sanity_path = _runtime_sanity_path(lut_name)
                sanity_path.write_text(json.dumps(sanity, indent=2, ensure_ascii=False), encoding="utf-8")
                if not sanity["passed"]:
                    raise ValueError(f"runtime calibration failed sanity: {sanity['reasons']}")
                base_raw = _load_json(Path(base_raw_path))
                hybrid_raw = _build_hybrid_raw(base_raw=base_raw, runtime=runtime)
                Path(paths["raw"]).write_text(json.dumps(hybrid_raw, indent=2, ensure_ascii=False), encoding="utf-8")
                generate_lut_for_model(lut_name)
                runtime_meta_path = Path(cfg.DATA_DIR) / f"runtime_calibration_{lut_name}.json"
                runtime_meta_path.write_text(json.dumps(runtime, indent=2, ensure_ascii=False), encoding="utf-8")
                row["status"] = "ok"
                row["runtime_meta"] = str(runtime_meta_path)
                row["runtime_sanity"] = str(sanity_path)
                row["base_raw_profile"] = base_raw_path
                row["base_gain"] = base_gain_path
                row["base_penalty"] = base_penalty_path
                row["hybrid_raw_profile"] = paths["raw"]
                row["hybrid_gain"] = paths["gain"]
                row["hybrid_penalty"] = paths["penalty"]
            except Exception as exc:
                row["status"] = "failed"
                row["error"] = repr(exc)
                sanity_path = _runtime_sanity_path(lut_name)
                if sanity_path.exists():
                    row["runtime_sanity"] = str(sanity_path)
                print(f"[HybridLUT] failed model={model_id}: {exc!r}")
            finally:
                rows.append(row)
                _write_registry(registry_path, rows)
                torch.cuda.empty_cache()
                _ensure_clean_gpu()

    ok = sum(1 for r in rows if r["status"] == "ok")
    print(f"[HybridLUT] done ok={ok}/{len(rows)} registry={registry_path}")


if __name__ == "__main__":
    main()
