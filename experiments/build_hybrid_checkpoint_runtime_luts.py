from __future__ import annotations

import argparse
import gc
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path
from statistics import median
from typing import Any

from waveslice.vllm.bootstrap import bootstrap_vllm_runtime

bootstrap_vllm_runtime()
import torch

from experiments.lut_fingerprint import current_lut_fingerprint
from experiments.model_assets import _hf_hub_dir
from profiler import offline_profiler as offline_profiler_mod
from profiler.lut_generator import generate_lut_for_model
from profiler.offline_profiler import ModelProfiler
from tools.experiment_lock import gpu_experiment_lock
from waveslice.lut import config as cfg


def _discover_local_snapshots() -> list[tuple[str, Path]]:
    output = []
    for repo in sorted(_hf_hub_dir().glob("models--*")):
        for snapshot in sorted((repo / "snapshots").glob("*")):
            if snapshot.is_dir() and (snapshot / "config.json").exists():
                output.append((repo.name.removeprefix("models--").replace("--", "/"), snapshot))
                break
    return output


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _infer_params(config: dict[str, Any]) -> dict[str, int]:
    def first(*names: str) -> Any:
        return next((config.get(name) for name in names if config.get(name) is not None), None)

    q_heads = first("num_attention_heads", "n_head", "num_heads")
    d_model = first("hidden_size", "n_embd", "d_model", "model_dim", "n_embed")
    kv_heads = first("num_key_value_heads", "n_head_kv", "multi_query_group_num")
    kv_heads = (1 if config.get("multi_query") else q_heads) if kv_heads is None else kv_heads
    if None in (q_heads, kv_heads, d_model):
        raise ValueError("cannot infer q_heads/kv_heads/d_model from config")
    q_heads, kv_heads, d_model = int(q_heads), int(kv_heads), int(d_model)
    if d_model % q_heads:
        raise ValueError(f"d_model={d_model} not divisible by q_heads={q_heads}")
    return {
        "q_heads": q_heads,
        "kv_heads": kv_heads,
        "d_model": d_model,
        "head_dim": d_model // q_heads,
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
    elements = (
        q_heads * max_bucket * head_dim
        + 2 * kv_heads * max_bucket * head_dim
        + 2 * max_bucket * d_model
        + 2 * d_model * cfg.LORA_RANK
    )
    return max(1, min(batch_cap, budget_bytes // max(int(elements * dtype_bytes * 1.35), 1)))


def _selected_models(local: list[tuple[str, Path]], models_arg: str) -> list[tuple[str, Path]]:
    if models_arg.strip().lower() in {"all", "all-local"}:
        return local
    chosen = {value.strip() for value in models_arg.split(",") if value.strip()}
    selected = [
        (model, path)
        for model, path in local
        if model in chosen or cfg.checkpoint_lut_name(model) in chosen
    ]
    known = {value for model, _ in selected for value in (model, cfg.checkpoint_lut_name(model))}
    if missing := chosen - known:
        raise ValueError(f"unknown local models: {sorted(missing)}")
    return selected


def _maybe_trust_remote_code(model_id: str) -> bool:
    return any(
        value in model_id.lower() for value in ("baichuan", "deci", "falcon", "qwen", "mixtral")
    )


def _safe_max_pos(config: dict[str, Any], default: int = 2048) -> int:
    return int(
        next(
            (
                config.get(name)
                for name in (
                    "max_position_embeddings",
                    "n_positions",
                    "n_ctx",
                    "seq_length",
                    "max_sequence_length",
                )
                if config.get(name)
            ),
            default,
        )
    )


def _pick_calibration_buckets(max_model_len: int, *, reserve_new_tokens: int = 8) -> list[int]:
    limit = max(1, max_model_len - max(1, reserve_new_tokens))
    output = [value for value in (256, 512, 1024, 1536, 2048, 3072) if value <= limit] or [
        min(256, limit)
    ]
    return sorted(
        set([output[0], output[1], output[-2], output[-1]] if len(output) > 4 else output)
    )


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
    output = Path(cfg.DATA_DIR) / f"runtime_calibration_{cfg.checkpoint_lut_name(model_id)}.json"
    command = [sys.executable, "-m", "experiments.run_runtime_calibration"]
    for name, value in (
        ("model-ref", snapshot),
        ("snapshot", snapshot),
        ("max-model-len", max_model_len),
        ("max-num-batched-tokens", max_num_batched_tokens),
        ("gpu-memory-utilization", gpu_memory_utilization),
        ("repeats", repeats),
        ("buckets", ",".join(map(str, buckets))),
        ("out", output),
    ):
        command.extend([f"--{name}", str(value)])
    if trust_remote_code:
        command.append("--trust-remote-code")
    env = dict(os.environ)
    env.setdefault("VLLM_USE_V1", "1")
    subprocess.run(command, env=env, check=True)
    return _load_json(output)


def _runtime_sanity_path(lut_name: str) -> Path:
    return Path(cfg.DATA_DIR) / f"runtime_sanity_{lut_name}.json"


def _runtime_sanity_check(runtime: dict[str, Any]) -> dict[str, Any]:
    buckets = [int(value) for value in runtime.get("buckets") or []]
    solo_values = [(bucket, float(runtime["solo_us"][str(bucket)])) for bucket in buckets]
    reasons = [
        f"nonpositive_solo:{bucket}"
        for bucket, value in solo_values
        if not math.isfinite(value) or value <= 0
    ]
    reasons.extend(
        f"solo_non_monotonic:{left[0]}->{right[0]}"
        for left, right in zip(solo_values, solo_values[1:], strict=False)
        if right[1] < left[1] * 0.90
    )
    solo_by_bucket = dict(solo_values)
    ratios = []
    for short in buckets:
        for chunk, value in runtime["concurrent_short_us"][str(short)].items():
            ratio = float(value) / max(solo_by_bucket[short], 1e-9)
            ratios.append(ratio)
            if not 0.20 <= ratio <= 10.0:
                reasons.append(f"ratio_out_of_range:{short}:{chunk}:{ratio:.3f}")
    return {
        "passed": bool(buckets) and not reasons,
        "buckets": buckets,
        "solo_bucket_count": len(solo_values),
        "ratio_count": len(ratios),
        "ratio_min": min(ratios) if ratios else None,
        "ratio_median": median(ratios) if ratios else None,
        "ratio_max": max(ratios) if ratios else None,
        "ratio_lower": 0.20,
        "ratio_upper": 10.0,
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
    force: bool = False,
) -> None:
    paths = cfg.get_lut_paths(lut_name)
    if not force and all(os.path.exists(paths[key]) for key in ("raw", "gain", "penalty")):
        return
    cfg.register_checkpoint_model(lut_name, aliases=[lut_name], **params)
    old_batch = cfg.BATCH_SIZE
    offline_profiler_mod.WARMUP_ITERS, offline_profiler_mod.ACTIVE_ITERS = (
        max(1, warmup_iters),
        max(1, active_iters),
    )
    try:
        cfg.BATCH_SIZE = batch_size
        profiler = ModelProfiler(lut_name, device, dtype)
        profiler.run()
        generate_lut_for_model(lut_name)
    finally:
        if "profiler" in locals():
            del profiler
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        cfg.BATCH_SIZE = old_batch


def _backup_profile(src: str, dst: str, *, force: bool = False) -> None:
    if os.path.exists(src) and (force or not os.path.exists(dst)):
        shutil.copy2(src, dst)


def _build_hybrid_raw(*, base_raw: dict[str, Any], runtime: dict[str, Any]) -> dict[str, Any]:
    output = json.loads(json.dumps(base_raw))
    solo = {int(key): float(value) for key, value in runtime["solo_us"].items()}
    ratios = [
        value / float(base_raw["T_solo"][str(bucket)])
        for bucket, value in solo.items()
        if float(base_raw["T_solo"].get(str(bucket), 0)) > 0
    ]
    scale = float(median(ratios)) if ratios else 1.0
    output["T_solo"].update((str(bucket), value) for bucket, value in solo.items())
    for short, row in runtime["concurrent_short_us"].items():
        output.setdefault("T_conc", {}).setdefault(str(int(short)), {}).update(
            (str(int(chunk)), float(value)) for chunk, value in row.items()
        )
    for row in output.get("T_read_amp", {}).values():
        row.update((chunk, float(value) * scale) for chunk, value in list(row.items()))
    return output


def _write_registry(path: Path, rows: list[dict[str, Any]]) -> None:
    existing = _load_json(path).get("models", []) if path.exists() else []
    merged = {
        str(row["model_id"]): row
        for row in existing
        if isinstance(row, dict) and row.get("model_id")
    }
    merged.update((str(row["model_id"]), row) for row in rows if row.get("model_id"))
    path.write_text(
        json.dumps({"models": list(merged.values())}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build hybrid checkpoint-runtime LUTs for all local models."
    )
    for name, default in (
        ("models", "all-local"),
        ("device", "cuda:0"),
        ("dtype", "fp16"),
        ("gpu-lock-path", ""),
    ):
        parser.add_argument(
            "--" + name,
            default=default,
            **({"choices": ["fp16", "bf16"]} if name == "dtype" else {}),
        )
    for name, default in (
        ("base-warmup-iters", 3),
        ("base-active-iters", 8),
        ("runtime-repeats", 3),
        ("batch-size-cap", 128),
        ("max-num-batched-tokens", 1536),
    ):
        parser.add_argument("--" + name, type=int, default=default)
    for name, default in (("budget-frac", 0.18), ("gpu-memory-utilization", 0.80)):
        parser.add_argument("--" + name, type=float, default=default)
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to build hybrid checkpoint-runtime LUTs.")
    targets = _selected_models(_discover_local_snapshots(), args.models)
    if not targets:
        raise RuntimeError("No local models discovered.")
    device, dtype = (
        torch.device(args.device),
        torch.float16 if args.dtype == "fp16" else torch.bfloat16,
    )
    budget = max(
        int(torch.cuda.get_device_properties(device).total_memory * args.budget_frac), 1 << 30
    )
    registry = Path(cfg.DATA_DIR) / "hybrid_checkpoint_registry.json"
    rows, fingerprint = [], current_lut_fingerprint()
    with gpu_experiment_lock(
        label="hybrid_checkpoint_runtime_lut", enabled=True, lock_path=args.gpu_lock_path or None
    ):
        for model_id, snapshot in targets:
            config, lut_name = (
                _load_json(snapshot / "config.json"),
                cfg.checkpoint_lut_name(model_id),
            )
            params = _infer_params(config)
            aliases = [model_id, model_id.split("/")[-1], lut_name]
            cfg.register_checkpoint_model(lut_name, aliases=aliases, **params)
            batch_size = _estimate_batch_size(
                **params,
                dtype_bytes=2,
                max_bucket=max(cfg.BUCKETS),
                batch_cap=args.batch_size_cap,
                budget_bytes=budget,
            )
            max_model_len = min(_safe_max_pos(config), max(1024, args.max_num_batched_tokens + 32))
            buckets = _pick_calibration_buckets(max_model_len)
            row = dict(
                model_id=model_id,
                lut_name=lut_name,
                snapshot=str(snapshot),
                aliases=aliases,
                **params,
                base_batch_size=batch_size,
                calibration_buckets=buckets,
                hardware_fingerprint=fingerprint,
                status="pending",
            )
            print(f"[HybridLUT] start model={model_id} lut={lut_name}")
            try:
                runtime = _runtime_calibration(
                    model_id=model_id,
                    snapshot=snapshot,
                    trust_remote_code=_maybe_trust_remote_code(model_id),
                    buckets=buckets,
                    max_model_len=max_model_len,
                    max_num_batched_tokens=args.max_num_batched_tokens,
                    gpu_memory_utilization=args.gpu_memory_utilization,
                    repeats=args.runtime_repeats,
                )
                runtime["hardware_fingerprint"] = fingerprint
                sanity = _runtime_sanity_check(runtime)
                sanity["hardware_fingerprint"] = fingerprint
                sanity_path = _runtime_sanity_path(lut_name)
                sanity_path.write_text(
                    json.dumps(sanity, indent=2, ensure_ascii=False), encoding="utf-8"
                )
                if not sanity["passed"]:
                    raise ValueError(f"runtime calibration failed sanity: {sanity['reasons']}")
                _ensure_base_profile(
                    lut_name=lut_name,
                    params=params,
                    device=device,
                    dtype=dtype,
                    batch_size=batch_size,
                    warmup_iters=args.base_warmup_iters,
                    active_iters=args.base_active_iters,
                    force=args.force,
                )
                paths = cfg.get_lut_paths(lut_name)
                backups = {
                    kind: str(Path(cfg.DATA_DIR) / f"{prefix}_base_{lut_name}.json")
                    for kind, prefix in (
                        ("raw", "raw_profile"),
                        ("gain", "lut_gain"),
                        ("penalty", "lut_penalty"),
                    )
                }
                for kind, path in backups.items():
                    _backup_profile(paths[kind], path, force=args.force)
                gc.collect()
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                hybrid = _build_hybrid_raw(
                    base_raw=_load_json(Path(backups["raw"])), runtime=runtime
                )
                Path(paths["raw"]).write_text(
                    json.dumps(hybrid, indent=2, ensure_ascii=False), encoding="utf-8"
                )
                generate_lut_for_model(lut_name)
                runtime_path = Path(cfg.DATA_DIR) / f"runtime_calibration_{lut_name}.json"
                runtime_path.write_text(
                    json.dumps(runtime, indent=2, ensure_ascii=False), encoding="utf-8"
                )
                row.update(
                    status="ok",
                    runtime_meta=str(runtime_path),
                    runtime_sanity=str(sanity_path),
                    base_raw_profile=backups["raw"],
                    base_gain=backups["gain"],
                    base_penalty=backups["penalty"],
                    hybrid_raw_profile=paths["raw"],
                    hybrid_gain=paths["gain"],
                    hybrid_penalty=paths["penalty"],
                )
            except Exception as error:
                row.update(status="failed", error=repr(error))
                if _runtime_sanity_path(lut_name).exists():
                    row["runtime_sanity"] = str(_runtime_sanity_path(lut_name))
                print(f"[HybridLUT] failed model={model_id}: {error!r}")
            finally:
                rows.append(row)
                _write_registry(registry, rows)
                torch.cuda.empty_cache()
    ok = sum(row["status"] == "ok" for row in rows)
    print(f"[HybridLUT] done ok={ok}/{len(rows)} registry={registry}")
    if ok != len(rows):
        raise RuntimeError(
            f"Hybrid LUT rebuild failed for {len(rows) - ok}/{len(rows)} models; see {registry}"
        )


if __name__ == "__main__":
    main()
