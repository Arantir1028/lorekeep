from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from config import hw_config as cfg
from profiler.lut_generator import generate_lut_for_model
from profiler import offline_profiler as offline_profiler_mod
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
        raise ValueError(f"cannot infer heads/d_model from config keys: {sorted(config_json.keys())[:20]}")
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
    # Safety multiplier covers allocator overhead and temporary tensors.
    per_sample_bytes = int(per_sample_elems * dtype_bytes * 1.35)
    if per_sample_bytes <= 0:
        return 1
    return max(1, min(batch_cap, budget_bytes // per_sample_bytes))


def _selected_models(local: list[tuple[str, Path]], models_arg: str) -> list[tuple[str, Path]]:
    if models_arg.strip().lower() in {"all", "all-local"}:
        return local
    chosen = {m.strip() for m in models_arg.split(",") if m.strip()}
    selected = [(mid, snap) for mid, snap in local if mid in chosen or cfg.checkpoint_lut_name(mid) in chosen]
    missing = chosen - {mid for mid, _ in selected} - {cfg.checkpoint_lut_name(mid) for mid, _ in selected}
    if missing:
        raise ValueError(f"unknown local models: {sorted(missing)}")
    return selected


def _write_registry(rows: list[dict[str, Any]]) -> None:
    payload = {
        "models": rows,
    }
    Path(cfg.CHECKPOINT_REGISTRY_PATH).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build checkpoint-level LUTs for all local Hugging Face models.")
    parser.add_argument("--models", default="all-local", help="Comma-separated local model_ids or LUT names, or all-local.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=["fp16", "bf16"], default="fp16")
    parser.add_argument("--warmup-iters", type=int, default=10)
    parser.add_argument("--active-iters", type=int, default=50)
    parser.add_argument("--budget-frac", type=float, default=0.18, help="Fraction of total GPU memory reserved for profiler tensor pools.")
    parser.add_argument("--batch-size-cap", type=int, default=128)
    parser.add_argument("--gpu-lock-path", default="")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to build checkpoint-level LUTs.")

    local = _discover_local_snapshots()
    targets = _selected_models(local, args.models)
    if not targets:
        raise RuntimeError("No local models discovered for checkpoint-level LUT generation.")

    device = torch.device(args.device)
    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    dtype_bytes = 2
    offline_profiler_mod.WARMUP_ITERS = max(1, int(args.warmup_iters))
    offline_profiler_mod.ACTIVE_ITERS = max(1, int(args.active_iters))
    total_mem = int(torch.cuda.get_device_properties(device).total_memory)
    budget_bytes = max(int(total_mem * float(args.budget_frac)), 1 << 30)
    max_bucket = max(cfg.BUCKETS)

    rows: list[dict[str, Any]] = []
    base_batch_size = int(cfg.BATCH_SIZE)

    with gpu_experiment_lock(
        label="checkpoint_lut_build",
        enabled=True,
        lock_path=args.gpu_lock_path or None,
    ):
        for model_id, snap in targets:
            config_json = _load_json(snap / "config.json")
            params = _infer_params(config_json)
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
            row = {
                "model_id": model_id,
                "lut_name": lut_name,
                "snapshot": str(snap),
                "q_heads": params["q_heads"],
                "kv_heads": params["kv_heads"],
                "head_dim": params["head_dim"],
                "d_model": params["d_model"],
                "aliases": aliases,
                "batch_size": batch_size,
                "status": "pending",
            }
            print(f"[CheckpointLUT] start model={model_id} lut={lut_name} batch_size={batch_size}")
            cfg.BATCH_SIZE = int(batch_size)
            try:
                profiler = ModelProfiler(lut_name, device, dtype)
                profiler.run()
                generate_lut_for_model(lut_name)
                row["status"] = "ok"
            except Exception as exc:
                row["status"] = "failed"
                row["error"] = repr(exc)
                print(f"[CheckpointLUT] failed model={model_id}: {exc!r}")
            finally:
                cfg.BATCH_SIZE = base_batch_size
                rows.append(row)
                _write_registry(rows)
                torch.cuda.empty_cache()

    ok = sum(1 for r in rows if r["status"] == "ok")
    print(f"[CheckpointLUT] done ok={ok}/{len(rows)} registry={cfg.CHECKPOINT_REGISTRY_PATH}")


if __name__ == "__main__":
    main()
