"""Profile objective-independent metadata for real local models.

This script is intentionally separated from the current scheduler objective.
It collects reusable metadata that can support future LUT construction,
objective redesign, and dataset-driven workload studies.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


@dataclass(frozen=True)
class ModelSpec:
    key: str
    model_id: str
    family_hint: str


DEFAULT_MODELS: list[ModelSpec] = [
    ModelSpec("mistral-7b-v0.1", "mistralai/Mistral-7B-v0.1", "mistral"),
    ModelSpec("mistral-7b-instruct-v0.2", "mistralai/Mistral-7B-Instruct-v0.2", "mistral"),
    ModelSpec("zephyr-7b-beta", "HuggingFaceH4/zephyr-7b-beta", "mistral"),
    ModelSpec("openchat-3.5-0106", "openchat/openchat-3.5-0106", "mistral"),
    ModelSpec("gemma-7b-it", "google/gemma-7b-it", "gemma"),
    ModelSpec("decilm-7b", "Deci/DeciLM-7B", "mistral"),
    ModelSpec("phi-2", "microsoft/phi-2", "phi"),
    ModelSpec("baichuan2-7b-chat", "baichuan-inc/Baichuan2-7B-Chat", "baichuan"),
]


def _safe_key(s: str) -> str:
    return s.replace("/", "--")


def _resolve_local_snapshot(model_id: str) -> Optional[str]:
    hub_dir = Path.home() / ".cache" / "huggingface" / "hub"
    repo_name = "models--" + model_id.replace("/", "--")
    snapshots_dir = hub_dir / repo_name / "snapshots"
    if not snapshots_dir.exists():
        return None
    dirs = [p for p in snapshots_dir.iterdir() if p.is_dir()]
    if not dirs:
        return None
    dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    for snap in dirs:
        if (snap / "config.json").exists():
            return str(snap)
    return None


def _load_json(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _infer_heads(cfg: dict[str, Any]) -> tuple[Optional[int], Optional[int], Optional[int], str]:
    q_heads = cfg.get("num_attention_heads") or cfg.get("n_head") or cfg.get("num_heads")
    kv_heads = (
        cfg.get("num_key_value_heads")
        or cfg.get("n_head_kv")
        or cfg.get("multi_query_group_num")
        or q_heads
    )
    d_model = (
        cfg.get("hidden_size")
        or cfg.get("n_embd")
        or cfg.get("d_model")
        or cfg.get("model_dim")
    )
    head_dim = None
    if q_heads and d_model:
        try:
            head_dim = int(d_model) // int(q_heads)
        except Exception:
            head_dim = None
    attn_type = "unknown"
    if q_heads and kv_heads:
        if int(kv_heads) == 1:
            attn_type = "MQA"
        elif int(kv_heads) == int(q_heads):
            attn_type = "MHA"
        else:
            attn_type = "GQA"
    return (
        int(q_heads) if q_heads is not None else None,
        int(kv_heads) if kv_heads is not None else None,
        int(head_dim) if head_dim is not None else None,
        attn_type,
    )


def _pick_tokenizer_class(cfg: dict[str, Any], tokenizer_cfg: dict[str, Any]) -> Optional[str]:
    for src in (tokenizer_cfg, cfg):
        for key in ("tokenizer_class", "auto_map"):
            if key in src:
                value = src[key]
                if isinstance(value, str):
                    return value
                if isinstance(value, dict):
                    return json.dumps(value, ensure_ascii=False)
    return None


def _lookup_existing_lut_artifacts(model_id: str) -> dict[str, Any]:
    base = Path("data/lut_tables")
    suffixes = {
        "raw_profile": f"raw_profile_{model_id}.json",
        "lut_gain": f"lut_gain_{model_id}.json",
        "lut_penalty": f"lut_penalty_{model_id}.json",
    }
    out: dict[str, Any] = {}
    for k, fname in suffixes.items():
        path = base / fname
        out[k] = str(path.resolve()) if path.exists() else None
    return out


def _summarize_raw_profile(path: Optional[str]) -> Optional[dict[str, Any]]:
    if not path:
        return None
    raw = _load_json(path)
    t_solo = {int(k): float(v) for k, v in raw.get("T_solo", {}).items()}
    t_conc = raw.get("T_conc", {})
    t_read_amp = raw.get("T_read_amp", {})
    return {
        "bucket_count": len(t_solo),
        "bucket_keys": sorted(t_solo.keys()),
        "t_solo_us_min": min(t_solo.values()) if t_solo else None,
        "t_solo_us_max": max(t_solo.values()) if t_solo else None,
        "t_conc_rows": len(t_conc),
        "t_read_amp_rows": len(t_read_amp),
    }


def profile_model_metadata(spec: ModelSpec) -> dict[str, Any]:
    snapshot = _resolve_local_snapshot(spec.model_id)
    row: dict[str, Any] = {
        "key": spec.key,
        "model_id": spec.model_id,
        "family_hint": spec.family_hint,
        "snapshot": snapshot,
        "available": snapshot is not None,
    }
    if snapshot is None:
        return row

    cfg = _load_json(os.path.join(snapshot, "config.json"))
    tokenizer_cfg_path = os.path.join(snapshot, "tokenizer_config.json")
    tokenizer_cfg = _load_json(tokenizer_cfg_path) if os.path.exists(tokenizer_cfg_path) else {}
    q_heads, kv_heads, head_dim, attn_type = _infer_heads(cfg)

    row.update(
        {
            "architectures": cfg.get("architectures"),
            "model_type": cfg.get("model_type"),
            "torch_dtype": cfg.get("torch_dtype"),
            "hidden_size": cfg.get("hidden_size") or cfg.get("n_embd") or cfg.get("d_model"),
            "intermediate_size": cfg.get("intermediate_size") or cfg.get("n_inner"),
            "num_hidden_layers": cfg.get("num_hidden_layers") or cfg.get("n_layer"),
            "num_attention_heads": q_heads,
            "num_key_value_heads": kv_heads,
            "head_dim": head_dim,
            "attn_type": attn_type,
            "vocab_size": cfg.get("vocab_size"),
            "max_position_embeddings": (
                cfg.get("max_position_embeddings")
                or cfg.get("seq_length")
                or cfg.get("max_sequence_length")
                or cfg.get("model_max_length")
            ),
            "sliding_window": cfg.get("sliding_window"),
            "rope_theta": cfg.get("rope_theta"),
            "bos_token_id": cfg.get("bos_token_id"),
            "eos_token_id": cfg.get("eos_token_id"),
            "tokenizer_class": _pick_tokenizer_class(cfg, tokenizer_cfg),
        }
    )

    lut_artifacts = _lookup_existing_lut_artifacts(spec.model_id.split("/")[-1])
    # Also try legacy canonical family names.
    if not lut_artifacts["raw_profile"]:
        family_map = {
            "mistral": "Mistral-7B-v0.1",
            "gemma": "Gemma-7B",
            "phi": "Mistral-7B-v0.1",
            "baichuan": "Mistral-7B-v0.1",
        }
        family_name = family_map.get(spec.family_hint)
        if family_name:
            lut_artifacts = _lookup_existing_lut_artifacts(family_name)
            row["lut_family_source"] = family_name
    row["lut_artifacts"] = lut_artifacts
    row["raw_profile_summary"] = _summarize_raw_profile(lut_artifacts.get("raw_profile"))
    return row


def _extract_longbench_prompt(example: dict[str, Any]) -> Optional[str]:
    pieces: list[str] = []
    for key in ("context", "input", "question", "instruction"):
        value = example.get(key)
        if isinstance(value, str) and value.strip():
            pieces.append(value.strip())
    if not pieces:
        return None
    return "\n\n".join(pieces)


def _extract_ultrachat_prompt(example: dict[str, Any]) -> Optional[str]:
    messages = example.get("messages")
    if isinstance(messages, list):
        user_turns = []
        for turn in messages:
            if not isinstance(turn, dict):
                continue
            role = str(turn.get("role", "")).lower()
            if role == "user":
                content = turn.get("content")
                if isinstance(content, str) and content.strip():
                    user_turns.append(content.strip())
        if user_turns:
            return "\n\n".join(user_turns)
    prompt = example.get("prompt")
    if isinstance(prompt, str) and prompt.strip():
        return prompt.strip()
    data = example.get("data")
    if isinstance(data, list):
        user_turns = []
        for turn in data:
            if not isinstance(turn, dict):
                continue
            role = str(turn.get("role", "")).lower()
            if role == "user":
                content = turn.get("content")
                if isinstance(content, str) and content.strip():
                    user_turns.append(content.strip())
        if user_turns:
            return "\n\n".join(user_turns)
    return None


def _percentile(sorted_vals: list[int], p: float) -> Optional[float]:
    if not sorted_vals:
        return None
    if p <= 0:
        return float(sorted_vals[0])
    if p >= 100:
        return float(sorted_vals[-1])
    k = (len(sorted_vals) - 1) * (p / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(sorted_vals) - 1)
    if lo == hi:
        return float(sorted_vals[lo])
    frac = k - lo
    return float(sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * frac)


def _token_stats(lengths: list[int]) -> dict[str, Any]:
    ordered = sorted(int(x) for x in lengths if int(x) > 0)
    if not ordered:
        return {
            "count": 0,
            "mean": None,
            "p50": None,
            "p90": None,
            "p95": None,
            "p99": None,
            "min": None,
            "max": None,
        }
    mean = sum(ordered) / len(ordered)
    return {
        "count": len(ordered),
        "mean": mean,
        "p50": _percentile(ordered, 50),
        "p90": _percentile(ordered, 90),
        "p95": _percentile(ordered, 95),
        "p99": _percentile(ordered, 99),
        "min": ordered[0],
        "max": ordered[-1],
    }


def profile_dataset_metadata(
    *,
    model_rows: list[dict[str, Any]],
    dataset_name: str,
    sample_count: int,
) -> dict[str, Any]:
    try:
        from datasets import load_dataset
        from transformers import AutoTokenizer
    except Exception as exc:
        return {
            "dataset": dataset_name,
            "status": "unavailable",
            "error": f"missing dependency: {exc}",
        }

    prompts: list[str] = []
    dataset_extra: dict[str, Any] = {}

    if dataset_name == "LongBench":
        # Use a small representative mix of common LongBench English tasks so the
        # resulting token-length metadata reflects long-context serving workloads
        # rather than a single task's prompt format.
        dataset_id = "Xnhyacinth/LongBench"
        longbench_configs = [
            "qmsum",
            "gov_report",
            "multifieldqa_en",
            "hotpotqa",
        ]
        per_config = max(1, int(math.ceil(sample_count / float(len(longbench_configs)))))
        config_samples: dict[str, int] = {}
        for config_name in longbench_configs:
            ds = load_dataset(dataset_id, config_name, split="test")
            taken = 0
            for example in ds:
                prompt = _extract_longbench_prompt(example)
                if prompt:
                    prompts.append(prompt)
                    taken += 1
                if taken >= per_config or len(prompts) >= sample_count:
                    break
            config_samples[config_name] = taken
            if len(prompts) >= sample_count:
                break
        dataset_extra = {
            "configs": longbench_configs,
            "config_sample_count": config_samples,
        }
    elif dataset_name == "C4":
        ds = load_dataset("allenai/c4", "en", split="train", streaming=True)
        extractor = lambda ex: ex.get("text") if isinstance(ex.get("text"), str) else None
        dataset_id = "allenai/c4/en"
        for example in ds:
            prompt = extractor(example)
            if prompt:
                prompts.append(prompt)
            if len(prompts) >= sample_count:
                break
    elif dataset_name == "UltraChat200k":
        ds = load_dataset(
            "HuggingFaceH4/ultrachat_200k",
            split="train_sft",
            streaming=True,
        )
        extractor = _extract_ultrachat_prompt
        dataset_id = "HuggingFaceH4/ultrachat_200k"
        for example in ds:
            prompt = extractor(example)
            if prompt:
                prompts.append(prompt)
            if len(prompts) >= sample_count:
                break
    else:
        raise ValueError(f"Unsupported dataset alias: {dataset_name}")

    dataset_row: dict[str, Any] = {
        "dataset": dataset_name,
        "dataset_id": dataset_id,
        "sample_count": len(prompts),
        "models": [],
    }
    dataset_row.update(dataset_extra)
    for model_row in model_rows:
        snapshot = model_row.get("snapshot")
        if not snapshot:
            dataset_row["models"].append(
                {"model_id": model_row["model_id"], "available": False}
            )
            continue
        try:
            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    snapshot,
                    trust_remote_code=False,
                )
                trust_remote_code = False
            except Exception:
                tokenizer = AutoTokenizer.from_pretrained(
                    snapshot,
                    trust_remote_code=True,
                )
                trust_remote_code = True
            lengths = [
                len(tokenizer(prompt, add_special_tokens=True).input_ids)
                for prompt in prompts
            ]
            dataset_row["models"].append(
                {
                    "model_id": model_row["model_id"],
                    "snapshot": snapshot,
                    "trust_remote_code": trust_remote_code,
                    "token_length_stats": _token_stats(lengths),
                }
            )
        except Exception as exc:
            dataset_row["models"].append(
                {
                    "model_id": model_row["model_id"],
                    "snapshot": snapshot,
                    "available": False,
                    "error": str(exc),
                }
            )
    return dataset_row


def main() -> int:
    parser = argparse.ArgumentParser(description="Profile reusable metadata for local real models.")
    parser.add_argument("--out-json", default="results/real_model_metadata.json")
    parser.add_argument(
        "--datasets",
        default="",
        help="Optional comma-separated dataset aliases: LongBench,UltraChat200k,C4",
    )
    parser.add_argument("--sample-count", type=int, default=256)
    args = parser.parse_args()

    rows = [profile_model_metadata(spec) for spec in DEFAULT_MODELS]
    dataset_rows: list[dict[str, Any]] = []
    if args.datasets.strip():
        aliases = [x.strip() for x in args.datasets.split(",") if x.strip()]
        for alias in aliases:
            try:
                dataset_rows.append(
                    profile_dataset_metadata(
                        model_rows=rows,
                        dataset_name=alias,
                        sample_count=max(1, int(args.sample_count)),
                    )
                )
            except Exception as exc:
                dataset_rows.append(
                    {
                        "dataset": alias,
                        "status": "failed",
                        "error": str(exc),
                    }
                )
    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump({"models": rows, "datasets": dataset_rows}, f, ensure_ascii=False, indent=2)
    print(f"[Saved] {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
