"""Build dataset-driven request JSON files for Wave-Slice evaluation.

The goal is to reuse the existing repeated evaluation harness while swapping
the synthetic prompts for prompts sampled from open datasets.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from typing import Any, Optional

from config.experiment_catalog import (
    DEFAULT_DATASET_SOURCES,
    DEFAULT_LONG_BENCH_CONFIGS,
)


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
            if str(turn.get("role", "")).lower() != "user":
                continue
            content = turn.get("content")
            if isinstance(content, str) and content.strip():
                user_turns.append(content.strip())
        if user_turns:
            return "\n\n".join(user_turns)
    return None


def _pick_by_quantile(items: list[dict[str, Any]], q: float) -> dict[str, Any]:
    if not items:
        raise ValueError("cannot pick from empty list")
    ordered = sorted(items, key=lambda x: x["tokens"])
    idx = int(round((len(ordered) - 1) * q))
    idx = max(0, min(idx, len(ordered) - 1))
    return ordered[idx]


def _pick_many_by_quantiles(
    items: list[dict[str, Any]],
    quantiles: list[float],
) -> list[dict[str, Any]]:
    ordered = sorted(items, key=lambda x: x["tokens"])
    if not ordered:
        return []
    chosen: list[dict[str, Any]] = []
    used: set[int] = set()
    for q in quantiles:
        idx = int(round((len(ordered) - 1) * q))
        idx = max(0, min(idx, len(ordered) - 1))
        if idx in used:
            left = idx - 1
            right = idx + 1
            picked = None
            while left >= 0 or right < len(ordered):
                if left >= 0 and left not in used:
                    picked = left
                    break
                if right < len(ordered) and right not in used:
                    picked = right
                    break
                left -= 1
                right += 1
            idx = picked if picked is not None else idx
        used.add(idx)
        chosen.append(ordered[idx])
    return chosen


def _assign_poisson_arrivals(
    items: list[dict[str, Any]],
    *,
    rate_per_s: float,
    seed: int,
) -> list[dict[str, Any]]:
    if rate_per_s <= 0:
        raise ValueError("rate_per_s must be > 0 for poisson arrivals")
    rng = random.Random(seed)
    cur = 0.0
    assigned: list[dict[str, Any]] = []
    for item in items:
        cur += rng.expovariate(rate_per_s)
        enriched = dict(item)
        enriched["arrival_offset_s"] = round(cur, 6)
        assigned.append(enriched)
    return assigned


def _shuffle_items(items: list[dict[str, Any]], rng: random.Random) -> list[dict[str, Any]]:
    copied = list(items)
    rng.shuffle(copied)
    return copied


def _mixed_arrival_order(
    shorts: list[dict[str, Any]],
    longs: list[dict[str, Any]],
    *,
    seed: int,
    early_short_frac: float,
    post_long_short_bias: float,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    short_pool = _shuffle_items(shorts, rng)
    long_pool = _shuffle_items(longs, rng)
    if not short_pool:
        return long_pool
    if not long_pool:
        return short_pool

    early_short_frac = max(0.0, min(0.95, float(early_short_frac)))
    post_long_short_bias = max(0.0, min(1.0, float(post_long_short_bias)))

    # Keep a small number of shorts ahead of the first long, then force a long
    # to arrive early so later shorts can benefit from Phase-I.
    max_prefix = max(0, len(short_pool) - 1)
    prefix_n = min(max_prefix, int(round(len(short_pool) * early_short_frac)))
    ordered: list[dict[str, Any]] = []
    ordered.extend(short_pool[:prefix_n])
    short_pool = short_pool[prefix_n:]

    ordered.append(long_pool.pop(0))

    while short_pool or long_pool:
        if short_pool and long_pool:
            pick_short = rng.random() < post_long_short_bias
            if pick_short:
                ordered.append(short_pool.pop(0))
            else:
                ordered.append(long_pool.pop(0))
        elif short_pool:
            ordered.append(short_pool.pop(0))
        else:
            ordered.append(long_pool.pop(0))
    return ordered


def _arrival_order(
    items: list[dict[str, Any]],
    *,
    seed: int,
    layout: str,
    early_short_frac: float,
    post_long_short_bias: float,
) -> list[dict[str, Any]]:
    if layout == "grouped":
        return list(items)
    shorts = [dict(item) for item in items if bool(item.get("is_short"))]
    longs = [dict(item) for item in items if not bool(item.get("is_short"))]
    if layout == "mixed":
        rng = random.Random(seed)
        shuffled = list(items)
        rng.shuffle(shuffled)
        return shuffled
    if layout == "beneficiary_rich":
        return _mixed_arrival_order(
            shorts,
            longs,
            seed=seed,
            early_short_frac=early_short_frac,
            post_long_short_bias=post_long_short_bias,
        )
    raise ValueError(f"unknown arrival layout: {layout}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build dataset request json files.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--out-prefix", required=True)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--max-prompt-tokens", type=int, default=3072)
    parser.add_argument("--sample-count", type=int, default=256)
    parser.add_argument("--phase1-short-count", type=int, default=24)
    parser.add_argument("--phase1-long-count", type=int, default=8)
    parser.add_argument("--phase2-short-count", type=int, default=24)
    parser.add_argument("--phase2-long-count", type=int, default=12)
    parser.add_argument(
        "--arrival-mode",
        choices=["burst", "poisson"],
        default="poisson",
        help="Whether requests arrive all at once or according to a Poisson process.",
    )
    parser.add_argument("--phase1-arrival-rate", type=float, default=6.0, help="Poisson arrival rate (req/s) for Phase-I requests.")
    parser.add_argument("--phase2-arrival-rate", type=float, default=6.0, help="Poisson arrival rate (req/s) for Phase-II requests.")
    parser.add_argument("--arrival-seed", type=int, default=7)
    parser.add_argument(
        "--phase1-arrival-layout",
        choices=["grouped", "mixed", "beneficiary_rich"],
        default="beneficiary_rich",
        help="How Phase-I request types are ordered before arrival timestamps are assigned.",
    )
    parser.add_argument(
        "--phase2-arrival-layout",
        choices=["grouped", "mixed", "beneficiary_rich"],
        default="beneficiary_rich",
        help="How Phase-II request types are ordered before arrival timestamps are assigned.",
    )
    parser.add_argument("--phase1-early-short-frac", type=float, default=0.25)
    parser.add_argument("--phase2-early-short-frac", type=float, default=0.20)
    parser.add_argument("--phase1-post-long-short-bias", type=float, default=0.70)
    parser.add_argument("--phase2-post-long-short-bias", type=float, default=0.60)
    parser.add_argument(
        "--datasets",
        default="ultrachat200k,longbench",
        help="Comma-separated dataset source keys.",
    )
    parser.add_argument(
        "--longbench-configs",
        default=",".join(DEFAULT_LONG_BENCH_CONFIGS),
        help="Comma-separated LongBench config names.",
    )
    args = parser.parse_args()

    from datasets import load_dataset
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=args.trust_remote_code,
    )

    dataset_keys = [k.strip().lower() for k in args.datasets.split(",") if k.strip()]
    selected_sources = [DEFAULT_DATASET_SOURCES[k] for k in dataset_keys if k in DEFAULT_DATASET_SOURCES]
    if len(selected_sources) != len(dataset_keys):
        missing = sorted(set(dataset_keys) - set(DEFAULT_DATASET_SOURCES))
        raise ValueError(f"unknown dataset source keys: {missing}")

    ultrachat_prompts: list[dict[str, Any]] = []
    if "ultrachat200k" in dataset_keys:
        source = DEFAULT_DATASET_SOURCES["ultrachat200k"]
        ds_uc = load_dataset(source.dataset_id, split=source.split, streaming=source.streaming)
        for ex in ds_uc:
            prompt = _extract_ultrachat_prompt(ex)
            if not prompt:
                continue
            tokens = len(tok(prompt, add_special_tokens=True).input_ids)
            if 8 <= tokens <= args.max_prompt_tokens:
                ultrachat_prompts.append({"prompt": prompt, "tokens": tokens, "source": source.key})
            if len(ultrachat_prompts) >= args.sample_count:
                break

    longbench_prompts: list[dict[str, Any]] = []
    if "longbench" in dataset_keys:
        source = DEFAULT_DATASET_SOURCES["longbench"]
        longbench_configs = [c.strip() for c in args.longbench_configs.split(",") if c.strip()]
        per_config = max(1, int(math.ceil(args.sample_count / float(len(longbench_configs)))))
        for cfg in longbench_configs:
            ds_lb = load_dataset(source.dataset_id, cfg, split=source.split)
            taken = 0
            for ex in ds_lb:
                prompt = _extract_longbench_prompt(ex)
                if not prompt:
                    continue
                tokens = len(tok(prompt, add_special_tokens=True).input_ids)
                if 8 <= tokens <= args.max_prompt_tokens:
                    longbench_prompts.append({"prompt": prompt, "tokens": tokens, "config": cfg, "source": source.key})
                    taken += 1
                if taken >= per_config or len(longbench_prompts) >= args.sample_count:
                    break
            if len(longbench_prompts) >= args.sample_count:
                break

    if len(ultrachat_prompts) < 4 or len(longbench_prompts) < 4:
        raise RuntimeError("insufficient dataset prompts collected")

    phase1_short_count = max(2, int(args.phase1_short_count))
    phase1_long_count = max(1, int(args.phase1_long_count))
    phase2_short_count = max(2, int(args.phase2_short_count))
    phase2_long_count = max(2, int(args.phase2_long_count))

    short_anchor_a = _pick_by_quantile(ultrachat_prompts, 0.15)
    short_anchor_b = _pick_by_quantile(ultrachat_prompts, 0.55)
    long_anchor_a = _pick_by_quantile(longbench_prompts, 0.50)
    long_anchor_b = _pick_by_quantile(longbench_prompts, 0.90)

    phase1_short_qs = [0.10 + (0.55 * i / max(1, phase1_short_count - 1)) for i in range(phase1_short_count)]
    phase1_long_qs = [0.45 + (0.45 * i / max(1, phase1_long_count - 1)) for i in range(phase1_long_count)]
    phase2_short_qs = [0.10 + (0.60 * i / max(1, phase2_short_count - 1)) for i in range(phase2_short_count)]
    phase2_long_qs = [0.45 + (0.50 * i / max(1, phase2_long_count - 1)) for i in range(phase2_long_count)]

    phase1_shorts = _pick_many_by_quantiles(ultrachat_prompts, phase1_short_qs)
    phase1_longs = _pick_many_by_quantiles(longbench_prompts, phase1_long_qs)
    phase2_shorts = _pick_many_by_quantiles(ultrachat_prompts, phase2_short_qs)
    phase2_longs = _pick_many_by_quantiles(longbench_prompts, phase2_long_qs)

    reqs = [
        {"req_id": "short_a", "prompt": short_anchor_a["prompt"], "is_short": True, "source": "UltraChat200k", "tokens": short_anchor_a["tokens"]},
        {"req_id": "short_b", "prompt": short_anchor_b["prompt"], "is_short": True, "source": "UltraChat200k", "tokens": short_anchor_b["tokens"]},
    ]
    for i, item in enumerate(phase1_shorts):
        reqs.append(
            {
                "req_id": f"short_{i:02d}",
                "prompt": item["prompt"],
                "is_short": True,
                "source": "UltraChat200k",
                "tokens": item["tokens"],
            }
        )
    reqs.append({"req_id": "long_b", "prompt": long_anchor_b["prompt"], "is_short": False, "source": "LongBench", "tokens": long_anchor_b["tokens"]})
    for i, item in enumerate(phase1_longs):
        reqs.append(
            {
                "req_id": f"long_{i:02d}",
                "prompt": item["prompt"],
                "is_short": False,
                "source": "LongBench",
                "tokens": item["tokens"],
            }
        )

    lora_reqs = [
        {"req_id": "short_a", "prompt": short_anchor_a["prompt"], "is_short": True, "lora_tag": "A", "source": "UltraChat200k", "tokens": short_anchor_a["tokens"]},
        {"req_id": "mid_b", "prompt": short_anchor_b["prompt"], "is_short": True, "lora_tag": "B", "source": "UltraChat200k", "tokens": short_anchor_b["tokens"]},
        {"req_id": "long_a", "prompt": long_anchor_a["prompt"], "is_short": False, "lora_tag": "A", "source": "LongBench", "tokens": long_anchor_a["tokens"]},
        {"req_id": "long_b", "prompt": long_anchor_b["prompt"], "is_short": False, "lora_tag": "B", "source": "LongBench", "tokens": long_anchor_b["tokens"]},
    ]
    for i, item in enumerate(phase2_shorts):
        lora_reqs.append(
            {
                "req_id": f"short_extra_{i:02d}",
                "prompt": item["prompt"],
                "is_short": True,
                "lora_tag": "A" if i % 2 == 0 else "B",
                "source": "UltraChat200k",
                "tokens": item["tokens"],
            }
        )
    for i, item in enumerate(phase2_longs):
        lora_reqs.append(
            {
                "req_id": f"long_extra_{i:02d}",
                "prompt": item["prompt"],
                "is_short": False,
                "lora_tag": "A" if i % 2 == 0 else "B",
                "source": "LongBench",
                "tokens": item["tokens"],
            }
        )

    reqs = _arrival_order(
        reqs,
        seed=int(args.arrival_seed) + 17,
        layout=str(args.phase1_arrival_layout),
        early_short_frac=float(args.phase1_early_short_frac),
        post_long_short_bias=float(args.phase1_post_long_short_bias),
    )
    lora_reqs = _arrival_order(
        lora_reqs,
        seed=int(args.arrival_seed) + 1017,
        layout=str(args.phase2_arrival_layout),
        early_short_frac=float(args.phase2_early_short_frac),
        post_long_short_bias=float(args.phase2_post_long_short_bias),
    )

    if args.arrival_mode == "poisson":
        reqs = _assign_poisson_arrivals(
            reqs,
            rate_per_s=float(args.phase1_arrival_rate),
            seed=int(args.arrival_seed),
        )
        lora_reqs = _assign_poisson_arrivals(
            lora_reqs,
            rate_per_s=float(args.phase2_arrival_rate),
            seed=int(args.arrival_seed) + 1009,
        )
    else:
        reqs = [dict(item, arrival_offset_s=0.0) for item in reqs]
        lora_reqs = [dict(item, arrival_offset_s=0.0) for item in lora_reqs]

    os.makedirs(os.path.dirname(args.out_prefix) or ".", exist_ok=True)
    req_path = f"{args.out_prefix}_requests.json"
    lora_req_path = f"{args.out_prefix}_lora_requests.json"
    meta_path = f"{args.out_prefix}_meta.json"
    with open(req_path, "w", encoding="utf-8") as f:
        json.dump(reqs, f, ensure_ascii=False, indent=2)
    with open(lora_req_path, "w", encoding="utf-8") as f:
        json.dump(lora_reqs, f, ensure_ascii=False, indent=2)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model_path": args.model_path,
                "trust_remote_code": args.trust_remote_code,
                "short_a_tokens": short_anchor_a["tokens"],
                "short_b_tokens": short_anchor_b["tokens"],
                "long_a_tokens": long_anchor_a["tokens"],
                "long_b_tokens": long_anchor_b["tokens"],
                "phase1_request_count": len(reqs),
                "phase2_request_count": len(lora_reqs),
                "arrival_mode": args.arrival_mode,
                "phase1_arrival_layout": args.phase1_arrival_layout,
                "phase2_arrival_layout": args.phase2_arrival_layout,
                "phase1_arrival_rate": args.phase1_arrival_rate,
                "phase2_arrival_rate": args.phase2_arrival_rate,
                "phase1_last_arrival_s": max((float(item.get("arrival_offset_s", 0.0)) for item in reqs), default=0.0),
                "phase2_last_arrival_s": max((float(item.get("arrival_offset_s", 0.0)) for item in lora_reqs), default=0.0),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"[Saved] {req_path}")
    print(f"[Saved] {lora_req_path}")
    print(f"[Saved] {meta_path}")
    return 0


if __name__ == "__main__":
    code = int(main())
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    finally:
        os._exit(code)
