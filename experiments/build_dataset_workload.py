"""Build dataset-driven request JSON files for Wave-Slice evaluation.

The goal is to reuse the existing repeated evaluation harness while swapping
the synthetic prompts for prompts sampled from open datasets.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from typing import Any, Optional


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


def main() -> int:
    parser = argparse.ArgumentParser(description="Build dataset request json files.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--out-prefix", required=True)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--max-prompt-tokens", type=int, default=3072)
    parser.add_argument("--sample-count", type=int, default=128)
    args = parser.parse_args()

    from datasets import load_dataset
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=args.trust_remote_code,
    )

    ultrachat_prompts: list[dict[str, Any]] = []
    ds_uc = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft", streaming=True)
    for ex in ds_uc:
        prompt = _extract_ultrachat_prompt(ex)
        if not prompt:
            continue
        tokens = len(tok(prompt, add_special_tokens=True).input_ids)
        if 8 <= tokens <= args.max_prompt_tokens:
            ultrachat_prompts.append({"prompt": prompt, "tokens": tokens})
        if len(ultrachat_prompts) >= args.sample_count:
            break

    longbench_prompts: list[dict[str, Any]] = []
    longbench_configs = ["qmsum", "gov_report", "multifieldqa_en", "hotpotqa"]
    per_config = max(1, int(math.ceil(args.sample_count / float(len(longbench_configs)))))
    for cfg in longbench_configs:
        ds_lb = load_dataset("Xnhyacinth/LongBench", cfg, split="test")
        taken = 0
        for ex in ds_lb:
            prompt = _extract_longbench_prompt(ex)
            if not prompt:
                continue
            tokens = len(tok(prompt, add_special_tokens=True).input_ids)
            if 8 <= tokens <= args.max_prompt_tokens:
                longbench_prompts.append({"prompt": prompt, "tokens": tokens, "config": cfg})
                taken += 1
            if taken >= per_config or len(longbench_prompts) >= args.sample_count:
                break
        if len(longbench_prompts) >= args.sample_count:
            break

    if len(ultrachat_prompts) < 4 or len(longbench_prompts) < 4:
        raise RuntimeError("insufficient dataset prompts collected")

    short_a = _pick_by_quantile(ultrachat_prompts, 0.15)
    short_b = _pick_by_quantile(ultrachat_prompts, 0.55)
    long_a = _pick_by_quantile(longbench_prompts, 0.50)
    long_b = _pick_by_quantile(longbench_prompts, 0.90)

    reqs = [
        {"req_id": "short_a", "prompt": short_a["prompt"], "is_short": True, "source": "UltraChat200k", "tokens": short_a["tokens"]},
        {"req_id": "short_b", "prompt": short_b["prompt"], "is_short": True, "source": "UltraChat200k", "tokens": short_b["tokens"]},
        {"req_id": "long_b", "prompt": long_b["prompt"], "is_short": False, "source": "LongBench", "tokens": long_b["tokens"]},
    ]
    lora_reqs = [
        {"req_id": "short_a", "prompt": short_a["prompt"], "is_short": True, "lora_tag": "A", "source": "UltraChat200k", "tokens": short_a["tokens"]},
        {"req_id": "mid_b", "prompt": short_b["prompt"], "is_short": True, "lora_tag": "B", "source": "UltraChat200k", "tokens": short_b["tokens"]},
        {"req_id": "long_a", "prompt": long_a["prompt"], "is_short": False, "lora_tag": "A", "source": "LongBench", "tokens": long_a["tokens"]},
        {"req_id": "long_b", "prompt": long_b["prompt"], "is_short": False, "lora_tag": "B", "source": "LongBench", "tokens": long_b["tokens"]},
    ]

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
                "short_a_tokens": short_a["tokens"],
                "short_b_tokens": short_b["tokens"],
                "long_a_tokens": long_a["tokens"],
                "long_b_tokens": long_b["tokens"],
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
    raise SystemExit(main())
