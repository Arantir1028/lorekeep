"""Build the two dataset-backed request files used by WaveSlice evaluations."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Any

from experiments.catalog import DEFAULT_DATASET_SOURCES, DEFAULT_LONG_BENCH_CONFIGS


def _load_dataset_sources(config_path: str | None) -> dict[str, Any]:
    if not config_path:
        return dict(DEFAULT_DATASET_SOURCES)
    payload = json.loads(Path(config_path).read_text(encoding="utf-8"))
    payload = payload.get("datasets", payload) if isinstance(payload, dict) else payload
    entries = (
        payload.values()
        if isinstance(payload, dict)
        else payload
        if isinstance(payload, list)
        else None
    )
    if entries is None:
        raise ValueError("dataset source config must be a list or dict")
    output = {}
    for raw in entries:
        if not isinstance(raw, dict):
            continue
        entry = {
            key: str(raw.get(key, "")).strip().lower()
            if key in {"key", "extractor"}
            else str(raw.get(key, "")).strip()
            for key in ("key", "dataset_id", "split", "extractor")
        }
        if not all(entry.values()):
            raise ValueError(f"invalid dataset source entry: {raw!r}")
        entry["streaming"] = bool(raw.get("streaming", False))
        output[entry["key"]] = entry
    if not output:
        raise ValueError("dataset source config produced no usable entries")
    return output


def _source_field(source: Any, name: str, default: Any = None) -> Any:
    return source.get(name, default) if isinstance(source, dict) else getattr(source, name, default)


def _extract_longbench_prompt(example: dict[str, Any]) -> str | None:
    pieces = [
        str(example[key]).strip()
        for key in ("context", "input", "question", "instruction")
        if isinstance(example.get(key), str) and str(example[key]).strip()
    ]
    return "\n\n".join(pieces) or None


def _extract_ultrachat_prompt(example: dict[str, Any]) -> str | None:
    messages = example.get("messages")
    if not isinstance(messages, list):
        return None
    pieces = [
        str(turn["content"]).strip()
        for turn in messages
        if isinstance(turn, dict)
        and str(turn.get("role", "")).lower() == "user"
        and isinstance(turn.get("content"), str)
        and str(turn["content"]).strip()
    ]
    return "\n\n".join(pieces) or None


def _pick_by_quantile(items: list[dict[str, Any]], q: float) -> dict[str, Any]:
    if not items:
        raise ValueError("cannot pick from empty list")
    ordered = sorted(items, key=lambda item: item["tokens"])
    return ordered[max(0, min(round((len(ordered) - 1) * q), len(ordered) - 1))]


def _pick_many_by_quantiles(
    items: list[dict[str, Any]], quantiles: list[float]
) -> list[dict[str, Any]]:
    ordered = sorted(items, key=lambda item: item["tokens"])
    used, output = set(), []
    for q in quantiles:
        index = max(0, min(round((len(ordered) - 1) * q), len(ordered) - 1))
        if index in used:
            alternatives = sorted(
                (candidate for candidate in range(len(ordered)) if candidate not in used),
                key=lambda candidate: (abs(candidate - index), candidate > index),
            )
            index = alternatives[0] if alternatives else index
        used.add(index)
        output.append(ordered[index])
    return output


def _assign_poisson_arrivals(
    items: list[dict[str, Any]], *, rate_per_s: float, seed: int
) -> list[dict[str, Any]]:
    if rate_per_s <= 0:
        raise ValueError("rate_per_s must be > 0 for poisson arrivals")
    rng, current, output = random.Random(seed), 0.0, []
    for item in items:
        current += rng.expovariate(rate_per_s)
        output.append(dict(item, arrival_offset_s=round(current, 6)))
    return output


def _mixed_arrival_order(
    shorts: list[dict[str, Any]],
    longs: list[dict[str, Any]],
    *,
    seed: int,
    early_short_frac: float,
    post_long_short_bias: float,
) -> list[dict[str, Any]]:
    rng, shorts, longs = random.Random(seed), list(shorts), list(longs)
    rng.shuffle(shorts)
    rng.shuffle(longs)
    if not shorts or not longs:
        return shorts + longs
    prefix = min(len(shorts) - 1, round(len(shorts) * max(0.0, min(0.95, early_short_frac))))
    output, shorts = shorts[:prefix] + [longs.pop(0)], shorts[prefix:]
    while shorts and longs:
        output.append(
            shorts.pop(0)
            if rng.random() < max(0.0, min(1.0, post_long_short_bias))
            else longs.pop(0)
        )
    return output + shorts + longs


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
    if layout == "mixed":
        output, rng = list(items), random.Random(seed)
        rng.shuffle(output)
        return output
    if layout == "beneficiary_rich":
        return _mixed_arrival_order(
            [dict(item) for item in items if item.get("is_short")],
            [dict(item) for item in items if not item.get("is_short")],
            seed=seed,
            early_short_frac=early_short_frac,
            post_long_short_bias=post_long_short_bias,
        )
    raise ValueError(f"unknown arrival layout: {layout}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build dataset request json files.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--out-prefix", required=True)
    parser.add_argument("--trust-remote-code", action="store_true")
    for name, default in (
        ("max-prompt-tokens", 3072),
        ("sample-count", 256),
        ("phase1-short-count", 24),
        ("phase1-long-count", 8),
        ("phase2-short-count", 24),
        ("phase2-long-count", 12),
        ("arrival-seed", 7),
    ):
        parser.add_argument("--" + name, type=int, default=default)
    for name, default in (
        ("phase1-arrival-rate", 6.0),
        ("phase2-arrival-rate", 6.0),
        ("phase1-early-short-frac", 0.25),
        ("phase2-early-short-frac", 0.20),
        ("phase1-post-long-short-bias", 0.70),
        ("phase2-post-long-short-bias", 0.60),
    ):
        parser.add_argument("--" + name, type=float, default=default)
    parser.add_argument("--arrival-mode", choices=["burst", "poisson"], default="poisson")
    for phase in (1, 2):
        parser.add_argument(
            f"--phase{phase}-arrival-layout",
            choices=["grouped", "mixed", "beneficiary_rich"],
            default="beneficiary_rich",
        )
    parser.add_argument("--datasets", default="ultrachat200k,longbench")
    parser.add_argument("--longbench-configs", default=",".join(DEFAULT_LONG_BENCH_CONFIGS))
    parser.add_argument("--dataset-source-config", default="")
    return parser


def _collect(
    load_dataset: Any,
    tokenizer: Any,
    source: Any,
    *,
    configs: list[str | None],
    limit: int,
    max_tokens: int,
    offline: bool,
) -> list[dict[str, Any]]:
    extractor = (
        _extract_ultrachat_prompt
        if _source_field(source, "extractor") == "ultrachat"
        else _extract_longbench_prompt
    )
    output, per_config = [], max(1, math.ceil(limit / len(configs)))
    for config in configs:
        positional = [_source_field(source, "dataset_id")] + ([config] if config else [])
        dataset = load_dataset(
            *positional,
            split=_source_field(source, "split"),
            streaming=False if offline else bool(_source_field(source, "streaming", False)),
        )
        taken = 0
        for example in dataset:
            prompt = extractor(example)
            if not prompt:
                continue
            tokens = len(tokenizer(prompt, add_special_tokens=True).input_ids)
            if 8 <= tokens <= max_tokens:
                item = {"prompt": prompt, "tokens": tokens}
                if config:
                    item["config"] = config
                item["source"] = _source_field(source, "key")
                output.append(item)
                taken += 1
            if taken >= per_config or len(output) >= limit:
                break
        if len(output) >= limit:
            break
    return output


def _quantiles(count: int, low: float, span: float) -> list[float]:
    return [low + span * index / max(1, count - 1) for index in range(count)]


def _request(
    item: dict[str, Any], request_id: str, short: bool, lora: str | None = None
) -> dict[str, Any]:
    output = {"req_id": request_id, "prompt": item["prompt"], "is_short": short}
    if lora:
        output["lora_tag"] = lora
    output.update(source="UltraChat200k" if short else "LongBench", tokens=item["tokens"])
    return output


def main() -> int:
    args = _parser().parse_args()
    from datasets import load_dataset
    from transformers import AutoTokenizer

    offline = any(
        str(os.environ.get(name, "")).strip().lower() in {"1", "true", "yes", "on"}
        for name in ("HF_DATASETS_OFFLINE", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=args.trust_remote_code
    )
    sources = _load_dataset_sources(args.dataset_source_config or None)
    dataset_keys = [key.strip().lower() for key in args.datasets.split(",") if key.strip()]
    missing = sorted(set(dataset_keys) - set(sources))
    if missing:
        raise ValueError(f"unknown dataset source keys: {missing}")
    ultra = (
        _collect(
            load_dataset,
            tokenizer,
            sources["ultrachat200k"],
            configs=[None],
            limit=args.sample_count,
            max_tokens=args.max_prompt_tokens,
            offline=offline,
        )
        if "ultrachat200k" in dataset_keys
        else []
    )
    configs = [value.strip() for value in args.longbench_configs.split(",") if value.strip()]
    long = (
        _collect(
            load_dataset,
            tokenizer,
            sources["longbench"],
            configs=configs,
            limit=args.sample_count,
            max_tokens=args.max_prompt_tokens,
            offline=offline,
        )
        if "longbench" in dataset_keys
        else []
    )
    if len(ultra) < 4 or len(long) < 4:
        raise RuntimeError("insufficient dataset prompts collected")
    counts = {
        "phase1_short": max(2, args.phase1_short_count),
        "phase1_long": max(1, args.phase1_long_count),
        "phase2_short": max(2, args.phase2_short_count),
        "phase2_long": max(2, args.phase2_long_count),
    }
    short_a, short_b = _pick_by_quantile(ultra, 0.15), _pick_by_quantile(ultra, 0.55)
    long_a, long_b = _pick_by_quantile(long, 0.50), _pick_by_quantile(long, 0.90)
    p1_shorts = _pick_many_by_quantiles(ultra, _quantiles(counts["phase1_short"], 0.10, 0.55))
    p1_longs = _pick_many_by_quantiles(long, _quantiles(counts["phase1_long"], 0.45, 0.45))
    p2_shorts = _pick_many_by_quantiles(ultra, _quantiles(counts["phase2_short"], 0.10, 0.60))
    p2_longs = _pick_many_by_quantiles(long, _quantiles(counts["phase2_long"], 0.45, 0.50))
    requests = [_request(short_a, "short_a", True), _request(short_b, "short_b", True)]
    requests += [_request(item, f"short_{index:02d}", True) for index, item in enumerate(p1_shorts)]
    requests += [_request(long_b, "long_b", False)] + [
        _request(item, f"long_{index:02d}", False) for index, item in enumerate(p1_longs)
    ]
    lora_requests = [
        _request(short_a, "short_a", True, "A"),
        _request(short_b, "mid_b", True, "B"),
        _request(long_a, "long_a", False, "A"),
        _request(long_b, "long_b", False, "B"),
    ]
    lora_requests += [
        _request(item, f"short_extra_{index:02d}", True, "A" if index % 2 == 0 else "B")
        for index, item in enumerate(p2_shorts)
    ]
    lora_requests += [
        _request(item, f"long_extra_{index:02d}", False, "A" if index % 2 == 0 else "B")
        for index, item in enumerate(p2_longs)
    ]
    requests = _arrival_order(
        requests,
        seed=args.arrival_seed + 17,
        layout=args.phase1_arrival_layout,
        early_short_frac=args.phase1_early_short_frac,
        post_long_short_bias=args.phase1_post_long_short_bias,
    )
    lora_requests = _arrival_order(
        lora_requests,
        seed=args.arrival_seed + 1017,
        layout=args.phase2_arrival_layout,
        early_short_frac=args.phase2_early_short_frac,
        post_long_short_bias=args.phase2_post_long_short_bias,
    )
    if args.arrival_mode == "poisson":
        requests = _assign_poisson_arrivals(
            requests, rate_per_s=args.phase1_arrival_rate, seed=args.arrival_seed
        )
        lora_requests = _assign_poisson_arrivals(
            lora_requests, rate_per_s=args.phase2_arrival_rate, seed=args.arrival_seed + 1009
        )
    else:
        requests = [dict(item, arrival_offset_s=0.0) for item in requests]
        lora_requests = [dict(item, arrival_offset_s=0.0) for item in lora_requests]
    prefix = Path(args.out_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    paths = [
        Path(str(prefix) + suffix)
        for suffix in ("_requests.json", "_lora_requests.json", "_meta.json")
    ]
    actual = {
        "phase1_short": sum(bool(item["is_short"]) for item in requests),
        "phase2_short": sum(bool(item["is_short"]) for item in lora_requests),
    }
    actual.update(
        phase1_long=len(requests) - actual["phase1_short"],
        phase2_long=len(lora_requests) - actual["phase2_short"],
    )
    metadata = {
        "model_path": args.model_path,
        "trust_remote_code": args.trust_remote_code,
        "short_a_tokens": short_a["tokens"],
        "short_b_tokens": short_b["tokens"],
        "long_a_tokens": long_a["tokens"],
        "long_b_tokens": long_b["tokens"],
    }
    for phase in (1, 2):
        for kind in ("short", "long"):
            metadata[f"phase{phase}_config_{kind}_count"] = counts[f"phase{phase}_{kind}"]
            metadata[f"phase{phase}_{kind}_count"] = actual[f"phase{phase}_{kind}"]
    metadata.update(
        {
            "phase1_long_fraction": actual["phase1_long"] / len(requests) if requests else 0.0,
            "phase2_long_fraction": actual["phase2_long"] / len(lora_requests)
            if lora_requests
            else 0.0,
            "phase1_request_count": len(requests),
            "phase2_request_count": len(lora_requests),
            "arrival_mode": args.arrival_mode,
            "phase1_arrival_layout": args.phase1_arrival_layout,
            "phase2_arrival_layout": args.phase2_arrival_layout,
            "phase1_arrival_rate": args.phase1_arrival_rate,
            "phase2_arrival_rate": args.phase2_arrival_rate,
            "phase1_last_arrival_s": max(
                (float(item.get("arrival_offset_s", 0.0)) for item in requests), default=0.0
            ),
            "phase2_last_arrival_s": max(
                (float(item.get("arrival_offset_s", 0.0)) for item in lora_requests), default=0.0
            ),
        }
    )
    for path, payload in zip(paths, (requests, lora_requests, metadata), strict=False):
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[Saved] {path}")
    return 0


if __name__ == "__main__":
    code = int(main())
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    finally:
        os._exit(code)
