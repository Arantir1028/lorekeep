from __future__ import annotations

import argparse
import contextlib
import gc
import json
import math
import time
from pathlib import Path
from statistics import median
from typing import Any

from engine.runtime_bootstrap import bootstrap_vllm_runtime

bootstrap_vllm_runtime()

from experiments.lut_fingerprint import current_lut_fingerprint
from transformers import AutoTokenizer
import torch
from vllm.engine.arg_utils import EngineArgs
from vllm.engine.llm_engine import LLMEngine
from vllm.sampling_params import SamplingParams


def _build_tokenizer(snapshot: Path, trust_remote_code: bool) -> Any:
    try:
        return AutoTokenizer.from_pretrained(str(snapshot), trust_remote_code=trust_remote_code)
    except Exception:
        return AutoTokenizer.from_pretrained(
            str(snapshot),
            trust_remote_code=trust_remote_code,
            use_fast=False,
        )


def _seed_token_ids(tokenizer: Any) -> list[int]:
    for text in (" hello", " the", "a", " test"):
        ids = tokenizer.encode(text, add_special_tokens=False)
        if ids:
            return [int(x) for x in ids]
    for attr in ("bos_token_id", "eos_token_id", "unk_token_id", "pad_token_id"):
        tok = getattr(tokenizer, attr, None)
        if tok is not None:
            return [int(tok)]
    return [1]


def _make_prompt_token_ids(tokenizer: Any, target_tokens: int) -> list[int]:
    seed = _seed_token_ids(tokenizer)
    repeats = (target_tokens + len(seed) - 1) // len(seed)
    ids = (seed * repeats)[:target_tokens]
    return [int(x) for x in ids]


def _build_engine(
    *,
    model_ref: str,
    trust_remote_code: bool,
    max_model_len: int,
    max_num_batched_tokens: int,
    gpu_memory_utilization: float,
) -> Any:
    args = EngineArgs(
        model=model_ref,
        trust_remote_code=trust_remote_code,
        seed=0,
        enable_lora=False,
        disable_sliding_window=False,
        max_num_batched_tokens=int(max_num_batched_tokens),
        enable_chunked_prefill=True,
        enforce_eager=True,
        max_model_len=int(max_model_len),
        gpu_memory_utilization=float(gpu_memory_utilization),
    )
    return LLMEngine.from_engine_args(args)


def _run_engine_case(
    *,
    engine: Any,
    prompt_token_batches: list[list[int]],
    max_new_tokens: int,
) -> list[dict[str, float | None]]:
    sampling = SamplingParams(max_tokens=max_new_tokens, temperature=0.0)
    trackers: dict[str, dict[str, float | None]] = {}
    for idx, ids in enumerate(prompt_token_batches):
        rid = f"calib-{idx}"
        trackers[rid] = {"arrival": time.perf_counter(), "first": None, "finish": None}
        engine.add_request(rid, {"prompt_token_ids": ids}, sampling)

    deadline = time.time() + 180
    while time.time() < deadline and engine.has_unfinished_requests():
        outputs = engine.step()
        now = time.perf_counter()
        for out in outputs:
            tr = trackers.get(out.request_id)
            if tr is None:
                continue
            tok_count = 0
            try:
                tok_count = len(out.outputs[0].token_ids)
            except Exception:
                tok_count = 0
            if tok_count > 0 and tr["first"] is None:
                tr["first"] = now
            if out.finished:
                tr["finish"] = now

    results: list[dict[str, float | None]] = []
    for idx in range(len(prompt_token_batches)):
        rid = f"calib-{idx}"
        tr = trackers[rid]
        arrival = float(tr["arrival"] or 0.0)
        first = tr["first"]
        finish = tr["finish"]
        results.append(
            {
                "first_ms": ((float(first) - arrival) * 1000.0) if first is not None else None,
                "finish_ms": ((float(finish) - arrival) * 1000.0) if finish is not None else None,
            }
        )
    return results


def _clean_samples(samples: list[float | None]) -> list[float]:
    vals = [float(v) for v in samples if v is not None and math.isfinite(float(v)) and float(v) > 0.0]
    if not vals:
        return []
    if len(vals) == 2:
        lo, hi = sorted(vals)
        if hi / max(lo, 1e-9) >= 4.0:
            return [lo]
        return vals
    med = float(median(vals))
    abs_dev = [abs(v - med) for v in vals]
    mad = float(median(abs_dev))
    if mad > 0:
        keep = [v for v in vals if abs(v - med) <= 4.5 * mad]
        if keep:
            return keep
    lo = med / 2.0
    hi = med * 2.0
    keep = [v for v in vals if lo <= v <= hi]
    return keep or vals


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-ref", required=True)
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--max-model-len", type=int, required=True)
    parser.add_argument("--max-num-batched-tokens", type=int, required=True)
    parser.add_argument("--gpu-memory-utilization", type=float, required=True)
    parser.add_argument("--repeats", type=int, required=True)
    parser.add_argument("--buckets", required=True, help="Comma-separated token buckets")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    buckets = [int(x) for x in args.buckets.split(",") if x.strip()]
    tokenizer = _build_tokenizer(Path(args.snapshot), trust_remote_code=bool(args.trust_remote_code))
    engine = _build_engine(
        model_ref=str(args.model_ref),
        trust_remote_code=bool(args.trust_remote_code),
        max_model_len=int(args.max_model_len),
        max_num_batched_tokens=int(args.max_num_batched_tokens),
        gpu_memory_utilization=float(args.gpu_memory_utilization),
    )

    warm_ids = _make_prompt_token_ids(tokenizer, min(256, buckets[0]))
    _run_engine_case(engine=engine, prompt_token_batches=[warm_ids], max_new_tokens=1)

    solo_us: dict[int, float] = {}
    conc_us: dict[int, dict[int, float]] = {int(b): {} for b in buckets}
    solo_samples_raw: dict[int, list[float]] = {int(b): [] for b in buckets}
    solo_samples: dict[int, list[float]] = {int(b): [] for b in buckets}
    conc_samples_raw: dict[tuple[int, int], list[float]] = {}
    conc_samples: dict[tuple[int, int], list[float]] = {}

    for bucket in buckets:
        ids = _make_prompt_token_ids(tokenizer, int(bucket))
        for _ in range(int(args.repeats)):
            out = _run_engine_case(engine=engine, prompt_token_batches=[ids], max_new_tokens=1)
            first_ms = out[0].get("first_ms")
            if first_ms is not None:
                solo_samples_raw[int(bucket)].append(float(first_ms) * 1000.0)
        solo_samples[int(bucket)] = _clean_samples(solo_samples_raw[int(bucket)])
        if solo_samples[int(bucket)]:
            solo_us[int(bucket)] = float(median(solo_samples[int(bucket)]))

    for short_bucket in buckets:
        for chunk_bucket in buckets:
            if chunk_bucket < short_bucket:
                continue
            short_ids = _make_prompt_token_ids(tokenizer, int(short_bucket))
            chunk_ids = _make_prompt_token_ids(tokenizer, int(chunk_bucket))
            key = (int(short_bucket), int(chunk_bucket))
            conc_samples_raw[key] = []
            for _ in range(int(args.repeats)):
                out = _run_engine_case(
                    engine=engine,
                    prompt_token_batches=[short_ids, chunk_ids],
                    max_new_tokens=1,
                )
                first_ms = out[0].get("first_ms")
                if first_ms is not None:
                    conc_samples_raw[key].append(float(first_ms) * 1000.0)
            conc_samples[key] = _clean_samples(conc_samples_raw[key])
            if conc_samples[key]:
                conc_us[int(short_bucket)][int(chunk_bucket)] = float(median(conc_samples[key]))

    result = {
        "model_ref": str(args.model_ref),
        "snapshot": str(args.snapshot),
        "hardware_fingerprint": current_lut_fingerprint(),
        "max_model_len": int(args.max_model_len),
        "max_num_batched_tokens": int(args.max_num_batched_tokens),
        "gpu_memory_utilization": float(args.gpu_memory_utilization),
        "repeats": int(args.repeats),
        "buckets": buckets,
        "solo_us": {str(k): v for k, v in solo_us.items()},
        "concurrent_short_us": {
            str(k): {str(kk): vv for kk, vv in row.items()} for k, row in conc_us.items()
        },
        "solo_samples_us": {str(k): vals for k, vals in solo_samples.items()},
        "solo_samples_us_raw": {str(k): vals for k, vals in solo_samples_raw.items()},
        "concurrent_samples_us": {f"{k[0]}:{k[1]}": vals for k, vals in conc_samples.items()},
        "concurrent_samples_us_raw": {f"{k[0]}:{k[1]}": vals for k, vals in conc_samples_raw.items()},
    }
    try:
        shutdown = getattr(engine, "shutdown", None)
        if callable(shutdown):
            shutdown()
    except Exception:
        pass
    del engine
    gc.collect()
    with contextlib.suppress(Exception):
        torch.cuda.empty_cache()
    Path(args.out).write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
