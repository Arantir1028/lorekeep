from __future__ import annotations

import argparse
import gc
import json
import math
import time
from pathlib import Path
from statistics import median
from typing import Any

from waveslice.vllm.bootstrap import bootstrap_vllm_runtime, shutdown_vllm_engine

bootstrap_vllm_runtime()
import torch
from transformers import AutoTokenizer
from vllm.engine.arg_utils import EngineArgs
from vllm.engine.llm_engine import LLMEngine
from vllm.sampling_params import SamplingParams

from experiments.lut_fingerprint import current_lut_fingerprint


def _build_tokenizer(snapshot: Path, trust_remote_code: bool) -> Any:
    return AutoTokenizer.from_pretrained(str(snapshot), trust_remote_code=trust_remote_code)


def _seed_token_ids(tokenizer: Any) -> list[int]:
    ids = tokenizer.encode(" calibration", add_special_tokens=False)
    if not ids:
        raise ValueError("tokenizer produced no calibration token ids")
    return [int(value) for value in ids]


def _make_prompt_token_ids(tokenizer: Any, target_tokens: int) -> list[int]:
    seed = _seed_token_ids(tokenizer)
    return (seed * math.ceil(target_tokens / len(seed)))[:target_tokens]


def _build_engine(
    *,
    model_ref: str,
    trust_remote_code: bool,
    max_model_len: int,
    max_num_batched_tokens: int,
    gpu_memory_utilization: float,
) -> Any:
    return LLMEngine.from_engine_args(
        EngineArgs(
            model=model_ref,
            trust_remote_code=trust_remote_code,
            seed=0,
            enable_lora=False,
            disable_sliding_window=False,
            max_num_batched_tokens=max_num_batched_tokens,
            enable_chunked_prefill=True,
            enable_prefix_caching=False,
            enforce_eager=True,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
        )
    )


def _run_engine_case(
    *, engine: Any, prompt_token_batches: list[list[int]], max_new_tokens: int
) -> list[dict[str, float | None]]:
    trackers = {}
    for index, token_ids in enumerate(prompt_token_batches):
        request_id = f"calib-{index}"
        trackers[request_id] = {"arrival": time.perf_counter(), "first": None, "finish": None}
        engine.add_request(
            request_id,
            {"prompt_token_ids": token_ids},
            SamplingParams(max_tokens=max_new_tokens, temperature=0.0),
        )
    deadline = time.time() + 180
    while time.time() < deadline and engine.has_unfinished_requests():
        outputs = engine.step()
        now = time.perf_counter()
        for output in outputs:
            tracker = trackers.get(output.request_id)
            if tracker is None:
                continue
            if output.outputs and output.outputs[0].token_ids and tracker["first"] is None:
                tracker["first"] = now
            if output.finished:
                tracker["finish"] = now
    if engine.has_unfinished_requests():
        engine.abort_request(list(trackers))
        raise TimeoutError("runtime calibration exceeded 180 seconds")
    results = []
    for tracker in trackers.values():
        arrival = float(tracker["arrival"])
        results.append(
            {
                key + "_ms": (float(tracker[key]) - arrival) * 1000.0
                if tracker[key] is not None
                else None
                for key in ("first", "finish")
            }
        )
    return results


def _clean_samples(samples: list[float | None]) -> list[float]:
    values = [
        float(value)
        for value in samples
        if value is not None and math.isfinite(float(value)) and float(value) > 0
    ]
    if len(values) == 2 and max(values) / min(values) >= 4:
        return [min(values)]
    if len(values) < 2:
        return values
    center = float(median(values))
    mad = float(median(abs(value - center) for value in values))
    filtered = (
        [value for value in values if abs(value - center) <= 4.5 * mad]
        if mad
        else [value for value in values if center / 2 <= value <= center * 2]
    )
    return filtered


def _measure(
    engine: Any, batches: list[list[int]], repeats: int
) -> tuple[list[float], list[float]]:
    raw = []
    for _ in range(repeats):
        value = _run_engine_case(engine=engine, prompt_token_batches=batches, max_new_tokens=1)[0][
            "first_ms"
        ]
        if value is None:
            raise RuntimeError("runtime calibration request produced no first-token timestamp")
        raw.append(float(value) * 1000.0)
    return raw, _clean_samples(raw)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-ref", required=True)
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--trust-remote-code", action="store_true")
    for name, cast in (
        ("max-model-len", int),
        ("max-num-batched-tokens", int),
        ("gpu-memory-utilization", float),
        ("repeats", int),
    ):
        parser.add_argument("--" + name, type=cast, required=True)
    parser.add_argument("--buckets", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    buckets = [int(value) for value in args.buckets.split(",") if value.strip()]
    tokenizer = _build_tokenizer(Path(args.snapshot), args.trust_remote_code)
    engine = _build_engine(
        model_ref=args.model_ref,
        trust_remote_code=args.trust_remote_code,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_num_batched_tokens,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    _run_engine_case(
        engine=engine,
        prompt_token_batches=[_make_prompt_token_ids(tokenizer, min(256, buckets[0]))],
        max_new_tokens=1,
    )
    token_ids = {bucket: _make_prompt_token_ids(tokenizer, bucket) for bucket in buckets}
    solo_us, solo_raw, solo_samples = {}, {}, {}
    concurrent_us = {bucket: {} for bucket in buckets}
    concurrent_raw, concurrent_samples = {}, {}
    for bucket in buckets:
        raw, clean = _measure(engine, [token_ids[bucket]], args.repeats)
        solo_raw[bucket], solo_samples[bucket] = raw, clean
        if clean:
            solo_us[bucket] = float(median(clean))
    for short in buckets:
        for chunk in (value for value in buckets if value >= short):
            raw, clean = _measure(engine, [token_ids[short], token_ids[chunk]], args.repeats)
            concurrent_raw[(short, chunk)], concurrent_samples[(short, chunk)] = raw, clean
            if clean:
                concurrent_us[short][chunk] = float(median(clean))
    result = {
        "model_ref": args.model_ref,
        "snapshot": args.snapshot,
        "hardware_fingerprint": current_lut_fingerprint(),
        "max_model_len": args.max_model_len,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "repeats": args.repeats,
        "buckets": buckets,
        "solo_us": {str(key): value for key, value in solo_us.items()},
        "concurrent_short_us": {
            str(key): {str(inner): value for inner, value in row.items()}
            for key, row in concurrent_us.items()
        },
        "solo_samples_us": {str(key): value for key, value in solo_samples.items()},
        "solo_samples_us_raw": {str(key): value for key, value in solo_raw.items()},
        "concurrent_samples_us": {
            f"{key[0]}:{key[1]}": value for key, value in concurrent_samples.items()
        },
        "concurrent_samples_us_raw": {
            f"{key[0]}:{key[1]}": value for key, value in concurrent_raw.items()
        },
    }
    shutdown_vllm_engine(engine)
    del engine
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    Path(args.out).write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
