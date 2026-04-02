"""Repeated Wave-Slice claim evaluation with warmup and error-rate reporting.

What this script measures:
1) Phase-I slicing gain (non-LoRA): TTFT p99 gain + text mismatch rate.
2) Phase-II multi-stream gain (LoRA): TTFT/slowdown gain + text mismatch rate.
3) Baseline LoRA noise floor: baseline-vs-baseline mismatch rate.

The script is designed for paper-style experiments:
- warmup iterations
- many repeats
- per-repeat records + aggregate statistics
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import time
from dataclasses import dataclass
from typing import Any, Optional

os.environ.setdefault("VLLM_NO_USAGE_STATS", "1")

from engine.runtime_bootstrap import bootstrap_vllm_runtime
from engine.vllm_hijacker import (
    WaveSlicePolicy,
    get_wave_slice_metrics,
    inject_wave_slice,
    reset_wave_slice_metrics,
    uninject_wave_slice,
)

bootstrap_vllm_runtime()


@dataclass(frozen=True)
class Req:
    req_id: str
    prompt: str
    is_short: bool
    lora_tag: Optional[str] = None


def _load_reqs_json(path: str) -> list[Req]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, list):
        raise ValueError(f"requests json must be a list: {path}")
    reqs: list[Req] = []
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"request item #{i} must be an object")
        req_id = str(item.get("req_id") or item.get("id") or f"req_{i}")
        prompt = str(item.get("prompt") or "")
        if not prompt.strip():
            raise ValueError(f"request item #{i} has empty prompt")
        is_short = bool(item.get("is_short"))
        lora_tag = item.get("lora_tag")
        reqs.append(Req(req_id=req_id, prompt=prompt, is_short=is_short, lora_tag=lora_tag))
    return reqs


def _percentile(values: list[float], p: float) -> Optional[float]:
    if not values:
        return None
    if p <= 0:
        return min(values)
    if p >= 100:
        return max(values)
    ordered = sorted(values)
    k = (len(ordered) - 1) * (p / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(ordered) - 1)
    if lo == hi:
        return ordered[lo]
    frac = k - lo
    return ordered[lo] + (ordered[hi] - ordered[lo]) * frac


def _stats(values: list[Optional[float]]) -> dict[str, Optional[float]]:
    data = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not data:
        return {
            "count": 0.0,
            "mean": None,
            "p50": None,
            "p95": None,
            "p99": None,
            "min": None,
            "max": None,
        }
    return {
        "count": float(len(data)),
        "mean": float(sum(data) / len(data)),
        "p50": _percentile(data, 50.0),
        "p95": _percentile(data, 95.0),
        "p99": _percentile(data, 99.0),
        "min": min(data),
        "max": max(data),
    }


def _ratio(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None or b <= 0:
        return None
    return a / b


def _mk_requests(short_a_repeat: int, short_b_repeat: int, long_repeat: int) -> list[Req]:
    short_1 = (
        (
            "Background note: efficient ML serving systems benefit from stable pipelines, "
            "careful batching, and predictable latency. "
        )
        * short_a_repeat
        + "Task: Translate the quoted sentence into French and output only the translation. "
        + "Sentence: 'I love machine learning systems and efficient serving pipelines.'"
    )
    short_2 = (
        (
            "Observed examples: 10+20=30. 20+30=50. 30+40=70. 40+50=90. "
            "The first addend grows by 10 each step and the second addend also grows by 10. "
        )
        * short_b_repeat
        + "Task: Continue the pattern with the next three equations and then give one short rule sentence. "
        + "Answer without repeating the prompt."
    )
    long_1 = (
        (
            "Passage: artificial intelligence changes systems engineering and deployment, "
            "especially when the serving stack must handle heterogeneous LoRA workloads, "
            "long-tail prompt lengths, fairness constraints, and resource contention. "
        )
        * long_repeat
        + "Task: Write exactly one sentence that summarizes the passage above. "
        + "Do not use section headers, numbering, bullet points, or repeated outline phrases."
    )
    return [
        Req("short_a", short_1, True, "A"),
        Req("short_b", short_2, True, "B"),
        Req("long_b", long_1, False, "B"),
    ]


def _mk_lora_requests(short_a_repeat: int, short_b_repeat: int, long_repeat: int) -> list[Req]:
    short_1 = (
        "Task: Translate the quoted sentence into French and output only the translation. "
        + "Sentence: 'I love machine learning.'"
    )
    mid_1 = (
        (
            "Observed examples: 10+20=30. 20+30=50. 30+40=70. 40+50=90. "
            "The first addend grows by 10 each step and the second addend also grows by 10. "
        )
        * short_b_repeat
        + "Task: Continue the pattern with the next three equations and then give one short rule sentence. "
        + "Answer without repeating the prompt."
    )
    long_1 = (
        "Task: Write exactly one sentence that summarizes the passage. "
        + ("Artificial intelligence is transforming industry, deployment, and systems engineering. " * max(1, int(long_repeat * 0.6)))
    )
    long_2 = (
        "Task: Write exactly one sentence that summarizes the passage. "
        + ("Artificial intelligence is transforming industry, deployment, and systems engineering. " * long_repeat)
    )
    return [
        Req("short_a", short_1, True, "A"),
        Req("mid_b", mid_1, True, "B"),
        Req("long_a", long_1, False, "A"),
        Req("long_b", long_2, False, "B"),
    ]


def _measure_input_tokens(
    model_path: str,
    reqs: list[Req],
    *,
    trust_remote_code: bool = False,
) -> dict[str, int]:
    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=trust_remote_code)
        return {
            req.req_id: len(tok.encode(req.prompt, add_special_tokens=False))
            for req in reqs
        }
    except Exception:
        return {
            req.req_id: max(1, len(req.prompt.split()))
            for req in reqs
        }


def _fit_requests_to_context(
    *,
    model_path: str,
    short_a_repeat: int,
    short_b_repeat: int,
    long_repeat: int,
    max_prompt_tokens: int,
    trust_remote_code: bool = False,
) -> tuple[list[Req], dict[str, int], int]:
    cur_long_repeat = max(1, long_repeat)
    for _ in range(8):
        reqs = _mk_requests(short_a_repeat, short_b_repeat, cur_long_repeat)
        tok_lens = _measure_input_tokens(model_path, reqs, trust_remote_code=trust_remote_code)
        if max(tok_lens.values()) <= max_prompt_tokens:
            return reqs, tok_lens, cur_long_repeat
        long_len = tok_lens.get("long_b", max(tok_lens.values()))
        if long_len <= 0 or cur_long_repeat <= 1:
            break
        # Scale down proportionally and leave a small margin.
        cur_long_repeat = max(1, int(cur_long_repeat * (max_prompt_tokens / long_len) * 0.96))
    reqs = _mk_requests(short_a_repeat, short_b_repeat, cur_long_repeat)
    tok_lens = _measure_input_tokens(model_path, reqs, trust_remote_code=trust_remote_code)
    return reqs, tok_lens, cur_long_repeat


def _fit_lora_requests_to_context(
    *,
    model_path: str,
    short_a_repeat: int,
    short_b_repeat: int,
    long_repeat: int,
    max_prompt_tokens: int,
    trust_remote_code: bool = False,
) -> tuple[list[Req], dict[str, int], int]:
    cur_long_repeat = max(1, long_repeat)
    for _ in range(8):
        reqs = _mk_lora_requests(short_a_repeat, short_b_repeat, cur_long_repeat)
        tok_lens = _measure_input_tokens(model_path, reqs, trust_remote_code=trust_remote_code)
        if max(tok_lens.values()) <= max_prompt_tokens:
            return reqs, tok_lens, cur_long_repeat
        long_len = max(v for k, v in tok_lens.items() if "long" in k)
        if long_len <= 0 or cur_long_repeat <= 1:
            break
        cur_long_repeat = max(1, int(cur_long_repeat * (max_prompt_tokens / long_len) * 0.96))
    reqs = _mk_lora_requests(short_a_repeat, short_b_repeat, cur_long_repeat)
    tok_lens = _measure_input_tokens(model_path, reqs, trust_remote_code=trust_remote_code)
    return reqs, tok_lens, cur_long_repeat


def _mk_lora_request(LoRARequest: Any, name: str, req_id: int, path: str) -> Any:
    try:
        return LoRARequest(lora_name=name, lora_int_id=req_id, lora_path=path)
    except TypeError:
        return LoRARequest(lora_name=name, lora_int_id=req_id, lora_local_path=path)


def _configure_mode(
    *,
    model_name: str,
    mode: str,
    phase2_dispatch_mode: str,
    phase1_objective_mode: str,
    phase1_gamma: float,
    phase1_ingress_target_chunk: int,
    phase1_ingress_direct_authoritative: bool,
    phase1_ingress_exact_chunk: bool,
) -> None:
    if mode == "baseline":
        uninject_wave_slice()
        return

    if mode == "phase1_only":
        policy = WaveSlicePolicy(
            enable_phase1_scheduler=True,
            enable_phase2_modelrunner=False,
            enable_metrics_hook=True,
            enable_sjf_reorder=True,
            enable_tick_hide=False,
            enable_vllm_lora_compat_patch=True,
            scheduler_objective_mode=phase1_objective_mode,
            phase1_ingress_target_chunk=int(phase1_ingress_target_chunk),
            phase1_ingress_direct_authoritative=bool(phase1_ingress_direct_authoritative),
            phase1_ingress_exact_chunk=bool(phase1_ingress_exact_chunk),
        )
        inject_wave_slice(model_name, gamma=float(phase1_gamma), policy=policy, force=True)
        return

    if mode == "phase2_lora":
        policy = WaveSlicePolicy(
            enable_phase1_scheduler=False,
            enable_phase2_modelrunner=True,
            enable_metrics_hook=True,
            enable_sjf_reorder=False,
            enable_tick_hide=False,
            enable_vllm_lora_compat_patch=True,
            phase2_consistency_mode="balanced",
            phase2_dispatch_mode=phase2_dispatch_mode,
        )
        inject_wave_slice(model_name, policy=policy, force=True)
        return

    if mode == "phase12_lora":
        policy = WaveSlicePolicy(
            enable_phase1_scheduler=True,
            enable_phase2_modelrunner=True,
            enable_metrics_hook=True,
            enable_sjf_reorder=True,
            enable_tick_hide=False,
            enable_vllm_lora_compat_patch=True,
            scheduler_objective_mode=phase1_objective_mode,
            phase1_ingress_target_chunk=int(phase1_ingress_target_chunk),
            phase1_ingress_direct_authoritative=bool(phase1_ingress_direct_authoritative),
            phase1_ingress_exact_chunk=bool(phase1_ingress_exact_chunk),
            phase2_consistency_mode="balanced",
            phase2_dispatch_mode=phase2_dispatch_mode,
        )
        inject_wave_slice(model_name, gamma=float(phase1_gamma), policy=policy, force=True)
        return

    if mode == "phase2_lora_strict":
        policy = WaveSlicePolicy(
            enable_phase1_scheduler=False,
            enable_phase2_modelrunner=True,
            enable_metrics_hook=True,
            enable_sjf_reorder=False,
            enable_tick_hide=False,
            enable_vllm_lora_compat_patch=True,
            phase2_consistency_mode="strict",
            phase2_dispatch_mode=phase2_dispatch_mode,
        )
        inject_wave_slice(model_name, policy=policy, force=True)
        return

    if mode == "phase12_lora_strict":
        policy = WaveSlicePolicy(
            enable_phase1_scheduler=True,
            enable_phase2_modelrunner=True,
            enable_metrics_hook=True,
            enable_sjf_reorder=True,
            enable_tick_hide=False,
            enable_vllm_lora_compat_patch=True,
            scheduler_objective_mode=phase1_objective_mode,
            phase1_ingress_target_chunk=int(phase1_ingress_target_chunk),
            phase1_ingress_direct_authoritative=bool(phase1_ingress_direct_authoritative),
            phase1_ingress_exact_chunk=bool(phase1_ingress_exact_chunk),
            phase2_consistency_mode="strict",
            phase2_dispatch_mode=phase2_dispatch_mode,
        )
        inject_wave_slice(model_name, gamma=float(phase1_gamma), policy=policy, force=True)
        return

    if mode == "baseline_lora_compat":
        policy = WaveSlicePolicy(
            enable_phase1_scheduler=False,
            enable_phase2_modelrunner=False,
            enable_metrics_hook=True,
            enable_sjf_reorder=False,
            enable_tick_hide=False,
            enable_vllm_lora_compat_patch=True,
        )
        inject_wave_slice(model_name, policy=policy, force=True)
        return

    raise ValueError(f"Unknown mode: {mode}")


def _cleanup_engine(engine: Optional[Any]) -> None:
    if engine is not None:
        del engine
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    uninject_wave_slice()


def _build_engine(
    *,
    model_path: str,
    model_name: str,
    mode: str,
    enable_lora: bool,
    max_num_batched_tokens: int,
    max_model_len: int,
    gpu_memory_utilization: float,
    phase1_objective_mode: str,
    phase1_gamma: float,
    phase1_ingress_target_chunk: int,
    phase1_ingress_direct_authoritative: bool,
    phase1_ingress_exact_chunk: bool,
    enable_chunked_prefill: bool,
    adapter_a: Optional[str] = None,
    adapter_b: Optional[str] = None,
    phase2_dispatch_mode: str = "synchronized",
    trust_remote_code: bool = False,
) -> tuple[Any, dict[str, Any]]:
    from vllm.engine.arg_utils import EngineArgs
    from vllm.engine.llm_engine import LLMEngine

    _configure_mode(
        model_name=model_name,
        mode=mode,
        phase2_dispatch_mode=phase2_dispatch_mode,
        phase1_objective_mode=phase1_objective_mode,
        phase1_gamma=phase1_gamma,
        phase1_ingress_target_chunk=phase1_ingress_target_chunk,
        phase1_ingress_direct_authoritative=phase1_ingress_direct_authoritative,
        phase1_ingress_exact_chunk=phase1_ingress_exact_chunk,
    )

    effective_batched_tokens = int(max_num_batched_tokens)
    if not enable_chunked_prefill:
        effective_batched_tokens = max(effective_batched_tokens, int(max_model_len))

    engine_args = EngineArgs(
        model=model_path,
        trust_remote_code=trust_remote_code,
        seed=0,
        enable_lora=enable_lora,
        max_lora_rank=32,
        max_num_batched_tokens=effective_batched_tokens,
        enable_chunked_prefill=enable_chunked_prefill,
        disable_sliding_window=True,
        enforce_eager=True,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
    )
    engine = LLMEngine.from_engine_args(engine_args)

    lora_map: dict[str, Any] = {}
    if enable_lora:
        from vllm.lora.request import LoRARequest

        if not adapter_a or not adapter_b:
            raise ValueError("LoRA mode requires adapter_a and adapter_b.")
        lora_map = {
            "A": _mk_lora_request(LoRARequest, "adapter_A", 1, adapter_a),
            "B": _mk_lora_request(LoRARequest, "adapter_B", 2, adapter_b),
        }
    return engine, lora_map


def _run_round(
    *,
    engine: Any,
    reqs: list[Req],
    max_new_tokens: int,
    timeout_sec: int,
    enable_lora: bool,
    lora_map: dict[str, Any],
    run_tag: str,
) -> dict[str, Any]:
    from vllm.sampling_params import SamplingParams

    sampling = SamplingParams(max_tokens=max_new_tokens, temperature=0.0)
    trackers: dict[str, dict[str, Any]] = {}
    reset_wave_slice_metrics()

    arrival = time.perf_counter()
    for r in reqs:
        rid = f"{run_tag}:{r.req_id}"
        if enable_lora:
            engine.add_request(rid, r.prompt, sampling, lora_request=lora_map[r.lora_tag or "A"])
        else:
            engine.add_request(rid, r.prompt, sampling)
        trackers[rid] = {
            "orig_req_id": r.req_id,
            "arrival_s": arrival,
            "first_s": None,
            "finish_s": None,
            "is_short": r.is_short,
            "text": "",
        }

    round_start = time.perf_counter()
    deadline = time.time() + timeout_sec
    while time.time() < deadline and engine.has_unfinished_requests():
        outputs = engine.step()
        now = time.perf_counter()
        for out in outputs:
            rid = out.request_id
            if rid not in trackers:
                continue
            tok_count = 0
            txt = ""
            try:
                payload = out.outputs[0]
                tok_count = len(payload.token_ids)
                txt = str(payload.text or "")
            except Exception:
                pass
            if tok_count > 0 and trackers[rid]["first_s"] is None:
                trackers[rid]["first_s"] = now
            if out.finished:
                trackers[rid]["finish_s"] = now
                trackers[rid]["text"] = txt
    round_end = time.perf_counter()

    ttft_short_ms: list[float] = []
    finished_count = 0
    for tr in trackers.values():
        if tr["is_short"] and tr["first_s"] is not None:
            ttft_short_ms.append((tr["first_s"] - tr["arrival_s"]) * 1000.0)
        if tr["finish_s"] is not None:
            finished_count += 1

    report = get_wave_slice_metrics(reset=True)
    result = {
        "texts": {tr["orig_req_id"]: tr["text"] for tr in trackers.values()},
        "ttft_short_p99_ms": _percentile(ttft_short_ms, 99.0),
        "round_wall_ms": (round_end - round_start) * 1000.0,
        "timed_out": finished_count != len(trackers),
        "finished_requests": finished_count,
        "total_requests": len(trackers),
        "hook_report": report,
    }
    return result


def _run_mode_series(
    *,
    model_path: str,
    model_name: str,
    reqs: list[Req],
    max_new_tokens: int,
    timeout_sec: int,
    mode: str,
    enable_lora: bool,
    warmup_iters: int,
    repeats: int,
    max_num_batched_tokens: int,
    max_model_len: int,
    gpu_memory_utilization: float,
    phase1_objective_mode: str,
    phase1_gamma: float,
    phase1_ingress_target_chunk: int,
    phase1_ingress_direct_authoritative: bool,
    phase1_ingress_exact_chunk: bool,
    enable_chunked_prefill: bool,
    adapter_a: Optional[str] = None,
    adapter_b: Optional[str] = None,
    phase2_dispatch_mode: str = "synchronized",
    trust_remote_code: bool = False,
) -> list[dict[str, Any]]:
    engine = None
    try:
        engine, lora_map = _build_engine(
            model_path=model_path,
            model_name=model_name,
            mode=mode,
            enable_lora=enable_lora,
            max_num_batched_tokens=max_num_batched_tokens,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            phase1_objective_mode=phase1_objective_mode,
            phase1_gamma=phase1_gamma,
            phase1_ingress_target_chunk=phase1_ingress_target_chunk,
            phase1_ingress_direct_authoritative=phase1_ingress_direct_authoritative,
            phase1_ingress_exact_chunk=phase1_ingress_exact_chunk,
            enable_chunked_prefill=enable_chunked_prefill,
            adapter_a=adapter_a,
            adapter_b=adapter_b,
            phase2_dispatch_mode=phase2_dispatch_mode,
            trust_remote_code=trust_remote_code,
        )
        for i in range(warmup_iters):
            _run_round(
                engine=engine,
                reqs=reqs,
                max_new_tokens=max_new_tokens,
                timeout_sec=timeout_sec,
                enable_lora=enable_lora,
                lora_map=lora_map,
                run_tag=f"warmup_{mode}_{i}",
            )
        rows: list[dict[str, Any]] = []
        for i in range(repeats):
            rows.append(
                _run_round(
                    engine=engine,
                    reqs=reqs,
                    max_new_tokens=max_new_tokens,
                    timeout_sec=timeout_sec,
                    enable_lora=enable_lora,
                    lora_map=lora_map,
                    run_tag=f"repeat_{mode}_{i}",
                )
            )
        return rows
    finally:
        _cleanup_engine(engine)


def _text_match_rate(a: dict[str, str], b: dict[str, str]) -> float:
    keys = sorted(set(a.keys()) & set(b.keys()))
    if not keys:
        return 0.0
    hit = sum(1 for k in keys if a.get(k, "") == b.get(k, ""))
    return float(hit) / float(len(keys))


def _text_mismatch_details(
    a: dict[str, str],
    b: dict[str, str],
    *,
    max_preview: int = 160,
) -> list[dict[str, Any]]:
    keys = sorted(set(a.keys()) & set(b.keys()))
    details: list[dict[str, Any]] = []
    for k in keys:
        left = str(a.get(k, "") or "")
        right = str(b.get(k, "") or "")
        if left == right:
            continue
        common = 0
        for ch_l, ch_r in zip(left, right):
            if ch_l != ch_r:
                break
            common += 1
        details.append(
            {
                "req_id": k,
                "common_prefix_chars": int(common),
                "base_preview": left[:max_preview],
                "wave_preview": right[:max_preview],
            }
        )
    return details


def _semantic_check(req_id: str, text: str) -> dict[str, Any]:
    raw = str(text or "")
    s = raw.strip()
    low = s.lower()
    result: dict[str, Any] = {
        "pass": False,
        "score": 0.0,
        "reason": "empty",
    }
    if not s:
        return result

    if req_id == "short_a":
        french_markers = [
            " je ", " j'", " les ", " des ", " une ", " un ", " et ",
            " apprentissage ", " automatique", " systemes", " systèmes",
            " efficaces", " efficaces", " pipelines",
        ]
        has_french = any(m in f" {low} " for m in french_markers)
        leaked_prompt = "translate to french" in low
        score = 0.0
        if has_french:
            score += 1.0
        if not leaked_prompt:
            score += 0.5
        result.update(
            {
                "pass": bool(has_french and not leaked_prompt),
                "score": score,
                "reason": "french_like" if has_french and not leaked_prompt else "not_french_like",
            }
        )
        return result

    if req_id == "short_b":
        expected_markers = ["110", "130", "150", "50+60", "60+70", "70+80", "pattern"]
        hits = sum(1 for m in expected_markers if m in low)
        leaked_prompt = "continue the arithmetic pattern" in low
        result.update(
            {
                "pass": bool(hits >= 1 and not leaked_prompt),
                "score": float(hits) - (0.5 if leaked_prompt else 0.0),
                "reason": "pattern_continuation" if hits >= 1 and not leaked_prompt else "weak_pattern_continuation",
            }
        )
        return result

    if req_id == "long_b":
        topic_markers = [
            "artificial intelligence", "ai", "systems", "engineering", "deployment",
            "lora", "workload", "workloads", "serving", "heterogeneous",
        ]
        bad_markers = [
            "1. introduction", "2. related work", "3. methodology",
            "4. experiments", "5. conclusion", "6. references",
        ]
        topic_hits = sum(1 for m in topic_markers if m in low)
        bad_hits = sum(1 for m in bad_markers if m in low)
        one_sentence_like = s.count(".") <= 2 and s.count("\n") == 0
        result.update(
            {
                "pass": bool(topic_hits >= 2 and one_sentence_like and bad_hits == 0),
                "score": float(topic_hits) - float(bad_hits),
                "reason": (
                    "topic_summary"
                    if topic_hits >= 2 and one_sentence_like and bad_hits == 0
                    else "off_topic_or_outline"
                ),
            }
        )
        return result

    result.update(
        {
            "pass": bool(s),
            "score": 1.0,
            "reason": "non_empty",
        }
    )
    return result


def _semantic_summary(texts: dict[str, str]) -> dict[str, Any]:
    details = {req_id: _semantic_check(req_id, txt) for req_id, txt in texts.items()}
    passes = [1.0 if item.get("pass") else 0.0 for item in details.values()]
    return {
        "pass_rate": float(sum(passes) / len(passes)) if passes else 0.0,
        "details": details,
    }


def _run_phase1_pair(
    *,
    base_rows: list[dict[str, Any]],
    base_repeat_rows: list[dict[str, Any]],
    wave_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for base, base_repeat, p1 in zip(base_rows, base_repeat_rows, wave_rows):
        match = _text_match_rate(base["texts"], p1["texts"])
        match_noise = _text_match_rate(base["texts"], base_repeat["texts"])
        mismatch_details = _text_mismatch_details(base["texts"], p1["texts"])
        base_sem = _semantic_summary(base["texts"])
        wave_sem = _semantic_summary(p1["texts"])
        p1_sched = (p1["hook_report"] or {}).get("scheduler", {})
        rows.append(
            {
                "base_ttft_short_p99_ms": base["ttft_short_p99_ms"],
                "wave_ttft_short_p99_ms": p1["ttft_short_p99_ms"],
                "ttft_improve_ratio": _ratio(base["ttft_short_p99_ms"], p1["ttft_short_p99_ms"]),
                "base_round_wall_ms": base.get("round_wall_ms"),
                "wave_round_wall_ms": p1.get("round_wall_ms"),
                "round_wall_improve_ratio": _ratio(base.get("round_wall_ms"), p1.get("round_wall_ms")),
                "text_match_rate": match,
                "mismatch_details": mismatch_details,
                "error_rate": 1.0 - match,
                "baseline_noise_match_rate": match_noise,
                "baseline_noise_error_rate": 1.0 - match_noise,
                "incremental_error_rate": (1.0 - match) - (1.0 - match_noise),
                "base_semantic_pass_rate": base_sem.get("pass_rate"),
                "wave_semantic_pass_rate": wave_sem.get("pass_rate"),
                "semantic_pass_delta": (wave_sem.get("pass_rate") or 0.0) - (base_sem.get("pass_rate") or 0.0),
                "base_semantic_details": base_sem.get("details"),
                "wave_semantic_details": wave_sem.get("details"),
                "base_texts": base.get("texts"),
                "wave_texts": p1.get("texts"),
                "scheduler_apply_ratio": p1_sched.get("apply_ratio"),
                "scheduler_applied": p1_sched.get("applied"),
                "scheduler_attempts": p1_sched.get("attempts"),
                "baseline_chunk_avg": p1_sched.get("baseline_chunk_avg"),
                "chosen_chunk_avg": p1_sched.get("chosen_chunk_avg"),
                "chosen_vs_baseline_ratio_avg": p1_sched.get("chosen_vs_baseline_ratio_avg"),
                "explicit_plan_ratio": p1_sched.get("explicit_plan_ratio"),
                "rewrite_apply_ratio": p1_sched.get("rewrite_apply_ratio"),
                "rewrite_old_chunk_avg": p1_sched.get("rewrite_old_chunk_avg"),
                "rewrite_new_chunk_avg": p1_sched.get("rewrite_new_chunk_avg"),
                "rewrite_token_delta_avg": p1_sched.get("rewrite_token_delta_avg"),
                "virtual_cap_apply_ratio": p1_sched.get("virtual_cap_apply_ratio"),
                "virtual_cap_old_avg": p1_sched.get("virtual_cap_old_avg"),
                "virtual_cap_new_avg": p1_sched.get("virtual_cap_new_avg"),
                "virtual_cap_target_set": p1_sched.get("virtual_cap_target_set"),
                "virtual_cap_helper_calls": p1_sched.get("virtual_cap_helper_calls"),
                "virtual_cap_prefill_calls": p1_sched.get("virtual_cap_prefill_calls"),
                "virtual_cap_target_hits": p1_sched.get("virtual_cap_target_hits"),
                "probe_total": p1_sched.get("probe_total"),
                "probe_slice_eligible_ratio": p1_sched.get("probe_slice_eligible_ratio"),
                "probe_best_lt_long_ratio": p1_sched.get("probe_best_lt_long_ratio"),
                "probe_short_avg": p1_sched.get("probe_short_avg"),
                "probe_long_avg": p1_sched.get("probe_long_avg"),
                "probe_baseline_avg": p1_sched.get("probe_baseline_avg"),
                "probe_best_avg": p1_sched.get("probe_best_avg"),
                "probe_queue_avg": p1_sched.get("probe_queue_avg"),
                "probe_wait_us_avg": p1_sched.get("probe_wait_us_avg"),
                "probe_reasons": p1_sched.get("probe_reasons"),
                "hook_report": p1.get("hook_report"),
                "baseline_timed_out": base.get("timed_out"),
                "wave_timed_out": p1.get("timed_out"),
            }
        )
    return {
        "rows": rows,
        "summary": {
            "ttft_improve_ratio": _stats([r.get("ttft_improve_ratio") for r in rows]),
            "round_wall_improve_ratio": _stats([r.get("round_wall_improve_ratio") for r in rows]),
            "error_rate": _stats([r.get("error_rate") for r in rows]),
            "baseline_noise_error_rate": _stats([r.get("baseline_noise_error_rate") for r in rows]),
            "incremental_error_rate": _stats([r.get("incremental_error_rate") for r in rows]),
            "base_semantic_pass_rate": _stats([r.get("base_semantic_pass_rate") for r in rows]),
            "wave_semantic_pass_rate": _stats([r.get("wave_semantic_pass_rate") for r in rows]),
            "semantic_pass_delta": _stats([r.get("semantic_pass_delta") for r in rows]),
            "scheduler_apply_ratio": _stats([r.get("scheduler_apply_ratio") for r in rows]),
            "baseline_chunk_avg": _stats([r.get("baseline_chunk_avg") for r in rows]),
            "chosen_chunk_avg": _stats([r.get("chosen_chunk_avg") for r in rows]),
            "chosen_vs_baseline_ratio_avg": _stats([r.get("chosen_vs_baseline_ratio_avg") for r in rows]),
            "explicit_plan_ratio": _stats([r.get("explicit_plan_ratio") for r in rows]),
            "rewrite_apply_ratio": _stats([r.get("rewrite_apply_ratio") for r in rows]),
            "rewrite_old_chunk_avg": _stats([r.get("rewrite_old_chunk_avg") for r in rows]),
            "rewrite_new_chunk_avg": _stats([r.get("rewrite_new_chunk_avg") for r in rows]),
            "rewrite_token_delta_avg": _stats([r.get("rewrite_token_delta_avg") for r in rows]),
            "virtual_cap_apply_ratio": _stats([r.get("virtual_cap_apply_ratio") for r in rows]),
            "virtual_cap_old_avg": _stats([r.get("virtual_cap_old_avg") for r in rows]),
            "virtual_cap_new_avg": _stats([r.get("virtual_cap_new_avg") for r in rows]),
            "virtual_cap_target_set": _stats([r.get("virtual_cap_target_set") for r in rows]),
            "virtual_cap_helper_calls": _stats([r.get("virtual_cap_helper_calls") for r in rows]),
            "virtual_cap_prefill_calls": _stats([r.get("virtual_cap_prefill_calls") for r in rows]),
            "virtual_cap_target_hits": _stats([r.get("virtual_cap_target_hits") for r in rows]),
            "probe_total": _stats([r.get("probe_total") for r in rows]),
            "probe_slice_eligible_ratio": _stats([r.get("probe_slice_eligible_ratio") for r in rows]),
            "probe_best_lt_long_ratio": _stats([r.get("probe_best_lt_long_ratio") for r in rows]),
            "probe_short_avg": _stats([r.get("probe_short_avg") for r in rows]),
            "probe_long_avg": _stats([r.get("probe_long_avg") for r in rows]),
            "probe_baseline_avg": _stats([r.get("probe_baseline_avg") for r in rows]),
            "probe_best_avg": _stats([r.get("probe_best_avg") for r in rows]),
            "probe_queue_avg": _stats([r.get("probe_queue_avg") for r in rows]),
            "probe_wait_us_avg": _stats([r.get("probe_wait_us_avg") for r in rows]),
        },
    }


def _run_phase2_block(
    *,
    base_rows: list[dict[str, Any]],
    base_repeat_rows: list[dict[str, Any]],
    wave_rows: list[dict[str, Any]],
    strict_rows: Optional[list[dict[str, Any]]],
    include_strict: bool,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    strict_eval_rows: list[dict[str, Any]] = []
    for idx, (lora_base, lora_base_repeat, lora_p2) in enumerate(zip(base_rows, base_repeat_rows, wave_rows)):
        base_report = lora_base["hook_report"] or {}
        wave_report = lora_p2["hook_report"] or {}
        base_slow = (base_report.get("slowdown_short") or {}).get("p99")
        wave_slow = (wave_report.get("slowdown_short") or {}).get("p99")

        match_wave = _text_match_rate(lora_base["texts"], lora_p2["texts"])
        match_noise = _text_match_rate(lora_base["texts"], lora_base_repeat["texts"])
        p2_info = wave_report.get("phase2", {})

        row: dict[str, Any] = {
            "repeat_index": idx,
            "base_ttft_short_p99_ms": lora_base["ttft_short_p99_ms"],
            "wave_ttft_short_p99_ms": lora_p2["ttft_short_p99_ms"],
            "ttft_improve_ratio": _ratio(lora_base["ttft_short_p99_ms"], lora_p2["ttft_short_p99_ms"]),
            "base_slowdown_short_p99": base_slow,
            "wave_slowdown_short_p99": wave_slow,
            "slowdown_improve_ratio": _ratio(base_slow, wave_slow),
            "base_round_wall_ms": lora_base.get("round_wall_ms"),
            "wave_round_wall_ms": lora_p2.get("round_wall_ms"),
            "round_wall_improve_ratio": _ratio(lora_base.get("round_wall_ms"), lora_p2.get("round_wall_ms")),
            "baseline_noise_match_rate": match_noise,
            "baseline_noise_error_rate": 1.0 - match_noise,
            "wave_match_rate": match_wave,
            "wave_error_rate": 1.0 - match_wave,
            "incremental_error_rate": (1.0 - match_wave) - (1.0 - match_noise),
            "phase2_apply_ratio": p2_info.get("apply_ratio"),
            "phase2_applied": p2_info.get("applied"),
            "phase2_attempts": p2_info.get("attempts"),
            "baseline_timed_out": lora_base.get("timed_out"),
            "wave_timed_out": lora_p2.get("timed_out"),
        }
        rows.append(row)

    if include_strict:
        strict_rows = strict_rows or []
        for idx, (lora_base, lora_base_repeat, lora_p2_strict) in enumerate(
            zip(base_rows, base_repeat_rows, strict_rows)
        ):
            strict_report = lora_p2_strict["hook_report"] or {}
            base_slow = ((lora_base["hook_report"] or {}).get("slowdown_short") or {}).get("p99")
            strict_slow = (strict_report.get("slowdown_short") or {}).get("p99")
            strict_info = strict_report.get("phase2", {})
            strict_match = _text_match_rate(lora_base["texts"], lora_p2_strict["texts"])
            match_noise = _text_match_rate(lora_base["texts"], lora_base_repeat["texts"])
            strict_eval_rows.append(
                {
                    "repeat_index": idx,
                    "strict_ttft_short_p99_ms": lora_p2_strict["ttft_short_p99_ms"],
                    "strict_ttft_improve_ratio": _ratio(
                        lora_base["ttft_short_p99_ms"],
                        lora_p2_strict["ttft_short_p99_ms"],
                    ),
                    "strict_slowdown_short_p99": strict_slow,
                    "strict_slowdown_improve_ratio": _ratio(base_slow, strict_slow),
                    "strict_round_wall_ms": lora_p2_strict.get("round_wall_ms"),
                    "strict_round_wall_improve_ratio": _ratio(
                        lora_base.get("round_wall_ms"),
                        lora_p2_strict.get("round_wall_ms"),
                    ),
                    "strict_match_rate": strict_match,
                    "strict_error_rate": 1.0 - strict_match,
                    "strict_incremental_error_rate": (1.0 - strict_match) - (1.0 - match_noise),
                    "strict_phase2_apply_ratio": strict_info.get("apply_ratio"),
                    "strict_phase2_applied": strict_info.get("applied"),
                    "strict_phase2_attempts": strict_info.get("attempts"),
                }
            )
    return {
        "rows": rows,
        "summary": {
            "ttft_improve_ratio": _stats([r.get("ttft_improve_ratio") for r in rows]),
            "slowdown_improve_ratio": _stats([r.get("slowdown_improve_ratio") for r in rows]),
            "round_wall_improve_ratio": _stats([r.get("round_wall_improve_ratio") for r in rows]),
            "wave_error_rate": _stats([r.get("wave_error_rate") for r in rows]),
            "baseline_noise_error_rate": _stats([r.get("baseline_noise_error_rate") for r in rows]),
            "incremental_error_rate": _stats([r.get("incremental_error_rate") for r in rows]),
            "phase2_apply_ratio": _stats([r.get("phase2_apply_ratio") for r in rows]),
        },
        "strict_rows": strict_eval_rows,
        "strict_summary": {
            "ttft_improve_ratio": _stats([r.get("strict_ttft_improve_ratio") for r in strict_eval_rows]),
            "slowdown_improve_ratio": _stats([r.get("strict_slowdown_improve_ratio") for r in strict_eval_rows]),
            "round_wall_improve_ratio": _stats([r.get("strict_round_wall_improve_ratio") for r in strict_eval_rows]),
            "error_rate": _stats([r.get("strict_error_rate") for r in strict_eval_rows]),
            "incremental_error_rate": _stats([r.get("strict_incremental_error_rate") for r in strict_eval_rows]),
            "apply_ratio": _stats([r.get("strict_phase2_apply_ratio") for r in strict_eval_rows]),
        },
    }


def _raw_mode_summary(rows: list[dict[str, Any]]) -> dict[str, dict[str, Optional[float]]]:
    return {
        "ttft_short_p99_ms": _stats([r.get("ttft_short_p99_ms") for r in rows]),
        "round_wall_ms": _stats([r.get("round_wall_ms") for r in rows]),
        "timeout_rate": _stats([1.0 if r.get("timed_out") else 0.0 for r in rows]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Repeated Wave-Slice evaluation with warmup/repeats and error rates.",
    )
    parser.add_argument(
        "--model-path",
        default="/home/onceas/.cache/huggingface/hub/models--mistralai--Mistral-7B-v0.1/snapshots/27d67f1b5f57dc0953326b2601d68371d40ea8da",
    )
    parser.add_argument("--model-name", default="Mistral-7B-v0.1")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--timeout-sec", type=int, default=240)
    parser.add_argument("--warmup-iters", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--short-repeat", type=int, default=16)
    parser.add_argument("--short-a-repeat", type=int, default=None)
    parser.add_argument("--short-b-repeat", type=int, default=None)
    parser.add_argument("--long-repeat", type=int, default=320)
    parser.add_argument("--max-model-len", type=int, default=3072)
    parser.add_argument("--max-num-batched-tokens", type=int, default=1536)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.60)
    parser.add_argument(
        "--phase1-objective-mode",
        choices=["fair_escape", "pure_gain"],
        default="fair_escape",
    )
    parser.add_argument(
        "--phase1-baseline-mode",
        choices=["chunked", "no_chunk", "both"],
        default="both",
        help="Phase-I controls: chunked baseline only, no-chunk baseline only, or both.",
    )
    parser.add_argument(
        "--phase1-ingress-target-chunk",
        type=int,
        default=384,
        help="Authoritative ingress target chunk for Phase-I exact-chunk experiments.",
    )
    parser.add_argument(
        "--phase1-gamma",
        type=float,
        default=2.0,
        help="Penalty amplification gamma for Phase-I scheduler.",
    )
    parser.add_argument(
        "--phase1-ingress-direct-authoritative",
        action="store_true",
        default=False,
        help="Enable ingress direct authoritative chunk override for Phase-I.",
    )
    parser.add_argument(
        "--phase1-ingress-exact-chunk",
        action="store_true",
        default=False,
        help="When authoritative ingress is enabled, use the exact target chunk instead of bucket-down mapping.",
    )
    parser.add_argument("--include-strict", action="store_true")
    parser.add_argument(
        "--include-phase12",
        action="store_true",
        help="Also run the combined Phase-I + Phase-II LoRA series.",
    )
    parser.add_argument(
        "--phase2-dispatch-mode",
        choices=["synchronized", "async_experimental"],
        default="synchronized",
    )
    parser.add_argument(
        "--adapter-a",
        default="/tmp/waveslice_synthetic_adapters/mistral-7b-v0.1/adapter_rank8_seed7",
    )
    parser.add_argument(
        "--adapter-b",
        default="/tmp/waveslice_synthetic_adapters/mistral-7b-v0.1/adapter_rank16_seed11",
    )
    parser.add_argument(
        "--out-json",
        default=None,
        help="Optional output json path. Default: results/waveslice_repeated_eval_<ts>.json",
    )
    parser.add_argument(
        "--skip-phase2",
        action="store_true",
        help="Run Phase-I only and skip LoRA Phase-II blocks.",
    )
    parser.add_argument(
        "--requests-json",
        default="",
        help="Optional JSON file containing non-LoRA requests.",
    )
    parser.add_argument(
        "--lora-requests-json",
        default="",
        help="Optional JSON file containing LoRA requests.",
    )
    args = parser.parse_args()

    short_a_repeat = args.short_a_repeat if args.short_a_repeat is not None else args.short_repeat
    short_b_repeat = args.short_b_repeat if args.short_b_repeat is not None else args.short_repeat

    max_prompt_tokens = max(16, args.max_model_len - args.max_new_tokens - 16)
    if args.requests_json.strip():
        reqs = _load_reqs_json(args.requests_json)
        tok_lens = _measure_input_tokens(
            args.model_path,
            reqs,
            trust_remote_code=args.trust_remote_code,
        )
        fitted_long_repeat = args.long_repeat
    else:
        reqs, tok_lens, fitted_long_repeat = _fit_requests_to_context(
            model_path=args.model_path,
            short_a_repeat=short_a_repeat,
            short_b_repeat=short_b_repeat,
            long_repeat=args.long_repeat,
            max_prompt_tokens=max_prompt_tokens,
            trust_remote_code=args.trust_remote_code,
        )
    if args.lora_requests_json.strip():
        lora_reqs = _load_reqs_json(args.lora_requests_json)
        lora_tok_lens = _measure_input_tokens(
            args.model_path,
            lora_reqs,
            trust_remote_code=args.trust_remote_code,
        )
        fitted_lora_long_repeat = args.long_repeat
    else:
        lora_reqs, lora_tok_lens, fitted_lora_long_repeat = _fit_lora_requests_to_context(
            model_path=args.model_path,
            short_a_repeat=short_a_repeat,
            short_b_repeat=short_b_repeat,
            long_repeat=args.long_repeat,
            max_prompt_tokens=max_prompt_tokens,
            trust_remote_code=args.trust_remote_code,
        )
    short_lens = [v for k, v in tok_lens.items() if "short" in k]
    long_lens = [v for k, v in tok_lens.items() if "long" in k]

    print("[Eval] Request token lengths")
    print(f"  per_request={tok_lens}")
    print(f"  short_range=[{min(short_lens)}, {max(short_lens)}], long_range=[{min(long_lens)}, {max(long_lens)}]")
    print(f"  decode_max_new_tokens={args.max_new_tokens}")
    if fitted_long_repeat != args.long_repeat:
        print(f"  long_repeat auto-adjusted: {args.long_repeat} -> {fitted_long_repeat}")
    print("[Eval] LoRA request token lengths")
    print(f"  per_request={lora_tok_lens}")
    if fitted_lora_long_repeat != args.long_repeat:
        print(f"  lora_long_repeat auto-adjusted: {args.long_repeat} -> {fitted_lora_long_repeat}")

    if not (os.path.exists(args.adapter_a) and os.path.exists(args.adapter_b)):
        print("[Eval] adapters not found; cannot run Phase-II LoRA repeated test.")
        print(f"  expected A={args.adapter_a}")
        print(f"  expected B={args.adapter_b}")
        return 1

    print(f"[Eval] warmup_iters={args.warmup_iters}, repeats={args.repeats}")

    need_chunked_baseline = args.phase1_baseline_mode in {"chunked", "both"}
    need_no_chunk_baseline = args.phase1_baseline_mode in {"no_chunk", "both"}

    phase1_base_rounds: list[dict[str, Any]] = []
    phase1_base_repeat_rounds: list[dict[str, Any]] = []
    if need_chunked_baseline:
        print("[Eval] Running Phase-I chunked baseline series")
        phase1_base_rounds = _run_mode_series(
            model_path=args.model_path,
            model_name=args.model_name,
            reqs=reqs,
            max_new_tokens=args.max_new_tokens,
            timeout_sec=args.timeout_sec,
            mode="baseline",
            enable_lora=False,
            warmup_iters=args.warmup_iters,
            repeats=args.repeats,
            max_num_batched_tokens=args.max_num_batched_tokens,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
            phase1_objective_mode=args.phase1_objective_mode,
            phase1_gamma=args.phase1_gamma,
            phase1_ingress_target_chunk=args.phase1_ingress_target_chunk,
            phase1_ingress_direct_authoritative=args.phase1_ingress_direct_authoritative,
            phase1_ingress_exact_chunk=args.phase1_ingress_exact_chunk,
            enable_chunked_prefill=True,
            trust_remote_code=args.trust_remote_code,
        )
        print("[Eval] Running Phase-I chunked baseline noise series")
        phase1_base_repeat_rounds = _run_mode_series(
            model_path=args.model_path,
            model_name=args.model_name,
            reqs=reqs,
            max_new_tokens=args.max_new_tokens,
            timeout_sec=args.timeout_sec,
            mode="baseline",
            enable_lora=False,
            warmup_iters=args.warmup_iters,
            repeats=args.repeats,
            max_num_batched_tokens=args.max_num_batched_tokens,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
            phase1_objective_mode=args.phase1_objective_mode,
            phase1_gamma=args.phase1_gamma,
            phase1_ingress_target_chunk=args.phase1_ingress_target_chunk,
            phase1_ingress_direct_authoritative=args.phase1_ingress_direct_authoritative,
            phase1_ingress_exact_chunk=args.phase1_ingress_exact_chunk,
            enable_chunked_prefill=True,
            trust_remote_code=args.trust_remote_code,
        )

    phase1_no_chunk_rounds: list[dict[str, Any]] = []
    phase1_no_chunk_repeat_rounds: list[dict[str, Any]] = []
    if need_no_chunk_baseline:
        print("[Eval] Running Phase-I no-chunk baseline series")
        phase1_no_chunk_rounds = _run_mode_series(
            model_path=args.model_path,
            model_name=args.model_name,
            reqs=reqs,
            max_new_tokens=args.max_new_tokens,
            timeout_sec=args.timeout_sec,
            mode="baseline",
            enable_lora=False,
            warmup_iters=args.warmup_iters,
            repeats=args.repeats,
            max_num_batched_tokens=args.max_num_batched_tokens,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
            phase1_objective_mode=args.phase1_objective_mode,
            phase1_gamma=args.phase1_gamma,
            phase1_ingress_target_chunk=args.phase1_ingress_target_chunk,
            phase1_ingress_direct_authoritative=args.phase1_ingress_direct_authoritative,
            phase1_ingress_exact_chunk=args.phase1_ingress_exact_chunk,
            enable_chunked_prefill=False,
            trust_remote_code=args.trust_remote_code,
        )
        print("[Eval] Running Phase-I no-chunk baseline noise series")
        phase1_no_chunk_repeat_rounds = _run_mode_series(
            model_path=args.model_path,
            model_name=args.model_name,
            reqs=reqs,
            max_new_tokens=args.max_new_tokens,
            timeout_sec=args.timeout_sec,
            mode="baseline",
            enable_lora=False,
            warmup_iters=args.warmup_iters,
            repeats=args.repeats,
            max_num_batched_tokens=args.max_num_batched_tokens,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
            phase1_objective_mode=args.phase1_objective_mode,
            phase1_gamma=args.phase1_gamma,
            phase1_ingress_target_chunk=args.phase1_ingress_target_chunk,
            phase1_ingress_direct_authoritative=args.phase1_ingress_direct_authoritative,
            phase1_ingress_exact_chunk=args.phase1_ingress_exact_chunk,
            enable_chunked_prefill=False,
            trust_remote_code=args.trust_remote_code,
        )

    if not phase1_base_rounds:
        phase1_base_rounds = phase1_no_chunk_rounds
        phase1_base_repeat_rounds = phase1_no_chunk_repeat_rounds

    print("[Eval] Running Phase-I Wave-Slice series")
    phase1_wave_rounds = _run_mode_series(
        model_path=args.model_path,
        model_name=args.model_name,
        reqs=reqs,
        max_new_tokens=args.max_new_tokens,
        timeout_sec=args.timeout_sec,
        mode="phase1_only",
        enable_lora=False,
        warmup_iters=args.warmup_iters,
        repeats=args.repeats,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        phase1_objective_mode=args.phase1_objective_mode,
        phase1_gamma=args.phase1_gamma,
        phase1_ingress_target_chunk=args.phase1_ingress_target_chunk,
        phase1_ingress_direct_authoritative=args.phase1_ingress_direct_authoritative,
        phase1_ingress_exact_chunk=args.phase1_ingress_exact_chunk,
        enable_chunked_prefill=True,
        trust_remote_code=args.trust_remote_code,
    )
    phase1 = _run_phase1_pair(
        base_rows=phase1_base_rounds,
        base_repeat_rows=phase1_base_repeat_rounds,
        wave_rows=phase1_wave_rounds,
    )
    phase1_no_chunk_control = None
    if need_chunked_baseline and need_no_chunk_baseline:
        phase1_no_chunk_control = _run_phase1_pair(
            base_rows=phase1_no_chunk_rounds,
            base_repeat_rows=phase1_no_chunk_repeat_rounds,
            wave_rows=phase1_base_rounds,
        )

    for i, row in enumerate(phase1["rows"], start=1):
        print(
            f"[Repeat] {i}/{args.repeats} Phase-I "
            f"ttft_gain={row.get('ttft_improve_ratio')} "
            f"wall_gain={row.get('round_wall_improve_ratio')} "
            f"error_rate={row.get('error_rate')} "
            f"base_noise_err={row.get('baseline_noise_error_rate')} "
            f"apply_ratio={row.get('scheduler_apply_ratio')}"
        )

    if args.skip_phase2:
        ts = int(time.time())
        result = {
            "phase1": phase1["summary"],
            "phase1_rows": phase1["rows"],
            "phase1_chunked_vs_no_chunk": (
                phase1_no_chunk_control["summary"] if phase1_no_chunk_control is not None else None
            ),
            "phase1_chunked_vs_no_chunk_rows": (
                phase1_no_chunk_control["rows"] if phase1_no_chunk_control is not None else None
            ),
            "request_token_lengths": tok_lens,
            "model_name": args.model_name,
            "model_path": args.model_path,
            "phase1_objective_mode": args.phase1_objective_mode,
            "phase1_baseline_mode": args.phase1_baseline_mode,
        }
        out_path = args.out_json or os.path.join(
            "results",
            f"waveslice_phase1_only_eval_{ts}.json",
        )
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print("\n[Summary] Phase-I")
        for key, stats in phase1["summary"].items():
            print(f"  {key}: {stats}")
        if phase1_no_chunk_control is not None:
            print("\n[Summary] Chunked-vs-NoChunk")
            for key, stats in phase1_no_chunk_control["summary"].items():
                print(f"  {key}: {stats}")
        print(f"\n[Saved] {out_path}")
        return 0

    print("[Eval] Running Phase-II baseline series A")
    phase2_base_rounds = _run_mode_series(
        model_path=args.model_path,
        model_name=args.model_name,
        reqs=lora_reqs,
        max_new_tokens=args.max_new_tokens,
        timeout_sec=args.timeout_sec,
        mode="baseline_lora_compat",
        enable_lora=True,
        warmup_iters=args.warmup_iters,
        repeats=args.repeats,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        phase1_objective_mode=args.phase1_objective_mode,
        phase1_gamma=args.phase1_gamma,
        phase1_ingress_target_chunk=args.phase1_ingress_target_chunk,
        phase1_ingress_direct_authoritative=args.phase1_ingress_direct_authoritative,
        phase1_ingress_exact_chunk=args.phase1_ingress_exact_chunk,
        enable_chunked_prefill=True,
        adapter_a=args.adapter_a,
        adapter_b=args.adapter_b,
        phase2_dispatch_mode=args.phase2_dispatch_mode,
        trust_remote_code=args.trust_remote_code,
    )
    print("[Eval] Running Phase-II baseline series B (noise floor)")
    phase2_base_repeat_rounds = _run_mode_series(
        model_path=args.model_path,
        model_name=args.model_name,
        reqs=lora_reqs,
        max_new_tokens=args.max_new_tokens,
        timeout_sec=args.timeout_sec,
        mode="baseline_lora_compat",
        enable_lora=True,
        warmup_iters=args.warmup_iters,
        repeats=args.repeats,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        phase1_objective_mode=args.phase1_objective_mode,
        phase1_gamma=args.phase1_gamma,
        phase1_ingress_target_chunk=args.phase1_ingress_target_chunk,
        phase1_ingress_direct_authoritative=args.phase1_ingress_direct_authoritative,
        phase1_ingress_exact_chunk=args.phase1_ingress_exact_chunk,
        enable_chunked_prefill=True,
        adapter_a=args.adapter_a,
        adapter_b=args.adapter_b,
        phase2_dispatch_mode=args.phase2_dispatch_mode,
        trust_remote_code=args.trust_remote_code,
    )
    print("[Eval] Running Phase-II Wave-Slice series")
    phase2_wave_rounds = _run_mode_series(
        model_path=args.model_path,
        model_name=args.model_name,
        reqs=lora_reqs,
        max_new_tokens=args.max_new_tokens,
        timeout_sec=args.timeout_sec,
        mode="phase2_lora",
        enable_lora=True,
        warmup_iters=args.warmup_iters,
        repeats=args.repeats,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        phase1_objective_mode=args.phase1_objective_mode,
        phase1_gamma=args.phase1_gamma,
        phase1_ingress_target_chunk=args.phase1_ingress_target_chunk,
        phase1_ingress_direct_authoritative=args.phase1_ingress_direct_authoritative,
        phase1_ingress_exact_chunk=args.phase1_ingress_exact_chunk,
        enable_chunked_prefill=True,
        adapter_a=args.adapter_a,
        adapter_b=args.adapter_b,
        phase2_dispatch_mode=args.phase2_dispatch_mode,
        trust_remote_code=args.trust_remote_code,
    )
    phase2_strict_rounds: Optional[list[dict[str, Any]]] = None
    if args.include_strict:
        print("[Eval] Running Phase-II strict Wave-Slice series")
        phase2_strict_rounds = _run_mode_series(
            model_path=args.model_path,
            model_name=args.model_name,
            reqs=lora_reqs,
            max_new_tokens=args.max_new_tokens,
            timeout_sec=args.timeout_sec,
            mode="phase2_lora_strict",
            enable_lora=True,
            warmup_iters=args.warmup_iters,
            repeats=args.repeats,
            max_num_batched_tokens=args.max_num_batched_tokens,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
            phase1_objective_mode=args.phase1_objective_mode,
            phase1_gamma=args.phase1_gamma,
            phase1_ingress_target_chunk=args.phase1_ingress_target_chunk,
            phase1_ingress_direct_authoritative=args.phase1_ingress_direct_authoritative,
            phase1_ingress_exact_chunk=args.phase1_ingress_exact_chunk,
            enable_chunked_prefill=True,
            adapter_a=args.adapter_a,
            adapter_b=args.adapter_b,
            phase2_dispatch_mode=args.phase2_dispatch_mode,
            trust_remote_code=args.trust_remote_code,
        )
    phase2 = _run_phase2_block(
        base_rows=phase2_base_rounds,
        base_repeat_rows=phase2_base_repeat_rounds,
        wave_rows=phase2_wave_rounds,
        strict_rows=phase2_strict_rounds,
        include_strict=args.include_strict,
    )
    phase12: Optional[dict[str, Any]] = None
    if args.include_phase12:
        print("[Eval] Running Phase-I + Phase-II Wave-Slice series")
        phase12_rounds = _run_mode_series(
            model_path=args.model_path,
            model_name=args.model_name,
            reqs=lora_reqs,
            max_new_tokens=args.max_new_tokens,
            timeout_sec=args.timeout_sec,
            mode="phase12_lora",
            enable_lora=True,
            warmup_iters=args.warmup_iters,
            repeats=args.repeats,
            max_num_batched_tokens=args.max_num_batched_tokens,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
            phase1_objective_mode=args.phase1_objective_mode,
            phase1_gamma=args.phase1_gamma,
            phase1_ingress_target_chunk=args.phase1_ingress_target_chunk,
            phase1_ingress_direct_authoritative=args.phase1_ingress_direct_authoritative,
            phase1_ingress_exact_chunk=args.phase1_ingress_exact_chunk,
            enable_chunked_prefill=True,
            adapter_a=args.adapter_a,
            adapter_b=args.adapter_b,
            phase2_dispatch_mode=args.phase2_dispatch_mode,
            trust_remote_code=args.trust_remote_code,
        )
        phase12 = _run_phase2_block(
            base_rows=phase2_base_rounds,
            base_repeat_rows=phase2_base_repeat_rounds,
            wave_rows=phase12_rounds,
            strict_rows=None,
            include_strict=False,
        )

    for i, p2 in enumerate(phase2["rows"], start=1):
        print(
            f"[Repeat] {i}/{args.repeats} Phase-II "
            f"ttft_gain={p2.get('ttft_improve_ratio')} "
            f"slow_gain={p2.get('slowdown_improve_ratio')} "
            f"wall_gain={p2.get('round_wall_improve_ratio')} "
            f"err={p2.get('wave_error_rate')} "
            f"base_noise_err={p2.get('baseline_noise_error_rate')} "
            f"apply_ratio={p2.get('phase2_apply_ratio')}"
        )

    summary = {
        "config": {
            "model_name": args.model_name,
            "model_path": args.model_path,
            "max_new_tokens": args.max_new_tokens,
            "timeout_sec": args.timeout_sec,
            "warmup_iters": args.warmup_iters,
            "repeats": args.repeats,
            "short_repeat": args.short_repeat,
            "short_a_repeat": short_a_repeat,
            "short_b_repeat": short_b_repeat,
            "long_repeat": args.long_repeat,
            "max_model_len": args.max_model_len,
            "max_num_batched_tokens": args.max_num_batched_tokens,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "phase1_objective_mode": args.phase1_objective_mode,
            "phase1_baseline_mode": args.phase1_baseline_mode,
            "include_strict": args.include_strict,
            "phase2_dispatch_mode": args.phase2_dispatch_mode,
            "adapter_a": args.adapter_a,
            "adapter_b": args.adapter_b,
        },
        "token_lengths": tok_lens,
        "phase1_baseline_chunked_raw": _raw_mode_summary(phase1_base_rounds) if need_chunked_baseline else None,
        "phase1_baseline_no_chunk_raw": _raw_mode_summary(phase1_no_chunk_rounds) if need_no_chunk_baseline else None,
        "phase1": phase1["summary"],
        "phase2": phase2["summary"],
        "per_repeat": {
            "phase1": phase1["rows"],
            "phase2": phase2["rows"],
        },
    }
    if phase1_no_chunk_control is not None:
        summary["phase1_chunked_vs_no_chunk"] = phase1_no_chunk_control["summary"]
        summary["per_repeat"]["phase1_chunked_vs_no_chunk"] = phase1_no_chunk_control["rows"]

    if args.include_strict:
        summary["phase2_strict"] = phase2["strict_summary"]
        summary["per_repeat"]["phase2_strict"] = phase2["strict_rows"]
    if phase12 is not None:
        summary["phase12"] = phase12["summary"]
        summary["per_repeat"]["phase12"] = phase12["rows"]

    print("\n[Summary] Phase-I")
    if summary.get("phase1_baseline_no_chunk_raw") is not None:
        print(f"  no_chunk_ttft_short_p99_ms={summary['phase1_baseline_no_chunk_raw']['ttft_short_p99_ms']}")
        print(f"  no_chunk_round_wall_ms={summary['phase1_baseline_no_chunk_raw']['round_wall_ms']}")
    if summary.get("phase1_baseline_chunked_raw") is not None:
        print(f"  chunked_ttft_short_p99_ms={summary['phase1_baseline_chunked_raw']['ttft_short_p99_ms']}")
        print(f"  chunked_round_wall_ms={summary['phase1_baseline_chunked_raw']['round_wall_ms']}")
    if summary.get("phase1_chunked_vs_no_chunk") is not None:
        print(f"  chunked_vs_no_chunk_ttft_ratio={summary['phase1_chunked_vs_no_chunk']['ttft_improve_ratio']}")
        print(f"  chunked_vs_no_chunk_wall_ratio={summary['phase1_chunked_vs_no_chunk']['round_wall_improve_ratio']}")
    print(f"  ttft_improve_ratio={summary['phase1']['ttft_improve_ratio']}")
    print(f"  round_wall_improve_ratio={summary['phase1']['round_wall_improve_ratio']}")
    print(f"  error_rate={summary['phase1']['error_rate']}")
    print(f"  baseline_noise_error_rate={summary['phase1']['baseline_noise_error_rate']}")
    print(f"  incremental_error_rate={summary['phase1']['incremental_error_rate']}")
    print(f"  scheduler_apply_ratio={summary['phase1']['scheduler_apply_ratio']}")
    print(f"  baseline_chunk_avg={summary['phase1']['baseline_chunk_avg']}")
    print(f"  chosen_chunk_avg={summary['phase1']['chosen_chunk_avg']}")
    print(f"  chosen_vs_baseline_ratio_avg={summary['phase1']['chosen_vs_baseline_ratio_avg']}")
    print(f"  explicit_plan_ratio={summary['phase1']['explicit_plan_ratio']}")
    print(f"  rewrite_apply_ratio={summary['phase1']['rewrite_apply_ratio']}")
    print(f"  rewrite_old_chunk_avg={summary['phase1']['rewrite_old_chunk_avg']}")
    print(f"  rewrite_new_chunk_avg={summary['phase1']['rewrite_new_chunk_avg']}")
    print(f"  rewrite_token_delta_avg={summary['phase1']['rewrite_token_delta_avg']}")
    print(f"  virtual_cap_apply_ratio={summary['phase1']['virtual_cap_apply_ratio']}")
    print(f"  virtual_cap_old_avg={summary['phase1']['virtual_cap_old_avg']}")
    print(f"  virtual_cap_new_avg={summary['phase1']['virtual_cap_new_avg']}")

    print("\n[Summary] Phase-II")
    print(f"  ttft_improve_ratio={summary['phase2']['ttft_improve_ratio']}")
    print(f"  slowdown_improve_ratio={summary['phase2']['slowdown_improve_ratio']}")
    print(f"  round_wall_improve_ratio={summary['phase2']['round_wall_improve_ratio']}")
    print(f"  wave_error_rate={summary['phase2']['wave_error_rate']}")
    print(f"  baseline_noise_error_rate={summary['phase2']['baseline_noise_error_rate']}")
    print(f"  incremental_error_rate={summary['phase2']['incremental_error_rate']}")
    print(f"  phase2_apply_ratio={summary['phase2']['phase2_apply_ratio']}")

    if args.include_strict:
        print("\n[Summary] Phase-II Strict")
        print(f"  ttft_improve_ratio={summary['phase2_strict']['ttft_improve_ratio']}")
        print(f"  slowdown_improve_ratio={summary['phase2_strict']['slowdown_improve_ratio']}")
        print(f"  round_wall_improve_ratio={summary['phase2_strict']['round_wall_improve_ratio']}")
        print(f"  error_rate={summary['phase2_strict']['error_rate']}")
        print(f"  incremental_error_rate={summary['phase2_strict']['incremental_error_rate']}")
        print(f"  apply_ratio={summary['phase2_strict']['apply_ratio']}")

    if phase12 is not None:
        print("\n[Summary] Phase-I + Phase-II")
        print(f"  ttft_improve_ratio={summary['phase12']['ttft_improve_ratio']}")
        print(f"  slowdown_improve_ratio={summary['phase12']['slowdown_improve_ratio']}")
        print(f"  round_wall_improve_ratio={summary['phase12']['round_wall_improve_ratio']}")
        print(f"  wave_error_rate={summary['phase12']['wave_error_rate']}")
        print(f"  baseline_noise_error_rate={summary['phase12']['baseline_noise_error_rate']}")
        print(f"  incremental_error_rate={summary['phase12']['incremental_error_rate']}")
        print(f"  phase2_apply_ratio={summary['phase12']['phase2_apply_ratio']}")

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_json = args.out_json or f"results/waveslice_repeated_eval_{ts}.json"
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n[Output] {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
