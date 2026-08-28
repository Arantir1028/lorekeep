"""Evaluate Phase I/II latency, slowdown, and output quality over repeated vLLM runs."""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("VLLM_NO_USAGE_STATS", "1")
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from eval_config import build_wave_slice_config
from eval_output import build_summary, print_summary, write_summary_json
from eval_support import (
    Req,
    bool_arg_from_argv,
    load_reqs_json,
    measure_input_tokens,
    percentile as _percentile,
    run_phase1_pair,
    run_phase2_block,
    str_arg_from_argv,
)

from experiments.catalog import safe_key
from experiments.model_assets import ensure_adapters
from experiments.result_io import write_json
from experiments.run_frozen_eval_config import apply_eval_config
from tools.experiment_lock import gpu_experiment_lock
from waveslice import get_wave_slice_metrics, reset_wave_slice_metrics
from waveslice.vllm.bootstrap import bootstrap_vllm_runtime, shutdown_vllm_engine
from waveslice.vllm.integration import deactivate_wave_slice

bootstrap_vllm_runtime()


def _ensure_eval_adapters(
    *, model_path: str, adapters_root: str, trust_remote_code: bool
) -> tuple[str, str]:
    model_key = safe_key(model_path)
    out_dir = os.path.join(adapters_root, model_key)
    os.makedirs(out_dir, exist_ok=True)
    return ensure_adapters(
        base_model_path=model_path, out_dir=out_dir, trust_remote_code=trust_remote_code
    )


def _cleanup_engine(engine: Any | None) -> None:
    shutdown_vllm_engine(engine)
    deactivate_wave_slice()
    gc.collect()
    import torch

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _build_engine(
    args: argparse.Namespace,
    *,
    mode: str,
    enable_lora: bool,
    enable_chunked_prefill: bool,
    adapter_a: str = "",
    adapter_b: str = "",
) -> tuple[Any, dict[str, Any]]:
    from vllm.engine.llm_engine import LLMEngine

    from waveslice import EngineArgs

    options = vars(args).copy()
    options.pop("model_name", None)
    wave_slice_config = build_wave_slice_config(
        model_name=args.model_name,
        mode=mode,
        **options,
    )
    effective_batched_tokens = int(args.max_num_batched_tokens)
    if not enable_chunked_prefill:
        effective_batched_tokens = max(effective_batched_tokens, int(args.max_model_len))
    engine_args = EngineArgs(
        model=args.model_path,
        trust_remote_code=args.trust_remote_code,
        seed=0,
        enable_lora=enable_lora,
        max_loras=max(2, len([path for path in (adapter_a, adapter_b) if path]))
        if enable_lora
        else 1,
        max_lora_rank=32,
        max_num_batched_tokens=effective_batched_tokens,
        max_num_partial_prefills=max(1, int(args.max_num_partial_prefills)),
        max_long_partial_prefills=max(1, int(args.max_long_partial_prefills)),
        enable_chunked_prefill=enable_chunked_prefill,
        disable_sliding_window=True,
        enforce_eager=True,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_wave_slice=wave_slice_config is not None,
        wave_slice_config=wave_slice_config,
    )
    engine = LLMEngine.from_engine_args(engine_args)
    lora_map: dict[str, Any] = {}
    if enable_lora:
        from vllm.lora.request import LoRARequest

        if not adapter_a or not adapter_b:
            raise ValueError("LoRA mode requires adapter_a and adapter_b.")
        lora_map = {
            "A": LoRARequest(lora_name="adapter_A", lora_int_id=1, lora_path=adapter_a),
            "B": LoRARequest(lora_name="adapter_B", lora_int_id=2, lora_path=adapter_b),
        }
    return (engine, lora_map)


def _run_round(
    *,
    engine: Any,
    reqs: list[Req],
    max_new_tokens: int,
    ignore_eos: bool,
    timeout_sec: int,
    enable_lora: bool,
    lora_map: dict[str, Any],
    run_tag: str,
) -> dict[str, Any]:
    from vllm.sampling_params import SamplingParams

    sampling = SamplingParams(max_tokens=max_new_tokens, temperature=0.0, ignore_eos=ignore_eos)
    trackers: dict[str, dict[str, Any]] = {}
    reset_wave_slice_metrics()
    pending_reqs = sorted(
        reqs, key=lambda r: (float(getattr(r, "arrival_offset_s", 0.0) or 0.0), str(r.req_id))
    )
    next_idx = 0
    round_start = time.perf_counter()
    deadline = time.time() + timeout_sec
    while time.time() < deadline and (
        next_idx < len(pending_reqs) or engine.has_unfinished_requests()
    ):
        now = time.perf_counter()
        elapsed_s = now - round_start
        while next_idx < len(pending_reqs):
            r = pending_reqs[next_idx]
            if float(getattr(r, "arrival_offset_s", 0.0) or 0.0) > elapsed_s:
                break
            rid = f"{run_tag}:{r.req_id}"
            if enable_lora:
                engine.add_request(
                    rid, r.prompt, sampling, lora_request=lora_map[r.lora_tag or "A"]
                )
            else:
                engine.add_request(rid, r.prompt, sampling)
            trackers[rid] = {
                "orig_req_id": r.req_id,
                "arrival_s": now,
                "round_start_s": round_start,
                "scheduled_arrival_offset_s": float(getattr(r, "arrival_offset_s", 0.0) or 0.0),
                "first_s": None,
                "finish_s": None,
                "is_short": r.is_short,
                "text": "",
            }
            next_idx += 1
        if not engine.has_unfinished_requests():
            if next_idx < len(pending_reqs):
                next_arrival = float(
                    getattr(pending_reqs[next_idx], "arrival_offset_s", 0.0) or 0.0
                )
                sleep_s = max(0.0, min(0.01, next_arrival - (time.perf_counter() - round_start)))
                if sleep_s > 0:
                    time.sleep(sleep_s)
            continue
        outputs = engine.step()
        now = time.perf_counter()
        for out in outputs:
            rid = out.request_id
            if rid not in trackers:
                continue
            payload = out.outputs[0] if out.outputs else None
            tok_count = len(payload.token_ids) if payload else 0
            txt = str(payload.text or "") if payload else ""
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
    unfinished = [rid for rid, tracker in trackers.items() if tracker["finish_s"] is None]
    if unfinished:
        engine.abort_request(unfinished)
    timed_out = bool(unfinished or next_idx < len(pending_reqs))
    report = get_wave_slice_metrics(reset=True)
    result = {
        "texts": {tr["orig_req_id"]: tr["text"] for tr in trackers.values()},
        "request_timings": {
            tr["orig_req_id"]: {
                "arrival_offset_s": float(tr["scheduled_arrival_offset_s"]),
                "first_latency_ms": (tr["first_s"] - tr["arrival_s"]) * 1000.0
                if tr["first_s"] is not None
                else None,
                "finish_latency_ms": (tr["finish_s"] - tr["arrival_s"]) * 1000.0
                if tr["finish_s"] is not None
                else None,
                "scheduled_first_latency_ms": (
                    (tr["first_s"] - (tr["round_start_s"] + tr["scheduled_arrival_offset_s"]))
                    * 1000.0
                    if tr["first_s"] is not None
                    else None
                ),
                "scheduled_finish_latency_ms": (
                    (tr["finish_s"] - (tr["round_start_s"] + tr["scheduled_arrival_offset_s"]))
                    * 1000.0
                    if tr["finish_s"] is not None
                    else None
                ),
                "is_short": bool(tr["is_short"]),
            }
            for tr in trackers.values()
        },
        "ttft_short_p99_ms": _percentile(ttft_short_ms, 99.0),
        "round_wall_ms": (round_end - round_start) * 1000.0,
        "timed_out": timed_out,
        "finished_requests": finished_count,
        "total_requests": len(pending_reqs),
        "hook_report": report,
    }
    return result


def _run_series(
    args: argparse.Namespace,
    *,
    reqs: list[Req],
    enable_lora: bool,
    mode: str,
    enable_chunked_prefill: bool,
    adapter_a: str = "",
    adapter_b: str = "",
) -> list[dict[str, Any]]:
    engine = None
    try:
        (engine, lora_map) = _build_engine(
            args,
            mode=mode,
            enable_lora=enable_lora,
            enable_chunked_prefill=enable_chunked_prefill,
            adapter_a=adapter_a,
            adapter_b=adapter_b,
        )

        def run(tag: str) -> dict[str, Any]:
            return _run_round(
                engine=engine,
                reqs=reqs,
                max_new_tokens=args.max_new_tokens,
                ignore_eos=args.ignore_eos,
                timeout_sec=args.timeout_sec,
                enable_lora=enable_lora,
                lora_map=lora_map,
                run_tag=tag,
            )

        for index in range(args.warmup_iters):
            run(f"warmup_{mode}_{index}")
        return [run(f"repeat_{mode}_{index}") for index in range(args.repeats)]
    finally:
        _cleanup_engine(engine)


def _eval_args(cli: argparse.Namespace) -> Any:
    from dataclasses import asdict
    from types import SimpleNamespace

    from waveslice.policy import WaveSlicePolicy

    config = json.loads(os.environ.get("WAVESLICE_EVAL_CONFIG_JSON", "{}"))
    args = SimpleNamespace(
        **asdict(WaveSlicePolicy()),
        trust_remote_code=False,
        warmup_iters=2,
        repeats=8,
        timeout_sec=240,
        max_new_tokens=128,
        max_model_len=3072,
        max_num_batched_tokens=1536,
        max_num_partial_prefills=1,
        max_long_partial_prefills=1,
        gpu_memory_utilization=0.6,
        model_name="Mistral-7B-v0.1",
        model_path="mistralai/Mistral-7B-v0.1",
        requests_json="",
        lora_requests_json="",
        adapter_a="",
        adapter_b="",
        adapters_root=os.path.join("results", "synthetic_adapters"),
        no_auto_build_adapters=False,
        ignore_eos=False,
        short_repeat=16,
        short_a_repeat=None,
        short_b_repeat=None,
        long_repeat=320,
        include_phase12=bool(config),
        skip_phase2=False,
        baseline_only=False,
        out_json=None,
        phase1_objective_mode="fair_escape",
        phase1_baseline_mode="both",
        phase1_gamma=2.0,
        phase2_dispatch_mode="synchronized",
        phase2_baseline_enable_chunked_prefill=True,
    )
    apply_eval_config(args, config)
    adapters = dict(config.get("adapters") or {})
    args.no_auto_build_adapters = adapters.get("auto_build") is False or bool(
        args.adapter_a and args.adapter_b
    )
    for key, value in vars(cli).items():
        if key not in {"serialize_gpu_tests", "gpu_lock_path"} and value is not None:
            setattr(args, key, value)
    args.serialize_gpu_tests = cli.serialize_gpu_tests
    args.gpu_lock_path = cli.gpu_lock_path
    return args


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the WaveSlice contract evaluation.")
    for name in (
        "out-json",
        "model-name",
        "model-path",
        "requests-json",
        "lora-requests-json",
        "adapter-a",
        "adapter-b",
    ):
        parser.add_argument(f"--{name}")
    parser.add_argument("--repeats", type=int)
    parser.add_argument("--warmup-iters", type=int)
    for flag in ("baseline-only", "skip-phase2", "include-phase12"):
        parser.add_argument(f"--{flag}", action="store_true", default=None)
    parser.add_argument(
        "--serialize-gpu-tests", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--gpu-lock-path", default="")
    cli = parser.parse_args()
    args = _eval_args(cli)
    if args.baseline_only:
        args.skip_phase2 = True
        if args.phase1_baseline_mode != "chunked":
            parser.error("--baseline-only requires phase1.baseline_mode=chunked")
    short_a = args.short_repeat if args.short_a_repeat is None else args.short_a_repeat
    short_b = args.short_repeat if args.short_b_repeat is None else args.short_b_repeat
    if not args.requests_json:
        parser.error("workload.requests_json is required")
    requests = load_reqs_json(args.requests_json)
    token_lengths = measure_input_tokens(
        args.model_path, requests, trust_remote_code=args.trust_remote_code
    )
    need_adapters = not args.skip_phase2 or args.include_phase12
    if need_adapters and (not args.lora_requests_json):
        parser.error("workload.lora_requests_json is required for LoRA evaluation")
    if need_adapters:
        lora_requests = load_reqs_json(args.lora_requests_json)
        lora_token_lengths = measure_input_tokens(
            args.model_path, lora_requests, trust_remote_code=args.trust_remote_code
        )
    else:
        (lora_requests, lora_token_lengths) = ([], {})
    print(f"[Eval] request_tokens={token_lengths}")
    print(f"[Eval] lora_request_tokens={lora_token_lengths}")
    if need_adapters and (
        not (
            args.adapter_a
            and args.adapter_b
            and os.path.exists(args.adapter_a)
            and os.path.exists(args.adapter_b)
        )
    ):
        if args.no_auto_build_adapters:
            print("[Eval] configured adapters are missing")
            return 1
        (args.adapter_a, args.adapter_b) = _ensure_eval_adapters(
            model_path=args.model_path,
            adapters_root=args.adapters_root,
            trust_remote_code=args.trust_remote_code,
        )

    def series(mode: str, *, lora: bool = False, chunked: bool = True) -> list[dict[str, Any]]:
        return _run_series(
            args,
            reqs=lora_requests if lora else requests,
            enable_lora=lora,
            mode=mode,
            enable_chunked_prefill=chunked,
            adapter_a=args.adapter_a if lora else "",
            adapter_b=args.adapter_b if lora else "",
        )

    need_chunked = args.phase1_baseline_mode in {"chunked", "both"}
    need_no_chunk = args.phase1_baseline_mode in {"no_chunk", "both"}
    chunked = series("baseline") if need_chunked else []
    chunked_repeat = series("baseline") if need_chunked else []
    no_chunk = series("baseline", chunked=False) if need_no_chunk else []
    no_chunk_repeat = series("baseline", chunked=False) if need_no_chunk else []
    if args.baseline_only:
        out = args.out_json or os.path.join(
            "results", f"waveslice_fixed_baseline_{int(time.time())}.json"
        )
        write_json(
            Path(out),
            {
                "schema_version": "waveslice-fixed-baseline-v1",
                "model_name": args.model_name,
                "model_path": args.model_path,
                "request_token_lengths": token_lengths,
                "mode": "baseline",
                "enable_chunked_prefill": True,
                "config": {
                    key: getattr(args, key)
                    for key in (
                        "warmup_iters",
                        "repeats",
                        "max_new_tokens",
                        "max_model_len",
                        "max_num_batched_tokens",
                        "max_num_partial_prefills",
                        "max_long_partial_prefills",
                        "gpu_memory_utilization",
                        "timeout_sec",
                        "phase1_baseline_mode",
                    )
                },
                "chunked_rows": chunked,
            },
        )
        print(f"[Saved] {out}")
        return 0
    (base, base_repeat) = (chunked, chunked_repeat) if chunked else (no_chunk, no_chunk_repeat)
    phase1 = run_phase1_pair(
        base_rows=base, base_repeat_rows=base_repeat, wave_rows=series("phase1_only")
    )
    no_chunk_control = (
        run_phase1_pair(base_rows=no_chunk, base_repeat_rows=no_chunk_repeat, wave_rows=chunked)
        if need_chunked and need_no_chunk
        else None
    )
    if args.skip_phase2:
        out = args.out_json or os.path.join(
            "results", f"waveslice_phase1_only_eval_{int(time.time())}.json"
        )
        write_json(
            Path(out),
            {
                "phase1": phase1["summary"],
                "phase1_rows": phase1["rows"],
                "phase1_chunked_vs_no_chunk": no_chunk_control["summary"]
                if no_chunk_control
                else None,
                "phase1_chunked_vs_no_chunk_rows": no_chunk_control["rows"]
                if no_chunk_control
                else None,
                "request_token_lengths": token_lengths,
                "model_name": args.model_name,
                "model_path": args.model_path,
                "phase1_objective_mode": args.phase1_objective_mode,
                "phase1_baseline_mode": args.phase1_baseline_mode,
            },
        )
        print(f"[Saved] {out}")
        return 0
    phase2_base = series(
        "baseline_lora_compat", lora=True, chunked=bool(args.phase2_baseline_enable_chunked_prefill)
    )
    phase2_repeat = series(
        "baseline_lora_compat", lora=True, chunked=bool(args.phase2_baseline_enable_chunked_prefill)
    )
    phase2 = run_phase2_block(
        base_rows=phase2_base,
        base_repeat_rows=phase2_repeat,
        wave_rows=series("phase2_lora", lora=True),
    )
    phase12 = (
        run_phase2_block(
            base_rows=phase2_base,
            base_repeat_rows=phase2_repeat,
            wave_rows=series("phase12_lora", lora=True),
        )
        if args.include_phase12
        else None
    )
    summary = build_summary(
        args=args,
        short_a_repeat=short_a,
        short_b_repeat=short_b,
        tok_lens=token_lengths,
        need_chunked_baseline=need_chunked,
        need_no_chunk_baseline=need_no_chunk,
        phase1_base_rounds=chunked,
        phase1_no_chunk_rounds=no_chunk,
        phase1=phase1,
        phase2=phase2,
        phase1_no_chunk_control=no_chunk_control,
        phase12=phase12,
    )
    print_summary(summary, include_phase12=args.include_phase12)
    print(f"[Output] {write_summary_json(summary, args.out_json)}")
    return 0


if __name__ == "__main__":
    serialize_gpu_tests = bool_arg_from_argv("serialize-gpu-tests", True)
    gpu_lock_path = str_arg_from_argv("gpu-lock-path", "")
    model_name = str_arg_from_argv("model-name", "unknown-model")
    with gpu_experiment_lock(
        label=f"evaluate:{model_name}", enabled=serialize_gpu_tests, lock_path=gpu_lock_path or None
    ):
        raise SystemExit(main())
