"""Validation entrypoint for Wave-Slice.

This script validates three layers:
1) Code-level correctness (scheduler + metrics core, no vLLM needed).
2) vLLM embedding correctness (monkey patch flags and rollback).
3) Optional live vLLM smoke run (real engine + requests).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from pathlib import Path

from engine.runtime_bootstrap import bootstrap_vllm_runtime
from engine.vllm_hijacker import (
    WaveSlicePolicy,
    WaveSliceMetrics,
    get_wave_slice_metrics,
    inject_wave_slice,
    reset_wave_slice_metrics,
    uninject_wave_slice,
)
from scheduler.wave_scheduler import WaveScheduler
from tools.experiment_lock import gpu_experiment_lock

bootstrap_vllm_runtime()


def _print(title: str) -> None:
    print(f"\n[WaveSlice-Validate] {title}")


def _extract_text(out: object) -> str:
    try:
        return str(out.outputs[0].text or "")
    except Exception:
        return ""


def run_offline_core_check(model_name: str) -> bool:
    _print("Offline core check (no vLLM required)")
    try:
        ws = WaveScheduler(model_name=model_name)
        best = ws.schedule_real(S_s=64, S_l=2048, t_wait_us=1000.0, queue_length=8)
        print(f"  scheduler schedule_real output: {best}")

        metrics = WaveSliceMetrics(short_threshold_tokens=256)
        metrics.register_request("r1", arrival_s=1.0, input_tokens=64, solo_us=1000.0, is_short=True)

        class _Payload:
            token_ids = [1]

        class _Output:
            request_id = "r1"
            outputs = [_Payload()]
            finished = True

        metrics.observe_engine_outputs([_Output()], now_s=1.02)
        report = metrics.summary()
        print(
            "  metrics sample: "
            f"ttft_p99={report['ttft_ms_short']['p99']}, "
            f"slowdown_p99={report['slowdown_short']['p99']}"
        )
        return True
    except Exception as exc:
        print(f"  FAIL: {exc}")
        return False


def run_vllm_embedding_check(model_name: str) -> bool:
    _print("vLLM embedding hook check")
    import importlib

    def _load(mod_name: str):
        try:
            return importlib.import_module(mod_name)
        except Exception:
            return None

    try:
        inject_wave_slice(model_name, force=True)

        p1_ok = False
        for mod_name, cls_name, meth_name, attr_name in [
            ("vllm.core.scheduler", "Scheduler", "_schedule", "__wave_slice_hook__"),
            ("vllm.v1.core.sched.scheduler", "Scheduler", "schedule", "__wave_slice_hook__"),
        ]:
            mod = _load(mod_name)
            cls = getattr(mod, cls_name, None) if mod else None
            meth = getattr(cls, meth_name, None) if cls else None
            p1_ok = p1_ok or bool(getattr(meth, attr_name, False))

        p2_ok = False
        for mod_name, cls_name in [
            ("vllm.worker.model_runner", "ModelRunner"),
            ("vllm.v1.worker.gpu_model_runner", "GPUModelRunner"),
        ]:
            mod = _load(mod_name)
            cls = getattr(mod, cls_name, None) if mod else None
            meth = getattr(cls, "execute_model", None) if cls else None
            p2_ok = p2_ok or bool(getattr(meth, "__wave_slice_phase2_hook__", False))

        m_ok = False
        for mod_name in ["vllm.engine.llm_engine", "vllm.v1.engine.llm_engine"]:
            mod = _load(mod_name)
            cls = getattr(mod, "LLMEngine", None) if mod else None
            meth = getattr(cls, "step", None) if cls else None
            m_ok = m_ok or bool(getattr(meth, "__wave_slice_metrics_hook__", False))

        print(f"  phase1_hook_installed={p1_ok}")
        print(f"  phase2_hook_installed={p2_ok}")
        print(f"  metrics_hook_installed={m_ok}")
        return p1_ok and p2_ok and m_ok
    except Exception as exc:
        print(f"  FAIL: hook install failed: {exc}")
        return False
    finally:
        try:
            uninject_wave_slice()
        except Exception:
            pass


def run_live_vllm_smoke(model_path: str, model_name: str, max_new_tokens: int) -> bool:
    _print("Live vLLM smoke run")
    try:
        from vllm.engine.arg_utils import EngineArgs
        from vllm.engine.llm_engine import LLMEngine
        from vllm.sampling_params import SamplingParams
    except Exception as exc:
        print(f"  FAIL: cannot import vllm runtime: {exc}")
        return False

    try:
        inject_wave_slice(model_name, force=True)
        reset_wave_slice_metrics()

        engine_args = EngineArgs(
            model=model_path,
            # NOTE:
            # vLLM 0.4.3 may crash with enable_lora=True when no LoRARequest is attached.
            # Keep this smoke test for embedding correctness only.
            enable_lora=False,
            max_lora_rank=32,
            max_num_batched_tokens=2048,
            enable_chunked_prefill=True,
            disable_sliding_window=True,
            enforce_eager=True,
        )
        engine = LLMEngine.from_engine_args(engine_args)

        engine.add_request(
            "short_req",
            "Translate to French: hello world",
            SamplingParams(max_tokens=max_new_tokens, temperature=0.0),
        )
        engine.add_request(
            "long_req",
            "The quick brown fox jumps over the lazy dog. " * 180,
            SamplingParams(max_tokens=max_new_tokens, temperature=0.0),
        )

        deadline = time.time() + 180
        while time.time() < deadline and engine.has_unfinished_requests():
            engine.step()

        report = get_wave_slice_metrics(reset=True)
        print(f"  report.ttft_short_p99={report.get('ttft_ms_short', {}).get('p99')}")
        print(f"  report.slowdown_short_p99={report.get('slowdown_short', {}).get('p99')}")
        print(f"  report.phase2_apply_ratio={report.get('phase2', {}).get('apply_ratio')}")
        return True
    except Exception as exc:
        print(f"  FAIL: live run failed: {exc}")
        return False
    finally:
        try:
            uninject_wave_slice()
        except Exception:
            pass


def run_lora_effectiveness_check(
    model_path: str,
    model_name: str,
    lora_adapter_path_a: str,
    lora_adapter_path_b: str,
    lora_name_a: str,
    lora_name_b: str,
    max_new_tokens: int,
    phase2_consistency_mode: str,
    phase2_dispatch_mode: str,
    enable_phase1_scheduler: bool,
) -> bool:
    _print("LoRA effectiveness check (real vLLM LoRARequest)")
    try:
        from vllm.engine.arg_utils import EngineArgs
        from vllm.engine.llm_engine import LLMEngine
        from vllm.lora.request import LoRARequest
        from vllm.sampling_params import SamplingParams
    except Exception as exc:
        print(f"  FAIL: cannot import vllm runtime/lora modules: {exc}")
        return False

    try:
        policy = WaveSlicePolicy(
            enable_phase1_scheduler=enable_phase1_scheduler,
            enable_phase2_modelrunner=True,
            enable_metrics_hook=True,
            enable_sjf_reorder=False,
            enable_tick_hide=False,
            enable_vllm_lora_compat_patch=True,
            phase2_consistency_mode=phase2_consistency_mode,
            phase2_dispatch_mode=phase2_dispatch_mode,
        )
        inject_wave_slice(model_name, policy=policy, force=True)
        reset_wave_slice_metrics()

        def _mk_lora_req(name: str, req_id: int, path: str):
            try:
                return LoRARequest(lora_name=name, lora_int_id=req_id, lora_path=path)
            except TypeError:
                return LoRARequest(lora_name=name, lora_int_id=req_id, lora_local_path=path)

        engine_args = EngineArgs(
            model=model_path,
            enable_lora=True,
            max_lora_rank=32,
            max_num_batched_tokens=2048,
            enable_chunked_prefill=True,
            disable_sliding_window=True,
            enforce_eager=True,
        )
        engine = LLMEngine.from_engine_args(engine_args)

        lora_req_a = _mk_lora_req(lora_name_a, 1, lora_adapter_path_a)
        lora_req_b = _mk_lora_req(lora_name_b, 2, lora_adapter_path_b)
        sampling = SamplingParams(max_tokens=max_new_tokens, temperature=0.0)

        base_id = "base_short"
        lora_short_a_id = "lora_short_a"
        lora_mid_b_id = "lora_mid_b"
        lora_long_a_id = "lora_long_a"
        lora_long_b_id = "lora_long_b"

        short_prompt = "Translate to French: I love machine learning."
        mid_prompt = (
            "Observed examples: 10+20=30. 20+30=50. 30+40=70. 40+50=90. "
            * 24
            + "Task: Continue the pattern with the next three equations and then give one short rule sentence."
        )
        long_prompt_a = (
            "Summarize this paragraph in exactly one sentence: "
            + ("Artificial intelligence is transforming industry, deployment, and systems engineering. " * 140)
        )
        long_prompt_b = (
            "Summarize this paragraph in exactly one sentence: "
            + ("Artificial intelligence is transforming industry, deployment, and systems engineering. " * 220)
        )

        engine.add_request(base_id, short_prompt, sampling)
        engine.add_request(lora_short_a_id, short_prompt, sampling, lora_request=lora_req_a)
        engine.add_request(lora_mid_b_id, mid_prompt, sampling, lora_request=lora_req_b)
        engine.add_request(lora_long_a_id, long_prompt_a, sampling, lora_request=lora_req_a)
        engine.add_request(lora_long_b_id, long_prompt_b, sampling, lora_request=lora_req_b)

        texts: dict[str, str] = {
            base_id: "",
            lora_short_a_id: "",
            lora_mid_b_id: "",
            lora_long_a_id: "",
            lora_long_b_id: "",
        }
        deadline = time.time() + 240
        while time.time() < deadline and engine.has_unfinished_requests():
            outputs = engine.step()
            for out in outputs:
                if out.finished:
                    texts[out.request_id] = _extract_text(out)

        report = get_wave_slice_metrics(reset=True)
        print(f"  base_output={texts[base_id]!r}")
        print(f"  lora_short_a_output={texts[lora_short_a_id]!r}")
        print(f"  lora_mid_b_output={texts[lora_mid_b_id]!r}")
        print(f"  lora_long_a_output={texts[lora_long_a_id]!r}")
        print(f"  lora_long_b_output={texts[lora_long_b_id]!r}")
        print(f"  base_vs_loraShortA_differ={texts[base_id] != texts[lora_short_a_id]}")
        print(f"  report.phase2_apply_ratio={report.get('phase2', {}).get('apply_ratio')}")
        print(f"  report.phase2_reasons={report.get('phase2', {}).get('reasons')}")
        print(f"  report.phase1_apply_ratio={report.get('scheduler', {}).get('apply_ratio')}")
        print(f"  report.phase1_chosen_chunk_avg={report.get('scheduler', {}).get('chosen_chunk_avg')}")
        print(f"  report.ttft_short_p99={report.get('ttft_ms_short', {}).get('p99')}")
        print(f"  report.slowdown_short_p99={report.get('slowdown_short', {}).get('p99')}")
        return True
    except Exception as exc:
        print(f"  FAIL: LoRA effectiveness run failed: {exc}")
        print(traceback.format_exc())
        return False
    finally:
        try:
            uninject_wave_slice()
        except Exception:
            pass


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate Wave-Slice correctness/effectiveness/embedding.")
    parser.add_argument("--model-name", default="Mistral-7B-v0.1", help="Wave-Slice LUT model key")
    parser.add_argument("--model-path", default="mistralai/Mistral-7B-v0.1", help="HF/local model path for live run")
    parser.add_argument(
        "--run-live",
        action="store_true",
        help="Run real vLLM engine smoke test (requires vllm + GPU + model availability).",
    )
    parser.add_argument(
        "--run-lora-live",
        action="store_true",
        help="Run real heterogeneous LoRARequest effectiveness test.",
    )
    parser.add_argument(
        "--lora-adapter-path-a",
        default=None,
        help="Local path to LoRA adapter A.",
    )
    parser.add_argument(
        "--lora-adapter-path-b",
        default=None,
        help="Local path to LoRA adapter B. If omitted, adapter A is reused.",
    )
    parser.add_argument("--lora-name-a", default="adapter_a", help="LoRA A logical name.")
    parser.add_argument("--lora-name-b", default="adapter_b", help="LoRA B logical name.")
    parser.add_argument(
        "--auto-build-lora",
        action="store_true",
        help="Auto-generate two synthetic adapters under the configured adapters root for the given base model (requires peft).",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=32,
        help="Decode length used in live validation. Set >=16 for non-trivial serving behavior.",
    )
    parser.add_argument(
        "--phase2-consistency-mode",
        choices=["balanced", "strict"],
        default="strict",
        help="Phase-II consistency mode used in LoRA live check.",
    )
    parser.add_argument(
        "--phase2-dispatch-mode",
        choices=["synchronized", "async_experimental"],
        default="synchronized",
        help="Phase-II dispatch mode used in LoRA live check.",
    )
    parser.add_argument(
        "--enable-phase1-in-lora-live",
        action="store_true",
        help="Run LoRA live validation with current Phase-I scheduler enabled together with Phase-II.",
    )
    parser.add_argument(
        "--serialize-gpu-tests",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Serialize GPU-backed validation runs through a global file lock.",
    )
    parser.add_argument(
        "--gpu-lock-path",
        default="",
        help="Optional file path used for the global GPU experiment lock.",
    )
    args = parser.parse_args()

    ok = True
    ok = run_offline_core_check(args.model_name) and ok
    ok = run_vllm_embedding_check(args.model_name) and ok
    needs_gpu_lock = bool(args.run_live or args.run_lora_live)
    with gpu_experiment_lock(
        label=f"validate:{args.model_name}",
        enabled=bool(args.serialize_gpu_tests and needs_gpu_lock),
        lock_path=args.gpu_lock_path or None,
    ):
        if args.run_live:
            ok = run_live_vllm_smoke(args.model_path, args.model_name, args.max_new_tokens) and ok
        if args.run_lora_live:
            lora_a = args.lora_adapter_path_a
            lora_b = args.lora_adapter_path_b

            if args.auto_build_lora and not lora_a:
                out_dir = Path("/tmp") / "waveslice_synth_lora" / str(int(time.time()))
                try:
                    from config.experiment_catalog import DEFAULT_SYNTHETIC_ADAPTER_PRESETS
                    from tools.synthetic_lora_builder import AdapterSpec, build_synthetic_adapters

                    generated = build_synthetic_adapters(
                        base_model=args.model_path,
                        out_dir=str(out_dir),
                        specs=[
                            AdapterSpec(
                                name=spec.name,
                                rank=spec.rank,
                                alpha=spec.alpha,
                                seed=spec.seed,
                                init_std=spec.init_std,
                            )
                            for spec in DEFAULT_SYNTHETIC_ADAPTER_PRESETS
                        ],
                    )
                    lora_a = generated[0]
                    lora_b = generated[1]
                    print(f"\n[WaveSlice-Validate] auto generated adapters:\n  A={lora_a}\n  B={lora_b}")
                except Exception as exc:
                    print(f"\n[WaveSlice-Validate] FAIL: auto-build-lora failed: {exc}")
                    ok = False

            if not lora_a:
                print("\n[WaveSlice-Validate] FAIL: --run-lora-live requires --lora-adapter-path-a or --auto-build-lora")
                ok = False
            else:
                if not lora_b:
                    lora_b = lora_a
                if not os.path.exists(lora_a) or not os.path.exists(lora_b):
                    print(f"\n[WaveSlice-Validate] FAIL: adapter path not found: A={lora_a}, B={lora_b}")
                    ok = False
                else:
                    ok = run_lora_effectiveness_check(
                        model_path=args.model_path,
                        model_name=args.model_name,
                        lora_adapter_path_a=lora_a,
                        lora_adapter_path_b=lora_b,
                        lora_name_a=args.lora_name_a,
                        lora_name_b=args.lora_name_b,
                        max_new_tokens=args.max_new_tokens,
                        phase2_consistency_mode=args.phase2_consistency_mode,
                        phase2_dispatch_mode=args.phase2_dispatch_mode,
                        enable_phase1_scheduler=args.enable_phase1_in_lora_live,
                    ) and ok

    _print("DONE")
    print(f"  overall_pass={ok}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
