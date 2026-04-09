from __future__ import annotations

import argparse
import json
import os
import traceback


def _print_stage(label: str, payload: dict) -> None:
    print(f"\n[{label}]")
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Minimal vLLM v1 CUDA/platform initialization debugger."
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--enable-lora", action="store_true")
    parser.add_argument("--max-model-len", type=int, default=3072)
    parser.add_argument("--max-num-batched-tokens", type=int, default=1536)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.60)
    parser.add_argument("--enable-chunked-prefill", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    os.environ.setdefault("VLLM_NO_USAGE_STATS", "1")
    os.environ.setdefault("WAVESLICE_VLLM_MODE", "v1")
    os.environ.setdefault("VLLM_USE_V1", "1")

    from engine.runtime_bootstrap import bootstrap_vllm_runtime

    _print_stage(
        "env_before_bootstrap",
        {
            "WAVESLICE_VLLM_MODE": os.environ.get("WAVESLICE_VLLM_MODE"),
            "VLLM_USE_V1": os.environ.get("VLLM_USE_V1"),
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "PYTHONPATH": os.environ.get("PYTHONPATH"),
        },
    )

    ok = bootstrap_vllm_runtime()
    _print_stage("bootstrap", {"ok": ok})

    import torch

    _print_stage(
        "torch_after_bootstrap",
        {
            "cuda_available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count(),
            "version_cuda": getattr(torch.version, "cuda", None),
        },
    )

    import vllm.platforms as platforms

    _print_stage(
        "platform_after_bootstrap",
        {
            "platform_cls": type(platforms.current_platform).__name__,
            "device_type": getattr(platforms.current_platform, "device_type", None),
            "device_name": getattr(platforms.current_platform, "device_name", None),
        },
    )

    try:
        from vllm.engine.arg_utils import EngineArgs
        from vllm.engine.llm_engine import LLMEngine

        engine_args = EngineArgs(
            model=args.model_path,
            trust_remote_code=args.trust_remote_code,
            seed=0,
            enable_lora=args.enable_lora,
            max_lora_rank=32,
            max_num_batched_tokens=int(args.max_num_batched_tokens),
            enable_chunked_prefill=bool(args.enable_chunked_prefill),
            disable_sliding_window=True,
            enforce_eager=True,
            max_model_len=int(args.max_model_len),
            gpu_memory_utilization=float(args.gpu_memory_utilization),
        )
        _print_stage(
            "engine_args",
            {
                "model": args.model_path,
                "enable_lora": args.enable_lora,
                "max_model_len": args.max_model_len,
                "max_num_batched_tokens": args.max_num_batched_tokens,
                "gpu_memory_utilization": args.gpu_memory_utilization,
            },
        )
    except Exception as exc:
        _print_stage(
            "engine_args_import_failure",
            {
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            },
        )
        return 1

    try:
        engine = LLMEngine.from_engine_args(engine_args)
    except Exception as exc:
        _print_stage(
            "engine_create_failure",
            {
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "torch_cuda_available_now": torch.cuda.is_available(),
                "torch_device_count_now": torch.cuda.device_count(),
                "platform_cls_now": type(platforms.current_platform).__name__,
                "platform_device_type_now": getattr(platforms.current_platform, "device_type", None),
            },
        )
        return 2

    _print_stage(
        "engine_create_success",
        {
            "engine_cls": type(engine).__name__,
            "platform_cls_now": type(platforms.current_platform).__name__,
            "platform_device_type_now": getattr(platforms.current_platform, "device_type", None),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
