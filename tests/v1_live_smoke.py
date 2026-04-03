from __future__ import annotations

import argparse
import logging
import os

from engine.runtime_bootstrap import bootstrap_vllm_runtime
from engine.vllm_hijacker import (
    WaveSlicePolicy,
    get_wave_slice_metrics,
    inject_wave_slice,
    reset_wave_slice_metrics,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Wave-Slice v1 live smoke test.")
    parser.add_argument(
        "--model-path",
        default="mistralai/Mistral-7B-v0.1",
    )
    parser.add_argument("--model-name", default="Mistral-7B-v0.1")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    args = parser.parse_args()

    os.environ.setdefault("WAVESLICE_VLLM_MODE", "v1")

    from vllm import LLM, SamplingParams

    bootstrap_vllm_runtime()
    logging.basicConfig(level=logging.INFO)

    policy = WaveSlicePolicy(
        enable_phase2_modelrunner=True,
        phase2_enable_v1_true_unbind=True,
        phase2_consistency_mode="strict",
        enable_sjf_reorder=False,
        enable_tick_hide=False,
    )
    inject_wave_slice(args.model_name, policy=policy, force=True)
    reset_wave_slice_metrics()

    llm = LLM(
        model=args.model_path,
        enforce_eager=True,
        enable_lora=False,
        max_num_batched_tokens=2048,
        enable_chunked_prefill=True,
    )

    prompts = [
        "Translate to French: I love machine learning systems and efficient serving pipelines. "
        * 16,
        "Summarize in one sentence: "
        + (
            "Artificial intelligence changes systems engineering and deployment under heterogeneous LoRA workloads. "
            * 180
        ),
    ]
    outputs = llm.generate(
        prompts,
        SamplingParams(max_tokens=args.max_new_tokens, temperature=0.0),
    )
    for out in outputs:
        print(
            "prompt_tokens",
            len(out.prompt_token_ids),
            "text",
            repr(out.outputs[0].text[:80]),
        )

    print("phase2_metrics", get_wave_slice_metrics(reset=True).get("phase2"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
