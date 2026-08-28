"""Request-lifecycle hooks for the vLLM V1 engine."""

from __future__ import annotations

import functools
import time
from collections.abc import Callable
from typing import Any


def _observe(
    state: Any,
    request_id: str,
    tokens: int | None,
    arrival: Any,
    estimate_solo_us: Callable[..., Any],
    seed: Callable[..., Any],
) -> None:
    if tokens is None:
        return
    state.metrics.register_request(
        request_id,
        arrival_s=float(arrival) if arrival is not None else None,
        input_tokens=tokens,
        solo_us=estimate_solo_us(state.brain, tokens),
        is_short=tokens <= state.policy.metrics_short_request_tokens,
    )
    seed(state, request_id=request_id, input_tokens=tokens)


def build_v1_process_inputs_hook(
    state: Any,
    *,
    estimate_solo_us: Callable[..., Any],
    phase1_maybe_seed_ingress_virtual: Callable[..., Any],
):
    original = state.original_v1_processor_process_inputs

    @functools.wraps(original)
    def process(processor: Any, *args: Any, **kwargs: Any):
        result = original(processor, *args, **kwargs)
        request_id = str(args[0] if args else kwargs.get("request_id", ""))
        request = result[1] if isinstance(result, tuple) and len(result) > 1 else None
        token_ids = getattr(request, "prompt_token_ids", None)
        arrival = args[3] if len(args) > 3 else kwargs.get("arrival_time")
        if request_id:
            _observe(
                state,
                request_id,
                None if token_ids is None else len(token_ids),
                arrival,
                estimate_solo_us,
                phase1_maybe_seed_ingress_virtual,
            )
        return result

    process.__wave_slice_metrics_hook__ = True
    return process


def build_v1_engine_core_add_request_hook(
    state: Any,
    *,
    estimate_solo_us: Callable[..., Any],
    phase1_maybe_seed_ingress_virtual: Callable[..., Any],
):
    original = state.original_v1_engine_core_add_request

    @functools.wraps(original)
    def add_request(core: Any, *args: Any, **kwargs: Any):
        request = args[0] if args else kwargs.get("request")
        request_id = str(getattr(request, "request_id", "") or "")
        token_ids = getattr(request, "prompt_token_ids", None)
        if request_id:
            _observe(
                state,
                request_id,
                None if token_ids is None else len(token_ids),
                getattr(request, "arrival_time", None),
                estimate_solo_us,
                phase1_maybe_seed_ingress_virtual,
            )
        return original(core, *args, **kwargs)

    add_request.__wave_slice_metrics_hook__ = True
    return add_request


def build_add_request_hook(
    state: Any,
    *,
    estimate_prompt_tokens: Callable[..., Any],
    estimate_solo_us: Callable[..., Any],
    lookup_engine_prompt_tokens: Callable[..., Any],
    phase1_maybe_seed_ingress_virtual: Callable[..., Any],
):
    original = state.original_add_request

    @functools.wraps(original)
    def add_request(engine: Any, *args: Any, **kwargs: Any):
        result = original(engine, *args, **kwargs)
        request_id = str(args[0] if args else kwargs.get("request_id", ""))
        if not request_id:
            return result
        prompt = args[1] if len(args) > 1 else kwargs.get("prompt", kwargs.get("prompt_token_ids"))
        lora = kwargs.get("lora_request", args[3] if len(args) > 3 else None)
        tokens = state.phase1_active_prompt_tokens.get(request_id)
        if tokens is None:
            tokens = lookup_engine_prompt_tokens(engine, request_id=request_id)
        if tokens is None:
            tokens = estimate_prompt_tokens(prompt, engine_self=engine, lora_request=lora)
        _observe(
            state,
            request_id,
            tokens,
            time.perf_counter(),
            estimate_solo_us,
            phase1_maybe_seed_ingress_virtual,
        )
        return result

    add_request.__wave_slice_metrics_hook__ = True
    return add_request


def build_step_hook(state: Any):
    original = state.original_step

    @functools.wraps(original)
    def step(engine: Any, *args: Any, **kwargs: Any):
        outputs = original(engine, *args, **kwargs)
        active = set(state.phase2_priority_active_ids)
        finished = [
            str(output.request_id)
            for output in outputs or ()
            if getattr(output, "finished", False) and getattr(output, "request_id", None)
        ]
        for request_id in finished:
            for mapping in (
                state.phase1_explicit_plans,
                state.phase1_active_prompt_tokens,
                state.phase1_ingress_virtuals,
                state.phase1_virtual_token_caps,
            ):
                mapping.pop(request_id, None)
        state.metrics.record_priority_lane_observation(
            active_ids=active,
            seen_request_ids=[str(getattr(output, "request_id", "")) for output in outputs or ()],
            finished_request_ids=finished,
        )
        state.metrics.observe_engine_outputs(outputs, now_s=time.perf_counter())
        return outputs

    step.__wave_slice_metrics_hook__ = True
    return step
