from __future__ import annotations

import functools
import logging
import time
from typing import Any, Callable


logger = logging.getLogger("WaveSlice")
logger.addHandler(logging.NullHandler())


def _extract_processed_prompt_tokens(processed_inputs: Any) -> int | None:
    if not isinstance(processed_inputs, dict):
        return None

    prompt_token_ids = processed_inputs.get("prompt_token_ids")
    if prompt_token_ids is not None:
        try:
            return len(prompt_token_ids)
        except Exception:
            return None

    decoder_inputs = processed_inputs.get("decoder")
    decoder_tokens = _extract_processed_prompt_tokens(decoder_inputs)
    if decoder_tokens is not None:
        return decoder_tokens

    encoder_inputs = processed_inputs.get("encoder")
    return _extract_processed_prompt_tokens(encoder_inputs)


def build_add_processed_request_hook(
    state: Any,
    *,
    estimate_solo_us: Callable[..., Any],
    phase1_maybe_seed_ingress_virtual: Callable[..., Any],
) -> Callable[..., Any]:
    original_add_processed_request = state.original_add_processed_request
    if original_add_processed_request is None:
        raise RuntimeError("internal error: original _add_processed_request is missing")

    @functools.wraps(original_add_processed_request)
    def _wave_add_processed_request_hook(self: Any, *args: Any, **kwargs: Any) -> Any:
        try:
            request_id = None
            if len(args) >= 1:
                request_id = str(args[0])
            elif "request_id" in kwargs:
                request_id = str(kwargs["request_id"])

            processed_inputs = args[1] if len(args) >= 2 else kwargs.get("processed_inputs")
            arrival_time = args[3] if len(args) >= 4 else kwargs.get("arrival_time")
            input_tokens = _extract_processed_prompt_tokens(processed_inputs)
            if request_id:
                state.metrics.record_phase1_step_trace(
                    request_id=str(request_id),
                    event=(
                        "phase1_add_processed_request_input_tokens_none"
                        if input_tokens is None
                        else "phase1_add_processed_request_observed"
                    ),
                    is_prefill=True,
                    num_computed_tokens=0,
                    uncached=(int(input_tokens) if input_tokens is not None else None),
                    cached=0,
                )
            if request_id and input_tokens is not None:
                solo_us = estimate_solo_us(state.brain, input_tokens)
                arrival_s = None
                try:
                    if arrival_time is not None:
                        arrival_s = float(arrival_time)
                except Exception:
                    arrival_s = None
                state.metrics.register_request(
                    str(request_id),
                    arrival_s=arrival_s,
                    input_tokens=int(input_tokens),
                    solo_us=solo_us,
                    is_short=(int(input_tokens) <= state.policy.metrics_short_request_tokens),
                )
                phase1_maybe_seed_ingress_virtual(
                    state,
                    request_id=str(request_id),
                    input_tokens=int(input_tokens),
                )
        except Exception:
            logger.exception("Wave-Slice metrics _add_processed_request hook failed.")
        return original_add_processed_request(self, *args, **kwargs)

    _wave_add_processed_request_hook.__wave_slice_metrics_hook__ = True  # type: ignore[attr-defined]
    return _wave_add_processed_request_hook


def build_v1_process_inputs_hook(
    state: Any,
    *,
    estimate_solo_us: Callable[..., Any],
    phase1_maybe_seed_ingress_virtual: Callable[..., Any],
) -> Callable[..., Any]:
    original_process_inputs = state.original_v1_processor_process_inputs
    if original_process_inputs is None:
        raise RuntimeError("internal error: original v1 Processor.process_inputs is missing")

    @functools.wraps(original_process_inputs)
    def _wave_v1_process_inputs_hook(self: Any, *args: Any, **kwargs: Any) -> Any:
        result = original_process_inputs(self, *args, **kwargs)
        try:
            request_id = None
            if len(args) >= 1:
                request_id = str(args[0])
            elif "request_id" in kwargs:
                request_id = str(kwargs["request_id"])

            arrival_time = args[3] if len(args) >= 4 else kwargs.get("arrival_time")
            input_tokens = None
            request = result[1] if isinstance(result, tuple) and len(result) >= 2 else None
            prompt_token_ids = getattr(request, "prompt_token_ids", None)
            if prompt_token_ids is not None:
                try:
                    input_tokens = len(prompt_token_ids)
                except Exception:
                    input_tokens = None

            if request_id:
                state.metrics.record_phase1_step_trace(
                    request_id=str(request_id),
                    event=(
                        "phase1_v1_process_inputs_input_tokens_none"
                        if input_tokens is None
                        else "phase1_v1_process_inputs_observed"
                    ),
                    is_prefill=True,
                    num_computed_tokens=0,
                    uncached=(int(input_tokens) if input_tokens is not None else None),
                    cached=0,
                )
            if request_id and input_tokens is not None:
                solo_us = estimate_solo_us(state.brain, input_tokens)
                arrival_s = None
                try:
                    if arrival_time is not None:
                        arrival_s = float(arrival_time)
                except Exception:
                    arrival_s = None
                state.metrics.register_request(
                    str(request_id),
                    arrival_s=arrival_s,
                    input_tokens=int(input_tokens),
                    solo_us=solo_us,
                    is_short=(int(input_tokens) <= state.policy.metrics_short_request_tokens),
                )
                phase1_maybe_seed_ingress_virtual(
                    state,
                    request_id=str(request_id),
                    input_tokens=int(input_tokens),
                )
        except Exception:
            logger.exception("Wave-Slice metrics v1 Processor.process_inputs hook failed.")
        return result

    _wave_v1_process_inputs_hook.__wave_slice_metrics_hook__ = True  # type: ignore[attr-defined]
    return _wave_v1_process_inputs_hook


def build_v1_engine_core_add_request_hook(
    state: Any,
    *,
    estimate_solo_us: Callable[..., Any],
    phase1_maybe_seed_ingress_virtual: Callable[..., Any],
) -> Callable[..., Any]:
    original_add_request = state.original_v1_engine_core_add_request
    if original_add_request is None:
        raise RuntimeError("internal error: original v1 EngineCore.add_request is missing")

    @functools.wraps(original_add_request)
    def _wave_v1_engine_core_add_request_hook(self: Any, *args: Any, **kwargs: Any) -> Any:
        try:
            request = args[0] if len(args) >= 1 else kwargs.get("request")
            request_id = str(getattr(request, "request_id", "") or "")
            prompt_token_ids = getattr(request, "prompt_token_ids", None)
            input_tokens = None
            if prompt_token_ids is not None:
                try:
                    input_tokens = len(prompt_token_ids)
                except Exception:
                    input_tokens = None
            arrival_time = getattr(request, "arrival_time", None)
            if request_id:
                state.metrics.record_phase1_step_trace(
                    request_id=request_id,
                    event=(
                        "phase1_v1_engine_core_add_request_input_tokens_none"
                        if input_tokens is None
                        else "phase1_v1_engine_core_add_request_observed"
                    ),
                    is_prefill=True,
                    num_computed_tokens=0,
                    uncached=(int(input_tokens) if input_tokens is not None else None),
                    cached=0,
                )
            if request_id and input_tokens is not None:
                solo_us = estimate_solo_us(state.brain, input_tokens)
                arrival_s = None
                try:
                    if arrival_time is not None:
                        arrival_s = float(arrival_time)
                except Exception:
                    arrival_s = None
                state.metrics.register_request(
                    request_id,
                    arrival_s=arrival_s,
                    input_tokens=int(input_tokens),
                    solo_us=solo_us,
                    is_short=(int(input_tokens) <= state.policy.metrics_short_request_tokens),
                )
                phase1_maybe_seed_ingress_virtual(
                    state,
                    request_id=request_id,
                    input_tokens=int(input_tokens),
                )
        except Exception:
            logger.exception("Wave-Slice metrics v1 EngineCore.add_request hook failed.")
        return original_add_request(self, *args, **kwargs)

    _wave_v1_engine_core_add_request_hook.__wave_slice_metrics_hook__ = True  # type: ignore[attr-defined]
    return _wave_v1_engine_core_add_request_hook


def build_add_request_hook(
    state: Any,
    *,
    estimate_prompt_tokens: Callable[..., Any],
    estimate_solo_us: Callable[..., Any],
    lookup_engine_prompt_tokens: Callable[..., Any],
    phase1_maybe_seed_ingress_virtual: Callable[..., Any],
) -> Callable[..., Any]:
    original_add_request = state.original_add_request
    if original_add_request is None:
        raise RuntimeError("internal error: original add_request is missing")

    @functools.wraps(original_add_request)
    def _wave_add_request_hook(self: Any, *args: Any, **kwargs: Any) -> Any:
        result = original_add_request(self, *args, **kwargs)
        try:
            request_id = None
            if len(args) >= 1:
                request_id = str(args[0])
            elif "request_id" in kwargs:
                request_id = str(kwargs["request_id"])
            if not request_id:
                return result

            prompt_obj = args[1] if len(args) >= 2 else kwargs.get("prompt")
            if prompt_obj is None:
                prompt_obj = kwargs.get("prompt_token_ids")
            lora_request = kwargs.get("lora_request")
            if lora_request is None and len(args) >= 4:
                lora_request = args[3]
            input_tokens = state.phase1_active_prompt_tokens.get(str(request_id))
            prompt_source = "seeded"
            if input_tokens is None:
                input_tokens = lookup_engine_prompt_tokens(
                    self,
                    request_id=str(request_id),
                )
                prompt_source = "engine"
            if input_tokens is None:
                input_tokens = estimate_prompt_tokens(
                    prompt_obj,
                    engine_self=self,
                    lora_request=lora_request,
                )
                prompt_source = "estimate"
            prompt_type = type(prompt_obj).__name__ if prompt_obj is not None else "NoneType"
            state.metrics.record_phase1_step_trace(
                request_id=str(request_id),
                event=(
                    f"phase1_add_request_input_tokens_none:{prompt_type}"
                    if input_tokens is None
                    else f"phase1_add_request_observed:{prompt_type}:{prompt_source}"
                ),
                is_prefill=True,
                num_computed_tokens=0,
                uncached=(int(input_tokens) if input_tokens is not None else None),
                cached=0,
            )
            solo_us = estimate_solo_us(state.brain, input_tokens)
            is_short = (input_tokens is not None) and (input_tokens <= state.policy.metrics_short_request_tokens)
            state.metrics.register_request(
                request_id,
                arrival_s=time.perf_counter(),
                input_tokens=input_tokens,
                solo_us=solo_us,
                is_short=is_short,
            )
            phase1_maybe_seed_ingress_virtual(
                state,
                request_id=request_id,
                input_tokens=input_tokens,
            )
        except Exception:
            logger.exception("Wave-Slice metrics add_request hook failed.")
        return result

    _wave_add_request_hook.__wave_slice_metrics_hook__ = True  # type: ignore[attr-defined]
    return _wave_add_request_hook


def build_step_hook(
    state: Any,
    *,
    maybe_install_v1_runtime_lifecycle_hooks: Callable[[Any], None],
    phase12_clear_escape_lane: Callable[..., Any],
) -> Callable[..., Any]:
    original_step = state.original_step
    if original_step is None:
        raise RuntimeError("internal error: original step is missing")

    @functools.wraps(original_step)
    def _wave_step_hook(self: Any, *args: Any, **kwargs: Any) -> Any:
        maybe_install_v1_runtime_lifecycle_hooks(state)
        outputs = original_step(self, *args, **kwargs)
        try:
            active_ids = set(getattr(state, "phase2_escape_active_ids", set()) or set())
            deferred_outputs = list(getattr(state, "phase2_deferred_request_outputs", []) or [])
            if (not active_ids) and deferred_outputs and isinstance(outputs, list):
                outputs = list(outputs or []) + deferred_outputs
                state.phase2_deferred_request_outputs = []
            if active_ids and isinstance(outputs, list):
                outputs.sort(
                    key=lambda out: (
                        0 if str(getattr(out, "request_id", "") or "") in active_ids else 1,
                        str(getattr(out, "request_id", "") or ""),
                    )
                )
            finished_request_ids: list[str] = []
            for out in outputs or []:
                try:
                    if bool(getattr(out, "finished", False)):
                        req_id = str(getattr(out, "request_id", ""))
                        if req_id:
                            finished_request_ids.append(req_id)
                            state.phase1_explicit_plans.pop(req_id, None)
                            state.phase1_active_prompt_tokens.pop(req_id, None)
                            state.phase1_ingress_virtuals.pop(req_id, None)
                            state.phase1_virtual_token_caps.pop(req_id, None)
                except Exception:
                    continue
            state.metrics.record_escape_lane_observation(
                active_ids=active_ids,
                seen_request_ids=[str(getattr(out, "request_id", "") or "") for out in (outputs or [])],
                finished_request_ids=finished_request_ids,
            )
            if finished_request_ids:
                clear_ids = [] if bool(getattr(state.policy, "phase2_enable_execution_escape", False)) else finished_request_ids
                if clear_ids:
                    phase12_clear_escape_lane(state, request_ids=clear_ids)
            state.metrics.observe_engine_outputs(outputs, now_s=time.perf_counter())
        except Exception:
            logger.exception("Wave-Slice metrics step hook failed.")
        return outputs

    _wave_step_hook.__wave_slice_metrics_hook__ = True  # type: ignore[attr-defined]
    _wave_step_hook.__wave_slice_lifecycle_lazy__ = True  # type: ignore[attr-defined]
    return _wave_step_hook
