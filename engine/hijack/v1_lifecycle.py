from __future__ import annotations

import functools
import logging
from typing import Any, Callable

logger = logging.getLogger("WaveSlice")
logger.addHandler(logging.NullHandler())


def maybe_install_v1_runtime_lifecycle_hooks(
    state: Any,
    *,
    patch_lock: Any,
    load_v1_output_processor_cls: Callable[[], Any],
    build_output_processor_add_request_hook: Callable[[Any], Callable[..., Any]],
    build_output_processor_process_outputs_hook: Callable[[Any], Callable[..., Any]],
    build_v1_scheduler_update_from_output_hook: Callable[[Any], Callable[..., Any]],
    build_v1_scheduler_finish_requests_hook: Callable[[Any], Callable[..., Any]],
) -> None:
    if not state.policy.enable_v1_runtime_lifecycle_patch:
        return
    if state.v1_lifecycle_hooks_installed:
        return
    with patch_lock:
        if state.v1_lifecycle_hooks_installed:
            return
        try:
            output_processor_cls = load_v1_output_processor_cls()
            original_op_add_request = getattr(output_processor_cls, "add_request", None)
            original_op_process_outputs = getattr(output_processor_cls, "process_outputs", None)
            if callable(original_op_add_request) and callable(original_op_process_outputs):
                if state.original_output_processor_add_request is None:
                    state.original_output_processor_add_request = original_op_add_request
                if state.original_output_processor_process_outputs is None:
                    state.original_output_processor_process_outputs = original_op_process_outputs
                state.v1_output_processor_cls = output_processor_cls
                if not bool(getattr(output_processor_cls.add_request, "__wave_slice_lifecycle_hook__", False)):
                    output_processor_cls.add_request = build_output_processor_add_request_hook(state)
                if not bool(getattr(output_processor_cls.process_outputs, "__wave_slice_lifecycle_hook__", False)):
                    output_processor_cls.process_outputs = build_output_processor_process_outputs_hook(state)
            else:
                logger.warning("[Wave-Slice] skip v1 lifecycle patch: OutputProcessor methods missing.")

            scheduler_cls = state.scheduler_cls
            if hasattr(scheduler_cls, "update_from_output"):
                original_update = getattr(scheduler_cls, "update_from_output", None)
                if callable(original_update):
                    if state.original_scheduler_update_from_output is None:
                        state.original_scheduler_update_from_output = original_update
                    if not bool(getattr(scheduler_cls.update_from_output, "__wave_slice_lifecycle_hook__", False)):
                        setattr(scheduler_cls, "update_from_output", build_v1_scheduler_update_from_output_hook(state))
            if hasattr(scheduler_cls, "finish_requests"):
                original_finish = getattr(scheduler_cls, "finish_requests", None)
                if callable(original_finish):
                    if state.original_scheduler_finish_requests is None:
                        state.original_scheduler_finish_requests = original_finish
                    if not bool(getattr(scheduler_cls.finish_requests, "__wave_slice_lifecycle_hook__", False)):
                        setattr(scheduler_cls, "finish_requests", build_v1_scheduler_finish_requests_hook(state))

            state.v1_lifecycle_hooks_installed = True
        except Exception:
            logger.exception("[Wave-Slice] failed to lazily install v1 lifecycle hooks.")


def build_output_processor_add_request_hook(state: Any) -> Callable[..., Any]:
    original_add_request = state.original_output_processor_add_request
    if original_add_request is None:
        raise RuntimeError("internal error: original OutputProcessor.add_request is missing")

    @functools.wraps(original_add_request)
    def _wave_output_processor_add_request_hook(self: Any, *args: Any, **kwargs: Any) -> Any:
        result = original_add_request(self, *args, **kwargs)
        try:
            request = args[0] if len(args) >= 1 else kwargs.get("request")
            request_id = str(getattr(request, "request_id", "") or "")
            if request_id:
                state.phase2_runtime_last_output_request_ids = [request_id]
        except Exception:
            logger.exception("Wave-Slice OutputProcessor.add_request hook failed.")
        return result

    _wave_output_processor_add_request_hook.__wave_slice_lifecycle_hook__ = True  # type: ignore[attr-defined]
    return _wave_output_processor_add_request_hook


def build_output_processor_process_outputs_hook(
    state: Any,
    *,
    phase12_clear_escape_lane: Callable[..., Any],
) -> Callable[..., Any]:
    original_process_outputs = state.original_output_processor_process_outputs
    if original_process_outputs is None:
        raise RuntimeError("internal error: original OutputProcessor.process_outputs is missing")

    @functools.wraps(original_process_outputs)
    def _wave_output_processor_process_outputs_hook(self: Any, *args: Any, **kwargs: Any) -> Any:
        outputs = original_process_outputs(self, *args, **kwargs)
        try:
            engine_core_outputs = args[0] if len(args) >= 1 else kwargs.get("engine_core_outputs")
            seen_request_ids: list[str] = []
            finished_request_ids: list[str] = []
            for engine_core_output in engine_core_outputs or []:
                request_id = str(getattr(engine_core_output, "request_id", "") or "")
                if not request_id:
                    continue
                seen_request_ids.append(request_id)
                if getattr(engine_core_output, "finish_reason", None) is not None:
                    finished_request_ids.append(request_id)
            state.phase2_runtime_last_output_request_ids = seen_request_ids
            state.phase2_runtime_last_finished_request_ids = finished_request_ids
            active_ids = set(getattr(state, "phase2_escape_active_ids", set()) or set())
            state.metrics.record_escape_lane_observation(
                active_ids=active_ids,
                seen_request_ids=seen_request_ids,
                finished_request_ids=finished_request_ids,
            )
            if active_ids and outputs is not None:
                try:
                    request_outputs = list(getattr(outputs, "request_outputs", []) or [])
                    if bool(getattr(state.policy, "phase2_enable_execution_escape", False)) and bool(
                        getattr(state.policy, "phase2_execution_escape_defer_finished_nonactive", True)
                    ):
                        keep_outputs: list[Any] = []
                        deferred_outputs: list[Any] = []
                        for out in request_outputs:
                            request_id = str(getattr(out, "request_id", "") or "")
                            finished = bool(getattr(out, "finished", False))
                            if finished and request_id and request_id not in active_ids:
                                deferred_outputs.append(out)
                            else:
                                keep_outputs.append(out)
                        if deferred_outputs:
                            state.phase2_deferred_request_outputs.extend(deferred_outputs)
                        request_outputs = keep_outputs
                    request_outputs.sort(
                        key=lambda out: (
                            0 if str(getattr(out, "request_id", "") or "") in active_ids else 1,
                            str(getattr(out, "request_id", "") or ""),
                        )
                    )
                    outputs.request_outputs = request_outputs
                except Exception:
                    logger.exception("Wave-Slice escape-lane output reorder failed.")
            elif outputs is not None and not active_ids:
                deferred_outputs = list(getattr(state, "phase2_deferred_request_outputs", []) or [])
                if deferred_outputs:
                    try:
                        request_outputs = list(getattr(outputs, "request_outputs", []) or [])
                        request_outputs.extend(deferred_outputs)
                        outputs.request_outputs = request_outputs
                        state.phase2_deferred_request_outputs = []
                    except Exception:
                        logger.exception("Wave-Slice deferred output flush failed.")
            if finished_request_ids:
                clear_ids = [] if bool(getattr(state.policy, "phase2_enable_execution_escape", False)) else finished_request_ids
                if clear_ids:
                    phase12_clear_escape_lane(state, request_ids=clear_ids)
        except Exception:
            logger.exception("Wave-Slice OutputProcessor.process_outputs hook failed.")
        return outputs

    _wave_output_processor_process_outputs_hook.__wave_slice_lifecycle_hook__ = True  # type: ignore[attr-defined]
    return _wave_output_processor_process_outputs_hook


def build_v1_scheduler_update_from_output_hook(
    state: Any,
    *,
    phase12_clear_escape_lane: Callable[..., Any],
) -> Callable[..., Any]:
    original_update = state.original_scheduler_update_from_output
    if original_update is None:
        raise RuntimeError("internal error: original Scheduler.update_from_output is missing")

    @functools.wraps(original_update)
    def _wave_v1_scheduler_update_from_output_hook(self: Any, *args: Any, **kwargs: Any) -> Any:
        outputs = original_update(self, *args, **kwargs)
        try:
            active_ids = set(getattr(state, "phase2_escape_active_ids", set()) or set())
            seen_request_ids: list[str] = []
            if active_ids:
                try:
                    for _virt_rank, engine_core_outputs in (outputs or {}).items():
                        if not isinstance(engine_core_outputs, list):
                            continue
                        engine_core_outputs.sort(
                            key=lambda out: (
                                0 if str(getattr(out, "request_id", "") or "") in active_ids else 1,
                                str(getattr(out, "request_id", "") or ""),
                            )
                        )
                except Exception:
                    logger.exception("Wave-Slice escape-lane core-output reorder failed.")
            finished_request_ids: list[str] = []
            for _virt_rank, engine_core_outputs in (outputs or {}).items():
                for engine_core_output in engine_core_outputs or []:
                    request_id = str(getattr(engine_core_output, "request_id", "") or "")
                    if not request_id:
                        continue
                    seen_request_ids.append(request_id)
                    if getattr(engine_core_output, "finish_reason", None) is not None:
                        finished_request_ids.append(request_id)
            state.metrics.record_escape_lane_observation(
                active_ids=active_ids,
                seen_request_ids=seen_request_ids,
                finished_request_ids=finished_request_ids,
            )
            if finished_request_ids:
                state.phase2_runtime_last_finished_request_ids = finished_request_ids
                clear_ids = (
                    [rid for rid in finished_request_ids if rid in active_ids]
                    if bool(getattr(state.policy, "phase2_enable_execution_escape", False))
                    else finished_request_ids
                )
                if clear_ids:
                    phase12_clear_escape_lane(state, request_ids=clear_ids)
        except Exception:
            logger.exception("Wave-Slice Scheduler.update_from_output hook failed.")
        return outputs

    _wave_v1_scheduler_update_from_output_hook.__wave_slice_lifecycle_hook__ = True  # type: ignore[attr-defined]
    return _wave_v1_scheduler_update_from_output_hook


def build_v1_scheduler_finish_requests_hook(
    state: Any,
    *,
    phase12_clear_escape_lane: Callable[..., Any],
) -> Callable[..., Any]:
    original_finish = state.original_scheduler_finish_requests
    if original_finish is None:
        raise RuntimeError("internal error: original Scheduler.finish_requests is missing")

    @functools.wraps(original_finish)
    def _wave_v1_scheduler_finish_requests_hook(self: Any, *args: Any, **kwargs: Any) -> Any:
        seen_request_ids: list[str] = []
        active_ids = set(getattr(state, "phase2_escape_active_ids", set()) or set())
        try:
            request_ids = args[0] if len(args) >= 1 else kwargs.get("request_ids")
            finished_status = args[1] if len(args) >= 2 else kwargs.get("finished_status")
            if isinstance(request_ids, str):
                seen_request_ids = [request_ids]
            else:
                seen_request_ids = [str(v) for v in (request_ids or []) if str(v)]
            state.phase2_runtime_last_finished_request_ids = seen_request_ids
            state.phase2_runtime_last_finished_status = str(finished_status) if finished_status is not None else None
            state.metrics.record_escape_lane_observation(
                active_ids=active_ids,
                finished_request_ids=seen_request_ids,
            )
        except Exception:
            logger.exception("Wave-Slice Scheduler.finish_requests pre-hook failed.")
        if bool(getattr(state.policy, "phase2_enable_execution_escape", False)):
            try:
                if (not active_ids) and state.phase2_deferred_finish_ids:
                    flush_ids = list(state.phase2_deferred_finish_ids)
                    flush_status = state.phase2_deferred_finish_status
                    state.phase2_deferred_finish_ids = []
                    state.phase2_deferred_finish_status = None
                    original_finish(self, flush_ids, flush_status)
                if active_ids and bool(getattr(state.policy, "phase2_execution_escape_defer_finished_nonactive", True)):
                    active_now = [rid for rid in seen_request_ids if rid in active_ids]
                    deferred_now = [rid for rid in seen_request_ids if rid not in active_ids]
                    result = None
                    if active_now:
                        result = original_finish(self, active_now, finished_status)
                    if deferred_now:
                        state.phase2_deferred_finish_ids.extend(deferred_now)
                        state.phase2_deferred_finish_status = finished_status
                    return result
            except Exception:
                logger.exception("Wave-Slice Scheduler.finish_requests execution-escape hook failed.")
        result = original_finish(self, *args, **kwargs)
        try:
            if seen_request_ids:
                clear_ids = [] if bool(getattr(state.policy, "phase2_enable_execution_escape", False)) else seen_request_ids
                if clear_ids:
                    phase12_clear_escape_lane(state, request_ids=clear_ids)
        except Exception:
            logger.exception("Wave-Slice Scheduler.finish_requests post-hook failed.")
        return result

    _wave_v1_scheduler_finish_requests_hook.__wave_slice_lifecycle_hook__ = True  # type: ignore[attr-defined]
    return _wave_v1_scheduler_finish_requests_hook
