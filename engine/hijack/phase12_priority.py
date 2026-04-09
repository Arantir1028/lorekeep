from __future__ import annotations

import math
from typing import Any, Callable

from engine.hijack.types import _Phase12BeneficiarySignal

RemainingGetter = Callable[[Any], int | None]
WaitGetter = Callable[[Any, float], float]
RequestIdGetter = Callable[[Any], str | None]
SoloUsEstimator = Callable[[Any, int], float | None]
QueueRebuilder = Callable[[Any, list[Any]], Any]


def phase12_priority_score(
    seq_group: Any,
    *,
    beneficiary_signal: _Phase12BeneficiarySignal,
    beneficiary_ids: set[str],
    strength: float,
    brain: Any,
    now_s: float,
    remaining_getter: RemainingGetter,
    wait_getter: WaitGetter,
    request_id_getter: RequestIdGetter,
    solo_us_estimator: SoloUsEstimator,
) -> tuple[float, float, int]:
    remaining = max(1, int(remaining_getter(seq_group) or 1))
    service_us = solo_us_estimator(brain, remaining) or float(remaining)
    wait_us = wait_getter(seq_group, now_s)
    wait_quality = min(1.0, wait_us / max(1.0, 200000.0))
    short_quality = 1.0 / math.sqrt(float(max(1, remaining)))
    request_id = request_id_getter(seq_group)
    request_id_str = str(request_id) if request_id is not None else ""
    beneficiary_quality = float(beneficiary_signal.beneficiary_score_map.get(request_id_str, 0.0))
    try:
        is_prefill = bool(seq_group.is_prefill())
    except Exception:
        is_prefill = False
    decode_bonus = 0.15 if not is_prefill else 0.0
    beneficiary_bonus = (1.25 + (0.75 * strength)) if request_id_str in beneficiary_ids else 0.0
    score = (
        beneficiary_bonus
        + (0.55 * beneficiary_quality)
        + (0.20 * wait_quality)
        + (0.15 * short_quality)
        + decode_bonus
    )
    return (-score, service_us, remaining)


def phase12_priority_bubble_waiting_queue(
    queue_obj: Any,
    *,
    beneficiary_signal: _Phase12BeneficiarySignal,
    beneficiary_ids: set[str],
    strength: float,
    brain: Any,
    now_s: float,
    remaining_getter: RemainingGetter,
    wait_getter: WaitGetter,
    request_id_getter: RequestIdGetter,
    solo_us_estimator: SoloUsEstimator,
    queue_rebuilder: QueueRebuilder,
) -> Any:
    queue = list(queue_obj)
    if not queue or not beneficiary_ids:
        return queue_rebuilder(queue_obj, queue)

    max_promotions = max(1, min(len(beneficiary_ids), int(round(1 + strength))))
    max_shift = max(1, min(2, int(round(1 + strength))))
    top_window = max(4, min(len(queue), 6 + max_shift))

    scored_ids = sorted(
        (rid for rid in beneficiary_ids if rid),
        key=lambda rid: float(beneficiary_signal.beneficiary_score_map.get(str(rid), 0.0)),
        reverse=True,
    )[:max_promotions]

    for rid in scored_ids:
        idx = next(
            (i for i, sg in enumerate(queue) if str(request_id_getter(sg) or "") == str(rid)),
            None,
        )
        if idx is None:
            continue
        moves = 0
        while idx > 0 and moves < max_shift:
            prev_idx = idx - 1
            if idx >= top_window and prev_idx >= top_window:
                break
            cur = queue[idx]
            prev = queue[prev_idx]
            prev_id = str(request_id_getter(prev) or "")
            if prev_id in beneficiary_ids:
                break
            cur_key = phase12_priority_score(
                cur,
                beneficiary_signal=beneficiary_signal,
                beneficiary_ids=beneficiary_ids,
                strength=strength,
                brain=brain,
                now_s=now_s,
                remaining_getter=remaining_getter,
                wait_getter=wait_getter,
                request_id_getter=request_id_getter,
                solo_us_estimator=solo_us_estimator,
            )
            prev_key = phase12_priority_score(
                prev,
                beneficiary_signal=beneficiary_signal,
                beneficiary_ids=beneficiary_ids,
                strength=strength,
                brain=brain,
                now_s=now_s,
                remaining_getter=remaining_getter,
                wait_getter=wait_getter,
                request_id_getter=request_id_getter,
                solo_us_estimator=solo_us_estimator,
            )
            if cur_key >= prev_key:
                break
            queue[prev_idx], queue[idx] = queue[idx], queue[prev_idx]
            idx = prev_idx
            moves += 1

    return queue_rebuilder(queue_obj, queue)
