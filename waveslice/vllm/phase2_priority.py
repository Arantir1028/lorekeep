from __future__ import annotations

from collections.abc import Callable
from typing import Any

from waveslice.vllm.state import Phase12BeneficiarySignal

RequestIdGetter = Callable[[Any], str | None]
QueueRebuilder = Callable[[Any, list[Any]], Any]


def phase12_priority_bubble_waiting_queue(
    queue_obj: Any,
    *,
    beneficiary_signal: Phase12BeneficiarySignal,
    beneficiary_ids: set[str],
    request_id_getter: RequestIdGetter,
    queue_rebuilder: QueueRebuilder,
) -> Any:
    queue = list(queue_obj)
    if not queue or not beneficiary_ids:
        return queue_rebuilder(queue_obj, queue)
    selected = [item for item in queue if str(request_id_getter(item) or "") in beneficiary_ids]
    selected.sort(
        key=lambda item: beneficiary_signal.beneficiary_score_map.get(
            str(request_id_getter(item) or ""), 0.0
        ),
        reverse=True,
    )
    rest = [item for item in queue if str(request_id_getter(item) or "") not in beneficiary_ids]
    return queue_rebuilder(queue_obj, selected + rest)
