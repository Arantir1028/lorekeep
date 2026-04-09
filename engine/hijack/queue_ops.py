from __future__ import annotations

from typing import Any, Callable


QueueRebuilder = Callable[[Any, list[Any]], Any]


def restore_hidden_queue_items(
    queue_obj: Any,
    hidden_items: list[Any],
    *,
    queue_rebuilder: QueueRebuilder,
) -> Any:
    if not hidden_items:
        return queue_obj
    current = list(queue_obj)
    restored = list(hidden_items) + current
    return queue_rebuilder(queue_obj, restored)


def capture_queue_pair(owner: Any) -> tuple[Any, Any]:
    return getattr(owner, "running", None), getattr(owner, "waiting", None)


def restore_queue_pair(owner: Any, running: Any, waiting: Any) -> None:
    if running is not None and hasattr(owner, "running"):
        owner.running = running
    if waiting is not None and hasattr(owner, "waiting"):
        owner.waiting = waiting
