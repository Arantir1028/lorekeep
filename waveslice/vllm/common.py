"""Small adapters shared by the WaveSlice vLLM hooks."""

from __future__ import annotations

import collections
import json
import os
import re
import time
from collections.abc import Callable, Iterable
from typing import Any

from waveslice.scheduling.scheduler import WaveScheduler

_RANK_CACHE: dict[str, int] = {}
_RANK_RE = re.compile(r"rank[_-]?(\d+)", re.I)


def safe_first_seq(group: Any) -> Any | None:
    get_seqs = getattr(group, "get_seqs", None)
    return next(iter(get_seqs()), None) if callable(get_seqs) else None


def _int_attr(obj: Any, *names: str) -> int | None:
    for name in names:
        value = getattr(obj, name, None)
        if value is not None:
            return int(value)
    return None


def safe_total_tokens(group: Any) -> int | None:
    seq = safe_first_seq(group)
    return (
        int(seq.get_len())
        if seq is not None
        else _int_attr(group, "num_tokens_with_spec", "num_prompt_tokens")
    )


def safe_remaining_tokens(group: Any) -> int | None:
    seq = safe_first_seq(group)
    if seq is not None:
        return max(0, int(seq.get_len()) - int(seq.data.get_num_computed_tokens()))
    total = _int_attr(group, "num_tokens_with_spec")
    done = _int_attr(group, "num_computed_tokens")
    return None if total is None or done is None else max(0, total - done)


def safe_prefill_uncomputed_tokens(group: Any) -> int | None:
    seq = safe_first_seq(group)
    if seq is not None:
        return max(0, int(seq.get_num_uncomputed_tokens())) if seq.is_prefill() else 0
    prefill = getattr(group, "is_prefill", None)
    prefill = prefill() if callable(prefill) else prefill
    prompt, computed = (
        _int_attr(group, name) for name in ("num_prompt_tokens", "num_computed_tokens")
    )
    if prefill is None:
        prefill = prompt is not None and computed is not None and computed < prompt
    if not prefill:
        return 0
    total = prompt if prompt is not None else _int_attr(group, "num_tokens_with_spec")
    return None if total is None or computed is None else max(0, total - computed)


def safe_request_id(group: Any) -> str | None:
    for obj in (group, safe_first_seq(group)):
        if obj is not None and (value := getattr(obj, "request_id", None)) is not None:
            return str(value)
    return None


def phase12_expected_chunk_tokens(group: Any, *, state: Any, remaining: int) -> int:
    remaining = max(0, int(remaining))
    if not remaining:
        return 0
    chunk = int(getattr(group, "token_chunk_size", 0) or 0)
    request_id = safe_request_id(group)
    if not chunk and state is not None and request_id:
        chunk = int(state.phase1_virtual_token_caps.get(request_id, 0) or 0)
    if not chunk and state is not None:
        chunk = int(getattr(state, "phase12_recent_phase1_chunk", 0) or 0)
    return min(remaining, max(1, chunk or 512))


def safe_wait_us(group: Any, now_s: float) -> float:
    metrics = getattr(group, "metrics", None)
    arrival = getattr(metrics, "arrival_time", getattr(group, "arrival_time", 0.0))
    return max(0.0, (now_s - float(arrival or 0.0)) * 1e6) if arrival else 0.0


def estimate_solo_us(brain: WaveScheduler, tokens: int | None) -> float | None:
    if not tokens or tokens <= 0:
        return None
    value = float(brain.t_solo_dict.get(brain._conservative_map_up(tokens), 0.0))
    return value or None


def queue_reorder_key(
    group: Any, *, brain: WaveScheduler, now_s: float, mode: str, aging_quantum_us: float
) -> Any:
    remaining = max(1, int(safe_remaining_tokens(group) or 1))
    service = estimate_solo_us(brain, remaining) or float(remaining)
    wait = safe_wait_us(group, now_s)
    if mode.strip().lower() == "hrrn":
        return (-(wait + service) / max(1.0, service), service, remaining)
    if mode.strip().lower() == "aging":
        return (service / (1.0 + wait / max(1.0, aging_quantum_us)), service, remaining)
    return service, remaining


def rebuild_queue_like(queue: Any, items: Iterable[Any]) -> Any:
    items = list(items)
    if isinstance(queue, list):
        return items
    if isinstance(queue, collections.deque):
        return type(queue)(items)
    if hasattr(queue, "add_request"):
        rebuilt = type(queue)()
        for item in items:
            rebuilt.add_request(item)
        return rebuilt
    return type(queue)(items)


def restore_hidden_queue_items(
    queue_obj: Any,
    hidden_items: list[Any],
    *,
    queue_rebuilder: Callable[[Any, list[Any]], Any],
) -> Any:
    if not hidden_items:
        return queue_obj
    return queue_rebuilder(queue_obj, [*hidden_items, *queue_obj])


def has_running_waiting_queues(obj: Any) -> bool:
    return hasattr(obj, "running") and hasattr(obj, "waiting")


def phase2_scheduler_cashout_enabled(policy: Any) -> bool:
    return bool(getattr(policy, "phase2_enable_scheduler_cashout", False))


def reorder_queue(
    queue: Iterable[Any], *, brain: WaveScheduler, now_s: float, mode: str, aging_quantum_us: float
) -> Any:
    ordered = sorted(
        queue,
        key=lambda group: queue_reorder_key(
            group, brain=brain, now_s=now_s, mode=mode, aging_quantum_us=aging_quantum_us
        ),
    )
    return rebuild_queue_like(queue, ordered)


def collect_live_snapshot(waiting: Iterable[Any], running: Iterable[Any]):
    now = time.time()
    snapshot = [
        (group, int(remaining))
        for group in [*waiting, *running]
        if (remaining := safe_prefill_uncomputed_tokens(group)) and remaining > 0
    ]
    wait = max((safe_wait_us(group, now) for group, _ in snapshot), default=0.0)
    return snapshot, wait


def compute_long_prefill_threshold(
    best_chunk: int, original_threshold: Any, scheduler_obj: Any
) -> int | None:
    if best_chunk <= 0:
        return None
    config = getattr(scheduler_obj, "scheduler_config", None)
    candidates = [int(best_chunk)]
    for value in (getattr(config, "max_model_len", None), original_threshold):
        if isinstance(value, int) and value > 0:
            candidates.append(value)
    return max(1, min(candidates))


def estimate_prompt_tokens(
    prompt: Any, *, engine_self: Any = None, lora_request: Any = None
) -> int | None:
    if prompt is None:
        return None
    if isinstance(prompt, dict):
        ids = prompt.get("prompt_token_ids")
        if ids is not None:
            return len(ids)
        prompt = prompt.get("prompt")
    if isinstance(prompt, (list, tuple)):
        return len(prompt)
    if not isinstance(prompt, str):
        return None
    tokenizer = engine_self.get_tokenizer(lora_request=lora_request) if engine_self else None
    return (
        len(tokenizer.encode(prompt, add_special_tokens=False)) if tokenizer is not None else None
    )


def safe_lora_path(request: Any) -> str | None:
    if request is None:
        return None
    value = getattr(request, "lora_path", None)
    return value.strip() if isinstance(value, str) and value.strip() else None


def extract_rank_from_text(text: str) -> int:
    match = _RANK_RE.search(text or "")
    return max(0, int(match.group(1))) if match else 0


def infer_lora_rank(request: Any) -> int:
    if request is None:
        return 0
    path = safe_lora_path(request)
    if not path:
        return extract_rank_from_text(str(getattr(request, "lora_name", "") or ""))
    if path not in _RANK_CACHE:
        rank = 0
        config = os.path.join(path, "adapter_config.json")
        if os.path.exists(config):
            with open(config, encoding="utf-8") as stream:
                payload = json.load(stream)
            rank = int(payload.get("r") or payload.get("rank") or 0)
        _RANK_CACHE[path] = max(
            rank,
            extract_rank_from_text(path),
            extract_rank_from_text(str(getattr(request, "lora_name", "") or "")),
        )
    return _RANK_CACHE[path]
