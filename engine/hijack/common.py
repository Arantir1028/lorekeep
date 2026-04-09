from __future__ import annotations

import collections
import json
import os
import re
import time
from typing import Any, Iterable, Optional

from scheduler.wave_scheduler import WaveScheduler

_LORA_RANK_CACHE: dict[str, int] = {}
_LORA_RANK_RE = re.compile(r"rank[_-]?(\d+)", re.IGNORECASE)


def is_phase2_strict(policy: Any) -> bool:
    return str(policy.phase2_consistency_mode).strip().lower() == "strict"


def is_phase2_async_experimental(policy: Any) -> bool:
    return str(policy.phase2_dispatch_mode).strip().lower() == "async_experimental"


def safe_first_seq(seq_group: Any) -> Optional[Any]:
    try:
        return next(iter(seq_group.get_seqs()), None)
    except Exception:
        return None


def safe_total_tokens(seq_group: Any) -> Optional[int]:
    seq = safe_first_seq(seq_group)
    if seq is None:
        for attr in ("num_tokens_with_spec", "num_prompt_tokens"):
            val = getattr(seq_group, attr, None)
            if val is not None:
                try:
                    return int(val)
                except Exception:
                    pass
        return None
    try:
        return int(seq.get_len())
    except Exception:
        return None


def safe_remaining_tokens(seq_group: Any) -> Optional[int]:
    seq = safe_first_seq(seq_group)
    if seq is None:
        try:
            total = int(getattr(seq_group, "num_tokens_with_spec"))
            done = int(getattr(seq_group, "num_computed_tokens"))
            return max(0, total - done)
        except Exception:
            return None
    try:
        total = int(seq.get_len())
        done = int(seq.data.get_num_computed_tokens())
    except Exception:
        return None
    return max(0, total - done)


def safe_prefill_uncomputed_tokens(seq_group: Any) -> Optional[int]:
    seq = safe_first_seq(seq_group)
    if seq is None:
        is_prefill = False
        is_prefill_attr = getattr(seq_group, "is_prefill", None)
        if callable(is_prefill_attr):
            try:
                is_prefill = bool(is_prefill_attr())
            except Exception:
                is_prefill = False
        elif is_prefill_attr is not None:
            try:
                is_prefill = bool(is_prefill_attr)
            except Exception:
                is_prefill = False
        if not is_prefill:
            try:
                num_prompt_tokens = int(getattr(seq_group, "num_prompt_tokens"))
                num_computed_tokens = int(getattr(seq_group, "num_computed_tokens"))
                is_prefill = num_computed_tokens < num_prompt_tokens
            except Exception:
                is_prefill = False
        if not is_prefill:
            return 0
        try:
            num_prompt_tokens = int(getattr(seq_group, "num_prompt_tokens"))
            num_computed_tokens = int(getattr(seq_group, "num_computed_tokens"))
            return max(0, num_prompt_tokens - num_computed_tokens)
        except Exception:
            pass
        try:
            total = int(getattr(seq_group, "num_tokens_with_spec"))
            done = int(getattr(seq_group, "num_computed_tokens"))
            return max(0, total - done)
        except Exception:
            return None
    try:
        if not bool(seq.is_prefill()):
            return 0
    except Exception:
        return 0
    try:
        return max(0, int(seq.get_num_uncomputed_tokens()))
    except Exception:
        pass
    try:
        total = int(seq.get_len())
        done = int(seq.data.get_num_computed_tokens())
        return max(0, total - done)
    except Exception:
        return None


def phase12_expected_chunk_tokens(seq_group: Any, *, state: Any, remaining: int) -> int:
    remaining_i = max(0, int(remaining))
    if remaining_i <= 0:
        return 0
    try:
        token_chunk_size = int(getattr(seq_group, "token_chunk_size", 0) or 0)
        if token_chunk_size > 0:
            return max(1, min(remaining_i, token_chunk_size))
    except Exception:
        pass
    req_id = safe_request_id(seq_group)
    if state is not None and req_id:
        try:
            virtual = int((state.phase1_virtual_token_caps or {}).get(str(req_id), 0) or 0)
            if virtual > 0:
                return max(1, min(remaining_i, virtual))
        except Exception:
            pass
        recent_chunk = max(0, int(getattr(state, "phase12_recent_phase1_chunk", 0) or 0))
        if recent_chunk > 0:
            return max(1, min(remaining_i, recent_chunk))
    fallback = min(remaining_i, 512)
    return max(1, fallback)


def safe_request_id(seq_group: Any) -> Optional[str]:
    for attr in ("request_id", "req_id", "id"):
        val = getattr(seq_group, attr, None)
        if val is not None:
            return str(val)
    seq = safe_first_seq(seq_group)
    if seq is None:
        return None
    for attr in ("request_id", "req_id", "id"):
        val = getattr(seq, attr, None)
        if val is not None:
            return str(val)
    return None


def safe_wait_us(seq_group: Any, now_s: float) -> float:
    try:
        arrival_s = float(seq_group.metrics.arrival_time)
    except Exception:
        try:
            arrival_s = float(getattr(seq_group, "arrival_time"))
        except Exception:
            return 0.0
    if arrival_s <= 0:
        return 0.0
    return max(0.0, (now_s - arrival_s) * 1e6)


def queue_reorder_key(seq_group: Any, *, brain: WaveScheduler, now_s: float, mode: str, aging_quantum_us: float) -> Any:
    remaining = max(1, int(safe_remaining_tokens(seq_group) or 1))
    service_us = estimate_solo_us(brain, remaining) or float(remaining)
    wait_us = safe_wait_us(seq_group, now_s)
    mode = str(mode or "sjf").strip().lower()
    if mode == "hrrn":
        response_ratio = (wait_us + service_us) / max(1.0, service_us)
        return (-response_ratio, service_us, remaining)
    if mode == "aging":
        quantum = max(1.0, float(aging_quantum_us))
        aged_service = service_us / (1.0 + (wait_us / quantum))
        return (aged_service, service_us, remaining)
    return (service_us, remaining)


def reorder_queue(queue_like: Iterable[Any], *, brain: WaveScheduler, now_s: float, mode: str, aging_quantum_us: float) -> Any:
    queue = list(queue_like)
    queue.sort(key=lambda sg: queue_reorder_key(sg, brain=brain, now_s=now_s, mode=mode, aging_quantum_us=aging_quantum_us))
    queue_type = type(queue_like)
    if isinstance(queue_like, collections.deque):
        try:
            return queue_type(queue)
        except Exception:
            return collections.deque(queue)
    if isinstance(queue_like, list):
        return queue
    if hasattr(queue_like, "add_request"):
        try:
            rebuilt = queue_type()
            for item in queue:
                rebuilt.add_request(item)
            return rebuilt
        except Exception:
            pass
    return queue


def rebuild_queue_like(queue_like: Any, items: Iterable[Any]) -> Any:
    materialized = list(items)
    queue_type = type(queue_like)
    if isinstance(queue_like, list):
        return materialized
    if hasattr(queue_like, "add_request"):
        try:
            rebuilt = queue_type()
            for item in materialized:
                rebuilt.add_request(item)
            return rebuilt
        except Exception:
            pass
    if isinstance(queue_like, collections.deque):
        try:
            return queue_type(materialized)
        except Exception:
            return collections.deque(materialized)
    try:
        return queue_type(materialized)
    except Exception:
        return materialized


def collect_live_lengths(waiting: Iterable[Any], running: Iterable[Any]) -> tuple[list[int], float]:
    lengths: list[int] = []
    max_wait_us = 0.0
    now_s = time.time()
    for sg in waiting:
        remaining = safe_prefill_uncomputed_tokens(sg)
        if remaining and remaining > 0:
            lengths.append(remaining)
            max_wait_us = max(max_wait_us, safe_wait_us(sg, now_s))
    for sg in running:
        remaining = safe_prefill_uncomputed_tokens(sg)
        if remaining and remaining > 0:
            lengths.append(remaining)
    return lengths, max_wait_us


def collect_live_snapshot(waiting: Iterable[Any], running: Iterable[Any]) -> tuple[list[tuple[Any, int]], float]:
    snapshot: list[tuple[Any, int]] = []
    max_wait_us = 0.0
    now_s = time.time()
    for sg in list(waiting) + list(running):
        remaining = safe_prefill_uncomputed_tokens(sg)
        if remaining and remaining > 0:
            rem = int(remaining)
            snapshot.append((sg, rem))
            max_wait_us = max(max_wait_us, safe_wait_us(sg, now_s))
    return snapshot, max_wait_us


def compute_long_prefill_threshold(best_chunk: int, original_threshold: Any, scheduler_obj: Any) -> Optional[int]:
    if best_chunk <= 0:
        return None
    max_model_len = None
    try:
        scheduler_cfg = getattr(scheduler_obj, "scheduler_config", None)
        max_model_len = getattr(scheduler_cfg, "max_model_len", None)
    except Exception:
        max_model_len = None
    threshold = int(best_chunk)
    if isinstance(max_model_len, int) and max_model_len > 0:
        threshold = min(threshold, max_model_len)
    if isinstance(original_threshold, int) and original_threshold > 0:
        threshold = min(threshold, max(original_threshold, 1))
    return max(1, threshold)


def estimate_prompt_tokens(prompt_or_ids: Any, *, engine_self: Any = None, lora_request: Any = None) -> Optional[int]:
    if prompt_or_ids is None:
        return None
    if isinstance(prompt_or_ids, dict):
        prompt_token_ids = prompt_or_ids.get("prompt_token_ids")
        if isinstance(prompt_token_ids, (list, tuple)):
            return len(prompt_token_ids)
        prompt_text = prompt_or_ids.get("prompt")
        if isinstance(prompt_text, str):
            prompt_or_ids = prompt_text
    if isinstance(prompt_or_ids, str):
        if engine_self is not None:
            try:
                tokenizer = engine_self.get_tokenizer(lora_request=lora_request)
            except Exception:
                tokenizer = None
            if tokenizer is not None:
                try:
                    encoded = tokenizer.encode(prompt_or_ids, add_special_tokens=False)
                    if isinstance(encoded, (list, tuple)):
                        return len(encoded)
                except TypeError:
                    try:
                        encoded = tokenizer.encode(prompt_or_ids)
                        if isinstance(encoded, (list, tuple)):
                            return len(encoded)
                    except Exception:
                        pass
                except Exception:
                    pass
        return max(1, len(prompt_or_ids.split()))
    if isinstance(prompt_or_ids, (list, tuple)):
        return len(prompt_or_ids)
    return None


def estimate_solo_us(brain: WaveScheduler, input_tokens: Optional[int]) -> Optional[float]:
    if input_tokens is None or input_tokens <= 0:
        return None
    try:
        bucket = brain._conservative_map_up(input_tokens)
        return float(brain.t_solo_dict.get(bucket, 0.0)) or None
    except Exception:
        return None


def safe_lora_path(lora_request: Any) -> Optional[str]:
    if lora_request is None:
        return None
    for attr in ("lora_path", "path", "lora_local_path", "local_path"):
        try:
            val = getattr(lora_request, attr, None)
        except Exception:
            val = None
        if isinstance(val, str) and val.strip():
            return val.strip()
    return None


def extract_rank_from_text(text: str) -> int:
    if not text:
        return 0
    match = _LORA_RANK_RE.search(text)
    if match is None:
        return 0
    try:
        return max(0, int(match.group(1)))
    except Exception:
        return 0


def infer_lora_rank(lora_request: Any) -> int:
    if lora_request is None:
        return 0

    path = safe_lora_path(lora_request)
    if path:
        cached = _LORA_RANK_CACHE.get(path)
        if cached is not None:
            return cached

        rank = 0
        cfg_path = os.path.join(path, "adapter_config.json")
        if os.path.exists(cfg_path):
            try:
                with open(cfg_path, "r", encoding="utf-8") as f:
                    payload = json.load(f)
                rank = int(payload.get("r") or payload.get("rank") or 0)
            except Exception:
                rank = 0

        if rank <= 0:
            rank = extract_rank_from_text(path)
        if rank <= 0:
            rank = extract_rank_from_text(str(getattr(lora_request, "lora_name", "") or ""))

        rank = max(0, int(rank))
        _LORA_RANK_CACHE[path] = rank
        return rank

    return extract_rank_from_text(str(getattr(lora_request, "lora_name", "") or ""))
