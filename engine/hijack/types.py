from __future__ import annotations

import collections
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, Optional

from engine.base_slicer import SlicePlan, WaveBaseSlicer
from engine.hijack.runtime_state import WaveSliceMetrics, WaveSlicePolicy
from scheduler.wave_scheduler import WaveScheduler


@dataclass
class _PatchState:
    scheduler_cls: type
    scheduler_method_name: str
    original_schedule: Callable[..., Any]
    brain: WaveScheduler
    policy: WaveSlicePolicy
    model_name: str
    original_scheduler_add_request: Optional[Callable[..., Any]] = None
    original_public_schedule: Optional[Callable[..., Any]] = None
    metrics: WaveSliceMetrics = field(default_factory=WaveSliceMetrics)
    slicer: WaveBaseSlicer = field(default_factory=WaveBaseSlicer)
    model_runner_cls: Optional[type] = None
    original_execute_model: Optional[Callable[..., Any]] = None
    llm_engine_cls: Optional[type] = None
    original_add_request: Optional[Callable[..., Any]] = None
    original_add_processed_request: Optional[Callable[..., Any]] = None
    original_step: Optional[Callable[..., Any]] = None
    v1_processor_cls: Optional[type] = None
    original_v1_processor_process_inputs: Optional[Callable[..., Any]] = None
    v1_engine_core_cls: Optional[type] = None
    original_v1_engine_core_add_request: Optional[Callable[..., Any]] = None
    v1_output_processor_cls: Optional[type] = None
    original_output_processor_add_request: Optional[Callable[..., Any]] = None
    original_output_processor_process_outputs: Optional[Callable[..., Any]] = None
    original_scheduler_update_after_schedule: Optional[Callable[..., Any]] = None
    original_scheduler_update_from_output: Optional[Callable[..., Any]] = None
    original_scheduler_finish_requests: Optional[Callable[..., Any]] = None
    logits_processor_lora_cls: Optional[type] = None
    original_lora_get_logits: Optional[Callable[..., Any]] = None
    sequence_data_cls: Optional[type] = None
    original_sequence_data_get_len: Optional[Callable[..., Any]] = None
    v1_request_cls: Optional[type] = None
    original_v1_request_num_tokens: Optional[Any] = None
    original_v1_request_num_tokens_with_spec: Optional[Any] = None
    original_get_new_uncached_and_cached_tokens: Optional[Callable[..., Any]] = None
    phase1_sticky_req_id: Optional[str] = None
    phase1_sticky_chunk: Optional[int] = None
    phase1_sticky_ttl_left: int = 0
    phase1_explicit_plans: dict[str, list[SlicePlan]] = field(default_factory=dict)
    phase1_shadow_seq_lens: dict[int, int] = field(default_factory=dict)
    phase1_virtual_token_caps: dict[str, int] = field(default_factory=dict)
    phase1_active_prompt_tokens: dict[str, int] = field(default_factory=dict)
    phase1_ingress_virtuals: dict[str, _Phase1IngressVirtualSlice] = field(default_factory=dict)
    phase1_public_skip_rewrite_requests: set[str] = field(default_factory=set)
    phase12_recent_phase1_apply_ttl: int = 0
    phase12_last_phase1_req_id: Optional[str] = None
    phase12_recent_phase1_strength: float = 0.0
    phase12_recent_phase1_chunk: int = 0
    phase12_recent_phase2_cashout_cooldown: int = 0
    phase2_escape_active_ids: set[str] = field(default_factory=set)
    phase2_escape_deferred_ids: set[str] = field(default_factory=set)
    phase2_escape_lane_ttl: int = 0
    phase2_deferred_request_outputs: list[Any] = field(default_factory=list)
    phase2_deferred_finish_ids: list[str] = field(default_factory=list)
    phase2_deferred_finish_status: Any = None
    phase2_runtime_last_output_request_ids: list[str] = field(default_factory=list)
    phase2_runtime_last_finished_request_ids: list[str] = field(default_factory=list)
    phase2_runtime_last_finished_status: Optional[str] = None
    v1_lifecycle_hooks_installed: bool = False


@dataclass
class _Phase2Decision:
    apply: bool
    reason: str
    prefill_lens: list[int]
    num_prefills: int
    num_decode_tokens: int
    lora_ranks: list[int] = field(default_factory=list)


@dataclass
class _RunnerStreamState:
    device: Any
    fast_stream: Any
    inflight_events: Deque[Any] = field(default_factory=collections.deque)


@dataclass(frozen=True)
class _ScheduledReqInfo:
    request_id: str
    scheduled_tokens: int
    remaining_tokens: int
    expected_chunk_tokens: int
    input_tokens: Optional[int]
    arrival_s: Optional[float]
    is_short: bool
    lora_rank: int


@dataclass(frozen=True)
class _Phase12BeneficiarySignal:
    long_anchor_id: Optional[str]
    beneficiary_prefill_ids: list[str]
    beneficiary_prefill_count: int
    beneficiary_fraction: float
    beneficiary_wait_quality: float
    beneficiary_size_quality: float
    beneficiary_cashout_quality: float
    beneficiary_selected_quality: float
    beneficiary_selected_ids: list[str]
    beneficiary_score_map: dict[str, float]


@dataclass(frozen=True)
class _Phase1CohortStats:
    representative_short_len: int
    short_count: int
    short_token_mass: int
    short_lengths: list[int]
    long_len: int
    long_req_id: Optional[str]
    total_count: int


@dataclass(frozen=True)
class _Phase1IngressVirtualSlice:
    long_req_id: str
    representative_short_len: int
    short_count: int
    short_token_mass: int
    short_lengths: list[int]
    original_long_len: int
    active_count: int
