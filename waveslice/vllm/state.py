from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from waveslice.metrics import WaveSliceMetrics
from waveslice.policy import WaveSlicePolicy
from waveslice.scheduling.scheduler import WaveScheduler
from waveslice.scheduling.slicer import SlicePlan, WaveBaseSlicer


@dataclass
class RuntimeState:
    scheduler_cls: type
    original_schedule: Callable[..., Any]
    brain: WaveScheduler
    policy: WaveSlicePolicy
    model_name: str
    original_scheduler_add_request: Callable[..., Any] | None = None
    metrics: WaveSliceMetrics = field(default_factory=WaveSliceMetrics)
    slicer: WaveBaseSlicer = field(default_factory=WaveBaseSlicer)
    llm_engine_cls: type | None = None
    original_add_request: Callable[..., Any] | None = None
    original_step: Callable[..., Any] | None = None
    v1_processor_cls: type | None = None
    original_v1_processor_process_inputs: Callable[..., Any] | None = None
    v1_engine_core_cls: type | None = None
    original_v1_engine_core_add_request: Callable[..., Any] | None = None
    original_scheduler_update_after_schedule: Callable[..., Any] | None = None
    v1_request_cls: type | None = None
    original_v1_request_num_tokens: Any | None = None
    original_v1_request_num_tokens_with_spec: Any | None = None
    phase1_explicit_plans: dict[str, list[SlicePlan]] = field(default_factory=dict)
    phase1_shadow_seq_lens: dict[int, int] = field(default_factory=dict)
    phase1_virtual_token_caps: dict[str, int] = field(default_factory=dict)
    phase1_active_prompt_tokens: dict[str, int] = field(default_factory=dict)
    phase1_ingress_virtuals: dict[str, Phase1IngressVirtualSlice] = field(default_factory=dict)
    phase1_public_skip_rewrite_requests: set[str] = field(default_factory=set)
    phase12_recent_phase1_apply_ttl: int = 0
    phase12_last_phase1_req_id: str | None = None
    phase12_recent_phase1_strength: float = 0.0
    phase12_recent_phase1_chunk: int = 0
    phase12_recent_phase2_cashout_cooldown: int = 0
    phase1_runtime_pressure_ema: float = 0.0
    phase1_runtime_wall_pressure_ema: float = 0.0
    phase1_runtime_last_meta: dict[str, Any] = field(default_factory=dict)
    phase2_priority_active_ids: set[str] = field(default_factory=set)
    phase2_priority_deferred_ids: set[str] = field(default_factory=set)
    phase2_priority_lane_ttl: int = 0


@dataclass(frozen=True)
class ScheduledRequestInfo:
    request_id: str
    scheduled_tokens: int
    remaining_tokens: int
    expected_chunk_tokens: int
    input_tokens: int | None
    arrival_s: float | None
    is_short: bool
    lora_rank: int


@dataclass(frozen=True)
class Phase12BeneficiarySignal:
    long_anchor_id: str | None
    beneficiary_fraction: float
    beneficiary_selected_quality: float
    beneficiary_selected_ids: list[str]
    beneficiary_score_map: dict[str, float]


@dataclass(frozen=True)
class Phase1CohortStats:
    representative_short_len: int
    short_count: int
    short_token_mass: int
    short_lengths: list[int]
    long_len: int
    long_req_id: str | None
    total_count: int


@dataclass(frozen=True)
class Phase1IngressVirtualSlice:
    long_req_id: str
    representative_short_len: int
    short_count: int
    short_token_mass: int
    short_lengths: list[int]
    original_long_len: int
    active_count: int


@dataclass(frozen=True)
class Phase1ScheduleDecision:
    snapshot: list[tuple[Any, int]]
    selected_snapshot: list[tuple[Any, int]]
    cohort: Phase1CohortStats
    long_group: Any
    queue_len: int
    max_wait_us: float
    best_chunk: int
    baseline_chunk: int | None
    explicit_kind: str | None
