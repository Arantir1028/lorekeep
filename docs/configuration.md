# Configuration reference

Applications enable WaveSlice through `waveslice.EngineArgs`. `WaveSliceConfig`
contains the LUT identity override, objective coefficient, and an immutable
`WaveSlicePolicy`.

```python
from waveslice import EngineArgs, WaveSliceConfig, WaveSlicePolicy

engine_args = EngineArgs(
    model="mistralai/Mistral-7B-v0.1",
    enable_wave_slice=True,
    wave_slice_config=WaveSliceConfig(
        gamma=2.0,
        policy=WaveSlicePolicy(),
    ),
)
```

The normal path omits `lut_model`; WaveSlice derives the LUT name from
`EngineArgs.model`. Policy values should be changed only for a controlled
experiment or a documented deployment profile.

## Public activation settings

| Setting | Default | Meaning |
| --- | --- | --- |
| `enable_wave_slice` | `False` | Select WaveSlice or the native vLLM V1 scheduler |
| `wave_slice_config` | `WaveSliceConfig()` | WaveSlice configuration object or equivalent mapping |
| `lut_model` | derived from `model` | Optional exact LUT-name override |
| `gamma` | `2.0` | Queue-pressure coefficient in the LUT objective |
| `policy` | `WaveSlicePolicy()` | Scheduler and metrics policy |

## Recommended profiles

The default policy enables Phase I and metrics while leaving Phase II disabled.
The smallest explicit Phase II profile is:

```python
WaveSlicePolicy(
    enable_phase1_scheduler=True,
    enable_phase2_scheduler=True,
    phase2_enable_scheduler_cashout=True,
)
```

Setting `enable_wave_slice=False` is the native-vLLM comparison. Do not emulate
the native comparison by selectively disabling internal Phase I fields.

## Phase I eligibility and queue handling

| Field | Default | Unit or values | Meaning |
| --- | --- | --- | --- |
| `enable_phase1_scheduler` | `True` | boolean | Enable prefill slicing decisions |
| `min_hetero_ratio` | `3.0` | ratio | Minimum long-to-short length ratio for ordinary eligibility |
| `min_long_seq` | `384` | tokens | Minimum long-prefill size for ordinary eligibility |
| `short_escape_multiplier` | `12` | multiplier | Short-length contribution to the scheduler budget proxy |
| `max_budget_cap` | `8192` | tokens | Upper bound for a Phase I-guided scheduler budget |
| `enable_sjf_reorder` | `True` | boolean | Allow waiting-queue reordering before the native call |
| `queue_reorder_mode` | `"sjf"` | `sjf`, `hrrn`, `aging` | Waiting-queue ordering rule |
| `queue_reorder_aging_quantum_us` | `20000.0` | microseconds | Aging interval used by HRRN/aging ordering |
| `allow_phase1_with_lora` | `False` | boolean | Allow the complete Phase I path when LoRA serving is active |
| `allow_phase1_threshold_with_lora` | `True` | boolean | Allow temporary threshold control for LoRA requests |
| `allow_phase1_budget_with_lora` | `False` | boolean | Allow temporary scheduler-budget control for LoRA requests |

## Phase I decision and budget controls

| Field | Default | Unit or values | Meaning |
| --- | --- | --- | --- |
| `enable_phase1_dynamic_threshold` | `True` | boolean | Allow a temporary long-prefill threshold derived from the plan |
| `enable_phase1_budget_guidance` | `True` | boolean | Allow a temporary token-budget bound derived from the plan |
| `enable_phase1_baseline_relative` | `True` | boolean | Compare the selected chunk with the native scheduler baseline |
| `enable_phase1_explicit_plan` | `True` | boolean | Retain an explicit per-request slice plan across request admission |
| `enable_phase1_direct_explicit_override` | `True` | boolean | Let a valid direct explicit plan override the LUT candidate |
| `phase1_ingress_direct_authoritative` | `True` | boolean | Install the ingress request cap before the native scheduler call |
| `scheduler_objective_mode` | `"fair_escape"` | `fair_escape`, `pure_gain` | LUT objective used by the scheduler brain |
| `phase1_force_extreme_ratio` | `6.0` | ratio | Long-to-short ratio for the extreme-ratio fallback |
| `phase1_force_queue_len` | `1` | requests | Minimum queue size for the extreme-ratio fallback |
| `phase1_force_min_chunk` | `128` | tokens | Minimum size used by forced eligibility and target calculations |
| `phase1_ingress_exact_chunk` | `True` | boolean | Use the requested ingress target instead of bucket rounding |
| `phase1_ingress_target_chunk` | `384` | tokens | Normal ingress target |
| `phase1_ingress_min_chunk` | `256` | tokens | Lower bound for a non-exact ingress target |
| `phase1_ingress_max_chunk` | `512` | tokens | Upper bound for an ingress target |
| `phase1_target_short_mul` | `4.0` | multiplier | Short-length contribution to the cohort target |
| `phase1_target_long_fraction` | `0.33` | fraction | Long-prefill contribution to the cohort target |
| `phase1_budget_short_mass_factor` | `1.75` | multiplier | Short-token-mass contribution to budget inflation |
| `phase1_budget_bonus_tokens` | `256` | tokens | Fixed budget headroom added to a chosen chunk |
| `phase1_budget_queue_bonus` | `64` | tokens/request | Queue-depth contribution to budget inflation |
| `phase1_explicit_budget_cap_tokens` | `512` | tokens | Maximum explicit-plan budget headroom |
| `phase1_cohort_queue_bonus` | `2` | queue units/request | Extra queue pressure for additional short requests |
| `phase1_cohort_mass_queue_factor` | `0.5` | multiplier | Short-token-mass contribution to adjusted queue pressure |
| `phase1_cohort_target_mass_factor` | `1.0` | multiplier | Short-token-mass contribution to the target chunk |

For exact ingress mode, the target can be below
`phase1_ingress_min_chunk`; this is intentional and covered by the Phase I
contract tests.

## Phase I runtime adaptation

| Field | Default | Unit | Meaning |
| --- | --- | --- | --- |
| `phase1_runtime_adaptive_enabled` | `False` | boolean | Enable queue-pressure interpolation of Phase I targets |
| `phase1_runtime_aggressive_long_fraction` | `0.33` | fraction | Long fraction at short-urgency pressure |
| `phase1_runtime_conservative_long_fraction` | `0.50` | fraction | Long fraction at sustained wall pressure |
| `phase1_runtime_aggressive_ingress_target_chunk` | `768` | tokens | Ingress target at the aggressive end |
| `phase1_runtime_conservative_ingress_target_chunk` | `1536` | tokens | Ingress target at the conservative end |
| `phase1_runtime_queue_high_watermark` | `8` | requests | Queue length treated as full wall pressure |
| `phase1_runtime_waiting_short_high_watermark` | `4` | requests | Waiting-short count treated as full urgency |
| `phase1_runtime_wait_us_high_watermark` | `1000000.0` | microseconds | Wait time treated as full urgency |
| `phase1_runtime_long_high_watermark` | `3072` | tokens | Long-prefill size treated as full wall pressure |
| `phase1_runtime_urgency_discount` | `0.55` | fraction | Short urgency subtracted from wall pressure |
| `phase1_runtime_ema_alpha` | `0.35` | fraction | Exponential moving-average update weight |

Runtime adaptation changes the active policy only for one scheduling decision;
the base policy is restored afterward.

## Phase II eligibility and coordination

| Field | Default | Unit or values | Meaning |
| --- | --- | --- | --- |
| `enable_phase2_scheduler` | `False` | boolean | Enable Phase II eligibility and priority processing |
| `phase2_min_prefill_count` | `1` | requests | Minimum active prefill count |
| `phase2_min_hetero_ratio` | `2.0` | ratio | Minimum prefill-length heterogeneity |
| `phase2_min_long_prefill` | `256` | tokens | Minimum high-cost prefill size |
| `phase2_enable_scheduler_cashout` | `False` | boolean | Allow one-call anchor deferral |
| `phase2_lora_rank_aware` | `True` | boolean | Include LoRA rank in the service-cost proxy |
| `phase2_min_lora_count` | `2` | requests | Minimum LoRA request count for rank heterogeneity |
| `phase2_min_rank_ratio` | `1.5` | ratio | Minimum high-to-low LoRA rank ratio |
| `phase2_min_rank_gap` | `4` | rank | Minimum absolute LoRA rank difference |
| `phase2_min_pressure_ratio` | `2.0` | ratio | Minimum high-to-low service-pressure ratio |
| `phase2_require_rank_hetero` | `False` | boolean | Require LoRA rank heterogeneity rather than accepting length heterogeneity |
| `phase12_joint_coordination` | `True` | boolean | Coordinate Phase II eligibility with Phase I state |
| `phase12_joint_min_chunk` | `512` | tokens | Minimum Phase I floor while joint coordination is active |
| `phase12_phase2_requires_recent_phase1` | `True` | boolean | Include recent Phase I evidence in the joint gate |
| `phase12_phase2_recent_ttl` | `4` | scheduler calls | Lifetime of recent Phase I evidence |
| `phase12_phase2_gate_mode` | `"soft"` | `hard`, `soft` | Strict recent-Phase-I gate or broader guarded gate |
| `phase12_phase2_soft_min_long_prefill` | `512` | tokens | Long-prefill threshold for the soft gate |
| `phase12_phase2_soft_allow_mixed_decode` | `True` | boolean | Allow a guarded mixed prefill/decode window |
| `phase12_phase2_beneficiary_prefill_scale` | `1.5` | multiplier | Recent Phase I chunk scale used when scoring beneficiaries |
| `phase12_phase2_beneficiary_score_threshold` | `0.55` | fraction | Minimum normalized beneficiary score |
| `phase12_phase2_beneficiary_quality_floor` | `0.60` | fraction | Minimum selected-beneficiary quality at the joint gate |
| `phase12_phase2_scheduler_cashout_soft_floor` | `0.55` | fraction | Lower cashout grade boundary |
| `phase12_phase2_scheduler_cashout_quality_floor` | `0.78` | fraction | High-confidence cashout grade boundary |
| `phase12_phase2_scheduler_cashout_cooldown_ticks` | `2` | scheduler calls | Base cooldown after a cashout |
| `phase12_phase2_priority_lane_ttl` | `2` | scheduler calls | Lifetime of the beneficiary priority lane |

Phase II requires both `enable_phase2_scheduler=True` and
`phase2_enable_scheduler_cashout=True` to defer an anchor. A cashout affects at
most one anchor and one native scheduler call.

## Phase II runtime adaptation

| Field | Default | Unit | Meaning |
| --- | --- | --- | --- |
| `phase2_runtime_adaptive_enabled` | `False` | boolean | Interpolate Phase II thresholds from runtime pressure |
| `phase2_runtime_low_pressure_min_hetero_ratio` | `6.0` | ratio | Length threshold at low pressure |
| `phase2_runtime_high_pressure_min_hetero_ratio` | `4.0` | ratio | Length threshold at high pressure |
| `phase2_runtime_low_pressure_min_pressure_ratio` | `6.0` | ratio | Service-pressure threshold at low pressure |
| `phase2_runtime_high_pressure_min_pressure_ratio` | `4.0` | ratio | Service-pressure threshold at high pressure |
| `phase2_runtime_low_pressure_min_long_prefill` | `1024` | tokens | Long-prefill threshold at low pressure |
| `phase2_runtime_high_pressure_min_long_prefill` | `768` | tokens | Long-prefill threshold at high pressure |

## Metrics

| Field | Default | Unit | Meaning |
| --- | --- | --- | --- |
| `enable_metrics_hook` | `True` | boolean | Observe engine requests and outputs in addition to scheduler decisions |
| `metrics_short_request_tokens` | `256` | tokens | Boundary used to classify short requests in metrics |

Metrics do not affect scheduling decisions except that already-recorded Phase I
cap-hit counts can be read by explicitly enabled runtime adaptation.

## Configuration invariants

- `queue_reorder_mode` must be `sjf`, `hrrn`, or `aging`.
- `scheduler_objective_mode` must be `fair_escape` or `pure_gain`.
- `phase12_phase2_gate_mode` must be `hard` or `soft`.
- token counts, queue counts, ranks, and TTL values should be non-negative.
- ratios and multipliers should be non-negative.
- quality, score, EMA, and discount fields are interpreted on a zero-to-one
  scale and are clamped by the implementation where required.
- WaveSlice cannot share an engine with another custom `scheduler_cls`.
- different active WaveSlice configurations require different processes.
