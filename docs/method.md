# Method and vLLM integration

WaveSlice is a scheduler-side extension for vLLM V1. It changes which prefill
work is admitted to a scheduler step and how much of a long prefill is visible
in that step. It does not change model kernels, KV-cache contents, generated
tokens, or output delivery.

The implementation uses two internal phase names. **Phase I** means prefill
slicing: bounding a long prefill at a scheduler boundary. **Phase II** means
priority cashout: temporarily promoting one lower-cost prefill while deferring
one higher-cost competitor for a single scheduler call.

## Integration boundary

Applications use `waveslice.EngineArgs`, a subclass of vLLM `EngineArgs` with
two additional fields:

- `enable_wave_slice`: selects WaveSlice or the native V1 scheduler path;
- `wave_slice_config`: contains an optional LUT override, the objective
  coefficient, and `WaveSlicePolicy`.

Unless explicitly overridden, the LUT model is derived from `EngineArgs.model`
using the same naming convention as the LUT builder. Hugging Face identifiers
replace `/` with `--`; local model references use the directory name, and
Hugging Face cache paths use their `models--org--model` component.
Selection uses exact generated names and registered aliases. It does not use
substring matching or fall back to a generic LUT when a model-specific triplet
is missing.

When WaveSlice is enabled, `create_engine_config()` performs the following
operations:

1. validates that no unrelated custom scheduler is configured;
2. selects `WaveSliceScheduler`, a subclass of the native V1 scheduler;
3. records the WaveSlice configuration under vLLM `additional_config`;
4. installs the scheduler and metrics adapters in the parent process;
5. publishes the configuration for worker processes.

The package registers a `vllm.general_plugins` entry point. A vLLM worker loads
that entry point and activates the same configuration from its environment.
Additive worker metrics are written to a temporary JSONL file and merged into
the parent-process report.

The adapter replaces a limited set of methods and properties at runtime:

- V1 scheduler `schedule`, `add_request`, and `_update_after_schedule`;
- V1 request token-count properties;
- engine and processor request-observation methods when metrics are enabled.

The original objects are retained and restored when WaveSlice is deactivated.
The scheduling methods are patched on `WaveSliceScheduler`, not on the native
scheduler class. Request and engine observation hooks are process-wide while
WaveSlice is active, which is why one process supports one active WaveSlice
configuration.

This design avoids a maintained vLLM fork, but it still depends on V1 internal
interfaces. A vLLM upgrade therefore requires an integration test and a GPU
smoke test.

## Phase I: prefill-boundary control

Phase I shortens the current scheduling window for a selected long prefill. It
does not truncate the request; the remaining prompt tokens are considered by a
later scheduler step.

### Eligibility and cohort selection

At each scheduler call, WaveSlice observes the waiting and running prefill
requests. A cohort is eligible when it contains at least two positive lengths
and the long/short length ratio or the configured extreme-ratio condition is
met.

The representative cohort contains:

- a selected long prefill;
- representative short length and short-token mass;
- current queue length and maximum wait time;
- LoRA cohort information when LoRA serving is active.

For LoRA workloads, requests can be grouped by adapter path or inferred rank so
that a chunk decision is not based on an unrelated adapter cohort.

### LUT objective

The scheduler loads three tables for the selected model:

- `raw_profile_<model>.json`: standalone service time by token bucket;
- `lut_gain_<model>.json`: expected short-request benefit by chunk;
- `lut_penalty_<model>.json`: expected long-request penalty by chunk.

Request sizes are mapped conservatively to the configured token buckets. For
each candidate chunk, the decision engine evaluates an objective of the form:

```text
score = wait_weight * short_utility
        - long_penalty * (1 + gamma * queue_pressure)
```

`wait_weight` increases with normalized request waiting time. Queue pressure is
derived from queue depth and maintained as an exponential moving average. If
no candidate improves the objective, WaveSlice keeps the native baseline
chunk.

### Applying a chunk

An accepted decision is applied only for the current scheduler call. Depending
on the active policy and vLLM request state, the adapter may:

- expose a virtual token cap on the selected request;
- adjust the long-prefill threshold;
- constrain the scheduler token budget;
- clamp the scheduler output before vLLM advances request state.

Temporary caps and scheduler configuration values are restored after the
native scheduler returns. The scheduler can also reorder queues using SJF,
HRRN, or aging keys; SJF is the default.

### Runtime adaptation

Runtime adaptation is optional. When enabled, it combines queue length,
waiting-short count, maximum wait, long-prefill size, and virtual-cap hit rate
into two signals:

- short urgency, which favors a more aggressive boundary;
- sustained wall pressure, which favors a more conservative chunk.

The signals interpolate between configured chunk targets and can also adjust
the Phase II heterogeneity thresholds. Workload density names are not used as
runtime decisions.

## Phase II: scheduler priority cashout

Phase II is disabled by default. It is enabled by setting both
`enable_phase2_scheduler` and `phase2_enable_scheduler_cashout` in
`WaveSlicePolicy`.

Phase II operates before model execution:

1. collect active prefill lengths and LoRA ranks;
2. require sufficient length or service-cost heterogeneity;
3. score candidate beneficiaries from size and waiting time;
4. choose one beneficiary and one competing high-cost anchor;
5. move the beneficiary to the front of the waiting queue;
6. hide the anchor for one native scheduler call;
7. restore only the hidden anchor after that call.

The service-cost proxy is:

```text
remaining_prefill_tokens * LoRA_rank
```

The value gate compares beneficiary quality and coverage with the cost of
deferring the anchor. A cashout affects at most one anchor and one scheduler
call. A cooldown and a short-lived priority-lane state prevent the mechanism
from becoming permanent starvation.

When Phase I and Phase II are enabled together, the default soft gate requires
an eligible beneficiary and sufficient heterogeneity, plus one of the
following:

- a live or recent Phase I cap;
- a strong long-prefill window;
- an allowed mixed prefill/decode window.

## Metrics

The runtime report contains:

- request count and completion state;
- all/short/long TTFT p50, p95, and p99;
- all/short/long completion slowdown p50, p95, and p99;
- Phase I attempts, selected chunks, cap hits, rewrites, and adaptation data;
- Phase II attempts, apply ratio, rejection reasons, and priority-lane data.

Metrics are intended for diagnosis and experiment collection. They do not
replace the request-level timing records written by the experiment evaluator.

## Operational constraints

- WaveSlice supports vLLM V1 only.
- One WaveSlice configuration is active per process.
- WaveSlice cannot share an engine with another custom `scheduler_cls`.
- All three model-specific LUT files must exist before activation.
- Model-specific LUTs should be regenerated when the hardware or relevant
  runtime environment changes.
- Phase II is opt-in; `enable_wave_slice=True` alone enables Phase I and
  metrics, not the full Phase I+II method.

## Result compatibility

The tracked result bundles were produced by frozen historical experiment runs.
Some predate the current scheduler-bound Phase II implementation. They remain
valid records of those runs, but they must not be presented as measurements of
the current source revision. A performance claim for the current implementation
requires a new full sweep with its manifest and commit recorded.
