# Current Method Definition

The current CUCUMIS path is fixed as Phase I chunk-size boundary control plus
Phase II execution escape / priority promotion. It should not be described as
true unbind, dual stream, or execution-level long/short co-running.

## Phase I

Phase I controls the chunk size of long prefill work so that a long request
returns to the scheduler boundary earlier. The goal is not to make long and
short requests run in separate streams. The goal is to create an earlier
dispatch opportunity where waiting short or beneficiary requests can be
considered.

## Phase II

Phase II reshapes the next scheduled window after Phase I has returned control.
The active implementation uses execution escape and scheduler-side priority
promotion: selected beneficiary requests are kept active while lower-value
work can be deferred within bounded gates. This is the current paper path.

The abandoned true-unbind / dual-stream path is not part of the current method
definition and should not be used in the paper narrative or current
experiment descriptions.

## Runtime Pressure Adaptation

The current adaptive policy is runtime queue-pressure adaptation. The
open-workload density still controls how requests arrive, but it no longer
directly chooses the active chunk size. Instead, each scheduler step computes a
pressure signal from live state:

```text
queue length
waiting short-prefill count
max observed waiting time
current long remaining tokens
virtual-cap hit EMA
```

The policy separates two forces. Short urgency pushes Phase I toward smaller
chunks so short requests can reach the next scheduling boundary earlier.
Sustained wall pressure pushes Phase I toward larger chunks and stricter Phase
II gates so long-request wall time is protected. The effective pressure is an
EMA-smoothed weighted wall-pressure score discounted by short urgency.

The suite still records a workload-pressure score for reproducibility, but that
score is metadata only when `runtime_queue_pressure=true`.
