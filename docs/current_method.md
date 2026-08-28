# Current Method Definition

The maintained CUCUMIS method is Phase I prefill-boundary control plus Phase II
scheduler-bound priority cashout. It does not modify model execution, defer
already-computed outputs, unbind requests from vLLM, or create a second CUDA
stream.

## Phase I: earlier scheduler boundaries

Phase I observes the live request cohort and chooses a bounded chunk for a long
prefill. The V1 adapter exposes that cap at the native scheduler boundary, so
vLLM returns to scheduling sooner and can reconsider waiting short requests.
The runtime policy may adjust the chunk from queue length, waiting-short count,
wait time, long remaining tokens, and virtual-cap hit rate.

## Phase II: one-tick priority cashout

Phase II operates only on scheduler queues:

1. It identifies heterogeneous prefill work and scores short beneficiaries.
2. It selects one beneficiary and one competing high-service-cost anchor.
3. It hides the anchor for one native scheduler call and stably promotes the
   beneficiary in the waiting queue.
4. It restores only the deferred anchor after the call, preserving every queue
   mutation made by the native scheduler.

The service-cost proxy is remaining prefill tokens multiplied by LoRA rank. A
cashout is bounded to one request and one scheduler tick; cooldown prevents it
from becoming permanent starvation.

For joint Phase I+II mode, the gate is deliberately small: workload
heterogeneity must be sufficient, a beneficiary must exist, and at least one of
recent Phase I activity, a strong long-prefill window, or an allowed mixed-decode
window must hold. The former weighted window-score and sparse-exception tree is
not part of the maintained method.

## Runtime metrics

The plugin records scheduler decisions, Phase I chunk/cap activity, Phase II
reasons, priority-lane activation, TTFT, completion slowdown, and request timing.
Parent and child processes exchange additive metric events through a temporary
JSONL file.

## Result compatibility

Results produced by the historical execution-escape implementation remain
archived evidence for that version only. The current scheduler-bound method has
a real-GPU functional validation, but paper-facing Phase I/II performance values
must come from a fresh full sweep before they are attributed to this version.
