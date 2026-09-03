# Architecture

WaveSlice extends the vLLM V1 scheduler without maintaining a vLLM fork. The
package owns configuration, scheduling decisions, temporary scheduler-bound
state, and metrics. vLLM continues to own request lifecycle, model execution,
KV cache management, and output delivery.

## Runtime path

```text
waveslice.EngineArgs
    -> WaveSliceEngineArgs.create_engine_config()
    -> WaveSliceScheduler selected in vLLMConfig
    -> activate_wave_slice()
    -> RuntimeState and method adapters installed
    -> WaveSliceScheduler.schedule()
       -> runtime coordinator
       -> Phase I plan and temporary caps
       -> Phase II priority cashout
       -> native vLLM V1 schedule()
       -> temporary state restored and metrics recorded
```

The vLLM plugin entry point activates equivalent configuration in worker
processes. Worker metrics are appended to a process-shared JSONL stream and
merged into the parent report.

## Package map

| Module | Responsibility |
| --- | --- |
| `waveslice/engine_args.py` | Declarative vLLM entry point and scheduler selection |
| `waveslice/config.py` | Public WaveSlice configuration and LUT-model resolution |
| `waveslice/policy.py` | Immutable scheduler policy values |
| `waveslice/lut/` | Exact model-name resolution and LUT loading |
| `waveslice/scheduling/` | LUT-backed objective, chunk selection, and fairness calculations |
| `waveslice/vllm/bootstrap.py` | V1 process configuration before vLLM imports |
| `waveslice/vllm/scheduler.py` | WaveSlice subclass of the native V1 scheduler |
| `waveslice/vllm/integration.py` | Activation lifecycle, adapter installation, and global runtime ownership |
| `waveslice/vllm/state.py` | Runtime and scheduling decision records |
| `waveslice/vllm/runtime.py` | One scheduler-call coordinator |
| `waveslice/vllm/phase1_cohorts.py` | Live cohort discovery and per-LoRA candidate selection |
| `waveslice/vllm/phase1_math.py` | Pure Phase I thresholds, budgets, and adaptation calculations |
| `waveslice/vllm/phase1_planning.py` | Phase I policy adaptation and final schedule plan |
| `waveslice/vllm/phase1_runtime.py` | Temporary Phase I scheduler limits and output reconciliation |
| `waveslice/vllm/phase1_selection.py` | Request lookup and cohort construction helpers |
| `waveslice/vllm/phase1_state.py` | Ingress virtual slices and explicit plan state |
| `waveslice/vllm/phase2_beneficiaries.py` | Beneficiary signal construction |
| `waveslice/vllm/phase2_gates.py` | Phase I/II coordination and eligibility gates |
| `waveslice/vllm/phase2_cashout.py` | Cashout value and cooldown calculations |
| `waveslice/vllm/phase2_priority.py` | Priority-lane queue transformations |
| `waveslice/vllm/phase2_runtime.py` | One-call anchor deferral, restoration, and Phase II metrics |
| `waveslice/vllm/request_hooks.py` | Temporary request token-count visibility |
| `waveslice/vllm/engine_hooks.py` | Request and output observation for metrics |
| `waveslice/vllm/subprocess.py` | Worker configuration and cross-process metric exchange |
| `waveslice/metrics.py` | Thread-safe request and scheduler metrics |

Imports within `waveslice` form an acyclic graph. Lower-level math and state
modules must not import the integration lifecycle or public `EngineArgs`.

## State ownership

`integration.py` owns the active process-wide `RuntimeState`. It retains the
native methods and properties needed for restoration. A scheduler call may add
temporary Phase I caps or temporarily remove one Phase II anchor, but the call
coordinator must restore those changes before returning to vLLM.

The process supports one active WaveSlice configuration. Engines that require
different WaveSlice policies must run in separate processes.

## Semantic invariants

Changes to the runtime must preserve the following rules unless the method is
being deliberately redesigned and re-evaluated:

- vLLM V1 is the only supported engine path;
- disabling WaveSlice selects the native V1 scheduler;
- model kernels, KV-cache contents, generated tokens, and delivered outputs are
  not modified;
- a Phase I chunk limits only the current scheduler window and does not truncate
  a request;
- a Phase II cashout defers at most one anchor for one native scheduler call;
- hidden or capped scheduler state is restored even when the native call fails;
- LoRA-aware decisions do not derive a chunk from an unrelated adapter cohort;
- LUT selection requires an exact model-specific raw/gain/penalty triplet;
- metrics observe decisions but do not control the scheduler;
- frozen results remain associated with the implementation and environment that
  produced them.

## Where to make a change

| Change | Primary location | Required nearby checks |
| --- | --- | --- |
| Public enable/disable behavior | `engine_args.py`, `config.py` | `tests/test_engine_args.py` |
| LUT naming or lookup | `waveslice/lut/` | project-contract and EngineArgs tests |
| Phase I eligibility or chunk math | `phase1_math.py` | Phase I and scenario-fixture tests |
| Phase I cohort selection | `phase1_cohorts.py` | Phase I and LoRA cohort tests |
| Applying/restoring Phase I limits | `phase1_runtime.py` | runtime hook tests and GPU smoke |
| Phase II eligibility/value | Phase II beneficiary, gate, and cashout modules | Phase II tests |
| Queue deferral/restoration | `phase2_runtime.py`, `runtime.py` | Phase II tests and GPU smoke |
| Metrics schema | `metrics.py`, observation hooks | runtime-metrics and result-parser tests |
| Chapter 5 workflow | `experiments/run_chapter5_suite.py` and stage modules | config validation and dry run |
| Paper-facing data | result exporters and bundle manifest | checksum and provenance validation |

## Experiment layers

`experiments/` contains reusable workload, preflight, and evaluation stages.
`scripts/` composes those stages for ratio sweeps, DistServe replay, multi-GPU
dispatch, and hardware portability. Results are data products, not import-time
inputs to the WaveSlice runtime. Runtime code must not depend on `experiments/`,
`scripts/`, or `results/`.
