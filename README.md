# WaveSlice

[简体中文](README.zh-CN.md)

WaveSlice is a scheduler extension for vLLM V1. It controls long-prefill
boundaries and scheduler priority for workloads with heterogeneous prompt
lengths and LoRA service costs. WaveSlice is installed as a Python package and
enabled through its `EngineArgs` subclass; it does not require changes to the
vLLM source tree.

This repository also contains the experiment harness and compact result bundles
used by the CUCUMIS research project.

## Requirements

- Python 3.10 or later
- Linux with a CUDA environment supported by vLLM
- vLLM with the V1 scheduler API
- A WaveSlice LUT for the selected model

The package does not pin a vLLM release. The integration uses vLLM V1 internal
interfaces, so compatibility must be checked when vLLM is upgraded. WaveSlice
does not support the legacy V0 engine.

## Installation

Install the WaveSlice package in an existing vLLM environment:

```bash
python -m pip install -e .
```

To install vLLM through the optional dependency:

```bash
python -m pip install -e ".[vllm]"
```

For code reading and CPU tests, use the lightweight development environment:

```bash
make install-dev
make check
```

The complete experiment environment uses the validated constraints referenced
by `requirements.txt`:

```bash
make install-runtime
```

See [Local development](docs/development.md) for environment and validation
levels. Runtime versions are recorded in `constraints/validated-vllm.txt`; the
WaveSlice runtime does not contain a hard-coded vLLM version check.

## Terminology

WaveSlice contains two scheduler mechanisms. The configuration and metrics use
the following internal phase names:

- **Prefill slicing (Phase I)** limits the amount of a selected long prefill
  admitted to one scheduler call, allowing waiting requests to be reconsidered
  sooner. It is enabled by default with WaveSlice.
- **Priority cashout (Phase II)** temporarily promotes one lower-cost prefill
  and defers one competing higher-cost prefill for a single scheduler call. It
  is optional and disabled by default.

## Quick start

Import `EngineArgs` from `waveslice`, not from `vllm.engine.arg_utils`. Import it
before constructing the vLLM engine so that the process is configured for V1.

```python
from waveslice import EngineArgs, get_wave_slice_metrics

from vllm.engine.llm_engine import LLMEngine
from vllm.sampling_params import SamplingParams

engine_args = EngineArgs(
    model="mistralai/Mistral-7B-v0.1",
    enable_wave_slice=True,
    enable_chunked_prefill=True,
    max_num_batched_tokens=1536,
    max_num_partial_prefills=1,
    max_long_partial_prefills=1,
    max_model_len=4096,
    enforce_eager=True,
)
engine = LLMEngine.from_engine_args(engine_args)

engine.add_request(
    request_id="req-001",
    prompt="Explain scheduler-level prefill chunking.",
    params=SamplingParams(max_tokens=64, temperature=0.0),
)

while engine.has_unfinished_requests():
    for output in engine.step():
        if output.finished:
            result = output.outputs[0]
            print(result.text)

print(get_wave_slice_metrics(reset=True))
```

WaveSlice derives the LUT name from `EngineArgs.model`; `lut_model` does not
need to be repeated in the normal case. A Hugging Face identifier such as
`mistralai/Mistral-7B-v0.1` maps to `mistralai--Mistral-7B-v0.1`, matching the
name used by the LUT builder. For a local model directory, WaveSlice uses the
directory name. Hugging Face cache snapshot paths are recognized from their
`models--org--model` component.

Set `WaveSliceConfig(lut_model=...)` only when a local directory name is
ambiguous or when deliberately selecting a different compatible profile.
`enable_chunked_prefill=True` is the intended configuration for prefill
slicing. Other vLLM arguments remain regular `EngineArgs` fields.

The selected model-specific raw, gain, and penalty tables must all exist.
WaveSlice reports a missing-table error instead of substituting a generic LUT.

### Disable WaveSlice

Use the same `EngineArgs` class and set the flag to `False`:

```python
from waveslice import EngineArgs

engine_args = EngineArgs(
    model="mistralai/Mistral-7B-v0.1",
    enable_wave_slice=False,
)
```

This selects the native vLLM V1 scheduler. The declarative `EngineArgs` path is
the application API; WaveSlice does not expose a separate inject or uninject
entry point.

### Enable priority cashout (Phase II)

The default policy enables Phase I and runtime metrics. Phase II priority
cashout is opt-in:

```python
from waveslice import EngineArgs, WaveSliceConfig, WaveSlicePolicy

engine_args = EngineArgs(
    model="mistralai/Mistral-7B-v0.1",
    enable_wave_slice=True,
    wave_slice_config=WaveSliceConfig(
        policy=WaveSlicePolicy(
            enable_phase1_scheduler=True,
            enable_phase2_scheduler=True,
            phase2_enable_scheduler_cashout=True,
        ),
    ),
    enable_chunked_prefill=True,
)
```

Phase II defers at most one competing prefill for one scheduler call and then
restores it. The default joint policy coordinates Phase II with recent Phase I
activity and applies a cooldown between cashouts.

## Configuration

`WaveSliceConfig` has three public fields:

| Field | Default | Description |
| --- | --- | --- |
| `lut_model` | inferred from `EngineArgs.model` | Optional override for LUT selection |
| `gamma` | `2.0` | Queue-pressure penalty in the LUT objective |
| `policy` | `WaveSlicePolicy()` | Prefill slicing, priority cashout, queue ordering, and metrics settings |

Model-specific LUTs are stored under `waveslice/data/lut_tables/`. Formal
experiments should use a model-specific profile generated on compatible
hardware. The Chapter 5 preflight checks the recorded hardware fingerprint and
rebuilds stale or missing profiles before a run.

The maintained LUT builder applies the same model-name rule:

```bash
python experiments/build_hybrid_checkpoint_runtime_luts.py \
  --models mistralai/Mistral-7B-v0.1
```

The selected checkpoint must already be present in the configured Hugging Face
cache. The command performs GPU profiling and runtime calibration.

For this model, the generated triplet is named
`raw_profile_mistralai--Mistral-7B-v0.1.json`,
`lut_gain_mistralai--Mistral-7B-v0.1.json`, and
`lut_penalty_mistralai--Mistral-7B-v0.1.json`. Runtime selection derives the
same stem from `EngineArgs.model`.

WaveSlice cannot be enabled together with another custom `scheduler_cls`. Its
runtime state is process-wide; run engines that require different WaveSlice
configurations in separate processes.

## Runtime metrics

`get_wave_slice_metrics()` returns request and scheduler statistics collected
in the parent process and vLLM worker processes. The report includes:

- request counts, TTFT, and completion slowdown;
- Phase I eligibility, selected chunks, virtual caps, and runtime adaptation;
- Phase II attempts, decisions, reasons, cooldowns, and priority-lane activity.

Pass `reset=True` to return the current report and clear the counters. Use
`reset_wave_slice_metrics()` when a report is not needed.

## Method

WaveSlice has two scheduler-bound phases:

- **Phase I** chooses a bounded chunk for a long prefill so that vLLM returns to
  the scheduler earlier and can reconsider waiting requests.
- **Phase II** selects one low-service-cost beneficiary and one competing
  high-service-cost anchor, promotes the beneficiary, and defers the anchor for
  one scheduler call.

Neither phase changes model execution, delays computed outputs, unbinds
requests, or creates an additional CUDA stream. See [Method and integration](docs/method.md)
for the decision rules and vLLM integration boundary.

The internal call path and module responsibilities are documented in
[Architecture](docs/architecture.md). All policy fields are listed in the
[Configuration reference](docs/configuration.md).

## Experiments and results

The repository includes configuration-driven runners for Chapter 2
observations, Chapter 5 open-workload evaluation, request-ratio sweeps,
DistServe comparison, and A100/RTX 4090/RTX 5090 portability runs.

- [Experiment guide](docs/experiments.md)
- [Result bundles and provenance](docs/results.md)

Generated run trees, logs, model snapshots, and synthetic adapters are ignored
by Git. The repository tracks packaged LUTs and selected compact result bundles.

## Repository layout

| Path | Purpose |
| --- | --- |
| `waveslice/` | Installable package and vLLM V1 integration |
| `waveslice/scheduling/` | LUT-backed chunk and fairness calculations |
| `waveslice/vllm/` | Scheduler, request, process, and metrics adapters |
| `waveslice/data/lut_tables/` | Packaged profiles and LUTs |
| `experiments/` | Workload generation, preflight, and experiment runners |
| `scripts/` | Ratio, DistServe, multi-GPU, and portability orchestration |
| `tests/` | Unit, integration-contract, and evaluator code |
| `results/` | Local run trees and tracked compact exports |
| `constraints/` | Tested runtime dependency combinations |

## Development

Install the lightweight development environment and run all non-GPU checks:

```bash
make install-dev
make check
```

Individual targets are available when iterating on one area:

```bash
make test
make lint
make validate-configs
make check-docs
make verify-results
```

The test suite does not replace a real-GPU validation when scheduler behavior
or vLLM compatibility changes.
