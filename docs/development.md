# Local development

This guide is for reading, testing, and changing WaveSlice locally. It does not
define a formal contribution or release process.

## Choose an environment

Most scheduling, configuration, and result-processing work does not require a
GPU or an installed vLLM runtime. Use a lightweight virtual environment for
that work:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
make install-dev
make check
```

If the system Python does not provide `venv`, install the distribution's
`python3-venv` package or create an equivalent Conda environment:

```bash
conda create -n waveslice-dev python=3.10
conda activate waveslice-dev
make install-dev
```

`make check` runs Ruff, the CPU unit and contract tests, experiment-config
validation, Markdown-link validation, and tracked-result checksum validation.
It does not construct a vLLM engine or start an experiment.

Use the full runtime environment for engine integration, LUT profiling, or
experiments:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
make install-runtime
```

`requirements.txt` uses `constraints/validated-vllm.txt`. The constraints file
records the stack on which the current V1 integration has been exercised; it is
not a version check in WaveSlice itself. Updating that file requires both the
CPU checks and a real-GPU engine smoke test.

## Common commands

```bash
make test              # CPU unit and contract tests
make lint              # Ruff
make validate-configs  # validate maintained JSON experiment files
make check-docs        # validate local Markdown links
make verify-results    # validate compact result bundles
make check             # all non-GPU checks
```

All commands are intended to run from the `lorekeep` repository root. Set
`PYTHON=/path/to/python` when the desired interpreter is not named `python3`.

## Validation levels

### CPU contracts

Run `make check` for documentation, configuration, LUT selection, scheduling
math, result processing, and refactors that do not change vLLM-facing behavior.
The tests replace vLLM with a small compatibility stub where necessary.

### Engine smoke test

Changes to `EngineArgs`, scheduler hooks, request token visibility, process
configuration, or runtime metrics require a real vLLM V1 engine smoke test in
addition to `make check`. Use one model, one short request, and one long request;
verify both `enable_wave_slice=False` and `enable_wave_slice=True`.

An engine smoke test establishes integration compatibility. It is not evidence
of a performance improvement.

### Workload regression

Changes to policy defaults or scheduler decisions require a fixed-workload
comparison. Record the source commit, resolved configuration, model and LUT
identity, hardware information, per-request timings, and WaveSlice metrics.
Do not replace a frozen result bundle with diagnostic output.

### Formal experiment

Paper-facing results use the pipeline and provenance rules in
`docs/experiments.md` and `docs/results.md`. A formal run must retain its
manifest, resolved configuration, completion state, and hardware metadata.

## Reading order

For the user-facing path, read:

1. `waveslice/engine_args.py`;
2. `waveslice/config.py` and `waveslice/policy.py`;
3. `waveslice/vllm/integration.py`;
4. `waveslice/vllm/runtime.py`;
5. the Phase I or Phase II modules relevant to the change.

For experiments, start at `experiments/run_chapter5_suite.py`, then follow the
stage command into preflight, open-workload execution, or baseline execution.
The files under `scripts/` orchestrate multi-run analysis and multi-GPU work;
they are not part of the importable WaveSlice package.

See `docs/architecture.md` for module responsibilities and
`docs/configuration.md` for policy fields and invariants.

## Small deterministic scenarios

`tests/fixtures/scheduler_scenarios.json` contains model-free Phase I and Phase
II cases. `experiments/configs/openworkload_smoke.json` is a one-model,
one-dataset, one-density configuration for checking experiment resolution:

```bash
python experiments/run_openworkload_suite.py \
  --config experiments/configs/openworkload_smoke.json \
  --run-name local_smoke \
  --dry-run
```

The dry run writes only metadata and a planned row under the ignored
`results/development_smoke/` tree. It does not download a model or dataset and
does not produce a performance result.
