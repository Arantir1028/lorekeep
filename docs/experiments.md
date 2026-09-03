# Experiment guide

The experiment code is configuration-driven and is intended to be run from the
repository root. Paths stored in maintained configuration files are relative to
the repository; machine-specific model and cache locations are resolved during
preflight.

## Environment setup

Activate a CUDA environment that can run vLLM, then install the experiment
dependencies and the local package:

```bash
python -m pip install -r requirements.txt
python -m pip install -e ".[dev]"
```

Model and dataset selection is controlled by the resource catalog and suite
configuration. With `resource_selection.auto_download=true`, missing Hugging
Face resources may be downloaded. Set `resource_selection.offline=true` only
when all selected resources are already available locally.

Model identity is maintained in `experiments/catalog.py`. Resource
configurations normally select a model by `key`; its Hugging Face identifier,
maximum-length override, and derived LUT name come from the shared catalog.
An inline `model_id` is needed only for an uncatalogued model. Set `lut_name`
only when intentionally overriding the normal model-derived name.

Maintained files under `experiments/configs/` carry a `schema_version` and are
validated against `experiments/schemas/experiment-config.schema.json`:

```bash
make validate-configs
```

For a model-free check of configuration resolution and output layout, use the
small development configuration:

```bash
python experiments/run_openworkload_suite.py \
  --config experiments/configs/openworkload_smoke.json \
  --run-name local_smoke \
  --dry-run
```

This command does not download resources or produce a performance result.

## Chapter 5 pipeline

The maintained Chapter 5 entry point is:

```bash
python experiments/run_chapter5_suite.py \
  --config experiments/configs/chapter5_pipeline_default.json \
  --run-tag chapter5_demo
```

The command runs three stages in order:

1. `preflight`: inspect the environment, validate model execution, validate or
   rebuild LUTs, and produce a resolved configuration;
2. `main`: build dataset-backed workloads and run the WaveSlice modes;
3. `baseline`: reuse the main-run workloads for native and mechanism-baseline
   cases.

`run_chapter5_suite.py` accepts only `preflight`, `main`, and `baseline` as stage
names. Figures and aggregate exports are produced by their dedicated scripts,
not by a `figures` pipeline stage.

The default pipeline references:

- `experiments/configs/openworkload_v1_local_realworld_lora8.json`;
- `experiments/configs/chapter5_baseline_variants_lora7.json`.

### Preflight only

```bash
python experiments/run_chapter5_suite.py \
  --config experiments/configs/chapter5_pipeline_default.json \
  --stages preflight \
  --run-tag chapter5_demo
```

To validate configuration and paths without constructing an engine:

```bash
python experiments/run_chapter5_suite.py \
  --config experiments/configs/chapter5_pipeline_default.json \
  --stages preflight \
  --run-tag chapter5_demo \
  --skip-preflight-engine-smoke
```

Preflight writes the following files below
`results/openworkload_v1_local_realworld_lora8/<run_tag>_preflight/metadata/`:

- `resolved_environment.json`;
- `model_preflight.json`;
- `runtime_capacity.json`;
- `workload_capacity.json`;
- `resolved_config.json`;
- `preflight_summary.json`.

On smaller GPUs, preflight may reduce batch-token capacity, output length,
request counts, repeats, sample count, and arrival pressure. The applied values
are recorded in `resolved_config.json` and `workload_capacity.json`.

### Reuse a completed preflight

```bash
python experiments/run_chapter5_suite.py \
  --config experiments/configs/chapter5_pipeline_default.json \
  --stages main,baseline \
  --run-tag chapter5_demo \
  --preflight-run-root \
    results/openworkload_v1_local_realworld_lora8/chapter5_demo_preflight
```

### Run a focused trial

Use command-line filters for temporary trials; keep the long-lived experiment
definition in JSON configuration:

```bash
python experiments/run_chapter5_suite.py \
  --config experiments/configs/chapter5_pipeline_default.json \
  --run-tag chapter5_trial \
  --model-keys mistral-7b-v0.1 \
  --dataset-keys ultrachat200k \
  --densities mid
```

Available filters are `--model-keys`, `--dataset-keys`, `--densities`, and
`--variants`. Each takes a comma-separated list.

### Run baselines from an existing main run

```bash
python experiments/run_chapter5_suite.py \
  --config experiments/configs/chapter5_pipeline_default.json \
  --stages baseline \
  --main-run-root \
    results/openworkload_v1_local_realworld_lora8/chapter5_demo_main \
  --run-tag chapter5_demo \
  --variants priority_no_chunk \
  --model-keys mistral-7b-v0.1 \
  --densities mid
```

The maintained baseline variants are defined in
`experiments/configs/chapter5_baseline_variants_lora7.json`.

### Output layout

| Stage | Output path |
| --- | --- |
| preflight | `results/openworkload_v1_local_realworld_lora8/<run_tag>_preflight/` |
| main | `results/openworkload_v1_local_realworld_lora8/<run_tag>_main/` |
| baseline | `results/chapter5_baseline_variants/<run_tag>_baseline/` |
| pipeline manifest | `results/chapter5_exports/<run_tag>/chapter5_pipeline_manifest.json` |

Run directories contain case-level JSON, request and method metrics, progress
metadata, logs, and summaries. Completed rows are reused on restart where the
runner supports resume.

## Request-ratio sweep

The maintained ratio-sweep pipeline uses five long-request fractions at two
traffic densities:

```bash
python experiments/run_chapter5_suite.py \
  --config experiments/configs/chapter5_pipeline_ratio_sweep.json \
  --run-tag ratio_sweep
```

To rebuild ratio tables and figures from complete main and baseline run trees:

```bash
python scripts/regenerate_ratio_sweep.py \
  --main-run results/openworkload_ratio_sweep_lora8/<main_run> \
  --baseline-run results/chapter5_baseline_variants/<baseline_run> \
  --out-dir results/chapter5_exports/<export_name>
```

The compact Git result bundle contains exported CSV, JSON, and PDF files but not
the complete main and baseline run trees. The regeneration command therefore
requires local full-run data or a newly completed sweep.

## Chapter 2 observations

Chapter 2 observations run native vLLM without WaveSlice hooks:

```bash
python experiments/run_chapter2_observations.py \
  --config experiments/configs/chapter2_observations_v2.json \
  --observations all \
  --run-name chapter2_demo
```

Use `--observations obs1` for the one-versus-two-long-request observation and
`--observations obs2` for the fixed token-budget sweep. Outputs are written to
`results/chapter2_observations_v2/<run_name>/`.

## DistServe and two-replica comparison

The unified runner has two components:

- physical single-GPU measurement of per-request prefill/decode stage costs,
  followed by logical token-level continuous-batching DistServe replay;
- real two-replica CUCUMIS runs using round-robin or least-backlog dispatch.

Inspect all machine- and run-specific options before launching:

```bash
python scripts/run_unified_distserve_cucumis_experiment.py --help
```

The default source is a completed ratio-sweep main run. A typical invocation
must identify the source run, output directory, models, densities, and the two
CUDA devices used by the CUCUMIS replicas. The comparison is not a physical
two-GPU DistServe deployment; it is a logical two-GPU replay based on physical
single-GPU stage measurements.

## Hardware portability

The hardware runner supports `a100`, `rtx4090`, and `rtx5090` profiles. A full
run requires two visible GPUs:

```bash
python scripts/run_hardware_portability_experiment.py \
  --gpu-profile a100 \
  --cuda-visible-devices 0,1 \
  --dry-run
```

Remove `--dry-run` after verifying paths, selected models, and the output root.
The runner rebuilds LUTs by default, performs DistServe stage measurement and
replay, runs the two CUCUMIS replicas, and writes profile metadata. Use
`--merge-only`, `--skip-distserve`, or `--skip-cucumis` only when resuming a
known partial run.

Merged hardware summaries are generated with:

```bash
python scripts/summarize_hardware_portability_results.py \
  --export-root results/chapter5_exports/hardware_portability_a100_4090_5090
```

The summarizer requires the full copied-back source roots recorded by the
hardware result manifest. Those roots are not part of the compact Git bundle.

To stage the repository and ratio-sweep source data on a remote host:

```bash
scripts/sync_hardware_portability_server.sh a100 user@example-host
```

The script derives the local repository path from its own location and uses a
directory relative to the remote user's home by default. Override
`WAVESLICE_SYNC_LOCAL_ROOT`, `WAVESLICE_SYNC_REMOTE_ROOT`,
`WAVESLICE_SYNC_SOURCE_RUN`, or `WAVESLICE_SYNC_HOST_LABEL` when the remote
layout differs.

## Operational rules

- Run commands from the repository root.
- Keep maintained configuration paths repository-relative.
- Use a distinct `run-tag` or output root for each formal run.
- Do not overwrite frozen result bundles with diagnostic output.
- Keep model snapshots, generated adapters, logs, and full run trees outside
  Git.
- Record the source commit, resolved configuration, hardware metadata, and
  completion state before using a run in a paper.
- Use the GPU experiment lock for concurrent local jobs; the evaluator enables
  it by default where configured.

The tracked result bundles and their provenance are listed in
[results.md](results.md).
