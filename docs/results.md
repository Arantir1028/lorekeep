# Result bundles and provenance

The repository tracks a small set of completed result exports for analysis and
paper auditing. Full run trees remain local because they contain logs, workload
copies, generated adapters, and machine-specific artifacts.

## Version rule

Tracked results are immutable records of the source revision and runtime used
to produce them. They are not automatically measurements of the current
`main` branch.

The current WaveSlice implementation uses scheduler-bound priority cashout,
called Phase II in configuration and metrics. It temporarily promotes one
lower-cost prefill and defers one higher-cost competitor for a single scheduler
call. Historical runs that used the retired execution-escape implementation
must not be attributed to the current method. A new performance claim for the
current implementation requires a fresh run with its source commit, resolved
configuration, and hardware metadata recorded.

## Tracked bundles

| Experiment | Scope | Primary path |
| --- | --- | --- |
| Request-ratio sweep | 5 models, 2 densities, 5 long-request fractions | `results/chapter5_exports/ratio_sweep_20step_5models_a100_overnight/` |
| DistServe comparison | continuous-batching replay and real two-replica CUCUMIS results | `results/openworkload_ratio_sweep_lora8/unified_distserve_continuous_cucumis_2a100_replay/` |
| Hardware portability | A100, RTX 4090, and RTX 5090; 50 cases per profile | `results/chapter5_exports/hardware_portability_a100_4090_5090/` |

The packaged scheduler inputs are stored separately under
`waveslice/data/lut_tables/` and are included in the Python package.

The repository-level bundle catalog is `results/result_bundles.json`. Verify
the file counts, byte counts, and aggregate SHA256 values with:

```bash
python tools/verify_result_bundles.py
```

The digest covers CSV, JSON, and TeX data products. README and PDF presentation
files are excluded so that editorial changes do not alter the data checksum.

## Request-ratio sweep

The frozen ratio sweep contains `mid` and `high` traffic densities with target
long-request fractions of 10%, 30%, 50%, 70%, and 90%. It covers Baichuan2-7B,
Gemma-7B, Gemma-2-9B, Mistral-7B-Instruct, and Qwen2.5-7B-Instruct.

Important files:

- [summary CSV](../results/chapter5_exports/ratio_sweep_20step_5models_a100_overnight/ratio_sweep_summary.csv);
- [guardrail CSV](../results/chapter5_exports/ratio_sweep_20step_5models_a100_overnight/ratio_sweep_guardrails.csv);
- [per-model comparison](../results/chapter5_exports/ratio_sweep_20step_5models_a100_overnight/ratio_sweep_per_model_comparison.csv);
- [method metrics](../results/chapter5_exports/ratio_sweep_20step_5models_a100_overnight/ratio_sweep_method_metrics.csv);
- [repeat metrics](../results/chapter5_exports/ratio_sweep_20step_5models_a100_overnight/ratio_sweep_repeat_metrics.csv);
- `ratio_sweep_ttft_improvement.pdf` and `ratio_sweep_guardrails.pdf`;
- [pipeline manifest](../results/chapter5_exports/ratio_sweep_20step_5models_a100_overnight/chapter5_pipeline_manifest.json).

The bundle contains 600 repeat rows, 300 method rows, and 50 per-model
comparison rows. Its raw main and baseline run roots are not tracked. The
exported tables and figures can be analyzed directly, but they cannot be fully
regenerated from a clean clone alone.

The historical result uses the stored label `strict_no_chunk`. The maintained
baseline configuration now calls the corresponding variant
`priority_no_chunk`; the frozen data label is not rewritten.

## DistServe comparison

The tracked DistServe bundle contains:

- physical single-GPU per-request stage measurements;
- logical two-GPU P/D replay with token-level continuous batching;
- real CUCUMIS two-replica results using round-robin and least-backlog dispatch;
- equal-resource comparison tables.

Important files:

- [unified request metrics](../results/openworkload_ratio_sweep_lora8/unified_distserve_continuous_cucumis_2a100_replay/unified_request_metrics.csv);
- [unified method metrics](../results/openworkload_ratio_sweep_lora8/unified_distserve_continuous_cucumis_2a100_replay/unified_method_metrics.csv);
- [equal-resource comparison](../results/openworkload_ratio_sweep_lora8/unified_distserve_continuous_cucumis_2a100_replay/unified_equal_resource_comparison.csv);
- `distserve_serial/raw/`, which contains the tracked per-case stage payloads;
- [manifest](../results/openworkload_ratio_sweep_lora8/unified_distserve_continuous_cucumis_2a100_replay/metadata/manifest.json).

The bundle contains 12,000 request rows, 150 method rows, 100 comparison rows,
and 50 raw DistServe case payloads. Unlike the ratio and hardware exports, it
supports request-level secondary analysis.

The comparison must be described as physical single-GPU stage-cost measurement
followed by a logical equal-resource two-GPU replay. It is not a physical
two-GPU DistServe deployment. The manifest retains absolute paths from the
original host; consumers should resolve data relative to the bundle directory.

The current simulator conserves measured decode service by assigning the
post-first-token cost as `(total_decode_service - first_token_service) / (N -
1)`. The tracked bundle does not record the corrected formula version in its
manifest, so it remains historical evidence rather than a current-code
performance result.

## Hardware portability

The hardware export combines 50 cases from each of three GPU profiles. It
compares logical two-GPU DistServe replay with real two-replica CUCUMIS using
least-backlog dispatch.

Important files:

- [aggregate summary](../results/chapter5_exports/hardware_portability_a100_4090_5090/hardware_portability_summary.csv);
- [case comparison](../results/chapter5_exports/hardware_portability_a100_4090_5090/hardware_portability_case_comparison.csv);
- [long-ratio trends](../results/chapter5_exports/hardware_portability_a100_4090_5090/hardware_portability_long_ratio_trend.csv);
- [exception audit](../results/chapter5_exports/hardware_portability_a100_4090_5090/hardware_portability_anomalies.csv);
- [source manifest](../results/chapter5_exports/hardware_portability_a100_4090_5090/hardware_portability_manifest.json).

The case table contains 150 rows. Ratios are defined as DistServe divided by
CUCUMIS. For latency and wall time, values above one favor CUCUMIS; for
throughput, values below one favor CUCUMIS.

The source run roots named by the manifest are not tracked. The compact export
supports aggregate and case-level analysis, but not reconstruction from the
original request-level hardware runs.

Hardware claims must use averages, win counts, and the exception table. The
data does not support a claim that CUCUMIS wins every workload; RTX 4090 in
particular is a memory-constrained stress profile with mixed TTFT results.

## Data available from a clean clone

A clean clone provides:

- packaged model profiles, gain tables, penalty tables, calibration records,
  and runtime-sanity records;
- the three compact result bundles listed above;
- request-level DistServe/CUCUMIS metrics for the tracked unified comparison;
- aggregate and case-level ratio-sweep and hardware-portability data.

A clean clone does not provide:

- model weights or Hugging Face cache contents;
- generated LoRA adapters;
- complete Chapter 5 main and baseline run trees;
- hardware source mirrors;
- local logs, quarantined runs, or temporary profiles.

Consequently, the tracked data is sufficient for table reconstruction,
plotting, aggregate statistics, and DistServe request-level analysis. Replaying
the complete experiment pipeline requires the original full-run data or a new
experiment.

## Repository data policy

Only compact, reviewable artifacts are committed to Git. The following remain
local by default:

- `results/*` outside the explicitly allowed export directories;
- `archived_results/`;
- generated `*.safetensors` adapters;
- model and dataset caches;
- profiler traces and run logs.

New formal result bundles should include a manifest, resolved configuration,
hardware metadata, tabular exports, and a short README that defines metric
directions and known limitations.
