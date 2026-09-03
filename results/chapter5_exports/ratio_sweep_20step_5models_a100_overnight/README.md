# Request-ratio sweep result bundle

This directory contains the frozen export for
`ratio_sweep_20step_5models_a100_overnight`.

## Scope

- 5 models
- `mid` and `high` traffic densities
- 10%, 30%, 50%, 70%, and 90% target long-request fractions
- 50 completed main cases
- 100 completed baseline-variant cases

## Files

| File | Contents |
| --- | --- |
| `ratio_sweep_summary.csv` | aggregate sensitivity table with confidence intervals |
| `ratio_sweep_guardrails.csv` | completion, wall-time, and throughput guardrails |
| `ratio_sweep_per_model_comparison.csv` | model-level comparison rows |
| `ratio_sweep_method_metrics.csv` | aggregated method metrics |
| `ratio_sweep_repeat_metrics.csv` | repeat-level metrics |
| `ratio_sweep_ttft_improvement.pdf` | TTFT sensitivity figure |
| `ratio_sweep_guardrails.pdf` | guardrail figure |
| `chapter5_pipeline_manifest.json` | source configuration and original run roots |

The JSON files mirror the corresponding CSV tables.

## Provenance

This is a historical result bundle. The original main and baseline run trees
are not tracked, and the measurements must not be presented as results from the
current WaveSlice source revision without a new full sweep.

The stored baseline label `strict_no_chunk` belongs to this frozen result. The
maintained configuration uses the name `priority_no_chunk`; existing rows are
left unchanged for reproducibility.

See [the repository result notes](../../../docs/results.md) for provenance and
data-availability rules.
