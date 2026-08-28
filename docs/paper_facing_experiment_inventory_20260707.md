# Paper-Facing Experiment Inventory

Date: 2026-07-07

This inventory consolidates the current WaveSlice experiments for the CUCUMIS
paper revision. It separates completed paper-facing results from auxiliary
audits and candidates that should not block evaluation writing.

## Completed Paper-Facing Experiments

| Experiment | Status | Paper use | Primary artifact |
| --- | --- | --- | --- |
| Single-GPU CUCUMIS ratio sweep | Complete | Main request-heterogeneity sensitivity figure/table | `docs/ratio_sweep_20step_5models_a100_overnight.md`; `results/chapter5_exports/ratio_sweep_20step_5models_a100_overnight/ratio_sweep_summary.csv` |
| Single-GPU guardrails | Complete | Show TTFT gains are not only from sacrificing completion, wall time, or throughput | `results/chapter5_exports/ratio_sweep_20step_5models_a100_overnight/ratio_sweep_guardrails.csv` |
| DistServe continuous-batching equal-resource comparison | Complete | Core DistServe baseline credibility result | `results/openworkload_ratio_sweep_lora8/unified_distserve_continuous_cucumis_2a100_replay/cucumis_2a100/distserve_equal_resource_real_comparison.csv` |
| DistServe replay method metrics | Complete | Provenance for method semantics | `results/openworkload_ratio_sweep_lora8/unified_distserve_continuous_cucumis_2a100_replay/distserve_serial/distserve_serial_method_metrics.csv` |
| A100 / RTX 4090 / RTX 5090 hardware portability | Complete | Portability subsection or appendix table | `results/chapter5_exports/hardware_portability_a100_4090_5090/hardware_portability_summary.csv` |
| Hardware exception audit | Complete offline listing | Caveat wording and optional spot-rerun shortlist | `results/chapter5_exports/hardware_portability_a100_4090_5090/hardware_portability_anomalies.csv` |

## Key Numbers

DistServe continuous-batching A100 equal-resource comparison, least-backlog
CUCUMIS rows only:

- all-request p99 TTFT: 3.79x mean DistServe/CUCUMIS
- short-request p99 TTFT: 6.01x
- all-request p99 completion: 4.69x
- round wall time: 3.14x
- throughput ratio: 0.32x

Hardware portability, least-backlog CUCUMIS:

| GPU | all TTFT p99 | short TTFT p99 | completion p99 | wall time | throughput Dist/CUC | note |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| A100 | 2.54x | 4.61x | 3.75x | 2.48x | 0.41x | Strong average and guardrail wins; all-TTFT has 11 exceptions |
| RTX 4090 | 1.34x | 3.33x | 2.09x | 1.61x | 0.79x | Memory-stress profile; do not claim universal wins |
| RTX 5090 | 1.50x | 3.35x | 2.60x | 1.84x | 0.56x | Short-TTFT exceptions are limited to 3 Gemma-7B cases |

## Auxiliary / Diagnostic Results

| Result | Status | Use |
| --- | --- | --- |
| Early DistServe functional reproduction and sensitivity results | Complete but auxiliary | Calibration and sensitivity audit only |
| Request-serial DistServe replay | Diagnostic only | Do not use as paper-facing baseline; continuous batching supersedes it |
| Partial CUCUMIS split-run smoke/formal roots | Diagnostic only | Do not use as formal comparison |
| Partial or interrupted hardware-portability roots | Diagnostic only | Use copied-back supervised full5 roots and merged exports instead |

## Remaining Work

1. Update `evaluation.tex` with the consolidated tables and cautious wording.
2. Decide whether RTX 4090/5090 exception cases need spot reruns. The exception
   list is already generated, so this is a judgment call rather than a
   prerequisite for writing.
3. Keep official multi-GPU DistServe deployment reproduction as optional future
   work; it should not block the current revision.

## Required DistServe Wording

Use:

> continuous-batching replay under equal-resource comparison, using physical
> single-GPU stage-cost measurements and a logical 2-GPU DistServe setting.

Avoid:

> single-GPU measured DistServe comparison.

The shorter wording is ambiguous and can make the baseline look weaker or less
carefully controlled than it is.
