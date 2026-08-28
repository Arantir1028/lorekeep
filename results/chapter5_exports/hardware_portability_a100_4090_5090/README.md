# Frozen Result: Hardware Portability A100 / RTX 4090 / RTX 5090

This directory is the paper-facing export root for `hardware_portability_a100_4090_5090`.

## Status

- Complete: yes
- Main cases: 3 GPU profiles x 50 cases
- Methods: `DistServe-2GPU` and `CUCUMIS-2GPU-LB`
- DistServe semantics: single-GPU physical stage-cost measurement, token-level continuous-batching replay, equal-resource logical 2-GPU comparison
- CUCUMIS semantics: two real CUCUMIS replicas on the same two local GPUs, least-backlog dispatch

## Important Files

- `hardware_portability_summary.csv`: compact aggregate table
- `hardware_portability_summary.tex`: LaTeX table for paper drafting
- `hardware_portability_case_comparison.csv`: normalized 150-case comparison table
- `hardware_portability_long_ratio_trend.csv`: ratio trend by long-request fraction
- `hardware_portability_anomalies.csv`: cases where at least one paper-facing metric does not favor CUCUMIS
- `hardware_portability_manifest.json`: source-result provenance

## Aggregate Results

| GPU | cases | all TTFT p99 | short TTFT p99 | completion p99 | wall | throughput | TTFT wins |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A100 | 50 | 2.54x | 4.61x | 3.75x | 2.48x | 0.41x | 39 / 50 |
| RTX 4090 | 50 | 1.34x | 3.33x | 2.09x | 1.61x | 0.79x | 19 / 50 |
| RTX 5090 | 50 | 1.50x | 3.35x | 2.60x | 1.84x | 0.56x | 22 / 50 |

Ratios are `DistServe / CUCUMIS`. For latency and wall time, larger than 1 means CUCUMIS is better. For throughput, smaller than 1 means CUCUMIS has higher throughput.

## Caveats

- Do not write that CUCUMIS wins every workload. Use average and win-count wording.
- RTX 4090 is the memory-stress profile; its all-request TTFT is mixed, while completion, wall time, and throughput still favor CUCUMIS on most cases.
- RTX 5090 has only a few short-TTFT exceptions, concentrated in Gemma-7B cases; completion, wall time, and throughput win in all 50 cases.

Exception counts by GPU:

- A100: 11 cases with at least one losing metric; 0 short-TTFT exceptions.
- RTX 4090: 31 cases with at least one losing metric; 19 short-TTFT exceptions.
- RTX 5090: 28 cases with at least one losing metric; 3 short-TTFT exceptions.
