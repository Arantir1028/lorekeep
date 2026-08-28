# Hardware Portability: A100, RTX 4090, RTX 5090

Status: completed server runs; paper-facing exports generated.

Date: 2026-07-07

## Goal

This experiment answers the reviewer concern that CUCUMIS was evaluated on only
one GPU type. It compares the same 50 open-workload cases on:

- `A100`: datacenter HBM reference.
- `RTX 4090`: high-end consumer GPU with tight 24 GB memory headroom.
- `RTX 5090`: new-generation consumer GPU with larger VRAM than RTX 4090.

The hardware sweep should be used as portability and robustness evidence. The
cleanest main DistServe-vs-CUCUMIS controlled comparison remains the A100
continuous-batching replay experiment under
`results/openworkload_ratio_sweep_lora8/unified_distserve_continuous_cucumis_2a100_replay`.

## Scope

Source workload:

- `results/openworkload_ratio_sweep_lora8/ratio_sweep_20step_5models_a100_overnight_main`

Cases:

- 5 models:
  `baichuan2-7b-chat`, `gemma-2-9b-it`, `gemma-7b-it`,
  `mistral-7b-instruct-v0.2`, `qwen2.5-7b-instruct`
- 10 workloads:
  `mid_l10/30/50/70/90`, `high_l10/30/50/70/90`

Methods:

- `DistServe-2GPU`: physical single-GPU stage-cost measurement on the same GPU
  type, followed by token-level continuous-batching replay under a logical
  equal-resource 2-GPU comparison.
- `CUCUMIS-2GPU-LB`: two real split-workload CUCUMIS replicas on the same two
  local GPUs, using least-backlog dispatch.

This sweep intentionally uses only `least_backlog`. The earlier RR/LB dispatcher
comparison remains in the A100 continuous-batching DistServe experiment; this
hardware sweep is for portability, not dispatcher ablation.

## Completed Result Roots

Use the copied-back completed roots below. Do not use partial or interrupted
roots under `results/hardware_portability_a100_4090_5090` as paper-facing
summaries.

| GPU | Completed root |
| --- | --- |
| A100 | `results/hardware_portability_remote_mirror/a100/results/hardware_portability_supervised_full5/a100/unified_distserve_cucumis_2gpu_lb` |
| RTX 4090 | `results/hardware_portability_remote_mirror/rtx4090/results/hardware_portability_supervised_full5/rtx4090/unified_distserve_cucumis_2gpu_lb` |
| RTX 5090 | `results/hardware_portability_remote_mirror/rtx5090/results/hardware_portability_supervised_full5/rtx5090/unified_distserve_cucumis_2gpu_lb` |

Paper-facing merged exports:

- `results/chapter5_exports/hardware_portability_a100_4090_5090/README.md`
- `results/chapter5_exports/hardware_portability_a100_4090_5090/hardware_portability_summary.csv`
- `results/chapter5_exports/hardware_portability_a100_4090_5090/hardware_portability_summary.tex`
- `results/chapter5_exports/hardware_portability_a100_4090_5090/hardware_portability_case_comparison.csv`
- `results/chapter5_exports/hardware_portability_a100_4090_5090/hardware_portability_anomalies.csv`

Regenerate exports with:

```bash
python3 scripts/summarize_hardware_portability_results.py
```

## Aggregate Results

Ratios are `DistServe / CUCUMIS`. For latency and wall time, larger than 1 means
CUCUMIS is better. For throughput, smaller than 1 means CUCUMIS has higher
throughput.

| GPU | cases | all TTFT p99 | short TTFT p99 | completion p99 | wall time | throughput | all TTFT wins |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A100 | 50 | 2.54x | 4.61x | 3.75x | 2.48x | 0.41x | 39 / 50 |
| RTX 4090 | 50 | 1.34x | 3.33x | 2.09x | 1.61x | 0.79x | 19 / 50 |
| RTX 5090 | 50 | 1.50x | 3.35x | 2.60x | 1.84x | 0.56x | 22 / 50 |

Win counts:

| GPU | all TTFT | short TTFT | completion | wall | throughput |
| --- | ---: | ---: | ---: | ---: | ---: |
| A100 | 39 / 50 | 50 / 50 | 50 / 50 | 50 / 50 | 50 / 50 |
| RTX 4090 | 19 / 50 | 31 / 50 | 37 / 50 | 40 / 50 | 40 / 50 |
| RTX 5090 | 22 / 50 | 47 / 50 | 50 / 50 | 50 / 50 | 50 / 50 |

## Long-Ratio Trend

Average all/short p99 TTFT ratios by target long-request fraction:

| GPU | 10% | 30% | 50% | 70% | 90% |
| --- | ---: | ---: | ---: | ---: | ---: |
| A100 | 0.94x / 2.58x | 2.01x / 6.21x | 2.29x / 5.89x | 3.20x / 5.20x | 4.27x / 3.14x |
| RTX 4090 | 0.74x / 1.97x | 0.81x / 2.47x | 1.30x / 3.12x | 1.70x / 4.81x | 2.14x / 4.30x |
| RTX 5090 | 0.92x / 2.73x | 1.01x / 2.73x | 1.15x / 3.12x | 1.93x / 4.04x | 2.47x / 4.12x |

The A100 and RTX 5090 trends match the request-heterogeneity story: as the
long-request fraction rises, CUCUMIS's all-request TTFT advantage becomes more
visible, while short-request TTFT stays strongly protected. RTX 4090 is a memory
stress case: short-request TTFT still improves on average, but all-request TTFT
is mixed and should be described conservatively.

## Exception Audit

The export `hardware_portability_anomalies.csv` lists every case where at least
one paper-facing metric does not favor CUCUMIS.

Summary:

- A100: 11 all-TTFT exceptions; 0 short-TTFT, completion, wall, or throughput
  exceptions.
- RTX 4090: 31 cases with at least one exception; 19 short-TTFT exceptions.
  Exceptions concentrate in Baichuan, Gemma-2, and Gemma-7B under the tightest
  memory profile.
- RTX 5090: 28 all-TTFT exceptions, but only 3 short-TTFT exceptions. The
  short-TTFT exceptions are all Gemma-7B cases; completion, wall time, and
  throughput win in all 50 cases.

Do not write that CUCUMIS wins every workload. The safe paper wording is:

> Across A100, RTX 4090, and RTX 5090 two-GPU runs, CUCUMIS preserves average
> short-request p99 TTFT advantages and improves completion tail latency, round
> wall time, and throughput on average; RTX 4090 is a memory-constrained stress
> case with mixed all-request TTFT.

## DistServe Wording

Use this exact method wording in the paper:

> We compare against a DistServe P/D-separated baseline using physical
> single-GPU stage-cost measurements and token-level continuous-batching replay
> under an equal-resource logical 2-GPU setting.

Avoid calling this a plain "single-GPU measured comparison"; that loses the
important equal-resource and continuous-batching semantics.
