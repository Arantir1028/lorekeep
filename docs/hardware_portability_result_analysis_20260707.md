# Hardware Portability Result Analysis

Date: 2026-07-07

This note summarizes the copied-back A100 / RTX 4090 / RTX 5090 hardware
portability results for the CUCUMIS paper revision. The numbers here are
regenerated from the paper-facing export:

`results/chapter5_exports/hardware_portability_a100_4090_5090`

## Source Roots

Use these completed roots:

- A100: `results/hardware_portability_remote_mirror/a100/results/hardware_portability_supervised_full5/a100/unified_distserve_cucumis_2gpu_lb`
- RTX 4090: `results/hardware_portability_remote_mirror/rtx4090/results/hardware_portability_supervised_full5/rtx4090/unified_distserve_cucumis_2gpu_lb`
- RTX 5090: `results/hardware_portability_remote_mirror/rtx5090/results/hardware_portability_supervised_full5/rtx5090/unified_distserve_cucumis_2gpu_lb`

Do not use the partial roots under `hardware_portability_a100_4090_5090` as
paper-facing summaries; for some hosts those contain only metadata or
interrupted progress.

## Scope

- Methods:
  - `DistServe-2GPU`: physical single-GPU stage-cost measurement on the same GPU
    type, token-level continuous-batching replay, logical equal-resource 2-GPU
    comparison.
  - `CUCUMIS-2GPU-LB`: two real CUCUMIS replicas on two local GPUs, using
    least-backlog dispatch.
- Models: `baichuan2-7b-chat`, `gemma-2-9b-it`, `gemma-7b-it`,
  `mistral-7b-instruct-v0.2`, `qwen2.5-7b-instruct`.
- Workloads: `mid_l10/30/50/70/90` and `high_l10/30/50/70/90`.
- Output length: 64 tokens.
- Completed rows per hardware: 50 DistServe method rows, 50 CUCUMIS method
  rows, 50 comparison rows, and 8000 request rows.

Ratios below are `DistServe / CUCUMIS`. For latency and wall time, larger than
1 means CUCUMIS is better. For throughput, smaller than 1 means CUCUMIS has
higher throughput.

## Aggregate Results

| GPU | all TTFT p99 | short TTFT p99 | all completion p99 | round wall | throughput Dist/CUC | all TTFT wins | short TTFT wins |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A100 | 2.54x | 4.61x | 3.75x | 2.48x | 0.41x | 39 / 50 | 50 / 50 |
| RTX 4090 | 1.34x | 3.33x | 2.09x | 1.61x | 0.79x | 19 / 50 | 31 / 50 |
| RTX 5090 | 1.50x | 3.35x | 2.60x | 1.84x | 0.56x | 22 / 50 | 47 / 50 |

Guardrail win counts:

| GPU | completion | wall | throughput |
| --- | ---: | ---: | ---: |
| A100 | 50 / 50 | 50 / 50 | 50 / 50 |
| RTX 4090 | 37 / 50 | 40 / 50 | 40 / 50 |
| RTX 5090 | 50 / 50 | 50 / 50 | 50 / 50 |

## Long-Ratio Trend

Average all/short p99 TTFT ratios:

| GPU | l10 | l30 | l50 | l70 | l90 |
| --- | ---: | ---: | ---: | ---: | ---: |
| A100 | 0.94x / 2.58x | 2.01x / 6.21x | 2.29x / 5.89x | 3.20x / 5.20x | 4.27x / 3.14x |
| RTX 4090 | 0.74x / 1.97x | 0.81x / 2.47x | 1.30x / 3.12x | 1.70x / 4.81x | 2.14x / 4.30x |
| RTX 5090 | 0.92x / 2.73x | 1.01x / 2.73x | 1.15x / 3.12x | 1.93x / 4.04x | 2.47x / 4.12x |

The trend supports a cautious request-heterogeneity claim: higher long-request
fractions generally make the CUCUMIS all-request TTFT advantage more visible,
while short-request TTFT remains protected on average. RTX 4090 should be
treated as a memory-constrained stress case rather than a clean algorithmic
counterexample.

## Exception Audit

Detailed exception rows are in:

`results/chapter5_exports/hardware_portability_a100_4090_5090/hardware_portability_anomalies.csv`

Important exception groups:

- A100 has 11 all-TTFT exceptions, but no short-TTFT or guardrail exceptions.
- RTX 4090 has 31 all-TTFT exceptions and 19 short-TTFT exceptions. The short
  exceptions concentrate in Baichuan, Gemma-2, and Gemma-7B. This profile used
  tight memory settings: default `gpu_memory_utilization=0.75`,
  Gemma-2 at `0.90` with `max_num_batched_tokens=768`, and Gemma-7B at `0.85`.
- RTX 5090 has 28 all-TTFT exceptions, but only 3 short-TTFT exceptions, all in
  Gemma-7B. Completion, wall time, and throughput win in every RTX 5090 case.

## Paper Use

Strong claims:

- CUCUMIS ports across datacenter and consumer GPUs.
- Average short-request p99 TTFT improves across A100, RTX 4090, and RTX 5090.
- Completion tail latency, round wall time, and throughput improve on average
  across all three hardware profiles.

Cautious claims:

- Do not claim every workload wins.
- RTX 4090 is memory constrained and has mixed all-request TTFT; discuss it as
  a stress case or put it in appendix.
- The hardware sweep is valuable portability evidence, but the A100
  continuous-batching replay remains the cleaner primary DistServe baseline
  table.
