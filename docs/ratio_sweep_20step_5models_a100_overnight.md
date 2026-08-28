# Ratio Sweep 20-Step 5-Model A100 Overnight

> Historical run note: `strict_no_chunk` is the label stored in this result
> tree. The active baseline is now named `priority_no_chunk`; the old label is
> retained only so the legacy result exporter can read this run.

This document freezes the completed request-ratio sensitivity experiment used
to answer whether CUCUMIS remains effective when agent and RAG long requests
become more common.

## Status

- Status: complete, frozen formal result
- Paper integration: 预定加入论文的部分; do not insert into the manuscript until
  all planned experiments finish
- Planned paper section: "Sensitivity to request heterogeneity"
- Run tag: `ratio_sweep_20step_5models_a100_overnight`
- Date completed: `2026-07-03`
- Branch at inspection: `v1-investigation`
- Upstream at inspection: `origin/v1-investigation`
- HEAD at inspection: `cc926e623c61ba17470959e76e3d05772febe3d7`
- GPU: single A100 80GB

This run supersedes the aborted 10%-step run
`ratio_sweep_formal_baichuan2_a100_r1`.

## Paper Integration Holding Note

Keep this result as a paper-ready evidence card for now. The manuscript should
not be edited yet; once all planned follow-up experiments finish, integrate this
card into the planned "Sensitivity to request heterogeneity" section together
with the guardrail table and any proxy-baseline results.

## Scope

- Models:
  - `baichuan2-7b-chat`
  - `gemma-7b-it`
  - `qwen2.5-7b-instruct`
  - `mistral-7b-instruct-v0.2`
  - `gemma-2-9b-it`
- Workload cases per model:
  - `mid_l10`, `mid_l30`, `mid_l50`, `mid_l70`, `mid_l90`
  - `high_l10`, `high_l30`, `high_l50`, `high_l70`, `high_l90`
- Ratio points: 10%, 30%, 50%, 70%, 90% long requests
- Density levels: `mid`, `high`
- Main cases: 50/50 ok
- Baseline variant cases: 100/100 ok
- Baseline variants:
  - `fixed_chunk_vs_sarathi`
  - `strict_no_chunk`

## Command

```bash
setsid env PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 \
  /home/onceas/anaconda3/envs/sara/bin/python experiments/run_chapter5_suite.py \
  --config experiments/configs/chapter5_pipeline_ratio_sweep.json \
  --stages preflight,main,baseline \
  --run-tag ratio_sweep_20step_5models_a100_overnight \
  --model-keys baichuan2-7b-chat,gemma-7b-it,qwen2.5-7b-instruct,mistral-7b-instruct-v0.2,gemma-2-9b-it \
  --variants fixed_chunk_vs_sarathi,strict_no_chunk \
  > results/run_logs/ratio_sweep_20step_5models_a100_overnight.log 2>&1 < /dev/null &
```

## Artifacts

- Log:
  `results/run_logs/ratio_sweep_20step_5models_a100_overnight.log`
- Pipeline manifest:
  `results/chapter5_exports/ratio_sweep_20step_5models_a100_overnight/chapter5_pipeline_manifest.json`
- Preflight root:
  `results/openworkload_ratio_sweep_lora8/ratio_sweep_20step_5models_a100_overnight_preflight`
- Main run root:
  `results/openworkload_ratio_sweep_lora8/ratio_sweep_20step_5models_a100_overnight_main`
- Baseline run root:
  `results/chapter5_baseline_variants/ratio_sweep_20step_5models_a100_overnight_baseline`
- Export root:
  `results/chapter5_exports/ratio_sweep_20step_5models_a100_overnight`

Key exported files:

- `ratio_sweep_summary.csv`
- `ratio_sweep_guardrails.csv`
- `ratio_sweep_method_metrics.csv`
- `ratio_sweep_per_model_comparison.csv`
- `ratio_sweep_ttft_improvement.pdf`
- `ratio_sweep_guardrails.pdf`

## Result Summary

Five-model aggregate:

| Metric | Mean | Min | Max |
| --- | ---: | ---: | ---: |
| CUCUMIS vs vLLM all p99 TTFT improvement | 2.385x | 1.351x | 6.420x |
| CUCUMIS vs vLLM short p99 TTFT improvement | 1.032x | 0.781x | 1.808x |
| CUCUMIS vs vLLM round-wall ratio | 0.931x | 0.557x | 1.064x |
| CUCUMIS vs vLLM throughput ratio | 1.093x | 0.940x | 1.796x |
| CUCUMIS vs vLLM all completion p99 improvement | 1.218x | 0.917x | 1.814x |

Formal three-model aggregate (`Baichuan2`, `Gemma-7B`, `Qwen2.5`):

| Metric | Mean | Min | Max |
| --- | ---: | ---: | ---: |
| CUCUMIS vs vLLM all p99 TTFT improvement | 2.413x | 1.356x | 6.420x |
| CUCUMIS vs vLLM short p99 TTFT improvement | 1.043x | 0.781x | 1.808x |
| CUCUMIS vs vLLM round-wall ratio | 0.922x | 0.557x | 1.057x |
| CUCUMIS vs vLLM throughput ratio | 1.109x | 0.946x | 1.796x |
| CUCUMIS vs vLLM all completion p99 improvement | 1.242x | 0.917x | 1.814x |

Aggregate by density and target long-request fraction:

| Density | Long % | All p99 TTFT | Short p99 TTFT | Round-wall ratio | Throughput ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| high | 10 | 1.544x | 1.068x | 1.017x | 0.984x |
| high | 30 | 1.912x | 1.004x | 1.029x | 0.972x |
| high | 50 | 1.885x | 1.214x | 0.915x | 1.094x |
| high | 70 | 3.244x | 1.091x | 0.829x | 1.213x |
| high | 90 | 4.186x | 0.978x | 0.733x | 1.390x |
| mid | 10 | 1.464x | 0.964x | 1.009x | 0.991x |
| mid | 30 | 1.500x | 0.922x | 1.025x | 0.976x |
| mid | 50 | 1.517x | 0.973x | 0.980x | 1.021x |
| mid | 70 | 2.766x | 0.980x | 0.910x | 1.103x |
| mid | 90 | 3.837x | 1.123x | 0.858x | 1.188x |

## Interpretation

- The main paper story should emphasize all-request p99 TTFT: gains grow as
  the long-request fraction rises, especially at 70%-90%.
- Short-request p99 TTFT should be framed as mostly preserved, not as a
  uniformly strong win. Some low-ratio points are slightly below 1.0.
- Round wall time and throughput are acceptable guardrails overall. The
  aggregate round-wall ratio is below 1.0 for the formal three-model subset
  and for all five models, while throughput is above 1.0 on average.
- Use this experiment for the planned section "Sensitivity to request
  heterogeneity".

## Rebuilding Tables And Figures

```bash
/home/onceas/anaconda3/envs/sara/bin/python scripts/regenerate_ratio_sweep.py \
  --main-run results/openworkload_ratio_sweep_lora8/ratio_sweep_20step_5models_a100_overnight_main \
  --baseline-run results/chapter5_baseline_variants/ratio_sweep_20step_5models_a100_overnight_baseline \
  --out-dir results/chapter5_exports/ratio_sweep_20step_5models_a100_overnight
```

## Do Not Confuse With

- `ratio_sweep_formal_baichuan2_a100_r1`: aborted after one completed 10%-step
  point when the sweep was changed from 10% increments to the five-point
  20%-step design.
- `ratio_sweep_ratiofix_l90_retry_a100`: smoke/retry run used to verify the
  90% long-request endpoint and the V1 finished-request queue fix.
- `ratio_sweep_ratiofix_l90_baseline_smoke_a100`: baseline smoke run for the
  90% long-request endpoint.
