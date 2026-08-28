# Frozen Result: Ratio Sweep 20-Step 5-Model A100 Overnight

This directory is the frozen export root for:

`ratio_sweep_20step_5models_a100_overnight`

Use this directory as the formal ratio-sweep result for the request
heterogeneity sensitivity experiment.

## Status

- Complete: yes
- Paper integration: 预定加入论文的部分; do not insert into the manuscript yet
- Main cases: 50/50 ok
- Baseline variant cases: 100/100 ok
- Models: Baichuan2, Gemma-7B, Qwen2.5, Mistral-Instruct, Gemma-2-9B
- Workloads: `mid/high x {10%,30%,50%,70%,90%}` long-request fraction

## Important Files

- `ratio_sweep_summary.csv`: compact aggregate table
- `ratio_sweep_guardrails.csv`: completion, wall-time, throughput guardrails
- `ratio_sweep_per_model_comparison.csv`: per-model CUCUMIS vs baselines
- `ratio_sweep_method_metrics.csv`: method-level metric table
- `ratio_sweep_ttft_improvement.pdf`: paper-facing TTFT trend figure
- `ratio_sweep_guardrails.pdf`: paper-facing guardrail figure
- `chapter5_pipeline_manifest.json`: links preflight, main, and baseline roots

## Key Takeaway

All-request p99 TTFT gains grow with the long-request fraction. Across five
models, CUCUMIS averages 2.385x all p99 TTFT improvement over vLLM, while
round-wall ratio averages 0.931x and throughput averages 1.093x. Short-request
p99 TTFT is mostly preserved rather than uniformly improved.

For the full frozen experiment card, see:

`docs/ratio_sweep_20step_5models_a100_overnight.md`
