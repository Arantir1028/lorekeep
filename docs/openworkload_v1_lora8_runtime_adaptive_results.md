# Open-Workload V1 LoRA8 Runtime-Adaptive Result Snapshot

This snapshot fixes the current full 8-model open-workload result after the
V1 RUNNING-state queue reconciliation fix.

## Run

- Run root: `results/openworkload_v1_local_realworld_lora8/runtime_adaptive_full8_20260427_143128`
- Config: `experiments/configs/openworkload_v1_local_realworld_lora8.json`
- Runner: `experiments/run_openworkload_execescape_suite.py`
- Dataset keys: `ultrachat200k,longbench`
- Densities: `low,mid,high,peak`
- Status: `32/32` cases completed successfully

## Overall

| Metric | Mean |
| --- | ---: |
| Phase I+II TTFT improvement | 2.309x |
| Phase I+II wall-time improvement | 1.037x |
| Phase I+II slowdown improvement | 1.117x |

## By Model

| Model | Cases | TTFT | Wall | Slowdown |
| --- | ---: | ---: | ---: | ---: |
| Mistral-7B-v0.1 | 4/4 | 2.591x | 1.012x | 1.096x |
| Mistral-7B-Instruct-v0.2 | 4/4 | 2.091x | 1.040x | 1.124x |
| Zephyr-7B-Beta | 4/4 | 2.200x | 1.021x | 1.105x |
| OpenChat-3.5-0106 | 4/4 | 2.629x | 1.018x | 1.115x |
| Gemma-7B-IT | 4/4 | 2.984x | 1.102x | 1.217x |
| Baichuan2-7B-Chat | 4/4 | 2.067x | 1.012x | 1.067x |
| Qwen2.5-7B-Instruct | 4/4 | 1.927x | 1.042x | 1.133x |
| Gemma-2-9B-IT | 4/4 | 1.983x | 1.046x | 1.081x |

## By Density

| Density | Cases | TTFT | Wall | Slowdown |
| --- | ---: | ---: | ---: | ---: |
| low | 8 | 1.045x | 1.012x | 1.036x |
| mid | 8 | 1.753x | 1.011x | 1.107x |
| high | 8 | 3.043x | 1.025x | 1.142x |
| peak | 8 | 3.395x | 1.099x | 1.184x |

## Notes

- The current method is Phase I chunk-size boundary control plus Phase II
  execution escape / priority promotion with runtime queue-pressure adaptation.
- The result directory itself is ignored by git through `results/`; this
  document is the repository-tracked result snapshot.
- The two previously failing cases were rerun successfully after adding V1
  queue reconciliation:
  - `Baichuan2-7B-Chat / mid`: TTFT 1.882x, wall 0.990x, slowdown 1.052x.
  - `Gemma-2-9B-IT / high`: TTFT 2.262x, wall 1.005x, slowdown 1.028x.
