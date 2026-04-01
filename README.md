# Wave-Slice

Wave-Slice is a fairness-aware spatio-temporal scheduling layer for heterogeneous LoRA concurrent inference.

This repository keeps a strict decoupling from vLLM source code by using runtime scheduler hijacking (monkey patching) only.

## Quick start

```python
from waveslice import inject_wave_slice, uninject_wave_slice

inject_wave_slice("Mistral-7B-v0.1")
# build and run your vLLM engine
uninject_wave_slice()
```
