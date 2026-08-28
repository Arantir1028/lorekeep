from __future__ import annotations

import json
import os
from collections.abc import Iterable
from dataclasses import dataclass


@dataclass
class AdapterSpec:
    name: str
    rank: int
    alpha: int
    seed: int
    init_std: float


def _infer_target_modules_from_config(config: object) -> list[str]:
    model_type = str(getattr(config, "model_type", "") or "").lower()
    if model_type == "baichuan":
        return ["W_pack", "o_proj", "gate_proj", "up_proj", "down_proj"]
    if model_type in {"deci", "gemma", "gemma2", "llama", "mistral", "qwen2"}:
        return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    raise RuntimeError(f"Unsupported config-only LoRA generation for model_type={model_type!r}")


def _build_config_linear_specs(config: object) -> list[tuple[str, int, int]]:
    model_type = str(getattr(config, "model_type", "") or "").lower()
    hidden = int(config.hidden_size)
    intermediate = int(config.intermediate_size)
    layers = int(config.num_hidden_layers)
    specs: list[tuple[str, int, int]] = []
    if model_type == "baichuan":
        for i in range(layers):
            prefix = f"model.layers.{i}"
            specs.extend(
                [
                    (f"{prefix}.self_attn.W_pack", 3 * hidden, hidden),
                    (f"{prefix}.self_attn.o_proj", hidden, hidden),
                    (f"{prefix}.mlp.gate_proj", intermediate, hidden),
                    (f"{prefix}.mlp.up_proj", intermediate, hidden),
                    (f"{prefix}.mlp.down_proj", hidden, intermediate),
                ]
            )
        return specs
    if model_type in {"deci", "gemma", "gemma2", "llama", "mistral", "qwen2"}:
        num_heads = int(config.num_attention_heads)
        head_dim = hidden // max(1, num_heads)
        kv_heads_list = getattr(config, "num_key_value_heads_per_layer", None)
        default_kv_heads = int(
            getattr(config, "num_key_value_heads", 0)
            or getattr(config, "num_key_value_heads", num_heads)
        )
        for i in range(layers):
            kv_heads = int(kv_heads_list[i]) if kv_heads_list is not None else default_kv_heads
            kv_dim = kv_heads * head_dim
            prefix = f"model.layers.{i}"
            specs.extend(
                [
                    (f"{prefix}.self_attn.q_proj", hidden, hidden),
                    (f"{prefix}.self_attn.k_proj", kv_dim, hidden),
                    (f"{prefix}.self_attn.v_proj", kv_dim, hidden),
                    (f"{prefix}.self_attn.o_proj", hidden, hidden),
                    (f"{prefix}.mlp.gate_proj", intermediate, hidden),
                    (f"{prefix}.mlp.up_proj", intermediate, hidden),
                    (f"{prefix}.mlp.down_proj", hidden, intermediate),
                ]
            )
        return specs
    raise RuntimeError(f"Unsupported config-only LoRA generation for model_type={model_type!r}")


def _write_manual_adapter_from_specs(
    *,
    base_model: str,
    target_path: str,
    spec: AdapterSpec,
    target_modules: list[str],
    linear_specs: list[tuple[str, int, int]],
) -> str:
    import torch
    from safetensors.torch import save_file

    tensors: dict[str, torch.Tensor] = {}
    g = torch.Generator(device="cpu")
    g.manual_seed(spec.seed)
    for full_name, out_features, in_features in linear_specs:
        lora_a = torch.empty((spec.rank, in_features), dtype=torch.float16)
        lora_b = torch.empty((out_features, spec.rank), dtype=torch.float16)
        lora_a.normal_(mean=0.0, std=spec.init_std, generator=g)
        lora_b.normal_(mean=0.0, std=spec.init_std, generator=g)
        tensors[f"base_model.model.{full_name}.lora_A.weight"] = lora_a
        tensors[f"base_model.model.{full_name}.lora_B.weight"] = lora_b
    adapter_config = {
        "base_model_name_or_path": base_model,
        "bias": "none",
        "fan_in_fan_out": False,
        "inference_mode": True,
        "lora_alpha": spec.alpha,
        "lora_dropout": 0.0,
        "peft_type": "LORA",
        "r": spec.rank,
        "target_modules": target_modules,
        "task_type": "CAUSAL_LM",
    }
    with open(os.path.join(target_path, "adapter_config.json"), "w", encoding="utf-8") as f:
        json.dump(adapter_config, f, ensure_ascii=False, indent=2)
    save_file(tensors, os.path.join(target_path, "adapter_model.safetensors"))
    return target_path


def _build_one_adapter(
    *, base_model: str, out_dir: str, spec: AdapterSpec, trust_remote_code: bool = False
) -> str:
    from transformers import AutoConfig

    os.makedirs(out_dir, exist_ok=True)
    target_path = os.path.join(out_dir, spec.name)
    os.makedirs(target_path, exist_ok=True)
    config = AutoConfig.from_pretrained(base_model, trust_remote_code=trust_remote_code)
    target_modules = _infer_target_modules_from_config(config)
    linear_specs = _build_config_linear_specs(config)
    return _write_manual_adapter_from_specs(
        base_model=base_model,
        target_path=target_path,
        spec=spec,
        target_modules=target_modules,
        linear_specs=linear_specs,
    )


def build_synthetic_adapters(
    *, base_model: str, out_dir: str, specs: Iterable[AdapterSpec], trust_remote_code: bool = False
) -> list[str]:
    paths: list[str] = []
    for spec in specs:
        paths.append(
            _build_one_adapter(
                base_model=base_model,
                out_dir=out_dir,
                spec=spec,
                trust_remote_code=trust_remote_code,
            )
        )
    return paths
