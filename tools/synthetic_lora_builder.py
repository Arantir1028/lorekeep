"""Build synthetic LoRA adapters for a given base model.

This is useful when you do not have real task-trained adapters yet, but still
need to validate same-base + heterogeneous-LoRA serving pipelines end-to-end.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Iterable

from config.experiment_catalog import DEFAULT_SYNTHETIC_ADAPTER_PRESETS
from engine.runtime_bootstrap import bootstrap_vllm_runtime

bootstrap_vllm_runtime()

import safetensors.torch
import torch
from transformers import AutoConfig, AutoModelForCausalLM


@dataclass
class AdapterSpec:
    name: str
    rank: int
    alpha: int
    seed: int
    init_std: float


def _infer_target_modules(model: torch.nn.Module) -> list[str]:
    linear_leaf_names: set[str] = set()
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            linear_leaf_names.add(name.split(".")[-1])

    preferred = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "up_proj",
        "down_proj",
        "gate_proj",
        "query_key_value",
        "dense",
        "dense_h_to_4h",
        "dense_4h_to_h",
    ]
    selected = [n for n in preferred if n in linear_leaf_names]
    if selected:
        return selected

    # Conservative fallback for unknown architectures.
    fallback = sorted(
        n for n in linear_leaf_names if any(k in n for k in ("proj", "dense", "query", "value", "key"))
    )
    if fallback:
        return fallback[:16]
    raise RuntimeError("Unable to infer target_modules for LoRA from base model.")


def _infer_target_modules_from_config(config: object) -> list[str]:
    model_type = str(getattr(config, "model_type", "") or "").lower()
    if model_type == "baichuan":
        return ["W_pack", "o_proj", "gate_proj", "up_proj", "down_proj"]
    if model_type in {"deci", "llama", "mistral"}:
        return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    raise RuntimeError(f"Unsupported config-only LoRA generation for model_type={model_type!r}")


def _build_config_linear_specs(config: object) -> list[tuple[str, int, int]]:
    model_type = str(getattr(config, "model_type", "") or "").lower()
    hidden = int(getattr(config, "hidden_size"))
    intermediate = int(getattr(config, "intermediate_size"))
    layers = int(getattr(config, "num_hidden_layers"))

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

    if model_type in {"deci", "llama", "mistral"}:
        num_heads = int(getattr(config, "num_attention_heads"))
        head_dim = hidden // max(1, num_heads)
        kv_heads_list = getattr(config, "num_key_value_heads_per_layer", None)
        default_kv_heads = int(getattr(config, "num_key_value_heads", 0) or getattr(config, "num_key_value_heads", num_heads))
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
    safetensors.torch.save_file(tensors, os.path.join(target_path, "adapter_model.safetensors"))
    return target_path


def _build_one_adapter(
    *,
    base_model: str,
    out_dir: str,
    spec: AdapterSpec,
    trust_remote_code: bool = False,
) -> str:
    os.makedirs(out_dir, exist_ok=True)
    target_path = os.path.join(out_dir, spec.name)
    os.makedirs(target_path, exist_ok=True)
    model: torch.nn.Module | None = None
    model_exc: Exception | None = None
    try:
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map="cpu",
            trust_remote_code=trust_remote_code,
        )
    except Exception as exc:  # pragma: no cover - exercised in integration
        model_exc = exc

    if model is not None:
        target_modules = _infer_target_modules(model)
        try:
            from peft import LoraConfig, get_peft_model
            lora_cfg = LoraConfig(
                r=spec.rank,
                lora_alpha=spec.alpha,
                lora_dropout=0.0,
                bias="none",
                target_modules=target_modules,
                task_type="CAUSAL_LM",
            )
            peft_model = get_peft_model(model, lora_cfg)

            g = torch.Generator(device="cpu")
            g.manual_seed(spec.seed)
            for name, param in peft_model.named_parameters():
                if "lora_A" in name or "lora_B" in name:
                    with torch.no_grad():
                        param.normal_(mean=0.0, std=spec.init_std, generator=g)

            peft_model.save_pretrained(target_path, safe_serialization=True)
            return target_path
        except Exception as peft_exc:
            selected_linear_modules: list[tuple[str, torch.nn.Linear]] = []
            for full_name, module in model.named_modules():
                if not isinstance(module, torch.nn.Linear):
                    continue
                if full_name.split(".")[-1] in target_modules:
                    selected_linear_modules.append((full_name, module))
            if selected_linear_modules:
                linear_specs = [
                    (full_name, module.out_features, module.in_features)
                    for full_name, module in selected_linear_modules
                ]
                print(
                    "[synthetic_lora_builder] fallback manual adapter generation was used "
                    f"because peft import/setup failed: {peft_exc}"
                )
                return _write_manual_adapter_from_specs(
                    base_model=base_model,
                    target_path=target_path,
                    spec=spec,
                    target_modules=target_modules,
                    linear_specs=linear_specs,
                )

    config = AutoConfig.from_pretrained(base_model, trust_remote_code=trust_remote_code)
    target_modules = _infer_target_modules_from_config(config)
    linear_specs = _build_config_linear_specs(config)
    if model_exc is not None:
        print(
            "[synthetic_lora_builder] config-only adapter generation was used "
            f"because model loading failed: {model_exc}"
        )
    return _write_manual_adapter_from_specs(
        base_model=base_model,
        target_path=target_path,
        spec=spec,
        target_modules=target_modules,
        linear_specs=linear_specs,
    )


def build_synthetic_adapters(
    *,
    base_model: str,
    out_dir: str,
    specs: Iterable[AdapterSpec],
    trust_remote_code: bool = False,
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Create synthetic LoRA adapters for local validation.")
    parser.add_argument("--base-model", required=True, help="Base model id/path (non-LLaMA allowed).")
    parser.add_argument("--out-dir", required=True, help="Output directory for generated adapters.")
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()

    specs = [
        AdapterSpec(
            name=spec.name,
            rank=spec.rank,
            alpha=spec.alpha,
            seed=spec.seed,
            init_std=spec.init_std,
        )
        for spec in DEFAULT_SYNTHETIC_ADAPTER_PRESETS
    ]
    paths = build_synthetic_adapters(
        base_model=args.base_model,
        out_dir=args.out_dir,
        specs=specs,
        trust_remote_code=args.trust_remote_code,
    )
    print("Synthetic LoRA adapters generated:")
    for p in paths:
        print(p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
