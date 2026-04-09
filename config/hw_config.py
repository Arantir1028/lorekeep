# lorekeep/config/hw_config.py
import json
import os
import re

# 候选的 Token 切分桶
# Include higher buckets for long-context serving experiments and
# bucket-limit ablations (e.g., 2048 vs 4096 ceiling).
BUCKETS = [256, 288, 320, 352, 384, 448, 512, 768, 1024, 1536, 2048, 3072, 4096]
BATCH_SIZE = 128
LORA_RANK = 32

# ---------------------------------------------------------
# 多模型异构架构矩阵 (The Evaluation Triad)
# ---------------------------------------------------------
# ---------------------------------------------------------
# 多模型异构架构矩阵 (完全规避 LLaMA 版权，符合 Apache/MIT 协议)
# ---------------------------------------------------------
SUPPORTED_MODELS = {
    # --- MHA (多头注意力：读放大灾难区) ---
    "Qwen1.5-7B":     {"attn_type": "MHA", "q_heads": 32, "kv_heads": 32, "head_dim": 128, "d_model": 4096},
    "BLOOM-7B":       {"attn_type": "MHA", "q_heads": 32, "kv_heads": 32, "head_dim": 128, "d_model": 4096},
    "Phi-2":          {"attn_type": "MHA", "q_heads": 32, "kv_heads": 32, "head_dim": 80,  "d_model": 2560},
    "Baichuan2-7B-Chat": {"attn_type": "MHA", "q_heads": 32, "kv_heads": 32, "head_dim": 128, "d_model": 4096},
    "DeciLM-7B":      {"attn_type": "MHA", "q_heads": 32, "kv_heads": 32, "head_dim": 128, "d_model": 4096},
    
    # --- GQA (分组注意力：现代模型主流，读放大温和) ---
    "Mistral-7B-v0.1":{"attn_type": "GQA", "q_heads": 32, "kv_heads": 8,  "head_dim": 128, "d_model": 4096},
    "Qwen2-7B":       {"attn_type": "GQA", "q_heads": 28, "kv_heads": 4,  "head_dim": 128, "d_model": 3584},
    
    # --- Other architectures / robustness probes ---
    "Gemma-7B":       {"attn_type": "MHA", "q_heads": 16, "kv_heads": 16, "head_dim": 192, "d_model": 3072},

    # --- MQA (多查询注意力：极低读放大，容易诱发过度调度的陷阱区) ---
    "Falcon-7B":      {"attn_type": "MQA", "q_heads": 71, "kv_heads": 1,  "head_dim": 64,  "d_model": 4544}
}

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "lut_tables")
os.makedirs(DATA_DIR, exist_ok=True)
CHECKPOINT_REGISTRY_PATH = os.path.join(DATA_DIR, "checkpoint_registry.json")

MODEL_ALIASES = {
    "qwen2.5-7b-instruct": "Qwen2-7B",
    "qwen2.5-7b": "Qwen2-7B",
    "qwen2-7b-instruct": "Qwen2-7B",
    "falcon-7b-instruct": "Falcon-7B",
    "mistral-7b-v0.1": "Mistral-7B-v0.1",
    "gemma-7b-it": "Gemma-7B",
}


def checkpoint_lut_name(model_id: str) -> str:
    return model_id.replace("/", "--")


def _normalize_model_key(model_name: str) -> str:
    return re.sub(r"[^a-z0-9.-]+", "-", model_name.strip().lower().replace("_", "-")).strip("-")


def register_checkpoint_model(
    lut_name: str,
    *,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
    d_model: int,
    aliases: list[str] | None = None,
) -> None:
    if kv_heads <= 0 or q_heads <= 0:
        raise ValueError(f"invalid heads for {lut_name}: q={q_heads} kv={kv_heads}")
    if kv_heads == 1:
        attn_type = "MQA"
    elif kv_heads == q_heads:
        attn_type = "MHA"
    else:
        attn_type = "GQA"
    SUPPORTED_MODELS[lut_name] = {
        "attn_type": attn_type,
        "q_heads": int(q_heads),
        "kv_heads": int(kv_heads),
        "head_dim": int(head_dim),
        "d_model": int(d_model),
    }
    for alias in aliases or []:
        MODEL_ALIASES[_normalize_model_key(alias)] = lut_name


def _load_checkpoint_registry() -> None:
    if not os.path.exists(CHECKPOINT_REGISTRY_PATH):
        return
    try:
        with open(CHECKPOINT_REGISTRY_PATH, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return

    entries = payload.get("models", payload)
    if isinstance(entries, dict):
        entries = list(entries.values())
    if not isinstance(entries, list):
        return

    for row in entries:
        if not isinstance(row, dict):
            continue
        lut_name = str(row.get("lut_name") or "").strip()
        q_heads = row.get("q_heads")
        kv_heads = row.get("kv_heads")
        head_dim = row.get("head_dim")
        d_model = row.get("d_model")
        if not lut_name or not all(v is not None for v in (q_heads, kv_heads, head_dim, d_model)):
            continue
        try:
            register_checkpoint_model(
                lut_name,
                q_heads=int(q_heads),
                kv_heads=int(kv_heads),
                head_dim=int(head_dim),
                d_model=int(d_model),
                aliases=list(row.get("aliases") or []),
            )
        except Exception:
            continue


def resolve_model_name(model_name: str) -> str:
    if model_name in SUPPORTED_MODELS:
        return model_name
    normalized = _normalize_model_key(model_name.split("/")[-1])
    if normalized in MODEL_ALIASES:
        return MODEL_ALIASES[normalized]
    full_normalized = _normalize_model_key(model_name)
    if full_normalized in MODEL_ALIASES:
        return MODEL_ALIASES[full_normalized]
    for supported in SUPPORTED_MODELS.keys():
        supported_norm = _normalize_model_key(supported)
        if supported_norm in normalized or normalized in supported_norm:
            return supported
    return model_name


_load_checkpoint_registry()

def get_lut_paths(model_name: str):
    """根据模型名称动态获取对应的 LUT 路径"""
    return {
        "raw": os.path.join(DATA_DIR, f"raw_profile_{model_name}.json"),
        "gain": os.path.join(DATA_DIR, f"lut_gain_{model_name}.json"),
        "penalty": os.path.join(DATA_DIR, f"lut_penalty_{model_name}.json")
    }
