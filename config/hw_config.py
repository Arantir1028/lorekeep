# lorekeep/config/hw_config.py
import os

# 候选的 Token 切分桶
# Include higher buckets for long-context serving experiments and
# bucket-limit ablations (e.g., 2048 vs 4096 ceiling).
BUCKETS = [32, 64, 128, 256, 288, 320, 352, 384, 448, 512, 768, 1024, 1536, 2048, 3072, 4096]
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
    
    # --- GQA (分组注意力：现代模型主流，读放大温和) ---
    "Mistral-7B-v0.1":{"attn_type": "GQA", "q_heads": 32, "kv_heads": 8,  "head_dim": 128, "d_model": 4096},
    "Gemma-7B":       {"attn_type": "GQA", "q_heads": 16, "kv_heads": 16, "head_dim": 256, "d_model": 3072}, # Gemma 维度较特殊，适合做鲁棒性测试
    "Qwen2-7B":       {"attn_type": "GQA", "q_heads": 28, "kv_heads": 4,  "head_dim": 128, "d_model": 3584},
    
    # --- MQA (多查询注意力：极低读放大，容易诱发过度调度的陷阱区) ---
    "Falcon-7B":      {"attn_type": "MQA", "q_heads": 71, "kv_heads": 1,  "head_dim": 64,  "d_model": 4544}
}

# 路径配置
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "lut_tables")
os.makedirs(DATA_DIR, exist_ok=True)

def get_lut_paths(model_name: str):
    """根据模型名称动态获取对应的 LUT 路径"""
    return {
        "raw": os.path.join(DATA_DIR, f"raw_profile_{model_name}.json"),
        "gain": os.path.join(DATA_DIR, f"lut_gain_{model_name}.json"),
        "penalty": os.path.join(DATA_DIR, f"lut_penalty_{model_name}.json")
    }
