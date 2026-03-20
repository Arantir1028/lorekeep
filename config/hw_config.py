# lorekeep/config/hw_config.py

import os

# ---------------------------------------------------------
# 系统离散状态空间 (Hardware-Aware Bucketization)
# ---------------------------------------------------------
# 候选的 Token 切分桶 (从 32 到 2048)
BUCKETS = [32, 64, 128, 256, 512, 1024, 2048]

# ---------------------------------------------------------
# 物理硬件与模型超参数 (以 Llama-7B 级别的单层 Attention 为例)
# ---------------------------------------------------------
BATCH_SIZE = 128
NUM_HEADS = 32
HEAD_DIM = 128
D_MODEL = NUM_HEADS * HEAD_DIM

# 路径配置
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "lut_tables")
RAW_PROFILE_PATH = os.path.join(DATA_DIR, "raw_profile.json")
LUT_GAIN_PATH = os.path.join(DATA_DIR, "lut_gain.json")
LUT_PENALTY_PATH = os.path.join(DATA_DIR, "lut_penalty.json")

# 确保数据目录存在
os.makedirs(DATA_DIR, exist_ok=True)