# lorekeep/profiler/lut_generator.py

import json
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import hw_config as cfg

def generate_lut():
    print("=== Generating O(1) Lookup Tables (LUT) ===")
    
    if not os.path.exists(cfg.RAW_PROFILE_PATH):
        raise FileNotFoundError(f"Raw profile not found at {cfg.RAW_PROFILE_PATH}. Run offline_profiler.py first.")
        
    with open(cfg.RAW_PROFILE_PATH, "r") as f:
        raw = json.load(f)

    T_solo = {int(k): v for k, v in raw["T_solo"].items()}
    T_conc = {int(k): {int(kk): vv for kk, vv in v.items()} for k, v in raw["T_conc"].items()}
    T_read_amp = {int(k): {int(kk): vv for kk, vv in v.items()} for k, v in raw["T_read_amp"].items()}

    LUT_Gain = {}    # 维度: [S_s][S_c] (近似收益，假定基线被打破)
    LUT_Penalty = {} # 维度: [S_l][S_c]

    # 常数开销 (驱动层 Launch 开销预估，单位：微秒)
    C_LAUNCH_US = 5.0 

    # ---------------------------------------------------------
    # 1. 计算 LUT_Gain: ΔU = T_solo(S_c) - Contention_Cost
    # 逻辑简化：由于 T_grp(S_s, S_l) 在运行时动态变化，
    # 查表时我们记录纯粹的“短任务相对其自身 Chunk 的释放收益”。
    # 真实 ΔU_online = T_solo(S_l) - T_conc(S_s, S_c)
    # 我们将 T_conc 存入 LUT_Gain，让引擎在线做减法。
    # ---------------------------------------------------------
    for s_s in cfg.BUCKETS:
        LUT_Gain[s_s] = {}
        for s_c in cfg.BUCKETS:
            if s_c >= s_s:
                LUT_Gain[s_s][s_c] = T_conc[s_s][s_c]
            else:
                # 非法状态 (短任务大于 Chunk 无意义)
                LUT_Gain[s_s][s_c] = float('inf') 

    # ---------------------------------------------------------
    # 2. 计算 LUT_Penalty: T_penalty = C_contention + C_read_amp + C_launch
    # ---------------------------------------------------------
    for s_l in cfg.BUCKETS:
        LUT_Penalty[s_l] = {}
        for s_c in cfg.BUCKETS:
            if s_c >= s_l:
                # 不切分，惩罚为 0
                LUT_Penalty[s_l][s_c] = 0.0
            else:
                k = s_l // s_c + (1 if s_l % s_c != 0 else 0)
                
                # 争用开销：长任务的第一个 Chunk 被短任务拖慢的时间
                # 极端保守估计：假设短任务的 S_s 也是 S_c (最大争用)
                c_contention = T_conc[s_c][s_c] - T_solo[s_c] if s_c in T_conc[s_c] else 0
                c_contention = max(0, c_contention) # 防止由于测量抖动出现负数
                
                c_read_amp = T_read_amp[s_l][s_c]
                c_launch = (k - 1) * C_LAUNCH_US
                
                total_penalty = c_contention + c_read_amp + c_launch
                LUT_Penalty[s_l][s_c] = total_penalty

    # 写入最终的查表文件
    with open(cfg.LUT_GAIN_PATH, "w") as f:
        json.dump(LUT_Gain, f, indent=4)
        
    with open(cfg.LUT_PENALTY_PATH, "w") as f:
        json.dump(LUT_Penalty, f, indent=4)

    print(f"✅ LUT_Gain generated. Dim: {len(cfg.BUCKETS)}x{len(cfg.BUCKETS)}.")
    print(f"✅ LUT_Penalty generated. Dim: {len(cfg.BUCKETS)}x{len(cfg.BUCKETS)}.")
    print("System is ready for O(1) Online Scheduling!")

if __name__ == "__main__":
    generate_lut()