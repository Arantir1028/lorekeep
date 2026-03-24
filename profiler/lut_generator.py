# lorekeep/profiler/lut_generator.py

import json
import sys
import os
import math

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import hw_config as cfg

BASE_LAUNCH_US = 5.0 # 单次底层 CUDA Graph/Kernel Launch 基础物理开销

def _calculate_dynamic_penalty(S_l: int, S_c: int, raw_contention: float, raw_read_amp: float) -> float:
    """
    计算消除固定常数的动态系统惩罚。
    包含：底层总线争用 + L2 Cache 读放大 + 动态算子发射开销
    """
    if S_c >= S_l:
        return 0.0
        
    # 物理事实：长任务被切得越碎，需要下发算子的次数就越多
    num_chunks = math.ceil(S_l / S_c)
    
    # 动态发射开销 = 基础发射耗时 * 额外产生的算子块数
    dynamic_launch_overhead = BASE_LAUNCH_US * (num_chunks - 1)
    
    # 总物理惩罚
    total_penalty = raw_contention + raw_read_amp + dynamic_launch_overhead
    
    return total_penalty

def generate_lut_for_model(model_name: str):
    paths = cfg.get_lut_paths(model_name)
    
    if not os.path.exists(paths["raw"]):
        print(f"⚠️ 跳过 {model_name}: 找不到 raw_profile，请先运行 offline_profiler.py")
        return
        
    with open(paths["raw"], "r") as f:
        raw = json.load(f)

    T_solo = {int(k): v for k, v in raw["T_solo"].items()}
    T_conc = {int(k): {int(kk): vv for kk, vv in v.items()} for k, v in raw["T_conc"].items()}
    T_read_amp = {int(k): {int(kk): vv for kk, vv in v.items()} for k, v in raw["T_read_amp"].items()}

    LUT_Gain = {}
    LUT_Penalty = {}

    for s_s in cfg.BUCKETS:
        LUT_Gain[s_s] = {}
        for s_c in cfg.BUCKETS:
            # 收益 = 短任务的独立逃逸时间 (直接读取 Profiler 修正后的 T_conc_s)
            LUT_Gain[s_s][s_c] = T_conc[s_s][s_c] if s_c >= s_s else float('inf')

    for s_l in cfg.BUCKETS:
        LUT_Penalty[s_l] = {}
        for s_c in cfg.BUCKETS:
            if s_c >= s_l:
                LUT_Penalty[s_l][s_c] = 0.0
            else:
                # 争用开销：保守假设短任务 S_s 最大为 S_c
                c_contention = max(0, T_conc[s_c][s_c] - T_solo[s_c]) if s_c in T_conc[s_c] else 0
                c_read_amp = T_read_amp[s_l][s_c]
                
                # 接入学术严谨的动态惩罚函数
                LUT_Penalty[s_l][s_c] = _calculate_dynamic_penalty(s_l, s_c, c_contention, c_read_amp)

    with open(paths["gain"], "w") as f:
        json.dump(LUT_Gain, f, indent=4)
    with open(paths["penalty"], "w") as f:
        json.dump(LUT_Penalty, f, indent=4)

    print(f"✅ [{model_name}] O(1) LUT 表已生成完毕。")

if __name__ == "__main__":
    print("=== 批量生成多架构 O(1) 决断表 ===")
    for model in cfg.SUPPORTED_MODELS.keys():
        generate_lut_for_model(model)