# lorekeep/profiler/lut_generator.py

import json
import sys
import os
import math
import argparse

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

    T_solo = {int(k): float(v) for k, v in raw["T_solo"].items()}
    T_conc = {int(k): {int(kk): float(vv) for kk, vv in v.items()} for k, v in raw["T_conc"].items()}
    T_read_amp = {int(k): {int(kk): float(vv) for kk, vv in v.items()} for k, v in raw["T_read_amp"].items()}

    LUT_Gain = {}
    LUT_Penalty = {}

    available_rows = sorted(T_conc.keys())
    available_cols = sorted(next(iter(T_conc.values())).keys()) if T_conc else []

    def _nearest_ge_or_max(keys: list[int], target: int) -> int:
        for k in keys:
            if k >= target:
                return k
        return keys[-1]

    for s_s in cfg.BUCKETS:
        LUT_Gain[s_s] = {}
        row = _nearest_ge_or_max(available_rows, s_s) if available_rows else s_s
        solo_row = _nearest_ge_or_max(sorted(T_solo.keys()), s_s) if T_solo else s_s
        t_solo_s = float(T_solo.get(solo_row, 0.0))
        row_conc = T_conc.get(row, {})
        cols = sorted(row_conc.keys()) if row_conc else available_cols
        for s_c in cfg.BUCKETS:
            if s_c < s_s:
                LUT_Gain[s_s][s_c] = 0.0
                continue
            col = _nearest_ge_or_max(cols, s_c) if cols else s_c
            t_conc_s = float(row_conc.get(col, 0.0))
            # 收益定义修正: Gain = T_solo(short) - T_conc(short, chunked-long)
            LUT_Gain[s_s][s_c] = max(0.0, t_solo_s - t_conc_s)

    for s_l in cfg.BUCKETS:
        LUT_Penalty[s_l] = {}
        row_l = _nearest_ge_or_max(sorted(T_read_amp.keys()), s_l) if T_read_amp else s_l
        for s_c in cfg.BUCKETS:
            if s_c >= s_l:
                LUT_Penalty[s_l][s_c] = 0.0
            else:
                # 争用开销：保守假设短任务 S_s 最大为 S_c
                row_c = _nearest_ge_or_max(available_rows, s_c) if available_rows else s_c
                row_conc = T_conc.get(row_c, {})
                cols_conc = sorted(row_conc.keys()) if row_conc else available_cols
                col_c = _nearest_ge_or_max(cols_conc, s_c) if cols_conc else s_c
                t_conc_cc = float(row_conc.get(col_c, 0.0))
                t_solo_c = float(T_solo.get(_nearest_ge_or_max(sorted(T_solo.keys()), s_c), 0.0))
                c_contention = max(0.0, t_conc_cc - t_solo_c)

                read_row = T_read_amp.get(row_l, {})
                read_cols = sorted(read_row.keys()) if read_row else []
                col_r = _nearest_ge_or_max(read_cols, s_c) if read_cols else s_c
                c_read_amp = float(read_row.get(col_r, 0.0))
                
                # 接入学术严谨的动态惩罚函数
                LUT_Penalty[s_l][s_c] = _calculate_dynamic_penalty(s_l, s_c, c_contention, c_read_amp)

    with open(paths["gain"], "w") as f:
        json.dump(LUT_Gain, f, indent=4)
    with open(paths["penalty"], "w") as f:
        json.dump(LUT_Penalty, f, indent=4)

    print(f"✅ [{model_name}] O(1) LUT 表已生成完毕。")

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Wave-Slice LUTs from raw profiles.")
    parser.add_argument(
        "--models",
        default="all",
        help="Comma-separated model keys in config/hw_config.py, or 'all'.",
    )
    parser.add_argument(
        "--buckets",
        default=None,
        help="Optional comma-separated buckets override.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.buckets:
        cfg.BUCKETS = [int(x.strip()) for x in args.buckets.split(",") if x.strip()]
        cfg.BUCKETS = sorted({b for b in cfg.BUCKETS if b > 0})
        if not cfg.BUCKETS:
            raise ValueError("Invalid --buckets")

    models = list(cfg.SUPPORTED_MODELS.keys())
    if args.models.strip().lower() != "all":
        models = [m.strip() for m in args.models.split(",") if m.strip()]
    print("=== 生成 O(1) LUT 表 ===")
    print(f"models={models}")
    print(f"buckets={cfg.BUCKETS}")
    for model in models:
        generate_lut_for_model(model)
