import pandas as pd
import numpy as np

def quantify_efficiency(csv_path):
    df = pd.read_csv(csv_path)
    
    # 定义阶段切分点
    solo_phase_end = 3.50367
    
    # 划分阶段
    solo_df = df[df['gui_time_s'] < solo_phase_end]
    wave_slice_df = df[df['gui_time_s'] >= solo_phase_end]
    
    stats = {
        "Solo Phase (Baseline)": {
            "Mean Utilization": f"{solo_df['gr_active'].mean():.2f}%",
            "Stability (Std)": f"{solo_df['gr_active'].std():.2f}",
            "Min Utilization": f"{solo_df['gr_active'].min():.2f}%"
        },
        "Wave-Slice Phase (Optimized)": {
            "Mean Utilization": f"{wave_slice_df['gr_active'].mean():.2f}%",
            "Stability (Std)": f"{wave_slice_df['gr_active'].std():.2f}",
            "Min Utilization": f"{wave_slice_df['gr_active'].min():.2f}%"
        }
    }
    
    print("--- Wave-Slice 调度性能量化报告 ---")
    for phase, data in stats.items():
        print(f"\n[{phase}]")
        for k, v in data.items():
            print(f"  {k}: {v}")
            
    # 计算理论增益
    gain = (wave_slice_df['gr_active'].mean() - solo_df['gr_active'].mean()) / solo_df['gr_active'].mean()
    print(f"\n[Conclusion] 相对利用率提升: {gain*100:.2f}%")

if __name__ == "__main__":
    quantify_efficiency("wave_slice_metrics_final.csv")