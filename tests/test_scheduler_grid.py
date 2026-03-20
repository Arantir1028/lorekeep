import sys
import os

# 确保能找到同级模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scheduler.wave_scheduler import WaveScheduler

def run_decision_matrix_test():
    print("=== Wave-Slice Scheduler 决策矩阵边缘测试 ===")
    
    # 实例化大脑: 惩罚放大因子 gamma=0.5, 最大队列深度=100
    scheduler = WaveScheduler(gamma=0.5, max_queue_depth=100)
    
    # 锁定一组异构请求
    S_s = 45       # 短任务 Token 数
    S_l = 2048     # 长任务 Token 数
    t_solo_s = 50.0 # 短任务理想独占时间 50us
    
    # 扫描变量 1: 短任务已经等待的时间 (衡量饥饿度)
    wait_times_us = [0.0, 50.0, 100.0, 200.0, 500.0]
    
    # 扫描变量 2: 当前系统队列深度 (衡量负载饱和度)
    queue_depths = [0, 25, 50, 75, 100]
    
    print(f"\n物理设定: 短任务 {S_s} Tokens (T_solo={t_solo_s}us) vs 长任务 {S_l} Tokens")
    print("输出指标: 调度器决断的最佳切分粒度 S_c (若为 2048 则代表拒绝切分)")
    print("-" * 75)
    
    # 打印表头
    header = "{:<35}".format('Queue Depth (负载) \\ T_wait (饥饿)')
    for tw in wait_times_us:
        header += f"| {tw:>6}us "
    print(header)
    print("-" * 75)
    
    # 生成二维决策矩阵
    for qd in queue_depths:
        row_str = f"QD={qd:<3} (rho={qd/100:>4.2f})".ljust(35)
        for tw in wait_times_us:
            # 核心调用：O(1) 决断
            best_Sc = scheduler.schedule(
                S_s=S_s, 
                S_l=S_l, 
                t_wait_s_us=tw, 
                t_solo_s_us=t_solo_s, 
                current_queue_depth=qd
            )
            
            # 格式化输出 (高亮熔断状态)
            if best_Sc == S_l:
                row_str += f"| \033[91m{best_Sc:>6}\033[0m " # 红色表示拒绝切分(Grouped GEMM)
            else:
                row_str += f"| \033[92m{best_Sc:>6}\033[0m " # 绿色表示执行切分
                
        print(row_str)
    print("-" * 75)
    print("注: 红色 2048 代表触发弹性熔断保吞吐；绿色数值代表执行切分解救短任务。")

if __name__ == "__main__":
    run_decision_matrix_test()