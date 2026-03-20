import torch
import triton
import triton.language as tl

# =====================================================================
# 1. 核心算子：零补零的异构工作负载模拟 (Zero-Padding Heterogeneous Kernel)
# 逻辑：每个 Program ID (Block) 负责一个请求。它读取自己专属的 seq_len，
# 执行精确匹配该长度的计算量。彻底消灭 Padding。
# =====================================================================
@triton.jit
def simulate_heterogeneous_kernel(
    seq_lens_ptr,    # 指向每个请求序列长度的指针
    out_ptr,         # 输出指针，防止计算被编译器优化掉
    d_model: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    # 获取当前 Thread Block 负责的 Request ID
    pid = tl.program_id(0)
    
    # 动态加载当前请求的真实序列长度 (短任务加载 32，长任务加载 2048)
    actual_seq_len = tl.load(seq_lens_ptr + pid)

    # 模拟纯净的 Compute-bound 负载：执行时间严格正比于 actual_seq_len
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for i in range(actual_seq_len):
        # 增加内部循环以模拟 d_model 维度的矩阵乘法耗时，确保在 nsys 中清晰可见
        for j in range(d_model // 64): 
            acc += 0.001
            
    # 写回显存，完成当前 Block 的生命周期
    tl.store(out_ptr + (pid * BLOCK_SIZE) + tl.arange(0, BLOCK_SIZE), acc)


# =====================================================================
# 2. 实验编排与 NVTX 埋点
# =====================================================================
def run_nsys_profiling():
    d_model = 4096
    S_short, S_long = 32, 2048
    device = torch.device("cuda:0")

    # 分配内存
    out_buffer = torch.empty((2 * 1024,), device=device, dtype=torch.float32)
    
    # 构造精确的长度描述符
    len_short_only = torch.tensor([S_short], device=device, dtype=torch.int32)
    len_long_only = torch.tensor([S_long], device=device, dtype=torch.int32)
    len_hetero_grouped = torch.tensor([S_short, S_long], device=device, dtype=torch.int32) # [短任务, 长任务]

    # 常量配置
    BLOCK_SIZE = 1024

    print("=== 正在生成 Nsys Profiling 轨迹 (Zero-Padding) ===")

    # 0. 预热阶段 (消除编译和初始化抖动)
    torch.cuda.nvtx.range_push("Warmup")
    for _ in range(5):
        simulate_heterogeneous_kernel[(2,)](len_hetero_grouped, out_buffer, d_model, BLOCK_SIZE)
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()

    # 1. 对照组：短任务独占执行 (Grid Size = 1)
    torch.cuda.nvtx.range_push("1_Solo_Short_Ideal_SLA")
    simulate_heterogeneous_kernel[(1,)](len_short_only, out_buffer, d_model, BLOCK_SIZE)
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()

    # 2. 对照组：长任务独占执行 (Grid Size = 1)
    torch.cuda.nvtx.range_push("2_Solo_Long_Baseline")
    simulate_heterogeneous_kernel[(1,)](len_long_only, out_buffer, d_model, BLOCK_SIZE)
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()

    # 3. 实验组：真正的 Grouped GEMM 屏障 (Grid Size = 2)
    # 物理现象：Grid 0 执行短任务，Grid 1 执行长任务。两者在同一个 Kernel 生命周期内。
    torch.cuda.nvtx.range_push("3_Grouped_ZeroPadding_Barrier")
    simulate_heterogeneous_kernel[(2,)](len_hetero_grouped, out_buffer, d_model, BLOCK_SIZE)
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()

    print("执行完毕。请使用 Nsight Systems 查看生成的 .nsys-rep 文件。")

if __name__ == "__main__":
    run_nsys_profiling()