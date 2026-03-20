# lorekeep/engine/base_slicer.py
import torch

class WaveBaseSlicer:
    def __init__(self, d_model: int):
        self.d_model = d_model

    def slice_and_pack(self, X_short: torch.Tensor, X_long: torch.Tensor, S_c: int):
        """
        在 Host 端执行零拷贝（或低拷贝）的 Token 切分与聚合。
        
        参数:
            X_short: 短任务的输入张量 [S_s, d_model]
            X_long: 长任务的完整输入张量 [S_l, d_model]
            S_c: 调度器决断的最佳切分粒度 (Chunk Size)
            
        返回:
            X_packed: 送入 Base Model 的展平 1D 张量 [S_s + S_c, d_model]
            metadata: 记录拼接边界，供 Unbinder 解绑使用
        """
        S_s = X_short.size(0)
        S_l = X_long.size(0)
        
        # 1. 物理切断：截取长任务的第一个 Chunk
        # 注意：这里是张量视图 (View) 操作，时间复杂度为 O(1)，无显存拷贝开销
        X_long_chunk = X_long[:S_c, :] 
        
        # 2. 物理聚合：构建无 Padding 的 1D 连续计算块
        # 这是为了适配 FlashAttention 的 varlen (变长) 接口，彻底消除 SM 气泡
        X_packed = torch.cat([X_short, X_long_chunk], dim=0)
        
        # 3. 构造元数据 (Metadata)
        metadata = {
            "S_s": S_s,
            "S_c": S_c,
            "S_l_residual": S_l - S_c # 记录剩余待计算的 Token 数量
        }
        
        return X_packed, metadata

# 纯逻辑验证
if __name__ == "__main__":
    slicer = WaveBaseSlicer(d_model=4096)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # 模拟输入
    X_short = torch.randn(45, 4096, device=device)
    X_long = torch.randn(2048, 4096, device=device)
    
    # 假设 Scheduler 决断 S_c = 256
    X_packed, meta = slicer.slice_and_pack(X_short, X_long, S_c=256)
    
    print(f"=== Base Slicer 物理状态 ===")
    print(f"原始短任务形状: {X_short.shape}")
    print(f"原始长任务形状: {X_long.shape}")
    print(f"输入 Base Model 的聚合形状: {X_packed.shape}")
    print(f"向后传递的元数据: {meta}")