# lorekeep/engine/vllm_hijacker.py

import time
import logging
from typing import List, Tuple

# 引入 vLLM 核心类
from vllm.core.scheduler import Scheduler, SchedulerOutputs
from vllm.worker.model_runner import ModelRunner

from scheduler.wave_scheduler import WaveScheduler

# 屏蔽繁杂日志，仅保留 Wave-Slice 关键输出
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("WaveSlice")

# 1. 保存底层原生方法的内存引用
_ORIGINAL_SCHEDULE = Scheduler._schedule
_ORIGINAL_EXECUTE_MODEL = ModelRunner.execute_model

_WAVE_BRAIN = None

def _wave_schedule_hook(self: Scheduler) -> Tuple[List[SchedulerOutputs], bool]:
    global _WAVE_BRAIN
    
    waiting_queue = self.waiting
    running_queue = self.running

    # 无排队压力时，放行原生调度器，零损耗
    if not waiting_queue or len(waiting_queue) == 0:
        return _ORIGINAL_SCHEDULE(self)

    current_time = time.time()
    
    # ==========================================================
    # 状态翻译：从 vLLM 内存树中提取真实的物理特征
    # ==========================================================
    seq_lens = []
    max_wait_time_us = 0.0
    
    for seq_group in waiting_queue:
        # vLLM 的 SequenceGroup 包含了所有的 Token 数据
        seq = next(iter(seq_group.get_seqs()))
        seq_len = seq.get_len()
        seq_lens.append(seq_len)
        
        # 提取真实到达时间，转换为微秒 (us) 供 FairnessEngine 计算
        wait_time_us = (current_time - seq_group.metrics.arrival_time) * 1e6
        if wait_time_us > max_wait_time_us:
            max_wait_time_us = wait_time_us

    S_s_real = min(seq_lens)
    S_l_real = max(seq_lens)

    # 边界条件判定：如果没有异构（长短任务差距小于 4 倍）或长任务本身够短，则不触发切分
    if S_l_real < S_s_real * 4 or S_l_real <= 512:
        return _ORIGINAL_SCHEDULE(self)

    # ==========================================================
    # 排队论映射与 O(1) 决断
    # ==========================================================
    total_reqs = len(waiting_queue) + len(running_queue)
    rho_est = total_reqs / (total_reqs + 1.0)

    best_S_c = _WAVE_BRAIN.schedule_real(S_s_real, S_l_real, max_wait_time_us, rho_est)

    # ==========================================================
    # 物理劫持：动态修改 vLLM 原生的 Chunk 预算
    # ==========================================================
    original_budget = self.scheduler_config.max_num_batched_tokens
    
    # 核心逻辑：预算必须覆盖 "最优切分块 S_c" + "当前准备插入的短任务 Token 总量"
    # 如果仅设置为 S_c，vLLM 会只调度长任务切片而抛弃短任务，彻底丧失并发意义
    short_tasks_tokens = sum(l for l in seq_lens if l < best_S_c)
    hijacked_budget = best_S_c + short_tasks_tokens
    
    self.scheduler_config.max_num_batched_tokens = hijacked_budget

    # 带着被篡改的全局预算，放行原生调度器生成执行图
    outputs = _ORIGINAL_SCHEDULE(self)

    # 状态恢复：擦除劫持痕迹，避免污染系统的静态配置
    self.scheduler_config.max_num_batched_tokens = original_budget

    return outputs

def _wave_execute_model_hook(self: ModelRunner, *args, **kwargs):
    """
    执行层探针钩子：截获发往 CUDA 的计算图元数据。
    此处仅做真实物理拼装的日志验证，不破坏 vLLM 原生的 PagedAttention C++ 调用链路。
    """
    model_input = args[0] if len(args) > 0 else kwargs.get("model_input")
    
    if model_input is not None and hasattr(model_input, "attn_metadata"):
        attn_meta = model_input.attn_metadata
        # 提取当前 Batch 内部真实的连续 Token 分布块
        if hasattr(attn_meta, "prompt_lens_tensor") and attn_meta.prompt_lens_tensor is not None:
            real_chunk_sizes = attn_meta.prompt_lens_tensor.tolist()
            if len(real_chunk_sizes) > 1:
                logger.info(f"🌊 [Wave-Slice] 底层物理拦截: 侦测到被切分的异构 Batch -> 长度分布 {real_chunk_sizes}")

    return _ORIGINAL_EXECUTE_MODEL(self, *args, **kwargs)

def inject_wave_slice(model_name: str):
    """全局注入入口"""
    global _WAVE_BRAIN
    # 实例化决断大脑
    _WAVE_BRAIN = WaveScheduler(model_name, gamma=2.0)
    
    Scheduler._schedule = _wave_schedule_hook
    ModelRunner.execute_model = _wave_execute_model_hook
    logger.info(f"🌊 [Wave-Slice] 针对 {model_name} 的底层劫持完成。进入异构感知时空调度模式。")