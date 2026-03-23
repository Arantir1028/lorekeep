# lorekeep/engine/vllm_hijacker.py

import time
import logging
import collections
from typing import List, Tuple

from vllm.core.scheduler import Scheduler, SchedulerOutputs
from vllm.worker.model_runner import ModelRunner
from scheduler.wave_scheduler import WaveScheduler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("WaveSlice")

# 1. 保存所有底层执行引擎的方法引用
_ORIGINAL_SCHEDULE = Scheduler._schedule
_ORIGINAL_SCHEDULE_RUNNING = Scheduler._schedule_running
_ORIGINAL_EXECUTE_MODEL = ModelRunner.execute_model
_WAVE_BRAIN = None

def _wave_schedule_running_hook(self, running_queue, budget, *args, **kwargs):
    """
    【二级劫持】：精准掐断 Running 队列中长任务的贪婪吞噬
    核心修复：严格匹配 vLLM v0.4.3 的 API 签名 (self, running_queue, budget, ...)
    """
    if getattr(self, "wave_running_budget_limit", None) is not None:
        original_token_budget = budget.token_budget
        # 核心：强制将 Running 队列的可用预算缩减至 (短任务所需 + 长任务最佳切片 S_c)
        budget.token_budget = min(original_token_budget, self.wave_running_budget_limit)
        
        # 带着被缩减的预算，放行去调度 Running 队列
        outputs = _ORIGINAL_SCHEDULE_RUNNING(self, running_queue, budget, *args, **kwargs)
        
        # 恢复真实预算，把被拦截下来的空余配额完美让渡给后续的 Waiting 队列短任务
        budget.token_budget = original_token_budget
        return outputs
    else:
        return _ORIGINAL_SCHEDULE_RUNNING(self, running_queue, budget, *args, **kwargs)

def _wave_schedule_hook(self: Scheduler) -> Tuple[List[SchedulerOutputs], bool]:
    global _WAVE_BRAIN
    
    waiting_queue = self.waiting
    running_queue = self.running

    # 1. 提取全生命周期的剩余 Token 长度
    all_seqs = []
    max_wait_time_us = 0.0
    current_time = time.time()
    
    for sg in waiting_queue:
        seq = next(iter(sg.get_seqs()))
        l = seq.get_len() - seq.data.get_num_computed_tokens()
        if l > 0:
            all_seqs.append(l)
            wait_time_us = (current_time - sg.metrics.arrival_time) * 1e6
            if wait_time_us > max_wait_time_us: max_wait_time_us = wait_time_us
            
    for sg in running_queue:
        seq = next(iter(sg.get_seqs()))
        l = seq.get_len() - seq.data.get_num_computed_tokens()
        if l > 0:
            all_seqs.append(l)

    if len(all_seqs) < 2:
        return _ORIGINAL_SCHEDULE(self)

    S_s_real = min(all_seqs)
    S_l_real = max(all_seqs)

    if S_l_real < S_s_real * 4 or S_l_real <= 512:
        return _ORIGINAL_SCHEDULE(self)

    # 2. O(1) 物理大脑决断
    total_reqs = len(waiting_queue) + len(running_queue)
    rho_est = total_reqs / (total_reqs + 1.0)
    best_S_c = _WAVE_BRAIN.schedule_real(S_s_real, S_l_real, max_wait_time_us, rho_est)

    if best_S_c >= S_l_real:
        return _ORIGINAL_SCHEDULE(self)

    # 3. 双队列重排 (SJF): 确保无论是 waiting 还是 running 队列，短任务永远顶在最前面
    waiting_short, waiting_long = [], []
    waiting_short_tokens = 0
    for sg in self.waiting:
        seq = next(iter(sg.get_seqs()))
        l = seq.get_len() - seq.data.get_num_computed_tokens()
        if l <= best_S_c:
            waiting_short.append(sg)
            waiting_short_tokens += l
        else:
            waiting_long.append(sg)
    self.waiting = collections.deque(waiting_short + waiting_long)

    running_short, running_long = [], []
    running_short_tokens = 0
    for sg in self.running:
        seq = next(iter(sg.get_seqs()))
        l = seq.get_len() - seq.data.get_num_computed_tokens()
        if l <= best_S_c:
            running_short.append(sg)
            running_short_tokens += l
        else:
            running_long.append(sg)
    self.running = collections.deque(running_short + running_long)

    # 4. 跨界物理预算微操 (Micro-Budgeting)
    original_budget = self.scheduler_config.max_num_batched_tokens
    
    # 限制原 Running 队列最多消耗多少 (避免 Titan 贪吃)
    running_budget_limit = running_short_tokens + best_S_c
    # 扩大全局预算，将跨界调度的 waiting 短任务包含进来
    global_hijacked_budget = running_budget_limit + waiting_short_tokens
    
    self.scheduler_config.max_num_batched_tokens = global_hijacked_budget
    self.wave_running_budget_limit = running_budget_limit
    
    logger.info(f"🌊 [Wave-Slice] 决断介入: 检测到短任务排队。将长任务 {S_l_real} 切分为 {best_S_c}。")
    if waiting_short_tokens > 0:
        logger.info(f"🌊 [Wave-Slice] 物理插队: {len(waiting_short)} 个短任务跃迁至队首, 跨界并发预算锁死为 {global_hijacked_budget}")

    # 触发原生调度，原生调度器将完美按照我们的粒度组装异构 Batch
    outputs = _ORIGINAL_SCHEDULE(self)
    
    # 5. 状态擦除，做到无痕劫持
    self.scheduler_config.max_num_batched_tokens = original_budget
    self.wave_running_budget_limit = None
    
    return outputs

def _wave_execute_model_hook(self: ModelRunner, *args, **kwargs):
    model_input = args[0] if len(args) > 0 else kwargs.get("model_input")
    if model_input is not None and hasattr(model_input, "attn_metadata"):
        attn_meta = model_input.attn_metadata
        if hasattr(attn_meta, "prompt_lens_tensor") and attn_meta.prompt_lens_tensor is not None:
            real_chunk_sizes = attn_meta.prompt_lens_tensor.tolist()
            if len(real_chunk_sizes) > 1:
                pass # 已证明切分生效，可不打印日志以免刷屏
    return _ORIGINAL_EXECUTE_MODEL(self, *args, **kwargs)

def inject_wave_slice(model_name: str):
    global _WAVE_BRAIN
    _WAVE_BRAIN = WaveScheduler(model_name, gamma=2.0)
    
    Scheduler._schedule = _wave_schedule_hook
    Scheduler._schedule_running = _wave_schedule_running_hook # 注入二级拦截器
    ModelRunner.execute_model = _wave_execute_model_hook
    logger.info(f"🌊 [Wave-Slice] 针对 {model_name} 的底层双重劫持完成。")