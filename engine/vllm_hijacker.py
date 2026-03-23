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

_ORIGINAL_SCHEDULE = Scheduler._schedule
_ORIGINAL_SCHEDULE_RUNNING = Scheduler._schedule_running
_ORIGINAL_EXECUTE_MODEL = ModelRunner.execute_model
_WAVE_BRAIN = None

def _wave_schedule_running_hook(self, running_queue, budget, *args, **kwargs):
    """
    【二级劫持】：精准掐断 Running 队列中长任务的贪婪吞噬
    """
    if getattr(self, "wave_running_budget_limit", None) is not None:
        original_token_budget = budget.token_budget
        # 将 Running 队列的可用预算缩减，逼迫引擎释放剩余算力
        budget.token_budget = min(original_token_budget, self.wave_running_budget_limit)
        outputs = _ORIGINAL_SCHEDULE_RUNNING(self, running_queue, budget, *args, **kwargs)
        # 恢复预算，让渡给后面的 Waiting 短任务
        budget.token_budget = original_token_budget
        return outputs
    else:
        return _ORIGINAL_SCHEDULE_RUNNING(self, running_queue, budget, *args, **kwargs)

def _wave_schedule_hook(self: Scheduler) -> Tuple[List[SchedulerOutputs], bool]:
    global _WAVE_BRAIN
    
    waiting_queue = self.waiting
    running_queue = self.running

    # 1. 提取剩余 Token 长度
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

    # 2. 动态大脑决断
    total_reqs = len(waiting_queue) + len(running_queue)
    rho_est = total_reqs / (total_reqs + 1.0)
    best_S_c = _WAVE_BRAIN.schedule_real(S_s_real, S_l_real, max_wait_time_us, rho_est)

    if best_S_c >= S_l_real:
        return _ORIGINAL_SCHEDULE(self)

    # =====================================================================
    # 核心修复区：保护 Running 队列，只重排 Waiting 队列
    # =====================================================================
    # 只针对 Waiting 队列进行 SJF 置顶，确保短任务第一时间挤入调度
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

    # 绝对禁止重排 self.running 队列！保持长任务的 RoPE 连续性！
    
    # 动态计算 Running 队列的合理预算上限
    running_budget_limit = 0
    for sg in self.running:
        seq = next(iter(sg.get_seqs()))
        l = seq.get_len() - seq.data.get_num_computed_tokens()
        # 让每个在运行的任务最多只能吃掉 best_S_c 的份额
        running_budget_limit += min(l, best_S_c)

    original_budget = self.scheduler_config.max_num_batched_tokens
    global_hijacked_budget = running_budget_limit + waiting_short_tokens
    
    self.scheduler_config.max_num_batched_tokens = global_hijacked_budget
    self.wave_running_budget_limit = running_budget_limit
    
    logger.info(f"🌊 [Wave-Slice] 决断介入: 检测到短任务排队。将长任务 {S_l_real} 切分为 {best_S_c}。")
    if waiting_short_tokens > 0:
        logger.info(f"🌊 [Wave-Slice] 物理插队: {len(waiting_short)} 个短任务跃迁至 Waiting 队首, 跨界并发预算锁死为 {global_hijacked_budget}")

    outputs = _ORIGINAL_SCHEDULE(self)
    
    self.scheduler_config.max_num_batched_tokens = original_budget
    self.wave_running_budget_limit = None
    
    return outputs

def _wave_execute_model_hook(self: ModelRunner, *args, **kwargs):
    return _ORIGINAL_EXECUTE_MODEL(self, *args, **kwargs)

def inject_wave_slice(model_name: str):
    global _WAVE_BRAIN
    _WAVE_BRAIN = WaveScheduler(model_name, gamma=2.0)
    Scheduler._schedule = _wave_schedule_hook
    Scheduler._schedule_running = _wave_schedule_running_hook 
    ModelRunner.execute_model = _wave_execute_model_hook
    logger.info(f"🌊 [Wave-Slice] 针对 {model_name} 的底层安全劫持完成。")