"""Wave-Slice decision engine backed by LUTs."""

import json
import os
from typing import Any, Optional

from config import hw_config as cfg
from scheduler.fairness_engine import FairnessEngine

class WaveScheduler:
    def __init__(
        self,
        model_name: str = "Mistral-7B-v0.1",
        gamma: float = 2.0,
        objective_mode: str = "fair_escape",
        **_: Any,
    ):
        self.model_name = self._resolve_model_name(model_name)
        self.gamma = gamma
        self.objective_mode = objective_mode
        self.buckets = sorted(cfg.BUCKETS)
        self.fairness_engine = FairnessEngine()
        self._load_luts()

    @staticmethod
    def _normalize_model_key(model_name: str) -> str:
        return model_name.strip().lower().replace("_", "-")

    def _resolve_model_name(self, model_name: str) -> str:
        # Direct hit first.
        if model_name in cfg.SUPPORTED_MODELS:
            return model_name

        normalized = self._normalize_model_key(model_name.split("/")[-1])
        alias_map = {
            "qwen2.5-7b-instruct": "Qwen2-7B",
            "qwen2.5-7b": "Qwen2-7B",
            "qwen2-7b-instruct": "Qwen2-7B",
            "falcon-7b-instruct": "Falcon-7B",
            "mistral-7b-instruct-v0.2": "Mistral-7B-v0.1",
            "mistral-7b-v0.1": "Mistral-7B-v0.1",
            "gemma-7b-it": "Gemma-7B",
        }
        if normalized in alias_map:
            return alias_map[normalized]

        # Soft contains match against supported keys.
        for supported in cfg.SUPPORTED_MODELS.keys():
            supported_norm = self._normalize_model_key(supported)
            if supported_norm in normalized or normalized in supported_norm:
                return supported

        # Keep original if no mapping found; downstream loader will raise.
        return model_name

    def _load_luts(self):
        paths = cfg.get_lut_paths(self.model_name)
        try:
            with open(paths["raw"], "r") as f:
                raw_data = json.load(f)
                self.t_solo_dict = {int(k): float(v) for k, v in raw_data["T_solo"].items()}
                
            with open(paths["gain"], "r") as f:
                self.lut_gain = {
                    int(k): {int(kk): float(vv) for kk, vv in v.items()}
                    for k, v in json.load(f).items()
                }
            with open(paths["penalty"], "r") as f:
                self.lut_penalty = {
                    int(k): {int(kk): float(vv) for kk, vv in v.items()}
                    for k, v in json.load(f).items()
                }
        except FileNotFoundError:
            # Backward-compatible fallback for single-model filenames.
            legacy_paths = {
                "raw": os.path.join(cfg.DATA_DIR, "raw_profile.json"),
                "gain": os.path.join(cfg.DATA_DIR, "lut_gain.json"),
                "penalty": os.path.join(cfg.DATA_DIR, "lut_penalty.json"),
            }
            try:
                with open(legacy_paths["raw"], "r") as f:
                    raw_data = json.load(f)
                    self.t_solo_dict = {int(k): float(v) for k, v in raw_data["T_solo"].items()}
                with open(legacy_paths["gain"], "r") as f:
                    self.lut_gain = {
                        int(k): {int(kk): float(vv) for kk, vv in v.items()}
                        for k, v in json.load(f).items()
                    }
                with open(legacy_paths["penalty"], "r") as f:
                    self.lut_penalty = {
                        int(k): {int(kk): float(vv) for kk, vv in v.items()}
                        for k, v in json.load(f).items()
                    }
            except FileNotFoundError as legacy_exc:
                raise RuntimeError(
                    f"Fatal Error: missing LUT/profile files for model={self.model_name}. "
                    "Please run offline profiler first."
                ) from legacy_exc

    def _conservative_map_up(self, seq_len: int) -> int:
        for b in self.buckets:
            if b >= seq_len:
                return b
        return self.buckets[-1]

    @staticmethod
    def _nearest_key_ge_or_max(keys: list[int], target: int) -> int:
        for k in keys:
            if k >= target:
                return k
        return keys[-1]

    def _lookup_2d(self, table: dict[int, dict[int, float]], row: int, col: int) -> float:
        if not table:
            return 0.0
        row_keys = sorted(int(k) for k in table.keys())
        mapped_row = self._nearest_key_ge_or_max(row_keys, int(row))
        row_data = table.get(mapped_row, {})
        if not row_data:
            return 0.0
        col_keys = sorted(int(k) for k in row_data.keys())
        mapped_col = self._nearest_key_ge_or_max(col_keys, int(col))
        return float(row_data[mapped_col])

    def _lookup_1d(self, table: dict[int, float], key: int) -> float:
        if not table:
            return 0.0
        keys = sorted(int(k) for k in table.keys())
        mapped = self._nearest_key_ge_or_max(keys, int(key))
        return float(table[mapped])

    @staticmethod
    def _queue_length_from_rho(rho: float) -> int:
        # M/D/1: L = rho + rho^2 / (2 * (1 - rho))
        rho = max(0.0, min(0.99, float(rho)))
        if rho <= 0:
            return 0
        l_val = rho + (rho * rho) / (2.0 * (1.0 - rho))
        return max(1, int(round(l_val)))

    def _chunk_utility(
        self,
        *,
        short_bucket: int,
        long_bucket: int,
        chunk_bucket: int,
        t_solo_s: float,
        t_solo_l: float,
    ) -> tuple[float, float]:
        _ = t_solo_s
        t_conc_s = self._lookup_2d(self.lut_gain, short_bucket, chunk_bucket)
        t_penalty = self._lookup_2d(self.lut_penalty, long_bucket, chunk_bucket)
        utility = max(0.0, t_solo_l - t_conc_s)
        return utility, t_penalty

    def schedule_real(
        self,
        S_s: int,
        S_l: int,
        t_wait_us: float,
        queue_length: int,
        baseline_chunk: Optional[int] = None,
    ) -> int:
        """
        核心物理决断引擎。
        输入变更为 queue_length，由系统内部严格推导 rho。
        """
        b_s = self._conservative_map_up(S_s)
        b_l = self._conservative_map_up(S_l)

        t_solo_s = self._lookup_1d(self.t_solo_dict, b_s)
        t_solo_l = self._lookup_1d(self.t_solo_dict, b_l)

        # 1. M/D/1 排队论感知 & 动态 SLA 杠杆
        rho = self.fairness_engine.compute_rho_md1(queue_length)
        if self.fairness_engine.should_elastic_bypass(rho):
            return int(baseline_chunk) if baseline_chunk is not None else S_l
        w_fairness = self.fairness_engine.compute_weight(t_wait_us, t_solo_s)

        best_S_c = int(baseline_chunk) if baseline_chunk is not None else S_l
        max_net_benefit = 0.0

        baseline_bucket: Optional[int] = None
        ref_utility = 0.0
        ref_penalty = 0.0
        if baseline_chunk is not None:
            baseline_chunk = max(1, min(int(baseline_chunk), int(S_l)))
            if baseline_chunk < S_l:
                baseline_bucket = self._conservative_map_up(baseline_chunk)
                if baseline_bucket >= b_l:
                    baseline_bucket = None
            else:
                best_S_c = S_l
        if baseline_bucket is not None:
            ref_utility, ref_penalty = self._chunk_utility(
                short_bucket=b_s,
                long_bucket=b_l,
                chunk_bucket=baseline_bucket,
                t_solo_s=t_solo_s,
                t_solo_l=t_solo_l,
            )

        valid_chunk_candidates = [b for b in self.buckets if b_s <= b < b_l]
        if baseline_bucket is not None:
            valid_chunk_candidates = [b for b in valid_chunk_candidates if b < baseline_bucket]

        for S_c in valid_chunk_candidates:
            utility, t_penalty = self._chunk_utility(
                short_bucket=b_s,
                long_bucket=b_l,
                chunk_bucket=S_c,
                t_solo_s=t_solo_s,
                t_solo_l=t_solo_l,
            )

            if baseline_bucket is not None:
                delta_u = utility - ref_utility
                delta_p = t_penalty - ref_penalty
                cost_global = delta_p * (1.0 + self.gamma * rho)
                net_benefit = w_fairness * delta_u - cost_global
            else:
                # 3. 拥塞惩罚放大
                cost_global = t_penalty * (1.0 + self.gamma * rho)
                # 4. 目标函数决断
                net_benefit = w_fairness * utility - cost_global

            if net_benefit > max_net_benefit:
                max_net_benefit = net_benefit
                best_S_c = S_c

        return best_S_c

    def schedule(self, S_s: int, S_l: int, *args: Any, **kwargs: Any) -> int:
        """Backward-compatible wrapper over `schedule_real`.

        Supports both current and legacy call styles used in this repo:
        - schedule(S_s, S_l, t_wait_us=<float>, queue_length=<int>)
        - schedule(S_s, S_l, ..., ..., t_wait_us, rho_est)
        """
        t_wait_us: float = float(
            kwargs.get("t_wait_us", kwargs.get("t_wait_s_us", 0.0))
        )

        queue_length: Optional[int] = kwargs.get(
            "queue_length", kwargs.get("current_queue_depth")
        )
        if queue_length is not None:
            queue_length = int(queue_length)
        baseline_chunk: Optional[int] = kwargs.get("baseline_chunk")
        if baseline_chunk is not None:
            baseline_chunk = int(baseline_chunk)

        # Legacy positional pattern in e2e_simulator:
        # schedule(S_s, S_l, t_solo_s, t_solo_l, t_wait_us, rho_est)
        if queue_length is None and len(args) >= 4:
            t_wait_us = float(args[2])
            queue_length = self._queue_length_from_rho(float(args[3]))
        elif queue_length is None and len(args) >= 2:
            # Optional short form: schedule(S_s, S_l, t_wait_us, queue_length)
            t_wait_us = float(args[0])
            queue_length = int(args[1])

        if queue_length is None:
            queue_length = 0

        return self.schedule_real(
            S_s=S_s,
            S_l=S_l,
            t_wait_us=t_wait_us,
            queue_length=queue_length,
            baseline_chunk=baseline_chunk,
        )
