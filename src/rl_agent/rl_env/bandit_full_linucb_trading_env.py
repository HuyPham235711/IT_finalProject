# src/rl_agent/rl_env/bandit_trading_env_v2.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


@dataclass
class BanditCostConfigV2:
    switch_cost: float = 0.0006    # phí đổi vị thế
    hold_cost: float = 0.00002     # phạt giữ lệnh lâu
    flat_bonus: float = 0.00001    # thưởng nhỏ cho FLAT khi thị trường nhiễu


def action_to_pos(a: int) -> int:
    # 0=FLAT, 1=LONG, 2=SHORT
    if a == 1:
        return 1
    if a == 2:
        return -1
    return 0


class TradingEnvBanditFusionV2:
    """
    Bandit trading env (V2 – reward kinh tế chuẩn)

    Timeline:
    - observe embedding at time i
    - choose action_i
    - execute at bar (i+1): open -> close
    """

    def __init__(
        self,
        open_raw: np.ndarray,
        close_raw: np.ndarray,
        embeddings: np.ndarray,
        *,
        costs: BanditCostConfigV2 = BanditCostConfigV2(),
        initial_equity: float = 1.0,
    ):
        self.open_raw = np.asarray(open_raw, dtype=np.float64)
        self.close_raw = np.asarray(close_raw, dtype=np.float64)
        self.emb = np.asarray(embeddings, dtype=np.float32)

        assert len(self.open_raw) == len(self.close_raw) == len(self.emb)
        assert len(self.open_raw) >= 3

        self.n = len(self.open_raw)
        self.costs = costs
        self.initial_equity = float(initial_equity)

        self.reset()

    @staticmethod
    def _logret(open_p: float, close_p: float) -> float:
        if open_p <= 0 or close_p <= 0:
            return 0.0
        return float(np.log(close_p / open_p))

    # ======================================================
    # API
    # ======================================================
    def reset(self) -> Tuple[np.ndarray, Dict]:
        self.i = 0
        self.prev_pos = 0
        self.cur_action = 0
        self.log_equity = np.log(max(self.initial_equity, 1e-12))

        obs = self.emb[self.i]
        info = {
            "t": self.i,
            "equity": float(np.exp(self.log_equity)),
            "pos": 0,
            "action": 0,
        }
        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        a = int(action)
        pos = action_to_pos(a)

        # reward realized on next bar
        lr = self._logret(
            self.open_raw[self.i + 1],
            self.close_raw[self.i + 1],
        )

        delta_pos = abs(pos - self.prev_pos)
        traded = int(delta_pos > 0)

        # --------------------------
        # COSTS
        # --------------------------
        cost = (
            self.costs.switch_cost * delta_pos
            + self.costs.hold_cost * abs(pos)
        )

        # --------------------------
        # REWARD
        # --------------------------
        reward = pos * lr - cost

        if pos == 0:
            reward += self.costs.flat_bonus

        self.log_equity += reward
        equity = float(np.exp(self.log_equity))

        # advance
        self.i += 1
        done = self.i >= self.n - 2

        self.prev_pos = pos
        self.cur_action = a

        obs = self.emb[self.i] if not done else self.emb[self.n - 2]

        info = {
            "t": self.i,
            "action": a,
            "pos": pos,
            "traded": traded,
            "bar_logret": lr,
            "cost": cost,
            "reward": reward,
            "equity": equity,
        }

        return obs, float(reward), bool(done), info
