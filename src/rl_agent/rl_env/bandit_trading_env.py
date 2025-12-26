from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import numpy as np


@dataclass
class BanditCostConfig:
    switch_cost: float = 0.0006  # gộp fee+spread+slippage (vừa phải cho đồ án)
    hold_cost: float = 0.0


def action_to_pos(a: int) -> int:
    # 0=FLAT, 1=LONG, 2=SHORT
    if a == 1:
        return 1
    if a == 2:
        return -1
    return 0


class TradingEnvBanditFusion:
    """
    Bandit-style env (NO Gym dependency) that matches your execution rule:

    - Observe embedding at time i
    - Choose action_i
    - Execute at next open => PnL measured on bar (i+1): open[i+1] -> close[i+1]
    - This "lag 1 step" prevents look-ahead leakage

    State: fusion_embeddings[i]
    Action: 0 flat, 1 long, 2 short
    Reward: log-wealth increment = pos*log(open_{i+1} -> close_{i+1}) - costs
    """

    def __init__(
        self,
        open_raw: np.ndarray,
        close_raw: np.ndarray,
        embeddings: np.ndarray,
        *,
        costs: BanditCostConfig = BanditCostConfig(),
        initial_equity: float = 1.0,
    ):
        self.open_raw = np.asarray(open_raw, dtype=np.float64).reshape(-1)
        self.close_raw = np.asarray(close_raw, dtype=np.float64).reshape(-1)
        self.emb = np.asarray(embeddings, dtype=np.float32)

        if self.emb.ndim != 2:
            raise ValueError(f"embeddings must be (T,D), got {self.emb.shape}")

        if len(self.open_raw) != len(self.close_raw) or len(self.open_raw) != len(self.emb):
            raise ValueError(
                f"Length mismatch: open={len(self.open_raw)}, close={len(self.close_raw)}, emb={len(self.emb)}"
            )

        self.n = len(self.open_raw)
        if self.n < 3:
            raise ValueError("Need at least 3 timesteps for lagged execution simulation.")

        self.costs = costs
        self.initial_equity = float(initial_equity)

        # runtime
        self.i = 0
        self.cur_action = 0
        self.prev_pos = 0
        self.log_equity = float(np.log(max(self.initial_equity, 1e-12)))

    @staticmethod
    def _bar_logret(open_p: float, close_p: float) -> float:
        if open_p <= 0 or close_p <= 0:
            return 0.0
        return float(np.log(close_p / open_p))

    def reset(self) -> Tuple[np.ndarray, Dict]:
        self.i = 0
        self.cur_action = 0
        self.prev_pos = 0
        self.log_equity = float(np.log(max(self.initial_equity, 1e-12)))

        obs = self.emb[self.i]
        info = {"equity": float(np.exp(self.log_equity)), "pos": 0, "action": 0, "t": self.i}
        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Step i uses action_i and realizes reward on bar (i+1).
        Done when i reaches n-2 (since we access i+1).
        """
        a = int(action)
        pos = action_to_pos(a)

        # reward realized on bar (i+1)
        lr = self._bar_logret(self.open_raw[self.i + 1], self.close_raw[self.i + 1])

        delta_pos = abs(pos - self.prev_pos)
        traded = 1 if delta_pos > 0 else 0
        cost = self.costs.switch_cost * float(delta_pos) + self.costs.hold_cost * float(abs(pos))

        r = (pos * lr) - cost  # log-wealth increment
        self.log_equity += float(r)
        equity = float(np.exp(self.log_equity))

        # advance
        self.i += 1
        done = (self.i >= self.n - 1)  # but we only valid until n-2 for next step
        # practical termination: when we cannot access i+1 next time
        done = (self.i >= self.n - 1) or (self.i >= self.n - 1) or (self.i >= self.n - 1)
        # stricter:
        done = (self.i >= self.n - 1) or (self.i >= self.n - 2)

        self.cur_action = a
        self.prev_pos = pos

        obs = self.emb[self.i] if not done else self.emb[self.n - 2]
        info = {
            "t": self.i,
            "action": a,
            "pos": pos,
            "traded": traded,
            "delta_pos": float(delta_pos),
            "bar_logret": float(lr),
            "cost": float(cost),
            "equity": equity,
        }
        return obs, float(r), bool(done), info
