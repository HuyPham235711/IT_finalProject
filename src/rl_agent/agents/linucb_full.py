# src/rl_agent/agents/linucb_full.py
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class LinUCBFullConfig:
    alpha: float = 0.6          # exploration strength
    ridge: float = 1.0          # L2 regularization
    tau: float = 0.0            # switch threshold
    cooldown: int = 2

    warmup_steps: int = 600
    epsilon: float = 0.02       # random exploration
    tie_eps: float = 1e-12

    min_adv: float = 0.0        # require advantage over FLAT to enter
    gamma: float = 1.0          # optional decay for non-stationary (<=1)

    seed: int = 42

    # performance/safety
    symmetrize_every: int = 500  # keep A_inv stable


class LinUCBFull:
    """
    Full LinUCB with per-action covariance:
      A_a = ridge*I + sum x x^T
      b_a = sum r x
      theta_a = A_a^{-1} b_a
      score_a = theta_a^T x + alpha * sqrt(x^T A_a^{-1} x)

    We maintain A_inv incrementally using Sherman–Morrison rank-1 updates.
    Complexity per step: O(n_actions * d^2), OK for d~260, actions=3.
    """

    def __init__(self, d: int, cfg: LinUCBFullConfig, n_actions: int = 3, eps: float = 1e-12):
        self.d = int(d)
        self.cfg = cfg
        self.n_actions = int(n_actions)
        self.eps = float(eps)

        ridge = max(float(cfg.ridge), self.eps)

        # Per-action matrices
        self.A = np.zeros((self.n_actions, self.d, self.d), dtype=np.float64)
        self.A_inv = np.zeros((self.n_actions, self.d, self.d), dtype=np.float64)
        self.b = np.zeros((self.n_actions, self.d), dtype=np.float64)
        self.theta = np.zeros((self.n_actions, self.d), dtype=np.float64)

        I = np.eye(self.d, dtype=np.float64)
        for a in range(self.n_actions):
            self.A[a] = ridge * I
            self.A_inv[a] = (1.0 / ridge) * I

        self.cooldown_left = 0
        self.rng = np.random.default_rng(int(cfg.seed))
        self._updates = 0

    def reset_cooldown(self):
        self.cooldown_left = 0

    def _maybe_decay(self, a: int):
        g = float(self.cfg.gamma)
        if g >= 1.0:
            return
        # decay A and b => keep ridge floor by re-adding (1-g)*ridge*I
        # A <- gA + (1-g)*ridge*I
        # b <- g b
        ridge = max(float(self.cfg.ridge), self.eps)
        I = np.eye(self.d, dtype=np.float64)

        self.A[a] = g * self.A[a] + (1.0 - g) * ridge * I
        self.b[a] = g * self.b[a]

        # recompute inverse safely (d~260, actions=3 => acceptable if called rarely)
        self.A_inv[a] = np.linalg.inv(self.A[a])
        self.theta[a] = self.A_inv[a] @ self.b[a]

    def _sherman_morrison_update_inv(self, A_inv: np.ndarray, x: np.ndarray) -> np.ndarray:
        """
        (A + x x^T)^{-1} = A^{-1} - (A^{-1} x x^T A^{-1}) / (1 + x^T A^{-1} x)
        """
        # u = A_inv x
        u = A_inv @ x
        denom = 1.0 + float(x.T @ u)
        if denom <= self.eps:
            # fallback: return original (rare)
            return A_inv
        return A_inv - np.outer(u, u) / denom

    def _scores(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        scores = np.zeros(self.n_actions, dtype=np.float64)

        for a in range(self.n_actions):
            mean = float(self.theta[a].dot(x))
            # bonus = alpha * sqrt(x^T A_inv x)
            quad = float(x.T @ (self.A_inv[a] @ x))
            quad = max(quad, 0.0)
            bonus = float(self.cfg.alpha) * np.sqrt(quad)
            scores[a] = mean + bonus

        scores[0] = 0.0  # FLAT baseline anchor (giữ giống design cũ của bạn)
        return scores

    def select_action(self, x: np.ndarray, cur_action: int, step: int) -> int:
        if step < int(self.cfg.warmup_steps):
            return int(step % self.n_actions)

        if self.cfg.epsilon > 0 and self.rng.random() < float(self.cfg.epsilon):
            return int(self.rng.integers(0, self.n_actions))

        scores = self._scores(x)
        max_s = float(np.max(scores))
        best_idxs = np.where(scores >= (max_s - float(self.cfg.tie_eps)))[0]
        best = int(self.rng.choice(best_idxs)) if len(best_idxs) > 1 else int(best_idxs[0])

        adv_best = float(scores[best] - scores[0])
        adv_cur = float(scores[cur_action] - scores[0])

        # nếu đang có vị thế mà advantage <= 0 => thoát FLAT
        if cur_action != 0 and adv_cur <= 0.0:
            return 0

        # vào lệnh nếu advantage > min_adv
        if best != 0 and adv_best <= float(self.cfg.min_adv):
            return 0

        if self.cooldown_left > 0:
            self.cooldown_left -= 1
            return int(cur_action)

        if best == cur_action:
            return best

        if (scores[best] - scores[cur_action]) > float(self.cfg.tau):
            self.cooldown_left = int(self.cfg.cooldown)
            return best

        return int(cur_action)

    def update(self, x: np.ndarray, a: int, r: float):
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        a = int(a)
        r = float(r)

        # optional non-stationary decay
        if float(self.cfg.gamma) < 1.0:
            self._maybe_decay(a)

        # A <- A + x x^T ; b <- b + r x
        self.A[a] += np.outer(x, x)
        self.b[a] += r * x

        # update inverse via Sherman-Morrison
        self.A_inv[a] = self._sherman_morrison_update_inv(self.A_inv[a], x)
        self.theta[a] = self.A_inv[a] @ self.b[a]

        self._updates += 1
        if self.cfg.symmetrize_every > 0 and (self._updates % int(self.cfg.symmetrize_every) == 0):
            # numerical stabilization
            for k in range(self.n_actions):
                self.A_inv[k] = 0.5 * (self.A_inv[k] + self.A_inv[k].T)
