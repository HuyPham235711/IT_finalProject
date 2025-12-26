from __future__ import annotations

from typing import Optional, Tuple, Dict, Any
import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except Exception:
    import gym
    from gym import spaces


class TradingEnvSACHybrid(gym.Env):
    """
    SAC Continuous Trading Env (PATCH v2.1: delta-action + anti-churn + anti-flip, deadband on DELTA).

    Action raw_a in [-1,1]
      desired_delta = raw_a * max_delta_pos
      if |desired_delta| < deadband_delta => desired_delta = 0
      position_next = clip(position_prev + desired_delta, [-max_position, +max_position])

    step_pnl = position_next * ret - cost(|delta_pos|) - holding_cost*|position_next|

    reward = alpha * step_pnl
           - turnover_penalty * (delta_pos^2)
           - pos_penalty * (position_next^2)
           - action_penalty * (raw_a^2)
           - flip_penalty * 1{sign(raw_a)!=sign(prev_raw_a)}   (optional)

    Notes:
      - Keep action-deadband (deadband) at 0.0 for delta-action; use deadband_delta instead.
      - This avoids the "raw action ~0.07 but deadband=0.10 => no trading" failure mode.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        prices: np.ndarray,
        embeddings: np.ndarray,

        episode_length: int = 256,
        is_backtest: bool = False,

        start_index: int = 0,
        end_index: Optional[int] = None,
        seed: Optional[int] = None,

        # account / limits
        initial_balance: float = 1000.0,
        max_position: float = 1.0,

        # costs
        transaction_cost: float = 0.001,
        slippage_rate: float = 0.0005,

        # action shaping
        deadband: Optional[float] = None,     
        max_delta_pos: float = 0.02,
        deadband_delta: float = 0.0005,       

        # reward shaping
        alpha: float = 200.0,
        turnover_penalty: float = 4.0,
        pos_penalty: float = 0.0,
        holding_cost: float = 2e-4,

        action_penalty: float = 0.05,
        flip_penalty: float = 0.002,

        # drawdown penalty (optional)
        dd_threshold: float = 0.10,
        dd_penalty: float = 0.0,

        # reward clip
        reward_clip: float = 10.0,

        # obs extras
        include_position_in_obs: bool = True,
        include_equity_in_obs: bool = False,
    ):
        super().__init__()

        self.is_backtest = bool(is_backtest)

        self.prices = np.asarray(prices, dtype=np.float32)
        self.embeddings = np.asarray(embeddings, dtype=np.float32)

        if self.embeddings.ndim != 2:
            raise ValueError("embeddings must have shape (T, D)")
        if self.prices.ndim != 1:
            raise ValueError("prices must have shape (T,)")
        if len(self.prices) != len(self.embeddings):
            raise ValueError("prices and embeddings must have same length T")
        if len(self.prices) < 2:
            raise ValueError("Need at least 2 timesteps of prices")

        self.T = len(self.prices)
        self.state_dim = int(self.embeddings.shape[1])

        self.episode_length = int(episode_length)
        if self.episode_length < 1:
            raise ValueError("episode_length must be >= 1")

        self.start_index = int(start_index)
        self.end_index = int(end_index) if end_index is not None else (self.T - 1)
        self.end_index = min(self.end_index, self.T - 1)
        if not (0 <= self.start_index < self.end_index):
            raise ValueError("Invalid index range: require 0 <= start_index < end_index")

        # params
        self.initial_balance = float(initial_balance)
        self.max_position = float(max_position)
        if self.max_position <= 0:
            raise ValueError("max_position must be > 0")

        self.transaction_cost = float(transaction_cost)
        self.slippage_rate = float(slippage_rate)

        # action shaping
        self.deadband = float(deadband) if deadband is not None else 0.0
        self.max_delta_pos = float(max_delta_pos)
        if self.max_delta_pos <= 0:
            raise ValueError("max_delta_pos must be > 0")

        self.deadband_delta = float(deadband_delta)
        if self.deadband_delta < 0:
            raise ValueError("deadband_delta must be >= 0")

        # reward shaping
        self.alpha = float(alpha)
        self.turnover_penalty = float(turnover_penalty)
        self.pos_penalty = float(pos_penalty)
        self.holding_cost_rate = float(holding_cost)

        self.action_penalty = float(action_penalty)
        self.flip_penalty = float(flip_penalty)

        self.dd_threshold = float(dd_threshold)
        self.dd_penalty = float(dd_penalty)

        self.reward_clip = float(reward_clip) if reward_clip is not None else 0.0

        # obs config
        self.include_position_in_obs = bool(include_position_in_obs)
        self.include_equity_in_obs = bool(include_equity_in_obs)

        self.rng = np.random.default_rng(seed)

        # spaces
        obs_dim = self.state_dim
        if self.include_position_in_obs:
            obs_dim += 1
        if self.include_equity_in_obs:
            obs_dim += 1

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        # runtime states
        self._t0: int = 0
        self._t: int = 0
        self._steps: int = 0

        self.position: float = 0.0
        self.equity: float = self.initial_balance
        self.peak_equity: float = self.initial_balance

        self.prev_action: float = 0.0  # previous RAW action for flip penalty

    def _get_obs(self) -> np.ndarray:
        x = self.embeddings[self._t].astype(np.float32, copy=False)
        parts = [x]

        if self.include_position_in_obs:
            parts.append(np.array([self.position], dtype=np.float32))

        if self.include_equity_in_obs:
            eq_feat = (self.equity / max(self.initial_balance, 1e-9)) - 1.0
            parts.append(np.array([eq_feat], dtype=np.float32))

        return np.concatenate(parts, axis=0).astype(np.float32, copy=False)

    def _compute_ret(self, t: int) -> float:
        p0 = float(self.prices[t])
        p1 = float(self.prices[t + 1])
        if p0 <= 0:
            return 0.0
        return (p1 - p0) / p0

    def _exec_cost(self, delta_pos: float) -> float:
        return (self.transaction_cost + self.slippage_rate) * abs(delta_pos)

    def _parse_action(self, action: np.ndarray) -> float:
        if isinstance(action, (list, tuple)):
            a = float(action[0])
        else:
            a = float(np.asarray(action, dtype=np.float32).reshape(-1)[0])
        a = float(np.clip(a, -1.0, 1.0))

        # legacy action-deadband (normally keep 0.0 when using deadband_delta)
        if self.deadband > 0 and abs(a) < self.deadband:
            a = 0.0
        return a

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        max_t0 = self.end_index - (self.episode_length + 1)
        if max_t0 < self.start_index:
            raise ValueError(
                f"Not enough data for episode_length={self.episode_length}. "
                f"Need end_index-start_index >= episode_length+1."
            )

        self._t0 = self.start_index if self.is_backtest else int(self.rng.integers(self.start_index, max_t0 + 1))
        self._t = self._t0
        self._steps = 0

        self.position = 0.0
        self.equity = float(self.initial_balance)
        self.peak_equity = float(self.initial_balance)
        self.prev_action = 0.0

        obs = self._get_obs()
        info = {"t0": self._t0, "equity": self.equity, "position": self.position}
        return obs, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        raw_a = self._parse_action(action)
        prev_a = float(self.prev_action)

        prev_pos = float(self.position)

        # delta-action
        desired_delta = raw_a * self.max_delta_pos

        # deadband on delta-position (unit-consistent)
        if self.deadband_delta > 0 and abs(desired_delta) < self.deadband_delta:
            desired_delta = 0.0

        target_pos = float(np.clip(prev_pos + desired_delta, -self.max_position, self.max_position))
        delta_pos = target_pos - prev_pos

        ret = self._compute_ret(self._t)

        cost = self._exec_cost(delta_pos)
        holding_cost = self.holding_cost_rate * abs(target_pos)

        step_pnl = (target_pos * ret) - cost - holding_cost

        self.equity *= (1.0 + step_pnl)
        self.peak_equity = max(self.peak_equity, self.equity)

        drawdown = 0.0
        if self.peak_equity > 0:
            drawdown = (self.peak_equity - self.equity) / self.peak_equity

        # apply
        self.position = target_pos
        self.prev_action = raw_a

        # flip penalty (based on RAW action sign)
        flip = 0.0
        if (raw_a != 0.0) and (prev_a != 0.0) and (np.sign(raw_a) != np.sign(prev_a)):
            flip = 1.0

        reward = self.alpha * step_pnl
        reward -= self.turnover_penalty * (delta_pos ** 2)
        reward -= self.pos_penalty * (self.position ** 2)
        reward -= self.action_penalty * (raw_a ** 2)
        reward -= self.flip_penalty * flip

        if self.dd_penalty > 0.0 and drawdown > self.dd_threshold:
            dd_excess = (drawdown - self.dd_threshold)
            reward -= self.dd_penalty * (dd_excess ** 2)

        if self.reward_clip and self.reward_clip > 0:
            reward = float(np.clip(reward, -self.reward_clip, self.reward_clip))

        # advance
        self._t += 1
        self._steps += 1

        terminated = False
        truncated = False
        if self._steps >= self.episode_length:
            truncated = True
        if self._t >= self.end_index:
            truncated = True
        if (not np.isfinite(self.equity)) or self.equity <= 0:
            terminated = True

        obs = self._get_obs()

        info = {
            "t": self._t,
            "ret": ret,
            "raw_action": raw_a,
            "prev_action": prev_a,
            "flip": flip,
            "prev_pos": prev_pos,
            "target_pos": target_pos,
            "position": self.position,
            "desired_delta": desired_delta,
            "delta_pos": delta_pos,
            "cost": cost,
            "holding_cost": holding_cost,
            "step_pnl": step_pnl,
            "equity": self.equity,
            "roi": (self.equity / max(self.initial_balance, 1e-9)) - 1.0,
            "peak_equity": self.peak_equity,
            "drawdown": drawdown,
            "deadband_action": self.deadband,
            "deadband_delta": self.deadband_delta,
            "max_delta_pos": self.max_delta_pos,
        }

        return obs, float(reward), bool(terminated), bool(truncated), info

    def render(self):
        pass

    def close(self):
        pass
