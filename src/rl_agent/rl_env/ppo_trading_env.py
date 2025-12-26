
import numpy as np
import gym
from gym import spaces


class TradingEnvPPOHybrid(gym.Env):
    """
    Trading env dùng chung cho PPO/SAC.

    - Observation: fusion embeddings (shape = [embed_dim])
    - Action: continuous in [-1, 1] -> target position in [-max_position, max_position]
    - Reward: SAC-friendly (advantage vs flat + wrong-way penalty + smooth turnover + soft DD penalty)
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        prices: np.ndarray,
        embeddings: np.ndarray,
        episode_length: int = 2048,
        initial_balance: float = 1000.0,
        is_backtest: bool = False,
        transaction_cost: float = 0.001,
        slippage_rate: float = 0.0005,
        max_position: float = 1.0,
        max_step_return: float = 0.03,
        reward_scale: float = 1.0,
        seed: int = 42,

        # ===== Reward knobs  =====
        alpha: float = 200.0,          
        tc: float = 0.01,             
        dd_target: float = 0.06,      
        c_dd: float = 1.5,            
        wrong_way_mult: float = 0.7,  
        reward_clip: float = 5.0,     

        # ===== Action/Position dynamics =====
        action_scale: float = 3.0,    
        inertia: float = 0.05,     
    ):
        super().__init__()
        np.random.seed(seed)

        # Data
        self.prices = prices.astype(np.float64)
        self.embeddings = embeddings.astype(np.float32)

        self.n = len(self.prices)
        self.episode_length = int(episode_length)
        self.initial_balance = float(initial_balance)
        self.is_backtest = bool(is_backtest)

        # Execution params
        self.transaction_cost = float(transaction_cost)
        self.slippage_rate = float(slippage_rate)
        self.max_position = float(max_position)
        self.max_step_return = float(max_step_return)

        # Reward params
        self.alpha = float(alpha)
        self.tc = float(tc)
        self.dd_target = float(dd_target)
        self.c_dd = float(c_dd)
        self.wrong_way_mult = float(wrong_way_mult)
        self.reward_scale = float(reward_scale)
        self.reward_clip = float(reward_clip)

        # Action dynamics
        self.action_scale = float(action_scale)
        self.inertia = float(inertia)

        # Spaces
        self.action_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.embeddings.shape[1],),
            dtype=np.float32,
        )

        self.reset()

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)

        if self.is_backtest:
            self.start_idx = 0
            self.end_idx = self.n - 1
        else:
            max_start = self.n - self.episode_length - 2
            if max_start <= 0:
                raise ValueError(
                    f"Episode length ({self.episode_length}) is too long for data length ({self.n})."
                )
            self.start_idx = np.random.randint(0, max_start)
            self.end_idx = self.start_idx + self.episode_length

        self.idx = self.start_idx
        self.balance = self.initial_balance
        self.peak_balance = self.initial_balance

        self.position = 0.0
        self.prev_position = 0.0

        self.step_count = 0

        return self.embeddings[self.idx], {}

    @staticmethod
    def _log_return(p, np_, eps=1e-12):
        p = max(float(p), eps)
        np_ = max(float(np_), eps)
        return np.log(np_ / p)

    def step(self, action):
        # -------------------------------
        # 1) Action -> target position
        # -------------------------------
        a = float(action[0])
        a = float(np.clip(a, -1.0, 1.0))

        target_pos = float(
            np.clip(a * self.action_scale, -1.0, 1.0) * self.max_position
        )

        self.prev_position = float(self.position)
        self.position = float(
            np.clip(
                (1.0 - self.inertia) * target_pos + self.inertia * self.prev_position,
                -self.max_position,
                self.max_position,
            )
        )

        delta_pos = abs(self.position - self.prev_position)

        # -------------------------------
        # 2) Market return
        # -------------------------------
        p = self.prices[self.idx]
        np_ = self.prices[self.idx + 1]
        ret = float(
            np.clip(
                self._log_return(p, np_),
                -self.max_step_return,
                self.max_step_return,
            )
        )

        # -------------------------------
        # 3) PnL update 
        # -------------------------------
        # execution cost as return drag proportional to turnover
        exec_cost = float(delta_pos * (self.transaction_cost + self.slippage_rate))

        step_pnl = (self.position * ret) - exec_cost
        self.balance *= (1.0 + step_pnl)

        self.peak_balance = max(self.peak_balance, self.balance)
        drawdown = float((self.peak_balance - self.balance) / max(self.peak_balance, 1e-9))

        # -------------------------------
        # 4) Reward 
        # -------------------------------
        reward = 0.0

        # (A) Advantage vs flat
        reward += self.alpha * (self.position * ret)

        # (B) Penalize being wrong-way (forces cutting exposure when wrong)
        if self.position * ret < 0:
            reward -= self.wrong_way_mult * self.alpha * abs(self.position * ret)

        # (C) Smooth turnover penalty
        reward -= self.tc * (delta_pos ** 2)

        # (D) Soft drawdown penalty beyond threshold
        if drawdown > self.dd_target:
            reward -= self.c_dd * ((drawdown - self.dd_target) ** 2)

        reward *= self.reward_scale
        reward = float(np.clip(reward, -self.reward_clip, self.reward_clip))

        # -------------------------------
        # 5) Advance
        # -------------------------------
        self.idx += 1
        self.step_count += 1

        terminated = self.idx >= self.end_idx
        truncated = False

        obs = self.embeddings[self.idx]
        info = {
            "balance": float(self.balance),
            "drawdown": float(drawdown),
            "roi": float(self.balance / self.initial_balance - 1.0),
            "position": float(self.position),
            "pnl": float(step_pnl),
            "ret": float(ret),
            "delta_pos": float(delta_pos),
            "exec_cost": float(exec_cost),
        }

        return obs, reward, terminated, truncated, info
