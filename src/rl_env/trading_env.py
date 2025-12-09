import gym
from gym import spaces
import numpy as np


class TradingEnvPPOHybrid(gym.Env):
    """
    PPO Hybrid Environment:
    - Reward = Direction-based (easy for PPO to learn)
    - PnL, ROI, Balance = logged separately for backtesting
    - Balance does NOT affect training reward
    """

    metadata = {"render.modes": ["human"]}

    def __init__(self, fusion_emb_path, ohlcv_df, initial_balance=1000.0):
        super().__init__()

        # Load data
        self.fusion_emb = np.load(fusion_emb_path)
        self.close_prices = ohlcv_df["close"].to_numpy()
        self.close_prices = self.close_prices[-len(self.fusion_emb):]

        assert len(self.fusion_emb) == len(self.close_prices)

        state_dim = self.fusion_emb.shape[1]

        # 0 = Sell, 1 = Hold, 2 = Buy
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float32
        )

        self.initial_balance = initial_balance
        self.reset()

    def reset(self):
        self.current_step = 0
        self.balance = float(self.initial_balance)
        self.position = 0
        self.equity_curve = [self.balance]
        return self._get_state()

    def _get_state(self):
        return self.fusion_emb[self.current_step].astype(np.float32)

    def step(self, action):
        if isinstance(action, np.ndarray):
            action = int(action.item())   # ⭐ chuyển từ array([2]) → 2


        price_now = self.close_prices[self.current_step]
        price_next = self.close_prices[self.current_step + 1]
        delta = price_next - price_now

        # True direction
        true_move = 1 if delta > 0 else -1 if delta < 0 else 0

        # Action → directional move
        action_move = {0: -1, 1: 0, 2: 1}[action]

        # ==========================================================
        # 1) REWARD FOR PPO (simple, stable)
        # ==========================================================
        if action_move == 0:
            reward = -0.05          # light penalty for HOLD
        elif action_move == true_move:
            reward = +1.0           # correct prediction
        else:
            reward = -1.0           # wrong prediction

        # ==========================================================
        # 2) FINANCIAL METRICS (NOT part of reward)
        # ==========================================================
        # Position PnL based on direction
        pnl = action_move * delta          # raw PnL per unit
        roi = pnl / price_now              # simple ROI
        equity_change = self.balance * roi # actual money change
        self.balance += equity_change      # update balance

        self.equity_curve.append(self.balance)
        self.position = action_move

        # ==========================================================
        # Step forward
        # ==========================================================
        self.current_step += 1
        done = self.current_step >= len(self.close_prices) - 1

        info = {
            "step": int(self.current_step),
            "price": float(price_now),
            "price_next": float(price_next),
            "delta": float(delta),
            "true_move": int(true_move),
            "action": int(action),
            "action_move": int(action_move),

            # PPO reward
            "reward": float(reward),

            # Real market metrics
            "pnl": float(pnl),
            "roi": float(roi),
            "balance": float(self.balance),
        }

        return self._get_state(), reward, done, info

    def render(self, mode="human"):
        print(f"Step {self.current_step} | Balance={self.balance:.2f}")
