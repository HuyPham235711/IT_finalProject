import numpy as np
import gym
from gym import spaces


class TradingEnvDQNBehavioralROI(gym.Env):
    """
    DQN discrete action env (segment-aware, hourly):
      action: 0=HOLD, 1=LONG, 2=SHORT
      position: -1(short), 0(flat), +1(long)
      balance/equity: chạy liên tục xuyên chuỗi concat

    Reward = ROI_step (sau cost/slippage) + behavioral shaping nhỏ:
      - trade_penalty nếu đổi position
      - churn_penalty theo |Δpos|
      (tùy chọn) hold_bonus nếu giữ nguyên position

    Segment boundary handling:
      - segment_id array (len N)
      - nếu bước i -> i+1 là sang segment mới:
          + freeze step return: effective_logret = 0
          + vẫn cho phép agent đổi position (nhưng PnL step đó = 0), chi phí vẫn áp dụng nếu bạn muốn (mặc định: có áp dụng cost nếu đổi pos)
      - clamp log-return để tránh outlier
    """

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        prices,
        embeddings,
        segment_id=None,
        *,
        initial_balance=100000.0,
        transaction_cost=0.001,
        slippage_rate=0.0005,
        max_position=1.0,  # giữ để đồng nhất interface (DQN dùng -1/0/+1)
        seed=42,
        # boundary handling
        max_step_return=0.03,
        freeze_first_step_of_segment=True,
        # reward shaping
        roi_scale=1.0,
        trade_penalty=0.00005,
        churn_penalty=0.00005,
        hold_bonus=0.0,
        # episode
        episode_length=None,       # nếu None: chạy hết dữ liệu
        start_index=0,             # backtest: start=0
        random_start=False,        # train: có thể True
    ):
        super().__init__()
        self.rng = np.random.default_rng(seed)

        self.prices = np.asarray(prices, dtype=np.float64).reshape(-1)
        self.embeddings = np.asarray(embeddings, dtype=np.float32)

        if len(self.prices) != len(self.embeddings):
            raise ValueError(f"prices len ({len(self.prices)}) != embeddings len ({len(self.embeddings)})")

        self.n = len(self.prices)
        if segment_id is None:
            self.segment_id = np.zeros(self.n, dtype=np.int64)
        else:
            seg = np.asarray(segment_id, dtype=np.int64).reshape(-1)
            if len(seg) != self.n:
                raise ValueError(f"segment_id len ({len(seg)}) != data len ({self.n})")
            self.segment_id = seg

        self.initial_balance = float(initial_balance)
        self.transaction_cost = float(transaction_cost)
        self.slippage_rate = float(slippage_rate)
        self.max_position = float(max_position)

        self.max_step_return = float(max_step_return)
        self.freeze_first_step_of_segment = bool(freeze_first_step_of_segment)

        self.roi_scale = float(roi_scale)
        self.trade_penalty = float(trade_penalty)
        self.churn_penalty = float(churn_penalty)
        self.hold_bonus = float(hold_bonus)

        self.start_index = int(start_index)
        self.random_start = bool(random_start)

        if episode_length is None:
            self.episode_length = self.n - 2
        else:
            self.episode_length = int(episode_length)

        self.state_dim = int(self.embeddings.shape[1])
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=np.float32
        )

        # runtime
        self.reset()

    def _pick_start(self):
        if not self.random_start:
            return self.start_index
        max_start = max(0, self.n - self.episode_length - 2)
        return int(self.rng.integers(0, max_start + 1))

    def _get_obs(self):
        return self.embeddings[self.i].astype(np.float32)

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.i0 = self._pick_start()
        self.i = self.i0
        self.t = 0
        self.done = False

        self.position = 0     # -1/0/+1
        self.prev_position = 0
        self.prev_action = 0

        self.balance = self.initial_balance
        self.peak_balance = self.initial_balance

        obs = self._get_obs()
        return obs, {}

    @staticmethod
    def _action_to_pos(action: int, cur_pos: int) -> int:
        if action == 0:
            return cur_pos
        if action == 1:
            return 1
        if action == 2:
            return -1
        return cur_pos

    def step(self, action):
        if self.done:
            return self._get_obs(), 0.0, True, False, {}

        action = int(action)
        cur_price = float(self.prices[self.i])
        next_price = float(self.prices[self.i + 1])

        # segment boundary check: i -> i+1
        seg_now = int(self.segment_id[self.i])
        seg_next = int(self.segment_id[self.i + 1])
        boundary = (seg_next != seg_now)

        # target position
        target_pos = self._action_to_pos(action, self.position)

        # turnover / trading detection
        delta_pos = abs(target_pos - self.position)
        traded = 1 if delta_pos > 0 else 0

        # transaction cost + slippage (áp dụng khi đổi position)
        # (giữ đúng tinh thần PPO: có cost/slippage làm giảm pnl thực)
        cost = 0.0
        if traded:
            # simple proportional cost on balance
            cost = self.balance * (self.transaction_cost + self.slippage_rate) * (delta_pos / 1.0)

        # compute log return
        if cur_price <= 0.0 or next_price <= 0.0:
            raw_logret = 0.0
        else:
            raw_logret = float(np.log(next_price / cur_price))

        # clamp step return
        raw_logret = float(np.clip(raw_logret, -self.max_step_return, self.max_step_return))

        # freeze first step of new segment (avoid jump exploit)
        if self.freeze_first_step_of_segment and boundary:
            effective_logret = 0.0
        else:
            effective_logret = raw_logret


        pnl_gross = self.balance * (target_pos * effective_logret)
        pnl = pnl_gross - cost

        prev_balance = self.balance
        self.balance = float(self.balance + pnl)
        self.balance = max(self.balance, 0.0)

        self.peak_balance = max(self.peak_balance, self.balance)
        drawdown = 0.0 if self.peak_balance <= 0 else (self.peak_balance - self.balance) / self.peak_balance

        # ROI step (after cost)
        roi_step = 0.0 if prev_balance <= 0 else (self.balance - prev_balance) / prev_balance

        # Behavioral shaping (keep small)
        behavioral = 0.0
        if traded:
            behavioral -= self.trade_penalty
            behavioral -= self.churn_penalty * float(delta_pos)
        else:
            behavioral += self.hold_bonus

        reward = self.roi_scale * roi_step + behavioral
        reward = float(reward)

        # advance
        self.prev_position = self.position
        self.position = int(target_pos)
        self.prev_action = action

        self.i += 1
        self.t += 1

        # done conditions
        if (self.i >= self.n - 2) or (self.t >= self.episode_length):
            self.done = True

        obs = self._get_obs()
        info = {
            "price": cur_price,
            "next_price": next_price,
            "raw_logret": raw_logret,
            "effective_logret": effective_logret,
            "boundary": boundary,
            "segment_now": seg_now,
            "segment_next": seg_next,
            "position": self.position,
            "action": action,
            "pnl": pnl,
            "pnl_gross": pnl_gross,
            "cost": cost,
            "balance": self.balance,
            "roi_step": roi_step,
            "drawdown": drawdown,
        }

        return obs, reward, self.done, False, info
