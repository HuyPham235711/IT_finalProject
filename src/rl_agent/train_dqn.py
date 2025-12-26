import os
import json
import time
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sqlalchemy import create_engine

from src.rl_agent.rl_env.dqn_trading_env import TradingEnvDQNBehavioralROI


# ============================================================
# CONFIG
# ============================================================
@dataclass
class DQNConfig:
    seed: int = 42

    # data
    schema: str = "it_final"
    table_train: str = "processed_ohlcv_train"
    price_col: str = "close"
    datetime_col: str = "datetime"

    # embeddings path (train)
    embeddings_path: str = "E:/TDTu/TAI_LIEU/KY1-NAM5/DU_AN_CNTT/results/fusion_rl/v2/fusion_embeddings_train_v2.npy"

    # env / portfolio
    initial_balance: float = 100000.0
    transaction_cost: float = 0.001
    slippage_rate: float = 0.0005
    max_step_return: float = 0.03
    freeze_first_step_of_segment: bool = True

    # reward
    roi_scale: float = 1000.0      # ROI theo giờ nhỏ → scale để DQN học ổn
    trade_penalty: float = 0.02    # scale-level penalty (tùy)
    churn_penalty: float = 0.02
    hold_bonus: float = 0.0

    # training
    episodes: int = 60
    episode_length: int = 2000
    gamma: float = 0.99
    lr: float = 1e-4
    batch_size: int = 128
    replay_size: int = 200_000
    warmup_steps: int = 10_000
    target_update: int = 2000
    train_every: int = 1
    grad_clip: float = 1.0

    # epsilon
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_steps: int = 200_000

    # output
    project_root: str = "E:/TDTu/TAI_LIEU/KY1-NAM5/DU_AN_CNTT"
    out_dir: str = "results/rl_agent/dqn/behavioral_roi"
    log_dir: str = "logs/rl/dqn_behavioral_roi"


# ============================================================
# LOGGING
# ============================================================
def setup_logging(log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers = []

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)


# ============================================================
# DATA
# ============================================================
def load_table_from_postgres(engine, schema: str, table: str, cols: list[str]):
    q = f'SELECT {",".join(cols)} FROM "{schema}"."{table}" ORDER BY datetime'
    df = pd.read_sql(q, engine)
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df


def infer_segment_id(df: pd.DataFrame, max_gap_hours: int = 1) -> np.ndarray:
    dt = df["datetime"]
    gap_hours = dt.diff().dt.total_seconds().fillna(0) / 3600.0
    seg = (gap_hours > max_gap_hours).cumsum().astype(int)
    return seg.values


# ============================================================
# DQN CORE
# ============================================================
class ReplayBuffer:
    def __init__(self, state_dim: int, capacity: int, seed: int = 42):
        self.capacity = int(capacity)
        self.rng = np.random.default_rng(seed)

        self.s = np.zeros((self.capacity, state_dim), dtype=np.float32)
        self.a = np.zeros((self.capacity,), dtype=np.int64)
        self.r = np.zeros((self.capacity,), dtype=np.float32)
        self.s2 = np.zeros((self.capacity, state_dim), dtype=np.float32)
        self.d = np.zeros((self.capacity,), dtype=np.float32)

        self.ptr = 0
        self.size = 0

    def add(self, s, a, r, s2, done):
        i = self.ptr
        self.s[i] = s
        self.a[i] = a
        self.r[i] = r
        self.s2[i] = s2
        self.d[i] = 1.0 if done else 0.0

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        idx = self.rng.integers(0, self.size, size=batch_size)
        return self.s[idx], self.a[idx], self.r[idx], self.s2[idx], self.d[idx]


class QNet(nn.Module):
    def __init__(self, state_dim: int, n_actions: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions),
        )

    def forward(self, x):
        return self.net(x)


def linear_epsilon(step: int, eps_start: float, eps_end: float, decay_steps: int) -> float:
    if step >= decay_steps:
        return eps_end
    frac = step / float(decay_steps)
    return eps_start + frac * (eps_end - eps_start)


# ============================================================
# TRAIN
# ============================================================
def main():
    cfg = DQNConfig()
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    project_root = Path(cfg.project_root)
    out_dir = project_root / cfg.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    log_path = project_root / cfg.log_dir / "train_dqn_behavioral_roi.log"
    setup_logging(log_path)

    logging.info("=== TRAIN DQN (Behavioral ROI, segment-aware, hourly) ===")

    # PG
    conn = os.environ.get("PG_CONN_STR", "postgresql+psycopg2://postgres:123456789@localhost:5432/postgres")
    engine = create_engine(conn)

    # Load df
    df = load_table_from_postgres(engine, cfg.schema, cfg.table_train, [cfg.datetime_col, cfg.price_col])
    seg = infer_segment_id(df, max_gap_hours=1)
    emb = np.load(cfg.embeddings_path)
    lookback = len(df) - len(emb)

    prices = df[cfg.price_col].astype(float).values[lookback:]
    seg = seg[lookback:]

    if len(emb) != len(prices):
        raise ValueError(
            f"After align: embeddings len {len(emb)} != prices len {len(prices)}"
        )


    state_dim = int(emb.shape[1])
    env = TradingEnvDQNBehavioralROI(
        prices=prices,
        embeddings=emb,
        segment_id=seg,
        initial_balance=cfg.initial_balance,
        transaction_cost=cfg.transaction_cost,
        slippage_rate=cfg.slippage_rate,
        max_step_return=cfg.max_step_return,
        freeze_first_step_of_segment=cfg.freeze_first_step_of_segment,
        roi_scale=cfg.roi_scale,
        trade_penalty=cfg.trade_penalty,
        churn_penalty=cfg.churn_penalty,
        hold_bonus=cfg.hold_bonus,
        episode_length=cfg.episode_length,
        start_index=0,
        random_start=True,  # train random start
        seed=cfg.seed,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    q = QNet(state_dim).to(device)
    qt = QNet(state_dim).to(device)
    qt.load_state_dict(q.state_dict())
    qt.eval()

    opt = optim.Adam(q.parameters(), lr=cfg.lr)
    rb = ReplayBuffer(state_dim, cfg.replay_size, seed=cfg.seed)

    global_step = 0
    best_final_balance = -1e18
    best_path = out_dir / "best.pt"

    stats = []

    for ep in range(1, cfg.episodes + 1):
        s, _ = env.reset()
        total_reward = 0.0
        steps = 0
        t0 = time.time()

        while True:
            eps = linear_epsilon(global_step, cfg.eps_start, cfg.eps_end, cfg.eps_decay_steps)
            if np.random.rand() < eps:
                a = np.random.randint(0, 3)
            else:
                with torch.no_grad():
                    x = torch.tensor(s, dtype=torch.float32, device=device).unsqueeze(0)
                    a = int(torch.argmax(q(x), dim=1).item())

            s2, r, done, _, info = env.step(a)
            rb.add(s, a, r, s2, done)

            s = s2
            total_reward += float(r)
            steps += 1
            global_step += 1

            # train
            if rb.size >= cfg.warmup_steps and (global_step % cfg.train_every == 0):
                bs, ba, br, bs2, bd = rb.sample(cfg.batch_size)

                bs = torch.tensor(bs, dtype=torch.float32, device=device)
                ba = torch.tensor(ba, dtype=torch.int64, device=device).unsqueeze(1)
                br = torch.tensor(br, dtype=torch.float32, device=device).unsqueeze(1)
                bs2 = torch.tensor(bs2, dtype=torch.float32, device=device)
                bd = torch.tensor(bd, dtype=torch.float32, device=device).unsqueeze(1)

                qv = q(bs).gather(1, ba)

                with torch.no_grad():
                    # Double DQN-lite: action from q, value from qt
                    a2 = torch.argmax(q(bs2), dim=1, keepdim=True)
                    q2 = qt(bs2).gather(1, a2)
                    target = br + (1.0 - bd) * cfg.gamma * q2

                loss = nn.functional.smooth_l1_loss(qv, target)

                opt.zero_grad(set_to_none=True)
                loss.backward()
                if cfg.grad_clip is not None:
                    nn.utils.clip_grad_norm_(q.parameters(), cfg.grad_clip)
                opt.step()

            # target update
            if global_step % cfg.target_update == 0:
                qt.load_state_dict(q.state_dict())

            if done:
                break

        final_balance = float(env.balance)
        roi_total = (final_balance / cfg.initial_balance) - 1.0
        dt = time.time() - t0

        logging.info(
            f"Ep {ep}/{cfg.episodes} | steps={steps} | total_reward={total_reward:.6f} | "
            f"final_balance={final_balance:.2f} | ROI={roi_total:.4f} | eps={eps:.4f} | {dt:.1f}s"
        )

        stats.append(
            {
                "episode": ep,
                "steps": steps,
                "total_reward": total_reward,
                "final_balance": final_balance,
                "roi_total": roi_total,
                "epsilon": eps,
                "time_sec": dt,
            }
        )

        # save best
        if final_balance > best_final_balance:
            best_final_balance = final_balance
            torch.save(
                {
                    "state_dim": state_dim,
                    "model_state_dict": q.state_dict(),
                    "config": cfg.__dict__,
                },
                best_path,
            )
            logging.info(f"Saved best checkpoint: {best_path}")

        # periodic save stats
        if ep % 5 == 0:
            pd.DataFrame(stats).to_csv(out_dir / "train_stats.csv", index=False)
            with open(out_dir / "train_config.json", "w", encoding="utf-8") as f:
                json.dump(cfg.__dict__, f, ensure_ascii=False, indent=2)

    # final dumps
    pd.DataFrame(stats).to_csv(out_dir / "train_stats.csv", index=False)
    with open(out_dir / "train_config.json", "w", encoding="utf-8") as f:
        json.dump(cfg.__dict__, f, ensure_ascii=False, indent=2)

    logging.info("=== TRAIN DQN DONE ===")


if __name__ == "__main__":
    main()
