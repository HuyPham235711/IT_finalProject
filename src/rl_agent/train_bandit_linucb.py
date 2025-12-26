from __future__ import annotations

import os
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Dict

import numpy as np
import pandas as pd
from sqlalchemy import create_engine

from src.rl_agent.rl_env.bandit_trading_env import TradingEnvBanditFusion, BanditCostConfig


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

    return logger


# ============================================================
# TECH FEATURES (computed from raw prices; no dependency on processed tables)
# ============================================================
def compute_sma(close: np.ndarray, window: int = 14) -> np.ndarray:
    close = np.asarray(close, dtype=np.float64)
    out = np.full_like(close, np.nan, dtype=np.float64)
    if len(close) < window:
        return np.nan_to_num(out, nan=0.0)
    csum = np.cumsum(close, dtype=np.float64)
    csum[window:] = csum[window:] - csum[:-window]
    out[window - 1 :] = csum[window - 1 :] / float(window)
    return np.nan_to_num(out, nan=0.0)


def compute_rsi(close: np.ndarray, window: int = 14) -> np.ndarray:
    close = np.asarray(close, dtype=np.float64)
    n = len(close)
    if n < 2:
        return np.zeros(n, dtype=np.float64)

    delta = np.diff(close, prepend=close[0])
    gain = np.maximum(delta, 0.0)
    loss = np.maximum(-delta, 0.0)

    # Wilder smoothing
    rsi = np.zeros(n, dtype=np.float64)
    avg_gain = 0.0
    avg_loss = 0.0

    for i in range(n):
        if i == 0:
            rsi[i] = 50.0
            continue

        if i < window:
            avg_gain = np.mean(gain[1 : i + 1])
            avg_loss = np.mean(loss[1 : i + 1])
        elif i == window:
            avg_gain = np.mean(gain[1 : window + 1])
            avg_loss = np.mean(loss[1 : window + 1])
        else:
            avg_gain = (avg_gain * (window - 1) + gain[i]) / window
            avg_loss = (avg_loss * (window - 1) + loss[i]) / window

        rs = avg_gain / max(avg_loss, 1e-12)
        rsi[i] = 100.0 - (100.0 / (1.0 + rs))

    return np.clip(rsi, 0.0, 100.0)


def make_context(emb: np.ndarray, open_raw: np.ndarray, close_raw: np.ndarray) -> np.ndarray:
    """
    Context = [fusion_embedding (D), bias(1), mom1, sma14_rel, rsi14_norm]

    All are computed at time t (known by decision time if you trade next open)
    - mom1: log(close/open) of current bar, clipped
    - sma14_rel: (close - sma14)/close
    - rsi14_norm: rsi14 / 100
    """
    emb = np.asarray(emb, dtype=np.float32)
    open_raw = np.asarray(open_raw, dtype=np.float64)
    close_raw = np.asarray(close_raw, dtype=np.float64)

    eps = 1e-12
    mom1 = np.log(np.maximum(close_raw, eps) / np.maximum(open_raw, eps))
    mom1 = np.clip(mom1, -0.05, 0.05)  # stabilize

    sma14 = compute_sma(close_raw, window=14)
    sma_rel = (close_raw - sma14) / np.maximum(np.abs(close_raw), eps)
    sma_rel = np.clip(sma_rel, -0.2, 0.2)

    rsi14 = compute_rsi(close_raw, window=14)
    rsi_norm = np.clip(rsi14 / 100.0, 0.0, 1.0)

    bias = np.ones_like(mom1, dtype=np.float64)

    extra = np.stack([bias, mom1, sma_rel, rsi_norm], axis=1).astype(np.float32)
    ctx = np.concatenate([emb, extra], axis=1).astype(np.float32)

    return ctx


# ============================================================
# BANDIT AGENT (Diagonal LinUCB / LinDiagUCB) - FLAT baseline + min_adv + gamma
# ============================================================
@dataclass
class LinUCBConfig:
    alpha: float = 0.6
    ridge: float = 1.0
    tau: float = 0.0
    cooldown: int = 2

    warmup_steps: int = 600
    epsilon: float = 0.02
    tie_eps: float = 1e-12

    min_adv: float = 0.0
    gamma: float = 0.9995

    seed: int = 42


class LinUCB:
    def __init__(self, d: int, cfg: LinUCBConfig, n_actions: int = 3, eps: float = 1e-8):
        self.d = int(d)
        self.cfg = cfg
        self.n_actions = int(n_actions)
        self.eps = float(eps)

        ridge = max(float(cfg.ridge), self.eps)
        self.A_diag = np.ones((self.n_actions, self.d), dtype=np.float64) * ridge
        self.b = np.zeros((self.n_actions, self.d), dtype=np.float64)
        self.theta = np.zeros((self.n_actions, self.d), dtype=np.float64)

        self.cooldown_left = 0
        self.rng = np.random.default_rng(cfg.seed)

    def reset_cooldown(self):
        self.cooldown_left = 0

    def _scores(self, x: np.ndarray) -> np.ndarray:
        x = x.astype(np.float64, copy=False)
        x2 = x * x
        mean = (self.theta * x).sum(axis=1)
        bonus = self.cfg.alpha * np.sqrt(np.maximum((x2 / self.A_diag).sum(axis=1), 0.0))
        s = mean + bonus
        s[0] = 0.0  # FLAT baseline
        return s

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

        # exit if no positive advantage
        if cur_action != 0 and adv_cur <= 0.0:
            return 0

        # enter only if advantage > min_adv
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
        x = x.astype(np.float64, copy=False)
        a = int(a)
        r = float(r)

        g = float(self.cfg.gamma)
        if g < 1.0:
            self.A_diag[a] *= g
            self.b[a] *= g

        x2 = x * x
        self.A_diag[a] += x2
        self.b[a] += r * x
        self.theta[a] = self.b[a] / np.maximum(self.A_diag[a], self.eps)


# ============================================================
# DB + ALIGN
# ============================================================
def get_engine():
    conn = os.getenv("PG_CONN_STR")
    if not conn:
        raise RuntimeError("Missing PG_CONN_STR")
    return create_engine(conn)


def load_prices(schema: str, table: str, ts_col="time_stamp", open_col="open", close_col="close") -> pd.DataFrame:
    eng = get_engine()
    q = f"""
    SELECT {ts_col} AS time_stamp, {open_col} AS open_raw, {close_col} AS close_raw
    FROM {schema}.{table}
    ORDER BY 1
    """
    df = pd.read_sql(q, eng)
    df["time_stamp"] = pd.to_datetime(df["time_stamp"])
    return df


def align_price_with_emb(df_price: pd.DataFrame, emb: np.ndarray, logger: logging.Logger) -> Tuple[pd.DataFrame, np.ndarray]:
    T = int(emb.shape[0])
    if len(df_price) < T:
        raise ValueError(f"price rows {len(df_price)} < emb rows {T}")
    if len(df_price) > T:
        offset = len(df_price) - T
        df_price = df_price.iloc[offset:].reset_index(drop=True)
        logger.info(f"[ALIGN] Dropped first {offset} price rows to match embeddings length T={T}.")
    return df_price, emb


def compute_metrics(equity: np.ndarray, traded: np.ndarray, steps_per_year: int = 365 * 24) -> Dict[str, float]:
    eq = np.asarray(equity, dtype=np.float64)
    if len(eq) < 2:
        return {"final_equity": float(eq[-1]) if len(eq) else 1.0, "sharpe": 0.0, "max_drawdown": 0.0, "turnover_rate": 0.0}
    rets = np.diff(eq, prepend=eq[0]) / np.maximum(eq, 1e-12)
    mu = float(np.mean(rets))
    sigma = float(np.std(rets) + 1e-12)
    sharpe = (mu / sigma) * np.sqrt(steps_per_year)

    peak = np.maximum.accumulate(eq)
    dd = (peak - eq) / np.maximum(peak, 1e-12)
    mdd = float(np.max(dd))

    turnover = float(np.mean(traded)) if len(traded) else 0.0
    return {"final_equity": float(eq[-1]), "sharpe": float(sharpe), "max_drawdown": float(mdd), "turnover_rate": float(turnover)}


def train_linucb(
    df_price: pd.DataFrame,
    emb: np.ndarray,
    cfg: LinUCBConfig,
    costs: BanditCostConfig,
    logger: logging.Logger,
    log_every: int = 5000,
) -> Tuple[LinUCB, int]:
    # Build context features (fusion + bias + techs)
    ctx = make_context(
        emb=emb,
        open_raw=df_price["open_raw"].to_numpy(),
        close_raw=df_price["close_raw"].to_numpy(),
    )
    d = int(ctx.shape[1])

    agent = LinUCB(d=d, cfg=cfg)

    env = TradingEnvBanditFusion(
        open_raw=df_price["open_raw"].to_numpy(),
        close_raw=df_price["close_raw"].to_numpy(),
        embeddings=ctx,  # pass context as "embeddings"
        costs=costs,
        initial_equity=1.0,
    )

    obs, _ = env.reset()
    agent.reset_cooldown()

    equity_hist, traded_hist, reward_hist = [], [], []
    actions_hist, pos_hist = [], []

    cur_action = 0
    done = False
    step = 0

    logger.info(f"Env ready | T={len(df_price)} | context_dim={d} (fusion_dim={emb.shape[1]} + 4 extras)")

    while not done:
        a = agent.select_action(obs, cur_action, step)
        next_obs, r, done, info = env.step(a)

        agent.update(obs, a, r)

        equity_hist.append(info["equity"])
        traded_hist.append(info["traded"])
        reward_hist.append(float(r))
        actions_hist.append(int(info["action"]))
        pos_hist.append(int(info["pos"]))

        obs = next_obs
        cur_action = a
        step += 1

        if (step % log_every == 0) or done:
            met = compute_metrics(np.asarray(equity_hist), np.asarray(traded_hist))
            a_counts = dict(pd.Series(actions_hist).value_counts().sort_index())
            p_counts = dict(pd.Series(pos_hist).value_counts().sort_index())

            logger.info(
                f"TRAIN step={step} | equity={met['final_equity']:.4f} | sharpe={met['sharpe']:.3f} | "
                f"mdd={met['max_drawdown']:.3f} | turnover={met['turnover_rate']:.3f} | "
                f"avg_reward={float(np.mean(reward_hist)):.8f} | "
                f"action_counts={a_counts} | pos_counts={p_counts}"
            )

    return agent, d


def save_agent(path: Path, agent: LinUCB, cfg: LinUCBConfig, feature_dim: int, logger: logging.Logger):
    path.parent.mkdir(parents=True, exist_ok=True)

    cfg5 = np.array([cfg.alpha, cfg.ridge, cfg.tau, cfg.cooldown, cfg.seed], dtype=np.float64)
    # ext: [alpha,ridge,tau,cooldown,seed,warmup,epsilon,tie_eps,min_adv,gamma]
    cfg_ext = np.array(
        [cfg.alpha, cfg.ridge, cfg.tau, cfg.cooldown, cfg.seed, cfg.warmup_steps, cfg.epsilon, cfg.tie_eps, cfg.min_adv, cfg.gamma],
        dtype=np.float64
    )

    np.savez_compressed(
        str(path),
        A_diag=agent.A_diag,
        b=agent.b,
        cfg=cfg5,
        cfg_ext=cfg_ext,
        feature_dim=np.array([feature_dim], dtype=np.int64),
        model_type=np.array(["lindiagucb_pa1_ctx"], dtype="<U32"),
    )
    logger.info(f"Saved checkpoint: {path}")


def main():
    os.environ.setdefault("PG_CONN_STR", "postgresql+psycopg2://postgres:123456789@localhost:5432/postgres")

    schema = "it_final"
    PROJECT_ROOT = Path("E:/TDTu/TAI_LIEU/KY1-NAM5/DU_AN_CNTT")

    log_path = PROJECT_ROOT / "logs" / "rl_agent" / "bandit_fusion" / "train_bandit_linucb.log"
    logger = setup_logging(log_path)

    logger.info("=== TRAIN BANDIT LinDiagUCB (PA1 ctx: fusion + tech features) ===")

    price_table = "ohlcv_train"  # raw prices table (your current setup)
    emb_path = PROJECT_ROOT / "results" / "fusion_rl" / "v2" / "fusion_embeddings_train_v2.npy"

    logger.info(f"price_table={schema}.{price_table}")
    logger.info(f"emb_path={emb_path}")

    df_price = load_prices(schema, price_table)
    emb = np.load(str(emb_path)).astype(np.float32)
    df_price, emb = align_price_with_emb(df_price, emb, logger)

    costs = BanditCostConfig(switch_cost=0.0006, hold_cost=0.0)

    # Practical hyperparams for your setting
    cfg = LinUCBConfig(
        alpha=0.6,
        ridge=1.0,
        tau=0.0,
        cooldown=2,        # helps avoid flip/churn
        warmup_steps=600,
        epsilon=0.02,
        tie_eps=1e-12,
        min_adv=0.0,
        gamma=0.9995,
        seed=42,
    )

    logger.info(f"cfg={cfg}")
    logger.info(f"costs={costs}")
    logger.info(f"aligned_rows={len(df_price)} | emb_shape={emb.shape}")

    agent, ctx_dim = train_linucb(df_price, emb, cfg, costs, logger, log_every=5000)

    ckpt = PROJECT_ROOT / "results" / "rl_agent" / "bandit_fusion" / "linucb_fusion_train.npz"
    save_agent(ckpt, agent, cfg, feature_dim=ctx_dim, logger=logger)

    logger.info("=== DONE ===")


if __name__ == "__main__":
    main()
