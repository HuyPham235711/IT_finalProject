# src/rl_agent/train_bandit_full_linucb.py
from __future__ import annotations

import os
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Dict

import numpy as np
import pandas as pd
from sqlalchemy import create_engine

# ===== NEW ENV + AGENT =====
from src.rl_agent.rl_env.bandit_full_linucb_trading_env import (
    TradingEnvBanditFusionV2,
    BanditCostConfigV2,
)
from src.rl_agent.agents.linucb_full import LinUCBFull, LinUCBFullConfig


# ============================================================
# LOGGING
# ============================================================
def setup_logging(log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("train_full_linucb")
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
# TECH FEATURES (Y HỆT BẢN CŨ)
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
    Context = [fusion_embedding (D), bias, mom1, sma14_rel, rsi14_norm]
    """
    emb = np.asarray(emb, dtype=np.float32)
    open_raw = np.asarray(open_raw, dtype=np.float64)
    close_raw = np.asarray(close_raw, dtype=np.float64)

    eps = 1e-12
    mom1 = np.log(np.maximum(close_raw, eps) / np.maximum(open_raw, eps))
    mom1 = np.clip(mom1, -0.05, 0.05)

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
# DB + ALIGN
# ============================================================
def get_engine():
    conn = os.getenv("PG_CONN_STR")
    if not conn:
        raise RuntimeError("Missing PG_CONN_STR")
    return create_engine(conn)


def load_prices(schema: str, table: str) -> pd.DataFrame:
    eng = get_engine()
    q = f"""
    SELECT time_stamp, open AS open_raw, close AS close_raw
    FROM {schema}.{table}
    ORDER BY 1
    """
    df = pd.read_sql(q, eng)
    df["time_stamp"] = pd.to_datetime(df["time_stamp"])
    return df


def align_price_with_emb(df_price: pd.DataFrame, emb: np.ndarray, logger: logging.Logger):
    T = emb.shape[0]
    if len(df_price) > T:
        offset = len(df_price) - T
        df_price = df_price.iloc[offset:].reset_index(drop=True)
        logger.info(f"[ALIGN] Dropped first {offset} price rows.")
    return df_price, emb


# ============================================================
# METRICS
# ============================================================
def compute_metrics(equity: np.ndarray, traded: np.ndarray, steps_per_year=365 * 24):
    eq = np.asarray(equity, dtype=np.float64)
    if len(eq) < 2:
        return {"final_equity": float(eq[-1]), "sharpe": 0.0, "mdd": 0.0}

    rets = np.diff(eq, prepend=eq[0]) / np.maximum(eq, 1e-12)
    sharpe = np.mean(rets) / (np.std(rets) + 1e-12) * np.sqrt(steps_per_year)

    peak = np.maximum.accumulate(eq)
    mdd = np.max((peak - eq) / np.maximum(peak, 1e-12))

    return {
        "final_equity": float(eq[-1]),
        "sharpe": float(sharpe),
        "mdd": float(mdd),
        "turnover": float(np.mean(traded)),
    }


# ============================================================
# TRAIN LOOP
# ============================================================
def train_full_linucb(
    df_price: pd.DataFrame,
    emb: np.ndarray,
    cfg: LinUCBFullConfig,
    costs: BanditCostConfigV2,
    logger: logging.Logger,
    log_every: int = 5000,
):
    ctx = make_context(
        emb=emb,
        open_raw=df_price["open_raw"].to_numpy(),
        close_raw=df_price["close_raw"].to_numpy(),
    )
    d = ctx.shape[1]

    agent = LinUCBFull(d=d, cfg=cfg)

    env = TradingEnvBanditFusionV2(
        open_raw=df_price["open_raw"].to_numpy(),
        close_raw=df_price["close_raw"].to_numpy(),
        embeddings=ctx,
        costs=costs,
        initial_equity=1.0,
    )

    obs, _ = env.reset()
    agent.reset_cooldown()

    equity_hist, traded_hist = [], []
    actions_hist, pos_hist = [], []

    cur_action = 0
    step = 0
    done = False

    logger.info(f"Env ready | T={len(df_price)} | context_dim={d}")

    while not done:
        a = agent.select_action(obs, cur_action, step)
        next_obs, r, done, info = env.step(a)

        agent.update(obs, a, r)

        equity_hist.append(info["equity"])
        traded_hist.append(info["traded"])
        actions_hist.append(info["action"])
        pos_hist.append(info["pos"])

        obs = next_obs
        cur_action = a
        step += 1

        if (step % log_every == 0) or done:
            met = compute_metrics(np.asarray(equity_hist), np.asarray(traded_hist))
            logger.info(
                f"TRAIN step={step} | equity={met['final_equity']:.4f} "
                f"| sharpe={met['sharpe']:.3f} | mdd={met['mdd']:.3f} "
                f"| turnover={met['turnover']:.3f}"
            )

    return agent, d


# ============================================================
# SAVE
# ============================================================
def save_agent(path: Path, agent: LinUCBFull, cfg: LinUCBFullConfig, feature_dim: int, logger):
    path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        str(path),
        A=agent.A,
        A_inv=agent.A_inv,
        b=agent.b,
        theta=agent.theta,
        feature_dim=np.array([feature_dim], dtype=np.int64),
        cfg=np.array(
            [
                cfg.alpha,
                cfg.ridge,
                cfg.tau,
                cfg.cooldown,
                cfg.warmup_steps,
                cfg.epsilon,
                cfg.min_adv,
                cfg.gamma,
            ],
            dtype=np.float64,
        ),
        model_type=np.array(["linucb_full_ctx"], dtype="<U32"),
    )
    logger.info(f"Saved checkpoint: {path}")


# ============================================================
# MAIN
# ============================================================
def main():
    os.environ.setdefault(
        "PG_CONN_STR",
        "postgresql+psycopg2://postgres:123456789@localhost:5432/postgres",
    )

    PROJECT_ROOT = Path("E:/TDTu/TAI_LIEU/KY1-NAM5/DU_AN_CNTT")
    schema = "it_final"

    log_path = PROJECT_ROOT / "logs/rl_agent/bandit_fusion/train_full_linucb.log"
    logger = setup_logging(log_path)

    price_table = "ohlcv_train"
    emb_path = PROJECT_ROOT / "results/fusion_rl/v2/fusion_embeddings_train_v2.npy"

    df_price = load_prices(schema, price_table)
    emb = np.load(emb_path).astype(np.float32)
    df_price, emb = align_price_with_emb(df_price, emb, logger)

    costs = BanditCostConfigV2(
        switch_cost=0.0006,
        hold_cost=0.00002,
        flat_bonus=0.00001,
    )

    cfg = LinUCBFullConfig(
        alpha=0.6,
        ridge=1.0,
        cooldown=2,
        warmup_steps=600,
        epsilon=0.02,
        min_adv=0.0,
        gamma=1.0,
        seed=42,
    )

    agent, d = train_full_linucb(df_price, emb, cfg, costs, logger)

    ckpt = PROJECT_ROOT / "results/rl_agent/bandit_fusion/linucb_full_fusion_train_v1.npz"
    save_agent(ckpt, agent, cfg, d, logger)

    logger.info("=== DONE (FULL LinUCB) ===")


if __name__ == "__main__":
    main()
