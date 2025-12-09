import os
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from stable_baselines3 import PPO
from sqlalchemy import create_engine

# ====== ENV CLASS IMPORT ======
from src.rl_env.trading_env import TradingEnvPPOHybrid


# ============================================================
# LOGGING
# ============================================================
def setup_logging(path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers = []

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    fh = logging.FileHandler(path, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)


# ============================================================
# MAIN BACKTEST
# ============================================================
def main():
    PROJECT_ROOT = Path("E:/TDTu/TAI_LIEU/KY1-NAM5/DU_AN_CNTT")

    # ========== LOGGING ==========
    out_dir = PROJECT_ROOT / "results" / "backtest" / "rl_ppo"
    out_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(str(out_dir / "backtest_ppo.log"))

    logging.info("=== PPO BACKTEST START ===")

    # ============================================================
    # 1) Hard-coded config
    # ============================================================
    PG_CONN_STR = "postgresql+psycopg2://postgres:123456789@localhost:5432/postgres"
    FUSION_PATH = PROJECT_ROOT / "results/backtest/fusion/fusion_embeddings_backtest.npy"
    OHLVC_SCHEMA = "it_final"
    OHLVC_TABLE = "processed_ohlcv_backtest"
    CHECKPOINT_PATH = PROJECT_ROOT / "results/rl_agent/ppo/ppo_trading.zip"

    # ============================================================
    # 2) LOAD OUT-OF-SAMPLE (OOS) PRICE DATA
    # ============================================================
    engine = create_engine(PG_CONN_STR)
    q = f'SELECT "datetime","close" FROM "{OHLVC_SCHEMA}"."{OHLVC_TABLE}" ORDER BY "datetime"'
    df_bt = pd.read_sql(q, engine)
    logging.info(f"OOS rows={len(df_bt)}")

    # ============================================================
    # 3) INIT ENVIRONMENT
    # ============================================================
    env = TradingEnvPPOHybrid(
        fusion_emb_path=str(FUSION_PATH),
        ohlcv_df=df_bt,
        initial_balance=1000.0,
    )

    # ============================================================
    # 4) LOAD TRAINED PPO MODEL
    # ============================================================
    logging.info(f"Loading PPO model: {CHECKPOINT_PATH}")
    model = PPO.load(str(CHECKPOINT_PATH))

    # ============================================================
    # 5) RUN BACKTEST
    # ============================================================
    obs = env.reset()
    done = False
    total_reward = 0.0

    logging.info("=== TRADING START ===")

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)

        total_reward += reward

        # ⭐ LOG FULL FINANCIAL METRICS ⭐
        logging.info(
            f"STEP {info['step']:04d} | "
            f"Price={info['price']:.4f} | Next={info['price_next']:.4f} | "
            f"Δ={info['delta']:.6f} | TrueMove={info['true_move']} | "
            f"Action={info['action']} | Move={info['action_move']} | "
            f"Reward={info['reward']:.6f} | "
            f"PnL={info['pnl']:.6f} | ROI={info['roi']:.6f} | "
            f"Bal={info['balance']:.2f}"
        )

    logging.info(f"=== BACKTEST DONE — Total Reward={total_reward:.4f} ===")
    logging.info(f"FINAL BALANCE = {info['balance']:.2f}")
    logging.info(f"FINAL ROI     = {(info['balance'] / 1000.0 - 1) * 100:.2f}%")

    # ============================================================
    # SAVE METRICS
    # ============================================================
    metrics = {
        "total_reward": float(total_reward),
        "steps": int(info["step"]),
        "final_balance": float(info["balance"]),
        "total_roi_percent": float((info["balance"] / 1000.0 - 1) * 100),
    }
    pd.Series(metrics).to_json(out_dir / "backtest_metrics_oos.json", indent=2)

    logging.info("Saved metrics JSON")


if __name__ == "__main__":
    main()
