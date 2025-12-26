import os
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import create_engine

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from src.rl_agent.rl_env.ppo_trading_env import TradingEnvPPOHybrid


# ============================================================
# PATHS & CONFIG
# ============================================================
PROJECT_ROOT = Path("E:/TDTu/TAI_LIEU/KY1-NAM5/DU_AN_CNTT")

MODEL_DIR = PROJECT_ROOT / "results" / "rl_agent" / "ppo" / "v3"
MODEL_PATH = MODEL_DIR / "ppo_trading.zip"
VNORM_PATH = MODEL_DIR / "vecnormalize.pkl"

SCHEMA = "it_final"

BACKTEST_PARTS = {
    "part1": {
        "ohlcv_table": "processed_ohlcv_backtest_part1",
        "fusion_emb": "fusion_embeddings_backtest_v2_part1.npy",
    },
    "part2": {
        "ohlcv_table": "processed_ohlcv_backtest_part2",
        "fusion_emb": "fusion_embeddings_backtest_v2_part2.npy",
    },
    "part3": {
        "ohlcv_table": "processed_ohlcv_backtest_part3",
        "fusion_emb": "fusion_embeddings_backtest_v2_part3.npy",
    },
}

OUT_DIR = PROJECT_ROOT / "results" / "backtest" / "rl_ppo"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# HELPERS
# ============================================================
def load_prices(table: str) -> np.ndarray:
    engine = create_engine(os.environ["PG_CONN_STR"])
    df = pd.read_sql(
        f'SELECT close FROM "{SCHEMA}"."{table}" ORDER BY datetime',
        engine,
    )
    return df["close"].to_numpy(dtype=np.float64)


def run_single_backtest(
    part_name: str,
    prices: np.ndarray,
    embeddings: np.ndarray,
    model: PPO,
):
    print(f"\n=== BACKTEST {part_name.upper()} ===")

    n = min(len(prices), len(embeddings))
    prices = prices[:n]
    embeddings = embeddings[:n]

    # --------------------------------------------------------
    # Create env (BACKTEST MODE)
    # --------------------------------------------------------
    env = DummyVecEnv([
        lambda: TradingEnvPPOHybrid(
            prices=prices,
            embeddings=embeddings,
            is_backtest=True,     # <<< QUAN TRỌNG
        )
    ])

    # Load VecNormalize stats (from TRAIN)
    env = VecNormalize.load(VNORM_PATH, env)
    env.training = False
    env.norm_reward = False

    obs = env.reset()

    logs = []
    done = False

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        logs.append(info[0])

    df = pd.DataFrame(logs)

    # --------------------------------------------------------
    # Save CSV
    # --------------------------------------------------------
    csv_path = OUT_DIR / f"backtest_{part_name}.csv"
    df.to_csv(csv_path, index=False)

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------
    metrics = {
        "final_balance": float(df["balance"].iloc[-1]),
        "total_return": float(df["balance"].iloc[-1] / df["balance"].iloc[0] - 1.0),
        "max_drawdown": float(df["drawdown"].max()),
        "avg_position": float(df["position"].mean()),
        "position_std": float(df["position"].std()),
        "n_steps": int(len(df)),
    }

    metrics_path = OUT_DIR / f"metrics_{part_name}.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"[OK] Saved CSV    → {csv_path}")
    print(f"[OK] Saved metric → {metrics_path}")
    print(metrics)

    return df, metrics


# ============================================================
# MAIN
# ============================================================
def main():
    print("=== PPO BACKTEST (3 PARTS – CONTINUOUS) ===")

    # Load trained PPO
    model = PPO.load(MODEL_PATH, device="cpu")

    for part, cfg in BACKTEST_PARTS.items():
        prices = load_prices(cfg["ohlcv_table"])
        emb_path = PROJECT_ROOT / "results" / "backtest" / "fusion" / cfg["fusion_emb"]
        embeddings = np.load(emb_path)

        run_single_backtest(
            part_name=part,
            prices=prices,
            embeddings=embeddings,
            model=model,
        )

    print("\n=== ALL BACKTESTS DONE ===")


if __name__ == "__main__":
    main()
