import os
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import create_engine

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from src.rl_agent.rl_env.sac_trading_env import TradingEnvSACHybrid


# ============================================================
# CONFIG – PROJECT
# ============================================================
PROJECT_ROOT = Path("E:/TDTu/TAI_LIEU/KY1-NAM5/DU_AN_CNTT")

SCHEMA = "it_final"
TABLE_TRAIN = "processed_ohlcv_train"

FUSION_TRAIN_PATH = (
    PROJECT_ROOT
    / "results"
    / "fusion_rl"
    / "v2"
    / "fusion_embeddings_train_v2.npy"
)

OUT_DIR = PROJECT_ROOT / "results" / "rl_agent" / "sac" / "v1"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# LOCKED ENV PARAMS (MUST MATCH BACKTEST)
# ============================================================
ENV_CFG = dict(
    # account/limits
    initial_balance=1000.0,
    max_position=1.0,

    # costs
    transaction_cost=0.001,
    slippage_rate=0.0005,
    holding_cost=2e-4,

    # action shaping
    deadband=None,          # IMPORTANT: keep action-deadband off
    max_delta_pos=0.02,
    deadband_delta=0.0005,  # delta-position deadband (hardcore but not "freeze")

    # reward scaling
    alpha=200.0,

    # penalties (hardcore)
    turnover_penalty=4.0,
    pos_penalty=0.0,
    action_penalty=0.05,
    flip_penalty=0.002,

    # dd penalty off unless you explicitly use it
    dd_threshold=0.10,
    dd_penalty=0.0,

    # stabilize
    reward_clip=10.0,

    # obs config (MUST MATCH BACKTEST)
    include_position_in_obs=True,
    include_equity_in_obs=False,

    seed=42,
)

EPISODE_LENGTH = 1024

# VecNormalize (MUST MATCH BACKTEST load)
VNORM_CFG = dict(
    norm_obs=True,
    norm_reward=False,
    clip_obs=10.0,
)

# ============================================================
# LOAD DATA
# ============================================================
def load_train_prices():
    engine = create_engine(os.environ["PG_CONN_STR"])
    df = pd.read_sql(
        f"""
        SELECT datetime, close
        FROM "{SCHEMA}"."{TABLE_TRAIN}"
        ORDER BY datetime
        """,
        engine,
    )
    return df["close"].to_numpy(dtype=np.float64)


def load_train_embeddings():
    return np.load(FUSION_TRAIN_PATH)


# ============================================================
# MAIN
# ============================================================
def main():
    print("=== TRAIN SAC (LOCKED CONFIG) ===")

    prices = load_train_prices()
    embeddings = load_train_embeddings()

    n = min(len(prices), len(embeddings))
    prices = prices[:n]
    embeddings = embeddings[:n]

    print(f"[INFO] Train samples: {n}")
    print(f"[INFO] Embedding dim : {embeddings.shape[1]}")
    print(f"[INFO] ENV_CFG       : {ENV_CFG}")
    print(f"[INFO] EPISODE_LEN   : {EPISODE_LENGTH}")
    print(f"[INFO] VNORM_CFG     : {VNORM_CFG}")

    def make_env():
        return TradingEnvSACHybrid(
            prices=prices,
            embeddings=embeddings,
            episode_length=EPISODE_LENGTH,
            is_backtest=False,
            **ENV_CFG,
        )

    env = DummyVecEnv([make_env])

    env = VecNormalize(
        env,
        **VNORM_CFG,
    )

    model = SAC(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        buffer_size=200_000,
        learning_starts=10_000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=(1, "step"),
        gradient_steps=1,
        ent_coef="auto",
        target_update_interval=1,
        verbose=1,
        device="cpu",  # "cuda" if available
    )

    TOTAL_TIMESTEPS = 600_000
    print(f"[INFO] Start training SAC for {TOTAL_TIMESTEPS} timesteps...")
    model.learn(total_timesteps=TOTAL_TIMESTEPS)

    model_path = OUT_DIR / "sac_trading.zip"
    vnorm_path = OUT_DIR / "vecnormalize.pkl"

    model.save(model_path)
    env.save(vnorm_path)

    print(" TRAIN SAC DONE")
    print(f"   Model        : {model_path}")
    print(f"   VecNormalize : {vnorm_path}")
    print("==================================================")


if __name__ == "__main__":
    main()
