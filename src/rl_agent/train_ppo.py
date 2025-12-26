# src/rl_agent/train_ppo.py
import os
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from src.rl_agent.rl_env.ppo_trading_env import TradingEnvPPOHybrid

PROJECT_ROOT = Path("E:/TDTu/TAI_LIEU/KY1-NAM5/DU_AN_CNTT")
SCHEMA = "it_final"
TABLE_TRAIN = "processed_ohlcv_train"
FUSION_PATH = PROJECT_ROOT / "results/fusion_rl/v2/fusion_embeddings_train_v2.npy"

OUT_DIR = PROJECT_ROOT / "results/rl_agent/ppo/v3"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_prices():
    engine = create_engine(os.environ["PG_CONN_STR"])
    df = pd.read_sql(
        f'SELECT close FROM "{SCHEMA}"."{TABLE_TRAIN}" ORDER BY datetime',
        engine,
    )
    return df["close"].to_numpy()


def main():
    prices = load_prices()
    embeddings = np.load(FUSION_PATH)

    n = min(len(prices), len(embeddings))
    prices, embeddings = prices[:n], embeddings[:n]

    def make_env():
        return TradingEnvPPOHybrid(
            prices=prices,
            embeddings=embeddings,
            episode_length=2048,
        )

    env = VecNormalize(DummyVecEnv([make_env]), norm_obs=True, norm_reward=True)

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=1e-4,
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.02,
        device="cpu",
        verbose=1,
    )

    model.learn(total_timesteps=500_000)

    model.save(OUT_DIR / "ppo_trading.zip")
    env.save(OUT_DIR / "vecnormalize.pkl")

    print(" PPO TRAIN DONE")


if __name__ == "__main__":
    main()
