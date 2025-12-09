# src/rl_agent/train_ppo.py
import os
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import DummyVecEnv

from src.rl_env.trading_env import TradingEnvPPOHybrid  # env reward đơn giản của bạn

# PG_CONN_STR dùng như DQN
os.environ["PG_CONN_STR"] = "postgresql+psycopg2://postgres:123456789@localhost:5432/postgres"


PROJECT_ROOT = Path("E:/TDTu/TAI_LIEU/KY1-NAM5/DU_AN_CNTT")
FUSION_TRAIN_PATH = PROJECT_ROOT / "results" / "fusion_rl" / "train_inference" / "fusion_embeddings.npy"

SCHEMA = "it_final"
TABLE_TRAIN = "processed_ohlcv_train"


def load_ohlcv_train():
    engine = create_engine(os.environ["PG_CONN_STR"])
    q = f'SELECT "datetime", "close" FROM "{SCHEMA}"."{TABLE_TRAIN}" ORDER BY "datetime"'
    df = pd.read_sql(q, engine)
    return df


def make_env():
    df = load_ohlcv_train()
    env = TradingEnvPPOHybrid(
        fusion_emb_path=str(FUSION_TRAIN_PATH),
        ohlcv_df=df,
        initial_balance=1000.0,
    )
    return env


def main():
    log_dir = PROJECT_ROOT / "results" / "rl_agent" / "ppo"
    log_dir.mkdir(parents=True, exist_ok=True)

    # VecEnv (bắt buộc với stable-baselines3, 1 env vẫn dùng DummyVecEnv)
    vec_env = DummyVecEnv([make_env])

    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        learning_rate=3e-4,
        gamma=0.99,
        n_steps=2048,
        batch_size=256,
        tensorboard_log=str(log_dir / "tb"),
    )

    total_steps = 1_000_000  # bạn có thể chỉnh
    model.learn(total_timesteps=total_steps)

    model_path = log_dir / "ppo_trading.zip"
    model.save(model_path)
    print(f"✅ Saved PPO model → {model_path}")


if __name__ == "__main__":
    main()
