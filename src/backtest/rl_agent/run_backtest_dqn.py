import importlib.util
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

from src.rl_agent.dqn_agent import DQNAgent

os.environ["PG_CONN_STR"] = "postgresql+psycopg2://postgres:123456789@localhost:5432/postgres"


# ============================================================
# 1. CONFIG + HELPERS
# ============================================================

@dataclass
class Config:
    agent: dict
    env: dict
    data: dict
    storage: dict
    postgres: dict


def load_config(path: str) -> Config:
    with open(path, "r", encoding="utf-8") as f:
        return Config(**yaml.safe_load(f))


def setup_logging(log_file: str):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers = []

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)


def get_engine_from_env(env_var: str) -> Engine:
    conn_str = os.getenv(env_var)
    if not conn_str:
        raise RuntimeError(f"{env_var} not set")
    return create_engine(conn_str)


def load_prices(engine: Engine, schema: str, table: str, order_by: str, cols: list) -> pd.DataFrame:
    col_list = ", ".join([f'"{c}"' for c in cols])
    q = f'SELECT {col_list} FROM "{schema}"."{table}" ORDER BY "{order_by}"'
    df = pd.read_sql(q, engine)
    return df


def dynamic_import(module_path: str, class_name: str):
    spec = importlib.util.spec_from_file_location("rl_env_module", module_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, class_name)


# ============================================================
# 2. GREEDY ACTION
# ============================================================

def select_greedy_action(agent: DQNAgent, state: np.ndarray) -> int:
    import torch
    state_t = torch.tensor(state, dtype=torch.float32, device=agent.device).unsqueeze(0)
    with torch.no_grad():
        q_values = agent.policy_net(state_t)
        action = int(q_values.argmax(dim=1).item())
    return action


# ============================================================
# 3. MAIN BACKTEST
# ============================================================

def main():
    PROJECT_ROOT = Path("E:/TDTu/TAI_LIEU/KY1-NAM5/DU_AN_CNTT")
    cfg = load_config(str(PROJECT_ROOT / "config" / "rl_agent_config.yaml"))

    # Logging
    backtest_dir = PROJECT_ROOT / "results" / "backtest" / "rl_dqn"
    backtest_dir.mkdir(parents=True, exist_ok=True)
    log_file = backtest_dir / "backtest_dqn.log"
    setup_logging(str(log_file))

    logging.info("=== Running DQN BACKTEST (OOS) ===")

    # Load OOS data
    engine = get_engine_from_env(cfg.postgres["conn_env"])
    df_bt = load_prices(
        engine,
        cfg.postgres["schema"],
        cfg.postgres["tables"]["ohlcv_backtest"],
        cfg.data.get("price_table_order_by", "datetime"),
        cfg.data.get("price_columns", ["datetime", "close"]),
    )
    logging.info(f"OOS rows={len(df_bt)}")

    # Load fusion embeddings
    fusion_path = PROJECT_ROOT / "results" / "backtest" / "fusion" / "fusion_embeddings_backtest.npy"
    logging.info(f"Using fusion: {fusion_path}")

    EnvClass = dynamic_import(cfg.env["module_path"], cfg.env["class_name"])
    env = EnvClass(
        fusion_emb_path=str(fusion_path),
        ohlcv_df=df_bt,
        initial_balance=float(cfg.env.get("initial_balance", 1000.0)),
    )

    # Init DQN Agent
    a = cfg.agent
    agent = DQNAgent(
        state_dim=int(env.observation_space.shape[0]),
        n_actions=int(env.action_space.n),
        learning_rate=float(a.get("learning_rate", 1e-3)),
        gamma=float(a.get("gamma", 0.99)),
        epsilon_start=float(a.get("epsilon_start", 1.0)),
        epsilon_end=float(a.get("epsilon_end", 0.05)),
        epsilon_decay_steps=int(a.get("epsilon_decay_steps", 20000)),
        memory_size=int(a.get("memory_size", 100000)),
        batch_size=int(a.get("batch_size", 128)),
        target_update_every=int(a.get("target_update_every", 1000)),
        gradient_clip_norm=float(a.get("gradient_clip_norm", 1.0)),
        seed=int(a.get("seed", 42)),
    )

    ckpt_path = Path(cfg.storage["checkpoints_dir"]) / "final.pt"
    agent.load(str(ckpt_path))
    logging.info(f"Loaded checkpoint: {ckpt_path}")

    # Run backtest
    state = env.reset()
    done = False
    total_reward = 0.0
    steps = 0
    last_info = None

    logging.info("=== START TRADING ===")

    while not done:
        action = select_greedy_action(agent, state)
        next_state, reward, done, info = env.step(action)
        step_idx = env.current_step

        # === LOG EVERY TRADE STEP ===
        logging.info(
            f"STEP {info['step']:04d} | "
            f"Price={info['price']:.4f} | "
            f"Next={info['price_next']:.4f} | "
            f"Δ={info['delta']:.6f} | "
            f"TrueMove={info['true_move']} | "
            f"Action={info['action']} | "
            f"Move={info['action_move']} | "
            f"Reward={info['reward']:.6f}"
        )



        total_reward += float(reward)
        steps += 1
        state = next_state
        last_info = info

    logging.info(f"=== BACKTEST DONE: steps={steps}, total_reward={total_reward:.4f} ===")

    # Save final metrics
    metrics = {k: float(v) for k, v in last_info.items() if isinstance(v, (int, float))}
    metrics["total_reward"] = float(total_reward)
    metrics["steps"] = steps

    metrics_path = backtest_dir / "backtest_metrics_oos.json"
    pd.Series(metrics).to_json(metrics_path, indent=2)
    logging.info(f"Saved metrics: {metrics_path}")


if __name__ == "__main__":
    main()
