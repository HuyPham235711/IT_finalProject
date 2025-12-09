import importlib.util
import logging
import os
from dataclasses import dataclass
from datetime import datetime
import numpy as np
import pandas as pd
import yaml
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

from .dqn_agent import DQNAgent, Transition


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


def ensure_dirs(*paths):
    for p in paths: os.makedirs(p, exist_ok=True)


def setup_logging(log_file: str):
    logger = logging.getLogger(); logger.setLevel(logging.INFO); logger.handlers = []
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    ch = logging.StreamHandler(); ch.setFormatter(fmt); logger.addHandler(ch)
    fh = logging.FileHandler(log_file, encoding="utf-8"); fh.setFormatter(fmt); logger.addHandler(fh)


def get_engine_from_env(env_var: str) -> Engine:
    conn_str = os.getenv(env_var)
    if not conn_str: raise RuntimeError(f"Environment variable {env_var} not set.")
    return create_engine(conn_str)


def load_train_prices(engine: Engine, schema: str, table: str, order_by: str, cols: list) -> pd.DataFrame:
    col_list = ", ".join([f'"{c}"' for c in cols])
    q = f'SELECT {col_list} FROM "{schema}"."{table}" ORDER BY "{order_by}"'
    df = pd.read_sql(q, engine)
    if "close" not in df.columns:
        raise ValueError("DataFrame must contain 'close' column for env.")
    return df


def dynamic_import(module_path: str, class_name: str):
    spec = importlib.util.spec_from_file_location("rl_env_module", module_path)
    if spec is None or spec.loader is None: raise ImportError(f"Cannot load module from {module_path}")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)  # type: ignore
    if not hasattr(mod, class_name): raise ImportError(f"{class_name} not found in {module_path}")
    return getattr(mod, class_name)


def main():
    cfg = load_config("config/rl_agent_config.yaml")
    ensure_dirs(
        cfg.storage["save_dir"], cfg.storage["checkpoints_dir"],
        os.path.dirname(cfg.storage["train_log_csv"]), os.path.dirname(cfg.storage["train_log_file"])
    )
    setup_logging(cfg.storage["train_log_file"])
    logging.info("Loaded config (DQN).")

    # --- Load train prices from Postgres ---
    engine = get_engine_from_env(cfg.postgres["conn_env"])
    df_train = load_train_prices(
        engine=engine,
        schema=cfg.postgres["schema"],
        table=cfg.postgres["tables"]["ohlcv_train"],
        order_by=cfg.data.get("price_table_order_by", "datetime"),
        cols=cfg.data.get("price_columns", ["datetime", "close"]),
    )
    logging.info(f"Train rows: {len(df_train)} | cols: {list(df_train.columns)}")

    # --- Import env and instantiate exactly per your signature ---
    EnvClass = dynamic_import(cfg.env["module_path"], cfg.env["class_name"])
    env = EnvClass(
        fusion_emb_path=cfg.data["fusion_embeddings_path"],
        ohlcv_df=df_train,
        initial_balance=float(cfg.env.get("initial_balance", 1000.0)),
    )

    # Infer dims from env/gym spaces
    state_dim = int(env.observation_space.shape[0])
    n_actions = int(env.action_space.n)  # Discrete(2): 0=Sell, 1=Buy
    logging.info(f"Env ready | state_dim={state_dim} | n_actions={n_actions}")

    # --- Init Agent ---
    a = cfg.agent
    agent = DQNAgent(
        state_dim=state_dim,
        n_actions=n_actions,
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

    # --- Train ---
    num_episodes = int(a.get("num_episodes", 50))
    max_steps = int(a.get("max_steps_per_episode", 5000))
    csv_path = cfg.storage["train_log_csv"]

    if not os.path.exists(csv_path):
        pd.DataFrame(columns=["timestamp","episode","steps","total_reward","avg_loss","epsilon"]
        ).to_csv(csv_path, index=False)

    best_reward = -1e18
    for ep in range(1, num_episodes + 1):
        state = env.reset()
        total_reward, steps, losses = 0.0, 0, []

        for t in range(max_steps):
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)

            agent.remember(Transition(state, action, float(reward), next_state, bool(done)))
            loss = agent.train_step()
            if loss is not None: losses.append(loss)

            total_reward += float(reward)
            steps += 1
            state = next_state
            if done: break

        avg_loss = float(np.mean(losses)) if losses else np.nan
        eps_now = agent.epsilon()
        logging.info(f"Ep {ep}/{num_episodes} | steps={steps} | reward={total_reward:.4f} | "
                     f"avg_loss={avg_loss:.6f} | epsilon={eps_now:.4f}")

        pd.DataFrame([[datetime.utcnow().isoformat(), ep, steps, total_reward, avg_loss, eps_now]],
            columns=["timestamp","episode","steps","total_reward","avg_loss","epsilon"]
        ).to_csv(csv_path, mode="a", header=False, index=False)

        if total_reward > best_reward:
            best_reward = total_reward
            ckpt = os.path.join(cfg.storage["checkpoints_dir"], f"best_ep{ep}.pt")
            agent.save(ckpt); logging.info(f"Saved best checkpoint: {ckpt}")

    final_ckpt = os.path.join(cfg.storage["checkpoints_dir"], "final.pt")
    agent.save(final_ckpt); logging.info(f"Done. Final model: {final_ckpt}")


if __name__ == "__main__":
    main()
