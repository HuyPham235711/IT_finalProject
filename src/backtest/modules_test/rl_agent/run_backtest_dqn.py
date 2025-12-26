import os
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sqlalchemy import create_engine

from src.rl_agent.rl_env.dqn_trading_env import TradingEnvDQNBehavioralROI
from src.rl_agent.train_dqn import QNet


# ============================================================
# LOGGING
# ============================================================
def setup_logging(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
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
# LOAD BACKTEST DATA
# ============================================================
def load_backtest_df(engine, schema: str, table: str) -> pd.DataFrame:
    q = f'''
        SELECT datetime, open, high, low, close, volume
        FROM "{schema}"."{table}"
        ORDER BY datetime
    '''
    df = pd.read_sql(q, engine)
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df


# ============================================================
# INFER SEGMENT ID (HOURLY)
# ============================================================
def infer_segment_id(df: pd.DataFrame, max_gap_hours: int = 1) -> np.ndarray:
    dt = df["datetime"]
    gap_hours = dt.diff().dt.total_seconds().fillna(0) / 3600.0
    seg = (gap_hours > max_gap_hours).cumsum().astype(int)
    return seg.values


# ============================================================
# LOAD MODEL
# ============================================================
def load_dqn_checkpoint(ckpt_path: Path, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device)
    state_dim = int(ckpt["state_dim"])
    model = QNet(state_dim).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


# ============================================================
# MAIN
# ============================================================
def main():
    PROJECT_ROOT = Path("E:/TDTu/TAI_LIEU/KY1-NAM5/DU_AN_CNTT")

    # ===== OUTPUT =====
    out_dir = PROJECT_ROOT / "results" / "backtest" / "rl_dqn" / "dqn_behavioral_roi"
    out_dir.mkdir(parents=True, exist_ok=True)

    log_path = PROJECT_ROOT / "logs" / "backtest" / "dqn_behavioral_roi.log"
    setup_logging(log_path)

    logging.info("=== DQN BACKTEST START (Behavioral ROI | ε-debug) ===")

    # ===== POSTGRES =====
    conn = os.environ.get(
        "PG_CONN_STR",
        "postgresql+psycopg2://postgres:123456789@localhost:5432/postgres",
    )
    engine = create_engine(conn)

    schema = "it_final"
    table_backtest = "processed_ohlcv_backtest"

    # ===== LOAD DATA =====
    df = load_backtest_df(engine, schema, table_backtest)

    # ===== LOAD EMBEDDINGS =====
    EMB_PATH = (
        PROJECT_ROOT
        / "results"
        / "backtest"
        / "fusion"
        / "fusion_embeddings_backtest_v2.npy"
    )
    emb = np.load(EMB_PATH)

    # ===== ALIGN LOOKBACK =====
    lookback = len(df) - len(emb)
    if lookback <= 0:
        raise ValueError(f"Invalid lookback={lookback}")

    df = df.iloc[lookback:].reset_index(drop=True)
    prices = df["close"].astype(float).values
    seg = infer_segment_id(df, max_gap_hours=1)

    if len(prices) != len(emb):
        raise ValueError(
            f"After align: prices len {len(prices)} != embeddings len {len(emb)}"
        )

    logging.info(
        f"Backtest aligned | lookback={lookback} | steps={len(prices)}"
    )

    # ===== LOAD MODEL =====
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    CKPT_PATH = (
        PROJECT_ROOT
        / "results"
        / "rl_agent"
        / "dqn"
        / "behavioral_roi"
        / "best.pt"
    )
    model = load_dqn_checkpoint(CKPT_PATH, device)

    # ===== ENV =====
    env = TradingEnvDQNBehavioralROI(
        prices=prices,
        embeddings=emb,
        segment_id=seg,
        initial_balance=1000.0,
        transaction_cost=0.001,
        slippage_rate=0.0005,
        max_step_return=0.03,
        freeze_first_step_of_segment=True,
        roi_scale=1.0,
        trade_penalty=0.0,
        churn_penalty=0.0,
        hold_bonus=0.0,
        episode_length=None,
        start_index=0,
        random_start=False,
        seed=42,
    )

    obs, _ = env.reset()

    EPS_DEBUG = 0.02  # ===== 2% RANDOM ACTION FOR DEBUG =====

    rows = []
    step = 0

    while True:
        # ===== GREEDY ACTION =====
        with torch.no_grad():
            x = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            greedy_action = int(torch.argmax(model(x), dim=1).item())

        # ===== EPSILON DEBUG =====
        if np.random.rand() < EPS_DEBUG:
            action = np.random.randint(0, 3)
        else:
            action = greedy_action

        obs2, reward, done, _, info = env.step(action)

        bal = info["balance"]
        roi_total = (bal / env.initial_balance) - 1.0

        logging.info(
            f"STEP {step} | "
            f"Action={action} (greedy={greedy_action}) | Pos={info['position']} | "
            f"Price={info['price']:.6f}->{info['next_price']:.6f} | "
            f"eff_logret={info['effective_logret']:.6f} | "
            f"PnL={info['pnl']:.6f} | "
            f"Bal={bal:.2f} | ROI={roi_total:.6f}"
        )

        rows.append(
            {
                "step": step,
                "datetime": df.loc[env.i, "datetime"],
                "price": info["price"],
                "next_price": info["next_price"],
                "effective_logret": info["effective_logret"],
                "action": action,
                "greedy_action": greedy_action,
                "position": info["position"],
                "pnl": info["pnl"],
                "balance": bal,
                "roi_total": roi_total,
            }
        )

        obs = obs2
        step += 1

        if done:
            break

    # ===== SAVE =====
    out_csv = out_dir / "backtest_steps.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)

    summary = {
        "final_balance": float(env.balance),
        "roi_total": float((env.balance / env.initial_balance) - 1.0),
        "steps": int(step),
        "lookback": int(lookback),
        "epsilon_debug": EPS_DEBUG,
        "ckpt_path": str(CKPT_PATH),
        "emb_path": str(EMB_PATH),
    }

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    logging.info(
        f"=== DQN BACKTEST DONE | final_balance={summary['final_balance']:.2f} "
        f"| ROI={summary['roi_total']:.6f} ==="
    )


if __name__ == "__main__":
    main()
