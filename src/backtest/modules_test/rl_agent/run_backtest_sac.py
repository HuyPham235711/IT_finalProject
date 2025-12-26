import os
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import create_engine

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from src.rl_agent.rl_env.sac_trading_env import TradingEnvSACHybrid


# ============================================================
# CONFIG
# ============================================================
PROJECT_ROOT = Path("E:/TDTu/TAI_LIEU/KY1-NAM5/DU_AN_CNTT")
SCHEMA = "it_final"

MODEL_DIR = PROJECT_ROOT / "results" / "rl_agent" / "sac" / "v1"
MODEL_PATH = MODEL_DIR / "sac_trading.zip"
VNORM_PATH = MODEL_DIR / "vecnormalize.pkl"

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

OUT_DIR = PROJECT_ROOT / "results" / "backtest" / "rl_sac"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ===== DEBUG =====
DEBUG = True
DEBUG_STEP_INTERVAL = 200
DEBUG_CLIP_THRESH = 9.9
SENSITIVITY_EPS_LIST = [0.01, 0.05]
SENSITIVITY_SAMPLES = 256
BASELINE_CONST_POS_LIST = [-0.40, 0.0, +0.40]

# ============================================================
# LOCKED ENV PARAMS (MUST MATCH TRAIN)
# ============================================================
ENV_CFG = dict(
    initial_balance=1000.0,
    max_position=1.0,

    transaction_cost=0.001,
    slippage_rate=0.0005,
    holding_cost=2e-4,

    deadband=None,          # action-deadband OFF
    max_delta_pos=0.02,
    deadband_delta=0.0005,  # delta deadband ON

    alpha=200.0,

    turnover_penalty=4.0,
    pos_penalty=0.0,
    action_penalty=0.05,
    flip_penalty=0.002,

    dd_threshold=0.10,
    dd_penalty=0.0,

    reward_clip=10.0,

    include_position_in_obs=True,
    include_equity_in_obs=False,
)

# ============================================================
# IO
# ============================================================
def load_prices(table: str) -> np.ndarray:
    engine = create_engine(os.environ["PG_CONN_STR"])
    df = pd.read_sql(
        f'SELECT datetime, close FROM "{SCHEMA}"."{table}" ORDER BY datetime',
        engine,
    )
    return df["close"].to_numpy(dtype=np.float64)


# ============================================================
# BASELINE (delta-limited constant target position)
# ============================================================
def simulate_constant_position_delta_action(
    prices: np.ndarray,
    const_pos: float,
    *,
    initial_balance: float,
    transaction_cost: float,
    slippage_rate: float,
    holding_cost: float,
    max_position: float,
    max_delta_pos: float,
    deadband_delta: float,
    n_steps: int | None = None,
) -> dict:
    prices = np.asarray(prices, dtype=np.float64)
    if len(prices) < 2:
        return {"final_equity": initial_balance, "total_return": 0.0, "max_drawdown": 0.0}

    if n_steps is None:
        n_steps = max(1, len(prices) - 2)
    n_steps = int(min(n_steps, len(prices) - 1))

    cost_rate = float(transaction_cost + slippage_rate)

    equity = float(initial_balance)
    peak = float(initial_balance)
    pos = 0.0
    max_dd = 0.0

    const_pos = float(np.clip(const_pos, -max_position, +max_position))

    for t in range(n_steps):
        p0 = float(prices[t])
        p1 = float(prices[t + 1])
        ret = 0.0 if p0 <= 0 else (p1 - p0) / p0

        desired = const_pos - pos
        dpos = float(np.clip(desired, -max_delta_pos, +max_delta_pos))

        # apply delta deadband (same logic style as env)
        if deadband_delta > 0 and abs(dpos) < deadband_delta:
            dpos = 0.0

        target_pos = float(np.clip(pos + dpos, -max_position, +max_position))
        dpos = target_pos - pos  # recompute after clip

        cost = cost_rate * abs(dpos)
        hold = float(holding_cost) * abs(target_pos)

        step_pnl = (target_pos * ret) - cost - hold
        equity *= (1.0 + step_pnl)

        peak = max(peak, equity)
        dd = 0.0 if peak <= 0 else (peak - equity) / peak
        max_dd = max(max_dd, dd)

        pos = target_pos

        if (not np.isfinite(equity)) or equity <= 0:
            break

    return {
        "final_equity": float(equity),
        "total_return": float(equity / max(initial_balance, 1e-9) - 1.0),
        "max_drawdown": float(max_dd),
    }


# ============================================================
# DEBUG HELPERS
# ============================================================
def _obs_stats(obs: np.ndarray) -> tuple[float, float, float]:
    x = np.asarray(obs, dtype=np.float32).reshape(-1)
    omin = float(np.min(x)) if x.size else 0.0
    omax = float(np.max(x)) if x.size else 0.0
    clip_rate = float(np.mean(np.abs(x) >= DEBUG_CLIP_THRESH)) if x.size else 0.0
    return omin, omax, clip_rate


def debug_sensitivity(model: SAC, obs: np.ndarray, eps: float, n: int, seed: int = 0) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    obs = np.asarray(obs, dtype=np.float32)

    base_action, _ = model.predict(obs, deterministic=True)
    base_a = float(np.asarray(base_action).reshape(-1)[0])

    deltas = []
    for _ in range(int(n)):
        noise = rng.normal(0.0, 1.0, size=obs.shape).astype(np.float32)
        obs_pert = obs + float(eps) * noise
        a2, _ = model.predict(obs_pert, deterministic=True)
        a2 = float(np.asarray(a2).reshape(-1)[0])
        deltas.append(abs(a2 - base_a))

    deltas = np.asarray(deltas, dtype=np.float64)
    return base_a, float(deltas.mean()), float(deltas.max())


# ============================================================
# BACKTEST (single part)
# ============================================================
def run_single(part: str, prices: np.ndarray, embeddings: np.ndarray, model: SAC):
    print(f"\n=== BACKTEST {part.upper()} (SAC) ===")

    n = min(len(prices), len(embeddings))
    prices = prices[:n]
    embeddings = embeddings[:n]

    episode_length = max(1, n - 2)

    # ---- BASELINES ----
    if DEBUG:
        for cp in BASELINE_CONST_POS_LIST:
            b = simulate_constant_position_delta_action(
                prices,
                cp,
                initial_balance=ENV_CFG["initial_balance"],
                transaction_cost=ENV_CFG["transaction_cost"],
                slippage_rate=ENV_CFG["slippage_rate"],
                holding_cost=ENV_CFG["holding_cost"],
                max_position=ENV_CFG["max_position"],
                max_delta_pos=ENV_CFG["max_delta_pos"],
                deadband_delta=ENV_CFG["deadband_delta"],
                n_steps=episode_length,
            )
            print(
                f"[BASELINE] const_pos={cp:+.2f} "
                f"final={b['final_equity']:.3f} "
                f"ret={b['total_return']*100:.4f}% "
                f"mdd={b['max_drawdown']*100:.4f}%"
            )

    # ---- ENV ----
    env = DummyVecEnv([
        lambda: TradingEnvSACHybrid(
            prices=prices,
            embeddings=embeddings,
            episode_length=episode_length,
            is_backtest=True,
            start_index=0,
            end_index=n - 1,
            **ENV_CFG,
        )
    ])

    env = VecNormalize.load(VNORM_PATH, env)
    env.training = False
    env.norm_reward = False

    obs = env.reset()

    # ---- DEBUG obs0 ----
    if DEBUG:
        print(f"[SANITY] ENV_CFG: deadband={ENV_CFG['deadband']} deadband_delta={ENV_CFG['deadband_delta']} max_delta_pos={ENV_CFG['max_delta_pos']}")
        omin, omax, clip_rate = _obs_stats(obs)
        print(f"[DEBUG] obs0 min={omin:.4f} max={omax:.4f} clip_rate(|obs|>={DEBUG_CLIP_THRESH})={clip_rate*100:.4f}%")

        for eps in SENSITIVITY_EPS_LIST:
            base_a, mean_da, max_da = debug_sensitivity(model, obs, eps=float(eps), n=SENSITIVITY_SAMPLES, seed=0)
            print(
                f"[DEBUG] sensitivity eps={eps:.2f}: base_action={base_a:+.6f} "
                f"| mean|Δa|={mean_da:.6e} max|Δa|={max_da:.6e}"
            )

    # ---- ROLLOUT ----
    logs = []
    obs_min_overall = +np.inf
    obs_max_overall = -np.inf
    clip_rates = []

    raw_actions = []
    desired_deltas = []
    delta_pos_list = []
    pos_list = []

    while True:
        action, _ = model.predict(obs, deterministic=True)  # raw action from policy
        raw_a = float(np.asarray(action).reshape(-1)[0])
        raw_actions.append(raw_a)

        obs, rewards, dones, infos = env.step(action)

        info = infos[0]
        logs.append(info)

        # pull env-reported quantities (post-deadband_delta)
        desired_delta = float(info.get("desired_delta", 0.0))
        dpos = float(info.get("delta_pos", 0.0))
        pos = float(info.get("position", 0.0))

        desired_deltas.append(desired_delta)
        delta_pos_list.append(dpos)
        pos_list.append(pos)

        if DEBUG:
            omin, omax, cr = _obs_stats(obs)
            obs_min_overall = min(obs_min_overall, omin)
            obs_max_overall = max(obs_max_overall, omax)
            clip_rates.append(cr)

            step = int(info.get("t", len(logs) - 1))
            if step == 0 or (DEBUG_STEP_INTERVAL > 0 and step % DEBUG_STEP_INTERVAL == 0):
                cost = float(info.get("cost", np.nan))
                pnl = float(info.get("step_pnl", np.nan))
                eq = float(info.get("equity", np.nan))
                dd = float(info.get("drawdown", np.nan))
                print(
                    f"[DEBUG] step={step:5d} obs_min={omin:.4f} obs_max={omax:.4f} clip_rate={cr*100:.4f}% "
                    f"| raw_a={raw_a:+.6f} desiredΔ={desired_delta:+.6f} dpos={dpos:+.6f} pos={pos:+.6f} "
                    f"cost={cost:.6g} pnl={pnl:.6g} eq={eq:.3f} dd={dd*100:.4f}%"
                )

        if bool(dones[0]):
            break

    df = pd.DataFrame(logs)

    csv_path = OUT_DIR / f"backtest_{part}.csv"
    df.to_csv(csv_path, index=False)

    eq_col = "equity" if "equity" in df.columns else ("balance" if "balance" in df.columns else None)
    if eq_col is None:
        raise KeyError("Backtest log must contain 'equity' or 'balance'")

    raw_actions_np = np.asarray(raw_actions, dtype=np.float64)
    desired_deltas_np = np.asarray(desired_deltas, dtype=np.float64)
    delta_pos_np = np.asarray(delta_pos_list, dtype=np.float64)
    pos_np = np.asarray(pos_list, dtype=np.float64)

    metrics = {
        "final_equity": float(df[eq_col].iloc[-1]),
        "total_return": float(df[eq_col].iloc[-1] / df[eq_col].iloc[0] - 1.0),
        "max_drawdown": float(df["drawdown"].max()) if "drawdown" in df.columns else 0.0,
        "avg_position": float(np.mean(pos_np)) if len(pos_np) else 0.0,
        "position_std": float(np.std(pos_np)) if len(pos_np) else 0.0,

        # raw action from model
        "raw_action_mean": float(np.mean(raw_actions_np)) if len(raw_actions_np) else 0.0,
        "raw_action_abs_mean": float(np.mean(np.abs(raw_actions_np))) if len(raw_actions_np) else 0.0,

        # applied movement
        "desired_delta_abs_mean": float(np.mean(np.abs(desired_deltas_np))) if len(desired_deltas_np) else 0.0,
        "delta_pos_abs_mean": float(np.mean(np.abs(delta_pos_np))) if len(delta_pos_np) else 0.0,
        "pct_desired_delta_is_zero": float(np.mean(np.abs(desired_deltas_np) < 1e-12)) if len(desired_deltas_np) else 0.0,

        "n_steps": int(len(df)),
        "deadband_action_used": float(0.0 if ENV_CFG["deadband"] is None else ENV_CFG["deadband"]),
        "deadband_delta_used": float(ENV_CFG["deadband_delta"]),
        "max_delta_pos": float(ENV_CFG["max_delta_pos"]),
    }

    metrics_path = OUT_DIR / f"metrics_{part}.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # ---- DEBUG SUMMARY ----
    if DEBUG:
        clip_rates_np = np.asarray(clip_rates, dtype=np.float64) if len(clip_rates) else np.asarray([0.0])

        print(
            f"[DEBUG] raw action stats: mean={metrics['raw_action_mean']:+.6f} "
            f"abs_mean={metrics['raw_action_abs_mean']:.6f} "
            f"min={float(np.min(raw_actions_np)):+.6f} max={float(np.max(raw_actions_np)):+.6f}"
        )
        print(
            f"[DEBUG] desiredΔ abs_mean={metrics['desired_delta_abs_mean']:.6f} "
            f"| pct_desiredΔ_is_zero={metrics['pct_desired_delta_is_zero']*100:.2f}%"
        )
        print(
            f"[DEBUG] clip_rate summary: mean={float(np.mean(clip_rates_np))*100:.4f}% "
            f"max={float(np.max(clip_rates_np))*100:.4f}% | "
            f"obs_min(min)={float(obs_min_overall if np.isfinite(obs_min_overall) else 0.0):.4f} "
            f"obs_max(max)={float(obs_max_overall if np.isfinite(obs_max_overall) else 0.0):.4f}"
        )

    print(f"[OK] Saved CSV    : {csv_path}")
    print(f"[OK] Saved metrics: {metrics_path}")
    print(json.dumps(metrics, indent=2))

    return df, metrics


# ============================================================
# MAIN
# ============================================================
def main():
    print("=== SAC BACKTEST (3 PARTS) ===")
    print(f"[INFO] MODEL_PATH={MODEL_PATH}")
    print(f"[INFO] VNORM_PATH={VNORM_PATH}")
    print(f"[INFO] ENV_CFG={ENV_CFG}")

    model = SAC.load(MODEL_PATH, device="cpu")

    for part, cfg in BACKTEST_PARTS.items():
        prices = load_prices(cfg["ohlcv_table"])
        emb_path = PROJECT_ROOT / "results" / "backtest" / "fusion" / cfg["fusion_emb"]
        embeddings = np.load(emb_path)

        run_single(part, prices, embeddings, model)

    print("\n=== DONE ===")


if __name__ == "__main__":
    main()
