from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, List, Any, Optional

import numpy as np
import pandas as pd
from sqlalchemy import create_engine

from src.rl_agent.rl_env.bandit_trading_env import TradingEnvBanditFusion, BanditCostConfig


# ============================================================
# PATHS / SPLITS
# ============================================================
PROJECT_ROOT = Path("E:/TDTu/TAI_LIEU/KY1-NAM5/DU_AN_CNTT")
SCHEMA = "it_final"

CKPT_PATH = PROJECT_ROOT / "results" / "rl_agent" / "bandit_fusion" / "linucb_fusion_train.npz"
EMB_DIR = PROJECT_ROOT / "results" / "backtest" / "fusion"
OUT_DIR = PROJECT_ROOT / "results" / "backtest" / "bandit_fusion"

SPLITS = {
    "bt1": ("ohlcv_backtest_part1", "fusion_embeddings_backtest_v2_part1.npy"),
    "bt2": ("ohlcv_backtest_part2", "fusion_embeddings_backtest_v2_part2.npy"),
    "bt3": ("ohlcv_backtest_part3", "fusion_embeddings_backtest_v2_part3.npy"),
}

# ============================================================
# ACCOUNT (report/plot only)
# ============================================================
INITIAL_BALANCE = 1000.0  # USD

# ============================================================
# COSTS
# ============================================================
SWITCH_COST = 0.0002
HOLD_COST = 0.0

# ============================================================
# HOLD / TREND DEFAULTS
# ============================================================
MAX_HOLD = 24

# ============================================================
# CONSTRAINTS: total dương và >=2 split dương
# ============================================================
REQUIRE_AT_LEAST_POS_SPLITS = 2
POS_EPS = 1e-4             # > 1 + POS_EPS mới tính là dương
REQUIRE_TOTAL_POSITIVE = True


# ============================================================
# BANDIT CORE (LinDiagUCB + online update)
# ============================================================
@dataclass
class LinUCBConfig:
    alpha: float
    ridge: float
    tau: float
    cooldown: int
    gamma: float
    min_adv: float
    max_hold: int

    # anti-noise gate
    eps_abs: float = 0.0
    eps_pct: float = 0.0

    # require advantage over FLAT to enter non-flat position
    flat_adv: float = 0.0

    # reset prior each split (shrink confidence keep theta)
    reset_prior_each_split: bool = False

    # trend options
    use_trend: bool = True
    sma_fast: int = 20
    sma_slow: int = 50
    forbid_to_flat: int = 0     # 1: forbid -> FLAT, 0: forbid -> switch opposite
    zero_trend_to_flat: bool = True

    # anti flip: LONG<->SHORT must go through FLAT 1 step
    flip_to_flat: bool = False


class LinDiagUCBOnline:
    """
    Diagonal LinUCB, online update.

    IMPORTANT:
      - ckpt dimension d is fixed.
      - obs from env may differ (vd 256 vs ckpt 260): we pad/truncate obs inside agent.
    """

    def __init__(self, A_diag: np.ndarray, b: np.ndarray, cfg: LinUCBConfig):
        self.A_diag = np.asarray(A_diag, dtype=np.float64)
        self.b = np.asarray(b, dtype=np.float64)
        self.cfg = cfg

        if self.A_diag.ndim != 2:
            raise ValueError("A_diag must have shape (n_actions, d)")
        if self.b.shape != self.A_diag.shape:
            raise ValueError(f"b shape {self.b.shape} != A_diag shape {self.A_diag.shape}")

        self.n_actions, self.d = self.A_diag.shape

        # runtime states
        self.cooldown_left = 0
        self.cur_action = 0  # 0=FLAT,1=LONG,2=SHORT
        self.cur_pos = 0     # -1/0/+1
        self.hold_steps = 0

    def reset_runtime(self):
        self.cooldown_left = 0
        self.cur_action = 0
        self.cur_pos = 0
        self.hold_steps = 0

    def shrink_confidence_keep_theta(self):
        ridge = float(self.cfg.ridge)
        theta0 = self.b / np.maximum(self.A_diag, 1e-12)
        self.A_diag[:] = ridge
        self.b[:] = theta0 * ridge

    def _prep_x(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        if x.shape[0] == self.d:
            return x
        if x.shape[0] > self.d:
            return x[: self.d]
        out = np.zeros((self.d,), dtype=np.float64)
        out[: x.shape[0]] = x
        return out

    def scores(self, x: np.ndarray) -> np.ndarray:
        x = self._prep_x(x)
        A = np.maximum(self.A_diag, 1e-12)
        theta = self.b / A
        mean = theta @ x
        bonus = float(self.cfg.alpha) * np.sqrt(np.sum((x * x) / A, axis=1))
        return mean + bonus

    def select_action(self, x: np.ndarray) -> int:
        scores = self.scores(x)
        best = int(np.argmax(scores))
        cur = int(self.cur_action)

        # optional: require best beats FLAT by flat_adv if best != 0
        if best != 0 and float(self.cfg.flat_adv) > 0.0:
            if float(scores[best] - scores[0]) < float(self.cfg.flat_adv):
                best = 0

        # cooldown
        if self.cooldown_left > 0:
            self.cooldown_left -= 1
            return cur

        if best == cur:
            return best

        adv = float(scores[best] - scores[cur])

        edge = max(float(self.cfg.tau), float(self.cfg.min_adv), float(self.cfg.eps_abs))
        if float(self.cfg.eps_pct) > 0.0:
            edge = max(edge, float(self.cfg.eps_pct) * max(1.0, abs(float(scores[cur]))))

        if adv > edge:
            self.cooldown_left = int(self.cfg.cooldown)
            return best

        return cur

    def observe(self, action: int, x: np.ndarray, reward: float):
        a = int(action)
        x = self._prep_x(x)
        r = float(reward)

        g = float(self.cfg.gamma)
        self.A_diag[a] = g * self.A_diag[a] + (x * x)
        self.b[a] = g * self.b[a] + (r * x)


# ============================================================
# HELPERS
# ============================================================
def get_engine():
    conn = os.getenv("PG_CONN_STR")
    if not conn:
        raise RuntimeError("Missing PG_CONN_STR")
    return create_engine(conn)


def load_prices(schema: str, table: str, ts_col="time_stamp", open_col="open", close_col="close") -> pd.DataFrame:
    eng = get_engine()
    q = f"""
    SELECT {ts_col} AS time_stamp, {open_col} AS open_raw, {close_col} AS close_raw
    FROM {schema}.{table}
    ORDER BY 1
    """
    df = pd.read_sql(q, eng)
    df["time_stamp"] = pd.to_datetime(df["time_stamp"])
    return df


def align_price_with_emb(df_price: pd.DataFrame, emb: np.ndarray) -> Tuple[pd.DataFrame, np.ndarray]:
    T = emb.shape[0]
    if len(df_price) < T:
        raise ValueError(f"price rows {len(df_price)} < emb rows {T}")
    if len(df_price) > T:
        offset = len(df_price) - T
        df_price = df_price.iloc[offset:].reset_index(drop=True)
        print(f"[ALIGN] dropped first {offset} rows to match embeddings.")
    return df_price, emb


def compute_trend_sma(close: np.ndarray, fast: int, slow: int) -> np.ndarray:
    s = pd.Series(np.asarray(close, dtype=np.float64))
    sma_fast = s.rolling(fast, min_periods=fast).mean()
    sma_slow = s.rolling(slow, min_periods=slow).mean()
    diff = (sma_fast - sma_slow).fillna(0.0).to_numpy()
    return np.sign(diff).astype(np.int8)  # -1,0,+1


def load_ckpt_diag_or_full(npz_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    z = np.load(str(npz_path), allow_pickle=False)

    if "A_diag" in z.files and "b" in z.files:
        return z["A_diag"], z["b"]

    if "A" in z.files and "b" in z.files:
        A = z["A"]
        b = z["b"]
        if A.ndim != 3:
            raise ValueError("A must be (n_actions, d, d)")
        A_diag = np.diagonal(A, axis1=1, axis2=2).copy()
        return A_diag, b

    raise ValueError(f"Checkpoint missing keys. Found keys={z.files}")


def compute_metrics(
    equity: np.ndarray,
    traded: np.ndarray,
    initial_balance: float,
    steps_per_year: int = 365 * 24
) -> Dict[str, float]:
    eq = np.asarray(equity, dtype=np.float64)
    if len(eq) < 2:
        final_eq = float(eq[-1]) if len(eq) else 1.0
        final_bal = float(final_eq * initial_balance)
        roi = (final_bal / initial_balance) - 1.0
        return {
            "final_equity": final_eq,
            "final_balance": final_bal,
            "roi": float(roi),
            "sharpe": 0.0,
            "max_drawdown": 0.0,
            "turnover_rate": 0.0,
        }

    rets = np.diff(eq) / np.maximum(eq[:-1], 1e-12)
    mu = float(np.mean(rets))
    sigma = float(np.std(rets) + 1e-12)
    sharpe = (mu / sigma) * np.sqrt(steps_per_year)

    peak = np.maximum.accumulate(eq)
    dd = (peak - eq) / np.maximum(peak, 1e-12)
    mdd = float(np.max(dd))

    turnover = float(np.mean(traded)) if len(traded) else 0.0

    final_eq = float(eq[-1])
    final_bal = float(final_eq * initial_balance)
    roi = (final_bal / initial_balance) - 1.0

    return {
        "final_equity": final_eq,
        "final_balance": final_bal,
        "roi": float(roi),
        "sharpe": float(sharpe),
        "max_drawdown": float(mdd),
        "turnover_rate": float(turnover),
    }


# ============================================================
# RUN ONE SPLIT
# ============================================================
def run_one_split(
    split_name: str,
    data: Dict[str, Any],
    trend_cache: Dict[Tuple[int, int], np.ndarray],
    A_diag0: np.ndarray,
    b0: np.ndarray,
    cfg: LinUCBConfig,
    save_csv: bool,
) -> Dict[str, Any]:
    open_raw = data["open_raw"]
    close_raw = data["close_raw"]
    emb = data["emb"]

    trend: Optional[np.ndarray] = None
    if cfg.use_trend:
        key = (int(cfg.sma_fast), int(cfg.sma_slow))
        if key not in trend_cache:
            trend_cache[key] = compute_trend_sma(close_raw, fast=int(cfg.sma_fast), slow=int(cfg.sma_slow))
        trend = trend_cache[key]

    costs = BanditCostConfig(switch_cost=SWITCH_COST, hold_cost=HOLD_COST)
    env = TradingEnvBanditFusion(
        open_raw=open_raw,
        close_raw=close_raw,
        embeddings=emb,
        costs=costs,
        initial_equity=1.0,
    )

    agent = LinDiagUCBOnline(A_diag=A_diag0.copy(), b=b0.copy(), cfg=cfg)
    agent.reset_runtime()

    if cfg.reset_prior_each_split:
        agent.shrink_confidence_keep_theta()

    obs, _ = env.reset()

    equity: List[float] = []
    traded: List[int] = []
    positions: List[int] = []
    actions: List[int] = []
    rewards: List[float] = []
    ts: List[int] = []

    done = False
    step_idx = 0

    while not done:
        # choose action
        a = agent.select_action(obs)

        # max-hold
        if agent.cur_pos != 0:
            agent.hold_steps += 1
        else:
            agent.hold_steps = 0

        if agent.cur_pos != 0 and agent.hold_steps >= int(cfg.max_hold):
            a = 0

        # anti flip: long<->short must pass flat
        if cfg.flip_to_flat:
            if (agent.cur_action == 1 and a == 2) or (agent.cur_action == 2 and a == 1):
                a = 0

        # trend gating
        if cfg.use_trend and trend is not None:
            tr = int(trend[step_idx]) if step_idx < len(trend) else 0

            if tr == 0 and cfg.zero_trend_to_flat:
                a = 0
            elif tr > 0 and a == 2:  # bull forbid short
                a = 0 if int(cfg.forbid_to_flat) == 1 else 1
            elif tr < 0 and a == 1:  # bear forbid long
                a = 0 if int(cfg.forbid_to_flat) == 1 else 2

        obs2, r, done, info = env.step(int(a))

        agent.observe(int(a), obs, float(r))
        agent.cur_action = int(a)
        agent.cur_pos = int(info.get("pos", 0))

        equity.append(float(info.get("equity", np.nan)))
        traded.append(int(info.get("traded", 0)))
        positions.append(int(info.get("pos", 0)))
        actions.append(int(info.get("action", a)))
        rewards.append(float(r))
        ts.append(int(info.get("t", step_idx)))

        obs = obs2
        step_idx += 1

    eq = np.asarray(equity, dtype=np.float64)
    trd = np.asarray(traded, dtype=np.int32)

    met = compute_metrics(eq, trd, initial_balance=INITIAL_BALANCE)

    out: Dict[str, Any] = {"metrics": met}

    if save_csv:
        balance = eq * float(INITIAL_BALANCE)
        roi_cum = (balance / float(INITIAL_BALANCE)) - 1.0

        pnl_step = np.zeros_like(balance, dtype=np.float64)
        ret_step = np.zeros_like(balance, dtype=np.float64)
        if len(balance) >= 2:
            pnl_step[1:] = balance[1:] - balance[:-1]
            ret_step[1:] = pnl_step[1:] / np.maximum(balance[:-1], 1e-12)

        df_out = pd.DataFrame(
            {
                "t": ts,
                "equity": eq,
                "balance": balance,
                "roi": roi_cum,
                "pnl_step": pnl_step,
                "ret_step": ret_step,
                "traded": trd,
                "position": positions,
                "action": actions,
                "reward": rewards,
            }
        )
        df_out.to_csv(OUT_DIR / f"equity_{split_name}.csv", index=False, encoding="utf-8")

        trade_count = int(np.sum(trd))
        a0 = int(np.sum(np.asarray(actions, dtype=np.int32) == 0))
        a1 = int(np.sum(np.asarray(actions, dtype=np.int32) == 1))
        a2 = int(np.sum(np.asarray(actions, dtype=np.int32) == 2))
        out["debug"] = {
            "trade_count": trade_count,
            "n_steps": int(len(trd)),
            "actions_tuple": (a0, a1, a2),
        }

    return out


def run_all_splits(
    splits_data: Dict[str, Dict[str, Any]],
    A_diag0: np.ndarray,
    b0: np.ndarray,
    cfg: LinUCBConfig,
    save_csv: bool = False,
) -> Tuple[float, Dict[str, float], Dict[str, Dict[str, Any]]]:
    total_eq = 1.0
    split_final_eq: Dict[str, float] = {}
    split_outputs: Dict[str, Dict[str, Any]] = {}

    trend_cache_by_split: Dict[str, Dict[Tuple[int, int], np.ndarray]] = {k: {} for k in splits_data.keys()}

    for name in ("bt1", "bt2", "bt3"):
        out = run_one_split(
            split_name=name,
            data=splits_data[name],
            trend_cache=trend_cache_by_split[name],
            A_diag0=A_diag0,
            b0=b0,
            cfg=cfg,
            save_csv=save_csv,
        )
        feq = float(out["metrics"]["final_equity"])
        split_final_eq[name] = feq
        total_eq *= feq
        split_outputs[name] = out

    return float(total_eq), split_final_eq, split_outputs


# ============================================================
# MAIN
# ============================================================
def main():
    os.environ.setdefault("PG_CONN_STR", "postgresql+psycopg2://postgres:123456789@localhost:5432/postgres")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    A_diag0_full, b0_full = load_ckpt_diag_or_full(CKPT_PATH)
    ckpt_d = int(A_diag0_full.shape[1])

    print(f"[ACCOUNT] initial_balance=${INITIAL_BALANCE:,.2f}")
    print(f"[COST] switch_cost={SWITCH_COST} hold_cost={HOLD_COST}")
    print(f"[CKPT] d={ckpt_d} | obs will be padded/truncated inside agent")

    # ------------------------------------------------------------
    # Preload data once
    # ------------------------------------------------------------
    splits_data: Dict[str, Dict[str, Any]] = {}
    for name, (price_table, emb_file) in SPLITS.items():
        df_price = load_prices(SCHEMA, price_table)
        emb = np.load(str(EMB_DIR / emb_file)).astype(np.float32)
        df_price, emb = align_price_with_emb(df_price, emb)

        splits_data[name] = {
            "open_raw": df_price["open_raw"].to_numpy(),
            "close_raw": df_price["close_raw"].to_numpy(),
            "emb": emb,
        }

    # ------------------------------------------------------------
    # SWEEP (random) — mở rộng không gian tìm kiếm
    # ------------------------------------------------------------
    rng = np.random.default_rng(1337)

    MAX_SWEEP = 220

    alpha_space = [0.06, 0.08, 0.10, 0.12, 0.15, 0.18, 0.22, 0.28]
    gamma_space = [1.0, 0.99995, 0.9998, 0.9995, 0.999, 0.998]
    cooldown_space = [6, 8, 10, 12, 14, 16, 18, 24]
    min_adv_space = [0.0015, 0.002, 0.003, 0.004, 0.005, 0.006]
    eps_abs_space = [0.0, 1e-4, 3e-4, 5e-4, 1e-3]
    eps_pct_space = [0.0, 0.0002, 0.0005, 0.001, 0.0015, 0.002]
    flat_adv_space = [0.0, 0.0005, 0.001, 0.0015, 0.002, 0.003]

    use_trend_space = [0, 1]
    forbid_to_flat_space = [0, 1]
    reset_prior_space = [0, 1]
    zero_to_flat_space = [0, 1]
    flip_to_flat_space = [0, 1]

    sma_fast_space = [10, 20, 30]
    sma_slow_space = [40, 50, 80, 120]

    seed_configs: List[Dict[str, Any]] = [
        dict(
            alpha=0.12, gamma=1.0, cooldown=14, min_adv=0.004,
            eps_abs=0.0, eps_pct=0.0, flat_adv=0.0,
            reset_prior=0, use_trend=1, sma_fast=20, sma_slow=50,
            forbid_to_flat=0, zero_to_flat=1, flip_to_flat=0
        )
    ]

    configs: List[Dict[str, Any]] = []
    configs.extend(seed_configs)

    while len(configs) < MAX_SWEEP:
        sf = int(rng.choice(sma_fast_space))
        ss = int(rng.choice([x for x in sma_slow_space if x > sf]))

        c = dict(
            alpha=float(rng.choice(alpha_space)),
            gamma=float(rng.choice(gamma_space)),
            cooldown=int(rng.choice(cooldown_space)),
            min_adv=float(rng.choice(min_adv_space)),
            eps_abs=float(rng.choice(eps_abs_space)),
            eps_pct=float(rng.choice(eps_pct_space)),
            flat_adv=float(rng.choice(flat_adv_space)),
            reset_prior=int(rng.choice(reset_prior_space)),
            use_trend=int(rng.choice(use_trend_space)),
            sma_fast=sf,
            sma_slow=ss,
            forbid_to_flat=int(rng.choice(forbid_to_flat_space)),
            zero_to_flat=int(rng.choice(zero_to_flat_space)),
            flip_to_flat=int(rng.choice(flip_to_flat_space)),
        )
        configs.append(c)

    best: Optional[Dict[str, Any]] = None
    best_total_eq = -1e18

    for i, p in enumerate(configs, start=1):
        cfg = LinUCBConfig(
            alpha=float(p["alpha"]),
            ridge=1.0,
            tau=0.0,
            cooldown=int(p["cooldown"]),
            gamma=float(p["gamma"]),
            min_adv=float(p["min_adv"]),
            max_hold=int(MAX_HOLD),
            eps_abs=float(p["eps_abs"]),
            eps_pct=float(p["eps_pct"]),
            flat_adv=float(p["flat_adv"]),
            reset_prior_each_split=bool(int(p["reset_prior"])),
            use_trend=bool(int(p["use_trend"])),
            sma_fast=int(p["sma_fast"]),
            sma_slow=int(p["sma_slow"]),
            forbid_to_flat=int(p["forbid_to_flat"]),
            zero_trend_to_flat=bool(int(p["zero_to_flat"])),
            flip_to_flat=bool(int(p["flip_to_flat"])),
        )

        total_eq, split_final_eq, _ = run_all_splits(
            splits_data=splits_data,
            A_diag0=A_diag0_full,
            b0=b0_full,
            cfg=cfg,
            save_csv=False,
        )

        pos_splits = [k for k, v in split_final_eq.items() if float(v) > (1.0 + POS_EPS)]
        pos_count = len(pos_splits)

        feasible = True
        if REQUIRE_TOTAL_POSITIVE:
            feasible = feasible and (float(total_eq) > (1.0 + POS_EPS))
        feasible = feasible and (pos_count >= REQUIRE_AT_LEAST_POS_SPLITS)

        tag = f"r{i}"
        print(
            f"[SWEEP] {tag} | alpha={cfg.alpha} gamma={cfg.gamma} cooldown={cfg.cooldown} min_adv={cfg.min_adv} "
            f"eps_abs={cfg.eps_abs} eps_pct={cfg.eps_pct} flat_adv={cfg.flat_adv} "
            f"use_trend={int(cfg.use_trend)} sma=({cfg.sma_fast},{cfg.sma_slow}) "
            f"reset_prior={int(cfg.reset_prior_each_split)} forbid_to_flat={int(cfg.forbid_to_flat)} "
            f"zero_to_flat={int(cfg.zero_trend_to_flat)} flip_to_flat={int(cfg.flip_to_flat)} "
            f"-> TOTAL eq={total_eq:.6f} ROI={(total_eq-1)*100:.2f}% | pos={pos_count}/3 {pos_splits}"
        )

        if feasible:
            if float(total_eq) > best_total_eq:
                best_total_eq = float(total_eq)
                best = {
                    "id": tag,
                    "params": dict(p),
                    "total_eq": float(total_eq),
                    "split_final_eq": dict(split_final_eq),
                    "pos_splits": list(pos_splits),
                    "pos_count": int(pos_count),
                }

    if best is None:
        print("\n=== NO FEASIBLE CONFIG FOUND ===")
        print("Condition: total>1 and at least 2/3 splits positive.")
        print(f"Tip: increase MAX_SWEEP (currently {MAX_SWEEP}), or widen spaces (alpha/min_adv/flat_adv/sma).")
        return

    bp = best["params"]
    print("\n=== BEST CONFIG (CONSTRAINED) ===")
    print(
        f"[BEST] {best['id']} | TOTAL eq={best['total_eq']:.6f} ROI={(best['total_eq']-1)*100:.2f}% "
        f"| pos={best['pos_count']}/3 {best['pos_splits']} | split={best['split_final_eq']}"
    )
    print(f"[BEST] params={bp}")

    best_cfg = LinUCBConfig(
        alpha=float(bp["alpha"]),
        ridge=1.0,
        tau=0.0,
        cooldown=int(bp["cooldown"]),
        gamma=float(bp["gamma"]),
        min_adv=float(bp["min_adv"]),
        max_hold=int(MAX_HOLD),
        eps_abs=float(bp["eps_abs"]),
        eps_pct=float(bp["eps_pct"]),
        flat_adv=float(bp["flat_adv"]),
        reset_prior_each_split=bool(int(bp["reset_prior"])),
        use_trend=bool(int(bp["use_trend"])),
        sma_fast=int(bp["sma_fast"]),
        sma_slow=int(bp["sma_slow"]),
        forbid_to_flat=int(bp["forbid_to_flat"]),
        zero_trend_to_flat=bool(int(bp["zero_to_flat"])),
        flip_to_flat=bool(int(bp["flip_to_flat"])),
    )

    total_eq, _, split_outputs = run_all_splits(
        splits_data=splits_data,
        A_diag0=A_diag0_full,
        b0=b0_full,
        cfg=best_cfg,
        save_csv=True,
    )

    for name in ("bt1", "bt2", "bt3"):
        met = split_outputs[name]["metrics"]
        print(f"\n=== BANDIT FUSION {name} (BACKTEST BEST) ===")
        print(met)
        print(f"[RESULT] final_balance=${met['final_balance']:,.2f} | ROI={met['roi']*100:.2f}%")

        dbg = split_outputs[name].get("debug", None)
        if dbg:
            a0, a1, a2 = dbg["actions_tuple"]
            print(f"[DEBUG] trade_count={dbg['trade_count']} / {dbg['n_steps']} steps | actions(FLAT/LONG/SHORT)=({a0}/{a1}/{a2})")

    final_balance = float(total_eq) * float(INITIAL_BALANCE)
    roi_total = (final_balance / float(INITIAL_BALANCE)) - 1.0
    print("\n=== TOTAL (bt1 * bt2 * bt3) BEST (CONSTRAINED) ===")
    print(f"[TOTAL] final_equity={total_eq:.6f} | final_balance=${final_balance:,.2f} | ROI={roi_total*100:.2f}%")

    print(f"\nSaved to {OUT_DIR}")


if __name__ == "__main__":
    main()
