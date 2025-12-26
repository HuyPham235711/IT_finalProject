from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, List

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

# NOTE: đúng theo folder fusion backtest của bạn
EMB_DIR = PROJECT_ROOT / "results" / "backtest" / "fusion"
OUT_DIR = PROJECT_ROOT / "results" / "backtest" / "bandit_fusion"

SPLITS = {
    "bt1": ("ohlcv_backtest_part1", "fusion_embeddings_backtest_v2_part1.npy"),
    "bt2": ("ohlcv_backtest_part2", "fusion_embeddings_backtest_v2_part2.npy"),
    "bt3": ("ohlcv_backtest_part3", "fusion_embeddings_backtest_v2_part3.npy"),
}

# ============================================================
# ACCOUNT 
# ============================================================
INITIAL_BALANCE = 1000.0  # USD

# ============================================================
# COSTS
# ============================================================
SWITCH_COST = 0.0002
HOLD_COST = 0.0

# ============================================================
# BEST CONFIG
# ============================================================
BEST = {
    "alpha": 0.22,
    "gamma": 0.999,
    "cooldown": 10,
    "min_adv": 0.002,
    "eps_abs": 0.0003,   # absolute advantage gate (thêm 1 lớp filter)
    "eps_pct": 0.0,      # pct advantage gate
    "flat_adv": 0.0,     # extra gate khi chuyển về FLAT (0 => off)
    "reset_prior": 1,    # shrink confidence each split, keep theta0
    "use_trend": 0,      # tắt trend gating theo best log
    "sma_fast": 30,      # giữ để log/khả năng bật lại, nhưng best đang use_trend=0
    "sma_slow": 120,
    "forbid_to_flat": 1, # nếu trend gating bật: cấm thì về FLAT
    "zero_to_flat": 1,   # nếu trend gating bật: trend=0 => FLAT
    "flip_to_flat": 1,   # HARD RULE: LONG <-> SHORT thì ép qua FLAT (giảm flip)
}

# max hold giữ như bạn đang dùng trước đó (để chống hold quá lâu)
MAX_HOLD = 24  # giờ/steps (1 step = 1h)


# ============================================================
# BANDIT CORE (LinDiagUCB + online update)
# ============================================================
@dataclass
class LinUCBConfig:
    alpha: float
    ridge: float
    cooldown: int
    gamma: float
    min_adv: float
    eps_abs: float
    eps_pct: float
    flat_adv: float
    max_hold: int


class LinDiagUCBOnline:
    """
    Diagonal LinUCB, online update.
    - A_diag[a, j] >= ridge
    - b[a, j]
    theta[a] = b[a] / A_diag[a]
    score = theta·x + alpha*sqrt(sum(x^2 / A_diag))

    IMPORTANT: obs_dim có thể != ckpt_d.
    -> pad/truncate x ngay trong agent để luôn khớp d=ckpt_d.
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
        self.cur_pos = 0     # -1/0/+1 from env
        self.hold_steps = 0  # consecutive steps where cur_pos != 0

    def reset_runtime(self):
        self.cooldown_left = 0
        self.cur_action = 0
        self.cur_pos = 0
        self.hold_steps = 0

    def shrink_confidence_keep_theta(self):
        """
        RESET_PRIOR behavior:
          Keep theta0 but shrink confidence by resetting A_diag to ridge
          theta0 = b/A_diag => keep theta0 by setting b = theta0 * ridge
        """
        ridge = float(self.cfg.ridge)
        theta0 = self.b / np.maximum(self.A_diag, 1e-12)
        self.A_diag[:] = ridge
        self.b[:] = theta0 * ridge

    def _fix_dim(self, x: np.ndarray) -> np.ndarray:
        """
        Pad with zeros or truncate to match ckpt_d.
        """
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        if x.shape[0] == self.d:
            return x
        if x.shape[0] > self.d:
            return x[: self.d]
        out = np.zeros((self.d,), dtype=np.float64)
        out[: x.shape[0]] = x
        return out

    def _scores(self, x: np.ndarray) -> np.ndarray:
        x = self._fix_dim(x)
        A = np.maximum(self.A_diag, 1e-12)
        theta = self.b / A
        mean = theta @ x  # (n_actions,)
        bonus = float(self.cfg.alpha) * np.sqrt(np.sum((x * x) / A, axis=1))
        return mean + bonus

    @staticmethod
    def _sign_from_action(a: int) -> int:
        if a == 1:
            return 1
        if a == 2:
            return -1
        return 0

    def select_action(self, obs: np.ndarray) -> int:
        scores = self._scores(obs)
        best = int(np.argmax(scores))

        # cooldown: giữ nguyên action hiện tại
        if self.cooldown_left > 0:
            self.cooldown_left -= 1
            return int(self.cur_action)

        cur = int(self.cur_action)
        if best == cur:
            return best

        adv = float(scores[best] - scores[cur])
        pct_gate = float(self.cfg.eps_pct) * max(abs(float(scores[cur])), 1e-12)
        gate = max(float(self.cfg.min_adv), float(self.cfg.eps_abs), pct_gate)

        # special gate cho chuyển về FLAT (nếu flat_adv>0)
        if best == 0 and float(self.cfg.flat_adv) > 0.0:
            gate = max(gate, float(self.cfg.flat_adv))

        if adv > gate:
            self.cooldown_left = int(self.cfg.cooldown)
            return best

        return cur

    def observe(self, action: int, obs: np.ndarray, reward: float):
        """
        Online update with forgetting:
          A_diag[a] = gamma*A_diag[a] + x^2
          b[a]      = gamma*b[a]      + reward*x
        """
        a = int(action)
        x = self._fix_dim(obs)
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
    T = int(emb.shape[0])
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
    return np.sign(diff).astype(np.int8)  # -1/0/+1


def compute_metrics(
    equity: np.ndarray,
    traded: np.ndarray,
    initial_balance: float,
    steps_per_year: int = 365 * 24,
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
            "turnover_rate": float(np.mean(traded)) if len(traded) else 0.0,
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
# MAIN
# ============================================================
def main():
    os.environ.setdefault(
        "PG_CONN_STR",
        "postgresql+psycopg2://postgres:123456789@localhost:5432/postgres",
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[ACCOUNT] initial_balance=${INITIAL_BALANCE:,.2f}")
    print(f"[COST] switch_cost={SWITCH_COST} hold_cost={HOLD_COST}")

    # load trained params
    z = np.load(str(CKPT_PATH), allow_pickle=False)
    if "A_diag" in z.files and "b" in z.files:
        A_diag0_full = z["A_diag"]
        b0_full = z["b"]
    elif "A" in z.files and "b" in z.files:
        A = z["A"]  # (n_actions, d, d)
        b0_full = z["b"]
        if A.ndim != 3:
            raise ValueError("A must be (n_actions, d, d)")
        A_diag0_full = np.diagonal(A, axis1=1, axis2=2).copy()
    else:
        raise ValueError(f"Checkpoint missing keys. Found keys={z.files}")

    ckpt_d = int(A_diag0_full.shape[1])
    print(f"[CKPT] d={ckpt_d} | obs will be padded/truncated inside agent")

    # config
    cfg = LinUCBConfig(
        alpha=float(BEST["alpha"]),
        ridge=1.0,  # giữ ridge=1.0 (consistent với các bản trước)
        cooldown=int(BEST["cooldown"]),
        gamma=float(BEST["gamma"]),
        min_adv=float(BEST["min_adv"]),
        eps_abs=float(BEST["eps_abs"]),
        eps_pct=float(BEST["eps_pct"]),
        flat_adv=float(BEST["flat_adv"]),
        max_hold=int(MAX_HOLD),
    )

    print(
        "[BEST] "
        f"alpha={cfg.alpha} gamma={cfg.gamma} cooldown={cfg.cooldown} "
        f"min_adv={cfg.min_adv} eps_abs={cfg.eps_abs} eps_pct={cfg.eps_pct} flat_adv={cfg.flat_adv} "
        f"reset_prior={int(BEST['reset_prior'])} use_trend={int(BEST['use_trend'])} "
        f"sma=({BEST['sma_fast']},{BEST['sma_slow']}) forbid_to_flat={int(BEST['forbid_to_flat'])} "
        f"zero_to_flat={int(BEST['zero_to_flat'])} flip_to_flat={int(BEST['flip_to_flat'])}"
    )

    split_final_eq: Dict[str, float] = {}

    for name, (price_table, emb_file) in SPLITS.items():
        print(f"\n[RUN] {name} | price_table={price_table} | emb={emb_file}")

        df_price = load_prices(SCHEMA, price_table)
        emb = np.load(str(EMB_DIR / emb_file)).astype(np.float32)
        df_price, emb = align_price_with_emb(df_price, emb)

        # arrays for plotting (aligned)
        ts_arr = df_price["time_stamp"].to_numpy()
        open_arr = df_price["open_raw"].to_numpy(dtype=np.float64)
        close_arr = df_price["close_raw"].to_numpy(dtype=np.float64)

        costs = BanditCostConfig(switch_cost=SWITCH_COST, hold_cost=HOLD_COST)
        env = TradingEnvBanditFusion(
            open_raw=open_arr,
            close_raw=close_arr,
            embeddings=emb,
            costs=costs,
            initial_equity=1.0,  # giữ nguyên logic equity
        )

        # optional trend array (best đang use_trend=0 => không dùng)
        trend = None
        if int(BEST["use_trend"]) == 1:
            trend = compute_trend_sma(
                close_arr,
                fast=int(BEST["sma_fast"]),
                slow=int(BEST["sma_slow"]),
            )

        # fresh agent per split
        agent = LinDiagUCBOnline(A_diag=A_diag0_full.copy(), b=b0_full.copy(), cfg=cfg)
        agent.reset_runtime()

        if int(BEST["reset_prior"]) == 1:
            agent.shrink_confidence_keep_theta()

        obs, _ = env.reset()

        equity: List[float] = []
        traded: List[int] = []
        positions: List[int] = []
        actions: List[int] = []
        t_idx: List[int] = []
        rewards: List[float] = []

        # NEW: market/time + entry info for plotting
        time_stamp: List[str] = []
        market_open: List[float] = []
        market_close: List[float] = []
        entry_price: List[float] = []
        entry_side: List[int] = []  # 1=LONG entry, -1=SHORT entry, 0=no entry

        done = False
        step_idx = 0

        while not done:
            # guard index
            px_i = step_idx if step_idx < len(close_arr) else (len(close_arr) - 1)

            # 1) choose action
            a = int(agent.select_action(obs))

            # 2) max-hold rule (based on current pos from previous step)
            if agent.cur_pos != 0:
                agent.hold_steps += 1
            else:
                agent.hold_steps = 0

            if agent.cur_pos != 0 and agent.hold_steps >= agent.cfg.max_hold:
                a = 0  # force FLAT

            # 3) flip-to-flat (HARDCORE): LONG <-> SHORT thì bắt buộc qua FLAT
            if int(BEST["flip_to_flat"]) == 1:
                cur_sign = LinDiagUCBOnline._sign_from_action(agent.cur_action)
                nxt_sign = LinDiagUCBOnline._sign_from_action(a)
                if cur_sign != 0 and nxt_sign != 0 and (cur_sign != nxt_sign):
                    a = 0

            # 4) optional trend gating (best đang OFF)
            if int(BEST["use_trend"]) == 1 and trend is not None:
                tr = int(trend[px_i]) if px_i < len(trend) else 0

                if tr == 0 and int(BEST["zero_to_flat"]) == 1:
                    a = 0
                elif tr > 0 and a == 2:  # bull forbid SHORT
                    a = 0 if int(BEST["forbid_to_flat"]) == 1 else 1
                elif tr < 0 and a == 1:  # bear forbid LONG
                    a = 0 if int(BEST["forbid_to_flat"]) == 1 else 2

            # 5) env step
            prev_pos = int(agent.cur_pos)  # NEW: for entry detection
            obs2, r, done, info = env.step(int(a))
            new_pos = int(info.get("pos", 0))

            # 6) online update
            agent.observe(int(a), obs, float(r))

            # 7) update runtime state using env info
            agent.cur_action = int(a)
            agent.cur_pos = new_pos

            # core logs
            equity.append(float(info.get("equity", np.nan)))
            traded.append(int(info.get("traded", 0)))
            positions.append(new_pos)
            actions.append(int(info.get("action", a)))
            t_idx.append(int(info.get("t", step_idx)))
            rewards.append(float(r))

            # NEW: per-step market/time
            time_stamp.append(str(ts_arr[px_i]))
            market_open.append(float(open_arr[px_i]))
            market_close.append(float(close_arr[px_i]))

            # NEW: entry detection (0 -> +/-1)
            if prev_pos == 0 and new_pos == 1:
                entry_side.append(1)
                entry_price.append(float(open_arr[px_i]))   # entry at OPEN of this step
            elif prev_pos == 0 and new_pos == -1:
                entry_side.append(-1)
                entry_price.append(float(open_arr[px_i]))   # entry at OPEN of this step
            else:
                entry_side.append(0)
                entry_price.append(np.nan)

            obs = obs2
            step_idx += 1

        eq = np.asarray(equity, dtype=np.float64)
        trd = np.asarray(traded, dtype=np.int32)

        met = compute_metrics(eq, trd, initial_balance=INITIAL_BALANCE)

        print(f"\n=== BANDIT FUSION {name} (BACKTEST BEST) ===")
        print(met)
        print(f"[RESULT] final_balance=${met['final_balance']:,.2f} | ROI={met['roi']*100:.2f}%")

        # quick debug
        u_a, c_a = np.unique(np.asarray(actions, dtype=np.int32), return_counts=True)
        counts = {int(k): int(v) for k, v in zip(u_a, c_a)}
        flat_n = counts.get(0, 0)
        long_n = counts.get(1, 0)
        short_n = counts.get(2, 0)
        trade_count = int(np.sum(trd))
        print(f"[DEBUG] trade_count={trade_count} / {len(trd)} steps | actions(FLAT/LONG/SHORT)=({flat_n}/{long_n}/{short_n})")

        # store final equity for TOTAL product
        split_final_eq[name] = float(met["final_equity"])

        # === add balance/roi for plotting ===
        balance = eq * float(INITIAL_BALANCE)
        roi_cum = (balance / float(INITIAL_BALANCE)) - 1.0

        pnl_step = np.zeros_like(balance, dtype=np.float64)
        ret_step = np.zeros_like(balance, dtype=np.float64)
        if len(balance) >= 2:
            pnl_step[1:] = balance[1:] - balance[:-1]
            ret_step[1:] = pnl_step[1:] / np.maximum(balance[:-1], 1e-12)

        # SAVE CSV (NEW: market + entry columns)
        df_out = pd.DataFrame(
            {
                "t": t_idx,
                "time_stamp": time_stamp,      # NEW
                "market_open": market_open,    # NEW
                "market_close": market_close,  # NEW

                "equity": eq,
                "balance": balance,
                "roi": roi_cum,
                "pnl_step": pnl_step,
                "ret_step": ret_step,

                "traded": trd,
                "position": positions,
                "action": actions,
                "reward": rewards,

                "entry_price": entry_price,    # NEW
                "entry_side": entry_side,      # NEW (1=LONG entry, -1=SHORT entry)
            }
        )
        df_out.to_csv(OUT_DIR / f"equity_{name}.csv", index=False, encoding="utf-8")

    # ============================================================
    # TOTAL
    # ============================================================
    total_eq = 1.0
    for k in ("bt1", "bt2", "bt3"):
        total_eq *= float(split_final_eq.get(k, 1.0))

    total_balance = float(total_eq * INITIAL_BALANCE)
    total_roi = (total_balance / float(INITIAL_BALANCE)) - 1.0

    print("\n=== TOTAL (bt1 * bt2 * bt3) BEST (CONSTRAINED) ===")
    print(f"[TOTAL] final_equity={total_eq:.6f} | final_balance=${total_balance:,.2f} | ROI={total_roi*100:.2f}%")

    print(f"\nSaved to {OUT_DIR}")


if __name__ == "__main__":
    main()
