import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# ============================================================
# CONFIG
# ============================================================
PROJECT_ROOT = Path("E:/TDTu/TAI_LIEU/KY1-NAM5/DU_AN_CNTT")

BACKTEST_DIR = (
    PROJECT_ROOT
    / "results"
    / "backtest"
    / "rl_dqn"
    / "dqn_behavioral_roi"
)

CSV_PATH = BACKTEST_DIR / "backtest_steps.csv"
OUT_DIR = BACKTEST_DIR / "analysis_plots_wide"
OUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "lines.linewidth": 1.8,
})


# ============================================================
# LOAD DATA
# ============================================================
df = pd.read_csv(CSV_PATH)
df["datetime"] = pd.to_datetime(df["datetime"])


# ============================================================
# 1. EQUITY CURVE
# ============================================================
plt.figure(figsize=(20, 6))
plt.plot(df["datetime"], df["balance"])
plt.title("DQN Backtest – Equity Curve")
plt.xlabel("Time")
plt.ylabel("Balance")
plt.grid(True)
plt.tight_layout()
plt.savefig(OUT_DIR / "01_equity_curve.png", dpi=150)
plt.close()


# ============================================================
# 2. ROI CURVE
# ============================================================
plt.figure(figsize=(20, 6))
plt.plot(df["datetime"], df["roi_total"])
plt.title("DQN Backtest – Total ROI")
plt.xlabel("Time")
plt.ylabel("ROI")
plt.grid(True)
plt.tight_layout()
plt.savefig(OUT_DIR / "02_roi_curve.png", dpi=150)
plt.close()


# ============================================================
# 3. PRICE + POSITION
# ============================================================
plt.figure(figsize=(20, 6))
plt.plot(df["datetime"], df["price"], label="Price", alpha=0.9)

long_idx = df["position"] == 1
short_idx = df["position"] == -1

plt.scatter(
    df.loc[long_idx, "datetime"],
    df.loc[long_idx, "price"],
    marker="^",
    s=40,
    label="LONG",
)
plt.scatter(
    df.loc[short_idx, "datetime"],
    df.loc[short_idx, "price"],
    marker="v",
    s=40,
    label="SHORT",
)

plt.title("Price with DQN Positions")
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(OUT_DIR / "03_price_with_positions.png", dpi=150)
plt.close()


# ============================================================
# 4. PnL PER STEP
# ============================================================
plt.figure(figsize=(20, 6))
plt.plot(df["datetime"], df["pnl"])
plt.axhline(0, linestyle="--", linewidth=1)
plt.title("PnL per Step")
plt.xlabel("Time")
plt.ylabel("PnL")
plt.grid(True)
plt.tight_layout()
plt.savefig(OUT_DIR / "04_pnl_per_step.png", dpi=150)
plt.close()


# ============================================================
# 5. ACTION DISTRIBUTION
# ============================================================
plt.figure(figsize=(10, 6))
df["action"].value_counts().sort_index().plot(kind="bar")
plt.xticks([0, 1, 2], ["HOLD", "LONG", "SHORT"], rotation=0)
plt.title("Action Distribution")
plt.ylabel("Count")
plt.grid(axis="y")
plt.tight_layout()
plt.savefig(OUT_DIR / "05_action_distribution.png", dpi=150)
plt.close()


# ============================================================
# 6. EPSILON DEBUG
# ============================================================
if "greedy_action" in df.columns:
    plt.figure(figsize=(20, 4))
    mismatch = (df["action"] != df["greedy_action"]).astype(int)
    plt.plot(df["datetime"], mismatch)
    plt.title("Epsilon Debug – Action != Greedy Action (1 = Random)")
    plt.xlabel("Time")
    plt.ylabel("Mismatch")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "06_epsilon_mismatch.png", dpi=150)
    plt.close()


print("=== DONE: WIDE DQN BACKTEST PLOTS SAVED ===")
print(f"Output folder: {OUT_DIR}")
