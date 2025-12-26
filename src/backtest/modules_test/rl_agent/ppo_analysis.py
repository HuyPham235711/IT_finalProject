import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================
# PATHS
# ============================================================
PROJECT_ROOT = Path("E:/TDTu/TAI_LIEU/KY1-NAM5/DU_AN_CNTT")
BT_DIR = PROJECT_ROOT / "results" / "backtest" / "rl_ppo"
FIG_DIR = BT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

PARTS = ["part1", "part2", "part3"]

# ============================================================
# LOAD DATA
# ============================================================
def load_part(part):
    path = BT_DIR / f"backtest_{part}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing backtest file: {path}")
    return pd.read_csv(path)


# ============================================================
# PLOT FUNCTIONS
# ============================================================
def plot_equity(ax, dfs):
    for part, df in dfs.items():
        ax.plot(df["balance"], label=part)
    ax.set_title("Equity Curve (Balance)")
    ax.set_xlabel("Step")
    ax.set_ylabel("Balance")
    ax.legend()
    ax.grid(True)


def plot_drawdown(ax, dfs):
    for part, df in dfs.items():
        ax.plot(df["drawdown"], label=part)
    ax.set_title("Drawdown")
    ax.set_xlabel("Step")
    ax.set_ylabel("Drawdown")
    ax.legend()
    ax.grid(True)


def plot_position(ax, dfs):
    for part, df in dfs.items():
        ax.plot(df["position"], label=part)
    ax.set_title("Position Over Time")
    ax.set_xlabel("Step")
    ax.set_ylabel("Position")
    ax.legend()
    ax.grid(True)


def plot_roi(ax, dfs, initial_balance=1000.0):
    for part, df in dfs.items():
        roi = df["balance"] / initial_balance - 1.0
        ax.plot(roi, label=part)
    ax.set_title("ROI")
    ax.set_xlabel("Step")
    ax.set_ylabel("ROI")
    ax.legend()
    ax.grid(True)


# ============================================================
# MAIN
# ============================================================
def main():
    print("=== PPO ANALYSIS – 3 BACKTEST PARTS ===")

    dfs = {part: load_part(part) for part in PARTS}

    # --------------------------------------------------------
    # FIGURE 1: Equity + Drawdown
    # --------------------------------------------------------
    fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True)

    plot_equity(axes[0], dfs)
    plot_drawdown(axes[1], dfs)

    fig.suptitle("PPO Backtest – Equity & Drawdown (3 Parts)", fontsize=14)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "ppo_equity_drawdown_3parts.png", dpi=200)
    plt.close(fig)

    # --------------------------------------------------------
    # FIGURE 2: Position
    # --------------------------------------------------------
    fig, ax = plt.subplots(figsize=(16, 4))
    plot_position(ax, dfs)

    fig.suptitle("PPO Backtest – Position (3 Parts)", fontsize=14)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "ppo_position_3parts.png", dpi=200)
    plt.close(fig)

    # --------------------------------------------------------
    # FIGURE 3: ROI
    # --------------------------------------------------------
    fig, ax = plt.subplots(figsize=(16, 4))
    plot_roi(ax, dfs)

    fig.suptitle("PPO Backtest – ROI (3 Parts)", fontsize=14)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "ppo_roi_3parts.png", dpi=200)
    plt.close(fig)

    # --------------------------------------------------------
    # METRICS SUMMARY TABLE
    # --------------------------------------------------------
    summary = []

    for part, df in dfs.items():
        summary.append({
            "part": part,
            "final_balance": df["balance"].iloc[-1],
            "total_return": df["balance"].iloc[-1] / df["balance"].iloc[0] - 1.0,
            "max_drawdown": df["drawdown"].max(),
            "avg_position": df["position"].mean(),
            "position_std": df["position"].std(),
            "n_steps": len(df),
        })

    summary_df = pd.DataFrame(summary)
    summary_path = FIG_DIR / "ppo_metrics_summary_3parts.csv"
    summary_df.to_csv(summary_path, index=False)

    print("✅ Saved figures:")
    print(f" - {FIG_DIR / 'ppo_equity_drawdown_3parts.png'}")
    print(f" - {FIG_DIR / 'ppo_position_3parts.png'}")
    print(f" - {FIG_DIR / 'ppo_roi_3parts.png'}")
    print("✅ Saved metrics summary:")
    print(f" - {summary_path}")

    print("\n=== DONE ===")


if __name__ == "__main__":
    main()
