from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


DEFAULT_PROJECT_ROOT = Path("E:/TDTu/TAI_LIEU/KY1-NAM5/DU_AN_CNTT")
DEFAULT_CSV_DIR = DEFAULT_PROJECT_ROOT / "results" / "backtest" / "bandit_fusion"
DEFAULT_OUT_DIR = DEFAULT_CSV_DIR / "analysis_plots"


def _require_cols(df: pd.DataFrame, cols: Tuple[str, ...], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"[{name}] Missing columns: {missing}. Found columns={list(df.columns)}")


def load_split(csv_dir: Path, split_name: str) -> pd.DataFrame:
    path = csv_dir / f"equity_{split_name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")

    df = pd.read_csv(path)

    # required for your plots
    _require_cols(
        df,
        (
            "t",
            "time_stamp",
            "market_close",
            "balance",
            "entry_side",
            "entry_price",
        ),
        name=split_name,
    )

    # parse timestamp (safe)
    df["time_stamp"] = pd.to_datetime(df["time_stamp"], errors="coerce")

    # enforce numeric
    for c in ["t", "market_close", "balance", "entry_side", "entry_price"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # drop rows with no step index
    df = df.dropna(subset=["t"]).sort_values("t").reset_index(drop=True)
    return df


def compute_drawdown_pct(balance: np.ndarray) -> np.ndarray:
    bal = np.asarray(balance, dtype=np.float64)
    if bal.size == 0:
        return bal
    peak = np.maximum.accumulate(bal)
    dd = (peak - bal) / np.maximum(peak, 1e-12)
    return dd * 100.0


def plot_balance_vs_market(df: pd.DataFrame, split_name: str, out_dir: Path) -> Path:
    # series
    x = df["t"].to_numpy()
    ts = df["time_stamp"]
    bal = df["balance"].to_numpy(dtype=np.float64)
    px = df["market_close"].to_numpy(dtype=np.float64)

    # entry markers
    entry_side = df["entry_side"].to_numpy(dtype=np.float64)
    entry_px = df["entry_price"].to_numpy(dtype=np.float64)

    long_mask = (entry_side == 1) & np.isfinite(entry_px)
    short_mask = (entry_side == -1) & np.isfinite(entry_px)

    fig, ax1 = plt.subplots(figsize=(20, 7), dpi=130)

    # Left axis: balance
    ax1.plot(x, bal, linewidth=1.6, label="Balance (USD)")
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Balance (USD)")
    ax1.grid(True, alpha=0.25)

    # Right axis: market price + entry points
    ax2 = ax1.twinx()
    ax2.plot(x, px, linewidth=1.2, alpha=0.9, label="Market Close Price")

    # ENTRY markers: required colors (user asked explicitly)
    # LONG = green, SHORT = red
    ax2.scatter(
        x[long_mask],
        entry_px[long_mask],
        s=28,
        marker="^",
        c="green",
        label="LONG Entry",
        zorder=5,
    )
    ax2.scatter(
        x[short_mask],
        entry_px[short_mask],
        s=28,
        marker="v",
        c="red",
        label="SHORT Entry",
        zorder=5,
    )

    ax2.set_ylabel("Market Price")

    # Title with time range
    ts_min = ts.min()
    ts_max = ts.max()
    title_range = ""
    if pd.notna(ts_min) and pd.notna(ts_max):
        title_range = f" | {ts_min} → {ts_max}"
    plt.title(f"{split_name}: Balance vs Market Price + Agent Entry Price{title_range}")

    # combined legend
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper left")

    out_path = out_dir / f"{split_name}_balance_vs_market.png"
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_drawdown(df: pd.DataFrame, split_name: str, out_dir: Path) -> Path:
    x = df["t"].to_numpy()
    bal = df["balance"].to_numpy(dtype=np.float64)
    dd_pct = compute_drawdown_pct(bal)

    fig, ax = plt.subplots(figsize=(20, 6), dpi=130)
    ax.plot(x, dd_pct, linewidth=1.6, label="Drawdown (%)")
    ax.set_title(f"{split_name}: Drawdown (%)")
    ax.set_xlabel("Step")
    ax.set_ylabel("Drawdown (%)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left")

    out_path = out_dir / f"{split_name}_drawdown.png"
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_dir", type=str, default=str(DEFAULT_CSV_DIR), help="Directory containing equity_bt*.csv")
    parser.add_argument("--out_dir", type=str, default=str(DEFAULT_OUT_DIR), help="Output directory for plots")
    args = parser.parse_args()

    csv_dir = Path(args.csv_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    splits = ["bt1", "bt2", "bt3"]

    print(f"[ANALYSIS] csv_dir={csv_dir}")
    print(f"[ANALYSIS] out_dir={out_dir}")
    print("[ANALYSIS] splits=", splits)

    saved = []
    for s in splits:
        df = load_split(csv_dir, s)

        p1 = plot_balance_vs_market(df, s, out_dir)
        p2 = plot_drawdown(df, s, out_dir)

        saved.append((s, p1, p2))
        print(f"[SAVED] {s}:")
        print(f"  - {p1}")
        print(f"  - {p2}")

    print("\nDone.")


if __name__ == "__main__":
    main()
