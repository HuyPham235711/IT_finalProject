from pathlib import Path
import os
from typing import Dict, Optional, Tuple

import pandas as pd
import matplotlib.pyplot as plt
from sqlalchemy import create_engine


PROJECT_ROOT = Path("E:/TDTu/TAI_LIEU/KY1-NAM5/DU_AN_CNTT")
BT_DIR = PROJECT_ROOT / "results" / "backtest" / "rl_sac"
FIG_DIR = BT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

SCHEMA = "it_final"

PARTS = ["part1", "part2", "part3"]

# Backtest CSV outputs (from run_backtest_sac)
# backtest_part1.csv ...
# backtest_part2.csv ...
# backtest_part3.csv ...

# Media tables you showed: it_final.media_backtest_part1 ...
MEDIA_TABLES = {
    "part1": "media_backtest_part1",
    "part2": "media_backtest_part2",
    "part3": "media_backtest_part3",
}


# ------------------------------------------------------------
# Backtest CSV load + normalize columns
# ------------------------------------------------------------
def _ensure_equity_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Chuẩn hóa tên cột để analysis không bị lệ thuộc 'balance' vs 'equity'.
    """
    df = df.copy()

    if "equity" not in df.columns and "balance" in df.columns:
        df["equity"] = df["balance"]
    if "balance" not in df.columns and "equity" in df.columns:
        df["balance"] = df["equity"]

    # if missing drawdown but has peak_equity
    if "drawdown" not in df.columns and ("peak_equity" in df.columns and "equity" in df.columns):
        peak = df["peak_equity"].replace(0, 1e-9)
        df["drawdown"] = (peak - df["equity"]) / peak

    return df


def load_part(part: str) -> pd.DataFrame:
    p = BT_DIR / f"backtest_{part}.csv"
    if not p.exists():
        raise FileNotFoundError(f"Missing: {p}")
    df = pd.read_csv(p)
    df = _ensure_equity_columns(df)
    return df


# ------------------------------------------------------------
# Media loading (Postgres)
# ------------------------------------------------------------
def _get_engine():
    if "PG_CONN_STR" not in os.environ:
        raise EnvironmentError("Missing PG_CONN_STR env var")
    return create_engine(os.environ["PG_CONN_STR"])


def load_media_part(part: str) -> pd.DataFrame:
    """
    Load media table: it_final.media_backtest_partX
    Expected columns (from your screenshot):
      - datetime
      - sentiment_label (negative/neutral/positive)
    """
    table = MEDIA_TABLES.get(part)
    if table is None:
        raise ValueError(f"Unknown part: {part}")

    engine = _get_engine()

    q = f"""
        SELECT datetime, sentiment_label
        FROM "{SCHEMA}"."{table}"
        ORDER BY datetime
    """
    df = pd.read_sql(q, engine)

    # normalize
    if "sentiment_label" in df.columns:
        df["sentiment_label"] = (
            df["sentiment_label"].astype(str).str.strip().str.lower()
        )

    return df


def media_label_counts(df_media: pd.DataFrame) -> pd.Series:
    if df_media is None or df_media.empty:
        return pd.Series(dtype=int)
    if "sentiment_label" not in df_media.columns:
        return pd.Series(dtype=int)
    return df_media["sentiment_label"].value_counts(dropna=False)


# ------------------------------------------------------------
# Plots (SAC)
# ------------------------------------------------------------
def plot_equity_drawdown(dfs: dict):
    fig, axes = plt.subplots(2, 1, figsize=(20, 10), sharex=True)

    for part, df in dfs.items():
        axes[0].plot(df["equity"], label=part)
    axes[0].set_title("Equity Curve")
    axes[0].set_ylabel("Equity")
    axes[0].grid(True)
    axes[0].legend()

    for part, df in dfs.items():
        axes[1].plot(df["drawdown"], label=part)
    axes[1].set_title("Drawdown")
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Drawdown")
    axes[1].grid(True)
    axes[1].legend()

    fig.suptitle("SAC Backtest – Equity & Drawdown (3 Parts)", fontsize=14)
    fig.tight_layout()
    out = FIG_DIR / "sac_equity_drawdown_3parts.png"
    fig.savefig(out, dpi=220)
    plt.close(fig)
    return out


def plot_position(dfs: dict):
    fig, ax = plt.subplots(figsize=(20, 4))

    for part, df in dfs.items():
        if "position" in df.columns:
            ax.plot(df["position"], label=part)

    ax.set_title("Position Over Time")
    ax.set_xlabel("Step")
    ax.set_ylabel("Position")
    ax.grid(True)
    ax.legend()

    fig.suptitle("SAC Backtest – Position (3 Parts)", fontsize=14)
    fig.tight_layout()
    out = FIG_DIR / "sac_position_3parts.png"
    fig.savefig(out, dpi=220)
    plt.close(fig)
    return out


def plot_roi(dfs: dict, initial_equity: float = 1000.0):
    fig, ax = plt.subplots(figsize=(20, 4))

    for part, df in dfs.items():
        roi = df["equity"] / float(initial_equity) - 1.0
        ax.plot(roi, label=part)

    ax.set_title("ROI")
    ax.set_xlabel("Step")
    ax.set_ylabel("ROI")
    ax.grid(True)
    ax.legend()

    fig.suptitle("SAC Backtest – ROI (3 Parts)", fontsize=14)
    fig.tight_layout()
    out = FIG_DIR / "sac_roi_3parts.png"
    fig.savefig(out, dpi=220)
    plt.close(fig)
    return out


# ------------------------------------------------------------
# Plots (Media)
# ------------------------------------------------------------
def plot_media_label_timeline(media_dfs: Dict[str, pd.DataFrame]):
    """
    Vẽ timeline label theo datetime (scatter/step-like).
    Map:
      negative -> -1
      neutral  ->  0
      positive -> +1
    """
    mapping = {"negative": -1, "neutral": 0, "positive": 1}

    fig, ax = plt.subplots(figsize=(20, 6))

    for part, dfm in media_dfs.items():
        if dfm is None or dfm.empty:
            continue
        if "datetime" not in dfm.columns or "sentiment_label" not in dfm.columns:
            continue

        x = pd.to_datetime(dfm["datetime"])
        y = dfm["sentiment_label"].map(mapping)
        # Nếu có label lạ, giữ NaN -> drop
        mask = y.notna()
        ax.plot(x[mask], y[mask], label=part, linewidth=1.2)

    ax.set_title("Media Sentiment Label Timeline (negative=-1, neutral=0, positive=+1)")
    ax.set_xlabel("Datetime")
    ax.set_ylabel("Label")
    ax.grid(True)
    ax.legend()

    fig.tight_layout()
    out = FIG_DIR / "media_sentiment_timeline_3parts.png"
    fig.savefig(out, dpi=220)
    plt.close(fig)
    return out


def plot_media_label_distribution(media_dfs: Dict[str, pd.DataFrame]):
    """
    Bar chart distribution label cho từng part.
    """
    fig, ax = plt.subplots(figsize=(18, 6))

    # build a combined table counts
    rows = []
    for part, dfm in media_dfs.items():
        if dfm is None or dfm.empty or "sentiment_label" not in dfm.columns:
            rows.append({"part": part, "negative": 0, "neutral": 0, "positive": 0, "total": 0})
            continue

        counts = dfm["sentiment_label"].value_counts()
        neg = int(counts.get("negative", 0))
        neu = int(counts.get("neutral", 0))
        pos = int(counts.get("positive", 0))
        total = int(len(dfm))

        rows.append({"part": part, "negative": neg, "neutral": neu, "positive": pos, "total": total})

    cdf = pd.DataFrame(rows).set_index("part")

    # stacked bars
    ax.bar(cdf.index, cdf["negative"], label="negative")
    ax.bar(cdf.index, cdf["neutral"], bottom=cdf["negative"], label="neutral")
    ax.bar(cdf.index, cdf["positive"], bottom=cdf["negative"] + cdf["neutral"], label="positive")

    ax.set_title("Media Sentiment Distribution (3 Parts)")
    ax.set_xlabel("Part")
    ax.set_ylabel("Count")
    ax.grid(True, axis="y")
    ax.legend()

    fig.tight_layout()
    out = FIG_DIR / "media_sentiment_distribution_3parts.png"
    fig.savefig(out, dpi=220)
    plt.close(fig)

    # also save the counts to csv for debugging
    out_csv = FIG_DIR / "media_sentiment_counts_3parts.csv"
    cdf.reset_index().to_csv(out_csv, index=False)

    return out, out_csv


# ------------------------------------------------------------
# Metrics summary (SAC backtest)
# ------------------------------------------------------------
def save_metrics_summary(dfs: dict):
    rows = []
    for part, df in dfs.items():
        rows.append({
            "part": part,
            "final_equity": float(df["equity"].iloc[-1]),
            "total_return": float(df["equity"].iloc[-1] / df["equity"].iloc[0] - 1.0),
            "max_drawdown": float(df["drawdown"].max()) if "drawdown" in df.columns else float("nan"),
            "avg_position": float(df["position"].mean()) if "position" in df.columns else float("nan"),
            "position_std": float(df["position"].std()) if "position" in df.columns else float("nan"),
            "avg_abs_action": float(df["action"].abs().mean()) if "action" in df.columns else float("nan"),
            "pct_abs_action_lt_0_05": float((df["action"].abs() < 0.05).mean()) if "action" in df.columns else float("nan"),
            "n_steps": int(len(df)),
        })

    out = FIG_DIR / "sac_metrics_summary_3parts.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    return out


def save_media_summary(media_dfs: Dict[str, pd.DataFrame]):
    rows = []
    for part, dfm in media_dfs.items():
        if dfm is None or dfm.empty or "sentiment_label" not in dfm.columns:
            rows.append({
                "part": part,
                "negative": 0,
                "neutral": 0,
                "positive": 0,
                "total": 0,
                "pct_negative": 0.0,
                "pct_neutral": 0.0,
                "pct_positive": 0.0,
            })
            continue

        counts = dfm["sentiment_label"].value_counts()
        neg = int(counts.get("negative", 0))
        neu = int(counts.get("neutral", 0))
        pos = int(counts.get("positive", 0))
        total = int(len(dfm)) if len(dfm) > 0 else 1

        rows.append({
            "part": part,
            "negative": neg,
            "neutral": neu,
            "positive": pos,
            "total": total,
            "pct_negative": float(neg / total),
            "pct_neutral": float(neu / total),
            "pct_positive": float(pos / total),
        })

    out = FIG_DIR / "media_metrics_summary_3parts.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    return out


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
def main():
    print("=== SAC ANALYSIS (3 PARTS) ===")

    # Load backtest csv
    dfs = {p: load_part(p) for p in PARTS}

    # Load media labels from Postgres tables
    media_dfs = {}
    for p in PARTS:
        try:
            mdf = load_media_part(p)
            media_dfs[p] = mdf
            c = media_label_counts(mdf)
            print(f"\n[MEDIA] {p} label counts:")
            print(c.to_string())
        except Exception as e:
            media_dfs[p] = pd.DataFrame()
            print(f"\n[MEDIA] {p} load failed: {e}")

    # SAC plots
    f1 = plot_equity_drawdown(dfs)
    f2 = plot_position(dfs)
    f3 = plot_roi(dfs)
    f4 = save_metrics_summary(dfs)

    # Media plots
    f5 = plot_media_label_timeline(media_dfs)
    f6, f6_csv = plot_media_label_distribution(media_dfs)
    f7 = save_media_summary(media_dfs)

    print("\n✅ Saved:")
    print(f" - {f1}")
    print(f" - {f2}")
    print(f" - {f3}")
    print(f" - {f4}")
    print(f" - {f5}")
    print(f" - {f6}")
    print(f" - {f6_csv}")
    print(f" - {f7}")
    print("=== DONE ===")


if __name__ == "__main__":
    main()
