import os
import json
import psycopg2
from pathlib import Path
from typing import List, Tuple

# =========================================================
# CONFIG
# =========================================================
DB_CONFIG = {
    "dbname": "postgres",
    "user": "postgres",
    "password": "123456789",
    "host": "localhost",
    "port": "5432"
}

BATCH_SIZE = 1000

PROJECT_ROOT = Path("E:/TDTu/TAI_LIEU/KY1-NAM5/DU_AN_CNTT")
SCALER_DIR = PROJECT_ROOT / "results" / "scaler"
SCALER_DIR.mkdir(parents=True, exist_ok=True)
SCALER_PATH = SCALER_DIR / "ohlcv_minmax_train.json"

# =========================================================
# TABLE MAPPING
# =========================================================
TABLES = {
    "train": (
        "it_final.ohlcv_train",
        "it_final.processed_ohlcv_train2"
    ),
    "valid": (
        "it_final.ohlcv_valid",
        "it_final.processed_ohlcv_valid2"
    ),
    "test": (
        "it_final.ohlcv_test",
        "it_final.processed_ohlcv_test2"
    ),
    "backtest_part1": (
        "it_final.ohlcv_backtest_part1",
        "it_final.processed_ohlcv_backtest_part1"
    ),
    "backtest_part2": (
        "it_final.ohlcv_backtest_part2",
        "it_final.processed_ohlcv_backtest_part2"
    ),
    "backtest_part3": (
        "it_final.ohlcv_backtest_part3",
        "it_final.processed_ohlcv_backtest_part3"
    ),
}

# =========================================================
# INDICATORS
# =========================================================
def calc_sma(values: List[float], period: int = 14):
    out = []
    for i in range(len(values)):
        if i < period - 1:
            out.append(None)
        else:
            out.append(sum(values[i - period + 1:i + 1]) / period)
    return out


def calc_rsi(closes: List[float], period: int = 14):
    rsi = [None] * len(closes)
    gains, losses = [], []

    for i in range(1, len(closes)):
        diff = closes[i] - closes[i - 1]
        gains.append(max(diff, 0))
        losses.append(abs(min(diff, 0)))

        if i >= period:
            avg_gain = sum(gains[i - period:i]) / period
            avg_loss = sum(losses[i - period:i]) / period
            if avg_loss == 0:
                rsi[i] = 100.0
            else:
                rs = avg_gain / avg_loss
                rsi[i] = 100 - (100 / (1 + rs))
    return rsi

# =========================================================
# SCALER
# =========================================================
def fit_minmax_scaler(rows: List[Tuple]):
    cols = list(zip(*rows))[1:]  # bỏ datetime
    mins = [min(col) for col in cols]
    maxs = [max(col) for col in cols]
    return mins, maxs


def save_scaler(mins, maxs):
    payload = {
        "scaler_type": "minmax",
        "fit_on": "it_final.ohlcv_train",
        "fit_period": "2017-09-23 → 2024-11-25",
        "features": ["open", "high", "low", "close", "volume"],
        "mins": mins,
        "maxs": maxs
    }
    with open(SCALER_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved scaler → {SCALER_PATH}")


def load_scaler():
    with open(SCALER_PATH, "r", encoding="utf-8") as f:
        s = json.load(f)
    return s["mins"], s["maxs"]


def apply_minmax(rows, mins, maxs):
    out = []
    for r in rows:
        ts, o, h, l, c, v = r
        vals = [o, h, l, c, v]
        norm = []
        for i, val in enumerate(vals):
            if maxs[i] == mins[i]:
                norm.append(0.0)
            else:
                norm.append((val - mins[i]) / (maxs[i] - mins[i]))
        out.append((ts, *norm))
    return out

# =========================================================
# ETL CORE
# =========================================================
def process_table(src: str, dst: str, fit_scaler: bool = False):
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    cur.execute(f"SELECT COUNT(*) FROM {src}")
    total = cur.fetchone()[0]
    print(f"\nProcessing {src} → {dst} ({total:,} rows)")

    if fit_scaler:
        cur.execute(
            f"SELECT time_stamp, open, high, low, close, volume "
            f"FROM {src} ORDER BY time_stamp"
        )
        rows = cur.fetchall()
        mins, maxs = fit_minmax_scaler(rows)
        save_scaler(mins, maxs)
    else:
        mins, maxs = load_scaler()

    for offset in range(0, total, BATCH_SIZE):
        cur.execute(
            f"""
            SELECT time_stamp, open, high, low, close, volume
            FROM {src}
            ORDER BY time_stamp
            LIMIT {BATCH_SIZE} OFFSET {offset}
            """
        )
        rows = cur.fetchall()
        if not rows:
            break

        normalized = apply_minmax(rows, mins, maxs)
        closes = [r[4] for r in rows]
        sma14 = calc_sma(closes, 14)
        rsi14 = calc_rsi(closes, 14)

        batch = []
        for i, r in enumerate(normalized):
            batch.append((
                r[0], r[1], r[2], r[3], r[4], r[5],
                sma14[i], rsi14[i]
            ))

        cur.executemany(
            f"""
            INSERT INTO {dst}
            (datetime, open, high, low, close, volume, sma14, rsi14)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
            """,
            batch
        )
        conn.commit()
        print(f"  Batch {offset // BATCH_SIZE + 1} done")

    cur.close()
    conn.close()


# =========================================================
# MAIN
# =========================================================
def main():
    # 1. TRAIN → fit scaler
    process_table(*TABLES["train"], fit_scaler=True)

    # 2. VALID / TEST
    process_table(*TABLES["valid"])
    process_table(*TABLES["test"])

    # 3. BACKTEST PARTS
    process_table(*TABLES["backtest_part1"])
    process_table(*TABLES["backtest_part2"])
    process_table(*TABLES["backtest_part3"])

    print("\nAll datasets processed successfully.")


if __name__ == "__main__":
    main()
