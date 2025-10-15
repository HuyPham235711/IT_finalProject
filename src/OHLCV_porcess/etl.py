import psycopg2
from .config import DB_CONFIG, BATCH_SIZE, TABLES
from .normalize import compute_scaler, save_scaler, load_scaler, apply_scaler
from .indicators import calc_sma, calc_rsi


def process_table(src, dst, scaler=None, fit_scaler=False):
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    cur.execute(f"SELECT COUNT(*) FROM {src}")
    total_rows = cur.fetchone()[0]

    print(f"🔹 Processing {src} ({total_rows:,} rows)...")

    if fit_scaler:
        cur.execute(f"SELECT time_stamp, open, high, low, close, volume FROM {src} ORDER BY time_stamp ASC")
        rows = cur.fetchall()
        scaler = compute_scaler(rows)
        save_scaler(scaler)
        print(f"Scaler fitted and saved: {scaler}")

    if not scaler:
        scaler = load_scaler()

    for offset in range(0, total_rows, BATCH_SIZE):
        cur.execute(f"""
            SELECT time_stamp, open, high, low, close, volume
            FROM {src}
            ORDER BY time_stamp ASC
            LIMIT {BATCH_SIZE} OFFSET {offset}
        """)
        rows = cur.fetchall()
        if not rows:
            break

        # normalize
        normalized = apply_scaler(rows, scaler)

        # indicators
        closes = [r[4] for r in rows]
        sma14 = calc_sma(closes, 14)
        rsi14 = calc_rsi(closes, 14)

        processed = []
        for i, row in enumerate(normalized):
            processed.append((
                row[0], row[1], row[2], row[3], row[4], row[5],
                sma14[i], rsi14[i]
            ))

        insert_query = f"""
        INSERT INTO {dst}
        (datetime, open, high, low, close, volume, sma14, rsi14)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
        """
        cur.executemany(insert_query, processed)
        conn.commit()
        print(f" Batch {offset//BATCH_SIZE + 1} done")

    cur.close()
    conn.close()

def run_etl_all():
    # Step 1: Train — fit scaler
    process_table(*TABLES["train"], fit_scaler=True)

    # Step 2–4: Apply same scaler cho valid/test/backtest
    process_table(*TABLES["valid"])
    process_table(*TABLES["test"])
    process_table(*TABLES["backtest"])

    print(" All datasets processed successfully.")
