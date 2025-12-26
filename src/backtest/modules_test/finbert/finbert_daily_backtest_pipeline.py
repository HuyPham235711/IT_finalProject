import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sqlalchemy import create_engine
from transformers import AutoModelForSequenceClassification, AutoTokenizer


# =========================================================
# CONFIG
# =========================================================
MODEL_PATH = Path(
    r"E:\TDTu\TAI_LIEU\KY1-NAM5\DU_AN_CNTT\models\finBERT\finbert_finetuned_sampler_v2"
)

OUTPUT_DIR = Path(
    r"E:\TDTu\TAI_LIEU\KY1-NAM5\DU_AN_CNTT\results\backtest\finbert"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PG_CONN_STR = "postgresql+psycopg2://postgres:123456789@localhost:5432/postgres"
BATCH_SIZE = 32
LOOKBACK = 60

BACKTEST_PARTS = {
    "part1": ("it_final.media_backtest_part1", "it_final.processed_ohlcv_backtest_part1"),
    "part2": ("it_final.media_backtest_part2", "it_final.processed_ohlcv_backtest_part2"),
    "part3": ("it_final.media_backtest_part3", "it_final.processed_ohlcv_backtest_part3"),
}


# =========================================================
# MAIN
# =========================================================
def main():
    start = time.time()
    print("=" * 80)
    print("[INFO] FinBERT DAILY BACKTEST PIPELINE (MULTI-PART)")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Device: {device}")

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    model.to(device)
    model.eval()

    engine = create_engine(PG_CONN_STR)

    for part, (media_tbl, ohlcv_tbl) in BACKTEST_PARTS.items():
        print(f"\n=== BACKTEST {part.upper()} ===")

        news_df = pd.read_sql(
            f"SELECT datetime, title FROM {media_tbl} ORDER BY datetime",
            engine,
        )

        ohlcv_df = pd.read_sql(
            f"SELECT datetime FROM {ohlcv_tbl} ORDER BY datetime",
            engine,
        )

        titles = news_df["title"].astype(str).tolist()
        all_logits = []

        for i in range(0, len(titles), BATCH_SIZE):
            batch = titles[i:i + BATCH_SIZE]
            inputs = tokenizer(
                batch,
                padding=True,
                truncation=True,
                return_tensors="pt"
            ).to(device)

            with torch.no_grad():
                logits = model(**inputs).logits

            all_logits.append(logits.cpu().numpy())

        if len(all_logits) == 0:
            print(f"[WARN] No news for {part}, skipping.")
            continue

        all_logits = np.concatenate(all_logits, axis=0)

        news_df["date"] = news_df["datetime"].dt.date
        news_df[["pos", "neg", "neu"]] = all_logits

        daily_mean = (
            news_df.groupby("date")[["pos", "neg", "neu"]]
            .mean()
            .reset_index()
        )

        ohlcv_df["date"] = ohlcv_df["datetime"].dt.date
        merged = (
            pd.merge(ohlcv_df, daily_mean, on="date", how="left")
            .sort_values("datetime")
            .reset_index(drop=True)
        )

        merged[["pos", "neg", "neu"]] = (
            merged[["pos", "neg", "neu"]].ffill().fillna(0.0)
        )

        sentiment = merged[["pos", "neg", "neu"]].to_numpy().astype(np.float32)

        if len(sentiment) > LOOKBACK:
            sentiment = sentiment[LOOKBACK:]

        out_path = OUTPUT_DIR / f"finbert_daily_embeddings_backtest_{part}.npy"
        np.save(out_path, sentiment)

        print(f"[OK] Saved {out_path} | shape={sentiment.shape}")

    print(f"[DONE] Runtime: {time.time() - start:.2f}s")
    print("=" * 80)


if __name__ == "__main__":
    main()
