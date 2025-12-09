# src/models/finbert/finbert_daily_backtest_pipeline.py
"""
Pipeline FinBERT OOS Backtest (one-shot):

Input:
    - Postgres:
        + it_final.media_backtest (datetime, title, ...)
        + it_final.processed_ohlcv_backtest (datetime, ...)
    - Model:
        + FinBERT đã fine-tune (classification 3 lớp: pos/neg/neu)

Output:
    - Numpy: finbert_daily_embeddings_backtest.npy
      (shape: [n_ohlcv_backtest_rows, 3])

Logic:
    1. Load media_backtest, chạy FinBERT để lấy logits (embeddings) cho từng bài.
    2. Gộp embeddings theo date (mean pooling).
    3. Align với processed_ohlcv_backtest (1h) theo date, ffill, fill 0.
    4. Lưu matrix [pos, neg, neu] theo từng timestamp OHLCV backtest.
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sqlalchemy import create_engine
from torch.nn.functional import softmax
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# === 0. Cấu hình chung ===

# Path model FinBERT đã fine-tune
MODEL_PATH = Path(
    r"E:\TDTu\TAI_LIEU\KY1-NAM5\DU_AN_CNTT\models\finBERT\finbert_finetuned_sampler_v2"
)

# Thư mục output cho backtest
OUTPUT_DIR = Path(
    r"E:\TDTu\TAI_LIEU\KY1-NAM5\DU_AN_CNTT\results\backtest\finbert"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Nơi lưu file daily embeddings cuối cùng
OUT_NPY_PATH = OUTPUT_DIR / "finbert_daily_embeddings_backtest.npy"

# Kết nối Postgres (sửa nếu bạn đổi password/DB)
PG_CONN_STR = "postgresql+psycopg2://postgres:123456789@localhost:5432/postgres"

BATCH_SIZE = 32


def main():
    start = time.time()
    print("=" * 80)
    print("[INFO] FinBERT DAILY BACKTEST PIPELINE starting...")

    # === 1. Load model FinBERT ===
    device = 0 if torch.cuda.is_available() else -1
    print(f"[INFO] Using device: {'cuda:0' if device == 0 else 'cpu'}")

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    # Mapping label (giống lúc fine-tune)
    model.config.id2label = {0: "positive", 1: "negative", 2: "neutral"}
    model.config.label2id = {"positive": 0, "negative": 1, "neutral": 2}

    model.to("cuda:0" if device == 0 else "cpu")
    model.eval()

    # === 2. Kết nối Postgres & load data ===
    engine = create_engine(PG_CONN_STR)

    # 2.1. Load media_backtest
    print("[INFO] Loading it_final.media_backtest ...")
    news_df = pd.read_sql(
        "SELECT datetime, title FROM it_final.media_backtest ORDER BY datetime",
        engine,
    )
    print(f"[INFO] media_backtest rows: {len(news_df)}")
    print(
        f"[INFO] media_backtest range: {news_df['datetime'].min()} → {news_df['datetime'].max()}"
    )

    # 2.2. Load OHLCV backtest timeline
    print("[INFO] Loading it_final.processed_ohlcv_backtest ...")
    ohlcv_df = pd.read_sql(
        "SELECT datetime FROM it_final.processed_ohlcv_backtest ORDER BY datetime",
        engine,
    )
    print(f"[INFO] processed_ohlcv_backtest rows: {len(ohlcv_df)}")
    print(
        f"[INFO] processed_ohlcv_backtest range: {ohlcv_df['datetime'].min()} → {ohlcv_df['datetime'].max()}"
    )

    # === 3. Chạy FinBERT để lấy logits (embeddings) ===
    print("[INFO] Running FinBERT inference to get logits...")
    all_logits = []

    titles = news_df["title"].astype(str).tolist()

    device_str = "cuda:0" if device == 0 else "cpu"

    for i in range(0, len(titles), BATCH_SIZE):
        batch_titles = titles[i : i + BATCH_SIZE]
        inputs = tokenizer(
            batch_titles,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(device_str)

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits  # shape: (batch, 3)

        all_logits.append(logits.cpu().numpy())

        if (i // BATCH_SIZE) % 50 == 0:
            print(f"[INFO] Processed {i}/{len(titles)} samples...")

    all_logits = np.concatenate(all_logits, axis=0)
    print(f"[INFO] Done inference. Logits shape: {all_logits.shape}")

    # === 4. Gộp theo ngày (mean pooling) ===
    print("[INFO] Aggregating sentiment by date...")

    news_df["date"] = news_df["datetime"].dt.date
    # all_logits: (N, 3) -> gán vào cột pos/neg/neu (logits)
    news_df[["pos", "neg", "neu"]] = all_logits

    daily_mean = (
        news_df.groupby("date")[["pos", "neg", "neu"]].mean().reset_index()
    )

    print(f"[INFO] Unique sentiment days (backtest): {len(daily_mean)}")

    # === 5. Align với OHLCV backtest (1h), ffill, fill 0.0 ===
    print("[INFO] Aligning sentiment with processed_ohlcv_backtest timeline...")

    ohlcv_df["date"] = ohlcv_df["datetime"].dt.date

    merged = (
        pd.merge(ohlcv_df, daily_mean, on="date", how="left")
        .sort_values("datetime")
        .reset_index(drop=True)
    )

    # FFill theo thời gian; đầu chuỗi chưa có gì -> 0.0
    merged[["pos", "neg", "neu"]] = (
        merged[["pos", "neg", "neu"]].ffill().fillna(0.0)
    )

    # === 6. Xuất numpy matrix ===
    sentiment_matrix = (
        merged[["pos", "neg", "neu"]].to_numpy().astype(np.float32)
    )
    
    # === CUT 60 rows để khớp với LSTM/Transformer lookback ===
    LOOKBACK = 60
    if len(sentiment_matrix) > LOOKBACK:
        sentiment_matrix = sentiment_matrix[LOOKBACK:]
        print(f"[INFO] Cut the first {LOOKBACK} rows → new shape: {sentiment_matrix.shape}")


    np.save(OUT_NPY_PATH, sentiment_matrix)

    print("✅ Saved FinBERT daily embeddings aligned with OHLCV 1h (BACKTEST)")
    print(f"[INFO] Output path: {OUT_NPY_PATH}")
    print(f"[INFO] Final shape: {sentiment_matrix.shape}")
    print(
        f"[INFO] Coverage days in media_backtest: {len(daily_mean['date'].unique())}"
    )

    runtime = time.time() - start
    print(f"[DONE] Pipeline finished in {runtime:.2f} s")
    print("=" * 80)


if __name__ == "__main__":
    main()
