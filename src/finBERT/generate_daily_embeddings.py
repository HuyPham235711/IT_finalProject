import numpy as np
import pandas as pd
from sqlalchemy import create_engine

# === 1. PostgreSQL connection ===
PG_CONN_STR = "postgresql+psycopg2://postgres:123456789@localhost:5432/postgres"
engine = create_engine(PG_CONN_STR)

# === 2. Load FinBERT embeddings & metadata ===
# Embeddings: [positive, negative, neutral] từ FinBERT inference
embeddings = np.load(
    "E:/TDTu/TAI_LIEU/KY1-NAM5/DU_AN_CNTT/results/finbert/sampler_v2/media_test_embeddings.npy"
)

# Metadata: cột datetime của từng bài viết trong media_test
news_df = pd.read_sql(
    "SELECT datetime FROM it_final.media_test ORDER BY datetime",
    engine
)

# Kiểm tra khớp số dòng
assert len(news_df) == embeddings.shape[0], (
    f"❌ Mismatch rows: news={len(news_df)}, emb={embeddings.shape[0]}"
)

# === 3. Gộp FinBERT sentiment theo ngày ===
# (mean pooling để giảm nhiễu, phản ánh tâm lý thị trường trong ngày)
news_df["date"] = news_df["datetime"].dt.date
news_df[["pos", "neg", "neu"]] = embeddings
daily_mean = news_df.groupby("date")[["pos", "neg", "neu"]].mean().reset_index()

print(f"[INFO] Unique sentiment days: {len(daily_mean)}")

# === 4. Lấy timeline OHLCV (1h timeframe) ===
ohlcv_df = pd.read_sql(
    "SELECT datetime FROM it_final.processed_ohlcv_test ORDER BY datetime",
    engine
)
ohlcv_df["date"] = ohlcv_df["datetime"].dt.date

print(f"[INFO] OHLCV timeline hours: {len(ohlcv_df)}")
print(f"[INFO] OHLCV date range: {ohlcv_df['datetime'].min()} → {ohlcv_df['datetime'].max()}")

# === 5. Merge sentiment theo ngày, forward fill ngày thiếu ===
merged = pd.merge(ohlcv_df, daily_mean, on="date", how="left").sort_values("datetime")
merged[["pos", "neg", "neu"]] = merged[["pos", "neg", "neu"]].ffill().fillna(0.0)

# === 6. Xuất file numpy ===
sentiment_matrix = merged[["pos", "neg", "neu"]].to_numpy().astype(np.float32)
out_path = (
    "E:/TDTu/TAI_LIEU/KY1-NAM5/DU_AN_CNTT/results/finbert/sampler_v2/finbert_daily_embeddings_test.npy"
)
np.save(out_path, sentiment_matrix)

print("✅ Saved FinBERT daily embeddings aligned with OHLCV 1h (test)")
print(f"[INFO] Output path: {out_path}")
print(f"[INFO] Final shape: {sentiment_matrix.shape}")
print(f"[INFO] Coverage days: {len(daily_mean['date'].unique())}")
