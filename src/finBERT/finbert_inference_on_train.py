import numpy as np
import pandas as pd
import torch
from pathlib import Path
from sqlalchemy import create_engine, text
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ======================================================
# CONFIG
# ======================================================
PG = "postgresql+psycopg2://postgres:123456789@localhost:5432/postgres"

MODEL_DIR = Path(
    "E:/TDTu/TAI_LIEU/KY1-NAM5/DU_AN_CNTT/models/finBERT/finbert_finetuned_sampler_v2"
)

OUT_PATH = Path(
    "E:/TDTu/TAI_LIEU/KY1-NAM5/DU_AN_CNTT/results/finbert/train_inference/finbert_daily_embeddings_train.npy"
)

TABLE_MEDIA = "it_final.media_train"
TABLE_OHLCV = "it_final.processed_ohlcv_train"


# ======================================================
# MAIN
# ======================================================
def main():
    print("=== FINBERT DAILY TRAIN PIPELINE ===")

    engine = create_engine(PG)

    # --------------------------------------------------
    # Load media
    # --------------------------------------------------
    q = text(f"""
        SELECT datetime, title
        FROM {TABLE_MEDIA}
        WHERE title IS NOT NULL
    """)
    df = pd.read_sql(q, engine)
    df["datetime"] = pd.to_datetime(df["datetime"]).dt.floor("h")
    print(f"[INFO] media rows = {len(df)}")

    # --------------------------------------------------
    # Load OHLCV timeline (for alignment)
    # --------------------------------------------------
    q2 = text(f"""
        SELECT datetime FROM {TABLE_OHLCV} ORDER BY datetime
    """)
    oh = pd.read_sql(q2, engine)
    oh["datetime"] = pd.to_datetime(oh["datetime"])
    print(f"[INFO] OHLCV rows = {len(oh)}")

    # --------------------------------------------------
    # Load FinBERT model
    # --------------------------------------------------
    print("[INFO] Loading FinBERT v2...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.eval().cuda()

    # --------------------------------------------------
    # Inference for each row
    # --------------------------------------------------
    logits_list = []
    titles = df["title"].tolist()

    with torch.no_grad():
        for i, text_str in enumerate(titles):
            tok = tokenizer(
                text_str,
                truncation=True,
                max_length=128,
                return_tensors="pt",
            ).to("cuda")

            out = model(**tok)
            logits = out.logits.squeeze().cpu().numpy()
            logits_list.append(logits)

            if i % 1000 == 0:
                print(f"[INFO] processed {i}/{len(titles)}")

    logits_arr = np.vstack(logits_list)
    df["logit_pos"] = logits_arr[:, 0]
    df["logit_neg"] = logits_arr[:, 1]
    df["logit_neu"] = logits_arr[:, 2]

    # --------------------------------------------------
    # DAILY aggregate → mean
    # --------------------------------------------------
    daily = (
        df.groupby("datetime")[["logit_pos", "logit_neg", "logit_neu"]]
        .mean()
        .reset_index()
    )
    print("[INFO] daily sentiment rows =", len(daily))

    # --------------------------------------------------
    # Align to OHLCV timeline
    # --------------------------------------------------
    merged = oh.merge(daily, on="datetime", how="left")
    merged.fillna(0.0, inplace=True)

    emb = merged[["logit_pos", "logit_neg", "logit_neu"]].values
    print("[INFO] Final embedding shape:", emb.shape)

    # --------------------------------------------------
    # SAVE
    # --------------------------------------------------
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.save(OUT_PATH, emb)
    print(f"Saved → {OUT_PATH}")


if __name__ == "__main__":
    main()
