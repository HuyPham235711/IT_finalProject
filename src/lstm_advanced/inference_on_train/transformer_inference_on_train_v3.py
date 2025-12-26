import torch
import numpy as np
from pathlib import Path

from src.lstm.pipeline_lstm_baseline.data_loader import load_table_from_postgres
from src.lstm_advanced.train.train_transformer_v3 import TimeSeriesTransformerV3


# ============================================================
# Create sequences (GIỐNG ATT-CNN)
# ============================================================
def create_sequences(df, feature_cols, target_col, lookback):
    df_seq = df[feature_cols + [target_col]].copy()
    data = df_seq.values.astype(np.float32)

    X = []
    for i in range(len(data) - lookback):
        X.append(data[i:i + lookback, :-1])
    return np.array(X)


# ============================================================
# MAIN
# ============================================================
def main():
    print("=== TRANSFORMER v3 | TRAIN EMBEDDINGS ===")

    SCHEMA = "it_final"
    TABLE = "processed_ohlcv_train"
    LOOKBACK = 60
    BATCH_SIZE = 256

    FEATURES = ["open", "high", "low", "close", "volume", "sma14", "rsi14"]
    TARGET = "close"

    OUT_DIR = Path("results/lstm_advanced/v3/train_inference")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------
    df = load_table_from_postgres(TABLE, SCHEMA)
    X_np = create_sequences(df, FEATURES, TARGET, LOOKBACK)

    print(f"[INFO] Train sequences: {X_np.shape}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --------------------------------------------------------
    # Load Transformer v3
    # --------------------------------------------------------
    model = TimeSeriesTransformerV3(
        input_dim=X_np.shape[2],
        embed_dim=256,      # ⚠️ PHẢI KHỚP CHECKPOINT
        num_heads=8,
        num_layers=2,
        dropout=0.2,
    ).to(device)

    ckpt = "results/lstm_advanced/v3/transformer_v3.pt"
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    # --------------------------------------------------------
    # Inference batched
    # --------------------------------------------------------
    embs = []
    with torch.no_grad():
        for i in range(0, len(X_np), BATCH_SIZE):
            xb = torch.tensor(X_np[i:i + BATCH_SIZE]).to(device)
            _, emb = model(xb)                 # emb: [B, 2*embed_dim]
            embs.append(emb.cpu().numpy())

            if i % 5000 == 0:
                print(f"[INFO] processed {i}/{len(X_np)}")

    final_emb = np.concatenate(embs, axis=0)

    out_path = OUT_DIR / "transformer_embeddings_train_v3.npy"
    np.save(out_path, final_emb)

    print(f"✅ Saved Transformer embeddings: {final_emb.shape} → {out_path}")


if __name__ == "__main__":
    main()
