import torch
import numpy as np
from pathlib import Path
from src.pipeline_baseline.data_loader import load_table_from_postgres
from src.lstm_advanced.train_att_cnn_lstm import AttCNNLSTM


# ============================================================
# Create sequences
# ============================================================
def create_sequences(df, lookback=60):
    data = df.drop(columns=["datetime"]).values.astype(np.float32)
    X = []

    for i in range(len(data) - lookback):
        X.append(data[i:i+lookback, :-1])  # 60 × 6

    return np.array(X)


# ============================================================
# MAIN
# ============================================================
def main():
    print("=== Generating ATT-CNN-LSTM TRAIN Embeddings (BATCHED) ===")

    SCHEMA = "it_final"
    TABLE_TRAIN = "processed_ohlcv_train"
    LOOKBACK = 60
    BATCH_SIZE = 256   # GPU-safe

    OUT_DIR = Path("results/lstm_advanced/train_inference/")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1) LOAD TRAIN DATA
    df = load_table_from_postgres(TABLE_TRAIN, SCHEMA)
    print("[INFO] Loaded TRAIN:", df.shape)

    X_np = create_sequences(df, LOOKBACK)
    print("[INFO] Train sequences:", X_np.shape)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2) LOAD TRAINED MODEL
    ckpt_path = "results/lstm_advanced/att_cnn_lstm_baseline.pt"
    print("[INFO] Loading model:", ckpt_path)

    model = AttCNNLSTM(
        input_dim=X_np.shape[2],
        hidden_dim=768,
        num_layers=2,
        dropout=0.2
    ).to(device)

    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    # 3) INFERENCE BATCHED
    all_embeddings = []

    with torch.no_grad():
        n = X_np.shape[0]
        for i in range(0, n, BATCH_SIZE):
            batch = X_np[i:i+BATCH_SIZE]

            batch_t = torch.tensor(batch, dtype=torch.float32).to(device)

            _, emb = model(batch_t)    # (batch, 2304)

            all_embeddings.append(emb.cpu().numpy())

            if i % 2000 == 0:
                print(f"[INFO] processed {i}/{n}")

    # 4) CONCAT ALL EMBEDDINGS
    final_emb = np.concatenate(all_embeddings, axis=0)

    out_path = OUT_DIR / "att_cnn_lstm_embeddings_train.npy"
    np.save(out_path, final_emb)

    print(f"✅ Saved ATT-CNN-LSTM TRAIN embeddings → {out_path} | shape={final_emb.shape}")


if __name__ == "__main__":
    main()
