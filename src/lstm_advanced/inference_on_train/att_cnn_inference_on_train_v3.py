import torch
import numpy as np
from pathlib import Path

from src.lstm.pipeline_lstm_baseline.data_loader import load_table_from_postgres
from src.lstm_advanced.train.train_att_cnn_lstm_v3 import AttCNNLSTM_v3


def create_sequences(df, feature_cols, target_col, lookback):
    df_seq = df[feature_cols + [target_col]].copy()
    data = df_seq.values.astype(np.float32)

    X = []
    for i in range(len(data) - lookback):
        X.append(data[i:i+lookback, :-1])
    return np.array(X)


def main():
    print("=== ATT-CNN-LSTM v3 | TRAIN EMBEDDINGS ===")

    SCHEMA = "it_final"
    TABLE = "processed_ohlcv_train"
    LOOKBACK = 60
    BATCH_SIZE = 256

    FEATURES = ["open","high","low","close","volume","sma14","rsi14"]
    TARGET = "close"

    OUT_DIR = Path("results/lstm_advanced/v3/train_inference")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_table_from_postgres(TABLE, SCHEMA)
    X_np = create_sequences(df, FEATURES, TARGET, LOOKBACK)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AttCNNLSTM_v3(
        input_dim=X_np.shape[2],
        hidden_dim=256,
        num_layers=2,
        dropout=0.2
    ).to(device)

    ckpt = "results/lstm_advanced/v3/att_cnn_lstm_v3.pt"
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    embs = []
    with torch.no_grad():
        for i in range(0, len(X_np), BATCH_SIZE):
            xb = torch.tensor(X_np[i:i+BATCH_SIZE]).to(device)
            _, emb = model(xb)
            embs.append(emb.cpu().numpy())

    final_emb = np.concatenate(embs, axis=0)
    out_path = OUT_DIR / "att_cnn_embeddings_train_v3.npy"
    np.save(out_path, final_emb)

    print(f"✅ Saved ATT embeddings: {final_emb.shape} → {out_path}")


if __name__ == "__main__":
    main()
