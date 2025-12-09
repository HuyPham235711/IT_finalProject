import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import yaml, os
from pathlib import Path
from src.pipeline_baseline.data_loader import load_table_from_postgres
from src.lstm_advanced.train_att_cnn_lstm import AttCNNLSTM  # nếu bạn đặt riêng
from src.lstm_advanced.train_transformer import TimeSeriesTransformer  # nếu có
# ⚠️ chỉnh import theo đúng cấu trúc project của bạn

# ============================================================
# Helper: tạo sequence
# ============================================================
def create_sequences(df, lookback=60):
    data = df.drop(columns=["datetime"]).values
    X, y = [], []
    for i in range(len(data) - lookback):
        X.append(data[i:i+lookback, :-1])
        y.append(data[i+lookback, -1])
    return np.array(X), np.array(y)


# ============================================================
# Main inference
# ============================================================
def main():
    with open("config/config_baseline.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg = cfg["lstm_advanced"]
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")

    schema = cfg["data"]["schema"]
    test_table = cfg["data"]["test_table"]
    lookback = cfg["lookback"]

    out_dir = Path("results/lstm_advanced/")
    out_dir.mkdir(parents=True, exist_ok=True)

    # === Load test data ===
    test_df = load_table_from_postgres(test_table, schema)
    X_test, y_test = create_sequences(test_df, lookback)
    X_test = torch.tensor(X_test, dtype=torch.float32).to(device)

    print(f"[INFO] Loaded {len(X_test)} test sequences")

    # ============================================================
    # 1️⃣ Inference ATT-CNN-LSTM
    # ============================================================
    att_model = AttCNNLSTM(
        input_dim=X_test.shape[2],
        hidden_dim=768,      # theo config train
        num_layers=2,
        dropout=0.2
    ).to(device)
    att_ckpt = "results/lstm_advanced/att_cnn_lstm_baseline.pt"
    att_model.load_state_dict(torch.load(att_ckpt, map_location=device))
    att_model.eval()

    with torch.no_grad():
        _, att_emb = att_model(X_test)
        att_emb = att_emb.cpu().numpy()
        np.save(out_dir / "att_cnn_lstm_embeddings_test.npy", att_emb)
    print(f"✅ Saved ATT-CNN-LSTM test embeddings → {out_dir/'att_cnn_lstm_embeddings_test.npy'} | shape={att_emb.shape}")

    # ============================================================
    # 2️⃣ Inference Transformer Encoder
    # ============================================================
    trans_model = TimeSeriesTransformer(
        input_dim=X_test.shape[2],
        embed_dim=768,
        num_heads=8,
        num_layers=2,
        dropout=0.2
    ).to(device)
    trans_ckpt = "results/lstm_advanced/transformer_baseline.pt"
    trans_model.load_state_dict(torch.load(trans_ckpt, map_location=device))
    trans_model.eval()

    with torch.no_grad():
        _, trans_emb = trans_model(X_test)
        trans_emb = trans_emb.cpu().numpy()
        np.save(out_dir / "transformer_embeddings_test.npy", trans_emb)
    print(f"✅ Saved Transformer test embeddings → {out_dir/'transformer_embeddings_test.npy'} | shape={trans_emb.shape}")

    print("🎯 Inference test embeddings completed for both models.")


if __name__ == "__main__":
    main()
