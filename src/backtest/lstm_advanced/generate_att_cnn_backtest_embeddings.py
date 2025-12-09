import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sqlalchemy import create_engine
import pandas as pd
import yaml
import os
os.environ["PG_CONN_STR"] = "postgresql+psycopg2://postgres:123456789@localhost:5432/postgres"

from src.pipeline_baseline.data_loader import load_table_from_postgres
from src.lstm_advanced.train_att_cnn_lstm import AttCNNLSTM  # bạn tạo file nhỏ riêng
# hoặc copy class AttCNNLSTM từ train file nếu muốn


def create_sequences(df, lookback=60):
    data = df.drop(columns=["datetime"]).values
    X = []
    for i in range(len(data) - lookback):
        X.append(data[i:i+lookback, :-1])  # bỏ target
    return np.array(X)


def main():
    print("=== Generating ATT-CNN-LSTM backtest embeddings ===")

    # Load config
    with open("config/config_baseline.yaml", "r") as f:
        cfg = yaml.safe_load(f)["lstm_advanced1"]

    schema = "it_final"
    backtest_table = "processed_ohlcv_backtest"

    # Load model hyperparams
    lookback = cfg["lookback"]
    hidden_dim = cfg.get("hidden_size", 768)
    num_layers = cfg["num_layers"]
    dropout = cfg["dropout"]

    # Load backtest OHLCV
    df = load_table_from_postgres(backtest_table, schema)
    print(f"Backtest OHLCV size = {len(df)} rows")

    # Create sequences
    X = create_sequences(df, lookback)
    print("Sequence tensor:", X.shape)

    # Convert to tensor
    X = torch.tensor(X, dtype=torch.float32)

    # Load model
    model = AttCNNLSTM(
        input_dim=X.shape[2],
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout
    )
    model.load_state_dict(torch.load("results/lstm_advanced/att_cnn_lstm_baseline.pt"))
    model.eval()

    # Generate embeddings
    embs = []
    bs = 32

    with torch.no_grad():
        for i in range(0, len(X), bs):
            batch = X[i:i+bs]
            _, emb = model(batch)
            embs.append(emb.numpy())

    embs = np.concatenate(embs, axis=0)
    print("Final embeddings shape:", embs.shape)

    out_path = "results/backtest/lstm_advanced/att_cnn_backtest_embeddings.npy"
    os.makedirs("results/backtest/lstm_advanced", exist_ok=True)
    np.save(out_path, embs)
    print("Saved:", out_path)


if __name__ == "__main__":
    main()
