import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd
import yaml, os, json
from datetime import datetime
from src.pipeline_baseline.data_loader import load_table_from_postgres
from src.pipeline_baseline.utils.logger import get_logger
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ============================================================
# 1. Attention Layer
# ============================================================
class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, lstm_output):
        # lstm_output: [batch, seq_len, hidden_dim]
        weights = torch.softmax(self.attn(lstm_output), dim=1)  # [batch, seq_len, 1]
        context = torch.sum(weights * lstm_output, dim=1)       # [batch, hidden_dim]
        return context, weights

# ============================================================
# 2. CNN-LSTM-Attention Model
# ============================================================
class AttCNNLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout):
        super().__init__()
        self.conv = nn.Conv1d(in_channels=input_dim, out_channels=64, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(
            input_size=64, hidden_size=hidden_dim, num_layers=num_layers,
            batch_first=True, dropout=dropout
        )
        self.attention = Attention(hidden_dim)
        self.dropout = nn.Dropout(dropout)          
        self.fc = nn.Linear(hidden_dim * 3, 1)
        

    def forward(self, x):
        # x: [batch, seq_len, features]
        x = x.permute(0, 2, 1)           # [batch, features, seq_len]
        x = self.relu(self.conv(x))      # [batch, 64, seq_len]
        x = x.permute(0, 2, 1)           # [batch, seq_len, 64]
        lstm_out, _ = self.lstm(x)
        context, _ = self.attention(lstm_out)  # [batch, hidden_size]

        # --- Combine Pooling (mean + max) giống Transformer ---
        mean_pool = lstm_out.mean(dim=1)
        max_pool, _ = lstm_out.max(dim=1)
        emb = torch.cat([context, mean_pool, max_pool], dim=-1)  # [batch, hidden_size*3]

        emb = self.dropout(emb)
        out = self.fc(emb)
        return out.squeeze(1), emb  # return both prediction and embedding


# ============================================================
# 3. Helper functions
# ============================================================
def create_sequences(df, lookback=60):
    data = df.drop(columns=["datetime"]).values
    X, y = [], []
    for i in range(len(data) - lookback):
        X.append(data[i:i+lookback, :-1])  # all features except target
        y.append(data[i+lookback, -1])     # close(t+1)
    return np.array(X), np.array(y)

# ============================================================
# 4. Main training
# ============================================================
def main():
    # === Load config ===
    with open("config/config_baseline.yaml", "r") as f:
        cfg = yaml.safe_load(f)["lstm_advanced"]

    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")

    log_path = "logs/train_att_cnn_lstm.log"
    logger = get_logger(log_path)
    logger.info("=== Training ATT-CNN-LSTM (Week4 Task1) ===")

    schema = cfg["data"]["schema"]
    train_table = cfg["data"]["train_table"]
    valid_table = cfg["data"]["valid_table"]
    lookback = cfg["lookback"]

    # === Load data ===
    train_df = load_table_from_postgres(train_table, schema)
    valid_df = load_table_from_postgres(valid_table, schema)
    logger.info(f"Train shape={train_df.shape}, Valid shape={valid_df.shape}")

    # === Create sequences ===
    X_train, y_train = create_sequences(train_df, lookback)
    X_valid, y_valid = create_sequences(valid_df, lookback)
    logger.info(f"Created {len(X_train)} train sequences, {len(X_valid)} valid sequences")

    # === Convert to tensors ===
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_valid = torch.tensor(X_valid, dtype=torch.float32)
    y_valid = torch.tensor(y_valid, dtype=torch.float32)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=cfg["batch_size"], shuffle=True)
    valid_loader = DataLoader(TensorDataset(X_valid, y_valid), batch_size=cfg["batch_size"], shuffle=False)

    # === Model setup ===
    model = AttCNNLSTM(input_dim=X_train.shape[2],
                       hidden_dim=cfg["hidden_size"],
                       num_layers=cfg["num_layers"],
                       dropout=cfg["dropout"]).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg["lr"])

    # === Training loop ===
    best_val_loss = float("inf")
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        total_loss = 0
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            preds, emb = model(Xb)         
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * Xb.size(0)
        train_loss = total_loss / len(train_loader.dataset)

        # === Validation ===
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for Xb, yb in valid_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                val_preds, val_emb = model(Xb)
                val_loss += criterion(val_preds, yb).item() * Xb.size(0)
        val_loss /= len(valid_loader.dataset)

        logger.info(f"Epoch {epoch}/{cfg['epochs']} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "results/lstm_advanced/att_cnn_lstm_baseline.pt")

    # === Evaluate on valid set ===
    model.load_state_dict(torch.load("results/lstm_advanced/att_cnn_lstm_baseline.pt"))
    model.eval()
    with torch.no_grad():
        X_valid = X_valid.to(device)
        preds, emb = model(X_valid)                   # ✅ unpack tuple
        preds = preds.cpu().numpy().flatten()          # ✅ chỉ lấy giá trị dự báo
        y_valid_np = y_valid.numpy().flatten()
    mse = mean_squared_error(y_valid_np, preds)
    mae = mean_absolute_error(y_valid_np, preds)

    metrics = {"MSE": float(mse), "MAE": float(mae)}
    with open("results/lstm_advanced/att_cnn_lstm_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"[DONE] Saved metrics: {metrics}")

    # === Save embeddings ===
    model.eval()
    with torch.no_grad():
        X_valid = X_valid.to(device)
        _, emb = model(X_valid)  # ✅ model đã trả về (out, emb)
        emb = emb.cpu().numpy()
        np.save("results/lstm_advanced/att_cnn_lstm_embeddings.npy", emb)

    logger.info(f"Saved embeddings to results/lstm_advanced/att_cnn_lstm_embeddings.npy, shape={emb.shape}")


if __name__ == "__main__":
    main()
