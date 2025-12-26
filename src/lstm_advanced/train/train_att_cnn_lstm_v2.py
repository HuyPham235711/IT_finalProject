import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import yaml, json
from pathlib import Path

from src.pipeline_baseline.data_loader import load_table_from_postgres
from src.pipeline_baseline.utils.logger import get_logger
from sklearn.metrics import mean_squared_error, mean_absolute_error


# ============================================================
# Attention
# ============================================================
class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, lstm_output):
        # lstm_output: [B, T, H]
        weights = torch.softmax(self.attn(lstm_output), dim=1)
        context = (weights * lstm_output).sum(dim=1)  # [B, H]
        return context, weights


# ============================================================
# AttCNNLSTM
# ============================================================
class AttCNNLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout):
        super().__init__()
        self.conv = nn.Conv1d(in_channels=input_dim, out_channels=64, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )

        self.attention = Attention(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 3, 1)

    def forward(self, x):
        # x: [B, T, F]
        x = x.permute(0, 2, 1)          # [B, F, T]
        x = self.relu(self.conv(x))     # [B, 64, T]
        x = x.permute(0, 2, 1)          # [B, T, 64]

        lstm_out, _ = self.lstm(x)      # [B, T, H]
        context, _ = self.attention(lstm_out)  # [B, H]

        mean_pool = lstm_out.mean(dim=1)       # [B, H]
        max_pool, _ = lstm_out.max(dim=1)      # [B, H]

        emb = torch.cat([context, mean_pool, max_pool], dim=-1)  # [B, 3H]
        emb = self.dropout(emb)
        out = self.fc(emb)                      # [B, 1]
        return out.squeeze(1), emb


# ============================================================
# Helper: sequences
# ============================================================
def create_sequences(df, feature_cols, target_col, lookback):
    df_seq = df[feature_cols + [target_col]].copy()
    data = df_seq.values

    X, y = [], []
    for i in range(len(data) - lookback):
        X.append(data[i:i + lookback, :-1])
        y.append(data[i + lookback, -1])
    return np.array(X), np.array(y)


def main():
    # === Load config ===
    config_path = Path("config/config_baseline.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg_all = yaml.safe_load(f)

    data_cfg = cfg_all["data"]
    adv_cfg = cfg_all["lstm_advanced1"]

    features = data_cfg["features"]
    target = data_cfg["target"]
    lookback = adv_cfg["lookback"]

    root_dir = Path(cfg_all["project"]["root_dir"])
    log_dir = Path(cfg_all["logging"]["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = get_logger(log_dir / "train_att_cnn_lstm_v2.log")
    logger.info("=== Training ATT-CNN-LSTM v2 (predict CLOSE t+1) ===")

    schema = adv_cfg["data"]["schema"]
    train_table = adv_cfg["data"]["train_table"]
    valid_table = adv_cfg["data"]["valid_table"]

    # === Load data ===
    train_df = load_table_from_postgres(train_table, schema)
    valid_df = load_table_from_postgres(valid_table, schema)
    logger.info(f"Train shape={train_df.shape}, Valid shape={valid_df.shape}")

    # === Sequences ===
    X_train, y_train = create_sequences(train_df, features, target, lookback)
    X_valid, y_valid = create_sequences(valid_df, features, target, lookback)
    logger.info(f"Created {len(X_train)} train sequences, {len(X_valid)} valid sequences")

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_valid = torch.tensor(X_valid, dtype=torch.float32)
    y_valid = torch.tensor(y_valid, dtype=torch.float32)

    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=adv_cfg["batch_size"],
        shuffle=True,
    )
    valid_loader = DataLoader(
        TensorDataset(X_valid, y_valid),
        batch_size=adv_cfg["batch_size"],
        shuffle=False,
    )

    # === Model & optimizer ===
    device = torch.device(adv_cfg["device"] if torch.cuda.is_available() else "cpu")
    model = AttCNNLSTM(
        input_dim=X_train.shape[2],
        hidden_dim=adv_cfg["hidden_size"],
        num_layers=adv_cfg["num_layers"],
        dropout=adv_cfg["dropout"],
    ).to(device)

    # weight_decay nhỏ để tránh overfit/đổ về hằng số
    optimizer = optim.Adam(model.parameters(), lr=adv_cfg["lr"], weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5)
    criterion = nn.MSELoss()

    # === Output dir v2 ===
    out_dir = root_dir / "results/lstm_advanced/v2"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "att_cnn_lstm_baseline.pt"
    metrics_path = out_dir / "att_cnn_lstm_metrics.json"
    emb_path = out_dir / "att_cnn_lstm_embeddings.npy"

    # === Train loop với early stopping ===
    best_val = float("inf")
    patience = 0
    patience_limit = 20

    num_epochs = adv_cfg["epochs"]  # hiện là 10 trong config
    for epoch in range(1, num_epochs + 1):
        model.train()
        train_loss = 0.0

        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            preds, _ = model(Xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for Xb, yb in valid_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                preds, _ = model(Xb)
                val_loss += criterion(preds, yb).item()
        val_loss /= len(valid_loader)

        scheduler.step(val_loss)

        logger.info(f"Epoch {epoch}/{num_epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

        if val_loss < best_val:
            best_val = val_loss
            patience = 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            patience += 1

        if patience >= patience_limit:
            logger.info("Early stopping triggered.")
            break

    logger.info(f"[DONE] Best Val Loss: {best_val:.6f}, saved to {ckpt_path}")

    # === Evaluate trên valid + save metrics/embeddings ===
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    with torch.no_grad():
        X_valid_dev = X_valid.to(device)
        preds, emb = model(X_valid_dev)
        preds = preds.cpu().numpy().flatten()
        emb = emb.cpu().numpy()
    y_valid_np = y_valid.numpy().flatten()

    mse = float(mean_squared_error(y_valid_np, preds))
    mae = float(mean_absolute_error(y_valid_np, preds))
    metrics = {"MSE": mse, "MAE": mae}

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    np.save(emb_path, emb)

    logger.info(f"[DONE] Saved metrics: {metrics} -> {metrics_path}")
    logger.info(f"[DONE] Saved embeddings to {emb_path}, shape={emb.shape}")


if __name__ == "__main__":
    main()
