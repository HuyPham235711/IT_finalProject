import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import yaml, json
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error

from src.lstm.pipeline_lstm_baseline.data_loader import load_table_from_postgres
from src.lstm.pipeline_lstm_baseline.utils.logger import get_logger


# ============================================================
# 1. Attention Layer
# ============================================================
class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, lstm_out):
        # [B, T, H]
        weights = torch.softmax(self.attn(lstm_out), dim=1)   # [B, T, 1]
        context = torch.sum(weights * lstm_out, dim=1)        # [B, H]
        return context, weights


# ============================================================
# 2. ATT-CNN-LSTM v3 (stable version)
# ============================================================
class AttCNNLSTM_v3(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout):
        super().__init__()

        # CNN → feature extraction
        self.conv = nn.Conv1d(in_channels=input_dim, out_channels=64, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

        # LayerNorm sau CNN để tránh explode scale
        self.cnn_norm = nn.LayerNorm(64)

        # LSTM
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )

        self.attention = Attention(hidden_dim)

        # Chuẩn hóa embedding trước FC
        self.post_ln = nn.LayerNorm(hidden_dim * 3)

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 3, 1)

    def forward(self, x):
        # x: [B, T, F]
        x = x.permute(0, 2, 1)                 # -> [B, F, T]
        x = self.relu(self.conv(x))            # -> [B, 64, T]
        x = x.permute(0, 2, 1)                 # -> [B, T, 64]

        # LayerNorm cải thiện ổn định
        x = self.cnn_norm(x)

        lstm_out, _ = self.lstm(x)            # -> [B, T, H]

        context, _ = self.attention(lstm_out)  # [B, H]
        mean_pool = lstm_out.mean(dim=1)       # [B, H]
        max_pool, _ = lstm_out.max(dim=1)      # [B, H]

        emb = torch.cat([context, mean_pool, max_pool], dim=-1)   # [B, 3H]
        emb = self.post_ln(emb)
        emb = self.dropout(emb)

        out = self.fc(emb)
        return out.squeeze(1), emb


# ============================================================
# 3. Helper: Create sequences
# ============================================================
def create_sequences(df, feature_cols, target_col, lookback):
    df_seq = df[feature_cols + [target_col]].copy()
    data = df_seq.values

    X, y = [], []
    for i in range(len(data) - lookback):
        X.append(data[i:i+lookback, :-1])
        y.append(data[i+lookback, -1])
    return np.array(X), np.array(y)


# ============================================================
# 4. Training script
# ============================================================
def main():

    config_path = Path("E:/TDTu/TAI_LIEU/KY1-NAM5/DU_AN_CNTT/config/config_lstm_v2.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg_all = yaml.safe_load(f)

    data_cfg = cfg_all["data"]
    adv_cfg = cfg_all["lstm_advanced1"]

    features = data_cfg["features"]
    target = data_cfg["target"]
    lookback = adv_cfg["lookback"]

    device = torch.device(adv_cfg["device"] if torch.cuda.is_available() else "cpu")

    # Logger
    log_dir = Path(cfg_all["logging"]["log_dir"])
    logger = get_logger(log_dir / "train_att_cnn_lstm_v3.log")
    logger.info("=== Training ATT-CNN-LSTM v3 (stable) ===")

    # Load data
    schema = adv_cfg["data"]["schema"]
    train_df = load_table_from_postgres(adv_cfg["data"]["train_table"], schema)
    valid_df = load_table_from_postgres(adv_cfg["data"]["valid_table"], schema)

    logger.info(f"Train shape={train_df.shape}, Valid shape={valid_df.shape}")

    X_train, y_train = create_sequences(train_df, features, target, lookback)
    X_valid, y_valid = create_sequences(valid_df, features, target, lookback)

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_valid = torch.tensor(X_valid, dtype=torch.float32)
    y_valid = torch.tensor(y_valid, dtype=torch.float32)

    train_loader = DataLoader(
        TensorDataset(X_train, y_train), batch_size=adv_cfg["batch_size"], shuffle=True
    )
    valid_loader = DataLoader(
        TensorDataset(X_valid, y_valid), batch_size=adv_cfg["batch_size"], shuffle=False
    )

    # Model v3
    model = AttCNNLSTM_v3(
        input_dim=X_train.shape[2],
        hidden_dim=adv_cfg["hidden_size"],
        num_layers=adv_cfg["num_layers"],
        dropout=adv_cfg["dropout"],
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=adv_cfg["lr"],
        weight_decay=1e-5,              # rất quan trọng để giữ embedding ổn định
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5
    )

    # Output path
    root_dir = Path(cfg_all["project"]["root_dir"])
    out_dir = root_dir / "results/lstm_advanced/v2"
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = out_dir / "att_cnn_lstm_v3.pt"
    metrics_path = out_dir / "att_cnn_metrics_v3.json"
    emb_path = out_dir / "att_cnn_embeddings_v3.npy"

    best_val = float("inf")

    # ============================================================
    # Training loop
    # ============================================================
    for epoch in range(1, adv_cfg["epochs"] + 1):
        model.train()
        train_loss = 0

        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)

            optimizer.zero_grad()
            pred, _ = model(Xb)
            loss = criterion(pred, yb)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item() * len(Xb)

        train_loss /= len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for Xb, yb in valid_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                pred, _ = model(Xb)
                val_loss += criterion(pred, yb).item() * len(Xb)

        val_loss /= len(valid_loader.dataset)
        scheduler.step(val_loss)

        logger.info(f"Epoch {epoch}/{adv_cfg['epochs']}  Train={train_loss:.6f}  Valid={val_loss:.6f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), ckpt_path)

    # ============================================================
    # Evaluate best model
    # ============================================================
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    preds_all, emb_all = [], []

    with torch.no_grad():
        for Xb, _ in DataLoader(TensorDataset(X_valid, y_valid), batch_size=adv_cfg["batch_size"]):
            Xb = Xb.to(device)
            preds, emb = model(Xb)
            preds_all.extend(preds.cpu().numpy().flatten())
            emb_all.append(emb.cpu().numpy())

    emb_all = np.concatenate(emb_all)

    y_np = y_valid.numpy()
    preds_np = np.array(preds_all)

    mse = mean_squared_error(y_np, preds_np)
    mae = mean_absolute_error(y_np, preds_np)

    metrics = {"MSE": float(mse), "MAE": float(mae)}

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    np.save(emb_path, emb_all)

    logger.info(f"Saved ATT-CNN-LSTM v3 checkpoint + metrics + embeddings.")
    logger.info(f"Metrics = {metrics}")


if __name__ == "__main__":
    main()
