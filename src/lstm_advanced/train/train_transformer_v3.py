import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import yaml, json
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error

from src.lstm.pipeline_lstm_baseline.data_loader import load_table_from_postgres
from src.lstm.pipeline_lstm_baseline.utils.logger import get_logger


# ============================================================
# 1. Positional Encoding
# ============================================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe.unsqueeze(0))


    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


# ============================================================
# 2. Time-Series Transformer v3
# ============================================================
class TimeSeriesTransformerV3(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads, num_layers, dropout):
        super().__init__()

        self.input_projection = nn.Linear(input_dim, embed_dim)
        self.pos_encoder = PositionalEncoding(embed_dim)
        self.pre_ln = nn.LayerNorm(embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True,
            dim_feedforward=embed_dim * 4  # transformer best practice
        )

        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Pooling + LayerNorm
        self.post_ln = nn.LayerNorm(embed_dim * 2)
        self.dropout = nn.Dropout(dropout)

        # Final regression head
        self.fc = nn.Linear(embed_dim * 2, 1)


    def forward(self, x):
        # x: [batch, seq, features]
        x = self.input_projection(x)
        x = self.pos_encoder(x)
        x = self.pre_ln(x)

        x = self.encoder(x)

        # mean & max pooling
        mean_pool = x.mean(dim=1)
        max_pool, _ = x.max(dim=1)

        emb = torch.cat([mean_pool, max_pool], dim=-1)
        emb = self.post_ln(emb)
        emb = self.dropout(emb)

        out = self.fc(emb)

        return out, emb


# ============================================================
# 3. Create sequences (features + target)
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
# 4. Main training
# ============================================================
def main():

    config_path = Path("E:/TDTu/TAI_LIEU/KY1-NAM5/DU_AN_CNTT/config/config_lstm_v2.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg_all = yaml.safe_load(f)

    data_cfg = cfg_all["data"]
    adv_cfg = cfg_all["lstm_advanced2"]

    features = data_cfg["features"]
    target = data_cfg["target"]
    lookback = adv_cfg["lookback"]

    device = torch.device(adv_cfg["device"] if torch.cuda.is_available() else "cpu")

    # Logger
    log_dir = Path(cfg_all["logging"]["log_dir"])
    logger = get_logger(log_dir / "train_transformer_v3.log")
    logger.info("=== Training Transformer v3 (predict CLOSE t+1) ===")

    # Load data
    schema = adv_cfg["data"]["schema"]
    train_table = adv_cfg["data"]["train_table"]
    valid_table = adv_cfg["data"]["valid_table"]

    train_df = load_table_from_postgres(train_table, schema)
    valid_df = load_table_from_postgres(valid_table, schema)

    logger.info(f"Train DF shape = {train_df.shape}, Valid DF shape = {valid_df.shape}")

    # Prepare sequences
    X_train, y_train = create_sequences(train_df, features, target, lookback)
    X_valid, y_valid = create_sequences(valid_df, features, target, lookback)

    logger.info(f"Train sequences: {len(X_train)}, Valid sequences: {len(X_valid)}")

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_valid = torch.tensor(X_valid, dtype=torch.float32)
    y_valid = torch.tensor(y_valid, dtype=torch.float32).unsqueeze(1)

    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=adv_cfg["batch_size"],
        shuffle=True
    )
    valid_loader = DataLoader(
        TensorDataset(X_valid, y_valid),
        batch_size=adv_cfg["batch_size"],
        shuffle=False
    )

    # Model v3
    model = TimeSeriesTransformerV3(
        input_dim=X_train.shape[2],
        embed_dim=adv_cfg["embedding_dim"],     # now = 256
        num_heads=adv_cfg["num_heads"],         # = 4
        num_layers=adv_cfg["num_layers"],
        dropout=adv_cfg["dropout"]
    ).to(device)

    optimizer = optim.Adam(
        model.parameters(),
        lr=adv_cfg["lr"],
        weight_decay=1e-5                      # bắt buộc cho stability
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        patience=5,
        factor=0.5
    )

    criterion = nn.MSELoss()

    # Output paths
    root_dir = Path(cfg_all["project"]["root_dir"])
    out_dir = root_dir / "results/lstm_advanced/v2"
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = out_dir / "transformer_v3.pt"
    metrics_path = out_dir / "transformer_v3_metrics.json"
    emb_path = out_dir / "transformer_v3_embeddings.npy"

    # Train loop
    best_val = float("inf")

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

    # Load best checkpoint
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    # Extract valid predictions & embeddings
    preds_all, emb_all = [], []

    with torch.no_grad():
        for Xb, _ in DataLoader(TensorDataset(X_valid, y_valid), batch_size=adv_cfg["batch_size"]):
            Xb = Xb.to(device)
            preds, emb = model(Xb)
            preds_all.extend(preds.cpu().numpy().flatten())
            emb_all.append(emb.cpu().numpy())

    emb_all = np.concatenate(emb_all)

    mse = mean_squared_error(y_valid.numpy(), preds_all)
    mae = mean_absolute_error(y_valid.numpy(), preds_all)

    metrics = {"MSE": float(mse), "MAE": float(mae)}

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    np.save(emb_path, emb_all)

    logger.info(f"Saved Transformer v3 checkpoint + metrics + embeddings.")
    logger.info(f"Metrics = {metrics}")


if __name__ == "__main__":
    main()
