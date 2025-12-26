import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import yaml, os, json
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error

from src.pipeline_baseline.data_loader import load_table_from_postgres
from src.pipeline_baseline.utils.logger import get_logger


# ==== 1. Positional Encoding ====
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x):
        # x: [batch, seq_len, d_model]
        return x + self.pe[:, :x.size(1)]


# ==== 2. Transformer Model ====
class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads, num_layers, dropout=0.1):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, embed_dim)
        self.pos_encoder = PositionalEncoding(embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        # LayerNorm trước encoder (embed_dim) và sau pooling (2*embed_dim)
        self.pre_ln = nn.LayerNorm(embed_dim)
        self.post_ln = nn.LayerNorm(embed_dim * 2)

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(embed_dim * 2, 1)  # nhận [mean, max] concat

    def forward(self, x):
        # x: [batch, seq_len, features]
        x = self.input_projection(x)     # -> [B, T, embed_dim]
        x = self.pos_encoder(x)
        x = self.pre_ln(x)
        x = self.transformer_encoder(x)  # -> [B, T, embed_dim]

        # === Combine Pooling ===
        x_mean = x.mean(dim=1)           # [B, embed_dim]
        x_max, _ = x.max(dim=1)          # [B, embed_dim]
        x = torch.cat([x_mean, x_max], dim=-1)  # [B, 2*embed_dim]

        x = self.post_ln(x)
        x = self.dropout(x)
        out = self.fc(x)                 # [B, 1]
        return out, x                    # (pred, emb 2*embed_dim)


# ==== 3. Utility: tạo sequences với feature_cols + target_col ====
def create_sequences(df, feature_cols, target_col, lookback):
    """
    df: DataFrame đầy đủ
    feature_cols: list tên cột feature (giống config.data.features)
    target_col: tên cột target (vd: "close")
    """
    df_seq = df[feature_cols + [target_col]].copy()
    data = df_seq.values

    X, y = [], []
    for i in range(len(data) - lookback):
        X.append(data[i:i+lookback, :-1])  # all features
        y.append(data[i+lookback, -1])     # target tại t+lookback
    return np.array(X), np.array(y)


# ==== 4. Main ====
def main():
    # === Load config gốc ===
    config_path = Path("E:/TDTu/TAI_LIEU/KY1-NAM5/DU_AN_CNTT/config/config_baseline.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg_all = yaml.safe_load(f)

    data_cfg = cfg_all["data"]              # features, target, lookback (baseline)
    adv_cfg = cfg_all["lstm_advanced2"]     # cấu hình cho transformer
    features = data_cfg["features"]
    target = data_cfg["target"]             # "close"
    lookback = adv_cfg["lookback"]

    device = torch.device(adv_cfg["device"] if torch.cuda.is_available() else "cpu")

    # Logger
    log_dir = Path(cfg_all["logging"]["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = get_logger(log_dir / "train_transformer_v2.log")
    logger.info("=== Training Transformer Encoder v2 (predict CLOSE t+1) ===")

    schema = adv_cfg["data"]["schema"]
    train_table = adv_cfg["data"]["train_table"]
    valid_table = adv_cfg["data"]["valid_table"]

    # === Load data ===
    train_df = load_table_from_postgres(train_table, schema)
    valid_df = load_table_from_postgres(valid_table, schema)
    logger.info(f"Train shape={train_df.shape}, Valid shape={valid_df.shape}")

    # === Prepare sequences (target = close) ===
    X_train, y_train = create_sequences(train_df, features, target, lookback)
    X_valid, y_valid = create_sequences(valid_df, features, target, lookback)
    logger.info(f"Created {len(X_train)} train seq, {len(X_valid)} valid seq")

    # Tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_valid = torch.tensor(X_valid, dtype=torch.float32)
    y_valid = torch.tensor(y_valid, dtype=torch.float32).unsqueeze(1)

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

    # === Model ===
    model = TimeSeriesTransformer(
        input_dim=X_train.shape[2],
        embed_dim=adv_cfg["embedding_dim"],
        num_heads=adv_cfg["num_heads"],
        num_layers=adv_cfg["num_layers"],
        dropout=adv_cfg["dropout"],
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=adv_cfg["lr"])
    criterion = nn.MSELoss()

    # === Output dir v2 ===
    root_dir = Path(cfg_all["project"]["root_dir"])
    out_dir = root_dir / "results/lstm_advanced/v2"
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = out_dir / "transformer_baseline.pt"
    metrics_path = out_dir / "transformer_metrics.json"
    emb_path = out_dir / "transformer_embeddings.npy"

    # === Train loop + best val ===
    best_val_loss = float("inf")
    for epoch in range(1, adv_cfg["epochs"] + 1):
        model.train()
        total_loss = 0.0
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred, _ = model(Xb)
            loss = criterion(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item() * Xb.size(0)

        train_loss = total_loss / len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for Xb, yb in valid_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                pred, _ = model(Xb)
                val_loss += criterion(pred, yb).item() * Xb.size(0)
        val_loss /= len(valid_loader.dataset)

        logger.info(
            f"Epoch {epoch}/{adv_cfg['epochs']} - "
            f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), ckpt_path)

    # === Evaluate trên valid set với checkpoint tốt nhất ===
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    with torch.no_grad():
        X_valid = X_valid.to(device)
        pred_all, emb_all_list = [], []
        bs = adv_cfg["batch_size"]
        for i in range(0, len(X_valid), bs):
            Xb = X_valid[i:i+bs].to(device)
            preds, emb = model(Xb)
            pred_all.extend(preds.cpu().numpy().flatten())
            emb_all_list.append(emb.cpu().numpy())

    emb_all = np.concatenate(emb_all_list, axis=0)
    y_valid_np = y_valid.numpy().flatten()
    y_pred_np = np.array(pred_all)

    mse = mean_squared_error(y_valid_np, y_pred_np)
    mae = mean_absolute_error(y_valid_np, y_pred_np)
    metrics = {"MSE": float(mse), "MAE": float(mae)}

    # Save results
    torch.save(model.state_dict(), ckpt_path)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    np.save(emb_path, emb_all)

    logger.info(f"Saved embeddings to {emb_path}, shape={emb_all.shape}")
    logger.info(f"[DONE] Saved model & metrics to {ckpt_path}, {metrics_path}. Metrics: {metrics}")


if __name__ == "__main__":
    main()
