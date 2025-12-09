import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd
import yaml, os, json
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
            d_model=embed_dim, nhead=num_heads, dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)


        # ⚠️ Split LayerNorm: trước encoder (embed_dim) và sau pooling (2*embed_dim)
        self.pre_ln = nn.LayerNorm(embed_dim)
        self.post_ln = nn.LayerNorm(embed_dim * 2)

        self.dropout = nn.Dropout(dropout)
        # ⚠️ FC nhận vào 2*embed_dim vì concat [mean, max]
        self.fc = nn.Linear(embed_dim * 2, 1)

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


# ==== 3. Utility ====
def create_sequences(data, lookback):
    X, y = [], []
    for i in range(len(data) - lookback):
        X.append(data[i : i + lookback, :-1])  # all features except target at last col
        y.append(data[i + lookback, -1])       # target = last col (e.g., close)
    return np.array(X), np.array(y)


# ==== 4. Main ====
def main():
    logger = get_logger("logs/train_transformer.log")
    logger.info("=== Training Transformer Encoder (Week4 Task1) ===")

    # Load config
    with open("config/config_baseline.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)["lstm_advanced"]

    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    schema = "it_final"
    train_table, valid_table = "processed_ohlcv_train", "processed_ohlcv_valid"

    # Load data
    train_df = load_table_from_postgres(train_table, schema)
    valid_df = load_table_from_postgres(valid_table, schema)
    logger.info(f"Train shape={train_df.shape}, Valid shape={valid_df.shape}")

    # Prepare sequences
    lookback = cfg["lookback"]
    train_data = train_df.drop(columns=["datetime"]).values
    valid_data = valid_df.drop(columns=["datetime"]).values
    X_train, y_train = create_sequences(train_data, lookback)
    X_valid, y_valid = create_sequences(valid_data, lookback)
    logger.info(f"Created {len(X_train)} train seq, {len(X_valid)} valid seq")

    # Tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_valid = torch.tensor(X_valid, dtype=torch.float32)
    y_valid = torch.tensor(y_valid, dtype=torch.float32).unsqueeze(1)

    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=cfg["batch_size"],
        shuffle=True
    )

    # Model
    model = TimeSeriesTransformer(
        input_dim=X_train.shape[2],
        embed_dim=cfg["embedding_dim"],
        num_heads=cfg["num_heads"],      # lấy từ config
        num_layers=cfg["num_layers"],    # lấy từ config
        dropout=cfg["dropout"]
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=cfg["lr"])
    criterion = nn.MSELoss()

    # Train loop
    for epoch in range(1, cfg["epochs"] + 1):
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
            total_loss += loss.item()
        logger.info(f"Epoch {epoch}/{cfg['epochs']} - Train Loss: {total_loss/len(train_loader):.6f}")

    # Evaluate (+ save embeddings)
    model.eval()
    with torch.no_grad():
        y_pred, emb_list = [], []
        bs = cfg["batch_size"]
        for i in range(0, len(X_valid), bs):
            Xb = X_valid[i:i+bs].to(device)
            preds, emb = model(Xb)
            y_pred.extend(preds.cpu().numpy().flatten())
            emb_list.append(emb.cpu().numpy())

        emb_all = np.concatenate(emb_list, axis=0)     # [N_valid, 2*embed_dim]
        y_valid_np = y_valid.numpy().flatten()
        mse = mean_squared_error(y_valid_np, np.array(y_pred))
        mae = mean_absolute_error(y_valid_np, np.array(y_pred))
        metrics = {"MSE": float(mse), "MAE": float(mae)}

    # Save results
    os.makedirs("results/lstm_advanced", exist_ok=True)
    torch.save(model.state_dict(), "results/lstm_advanced/transformer_baseline.pt")
    with open("results/lstm_advanced/transformer_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    np.save("results/lstm_advanced/transformer_embeddings.npy", emb_all)

    logger.info(f"Saved embeddings shape={emb_all.shape}")
    logger.info(f"[DONE] Saved model and embeddings.")
    logger.info(f"Metrics: {metrics}")


if __name__ == "__main__":
    main()
