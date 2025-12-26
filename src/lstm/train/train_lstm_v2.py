import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import yaml, json
from pathlib import Path

from src.pipeline_baseline.data_loader import load_table_from_postgres, make_time_windows
from src.pipeline_baseline.utils.logger import get_logger


class LSTMRegressor(nn.Module):
    def __init__(self, input_dim, hidden_size, num_layers, dropout, bidirectional=False):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden_size,
            num_layers,
            dropout=dropout,
            batch_first=True,
            bidirectional=bidirectional,
        )
        self.fc = nn.Linear(hidden_size * (2 if bidirectional else 1), 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.fc(last)


def main():
    # === Load config ===
    config_path = Path("config/config_lstm_v2.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg["data"]
    lstm_cfg = cfg["lstm"]

    root_dir = Path(cfg["project"]["root_dir"])
    log_dir = Path(cfg["logging"]["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = get_logger(log_dir / "train_lstm_v2.log")
    logger.info("=== Training LSTM Baseline v2 (predict CLOSE t+1) ===")

    # === Load data (đã được scale, xử lý NaN trong data_loader) ===
    train_df = load_table_from_postgres(cfg["postgres"]["tables"]["ohlcv_train"], cfg["postgres"]["schema"])
    valid_df = load_table_from_postgres(cfg["postgres"]["tables"]["ohlcv_valid"], cfg["postgres"]["schema"])

    features = data_cfg["features"]
    target = data_cfg["target"]
    lookback = data_cfg["lookback"]

    X_train, y_train = make_time_windows(train_df, features, target, lookback)
    X_valid, y_valid = make_time_windows(valid_df, features, target, lookback)

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_valid = torch.tensor(X_valid, dtype=torch.float32)
    y_valid = torch.tensor(y_valid, dtype=torch.float32).unsqueeze(1)

    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=lstm_cfg["batch_size"],
        shuffle=True,
    )
    valid_loader = DataLoader(
        TensorDataset(X_valid, y_valid),
        batch_size=lstm_cfg["batch_size"],
        shuffle=False,
    )

    # === Model & optimizer ===
    device = torch.device(lstm_cfg["device"] if torch.cuda.is_available() else "cpu")
    model = LSTMRegressor(
        input_dim=len(features),
        hidden_size=lstm_cfg["hidden_size"],
        num_layers=lstm_cfg["num_layers"],
        dropout=lstm_cfg["dropout"],
        bidirectional=lstm_cfg["bidirectional"],
    ).to(device)

    criterion = nn.MSELoss()
    wd = float(lstm_cfg.get("weight_decay", 0.0))
    if wd == 0.0:
        wd = 1e-5

    optimizer = torch.optim.Adam(model.parameters(), lr=lstm_cfg["lr"], weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5)

    # === Output dir v2 ===
    out_dir = root_dir / "results/lstm/v2"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "lstm_baseline.pt"
    metrics_path = out_dir / "lstm_baseline_valid_metrics.json"

    # === Train loop với early stopping ===
    best_val = float("inf")
    patience = 0
    patience_limit = 15  # có thể chỉnh nếu muốn

    num_epochs = lstm_cfg["num_epochs"]  # hiện là 5 trong config
    for epoch in range(1, num_epochs + 1):
        model.train()
        train_loss = 0.0

        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(Xb)
            loss = criterion(pred, yb)
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
                pred = model(Xb)
                val_loss += criterion(pred, yb).item()
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

    logger.info(f"[DONE] Best Val Loss: {best_val:.6f}, model saved to {ckpt_path}")

    # === Tính metrics MSE/MAE trên valid (dùng best model) ===
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    with torch.no_grad():
        X_valid = X_valid.to(device)
        preds = model(X_valid).cpu().numpy().squeeze()
        y_true = y_valid.numpy().squeeze()

    mse = float(((preds - y_true) ** 2).mean())
    mae = float((abs(preds - y_true)).mean())
    metrics = {"MSE": mse, "MAE": mae}

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"[DONE] Valid metrics: {metrics} -> {metrics_path}")


if __name__ == "__main__":
    main()
