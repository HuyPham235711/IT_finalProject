import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import yaml, os, json
from pathlib import Path
from src.pipeline_baseline.data_loader import load_table_from_postgres, make_time_windows
from src.pipeline_baseline.utils.logger import get_logger

# === Mô hình LSTM cơ bản ===
class LSTMRegressor(nn.Module):
    def __init__(self, input_dim, hidden_size, num_layers, dropout, bidirectional=False):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_size, num_layers,
            dropout=dropout, batch_first=True, bidirectional=bidirectional
        )
        self.fc = nn.Linear(hidden_size * (2 if bidirectional else 1), 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.fc(last)


def main():
    # === Load config ===
    config_path = Path("E:/TDTu/TAI_LIEU/KY1-NAM5/DU_AN_CNTT/config/config_baseline.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    logger = get_logger(Path(cfg["logging"]["log_dir"]) / "train_lstm.log")
    logger.info("=== Training LSTM Baseline ===")

    # === Load data ===
    train_df = load_table_from_postgres(cfg["postgres"]["tables"]["ohlcv_train"], cfg["postgres"]["schema"])
    valid_df = load_table_from_postgres(cfg["postgres"]["tables"]["ohlcv_valid"], cfg["postgres"]["schema"])

    features = cfg["data"]["features"]
    target = cfg["data"]["target"]
    lookback = cfg["data"]["lookback"]

    X_train, y_train = make_time_windows(train_df, features, target, lookback)
    X_valid, y_valid = make_time_windows(valid_df, features, target, lookback)

    X_train, y_train = torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_valid, y_valid = torch.tensor(X_valid, dtype=torch.float32), torch.tensor(y_valid, dtype=torch.float32).unsqueeze(1)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=cfg["lstm"]["batch_size"], shuffle=True)
    valid_loader = DataLoader(TensorDataset(X_valid, y_valid), batch_size=cfg["lstm"]["batch_size"], shuffle=False)

    # === Model setup ===
    model = LSTMRegressor(
        input_dim=len(features),
        hidden_size=cfg["lstm"]["hidden_size"],
        num_layers=cfg["lstm"]["num_layers"],
        dropout=cfg["lstm"]["dropout"],
        bidirectional=cfg["lstm"]["bidirectional"]
    ).to(cfg["lstm"]["device"])

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lstm"]["lr"])

    # === Train loop (baseline, ngắn) ===
    for epoch in range(cfg["lstm"]["num_epochs"]):
        model.train()
        total_loss = 0
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(cfg["lstm"]["device"]), yb.to(cfg["lstm"]["device"])
            pred = model(Xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        logger.info(f"Epoch {epoch+1}/{cfg['lstm']['num_epochs']} - Train Loss: {total_loss/len(train_loader):.6f}")

    # === Save model ===
    save_path = Path(cfg["evaluation"]["save_dir"]) / cfg["evaluation"]["outputs"]["model_file"]
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_path)
    logger.info(f"Model saved to {save_path}")


if __name__ == "__main__":
    main()
