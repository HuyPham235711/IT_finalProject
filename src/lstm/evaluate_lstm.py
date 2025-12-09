import torch
import torch.nn as nn
import numpy as np
import json, yaml
from pathlib import Path
from src.pipeline_baseline.data_loader import load_table_from_postgres, make_time_windows
from src.lstm.train_lstm import LSTMRegressor
from src.pipeline_baseline.utils.logger import get_logger

def main():
    # === Load config ===
    config_path = Path("E:/TDTu/TAI_LIEU/KY1-NAM5/DU_AN_CNTT/config/config_baseline.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    logger = get_logger(Path(cfg["logging"]["log_dir"]) / "eval_lstm.log")
    logger.info("=== Evaluating LSTM Baseline ===")

    # === Load test data ===
    test_df = load_table_from_postgres(cfg["postgres"]["tables"]["ohlcv_test"], cfg["postgres"]["schema"])
    features = cfg["data"]["features"]
    target = cfg["data"]["target"]
    lookback = cfg["data"]["lookback"]

    X_test, y_test = make_time_windows(test_df, features, target, lookback)
    X_test = torch.from_numpy(X_test)
    y_test = torch.from_numpy(y_test).unsqueeze(1)

    # === Load model ===
    model_path = Path("E:/TDTu/TAI_LIEU/KY1-NAM5/DU_AN_CNTT/results/lstm/lstm_baseline.pt")
    model = LSTMRegressor(
        input_dim=len(features),
        hidden_size=cfg["lstm"]["hidden_size"],
        num_layers=cfg["lstm"]["num_layers"],
        dropout=cfg["lstm"]["dropout"],
        bidirectional=cfg["lstm"]["bidirectional"]
    )
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    logger.info(f"Model loaded from {model_path}")

    # === Predict ===
    preds, actuals, embeddings = [], [], []
    with torch.no_grad():
        for i in range(len(X_test)):
            out, (hn, _) = model.lstm(X_test[i:i+1])
            last_hidden = hn[-1].numpy()
            pred = model.fc(torch.from_numpy(last_hidden)).item()
            preds.append(pred)
            actuals.append(y_test[i].item())
            embeddings.append(last_hidden.squeeze())

    preds, actuals = np.array(preds), np.array(actuals)
    mse = float(np.mean((preds - actuals) ** 2))
    mae = float(np.mean(np.abs(preds - actuals)))
    metrics = {"MSE": mse, "MAE": mae}

    # === Save metrics + embeddings ===
    save_dir = Path("E:/TDTu/TAI_LIEU/KY1-NAM5/DU_AN_CNTT/results/lstm")
    metrics_path = save_dir / "lstm_baseline_metrics.json"
    npy_path = save_dir / "ohlcv_embeddings.npy"

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    np.save(npy_path, np.array(embeddings))

    logger.info(f"Metrics: {metrics}")
    logger.info(f"Saved metrics to {metrics_path}")
    logger.info(f"Saved embeddings to {npy_path}")

if __name__ == "__main__":
    main()
