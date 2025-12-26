import torch
import numpy as np
import json
import yaml
import matplotlib.pyplot as plt
from pathlib import Path

from src.pipeline_baseline.data_loader import load_table_from_postgres
from src.lstm.train_lstm_v2 import LSTMRegressor
from src.lstm_advanced.train_att_cnn_lstm_v3 import AttCNNLSTM_v3
from src.lstm_advanced.train_transformer_v3 import TimeSeriesTransformerV3


# ============================================================
# Helper: create unified sequences
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
# Metrics
# ============================================================
def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def mape(y_true, y_pred, eps=1e-8):
    mask = np.abs(y_true) > eps
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


# ============================================================
# Main
# ============================================================
def main():

    config_path = Path("E:/TDTu/TAI_LIEU/KY1-NAM5/DU_AN_CNTT/config/config_baseline.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg["data"]
    lstm_cfg = cfg["lstm"]
    adv1_cfg = cfg["lstm_advanced1"]
    adv2_cfg = cfg["lstm_advanced2"]

    features = data_cfg["features"]
    target = data_cfg["target"]
    lookback = data_cfg["lookback"]

    device = torch.device(lstm_cfg["device"] if torch.cuda.is_available() else "cpu")

    # Output dir
    root_dir = Path(cfg["project"]["root_dir"])
    out_dir = root_dir / "results/lstm_advanced/v2/test_inference_v3"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ========================================================
    # Load TEST data
    # ========================================================
    df_test = load_table_from_postgres(
        cfg["postgres"]["tables"]["ohlcv_test"],
        cfg["postgres"]["schema"]
    )

    X, y = create_sequences(df_test, features, target, lookback)
    X = torch.tensor(X, dtype=torch.float32).to(device)
    y_np = y.astype(np.float32)

    print(f"[INFO] Unified test samples = {len(X)}, shape = {X.shape}")

    # ========================================================
    # Load models
    # ========================================================
    # LSTM v2
    lstm = LSTMRegressor(
        input_dim=len(features),
        hidden_size=lstm_cfg["hidden_size"],
        num_layers=lstm_cfg["num_layers"],
        dropout=lstm_cfg["dropout"],
        bidirectional=lstm_cfg["bidirectional"]
    ).to(device)

    lstm.load_state_dict(torch.load(
        root_dir / "results/lstm/v2/lstm_baseline.pt",
        map_location=device
    ))
    lstm.eval()

    # ATT-CNN-LSTM v3
    ATT_HIDDEN_DIM = 256  # khớp checkpoint

    att = AttCNNLSTM_v3(
        input_dim=X.shape[2],
        hidden_dim=ATT_HIDDEN_DIM,
        num_layers=adv1_cfg["num_layers"],
        dropout=adv1_cfg["dropout"]
    ).to(device)

    att.load_state_dict(torch.load(
        root_dir / "results/lstm_advanced/v2/att_cnn_lstm_v3.pt",
        map_location=device
    ))
    att.eval()

    # Transformer v3
    TRANS_EMBED_DIM = 256  # KHỚP CHECKPOINT

    trans = TimeSeriesTransformerV3(
        input_dim=X.shape[2],
        embed_dim=TRANS_EMBED_DIM,
        num_heads=adv2_cfg["num_heads"],
        num_layers=adv2_cfg["num_layers"],
        dropout=adv2_cfg["dropout"]
    ).to(device)


    trans.load_state_dict(torch.load(
        root_dir / "results/lstm_advanced/v2/transformer_v3.pt",
        map_location=device
    ))
    trans.eval()

    # ========================================================
    # Predict
    # ========================================================
    with torch.no_grad():
        lstm_pred = lstm(X).cpu().numpy().flatten()
        att_pred, _ = att(X)
        att_pred = att_pred.cpu().numpy().flatten()
        trans_pred, _ = trans(X)
        trans_pred = trans_pred.cpu().numpy().flatten()

    # ========================================================
    # Metrics
    # ========================================================
    metrics = {
        "LSTM_v2": {
            "MSE": float(np.mean((lstm_pred - y_np) ** 2)),
            "MAE": float(np.mean(np.abs(lstm_pred - y_np))),
            "RMSE": rmse(y_np, lstm_pred),
            "MAPE": mape(y_np, lstm_pred),
        },
        "ATT_CNN_LSTM_v3": {
            "MSE": float(np.mean((att_pred - y_np) ** 2)),
            "MAE": float(np.mean(np.abs(att_pred - y_np))),
            "RMSE": rmse(y_np, att_pred),
            "MAPE": mape(y_np, att_pred),
        },
        "Transformer_v3": {
            "MSE": float(np.mean((trans_pred - y_np) ** 2)),
            "MAE": float(np.mean(np.abs(trans_pred - y_np))),
            "RMSE": rmse(y_np, trans_pred),
            "MAPE": mape(y_np, trans_pred),
        }
    }

    with open(out_dir / "metrics_all_v3.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(metrics, indent=2))

    # ========================================================
    # Plot
    # ========================================================
    n = min(len(y_np), len(lstm_pred), len(att_pred), len(trans_pred))

    plt.figure(figsize=(16, 6))
    plt.plot(y_np[:n], label="Actual (scaled)", color="black", linewidth=2)
    plt.plot(lstm_pred[:n], label="LSTM v2", alpha=0.8)
    plt.plot(att_pred[:n], label="ATT-CNN-LSTM v3", alpha=0.8)
    plt.plot(trans_pred[:n], label="Transformer v3", alpha=0.9)

    plt.title("Close(t+1) Prediction – LSTM v2 vs ATT-CNN-LSTM v3 vs Transformer v3 (scaled)")
    plt.xlabel("Time step")
    plt.ylabel("Scaled close")
    plt.legend()
    plt.grid(True)

    plt.savefig(out_dir / "comparison_all_v3.png", dpi=300)
    plt.close()

    print(f"[DONE] Saved metrics + plot to {out_dir}")


if __name__ == "__main__":
    main()
