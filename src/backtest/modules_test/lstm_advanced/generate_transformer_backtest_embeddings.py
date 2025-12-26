import torch
import numpy as np
from pathlib import Path

from src.lstm.pipeline_lstm_baseline.data_loader import load_table_from_postgres
from src.lstm_advanced.train.train_transformer_v3 import TimeSeriesTransformerV3


SCHEMA = "it_final"
FEATURES = ["open", "high", "low", "close", "volume", "sma14", "rsi14"]
LOOKBACK = 60
BATCH_SIZE = 256

CHECKPOINT_PATH = "results/lstm_advanced/v3/transformer_v3.pt"
OUT_DIR = Path("results/backtest/lstm_advanced/v3")

BACKTEST_PARTS = {
    "part1": "processed_ohlcv_backtest_part1",
    "part2": "processed_ohlcv_backtest_part2",
    "part3": "processed_ohlcv_backtest_part3",
}


def create_sequences(df, lookback):
    data = df[FEATURES].values.astype(np.float32)
    return np.array([data[i:i + lookback] for i in range(len(data) - lookback)])


def main():
    print("=== TRANSFORMER v3 BACKTEST (MULTI-PART) ===")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = TimeSeriesTransformerV3(
        input_dim=len(FEATURES),
        embed_dim=256,
        num_heads=8,
        num_layers=2,
        dropout=0.2
    ).to(device)

    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    model.eval()

    for part, table in BACKTEST_PARTS.items():
        print(f"\n--- {part.upper()} ---")

        df = load_table_from_postgres(table, SCHEMA)
        X_np = create_sequences(df, LOOKBACK)

        all_emb = []
        with torch.no_grad():
            for i in range(0, len(X_np), BATCH_SIZE):
                xb = torch.tensor(X_np[i:i + BATCH_SIZE], device=device)
                _, emb = model(xb)
                all_emb.append(emb.cpu().numpy())

        final_emb = np.concatenate(all_emb, axis=0)
        out_path = OUT_DIR / f"transformer_backtest_embeddings_v3_{part}.npy"
        np.save(out_path, final_emb)

        print(f"[OK] {out_path} | shape={final_emb.shape}")


if __name__ == "__main__":
    main()
