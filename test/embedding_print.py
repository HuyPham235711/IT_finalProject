import numpy as np

fin_train = np.load("results/finbert/train_inference/finbert_daily_embeddings_train.npy")
fin_bt    = np.load("results/backtest/finbert/finbert_daily_embeddings_backtest.npy")

att_train = np.load("results/lstm_advanced/train_inference/att_cnn_lstm_embeddings_train.npy")
att_bt    = np.load("results/backtest/lstm_advanced/att_cnn_backtest_embeddings.npy")

trf_train = np.load("results/lstm_advanced/train_inference/transformer_embeddings_train.npy")
trf_bt    = np.load("results/backtest/lstm_advanced/transformer_backtest_embeddings.npy")

for name, a, b in [
    ("FinBERT", fin_train, fin_bt),
    ("ATT-CNN", att_train, att_bt),
    ("Transformer", trf_train, trf_bt),
]:
    print(f"=== {name} ===")
    print("train mean/std:", a.mean(), a.std())
    print("bt    mean/std:", b.mean(), b.std())
