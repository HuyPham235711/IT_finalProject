
# LSTM Baseline – Task C (Week 3)

### 🎯 Mục tiêu
Huấn luyện mô hình **LSTM regression** dự đoán `close(t+1)` cho BTCUSD từ dữ liệu OHLCV.

---

## 1️⃣ Cấu trúc

```

src/lstm/
├─ train_lstm.py
├─ evaluate_lstm.py
└─ README_LSTM.md   ← file này

results/lstm/
├─ lstm_baseline.pt
├─ lstm_baseline_metrics.json
└─ ohlcv_embeddings.npy

````

---

## 2️⃣ Lệnh chạy

**Train**
```bash
cd E:/TDTu/TAI_LIEU/KY1-NAM5/DU_AN_CNTT
python -m src.lstm.train_lstm
````

**Evaluate**

```bash
python -m src.lstm.evaluate_lstm
```

---

## 3️⃣ Đường dẫn quan trọng

| Mục        | File                         | Đường dẫn                                                                      |
| ---------- | ---------------------------- | ------------------------------------------------------------------------------ |
| Model      | `lstm_baseline.pt`           | `E:/TDTu/TAI_LIEU/KY1-NAM5/DU_AN_CNTT/results/lstm/lstm_baseline.pt`           |
| Metrics    | `lstm_baseline_metrics.json` | `E:/TDTu/TAI_LIEU/KY1-NAM5/DU_AN_CNTT/results/lstm/lstm_baseline_metrics.json` |
| Embeddings | `ohlcv_embeddings.npy`       | `E:/TDTu/TAI_LIEU/KY1-NAM5/DU_AN_CNTT/results/lstm/ohlcv_embeddings.npy`       |
| Log train  | `train_lstm.log`             | `E:/TDTu/TAI_LIEU/KY1-NAM5/DU_AN_CNTT/logs/train_lstm.log`                     |
| Log eval   | `eval_lstm.log`              | `E:/TDTu/TAI_LIEU/KY1-NAM5/DU_AN_CNTT/logs/eval_lstm.log`                      |

---

## 4️⃣ Kết quả tuần

|  Metric | Giá trị |
| ------: | ------: |
| **MSE** |  0.6911 |
| **MAE** |  0.8260 |

Baseline chạy ổn định, không NaN, pipeline đầy đủ.

---

## 5️⃣ Ghi chú

* Data đọc từ PostgreSQL schema `it_final`
  (`processed_ohlcv_train`, `processed_ohlcv_valid`, `processed_ohlcv_test`)
* Đã chuẩn hóa MinMax trước khi huấn luyện.
* Chưa tune hyperparameter (baseline only).

---

✅ **Task C – LSTM Baseline hoàn thành.**

```
