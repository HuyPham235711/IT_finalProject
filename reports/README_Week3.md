
## 🎯 Mục tiêu Task A

> Viết mô tả và checklist cấu trúc pipeline:
> Data → Model → Output, bao gồm cả FinBERT và LSTM.
>
> Đây là phần “README_Week3.md” tổng hợp toàn tuần 3 —
> mô tả kiến trúc, đường dẫn, cách chạy, và deliverables.

---

### 📁 Đường dẫn file

```
E:/TDTu/TAI_LIEU/KY1-NAM5/DU_AN_CNTT/README_Week3.md
```

---

### 📘 Nội dung đầy đủ – gọn rõ, đúng format của bạn

```markdown
# 🧩 Week 3 – Phát triển Mô hình Độc lập I (Baseline)

Tuần 3 gồm hai task chính:

| Task | Tên | Mục tiêu |
|------|-----|-----------|
| **A** | Pipeline & Kiến trúc | Thiết kế cấu trúc xử lý I/O – Train – Eval – Lưu kết quả |
| **B** | FinBERT Baseline | Đánh giá sentiment model, logits + scores |
| **C** | LSTM Baseline | Dự đoán giá BTCUSD t+1 từ chuỗi OHLCV |

---

## ⚙️ 1️⃣ Cấu trúc thư mục dự án

```

DU_AN_CNTT/
├─ config/
│   └─ config_baseline.yaml
├─ models/
│   ├─ finBERT/
│   └─ lstm/                      ← code LSTM (không lưu checkpoint)
├─ results/
│   ├─ finbert/
│   └─ lstm/
├─ src/
│   ├─ finBERT/
│   │   ├─ finetune_finBERT.py
│   │   └─ evaluate_finbert.py
│   ├─ lstm/
│   │   ├─ train_lstm.py
│   │   ├─ evaluate_lstm.py
│   │   └─ README_LSTM.md
│   ├─ pipeline_baseline/
│   │   ├─ data_loader.py
│   │   └─ utils/logger.py
│   └─ db/postgres_conn.py
├─ logs/
└─ README_Week3.md   ← file này

```

---

## 🧠 2️⃣ Luồng pipeline tổng quát

```

PostgreSQL
│
├─ src/pipeline_baseline/data_loader.py
│      • Load bảng OHLCV / media_test
│      • Xử lý NaN, convert float32
│
├─ src/lstm/train_lstm.py
│      • Huấn luyện LSTM regression
│      • Lưu model + log loss
│
├─ src/lstm/evaluate_lstm.py
│      • Tính MSE/MAE + lưu embeddings
│
├─ src/finBERT/evaluate_finbert.py
│      • Inference + tính Accuracy/F1
│      • Xuất logits/scores
│
└─ results/
├─ finbert/
└─ lstm/

```

---

## 🧩 3️⃣ Cấu hình cơ bản

**File:**  
`E:/TDTu/TAI_LIEU/KY1-NAM5/DU_AN_CNTT/config/config_baseline.yaml`

- Data source: PostgreSQL (`it_final`)
- Feature set LSTM: `[open, high, low, close, volume, sma14, rsi14]`
- Target: `close`
- Lookback: 60
- Device: `cuda:0`
- Epoch: 5

---

## 🧾 4️⃣ Deliverables tuần 3

| Nhóm | File / Output | Đường dẫn |
|------|----------------|------------|
| **FinBERT** | finbert_baseline_metrics.json | `results/finbert/` |
|  | finbert_baseline_predictions.csv | `results/finbert/` |
| **LSTM** | lstm_baseline.pt | `results/lstm/` |
|  | lstm_baseline_metrics.json | `results/lstm/` |
|  | ohlcv_embeddings.npy | `results/lstm/` |
| **Tài liệu** | README_LSTM.md | `src/lstm/` |
|  | README_Week3.md | project root |

---

## 📊 5️⃣ Kết quả tuần

| Mô hình | Metric | Giá trị |
|---------|---------|---------|
| **FinBERT** | F1 (macro) | 0.347 |
| **LSTM** | MSE / MAE | 0.691 / 0.826 |

---

## ✅ 6️⃣ Definition of Done (DoD)

| Mục | Tiêu chí | Trạng thái |
|------|-----------|-------------|
| Pipeline hoạt động đầy đủ | Data → Model → Output | ✅ |
| FinBERT Inference & Metrics | Accuracy / F1 log chuẩn | ✅ |
| LSTM Train & Eval | MSE/MAE được ghi log | ✅ |
| Kết quả lưu vào results/ | Có đủ 3 file (pt/json/npy) | ✅ |
| Tài liệu pipeline hoàn chỉnh | README_LSTM.md + README_Week3.md | ✅ |

---

## 🗓 Ngày hoàn thành
**16 / 10 / 2025**  
→ Toàn bộ Task A, B, C tuần 3 đã hoàn thành.
```

---

## ✅ Tóm tắt nhanh

* File: `E:/TDTu/TAI_LIEU/KY1-NAM5/DU_AN_CNTT/README_Week3.md`
* Mục đích: tổng hợp **pipeline + kết quả + deliverables** cho toàn tuần 3.
* Tài liệu này dùng để **nộp, commit, hoặc review sprint**.