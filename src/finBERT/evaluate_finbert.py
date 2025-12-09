# src/models/finbert/evaluate_finbert_full.py
"""
Chạy toàn bộ inference FinBERT:
- Inference + đánh giá trên media_test (Postgres)
- Tính metrics (accuracy, precision, recall, F1)
- Vẽ confusion matrix & bar chart F1-score
- Trích xuất score 3 lớp (pos/neg/neu) + logits embedding
- Lưu CSV, JSON, TXT, PNG, NPY
"""

from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from torch.nn.functional import softmax
from sklearn.metrics import classification_report, confusion_matrix
from pathlib import Path
import torch, json, numpy as np, pandas as pd, seaborn as sns, matplotlib.pyplot as plt
from src.db.postgres_conn import load_table_to_df, get_postgres_engine
import time

# === 0. Cấu hình chung ===
MODEL_PATH = Path(r"E:\TDTu\TAI_LIEU\KY1-NAM5\DU_AN_CNTT\models\finBERT\finbert_finetuned_sampler_v2")
OUTPUT_DIR = Path(r"E:\TDTu\TAI_LIEU\KY1-NAM5\DU_AN_CNTT\results\finbert\sampler_v2")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

start = time.time()
device = 0 if torch.cuda.is_available() else -1
print("=" * 70)
print("[INFO] FinBERT FULL Inference starting...")
print(f"[INFO] Using device: {'cuda:0' if device == 0 else 'cpu'}")

# === 1. Load model ===
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
# Mapping nhãn chuẩn theo model đã fix
model.config.id2label = {0: 'positive', 1: 'negative', 2: 'neutral'}
model.config.label2id = {'positive': 0, 'negative': 1, 'neutral': 2}
clf = pipeline("text-classification", model=model, tokenizer=tokenizer, device=device)
print(f"[INFO] Model loaded from: {MODEL_PATH}")

# === 2. Load data từ Postgres ===
df = load_table_to_df("media_test", schema="it_final")
print(f"[INFO] Loaded {len(df)} rows from media_test")

texts = df["title"].astype(str).tolist()
true_labels = df["sentiment_label"].astype(str).tolist()

# === 3. Inference ===
print("[INFO] Running inference...")
preds = clf(texts, truncation=True, padding=True)
df["pred_label"] = [p["label"] for p in preds]
df["pred_score"] = [p["score"] for p in preds]

# === 4. Evaluation ===
print("[INFO] Calculating metrics...")
report = classification_report(true_labels, df["pred_label"], output_dict=True, digits=3)
conf_mat = confusion_matrix(true_labels, df["pred_label"])

# --- Tổng hợp các chỉ số ---
acc = report["accuracy"]
macro_f1 = report["macro avg"]["f1-score"]
weighted_f1 = report["weighted avg"]["f1-score"]
macro_precision = report["macro avg"]["precision"]
macro_recall = report["macro avg"]["recall"]

summary_txt = []
summary_txt.append("========== 📊 METRICS SUMMARY ==========\n")
summary_txt.append(f"Accuracy        : {acc:.4f}\n")
summary_txt.append(f"Macro Precision : {macro_precision:.4f}\n")
summary_txt.append(f"Macro Recall    : {macro_recall:.4f}\n")
summary_txt.append(f"Macro F1-score  : {macro_f1:.4f}\n")
summary_txt.append(f"Weighted F1     : {weighted_f1:.4f}\n")
summary_txt.append("========================================\n")

for cls in sorted(report.keys()):
    if cls in ["accuracy", "macro avg", "weighted avg"]:
        continue
    summary_txt.append(
        f"{cls.capitalize():<10} → "
        f"P={report[cls]['precision']:.3f}  "
        f"R={report[cls]['recall']:.3f}  "
        f"F1={report[cls]['f1-score']:.3f}  "
        f"(Support={report[cls]['support']})\n"
    )

summary_txt.append("========================================\n")

# In ra console
print("".join(summary_txt))

# --- Lưu file metrics ---
with open(OUTPUT_DIR / "finbert_metrics_summary.txt", "w", encoding="utf-8") as f:
    f.writelines(summary_txt)

with open(OUTPUT_DIR / "finbert_metrics_detail.json", "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2, ensure_ascii=False)

# --- Vẽ confusion matrix ---
plt.figure(figsize=(6, 4))
sns.heatmap(
    conf_mat, annot=True, fmt="d", cmap="Blues",
    xticklabels=model.config.id2label.values(),
    yticklabels=model.config.id2label.values()
)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("FinBERT Confusion Matrix (media_test)")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=200)
plt.close()

# --- Vẽ biểu đồ F1-score từng lớp ---
f1_per_class = {cls: report[cls]["f1-score"] for cls in report if cls not in ["accuracy", "macro avg", "weighted avg"]}
plt.figure(figsize=(5, 3))
sns.barplot(x=list(f1_per_class.keys()), y=list(f1_per_class.values()), palette="crest")
plt.title("FinBERT per-class F1-score")
plt.ylabel("F1-score")
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "f1_per_class.png", dpi=200)
plt.close()
print("[DONE] Saved metrics summary, JSON, confusion matrix, and F1 bar chart.")

# === 5. Lưu CSV kết quả dự đoán ===
df.to_csv(OUTPUT_DIR / "finbert_predictions.csv", index=False, encoding="utf-8")

# === 6. Trích xuất toàn bộ score 3 lớp + logits (embeddings) ===
print("[INFO] Extracting sentiment score vectors & logits...")
model.eval()
all_probs, all_logits = [], []
BATCH_SIZE = 32

for i in range(0, len(df), BATCH_SIZE):
    batch = df["title"].iloc[i:i+BATCH_SIZE].tolist()
    inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = softmax(logits, dim=-1)
    all_logits.append(logits.cpu().numpy())
    all_probs.append(probs.cpu().numpy())

all_logits = np.concatenate(all_logits, axis=0)
all_probs = np.concatenate(all_probs, axis=0)
df["score_pos"], df["score_neg"], df["score_neu"] = all_probs[:, 0], all_probs[:, 1], all_probs[:, 2]

np.save(OUTPUT_DIR / "media_test_embeddings.npy", all_logits)
df[["title", "pred_label", "score_pos", "score_neg", "score_neu"]].to_csv(
    OUTPUT_DIR / "finbert_sentiment_scores.csv", index=False, encoding="utf-8"
)

print(f"[DONE] Saved embeddings → {OUTPUT_DIR / 'media_test_embeddings.npy'}")
print(f"[DONE] Saved sentiment scores → {OUTPUT_DIR / 'finbert_sentiment_scores.csv'}")

# # === 7. (Optional) Ghi vào PostgreSQL ===
# try:
#     engine = get_postgres_engine()
#     df_to_save = df[["id", "score_pos", "score_neg", "score_neu", "pred_label"]].copy()
#     df_to_save.to_sql("sentiment_scores", engine, schema="it_result",
#                       if_exists="replace", index=False)
#     print("[DONE] Written sentiment_scores → it_result.sentiment_scores")
# except Exception as e:
#     print(f"[WARN] Cannot write to PostgreSQL: {e}")

runtime = time.time() - start
print(f"[DONE] FinBERT FULL Inference finished in {runtime:.2f} s")
print("=" * 70)
