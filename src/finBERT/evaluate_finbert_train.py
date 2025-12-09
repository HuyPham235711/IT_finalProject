# src/models/finbert/evaluate_finbert_full.py
"""
Chạy toàn bộ inference FinBERT:
- Inference + đánh giá trên media_train (Postgres)
- Tính metrics, vẽ confusion matrix
- Trích xuất score 3 lớp (pos/neg/neu) + logits embedding
- Lưu CSV, JSON, PNG, NPY
"""

from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from torch.nn.functional import softmax
from sklearn.metrics import classification_report, confusion_matrix
from pathlib import Path
import torch, json, numpy as np, pandas as pd, seaborn as sns, matplotlib.pyplot as plt
from src.db.postgres_conn import load_table_to_df, get_postgres_engine
import time

# === 0. Cấu hình chung ===
MODEL_PATH = Path(r"E:\TDTu\TAI_LIEU\KY1-NAM5\DU_AN_CNTT\models\finBERT\finbert_finetuned_safe_dataBalanced")
OUTPUT_DIR = Path(r"E:\TDTu\TAI_LIEU\KY1-NAM5\DU_AN_CNTT\results\finbert")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

start = time.time()
device = 0 if torch.cuda.is_available() else -1
print("=" * 70)
print("[INFO] FinBERT FULL Inference starting...")
print(f"[INFO] Using device: {'cuda:0' if device == 0 else 'cpu'}")

# === 1. Load model ===
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model.config.id2label = {0: 'positive', 1: 'negative', 2: 'neutral'}
model.config.label2id = {'positive': 0, 'negative': 1, 'neutral': 2}
clf = pipeline("text-classification", model=model, tokenizer=tokenizer, device=device)
print(f"[INFO] Model loaded from: {MODEL_PATH}")

# === 2. Load data từ Postgres ===
df = load_table_to_df("media_train", schema="it_final")
print(f"[INFO] Loaded {len(df)} rows from media_train")

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
print(json.dumps(report, indent=2))

# === 5. Save metrics & predictions ===
with open(OUTPUT_DIR / "finbert_baseline_metrics_train.json", "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2)
df.to_csv(OUTPUT_DIR / "finbert_baseline_predictions_train.csv", index=False, encoding="utf-8")

plt.figure(figsize=(6, 4))
sns.heatmap(conf_mat, annot=True, fmt="d", cmap="Blues",
            xticklabels=model.config.id2label.values(),
            yticklabels=model.config.id2label.values())
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("FinBERT Baseline Confusion Matrix (media_train)")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "confusion_matrix_train.png", dpi=200)
plt.close()
print("[DONE] Saved metrics, CSV, and confusion matrix.")

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
df["score_pos"], df["score_neg"], df["score_neu"] = all_probs[:,0], all_probs[:,1], all_probs[:,2]

np.save(OUTPUT_DIR / "media_train_embeddings.npy", all_logits)
df[["title", "pred_label", "score_pos", "score_neg", "score_neu"]].to_csv(
    OUTPUT_DIR / "finbert_sentiment_scores_train.csv", index=False, encoding="utf-8"
)

print(f"[DONE] Saved embeddings → {OUTPUT_DIR / 'media_train_embeddings.npy'}")
print(f"[DONE] Saved sentiment scores → {OUTPUT_DIR / 'finbert_sentiment_scores_train.csv'}")

# # === 7. (Optional) Ghi vào PostgreSQL ===
# try:
#     engine = get_postgres_engine()
#     df_to_save = df[["id", "score_pos", "score_neg", "score_neu", "pred_label"]].copy()
#     df_to_save.to_sql("sentiment_scores", engine, schema="it_result",
#                       if_exists="replace", index=False)
#     print("[DONE] Written sentiment_scores → it_result.sentiment_scores")
# except Exception as e:
#     print(f"[WARN] Cannot write to PostgreSQL: {e}")

# runtime = time.time() - start
# print(f"[DONE] FinBERT FULL Inference finished in {runtime:.2f} s")
# print("=" * 70)
