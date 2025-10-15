# src/models/finbert/evaluate_finbert_baseline.py

from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from sklearn.metrics import classification_report, confusion_matrix
from pathlib import Path
import torch, json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from src.db.postgres_conn import load_table_to_df

# === 1. Load model ===
MODEL_PATH = Path(r"E:\TDTu\TAI_LIEU\KY1-NAM5\DU_AN_CNTT\models\finBERT\finbert_finetuned_safe_dataBalanced")
OUTPUT_DIR = Path(r"E:\TDTu\TAI_LIEU\KY1-NAM5\DU_AN_CNTT\results\finbert")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

# đảm bảo mapping đúng với file config gốc
model.config.id2label = {0: 'positive', 1: 'negative', 2: 'neutral'}
model.config.label2id = {'positive': 0, 'negative': 1, 'neutral': 2}
print("[INFO] Label mapping confirmed:", model.config.id2label)

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
device = 0 if torch.cuda.is_available() else -1
clf = pipeline("text-classification", model=model, tokenizer=tokenizer, device=device)

# === 2. Load data từ Postgres ===
df = load_table_to_df("media_test", schema="it_final")
print(f"Loaded {len(df)} rows from media_test")

# === 3. Chuẩn bị dữ liệu ===
texts = df["title"].astype(str).tolist()
true_labels = df["sentiment_label"].tolist()

# === 4. Inference ===
preds = clf(texts, truncation=True, padding=True)
df["pred_label"] = [p["label"] for p in preds]
df["pred_score"] = [p["score"] for p in preds]

# === 5. Evaluation ===
report = classification_report(true_labels, df["pred_label"], output_dict=True, digits=3)
conf_mat = confusion_matrix(true_labels, df["pred_label"])
print(json.dumps(report, indent=2))

# === 6. Save metrics ===
with open(OUTPUT_DIR / "finbert_baseline_metrics.json", "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2)

df.to_csv(OUTPUT_DIR / "finbert_baseline_predictions.csv", index=False, encoding="utf-8")

# === 7. Optional: Vẽ Confusion Matrix ===
plt.figure(figsize=(6,4))
sns.heatmap(conf_mat, annot=True, fmt="d", cmap="Blues",
            xticklabels=model.config.id2label.values(),
            yticklabels=model.config.id2label.values())
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("FinBERT Baseline Confusion Matrix (media_test)")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=200)
plt.close()
