import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os

# === Đường dẫn model cần test ===
MODEL_DIR = r"E:\TDTu\TAI_LIEU\KY1-NAM5\DU_AN_CNTT\models\finBERT\finbert_finetuned_sampler_fixed"

print(f"[🔍] Loading model from: {MODEL_DIR}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.eval()

# === Hiển thị mapping trong config ===
print("\n=== CONFIG MAPPING ===")
print("label2id:", model.config.label2id)
print("id2label:", model.config.id2label)

# === Các câu test cơ bản ===
sentences = {
    "NEGATIVE": "This stock is performing terribly and will probably crash soon.",
    "NEUTRAL": "The company announced its quarterly earnings as expected.",
    "POSITIVE": "The financial results exceeded all market expectations and look very promising."
}

print("\n=== TEST SENTENCES ===")
for key, text in sentences.items():
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)
        pred_id = torch.argmax(probs, dim=-1).item()
        pred_label = model.config.id2label.get(str(pred_id), model.config.id2label.get(pred_id, f"UNK_{pred_id}"))

        print(f"{key:<8} → Pred: {pred_label:<8} | logits={logits.tolist()}")

print("""
=== CHECK RESULT ===
✅ Nếu NEGATIVE → Pred: Negative, v.v... thì nhãn mapping là đúng.
⚠️ Nếu bị đảo (vd. NEGATIVE → Neutral), model vẫn đang dùng mapping cũ (neutral=0, positive=1, negative=2).
""")
