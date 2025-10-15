import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, default_data_collator
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch import nn
import numpy as np

# ==================== CONFIG ====================
MODEL = "ProsusAI/finbert"
LABELS = ["Negative", "Neutral", "Positive"]
label2id = {lbl: i for i, lbl in enumerate(LABELS)}
id2label = {i: lbl for lbl, i in label2id.items()}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Device] {device}")

# ==================== DATA MINI ====================
texts = [
    "Bitcoin crashes after SEC warning",
    "The stock market remains stable today",
    "Tesla reports record profits this quarter",
    "Oil prices fall due to lower demand",
    "Gold is a safe-haven asset",
    "Interest rates expected to rise soon",
]
labels = [0, 1, 2, 0, 1, 2]  # random mini labels

tokenizer = AutoTokenizer.from_pretrained(MODEL)
enc = tokenizer(texts, padding="max_length", truncation=True, max_length=64, return_tensors="pt")
dataset = [{"input_ids": enc["input_ids"][i],
            "attention_mask": enc["attention_mask"][i],
            "labels": torch.tensor(labels[i])} for i in range(len(labels))]

# ==================== MODEL ====================
model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=len(LABELS))
model.to(device)

# ==================== METRICS ====================
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}

# ==================== TRAINING ====================
args = TrainingArguments(
    output_dir="./finbert_test_run",
    num_train_epochs=1,
    per_device_train_batch_size=2,
    learning_rate=5e-5,
    eval_strategy="epoch",
    save_strategy="no",
    logging_strategy="steps",
    logging_steps=1,
    report_to="none",
    fp16=torch.cuda.is_available(),
    bf16=False,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=dataset,
    eval_dataset=dataset,
    data_collator=default_data_collator,
    compute_metrics=compute_metrics,
)

print("[Test] Starting mini fine-tune...")
trainer.train()
print("[✅ OK] Mini finetune ran without crash!")
