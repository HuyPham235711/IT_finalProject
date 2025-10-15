# finbert_finetune_full.py
import os
import math
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    default_data_collator,
)
from transformers.modeling_outputs import SequenceClassifierOutput
from sqlalchemy import create_engine, text
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import torch
from torch import nn
from torch.utils.data import WeightedRandomSampler

# ================== ENV ==================
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["PYTORCH_HIP_ALLOC_CONF"] = "expandable_segments:True"
torch.backends.cudnn.benchmark = True

# ================== CONFIG ==================
PG_CONN_STR = "postgresql+psycopg2://postgres:123456@localhost:5432/postgres"
TABLE_TRAIN = "it.media_train"
TABLE_VALID = "it.media_valid"
TABLE_TEST  = "it.media_test"

TEXT_COLUMN = "title"
LABELS = ["negative", "neutral", "positive"]
label2id = {lbl: i for i, lbl in enumerate(LABELS)}
id2label = {i: lbl.capitalize() for i, lbl in enumerate(LABELS)}

BASE_MODEL = "ProsusAI/finbert"
OUTPUT_DIR = "./finbert_finetuned_safe"
MAX_LENGTH = 512
TRAIN_BS = 16
EVAL_BS = 32
LEARNING_RATE = 2e-5
NUM_EPOCHS = 4
WARMUP_RATIO = 0.06
WEIGHT_DECAY = 0.05

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"[Device] Using: {device}")

bf16_ok = getattr(torch.cuda, "is_bf16_supported", lambda: False)()
fp16_ok = not bf16_ok and torch.cuda.is_available()

# ================== TOKENIZER ==================
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)

# ================== LOAD DATA ==================
def load_split(table_name: str):
    engine = create_engine(PG_CONN_STR)
    query = text(f"""
        SELECT datetime, {TEXT_COLUMN} AS text, sentiment_label
        FROM {table_name}
        WHERE {TEXT_COLUMN} IS NOT NULL AND sentiment_label IS NOT NULL
    """)
    df = pd.read_sql(query, engine)
    df["text"] = df["text"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    df = df[df["text"].str.len() > 0]

    tqdm.pandas(desc=f"Filtering empty texts")
    df["token_len"] = df["text"].progress_apply(lambda x: len(tokenizer(x)["input_ids"]))
    df = df[df["token_len"] > 0]

    # chuẩn hóa nhãn
    df["sentiment_label"] = df["sentiment_label"].astype(str).str.strip().str.lower()
    df = df[df["sentiment_label"].isin(LABELS)]
    df["label_id"] = df["sentiment_label"].map(label2id).astype(int)
    return df[["text", "label_id", "sentiment_label"]]

train_df = load_split(TABLE_TRAIN)
valid_df = load_split(TABLE_VALID)
test_df  = load_split(TABLE_TEST)
print(f"[Data] Train={len(train_df):,}, Valid={len(valid_df):,}, Test={len(test_df):,}")
print("[Label Distribution]")
print(train_df["sentiment_label"].value_counts(normalize=True))

# ================== DATASET ==================
class FinDataset(torch.utils.data.Dataset):
    def __init__(self, df, tokenizer, max_length):
        self.texts = df["text"].tolist()
        self.labels = df["label_id"].tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self): return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx] or "[EMPTY]"
        enc = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

train_ds = FinDataset(train_df, tokenizer, MAX_LENGTH)
valid_ds = FinDataset(valid_df, tokenizer, MAX_LENGTH)
test_ds  = FinDataset(test_df, tokenizer, MAX_LENGTH)

# ================== MODEL ==================
config = AutoConfig.from_pretrained(BASE_MODEL, num_labels=len(LABELS), label2id=label2id, id2label=id2label)
base_model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, config=config)

class_weights = compute_class_weight("balanced", classes=np.arange(len(LABELS)), y=train_df["label_id"])
class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

class WeightedLossModel(nn.Module):
    def __init__(self, model, weights):
        super().__init__()
        self.model = model
        self.register_buffer("weights", weights)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=None  # bỏ loss gốc
        )
        logits = outputs.logits
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(weight=self.weights)
            loss = loss_fct(logits, labels)
        return SequenceClassifierOutput(loss=loss, logits=logits)

model = WeightedLossModel(base_model, class_weights).to(device)

# ================== METRICS ==================
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}

# ================== SAMPLER ==================
class_sample_count = np.array(
    [len(train_df[train_df.label_id == t]) for t in np.unique(train_df.label_id)]
)
weights = 1. / class_sample_count
samples_weights = np.array([weights[t] for t in train_df.label_id])
sampler = WeightedRandomSampler(torch.DoubleTensor(samples_weights), len(samples_weights))

# ================== TRAINING ==================
args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=TRAIN_BS,
    per_device_eval_batch_size=EVAL_BS,
    learning_rate=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    warmup_ratio=WARMUP_RATIO,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="steps",
    logging_steps=100,
    load_best_model_at_end=True,
    metric_for_best_model="eval_f1",
    greater_is_better=True,
    bf16=bf16_ok,
    fp16=fp16_ok,
    report_to="none",
    remove_unused_columns=False,
    dataloader_num_workers=4,
    dataloader_pin_memory=True,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=valid_ds,
    data_collator=default_data_collator,
    compute_metrics=compute_metrics,
)

# ================== RUN ==================
print("[Train] Start full training...")
trainer.train()

print("[Eval] on test set...")
test_result = trainer.predict(test_ds)
print(test_result.metrics)
print(classification_report(
    test_result.label_ids,
    np.argmax(test_result.predictions, axis=-1),
    target_names=[lbl.capitalize() for lbl in LABELS]
))

print("[Save] Saving model...")
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"[✅ Done] Model saved at {OUTPUT_DIR}")
