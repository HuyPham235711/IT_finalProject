# finbert_finetune_sampler.py
import os
import platform
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import WeightedRandomSampler, DataLoader

from sqlalchemy import create_engine, text
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report

from transformers import (
    AutoTokenizer, AutoConfig, AutoModelForSequenceClassification,
    Trainer, TrainingArguments, default_data_collator
)
from transformers.modeling_outputs import SequenceClassifierOutput


# ================== GPU / PRECISION CHECK ==================
REQUIRE_CUDA = True  # đặt False nếu chấp nhận chạy CPU

def setup_device():
    print(f"[Python] {platform.python_version()}  [PyTorch] {torch.__version__}")
    cuda_ok = torch.cuda.is_available()
    print(f"[CUDA available]: {cuda_ok}")
    if cuda_ok:
        n = torch.cuda.device_count()
        print(f"[GPU count]: {n}")
        for i in range(n):
            print(f"[GPU {i}]: {torch.cuda.get_device_name(i)}")
        device = torch.device("cuda:0")
        # bf16 nếu GPU hỗ trợ, nếu không dùng fp16
        bf16_ok = getattr(torch.cuda, "is_bf16_supported", lambda: False)()
        fp16_ok = (not bf16_ok)
        print(f"[Using device]: {device}")
        print(f"[Precision support] bf16={bf16_ok} fp16={fp16_ok}")
    else:
        device = torch.device("cpu")
        bf16_ok = False
        fp16_ok = False
        print(f"[Using device]: {device}")
        if REQUIRE_CUDA:
            raise SystemExit("❌ CUDA không khả dụng — đang chạy CPU. Kiểm tra driver/PyTorch CUDA.")
    return device, bf16_ok, fp16_ok

device, bf16_ok, fp16_ok = setup_device()
torch.set_num_threads(1)  # tránh tạo nhiều thread CPU trên Windows


# ================== PATHS / OUTPUT ==================
# file này: DU_AN_CNTT/src/finBERT/finetune_finBERT/finbert_finetune_sampler.py
HERE = Path(__file__).resolve()
PKG_DIR = HERE.parent                 # .../finetune_finBERT
FINBERT_DIR = PKG_DIR.parent          # .../finBERT
SRC_DIR = FINBERT_DIR.parent          # .../src
PROJECT_ROOT = SRC_DIR.parent         # .../DU_AN_CNTT

OUTPUT_DIR = PROJECT_ROOT / "models" / "finBERT" / "finbert_finetuned_sampler"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"[Paths] PROJECT_ROOT={PROJECT_ROOT}")
print(f"[Paths] OUTPUT_DIR={OUTPUT_DIR}")


# ================== CONFIG ==================
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["PYTORCH_HIP_ALLOC_CONF"] = "expandable_segments:True"
torch.backends.cudnn.benchmark = True
rng_seed = 42
np.random.seed(rng_seed)
torch.manual_seed(rng_seed)

PG_CONN_STR = "postgresql+psycopg2://postgres:123456789@localhost:5432/postgres"
TABLE_TRAIN = "it_final.media_train"
TABLE_VALID = "it_final.media_valid"
TABLE_TEST  = "it_final.media_test"

TEXT_COLUMN = "title"
LABELS = ["negative", "neutral", "positive"]
label2id = {lbl: i for i, lbl in enumerate(LABELS)}
id2label = {i: lbl.capitalize() for i, lbl in enumerate(LABELS)}

BASE_MODEL = "ProsusAI/finbert"
MAX_LENGTH = 512
TRAIN_BS = 16
EVAL_BS = 32
LR = 2e-5
EPOCHS = 4
WARMUP_RATIO = 0.06
WEIGHT_DECAY = 0.05

USE_FOCAL = True
FOCAL_GAMMA = 2.0
PRIOR_TARGET = "uniform"  # "uniform" | "valid" | "train"


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
    tqdm.pandas(desc=f"Token len")
    df["token_len"] = df["text"].progress_apply(lambda x: len(tokenizer(x, truncation=True)["input_ids"]))
    df = df[df["token_len"] > 0]
    df["sentiment_label"] = df["sentiment_label"].astype(str).str.strip().str.lower()
    df = df[df["sentiment_label"].isin(LABELS)]
    df["label_id"] = df["sentiment_label"].map(label2id).astype(int)
    return df[["text", "label_id", "sentiment_label"]]

train_df = load_split(TABLE_TRAIN)
valid_df = load_split(TABLE_VALID)
test_df  = load_split(TABLE_TEST)

print(f"[Data] Train={len(train_df):,} Valid={len(valid_df):,} Test={len(test_df):,}")
print("[Train label dist]\n", train_df["sentiment_label"].value_counts(normalize=True))


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


# ================== MODEL + LOSS ==================
config = AutoConfig.from_pretrained(BASE_MODEL, num_labels=len(LABELS),
                                    label2id=label2id, id2label=id2label)
base_model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, config=config)

# class weights theo train
class_weights = compute_class_weight("balanced",
                                     classes=np.arange(len(LABELS)),
                                     y=train_df["label_id"])
class_weights = torch.tensor(class_weights, dtype=torch.float, device=device)

class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0):
        super().__init__()
        self.weight = weight
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(weight=weight, reduction="none")
    def forward(self, logits, y):
        ce = self.ce(logits, y)  # (B,)
        with torch.no_grad():
            pt = torch.softmax(logits, dim=-1).gather(1, y.view(-1,1)).squeeze(1).clamp_(1e-6, 1.0)
        loss = ((1 - pt) ** self.gamma) * ce
        return loss.mean()

class WeightedLossModel(nn.Module):
    def __init__(self, model, weights, use_focal=False, gamma=2.0):
        super().__init__()
        self.model = model
        self.use_focal = use_focal
        self.gamma = gamma
        self.register_buffer("weights", weights)
        self.ce = nn.CrossEntropyLoss(weight=self.weights)
        self.focal = FocalLoss(weight=self.weights, gamma=self.gamma)
    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None):
        out = self.model(input_ids=input_ids, attention_mask=attention_mask,
                         token_type_ids=token_type_ids, labels=None)
        logits = out.logits
        loss = None
        if labels is not None:
            loss = self.focal(logits, labels) if self.use_focal else self.ce(logits, labels)
        return SequenceClassifierOutput(loss=loss, logits=logits)

model = WeightedLossModel(base_model, class_weights, use_focal=USE_FOCAL, gamma=FOCAL_GAMMA).to(device)


# ================== METRICS ==================
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


# ================== SAMPLER (cân bằng) ==================
class_sample_count = np.array([(train_df["label_id"] == i).sum() for i in range(len(LABELS))], dtype=float)
weights = 1.0 / class_sample_count
samples_weights = np.array([weights[i] for i in train_df["label_id"]], dtype=float)
sampler = WeightedRandomSampler(torch.DoubleTensor(samples_weights), len(samples_weights), replacement=True)


# ================== TRAINER (override để dùng sampler) ==================
class ImbTrainer(Trainer):
    def get_train_dataloader(self):
        # Không đa luồng: num_workers=0 (Windows an toàn), pin_memory vẫn bật
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            sampler=sampler,
            collate_fn=self.data_collator,
            num_workers=0,                    # << quan trọng: không tạo worker con
            pin_memory=self.args.dataloader_pin_memory,
        )


# ================== TRAINING ARGS ==================
args = TrainingArguments(
    output_dir=str(OUTPUT_DIR),
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=TRAIN_BS,
    per_device_eval_batch_size=EVAL_BS,
    learning_rate=LR,
    weight_decay=WEIGHT_DECAY,
    warmup_ratio=WARMUP_RATIO,

    eval_strategy="epoch",           # transformers bản cũ dùng eval_strategy
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

    dataloader_num_workers=0,        # << đảm bảo không spawn process phụ
    dataloader_pin_memory=True,
    seed=rng_seed,
)

trainer = ImbTrainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=valid_ds,
    data_collator=default_data_collator,
    compute_metrics=compute_metrics,
)


# ================== MAIN (Windows guard) ==================
def get_prior(df):
    counts = df["label_id"].value_counts(normalize=True).reindex(range(len(LABELS))).fillna(0).to_numpy()
    return counts

def main():
    print("[Train] Start...")
    trainer.train()

    # === Eval raw ===
    print("[Eval] raw on test...")
    raw_pred = trainer.predict(test_ds)
    print(raw_pred.metrics)
    print(classification_report(raw_pred.label_ids,
                                raw_pred.predictions.argmax(-1),
                                target_names=[s.capitalize() for s in LABELS]))

    # === Prior adjust (tùy chọn) ===
    train_prior = get_prior(train_df)
    valid_prior = get_prior(valid_df)
    if PRIOR_TARGET == "uniform":
        target_prior = np.array([1/len(LABELS)]*len(LABELS))
    elif PRIOR_TARGET == "valid":
        target_prior = valid_prior
    else:
        target_prior = train_prior

    log_adj = (np.log(target_prior + 1e-12) - np.log(train_prior + 1e-12))
    adj_logits = raw_pred.predictions + log_adj
    y_pred = adj_logits.argmax(-1)
    acc = accuracy_score(test_df["label_id"], y_pred)
    f1 = precision_recall_fscore_support(test_df["label_id"], y_pred, average="macro", zero_division=0)[2]
    print("\n[Eval] prior-adjusted on test...")
    print({"accuracy": acc, "macro_f1": f1})
    print(classification_report(test_df["label_id"], y_pred,
                                target_names=[s.capitalize() for s in LABELS]))

    # === Save ===
    print("[Save] Model + tokenizer...")
    trainer.save_model(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))
    print(f"[✅ Done] Saved at {OUTPUT_DIR}")

if __name__ == "__main__":
    # Windows: đảm bảo start method "spawn"
    import torch.multiprocessing as mp
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main()
