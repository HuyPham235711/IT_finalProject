# finbert_finetune_sampler_I_am_speed.py
import os
import platform
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import WeightedRandomSampler, DataLoader

from sqlalchemy import create_engine, text
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report

from transformers import (
    AutoTokenizer, AutoConfig, AutoModelForSequenceClassification,
    Trainer, TrainingArguments, DataCollatorWithPadding
)
from transformers.modeling_outputs import SequenceClassifierOutput

# ================== GPU / DEVICE ==================
REQUIRE_CUDA = True

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
        print(f"[Using device]: {device}")
    else:
        device = torch.device("cpu")
        print(f"[Using device]: {device}")
        if REQUIRE_CUDA:
            raise SystemExit("❌ CUDA không khả dụng — đang chạy CPU.")
    return device

device = setup_device()
torch.set_num_threads(1)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")

# ================== PATHS ==================
HERE = Path(__file__).resolve()
PROJECT_ROOT = HERE.parents[3]               # .../DU_AN_CNTT
OUTPUT_DIR = PROJECT_ROOT / "models" / "finBERT" / "finbert_finetuned_sampler_v3"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"[Paths] PROJECT_ROOT={PROJECT_ROOT}")
print(f"[Paths] OUTPUT_DIR={OUTPUT_DIR}")

# ================== CONFIG ==================
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
torch.backends.cudnn.benchmark = True

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

PG_CONN_STR = "postgresql+psycopg2://postgres:123456789@localhost:5432/postgres"
TABLE_TRAIN = "it_final.media_train"
TABLE_VALID = "it_final.media_valid"
TABLE_TEST  = "it_final.media_test"

TEXT_COLUMN = "title"
LABELS = ["negative", "neutral", "positive"]
label2id = {lbl: i for i, lbl in enumerate(LABELS)}
id2label = {i: lbl.capitalize() for i, lbl in enumerate(LABELS)}

BASE_MODEL = "ProsusAI/finbert"

TRAIN_BS = 8
EVAL_BS  = 16
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
    tqdm.pandas(desc="Token len")
    df["token_len"] = df["text"].progress_apply(
        lambda x: len(tokenizer(x, truncation=False, add_special_tokens=False)["input_ids"])
    )
    df = df[df["token_len"] > 0]
    df["sentiment_label"] = df["sentiment_label"].astype(str).str.strip().str.lower()
    df = df[df["sentiment_label"].isin(LABELS)]
    df["label_id"] = df["sentiment_label"].map(label2id).astype(int)
    return df[["text", "label_id", "sentiment_label", "token_len"]]

train_df = load_split(TABLE_TRAIN)
valid_df = load_split(TABLE_VALID)
test_df  = load_split(TABLE_TEST)

print(f"[Data] Train={len(train_df):,} Valid={len(valid_df):,} Test={len(test_df):,}")
print("[Train label dist]\n", train_df["sentiment_label"].value_counts(normalize=True))

# ================== SAFE MAX_LENGTH PICKER ==================
def pick_max_len(*dfs, p=99, cap=512, multiple_of=8, floor=256):
    all_lens = pd.concat([d["token_len"] for d in dfs], ignore_index=True).to_numpy()
    pxx = int(np.percentile(all_lens, p))
    rounded = ((pxx + multiple_of - 1) // multiple_of) * multiple_of
    return max(min(rounded, cap), floor)

MAX_LENGTH = pick_max_len(train_df, valid_df, test_df, p=99, cap=512, multiple_of=8, floor=256)
print(f"[MAX_LENGTH] = {MAX_LENGTH} tokens (floor=256, cap=512)")

# ================== HEAD+TAIL ENCODER ==================
def encode_head_tail(text: str, tokenizer, max_len: int, tail_ratio: float = 0.30):
    enc_full = tokenizer(text, add_special_tokens=False, return_attention_mask=False)
    ids = enc_full["input_ids"]
    if len(ids) <= max_len - 2:
        input_ids = [tokenizer.cls_token_id] + ids + [tokenizer.sep_token_id]
        attn = [1] * len(input_ids)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
        }
    avail = max_len - 2
    tail_len = max(1, int(avail * tail_ratio))
    head_len = avail - tail_len
    head = ids[:head_len]
    tail = ids[-tail_len:]
    input_ids = [tokenizer.cls_token_id] + head + tail + [tokenizer.sep_token_id]
    attn = [1] * len(input_ids)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attn, dtype=torch.long),
    }

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
        enc = encode_head_tail(text, self.tokenizer, max_len=self.max_length, tail_ratio=0.30)
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }

train_ds = FinDataset(train_df, tokenizer, MAX_LENGTH)
valid_ds = FinDataset(valid_df, tokenizer, MAX_LENGTH)
test_ds  = FinDataset(test_df, tokenizer, MAX_LENGTH)

# Dynamic padding
data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)

# ================== MODEL + LOSS (fixed: no shared tensors) ==================
config = AutoConfig.from_pretrained(
    BASE_MODEL, num_labels=len(LABELS), label2id=label2id, id2label=id2label
)
base_model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, config=config)

class_weights = compute_class_weight(
    "balanced", classes=np.arange(len(LABELS)), y=train_df["label_id"]
)
class_weights = torch.tensor(class_weights, dtype=torch.float, device=device)

class WeightedLossModel(nn.Module):
    """
    - Không register_buffer.
    - Không giữ nn.CrossEntropyLoss / FocalLoss làm submodule.
    - Tính loss bằng F.cross_entropy để tránh shared-storage trong state_dict.
    """
    def __init__(self, model, weights, use_focal=False, gamma=2.0):
        super().__init__()
        self.model = model
        self.use_focal = bool(use_focal)
        self.gamma = float(gamma)
        # clone về device, KHÔNG register_buffer
        self.class_weights = weights.detach().clone()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None):
        out = self.model(input_ids=input_ids, attention_mask=attention_mask,
                         token_type_ids=token_type_ids, labels=None)
        logits = out.logits
        loss = None
        if labels is not None:
            if self.use_focal:
                ce = F.cross_entropy(logits, labels, weight=self.class_weights, reduction="none")
                with torch.no_grad():
                    pt = torch.softmax(logits, dim=-1).gather(1, labels.view(-1,1)).squeeze(1).clamp_(1e-6, 1.0)
                loss = ((1.0 - pt) ** self.gamma * ce).mean()
            else:
                loss = F.cross_entropy(logits, labels, weight=self.class_weights, reduction="mean")
        return SequenceClassifierOutput(loss=loss, logits=logits)

model = WeightedLossModel(base_model, class_weights, use_focal=USE_FOCAL, gamma=FOCAL_GAMMA).to(device)

# ================== METRICS ==================
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}

# ================== SAMPLER ==================
class_sample_count = np.array([(train_df["label_id"] == i).sum() for i in range(len(LABELS))], dtype=float)
weights = 1.0 / class_sample_count
samples_weights = np.array([weights[i] for i in train_df["label_id"]], dtype=float)
sampler = WeightedRandomSampler(torch.DoubleTensor(samples_weights), len(samples_weights), replacement=True)

# ================== TRAINER (override) ==================
class ImbTrainer(Trainer):
    def get_train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            sampler=sampler,
            collate_fn=self.data_collator,
            num_workers=0,
            pin_memory=self.args.dataloader_pin_memory,
        )

# ================== ARGS ==================
args = TrainingArguments(
    output_dir=str(OUTPUT_DIR),
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=TRAIN_BS,
    per_device_eval_batch_size=EVAL_BS,
    learning_rate=LR,
    weight_decay=WEIGHT_DECAY,
    warmup_ratio=WARMUP_RATIO,

    eval_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="steps",
    logging_steps=100,
    load_best_model_at_end=True,
    metric_for_best_model="eval_f1",
    greater_is_better=True,

    bf16=False,
    fp16=True,

    report_to="none",
    remove_unused_columns=False,

    dataloader_num_workers=0,
    dataloader_pin_memory=False,
    seed=seed,

    gradient_accumulation_steps=2,
    # (giữ mặc định save_safetensors=True; đã fix code để an toàn)
)

trainer = ImbTrainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=valid_ds,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# ================== PRIOR & DIAGNOSTICS ==================
def get_prior(df):
    counts = df["label_id"].value_counts(normalize=True).reindex(range(len(LABELS))).fillna(0).to_numpy()
    return counts

def trunc_rate(df, max_len):
    return (df["token_len"].to_numpy() > (max_len - 2)).mean()

def eval_by_len(y_true, y_pred, lens, max_len):
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    m_short = lens <= (max_len - 2)
    m_long  = ~m_short
    def rep(mask, name):
        yt, yp = y_true[mask], y_pred[mask]
        if yt.size == 0:
            print(f"[{name}] n=0"); return
        acc = accuracy_score(yt, yp)
        f1 = precision_recall_fscore_support(yt, yp, average="macro", zero_division=0)[2]
        print(f"[{name}] n={yt.size} acc={acc:.4f} macroF1={f1:.4f}")
    print("\n[By length buckets]")
    rep(m_short, "NO-TRUNC (<= MAX_LENGTH)")
    rep(m_long , "TRUNC   (> MAX_LENGTH)")

# ================== MAIN ==================
def main():
    print(f"[Truncation rate] train={trunc_rate(train_df, MAX_LENGTH):.3f}  "
          f"valid={trunc_rate(valid_df, MAX_LENGTH):.3f}  "
          f"test={trunc_rate(test_df, MAX_LENGTH):.3f}")

    print("[Train] Start...")
    trainer.train()

    # Raw eval
    print("\n[Eval] raw on test...")
    raw = trainer.predict(test_ds)
    print(raw.metrics)
    y_true = test_df["label_id"].to_numpy()
    y_pred = raw.predictions.argmax(-1)
    lens   = test_df["token_len"].to_numpy()
    print(classification_report(y_true, y_pred, target_names=[s.capitalize() for s in LABELS]))
    eval_by_len(y_true, y_pred, lens, MAX_LENGTH)

    # Prior adjust
    train_prior = get_prior(train_df)
    valid_prior = get_prior(valid_df)
    if PRIOR_TARGET == "uniform":
        target_prior = np.array([1/len(LABELS)]*len(LABELS))
    elif PRIOR_TARGET == "valid":
        target_prior = valid_prior
    else:
        target_prior = train_prior
    log_adj = (np.log(target_prior + 1e-12) - np.log(train_prior + 1e-12))
    adj_logits = raw.predictions + log_adj
    y_pred_adj = adj_logits.argmax(-1)

    print("\n[Eval] prior-adjusted on test...")
    acc = accuracy_score(y_true, y_pred_adj)
    f1  = precision_recall_fscore_support(y_true, y_pred_adj, average="macro", zero_division=0)[2]
    print({"accuracy": acc, "macro_f1": f1})
    print(classification_report(y_true, y_pred_adj, target_names=[s.capitalize() for s in LABELS]))
    eval_by_len(y_true, y_pred_adj, lens, MAX_LENGTH)

    # Save
    print("\n[Save] Model + tokenizer...")
    trainer.save_model(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))
    print(f"[✅ Done] Saved at {OUTPUT_DIR}")

if __name__ == "__main__":
    import torch.multiprocessing as mp
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main()
