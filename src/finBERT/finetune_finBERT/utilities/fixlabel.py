# import json, os

# MODEL_DIR = r"E:\TDTu\TAI_LIEU\KY1-NAM5\DU_AN_CNTT\models\finBERT\finbert_finetuned_sampler_fixed"
# cfg_path = os.path.join(MODEL_DIR, "config.json")

# with open(cfg_path, "r", encoding="utf-8") as f:
#     cfg = json.load(f)

# cfg["id2label"] = {"0": "Neutral", "1": "Positive", "2": "Negative"}
# cfg["label2id"] = {"Neutral": 0, "Positive": 1, "Negative": 2}
# cfg["num_labels"] = 3

# with open(cfg_path, "w", encoding="utf-8") as f:
#     json.dump(cfg, f, indent=2)

# print("✅ Fixed label order: Neutral=0, Positive=1, Negative=2")

# import json, os

# MODEL_DIR = r"E:\TDTu\TAI_LIEU\KY1-NAM5\DU_AN_CNTT\models\finBERT\finbert_finetuned_sampler_fixed"
# cfg_path = os.path.join(MODEL_DIR, "config.json")

# with open(cfg_path, "r", encoding="utf-8") as f:
#     cfg = json.load(f)

# # convert keys to strings
# cfg["id2label"] = {str(k): v for k, v in cfg["id2label"].items()}
# cfg["label2id"] = {k: int(v) for k, v in cfg["label2id"].items()}

# with open(cfg_path, "w", encoding="utf-8") as f:
#     json.dump(cfg, f, indent=2)

# print("✅ Fixed id2label keys to strings.")


# import json, os

# MODEL_DIR = r"E:\TDTu\TAI_LIEU\KY1-NAM5\DU_AN_CNTT\models\finBERT\finbert_finetuned_sampler_fixed"
# cfg_path = os.path.join(MODEL_DIR, "config.json")

# print(f"[🔧] Fixing id2label keys → strings: {cfg_path}")

# with open(cfg_path, "r", encoding="utf-8") as f:
#     cfg = json.load(f)

# # 🔄 convert keys to strings (HuggingFace standard)
# cfg["id2label"] = {str(k): v for k, v in cfg["id2label"].items()}
# cfg["label2id"] = {k: int(v) for k, v in cfg["label2id"].items()}

# with open(cfg_path, "w", encoding="utf-8") as f:
#     json.dump(cfg, f, indent=2)

# print("✅ All id2label keys converted to string successfully.")

import json, os

MODEL_DIR = r"E:\TDTu\TAI_LIEU\KY1-NAM5\DU_AN_CNTT\models\finBERT\finbert_finetuned_sampler_fixed"
cfg_path = os.path.join(MODEL_DIR, "config.json")

print(f"[🔧] Forcing id2label keys to string (HuggingFace-safe format): {cfg_path}")

with open(cfg_path, "r", encoding="utf-8") as f:
    cfg = json.load(f)

# ép key thành string đúng chuẩn
new_id2label = {}
for k, v in cfg["id2label"].items():
    new_id2label[str(k)] = v

cfg["id2label"] = new_id2label
cfg["label2id"] = {str(k): int(v) for k, v in cfg["label2id"].items()}

# save lại
with open(cfg_path, "w", encoding="utf-8") as f:
    json.dump(cfg, f, indent=2, ensure_ascii=False)

print("✅ id2label keys converted to string successfully.")

