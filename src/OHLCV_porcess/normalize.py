import json
import os

SCALER_PATH = "scaler.json"

def compute_scaler(rows):
    """Tính min/max cho toàn bộ tập train"""
    cols = list(zip(*rows))[1:]  # bỏ timestamp
    mins = [min(col) for col in cols]
    maxs = [max(col) for col in cols]
    return {"mins": mins, "maxs": maxs}

def save_scaler(scaler, path=SCALER_PATH):
    with open(path, "w") as f:
        json.dump(scaler, f)

def load_scaler(path=SCALER_PATH):
    with open(path, "r") as f:
        return json.load(f)

def apply_scaler(rows, scaler):
    mins = scaler["mins"]
    maxs = scaler["maxs"]

    result = []
    for r in rows:
        ts, o, h, l, c, v = r
        vals = [o, h, l, c, v]
        norm = []
        for i, val in enumerate(vals):
            vmin, vmax = mins[i], maxs[i]
            if vmax == vmin:
                norm.append(0.0)
            else:
                norm.append((val - vmin) / (vmax - vmin))
        result.append((ts, *norm))
    return result
