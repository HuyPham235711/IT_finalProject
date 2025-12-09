import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import yaml
import os


# ============================================================================
#  Fusion MLP AutoEncoder (đúng kiến trúc bạn đã train trước đây)
# ============================================================================
class FusionMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon, z


# ============================================================================
#  MAIN
# ============================================================================
def main():

    # ---- Load backtest config ----
    CFG_PATH = Path("E:/TDTu/TAI_LIEU/KY1-NAM5/DU_AN_CNTT/config/backtest/fusion_config_backtest.yaml")

    with open(CFG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    print("=== Generating FUSION (BACKTEST) 3843 → 256 (using trained AutoEncoder) ===")

    # ---- Load backtest embeddings ----
    fin_path = cfg["sources"]["finbert_embeddings"]
    att_path = cfg["sources"]["attcnn_embeddings"]
    trf_path = cfg["sources"]["transformer_embeddings"]

    print("[LOAD]", fin_path)
    print("[LOAD]", att_path)
    print("[LOAD]", trf_path)

    fin = np.load(fin_path)
    att = np.load(att_path)
    trf = np.load(trf_path)

    print("finBERT:", fin.shape)
    print("ATT-CNN:", att.shape)
    print("Transformer:", trf.shape)

    # ---- Align rows (avoid mismatch) ----
    n = min(len(fin), len(att), len(trf))
    fin = fin[:n]
    att = att[:n]
    trf = trf[:n]

    fusion_raw = np.concatenate([trf, att, fin], axis=1)
    print("Fusion raw shape (3843-D):", fusion_raw.shape)

    # ============================================================================
    # Load trained Fusion AutoEncoder checkpoint
    # ============================================================================
    ckpt_path = Path(cfg["save_path"])
    print(f"[CHECKPOINT] Loading AE weights: {ckpt_path}")

    model = FusionMLP(
        input_dim=cfg["input_dim"],
        hidden_dim=cfg["hidden_dim"],
        output_dim=cfg["output_dim"],
        dropout=cfg["dropout"]
    )

    state = torch.load(ckpt_path, map_location="cpu")

    # Checkpoint là 1 state_dict thuần với encoder.*, decoder.*
    model.load_state_dict(state)
    model.eval()

    # ============================================================================
    # Encode (compress 3843 → 256)
    # ============================================================================
    print("[ENCODE] Compressing fusion embeddings to 256-D...")

    with torch.no_grad():
        x = torch.tensor(fusion_raw, dtype=torch.float32)
        _, z = model(x)
        fusion_256 = z.cpu().numpy()

    print("Compressed shape:", fusion_256.shape)

    # ============================================================================
    # SAVE
    # ============================================================================
    out_dir = Path("results/backtest/fusion")
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / "fusion_embeddings_backtest.npy"
    np.save(out_path, fusion_256)

    print(f"✅ Saved fusion compressed embeddings → {out_path}")
    print("=== DONE ===")


if __name__ == "__main__":
    main()
