import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import yaml


class FusionMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, dropout=0.2):
        super().__init__()
        enc = []
        prev = input_dim
        for h in hidden_dims:
            enc += [
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            prev = h
        enc.append(nn.Linear(prev, output_dim))
        self.encoder = nn.Sequential(*enc)

        dec = []
        prev = output_dim
        for h in reversed(hidden_dims):
            dec += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        dec.append(nn.Linear(prev, input_dim))
        self.decoder = nn.Sequential(*dec)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z


def main():
    print("=== FUSION BACKTEST (MULTI-PART) ===")

    CFG_PATH = Path(
        "E:/TDTu/TAI_LIEU/KY1-NAM5/DU_AN_CNTT/config/backtest/fusion_config_backtest.yaml"
    )

    with open(CFG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for part in ["part1", "part2", "part3"]:
        print(f"\n--- {part.upper()} ---")

        fin = np.load(cfg["sources"]["finbert_embeddings"].replace(".npy", f"_{part}.npy"))
        att = np.load(cfg["sources"]["attcnn_embeddings"].replace(".npy", f"_{part}.npy"))
        trf = np.load(cfg["sources"]["transformer_embeddings"].replace(".npy", f"_{part}.npy"))

        n = min(len(fin), len(att), len(trf))
        fusion_raw = np.concatenate([trf[:n], att[:n], fin[:n]], axis=1)

        model = FusionMLP(
            input_dim=fusion_raw.shape[1],
            hidden_dims=[1024, 512],
            output_dim=cfg["output_dim"],
            dropout=cfg["dropout"]
        ).to(device)

        state = torch.load(cfg["save_path"], map_location=device)
        model.load_state_dict(state, strict=False)
        model.eval()

        with torch.no_grad():
            z = model.encoder(
                torch.tensor(fusion_raw, dtype=torch.float32, device=device)
            ).cpu().numpy()

        out_dir = Path("results/backtest/fusion")
        out_dir.mkdir(parents=True, exist_ok=True)

        out_path = out_dir / f"fusion_embeddings_backtest_v2_{part}.npy"
        np.save(out_path, z)

        print(f"[OK] {out_path} | shape={z.shape}")


if __name__ == "__main__":
    main()
