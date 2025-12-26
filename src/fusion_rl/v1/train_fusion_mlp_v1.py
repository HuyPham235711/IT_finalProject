import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import yaml, os
from pathlib import Path

# ============================================================
#  Fusion MLP Model (autoencoder-like)
# ============================================================
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


# ============================================================
#  Main Training Function
# ============================================================
def main():
    with open("config/fusion_rl/fusion_config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # === Load embeddings ===
    paths = cfg["sources"]
    print("[INFO] Loading embeddings...")
    emb_transformer = np.load(paths["transformer_embeddings"])
    emb_attcnn = np.load(paths["attcnn_embeddings"])
    emb_finbert = np.load(paths["finbert_embeddings"])

    # === Check alignment ===
    min_len = min(len(emb_transformer), len(emb_attcnn), len(emb_finbert))
    emb_transformer = emb_transformer[-min_len:]
    emb_attcnn = emb_attcnn[-min_len:]
    emb_finbert = emb_finbert[-min_len:]

    X = np.concatenate([emb_transformer, emb_attcnn, emb_finbert], axis=1)
    print(f"[INFO] Fusion input shape: {X.shape}")

    X_tensor = torch.tensor(X, dtype=torch.float32)
    dataset = TensorDataset(X_tensor)
    loader = DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=True)

    # === Model setup ===
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FusionMLP(
        input_dim=cfg["input_dim"],
        hidden_dim=cfg["hidden_dim"],
        output_dim=cfg["output_dim"],
        dropout=cfg["dropout"]
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg["lr"]))

    # === Training loop ===
    print("[INFO] Start training Fusion MLP...")
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        total_loss = 0.0
        for (xb,) in loader:
            xb = xb.to(device)
            optimizer.zero_grad()
            recon, z = model(xb)
            loss = criterion(recon, xb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
        epoch_loss = total_loss / len(loader.dataset)
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch}/{cfg['epochs']} - Loss: {epoch_loss:.6f}")

    # === Save model & embeddings ===
    Path(cfg["save_path"]).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), cfg["save_path"])

    model.eval()
    with torch.no_grad():
        _, fused_z = model(torch.tensor(X, dtype=torch.float32).to(device))
    fused_z = fused_z.cpu().numpy()

    out_path = Path(cfg["save_path"]).parent / "fusion_embeddings.npy"
    np.save(out_path, fused_z)

    print(f"✅ Saved checkpoint → {cfg['save_path']}")
    print(f"✅ Saved fused embeddings → {out_path} | shape={fused_z.shape}")


if __name__ == "__main__":
    main()
