import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import yaml
from pathlib import Path


# ============================================================
# Fusion MLP v3 
# ============================================================
class FusionMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, dropout=0.2, layer_norm=True):
        super().__init__()

        # -------- Encoder --------
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            if layer_norm:
                layers.append(nn.LayerNorm(h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h

        layers.append(nn.Linear(prev_dim, output_dim))
        self.encoder = nn.Sequential(*layers)

        # -------- Decoder (regularization) --------
        dec_layers = []
        prev_dim = output_dim
        for h in reversed(hidden_dims):
            dec_layers.append(nn.Linear(prev_dim, h))
            dec_layers.append(nn.ReLU())
            prev_dim = h
        dec_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*dec_layers)

    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon, z


# ============================================================
# Main
# ============================================================
def main():

    # --------------------------------------------------------
    # Load config
    # --------------------------------------------------------
    cfg_path = Path("config/fusion_rl/fusion_config_v2.yaml")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    src_cfg = cfg["sources"]
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]
    out_cfg = cfg["output"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --------------------------------------------------------
    # Load embeddings
    # --------------------------------------------------------
    print("[INFO] Loading embedding sources...")

    emb_att = np.load(src_cfg["attcnn_embeddings"]["path"])
    emb_trans = np.load(src_cfg["transformer_embeddings"]["path"])
    emb_fin = np.load(src_cfg["finbert_embeddings"]["path"])

    # Align length (IMPORTANT)
    min_len = min(len(emb_att), len(emb_trans), len(emb_fin))
    emb_att = emb_att[-min_len:]
    emb_trans = emb_trans[-min_len:]
    emb_fin = emb_fin[-min_len:]

    # Validate dims
    assert emb_att.shape[1] == src_cfg["attcnn_embeddings"]["dim"]
    assert emb_trans.shape[1] == src_cfg["transformer_embeddings"]["dim"]
    assert emb_fin.shape[1] == src_cfg["finbert_embeddings"]["dim"]

    X = np.concatenate([emb_att, emb_trans, emb_fin], axis=1)
    print(f"[INFO] Fusion input shape = {X.shape}")

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------
    X_tensor = torch.tensor(X, dtype=torch.float32)
    loader = DataLoader(
        TensorDataset(X_tensor),
        batch_size=train_cfg["batch_size"],
        shuffle=train_cfg["shuffle"],
        num_workers=train_cfg["num_workers"],
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------
    model = FusionMLP(
        input_dim=model_cfg["input_dim"],
        hidden_dims=model_cfg["hidden_dims"],
        output_dim=model_cfg["output_dim"],
        dropout=model_cfg["dropout"],
        layer_norm=model_cfg["layer_norm"],
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=train_cfg["lr"],
        weight_decay=train_cfg["weight_decay"],
    )
    criterion = nn.MSELoss()

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------
    print("[INFO] Training Fusion MLP v3...")
    for epoch in range(1, train_cfg["epochs"] + 1):
        model.train()
        total_loss = 0.0

        for (xb,) in loader:
            xb = xb.to(device)
            optimizer.zero_grad()
            recon, _ = model(xb)
            loss = criterion(recon, xb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)

        avg_loss = total_loss / len(loader.dataset)
        if epoch == 1 or epoch % 10 == 0:
            print(f"Epoch {epoch}/{train_cfg['epochs']} | Loss = {avg_loss:.6f}")

    # --------------------------------------------------------
    # Save outputs
    # --------------------------------------------------------
    out_dir = Path(out_cfg["save_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = out_dir / out_cfg["checkpoint"]
    torch.save(model.state_dict(), ckpt_path)

    model.eval()
    with torch.no_grad():
        _, fused_z = model(X_tensor.to(device))
    fused_z = fused_z.cpu().numpy()

    emb_path = out_dir / out_cfg["embeddings_train"]
    np.save(emb_path, fused_z)

    print(f" Saved checkpoint → {ckpt_path}")
    print(f" Saved fused embeddings → {emb_path} | shape={fused_z.shape}")


if __name__ == "__main__":
    main()
