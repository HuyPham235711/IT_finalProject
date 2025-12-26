import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path

# ============================================================
# Config
# ============================================================
FUSED_EMB_PATH = Path(
    "results/fusion_rl/v2/fusion_embeddings_train_v2.npy"   # chỉnh đúng path của bạn
)

SAVE_DIR = Path("results/fusion_rl/v2/analysis_plots")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# Load embedding
# ============================================================
print("[INFO] Loading fusion embeddings...")
Z = np.load(FUSED_EMB_PATH)
print(f"[INFO] Embedding shape: {Z.shape}")

# Optional: standardize before PCA
Z_std = StandardScaler().fit_transform(Z)

# ============================================================
# 1. PCA Visualization (2D)
# ============================================================
print("[INFO] Running PCA...")
pca = PCA(n_components=2)
Z_pca = pca.fit_transform(Z_std)

plt.figure(figsize=(8, 6))
plt.scatter(
    Z_pca[:, 0],
    Z_pca[:, 1],
    s=8,
    alpha=0.6
)
plt.title("PCA of Fusion Embeddings")
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.2f}%)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.2f}%)")
plt.grid(alpha=0.3)

pca_path = SAVE_DIR / "fusion_pca.png"
plt.tight_layout()
plt.savefig(pca_path, dpi=200)
plt.close()

print(f" Saved PCA plot → {pca_path}")

# ============================================================
# 2. Explained Variance Bar Chart
# ============================================================
plt.figure(figsize=(6, 4))
plt.bar(
    ["PC1", "PC2"],
    pca.explained_variance_ratio_ * 100
)
plt.ylabel("Explained Variance (%)")
plt.title("Explained Variance of Fusion Embeddings")
plt.grid(axis="y", alpha=0.3)

var_path = SAVE_DIR / "fusion_pca_variance.png"
plt.tight_layout()
plt.savefig(var_path, dpi=200)
plt.close()

print(f" Saved variance plot → {var_path}")

# ============================================================
# 3. Cosine Similarity over Time (Regime Stability)
# ============================================================
print("[INFO] Computing cosine similarity...")
cos_sim = [
    cosine_similarity(
        Z[i].reshape(1, -1),
        Z[i - 1].reshape(1, -1)
    )[0, 0]
    for i in range(1, len(Z))
]

plt.figure(figsize=(10, 4))
plt.plot(cos_sim, linewidth=1.2)
plt.title("Cosine Similarity Between Consecutive Fusion Embeddings")
plt.xlabel("Time Step")
plt.ylabel("Cosine Similarity")
plt.grid(alpha=0.3)

cos_path = SAVE_DIR / "fusion_cosine_similarity.png"
plt.tight_layout()
plt.savefig(cos_path, dpi=200)
plt.close()

print(f"Saved cosine similarity plot → {cos_path}")

print("\n=== DONE: Fusion analysis plots generated ===")
