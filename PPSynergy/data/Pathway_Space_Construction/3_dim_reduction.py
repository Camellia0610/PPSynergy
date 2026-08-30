"""
Dimensionality reduction (PCA / t-SNE / UMAP) of the ssGSEA pathway activity matrix and visualization
- Replace {CANCER} with your cancer type (e.g. LIHC, COAD) before running.
- Input: TCGA-{CANCER}_ssGSEA_scores.csv (pathways x samples)
- Note: We reduce the dimensions of the PATHWAYS -- each pathway is a vector across the samples
- Output: Pathway reduction coordinate CSV + visualization images + PCA explained variance
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

import umap
import time

t0 = time.time()

# ============================
# 1. Load data (pathways x samples)
# ============================
print("1. Loading ssGSEA score matrix...")
ss = pd.read_csv(r'e:\PPSynergy\TCGA-{CANCER}_ssGSEA_scores.csv', index_col=0)
print(f"   Raw: {ss.shape[0]} pathways x {ss.shape[1]} samples")

# Each PATHWAY is a high-dimensional vector (its activity pattern across samples)
X = ss  # shape: pathways x samples
print(f"   Reducing pathways: {X.shape[0]} pathways x {X.shape[1]} samples (each pathway is a high-dimensional vector)")

# Standardize by sample (z-score of each pathway's activity across samples)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ============================
# 2. PCA dimensionality reduction + explained variance
# ============================
print("\n2. PCA dimensionality reduction...")
pca = PCA(n_components=2, random_state=42)
pca_coords = pca.fit_transform(X_scaled)

evr = pca.explained_variance_ratio_
print(f"   PC1 explained variance: {evr[0]*100:.2f}%")
print(f"   PC2 explained variance: {evr[1]*100:.2f}%")
print(f"   PC1+PC2 cumulative explained variance: {evr.sum()*100:.2f}%")

# Save PCA explained variance to text
with open(r'e:\PPSynergy\PCA_variance.txt', 'w', encoding='utf-8') as f:
    f.write("PCA Explained Variance Ratio\n")
    f.write("=" * 50 + "\n")
    for i, v in enumerate(evr):
        f.write(f"PC{i+1}: {v*100:.4f}%\n")
    f.write(f"PC1+PC2 cumulative: {evr.sum()*100:.4f}%\n")
print(f"   Explained variance saved: PCA_variance.txt")

# ============================
# 3. t-SNE dimensionality reduction
# ============================
print("\n3. t-SNE dimensionality reduction...")
tsne = TSNE(n_components=2, random_state=42, perplexity=30, init='pca', learning_rate='auto')
tsne_coords = tsne.fit_transform(X_scaled)

# ============================
# 4. UMAP dimensionality reduction
# ============================
print("4. UMAP dimensionality reduction...")
reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
umap_coords = reducer.fit_transform(X_scaled)

# ============================
# 5. Save the reduced pathway coordinates
# ============================
pathway_names = X.index.tolist()
result_df = pd.DataFrame({
    'pathway': pathway_names,
    'PCA_1': pca_coords[:, 0],
    'PCA_2': pca_coords[:, 1],
    'tSNE_1': tsne_coords[:, 0],
    'tSNE_2': tsne_coords[:, 1],
    'UMAP_1': umap_coords[:, 0],
    'UMAP_2': umap_coords[:, 1],
})
result_df.to_csv(r'e:\PPSynergy\TCGA-{CANCER}_pathway_dim_reduction.csv', index=False)
print("\n5. Reduced pathway coordinates saved: TCGA-{CANCER}_pathway_dim_reduction.csv")

# ============================
# 6. Visualization (pathways)
# ============================
print("6. Generating visualization figures...")
fig, axes = plt.subplots(1, 3, figsize=(21, 6.5))

# PCA
axes[0].scatter(pca_coords[:, 0], pca_coords[:, 1], s=8, alpha=0.6, c='#4C72B0', edgecolors='none')
axes[0].set_title(f'PCA (Pathways)\n(PC1: {evr[0]*100:.2f}%, PC2: {evr[1]*100:.2f}%)', fontsize=13)
axes[0].set_xlabel('PC1')
axes[0].set_ylabel('PC2')
axes[0].grid(True, alpha=0.3)

# t-SNE
axes[1].scatter(tsne_coords[:, 0], tsne_coords[:, 1], s=8, alpha=0.6, c='#55A868', edgecolors='none')
axes[1].set_title('t-SNE (Pathways)\n(perplexity=30)', fontsize=13)
axes[1].set_xlabel('t-SNE 1')
axes[1].set_ylabel('t-SNE 2')
axes[1].grid(True, alpha=0.3)

# UMAP
axes[2].scatter(umap_coords[:, 0], umap_coords[:, 1], s=8, alpha=0.6, c='#C44E52', edgecolors='none')
axes[2].set_title('UMAP (Pathways)\n(n_neighbors=15, min_dist=0.1)', fontsize=13)
axes[2].set_xlabel('UMAP 1')
axes[2].set_ylabel('UMAP 2')
axes[2].grid(True, alpha=0.3)

fig.suptitle('TCGA-{CANCER} ssGSEA Pathway Activity - Pathway Dimensionality Reduction', fontsize=15, y=1.02)
plt.tight_layout()
plt.savefig(r'e:\PPSynergy\TCGA-{CANCER}_pathway_dim_reduction.png', dpi=150, bbox_inches='tight')
print("   Visualization saved: TCGA-{CANCER}_pathway_dim_reduction.png")

# Extra: PCA cumulative variance scree plot
fig2, ax2 = plt.subplots(figsize=(8, 5))
pca_full = PCA(n_components=min(20, X.shape[0])).fit(X_scaled)
cumsum = np.cumsum(pca_full.explained_variance_ratio_)
ax2.plot(range(1, len(cumsum)+1), cumsum, 'o-', color='#4C72B0')
ax2.axhline(y=0.9, color='red', linestyle='--', alpha=0.6, label='90% threshold')
ax2.axhline(y=0.8, color='orange', linestyle='--', alpha=0.6, label='80% threshold')
ax2.set_xlabel('Number of Principal Components')
ax2.set_ylabel('Cumulative Explained Variance')
ax2.set_title('PCA Cumulative Explained Variance (top 20 PCs)')
ax2.legend()
ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(r'e:\PPSynergy\PCA_cumulative_variance.png', dpi=150, bbox_inches='tight')
print(f"   PCA cumulative variance figure saved: PCA_cumulative_variance.png")

print(f"\nDone! Elapsed {time.time()-t0:.1f} seconds")
