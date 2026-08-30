"""
UMAP pathway coordinate repulsion refinement
- Replace {CANCER} with your cancer type (e.g. LIHC, COAD) before running.
- Input: TCGA-{CANCER}_pathway_dim_reduction.csv (UMAP_1, UMAP_2)
- Only repulsion refinement, no Y-axis stretching
- Output: all results in the repulsion_umap/ directory
"""

import pandas as pd
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT_DIR = r'e:\PPSynergy\repulsion_umap'
os.makedirs(OUT_DIR, exist_ok=True)

# ============================
# 1. Load UMAP coordinates
# ============================
print("1. Loading UMAP coordinates...")
df = pd.read_csv(r'e:\PPSynergy\TCGA-{CANCER}_pathway_dim_reduction.csv')
coords = df[['UMAP_1', 'UMAP_2']].values.astype(float)
names = df['pathway'].values
n = len(coords)
print(f"   Total {n} pathways")

# ============================
# 2. Repulsion refinement (core)
# ============================
def repulsion_refinement(X, iterations=6, lr=0.005, eps=1e-6, decay=0.95):
    """
    Pure repulsion model: all points repel each other, keep the original scale, only fine-tune
    force = diff / dist2  (direction outward, magnitude proportional to 1/d)
    """
    X = X.copy()
    for it in range(iterations):
        grad = np.zeros_like(X)
        for i in range(len(X)):
            diff = X[i] - X
            dist2 = np.sum(diff**2, axis=1) + eps
            force = diff / dist2[:, None]
            force[i] = 0
            grad[i] = np.sum(force, axis=0)
        X = X + lr * grad
        lr *= decay
    return X

print("\n2. Running repulsion refinement (iterations=6, lr=0.005, decay=0.95)...")
coords_refined = repulsion_refinement(coords)

# ============================
# 3. Save results
# ============================
print("\n3. Saving results...")
out = pd.DataFrame({
    "pathway": names,
    "UMAP_1_refined": coords_refined[:, 0],
    "UMAP_2_refined": coords_refined[:, 1],
    "UMAP_1_original": coords[:, 0],
    "UMAP_2_original": coords[:, 1],
})
out.to_csv(os.path.join(OUT_DIR, 'refined_umap.csv'), index=False)
print(f"   Saved: {OUT_DIR}\\refined_umap.csv")

# ============================
# 4. Range comparison
# ============================
print("\n4. Range comparison (original vs refined):")
print(f"   X: [{coords[:,0].min():.4f}, {coords[:,0].max():.4f}]  ->  "
      f"[{coords_refined[:,0].min():.4f}, {coords_refined[:,0].max():.4f}]")
print(f"   Y: [{coords[:,1].min():.4f}, {coords[:,1].max():.4f}]  ->  "
      f"[{coords_refined[:,1].min():.4f}, {coords_refined[:,1].max():.4f}]")
print(f"   X span: {coords[:,0].max()-coords[:,0].min():.4f} -> "
      f"{coords_refined[:,0].max()-coords_refined[:,0].min():.4f}")
print(f"   Y span: {coords[:,1].max()-coords[:,1].min():.4f} -> "
      f"{coords_refined[:,1].max()-coords_refined[:,1].min():.4f}")

# ============================
# 5. Pixel overlap analysis (128x128 & 64x128)
# ============================
def analyze_overlap(X, res_x, res_y):
    """Count pixel overlap at the given resolution"""
    def to_pixel(vals, res):
        vmin, vmax = vals.min(), vals.max()
        span = (vmax - vmin) if vmax > vmin else 1.0
        px = ((vals - vmin) / span * (res - 1)).astype(int)
        return np.clip(px, 0, res - 1)

    px = to_pixel(X[:, 0], res_x)
    py = to_pixel(X[:, 1], res_y)
    grid = np.zeros((res_y, res_x), dtype=int)
    for i in range(len(X)):
        grid[py[i], px[i]] += 1

    nonempty = np.count_nonzero(grid)
    overlap = np.count_nonzero(grid >= 2)
    max_overlap = grid.max()
    single_px = np.count_nonzero(grid == 1)
    return {
        'res': f'{res_x}x{res_y}',
        'nonempty': nonempty,
        'overlap': overlap,
        'max_overlap': max_overlap,
        'single_ratio': single_px / nonempty if nonempty else 0,
    }

print("\n5. Pixel overlap comparison (before vs after refinement):")
stats_rows = []
for rx, ry in [(128, 128), (64, 128)]:
    before = analyze_overlap(coords, rx, ry)
    after = analyze_overlap(coords_refined, rx, ry)
    stats_rows.append({
        'resolution': before['res'],
        'before_nonempty': before['nonempty'],
        'after_nonempty': after['nonempty'],
        'before_overlap': before['overlap'],
        'after_overlap': after['overlap'],
        'before_max': before['max_overlap'],
        'after_max': after['max_overlap'],
        'before_single_ratio': round(before['single_ratio'], 4),
        'after_single_ratio': round(after['single_ratio'], 4),
    })
    print(f"   {before['res']}: overlapping pixels {before['overlap']} -> {after['overlap']}  |  "
          f"max overlap {before['max_overlap']} -> {after['max_overlap']}  |  "
          f"non-overlap ratio {before['single_ratio']*100:.1f}% -> {after['single_ratio']*100:.1f}%")

pd.DataFrame(stats_rows).to_csv(os.path.join(OUT_DIR, 'overlap_comparison.csv'), index=False)
print(f"   Comparison table saved: {OUT_DIR}\\overlap_comparison.csv")

# ============================
# 6. Visualization comparison
# ============================
print("\n6. Generating visualizations...")

# Figure 1: scatter comparison before/after refinement
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
axes[0].scatter(coords[:, 0], coords[:, 1], s=8, alpha=0.6, c='#4C72B0')
axes[0].set_title('Before Repulsion Refinement')
axes[0].set_xlabel('UMAP 1'); axes[0].set_ylabel('UMAP 2')
axes[0].grid(True, alpha=0.3)
axes[1].scatter(coords_refined[:, 0], coords_refined[:, 1], s=8, alpha=0.6, c='#55A868')
axes[1].set_title('After Repulsion Refinement')
axes[1].set_xlabel('UMAP 1'); axes[1].set_ylabel('UMAP 2')
axes[1].grid(True, alpha=0.3)
fig.suptitle(f'UMAP Pathway Coordinates - Repulsion Refinement ({n} pathways)', fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'before_after_scatter.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"   Saved: before_after_scatter.png")

# Figure 2: 128x128 overlap heatmap comparison before/after refinement
fig2, axes2 = plt.subplots(1, 2, figsize=(14, 6))

def draw_heatmap(ax, X, title, res_x=128, res_y=128):
    def to_pixel(vals, res):
        vmin, vmax = vals.min(), vals.max()
        span = (vmax - vmin) if vmax > vmin else 1.0
        return np.clip(((vals - vmin) / span * (res - 1)).astype(int), 0, res - 1)
    px = to_pixel(X[:, 0], res_x); py = to_pixel(X[:, 1], res_y)
    grid = np.zeros((res_y, res_x), dtype=int)
    for i in range(len(X)):
        grid[py[i], px[i]] += 1
    im = ax.imshow(grid, cmap='viridis', interpolation='nearest', origin='lower')
    ax.set_title(title)
    ax.set_xlabel('X (pixel)'); ax.set_ylabel('Y (pixel)')
    return im

im1 = draw_heatmap(axes2[0], coords, 'Before (128x128)')
im2 = draw_heatmap(axes2[1], coords_refined, 'After (128x128)')
fig2.colorbar(im1, ax=axes2[0]); fig2.colorbar(im2, ax=axes2[1])
fig2.suptitle('Pathway Overlap Heatmap - Before vs After Repulsion', fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'before_after_heatmap.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"   Saved: before_after_heatmap.png")

print(f"\nDone! All results saved to: {OUT_DIR}")
