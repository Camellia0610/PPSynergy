"""
Compute per-sample ssGSEA pathway activity scores
- Replace {CANCER} with your cancer type (e.g. LIHC, COAD) before running.
- Input: TCGA-{CANCER}_TPM_data_symbol.csv (gene_symbol x tumor samples)
- Input: c2.cp.v2023.1.Hs.symbols.gmt (pathway gene sets)
- Output: TCGA-{CANCER}_ssGSEA_scores.csv (pathway x sample matrix)
"""

import pandas as pd
import gseapy as gp
import time

t0 = time.time()

# ============================
# 1. Load expression data and preprocess
# ============================
print("1. Loading expression data...")
expr = pd.read_csv(r'e:\PPSynergy\TCGA-LIHC_TPM_data_symbol.csv')
print(f"   Raw: {expr.shape[0]} genes x {expr.shape[1]-2} samples")

# Drop genes with an empty gene_symbol
expr = expr.dropna(subset=['gene_symbol'])
print(f"   After dropping empty gene symbols: {expr.shape[0]} genes")

# Handle duplicated gene symbols (merge by mean) to guarantee a unique index
if expr['gene_symbol'].duplicated().any():
    print(f"   Found {expr['gene_symbol'].duplicated().sum()} duplicated gene symbols, merging by mean...")
    sample_cols = [c for c in expr.columns if c.startswith('TCGA-')]
    expr = expr.groupby('gene_symbol')[sample_cols].mean().reset_index()
    print(f"   After merging: {expr.shape[0]} genes")

# Set gene_symbol as the index to form a genes x samples matrix
expr = expr.set_index('gene_symbol')
expr.index.name = None
print(f"   Expression matrix: {expr.shape[0]} genes x {expr.shape[1]} samples")

# ============================
# 2. Load the GMT pathway file
# ============================
print("\n2. Loading GMT pathway file...")
gmt_path = r'e:\PPSynergy\通路对应基因c2.cp.v2023.1.Hs.symbols.gmt'
gene_sets = {}
with open(gmt_path, 'r', encoding='utf-8', errors='ignore') as f:
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) >= 3:
            term = parts[0]
            genes = parts[2:]  # Skip pathway name and description
            gene_sets[term] = genes

print(f"   Total {len(gene_sets)} pathways")

# ============================
# 3. Compute ssGSEA scores
# ============================
print("\n3. Computing ssGSEA scores (samples x pathways, please wait)...")
res = gp.ssgsea(
    data=expr,
    gene_sets=gene_sets,
    outdir=None,
    min_size=5,          # Minimum number of genes per pathway
    max_size=5000,       # Maximum number of genes per pathway
    sample_norm_method='rank',
    no_plot=True
)

# Convert to long format: Name (sample), Term (pathway), ES (enrichment score), NES (normalized score)
res2d = res.res2d
print(f"   Result rows (pathway x sample): {len(res2d)}")
print(f"   Result columns: {list(res2d.columns)}")

# Pivot into a pathway x sample matrix (using the ES enrichment score)
ss_df = res2d.pivot(index='Term', columns='Name', values='ES')
print(f"   Pathway activity matrix: {ss_df.shape[0]} pathways x {ss_df.shape[1]} samples")

# ============================
# 4. Save results
# ============================
out_path = r'e:\PPSynergy\TCGA-{CANCER}_ssGSEA_scores.csv'
ss_df.to_csv(out_path)
print(f"\n4. Saved to: {out_path}")

# Also save the long format (one row per: pathway, sample, ES, NES)
long_path = r'e:\PPSynergy\TCGA-{CANCER}_ssGSEA_scores_long.csv'
res2d.rename(columns={'Term': 'pathway', 'Name': 'sample'}) \
     .to_csv(long_path, index=False)
print(f"   Long format saved: {long_path}")

print(f"\nDone! Elapsed {time.time()-t0:.1f} seconds")
print(f"Example (first 5 pathways x first 3 samples):")
print(ss_df.iloc[:5, :3].round(4).to_string())
