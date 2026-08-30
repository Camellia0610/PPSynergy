"""
Convert ENSG gene IDs to gene symbols using mygene
- Replace {CANCER} with your cancer type (e.g. LIHC, COAD) before running.
"""

import pandas as pd
import mygene

# ============================
# 1. Load the cleaned data
# ============================
print("Loading cleaned data...")
df = pd.read_csv(r'e:\PPSynergy\TCGA-{CANCER}_TPM_data_clean.csv')
print(f"Data: {df.shape[0]} genes x {df.shape[1]-1} samples\n")

# ============================
# 2. Extract gene IDs and query
# ============================
gene_ids = df['gene_id'].astype(str).tolist()
print(f"Total {len(gene_ids)} genes to convert...")

# Remove the version suffix (e.g. ENSG00000000003.15 -> ENSG00000000003)
clean_ids = [g.split('.')[0] for g in gene_ids]

# Batch query mygene (up to 1000 per batch)
mg = mygene.MyGeneInfo()
results = []

for i in range(0, len(clean_ids), 1000):
    batch = clean_ids[i:i+1000]
    print(f"  Querying batch {i//1000 + 1} ({len(batch)} genes)...")
    res = mg.querymany(batch, scopes='ensembl.gene', fields='symbol', species='human', verbose=False)
    results.extend(res)

# ============================
# 3. Organize query results
# ============================
id_to_symbol = {}
for r in results:
    qid = r.get('query')
    if qid and qid not in id_to_symbol:
        if 'notfound' in r:
            id_to_symbol[qid] = None  # Not found
        else:
            sym = r.get('symbol')
            id_to_symbol[qid] = sym if sym else None

# Map back to the original gene_id
df['gene_symbol'] = [id_to_symbol.get(g.split('.')[0], None) for g in gene_ids]

# Statistics
not_found = df['gene_symbol'].isna().sum()
found = df['gene_symbol'].notna().sum()
print(f"\nConverted successfully: {found}")
print(f"Not found: {not_found}")
if not_found > 0:
    print(f"  Examples of not-found genes: {df.loc[df['gene_symbol'].isna(), 'gene_id'].head(10).tolist()}")

# ============================
# 4. Preview results & save
# ============================
print(f"\nConversion result preview:")
print(df[['gene_id', 'gene_symbol']].head(10).to_string(index=False))

# Put gene_symbol as the first column
cols = ['gene_symbol', 'gene_id'] + [c for c in df.columns if c not in ('gene_symbol', 'gene_id')]
df_out = df[cols]

output_path = r'e:\PPSynergy\TCGA-{CANCER}_TPM_data_symbol.csv'
df_out.to_csv(output_path, index=False)
print(f"\nSaved to: {output_path}")
print(f"Final data: {df_out.shape[0]} genes x {df_out.shape[1]-2} samples (+ gene_symbol, gene_id)")
