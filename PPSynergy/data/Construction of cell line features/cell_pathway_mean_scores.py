# -*- coding: utf-8 -*-
"""
Compute the activity score of each pathway in each cell line from a gene
expression matrix, using the mean method.

Usage:
  python cell_pathway_mean_scores.py --expr <expr.csv> --pathways <pathways.csv> -o <out.csv>

Inputs:
  --expr       Gene expression matrix (CSV, supplied via --expr)
               Format: header row first; column 1 = gene symbol (uppercase),
               remaining columns = cell lines; values are TPM expression.
               e.g. gene,CellLine1,CellLine2,...
  --pathways   Pathway list (CSV, supplied via --pathways)
               Format: header row first; column 1 = pathway name, one pathway
               per row (any number of pathways, as needed).
  MSigDB.gmt   Pathway-gene mapping (GMT), fixed file located next to this script.
               Format: tab-separated; column 1 = pathway name, column 2 =
               description, remaining columns = member gene symbols.

Method: for each pathway, take the mean expression of its member genes
(present in the expression matrix) across all cell lines.

Output:
  -o/--output  CSV with columns [pathway, <cell lines>]; each cell = mean
               expression of the pathway's genes in that cell line,
               rounded to 4 decimals.
"""
import os
import csv
import argparse

BASE = os.path.dirname(os.path.abspath(__file__))
GMT = os.path.join(BASE, "MSigDB.gmt")  # fixed pathway-gene mapping (GMT)


def main():
    parser = argparse.ArgumentParser(
        description="Compute per-cell-line pathway activity scores from a gene expression matrix."
    )
    parser.add_argument("--expr", required=True,
                        help="gene expression matrix CSV: gene symbol x cell lines (TPM)")
    parser.add_argument("--pathways", required=True,
                        help="pathway list CSV: first column = pathway names")
    parser.add_argument("-o", "--output", required=True,
                        help="output CSV: pathway + cell lines, rounded to 4 decimals")
    args = parser.parse_args()
    # 1. Read expression matrix: gene -> [values per cell line]
    with open(args.expr, encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        cell_lines = header[1:]
        expr = {}
        for row in reader:
            if not row or not row[0]:
                continue
            expr[row[0].strip().upper()] = [float(x) for x in row[1:]]
    n_cl = len(cell_lines)
    print(f"Number of expressed genes: {len(expr)}, cell lines ({n_cl}): {cell_lines}")

    # 2. GMT: pathway name -> gene list
    pw_genes = {}
    with open(GMT, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2:
                pw_genes[parts[0]] = [g.upper() for g in parts[2:]]
    print(f"Number of pathways in GMT: {len(pw_genes)}")

    # 3. Pathway list
    with open(args.pathways, encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)
        pathways = [row[0] for row in reader if row]
    print(f"Number of pathways in unique_pathways: {len(pathways)}")

    # 4. Compute mean scores
    out = []
    n_used = 0
    n_avg_genes = []
    for pw in pathways:
        genes = pw_genes.get(pw, [])
        present = [g for g in genes if g in expr]
        if not present:
            out.append([pw] + [""] * n_cl)
            continue
        n_used += 1
        n_avg_genes.append(len(present))
        n = len(present)
        means = [round(sum(expr[g][i] for g in present) / n, 4) for i in range(n_cl)]
        out.append([pw] + [f"{m:.4f}" for m in means])

    with open(args.output, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pathway"] + cell_lines)
        w.writerows(out)

    if n_avg_genes:
        print(f"Successfully computed: {n_used} / {len(pathways)} pathways")
        print(f"Genes averaged per pathway: {min(n_avg_genes)} ~ {max(n_avg_genes)}")
    print(f"Output written to: {args.output}")


if __name__ == "__main__":
    main()
