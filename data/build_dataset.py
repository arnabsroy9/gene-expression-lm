"""
Join NCBI sequences with GTEx expression labels -> labeled_genes.csv

Pipeline:
  1. Load ncbi_sequences.csv       (gene_symbol, sequence)
  2. Load gtex_whole_blood.csv     (gene_id [Ensembl], median_tpm)
  3. Load ensembl_to_symbol.csv    (ensembl_id -> gene_symbol)
  4. Join on gene_symbol
  5. Log-transform TPM -> percentile bins -> High / Medium / Low
  6. Save labeled_genes.csv

Outputs: data/labeled_genes.csv
Columns: gene_id, gene_symbol, sequence, tpm, log_tpm, expression_label
"""

import os
import numpy as np
import pandas as pd

DATA_DIR     = os.path.dirname(__file__)
NCBI_PATH    = os.path.join(DATA_DIR, "ncbi_sequences.csv")
GTEX_PATH    = os.path.join(DATA_DIR, "gtex_expression.csv")
MAP_PATH     = os.path.join(DATA_DIR, "ensembl_to_symbol.csv")
OUT_PATH     = os.path.join(DATA_DIR, "labeled_genes.csv")

LOW_PCT      = 33
HIGH_PCT     = 67
LOG_OFFSET   = 1.0   # log(tpm + 1) to handle zeros


def _bin_labels(log_tpm: pd.Series) -> pd.Series:
    # Rank-based split avoids duplicate bin edges when many genes share TPM=0
    n     = len(log_tpm)
    ranks = log_tpm.rank(method="first", ascending=True)
    result = pd.Series("Medium", index=log_tpm.index, dtype=object)
    result[ranks <= n * LOW_PCT / 100]  = "Low"
    result[ranks >  n * HIGH_PCT / 100] = "High"
    return result


def build(
    ncbi_path: str = NCBI_PATH,
    gtex_path: str = GTEX_PATH,
    map_path:  str = MAP_PATH,
    out_path:  str = OUT_PATH,
) -> pd.DataFrame:

    ncbi = pd.read_csv(ncbi_path)
    gtex = pd.read_csv(gtex_path)
    sym_map = pd.read_csv(map_path)

    # Normalise gene symbols to uppercase for robust matching
    for df, col in [(ncbi, "gene_symbol"), (sym_map, "gene_symbol")]:
        df[col] = df[col].astype(str).str.strip().str.upper()

    # Rename GTEx gene_id to ensembl_id to avoid collision with NCBI gene_id after merge
    gtex = gtex.rename(columns={"gene_id": "ensembl_id"})
    gtex = gtex.merge(sym_map, on="ensembl_id", how="inner")
    gtex["gene_symbol"] = gtex["gene_symbol"].str.upper()

    # Keep highest TPM per symbol (some Ensembl IDs map to same symbol)
    gtex = gtex.sort_values("median_tpm", ascending=False).drop_duplicates("gene_symbol")

    merged = ncbi.merge(gtex[["gene_symbol", "median_tpm"]], on="gene_symbol", how="inner")
    merged = merged.dropna(subset=["median_tpm", "sequence"])
    merged["log_tpm"] = np.log(merged["median_tpm"].clip(lower=0) + LOG_OFFSET)

    merged["expression_label"] = _bin_labels(merged["log_tpm"])
    merged = merged.dropna(subset=["expression_label"]).reset_index(drop=True)

    print(f"Final dataset: {len(merged)} genes")
    print(merged["expression_label"].value_counts().to_string())

    merged = merged[["gene_id", "gene_symbol", "sequence", "median_tpm", "log_tpm", "expression_label"]]
    merged.to_csv(out_path, index=False)
    print(f"Saved -> {out_path}")
    return merged


if __name__ == "__main__":
    build()