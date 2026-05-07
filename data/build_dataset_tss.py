"""
Join TSS-centred sequences with GTEx expression labels → labeled_genes_tss.csv

Pipeline:
  1. Load tss_sequences.csv    (ensembl_id, gene_symbol, chromosome, tss, strand, sequence)
  2. Load gtex_expression.csv  (gene_id [Ensembl], median_tpm)
  3. Join on Ensembl ID
  4. Log-transform TPM → rank-percentile bins → Low / Medium / High
  5. Save labeled_genes_tss.csv

Uses the same label-binning logic as build_dataset.py so results are comparable.
Existing files (labeled_genes.csv, ncbi_sequences.csv, etc.) are not modified.

Outputs: data/labeled_genes_tss.csv
Columns: ensembl_id, gene_symbol, chromosome, tss, strand,
         sequence, median_tpm, log_tpm, expression_label

Usage:
    python data/build_dataset_tss.py
    python data/build_dataset_tss.py --tss data/tss_sequences.csv --out data/labeled_genes_tss.csv
"""

import os
import argparse
import numpy as np
import pandas as pd

DATA_DIR   = os.path.dirname(__file__)
TSS_PATH   = os.path.join(DATA_DIR, "tss_sequences.csv")
GTEX_PATH  = os.path.join(DATA_DIR, "gtex_expression.csv")
OUT_PATH   = os.path.join(DATA_DIR, "labeled_genes_tss.csv")

LOW_PCT    = 33
HIGH_PCT   = 67
LOG_OFFSET = 1.0    # log(tpm + 1) to handle zeros, matches build_dataset.py


def _bin_labels(log_tpm: pd.Series) -> pd.Series:
    # Rank-based split: avoids duplicate bin edges when many genes share TPM=0
    n     = len(log_tpm)
    ranks = log_tpm.rank(method="first", ascending=True)
    result = pd.Series("Medium", index=log_tpm.index, dtype=object)
    result[ranks <= n * LOW_PCT  / 100] = "Low"
    result[ranks >  n * HIGH_PCT / 100] = "High"
    return result


def build(
    tss_path:  str = TSS_PATH,
    gtex_path: str = GTEX_PATH,
    out_path:  str = OUT_PATH,
) -> pd.DataFrame:

    tss  = pd.read_csv(tss_path)
    gtex = pd.read_csv(gtex_path)

    print(f"TSS sequences loaded:  {len(tss):,} genes")
    print(f"GTEx expression rows:  {len(gtex):,}")

    # Strip Ensembl version suffixes (ENSG00000001.5 → ENSG00000001) for robust join
    tss["ensembl_id"]  = tss["ensembl_id"].astype(str).str.split(".").str[0]
    gtex["ensembl_id"] = gtex["gene_id"].astype(str).str.split(".").str[0]

    # Keep highest TPM per Ensembl ID (duplicate Ensembl entries are rare but possible)
    gtex = gtex.sort_values("median_tpm", ascending=False).drop_duplicates("ensembl_id")

    merged = tss.merge(gtex[["ensembl_id", "median_tpm"]], on="ensembl_id", how="inner")
    merged = merged.dropna(subset=["median_tpm", "sequence"])

    merged["log_tpm"]          = np.log(merged["median_tpm"].clip(lower=0) + LOG_OFFSET)
    merged["expression_label"] = _bin_labels(merged["log_tpm"])
    merged = merged.dropna(subset=["expression_label"]).reset_index(drop=True)

    print(f"\nFinal dataset: {len(merged):,} genes")
    print(merged["expression_label"].value_counts().to_string())
    print(f"\nSequence length stats (bp):")
    lens = merged["sequence"].str.len()
    print(f"  min={lens.min()}  median={lens.median():.0f}  max={lens.max()}")

    cols = [
        "ensembl_id", "gene_symbol", "chromosome", "tss", "strand",
        "sequence", "median_tpm", "log_tpm", "expression_label",
    ]
    merged[cols].to_csv(out_path, index=False)
    print(f"\nSaved -> {out_path}")
    return merged


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--tss",  default=TSS_PATH,  help="Path to tss_sequences.csv")
    p.add_argument("--gtex", default=GTEX_PATH, help="Path to gtex_expression.csv")
    p.add_argument("--out",  default=OUT_PATH,  help="Output CSV path")
    args = p.parse_args()
    build(tss_path=args.tss, gtex_path=args.gtex, out_path=args.out)
