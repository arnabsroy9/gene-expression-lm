"""
Build a CLEAN labeled dataset by stacking multiple noise-reducing filters.

Filters applied (in order):
  1. Protein-coding genes only          (drops pseudogenes, lncRNAs, miRNAs, ...)
  2. Optional min_tpm filter            (drops genes with near-zero expression everywhere)
  3. Strict percentile cutoffs          (bottom LOW_PCT% = Low, top HIGH_PCT% = High)
  4. Binary classification              (drops the middle class entirely)

Inputs (defaults):
  - data/tss_sequences.csv                  TSS-centred sequences
  - data/gene_biotypes.csv                  ensembl_id -> biotype
  - data/gtex_expression_all_tissues.csv    median TPM across all GTEx tissues

Output: data/labeled_genes_tss_clean.csv
Columns: ensembl_id, gene_symbol, chromosome, tss, strand,
         sequence, median_tpm, log_tpm, expression_label
Labels:  "Low" or "High" only (binary)

Usage:
    python data/build_dataset_clean.py
    python data/build_dataset_clean.py --min_tpm 0.5
    python data/build_dataset_clean.py --low_pct 25 --high_pct 75   # tighter middle band
    python data/build_dataset_clean.py --gtex data/gtex_expression.csv  # use single-tissue file
"""

import os
import argparse
import numpy as np
import pandas as pd

DATA_DIR     = os.path.dirname(__file__)
TSS_PATH     = os.path.join(DATA_DIR, "tss_sequences.csv")
BIOTYPE_PATH = os.path.join(DATA_DIR, "gene_biotypes.csv")
GTEX_PATH    = os.path.join(DATA_DIR, "gtex_expression_all_tissues.csv")
OUT_PATH     = os.path.join(DATA_DIR, "labeled_genes_tss_clean.csv")

DEFAULT_MIN_TPM  = 0.1   # very permissive — drops only genes with near-zero expression everywhere
DEFAULT_LOW_PCT  = 30    # bottom 30% of remaining genes labelled "Low"
DEFAULT_HIGH_PCT = 70    # top 30% labelled "High" (everything > 70th pct)
LOG_OFFSET       = 1.0


def build(
    tss_path:     str   = TSS_PATH,
    biotype_path: str   = BIOTYPE_PATH,
    gtex_path:    str   = GTEX_PATH,
    out_path:     str   = OUT_PATH,
    min_tpm:      float = DEFAULT_MIN_TPM,
    low_pct:      float = DEFAULT_LOW_PCT,
    high_pct:     float = DEFAULT_HIGH_PCT,
) -> pd.DataFrame:

    tss      = pd.read_csv(tss_path)
    biotypes = pd.read_csv(biotype_path)
    gtex     = pd.read_csv(gtex_path)

    # Strip Ensembl version suffixes for robust joining
    tss["ensembl_id"]      = tss["ensembl_id"].astype(str).str.split(".").str[0]
    biotypes["ensembl_id"] = biotypes["ensembl_id"].astype(str).str.split(".").str[0]
    gtex["ensembl_id"]     = gtex["gene_id"].astype(str).str.split(".").str[0]

    print(f"\nFilter pipeline:")
    print(f"  Initial TSS sequences:       {len(tss):>7,} genes")

    # 1. Protein-coding filter
    coding = biotypes[biotypes["gene_biotype"] == "protein_coding"]
    tss = tss.merge(coding[["ensembl_id"]], on="ensembl_id", how="inner")
    print(f"  After protein-coding only:   {len(tss):>7,} genes")

    # 2. Join with GTEx expression
    gtex = gtex.sort_values("median_tpm", ascending=False).drop_duplicates("ensembl_id")
    merged = tss.merge(gtex[["ensembl_id", "median_tpm"]], on="ensembl_id", how="inner")
    merged = merged.dropna(subset=["median_tpm", "sequence"])
    print(f"  After joining with GTEx:     {len(merged):>7,} genes")

    # 3. Drop genes that are essentially unexpressed
    merged = merged[merged["median_tpm"] >= min_tpm]
    print(f"  After min_tpm >= {min_tpm}:        {len(merged):>7,} genes")

    # 4. log-TPM
    merged["log_tpm"] = np.log(merged["median_tpm"].clip(lower=0) + LOG_OFFSET)

    # 5. Binary labels via strict percentile cutoffs (drop the middle band)
    n     = len(merged)
    ranks = merged["log_tpm"].rank(method="first", ascending=True)
    low_mask  = ranks <= n * low_pct  / 100
    high_mask = ranks >  n * high_pct / 100

    merged["expression_label"] = "Drop"
    merged.loc[low_mask,  "expression_label"] = "Low"
    merged.loc[high_mask, "expression_label"] = "High"
    merged = merged[merged["expression_label"] != "Drop"].reset_index(drop=True)
    print(f"  After binary cutoffs ({low_pct}/{high_pct}): {len(merged):>7,} genes")

    print(f"\nFinal class balance:")
    print(merged["expression_label"].value_counts().to_string())

    print(f"\nExpression range per class:")
    for label in ["Low", "High"]:
        sub = merged[merged["expression_label"] == label]
        print(f"  {label:>4}: median TPM = {sub['median_tpm'].median():.3f}  "
              f"({sub['median_tpm'].min():.4f} – {sub['median_tpm'].max():.1f})")

    cols = ["ensembl_id", "gene_symbol", "chromosome", "tss", "strand",
            "sequence", "median_tpm", "log_tpm", "expression_label"]
    merged[cols].to_csv(out_path, index=False)
    print(f"\nSaved -> {out_path}")
    return merged


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--tss",      default=TSS_PATH,     help="TSS sequences CSV")
    p.add_argument("--biotype",  default=BIOTYPE_PATH, help="Gene biotypes CSV")
    p.add_argument("--gtex",     default=GTEX_PATH,    help="GTEx expression CSV (default: median across tissues)")
    p.add_argument("--out",      default=OUT_PATH,     help="Output CSV")
    p.add_argument("--min_tpm",  type=float, default=DEFAULT_MIN_TPM,
                   help="Drop genes below this median TPM")
    p.add_argument("--low_pct",  type=float, default=DEFAULT_LOW_PCT,
                   help="Bottom percentile threshold for Low class")
    p.add_argument("--high_pct", type=float, default=DEFAULT_HIGH_PCT,
                   help="Top percentile threshold for High class")
    args = p.parse_args()

    build(
        tss_path=args.tss, biotype_path=args.biotype, gtex_path=args.gtex,
        out_path=args.out, min_tpm=args.min_tpm,
        low_pct=args.low_pct, high_pct=args.high_pct,
    )
