"""
Master pipeline script — runs all data steps in order.

Usage:
    python run_pipeline.py           # full pipeline
    python run_pipeline.py --skip_download   # skip if CSVs already exist
"""

import os
import argparse

DATA_DIR = "data"


def step(name, fn):
    print(f"\n{'─'*60}")
    print(f"  STEP: {name}")
    print(f"{'─'*60}")
    fn()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--skip_download", action="store_true",
                   help="Skip downloading if raw CSVs already exist")
    args = p.parse_args()

    ncbi_path = os.path.join(DATA_DIR, "ncbi_sequences.csv")
    gtex_path = os.path.join(DATA_DIR, "gtex_expression.csv")
    map_path  = os.path.join(DATA_DIR, "ensembl_to_symbol.csv")
    out_path  = os.path.join(DATA_DIR, "labeled_genes.csv")

    if not args.skip_download or not os.path.exists(ncbi_path):
        from data.download_ncbi import download as dl_ncbi
        step("Download NCBI gene sequences", dl_ncbi)
    else:
        print(f"[skip] {ncbi_path} exists.")

    if not args.skip_download or not os.path.exists(gtex_path):
        from data.download_gtex import download as dl_gtex
        step("Download GTEx TPM data", dl_gtex)
    else:
        print(f"[skip] {gtex_path} exists.")

    if not args.skip_download or not os.path.exists(map_path):
        from data.map_ids import map_ensembl_to_symbol
        step("Map Ensembl IDs -> gene symbols", map_ensembl_to_symbol)
    else:
        print(f"[skip] {map_path} exists.")

    if not args.skip_download or not os.path.exists(out_path):
        from data.build_dataset import build
        step("Build labeled_genes.csv", build)
    else:
        print(f"[skip] {out_path} exists.")

    print("\nPipeline complete.  Next: python train.py")


if __name__ == "__main__":
    main()
