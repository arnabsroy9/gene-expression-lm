"""
Download GTEx median TPM data for a chosen tissue and persist locally.

The GCT.gz (~50 MB) contains all tissues and is downloaded once then cached.
Re-running with a different --tissue just re-parses the cached file.

Usage:
    python data/download_gtex.py                        # default: Liver
    python data/download_gtex.py --tissue "Whole Blood"
    python data/download_gtex.py --list_tissues         # show all available tissues

Outputs: data/gtex_expression.csv  (gene_id [Ensembl], median_tpm)
"""

import os
import gzip
import shutil
import argparse
import requests
import pandas as pd

GTEX_URL = (
    "https://storage.googleapis.com/adult-gtex/bulk-gex/v8/rna-seq/"
    "GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_median_tpm.gct.gz"
)

DEFAULT_TISSUE = "Liver"
GCT_GZ_PATH   = os.path.join(os.path.dirname(__file__), "gtex_median_tpm.gct.gz")
OUT_PATH       = os.path.join(os.path.dirname(__file__), "gtex_expression.csv")


def _download_gct(url: str, dest: str) -> None:
    print(f"Downloading GTEx TPM table ({url}) ...")
    with requests.get(url, stream=True, timeout=300) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            shutil.copyfileobj(r.raw, f)
    print(f"Saved GCT.gz -> {dest}")


def _parse_gct(gz_path: str, tissue_col: str) -> pd.DataFrame:
    print(f"Parsing GCT file (tissue: {tissue_col}) ...")
    with gzip.open(gz_path, "rt") as fh:
        fh.readline()  # #1.2
        fh.readline()  # dimensions
        df = pd.read_csv(fh, sep="\t")

    df["gene_id"] = df["Name"].str.split(".").str[0]

    if tissue_col not in df.columns:
        available = [c for c in df.columns if c not in ("Name", "Description", "gene_id")]
        raise ValueError(
            f"Tissue '{tissue_col}' not found.\nAvailable tissues (first 20):\n"
            + "\n".join(f"  {t}" for t in available[:20])
        )

    result = df[["gene_id", tissue_col]].copy()
    result.columns = ["gene_id", "median_tpm"]
    result = result.dropna().reset_index(drop=True)
    print(f"Genes with TPM data: {len(result)}")
    return result


def list_tissues(gz_path: str = GCT_GZ_PATH) -> list:
    if not os.path.exists(gz_path):
        raise FileNotFoundError(f"GCT file not found: {gz_path}. Run without --list_tissues first.")
    with gzip.open(gz_path, "rt") as fh:
        fh.readline()
        fh.readline()
        header = fh.readline().strip().split("\t")
    return [c for c in header if c not in ("Name", "Description")]


def download(
    url:        str = GTEX_URL,
    tissue_col: str = DEFAULT_TISSUE,
    out_path:   str = OUT_PATH,
) -> pd.DataFrame:
    if not os.path.exists(GCT_GZ_PATH):
        _download_gct(url, GCT_GZ_PATH)
    else:
        print(f"Using cached GCT.gz: {GCT_GZ_PATH}")

    df = _parse_gct(GCT_GZ_PATH, tissue_col)
    df.to_csv(out_path, index=False)
    print(f"Saved -> {out_path}")
    return df


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--tissue",       default=DEFAULT_TISSUE, help="GTEx tissue column name")
    p.add_argument("--list_tissues", action="store_true",    help="Print all available tissue names and exit")
    args = p.parse_args()

    if args.list_tissues:
        for t in list_tissues():
            print(t)
    else:
        download(tissue_col=args.tissue)
