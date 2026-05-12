"""
Fetch human gene nucleotide sequences (NCBI Gene database) and persist locally.

The sequences are pulled from a publicly redistributed CSV mirror to avoid
making thousands of small NCBI API calls. The biological provenance is the
NCBI Gene database — the mirror simply packages already-public sequences.

Outputs: data/ncbi_sequences.csv  (gene_id, gene_symbol, sequence)
"""

import os
import requests
import pandas as pd
from io import StringIO

# Public CSV mirror that packages NCBI Gene sequences for bulk download.
# Used purely to avoid rate-limited NCBI API calls; the underlying sequences
# are NCBI's.
RAW_URL = (
    "https://raw.githubusercontent.com/boun-tabi/GenerativeLM-Genes/"
    "main/Datasets/unsplitted_dataset.csv"
)
OUT_PATH = os.path.join(os.path.dirname(__file__), "ncbi_sequences.csv")

MAX_LEN = 1_000
VALID_BASES = set("ACGT")


def _is_valid(seq: str) -> bool:
    return bool(seq) and all(c in VALID_BASES for c in seq)


def download(url: str = RAW_URL, out_path: str = OUT_PATH) -> pd.DataFrame:
    print(f"Fetching sequences from {url} ...")
    response = requests.get(url, timeout=120)
    response.raise_for_status()

    df = pd.read_csv(StringIO(response.text))
    print(f"Raw rows: {len(df)}")

    # Actual columns: NCBIGeneID, Symbol, NucleotideSequence (+ others we don't need)
    df = df.rename(columns={
        "NCBIGeneID":        "gene_id",
        "Symbol":            "gene_symbol",
        "NucleotideSequence": "sequence",
    })

    # Sequences are wrapped in angle brackets e.g. <ATCG...> — strip them
    df["sequence"] = (
        df["sequence"].astype(str)
        .str.strip()
        .str.strip("<>")
        .str.upper()
    )

    df = df[df["sequence"].apply(_is_valid)]
    df = df[df["sequence"].str.len() <= MAX_LEN]
    df = df[["gene_id", "gene_symbol", "sequence"]].drop_duplicates().reset_index(drop=True)

    print(f"After filtering: {len(df)} sequences")
    df.to_csv(out_path, index=False)
    print(f"Saved -> {out_path}")
    return df


if __name__ == "__main__":
    download()