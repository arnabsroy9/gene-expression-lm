"""
Download TSS-centred sequences (±WINDOW bp) for human genes via Ensembl.

Steps:
  1. Bulk-fetch TSS coordinates from Ensembl BioMart (all human genes)
  2. Select canonical TSS per gene (most 5′ transcript on each strand)
  3. Fetch genomic sequence via Ensembl REST API (batches of 50 regions)
  4. Filter to valid ACGT-only sequences on standard chromosomes

The returned sequence is always oriented 5′→3′ relative to the gene:
  + strand genes: genomic forward strand, upstream on the left
  - strand genes: reverse-complement, so upstream is still on the left

Outputs: data/tss_sequences.csv
Columns: ensembl_id, gene_symbol, chromosome, tss, strand, sequence

Usage:
    python data/download_tss.py                  # default ±2 kb window
    python data/download_tss.py --window 1000    # ±1 kb (4× faster)
    python data/download_tss.py --out data/my_tss.csv
"""

import os
import time
import argparse
import requests
import pandas as pd
from io import StringIO

BIOMART_URL     = "https://www.ensembl.org/biomart/martservice"
ENSEMBL_API     = "https://rest.ensembl.org"
DEFAULT_WINDOW  = 2_000          # bp on each side of TSS
BATCH_SIZE      = 50             # Ensembl REST API max regions per POST
REQUEST_DELAY   = 0.07           # ~14 req/s, safely under the 15/s limit
MAX_RETRIES     = 4
VALID_BASES     = set("ACGT")
STANDARD_CHROMS = {str(i) for i in range(1, 23)} | {"X", "Y"}
OUT_PATH        = os.path.join(os.path.dirname(__file__), "tss_sequences.csv")

_BIOMART_XML = (
    '<?xml version="1.0" encoding="UTF-8"?>'
    '<!DOCTYPE Query>'
    '<Query virtualSchemaName="default" formatter="TSV" header="1"'
    ' uniqueRows="0" count="">'
    '<Dataset name="hsapiens_gene_ensembl" interface="default">'
    '<Attribute name="ensembl_gene_id"/>'
    '<Attribute name="external_gene_name"/>'
    '<Attribute name="chromosome_name"/>'
    '<Attribute name="transcription_start_site"/>'
    '<Attribute name="strand"/>'
    '</Dataset>'
    '</Query>'
)


def _fetch_tss_coords() -> pd.DataFrame:
    print("Querying Ensembl BioMart for TSS coordinates ...")
    r = requests.get(BIOMART_URL, params={"query": _BIOMART_XML}, timeout=300)
    r.raise_for_status()

    df = pd.read_csv(StringIO(r.text), sep="\t", dtype=str)
    df.columns = ["ensembl_id", "gene_symbol", "chromosome", "tss", "strand"]

    df["tss"]    = pd.to_numeric(df["tss"],    errors="coerce")
    df["strand"] = pd.to_numeric(df["strand"], errors="coerce")
    df = df.dropna(subset=["tss", "strand"])
    df["tss"]    = df["tss"].astype(int)
    df["strand"] = df["strand"].astype(int)

    print(f"  Transcript records fetched: {len(df):,}")
    return df


def _select_canonical(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df["chromosome"].isin(STANDARD_CHROMS)].copy()

    # Most 5′ TSS per gene:
    #   + strand → smallest genomic coordinate is most upstream → keep first after sort asc
    #   - strand → largest genomic coordinate is most upstream → keep first after sort desc
    pos = (
        df[df["strand"] == 1]
        .sort_values("tss", ascending=True)
        .drop_duplicates("ensembl_id", keep="first")
    )
    neg = (
        df[df["strand"] == -1]
        .sort_values("tss", ascending=False)
        .drop_duplicates("ensembl_id", keep="first")
    )
    canonical = pd.concat([pos, neg], ignore_index=True)
    print(f"  Canonical TSS per gene (standard chroms): {len(canonical):,}")
    return canonical


def _post_batch(session: requests.Session, regions: list) -> list:
    for attempt in range(MAX_RETRIES):
        try:
            r = session.post(
                f"{ENSEMBL_API}/sequence/region/human",
                json={"regions": regions},
                timeout=90,
            )
            if r.status_code == 429:
                wait = 2 ** attempt
                print(f"  Rate-limited, retrying in {wait}s ...")
                time.sleep(wait)
                continue
            r.raise_for_status()
            return r.json()
        except Exception as exc:
            if attempt == MAX_RETRIES - 1:
                print(f"  Warning: batch failed after {MAX_RETRIES} attempts: {exc}")
                return []
            time.sleep(2 ** attempt)
    return []


def _fetch_sequences(coords: pd.DataFrame, window: int) -> pd.DataFrame:
    session = requests.Session()
    session.headers.update({
        "Content-Type": "application/json",
        "Accept":       "application/json",
    })

    rows    = coords.to_dict("records")
    total   = len(rows)
    records = []

    print(f"Fetching ±{window:,} bp sequences from Ensembl REST API ...")

    for i in range(0, total, BATCH_SIZE):
        batch = rows[i : i + BATCH_SIZE]

        regions = []
        for row in batch:
            start = max(1, row["tss"] - window)
            end   = row["tss"] + window
            # Embedding strand in region string makes Ensembl return reverse-complement
            # for - strand genes, so position 0 is always 5′ (upstream) of TSS.
            regions.append(f"{row['chromosome']}:{start}..{end}:{row['strand']}")

        seq_objects = _post_batch(session, regions)

        for row, seq_obj in zip(batch, seq_objects):
            seq = seq_obj.get("seq", "").upper() if isinstance(seq_obj, dict) else ""
            if seq and all(c in VALID_BASES for c in seq):
                records.append({
                    "ensembl_id":  row["ensembl_id"],
                    "gene_symbol": row["gene_symbol"],
                    "chromosome":  row["chromosome"],
                    "tss":         row["tss"],
                    "strand":      row["strand"],
                    "sequence":    seq,
                })

        done = min(i + BATCH_SIZE, total)
        if done % (BATCH_SIZE * 20) == 0 or done == total:
            print(f"  Processed {done:,}/{total:,} genes — {len(records):,} valid so far")

        time.sleep(REQUEST_DELAY)

    return pd.DataFrame(records)


def download(window: int = DEFAULT_WINDOW, out_path: str = OUT_PATH) -> pd.DataFrame:
    coords = _fetch_tss_coords()
    coords = _select_canonical(coords)

    df = _fetch_sequences(coords, window)

    print(f"\nValid sequences: {len(df):,} / {len(coords):,}")
    df.to_csv(out_path, index=False)
    print(f"Saved -> {out_path}")
    return df


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "--window", type=int, default=DEFAULT_WINDOW,
        help="Bp on each side of TSS (default 2000 → 4001 bp total)",
    )
    p.add_argument("--out", default=OUT_PATH, help="Output CSV path")
    args = p.parse_args()
    download(window=args.window, out_path=args.out)
