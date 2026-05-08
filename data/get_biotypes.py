"""
Fetch gene biotypes (protein_coding, lncRNA, pseudogene, ...) for all human Ensembl genes.

Used to filter noisy non-coding genes out of the labeled dataset before training.

Outputs: data/gene_biotypes.csv  (ensembl_id, gene_biotype)
"""

import os
import requests
import pandas as pd
from io import StringIO

BIOMART_URL = "https://www.ensembl.org/biomart/martservice"
OUT_PATH    = os.path.join(os.path.dirname(__file__), "gene_biotypes.csv")

_BIOMART_XML = (
    '<?xml version="1.0" encoding="UTF-8"?>'
    '<!DOCTYPE Query>'
    '<Query virtualSchemaName="default" formatter="TSV" header="1"'
    ' uniqueRows="1" count="">'
    '<Dataset name="hsapiens_gene_ensembl" interface="default">'
    '<Attribute name="ensembl_gene_id"/>'
    '<Attribute name="gene_biotype"/>'
    '</Dataset>'
    '</Query>'
)


def fetch(out_path: str = OUT_PATH) -> pd.DataFrame:
    print("Querying Ensembl BioMart for gene biotypes ...")
    r = requests.get(BIOMART_URL, params={"query": _BIOMART_XML}, timeout=300)
    r.raise_for_status()

    df = pd.read_csv(StringIO(r.text), sep="\t", dtype=str)
    df.columns = ["ensembl_id", "gene_biotype"]
    df = df.dropna().drop_duplicates("ensembl_id").reset_index(drop=True)

    print(f"Fetched biotypes for {len(df):,} genes")
    print("Top biotypes:")
    print(df["gene_biotype"].value_counts().head(10).to_string())

    df.to_csv(out_path, index=False)
    print(f"Saved -> {out_path}")
    return df


if __name__ == "__main__":
    fetch()
