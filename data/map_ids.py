"""
Map Ensembl gene IDs (GTEx) to HGNC gene symbols for joining with NCBI sequences.

Uses the `mygene` package which wraps the MyGene.info REST API.
Outputs: data/ensembl_to_symbol.csv  (ensembl_id, gene_symbol)
"""

import os
import pandas as pd
import mygene

GTEX_PATH   = os.path.join(os.path.dirname(__file__), "gtex_whole_blood.csv")
OUT_PATH    = os.path.join(os.path.dirname(__file__), "ensembl_to_symbol.csv")

BATCH_SIZE  = 1_000


def map_ensembl_to_symbol(gtex_path: str = GTEX_PATH, out_path: str = OUT_PATH) -> pd.DataFrame:
    gtex = pd.read_csv(gtex_path)
    ensembl_ids = gtex["gene_id"].unique().tolist()
    print(f"Mapping {len(ensembl_ids)} Ensembl IDs -> gene symbols ...")

    mg = mygene.MyGeneInfo()
    results = []

    for i in range(0, len(ensembl_ids), BATCH_SIZE):
        batch = ensembl_ids[i : i + BATCH_SIZE]
        hits = mg.querymany(batch, scopes="ensembl.gene", fields="symbol", species="human", verbose=False)
        for h in hits:
            if "symbol" in h and not h.get("notfound", False):
                results.append({"ensembl_id": h["query"], "gene_symbol": h["symbol"]})
        print(f"  Processed {min(i + BATCH_SIZE, len(ensembl_ids))}/{len(ensembl_ids)}")

    df = pd.DataFrame(results).drop_duplicates("ensembl_id").reset_index(drop=True)
    print(f"Mapped: {len(df)} / {len(ensembl_ids)} ({100*len(df)/len(ensembl_ids):.1f}%)")
    df.to_csv(out_path, index=False)
    print(f"Saved -> {out_path}")
    return df


if __name__ == "__main__":
    map_ensembl_to_symbol()