# Data Directory

| File | Created by | Description |
|---|---|---|
| `ncbi_sequences.csv` | `download_ncbi.py` | Gene sequences from GenerativeLM-Genes GitHub repo |
| `gtex_whole_blood.csv` | `download_gtex.py` | Whole-blood median TPM from GTEx v8 |
| `gtex_median_tpm.gct.gz` | `download_gtex.py` | Cached raw GCT file (~50 MB) |
| `ensembl_to_symbol.csv` | `map_ids.py` | Ensembl ID → HGNC symbol mapping via MyGene.info |
| `labeled_genes.csv` | `build_dataset.py` | **Final dataset** — sequence + expression label |

## Column definitions — `labeled_genes.csv`

| Column | Type | Description |
|---|---|---|
| `gene_id` | str | Ensembl gene ID |
| `gene_symbol` | str | HGNC gene symbol |
| `sequence` | str | Nucleotide sequence (A/C/G/T, ≤1000 bp) |
| `median_tpm` | float | Raw whole-blood TPM from GTEx |
| `log_tpm` | float | log(TPM + 1) |
| `expression_label` | str | Low / Medium / High (33rd/67th percentile split) |
