"""
Download the Xpresso benchmark dataset (Agarwal & Shendure, 2020).

Source: Vikram Agarwal's distribution at washington.edu
        https://krishna.gs.washington.edu/content/members/vagar/Xpresso/data/datasets/pM10Kb_1KTest/

Files (auto-downloaded, ~281 MB total):
  - train.h5  (~251 MB, 16,377 genes)
  - valid.h5  (~15 MB,   1,000 genes)
  - test.h5   (~15 MB,   1,000 genes)

Schema:
  - Sequences are one-hot encoded ±10 kb around the TSS, decoded to ACGT here.
  - Labels are log10(mRNA + ε) — a normalised expression value (NOT raw GTEx TPM).
  - The canonical Xpresso train/valid/test split is preserved in `xpresso_split`.

Output: data/labeled_genes_xpresso.csv
Columns: gene_id, gene_symbol, sequence, log_tpm, expression_label, xpresso_split

For binary classification, run `train.py` on the output normally — the rank-percentile
binning produces 3 classes by default; pass `--data` and the standard split kicks in.

Usage:
    python data/download_xpresso.py
    python data/download_xpresso.py --out data/my_xpresso.csv
"""

import os
import argparse
import gc
import requests
import numpy as np
import pandas as pd

DATA_DIR     = os.path.dirname(__file__)
BASE_URL     = ("https://krishna.gs.washington.edu/content/members/vagar/"
                "Xpresso/data/datasets/pM10Kb_1KTest")
SPLITS       = ["train", "valid", "test"]
H5_PATH_FMT  = os.path.join(DATA_DIR, "xpresso_{split}.h5")
OUT_PATH     = os.path.join(DATA_DIR, "labeled_genes_xpresso.csv")

LOW_PCT      = 33
HIGH_PCT     = 67


def _download_file(url: str, dest: str) -> None:
    if os.path.exists(dest) and os.path.getsize(dest) > 1_000_000:
        print(f"  Cached: {os.path.basename(dest)} ({os.path.getsize(dest) // 1024 // 1024} MB)")
        return

    print(f"  Downloading {url}")
    with requests.get(url, stream=True, timeout=600) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        downloaded = 0
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                f.write(chunk)
                downloaded += len(chunk)
                if total and downloaded % (50 * 1024 * 1024) < 1024 * 1024:
                    print(f"    {downloaded // 1024 // 1024}/{total // 1024 // 1024} MB"
                          f" ({downloaded / total * 100:.0f}%)")
    print(f"    saved -> {dest}")


def _onehot_batch_to_seqs(arr: np.ndarray) -> list:
    """Decode (N, L, 4) one-hot array to a list of N ACGT strings."""
    bases   = np.array(['A', 'C', 'G', 'T'], dtype='U1')
    indices = arr.argmax(axis=-1)             # (N, L)
    is_zero = arr.sum(axis=-1) == 0           # (N, L)
    chars   = bases[indices]                  # (N, L) of single-char strings
    chars[is_zero] = 'N'

    # Fast vectorised path: view (N, L) U1 array as (N,) of length-L strings.
    # Falls back to per-row join if memory layout doesn't allow the view.
    try:
        contig = np.ascontiguousarray(chars)
        L = contig.shape[1]
        return contig.view(f'U{L}').reshape(-1).tolist()
    except (ValueError, TypeError):
        return [''.join(row.tolist()) for row in chars]


def _pick_sequence_key(f, keys: list) -> str:
    """Return the HDF5 key whose data is 3-D with last axis 4 (i.e. one-hot DNA)."""
    candidates = []
    for k in keys:
        try:
            shape = f[k].shape
        except Exception:
            continue
        if len(shape) == 3 and shape[-1] == 4:
            candidates.append((k, shape))
        elif len(shape) == 3 and shape[1] == 4:
            # (N, 4, L) layout — also valid, will be transposed later
            candidates.append((k, shape))

    if not candidates:
        return None

    # Prefer keys named 'promoter'/'sequence'/'X' if multiple match
    preferred = ("promoter", "sequence", "seqs", "input", "x")
    for name in preferred:
        for k, _ in candidates:
            if k.lower() == name:
                return k
    return candidates[0][0]


def _load_h5_split(path: str) -> pd.DataFrame:
    """Parse an Xpresso HDF5 file. Schema is auto-detected from keys + shapes."""
    import h5py

    with h5py.File(path, "r") as f:
        keys = list(f.keys())
        print(f"    HDF5 keys + shapes: {[(k, f[k].shape) for k in keys]}")

        seq_key = _pick_sequence_key(f, keys)
        label_key = next((k for k in keys if k.lower() in
                          ("label", "y", "expression", "mrna", "labels")), None)
        name_key = next((k for k in keys if "gene" in k.lower() or "name" in k.lower()), None)

        if seq_key is None or label_key is None:
            raise RuntimeError(f"Couldn't auto-detect sequence / label keys in {path}. "
                               f"Found keys: {keys}")

        print(f"    using seq='{seq_key}'  label='{label_key}'  name='{name_key}'")

        seqs_oh = np.asarray(f[seq_key][:])
        labels  = np.asarray(f[label_key][:]).reshape(-1)
        if name_key is not None:
            raw_names = f[name_key][:]
        else:
            raw_names = np.array([f"gene_{i}" for i in range(len(labels))])

    # Normalise to (N, L, 4) layout
    if seqs_oh.ndim == 3 and seqs_oh.shape[1] == 4 and seqs_oh.shape[2] != 4:
        seqs_oh = seqs_oh.transpose(0, 2, 1)

    print(f"    decoding {len(seqs_oh):,} sequences of shape {seqs_oh.shape[1:]}")
    sequences = _onehot_batch_to_seqs(seqs_oh)

    # Decode HDF5 bytes -> str if necessary
    if raw_names.dtype.kind in ('S', 'O'):
        names = [n.decode() if isinstance(n, (bytes, np.bytes_)) else str(n)
                 for n in raw_names.tolist()]
    else:
        names = [str(n) for n in raw_names.tolist()]

    df = pd.DataFrame({
        "gene_symbol": names,
        "sequence":    sequences,
        "log_tpm":     labels.astype(float),
    })

    # Free large arrays before the next split is loaded
    del seqs_oh, labels, raw_names
    gc.collect()
    return df


def _bin_labels(values: pd.Series) -> pd.Series:
    """Rank-based percentile binning into Low / Medium / High."""
    n     = len(values)
    ranks = values.rank(method="first", ascending=True)
    out   = pd.Series("Medium", index=values.index, dtype=object)
    out[ranks <= n * LOW_PCT  / 100] = "Low"
    out[ranks >  n * HIGH_PCT / 100] = "High"
    return out


def download(out_path: str = OUT_PATH) -> pd.DataFrame:
    print("=== Xpresso dataset download ===\n")

    print("Step 1/3: Downloading HDF5 files ...")
    h5_paths = {}
    for split in SPLITS:
        path = H5_PATH_FMT.format(split=split)
        _download_file(f"{BASE_URL}/{split}.h5", path)
        h5_paths[split] = path

    print("\nStep 2/3: Parsing HDF5 files ...")
    frames = []
    for split in SPLITS:
        print(f"  -> {split}.h5")
        df = _load_h5_split(h5_paths[split])
        df["xpresso_split"] = split
        frames.append(df)
        print(f"     {len(df):,} genes")

    df = pd.concat(frames, ignore_index=True)
    del frames
    gc.collect()

    print(f"\nStep 3/3: Building labeled CSV ({len(df):,} total genes)")

    # Compute class labels using the rank-percentile of log_tpm across the full dataset
    df["expression_label"] = _bin_labels(df["log_tpm"])

    # gene_id mirrors gene_symbol (Xpresso doesn't ship NCBI IDs)
    df["gene_id"] = df["gene_symbol"]

    print("\nClass balance:")
    print(df["expression_label"].value_counts().to_string())

    print("\nSequence length stats:")
    lens = df["sequence"].str.len()
    print(f"  min={lens.min()}  median={int(lens.median())}  max={lens.max()}")

    print("\nlog_tpm value stats:")
    print(f"  min={df['log_tpm'].min():.3f}  median={df['log_tpm'].median():.3f}  "
          f"max={df['log_tpm'].max():.3f}")

    cols = ["gene_id", "gene_symbol", "sequence", "log_tpm",
            "expression_label", "xpresso_split"]
    df[cols].to_csv(out_path, index=False)
    print(f"\nSaved -> {out_path}")
    return df


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=OUT_PATH, help="Output CSV path")
    args = p.parse_args()
    download(out_path=args.out)
