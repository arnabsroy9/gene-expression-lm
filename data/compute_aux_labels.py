"""
Compute biologically-meaningful auxiliary labels for sequences in a labeled CSV.

Adds three columns:
  - cpg_island   : binary (0/1)  - any 200 bp window meets Gardiner-Garden & Frommer
                                   criteria (GC% >= 50, CpG obs/exp >= 0.6)
  - tata_box     : binary (0/1)  - TATAA-like motif found within --tata_window bp
                                   upstream of the TSS (assumed at sequence centre)
  - gc_content   : float [0, 1]  - fraction of G/C nucleotides over the full sequence

These act as auxiliary tasks for multi-task transformer training, providing
biologically-grounded inductive bias to the shared encoder.

Usage:
    python data/compute_aux_labels.py --in data/labeled_genes_xpresso.csv
    python data/compute_aux_labels.py --in data/labeled_genes_xpresso.csv --out data/xpresso_aux.csv
    python data/compute_aux_labels.py --in ... --tata_window 100   # wider TATA search
"""

import re
import argparse
import pandas as pd

# CpG island parameters (Gardiner-Garden & Frommer 1987)
CPG_GC_MIN     = 0.50
CPG_OE_MIN     = 0.60
CPG_WINDOW     = 200
CPG_STEP       = 50

# TATA box pattern: TATAA[AT]A[AT]? - canonical and common variants
TATA_PATTERN   = re.compile(r"TATA[AT]A[AT]?")
TATA_WINDOW_BP = 50    # search this many bp upstream of TSS


def gc_content(seq: str) -> float:
    """Fraction of G or C nucleotides over the entire sequence."""
    if not seq:
        return 0.0
    seq = seq.upper()
    return (seq.count("G") + seq.count("C")) / len(seq)


def _cpg_oe(seq: str) -> float:
    """Observed/Expected CpG ratio for a single window.
    Expected = (#C * #G) / length.
    """
    L = len(seq)
    if L < 2:
        return 0.0
    g = seq.count("G")
    c = seq.count("C")
    if g == 0 or c == 0:
        return 0.0
    obs = seq.count("CG")
    exp = (g * c) / L
    return obs / exp if exp > 0 else 0.0


def has_cpg_island(seq: str, window: int = CPG_WINDOW, step: int = CPG_STEP) -> int:
    """1 if any sliding window meets CpG-island criteria."""
    seq = seq.upper()
    L = len(seq)
    if L < window:
        return 0
    for start in range(0, L - window + 1, step):
        sub = seq[start:start + window]
        if gc_content(sub) >= CPG_GC_MIN and _cpg_oe(sub) >= CPG_OE_MIN:
            return 1
    return 0


def has_tata_box(seq: str, tss_pos: int = None, window: int = TATA_WINDOW_BP) -> int:
    """1 if a TATA-like motif occurs within `window` bp upstream of the TSS.
    For TSS-centred sequences (Xpresso, our TSS pipeline), TSS = sequence centre.
    """
    if tss_pos is None:
        tss_pos = len(seq) // 2
    start  = max(0, tss_pos - window)
    region = seq[start:tss_pos].upper()
    return 1 if TATA_PATTERN.search(region) else 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in",  dest="inp", required=True, help="Input labeled CSV")
    p.add_argument("--out", default=None, help="Output CSV (default: overwrite input)")
    p.add_argument("--tata_window", type=int, default=TATA_WINDOW_BP,
                   help=f"Bp upstream of TSS to search for TATA box (default {TATA_WINDOW_BP})")
    args = p.parse_args()

    out_path = args.out or args.inp

    print(f"Reading {args.inp} ...")
    df = pd.read_csv(args.inp)
    print(f"  {len(df):,} sequences")

    print("Computing auxiliary labels ...")
    df["gc_content"] = df["sequence"].apply(gc_content)
    df["cpg_island"] = df["sequence"].apply(has_cpg_island)
    df["tata_box"]   = df["sequence"].apply(lambda s: has_tata_box(s, window=args.tata_window))

    n = len(df)
    cpg_pos  = int(df["cpg_island"].sum())
    tata_pos = int(df["tata_box"].sum())
    print(f"\nAuxiliary label statistics:")
    print(f"  CpG island present : {cpg_pos:>6,} / {n:,}  ({cpg_pos/n*100:5.1f}%)")
    print(f"  TATA box present   : {tata_pos:>6,} / {n:,}  ({tata_pos/n*100:5.1f}%)")
    print(f"  GC content         : min={df['gc_content'].min():.3f}  "
          f"median={df['gc_content'].median():.3f}  max={df['gc_content'].max():.3f}")

    print(f"\nClass-conditional aux label rates (sanity check):")
    if "expression_label" in df.columns:
        for lbl, grp in df.groupby("expression_label"):
            print(f"  {lbl:>6}: cpg={grp['cpg_island'].mean():.2f}  "
                  f"tata={grp['tata_box'].mean():.2f}  "
                  f"gc={grp['gc_content'].mean():.3f}")

    df.to_csv(out_path, index=False)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
