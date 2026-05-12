"""
Attention rollout + input-gradient saliency visualisation for the Transformer.

Usage:
    python interpret.py                       # 3 random examples
    python interpret.py --gene IFNB1          # Low-expression example
    python interpret.py --gene CCDC182        # Medium-expression example
    python interpret.py --gene HBA1           # High-expression example
    python interpret.py --sequence ATCG...    # custom DNA string
    python interpret.py --n_examples 5

Gene symbols must exist in data/labeled_genes.csv (source = NCBI / GenerativeLM-Genes).
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import torch

from preprocess.tokenizer import KmerTokenizer

CKPT_DIR = "checkpoints"
OUT_DIR  = "outputs"
DATA_CSV = os.path.join("data", "labeled_genes.csv")
MAX_LEN    = 994
K          = 6
CLASSES    = ["Low", "Medium", "High"]
D_MODEL    = 128
NHEAD      = 4
NUM_LAYERS = 4
DIM_FF     = 256

os.makedirs(OUT_DIR, exist_ok=True)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_task():
    task_file = os.path.join(CKPT_DIR, "transformer_task.txt")
    if os.path.exists(task_file):
        return open(task_file).read().strip()
    return "classification"


def _plot_heatmap(tokens, scores, title, out_path):
    n      = min(len(tokens), 80)
    tokens = tokens[:n]
    scores = scores[:n]

    norm   = mcolors.Normalize(vmin=0, vmax=scores.max() + 1e-9)
    cmap   = plt.cm.YlOrRd
    colors = cmap(norm(scores))

    fig_w  = max(12, n * 0.28)
    fig, ax = plt.subplots(figsize=(fig_w, 2.5))
    for i, (tok, col) in enumerate(zip(tokens, colors)):
        ax.add_patch(plt.Rectangle((i, 0), 1, 1, color=col))
        ax.text(i + 0.5, 0.5, tok, ha="center", va="center", fontsize=6, rotation=90)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    plt.colorbar(sm, ax=ax, orientation="horizontal", pad=0.02, shrink=0.6)
    ax.set_xlim(0, n)
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {out_path}")


def _load_model(device, task):
    from models.transformer_model import TransformerClassifier
    tag = "transformer_regression" if task == "regression" else "transformer"
    ckpt = os.path.join(CKPT_DIR, f"{tag}.pt")
    if not os.path.exists(ckpt):
        ckpt = os.path.join(CKPT_DIR, "transformer.pt")

    tok         = KmerTokenizer(k=K, max_len=MAX_LEN)
    num_classes = 1 if task == "regression" else 3
    model       = TransformerClassifier(
        vocab_size=tok.vocab_size, pad_idx=tok.pad_id,
        max_len=MAX_LEN + 1, num_classes=num_classes,
        d_model=D_MODEL, nhead=NHEAD, num_layers=NUM_LAYERS, dim_ff=DIM_FF,
    ).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    return model, tok


def _pred_label(model, ids, mask, task):
    with torch.no_grad():
        out = model(ids, attention_mask=mask)

    if task == "regression":
        thresholds = np.load(os.path.join(CKPT_DIR, "transformer_thresholds.npy"))
        score = out[0, 0].item()
        if   score <= thresholds[0]: return CLASSES[0]
        elif score >  thresholds[1]: return CLASSES[2]
        else:                        return CLASSES[1]
    else:
        return CLASSES[torch.softmax(out, dim=-1).argmax().item()]


def interpret(sequence: str, label: str = None, tag: str = "seq"):
    device       = get_device()
    task         = _load_task()
    model, tok   = _load_model(device, task)

    ids_np = tok.encode_padded(sequence, add_cls=True)
    ids    = torch.from_numpy(ids_np).unsqueeze(0).long().to(device)
    mask   = (ids != 0).long()
    kmers  = tok.tokenize(sequence)
    n      = len(kmers)

    pred_label = _pred_label(model, ids, mask, task)
    true_str   = f"  |  True: {label}" if label else ""

    # ── Attention rollout ──────────────────────────────────────────────────────
    rollout = model.get_attention_rollout(ids, mask)  # (L-1,)
    rollout = rollout[:n]
    _plot_heatmap(
        kmers, rollout,
        title=f"Attention Rollout  |  Pred: {pred_label}{true_str}",
        out_path=os.path.join(OUT_DIR, f"rollout_{tag}.png"),
    )

    # ── Input-gradient saliency ────────────────────────────────────────────────
    saliency = model.get_input_saliency(ids, mask)  # (L,) including CLS
    saliency = saliency[1:n + 1]                     # drop CLS, trim to seq length
    _plot_heatmap(
        kmers, saliency,
        title=f"Input Gradient Saliency  |  Pred: {pred_label}{true_str}",
        out_path=os.path.join(OUT_DIR, f"saliency_{tag}.png"),
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gene",       default=None, help="Gene symbol to look up")
    p.add_argument("--sequence",   default=None, help="Raw DNA sequence")
    p.add_argument("--n_examples", type=int, default=3)
    args = p.parse_args()

    if args.sequence:
        interpret(args.sequence, tag="custom")
    elif args.gene:
        df  = pd.read_csv(DATA_CSV)
        row = df[df["gene_symbol"].str.upper() == args.gene.upper()]
        if row.empty:
            print(f"Gene '{args.gene}' not found.")
            return
        interpret(row.iloc[0]["sequence"], label=row.iloc[0]["expression_label"], tag=args.gene.upper())
    else:
        df = pd.read_csv(DATA_CSV).dropna(subset=["sequence", "expression_label"]).sample(args.n_examples, random_state=0)
        for i, (_, row) in enumerate(df.iterrows()):
            interpret(row["sequence"], label=row["expression_label"], tag=f"example_{i}")


if __name__ == "__main__":
    main()
