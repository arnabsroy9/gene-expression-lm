"""
Generate report figures into latex/figures/.

Produces:
  - class_distribution.png      (computed from labelled CSV + reproduced splits)
  - architecture_diagram.png    (matplotlib block diagram)
  - roc_curves.png              (computed from trained checkpoint + test set)
  - confusion_matrix.png        (copied from outputs/)
  - attention_hba1.png          (copied from outputs/)
  - saliency_hba1.png           (copied from outputs/)
  - webapp_screenshot.png       (copied from assets/)

Usage:
    python make_figures.py
"""

import os
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

# --- Paths ---------------------------------------------------------------
DATA_CSV       = os.path.join("data", "labeled_genes.csv")
CKPT_DIR       = "checkpoints"
FIG_DIR        = os.path.join("latex", "figures")

LABEL_ORDER    = ["Low", "Medium", "High"]
CLASS_COLOURS  = {"Low": "#3b82f6", "Medium": "#f59e0b", "High": "#ef4444"}

os.makedirs(FIG_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────
# 1. class_distribution.png
# ─────────────────────────────────────────────────────────────────────────
def make_class_distribution():
    print("  Building class_distribution.png ...")
    df = pd.read_csv(DATA_CSV).dropna(subset=["sequence", "expression_label"])
    labels = df["expression_label"].astype(str).tolist()

    # Reproduce the train/val/test split exactly as train.py does
    idx = np.arange(len(labels))
    idx_tv, idx_test = train_test_split(
        idx, test_size=0.20, stratify=labels, random_state=42)
    idx_train, idx_val = train_test_split(
        idx_tv, test_size=0.25,
        stratify=[labels[i] for i in idx_tv],
        random_state=42,
    )

    def _counts(idxs):
        sub = [labels[i] for i in idxs]
        return [sub.count(c) for c in LABEL_ORDER]

    train_counts = _counts(idx_train)
    val_counts   = _counts(idx_val)
    test_counts  = _counts(idx_test)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    splits = ["Train", "Validation", "Test"]
    counts = np.array([train_counts, val_counts, test_counts])
    x = np.arange(len(splits))
    width = 0.26

    for i, cls in enumerate(LABEL_ORDER):
        ax.bar(x + (i - 1) * width, counts[:, i],
               width, label=cls, color=CLASS_COLOURS[cls],
               edgecolor="white", linewidth=0.6)
        # value labels on top of each bar
        for xi, val in zip(x + (i - 1) * width, counts[:, i]):
            ax.text(xi, val + 60, f"{val:,}",
                    ha="center", va="bottom", fontsize=8.5)

    ax.set_xticks(x)
    ax.set_xticklabels(splits)
    ax.set_ylabel("Number of genes")
    ax.set_title("Class balance across the train / validation / test splits", pad=12)
    ax.legend(title="Class", frameon=False, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylim(0, counts.max() * 1.18)

    fig.tight_layout()
    out = os.path.join(FIG_DIR, "class_distribution.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"     -> {out}")


# ─────────────────────────────────────────────────────────────────────────
# 2. architecture_diagram.png
# ─────────────────────────────────────────────────────────────────────────
def make_architecture_diagram():
    print("  Building architecture_diagram.png ...")
    blocks = [
        ("DNA sequence  (≤ 1,000 bp,  A/C/G/T)",        "#e8eaf6"),
        ("6-mer Tokenizer\n(k = 6, stride = 1, vocab = 4,099)", "#c5cae9"),
        ("Token Embedding (d = 128)  +  Sinusoidal Positional Encoding", "#9fa8da"),
        ("Transformer Encoder Block  × 4\n(4 attention heads, dim_ff = 256, dropout = 0.1, pre-LN)", "#7986cb"),
        ("CLS-token pooling  (sequence-level representation)", "#5c6bc0"),
        ("LayerNorm  →  Dropout  →  Linear (128 → 3)", "#3f51b5"),
        ("Logits over  {Low, Medium, High}", "#283593"),
    ]
    # Text colour: dark on light, white on dark
    text_colours = ["#1a237e", "#1a237e", "#1a237e", "white", "white", "white", "white"]

    box_w, box_h = 7.0, 0.95
    gap = 0.55
    n = len(blocks)
    total_h = n * box_h + (n - 1) * gap

    fig, ax = plt.subplots(figsize=(9.0, total_h + 0.9))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, total_h + 0.6)
    ax.axis("off")

    # Draw boxes top-to-bottom; collect (cx, y_bottom, y_top) for each
    y_top = total_h
    centres = []
    for (label, colour), txtcol in zip(blocks, text_colours):
        y_bottom = y_top - box_h
        rect = FancyBboxPatch(
            (1.5, y_bottom), box_w, box_h,
            boxstyle="round,pad=0.04,rounding_size=0.12",
            linewidth=1.0, edgecolor="#1a237e", facecolor=colour,
        )
        ax.add_patch(rect)
        ax.text(1.5 + box_w / 2, y_bottom + box_h / 2, label,
                ha="center", va="center", fontsize=13, color=txtcol,
                wrap=True)
        centres.append((1.5 + box_w / 2, y_bottom, y_bottom + box_h))
        y_top = y_bottom - gap

    # Arrows live only in the gap between consecutive boxes (bottom of upper → top of lower)
    for (cx, y_bot_upper, _), (_, _, y_top_lower) in zip(centres[:-1], centres[1:]):
        arrow = FancyArrowPatch(
            (cx, y_bot_upper - 0.02),     # just below upper box
            (cx, y_top_lower + 0.02),     # just above lower box
            mutation_scale=14, arrowstyle="-|>",
            color="#1a237e", linewidth=1.3,
        )
        ax.add_patch(arrow)

    fig.tight_layout()
    out = os.path.join(FIG_DIR, "architecture_diagram.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"     -> {out}")


# ─────────────────────────────────────────────────────────────────────────
# 3. roc_curves.png
# ─────────────────────────────────────────────────────────────────────────
def make_roc_curves():
    print("  Building roc_curves.png ...")
    import torch
    from preprocess.tokenizer import KmerTokenizer
    from models.transformer_model import TransformerClassifier

    MAX_LEN, K = 994, 6
    D_MODEL, NHEAD, NUM_LAYERS, DIM_FF = 128, 4, 4, 256

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tok = KmerTokenizer(k=K, max_len=MAX_LEN)
    model = TransformerClassifier(
        vocab_size=tok.vocab_size, pad_idx=tok.pad_id,
        max_len=MAX_LEN + 1, num_classes=3,
        d_model=D_MODEL, nhead=NHEAD, num_layers=NUM_LAYERS, dim_ff=DIM_FF,
    ).to(device)
    state = torch.load(os.path.join(CKPT_DIR, "transformer.pt"),
                       map_location=device, weights_only=False)
    model.load_state_dict(state)
    model.eval()

    seqs   = np.load(os.path.join(CKPT_DIR, "transformer_test_seqs.npy"),
                     allow_pickle=True)
    labels = np.load(os.path.join(CKPT_DIR, "transformer_test_labels.npy"))

    # Inference in mini-batches
    BATCH = 32
    probs = np.zeros((len(seqs), 3), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, len(seqs), BATCH):
            batch = seqs[i : i + BATCH]
            ids = np.stack([tok.encode_padded(s, add_cls=True) for s in batch])
            ids_t = torch.from_numpy(ids).long().to(device)
            mask  = (ids_t != 0).long()
            logits = model(ids_t, attention_mask=mask)
            probs[i : i + BATCH] = torch.softmax(logits, dim=-1).cpu().numpy()

    y_true_bin = label_binarize(labels, classes=[0, 1, 2])

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for i, cls in enumerate(LABEL_ORDER):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], probs[:, i])
        auroc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=CLASS_COLOURS[cls], linewidth=2.0,
                label=f"{cls}  (AUROC = {auroc:.3f})")

    ax.plot([0, 1], [0, 1], "--", color="#9ca3af", linewidth=1.0, label="Chance")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("One-vs-rest ROC curves on the held-out test set", pad=12)
    ax.legend(loc="lower right", frameon=False)
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.02)
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    out = os.path.join(FIG_DIR, "roc_curves.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"     -> {out}")


# ─────────────────────────────────────────────────────────────────────────
# 4. Copy pre-existing figures
# ─────────────────────────────────────────────────────────────────────────
def copy_existing():
    pairs = [
        (os.path.join("outputs", "confusion_transformer.png"),
         os.path.join(FIG_DIR, "confusion_matrix.png")),
        (os.path.join("outputs", "rollout_HBA1.png"),
         os.path.join(FIG_DIR, "attention_hba1.png")),
        (os.path.join("outputs", "saliency_HBA1.png"),
         os.path.join(FIG_DIR, "saliency_hba1.png")),
        (os.path.join("assets", "webapp_demo.png"),
         os.path.join(FIG_DIR, "webapp_screenshot.png")),
    ]
    for src, dst in pairs:
        if os.path.exists(src):
            shutil.copy2(src, dst)
            print(f"  Copied  {src}  ->  {dst}")
        else:
            print(f"  Missing source: {src}")


# ─────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Generating report figures into latex/figures/ ...")
    make_class_distribution()
    make_architecture_diagram()
    make_roc_curves()
    copy_existing()
    print("\nDone.")
