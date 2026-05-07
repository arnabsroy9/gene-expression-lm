"""
Evaluate the saved Transformer checkpoint on the test set.

Outputs:
    outputs/eval_results.csv
    outputs/confusion_transformer.png

Usage:
    python evaluate.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch

from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, confusion_matrix, classification_report
)

from preprocess.tokenizer import KmerTokenizer

CKPT_DIR    = "checkpoints"
OUT_DIR     = "outputs"
MAX_LEN     = 994
K           = 6
CLASSES     = ["Low", "Medium", "High"]
EVAL_BATCH  = 32

D_MODEL    = 128
NHEAD      = 4
NUM_LAYERS = 4
DIM_FF     = 256

os.makedirs(OUT_DIR, exist_ok=True)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _plot_cm(cm):
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASSES, yticklabels=CLASSES, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix - Transformer")
    path = os.path.join(OUT_DIR, "confusion_transformer.png")
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved confusion matrix -> {path}")


def _load_task():
    task_file = os.path.join(CKPT_DIR, "transformer_task.txt")
    if os.path.exists(task_file):
        return open(task_file).read().strip()
    return "classification"


def _regression_to_proba(pred_scores, thresholds):
    """Convert continuous regression scores to soft class probabilities."""
    low_t, high_t = thresholds
    sigma = max((high_t - low_t) / 4.0, 1e-6)

    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))

    p_low  = sigmoid((low_t  - pred_scores) / sigma)
    p_high = sigmoid((pred_scores - high_t) / sigma)
    p_med  = np.clip(1.0 - p_low - p_high, 0, 1)
    stack  = np.stack([p_low, p_med, p_high], axis=1)
    return stack / stack.sum(axis=1, keepdims=True).clip(min=1e-9)


def main():
    from models.transformer_model import TransformerClassifier

    task = _load_task()
    is_regression = (task == "regression")
    tag = "transformer_regression" if is_regression else "transformer"

    ckpt = os.path.join(CKPT_DIR, f"{tag}.pt")
    if not os.path.exists(ckpt):
        # Fall back to the default name
        ckpt = os.path.join(CKPT_DIR, "transformer.pt")
        if not os.path.exists(ckpt):
            print("Checkpoint not found:", ckpt)
            return

    tok       = KmerTokenizer(k=K, max_len=MAX_LEN)
    seqs      = np.load(os.path.join(CKPT_DIR, f"{tag}_test_seqs.npy"), allow_pickle=True).tolist()
    y_test    = np.load(os.path.join(CKPT_DIR, f"{tag}_test_labels.npy"))
    X_test    = tok.batch_encode_padded(seqs, add_cls=True)

    num_classes = 1 if is_regression else 3
    device = get_device()
    model  = TransformerClassifier(
        vocab_size=tok.vocab_size, pad_idx=tok.pad_id,
        max_len=MAX_LEN + 1, num_classes=num_classes,
        d_model=D_MODEL, nhead=NHEAD, num_layers=NUM_LAYERS, dim_ff=DIM_FF,
    ).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    # Batched inference
    raw_preds = []
    ids_t     = torch.from_numpy(X_test).long()
    for i in range(0, len(ids_t), EVAL_BATCH):
        batch_ids  = ids_t[i : i + EVAL_BATCH].to(device)
        batch_mask = (batch_ids != 0).long()
        with torch.no_grad():
            out = model(batch_ids, attention_mask=batch_mask)
        raw_preds.append(out.cpu())
    raw_preds = torch.cat(raw_preds, dim=0).numpy()

    if is_regression:
        thresholds = np.load(os.path.join(CKPT_DIR, "transformer_thresholds.npy"))
        pred_scores = raw_preds.squeeze(-1)
        y_proba = _regression_to_proba(pred_scores, thresholds)
        y_pred  = np.full(len(pred_scores), 1, dtype=int)
        y_pred[pred_scores <= thresholds[0]] = 0
        y_pred[pred_scores >  thresholds[1]] = 2
        # For regression the "true" labels are log-TPM; need class labels for metrics
        # Recompute class labels from test log-TPM using the same thresholds
        y_true = np.full(len(y_test), 1, dtype=int)
        y_true[y_test <= thresholds[0]] = 0
        y_true[y_test >  thresholds[1]] = 2
    else:
        y_proba = torch.softmax(torch.from_numpy(raw_preds), dim=-1).numpy()
        y_pred  = y_proba.argmax(axis=1)
        y_true  = y_test.astype(int)

    acc   = accuracy_score(y_true, y_pred)
    f1    = f1_score(y_true, y_pred, average="macro", zero_division=0)
    try:
        auroc = roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
    except Exception:
        auroc = float("nan")

    print(f"\n{'='*50}")
    print(f"Model: Transformer ({task})")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  Macro F1 : {f1:.4f}")
    print(f"  AUROC    : {auroc:.4f}")
    print(classification_report(y_true, y_pred, target_names=CLASSES, zero_division=0))

    _plot_cm(confusion_matrix(y_true, y_pred))

    df = pd.DataFrame([{"model": f"transformer_{task}", "accuracy": acc, "macro_f1": f1, "auroc": auroc}])
    out_csv = os.path.join(OUT_DIR, "eval_results.csv")
    df.to_csv(out_csv, index=False)
    print(f"\nResults saved -> {out_csv}")


if __name__ == "__main__":
    main()
