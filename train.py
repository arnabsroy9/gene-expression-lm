"""
Training entry point for the Transformer model.

Improvements applied:
  - 6-mer tokenization (richer biological context per token)
  - Reverse-complement augmentation (strand invariance, p=0.5 per sample)
  - Label smoothing = 0.1  (classification only)
  - Class-weighted loss    (classification only)
  - Linear warmup (2 epochs) + cosine annealing
  - Regression mode: predict log-TPM with Huber loss, bin at eval time

Usage:
    python train.py                                          # original CDS dataset
    python train.py --data data/labeled_genes_xpresso.csv    # alternate dataset
    python train.py --task regression
    python train.py --epochs 30 --batch_size 32
    python train.py --max_len 2000                           # override auto-computed sequence length
"""

import os
import math
import random
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from preprocess.tokenizer import KmerTokenizer, LABEL_MAP

DEFAULT_DATA_CSV = os.path.join("data", "labeled_genes.csv")
CKPT_DIR         = "checkpoints"
K                = 6
NUM_CLASSES      = 3
WARMUP_EPOCHS    = 3

# Model hyperparameters
D_MODEL    = 128
NHEAD      = 4
NUM_LAYERS = 4
DIM_FF     = 256

# Memory budget: attention is O(L²); warn when batch×L² exceeds this threshold
_ATTN_WARN_TOKENS = 2_000

RC_TABLE = str.maketrans("ACGT", "TGCA")


def reverse_complement(seq: str) -> str:
    return seq.translate(RC_TABLE)[::-1]


# ── Dataset ────────────────────────────────────────────────────────────────────

class DNADataset:
    def __init__(self, seqs, labels, tokenizer, add_cls=True, augment=False, is_float=False):
        self.seqs     = seqs
        self.labels   = labels
        self.tok      = tokenizer
        self.add_cls  = add_cls
        self.augment  = augment
        self.is_float = is_float

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        import torch
        seq = self.seqs[idx]
        if self.augment and random.random() < 0.5:
            seq = reverse_complement(seq)
        ids  = torch.from_numpy(self.tok.encode_padded(seq, add_cls=self.add_cls)).long()
        mask = (ids != 0).long()
        dtype = torch.float if self.is_float else torch.long
        lbl  = torch.tensor(self.labels[idx], dtype=dtype)
        return ids, mask, lbl


def make_loader(dataset, batch_size, shuffle):
    import torch
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)


# ── Scheduler ─────────────────────────────────────────────────────────────────

def make_scheduler(optimizer, warmup_epochs, total_epochs):
    import torch
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / max(1, warmup_epochs)
        t = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * t))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ── Training loops ─────────────────────────────────────────────────────────────

def _run_epoch_clf(model, loader, criterion, optimizer, device, train=True, desc=""):
    import torch
    import torch.nn as nn
    model.train() if train else model.eval()
    total_loss, correct, total = 0.0, 0, 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    n_batches = len(loader)
    log_every = max(1, n_batches // 10)   # ~10 progress prints per epoch
    with ctx:
        for i, (ids, mask, labels) in enumerate(loader, start=1):
            ids, mask, labels = ids.to(device), mask.to(device), labels.to(device)
            logits = model(ids, attention_mask=mask)
            loss   = criterion(logits, labels)
            if train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            total_loss += loss.item() * len(labels)
            correct    += (logits.argmax(1) == labels).sum().item()
            total      += len(labels)
            if i % log_every == 0 or i == n_batches:
                print(f"  {desc} batch {i:>5d}/{n_batches}  "
                      f"loss={total_loss/total:.4f}  acc={correct/total:.4f}",
                      flush=True)
    return total_loss / total, correct / total


def _run_epoch_reg(model, loader, criterion, optimizer, device, train=True, desc=""):
    import torch
    import torch.nn as nn
    model.train() if train else model.eval()
    total_loss, total = 0.0, 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    n_batches = len(loader)
    log_every = max(1, n_batches // 10)
    with ctx:
        for i, (ids, mask, targets) in enumerate(loader, start=1):
            ids, mask, targets = ids.to(device), mask.to(device), targets.to(device)
            preds = model(ids, attention_mask=mask).squeeze(-1)
            loss  = criterion(preds, targets)
            if train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            total_loss += loss.item() * len(targets)
            total      += len(targets)
            if i % log_every == 0 or i == n_batches:
                print(f"  {desc} batch {i:>5d}/{n_batches}  "
                      f"loss={total_loss/total:.4f}",
                      flush=True)
    return total_loss / total


# ── Data loading ───────────────────────────────────────────────────────────────

def _auto_max_len(data_csv: str) -> int:
    """Compute max k-mer token length from the actual sequences in the dataset."""
    df = pd.read_csv(data_csv, usecols=["sequence"]).dropna()
    max_bp = int(df["sequence"].str.len().max())
    return max_bp - K + 1   # number of 6-mers per sequence


def _load_splits(task, data_csv: str):
    df = pd.read_csv(data_csv).dropna(subset=["sequence", "expression_label"])
    df["class_label"] = df["expression_label"].map(LABEL_MAP)
    df = df.dropna(subset=["class_label"]).copy()

    # Auto-detect classes present (binary datasets skip "Medium") and remap
    # to a contiguous range 0..K-1 so CrossEntropyLoss/num_classes work cleanly.
    unique_labels = sorted(df["class_label"].astype(int).unique())
    remap = {old: new for new, old in enumerate(unique_labels)}
    df["class_label"] = df["class_label"].astype(int).map(remap)
    num_classes = len(unique_labels)

    inverse = {v: k for k, v in LABEL_MAP.items()}
    label_names = [inverse[old] for old in unique_labels]
    print(f"Detected {num_classes}-class problem: {label_names} → {list(range(num_classes))}")

    seqs         = df["sequence"].tolist()
    class_labels = df["class_label"].astype(int).tolist()

    if task == "regression":
        df = df.dropna(subset=["log_tpm"])
        targets = df["log_tpm"].astype(float).tolist()
    else:
        targets = class_labels

    X_tv, X_test, y_tv, y_test = train_test_split(
        seqs, targets, test_size=0.20, stratify=class_labels, random_state=42)
    _, _, cl_tv, _ = train_test_split(
        seqs, class_labels, test_size=0.20, stratify=class_labels, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv, test_size=0.25, stratify=cl_tv, random_state=42)

    print(f"Split sizes -- train: {len(X_train)}, val: {len(X_val)}, test: {len(X_test)}")
    return X_train, X_val, X_test, y_train, y_val, y_test, num_classes


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    import torch
    import torch.nn as nn
    from torch.optim import AdamW
    from models.transformer_model import TransformerClassifier

    p = argparse.ArgumentParser()
    p.add_argument("--task",       choices=["classification", "regression"], default="classification")
    p.add_argument("--epochs",     type=int,   default=30)
    p.add_argument("--batch_size", type=int,   default=16)
    p.add_argument("--lr",         type=float, default=1e-3)
    p.add_argument("--data",       default=DEFAULT_DATA_CSV,
                   help="Labeled CSV to train on (default: labeled_genes.csv)")
    p.add_argument("--max_len",    type=int, default=-1,
                   help="Token sequence length cap; -1 = auto-compute from data")
    args = p.parse_args()

    # Compute MAX_LEN from data unless the user overrides it
    if args.max_len > 0:
        MAX_LEN = args.max_len
    else:
        print(f"Auto-computing MAX_LEN from {args.data} ...")
        MAX_LEN = _auto_max_len(args.data)

    print(f"MAX_LEN = {MAX_LEN} tokens  (K={K}, covers up to {MAX_LEN + K - 1} bp)")

    if MAX_LEN > _ATTN_WARN_TOKENS and args.batch_size > 8:
        print(
            f"WARNING: MAX_LEN={MAX_LEN} with batch_size={args.batch_size} requires large attention "
            f"matrices (O(L²)). Consider --batch_size 4 to avoid OOM errors."
        )

    # Derive checkpoint tag from the data filename so runs don't overwrite each other
    data_stem = os.path.splitext(os.path.basename(args.data))[0]  # e.g. "labeled_genes_tss"
    ckpt_tag  = "transformer" if data_stem == "labeled_genes" else f"transformer_{data_stem}"

    tok = KmerTokenizer(k=K, max_len=MAX_LEN)
    X_train, X_val, X_test, y_train, y_val, y_test, detected_classes = _load_splits(args.task, args.data)

    is_regression = (args.task == "regression")
    num_classes   = 1 if is_regression else detected_classes

    train_ds = DNADataset(X_train, y_train, tok, add_cls=True, augment=True,  is_float=is_regression)
    val_ds   = DNADataset(X_val,   y_val,   tok, add_cls=True, augment=False, is_float=is_regression)

    train_loader = make_loader(train_ds, args.batch_size, shuffle=True)
    val_loader   = make_loader(val_ds,   args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = TransformerClassifier(
        vocab_size=tok.vocab_size, pad_idx=tok.pad_id,
        max_len=MAX_LEN + 1,   # +1 for the CLS token
        num_classes=num_classes,
        d_model=D_MODEL, nhead=NHEAD, num_layers=NUM_LAYERS, dim_ff=DIM_FF,
    ).to(device)

    opt   = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = make_scheduler(opt, warmup_epochs=WARMUP_EPOCHS, total_epochs=args.epochs)

    if is_regression:
        crit = nn.HuberLoss(delta=1.0)
    else:
        # Class weights from training label distribution
        counts  = np.bincount(np.array(y_train, dtype=int), minlength=num_classes).astype(float)
        weights = torch.tensor(1.0 / counts, dtype=torch.float).to(device)
        weights = weights / weights.sum() * num_classes
        crit = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1)

    if is_regression:
        ckpt_tag = f"{ckpt_tag}_regression"
    print(f"[Transformer] task={args.task}  data={args.data}  vocab_size={tok.vocab_size}  K={K}  max_len={MAX_LEN}")

    if is_regression:
        best_val, best_state = float("inf"), None
        for epoch in range(1, args.epochs + 1):
            print(f"\n[Transformer] === Epoch {epoch:02d}/{args.epochs} ===", flush=True)
            tr_loss = _run_epoch_reg(model, train_loader, crit, opt, device, train=True,  desc=f"Ep{epoch:02d} train")
            vl_loss = _run_epoch_reg(model, val_loader,   crit, opt, device, train=False, desc=f"Ep{epoch:02d} val  ")
            sched.step()
            print(f"[Transformer] Ep {epoch:02d}  tr_loss={tr_loss:.4f}  val_loss={vl_loss:.4f}", flush=True)
            if vl_loss < best_val:
                best_val   = vl_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        # Compute and save binning thresholds from training targets
        y_arr = np.array(y_train, dtype=float)
        thresholds = np.array([np.percentile(y_arr, 33), np.percentile(y_arr, 67)])
        os.makedirs(CKPT_DIR, exist_ok=True)
        np.save(os.path.join(CKPT_DIR, f"{ckpt_tag}_thresholds.npy"), thresholds)
        print(f"[Transformer] Thresholds: low<={thresholds[0]:.3f}, high>{thresholds[1]:.3f}", flush=True)
    else:
        best_val, best_state = 0.0, None
        for epoch in range(1, args.epochs + 1):
            print(f"\n[Transformer] === Epoch {epoch:02d}/{args.epochs} ===", flush=True)
            tr_loss, tr_acc = _run_epoch_clf(model, train_loader, crit, opt, device, train=True,  desc=f"Ep{epoch:02d} train")
            vl_loss, vl_acc = _run_epoch_clf(model, val_loader,   crit, opt, device, train=False, desc=f"Ep{epoch:02d} val  ")
            sched.step()
            print(f"[Transformer] Ep {epoch:02d}  tr_loss={tr_loss:.4f} tr_acc={tr_acc:.4f}  val_loss={vl_loss:.4f} val_acc={vl_acc:.4f}", flush=True)
            if vl_acc > best_val:
                best_val   = vl_acc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    os.makedirs(CKPT_DIR, exist_ok=True)
    torch.save(best_state, os.path.join(CKPT_DIR, f"{ckpt_tag}.pt"))
    np.save(os.path.join(CKPT_DIR, f"{ckpt_tag}_test_seqs.npy"),   np.array(X_test, dtype=object))
    np.save(os.path.join(CKPT_DIR, f"{ckpt_tag}_test_labels.npy"), np.array(y_test))

    # Save task type so evaluate.py and app.py know which mode was trained
    with open(os.path.join(CKPT_DIR, f"{ckpt_tag}_task.txt"), "w") as f:
        f.write(args.task)

    result_str = f"Best val loss: {best_val:.4f}" if is_regression else f"Best val acc: {best_val:.4f}"
    print(f"[Transformer] {result_str}  checkpoint -> checkpoints/{ckpt_tag}.pt", flush=True)


if __name__ == "__main__":
    main()
