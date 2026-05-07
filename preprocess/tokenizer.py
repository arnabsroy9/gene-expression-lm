"""
K-mer tokenization and train/val/test splitting shared across all models.

Usage:
    from preprocess.tokenizer import KmerTokenizer, load_splits

    tok = KmerTokenizer(k=3, max_len=512)
    tokens = tok.tokenize("ATCGATCG")          # list of str k-mers
    ids    = tok.encode("ATCGATCG")            # list of int IDs
    padded = tok.encode_padded("ATCGATCG")     # numpy array of length max_len
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import List, Tuple, Dict

LABEL_MAP = {"Low": 0, "Medium": 1, "High": 2}
IDX_MAP   = {v: k for k, v in LABEL_MAP.items()}

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
CLS_TOKEN = "<CLS>"


class KmerTokenizer:
    def __init__(self, k: int = 3, max_len: int = 512, stride: int = 1):
        self.k       = k
        self.max_len = max_len
        self.stride  = stride
        self.vocab: Dict[str, int] = {}
        self._build_vocab()

    def _build_vocab(self) -> None:
        special = [PAD_TOKEN, UNK_TOKEN, CLS_TOKEN]
        bases   = "ACGT"
        kmers   = [
            "".join([bases[i >> (2 * j) & 3] for j in range(self.k - 1, -1, -1)])
            for i in range(4 ** self.k)
        ]
        all_tokens = special + sorted(kmers)
        self.vocab = {tok: idx for idx, tok in enumerate(all_tokens)}

    @property
    def pad_id(self) -> int:
        return self.vocab[PAD_TOKEN]

    @property
    def unk_id(self) -> int:
        return self.vocab[UNK_TOKEN]

    @property
    def cls_id(self) -> int:
        return self.vocab[CLS_TOKEN]

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def tokenize(self, sequence: str) -> List[str]:
        seq = sequence.upper().replace(" ", "")
        return [
            seq[i : i + self.k]
            for i in range(0, len(seq) - self.k + 1, self.stride)
        ]

    def encode(self, sequence: str, add_cls: bool = False) -> List[int]:
        kmers = self.tokenize(sequence)
        ids   = [self.vocab.get(km, self.unk_id) for km in kmers]
        if add_cls:
            ids = [self.cls_id] + ids
        return ids

    def encode_padded(self, sequence: str, add_cls: bool = False) -> np.ndarray:
        ids = self.encode(sequence, add_cls=add_cls)
        # Truncate
        ids = ids[: self.max_len]
        # Pad
        pad_len = self.max_len - len(ids)
        ids = ids + [self.pad_id] * pad_len
        return np.array(ids, dtype=np.int64)

    def batch_encode_padded(self, sequences: List[str], add_cls: bool = False) -> np.ndarray:
        return np.stack([self.encode_padded(s, add_cls=add_cls) for s in sequences])

    def kmers_of(self, sequence: str) -> List[str]:
        """Return k-mer list for a sequence (useful for N-gram features)."""
        return self.tokenize(sequence)

    def kmer_freqs(self, sequence: str) -> Dict[str, float]:
        """Normalised k-mer frequency dictionary (for N-gram baseline)."""
        kmers = self.tokenize(sequence)
        if not kmers:
            return {}
        counts: Dict[str, int] = {}
        for km in kmers:
            counts[km] = counts.get(km, 0) + 1
        total = len(kmers)
        return {km: cnt / total for km, cnt in counts.items()}


def load_splits(
    csv_path: str,
    tokenizer: KmerTokenizer,
    test_size: float = 0.20,
    val_size:  float = 0.25,
    random_state: int = 42,
    add_cls: bool = False,
) -> Tuple:
    """
    Load labeled_genes.csv and return train/val/test splits as numpy arrays.

    Returns:
        X_train, X_val, X_test   : np.ndarray of shape (N, max_len)
        y_train, y_val, y_test   : np.ndarray of int labels
        sequences_test           : list of raw sequences (for attention viz)
    """
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["sequence", "expression_label"])
    df["label"] = df["expression_label"].map(LABEL_MAP)
    df = df.dropna(subset=["label"])

    seqs   = df["sequence"].tolist()
    labels = df["label"].astype(int).tolist()

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        seqs, labels, test_size=test_size, random_state=random_state, stratify=labels
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=val_size, random_state=random_state, stratify=y_trainval
    )

    print(f"Split sizes — train: {len(X_train)}, val: {len(X_val)}, test: {len(X_test)}")

    X_train_enc = tokenizer.batch_encode_padded(X_train, add_cls=add_cls)
    X_val_enc   = tokenizer.batch_encode_padded(X_val,   add_cls=add_cls)
    X_test_enc  = tokenizer.batch_encode_padded(X_test,  add_cls=add_cls)

    return (
        X_train_enc, X_val_enc, X_test_enc,
        np.array(y_train), np.array(y_val), np.array(y_test),
        X_test,
    )