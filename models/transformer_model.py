"""
Encoder-only Transformer for gene expression classification / regression.

Architecture:
    Token Embedding + Sinusoidal Positional Encoding
    -> N x TransformerEncoderLayer (pre-LN)
    -> CLS token representation
    -> LayerNorm -> Dropout -> Linear -> num_classes outputs

num_classes=3  : classification (Low / Medium / High)
num_classes=1  : regression     (log-TPM scalar)
"""

import math
import torch
import torch.nn as nn
import numpy as np


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5_000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10_000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class TransformerClassifier(nn.Module):
    def __init__(
        self,
        vocab_size:  int,
        d_model:     int   = 128,
        nhead:       int   = 4,
        num_layers:  int   = 3,
        dim_ff:      int   = 256,
        num_classes: int   = 3,
        max_len:     int   = 5_000,
        dropout:     float = 0.1,
        pad_idx:     int   = 0,
        aux_specs:   list  = None,    # list of (name, "binary" | "regression") tuples
    ):
        super().__init__()
        self.pad_idx    = pad_idx
        self.num_classes = num_classes
        self.embedding  = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_enc    = PositionalEncoding(d_model, max_len=max_len, dropout=dropout)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder    = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.norm       = nn.LayerNorm(d_model)
        self.dropout    = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_model, num_classes)

        # Optional auxiliary task heads (multi-task learning).
        # Each head reads the same CLS representation and outputs a single scalar.
        # Binary tasks → BCE-with-logits at training time; regression → MSE.
        self.aux_specs = list(aux_specs) if aux_specs else []
        self.aux_heads = nn.ModuleDict({
            name: nn.Linear(d_model, 1) for name, _ in self.aux_specs
        })

    def _encode_cls(self, input_ids, attention_mask=None):
        """Run encoder, return the CLS-token representation (B, d_model)."""
        src_key_padding_mask = None
        if attention_mask is not None:
            src_key_padding_mask = (attention_mask == 0)
        emb = self.pos_enc(self.embedding(input_ids))
        out = self.encoder(emb, src_key_padding_mask=src_key_padding_mask)
        return self.norm(out[:, 0, :])

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None):
        cls_repr = self._encode_cls(input_ids, attention_mask)
        return self.classifier(self.dropout(cls_repr))

    def forward_multi(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None):
        """Multi-task forward: returns dict with main logits + each aux head's output."""
        cls_repr = self._encode_cls(input_ids, attention_mask)
        out = {"main": self.classifier(self.dropout(cls_repr))}
        for name, head in self.aux_heads.items():
            out[name] = head(cls_repr).squeeze(-1)   # (B,)
        return out

    # ── Interpretability ────────────────────────────────────────────────────────

    def _padding_mask(self, attention_mask):
        if attention_mask is not None:
            return (attention_mask == 0)
        return None

    def get_attention_weights(self, input_ids, attention_mask=None):
        """Last-layer attention. Shape: (B, nhead, L, L)."""
        km = self._padding_mask(attention_mask)
        with torch.no_grad():
            x = self.pos_enc(self.embedding(input_ids))
            for layer in self.encoder.layers[:-1]:
                x = layer(x, src_key_padding_mask=km)
            last   = self.encoder.layers[-1]
            x_norm = last.norm1(x)
            _, attn = last.self_attn(
                x_norm, x_norm, x_norm,
                key_padding_mask=km,
                need_weights=True,
                average_attn_weights=False,
            )
        return attn

    def _all_layer_attention(self, input_ids, attention_mask=None):
        """Attention averaged over heads for every layer. Returns list of (B, L, L)."""
        km = self._padding_mask(attention_mask)
        all_attn = []
        with torch.no_grad():
            x = self.pos_enc(self.embedding(input_ids))
            for layer in self.encoder.layers:
                x_norm = layer.norm1(x)
                _, attn = layer.self_attn(
                    x_norm, x_norm, x_norm,
                    key_padding_mask=km,
                    need_weights=True,
                    average_attn_weights=True,  # (B, L, L)
                )
                all_attn.append(attn)
                x = layer(x, src_key_padding_mask=km)
        return all_attn

    def get_attention_rollout(self, input_ids, attention_mask=None) -> np.ndarray:
        """
        Attention rollout (Abnar & Zuidema 2020).
        Propagates attention across all layers with residual correction.
        Returns CLS-to-token relevance scores, shape (seq_len,).
        """
        all_attn = self._all_layer_attention(input_ids, attention_mask)
        device   = all_attn[0].device
        L        = all_attn[0].shape[-1]
        rollout  = torch.eye(L, device=device).unsqueeze(0)  # (1, L, L)

        for attn in all_attn:
            a = attn[0:1]                                              # (1, L, L)
            a = 0.5 * a + 0.5 * torch.eye(L, device=device).unsqueeze(0)
            a = a / a.sum(dim=-1, keepdim=True).clamp(min=1e-9)
            rollout = torch.bmm(rollout, a)

        # Row 0 = CLS; skip CLS-to-CLS position
        return rollout[0, 0, 1:].cpu().numpy()  # (L-1,)

    def get_input_saliency(self, input_ids, attention_mask=None) -> np.ndarray:
        """
        Input x gradient saliency.
        Call with model in eval mode and outside torch.no_grad().
        Returns per-token scores, shape (L,) including CLS at pos 0.
        """
        emb_ref = []

        def _hook(module, inp, out):
            out.retain_grad()
            emb_ref.append(out)

        handle = self.embedding.register_forward_hook(_hook)
        logits  = self.forward(input_ids, attention_mask)
        handle.remove()

        if self.num_classes > 1:
            score = logits[0, logits.argmax(dim=-1).item()]
        else:
            score = logits[0, 0]

        score.backward()

        emb      = emb_ref[0]                          # (1, L, D)
        saliency = (emb.grad * emb).norm(dim=-1)[0]   # (L,)
        return saliency.detach().cpu().numpy()
