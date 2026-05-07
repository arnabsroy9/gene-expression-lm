"""
Gradio web application for gene expression prediction.

Features:
  - Paste a DNA sequence or pick a pre-loaded example
  - Confidence bar chart
  - Attention rollout heatmap
  - Input-gradient saliency heatmap
  - Supports both classification and regression checkpoints

Run locally:
    python app.py
"""

import os
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import gradio as gr

sys.path.insert(0, os.path.dirname(__file__))

from preprocess.tokenizer import KmerTokenizer

CKPT_DIR     = "checkpoints"
MAX_LEN      = 994
K            = 6
D_MODEL      = 128
NHEAD        = 4
NUM_LAYERS   = 4
DIM_FF       = 256
CLASSES      = ["Low", "Medium", "High"]
CLASS_COLORS = ["#3b82f6", "#f59e0b", "#ef4444"]

EXAMPLES = {
    "BRCA1 (tumor suppressor)": (
        "ATGGATTTATCTGCTCTTCGCGTTGAAGAAGTACAAAATGTCATTAATGCTATGCAGAAAATCTTAGAG"
        "TGTCCCATCTGTCTGGAGTTGATCAAGGAACCTGTCTCCACAAAGTGTGACCACATATTTTGCAAATTT"
        "TGCATGCTGAAACTTCTCAACCAGAAGAAAGGGCCTTCACAGTGTCCTTTATGTAAGAATGATATAAC"
        "CAAAAGAGCAGAATTAAAGAAAATAAATGAGAGCTTAGTACAGACAGCTTCAAAGAAATGGCTTCAGG"
    ),
    "TP53 (guardian of the genome)": (
        "ATGGAGGAGCCGCAGTCAGATCCTAGCGTTGAATCGCACTGCCCACGGAGCGGAGCAGCTTCTCCGAG"
        "GACCAGGTCAGCTCCGAAGCCGCAGTCAGATCCTAGCGTTGAATCGCACTGCCCACGGAGCGGAGCAG"
        "CTTCTCCGAGGACCAGGTCAGCTCCGAAGCCGCAGTCAGATCCTAGCGTTGAATCGCACTGCCCACGG"
        "AGCGGAGCAGCTTCTCCGAGGACCAGGTCAGCTCCGAAGCCGCAGTCAGATCCTAGCGTTGAATCGCA"
    ),
    "ACTB (housekeeping - actin)": (
        "ATGGATGATGATATCGCCGCGCTCGTCGTCGACAACGGCTCCGGCATGTGCAAAGCCGGCTTCGCGGG"
        "CGACGATGCCCCGAGGGCCGTCTTCCCCTCCATCGTGGGGCGCCCCAGGCACCAGGGCGTGATGGTGG"
        "GCATGGGTCAGAAGGATTCCTATGTGGGCGACGAGGCCCAGAGCAAGAGAGGCATCCTCACCCTGAAG"
        "TACCCCATCGAGCACGGCATCGTCACCAACTGGGACGACATGGAGAAAATCTGGCACCACACCTTCTA"
    ),
}

_cache: dict = {}


# ── Model loading ──────────────────────────────────────────────────────────────

def _load_task():
    task_file = os.path.join(CKPT_DIR, "transformer_task.txt")
    if os.path.exists(task_file):
        return open(task_file).read().strip()
    return "classification"


def _get_tok():
    if "tok" not in _cache:
        _cache["tok"] = KmerTokenizer(k=K, max_len=MAX_LEN)
    return _cache["tok"]


def _get_model():
    if "model" not in _cache:
        from models.transformer_model import TransformerClassifier
        task = _load_task()
        tag  = "transformer_regression" if task == "regression" else "transformer"
        path = os.path.join(CKPT_DIR, f"{tag}.pt")
        if not os.path.exists(path):
            path = os.path.join(CKPT_DIR, "transformer.pt")

        tok         = _get_tok()
        num_classes = 1 if task == "regression" else 3
        if os.path.exists(path):
            model = TransformerClassifier(
                vocab_size=tok.vocab_size, pad_idx=tok.pad_id,
                max_len=MAX_LEN + 1, num_classes=num_classes,
                d_model=D_MODEL, nhead=NHEAD, num_layers=NUM_LAYERS, dim_ff=DIM_FF,
            )
            model.load_state_dict(torch.load(path, map_location="cpu"))
            model.eval()
            _cache["model"] = model
            _cache["task"]  = task
            if task == "regression":
                _cache["thresholds"] = np.load(os.path.join(CKPT_DIR, "transformer_thresholds.npy"))
        else:
            _cache["model"] = None
    return _cache.get("model"), _cache.get("task", "classification"), _cache.get("thresholds")


# ── Inference helpers ──────────────────────────────────────────────────────────

def _clean_seq(seq: str) -> str:
    return seq.upper().replace(" ", "").replace("\n", "").strip()


def _regression_proba(score, thresholds):
    low_t, high_t = thresholds
    sigma = max((high_t - low_t) / 4.0, 1e-6)
    def _sig(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))
    p_low  = float(_sig((low_t  - score) / sigma))
    p_high = float(_sig((score  - high_t) / sigma))
    p_med  = max(0.0, 1.0 - p_low - p_high)
    arr    = np.array([p_low, p_med, p_high])
    return arr / arr.sum()


# ── Plots ──────────────────────────────────────────────────────────────────────

def _confidence_fig(proba, pred_label):
    fig, ax = plt.subplots(figsize=(5, 3))
    bars = ax.bar(CLASSES, proba, color=CLASS_COLORS, edgecolor="white", linewidth=0.8)
    for bar, p in zip(bars, proba):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{p:.2%}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Probability")
    ax.set_title(f"Prediction: {pred_label}", fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def _heatmap_fig(tokens, scores, title):
    if tokens is None or scores is None or len(tokens) == 0:
        return None
    n      = min(len(tokens), 80)
    tokens = tokens[:n]
    scores = scores[:n]

    norm   = mcolors.Normalize(vmin=0, vmax=scores.max() + 1e-9)
    cmap   = plt.cm.YlOrRd
    colors = cmap(norm(scores))

    fig_w  = max(12, n * 0.35)
    fig, ax = plt.subplots(figsize=(fig_w, 2.5))
    for i, (tok, col) in enumerate(zip(tokens, colors)):
        ax.add_patch(plt.Rectangle((i, 0), 1, 1, color=col))
        ax.text(i + 0.5, 0.5, tok, ha="center", va="center",
                fontsize=5.5, rotation=90, color="black")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    plt.colorbar(sm, ax=ax, orientation="horizontal", pad=0.05, shrink=0.6)
    ax.set_xlim(0, n)
    ax.set_ylim(0, 1)
    ax.set_title(f"{title} (first {n} 6-mers)", fontweight="bold")
    ax.axis("off")
    fig.tight_layout()
    return fig


# ── Main prediction function ───────────────────────────────────────────────────

def run_prediction(sequence_input, example_choice):
    seq = _clean_seq(sequence_input if sequence_input.strip() else
                     (EXAMPLES.get(example_choice, "") if example_choice else ""))

    if not seq:
        return "Please enter or select a DNA sequence.", None, None, None

    valid = set("ACGT")
    if not all(c in valid for c in seq):
        bad = sorted({c for c in seq if c not in valid})
        return f"Invalid characters: {bad}. Only A, C, G, T are allowed.", None, None, None

    model, task, thresholds = _get_model()
    if model is None:
        return "Transformer checkpoint not found. Run train.py first.", None, None, None

    tok    = _get_tok()
    ids_np = tok.encode_padded(seq, add_cls=True)
    ids    = torch.from_numpy(ids_np).unsqueeze(0).long()
    mask   = (ids != 0).long()

    # Classification or regression output
    if task == "regression":
        with torch.no_grad():
            raw = model(ids, attention_mask=mask)[0, 0].item()
        proba      = _regression_proba(raw, thresholds)
        pred_label = CLASSES[int(proba.argmax())]
        score_str  = f" (log-TPM: {raw:.3f})"
    else:
        with torch.no_grad():
            logits = model(ids, attention_mask=mask)
        proba      = torch.softmax(logits, dim=-1).squeeze().numpy()
        pred_label = CLASSES[int(proba.argmax())]
        score_str  = ""

    status = (
        f"**Predicted expression level: {pred_label}**{score_str}\n\n"
        f"Sequence length: {len(seq)} bp  |  Task: {task}  |  K-mer: {K}"
    )

    conf_fig = _confidence_fig(proba, pred_label)

    kmers = tok.tokenize(seq)
    n     = len(kmers)

    # Attention rollout
    rollout     = model.get_attention_rollout(ids, mask)
    rollout_fig = _heatmap_fig(kmers, rollout[:n], "Attention Rollout")

    # Input-gradient saliency
    saliency     = model.get_input_saliency(ids, mask)   # includes CLS
    saliency_fig = _heatmap_fig(kmers, saliency[1:n + 1], "Input Gradient Saliency")

    return status, conf_fig, rollout_fig, saliency_fig


def fill_example(choice):
    return EXAMPLES.get(choice, "")


# ── Gradio UI ──────────────────────────────────────────────────────────────────

with gr.Blocks(title="Gene Expression Predictor") as demo:
    gr.Markdown(
        """
# Gene Expression Predictor
Predict whether a human gene sequence has **High**, **Medium**, or **Low** expression
using an encoder-only Transformer trained on 6-mer tokenized DNA sequences.
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            example_dd = gr.Dropdown(
                label="Pre-loaded example genes",
                choices=[""] + list(EXAMPLES.keys()),
                value="",
            )
            seq_input = gr.Textbox(
                label="DNA sequence (A/C/G/T)",
                placeholder="Paste your sequence here, or pick an example above...",
                lines=6,
            )
            submit_btn = gr.Button("Predict", variant="primary")

        with gr.Column(scale=2):
            status_md    = gr.Markdown()
            conf_plot    = gr.Plot(label="Confidence")
            rollout_plot = gr.Plot(label="Attention Rollout")
            saliency_plot = gr.Plot(label="Input Gradient Saliency")

    example_dd.change(fill_example, inputs=example_dd, outputs=seq_input)

    submit_btn.click(
        fn=run_prediction,
        inputs=[seq_input, example_dd],
        outputs=[status_md, conf_plot, rollout_plot, saliency_plot],
    )

    gr.Markdown(
        """
---
**Note:** Coding sequences carry limited expression information -- regulatory elements
(promoters, enhancers) that drive expression typically lie outside the coding region.
Moderate accuracy is a scientifically expected finding, not a model failure.
        """
    )

if __name__ == "__main__":
    demo.launch(share=False, theme=gr.themes.Soft())
