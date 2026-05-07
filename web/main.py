"""
FastAPI backend for Gene Expression Predictor.

Usage:
    python web/main.py
    uvicorn web.main:app --reload   (from project root)
"""

import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from preprocess.tokenizer import KmerTokenizer
from models.transformer_model import TransformerClassifier

# ── Config ─────────────────────────────────────────────────────────────────────

CKPT_DIR    = os.path.join(os.path.dirname(os.path.dirname(__file__)), "checkpoints")
MAX_LEN     = 994
K           = 6
CLASSES     = ["Low", "Medium", "High"]
D_MODEL     = 128
NHEAD       = 4
NUM_LAYERS  = 4
DIM_FF      = 256

app = FastAPI(title="Gene Expression Predictor")

# ── Model loading (once at startup) ───────────────────────────────────────────

_model      = None
_tok        = None
_task       = None
_thresholds = None


def _load_task():
    task_file = os.path.join(CKPT_DIR, "transformer_task.txt")
    if os.path.exists(task_file):
        return open(task_file).read().strip()
    return "classification"


def _get_model():
    global _model, _tok, _task, _thresholds
    if _model is not None:
        return

    _task = _load_task()
    tag   = "transformer_regression" if _task == "regression" else "transformer"
    ckpt  = os.path.join(CKPT_DIR, f"{tag}.pt")
    if not os.path.exists(ckpt):
        ckpt = os.path.join(CKPT_DIR, "transformer.pt")
    if not os.path.exists(ckpt):
        raise RuntimeError(f"Checkpoint not found: {ckpt}")

    _tok        = KmerTokenizer(k=K, max_len=MAX_LEN)
    num_classes = 1 if _task == "regression" else 3
    _model      = TransformerClassifier(
        vocab_size=_tok.vocab_size, pad_idx=_tok.pad_id,
        max_len=MAX_LEN + 1, num_classes=num_classes,
        d_model=D_MODEL, nhead=NHEAD, num_layers=NUM_LAYERS, dim_ff=DIM_FF,
    )
    _model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    _model.eval()

    if _task == "regression":
        _thresholds = np.load(os.path.join(CKPT_DIR, "transformer_thresholds.npy"))

    print(f"[startup] Model loaded  task={_task}  vocab={_tok.vocab_size}  K={K}")


@app.on_event("startup")
def startup():
    try:
        _get_model()
    except Exception as e:
        print(f"[startup] Warning: could not load model: {e}")


# ── Helpers ────────────────────────────────────────────────────────────────────

def _regression_proba(score, thresholds):
    low_t, high_t = thresholds
    sigma = max((high_t - low_t) / 4.0, 1e-6)
    def _sig(x):
        return 1.0 / (1.0 + np.exp(-np.clip(float(x), -50, 50)))
    p_low  = _sig((low_t  - score) / sigma)
    p_high = _sig((score  - high_t) / sigma)
    p_med  = max(0.0, 1.0 - p_low - p_high)
    arr    = np.array([p_low, p_med, p_high])
    return arr / arr.sum()


# ── API ────────────────────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    sequence: str


class PredictResponse(BaseModel):
    predicted_class: str
    probabilities:   dict       # {"Low": float, "Medium": float, "High": float}
    rollout:         list       # per-kmer attention rollout scores
    saliency:        list       # per-kmer saliency scores
    kmers:           list       # kmer strings for labelling the heatmaps
    task:            str
    score:           float | None   # log-TPM if regression, else null


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    _get_model()   # no-op if already loaded
    seq = req.sequence.upper().replace(" ", "").replace("\n", "").strip()

    if not seq:
        raise HTTPException(400, "Empty sequence")

    invalid = sorted({c for c in seq if c not in "ACGT"})
    if invalid:
        raise HTTPException(400, f"Invalid characters: {invalid}. Only A, C, G, T allowed.")

    ids_np = _tok.encode_padded(seq, add_cls=True)
    ids    = torch.from_numpy(ids_np).unsqueeze(0).long()
    mask   = (ids != 0).long()

    # Classification or regression
    if _task == "regression":
        with torch.no_grad():
            raw_score = _model(ids, attention_mask=mask)[0, 0].item()
        proba = _regression_proba(raw_score, _thresholds)
    else:
        with torch.no_grad():
            logits = _model(ids, attention_mask=mask)
        proba     = torch.softmax(logits, dim=-1).squeeze().numpy()
        raw_score = None

    pred_class = CLASSES[int(proba.argmax())]

    # Attention rollout
    rollout = _model.get_attention_rollout(ids, mask).tolist()

    # Input-gradient saliency
    saliency_full = _model.get_input_saliency(ids, mask)  # includes CLS at [0]

    kmers   = _tok.tokenize(seq)
    n       = len(kmers)
    rollout  = rollout[:n]
    saliency = saliency_full[1:n + 1].tolist()

    return PredictResponse(
        predicted_class = pred_class,
        probabilities   = {c: float(p) for c, p in zip(CLASSES, proba)},
        rollout         = rollout,
        saliency        = saliency,
        kmers           = kmers,
        task            = _task,
        score           = raw_score,
    )


@app.get("/health")
def health():
    _get_model()
    return {"status": "ok", "task": _task, "kmer": K}


# ── Serve frontend ─────────────────────────────────────────────────────────────

_static = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=_static), name="static")


@app.get("/")
def index():
    return FileResponse(os.path.join(_static, "index.html"))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("web.main:app", host="0.0.0.0", port=8000, reload=False)
