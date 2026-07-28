#!/usr/bin/env python3
"""
Application 12 — the multimodal catalogue. Every figure and every number quoted
in Lectures 23 and 24.

    python3 tools/figures_app12.py

The corpus, stated plainly because the slides state it too
-----------------------------------------------------------------
**We do not download COCO.** The 2017 release is about 20 GB and no lecture
needs it. What we use instead:

  * the Karpathy *test* split index — a 5 MB CSV listing 5,000 COCO 2014
    validation images with their five human-written captions each, from
    `nlphuji/mscoco_2014_5k_test_image_text_retrieval` on the Hugging Face Hub;
  * the first **200** of those images by COCO id, fetched one at a time from
    `images.cocodataset.org`. That is roughly 30 MB.

So the catalogue is 200 real COCO images with real human captions, and every
retrieval number in these two lectures is measured over 200 candidates. The
slides say "200" wherever they say a recall, because a recall without its
candidate-set size is not a measurement.

The second corpus is CIFAR-10 (170 MB, the torchvision copy), of which we use
the first 2,000 test images. It is the *known-answer* test from the course's
working loop: before we trust a zero-shot model on a catalogue with no labels,
we run it on a set whose labels we have.

Models
------
  * `openai/clip-vit-base-patch32`  — the jointly trained dual encoder
  * `google/vit-base-patch16-224`   — an image-only encoder (ImageNet-supervised)
  * `sentence-transformers/all-MiniLM-L6-v2` — a text-only sentence encoder
  * `Salesforce/blip-image-captioning-base`  — the captioner, Lecture 24
  * `Qwen/Qwen2.5-0.5B-Instruct`             — the generator for RAG, Lecture 24

Every one of them can fail to download. When one does, this script records
`available: false` for that block and the deck says so on the slide. A missing
measurement stated plainly is acceptable; a fabricated one is not.

Everything expensive is cached in CACHE/fits-app12.pkl — this application keeps
its own cache rather than sharing make_figures.py's, so two authors running
their scripts at the same time cannot corrupt each other's fits.

Read TRICKS §6 and §11.6 before adding a figure.
"""

from __future__ import annotations

import ast
import json
import pickle
import re
import time
import urllib.request
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

from figkit import (ACCENT, AXIS, MATH, MUTED, PRIMARY, RULE, SMALL, SUCCESS,
                    check_text_floor, export, save, setup)

ROOT = Path(__file__).resolve().parent.parent
CACHE = Path("/private/tmp/claude-501/aiml-data/app12")
CACHE_FILE = CACHE / "fits-app12.pkl"

SEED = 42

N_CATALOGUE = 200          # candidates in every retrieval number in these decks
N_CIFAR = 2000             # the known-answer test
N_BLANKED = 60             # catalogue entries whose description we delete
EMBED_DIM = 512            # CLIP ViT-B/32's shared space

CLIP_ID = "openai/clip-vit-base-patch32"
VIT_ID = "google/vit-base-patch16-224"
TEXT_ID = "sentence-transformers/all-MiniLM-L6-v2"
BLIP_ID = "Salesforce/blip-image-captioning-base"
LLM_ID = "Qwen/Qwen2.5-0.5B-Instruct"

CSV_URL = ("https://huggingface.co/datasets/nlphuji/"
           "mscoco_2014_5k_test_image_text_retrieval/resolve/main/"
           "test_5k_mscoco_2014.csv")
IMG_URL = "http://images.cocodataset.org/val2014/{}"

# The twelve ambiguous queries of Lecture 24. Written by hand, on purpose: an
# ambiguous query is one a keyword index cannot serve, and there is no dataset
# of those. They are listed on the slide so nothing is hidden.
AMBIGUOUS = [
    "something to sit on",
    "somewhere to eat outdoors",
    "a way to get across town without a car",
    "something for a child's birthday",
    "gear for bad weather",
    "an animal that would suit a small flat",
    "something to put flowers in",
    "equipment for a game played on grass",
    "a place to sleep when travelling",
    "something warm to wear in the snow",
    "a machine that heats food",
    "something you would take to the beach",
]


# ------------------------------------------------------------------- caching

_cache: dict = {}


def load_cache() -> None:
    global _cache
    _cache = pickle.loads(CACHE_FILE.read_bytes()) if CACHE_FILE.is_file() else {}


def cached(key, fn):
    if key in _cache:
        print(f"    [cached] {key}")
        return _cache[key]
    print(f"    [computing] {key}")
    t0 = time.perf_counter()
    value = fn()
    print(f"    [done] {key} in {time.perf_counter() - t0:.1f}s")
    _cache[key] = value
    CACHE_FILE.write_bytes(pickle.dumps(_cache))
    return value


# ---------------------------------------------------------------- the corpus

def device():
    import torch
    return "mps" if torch.backends.mps.is_available() else "cpu"


def load_catalogue() -> list[dict]:
    """200 COCO validation images with their five human captions each.

    Deterministic: the split index is sorted by COCO id and the first 200 rows
    are taken, so every student's catalogue is the same catalogue.
    """
    import pandas as pd

    CACHE.mkdir(parents=True, exist_ok=True)
    csv = CACHE / "coco_karpathy_test.csv"
    if not csv.is_file():
        urllib.request.urlretrieve(CSV_URL, csv)
    df = pd.read_csv(csv).sort_values("cocoid").reset_index(drop=True)
    rows = df.head(N_CATALOGUE)

    imgdir = CACHE / "images"
    imgdir.mkdir(exist_ok=True)
    out, total_bytes = [], 0
    for _, r in rows.iterrows():
        dest = imgdir / r["filename"]
        if not dest.is_file():
            urllib.request.urlretrieve(IMG_URL.format(r["filename"]), dest)
        total_bytes += dest.stat().st_size
        caps = [" ".join(c.split()) for c in ast.literal_eval(r["raw"])]
        out.append({"cocoid": int(r["cocoid"]), "file": str(dest),
                    "sku": f"CAT-{int(r['cocoid']):06d}", "captions": caps})
    assert len(out) == N_CATALOGUE, f"{len(out)} entries, expected {N_CATALOGUE}"
    assert all(len(e["captions"]) >= 2 for e in out), "an entry has < 2 captions"
    print(f"    catalogue: {len(out)} images, {total_bytes / 1e6:.1f} MB on disk")
    load_catalogue.bytes = total_bytes
    return out


def open_images(entries):
    from PIL import Image
    return [Image.open(e["file"]).convert("RGB") for e in entries]


# ------------------------------------------------------------------ encoders

def l2(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def clip_features(entries) -> dict:
    """Unnormalised CLIP embeddings for 200 images and their first captions.

    Unnormalised on purpose: Lecture 24's assistant failure is what happens
    when you forget to normalise, and you cannot demonstrate that from vectors
    somebody already normalised for you.
    """
    import torch
    from transformers import CLIPModel, CLIPProcessor

    dev = device()
    proc = CLIPProcessor.from_pretrained(CLIP_ID)
    model = CLIPModel.from_pretrained(CLIP_ID).to(dev).eval()

    images = open_images(entries)
    texts = [e["captions"][0] for e in entries]

    img, txt = [], []
    with torch.no_grad():
        for i in range(0, len(images), 32):
            b = proc(images=images[i:i + 32], return_tensors="pt").to(dev)
            img.append(model.get_image_features(**b).cpu().numpy())
        for i in range(0, len(texts), 64):
            b = proc(text=texts[i:i + 64], return_tensors="pt",
                     padding=True, truncation=True, max_length=77).to(dev)
            txt.append(model.get_text_features(**b).cpu().numpy())

    return {"image": np.concatenate(img).astype(np.float64),
            "text": np.concatenate(txt).astype(np.float64),
            "logit_scale": float(model.logit_scale.exp().item()),
            "n_params": int(sum(p.numel() for p in model.parameters())),
            "dim": int(model.config.projection_dim),
            "device": dev}


def vit_features(entries) -> np.ndarray:
    """CLS token of an image-only ViT. 768 dimensions, no text anywhere."""
    import torch
    from transformers import AutoImageProcessor, ViTModel

    dev = device()
    proc = AutoImageProcessor.from_pretrained(VIT_ID)
    model = ViTModel.from_pretrained(VIT_ID, add_pooling_layer=False).to(dev).eval()

    images = open_images(entries)
    out = []
    with torch.no_grad():
        for i in range(0, len(images), 32):
            b = proc(images=images[i:i + 32], return_tensors="pt").to(dev)
            h = model(**b).last_hidden_state[:, 0]      # the CLS token
            out.append(h.cpu().numpy())
    return np.concatenate(out).astype(np.float64)


def text_features(sentences: list[str]) -> np.ndarray:
    """Mean-pooled MiniLM sentence embeddings. 384 dimensions, no image anywhere."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    dev = device()
    tok = AutoTokenizer.from_pretrained(TEXT_ID)
    model = AutoModel.from_pretrained(TEXT_ID).to(dev).eval()

    out = []
    with torch.no_grad():
        for i in range(0, len(sentences), 64):
            b = tok(sentences[i:i + 64], return_tensors="pt",
                    padding=True, truncation=True, max_length=128).to(dev)
            h = model(**b).last_hidden_state
            m = b["attention_mask"].unsqueeze(-1).float()
            out.append(((h * m).sum(1) / m.sum(1)).cpu().numpy())
    return np.concatenate(out).astype(np.float64)


# ------------------------------------------------------------------- metrics

def recall_at_k(sim: np.ndarray, ks=(1, 5, 10)) -> dict:
    """Text-to-image recall when row i's one relevant item is column i.

    `sim[i, j]` is the score of query i against candidate j. Ties are broken
    against us: a tie counts as the worse rank, so nothing here is flattered by
    a degenerate score matrix.
    """
    n = sim.shape[0]
    correct = sim[np.arange(n), np.arange(n)][:, None]
    rank = (sim >= correct).sum(axis=1)          # 1 = best; ties count against us
    out = {f"r{k}": float((rank <= k).mean()) for k in ks}
    out["median_rank"] = float(np.median(rank))
    out["mean_rank"] = float(rank.mean())
    out["mrr"] = float((1.0 / rank).mean())
    out["ranks"] = rank.tolist()
    return out


def matched_vs_random(sim: np.ndarray) -> dict:
    """Diagonal similarity against off-diagonal, the only comparison that means
    anything. A cosine of 0.3 is a fact about nothing until you know what an
    unrelated pair scores."""
    n = sim.shape[0]
    eye = np.eye(n, dtype=bool)
    matched, other = sim[eye], sim[~eye]
    pooled = np.sqrt((matched.var(ddof=1) + other.var(ddof=1)) / 2)
    return {"matched_mean": float(matched.mean()),
            "matched_sd": float(matched.std(ddof=1)),
            "random_mean": float(other.mean()),
            "random_sd": float(other.std(ddof=1)),
            "gap": float(matched.mean() - other.mean()),
            "cohens_d": float((matched.mean() - other.mean()) / pooled),
            "matched": matched.tolist(),
            "random_sample": other[::37][:4000].tolist()}


def random_ranking_baseline(n: int, ks=(1, 5, 10)) -> dict:
    """The trivial baseline, computed exactly rather than simulated.

    One relevant item among n, ranked uniformly at random: P(rank <= k) = k/n,
    E[rank] = (n+1)/2. TRICKS §3.1 — the commitment needs an anchor, and this
    one needs no experiment at all.
    """
    out = {f"r{k}": k / n for k in ks}
    out["expected_rank"] = (n + 1) / 2
    out["mrr"] = float(np.mean(1.0 / np.arange(1, n + 1)))
    return out


# --------------------------------------------------- L23: the two spaces
def space_mismatch(entries) -> dict:
    """Encode the images with a vision model and the captions with a text
    model, and try to compare them. Three ways of forcing the dimensions to
    agree, so nobody can blame the projection."""
    rng = np.random.default_rng(SEED)
    V = vit_features(entries)                    # 200 x 768
    T = text_features([e["captions"][0] for e in entries])   # 200 x 384
    Vn, Tn = l2(V), l2(T)

    out = {"vit_dim": int(V.shape[1]), "text_dim": int(T.shape[1]),
           "n": len(entries)}

    # (a) Johnson-Lindenstrauss: a Gaussian projection of the image space down
    #     to the text space's dimension. Thread 5 says this preserves the image
    #     geometry; it says nothing about aligning it with anything else.
    R = rng.normal(0, 1 / np.sqrt(T.shape[1]), size=(V.shape[1], T.shape[1]))
    sims = {"jl": l2(Vn @ R) @ Tn.T,
            "truncate": l2(Vn[:, :T.shape[1]]) @ Tn.T,
            "pad": Vn @ l2(np.pad(Tn, ((0, 0), (0, V.shape[1] - T.shape[1])))).T}
    for name, S in sims.items():
        out[name] = {**matched_vs_random(S.T), **recall_at_k(S.T)}
    # the projection did preserve the image geometry — thread 5's promise, kept.
    # It is the alignment with the *other* space that was never promised.
    P = l2(Vn @ R)
    before, after = Vn @ Vn.T, P @ P.T
    keep = ~np.eye(len(Vn), dtype=bool)
    out["jl_pairwise_corr"] = float(
        np.corrcoef(before[keep], after[keep])[0, 1])
    out["jl_max_distortion"] = float(np.abs(after[keep] - before[keep]).max())
    return out


def clip_retrieval(F) -> dict:
    I, T = l2(F["image"]), l2(F["text"])
    t2i = T @ I.T                     # rows: text queries, columns: images
    i2t = I @ T.T
    t2i_stats = {**matched_vs_random(t2i), **recall_at_k(t2i)}
    t2i_stats["miss1_pct"] = 100.0 * (1.0 - t2i_stats["r1"])
    return {"t2i": t2i_stats,
            "i2t": recall_at_k(i2t),
            "n": int(I.shape[0]), "dim": int(I.shape[1]),
            "n_params_m": round(F["n_params"] / 1e6),
            "logit_scale": F["logit_scale"], "device": F["device"]}


# ------------------------------------------------- L23: the known-answer test
def zero_shot_cifar() -> dict:
    """CLIP on 2,000 CIFAR-10 test images it has never been fine-tuned on.

    Three prompt templates, because the difference between them is the first
    thing a student should be suspicious of: a "zero-shot" number is a number
    for one particular sentence.
    """
    import torch
    from torchvision.datasets import CIFAR10
    from transformers import CLIPModel, CLIPProcessor

    dev = device()
    ds = CIFAR10(root=str(CACHE / "cifar"), train=False, download=True)
    classes = list(ds.classes)
    images = [ds[i][0].convert("RGB") for i in range(N_CIFAR)]
    y = np.array([ds[i][1] for i in range(N_CIFAR)])

    proc = CLIPProcessor.from_pretrained(CLIP_ID)
    model = CLIPModel.from_pretrained(CLIP_ID).to(dev).eval()

    feats = []
    with torch.no_grad():
        for i in range(0, len(images), 64):
            b = proc(images=images[i:i + 64], return_tensors="pt").to(dev)
            feats.append(model.get_image_features(**b).cpu().numpy())
    F = l2(np.concatenate(feats).astype(np.float64))

    templates = {"bare": "{}",
                 "photo": "a photo of a {}",
                 "lowres": "a low-resolution photo of a {}"}
    out = {"n": N_CIFAR, "classes": classes, "chance": 1.0 / len(classes),
           "majority": float(np.bincount(y).max() / len(y))}
    for name, tpl in templates.items():
        prompts = [tpl.format(c) for c in classes]
        with torch.no_grad():
            b = proc(text=prompts, return_tensors="pt", padding=True).to(dev)
            W = l2(model.get_text_features(**b).cpu().numpy().astype(np.float64))
        pred = (F @ W.T).argmax(1)
        out[name] = {
            "accuracy": float((pred == y).mean()),
            "per_class": {c: float((pred[y == i] == i).mean())
                          for i, c in enumerate(classes)},
        }
    out["template_spread"] = float(
        max(out[t]["accuracy"] for t in templates)
        - min(out[t]["accuracy"] for t in templates))
    return out


# ------------------------------------------- L24: concentration and the sphere
def concentration() -> dict:
    """Thread 5 returning as thread 12: in high dimensions two random unit
    vectors are nearly orthogonal, not antipodal. Measured, then compared with
    what CLIP's own embeddings do."""
    rng = np.random.default_rng(SEED)
    dims = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
    sd, mean_abs, frac_beyond = [], [], []
    for d in dims:
        A = l2(rng.normal(size=(4000, d)))
        B = l2(rng.normal(size=(4000, d)))
        c = (A * B).sum(1)
        sd.append(float(c.std(ddof=1)))
        mean_abs.append(float(np.abs(c).mean()))
        frac_beyond.append(float((np.abs(c) > 0.5).mean()))
    A = l2(rng.normal(size=(20000, EMBED_DIM)))
    B = l2(rng.normal(size=(20000, EMBED_DIM)))
    c = (A * B).sum(1)
    return {"dims": dims, "sd": sd, "mean_abs": mean_abs,
            "frac_beyond_half": frac_beyond,
            "theory_sd": [float(1 / np.sqrt(d)) for d in dims],
            "d": EMBED_DIM, "n_pairs": 20000,
            "gauss_mean": float(c.mean()), "gauss_sd": float(c.std(ddof=1)),
            "gauss_min": float(c.min()), "gauss_max": float(c.max()),
            "gauss_theory_sd": float(1 / np.sqrt(EMBED_DIM)),
            "gauss_frac_negative": float((c < 0).mean()),
            "gauss_sample": c[::4].tolist()}


def clip_geometry(F) -> dict:
    """What CLIP's own unrelated pairs actually score. Not zero — and the
    deviation has a name."""
    I, T = l2(F["image"]), l2(F["text"])
    n = I.shape[0]
    off = ~np.eye(n, dtype=bool)
    ii, tt, it = I @ I.T, T @ T.T, I @ T.T
    gap = np.linalg.norm(I.mean(0) - T.mean(0))

    # a 2-D view of the two cones, for the figure
    X = np.vstack([I, T])
    Xc = X - X.mean(0)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    P = Xc @ Vt[:2].T

    return {"n": n,
            "img_img": {"mean": float(ii[off].mean()), "sd": float(ii[off].std(ddof=1)),
                        "min": float(ii[off].min()), "max": float(ii[off].max())},
            "txt_txt": {"mean": float(tt[off].mean()), "sd": float(tt[off].std(ddof=1)),
                        "min": float(tt[off].min()), "max": float(tt[off].max())},
            "img_txt_unrelated": {"mean": float(it[off].mean()),
                                  "sd": float(it[off].std(ddof=1)),
                                  "min": float(it[off].min()),
                                  "max": float(it[off].max())},
            "img_txt_matched": {"mean": float(np.diag(it).mean()),
                                "sd": float(np.diag(it).std(ddof=1)),
                                "min": float(np.diag(it).min()),
                                "max": float(np.diag(it).max())},
            "frac_unrelated_negative": float((it[off] < 0).mean()),
            "modality_gap": float(gap),
            "img_norm_mean": float(np.linalg.norm(F["image"], axis=1).mean()),
            "txt_norm_mean": float(np.linalg.norm(F["text"], axis=1).mean()),
            "img_norm_ratio": float(np.linalg.norm(F["image"], axis=1).max()
                                    / np.linalg.norm(F["image"], axis=1).min()),
            "pca_img": P[:n].tolist(), "pca_txt": P[n:].tolist(),
            "pca_evr": (S[:2] ** 2 / (S ** 2).sum()).tolist(),
            "ii_sample": ii[off][::17][:4000].tolist(),
            "it_sample": it[off][::17][:4000].tolist()}


# ------------------------------------------------------ L24: the temperature
def _softmax(x, axis=-1):
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def infonce(sim: np.ndarray, tau: float) -> dict:
    """The symmetric contrastive loss, on a batch whose positives are the
    diagonal. `sim` must already be cosines."""
    n = sim.shape[0]
    logits = sim / tau
    p_rows = _softmax(logits, axis=1)
    p_cols = _softmax(logits, axis=0)
    d = np.arange(n)
    loss = 0.5 * (-np.log(p_rows[d, d] + 1e-300).mean()
                  - np.log(p_cols[d, d] + 1e-300).mean())
    off = p_rows.copy()
    off[d, d] = 0.0
    return {"loss": float(loss),
            "p_positive": float(p_rows[d, d].mean()),
            "p_hardest_negative": float(off.max(axis=1).mean()),
            "entropy": float(-(p_rows * np.log(p_rows + 1e-300)).sum(1).mean()),
            "accuracy": float((logits.argmax(1) == d).mean())}


def temperature_sweep(F) -> dict:
    I, T = l2(F["image"]), l2(F["text"])
    sim = I @ T.T
    taus = [1.0, 0.5, 0.2, 0.1, 0.07, 0.05, 0.03, 0.02, 0.01, 0.007, 0.005, 0.002]
    rows = {t: infonce(sim, t) for t in taus}
    learned = 1.0 / F["logit_scale"]
    return {"taus": taus,
            "loss": [rows[t]["loss"] for t in taus],
            "p_positive": [rows[t]["p_positive"] for t in taus],
            "p_hardest": [rows[t]["p_hardest_negative"] for t in taus],
            "entropy": [rows[t]["entropy"] for t in taus],
            "accuracy": [rows[t]["accuracy"] for t in taus],
            "learned_tau": learned,
            "learned_logit_scale": F["logit_scale"],
            "at_learned": infonce(sim, learned),
            "at_one": infonce(sim, 1.0),
            "at_tiny": infonce(sim, 0.002),
            "chance_loss": float(np.log(sim.shape[0])),
            "n": int(sim.shape[0])}


def batch_size_sweep(F) -> dict:
    """The number of negatives is the difficulty knob, and it is the batch
    size. Chance is 1/B, so a bigger batch is a harder task with a harder
    baseline — that is the point, not a side effect."""
    rng = np.random.default_rng(SEED)
    I, T = l2(F["image"]), l2(F["text"])
    n = I.shape[0]
    tau = 1.0 / F["logit_scale"]
    Bs = [2, 4, 8, 16, 32, 64, 128, n]
    acc_mean, acc_sd, loss_mean, chance = [], [], [], []
    for B in Bs:
        reps = 1 if B == n else 200
        a, l = [], []
        for _ in range(reps):
            idx = rng.choice(n, size=B, replace=False)
            s = I[idx] @ T[idx].T
            r = infonce(s, tau)
            a.append(r["accuracy"])
            l.append(r["loss"])
        acc_mean.append(float(np.mean(a)))
        acc_sd.append(float(np.std(a, ddof=1)) if reps > 1 else 0.0)
        loss_mean.append(float(np.mean(l)))
        chance.append(1.0 / B)
    return {"batch": Bs, "accuracy": acc_mean, "accuracy_sd": acc_sd,
            "loss": loss_mean, "chance": chance,
            "log_chance": [float(np.log(B)) for B in Bs], "tau": tau}


def normalisation_failure(F) -> dict:
    """Lecture 24's assistant failure: the same loss without the unit sphere.

    Not a crash, not a NaN — a number, and a ranking driven by vector length
    instead of by direction.
    """
    I_raw, T_raw = F["image"], F["text"]
    I, T = l2(I_raw), l2(T_raw)
    tau = 1.0 / F["logit_scale"]
    n = I.shape[0]

    good = infonce(I @ T.T, tau)
    raw_sim = I_raw @ T_raw.T
    logits = raw_sim / tau
    p = _softmax(logits, axis=1)
    d = np.arange(n)
    bad = {"loss": float(-np.log(p[d, d] + 1e-300).mean()),
           "accuracy": float((logits.argmax(1) == d).mean()),
           "p_positive": float(p[d, d].mean())}

    # Who wins a text query when the vectors are not normalised? For a fixed
    # query the text norm is a constant across candidates, so the ranking within
    # a row is by ||image|| * cos — length competes with direction. Count, per
    # image, how many of the n queries put it first.
    norms = np.linalg.norm(I_raw, axis=1)
    top1 = np.bincount((T_raw @ I_raw.T).argmax(1), minlength=n)
    order = np.argsort(np.argsort(norms))
    rank_top1 = np.argsort(np.argsort(top1))
    corr = float(np.corrcoef(order, rank_top1)[0, 1])
    # and the same count with the sphere restored, as the control
    top1_norm = np.bincount((T @ I.T).argmax(1), minlength=n)
    return {"normalised": good, "unnormalised": bad,
            "max_wins_normalised": int(top1_norm.max()),
            "n_never_first_normalised": int((top1_norm == 0).sum()),
            "loss_ratio": float(bad["loss"] / good["loss"]),
            "acc_drop_pts": float(100 * (good["accuracy"] - bad["accuracy"])),
            "norm_min": float(norms.min()), "norm_max": float(norms.max()),
            "norm_ratio": float(norms.max() / norms.min()),
            "spearman_norm_vs_wins": corr,
            "max_wins_one_image": int(top1.max()),
            "n_images_never_first": int((top1 == 0).sum()),
            "norms": norms.tolist(), "wins": top1.tolist(), "n": n}


# ------------------------------------------- L24: repair 1 — missing captions
def caption_repair(entries, blanked_idx) -> dict:
    """Text-only search over the catalogue descriptions, and what happens to
    the entries that have none.

    The query is a *different* human caption of the same image, so this is a
    paraphrase-matching task rather than a lookup.
    """
    descriptions = [e["captions"][0] for e in entries]
    queries = [e["captions"][1] for e in entries]
    n = len(entries)
    keep = np.array([i not in set(blanked_idx) for i in range(n)])

    D = l2(text_features(descriptions))
    Q = l2(text_features(queries))
    full = recall_at_k(Q @ D.T)

    # with the blanked entries simply absent from the index
    S = Q @ D.T
    S_missing = S.copy()
    S_missing[:, ~keep] = -np.inf
    miss = recall_at_k(S_missing)
    blank = np.array(sorted(blanked_idx))
    miss_on_blanked = float(np.mean(np.array(miss["ranks"])[blank] <= 1))

    return {"n": n, "n_blanked": len(blanked_idx),
            "full": {k: v for k, v in full.items() if k != "ranks"},
            "full_on_blanked": float(np.mean(np.array(full["ranks"])[blank] <= 1)),
            "missing": {k: v for k, v in miss.items() if k != "ranks"},
            "missing_on_blanked": miss_on_blanked,
            "descriptions": descriptions, "queries": queries,
            "D": D.tolist(), "Q": Q.tolist(), "keep": keep.tolist()}


def blip_captions(entries, blanked_idx) -> dict:
    from PIL import Image
    from transformers import BlipForConditionalGeneration, BlipProcessor
    import torch

    dev = device()
    proc = BlipProcessor.from_pretrained(BLIP_ID)
    model = BlipForConditionalGeneration.from_pretrained(BLIP_ID).to(dev).eval()

    out = {}
    idx = sorted(blanked_idx)
    t0 = time.perf_counter()
    with torch.no_grad():
        for i in range(0, len(idx), 8):
            chunk = idx[i:i + 8]
            imgs = [Image.open(entries[j]["file"]).convert("RGB") for j in chunk]
            b = proc(images=imgs, return_tensors="pt").to(dev)
            ids = model.generate(**b, max_new_tokens=30, num_beams=3)
            for j, txt in zip(chunk, proc.batch_decode(ids, skip_special_tokens=True)):
                out[j] = " ".join(txt.strip().split())
    return {"captions": out, "seconds": time.perf_counter() - t0,
            "n_params": int(sum(p.numel() for p in model.parameters()))}


def caption_repair_measured(entries, base, blip, F) -> dict:
    """Re-index with the generated captions and re-measure. Also: what CLIP's
    image side does on exactly the same 60 queries, which is unaffected because
    it never reads a description at all."""
    n = len(entries)
    blank = np.array(sorted(blip["captions"]))
    D = np.array(base["D"])
    Q = np.array(base["Q"])
    gen = [blip["captions"][j] for j in blank]
    Dg = D.copy()
    Dg[blank] = l2(text_features(gen))
    filled = recall_at_k(Q @ Dg.T)
    filled_on_blanked = float(np.mean(np.array(filled["ranks"])[blank] <= 1))

    # the other route: CLIP text-to-image on the same 60 queries
    I = l2(F["image"])
    Qc = l2(_clip_text(base["queries"]))
    clip_ranks = np.array(recall_at_k(Qc @ I.T)["ranks"])
    return {"filled": {k: v for k, v in filled.items() if k != "ranks"},
            "filled_on_blanked": filled_on_blanked,
            "clip_on_blanked": float(np.mean(clip_ranks[blank] <= 1)),
            "clip_overall": float(np.mean(clip_ranks <= 1)),
            "examples": [{"sku": entries[j]["sku"],
                          "human": entries[j]["captions"][0],
                          "generated": blip["captions"][j]}
                         for j in blank[:6]],
            "n_blanked": int(len(blank))}


def _clip_text(sentences):
    import torch
    from transformers import CLIPModel, CLIPProcessor
    dev = device()
    proc = CLIPProcessor.from_pretrained(CLIP_ID)
    model = CLIPModel.from_pretrained(CLIP_ID).to(dev).eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(sentences), 64):
            b = proc(text=sentences[i:i + 64], return_tensors="pt",
                     padding=True, truncation=True, max_length=77).to(dev)
            out.append(model.get_text_features(**b).cpu().numpy())
    return np.concatenate(out).astype(np.float64)


# --------------------------------------------------------- L24: repair 2 — RAG
SKU_RE = re.compile(r"CAT-\d{6}")


def rag(entries, F) -> dict:
    """An ambiguous query is one whose answer is a *shortlist with reasons*.

    The check is mechanical: the assistant must cite catalogue SKUs, and a SKU
    either exists in the catalogue or it does not. Closed-book, it cannot know
    any real one.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dev = device()
    tok = AutoTokenizer.from_pretrained(LLM_ID)
    model = AutoModelForCausalLM.from_pretrained(LLM_ID).to(dev).eval()

    valid = {e["sku"] for e in entries}
    I = l2(F["image"])
    Qc = l2(_clip_text(AMBIGUOUS))
    sim = Qc @ I.T
    top5 = np.argsort(-sim, axis=1)[:, :5]

    def ask(prompt: str) -> str:
        msgs = [{"role": "system",
                 "content": "You are a product-catalogue assistant. Cite catalogue "
                            "SKUs, which always look like CAT-123456."},
                {"role": "user", "content": prompt}]
        text = tok.apply_chat_template(msgs, tokenize=False,
                                       add_generation_prompt=True)
        b = tok([text], return_tensors="pt").to(dev)
        with torch.no_grad():
            ids = model.generate(**b, max_new_tokens=140, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
        return tok.decode(ids[0][b["input_ids"].shape[1]:], skip_special_tokens=True)

    rows = []
    for qi, q in enumerate(AMBIGUOUS):
        closed = ask(f"Our catalogue has {len(entries)} entries. "
                     f"A customer asks: \"{q}\". "
                     f"Recommend three entries and cite their SKUs.")
        shortlist = "\n".join(
            f"- {entries[j]['sku']}: {entries[j]['captions'][0]}" for j in top5[qi])
        grounded = ask(
            f"Here are the catalogue entries our search returned for the query "
            f"\"{q}\":\n{shortlist}\n\n"
            f"Recommend the best ones for the customer, citing only SKUs from "
            f"the list above. If none fit, say so.")
        rows.append({
            "query": q,
            "retrieved": [entries[j]["sku"] for j in top5[qi]],
            "retrieved_captions": [entries[j]["captions"][0] for j in top5[qi]],
            "closed_text": closed.strip(),
            "closed_skus": SKU_RE.findall(closed),
            "grounded_text": grounded.strip(),
            "grounded_skus": SKU_RE.findall(grounded)})

    def tally(key):
        cited = [s for r in rows for s in r[key]]
        ok = [s for s in cited if s in valid]
        return {"citations": len(cited), "valid": len(ok),
                "valid_pct": 100 * len(ok) / len(cited) if cited else 0.0,
                "queries_with_a_citation": sum(1 for r in rows if r[key])}

    n_params = int(sum(p.numel() for p in model.parameters()))
    return {"n_queries": len(AMBIGUOUS), "rows": rows,
            "closed": tally("closed_skus"), "grounded": tally("grounded_skus"),
            "catalogue_size": len(entries),
            "llm": LLM_ID, "n_params": n_params,
            "n_params_m": round(n_params / 1e6)}


# --------------------------------------------------------------------- figures

def fig_catalogue(entries):
    """Sixteen thumbnails, no per-image labels.

    Two rows rather than three, and no captions under the images: at three rows
    the figure is clamped by .fig-wide's 420px cap to 0.42, which takes a 15pt
    label to 14.1px on the slide — under the floor. See TRICKS §11.6.
    """
    from PIL import Image

    def square(path):
        im = Image.open(path).convert("RGB")
        s = min(im.size)
        left, top = (im.width - s) // 2, (im.height - s) // 2
        return im.crop((left, top, left + s, top + s)).resize((224, 224))

    fig, axes = plt.subplots(2, 8, figsize=(11.5, 3.3))
    for ax, e in zip(axes.ravel(), entries[:16]):
        ax.imshow(square(e["file"]))
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        for s in ax.spines.values():
            s.set_edgecolor(RULE)
    fig.suptitle(f"16 of the {N_CATALOGUE} catalogue entries — COCO validation "
                 f"images, each with five human captions we hold back",
                 fontsize=SMALL, color=MUTED, x=0.004, y=0.975, ha="left")
    # subplots_adjust rather than tight_layout: tight_layout reserves gutters for
    # tick labels these axes do not have, and left the grid floating in white
    fig.subplots_adjust(left=0.004, right=0.996, top=0.90, bottom=0.008,
                        wspace=0.045, hspace=0.055)
    return save(fig, "l23-catalogue", raster=True)


def plain_log(ax, axis, ticks, fmt=lambda v: f"{v:g}"):
    """Plain-text log tick labels.

    matplotlib's log formatter writes mathtext, and `text.parse_math` is off
    project-wide (TRICKS §9.1), so the default labels ship as the literal string
    `$\\mathdefault{10^{-1}}$`. Set the ticks by hand instead.
    """
    a = ax.xaxis if axis == "x" else ax.yaxis
    a.set_major_locator(FixedLocator(ticks))
    a.set_minor_locator(NullLocator())
    a.set_major_formatter(FuncFormatter(lambda v, _: fmt(v)))


def _pair_hist(ax, matched, random_, *, title, xlabel, headroom=1.42,
               labels=("unrelated pairs", "the caption of that image")):
    lo = min(min(matched), min(random_))
    hi = max(max(matched), max(random_))
    bins = np.linspace(lo, hi, 42)
    ax.hist(random_, bins=bins, density=True, color=RULE, edgecolor=AXIS,
            linewidth=0.6, label=labels[0])
    ax.hist(matched, bins=bins, density=True, color=PRIMARY, alpha=0.72,
            label=labels[1])
    ax.axvline(np.mean(random_), color=AXIS, ls=":", lw=2)
    ax.axvline(np.mean(matched), color=PRIMARY, ls="--", lw=2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("density")
    ax.set_title(title)
    # headroom, so the legend and the callout have somewhere to sit that is not
    # on top of the data. TRICKS §6.1.
    ax.set_ylim(0, ax.get_ylim()[1] * headroom)


def fig_mismatch(sm):
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.0))
    jl = sm["jl"]
    _pair_hist(axes[0], jl["matched"], jl["random_sample"],
               title="ViT-B/16 images against MiniLM captions",
               xlabel="cosine similarity", headroom=1.75,
               labels=("unrelated", "matched"))
    # both corners of this panel are empty; the middle is all data. The labels
    # are short here because the callout needs the other corner.
    axes[0].legend(loc="upper right", framealpha=1.0, frameon=True,
                   facecolor="white", edgecolor="none")
    axes[0].annotate(f"means differ by {jl['gap']:+.3f}\n"
                     f"sd of unrelated: {jl['random_sd']:.3f}",
                     xy=(0.02, 0.985), xycoords="axes fraction", fontsize=SMALL,
                     color=ACCENT, va="top",
                     bbox=dict(fc="white", ec=ACCENT, boxstyle="round,pad=0.35"))
    names = ["JL projection\n768 to 384", "truncate\nimage to 384",
             "zero-pad\ntext to 768"]
    keys = ["jl", "truncate", "pad"]
    r1 = [100 * sm[k]["r1"] for k in keys]
    axes[1].bar(names, r1, color=ACCENT, width=0.55)
    for i, v in enumerate(r1):
        axes[1].text(i, v + 0.05, f"{v:.1f}%", ha="center", fontsize=SMALL,
                     color=ACCENT)
    axes[1].axhline(100 / sm["n"], color=PRIMARY, ls="--", lw=2)
    axes[1].text(2.45, 100 / sm["n"] + 0.55,
                 f"random ranking = {100 / sm['n']:.1f}%",
                 ha="right", fontsize=SMALL, color=PRIMARY)
    axes[1].set_ylabel("Recall@1, %")
    axes[1].set_ylim(0, max(2.0, max(r1) * 1.9))
    axes[1].set_title(f"three ways to make the dimensions agree "
                      f"(n = {sm['n']})")
    fig.tight_layout()
    return save(fig, "l23-space-mismatch")


def fig_clip_sim(cl):
    t2i = cl["t2i"]
    fig, ax = plt.subplots(figsize=(9.6, 4.0))
    _pair_hist(ax, t2i["matched"], t2i["random_sample"],
               title=f"CLIP ViT-B/32, the same {cl['n']} images and captions",
               xlabel="cosine similarity")
    ax.legend(loc="upper left", framealpha=1.0, frameon=True,
              facecolor="white", edgecolor="none")
    ax.annotate(f"means differ by {t2i['gap']:+.3f}\n"
                f"Cohen's d = {t2i['cohens_d']:.2f}",
                xy=(0.66, 0.72), xycoords="axes fraction", fontsize=SMALL,
                color=SUCCESS,
                bbox=dict(fc="white", ec=SUCCESS, boxstyle="round,pad=0.35"))
    fig.tight_layout()
    return save(fig, "l23-clip-sim")


def fig_recall(sm, cl, base):
    ks = [1, 5, 10]
    x = np.arange(len(ks))
    w = 0.26
    fig, ax = plt.subplots(figsize=(9.6, 4.2))
    ax.bar(x - w, [100 * base[f"r{k}"] for k in ks], w, color=AXIS,
           label="random ranking (exact)")
    ax.bar(x, [100 * sm["jl"][f"r{k}"] for k in ks], w, color=ACCENT,
           label="ViT + MiniLM, two separate spaces")
    ax.bar(x + w, [100 * cl["t2i"][f"r{k}"] for k in ks], w, color=SUCCESS,
           label="CLIP, one joint space")
    # inside the bar, not above it: above it collides with the legend
    for i, k in enumerate(ks):
        ax.text(i + w, 100 * cl["t2i"][f"r{k}"] - 3.5,
                f"{100 * cl['t2i'][f'r{k}']:.1f}", ha="center", va="top",
                fontsize=SMALL, color="white")
    ax.set_xticks(x, [f"Recall@{k}" for k in ks])
    ax.set_ylabel("%")
    ax.set_ylim(0, 122)
    ax.legend(loc="upper left", framealpha=1.0, frameon=True,
              facecolor="white", edgecolor="none")
    ax.set_title(f"text-to-image retrieval over {cl['n']} candidates")
    fig.tight_layout()
    return save(fig, "l23-recall")


def fig_ranks(cl, sm):
    fig, ax = plt.subplots(figsize=(9.6, 3.9))
    n = cl["n"]
    bins = np.arange(0.5, n + 1.5, 4)
    ax.hist(sm["jl"]["ranks"], bins=bins, color=ACCENT, alpha=0.75,
            label="ViT + MiniLM")
    ax.hist(cl["t2i"]["ranks"], bins=bins, color=SUCCESS, alpha=0.8,
            label="CLIP")
    ax.set_ylim(0, ax.get_ylim()[1] * 1.15)
    ax.axvline((n + 1) / 2, color=PRIMARY, ls="--", lw=2)
    ax.text((n + 1) / 2 + 5, ax.get_ylim()[1] * 0.34,
            f"random ranking\nputs it at {(n + 1) / 2:.0f} on average",
            fontsize=SMALL, color=PRIMARY,
            bbox=dict(fc="white", ec="none", pad=2))
    ax.set_xlabel(f"rank of the correct image, out of {n}")
    ax.set_ylabel("queries")
    ax.legend(loc="upper right", framealpha=1.0, frameon=True,
              facecolor="white", edgecolor="none")
    ax.set_title("where the right answer actually landed")
    fig.tight_layout()
    return save(fig, "l23-ranks")


def fig_zeroshot(zs):
    classes = zs["classes"]
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.0),
                             gridspec_kw={"width_ratios": [1.25, 1]})
    v = [100 * zs["photo"]["per_class"][c] for c in classes]
    axes[0].bar(classes, v, color=PRIMARY)
    axes[0].axhline(100 * zs["chance"], color=ACCENT, ls="--", lw=2)
    # every bar is tall, so there is no empty region: a white bbox it is
    axes[0].text(9.4, 100 * zs["chance"] + 3.0, "chance = 10%", ha="right",
                 fontsize=SMALL, color=ACCENT,
                 bbox=dict(fc="white", ec="none", pad=1.6))
    axes[0].set_ylabel("accuracy, %")
    axes[0].set_ylim(0, 105)
    axes[0].tick_params(axis="x", rotation=45)
    for lbl in axes[0].get_xticklabels():
        lbl.set_ha("right")
    axes[0].set_title(f"per class, \"a photo of a …\", {zs['n']:,} CIFAR-10 images")

    names = ["\"dog\"", "\"a photo\nof a dog\"", "\"a low-resolution\nphoto of a dog\""]
    accs = [100 * zs[k]["accuracy"] for k in ("bare", "photo", "lowres")]
    axes[1].bar(names, accs, color=MATH, width=0.55)
    for i, a in enumerate(accs):
        axes[1].text(i, a + 1.5, f"{a:.1f}", ha="center", fontsize=SMALL, color=MATH)
    axes[1].set_ylim(0, 105)
    axes[1].set_ylabel("accuracy, %")
    axes[1].set_title("the same model, three sentences")
    fig.tight_layout()
    return save(fig, "l23-zeroshot")


def fig_concentration(co):
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.0))
    axes[0].loglog(co["dims"], co["sd"], "o-", color=MATH, lw=2,
                   label="measured sd of the cosine")
    axes[0].loglog(co["dims"], co["theory_sd"], "--", color=AXIS, lw=2,
                   label="1 / sqrt(d)")
    plain_log(axes[0], "x", [2, 8, 32, 128, 512, 2048], lambda v: f"{v:.0f}")
    plain_log(axes[0], "y", [0.01, 0.03, 0.1, 0.3, 0.7])
    axes[0].set_xlabel("dimension d")
    axes[0].set_ylabel("sd of the cosine")
    axes[0].legend(loc="upper right")
    axes[0].set_title("4,000 random pairs per dimension")
    axes[0].annotate(f"at d = {co['d']}: sd = {co['gauss_sd']:.4f}",
                     xy=(co["d"], co["gauss_sd"]), xytext=(0.04, 0.10),
                     textcoords="axes fraction", fontsize=SMALL, color=MATH,
                     arrowprops=dict(arrowstyle="->", color=MATH),
                     bbox=dict(fc="white", ec=MATH, boxstyle="round,pad=0.3"))

    axes[1].hist(co["gauss_sample"], bins=60, density=True, color=MATH, alpha=0.8)
    top = axes[1].get_ylim()[1] * 1.25
    axes[1].axvline(0, color=SUCCESS, lw=2.5)
    axes[1].axvline(-1, color=ACCENT, lw=2.5)
    axes[1].set_xlim(-1.05, 1.05)
    axes[1].set_ylim(0, top)
    axes[1].set_xlabel(f"cosine between random unit vectors, d = {co['d']}")
    axes[1].set_ylabel("density")
    axes[1].set_title("nothing is anywhere near minus one")
    axes[1].annotate("an unrelated pair\nlands here: 0",
                     xy=(0.02, top * 0.60), xytext=(0.58, 0.66),
                     textcoords="axes fraction",
                     fontsize=SMALL, color=SUCCESS,
                     arrowprops=dict(arrowstyle="->", color=SUCCESS),
                     bbox=dict(fc="white", ec=SUCCESS, boxstyle="round,pad=0.3"))
    axes[1].annotate("where \"opposite\"\nwould be", xy=(-1, top * 0.30),
                     xytext=(0.04, 0.52), textcoords="axes fraction",
                     fontsize=SMALL, color=ACCENT,
                     arrowprops=dict(arrowstyle="->", color=ACCENT),
                     bbox=dict(fc="white", ec=ACCENT, boxstyle="round,pad=0.3"))
    fig.tight_layout()
    return save(fig, "l24-concentration")


def fig_geometry(g, co):
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.0))
    bins = np.linspace(-0.4, 1.0, 60)
    axes[0].hist(co["gauss_sample"], bins=bins, density=True, color=RULE,
                 edgecolor=AXIS, linewidth=0.5, label="random unit vectors")
    axes[0].hist(g["it_sample"], bins=bins, density=True, color=ACCENT,
                 alpha=0.7, label="CLIP: unrelated image & caption")
    axes[0].hist(g["ii_sample"], bins=bins, density=True, color=PRIMARY,
                 alpha=0.6, label="CLIP: two unrelated images")
    axes[0].axvline(0, color=SUCCESS, lw=2)
    axes[0].set_xlabel("cosine similarity")
    axes[0].set_ylabel("density")
    axes[0].set_ylim(0, axes[0].get_ylim()[1] * 1.45)
    axes[0].legend(loc="upper right", fontsize=SMALL, framealpha=1.0,
                   frameon=True, facecolor="white", edgecolor="none")
    axes[0].set_title("unrelated pairs sit near zero — but not at zero")

    pi = np.array(g["pca_img"]); pt = np.array(g["pca_txt"])
    axes[1].scatter(pi[:, 0], pi[:, 1], s=14, color=PRIMARY)
    axes[1].scatter(pt[:, 0], pt[:, 1], s=14, color=ACCENT)
    axes[1].annotate("captions", xy=(pt[:, 0].mean(), pt[:, 1].max() * 1.7),
                     ha="center", fontsize=SMALL, color=ACCENT)
    axes[1].annotate("images", xy=(pi[:, 0].mean(), pi[:, 1].max() * 1.7),
                     ha="center", fontsize=SMALL, color=PRIMARY)
    axes[1].set_ylim(pt[:, 1].min() * 1.25, pt[:, 1].max() * 2.1)
    axes[1].set_xlabel("first principal direction")
    axes[1].set_ylabel("second")
    axes[1].set_title(f"one space, two cones, {g['modality_gap']:.2f} apart")
    fig.tight_layout()
    return save(fig, "l24-geometry")


def fig_temperature(ts):
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.0))
    ticks = [1.0, 0.1, 0.01, 0.002]
    axes[0].semilogx(ts["taus"], ts["loss"], "o-", color=MATH, lw=2)
    axes[0].axhline(ts["chance_loss"], color=AXIS, ls=":", lw=2)
    axes[0].set_ylim(0, ts["chance_loss"] * 1.30)
    axes[0].text(0.0022, ts["chance_loss"] + 0.14,
                 f"a scorer that knows nothing: log {ts['n']} = "
                 f"{ts['chance_loss']:.2f}",
                 ha="left", fontsize=SMALL, color=AXIS)
    axes[0].axvline(ts["learned_tau"], color=SUCCESS, ls="--", lw=2)
    axes[0].annotate(f"what CLIP learned:\ntau = {ts['learned_tau']:.4f}",
                     xy=(ts["learned_tau"], ts["at_learned"]["loss"]),
                     xytext=(0.34, 0.46), textcoords="axes fraction",
                     fontsize=SMALL, color=SUCCESS,
                     arrowprops=dict(arrowstyle="->", color=SUCCESS),
                     bbox=dict(fc="white", ec=SUCCESS, boxstyle="round,pad=0.3"))
    axes[0].invert_xaxis()
    plain_log(axes[0], "x", ticks)
    axes[0].set_xlabel("temperature tau  (colder to the right)")
    axes[0].set_ylabel("contrastive loss")
    axes[0].set_title(f"one set of {ts['n']} cosines, twelve temperatures")

    axes[1].semilogx(ts["taus"], ts["p_positive"], "o-", color=SUCCESS, lw=2,
                     label="probability on the right answer")
    axes[1].semilogx(ts["taus"], ts["p_hardest"], "s-", color=ACCENT, lw=2,
                     label="probability on the hardest wrong one")
    axes[1].semilogx(ts["taus"], ts["accuracy"], "--", color=PRIMARY, lw=2,
                     label="top-1 accuracy — flat")
    axes[1].axvline(ts["learned_tau"], color=SUCCESS, ls="--", lw=1.5)
    axes[1].invert_xaxis()
    plain_log(axes[1], "x", ticks)
    axes[1].set_xlabel("temperature tau")
    axes[1].set_ylabel("mean probability")
    axes[1].set_ylim(-0.03, 1.28)
    axes[1].legend(loc="upper left", fontsize=SMALL, framealpha=1.0,
                   frameon=True, facecolor="white", edgecolor="none")
    axes[1].set_title("what the loss is paying attention to")
    fig.tight_layout()
    return save(fig, "l24-temperature")


def fig_batch(bs):
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.0))
    B = bs["batch"]
    acc = [100 * a for a in bs["accuracy"]]
    chance = [100 * c for c in bs["chance"]]
    axes[0].semilogx(B, acc, "o-", color=PRIMARY, lw=2, base=2)
    axes[0].semilogx(B, chance, "--", color=AXIS, lw=2, base=2)
    axes[0].fill_between(B,
                         [100 * (a - s) for a, s in zip(bs["accuracy"], bs["accuracy_sd"])],
                         [100 * (a + s) for a, s in zip(bs["accuracy"], bs["accuracy_sd"])],
                         color=PRIMARY, alpha=0.18)
    axes[0].annotate("CLIP, measured", xy=(B[2], acc[2] + 7), fontsize=SMALL,
                     color=PRIMARY)
    axes[0].annotate("chance = 1/B", xy=(B[2], chance[2] + 7), fontsize=SMALL,
                     color=AXIS)
    plain_log(axes[0], "x", B, lambda v: f"{v:.0f}")
    axes[0].set_xlabel("batch size B = 1 positive and B-1 negatives")
    axes[0].set_ylabel("in-batch top-1, %")
    axes[0].set_ylim(0, 112)
    axes[0].set_title("200 random batches at each size")

    axes[1].semilogx(B, bs["loss"], "o-", color=MATH, lw=2, base=2,
                     label="measured loss")
    axes[1].semilogx(B, bs["log_chance"], "--", color=AXIS, lw=2,
                     base=2, label="log B — a scorer that knows nothing")
    plain_log(axes[1], "x", B, lambda v: f"{v:.0f}")
    axes[1].set_xlabel("batch size B")
    axes[1].set_ylabel("contrastive loss")
    axes[1].set_ylim(0, max(bs["log_chance"]) * 1.32)
    axes[1].legend(loc="upper left", fontsize=SMALL, framealpha=1.0,
                   frameon=True, facecolor="white", edgecolor="none")
    axes[1].set_title(f"the ceiling moves with the batch, tau = {bs['tau']:.4f}")
    fig.tight_layout()
    return save(fig, "l24-batch")


def fig_normalisation(nf):
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.0))
    labels = ["on the unit sphere", "raw dot products"]
    acc = [100 * nf["normalised"]["accuracy"], 100 * nf["unnormalised"]["accuracy"]]
    axes[0].bar(labels, acc, color=[SUCCESS, ACCENT], width=0.5)
    for i, a in enumerate(acc):
        axes[0].text(i, a + 1.5, f"{a:.1f}%", ha="center", fontsize=SMALL,
                     color=[SUCCESS, ACCENT][i])
    axes[0].set_ylim(0, 105)
    axes[0].set_ylabel("in-batch top-1, %")
    axes[0].set_title(f"the same loss, the same tau, {nf['n']} pairs")

    axes[1].scatter(nf["norms"], nf["wins"], s=18, color=ACCENT)
    axes[1].set_xlabel("length of the image embedding")
    axes[1].set_ylabel("queries that ranked it first")
    axes[1].set_title("without normalising, length competes with direction")
    axes[1].annotate(f"rank correlation {nf['spearman_norm_vs_wins']:+.2f}\n"
                     f"{nf['n_images_never_first']} of {nf['n']} images are "
                     f"never first\n(against "
                     f"{nf['n_never_first_normalised']} on the sphere)",
                     xy=(0.03, 0.70), xycoords="axes fraction", fontsize=SMALL,
                     color=ACCENT,
                     bbox=dict(fc="white", ec=ACCENT, boxstyle="round,pad=0.35"))
    fig.tight_layout()
    return save(fig, "l24-normalisation")


def fig_captions(cr, cm):
    fig, ax = plt.subplots(figsize=(9.8, 4.0))
    names = ["human description\npresent", "description missing\n(entry invisible)",
             "auto-caption from\na multimodal model"]
    vals = [100 * cr["full_on_blanked"], 100 * cr["missing_on_blanked"],
            100 * cm["filled_on_blanked"]]
    ax.bar(names, vals, color=[PRIMARY, ACCENT, SUCCESS], width=0.5)
    for i, v in enumerate(vals):
        ax.text(i, v + 1.5, f"{v:.1f}%", ha="center", fontsize=SMALL,
                color=[PRIMARY, ACCENT, SUCCESS][i])
    ax.axhline(100 * cm["clip_on_blanked"], color=MATH, ls="--", lw=2)
    ax.text(2.45, 100 * cm["clip_on_blanked"] + 3.4,
            f"CLIP's image route, unaffected: {100 * cm['clip_on_blanked']:.1f}%",
            ha="right", fontsize=SMALL, color=MATH)
    ax.set_ylabel("Recall@1, %")
    ax.set_ylim(0, 105)
    ax.set_title(f"text-only search, measured on the {cm['n_blanked']} entries "
                 f"whose description we deleted")
    fig.tight_layout()
    return save(fig, "l24-captions")


def fig_rag(rg):
    fig, ax = plt.subplots(figsize=(9.8, 3.8))
    names = ["closed book", "retrieval-augmented"]
    vals = [rg["closed"]["valid_pct"], rg["grounded"]["valid_pct"]]
    ax.bar(names, vals, color=[ACCENT, SUCCESS], width=0.45)
    for i, v in enumerate(vals):
        n_cit = [rg["closed"], rg["grounded"]][i]["citations"]
        n_ok = [rg["closed"], rg["grounded"]][i]["valid"]
        ax.text(i, v + 2.5, f"{v:.0f}%\n({n_ok} of {n_cit} citations)",
                ha="center", fontsize=SMALL, color=[ACCENT, SUCCESS][i])
    ax.set_ylim(0, 132)
    ax.set_ylabel("cited SKUs that exist, %")
    ax.set_title(f"{rg['n_queries']} ambiguous queries, "
                 f"{rg['catalogue_size']}-entry catalogue")
    fig.tight_layout()
    return save(fig, "l24-rag")


def fig_commitment(cl, base, sm):
    """The scoring slide: what a random ranker gets, what two separate spaces
    get, what the joint one gets."""
    fig, ax = plt.subplots(figsize=(9.8, 3.6))
    names = ["random ranking\n(exact, k/n)", "two separate\nencoders",
             "one joint\nencoder"]
    vals = [100 * base["r1"], 100 * sm["jl"]["r1"], 100 * cl["t2i"]["r1"]]
    ax.barh(names[::-1], vals[::-1], color=[SUCCESS, ACCENT, AXIS], height=0.55)
    for i, v in enumerate(vals[::-1]):
        ax.text(v + 1.2, i, f"{v:.1f}%", va="center", fontsize=SMALL,
                color=[SUCCESS, ACCENT, AXIS][i])
    ax.set_xlim(0, 100)
    ax.set_xlabel(f"Recall@1 over {cl['n']} candidates, %")
    ax.set_title("the number you committed to, and the three anchors")
    fig.tight_layout()
    return save(fig, "l23-commitment")


# ------------------------------------------------------------------------ main

def strip_heavy(d: dict) -> dict:
    """figures.json is read by check_provenance and by substitute; the raw
    embedding matrices belong in the pickle, not there."""
    drop = {"D", "Q", "keep", "matched", "random_sample", "gauss_sample",
            "ii_sample", "it_sample", "pca_img", "pca_txt", "ranks",
            "norms", "wins", "descriptions", "queries"}
    out = {}
    for k, v in d.items():
        if k in drop:
            continue
        out[k] = strip_heavy(v) if isinstance(v, dict) else v
    return out


def main() -> int:
    setup()
    load_cache()
    np.random.seed(SEED)

    print("Corpus:")
    entries = load_catalogue()
    corpus = {
        "n_catalogue": N_CATALOGUE,
        "n_captions_each": 5,
        "megabytes": round(load_catalogue.bytes / 1e6, 1),
        "coco_full_gb": 20,
        "n_cifar": N_CIFAR,
        "source": "COCO 2014 val, Karpathy test split, first 200 by image id",
    }

    print("Lecture 23 — the two spaces:")
    sm = cached("space_mismatch", lambda: space_mismatch(entries))
    print(f"    ViT+MiniLM matched cosine {sm['jl']['matched_mean']:+.4f} "
          f"vs unrelated {sm['jl']['random_mean']:+.4f}  "
          f"(R@1 {sm['jl']['r1']:.3%})")

    print("Lecture 23 — the joint space:")
    F = cached("clip_features", lambda: clip_features(entries))
    cl = cached("clip_retrieval", lambda: clip_retrieval(F))
    base = random_ranking_baseline(N_CATALOGUE)
    print(f"    CLIP matched cosine {cl['t2i']['matched_mean']:+.4f} "
          f"vs unrelated {cl['t2i']['random_mean']:+.4f}  "
          f"(R@1 {cl['t2i']['r1']:.1%}, R@5 {cl['t2i']['r5']:.1%})")
    zs = cached("zero_shot", zero_shot_cifar)
    print(f"    CIFAR-10 zero-shot: {zs['photo']['accuracy']:.1%} "
          f"(chance {zs['chance']:.0%})")

    print("Lecture 24 — the thread:")
    co = cached("concentration", concentration)
    print(f"    d = {co['d']}: cosine mean {co['gauss_mean']:+.5f}, "
          f"sd {co['gauss_sd']:.4f} against 1/sqrt(d) = "
          f"{co['gauss_theory_sd']:.4f}")
    geo = cached("clip_geometry", lambda: clip_geometry(F))
    print(f"    CLIP unrelated image/caption cosine "
          f"{geo['img_txt_unrelated']['mean']:+.4f}; modality gap "
          f"{geo['modality_gap']:.3f}")
    ts = cached("temperature", lambda: temperature_sweep(F))
    print(f"    learned temperature {ts['learned_tau']:.5f} "
          f"(logit scale {ts['learned_logit_scale']:.2f})")
    bs = cached("batch_sweep", lambda: batch_size_sweep(F))
    nf = cached("normalisation", lambda: normalisation_failure(F))
    print(f"    without normalising: top-1 {nf['unnormalised']['accuracy']:.1%} "
          f"against {nf['normalised']['accuracy']:.1%}")

    print("Lecture 24 — the repairs:")
    blanked = [i for i in range(N_CATALOGUE) if i % 10 < 3][:N_BLANKED]
    assert len(blanked) == N_BLANKED
    cr = cached("caption_repair", lambda: caption_repair(entries, blanked))
    blip = cached("blip", lambda: blip_captions(entries, blanked))
    cm = cached("caption_measured",
                lambda: caption_repair_measured(entries, cr, blip, F))
    print(f"    text-only R@1 on the blanked 60: "
          f"{cr['full_on_blanked']:.1%} -> {cr['missing_on_blanked']:.1%} "
          f"-> {cm['filled_on_blanked']:.1%}")
    rg = cached("rag", lambda: rag(entries, F))
    print(f"    RAG: cited SKUs that exist "
          f"{rg['closed']['valid_pct']:.0f}% -> {rg['grounded']['valid_pct']:.0f}%")

    print("Lecture 23 figures:")
    fig_catalogue(entries)
    fig_mismatch(sm)
    fig_clip_sim(cl)
    fig_recall(sm, cl, base)
    fig_ranks(cl, sm)
    fig_zeroshot(zs)
    fig_commitment(cl, base, sm)

    print("Lecture 24 figures:")
    fig_concentration(co)
    fig_geometry(geo, co)
    fig_temperature(ts)
    fig_batch(bs)
    fig_normalisation(nf)
    fig_captions(cr, cm)
    fig_rag(rg)

    export(**{
        "l23_corpus": corpus,
        "l23_mismatch": strip_heavy(sm),
        "l23_clip": strip_heavy(cl),
        "l23_random": base,
        "l23_zeroshot": strip_heavy(zs),
        "l24_concentration": strip_heavy(co),
        "l24_geometry": strip_heavy(geo),
        "l24_temperature": strip_heavy(ts),
        "l24_batch": bs,
        "l24_normalisation": strip_heavy(nf),
        # The text-only route is built and measured in Lecture 23; Lecture 24
        # repairs it. Two keys, so a slide's number says which lecture produced
        # it without anyone having to check.
        "l23_textroute": strip_heavy(cr),
        "l24_captions": {**strip_heavy(cm),
                         "blip_seconds": round(blip["seconds"], 1),
                         "blip_params": blip["n_params"],
                         "blip_params_m": round(blip["n_params"] / 1e6)},
        "l24_rag": {k: v for k, v in rg.items() if k != "rows"},
    })
    (CACHE / "rag_transcripts.json").write_text(json.dumps(rg["rows"], indent=2))
    (CACHE / "blip_captions.json").write_text(
        json.dumps({str(k): v for k, v in blip["captions"].items()}, indent=2))

    problems = check_text_floor()
    if problems:
        print("\ntext floor:")
        for p in problems:
            print("  " + p)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
