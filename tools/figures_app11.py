#!/usr/bin/env python3
"""
Application 11 — Lectures 21 and 22. IMDb customer feedback: subword
tokenisation, trainable embeddings, a recurrent classifier from scratch, and
what happens when someone else's pretraining is dropped into the same
architecture.

    python3 tools/figures_app11.py

Everything printed on slides/lecture-21.html and slides/lecture-22.html comes
from here, via figkit.export() into assets/figures/figures.json. Expensive fits
are cached (figkit.cached) so a cosmetic re-run takes seconds; delete
/private/tmp/claude-501/aiml-data/fits-app11.pkl to refit from scratch.

The data is the Large Movie Review Dataset (Maas et al. 2011), fetched from
Stanford directly rather than through the `datasets` package, which is not
installed. 50,000 reviews, 25,000 train / 25,000 test, balanced by
construction.

Timings are wall-clock on the machine that generated the figures — an Apple
Silicon laptop with an MPS backend and no CUDA. They are measurements, not
guarantees, and the decks say so.
"""

from __future__ import annotations

import os

# Three env vars, all of which must be set before torch is imported, and all of
# which were paid for.
#
# TOKENIZERS_PARALLELISM: the Rust tokenizer keeps a thread pool and
# scikit-learn forks; a fork with that pool live warns loudly and deadlocks.
#
# OMP_NUM_THREADS / KMP_DUPLICATE_LIB_OK: torch and scikit-learn each bring
# their own OpenMP runtime, and on macOS loading both kills the process inside
# KMeans with `OMP: Error #179: pthread_mutex_init failed` and no Python
# traceback at all. It killed this script four times, always on the same line,
# and the log showed only that the process had stopped existing. Verified:
# KMP_DUPLICATE_LIB_OK alone still segfaults; the thread limit is the load-
# bearing one.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import pickle
import re
import sys
import tarfile
import time
import urllib.request
import xml.dom.minidom
from collections import Counter
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))

from figkit import (setup, save, export, OUT, SEED, plain_log,        # noqa: E402
                    PRIMARY, ACCENT, SUCCESS, MATH, MUTED, RULE, AXIS,
                    BODY, SMALL, TICK, check_text_floor)

import torch                                                        # noqa: E402
import torch.nn as nn                                               # noqa: E402

CACHE = Path("/private/tmp/claude-501/aiml-data")
DATA = CACHE / "aclImdb"
URL = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

# --- the one configuration the whole application is built around ------------
MAXLEN = 256          # tokens; the mean review is ~230 words
EMB_DIM = 128         # every embedding table in Lectures 21-22, so that the
                      # architecture is literally unchanged across the swap
HIDDEN = 64           # bidirectional GRU, so the head sees 128
BATCH = 64
EPOCHS = 4
LR = 1e-3
N_FIT = 20_000        # carved out of the 25,000 training reviews
N_VAL = 5_000
WORD_VOCAB = 20_000   # our own vocabulary, built on the fit split only

BERT = "distilbert-base-uncased"
MINILM = "sentence-transformers/all-MiniLM-L6-v2"

WORD_RE = re.compile(r"[a-z0-9']+")

# app11 keeps its own pickle so a re-run of another application's script cannot
# invalidate half of ours.
_CACHE_FILE = CACHE / "fits-app11.pkl"
_cache: dict = {}


def load_cache():
    global _cache
    _cache = pickle.loads(_CACHE_FILE.read_bytes()) if _CACHE_FILE.is_file() else {}


def cached(key, fn):
    if key in _cache:
        print(f"    [cached] {key}")
        return _cache[key]
    print(f"    [computing] {key}")
    value = fn()
    _cache[key] = value
    _CACHE_FILE.write_bytes(pickle.dumps(_cache))
    return value


# ------------------------------------------------------------------ the data

def load_imdb() -> dict:
    """The 50,000 reviews, exactly the loader the slides show.

    ~80 MB the first time, instant afterwards. The train/test split is the one
    shipped with the corpus, so every course in the world reports on the same
    25,000 test reviews.
    """
    if not DATA.is_dir():
        tarball = CACHE / "aclImdb_v1.tar.gz"
        if not tarball.is_file():
            CACHE.mkdir(parents=True, exist_ok=True)
            print(f"  downloading {URL}")
            urllib.request.urlretrieve(URL, tarball)
        with tarfile.open(tarball) as t:
            t.extractall(path=CACHE, filter="data")

    def read(split):
        texts, labels = [], []
        for lab, name in ((1, "pos"), (0, "neg")):
            for p in sorted((DATA / split / name).iterdir()):
                texts.append(p.read_text(encoding="utf-8"))
                labels.append(lab)
        return texts, np.array(labels, dtype=np.int64)

    tr_x, tr_y = read("train")
    te_x, te_y = read("test")

    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(tr_x))
    fit_i, val_i = order[:N_FIT], order[N_FIT:N_FIT + N_VAL]

    # The corpus ships positives first and negatives second, so te_x[:n] is
    # every positive review and nothing else. Any "accuracy" measured on that
    # slice is positive-class recall wearing a different name -- it read 97.7%
    # once, above the full-test score, which is what gave it away. Shuffle once,
    # here, and take every subset through test_subset().
    test_order = np.random.default_rng(SEED + 7).permutation(len(te_x))
    return {
        "train_x": tr_x, "train_y": tr_y,
        "test_x": te_x, "test_y": te_y, "test_order": test_order,
        "fit_x": [tr_x[i] for i in fit_i], "fit_y": tr_y[fit_i],
        "val_x": [tr_x[i] for i in val_i], "val_y": tr_y[val_i],
    }


def test_subset(d, n: int):
    """A class-balanced slice of the test set, the same one every run."""
    idx = d["test_order"][:n]
    return [d["test_x"][i] for i in idx], d["test_y"][idx]


def word_tokens(s: str) -> list[str]:
    """The tokenizer the students write in ten minutes, and its whole spec."""
    return WORD_RE.findall(s.lower().replace("<br />", " "))


# ------------------------------------------------------------- corpus figures

def corpus_facts(d) -> dict:
    tr_tok = [word_tokens(s) for s in d["train_x"]]
    te_tok = [word_tokens(s) for s in d["test_x"]]
    lens = np.array([len(t) for t in tr_tok])
    cnt = Counter(w for t in tr_tok for w in t)

    # the fit-split vocabulary, which is the only one a model may see
    fit_cnt = Counter(w for s in d["fit_x"] for w in word_tokens(s))
    vocab = {w for w, _ in fit_cnt.most_common(WORD_VOCAB - 2)}

    te_flat = [w for t in te_tok for w in t]
    oov = sum(1 for w in te_flat if w not in vocab)

    # how much of a review survives a 256-token cut
    kept = np.minimum(lens, MAXLEN).sum() / lens.sum()

    hapax = sum(1 for _, c in cnt.items() if c == 1)
    return {
        "n_train": len(d["train_x"]), "n_test": len(d["test_x"]),
        "n_fit": N_FIT, "n_val": N_VAL,
        "pos_rate_train": float(d["train_y"].mean()),
        "pos_rate_test": float(d["test_y"].mean()),
        "len_mean": float(lens.mean()), "len_median": float(np.median(lens)),
        "len_p90": float(np.percentile(lens, 90)), "len_max": int(lens.max()),
        "over_maxlen": float((lens > MAXLEN).mean()),
        "tokens_kept": float(kept),
        "distinct_words": len(cnt), "hapax": hapax,
        "hapax_frac": hapax / len(cnt),
        "oov_rate": oov / len(te_flat),
        "n_test_tokens": len(te_flat),
        "lens": lens,
    }


def oov_curve(d) -> dict:
    """OOV rate on the test set against the size of a word vocabulary.

    The subword tokenizer's own rate is measured, not asserted: WordPiece backs
    off to characters, so nothing in the test set can miss.
    """
    from transformers import AutoTokenizer
    fit_cnt = Counter(w for s in d["fit_x"] for w in word_tokens(s))
    ranked = [w for w, _ in fit_cnt.most_common()]
    te_flat = [w for s in d["test_x"] for w in word_tokens(s)]
    te_cnt = Counter(te_flat)
    total = len(te_flat)

    sizes = [2_000, 5_000, 10_000, 20_000, 50_000, len(ranked)]
    rates, types = [], []
    for v in sizes:
        keep = set(ranked[:v])
        covered = sum(c for w, c in te_cnt.items() if w in keep)
        rates.append(1 - covered / total)
        types.append(1 - sum(1 for w in te_cnt if w in keep) / len(te_cnt))

    tk = AutoTokenizer.from_pretrained(BERT)
    sample = [d["test_x"][i] for i in d["test_order"][:2_000]]
    unk = tk.unk_token_id
    n_unk = n_tok = 0
    for s in sample:
        ids = tk(s, truncation=True, max_length=512)["input_ids"]
        n_unk += sum(1 for i in ids if i == unk)
        n_tok += len(ids)
    return {"sizes": sizes, "token_oov": rates, "type_oov": types,
            "subword_vocab": int(tk.vocab_size),
            "subword_oov": n_unk / n_tok, "subword_sample": len(sample),
            "full_vocab": len(ranked)}


def tokenisation_example(d) -> dict:
    """The same sentence under both tokenizers, with a word neither has seen.

    Printed so the hand-drawn diagram quotes real output rather than a
    plausible reconstruction.
    """
    from transformers import AutoTokenizer
    tk = AutoTokenizer.from_pretrained(BERT)
    sentence = "The plot was unwatchable and utterly discombobulating."
    fit_cnt = Counter(w for s in d["fit_x"] for w in word_tokens(s))
    vocab = {w for w, _ in fit_cnt.most_common(WORD_VOCAB - 2)}

    words = word_tokens(sentence)
    word_view = [w if w in vocab else "[UNK]" for w in words]
    pieces = tk.tokenize(sentence)

    counts = {w: fit_cnt.get(w, 0) for w in words}
    return {"sentence": sentence, "words": words, "word_view": word_view,
            "pieces": pieces, "n_words": len(words), "n_pieces": len(pieces),
            "n_unk": sum(1 for w in word_view if w == "[UNK]"),
            "counts": counts,
            "expansion": len(pieces) / len(words)}


def fig_lengths(cf):
    lens = cf["lens"]
    fig, ax = plt.subplots(figsize=(10.4, 3.1))
    ax.hist(np.clip(lens, 0, 1200), bins=80, color=PRIMARY, alpha=0.85)
    ax.axvline(MAXLEN, color=ACCENT, lw=2.2, ls="--")
    ax.annotate(f"cut at {MAXLEN} tokens\n{cf['over_maxlen']*100:.0f}% of reviews "
                f"are longer\nbut {cf['tokens_kept']*100:.0f}% of all words survive",
                xy=(MAXLEN, ax.get_ylim()[1] * 0.30),
                xytext=(470, ax.get_ylim()[1] * 0.42),
                color=ACCENT, fontsize=SMALL,
                bbox=dict(fc="white", ec=ACCENT, lw=1.0, boxstyle="round,pad=0.5"),
                arrowprops=dict(arrowstyle="->", color=ACCENT, lw=1.8))
    ax.set_xlabel("review length, in words")
    ax.set_ylabel("reviews")
    ax.set_title("25,000 training reviews, clipped at 1,200 for the plot")
    fig.tight_layout()
    save(fig, "l21-lengths")


def fig_oov(oc):
    fig, ax = plt.subplots(figsize=(10.4, 3.2))
    x = np.arange(len(oc["sizes"]))
    ax.plot(x, 100 * np.array(oc["token_oov"]), "o-", color=ACCENT, lw=2.4,
            ms=7, label="word tokenizer, tokens unseen")
    ax.plot(x, 100 * np.array(oc["type_oov"]), "s--", color=MUTED, lw=1.8,
            ms=6, label="word tokenizer, distinct words unseen")
    ax.axhline(100 * oc["subword_oov"], color=SUCCESS, lw=2.6)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{v//1000}k" if v >= 1000 else str(v)
                        for v in oc["sizes"]])
    ax.set_xlabel("size of the word vocabulary")
    ax.set_ylabel("out of vocabulary, %")
    ax.annotate(f"subword, {oc['subword_vocab']:,} pieces: "
                f"{100*oc['subword_oov']:.2f}%",
                xy=(4.6, 100 * oc["subword_oov"]), xytext=(3.5, 33),
                color=SUCCESS, fontsize=SMALL, ha="center",
                bbox=dict(fc="white", ec=SUCCESS, lw=1.0,
                          boxstyle="round,pad=0.5"),
                arrowprops=dict(arrowstyle="->", color=SUCCESS, lw=1.8))
    ax.set_ylim(-6, 108)
    ax.legend(loc="center left", bbox_to_anchor=(0.015, 0.60), fontsize=SMALL)
    ax.set_title("measured on the 25,000 test reviews")
    fig.tight_layout()
    save(fig, "l21-oov")


# ------------------------------------------------------------- the baselines

def bow_baselines(d) -> dict:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    out = {}
    for name, ngram in (("unigram", (1, 1)), ("bigram", (1, 2))):
        t0 = time.perf_counter()
        vec = TfidfVectorizer(min_df=2, ngram_range=ngram, max_features=200_000)
        Xtr = vec.fit_transform(d["train_x"])
        clf = LogisticRegression(max_iter=2000, C=4.0).fit(Xtr, d["train_y"])
        acc = float((clf.predict(vec.transform(d["test_x"])) == d["test_y"]).mean())
        out[name] = {"acc": acc, "n_features": int(Xtr.shape[1]),
                     "seconds": time.perf_counter() - t0}
        print(f"      tf-idf {name}: {acc:.4f} "
              f"({Xtr.shape[1]:,} features, {out[name]['seconds']:.0f}s)")
    out["majority"] = float(max(d["test_y"].mean(), 1 - d["test_y"].mean()))
    return out


# --------------------------------------------------------- the recurrent net

def pad_batch(seqs) -> tuple[torch.Tensor, torch.Tensor]:
    X = np.zeros((len(seqs), MAXLEN), dtype=np.int64)
    L = np.zeros(len(seqs), dtype=np.int64)
    for i, s in enumerate(seqs):
        s = s[:MAXLEN]
        X[i, :len(s)] = s
        L[i] = max(len(s), 1)
    return torch.from_numpy(X), torch.from_numpy(L)


def word_encoder(d):
    fit_cnt = Counter(w for s in d["fit_x"] for w in word_tokens(s))
    w2i = {w: i + 2 for i, (w, _) in enumerate(fit_cnt.most_common(WORD_VOCAB - 2))}

    def enc(texts):
        return pad_batch([[w2i.get(w, 1) for w in word_tokens(s)] for s in texts])
    return enc, WORD_VOCAB


def subword_encoder():
    from transformers import AutoTokenizer
    tk = AutoTokenizer.from_pretrained(BERT)

    def enc(texts):
        out = tk(list(texts), truncation=True, max_length=MAXLEN,
                 padding="max_length")
        ids = np.asarray(out["input_ids"], dtype=np.int64)
        lens = np.asarray(out["attention_mask"], dtype=np.int64).sum(1)
        return torch.from_numpy(ids), torch.from_numpy(np.maximum(lens, 1))
    return enc, int(tk.vocab_size)


def pretrained_embeddings(n_rows: int) -> tuple[np.ndarray, float]:
    """DistilBERT's input embedding matrix, projected 768 -> 128 by PCA.

    The projection is not decoration: the whole point of the lecture is that
    the architecture does not change, and the architecture has a 128-wide
    embedding. PCA is thread 5, from Lecture 10, doing work again.
    """
    from sklearn.decomposition import PCA
    from transformers import AutoModel
    m = AutoModel.from_pretrained(BERT)
    E = m.embeddings.word_embeddings.weight.detach().numpy()[:n_rows]
    pca = PCA(n_components=EMB_DIM, random_state=SEED)
    Z = pca.fit_transform(E)
    Z = Z / Z.std() * 0.1                    # match nn.Embedding's default scale
    return Z.astype(np.float32), float(pca.explained_variance_ratio_.sum())


class GRUClassifier(nn.Module):
    """784-style simplicity: embed, run a bidirectional GRU, read the last
    state of each direction, one linear head. Identical in every run."""

    def __init__(self, vocab, init=None, freeze=False, last_of_padding=False):
        super().__init__()
        self.emb = nn.Embedding(vocab, EMB_DIM, padding_idx=0)
        if init is not None:
            self.emb.weight.data.copy_(torch.from_numpy(init))
            self.emb.weight.requires_grad = not freeze
        self.rnn = nn.GRU(EMB_DIM, HIDDEN, batch_first=True, bidirectional=True)
        self.head = nn.Linear(2 * HIDDEN, 2)
        self.last_of_padding = last_of_padding

    def forward(self, x, lengths):
        e = self.emb(x)
        if self.last_of_padding:
            # the assistant's version: read position -1 of a padded sequence
            out, _ = self.rnn(e)
            return self.head(out[:, -1, :])
        packed = nn.utils.rnn.pack_padded_sequence(
            e, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, h = self.rnn(packed)
        return self.head(torch.cat([h[0], h[1]], dim=1))


@torch.no_grad()
def rnn_accuracy(net, X, L, y, batch=256) -> float:
    net.eval()
    hits = 0
    for i in range(0, len(X), batch):
        out = net(X[i:i + batch].to(DEVICE), L[i:i + batch])
        hits += int((out.argmax(1).cpu().numpy() == y[i:i + batch]).sum())
    return hits / len(y)


def train_rnn(d, enc, vocab, *, init=None, freeze=False, last_of_padding=False,
              double_softmax=False, epochs=EPOCHS, tag="") -> dict:
    """One run. Same seed, same optimiser, same epochs, every time."""
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    Xf, Lf = enc(d["fit_x"])
    Xv, Lv = enc(d["val_x"])
    Xt, Lt = enc(d["test_x"])
    yf = torch.from_numpy(d["fit_y"])

    net = GRUClassifier(vocab, init=init, freeze=freeze,
                        last_of_padding=last_of_padding).to(DEVICE)
    trainable = [p for p in net.parameters() if p.requires_grad]
    opt = torch.optim.Adam(trainable, lr=LR)
    lossf = nn.CrossEntropyLoss()

    Xf_d, yf_d = Xf.to(DEVICE), yf.to(DEVICE)
    n_par = sum(p.numel() for p in net.parameters())
    n_trainable = sum(p.numel() for p in trainable)

    curve, losses = [], []
    best_state, best_val, best_epoch = None, -1.0, 0
    t0 = time.perf_counter()
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(len(Xf_d))
        running = 0.0
        for i in range(0, len(Xf_d), BATCH):
            j = perm[i:i + BATCH]
            opt.zero_grad()
            out = net(Xf_d[j], Lf[j])
            if double_softmax:
                # the assistant's version: probabilities into a loss that
                # expects logits
                out = torch.softmax(out, dim=1)
            loss = lossf(out, yf_d[j])
            loss.backward()
            opt.step()
            running += float(loss.detach()) * len(j)
        losses.append(running / len(Xf_d))
        curve.append(rnn_accuracy(net, Xv, Lv, d["val_y"]))
        if curve[-1] > best_val:
            best_val, best_epoch = curve[-1], ep + 1
            best_state = {k: v.detach().cpu().clone()
                          for k, v in net.state_dict().items()}
        print(f"        {tag} epoch {ep+1}: train loss {losses[-1]:.4f}  "
              f"val {curve[-1]:.4f}  ({time.perf_counter()-t0:.0f}s)")
    seconds = time.perf_counter() - t0

    # Early stopping on validation, as in Lecture 6. Reporting the last epoch
    # instead cost the pretrained-embedding run 3.5 points here: its training
    # loss reaches 0.03 by epoch 4 and it has memorised the fit split.
    test_last = rnn_accuracy(net, Xt, Lt, d["test_y"])
    net.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})
    test_acc = rnn_accuracy(net, Xt, Lt, d["test_y"])
    print(f"        {tag}: best val at epoch {best_epoch} -> test {test_acc:.4f}"
          f"   (last epoch would have been {test_last:.4f})")
    return {"val_curve": curve, "train_loss": losses, "best_val": best_val,
            "best_epoch": best_epoch, "test_acc": test_acc,
            "test_acc_last_epoch": test_last,
            "seconds": seconds, "epochs": epochs,
            "n_params": n_par, "n_trainable": n_trainable, "vocab": vocab}


# ---------------------------------------------------- thread 11 — the numbers

def softmax_shift() -> dict:
    """Softmax is invariant under a constant shift — and float32 is not."""
    z = np.array([1000.0, 1001.0, 1002.0], dtype=np.float32)
    naive_num = np.exp(z)                            # inf, inf, inf
    naive = naive_num / naive_num.sum()              # nan
    shifted = np.exp(z - z.max()) / np.exp(z - z.max()).sum()

    small = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    a = np.exp(small) / np.exp(small).sum()
    b = np.exp(small + 50) / np.exp(small + 50).sum()
    # inf and nan are exactly the point here, but they are not JSON, so the
    # broken column is exported as the strings a student would see printed.
    show = lambda a: [("inf" if np.isposinf(v) else "nan" if np.isnan(v)
                       else float(v)) for v in a]
    return {
        "z": z.tolist(),
        "naive_numerator": show(naive_num),
        "naive": show(naive),
        "shifted": [float(v) for v in shifted],
        "small": small.tolist(),
        "p_small": [float(v) for v in a],
        "p_shifted_50": [float(v) for v in b],
        "max_shift_diff": float(np.abs(a - b).max()),
        "overflow_at": float(np.log(np.finfo(np.float32).max)),
    }


def gradient_check() -> dict:
    """d(cross-entropy)/d(logits) = softmax(z) - y, verified against autograd."""
    torch.manual_seed(SEED)
    z = torch.randn(7, 5, dtype=torch.float64, requires_grad=True)
    y = torch.randint(0, 5, (7,))
    loss = nn.CrossEntropyLoss(reduction="sum")(z, y)
    loss.backward()

    p = torch.softmax(z.detach(), dim=1)
    onehot = torch.zeros_like(p).scatter_(1, y[:, None], 1.0)
    analytic = p - onehot
    err = float((z.grad - analytic).abs().max())

    row = {"logits": [float(v) for v in z[0].detach()],
           "probs": [float(v) for v in p[0]],
           "target": int(y[0]),
           "grad": [float(v) for v in z.grad[0]]}
    return {"max_abs_error": err, "row": row,
            "grad_sums_to_zero": float(analytic.sum(1).abs().max())}


def ce_stability() -> dict:
    """Compute the loss both ways in float32 and find where naive breaks.

    naive  :  p = softmax(z);  L = -log(p[y])
    stable :  L = -(z[y] - logsumexp(z))
    The float64 stable form is the reference both are scored against.

    The sweep is over **how wrong the model is**, not over the scale of the
    logits, because that is the quantity the failure depends on. A row whose
    true class sits `gap` below the maximum has a loss of at least `gap`, and
    the naive form has to represent exp(-gap) as a float32. Sweeping the
    logits' standard deviation instead buries the effect: most rows then have a
    loss near zero, where relative error is meaningless and both forms score
    the same.
    """
    rng = np.random.default_rng(SEED)
    K, N = 10, 2_000
    gaps = [1, 5, 10, 20, 40, 60, 80, 90, 100, 110]
    idx = np.arange(N)
    rows = []
    for gap in gaps:
        z64 = rng.normal(0, 1.0, size=(N, K))
        y = rng.integers(0, K, size=N)
        z64[idx, y] = z64.max(1) - gap          # the true class, `gap` below
        z32 = z64.astype(np.float32)

        lse64 = z64.max(1) + np.log(np.exp(z64 - z64.max(1, keepdims=True)).sum(1))
        ref = -(z64[idx, y] - lse64)            # >= gap, so relative error is safe

        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            e = np.exp(z32)
            p = e / e.sum(1, keepdims=True)
            naive = -np.log(p[idx, y])
        m = z32.max(1, keepdims=True)
        stable = -(z32[idx, y] - (m[:, 0] + np.log(np.exp(z32 - m).sum(1))))

        def score(a):
            a = a.astype(np.float64)
            ok = np.isfinite(a)
            return (float(1 - ok.mean()),
                    float(np.median(np.abs(a[ok] - ref[ok]) / ref[ok]))
                    if ok.any() else None)

        n_bad, n_err = score(naive)
        s_bad, s_err = score(stable)
        rows.append({"gap": gap, "mean_loss": float(ref.mean()),
                     "naive_nonfinite": n_bad, "naive_rel_err": n_err,
                     "stable_nonfinite": s_bad, "stable_rel_err": s_err})
        print(f"      gap {gap:>3}: naive non-finite {n_bad*100:5.1f}%  "
              f"median rel err  naive {n_err}  stable {s_err}")

    # Two concrete rows, small enough to print on a slide and check by hand.
    # The first is the loud failure; the second is the quiet one, and it is the
    # quiet one that ships.
    lse = lambda v: v.max() + np.log(np.exp(v - v.max()).sum())
    finite = lambda v: float(v) if np.isfinite(v) else (
        "inf" if np.isposinf(v) else "-inf" if np.isneginf(v) else "nan")

    def one_row(logits, tgt=2):
        z = np.array(logits, dtype=np.float32)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            e = np.exp(z)
            p32 = e / e.sum()
            naive = float(-np.log(p32[tgt]))
        stable = float(-(z[tgt] - lse(z)))
        z64 = z.astype(np.float64)
        exact = float(-(z64[tgt] - lse(z64)))
        torch_val = float(nn.CrossEntropyLoss()(torch.tensor(z)[None, :],
                                                torch.tensor([tgt])))
        return {"logits": [float(v) for v in z], "target": tgt,
                "p_true": finite(p32[tgt]),
                "denominator": finite(e.sum()),
                "naive": finite(naive), "stable": stable, "exact": exact,
                "naive_err": finite(abs(naive - exact)),
                "stable_err": abs(stable - exact),
                "torch": torch_val, "torch_err": abs(torch_val - exact)}

    loud = one_row([100.0, 0.0, -100.0])
    quiet = one_row([0.0, 0.0, -100.0])
    print(f"      loud : naive {loud['naive']}  stable {loud['stable']:.6f}  "
          f"exact {loud['exact']:.6f}")
    print(f"      quiet: naive {quiet['naive']}  stable {quiet['stable']:.6f}  "
          f"exact {quiet['exact']:.6f}  naive off by {quiet['naive_err']}")
    return {"rows": rows, "loud": loud, "quiet": quiet,
            "torch_agrees": loud["torch_err"]}


def ce_vs_kl() -> dict:
    """H(p, q) = H(p) + KL(p || q), and H(p) = 0 for a one-hot target."""
    rng = np.random.default_rng(SEED)
    q = rng.dirichlet(np.ones(5))
    onehot = np.zeros(5); onehot[2] = 1.0
    smooth = np.full(5, 0.1 / 4); smooth[2] = 0.9

    def H(p):
        nz = p > 0
        return float(-(p[nz] * np.log(p[nz])).sum())

    def CE(p, q):
        return float(-(p * np.log(q)).sum())

    def KL(p, q):
        nz = p > 0
        return float((p[nz] * np.log(p[nz] / q[nz])).sum())

    return {
        "q": [float(v) for v in q],
        "onehot": {"H": H(onehot), "CE": CE(onehot, q), "KL": KL(onehot, q)},
        "smooth": {"H": H(smooth), "CE": CE(smooth, q), "KL": KL(smooth, q),
                   "H_plus_KL": H(smooth) + KL(smooth, q)},
        "residual": abs(CE(smooth, q) - H(smooth) - KL(smooth, q)),
    }


def fig_softmax_shift(ss):
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.0), sharey=True)
    x = np.arange(3)
    for ax, p, title, c in (
            (axes[0], ss["p_small"], "logits  1, 2, 3", PRIMARY),
            (axes[1], ss["p_shifted_50"], "logits  51, 52, 53", SUCCESS)):
        ax.bar(x, p, color=c, width=0.6)
        ax.set_xticks(x); ax.set_xticklabels(["class 0", "class 1", "class 2"])
        ax.set_title(title)
        for xi, pi in zip(x, p):
            ax.text(xi, pi + 0.02, f"{pi:.4f}", ha="center", fontsize=SMALL,
                    color=MUTED)
    axes[0].set_ylabel("softmax probability")
    axes[0].set_ylim(0, 0.85)
    fig.suptitle(f"identical to {ss['max_shift_diff']:.0e} — adding a constant "
                 f"to every logit changes nothing",
                 fontsize=SMALL, color=MUTED, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save(fig, "l22-softmax-shift")


def fig_ce_stability(cs):
    rows = cs["rows"]
    g = np.array([r["gap"] for r in rows], dtype=float)
    frac = np.array([r["naive_nonfinite"] for r in rows])

    # Where the naive form returns nothing there is no error to plot, so its
    # line stops. Substituting a floor there would draw a number that does not
    # exist -- and would draw it going the reassuring way.
    n_ok = np.array([r["naive_rel_err"] is not None for r in rows])
    naive = np.array([r["naive_rel_err"] if ok else np.nan
                      for r, ok in zip(rows, n_ok)], dtype=float)
    stable = np.array([r["stable_rel_err"] for r in rows], dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.2))

    ax = axes[0]
    ax.plot(g[n_ok], naive[n_ok], "o-", color=ACCENT, lw=2.4, ms=7,
            label="naive")
    ax.plot(g, stable, "s-", color=SUCCESS, lw=2.4, ms=6, label="combined")
    ax.set_yscale("log")
    ax.set_ylim(6e-9, 3e-3)
    # after plotting, not before: set_yscale reinstalls the mathtext formatter
    plain_log(ax, "y", [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3], fmt="{:.0e}")
    ax.set_xlabel("true loss of the row, in nats")
    ax.set_ylabel("median relative error")
    ax.legend(loc="upper left", fontsize=SMALL)
    ax.set_title("where it answers", fontsize=SMALL)
    if (~n_ok).any():
        ax.annotate("the naive line stops where\nthere is nothing left to score",
                    xy=(55, 1.3e-6), ha="center", va="center",
                    fontsize=SMALL, color=MUTED)

    ax = axes[1]
    ax.plot(g, 100 * frac, "o-", color=ACCENT, lw=2.4, ms=7, label="naive")
    ax.plot(g, 100 * np.array([r["stable_nonfinite"] for r in rows]), "s-",
            color=SUCCESS, lw=2.4, ms=6, label="combined")
    ax.set_xlabel("true loss of the row, in nats")
    ax.set_ylabel("rows returning inf, %")
    ax.set_ylim(-5, 108)
    ax.legend(loc="center left", fontsize=SMALL)
    ax.set_title("whether it answers at all", fontsize=SMALL)
    first = int(np.argmax(frac > 0)) if (frac > 0).any() else None
    if first is not None:
        ax.annotate(f"every row, past\n{g[first]:.0f} nats", xy=(g[first], 100),
                    xytext=(g[first] - 6, 78), ha="right", fontsize=SMALL,
                    color=ACCENT,
                    arrowprops=dict(arrowstyle="->", color=ACCENT, lw=1.8))

    fig.tight_layout(w_pad=2.5)
    save(fig, "l22-ce-stability")


def fig_gradient(gc):
    r = gc["row"]
    x = np.arange(5)
    onehot = np.zeros(5); onehot[r["target"]] = 1.0
    fig, ax = plt.subplots(figsize=(10.4, 3.1))
    ax.bar(x - 0.21, r["probs"], width=0.4, color=PRIMARY, label="softmax(z)")
    ax.bar(x + 0.21, onehot, width=0.4, color=RULE, label="one-hot target y")
    ax.plot(x, r["grad"], "o", color=MATH, ms=11, zorder=5,
            label="∂L/∂z from autograd")
    ax.axhline(0, color=AXIS, lw=1.0)
    ax.set_xticks(x); ax.set_xticklabels([f"class {i}" for i in x])
    ax.set_ylabel("probability / gradient")
    ax.legend(loc="upper left", ncols=3)
    ax.set_ylim(-1.15, 1.5)
    ax.set_title("one row; the dots sit exactly on blue minus grey")
    fig.tight_layout()
    save(fig, "l22-gradient")


# ------------------------------------------------------------- the fine-tune

def finetune_distilbert(d, n_fit=N_FIT, epochs=2, batch=16, lr=2e-5) -> dict:
    """Fine-tune, scoring validation AND test after every epoch.

    The number of epochs is a hyperparameter, so it is chosen on the validation
    split like every other one in this course. Both epochs' test scores are
    recorded so the deck can show that the choice was not made on the test set —
    but the headline is whichever epoch validation preferred.
    """
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    torch.manual_seed(SEED)
    tk = AutoTokenizer.from_pretrained(BERT)
    model = AutoModelForSequenceClassification.from_pretrained(
        BERT, num_labels=2).to(DEVICE)

    def encode(texts):
        out = tk(list(texts), truncation=True, max_length=MAXLEN,
                 padding="max_length", return_tensors="pt")
        return out["input_ids"], out["attention_mask"]

    ids, am = encode(d["fit_x"][:n_fit])
    y = torch.from_numpy(d["fit_y"][:n_fit])
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    lossf = nn.CrossEntropyLoss()

    ids_v, am_v = encode(d["val_x"])
    ids_t, am_t = encode(d["test_x"])

    @torch.no_grad()
    def score(i_all, a_all, labels, chunk=64):
        model.eval()
        preds = np.zeros(len(i_all), dtype=np.int64)
        for i in range(0, len(i_all), chunk):
            out = model(input_ids=i_all[i:i + chunk].to(DEVICE),
                        attention_mask=a_all[i:i + chunk].to(DEVICE)).logits
            preds[i:i + chunk] = out.argmax(1).cpu().numpy()
        return float((preds == labels).mean()), preds

    val_curve, test_curve, epoch_seconds = [], [], []
    t0 = time.perf_counter()
    preds = None
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(ids))
        for k, i in enumerate(range(0, len(ids), batch)):
            j = perm[i:i + batch]
            opt.zero_grad()
            out = model(input_ids=ids[j].to(DEVICE),
                        attention_mask=am[j].to(DEVICE)).logits
            loss = lossf(out, y[j].to(DEVICE))
            loss.backward()
            opt.step()
            if k % 200 == 0:
                print(f"        step {k}: loss {float(loss.detach()):.4f} "
                      f"({time.perf_counter()-t0:.0f}s)")
        epoch_seconds.append(time.perf_counter() - t0)
        v, _ = score(ids_v, am_v, d["val_y"])
        t1 = time.perf_counter()
        a, preds = score(ids_t, am_t, d["test_y"])
        val_curve.append(v)
        test_curve.append(a)
        print(f"        epoch {ep+1}: val {v:.4f}  test {a:.4f} "
              f"({epoch_seconds[-1]:.0f}s train, "
              f"{time.perf_counter()-t1:.0f}s to score the test set)")

    # the epoch is a hyperparameter: chosen on validation, never on test
    best = int(np.argmax(val_curve))
    acc = test_curve[best]
    n_par = sum(p.numel() for p in model.parameters())
    print(f"      DistilBERT fine-tuned: validation prefers epoch {best+1}, "
          f"test {acc:.4f}")
    return {"test_acc": acc, "chosen_epoch": best + 1,
            "val_curve": val_curve, "test_curve": test_curve,
            "train_seconds": epoch_seconds[best],
            "test_seconds": time.perf_counter() - t1, "n_params": n_par,
            "n_fit": n_fit, "epochs": epochs, "batch": batch, "lr": lr,
            "errors": int(round((1 - acc) * len(d["test_y"])))}


def label_efficiency(d, sizes=(200, 1_000, 5_000, 20_000), n_test=5_000,
                     checkpoints=(100, 200, 400, 800, 1600), batch=16,
                     n_val=2_000) -> dict:
    """Where the pretrained model actually earns its place: few labels.

    At 20,000 labelled reviews a linear model on bigrams is nearly as good, so
    the honest question is not "is the transformer better" but "better at what
    budget".

    The trap here is the training protocol, and both obvious choices are unfair
    at one end. A fixed number of *epochs* gives the 200-review model 13
    optimisation steps, which does not train the randomly initialised head at
    all -- that measures undertraining and calls it label efficiency. A fixed
    number of *steps* then undertrains the 20,000-review model, because 400
    steps is a third of an epoch there. Measured, those two protocols disagree
    about the sign of the effect at 20,000 labels.

    So neither is used. Each model gets **its own budget, chosen on the
    validation split** at every size, exactly as every other hyperparameter in
    this course: the transformer's number of steps from `checkpoints`, and the
    logistic regression's regularisation from a small grid. The test subset is
    scored once, at the chosen budget.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tk = AutoTokenizer.from_pretrained(BERT)
    test_x, test_y = test_subset(d, n_test)
    val_x, val_y = d["val_x"][:n_val], d["val_y"][:n_val]

    def encode(texts):
        out = tk(list(texts), truncation=True, max_length=MAXLEN,
                 padding="max_length", return_tensors="pt")
        return out["input_ids"], out["attention_mask"]

    ids_v, am_v = encode(val_x)
    ids_t, am_t = encode(test_x)

    @torch.no_grad()
    def predict(model, i_all, a_all, chunk=64):
        model.eval()
        out = np.zeros(len(i_all), dtype=np.int64)
        for i in range(0, len(i_all), chunk):
            lg = model(input_ids=i_all[i:i + chunk].to(DEVICE),
                       attention_mask=a_all[i:i + chunk].to(DEVICE)).logits
            out[i:i + chunk] = lg.argmax(1).cpu().numpy()
        return out

    bow, bert, chosen_C, chosen_steps = [], [], [], []
    for n in sizes:
        fx, fy = d["fit_x"][:n], d["fit_y"][:n]

        # --- the linear model: C chosen on validation
        vec = TfidfVectorizer(min_df=1, ngram_range=(1, 2), max_features=200_000)
        Z, Zv, Zt = vec.fit_transform(fx), None, None
        Zv, Zt = vec.transform(val_x), vec.transform(test_x)
        best = (-1.0, None, None)
        for C in (0.25, 1.0, 4.0, 16.0):
            clf = LogisticRegression(max_iter=2000, C=C).fit(Z, fy)
            v = float((clf.predict(Zv) == val_y).mean())
            if v > best[0]:
                best = (v, C, float((clf.predict(Zt) == test_y).mean()))
        bow.append(best[2]); chosen_C.append(best[1])

        # --- the transformer: number of steps chosen on validation
        torch.manual_seed(SEED)
        g = torch.Generator().manual_seed(SEED)
        model = AutoModelForSequenceClassification.from_pretrained(
            BERT, num_labels=2).to(DEVICE)
        opt = torch.optim.AdamW(model.parameters(), lr=2e-5)
        lossf = nn.CrossEntropyLoss()
        ids, am = encode(fx)
        yb = torch.from_numpy(fy)

        done, best_b = 0, (-1.0, None, None)
        for target in checkpoints:
            model.train()
            for _ in range(target - done):
                j = torch.randint(0, len(ids), (batch,), generator=g)
                opt.zero_grad()
                out = model(input_ids=ids[j].to(DEVICE),
                            attention_mask=am[j].to(DEVICE)).logits
                lossf(out, yb[j].to(DEVICE)).backward()
                opt.step()
            done = target
            v = float((predict(model, ids_v, am_v) == val_y).mean())
            if v > best_b[0]:
                best_b = (v, target,
                          float((predict(model, ids_t, am_t) == test_y).mean()))
        bert.append(best_b[2]); chosen_steps.append(best_b[1])
        del model

        print(f"      {n:>6,} labels: bag of words {bow[-1]:.4f} (C={chosen_C[-1]})"
              f"   DistilBERT {bert[-1]:.4f} ({chosen_steps[-1]} steps)"
              f"   gap {100*(bert[-1]-bow[-1]):+.2f} pts")

    return {"sizes": list(sizes), "bow": bow, "bert": bert, "n_test": n_test,
            "n_val": n_val, "batch": batch, "chosen_C": chosen_C,
            "chosen_steps": chosen_steps, "checkpoints": list(checkpoints),
            "test_pos_rate": float(test_y.mean()),
            "gaps": [100 * (b - w) for b, w in zip(bert, bow)]}


def fig_label_efficiency(le):
    fig, ax = plt.subplots(figsize=(10.4, 3.1))
    x = np.array(le["sizes"], dtype=float)
    ax.plot(x, 100 * np.array(le["bow"]), "s--", color=MUTED, lw=2.2, ms=7,
            label="tf-idf + logistic regression")
    ax.plot(x, 100 * np.array(le["bert"]), "o-", color=SUCCESS, lw=2.6, ms=8,
            label="DistilBERT, fine-tuned")
    ax.set_xscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(v):,}" for v in x])
    ax.set_xlabel("labelled reviews used to fit")
    ax.set_ylabel("test accuracy, %")
    lo = 100 * min(min(le["bow"]), min(le["bert"]))
    hi = 100 * max(max(le["bow"]), max(le["bert"]))
    ax.set_ylim(lo - 4, hi + 8)          # headroom for the legend, above the data
    ax.legend(loc="upper left", fontsize=SMALL)
    i = 0
    ax.annotate(f"{le['gaps'][i]:+.1f} points at {le['sizes'][i]:,} labels",
                xy=(x[i], 100 * (le["bow"][i] + le["bert"][i]) / 2),
                xytext=(x[1] * 2.6, 100 * le["bow"][0] - 1.0),
                color=SUCCESS, fontsize=SMALL, ha="center",
                bbox=dict(fc="white", ec=SUCCESS, lw=1.0,
                          boxstyle="round,pad=0.5"),
                arrowprops=dict(arrowstyle="->", color=SUCCESS, lw=1.8))
    ax.set_title(f"same reviews, same {le['n_test']:,} test reviews; each "
                 f"model's budget chosen on validation")
    fig.tight_layout()
    save(fig, "l22-label-efficiency")


def zero_shot_head(d, n=5_000) -> dict:
    """The same backbone with an untrained head, to show what fine-tuning did."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    torch.manual_seed(SEED)
    tk = AutoTokenizer.from_pretrained(BERT)
    model = AutoModelForSequenceClassification.from_pretrained(
        BERT, num_labels=2).to(DEVICE).eval()
    tx, ty = test_subset(d, n)
    out = tk(tx, truncation=True, max_length=MAXLEN,
             padding="max_length", return_tensors="pt")
    preds = np.zeros(n, dtype=np.int64)
    with torch.no_grad():
        for i in range(0, n, 64):
            lg = model(input_ids=out["input_ids"][i:i + 64].to(DEVICE),
                       attention_mask=out["attention_mask"][i:i + 64].to(DEVICE)).logits
            preds[i:i + 64] = lg.argmax(1).cpu().numpy()
    return {"acc": float((preds == ty).mean()), "n": n,
            "pos_rate": float(ty.mean())}


def inference_cost(d, repeats=5) -> dict:
    """Seconds to score every test review, from raw strings.

    The brief says "within the hour", which makes inference cost a stated
    requirement -- and the deck quoted accuracy against it without ever timing
    anything. This times all four models the same way, end to end: tokenising
    or vectorising the raw text *and* the forward pass, because that is what the
    desk actually pays per review.

    Two things make it cheap enough to do properly. Inference cost depends on
    the architecture, the vocabulary and the data, not on the values in the
    weights, so the networks here are constructed rather than retrained. And a
    wall clock on a shared machine is the least reproducible number in this
    course, so the **median** of `repeats` full passes is reported rather than
    one pass -- the mean is dominated by whichever pass collided with another
    job.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    texts, n = d["test_x"], len(d["test_x"])
    out: dict = {"n_reviews": n, "repeats": repeats, "device": DEVICE}

    def record(name, fn):
        runs = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            fn()
            runs.append(time.perf_counter() - t0)
        runs.sort()
        med = runs[len(runs) // 2]
        out[name] = {"seconds": med, "seconds_min": runs[0],
                     "seconds_max": runs[-1],
                     "ms_per_review": 1000 * med / n,
                     "spread_pct": 100 * (runs[-1] - runs[0]) / med}
        print(f"      {name:22s} {med:7.1f} s   "
              f"{1000*med/n:6.2f} ms/review   "
              f"(min {runs[0]:.1f}, max {runs[-1]:.1f})")

    # --- bag of words: fitted once, untimed; only scoring is timed
    for tag, ngram in (("bow_unigram", (1, 1)), ("bow", (1, 2))):
        vec = TfidfVectorizer(min_df=2, ngram_range=ngram, max_features=200_000)
        Xtr = vec.fit_transform(d["train_x"])
        clf = LogisticRegression(max_iter=2000, C=4.0).fit(Xtr, d["train_y"])
        record(tag, (lambda v, c: lambda: c.predict(v.transform(texts)))(vec, clf))

    @torch.no_grad()
    def run_rnn(net, enc):
        X, L = enc(texts)
        net.eval()
        for i in range(0, len(X), 256):
            net(X[i:i + 256].to(DEVICE), L[i:i + 256])

    enc_w, v_w = word_encoder(d)
    net_w = GRUClassifier(v_w).to(DEVICE)
    record("gru_word", lambda: run_rnn(net_w, enc_w))

    enc_s, v_s = subword_encoder()
    net_s = GRUClassifier(v_s).to(DEVICE)
    record("gru_subword", lambda: run_rnn(net_s, enc_s))

    tk = AutoTokenizer.from_pretrained(BERT)
    bert = AutoModelForSequenceClassification.from_pretrained(
        BERT, num_labels=2).to(DEVICE).eval()

    @torch.no_grad()
    def run_bert():
        enc = tk(list(texts), truncation=True, max_length=MAXLEN,
                 padding="max_length", return_tensors="pt")
        for i in range(0, n, 64):
            bert(input_ids=enc["input_ids"][i:i + 64].to(DEVICE),
                 attention_mask=enc["attention_mask"][i:i + 64].to(DEVICE))

    record("distilbert", run_bert)

    out["bert_over_bow"] = out["distilbert"]["seconds"] / out["bow"]["seconds"]
    out["bert_over_gru"] = out["distilbert"]["seconds"] / out["gru_word"]["seconds"]
    print(f"      DistilBERT is {out['bert_over_bow']:.0f}x the bag of words "
          f"and {out['bert_over_gru']:.1f}x the GRU")
    return out


# ------------------------------------------------- embeddings, search, groups

def sentence_embeddings(texts, batch=64) -> np.ndarray:
    from transformers import AutoModel, AutoTokenizer
    tk = AutoTokenizer.from_pretrained(MINILM)
    m = AutoModel.from_pretrained(MINILM).to(DEVICE).eval()
    vecs = []
    with torch.no_grad():
        for i in range(0, len(texts), batch):
            enc = tk(list(texts[i:i + batch]), truncation=True, max_length=256,
                     padding=True, return_tensors="pt").to(DEVICE)
            out = m(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).float()
            pooled = (out * mask).sum(1) / mask.sum(1)
            pooled = torch.nn.functional.normalize(pooled, dim=1)
            vecs.append(pooled.cpu().numpy())
    return np.vstack(vecs)


def semantic_search(d, n=5_000) -> dict:
    """Search the negative reviews for a complaint that shares no words."""
    neg = [t for t, y in zip(d["test_x"], d["test_y"]) if y == 0][:n]
    t0 = time.perf_counter()
    V = sentence_embeddings(neg)
    seconds = time.perf_counter() - t0

    queries = ["the acting was wooden and unconvincing",
               "the sound mix made the dialogue impossible to follow",
               "far too long, it should have ended an hour earlier"]
    Q = sentence_embeddings(queries)
    sims = Q @ V.T

    # the keyword baseline the search has to beat
    from sklearn.feature_extraction.text import TfidfVectorizer
    vec = TfidfVectorizer(min_df=2).fit(neg)
    Vk = vec.transform(neg)
    Qk = vec.transform(queries)
    ksim = (Qk @ Vk.T).toarray()

    hits = []
    for i, q in enumerate(queries):
        top = np.argsort(-sims[i])[:3]
        ktop = np.argsort(-ksim[i])[:3]
        overlap = len(set(top.tolist()) & set(ktop.tolist()))
        hits.append({
            "query": q,
            "top_sim": [float(sims[i, j]) for j in top],
            "top_text": [neg[j][:260] for j in top],
            "keyword_top_sim": [float(ksim[i, j]) for j in ktop],
            "keyword_top_text": [neg[j][:260] for j in ktop],
            "overlap_at_3": overlap,
        })
    return {"n": len(neg), "dim": int(V.shape[1]), "seconds": seconds,
            "hits": hits, "embeddings": V, "texts": neg}


def cluster_complaints(sr, ks=(3, 4, 5, 6, 8, 10, 12)) -> dict:
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from sklearn.feature_extraction.text import TfidfVectorizer

    V, texts = sr["embeddings"], sr["texts"]
    sil = {}
    for k in ks:
        km = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(V)
        sil[k] = float(silhouette_score(V, km.labels_, sample_size=3_000,
                                        random_state=SEED))
        print(f"      k={k:2d}  silhouette {sil[k]:.4f}")
    best_k = max(sil, key=sil.get)
    km = KMeans(n_clusters=best_k, n_init=10, random_state=SEED).fit(V)

    vec = TfidfVectorizer(min_df=5, max_df=0.4, stop_words="english",
                          ngram_range=(1, 2))
    X = vec.fit_transform(texts)
    names = np.array(vec.get_feature_names_out())
    groups = []
    for c in range(best_k):
        m = km.labels_ == c
        mean_in = np.asarray(X[m].mean(0)).ravel()
        mean_out = np.asarray(X[~m].mean(0)).ravel()
        lift = mean_in - mean_out
        top = names[np.argsort(-lift)[:6]].tolist()
        groups.append({"size": int(m.sum()), "terms": top,
                       "example": texts[np.argmax(m * 1.0)][:220]})
    groups.sort(key=lambda g: -g["size"])
    return {"silhouette": sil, "best_k": best_k, "groups": groups,
            "n": len(texts), "labels": km.labels_.tolist()}


def fig_clusters(cl):
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.2),
                             gridspec_kw={"width_ratios": [1, 1.35]})
    ks = sorted(cl["silhouette"])
    vals = [cl["silhouette"][k] for k in ks]
    axes[0].plot(ks, vals, "o-", color=PRIMARY, lw=2.4, ms=7)
    best = cl["best_k"]
    axes[0].plot([best], [cl["silhouette"][best]], "o", color=SUCCESS, ms=13,
                 zorder=5)
    axes[0].annotate(f"k = {best}", xy=(best, cl["silhouette"][best]),
                     xytext=(best + 1.4, cl["silhouette"][best] - 0.0015),
                     color=SUCCESS, fontsize=SMALL)
    axes[0].set_xlabel("clusters, k"); axes[0].set_ylabel("silhouette")
    axes[0].set_title("choose k as in Lecture 9")

    sizes = [g["size"] for g in cl["groups"]]
    # bigram terms get long; a y-label wider than the axes shrinks the whole
    # panel and takes the tick labels under the 15px floor with it
    def _label(terms):
        out = []
        for t in terms:
            if sum(len(x) + 2 for x in out) + len(t) > 30:
                break
            out.append(t)
        return ", ".join(out or terms[:1])
    labels = [_label(g["terms"]) for g in cl["groups"]]
    y = np.arange(len(sizes))[::-1]
    axes[1].barh(y, sizes, color=PRIMARY, height=0.62)
    axes[1].set_yticks(y); axes[1].set_yticklabels(labels, fontsize=SMALL)
    axes[1].set_xlabel("negative reviews in the group")
    axes[1].set_title("the groups, named by their own words")
    axes[1].grid(axis="y", visible=False)
    fig.tight_layout()
    save(fig, "l22-clusters")


# ------------------------------------------------------------------ the leak

def tokeniser_leak(d, n_seeds=20) -> dict:
    """Fit the vectoriser on everything, then split — at two corpus sizes.

    The point is the contrast. At 25,000 documents the idf vector is a stable
    statistic and the leak is nearly free; at 400 it is not, and the same
    three lines of code buy a visible amount of accuracy that does not exist.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split

    all_x = list(d["train_x"]) + list(d["test_x"])
    all_y = np.concatenate([d["train_y"], d["test_y"]])

    out = {}
    for label, n_docs, seeds in (("small", 400, n_seeds), ("full", 25_000, 5)):
        leaky, honest = [], []
        for s in range(seeds):
            rng = np.random.default_rng(SEED + s)
            idx = rng.choice(len(all_x), size=n_docs, replace=False)
            X = [all_x[i] for i in idx]
            y = all_y[idx]
            tr_i, te_i = train_test_split(np.arange(n_docs), test_size=0.25,
                                          random_state=SEED + s, stratify=y)

            # what the assistant wrote: fit on everything, then split
            v_bad = TfidfVectorizer(min_df=1, ngram_range=(1, 2),
                                    max_features=60_000)
            Z = v_bad.fit_transform(X)
            clf = LogisticRegression(max_iter=2000, C=4.0).fit(Z[tr_i], y[tr_i])
            leaky.append(float((clf.predict(Z[te_i]) == y[te_i]).mean()))

            # what it should have written: split, then fit on the training half
            v_ok = TfidfVectorizer(min_df=1, ngram_range=(1, 2),
                                   max_features=60_000)
            Ztr = v_ok.fit_transform([X[i] for i in tr_i])
            clf = LogisticRegression(max_iter=2000, C=4.0).fit(Ztr, y[tr_i])
            Zte = v_ok.transform([X[i] for i in te_i])
            honest.append(float((clf.predict(Zte) == y[te_i]).mean()))

            if label == "small" and s == 0:
                out["small_vocab_leaky"] = int(Z.shape[1])
                out["small_vocab_honest"] = int(Ztr.shape[1])
        leaky, honest = np.array(leaky), np.array(honest)
        gap = leaky - honest
        out[label] = {
            "n_docs": n_docs, "seeds": seeds,
            "leaky_mean": float(leaky.mean()), "leaky_sd": float(leaky.std()),
            "honest_mean": float(honest.mean()), "honest_sd": float(honest.std()),
            "gap_mean": float(gap.mean()), "gap_sd": float(gap.std()),
            "gap_points": float(100 * gap.mean()),
            "seeds_leak_wins": int((gap > 0).sum()),
            "gaps": [float(g) for g in gap],
        }
        print(f"      {label} (n={n_docs}, {seeds} seeds): leaky "
              f"{leaky.mean():.4f} honest {honest.mean():.4f}  "
              f"gap {100*gap.mean():+.2f} points (sd {100*gap.std():.2f})")
    return out


def _tfidf_score(X, y, tr, te) -> float:
    """Fit the vectoriser on the training rows only, then score. No leak here —
    the split is the only thing under test."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    vec = TfidfVectorizer(min_df=1, ngram_range=(1, 2), max_features=60_000)
    Ztr = vec.fit_transform([X[i] for i in tr])
    clf = LogisticRegression(max_iter=2000, C=4.0).fit(Ztr, y[tr])
    Zte = vec.transform([X[i] for i in te])
    return float((clf.predict(Zte) == y[te]).mean())


def duplicate_leak(d, n_unique=1_500, dup_frac=0.3, seeds=10) -> dict:
    """The text leak that is expensive: the same document on both sides.

    IMDb is deduplicated, so a corpus that is not has to be constructed — and
    the slide says so. A third of the entries are submitted twice, which is
    ordinary for a feedback desk with a resubmit button and no de-duplication.
    Everything else is done correctly: the vectoriser is fitted inside each
    split. The only difference between the two rows is whether the split keeps
    both copies of an entry on the same side.
    """
    from sklearn.model_selection import GroupShuffleSplit, train_test_split

    all_x = list(d["train_x"]) + list(d["test_x"])
    all_y = np.concatenate([d["train_y"], d["test_y"]])

    naive, grouped, twins = [], [], []
    for s in range(seeds):
        rng = np.random.default_rng(SEED + 100 + s)
        idx = rng.choice(len(all_x), size=n_unique, replace=False)
        X = [all_x[i] for i in idx]
        y = all_y[idx]
        g = np.arange(n_unique)

        dup = rng.choice(n_unique, size=int(dup_frac * n_unique), replace=False)
        X = X + [X[i] for i in dup]
        y = np.concatenate([y, y[dup]])
        g = np.concatenate([g, g[dup]])

        # (a) the split a careless pipeline makes: random over rows
        tr, te = train_test_split(np.arange(len(X)), test_size=0.25,
                                  random_state=SEED + s, stratify=y)
        naive.append(_tfidf_score(X, y, tr, te))
        twins.append(float(np.isin(g[te], g[tr]).mean()))

        # (b) the split that keeps every copy of an entry on one side
        gss = GroupShuffleSplit(n_splits=1, test_size=0.25,
                                random_state=SEED + s)
        tr2, te2 = next(gss.split(np.arange(len(X)), y, groups=g))
        grouped.append(_tfidf_score(X, y, tr2, te2))

    naive, grouped = np.array(naive), np.array(grouped)
    gap = naive - grouped
    print(f"      duplicates: random split {naive.mean():.4f}  "
          f"grouped split {grouped.mean():.4f}  "
          f"gap {100*gap.mean():+.2f} points (sd {100*gap.std():.2f})")
    return {"n_unique": n_unique, "dup_frac": dup_frac, "seeds": seeds,
            "n_rows": n_unique + int(dup_frac * n_unique),
            "naive_mean": float(naive.mean()), "naive_sd": float(naive.std()),
            "grouped_mean": float(grouped.mean()),
            "grouped_sd": float(grouped.std()),
            "gap_mean": float(gap.mean()), "gap_sd": float(gap.std()),
            "gap_points": float(100 * gap.mean()),
            "seeds_leak_wins": int((gap > 0).sum()),
            "twin_share": float(np.mean(twins)),
            "gaps": [float(v) for v in gap]}


def duplicate_check(d) -> dict:
    """Is any test review also a training review? On text this is the leak
    that costs the most and is checked the least."""
    norm = lambda s: " ".join(s.lower().split())
    tr = set(norm(s) for s in d["train_x"])
    dup = sum(1 for s in d["test_x"] if norm(s) in tr)
    within_tr = len(d["train_x"]) - len(tr)
    return {"train_test_dupes": dup, "within_train_dupes": within_tr,
            "n_train": len(d["train_x"]), "n_test": len(d["test_x"])}


def fig_leak(lk):
    fig, ax = plt.subplots(figsize=(10.4, 3.1))
    rng = np.random.default_rng(SEED)
    for i, (label, name) in enumerate((("small", "400 reviews"),
                                       ("full", "25,000 reviews"))):
        g = 100 * np.array(lk[label]["gaps"])
        ax.scatter(g, i + rng.normal(0, 0.055, len(g)), s=52,
                   color=ACCENT if label == "small" else PRIMARY,
                   alpha=0.75, zorder=3)
        ax.plot([g.mean()], [i], "|", ms=34, mew=3.4, color="#16212b", zorder=4)
    ax.axvline(0, color=AXIS, lw=1.4)
    ax.set_yticks([0, 1]); ax.set_yticklabels(["400 reviews", "25,000 reviews"])
    ax.set_ylim(-0.5, 1.5)
    ax.set_xlabel("accuracy the leak buys, in points")
    ax.set_title("one dot per seed; the bar is the mean")
    ax.grid(axis="y", visible=False)
    ax.annotate(f"{lk['small']['gap_points']:+.2f} points on average",
                xy=(lk["small"]["gap_points"], 0.28), fontsize=SMALL,
                color=ACCENT, ha="center")
    ax.annotate(f"{lk['full']['gap_points']:+.2f} points",
                xy=(lk["full"]["gap_points"], 1.28), fontsize=SMALL,
                color=PRIMARY, ha="center")
    fig.tight_layout()
    save(fig, "l22-leak")


def fig_duplicate_leak(dl):
    fig, ax = plt.subplots(figsize=(10.4, 3.0))
    x = np.arange(2)
    vals = [100 * dl["naive_mean"], 100 * dl["grouped_mean"]]
    errs = [100 * dl["naive_sd"], 100 * dl["grouped_sd"]]
    ax.bar(x, vals, yerr=errs, capsize=7, width=0.5,
           color=[ACCENT, SUCCESS], error_kw=dict(ecolor="#16212b", lw=1.6))
    for xi, v in zip(x, vals):
        ax.text(xi, v + 1.6, f"{v:.1f}%", ha="center", fontsize=SMALL,
                color="#16212b")
    ax.set_xticks(x)
    ax.set_xticklabels(["random split\n(copies land on both sides)",
                        "grouped split\n(copies kept together)"],
                       fontsize=SMALL)
    ax.set_ylabel("reported accuracy, %")
    ax.set_ylim(0, 108)
    ax.set_title(f"{dl['seeds']} seeds; bars are the mean, whiskers one "
                 f"standard deviation")
    fig.tight_layout()
    save(fig, "l22-duplicates")


# ------------------------------------------------------------ result figures

def fig_baselines(bl, scratch):
    names = ["always positive", "tf-idf unigram\n+ logistic regression",
             "tf-idf 1-2 gram\n+ logistic regression",
             "GRU from scratch\n(trainable embeddings)"]
    vals = [bl["majority"], bl["unigram"]["acc"], bl["bigram"]["acc"],
            scratch["test_acc"]]
    colors = [RULE, MUTED, PRIMARY,
              SUCCESS if vals[3] > vals[2] else ACCENT]
    fig, ax = plt.subplots(figsize=(10.4, 3.2))
    x = np.arange(4)
    ax.bar(x, 100 * np.array(vals), color=colors, width=0.62)
    for xi, v in zip(x, vals):
        ax.text(xi, 100 * v + 1.0, f"{100*v:.1f}%", ha="center",
                fontsize=SMALL, color="#16212b")
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=SMALL)
    ax.set_ylim(0, 105)
    ax.set_ylabel("test accuracy, %")
    ax.set_title("25,000 held-out reviews")
    fig.tight_layout()
    save(fig, "l21-baselines")


def fig_swap(runs, bl):
    order = ["word_random", "wp_random", "wp_frozen", "wp_tuned"]
    names = ["our words\nrandom", "subword\nrandom",
             "subword\npretrained, frozen", "subword\npretrained, tuned"]
    vals = [runs[k]["test_acc"] for k in order]
    base = vals[0]
    # A single-seed difference under a third of a point is not a ranking, so it
    # is not painted as one — see the null result on the swap-1 slide.
    tie = 0.003
    colors = [PRIMARY] + [SUCCESS if v > base + tie else
                          ACCENT if v < base - tie else PRIMARY
                          for v in vals[1:]]
    fig, ax = plt.subplots(figsize=(10.4, 3.2))
    x = np.arange(4)
    ax.bar(x, 100 * np.array(vals), color=colors, width=0.6)
    ax.axhline(100 * base, color=PRIMARY, ls="--", lw=1.6)
    ax.axhline(100 * bl["bigram"]["acc"], color=MUTED, ls=":", lw=2.0)
    ax.text(3.42, 100 * bl["bigram"]["acc"] + 0.6, "bag of words",
            fontsize=SMALL, color=MUTED, ha="right")
    # inside the bars: the dashed baseline sits within a point of three of the
    # four tops, so a label above the bar lands on the rule
    for xi, v in zip(x, vals):
        ax.text(xi, 100 * v - 2.4, f"{100*v:.1f}%", ha="center", fontsize=SMALL,
                color="white", fontweight="semibold")
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=SMALL)
    # the bag-of-words rule has to stay inside the axes even when it is above
    # every bar, which on this corpus it may well be
    ax.set_ylim(min(70, 100 * min(vals) - 5),
                max(100 * max(vals), 100 * bl["bigram"]["acc"]) + 4)
    ax.set_ylabel("test accuracy, %")
    ax.set_title("identical GRU, identical seed, identical four epochs")
    fig.tight_layout()
    save(fig, "l21-swap")


def fig_curves(runs):
    fig, ax = plt.subplots(figsize=(10.4, 3.1))
    style = {
        "word_random": ("our words, random", PRIMARY, "o-"),
        "wp_random": ("subword, random", ACCENT, "s--"),
        "wp_frozen": ("subword, pretrained frozen", MATH, "^-"),
        "wp_tuned": ("subword, pretrained tuned", SUCCESS, "d-"),
    }
    for k, (label, c, m) in style.items():
        y = 100 * np.array(runs[k]["val_curve"])
        ax.plot(np.arange(1, len(y) + 1), y, m, color=c, lw=2.3, ms=7,
                label=label)
    ax.set_xticks(np.arange(1, EPOCHS + 1))
    ax.set_xlabel("epoch"); ax.set_ylabel("validation accuracy, %")
    ax.legend(loc="lower right", fontsize=SMALL)
    ax.set_title(f"{N_VAL:,} held-out reviews from the training split")
    fig.tight_layout()
    save(fig, "l21-curves")


def fig_final(bl, scratch, swap, ft, zs):
    names = ["always\npositive", "bag of\nwords", "GRU from\nscratch",
             "GRU, pretrained\nembeddings", "DistilBERT,\nhead only",
             "DistilBERT,\nfine-tuned"]
    vals = [bl["majority"], bl["bigram"]["acc"], scratch["test_acc"],
            swap["test_acc"], zs["acc"], ft["test_acc"]]
    colors = [RULE, MUTED, PRIMARY, PRIMARY, ACCENT, SUCCESS]
    fig, ax = plt.subplots(figsize=(10.4, 3.2))
    x = np.arange(len(vals))
    ax.bar(x, 100 * np.array(vals), color=colors, width=0.62)
    for xi, v in zip(x, vals):
        ax.text(xi, 100 * v + 1.2, f"{100*v:.1f}%", ha="center", fontsize=SMALL,
                color="#16212b")
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=SMALL)
    ax.set_ylim(0, 108)
    ax.set_ylabel("test accuracy, %")
    ax.set_title("25,000 held-out reviews, every row measured the same way")
    fig.tight_layout()
    save(fig, "l22-final")


# ----------------------------------------------------------------- diagrams

def _snippet(text: str, n: int = 150) -> str:
    """A quotable fragment: no HTML, no line breaks, no dollar amounts.

    A retrieved review goes straight onto a slide, so it has to survive both
    KaTeX (TRICKS 9.1) and the HTML parser without a manual clean-up step that
    someone will forget.
    """
    t = " ".join(text.replace("<br />", " ").split())
    t = t.replace("&", "and").replace("<", "(").replace(">", ")").replace("$", "")
    if len(t) > n:
        t = t[:n].rsplit(" ", 1)[0] + "..."
    return t


def validate_diagrams() -> list[str]:
    bad = []
    for name in ("d-tokenise.svg", "d-cetrap.svg"):
        p = OUT / name
        if not p.is_file():
            bad.append(f"{name}: missing")
            continue
        try:
            xml.dom.minidom.parse(str(p))
        except Exception as exc:                              # noqa: BLE001
            bad.append(f"{name}: {exc}")
    return bad


# --------------------------------------------------------------------- main

def main() -> int:
    setup()
    load_cache()
    facts: dict = {}

    print("Loading IMDb:")
    d = load_imdb()
    print(f"  {len(d['train_x']):,} train, {len(d['test_x']):,} test, "
          f"{N_FIT:,} fit / {N_VAL:,} validation")

    print("Lecture 21 — the corpus:")
    cf = cached("app11_corpus", lambda: corpus_facts(d))
    fig_lengths(cf)
    facts["l21_corpus"] = {k: v for k, v in cf.items() if k != "lens"}
    facts["l21_n_train"] = cf["n_train"]
    facts["l21_n_test"] = cf["n_test"]
    facts["l21_n_fit"] = N_FIT
    facts["l21_n_val"] = N_VAL
    facts["l21_maxlen"] = MAXLEN
    facts["l21_word_vocab"] = WORD_VOCAB
    facts["l21_under_maxlen"] = 1 - cf["over_maxlen"]

    print("Lecture 21 — tokenisation:")
    oc = cached("app11_oov_v2", lambda: oov_curve(d))
    fig_oov(oc)
    facts["l21_oov"] = oc
    te = cached("app11_tokexample_v2", lambda: tokenisation_example(d))
    facts["l21_tok_example"] = te
    print(f"      words   : {' | '.join(te['word_view'])}")
    print(f"      pieces  : {' | '.join(te['pieces'])}")
    print(f"      counts  : {te['counts']}")

    print("Lecture 21 — the baselines:")
    bl = cached("app11_bow", lambda: bow_baselines(d))
    facts["l21_bow"] = bl
    facts["l21_majority"] = bl["majority"]

    print("Lecture 21 — the recurrent classifier:")
    enc_w, v_w = word_encoder(d)
    runs = {}
    runs["word_random"] = cached(
        "app11_rnn_word_random_es",
        lambda: train_rnn(d, enc_w, v_w, tag="word/random"))

    enc_s, v_s = subword_encoder()
    runs["wp_random"] = cached(
        "app11_rnn_wp_random_es",
        lambda: train_rnn(d, enc_s, v_s, tag="wp/random"))

    E, evr = cached("app11_pretrained_emb", lambda: pretrained_embeddings(v_s))
    facts["l21_pca_explained"] = evr
    print(f"      PCA 768 -> {EMB_DIM}: {evr:.3f} of the variance")

    runs["wp_frozen"] = cached(
        "app11_rnn_wp_frozen_es",
        lambda: train_rnn(d, enc_s, v_s, init=E, freeze=True, tag="wp/frozen"))
    runs["wp_tuned"] = cached(
        "app11_rnn_wp_tuned_es",
        lambda: train_rnn(d, enc_s, v_s, init=E, freeze=False, tag="wp/tuned"))

    fig_baselines(bl, runs["word_random"])
    fig_swap(runs, bl)
    fig_curves(runs)
    facts["l21_runs"] = runs
    facts["l21_scratch_acc"] = runs["word_random"]["test_acc"]
    facts["l21_pretrained_acc"] = runs["wp_tuned"]["test_acc"]
    facts["l21_swap_gain_points"] = 100 * (runs["wp_tuned"]["test_acc"]
                                           - runs["word_random"]["test_acc"])
    facts["l21_wp_random_delta"] = 100 * (runs["wp_random"]["test_acc"]
                                          - runs["word_random"]["test_acc"])
    facts["l21_earlystop_gain"] = 100 * (runs["wp_tuned"]["test_acc"]
                                         - runs["wp_tuned"]["test_acc_last_epoch"])
    facts["l21_frozen_delta"] = 100 * (runs["wp_frozen"]["test_acc"]
                                       - runs["word_random"]["test_acc"])
    facts["l21_bow_minus_rnn_points"] = 100 * (bl["bigram"]["acc"]
                                               - runs["word_random"]["test_acc"])

    print("Lecture 21 — the assistant's padding bug:")
    pad_run = cached(
        "app11_rnn_padding_bug_es",
        lambda: train_rnn(d, enc_w, v_w, last_of_padding=True, tag="pad-bug"))
    facts["l21_padding_bug"] = pad_run
    facts["l21_padding_cost_points"] = 100 * (runs["word_random"]["test_acc"]
                                              - pad_run["test_acc"])
    print(f"      last-of-padding costs "
          f"{facts['l21_padding_cost_points']:.2f} points")

    print("Lecture 22 — thread 11:")
    ss = softmax_shift()
    fig_softmax_shift(ss)
    facts["l22_softmax"] = ss
    gc = gradient_check()
    fig_gradient(gc)
    facts["l22_gradient"] = gc
    print(f"      |autograd - (p - y)| = {gc['max_abs_error']:.3e}")
    cs = cached("app11_ce_stability_v3", ce_stability)
    fig_ce_stability(cs)
    facts["l22_stability"] = cs
    kl = ce_vs_kl()
    facts["l22_kl"] = kl
    print(f"      CE - H(p) - KL = {kl['residual']:.3e}")

    print("Lecture 22 — the double-softmax bug:")
    ds = cached("app11_rnn_double_softmax_es",
                lambda: train_rnn(d, enc_w, v_w, double_softmax=True,
                                  tag="double-softmax"))
    facts["l22_double_softmax"] = ds
    facts["l22_double_softmax_cost"] = 100 * (runs["word_random"]["test_acc"]
                                              - ds["test_acc"])
    print(f"      softmax before the loss costs "
          f"{facts['l22_double_softmax_cost']:.2f} points")

    print("Lecture 22 — fine-tuning:")
    ft = cached("app11_finetune_v2", lambda: finetune_distilbert(d))
    facts["l22_finetune"] = ft
    zs = cached("app11_zeroshot_v2", lambda: zero_shot_head(d))
    facts["l22_zeroshot"] = zs

    print("Lecture 22 — where the pretrained model earns its place:")
    le = cached("app11_label_efficiency_v3", lambda: label_efficiency(d))
    fig_label_efficiency(le)
    facts["l22_label_efficiency"] = le
    facts["l22_finetune_acc"] = ft["test_acc"]
    facts["l22_gain_over_scratch"] = 100 * (ft["test_acc"]
                                            - runs["word_random"]["test_acc"])
    facts["l22_gain_over_bow"] = 100 * (ft["test_acc"] - bl["bigram"]["acc"])
    e_scratch = 1 - runs["word_random"]["test_acc"]
    e_ft = 1 - ft["test_acc"]
    facts["l22_error_drop"] = (e_scratch - e_ft) / e_scratch    # errors removed
    facts["l22_errors_scratch"] = round(e_scratch * cf["n_test"])
    facts["l22_errors_finetuned"] = round(e_ft * cf["n_test"])
    fig_final(bl, runs["word_random"], runs["wp_tuned"], ft, zs)

    print("Lecture 21/22 — what inference costs:")
    ic = cached("app11_inference_v2", lambda: inference_cost(d))
    # Both lectures put these in a class="num" cell, and check_provenance scopes
    # those to the lecture that owns them, so both namespaces carry the values.
    # One measurement, two keys -- not two measurements.
    facts["l21_inference"] = ic
    facts["l22_inference"] = ic

    print("Lecture 22 — search and grouping:")
    sr = cached("app11_search", lambda: semantic_search(d))
    facts["l22_search"] = {k: v for k, v in sr.items()
                           if k not in ("embeddings", "texts")}
    for h in sr["hits"]:
        print(f"      query: {h['query']}")
        print(f"        semantic top-1 ({h['top_sim'][0]:.3f}): "
              f"{h['top_text'][0][:120]}")
        print(f"        keyword  top-1 ({h['keyword_top_sim'][0]:.3f}): "
              f"{h['keyword_top_text'][0][:120]}")
    cl = cached("app11_clusters", lambda: cluster_complaints(sr))
    fig_clusters(cl)
    facts["l22_clusters"] = {k: v for k, v in cl.items() if k != "labels"}
    # display-ready strings, so the deck quotes the measurement rather than a
    # lecturer's memory of it
    facts["l22_cluster_terms"] = [", ".join(g["terms"][:4]) for g in cl["groups"]]
    facts["l22_best_silhouette"] = cl["silhouette"][cl["best_k"]]
    facts["l22_query"] = [h["query"] for h in sr["hits"]]
    facts["l22_hit_text"] = [_snippet(h["top_text"][0]) for h in sr["hits"]]
    facts["l22_keyword_text"] = [_snippet(h["keyword_top_text"][0])
                                 for h in sr["hits"]]

    print("Lecture 22 — the tokeniser leak:")
    lk = cached("app11_leak", lambda: tokeniser_leak(d))
    fig_leak(lk)
    facts["l22_leak"] = lk
    dup = cached("app11_dupes", lambda: duplicate_check(d))
    facts["l22_dupes"] = dup
    print(f"      exact duplicates across the official split: "
          f"{dup['train_test_dupes']}")

    print("Lecture 22 — the leak that is expensive:")
    dl = cached("app11_duplicate_leak", lambda: duplicate_leak(d))
    fig_duplicate_leak(dl)
    facts["l22_duplicate_leak"] = dl

    export(**facts)

    bad = validate_diagrams()
    if bad:
        print("\nmalformed or missing diagrams:")
        for b in bad:
            print("  " + b)
        return 1

    problems = check_text_floor()
    if problems:
        print("\ntext floor:")
        for p in problems:
            print("  " + p)
        return 1
    print("\nall figures clear the 15px text floor; all d-*.svg parse as XML")
    return 0


if __name__ == "__main__":
    sys.exit(main())
