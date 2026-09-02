"""
Lecture 21 — Reading the customers.

IMDb film reviews: subword tokenisation, trainable embeddings, a recurrent
classifier from scratch, and the swap that changes nothing but the embedding
table.

Exports build() -> list[nbformat cell]. Self-contained: it downloads and splits
the corpus itself rather than assuming another notebook's kernel is alive.

The deck's numbers come from 20,000 fit reviews and four epochs, which is about
ten minutes per run. This notebook uses 5,000 and two epochs so that a free
Colab runtime finishes inside the hour; every cell says which it is, and the
ordering of the results is the same.
"""

from __future__ import annotations

import nbformat as nbf
from _prompt import prompt                                # noqa: E402


def md(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(text.strip("\n"))


def code(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(text.strip("\n"))


HEADER = """
# Text

**Lecture 17** · Géron, Chapter 14

Applications of Machine Learning — BSc Mathematics of Artificial Intelligence

---

**How to use this notebook.** Read before you run. Every code cell is preceded
by the specification that would produce it — input, output, constraint, check.

Cells marked **⚠** deliberately run code that is wrong, and say so in the
heading before you reach them. They are the failures this lecture is about;
each runs the broken version beside the correct one and prices the difference.

Runs on CPU. Nothing here needs an accelerator.

**Scale.** The lecture's numbers come from 20,000 training reviews and four
epochs — about ten minutes per run on a GPU. Here we use **5,000 reviews and
two epochs** so the whole notebook finishes in a few minutes. The accuracies are
lower than the deck's; the *ordering* of the four configurations is the same,
and the ordering is the point.
"""


def build() -> list:
    return [
        md(HEADER),

        md("## 1 · Setup"),
        prompt(
            label="setup",
            input="nothing",
            output="versions, seeds, device",
            constraint="say what to do if there is no accelerator — the GRU cells here are the slowest in the course on CPU",
            check="when a notebook is deliberately smaller than the deck, say by how much and say what is preserved. 'The ordering is the point' is a claim you can check."),
        code('''
# Not examinable, and only needed on some machines: PyTorch, numpy and
# torchvision can each end up loading their own OpenMP runtime, and with more
# than one loaded a training cell can deadlock -- no error, no output, and no
# CPU use. These have to be set BEFORE torch is imported, because they are read
# at import time and after that they do nothing.
import os
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# --- setup -------------------------------------------------------------------
# Not examinable: engineering hygiene. It is here because a version mismatch
# produces a confusing error twenty cells later.
import sys, re, time, tarfile, urllib.request
from pathlib import Path
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import sklearn

print(f"python        {sys.version.split()[0]}")
print(f"torch         {torch.__version__}")
print(f"scikit-learn  {sklearn.__version__}")

RANDOM_STATE = 42                 # every split, every model, every shuffle
torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
print(f"\\ndevice        {device}")
if device == "cpu":
    print("No accelerator. Everything below still runs; the GRU cells are slow.")
    print("In Colab: Runtime -> Change runtime type -> T4 GPU.")
'''),

        md("""
## 2 · The corpus

A function, not a manual download — the same rule as Lecture 1. About 80 MB;
**⏱ 30–90 seconds** the first time, instant afterwards.

The split into `train` and `test` ships with the corpus. We do not make our own.
"""),
        prompt(
            label="⏱ 30-90 s first time — the corpus",
            input="the IMDb tarball",
            output="25,000 training and 25,000 test reviews with their labels",
            constraint="use the split that SHIPS with the corpus — we do not make our own, because every published number on this dataset uses that cut",
            check="assert both halves are 25,000 and that the labels are 0/1. Print the positive share of BOTH halves. Balanced-by-construction is a claim about the file, and it costs one line to verify."),
        code('''
URL  = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
ROOT = Path("datasets")
DATA = ROOT / "aclImdb"

def load_imdb():
    if not DATA.is_dir():
        ROOT.mkdir(parents=True, exist_ok=True)
        tarball = ROOT / "aclImdb_v1.tar.gz"
        if not tarball.is_file():
            urllib.request.urlretrieve(URL, tarball)
        with tarfile.open(tarball) as t:
            t.extractall(path=ROOT, filter="data")

    def read(split):
        texts, labels = [], []
        for lab, name in ((1, "pos"), (0, "neg")):
            for p in sorted((DATA / split / name).iterdir()):
                texts.append(p.read_text(encoding="utf-8"))
                labels.append(lab)
        return texts, np.array(labels, dtype=np.int64)

    return read("train"), read("test")

(train_x, train_y), (test_x, test_y) = load_imdb()

assert len(train_x) == 25_000 and len(test_x) == 25_000, "unexpected corpus size"
assert set(train_y) == {0, 1}
print(f"{len(train_x):,} train, {len(test_x):,} test")
print(f"positive share: train {train_y.mean():.3f}, test {test_y.mean():.3f}")
'''),

        md("""
Balanced by construction, in both halves. That is what makes accuracy a
defensible metric here — the condition Lecture 4 spent ninety minutes on.

Now carve a validation set out of the **training** half. The test half is not
touched again until the very last cell.
"""),
        prompt(
            label="carve the validation set out of TRAINING",
            input="the 25,000 training reviews",
            output="5,000 fit and 2,000 validation reviews",
            constraint="both from the TRAINING half — the test half is not touched again until the very last cell",
            check="assert the two index sets are disjoint and the sizes are right. Whenever you slice a corpus, ask whether it is sorted by label. This one is, and so are most of the classic text datasets."),
        code('''
N_FIT, N_VAL = 5_000, 2_000        # the deck uses 20,000 and 5,000

rng   = np.random.default_rng(RANDOM_STATE)
order = rng.permutation(len(train_x))
fit_i, val_i = order[:N_FIT], order[N_FIT:N_FIT + N_VAL]

fit_x = [train_x[i] for i in fit_i]; fit_y = train_y[fit_i]
val_x = [train_x[i] for i in val_i]; val_y = train_y[val_i]

assert set(fit_i).isdisjoint(val_i), "the split overlaps"
assert len(fit_x) == N_FIT and len(val_x) == N_VAL
print(f"fit {len(fit_x):,}   val {len(val_x):,}   test {len(test_x):,} (untouched)")
'''),

        md("""
## 3 · Look at it — at the training half only

Two things to notice: how long a review is, and how many distinct words there
are. Both decide something about the model.
"""),
        prompt(
            label="look at it — lengths first",
            input="the training reviews",
            output="the length distribution, and what a cut at MAXLEN costs",
            constraint="report BOTH what fraction of REVIEWS get truncated and what fraction of WORDS survive — they are very different numbers",
            check="a truncation length is a hyperparameter. State what it discards, in both units, before adopting it."),
        code('''
WORD_RE = re.compile(r"[a-z0-9']+")

def word_tokens(s):
    """The whole tokenizer. Three decisions are already made in it."""
    return WORD_RE.findall(s.lower().replace("<br />", " "))

train_tok = [word_tokens(s) for s in train_x]
lens = np.array([len(t) for t in train_tok])

print(f"length: mean {lens.mean():.0f}   median {np.median(lens):.0f}   "
      f"90th pct {np.percentile(lens, 90):.0f}   max {lens.max()}")

MAXLEN = 192                       # the deck uses 256
over = (lens > MAXLEN).mean()
kept = np.minimum(lens, MAXLEN).sum() / lens.sum()
print(f"cutting at {MAXLEN}: truncates {over:.1%} of reviews, "
      f"keeps {kept:.1%} of all words")
'''),
        prompt(
            label="and the vocabulary",
            input="every token in the training half",
            output="how many distinct words, how many appear exactly once, and the length histogram",
            constraint="count the HAPAX words — the ones seen exactly once — as a share of the vocabulary",
            check="clip the histogram before plotting. One review of 2,470 words stretches the axis so that the bulk of the distribution is three pixels wide."),
        code('''
counts = Counter(w for t in train_tok for w in t)
hapax  = sum(1 for _, c in counts.items() if c == 1)
print(f"{len(counts):,} distinct words")
print(f"{hapax:,} of them appear exactly once  ({hapax / len(counts):.1%})")

plt.figure(figsize=(9, 3))
plt.hist(np.clip(lens, 0, 1200), bins=80)
plt.axvline(MAXLEN, color="C3", ls="--")
plt.xlabel("review length, in words"); plt.ylabel("reviews")
plt.tight_layout(); plt.show()
'''),
        md("""
Two words in five are seen once and never again. You cannot learn a
128-dimensional vector for a word from one example — remember that when the
recurrent model underperforms.
"""),

        md("""
## 4 · The baselines, before anything is built

Rule 2 of this course: *a metric with nothing to compare it to is decoration.*

Two anchors. The trivial one, and the one that is actually hard to beat.
"""),
        prompt(
            label="the trivial anchor",
            input="the test labels",
            output="the majority-class accuracy",
            constraint="take the max of the two shares rather than assuming which class is commoner",
            check="a trivial baseline that is trivially beaten still belongs in the table. It is the zero of the scale."),
        code('''
majority = max(test_y.mean(), 1 - test_y.mean())
print(f"always predict one class:  {majority:.1%}")
'''),
        prompt(
            label="⏱ 15 s — the anchor that is actually hard to beat",
            input="all 25,000 training reviews, unigrams and bigrams",
            output="a tf-idf logistic regression and its test accuracy",
            constraint="`fit_transform` on train and `transform` on test — different verbs, and the vocabulary is a fitted object like any other",
            check="assert the two matrices have the same number of COLUMNS, which is what fails if `fit_transform` was called twice. Thirty seconds of counting words is the thing your recurrent network has to beat. Measure it before you build anything."),
        code('''
# ⏱ about 15 seconds: 25,000 documents, unigrams and bigrams.
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

t0 = time.perf_counter()
vec = TfidfVectorizer(min_df=2, ngram_range=(1, 2), max_features=200_000)
X_bow_train = vec.fit_transform(train_x)          # fit on TRAINING reviews only
X_bow_test  = vec.transform(test_x)               # transform, a different verb

bow = LogisticRegression(max_iter=2000, C=4.0).fit(X_bow_train, train_y)
bow_acc = (bow.predict(X_bow_test) == test_y).mean()

assert X_bow_train.shape[0] == 25_000
assert X_bow_train.shape[1] == X_bow_test.shape[1], "different feature spaces"
print(f"tf-idf + logistic regression: {bow_acc:.1%}  "
      f"({X_bow_train.shape[1]:,} features, {time.perf_counter() - t0:.0f}s)")
'''),

        md("""
## 5 · The derivation — softmax, and the shift that changes nothing

$$\\sigma(\\mathbf{z})_k = \\frac{e^{z_k}}{\\sum_j e^{z_j}}$$

Adding a constant to every logit multiplies numerator and denominator by the
same $e^{c}$, so

$$\\sigma(\\mathbf{z} + c\\mathbf{1}) = \\sigma(\\mathbf{z}).$$

The **function** ignores the shift. `float32` does not.
"""),
        prompt(
            label="softmax, and the shift that changes nothing",
            input="three logits near 1000",
            output="the naive softmax, and the same thing with the max subtracted",
            constraint="suppress the overflow warnings deliberately with `np.errstate` — the overflow is the demonstration, not an accident, and an unsuppressed warning reads as a bug in the notebook",
            check="print where exp overflows float32. It is about 88.7, which is a much smaller number than people expect."),
        code('''
z = np.array([1000., 1001., 1002.], dtype=np.float32)

with np.errstate(over="ignore", invalid="ignore"):
    print("exp(z)                 ", np.exp(z))
    print("exp(z) / exp(z).sum()  ", np.exp(z) / np.exp(z).sum())

m = z.max()
print("with the max subtracted", np.exp(z - m) / np.exp(z - m).sum())
print(f"\\nexp overflows float32 past x = {np.log(np.finfo(np.float32).max):.2f}")
'''),
        prompt(
            label="the invariance itself",
            input="three small logits, shifted by 50",
            output="both softmaxes",
            constraint="use values that do NOT overflow, so the invariance is demonstrated separately from the numerical failure",
            check="assert the two agree to 1e-6. When a property holds mathematically and fails numerically, demonstrate each on its own inputs."),
        code('''
# and the invariance itself, on values that do not overflow
a = np.array([1., 2., 3.], dtype=np.float32)
p1 = np.exp(a) / np.exp(a).sum()
p2 = np.exp(a + 50) / np.exp(a + 50).sum()
print(p1, p2, sep="\\n")
assert np.abs(p1 - p2).max() < 1e-6, "softmax is not shift invariant here"
print(f"\\nlargest difference: {np.abs(p1 - p2).max():.2e}")
'''),

        md("""
## 6 · Cross-entropy, and its gradient with respect to the logits

For a one-hot target every term of $-\\sum_k y_k \\log p_k$ dies but one:

$$L = -\\log p_c = -\\log \\frac{e^{z_c}}{\\sum_j e^{z_j}}
    = -z_c + \\log\\sum_j e^{z_j}$$

**The exponential of the true class has cancelled.** Differentiating: the first
term contributes $-y_k$, and the derivative of the log-sum-exp is softmax, so

$$\\frac{\\partial L}{\\partial \\mathbf{z}} = \\mathbf{p} - \\mathbf{y}.$$

Three lines, no chain rule through the softmax. Verify it rather than believing
it.
"""),
        prompt(
            label="the gradient, derived and verified",
            input="random logits and targets",
            output="autograd's gradient beside the analytic p − y",
            constraint="`reduction='sum'` — the mean would divide every gradient by the batch size, and the assertion would then fail for a reason that has nothing to do with the mathematics",
            check="assert agreement to 1e-10, that each row of the gradient sums to zero, and that every component is in [−1, 1]. The row-sum assert is the interesting one. The gradient has no component along the all-ones direction, which is the derivative form of the shift invariance above."),
        code('''
torch.manual_seed(RANDOM_STATE)
z = torch.randn(7, 5, dtype=torch.float64, requires_grad=True)
y = torch.randint(0, 5, (7,))

# reduction="sum": the mean would divide every gradient by the batch size, and
# then the assertion fails for a reason that has nothing to do with the maths.
loss = nn.CrossEntropyLoss(reduction="sum")(z, y)
loss.backward()

p        = torch.softmax(z.detach(), dim=1)
onehot   = torch.zeros_like(p).scatter_(1, y[:, None], 1.0)
analytic = p - onehot

err = (z.grad - analytic).abs().max().item()
print(f"|autograd - (p - y)| = {err:.3e}")
assert err < 1e-10, "the derivation and the library disagree"

print(f"each row of the gradient sums to "
      f"{analytic.sum(1).abs().max().item():.2e} — no component along 1")
assert analytic.abs().max() <= 1.0, "p - y must lie in [-1, 1]"
'''),

        md("""
## 7 · Why the loss consumes logits

Two ways to compute the same number in `float32`, scored against `float64`.
"""),
        md("""
Sweep **how wrong the row is**, not the scale of the logits: the naive form has
to represent $e^{-\\text{loss}}$ as a `float32`, so the loss is the quantity the
failure depends on. Sweeping the standard deviation instead buries the effect,
because most rows then have a loss near zero where both forms agree trivially.
"""),
        prompt(
            label="why the loss consumes logits",
            input="2,000 rows at each of ten loss levels",
            output="the non-finite rate and median relative error of the naive form against the combined one",
            constraint="sweep HOW WRONG THE ROW IS, not the scale of the logits — the naive form has to represent e^(−loss) as a float32, so the loss is the quantity the failure depends on",
            check="assert the stable form never fails, at either end of the sweep. Score against float64 rather than against each other. Two float32 computations agreeing tells you nothing about either."),
        code('''
K, N = 10, 2_000
idx = np.arange(N)
rng32 = np.random.default_rng(RANDOM_STATE)
lse = lambda v: v.max(1) + np.log(np.exp(v - v.max(1, keepdims=True)).sum(1))

rows = []
for gap in (1, 5, 10, 20, 40, 60, 80, 90, 100, 110):
    z64 = rng32.normal(0, 1.0, size=(N, K))
    yy  = rng32.integers(0, K, size=N)
    z64[idx, yy] = z64.max(1) - gap        # true class `gap` below the largest
    z32 = z64.astype(np.float32)
    ref = -(z64[idx, yy] - lse(z64))       # >= gap, so relative error is safe

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        e     = np.exp(z32)
        p32   = e / e.sum(1, keepdims=True)
        naive = -np.log(p32[idx, yy])
    stable = -(z32[idx, yy] - lse(z32))

    def score(a):
        a = a.astype(np.float64)
        ok = np.isfinite(a)
        return (1 - ok.mean(),
                np.median(np.abs(a[ok] - ref[ok]) / ref[ok]) if ok.any() else np.nan)

    nb, ne = score(naive)
    sb, se = score(stable)
    rows.append((gap, nb, ne, sb, se))
    print(f"loss {gap:>3}   naive: {nb:6.1%} non-finite, median err {ne:.2e}   "
          f"stable: {sb:6.1%}, {se:.2e}")

assert rows[0][3] == 0.0 and rows[-1][3] == 0.0, "the stable form should never fail"
'''),
        prompt(
            label="the two failure modes, drawn",
            input="the sweep",
            output="median relative error on a log axis, and the non-finite rate as a percentage",
            constraint="two panels, because the two failures are different kinds — silent inaccuracy and outright inf/nan — and one axis cannot show both",
            check="clamp the log-axis values away from zero before plotting. A median error of exactly 0.0 is not plottable on a log scale and matplotlib's response is to drop the point silently."),
        code('''
g = np.array([r[0] for r in rows], dtype=float)
fig, ax = plt.subplots(1, 2, figsize=(10, 3))
ax[0].semilogy(g, np.maximum([r[2] for r in rows], 1e-9), "o-", label="naive")
ax[0].semilogy(g, np.maximum([r[4] for r in rows], 1e-9), "s-", label="combined")
ax[0].set_xlabel("true loss of the row, in nats")
ax[0].set_ylabel("median relative error"); ax[0].legend()
ax[1].plot(g, [100 * r[1] for r in rows], "o-", label="naive")
ax[1].plot(g, [100 * r[3] for r in rows], "s-", label="combined")
ax[1].set_xlabel("true loss of the row, in nats")
ax[1].set_ylabel("rows returning inf or nan, %"); ax[1].legend()
plt.tight_layout(); plt.show()
'''),
        md("""
### Two rows, small enough to check by hand

The first failure is loud. The second is the one that ships.
"""),
        prompt(
            label="two rows small enough to check by hand",
            input="[100, 0, −100] and [0, 0, −100], target class 2",
            output="the naive, combined, float64 and PyTorch values for each",
            constraint="show the LOUD failure and the QUIET one, in that order — the first overflows and the second returns a finite, plausible, wrong number",
            check="assert the first is non-finite and the second is off by more than 1e-3, so both failures are pinned. Print PyTorch's answer beside your own. It agrees with the combined form, which is evidence that the combined form is what the library does."),
        code('''
lse1 = lambda v: v.max() + np.log(np.exp(v - v.max()).sum())

def both_ways(logits, tgt=2):
    z = np.array(logits, dtype=np.float32)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        e = np.exp(z)
        p = e / e.sum()
        naive = -np.log(p[tgt])
    stable = -(z[tgt] - lse1(z))
    z64    = z.astype(np.float64)
    exact  = -(z64[tgt] - lse1(z64))
    torch_ = float(nn.CrossEntropyLoss()(torch.tensor(z)[None, :],
                                         torch.tensor([tgt])))
    print(f"z = {list(logits)}")
    print(f"  p(true class) {p[tgt]:.4e}   denominator {e.sum():.4e}")
    print(f"  naive    {naive}")
    print(f"  combined {stable}")
    print(f"  float64  {exact}")
    print(f"  PyTorch  {torch_}")
    return float(naive), float(stable), float(exact), torch_

print("--- the loud failure ---")
n1, s1, e1, t1 = both_ways([100., 0., -100.])
print("\\n--- the quiet failure ---")
n2, s2, e2, t2 = both_ways([0., 0., -100.])
print(f"\\nnaive is off by {abs(n2 - e2):.4f} — finite, plausible, and wrong")

assert not np.isfinite(n1), "expected the naive form to overflow here"
assert abs(s1 - e1) < 1e-3 and abs(t1 - e1) < 1e-3
assert abs(n2 - e2) > 1e-3, "expected the naive form to lose precision here"
assert abs(s2 - e2) < 1e-4, "the combined form should be exact here"
'''),

        md("""
## 8 · Cross-entropy and KL divergence

$$H(\\mathbf{y}, \\mathbf{p}) = H(\\mathbf{y})
  + D_{\\mathrm{KL}}(\\mathbf{y} \\Vert \\mathbf{p})$$

Add and subtract $\\sum_k y_k \\log y_k$; that is the whole proof. For a
**one-hot** target $H(\\mathbf{y}) = 0$, so minimising cross-entropy *is*
minimising the KL divergence to the label.
"""),
        prompt(
            label="cross-entropy and KL divergence",
            input="a random distribution, against a one-hot and a label-smoothed target",
            output="entropy, KL, their sum, and the cross-entropy",
            constraint="test with BOTH targets — for a one-hot target the entropy is zero and the identity is invisible",
            check="assert H + KL equals CE to 1e-12 for both, and that the one-hot entropy is exactly zero. Guard the logs against zeros with a mask on p > 0. 0·log 0 is 0 by convention and nan in floating point."),
        code('''
rngk = np.random.default_rng(RANDOM_STATE)
q      = rngk.dirichlet(np.ones(5))
onehot = np.eye(5)[2]
smooth = np.full(5, 0.1 / 4); smooth[2] = 0.9

H  = lambda p: float(-(p[p > 0] * np.log(p[p > 0])).sum())
CE = lambda p, q: float(-(p * np.log(q)).sum())
KL = lambda p, q: float((p[p > 0] * np.log(p[p > 0] / q[p > 0])).sum())

for name, p in (("one-hot", onehot), ("label-smoothed", smooth)):
    print(f"{name:15s} H {H(p):.4f}   KL {KL(p, q):.4f}   "
          f"H+KL {H(p) + KL(p, q):.4f}   CE {CE(p, q):.4f}")
    assert abs(CE(p, q) - H(p) - KL(p, q)) < 1e-12
assert H(onehot) == 0.0, "a one-hot distribution has zero entropy"
'''),

        md("""
## 9 · Tokenisation, and why a word vocabulary breaks

Build the vocabulary from the **fit split only**. Building a vocabulary is
*fitting*, and it obeys the same rule as every other fitted object in this
course.
"""),
        prompt(
            label="the vocabulary, fitted on the FIT split only",
            input="the 5,000 fit reviews",
            output="a word-to-index map, and the out-of-vocabulary rate on test",
            constraint="build it from the fit split ONLY — building a vocabulary is FITTING, and it obeys the same rule as every other fitted object in this course",
            check="assert indices 0 and 1 are reserved, for padding and [UNK]. The problem is not the SIZE of the vocabulary. A word vocabulary is closed and language is not, so no size fixes it."),
        code('''
VOCAB = 20_000
fit_counts = Counter(w for s in fit_x for w in word_tokens(s))
w2i = {w: i + 2 for i, (w, _) in enumerate(fit_counts.most_common(VOCAB - 2))}
# 0 = padding, 1 = [UNK]

assert 0 not in w2i.values() and 1 not in w2i.values()
assert len(w2i) <= VOCAB - 2
print(f"{len(w2i):,} words in the vocabulary")

test_flat = [w for s in test_x for w in word_tokens(s)]
oov = sum(1 for w in test_flat if w not in w2i)
distinct_test = set(test_flat)
oov_types = sum(1 for w in distinct_test if w not in w2i)
print(f"unseen test tokens:         {oov / len(test_flat):.1%}")
print(f"unseen DISTINCT test words: {oov_types / len(distinct_test):.1%}")
'''),
        md("""
The token rate looks survivable. The rate over *distinct* words never does — and
those are the informative ones. The problem is not the size of the vocabulary:
a word vocabulary is **closed** and language is not.

Now the same sentence under a subword tokenizer, which was trained once on some
other corpus and shipped.
"""),
        prompt(
            label="the same sentence, under subwords",
            input="one sentence with two rare words in it",
            output="the word tokenizer's view beside WordPiece's",
            constraint="show the word tokenizer producing [UNK] for exactly the words that carry the sentiment",
            check="print the pieces-per-word ratio and both vocabulary sizes. A subword tokenizer buys coverage and spends sequence length, and the next sections are about that trade."),
        code('''
from transformers import AutoTokenizer

BERT = "distilbert-base-uncased"
tk = AutoTokenizer.from_pretrained(BERT)

sentence = "The plot was unwatchable and utterly discombobulating."
words  = word_tokens(sentence)
pieces = tk.tokenize(sentence)

print("word tokenizer :", [w if w in w2i else "[UNK]" for w in words])
print("WordPiece      :", pieces)
print(f"\\n{len(words)} words -> {len(pieces)} pieces "
      f"({len(pieces) / len(words):.2f} per word)")
print(f"vocabulary: ours {VOCAB:,}   WordPiece {tk.vocab_size:,}")
'''),
        prompt(
            label="how often does WordPiece give up",
            input="2,000 randomly chosen test reviews",
            output="the [UNK] rate over subword tokens",
            constraint="sample RANDOMLY — the corpus ships positives first, so `test_x[:2000]` is all positives",
            check="assert the rate is below 0.1%, which is what 'almost never misses' should mean. Measure rather than assume. 'Subword tokenizers have no OOV problem' is nearly true and the nearly is worth four lines."),
        code('''
# How often does WordPiece have to give up? Measure rather than assume.
# NB: the corpus ships positives first, so test_x[:2000] is all positives. It
# does not matter for a rate that ignores the label, but take the habit anyway.
sample_i = np.random.default_rng(RANDOM_STATE + 7).permutation(len(test_x))[:2_000]
sample = [test_x[i] for i in sample_i]
n_unk = n_tok = 0
for s in sample:
    ids = tk(s, truncation=True, max_length=512)["input_ids"]
    n_unk += sum(1 for i in ids if i == tk.unk_token_id)
    n_tok += len(ids)
print(f"[UNK] rate over {n_tok:,} subword tokens: {n_unk / n_tok:.4%}")
assert n_unk / n_tok < 0.001, "a subword tokenizer should almost never miss"
'''),

        md("""
## 10 · From integers to vectors

Token 4,271 is not four thousand of anything — the integer is a name. An
embedding is a learned table with one row per token, and looking up row *i* is
exactly multiplying a one-hot vector by that table, without ever forming it.

`padding_idx=0` pins row 0 at zero and keeps it there. Padding must not learn
anything.
"""),
        prompt(
            label="from integers to vectors",
            input="a batch of token ids",
            output="the embedded batch, and the table's parameter count",
            constraint="`padding_idx=0` — it pins row 0 at zero and KEEPS it there, so padding never learns anything",
            check="assert the output shape, and assert row 0 is still exactly zero. Assert the padding row is zero AFTER training too, not just at construction. That is the only way to know the flag did what it claims."),
        code('''
emb = nn.Embedding(num_embeddings=VOCAB, embedding_dim=128, padding_idx=0)
x   = torch.randint(0, VOCAB, (32, MAXLEN))

assert emb(x).shape == (32, MAXLEN, 128)
assert torch.equal(emb.weight[0], torch.zeros(128)), "padding row is not zero"
print(f"embedding table: {emb.weight.numel():,} parameters")
print(f"one batch: {tuple(x.shape)} -> {tuple(emb(x).shape)}")
'''),

        md("""
## 11 · Padding, and the length you must keep

A batch is a rectangle; reviews are not. Pad to `MAXLEN`, truncate what is
longer, and **keep the true lengths**. A padded batch has lost the information
about where each review ends.
"""),
        prompt(
            label="padding, and the length you must keep",
            input="variable-length token sequences",
            output="a rectangular batch and the true lengths",
            constraint="keep the LENGTHS. A padded batch has lost the information about where each review ends, and the model needs it back",
            check="assert every length is between 1 and MAXLEN, and that nothing is written past the true length of the first row. Assert that the region past the true length is zero. It is one line and it is the invariant the whole padding scheme depends on."),
        code('''
def pad_batch(seqs):
    X = np.zeros((len(seqs), MAXLEN), dtype=np.int64)
    L = np.zeros(len(seqs), dtype=np.int64)
    for i, s in enumerate(seqs):
        s = s[:MAXLEN]
        X[i, :len(s)] = s
        L[i] = max(len(s), 1)          # a length of 0 would crash the packing
    return torch.from_numpy(X), torch.from_numpy(L)

def encode_words(texts):
    return pad_batch([[w2i.get(w, 1) for w in word_tokens(s)] for s in texts])

Xf_w, Lf_w = encode_words(fit_x)
Xv_w, Lv_w = encode_words(val_x)

assert Xf_w.shape == (N_FIT, MAXLEN)
assert (Lf_w >= 1).all() and (Lf_w <= MAXLEN).all()
assert (Xf_w[0, Lf_w[0]:] == 0).all(), "something is written past the true length"
print(f"fit batch {tuple(Xf_w.shape)}   lengths {Lf_w.min()}–{Lf_w.max()}")
'''),

        md("""
## 12 · The classifier

Thirteen lines, and eleven of them are Lecture 12. The two new ones are the
embedding and the packing.
"""),
        prompt(
            label="the classifier",
            input="a padded batch and its lengths",
            output="two logits per review",
            constraint="`pack_padded_sequence` with `enforce_sorted=False`, and take the final hidden state of BOTH directions",
            check="assert the head returns exactly two logits for a batch of four. `lengths.cpu()` in the pack call. It requires a CPU tensor and the error message when it is on the GPU names neither the argument nor the fix."),
        code('''
EMB_DIM, HIDDEN = 128, 64

class GRUClassifier(nn.Module):
    def __init__(self, vocab, init=None, freeze=False, last_of_padding=False):
        super().__init__()
        self.emb = nn.Embedding(vocab, EMB_DIM, padding_idx=0)
        if init is not None:
            self.emb.weight.data.copy_(torch.from_numpy(init))
            self.emb.weight.requires_grad = not freeze
        self.rnn  = nn.GRU(EMB_DIM, HIDDEN, batch_first=True, bidirectional=True)
        self.head = nn.Linear(2 * HIDDEN, 2)
        self.last_of_padding = last_of_padding

    def forward(self, x, lengths):
        e = self.emb(x)
        if self.last_of_padding:                 # the assistant's version
            out, _ = self.rnn(e)
            return self.head(out[:, -1, :])
        packed = nn.utils.rnn.pack_padded_sequence(
            e, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, h = self.rnn(packed)
        return self.head(torch.cat([h[0], h[1]], dim=1))

_m = GRUClassifier(VOCAB)
assert _m(Xf_w[:4], Lf_w[:4]).shape == (4, 2), "the head must return two logits"
print(f"{sum(p.numel() for p in _m.parameters()):,} parameters")
'''),

        md("""
The training loop is Lecture 12's five lines, unchanged. Nothing about text
changes the loop.

⏱ **about 40–90 seconds per run** on a GPU, several minutes on a CPU.
"""),
        prompt(
            label="the loop, with early stopping",
            input="the training and validation splits",
            output="a trained model, keeping the weights from the BEST validation epoch",
            constraint="count accuracy over the SET, not as a mean of batch accuracies — the entry from application 6",
            check="clone the state dict when you save it. `net.state_dict()` returns references to live tensors, so without the clone your 'best' weights keep training."),
        code('''
BATCH, EPOCHS, LR = 64, 2, 1e-3     # the deck uses 4 epochs

@torch.no_grad()
def accuracy(net, X, L, y, batch=256):
    net.eval()
    hits = 0
    for i in range(0, len(X), batch):
        out = net(X[i:i + batch].to(device), L[i:i + batch])
        hits += int((out.argmax(1).cpu().numpy() == y[i:i + batch]).sum())
    return hits / len(y)            # counted over the SET, not a mean of batches

def train(net, Xf, Lf, yf, Xv, Lv, yv, double_softmax=False, tag=""):
    """Train, and keep the weights from the best validation epoch.

    Early stopping, from Lecture 6. Reporting the last epoch instead is worth
    several points to whichever run overfits hardest — and the run that
    overfits hardest is the one with the pretrained embeddings, because it
    starts from vectors that already mean something.
    """
    net = net.to(device)
    params = [p for p in net.parameters() if p.requires_grad]
    opt    = torch.optim.Adam(params, lr=LR)
    lossf  = nn.CrossEntropyLoss()
    Xf_d, yf_d = Xf.to(device), torch.from_numpy(yf).to(device)
    curve, losses = [], []
    best_state, best_val, best_epoch = None, -1.0, 0
    t0 = time.perf_counter()
    for ep in range(EPOCHS):
        net.train()
        perm, running = torch.randperm(len(Xf_d)), 0.0
        for i in range(0, len(Xf_d), BATCH):
            j = perm[i:i + BATCH]
            opt.zero_grad()
            out = net(Xf_d[j], Lf[j])
            if double_softmax:
                out = torch.softmax(out, dim=1)
            loss = lossf(out, yf_d[j])
            loss.backward()
            opt.step()
            running += float(loss.detach()) * len(j)
        losses.append(running / len(Xf_d))
        curve.append(accuracy(net, Xv, Lv, yv))
        if curve[-1] > best_val:
            best_val, best_epoch = curve[-1], ep + 1
            best_state = {k: v.detach().cpu().clone()
                          for k, v in net.state_dict().items()}
        print(f"  {tag} epoch {ep + 1}: loss {losses[-1]:.4f}  "
              f"val {curve[-1]:.4f}  ({time.perf_counter() - t0:.0f}s)")
    net.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    print(f"  {tag}: keeping epoch {best_epoch} (val {best_val:.4f})")
    return net, curve, losses
'''),
        prompt(
            label="⏱ 40-90 s — our words, random table",
            input="the word-tokenised batches",
            output="the trained model and its validation curve",
            constraint="re-seed immediately before, so every configuration below starts from the same initialisation",
            check="`torch.manual_seed` before EACH run, not once at the top. Any cell that consumes randomness in between shifts every configuration after it."),
        code('''
torch.manual_seed(RANDOM_STATE)
scratch, curve_scratch, loss_scratch = train(
    GRUClassifier(VOCAB), Xf_w, Lf_w, fit_y, Xv_w, Lv_w, val_y,
    tag="our words / random")
'''),

        md("""
## 13 · ⚠ Read before running — an assistant writes the same model

> *"Write a PyTorch model that classifies a padded batch of token ids with an
> embedding and a GRU, and returns two logits."*

Under-specified in exactly one place. The code it returns is the
`last_of_padding=True` branch above: it summarises at `out[:, -1, :]`, the last
position of the **padded** batch.

**Reviewer question 3: what is the shape here?** Position `MAXLEN - 1` is
padding for every review shorter than `MAXLEN` — which is most of them.
"""),
        prompt(
            label="⚠ what the assistant returns",
            input="'write a PyTorch model that classifies a padded batch of token ids with an embedding and a GRU, returning two logits'",
            output="the same model summarising at `out[:, -1, :]`, and what it costs",
            constraint="run it and compare — under-specified in exactly ONE place, and the code is otherwise correct",
            check="the prompt named the batch as padded and did not say what to do about it. A specification that mentions padding must say how the summary avoids it."),
        code('''
torch.manual_seed(RANDOM_STATE)
padbug, curve_padbug, _ = train(
    GRUClassifier(VOCAB, last_of_padding=True), Xf_w, Lf_w, fit_y,
    Xv_w, Lv_w, val_y, tag="last-of-padding")

print(f"\\ncorrect        {curve_scratch[-1]:.1%}")
print(f"last of padding {curve_padbug[-1]:.1%}")
print(f"the bug costs   {100 * (curve_scratch[-1] - curve_padbug[-1]):.2f} points")
'''),
        md("""
### The test that catches it

A padding bug is exactly the class of error that a test on **invariance**
catches and a test on output shape does not: pad the same review two different
ways and the logits must not move.
"""),
        prompt(
            label="the test that catches it",
            input="the same review, padded two different ways",
            output="whether the logits move",
            constraint="test INVARIANCE, not shape — a padding bug is exactly the class of error that an output-shape test cannot see",
            check="assert the correct model's logits agree to 1e-4, and print how far the buggy one moves. Feed the same content two ways and require the same answer. It is the text equivalent of evaluating the same model twice."),
        code('''
scratch.eval(); padbug.eval()
n = int(Lf_w[0])
one = scratch(Xf_w[:1, :n].to(device), torch.tensor([n]))
two = scratch(Xf_w[:1, :].to(device),  torch.tensor([n]))
assert torch.allclose(one, two, atol=1e-4), "padding is being read"
print("padding invariance holds for the correct model")

one_b = padbug(Xf_w[:1, :n].to(device), torch.tensor([n]))
two_b = padbug(Xf_w[:1, :].to(device),  torch.tensor([n]))
print(f"the assistant's model moves by "
      f"{(one_b - two_b).abs().max().item():.3f} — same review, different padding")
'''),

        md("""
## 14 · The swap: change the table, and nothing else

The diagnosis is not the architecture. It is that the embedding table is being
asked to learn English from 5,000 sentiment labels.

Two changes, applied **one at a time**, so the measurement says which one did
the work.
"""),
        prompt(
            label="change ONE thing — the tokenizer",
            input="the same reviews, tokenised into subwords",
            output="the subword batches, and a model trained on them from random vectors",
            constraint="change the tokenizer and NOTHING else — same architecture, same seed, same epochs — so the measurement says which change did the work",
            check="assert the batch shape before training on it. A subword tokenizer is not a better tokenizer by itself. It is a vocabulary that someone else's pretraining can be poured into."),
        code('''
def encode_pieces(texts):
    out  = tk(list(texts), truncation=True, max_length=MAXLEN,
              padding="max_length")
    ids  = np.asarray(out["input_ids"], dtype=np.int64)
    lens = np.asarray(out["attention_mask"], dtype=np.int64).sum(1)
    return torch.from_numpy(ids), torch.from_numpy(np.maximum(lens, 1))

Xf_s, Lf_s = encode_pieces(fit_x)
Xv_s, Lv_s = encode_pieces(val_x)
assert Xf_s.shape == (N_FIT, MAXLEN)
print(f"subword batch {tuple(Xf_s.shape)}   vocabulary {tk.vocab_size:,}")

torch.manual_seed(RANDOM_STATE)
_, curve_wp_random, _ = train(
    GRUClassifier(tk.vocab_size), Xf_s, Lf_s, fit_y, Xv_s, Lv_s, val_y,
    tag="subword / random")
'''),
        md("""
At the deck's scale these two land within a few hundredths of a point of each
other — a **null result, not a ranking**, since two single-seed numbers that
close say only that the effect is smaller than the seed-to-seed spread. At this
notebook's smaller scale it is usually worse. Either way the reading is the
same:

* there was little to win — the token-level OOV rate was already a few per cent;
* sequences are longer in pieces, so the same budget of positions holds fewer
  words;
* `un`, `##watch`, `##able` are three random vectors, and the model has to learn
  that their *composition* is negative.

A subword tokenizer is not a better tokenizer by itself. It is a vocabulary that
someone else's pretraining can be poured into.
"""),
        prompt(
            label="pour the pretraining in",
            input="DistilBERT's 768-wide embedding table",
            output="a 128-wide projection of it, rescaled to match nn.Embedding's own scale",
            constraint="rescale after projecting — PCA components have the variance of the data, and dropping a table with the wrong scale into a randomly-initialised architecture changes the effective learning rate of everything downstream",
            check="assert the projected shape, and report how much variance 128 components keep. When you transplant a fitted object into a different architecture, match its scale to what the architecture expects. `Z / Z.std() * 0.1` is that line, and it is easy to leave out."),
        code('''
# The same vocabulary already has a trained embedding table. It is 768 wide and
# our architecture is 128 wide, so project with PCA — thread 5, from Lecture 10.
from sklearn.decomposition import PCA
from transformers import AutoModel

bert = AutoModel.from_pretrained(BERT)
E = bert.embeddings.word_embeddings.weight.detach().numpy()[:tk.vocab_size]
print(f"pretrained table: {E.shape}")

pca = PCA(n_components=EMB_DIM, random_state=RANDOM_STATE)
Z = pca.fit_transform(E)
Z = (Z / Z.std() * 0.1).astype(np.float32)      # match nn.Embedding's own scale

assert Z.shape == (tk.vocab_size, EMB_DIM)
print(f"{EMB_DIM} components keep "
      f"{pca.explained_variance_ratio_.sum():.1%} of the variance")
'''),
        prompt(
            label="frozen, then tuned",
            input="the pretrained table, once held fixed and once allowed to move",
            output="both validation curves",
            constraint="run BOTH — frozen isolates what the pretrained vectors are worth, and tuned shows what adapting them adds",
            check="two runs, one variable. Frozen against tuned is the cheapest possible ablation and it answers a question the single tuned run cannot."),
        code('''
torch.manual_seed(RANDOM_STATE)
_, curve_frozen, _ = train(
    GRUClassifier(tk.vocab_size, init=Z, freeze=True),
    Xf_s, Lf_s, fit_y, Xv_s, Lv_s, val_y, tag="subword / frozen")

torch.manual_seed(RANDOM_STATE)
tuned, curve_tuned, _ = train(
    GRUClassifier(tk.vocab_size, init=Z, freeze=False),
    Xf_s, Lf_s, fit_y, Xv_s, Lv_s, val_y, tag="subword / tuned")
'''),

        md("""
## 15 · The test set. Once.

Everything above used the fit and validation splits only. This is the first and
last time the test half is scored.

⏱ **about a minute** — encoding 25,000 reviews twice and two forward passes.
"""),
        prompt(
            label="the test set, once",
            input="all 25,000 test reviews, encoded both ways",
            output="every configuration's test accuracy, beside both anchors",
            constraint="encode with BOTH tokenizers — the word models and the subword models cannot share a test batch",
            check="assert both encodings produced 25,000 rows. Keep the anchors in the final table. The bag of words is in that column for a reason, and it is not there to be flattered."),
        code('''
Xt_w, Lt_w = encode_words(test_x)
Xt_s, Lt_s = encode_pieces(test_x)
assert Xt_w.shape[0] == 25_000 and Xt_s.shape[0] == 25_000

results = {
    "always one class":               majority,
    "tf-idf + logistic regression":   bow_acc,
    "GRU, our words, random":         accuracy(scratch, Xt_w, Lt_w, test_y),
    "GRU, subword, pretrained tuned": accuracy(tuned,   Xt_s, Lt_s, test_y),
    "GRU, the assistant's version":   accuracy(padbug,  Xt_w, Lt_w, test_y),
}
for name, acc in results.items():
    print(f"{name:34s} {acc:.1%}")
'''),
        prompt(
            label="four curves",
            input="every configuration's validation curve",
            output="all four on one axis",
            constraint="one axis, so the four are comparable, and integer epoch ticks — there are two of them and matplotlib will otherwise offer 1.5",
            check="when you have very few points, plot markers and not just lines. A line between two points invites extrapolation that the data cannot support."),
        code('''
plt.figure(figsize=(9, 3))
for label, c in (("our words, random", curve_scratch),
                 ("subword, random", curve_wp_random),
                 ("subword, pretrained frozen", curve_frozen),
                 ("subword, pretrained tuned", curve_tuned)):
    plt.plot(range(1, EPOCHS + 1), [100 * v for v in c], "o-", label=label)
plt.xticks(range(1, EPOCHS + 1))
plt.xlabel("epoch"); plt.ylabel("validation accuracy, %")
plt.legend(); plt.tight_layout(); plt.show()
'''),

        md("""
## 12b · The requirement the metric cannot see

The brief said *"within the hour"*, which makes inference cost a stated
requirement — and accuracy says nothing about it. So time it: raw strings
in, labels out, tokenising **included**, because that is what the desk pays
per review.

A wall clock is the least reproducible number you will produce, so take the
**median of several passes** and never a single one.

⏱ **two to three minutes** — it re-tokenises the whole test set on every
pass, which is exactly the cost being measured.
"""),
        prompt(
            label="⏱ 2-3 min — the requirement the metric cannot see",
            input="the full test set, three times per configuration",
            output="the median seconds per pass and the milliseconds per review",
            constraint="time raw strings IN and labels OUT, tokenising INCLUDED — that is what the desk pays per review — and take the MEDIAN of several passes, never a single one",
            check="assert counting words is the cheapest, which is the expected ordering and worth failing loudly if it is not. None of these is anywhere near the stated hour, so the cost column does not decide anything here. A requirement that turns out not to bind is still worth measuring: you did not know it did not bind until you measured."),
        code('''
def score_time(fn, repeats=3):
    """Median of `repeats` full passes. The mean is dominated by whichever
    pass collided with something else on the machine."""
    runs = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        runs.append(time.perf_counter() - t0)
    return sorted(runs)[len(runs) // 2], min(runs), max(runs)

@torch.no_grad()
def rnn_pass(net, enc):
    X, L = enc(test_x)
    net.eval()
    for i in range(0, len(X), 256):
        net(X[i:i + 256].to(device), L[i:i + 256])

costs = {
    "tf-idf + logistic regression": score_time(
        lambda: bow.predict(vec.transform(test_x))),
    "GRU, our words":  score_time(lambda: rnn_pass(scratch, encode_words)),
    "GRU, subword":    score_time(lambda: rnn_pass(tuned,   encode_pieces)),
}
for name, (med, lo, hi) in costs.items():
    print(f"{name:30s} {med:6.1f} s   {1000 * med / len(test_x):5.2f} ms/review"
          f"   (min {lo:.1f}, max {hi:.1f})")

cheapest = min(costs, key=lambda k: costs[k][0])
assert cheapest.startswith("tf-idf"), f"expected counting words to win, got {cheapest}"
'''),
        md("""
None of these is anywhere near an hour, so on this brief the cost column does
not decide anything. A requirement that turns out not to bind is still worth
measuring — you did not know it did not bind until you measured. It starts
to bind in the next lecture, where the model is 25 times larger.
"""),

        md("""
## 16 · Where we are

Write your **best accuracy** on the same sheet of paper, next to what you
predicted. Bring it to the next lecture — we open by comparing them.

Four questions we did not answer, and all four are the next lecture:

1. What exactly does `CrossEntropyLoss` receive, and why does it want the raw
   two numbers rather than probabilities?
2. We borrowed a tokenizer and a table. What if we borrowed the whole model?
3. The desk wants complaints *grouped*, not only flagged.
4. Is the last number good? Compared with what ceiling?

Do not fix anything yet.
""")]
