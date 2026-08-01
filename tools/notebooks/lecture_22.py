"""
Lecture 22 — Reusing what someone else learned.

Thread 11: cross-entropy, softmax and logits. Then fine-tuning a pretrained
transformer, sentence embeddings for search, clustering for recurring
complaints, and the tokeniser leak.

Exports build() -> list[nbformat cell]. Self-contained: it reloads and re-splits
the corpus rather than assuming Lecture 21's kernel is still alive.

The deck fine-tunes on 20,000 reviews; this notebook uses 2,000 and scores on
3,000, so a free Colab runtime finishes inside the hour. Every cell says which.
"""

from __future__ import annotations

import nbformat as nbf
from _prompt import prompt                                # noqa: E402


def md(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(text.strip("\n"))


def code(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(text.strip("\n"))


HEADER = """
# Reusing what someone else learned

**Lecture 22 · Fix** · Géron, Chapters 14–15 · *Mathematical thread:
cross-entropy, softmax and logits*

Applications of Machine Learning — BSc Mathematics of Artificial Intelligence

---

**How to use this notebook.** Read before you run. Cells marked
**⚠ read before running** contain a defect on purpose, and neither of today's
two defects raises an exception.

**Scale.** The deck fine-tunes on 20,000 reviews and scores on all 25,000 test
reviews. Here we fine-tune on **2,000** and score on **3,000** so the notebook
finishes in a few minutes. The accuracies are lower than the deck's; the
ordering is the same.
"""


def build() -> list:
    return [
        md(HEADER),

        md("## 1 · Setup and the same corpus"),
        prompt(
            label="setup, and one line that has to be first",
            input="nothing",
            output="thread settings, then versions, seeds and the device",
            constraint="set the OpenMP variables BEFORE torch is imported — they are read at import time, and after that they do nothing",
            left_open="why they are there at all. On macOS, PyTorch and scikit-learn each ship their own OpenMP runtime, and with both loaded the KMeans cell near the end of this notebook deadlocks.",
            student="moving the imports to the top for tidiness, which restores the deadlock — and a deadlock is not an error message, it is a cell that never finishes.",
            catch="when an environment variable must precede an import, say so in a comment beside it. The next person to tidy the file has no other way to know."),
        code('''
# Not examinable, and only needed on macOS: PyTorch and scikit-learn each ship
# their own OpenMP runtime, and with both loaded the KMeans cell near the end of
# this notebook deadlocks. It has to be set BEFORE torch is imported, which is
# why it is the first thing in the notebook.
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import sys, re, time, tarfile, urllib.request
from pathlib import Path
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

print(f"python  {sys.version.split()[0]}")
print(f"torch   {torch.__version__}")

RANDOM_STATE = 42
torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
print(f"device  {device}")
'''),
        prompt(
            label="the same corpus, the same split",
            input="the IMDb tarball",
            output="the identical fit / validation split as the previous lecture",
            constraint="rebuild from the same seed — a notebook that only runs because a previous one is still in memory is not reproducible",
            check="assert both halves are 25,000 and that the two index sets are disjoint",
            left_open="that the seed is what guarantees the comparison. Nothing else in this notebook checks that the two lectures share a split.",
            student="importing the variables from the other notebook, or trusting that the kernel still holds them.",
            catch="reproduce, do not inherit. It costs thirty seconds of download and it is the difference between a comparison and a coincidence."),
        code('''
# Reloaded here rather than inherited. A notebook that only runs because a
# previous one is still in memory is not reproducible.
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

N_FIT, N_VAL = 5_000, 2_000
rng   = np.random.default_rng(RANDOM_STATE)
order = rng.permutation(len(train_x))
fit_i, val_i = order[:N_FIT], order[N_FIT:N_FIT + N_VAL]
fit_x = [train_x[i] for i in fit_i]; fit_y = train_y[fit_i]
val_x = [train_x[i] for i in val_i]; val_y = train_y[val_i]

assert len(train_x) == 25_000 and len(test_x) == 25_000
assert set(fit_i).isdisjoint(val_i)
print(f"fit {len(fit_x):,}   val {len(val_x):,}   test {len(test_x):,}")
print("same split as the previous lecture — the seed guarantees it")
'''),

        md("""
## 2 · Thread 11 — softmax, and the shift that changes nothing

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
            left_open="the algebra: adding a constant to every logit multiplies numerator and denominator by the same e^c, so the FUNCTION ignores the shift. float32 does not.",
            student="assuming softmax is safe because it is a library function. It is, because the library subtracts the max — this cell is what that subtraction is for.",
            catch="print where exp overflows float32. It is about 88.7, which is a much smaller number than people expect."),
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
            check="assert the two agree to 1e-6",
            left_open="that these are two different claims. One is about mathematics and one is about float32, and mixing them in a single example lets you conclude the wrong thing from the right output.",
            student="testing the invariance on the overflowing example and concluding softmax is not shift-invariant.",
            catch="when a property holds mathematically and fails numerically, demonstrate each on its own inputs."),
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
## 3 · Cross-entropy, and its gradient with respect to the logits

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
            check="assert agreement to 1e-10, that each row of the gradient sums to zero, and that every component is in [−1, 1]",
            left_open="why the derivation is three lines. The exponential of the true class CANCELS: L = −z_c + logsumexp(z), so differentiating needs no chain rule through the softmax at all.",
            student="taking `p − y` on faith. It is the most-quoted gradient in machine learning and it is four lines to check against the library.",
            catch="the row-sum assert is the interesting one. The gradient has no component along the all-ones direction, which is the derivative form of the shift invariance above."),
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
## 4 · Why the loss consumes logits

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
            check="assert the stable form never fails, at either end of the sweep",
            left_open="why sweeping the logit standard deviation instead buries the effect: most rows then have a loss near zero, where both forms agree trivially.",
            student="sweeping the logit scale, seeing no difference, and concluding the numerical argument is folklore. The experiment has to vary the thing the failure depends on.",
            catch="score against float64 rather than against each other. Two float32 computations agreeing tells you nothing about either."),
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
            left_open="that the loud failure is the harmless one. A nan propagates and someone notices; a finite answer that is wrong in the fourth decimal place ships.",
            student="plotting only the non-finite rate, which is zero across most of the range and suggests the naive form is fine there.",
            catch="clamp the log-axis values away from zero before plotting. A median error of exactly 0.0 is not plottable on a log scale and matplotlib's response is to drop the point silently."),
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
            check="assert the first is non-finite and the second is off by more than 1e-3, so both failures are pinned",
            left_open="which of the two is dangerous. The first stops your run. The second is the one that ships.",
            student="testing numerical stability only on the case that produces inf, concluding the library handles it, and never seeing the precision loss.",
            catch="print PyTorch's answer beside your own. It agrees with the combined form, which is evidence that the combined form is what the library does."),
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
## 5 · Cross-entropy and KL divergence

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
            check="assert H + KL equals CE to 1e-12 for both, and that the one-hot entropy is exactly zero",
            left_open="the proof, which is one line: add and subtract Σ y log y. And the consequence: for a one-hot target, minimising cross-entropy IS minimising the KL divergence to the label.",
            student="reading 'cross-entropy is KL divergence' as an identity. It is an identity up to the entropy of the target, and that term is zero only for hard labels.",
            catch="guard the logs against zeros with a mask on p > 0. 0·log 0 is 0 by convention and nan in floating point."),
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
## 6 · ⚠ Read before running — an assistant "improves" the model

> *"The model returns raw numbers. Add a softmax to the output so it returns
> probabilities, and keep the training loop working."*

**Reviewer question 5: what is the default I did not ask for?**
`nn.CrossEntropyLoss` applies its own `log_softmax`. Applying one yourself gives
a softmax of a softmax, whose input lives in $[0, 1]$ — so on two classes the
output can never exceed $e/(e+1) \\approx 0.731$, and the loss is floored near
$-\\log 0.731 \\approx 0.313$.

Nothing raises. The loss still goes down.
"""),
        prompt(
            label="the previous lecture's model, rebuilt",
            input="the fit and validation reviews",
            output="the vocabulary, the encoders, the classifier, and the padded batches",
            constraint="everything rebuilt here — vocabulary from the fit split only, packing rather than last-of-padding",
            check="assert the batch shape",
            left_open="that this is a straight copy of the previous lecture's working version, defects removed. The defect this lecture is about is added in the NEXT cell, as an argument.",
            student="skimming this because it is familiar. The `padding_idx=0` and the packing are both here, and both were lessons.",
            catch="when you carry an architecture forward, carry the fixes with it. The last-of-padding bug is not in this version and that is deliberate."),
        code('''
MAXLEN, EMB_DIM, HIDDEN = 192, 128, 64
VOCAB, BATCH, EPOCHS, LR = 20_000, 64, 2, 1e-3

WORD_RE = re.compile(r"[a-z0-9']+")
word_tokens = lambda s: WORD_RE.findall(s.lower().replace("<br />", " "))

fit_counts = Counter(w for s in fit_x for w in word_tokens(s))
w2i = {w: i + 2 for i, (w, _) in enumerate(fit_counts.most_common(VOCAB - 2))}

def pad_batch(seqs):
    X = np.zeros((len(seqs), MAXLEN), dtype=np.int64)
    L = np.zeros(len(seqs), dtype=np.int64)
    for i, s in enumerate(seqs):
        s = s[:MAXLEN]
        X[i, :len(s)] = s
        L[i] = max(len(s), 1)
    return torch.from_numpy(X), torch.from_numpy(L)

def encode_words(texts):
    return pad_batch([[w2i.get(w, 1) for w in word_tokens(s)] for s in texts])

class GRUClassifier(nn.Module):
    def __init__(self, vocab):
        super().__init__()
        self.emb  = nn.Embedding(vocab, EMB_DIM, padding_idx=0)
        self.rnn  = nn.GRU(EMB_DIM, HIDDEN, batch_first=True, bidirectional=True)
        self.head = nn.Linear(2 * HIDDEN, 2)

    def forward(self, x, lengths):
        e = self.emb(x)
        packed = nn.utils.rnn.pack_padded_sequence(
            e, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, h = self.rnn(packed)
        return self.head(torch.cat([h[0], h[1]], dim=1))

Xf, Lf = encode_words(fit_x)
Xv, Lv = encode_words(val_x)
assert Xf.shape == (N_FIT, MAXLEN)
print(f"fit batch {tuple(Xf.shape)}")
'''),
        prompt(
            label="⏱ 1-3 min — ⚠ two runs, one extra softmax",
            input="'add a softmax to the output so it returns probabilities, and keep the training loop working'",
            output="both models trained, with their loss curves",
            constraint="identical seeds and identical everything else — the only difference is one `torch.softmax` before the loss",
            left_open="reviewer question 5. `nn.CrossEntropyLoss` applies its OWN log_softmax, so applying one yourself gives a softmax of a softmax, whose input lives in [0,1] — on two classes the output can never exceed e/(e+1) ≈ 0.731.",
            student="exactly this request, which is reasonable on its face. Nothing raises, and the loss still goes down.",
            catch="the loss is FLOORED near −log(0.731) ≈ 0.313. That is the tell, and it is visible in the training output long before the accuracy comparison."),
        code('''
# ⏱ about 1-3 minutes for the two runs together, on a GPU.
@torch.no_grad()
def accuracy(net, X, L, y, batch=256):
    net.eval()
    hits = 0
    for i in range(0, len(X), batch):
        out = net(X[i:i + batch].to(device), L[i:i + batch])
        hits += int((out.argmax(1).cpu().numpy() == y[i:i + batch]).sum())
    return hits / len(y)

def train(net, double_softmax=False, tag=""):
    net = net.to(device)
    opt   = torch.optim.Adam(net.parameters(), lr=LR)
    lossf = nn.CrossEntropyLoss()
    Xd, yd = Xf.to(device), torch.from_numpy(fit_y).to(device)
    losses = []
    for ep in range(EPOCHS):
        net.train()
        perm, running = torch.randperm(len(Xd)), 0.0
        for i in range(0, len(Xd), BATCH):
            j = perm[i:i + BATCH]
            opt.zero_grad()
            out = net(Xd[j], Lf[j])
            if double_softmax:
                out = torch.softmax(out, dim=1)
            loss = lossf(out, yd[j])
            loss.backward()
            opt.step()
            running += float(loss) * len(j)
        losses.append(running / len(Xd))
        print(f"  {tag} epoch {ep + 1}: loss {losses[-1]:.4f}  "
              f"val {accuracy(net, Xv, Lv, val_y):.4f}")
    return net, losses

torch.manual_seed(RANDOM_STATE)
good, loss_good = train(GRUClassifier(VOCAB), tag="logits into the loss")
torch.manual_seed(RANDOM_STATE)
bad,  loss_bad  = train(GRUClassifier(VOCAB), double_softmax=True,
                        tag="probabilities into the loss")
'''),
        prompt(
            label="look at the loss column first",
            input="both trained models",
            output="final loss and validation accuracy for each, and the predicted floor",
            constraint="print the ALGEBRAIC floor, −log(e/(e+1)), beside the measured loss — a predicted number matching a measured one is the argument",
            left_open="that the accuracy difference is the smaller signal. The loss stopping exactly where the algebra said it would is what identifies the cause.",
            student="diagnosing from the accuracy, which is a few points down and could be anything. The floored loss could only be one thing.",
            catch="when a loss plateaus at a value you can compute in closed form, compute it. It names the bug rather than merely detecting it."),
        code('''
acc_good = accuracy(good, Xv, Lv, val_y)
acc_bad  = accuracy(bad,  Xv, Lv, val_y)
print(f"logits        loss {loss_good[-1]:.4f}   val {acc_good:.1%}")
print(f"probabilities loss {loss_bad[-1]:.4f}   val {acc_bad:.1%}")
print(f"\\nthe extra softmax costs {100 * (acc_good - acc_bad):.2f} points")
print(f"and its loss is floored near -log(e/(e+1)) = "
      f"{-np.log(np.e / (np.e + 1)):.4f}")
'''),
        md("""
**Look at the loss column first.** It stops above 0.3, exactly where the algebra
said it would — *that* is the tell, not the accuracy.

The corrected specification: keep `forward` returning logits; add a separate
`predict_proba` used only at inference; and assert that an untrained model's
loss on balanced classes is within 0.05 of $\\log 2$.
"""),
        prompt(
            label="the assertion that would have caught it",
            input="an untrained model on a balanced batch",
            output="its loss, against log 2",
            constraint="assert it is within 0.05 of log 2 — an untrained two-class model on balanced data has no information and must score the entropy of a coin flip",
            left_open="how many bugs this catches. A wrong head size, transposed targets, a double softmax, and labels that are not what you think they are all fail it.",
            student="having no smoke test at all, so the first evidence that something is wrong is a slightly disappointing accuracy at the end of training.",
            catch="the corrected specification is three parts: `forward` returns logits, a separate `predict_proba` is used only at inference, and this assertion runs before training starts."),
        code('''
untrained = GRUClassifier(VOCAB).to(device)
with torch.no_grad():
    l0 = float(nn.CrossEntropyLoss()(untrained(Xf[:512].to(device), Lf[:512]),
                                     torch.from_numpy(fit_y[:512]).to(device)))
print(f"untrained loss {l0:.4f}   log 2 = {np.log(2):.4f}")
assert abs(l0 - np.log(2)) < 0.05, "the head or the targets are wrong"
'''),

        md("""
## 7 · Borrow the whole model

Our GRU saw 5,000 reviews. A pretrained language model saw billions of words —
and needed no labels at all to do it, because its task was predicting missing
words.

⏱ **about 1–3 minutes** to fine-tune, on a GPU.
"""),
        prompt(
            label="⏱ 1-3 min — borrow the whole model",
            input="DistilBERT and 2,000 fit reviews",
            output="the model, the encoded batches, and a balanced scoring subset",
            constraint="SHUFFLE before taking the scoring subset — the corpus ships every positive first, so `test_x[:3000]` is all positives and any 'accuracy' on it is really recall",
            check="assert the scoring subset is within three points of balanced",
            left_open="what was borrowed. Our GRU saw 5,000 reviews; a pretrained language model saw billions of words, and needed no labels at all to do it, because its task was predicting missing words.",
            student="taking a prefix of the test set. On this corpus that is a 100% positive evaluation set, and a model that always answers positive scores 100%.",
            catch="assert the class balance of any subset you score on. It is one line and it catches the most embarrassing possible result."),
        code('''
from transformers import AutoModelForSequenceClassification, AutoTokenizer

BERT = "distilbert-base-uncased"
tk = AutoTokenizer.from_pretrained(BERT)
model = AutoModelForSequenceClassification.from_pretrained(
    BERT, num_labels=2).to(device)
print(f"{sum(p.numel() for p in model.parameters()):,} parameters")

def encode_bert(texts):
    out = tk(list(texts), truncation=True, max_length=MAXLEN,
             padding="max_length", return_tensors="pt")
    return out["input_ids"], out["attention_mask"]

N_FT, N_SCORE = 2_000, 3_000        # the deck uses 20,000 and all 25,000
ids, am = encode_bert(fit_x[:N_FT])
yb = torch.from_numpy(fit_y[:N_FT])
assert ids.shape == (N_FT, MAXLEN)

# The corpus ships every positive review first and every negative second, so
# test_x[:3000] is all positives and any "accuracy" on it is really recall.
# Shuffle once and take every subset through this.
score_i = np.random.default_rng(RANDOM_STATE + 7).permutation(len(test_x))[:N_SCORE]
score_x = [test_x[i] for i in score_i]
score_y = test_y[score_i]
assert abs(score_y.mean() - 0.5) < 0.03, "the scoring subset is not balanced"
print(f"scoring on {N_SCORE:,} test reviews, {score_y.mean():.1%} positive")
'''),
        prompt(
            label="the floor, before any training",
            input="the pretrained body with a random head",
            output="its accuracy on the scoring subset",
            constraint="measure BEFORE fine-tuning — the head is random and nothing has been trained, so this is the floor the fine-tune has to beat",
            left_open="that a random head on a pretrained body is near chance, which is the expected result and worth seeing rather than assuming.",
            student="reporting the fine-tuned number with no zero-shot baseline, so there is no evidence that the fine-tuning did anything.",
            catch="a pretrained model with a fresh head is not a zero-shot classifier. The body is informative and the head is noise, and this number measures the pair."),
        code('''
@torch.no_grad()
def bert_accuracy(model, texts, labels, batch=32):
    model.eval()
    i_all, a_all = encode_bert(texts)
    preds = np.zeros(len(texts), dtype=np.int64)
    for i in range(0, len(texts), batch):
        lg = model(input_ids=i_all[i:i + batch].to(device),
                   attention_mask=a_all[i:i + batch].to(device)).logits
        preds[i:i + batch] = lg.argmax(1).cpu().numpy()
    return (preds == labels).mean()

# the head is random and nothing has been trained: this is the floor
zero_shot = bert_accuracy(model, score_x, score_y)
print(f"pretrained body, random head, no training: {zero_shot:.1%}")
'''),
        prompt(
            label="fine-tune, at a hundredth of the learning rate",
            input="2,000 reviews, one epoch, batch 16",
            output="the loss every 25 steps and the wall clock",
            constraint="`lr=2e-5`, about a hundred times smaller than the 1e-3 used for the from-scratch model — the body already encodes something and a large step destroys it in a few dozen steps",
            left_open="that the loss goes into the loop as LOGITS, exactly as before. Nothing about a pretrained model changes that.",
            student="reusing 1e-3 because it worked for the GRU. The pretrained weights are wiped in the first few dozen steps and the result is worse than training from scratch.",
            catch="print the loss during training, not only at the end. A fine-tune going wrong is visible in the first twenty steps and takes minutes to confirm at the end."),
        code('''
torch.manual_seed(RANDOM_STATE)
opt   = torch.optim.AdamW(model.parameters(), lr=2e-5)   # ~100x smaller than 1e-3
lossf = nn.CrossEntropyLoss()                            # logits in, as always

t0 = time.perf_counter()
model.train()
perm = torch.randperm(len(ids))
for k, i in enumerate(range(0, len(ids), 16)):
    j = perm[i:i + 16]
    opt.zero_grad()
    out = model(input_ids=ids[j].to(device),
                attention_mask=am[j].to(device)).logits
    loss = lossf(out, yb[j].to(device))
    loss.backward()
    opt.step()
    if k % 25 == 0:
        print(f"  step {k:3d}: loss {float(loss):.4f} "
              f"({time.perf_counter() - t0:.0f}s)")
print(f"one epoch on {N_FT:,} reviews: {time.perf_counter() - t0:.0f}s")
'''),
        prompt(
            label="the comparison",
            input="every model, on the same balanced scoring subset",
            output="four accuracies, and the error count removed",
            constraint="report ERRORS REMOVED as well as points gained — going from 85% to 90% removes a third of the mistakes, and the desk cares about the mistakes",
            check="assert the fine-tune is above 0.75, with a message naming the learning rate as the likely cause",
            left_open="that this assert is a FLOOR and not the headline. At this scale one epoch on 2,000 reviews should still be far above the majority class; if it is not, something is broken rather than disappointing.",
            student="quoting the accuracy gain and stopping. Percentage points compress as accuracy rises, and the error-reduction figure is what communicates.",
            catch="score the GRU on the SAME shuffled subset, not on its own. Two accuracies on two different subsets are not a comparison."),
        code('''
ft_acc      = bert_accuracy(model, score_x, score_y)
Xt, Lt      = encode_words(score_x)
scratch_acc = accuracy(good, Xt, Lt, score_y)

print(f"{'always one class':32s} {max(test_y.mean(), 1-test_y.mean()):.1%}")
print(f"{'DistilBERT, untrained head':32s} {zero_shot:.1%}")
print(f"{'GRU from scratch':32s} {scratch_acc:.1%}")
print(f"{'DistilBERT, fine-tuned':32s} {ft_acc:.1%}")
print(f"\\ngain over the from-scratch model: "
      f"{100 * (ft_acc - scratch_acc):.2f} points")

errs_scratch = int(round((1 - scratch_acc) * N_SCORE))
errs_ft      = int(round((1 - ft_acc) * N_SCORE))
print(f"errors out of {N_SCORE:,}: {errs_scratch} -> {errs_ft}  "
      f"({(errs_scratch - errs_ft) / max(errs_scratch, 1):.0%} of them removed)")

# A floor, not the headline: at this scale one epoch on 2,000 reviews should
# still be far above the majority class. If it is not, the learning rate is
# wrong — 1e-3 destroys a pretrained body in a few dozen steps.
assert ft_acc > 0.75, f"fine-tuning failed ({ft_acc:.3f}) — check the learning rate"
'''),

        md("""
## 8 · Past classification: one vector per review

The desk asked for two more things — *"find me the ones like this one"* and
*"tell me what people keep complaining about"*. Neither is classification, and
both fall out of the same representation.

Note line 4 below: the mean is over the **real** tokens, using the attention
mask. Dividing by the padded length instead is Lecture 21's padding bug in a new
costume.
"""),
        prompt(
            label="⏱ 30-60 s — one vector per review",
            input="2,000 negative reviews",
            output="unit-norm sentence embeddings",
            constraint="mean-pool over the REAL tokens using the attention mask — dividing by the padded length instead is the previous lecture's padding bug in a new costume",
            check="assert every vector has norm 1",
            left_open="that neither of the desk's other two requests is classification, and both fall out of the same representation.",
            student="`out.last_hidden_state.mean(1)`, which averages over the padding too. Longer reviews are diluted more than short ones, and the effect is invisible in every shape check.",
            catch="normalise, and assert the norms. On the unit sphere the dot product IS the cosine, which makes the next cell one matrix product."),
        code('''
from transformers import AutoModel

MINILM = "sentence-transformers/all-MiniLM-L6-v2"
stk = AutoTokenizer.from_pretrained(MINILM)
smodel = AutoModel.from_pretrained(MINILM).to(device).eval()

@torch.no_grad()
def embed(texts, batch=64):
    vecs = []
    for i in range(0, len(texts), batch):
        enc = stk(list(texts[i:i + batch]), truncation=True, max_length=256,
                  padding=True, return_tensors="pt").to(device)
        out  = smodel(**enc).last_hidden_state
        mask = enc["attention_mask"].unsqueeze(-1).float()
        pooled = (out * mask).sum(1) / mask.sum(1)      # real tokens only
        vecs.append(torch.nn.functional.normalize(pooled, dim=1).cpu().numpy())
    return np.vstack(vecs)

# ⏱ about 30-60 seconds for 2,000 reviews.
neg = [t for t, y in zip(test_x, test_y) if y == 0][:2_000]
V = embed(neg)

assert V.shape[0] == len(neg)
assert np.allclose(np.linalg.norm(V, axis=1), 1.0, atol=1e-5), "not normalised"
print(f"{V.shape[0]:,} reviews, {V.shape[1]} dimensions, on the unit sphere")
'''),
        md("""
On the unit sphere the dot product *is* the cosine, so searching the corpus is
one matrix product.
"""),
        prompt(
            label="search the corpus",
            input="three complaint-shaped queries",
            output="the nearest review to each, with its cosine",
            constraint="embed the queries with the SAME function as the corpus — a different pooling or a different normalisation makes the cosines meaningless",
            check="assert the similarity matrix has the shape you expect",
            left_open="a failure mode worth measuring on your own queries: cosine similarity has no notion of negation, so 'the sound was perfect' and 'the sound was not perfect' sit close together.",
            student="reading three good results and deploying. Three queries is an anecdote, and the negation failure will not appear in three hand-written examples.",
            catch="print the cosine beside every result. A nearest neighbour at 0.31 and one at 0.78 are very different claims and the ranking hides that."),
        code('''
queries = ["the acting was wooden and unconvincing",
           "the sound mix made the dialogue impossible to follow",
           "far too long, it should have ended an hour earlier"]
Q = embed(queries)
sims = Q @ V.T
assert sims.shape == (len(queries), len(neg))

for i, q in enumerate(queries):
    j = int(np.argmax(sims[i]))
    print(f"\\nquery: {q}")
    print(f"  cosine {sims[i, j]:.3f}: {' '.join(neg[j].split())[:180]}")
'''),
        prompt(
            label="what keyword search returns instead",
            input="the same three queries",
            output="how many of the top three overlap between the two methods",
            constraint="compare the top-k SETS rather than the top-1 — a single disagreement could be a tie",
            left_open="why they differ: keyword search can only retrieve documents that reuse the query's words, and a complaint rarely uses the desk's vocabulary.",
            student="assuming semantic search is simply better. It is better at this, and it is worse at exact-match retrieval, which is what the desk will ask for next.",
            catch="when you replace a method, measure the overlap with what it replaced. Zero overlap and total overlap are both worth knowing about."),
        code('''
# What does keyword search return for the same queries?
from sklearn.feature_extraction.text import TfidfVectorizer

kvec = TfidfVectorizer(min_df=2).fit(neg)
ksim = (kvec.transform(queries) @ kvec.transform(neg).T).toarray()

for i, q in enumerate(queries):
    top_sem = set(np.argsort(-sims[i])[:3].tolist())
    top_key = set(np.argsort(-ksim[i])[:3].tolist())
    print(f"{len(top_sem & top_key)}/3 shared for: {q}")
'''),
        md("""
Keyword search can only retrieve documents that reuse the query's words, and a
complaint rarely uses the desk's vocabulary.

It also fails differently: cosine similarity has no notion of negation, so *the
sound was perfect* and *the sound was not perfect* sit close together. Measure
it on your own queries before deploying it.
"""),

        md("""
## 9 · Grouping the complaints

Lecture 9, unchanged: k-means, with $k$ chosen by silhouette rather than by eye.
"""),
        prompt(
            label="grouping the complaints",
            input="the sentence embeddings",
            output="the silhouette at six values of k, and the best clustering",
            constraint="choose k by silhouette rather than by eye — the same rule as application 5, unchanged by the data being text",
            check="assert the fitted clustering really has best_k distinct labels",
            left_open="`sample_size=1500` on the silhouette. It is an approximation for speed and it is seeded, so it is reproducible — but it is an approximation.",
            student="picking k = 5 because the desk asked for about five themes. The silhouette is what makes it a measurement rather than a preference.",
            catch="k-means on unit vectors is spherical k-means in all but name. That is appropriate here and it is worth knowing you have chosen it."),
        code('''
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

sil = {}
for k in (3, 4, 5, 6, 8, 10):
    km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit(V)
    sil[k] = silhouette_score(V, km.labels_, sample_size=1_500,
                              random_state=RANDOM_STATE)
    print(f"k={k:2d}  silhouette {sil[k]:.4f}")

best_k = max(sil, key=sil.get)
km = KMeans(n_clusters=best_k, n_init=10, random_state=RANDOM_STATE).fit(V)
assert len(np.unique(km.labels_)) == best_k
print(f"\\nbest k = {best_k}")
'''),
        prompt(
            label="name each group by what makes it different",
            input="the clusters and a tf-idf matrix",
            output="the five most distinctive terms per cluster, with the cluster size",
            constraint="rank by LIFT — the term's mean weight inside the cluster minus its mean outside — not by frequency inside it",
            left_open="`max_df=0.4` and the stop words. Without them every cluster is named 'film, movie, one, like', which is true and useless.",
            student="taking the commonest terms in each cluster. They are the commonest terms in the corpus, and every cluster gets the same name.",
            catch="print the cluster SIZE beside its terms. A theme covering nine reviews is not a theme the desk needs to hear about."),
        code('''
# Name each group by what makes it DIFFERENT, not by what is common everywhere.
gvec  = TfidfVectorizer(min_df=5, max_df=0.4, stop_words="english",
                        ngram_range=(1, 2))
G     = gvec.fit_transform(neg)
names = np.array(gvec.get_feature_names_out())

groups = []
for c in range(best_k):
    m    = km.labels_ == c
    lift = np.asarray(G[m].mean(0)).ravel() - np.asarray(G[~m].mean(0)).ravel()
    groups.append((int(m.sum()), names[np.argsort(-lift)[:5]].tolist()))

for size, terms in sorted(groups, reverse=True):
    print(f"{size:5d}  {', '.join(terms)}")
'''),

        md("""
## 10 · Red-team — ⚠ read before running

> *"Build a scikit-learn pipeline that vectorises the reviews with tf-idf and
> classifies them with logistic regression, and report the test accuracy."*

**Reviewer question 2: what was fitted, and on what?** `TfidfVectorizer.fit`
learns *which columns exist* and *the weight on every column*, from every
document it is given.

Measure the damage at two corpus sizes, because the answer depends on the size
and that is the lesson.
"""),
        prompt(
            label="⚠ the vectoriser fitted before the split",
            input="'build a pipeline that vectorises with tf-idf and classifies with logistic regression, and report the test accuracy'",
            output="the leaky and honest accuracies at 400 documents, over 20 seeds",
            constraint="measure at TWO corpus sizes, because the answer depends on the size and that is the lesson",
            check="assert the leaky vocabulary is at least as large as the honest one, which is what makes it the leaky one",
            left_open="reviewer question 2. `TfidfVectorizer.fit` learns which columns EXIST and the weight on every column, from every document it is given.",
            student="fitting the vectoriser on everything and then splitting, which is the natural reading of 'vectorise the reviews and classify them'.",
            catch="20 seeds at the small size. The effect is a fraction of a point and the seed-to-seed spread is larger, so a single split proves nothing in either direction."),
        code('''
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

all_x = list(train_x) + list(test_x)
all_y = np.concatenate([train_y, test_y])

def leak_experiment(n_docs, seeds):
    gaps = []
    for s in range(seeds):
        r   = np.random.default_rng(RANDOM_STATE + s)
        idx = r.choice(len(all_x), size=n_docs, replace=False)
        X   = [all_x[i] for i in idx]
        y   = all_y[idx]
        tr, te = train_test_split(np.arange(n_docs), test_size=0.25,
                                  random_state=RANDOM_STATE + s, stratify=y)

        # what the assistant wrote: fit on everything, then split
        v_bad = TfidfVectorizer(min_df=1, ngram_range=(1, 2), max_features=60_000)
        Z = v_bad.fit_transform(X)
        leaky = (LogisticRegression(max_iter=2000, C=4.0)
                 .fit(Z[tr], y[tr]).predict(Z[te]) == y[te]).mean()

        # what it should have written: split, then fit on the training half
        v_ok = TfidfVectorizer(min_df=1, ngram_range=(1, 2), max_features=60_000)
        Ztr  = v_ok.fit_transform([X[i] for i in tr])
        honest = (LogisticRegression(max_iter=2000, C=4.0)
                  .fit(Ztr, y[tr]).predict(v_ok.transform([X[i] for i in te]))
                  == y[te]).mean()

        assert Z.shape[1] >= Ztr.shape[1], "the leaky vocabulary must be larger"
        gaps.append(leaky - honest)
    return np.array(gaps)

# ⏱ about a minute.
small = leak_experiment(400, 20)
print(f"400 docs, 20 seeds:  {100 * small.mean():+.2f} points "
      f"(sd {100 * small.std():.2f}), leak wins on {(small > 0).sum()}/20")
'''),
        prompt(
            label="the same leak at full size",
            input="25,000 documents, three seeds",
            output="the gap, and both distributions as a strip plot",
            constraint="plot the individual seeds, not just the means — with three points at one size and twenty at the other, error bars would imply a precision neither has",
            left_open="why it shrinks. At 25,000 documents the inverse document frequencies are an average over 25,000 draws and removing a quarter of them barely moves any of them. At 400 the leaky vocabulary has columns that exist BECAUSE a test document used them.",
            student="measuring at full size only, finding nothing, and concluding the rule is superstition.",
            catch="the decision rule: fit the vectoriser inside the pipeline always — not because the damage is always large, but because it scales with the reciprocal of your corpus size, and the corpus is smallest exactly when the project starts."),
        code('''
# ⏱ about a minute: five seeds at the full corpus size.
full = leak_experiment(25_000, 3)
print(f"25,000 docs, 3 seeds: {100 * full.mean():+.2f} points "
      f"(sd {100 * full.std():.2f})")

plt.figure(figsize=(9, 2.6))
for i, (g, label) in enumerate(((small, "400 reviews"), (full, "25,000 reviews"))):
    plt.scatter(100 * g, np.full(len(g), i) + np.random.normal(0, .05, len(g)))
    plt.plot([100 * g.mean()], [i], "|", ms=30, mew=3, color="k")
plt.axvline(0, color="grey")
plt.yticks([0, 1], ["400 reviews", "25,000 reviews"])
plt.xlabel("accuracy the leak buys, in points")
plt.tight_layout(); plt.show()
'''),
        md("""
At 25,000 documents the inverse document frequencies are an average over 25,000
draws and removing a quarter of them barely moves any of them. At 400 the leaky
vocabulary has columns that exist *because* a test document used them.

**The decision rule:** fit the vectoriser inside the pipeline, always — not
because the damage is always large, but because it scales with the reciprocal of
your corpus size, and the corpus is smallest exactly when the project starts.
"""),
        prompt(
            label="the leak no pipeline protects you from",
            input="the training and test texts, whitespace-normalised",
            output="how many test reviews also appear in training, and how many duplicates are within training",
            constraint="normalise whitespace and case before comparing — an exact string match finds fewer duplicates than there are",
            left_open="that IMDb is clean because its authors deduplicated it, which also means the cost of duplication cannot be measured on it. The next cell builds a corpus that is not clean, and says so.",
            student="never checking. Three lines, and you know rather than assume.",
            catch="a duplicate across the split is not fixed by any pipeline, any cross-validation, or any amount of care about `fit` and `transform`. It is fixed by grouping."),
        code('''
# And the text leak no pipeline protects you from: the same document in both
# halves. Three lines, and you know rather than assume.
norm = lambda s: " ".join(s.lower().split())
train_norm = set(norm(s) for s in train_x)
dupes = sum(1 for s in test_x if norm(s) in train_norm)
print(f"test reviews also present in training: {dupes}")
print(f"duplicate reviews within training:     {len(train_x) - len(train_norm)}")
'''),
        md("""
IMDb is clean, because its authors deduplicated it — which also means its cost
cannot be measured here. So build a corpus that is **not** clean, and say so:
1,500 reviews of which a third were submitted twice. Everything else stays
correct; the vectoriser is fitted inside each split. The only difference is
whether the split keeps both copies of an entry on the same side.

⏱ **about a minute.**
"""),
        prompt(
            label="⏱ 1 min — rows split wrongly, against objects fitted wrongly",
            input="1,500 unique reviews, a third of them submitted twice, six seeds",
            output="the random-split and grouped-split accuracies, and how many test rows had a twin in training",
            constraint="everything else stays CORRECT — the vectoriser is fitted inside each split both times. The only difference is whether both copies of an entry land on the same side",
            check="assert no group straddles the grouped split",
            left_open="that nothing was fitted on the test set in either row. The whole difference is which rows the split happened to separate.",
            student="assuming a correct pipeline is sufficient. The duplicate leak here is LARGER than the vectoriser leak above — the object fitted wrongly cost less than the rows split wrongly.",
            catch="report what fraction of test rows had a copy of themselves in training. That number explains the gap and it is the one to look for in a real corpus."),
        code('''
from sklearn.model_selection import GroupShuffleSplit

def honest_score(X, y, tr, te):
    vec = TfidfVectorizer(min_df=1, ngram_range=(1, 2), max_features=60_000)
    Ztr = vec.fit_transform([X[i] for i in tr])           # training rows only
    clf = LogisticRegression(max_iter=2000, C=4.0).fit(Ztr, y[tr])
    return (clf.predict(vec.transform([X[i] for i in te])) == y[te]).mean()

N_UNIQUE, DUP_FRAC, SEEDS = 1_500, 0.3, 6
naive, grouped, twins = [], [], []
for s in range(SEEDS):
    r   = np.random.default_rng(RANDOM_STATE + 100 + s)
    idx = r.choice(len(all_x), size=N_UNIQUE, replace=False)
    X, y, g = [all_x[i] for i in idx], all_y[idx], np.arange(N_UNIQUE)

    dup = r.choice(N_UNIQUE, size=int(DUP_FRAC * N_UNIQUE), replace=False)
    X = X + [X[i] for i in dup]
    y = np.concatenate([y, y[dup]])
    g = np.concatenate([g, g[dup]])
    assert len(X) == len(y) == len(g)

    tr, te = train_test_split(np.arange(len(X)), test_size=0.25,
                              random_state=RANDOM_STATE + s, stratify=y)
    naive.append(honest_score(X, y, tr, te))
    twins.append(np.isin(g[te], g[tr]).mean())

    gss = GroupShuffleSplit(n_splits=1, test_size=0.25,
                            random_state=RANDOM_STATE + s)
    tr2, te2 = next(gss.split(np.arange(len(X)), y, groups=g))
    assert set(g[tr2]).isdisjoint(g[te2]), "a group straddles the grouped split"
    grouped.append(honest_score(X, y, tr2, te2))

naive, grouped = np.array(naive), np.array(grouped)
print(f"random split  {naive.mean():.4f} (sd {naive.std():.4f})")
print(f"grouped split {grouped.mean():.4f} (sd {grouped.std():.4f})")
print(f"the duplicate leak is worth {100 * (naive - grouped).mean():+.2f} points")
print(f"{np.mean(twins):.0%} of test rows had a copy of themselves in training")
'''),
        md("""
Nothing was fitted on the test set in either row. The whole difference is which
rows the split happened to separate — and it is larger than the vectoriser leak
above. **The object fitted wrongly cost less than the rows split wrongly.**
"""),

        md("""
## 11 · Red-team a peer's notebook

Swap with the team beside you. Ten minutes. Nine questions:

1. What touched the test set?
2. What was fitted, and on what? (`fit` and `transform` are different verbs)
3. What is the shape here?
4. What was dropped — rows, columns, NaNs? Count them.
5. What is the default I did not ask for?
6. Was the tokenizer or vectoriser fitted before the split?
7. Does `forward` return logits, or probabilities?
8. Is padding excluded from every pooling and every mean?
9. Are there duplicate documents across the split? Count them.

Report what you **found**, not what you would have done differently.
"""),
    ]
