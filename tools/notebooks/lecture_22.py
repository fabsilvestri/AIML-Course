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
        code('''
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
        code('''
z = np.array([1000., 1001., 1002.], dtype=np.float32)

with np.errstate(over="ignore", invalid="ignore"):
    print("exp(z)                 ", np.exp(z))
    print("exp(z) / exp(z).sum()  ", np.exp(z) / np.exp(z).sum())

m = z.max()
print("with the max subtracted", np.exp(z - m) / np.exp(z - m).sum())
print(f"\\nexp overflows float32 past x = {np.log(np.finfo(np.float32).max):.2f}")
'''),
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
        code('''
rows = []
rng32 = np.random.default_rng(RANDOM_STATE)
for scale in (1, 3, 10, 30, 60, 88, 100, 200):
    z64 = rng32.normal(0, scale, size=(2_000, 10))
    z32 = z64.astype(np.float32)
    yy  = rng32.integers(0, 10, size=2_000)
    idx = np.arange(2_000)

    lse = lambda v: v.max(1) + np.log(np.exp(v - v.max(1, keepdims=True)).sum(1))
    ref = -(z64[idx, yy] - lse(z64))

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        e     = np.exp(z32)
        p32   = e / e.sum(1, keepdims=True)
        naive = -np.log(p32[idx, yy])
    stable = -(z32[idx, yy] - lse(z32))

    bad = ~np.isfinite(naive)
    rel = lambda a: (np.abs(a[~bad] - ref[~bad]) / np.abs(ref[~bad])).max() \\
        if (~bad).any() else np.nan
    rows.append((scale, bad.mean(), rel(naive.astype(np.float64)),
                 rel(stable.astype(np.float64))))
    print(f"sd {scale:>3}   non-finite {bad.mean():6.1%}   "
          f"naive {rows[-1][2]:.2e}   stable {rows[-1][3]:.2e}")
'''),
        code('''
s = np.array([r[0] for r in rows], dtype=float)
plt.figure(figsize=(9, 3))
plt.loglog(s, np.maximum([r[2] for r in rows], 1e-17), "o-",
           label="naive:  -log(softmax(z))")
plt.loglog(s, np.maximum([r[3] for r in rows], 1e-17), "s-",
           label="stable: -(z[y] - logsumexp z)")
plt.xlabel("standard deviation of the logits")
plt.ylabel("relative error against float64")
plt.legend(); plt.tight_layout(); plt.show()
'''),
        code('''
# one row, small enough to check by hand
z1 = np.array([100., 0., -100.], dtype=np.float32)
with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
    e = np.exp(z1)
    print(f"numerator   exp(-100) -> {e[2]}")
    print(f"denominator exp(z).sum() -> {e.sum()}")
    print(f"naive loss  -> {-np.log(e[2] / e.sum())}")

lse1   = lambda v: v.max() + np.log(np.exp(v - v.max()).sum())
stable = -(z1[2] - lse1(z1))
exact  = -(z1.astype(np.float64)[2] - lse1(z1.astype(np.float64)))
torch_ = float(nn.CrossEntropyLoss()(torch.tensor(z1)[None, :], torch.tensor([2])))
print(f"\\nstable      -> {stable:.4f}")
print(f"float64     -> {exact:.4f}")
print(f"PyTorch     -> {torch_:.4f}")
assert abs(torch_ - exact) < 1e-3, "the library is not doing the stable thing"
'''),

        md("""
## 5 · Cross-entropy and KL divergence

$$H(\\mathbf{y}, \\mathbf{p}) = H(\\mathbf{y})
  + D_{\\mathrm{KL}}(\\mathbf{y} \\Vert \\mathbf{p})$$

Add and subtract $\\sum_k y_k \\log y_k$; that is the whole proof. For a
**one-hot** target $H(\\mathbf{y}) = 0$, so minimising cross-entropy *is*
minimising the KL divergence to the label.
"""),
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
'''),
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
zero_shot = bert_accuracy(model, test_x[:N_SCORE], test_y[:N_SCORE])
print(f"pretrained body, random head, no training: {zero_shot:.1%}")
'''),
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
        code('''
ft_acc      = bert_accuracy(model, test_x[:N_SCORE], test_y[:N_SCORE])
Xt, Lt      = encode_words(test_x[:N_SCORE])
scratch_acc = accuracy(good, Xt, Lt, test_y[:N_SCORE])

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
