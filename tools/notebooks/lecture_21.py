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


def md(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(text.strip("\n"))


def code(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(text.strip("\n"))


HEADER = """
# Reading the customers

**Lecture 21 · Build** · Géron, Chapter 14

Applications of Machine Learning — BSc Mathematics of Artificial Intelligence

---

**How to use this notebook.** You are not expected to type the code. You are
expected to *read* it before you run it, and to be able to say what every line
does and what would break if it changed. Cells marked **⚠ read before running**
contain a defect on purpose.

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
        code('''
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
        code('''
majority = max(test_y.mean(), 1 - test_y.mean())
print(f"always predict one class:  {majority:.1%}")
'''),
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
## 5 · Commit

**Stop. On paper, now.** Not in this notebook — on paper, where you cannot
quietly revise it.

```
Metric:                                            ____________
Accuracy the desk would need to trust it:        ____________ %
Accuracy I expect from the model I build today:  ____________ %
Best accuracy I actually obtain:                 ____________ %
```

You have two anchors: the trivial one and the bag of words. Estimate against the
second, not the first.
"""),

        md("""
## 6 · Tokenisation, and why a word vocabulary breaks

Build the vocabulary from the **fit split only**. Building a vocabulary is
*fitting*, and it obeys the same rule as every other fitted object in this
course.
"""),
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
        code('''
# How often does WordPiece have to give up? Measure rather than assume.
sample = test_x[:2_000]
n_unk = n_tok = 0
for s in sample:
    ids = tk(s, truncation=True, max_length=512)["input_ids"]
    n_unk += sum(1 for i in ids if i == tk.unk_token_id)
    n_tok += len(ids)
print(f"[UNK] rate over {n_tok:,} subword tokens: {n_unk / n_tok:.4%}")
assert n_unk / n_tok < 0.001, "a subword tokenizer should almost never miss"
'''),

        md("""
## 7 · From integers to vectors

Token 4,271 is not four thousand of anything — the integer is a name. An
embedding is a learned table with one row per token, and looking up row *i* is
exactly multiplying a one-hot vector by that table, without ever forming it.

`padding_idx=0` pins row 0 at zero and keeps it there. Padding must not learn
anything.
"""),
        code('''
emb = nn.Embedding(num_embeddings=VOCAB, embedding_dim=128, padding_idx=0)
x   = torch.randint(0, VOCAB, (32, MAXLEN))

assert emb(x).shape == (32, MAXLEN, 128)
assert torch.equal(emb.weight[0], torch.zeros(128)), "padding row is not zero"
print(f"embedding table: {emb.weight.numel():,} parameters")
print(f"one batch: {tuple(x.shape)} -> {tuple(emb(x).shape)}")
'''),

        md("""
## 8 · Padding, and the length you must keep

A batch is a rectangle; reviews are not. Pad to `MAXLEN`, truncate what is
longer, and **keep the true lengths**. A padded batch has lost the information
about where each review ends.
"""),
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
## 9 · The classifier

Thirteen lines, and eleven of them are Lecture 12. The two new ones are the
embedding and the packing.
"""),
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
    net = net.to(device)
    params = [p for p in net.parameters() if p.requires_grad]
    opt    = torch.optim.Adam(params, lr=LR)
    lossf  = nn.CrossEntropyLoss()
    Xf_d, yf_d = Xf.to(device), torch.from_numpy(yf).to(device)
    curve, losses = [], []
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
            running += float(loss) * len(j)
        losses.append(running / len(Xf_d))
        curve.append(accuracy(net, Xv, Lv, yv))
        print(f"  {tag} epoch {ep + 1}: loss {losses[-1]:.4f}  "
              f"val {curve[-1]:.4f}  ({time.perf_counter() - t0:.0f}s)")
    return net, curve, losses
'''),
        code('''
torch.manual_seed(RANDOM_STATE)
scratch, curve_scratch, loss_scratch = train(
    GRUClassifier(VOCAB), Xf_w, Lf_w, fit_y, Xv_w, Lv_w, val_y,
    tag="our words / random")
'''),

        md("""
## 10 · ⚠ Read before running — an assistant writes the same model

> *"Write a PyTorch model that classifies a padded batch of token ids with an
> embedding and a GRU, and returns two logits."*

Under-specified in exactly one place. The code it returns is the
`last_of_padding=True` branch above: it summarises at `out[:, -1, :]`, the last
position of the **padded** batch.

**Reviewer question 3: what is the shape here?** Position `MAXLEN - 1` is
padding for every review shorter than `MAXLEN` — which is most of them.
"""),
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
        code('''
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
## 11 · The swap: change the table, and nothing else

The diagnosis is not the architecture. It is that the embedding table is being
asked to learn English from 5,000 sentiment labels.

Two changes, applied **one at a time**, so the measurement says which one did
the work.
"""),
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
Usually **worse**. Solving the out-of-vocabulary problem on its own made the
model worse, and that is a measurement worth keeping:

* sequences are longer in pieces, so the same budget of positions holds fewer
  words;
* `un`, `##watch`, `##able` are three random vectors, and the model has to learn
  that their *composition* is negative;
* half again as many rows to train, from the same labels.

A subword tokenizer is not a better tokenizer by itself. It is a vocabulary that
someone else's pretraining can be poured into.
"""),
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
## 12 · The test set. Once.

Everything above used the fit and validation splits only. This is the first and
last time the test half is scored.

⏱ **about a minute** — encoding 25,000 reviews twice and two forward passes.
"""),
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
## 13 · Where we are

Write your **best accuracy** on the same sheet of paper, next to what you
predicted. Bring it to the next lecture — we open by comparing them.

Four questions we did not answer, and all four are the next lecture:

1. What exactly does `CrossEntropyLoss` receive, and why does it want the raw
   two numbers rather than probabilities?
2. We borrowed a tokenizer and a table. What if we borrowed the whole model?
3. The desk wants complaints *grouped*, not only flagged.
4. Is the last number good? Compared with what ceiling?

Do not fix anything yet.
"""),
    ]
