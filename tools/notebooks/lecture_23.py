"""
Lecture 23 — One catalogue, two modalities.

Build. Chapters 15–16. The surprise is that two separately trained encoders
place an image and its own caption no closer than an image and a stranger's
caption, and the lecture turns on measuring that rather than asserting it.

Exports build() -> list[nbformat cell]. Self-contained: it assembles its own
catalogue and does not assume any previous lecture's kernel is alive.

Sizes are stated everywhere, because a recall without its candidate-set size is
not a measurement. The catalogue is 200 images. Full COCO is about 20 GB and
this notebook does not download it.
"""

from __future__ import annotations

import nbformat as nbf


def md(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(text.strip("\n"))


def code(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(text.strip("\n"))


HEADER = """
# One catalogue, two modalities

**Lecture 23 · Build** · Géron, Chapters 15–16

Applications of Machine Learning — BSc Mathematics of Artificial Intelligence

---

**How to use this notebook.** You are not expected to type the code. You are
expected to *read* it before you run it, and to be able to say what every line
does and what would break if it changed. Cells marked **⚠ read before running**
contain a defect on purpose.

**What it downloads.** A 5 MB index of the COCO validation split, then 200
individual images (about 30 MB), then three model checkpoints (about 1 GB in
total, cached after the first run). It does **not** download COCO, which is
about 20 GB.

**Expected wall clock on a Colab GPU runtime:** three to five minutes end to
end, most of it the first model download.
"""


def build() -> list:
    return [
        md(HEADER),

        md("## 1 · Setup"),
        code('''
# --- setup -------------------------------------------------------------------
# Not examinable: version hygiene. It is here because a mismatch produces a
# confusing error twenty cells later rather than here.
import ast, io, sys, time, urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import transformers
from PIL import Image

print(f"python        {sys.version.split()[0]}")
print(f"torch         {torch.__version__}")
print(f"transformers  {transformers.__version__}")

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
print(f"device        {device}")

N_CATALOGUE = 200          # every recall below is over this many candidates
'''),

        md("""
## 2 · The catalogue

Two downloads, and neither of them is COCO.

1. The **split index**: a CSV listing 5,000 COCO 2014 validation images with the
   five human-written captions each. About 5 MB.
2. The **images we actually use**: the first 200 by COCO id, fetched one at a
   time. About 30 MB.

Sorting by image id and taking the first 200 makes the catalogue deterministic —
two students with differently ordered frames get the same 200 images, for the
same reason `KFold(shuffle=True, random_state=42)` gives them the same folds.

**Expected wall clock: about one minute** the first time, instant afterwards.
"""),
        code('''
CACHE = Path("datasets/app12")
CACHE.mkdir(parents=True, exist_ok=True)

CSV_URL = ("https://huggingface.co/datasets/nlphuji/"
           "mscoco_2014_5k_test_image_text_retrieval/resolve/main/"
           "test_5k_mscoco_2014.csv")

csv_path = CACHE / "coco_karpathy_test.csv"
if not csv_path.is_file():
    urllib.request.urlretrieve(CSV_URL, csv_path)

split = pd.read_csv(csv_path).sort_values("cocoid").reset_index(drop=True)
print(f"the split index lists {len(split):,} images")

imgdir = CACHE / "images"
imgdir.mkdir(exist_ok=True)

catalogue = []
for _, row in split.head(N_CATALOGUE).iterrows():
    dest = imgdir / row["filename"]
    if not dest.is_file():
        urllib.request.urlretrieve(
            f"http://images.cocodataset.org/val2014/{row['filename']}", dest)
    captions = [" ".join(c.split()) for c in ast.literal_eval(row["raw"])]
    catalogue.append({"sku": f"CAT-{int(row['cocoid']):06d}",
                      "file": dest, "captions": captions})

# assert, do not hope
assert len(catalogue) == N_CATALOGUE, len(catalogue)
assert all(len(e["captions"]) >= 2 for e in catalogue), "an entry has < 2 captions"
assert len({e["sku"] for e in catalogue}) == N_CATALOGUE, "duplicate SKU"

megabytes = sum(e["file"].stat().st_size for e in catalogue) / 1e6
print(f"catalogue: {len(catalogue)} entries, {megabytes:.1f} MB on disk")
'''),

        md("""
### One entry

The five captions are **evidence**, not part of the product. We use caption #1 as
the entry's written description and caption #2 as the customer's query, so the
two sides are different people's sentences about the same picture. If we used
the same sentence on both sides a hash table would score 100% and the metric
would measure nothing.
"""),
        code('''
e = catalogue[0]
print(e["sku"])
for i, c in enumerate(e["captions"][:3], start=1):
    print(f"  caption {i}: {c}")

images       = [Image.open(x["file"]).convert("RGB") for x in catalogue]
descriptions = [x["captions"][0] for x in catalogue]
queries      = [x["captions"][1] for x in catalogue]

assert len(images) == len(descriptions) == len(queries) == N_CATALOGUE
assert descriptions[0] != queries[0], "query and description must differ"
'''),

        md("""
## 3 · The metric, and the trivial baseline

The system returns a ranking, so the quantity that matters is the **rank of the
relevant entry**. Recall@k is the share of queries whose answer landed in the
top *k*.

Ties count **against** the model: a tie is scored as the worse rank. A metric
that flatters a degenerate score matrix is not a metric, and you will see why
three cells from now.

The trivial baseline needs no experiment at all. One relevant entry among *n*,
ranked uniformly at random, gives `P(rank <= k) = k/n` exactly.
"""),
        code('''
def ranks_of_truth(sim):
    """sim[i, j] = score of query i against candidate j; truth is j == i."""
    assert sim.shape[0] == sim.shape[1], sim.shape
    correct = np.diag(sim)[:, None]
    return (sim >= correct).sum(axis=1)          # 1 = best; ties count against us


def report(name, sim):
    r = ranks_of_truth(sim)
    n = len(r)
    out = {f"R@{k}": (r <= k).mean() for k in (1, 5, 10)}
    print(f"{name:34s} " + "  ".join(f"{k} {v:6.1%}" for k, v in out.items())
          + f"   median rank {np.median(r):5.0f} of {n}")
    return out


n = N_CATALOGUE
print(f"random ranking over {n} candidates, computed exactly:")
for k in (1, 5, 10):
    print(f"  R@{k:<3d} {k / n:6.1%}")
print(f"  expected rank {(n + 1) / 2:.1f}")
print(f"  MRR {np.mean(1 / np.arange(1, n + 1)):.3f}")

# the tie rule, demonstrated: a model that has learned nothing
zeros = np.zeros((n, n))
strict = 1 + (zeros > np.diag(zeros)[:, None]).sum(axis=1)
print(f"\\nwith a strict '>' a constant scorer reports R@1 = {(strict == 1).mean():.0%}")
print(f"with '>=' it reports R@1 = {(ranks_of_truth(zeros) <= 1).mean():.1%}")
'''),

        md("""
### ✍️ Commit

Before running another cell, write down on paper:

* the metric, and the candidate-set size;
* the Recall@1 a good system should reach;
* the Recall@1 you expect from what we are about to build.

Shuffling scores 0.5% at rank 1 and 5.0% in the top ten over 200 candidates.
Your number belongs somewhere above those, and saying *where* is the exercise.
"""),

        md("""
## 4 · Encode the images

An image-only encoder: a vision transformer trained on ImageNet labels. No text
appears anywhere in its training.

**Expected wall clock: 20–60 s** for the model download, then about 15 s of
inference for 200 images.
"""),
        code('''
from transformers import AutoImageProcessor, ViTModel

vit_proc = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
vit = ViTModel.from_pretrained("google/vit-base-patch16-224",
                               add_pooling_layer=False).to(device).eval()
# add_pooling_layer=False because that checkpoint carries no pooler weights.
# Ask for one and you get a randomly initialised layer, with no error.

t0 = time.perf_counter()
chunks = []
with torch.no_grad():
    for i in range(0, len(images), 32):
        batch = vit_proc(images=images[i:i + 32], return_tensors="pt").to(device)
        chunks.append(vit(**batch).last_hidden_state[:, 0].cpu().numpy())

V = np.concatenate(chunks).astype(np.float64)
assert V.shape == (N_CATALOGUE, 768), V.shape
print(f"images encoded: {V.shape} in {time.perf_counter() - t0:.1f}s")
'''),

        md("""
## 5 · Encode the queries

A text-only encoder: mean-pooled sentence embeddings. No image appears anywhere
in its training. Note the attention mask in the pooling — forget it and you
average in the padding.
"""),
        code('''
from transformers import AutoModel, AutoTokenizer

TEXT_ID = "sentence-transformers/all-MiniLM-L6-v2"
tok = AutoTokenizer.from_pretrained(TEXT_ID)
enc = AutoModel.from_pretrained(TEXT_ID).to(device).eval()


def minilm(sentences, batch=64):
    out = []
    with torch.no_grad():
        for i in range(0, len(sentences), batch):
            b = tok(sentences[i:i + batch], return_tensors="pt", padding=True,
                    truncation=True, max_length=128).to(device)
            h = enc(**b).last_hidden_state
            m = b["attention_mask"].unsqueeze(-1).float()
            out.append(((h * m).sum(1) / m.sum(1)).cpu().numpy())
    return np.concatenate(out).astype(np.float64)


T = minilm(queries)
assert T.shape == (N_CATALOGUE, 384), T.shape
print(f"queries encoded: {T.shape}")
'''),

        md("""
## 6 · Now take the cosine

Reviewer question 3: **what is the shape here?**
"""),
        code('''
print(f"V {V.shape}   T {T.shape}")
try:
    V @ T.T
except ValueError as exc:
    print(f"\\nValueError: {exc}")
    print("\\nThere is no cosine between them, because there is no inner "
          "product between them.")
'''),

        md("""
### Force the dimensions to agree — three ways

So nobody can blame the fix:

1. a random Gaussian projection 768 → 384 (Johnson–Lindenstrauss, thread 5);
2. keep the first 384 coordinates of the image vector;
3. zero-pad the text vector to 768.

Thread 5 promised that a random projection nearly preserves the geometry *of the
image space*. It promised nothing about aligning that geometry with anything
else.
"""),
        code('''
def unit(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)


rng = np.random.default_rng(SEED)
R = rng.normal(0, 1 / np.sqrt(384), size=(768, 384))

Vn, Tn = unit(V), unit(T)
sims_two_spaces = {
    "JL projection 768->384": unit(Vn @ R) @ Tn.T,
    "truncate image to 384":  unit(Vn[:, :384]) @ Tn.T,
    "zero-pad text to 768":   Vn @ unit(np.pad(Tn, ((0, 0), (0, 768 - 384)))).T,
}
for name, S in sims_two_spaces.items():
    assert S.shape == (N_CATALOGUE, N_CATALOGUE), (name, S.shape)
    report(name, S.T)                      # transpose: rows are text queries
print(f"\\nrandom ranking over {n} candidates: R@1 {1 / n:.1%}  "
      f"R@5 {5 / n:.1%}  R@10 {10 / n:.1%}")
'''),

        md("""
## 7 · ⚠ Read before running — the assistant failure

**The prompt:** *"Encode the catalogue images with a ViT and the captions with a
sentence transformer, then check whether an image and its caption are similar."*

Under-specified in exactly one place. Find it before you run the cell.
"""),
        code('''
# --- what the weak prompt returns --------------------------------------------
Vp = unit(unit(V) @ R)
sim_matched = (Vp * unit(T)).sum(axis=1)      # cosine of each matched pair

print(f"mean similarity of an image and its caption: {sim_matched.mean():.3f}")
print(f"{(sim_matched > 0.05).mean():.0%} of pairs score above 0.05")
'''),

        md("""
**The review question:** *what does an unrelated pair score?*

The cell computed the diagonal of the similarity matrix and never touched the
other 200 × 200 − 200 = 39,800 entries. It reports a **level** where only a
**difference** means anything. In the five reviewer questions this is number 5 —
the default nobody asked for is the missing control group.
"""),
        code('''
S = sims_two_spaces["JL projection 768->384"]
off = ~np.eye(N_CATALOGUE, dtype=bool)
matched, unrelated = np.diag(S), S[off]

pooled = np.sqrt((matched.var(ddof=1) + unrelated.var(ddof=1)) / 2)
print(f"matched pairs    mean {matched.mean():+.4f}   sd {matched.std(ddof=1):.4f}"
      f"   (n = {len(matched)})")
print(f"unrelated pairs  mean {unrelated.mean():+.4f}   sd {unrelated.std(ddof=1):.4f}"
      f"   (n = {len(unrelated):,})")
print(f"difference       {matched.mean() - unrelated.mean():+.4f}"
      f"   Cohen's d {(matched.mean() - unrelated.mean()) / pooled:+.3f}")
'''),

        md("""
**The corrected specification:**

> Build the full 200 × 200 cosine matrix. Report the mean and sd of the diagonal
> **and** of the off-diagonal, their standardised difference, and Recall@1, 5
> and 10 for text-to-image ranking. Print the exact random-ranking values `k/n`
> beside them. Break ties against the model.

Every clause is there because leaving it out produced a number somebody
believed.

---

### Why it happened

Both encoders work. The ViT separates images from images; MiniLM separates
sentences from sentences. Neither ever saw a single (image, sentence) pair, so
neither has any reason to place them anywhere in particular relative to each
other.

Apply any rotation to the image space: every image–image cosine is unchanged, so
the ViT's loss is unchanged, and every image–text cosine changes. The loss
cannot tell those worlds apart, so it did not pick one.
"""),

        md("""
## 8 · A jointly trained pair

Same architecture family, one difference: both towers were trained by a single
objective that compared them. **Expected wall clock: 1–2 min** for the download
(about 600 MB), then a few seconds of inference.
"""),
        code('''
from transformers import CLIPModel, CLIPProcessor

CLIP_ID = "openai/clip-vit-base-patch32"
clip_proc = CLIPProcessor.from_pretrained(CLIP_ID)
clip = CLIPModel.from_pretrained(CLIP_ID).to(device).eval()

print(f"projection dim {clip.config.projection_dim}")
print(f"parameters     {sum(p.numel() for p in clip.parameters()) / 1e6:.0f}M")


def clip_images(imgs, batch=32):
    out = []
    with torch.no_grad():
        for i in range(0, len(imgs), batch):
            b = clip_proc(images=imgs[i:i + batch], return_tensors="pt").to(device)
            out.append(clip.get_image_features(**b).cpu().numpy())
    return np.concatenate(out).astype(np.float64)


def clip_text(sentences, batch=64):
    out = []
    with torch.no_grad():
        for i in range(0, len(sentences), batch):
            b = clip_proc(text=sentences[i:i + batch], return_tensors="pt",
                          padding=True, truncation=True, max_length=77).to(device)
            out.append(clip.get_text_features(**b).cpu().numpy())
    return np.concatenate(out).astype(np.float64)


I_raw = clip_images(images)
Q_raw = clip_text(queries)
assert I_raw.shape == Q_raw.shape == (N_CATALOGUE, clip.config.projection_dim)

I, Q = unit(I_raw), unit(Q_raw)      # onto the sphere — both sides, always
print(f"\\nimage features {I.shape}   text features {Q.shape}")
'''),

        md("""
`get_image_features` and `get_text_features` return vectors that are **not**
normalised. Skipping `unit()` still runs, still returns a matrix, and still
produces a ranking — a different one. That is the assistant failure of the next
lecture, and it costs more than you would guess.

## 9 · The known-answer test, before the real one

Step 4 of the working loop: *test against a case whose answer you know*. The
catalogue has no labels, so nothing in it can tell us the model is loaded
correctly and preprocessing its inputs the way it was trained to.

CIFAR-10 can. **Expected wall clock: 1–2 min** for the 170 MB download the first
time, then about 10 s of inference for 500 images.
"""),
        code('''
from torchvision.datasets import CIFAR10

cifar = CIFAR10(root="datasets", train=False, download=True)
N_CIFAR = 500                      # a subset, and we say so wherever we quote it
cifar_images = [cifar[i][0].convert("RGB") for i in range(N_CIFAR)]
y = np.array([cifar[i][1] for i in range(N_CIFAR)])
classes = list(cifar.classes)

assert len(cifar_images) == len(y) == N_CIFAR
assert len(classes) == 10, classes

F = unit(clip_images(cifar_images))
print(f"{N_CIFAR} CIFAR-10 images encoded: {F.shape}")
'''),

        code('''
for template in ["{}", "a photo of a {}", "a low-resolution photo of a {}"]:
    W = unit(clip_text([template.format(c) for c in classes]))
    pred = (F @ W.T).argmax(axis=1)
    print(f"{template.format('dog')!r:38s} accuracy {(pred == y).mean():6.1%}"
          f"   over {N_CIFAR} images   (chance 10.0%)")
'''),

        md("""
The classifier is ten sentences; there is no fitted parameter anywhere in that
cell. But notice the spread across templates. A "zero-shot" number is a number
for **one particular sentence**, and the sentence is a choice you made.

Choosing the best template by looking at the test accuracy is a hyperparameter
chosen on the test set — Lecture 6, in a new costume.

## 10 · Back to the catalogue

Same 200 images, same 200 queries, same metric, same tie rule, same baseline.
Only the encoders changed.
"""),
        code('''
sim_clip = Q @ I.T                        # rows: text queries, columns: images
assert sim_clip.shape == (N_CATALOGUE, N_CATALOGUE)

print(f"over {N_CATALOGUE} candidates, ties against the model:\\n")
report("ViT + MiniLM, JL projection",
       sims_two_spaces["JL projection 768->384"].T)
report("joint dual encoder, text->image", sim_clip)
report("joint dual encoder, image->text", I @ Q.T)
print(f"\\n{'random ranking (arithmetic)':34s} " +
      "  ".join(f"R@{k} {k / n:6.1%}" for k in (1, 5, 10)) +
      f"   expected rank {(n + 1) / 2:5.0f} of {n}")
'''),
        code('''
matched, unrelated = np.diag(sim_clip), sim_clip[~np.eye(N_CATALOGUE, dtype=bool)]
pooled = np.sqrt((matched.var(ddof=1) + unrelated.var(ddof=1)) / 2)
print(f"matched pairs    mean {matched.mean():+.4f}   sd {matched.std(ddof=1):.4f}")
print(f"unrelated pairs  mean {unrelated.mean():+.4f}   sd {unrelated.std(ddof=1):.4f}")
print(f"difference       {matched.mean() - unrelated.mean():+.4f}"
      f"   Cohen's d {(matched.mean() - unrelated.mean()) / pooled:.2f}")
'''),

        md("""
## 11 · The other route, and where it breaks

Some entries have a written description. For those, the previous application's
semantic search applies directly: embed the description, embed the query, rank by
cosine. Same space on both sides, because both sides are sentences.

Then model the catalogue as it really arrives: delete the description of 60 of
the 200 entries, by a fixed rule so the experiment repeats.
"""),
        code('''
D = unit(minilm(descriptions))
Qt = unit(minilm(queries))
sim_text = Qt @ D.T
report("text query -> text description", sim_text)

blanked = np.array([i for i in range(N_CATALOGUE) if i % 10 < 3])
assert len(blanked) == 60, len(blanked)

sim_missing = sim_text.copy()
sim_missing[:, blanked] = -np.inf          # not in the index at all
report("...with 60 descriptions deleted", sim_missing)

r_full    = ranks_of_truth(sim_text)
r_missing = ranks_of_truth(sim_missing)
print(f"\\nR@1 on those 60 entries: with a description "
      f"{(r_full[blanked] <= 1).mean():.1%}, with none "
      f"{(r_missing[blanked] <= 1).mean():.1%}")
print(f"R@1 on the same 60 via the joint image route: "
      f"{(ranks_of_truth(sim_clip)[blanked] <= 1).mean():.1%}")
'''),

        md("""
Not "worse" — **zero, structurally**. An entry with no text is not in a text
index. The image route is untouched, because it never read a description.

That zero is the number the next lecture repairs.

## 12 · Red-team

Swap notebooks with the team beside you. Fifteen minutes. Five questions:

1. What touched the test set?
2. What was fitted, and on what? (`fit` and `transform` are different verbs)
3. What is the shape here?
4. What was dropped — rows, columns, NaNs? Count them.
5. What is the default I did not ask for?

And six ways to leak in a *retrieval* evaluation specifically:

1. the query text is also the indexed description — you are measuring string
   equality;
2. the shortlist a query is scored against was built using that query;
3. the prompt template was chosen by looking at the number it produced;
4. recall reported without the candidate-set size;
5. ties broken in the model's favour (see section 3 — a constant scorer then
   reports 100%);
6. normalisation applied to one side only, so "cosine" is not a cosine.

Report what you **found**, not what you would have done differently.

---

### ✍️ Before you leave

Add to your sheet: the best Recall@1 you obtained, over how many candidates, and
one query the system got wrong together with your explanation of why.
"""),
    ]
