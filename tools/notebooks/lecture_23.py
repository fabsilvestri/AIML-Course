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
from _prompt import prompt                                # noqa: E402


def md(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(text.strip("\n"))


def code(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(text.strip("\n"))


HEADER = """
# Vision transformers and multimodal retrieval

**Lecture 23** · Géron, Chapters 15–16

Applications of Machine Learning — BSc Mathematics of Artificial Intelligence

---

**How to use this notebook.** Read before you run. Every code cell is preceded
by the specification that would produce it — input, output, constraint, check.

Cells marked **⚠** deliberately run code that is wrong, and say so in the
heading before you reach them. They are the failures this lecture is about;
each runs the broken version beside the correct one and prices the difference.

Runs on CPU. Nothing here needs an accelerator.

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
        prompt(
            label="setup",
            input="nothing",
            output="versions, seeds, device, and N_CATALOGUE",
            constraint="`N_CATALOGUE = 200` as a named constant — every recall in this notebook is over that many candidates, and a recall without its candidate-set size is not a number",
            check="the candidate-set size belongs in the same sentence as the recall, every time. Put it in a constant so the printout carries it.",
            **{"try": "set N_CATALOGUE = 50 and run the notebook through. "
                      "Every recall rises and not one model improved: R@10 "
                      "over 50 candidates is a different question from R@10 "
                      "over 200. That is the whole reason this is a named "
                      "constant rather than a literal."}),
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
        prompt(
            label="⏱ 1 min first time — the catalogue",
            input="the COCO split index and the first 200 images by id",
            output="200 catalogue entries with their captions and files",
            constraint="sort by `cocoid` and take the first 200 — two students with differently ordered frames then get the same 200 images, for the same reason a seeded KFold gives them the same folds",
            check="assert the count, that every entry has at least two captions, and that the SKUs are unique. Normalise the caption whitespace on the way in. COCO's captions carry stray newlines, and a query that differs from a description only in whitespace is a bug you will chase later.",
            **{"try": "sort by filename instead of cocoid and take the first "
                      "200. You get a different catalogue, every number in "
                      "the notebook moves, and nothing warns you. Then say "
                      "what would have to be true for the two catalogues to "
                      "be interchangeable."}),
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
        prompt(
            label="caption 1 is the description, caption 2 is the query",
            input="the catalogue entries",
            output="the images, the descriptions and the queries as three parallel lists",
            constraint="use DIFFERENT captions for the two sides — the five captions are evidence, not part of the product, and caption 1 as description with caption 2 as query means two different people's sentences about the same picture",
            check="assert the three lists are the same length AND that the first description differs from the first query. The assert that the two sides differ. One line, and it is the difference between measuring retrieval and measuring string equality.",
            **{"try": "set queries = descriptions. The assert on the last "
                      "line fires immediately. Comment it out and run Section "
                      "15: R@1 is 100%, because you are now measuring string "
                      "equality with extra steps and a hash table would win."}),
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
        prompt(
            label="the metric, and the exact baseline",
            input="a square similarity matrix whose truth is the diagonal",
            output="Recall@1, 5 and 10, the median rank, and the random-ranking values",
            constraint="ties count AGAINST the model — use `>=`, not `>` — and compute the random baseline ARITHMETICALLY as k/n rather than by simulation",
            check="assert the matrix is square before reading a diagonal out of it. A baseline you can compute in closed form beats a simulated one. One relevant entry among n, ranked uniformly, gives P(rank ≤ k) = k/n exactly — no seeds, no noise.",
            **{"try": "change the >= in ranks_of_truth to >. A constant "
                      "scorer now reports R@1 = 100%, as the cell already "
                      "prints. Then decide which way the rule should break "
                      "for two candidates that genuinely score the same, and "
                      "say why the pessimistic choice is the only defensible "
                      "default."}),
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
### The floor is now on the table

Shuffling scores 0.5% at rank 1 and 5.0% in the top ten over 200 candidates.
Every number in the rest of this notebook is read against those two, and a
method that does not clear them has not been shown to do anything.
"""),

        md("""
## 4 · Encode the images

An image-only encoder: a vision transformer trained on ImageNet labels. No text
appears anywhere in its training.

**Expected wall clock: 20–60 s** for the model download, then about 15 s of
inference for 200 images.
"""),
        prompt(
            label="⏱ 20-60 s — encode the images",
            input="the 200 catalogue images",
            output="768-dimensional CLS vectors",
            constraint="`add_pooling_layer=False` — that checkpoint carries no pooler weights, so asking for one gives you a RANDOMLY INITIALISED layer with no error and no warning",
            check="assert the shape is (200, 768). When a checkpoint warns that weights were newly initialised, read it. It is the single most-ignored message in the transformers library.",
            **{"try": "drop add_pooling_layer=False and read the warning "
                      "transformers prints. You now have a randomly "
                      "initialised pooler; use pooler_output instead of the "
                      "CLS token and watch the recalls. Nothing raises, and "
                      "that warning was the only signal you were ever going "
                      "to get."}),
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
        prompt(
            label="encode the queries",
            input="the 200 query sentences",
            output="384-dimensional mean-pooled embeddings",
            constraint="pool with the ATTENTION MASK — forget it and you average in the padding, and longer sentences are diluted more than short ones",
            check="assert the shape is (200, 384). Any mean over a padded sequence needs the mask. If your pooling line has no `attention_mask` in it, it is wrong.",
            **{"try": "delete the mask from the pooling and average over all "
                      "positions. Longer sentences are diluted more than "
                      "short ones, so the ranking now partly encodes sentence "
                      "length. How would you detect that from the similarity "
                      "matrix alone?"}),
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
        prompt(
            label="reviewer question 3 — what is the shape here",
            input="the two feature matrices",
            output="their shapes, and the ValueError from multiplying them",
            constraint="catch the error and PRINT it rather than letting the notebook stop — the exception is the content of the cell",
            check="768 and 384 is a loud failure. The quiet version is two encoders that happen to share a dimension, where the multiplication succeeds and means nothing.",
            **{"try": "truncate first and run V[:, :384] @ T.T. It succeeds, "
                      "returns a 200 by 200 matrix of entirely plausible "
                      "numbers, and means exactly as much as the ValueError "
                      "did. A loud failure is a gift; this is the quiet "
                      "version of the same mistake."}),
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
        prompt(
            label="force the dimensions to agree — three ways",
            input="the two feature matrices",
            output="Recall@1, 5, 10 for a JL projection, a truncation and a zero-pad",
            constraint="do it THREE ways, so nobody can blame the particular fix",
            check="assert each similarity matrix is square and 200 by 200 before reporting it. Transpose before reporting. `report` expects rows to be queries, and the matrices here are built image-major — a silent transpose measures image-to-text and calls it text-to-image.",
            **{"try": "delete the .T from one of the three report calls. The "
                      "recalls change, because you are now measuring image- "
                      "to-text and calling it text-to-image. On a square "
                      "matrix a transpose is silent, which is why the "
                      "constraint names it."}),
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
## 7 · A jointly trained pair, before the derivation

Section 6 showed what two separately trained encoders give you. The derivation
that follows is about what a *jointly* trained pair looks like, so we need one
first — same architecture family, one difference: both towers were trained by a
single objective that compared them. **Expected wall clock: 1–2 min** for the download
(about 600 MB), then a few seconds of inference.
"""),
        prompt(
            label="⏱ 1-2 min — a jointly trained pair",
            input="the same images and queries",
            output="image and text features in a shared 512-dimensional space",
            constraint="`unit()` BOTH sides — `get_image_features` and `get_text_features` return vectors that are NOT normalised",
            check="assert both matrices have the same shape and the model's own projection dimension. One difference from section 4: both towers were trained by a single objective that COMPARED them. Same architecture family, one difference, and it is the whole result.",
            **{"try": "skip the unit() on one side only, then re-run the "
                      "recall table in Section 14. The ranking changes, "
                      "because an unnormalised inner product mixes direction "
                      "with loudness. Which side matters more, and why does "
                      "that depend on the length ratio printed in Section 8?"}),
        code('''
from transformers import CLIPModel, CLIPProcessor

CLIP_ID = "openai/clip-vit-base-patch32"
clip_proc = CLIPProcessor.from_pretrained(CLIP_ID)
clip = CLIPModel.from_pretrained(CLIP_ID).to(device).eval()

print(f"projection dim {clip.config.projection_dim}")
print(f"parameters     {sum(p.numel() for p in clip.parameters()) / 1e6:.0f}M")


def features(out):
    """The projected embedding, whichever transformers version is installed.

    In transformers 4.x `get_image_features` returned the tensor itself; in 5.x
    it returns an output object whose `pooler_output` holds the same vector.
    Both are the PROJECTED and UNNORMALISED embedding, which is the whole
    subject of the section below -- so this helper must not normalise.
    """
    return out if torch.is_tensor(out) else out.pooler_output


def clip_images(imgs, batch=32):
    out = []
    with torch.no_grad():
        for i in range(0, len(imgs), batch):
            b = clip_proc(images=imgs[i:i + batch], return_tensors="pt").to(device)
            out.append(features(clip.get_image_features(**b)).cpu().numpy())
    return np.concatenate(out).astype(np.float64)


def clip_text(sentences, batch=64):
    out = []
    with torch.no_grad():
        for i in range(0, len(sentences), batch):
            b = clip_proc(text=sentences[i:i + batch], return_tensors="pt",
                          padding=True, truncation=True, max_length=77).to(device)
            out.append(features(clip.get_text_features(**b)).cpu().numpy())
    return np.concatenate(out).astype(np.float64)


I_raw = clip_images(images)
Q_raw = clip_text(queries)
assert I_raw.shape == Q_raw.shape == (N_CATALOGUE, clip.config.projection_dim)

I, Q = unit(I_raw), unit(Q_raw)      # onto the sphere — both sides, always
print(f"\\nimage features {I.shape}   text features {Q.shape}")
'''),

        md("""
## 8 · The derivation, part one — why the sphere

The unnormalised inner product factorises as
`<a, b> = ||a|| ||b|| cos(theta)`. Two quantities are mixed: **which direction**
the encoder chose, and **how loudly** it said it. Only the direction carries the
semantics; the length is whatever the last linear layer happened to scale to.
"""),
        prompt(
            label="why the sphere",
            input="the raw embedding lengths",
            output="their range, for images and for text",
            constraint="report the RATIO of longest to shortest, not just the extremes",
            check="four lines, and they are the setup for the entire assistant failure. Measure the thing you are about to claim matters.",
            **{"try": "sort the images by their embedding length and look at "
                      "the longest and the shortest few. Is there anything "
                      "visibly in common within either group? If the length "
                      "carried semantics there would be."}),
        code('''
norms = np.linalg.norm(I_raw, axis=1)
print(f"image embedding lengths: min {norms.min():.2f}   max {norms.max():.2f}"
      f"   ratio {norms.max() / norms.min():.2f}x")
print(f"text  embedding lengths: min {np.linalg.norm(Q_raw, axis=1).min():.2f}"
      f"   max {np.linalg.norm(Q_raw, axis=1).max():.2f}")
'''),

        md("""
## 9 · The derivation, part two — what should an *unrelated* pair score?

Most rooms vote for **−1**: opposite meaning, opposite vector. It is the wrong
answer, and the reason is the concentration result from Lecture 10.

For a fixed unit vector there is **exactly one** point at cosine −1. Asking 200
unrelated captions all to sit there is asking for a configuration that does not
exist. The right question is not "what is the most different?" but "what does a
pair with *no relationship at all* look like?" — and that is a question about a
random vector.

Fix `u = e1` by rotational symmetry and write `v = z / ||z||` with
`z ~ N(0, I_d)`. Then `u.v = z1 / ||z||`, whose expectation is zero by the
symmetry `v -> -v`, and whose second moment is `1/d` because the *d* coordinates
share the total equally. So the standard deviation is exactly `1 / sqrt(d)`.
"""),
        prompt(
            label="what should an unrelated pair score",
            input="random unit vectors at six dimensionalities",
            output="the measured sd of the cosine against 1/√d, and how often |cos| exceeds 0.5",
            constraint="sweep the DIMENSION — the effect is entirely about d, and a single dimensionality shows a number rather than a law",
            check="in high dimensions two unrelated things are ORTHOGONAL, not opposite. That is why the contrastive loss targets zero for a non-matching pair, and it is the concentration result from application 5 in a new costume.",
            **{"try": "add d = 1 and d = 3 to the sweep. At d = 1 a unit "
                      "vector is +1 or -1 and nothing else, so the cosine is "
                      "+/-1 and its sd is 1 — which is still exactly "
                      "1/sqrt(d). The law does not break at small d. The "
                      "intuition that unrelated means opposite is what only "
                      "holds there."}),
        code('''
rng = np.random.default_rng(SEED)

print(f"{'d':>6}  {'sd measured':>12}  {'1/sqrt(d)':>10}  {'|cos| > 0.5':>12}")
for dim in [2, 8, 32, 128, 512, 2048]:
    A = unit(rng.normal(size=(4000, dim)))
    B = unit(rng.normal(size=(4000, dim)))
    c = (A * B).sum(1)
    print(f"{dim:6d}  {c.std(ddof=1):12.4f}  {1 / np.sqrt(dim):10.4f}"
          f"  {(np.abs(c) > 0.5).mean():11.1%}")

D_DEMO = 512                      # the dimension the argument is made at
A = unit(rng.normal(size=(20000, D_DEMO)))
B = unit(rng.normal(size=(20000, D_DEMO)))
c = (A * B).sum(1)
print(f"\\nat d = {D_DEMO}, over 20,000 pairs:")
print(f"  mean          {c.mean():+.5f}")
print(f"  sd            {c.std(ddof=1):.4f}   (1/sqrt(d) = {1 / np.sqrt(D_DEMO):.4f})")
print(f"  most negative {c.min():+.3f}   — nothing is anywhere near -1")
'''),

        md("""
**In high dimensions two unrelated things are orthogonal, not opposite.** That
is why the contrastive loss targets zero for a non-matching pair, and it is the
Lecture 10 concentration result arriving in a new costume seven applications
later.

Now the same measurement on the trained embeddings themselves.
"""),
        prompt(
            label="the same measurement on the trained embeddings",
            input="the CLIP features",
            output="mean, sd and minimum for image-image, image-unrelated-caption and matched pairs, plus the distance between the two centroids",
            constraint="report the MINIMUM as well as the mean — the claim is that nothing is anywhere near −1, and only the minimum tests it",
            check="why the gap exists is an open research question, outside the book and not examinable. THAT it exists is measurable in four lines and is on the exam.",
            **{"try": "add the same constant vector to every image embedding "
                      "BEFORE the unit(). For a fixed query every image's "
                      "score moves by the same amount, so no ranking within a "
                      "row can change — and the modality gap does. Now add it "
                      "AFTER the unit() instead, and explain why the ranking "
                      "moves this time."}),
        code('''
off = ~np.eye(N_CATALOGUE, dtype=bool)
ii, it = I @ I.T, I @ Q.T

print(f"{'pairs':38s} {'mean':>8} {'sd':>8} {'min':>8}")
for name, v in [("two unrelated images", ii[off]),
                ("an image and an unrelated caption", it[off]),
                ("an image and its own caption", np.diag(it))]:
    print(f"{name:38s} {v.mean():+8.3f} {v.std(ddof=1):8.3f} {v.min():+8.3f}")

gap = np.linalg.norm(I.mean(0) - Q.mean(0))
print(f"\\ndistance between the two centroids (the modality gap): {gap:.3f}")
print(f"fraction of unrelated image/caption pairs below zero: "
      f"{(it[off] < 0).mean():.1%}")
'''),

        md("""
Unrelated pairs sit near zero and nowhere near −1, exactly as the geometry says.
But they are not *at* zero either: images occupy one region of the sphere and
captions another. Only the ranking within a row is trained, so adding a constant
offset to every image embedding changes no ranking and no loss — the objective
has no reason to remove the gap, and it does not.

**Consequence you can be caught by:** an absolute cosine threshold tuned on
image–image pairs is meaningless for image–text pairs. Why the gap exists is an
open research question, outside Chapters 1–16 and not examinable. That it exists
is measurable and is on the exam.

## 10 · The derivation, part three — the temperature

Thread 11 gave us the machinery. Row *i* of the similarity matrix is a *B*-class
problem whose correct answer is column *i*:

`L = -mean_i log( exp(S_ii / tau) / sum_j exp(S_ij / tau) )`

Everything is familiar except `tau`. Set it to 1 and watch.
"""),
        prompt(
            label="the temperature",
            input="the similarity matrix at six temperatures",
            output="the loss, p(correct), p(hardest wrong) and top-1 at each",
            constraint="print log B beside the table — a contrastive loss is measured against a batch-dependent ceiling and is meaningless without it",
            check="from thread 11, ∂L/∂S_ij = (p_ij − 1[j=i])/τ, so a small τ concentrates the push on the few hardest negatives. That is what the temperature is for.",
            **{"try": "add tau = 100 to the table. The loss climbs towards "
                      "log 200 and the top-1 column still does not move. Say "
                      "in one sentence why no temperature whatever can change "
                      "the accuracy of a fixed similarity matrix."}),
        code('''
def infonce(sim, tau):
    """Symmetric InfoNCE on a matrix of cosines. Returns a dict of diagnostics."""
    B = sim.shape[0]
    logits = sim / tau
    p = torch.softmax(torch.tensor(logits), dim=1).numpy()
    q = torch.softmax(torch.tensor(logits), dim=0).numpy()
    diag = np.arange(B)
    loss = 0.5 * (-np.log(p[diag, diag]).mean() - np.log(q[diag, diag]).mean())
    offd = p.copy()
    offd[diag, diag] = 0.0
    return {"loss": float(loss),
            "p_positive": float(p[diag, diag].mean()),
            "p_hardest_neg": float(offd.max(axis=1).mean()),
            "accuracy": float((logits.argmax(1) == diag).mean())}


sim = I @ Q.T
print(f"a scorer that knows nothing would sit at log {N_CATALOGUE} = "
      f"{np.log(N_CATALOGUE):.3f}\\n")
print(f"{'tau':>8} {'loss':>8} {'p(correct)':>12} {'p(hardest wrong)':>18} {'top-1':>8}")
for tau in [1.0, 0.3, 0.1, 0.03, 0.01, 0.003]:
    r = infonce(sim, tau)
    print(f"{tau:8.3f} {r['loss']:8.3f} {r['p_positive']:12.4f}"
          f" {r['p_hardest_neg']:18.4f} {r['accuracy']:8.1%}")
'''),

        md("""
Two things to read off that table.

* At `tau = 1` the logits are cosines, so the whole spread of a row is at most 2.
  `exp` of a range of 2 is a ratio of at most 7.4 across 200 competitors, the
  softmax is nearly uniform whatever the model says, and the loss sits near
  `log B`.
* **The top-1 column does not move.** `tau` cannot change which column is
  largest, so it cannot change the accuracy of a fixed model. What it changes is
  where the gradient goes: from thread 11, `dL/dS_ij = (p_ij - 1[j=i]) / tau`, so
  a small `tau` concentrates the push on the few hardest negatives.

The temperature is not a hyperparameter anybody tunes by hand. The model stores
`log(1/tau)` and learns it by gradient descent, clamped from above.
"""),
        prompt(
            label="the temperature the model learned",
            input="CLIP's own logit_scale parameter",
            output="1/τ, τ, and the loss at that temperature",
            constraint="read it OUT OF THE MODEL rather than choosing one — it is not a hyperparameter anybody tunes by hand",
            check="when a model has learned a hyperparameter, ask it. `clip.logit_scale` is one attribute access and it is the correct value by construction.",
            **{"try": "print clip.logit_scale itself rather than its "
                      "exponential, and compare it with the clamp CLIP "
                      "applies at log 100. The learned value sits at its own "
                      "ceiling. What does that say about the direction the "
                      "objective was still pushing when training stopped?"}),
        code('''
scale = clip.logit_scale.exp().item()
print(f"learned logit scale 1/tau = {scale:.2f}")
print(f"learned temperature   tau = {1 / scale:.5f}")
r = infonce(sim, 1 / scale)
print(f"\\nat the learned temperature: loss {r['loss']:.3f}   "
      f"p(correct) {r['p_positive']:.3f}   top-1 {r['accuracy']:.1%}")
'''),

        md("""
## 11 · The derivation, part four — the negatives, and the batch size

The loss needs, for each image, a set of captions it should *not* match. Nobody
labels those: they are the other members of the batch, free and correct with high
probability on a large corpus.

So **the batch is the label set**. The batch size is not a memory setting; it is
the number of classes in the problem you are solving, and the chance level is
`1/B`.
"""),
        prompt(
            label="the batch IS the label set",
            input="batches of 2, 8, 32, 128 and 200",
            output="top-1 and loss at each, beside chance 1/B and the ceiling log B",
            constraint="average over many random batches at each size — a single draw at B=2 is one coin flip",
            check="a contrastive loss value is not comparable across papers. It is measured against a batch-dependent ceiling of log B, and almost nobody states their B beside it.",
            **{"try": "add B = 1 to the table. The loss is exactly zero and "
                      "top-1 is 100%, because a one-class problem has no "
                      "wrong answer available. Every contrastive loss you "
                      "will ever read has a B behind it, and this row is the "
                      "limit that shows why quoting one without it is "
                      "meaningless."}),
        code('''
tau = 1 / scale
print(f"{'B':>5} {'top-1':>8} {'chance 1/B':>12} {'loss':>8} {'log B':>8}")
for B in [2, 8, 32, 128, N_CATALOGUE]:
    reps = 1 if B == N_CATALOGUE else 200
    accs, losses = [], []
    for _ in range(reps):
        idx = rng.choice(N_CATALOGUE, size=B, replace=False)
        r = infonce(I[idx] @ Q[idx].T, tau)
        accs.append(r["accuracy"])
        losses.append(r["loss"])
    print(f"{B:5d} {np.mean(accs):8.1%} {1 / B:12.1%} {np.mean(losses):8.3f}"
          f" {np.log(B):8.3f}")
'''),

        md("""
## 12 · ⚠ Read before running — the assistant failure

**The prompt:** *"Encode the catalogue images with a ViT and the captions with a
sentence transformer, then check whether an image and its caption are similar."*

Under-specified in exactly one place. Find it before you run the cell.
"""),
        prompt(
            label="⚠ what the weak prompt returns",
            input="'encode the images with a ViT and the captions with a sentence transformer, then check whether an image and its caption are similar'",
            output="the mean cosine of each matched pair, and the share above 0.05",
            constraint="compute only the DIAGONAL, as the prompt implies — this is the failure, not the fix",
            check="reviewer question 5, in an unusual form. The default nobody asked for here is the missing control group.",
            **{"try": "print the mean of the whole 200 by 200 matrix beside "
                      "the diagonal mean. How far apart are they? This cell "
                      "reports a level, and only that difference means "
                      "anything — which is exactly what the prompt forgot to "
                      "ask for."}),
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
        prompt(
            label="the control that was missing",
            input="the full 200 × 200 similarity matrix",
            output="the mean and sd of the diagonal and off-diagonal, and Cohen's d",
            constraint="report a STANDARDISED difference — the raw gap between two means is unreadable without the spread they are drawn from",
            check="`ddof=1` on both variances, and n printed beside each. 200 against 39,800 is a very asymmetric comparison and the reader should see it.",
            **{"try": "recompute d using the truncation from Section 6 "
                      "instead of the JL projection. It changes, and none of "
                      "the three fixes is more correct than the others. An "
                      "effect size near zero that survives all three is "
                      "stronger evidence than any single one of them."}),
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
`get_image_features` and `get_text_features` return vectors that are **not**
normalised. Skipping `unit()` still runs, still returns a matrix, and still
produces a ranking — a different one. That is the assistant failure of the next
lecture, and it costs more than you would guess.

## 13 · The known-answer test, before the real one

Step 4 of the working loop: *test against a case whose answer you know*. The
catalogue has no labels, so nothing in it can tell us the model is loaded
correctly and preprocessing its inputs the way it was trained to.

CIFAR-10 can. **Expected wall clock: 1–2 min** for the 170 MB download the first
time, then about 10 s of inference for 500 images.
"""),
        prompt(
            label="⏱ 1-2 min — a known-answer test first",
            input="500 CIFAR-10 test images",
            output="their CLIP features",
            constraint="use a dataset WITH LABELS — the catalogue has none, so nothing in it can tell us the model is loaded correctly and preprocessing its inputs the way it was trained to",
            check="assert the counts and that there are ten classes. Say `over 500 images` wherever you quote the accuracy. A subset is a subset even when the point of the cell is not the number.",
            **{"try": "try to run this known-answer test on the ViT and "
                      "MiniLM pair instead. You cannot: the ViT has no text "
                      "tower, so there are no ten sentences to compare "
                      "against. Which of the two systems can be tested at all "
                      "before it is deployed?"}),
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

        prompt(
            label="the classifier is ten sentences",
            input="three prompt templates",
            output="zero-shot accuracy under each, against chance",
            constraint="try SEVERAL templates and print all of them — a single template hides that the number depends on it",
            check="a 'zero-shot' number is a number for ONE PARTICULAR SENTENCE. The spread across templates is part of the result.",
            **{"try": "add three templates of your own and report all six. "
                      "Then pick the best one and work out what you would "
                      "have to do to be allowed to report its number. The "
                      "answer is a validation split, and CIFAR-10 ships with "
                      "one."}),
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

## 14 · Back to the catalogue

Same 200 images, same 200 queries, same metric, same tie rule, same baseline.
Only the encoders changed.
"""),
        prompt(
            label="back to the catalogue",
            input="the CLIP features",
            output="recall for the two-space route and both directions of the joint one, with the arithmetic baseline",
            constraint="same 200 images, same 200 queries, same metric, same tie rule, same baseline — ONLY the encoders changed",
            check="assert the similarity matrix is square. Print the random baseline in the same table, every time. It is the only thing that makes 34% legible.",
            **{"try": "report R@1 for the joint model over a random 50-image "
                      "subset of this catalogue as well. It rises sharply, "
                      "and the model is byte-for-byte identical. A recall is "
                      "a recall over a stated candidate set, or it is not a "
                      "measurement."}),
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
        prompt(
            label="the same control, on the joint model",
            input="the CLIP similarity matrix",
            output="matched against unrelated, and Cohen's d",
            constraint="run the IDENTICAL analysis as section 7 — the comparison is between two values of d, and it is only a comparison if the computation is the same",
            check="when you fix something, re-run the diagnostic that detected it, unchanged. A new diagnostic on the fixed version proves nothing about the old one.",
            **{"try": "run this same block on the two-space matrix and print "
                      "both effect sizes on one line. The same computation, "
                      "two encoders, two numbers: that pair is the entire "
                      "result of the lecture, and neither number means "
                      "anything without the other."}),
        code('''
matched, unrelated = np.diag(sim_clip), sim_clip[~np.eye(N_CATALOGUE, dtype=bool)]
pooled = np.sqrt((matched.var(ddof=1) + unrelated.var(ddof=1)) / 2)
print(f"matched pairs    mean {matched.mean():+.4f}   sd {matched.std(ddof=1):.4f}")
print(f"unrelated pairs  mean {unrelated.mean():+.4f}   sd {unrelated.std(ddof=1):.4f}")
print(f"difference       {matched.mean() - unrelated.mean():+.4f}"
      f"   Cohen's d {(matched.mean() - unrelated.mean()) / pooled:.2f}")
'''),

        md("""
## 15 · The other route, and where it breaks

Some entries have a written description. For those, the previous application's
semantic search applies directly: embed the description, embed the query, rank by
cosine. Same space on both sides, because both sides are sentences.

Then model the catalogue as it really arrives: delete the description of 60 of
the 200 entries, by a fixed rule so the experiment repeats.
"""),
        prompt(
            label="the other route, and where it breaks",
            input="the written descriptions, with 60 of them deleted by a fixed rule",
            output="recall with and without the descriptions, on the affected entries and via the image route",
            constraint="delete by a FIXED RULE (`i % 10 < 3`) so the experiment repeats, and set the deleted columns to −inf so they are not in the index at all",
            check="assert exactly 60 entries were blanked. The image route is untouched on exactly those 60 entries, because it never read a description. That contrast is the number the next lecture repairs.",
            **{"try": "change the deletion rule to i % 10 < 5, so half the "
                      "catalogue loses its description. The text route's R@1 "
                      "on the affected entries stays at exactly zero and the "
                      "overall recall falls further. A structural zero does "
                      "not get worse; more of the catalogue simply falls into "
                      "it."}),
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

## 16 · Where we are

Five questions to ask of any retrieval result:

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

Each has a specific answer in this lecture, and each failure is silent.

---

### ✍️ Before you leave

Add to your sheet: the best Recall@1 you obtained, over how many candidates, and
one query the system got wrong together with your explanation of why.
""")]
