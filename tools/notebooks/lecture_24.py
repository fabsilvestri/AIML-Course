"""
Lecture 24 — Closing the loop, and closing the course.

Fix. Chapters 15–16. Thread 12: the contrastive objective and its temperature.

Exports build() -> list[nbformat cell]. Self-contained: it rebuilds the 200-entry
catalogue and re-encodes it rather than assuming Lecture 23's kernel is alive. A
notebook that only runs because a previous one left variables in memory is not
reproducible.

Two things are deliberately smaller here than on the slides, and both are stated
in the notebook where they are used: the retrieval-augmented section uses six of
the twelve ambiguous queries, and the captioner runs on the same sixty entries
but with a shorter beam. The deck quotes the full run.
"""

from __future__ import annotations

import nbformat as nbf
from _prompt import prompt                                # noqa: E402


def md(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(text.strip("\n"))


def code(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(text.strip("\n"))


HEADER = """
# Closing the loop, and closing the course

**Lecture 24 · Fix** · Géron, Chapters 15–16 · *Mathematical thread: the
contrastive objective and its temperature*

Applications of Machine Learning — BSc Mathematics of Artificial Intelligence

---

**How to use this notebook.** Read before you run. The cell marked
**⚠ read before running** contains the defect this lecture is about, and it does
not raise an exception.

**What it downloads.** The same 200-entry catalogue as the previous lecture
(cached), plus two more checkpoints: a captioner (about 1 GB) and a small
instruction-tuned language model (about 1 GB). It does **not** download COCO.

**Expected wall clock on a Colab GPU runtime:** five to eight minutes end to
end.
"""


def build() -> list:
    return [
        md(HEADER),

        md("## 1 · Setup, and the catalogue again"),
        prompt(
            label="setup",
            input="nothing",
            output="versions, seeds, device, N_CATALOGUE",
            constraint="the same constants as the previous lecture, so the numbers are comparable",
            check="`torch.nn.functional as Fn` rather than `F` — `F` is already the feature matrix in the previous notebook's namespace, and a collision there is the kind of bug that produces a confident wrong number."),
        code('''
# --- setup -------------------------------------------------------------------
import ast, re, sys, time, urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as Fn
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

N_CATALOGUE = 200
'''),

        md("""
The same deterministic catalogue: the split index sorted by COCO id, first 200
entries. Cached from the previous lecture if you ran it in this runtime.
**Expected wall clock: under a minute** if the cache is cold, instant otherwise.
"""),
        prompt(
            label="the same catalogue, rebuilt",
            input="the split index and the first 200 images by id",
            output="the identical 200 entries, with descriptions and queries",
            constraint="the same deterministic rule — sorted by cocoid, first 200 — so this notebook and the last are talking about the same corpus",
            check="assert the count and that the SKUs are unique. Whitespace-normalise the captions on the way in, exactly as before. Two notebooks that normalise differently have different queries."),
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

assert len(catalogue) == N_CATALOGUE, len(catalogue)
assert len({e["sku"] for e in catalogue}) == N_CATALOGUE, "duplicate SKU"

images       = [Image.open(e["file"]).convert("RGB") for e in catalogue]
descriptions = [e["captions"][0] for e in catalogue]
queries      = [e["captions"][1] for e in catalogue]
valid_skus   = {e["sku"] for e in catalogue}
print(f"catalogue: {len(catalogue)} entries")
'''),

        prompt(
            label="the embeddings, and the raw ones kept",
            input="the images and queries",
            output="both the RAW features and the unit-normalised ones",
            constraint="keep `I_raw` and `Q_raw` deliberately — the assistant failure in section 6 is about what happens when you use them, and it cannot be demonstrated if they were normalised in place",
            check="assert both normalised matrices really have unit rows. `np.allclose(norm, 1.0)` as an assert, not a print. It is two lines, and section 6 is entirely about the run where it would have fired."),
        code('''
from transformers import CLIPModel, CLIPProcessor

CLIP_ID = "openai/clip-vit-base-patch32"
clip_proc = CLIPProcessor.from_pretrained(CLIP_ID)
clip = CLIPModel.from_pretrained(CLIP_ID).to(device).eval()


def unit(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)


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


I_raw = clip_images(images)          # NOT normalised — deliberately
Q_raw = clip_text(queries)
I, Q = unit(I_raw), unit(Q_raw)
d = I.shape[1]

assert np.allclose(np.linalg.norm(I, axis=1), 1.0), "images are not unit vectors"
assert np.allclose(np.linalg.norm(Q, axis=1), 1.0), "texts are not unit vectors"
print(f"embeddings {I.shape}, shared dimension d = {d}")
'''),

        md("""
## 2 · Thread 12, part one — why the sphere

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
            check="four lines, and they are the setup for the entire assistant failure. Measure the thing you are about to claim matters."),
        code('''
norms = np.linalg.norm(I_raw, axis=1)
print(f"image embedding lengths: min {norms.min():.2f}   max {norms.max():.2f}"
      f"   ratio {norms.max() / norms.min():.2f}x")
print(f"text  embedding lengths: min {np.linalg.norm(Q_raw, axis=1).min():.2f}"
      f"   max {np.linalg.norm(Q_raw, axis=1).max():.2f}")
'''),

        md("""
## 3 · Thread 12, part two — what should an *unrelated* pair score?

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
            check="in high dimensions two unrelated things are ORTHOGONAL, not opposite. That is why the contrastive loss targets zero for a non-matching pair, and it is the concentration result from application 5 in a new costume."),
        code('''
rng = np.random.default_rng(SEED)

print(f"{'d':>6}  {'sd measured':>12}  {'1/sqrt(d)':>10}  {'|cos| > 0.5':>12}")
for dim in [2, 8, 32, 128, 512, 2048]:
    A = unit(rng.normal(size=(4000, dim)))
    B = unit(rng.normal(size=(4000, dim)))
    c = (A * B).sum(1)
    print(f"{dim:6d}  {c.std(ddof=1):12.4f}  {1 / np.sqrt(dim):10.4f}"
          f"  {(np.abs(c) > 0.5).mean():11.1%}")

A = unit(rng.normal(size=(20000, d)))
B = unit(rng.normal(size=(20000, d)))
c = (A * B).sum(1)
print(f"\\nat d = {d}, over 20,000 pairs:")
print(f"  mean          {c.mean():+.5f}")
print(f"  sd            {c.std(ddof=1):.4f}   (1/sqrt(d) = {1 / np.sqrt(d):.4f})")
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
            check="why the gap exists is an open research question, outside the book and not examinable. THAT it exists is measurable in four lines and is on the exam."),
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

## 4 · Thread 12, part three — the temperature

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
            check="from thread 11, ∂L/∂S_ij = (p_ij − 1[j=i])/τ, so a small τ concentrates the push on the few hardest negatives. That is what the temperature is for."),
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
            check="when a model has learned a hyperparameter, ask it. `clip.logit_scale` is one attribute access and it is the correct value by construction."),
        code('''
scale = clip.logit_scale.exp().item()
print(f"learned logit scale 1/tau = {scale:.2f}")
print(f"learned temperature   tau = {1 / scale:.5f}")
r = infonce(sim, 1 / scale)
print(f"\\nat the learned temperature: loss {r['loss']:.3f}   "
      f"p(correct) {r['p_positive']:.3f}   top-1 {r['accuracy']:.1%}")
'''),

        md("""
## 5 · Thread 12, part four — the negatives, and the batch size

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
            check="a contrastive loss value is not comparable across papers. It is measured against a batch-dependent ceiling of log B, and almost nobody states their B beside it."),
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
Accuracy falls with *B* and chance falls faster, so the gap — the learning signal
— widens. That is the whole argument for a large batch, and it has two
consequences worth carrying away:

1. **A contrastive loss value is not comparable across papers.** It is measured
   against a batch-dependent ceiling of `log B`.
2. **Doubling the batch changes the task**, not just the gradient noise. Which is
   why these models are trained with batches in the tens of thousands, across
   many devices. That engineering is outside Chapters 1–16.

## 6 · ⚠ Read before running — the assistant failure

**The prompt:** *"Write the symmetric contrastive loss for a batch of image and
text embeddings, with a temperature of 0.01."*

One clause missing — the clause section 2 spent five minutes on.
"""),
        prompt(
            label="⚠ what the assistant returns",
            input="'write the symmetric contrastive loss for a batch of image and text embeddings, with a temperature of 0.01'",
            output="the loss, computed on the RAW features",
            constraint="run it exactly as returned. It runs, the docstring is accurate, and the shapes, the target and the factor of one half are all right",
            check="reviewer question 5 again. The default nobody asked for is that a function called `get_*_features` returns something unnormalised."),
        code('''
# --- what the weak prompt returns --------------------------------------------
def contrastive_loss(img, txt, tau=0.01):
    """Symmetric InfoNCE over a batch of paired embeddings."""
    logits = img @ txt.T / tau                 # (B, B)
    target = torch.arange(len(img), device=img.device)
    return 0.5 * (Fn.cross_entropy(logits, target) +
                  Fn.cross_entropy(logits.T, target))


img_t = torch.tensor(I_raw, dtype=torch.float32)     # as get_*_features returns
txt_t = torch.tensor(Q_raw, dtype=torch.float32)
print(f"loss on raw features: {contrastive_loss(img_t, txt_t, tau).item():.3f}")
'''),

        md("""
It runs. The docstring is accurate. The shapes, the target and the factor of one
half are all right.

**The review question:** *is `img @ txt.T` a cosine?* Only if both sides are unit
vectors — and `get_image_features` does not return unit vectors. So the entries
are `||a|| ||b|| cos(theta)`, the temperature is dividing a quantity with no
fixed scale, and the row-wise softmax compares lengths as much as directions.
Reviewer question 5 again: the default nobody asked for.
"""),
        prompt(
            label="what it costs, and how",
            input="the loss and accuracy both ways",
            output="both, plus the rank correlation between embedding length and queries won",
            constraint="show the MECHANISM, not just the loss difference — count how many queries each image wins and correlate that with its length",
            check="the corrected specification's assertion is the part that would have caught it in silence: assert every row of both matrices has unit norm to within 1e-5. Two lines, and the bug becomes a crash."),
        code('''
good = infonce(I @ Q.T, tau)
bad  = infonce(I_raw @ Q_raw.T, tau)
print(f"{'':22s} {'loss':>10} {'top-1':>8}")
print(f"{'on the unit sphere':22s} {good['loss']:10.3f} {good['accuracy']:8.1%}")
print(f"{'raw dot products':22s} {bad['loss']:10.3f} {bad['accuracy']:8.1%}")

wins = np.bincount((I_raw @ Q_raw.T).argmax(0), minlength=N_CATALOGUE)
order = np.argsort(np.argsort(norms))
corr = np.corrcoef(order, np.argsort(np.argsort(wins)))[0, 1]
print(f"\\nrank correlation between embedding length and queries won: {corr:+.2f}")
print(f"one image took {wins.max()} of {N_CATALOGUE} queries; "
      f"{(wins == 0).sum()} images were never ranked first")
'''),

        md("""
**The corrected specification:**

> Symmetric InfoNCE. **L2-normalise both embedding sets along the feature axis
> before the matrix product**, so the logits are cosines divided by `tau`. Assert
> that every row of both matrices has unit norm to within 1e-5. Report the loss
> *and* the in-batch top-1 accuracy, and print `log B` beside the loss.

The assertion is the part that would have caught it in silence. Two lines, and
the bug becomes a crash.

## 7 · Repair 1 — entries with no description

Sixty of the two hundred entries have no description, so they score exactly zero
on the text route. Use a model built for *generating* text from an image and
write the missing ones.

**Expected wall clock: 1–3 min**, most of it the 1 GB download.
"""),
        prompt(
            label="rebuild the text route and its hole",
            input="the descriptions and queries through MiniLM",
            output="R@1 on the 60 blanked entries, with and without their descriptions",
            constraint="the same fixed blanking rule as the previous lecture, and the same −inf convention for 'not in the index at all'",
            check="assert exactly 60 entries were blanked. Reproduce the fault before repairing it, from the same rule and the same seed. A repair measured against a differently-broken baseline is not measured."),
        code('''
from transformers import AutoTokenizer, BlipForConditionalGeneration, BlipProcessor
from transformers import AutoModel

TEXT_ID = "sentence-transformers/all-MiniLM-L6-v2"
mtok = AutoTokenizer.from_pretrained(TEXT_ID)
menc = AutoModel.from_pretrained(TEXT_ID).to(device).eval()


def minilm(sentences, batch=64):
    out = []
    with torch.no_grad():
        for i in range(0, len(sentences), batch):
            b = mtok(sentences[i:i + batch], return_tensors="pt", padding=True,
                     truncation=True, max_length=128).to(device)
            h = menc(**b).last_hidden_state
            m = b["attention_mask"].unsqueeze(-1).float()
            out.append(((h * m).sum(1) / m.sum(1)).cpu().numpy())
    return np.concatenate(out).astype(np.float64)


def ranks_of_truth(sim):
    return (sim >= np.diag(sim)[:, None]).sum(axis=1)


D = unit(minilm(descriptions))
Qt = unit(minilm(queries))
blanked = np.array([i for i in range(N_CATALOGUE) if i % 10 < 3])
assert len(blanked) == 60, len(blanked)

sim_full = Qt @ D.T
sim_missing = sim_full.copy()
sim_missing[:, blanked] = -np.inf

r_full, r_missing = ranks_of_truth(sim_full), ranks_of_truth(sim_missing)
print(f"R@1 on the 60 blanked entries — described: "
      f"{(r_full[blanked] <= 1).mean():.1%},  deleted: "
      f"{(r_missing[blanked] <= 1).mean():.1%}")
'''),

        prompt(
            label="⏱ 1-3 min — write the missing descriptions",
            input="the 60 images with no description",
            output="a generated caption for each, with four shown beside their human captions",
            constraint="generate for the BLANKED entries only — captioning all 200 would replace descriptions that already exist and confound the measurement",
            check="assert one caption came back per blanked entry. Print human and generated side by side for a few. It is the only way to see that the generated ones are shorter, blander and occasionally wrong."),
        code('''
proc = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
cap = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base").to(device).eval()

t0 = time.perf_counter()
generated = []
with torch.no_grad():
    for i in range(0, len(blanked), 8):
        chunk = blanked[i:i + 8]
        b = proc(images=[images[j] for j in chunk], return_tensors="pt").to(device)
        ids = cap.generate(**b, max_new_tokens=30, num_beams=3)
        generated += [" ".join(t.split())
                      for t in proc.batch_decode(ids, skip_special_tokens=True)]

assert len(generated) == len(blanked), (len(generated), len(blanked))
print(f"{len(generated)} captions written in {time.perf_counter() - t0:.0f}s\\n")
for j, g in list(zip(blanked, generated))[:4]:
    print(f"human    : {descriptions[j]}")
    print(f"generated: {g}\\n")
'''),

        prompt(
            label="what the repair recovers",
            input="the description matrix with the generated captions filled in",
            output="R@1 on the 60, four ways, and the overall text-route figure",
            constraint="measure on the SAME 60 entries throughout — the overall number is diluted by the 140 that never changed",
            check="three things not to claim: a generated caption is not evidence about the product; 60 entries is a small sample with a wide interval; and we are scoring generated captions against human captions of the SAME image, which is a friendly test. The failure mode to watch for is an auto-caption that is WRONG, making an entry findable under the wrong query — worse than unfindable, and nothing measured here detects it."),
        code('''
D_filled = D.copy()
D_filled[blanked] = unit(minilm(generated))
r_filled = ranks_of_truth(Qt @ D_filled.T)

sim_clip = Q @ I.T
r_clip = ranks_of_truth(sim_clip)

print(f"R@1, measured on the same 60 entries:")
print(f"  human description present   {(r_full[blanked] <= 1).mean():6.1%}")
print(f"  description deleted         {(r_missing[blanked] <= 1).mean():6.1%}")
print(f"  auto-caption                {(r_filled[blanked] <= 1).mean():6.1%}")
print(f"  joint image route (unused)  {(r_clip[blanked] <= 1).mean():6.1%}")
print(f"\\nover all {N_CATALOGUE} queries, text route R@1: "
      f"{(r_full <= 1).mean():.1%} -> {(r_missing <= 1).mean():.1%} "
      f"-> {(r_filled <= 1).mean():.1%}")
'''),

        md("""
A generated caption recovers **part** of the loss, not all of it. Report the
part, not the direction.

Three things not to claim: a generated caption is not evidence about the product;
sixty entries is a small sample with a wide interval around it; and we are
scoring generated captions against human captions of the same image, which is a
friendly test. The failure mode to watch for is an auto-caption that is *wrong*,
making an entry findable under the wrong query — worse than unfindable, and
nothing measured here detects it.

## 8 · Repair 2 — queries with no single right answer

Customers do not type captions. They type "something to sit on". A ranked list of
pictures is a poor answer, because the customer wants a shortlist **with
reasons**, and a ranking has no place to put a reason.

First decide how to check it. "The answer is good" is not measurable in a
lecture, so require the assistant to cite stock numbers: a SKU either exists in
the catalogue or it does not. That measures **grounding**, not helpfulness — say
which one you measured.

**Expected wall clock: 2–4 min**, most of it the 1 GB download.
"""),
        prompt(
            label="⏱ 2-4 min — queries with no single right answer",
            input="six deliberately ambiguous queries",
            output="the model, the prompt helper, and the top-5 shortlist per query",
            constraint="decide how to CHECK it before generating anything — 'the answer is good' is not measurable in a lecture, so require cited SKUs, which either exist in the catalogue or do not",
            check="assert the shortlist has the shape you expect. `do_sample=False`. A sampled generation gives a different answer every run and the grounding rate becomes a random variable you have not characterised."),
        code('''
from transformers import AutoModelForCausalLM

LLM_ID = "Qwen/Qwen2.5-0.5B-Instruct"
ltok = AutoTokenizer.from_pretrained(LLM_ID)
llm = AutoModelForCausalLM.from_pretrained(LLM_ID).to(device).eval()


def ask(prompt, max_new_tokens=140):
    msgs = [{"role": "system",
             "content": "You are a product-catalogue assistant. Cite catalogue "
                        "SKUs, which always look like CAT-123456."},
            {"role": "user", "content": prompt}]
    text = ltok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    b = ltok([text], return_tensors="pt").to(device)
    with torch.no_grad():
        out = llm.generate(**b, max_new_tokens=max_new_tokens, do_sample=False,
                           pad_token_id=ltok.eos_token_id)
    return ltok.decode(out[0][b["input_ids"].shape[1]:], skip_special_tokens=True)


# six of the twelve queries on the slides, so this cell finishes in the lecture
AMBIGUOUS = ["something to sit on",
             "somewhere to eat outdoors",
             "a way to get across town without a car",
             "gear for bad weather",
             "something to put flowers in",
             "a machine that heats food"]

SKU_RE = re.compile(r"CAT-\\d{6}")
q_emb = unit(clip_text(AMBIGUOUS))
top5 = np.argsort(-(q_emb @ I.T), axis=1)[:, :5]
assert top5.shape == (len(AMBIGUOUS), 5)
'''),

        prompt(
            label="closed book against retrieval-augmented",
            input="the same six queries, asked both ways",
            output="how many cited SKUs actually exist, under each condition",
            constraint="identical prompts except for the shortlist — the only difference must be whether the model was given the catalogue",
            check="the retriever is now the ceiling. If the right entry is not in the top five, no amount of generation recovers it — which is why the R@5 from the previous lecture is the number that matters here."),
        code('''
t0 = time.perf_counter()
closed_cited, grounded_cited = [], []
for qi, q in enumerate(AMBIGUOUS):
    closed = ask(f"Our catalogue has {N_CATALOGUE} entries. A customer asks: "
                 f'"{q}". Recommend three entries and cite their SKUs.')
    shortlist = "\\n".join(f"- {catalogue[j]['sku']}: {catalogue[j]['captions'][0]}"
                          for j in top5[qi])
    grounded = ask(f'Here are the catalogue entries our search returned for the '
                   f'query "{q}":\\n{shortlist}\\n\\nRecommend the best ones for '
                   f'the customer, citing only SKUs from the list above. '
                   f'If none fit, say so.')
    closed_cited += SKU_RE.findall(closed)
    grounded_cited += SKU_RE.findall(grounded)

print(f"{len(AMBIGUOUS)} queries, both ways, in {time.perf_counter() - t0:.0f}s\\n")
for name, cited in [("closed book", closed_cited), ("retrieval-augmented", grounded_cited)]:
    ok = sum(s in valid_skus for s in cited)
    pct = 100 * ok / len(cited) if cited else 0.0
    print(f"{name:22s} {ok:3d} of {len(cited):3d} cited SKUs exist  ({pct:5.1f}%)")
'''),

        md("""
The closed-book failure is not that the model refuses. It is that it does *not*
refuse: it produces a fluent recommendation, in exactly the right SKU format,
and nothing in the output distinguishes an invented stock number from a real one.

What this does **not** fix:

* **Grounding is not correctness.** A cited entry can exist and still be a bad
  recommendation. We measured the cheap half.
* **The retriever is now the ceiling.** If the right entry is not in the top
  five, no amount of generation recovers it — which is why the R@5 from the
  previous lecture is the number that matters here.
* **Fluency is unchanged.** A wrong answer built from real SKUs reads better than
  one built from invented ones.

Evaluating generated answers — faithfulness, attribution, judging with another
model — is a live research area, outside Chapters 1–16 and not examinable.

## 9 · Red-team, and the end

Swap notebooks. Four questions for *this* notebook:

1. Are the embeddings on both sides unit vectors? Assert it, do not read it.
2. Is every recall reported with its candidate-set size?
3. Were the auto-captions generated for entries that are also in the evaluation
   queries — and does that matter here?
4. Does the grounding metric count a SKU cited twice as two citations? What would
   that do to the percentage?

---

### The course, in five rules

1. Split before anything is fitted.
2. All preprocessing inside the object that is cross-validated.
3. Nothing derived from the test set in the training path.
4. Fixed seeds; report per-fold scores, not only the mean.
5. Every number gets a baseline, and every baseline gets stated.

Twelve applications and twenty-four lectures, and those five never needed
extending — not for images, not for sequences, and not for two modalities at
once.

You will forget most of the syntax. Keep the five reviewer questions, the four
rules, and the habit of writing the number down first.
"""),
    ]
