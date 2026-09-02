#!/usr/bin/env python3
"""
Lecture 14 — Detection and segmentation. COCO, Géron Chapter 12.

Derivation: IoU's vanishing gradient, and mAP as a mean of a mean.

Merges the old lectures 17 and 18, which ran a detector and then, a lecture
later, derived a way to score it. Here the derivation comes first, because a
box has no notion of correct until IoU exists.

Exports build() -> list[nbformat cell]. Runs on CPU: 128 COCO images and a
pretrained detector, inference only. Nothing is trained.
"""

from __future__ import annotations

import nbformat as nbf
from _prompt import prompt                                # noqa: E402


def md(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(text.strip("\n"))


def code(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(text.strip("\n"))


HEADER = """
# Detection and segmentation

**Lecture 14** · Géron, Chapter 12

Applications of Machine Learning — BSc Mathematics of Artificial Intelligence

---

**How to use this notebook.** Read before you run. Every code cell is preceded
by the specification that would produce it — input, output, constraint, check.

Two cells marked **⚠** deliberately run code that is wrong, and say so in the
heading above them. They are the two failures this lecture is about: an IoU
that is silently wrong for disjoint boxes, and a mean taken over images where
it should be taken over instances. Both run, both print believable numbers, and
neither raises. Seeing what they print is the only way to learn to distrust
them.

**The corpus is 128 images.** COCO's `val2017` split is 5,000 images and the
full release is about 20 GB. Neither is downloaded here. Every number this
notebook prints is a measurement on 128 images, and you are expected to say
"128 images" whenever you quote one.
"""

SETUP = '''
# Not examinable, and only needed on some machines: PyTorch, numpy and
# torchvision can each end up loading their own OpenMP runtime, and with more
# than one loaded a training cell can deadlock -- no error, no output, and no
# CPU use. These have to be set BEFORE torch is imported, because they are read
# at import time and after that they do nothing.
import os
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# --- setup -------------------------------------------------------------------
# Not examinable: this is engineering hygiene, not machine learning.
import sys, json, time, itertools, urllib.request, zipfile, io
from pathlib import Path

import numpy as np
import torch, torchvision
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

print(f"python       {sys.version.split()[0]}")
print(f"torch        {torch.__version__}")
print(f"torchvision  {torchvision.__version__}")

RANDOM_STATE = 42                  # one seed, used everywhere
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)

DEVICE = ("cuda" if torch.cuda.is_available()
          else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"device       {DEVICE}")

N_IMAGES = 128                     # the corpus. Say it out loud every time.
DATA = Path("datasets/coco")
DATA.mkdir(parents=True, exist_ok=True)
'''

DOWNLOAD = '''
# --- the data ----------------------------------------------------------------
# Two downloads. The annotation file is the larger of them and it is the only
# way to have real ground-truth boxes at all; the images are 128 JPEGs, not
# 5,000 and certainly not the 20 GB training split.
#
# ⏱ about 60-90 seconds the first time, instant afterwards.
ANN = DATA / "instances_val2017.json"
IMG_DIR = DATA / "images"
IMG_DIR.mkdir(exist_ok=True)

if not ANN.is_file():
    url = ("http://images.cocodataset.org/annotations/"
           "annotations_trainval2017.zip")
    print(f"downloading annotations (~241 MB) from {url}")
    blob = urllib.request.urlopen(url).read()
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        ANN.write_bytes(z.read("annotations/instances_val2017.json"))

raw = json.loads(ANN.read_text())
print(f"{len(raw['images']):,} images in val2017, "
      f"{len(raw['categories'])} categories")

# The 128 numerically lowest image ids. A rule, not a selection: nobody chose
# which images make the detector look good.
images = sorted(raw["images"], key=lambda i: i["id"])[:N_IMAGES]
ids = {im["id"] for im in images}
assert len(images) == N_IMAGES

for im in images:
    p = IMG_DIR / im["file_name"]
    if not p.is_file():
        urllib.request.urlretrieve(
            "http://images.cocodataset.org/val2017/" + im["file_name"], p)
print(f"{len(list(IMG_DIR.glob('*.jpg')))} images on disk")
'''

GROUND_TRUTH = '''
# --- ground truth, converted once, at the edge --------------------------------
# COCO stores [x, y, w, h]. torchvision emits [x1, y1, x2, y2]. Mixing them is
# the commonest bug in this material, so the conversion happens HERE and
# nowhere else; from this cell on, every box in memory is corners.
cat_name = {c["id"]: c["name"] for c in raw["categories"]}

gt = {i: {"boxes": [], "labels": []} for i in ids}
n_crowd = 0
for a in raw["annotations"]:
    if a["image_id"] not in ids:
        continue
    if a["iscrowd"]:
        # One polygon drawn around many instances. Not one object, not n
        # objects — a refusal to decide. COCO's own evaluator ignores them.
        n_crowd += 1
        continue
    x, y, w, h = a["bbox"]
    gt[a["image_id"]]["boxes"].append([x, y, x + w, y + h])
    gt[a["image_id"]]["labels"].append(a["category_id"])

for iid, g in gt.items():
    g["boxes"] = np.asarray(g["boxes"], dtype=float).reshape(-1, 4)
    g["labels"] = np.asarray(g["labels"], dtype=np.int64)

# assert, do not hope
for iid, g in gt.items():
    assert (g["boxes"][:, 2] >= g["boxes"][:, 0]).all(), "x2 < x1: w read as x2"
    assert (g["boxes"][:, 3] >= g["boxes"][:, 1]).all(), "y2 < y1: same bug"

n_true = np.array([len(gt[im["id"]]["labels"]) for im in images])
assert n_true.shape == (N_IMAGES,)
print(f"{N_IMAGES} images, {n_true.sum()} objects, "
      f"{n_crowd} crowd regions dropped")
print(f"objects per image: mean {n_true.mean():.2f}  median "
      f"{np.median(n_true):.0f}  range {n_true.min()}-{n_true.max()}")
'''



def build() -> list:
    return [
        md(HEADER),

        md("## 1 · Setup"),prompt(
                                  label="setup",
                                  input="nothing",
                                  output="versions, one seed, the device, and the corpus size as a named constant",
                                  constraint="`N_IMAGES = 128` as a NAMED constant with a comment saying to say it out loud — every number this notebook prints is a measurement on 128 images",
                                  check="when a notebook works on a subset, put the subset size in a constant with a name, not a literal in a slice. It then appears in every printout that uses it."),
                            code(SETUP),
        md("## 2 · The corpus, and exactly how big it is"),prompt(
                                                                  label="⏱ 60-90 s — the corpus, and exactly how big it is",
                                                                  input="COCO's annotation file and 128 JPEGs",
                                                                  output="the annotation blob, the 128 chosen images, and the files on disk",
                                                                  constraint="choose the images by a RULE — the 128 numerically lowest image ids — so that nobody chose which images make the detector look good",
                                                                  check="assert exactly N_IMAGES were selected. A selection rule you can state in one sentence is a selection rule somebody can check. 'The lowest 128 ids' is; 'a representative sample' is not."),
                                                            code(DOWNLOAD),
        md("""
### 2.1 · Ground truth

Two things to notice in the next cell, both of which cost people an afternoon
the first time:

1. COCO stores a box as `[x, y, w, h]`; torchvision returns `[x1, y1, x2, y2]`.
   Convert once, at the edge of the program.
2. `iscrowd = 1` means the annotator drew one region around many instances
   rather than boxing them separately. Dropping those is a *choice*, it changes
   every count below, and this is where it is recorded.
"""),
        prompt(
            label="ground truth, converted once at the edge",
            input="COCO's annotations for the 128 images",
            output="corner-form boxes and labels per image, with the crowd regions counted and dropped",
            constraint="COCO stores [x, y, w, h] and torchvision emits [x1, y1, x2, y2]. Convert HERE and nowhere else — from this cell on, every box in memory is corners",
            check="assert x2 ≥ x1 and y2 ≥ y1 on every box, which is exactly the assertion that fires if w was read as x2. Convert at the boundary of your program, assert the invariant immediately, and never carry two conventions in the same variable name."),
        code(GROUND_TRUTH),

        md("""
### 2.2 · What is in it

`person` dominates. Remember that: in the next lecture we start averaging over
categories, and a mean over categories does not care that one of them is 39% of
the corpus.
"""),
        prompt(
            label="what is in it",
            input="the ground-truth labels",
            output="how many categories appear, and the eight commonest",
            constraint="print `person` as a SHARE of every annotated object, not just as a count",
            check="only some of the 80 categories appear in 128 images. Any per-category metric will have empty categories in it, and what you do about those changes the mean."),
        code('''
import collections

freq = collections.Counter()
for g in gt.values():
    for c in g["labels"]:
        freq[cat_name[int(c)]] += 1

print(f"{len(freq)} of 80 categories appear in these {N_IMAGES} images\\n")
for name, k in freq.most_common(8):
    print(f"  {name:14s} {k:4d}")
print(f"\\nperson is {freq['person'] / n_true.sum():.1%} of every "
      f"annotated object")
'''),

        md("""
## 3 · A metric, and the baseline that kills the obvious one

The obvious metric is: *a detection is correct when its box overlaps the true
box.* It is computable, unambiguous and parameter-free.

Before adopting any metric, this course computes what the stupidest possible
system scores under it. For detection, the stupidest possible system is
**one box per image, covering the whole image**.
"""),
        prompt(
            label="the baseline that kills the obvious metric",
            input="one box per image, covering the whole image",
            output="how many true objects it overlaps",
            constraint="test the OBVIOUS metric — a detection is correct when its box overlaps the true box — against the stupidest possible system before adopting it",
            check="assert it hits every object, which also verifies no annotated box lies outside its own image. Compute what the stupidest possible system scores under any metric BEFORE adopting it. Here that takes six lines and rules out the obvious choice."),
        code('''
def overlaps(a, b):
    """Do two corner-form boxes share any area at all?"""
    lt = np.maximum(a[:2], b[:2])
    rb = np.minimum(a[2:], b[2:])
    wh = np.clip(rb - lt, 0.0, None)
    return bool(wh[0] * wh[1] > 0)

hits = total = 0
for im in images:
    whole = np.array([0.0, 0.0, float(im["width"]), float(im["height"])])
    for b in gt[im["id"]]["boxes"]:
        hits += overlaps(whole, b)
        total += 1

print(f"the whole-image box overlaps {hits} of {total} true objects "
      f"= {hits / total:.1%}")
assert hits == total, "if this ever fails, a box lies outside its own image"
'''),
        md("""
**100%.** A system with no weights, no data and no idea scores perfectly under
the proposed metric. That metric is dead: it rewards a box for being enormous,
and nothing in it punishes size.

So today's metric is the one thing left that the whole-image box loses at:
**counting**.
"""),
        prompt(
            label="the metric we can defend",
            input="predicted and true object counts",
            output="the mean absolute error, and two trivial baselines",
            constraint="assert the two arrays have the same shape — a count vector of the wrong length broadcasts silently and gives a plausible number",
            check="a system that never opens the image scores 6.02. Every number below has to be read against that."),
        code('''
def count_mae(pred_counts, true_counts):
    pred_counts = np.asarray(pred_counts)
    assert pred_counts.shape == true_counts.shape
    return float(np.abs(pred_counts - true_counts).mean())

one_box   = count_mae(np.ones(N_IMAGES), n_true)
mean_box  = count_mae(np.full(N_IMAGES, round(n_true.mean())), n_true)

print(f"one box per image           MAE {one_box:.2f}")
print(f"predict the corpus mean     MAE {mean_box:.2f}")
print(f"perfect                     MAE 0.00")
'''),

        md("""
## 3 · The derivation, part 1 — intersection over union

Last time you were asked for a number that says how right a box is. It has to

1. be 1 for identical boxes and 0 for disjoint ones,
2. punish a box for being **too large**,
3. punish a box for being **too small**,
4. be dimensionless.

Requirement 2 forces the predicted area into the denominator; requirement 3
forces the true area in too; requirement 4 forces the numerator to be an area.
There is essentially one candidate:

$$\\mathrm{IoU}(A,B) \\;=\\; \\frac{|A \\cap B|}{|A \\cup B|}
  \\;=\\; \\frac{|A \\cap B|}{|A| + |B| - |A \\cap B|}$$
"""),
        prompt(
            label="intersection over union",
            input="two corner-form boxes",
            output="their IoU",
            constraint="CLAMP the overlap width and height at zero — this is not defensive programming, it is the whole function",
            check="assert 1.0 for identical boxes, exactly 0.0 for edge-to-edge, and 1/3 for half-overlapping. An exact `== 0.0` assert on the edge-to-edge case. A tolerance there would pass for a function that returns a small positive number, which is precisely the bug."),
        code('''
def iou(a, b):
    """IoU of two corner-form boxes [x1, y1, x2, y2].

    The clip is not defensive programming. Without it two disjoint boxes give
    a negative width AND a negative height, whose product is a positive
    "intersection".
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    lt = np.maximum(a[:2], b[:2])
    rb = np.minimum(a[2:], b[2:])
    wh = np.clip(rb - lt, 0.0, None)
    inter = wh[0] * wh[1]
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return float(inter / (area_a + area_b - inter))

box = np.array([0.0, 0.0, 100.0, 100.0])
assert iou(box, box) == 1.0
assert iou(box, box + np.array([100, 0, 100, 0])) == 0.0      # edge to edge
assert abs(iou(box, box + np.array([50, 0, 50, 0])) - 1 / 3) < 1e-12

for name, other in [("identical",        box),
                    ("half overlapping", box + [50, 0, 50, 0]),
                    ("edge to edge",     box + [100, 0, 100, 0]),
                    ("300 px away",      box + [300, 0, 300, 0])]:
    print(f"{name:18s} IoU = {iou(box, other):.3f}")
'''),

        # ------------------------------------------------ assistant failure
        md("""
### 3.1 · An assistant writes this function

> *"Write a NumPy function that takes two bounding boxes in `[x1, y1, x2, y2]`
> format and returns their intersection over union."*

**⚠ Read before running.** The implementation below is the one almost everyone
writes first: it handles overlapping boxes correctly and it is silently wrong
for disjoint ones, because the intersection width and height go negative and
their product goes positive again. Test it only on overlapping pairs and it
looks right.
"""),
        prompt(
            label="⚠ the IoU almost everyone writes first",
            input="'write a NumPy function taking two boxes in [x1,y1,x2,y2] and returning their intersection over union'",
            output="the IoU of two overlapping pairs",
            constraint="test it on OVERLAPPING boxes only, deliberately — that is the test suite this bug survives",
            check="a test suite made only of the cases you thought of tests the cases you thought of. Ask what input would make the output meaningless."),
        code('''
def iou_broken(a, b):
    x1 = max(a[0], b[0]);  y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]);  y2 = min(a[3], b[3])

    inter = (x2 - x1) * (y2 - y1)              # <- no clamp
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)

# every test a reasonable person writes first is a pair of OVERLAPPING boxes
print(iou_broken([0, 0, 100, 100], [50, 0, 150, 100]))     # 0.333... correct
print(iou_broken([0, 0, 100, 100], [10, 10, 110, 110]))    # 0.680... correct
'''),
        md("""
### Test against a case whose answer you know

*What does it return for two boxes that do not touch?*
"""),
        prompt(
            label="test against a case whose answer you know",
            input="boxes separated along one axis, and along both",
            output="what the broken function reports for each",
            constraint="show the ONE-axis case and the BOTH-axes case separately — they fail differently, and only one of them is obviously wrong",
            check="separation along one axis and along both are different tests. A single disjoint example is not a disjointness test."),
        code('''
print("one axis apart :", iou_broken([0, 0, 100, 100], [300, 0, 400, 100]))
print("both axes, 150 :", iou_broken([0, 0, 100, 100], [150, 150, 250, 250]))
print("both axes, 200 :", iou_broken([0, 0, 100, 100], [200, 200, 300, 300]))
print("\\nThe last line reports two boxes 200 px apart in BOTH directions")
print("as a perfect match. Two negative differences multiply to a positive.")
'''),
        md("""
Plot it and the failure is undeniable: the broken function is symmetric about
100 pixels, so it reports **more** overlap the further apart the boxes get.
"""),
        prompt(
            label="plot it and the failure is undeniable",
            input="two boxes pulled apart diagonally, 0 to 200 pixels",
            output="both functions' reported overlap against separation",
            constraint="pull them apart DIAGONALLY, so the broken function's symmetry about 100 px is visible",
            check="when a function is wrong on a region, plot it across that region. A curve that turns around where it should be flat at zero is not something anyone argues with."),
        code('''
d = np.arange(0, 201, 10, dtype=float)
ok  = [iou(box, box + [x, x, x, x]) for x in d]
bad = [iou_broken(box, box + np.array([x, x, x, x])) for x in d]

plt.figure(figsize=(9, 3.6))
plt.plot(d, ok,  color="#14663a", lw=3, marker="o", ms=4,
         label="IoU, with the clamp")
plt.plot(d, bad, color="#c0392b", lw=3, ls="--", marker="s", ms=4,
         label="IoU, clamp removed")
plt.axvline(100, color="#4b5563", lw=1.2, ls=":")
plt.xlabel("diagonal separation of two 100 x 100 boxes (pixels)")
plt.ylabel("reported overlap"); plt.legend(); plt.grid(alpha=0.3)
plt.title("the boxes are disjoint beyond 100 px; one curve does not know")
plt.tight_layout(); plt.show()

print(f"broken value at 200 px apart: {bad[-1]:.3f}   (should be 0.000)")
'''),
        md("""
### The corrected specification

> *"… returns their intersection over union. **Clamp the overlap width and
> height at zero.** Include tests for identical boxes, boxes sharing an edge,
> **boxes separated along one axis** and **boxes separated along both axes**.
> **Assert that the result is always in [0, 1].**"*

The last assertion is the one that fails immediately, on the first disjoint
pair, without anyone having to think of the diagonal case.
"""),
        prompt(
            label="the property test that would have caught it",
            input="a 5 by 5 grid of separations",
            output="the property holding on 25 pairs, and the broken version failing it",
            constraint="state the PROPERTY, not the values: the result is in [0,1], and it is zero if and only if the boxes are disjoint on some axis",
            check="run the same loop against the broken function and show that it fails. Demonstrate that your test catches the bug. A test suite nobody has seen fail is a test suite of unknown strength."),
        code('''
def iou_checked(a, b):
    v = iou(a, b)
    assert 0.0 <= v <= 1.0, f"IoU out of range: {v}"
    return v

# state the PROPERTY, not the values: zero if and only if disjoint on some axis
n_checked = 0
for dx, dy in itertools.product([0, 50, 100, 150, 200], repeat=2):
    other = box + np.array([dx, dy, dx, dy])
    v = iou_checked(box, other)
    assert (v == 0.0) == (dx >= 100 or dy >= 100), (dx, dy, v)
    n_checked += 1
print(f"{n_checked} pairs checked, property holds")

# and the broken one fails the same loop
try:
    for dx, dy in itertools.product([0, 200], repeat=2):
        v = iou_broken(box, box + np.array([dx, dy, dx, dy]))
        assert 0.0 <= v <= 1.0 and (v == 0.0) == (dx >= 100 or dy >= 100)
    print("broken version passed — it should not have")
except AssertionError:
    print("broken version fails the property test, as it must")
'''),

        # ------------------------------------------------ vanishing gradient
        md("""
## 4 · The derivation, part 2 — the gradient that is not there

IoU does three jobs: matching, suppression, and serving as a loss. Only the
third needs a derivative, and that is the one that breaks.

Pull two 100 × 100 boxes apart and ask autograd for the derivative at each
separation. Nothing here depends on a dataset: the conclusion is a property of
the formula.
"""),
        prompt(
            label="the gradient that is not there",
            input="two 100×100 boxes pulled apart, 0 to 300 px, with autograd",
            output="IoU and GIoU with their derivatives at each separation",
            constraint="ask AUTOGRAD for the derivative rather than differentiating by hand — the claim is about what an optimiser would receive",
            check="float64 and `torch.autograd.grad` on a scalar. This is a claim about an exact zero, and float32 would leave you unable to distinguish zero from 1e-9."),
        code('''
def t_iou(a, b):
    lt = torch.maximum(a[:2], b[:2])
    rb = torch.minimum(a[2:], b[2:])
    wh = torch.clamp(rb - lt, min=0.0)
    inter = wh[0] * wh[1]
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)

def t_giou(a, b):
    v = t_iou(a, b)
    lt_c = torch.minimum(a[:2], b[:2]); rb_c = torch.maximum(a[2:], b[2:])
    wh_c = torch.clamp(rb_c - lt_c, min=0.0)
    area_c = wh_c[0] * wh_c[1]
    lt_i = torch.maximum(a[:2], b[:2]); rb_i = torch.minimum(a[2:], b[2:])
    wh_i = torch.clamp(rb_i - lt_i, min=0.0)
    inter = wh_i[0] * wh_i[1]
    union = ((a[2] - a[0]) * (a[3] - a[1])
             + (b[2] - b[0]) * (b[3] - b[1]) - inter)
    return v - (area_c - union) / area_c

A = torch.tensor([0.0, 0.0, 100.0, 100.0], dtype=torch.float64)

rows = []
for dv in np.arange(0, 301, 10, dtype=float):
    d = torch.tensor(dv, dtype=torch.float64, requires_grad=True)
    B = torch.stack([d, torch.zeros_like(d),
                     d + 100.0, torch.full_like(d, 100.0)])
    vals = {}
    for nm, fn in (("iou", t_iou), ("giou", t_giou)):
        v = fn(A, B)
        g, = torch.autograd.grad(v, d)
        vals[nm] = (float(v.detach()), float(g))
    rows.append((dv, vals["iou"][0], vals["iou"][1],
                 vals["giou"][0], vals["giou"][1]))

print(f"{'d':>5s} {'IoU':>8s} {'dIoU/dd':>10s} {'GIoU':>8s} {'dGIoU/dd':>10s}")
for dv, i, gi, g, gg in rows[::5]:
    print(f"{dv:5.0f} {i:8.3f} {gi:10.5f} {g:8.3f} {gg:10.5f}")
'''),
        md("""
Read rows four and five. Two boxes 150 px apart and two boxes 300 px apart are,
to IoU, **exactly equally wrong** — same value, same gradient, and the gradient
is not small but zero.

Assert it rather than eyeballing it:
"""),
        prompt(
            label="assert it rather than eyeballing it",
            input="every separation past 100 pixels",
            output="the maximum IoU and maximum gradient magnitude there, and GIoU's gradient for contrast",
            constraint="assert IoU is identically zero AND its gradient is identically zero AND GIoU's gradient is still negative — three claims, three asserts",
            check="an assert on an exact equality with zero is available here because the quantity is structurally zero. Take it — it distinguishes 'vanishing' from 'absent'."),
        code('''
past = [r for r in rows if r[0] > 100]
assert all(r[1] == 0.0 for r in past), "IoU should be identically zero"
assert all(r[2] == 0.0 for r in past), "and so should its gradient"
assert all(r[4] < 0.0 for r in past), "GIoU should still be descending"
print(f"{len(past)} separations past 100 px:")
print(f"  max IoU there          {max(r[1] for r in past):.6f}")
print(f"  max |dIoU/dd| there    {max(abs(r[2]) for r in past):.6f}")
print(f"  GIoU at 300 px         {past[-1][3]:.3f}")
print(f"  dGIoU/dd at 300 px     {past[-1][4]:.5f}")
'''),
        md("""
### Why it is exactly zero, not merely small

For $d \\geq 100$ the overlap width is $\\max(0, 100 - d) = 0$, so the
intersection is identically zero, so IoU is **constant** on the whole disjoint
region. A constant has no descent direction — not a weak one, none. No
optimiser, learning rate or initialisation repairs that.

### The repairs

$$\\mathrm{GIoU} = \\mathrm{IoU} - \\frac{|C| - |A \\cup B|}{|C|}, \\qquad
  \\mathrm{CIoU} = \\mathrm{IoU} - \\frac{\\rho^2}{\\ell^2} - \\alpha v$$

where $C$ is the smallest box containing both, $\\rho$ is the distance between
centres, $\\ell$ is the diagonal of $C$, and $v$ measures aspect-ratio
disagreement. When the boxes are disjoint and move apart, $|C|$ grows and
$|A \\cup B|$ does not, so the penalty grows.
"""),
        prompt(
            label="CIoU, and its invisible term",
            input="a same-shaped disjoint box and a differently-shaped one",
            output="IoU and CIoU for each",
            constraint="test with TWO shapes — the aspect-ratio term is exactly zero when the shapes agree, and every pair in the figure above is square",
            check="when a term of a formula can be exactly zero on your test cases, construct a case where it is not. Otherwise you have tested a simpler function."),
        code('''
def t_ciou(a, b):
    v = t_iou(a, b)
    ca = torch.stack([(a[0] + a[2]) / 2, (a[1] + a[3]) / 2])
    cb = torch.stack([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2])
    rho2 = ((ca - cb) ** 2).sum()
    lt = torch.minimum(a[:2], b[:2]); rb = torch.maximum(a[2:], b[2:])
    c2 = ((rb - lt) ** 2).sum()
    wa, ha = a[2] - a[0], a[3] - a[1]
    wb, hb = b[2] - b[0], b[3] - b[1]
    vv = (4 / torch.pi ** 2) * (torch.atan(wa / ha) - torch.atan(wb / hb)) ** 2
    alpha = vv / (1 - v + vv + 1e-12)
    return v - rho2 / c2 - alpha * vv

same_shape = torch.tensor([120.0, 0.0, 220.0, 100.0], dtype=torch.float64)
thin       = torch.tensor([120.0, 0.0, 320.0,  50.0], dtype=torch.float64)

for nm, B in (("same shape 100x100", same_shape), ("different 200x50", thin)):
    print(f"{nm:20s} IoU {float(t_iou(A, B)):6.3f}   "
          f"CIoU {float(t_ciou(A, B)):7.3f}")
print("\\nThe aspect term is exactly zero when the shapes agree, which is why")
print("it is invisible in the figure above: every pair there is square.")
'''),
        md("""
**An honest caveat.** GIoU and CIoU are *losses*: their value is in the
backward pass of a detector you are fitting. Nothing in this application fits a
detector, so what you have just seen is a property of the functions rather than
a measured improvement in a model we built. The rest of the notebook is about
the evaluation, which we *can* measure.
"""),

        # ------------------------------------------------ AP
        md("""
## 5 · The derivation, part 3 — average precision

⏱ **a few minutes on CPU**: the same detector as last lecture, over the same
128 images.
"""),
        prompt(
            label="⏱ 1-2 min — the same detector, the same images",
            input="the 128 images",
            output="predictions, converted to float numpy with integer labels",
            constraint="cast labels back to int64 after the float conversion — a label of 1.0 will not match a category_id of 1 in a boolean mask, and the resulting comparison is silently all-False",
            check="assert one prediction per image. When you bulk-convert a dict of tensors, check the dtypes afterwards. One of them is an index and does not want to be a float."),
        code('''
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights)

weights = FasterRCNN_ResNet50_FPN_Weights.COCO_V1
model = fasterrcnn_resnet50_fpn(weights=weights).eval().to(DEVICE)
preprocess = weights.transforms()
names = weights.meta["categories"]

t0 = time.time()
preds = {}
with torch.inference_mode():
    for im in images:
        img = Image.open(IMG_DIR / im["file_name"]).convert("RGB")
        out = model([preprocess(img).to(DEVICE)])[0]
        preds[im["id"]] = {k: v.cpu().numpy().astype(float)
                           for k, v in out.items()}
        preds[im["id"]]["labels"] = preds[im["id"]]["labels"].astype(np.int64)

assert len(preds) == N_IMAGES
print(f"{N_IMAGES} images in {time.time() - t0:.1f} s on {DEVICE}")
'''),
        md("""
### 5.1 · Matching, precision, recall

A detector is a **ranking** — Lecture 4's shape of problem. For one class and
one IoU threshold:

1. sort every detection of that class, over the whole corpus, by score;
2. walk down the list; for each detection find the best **unmatched**
   annotation in the same image;
3. IoU at least $t$ → true positive, and that annotation is now used up.
   Otherwise → false positive.

Step 2's word *unmatched* is what makes a second box on the same bottle a false
positive rather than a second success.
"""),
        prompt(
            label="matching, precision and recall",
            input="one class, one IoU threshold, every detection in the corpus",
            output="the cumulative recall and precision along the ranking",
            constraint="sort by score over the WHOLE corpus, and match each detection to the best UNMATCHED annotation in its own image",
            check="handle the empty case in `iou_many` — a class with no ground truth in an image gives a zero-length array, and `argmax` on it raises. Mark the annotation as used the moment it is matched. The bookkeeping is three lines and it is the entire difference between AP and a count of overlaps."),
        code('''
def iou_many(one, many):
    """IoU of one box against an (M, 4) array of boxes."""
    if len(many) == 0:
        return np.zeros(0)
    lt = np.maximum(one[:2], many[:, :2])
    rb = np.minimum(one[2:], many[:, 2:])
    wh = np.clip(rb - lt, 0.0, None)
    inter = wh[:, 0] * wh[:, 1]
    area_1 = (one[2] - one[0]) * (one[3] - one[1])
    area_m = (many[:, 2] - many[:, 0]) * (many[:, 3] - many[:, 1])
    return inter / np.maximum(area_1 + area_m - inter, 1e-12)


def pr_curve(cls, t):
    """Cumulative precision and recall for one class at one IoU threshold."""
    n_gt, gt_by_img = 0, {}
    for iid, g in gt.items():
        m = g["labels"] == cls
        gt_by_img[iid] = g["boxes"][m]
        n_gt += int(m.sum())

    rows = []
    for iid, p in preds.items():
        m = p["labels"] == cls
        rows += [(float(s), iid, b) for b, s in zip(p["boxes"][m],
                                                    p["scores"][m])]
    rows.sort(key=lambda r: -r[0])

    used = {iid: np.zeros(len(g), bool) for iid, g in gt_by_img.items()}
    tp = np.zeros(len(rows))
    for k, (_s, iid, b) in enumerate(rows):
        g = gt_by_img[iid]
        free = ~used[iid]
        if len(g) and free.any():
            v = iou_many(b, g[free])
            j = int(v.argmax())
            if v[j] >= t:
                tp[k] = 1.0
                used[iid][np.flatnonzero(free)[j]] = True
    ctp = tp.cumsum()
    cfp = (1.0 - tp).cumsum()
    return ctp / max(n_gt, 1), ctp / np.maximum(ctp + cfp, 1e-12), n_gt


recall, precision, n_gt = pr_curve(cls=1, t=0.5)          # 1 == person
print(f"person: {n_gt} annotations, {len(precision)} detections in the ranking")
print(f"true positives: {int(round(precision[-1] * len(precision)))}")
print(f"highest recall reached: {recall[-1]:.3f}")
'''),
        md("""
### 5.2 · Precision is not monotone — Lecture 4, on boxes

Classify every step of the ranking exactly, the way Lecture 4 did for MNIST.
"""),
        prompt(
            label="precision is not monotone — application 2, on boxes",
            input="the person class's precision curve",
            output="how many steps go down, up and flat, checked against the true and false positive counts",
            constraint="classify every step EXACTLY and assert the identity: every false positive is a step down, and every true positive but the first is up or flat",
            check="when a curve has a structure you can predict from counts, assert the prediction. If it fails, either your matching or your counting is wrong and you now know which."),
        code('''
step = np.diff(precision)
down = int((step < -1e-12).sum())
up   = int((step >  1e-12).sum())
flat = int((np.abs(step) <= 1e-12).sum())
n_tp = int(round(precision[-1] * len(precision)))
n_fp = len(precision) - n_tp

print(f"steps down (precision falls) : {down}")
print(f"steps up   (precision rises) : {up}")
print(f"steps flat (already at 1)    : {flat}")
print(f"total steps                  : {down + up + flat}")

# the identity: every FP is a step down; every TP but the first is up or flat
assert down == n_fp, (down, n_fp)
assert up + flat == n_tp - 1, (up, flat, n_tp)
assert down + up + flat == len(precision) - 1
print(f"\\nfalse positives = {n_fp} = steps down, exactly")
print(f"true positives  = {n_tp}, less the one at the top of the ranking,")
print(f"                  = {n_tp - 1} = {up} up + {flat} flat")

leading = int(np.flatnonzero(precision < 1.0)[0])
print(f"\\nthe {flat} flat steps are one run: the top {leading} person")
print("detections in the whole corpus are all correct")
'''),
        md("""
### 5.3 · The repair Lecture 4 promised

Lecture 4 proved precision has no monotone envelope you can rely on. Average
precision is defined using the **maximum precision at or above each recall
level**, and that maximum exists for exactly one reason: to replace a
non-monotone quantity by a monotone one.

$$p_{\\text{env}}(r) = \\max_{\\tilde r \\geq r} p(\\tilde r),
  \\qquad \\mathrm{AP} = \\int_0^1 p_{\\text{env}}(r)\\,\\mathrm{d}r$$

No threshold appears anywhere. That is the whole point.
"""),
        prompt(
            label="average precision, checked by hand",
            input="the precision-recall curve",
            output="the AP, plus a four-detection case computed by hand in the comment",
            constraint="check it against a case you can do ON PAPER — four detections, two annotations, TP FP TP FP in score order, AP = 0.5 + 1/3",
            check="the hand-computable assert, to 1e-12. No threshold appears anywhere in AP. That is the whole point of it, and it is what makes it comparable across detectors."),
        code('''
def envelope(p):
    """Maximum precision at or above each recall level. One sweep, right to
    left. Monotone non-increasing by construction."""
    out = np.asarray(p, float).copy()
    for i in range(len(out) - 2, -1, -1):
        out[i] = max(out[i], out[i + 1])
    return out


def average_precision(precision, recall):
    """Area under the enveloped curve — the all-point definition."""
    if len(precision) == 0:
        return 0.0
    mrec = np.concatenate([[0.0], recall, [recall[-1]]])
    mpre = envelope(np.concatenate([[0.0], precision, [0.0]]))
    idx = np.flatnonzero(mrec[1:] != mrec[:-1])
    return float(((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]).sum())


# check it on a case you can do by hand: 4 detections, 2 annotations,
# labels TP, FP, TP, FP in score order.
#   after 1: P=1,   R=0.5
#   after 2: P=0.5, R=0.5
#   after 3: P=2/3, R=1.0
#   after 4: P=0.5, R=1.0
# envelope: max precision at recall >= 0.5 is 1 ... but at recall 1.0 it is 2/3
# AP = (0.5 - 0) * 1.0 + (1.0 - 0.5) * (2/3) = 0.5 + 1/3 = 0.8333...
hand_p = np.array([1.0, 0.5, 2 / 3, 0.5])
hand_r = np.array([0.5, 0.5, 1.0, 1.0])
assert abs(average_precision(hand_p, hand_r) - (0.5 + 1 / 3)) < 1e-12
print("hand-computable case passes:",
      f"{average_precision(hand_p, hand_r):.4f}")

ap_person = average_precision(precision, recall)
print(f"\\nAP for person at IoU 0.5, on 128 images: {ap_person:.3f}")
'''),
        md("""
### 5.4 · Draw it
"""),
        prompt(
            label="draw it, twice",
            input="the curve and its envelope",
            output="the full PR curve with the area shaded, and a zoomed window on the sawtooth",
            constraint="`where='post'` on the step plots — a precision-recall curve drawn with linear interpolation claims performance at recall levels that were never achieved",
            check="shade the area you are claiming to integrate. An AP quoted beside an unshaded curve is a number beside a picture."),
        code('''
env = envelope(precision)

fig, axes = plt.subplots(1, 2, figsize=(13, 4))
ax = axes[0]
ax.step(recall, precision, where="post", color="#c0392b", lw=2,
        label="precision as measured")
ax.step(recall, env, where="post", color="#14663a", lw=2.6,
        label="max precision at or above")
ax.fill_between(recall, 0, env, step="post", color="#14663a", alpha=0.12)
ax.set_xlabel("recall"); ax.set_ylabel("precision"); ax.set_ylim(0, 1.05)
ax.set_title(f"person, IoU >= 0.5, AP = {ap_person:.3f}")
ax.legend(loc="lower left"); ax.grid(alpha=0.3)

ax = axes[1]
lo, hi = 50, 140
k = np.arange(lo + 1, hi + 1)
ax.plot(k, precision[lo:hi], color="#c0392b", lw=2, marker="o", ms=3,
        label="precision")
ax.step(k, env[lo:hi], where="post", color="#14663a", lw=2.4,
        label="its maximum")
ax.set_xlabel("detections accepted, in score order")
ax.set_ylabel("precision")
ax.set_title("the sawtooth, and the staircase that repairs it")
ax.legend(loc="lower left"); ax.grid(alpha=0.3)
plt.tight_layout(); plt.show()
'''),

        # ------------------------------------------------ mAP
        md("""
## 6 · The derivation, part 4 — mAP, a mean of a mean

⏱ **about 60 seconds**: 73 classes × 10 IoU thresholds.
"""),
        prompt(
            label="⏱ 60 s — mAP, a mean of a mean",
            input="73 classes × 10 IoU thresholds",
            output="mAP at 0.50, at 0.75, and averaged across the ten",
            constraint="only classes that ACTUALLY APPEAR — a class with no annotations has no AP, and counting it as zero would be inventing a measurement",
            check="say '73 classes' out loud alongside '128 images'. Both are part of the number."),
        code('''
present = sorted({int(c) for g in gt.values() for c in g["labels"]})
print(f"{len(present)} of COCO's 80 categories appear in our 128 images")

IOU_TS = np.round(np.arange(0.50, 0.96, 0.05), 2)

t0 = time.time()
ap = {}                                    # ap[(class, t)] -> AP
for t in IOU_TS:
    for c in present:
        r, p, n = pr_curve(c, float(t))
        if n:
            ap[(c, float(t))] = average_precision(p, r)
print(f"{len(ap)} class-threshold pairs in {time.time() - t0:.0f} s")

map_at = {float(t): float(np.mean([ap[(c, float(t))] for c in present
                                   if (c, float(t)) in ap]))
          for t in IOU_TS}

print(f"\\nmAP @ 0.50            {map_at[0.50]:.3f}")
print(f"mAP @ 0.75            {map_at[0.75]:.3f}")
print(f"mAP @ [0.50:0.95]     {np.mean(list(map_at.values())):.3f}")
print("\\n...all on 128 images, and 73 classes.")
'''),
        md("""
### 6.1 · What the first mean hides
"""),
        prompt(
            label="what the first mean hides",
            input="the per-class APs at IoU 0.50, and the instance counts",
            output="best, median, mean and worst, with instance counts, and how many classes score exactly 1.000",
            constraint="print the INSTANCE COUNT beside every class you name — the mean is unweighted, and that is the finding",
            check="a mean over categories does not care that one of them is 39% of the corpus. If you want it to, you need a different statistic and you should say which."),
        code('''
per50 = sorted(((names[c], ap[(c, 0.5)]) for c in present if (c, 0.5) in ap),
               key=lambda r: -r[1])
inst = collections.Counter()
for g in gt.values():
    for c in g["labels"]:
        inst[names[int(c)]] += 1

m50 = map_at[0.50]
perfect = [n for n, v in per50 if v == 1.0]
print(f"classes scoring exactly 1.000: {len(perfect)}")
print("  and their instance counts:",
      sorted(inst[n] for n in perfect))
print(f"\\nbest   {per50[0][0]:12s} {per50[0][1]:.3f} "
      f"({inst[per50[0][0]]} instances)")
print(f"median {'':12s} {np.median([v for _n, v in per50]):.3f}")
print(f"mean   {'= the mAP':12s} {m50:.3f}")
print(f"worst  {per50[-1][0]:12s} {per50[-1][1]:.3f} "
      f"({inst[per50[-1][0]]} instances)")
print(f"\\nclasses below the mean: {sum(1 for _n, v in per50 if v < m50)}")
print("A class with one annotation scores 1.000 or 0.000 and nothing")
print("between, and it weighs as much in the mean as person with 350.")
'''),
        md("""
### 6.2 · What the second mean hides
"""),
        prompt(
            label="what the second mean hides",
            input="mAP at each of the ten IoU thresholds",
            output="the curve, with the mean drawn across it",
            constraint="draw the mean as a horizontal line ON the curve — the point is how far the endpoints are from it",
            check="when a headline metric is a mean over a parameter, plot it against that parameter once. It takes four lines and it changes how the number reads."),
        code('''
plt.figure(figsize=(9, 3.6))
plt.plot(IOU_TS, [map_at[float(t)] for t in IOU_TS], color="#0b3d62", lw=3,
         marker="o")
plt.axhline(np.mean(list(map_at.values())), color="#6c3483", ls="--", lw=2,
            label=f"mean over the ten = {np.mean(list(map_at.values())):.3f}")
plt.xlabel("IoU threshold at which a detection counts as correct")
plt.ylabel("mAP over the 73 classes present")
plt.legend(); plt.grid(alpha=0.3)
plt.title("128 images: 0.659 at the loosest threshold, 0.040 at the tightest")
plt.tight_layout(); plt.show()

print(f"mAP at 0.50: {map_at[0.50]:.3f}")
print(f"mAP at 0.95: {map_at[0.95]:.3f}")
print("One number in the middle stands for both.")
'''),

        # ------------------------------------------------ second failure
        md("""
## 7 · The detector

Nothing is trained here. These are the weights torchvision ships, trained on
COCO's training split by someone else, and this lecture is about evaluating
them rather than fitting them.

⏱ the weights are about 167 MB; the download happens once.
"""),
        prompt(
            label="the detector — nothing is trained here",
            input="torchvision's COCO-trained weights",
            output="the model in eval mode, its preprocessing, and its category names",
            constraint="assert that the model's label integers ARE COCO's category_ids — the comparison further down is only legitimate if they agree",
            check="the name-agreement assert, plus `names[1] == 'person'`. Print the weights' own reported metrics on the full 5,000 images. Your 128-image number should be read next to it, not instead of it."),
        code('''
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights)

weights = FasterRCNN_ResNet50_FPN_Weights.COCO_V1
model = fasterrcnn_resnet50_fpn(weights=weights)
model.eval().to(DEVICE)              # eval(), every time — Lecture 12
preprocess = weights.transforms()

names = weights.meta["categories"]
print(f"{len(names)} label slots for 80 categories")
print("slot 0 is", names[0], "| slot 12 is", names[12])
assert names[1] == "person"
# the integer in `labels` is the same integer as COCO's category_id, which is
# the only reason the comparison further down is legitimate
assert all(names[cid] == nm for cid, nm in cat_name.items())

print("\\ntorchvision's own reported score for these weights,")
print("on all 5,000 val2017 images:", weights.meta["_metrics"])
'''),

        md("""
### 5.1 · Run it

⏱ **a few minutes on CPU.** How many depends on what else the machine is
doing — the same loop took roughly 40 s on an idle laptop and roughly 100 s on
a busy one, which is why this notebook reports times to one significant figure
and never puts two of them in a column inviting a comparison. No output does
not mean it has hung.
"""),
        prompt(
            label="⏱ a few min on CPU — run it",
            input="the 128 images",
            output="predictions per image, and the wall clock per image",
            constraint="`torch.inference_mode()` — no graph and no gradients, which matters here because a detector's intermediate tensors are large",
            check="assert one prediction per image. Move the predictions to CPU numpy inside the loop. Keeping 128 sets of GPU tensors alive is how the next cell fails for a reason that has nothing to do with the next cell."),
        code('''
t0 = time.time()
preds = {}
with torch.inference_mode():                 # no graph, no gradients
    for im in images:
        img = Image.open(IMG_DIR / im["file_name"]).convert("RGB")
        out = model([preprocess(img).to(DEVICE)])[0]
        preds[im["id"]] = {k: v.cpu().numpy() for k, v in out.items()}
elapsed = time.time() - t0

assert len(preds) == N_IMAGES
print(f"{N_IMAGES} images in {elapsed:.1f} s "
      f"({elapsed / N_IMAGES:.2f} s per image on {DEVICE})")
'''),
        md("""
### 5.2 · Read the shape before you read the answer
"""),
        prompt(
            label="read the shape before the answer",
            input="one image's predictions",
            output="the shape and dtype of every returned array",
            constraint="assert the three arrays are the same length AND that scores come back sorted descending — everything below relies on both",
            check="assert the sort order rather than assuming it. Several detection APIs return unsorted boxes, and code that slices the 'top k' silently takes an arbitrary k."),
        code('''
p = preds[images[0]["id"]]
for k, v in p.items():
    print(f"{k:8s} {v.shape} {v.dtype}")

assert p["boxes"].shape[0] == p["labels"].shape[0] == p["scores"].shape[0]
assert np.all(np.diff(p["scores"]) <= 0), "not sorted by score"
print(f"\\nthis image has {len(p['boxes'])} boxes and "
      f"{len(gt[images[0]['id']]['labels'])} annotated objects")
'''),

        prompt(
            label="counting, and the threshold nobody chose",
            input="one image's predictions and a confidence threshold",
            output="the number of objects the detector claims to see",
            constraint="fix the reported threshold HERE, before the sweep that follows — a threshold picked after seeing which one minimises the error is a hyperparameter fitted on the test set",
            check="0.5 is a convention, not a result. It is written down first so the sweep below cannot quietly become a search."),
        code('''
THRESH = 0.5          # the convention, fixed before the sweep, not after it

def count_objects(pred, thresh=THRESH):
    """How many objects the detector claims, above a confidence threshold."""
    return int((pred["scores"] >= thresh).sum())

counts = np.array([count_objects(preds[im["id"]]) for im in images])
print(f"at threshold {THRESH}: {counts.mean():.2f} objects per image, "
      f"MAE {count_mae(counts, n_true):.2f}")
print(f"the truth:            {n_true.mean():.2f} objects per image")
'''),

        md("""
## 8 · The threshold is a knob, and nobody chose it

Sweep it and watch the answer to the stakeholder's question move by a factor of
ten.
"""),
        prompt(
            label="the threshold is a knob, and nobody chose it",
            input="nineteen thresholds",
            output="mean count and MAE at each, with the reported one marked",
            constraint="mark the value we report, and find the BEST one — then refuse to report the best",
            check="the answer to the stakeholder's question moves by a factor of ten across this sweep. A single count with no threshold stated is not an answer."),
        code('''
ts = np.round(np.arange(0.05, 0.96, 0.05), 2)
rows = []
for t in ts:
    c = np.array([count_objects(preds[im["id"]], t) for im in images])
    rows.append((t, c.mean(), count_mae(c, n_true)))

print(f"{'thresh':>7s} {'mean/img':>9s} {'MAE':>7s}")
for t, m, e in rows:
    mark = "  <- we report this" if abs(t - THRESH) < 1e-9 else ""
    print(f"{t:7.2f} {m:9.2f} {e:7.2f}{mark}")

best = min(rows, key=lambda r: r[2])
print(f"\\nlowest MAE is {best[2]:.2f} at threshold {best[0]:.2f}")
print("We do NOT report that one: it was found on the same 128 images we")
print("then report on, which is choosing a hyperparameter on the test set.")
'''),

        md("""
## 9 · Look at the pictures, not only at the number

Three images with their predicted boxes. Labels are drawn only for confident
detections, because a crowded image stacks fourteen captions on top of each
other and an illegible figure teaches nothing.
"""),
        prompt(
            label="look at the pictures, not only at the number",
            input="three images with their detections",
            output="boxes drawn, with labels only on confident detections",
            constraint="label only above a high score — a crowded image stacks fourteen captions on top of each other and an illegible figure teaches nothing",
            check="try to write down a number for 'how wrong is that box'. You cannot, and neither can a count of objects."),
        code('''
def show(iid, thresh=THRESH, label_above=0.90, ax=None):
    im = next(i for i in images if i["id"] == iid)
    ax = ax or plt.gca()
    ax.imshow(Image.open(IMG_DIR / im["file_name"]).convert("RGB"))
    p = preds[iid]
    keep = np.flatnonzero(p["scores"] >= thresh)
    for k in keep:
        x1, y1, x2, y2 = p["boxes"][k]
        ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False,
                               edgecolor="#c0392b", linewidth=2))
        if p["scores"][k] >= label_above:
            ax.text(x1 + 2, max(y1 - 4, 12),
                    f"{names[int(p['labels'][k])]} {p['scores'][k]:.2f}",
                    color="white", fontsize=8,
                    bbox=dict(fc="#c0392b", ec="none", pad=1.0))
    ax.set_title(f"{len(keep)} boxes, {len(gt[iid]['labels'])} true",
                 fontsize=10, loc="left")
    ax.set_xticks([]); ax.set_yticks([])

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, im in zip(axes, images[:3]):
    show(im["id"], ax=ax)
plt.tight_layout(); plt.show()
'''),
        md("""
Some of those boxes are visibly wrong: two on one object, one a little too
large, one confident about nothing at all.

**You have no way to say how wrong.** Try it: write down a number for "how
wrong is that box". You cannot, and neither can a count of objects.
"""),
        prompt(
            label="two systems the metric cannot tell apart",
            input="nine true boxes, and the same nine shifted 400 pixels",
            output="the count error of each",
            constraint="construct the counterexample rather than describing it — both systems emit nine boxes for an image with nine objects, and one puts them on the objects",
            check="when you suspect a metric is blind to something, build the pair it cannot separate. If you can, the metric is dead for that purpose."),
        code('''
# Two systems your metric cannot tell apart. Both emit nine boxes for an
# image with nine objects; one puts them on the objects and one does not.
true_boxes = gt[images[0]["id"]]["boxes"]
k = min(9, len(true_boxes))

system_a = true_boxes[:k].copy()                  # exactly right
system_b = true_boxes[:k].copy() + 400.0          # exactly the wrong places

print(f"system A: {len(system_a)} boxes, count error "
      f"{abs(len(system_a) - k)}")
print(f"system B: {len(system_b)} boxes, count error "
      f"{abs(len(system_b) - k)}")
print("\\nA count of objects scores both of them perfect.")
'''),

        md("""
## 10 · Per-image averaging

An assistant asked to *"report mAP over the dataset"* will sometimes compute AP
for each image and average those. It runs, and it is worth a great deal of free
mAP.

⏱ **about 30 seconds.**
"""),
        prompt(
            label="⏱ 30 s — ⚠ the second silent failure",
            input="AP computed per image and then averaged",
            output="the per-image figure beside the correctly accumulated one",
            constraint="restore `preds` and `gt` afterwards — this cell rebinds the globals the rest of the notebook uses, and forgetting to put them back breaks every cell below with no obvious cause",
            check="assert the per-image version is optimistic, since the whole point is that it is free mAP. This is the metric averaged per batch rather than over the set — the same entry in the silent-failure catalogue you met in application 6, wearing detection clothes."),
        code('''
# ⚠ read before running — this is the WRONG way, on purpose
all_preds, all_gt = preds, gt
per_image = []
for im in images:
    iid = im["id"]
    preds, gt = {iid: all_preds[iid]}, {iid: all_gt[iid]}       # one image
    vals = []
    for c in sorted({int(x) for x in all_gt[iid]["labels"]}):
        r, p, n = pr_curve(c, 0.5)
        if n:
            vals.append(average_precision(p, r))
    if vals:
        per_image.append(float(np.mean(vals)))
preds, gt = all_preds, all_gt                                    # put it back

wrong = float(np.mean(per_image))
print(f"accumulated over the corpus, correctly : {m50:.3f}")
print(f"computed per image, then averaged      : {wrong:.3f}")
print(f"free mAP                               : {wrong - m50:+.3f}")
assert wrong > m50, "the per-image version should be optimistic"
'''),
        md("""
This is *the metric averaged per batch rather than over the set* — the same
entry in the course's silent-failure catalogue you met in Lecture 12, wearing
detection clothes.

Why it inflates: a single image usually contains one or two classes and a
handful of objects, so its own AP is often exactly 1.0. Averaging a lot of easy
1.0s is not the same as ranking every detection in the corpus against every
other.
"""),

        # ------------------------------------------------ NMS
        md("""
## 11 · Non-maximum suppression

The detector you ran had already thrown away nine tenths of its own output
before you saw it, using IoU, at a threshold you did not set.

⏱ **about 60 seconds**: the same 128 images with suppression switched off.
"""),
        prompt(
            label="⏱ 60 s — non-maximum suppression",
            input="the same 128 images with suppression switched OFF",
            output="candidates per image before suppression, after suppression at IoU 0.5, and the true object count",
            constraint="suppress the SAME candidate pool — comparing against the stock pipeline's output would compare two different pipelines, since it caps at 100 detections after its own NMS, and the drop would look smaller than it is",
            check="`batched_nms`, not `nms` — suppression must be per class, or a person standing in front of a car suppresses the car."),
        code('''
from torchvision.ops import batched_nms

raw_model = fasterrcnn_resnet50_fpn(
    weights=weights, box_score_thresh=0.05, box_nms_thresh=1.0,
    box_detections_per_img=300).eval().to(DEVICE)

t0 = time.time()
raw_preds = {}
with torch.inference_mode():
    for im in images:
        img = Image.open(IMG_DIR / im["file_name"]).convert("RGB")
        out = raw_model([preprocess(img).to(DEVICE)])[0]
        raw_preds[im["id"]] = {k: v.cpu().numpy() for k, v in out.items()}
print(f"{time.time() - t0:.0f} s")

n_raw = np.mean([len(p["scores"]) for p in raw_preds.values()])

# Suppress the SAME candidate pool, so the two rows are comparable. Comparing
# against the stock pipeline's output instead would compare two different
# pipelines — it caps at 100 detections after its own NMS — and the drop would
# look smaller than it is.
n_sup = np.mean([
    len(batched_nms(torch.tensor(p["boxes"]), torch.tensor(p["scores"]),
                    torch.tensor(p["labels"]), 0.5))
    for p in raw_preds.values()])

print(f"\\ncandidate boxes per image, no suppression : {n_raw:.1f}")
print(f"the same pool after NMS at IoU 0.5        : {n_sup:.1f}")
print(f"actual objects per image                  : {n_true.mean():.2f}")
'''),
        prompt(
            label="another knob nobody set",
            input="five NMS thresholds",
            output="boxes kept per image at each",
            constraint="apply the rule by hand at several thresholds rather than trusting the default",
            check="count the defaults in this notebook: score 0.05, NMS 0.5, 100 detections per image, IoU 0.5 for matching. Four numbers nobody in the room chose."),
        code('''
# the rule, applied by hand at several thresholds
for t in [0.1, 0.3, 0.5, 0.7, 0.9]:
    kept = []
    for iid, p in raw_preds.items():
        keep = batched_nms(torch.tensor(p["boxes"]),
                           torch.tensor(p["scores"]),
                           torch.tensor(p["labels"]), t).numpy()
        kept.append(len(keep))
    print(f"NMS IoU {t:.1f} -> {np.mean(kept):6.1f} boxes per image")

print("\\nToo low and two people standing close together become one person.")
print("Too high and every object keeps its duplicates. It is another knob.")
'''),

        # ------------------------------------------------ segmentation
        md("""
## 12 · Per-pixel prediction

A box was always an approximation: a bicycle's box is mostly not bicycle. Ask
instead for a label on every pixel.

* **Semantic** segmentation: one class per pixel. Two people standing together
  are one `person` region and you cannot count them.
* **Instance** segmentation: one mask per object. You can.

Mask R-CNN is Faster R-CNN with one extra head. Two lines away from what you
already ran.

⏱ **about 20 seconds**, including the weight download the first time.
"""),
        prompt(
            label="⏱ 20 s — per-pixel prediction",
            input="the most crowded of the first forty images",
            output="soft masks, labels and scores, and how many objects survive a 0.7 cut",
            constraint="`masks` is (N, 1, H, W) and SOFT in [0,1] — indexing it as (N, H, W) silently gives you the first object, repeated",
            check="a box was always an approximation — a bicycle's box is mostly not bicycle. That is the argument for per-pixel prediction, and it is visible in the figure."),
        code('''
from torchvision.models.detection import (
    maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights)

mw = MaskRCNN_ResNet50_FPN_Weights.COCO_V1
mask_model = maskrcnn_resnet50_fpn(weights=mw).eval().to(DEVICE)
print("torchvision's reported scores on all 5,000 val2017 images:",
      mw.meta["_metrics"])

# The most crowded of the first forty images — the same one the lecture's
# figure uses, so the notebook and the slide show the same picture.
busy = max(images[:40], key=lambda im: len(gt[im["id"]]["labels"]))["id"]
im = next(i for i in images if i["id"] == busy)
pic = Image.open(IMG_DIR / im["file_name"]).convert("RGB")

with torch.inference_mode():
    out = mask_model([mw.transforms()(pic).to(DEVICE)])[0]

masks = out["masks"][:, 0].cpu().numpy()
labels = out["labels"].cpu().numpy()
scores = out["scores"].cpu().numpy()
print(f"\\nmasks shape {out['masks'].shape}  -- (N, 1, H, W), soft in [0, 1]")
keep = np.flatnonzero(scores >= 0.7)
print(f"{len(keep)} objects at score >= 0.7, "
      f"{len(set(labels[keep]))} distinct classes")
'''),
        prompt(
            label="semantic beside instance",
            input="the masks and their labels",
            output="the image, a semantic overlay and an instance overlay",
            constraint="colour by CLASS in one panel and by OBJECT in the other — that difference IS the distinction, and one panel cannot show it",
            check="`np.clip(img, 0, 1)` before imshow. Repeated alpha blending can drift outside the range, and matplotlib's response to that is to rescale the whole image."),
        code('''
base = np.asarray(pic, dtype=float) / 255.0
palette = plt.get_cmap("tab10")

inst = base.copy()
for k, j in enumerate(keep):
    inst = inst * (1 - 0.55 * masks[j][..., None]) \\
         + 0.55 * masks[j][..., None] * np.array(palette(k % 10)[:3])

sem, colour = base.copy(), {}
for j in keep:
    c = int(labels[j])
    colour.setdefault(c, np.array(palette(len(colour) % 10)[:3]))
    sem = sem * (1 - 0.55 * masks[j][..., None]) \\
        + 0.55 * masks[j][..., None] * colour[c]

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, img, ttl in zip(axes, [base, sem, inst],
                        ["the image",
                         f"semantic: {len(colour)} classes",
                         f"instance: {len(keep)} objects"]):
    ax.imshow(np.clip(img, 0, 1)); ax.set_title(ttl, fontsize=10, loc="left")
    ax.set_xticks([]); ax.set_yticks([])
plt.tight_layout(); plt.show()
'''),
        md("""
### 9.1 · And the annotation for that image

Look at what COCO actually recorded for it, and at what the detector found.
"""),
        prompt(
            label="and what the annotator recorded",
            input="the raw annotations for that image",
            output="what the detector found, beside what COCO annotates, with crowd regions marked",
            constraint="show the crowd regions explicitly — they are the reason the two counts disagree",
            check="when your model disagrees with the ground truth, look at the ground truth. It is a record of what somebody decided, not a record of what is there."),
        code('''
here = [a for a in raw["annotations"] if a["image_id"] == busy]
by = collections.Counter((cat_name[a["category_id"]], a["iscrowd"])
                         for a in here)
print(f"detector finds  {len(keep)} objects at score >= 0.7")
print("COCO annotates:")
for (nm, crowd), k in by.most_common():
    print(f"  {nm:10s} {k:3d}" + ("   (crowd region)" if crowd else ""))
print("\\nThe detector is probably right and the ground truth is probably not")
print("wrong. They answer different questions, and every metric in this")
print("lecture is measured against the annotator's decision.")
'''),

        md("""
## 13 · Where we are

- A label is not an answer when the question is *where* and *how many*.
- IoU is exactly zero for every pair of disjoint boxes, so as a loss it gives
  no gradient however far apart they are. GIoU and CIoU add a term that keeps
  moving.
- Average precision is defined with a **maximum** in it, and that maximum
  exists to repair the non-monotonicity of precision proved in Lecture 3. You
  computed both the sawtooth and its running maximum here.
- mAP is a mean over classes of a mean over IoU thresholds. Two averagings, and
  each hides something the one below it showed.

**Five questions to ask of any detection result:**

1. At what IoU threshold? A number without one is not a number.
2. At what score cut-off, and who chose it?
3. Was non-maximum suppression applied, and at what overlap?
4. Is the mean over images or over instances? They differ, and the difference
   grows with how unevenly the objects are distributed.
5. How many images? 128 is enough to see a shape and not enough to quote a
   figure to three decimals.

**Before the next lecture:** run this notebook top to bottom. Then move one
predicted box steadily away from its match and print IoU at each step. It sits
at exactly zero the whole way — the flat region the derivation predicts, and
the reason a plain IoU loss cannot train a detector.
"""),
    ]
    return cells
