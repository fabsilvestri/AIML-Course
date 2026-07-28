"""
Lecture 17 — Where is it, exactly?  (Build)

Application 9: run a pretrained detector over a stated 128-image subset of
COCO val2017, count the objects, and discover that counting cannot say how
wrong a box is.

Exports build() -> list[nbformat cell]. Self-contained: it downloads its own
data and imports everything it uses, so it does not depend on any other
notebook having been run first.

The corpus is 128 images and the notebook says so in three places. Full COCO
is about 20 GB and is never downloaded; only the official annotation file and
128 JPEGs are.
"""

from __future__ import annotations

import nbformat as nbf


def md(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(text.strip("\n"))


def code(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(text.strip("\n"))


HEADER = """
# Where is it, exactly?

**Lecture 17 · Build** · Géron, Chapter 12

Applications of Machine Learning — BSc Mathematics of Artificial Intelligence

---

**How to use this notebook.** Read before you run. The cell marked
**⚠ read before running** contains a defect on purpose, and it is the defect
this lecture is about: it runs, it prints a believable number, and the number
is wrong by a factor of nine.

**The corpus is 128 images.** COCO's `val2017` split is 5,000 images and the
full release is about 20 GB. Neither is downloaded here. Every number this
notebook prints is a measurement on 128 images, and you are expected to say
"128 images" whenever you quote one.
"""

SETUP = '''
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

        md("## 1 · Setup"), code(SETUP),
        md("## 2 · The corpus, and exactly how big it is"), code(DOWNLOAD),
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
        code(GROUND_TRUTH),

        md("""
### 2.2 · What is in it

`person` dominates. Remember that: in the next lecture we start averaging over
categories, and a mean over categories does not care that one of them is 39% of
the corpus.
"""),
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
## 4 · Commit

**Stop. On paper, now.** Not in this notebook — on paper, where you cannot
quietly revise it.

```
Metric:                                              ____________
Count MAE a useful shelf-audit system would need:    ____________
Count MAE I expect from what we build today:         ____________
```

You are not guessing in the dark: a system that never opens the image scores
6.02, and perfect is 0.00. Saying *where between them* is the exercise.
"""),

        md("""
## 5 · The detector

Nothing is trained here. These are the weights torchvision ships, trained on
COCO's training split by someone else, and this lecture is about evaluating
them rather than fitting them.

⏱ the weights are about 167 MB; the download happens once.
"""),
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

⏱ **1 to 2 minutes** on a GPU or an Apple Silicon MPS backend, and several
minutes on a CPU-only runtime. It varies with what else the machine is doing:
the same loop took 39 s on an idle laptop and 102 s on a busy one. No output
does not mean it has hung.
"""),
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
        code('''
p = preds[images[0]["id"]]
for k, v in p.items():
    print(f"{k:8s} {v.shape} {v.dtype}")

assert p["boxes"].shape[0] == p["labels"].shape[0] == p["scores"].shape[0]
assert np.all(np.diff(p["scores"]) <= 0), "not sorted by score"
print(f"\\nthis image has {len(p['boxes'])} boxes and "
      f"{len(gt[images[0]['id']]['labels'])} annotated objects")
'''),

        md("""
## 6 · An assistant writes the counting code

> *"Use a pretrained Faster R-CNN from torchvision to count how many objects
> are in each image in this folder, and print the mean absolute error against
> the COCO annotations."*

**⚠ Read before running.** Everything in that request is true and none of it is
wrong. One constraint is missing. Find it before you scroll.
"""),
        code('''
# the code the request returns — it runs, and it prints a plausible number
counts_naive = np.array([len(preds[im["id"]]["boxes"]) for im in images])

print(f"mean objects per image: {counts_naive.mean():.2f}")
print(f"count MAE: {count_mae(counts_naive, n_true):.2f}")
'''),
        md("""
### Reviewer question 5: what is the default I did not ask for?

`len(pred["boxes"])` is the number of rows the model chose to return. That is
everything above `box_score_thresh`, which defaults to **0.05**, capped at
`box_detections_per_img`, which defaults to **100**. It is not a count of
objects. It is a count of *candidates*.

Reviewer question 3 — *what is the shape here?* — finds it too: a shape of 88
for a photograph with twenty annotated objects in it should stop you.

**Now measure the damage.** Do not estimate it.
"""),
        code('''
scores = np.concatenate([preds[im["id"]]["scores"] for im in images])
print(f"boxes returned in total: {len(scores):,}")
print(f"  below score 0.10: {(scores < 0.10).sum():,}")
print(f"  at or above 0.50: {(scores >= 0.50).sum():,}")
print(f"  median score:     {np.median(scores):.3f}")

# the model's own cap, hit
print(f"\\nmost boxes returned for one image: {counts_naive.max()} "
      f"(box_detections_per_img is 100)")
'''),

        md("""
### The corrected specification

> *"… count how many objects are in each image, **keeping only detections whose
> score is at least a threshold I pass in**. Print the mean absolute error
> against the annotations, **and print the trivial baseline of one box per image
> beside it**. **Assert that the number kept never exceeds the number
> returned.**"*

Three additions: name the parameter, demand the baseline, assert the
relationship. The third is the one that fails loudly.
"""),
        code('''
def count_objects(pred, thresh):
    """Objects detected at or above `thresh`.

    There is deliberately no default: a count of objects is meaningless
    without saying which detections were counted.
    """
    keep = pred["scores"] >= thresh
    assert keep.sum() <= len(pred["scores"])
    return int(keep.sum())

THRESH = 0.5                     # chosen BEFORE looking at the error curve
n_pred = np.array([count_objects(preds[im["id"]], THRESH) for im in images])

mae = count_mae(n_pred, n_true)
bias = float((n_pred - n_true).mean())
print(f"count MAE  {mae:.2f}")
print(f"signed     {bias:+.2f}   (positive = too many boxes)")
print(f"baseline   {one_box:.2f}")
assert mae < one_box, "worse than predicting one box per image"
'''),
        md("""
The naive version was **27.51**, this one is **3.00**, and the baseline that
never opens the image is **6.02**. One missing line took the answer from "half
the error of a system that ignores the picture" to "four times worse than one".

Nothing raised. Nothing warned. The number just looked plausible.
"""),

        md("""
## 7 · The threshold is a knob, and nobody chose it

Sweep it and watch the answer to the stakeholder's question move by a factor of
ten.
"""),
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
## 8 · Look at the pictures, not only at the number

Three images with their predicted boxes. Labels are drawn only for confident
detections, because a crowded image stacks fourteen captions on top of each
other and an illegible figure teaches nothing.
"""),
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
wrong is that box". You cannot, and neither can the metric you committed to.
"""),
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
print("\\nYour committed metric scores both of them perfect.")
'''),

        md("""
## 9 · Propose the missing number

Whatever repairs this has to:

1. be 1 for identical boxes and 0 for boxes that do not touch
2. punish a box for being **too large**, or the whole-image baseline wins again
3. punish a box for being **too small**, or a one-pixel box in the right place
   wins
4. be dimensionless, so a 40-pixel cup and a 400-pixel sofa are on one scale

Write yours as a function of two corner-form boxes, and test it on two
identical boxes and on two that do not touch.
"""),
        code('''
def my_box_score(a, b):
    """Your formula. Replace the body.

    Requirements: 1 when a == b, 0 when disjoint, punishes both too-large and
    too-small, dimensionless.
    """
    raise NotImplementedError("this is yours to write")

same = np.array([0.0, 0.0, 100.0, 100.0])
away = np.array([300.0, 0.0, 400.0, 100.0])

try:
    print("identical:", my_box_score(same, same))
    print("disjoint: ", my_box_score(same, away))
except NotImplementedError as exc:
    print("not written yet —", exc)
    print("\\nBring your version to the next lecture. If it does something odd")
    print("for the disjoint pair, do NOT fix it. That is the interesting case.")
'''),

        md("""
## 10 · Where we are

| System | Count MAE, 128 images |
|---|---|
| One box per image | 6.02 |
| Every box the model returns | 27.51 |
| Faster R-CNN at score ≥ 0.5 | **3.00** |

Write **3.00** next to what you predicted, and keep the sheet.

Do not fix anything. Counting cannot distinguish nine right boxes from nine
wrong ones, and the repair is the next ninety minutes.
"""),
    ]
