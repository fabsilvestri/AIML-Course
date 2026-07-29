#!/usr/bin/env python3
"""
Application 9 — Lectures 17 and 18. Locating objects, and scoring the locations.

    python3 tools/figures_app09.py

Everything printed on slides/lecture-17.html and slides/lecture-18.html comes
from here, via figkit.export() into assets/figures/figures.json.

THE CORPUS, STATED ONCE AND STATED ON THE SLIDES
------------------------------------------------
COCO val2017 is 5,000 images; the full COCO release with train2017 is about
20 GB and is not downloaded here. This application uses

    * the 128 numerically lowest image ids of val2017  (about 20 MB of JPEG),
    * the official `instances_val2017.json` annotation file, which is the only
      way to have real ground-truth boxes at all.

128 images is not COCO. Every number below is a measurement on 128 images and
the decks say "128 images" beside each one. torchvision's own reported figure
for these weights on the full 5,000 — 37.0 box mAP — is quoted alongside ours
so the size of the sample is visible rather than implied.

The detectors are `fasterrcnn_resnet50_fpn` and `maskrcnn_resnet50_fpn` with
their COCO_V1 weights. Nothing is trained here; the lecture is about evaluation.

Timings are wall-clock on the machine that generated the figures — an Apple
Silicon laptop with an MPS backend and no CUDA. They are measurements, not
guarantees, and the decks say so.

WHY A LOCAL CACHE RATHER THAN figkit.cached
-------------------------------------------
`figkit.cached` writes one shared pickle that every application's script reads
and rewrites. Twelve applications are being built in parallel, so this one keeps
its own file next to its own data. Delete
/private/tmp/claude-501/aiml-data/coco-app09/fits-app09.pkl to recompute.
"""

from __future__ import annotations

import json
import pickle
import sys
import time
import urllib.request
import xml.dom.minidom
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter, NullFormatter

sys.path.insert(0, str(Path(__file__).resolve().parent))

from figkit import (setup, save, export, OUT, SEED,                  # noqa: E402
                    PRIMARY, ACCENT, SUCCESS, MATH, MUTED, RULE, AXIS,
                    BODY, SMALL, TICK, check_text_floor)

import torch                                                          # noqa: E402
import torchvision                                                    # noqa: E402
from torchvision.models.detection import (                            # noqa: E402
    fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights,
    maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights)

DATA = Path("/private/tmp/claude-501/aiml-data/coco-app09")
ANN_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
IMG_URL = "http://images.cocodataset.org/val2017/"

N_IMAGES = 128                # the corpus, stated on every slide that uses it
VAL_TOTAL = 5_000             # COCO val2017, for scale
SHOW_THRESH = 0.5             # the score cut-off the build lecture works at
IOU_THRESHOLDS = np.round(np.arange(0.50, 0.96, 0.05), 2)   # COCO's ten
TV_REPORTED_MAP = 37.0        # torchvision's own number, full val2017

DEVICE = ("mps" if torch.backends.mps.is_available()
          else "cuda" if torch.cuda.is_available() else "cpu")

_CACHE_FILE = DATA / "fits-app09.pkl"
_cache: dict = {}


def load_cache09() -> None:
    global _cache
    _cache = pickle.loads(_CACHE_FILE.read_bytes()) if _CACHE_FILE.is_file() else {}


def cached09(key, fn):
    if key in _cache:
        print(f"    [cached] {key}")
        return _cache[key]
    print(f"    [computing] {key}")
    value = fn()
    _cache[key] = value
    _CACHE_FILE.write_bytes(pickle.dumps(_cache))
    return value


# --------------------------------------------------------------- the corpus

def load_corpus() -> dict:
    """The 128 images, their ground-truth boxes, and the category table.

    Boxes are converted from COCO's [x, y, w, h] to the corner form
    [x1, y1, x2, y2] that torchvision emits, because mixing the two is the
    single commonest bug in this material and the lecture says so.
    """
    DATA.mkdir(parents=True, exist_ok=True)
    ann_path = DATA / "instances_val2017.json"
    if not ann_path.is_file():
        import zipfile, io
        print(f"  downloading {ANN_URL} (~241 MB, annotations only)")
        blob = urllib.request.urlopen(ANN_URL).read()
        with zipfile.ZipFile(io.BytesIO(blob)) as z:
            ann_path.write_bytes(z.read("annotations/instances_val2017.json"))

    raw = json.loads(ann_path.read_text())
    images = sorted(raw["images"], key=lambda i: i["id"])[:N_IMAGES]
    ids = {im["id"] for im in images}
    cat_name = {c["id"]: c["name"] for c in raw["categories"]}

    img_dir = DATA / "images"
    img_dir.mkdir(exist_ok=True)
    for im in images:
        p = img_dir / im["file_name"]
        if not p.is_file():
            urllib.request.urlretrieve(IMG_URL + im["file_name"], p)

    gt = {im["id"]: {"boxes": [], "labels": []} for im in images}
    n_crowd = 0
    for a in raw["annotations"]:
        if a["image_id"] not in ids:
            continue
        if a["iscrowd"]:
            # A crowd region is a single polygon covering many instances; it is
            # not one box and COCO's own evaluator ignores it. Eight of them in
            # this corpus. Dropping them is a choice, and the deck states it.
            n_crowd += 1
            continue
        x, y, w, h = a["bbox"]
        gt[a["image_id"]]["boxes"].append([x, y, x + w, y + h])
        gt[a["image_id"]]["labels"].append(a["category_id"])

    for iid in gt:
        gt[iid]["boxes"] = np.array(gt[iid]["boxes"], dtype=np.float64
                                    ).reshape(-1, 4)
        gt[iid]["labels"] = np.array(gt[iid]["labels"], dtype=np.int64)

    return {"images": images, "gt": gt, "cat_name": cat_name,
            "n_crowd": n_crowd, "img_dir": img_dir}


def _run_detector(corpus, *, score_thresh, nms_thresh, top_k):
    from PIL import Image
    weights = FasterRCNN_ResNet50_FPN_Weights.COCO_V1
    model = fasterrcnn_resnet50_fpn(weights=weights,
                                    box_score_thresh=score_thresh,
                                    box_nms_thresh=nms_thresh,
                                    box_detections_per_img=top_k)
    model.eval().to(DEVICE)
    tf = weights.transforms()
    out, t0 = {}, time.time()
    with torch.inference_mode():
        for im in corpus["images"]:
            img = Image.open(corpus["img_dir"] / im["file_name"]).convert("RGB")
            x = tf(img).to(DEVICE)
            p = model([x])[0]
            out[im["id"]] = {
                "boxes": p["boxes"].cpu().numpy().astype(np.float64),
                "labels": p["labels"].cpu().numpy().astype(np.int64),
                "scores": p["scores"].cpu().numpy().astype(np.float64)}
    return {"preds": out, "seconds": time.time() - t0}


def _run_masks(corpus, image_ids):
    from PIL import Image
    weights = MaskRCNN_ResNet50_FPN_Weights.COCO_V1
    model = maskrcnn_resnet50_fpn(weights=weights).eval().to(DEVICE)
    tf = weights.transforms()
    out = {}
    with torch.inference_mode():
        for iid in image_ids:
            im = next(i for i in corpus["images"] if i["id"] == iid)
            img = Image.open(corpus["img_dir"] / im["file_name"]).convert("RGB")
            p = model([tf(img).to(DEVICE)])[0]
            out[iid] = {"masks": p["masks"][:, 0].cpu().numpy().astype(np.float32),
                        "labels": p["labels"].cpu().numpy().astype(np.int64),
                        "scores": p["scores"].cpu().numpy().astype(np.float64)}
    return out


# ------------------------------------------------------ boxes and overlap

def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU between every box in a (N,4) and every box in b (M,4), corner form.

    The `clip(..., 0, None)` is the whole lecture in one call. Without it a
    disjoint pair produces a NEGATIVE width and a negative height, whose product
    is a positive "intersection", and the function returns a confident non-zero
    overlap for two boxes that do not touch. That bug runs, and it is Lecture
    18's worked assistant failure.
    """
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0.0, None)
    inter = wh[..., 0] * wh[..., 1]
    area_a = ((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]))[:, None]
    area_b = ((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]))[None, :]
    return inter / np.maximum(area_a + area_b - inter, 1e-12)


def iou_unclamped(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """The same function with the clamp removed. Kept so the deck can print
    what the defective version actually returns, rather than asserting it."""
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = rb - lt                                     # <-- no clip
    inter = wh[..., 0] * wh[..., 1]
    area_a = ((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]))[:, None]
    area_b = ((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]))[None, :]
    return inter / np.maximum(area_a + area_b - inter, 1e-12)


# ------------------------------------------------- precision, recall, AP

def pr_for_class(preds, gt, cls, t, score_min=0.0):
    """Ranked precision and recall for one class at one IoU threshold.

    Greedy matching in score order, against the best *unmatched* ground-truth
    box in the same image — which is what COCO does. Returns the raw,
    non-monotone precision, exactly as Lecture 4 left it.
    """
    n_gt, gt_by_img = 0, {}
    for iid, g in gt.items():
        m = g["labels"] == cls
        gt_by_img[iid] = g["boxes"][m]
        n_gt += int(m.sum())

    rows = []
    for iid, p in preds.items():
        m = (p["labels"] == cls) & (p["scores"] >= score_min)
        for b, s in zip(p["boxes"][m], p["scores"][m]):
            rows.append((float(s), iid, b))
    rows.sort(key=lambda r: -r[0])

    matched = {iid: np.zeros(len(g), bool) for iid, g in gt_by_img.items()}
    tp = np.zeros(len(rows))
    for k, (_s, iid, b) in enumerate(rows):
        g = gt_by_img[iid]
        free = ~matched[iid]
        if len(g) and free.any():
            ious = iou_matrix(b[None, :], g[free])[0]
            j = int(ious.argmax())
            if ious[j] >= t:
                tp[k] = 1.0
                matched[iid][np.flatnonzero(free)[j]] = True
    fp = 1.0 - tp
    ctp, cfp = tp.cumsum(), fp.cumsum()
    recall = ctp / max(n_gt, 1)
    precision = ctp / np.maximum(ctp + cfp, 1e-12)
    scores = np.array([r[0] for r in rows])
    return precision, recall, scores, n_gt


def envelope(precision: np.ndarray) -> np.ndarray:
    """The maximum precision at or above each recall level.

    Lecture 4 proved precision is not monotone in the threshold. This is the
    repair, and it is one line: sweep right to left taking running maxima.
    """
    out = precision.copy()
    for i in range(len(out) - 2, -1, -1):
        out[i] = max(out[i], out[i + 1])
    return out


def average_precision(precision, recall) -> float:
    """Area under the enveloped PR curve — the all-point definition."""
    if len(precision) == 0:
        return 0.0
    mrec = np.concatenate([[0.0], recall, [recall[-1]]])
    mpre = envelope(np.concatenate([[0.0], precision, [0.0]]))
    idx = np.flatnonzero(mrec[1:] != mrec[:-1])
    return float(((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]).sum())


def average_precision_101(precision, recall) -> float:
    """COCO's variant: the envelope sampled at 101 evenly spaced recalls."""
    if len(precision) == 0:
        return 0.0
    mpre = envelope(precision)
    q = np.linspace(0.0, 1.0, 101)
    idx = np.searchsorted(recall, q, side="left")
    vals = np.where(idx < len(mpre), mpre[np.clip(idx, 0, len(mpre) - 1)], 0.0)
    return float(vals.mean())


def reference_cocoeval(corpus, preds) -> dict | None:
    """Score the same predictions with the official COCO evaluator.

    Our AP is derived on the slides from its definition, so it has to be checked
    against something we did not write. `pycocotools` is an OPTIONAL import — it
    is not in requirements.txt and the course never asks a student to install
    it — so if it is absent this returns None and the previously measured values
    stay in figures.json. When it is present the two implementations are
    compared and the agreement is exported, because "we implemented AP and it
    looks about right" is exactly the claim this course refuses to accept.
    """
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError:
        print("    pycocotools absent — skipping the reference cross-check")
        return None
    import contextlib, io

    raw = json.loads((DATA / "instances_val2017.json").read_text())
    ids = {im["id"] for im in corpus["images"]}
    sub = {"info": raw.get("info", {}), "licenses": raw.get("licenses", []),
           "images": [i for i in raw["images"] if i["id"] in ids],
           "annotations": [a for a in raw["annotations"]
                           if a["image_id"] in ids and not a["iscrowd"]],
           "categories": raw["categories"]}
    sub_path = DATA / "sub_gt.json"
    sub_path.write_text(json.dumps(sub))

    dets = []
    for iid, p in preds.items():
        for b, l, s in zip(p["boxes"], p["labels"], p["scores"]):
            dets.append({"image_id": int(iid), "category_id": int(l),
                         "bbox": [float(b[0]), float(b[1]),
                                  float(b[2] - b[0]), float(b[3] - b[1])],
                         "score": float(s)})

    with contextlib.redirect_stdout(io.StringIO()):
        coco = COCO(str(sub_path))
        res = coco.loadRes(dets)
        ev = COCOeval(coco, res, "bbox")
        ev.evaluate(); ev.accumulate(); ev.summarize()
    st = ev.stats
    return {"map5095": float(st[0]), "map50": float(st[1]),
            "map75": float(st[2]), "small": float(st[3]),
            "medium": float(st[4]), "large": float(st[5]),
            "recall100": float(st[8])}


def classes_present(gt) -> list[int]:
    seen = set()
    for g in gt.values():
        seen.update(int(c) for c in g["labels"])
    return sorted(seen)


def map_at(preds, gt, t, classes) -> tuple[float, dict]:
    per = {}
    for c in classes:
        p, r, _s, n = pr_for_class(preds, gt, c, t)
        if n:
            per[c] = average_precision(p, r)
    return (float(np.mean(list(per.values()))) if per else 0.0), per


# ------------------------------------------------- the synthetic box pair

BOX_A = np.array([0.0, 0.0, 100.0, 100.0])
SEPARATIONS = np.arange(0, 301, 10, dtype=np.float64)


def _torch_iou(a, b):
    lt = torch.maximum(a[:2], b[:2])
    rb = torch.minimum(a[2:], b[2:])
    wh = torch.clamp(rb - lt, min=0.0)
    inter = wh[0] * wh[1]
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


def _torch_giou(a, b):
    iou = _torch_iou(a, b)
    lt = torch.minimum(a[:2], b[:2])
    rb = torch.maximum(a[2:], b[2:])
    wh = torch.clamp(rb - lt, min=0.0)
    area_c = wh[0] * wh[1]
    lt_i = torch.maximum(a[:2], b[:2])
    rb_i = torch.minimum(a[2:], b[2:])
    wh_i = torch.clamp(rb_i - lt_i, min=0.0)
    inter = wh_i[0] * wh_i[1]
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return iou - (area_c - union) / area_c


def _ciou_v(a, b):
    """CIoU's aspect-ratio term on its own. Zero when the shapes agree."""
    wa, ha = a[2] - a[0], a[3] - a[1]
    wb, hb = b[2] - b[0], b[3] - b[1]
    return (4 / torch.pi ** 2) * (torch.atan(wa / ha) - torch.atan(wb / hb)) ** 2


def _torch_ciou(a, b):
    """Complete IoU: IoU, minus a centre-distance term, minus an aspect term."""
    iou = _torch_iou(a, b)
    ca = torch.stack([(a[0] + a[2]) / 2, (a[1] + a[3]) / 2])
    cb = torch.stack([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2])
    rho2 = ((ca - cb) ** 2).sum()
    lt = torch.minimum(a[:2], b[:2])
    rb = torch.maximum(a[2:], b[2:])
    c2 = ((rb - lt) ** 2).sum()
    wa, ha = a[2] - a[0], a[3] - a[1]
    wb, hb = b[2] - b[0], b[3] - b[1]
    v = (4 / torch.pi ** 2) * (torch.atan(wa / ha) - torch.atan(wb / hb)) ** 2
    alpha = v / (1 - iou + v + 1e-12)
    return iou - rho2 / c2 - alpha * v


def separation_study() -> dict:
    """Two 100x100 boxes, pulled apart. The whole of thread 9, in one table.

    Every value and every gradient is taken from autograd on the same separation
    variable, so nothing here is a formula retyped by hand.
    """
    a = torch.tensor(BOX_A, dtype=torch.float64)
    rows = {k: [] for k in ("d", "iou", "giou", "ciou",
                            "g_iou", "g_giou", "g_ciou")}
    for d in SEPARATIONS:
        dv = torch.tensor(float(d), dtype=torch.float64, requires_grad=True)
        b = torch.stack([dv, torch.zeros_like(dv),
                         dv + 100.0, torch.full_like(dv, 100.0)])
        for name, fn in (("iou", _torch_iou), ("giou", _torch_giou),
                         ("ciou", _torch_ciou)):
            val = fn(a, b)
            g, = torch.autograd.grad(val, dv, retain_graph=False)
            rows[name].append(float(val.detach()))
            rows["g_" + name].append(float(g))
        rows["d"].append(float(d))
    return {k: np.array(v) for k, v in rows.items()}


# ---------------------------------------------------------------- figures

def fig_gt_class_counts(counts, path):
    names, vals = zip(*counts)
    fig, ax = plt.subplots(figsize=(11.5, 4.4))
    y = np.arange(len(names))[::-1]
    ax.barh(y, vals, color=PRIMARY, height=0.68)
    ax.set_yticks(y, names)
    ax.set_xlabel("ground-truth instances in the 128-image corpus")
    ax.set_title("person is half the corpus — a mean over classes will hide that")
    for yi, v in zip(y, vals):
        ax.text(v + 4, yi, str(v), va="center", color=MUTED, fontsize=TICK)
    ax.set_xlim(0, max(vals) * 1.16)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return save(fig, path)


def fig_score_hist(scores, kept, path):
    fig, ax = plt.subplots(figsize=(11.5, 4.4))
    bins = np.linspace(0.0, 1.0, 41)
    ax.hist(scores, bins=bins, color=PRIMARY, alpha=0.85,
            label=f"all {len(scores):,} boxes the model returns")
    ax.axvline(SHOW_THRESH, color=ACCENT, lw=2.5, ls="--")
    ax.annotate(f"score = {SHOW_THRESH}\n{kept:,} boxes survive",
                xy=(SHOW_THRESH, ax.get_ylim()[1] * 0.62),
                xytext=(0.60, ax.get_ylim()[1] * 0.72),
                color=ACCENT, fontsize=SMALL,
                bbox=dict(fc="white", ec=ACCENT, boxstyle="round,pad=0.35"),
                arrowprops=dict(arrowstyle="->", color=ACCENT, lw=2))
    ax.set_xlabel("detection score")
    ax.set_ylabel("boxes")
    ax.set_title("the detector's confidence, over 128 images")
    ax.legend(loc="upper center")
    fig.tight_layout()
    return save(fig, path)


def fig_count_vs_threshold(sweep, true_mean, path):
    ts, mean_pred, mae = sweep["t"], sweep["mean_pred"], sweep["mae"]
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.5))
    ax = axes[0]
    ax.plot(ts, mean_pred, color=PRIMARY, lw=2.5, marker="o", ms=4,
            label="predicted objects per image")
    ax.axhline(true_mean, color=SUCCESS, lw=2.5, ls="--",
               label=f"ground truth, {true_mean:.2f}")
    ax.set_xlabel("score threshold")
    ax.set_ylabel("mean objects per image")
    ax.set_title("the count is a function of a number nobody chose")
    ax.legend(loc="upper right")
    ax = axes[1]
    ax.plot(ts, mae, color=ACCENT, lw=2.5, marker="o", ms=4)
    k = int(np.argmin(mae))
    ax.plot([ts[k]], [mae[k]], "o", ms=11, mfc="none", mec=SUCCESS, mew=2.5)
    ax.annotate(f"best at {ts[k]:.2f}\nMAE {mae[k]:.2f}",
                xy=(ts[k], mae[k]), xytext=(ts[k] + 0.12, mae[k] + 2.0),
                color=SUCCESS, fontsize=SMALL,
                bbox=dict(fc="white", ec=SUCCESS, boxstyle="round,pad=0.35"),
                arrowprops=dict(arrowstyle="->", color=SUCCESS, lw=2))
    ax.set_xlabel("score threshold")
    ax.set_ylabel("mean absolute count error")
    ax.set_title("and so is the error")
    fig.tight_layout()
    return save(fig, path)


def fig_count_scatter(true_counts, pred_counts, path):
    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    hi = max(true_counts.max(), pred_counts.max()) + 2
    ax.plot([0, hi], [0, hi], color=MUTED, lw=1.6, ls="--", zorder=1)
    ax.scatter(true_counts, pred_counts, s=52, color=PRIMARY, alpha=0.55,
               edgecolor="white", linewidth=0.8, zorder=3)
    ax.set_xlim(0, hi); ax.set_ylim(0, hi)
    ax.set_xlabel("ground-truth objects in the image")
    ax.set_ylabel(f"objects detected at score \u2265 {SHOW_THRESH}")
    ax.set_title("128 images; points below the line are objects missed")
    below = int((pred_counts < true_counts).sum())
    ax.text(0.04, 0.94, f"{below} of {len(true_counts)} images undercounted",
            transform=ax.transAxes, va="top", color=ACCENT, fontsize=SMALL,
            bbox=dict(fc="white", ec=ACCENT, boxstyle="round,pad=0.35"))
    fig.tight_layout()
    return save(fig, path)


def _draw_boxes(ax, boxes, labels, scores, names, color, *, lw=2.2,
                show_score=True, label_above=1.0, fontsize=12):
    """Draw every box; label only the boxes scoring at or above `label_above`.

    Labelling all of them was unreadable — the crowded images stack fourteen
    captions on top of each other, and an illegible figure teaches nothing even
    when the mess it depicts is the point.
    """
    order = np.argsort(-scores) if scores is not None else range(len(boxes))
    for k in order:
        x1, y1, x2, y2 = boxes[k]
        ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False,
                               edgecolor=color, linewidth=lw))
        if labels is None:
            continue
        if scores is not None and scores[k] < label_above:
            continue
        tag = names.get(int(labels[k]), str(labels[k]))
        if show_score and scores is not None:
            tag += f" {scores[k]:.2f}"
        ax.text(x1 + 2, max(y1 - 4, 12), tag, color="white", fontsize=fontsize,
                bbox=dict(fc=color, ec="none", pad=1.4))


def _plural(n, word):
    return f"{n} {word}" + ("" if n == 1 else "es" if word.endswith("x")
                            else "s")


def fig_detection_grid(corpus, preds, image_ids, path, *, thresh=SHOW_THRESH,
                       label_above=0.90, title="", height=4.2):
    from PIL import Image
    n = len(image_ids)
    fig, axes = plt.subplots(1, n, figsize=(4.6 * n, height))
    axes = np.atleast_1d(axes)
    for ax, iid in zip(axes, image_ids):
        im = next(i for i in corpus["images"] if i["id"] == iid)
        ax.imshow(Image.open(corpus["img_dir"] / im["file_name"]).convert("RGB"))
        p = preds[iid]
        m = p["scores"] >= thresh
        _draw_boxes(ax, p["boxes"][m], p["labels"][m], p["scores"][m],
                    corpus["cat_name"], ACCENT, label_above=label_above)
        ax.set_title(f"{_plural(int(m.sum()), 'box')}   (truth: "
                     f"{len(corpus['gt'][iid]['labels'])})",
                     fontsize=TICK, color=MUTED, loc="left")
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    if title:
        fig.suptitle(title, fontsize=BODY, color=MUTED, x=0.01, ha="left")
    fig.tight_layout()
    return save(fig, path, raster=True)


def fig_wholebox(corpus, iid, path):
    from PIL import Image
    im = next(i for i in corpus["images"] if i["id"] == iid)
    pic = Image.open(corpus["img_dir"] / im["file_name"]).convert("RGB")
    W, H = pic.size
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.0))
    for ax in axes:
        ax.imshow(pic); ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    g = corpus["gt"][iid]
    _draw_boxes(axes[0], g["boxes"], g["labels"], None, corpus["cat_name"],
                SUCCESS, show_score=False)
    axes[0].set_title(f"ground truth: {len(g['labels'])} objects",
                      fontsize=TICK, color=MUTED, loc="left")
    axes[1].add_patch(Rectangle((0, 0), W - 1, H - 1, fill=False,
                                edgecolor=ACCENT, linewidth=4.0))
    axes[1].set_title("the baseline: one box, the whole image",
                      fontsize=TICK, color=MUTED, loc="left")
    axes[1].text(W * 0.5, H * 0.5, "it overlaps\nevery object",
                 color=ACCENT, fontsize=13, ha="center", va="center",
                 bbox=dict(fc="white", ec=ACCENT, boxstyle="round,pad=0.4"))
    fig.tight_layout()
    return save(fig, path, raster=True)


def fig_iou_flat(sep, path):
    d, iou, giou = sep["d"], sep["iou"], sep["giou"]
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.5))
    ax = axes[0]
    ax.plot(d, iou, color=ACCENT, lw=3.0, marker="o", ms=4, label="IoU")
    ax.axvline(100, color=MUTED, lw=1.4, ls=":")
    ax.set_xlabel("separation of two 100 \u00d7 100 boxes (pixels)")
    ax.set_ylabel("IoU")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("past 100 px the boxes are disjoint and IoU is exactly 0")
    ax.annotate("flat at zero for every separation beyond 100",
                xy=(210, 0.0), xytext=(115, 0.42), color=ACCENT, fontsize=SMALL,
                bbox=dict(fc="white", ec=ACCENT, boxstyle="round,pad=0.35"),
                arrowprops=dict(arrowstyle="->", color=ACCENT, lw=2))
    ax = axes[1]
    ax.plot(d, sep["g_iou"], color=ACCENT, lw=3.0, marker="o", ms=4,
            label="d(IoU) / d(separation)")
    ax.plot(d, sep["g_giou"], color=SUCCESS, lw=2.6, marker="s", ms=4,
            label="d(GIoU) / d(separation)")
    ax.axhline(0, color=MUTED, lw=1.2)
    ax.axvline(100, color=MUTED, lw=1.4, ls=":")
    ax.set_xlabel("separation (pixels)")
    ax.set_ylabel("gradient, from autograd")
    ax.set_title("no gradient means nothing to descend")
    ax.legend(loc="lower right")
    fig.tight_layout()
    return save(fig, path)


def fig_clamp_bug(d, broken, correct, path):
    fig, ax = plt.subplots(figsize=(11.5, 4.6))
    ax.plot(d, correct, color=SUCCESS, lw=3.0, marker="o", ms=5,
            label="IoU, with the clamp")
    ax.plot(d, broken, color=ACCENT, lw=3.0, ls="--", marker="s", ms=5,
            label="IoU, clamp removed")
    ax.axvline(100, color=MUTED, lw=1.4, ls=":")
    ax.set_xlabel("diagonal separation of two 100 \u00d7 100 boxes (pixels)")
    ax.set_ylabel("reported overlap")
    ax.set_title("the boxes are disjoint beyond 100 px; one curve does not know")
    ax.legend(loc="upper left")
    ax.annotate("it goes UP as they move apart",
                xy=(180, broken[d == 180][0]), xytext=(105, 0.42),
                color=ACCENT, fontsize=SMALL,
                bbox=dict(fc="white", ec=ACCENT, boxstyle="round,pad=0.35"),
                arrowprops=dict(arrowstyle="->", color=ACCENT, lw=2))
    fig.tight_layout()
    return save(fig, path)


def fig_giou_ciou(sep, path):
    d = sep["d"]
    fig, ax = plt.subplots(figsize=(11.5, 4.6))
    ax.plot(d, sep["iou"], color=ACCENT, lw=3.0, label="IoU")
    ax.plot(d, sep["giou"], color=SUCCESS, lw=2.6, ls="--", label="GIoU")
    ax.plot(d, sep["ciou"], color=MATH, lw=2.6, ls=":", label="CIoU")
    ax.axhline(0, color=MUTED, lw=1.2)
    ax.axvline(100, color=MUTED, lw=1.4, ls=":")
    ax.set_xlabel("separation of two 100 \u00d7 100 boxes (pixels)")
    ax.set_ylabel("value")
    ax.set_title("both repairs keep falling where IoU has stopped")
    ax.legend(loc="upper right")
    ax.annotate("IoU stops here", xy=(100, 0.0), xytext=(28, -0.55),
                color=ACCENT, fontsize=SMALL,
                bbox=dict(fc="white", ec=ACCENT, boxstyle="round,pad=0.35"),
                arrowprops=dict(arrowstyle="->", color=ACCENT, lw=2))
    fig.tight_layout()
    return save(fig, path)


def fig_iou_examples(pairs, path):
    fig, axes = plt.subplots(1, len(pairs), figsize=(3.1 * len(pairs), 2.9))
    for ax, (title, a, b, val) in zip(np.atleast_1d(axes), pairs):
        ax.add_patch(Rectangle((a[0], a[1]), a[2] - a[0], a[3] - a[1],
                               fill=False, edgecolor=SUCCESS, lw=2.8))
        ax.add_patch(Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1],
                               fill=False, edgecolor=ACCENT, lw=2.8, ls="--"))
        lt = np.maximum(a[:2], b[:2]); rb = np.minimum(a[2:], b[2:])
        if (rb > lt).all():
            ax.add_patch(Rectangle((lt[0], lt[1]), rb[0] - lt[0], rb[1] - lt[1],
                                   facecolor=MATH, alpha=0.30, edgecolor="none"))
        x0 = min(a[0], b[0]) - 20; x1 = max(a[2], b[2]) + 20
        y0 = min(a[1], b[1]) - 20; y1 = max(a[3], b[3]) + 20
        span = max(x1 - x0, y1 - y0)
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        ax.set_xlim(cx - span / 2, cx + span / 2)
        ax.set_ylim(cy + span / 2, cy - span / 2)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(f"{title}\nIoU = {val:.3f}", fontsize=TICK, color=MUTED,
                     loc="left")
    fig.tight_layout()
    return save(fig, path)


def fig_pr_sawtooth(precision, recall, ap, cls_name, n_gt, path):
    env = envelope(precision)
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.6))
    ax = axes[0]
    ax.step(recall, precision, where="post", color=ACCENT, lw=2.2,
            label="precision as measured")
    ax.step(recall, env, where="post", color=SUCCESS, lw=2.8,
            label="max precision at or above this recall")
    ax.fill_between(recall, 0, env, step="post", color=SUCCESS, alpha=0.13)
    ax.set_xlabel("recall"); ax.set_ylabel("precision")
    ax.set_ylim(0, 1.05); ax.set_xlim(0, 1.0)
    ax.set_title(f"{cls_name}, IoU \u2265 0.5, {n_gt} ground-truth instances")
    ax.legend(loc="lower left")
    ax.text(0.97, 0.94, f"AP = {ap:.3f}", transform=ax.transAxes, ha="right",
            va="top", color=SUCCESS, fontsize=BODY,
            bbox=dict(fc="white", ec=SUCCESS, boxstyle="round,pad=0.35"))
    ax = axes[1]
    lo, hi = 50, min(len(precision), 140)
    k = np.arange(lo + 1, hi + 1)
    ax.plot(k, precision[lo:hi], color=ACCENT, lw=2.2, marker="o", ms=3.5,
            label="precision")
    ax.step(k, env[lo:hi], where="post", color=SUCCESS, lw=2.6,
            label="the maximum at or above")
    down = int((np.diff(precision[lo:hi]) < -1e-12).sum())
    up = int((np.diff(precision[lo:hi]) > 1e-12).sum())
    ax.set_xlabel("detections accepted, in score order")
    ax.set_ylabel("precision")
    ax.set_title(f"steps {lo + 1}–{hi}: {down} down, {up} up — "
                 f"Lecture 4's sawtooth")
    ax.legend(loc="lower left")
    fig.tight_layout()
    return save(fig, path)


def fig_ap_by_iou(ts, aps, map_all, path):
    fig, ax = plt.subplots(figsize=(11.5, 4.5))
    ax.plot(ts, aps, color=PRIMARY, lw=3.0, marker="o", ms=6)
    ax.axhline(map_all, color=MATH, lw=2.4, ls="--",
               label=f"mean over the ten thresholds = {map_all:.3f}")
    ax.set_xlabel("IoU threshold at which a detection counts as correct")
    ax.set_ylabel("mAP over classes present")
    ax.set_title("the second mean: ten thresholds, collapsed to one number")
    ax.legend(loc="upper right")
    ax.annotate(f"mAP@0.50 = {aps[0]:.3f}", xy=(ts[0], aps[0]),
                xytext=(0.56, aps[0] * 0.96), color=PRIMARY, fontsize=SMALL,
                bbox=dict(fc="white", ec=PRIMARY, boxstyle="round,pad=0.35"),
                arrowprops=dict(arrowstyle="->", color=PRIMARY, lw=2))
    ax.annotate(f"mAP@0.95 = {aps[-1]:.3f}", xy=(ts[-1], aps[-1]),
                xytext=(0.70, aps[0] * 0.45), color=ACCENT, fontsize=SMALL,
                bbox=dict(fc="white", ec=ACCENT, boxstyle="round,pad=0.35"),
                arrowprops=dict(arrowstyle="->", color=ACCENT, lw=2))
    fig.tight_layout()
    return save(fig, path)


def fig_per_class_ap(rows, counts, mean_ap, path):
    """All 73 classes, then AP against how many instances the class had.

    Showing only the top of the ranking made every bar sit above the mean and
    hid the very thing the slide is about. The right-hand panel is the actual
    finding: the perfect scores belong to the classes with one or two
    instances, and each of them weighs exactly as much as `person`.
    """
    names = [r[0] for r in rows]
    vals = np.array([r[1] for r in rows])
    n = np.array([counts[r[0]] for r in rows], dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.7))
    ax = axes[0]
    x = np.arange(len(vals))
    ax.bar(x, vals, color=[SUCCESS if v >= mean_ap else ACCENT for v in vals],
           width=0.9)
    ax.axhline(mean_ap, color=MATH, lw=2.4, ls="--",
               label=f"mAP@0.50 = {mean_ap:.3f}")
    ax.set_xlabel(f"all {len(vals)} classes present, best to worst")
    ax.set_ylabel("AP at IoU \u2265 0.5")
    ax.set_ylim(0, 1.05)
    ax.set_xticks([])
    n_perfect = int((vals == 1.0).sum())
    ax.set_title(f"{n_perfect} classes score exactly 1.000")
    ax.legend(loc="upper right")
    ax.grid(axis="x", visible=False)

    ax = axes[1]
    ax.scatter(n, vals, s=58, color=PRIMARY, alpha=0.55, edgecolor="white",
               linewidth=0.8)
    ax.axhline(mean_ap, color=MATH, lw=2.0, ls="--")
    ax.set_xscale("log")
    # parse_math is off course-wide (TRICKS 9.1), so matplotlib's default log
    # tick labels would print as a literal "$\\mathdefault{10^{0}}$"
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("ground-truth instances of that class, in 128 images")
    ax.set_ylabel("AP at IoU \u2265 0.5")
    ax.set_ylim(-0.05, 1.08)
    ax.set_title("a perfect score is usually a class with one instance")
    k = int(np.argmax(n))
    ax.annotate(f"{names[k]}: {int(n[k])} instances, AP {vals[k]:.2f}",
                xy=(n[k], vals[k]), xytext=(3.0, 0.24), color=PRIMARY,
                fontsize=SMALL,
                bbox=dict(fc="white", ec=PRIMARY, boxstyle="round,pad=0.35"),
                arrowprops=dict(arrowstyle="->", color=PRIMARY, lw=2))
    fig.tight_layout()
    return save(fig, path)


def fig_nms_sweep(sweep, path):
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.5))
    ax = axes[0]
    ax.plot(sweep["t"], sweep["boxes"], color=PRIMARY, lw=2.8, marker="o", ms=5)
    ax.set_xlabel("NMS IoU threshold")
    ax.set_ylabel("boxes kept per image")
    ax.set_title("suppression, as a function of one number")
    ax = axes[1]
    ax.plot(sweep["t"], sweep["map50"], color=SUCCESS, lw=2.8, marker="o", ms=5)
    k = int(np.argmax(sweep["map50"]))
    ax.plot([sweep["t"][k]], [sweep["map50"][k]], "o", ms=12, mfc="none",
            mec=MATH, mew=2.5)
    ax.annotate(f"best at {sweep['t'][k]:.2f}", xy=(sweep["t"][k], sweep["map50"][k]),
                xytext=(sweep["t"][k] - 0.02, sweep["map50"][k] - 0.09),
                color=MATH, fontsize=SMALL, ha="center",
                bbox=dict(fc="white", ec=MATH, boxstyle="round,pad=0.35"),
                arrowprops=dict(arrowstyle="->", color=MATH, lw=2))
    ax.set_xlabel("NMS IoU threshold")
    ax.set_ylabel("mAP@0.50")
    ax.set_title("and the score it buys, on 128 images")
    fig.tight_layout()
    return save(fig, path)


def fig_nms_before_after(corpus, raw, iid, path, *, nms_t=0.5, score_t=0.5):
    from PIL import Image
    from torchvision.ops import batched_nms
    im = next(i for i in corpus["images"] if i["id"] == iid)
    pic = Image.open(corpus["img_dir"] / im["file_name"]).convert("RGB")
    p = raw[iid]
    m = p["scores"] >= score_t
    boxes, labels, scores = p["boxes"][m], p["labels"][m], p["scores"][m]
    keep = batched_nms(torch.tensor(boxes), torch.tensor(scores),
                       torch.tensor(labels), nms_t).numpy()
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.0))
    for ax, (bx, lb, sc, ttl) in zip(axes, [
            (boxes, labels, scores, f"no suppression: {len(boxes)} boxes"),
            (boxes[keep], labels[keep], scores[keep],
             f"NMS at IoU {nms_t}: {len(keep)} boxes")]):
        ax.imshow(pic)
        _draw_boxes(ax, bx, lb, sc, corpus["cat_name"], ACCENT, lw=1.8,
                    label_above=2.0)
        ax.set_title(ttl, fontsize=TICK, color=MUTED, loc="left")
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    fig.tight_layout()
    return save(fig, path, raster=True)


def fig_masks(corpus, masks, iid, path, *, thresh=0.7):
    from PIL import Image
    im = next(i for i in corpus["images"] if i["id"] == iid)
    pic = np.asarray(Image.open(corpus["img_dir"] / im["file_name"]
                                ).convert("RGB")).astype(np.float64) / 255.0
    m = masks[iid]
    keep = np.flatnonzero(m["scores"] >= thresh)
    palette = plt.get_cmap("tab10")

    inst = pic.copy()
    for k, j in enumerate(keep):
        col = np.array(palette(k % 10)[:3])
        soft = m["masks"][j][..., None]
        inst = inst * (1 - 0.55 * soft) + 0.55 * soft * col

    sem, class_colour = pic.copy(), {}
    for j in keep:
        c = int(m["labels"][j])
        class_colour.setdefault(c, np.array(palette(len(class_colour) % 10)[:3]))
        soft = m["masks"][j][..., None]
        sem = sem * (1 - 0.55 * soft) + 0.55 * soft * class_colour[c]

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.6))
    for ax, img, ttl in zip(axes, [pic, sem, inst], [
            "the image",
            f"semantic: {len(class_colour)} classes, one colour each",
            f"instance: {len(keep)} objects, one colour each"]):
        ax.imshow(np.clip(img, 0, 1))
        ax.set_title(ttl, fontsize=TICK, color=MUTED, loc="left")
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    fig.tight_layout()
    return save(fig, path, raster=True)


def fig_baseline_bars(rows, path):
    names = [r[0] for r in rows]
    vals = [r[1] for r in rows]
    good = [r[2] for r in rows]
    fig, ax = plt.subplots(figsize=(11.0, 4.2))
    x = np.arange(len(names))
    ax.bar(x, vals, color=[SUCCESS if g else ACCENT for g in good], width=0.6)
    for xi, v in zip(x, vals):
        ax.text(xi, v + max(vals) * 0.02, f"{v:.2f}", ha="center",
                color=MUTED, fontsize=TICK)
    ax.set_xticks(x, names)
    ax.set_ylabel("mean absolute error in the object count")
    ax.set_ylim(0, max(vals) * 1.2)
    ax.set_title("counting objects: three systems, 128 images")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    return save(fig, path)


# ------------------------------------------------------------------- main

def main() -> int:
    setup()
    load_cache09()
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    print("Corpus…")
    corpus = load_corpus()
    gt, names = corpus["gt"], corpus["cat_name"]
    counts_true = np.array([len(gt[im["id"]]["labels"]) for im in corpus["images"]])
    present = classes_present(gt)

    import collections
    freq = collections.Counter()
    for g in gt.values():
        for c in g["labels"]:
            freq[names[int(c)]] += 1

    print(f"  {N_IMAGES} images, {int(counts_true.sum())} ground-truth boxes, "
          f"{len(present)} classes, {corpus['n_crowd']} crowd regions dropped")

    print("Detector (stock settings)…")
    run = cached09("l09_fasterrcnn_stock",
                   lambda: _run_detector(corpus, score_thresh=0.05,
                                         nms_thresh=0.5, top_k=100))
    preds, infer_s = run["preds"], run["seconds"]

    print("Detector (suppression disabled, for the NMS study)…")
    raw = cached09("l09_fasterrcnn_nonms",
                   lambda: _run_detector(corpus, score_thresh=0.05,
                                         nms_thresh=1.0, top_k=300))["preds"]

    # ---------------------------------------------------- Lecture 17 numbers
    all_scores = np.concatenate([p["scores"] for p in preds.values()])
    counts_raw = np.array([len(preds[im["id"]]["scores"])
                           for im in corpus["images"]])
    counts_50 = np.array([int((preds[im["id"]]["scores"] >= SHOW_THRESH).sum())
                          for im in corpus["images"]])

    mae = lambda a, b: float(np.abs(a - b).mean())
    ts = np.round(np.arange(0.05, 0.96, 0.05), 2)
    sweep_mean, sweep_mae = [], []
    for t in ts:
        c = np.array([int((preds[im["id"]]["scores"] >= t).sum())
                      for im in corpus["images"]])
        sweep_mean.append(float(c.mean()))
        sweep_mae.append(mae(c, counts_true))
    sweep = {"t": ts, "mean_pred": np.array(sweep_mean),
             "mae": np.array(sweep_mae)}
    best_k = int(np.argmin(sweep["mae"]))

    # The trivial baseline: one box per image, covering the whole image.
    # Under "does the box overlap the object at all?" it is perfect, which is
    # the entire reason Lecture 18 needs a metric that punishes size.
    whole_hits, whole_total = 0, 0
    whole_ious = []
    for im in corpus["images"]:
        g = gt[im["id"]]["boxes"]
        if not len(g):
            continue
        whole = np.array([[0.0, 0.0, float(im["width"]), float(im["height"])]])
        lt = np.maximum(whole[:, None, :2], g[None, :, :2])
        rb = np.minimum(whole[:, None, 2:], g[None, :, 2:])
        wh = np.clip(rb - lt, 0.0, None)
        whole_hits += int((wh[..., 0] * wh[..., 1] > 0).sum())
        whole_total += len(g)
        whole_ious.extend(iou_matrix(whole, g)[0].tolist())
    whole_ious = np.array(whole_ious)

    baseline_counts = np.ones_like(counts_true)

    export(
        l17_n_images=N_IMAGES,
        l17_val_total=VAL_TOTAL,
        l17_n_gt=int(counts_true.sum()),
        l17_n_crowd=int(corpus["n_crowd"]),
        l17_n_classes_present=len(present),
        l17_n_coco_classes=80,
        l17_gt_per_image={"mean": float(counts_true.mean()),
                          "median": float(np.median(counts_true)),
                          "min": int(counts_true.min()),
                          "max": int(counts_true.max())},
        l17_class_counts={k: int(v) for k, v in freq.most_common(12)},
        l17_person_share=float(freq["person"] / counts_true.sum()),
        l17_device=DEVICE,
        l17_infer_seconds=float(infer_s),
        l17_infer_per_image=float(infer_s / N_IMAGES),
        l17_raw_boxes={"total": int(counts_raw.sum()),
                       "mean": float(counts_raw.mean()),
                       "max": int(counts_raw.max()),
                       "mae": mae(counts_raw, counts_true)},
        l17_at_50={"total": int(counts_50.sum()),
                   "mean": float(counts_50.mean()),
                   "max": int(counts_50.max()),
                   "mae": mae(counts_50, counts_true),
                   "bias": float((counts_50 - counts_true).mean()),
                   "undercounted": int((counts_50 < counts_true).sum()),
                   "overcounted": int((counts_50 > counts_true).sum()),
                   "exact": int((counts_50 == counts_true).sum())},
        l17_threshold_sweep={"t": ts.tolist(),
                             "mean_pred": sweep["mean_pred"].tolist(),
                             "mae": sweep["mae"].tolist()},
        l17_best_threshold=float(ts[best_k]),
        l17_best_mae=float(sweep["mae"][best_k]),
        # "predict the corpus mean, every image" — the detection analogue of
        # predicting the mean in Lecture 1. Rounded to an integer because a
        # count of objects is an integer.
        l17_baseline_mean_count=mae(
            np.full_like(counts_true, int(round(counts_true.mean()))),
            counts_true),
        l17_failure_cost=abs(mae(counts_raw, counts_true)
                             - mae(counts_50, counts_true)),
        l17_baseline_one_box={"mae": mae(baseline_counts, counts_true),
                              "overlap_recall": float(whole_hits / whole_total),
                              "mean_iou": float(whole_ious.mean()),
                              "max_iou": float(whole_ious.max()),
                              "iou_above_half": int((whole_ious >= 0.5).sum())},
        l17_score_quartiles={"q25": float(np.quantile(all_scores, 0.25)),
                             "median": float(np.median(all_scores)),
                             "q75": float(np.quantile(all_scores, 0.75)),
                             "above_half": int((all_scores >= 0.5).sum()),
                             "below_tenth": int((all_scores < 0.1).sum())},
    )

    # ---------------------------------------------------- Lecture 18 numbers
    print("Separation study (synthetic boxes)…")
    sep = separation_study()
    first_zero = int(np.flatnonzero(sep["iou"] <= 0.0)[0])
    # strictly greater: at exactly 100 the two boxes share an edge, the
    # intersection has zero area, and autograd still returns the one-sided
    # derivative from the overlapping side. The claim being made is about the
    # interior of the disjoint region.
    disjoint = sep["d"] > 100

    # The clamp bug, pushed until it is undeniable. Separate the two boxes
    # DIAGONALLY and both differences go negative, so their product is a
    # positive "intersection" that grows with the distance. The defective
    # function therefore reports MORE overlap the further apart the boxes are.
    a = np.array([[0.0, 0.0, 100.0, 100.0]])
    diag_d = np.arange(0, 201, 10, dtype=np.float64)
    bug_diag, ok_diag = [], []
    for dd in diag_d:
        b = np.array([[dd, dd, dd + 100.0, dd + 100.0]])
        bug_diag.append(float(iou_unclamped(a, b)[0, 0]))
        ok_diag.append(float(iou_matrix(a, b)[0, 0]))
    bug_diag, ok_diag = np.array(bug_diag), np.array(ok_diag)
    b_far = np.array([[300.0, 0.0, 400.0, 100.0]])
    bug_far = float(iou_unclamped(a, b_far)[0, 0])

    # CIoU's aspect term is zero whenever the two boxes have the same shape,
    # which is exactly the case drawn in the separation figure. One pair where
    # it is not, so the slide can say what the term is for.
    a_t = torch.tensor([0.0, 0.0, 100.0, 100.0], dtype=torch.float64)
    b_same = torch.tensor([120.0, 0.0, 220.0, 100.0], dtype=torch.float64)
    b_thin = torch.tensor([120.0, 0.0, 320.0, 50.0], dtype=torch.float64)

    print("Average precision…")
    person = 1
    p_prec, p_rec, _p_sc, p_ngt = pr_for_class(preds, gt, person, 0.5)
    ap_person = average_precision(p_prec, p_rec)
    ap_person_101 = average_precision_101(p_prec, p_rec)
    steps_down = int((np.diff(p_prec) < -1e-12).sum())
    steps_up = int((np.diff(p_prec) > 1e-12).sum())
    steps_flat = int((np.abs(np.diff(p_prec)) <= 1e-12).sum())
    n_tp = int(round(p_prec[-1] * len(p_prec)))
    # the leading run of correct detections, during which precision is pinned
    # at 1 and a further correct detection changes nothing
    leading_run = int(np.flatnonzero(p_prec < 1.0)[0]) if (p_prec < 1).any() \
        else len(p_prec)

    map_by_t, per_class_by_t = [], {}
    for t in IOU_THRESHOLDS:
        m, per = map_at(preds, gt, float(t), present)
        map_by_t.append(m)
        per_class_by_t[float(t)] = per
    map50 = map_by_t[0]
    map75 = map_by_t[int(np.flatnonzero(IOU_THRESHOLDS == 0.75)[0])]
    map5095 = float(np.mean(map_by_t))
    per50 = per_class_by_t[0.50]

    ranked = sorted(((names[c], v) for c, v in per50.items()),
                    key=lambda r: -r[1])
    zero_ap = [n for n, v in ranked if v == 0.0]

    # The wrong way: average precision computed per image, then averaged over
    # images. This is the "metric averaged per batch, not over the set" entry
    # in the course's silent-failure catalogue, in its detection costume.
    per_image = []
    for im in corpus["images"]:
        iid = im["id"]
        sub_p = {iid: preds[iid]}
        sub_g = {iid: gt[iid]}
        cls = sorted({int(c) for c in gt[iid]["labels"]})
        vals = []
        for c in cls:
            pr, rc, _s, n = pr_for_class(sub_p, sub_g, c, 0.5)
            if n:
                vals.append(average_precision(pr, rc))
        if vals:
            per_image.append(float(np.mean(vals)))
    map_per_image = float(np.mean(per_image))

    print("Reference cross-check…")
    ref = cached09("l09_cocoeval", lambda: reference_cocoeval(corpus, preds))
    if ref:
        export(l18_cocoeval=ref,
               l18_cocoeval_gap={"map50": abs(ref["map50"] - map50),
                                 "map75": abs(ref["map75"] - map75),
                                 "map5095": abs(ref["map5095"] - map5095)})

    print("NMS sweep…")
    from torchvision.ops import batched_nms
    nms_ts = np.round(np.arange(0.1, 0.91, 0.1), 2)
    nms_boxes, nms_map = [], []
    for t in nms_ts:
        sup = {}
        for iid, p in raw.items():
            keep = batched_nms(torch.tensor(p["boxes"]),
                               torch.tensor(p["scores"]),
                               torch.tensor(p["labels"]), float(t)).numpy()
            sup[iid] = {"boxes": p["boxes"][keep], "labels": p["labels"][keep],
                        "scores": p["scores"][keep]}
        nms_boxes.append(float(np.mean([len(s["scores"]) for s in sup.values()])))
        nms_map.append(map_at(sup, gt, 0.5, present)[0])
    nms_sweep = {"t": nms_ts, "boxes": np.array(nms_boxes),
                 "map50": np.array(nms_map)}
    raw_mean_boxes = float(np.mean([len(p["scores"]) for p in raw.values()]))

    print("Masks…")
    mask_ids = [im["id"] for im in corpus["images"][:40]]
    busy = sorted(mask_ids, key=lambda i: -len(gt[i]["labels"]))[:3]
    masks = cached09("l09_maskrcnn", lambda: _run_masks(corpus, busy))

    export(
        l18_sep_d=sep["d"].tolist(),
        l18_sep_iou=sep["iou"].tolist(),
        l18_sep_giou=sep["giou"].tolist(),
        l18_sep_ciou=sep["ciou"].tolist(),
        l18_sep_grad_iou=sep["g_iou"].tolist(),
        l18_sep_grad_giou=sep["g_giou"].tolist(),
        l18_box_side=100,
        l18_first_zero_at=float(sep["d"][first_zero]),
        l18_iou_flat={"n_disjoint": int(disjoint.sum()),
                      "max_iou_when_disjoint": float(sep["iou"][disjoint].max()),
                      "max_grad_when_disjoint":
                          float(np.abs(sep["g_iou"][disjoint]).max()),
                      "giou_at_150": float(sep["giou"][sep["d"] == 150][0]),
                      "giou_at_300": float(sep["giou"][sep["d"] == 300][0]),
                      "grad_giou_at_300":
                          float(sep["g_giou"][sep["d"] == 300][0])},
        l18_clamp_bug={"far": bug_far,
                       "correct": float(iou_matrix(a, b_far)[0, 0]),
                       "diag_d": diag_d.tolist(),
                       "diag_broken": bug_diag.tolist(),
                       "diag_correct": ok_diag.tolist(),
                       "broken_at_150": float(bug_diag[diag_d == 150][0]),
                       "broken_at_200": float(bug_diag[diag_d == 200][0]),
                       "broken_at_110": float(bug_diag[diag_d == 110][0])},
        # the two GIoU values printed inside assets/figures/d-giou.svg, which is
        # hand-drawn and therefore cannot compute them for itself
        l18_diagram_giou={"near": float(_torch_giou(
            torch.tensor([40., 70., 160., 180.], dtype=torch.float64),
            torch.tensor([200., 70., 320., 180.], dtype=torch.float64))),
            "far": float(_torch_giou(
                torch.tensor([580., 70., 700., 180.], dtype=torch.float64),
                torch.tensor([920., 70., 1040., 180.], dtype=torch.float64)))},
        l18_ciou_terms={"same_shape_v": float(_ciou_v(a_t, b_same)),
                        "thin_v": float(_ciou_v(a_t, b_thin)),
                        "same_shape_ciou": float(_torch_ciou(a_t, b_same)),
                        "thin_ciou": float(_torch_ciou(a_t, b_thin)),
                        "same_shape_iou": float(_torch_iou(a_t, b_same)),
                        "thin_iou": float(_torch_iou(a_t, b_thin))},
        l18_ap_person={"ap": ap_person, "ap_101": ap_person_101,
                       "n_gt": int(p_ngt), "n_pred": int(len(p_prec)),
                       "n_tp": n_tp, "n_fp": int(len(p_prec)) - n_tp,
                       "steps_down": steps_down, "steps_up": steps_up,
                       "steps_flat": steps_flat, "leading_run": leading_run,
                       "steps_total": int(len(p_prec)) - 1,
                       "max_precision": float(p_prec.max()),
                       "final_recall": float(p_rec[-1])},
        l18_map50=map50, l18_map75=map75, l18_map5095=map5095,
        l18_map_by_iou={"t": IOU_THRESHOLDS.tolist(), "map": map_by_t},
        l18_n_classes_scored=len(per50),
        l18_per_class_ap={n: float(v) for n, v in ranked},
        l18_ap_spread={"best": ranked[0][1], "worst": ranked[-1][1],
                       "n_zero": len(zero_ap),
                       "n_perfect": sum(1 for _n, v in ranked if v == 1.0),
                       "n_below_mean": sum(1 for _n, v in ranked if v < map50),
                       "best_name": ranked[0][0], "worst_name": ranked[-1][0],
                       "median": float(np.median([v for _n, v in ranked]))},
        l18_map_per_image=map_per_image,
        l18_map_gap=float(map_per_image - map50),
        l18_nms_sweep={"t": nms_ts.tolist(), "boxes": nms_boxes,
                       "map50": nms_map},
        l18_nms_best={"t": float(nms_ts[int(np.argmax(nms_map))]),
                      "map50": float(max(nms_map)),
                      "boxes_at_best":
                          float(nms_boxes[int(np.argmax(nms_map))])},
        l18_raw_boxes_per_image=raw_mean_boxes,
        l18_tv_reported_map=TV_REPORTED_MAP,
        # torchvision's own reported numbers for the Mask R-CNN weights, on the
        # full 5,000-image val2017. Read out of the weights metadata rather than
        # typed in from a web page.
        l18_maskrcnn_reported=MaskRCNN_ResNet50_FPN_Weights
            .COCO_V1.meta["_metrics"]["COCO-val2017"],
        # our 128-image estimate against the same model on all 5,000, in points
        l18_map_gap_vs_full=abs(map5095 * 100 - TV_REPORTED_MAP),
        # the window the right-hand panel of l18-pr-sawtooth.svg zooms into
        l18_sawtooth_window={
            "lo": 51, "hi": 140,
            "down": int((np.diff(p_prec[50:140]) < -1e-12).sum()),
            "up": int((np.diff(p_prec[50:140]) > 1e-12).sum())},
    )

    # ------------------------------------------------------------- figures
    print("\nFigures…")
    show_ids = [im["id"] for im in corpus["images"][:3]]
    busy_ids = sorted([im["id"] for im in corpus["images"][:40]],
                      key=lambda i: -len(gt[i]["labels"]))[:3]

    fig_gt_class_counts(freq.most_common(12), "l17-class-counts")
    fig_score_hist(all_scores, int((all_scores >= SHOW_THRESH).sum()),
                   "l17-score-hist")
    fig_count_vs_threshold(sweep, float(counts_true.mean()),
                           "l17-count-vs-threshold")
    fig_count_scatter(counts_true, counts_50, "l17-count-scatter")
    fig_detection_grid(corpus, preds, show_ids, "l17-detections",
                       title="Faster R-CNN, score \u2265 0.5, on three of our 128 "
                             "images; labels shown for scores \u2265 0.90")
    fig_detection_grid(corpus, preds, busy_ids[:2], "l17-detections-busy",
                       label_above=0.97, height=4.6,
                       title="the two most crowded images in the first forty; "
                             "labels shown for scores \u2265 0.97")
    fig_wholebox(corpus, show_ids[0], "l17-wholebox")
    fig_baseline_bars([
        ("one box per image", mae(baseline_counts, counts_true), False),
        ("every box returned", mae(counts_raw, counts_true), False),
        (f"score \u2265 {SHOW_THRESH}", mae(counts_50, counts_true), True),
    ], "l17-baselines")

    fig_iou_flat(sep, "l18-iou-flat")
    fig_clamp_bug(diag_d, bug_diag, ok_diag, "l18-clamp-bug")
    fig_giou_ciou(sep, "l18-giou-ciou")
    pairs = []
    for title, bb in (("300 px away", [300.0, 0.0, 400.0, 100.0]),
                      ("edge to edge", [100.0, 0.0, 200.0, 100.0]),
                      ("half overlapping", [50.0, 0.0, 150.0, 100.0]),
                      ("nearly identical", [10.0, 10.0, 110.0, 110.0])):
        bb = np.array(bb)
        pairs.append((title, BOX_A, bb,
                      float(iou_matrix(BOX_A[None], bb[None])[0, 0])))
    fig_iou_examples(pairs, "l18-iou-examples")
    fig_pr_sawtooth(p_prec, p_rec, ap_person, "person", int(p_ngt),
                    "l18-pr-sawtooth")
    fig_ap_by_iou(IOU_THRESHOLDS, map_by_t, map5095, "l18-ap-by-iou")
    fig_per_class_ap(ranked, freq, map50, "l18-per-class-ap")
    fig_nms_sweep(nms_sweep, "l18-nms-sweep")
    fig_nms_before_after(corpus, raw, busy_ids[0], "l18-nms")
    fig_masks(corpus, masks, busy[0], "l18-masks")
    _m = masks[busy[0]]
    _keep = _m["scores"] >= 0.7
    # How this image is annotated is itself worth a slide: COCO drew thirteen
    # people individually and put the rest of the class photo into one `iscrowd`
    # polygon that the official evaluator then ignores. The detector's count is
    # arguably the more faithful one, and the ground truth is a modelling
    # decision rather than a fact.
    _raw = json.loads((DATA / "instances_val2017.json").read_text())
    _here = [a for a in _raw["annotations"] if a["image_id"] == busy[0]]
    export(l18_masks={"image_id": int(busy[0]),
                      "instances": int(_keep.sum()),
                      "classes": len({int(c) for c in _m["labels"][_keep]}),
                      "gt_objects": int(len(gt[busy[0]]["labels"])),
                      "gt_person_individual": sum(
                          1 for a in _here
                          if names[a["category_id"]] == "person"
                          and not a["iscrowd"]),
                      "gt_person_crowd": sum(
                          1 for a in _here
                          if names[a["category_id"]] == "person"
                          and a["iscrowd"]),
                      "gt_tie": sum(1 for a in _here
                                    if names[a["category_id"]] == "tie")})

    # --------------------------------------------------------------- checks
    bad = []
    for d in sorted(OUT.glob("d-*.svg")):
        try:
            xml.dom.minidom.parse(str(d))
        except Exception as exc:                              # noqa: BLE001
            bad.append(f"{d.name}: {exc}")
    if bad:
        print("\nmalformed diagrams:")
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
