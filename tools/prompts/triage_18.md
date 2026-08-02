# Triage — lecture 18 defect report

Against `notebooks/lecture-18.ipynb` (74 cells, 24 code, **0 stored outputs** —
noted once here, not repeated per claim: no prose figure in this notebook can be
reconciled against a stored output, so every numeric claim below was re-derived
from scratch).

**What was available.** `notebooks/datasets/coco/instances_val2017.json` is
cached and genuine (5,000 images, 36,781 annotations, 80 categories), and all
128 selected JPEGs are on disk. The **detector was not run** — so every claim
that needs `preds` is either unverifiable or verified by a stated proxy.

Corpus sanity check, from the cached annotations, reproducing cell 6 exactly:

```
images 128   objects 898   crowd dropped 8
categories present: 73 of 80
person instances: 350 (39.0%)
objects per image (mean): 7.02
all 128 selected files present: True
```

The mathematical thread was run in full and **the notebook is right on all of
it** — this is the baseline against which the claims below are judged:

```
broken IoU, [0,0,100,100] vs [200,200,300,300]  ->  1.0        (exactly)
broken IoU, one axis 300 px                     -> -0.5
broken IoU, both axes 150 px                    ->  0.142857

autograd, float64, 20 separations past 100 px:
  max IoU 0.0    max |dIoU/dd| 0.0    all gradients exactly == 0.0: True
  GIoU at 300 px -0.5   dGIoU/dd at 300 px -0.00125   all GIoU grads < 0: True

AP hand case (4 detections, 2 annotations, TP FP TP FP):
  average_precision = 0.8333333333333333   target 0.5+1/3   |diff| = 0.0
envelope monotone non-increasing on 2000 random vectors: 0 violations

aspect term alpha*v when the two shapes agree: 0.0  (exactly zero: True)
```

---

### Claim 1 — four of six ⏱ figures are wrong, two by two orders of magnitude
**Verdict:** CONFIRMED (in part — two rows UNVERIFIABLE)

**Evidence:** Split three ways.

*(a) The two pure-NumPy rows — CONFIRMED by proxy.* Cells 49 and 58 contain no
torch and no device. I ran the notebook's own `pr_curve` / `average_precision` /
`envelope` over the **real** 128-image ground truth with synthetic detections at
two densities, the second being 100/image — the hard ceiling, since
`box_detections_per_img=100`:

```
proxy at  35 detections/image:  cell 49 sweep 1.72 s   cell 58 per-image AP 0.01 s
proxy at 100 detections/image:  cell 49 sweep 2.59 s   cell 58 per-image AP 0.01 s
      notebook says:                 "about 60 seconds"      "about 30 seconds"
```

730 class-threshold pairs in 2.6 s at the maximum possible detection count. The
cost is driven by detection count, not by box geometry, so the real run cannot
plausibly be 20× slower. Cell 58 is off by three orders of magnitude.

*(b) The CPU-coverage charge — CONFIRMED, structural.* Every ⏱ marker in the
file, extracted from the JSON:

```
cell  5 markdown  > **Prompt · ⏱ 60-90 s first time — the same 128 images**
cell  6 code      # ⏱ 60-90 s, once
cell 32 markdown  ⏱ **1 to 2 minutes** on a GPU or MPS, several minutes on CPU
cell 33 markdown  > **Prompt · ⏱ 1-2 min — the same detector, the same images**
cell 47 markdown  ⏱ **about 60 seconds**: 73 classes × 10 IoU thresholds.
cell 48 markdown  > **Prompt · ⏱ 60 s — mAP, a mean of a mean**
cell 56 markdown  ⏱ **about 30 seconds.**
cell 57 markdown  > **Prompt · ⏱ 30 s — ⚠ the second silent failure**
cell 60 markdown  ⏱ **about 60 seconds**: ... suppression switched off.
cell 61 markdown  > **Prompt · ⏱ 20 s — ...**  (cell 65: about 20 seconds)
cells mentioning CPU at all: [3, 32]
```

§7.1 requires the CPU number for every cell over ~20 s. Exactly one cell (32)
gives one. Cell 62 — the second GPU-bound inference loop — gives none.

*(c) The three detector rows — UNVERIFIABLE.* Cells 34, 62 and 67 require
running Faster R-CNN / Mask R-CNN, which was out of scope. 15.6 s, 11 s and 0.4 s
are untested.

**Severity:** misleads a student
**Origin:** hand-written prose
**Fix:** re-time every ⏱ cell on the target hardware and give the CPU figure for cells 47, 56, 60 and 65 as well as 32; the two NumPy cells should lose their ⏱ markers entirely.

---

### Claim 2 — cell 33's label-dtype failure does not happen
**Verdict:** CONFIRMED

**Evidence:** The claimed mechanism first:

```
np.array([1.,3.,1.]) == 1  ->  [ True False  True]     dtype bool, any True: True
```

Then the whole pipeline, using the notebook's verbatim `iou_many`, `pr_curve`,
`envelope` and `average_precision`, on a 40-image synthetic corpus built exactly
the way cell 34 builds `preds` (bulk `.astype(float)`), once with the fix-up line
and once without:

```
prediction label dtype WITHOUT the fix-up: float64
prediction label dtype WITH    the fix-up: int64
float64 labels (fix-up omitted)    mAP@0.50 = 0.158372   per-class [0.1615, 0.1393, 0.1426, 0.1901]
int64 labels (fix-up applied)      mAP@0.50 = 0.158372   per-class [0.1615, 0.1393, 0.1426, 0.1901]
detections of class 1 selected by the mask: float=58  int=58
```

Identical to every digit. The mask is not "silently all-False"; it selects the
same 58 rows. The constraint in cell 33 and the student bullet under it both
describe a failure NumPy does not have.

**Severity:** misleads a student
**Origin:** hand-written prose
**Fix:** cut the constraint clause and the student bullet, or keep `.astype(np.int64)` with the honest reason (dict keys and printing), not an invented all-False mask.

---

### Claim 3 — "rows four and five … 150 px apart and 300 px apart" does not resolve
**Verdict:** CONFIRMED (the cell-26 half of the claim is FALSE)

**Evidence:** Cell 24 prints `rows[::5]` where `rows` walks `np.arange(0, 301, 10)`:

```
printed rows: [0. 50. 100. 150. 200. 250. 300.]
  row 1: d = 0     row 2: d = 50    row 3: d = 100   row 4: d = 150
  row 5: d = 200   row 6: d = 250   row 7: d = 300
```

Cell 25 says *"Read rows four and five. Two boxes 150 px apart and two boxes 300
px apart…"*. Rows four and five are 150 and **200**. The sentence needs rows four
and **seven**.

The claim's second sentence — *"the same wrong reference is repeated in cell 26's
student bullet"* — does not hold. Cell 26's bullet reads *"reading rows four and
five off the table and saying the gradient 'goes to zero'"*, and names no
separations. Rows four and five (d = 150, 200) both sit past 100 px with an
exactly-zero gradient, so that reference is correct.

**Severity:** wrong but harmless
**Origin:** hand-written prose
**Fix:** cell 25 — "rows four and seven"; leave cell 26 alone.

---

### Claim 4 — "the figure above" in cells 29 and 30 refers to a figure that does not exist
**Verdict:** CONFIRMED — but the claim's own headline is wrong and must not be copied into the rebuild

**Evidence:** The figure does exist. Code cells that draw one:

```
code cells that draw a figure: [18, 46, 55, 69]
markdown cells containing 'figure above': [29, 30]   code cells: [30]
cells containing 'earlier figure': [29]
```

Cell 18 is 11 cells above cell 29, plots `iou` against `iou_broken` for two
100 × 100 boxes pulled diagonally — so cell 29's *constraint*, "every pair in the
figure above is square", is **true of it**. The claim overstates.

What is real is the rest:

```
cell 24: mentions GIoU/CIoU; draws a plot: False
cell 27: mentions GIoU/CIoU; draws a plot: False
cell 30: mentions GIoU/CIoU; draws a plot: False
```

**No figure in the notebook plots GIoU or CIoU.** So cell 29's student bullet —
*"concluding from the earlier figure that CIoU and GIoU behave identically. They
do on square boxes, which is all that figure contains"* — invites a conclusion no
figure in the file could support, and cell 30's printed line gives the wrong
reason: the aspect term is invisible in cell 18's figure because CIoU is not
drawn there at all, not because the pairs happen to be square.

**Severity:** wrong but harmless
**Origin:** hand-written prose
**Fix:** rewrite cell 29's bullet and cell 30's print to point at cell 24's **table** (which does carry GIoU), or add GIoU and CIoU to cell 18's figure so the reference becomes true.

---

### Claim 5 — the CIoU demonstration is confounded and only totals are printed
**Verdict:** CONFIRMED

**Evidence:** Decomposed with the notebook's own `t_ciou`, float64:

```
same 100x100       IoU 0.000  rho2/l2 0.2466  alpha*v 0.0000  CIoU -0.2466  diag(C) 241.66
different 200x50   IoU 0.000  rho2/l2 0.2627  alpha*v 0.0125  CIoU -0.2752  diag(C) 335.26
gap in totals   0.0286
  from aspect   0.0125   (44%)
  from rho2/l2  0.0161   (56%)
aspect term, same shapes: 0.0   exactly zero: True
```

The two boxes differ in aspect ratio, in centre distance and in the enclosing
diagonal (241.66 → 335.26) simultaneously — a §2.1 confound. The cell prints
`IoU` and `CIoU` only, so the reader cannot see that the majority of the gap the
cell attributes to the aspect term is centre distance.

Two corrections to the claim's own arithmetic: the gap is **0.0286**, not 0.0277,
and the aspect term is therefore **44%** of it, not 45%. The component split it
reports (0.0125 / 0.0161) is exact. Its final sentence — that the aspect term is
exactly zero when the shapes agree — reproduces to the bit.

**Severity:** misleads a student
**Origin:** hand-written prose
**Fix:** print the three terms separately, and add a third box that changes shape while holding the centre distance fixed.

---

### Claim 6 — "More than half the classes are below it" (cell 51) is false
**Verdict:** UNVERIFIABLE

**Evidence:** The claim needs 73 per-class APs at IoU 0.50, which require running
Faster R-CNN over the 128 images. That was out of scope for this triage. The
instance counts the claim leans on are checkable and do reproduce (person = 350
of 898, 39.0%; 73 of 80 categories present), but they cannot settle "29 of 73,
40%" or "the median 0.673 sits above the mean". **Nobody has tested this claim.**

Note that the sentence at issue is in a prompt-box *student bullet*, not in the
printed output — cell 52 prints `classes below the mean:` as a derived number, so
whatever the truth is, the cell and the prose can disagree without a reader
noticing.

**Severity:** unknown — if true, misleads a student
**Origin:** hand-written prose
**Fix:** run cell 52 and copy its printed `classes below the mean` figure into the bullet, or phrase the bullet so it refers to the printed number rather than asserting a second one.

---

### Claim 7 — two figures hard-coded as string literals in plotting code
**Verdict:** CONFIRMED

**Evidence:** Every title call in every code cell:

```
cell 18: plt.title("the boxes are disjoint beyond 100 px; one curve does not know")
cell 46: ax.set_title(f"person, IoU >= 0.5, AP = {ap_person:.3f}")      <- derived
cell 46: ax.set_title("the sawtooth, and the staircase that repairs it")
cell 55: plt.title("128 images: 0.659 at the loosest threshold, 0.040 at the tightest")
cell 69: ax.set_title(ttl, fontsize=10, loc="left")
```

And every appearance of the two numbers:

```
cell 54 (markdown): the range: 0.659 at the loosest threshold and 0.040 at the tightest
cell 55 (code)    : plt.title("128 images: 0.659 ... 0.040 ...")
cell 73 (markdown): | mAP at IoU 0.50 | **0.659** |
```

Cell 46 four cells earlier interpolates `ap_person` correctly, so the author knew
the idiom and did not use it here. Whether 0.659 and 0.040 are correct today is
UNVERIFIABLE without the detector — but that is beside the point of §1.1: the
title will keep printing them after the corpus changes and nothing warns anyone.

**Severity:** wrong but harmless (latent)
**Origin:** generated code
**Fix:** `plt.title(f"128 images: {map_at[0.50]:.3f} at the loosest threshold, {map_at[0.95]:.3f} at the tightest")`, and make cell 54's bullet refer to the printed pair rather than restating it.

---

### Claim 8 — the `np.clip(img, 0, 1)` catch in cell 68 describes an impossible failure
**Verdict:** CONFIRMED — both halves of the bullet fail

**Evidence:** *(a) The drift cannot happen.* Cell 69's blend is
`img*(1 - 0.55m) + 0.55m*c`, whose coefficients sum to exactly 1 — a convex
combination of values already in [0, 1]. 45 successive blends on random data:

```
after 45 blends: min 0.223511  max 0.825571
running extremes over all 45: 0.005476 .. 0.998235
any value <0 or >1 ever: False
```

*(b) matplotlib does not rescale.* The bullet says *"matplotlib's response to
that is to rescale the whole image"*. matplotlib 3.10.6, feeding
`imshow` an RGB float array containing 1.4 and −0.3 alongside 0.5 and 0.25:

```
Clipping input data to the valid range for imshow with RGB data
([0..1] for floats or [0..255] for integers). Got range [-0.3..1.4].

top-left  (1.4, -0.3, 0.5) -> [255   0 127]
top-right (0.5, 0.5,  0.5) -> [127 127 127]     0.5*255 = 128
bot-left  (0.25,...)       -> [ 63  63  63]     0.25*255 = 64
```

It **clips**, it leaves in-range values untouched, and it says so on stderr. The
`np.clip` is harmless and defensible as habit; the reason given for it is
invented twice over.

**Severity:** wrong but harmless
**Origin:** hand-written prose
**Fix:** replace the bullet with the real reason (`np.clip` is cheap insurance for the general case) or cut it — §6.2.

---

### Claim 9 — "indexing it as `(N, H, W)` silently gives you the first object, repeated" is wrong twice
**Verdict:** CONFIRMED

**Evidence:** Built an `(N, 1, H, W)` tensor whose object *j* is a constant-*j*
plane, so "the first object" is distinguishable by value:

```
out['masks'].shape : (5, 1, 8, 9)
masks[2].shape     : (1, 8, 9)   unique values: [2.0]
masks[0] unique    : [0.0]
-> masks[2] is object 2 with a spare leading axis, NOT the first object

blend shape from masks[j][...,None] against (H,W,3): (1, 8, 9, 3)
imshow raises: TypeError: Invalid shape (1, 8, 9, 3) for image data
```

So `masks[j]` is the *correct* object with a spare axis, and the consequence is
not silent — it raises, loudly, immediately. Both halves of the constraint are
false, and the same wording is repeated as red-team question 3 in cell 73.

For contrast, the notebook's own code is right and sidesteps the issue entirely:

```
cell 67 does masks = out['masks'][:,0] -> (5, 8, 9); masks[j] -> (8, 9); masks[j][...,None] -> (8, 9, 1)
```

**Severity:** misleads a student
**Origin:** hand-written prose
**Fix:** rewrite both places as "`out['masks'][j]` is `(1, H, W)`, not `(H, W)`; broadcasting it against an `(H, W, 3)` image gives `(1, H, W, 3)` and `imshow` raises `TypeError` — which is why cell 67 drops the axis with `[:, 0]` first."

---

### Claim 10 — "nine tenths of its own output" is about a pipeline the notebook never measures
**Verdict:** CONFIRMED (structural); the substituted figure 34.5 boxes/image is UNVERIFIABLE

**Evidence:** The sentence, in two places:

```
cell 60 (markdown): The detector you ran had already thrown away nine tenths of its own output
cell 61 (markdown): * Left open: that the detector you ran last lecture had already thrown away
                    nine tenths of its own output before you saw it ...
```

Everything cell 34 prints:

```
print(f"{N_IMAGES} images in {time.time() - t0:.1f} s on {DEVICE}")
```

That is the whole of it. The only cells that count boxes per image are:

```
cell 62: n_raw = np.mean([len(p["scores"]) for p in raw_preds.values()])
cell 64: print(f"NMS IoU {t:.1f} -> {np.mean(kept):6.1f} boxes per image")
```

Both operate on `raw_preds`, a **different model instance** built with
`box_score_thresh=0.05, box_nms_thresh=1.0, box_detections_per_img=300`. The
stock `model` from cell 34 (defaults 0.05 / 0.5 / 100) never has its box count
measured or printed anywhere in the notebook, so the reader cannot see the
substitution. The claim's structural charge holds exactly as stated.

The corrected ratio it offers — 34.5 stock boxes/image, 0.149, "six sevenths" —
would need a detector run and is untested.

**Severity:** misleads a student
**Origin:** hand-written prose
**Fix:** either change the sentence to "the raw candidate pool below" (one word of scope, no code change), or add `print(np.mean([len(p["boxes"]) for p in preds.values()]))` to cell 34 so the claim is measured where it is made.

---

### Claim 11 — the missing clamp is announced (at least) five times before the reader can be caught by it
**Verdict:** CONFIRMED — and undercounted

**Evidence:** Every mention before the defective cell 12, in source order:

```
cell  0 md   "an IoU function that reports two boxes 200 pixels apart as a perfect match"
cell  8 md   constraint: "CLAMP the overlap width and height at zero — this is not
             defensive programming, it is the whole function"            <- not in the claim's list
cell  8 md   student bullet: "omitting the clamp, which is the next cell ...
             Two disjoint boxes give a negative width AND a negative height"
cell  9 code the CORRECT function, docstring: "The clip is not defensive programming.
             Without it two disjoint boxes give a negative width AND a negative height"
cell 10 md   "**⚠ Read before running.** ... One thing is missing."
cell 11 md   "> **Prompt · ⚠ what the assistant returns**"
cell 12 code "inter = (x2 - x1) * (y2 - y1)              # <- no clamp"
```

Seven, not five, and the arithmetic of the bug ("two negative differences
multiply to a positive") is given in full in cell 8 — before the reader has seen
the broken function. §8.1 sets the ceiling at fewer than four.

The second defect, per-image AP, is announced three times before its cell, as the
claim says:

```
cell 56 md   "## 7 · The second silent failure: per-image averaging"
cell 57 md   "> **Prompt · ⏱ 30 s — ⚠ the second silent failure**"
cell 58 code "# ⚠ read before running — this is the WRONG way, on purpose"
```

**Severity:** wrong but harmless
**Origin:** notebook structure
**Fix:** move cell 9 (the correct function) to after cell 15, and cut the clamp wording from cell 8's constraint and bullet — broken function, overlapping tests, disjoint tests, then the contrast.

---

### Claim 12 — "which is the next cell" (cell 8) points three cells away
**Verdict:** CONFIRMED

**Evidence:** Cell 8's bullet: *"omitting the clamp, which is the next cell and
the lecture's assistant failure."* Cell 9 is the **correct** `iou`, with the
clamp. `iou_broken` is cell 12, with markdown cells 10 and 11 in between:

```
cell  8 markdown  prompt box, "... which is the next cell"
cell  9 code      def iou(a, b):        # has the clamp — this is the correct one
cell 10 markdown  "### 3.1 · An assistant writes this function"
cell 11 markdown  "> **Prompt · ⚠ what the assistant returns**"
cell 12 code      def iou_broken(a, b): # <- no clamp
```

**Severity:** wrong but harmless
**Origin:** hand-written prose
**Fix:** "which is three cells below" — or, better, delete the pointer as part of the claim-11 reordering.

---

### Claim 13 — `inst`, `d`, `g` and `t` are each bound to two different kinds of object
**Verdict:** CONFIRMED (with two corrections to the claim's detail)

**Evidence:** Module-scope bindings, walked with `ast` over all 24 code cells:

```
cell  6: g    = <for-loop target over gt.values()>      dict {"boxes","labels"}
cell 18: d    = np.arange(0, 201, 10, dtype=float)      np.ndarray
cell 24: d    = torch.tensor(dv, dtype=float64, requires_grad=True)   0-d Tensor
cell 24: g,   = torch.autograd.grad(v, d)               gradient Tensor   (line 32)
cell 24: g    = <for-loop target over rows[::5]>        the GIoU VALUE    (line 38)
cell 49: t    = <for-loop target over IOU_TS>           IoU matching threshold
cell 52: inst = collections.Counter()
cell 52: g    = <for-loop target over gt.values()>
cell 64: t    = <for-loop target over [0.1,0.3,0.5,0.7,0.9]>   NMS threshold
cell 69: inst = base.copy()                             (H, W, 3) float array
```

All four rebindings are real. Two details in the claim are wrong: the two `g`
bindings in cell 24 are **six** lines apart (32 → 38), not eleven; and cell 37's
`g = gt_by_img[iid]` is **function-local** to `pr_curve`, not module scope, so it
is not part of the collision. `t` colliding across cells 49 and 64 is the sharpest
of the four, exactly as the claim says — §8 of the notebook argues those are
different knobs and then gives them one name.

**Severity:** wrong but harmless
**Origin:** generated code
**Fix:** `inst_count` / `overlay`; `sep` / `d_t`; `grad` / `giou_val`; `iou_t` / `nms_t`.

---

### Claim 14 — cell 58 leaves the notebook broken if it does not finish
**Verdict:** CONFIRMED

**Evidence:** Cell 58's module-level AST node types, in order:

```
['Assign', 'Assign', 'For', 'Assign', 'Assign', 'Expr', 'Expr', 'Expr', 'Assert']
has try: False | has finally: False
```

The restore is a bare statement after the loop:

```python
all_preds, all_gt = preds, gt
for im in images:
    preds, gt = {iid: all_preds[iid]}, {iid: all_gt[iid]}       # one image
    ...
preds, gt = all_preds, all_gt                                    # put it back
```

Any interrupt or exception inside the loop leaves the module globals `preds` and
`gt` bound to **one-image dicts**, and every cell below silently scores a
one-image corpus. Cell 57's own box warns that this "breaks every cell below with
no obvious cause" and then provides no protection against it.

One correction to the claim: recovery needs **cell 6 as well as cell 34** — `gt`
is built in cell 6, `preds` in cell 34 — and the notebook names neither (§7.2).

**Severity:** misleads a student
**Origin:** generated code
**Fix:** wrap the loop in `try: ... finally: preds, gt = all_preds, all_gt`, and name cells 6 and 34 as the recovery path in the box.

---

### Claim 15 — the string "examinable" appears zero times
**Verdict:** CONFIRMED

**Evidence:** Counted case-insensitively over all 74 cells' source:

```
occurrences of 'examinable'      : 0
occurrences of 'not examinable'  : 0
occurrences of 'beyond the book' : 0
```

§8.3 requires every section to carry one of *examinable* / *not examinable —
engineering* / *beyond the book, for context*. The notebook has eleven sections
and zero markers.

**Severity:** wrong but harmless
**Origin:** notebook structure
**Fix:** add one marker per `##` heading — §3, §4 and §5 are examinable; §8 and §9 are engineering.

---

### Claim 16 — 24 code cells, 24 full three-bullet annotations
**Verdict:** CONFIRMED

**Evidence:** Counted from the notebook JSON, a "full" box being one carrying all
three of `**Left open:**`, `**The usual student version:**` and `**How you would
catch it:**`:

```
cells 74   code 24   md 50
prompt boxes: 24
with all three bullets: 24
box cell indices: [2,5,8,11,14,17,20,23,26,29,33,36,39,42,45,48,51,54,57,61,63,66,68,71]
code cells NOT immediately preceded by a prompt box: []
```

§6.1's budget is five to eight full annotations, never more than ten. This is 24 —
three times the ceiling — and it matches the count GUIDELINES.md already records
for this notebook. The measured consequence lands badly here: the readers stopped
around cell 30, which in lecture 18 is the CIoU property test, and cell 33 is the
detector's box.

**Severity:** wrong but harmless
**Origin:** notebook structure
**Fix:** keep all 24 short `input · output · constraint · check` boxes; cut the three-bullet annotation to the six or seven cells where the prompt genuinely failed (8/11, 14, 26, 29, 57, 66).

---

## Summary

```
confirmed: 15   false positive: 0   unverifiable: 1
of the confirmed, 7 mislead a student
origin split — prose: 10   code: 3   structure: 3
duplicates: claim 12 is sub-item (b) of claim 11 — the same "which is the next
            cell" pointer, counted twice. Within claim 9, the cell-66 constraint
            and cell-73 red-team question 3 are one defect stated twice, which
            the claim itself notes.
```

**Calibration note.** This is an unusually high confirmation rate, so here is what
would falsify it: eleven of the fifteen are structural facts I re-derived
independently (greps, AST walks, cell-index counts) rather than judgements, and
four are numerical re-derivations that ran in seconds. Against that, the report
itself contains three errors I had to correct while confirming it — claim 4's
headline ("a figure that does not exist") is false, claim 5's gap is 0.0286 not
0.0277, and claim 13 misplaces cell 37's `g` in module scope and the line
distance in cell 24. None of them change the verdict; all of them mean the
report's prose should not be copied into the rebuild verbatim.

**What nobody has tested.** Claim 6 in full, the three detector rows of claim 1,
claim 7's numbers (as opposed to their hard-coding), and claim 10's 34.5
boxes/image. All four need a Faster R-CNN pass over the 128 cached images, which
is one run away — the annotations and all 128 JPEGs are on disk.
