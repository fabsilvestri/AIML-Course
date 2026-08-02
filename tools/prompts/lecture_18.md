# Lecture 18 — Scoring a box, scoring a detector

**A script for rebuilding this notebook in Colab by prompting.**

Follow it at the keyboard, in order. Every prompt below is what a competent
person actually types — short, under-specified, and occasionally wrong. Where a
prompt is deliberately loose, the loose version is the lesson; do not tighten it
to be helpful.

---

## Before you start

**Numbers.** Every figure in this script was re-derived on 2 August 2026 by
running the notebook's own code against `notebooks/datasets/coco/` — 128 images
of COCO `val2017`, `instances_val2017.json` as published. Measured with
**torch 2.13.0 / torchvision 0.28.0**, Faster R-CNN `COCO_V1` weights, on
**MPS** (Apple silicon). The detector is deterministic, so the mAP figures
reproduce exactly on the same weights; a different backend can move the third
decimal. If yours does, that is the finding, not an error — write down what you
get and compare.

**Timing.** Wall clock below is measured, not estimated, and each figure names
the device. The CPU figures are Apple silicon with `torch.set_num_threads(2)`,
which is what a free Colab runtime gives you. **Two cells dominate the whole
notebook on CPU and nothing else comes close.**

| | MPS, measured | CPU (2 threads), measured |
|---|---|---|
| Cell 12 — detector over 128 images | 15.6 s | **4.06 s/image → ~8 min 40 s** |
| Cell 21 — the same 128 with NMS off | 11 s | **~9 min** |
| everything else combined | under 5 s | under 5 s |

If you are on a CPU runtime, start Cell 12 and read §2 while it runs. Do **not**
plan to re-run Cell 12 or Cell 21; every later cell reuses their output.

**Download.** Cell 2 fetches
`annotations_trainval2017.zip` — verified `Content-Length: 252,907,541` bytes
(**241 MiB**) — and then 128 JPEGs one at a time, 20 MB in **128 separate HTTP
requests**. The zip is bandwidth-bound; the 128 requests have a round-trip floor
you cannot beat by having a fast connection. I could not time this honestly
(both were already on disk), so budget by arithmetic: 241 MiB at your actual
download rate, plus 128 round trips. Once it is on disk the cell is instant, and
it is written to be.

**Prompt-box convention.** Every cell gets the structured box GUIDELINES §6.4
makes standard:

> **input** · what goes in · **output** · what comes out ·
> **constraint** · what must be true of the method · **check** · how you know it is wrong

Seven cells — 5, 7, 10, 12, 13, 20, 22 — additionally get the three-bullet
annotation. Those are the seven where the prompt genuinely fails. The other
nineteen get the specification only, which is what §6.1 asks for.

---

# §1 · Setup

## Cell 1 — setup

**Prompt to type:**

> Setup cell. numpy, torch, torchvision, matplotlib, PIL. Print the python,
> torch and torchvision versions. Seed numpy and torch with 42. Pick cuda, then
> mps, then cpu, and print which. Set `N_IMAGES = 128` and make a
> `datasets/coco` directory.

**Expect:** four printed lines — python, torch, torchvision, device. Nothing
else.

**Assert:** none.

**Annotate:** short

> **input** · nothing ·
> **output** · versions, seed, device, and `N_IMAGES = 128` ·
> **constraint** · `N_IMAGES` a named constant, not a literal `128` in a slice, so it travels into every printed line below ·
> **check** · the device line names one of cuda/mps/cpu and you recognise which machine you are on

---

## Cell 2 — the corpus

**Prompt to type:**

> Download COCO's `instances_val2017.json` from
> `http://images.cocodataset.org/annotations/annotations_trainval2017.zip`,
> only if it is not already on disk. Take the 128 images with the lowest ids.
> Download those images too, skipping any already present. Build a dict from
> image id to `{"boxes", "labels"}` with boxes in `[x1,y1,x2,y2]` corner form —
> COCO stores `x, y, w, h`. Skip `iscrowd` annotations and count how many you
> skipped. Print the totals.

**Expect:**
`128 images, 898 objects, 8 crowd regions dropped`

**Assert:**
```python
assert len(gt) == N_IMAGES and n_true.sum() == 898
```

**⏱** First run: 241 MiB + 128 HTTP round trips, bandwidth-bound (see above).
Every run after that: under a second on any device.

**Annotate:** short

> **input** · COCO's annotations and the rule "the 128 lowest ids" ·
> **output** · the identical corpus and ground truth as Lecture 17 ·
> **constraint** · rebuild from the RULE, not by inheriting anything from the other notebook's kernel ·
> **check** · assert the exact object count 898, which is what Lecture 17 reported. Shapes agree under a great many wrong reloads; a total does not

*Verified:* 898 objects and 8 crowd regions, from the published file. 73 of
COCO's 80 categories appear. `person` accounts for 350 of the 898 — **39.0%**.

---

# §2 · Intersection over union

Lecture 17 left you with a request: a number that says how right a box is. It
must be 1 for identical boxes and 0 for disjoint ones, must punish a box for
being too large *and* for being too small, and must be dimensionless.
Requirement two forces the predicted area into the denominator, three forces the
true area in with it, four forces the numerator to be an area. There is
essentially one candidate:

$$\mathrm{IoU}(A,B) = \frac{|A \cap B|}{|A \cup B|}
  = \frac{|A \cap B|}{|A| + |B| - |A \cap B|}$$

Ask for it.

## Cell 3 — IoU

**Prompt to type:**

> Write a NumPy function that takes two bounding boxes in `[x1, y1, x2, y2]`
> format and returns their intersection over union.

**Expect:** eight or ten lines. A `maximum` of the two top-left corners, a
`minimum` of the two bottom-right corners, a product for the intersection, the
two areas, one division.

**Assert:** none. Read it, then go to Cell 4.

**Annotate:** short

> **input** · two corner-form boxes ·
> **output** · their IoU ·
> **constraint** · NumPy, and the corner format named explicitly ·
> **check** · identical boxes give 1.0, and two 100×100 boxes overlapping by half give 1/3 — work that out on paper before you run anything: intersection 5,000, union 15,000

**Keep whatever comes back**, and call it `iou_naive`. If it happens to contain
a `clip`, a `clamp` or a `max(0, ...)` on the overlap width and height, delete
those three characters' worth of protection and keep the unprotected version
anyway — the rest of this section is about what that line is for, and you cannot
learn it from a function that already has it. You will write the protected
version yourself in Cell 8.

---

## Cell 4 — test it on two pairs

**Prompt to type:**

> Test it on two overlapping pairs: `[0,0,100,100]` against `[50,0,150,100]`,
> and `[0,0,100,100]` against `[10,10,110,110]`. Print both.

**Expect:**
```
0.3333333333333333
0.6803278688524591
```

**Assert:** none — but **write both numbers down**, and check the first against
the arithmetic you did in Cell 3's box. It is exactly 1/3. The function is
correct.

**Annotate:** short

> **input** · two pairs of overlapping boxes ·
> **output** · their IoU ·
> **constraint** · pairs you can check by hand — the first is 5,000/15,000 ·
> **check** · 0.3333… to the last digit, not "about a third"

---

## Cell 5 — ⚠ test it on a case whose answer you already know

**Prompt to type:**

> Now two boxes that do not touch. `[0,0,100,100]` against `[300,0,400,100]`,
> against `[150,150,250,250]`, and against `[200,200,300,300]`. Print all three.

**Expect:**
```
one axis apart :  -0.5
both axes, 150 :  0.14285714285714285
both axes, 200 :  1.0
```

**Assert:** none yet. Read the third line.

**Annotate:** full

Two boxes 200 pixels apart in **both** directions are reported as a **perfect
match**. Not approximately — `1.0`, the same value the function returns for a
box compared with itself.

The arithmetic: with the boxes 200 apart on each axis, the overlap width is
`min(100,300) - max(0,200) = -100` and the overlap height is also `-100`. Two
negative differences multiply to a *positive* 10,000, which the function reads
as an intersection of 10,000 px² between two boxes that share no pixel. The
union then computes as `10,000 + 10,000 - 10,000 = 10,000`, and 10,000/10,000
is 1.

* **Left open:** the prompt in Cell 3 says format, library and return value —
  more than most prompts specify — and never says what the function must do when
  the boxes do **not overlap**. "Intersection over union" has no vocabulary for
  that case, so neither does the prompt, so neither does the code.
* **The usual student version:** this **is** the usual version. It is also what
  the assistant returns. It is correct on every pair you would naturally try —
  both Cell 4 pairs are exactly right — because every pair you would naturally
  try overlaps. That is not bad luck; it is what "a reasonable first test" means.
* **How you would catch it:** the one-axis case gives **−0.5**, and a negative
  IoU is unmissable. The diagonal case gives 1.000 and is invisible. **Separation
  along one axis and separation along both are different tests.** A single
  disjoint example is not a disjointness test, and if you had only tried
  `[300,0,400,100]` you would have found the wrong bug.

---

## Cell 6 — plot it

**Prompt to type:**

> Plot the reported overlap against diagonal separation, 0 to 200 pixels in
> steps of 10, for the function above. Mark 100 pixels with a dotted vertical
> line — past that the boxes are disjoint.

**Expect:** a U. The curve falls from 1.0 to 0 at 100 px and then climbs back to
1.0 at 200 px. It is symmetric about 100, because the reported intersection is
$(100-d)^2$ and that is an even function of $d - 100$.

**Assert:** none. The picture is the assertion.

**Annotate:** short

> **input** · two 100×100 boxes pulled apart diagonally, 0 to 200 px ·
> **output** · reported overlap against separation ·
> **constraint** · pull them apart DIAGONALLY, so the symmetry about 100 px is visible — a one-axis sweep goes negative and looks like a different bug ·
> **check** · the value at 200 px should be 0.000 and will print as 1.000, and the curve turns around where it should be flat at zero

---

## Cell 7 — the test that would have caught it

**Prompt to type:**

> Write a property test instead of a table of expected values. Over a 5×5 grid
> of offsets from `[0, 50, 100, 150, 200]` in x and y, assert that the IoU is in
> [0, 1] and that it is zero exactly when the boxes are disjoint on at least one
> axis. Then run the same loop against the broken version and show that it fails.

**Expect:**
```
25 pairs checked, property holds
broken version fails the property test, as it must
```

**Assert:** the two the prompt asks for —
```python
assert 0.0 <= v <= 1.0
assert (v == 0.0) == (dx >= 100 or dy >= 100)
```
The second one is exact, not toleranced, and it must be. A tolerance there
passes for a function returning 1e-9, which is a different function.

**Annotate:** full

* **Left open:** whether the two halves of the property are separable. They are,
  and the difference matters. Only the **range** half is needed to catch the raw
  broken function — the very first disjoint pair returns −0.5 and trips
  `0 <= v` without anybody having had to think of the diagonal case. The
  **"zero iff disjoint"** half is what catches the *repaired* version, below.
* **The usual student version:** seeing `-0.5`, concluding the function needs a
  floor, and writing `return max(0.0, inter / (area_a + area_b - inter))`. I ran
  it. On the three Cell 5 pairs that repair gives **0.000, 0.143 and 1.000**, and
  **all three pass `0 <= v <= 1`**. Clamping the *result* fixes the case you saw
  and leaves the diagonal case reporting a perfect match, now with a range assert
  standing guard over it. The clamp belongs on the overlap **width and height**,
  before they are multiplied, because that is where the sign is lost.
* **How you would catch it:** run your test suite against the broken function
  and watch it fail. A test nobody has seen fail is a test of unknown strength —
  and in this case the version of the test that only checks the range would have
  passed the repaired function and told you the bug was gone.

---

## Cell 8 — the repaired function, and a vectorised one

**Prompt to type:**

> Rewrite it with the overlap width and height clipped at zero before they are
> multiplied. Then write `iou_many(one, many)` for one box against an `(M, 4)`
> array, returning an `(M,)` array. Re-run the property test on both.

**Expect:** the 25-pair loop passes for both. `iou_many` of a box against a
zero-length array returns a zero-length array, not an exception.

**Assert:**
```python
assert iou(box, box) == 1.0
assert iou(box, box + np.array([100, 0, 100, 0])) == 0.0      # edge to edge
assert abs(iou(box, box + np.array([50, 0, 50, 0])) - 1/3) < 1e-12
```
The edge-to-edge case is `== 0.0`, exactly. Everything downstream in this
notebook depends on it.

**Annotate:** short

> **input** · one box and an `(M, 4)` array ·
> **output** · an `(M,)` array of IoUs ·
> **constraint** · clip `rb - lt` at zero BEFORE multiplying, and handle `M = 0` — a class with no annotation in an image gives a zero-length array ·
> **check** · edge-to-edge asserts exactly `0.0`; `iou_many(b, np.zeros((0,4)))` returns shape `(0,)` and does not raise

---

# §3 · The gradient that is not there

IoU does three jobs: matching (§4), suppression (§7), and serving as a loss.
Only the third needs a derivative, and that is the one that breaks. Nothing in
this section touches the dataset — the conclusion is a property of the formula.

## Cell 9 — IoU and GIoU, with autograd

**Prompt to type:**

> Write torch versions of IoU and GIoU. GIoU is IoU minus `(|C| - |A∪B|)/|C|`
> where C is the smallest box containing both. Then pull two 100×100 boxes apart
> along x, 0 to 300 in steps of 10, and for each separation use
> `torch.autograd.grad` to get the derivative of each with respect to the
> separation. Use float64. Print every fifth row.

**Expect:**
```
    d      IoU    dIoU/dd     GIoU   dGIoU/dd
    0    1.000    0.00000    1.000    0.00000
   50    0.333   -0.00889    0.333   -0.00889
  100    0.000   -0.00500    0.000   -0.00500
  150    0.000    0.00000   -0.200   -0.00320
  200    0.000    0.00000   -0.333   -0.00222
  250    0.000    0.00000   -0.429   -0.00163
  300    0.000    0.00000   -0.500   -0.00125
```

**Assert:** none here — Cell 10 does the asserting.

**Annotate:** short

> **input** · two 100×100 boxes pulled apart along x, 0 to 300 px ·
> **output** · IoU and GIoU with their derivatives at each separation ·
> **constraint** · ask AUTOGRAD for the derivative rather than differentiating by hand — the claim is about what an optimiser would actually receive ·
> **check** · at d = 50 both functions agree exactly (the boxes still overlap, so GIoU's penalty is zero) and both gradients read −0.00889

Read rows four and seven — **150 px apart and 300 px apart**. To IoU they are
identical: same value, same gradient. To GIoU they are −0.200 and −0.500.

---

## Cell 10 — assert it rather than eyeballing it

**Prompt to type:**

> For every separation strictly past 100, assert that IoU is exactly zero, that
> its gradient is exactly zero, and that GIoU's gradient is still negative.
> Print how many separations that is, and GIoU's value and gradient at 300.

**Expect:**
```
20 separations past 100 px:
  max IoU there          0.000000
  max |dIoU/dd| there    0.000000
  GIoU at 300 px         -0.500
  dGIoU/dd at 300 px     -0.00125
```

**Assert:**
```python
assert all(r[1] == 0.0 for r in past)
assert all(r[2] == 0.0 for r in past)
assert all(r[4] <  0.0 for r in past)
```
Three claims, three asserts, and the first two are exact equalities with zero.

**Annotate:** full

* **Left open:** the prompt says "assert the gradient is zero" and never says
  *why you are entitled to an exact equality*. You are entitled to it because
  the quantity is structurally zero, not numerically small: for $d \geq 100$ the
  overlap width is $\max(0, 100-d) = 0$, so the intersection is identically zero,
  so IoU is **constant** on the whole disjoint region. The clamp's gradient is
  zero on its flat side and torch propagates that exactly. Had you not known
  that, a tolerance would have been the honest choice and the lesson would have
  been lost.
* **The usual student version:** two real defaults conspire here. `torch.tensor(150.0)`
  has **`requires_grad=False`** — asking for its gradient raises
  `RuntimeError: element 0 of tensors does not require grad and does not have a
  grad_fn`, which is the lucky failure because you find out. And
  `torch.get_default_dtype()` is **float32**, which is the unlucky one: the
  claim here is about an exact zero, and in float32 you cannot distinguish zero
  from 1e-9, so the strongest sentence available to you collapses into "it gets
  small". Ask for float64 explicitly.
* **How you would catch it:** the difference between "vanishing" and "absent" is
  the difference between a hard optimisation problem and an impossible one. A
  constant has **no descent direction** — not a weak one, none — so no optimiser,
  learning rate, warm-up or initialisation repairs it. Anything you read that
  says the IoU gradient "goes to zero" is describing a limit; this is an
  identity, and the assert is what tells the two apart.

---

## Cell 11 — CIoU, and the term you cannot see

CIoU adds two penalties to IoU: $\rho^2/\ell^2$, the squared distance between
centres over the squared diagonal of the enclosing box, and $\alpha v$, which
measures aspect-ratio disagreement.

**Prompt to type:**

> Add CIoU. Then compare `[0,0,100,100]` against a disjoint box of the same
> shape, `[120,0,220,100]`, and against a disjoint box of a different shape,
> `[120,0,320,50]`. **Print the three terms separately** — IoU, the centre-distance
> penalty and the aspect penalty — not just the CIoU total.

**Expect:**
```
same 100x100       IoU 0.000  rho2/l2 0.2466  alpha*v 0.0000  CIoU -0.2466
different 200x50   IoU 0.000  rho2/l2 0.2627  alpha*v 0.0125  CIoU -0.2752
```

**Assert:**
```python
assert alpha_v_same == 0.0        # exactly zero when the shapes agree
```

**Annotate:** short

> **input** · a same-shaped disjoint box and a differently-shaped one ·
> **output** · IoU, the centre-distance penalty and the aspect penalty, **as three columns** ·
> **constraint** · print the terms separately. The two boxes differ in aspect ratio AND in centre position AND in enclosing diagonal, so a comparison of the two totals cannot tell you which term moved ·
> **check** · the aspect term is exactly 0.0000 for the same-shaped pair, and 0.0125 for the other — which is **45% of the 0.0277 gap between the totals**, not all of it

*Verified:* totals differ by 0.0277; 0.0125 of that is the aspect term and
0.0161 is centre distance. This is why the prompt insists on three columns. The
current notebook prints only the totals and attributes the whole gap to aspect
ratio.

**An honest caveat, and say it out loud here.** GIoU and CIoU are **losses**:
their value is in the backward pass of a detector you are fitting. Nothing in
this notebook fits a detector. What you have just measured is a property of the
functions, not an improvement in a model we built. The rest of the notebook is
about evaluation, which we *can* measure.

*Examinable:* §2 and §3 — the formula, the clamp, and why the gradient is
identically zero on the disjoint region.

---

# §4 · Average precision

## Cell 12 — run the detector

**Prompt to type:**

> Load `fasterrcnn_resnet50_fpn` with the `COCO_V1` weights, put it in eval
> mode, and run it over the 128 images one at a time. Store each image's boxes,
> scores and labels as numpy. Print how long it took and on what device.

**Expect:** `128 images in 15.6 s on mps`, and a `preds` dict of 128 entries.
Average **34.5 detections per image** — the model's default score floor is 0.05,
so most of those are near-worthless and the ranking will sort them to the bottom.

**Assert:**
```python
assert len(preds) == N_IMAGES
```

**⏱ MPS 15.6 s. CPU (2 threads) 4.06 s/image → about 8 min 40 s for 128.**
Start it and read on. Every cell from here to Cell 20 reuses `preds`; do not
re-run this cell.

**Annotate:** full

* **Left open:** what the integers in `labels` mean. torchvision's detection
  models emit **COCO category ids**, which run 1 to 90 **with gaps** —
  `weights.meta["categories"]` is a 91-entry list whose index 0 is
  `'__background__'` and which contains **10 `'N/A'` placeholders** where COCO
  skipped a number. I checked: `categories[category_id]` gives the right name for
  all 80 real classes. So `names[c]` works and `names[label_index]` on a
  0..79 assumption does not.
* **The usual student version:** assuming a contiguous 0..79 label space,
  because that is what every classification head you have written emits and what
  `ImageFolder` gives you. Then `names[12]` returns `'N/A'`, your per-class
  tables are shifted, and nothing raises. This is the same shape of error as
  scikit-learn's ensembles re-encoding `y` to positions 0..k−1: the model's
  label space is a convention, and reading it off the weights object is one line.
* **How you would catch it:** print `names[1]` and check it says `person` — then
  print `names[12]`. Two lines, before you build anything on top. The second one
  is the test; the first one passes under both conventions and would have
  reassured you for the wrong reason.

While you wait, note the three defaults you did not choose and that this cell
just accepted: **`box_score_thresh=0.05`**, **`box_nms_thresh=0.5`**,
**`box_detections_per_img=100`**. All three are in `FasterRCNN.__init__`; §7 is
about the second one.

---

## Cell 13 — matching, precision, recall

A detector is a **ranking** — Lecture 4's shape of problem, on boxes. For one
class and one IoU threshold: sort every detection of that class **over the whole
corpus** by score; walk down the list; for each detection find the best
**unmatched** annotation in the same image; IoU at least $t$ makes it a true
positive and uses that annotation up, otherwise a false positive.

**Prompt to type:**

> Write `pr_curve(cls, t)` returning cumulative recall and precision for one
> class at one IoU threshold, plus the number of ground-truth instances. Collect
> every detection of that class across all 128 images, sort by score descending,
> and match each one greedily to the best still-unmatched annotation in its own
> image. Run it for class 1 at IoU 0.5 and print the counts.

**Expect:**
```
person: 350 annotations, 1209 detections in the ranking
true positives: 310
highest recall reached: 0.886
```

**Assert:** none directly — Cell 14 turns these three numbers into three asserts,
which is the better test.

**Annotate:** full

* **Left open:** the word **unmatched**. The prompt says it, and it is the only
  word in the prompt doing real work: it is what makes a second box on the same
  bottle a false positive rather than a second success. Remove it and a detector
  is rewarded for duplicating its confident predictions — which, given §7, is
  something detectors are extremely good at.
* **The usual student version:** ranking **within each image** rather than across
  the corpus. It is the natural loop to write, because your data is already a
  dict keyed by image. Cell 20 measures exactly what it costs: **+0.157 mAP**,
  free. And there is a second real one: `np.argmax` on an empty array raises
  `ValueError: attempt to get argmax of an empty sequence`, and a class with no
  annotation in a given image produces exactly that. The guard is one `if`, and
  without it the function works for `person` and dies on class 73.
* **How you would catch it:** mark the annotation used **at the moment it is
  matched**, inside the loop, not afterwards. Three lines of bookkeeping, and it
  is the entire difference between average precision and a count of overlaps. To
  test it: run the function on one image with two identical high-scoring boxes on
  one annotation. You must get one TP and one FP.

---

## Cell 14 — precision is not monotone

**Prompt to type:**

> Classify every step of the person precision curve as down, up or flat. Then
> assert the identity: every false positive is a step down, and every true
> positive except the first is a step up or flat.

**Expect:**
```
steps down (precision falls) : 899
steps up   (precision rises) : 254
steps flat (already at 1)    : 55
total steps                  : 1208
```
1209 detections, 310 true positives, 899 false positives. 254 + 55 = 309 = 310 − 1.

**Assert:**
```python
assert down == n_fp
assert up + flat == n_tp - 1
assert down + up + flat == len(precision) - 1
```

**Annotate:** short

> **input** · the person class's precision curve ·
> **output** · the step counts, checked against the true and false positive counts ·
> **constraint** · classify EXACTLY, with a tolerance of 1e-12, and assert the identity rather than observing that the curve is jagged ·
> **check** · you can predict all three numbers from 1209 and 310 before running it. If an assert fails, either the matching or the counting is wrong and the three numbers tell you which

The 55 flat steps are one run: `np.flatnonzero(precision < 1.0)[0]` is **56**, so
the top 56 person detections in the whole corpus are all correct and precision
sits at 1.0 with nowhere to rise to.

---

## Cell 15 — average precision

Lecture 4 proved precision has no monotone envelope you can rely on. AP is
defined against the **maximum precision at or above each recall level**, and that
maximum exists for exactly one reason: to replace a non-monotone quantity with a
monotone one.

$$p_{\text{env}}(r) = \max_{\tilde r \geq r} p(\tilde r),
  \qquad \mathrm{AP} = \int_0^1 p_{\text{env}}(r)\,\mathrm{d}r$$

No threshold appears anywhere. That is the whole point of it.

**Prompt to type:**

> Write `envelope(p)` — the maximum precision at or above each recall level, one
> right-to-left sweep. Then `average_precision(precision, recall)`, the area
> under the enveloped curve, all-point definition. Check it against a case I can
> do on paper: four detections, two annotations, labelled TP FP TP FP in score
> order, where the answer is 0.5 + 1/3.

**Expect:**
```
hand-computable case passes: 0.8333
AP for person at IoU 0.5, on 128 images: 0.751
```

**Assert:**
```python
assert abs(average_precision(hand_p, hand_r) - (0.5 + 1/3)) < 1e-12
```

**Annotate:** short

> **input** · a precision-recall curve ·
> **output** · the AP, plus the four-detection case worked on paper ·
> **constraint** · the all-point definition, and a hand case whose answer you compute BEFORE you run it — after 1: P=1, R=0.5; after 3: P=2/3, R=1.0; envelope gives (0.5−0)×1.0 + (1.0−0.5)×2/3 ·
> **check** · the hand case to 1e-12, not to two decimals. There is a real alternative — the 11-point interpolation of the 2007 VOC papers — which gives a different number on the same curve, and 1e-12 is what tells you which one you implemented

---

## Cell 16 — draw it

**Prompt to type:**

> Two panels. Left: the person PR curve and its envelope, as step plots with
> `where='post'`, with the area under the envelope shaded and the AP in the
> title. Right: zoom in on detections 50 to 140 in score order, so the sawtooth
> is visible, **and use a step plot there too**.

**Expect:** left panel a shaded staircase, `AP = 0.751` in the title. Right panel
a red sawtooth under a green staircase.

**Assert:** none.

**Annotate:** short

> **input** · the curve and its envelope ·
> **output** · the full PR curve with the area shaded, and a zoomed window on the sawtooth ·
> **constraint** · `where='post'` on **both** panels. A PR curve drawn with linear interpolation claims performance at recall levels that were never achieved, and the zoom panel is the one place where that matters most ·
> **check** · the shaded area should look like 0.75 of the box. If it looks like 0.9, you shaded under `precision` rather than under `env`

---

# §5 · mAP, a mean of a mean

## Cell 17 — the sweep

**Prompt to type:**

> Find which categories actually appear in the 128 images. For each of them, at
> each of the ten IoU thresholds 0.50 to 0.95 in steps of 0.05, compute the AP.
> Skip categories with no annotations rather than scoring them zero. Print mAP
> at 0.50, at 0.75, and averaged over the ten.

**Expect:**
```
73 of COCO's 80 categories appear in our 128 images
730 class-threshold pairs in 1 s

mAP @ 0.50            0.659
mAP @ 0.75            0.475
mAP @ [0.50:0.95]     0.439
```

**Assert:** none.

**⏱ 1 second.** Pure NumPy on cached predictions — the same on CPU and GPU. It
is 730 calls to `pr_curve` and each one sorts a few hundred detections.

**Annotate:** short

> **input** · 73 classes × 10 IoU thresholds ·
> **output** · mAP at 0.50, at 0.75, and averaged over the ten ·
> **constraint** · only classes that ACTUALLY APPEAR. A class with no annotations has no AP, and scoring it zero invents a measurement — and drags the mean down by an amount that depends on your sample, not on your detector ·
> **check** · 73 classes, and say it out loud beside "128 images". Both are part of the number

---

## Cell 18 — what the first mean hides

**Prompt to type:**

> Print the per-class APs at IoU 0.50 sorted, with each class's instance count
> beside it. Report best, median, mean and worst, how many classes score exactly
> 1.000, and how many are below the mean.

**Expect:**
```
classes scoring exactly 1.000: 13
  and their instance counts: [1, 1, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 6]

best   bus          1.000 (6 instances)
median              0.673
mean   = the mAP    0.659
worst  hot dog      0.000 (1 instance)

classes below the mean: 29
```

**Assert:** none.

**Annotate:** short

> **input** · the per-class APs at IoU 0.50 and the instance counts ·
> **output** · best, median, mean, worst, the perfect scorers, and the count below the mean ·
> **constraint** · print the INSTANCE COUNT beside every class you name. The mean is unweighted and that is the finding ·
> **check** · thirteen classes tie at exactly 1.000, so whichever one your sort puts first is not "the best class" — it is the first of thirteen. Print the tie rather than the winner

**Read the numbers, not the sentence you expected.** 29 of 73 classes — **40%**,
not "more than half" — are below the mean, and the median (0.673) sits *above*
it. The 13 perfect classes hold between 1 and 6 annotations each. `person`, with
350 of the corpus's 898 objects (39.0%), scores 0.751 and counts for exactly
1/73 of the mean, the same as `hot dog` with one annotation and an AP of zero.
Classes with three or fewer instances average **0.652**; classes with ten or more
average **0.559**. A mean over categories does not care that one category is 39%
of the corpus. If you want it to, you need a different statistic, and you should
say which.

---

## Cell 19 — what the second mean hides

**Prompt to type:**

> Plot mAP against the IoU threshold for the ten thresholds, with a horizontal
> line at the mean over the ten. **Take the two endpoint values from the data
> rather than typing them into the title.**

**Expect:** a curve falling from **0.659** at threshold 0.50 to **0.040** at 0.95,
with a dashed line at 0.439 crossing it between 0.75 and 0.80.

**Assert:** none.

**Annotate:** short

> **input** · mAP at each of the ten IoU thresholds ·
> **output** · the curve, with the mean drawn across it ·
> **constraint** · every number in the title interpolated from the array. A figure whose title is a string literal keeps printing yesterday's numbers after you change the data, and nothing warns you ·
> **check** · the mean line must cross the curve exactly once, between 0.75 and 0.80 — you can predict that from mAP@0.75 = 0.475 being above 0.439

"mAP 0.439" is an average over ten quite different questions about how precisely
a box must be placed, and one number in the middle stands for both ends.

*Examinable:* §4 and §5 — the ranking, the envelope, and both means.

---

# §6 · Report mAP over the dataset

## Cell 20 — per-image averaging

**Prompt to type:**

> Report mAP over the dataset: for each image, compute the AP of each class
> present in it at IoU 0.5, average those, and then average over images. Print it
> beside the number from Cell 17.

**Expect:**
```
accumulated over the corpus, correctly : 0.659
computed per image, then averaged      : 0.816
free mAP                               : +0.157
```

**Assert:**
```python
assert wrong > m50, "the per-image version should be optimistic"
```

**⏱ under a second.** NumPy on cached predictions.

**Annotate:** full

* **Left open:** the phrase **"over the dataset"**, which is the whole prompt.
  It reads as a scope — *all of it, not a subset* — and the code that comes back
  reads it as an iteration order. Both readings produce a loop over 128 images
  and an `np.mean`; only one of them is average precision.
* **The usual student version:** exactly this. It is what an assistant returns
  for "report mAP over the dataset", it runs, it produces a plausible number in
  the right range, and it is worth **+0.157**. Why it inflates is arithmetic, not
  bad luck: a single image usually holds one or two classes and a handful of
  objects, so its own AP is often exactly 1.0 — I counted **51 of the 128 images
  score exactly 1.000**. Averaging fifty-one easy 1.0s is not the same as ranking
  all 1,209 person detections in the corpus against each other.
* **How you would catch it:** name it. This is *the metric averaged per batch
  rather than accumulated over the set* — the same entry in the course's
  silent-failure catalogue you met in **Lecture 12**, wearing detection clothes.
  The signature is always the same: a loop over units of data with a mean at the
  end, where the metric is not a mean of per-unit means. Grep your own code for
  `np.mean` immediately after a `for ... in` over a data loader.

**A trap in the cell itself, and it is not the metric.** The natural way to write
this is to rebind the globals `preds` and `gt` to one-image dicts inside the
loop, because `pr_curve` reads them from module scope. That cell then leaves the
notebook in a broken state if it does not finish — and you *will* be tempted to
interrupt a cell that loops over 128 images. **Write it with `try` / `finally`,
or pass the dicts in as arguments.** If you interrupted it, the fix is:
re-run **Cell 12** (⏱ 8 min 40 s on CPU), then Cell 17, then Cell 18.
That is the re-run order, and it is why the argument-passing version is worth the
extra two minutes now.

---

# §7 · Non-maximum suppression

The detector in Cell 12 threw away most of its own output before you saw it,
using IoU, at a threshold you did not set.

## Cell 21 — the same 128 images with suppression off

**Prompt to type:**

> Build the same Faster R-CNN again with `box_score_thresh=0.05`,
> `box_nms_thresh=1.0` and `box_detections_per_img=300`, so almost nothing is
> suppressed, and run it over the same 128 images. Print how many candidate
> boxes per image that gives.

**Expect:** `231.7` candidate boxes per image.

**Assert:** none.

**⏱ MPS 11 s. CPU (2 threads) about 9 minutes** — it is the same forward pass as
Cell 12 with a bigger box head. On CPU this is the second of the two long cells
and there is no third.

**Annotate:** short

> **input** · the same 128 images with suppression switched off ·
> **output** · candidate boxes per image before any suppression ·
> **constraint** · raise `box_detections_per_img` to 300 as well. Leaving it at its default of 100 caps the pool at 100 and you would be measuring the cap ·
> **check** · 231.7 mean, comfortably under the 300 cap — if it comes back at 300.0 the cap is binding and the number is not a measurement

---

## Cell 22 — suppress that pool by hand

**Prompt to type:**

> Apply `batched_nms` at IoU 0.5 to that same candidate pool and print the mean
> number kept per image, next to the true number of objects per image.

**Expect:**
```
candidate boxes per image, no suppression : 231.7
the same pool after NMS at IoU 0.5        :  23.6
actual objects per image                  :   7.02
```

**Assert:** none.

**Annotate:** full

* **Left open:** which pool the "before" number describes. The prompt says *that
  same candidate pool*, and it has to, because the obvious alternative — compare
  against `preds` from Cell 12 — compares two different pipelines. Cell 12's
  model caps at 100 detections after its own NMS and returns **34.5 per image**,
  so that comparison would report a 6.7× reduction where the matched one reports
  9.8×. Both rows here come from the same 231.7 boxes; state that when you quote
  the drop.
* **The usual student version:** quoting the stock model's output as the "after"
  number without noticing that **`box_detections_per_img=100` is a documented
  default and a cap, not a result**. A number that is really a cap looks exactly
  like a number that is really a measurement, and the giveaway — the mean sitting
  suspiciously near a round default — is only visible if you know the default is
  there.
* **How you would catch it:** `batched_nms`, not `nms`. torchvision's `nms` is
  class-agnostic; `batched_nms` suppresses per class. With `nms`, a person
  standing in front of a car suppresses the car, your box count falls further,
  and the result looks *better* by the metric in this cell and worse by every
  metric in §5. Suppression that improves the number you are looking at while
  destroying the answer is the general shape to watch for.

---

## Cell 23 — another knob nobody set

**Prompt to type:**

> Run the same suppression at IoU 0.1, 0.3, 0.5, 0.7 and 0.9, and print the mean
> boxes kept per image at each.

**Expect:**
```
NMS IoU 0.1 ->   14.6 boxes per image
NMS IoU 0.3 ->   18.0
NMS IoU 0.5 ->   23.6
NMS IoU 0.7 ->   42.6
NMS IoU 0.9 ->  134.1
```

**Assert:** none.

**Annotate:** short

> **input** · five NMS thresholds on the fixed candidate pool ·
> **output** · boxes kept per image at each ·
> **constraint** · one pool, five thresholds, so the only thing varying is the knob ·
> **check** · monotone increasing, and 0.9 keeps roughly nine times what 0.1 keeps. Against 7.02 real objects per image, every one of these five is wrong in a different direction

Too low and two people standing close together become one person; too high and
every object keeps its duplicates. Count the defaults this notebook has now
accepted: **score 0.05, NMS 0.5, 100 detections per image, IoU 0.5 for matching**.
Four numbers, none of them chosen by anyone in the room — and note that the last
two are both "0.5" and are completely different quantities. Give them different
names in your code.

*Not examinable — engineering.* §7 is here so that you never again read a box
count as a property of a model.

---

# §8 · Per-pixel prediction

A box was always an approximation: a bicycle's box is mostly not bicycle.

* **Semantic** segmentation: one class per pixel. Two people standing together
  are one `person` region and you cannot count them.
* **Instance** segmentation: one mask per object. You can.

Mask R-CNN is Faster R-CNN with one extra head — two lines from what you already
ran.

## Cell 24 — Mask R-CNN on one image

**Prompt to type:**

> Load `maskrcnn_resnet50_fpn` with `COCO_V1` weights. Take the most crowded of
> the first forty images and run it. Print the shape of `masks`, and how many
> objects score at least 0.7.

**Expect:**
```
masks shape torch.Size([100, 1, 302, 500])  -- (N, 1, H, W), soft in [0, 1]
45 objects at score >= 0.7, 2 distinct classes
```
The image is `000000002299.jpg`, 500 × 302, with 22 non-crowd annotations.

**Assert:**
```python
assert out["masks"].ndim == 4 and out["masks"].shape[1] == 1
```

**⏱ 0.4 s on MPS, a couple of seconds on CPU — plus a 178 MB weight download the
first time.** The download dominates and is bandwidth-bound.

**Annotate:** short

> **input** · the most crowded of the first forty images ·
> **output** · soft masks, labels, scores, and how many objects survive a 0.7 cut ·
> **constraint** · `masks` is `(N, 1, H, W)` and SOFT in [0, 1], not binary. Take `masks[:, 0]` once, explicitly, rather than squeezing wherever you happen to need it ·
> **check** · assert the rank is 4 and axis 1 has length 1. `masks[j]` gives you object *j* with shape `(1, H, W)` — the right object with a spare axis, which then broadcasts your overlay to `(1, H, W, 3)` and `imshow` raises `TypeError: Invalid shape`. Loud, not silent, but only if you get that far

---

## Cell 25 — semantic beside instance

**Prompt to type:**

> Three panels: the image, the same image with the masks coloured **by class**,
> and with the masks coloured **by object**. Alpha-blend the soft masks at 0.55
> rather than thresholding them.

**Expect:** panel two shows two colours (person, tie); panel three shows 45.
That difference *is* the distinction between semantic and instance segmentation,
and one panel cannot show it.

**Assert:** none.

**Annotate:** short

> **input** · the masks and their labels ·
> **output** · the image, a semantic overlay and an instance overlay ·
> **constraint** · colour by CLASS in one panel and by OBJECT in the other ·
> **check** · panel titles carry the two counts — 2 classes, 45 objects — read off the data, not typed in. The blend `img*(1-0.55m) + 0.55m*c` has coefficients summing to exactly 1, so it is a convex combination and stays inside [0,1] however many times you apply it; I ran 45 blends and the extremes were 0.0126 and 0.9983. You do not need a clip, and if you find yourself wanting one, something else is wrong

The soft masks are blended rather than thresholded on purpose: the boundaries
come out visibly uncertain, and that uncertainty is real. Thresholding at 0.5
looks cleaner and asserts a confidence the model never expressed.

---

## Cell 26 — and what the annotator recorded

**Prompt to type:**

> For that same image, print what COCO actually annotates — class, count, and
> whether it is a crowd region — beside what the detector found, broken down by
> class.

**Expect:**
```
detector at score >= 0.7:  person 38,  tie 7
COCO annotates:
  person     13
  tie         9
  person      1   (crowd region)
```

**Assert:** none.

**Annotate:** short

> **input** · the raw annotations for that image, and the detector's output ·
> **output** · both counts **broken down by class**, with crowd regions marked ·
> **constraint** · break the detector's 45 down by class too. "45 against 22" is not a comparison you can reason about; "38 people against 13 people plus one crowd polygon" is ·
> **check** · the crowd polygon's area is **87,090 px, 57.7% of the image**. That is what a single annotation covering a crowd looks like, and it is where the other 25 people went

The conclusion is uncomfortable and it is the right one: **the detector is
probably right and the ground truth is probably not wrong.** They answer
different questions. The annotator drew one polygon around a crowd because that
is the COCO convention for a crowd; the detector found the people in it. Every
metric in this lecture — every AP, every mAP, the count MAE from Lecture 17 — is
measured against the annotator's decision, including the decision to draw one
polygon around twenty-five people. When your model disagrees with the ground
truth, look at the ground truth: it is a record of what somebody decided, not a
record of what is there.

*Beyond the book, for context:* §8. You are not examined on Mask R-CNN.

---

# §9 · Where we ended up

| Application 9, 128 images of COCO val2017 | Value |
|---|---|
| Count MAE, the metric committed to in Lecture 17 | 3.00 |
| mAP at IoU 0.50 | **0.659** |
| mAP averaged 0.50 to 0.95 | **0.439** |
| The same weights, official protocol, all 5,000 images | 0.370 |
| *(check)* the same weights, official protocol, **our** 128 images | *0.441* |

The first row is not wrong; it is blind — counting cannot distinguish nine right
boxes from nine wrong ones. The last two rows are the ones that keep the middle
two honest, and the fifth row is worth adding because it is the row that settles
*why* the numbers differ.

Our 0.439 is **6.9 points above** torchvision's published 0.370 for identical
weights. Two explanations are available: our sample of 128 is easier than the
full 5,000, or our hand-written AP is more generous than the official
`pycocotools` protocol (101-point interpolation, area ranges, `maxDets=100`,
crowd-region handling). **I ran `pycocotools` on our own 128 images: 0.4413,
against our 0.4388.** So the method accounts for **−0.25 points** and the sample
for **+7.1**. The gap is the sample, and now that is measured rather than
asserted. Add that row to your notebook — it costs one `COCOeval` call on
predictions you already have.

## §10 · Red-team

Swap notebooks. Fifteen minutes. Five questions:

1. What touched the test set? *(was any threshold chosen by looking at the 128?)*
2. What was fitted, and on what? *(nothing was fitted here — say so)*
3. What is the shape here? *(`masks` is `(N, 1, H, W)`, soft, not `(N, H, W)`
   and not binary)*
4. What was dropped — rows, columns, NaNs? Count them. *(8 crowd regions, and
   every detection below the 0.05 score floor)*
5. What is the default I did not ask for? *(score 0.05, NMS 0.5, 100 detections
   per image, IoU 0.5 for matching)*

Three bugs to hunt for by name. All three run. All three produce a plausible
number. Two of the three make the score look better.

* **The missing clamp.** Feed their IoU function `[0,0,100,100]` and
  `[200,200,300,300]`. If it says 1.0, you have it. Eight seconds. Then feed the
  same pair to the version they "fixed" with `max(0, result)` — it still says 1.0.
* **Per-image AP.** A loop over images with an `np.mean` at the end. Worth
  +0.157 on this corpus.
* **Corner against size.** Assert `x2 >= x1 and y2 >= y1` on every box in the
  notebook. COCO stores `x, y, w, h`; every model here emits corners; the two
  formats are indistinguishable by shape and by dtype.

---

# Defects found in the current notebook

Everything below is against `notebooks/lecture-18.ipynb` as it stands (74 cells,
24 code cells, no stored outputs). Each entry says how it was checked.

## Verified with `python3` — reproduced against the data

**1 · §7.1 — four of the six ⏱ figures are wrong, two of them by two orders of
magnitude.** Measured on MPS with torch 2.13.0 / torchvision 0.28.0:

| Cell | Notebook says | Measured, MPS | Measured, CPU (2 threads) |
|---|---|---|---|
| 34, detector | "1 to 2 minutes … several minutes on CPU" | **15.6 s** | **~8 min 40 s** (4.06 s/img) |
| 49, mAP sweep | "about 60 seconds" | **1 s** | **1 s** (pure NumPy) |
| 58, per-image AP | "about 30 seconds" | **<1 s** | **<1 s** (pure NumPy) |
| 62, NMS off | "about 60 seconds" | **11 s** | **~9 min** |
| 67, Mask R-CNN | "about 20 seconds" | **0.4 s** + 178 MB download | few s + download |

The two NumPy cells are device-independent, so 60 s and 30 s are wrong on every
machine. More seriously, §7.1 requires the CPU number and the notebook gives
"several minutes" once, for one cell, and nothing for cell 62 — which is the
*other* eight-minute cell. A reader on a Colab CPU runtime is told to expect
about four minutes of work in total and will spend about eighteen.

**2 · §6.2 — the label-dtype failure described in cell 33's box does not
happen.** The constraint says *"a label of 1.0 will not match a category_id of 1
in a boolean mask, and the resulting comparison is silently all-False"*, and the
student bullet says omitting the fix-up *"makes every per-class mask empty and
every AP zero, with no error anywhere"*. In NumPy, `np.array([1.,3.,1.]) == 1`
is `[True, False, True]`. I ran the entire AP sweep both ways:

```
mAP@0.50 with labels left as float64 : 0.659424
mAP@0.50 after .astype(np.int64)     : 0.659424
difference                           : 0.00e+00
```

The `.astype(np.int64)` line is cosmetic. Since this is the notebook's own
"observed failure" for that cell, it is the exact bullet §6.2 says to cut.

**3 · §3.3 — "Read rows four and five. Two boxes 150 px apart and two boxes 300
px apart" (cell 25) does not resolve.** The cell prints `rows[::5]`, so the
printed rows are d = 0, 50, 100, 150, 200, 250, 300. Rows four and five are
**150 and 200**. The sentence needs rows four and **seven**. Counted with
`np.arange(0, 301, 10)[::5]`. The same wrong reference is repeated in cell 26's
student bullet.

**4 · §3.3 — "the figure above" in cells 29 and 30 refers to a figure that does
not exist.** Cell 29's constraint says *"every pair in the figure above is
square"* and its student bullet warns against *"concluding from the earlier
figure that CIoU and GIoU behave identically"*. Cell 30 prints the same phrase as
program output. The notebook contains exactly four figures — cells 18, 46, 55 and
69 — and I checked all four: **GIoU and CIoU are never plotted**. GIoU appears
only in cell 24's printed table. The nearest figure above is cell 18's, which
plots `iou` against `iou_broken`, so a reader who goes back to look finds a
picture that says nothing about the claim.

**5 · §1.1 / §2.1 — the CIoU demonstration is confounded and the notebook prints
only the totals.** Cell 30 compares a same-shaped disjoint box against a
differently-shaped one and attributes the difference to the aspect-ratio term.
Decomposing both:

```
same 100x100       IoU 0.000  rho2/l2 0.2466  alpha*v 0.0000  CIoU -0.2466
different 200x50   IoU 0.000  rho2/l2 0.2627  alpha*v 0.0125  CIoU -0.2752
```

The two boxes differ in aspect ratio **and** in centre distance **and** in
enclosing diagonal. Of the 0.0277 gap between the printed totals, **0.0125 is the
aspect term and 0.0161 is centre distance** — the term the cell exists to
demonstrate is 45% of the effect it is credited with. The claim "the aspect term
is exactly zero when the shapes agree" is true (0.0000 exactly) and the cell's own
output cannot show it.

**6 · §1.1 — "More than half the classes are below it" (cell 51) is false.**
Measured: **29 of 73 classes, 40%**, are below the mAP of 0.659; the median
(0.673) sits above the mean. The companion claim, "the ones above are mostly the
rare ones", does hold in the direction stated — classes with ≤3 instances average
0.652, classes with ≥10 average 0.559 — but the headline fraction is wrong and it
is the number a reader would repeat.

**7 · §1.1 — two figures are hard-coded as string literals in plotting code.**
Cell 55 has `plt.title("128 images: 0.659 at the loosest threshold, 0.040 at the
tightest")` and the same two numbers appear in cell 54's box. They happen to be
correct today — I measured 0.6594 and 0.0403 — but they are transcribed, not
derived, so they will keep printing after the data changes and nothing will warn
anyone. This is the §5.3/§1.1 pattern that produced lecture 19's `<- lowest`
annotating the wrong number.

**8 · §6.2 — the `np.clip(img, 0, 1)` catch in cell 68 describes an impossible
failure.** The bullet says *"repeated alpha blending can drift outside the range,
and matplotlib's response to that is to rescale the whole image"*. The blend is
`img*(1-0.55m) + 0.55m*c` whose coefficients sum to exactly 1 — a convex
combination of values already in [0,1]. I ran 45 successive blends on random
data: extremes 0.0126 and 0.9983, never outside the range. The `np.clip` is
harmless and the reason given for it is invented.

**9 · §6.2 — "indexing it as `(N, H, W)` silently gives you the first object,
repeated" (cell 66's constraint, repeated as red-team question 3 in cell 73) is
wrong twice.** With `masks` of shape `(N, 1, H, W)`, `masks[j]` returns **object
j** with shape `(1, H, W)` — the correct object with a spare axis, not the first
one. And it is not silent: blending `masks[j][..., None]` against an `(H, W, 3)`
image broadcasts to `(1, H, W, 3)`, and `imshow` raises `TypeError: Invalid shape
(1, H, W, 3) for image data`. Both checked directly. The hazard is real; the
description of it is not, and it is stated twice.

**10 · §2.1 — "The detector you ran had already thrown away nine tenths of its
own output" (cell 60) is a claim about a pipeline the notebook never measures.**
The code is careful and its comment is right: it suppresses the same 231.7-box
pool and gets 23.6, a ratio of 0.102. But the sentence attributes that ratio to
*the detector you ran*, which is cell 34's stock model, and that model returns
**34.5 boxes per image** — a ratio of 0.149, six sevenths rather than nine tenths.
The stock figure is never printed, so the reader cannot see the substitution. The
fix is one word in the sentence, not a change to the code.

**11 · §8.1 — the missing clamp is announced five times before a reader can be
caught by it.** Counted in source order: (a) cell 0's header, *"an IoU function
that reports two boxes 200 pixels apart as a perfect match"*; (b) cell 8's student
bullet, *"omitting the clamp, which is the next cell"*; (c) cell 9's code, which
is the **correct** function, whose docstring is three lines explaining what the
clamp is for; (d) cell 10's `⚠ Read before running`; (e) the prompt label
`⚠ what the assistant returns`; and (f) the comment `# <- no clamp` on the
offending line itself. By the time the reader reaches the cell there is nothing
to find. The strongest single fix is (c): the correct function is shown *first*,
so the defect is a spot-the-difference against a solution the reader already has.
Reordering — broken function, overlapping tests, disjoint tests, *then* the
contrast and the clamp — costs nothing and restores the "would you have caught
it?" question. This script does that. The second defect (per-image AP) is
announced three times before its cell, in cells 56, the box label, and the code
comment.

**12 · §3.3 — "which is the next cell" (cell 8) points three cells away.** Cell 8
is the box for code cell 9, which is the *correct* IoU. The broken function is
code cell 12, with markdown cells 10 and 11 in between.

**13 · §4.1 — `inst` is bound to two different types.** Cell 52:
`inst = collections.Counter()`. Cell 69: `inst = base.copy()`, an `(H, W, 3)`
float array. Same module scope, seventeen cells apart, both live at the end of a
restart-and-run-all. Two more of the same shape: **`d`** is a NumPy array of
separations in cell 18 and a scalar `torch.Tensor` with `requires_grad=True` in
cell 24; **`g`** is a ground-truth dict in cells 6 and 37, a gradient tensor in
cell 24 (`g, = torch.autograd.grad(v, d)`) and then, *eleven lines later in the
same cell*, the GIoU **value** in `for dv, i, gi, g, gg in rows[::5]`. Found by
walking the AST of all 24 code cells. Also worth renaming: `t` is the IoU
matching threshold in cell 49 and the NMS threshold in cell 64 — the notebook's
own §8 argues these are different knobs that happen to share the value 0.5, and
then gives them the same name.

**14 · §4.3 — cell 58 leaves the notebook broken if it does not finish.** It
rebinds the globals `preds` and `gt` to one-image dicts and restores them on the
last line, with no `try`/`finally`. The cell's own box warns that forgetting to
restore them *"breaks every cell below with no obvious cause"* — and then does not
protect against the case a reader will actually hit, which is interrupting a cell
advertised as taking 30 seconds. Recovery requires re-running cell 34, the
eight-minute one, and the notebook does not say so.

**15 · §8.3 — the string "examinable" appears zero times in the notebook.** §8.3
requires every section to carry one of *examinable*, *not examinable —
engineering*, or *beyond the book, for context*. Lecture 19 has it once; lecture
18 has it never. Counted with `grep -ci`.

**16 · §6.1 — 24 code cells, 24 full three-bullet annotations.** Counted from the
notebook. The budget is five to eight. This is the measured defect: all three
student readers stopped reading the template around cell 30, and in this notebook
cell 30 is the property test and cell 33 is the detector's box — two of the
places most worth reading.

## Verified, and the notebook is right

Recorded because the brief asks which claims I checked, not only which ones
failed.

* **898 objects, 8 crowd regions, 73 of 80 categories, 350 `person` (39.0%)** —
  all reproduce from the published `instances_val2017.json`. `val2017` does hold
  5,000 images and 80 categories.
* **mAP 0.659 / 0.475 / 0.439, mAP@0.95 = 0.040** — reproduced to three decimals
  (0.6594, 0.4749, 0.4388, 0.0403). **Count MAE 3.00** reproduced exactly from
  Lecture 17's `THRESH = 0.5`. **torchvision's 37.0** is
  `FasterRCNN_ResNet50_FPN_Weights.COCO_V1.meta["_metrics"]`, checked.
  **"6.9 points higher"** is 0.4388 − 0.370, correct.
* **The 6.9-point gap really is the sample.** `pycocotools` `COCOeval` on the
  same 128 images gives **0.4413** against the notebook's 0.4388, so the method
  contributes −0.25 points and the sample +7.1. The notebook asserts this and is
  right; it is worth adding the row, because the alternative explanation is
  obvious and unaddressed.
* **The broken IoU returns exactly `1.0`** for `[0,0,100,100]` vs
  `[200,200,300,300]`, and −0.5 / 0.143 for the other two disjoint pairs.
* **The gradient really is identically zero** past 100 px — 20 separations, max
  IoU 0.0, max |gradient| 0.0, GIoU still descending at −0.00125.
* **The step identity holds**: 1,209 person detections, 310 TP, 899 FP; 899 down,
  254 up, 55 flat; 254 + 55 = 309; the 55 flat steps are one run and
  `precision < 1.0` first at index 56.
* **Per-image AP inflates by +0.157** (0.8165 vs 0.6594), and **51 of the 128
  images score exactly 1.000**, which is the mechanism the notebook names.
* **NMS**: 231.7 → 23.6 at IoU 0.5 against 7.02 true objects; the sweep gives
  14.6 / 18.0 / 23.6 / 42.6 / 134.1.
* **The crowd-region explanation for cell 72 holds.** The busiest of the first
  forty is `000000002299.jpg`; COCO records 13 `person`, 9 `tie` and one
  `person` crowd region whose area is **87,090 px, 57.7% of the image**; the
  detector finds 38 people and 7 ties at score ≥ 0.7. The gap is 25 people inside
  one polygon. The notebook's conclusion is right — but it prints only "45
  objects" against a class-by-class COCO list, so a reader cannot see that 25 of
  the 45 are people the annotator lumped. Print the detector's breakdown too.
* **`torchvision` defaults quoted in cell 63** — `box_score_thresh=0.05`,
  `box_nms_thresh=0.5`, `box_detections_per_img=100` — are the real signature
  defaults, checked with `inspect.signature`.
* **§5.1 / §5.2 markdown** — no line indented ≥4 spaces outside a fence, no
  indented fence marker, in any of the 50 markdown cells. **§3.1** — the notebook
  contains no ```` ```python ```` block in markdown, so there is nothing to
  mismatch.
* **The 241 MB download figure** is right as MiB: `Content-Length: 252,907,541`
  = 241.2 MiB.

## Could not check

* **The download timings.** Both the annotations zip and the 128 JPEGs were
  already on disk, and re-downloading 241 MiB to time it would be dishonest about
  a reader's bandwidth anyway. Defect 1 above therefore covers only the compute
  cells. What I can state is the byte count and that the image fetch is 128
  separate HTTP requests.
* **Whether cell 46's zoom window `lo, hi = 50, 140` is a good window.** It
  happens to sit right where the flat run ends (index 56), so the panel does show
  the sawtooth on this corpus. Whether it was chosen for that reason or is a
  coincidence I cannot tell from the source.
* **§4.4, provenance.** Cell 0 describes the boxes as specifications rather than
  transcripts, which is the honest claim and is what `_prompt.py` documents, so
  there is nothing to falsify — but it also means no bullet in the notebook can
  be defended as "observed" unless it names a library default. Defects 2, 8 and 9
  are the three where that shows.
* **Restart-and-run-all.** I did not execute the notebook end to end — the brief
  forbids running training cells and I chose not to run the notebook itself. I
  reproduced every cell's computation in a standalone script instead, in notebook
  order, and everything that script asserts, passes.
