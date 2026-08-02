# Lecture 17 — *Where is it, exactly?* — Colab prompt script

Rebuild `notebooks/lecture-17.ipynb` in Colab by prompting. Sixteen code cells.
Type the prompt, read what comes back, check it against **Expect**, run the
**Assert**. Where a cell is marked **Annotate: full**, type the three bullets
into the markdown box above it as well — those are the seven places in this
notebook where reading the prompt is the lesson.

**Application 9.** Run a pretrained detector over a stated **128-image** subset
of COCO `val2017`, count the objects, and discover that counting cannot say how
wrong a box is.

---

## Before you sit down

**Corpus.** `N_IMAGES = 128`. COCO `val2017` is 5,000 images and the full
release is about 20 GB; neither is downloaded. Every number in this notebook is
a measurement on 128 images and you are expected to say "128 images" whenever
you quote one.

**Wall clock.** Three cells cost real time. The byte counts are exact; the two
download rows are arithmetic from those bytes and your own link speed, and only
the last row is a stopwatch measurement.

| Cell | What it costs | Time |
|---|---|---|
| 2 | `annotations_trainval2017.zip`, 252,907,541 bytes (241 MiB), plus 128 JPEGs totalling 20,325,087 bytes in 128 sequential requests | link-bound; see below |
| 7 | `fasterrcnn_resnet50_fpn_coco-258fb6c6.pth`, 167,502,836 bytes | link-bound; ~13 s at 100 Mbit/s |
| 8 | 128 forward passes | **measured: 12.8 s** on Apple M4 MPS (0.10 s/image); **361.4 s — six minutes** on the same machine with MPS disabled, 16 CPU threads (2.82 s/image) |

For cell 2, do the division rather than trusting a number: 253 MB on the wire is
about 20 s at 100 Mbit/s and about 100 s at 20 Mbit/s, and the 128 JPEGs add 128
round trips on top of their 20.3 MB. **A fresh Colab VM has an empty disk**, so
this is a per-session cost, not a one-time one. If you are on a slow link, start
cell 2 and go and make coffee.

**Prerequisite.** None of the other notebooks. This one downloads its own data.

**GPU.** Not required, and the notebook is correct without one — but cell 8 is
six minutes on CPU against thirteen seconds on an accelerator, so if Colab
offers you one, take it. Nothing else in the notebook cares.

---

## Section 1 · Setup

*Not examinable — engineering hygiene.*

Type this markdown first:

> # Where is it, exactly?
>
> **Lecture 17 · Build** · Géron, Chapter 12
>
> **The corpus is 128 images.** COCO's `val2017` split is 5,000 images and the
> full release is about 20 GB. Neither is downloaded here.
>
> ## 1 · Setup

## Cell 1 — setup

**Prompt to type:**

> setup cell for a teaching notebook: import numpy, torch, torchvision, matplotlib, PIL, and json/time/pathlib. seed 42 for numpy and torch. print the python, torch and torchvision versions. pick cuda if available, else mps, else cpu, and print it. add a constant N_IMAGES = 128 and make a datasets/coco folder.

**Expect:** five printed lines — three versions, the device, and nothing else.
`N_IMAGES` a module-level constant with `128` on the right-hand side, not a
literal `[:128]` buried in a slice later.

**Assert:** none.

**Annotate:** short

*If the assistant writes `images[:128]` anywhere instead of `images[:N_IMAGES]`,
send it back. The constant is the reason the corpus size appears in every
printout downstream.*

---

## Section 2 · The corpus, and exactly how big it is

*Not examinable — engineering hygiene.*

## Cell 2 — download the corpus

**Prompt to type:**

> download the coco val2017 annotations from images.cocodataset.org — the file is annotations_trainval2017.zip — and pull instances_val2017.json out of the zip into datasets/coco without writing the rest of it. then take the 128 images with the lowest image ids and download just those jpegs from the val2017 bucket. skip anything already on disk. print how many images and categories val2017 has, and how many jpegs ended up on disk.

**Expect:**

```
5,000 images in val2017, 80 categories
128 images on disk
```

**Assert:** `assert len(images) == N_IMAGES`

**⏱** See the table above. The zip is 241 MiB and the JPEGs are 20.3 MB across
128 sequential requests. On a fresh Colab VM this runs in full every session.

**Annotate:** short

*Two things to insist on. The selection rule must be statable in one sentence —
"the 128 numerically lowest image ids" is, "a representative sample" is not — so
that nobody can be accused of choosing the images that make the detector look
good. And the zip must not be extracted whole: you want one member,
`annotations/instances_val2017.json`, and the rest of the archive is the other
four annotation sets.*

---

### 2.1 · Ground truth

*Examinable.*

Type this markdown:

> Two things to notice in the next cell, both of which cost people an afternoon
> the first time:
>
> 1. COCO stores a box as `[x, y, w, h]`; torchvision returns
>    `[x1, y1, x2, y2]`. Convert once, at the edge of the program.
> 2. `iscrowd = 1` means the annotator drew one region around many instances
>    rather than boxing them separately. Dropping those is a *choice*, it
>    changes every count below, and this is where it is recorded.

## Cell 3 — ground truth, converted at the edge

**Prompt to type:**

> build the ground truth for those 128 images: a dict from image id to boxes and labels as numpy arrays. coco bboxes are [x, y, w, h] and i want corners [x1, y1, x2, y2]. skip the iscrowd annotations but count how many you skipped. assert x2 >= x1 and y2 >= y1 on every box. print the total object count and the objects per image — mean, median, min, max.

**Expect:**

```
128 images, 898 objects, 8 crowd regions dropped
objects per image: mean 7.02  median 5  range 1-29
```

**Assert:**

```python
for iid, g in gt.items():
    assert (g["boxes"][:, 2] >= g["boxes"][:, 0]).all(), "x2 < x1: w read as x2"
    assert (g["boxes"][:, 3] >= g["boxes"][:, 1]).all(), "y2 < y1: same bug"
```

**Annotate:** full

* **Left open:** that dropping `iscrowd = 1` is a *choice*. One polygon drawn
  around many instances is not one object and it is not *n* objects — it is a
  refusal to decide, and COCO's own evaluator ignores those regions. Eight of
  them go here, and every count in the rest of the notebook is 8 short of the
  annotation file because of it. The prompt asked for the count; it did not ask
  for the decision to be defended.
* **The usual student version:** `boxes.append(a["bbox"])`, with no conversion.
  COCO's `bbox` field is `[x, y, w, h]` — that is the format spec, and it is
  true of every one of the 36,781 annotations in `instances_val2017.json`.
  torchvision's detection models return `boxes` as `[x1, y1, x2, y2]`, which is
  in the `FasterRCNN` docstring. Read one as the other and `x2` becomes the
  *width*, so a box that starts 300 px into the picture claims to end 60 px into
  it.
* **How you would catch it:** the assertion above, and nothing weaker. On these
  128 images, reading `[x, y, w, h]` as corners breaks `x2 >= x1` on **679 of
  the 898 boxes (75.6%), across 107 of the 128 images** — including 19 of the 20
  boxes on the very first image, id 139. The assert fires on the first
  iteration. Without it the boxes all collapse toward the top-left corner, every
  overlap you compute afterwards is near zero, and the number you get is
  believable.

---

### 2.2 · What is in it

*Examinable.*

## Cell 4 — the class distribution

**Prompt to type:**

> count how often each category name appears in the ground truth. print how many of the 80 categories show up at all, the eight commonest with their counts, and what share of all annotated objects is person.

**Expect:**

```
73 of 80 categories appear in these 128 images

  person          350
  cup              30
  chair            28
  wine glass       24
  car              24
  tie              20
  bottle           20
  backpack         19

person is 39.0% of every annotated object
```

**Assert:** none.

**Annotate:** short

*Ask for the **share**, not just the count. 350 is a number; 39.0% of 898 is the
fact you will need in the next lecture, when we start averaging over categories
and a mean over categories stops caring that one of them is two fifths of the
corpus. Note also that 7 of the 80 categories never appear at all — any
per-category mean computed on 128 images has empty categories in it, and what
you do about those changes the mean.*

---

## Section 3 · A metric, and the baseline that kills the obvious one

*Examinable. This is the section the lecture is built on.*

Type this markdown:

> The obvious metric is: *a detection is correct when its box overlaps the true
> box.* It is computable, unambiguous and parameter-free.
>
> Before adopting any metric, this course computes what the stupidest possible
> system scores under it. For detection, the stupidest possible system is
> **one box per image, covering the whole image**.

## Cell 5 — the baseline that kills the obvious metric

**Prompt to type:**

> i want to test a matching rule before i commit to it. the rule is: a detection counts as correct if its box overlaps the true box at all. score the dumbest possible detector under that rule — one box per image, covering the whole image — and print what fraction of the true objects it gets.

**Expect:**

```
the whole-image box overlaps 898 of 898 true objects = 100.0%
```

**Assert:** `assert hits == total, "if this ever fails, a box lies outside its own image"`

**Annotate:** full

* **Left open:** everything about *how much* two boxes overlap. The rule under
  test is the weakest possible one — any shared area at all — and the prompt
  never asks for a fraction, a threshold or an area ratio. That is deliberate:
  the point is that the weakest matcher is the one everybody reaches for first.
* **The usual student version:** `torchvision.ops.box_iou(pred, gt) > 0`. It is
  a real function, the arithmetic is correct, and it is exactly this rule. On
  these 128 images it scores the whole-image box **898 of 898 = 100.0%** — a
  system with no weights, no data and no idea, perfect. Nothing in the metric
  punishes a box for being enormous, so the metric is dead before it is adopted.
* **How you would catch it:** compute what the stupidest possible system scores
  *before* adopting any metric, not after. Here it is six lines and it rules out
  the obvious choice. The same assert does double duty: `hits == total` also
  proves no annotated box lies outside its own image, which is the other thing
  that would silently break every overlap computation below.

Type this markdown after the cell runs:

> **100%.** A system with no weights, no data and no idea scores perfectly under
> the proposed metric. That metric is dead: it rewards a box for being enormous,
> and nothing in it punishes size.
>
> So today's metric is the one thing left that the whole-image box loses at:
> **counting**.

## Cell 6 — the metric we can defend

**Prompt to type:**

> write count_mae(pred_counts, true_counts) returning the mean absolute error, and assert inside it that the two arrays have the same shape. then print the count MAE of two trivial predictors: always one object per image, and always the rounded corpus mean.

**Expect:**

```
one box per image           MAE 6.02
predict the corpus mean     MAE 4.88
perfect                     MAE 0.00
```

**Assert:** `assert pred_counts.shape == true_counts.shape`, inside `count_mae`.

**Annotate:** full

* **Left open:** what a *good* score is. 6.02 and 4.88 are the floor, not the
  target, and nothing in the prompt says where between 6.02 and 0.00 a useful
  system sits. That is the next section's job and it is yours, on paper.
* **The usual student version:** dropping the shape assert, because it looks like
  ceremony. NumPy will broadcast anything conformable and return a float that
  reads like a MAE. Two real ways to get one here: pass the scalar `n_true.mean()`
  instead of a length-128 vector and you get **4.88**, which happens to be the
  right answer to a different question; pass a column, `n_true[:, None]`, and
  numpy broadcasts `(128,)` against `(128, 1)` to shape **(128, 128)** and
  returns **6.43** — the mean pairwise difference between images, computed
  without a single error message.
* **How you would catch it:** the assert, which costs one line and turns both of
  those into a crash. This is also why the two trivial baselines are printed
  *now*, before any model exists: 6.02 is the number every result below has to be
  read against, and a system that never opens the image scores it.

---

## Section 4 · Commit

*Not examinable — but do it.*

Type this markdown. No code cell.

> **Stop. On paper, now.** Not in this notebook — on paper, where you cannot
> quietly revise it.
>
> ```
> Metric:                                              ____________
> Count MAE a useful shelf-audit system would need:    ____________
> Count MAE I expect from what we build today:         ____________
> ```
>
> You are not guessing in the dark: a system that never opens the image scores
> 6.02, and perfect is 0.00. Saying *where between them* is the exercise.

---

## Section 5 · The detector

*Beyond the book, for context — nothing here is trained.*

## Cell 7 — load the pretrained detector

**Prompt to type:**

> load torchvision's pretrained faster r-cnn resnet50 fpn with the COCO_V1 weights, put it in eval mode on DEVICE, and get its transforms and its category name list. print how many label slots it has, what slot 0 and slot 12 are, and whatever metrics the weights object ships with. check that the label integer the model emits is the same integer as coco's category_id.

**Expect:**

```
91 label slots for 80 categories
slot 0 is __background__ | slot 12 is N/A
```

then the weights' own reported score on all 5,000 val2017 images:
`{'COCO-val2017': {'box_map': 37.0}}`

**Assert:**

```python
assert names[1] == "person"
assert all(names[cid] == nm for cid, nm in cat_name.items())
```

**⏱** First run downloads `fasterrcnn_resnet50_fpn_coco-258fb6c6.pth`,
167,502,836 bytes. Cached afterwards on a local disk; re-downloaded every
session on Colab.

**Annotate:** short

*91 slots for 80 categories. COCO's ids are not contiguous — slot 12 is the
string `N/A` — and the gaps are the whole reason the second assert is worth
writing rather than assuming. It passes, and that is the only thing that makes
the label comparison further down legitimate. Do **not** let the assistant build
an index-order-to-COCO-id mapping by hand; it is unnecessary and it shifts every
category by a few positions.*

*One thing the printed `box_map` of 37.0 is **not**: comparable to anything below
it. That is mean average precision over 5,000 images. Everything from cell 10 on
is a count MAE over 128. They are different quantities on different corpora and
neither bounds the other.*

### 5.1 · Run it

## Cell 8 — run the detector over the 128 images

**Prompt to type:**

> run the detector over all 128 images. open each jpeg with PIL, convert to RGB, apply the preprocess, one image at a time inside torch.inference_mode, and move each output to cpu numpy before storing it in a dict keyed by image id. time the loop and print the seconds per image.

**Expect:**

```
128 images in 12.8 s (0.10 s per image on mps)
```

with your own two numbers. The dict has 128 entries, each `{"boxes", "labels",
"scores"}` as numpy arrays.

**Assert:** `assert len(preds) == N_IMAGES`

**⏱ Measured, on one machine (Apple M4, 16 threads):** **12.8 s** on the MPS
backend (0.10 s/image), **361.4 s — six minutes** with MPS disabled (2.82
s/image). That is a factor of **28**. A CPU-only Colab runtime is the case to
plan for: budget six minutes, not one. Timings also move with what else the
machine is doing, so treat your own first run as the estimate and not this
table. **No output does not mean it has hung** — this loop prints nothing until
it finishes.

*One consequence, measured rather than assumed:* the CPU and MPS runs do **not**
return byte-identical predictions. Over the 128 images the CPU run returned
4,420 boxes and the MPS run 4,419 — one box, on one image, that landed either
side of the 0.05 score cut. Every figure in this notebook is unaffected except
the naive count MAE in cell 10, which is **27.51** on MPS and **27.52** on CPU.
If your number ends in 2 rather than 1, nothing is wrong.

**Annotate:** short

*Two things to insist on in the prompt, both of which are about memory rather
than correctness. `inference_mode()` rather than `no_grad()` — it is strictly
stronger, it also skips version counting, and for a pure evaluation loop there
is no reason to use the weaker one. And `.cpu().numpy()` **inside** the loop:
keeping 128 sets of live GPU tensors alive is how the next cell fails for a
reason that has nothing to do with the next cell.*

### 5.2 · Read the shape before you read the answer

## Cell 9 — shapes and dtypes

**Prompt to type:**

> print the shape and dtype of every array the model returned for the first image, and check that boxes, labels and scores are the same length and that the scores come back sorted descending.

**Expect:**

```
boxes    (88, 4) float32
labels   (88,) int64
scores   (88,) float32

this image has 88 boxes and 20 annotated objects
```

**Assert:**

```python
assert p["boxes"].shape[0] == p["labels"].shape[0] == p["scores"].shape[0]
assert np.all(np.diff(p["scores"]) <= 0), "not sorted by score"
```

**Annotate:** short

*Assert the sort order rather than assuming it. Several detection APIs return
unsorted boxes, and code that slices "the top k" from an unsorted array silently
takes an arbitrary k. Write down the two numbers this cell prints before you go
on.*

---

## Section 6 · An assistant writes the counting code

*Examinable.*

Type this markdown — and nothing else. No warning, no ⚠, no hint:

> The stakeholder's question is *how many objects are in each picture*. Ask for
> it the way you would actually ask for it.

## Cell 10 — the assistant's counting code

**Prompt to type:**

> use a pretrained faster r-cnn from torchvision to count how many objects are in each image and print the mean absolute error against the coco annotations

**Expect:**

```
mean objects per image: 34.52
count MAE: 27.51
```

on the MPS backend; **34.53** and **27.52** on CPU. See the note under cell 8 —
this is the one figure in the notebook that moves with the device, by one box in
4,420.

**Assert:** none. Nothing raises. Nothing warns.

**Annotate:** short

*Write both numbers on the sheet from section 4, next to your prediction, before
you scroll. Then keep scrolling.*

---

Type this markdown **after** the cell has run and the numbers are written down:

> ### ⚠ Reviewer question 5: what is the default I did not ask for?
>
> Everything in that request was true and none of it was wrong. One constraint
> was missing, and the code that came back is one line:
>
> ```python
> counts_naive = np.array([len(preds[im["id"]]["boxes"]) for im in images])
> ```
>
> `len(...["boxes"])` is the number of rows the model **chose** to return. It is
> not a count of objects. It is a count of *candidates*.
>
> Reviewer question 3 — *what is the shape here?* — found it one cell earlier and
> nobody stopped: 88 boxes for a photograph with 20 annotated objects in it.
>
> **Now measure the damage. Do not estimate it.**

## Cell 11 — measure the damage

**Prompt to type:**

> summarise every score the model returned across the 128 images: how many boxes in total, how many below 0.10, how many at or above 0.50, and the median score. then print the largest number of boxes returned for any single image, and how many images returned exactly that number.

**Expect:**

```
boxes returned in total: 4,419
  below score 0.10: 1,224
  at or above 0.50: 1,206
  median score:     0.205

most boxes returned for one image: 100 (10 images returned exactly 100)
```

**Assert:** none, but see the third bullet — the round number *is* the check.

**Annotate:** full

* **Left open:** cell 10's prompt never said *which* detections to count, so the
  assistant answered a different question correctly. Half of the 4,419 boxes
  score below 0.205; 1,224 of them the model believes in at less than 0.10. The
  request contained no threshold, so all of them were counted, and the output
  contained no threshold either, so nothing in the printed result records that a
  decision was made.
* **The usual student version:** exactly `len(pred["boxes"])`. Two torchvision
  defaults decided that length and neither was mentioned in the prompt: on the
  constructed model, `model.roi_heads.score_thresh` is **0.05** and
  `model.roi_heads.detections_per_img` is **100** — the `box_score_thresh` and
  `box_detections_per_img` arguments of
  `torchvision.models.detection.FasterRCNN`. The result on these 128 images:
  34.52 boxes per image against a true mean of 7.02, and a count MAE of
  **27.51** against a baseline of **6.02** that never opens the image. Nothing
  raised. Nothing warned. The number just looked plausible.
* **How you would catch it:** whenever a library hands you a variable-length
  result, ask what decided the length. Here the answer is printed in the cell
  above — **10 of the 128 images returned exactly 100 boxes**. A per-image count
  that lands on a round number for eight percent of the corpus is a cap, not a
  property of the photographs, and it takes one lookup to confirm which cap.

---

### The corrected specification

Type this markdown:

> Three additions: name the parameter, demand the baseline, assert the
> relationship. The third is the one that fails loudly.

## Cell 12 — the corrected version

**Prompt to type:**

> write count_objects(pred, thresh) that counts detections scoring at least thresh, with no default for thresh — i want to have to say it every time. assert inside it that the kept count never exceeds the returned count. run it at 0.5 and print the count MAE, the signed mean error, and the one-box-per-image baseline next to it. assert the MAE beats the baseline.

**Expect:**

```
count MAE  3.00
signed     +2.41   (positive = too many boxes)
baseline   6.02
```

**Assert:**

```python
assert keep.sum() <= len(pred["scores"])
assert mae < one_box, "worse than predicting one box per image"
```

**Annotate:** full

* **Left open:** that `THRESH = 0.5` was fixed *before* anyone looked at an error
  curve. The prompt does not say so and the code cannot record it; only the
  comment can, and only if you write it. Section 7 is about what happens when it
  is not true.
* **The usual student version:** giving `thresh` a default of 0.5, because it is
  more convenient to call. torchvision already made that decision for you once —
  `box_score_thresh` has a default of 0.05 — and that default is precisely what
  produced 27.51 two cells ago. A silent default in your own function is the
  same bug in your own code, one layer up, and the next person to call it will
  not know a threshold was applied either.
* **How you would catch it:** print the **signed** error beside the absolute one.
  MAE is 3.00 and the signed bias is +2.41, so four fifths of the error is
  over-counting; a balanced system with the same MAE of 3.00 would be a
  completely different system and the absolute figure alone cannot tell them
  apart. Note also what the second assert is worth: it passes here, 3.00 against
  6.02, but on the naive counts it would have failed at 27.51 — the assert cell
  10 did not have.

Type this markdown after it runs:

> The naive version was **27.51**, this one is **3.00**, and the baseline that
> never opens the image is **6.02**. One missing constraint took the answer from
> "half the error of a system that ignores the picture" (3.00 ÷ 6.02 = 0.50) to
> "four and a half times worse than one" (27.51 ÷ 6.02 = 4.57).
>
> Nothing raised. Nothing warned. The number just looked plausible.

---

## Section 7 · The threshold is a knob, and nobody chose it

*Examinable.*

## Cell 13 — sweep the threshold

**Prompt to type:**

> sweep the threshold from 0.05 to 0.95 in steps of 0.05. for each one print the mean detections per image and the count MAE, and mark the row we report. then print which threshold gives the lowest MAE.

**Expect:** nineteen rows. The ends and the interesting middle:

| thresh | mean/img | MAE |
|---|---|---|
| 0.05 | 34.52 | 27.51 |
| 0.50 | 9.42 | 3.00 ← we report this |
| 0.75 | 6.16 | **1.92** |
| 0.95 | 3.41 | 3.70 |

and `lowest MAE is 1.92 at threshold 0.75`.

**Assert:** none.

**Annotate:** full

* **Left open:** why we then refuse to report 1.92. The prompt asked for the
  minimum and the assistant will hand it over; the refusal is a judgement the
  code cannot make and you have to type it as prose underneath.
* **The usual student version:** reporting the minimum. It is lower, it is
  honestly computed, and it is **36% below the figure we report** (1.92 against
  3.00). It is also selection on the evaluation set: the threshold 0.75 was
  chosen by looking at the MAE on the same 128 images the 1.92 is then quoted
  for. That is the failure from application 3 — choosing a hyperparameter on the
  test set — wearing a detection costume.
* **How you would catch it:** read the first column, not the third. The mean
  detections per image runs from **34.52 at threshold 0.05 to 3.41 at 0.95, a
  factor of 10.1**, and the stakeholder asked for exactly that number. A count
  reported without the threshold that produced it is not an answer to their
  question; it is one of nineteen answers, picked silently.

Type this markdown after it runs:

> We do **not** report 1.92. It was found on the same 128 images we then report
> on, and 3.00 at a threshold fixed in advance is the honest number even though
> it is worse.

---

## Section 8 · Look at the pictures, not only at the number

*Examinable.*

## Cell 14 — draw the detections

**Prompt to type:**

> show the first three images side by side with the detections at threshold 0.5 drawn as rectangles. only draw the text label when the score is above 0.9, otherwise the captions pile up on each other. title each panel with how many boxes were drawn and how many objects are annotated.

**Expect:** three panels, titled `25 boxes, 20 true`, `1 boxes, 1 true`,
`31 boxes, 17 true`. Panel 2 is a single bear filling the frame; panels 1 and 3
are a kitchen and a bedroom, both dense.

**Assert:** none. Look at it instead.

**Annotate:** full

* **Left open:** what to *do* with the figure. It is evidence, not decoration,
  and the label-suppression rule is what makes it evidence: at threshold 0.5,
  panel 3 draws 31 rectangles, and drawing 31 captions on a bedroom produces an
  illegible figure from which a reader correctly concludes that the visual check
  is not worth doing.
* **The usual student version:** assuming non-maximum suppression already removed
  the duplicates, so two boxes on one object must mean two objects.
  `box_nms_thresh` defaults to **0.5** and torchvision's NMS runs **per class**,
  so any pair below that IoU survives. Measured on these images: two `vase`
  boxes on panel 1 at **IoU 0.494**, two `book` boxes on panel 3 at **IoU
  0.487**. Both are one object, both are drawn, and both are counted.
* **How you would catch it:** try to write down a number for "how wrong is that
  box". You cannot, and neither can the metric you committed to in section 4.
  Concretely, on panel 1 there are **8 drawn boxes that touch no annotated
  object at all** (IoU below 0.3) — a `refrigerator` at 0.88, a `dining table`
  at 0.95 — and on panel 3 there are **22**. There is also a `dining table` box
  on panel 1 that is **1.19× the area** of the object it found and a `book` box
  on panel 3 at **1.56×**. Counting scores every one of those as correct
  whenever the totals happen to agree.

Type this markdown after it runs:

> Some of those boxes are visibly wrong: two on one object, one a little too
> large, one confident about nothing at all.
>
> **You have no way to say how wrong.**

## Cell 15 — two systems the metric cannot tell apart

**Prompt to type:**

> take the ground-truth boxes of the first image and keep the first nine. make system A those nine boxes exactly, and system B the same nine shifted 400 pixels right and down. print the count error of each against nine.

**Expect:**

```
system A: 9 boxes, count error 0
system B: 9 boxes, count error 0
```

**Assert:** none — the equality of the two lines *is* the result.

**Annotate:** short

*Insist on constructing the counterexample rather than describing it. "Counting
is a weak metric" is a remark somebody can argue with; two arrays and three
prints are not. Image 139 has 20 annotated objects so the slice takes 9 of them,
and both systems emit 9 boxes for an image with 9 objects. The committed metric
scores both perfect, which means no amount of tuning it can ever separate them.*

---

## Section 9 · Propose the missing number

*Examinable, and it is the whole of next lecture.*

Type this markdown:

> Whatever repairs this has to:
>
> 1. be 1 for identical boxes and 0 for boxes that do not touch
> 2. punish a box for being **too large**, or the whole-image baseline of section
>    3 wins again
> 3. punish a box for being **too small**, or a one-pixel box in the right place
>    wins
> 4. be dimensionless, so a 40-pixel cup and a 400-pixel sofa are on one scale

## Cell 16 — the stub you fill in

**Prompt to type:**

> give me a stub function my_box_score(a, b) taking two corner-form boxes, with a docstring listing those four requirements and a body that raises NotImplementedError. then call it on an identical pair and on a disjoint pair inside a try/except that prints a message if it is not written yet.

**Expect:**

```
not written yet — this is yours to write
```

**Assert:** none — `NotImplementedError` is the expected outcome, caught.

**Annotate:** short

*This is the one cell that is supposed to fail. Two test boxes, `[0, 0, 100,
100]` twice for the identical pair and `[300, 0, 400, 100]` for the disjoint
one. If your formula does something odd for the **disjoint** pair, do not fix
it — that is the interesting case, and it is most of next lecture.*

---

## Section 10 · Where we are

*Not examinable.*

Type this markdown:

> | System | Count MAE, 128 images |
> |---|---|
> | One box per image | 6.02 |
> | Every box the model returns | 27.51 |
> | Faster R-CNN at score ≥ 0.5 | **3.00** |
>
> All three rows are scored on the same 128 images and the same 898 objects.
>
> Write **3.00** next to what you predicted in section 4, and keep the sheet.
>
> Do not fix anything. Counting cannot distinguish nine right boxes from nine
> wrong ones, and the repair is the next ninety minutes.

---

## Exercises, with the cells to re-run

Each of these says which cells, in which order. **Cell 8 is the expensive one
(12.8 s on MPS, 361.4 s on CPU) and none of these exercises requires re-running
it** — `preds` does not depend on any threshold, so every exercise below is
seconds of compute on any machine.

1. **Report at 0.75 instead of 0.5.** Change `THRESH` in cell 12, then re-run
   **12 → 13 → 14**, in that order. Cell 14 draws at `THRESH`, so the panel
   titles change too. Expected: MAE **1.92**, and the three panels go from
   25 / 1 / 31 boxes to **18 / 1 / 9**. Then write one paragraph on why the
   notebook does not report 1.92.
2. **Move the label cut-off.** Change `label_above` from 0.90 in cell 14 and
   re-run **14 only**. Find the value at which panel 3 becomes unreadable. That
   value is your evidence for the suppression rule, not the assertion that
   captions pile up.
3. **Break the box convention on purpose.** In cell 3, change
   `[x, y, x + w, y + h]` to `[x, y, w, h]` and re-run **3 → 4 → 5 → 6**.
   Predict which assert fires first, then check: it is `x2 >= x1` in cell 3, on
   image 139, and 679 of the 898 boxes violate it. Cells 4–6 never run. Change
   it back before going on.
4. **Count the crowd regions back in.** In cell 3, stop skipping `iscrowd`, and
   re-run **3 → 6 → 12**. Cell 8 does not need re-running: only the ground truth
   changed. Eight objects return, `n_true.sum()` goes 898 → **906**, the mean
   7.02 → **7.08**, the baseline 6.02 → **6.08** and the reported MAE 3.00 →
   **2.97**. Note the direction: the notebook's choice to drop the crowd regions
   makes its own system look slightly *worse*, which is the direction a
   discretionary choice should point. Say in one sentence which of the two
   numbers you would put in a report.
5. **Score the whole-image baseline under counting instead of overlap.** New
   cell after 6: one box per image is `count_mae(np.ones(N_IMAGES), n_true)` and
   you already have it — 6.02. Now do the same for the *whole-image* box, which
   is also one box. They are the same number. Write down why the metric that
   killed the baseline in section 3 and the metric that kills it in section 6
   are two different metrics.
6. **Restart and run all.** From a cold kernel, top to bottom. On a local disk
   the two downloads are already there and the whole notebook is under a minute
   plus cell 8; on a fresh Colab VM it is the full 241 MiB plus 167 MB plus the
   128 JPEGs again. Nothing in this notebook is order-dependent except that cell
   8 must precede 9–15, and cell 12 defines `count_objects`, which 13 uses.

---

## Defects found in the current notebook

**A note on numbering.** The script above numbers *code cells* 1–16. This
section numbers *notebook cells* as `nbformat` indexes them, 0–51, so that every
claim can be checked with `json.load(...)["cells"][n]`. The defective cell is
code cell 10 in the script and cell **31** in the `.ipynb`.

All of the following were checked against `notebooks/lecture-17.ipynb`,
`tools/notebooks/lecture_17.py`, the annotation file at
`notebooks/datasets/coco/instances_val2017.json`, and a real inference run over
the 128 images. Where I could not check something, I say so.

### Verified with `python3`

**1. Not one code cell has a stored output. §1.2, §9.**
`sum(len(c["outputs"]) for c in cells)` is **0**, and every `execution_count` is
`None`, across all 16 code cells. Every figure quoted in the prose — 27.51,
3.00, 6.02, 39%, 100%, "88 boxes", "twenty annotated objects" — therefore
appears in no stored output, which is precisely what §1.2 forbids and what the
§9 checker tests. It also means the §7.1 machine check ("any cell whose stored
execution exceeded 20 s must have a ⏱ marker") cannot run at all. I re-derived
every one of those figures independently and **they are all correct** — 898
objects, 8 crowd regions, mean 7.02, median 5, range 1–29, person 39.0%, 898/898
= 100.0%, one-box MAE 6.02, image 139 returning `(88, 4)` boxes against 20
annotated objects, naive MAE 27.51, corrected MAE 3.00 — so this is a
provenance defect, not an arithmetic one. The reader cannot tell the difference,
which is the point of the rule.

**2. The trap is announced five times before the cell, and the fifth
announcement gives away the answer. §8.1, §8.2.** Counting forward to code cell
31: the header (cell 0) says a marked cell "prints a believable number, and the
number is wrong by a factor of nine"; cell 29 says "**⚠ Read before running** …
One constraint is missing. Find it before you scroll"; the prompt box label (cell
30) carries a ⚠; the same box's **Left open** bullet then prints the complete
answer — "`len(pred['boxes'])` … everything above `box_score_thresh`, which
defaults to 0.05, capped at `box_detections_per_img`, which defaults to 100" —
and its **The usual student version** bullet adds "wrong by a factor of nine".
All of that sits **above** the code cell. §8.1's limit is four; this is five, and
the last two are on the same screen as the instruction to find it unaided. My
script moves the entire disclosure to cell 11, after the number is written down.

**3. Every one of the 16 prompt boxes carries the full three-bullet annotation.
§6.1.** `"**Watch this prompt.**"` occurs **16 times** in 16 prompt boxes, and
`"> **Prompt"` occurs 16 times. The budget is five to eight, never more than ten.
Worse, the rendered notebook's header (a paragraph the build script adds, present
in the `.ipynb` but not in `lecture_17.py`) instructs the reader that those three
lines "are the part worth reading twice" — so the notebook explicitly asks for
the behaviour §6.1 measured readers abandoning around cell 30. Cell 30 here is
the defect cell.

**4. Four names are rebound to different types across cells. §4.1.** Parsed with
`ast` over the 16 code cells:

| Name | Bindings |
|---|---|
| `c` | cell 9: a category `dict` (`{c["id"]: c["name"] for c in raw["categories"]}`) · cell 12: an `np.int64` label · cell 41: an `ndarray` of counts |
| `p` | cell 6: a `pathlib.Path` (`p = IMG_DIR / im["file_name"]`) · cells 28 and 44: a prediction `dict` |
| `k` | cells 25, 28: a `str` dict key · cells 12, 44, 47: an `int` |
| `i` | cell 9: an `int` image id (`{i: … for i in ids}`) · cell 44: an image `dict` (`next(i for i in images …)`) |

`keep` is a third case worth naming: cell 37 binds it to a **boolean mask**
(`pred["scores"] >= thresh`) and cell 44 to an **integer index array**
(`np.flatnonzero(…)`). Same dtype family, opposite indexing semantics, one
name — which is exactly the confusion that silently selects the wrong rows.

**5. The timing claim for cell 25 does not reproduce, and there is no CPU
figure. §7.1.** Cell 23 states "⏱ **1 to 2 minutes** on a GPU or an Apple
Silicon MPS backend … the same loop took 39 s on an idle laptop and 102 s on a
busy one." I ran the identical loop on an Apple M4 with the MPS backend: **12.8
s for 128 images, 0.10 s per image** — three times faster than the fastest
figure quoted and a fifth of the stated lower bound. The CPU number, which §7.1
requires explicitly, is given only as "several minutes on a CPU-only runtime",
which is not a figure. Measured on the same machine with MPS disabled, 16
threads: **361.4 s, 2.82 s per image** — six minutes, and **28×** the MPS run.
So the reader is given a range that is wrong at both ends for the accelerated
case, and for the CPU case — the one the guidelines' literal reader is actually
in — no number at all. "Several minutes" and "six minutes" are not the same
planning decision.

**5b. The naive count MAE is device-dependent, and the prose hard-codes one
device's value. §1.2, §1.5.** I ran the full pipeline twice on the same machine,
once on MPS and once on CPU. Everything reconciles except cell 31: the CPU run
returns **4,420** boxes across the 128 images and the MPS run **4,419** — one
box, on one image, landing either side of the 0.05 score threshold. That makes
the headline naive figure **27.51 on MPS and 27.52 on CPU** (and the mean 34.52
against 34.53). Cell 38's prose and the §10 summary table both state 27.51 as
flat fact. A CPU-only Colab runtime — the default, and the one the notebook's
own header implies is fine — prints 27.52, and nothing in the notebook tells the
reader that the mismatch is a float-ordering artefact rather than their mistake.
Everything else is stable across both devices: 3.00, +2.41, 6.02, median 0.205,
88 boxes on image 139, sweep minimum 1.92 at 0.75.

**6. The download cell states a duration for something that is a bandwidth
division, and "instant afterwards" is false on Colab. §7.1, §7 (the literal
reader).** Cell 6 says "⏱ about 60-90 seconds the first time, instant
afterwards". I checked the archive with an HTTP `HEAD` against
`images.cocodataset.org`: **252,907,541 bytes**, i.e. 241.2 MiB — so the
notebook's "~241 MB" is MiB wearing an MB label, which is the conventional
reading and I am not counting it as an error. The 60–90 s figure implies a
sustained 2.8–4.2 MB/s (22–34 Mbit/s), which is a plausible link but is not
stated as an assumption; the same cell takes ~20 s at 100 Mbit/s and ~100 s at
20 Mbit/s, and a reader on a slow link has no way to tell whether their run has
hung. The budget also omits the second download entirely: 128 JPEGs,
20,325,087 bytes, fetched one `urlretrieve` at a time, so 128 sequential round
trips are unaccounted for.

The harder half is "instant afterwards", which holds only for a persistent
disk. `DATA = Path("datasets/coco")` is relative to the working directory, and
Colab discards the VM filesystem between sessions — so the student these
guidelines are written for pays the full 241 MiB, plus the 167 MB of weights in
cell 22, **every time they open the notebook**. Neither the cell nor the header
says so, and "instant afterwards" is the sentence that would stop them from
budgeting for it.

**7. "Wrong by a factor of nine" is a ratio of MAEs, presented as a property of
the printed number. §1.4, §1.5.** The header and the cell-30 annotation both
say the defective cell "prints a believable number, and the number is wrong by a
factor of nine". The cell prints two numbers. `mean objects per image: 34.52`
against a true mean of 7.02 is a factor of **4.92**; `count MAE: 27.51` against
the corrected 3.00 is a factor of **9.17**. Only the second is nine, and it is a
ratio of *errors*, not of the quantity the stakeholder asked about. §1.4 exists
for exactly this — name the operation.

**8. "Four times worse than one" is 4.57. §1.2.** Cell 38: "One missing line
took the answer from 'half the error of a system that ignores the picture' to
'four times worse than one'." 3.00 ÷ 6.02 = 0.498, so "half" is right. 27.51 ÷
6.02 = **4.57**, which rounds to four and a half, not four. Small, but §1.2
requires conventional round-half-up and this rounds the wrong way, in the
direction that understates the defect the section exists to dramatise.

**9. `len(pred["boxes"])` appears in no code cell of this notebook. §3.1, §3.2.**
Cell 32's prose is built around that expression. String search over all 16 code
cells: `len(pred["boxes"])` — **absent**. What cell 31 actually contains is
`len(preds[im["id"]]["boxes"])`. The name `pred` (singular) is not bound anywhere
in the notebook until cell 37 defines `count_objects(pred, thresh)`, five cells
*after* the prose invites the reader to think about `pred["boxes"]`. A reader who
searches for the quoted expression, as §3.1's evidence says two of three readers
did in lecture 19, finds nothing. Likewise `box_score_thresh` is named in prose
three times and appears in **no** code cell — it is a constructor default that
the notebook never prints, so the reader is asked to take 0.05 on trust. It is
checkable in one line, `model.roi_heads.score_thresh`, and I checked it: **0.05**,
with `detections_per_img` **100** and `nms_thresh` **0.5**.

**10. "Reviewer question 3" and "reviewer question 5" are used five cells before
they are named. §3.3, §7.5.** First occurrence is cell 27's **Left open** bullet
("reviewer question 3 answering reviewer question 5 before it is asked"); cell
30's bullet uses "reviewer question 5" again; only cell 32 finally states what
they are. The five questions are defined in `lecture_03.py` §13, so a reader who
has lecture 3 can resolve them and a reader who does not cannot — and neither
can tell which situation they are in from anything in this notebook.

**11. The weights' `box_map` of 37.0 is offered for comparison against numbers it
cannot be compared with. §2.1.** Cell 22 prints torchvision's reported metric on
all 5,000 val2017 images, and the cell-21 annotation instructs: "Your 128-image
number should be read next to it, not instead of it." The 128-image numbers are
count MAEs (27.51, 3.00, 6.02); 37.0 is mean average precision. Different
quantity, different corpus, no arithmetic relation. §2.1 asks that a comparison
hold the scoring window constant and say which rows; this one holds neither the
window nor the quantity, and the prose does not warn the reader that "next to"
cannot mean "against".

**12. The word "examinable" appears once in the notebook, and it is in a code
comment. §8.3.** Search of all markdown cells: **0** occurrences. Search of code
cells: **1**, the line `# Not examinable: this is engineering hygiene, not
machine learning.` inside cell 3. Sections 2 through 10 carry no marking at all.
§8.3 requires every section to carry one of three labels, and cites lecture 19's
single occurrence as the defect.

**13. `n_crowd` is computed and printed but its consequence is never quantified.
§1.3.** The notebook drops 8 crowd regions and calls the drop "a *choice* … it
changes every count below". It does not say by how much. Verified: 898 objects
kept, 906 with crowds included, so the corpus mean moves from 7.02 to 7.08 and
the one-box baseline from 6.02 to 6.08. A statistic that depends on an arbitrary
choice must either justify the choice in the sentence or report the version that
does not depend on it (§1.3); this does neither, and the numbers are small enough
that saying them would have cost nothing. Measured with the crowds included:
`n_true.sum()` 898 → **906**, mean 7.02 → **7.08**, baseline 6.02 → **6.08**,
reported MAE 3.00 → **2.97**. The choice moves the headline by 0.03 and moves it
in the conservative direction — which is a good defence of the choice, and the
notebook does not make it.

### Checked and found clean

I ran these and they pass, so they are **not** defects — recorded because the
brief asks which checks were made:

* **§5.1 / §5.2 markdown rendering.** No markdown line outside a fence is
  indented ≥ 4 spaces; no fence marker is indented at all. The lecture-19 cell-41
  failure has no analogue here.
* **§3.1 fenced code blocks.** There are **no** ```` ```python ```` fences in any
  markdown cell, so there is nothing to fail the verbatim check. (Defect 9 above
  is about *inline* code, which the §9 checker does not cover.)
* **§4.2 idempotence.** Nothing trains. Cell 25 rebuilds `preds` from an empty
  dict, cell 6 guards both downloads on `is_file()`, and re-running any cell
  top-to-bottom gives the same numbers. Restart-and-run-all has no state hazard
  beyond the obvious ordering.
* **Every headline figure.** 5,000 images / 80 categories / 36,781 annotations in
  the file; 898 objects; 8 crowd regions; mean 7.02, median 5, range 1–29; 73 of
  80 categories present; person 350 = 39.0%; whole-image box 898/898 = 100.0%;
  baselines 6.02 and 4.88; 91 label slots, slot 0 `__background__`, slot 12
  `N/A`; image 139 → `(88, 4)` boxes, 20 annotated objects, scores sorted
  descending; naive mean 34.52 and MAE 27.51; 4,419 boxes, 1,224 below 0.10,
  1,206 at or above 0.50, median 0.205; max 100 boxes; corrected MAE 3.00, bias
  +2.41; sweep minimum 1.92 at 0.75. **All correct.**
* **Both asserts in cell 22.** `names[1] == "person"` and
  `all(names[cid] == nm for cid, nm in cat_name.items())` both pass — the second
  is the non-obvious one and it holds.
* **The three visual claims under the figure** ("two on one object, one a little
  too large, one confident about nothing at all", §7.3). All three are findable,
  which I did not expect: two `vase` boxes at IoU 0.494 and two `book` boxes at
  IoU 0.487 (both under the 0.5 NMS default); a `dining table` box at 1.19× the
  true area and a `book` box at 1.56×; and 8 boxes on panel 1 plus 22 on panel 3
  that touch no annotated object at IoU ≥ 0.3. One caveat: all six of those
  boxes score below `label_above = 0.90`, so they are drawn **unlabelled**, and a
  reader cannot name what they are looking at from the figure alone.

### Not checked

* **Whether the figures hold on CUDA.** I compared MPS against CPU (defect 5b)
  but have no NVIDIA GPU here, so I cannot say whether a Colab T4 reproduces
  27.51, 27.52, or a third value. Given that MPS and CPU already disagree by one
  box, I would expect a third value and I have not verified it.
* **Download durations.** I could not time the 241 MiB download or the 167 MB
  weights download: both were already on disk. The byte counts are exact (HTTP
  `HEAD` and `stat` respectively); the durations in defect 6 are arithmetic from
  those byte counts, not measurements.
* **How the notebook renders in Colab specifically.** I checked the markdown
  source mechanically for §5.1/§5.2 and read it, but did not open it in Colab.
  §10.8 asks for the rendered page and that requires a browser.
* **Whether "the annotation file is the larger of them" (cell 6 comment) is what
  the author meant.** As *downloads* it is true — 241 MiB against 20.3 MB of
  JPEGs. On disk after extraction it is false by a hair: `instances_val2017.json`
  is 19,987,840 bytes and the 128 JPEGs total 20,325,087. I have left it out of
  the defect list because the sentence says "downloads".
