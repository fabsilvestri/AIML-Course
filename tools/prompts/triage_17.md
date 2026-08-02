# Triage — lecture 17

Claims triaged: the `Defects found in the current notebook` section of
`tools/prompts/lecture_17.md` (lines 823–1002). That section contains
**14** numbered claims, not 10: `1, 2, 3, 4, 5, 5b, 6, 7, 8, 9, 10, 11, 12, 13`.
All 14 are triaged below, in source order.

**Standing note (§1.2, applies to every claim below).** `notebooks/lecture-17.ipynb`
stores **no** cell outputs: 52 cells, 16 code cells, `sum(len(c["outputs"]))` = **0**,
`{c["execution_count"]}` = `{None}`. So no prose figure in this notebook can be
reconciled against a stored output. This is stated once here and not repeated
per claim.

**What I did not run.** Per instruction I did not execute the detector. Every
figure that requires inference over the 128 images — 4,419/4,420 boxes, mean
34.52, naive MAE 27.51, corrected MAE 3.00, bias +2.41, the score distribution,
the sweep, the wall-clock — is therefore not re-derived here, and I say so where
it matters. Everything derivable from `instances_val2017.json` (cached, 19,987,840
bytes), from torchvision's static metadata, or from the notebook text **is**
re-derived.

---

### Claim 1 — no code cell stores an output; every quoted figure is unreconcilable, and the §7.1 machine check cannot run

**Verdict:** CONFIRMED

**Evidence:**
```
$ python3 -c "import json; nb=json.load(open('notebooks/lecture-17.ipynb')); \
  co=[c for c in nb['cells'] if c['cell_type']=='code']; \
  print(len(nb['cells']),'cells',len(co),'code',sum(len(c['outputs']) for c in co),'outputs', \
        set(c['execution_count'] for c in co))"
52 cells 16 code 0 outputs {None}

$ python3 tools/check_notebooks.py --advisory   # lecture-17 section
note  lecture-17.ipynb  (28 advisory)
        cell 0: 5,000 not in any output — ...COCO's `val2017` split is 5,000 images...
        cell 17: 6.02 not in any output — ...a system that never opens the image scores 6.02...
        ... and 20 more
```
The §7.1 check keys off *stored* execution time; with `execution_count = None`
everywhere it has nothing to test.

The claim's second half — "I re-derived every one of those figures and they are
all correct" — I independently re-derived the annotation-only half of it from
`notebooks/datasets/coco/instances_val2017.json` and **every one matches**:

```
5000 images 80 cats 36781 anns
objects 898 crowd 8
mean 7.02 median 5 range 1-29
73 of 80 categories
person 350 = 39.0%
one_box MAE 6.02   mean_box MAE 4.88
whole-image box overlaps 898 of 898 true objects = 100.0%
image 139 (lowest id): 20 annotated objects
```
Torchvision statics also match: 91 label slots, `names[0]='__background__'`,
`names[1]='person'`, `names[12]='N/A'`, `_metrics = {'COCO-val2017': {'box_map': 37.0}}`,
and both cell-22 asserts pass (`names[1]=="person"` → True; `all(names[cid]==nm …)` → True).
The inference-dependent figures (27.51, 34.52, 3.00, 4,419, 1.92@0.75) I did not run.

**Severity:** misleads a student
**Origin:** notebook structure
**Fix:** execute the notebook once and commit the outputs, or drop every prose figure that no output backs.

---

### Claim 2 — the trap is announced five times before code cell 31, and the fifth announcement gives the answer

**Verdict:** CONFIRMED

**Evidence:** the five disclosures, all above code cell 31, quoted from the `.ipynb`:

1. cell 0 (header): *"The cell marked **⚠ read before running** contains a defect on purpose … it prints a believable number, and the number is wrong by a factor of nine."*
2. cell 29: *"**⚠ Read before running.** … One constraint is missing. Find it before you scroll."*
3. cell 30, box label: `> **Prompt · ⚠ what the assistant returns**`
4. cell 30, Left open: *"`len(pred['boxes'])` … everything above `box_score_thresh`, which defaults to 0.05, capped at `box_detections_per_img`, which defaults to 100."* — the complete answer
5. cell 30, usual student version: *"It runs, it prints a plausible number, and the number is wrong by a factor of nine."*

Items 3–5 are the same markdown cell, i.e. on the same screen as item 2's
instruction to find it unaided. §8.1's evidence for lecture 19 is four
announcements; this is five, and two of them state the answer outright.

One correction to the claim's own wording: §8.1 does not set "a limit of four" —
its title is *"Do not announce the trap four times before it"*, so four is
already the failure, not the budget. Verdict unaffected.

**Severity:** misleads a student
**Origin:** hand-written prose
**Fix:** move disclosure 1 and the two cell-30 bullets to a markdown cell *after* code cell 31.

---

### Claim 3 — all 16 prompt boxes carry the full three-bullet annotation (§6.1 budget is 5–8, max 10)

**Verdict:** CONFIRMED

**Evidence:**
```
'**Watch this prompt.**'  markdown: 16   code: 0
'> **Prompt'              markdown: 16   code: 0

$ python3 tools/check_notebooks.py --advisory
FAIL  lecture-17.ipynb
        16 full annotations, budget is 10 (§6.1)
```
The course's own checker fails the notebook on this rule. The aggravating
detail is also confirmed: cell 0 contains *"Those three lines are the part worth
reading twice"*, and that paragraph is **not** in `tools/notebooks/lecture_17.py`
— `grep -n "worth reading twice"` finds it only at `tools/make_notebooks.py:85`,
so the build script injects it into all notebooks.

**Severity:** misleads a student
**Origin:** notebook structure
**Fix:** reduce to 5–8 full annotations; short `input · output · constraint · check` boxes for the rest.

---

### Claim 4 — four names rebound to different types across cells, plus `keep`

**Verdict:** CONFIRMED (with two rows of the table wrong on scope)

**Evidence:** `ast` walk over the 16 code cells, separating bindings that leak
into cell scope (`for` targets, `Assign`) from comprehension/genexp-local ones,
which in Python 3 never coexist with anything:

```
c      leaky: [12, 41]        comprehension-local: [9]
p      leaky: [6, 28, 44]     comprehension-local: []
k      leaky: [12, 28, 44, 47] comprehension-local: [25]
i      leaky: []             comprehension-local: [9, 44]
keep   leaky: [37, 44]        comprehension-local: []
```

Real cross-cell rebindings of differing type, all at cell scope:
- `p` — cell 6 `p = IMG_DIR / im["file_name"]` (`pathlib.Path`) → cell 28 / cell 44 `p = preds[…]` (`dict`). **Real.**
- `c` — cell 12 `for c in g["labels"]` (`np.int64`) → cell 41 `c = np.array([...])` (`ndarray`). **Real.**
- `k` — cell 28 `for k, v in p.items()` (`str`) → cells 12/44/47 (`int`). **Real.**
- `keep` — cell 37 `keep = pred["scores"] >= thresh` (**boolean mask**) → cell 44 `keep = np.flatnonzero(…)` (**integer index array**). **Real, and the worst of them**: swapping the two silently selects different rows without raising.

Two rows of the claim's table are wrong: the `i` row cites cell 9
(`{i: … for i in ids}`) and cell 44 (`next(i for i in images …)`), **both** of
which are comprehension/genexp-scoped and never bind at cell scope — `i` is not
rebound anywhere in this notebook. Likewise the `c`-at-cell-9 and `k`-at-cell-25
entries are comprehension-local. The claim is right about the defect and
imprecise about the mechanism.

**Severity:** wrong but harmless for `c`/`k`/`i`; **misleads a student** for `keep`
**Origin:** generated code
**Fix:** rename cell 44's `keep` → `idx`, and cell 6's `p` → `jpg`.

---

### Claim 5 — cell 23's timing does not reproduce, and there is no CPU figure

**Verdict:** CONFIRMED for the §7.1 violation; the two measurements are UNVERIFIABLE here

**Evidence:** cell 23 verbatim —
```
⏱ **1 to 2 minutes** on a GPU or an Apple Silicon MPS backend, and several
minutes on a CPU-only runtime. It varies with what else the machine is doing:
the same loop took 39 s on an idle laptop and 102 s on a busy one.
```
Two things settle without running anything:

1. §7.1 requires the CPU number. The notebook's CPU entry is *"several minutes"*.
   That is not a figure. **Confirmed.**
2. The paragraph contradicts itself: it states a floor of **1 to 2 minutes**
   and then, two lines later, **39 s** for *"the same loop"*. 39 s is below its
   own stated lower bound. This is checkable from the text alone and the Phase A
   report does not mention it.

The report's own measurements (12.8 s on M4/MPS, 361.4 s on 16-thread CPU) I did
not reproduce — running them means running the detector, which I was instructed
not to do. Those two numbers are **untested**.

**Severity:** misleads a student (the literal reader with no GPU gets no planning number)
**Origin:** hand-written prose
**Fix:** replace with a measured GPU/MPS figure and a measured CPU figure, and delete the self-contradicting 39 s / 102 s sentence.

---

### Claim 5b — the naive MAE is device-dependent (27.51 MPS vs 27.52 CPU) and the prose hard-codes one device

**Verdict:** UNVERIFIABLE

**Evidence:** establishing this requires two full inference passes over the 128
images, on two devices, which is exactly what I was told not to run. Nothing in
the annotation file, the notebook source or torchvision's metadata bears on it.
What I *can* confirm is the exposure the claim describes: cell 38 states
`**27.51**` and cell 51's table states `27.51` as flat fact, with no device
named anywhere near either, and `DEVICE` in cell 3 is chosen at runtime
(`cuda` → `mps` → `cpu`), so different readers do run different devices. Whether
the number actually moves is untested.

**Severity:** misleads a student, *if* real
**Origin:** hand-written prose
**Fix:** verify on two devices; if it moves, print the number rather than hard-coding it, and say which device produced it.

---

### Claim 6 — cell 6's "⏱ 60-90 seconds, instant afterwards" is a bandwidth division, omits the second download, and is false on an ephemeral disk

**Verdict:** CONFIRMED

**Evidence:** byte counts, both exact:
```
$ curl -sIL http://images.cocodataset.org/annotations/annotations_trainval2017.zip
HTTP/1.1 200 OK
Content-Length: 252907541          # = 241.19 MiB

$ 128 jpegs 20325087 bytes         # notebooks/datasets/coco/images/*.jpg
$ ann bytes 19987840                # instances_val2017.json on disk
```
Arithmetic on those: 252,907,541 B in 60–90 s is 2.8–4.2 MB/s (22–34 Mbit/s), an
assumption the cell never states; the same archive is ~20 s at 100 Mbit/s and
~100 s at 20 Mbit/s. The cell's budget covers only the archive — the 128
sequential `urllib.request.urlretrieve` calls (20,325,087 B, 128 round trips) are
not in the ⏱ figure at all. Confirmed from the source: `tools/notebooks/lecture_17.py:109-113`.

"Instant afterwards": `DATA = Path("datasets/coco")` (`lecture_17.py:76`) is
relative to the working directory, so persistence depends entirely on the
runtime's disk. On a Colab VM — the platform the header targets — nothing
carries over, so the reader pays 241 MiB plus the 167 MB of cell-22 weights every
session. That last step is reasoning from known Colab behaviour, not a
measurement I made.

The claim's own concession that "~241 MB" is MiB-labelled-MB and should not be
counted as an error is right, and I agree with it.

**Severity:** misleads a student
**Origin:** hand-written prose (the ⏱ line is a code comment, hand-written)
**Fix:** state the assumed link speed, include the 128 JPEGs in the budget, and replace "instant afterwards" with "instant afterwards *on a persistent disk*; Colab re-downloads every session".

---

### Claim 7 — "wrong by a factor of nine" is a ratio of MAEs presented as a property of the printed number

**Verdict:** CONFIRMED

**Evidence:** code cell 31 prints two numbers (`mean objects per image`,
`count MAE`). Against the notebook's own figures:
```
34.52 / 7.02 = 4.9174     <- the mean, factor of FIVE
27.51 / 3.00 = 9.1700     <- the MAE ratio, factor of nine
```
7.02 is re-derived from the annotation file (`n_true.mean()`); 34.52, 27.51 and
3.00 are the notebook's own stated values, not re-derived. The structural point
does not depend on them: the "factor of nine" is a ratio of two *errors*, whereas
the phrase "the number is wrong by a factor of nine" (cell 0, and again in cell
30's usual-student bullet) attaches to the printed count, which is off by ~4.9.
§1.4 exists for exactly this — name the operation.

**Severity:** misleads a student
**Origin:** hand-written prose
**Fix:** "the count MAE it reports is nine times the MAE of the corrected version".

---

### Claim 8 — "four times worse than one" is 4.57

**Verdict:** CONFIRMED

**Evidence:** cell 38 verbatim: *"took the answer from 'half the error of a
system that ignores the picture' to 'four times worse than one'."* Using the
notebook's own three figures and the baseline I re-derived (6.02):
```
3.00 / 6.02  = 0.4983   -> "half"  ✓
27.51 / 6.02 = 4.5698   -> not four
```
Under §1.2's round-half-up, 4.57 rounds to **5**, not 4, so the prose rounds in
the direction that understates the defect the section exists to dramatise. The
claim's own gloss — *"which rounds to four and a half"* — is loose; 4.57 rounds
to 4.6 at one decimal and to 5 at zero. The finding stands either way.

**Severity:** cosmetic
**Origin:** hand-written prose
**Fix:** "four and a half times worse".

---

### Claim 9 — `len(pred["boxes"])` appears in no code cell; `box_score_thresh` appears in no code cell

**Verdict:** CONFIRMED

**Evidence:** string search across all 16 code cells and all 36 markdown cells:
```
'len(pred["boxes"])'    markdown: 1 (cell 32)   code: 0
"len(pred['boxes'])"    markdown: 1 (cell 30)   code: 0
'box_score_thresh'      markdown: 2 (cells 30, 32)   code: 0
'box_detections_per_img' markdown: 3   code: 1 (cell 34, a print string)
```
Code cell 31 actually contains `len(preds[im["id"]]["boxes"])`. The name `pred`
(singular) first binds at cell 37, in `def count_objects(pred, thresh)` — five
cells *after* cell 32 invites the reader to reason about `pred["boxes"]`. A
reader who string-searches the quoted expression, which §3.1's evidence says two
of three readers did, finds nothing.

The library defaults the prose asks the reader to take on trust are correct, and
the one-line check the claim proposes works:
```
$ python3 -c "from torchvision.models.detection.faster_rcnn import FasterRCNN; import inspect; \
  s=inspect.signature(FasterRCNN.__init__); print([ (n,s.parameters[n].default) for n in \
  ('box_score_thresh','box_nms_thresh','box_detections_per_img')])"
[('box_score_thresh', 0.05), ('box_nms_thresh', 0.5), ('box_detections_per_img', 100)]

$ m = fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None)
roi_heads.score_thresh 0.05 | detections_per_img 100 | nms_thresh 0.5
```
torchvision 0.28.0. One correction to the claim: `box_score_thresh` is named in
prose **twice** (cells 30 and 32), not three times.

**Severity:** misleads a student
**Origin:** hand-written prose
**Fix:** quote `len(preds[im["id"]]["boxes"])` verbatim, and add `print(model.roi_heads.score_thresh, model.roi_heads.detections_per_img)` to cell 22 so 0.05 and 100 are printed rather than asserted in prose.

---

### Claim 10 — "reviewer question 3" and "reviewer question 5" are used five cells before they are named

**Verdict:** CONFIRMED

**Evidence:** every occurrence, by cell:
```
27 markdown  reviewer question 3
27 markdown  reviewer question 5
30 markdown  reviewer question 5
32 markdown  Reviewer question 5     <- first time either is stated
32 markdown  Reviewer question 3
```
Cell 27's Left-open bullet reads *"That is reviewer question 3 answering reviewer
question 5 before it is asked"* — using both by number, five cells before cell 32
says what they are. Nothing in lecture 17 lists the five questions; a reader
without lecture 3 cannot resolve them, and cannot tell from this notebook that
lecture 3 is where they live. §7.5 (define vocabulary on first use) and §3.3
(cross-references must resolve).

**Severity:** misleads a student (second-language / working-alone reader specifically)
**Origin:** hand-written prose
**Fix:** at cell 27, write "reviewer question 3 (*what is the shape here?*)" and add a one-line pointer to lecture 3 §13.

---

### Claim 11 — `box_map` 37.0 is offered for comparison against count MAEs

**Verdict:** CONFIRMED

**Evidence:** cell 22 prints
```
weights.meta["_metrics"]  ->  {'COCO-val2017': {'box_map': 37.0}}
```
(verified from static metadata; no weights downloaded), and cell 21's How-you-
would-catch-it bullet instructs: *"print the weights' own reported metrics on the
full 5,000 images. Your 128-image number should be read next to it, not instead
of it."* The notebook's own numbers (cells 38 and 51) are count MAEs — 6.02,
27.51, 3.00. Mean average precision on 5,000 images and count MAE on 128 images
share neither quantity nor corpus nor scoring window; there is no arithmetic
relation in either direction. §2.1 requires the window be held constant and
stated; neither is.

**Severity:** misleads a student
**Origin:** hand-written prose
**Fix:** say explicitly that 37.0 is mAP on a different corpus and cannot be compared with any number in this notebook — it is there only to show that the weights ship with a published score.

---

### Claim 12 — "examinable" appears once, in a code comment; sections 2–10 carry no marking

**Verdict:** CONFIRMED

**Evidence:**
```
'examinable'   markdown cells: 0   code cells: 1
```
The single occurrence is `tools/notebooks/lecture_17.py:53`, inside the SETUP
string: `# Not examinable: this is engineering hygiene, not machine learning.`
It renders as a Python comment in code cell 3, not as a section label. Section
headings 2 through 10 (`## 2 · The corpus…` through `## 10 · Where we are`)
carry none of §8.3's three labels. §8.3 cites lecture 19's single occurrence as
the defect; lecture 17 reproduces it exactly, and its one occurrence is in a
place §8.3 does not even count.

**Severity:** wrong but harmless
**Origin:** notebook structure
**Fix:** add one of *examinable* / *not examinable — engineering* / *beyond the book* under every `##` heading.

---

### Claim 13 — `n_crowd` is printed but its consequence is never quantified

**Verdict:** CONFIRMED

**Evidence:** cell 7 states the drop *"is a **choice**, it changes every count
below, and this is where it is recorded"* — and no cell says by how much.
Re-derived from the annotation file, both ways:
```
crowds dropped (as shipped):   n_true.sum() 898   mean 7.02   one-box MAE 6.02
crowds included:               n_true.sum() 906   mean 7.08   one-box MAE 6.08
```
Every number the claim gives for the crowd-inclusive corpus is exactly right.
The one figure it gives that I did **not** verify is "reported MAE 3.00 → 2.97",
which needs inference. §1.3 requires the sentence either justify the choice or
report the version that does not depend on it; this cell does neither, and the
delta is 0.06 on the mean and 0.06 on the baseline — small enough that stating
it would have cost one line and would have *defended* the choice.

**Severity:** wrong but harmless
**Origin:** hand-written prose
**Fix:** append to cell 7: "with the 8 crowd regions kept, the corpus is 906 objects, mean 7.08, baseline 6.08 — the choice moves the headline by less than 0.1."

---

## Summary
```
confirmed: 13   false positive: 0   unverifiable: 1
of the confirmed, 9 mislead a student
origin split — prose: 9   code: 1   structure: 3
```

**Per-claim verdicts:** 1 C · 2 C · 3 C · 4 C · 5 C · 5b U · 6 C · 7 C · 8 C ·
9 C · 10 C · 11 C · 12 C · 13 C

**Duplicates / shared roots:**
- Claims **7** and **8** are two arithmetic errors in the same three sentences
  (cell 0 + cell 38 + cell 30's usual-student bullet) about the same three
  numbers. One fix addresses both. Counted twice.
- Claim **2**'s items 3–5 and claim **3** both indict cell 30, but for different
  rules (§8.1 staging vs §6.1 budget). Not duplicates.
- Claim **1** is the root that makes **5b** and half of **9** unresolvable for a
  reader. Not a duplicate; a precondition.

**Notes on claim quality.** No claim in this section is a false positive, which
is unusual and worth stating plainly — the corpus arithmetic in particular is
exact to the last digit across every figure I could re-derive from
`instances_val2017.json` and torchvision's metadata. Three claims are imprecise
inside a correct verdict, and the imprecisions are recorded above:

- **4** — the `i` row is not a rebinding at all (both bindings are
  comprehension-scoped); `c`@9 and `k`@25 likewise.
- **9** — `box_score_thresh` occurs twice in prose, not three times.
- **8** — 4.57 rounds to 5, not to "four and a half".

**Two findings the Phase A report missed**, both checkable from the text alone:

1. Cell 23 contradicts itself inside one paragraph: *"⏱ **1 to 2 minutes** …
   the same loop took **39 s** on an idle laptop"*. 39 s is below its own stated
   lower bound, independent of any device the report measured.
2. `tools/check_notebooks.py --advisory` already **FAILs** this notebook on §6.1
   (16 full annotations, budget 10) and raises 28 §1.2 advisories. The report
   quotes neither, and the checker is the course's own instrument.

**Not tested here (by instruction):** every inference-dependent figure — 34.52,
27.51, 3.00, +2.41, 4,419/4,420 boxes, the score distribution (1,224 / 1,206 /
median 0.205), the 100-box cap, the sweep minimum 1.92 at 0.75, the three visual
claims under the figure, and both wall-clock measurements in claims 5 and 5b.
The detector was not run.
