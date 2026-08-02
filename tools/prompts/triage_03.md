# Triage — lecture 3

Claims triaged: the 20 numbered items of the *Defects found in the current
notebook* section of `tools/prompts/lecture_03.md`, plus the 6 bullets of
*Checked and clean* (21–26) and the *Could not check* paragraph (27). 27 in
total, in the order they appear.

Environment for every re-derivation below: python 3.13.5, scikit-learn 1.7.2,
numpy 2.3.5, matplotlib 3.10.6 — the same versions the Phase A report names.
MNIST was **already cached** by scikit-learn at
`~/scikit_learn_data/openml/openml.org/data/v1/download/52667/mnist_784.arff.gz`,
so the data-dependent claims were re-derived against the real data, not guessed.
Loading it takes 2.3 s.

**Stated once, not repeated per claim (per the brief):** `notebooks/lecture-03.ipynb`
stores no outputs — all 16 code cells have `execution_count: null` and zero
`outputs`. Every prose figure in the file is therefore unreconcilable against
the notebook's own stored data. This is claim 6 and is true; it is not restated
under the individual numeric claims.

**Not run:** no cell that fits on the 60,000 training rows was executed. Where a
mechanism needed an empirical demonstration (claim 4) it was run on a 6,000-row
subset and that is stated in the evidence.

---

### Claim 1 — `i_pos = int(np.argmax(y_train))  # first 5` selects a nine
**Verdict:** CONFIRMED

**Evidence:** loaded the cached MNIST, cast `y` to `uint8` exactly as cell 10
does, sliced `y_train = y[:60000]`:

```
argmax(y_train) = 4   y_train[i_pos] = 9   bool -> True
argmin(y_train) = 1   y_train[i_neg] = 0   bool -> False
argmax(y_train_5) = 0  label 5
argmin(y_train_5) = 1  label 0
first index where y_train == 5: 0
first index where y_train != 5: 1
first 16 labels: [5, 0, 4, 1, 9, 2, 1, 3, 1, 4, 3, 5, 3, 6, 1, 7]
```

Every element of the report's claim reproduces. `np.argmax` on a `uint8` label
vector returns the first index whose *digit value* is largest — index 4, a nine —
not the first 5. The printed truth column is `bool(y_train[i])`, and `bool(9)` is
`True`, so the cell displays a nine under a column that the reader will read as
"is a 5". Symmetrically `i_neg = 1` is a zero, and it reads as a non-5 only
because `bool(0)` happens to be `False` — the column is really testing *is the
digit non-zero*, which coincides with *is it a 5* on exactly one of the two rows
shown. The corrected expressions `np.argmax(y_train_5)` and
`np.argmin(y_train_5)` give 0 and 1, i.e. a real 5 and a real non-5.

The two `assert`s below are on shapes (`clf.predict(X_train[:8]).shape == (8,)`
and `.dtype == bool`) and are unaffected, so nothing catches this.

I agree with the independent verification. The aggravating detail the report
names is right and worth keeping: this is the cell whose entire subject is
"assert the shape, not the answer", and the answer it displays is wrong.

**Severity:** misleads a student
**Origin:** generated code (`tools/notebooks/lecture_03.py:324`)
**Fix:** `i_pos = int(np.argmax(y_train_5))` / `i_neg = int(np.argmin(y_train_5))`,
and print `y_train[i]` rather than `bool(y_train[i])` in the truth column.

---

### Claim 2 — cell 31 says `X_train[0]` is the known 5, but cell 33 never uses index 0
**Verdict:** CONFIRMED

**Evidence:** `y_train[0] == 5` is true (`first 16 labels` above), and cell 12
plots `X[i]` for `i in range(16)`, so "the 5 we plotted above" is a correct
statement about the data. But cell 33's only indices are `i_pos` and `i_neg`,
which evaluate to **4** and **1**. Index 0 appears nowhere in cell 33. The prose
directs the reader to an example the code below it does not use.

**Severity:** misleads a student
**Origin:** hand-written prose (cell 31)
**Fix:** none needed separately — fixing claim 1 makes `i_pos` equal 0 and the
sentence becomes true. See *duplicates* in the summary.

---

### Claim 3 — "imshow of a 784-long vector is not an error, it is a stripe"
**Verdict:** CONFIRMED

**Evidence:** run directly, no MNIST needed:

```python
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, numpy as np
plt.imshow(np.random.rand(784))
```
```
TypeError: Invalid shape (784,) for image data
```

It raises; it does not render a stripe. I agree with the independent
verification. (For the record, matplotlib *would* draw a stripe from a
`(1, 784)` or `(784, 1)` array — but a flat 784-vector, which is exactly what
`X[i]` is, is rejected.) §3.2: this is a claim offered to the reader as a fact
about what happens, and it does not happen.

**Severity:** misleads a student
**Origin:** hand-written prose (prompt-box constraint, cell 11;
`tools/notebooks/lecture_03.py:106`)
**Fix:** "…imshow of a 784-long vector raises `TypeError: Invalid shape (784,)
for image data` — it does not silently draw something wrong."

---

### Claim 4 — the `clone` explanation is false; reusing a fitted estimator does not raise the accuracies
**Verdict:** CONFIRMED

**Evidence:** three separate checks.

Signature and docstring:
```
warm_start default: False
warm_start : bool, default=False
    When set to True, reuse the solution of the previous call to fit as
    initialization, otherwise, just erase the previous solution.
```

Empirical, on a **6,000-row subset** (not the notebook's training cell), running
the cell-43 loop twice — once with `clone(clf)` per fold, once with
`fold_clf = clf`, which is precisely the "usual student version" the box names:
```
clone each fold : [0.964  0.9645 0.9635] mean 0.964
reuse same obj  : [0.964  0.9645 0.9635] mean 0.964
identical? True   reuse higher? False
```

Coefficient identity — a fresh pipeline fitted once versus a pipeline fitted on
3,000 rows and then refitted on the same 6,000:
```
fresh vs refit coefs identical? True   max abs diff 0.0
intercepts: [-1015.40142031] [-1015.40142031]
```

`random_state` is a fixed int, so each `fit` re-seeds and the results are
bit-identical. Cell 41's "each fold starting from the previous fold's
parameters", cell 42's "the accuracies come out higher" and "The numbers rise"
are all false.

The report's second half also holds: `cv.split` on 60,000 rows with 3 folds
gives train parts of exactly 40,000 (verified — every fold's test part is
20,000), so a student who wrote `fold_clf = clf` would leave `clf` fitted on the
*last fold's* 40,000 rows, silently changing what cells 46 and 54 measure. That
is the real consequence and the notebook states it nowhere.

**Severity:** misleads a student
**Origin:** hand-written prose (cell 41 body and cell 42's *Left open* and
*usual student version* bullets)
**Fix:** replace the mechanism with the true one — `clone` is about *what `clf`
is left holding afterwards*, not about carried-over parameters; state that the
fold scores are unchanged and that the damage is to `clf`'s fitted state for
cells 46 and 54.

---

### Claim 5 — "11 MB from openml" understates the download
**Verdict:** CONFIRMED

**Evidence:**
```
$ stat -f "%z %N" ~/scikit_learn_data/.../52667/mnist_784.arff.gz
15469256 .../mnist_784.arff.gz
```
15,469,256 bytes = 15.5 MB decimal, 14.75 MiB. "11 MB" is wrong under either
unit convention. The figure in the report (14.8 MiB) is 14.75 rounded; both
readings contradict 11.

**Severity:** wrong but harmless
**Origin:** generated code (comment, cell 7; `tools/notebooks/lecture_03.py:26`)
**Fix:** "~30 s the first time (about 15 MB from openml)".

---

### Claim 6 — the notebook stores no outputs at all
**Verdict:** CONFIRMED

**Evidence:** over the notebook JSON, code cells whose `execution_count` is not
`None` **or** which carry any `outputs`: `[]` — none. 56 cells, 16 of them code
(indices 3, 7, 10, 12, 16, 19, 21, 25, 30, 33, 36, 39, 43, 46, 49, 54), 40
markdown. `python3 tools/check_notebooks.py --advisory` reports
`note lecture-03.ipynb (26 advisory)`, every one of them of the form
"*N* not in any output" — the §1.2 check has nothing to reconcile against.
The §7.1/§9 `⏱` rule keys off stored execution time and likewise has nothing to
run against.

**Severity:** wrong but harmless (it is the course-wide condition the brief
tells us to note once, not a lecture-3 defect)
**Origin:** notebook structure
**Fix:** none needed at lecture level; a course-wide decision about committing
executed notebooks.

---

### Claim 7 — the same gap is given three sizes in one prompt box, and "five points" is wrong
**Verdict:** CONFIRMED

**Evidence:** cell 35 contains all three of "six points sounds like a lot",
"five points from a model with no parameters", and "makes 96.9% shrink to 5.9".
Under the notebook's own two figures, `96.9 − 90.96 = 5.94`. "Six points" and
"5.9" are both defensible roundings of 5.94; "five points" is not — it is a
round-down of nearly a full point, in the bullet that tells the student what
number to carry away.

The 96.9% itself was **not** re-derived (it needs the training cell), so this
verdict is about the internal arithmetic only. It does not depend on 96.9 being
right: whatever the cross-validated accuracy turns out to be, the box quotes one
subtraction three ways.

**Severity:** wrong but harmless
**Origin:** hand-written prose (cell 35)
**Fix:** use one figure — "six points" — in all three places, or compute the gap
in the cell and quote nothing in prose.

---

### Claim 8 — "Six numbers" under a five-row table
**Verdict:** CONFIRMED

**Evidence:** cell 54's `summary` dict has exactly five entries: *positives in
the training set*, *base rate*, *never-fires accuracy*, *ours, cross-validated*,
*ours, scored on its own rows*. Cell 55 opens "Six numbers, and their **status**
matters as much as their value: four measured, one stated by the client, one
discarded." Five printed rows, all five measured; the client's figure (1,000
items per shift, cell 4) is not among them, and nothing in the table is
"discarded". The literal reader counting to six finds five.

**Severity:** wrong but harmless
**Origin:** hand-written prose (cell 55)
**Fix:** either add the client's 1,000-per-shift row to the dict (which is
probably what was intended, since the sentence's taxonomy needs it) or say
"Five numbers, all measured".

---

### Claim 9 — cell 24's box asks for agreement "to three places" when the identity is exact
**Verdict:** CONFIRMED

**Evidence:** ran the split and the anchor (not a training cell — `NeverFires.fit`
is `return self`):
```
fold 0: test n=20000 pos=1807 neg=18193
fold 1: test n=20000 pos=1807 neg=18193
fold 2: test n=20000 pos=1807 neg=18193
per fold: array([0.90965, 0.90965, 0.90965])
mean 0.9096500000   1 - base_rate 0.9096500000
exactly equal? True   abs diff 0.0
```
5,421 / 3 = 1,807 and 60,000 / 3 = 20,000, both exact, so stratification places
identical fold compositions and every fold scores 1 − 1807/20000 = 0.909650. The
box asks for agreement "to three places" and the code asserts `atol=1e-4`; the
difference is **0.0**. §6.3 — a check with an exactly knowable answer was
available and a weaker one was asked for.

**Severity:** cosmetic
**Origin:** hand-written prose (prompt-box `check` clause, cell 24;
`tools/notebooks/lecture_03.py:227`)
**Fix:** "its accuracy should equal 1 minus the base rate **exactly** — 5,421 and
60,000 are both divisible by 3, so every fold scores 0.909650"; assert equality.

---

### Claim 10 — cell 6's "the next cell is about that" (the dtype of `y`)
**Verdict:** FALSE POSITIVE

**Evidence:** the report counts raw notebook cells and lands on cell 7, the
loader. But cell 7 is the cell this very prompt box specifies, and it *does*
print the dtype: `print(f"y {y.shape} {y.dtype}")  # note the dtype of y`. The
cell after it, cell 8, opens "`y` came back as an array of **one-character
strings**" — literally about the dtype. And counting *code* cells, which is the
convention the notebook's prompt boxes use elsewhere (see claim 12), "the next
cell" after cell 7 is cell 10, which is exactly the before/after-cast
demonstration the report nominates as the correct target.

Under all three readings the reader arrives at material about the dtype. Nothing
misdirects them.

**Severity:** n/a
**Origin:** hand-written prose
**Fix:** none needed.

---

### Claim 11 — cell 18's "it also catches the missing cast **two cells up**"
**Verdict:** CONFIRMED

**Evidence:** the cast `y = y.astype(np.uint8)` is in cell 10. The assertion
that catches it is in cell 19; the bullet making the claim is in cell 18.

- counting all cells: 19 → 10 is **nine** cells; 18 → 10 is eight.
- counting code cells (indices 3, 7, 10, 12, 16, 19, …): from cell 19, one code
  cell up is 16, two is **12** — the imshow grid, not the cast. Three is 10.

"Two cells up" resolves to the wrong cell under every counting convention in the
file. This is the one cross-reference of the three that genuinely fails.

**Severity:** wrong but harmless
**Origin:** hand-written prose (cell 18; `tools/notebooks/lecture_03.py:171`)
**Fix:** "it also catches the missing cast — the `astype` three code cells up".
Better still, name the cell rather than counting.

---

### Claim 12 — cell 29's "which is **the next cell but one**" (cross-validation)
**Verdict:** FALSE POSITIVE

**Evidence:** the bullet is about code — `StandardScaler().fit_transform` "leaks
the moment it meets cross-validation". Counting code cells from the cell the box
specifies (cell 30): the next code cell is **33**, the next but one is **36** —
which is exactly `cross_val_score(clf, ...)`. The reference resolves precisely.

The report counts raw cells and gets six, which is true but is not the counting
the sentence uses. Note that `cross_val_score` also appears *earlier*, at cell
25 (the anchor), so "the next" cannot mean the first occurrence in the file
either.

I would still change the wording, because a reader scrolling counts what they
see and passes three markdown cells on the way — but the claim as stated, that
the reference does not resolve, does not hold.

**Severity:** n/a
**Origin:** hand-written prose
**Fix:** none needed; optionally "…which is cell 36" for readers who count
rendered cells.

---

### Claim 13 — §8.1, the trap is announced repeatedly before it fires
**Verdict:** CONFIRMED

**Evidence:** the defect fires in cell 46 (`accuracy_score(y_train_5, y_pred)`
on the fitted rows). Announcements before it, quoted from the cells:

1. cell 0 — "Cells marked **⚠ read before running** contain a defect on purpose."
2. cell 44 — "**⚠ Read before running.**"
3. cell 44 — "it prints a number *better* than the one we just measured"
4. cell 45 — box title "**Prompt · the number you must not report**"
5. cell 45 — constraint "…not to be quoted"
6. cell 45 — student bullet "**The usual student version:** reporting this one."

Six explicit signals across three cells, all before the cell runs. GUIDELINES
§8.1 cites lecture 19's **four** as the defect; this is more. The report says
"five times" and then enumerates seven items — its tally is loose, but the
substance (more than lecture 19, and enough that "would you have caught it?" has
no honest answer) reproduces exactly.

**Severity:** wrong but harmless
**Origin:** notebook structure (cell ordering)
**Fix:** run cell 46 unannounced, have the reader write the number down, then
open §12 with the ⚠ and the contrast — as the report's own script proposes.

---

### Claim 14 — §6.1, 16 code cells and 16 full three-bullet prompt boxes
**Verdict:** CONFIRMED

**Evidence:** string counts over the notebook JSON:
```
"**Watch this prompt.**"                16
"> **Prompt ·"                          16
"* **Left open:"                        16
"* **The usual student version:"        16
"* **How you would catch it:"           16
code cells                              16
```
And the project's own checker agrees:
```
$ python3 tools/check_notebooks.py --advisory
FAIL  lecture-03.ipynb
        16 full annotations, budget is 10 (§6.1)
```
Budget is five to eight, never more than ten.

**Severity:** wrong but harmless
**Origin:** notebook structure
**Fix:** keep a one-line specification box on every code cell; reserve the
three-bullet form for the five to eight places where the prompt genuinely fails
(cells 9, 11, 24, 32, 42, 45 are the candidates).

---

### Claim 15 — §8.3, "examinable" appears twice, both on the setup cell
**Verdict:** CONFIRMED

**Evidence:** `examinable` occurs exactly twice in the whole file — cell 2
(markdown, "not examinable, and it is here because a version mismatch…") and
cell 3 (code comment, "Not examinable: this is engineering hygiene"). Both are
the setup section. The file has **14** `## ` sections (verified by listing them:
§1 Setup … §14 Record it); twelve of them carry no examinability marker at all.

**Severity:** wrong but harmless
**Origin:** hand-written prose
**Fix:** tag each of the 14 sections with *examinable*, *not examinable —
engineering*, or *beyond the book, for context*.

---

### Claim 16 — cell 10 is not idempotent and nothing says so
**Verdict:** CONFIRMED

**Evidence:** ran cell 10 twice against the real data.

First run (as the reader sees it):
```
before: object '5'
y == 5 finds 0 images
after:  uint8 5
```
Re-run after the cast:
```
before: uint8 5
y == 5 finds 6313 images
```
The demonstration inverts. `y` is rebound from an `object` array of strings to
`uint8` (§4.1: one name, two types), and on a second execution the line whose
whole job is to print **zero** prints **6,313** — the number of fives across all
70,000 rows. The cell carries the lecture's headline lesson and nothing warns
that re-running it destroys the lesson.

(The report's 6,313 is exact.)

**Severity:** misleads a student
**Origin:** generated code (cell 10)
**Fix:** compute the "before" from a separate name — e.g. `y_raw = mnist.target`
kept alongside `y` — so the cell is idempotent; or add an explicit "⚠ do not
re-run this cell" with the reason.

---

### Claim 17 — `y_test_5` is computed and never used
**Verdict:** CONFIRMED

**Evidence:** occurrences of `y_test_5` across all 16 code cells: `[(19, 1)]` —
the line that creates it, and nowhere else. Zero occurrences in markdown.
Meanwhile cell 51's reviewer-question table answers "What touched the test set?"
with "Nothing. `X_test` appears in the split and not again". `X_test` is indeed
clean (code: cell 16 only; markdown: cells 15, 50, 51), but the test *labels*
are read at cell 19 to build a binding that is then dead. A dead binding built
from held-out data is exactly what reviewer question 1 exists to catch, and the
table answers "Nothing".

**Severity:** wrong but harmless
**Origin:** generated code (cell 19)
**Fix:** delete the `y_test_5` line, or move it to the next notebook where it is
first needed; either way the cell-51 answer becomes true as written.

---

### Claim 18 — no timing anywhere names hardware
**Verdict:** CONFIRMED

**Evidence:** every duration in the file, extracted by regex:
```
cell 7  (code):  # ~30 s the first time (11 MB from openml)
cell 29 (md):    ⏱ about 30 seconds, which the comment gives but the specification does not
cell 30 (code):  # ⏱ about 30 s: one pass fits the scaler and then the classifier on 60,000 rows
cell 34 (md):    ⏱ **about 70 seconds** — three complete refits of scaler and classifier
cell 37 (md):    ⏱ **about 30 seconds.**
cell 41 (md):    ⏱ **about 70 seconds** — the same three fits, done by hand
cell 47 (md):    ⏱ **about four minutes**
```
None names a machine, a core count, or "Colab CPU runtime". §7.1 requires the
CPU figure. The durations themselves could not be re-measured (they need the
training cells).

**Severity:** wrong but harmless
**Origin:** hand-written prose
**Fix:** append the hardware to each, e.g. "⏱ about 70 s on a Colab CPU runtime
(2 vCPU)".

---

### Claim 19 — two `⏱` figures are in the wrong place
**Verdict:** CONFIRMED — but only one of the two

**Evidence:** §9's machine rule wants the marker in the markdown *above* the
cell. Checked per code cell:

- cell 7 — the "~30 s" lives **only** in the code comment, has no `⏱` character
  at all, and neither cell 5 nor cell 6 mentions a duration. This half of the
  claim holds.
- cell 30 — "⏱ about 30 s" is in the code comment *and* in cell 29, which **is**
  the markdown cell immediately above it. §9's rule is satisfied. The report's
  objection is that the figure sits in the *Left open* slot, which is for what
  the specification omits — but that is precisely what the bullet says it is
  doing ("which the comment gives but the specification does not"), and it is a
  legitimate use of the slot. This half does not hold.

**Severity:** cosmetic
**Origin:** hand-written prose
**Fix:** add "⏱ ~30 s on first run (about 15 MB downloaded)" to cell 6's
markdown, above cell 7. Cell 30 needs no change.

---

### Claim 20 — no restart-and-run-all evidence
**Verdict:** UNVERIFIABLE

**Evidence:** establishing §10.1 requires executing cells 30, 36, 39, 43, 46 and
49, all of which fit on 60,000 rows; the brief forbids it. What *can* be said is
already said elsewhere: there are no stored outputs (claim 6) and cell 10 is not
idempotent (claim 16), so a cold-kernel pass has neither been recorded nor can
be inferred. Whether it would pass is untested by anyone.

**Severity:** n/a — untested
**Origin:** notebook structure
**Fix:** run it once from a cold kernel during the rebuild and record the
outputs.

---

### Claim 21 — §5.1 / §5.2 clean: no markdown line indented ≥ 4 outside a fence, no indented fence marker
**Verdict:** CONFIRMED (the notebook is clean, as claimed)

**Evidence:** walked all 40 markdown cells line by line, tracking fence state:
`violations: NONE`. `tools/check_notebooks.py` likewise reports no §5.1/§5.2
failure for lecture-03 (its only FAIL is the §6.1 box budget). The two ``` ```
fences in the file, cells 27 and 52, both open and close at column 0.

**Severity:** n/a
**Origin:** n/a
**Fix:** none needed.

---

### Claim 22 — §3.1 clean: no ```` ```python ```` fence in any markdown cell
**Verdict:** CONFIRMED (clean, as claimed)

**Evidence:** markdown cells containing ```` ```python ````: none. The only
fenced blocks are cells 27 and 52, which contain the blank fill-in-by-hand forms
("Metric: ____"), not code. `check_notebooks.py`'s §3.1 advisory fires on
lecture-05 and not on lecture-03, consistently.

**Severity:** n/a
**Origin:** n/a
**Fix:** none needed.

---

### Claim 23 — the `5,421 → 9.04% → 90.96%` chain is exact and internally consistent
**Verdict:** CONFIRMED (clean, as claimed)

**Evidence:**
```
y_train_5.sum() = 5421
base_rate       = 5421/60000 = 0.09035      printed as 9.04%
1 - base_rate   = 0.90965                   printed as 90.96%
anchor.mean()   = 0.90965                   exactly equal, diff 0.0
```
9.04 + 90.96 = 100.00, and every prose occurrence of 90.96% matches what the
code prints. One footnote for the rebuild, not a defect in this claim: 90.965
under GUIDELINES §1.2's round-half-up would be 90.97, and Python's `:.2f` gives
90.96 because the binary double is 90.96499…. The notebook is self-consistent
because it quotes what it prints.

**Severity:** n/a
**Origin:** n/a
**Fix:** none needed.

---

### Claim 24 — cell 19's assert really does catch a skipped cast
**Verdict:** CONFIRMED (clean, as claimed)

**Evidence:** compared the raw object array of strings with the integer 5, as a
reader who skipped cell 10 would:
```
dtype: bool   sum: 0   shape: (60000,)
warnings: []
assert y_train_5.dtype == bool  -> True
AssertionError: did the cast to uint8 happen?   <-- fires
```
All-`False`, sum 0, **no warning at all** from numpy, and the `dtype == bool`
assert on the line above passes — so the count assert is the only thing standing
between the reader and a silently empty label vector. It fires.

**Severity:** n/a
**Origin:** n/a
**Fix:** none needed.

---

### Claim 25 — `NeverFires(BaseEstimator)` runs under sklearn 1.7.2 with no warning
**Verdict:** CONFIRMED (clean, as claimed)

**Evidence:** ran cell 25 verbatim under `warnings.simplefilter("always")` with
warnings recorded (not a training cell — `fit` is `return self`):
```
WARNINGS CAUGHT: []
per fold: [0.90965 0.90965 0.90965]
mean 0.9096500000
```
No `ClassifierMixin`, no `_estimator_type`, `scoring="accuracy"` with
`StratifiedKFold` — and scikit-learn 1.7.2 emits nothing.

**Severity:** n/a
**Origin:** n/a
**Fix:** none needed.

---

### Claim 26 — cell 51's "`X_test` appears in the split and not again" is true of the code
**Verdict:** CONFIRMED (clean, as claimed)

**Evidence:** `X_test` occurrences — code cells: cell 16 only (5 hits, all
within the split and its asserts). Markdown cells: 15, 50, 51. No code cell
after 16 mentions it. The statement is true *of `X_test`*; see claim 17 for what
it misses about `y_test`.

**Severity:** n/a
**Origin:** n/a
**Fix:** none needed.

---

### Claim 27 — everything requiring a fit on 60,000 rows could not be checked
**Verdict:** CONFIRMED (the scoping is correct)

**Evidence:** the figures the report lists — 96.9%, the 5.9-point gap, "they
agree to two tenths of a point", "one fold of the unscaled version lands five
points below its own siblings", "the two means differ by under two points",
"Half a point", "positive in 15/15", and the five wall-clock figures — all come
from cells 30, 36, 39, 43, 46, 49, which fit on the full training set. The brief
forbids running them and I did not. Every one of them is therefore **untested by
anyone**, not refuted.

The one figure in that list that can be bounded without a fit is "positive in
15/15": cell 49 asserts `(gaps > 0).all()`, so if the assertion has ever passed
the figure is 15/15 by construction. That is circular as evidence and is not
offered as verification.

**Severity:** n/a — untested
**Origin:** n/a
**Fix:** measure all of them during the rebuild and store the outputs, which
also resolves claim 6.

---

## Summary

```
confirmed: 23   false positive: 2   unverifiable: 2
```

- **confirmed:** 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 13, 14, 15, 16, 17, 18, 19, 21,
  22, 23, 24, 25, 26 — where 21–26 are "clean" claims confirmed as clean, and 19
  is confirmed for only one of the two cells it names.
- **false positive:** 10, 12 — both cross-reference claims that resolve under
  the notebook's own code-cell counting convention.
- **unverifiable:** 20 (needs a cold-kernel run of the training cells), 27 (the
  report's own scoping of what needs a 60,000-row fit; correct as scoping, and
  the figures inside it remain untested).

```
of the confirmed, 5 mislead a student
```

**1** (a nine displayed as the known 5), **2** (prose points at an index the
code does not use), **3** (a stated behaviour that raises instead), **4** (a
false mechanism taught as the reason for `clone`), **16** (a demonstration that
reverses itself on re-run). Claim 2 counts only as long as claim 1 stands — see
*duplicates*. The remaining confirmed items are harmless-but-wrong or cosmetic.

```
origin split — prose: 10   code: 4   structure: 3   (no origin: 10)
```

Of the 23 confirmed claims, 17 name a real defect and so carry an origin:

- **hand-written prose (10):** 2, 3, 4, 7, 8, 9, 11, 15, 18, 19.
- **generated code (4):** 1, 5, 16, 17.
- **notebook structure (3):** 6, 13, 14.
- **no origin (10):** 21–26 are clean-claims and 27 is a scoping statement, so
  they carry no defect; 10 and 12 are false positives (both would have been
  prose had they been real); 20 is unverifiable (it would be structure).

The audit's "defects concentrate in hand-written prose" pattern **holds here,
but not overwhelmingly**: 10 prose against 4 code and 3 structure. The single
most serious defect in this lecture (claim 1) is in *generated code*, not prose —
so the strong form of the pattern, "every genuine defect is in the markdown",
is false for lecture 3.

```
duplicates
```

- **1 and 2 are one defect.** Cell 33 selects index 4 instead of index 0; cell
  31's "`X_train[0]` is the 5 we plotted above" is a correct sentence about the
  data that only reads as broken because of it. Fixing `i_pos` to
  `np.argmax(y_train_5)` makes it 0 and repairs both. Count as one for
  rebuild-effort purposes.
- **6 and 20 overlap.** "No stored outputs" is the reason restart-and-run-all
  cannot be shown to hold; 20 adds only the non-idempotent cell 10, which is
  claim 16. Executing the notebook once and committing the outputs closes 6, 20
  and 27 together.
- **6 and 27 overlap.** 27 is the list of figures that 6 makes unreconcilable.
```
