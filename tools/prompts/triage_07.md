# Triage — lecture 7 defect report

Claims triaged: the seventeen numbered entries (0–16) of the *Defects found in
the current notebook* section of `tools/prompts/lecture_07.md`.

**Environment.** Python 3.13, scikit-learn 1.7.2, numpy 2.3.5, pandas 2.3.3,
Apple M4 Max (16 cores). CoverType was already cached under
`~/scikit_learn_data/covertype`, so the whole pipeline was re-run end to end at
`RANDOM_STATE = 42` rather than taken on trust. Verification scripts:
`scratchpad/verify.py` (pipeline), `scratchpad/nbstruct.py` (notebook
structure), `scratchpad/cold.py` / `cold2.py` (cold-fetch timing).

**Two standing caveats.**

1. *No stored outputs.* `notebooks/lecture-07.ipynb` has 0 outputs across 69
   cells and every `execution_count` is `None`. Per the brief this is true of
   all 23 notebooks and is noted once here, not repeated per claim. Every
   numeric verdict below therefore compares prose against **my re-run**, not
   against the notebook's own output.
2. *The machine was under heavy load.* `uptime` reported load averages of
   **148–180** on 16 cores throughout (roughly fifty agents active in this
   session). Absolute wall-clock numbers are therefore unreliable in the
   *upward* direction. This does not affect claims that a stated duration is
   **overstated** (load only makes measurements slower, so a measurement below
   the stated figure is safe), but it makes the cold-fetch claim unsettleable —
   see Claim 4.

---

### Claim 0 — the notebook ships with no stored outputs at all

**Verdict:** CONFIRMED

**Evidence:**
```
$ python3 nbstruct.py
cells 69 stored outputs 0
execution_counts set: {None}
```
`sum(len(c.get("outputs", [])) for c in nb["cells"]) == 0` across all 69 cells,
and no cell has ever been executed in the committed file. The consequence the
claim draws is correct: under §1.2 no prose figure in this notebook can be
reconciled against an output on the page.

**Severity:** misleads a student
**Origin:** notebook structure
**Fix:** none needed in this lecture — this is a course-wide property of all 23
notebooks and belongs in a single rebuild-wide decision about whether notebooks
ship executed, not in lecture 7's defect list.

---

### Claim 1 — every headline number belongs to the `min_samples_leaf=1` tree the notebook says it rejected

**Verdict:** CONFIRMED

**Evidence:** both trees fitted on the same 48,000 training rows, scored on the
same 12,000 test rows, `random_state=42`:

```
--- min_samples_leaf=20: test acc 0.730250 (73.0%) mean cond 7.9661 max 8 leaves 163 cols 24 train 0.744062
    leaf sizes min 20 median 48 max 6203  n 163
    Aspen recall 0.025510 precision 0.5000 support 196.0
    traced leaf: Krummholz 0.9093 of 463
    481 in leaf sizes? False;  neighbours around 481: [460, 463, 487, 494]
--- min_samples_leaf=1:  test acc 0.733417 (73.3%) mean cond 7.9941 max 8 leaves 206 cols 30 train 0.748146
    leaf sizes min 1 median 27 max 6203  n 206
    Aspen recall 0.030612 precision 0.6667 support 196.0
    traced leaf: Krummholz 0.9002 of 481
    481 in leaf sizes? True
```

Every prose figure the claim lists matches the **rejected** tree, not the
shipped one:

| Prose | Notebook cell | shipped (`msl=20`) | rejected (`msl=1`) |
|---|---|---|---|
| "73.3%" test accuracy | 61 (§15 prompt), 68 (§16 table) | **73.0250%** | 73.3417% |
| "7.99" conditions | 68 (§16 table) | **7.9661** | 7.9941 |
| "90% of the 481 training patches" | 44 (§12), 45 (§12 prompt), 55 + 57 (§14) | **90.93% of 463** | 90.02% of **481** |
| Aspen "never once predicts correctly" | 61 (§15 prompt) | recall **2.55%**, 5 of 196 | 3.06% |

The `481` figure is decisive and cannot be a rounding artefact: no leaf of the
shipped tree holds 481 patches (nearest are 463 and 487), and the shipped
tree's traced leaf at row 27 holds exactly the 463 the report predicts.

Two things the report did **not** catch, found while verifying this claim:

* Cell 18 (§6's anchor bullet) also carries the rejected tree's number — *"the
  distance from 48.8 to 73.3 is what was earned"*. Same defect, a fourth site.
* The "90%" in cells 55 and 57 is wrong on its own terms as well: the shipped
  tree's leaf prints `91%` (`dist.max()` = 0.9093, `:.0%` → 91), so §14's prose
  will contradict §14's own output the moment the notebook is executed.

**Severity:** misleads a student
**Origin:** hand-written prose
**Fix:** re-derive §6, §12, §14, §15 and §16 from the `tree` object bound in
§12 — 73.0%, 7.97, 91% of 463, Aspen recall 2.6% — and fix the four sites
above, not just the three the report lists.

---

### Claim 2 — "73.3%" is also the cross-validated number, mislabelled as a test score

**Verdict:** CONFIRMED — but as a **duplicate** of Claim 1, and its causal
attribution is not establishable.

**Evidence:** the grid value the claim cites is real:

```
param_min_samples_leaf    1      5      20     50     200    500
param_max_depth
8.0                    0.7375 0.7372 0.7335 0.7314 0.7152 0.6969
```

0.7335 → 73.35% → "73.3%", and cell 68's column header is verbatim
`| Model | Test accuracy | Conditions per justification |`.

What is **not** establishable is that the CV figure is the *source* of the
"73.3%". The rejected tree's test accuracy is 73.3417%, which rounds to the same
three characters. Two candidate origins produce a byte-identical string, so the
attribution is a guess either way. The demonstrable defect — §16's *Test
accuracy* column does not hold the shipped tree's test accuracy (73.0%) — is
exactly the defect already counted as Claim 1.

**Severity:** misleads a student
**Origin:** hand-written prose
**Fix:** none beyond Claim 1's fix, which resolves this figure too. Do not count
it as a second defect.

---

### Claim 3 — §8's cross-validation figures are wrong twice in one bullet

**Verdict:** CONFIRMED, with the second half of the claim refuted.

**Evidence:** cell 25's prompt box reads *"A mean of 82.6% built from folds
spanning four points is a different object from one built from folds spanning
half a point."* Measured:

```
[cell8 cv njobs-1 1.69s] mean 0.820646 min 0.814167 max 0.828438 span 1.427pts
free test acc 0.8263333333333334
```

* **"82.6%" is wrong** — the cross-validated mean is **82.06%**, and 82.6% is
  the *same tree's test accuracy* (82.6333%), printed by cell 29 and quoted
  again in §16. Two different quantities under one figure, §1.5.
* **"folds spanning four points" is not a defect.** Read as printed, the
  sentence contrasts two *hypothetical* fold spreads ("four points" versus "half
  a point") to make a general point about what a mean conceals. Neither number
  is offered as this run's span. The report's *"both halves of the sentence are
  wrong"* overstates: one half is wrong, one half is a rhetorical contrast.

The real span is 1.427 points, which happens to sit between the two hypotheticals
— so the illustration is not even misleading, only the figure attached to it.

**Severity:** misleads a student
**Origin:** hand-written prose
**Fix:** replace "82.6%" with 82.1% in cell 25; leave the four-points/half-a-point
contrast alone.

---

### Claim 4 — the first data cell is thirteen times slower than it says (391.4 s against "~30 s")

**Verdict:** UNVERIFIABLE

**Evidence:** four cold fetches into a fresh empty `data_home`, plus an isolated
download:

```
COLD fetch_covtype:  34.7 s     cache size on disk: 14.3 MB   WARM: 0.7 s
trial 0 COLD:       196.5 s
trial 1 COLD:       155.7 s
CLEAN COLD:         161.0 s                                   WARM: 1.73 s
raw download of 11.2 MB: 1.0 s
```

`uptime` during these runs: load averages **148.78 / 172.47 / 180.50** on 16
cores. The bulk of `fetch_covtype` is a single-threaded CSV parse, so it is
exactly the workload that a 10× oversubscribed machine distorts, and the spread
(34.7 s to 196.5 s, a factor of 5.7 across four runs of identical work) shows
the distortion directly.

What **is** settled:

* the download is **1.0 s for 11.2 MB** — the notebook's *"about 11 MB"* is
  correct, and the report agrees;
* the cache is **14.3 MB** on disk;
* warm re-fetch is **0.7–1.7 s**.

What is **not** settled: the 391.4 s figure was not reproduced in four attempts,
and the fastest reading (34.7 s) matches the notebook's "~30 s" almost exactly.
I cannot distinguish "the comment understates by 13×" from "the comment is right
and this machine is saturated". The direction of the claim is plausible — three
of four readings exceeded 150 s — but the magnitude the report asserts is not
something I reproduced, and the brief's rule is to say so rather than guess.

**Severity:** misleads a student *if true* — this is the first cell a student
runs, and six minutes of silence reads as a hang.
**Origin:** hand-written prose (an authored comment inside `COVER_LOADER`, not
generated code)
**Fix:** re-measure on an idle machine before changing the comment. If it holds,
state a range and the cause (*"1 s to download, then several minutes to parse,
single-threaded"*) rather than a point estimate.

---

### Claim 5 — the other three ⏱ markers overstate by 3–12×

**Verdict:** CONFIRMED for the three timings; the *ordering* sub-claim is a
16-core artefact and does not hold for the runtime the notebook targets.

**Evidence:** measured under the same heavy load, which can only inflate these
numbers:

| Marker | Cell | notebook says | `n_jobs=-1`, 16 cores | `n_jobs=1` |
|---|---|---|---|---|
| §8 CV | 24 → 26 | about 30 seconds | **1.69 s** | 1.48 s |
| §10 depth sweep | 34 → 36 | about 90 seconds | **10.53 s** | 32.5 s |
| §11 grid search | 40 → 42 | about 2 minutes | **2.18 s** | 61.0 s |

All three are overstated on the notebook's own settings (both cells pass
`n_jobs=-1`), §8 by roughly 20×. Confirmed.

**The ordering sub-claim is wrong.** The report says §11's grid *"is in fact the
fastest of the three"* and that the ⏱ ordering is backwards. That is true only
at 16 cores. At `n_jobs=1` the ordering is the one the notebook states —
sweep 32.5 s, grid 61.0 s — because `GridSearchCV` does ten times the work and
only wins when there are enough cores to hide it. A Colab CPU runtime has two
cores, so the grid will be the slower cell there, exactly as the notebook says.
The report generalised a 16-core result to the student's machine.

**Severity:** misleads a student
**Origin:** hand-written prose
**Fix:** replace the three markers with measured 2-core figures, or drop them —
§7.1 only requires a marker above ~20 s, and on 16 cores none of these three
qualifies. Do not "fix" the ordering.

---

### Claim 6 — §5's "twelve sections from now" does not resolve

**Verdict:** CONFIRMED

**Evidence:** cell 15 (§5's prompt box) reads *"…decides that Aspen will be
invisible in the confusion matrix twelve sections from now."* The confusion
matrix is cell 64, under **§15** (heading in cell 60). Section headings in
order:

```
cell  1: ## 1 · Setup            cell 34: ## 10 · What does depth actually buy?
cell  4: ## 2 · The brief        cell 40: ## 11 · Tune what is left
cell  5: ## 3 · The data         cell 44: ## 12 · The model we are going to ship
cell 11: ## 4 · Split before…    cell 48: ## 13 · Read the tree
cell 14: ## 5 · Look at the labels
cell 17: ## 6 · A number to compare against
cell 20: ## 7 · Commit           cell 54: ## 14 · Trace one prediction
cell 21: ## 8 · One tree…        cell 60: ## 15 · The test set. Once.
cell 27: ## 9 · An assistant…    cell 68: ## 16 · Where we are
```

§5 → §15 is **ten** numbered sections. Counting the three `###` subsections
between them as well (*Reviewer question 3*, *The corrected specification*,
*Read the leaf carefully*) gives **thirteen** headings. Under neither counting
is the answer twelve. §3.3.

**Severity:** wrong but harmless
**Origin:** hand-written prose
**Fix:** "ten sections from now", or name the target: "in §15's confusion matrix".

---

### Claim 7 — the half-integer claim contradicts the output above it

**Verdict:** CONFIRMED

**Evidence:** cell 52 is
```python
print(export_text(tree, feature_names=short, class_names=COVER_NAMES,
                  max_depth=2, decimals=0))
```
and its actual output begins
```
|--- Elevation <= 3046
|   |--- Elevation <= 2510
```
Cell 53's markdown immediately below reads *"The thresholds sit at half-integers
— CART puts a split midway between two adjacent observed values, so nothing in
the data sits exactly on one."* The unrounded values are
`tree.tree_.threshold[0]` = **3046.5** and **2510.5**, but `decimals=0` means
the reader sees no half-integer anywhere on the page.

The report's secondary figure also checks out:
```
n thresholds 162   frac ending .5 0.7099
```
**71.0%** of the shipped tree's 162 thresholds end in `.5`, not all of them —
the remaining 29% are whole numbers, which is consistent with the same midpoint
rule when the two bracketing values are two units apart.

**Severity:** misleads a student
**Origin:** hand-written prose (the `decimals=0` in generated code is what makes
it visible)
**Fix:** pass `decimals=1`, or make the point about the unrounded value and
point the reader at `tree.tree_.threshold[0]`. Also change "the thresholds" to
"about seven in ten of the thresholds".

---

### Claim 8 — `2959.5` appears nowhere in the notebook

**Verdict:** CONFIRMED

**Evidence:** cell 51's student bullet says *"assuming a threshold of 2959.5
means something about the terrain. It means two training patches were at 2959
and 2960."* The shipped tree's complete set of Elevation thresholds:

```
[2286.0, 2307.5, 2329.0, 2379.5, 2415.5, 2510.5, 2616.5, 2694.5, 2695.5,
 2699.5, 2744.0, 2954.5, 3046.5, 3089.5, 3114.0, 3196.5, 3213.5, 3216.5,
 3259.5, 3306.5, 3330.5, 3338.0, 3347.0, 3359.5, 3368.5, 3401.5, 3452.0, 3474.0]
2959.5 present? False
```

It **is** a threshold of the unconstrained `free` tree:
```
2959.5 in free tree Elevation thresholds? True
```
which is never printed anywhere in the notebook. So the figure was transcribed
from the wrong tree, the same root cause as Claim 1, and a reader who goes
looking for it in the `export_text` output will not find it. §1.2.

**Severity:** wrong but harmless
**Origin:** hand-written prose
**Fix:** use 2954.5 or 3046.5 — both are real thresholds of the shipped tree and
2954.5 makes the same point.

---

### Claim 9 — §10's "usual student version" describes something that does not happen

**Verdict:** CONFIRMED, with the qualitative point partly surviving.

**Evidence:** cell 37 says a linear y-axis *"shows one point rising and eleven at
zero, and concludes leaf count 'explodes at depth 12' when it has been doubling
all along."* Measured leaf counts and their position on a linear axis scaled to
the 1,076 maximum:

```
leaves            2   4   8  16  32  60 113 206 350 539 782 1076
% of axis range   0   0   1   1   3   6  11  19  33  50  73  100
leaf ratios     2.0 2.0 2.0 2.0 1.88 1.88 1.82 1.70 1.54 1.45 1.38
```

* *"one point rising and eleven at zero"* — four points sit at 33%, 50%, 73%
  and 100%, clearly visible, plus depth 8 at 19% and depth 7 at 11%. Six points
  are near the floor, not eleven, and six are legible. The count is wrong,
  though the underlying observation that a linear axis compresses the low end
  is fair.
* *"doubling all along"* — it doubles for exactly four steps and then decays
  monotonically to 1.38. The decay is itself the interesting finding and the
  bullet asserts its opposite.

§6.2: an invented failure rather than an observed one.

**Severity:** wrong but harmless
**Origin:** hand-written prose
**Fix:** state the real reason for `semilogy` — growth per level is geometric
with a decaying ratio (2.0 → 1.38), and a log axis is where that reads as a
straight line.

---

### Claim 10 — §11's "barely matters" has no stated range

**Verdict:** CONFIRMED

**Evidence:** cells 41 and 43 both say *"Along the `max_depth=8` row,
`min_samples_leaf` barely matters"* with no numbers attached. The row:

```
param_min_samples_leaf    1      5      20     50     200    500
8.0                    0.7375 0.7372 0.7335 0.7314 0.7152 0.6969
```

0.7375 → 0.6969 is a **4.06-point** swing across the printed row — ten times the
0.40 points (0.7375 − 0.7335) that §12 calls "the price of auditability". The
"barely matters" reading holds only for `min_samples_leaf ≤ 50`, where the swing
is 0.61 points. The reader is given no way to tell which.

The report's supporting claim about the deck also checks out —
`slides/lecture-07.html` contains the string `73.8% at 1, 71.5% at 200`, so the
range exists upstream and was dropped on the way into the notebook. §1.3.

**Severity:** misleads a student
**Origin:** hand-written prose
**Fix:** *"from 1 to 50 it moves 0.6 points; across the whole row, 4.1"*.

---

### Claim 11 — §15's cell does not satisfy its own specification

**Verdict:** CONFIRMED

**Evidence:** cell 66's prompt box says *"**constraint** · report BOTH numbers —
one goes up and one goes down, and quoting either alone is an argument rather
than a measurement"* and its catch bullet says *"Show the pair"*. Cell 67:

```python
print(f"accuracy      {balanced.score(X_test, y_test):.1%}  "
      f"(was {acc:.1%})")
print(f"Aspen recall  {rep['Aspen']['recall']:.1%}")
```

The accuracy line prints a pair; the Aspen line prints one number. There is no
`rep0` — no `classification_report` on the unweighted tree — anywhere in the
notebook, so the second half of the pair is not merely unprinted, it is never
computed. Measured, the missing number and its counterpart:

```
balanced acc 0.5935833333333334  Aspen recall 0.8010204081632653
shipped  acc 0.730250            Aspen recall 0.025510
```

80.1% against **2.6%** — 77.5 points of recall bought for 13.7 points of
accuracy. The omitted half is the larger half of the argument, in the cell whose
whole subject is not quoting one number alone.

**Severity:** misleads a student
**Origin:** generated code
**Fix:** compute `rep0 = classification_report(y_test, tree.predict(X_test),
target_names=COVER_NAMES, output_dict=True, zero_division=0)` in §15 and print
`Aspen recall 80.1% (was 2.6%)`.

---

### Claim 12 — `ax` is rebound across types

**Verdict:** CONFIRMED

**Evidence:**
```
cell 38: fig, ax = plt.subplots(1, 2, figsize=(11, 3.4))   -> numpy.ndarray, shape (2,)
cell 50: fig, ax = plt.subplots(figsize=(13, 4.5))         -> matplotlib.axes.Axes
cell 64: fig, ax = plt.subplots(figsize=(7, 6))            -> matplotlib.axes.Axes
```
Cell 38 indexes it (`ax[0].plot`, `ax[1].semilogy`); cells 50 and 64 pass it
whole (`ax=ax`). One name, two kinds of object — a literal §4.1 violation.

In practice this is harmless: each binding is created and consumed inside its
own cell, nothing reads `ax` across a cell boundary, and this is the standard
matplotlib idiom. It is a rule violation, not a hazard.

**Severity:** cosmetic
**Origin:** generated code
**Fix:** `axes` for the two-panel case in cell 38, `ax` for the single-axis
cases. Low priority.

---

### Claim 13 — `acc` is a loop variable and a headline result

**Verdict:** CONFIRMED

**Evidence:**
```
cell 36: acc = cross_val_score(clf, X_train, y_train, cv=cv, n_jobs=-1).mean()   # inside for d in range(1,13)
cell 62: acc = tree.score(X_test, y_test)
cell 62: print(f"test accuracy {acc:.1%}   (baseline {baseline:.1%})")
cell 67:       f"(was {acc:.1%})")
```
The depth sweep leaves `acc` bound to its last iteration, d = 12:
```
12   0.7861   1076
```
So re-running cell 36 after cell 62 — a natural thing to do while reading the
depth table — makes cell 67 print `(was 78.6%)` for the shipped model's test
accuracy of 73.0%, with nothing raised and nothing warned. Unlike Claim 12 this
is a live hazard: the two bindings are 26 cells apart and the third cell reads
across both.

This is the same shape as the `target`-clobbered-by-a-loop-variable defect
lecture 19 spends 200 words on, in a notebook that does not mention it. §4.1.

**Severity:** misleads a student
**Origin:** generated code
**Fix:** rename the loop variable in cell 36 to `cv_acc`.

---

### Claim 14 — cell 13 is not idempotent, and it breaks cell 10

**Verdict:** CONFIRMED — executed, not merely inspected.

**Evidence:** cell 13 both reads and deletes `X_all` / `y_all`:
```python
X, _, y, _ = train_test_split(X_all, y_all, train_size=60_000, ...)
...
del cover, X_all, y_all          # 250 MB we no longer need
```
Re-running it, and then cell 10, on stand-in frames of the same shape:
```
first run ok
second run -> NameError: name 'X_all' is not defined
cell 10 -> NameError: name 'X_all' is not defined
```
The `del` itself is the right trade — `cover.data.memory_usage(deep=True).sum()`
measures **250,997,316 bytes = 251.0 MB**, so the comment's "250 MB" is accurate
— but restart-and-run-all is the only way to change the subsample, and nothing
in the notebook says so. §4.3 plus §7.2.

**Severity:** wrong but harmless
**Origin:** notebook structure
**Fix:** one line in §4's markdown: *"this cell deletes `X_all`; to change the
subsample size, restart and run all from cell 1. Cell 10 also stops working
after this."*

---

### Claim 15 — the defect is announced three times before it fires

**Verdict:** CONFIRMED

**Evidence:** the `feature_importances_` cell is **cell 29**. Everything the
reader has already met, in order:

* **cell 0** (header): *"Cells marked **⚠ read before running** contain a defect
  on purpose."*
* **cell 27** (§9 heading): *"**⚠ Read before running.**"*
* **cell 28** (prompt box): labelled *"⚠ what the assistant returns"*, and its
  `left_open` bullet gives the answer away outright — *"`feature_importances_`
  has 54 entries, one per column, not one per prediction. It is the same vector
  for every patch in Colorado."*

— and then **cell 30** repeats it as *Reviewer question 3*. Three flags before,
one after, with the third stating the finding the reader is about to be asked to
discover. §8.1 exactly; the skimmer's objection quoted in GUIDELINES applies
verbatim.

The lecture script itself already prescribes the fix (§ Cell 9: *"Type this one
exactly as a person would, with no warning in the markdown above it, no ⚠ in the
label, and no hint in the box"*), so this is a known-and-unapplied correction
rather than a disputed reading.

**Severity:** misleads a student — it defeats the one piece of pedagogy the
section exists for
**Origin:** notebook structure
**Fix:** strip the ⚠ from cells 27 and 28 and move the `left_open` bullet into
cell 30, after the reader has written the five feature names down.

---

### Claim 16 — fourteen of sixteen sections carry no examinable marker

**Verdict:** CONFIRMED

**Evidence:** the notebook has sixteen `##` numbered sections (listed under
Claim 6). The string "examinable" occurs exactly three times:

```
cell  2 (markdown, §1): "not examinable, and it is here because a version mismatch…"
cell  3 (code,     §1): "# Not examinable: this is engineering hygiene, not machine learning."
cell 48 (markdown, §13): "*(Not examinable: this is tooling, not machine learning.)*"
```

Three occurrences covering two sections — §1 and §13. The other fourteen (§2–§12,
§14–§16) carry none. §8.3 requires every section to be marked *examinable*, *not
examinable — engineering*, or *beyond the book*.

**Severity:** cosmetic
**Origin:** notebook structure
**Fix:** add one marker line per section heading.

---

## Summary

```
confirmed: 16   false positive: 0   unverifiable: 1
of the confirmed, 10 mislead a student
origin split — prose: 10   code: 3   structure: 4
duplicates: Claim 2 is Claim 1 (§16's "73.3%" counted twice, once per candidate
            explanation; the two explanations are indistinguishable because
            73.3417% and 73.35% round identically)
```

**Calibration note.** Lecture 7 contains none of the three pre-verified claims,
so nothing here re-tests the independent verification.

**Sub-claims refuted inside otherwise-confirmed claims.** The report is accurate
at the level of *is there a defect here*, but overstates inside four entries.
Recorded separately because the rebuild will act on the detail, not the headline:

* **Claim 3** — "wrong twice in one bullet" is wrong once. "82.6%" is a genuine
  §1.5 error; "folds spanning four points" is a hypothetical contrast, not a
  claim about this run, and the real span (1.427) sits between the two
  hypotheticals it names.
* **Claim 5** — the ⏱ *ordering* is not backwards. At `n_jobs=1` the grid
  (61.0 s) is slower than the sweep (32.5 s), as the notebook says. The report
  generalised a 16-core result to a 2-core Colab runtime.
* **Claim 9** — "eleven at zero" is wrong, but six of twelve points really are
  near the floor on a linear axis, so the qualitative reason for `semilogy`
  survives; only the count and "doubling all along" fail.
* **Claim 4** — see below.

**The one unverifiable claim.** Claim 4's 391.4 s cold fetch did not reproduce
in four attempts (34.7 / 155.7 / 161.0 / 196.5 s) on a machine running at load
average 148–180. The download portion (1.0 s, 11.2 MB) and the cache size
(14.3 MB) both confirm; the parse time does not settle. It needs one measurement
on an idle machine before the comment is rewritten.

**Two sites the report missed**, both instances of Claim 1's root cause:

* cell 18 (§6) — *"the distance from 48.8 to 73.3 is what was earned"* is the
  rejected tree's accuracy, a fourth site beyond the three listed.
* cells 55 and 57 (§14) — the "90%" leaf proportion will contradict §14's own
  output on execution: the shipped tree's leaf prints **91%** (0.9093).
