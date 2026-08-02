# Triage — lecture 6 defect report

Triaging the 34 numbered claims in the `Defects found in the current notebook`
section of `tools/prompts/lecture_06.md`, against
`notebooks/lecture-06.ipynb` (generated from `tools/notebooks/lecture_06.py`).

**Environment.** Python 3.13.5, scikit-learn 1.7.2, numpy 2.3.5, pandas 2.3.3 —
identical to the versions the Phase A report claims to have used. Data:
`notebooks/datasets/titanic/train.csv` (cached, 891 rows), re-derived through a
verbatim transcription of the notebook's own cells 1–6 into
`.../scratchpad/tri06/setup6.py`.

**Two standing notes, made once and not repeated per claim:**

1. **The notebook stores no outputs** (`any(c.get('outputs') for c in cells)` is
   `False` over all 29 code cells). §1.2 therefore cannot be checked against the
   artefact at all; every prose figure below was recomputed from scratch.
2. **The machine was under heavy load throughout this triage** — `uptime` load
   average between **153 and 180** on a 16-core box, because ~40 other agents
   were running concurrently. All *wall-clock* numbers I report are therefore
   upper bounds and are not comparable with the Phase A "idle M4 Max" figures.
   Where a claim is about timing I test the **ratio**, which survives
   contention, and say so. This is itself the phenomenon claim 28 is about, and
   I hit it accidentally rather than by design.

**On the claim count.** The task message says 37 claims. The Phase A report
contains **34 numbered items**; the remaining sections (`Checked and clean`,
`Not checked`) hold 5 and 4 unnumbered bullets. I have triaged the 34 numbered
claims and spot-checked the `Checked and clean` block (results at the end).

---

### Claim 1 — `1.957` for degree 5 is stale; measured 1.922

**Verdict:** CONFIRMED
**Evidence:** re-ran the notebook's own `cross_validate` loop (cell 16) verbatim:

```
degree 1   22 cols train 0.3952 held-out 0.4681
degree 2   32 cols train 0.3806 held-out 0.4671
degree 3   52 cols train 0.3554 held-out 0.5381
degree 4   87 cols train 0.3104 held-out 1.3903
degree 5  143 cols train 0.2857 held-out 1.9217
degree 6  227 cols train 0.2839 held-out 1.8360
```

`1.957` appears in three markdown cells (15, 38, 53) and in none of them is it
computed. The qualitative claim survives: 1.9217 / 0.6657 = **2.887**, so
"three times worse than saying nothing" still reads correctly.
**Severity:** misleads a student
**Origin:** hand-written prose
**Fix:** interpolate `sweep[5]['valid']` into those three sentences instead of
typing a number.

---

### Claim 2 — `0.468` attributed to degree 2 is degree 1's score

**Verdict:** CONFIRMED
**Evidence:** from the same run, degree 2 = **0.4671**, degree 1 = **0.4681**.
Cell 53 says *"none of them beats the plain degree-2 model you already had at
0.468."* 0.468 is degree 1.
**Severity:** misleads a student
**Origin:** hand-written prose
**Fix:** print `sweep[2]['valid']`; do not transcribe.

---

### Claim 3 — `75.4%` accuracy at degree 5 matches no row set

**Verdict:** CONFIRMED
**Evidence:** `cross_val_score(pipeline(degree=5), X_train, y_train, cv=cv,
scoring="accuracy").mean()` = **0.75988** (76.0%, 712 training rows, 10 folds).
On the test set the same model scores **0.74860** (74.9%, 179 rows). Neither
rounds to 75.4%, and cell 38 names no row set at all.
**Severity:** misleads a student
**Origin:** hand-written prose
**Fix:** compute it, and say which rows.

---

### Claim 4 — "a bootstrap with replacement gives ~253 distinct rows"; it gives 306

**Verdict:** CONFIRMED
**Evidence:**

```
closed form 712*(1-(711/712)**400) = 306.1899
simulated mean over 2000 draws     = 306.2655
400*(1-1/e)                        = 252.8482
```

253 is the *n-from-n* answer applied to a 400-from-712 draw. The string
`~253 distinct rows` is in cell 23's `constraint` line.
**Severity:** misleads a student — the bullet exists to teach the closed form
**Origin:** hand-written prose
**Fix:** `306` (or state the formula and let the cell evaluate it).

---

### Claim 5 — "At 12% of 712 that is 85 passengers"; it is 76

**Verdict:** CONFIRMED
**Evidence:** ran `learning_curve` exactly as cell 35 does:

```
degree 1: sizes [76, 157, 237, 318, 398, 479, 559, 640]
  first held-out 2.8525  last held-out 0.4736  final gap 0.0781
degree 5: sizes [76, 157, 237, 318, 398, 479, 559, 640]
  first held-out 9.8894  last held-out 1.9114  final gap 1.6254
```

`train_sizes` are fractions of the largest training set *inside a fold*
(10-fold on 712 → 640/641), not of 712. `0.12 × 712 = 85.44`; `0.12 × 640 = 76`.
Cell 34's `catch` bullet says 85 and cell 35 asserts `n[-1] == 640` one line
later.
**Severity:** misleads a student
**Origin:** hand-written prose
**Fix:** say 76, or print `lc[1]["n"][0]`.

---

### Claim 6 — "where saga fits 4.90" does not reproduce (calibration claim)

**Verdict:** CONFIRMED — I agree with the independent verification, and my
reproduction used the notebook's own 712-row train split rather than the full
frame, which does not change the answer.
**Evidence:** the configuration the comment names (`penalty="l1"`, `C=0.001`),
fitted on `X_train`:

```
base-rate log-odds (712 train): -0.47503

degree 1 liblinear  C=0.001 intercept +0.0000  nnz   0 of  22  n_iter [0]
degree 1 saga       C=0.001 intercept -0.5018  nnz   0 of  22  n_iter [1]
degree 5 liblinear  C=0.001 intercept +0.0000  nnz   4 of 143  n_iter [9]
degree 5 saga       C=0.001 intercept -0.0001  nnz   5 of 143  n_iter [4588]
```

So: **liblinear +0.0000 exactly**, at both degrees — the claim's liblinear half
is right, and the mechanism (liblinear penalises its synthetic intercept
column) is right. **saga gives −0.0001 at degree 5 and −0.5018 at degree 1**,
never 4.90. I then swept for it: degrees 1–3 × C ∈ {1e-4 … 1e4}, both solvers,
32 fits. saga's intercept crosses 4.90 only between C = 10 (+3.66) and C = 100
(+5.44) at degree 1 — four to five orders of magnitude away from the C = 0.001
the comment names. No configuration of this pipeline produces 4.90 at C = 0.001.

The contrast the comment is *trying* to draw is real and is available one line
away: at degree 1, liblinear +0.0000 with 0 of 22 weights alive against saga
**−0.5018**, essentially the base-rate log-odds −0.4750, which is what an
unpenalised intercept should be once every slope is crushed. The Phase A report
reaches the same conclusion.
**Severity:** misleads a student — the number is offered as the evidence for the
lecture's own claim that the bias term is not penalised
**Origin:** hand-written prose (the string appears twice: cell 48's `left_open`
bullet and a comment inside cell 49's code)
**Fix:** replace 4.90 with −0.5018, state the degree, and print the base-rate
log-odds beside it.

---

### Claim 7 — four prose claims hard-coded inside `print()` calls

**Verdict:** CONFIRMED
**Evidence:** string search over the 29 code cells:

```
cell 27: print("=> the squared bias of the degree-1 model is under 0.01. There was "
cell 37: print("Right: still 1.6 apart and falling — starved of rows, not of capacity.")
cell 55: print(f"degree 2, no penalty      {sweep[2]['valid']:.3f}   <- still the best")
cell 74: print("=> about 0.013 of Brier score is left on the table, in total, for any")
```

None is recomputed. Measured today: degree-5 gap **1.6254** (so "1.6" is right);
squared bias at degree 1 = 0.1300 − 0.1218 = **0.0082** (under 0.01, right);
0.1339 − 0.1206 = **0.0133** (right); `<- still the best` rests on a 0.0010
margin (claim 13, a coin flip). A fifth instance of the same pattern is
`assert abs(noise - 0.121) < 0.001` in cell 19.
**Severity:** wrong but harmless *today* — all four survive any change to the
cells above them, which is the actual defect
**Origin:** generated code (authored string literals inside code cells)
**Fix:** compute each in the f-string.

---

### Claim 8 — "the bias band barely moves"; it grows 0.051 against variance's 0.062

**Verdict:** CONFIRMED
**Evidence:** ran the 800-fit decomposition (cell 24) verbatim — 200 draws of
400 rows without replacement, degrees 1, 2, 3, 5:

```
degree 1: total 0.1360 = bias2+noise 0.1300 + variance 0.0060   resid 2.78e-17
degree 2: total 0.1455 = bias2+noise 0.1342 + variance 0.0113   resid 0
degree 3: total 0.1674 = bias2+noise 0.1383 + variance 0.0291   resid 0
degree 5: total 0.2484 = bias2+noise 0.1806 + variance 0.0679   resid 0
```

variance grows by **0.0619**, bias²+noise by **0.0505** — the bottom band
accounts for **45.0%** of the total growth. The ×11.3 *ratio* claim in the same
annotation is correct; the claim about the *picture*, which is drawn in absolute
units by `ax.stackplot`, is not.
**Severity:** misleads a student — the annotation tells the reader what to see in
a plot that shows the opposite
**Origin:** hand-written prose — cell 28's `left_open` bullet, *"that the bias
band barely moves. The squared bias at degree 1 is under 0.01, so almost the
whole bottom band is the floor"*
**Fix:** say "both bands grow by about the same amount; the ratio and the
difference tell opposite stories", and name which one you are using.

---

### Claim 9 — `assert growth > 10` passes at 11.3

**Verdict:** CONFIRMED
**Evidence:** from the run above, growth = 0.0678595 / 0.0059864 = **11.336**, a
13% margin on a quantity estimated from 200 random draws at a fixed seed. The
assert is in cell 27 and is presented as a structural claim.
**Severity:** wrong but harmless — it passes today; it is a flake risk, not a
false statement
**Origin:** generated code
**Fix:** `assert growth > 5` and print the real figure.

---

### Claim 10 — prose says "the factor of ten"; the cell prints 22

**Verdict:** CONFIRMED
**Evidence:** cell 31's student bullet: *"The factor of ten is what connects the
picture back to the decomposition you just measured."* Cell 32 prints
`f"the gap grew by a factor of {g5 / g1:.0f}"`;
measured `g1 = 0.0729`, `g5 = 1.6360`, ratio **22.44** → prints **22**. The
assert `g5 > 10 * g1` is a floor, not the claim.
**Severity:** wrong but harmless
**Origin:** hand-written prose
**Fix:** "the factor of twenty-two", or derive the word from the number.

---

### Claim 11 — the noise floor and the final Brier are measured on different people

**Verdict:** CONFIRMED
**Evidence:** cell 19 opens `d = full.copy()` — all 891 passengers. Cell 74
prints the winner's Brier, computed on the 179 test passengers, beside it.
Re-derived:

```
all 891 : 133 cells, 102 multi, 858 people, floor 0.1206
train712: 122 cells,  97 multi, 685 people, floor 0.1218
test 179:  81 cells,  40 multi, 138 people, floor 0.1185
winner (degree 2, no penalty) test Brier = 0.1339
```

Headroom **0.0133** against the all-891 floor, **0.0121** against the
training-row floor. The comparison survives the mismatch — but §2.1 requires the
row sets to be named, and the notebook names neither.
**Severity:** misleads a student — and it is the same defect as claim 12, since
the 891-row floor is what reads the test labels
**Origin:** generated code (`d = full.copy()`)
**Fix:** compute the floor on `X_train`'s rows, and print both row sets in the
sentence.

---

### Claim 12 — "179 passengers, untouched since the split" is false

**Verdict:** CONFIRMED
**Evidence:** every read of `X_test`/`y_test`, by cell, with section:

```
cell 11 (§1) the split itself
cell 24 (§2) m.predict_proba(X_test) inside boot_fit; y_test as ground truth
             — 4 degrees × 200 draws = 800 reads
cell 63 (§5) log_loss(y_test, m.predict_proba(X_test)...) inside a 17-iteration loop
cell 69 (§5) log_loss(y_test, gs.predict_proba(X_test)...)  — 1 read
cell 70 (§6) "179 passengers, untouched since the split at the top of this notebook."
```

Plus cell 19's floor, which reads the test labels through `full`. The notebook's
own red-team question 1 is *"Count every read, not every write."* By its own
rule it fails its own checklist.
**Severity:** misleads a student
**Origin:** hand-written prose
**Fix:** replace the sentence with the list of reads; it is two lines.

---

### Claim 13 — "degree 2 is still the best" is inside its own noise

**Verdict:** CONFIRMED
**Evidence:** per-fold held-out log losses from the sweep:

```
degree 2 mean 0.4671, degree 1 mean 0.4681
margin                          0.00103
paired per-fold sd (ddof=1)     0.02883
degree 2 wins                   5 of 10 folds
```

A margin 28× smaller than the paired spread, and a 5–5 split of the folds.
§2.4: a result inside its own noise is not a result. The `<- still the best`
marker (cell 55) and the closing "the winner is a two-line model" argument both
rest on it.
**Severity:** misleads a student
**Origin:** hand-written prose (with the printed marker in generated code)
**Fix:** print the fold spread and say "degree 1 or 2" — which the notebook's own
assert, `min(...) in (1, 2)`, already allows.

---

### Claim 14 — the penalty table and the degree table use different fold counts

**Verdict:** CONFIRMED
**Evidence:** `cv` is `StratifiedKFold(n_splits=10)`, `cv5` is `n_splits=5`.
Training-set sizes per fit, measured on the 712 rows:

```
10-fold train sizes: [640, 641]
 5-fold train sizes: [569, 570]
```

Cell 55 prints `sweep[5]`, three `reg[...]` values and `sweep[2]` as five rows
of one column with no marking. Different training sizes, different held-out
sets.
**Severity:** misleads a student
**Origin:** generated code (the print block)
**Fix:** print the fold count on every row, or re-run the penalty sweep at `cv`.

---

### Claim 15 — "every penalty makes it respectable"; none reaches the anchor

**Verdict:** CONFIRMED
**Evidence:** ran the penalty sweep (cell 52) verbatim, `n_jobs=1`:

```
ridge    best C = 0.0001  log loss 0.72347  nnz 143 of 143   [ 10.5 s]
lasso    best C = 0.001   log loss 0.68106  nnz   5 of 143   [170.4 s]
elastic  best C = 0.3162  log loss 0.68115  nnz 140 of 143   [ 18.5 s]
anchor                    0.66572
```

Rounded to three decimals these are 0.724 / 0.681 / 0.681, exactly as the
Phase A report gives them.

All three are worse than predicting the base rate for everybody. Cell 53 says
they make the model "respectable". The repair is real (from 1.9217) and the
sentence overstates where it lands.

The second half of the claim — that on the *test* set the tuned models are worse
than unpenalised degree 5 on Brier and accuracy — also reproduces:

```
degree 5, no penalty              log loss 1.7266  Brier 0.1892  acc 74.9%
degree 5, ridge tuned (C=1e-4)    log loss 0.7790  Brier 0.2297  acc 64.8%
degree 5, lasso tuned (C=1e-3)    log loss 0.6691  Brier 0.2402  acc 64.8%
degree 1, sklearn defaults        log loss 0.4329  Brier 0.1343  acc 82.7%
degree 2, no penalty              log loss 0.4265  Brier 0.1339  acc 82.7%
```

**Severity:** misleads a student
**Origin:** hand-written prose
**Fix:** "drags it back to the neighbourhood of the anchor without reaching it",
computed against `constant_log_loss` in the cell.

---

### Claim 16 — two of the three tuned `C` values sit on the grid boundary

**Verdict:** CONFIRMED
**Evidence:** `Cs = np.logspace(-4, -0.5, 8)` =
`[1.0e-4, 3.16e-4, 1.0e-3, 3.16e-3, 1.0e-2, 3.16e-2, 1.0e-1, 3.16e-1]`.
Argmin index per penalty: **ridge 0** (first), **lasso 2** (interior),
**elastic 7** (last). Full score vectors:

```
ridge  : 0.72347 0.74693 0.77057 0.76646 0.74570 0.76729 0.83814 0.92282
lasso  : 0.68578 0.68203 0.68106 0.68214 0.68343 0.68411 0.68439 0.68447
elastic: 0.68306 0.68179 0.68123 0.68118 0.68116 0.68115 0.68115 0.68115
```

Only lasso has an interior minimum. Ridge's score rises across the grid (not
monotonically — it wobbles at indices 3–5 — but its best value is the smallest C
on offer and its worst is the largest). Elastic net is worse: its last four
values are `0.68115582 0.68115457 0.68114962 0.68114680` — a spread of
**9.0e-06** — so `argmin` lands on the final grid point by a margin four orders
of magnitude below the fold spread. Neither endpoint is a tuned hyperparameter,
and both are carried into cell 72 as "tuned".
**Severity:** misleads a student
**Origin:** generated code (no boundary check in cell 52) plus the docstring
prose that claims the minimum is inside the range
**Fix:** flag endpoint winners in the print, as lecture 2 already does.

---

### Claim 17 — the stated cost reason for truncating the grid does not reproduce

**Verdict:** CONFIRMED
**Evidence:** one 5-fold cross-validation of the lasso row at degree 5,
`n_jobs=1`, timed at three C values:

```
C = 0.01      37.5 s   mean log loss 0.6834
C = 0.1       35.2 s   mean log loss 0.6844
C = 1         24.6 s   mean log loss 0.6845
```

The module (docstring, cell 49 comment, cell 51 `left_open`) says **163 s at
C = 1 against 1.6 s at C = 0.01** — a 100× *increase* with C. Measured, the cost
is flat and C = 1 is the **fastest** of the three, a ratio of **0.66**, not 100.
Absolute wall times here are inflated by the machine load described at the top;
the ratio is not, and the ratio is what the claim is about.
**Severity:** misleads a student — it is offered as the reason for a design
decision the reader is asked to accept
**Origin:** hand-written prose
**Fix:** delete the cost justification or re-measure it; the honest reason is
claim 16's — the grid is narrower than the problem needs.

---

### Claim 18 — "the assert on line 3" is on line 17

**Verdict:** CONFIRMED
**Evidence:** the rendered `engineer` cell (cell 9), numbered:

```
  1| # --- the four engineered columns, exactly as in the build session ---
  2| def engineer(d):
  3|     d = d.copy()
 ...
 16| # Keep this line in view. It is the third diagnosis, four sections from now.
 17| assert (full["SibSp"] + full["Parch"] + 1 - full["FamilySize"]).abs().max() == 0
```

Cell 8's `left_open` bullet: *"that the assert on line 3 is not a health check"*.
Line 3 is `d = d.copy()`.
**Severity:** cosmetic
**Origin:** hand-written prose
**Fix:** "the assert below `engineer`", or drop the line number.

---

### Claim 19 — "four sections from now" is two sections

**Verdict:** CONFIRMED
**Evidence:** the `##` section headings, by cell index:

```
cell  1: ## 1 · Setup and where we left off
cell 17: ## 2 · Thread 3 — the bias-variance decomposition
cell 38: ## 3 · Diagnose — three faults, one shape
cell 47: ## 4 · Fix — ...
cell 61: ## 5 · The worked assistant failure
cell 70: ## 6 · Re-measure — the test set, once
```

The phrase is in cell 8 (markdown) and cell 9 (code comment), both inside
section 1. "The third diagnosis" is in section 3 (cell 41). Two sections.
**Severity:** cosmetic
**Origin:** hand-written prose (and the same words as a code comment)
**Fix:** "two sections from now".

---

### Claim 20 — lecture 2 contains no such text, and the clause is six words not eleven

**Verdict:** CONFIRMED
**Evidence:** string search across `notebooks/lecture-01..06.ipynb` for
`Split before anything`, `preprocessing lives inside`, `Fixed random seed`,
`Nothing derived from the test set`:

```
lecture-06 cell 78  — all four (this notebook, the one asking the reader to extend it)
lecture-03 cell 28  — "The standing constraint from the previous lecture applies unchanged..."
lecture-03 cell 29  — "...the standing constraint from Lecture 2"
(no hit in lectures 01, 02, 04, 05)
```

So lecture 3 refers to a block lecture 2 never writes, and lecture 6 asks the
reader to *"Add one clause to what you wrote down in Lecture 2"*. The added
clause is *including the choice of any hyperparameter* = **6** words; cell 78
says "Eleven added words."
**Severity:** misleads a student — §7 requires instructions the reader can
actually carry out, and this one refers to something they were never given
**Origin:** hand-written prose
**Fix:** write the constraint out in lecture 2; and count the words.

---

### Claim 21 — lecture 2 taught the grid-boundary check and lecture 6 breaks it

**Verdict:** CONFIRMED
**Evidence:** `notebooks/lecture-02.ipynb` cell 22:

```
> **check** · detect whether the winner sits on the EDGE of the grid and say so
  — an optimum at the boundary means the optimum may lie outside it
* **How you would catch it:** ... If the largest value wins, the search was too small.
```

and cell 23 prints `"⚠ the winner sits on the EDGE of the grid — the optimum may
lie ..."`. Searching all 79 cells of lecture 6 for `EDGE`, `edge`, `boundary`,
`endpoint`: **zero hits**, while two of its three tuned values are at endpoints
(claim 16).
**Severity:** misleads a student
**Origin:** notebook structure (a rule taught earlier, unenforced later)
**Fix:** port lecture 2's four-line endpoint check into cell 52.

---

### Claim 22 — "rank 23 of 29 columns" describes a matrix this notebook never builds

**Verdict:** CONFIRMED
**Evidence:**

```
prep(1) cols 22   with intercept 23   np.linalg.matrix_rank = 23
```

The degree-1 design matrix of *this* notebook is 22 columns, 23 with the
intercept, and **full rank** — not rank 23 of 29. The figure is lecture 5's
unrepaired encoder. It appears twice in lecture 6, in cell 39's `catch` bullet
and cell 41's prose, both times as the second symptom of the degree-5 failure.
**Severity:** misleads a student
**Origin:** hand-written prose
**Fix:** either drop it or say explicitly that it is lecture 5's matrix.

---

### Claim 23 — diagnosis 3 is demonstrated on a matrix no pipeline fits

**Verdict:** CONFIRMED
**Evidence:** cell 43 builds `Z` from
`["SibSp", "Parch", "FamilySize", "Age", "Fare"]` and finds the singularity:

```
eigenvalues of XᵀX: [1.81e-12 4.29e+02 5.43e+02 7.90e+02 1.80e+03]
condition number at α = 0: 9.231e+14
```

But `NUM = ["Age", "Fare", "SibSp", "Parch"]` from cell 11 onward, so
`FamilySize` enters no pipeline in the notebook, and the degree-1 design is full
rank (claim 22). Cell 41 introduces this as *"Diagnosis 3 is Thread 1
returning"* — presenting as a cause of the 1.92 something cell 11 had already
removed. The notebook says this obliquely once, in cell 10's `left_open`
("that is the whole of the third repair"), and never in section 3 where the
conclusion is drawn.
**Severity:** misleads a student
**Origin:** hand-written prose
**Fix:** one sentence in section 3 saying this matrix is the fault that *would*
have been there.

---

### Claim 24 — the assistant failure is announced four times before it runs

**Verdict:** CONFIRMED
**Evidence:** every `⚠` in the notebook, in order:

```
cell  0 (md)   "Cells marked **⚠ read before running** contain a defect on purpose"
cell 61 (md)   "> ⚠ **read before running**"   [section-5 heading block]
cell 61 (md)   "...It runs, it warns about nothing, and it prints a number a reader will quote."
cell 62 (md)   "> **Prompt · ⚠ what the assistant returns**"  + a three-bullet
               annotation whose `left_open` gives away the whole answer
cell 63 (code) "# ⚠ WRONG — this is the failure, not the fix"
```

Four announcements before the cell, and a fifth on its first line. §8.1 is
written about exactly this; §8.2 says the best trap is unlabelled.
**Severity:** misleads a student — it destroys the only measurement the section
exists to take
**Origin:** notebook structure
**Fix:** move the ⚠ and the annotation into the *next* section, after the reader
has written the number down.

---

### Claim 25 — 29 of 29 prompt boxes carry the full three-bullet annotation

**Verdict:** CONFIRMED
**Evidence:** counted programmatically over the 79 cells — markdown cells
beginning `> **Prompt`, checking for all three of `Left open`, `The usual
student version`, `How you would catch it`:

```
boxes: 29    full three-bullet: 29    boxes missing a bullet: []
code cells: 29
```

§6.1's budget is five to eight, never more than ten.
**Severity:** misleads a student — §6.1's evidence is that readers stop reading
the template around cell 30, which is where this notebook's defect lives
**Origin:** notebook structure
**Fix:** keep the full form on 7 cells; short specification box on the rest.

---

### Claim 26 — invented "usual student version" bullets

**Verdict:** CONFIRMED for the headline, **but its supporting list is wrong on
four of six items** — flagging this because a rebuild that acts on the list will
delete bullets that are already correct.
**Evidence:** the three bullets it names do exist and do name neither a library
default nor a recorded failure:

```
cell  6: "pickling the frame at the end of the last notebook"
cell 23: "reusing one fitted pipeline across the 200 draws"
cell 71: "adding a sixth candidate after seeing the table"
```

The claim then says *"the genuinely real defaults available in the same cells go
unmentioned"* and lists six. Four of the six are in fact mentioned, and three of
those in the very bullets §6.2 asks for:

```
return_train_score=False  -> cell 15 constraint: "it is off by default and it is
                             the whole experiment"
rng.choice(replace=True)  -> cell 23 constraint: "draw WITHOUT replacement — a
                             bootstrap with replacement gives each fit ~253..."
learning_curve(shuffle=)  -> cell 33 prose + cell 34 constraint + cell 34's own
                             student bullet ("forgetting shuffle")
SGDClassifier warm_start  -> cell 57 student bullet ("`max_iter=1` without
                             `warm_start`") and catch bullet ("`penalty=None` is
                             deliberate. If you leave the default L2 on...")
```

Genuinely unmentioned, confirmed by search: `observed=` appears only as
`observed=True` in cell 19 with no note of the default, `dropna` appears
nowhere, and `scoring=None`/the accuracy fallback appears nowhere (cell 69 sets
`scoring="neg_log_loss"` without saying why).
**Severity:** wrong but harmless (the headline is a real §6.2 breach; the
supporting list would cause harm if acted on literally)
**Origin:** hand-written prose
**Fix:** cut the three invented bullets; add the two genuinely missing defaults
(`groupby(observed=False, dropna=True)`, `GridSearchCV(scoring=None)`).

---

### Claim 27 — no ⏱ note states a CPU

**Verdict:** CONFIRMED for the checkable part; the specific "20–60× faster"
multiples are UNVERIFIABLE on this machine.
**Evidence:** every `⏱` in the notebook:

```
cell 14/15  about 15 seconds   (degree sweep)
cell 22/23  about two minutes  (decomposition)
cell 33/34  about 30 seconds   (learning curves)
cell 50/51  about two minutes  (penalty sweep)
cell 56/57  about 20 seconds   (early stopping)
cell 64/65  about two minutes  (tuning trap)
```

Six markers; **none names a machine, a core count or `n_jobs`**, and "about two
minutes" appears three times. §7.1 requires the CPU figure. That much is
settled. The claimed idle-M4-Max timings I could not reproduce, because the box
was at load 150–180 for the whole triage: the six-fit convergence cell took
**4.7 s** here (Phase A: 1.0 s), the degree sweep **7.9 s** (Phase A: 6.0 s),
and the 800-fit decomposition well over ten minutes (Phase A: 21.2 s). The
direction of the claim — that the notebook's figures are far too pessimistic for
a modern laptop — is supported by the two cells I could time cleanly; the
multiples are not mine to confirm.
**Severity:** misleads a student
**Origin:** hand-written prose
**Fix:** two numbers per marker — measured-on-what, and Colab-scale.

---

### Claim 28 — a timing measured under load is worthless

**Verdict:** CONFIRMED, and independently reproduced — though note the report
itself says this is **not a defect in the notebook**, so it should not be
counted as one.
**Evidence:** I hit this without trying. Same code, same data, same machine:

```
5-fold CV of the lasso row at degree 5, n_jobs=1, load ~160:   37.5 s
the same row, Phase A, idle M4 Max, NJ=4:                       ~4.9 s (39.3/8)
800-fit decomposition, load ~160:                             > 10 min
the same cell, Phase A, idle:                                    21.2 s
convergence cell (6 fits), load ~160:                            4.7 s
the same cell, Phase A, idle:                                    1.0 s
```

`uptime` reported load averages of 153–180 on a 16-core box throughout. This is
the argument for `NJ = 4` rather than `n_jobs=-1`, and it belongs in the
notebook's timing note.
**Severity:** cosmetic — it is a recommendation, not a defect
**Origin:** n/a (a note about method)
**Fix:** one sentence in the timing note; none needed in the cells.

---

### Claim 29 — three untimed multi-fit cells

**Verdict:** CONFIRMED
**Evidence:** the ⏱ scan above lists markers at cells 14/15, 22/23, 33/34,
50/51, 56/57, 64/65. The three multi-fit cells with **no** marker are:

```
cell 40  the convergence table   — 6 fits at max_iter=4000   (4.7 s measured, loaded)
cell 63  the assistant-failure loop — 17 fits at degree 3
cell 69  GridSearchCV            — 17 candidates × 10 folds + refit = 171 fits
```

The notebook times its 6-second degree sweep and not its 171-fit grid search.
**Severity:** wrong but harmless
**Origin:** notebook structure
**Fix:** add ⏱ to cell 69 at minimum.

---

### Claim 30 — no exercise lists a re-run order

**Verdict:** CONFIRMED, with one qualification
**Evidence:** searching all 79 cells for `xercise`, `re-run`, `rerun`,
`change the seed`, `neighbour` returns exactly two hits, both about running code
on someone else's notebook (cells 75 and 76). Nothing anywhere states which
cells depend on `RANDOM_STATE`, and they are three cells apart: cell 3
(`RANDOM_STATE = 42`), cell 11 (the split), cell 13 (`cv` and the anchor).
**Qualification:** §7.2 governs *an exercise that requires re-running cells*, and
neither of this notebook's two reader tasks explicitly asks for one — so the
rule is only weakly triggered. The claim is factually true; its severity is
lower than the report implies.
**Severity:** wrong but harmless
**Origin:** notebook structure
**Fix:** list the seed-dependent cells in order where the reader is asked to
change anything.

---

### Claim 31 — `p` is rebound from a float to a 179-vector

**Verdict:** CONFIRMED
**Evidence:**

```
cell 13: p = y_train.mean()                    # float, defines constant_log_loss
cell 72: p = m.predict_proba(X_test)[:, 1]     # ndarray of 179, inside the loop
```

**Severity:** wrong but harmless — `constant_log_loss` is already bound by then
**Origin:** generated code
**Fix:** call the second one `pr`.

---

### Claim 32 — `best` is rebound from an epoch index to a log loss

**Verdict:** CONFIRMED
**Evidence:**

```
cell 58: best = int(np.argmin(valid_curve))    # epoch index, 0..499
cell 60: ax.axvline(best, ls="--", color="#14663a")
cell 74: best      = final[winner]["log_loss"]  # float
```

Measured value after cell 74: **0.4265**. Re-running cell 60 after cell 74
therefore draws the dashed "chosen epoch" line at x = 0.43 on a 0–500 axis, with
no error.
**Severity:** wrong but harmless — it only bites on an out-of-order re-run,
which §4.3 says must be named and is not
**Origin:** generated code
**Fix:** `best_epoch` in cell 58.

---

### Claim 33 — `A` and `B` mean two different things

**Verdict:** CONFIRMED, and the report's own characterisation ("harmless to
execution") is right
**Evidence:**

```
cell 58 (module level):  A, B, y_a, y_b = train_test_split(X_train, y_train, test_size=0.25, ...)
cell 66 (inside one_seed): A, B, ya, yb = train_test_split(X, y, test_size=0.2, random_state=seed, ...)
```

The second pair is function-local, so it shadows rather than clobbers; a
75/25 split of the training rows and an 80/20 split of the whole frame share two
names.
**Severity:** cosmetic
**Origin:** generated code
**Fix:** throwaway names inside `one_seed`.

---

### Claim 34 — "examinable" appears three times and never as a section marker

**Verdict:** CONFIRMED
**Evidence:** every occurrence, case-insensitive, over all 79 cells:

```
cell 2 (markdown, a prompt annotation): "not examinable, and it is here because a
                                         version mismatch produces a confusing error"
cell 3 (code comment):  "# Not examinable: this is engineering hygiene, not machine learning."
cell 5 (code comment):  "# Not examinable. A free Colab CPU runtime has two cores..."
```

Three, all in section 1, none a section marker. Sections 2–7 — including the
whole bias-variance derivation — carry no marking.
**Severity:** wrong but harmless
**Origin:** notebook structure
**Fix:** one marker per section heading.

---

## Spot-check of the report's own `Checked and clean` block

I re-derived these too, because a triage that only checks the accusations is
half a triage. All reproduce:

```
891 passengers, 342 survived; titles {'Mr':517,'Miss':185,'Mrs':126,'Master':40,'Rare':23}
split 712/179, rates 0.3834 / 0.3855
anchor 0.6657168967694276  (0.666)
columns: 22 at degree 1, 143 at degree 5
groups on all 891: 133 cells / 102 multi / 858 people, floor 0.1206  (0.121)
eigenvalues [1.81e-12 4.29e+02 5.43e+02 7.90e+02 1.80e+03], cond 9.231e+14
condition number at α=1: 1798.3
convergence: degrees 1-3 converge (74/256/935 iters), 4-6 do not (4000)
largest |θ|: 4.87 5.43 6.32 18.70 11.07 3.98;  ‖θ‖₂: 10.4 11.1 15.0 32.1 37.5 15.1
identity residuals: 2.78e-17 at degree 1, exactly 0.0 at degrees 2, 3, 5
final table: 1.7266 / 0.7790 / 0.6691 / 0.4329 / 0.4265
imputed Age 137 train / 40 test / 177 total; 2 missing Embarked
majority-class accuracy on the test set 61.45%
§5.1/5.2: 0 violations over 50 markdown cells;  §3.1: 0 fenced blocks in markdown
```

Two rounding notes, neither a defect claim: the rewrite script's cell 17 quotes
the third eigenvalue as `5.44e+02` where `np.array2string(precision=2)` prints
**5.43e+02**, and its cell 16 table quotes `‖θ‖₂ = 37.6` at degree 5 where the
fit gives **37.5**.

The report's `Not checked` items stand as it left them: I did not run the full
8-seed tuning trap either, and nobody has looked at the two rendered figures or
re-measured anything on a Colab runtime.

---

## Summary

```
confirmed: 34   false positive: 0   unverifiable: 0
```

**34 of 34 is the shape of a rubber stamp, so here is why it is not one.** The
brief's default is FALSE POSITIVE, and I went looking. What I found is that the
lecture-6 Phase A report is not a list of impressions: it marks almost every
item *Checked*, gives the number it measured, and names the row set. I
re-derived every one of those numbers independently from
`notebooks/datasets/titanic/train.csv` rather than reading its arithmetic, and
they came back to the printed precision — including the ones it would have been
easiest to fudge (the 800-fit decomposition, the 144-fit penalty sweep, the
learning-curve sizes, the intercept sweep). Where I could not reproduce a figure
I said so (claim 27's timing multiples). Where the claim contains a mistake I
said so (claim 26). Where it overstates a rule I said so (claim 30). Where it
concedes it is not a defect I said so (claim 28). If this report is wrong it is
wrong in a way that survived running the code, which is a different failure from
the one the brief is guarding against.

Three tiny disagreements with Phase A, all in its favour and none a defect
claim: `‖θ‖₂` at degree 5 is 37.5 where the rewrite script says 37.6; the third
eigenvalue prints 5.43e+02 where the script says 5.44e+02; and elastic net's
"minimum at the last grid value" is a tie to 5e-5 across the last four values
rather than a clean minimum — which strengthens claim 16 rather than weakening
it.

No claim was refuted outright. Two carry corrections:

- **claim 26** — headline confirmed, but four of the six "unmentioned real
  defaults" it lists *are* mentioned in the notebook, two of them in exactly the
  §6.2 bullets the claim says are missing. Acting on the list as written would
  delete correct material.
- **claim 27** — the "no CPU is named" half is settled; the "20–60× faster on an
  idle M4 Max" multiples are not reproducible on a machine at load 150–180 and
  are not confirmed by me.

One claim (**28**) is a methodological note the report itself labels *not a
defect in the notebook*; it is counted as confirmed above because the fact is
true and I reproduced it, but it should not be counted against the notebook.

```
of the confirmed, 21 mislead a student
  1 2 3 4 5 6 8 11 12 13 14 15 16 17 20 21 22 23 24 25 27
wrong but harmless: 7 9 10 26 29 30 31 32 34   (9)
cosmetic:           18 19 28 33                (4)

origin split — prose: 20   code: 8   structure: 6
  prose:     1 2 3 4 5 6 8 10 12 13 15 17 18 19 20 22 23 26 27 (19) + 28 (n/a, filed as prose)
  code:      7 9 11 14 16 31 32 33 (8)
  structure: 21 24 25 29 30 34 (6)
```

**Duplicates — the same underlying defect counted more than once:**

- **11 and 12** — one defect. The all-891 noise floor (`d = full.copy()`) *is*
  one of the reads of the test set that falsifies "untouched"; fixing cell 19 to
  use the training rows fixes both.
- **16 and 21** — one defect seen twice. 16 is "two tuned values are at
  endpoints"; 21 is "and lecture 2 already taught you to check for that". One
  four-line fix in cell 52 closes both.
- **13 and the fourth item of 7** — `<- still the best` is a hard-coded string
  (7) asserting a 0.0010 margin (13). Same line of code.
- **8 and 10** — both are ratio-versus-difference confusions in the same
  argument (×11.3 quoted as if it described a plot in absolute units; "factor of
  ten" quoted where the cell prints 22). Different cells, one habit.
- **18, 19, 22 and 20** are all §3.3 cross-reference failures and are
  independent of each other; they are *not* duplicates, but they are one class
  and would be caught by one pass of "count every reference".
- **1 and 15** overlap in the source: cell 53's paragraph carries both the stale
  1.957 and the "makes it respectable" overstatement, in adjacent sentences.
