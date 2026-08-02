# Triage — lecture 5

Claims triaged: the 16 numbered items of the *Defects found in the current
notebook* section of `tools/prompts/lecture_05.md`, split into 21 where an item
carries independently checkable sub-claims (item 7 → 7a–7d, item 8 → 8a–8b,
item 13 → 13a–13b).

**Environment.** Python 3.13.5, scikit-learn 1.7.2, numpy 2.3.5, pandas 2.3.3,
Apple M4 Max (`Mac16,6`, 16 cores). Data: `notebooks/datasets/titanic/train.csv`
(cached, not re-downloaded). Verification script:
`/private/tmp/claude-501/-Users-fabriziosilvestri-Documents-Codice-AIML-Course/f3a0270e-32b0-470f-a72d-bbb10b6a91ef/scratchpad/t05/run05.py`,
which reproduces cells 3–88 of the notebook line for line from the cached CSV;
output in `.../t05/out.txt`. Timings:
`.../t05/time05.py` → `.../t05/time.txt`.

**Stated once, not repeated per claim (per the brief):** all 29 code cells have
`outputs: []` and `execution_count: null`, so no prose figure in this notebook
can be reconciled against a stored output. Every figure below was re-derived by
executing the notebook's own logic.

**Note on timing measurement.** The machine was carrying a load average of
~150 on 16 cores while other agents ran, so wall-clock is worthless here. I
therefore measured **CPU-seconds** with `resource.getrusage` (self + children)
at `n_jobs=1` and `OMP_NUM_THREADS=1`, which is load-independent and is the
honest proxy for a single slow Colab core. Those are the numbers quoted in
claim 4.

---

### Claim 1 — `Deck_T` has no coefficient in this model; the largest weight belongs to `Deck_infrequent_sklearn`

**Verdict:** CONFIRMED

**Evidence:** re-running cells 30/46 against the cached CSV:

```
CLAIM1 'Deck_T' in names -> False
CLAIM1 infrequent_categories_ -> [None, None, None, None, array(['T'], dtype=object)]
CLAIM1 top6:
   Deck_infrequent_sklearn    -5.678  odds x0.003
   Title_Master               +3.484  odds x32.596
   Sex_female                 +3.308  odds x27.325
   Sex_male                   -2.833  odds x0.059
   Title_Miss                 -2.806  odds x0.060
   Title_Mrs                  -2.020  odds x0.133
```

Cell 47 says *"`Deck_T` has the largest weight in the model"*; cell 45's
`left_open` says *"Deck_T is fitted to one passenger"*; cell 18's `student`
bullet says `Deck_T` is *"precisely that failure, surviving the collapse"*. The
name `Deck_T` appears in none of cell 46's output. The pedagogical point
survives — the bucket is exactly `['T']`, one passenger, and it does carry the
largest weight — but the reader is told to look for a string that is not there.

A fourth cell compounds it, and the Phase A report does not list it: cell 48's
`left_open` reads *"`min_frequency=2` on the encoder would fold it away; the
notebook leaves it visible"*. Cell 30 already sets `min_frequency=2`. The
notebook states the counterfactual as though it had not taken it.

**Severity:** misleads a student
**Origin:** hand-written prose
**Fix:** rename the referent to `Deck_infrequent_sklearn` in cells 18, 45 and
47, print `infrequent_categories_` in cell 49, and delete the false
counterfactual in cell 48.

---

### Claim 2 — the condition number is quoted as 2.6e+16 in three places and is 3.16e+16

**Verdict:** CONFIRMED

**Evidence:**

```
CLAIM2 columns 29 rank 23 cond 3.1591e+16 deficiency 6
CLAIM2 six smallest singular values: [1.462e-14 1.013e-14 6.522e-15 3.779e-15 3.183e-15 1.721e-15]
```

The three sites are real: cell 51 `student` (*"the condition number is
2.6e+16"*), cell 61 `catch` (*"2.6e+16 to under a hundred"*), and cell 62 as a
literal inside a `print` — `print("from 2.6e+16 to under a hundred.")` — sitting
directly beneath a line that computes and prints the true value. The
quantity also has no significant digits: it is a ratio against six singular
values in the range 1.7e-15 to 1.5e-14, i.e. floating-point zero, and will
differ between BLAS builds.

One qualification on the report's wording: it says *"the notebook's stored
output and its own prose disagree by 20%"*. There are no stored outputs — the
disagreement is between the printed value at run time and the hard-coded string
two lines below it. The defect is real; that sentence describes it wrongly.

**Severity:** misleads a student
**Origin:** hand-written prose (and one hard-coded literal in generated code, cell 62)
**Fix:** say "singular" or quote the rank; if a number is wanted, interpolate
the computed one rather than hard-coding it.

---

### Claim 3 — a prompt box is duplicated verbatim (cells 74 and 75)

**Verdict:** CONFIRMED

**Evidence:** `grep -n 'label="the policy you can actually staff"'
tools/notebooks/lecture_05.py` → lines **946** and **971**, two `prompt(...)`
calls (opening at 945 and 970) for the single code cell 76. The rendered cells
are not byte-identical; the diff is confined to the last two bullets:

```
- ... and gives the WRONG 80 the moment the cut-off is not monotone in risk.
    The truncation also hides the fact that a constraint bound at all.
+ ... and the WRONG 80 the moment the cut-off is not monotone in risk.
- ... is not a capacity rule, it is a threshold rule that got lucky — and it wastes six crew.
+ ... is a threshold rule that got lucky, and it wastes six crew.
```

Same `input`, `output`, `constraint` and `check`, verbatim. Programmatic count:
**30 prompt boxes for 29 code cells**.

**Severity:** wrong but harmless
**Origin:** notebook structure
**Fix:** delete one of the two `prompt(...)` calls in `tools/notebooks/lecture_05.py`.

---

### Claim 4 — the ⏱ markers overstate by 10× to 100×

**Verdict:** CONFIRMED

**Evidence:** CPU-seconds, single-threaded, load-independent:

| Marker | Notebook claims | Measured (CPU-s, 1 thread) |
|---|---|---|
| cell 37 / 38 (20 seeds) | "about two minutes" | **0.94** |
| cell 41 / 42 (cross_validate ×3) | "about 20 seconds" | **0.14** |
| cell 65 / 66 (calibration) | "about 15 seconds" | **0.08** |
| cell 78 / 79 (degree sweep) | "about 30 seconds" | **9.76** |
| cell 85 (convergence, no marker) | — | **0.96** |

The 20-seed cell is the extreme case: 128× overstated, and *faster*
single-threaded than the Phase A report's own 7.2 s figure because at `n_jobs=4`
the cell is dominated by joblib process spawn, not arithmetic. Cells 41/42 and
65/66 are overstated by 140× and 190×. Only the degree sweep is in the right
neighbourhood — at 9.8 s CPU on an M4 Max core, a free Colab core could
plausibly reach 30 s, so that marker is defensible and the other three are not.

Cell 85, which carries no marker, is slower than two cells that do.

I did not measure on Colab. The direction is not in doubt: a 128× gap is not a
hardware difference.

**Severity:** misleads a student
**Origin:** hand-written prose
**Fix:** delete the ⏱ on cells 37/38, 41/42 and 65/66; keep one on 78/79 with a
real number.

---

### Claim 5 — the repair's 0.028 improvement is one fold out of ten

**Verdict:** CONFIRMED

**Evidence:** both models scored on the *same* `cv` object
(`StratifiedKFold(10, shuffle=True, random_state=42)`) over the same 712
training rows, so §2.1 is satisfied and the rows match:

```
CLAIM5 v1 folds [0.422 0.43  1.134 0.501 0.337 0.39  0.457 0.471 0.402 0.417]
CLAIM5 v2 folds [0.421 0.43  0.843 0.5   0.338 0.389 0.47  0.471 0.402 0.417]
CLAIM5 diff    [ 0.001  0.     0.291  0.001 -0.001  0.001 -0.013  0.001  0.001 -0.  ]
CLAIM5 drop fold3: v1 0.4254 v2 0.4264 delta +0.0011
CLAIM5 medians 0.4263 0.4257  stds 0.2169 0.1325
CLAIM5 ll_v2 0.4681 anchor 0.6657
```

Nine folds agree to within 0.013; the whole mean improvement is fold 3. Remove
it and the repaired model is 0.0011 **worse**. Medians are a tie. The fold
spreads (0.217 and 0.133) are 5–8× the effect. Cell 64 prints the mean only,
under the comment *"And it scored better, which was not the reason for doing
it."* The correct structural defence is present in cell 63's `left_open`, but
the headline sentence in the code cell asserts an improvement that is inside
the noise, with no spread beside it — §2.4.

**Severity:** misleads a student
**Origin:** hand-written prose (the comment on cell 64 line 1)
**Fix:** print the per-fold vector for both models and the elementwise
difference, and say the effect is one fold.

---

### Claim 6 — the `FamilySize` trap is announced four times before it fires, and cell 18 contradicts cell 20

**Verdict:** CONFIRMED

**Evidence:** by enumeration of the notebook between the feature (cell 19) and
the reveal (cell 55), all four sites verbatim:

- cell 18 `left_open`: *"one of these four lines manufactures an exact linear dependence and takes twenty minutes to diagnose in section 9. The markdown under this cell says so and does not say which."*
- cell 20, a ⚠ blockquote containing the fenced line `d["FamilySize"] = d["SibSp"] + d["Parch"] + 1` and *"Look at this line and remember it."*
- cell 22 `left_open`: *"that FamilySize, SibSp and Parch are all in NUM together. That is the trap, sitting in plain sight in the first line."*
- cell 29 `left_open`: *"some of those 28 are redundant by construction."*

Cell 20 is the markdown under cell 18's cell, and it does name the line, in a
fenced `python` block — so cell 18's *"does not say which"* is false about the
cell directly beneath it. §8.1 and §8.2 are both violated: the trap is not
merely flagged four times, it is solved in cells 20 and 22, thirty-five cells
before it fires.

**Severity:** misleads a student (it destroys the exercise, which is the point of §8.1/8.2)
**Origin:** hand-written prose
**Fix:** delete cell 20 entirely and the trap-naming clauses from cells 18, 22
and 29; move the reveal to cell 55 as the script's *Staging* section specifies.

---

### Claim 7a — cell 12's *"the anchor computed three cells down"* does not resolve, and the anchor is not built out of cell 13

**Verdict:** CONFIRMED (with a correction to the report's own arithmetic)

**Evidence:** code-cell indices are
`[3, 5, 8, 10, 13, 16, 19, 23, 26, 30, ...]`. `constant_log_loss` is computed in
cell **26**, which is **14 cells** and **5 code cells** after cell 12 (4 code
cells after cell 13, the cell the box belongs to). "Three cells down" is wrong
under every counting.

The Phase A report says *"fourteen cells later and thirteen code cells later"*.
Fourteen is right; **thirteen code cells is wrong** — there are only 29 code
cells in the whole notebook and only 5 of them lie in that span. The report is
right about the defect and wrong about the size of it.

Second half of the claim, re-derived:

```
CLAIM7a y_train.mean 0.383427  full.mean 0.383838  equal? False
```

Cell 13 prints `full["Survived"].mean()` = 0.38384; cell 26 computes the anchor
from `y_train.mean()` = 0.38343, a different set of rows. So *"built out of this
one"* is also false.

**Severity:** misleads a student (a reader who follows the pointer lands on cell 15's cross-tab)
**Origin:** hand-written prose
**Fix:** say "the anchor in section 5, computed from the training rows only —
0.3834, not this 0.3838".

---

### Claim 7b — *"Requirement 3 … got its own row in the table"*: there is no table of requirements

**Verdict:** CONFIRMED

**Evidence:** regex search for `[Rr]equirement\s*\d?` across all 89 cells
returns hits in cells **24, 65, 73, 76, 77** (plus generic uses of the word in
81 and 83). No cell anywhere enumerates or numbers the requirements. Search for
`\btable\b` returns cells 15, 27, 45, 57, 59, 61, 63, 73 — every one of them is
the sex×class cross-tab or the coefficient table; none is a requirements table.
The brief in cell 1 states two requirements in prose and never numbers them.

A student reading alone can resolve neither the numbering nor the table.

**Severity:** misleads a student (the whole of §12b hangs on "requirement 3")
**Origin:** hand-written prose
**Fix:** number the three requirements explicitly in cell 1, or drop the
reference to a table that lives only in the slide deck.

---

### Claim 7c — cell 77's *"Lecture 8 is about exactly that"* does not resolve

**Verdict:** CONFIRMED

**Evidence:** `LECTURES.md:108` — *"Lecture 8 — Retrain it and watch it
change"*; the notebook header of `lecture-08.ipynb` reads *"Géron, Chapters 5 &
6 · Mathematical thread: impurity, and why averaging reduces variance"*. Word
counts over the full text of `lecture-08.ipynb`:

```
'drift': 0    'distribution shift': 0    'passenger mix': 0    'population': 0
'bagging': 15   'bootstrap': 15   'Gini': 17   'entropy': 17   'variance': 27
```

The sentence promises material on the input distribution moving under a fixed
model. Lecture 8 covers variance under resampling — adjacent, and not what the
sentence sends the reader to find.

**Severity:** wrong but harmless
**Origin:** hand-written prose
**Fix:** drop the forward reference, or point at the lecture that actually
covers distribution shift.

---

### Claim 7d — cell 51's *"The next two cells answer it"* is off by a cell

**Verdict:** FALSE POSITIVE

**Evidence:** cell 51's `left_open` reads *"where the six come from. The next
two cells answer it: five one-hot blocks and one feature engineered by hand."*
The next two cells are 52 and 53. Cell 52 prints `rank deficiency: 6`. Cell 53
reads, in full:

> **Five of them are the one-hot blocks.** … Five categorical columns, five exact
> dependencies.
>
> **The sixth is the one you engineered yourself.**

Both halves of the promised answer are stated within the next two cells. Cell
55 *proves* the sixth, but the sentence says "answer", and it is answered. The
Phase A report's own reading (*"the five … are answered in the next cell
(53)"*) miscounts in the opposite direction — 53 is two cells on, not one.
The repair script itself keeps the same phrasing (*"Two cells answer it"*),
which is a further sign nobody found this reference broken on the page.

**Severity:** cosmetic
**Origin:** hand-written prose
**Fix:** none needed.

---

### Claim 8a — *"held-out accuracy falls by about seven points"* from degree 2 to degree 5

**Verdict:** CONFIRMED

**Evidence:** the degree sweep, re-derived:

```
  deg 1 cols  22 train 0.3952 valid 0.4681 acc 81.4574%
  deg 2 cols  32 train 0.3806 valid 0.4671 acc 82.1674%
  deg 3 cols  52 train 0.3554 valid 0.5381 acc 79.6401%
  deg 4 cols  87 train 0.3104 valid 1.3903 acc 77.3944%
  deg 5 cols 143 train 0.2857 valid 1.9217 acc 75.9879%
  deg 6 cols 227 train 0.2839 valid 1.8360 acc 76.6882%
CLAIM8a acc deg2 82.1674% deg5 75.9879% drop 6.18 pts
CLAIM8a ll ratio 4.114
```

**6.18 points**, not seven; round-half-up gives 6. The claim appears twice, in
cell 81's `left_open` (*"Accuracy falls seven points from degree 2 to 5"*) and
cell 83's prose (*"about seven points"*). The log-loss half of the same
sentence — *"multiplied by more than four"* — checks out at ×4.114.

**Severity:** wrong but harmless
**Origin:** hand-written prose
**Fix:** "about six points", in both cells.

---

### Claim 8b — cell 35's *"getting a monotone cost curve"* is the wrong tell

**Verdict:** CONFIRMED

**Evidence:** I ran the inverted rule (`flag = prob > t`) across the same
97-point grid on the same `p_oof`:

```
CLAIM8b inverted: min 371 at t=0.02; first 371 last 4401
CLAIM8b inverted monotone non-decreasing? False  #decreases 19
CLAIM8b correct rule monotone? False  min 210
CELL70 #grid points achieving min: 1
```

The inverted curve rises overall but decreases at **19 of its 96 steps** — it
is not monotone. Its minimum is indeed the first grid point, so the second half
of cell 35's diagnosis is right. A student who checks monotonicity, as
instructed, concludes their inverted curve is fine. Separately: exactly one
grid point achieves the correct minimum of 210, so `min(GRID, key=...)`
returning the first minimiser hides nothing here.

**Severity:** misleads a student (the offered check passes for the wrong reason — §3.2)
**Origin:** hand-written prose
**Fix:** replace "monotone" with "minimum on the edge of the grid", which is
the tell that actually fires.

---

### Claim 9 — the largest-coefficient hint contradicts its own table

**Verdict:** CONFIRMED

**Evidence:** cell 85 re-run:

```
  degree 1: converged True  iters   74  max|theta|  4.87
  degree 2: converged True  iters  256  max|theta|  5.43
  degree 3: converged True  iters  935  max|theta|  6.32
  degree 4: converged False iters 4000  max|theta| 18.70
  degree 5: converged False iters 4000  max|theta| 11.07
  degree 6: converged False iters 4000  max|theta|  3.98
```

Cell 84's `left_open`: *"why raising `max_iter` will not help. That is the next
lecture, and the largest-coefficient column is the hint."* The column is not
monotone, and degree 6 — 227 columns on 712 rows, the most over-parameterised
model in the notebook — has the **smallest** maximum coefficient of the six,
below degree 1's. A reader who follows the hint finds evidence against the
story it points at.

**Severity:** misleads a student
**Origin:** hand-written prose
**Fix:** point at the iteration column (74 → 256 → 935 → 4000) instead, and
state explicitly that the θ column does not carry the story.

---

### Claim 10 — §4.1, `v` is bound to three different kinds of object

**Verdict:** CONFIRMED (core), with the report's tail list overstated

**Evidence:** static scan of the code cells:

```
--- v ---   (16, 'for k, v in by_sex.items():')                         -> float
            (19, 'for k, v in ....agg(["mean","size"]).iterrows():')    -> pandas Series
            (58, 'v = np.zeros_like(theta)')                            -> ndarray
--- n ---   (10, 'for col, n in missing.items():')                      -> int
            (46, 46, 64, 64) 'for n, c in sorted(zip(names_…'           -> str
--- d ---   (19, 'def engineer(d):')                                    -> DataFrame
            (80, 85) 'for d in DEGREES:'                                -> int
```

Three genuine §4.1 violations: `v`, `n`, `d`. `v` is the serious one — it is
the dependency direction in the identifiability proof, the most load-bearing
object in section 9, wearing a name two earlier loops already used.

The report's tail list is partly overstated. `Z` (30, 52), `k` (16, 19), `c`
(46, 64) and `fig` (72, 82) are rebound across cells to the **same kind** of
object each time (ndarray, str, float, Figure), which §4.1 does not forbid —
its wording is *"rebound to a different kind of object"*. Calling those §4.1
defects inflates the count.

`r` (43, 80) is correctly described: both bindings are `cross_validate` result
dicts, nothing downstream reads it, and after a full run `r` no longer refers
to the section-8 experiment. A rebinding, not a type change, and harmless.

**Severity:** misleads a student (for `v` specifically; the rest is cosmetic)
**Origin:** generated code
**Fix:** rename cell 58's `v` to `shift_dir`, cell 19's loop variables to
`title, row`, and cell 80/85's `d` to `deg`.

---

### Claim 11 — `prep_v1` is a shared object mutated in place, and the hazard is not named

**Verdict:** CONFIRMED

**Evidence:** executed —

```
CLAIM11 m_v1.named_steps['prep'] is prep_v1 -> True
```

The same `ColumnTransformer` instance is fitted in cell 30, refitted by
`model_weak.fit(X_train, y_train)` in cell 33, refitted 20 more times by
`m.fit(A, ya)` inside cell 39's seed loop (on other splits, hence other
imputation medians and possibly other infrequent-category sets), and refitted
again in cell 46. (`cross_validate` and `cross_val_predict` clone, so cells 43
and 67 are not implicated.)

Top-to-bottom this costs nothing. A reader who re-runs cell 39 after cell 46 —
the obvious thing to try, since it is the cell with the seed loop — leaves
`m_v1`'s coefficients fitted to `X_train` on top of a preprocessing block
fitted to seed 19's split, and cells 52, 55 and 58 then describe a design matrix
that does not correspond to the coefficients they discuss. Cell 58's assert
still passes, because shift-invariance holds for any matrix in which
`Sex_female + Sex_male = 1` — the check passes for the wrong reason.

The hazard is not named: `clone` appears in no cell; `re-run` appears only in
cell 7 (about deleting `datasets/`); `Restart` only in cell 4 as generic
advice. §4.3 asks for the specific failure.

**Severity:** misleads a student
**Origin:** generated code
**Fix:** `clone(prep_v1)` inside `make_model`, or build the ColumnTransformer
in a function.

---

### Claim 12 — §8.3, nothing is marked examinable

**Verdict:** CONFIRMED

**Evidence:** the string "examinable" occurs in exactly three cells —

```
examinable in cell 2 markdown   ("not examinable, and it is here because…")
examinable in cell 3 code       ("# Not examinable: this is engineering hygiene…")
examinable in cell 5 code       ("# Not examinable. A free Colab CPU runtime…")
```

All three are in the setup, and all three say *not* examinable. Sections 2
through 14 carry no marking of any kind. §8.3 requires every section to carry
one of *examinable* / *not examinable — engineering* / *beyond the book*.

**Severity:** misleads a student (a student revising alone cannot tell what to revise)
**Origin:** notebook structure
**Fix:** add a marking line to each of the 14 section headings.

---

### Claim 13a — 30 prompt boxes, all 30 carrying the full three-bullet annotation

**Verdict:** CONFIRMED

**Evidence:** programmatic count over the notebook JSON —

```
prompt boxes: 30   with all three bullets: 30
```

and `python3 tools/check_notebooks.py`:

```
FAIL  lecture-05.ipynb
      30 full annotations, budget is 10 (§6.1) — every reader in the audit
      stopped reading the template around cell 30
```

§6.1 asks for five to eight, never more than ten. Cross-checking against the
same run: lecture 6 has 29, lecture 7 has 21, lecture 8 has 21 — lecture 5 is
the worst in the course, as the report says.

**Severity:** misleads a student (measured annotation fatigue — §6.1)
**Origin:** notebook structure
**Fix:** reduce to the seven the script nominates (script cells 10, 11, 13, 17,
21, 25, 28); the rest keep the specification only.

---

### Claim 13b — §6.2, cell 74/75's *"the WRONG 80 the moment the cut-off is not monotone in risk"* describes an impossible situation

**Verdict:** CONFIRMED

**Evidence:** the described student error is *"applying the cost-optimal
cut-off and then truncating the list at 80"*. The cut-off rule is
`p_oof < t` (cell 70) and the risk measure being ranked on is `p_oof` itself
(cell 76), so the flagged set is by construction an initial segment of the risk
ordering — a threshold on a quantity is monotone in that quantity, always. The
stated failure condition cannot occur, which makes the bullet invention rather
than observation.

The real hazard in that neighbourhood *is* checkable, and the bullet does not
name it:

```
CLAIM25 quicksort vs stable same 80? True   80th/81st gap 1.12e-04   n repeated values 15
```

`np.argsort` defaults to `kind="quicksort"`, which is not stable; `p_oof` has 15
repeated values; but the 80th and 81st differ by 1.1e-4, so `kind="stable"`
(which cell 76 uses) and the default select the same 80 here. Free insurance,
not a fix for an observed bug.

**Severity:** wrong but harmless
**Origin:** hand-written prose
**Fix:** replace the bullet with the argsort-stability default, and say
explicitly that it does not change the answer on this data.

---

### Claim 14 — the setup box is justified by `root_mean_squared_error`, which this notebook never calls

**Verdict:** CONFIRMED

**Evidence:** search across all 89 cells for `root_mean_squared_error` returns
exactly two hits: cell 2 (the prompt box's `constraint`) and cell 3 (a code
*comment*, `# root_mean_squared_error arrived in scikit-learn 1.4`). It is
called nowhere. It is boilerplate carried over from the regression lectures.

The notebook does have a real version dependency:
`OneHotEncoder(min_frequency=2, handle_unknown="infrequent_if_exist")` in cell
30, which requires scikit-learn ≥ 1.1 and without which cell 30 fails. That is
a better reason and it is not the one given.

**Severity:** wrong but harmless
**Origin:** hand-written prose (plus the comment in generated code, cell 3)
**Fix:** replace the justification with `OneHotEncoder`'s `min_frequency`, in
both cell 2 and cell 3's comment.

---

### Claim 15 — cell 24's six-space-indented line is flagged by the §9 checker

**Verdict:** FALSE POSITIVE

**Evidence:** the line exists — cell 24, source line index 13, indented six
spaces:

```
      + (1-y^{(i)})\log(1-\hat{p}^{(i)})\Big]$$
```

and it is the only line ≥4 spaces outside a fence in the whole notebook. But
the checker does **not** flag it. `python3 tools/check_notebooks.py` reports
exactly one failure for this file:

```
FAIL  lecture-05.ipynb
      30 full annotations, budget is 10 (§6.1)
```

`tools/check_notebooks.py:112–125` shows why: `check_indentation` fires only if
`lead >= 8` **or** the line contains one of `**`, `> `, a backtick or `](`.
This line has `lead == 6` and none of those tokens, so it passes by design —
the checker was deliberately written to tolerate a continuation line inside a
`$$…$$` span, which is exactly what this is.

The report marked this item ⚠ (reasoned, not executed) and said so. Its
conclusion that the line renders correctly is right — CommonMark does not let
an indented code block interrupt a paragraph, so this is a lazy continuation.
Its premise, that the machine check flags it, is wrong. I did not open the
notebook in Colab.

**Severity:** cosmetic
**Origin:** hand-written prose (of the report, not the notebook)
**Fix:** none needed.

---

### Claim 16 — §1.2 cannot be satisfied: the notebook stores no outputs

**Verdict:** CONFIRMED

**Evidence:**

```
total cells 89   code 29   md 60
outputs all empty: True
exec counts: {None}
```

Every code cell has `outputs: []` and `execution_count: null`, so the §1.2
advisory check has nothing to compare prose figures against and the §7.1 ⏱
check (*"any cell whose stored execution exceeded 20 s"*) can never fire —
which is presumably how claim 4 survived. Confirmed as a fact about this file;
it is a course-wide convention rather than a lecture-5 choice, so it is a
finding about the check suite more than about this notebook.

**Severity:** wrong but harmless
**Origin:** notebook structure
**Fix:** none needed at lecture level; the §1.2 and §7.1 checks need stored
outputs to be worth running at all.

---

## Summary

```
confirmed: 19   false positive: 2   unverifiable: 0
of the confirmed, 12 mislead a student
origin split — prose: 13   code: 3   structure: 4
   (claims 2, 10, 14 are mixed prose+code; counted at the site the reader
    is misled from — 2 and 14 prose, 10 code)
duplicates:
  - claims 1 and 13b are not duplicates but share a root cause: both are
    annotation bullets written about a code path the author did not re-run
    (min_frequency=2 in cell 48, monotonicity in cell 74/75).
  - claims 4 and 16 are causally linked: the ⏱ check cannot fire without
    stored outputs, which is why claim 4's markers went unchallenged.
  - claims 8b and 13b are the same species as claim 9: three "how you would
    catch it" bullets whose stated tell does not fire on this data.
  - no two claims are the same underlying defect counted twice.
```

**Calibration note.** The Phase A report for lecture 5 is unusually reliable:
every number marked ✅ reproduced *exactly* against the cached CSV — 342/891,
0.38384 vs 0.38343, 712/179, 0.666/0.617, 11→28 columns, 29 columns rank 23,
23=23 and cond 83.6, the 0.422/1.134 fold vector, 605 escorts at cost 210, 439
deaths, 3,678 at 80 crew, implied cut-off 0.0624, 5 feasible of 97, 22 and 143
columns, 6.18 accuracy points, ×4.114 log loss, `max|θ|` 4.87 → 3.98. Nothing
in the ✅ list failed to reproduce. Where I disagree it is on the report's
*reasoning about the notebook's structure*, not on its arithmetic: claim 7d
(the cross-reference resolves), claim 15 (the checker does not fire), claim 7a's
"thirteen code cells" (it is five), claim 2's "stored output" (there is none),
and claim 10's tail list (four of the eight named rebindings do not change type).

The origin split holds the audit's finding at scale: **13 of 19** confirmed
defects are in hand-written prose, and the three code-origin ones (`v`'s
retyping, `prep_v1`'s shared mutation, cell 62's hard-coded `2.6e+16`) all trace
back to a prose decision about what to say rather than a fault in what the
generated code computes. Every number this notebook prints is correct.
