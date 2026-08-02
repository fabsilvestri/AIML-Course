# Triage — lecture 11 defect report

Artefact: `notebooks/lecture-11.ipynb` (50 cells, 15 code cells).
Source: `tools/notebooks/lecture_11.py`. Claims: `tools/prompts/lecture_11.md`.
Environment: python 3.13.5, scikit-learn **1.7.2**, Fashion MNIST cached at
`notebooks/datasets/FashionMNIST` (read only, never re-downloaded).

**Claim count.** The task message said 11 claims. The Phase A report's
`Checked and confirmed` section contains **16** numbered claims. All 16 are
triaged below.

**Stated once, not repeated per claim (per the brief):** all 15 code cells of
this notebook have `execution_count: null` and zero stored outputs, so no prose
figure in it can be reconciled against the file itself. Verified course-wide:
of the 24 notebooks only `lecture-19.ipynb` stores anything (19 of its 20 code
cells, 21 output objects).

**No training cell was executed.** Every fit below is on synthetic arrays of
20–784 columns and ≤300 rows, used only to read library behaviour and
signatures, never to produce an accuracy.

---

### Claim 1 — every one of the 15 prompt boxes carries the full three-bullet annotation
**Verdict:** CONFIRMED
**Evidence:** parsed the notebook JSON for markdown cells beginning `> **Prompt`:

```
prompt boxes: 15 [2, 5, 8, 10, 14, 17, 22, 25, 29, 31, 35, 37, 41, 45, 47]
full (all 3 bullets): 15
```

Every box contains all of `**Left open:**`, `**The usual student version:**`,
`**How you would catch it:**`. The project's own checker agrees:

```
$ python3 tools/check_notebooks.py
FAIL  lecture-11.ipynb
        15 full annotations, budget is 10 (§6.1)
```

`GUIDELINES.md` §6.1: "aim for **five to eight per notebook**, never more than
ten."
**Severity:** misleads a student
**Origin:** notebook structure
**Fix:** demote ten boxes to specification-only; keep the full form on the
cells the prompt script marks `Annotate: full` (cells 6, 15, 23, 30, 46, 48).

---

### Claim 2 — the `partial_fit` failure mode is stated backwards
**Verdict:** CONFIRMED
**Evidence:** cell 29's student bullet says omitting `classes=` "works on the
first call and raises on the second, or worse, silently fits a model that has
never seen a class absent from the first batch." Run against sklearn 1.7.2:

```
first call WITHOUT classes=: ValueError: classes must be passed on the first call to partial_fit.
second call WITHOUT classes= (after first WITH): OK, classes_ [0 1 2 3 4 5 6 7 8 9]
first batch missing class 7, classes= given -> classes_ [0 1 ... 9] n_outputs_ 10
```

Both halves are wrong. The first call raises; the second succeeds; and the
"silently fits" case is unreachable, because you cannot get past call one. The
third line also shows the real risk is the opposite of the one described:
`classes=` is what fixes the output layer's width *before* any data is seen.
**Severity:** misleads a student
**Origin:** hand-written prose
**Fix:** rewrite the bullet to the observed behaviour — first call raises
loudly, later calls do not need it, and `classes=` exists to fix the output
width, not to insure a second call.

---

### Claim 3 — the deliberate defect is announced five times before the reader can fall into it
**Verdict:** CONFIRMED
**Evidence:** grep for `⚠` over the notebook JSON returns **cells 0, 21, 22** —
all ahead of the defective code cell 23. Contents:

- cell 0: "Cells marked **⚠ read before running** contain a defect on purpose"
- cell 21: "**⚠ Read before running.**"
- cell 22 label: `⏱ 40 s — ⚠ what the assistant returns`
- cell 22 `constraint`: "feed it the RAW uint8 pixels"
- cell 22 `left_open`: "``learning_rate_init=0.001`` is scikit-learn's — the
  default for inputs of order ONE. We handed the network integers up to 255."

Cell 24 ("Reviewer question 5") states the same diagnosis a sixth time, after.
`GUIDELINES.md` §8.1 prescribes the opposite ordering.
**Severity:** misleads a student
**Origin:** notebook structure
**Fix:** strip cell 22's `constraint` and `left_open` down to the specification,
drop the ⚠ from cells 0/21/22, and move the whole diagnosis into cell 24 where
it already exists.

---

### Claim 4 — "The scaling happens two sections down" is one section down
**Verdict:** CONFIRMED
**Evidence:** the sentence is in cell 5's student bullet. Parsed every `## N ·`
heading:

```
cell  4 : ## 2 · The brief
cell 13 : ## 3 · Scale, and split
```

Cell 5 sits inside section 2; the scaling is section 3. One section down.
**Severity:** wrong but harmless
**Origin:** hand-written prose
**Fix:** "the scaling happens in the next section, deliberately."

---

### Claim 5 — "The previous application spent an hour on why accuracy is worthless under imbalance"
**Verdict:** CONFIRMED
**Evidence:** the sentence is in cell 10's `left_open`. `LECTURES.md` states
"Applications are covered in **pairs of consecutive lectures**". Lecture 11
opens Part II, so the previous application is the **Lectures 9–10** pair
(Olivetti clustering → PCA/dimensionality reduction). The imbalance material is
`## Lecture 4 — It never fires` ("*why accuracy fails under imbalance*"), which
is the Lectures 3–4 pair. Cell 12, two cells later, gets it right:

> "**That single assertion decides the metric.** Lecture 4 spent an hour on why
> accuracy is worthless under imbalance."

The notebook contradicts itself about the same reference within three cells.
**Severity:** misleads a student
**Origin:** hand-written prose
**Fix:** cell 10 → "Lecture 4 spent an hour on…", matching cell 12.

---

### Claim 6 — "it warns; we come back to that" never comes back
**Verdict:** CONFIRMED
**Evidence:** the comment is on cell 23's `warnings.simplefilter("ignore")`.
Grep over all 50 cells: `"Convergence"` appears in **no cell**; `"warn"`
appears only in cells 3, 23, 26, 30, and in every case as the
`warnings.catch_warnings()` machinery, never as discussion. Nothing names or
returns to the suppressed warning.

Exact text of the suppressed warning, reproduced on a 60×20 synthetic array so
that no training cell was run:

```
ConvergenceWarning: Stochastic Optimizer: Maximum iterations (12) reached and
the optimization hasn't converged yet.
```

**Severity:** wrong but harmless
**Origin:** hand-written prose
**Fix:** add one paragraph after cell 26 quoting the warning and making the
point that it is about `max_iter` — which was stated — and silent about the
pixel range, which was not. Or delete the promise.

---

### Claim 7 — "Both come back in the error analysis" resolves half way
**Verdict:** CONFIRMED
**Evidence:** cell 8's `left_open` names two things: "several classes are
garments photographed the same way, and the background is exactly zero. Both
come back in the error analysis." Grep for `background` over all 50 cells
returns **cells 7 and 8 only**. The error analysis is cells 40–43; none of them
mentions the background. `GUIDELINES.md` §7.3: a reader who genuinely looks and
finds N−1 cannot tell whether the failure is theirs.
**Severity:** misleads a student
**Origin:** hand-written prose
**Fix:** name only the garment-similarity observation as the one that returns,
or add the background to the error analysis.

---

### Claim 8 — "the background is exactly zero" is not what the data says
**Verdict:** CONFIRMED
**Evidence:** measured on all 60,000 training images from the cached dataset:

```
pct pixels exactly 0:                          50.2051%
top-left pixel 0 in                            99.9783% of images
images with entirely-zero 1px border:           0.0350%
```

The report's three figures (50.21%, 99.98%, 0.03%) all reproduce. Half the
pixels are exactly zero; the *border* is almost never entirely zero, because
the garments are cropped to fill the frame. The sentence as written in cells 7
and 8 is false.
**Severity:** misleads a student
**Origin:** hand-written prose
**Fix:** "half the pixels are exactly zero" — checkable, and true.

---

### Claim 9 — the name `c` is bound to two different types across cells
**Verdict:** CONFIRMED
**Evidence:** grepped every `\bc\b` binding in the code cells:

```
cell  9 | for c in range(10):                                  int
cell 11 | for c, n in enumerate(counts):                       int
cell 36 |     c, r = train_curve(X_fit[:SMALL], ...)            MLPClassifier
cell 38 |     c, r = train_curve(X_fit[:SMALL], ...)            MLPClassifier
cell 42 | for c in np.argsort(recall):                         int
cell 42 | for c in np.argsort(-row)[:4]:                       int
```

`GUIDELINES.md` §4.1 names this exact failure. The report's own qualification
("harmless today") is right: cells 36/38 bind `c` inside their own loops and
cell 42 rebinds it before use, so no live value is clobbered.
**Severity:** wrong but harmless
**Origin:** generated code
**Fix:** rename the model in cells 36 and 38 to `m` and the tuple to `m, r`.

---

### Claim 10 — cell 26 is a full 12-epoch fit with no ⏱ marker anywhere above it
**Verdict:** CONFIRMED
**Evidence:** grep for `⏱` returns **cells 21, 22, 28, 29, 34, 35**. Section 6
is cell 21, section 7 is cell 28. The only ⏱ in that span is on cells 21/22,
and it reads "⏱ **about 40 seconds**" attached to cell 23. Cells 24, 25 and 26
carry no timing. Cell 26 is a second `MLPClassifier(hidden_layer_sizes=(300,
100), max_iter=12).fit(X_fit, y_fit)` on the same 12,000 rows — by the
notebook's own estimate for the identical fit in cell 23, about 40 seconds. The
header of cell 0 promises "Anything that takes more than a few seconds says
so."
**Severity:** misleads a student
**Origin:** notebook structure
**Fix:** add "⏱ **about 40 seconds** — a second fit of the same size" to cell 24
or 25.

---

### Claim 11 — "three walls" over a four-row table
**Verdict:** CONFIRMED
**Evidence:** cell 49 reads "You also have **three** walls" and is immediately
followed by a four-row markdown table (change the objective / see a gradient /
stop mid-epoch / use a GPU). Section 10 is titled "Three things you cannot do"
(cell 44) and contains two code cells demonstrating four things: cell 46 covers
the objective, the gradient and the epoch; cell 48 covers the GPU. §7.3 again —
a reader who counts finds four.
**Severity:** wrong but harmless
**Origin:** hand-written prose
**Fix:** "four walls, three of them demonstrated in section 10 and the fourth
just now" — and retitle section 10, or split the GPU row out of the table.

---

### Claim 12 — "Write your best validation accuracy" is ambiguous across five cells
**Verdict:** CONFIRMED (in part — see the two sub-claims below, which do not
hold)
**Evidence for the core claim:** cell 49 says "Write your **best validation
accuracy** on the same sheet of paper". Validation accuracies printed by the
notebook, from the code cells:

| cell | printed | count |
|---|---|---|
| 23 | `acc_raw` (the deliberately defective raw-pixel model) | 1 |
| 26 | `acc_raw` again, and `acc_scaled` | 1 new |
| 30 | `hist['val_acc'][-1]` — 12,000 images, 20 epochs | 1 |
| 36 | five architecture rows — 6,000 images, 8 epochs | 5 |
| 38 | five learning-rate rows — 6,000 images, 8 epochs | 5 |

**13 distinct numbers**, not the report's "twelve" (12 only if you exclude the
defective raw model, which the notebook does not tell the reader to). Nothing
in cell 49 says which one, and the candidates were fitted on two different data
sizes (12,000 vs 6,000) and two different epoch counts (20 vs 8). The defect is
real.

**Sub-claim (c), "the literal maximum will almost certainly come from a sweep
row fitted on 6,000 images for 8 epochs": UNVERIFIABLE, and the prior runs the
other way.** Establishing it requires the training cells. Cell 30's model has
twice the data and 2.5× the epochs of every sweep row, so it is the more likely
maximum, not less.

**Sub-claim (d), "Lecture 12 then compares that number against a PyTorch model
… on unmatched training budgets": FALSE.** `notebooks/lecture-12.ipynb` does not
inherit lecture 11's number. Cell 18 is headed "Time the Scikit-Learn model on
this machine, as the control", cell 19's `constraint` reads "time it here rather
than quoting the previous lecture — the speed-up claimed below is a ratio, and
both halves have to come from the same box", and cell 20 re-fits
`MLPClassifier(hidden_layer_sizes=(300,100), solver="adam",
learning_rate_init=LR, batch_size=BATCH, max_iter=EPOCHS)` with
`EPOCHS, BATCH, LR = 10, 128, 1e-3` on the same `X_fit`. Lecture 12's internal
comparison is matched by construction. What its section 15 asks the reader to
compare is the *committed prediction* from the sheet of paper, not lecture 11's
measured best.

A separate small defect surfaced while checking (d): cell 49's "Bring it to the
next lecture; **we open by comparing them**" is wrong about position — lecture
12 re-measures in its **section 15**, cell 57, not at its opening.
**Severity:** misleads a student
**Origin:** hand-written prose
**Fix:** cell 49 → "write the validation accuracy from section 7 — 20 epochs,
12,000 images, hidden layers (300, 100)", and change "we open by comparing" to
"we come back to it".

---

### Claim 13 — "examinable" appears exactly once, in a code comment
**Verdict:** CONFIRMED
**Evidence:** case-insensitive grep for `examinable` over all 50 cells returns
exactly one hit:

```
cell 3 (code): # Not examinable: engineering hygiene. It is here because a version
```

Sections 2 through 11 carry no marker of any kind. `GUIDELINES.md` §8.3
requires one of *examinable* / *not examinable — engineering* / *beyond the
book* on every section.
**Severity:** misleads a student
**Origin:** hand-written prose
**Fix:** add a marker to each of the eleven `## N ·` headings.

---

### Claim 14 — no figure can be reconciled with a stored output, because there are none
**Verdict:** CONFIRMED (one sub-figure imprecise)
**Evidence:** all 15 code cells of `lecture-11.ipynb` have `execution_count:
null` and zero outputs. Course-wide scan of `notebooks/lecture-*.ipynb`:

```
lecture-19.ipynb  code 20  cells with outputs 19  total output objects 21
every other notebook            cells with outputs 0
```

The report says lecture 19 stores "21 of them". 21 is the count of **output
objects**; it is **19** code cells. The claim's substance — this is course-wide
and the three ⏱ figures rest on a run nobody can see — holds.
**Severity:** wrong but harmless
**Origin:** notebook structure
**Fix:** none needed at lecture-11 level; it is a course-wide decision. If it
is ever revisited, storing outputs would make `check_notebooks.py`'s §1.2 and
§7.1 checks meaningful instead of vacuous.

---

### Claim 15 — "none of the six checks the guidelines say must be added to the tooling exists"
**Verdict:** FALSE POSITIVE
**Evidence:** `tools/check_notebooks.py` exists (279 lines) and implements all
six. Its docstring: "Six rules, from GUIDELINES.md §9." Its functions:

```
check_indentation      §5.1/5.2   blocking
check_quoted_code      §3.1       advisory
check_annotation_budget §6.1      blocking
check_clock_markers    §7.1       blocking
advise_numbers         §1.2       advisory
advise_rebinding       §4.1       advisory
```

It runs and produces findings on this very notebook:

```
$ python3 tools/check_notebooks.py --advisory
FAIL  lecture-11.ipynb
        15 full annotations, budget is 10 (§6.1)
note  lecture-11.ipynb  (22 advisory)
```

The claim's *second* sentence is true — `check_all.py`'s FAST list is decks,
provenance, fonts, notebook-build, and does not include it. But that omission
is deliberate and documented in the very section the claim cites.
`GUIDELINES.md` §9: "**It is deliberately not yet wired into
`tools/check_all.py`.** Every notebook currently violates §6.1 … it gets wired
in as the last step of the rebuild, when the budget is actually met. Until
then, run it by hand." The claim reports a documented plan as a missing
artefact.
**Severity:** n/a
**Origin:** n/a
**Fix:** none needed. (Claim 1's fix is the precondition for wiring it in.)

---

### Claim 16 — three results are stated in prose before the cells that produce them
**Verdict:** CONFIRMED (one attribution wrong)
**Evidence:** the three statements, and the cells that compute them:

1. cell 35 `catch`: "depth buys less than you expected. One hidden layer to two
   is worth a point or so; a third is worth roughly nothing here." — computed by
   cell **36**.
2. cell 37 `left_open`: "the reading to write down: the learning rate matters
   more than the architecture." — computed by cell **38**.
3. "The default is fine here and this cell is what tells you how much luck that
   was." — also above cell **38**.

Statement 3 is in cell 37's **`student`** bullet, not its `left_open` as the
report states. The count of three and the ordering defect are right. Cell 39
then repeats both readings as a numbered list *after* the cells, which is where
they belong.
**Severity:** misleads a student
**Origin:** hand-written prose
**Fix:** strip the findings out of cells 35 and 37; cell 39 already carries
them in the correct position.

---

## Defect the Phase A report missed — and got backwards

Not one of the 16 claims. It refutes an item the report filed under **"Checked
and found clean"**, and it is the "check that passed for the wrong reason"
pattern the brief warns about, so it is recorded here.

### The gradient wall does not demonstrate itself — cell 46 prints a non-empty list
**Verdict:** CONFIRMED (against the report, which claims the opposite)
**Evidence:** the report states, under *Checked and found clean*: "`dir(clf)`
for 'grad' returns `[]` … I ran all three." The prompt script
`tools/prompts/lecture_11.md` Cell 14 likewise gives as **Expect**:
"`attributes containing 'grad': []` — an empty list". Run against
scikit-learn 1.7.2, using cell 46's exact expression on a model constructed the
way `train_curve` constructs it:

```python
clf = MLPClassifier(hidden_layer_sizes=(300,100), activation="relu", solver="adam",
                    learning_rate_init=1e-3, batch_size=128, random_state=42)
clf.partial_fit(X, y, classes=np.arange(10))
[a for a in dir(clf) if "grad" in a.lower()]
```
```
attributes containing 'grad': ['_compute_loss_grad', '_loss_grad_lbfgs']
```

Two hits, not zero — and the same two on an *unfitted* instance, so it is not a
fitting artefact. `dir()` returns class attributes, and
`BaseMultilayerPerceptron` defines both methods. Only the public-name filter is
empty:

```
public only: []
```

Cell 46's printed line will read `attributes containing 'grad':
['_compute_loss_grad', '_loss_grad_lbfgs']`, directly under prose that promises
"the absence of any gradient attribute" (cell 45 `output`) and one section above
cell 49's table row "see a gradient | nothing is exposed". A student who reads
the output rather than the prose sees two gradient-named attributes and
concludes the notebook is wrong — or, worse, goes looking for a gradient in
`_compute_loss_grad`, which is a per-layer helper called inside `_backprop` and
returns nothing you can inspect after `fit`.

The other two walls in cell 46 do behave as claimed:

```
loss= -> TypeError: MLPClassifier.__init__() got an unexpected keyword argument 'loss'
partial_fit signature: (self, X, y, sample_weight=None, classes=None)
```

**Severity:** misleads a student
**Origin:** generated code (the filter), compounded by hand-written prose (the
`Expect` in the script and the "found clean" line in the report)
**Fix:** filter to public names —
`[a for a in dir(clf) if "grad" in a.lower() and not a.startswith("_")]` — and
say in one line why the private ones do not count: they are called inside
`_backprop` and nothing survives the call.

---

## Other findings while verifying, all clean

Re-derived independently from the cached dataset under `default_rng(42)`;
each matches the notebook and the script:

```
shapes (60000,28,28) (10000,28,28) uint8 0 255
train counts min==max==6000: True          test counts 1000 x 10
fit 55000  val 5000  test 10000   disjoint True
12k subsample counts: min 1170 (Bag)  max 1246 (Sandal)
majority Sandal -> baseline accuracy 0.1  (exactly)
raw == 255 * scaled, exactly: True   (max abs diff 0.0)
param counts: (30,) 23,860  (100,) 79,510  (300,) 238,510
              (300,100) 266,610  (300,200,100) 316,810
sklearn defaults: learning_rate_init=0.001  batch_size='auto'
                  learning_rate='constant'  solver='adam'  max_iter=200
55,000 / 12,000 = 4.5833  (|diff| 3.3e-05 < 1e-3)
```

So the raw-versus-scaled comparison really is on matched rows, the parameter
counts in the script's table are all correct, the 12,000-slice imbalance
(1,170–1,246) is real, and `learning_rate_init=0.001` is indeed the
`MLPClassifier.__init__` default in 1.7.2.

The report's "Unverified" section is correctly scoped: the three ⏱ figures,
cell 39's two readings, and the sign of `acc_scaled - acc_raw` all require the
training cells and were not run here either.

---

## Summary

```
confirmed: 15   false positive: 1   unverifiable: 0
   (+ 1 confirmed defect not in the report — the 'grad' wall, which the report
      filed as clean after reporting an output it does not produce)

of the confirmed, 10 mislead a student
   claims 1, 2, 3, 5, 7, 8, 10, 12, 13, 16
   (11 including the missed 'grad' defect, which is not one of the 16)

origin split — prose: 10   code: 1   structure: 4    (the 15 confirmed claims)
   prose:     2, 4, 5, 6, 7, 8, 11, 12, 13, 16
   code:      9
   structure: 1, 3, 10, 14
   claim 15 is a false positive and carries no origin.
   the missed 'grad' defect adds one more to code, giving prose 10 / code 2 /
   structure 4 over all 16 real defects.

partially confirmed, sub-claims that do not hold:
   12(c) "the maximum will come from a sweep row"  — UNVERIFIABLE, prior runs
         the other way (12,000 rows x 20 epochs vs 6,000 x 8)
   12(d) "lecture 12 compares on unmatched budgets" — FALSE; lecture 12 cell 20
         re-times scikit-learn itself at 10 epochs on the same X_fit
   14    "lecture 19 stores 21 of them" — 21 output objects across 19 code cells
   16    statement 3 is in cell 37's `student` bullet, not its `left_open`

duplicates: 7 and 8 are one sentence counted twice — "the background is exactly
   zero" in cells 7/8. Claim 8 says the sentence is false; claim 7 says its
   forward reference never resolves. Deleting the sentence fixes both, so they
   are one repair, but they are two distinct guideline violations (§1.2 and
   §7.3) and I have not merged them.
   3 and 16 are the same *shape* of defect — a result stated before the cell
   that produces it — but on different cells (23 vs 36/38) and with different
   fixes. Not duplicates.
```

The prose-concentration hypothesis holds here: 10 of the 15 confirmed defects
are in hand-written markdown, 4 are ordering/structure, and only 1 is in the
generated Python. The one exception the audit would care about is the missed
`grad` defect, which **is** in the generated code — and which slipped past
Phase A precisely because the prose asserted the output rather than showing it.
