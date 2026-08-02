# Triage — lecture 4 defect report

Artefact: `notebooks/lecture-04.ipynb` (56 cells, 18 code cells). Source:
`tools/notebooks/lecture_04.py`. Claims: `tools/prompts/lecture_04.md`,
section *Defects found in the current notebook*.

**Count.** My task message said 14 claims. The Phase A report's numbered list
contains **24**. All 24 are triaged below, in report order.

**Cell numbering.** The report indexes cells 0-based over *all* cells
(markdown included). I verified this: its "cell 4" is the import prompt box at
absolute index 4, its "cell 32" is the 90%-precision code cell at absolute
index 32. Where a claim turns on a cell number I also checked the alternative
reading — 1-based over *code cells only*, which is what Jupyter's `In [n]`
counter shows a student. The map is:

```
code cell #   1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18
absolute      3  5  7 10 13 15 18 21 23 27 29 32 36 40 44 46 50 53
```

**Stated once, not repeated per claim (per the brief).** The notebook stores
**zero** outputs — all 18 code cells have `outputs: []` and every
`execution_count` is `None`. No prose figure in this notebook can be reconciled
against a stored output. Any claim that a printed number and a prose number
disagree is therefore checked against *re-derivation*, not against the file.

**What I could and could not run.** MNIST is cached
(`~/scikit_learn_data/openml/openml.org/data/v1/download/52667/mnist_784.arff.gz`,
15 MB) and I loaded it — 17.9 s — so all label-derived figures are checked
against the real data. Per the brief I did **not** run the SGD or forest fits,
so nothing that depends on `y_scores` or `f_scores` is checked by execution.
Where a claim about those is nonetheless settleable by structure (monotonicity
of `recalls`, prefix/suffix arguments, array lengths) I settled it that way and
say so.

Environment: scikit-learn 1.7.2, numpy 2.3.5, Python 3.13.

---

### Claim 1 — no stored outputs anywhere; every prose figure violates §1.2
**Verdict:** CONFIRMED
**Evidence:**
```
$ python3 -c "import json; nb=json.load(open('notebooks/lecture-04.ipynb')); \
  c=nb['cells']; print('code cells', sum(1 for x in c if x['cell_type']=='code')); \
  print('total outputs', sum(len(x.get('outputs',[])) for x in c if x['cell_type']=='code'))"
code cells 18
total outputs 0
execution_counts set: {None}
```
**Severity:** misleads a student
**Origin:** notebook structure
**Fix:** execute the notebook once and commit the outputs, or the §1.2 rule is
unenforceable for this file and items 5, 6 and 7 cannot be closed.

---

### Claim 2 — four cross-references that do not resolve (§3.3)
**Verdict:** CONFIRMED — all four, under **both** numbering conventions
**Evidence:** each phrase is present verbatim in the cell named. Resolving the
targets:

| Reference | in cell | absolute target | code-cell-#N target |
|---|---|---|---|
| "cell **10** asserts the current shape" | 4 (markdown) | 10 = `for p in (0.5, 0.09035, ...)`, **no assert** | #10 = abs 27, the `clf.fit` cell, **no assert** |
| "cell **14** works today because cell **4** is still in the kernel" | 4 (markdown) | 14 and 4 are both **markdown** | #14 = abs 40 (forest CV); #4 = abs 10 (base-rate loop) |
| "the pipeline in cell **4** adds the scaler" | 6 (markdown) | 4 is **markdown** | #4 = abs 10, no pipeline |
| "nothing stops cell **14** reaching for the test set" | 12 (markdown) | 14 is **markdown** | #14 = abs 40, which uses `X_train` only |

Ground truth from parsing:
```
where the PR-length assert actually is: [29]
where make_pipeline( is called:         [7]
code cells that touch X_test:           [7, 44, 50]
```
So the shape assert is at absolute 29 / code cell #11; the pipeline is built in
absolute 7 / #3 — the very cell the cell-6 box annotates; and the first cell
that could reach for the test set to pick a threshold is absolute 44 / #15.
Every one is off, and off by a different amount, under either convention.
**Severity:** misleads a student
**Origin:** hand-written prose
**Fix:** replace the four numeric references with the cell *labels* already in
the prompt boxes ("the two-curves cell", "the choose-then-measure cell"), which
survive re-interleaving.

---

### Claim 3 — `root_mean_squared_error` is the stated reason for `>= 1.4` and is never used
**Verdict:** CONFIRMED
**Evidence:** two occurrences in the whole notebook, both in the setup cell and
its box — cell 3 `# root_mean_squared_error arrived in scikit-learn 1.4` above
`assert ... >= (1, 4)`, and cell 2's `constraint` slot. It is a regression
metric; the notebook imports no regression metric. The claim's supporting fact
about 1.3 also holds:
```
$ python3 -c "from sklearn.metrics import precision_recall_curve as f; import inspect; print(inspect.signature(f))"
(y_true, y_score, *, pos_label=None, sample_weight=None, drop_intermediate=False)
# docstring: "drop_intermediate : bool, default=False ... .. versionadded:: 1.3"
```
**Severity:** misleads a student
**Origin:** generated code (the assert), echoed in hand-written prose (the box)
**Fix:** float the assert to `>= (1, 3)` and cite `drop_intermediate` —
**but note the cell is `SETUP` in `tools/make_notebooks.py:137`, shared by every
lecture**, so this is a course-wide edit, not a lecture-4 one.

---

### Claim 4 — `roc_curve` and `cross_val_score` imported and never called
**Verdict:** CONFIRMED
**Evidence:** string search over all 56 cells — `roc_curve` 1 occurrence,
`cross_val_score` 1 occurrence, both on the import lines of cell 5
(`... roc_auc_score, roc_curve)` and `... cross_val_predict, cross_val_score)`).
No call site anywhere.
**Severity:** cosmetic
**Origin:** generated code
**Fix:** delete both names from cell 5.

---

### Claim 5 — "91% of the accuracy is bought by specificity" ≠ what cell 23 prints
**Verdict:** CONFIRMED
**Evidence:** cell 23 prints `100 * (1 - base_rate) * spec / lhs`. Re-derived
from the notebook's own three prose figures (accuracy 0.9690, "misses 22.8% of
the 5s" in cell 24, base rate 0.09035):
```
implied specificity = (0.9690 - 0.09035*0.772)/(1-0.09035) = 0.98857
printed share (1-p)*spec/acc                              = 92.8%
the weight 1-p                                            = 91.0%
```
And structurally, `share = 1 - p*recall/acc`, so the share equals the weight
only if `spec == acc`; here `spec 0.98857 > acc 0.9690`, so the share is
strictly above 91% for any recall below 1. Getting 91% would require
`recall = 0.9652`, which contradicts the notebook's own "22.8% missed".
**Severity:** misleads a student
**Origin:** hand-written prose
**Fix:** cell 22's bullet should read "≈93% of the accuracy" and name the
distinction — 91% is the weight $1-p$, the share is $(1-p)\cdot\text{spec}/\text{acc}$.

---

### Claim 6 — "a 1.87-point improvement" (cell 41) is derivable from nothing
**Verdict:** CONFIRMED
**Evidence:** every `accuracy_score` call site in the notebook:
```
cell  5: (import)
cell 13: accuracy_score(y_train_5, y_pred)          <- SGD, out-of-fold
cell 23: accuracy_score(y_train_5, y_pred)  x2      <- SGD, out-of-fold
cell 53: accuracy_score in the before/after loop    <- SGD, test set
cell 53: assert accuracy_score(y_test_5, after) ...
```
None takes `f_scores`. Cell 40, the cell the sentence follows, prints ROC AUC,
average precision and recall-at-90%-precision — no accuracy. The sentence names
no arithmetic, so §1.2's derivability escape does not apply.
**Severity:** misleads a student
**Origin:** hand-written prose
**Fix:** compute a forest accuracy in cell 40 and quote the printed delta, or
cut the sentence.

---

### Claim 7 — "90.39% recall on the cross-validated training scores" (cell 47) is printed by no cell
**Verdict:** CONFIRMED
**Evidence:** cell 44 prints exactly two lines — `SGD threshold {t_sgd:.2f}`
and `forest threshold {t_forest:.2f}`. It never evaluates
`recall_score(y_train_5, f_scores >= t_forest)` or reads `f_rec` at the chosen
index. No other cell prints a training recall for the forest. The figure is
self-consistent (`0.9039 * 5421 = 4900.04`), which is why it reads as safe.
**Severity:** misleads a student
**Origin:** hand-written prose
**Fix:** have cell 44 print the training recall achieved at each chosen
threshold; it is one line and it is the number the wrinkle argument rests on.

---

### Claim 8 — a hard-coded 2.12% in a print that is not an f-field
**Verdict:** CONFIRMED (with one wording correction)
**Evidence:** cell 32's last two relevant lines:
```python
print(f"\nat 99% precision, recall is {recalls[idx99]:.4f} — one 5 in fifty")
...
print("   -> quote 2.12% as an illustration, not as an operating point")
```
The second is a **plain string, not an f-string** — the report calls it "inside
an f-string that is not an f-field", which is inaccurate about the mechanism
but exactly right about the consequence: `recalls[idx99]` is computed three
lines above and the 2.12 is transcribed beside it with nothing tying them.
**Severity:** wrong but harmless (today it is consistent; it goes wrong silently)
**Origin:** generated code
**Fix:** `print(f"   -> quote {100*recalls[idx99]:.2f}% as an illustration, ...")`.

---

### Claim 9 — `row[0.0904]` in cell 36 survives on a rounding coincidence
**Verdict:** CONFIRMED — reproduced exactly
**Evidence:**
```
$ python3 -c "import numpy as np; v=5421/60000; print(repr(round(np.float64(v),4)), repr(round(float(v),4)))"
np.float64(0.0904) 0.0903
$ python3 -c "... d={round(float(5421/60000),4):'x'}; d[0.0904]"
KeyError: 0.0904
```
`base_rate = y_train_5.mean()` is a `np.float64` today, so the lookup works.
`float(base_rate)`, `base_rate.item()`, or a pandas round-trip all give
`0.0903` and the cell dies.
**Severity:** wrong but harmless (latent; fires on a refactor)
**Origin:** generated code
**Fix:** collect the three rows in a list of tuples and unpack by position.

---

### Claim 10 — `assert (d[~drops_a_five] < 0).all()` fails if the top-ranked instance is not a 5
**Verdict:** CONFIRMED, with a refinement — it needs the top **two** to be non-5s
**Evidence:** replayed cell 15's arithmetic on the real 60,000 labels with the
head of the ranking forced:
```
non-5 at top, 5 second           lab[:4]=[0 1 0 0]  assert passes: True   d[0]=+0.5000
two non-5s at the top            lab[:4]=[0 0 0 0]  assert passes: False  d[0]=+0.0000  prec[:3]=[0. 0. 0.]
5 at top (as in the notebook)    lab[:4]=[1 0 0 0]  assert passes: True   d[0]=-0.5000
```
With $T=0$ the step is flat, not a fall, so `< 0` is `False`. If the second
instance is a 5, precision moves 0 → 0.5 and the step lands in the other
bucket, so the assert survives. Over 500 random rankings of the real labels:
top-1 a non-5 in 458/500 (91.6%), top-**2** both non-5 in 426/500 (85.2%) — the
latter is the failing case. The report's one-line statement of the trigger is
slightly too broad; the defect, the mechanism and the proposed fix
(`assert lab[0] == 1`) are all correct.
**Severity:** misleads a student — the assert message ("the lemma says every one
of these rises") points at the sort, not at the empty-TP edge case
**Origin:** generated code
**Fix:** add `assert lab[0] == 1, "the lemma below assumes precision is never 0"`
above it, as the script already does.

---

### Claim 11 — "5,417 + 3 = 5,420" is transcribed; only the total is structural
**Verdict:** CONFIRMED, with a refinement to the report's own supporting figure
**Evidence:** replaying cell 15 on the real labels under random rankings whose
top instance is a 5 (the notebook's regime — cell 18's prose "5/6" implies it):
```
seed 0: rises/non-5 steps (d<0) = 54,579   falls (d>0) = 5,420   flat = 0   leading run of 5s = 1
seed 1: ... 54,579 ... 5,420 ... 0
... (6 seeds, identical)
(~y_train_5).sum() = 54579        y_train_5.sum()-1 = 5420
```
Both totals are forced. The 5,417/3 split is not: the flats are exactly
(leading run of 5s − 1), so "3" says the SGD ranking opens with **four**
consecutive 5s — a property of one run. Under a random ranking the split is
5,420/0.
**Refinement:** the report says 54,579 is "forced by the labels alone for any
ranking". It is not — it is forced *once the top instance is a 5*. With a non-5
on top I measured `(d<0).sum()` = 54,566 / 54,576 / 54,576 / 54,575 / 54,559
across five rankings, because the leading run of non-5s produces flat steps
while $T=0$. This is the same edge case as claim 10 and strengthens it.
**Severity:** misleads a student — cell 16 presents three numbers of two
different kinds in one sentence
**Origin:** hand-written prose
**Fix:** state 54,579 and 5,420 as `(~y_train_5).sum()` and `y_train_5.sum()-1`;
present 5,417/3 explicitly as "on this run".

---

### Claim 12 — `assert prec[4] < prec[5]` (cell 18) is a hard-coded fact about rank 6
**Verdict:** CONFIRMED
**Evidence:** the assert holds iff `lab[5] == 1` and precision at k=5 is below 1.
Over 500 random rankings of the real labels it held in **53/500**; the
lemma-based version (find the first fall, assert the instance accepted there is
a non-5) held in 500/500. The cell whose stated purpose is "a fact the reader
can check by hand" is the one asserting a fact that depends on the seed and the
scikit-learn version.
**Severity:** misleads a student
**Origin:** generated code
**Fix:** replace with `first_down = int(np.flatnonzero(d < 0)[0]) + 1; assert lab[first_down] == 0`.

---

### Claim 13 — the unguarded `argmax` the notebook argues against is used eight cells later
**Verdict:** CONFIRMED
**Evidence:** cell 31's box: *"**The usual student version:** `(precisions >= 0.90).argmax()` alone."*
Cell 32 builds `MIN_SUPPORT = 500` and `ok = (precisions[:-1] >= 0.90) & (flagged >= MIN_SUPPORT)`.
Cell 40 then prints the headline comparison row with:
```python
f"{recalls[(precisions >= 0.90).argmax()]:>10.4f}"
f"{f_rec[(f_prec >= 0.90).argmax()]:>10.4f}")
```
— the exact expression the box names, unguarded, for **both** models, and cell
41 calls that row "the row the brief is about". §2.3: the correction was not
propagated.
**Severity:** misleads a student
**Origin:** generated code
**Fix:** factor the guarded crossing into a two-line helper in cell 32 and call
it from cell 40 for both models.

---

### Claim 14 — the same two cells index arrays of different lengths for the same question
**Verdict:** CONFIRMED (harmless numerically)
**Evidence:** cell 32 uses `precisions[:-1]` throughout (`flagged`, `ok`,
`naive`, `idx99`, both `.sum()` counts); cell 40 uses the unsliced `precisions`
and `f_prec`. They agree, and agree structurally rather than by luck:
`precision_recall_curve` returns precision in increasing order with the
degenerate 1.0 **last**, so `argmax` on the full array finds the same first
crossing as on the sliced one whenever a crossing exists before the end. The
defect is pedagogical — cell 28's box teaches that the two lengths matter, and
the notebook then uses both conventions for one question.
**Severity:** cosmetic
**Origin:** generated code
**Fix:** slice `[:-1]` at every threshold-world boundary, without exception.
**Note:** near-duplicate of claim 13 — same two cells, same expression, two
different complaints about it.

---

### Claim 15 — `idx99` is bound twice to indices with different meanings
**Verdict:** CONFIRMED
**Evidence:** the only two assignments in the notebook:
```
cell 32: idx99 = int((precisions[:-1] >= 0.99).argmax())     # precision-based, length-T world
cell 50: idx99 = np.where(recalls >= 0.99)[0][-1]            # recall-based, length-T+1 world, other end
```
Both are integers, so §4.1's letter (no rebinding to a different *kind* of
object) survives; its purpose does not. No comment marks the change.
**Severity:** wrong but harmless
**Origin:** generated code
**Fix:** rename to `i_prec99` and `i_rec99`.

---

### Claim 16 — cell 26's "usual student version" describes what cell 44 correctly does
**Verdict:** CONFIRMED
**Evidence:** cell 26's bullet: *"**The usual student version:** reusing this
fitted `clf` for an accuracy figure later. It has seen everything; every score
it produces is optimistic."* Cell 44: `sgd_final = clf  # already fitted above`,
then `sgd_final.decision_function(X_test)`; cells 46 and 53 report test metrics
from it. `clf` was fitted in cell 27 on `X_train` only and is scored on
`X_test` — fitted on train, measured on test, which is correct. "Every score it
produces is optimistic" is false for test-set scores.
**Severity:** misleads a student — it teaches distrust of a valid procedure,
and §6.2 requires these bullets to be observed rather than invented
**Origin:** hand-written prose
**Fix:** narrow the bullet to "reusing it for a *training* accuracy figure", and
say that cell 44's reuse is legitimate and why.

---

### Claim 17 — "the only threshold that reaches 90% recall flags 1,377 items"
**Verdict:** CONFIRMED
**Evidence:** cell 44 computes `t_sgd = thresholds[np.where(recalls >= 0.90)[0][-1]]`.
`precision_recall_curve` returns `recall` non-increasing, so `{r >= 0.90}` is a
**prefix** of the array and `[0][-1]` selects its **last** element — the
*highest* threshold clearing 90% recall. Every index before it also clears 90%
recall, at a lower threshold, flagging strictly more. So thresholds reaching
90% recall are a whole interval, not one point. The intended claim — no
threshold reaches 90% recall inside 1,000 flagged — is true and strictly
stronger, since 1,377 is the minimum flagged count over that interval.
**Severity:** misleads a student
**Origin:** hand-written prose
**Fix:** "the *cheapest* threshold that reaches 90% recall still flags 1,377
items — every other one flags more."

---

### Claim 18 — the §7 trap is announced three times before the cell (§8.1)
**Verdict:** CONFIRMED
**Evidence:** three announcements, all above code cell 50:
- cell 48 (section heading): `**⚠ Read before running.** This one runs, it is correct code, it does exactly what was asked, and the sentence it reports is true.`
- cell 49 (`constraint` slot): `this is the cell to read before running — it is what an assistant returns when asked to 'improve recall', and it does exactly that`
- cell 49 (third bullet): `Ask what went down, and the next cell answers: 5,303 extra false alarms and an accuracy below the never-fires baseline.`

The third gives away both headline numbers of cell 53 before the reader has run
cell 50. This is the §8.1 defect verbatim.
**Severity:** misleads a student — it disarms the one cell in the notebook whose
whole value is that the reader falls in
**Origin:** notebook structure (ordering)
**Fix:** run cell 50 unannounced; move the ⚠ and the third bullet into the
markdown that opens §9, after the reader has written the two recalls down.

---

### Claim 19 — eighteen code cells, eighteen full three-bullet annotations (§6.1)
**Verdict:** CONFIRMED
**Evidence:**
```
prompt boxes: 18   with 'Left open': 18   code cells: 18
```
Budget is five to eight, never more than ten: 2.25× the ceiling. The §7 defect
sits at cell 49, past where all three audit readers stopped reading the
template.
**Severity:** misleads a student (via the mechanism §6.1 documents — this is
the same file whose §7 payload is at cell 49)
**Origin:** notebook structure
**Fix:** keep the short `input/output/constraint/check` spec on all 18; keep the
three-bullet form on the six that earn it — cells 6, 12, 28, 31, 43, 52.

---

### Claim 20 — "examinable" appears twice, both about the setup cell (§8.3)
**Verdict:** CONFIRMED
**Evidence:** both occurrences in the whole notebook:
```
cell 2 (markdown): * **How you would catch it:** not examinable, and it is here because ...
cell 3 (code):     # Not examinable: this is engineering hygiene, not machine learning.
```
No other section carries a mark, including §2, the mathematical thread the
lecture is named after.
**Severity:** wrong but harmless
**Origin:** notebook structure
**Fix:** mark all eight sections. Note cell 3 is the shared `SETUP` constant
(`tools/make_notebooks.py:137`), so its mark is course-wide, not lecture-4's.

---

### Claim 21 — vocabulary undefined on first use (§7.5)
**Verdict:** CONFIRMED, with one count corrected
**Evidence:** case-insensitive occurrence counts over all 56 cells:
```
'shift'            18      (report says "20+")
'operating point'   8
'support'           9
'plateau'           2
'red-team'          1
'flatters'          1
```
None is defined. First use of "shift" is cell 42's heading *"The operating
point, and the **test shift**"* with no gloss; the collision with the array
operation of the same name is real (lecture 19 uses `shift(7)` constantly).
"support" is partially glossed in context — cell 32's comment "a real operating
point needs support behind it" beside `MIN_SUPPORT = 500` — so it is the
weakest member of the list. "red-team" appears only as the §8 heading.
**Severity:** misleads a student (the second-language reader §7.5 protects)
**Origin:** hand-written prose
**Fix:** the vocabulary block the script already drafts, before §1.

---

### Claim 22 — lecture 4 and lecture 18 use the same words for opposite directions
**Verdict:** CONFIRMED
**Evidence:** the same array element, two opposite labels.

Lecture 4 cell 15 (`d = np.diff(prec)`, `prec` cumulative top-down):
```python
print(f"drop a non-5, precision rises      {(d < 0).sum():,}")
print(f"drop a 5,     precision falls      {(d > 0).sum():,}")
```
Lecture 18 cell 40 (`step = np.diff(precision)`, same construction):
```python
down = int((step < -1e-12).sum())
...
print(f"steps down (precision falls) : {down}")
assert down == n_fp, (down, n_fp)
```
`d < 0` counts the same steps in both — the ones where a negative is accepted —
and lecture 4 calls them "rises" while lecture 18 calls them "falls" and
asserts they equal the false-positive count. Lecture 18 cell 38 tells the reader
it is doing this *"the way Lecture 4 did for MNIST"*.
**Severity:** misleads a student
**Origin:** hand-written prose (the derivation direction in cell 11; the labels
themselves are print strings in cell 15)
**Fix:** read the ranking top-down in lecture 4, as the script does, and relabel
cell 15's two prints.

---

### Claim 23 — the two lectures use "average precision" for two different estimators
**Verdict:** CONFIRMED — both numbers reproduced
**Evidence:** ranking TP, FP, FP, TP, TP (3 annotations):
```
cumulative precision: [1.0  0.5  0.3333  0.5  0.6]
cumulative recall   : [0.3333 0.3333 0.3333 0.6667 1.0]
lecture 18 average_precision (cell 43, verbatim) : 0.7333333333333334  -> 0.7333
sklearn average_precision_score                  : 0.7                 -> 0.7000
```
Lecture 4 cells 36 and 40 call `average_precision_score`; lecture 18 cell 43
defines its own `average_precision` on the enveloped curve. Lecture 18 cell 41
says *"Lecture 4 proved precision has no monotone envelope"* — the string
`envelope` occurs **0** times in `lecture-04.ipynb`.
**Severity:** misleads a student
**Origin:** hand-written prose (the cross-reference), on top of two genuinely
different estimators in generated code
**Fix:** name them apart — "AP (sklearn, non-interpolated)" and "AP (enveloped,
all-point)" — and cut lecture 18's back-reference or give lecture 4 the envelope.

---

### Claim 24 — no CPU figure beside any ⏱ (§7.1), and the SGD timing looks optimistic
**Verdict:** CONFIRMED for the §7.1 half; the timing-accuracy half is UNVERIFIABLE
**Evidence:** every timing note in the notebook, and what each states:
```
cell  7 (code):     # ~30 s the first time; scikit-learn has it cached ...
cell 11 (markdown): ⏱ **about 60 seconds** — three refits ...
cell 26 (markdown): > **constraint** · ⏱ this refits on all 60,000 rows, about 30 seconds
cell 27 (code):     clf.fit(X_train, y_train_5)   # ⏱ about 30 s
cell 38 (markdown): ⏱ **under a minute.**
cell 42 (markdown): ⏱ **about 30 seconds** for the two final fits.
```
Regex search for `core|CPU|n_iter_|laptop` over all 56 cells returns **no
cells**. §7.1 requires a CPU number beside every ⏱; none is given. Confirmed.

The sub-claim that "about 30 s" is optimistic requires executing the SGD fit,
which the brief forbids, so I did not test it and neither did Phase A — the
report says so itself. Marked unverifiable, not refuted.

The cell-42 sub-claim is checkable and correct: cell 44 contains one fit, not
two — `sgd_final = clf  # already fitted above` (no `.fit`) and
`forest_final = forest.fit(X_train, y_train_5)`. The prose says "the two final
fits".
**Severity:** misleads a student (§7.1 evidence: untimed cells blocked four of
six exercises in the audited lecture)
**Origin:** hand-written prose
**Fix:** state cores and machine beside each ⏱; change cell 42 to "one final
forest fit — the SGD is reused from the cell above".

---

## Summary
confirmed: 24   false positive: 0   unverifiable: 0
of the confirmed, 18 mislead a student
origin split — prose: 11   code: 9   structure: 4

(prose: 2, 5, 6, 7, 11, 16, 17, 21, 22, 23, 24 · code: 3, 4, 8, 9, 10, 12, 13,
14, 15 · structure: 1, 18, 19, 20. Not misleading: 4 and 14 cosmetic; 8, 9, 15,
20 wrong but harmless.)

**Partial verdicts inside the 24.** Four claims are correct in substance but
overstated in a detail, and the detail is recorded in the entry:
- **10** — the assert fails when the top **two** are non-5s (85.2% of random
  rankings), not merely the top one (91.6%).
- **11** — 54,579 is forced *once the top instance is a 5*, not "by the labels
  alone for any ranking"; I measured 54,566 / 54,576 / 54,576 / 54,575 / 54,559
  under rankings led by a non-5. The claim itself stands.
- **21** — "shift" occurs 18 times, not "20+"; "support" is partially glossed.
- **24** — the "SGD timing is optimistic" half is untested by anyone, and is
  recorded as such rather than as confirmed.
- **8** — the offending line is a plain string, not "an f-string that is not an
  f-field"; the transcription defect is real.

**duplicates:** 13 and 14 are two complaints about one line
(`recalls[(precisions >= 0.90).argmax()]` in cell 40) — the missing support
guard and the unsliced array. Fixing cell 40 closes both. 10 and 11 share one
root cause, the $T=0$ head of the ranking, which surfaces once as a failing
assert and once as a mis-stated invariant. 5, 6 and 7 are three instances of
claim 1: a prose figure with no output to reconcile against.

**Calibration.** No lecture-3 or lecture-6 calibration claim appears in this
lecture's report, so this triage carries no calibration signal. What it does
carry: MNIST-derived figures re-checked against the cached copy all held —
`y_train_5.sum() = 5421`, `y_test_5.sum() = 892`, base rate `0.09035`,
never-fires `0.90965`, `(~y_train_5).sum() = 54579`, test baseline `0.91080`,
`X_train` `int64` 376.3 MB. Phase A got every one of those right.
