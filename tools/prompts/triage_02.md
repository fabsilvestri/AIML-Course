# Triage — lecture 2 defect report

Claims triaged: the seventeen lettered items **A–Q** at the end of
`tools/prompts/lecture_02.md`. (The task message said "6 claims"; the Phase A
report contains seventeen, and P is itself four sub-bullets. All seventeen are
triaged below.)

**Stated once, not repeated per claim:** `notebooks/lecture-02.ipynb` stores **no
outputs at all** — 35 cells, every `execution_count` is `None`, every `outputs`
list is empty. No prose figure in this notebook can be reconciled against a
stored output, so every numeric verdict below is a re-derivation from the
notebook's own code run against `notebooks/datasets/housing/housing.csv`.

**Environment:** scikit-learn 1.7.2, numpy 2.3.5, pandas 2.3.3, scipy 1.16.3,
Python 3.13, Apple silicon. Seed 42 throughout. The two long cells (10-fold CV,
grid search) were executed in a private scratchpad copy, never in the repository;
nothing outside this file was written.

**Reproduction scripts** (scratchpad, not in the repo):
`/private/tmp/claude-501/-Users-fabriziosilvestri-Documents-Codice-AIML-Course/f3a0270e-32b0-470f-a72d-bbb10b6a91ef/scratchpad/s1.py`
(fast checks) and `.../s2.py` (CV + three grid searches + bootstrap, 3 min).

---

### Claim A — §5 passes `cv=5`, which builds an unshuffled `KFold`, one section after §4 teaches that this is wrong

**Verdict:** CONFIRMED

**Evidence:** `notebooks/lecture-02.ipynb` cell 23 line 8: `grid, cv=5,
scoring="neg_root_mean_squared_error", n_jobs=-1)`. Cell 15 markdown, two
sections earlier: `` `shuffle=True` is not decoration. The default `KFold` does
**not** shuffle ``. Ran all three searches on the same grid, same seed, same
pipeline:

```
cv=5 (notebook)                    best={'model__max_features': 6, 'model__n_estimators': 200} rmse=$48,613 (48613.17) n_params=15 [26.4s]
KFold(5,shuffle=True,42)           best={'model__max_features': 8, 'model__n_estimators': 200} rmse=$48,629 (48629.49) n_params=15 [26.4s]
KFold(10,shuffle=True,42) (deck)   best={'model__max_features': 8, 'model__n_estimators': 200} rmse=$48,180 (48179.64) n_params=15 [66.8s]
```

Every figure in the report's table reproduced exactly. The shuffle, not the fold
count, changes the winner: five *shuffled* folds already pick 8.

**Severity:** misleads a student
**Origin:** generated code
**Fix:** `cv=KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)` in cell 23.

---

### Claim B — the notebook and its own slide deck disagree on the answer, and the markdown explains the difference away as wall clock

**Verdict:** CONFIRMED

**Evidence:** `slides/lecture-02.html` lines 814 and 816 print
`{'model__max_features': 8, 'model__n_estimators': 200}` and
`48179.64299597594`; line 836's figure alt-text says *"the best cell (8, 200)
outlined in red"*. The notebook produces `max_features: 6`, `$48,613` (claim A).
The notebook's explanation, cell 21 markdown: *"The lecture's figure uses `cv=10`,
which takes twice as long; five is enough here"*, and cell 22 prompt box: *"Five
is enough to choose between these fifteen and it halves the wall clock"*.

Fold count is not the cause: `KFold(5, shuffle=True, 42)` also picks 8 (claim A
table). The wall-clock half of the sentence is roughly right (26.4 s vs 66.8 s,
2.5×); the causal half is wrong, and it is the half a student acts on.

**Severity:** misleads a student
**Origin:** hand-written prose
**Fix:** delete the "five is enough" explanation; state that unshuffled folds
select a different model and shuffle instead.

---

### Claim C — §6 measures the tuned forest on the rows it was fitted to

**Verdict:** CONFIRMED — the defect, not the two numbers quoted for it

**Evidence:** cell 26 lines 1–2 verbatim:

```python
best = search.best_estimator_
pred_train_cv = best.predict(X_train)
```

Nothing is cross-validated; the name says otherwise. Re-derived on the model the
notebook actually builds (`max_features=6`, the `cv=5` winner):

```
max_features=6: TRAIN RMSE $17,670  TEST RMSE $48,817
max_features=8: TRAIN RMSE $17,680  TEST RMSE $49,037
```

The report quotes **$17,680** and **$49,037** — those are the `max_features=8`
model's figures, i.e. the deck's model, not the notebook's. The notebook's own
numbers are **$17,670** and **$48,817**. The substance is unchanged (2.76× rather
than 2.8× optimistic) but the report has itself compared two different models,
which is the error it is reporting.

Group ordering also differs between the two tables, as claimed — my run, cell
26's table vs the same code on test rows:

```
TRAIN  ISLAND 2 $70,137 | NEAR OCEAN 2089 $21,709 | NEAR BAY 1846 $20,328 | <1H OCEAN 7274 $18,414 | INLAND 5301 $13,259
TEST   ISLAND 3 $68,254 | NEAR BAY   444 $59,804 | NEAR OCEAN 569 $57,483 | <1H OCEAN 1862 $48,707 | INLAND 1250 $39,373
```

(The report's "train worst = NEAR OCEAN, test worst = NEAR BAY" is true only if
ISLAND is set aside; ISLAND is worst in both.) The deck does this table on the
**test** set — `slides/lecture-02.html`: *"INLAND 1,250 $39,829 … NEAR BAY 444
$60,195"*, overall $49,037.

**Severity:** misleads a student
**Origin:** generated code
**Fix:** score the group table on `X_test`/`y_test` (or on out-of-fold
predictions) and rename `pred_train_cv`.

---

### Claim D — group RMSEs printed without counts, and a count that contradicts the prose

**Verdict:** CONFIRMED

**Evidence:** cell 26 prints
`err.groupby("ocean", observed=True)["error"].apply(...).apply(lambda v: f"${v:,.0f}")`
— a Series of formatted strings, no `n` column, both tables. Counts re-derived:
ISLAND is **2** districts in train, **3** in test, **5** in California. Cell 27
markdown, immediately below the table: `` `ISLAND` — five districts in the whole
state — is the category the first lecture warned you about. `` The training
table's ISLAND row is therefore an RMSE over two rows sitting under a sentence
saying five.

`slides/lecture-02.html` has the three-column table (*"ISLAND districts | Total
5 | In training 2 | In test 3"*) and the rule the notebook breaks: *"the ISLAND
number is computed from three districts: report it with the count or do not
report it."*

**Severity:** misleads a student
**Origin:** hand-written prose (the "five districts" sentence); the missing `n`
is generated code
**Fix:** `.agg(n="size", rmse=...)` on both groupbys; change the sentence to
"five in California, two in this training set".

---

### Claim E — "with ten folds, some folds contain no ISLAND training row at all" is false

**Verdict:** CONFIRMED

**Evidence:** the sentence appears twice — cell 27 markdown and cell 28's prompt
box `left_open`. Four lines settle it:

```
ISLAND rows in each fold's TRAIN part, KFold(10, shuffle=True, random_state=42):
  [2, 1, 2, 2, 2, 2, 2, 1, 2, 2]   zero-folds: 0
same for the unshuffled cv=5 of §5:
  [2, 2, 2, 1, 1]
```

Zero folds train without ISLAND, on either splitter. Folds 1 and 7 (0-based) get
one of the two; the rest get both — exactly as the report says. The deck states
the correct version: *"With our seed no fold loses it; over 500 reshufflings,
11.2% produce at least one fold in which the encoder is fitted without ISLAND."*

**Severity:** misleads a student
**Origin:** hand-written prose
**Fix:** replace with "in two of the ten folds the model trains on a *single*
ISLAND district"; or quote the deck's 11.2%-over-reshufflings figure.

---

### Claim F — the `observed=True` bullet describes a failure that cannot occur here

**Verdict:** CONFIRMED, with one qualification on the word *invented*

**Evidence:** cell 25's `catch` bullet: *"`observed=True` on the groupby. Without
it pandas produces a row for every unobserved combination of categories, full of
NaN, and the table becomes unreadable."* Both groupbys run with `observed=False`:

```
ocean       dtype=object    observed=True: rows=5 NaN=0    observed=False: rows=5 NaN=0
income_cat  dtype=category  observed=True: rows=5 NaN=0    observed=False: rows=5 NaN=0
index in both cases: ['<1H OCEAN','INLAND','ISLAND','NEAR BAY','NEAR OCEAN'] / [1,2,3,4,5]
```

`observed` has no effect on an `object` grouper at all, and all five income bands
are observed. No NaN row is produced either way, and "every unobserved
*combination*" needs two or more categorical keys — both groupbys have one.

*Qualification:* the mechanism is a real, documented pandas behaviour (a
categorical grouper with unobserved levels does emit NaN rows under
`observed=False`). It is not invented; it simply cannot fire on this data. §6.2
is still breached — the bullet is a plausible failure rather than an observed
one — but "invented library default" overstates it.

**Severity:** wrong but harmless
**Origin:** hand-written prose
**Fix:** cut the bullet, or point it at a grouper where it bites (two categorical
keys).

---

### Claim G — the `np.linalg.inv` claim is backwards, and appears twice

**Verdict:** CONFIRMED

**Evidence:** the sentence appears in cell 9 (prompt box `catch`) and cell 11
(markdown), so a reader meets it twice. `np.linalg.inv` computes an inverse via
LAPACK `gesv`, not a pseudoinverse via SVD — its docstring contains no mention of
`pinv`. Measured on the design matrix plus one exactly collinear column
(`X[:,3]*2 - X[:,6]`):

```
cond(XtX) = 1.61e+17
np.linalg.inv: no exception, no warning      ||theta|| = 1.808e+06   training RMSE $160,835
LinearRegression (scipy.linalg.lstsq):                                training RMSE $69,188
constant predictor (predict y.mean()):                                training RMSE $115,311
orthogonality ratio: 3.44e-11 (clean)  ->  1.071e+04 (collinear); the assert fires
```

So it is `inv` that silently returns an answer — one worse than predicting a
constant — and scikit-learn that returns the safe minimum-norm solution. The
notebook credits `inv` with the pseudoinverse and presents "still returns an
answer" as the safe behaviour. Backwards on both counts.

**Severity:** misleads a student
**Origin:** hand-written prose
**Fix:** swap the subjects — `np.linalg.inv` is the unsafe one; scikit-learn's
`lstsq` is what survives a singular `XᵀX`. Fix both copies.

---

### Claim H — a cross-reference to a warning that was never given

**Verdict:** CONFIRMED

**Evidence:** cell 11: *"…it is why you were warned against engineering a feature
as a weighted sum of existing ones."* Searched both the source and the artefact:

```
tools/make_notebooks.py::lecture_01   'weighted sum': 0  'collinear': 0  'linear combination': 0
notebooks/lecture-01.ipynb            'weighted sum': 0  'collinear': 0  'linear combination': 0  'engineering a feature': 0
```

The ISLAND cross-reference does resolve — lecture 1 warns twice (a prompt box
`left_open` and a markdown cell). But lecture 1 promises the all-zero encoding
*"two sections into the next lecture"* and it lands in section **6** of eight.

Cell 24's *"Two things the previous lecture promised and never did"*: lecture 1
promises the three training RMSEs will be diagnosed (*"two of these numbers are
meaningless… the diagnosis is the next lecture"*) and that ISLAND comes back. It
never promises an error breakdown by group — `'breakdown': 0`, `'by group': 0` in
both lecture-1 files. So one of the two things is not findable (§7.3).

**Severity:** wrong but harmless
**Origin:** hand-written prose
**Fix:** delete the "weighted sum" clause; reword §6's opener to name the one
thing lecture 1 actually promised.

---

### Claim I — a conclusion printed as prose instead of being computed

**Verdict:** CONFIRMED

**Evidence:** cell 32, last statement, unconditional:

```python
print("The two agree within the interval. A gap smaller than the interval is "
      "not evidence of anything.")
```

Ran the cell's own code: test RMSE **$48,816.57**, 95% percentile interval
**$46,567 – $51,152**, cross-validated estimate **$48,613**, `lo <= cv <= hi` →
`True`. Both of the report's figures for this claim reproduce exactly. So the
sentence is currently true, and it will print unchanged on any seed where it is
not.

**Severity:** misleads a student
**Origin:** hand-written prose (an English conclusion hard-coded inside a code cell)
**Fix:** `print(f"CV estimate inside the interval: {lo <= cv_estimate <= hi}")`.

---

### Claim J — "bootstrap the squared errors, then square-root" is a free choice for `method="percentile"`, not a correctness requirement

**Verdict:** CONFIRMED, with one nuance on the BCa aside

**Evidence:** cell 31's `catch` bullet insists on the ordering. Ran both orderings
on the 4,128 test squared errors, same seed, all three methods:

```
percentile  sqrt(ci_mean)=46,566.5801,51,152.2908   ci_rmse=46,566.5801,51,152.2908   allclose=True   |diff|=[1.6e-06, 5.8e-09]
BCa         sqrt(ci_mean)=46,675.1434,51,310.0028   ci_rmse=46,675.1556,51,310.4016   allclose=True   |diff|=[0.012, 0.399]
basic       sqrt(ci_mean)=46,363.3270,50,967.3284   ci_rmse=46,480.8485,51,066.5593   allclose=False  |diff|=[117.5, 99.2]
```

For `percentile` the two orderings are identical to machine precision, as
claimed; only `basic` differs. *Nuance:* the report's parenthetical "and so, as
it happens, is scipy's BCa" holds only to `np.allclose` tolerance — BCa's
endpoints differ by $0.01 and $0.40 (relative 8e-6), not machine precision,
because the acceleration is estimated by jackknife on the transformed statistic.
Immaterial to the defect; worth not repeating as "machine precision".

Also confirmed: scipy's default is `method='BCa'`, and BCa here gives $46,675 –
$51,310 against percentile's $46,567 – $51,152.

**Severity:** misleads a student
**Origin:** hand-written prose
**Fix:** state that the ordering is free under the percentile bootstrap and name
`basic`/the *t*-interval as where it is not.

---

### Claim K — timings with no machine attached, and mutually inconsistent

**Verdict:** CONFIRMED (measured half); the Colab column is UNVERIFIABLE

**Evidence:** cell 15: *"⏱ **about 90 seconds** — thirty fits in total, ten of
them forests"*; cell 21: *"⏱ **2–4 minutes.** Fifteen combinations × five folds =
75 forest fits."* Both arithmetic counts check out (3 models × 10 folds = 30 fits,
10 of them forests; 15 × 5 = 75). Neither names a machine — the string "Colab"
appears in the notebook only in the header, the setup prompt box and the setup
cell's `%pip` hint, and "Timings" appears **zero** times, so the deck's convention
(`slides/lecture-02.html`: *"Timings are for a free Colab CPU runtime and are not
examinable"*) is not carried over.

Measured here (Apple silicon, `n_jobs=-1`):

```
10-fold CV cell:  linear 0.2 s, tree 1.1 s, forest 6.8 s   TOTAL 8.1 s
grid search cv=5:                                          26.4 s
grid search cv=10 (the deck's, 150 fits):                  66.8 s
```

The deck's two anchors do disagree: `# about 25 s` for the tree-only 10-fold
(1.1 s here → 23×) against `# 15 x 10 = 150 fits, about 4 minutes` (66.8 s here →
3.6×). That is a 6× inconsistency between the two anchors the notebook inherits,
close to the report's "roughly 5×".

I have no Colab session, so the report's Colab ranges (2–4 min for the CV cell,
10–20 min for the grid) are untested by me — the report labels them as estimates
and that labelling is honest.

**Severity:** misleads a student (a reader told "2–4 minutes" interrupts a working
cell — §7.1's founding evidence)
**Origin:** hand-written prose
**Fix:** name the machine beside every ⏱, and adopt the deck's "timings are for a
free Colab CPU runtime" sentence.

---

### Claim L — the header promises a `⚠ read before running` marker that no cell carries

**Verdict:** CONFIRMED

**Evidence:** scanned all 35 cells:

```
'read before running' : cell 0 (markdown) only — "Cells marked **⚠ read before running** contain a defect on purpose."
'⚠'                   : cell 0 (that same sentence); cell 23 (code) — print("\n⚠ the winner sits on the EDGE of the grid …")
```

One occurrence, in the promise itself. No cell is marked, and the notebook's one
real defect (claim A, `cv=5`) sits unmarked in cell 23.

**Severity:** wrong but harmless
**Origin:** notebook structure (the shared `header()` template in
`tools/make_notebooks.py:51`)
**Fix:** mark cell 23, or drop the sentence from lecture 2's header.

---

### Claim M — examinability marked on one section of eight

**Verdict:** CONFIRMED

**Evidence:**

```
'examinable': cell 2 (markdown, setup prompt box) — "not examinable, and it is here because a version mismatch…"
              cell 3 (code, setup)                 — "# Not examinable: this is engineering hygiene…"
```

Two occurrences, both about §1. Sections 2–8 carry no marking, against §8.3's
"every section gets one of: *examinable*, *not examinable — engineering*, or
*beyond the book*".

**Severity:** wrong but harmless
**Origin:** notebook structure
**Fix:** one line per section heading.

---

### Claim N — instructions a student alone cannot carry out, and no re-run order anywhere

**Verdict:** CONFIRMED

**Evidence:** cell 34: *"Swap notebooks with the team beside you. Ten minutes.
Five questions:"* — the whole of §8. Searched the notebook for exercises:
`'Exercise'`: 0, `'exercise'`: 0, `'rerun'`: 0; the only `'re-run'` is the
loader prompt box's *"delete `datasets/` and re-run"*. There is no exercise, so
a fortiori none states a re-run order.

The report's reconstruction of the one plausible exercise checks out against the
cell index: `RANDOM_STATE` is cell 3, the split cell 7, the 10-fold CV cell 17,
the paired comparison cell 20, the grid cell 23, the group table cell 26, the
test set cell 32 — seven cells, exactly as listed.

**Severity:** misleads a student (the reader alone at home cannot perform §8 at all)
**Origin:** notebook structure
**Fix:** give §8 a solo version, and add one exercise with its re-run order stated.

---

### Claim O — the import cell's own claim, and a dead import

**Verdict:** CONFIRMED

**Evidence:**

```
cell  7 (code)  line  1: "# Every import this notebook needs, in one place — a notebook that only runs"
cell 32 (code)  line  1: "from scipy import stats"
'plt' anywhere in the notebook: cell 7 line 13 only — "import matplotlib.pyplot as plt"
```

`scipy` is imported twenty-five cells after the cell that claims to hold every
import, and `plt` is bound and never used. (`import matplotlib` in cell 3 is
likewise unused, an extra instance the report does not mention.)

**Severity:** wrong but harmless
**Origin:** generated code
**Fix:** move `from scipy import stats` to cell 7; drop `plt` and `matplotlib`
— nothing in lecture 2 plots.

---

### Claim P — four smaller items

**Verdict:** CONFIRMED (all four)

**Evidence, in the order given:**

1. *Edge check tests `n_estimators` only.* Cell 23 lines 13–17:
   `best_n = search.best_params_["model__n_estimators"]` then
   `if best_n == max(grid["model__n_estimators"])`. `max_features` has edges at 4
   and 12 and is never tested; cell 22's prose calls it *"the edge-of-grid check
   is four lines"* without qualification, and its `check` slot says *"detect
   whether the winner sits on the EDGE of the grid"* — both parameters implied,
   one covered.

2. *`enc.transform([["ISLAND"]])` warns, and is called twice.* Cell 29 calls
   `enc.transform([["ISLAND"]]).toarray()` on lines 3 and 4 — same input, two
   prints. Captured warnings:

   ```
   list form:      [0. 0. 0. 0.]  ['UserWarning: X does not have valid feature names, but OneHotEncoder was fitted with feat…']
   DataFrame form: [0. 0. 0. 0.]  []          categories_ = ['<1H OCEAN','INLAND','NEAR BAY','NEAR OCEAN']
   ```

   A cell whose subject is a silent failure ships an unmentioned warning.

3. *§5 never says whether the tuning was worth it.* $48,613 (5 unshuffled folds)
   against $48,687 (10 shuffled folds) is a $74 difference between two different
   measurements. Re-scored on the same ten shuffled folds:

   ```
   tuned (cv=5 winner, max_features=6): $48,184     untuned 100-tree forest: $48,687     gain $503
   tuned (shuffled winner, max_features=8): $48,180                                      gain $507
   forest fold sd (ddof=0) $2,762, fold range $8,860
   ```

   The report's "$508" is the deck's figure for the `max_features=8` model; the
   notebook's own model gives **$503**. Either way the gain is under a fifth of
   the fold sd. The deck states this (*"150 fits bought us something we cannot
   cleanly distinguish from noise"*, `slides/lecture-02.html`); the notebook drops it.

4. *The tree's training RMSE prints as `$0` via `:,.0f`.* Cell 14 line 6:
   `f"${root_mean_squared_error(y_train, tree.predict(X_train)):,.0f}"`. Measured:
   `tr_rmse == 0.0` is `True`, `repr` is `0.0`, 15,830 leaves for 16,512 rows, and
   all 16,512 preprocessed rows are distinct. The exact-equality assert is
   available and is stronger than a formatted zero.

**Severity:** items 1 and 3 mislead a student; items 2 and 4 are cosmetic
**Origin:** generated code (all four)
**Fix:** test both grid axes for edges; pass a DataFrame and call `transform`
once; add the same-folds comparison to §5; assert `== 0.0`.

---

### Claim Q — the listed checks pass

**Verdict:** CONFIRMED (they do pass — re-run so a later reviewer need not)

**Evidence:**

```
§5.1 markdown lines indented >=4 outside a fence : NONE
§5.2 mis-indented fence markers                  : NONE
§3.1 ```python blocks in markdown                : NONE  (so §3.1 is vacuous)
§4.1 names bound to two kinds of object (AST)    : NONE
assert len(X_train)==16512 and len(X_test)==4128 : present in cell 7, and holds (16512 / 4128)
fold spans, all three models                     : linear $7,656   tree $5,459   forest $8,860
```

The "several thousand dollars" claim in cell 18 is therefore true on all three
models. Additional confirmations from the same run, for the record: the 10 folds
are a partition of `range(16512)` (`True`), fold test sizes are two of 1,652 and
eight of 1,651, `housing_full.shape == (20640, 10)` with 207 missing values in
`total_bedrooms` and no other column, and 201 of the 4,128 test districts sit at
the $500,001 cap.

**Severity:** n/a
**Origin:** n/a
**Fix:** none needed

---

## Summary

```
confirmed: 17   false positive: 0   unverifiable: 0
```

(Sixteen defect claims A–P, all confirmed, plus Q — a claim that six checks pass,
which they do. K is confirmed on its measured half; its Colab column alone is
untested here and is labelled as an estimate in the source report.)

```
of the confirmed, 10 mislead a student
   misleads a student   : A, B, C, D, E, G, I, J, K, N   (10)
   wrong but harmless   : F, H, L, M, O                  (5)
   mixed                : P — sub-items 1 and 3 mislead, 2 and 4 are cosmetic
   n/a                  : Q
origin split — prose: 9   code: 4   structure: 3   (+ Q, n/a)
   prose:     B, D, E, F, G, H, I, J, K
   code:      A, C, O, P (its four sub-items counted as one claim)
   structure: L, M, N
   D is mixed in origin — the "five districts" sentence is prose, the missing
   `n` column is code; counted under prose, its sharper half.
duplicates:
   A and B are one root cause (`cv=5`) seen twice — A is the code choice, B is
   the prose that misexplains it. Two distinct fixes; keep both.
   P.3 (§5 never says whether tuning was worth it) overlaps A/B: all three are
   downstream of the same unshuffled search. Fixing A does not fix P.3.
   C and D are both about the §6 table but are independent defects (wrong rows
   vs missing counts); fixing one leaves the other.
   E and D both concern ISLAND and are independent (fold coverage vs counts).
   G is a single defect that ships in two cells (9 and 11) — one claim, two
   edits. Likewise E's sentence appears in cells 27 and 28.
```

**Three corrections to the Phase A report itself**, all in the same family as the
defects it reports:

1. **Claim C's figures are the wrong model's.** $17,680 / $49,037 belong to
   `max_features=8` — the deck's and the shuffled search's winner. The notebook
   ships `cv=5`, so its own numbers are **$17,670 / $48,817**. Confirmed by
   fitting both: `max_features=6 → train $17,670, test $48,817`;
   `max_features=8 → train $17,680, test $49,037`.
2. **Claim P.3's "$508" is likewise the `max_features=8` figure.** The notebook's
   own tuned model gains **$503** on the same ten shuffled folds. The conclusion
   is unaffected.
3. **Claim J's "and so, as it happens, is scipy's BCa" is true only to
   `np.allclose` tolerance**, not to machine precision — BCa's two orderings
   differ by $0.01 and $0.40. Percentile's agreement *is* machine precision.

Everything else in A–Q reproduced to the digit, including the three grid-search
rows of claim A, the $46,567 – $51,152 interval of claim I, the per-fold ISLAND
counts of claim E, and every string-search claim in F, L, M, N and O.
