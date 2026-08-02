# Lecture 8 — Retrain it and watch it change

**Rebuild-by-prompting script.** Géron chapters 5 & 6 · CoverType · *Fix*.
Mathematical thread: impurity, and why averaging reduces variance.

This is what you type at a Colab keyboard, in order, to rebuild
`notebooks/lecture-08.ipynb`. Type the prompt, read what comes back **before**
you run it, run it, then check the output against **Expect**.

---

## Before you start

**Reference machine for every ⏱ in this file.** Apple M4 Max, 16 cores
(12 performance + 4 efficiency), 128 GB, Python 3.13, scikit-learn 1.7.2, macOS.
Every figure below was measured on that machine with `time.time()` around the
cell body. **Colab's free CPU runtime is 2 vCPU.** Every cell here that fits an
ensemble with `n_jobs=-1` will therefore be several times slower there — assume
4–8×, and time cell 9 against my 5.7 s so you know your own factor before you
reach the two long cells, 13 and 18.

**One thing to fix before you get there.** Cell 18 is the longest cell in the
notebook and `n_jobs=-1` is why: **68 s** with it, **6.8 s** without. Write
`n_jobs=1` in cell 18 and nowhere else. The reason is in that cell's annotation.

**Seed.** `RANDOM_STATE = 42`, set once in cell 1 and never shadowed.

**Data.** `fetch_covtype` caches to `~/scikit_learn_data/covertype` (~5 s cold,
instant afterwards). Nothing is downloaded from anywhere else. The full array is
581,012 × 54 float64 = **251 MB (239 MiB)**, and this notebook loads it
**twice** — once in cell 2 and again in cell 12. `del` both when you are done
with them; the instructions are in those two cells.

**Examinable.** Cells 1–11, 15–17, 19 and 21 are examinable (Géron ch. 5–6).
Cells 12–14 (measuring ρ) are *beyond the book, for context* — the derivation
in cell 5 **is** examinable, the ANOVA estimator is not. Cell 18
(`permutation_importance`) is *beyond the book, for context*; it is here because
the alternative is to ship the biased measurement. Cell 20 (boosting) is
*examinable at the level of the one-paragraph summary only*.

---

## Cell 1 — setup and seed

**Prompt to type:**

> Setup cell for a teaching notebook. Import numpy as np, pandas as pd, sklearn
> and matplotlib. Print the python, scikit-learn, numpy and pandas versions. Set
> a single `RANDOM_STATE = 42` that everything below will use.

**Expect:** four version lines and nothing else. `RANDOM_STATE` bound to 42.

**Assert:** the prompt above will not produce one — add it yourself:
`assert tuple(int(p) for p in sklearn.__version__.split(".")[:2]) >= (1, 4)`.
This notebook needs 1.4 for a concrete reason you will meet in cell 9:
`BaggingClassifier`'s `base_estimator` argument was renamed `estimator` in 1.2
and removed in 1.4, so on an older image the *wrong* spelling is the one that
works and everything you copy from this notebook fails. Assert the floor rather
than printing it; a printed version number is information, not an error.

**Annotate:** short

---

## Cell 2 — rebuild the previous lecture's split

**Prompt to type:**

> Load CoverType with `fetch_covtype(as_frame=True)`. Take a stratified 60,000
> row subsample of it with seed 42, then split *that* 80/20, stratified, same
> seed. Fit a `DecisionTreeClassifier(max_depth=8)` and a most-frequent
> `DummyClassifier` on the training half and print both test accuracies. Delete
> the full frame once you have the split.

**Expect:** `48000` train, `12000` test. Depth-8 tree **73.3%**, constant
baseline **48.8%** (the majority class is 2, Lodgepole Pine). Both numbers must
match to the tenth of a point or the seed did not reproduce.

**Assert:** `assert len(X_train) == 48_000 and len(X_test) == 12_000`

**Annotate:** short

> Note for the rest of the notebook: this tree is `min_samples_leaf=1`, the
> scikit-learn default. Lecture 7 shipped `min_samples_leaf=20` and got **73.0%**
> with 163 leaves; this one gets **73.3%** with **206** leaves. The *split* is
> identical — the *tree* is not. This notebook diagnoses instability and the
> least-constrained tree shows it most clearly.

---

## Cell 3 — Gini and entropy, drawn before they are compared

**Prompt to type:**

> Plot Gini impurity `2p(1-p)` and binary entropy `H(p)` in bits against p on
> one panel, and their difference on a second panel beside it. Put them on the
> same scale before you subtract. Print the largest gap and where it happens.

**Expect:** two panels. Both curves zero at 0 and 1, both maximal at p=0.5.
Printed: `largest gap 0.0545 at p = 0.095`. If you get `nan` you let `p` reach
0 — `np.log2(0)` warns and returns `-inf`, and the NaN then propagates silently
into the difference panel. Start at `1e-9`.

**Assert:** none. `assert np.isfinite(gap).all()` is worth adding.

**Annotate:** short

---

## Cell 4 — does the criterion actually matter

**Prompt to type:**

> Does gini or entropy give a better tree on this data? Fit depth-8 trees with
> each criterion on ten resamples of the training set and report the accuracy
> difference, whether the two trees pick the same root feature, and how often
> they make the same prediction on the test set.

**Expect:**

```
same root feature      10 of 10 resamples
predictions agreeing   85.5%
entropy - gini         -0.41 +/- 0.43 points
resamples entropy won  2 of 10
```

**Assert:** `assert same_root == len(rows)` — the root feature is Elevation
every time, on every resample, under both criteria.

**⏱** 2.9 s on the M4 Max (20 depth-8 fits on 38,400 rows). The current
notebook says "about 40 seconds"; expect something in that range on 2 vCPU.

**Annotate:** full

- **Left open:** the word "resamples" never says *paired*. Nothing in that
  prompt forces both criteria onto the *same* resample, and the code that comes
  back is free to draw a fresh subsample inside each criterion's loop. The
  effect being measured is 0.41 points; the resample-to-resample spread is
  larger than that, so unpaired the answer is noise.
- **The usual student version:** `train_test_split(X_train, y_train,
  train_size=0.8)` with `stratify` left at its default, which is `None`.
  `shuffle` defaults to `True`, so you do get a random 80%, but the class
  proportions now wobble from resample to resample — and CoverType's smallest
  class, Cottonwood/Willow, is **0.473%** of the rows, about 227 of the 48,000.
  That wobble is a second uncontrolled variable on top of the unpaired one.
- **How you would catch it:** the printed sd (**0.43**) is *larger* than the
  printed mean (**0.41**). Under §2.4 that is not a result. What survives is the
  sign count — 8 of 10 resamples favour gini — and the fact that both criteria
  choose the same root every time and agree on 85.5% of predictions. Report the
  effect size beside the sign count, always, and do not spend tuning budget
  here.

---

## Cell 5 — the variance of an average, checked numerically

**Prompt to type:**

> Simulate n predictors with unit variance and common pairwise correlation rho
> by mixing one shared standard normal with n independent ones. Measure the
> variance of their average for rho in 0, 0.3, 0.8 and n in 1, 5, 50, and print
> it next to `rho + (1 - rho)/n`.

**Expect:** a 4-column table, 9 rows. `measured` and `formula` agree to about
three decimals everywhere. Read off the three limits: ρ=0 gives σ²/n; ρ=1 gives
σ² (averaging is a no-op); n→∞ gives ρσ², the floor.

**Assert:** `assert abs(measured - formula) < 0.02` inside the loop, at every ρ
and every n.

**Annotate:** short

> The mixture `sqrt(rho)*shared + sqrt(1-rho)*own` is the point: it produces
> exactly the covariance structure the formula assumes. If the generated code
> builds correlation some other way (a Cholesky factor of a hand-written
> matrix, say), the assert still has to pass.

---

## Cell 6 — twenty nearly-identical training sets

**Prompt to type:**

> Fit twenty depth-8 trees, each on a different stratified 90% subsample of
> `X_train`, keeping the estimator's `random_state` fixed at 42 so only the rows
> change. Keep the trees, their test predictions, their root split and their
> test accuracy. Print the mean, sd and range of the accuracy.

**Expect:** `accuracy mean 73.33% sd 0.26% (72.52% - 73.71%)`. A range of about
**1.2 points** and a standard deviation of about **0.26** points across twenty
refits. Write both down. On the headline alone you would call this model
completely stable.

**Assert:** `assert len(preds) == 20 and all(len(p) == 12_000 for p in preds)`

**⏱** 3.5 s on the M4 Max. The current notebook says "about 30 seconds".

**Annotate:** short

> Say which number you mean. "A spread of about a point" is the **range**;
> the **sd** is 0.26. §1.4 — name the operation.

---

## Cell 7 — the other half of the answer

**Prompt to type:**

> Using those twenty prediction vectors, compute the disagreement rate for every
> pair of trees, and the share of test rows where all twenty agree.

**Expect:**

```
pairwise disagreement  9.0%  (5.0% - 15.3%) over 190 pairs
patches all 20 agree on  71.5%
```

190 = C(20,2). Two trees whose test accuracies differ by a quarter of a point
disagree on about **one prediction in eleven**, and **28.5%** of the test rows
have at least one of the twenty trees dissenting.

**Assert:** `assert len(disagree) == 190`

**Annotate:** short

---

## Cell 8 — what is stable and what is not

**Prompt to type:**

> For those same twenty trees, print which feature each one splits on at the
> root, how many distinct root thresholds there are, the range of leaf counts,
> and the range of how many columns each tree consults.

**Expect:**

```
root feature: {'Elevation': 20}
distinct root thresholds: 4
leaves per tree: 200 - 216
columns consulted: 26 - 31
```

Separate the root **feature** (identical twenty times out of twenty) from the
root **threshold** (four different values). The part you would put on a slide is
the stable part; the part that decides an individual prediction is not.

**Assert:** none.

**Annotate:** short

> The diagnosis, in one line: a stable metric is not a stable model. The root
> split is chosen from 43,200 rows and wins by a wide margin; a node eight
> levels down was chosen from a few hundred, where two candidates are separated
> by a hair.

---

## Cell 9 — bagging

**Prompt to type:**

> Bag 100 unconstrained decision trees on `X_train` — no depth limit, we are
> about to average the variance away. Turn on out-of-bag scoring and print the
> out-of-bag score beside the test score.

**Expect:** `out-of-bag 89.5%`, `test 89.7%`. They must land within a point of
each other.

**Assert:** `assert abs(bag.oob_score_ - bag.score(X_test, y_test)) < 0.01`

**⏱** 5.7 s on the M4 Max with `n_jobs=-1`. The current notebook says "about 40
seconds"; on 2 vCPU that is the right order.

**Annotate:** full

- **Left open:** the prompt says "unconstrained" and never says how many rows
  each member draws. `max_samples` defaults to `1.0`, so each member gets a
  bootstrap the same *size* as `X_train` — 48,000 draws with replacement, which
  is about **63.2%** distinct rows, `1 - (1 - 1/m)^m → 1 - e⁻¹`. Nothing in the
  prompt would have caught a member trained on 10% of the rows.
- **The usual student version:** two real library facts, both of which bite
  here. First, `base_estimator=` was deprecated in scikit-learn 1.2 and
  **removed in 1.4**; the parameter is `estimator=`. Every pre-2023 snippet and
  a good share of assistant completions still emit
  `BaggingClassifier(base_estimator=DecisionTreeClassifier(), …)`, which now
  raises `TypeError: BaggingClassifier.__init__() got an unexpected keyword
  argument 'base_estimator'`. Second, `oob_score` defaults to **`False`**, so
  asking for "the out-of-bag score" without setting it gives
  `AttributeError: 'BaggingClassifier' object has no attribute 'oob_score_'`.
  And `n_estimators` defaults to **10**, not 100.
- **How you would catch it:** oob and test must be close. The out-of-bag score
  replaces the **validation** set, not the test set — it is computed on rows the
  ensemble selected for itself. If oob comes out much *better* than test,
  something in the pipeline saw the out-of-bag rows anyway.

---

## Cell 10 — one member on its own

**Prompt to type:**

> Take `bag.estimators_[0]` and score it on the test set on its own, so I can
> see how much worse a single member is than the ensemble.

**Expect:** one accuracy figure. **Write it down before reading on.** Also print
`bag.classes_` and `member.classes_` side by side — you will want them in a
moment.

**Assert:** none yet.

**Annotate:** full

- **Left open:** nothing in that prompt says what `member.predict` *returns*.
  It is the only thing in the sentence that matters and there is no natural way
  to think of asking it.
- **The usual student version:** the one the prompt above actually produces.
  Every scikit-learn ensemble runs `y` through `np.unique`/`LabelEncoder` and
  hands its members **positions `0..k−1`**, keeping the real labels on the
  ensemble in `classes_`. CoverType's labels are `1..7`. So
  `bag.classes_` prints `[1 2 3 4 5 6 7]`, `member.classes_` prints
  `[0 1 2 3 4 5 6]`, and `(member.predict(X_test) == y_test).mean()` returns
  **7.7%**. No exception. No warning. A small, plausible, entirely wrong number.
  The correct comparison, `bag.classes_[member.predict(X_test).astype(int)]`,
  gives **78.8%**.
- **How you would catch it:** 7.7% is *below* the 14.3% you would get by
  guessing uniformly among seven classes, and below the 48.8% you get by always
  saying "Lodgepole Pine". A member of an 89.7% ensemble cannot be worse than a
  coin. That is the tell, and it is the only one you get. Reviewer question 5 —
  *what is the default I did not ask for?* — print `ensemble.classes_` beside
  `member.classes_` any time you reach inside an ensemble.

**Add the assert once you have seen both numbers:**
`assert mapped > naive, "the mapping should rescue the score"`

> ⚠ Everything downstream that scores a single member — cell 11's member column,
> cell 13's whole correctness array — has to go through `classes_`. Cell 13
> computes ρ from 400 member-level predictions; get this wrong there and ρ is
> computed from noise and nothing raises.

---

## Cell 11 — three ways to decorrelate

**Prompt to type:**

> Fit a random forest and an extra-trees classifier, 100 trees each, on the same
> training set, and print each one's test accuracy beside bagging's. Print the
> accuracy of one member of each as a second column. Do extra-trees both with
> and without the bootstrap.

**Expect:**

| | ensemble | one member |
|---|---|---|
| bagging | 89.7% | 78.8% |
| forest | 88.5% | 73.6% |
| extra | 88.8% | 77.8% |
| extra_bs | 87.9% | 72.8% |

**Assert:** none. (If your member column reads 7–20%, go back to cell 10.)

**⏱** ~1 s for the three fits on the M4 Max. The current notebook says "about 40
seconds for the three".

**Annotate:** full

- **Left open:** the prompt says "extra-trees" and never says whether it
  bootstraps. It does not, by default — so as written, the forest row and the
  extra row differ in **two** things, not one, and the table cannot attribute
  anything.
- **The usual student version:** reading this table as "extra-trees is simply
  better, and it makes its members worse". A real default asymmetry inside
  scikit-learn makes both halves of that wrong.
  `RandomForestClassifier(bootstrap=True)` is the default;
  `ExtraTreesClassifier(bootstrap=False)` is the default. The random thresholds
  **replace** the bootstrap rather than joining it. So `extra`'s member trained
  on all 48,000 distinct rows while `forest`'s member trained on ~63% of them,
  which is why extra's member (77.8%) *beats* forest's (73.6%). Turn the
  bootstrap back on — the `extra_bs` row — and it drops to 72.8%.
  (`max_features="sqrt"` is already the default for both, so writing it changes
  nothing; ⌊√54⌋ = 7 features per node.)
- **How you would catch it:** the second column is not monotone, and the
  notebook's own summary sentence — *"every mechanism that makes the members
  less alike also makes each of them worse"* — is refuted by its own table.
  Compare only rows that differ in one thing: bagging → forest isolates feature
  subsampling; extra → extra_bs isolates the bootstrap. §2.1.

---

## Cell 12 — the setup for measuring ρ

**Prompt to type:**

> ρ is a correlation over the randomness of the whole procedure, including which
> training set you were handed — so I need several *disjoint* training sets, not
> one. Take a permutation of all 581,012 CoverType rows, cut off 6,000 for test,
> then cut 10 disjoint training sets of 15,000 out of what is left. Write a
> function that fits an ensemble of 10 members per training set and records, for
> every member and every test row, whether that member got it right.

**Expect:** no printed table — two function definitions (`make`, `experiment`)
and the index arrays. The correctness array is `(10, 10, 6000)` float32 ≈ 2.4 MB.

**Assert:** `assert len(pool) == K * N_Z and len(set(pool) & set(te)) == 0`

**⏱** 1.0 s on the M4 Max — the cost here is `fetch_covtype(as_frame=False)` off
the local cache. **The current notebook puts a "⏱ about 90 seconds" marker on
this cell.** It is on the wrong cell: the work is in cell 13.

**Annotate:** short

> Two warnings for a free Colab runtime. (a) The disjointness half of that
> assert can never fail — a permutation is disjoint from itself by construction.
> Only `len(pool) == K * N_Z` is a real check. (b) `full` is a second complete
> copy of CoverType, 581,012 × 54 float64 = **251 MB (239 MiB)**, held alongside
> `X_train`/`X_test`. Cell 2 deleted exactly this to protect the runtime. Nothing
> after cell 13 uses it — add `del full` there and reclaim it.

---

## Cell 13 — ρ, σ², and the prediction

**Prompt to type:**

> From that correctness array, split one member's variance into the part its
> training set explains and the part it does not, and get ρ and σ² out. Then
> print, for each of bagging, forest, extra and extra-with-bootstrap: ρ, σ², the
> measured variance of an average of 1 member and of 10, and what
> `rho*sigma2 + (1-rho)*sigma2/10` predicts.

**Expect:**

```
                rho   sigma^2      V(1)     V(10)  predicted
bagging       0.078    0.1616    0.1622    0.0266     0.0275
forest        0.052    0.1740    0.1730    0.0240     0.0255
extra         0.072    0.1710    0.1701    0.0270     0.0281
extra_bs      0.051    0.1792    0.1757    0.0244     0.0262
```

The last two columns are the whole lecture: **measured** variance of an average
of ten, against the formula evaluated at n=10, computed by entirely separate
routes. They agree to within **3–7%** of each other on every row (3.4, 5.9, 3.9,
6.9 percent respectively) and nothing was fitted to make them.

**Assert:** none — and that is the point. Adding
`assert abs(V10 - predicted) / predicted < 0.15` afterwards is legitimate; do
not add it before you have looked.

**⏱ This is the expensive cell.** 10.6 s on the M4 Max — **8.3 s of it is the
bagging variant alone** (100 unconstrained trees fitted 10 times over), against
0.6–0.8 s for each of the three that subsample features. On 2 vCPU budget
**1–2 minutes**, most of it bagging. The current notebook gives this cell **no
⏱ marker at all**.

**Annotate:** full

- **Left open:** *which* variance estimator. Between-group variance is not τ²;
  it already contains `within/M`. Get that wrong and ρ comes out roughly
  `1/M = 10%` too high with no symptom.
- **The usual student version:** two real defaults, both silent. First,
  `np.var` defaults to **`ddof=0`** while `pandas.Series.var` defaults to
  **`ddof=1`** — with K=10 groups that is a factor 10/9 on the between-group
  term, about 11% straight onto ρ, and which one you get depends on whether the
  array happened to stay a DataFrame. Both `.var()` calls here pass `ddof=1`
  explicitly for that reason. Second, the naive route: computing ρ as the plain
  correlation between two members' 0/1 correctness vectors. That double-counts
  the within-group term and gives a number that does not predict the curve at
  all.
- **How you would catch it:** the ANOVA estimator `between - within/M` is
  *unbiased*, not non-negative — on K=10 groups it goes negative on individual
  test rows routinely, which is why it is clamped with `np.maximum(·, 0)`. The
  clamp biases ρ **upward**, and the smaller K is the worse that gets. Report K
  next to ρ. Then check the prediction: V(10) against the formula is the only
  evidence that any of this was estimated correctly.

> **What the ρ column says, read honestly.** bagging 0.078 → forest 0.052 →
> extra 0.072 → extra_bs 0.051. Removing the bootstrap (`extra`) puts ρ nearly
> back to bagging's; adding it back (`extra_bs`) recovers the forest's value.
> On these numbers the **bootstrap** does about 3.5× more decorrelation than
> feature subsampling plus random thresholds combined. And ρ ≈ 0.05–0.08 is
> *small*: V(1)≈0.17 falls to V(10)≈0.025, a factor of **6.1 to 7.2** out of a
> theoretical maximum of 10, so on these ten disjoint training sets averaging is
> doing most of what it could possibly do.

---

## Cell 14 — the curve and its floor

**Prompt to type:**

> Plot the measured variance against n for all four variants on a log y-axis,
> overlay the fitted `tau2 + within/n` curve for each, and draw each variant's
> ρσ² floor as a horizontal dotted line.

**Expect:** four sets of markers at n = 1, 2, 5, 10, four smooth curves through
them, and four dotted floors at τ² = 0.0126 (bagging), 0.0090 (forest), 0.0123
(extra), 0.0091 (extra_bs) — all four fit on one log axis with V(1) ≈ 0.17.
The curves flatten onto
*different floors*, not towards zero. On these data the floors are low, so the
gap between V(10) and the floor is still worth having — but it is the gap, not
the number of trees, that tells you whether more members will buy anything.

**Assert:** none.

**Annotate:** short

> The strongest evidence a notebook can offer is two independently computed
> things landing on top of each other. Plot the measured points *and* the
> predicted curve on the same axes, never one or the other.

---

## Cell 15 — now the bill

**Prompt to type:**

> Print the number of leaves and the test accuracy of the depth-8 tree beside the
> total number of leaves across all 100 trees of the forest.

**Expect:**

```
depth-8 tree            206 leaves   73.3%
100-tree forest     749,170 leaves   88.5%
```

**3,637×** more leaves. The justification for one prediction is now 100 decision
paths and a vote.

**Assert:** none.

**Annotate:** short

> The brief asked for a model whose every prediction comes with a human-readable
> justification. We have built one that is 15 points more accurate and cannot
> supply one. When a fix improves the metric, check what it did to the
> **requirement** — they are different objects and only one was written down by
> the customer.

---

## Cell 16 — what the assistant returns

**Prompt to type:**

> The forest replaced our decision tree. Show me which features it relies on, so
> I can put that in the report to the regulator.

**Expect:** the eight largest `feature_importances_`:

```
Elevation                             0.233
Horizontal_Distance_To_Roadways       0.101
Horizontal_Distance_To_Fire_Points    0.093
Horizontal_Distance_To_Hydrology      0.061
Vertical_Distance_To_Hydrology        0.059
Aspect                                0.056
Hillshade_Noon                        0.052
Hillshade_3pm                         0.051
```

It runs, it is instant, and the top of the list is entirely sensible: elevation
decides which trees grow where, and everyone in the room already believed that.

**Assert:** none.

**Annotate:** short

---

## Cell 17 — the control

Before you type this: **guess where a column of uniform random numbers would
rank among the 55.** Write the number down.

**Prompt to type:**

> Add a column of uniform random numbers to both the training and the test frame,
> refit the same forest, and tell me where that column ranks in
> `feature_importances_`.

**Expect:**

```
random_decoy ranks 10 of 55  (importance 0.0399)
45 real columns rank below it
```

The nine above it are nine of the ten continuous columns; the one continuous
column the decoy beats is **`Slope`** (0.0391). It also outranks every one of
the 44 binary columns — the highest of those, `Wilderness_Area_3`, is 12th at
0.0342. Adding a column of pure noise also cost the forest **0.8 points** of
test accuracy, 88.5% → 87.7%, which no one asked about and nothing printed.

**Assert:** `assert rank < 30, "expected the decoy to rank absurdly high"` — the
assert *is* the finding, not a sanity check.

**⏱** 0.5 s on the M4 Max. The current notebook says "about 20 seconds".

**Annotate:** full

- **Left open:** nothing in the prompt says what a *correct* answer would look
  like. That is the entire review question, and no ranking answers it about
  itself.
- **The usual student version:** guessing the decoy ranks last, or near it —
  which is what almost everyone writes down. It ranks **10th of 55**. This is
  not a quirk of this run; it is scikit-learn's own documented warning about
  `feature_importances_`, in two parts. (a) It is computed **on training-set
  statistics**: it sums the weighted impurity reduction each split achieved on
  the rows that *chose* that split, so a split that reduces impurity there does
  so whether or not the feature carries information. (b) It is **biased towards
  high-cardinality features**: a continuous column offers thousands of candidate
  thresholds and a 0/1 indicator offers one. Our decoy is continuous and **44 of
  the 54 real columns take only two values** (4 wilderness-area flags + 40
  soil-type flags). The 45 columns below the decoy are those 44 binaries plus
  exactly one of the ten continuous columns.
- **How you would catch it:** you add the control. It is one line, it costs half
  a second, and it converts an unfalsifiable ranking into a measurement. Add it
  to every importance table you produce. Note the constraint the prompt above
  *does* carry — the decoy goes into **both** frames and the model is **refit**;
  a control added to one side only is not a control.

---

## Cell 18 — the repair

**Prompt to type:**

> Redo the importances properly: permute one column at a time of 3,000 held-out
> rows, re-score the fitted forest, and report the mean drop and its standard
> deviation. Show me where the decoy lands.

**Expect:**

```
                                      mean      sd
Elevation                           0.2841  0.0081
Horizontal_Distance_To_Roadways     0.0891  0.0008
Horizontal_Distance_To_Fire_Points  0.0664  0.0004
Horizontal_Distance_To_Hydrology    0.0327  0.0003
Wilderness_Area_0                   0.0238  0.0046
Vertical_Distance_To_Hydrology      0.0221  0.0024
Hillshade_Noon                      0.0220  0.0012
Wilderness_Area_3                   0.0187  0.0017

random_decoy   mean -0.0   sd 0.00223
```

The decoy is **10th of 55** on the impurity ranking and **indistinguishable from
zero** here — its mean is negative and its standard deviation is wider than
itself. Two wilderness-area flags that are absent from the impurity top eight
(`Wilderness_Area_3` is 12th there, at 0.0342) are now in the permutation top
eight.

**Assert:** `assert abs(pi.loc["random_decoy", "mean"]) < 3 * pi.loc["random_decoy", "sd"]`

**⏱ The longest cell in the notebook, and the only one where a fast machine does
not save you — because of `n_jobs`.** Measured on the M4 Max, same call, same
data:

| | wall clock |
|---|---|
| `n_jobs=-1` (16 cores), as the notebook writes it | **68 s**, and **141 s** on a second call in the same process |
| `n_jobs=1` | **6.8 s** |

Parallelism makes this cell **10–20× slower**. `permutation_importance` farms out
one task per column, and each task carries the whole 100-tree, ~750,000-leaf
forest across a process boundary — 55 columns' worth of pickling to save 55
column shuffles. **Type `n_jobs=1` here.** For comparison, fitting the entire
100-tree bagging ensemble on 16× as many rows (cell 9) takes 5.7 s. The current
notebook says "about 60 seconds", writes `n_jobs=-1`, and names no machine.

**Annotate:** full

- **Left open:** how many repeats. `permutation_importance(n_repeats=…)` defaults
  to **5**; the notebook uses **3** for speed, and that is a decision about the
  precision of the standard deviation you are about to report, not a free
  saving. Also open: `scoring` defaults to `None`, which means "the estimator's
  own `.score`" — accuracy here. On a 7-class problem whose smallest class is 0.473%
  of the rows, that is a choice, not a neutral default. And `n_jobs` defaults to
  `None`, i.e. **one job** — the notebook's `n_jobs=-1` is an override that
  costs a factor of ten, and the prompt above never asked for it.
- **The usual student version:** two, and one of them raises. The forest was
  refit in cell 17 on the **55-column** `X_tr2`; passing the original 54-column
  `X_test` raises `ValueError: The feature names should match those that were
  passed during fit. Feature names seen at fit time, yet now missing: -
  random_decoy`. That one you find in a second. The one that does **not** raise
  is running it on `X_train` — it produces a clean, plausible, better-looking
  table and reproduces the exact training-set bias the cell exists to repair,
  through a different mechanism. It is the whole defect of impurity importance,
  wearing the name of the fix.
- **How you would catch it:** report the standard deviation, always. An
  importance of `0.002 ± 0.004` is zero, and the mean on its own reads as a small
  positive effect. Here the decoy's `-0.0 ± 0.00223` is what lets a reader see
  that its bar is not small but **absent**. And check §2.1 while you are there:
  this is scored on rows 0–2,999 of the test set, the impurity number on all
  48,000 training rows — the two panels of cell 19 are not measured on the same
  rows and the caption has to say so.

---

## Cell 19 — the two rankings, side by side

**Prompt to type:**

> Two horizontal bar charts side by side on the same twelve columns and the same
> y order — impurity importance on the left, permutation importance with error
> bars on the right. Colour the decoy differently.

**Expect:** twelve rows, identical order in both panels, taken from the
*impurity* ranking. The decoy's bar is near the top on the left and, on the
right, a bar you cannot see with an error bar straddling zero.

**Assert:** none. Check by eye that both panels use the same twelve labels in the
same order — `sharey=True` will not save you if you re-sort each panel.

**Annotate:** short

> The pair is the argument. Presenting only the permutation panel because it is
> the correct one leaves the reader with just another ranking to be trusted.
>
> | Question | Can importance answer it? |
> |---|---|
> | Which measurements should the survey keep collecting? | yes — this is what it is for |
> | Is the model using a variable it legally must not? | yes, as a screen |
> | Why was **this** parcel refused? | **no** |
> | Would removing this column hurt? | no — remove it, refit, and measure |
>
> Row three is the regulator's actual question.

---

## Cell 20 — boosting, briefly

**Prompt to type:**

> Fit a `HistGradientBoostingClassifier` on the same training rows with early
> stopping on and a learning rate of 0.2, and print its test accuracy beside
> bagging's and the depth-8 tree's. Report how many iterations it actually used.

**Expect:**

```
histogram gradient boosting  78.2%  (11 iterations, early stopping)
bagging                      89.7%
the legible depth-8 tree     73.3%
```

**Assert:** none.

**⏱** 4.9 s on the M4 Max. The current notebook says "about 60 seconds".

**Annotate:** short

> Read the iteration count, not the accuracy. `max_iter=100` with
> `early_stopping=True` is a **ceiling, not a setting**, and with the default
> `validation_fraction=0.1` and `n_iter_no_change=10` it stopped at **11** — so
> 78.2% is what an 11-iteration model scores, not what boosting scores. Raise
> `n_iter_no_change`, or turn early stopping off and watch it, before you quote
> this number anywhere.
>
> Today's ρ formula does **not** apply to boosting. Everything above trains
> members in parallel and averages them; boosting trains them in sequence, each
> correcting its predecessor. The members are neither identically distributed nor
> exchangeable, so neither the derivation nor the intuition transfers, and it
> reduces *bias* rather than variance.

---

## Cell 21 — test the thing we diagnosed

**Prompt to type:**

> Repeat the twenty-subsample experiment with 30-tree random forests instead of
> single trees — the same twenty seeds, the same 90% stratified subsamples — and
> print the pairwise disagreement beside the single tree's.

Predict the answer before you run it.

**Expect:**

```
single tree, pairwise disagreement  9.0%
30-tree forest                      6.5%
patches all 20 forests agree on     78.2%
```

Pairwise disagreement falls from **9.0%** to **6.5%** — a **28%** relative
reduction, on the same twenty subsamples and the same 12,000 test rows.
Unanimity rises from 71.5% to 78.2%. The accuracy gain was 15 points; the
stability gain is real, and it is smaller than the accuracy table would let you
assume.

**Assert:** `assert len(fdis) == 190` — and the seeds must be `1000 + seed` for
`seed in range(20)`, identical to cell 6. A stability comparison across different
subsamples is not a comparison.

**⏱** 11.9 s on the M4 Max (twenty 30-tree forests on 43,200 rows). The current
notebook says "about 2 minutes".

**Annotate:** short

---

## Closing table

Every row scored on the **same 12,000 test rows** — the ones held out in cell 2
and never touched since.

| Reality | Test accuracy | one member | ρ (cell 13) |
|---|---|---|---|
| always "Lodgepole Pine" | 48.8% | — | — |
| the depth-8 tree, `min_samples_leaf=1` | 73.3% | — | — |
| 100 bagged unconstrained trees | **89.7%** | 78.8% | 0.078 |
| 100-tree random forest | 88.5% | 73.6% | 0.052 |
| 100 extra-trees, `bootstrap=False` (the default) | 88.8% | 77.8% | 0.072 |
| 100 extra-trees, `bootstrap=True` | 87.9% | 72.8% | 0.051 |

Note which row wins the accuracy column and which wins the ρ column: **they are
not the same row.** Plain bagging has the highest ρ of the four and the best
accuracy; the forest cuts ρ by a third (0.078 → 0.052) and still loses 1.2
points of accuracy. That is the trade the formula warned about — σ² appears in **both** terms of
ρσ² + (1−ρ)σ²/n, and every mechanism that lowered ρ here also made the members
worse. Lower ρ is not a goal; lower variance of the average is, and ρ is only
half of it.

And per-instance stability, the thing actually diagnosed in cells 6–8, went from
9.0% pairwise disagreement to 6.5% (cell 21) — a real repair, and a smaller one
than the 15-point accuracy jump from tree to forest would let you assume.

---

## Exercises, with the re-run order (§7.2)

Every exercise below lists the cells to re-run, in order, from a **kernel that
has already run cells 1–21 once**. Cells not listed do not need re-running.
Timings marked *(measured)* I ran; timings marked *(scaled)* are the measured
figure for that cell multiplied by the extra work, and are estimates.

1. **Does the criterion conclusion survive more resamples?** Change `range(10)`
   to `range(30)` in cell 4 and re-run **cell 4 only**. ⏱ ~9 s, M4 Max
   *(scaled: 3× cell 4's measured 2.9 s)*. Does the sd still exceed the mean?

2. **Is the instability a depth-8 artefact?** Set `max_depth=None` in cell 6 and
   re-run **6 → 7 → 8**, in that order. Cell 7 reads `preds` from cell 6 and cell
   8 reads `trees` and `roots`; running 7 without 6 silently compares the old
   vectors. ⏱ ~20 s, M4 Max *(scaled from cell 9: 100 unconstrained trees on
   48,000 rows took 5.7 s across 16 cores, so ≈0.9 s per tree, and cell 6 fits
   twenty of them one at a time)*. Expect `get_n_leaves()` in the thousands, not
   200–216.

3. **How much of the decorrelation is the bootstrap?** Add a fifth entry to the
   `ensembles` dict in cell 11 — `RandomForestClassifier(bootstrap=False, …)` —
   and re-run **cell 11 only**. Then add `"forest_nobs"` to the tuple in cell 13
   and re-run **13 → 14**. ⏱ cell 13 ~11 s, M4 Max *(scaled: 10.6 s measured
   plus one more forest-shaped variant at 0.6 s)*; 1–2 min on 2 vCPU. Cell 12
   does *not* need re-running — `pool`, `Xte`, `yte`, `make` and `experiment`
   are all still bound.

4. **Does the decoy's rank depend on its distribution?** Replace
   `rng.random(...)` with `rng.integers(0, 2, ...)` in cell 17 — a *binary*
   decoy — and re-run **17 only**. Predict the rank first: it should fall a long
   way, because the bias is about the number of candidate thresholds a column
   offers. Then try a 10-category decoy and predict again.
   ⚠ Do **not** then re-run cell 18 expecting the old numbers: `rnd2` and
   `X_te2` have both been rebound. Re-run **17 → 18 → 19** together, ⏱ ~8 s
   with `n_jobs=1` in cell 18, ~70 s with `n_jobs=-1` *(measured)*.

5. **Is 3,000 rows enough for cell 18?** Change `sub = slice(0, 3000)` to
   `slice(0, 6000)` and `n_repeats=3` to `5` and re-run **cell 18 only**. ⏱ 3.3×
   the work: ~23 s with `n_jobs=1`, ~4 min with `n_jobs=-1`, M4 Max *(scaled
   from the two measured figures)*. Compare the standard deviations, not the
   means — that is the quantity more rows and more repeats actually buy.

6. **Does ρ depend on how much data each training set gets?** Change `N_Z` from
   15,000 to 5,000 in cell 12 and re-run **12 → 13**. ⏱ ~1 s + ~4 s, M4 Max
   *(scaled: cell 13's 10.6 s at a third of the rows per fit)*. Predict the
   direction first: less data per training set means the training set explains
   more of the variance, so ρ should **rise**. Cell 14 re-plots from the new
   `dec` — re-run it too if you want the figure.

---

## Defects found in the current notebook

`notebooks/lecture-08.ipynb`, 69 cells (21 code, 48 markdown), checked against
`GUIDELINES.md`. Every entry below is marked **[verified]** if I re-derived it
with `python3` against CoverType and the notebook's own code, or **[read]** if it
is a textual/structural fact I checked by parsing the `.ipynb` but did not need
to run.

### Numbers that do not reconcile (§1.1, §1.2)

1. **`member.predict` "getting 20%" is actually 7.7%.** [verified] Cell 30's
   annotation says *"comparing `member.predict(X)` with `y_test` directly,
   getting 20%"*. Running the notebook's own cell 31 code — `bag.estimators_[0]`,
   `bag` fitted with `random_state=42` — gives **0.0765**, i.e. **7.7%**. The
   mapped figure is 78.8%. The lesson is unharmed and arguably sharper (7.7% is
   below the 14.3% of uniform guessing), but the figure in the prose is wrong by
   a factor of 2.6 and appears nowhere in any output.

2. **§11's summary table quotes an ensemble the notebook never fits.**
   [verified] Cell 65: *"bagged, 200 unconstrained trees | 89.8%"*. Cell 28 fits
   `n_estimators=100`. The 100-tree ensemble scores **0.8968 → 89.7%**, not
   89.8%. Two errors in one row: the tree count, and a figure that does not round
   to the printed value.

3. **"forty of the fifty-four real columns are binary" — it is forty-four.**
   [verified] Cells 53 and 55. `X_train.nunique()` gives **44** columns with ≤ 2
   distinct values (4 `Wilderness_Area_*` + 40 `Soil_Type_*`) and **10**
   continuous. The claim undercounts by the four wilderness flags. The argument
   it supports (cardinality bias) is strengthened, not weakened, by the correct
   number.

4. **"the root split is chosen from 48,000 patches".** [verified] Cells 21 and
   25. The twenty trees in §4 are fitted on `train_size=0.9` subsamples —
   **43,200** rows, not 48,000. 48,000 is the full training set, which no tree in
   that experiment ever sees.

5. **"the sign is consistent" contradicts "gini wins on 8 of the 10
   resamples", in the same sentence.** [verified] Cell 12, item 1. Two of the ten
   resamples have the opposite sign — I measured the per-resample differences as
   `[+0.31, −0.47, −0.83, −0.93, +0.19, −0.26, −0.07, −0.58, −0.43, −1.02]`
   points. The 8-of-10 count is correct; "consistent" is not.

6. **A result stated as real that is inside its own noise (§2.4).** [verified]
   The same cell 12 says *"The effect is **real** — the sign is consistent"*.
   Cell 11 prints `entropy - gini` as **−0.41 ± 0.43** points. The standard
   deviation exceeds the mean. §2.4 requires this to be said where the headline
   is stated; the notebook prints both numbers and lets the headline stand — the
   identical shape to the lecture-19 defect §2.4 was written for.

7. **The ρ table contradicts the mechanism the prose attributes it to.**
   [verified] Cell 42: *"Feature subsampling does most of the work; random
   thresholds are largely a substitute for the bootstrap rather than an addition
   to it."* Running cells 39/41 gives ρ = bagging **0.078**, forest **0.052**,
   extra **0.072**, extra_bs **0.051**. Removing the bootstrap (`extra`) returns
   ρ almost to bagging's; restoring it (`extra_bs`) recovers the forest's value.
   The bootstrap accounts for a drop of ~0.021 and feature subsampling plus
   random thresholds together for ~0.006 — the bootstrap does about **3.5×**
   more, the opposite of what the sentence claims. (The neighbouring claim
   *"every variant is below bagging"* does hold.)

8. **"nothing gets ρ near zero, because all four ensembles ultimately saw the
   same rows" — both halves are wrong.** [verified] Cell 42. (a) The measured ρ
   is **0.051–0.078**, which is near zero on any reading; V(1)≈0.17 falls to
   V(10)≈0.025, a factor of **6.1 to 7.2** out of a theoretical maximum of 10.
   (b) The ten
   ensembles per variant are fitted on **disjoint** 15,000-row training sets —
   cell 39 asserts the disjointness two cells earlier. They did *not* see the
   same rows, which is the entire construction of §7.

9. **Cell 36's summary is refuted by the table cell 35 prints.** [verified]
   *"Every mechanism that makes the members less alike also makes each of them
   worse."* Measured member accuracies: bagging 78.8%, forest 73.6%, **extra
   77.8%**, extra_bs 72.8%. Extra-trees members are 4.2 points *better* than
   forest members. The cause is the very default the next paragraph explains
   (`ExtraTreesClassifier(bootstrap=False)`, so its members see all 48,000
   distinct rows) — the notebook has the explanation and still states the
   monotone claim.

10. **The notebook stores no cell outputs at all.** [verified] All 21 code cells
   have `outputs: []`. Under §1.2 every prose figure in this notebook is
   unreconcilable by construction, and `tools/check_notebook_numbers.py` has
   nothing to check against.

### Comparisons on mismatched rows (§2.1)

11. **Cell 19's two importance panels are measured on different rows and the
   caption says they are the same.** [verified] Cell 60: *"Same model, same
   columns, same row order."* True of the columns and the ordering, false of the
   rows: `feature_importances_` (left panel) is accumulated over the **48,000
   training** rows; `permutation_importance` (right panel) is computed on
   `X_te2.iloc[0:3000]` — **3,000 held-out** rows. That difference *is* the
   lesson of the section, and the caption denies it.

12. **The four-row ensemble table varies more than one thing per row.**
   [verified, and see 9] Cell 35 compares bagging (all 54 features per node,
   bootstrap) with forest (7 features, bootstrap), extra (7 features, random
   thresholds, **no** bootstrap) and extra_bs (7 features, random thresholds,
   bootstrap). Only `bagging → forest` and `extra → extra_bs` are one-variable
   comparisons; the prose reads the table as a progression.

13. **"the legible depth-8 tree" is not the tree lecture 7 called that.**
   [verified] Lecture 7 ships `min_samples_leaf=AUDITABLE_LEAF` (= 20) and
   scores **73.02% with 163 leaves**; lecture 8 rebuilds with
   `min_samples_leaf=1` and gets **73.34% with 206 leaves**, then reuses
   lecture 7's label and lecture 7's 73.3% figure in cell 65. Cell 5 prints
   *"same split as the previous lecture"* — the **split** is genuinely
   identical, which makes the substituted **tree** easier to miss. The prompt
   box at cell 4 does disclose the change in its "Left open" bullet.

### Timing and instructions the reader cannot carry out (§7.1)

14. **No ⏱ figure in the notebook names a machine.** [read] Nine cells carry ⏱
   markers (40 s, 40 s, 30 s, 40 s, 90 s, 20 s, 60 s, 60 s, 2 min) and not one
   says on what. §7.1 requires the CPU figure. Measured on an M4 Max /
   16 cores, the true figures are 2.9 s, 5.7 s, 3.5 s, ~1 s, 1.0 s, 0.5 s,
   68 s, 4.9 s, 11.9 s. Eight of the nine are overstated by 5–50×; a reader
   cannot tell whether that means their laptop is fine or their laptop is the
   problem.

15. **The 90-second marker is on the wrong cell, and the expensive cell has no
   marker.** [verified] Cell 37's *"⏱ about 90 seconds"* sits above cell 39,
   which loads the cached array and defines two functions — **1.0 s** measured.
   The work is cell 41, which runs `experiment()` four times: **10.6 s**
   measured (8.3 s of it the bagging variant), and cell 40 carries **no ⏱ at
   all**. This is exactly the §7.1 failure mode: the untimed cell is the long
   one.

16. **`permutation_importance` is the one cell where the estimate is too
   *low*.** [verified] Cell 55 says *"⏱ about 60 seconds"*. Measured **68.1 s**
   with `n_jobs=-1` on 16 cores — 12× longer than fitting the 100-tree bagging
   ensemble on 16× as many rows. On Colab's 2-vCPU free runtime this is the cell
   that will strand a reader, and it is the only one the notebook does not
   overstate.

17. **`n_jobs=-1` in cell 57 makes that cell ten to twenty times slower.**
   [verified] Identical call, identical data, M4 Max: `n_jobs=1` → **6.8 s**;
   `n_jobs=-1` → **68.1 s** on a cold process and **140.9 s** on a second call
   in the same process. `permutation_importance` dispatches one task per column
   and each task carries the fitted forest — 100 trees, 755,163 leaves — across
   a process boundary; 55 crossings to save 55 column shuffles. `n_jobs`
   defaults to `None` (one job), so this is an override the notebook added, and
   it is the single largest wall-clock cost in the notebook. Removing it would
   take the notebook's longest cell from over a minute to under seven seconds.
   Cells 28, 35, 39, 54 and 67 also pass `n_jobs=-1`, and there it genuinely
   helps — the parallel unit is a whole tree fit, not one shuffle.

18. **Peak memory is never stated and the notebook's own advice is reversed.**
   [verified] Cell 5 deletes `cover` with the annotation *"250 MB held for no
   reason is how a free runtime dies three cells later"*. Cell 39 then binds
   `full = fetch_covtype(as_frame=False)` — 581,012 × 54 float64 = **251 MB** —
   alongside the still-live `X_train`/`X_test`, and never frees it. Nothing
   after cell 41 uses `full`.

### Names and state (§4.1, §4.2)

19. **`perm` is bound to two different types.** [verified by parsing the
   notebook] Cell 39: `perm = rng.permutation(len(full.target))`, an
   `ndarray` of 581,012 ints. Cell 57:
   `perm = permutation_importance(...)`, a `sklearn.utils.Bunch`. §4.1
   forbids exactly this, and the notebook's own red-team question 2 tells the
   reader to *"check where `permutation_importance` was computed"* — sending
   them to a name that means two things. `rng` is also rebound three times
   (cells 15, 39, 54) but always to the same type and always re-seeded, so
   that one is harmless.

20. **A vacuous assert presented as a check.** [read] Cell 39:
   `assert len(pool) == K * N_Z and len(set(pool) & set(te)) == 0`. `pool` and
   `te` are disjoint slices of one permutation, so the second conjunct cannot
   fail under any input. §6.3 asks for checks with a knowable outcome; this one
   has a knowable outcome for the wrong reason.

### Staging the defect (§8.1)

21. **The `classes_` trap is announced three times before the reader reaches
   it.** [read] Cell 29 heading *"### One trap, worth ten minutes of your
   life"*, then *"⚠ **Read before running.** This is the shape question, and it
   costs an afternoon the first time"*, then the prompt label *"⚠ the trap that
   costs an afternoon"*, then the constraint *"show the WRONG number first"*,
   then the "Left open" bullet which states the answer outright before the cell
   runs. §8.1's evidence is that nobody falls into a trap flagged four times.
   §8's preferred shape — run it unannounced, write the number down, open the
   *next* section with the ⚠ — is available here at zero cost.

22. **Same pattern on the importance defect.** [read] Cell 49's *"⚠ **Read
   before running.** It runs, it is fast, and the top of the list is entirely
   sensible"* plus the prompt label *"⚠ what the assistant returns"* plus the
   constraint repeating the same sentence. Cell 52's *"Guess where it ranks
   among the 55 before running the cell"* is the good version and it arrives
   after three announcements have already given the game away.

23. **§8.3 — "examinable" appears nowhere in the notebook.** [read] String
   search over all 69 cells: zero occurrences of *examinable* except *"not
   examinable"* in cell 3's comment and cell 55's parenthetical about
   `permutation_importance`. §8.3 requires every section to carry one of the
   three labels; eleven of twelve sections carry none.

### Annotation budget (§6.1)

24. **All 21 prompt boxes carry the full three-bullet annotation.** [read]
   §6.1 caps this at five to eight per notebook. Twenty-one is the defect the
   guideline was written to repair, and the boxes that matter most here — the
   `classes_` trap at cell 30 and the decoy at cell 53 — sit at positions 10
   and 17 of 21, past the point all three audit readers stopped reading.

25. **Two "How you would catch it" bullets are not catches.** [read] Cell 2's
   reads *"not examinable, and it is here because a version mismatch produces a
   confusing error"* — a justification. Cell 4's reads *"`del cover` after
   splitting. 250 MB held for no reason…"* — a memory tip about a different
   line. Neither tells the reader how to detect a wrong answer.

26. **The setup box's constraint cites an API this notebook never uses.**
   [verified] Cell 2: *"`root_mean_squared_error` arrived in 1.4, and on an
   older Colab image the failure is an ImportError twenty cells from here"*.
   `root_mean_squared_error` appears nowhere in lecture 8 — it is a regression
   metric and this is a classification notebook. It is inherited verbatim from
   the shared `SETUP_PROMPT` in `tools/make_notebooks.py`. The version floor is
   still worth asserting; the reason given is false here.

### Cross-references (§3.3)

27. **"the next cell is the control that answers it" points at the biased
   table.** [verified by counting] Cell 50's bullet. Cell 50 is a markdown
   prompt box; the next cell is **51**, which prints the *impurity*
   importances — the thing being criticised. The control is cell **54**, four
   cells later.

28. **"a free runtime dies three cells later".** [verified by counting] Cell 4.
   Three cells later is cell 7, a markdown prompt box. The `del cover` it refers
   to is in cell 5, the very next cell.

29. **"a confusing error twenty cells later" / "twenty cells from here".**
   [verified by counting] Cell 2's constraint and cell 3's comment. Twenty cells
   from cell 3 is cell 23, a markdown box; nothing in this notebook imports
   anything that needs scikit-learn 1.4.

### Checked clean

- **§5.1 / §5.2 — markdown rendering.** [verified by parsing] No markdown line
  in any of the 48 markdown cells is indented ≥ 4 spaces outside a fence, and no
  fence marker is indented at all. The lecture-19 cell-41 failure does not recur
  here. All markdown tables are real markdown tables, not ASCII art.
- **§3.1 — code quoted in prose.** [verified by parsing] There are **no** fenced
  code blocks in any markdown cell of this notebook, so nothing is quoted that
  could fail to exist.
- **§4.2 — idempotency.** [read] Every training cell re-instantiates its
  estimators inside itself. Cell 35 rebuilds the `ensembles` dict from scratch
  (it reuses the already-fitted `bag` object deliberately and refits the other
  three); cells 15, 39 and 54 re-seed `rng` before use. Restart-and-run-all
  ordering is linear; I found no cell that depends on a name defined below it.
- **Reproducibility of the split.** [verified] Lecture 7 and lecture 8 pass the
  same objects, sizes, stratification and seed to `train_test_split`, so cell
  5's *"same split as the previous lecture — the seed guarantees it"* is true.
  `set(X_train.index).isdisjoint(X_test.index)` holds.
- **Arithmetic in prose.** [verified] ⌊√54⌋ = **7**; C(20,2) = **190**;
  1 − e⁻¹ = **0.368**; CoverType has **581,012** rows and **54** columns and
  labels `1..7`; 10 × 15,000 + 6,000 = 156,000 ≤ 581,012; the majority class is
  **2** at **48.76%**; the depth-8 tree has **206** leaves and the 100-tree
  forest **749,170**, a ratio of **3,637**; pairwise disagreement **9.0%** is
  "one prediction in eleven"; the decoy's `assert rank < 30` passes at rank
  **10**; the decoy is *"near the top on the left and indistinguishable from
  zero on the right"* — 10th of 55 at 0.0399, versus −0.0 ± 0.0022.

### Not checked

- **The published ⏱ figures.** I could not test them on their intended machine —
  I have no Colab 2-vCPU runtime here. My timings are M4 Max / 16 cores and are
  stated as such throughout. Defect 14 is that the notebook names no machine at
  all; defects 15 and 16 hold on any machine, because they are about *which cell*
  carries the marker and about a ratio measured on one machine.
- **The lecture's own ρ figure.** Cell 37 says *"the lecture's figure uses 20
  training sets of 20,000 rows and 20 members"*. That figure is not in the
  notebook and I did not run the larger version, so I cannot say whether *"the
  numbers will be close but not identical"* is true. My ρ values are from the
  notebook's own K=10, M=10, N_Z=15,000 configuration.
- **Cell 44's plot, and cells 8, 44 and 59 as figures.** I computed the four
  decompositions cell 44 draws from — the floors are `tau2` = **0.0126**
  (bagging), **0.0090** (forest), **0.0123** (extra), **0.0091** (extra_bs),
  obtained as ρ × σ² from the printed columns — but I did not render any of the
  three figures, so I have not checked axes, legends, error bars or the
  `set_yticks(..., labels, fontsize=7)` call in cell 59 by eye. §10.8 says to
  read the rendered page, and for the figures I did not.
- **Whether a second run reproduces every figure exactly.** I ran each verified
  experiment once. Everything is seeded with `random_state=42` and the joblib
  worker count does not enter any of these estimators' randomness, so I expect
  exact reproduction, but the only thing I ran twice was
  `permutation_importance` — and that one moved from 68.1 s to 140.9 s in wall
  clock (its *numbers* were identical). Treat every ⏱ in this file as one
  measurement, not a mean.
