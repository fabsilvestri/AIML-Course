# Lecture 6 — Reading a learning curve

*A script to follow at a Colab keyboard.* Twenty-nine code cells, in order. Type
the prompt, read what comes back, check it against **Expect**, run the
**Assert**. If the assert fails, the cell is wrong — not the assert.

## Before you start

**Environment.** Every number in this file was measured on **scikit-learn 1.7.2,
numpy 2.3.5, pandas 2.3.3**, Python 3, on an **Apple M4 Max (16 cores)** with
`NJ = 4`. If your scikit-learn is older, the held-out log losses move in the
third decimal and the two encoder-dependent column counts (22 and 143) may move
as well.

**Runtime:** CPU. There is nothing here a GPU would make faster.

**About the ⏱ figures — read this before you trust one.** Two numbers are given
for every slow cell: what it took **measured on an idle M4 Max**, and the
**Colab-scale figure** inherited from the lecture module, which is what a free
2-vCPU runtime should show. They differ by a factor of 10 to 30, and both are
useful: the first tells you the cell is not hung, the second is what you will
actually sit through.

I learned this the hard way while checking the notebook. The 17-fit loop in cell
24 measured **484 s** when three other Python processes were on the box and
**0.6 s** when nothing was — the same code, the same data, an 800× difference
from thread oversubscription alone. If a timing surprises you, close everything
else and take it again before you write it down. This is also the whole argument
for `NJ = 4` rather than `n_jobs=-1`.

**Total wall clock** for a cold restart-and-run-all: **under a minute** on an
idle 16-core laptop; **10–15 minutes** on a free Colab CPU runtime, nearly all
of it in cells 10, 14, 20, 22 and 25.

**Annotation budget (§6.1).** Seven cells carry the full three-bullet
annotation: **7, 8, 10, 14, 19, 22, 25**. Every other cell gets the short
specification box only. The current notebook annotates all twenty-nine, and
that is the defect this rewrite exists to repair.

**Two deliberate departures from the deck**, both stated in the cell that makes
them:

- the bias–variance decomposition is measured at degrees **1, 2, 3 and 5**, not
  all six, because degrees 4 and 6 do not carry the argument: degree 1 is the
  low-variance end and degree 5 is the model the build session committed to.
  Give that reason and not the cost one. Measured on an idle machine, the sweep
  spends 1.8 s on degree 6, 1.4 s on degree 4 and 1.2 s on degree 5, so "degrees
  4 and 6 are the two most expensive" is not what the clock says — degree 5,
  which you are keeping, costs about what degree 4 does. Dropping two of six
  saves a third of the fits, which is the honest arithmetic;
- the tuning-trap experiment uses **8 seeds**, not 20 — about two Colab minutes
  against five (6 s against 15 on the laptop). The conclusion is the same and
  the interval is wider, and the interval is the point.

---

## Cell 1 — versions and the one seed

**Prompt to type:**

> Setup cell for a teaching notebook: import sys, sklearn, numpy as np, pandas
> as pd, matplotlib, print all four versions, and assert scikit-learn is at
> least 1.4. Then set RANDOM_STATE = 42 with a comment saying it is used for
> every split, every model and every shuffle.

**Expect:** four lines, `python … / scikit-learn … / numpy … / pandas …`, and
nothing else. No plot, no dataframe.
**Assert:** `tuple(int(p) for p in sklearn.__version__.split(".")[:2]) >= (1, 4)`
**Annotate:** short

---

## Cell 2 — every import this notebook needs, in one place

**Prompt to type:**

> One import cell with everything else this notebook uses: tarfile,
> urllib.request, warnings, pathlib.Path, matplotlib.pyplot as plt,
> joblib Parallel and delayed, and from sklearn: ColumnTransformer,
> SimpleImputer, LogisticRegression, SGDClassifier, accuracy_score,
> brier_score_loss, log_loss, GridSearchCV, StratifiedKFold, cross_val_score,
> cross_validate, learning_curve, train_test_split, Pipeline, make_pipeline,
> OneHotEncoder, PolynomialFeatures, StandardScaler. Also set NJ = 4 for
> n_jobs.

**Expect:** no output at all. If it prints anything, something imported a
plotting backend it should not have.
**Assert:** none.
**Annotate:** short

> Why the imports are repeated rather than inherited from lecture 5: a notebook
> that only runs because another notebook is still in memory is not
> reproducible. Restart-and-run-all is the test, and it starts from a cold
> kernel.
>
> Why `NJ = 4` and not `n_jobs=-1`: a free Colab CPU runtime has two cores, so
> `-1` buys nothing there, and on a shared machine it makes everything slower by
> oversubscribing.

---

## Cell 3 — the data, rebuilt from scratch

**Prompt to type:**

> Write `load_titanic()`: if `datasets/titanic.tgz` is not on disk, download
> `https://github.com/ageron/data/raw/main/titanic.tgz` and extract it into
> `datasets`, then read `datasets/titanic/train.csv`. Call the result `full`,
> assert the shape, and print how many passengers there are and how many
> survived.

**Expect:** `891 passengers, 342 of whom survived`. Roughly 5 s the first time,
instant afterwards.
**Assert:** `full.shape == (891, 12)` and `full["Survived"].isin([0, 1]).all()`
**Annotate:** short

---

## Cell 4 — the same four engineered columns

**Prompt to type:**

> Add four columns to `full`. `Title`: whatever sits between the comma and the
> full stop in `Name`, with Mlle and Ms mapped to Miss, Mme to Mrs, and
> anything outside {Mr, Mrs, Miss, Master} mapped to Rare. `FamilySize` =
> SibSp + Parch + 1. `IsAlone` = 1 when FamilySize is 1. `Deck` = first letter
> of `Cabin`, "U" where it is missing. Print the Title counts.

**Expect:** `{'Mr': 517, 'Miss': 185, 'Mrs': 126, 'Master': 40, 'Rare': 23}` —
five titles, summing to 891.
**Assert:** `set(full["Title"]) == {"Mr", "Mrs", "Miss", "Master", "Rare"}`, and
`(full["SibSp"] + full["Parch"] + 1 - full["FamilySize"]).abs().max() == 0`
**Annotate:** short

> Keep `FamilySize`. It is an exact linear combination of two other columns and
> we know it — cell 17 needs the column in order to show what that does to
> XᵀX. Do not repair it here. A notebook that exists to repair something and
> repairs it in the setup diagnoses a fault that is no longer present.

---

## Cell 5 — split first, and one column left out on purpose

**Prompt to type:**

> Define NUM = Age, Fare, SibSp, Parch — deliberately without FamilySize —
> CAT = Pclass, Sex, Embarked, Title, Deck, BIN = IsAlone, and ALL = NUM +
> FamilySize + CAT + BIN. Take X = full[ALL], y = full["Survived"], and split
> 20% off for test, stratified on y, random_state=RANDOM_STATE. Assert the two
> sizes and that the indices are disjoint, then print the survival rate on each
> side.

**Expect:** `train rate 0.3834   test rate 0.3855`.
**Assert:** `len(X_train) == 712 and len(X_test) == 179`;
`set(X_train.index).isdisjoint(X_test.index)`
**Annotate:** short

> `ALL` and `NUM` differ by exactly one name. `FamilySize` stays in the frame,
> because cell 17 needs to measure it, and stays out of the model, because that
> is the repair. A column existing and a column being modelled are different
> things, and this is the only cell where the difference is visible.

---

## Cell 6 — the pipeline factory, and the anchor

**Prompt to type:**

> Write `prep(degree)` returning a ColumnTransformer: on NUM, median-impute →
> StandardScaler → PolynomialFeatures(degree, include_bias=False); on CAT,
> most-frequent impute → OneHotEncoder(drop="first", min_frequency=2,
> handle_unknown="infrequent_if_exist"); passthrough on BIN. Then
> `pipeline(degree=1, C=1e6, penalty="l2", solver="lbfgs", l1_ratio=None,
> max_iter=4000)` returning a **new** Pipeline of prep + LogisticRegression on
> every call. Make a 10-fold shuffled StratifiedKFold called `cv`. Finally
> compute the log loss of always predicting the training base rate and print it.

**Expect:** `anchor — report the base rate to everyone: 0.666`
(0.665717 before rounding; the training base rate is 0.3834).
**Assert:** `abs(constant_log_loss - 0.666) < 0.001`
**Annotate:** short

> Recompute the anchor here rather than copying 0.666 from lecture 5. If it
> comes out different, the two notebooks are not on the same data and every
> comparison below is void. It is one line and it is the only evidence you have
> that they agree.
>
> `pipeline()` is a **function**. A single Pipeline object built once at the top
> and refitted in a loop carries the previous iteration's fitted state into the
> next, and nothing warns you.

---

## Cell 7 — the degree sweep, rebuilt

**Prompt to type:**

> For degrees 1 to 6, cross-validate `pipeline(degree=d)` on the training set
> with `scoring="neg_log_loss"`, `cv=cv`, `n_jobs=NJ`, and give me the training
> score as well as the held-out one. Store the number of columns
> `prep(d)` produces alongside both means in a dict called `sweep`, and print a
> line per degree. Suppress the convergence warnings for now.

**Expect:**

| degree | columns | train | held-out |
|---|---|---|---|
| 1 | 22 | 0.395 | 0.468 |
| 2 | 32 | 0.381 | 0.467 |
| 3 | 52 | 0.355 | 0.538 |
| 4 | 87 | 0.310 | 1.390 |
| 5 | 143 | 0.286 | 1.922 |
| 6 | 227 | 0.284 | 1.836 |

The held-out minimum is degree 2 at 0.467, with degree 1 at 0.468. **Print the
fold spread too, and read it before you announce a winner:** the margin between
them is **0.0010**, the paired per-fold difference has a standard deviation of
**0.0288**, and degree 2 wins in **5 of the 10 folds**. That is a coin flip.
Treat the answer as *degree 1 or 2*, which is what the assert below already
says.

**Assert:**
`min(sweep, key=lambda d: sweep[d]["valid"]) in (1, 2)` and
`sweep[5]["valid"] > 2 * constant_log_loss`.
Do **not** assert the third decimal of anything here.
**⏱** **6.0 s measured** on an idle M4 Max (NJ=4); per degree, 1.3 / 0.1 / 0.3 /
1.4 / 1.2 / **1.8** s, and the 1.3 s at degree 1 is mostly joblib starting its
workers. Colab-scale figure: **about 15 s**. Note which degrees the time is in:
**6, then 4 and 5** — not "4 and 6".
**Annotate:** full

- **Left open:** the sign, and the rows. `neg_log_loss` returns negatives, so a
  cell that prints `r["test_score"]` raw shows −0.47 at degree 1 and −1.92 at
  degree 5 and a curve that appears to *improve* with degree. Also: the training
  score is measured on the rows that were fitted, and the held-out score on rows
  that were not — the prompt never says so, and the whole lecture is the
  difference between those two row sets.
- **The usual student version:** leaving out the request for the training score.
  `cross_validate(..., return_train_score=False)` is the scikit-learn default,
  so you get exactly one curve, and one curve cannot show a gap. This is the
  single most consequential default in the notebook.
- **How you would catch it:** ask what your assert does if someone improves the
  pipeline. `assert sweep[5]["valid"] == 1.957` was true when this lecture was
  written and is false today — this same pipeline now scores **1.922** — so it
  would turn a legitimate improvement into a red cell. Assert the claim (a low
  degree wins; degree 5 is worse than saying nothing), never the decimal.

---

## Cell 8 — the term you cannot fix, measured

**Prompt to type:**

> Take the 712 **training** passengers only. Band Age into 0-12, 13-25, 26-40,
> 41-60, 60+ plus an explicit "missing" band, band FamilySize into alone / 2-4 /
> 5+, and group by Sex, Pclass, the age band, the family band and Embarked with
> `observed=True`. For every group holding at least two passengers use the
> unbiased estimator k(m−k)/(m(m−1)) for p(1−p), average it weighted by group
> size, and print how many groups there are, how many hold two or more
> passengers, how many people that is, and the resulting Brier floor.

**Expect:** `122` groups, `97` with two or more passengers, `685` people, floor
**0.122**. Of those 97 groups, 51 are mixed (some lived, some died), covering
507 passengers.
**Assert:** `len(cells) == 122 and len(multi) == 97` and
`int(multi["m"].sum()) == 685` and `abs(noise - 0.122) < 0.001`
**Annotate:** full

- **Left open:** *which passengers*. Run the identical grouping on all 891 and
  you get 133 groups, 858 people and a floor of **0.121**; on the 712 training
  rows, 122 / 685 / **0.122**; on the 179 test rows alone, 81 / 40 / **0.119**,
  from only 138 people — thin, but it agrees. The three numbers agree to about
  0.003, which is the useful
  finding — the floor is a property of the columns, not of a row set — but it is
  a finding you only have because you computed more than one of them. The
  version that uses all 891 rows reads the labels of the test set, in section 2,
  in a notebook whose section 6 opens by saying the test set is untouched.
- **The usual student version:** dropping `observed=True`. Through pandas 2.x
  `observed=False` is still the default for categorical group keys, so the same
  grouping returns **324 groups of which 191 are empty**, and the printed "how
  many cells" is then a statement about the Cartesian product of your bands
  rather than about the ship. The second real default in the same line:
  `groupby(dropna=True)` silently drops the **2** passengers with a missing
  `Embarked` — 889 of 891, and nothing says so. Pass `dropna=False` if you want
  them; either way, print the count you kept.
- **How you would catch it:** a floor computed without fitting anything is the
  only one you can compare a fitted model against. If your "irreducible error"
  moves when you improve the model, it was never irreducible — it was a residual.

---

## Cell 9 — look at the four groups that disagree most

**Prompt to type:**

> From those groups, show the four where identical passengers disagreed most —
> rank by k(m−k), the number of disagreeing pairs, not by group size — printing
> the key, how many died and how many survived.

**Expect:** on the 712 training rows, exactly these four —

```
male · 3 · 26-40 · alone · S               37 died, 11 survived
male · 3 · 13-25 · alone · S               59 died,  5 survived
male · 3 · missing · alone · S             31 died,  3 survived
female · 3 · 13-25 · alone · S              7 died,  6 survived
```

— then a sentence saying no model of these five columns can separate the people
inside one line. (On all 891 the same four groups come out, larger: 49/13,
69/6, 38/4, 8/7.)
**Assert:** none.
**Annotate:** short

> Rank by k(m−k) rather than by m, or you will just rediscover the four biggest
> groups. A floor you can read as a list of real people is a floor you can
> defend in a viva.

---

## Cell 10 — the other two terms

**Prompt to type:**

> Draw 200 training sets of 400 rows each from X_train **without replacement**,
> fit `pipeline(degree=deg)` on each with joblib Parallel(n_jobs=NJ), and
> predict probabilities on X_test. For degrees 1, 2, 3 and 5 report the
> variance term, the bias²+noise term and the total expected Brier score.
> Assert the shape of the prediction matrix before averaging anything.

**Expect:** four lines,
`degree d: total … = bias²+noise … + variance …`. The variance term grows by
more than 10× from degree 1 to degree 5; the bias²+noise term barely moves.
**Assert:** `P.shape == (200, len(X_test))` — inside the loop, before any mean
is taken.
**⏱** **about 2 minutes** (author's figure, NJ=4; not re-measured for this
script). 800 fits. On a 2-vCPU Colab runtime budget 4–5 minutes.
**Annotate:** full

- **Left open:** what the 179 evaluation rows are. They are `X_test`, which
  means this cell reads the test set 800 times before section 6 claims it is
  untouched. Nothing here *selects* on it, so nothing is optimistic — but the
  red-team question in section 7 is "count every **read**", and by that rule
  this cell, cell 8 and cell 24 all count. Write the number down: reads of the
  test set before section 6 are cells **10, 24 and 26**, and cell 8 too if you
  compute the floor on all 891.
- **The usual student version:** dropping the words *without replacement*.
  `rng.choice` has `replace=True` as its **default**, so the natural code gives
  you a bootstrap, and the variance you measure is then partly the variance of
  the resampling scheme. Measured over 2,000 draws of 400 rows from 712 with
  replacement: **306 distinct rows** on average (closed form 712·(1−(711/712)⁴⁰⁰)
  = 306.2), so a fifth of every training set is duplicated rows. Note that the
  familiar 63.2% figure — 253 of 400 — is the *n-from-n* answer and does not
  apply to a 400-from-712 draw.
- **How you would catch it:** the shape assert, and only the shape assert. A
  silently broadcast array here produces a beautifully wrong stackplot, a
  plausible variance term and no error of any kind.

---

## Cell 11 — does the identity actually hold?

**Prompt to type:**

> For each of those four degrees, check that total − variance − (bias²+noise) is
> zero to floating point, and assert it. Then print how much the variance term
> grew from degree 1 to degree 5, and the bias²+noise at degree 1 next to the
> measured noise floor.

**Expect:** `identity holds at machine precision for every degree`, then a
growth factor above 10, then the two comparable numbers.
**Assert:** `resid < 1e-12` per degree, and `growth > 10`.
**Annotate:** short

> These three columns are not a *model* of the error, they are the error
> rearranged, so a residual above floating-point noise is a bug and not a
> modelling choice. Whenever you decompose a quantity, assert that the parts add
> up: one line, and it is the whole difference between a measurement and an
> illustration.
>
> Print the bias² number rather than describing it. "There was never much bias
> to buy back" is a claim about a specific difference — bias²+noise at degree 1
> minus the floor — and it should be computed in the cell, not typed into the
> markdown.

---

## Cell 12 — the decomposition, stacked

**Prompt to type:**

> Stackplot of bias²+noise and variance against degree for those four degrees,
> with the total drawn over it as a line, and a dashed horizontal line at the
> measured noise floor, labelled with its value. Axis labels and a legend.

**Expect:** two bands and a line; the bottom band nearly flat, the top band
opening out to the right; the dashed floor sitting just under the bottom band at
degree 1.
**Assert:** none.
**Annotate:** short

> The floor is drawn as a line because the point of the picture is how much of
> the bottom band is unreachable. If the label says 0.122 it must be the number
> cell 8 printed — read it off the variable, never retype it.

---

## Cell 13 — put it back on the two curves

**Prompt to type:**

> From `sweep`, print held-out minus training log loss at degree 1 and at degree
> 5, then the ratio of the two gaps, and assert the degree-5 gap is more than
> ten times the degree-1 gap.

**Expect:** degree 1 gap **0.073**, degree 5 gap **1.636**, ratio **22**.
**Assert:** `g5 > 10 * g1`
**Annotate:** short

> **The vertical gap between the training and held-out curves is the variance
> term.** That is the sentence to take out of this lecture. The training curve
> is measured on the rows that were fitted, so it carries no variance at all;
> the held-out curve carries all three terms.
>
> The assert says ten and the cell prints twenty-two. Say twenty-two in any
> prose you write: the assert is a floor on the claim, not the claim.

---

## Cell 14 — would more data help?

**Prompt to type:**

> Use `learning_curve` on `pipeline(degree=deg)` for deg in (1, 5) over the
> training set, eight training sizes from 0.12 to 1.0, `cv=cv`,
> `scoring="neg_log_loss"`, `n_jobs=NJ`, `shuffle=True`,
> `random_state=RANDOM_STATE`. Store n, the mean training score and the mean
> held-out score per degree, and print the first and last held-out score and the
> final gap for each.

**Expect:** eight sizes, `[76 157 237 318 398 479 559 640]`. At 640 rows the
degree-1 curves have met (gap ≈ 0.07); the degree-5 curves are about **1.6**
apart and the held-out one is still falling.
**Assert:** `len(n) == 8 and n[-1] == 640`
**⏱** **about 30 s** (author's figure; the cell fits 8 sizes × 10 folds × 2
degrees). Budget 1–2 minutes on Colab.
**Annotate:** full

- **Left open:** what the eight fractions are fractions *of*. Not of your 712
  training rows: `learning_curve` takes them as fractions of the largest
  training set available **inside a fold**, which with 10-fold CV is
  712 × 9/10 = **640**. So `0.12` is 76 passengers, not 85. The cell's own
  assert (`n[-1] == 640`) proves it, and the current notebook's annotation says
  85 in the box directly above that assert. Whenever a fraction and an absolute
  count appear in the same cell, divide one by the other before you believe
  either.
- **The usual student version:** omitting `shuffle=True`. It is `False` by
  default, so each sub-sample is the *first n rows of the frame in the order the
  frame happens to be in*. The curve comes out smooth, the shape is wrong, and
  nothing warns you. On a frame sorted by PassengerId the effect is mild, which
  is worse than if it were violent.
- **How you would catch it:** ask what the leftmost point is measured on. 76
  passengers, ten folds, one held-out log loss each — that point is noise, and
  reading a trend through it is the commonest way a learning curve lies. Plot
  the fold spread, or start the x-axis where you would be willing to defend the
  estimate.

---

## Cell 15 — two learning curves, one y-axis

**Prompt to type:**

> Two panels side by side, degree 1 and degree 5, training and held-out log loss
> against number of training passengers, `sharey=True`, titles "high bias" and
> "high variance". Then print, from the data rather than from memory, whether
> each pair of curves has met.

**Expect:** left panel, two curves that meet around 0.4; right panel, two curves
that stay far apart across the whole x-axis.
**Assert:** none.
**Annotate:** short

> `sharey=True` is the whole cell. Independent y-axes make a gap of 1.6 and a
> gap of 0.07 look identical, and that is the most common way a true plot tells
> a false story.
>
> Derive the verdict, do not type it. `f"{gap:.1f} apart"` from `lc[5]`, not the
> string "1.6 apart" — it happens to be right today, and it will still print
> "1.6" on the day it becomes 0.9.
>
> Curves that have met mean more data changes nothing. Curves still far apart
> and falling mean starved of rows, not of capacity. There are 891 passengers on
> the Titanic and there will never be more, which is the one remedy this dataset
> refuses you.

---

## Cell 16 — read the warning instead of silencing it

**Prompt to type:**

> Fit `pipeline(degree=d)` on the whole training set for d in 1..6 with
> `warnings.catch_warnings(record=True)` and `simplefilter("always")`, and print
> for each degree whether it converged, the number of iterations, and the
> largest absolute coefficient. Assert degrees 1 and 3 converge and degree 4
> does not.

**Expect:**

| degree | converged | iterations | largest \|θ\| |
|---|---|---|---|
| 1 | True | 74 | 4.87 |
| 2 | True | 256 | 5.43 |
| 3 | True | 935 | 6.32 |
| 4 | False | 4000 | 18.70 |
| 5 | False | 4000 | 11.07 |
| 6 | False | 4000 | 3.98 |

Iterations climb 74 → 256 → 935 and then hit the ceiling. **Read the last column
honestly:** it jumps at degree 4, which is the evidence, and then *falls* at
degrees 5 and 6 — because those two stop after 0.3 s on a failed line search
rather than after 4000 real iterations, so their coefficients never got the
chance to grow. "It grows without bound" is the right theory and this table is
weak evidence for it. Say so, or raise `max_iter` and show the growth properly.

**Assert:** `sep[1]["converged"] and sep[3]["converged"]` and
`not sep[4]["converged"]`
**⏱** **1.0 s measured** on an idle M4 Max, single-threaded — the whole cell,
six fits. Colab-scale: a few seconds. No ⏱ marker is needed in the notebook, and
the current notebook is right not to have one; I include the figure because I
first measured this cell at **106 s** under concurrent load and nearly wrote
that down.
**Annotate:** short

> The reflex is `max_iter=100000` and a coffee. It will not help. If some θ
> separates the classes in the expanded space then σ(cθᵀx) → {0,1} as c → ∞, so
> the training loss falls monotonically toward zero as ‖θ‖ → ∞ and the infimum
> is never attained. The optimiser is not slow; it is looking for something that
> does not exist, and the growing-coefficient column is what that looks like
> numerically.

---

## Cell 17 — the dependence, as an eigenvalue

**Prompt to type:**

> Print the largest absolute value of SibSp + Parch + 1 − FamilySize over the
> training rows. Then median-impute and standardise the five columns SibSp,
> Parch, FamilySize, Age, Fare, form G = ZᵀZ, and print its eigenvalues with
> `eigvalsh` and its condition number. Assert the smallest eigenvalue is below
> 1e-9.

**Expect:** `0` for the identity, then the five eigenvalues
`[1.81e-12  4.29e+02  5.44e+02  7.90e+02  1.80e+03]` — one of them at machine
zero — and a condition number of **9.2e+14**.
**Assert:** `eig.min() < 1e-9` — the dependence must show up as a *zero*, not
merely as a small number.
**Annotate:** short

> `eigvalsh`, not `eigvals`: G is symmetric, and the general routine hands back
> complex numbers with tiny imaginary parts that then have to be explained away.
>
> **State plainly what this matrix is and is not.** These five columns are not
> the design matrix any model in this notebook is fitted on — `NUM` excludes
> `FamilySize` from cell 5 onwards, precisely so that it does not appear. This
> cell measures the fault that *would* be there, on the frame where it still
> lives. A reader who thinks the degree-5 disaster in cell 7 was caused by this
> singularity has been misled: that pipeline never saw `FamilySize`. Its design
> matrix at degree 1 is 22 columns, 23 with the intercept, and full rank.

---

## Cell 18 — what α does to the eigenvalues

**Prompt to type:**

> For α in 0, 1e-6, 1e-3 and 1, print the condition number of G + αI computed
> from the shifted eigenvalues λ+α (not by refitting anything), next to the
> bound (λmax+α)/α. Assert α = 1 brings it under 2000.

**Expect:** 9.2e+14 at α = 0, falling to **1798** at α = 1 — a four-decade
collapse from one line of arithmetic on the eigenvalues.
**Assert:** `(eig.max() + 1.0) / (eig.min() + 1.0) < 2000`
**Annotate:** short

> Two lines of proof, not an empirical finding: if XᵀXv = λv then
> (XᵀX + αI)v = (λ+α)v — same eigenvectors, every eigenvalue moved up by exactly
> α — and XᵀX is positive semi-definite, so λ+α ≥ α > 0 and the matrix is
> invertible for every α > 0. The bound (λmax+α)/α does not mention λmin at all,
> which is why ridge works on a singular design with no column-dropping first.
>
> Say the honest thing in the print: ridge does **not** remove the
> multicollinearity. The dependence is exactly where it was. It is dominated.

---

## Cell 19 — the grid, and the solver you should not have picked

**Prompt to type:**

> Set up the penalty comparison: `Cs = np.logspace(-4, -0.5, 8)`, a 5-fold
> shuffled StratifiedKFold called `cv5`, and a dict `PENALTIES` with "ridge" =
> l2/lbfgs, "lasso" = l1/saga/max_iter=5000, "elastic" = elasticnet/saga/
> l1_ratio=0.5/max_iter=1000. Nothing fitted yet.

**Expect:** no output. `Cs` runs from 1e-4 to 0.316.
**Assert:** none. Check by eye that `Cs[0] < Cs[-1] < 1`.
**Annotate:** full

- **Left open:** which direction `C` runs in. Scikit-learn's `LogisticRegression`
  takes **C = 1/α**, so **small C is strong regularisation** — the grid above
  runs from *very strong* to *moderately strong* and never reaches the
  unpenalised end. And the default is `C=1.0, penalty="l2"`: a bare
  `LogisticRegression()` is already regularised, which is why cell 6 had to set
  `C=1e6` explicitly to get an unpenalised model to repair.
- **The usual student version:** `solver="liblinear"` on the lasso row. It is
  the obvious pick for an L1 penalty and it is wrong here: liblinear implements
  the intercept as a synthetic constant column and **penalises that column's
  weight like any other**. Measured on this pipeline at degree 1, `penalty="l1"`,
  `C=0.001`: liblinear returns an intercept of exactly **+0.0000** with 0 of 22
  weights surviving, while saga returns **−0.5018** — essentially the base-rate
  log-odds, −0.4750, which is what an unpenalised intercept should be when every
  slope has been crushed to zero. This lecture states that the bias term is not
  penalised. With that solver, that statement is false in your own output.
- **How you would catch it:** print the intercept. If it is suspiciously near
  zero under a strong penalty, your solver is penalising it and every
  coefficient in the table is compensating for that. `intercept_scaling` is the
  knob liblinear gives you, and its default of 1.0 is not a neutral choice.

> On the range: the deck sweeps 17 values from 1e-4 to 1e4 and this grid stops at
> 0.316. **Give the real reason.** The lecture module says the truncation is
> about cost — "163 s for a single cross-validation at C = 1 against 1.6 s at
> C = 0.01" — and on scikit-learn 1.7.2 that does not reproduce: a single 5-fold
> cross-validation of the lasso row at degree 5 measured **9.0 s at C = 0.01,
> 10.3 s at C = 0.1 and 9.6 s at C = 1**. Flat, within noise, no 100× cliff. The
> defensible reason to stop at 0.316 is that the minimum is well inside the range
> and the weak-penalty end is the model you already know fails; say that, and
> re-measure the cost claim on the machine you are actually using before you
> repeat it.

---

## Cell 20 — three penalties, eight values each

**Prompt to type:**

> For each entry in PENALTIES, cross-validate `pipeline(degree=5, C=C, **kw)`
> over the eight Cs with cv5 and neg_log_loss, and also fit once on the full
> training set to count how many coefficients are non-zero above 1e-8. Keep the
> best C per penalty with its log loss and its non-zero count, print them, and
> flag any penalty whose best C is the first or the last value in the grid.

**Expect:**

| penalty | best C | log loss (5-fold) | non-zero of 143 |
|---|---|---|---|
| ridge | **0.0001** — first in the grid | 0.724 | 143 |
| lasso | 0.001 | 0.681 | 5 |
| elastic | **0.3162** — last in the grid | 0.681 | 140 |

**Read all three columns before you write a sentence about this table.**

1. Every one of them is an enormous repair on the 1.92 that degree 5 scored
   unpenalised — and **none of them beats the 0.666 anchor**. "Regularisation
   made it respectable" is not what this output says. It made it *survivable*.
2. **Two of the three minima sit on the edge of the grid.** A best value at the
   boundary is not a minimum, it is the grid running out. Ridge wants *more*
   penalty than 1e-4 offers — i.e. it wants to shrink 143 columns to nothing and
   land on the constant model — and elastic net wants *less* than 0.316. Neither
   number is a tuned hyperparameter; both are artefacts of where you stopped.

**Assert:** `reg["lasso"]["nnz"] < 143` and `reg["ridge"]["nnz"] == 143`. Then
print, do not assert, whether each `best C` is at a grid endpoint — on this grid
two of them are, and an assert that always fires teaches nothing.
**⏱** **54.7 s measured** (NJ=4, idle machine): ridge 5.8 s, **lasso 39.3 s**,
elastic 9.6 s. The lasso row is three quarters of the cell. Colab-scale: about 2
minutes.
**Annotate:** short

> The non-zero count is the check that the penalty you asked for is the penalty
> you got. An L1 run with 143 of 143 weights non-zero did not apply an L1
> penalty, whatever the string said. Ridge keeping all 143 and lasso keeping 5
> is the entire difference between the two, and it is invisible in the score.
>
> **Do not put the cell-19 intercept assert here.** At degree 5 saga does not
> converge within `max_iter=5000` and returns an intercept of −0.0001 for lasso
> and −0.0000 for elastic net — indistinguishable from the liblinear pathology,
> for a completely different reason. The intercept check is diagnostic at degree
> 1, where saga converges in milliseconds and returns −0.5018; at degree 5 it
> tells you only that the solver gave up. Run diagnostics where they can
> discriminate.

---

## Cell 21 — read that table twice

**Prompt to type:**

> Print, in one column: the anchor, degree 5 unpenalised, degree 5 under each of
> the three penalties, and degree 2 unpenalised — and put the number of
> cross-validation folds beside every row, because the penalty sweep used 5 and
> the degree sweep used 10. Assert that no penalty beats degree 2, with a
> message saying that if this ever fires the story has changed and you must say
> so.

**Expect:**

```
anchor, no model at all               0.666
degree 5, no penalty     (10 folds)   1.922
degree 5, ridge tuned    ( 5 folds)   0.724
degree 5, lasso tuned    ( 5 folds)   0.681
degree 5, elastic tuned  ( 5 folds)   0.681
degree 2, no penalty     (10 folds)   0.467   <- still the best
```

**Assert:** `min(reg[n]["log_loss"] for n in PENALTIES) > sweep[2]["valid"]`
**Annotate:** short

> **The fold counts are not decoration.** Five folds trains each model on 569
> rows, ten folds on 640. The penalised rows and the unpenalised rows in this
> table were therefore measured on different amounts of data and scored on
> different held-out sets, and the notebook this replaces prints them in one
> column with no marking at all. Either re-run the penalty sweep at `cv` and pay
> the extra minute, or print the fold count on every row and say the comparison
> is indicative. Do not print them bare.

> Every penalty takes the worst model in the sweep — three times worse than
> saying nothing — and drags it back to the neighbourhood of the anchor. **Not
> one of them reaches the anchor, and none of them comes near the plain degree-2
> model you already had.** That is not a disappointment to be explained away:
> cell 11 predicted it an hour earlier, when the squared bias at degree 1 came
> out barely above the noise floor. There was never much bias for the extra
> capacity to buy back. Regularisation here is a **repair**, not an upgrade —
> and on this data it is a repair that does not finish the job, which is a
> stronger version of the same lesson and the one your own output supports.
>
> Put the unregularised baseline in the same list as the three repairs. Three
> repairs and no baseline is not a comparison.
>
> An assert with a message telling you what to do when it fires is a tripwire
> rather than a test. This is the right place for one.

---

## Cell 22 — early stopping

**Prompt to type:**

> Split the training rows 75/25 stratified into A and B. Fit `prep(degree=5)` on
> A **only**, outside the loop, and transform both. Then train an
> `SGDClassifier(loss="log_loss", penalty=None, learning_rate="constant",
> eta0=0.0015, warm_start=True, max_iter=1, tol=None, random_state=RANDOM_STATE)`
> for 500 epochs, calling fit once per epoch, recording log loss on A and on B
> each time. Print the best epoch, its validation loss, the epoch-500 loss, and
> the regret for not stopping.

**Expect:** both curves fall; the validation curve reaches its minimum well
before epoch 500 and then rises. The absolute numbers are poor — that is
expected and the cell should say so.
**Assert:** `Z_a.shape[1] == Z_b.shape[1] == 143`, `len(A) + len(B) == 712`,
and `best + 1 < 500` — a minimum at the last epoch means there was nothing to
stop early.
**⏱** **about 20 s** (author's figure). 500 one-epoch fits on 534 rows × 143
columns.
**Annotate:** full

- **Left open:** where the transform is fitted. Inside the loop it would be the
  leak from lecture 5's first application, committed 500 times over; fitted on
  `X_train` before the split it would be the same leak committed once. The
  prompt has to say *on A only, outside the loop* or you will get one of those
  two, and both of them produce a nicer-looking curve.
- **The usual student version:** `max_iter=1` without `warm_start=True`.
  `warm_start` defaults to **False**, so every `fit` call throws away the
  previous weights and restarts from zero: 500 independent one-epoch models
  rather than 500 epochs of one model. The curve comes out flat and reads as a
  model that cannot learn. The second default in the same constructor:
  `SGDClassifier` uses `penalty="l2"` with `alpha=1e-4` unless you say
  otherwise, so omitting `penalty=None` leaves a second regulariser in a cell
  whose entire subject is regularisation-by-stopping.
- **How you would catch it:** print the first three weights at epochs 1, 2 and
  3. If they are the same numbers each time, `warm_start` is not doing what you
  think. And say out loud in the cell that these are bad numbers in absolute
  terms — plain SGD at a constant learning rate on 143 correlated columns is a
  poor optimiser, and a number left unqualified will be quoted qualified.

---

## Cell 23 — the shape, not the number

**Prompt to type:**

> Plot both epoch curves with a dashed vertical line at the chosen epoch, then
> print a sentence saying these are deliberately not competitive numbers and
> that the stopping epoch is a hyperparameter chosen on held-out rows like any
> other.

**Expect:** the classic U on the validation curve, the training curve still
falling at 500.
**Assert:** none.
**Annotate:** short

> Early stopping is not free regularisation. It is regularisation whose knob
> happens to be time, chosen on held-out rows exactly like `C`. Nobody quotes an
> epoch as a fitted parameter, which is precisely why it escapes the accounting.

---

## Cell 24 — "find the best C and tell me how well it does"

Type this one exactly as written, including the vagueness. Run it. **Write the
printed number down on paper before you go on.**

**Prompt to type:**

> Find the best value of `C` for my logistic regression and tell me how well it
> does.

**Expect:** a loop over `np.logspace(-4, 4, 17)` at some fixed degree, each
model fitted on the training rows, each scored on the test rows, and one line.
At degree 3: **`best C = 1, log loss = 0.4317`**. It runs, it warns about
nothing, and it prints a number a reader will quote.
**Assert:** none. That is part of the point.
**⏱** **0.6 s measured** at degree 3 (17 fits, idle machine). A few seconds on
Colab.
**Annotate:** short

---

## Cell 25 — measure the optimism

⚠ **Now go back and look at cell 24.** *What touched the test set?*

Seventeen models did, and then it reported the score of whichever one it liked
best. That number is a **minimum over seventeen noisy estimates**, and the
minimum of noisy estimates is biased downward even when every estimate is
individually unbiased. The code never *writes* to the test set. It reads it
eighteen times, and reading is enough. Nothing in the prompt was wrong; it
simply never said **on what data** "best" was to be found, or **on what data**
"how well it does" was to be measured.

**One matched comparison, before the experiment.** Cell 24 reported **0.4317**.
Cell 26 — which you have not run yet — picks `C` by cross-validation inside the
training set and then scores once, and gets **0.4501**. *Same 179 test
passengers, same 17 candidates, same degree, same seed.* The only difference is
which rows chose `C`, and it is worth **0.018** of log loss, in the flattering
direction, every time.

Do not fix it yet. Measure the damage first — one split is one draw, and 0.018
means nothing until you know the spread.

**Prompt to type:**

> For each of 8 seeds: make a fresh 80/20 stratified split of X and y with that
> seed, loop over the same 17 values of C at degree 3, and record two things —
> the score on the held-out part of the C that scored best on the held-out part,
> and the score on the held-out part of the C that 5-fold cross-validation
> inside the training part picked. Also record whether the two procedures picked
> the same C. Run the seeds in parallel and print both means with their spreads,
> the difference, and the same-C count.

**Expect:** the dishonest number is lower — better-looking — than the honest one
on most or all seeds, by an amount **smaller than the seed-to-seed spread**.
Both halves of that sentence matter and the cell should print the numbers for
both. Seed 0 alone, measured: reported **0.4638**, honest **0.4770**, a gap of
0.013, and the two procedures picked **different** values of `C` — so on that
seed the whole gap is selection.
**Assert:** `(honest - reported).mean() > 0` — the bias is one-sided by
construction, so a negative mean means the experiment is wired wrong, not that
the leak is benign.
**⏱** Measured: one seed of this loop — 17 × (1 fit + 5 CV fits) = 102 fits —
takes **2.7 s** on an idle M4 Max, so eight seeds over four workers is **about
6 s**. Colab-scale figure: **about 2 minutes**, and twenty seeds about 5. The
deck uses twenty and this uses eight; the only thing that changes is the width
of the interval, and the interval is the point.
**Annotate:** full

- **Left open:** that both numbers must be scored on **the same rows**. Both are
  scored on `B`, the held-out fifth of that seed's split; only the *selection*
  differs. If you score the honest choice on a different split you are measuring
  the split, and the effect you are hunting is smaller than that difference.
- **The usual student version:** exactly cell 24 — this is not an invented
  mistake, it is what a competent request returns when it is under-specified,
  and it is what most people do by hand under time pressure. The second version:
  running this experiment on **one** seed, seeing a difference smaller than the
  noise, and concluding the leak does not matter. It never averages away, it is
  one-sided, and it grows with how hard you looked: seventeen candidates is a
  tiny search, and a randomised search over a thousand configurations does the
  same thing much louder.
- **How you would catch it:** report how often the two procedures picked the
  **same** C. On the seeds where they agree the leak costs nothing, so that
  count tells you how much of the effect is selection and how much is scoring —
  which is the only way to know whether you are looking at a real mechanism or
  at eight noisy differences with a sign.

---

## Cell 26 — the honest version

**The corrected specification**, and the only version of this request worth
typing:

**Prompt to type:**

> Choose `C` by 10-fold cross-validation **on the training set only**, using
> GridSearchCV over the whole pipeline with `scoring="neg_log_loss"`. Refit the
> winner on the full training set, then evaluate once on the test set and print
> both numbers separately with the fold spread.

**Expect:** `chosen C: 0.1`, `honest CV estimate: 0.462 (fold sd 0.071)`,
`the test set, once: 0.450`. Three lines, plus a sentence saying the test number
is allowed to be worse than the CV number — here it happens to come out
slightly better, and 0.071 of fold spread is why you should not read anything
into either direction.
**Assert:** `gs.best_index_ is not None`, and check by eye that
`gs.best_estimator_` is a `Pipeline` and not a bare `LogisticRegression`.
**⏱** **1.9 s measured** (17 candidates × 10 folds + refit, NJ=4, idle machine).
Ten times the fits of cell 24 for three times the wall clock, because they are
spread over four workers. Colab-scale: under a minute.
**Annotate:** short

> Three things this prompt has to say and the vague one did not: *on the
> training set only*, *over the whole pipeline*, and *both numbers separately*.
>
> **`scoring="neg_log_loss"` is load-bearing.** `GridSearchCV(scoring=None)`
> falls back to the estimator's own `.score()`, which for a classifier is
> **accuracy** — so the default search picks the C with the best accuracy while
> you report its log loss, and the two disagree here by construction (cell 7:
> degree 5 has 76% CV accuracy and a log loss three times the anchor).
>
> **Over the whole pipeline**, so the imputer, the scaler and the encoder are
> refitted inside each fold. Grid-searching the classifier alone on
> pre-transformed data is honest about C and dishonest about the scaler.
>
> The test number is allowed to be worse than the CV number. Say so in the
> cell, because the instinct when it is worse is to go back and adjust
> something, and that is the failure you have just spent two sections measuring.

---

## Cell 27 — the test set, once

**Prompt to type:**

> Build five candidates — degree 5 unpenalised, degree 5 with the tuned ridge C,
> degree 5 with the tuned lasso C, degree 1 with sklearn's defaults (C=1.0), and
> degree 2 unpenalised — fit each on the training set, score each on the test
> set, and print log loss, Brier and accuracy for all five. Pick the winner by
> log loss and assert it beats the anchor.

**Expect:**

| candidate | log loss | Brier | accuracy |
|---|---|---|---|
| degree 5, no penalty | 1.727 | 0.189 | 74.9% |
| degree 5, ridge tuned (C = 1e-4) | 0.779 | 0.230 | 64.8% |
| degree 5, lasso tuned (C = 1e-3) | 0.669 | 0.240 | 64.8% |
| degree 1, sklearn defaults | 0.433 | 0.134 | 82.7% |
| **degree 2, no penalty** | **0.427** | 0.134 | 82.7% |

Majority-class accuracy on these 179 passengers is 61.5%.

**Do not skip past the middle two rows.** Regularising degree 5 improves its log
loss enormously (1.727 → 0.669) while making its Brier score *worse* (0.189 →
0.240) and its accuracy *worse* (74.9% → 64.8%). Both directions are correct and
they are the same fact seen twice: the penalty shrinks every prediction toward
the base rate, which log loss rewards heavily — it is the metric that punishes
confident mistakes, −log(0.01) ≈ 4.6 — and which Brier and accuracy do not. **If
you had run this cell scoring accuracy you would have concluded that
regularisation made the model worse.** Report all three, and name the metric
before you name the winner.

**Assert:** `final[winner]["log_loss"] < constant_log_loss`
**⏱** **2.4 s measured**, all five candidates.
**Annotate:** short

> Every hyperparameter in this cell was fixed before the cell ran. No selection
> happens here; only measurement. **If you find yourself editing this cell after
> reading its output, stop** — the test set has now been read twice, and that is
> cell 24 committed by hand, three cells later.

---

## Cell 28 — the five numbers, and the floor

**Prompt to type:**

> Print the anchor, the best cross-validated log loss from the sweep, the final
> test log loss, majority-class accuracy on the test set, and the winner's test
> Brier next to the measured noise floor — and compute the difference between
> those last two rather than typing it. Say which rows each number came from.

**Expect:**

```
anchor — report the base rate to everyone   0.666   (712 training rows)
best cross-validated log loss               0.467   (712 rows, 10 folds)
final, on the test set                      0.427   (179 test rows)
majority-class accuracy on the test set     61.5%   (179 test rows)
improvement over where the build session ended: 1.300
Brier 0.134 (179 test rows) against a floor of 0.122 (712 training rows)
=> about 0.012 of Brier left for any model of these columns
```

**Assert:** `best < constant_log_loss`
**Annotate:** short

> **§2.1 applies here and cannot be fully satisfied, so say so.** The Brier
> score is on the 179 test passengers; the floor is on the 712 training
> passengers; they are different people. The defence is that the floor barely
> depends on which people: 0.121 over all 891, 0.122 over the 712, 0.119 over
> the 179. The headroom is **0.0121** against the training floor and **0.0133**
> against the all-891 floor, so quote it as *about 0.012*, state both row sets
> in the same sentence, and do not write a third decimal you cannot defend.
>
> **The winner is a model with no repair in it.** Degree 2, no penalty, chosen
> by reading a held-out curve — and `LogisticRegression()` with every default
> left alone is within 0.006 of it. Ninety minutes of ridge, lasso, elastic net
> and early stopping, and the best system on the test set is two lines you could
> have written before the build session started. Report it anyway. The
> alternative is choosing the interesting model over the better one, which is
> the failure this course exists to prevent.
>
> What the work bought is not a better number. It is knowing *why* degree 2
> wins, in three measured terms; knowing the floor; and knowing that the
> degree-5 model was not merely worse but **ill-posed** — and exactly which term
> repairs that.

---

## Cell 29 — red-team questions 3 and 4

**Prompt to type:**

> Print how many columns `prep(5)` produces, how many training rows there are,
> and the ratio; then how many Age values were imputed in the training and test
> sets and how many Embarked values are missing in the whole frame. Assert 143
> columns and that no passenger was dropped.

**Expect:** 143 columns, 712 rows, **5.0 rows per weight**; **137** imputed ages
in the training set and **40** in the test set (177 in the whole frame); 2
missing Embarked.
**Assert:** `n_cols == 143` and `len(X_train) + len(X_test) == 891`
**Annotate:** short

> Five rows per weight, printed. No amount of tuning repairs a ratio like that,
> and it is the kind of thing that is obvious in another person's notebook and
> invisible in your own.
>
> Nothing was dropped: the imputer fills, and it is fitted per fold.

---

## The red-team checklist, and how to run it (§7.2)

Swap notebooks with someone. Eight minutes. Report what you found, not what you
would have done differently.

1. **What touched the test set?** Count every *read*, not every write.
2. **What was fitted, and on what?** Is the scaler inside the cross-validated
   pipeline, or fitted once outside it?
3. **How many columns does the degree-5 pipeline produce, against how many
   rows?**
4. **Which passengers have an imputed `Age`, and how many?**
5. **Which `LogisticRegression` had its `C` left unstated?** Remember the
   default is `C=1.0`, not "no penalty".

Cell 29 answers 3 and 4 mechanically. **1, 2 and 5 need a person**, and on this
notebook the honest answer to question 1 is: cells **10, 24 and 26** read the
test set before cell 27 claims it is untouched — 800 reads, 17 reads and 1 read
respectively — and cell 8 does too if you compute the floor over all 891 rows
rather than the 712 this script specifies.

**Re-run order for the two exercises that need one:**

- *Change the seed and re-measure everything.* Edit `RANDOM_STATE` in **cell 1**,
  then re-run, in this order: **1 → 5 → 6 → 7**, then **8 → 10 → 11 → 13** if you
  want the decomposition, then **27 → 28**. Cells 2, 3 and 4 do not depend on
  the seed and cost 5 s and a download; re-running everything is simpler and
  takes about 9 minutes. Skipping cell 6 is the one that will bite you: `cv` and
  the anchor are both built there.
- *Reduce the tuning-trap experiment to 3 seeds to see it run quickly.* Change
  `range(8)` in **cell 25** only, and re-run **cell 25** alone. It depends on
  nothing but `X`, `y` and `pipeline`. Read the printed spread before you read
  the printed difference: at 3 seeds the difference will usually be inside it.

**The standing constraint, extended.** Add one clause to what you wrote down in
lecture 2:

> *"Split before anything is fitted. All preprocessing lives inside a `Pipeline`
> passed to cross-validation. Nothing derived from the test set may appear in
> the training path — **including the choice of any hyperparameter**. Fixed
> random seed. Print the fold scores, not just the mean."*

Eleven added words. They are the whole of cell 24.

**Examinable (§8.3).** Sections 2 and 3 — the decomposition, separation, and
what α does to the spectrum — are examinable. Section 4 is examinable for the
ridge/lasso objectives and the subgradient argument, not for the API. Sections
1, 6 and 7 are engineering and are not examinable. Section 5 is examinable as a
*procedure*: you may be asked what touched the test set.

---

## Defects found in the current notebook

Everything below refers to `notebooks/lecture-06.ipynb` as it stands, generated
from `tools/notebooks/lecture_06.py`. **Checked** means I re-derived it with
`python3` against `notebooks/datasets/titanic/train.csv` on scikit-learn 1.7.2 /
pandas 2.3.3 / numpy 2.3.5. The notebook ships with **no stored outputs**, so
§1.2 cannot be checked against it at all — every prose figure had to be
recomputed from scratch, which is itself worth recording.

### Numbers that do not reconcile (§1.1)

1. **`1.957` for degree 5 is stale — measured 1.922.** *Checked.* Section 3's
   opening sentence ("Degree 5 scored 1.957 against an anchor of 0.666") and the
   annotation on the sweep cell both quote it. Re-running that exact
   `cross_validate` gives **1.9217**. The notebook's own annotation names this
   failure mode — *"`assert sweep[5]['valid'] == 1.957`, which is true today"* —
   and it is no longer true. The qualitative claim ("three times worse than
   saying nothing", 1.922/0.6657 = 2.89) survives.
2. **`0.468` for degree 2 is the degree-*1* number.** *Checked.* Section 4 says
   "none of them beats the plain degree-2 model you already had at 0.468".
   Measured: degree 2 = **0.4671**, degree 1 = **0.4681**. 0.468 is degree 1's
   held-out score, attributed to degree 2.
3. **`75.4%` accuracy at degree 5 matches nothing.** *Checked.* Section 3 says
   "Accuracy at that degree was 75.4%" without saying on which rows. Measured:
   **76.0%** by 10-fold CV on the training rows, **74.9%** on the test set.
   Neither rounds to 75.4%, and §1.2 requires the row set to be named.
4. **"a bootstrap with replacement gives each fit ~253 distinct rows" — it gives
   306.** *Checked* by closed form and by 2,000 simulated draws: drawing 400 rows
   with replacement from 712 leaves 712·(1−(711/712)⁴⁰⁰) = **306.2** distinct
   rows, simulated mean 306.3. The figure 253 is 400·(1−1/e), the *n-from-n*
   bootstrap answer, applied to an n-from-N draw.
5. **"At 12% of 712 that is 85 passengers" — it is 76.** *Checked.*
   `learning_curve` scales `train_sizes` by the largest training set available
   inside a fold, which with `cv` = 10 folds is 640, not 712. The eight sizes
   are `[76 157 237 318 398 479 559 640]`. The same cell asserts `n[-1] == 640`,
   so the box and the assert contradict each other three lines apart.
6. **"where saga fits 4.90" does not reproduce.** *Checked, and it did not
   reproduce.* At degree 5, `penalty="l1"`, `C=0.001`, `max_iter=5000`, saga
   returned an intercept of **−0.0001** after 54 s (and had plainly not
   converged). The *liblinear* half of the claim is exactly right — intercept
   **+0.0000**, because liblinear penalises its synthetic intercept column — and
   the contrast is real and reproducible at degree 1, where saga converges in
   milliseconds: liblinear **+0.0000** with 0 of 22 weights alive, saga
   **−0.5018** against a base-rate log-odds of −0.4750. I could not find any
   configuration of this pipeline that produces 4.90.
7. **Hard-coded prose inside `print()` calls.** *Checked by reading the source.*
   Four claims are printed as string literals rather than computed, so they will
   keep printing after they stop being true: `"still 1.6 apart and falling"`,
   `"the squared bias of the degree-1 model is under 0.01"`, `"about 0.013 of
   Brier score is left on the table"`, and `"<- still the best"`. The first and
   third happen to be right today (the gap at degree 5 is 1.636; 0.134 − 0.121 =
   0.013). The second I could not check without running the two-minute
   decomposition cell.
8. **The prose says "the factor of ten"; the cell prints 22.** *Checked.* The
   annotation on the gap cell says "The factor of ten is what connects the
   picture back to the decomposition", but the assert is a floor (`g5 > 10*g1`)
   and the measured ratio is **22.4**.

### Comparisons on mismatched rows (§2.1, §2.4)

9. **The noise floor and the final Brier score are measured on different
   people.** *Checked.* The floor cell does `d = full.copy()` — all 891
   passengers — and section 6 sets the winner's Brier score, computed on the 179
   test passengers, beside it. Measured: floor **0.1206** over all 891 (858
   people in multi-passenger cells), **0.1218** over the 712 training rows. The
   headroom claim is robust (0.013 vs 0.012), which is exactly the sentence the
   notebook should be printing instead of the comparison it currently makes
   silently.
10. **"179 passengers, untouched since the split at the top of this notebook" is
    false.** *Checked by static analysis of the cell sources.* Before section 6,
    `X_test`/`y_test` are read by the bias-variance cell (200 draws × 4 degrees =
    800 `predict_proba` calls plus `y_test` as ground truth), by the assistant-
    failure cell (17 reads) and by the honest GridSearchCV cell (1 read); and
    `y_test`'s labels enter the noise floor through `full`. None of those reads
    *selects* anything except the deliberate one — but the notebook's own
    red-team question 1 is "count every *read*, not every write", and by its own
    rule it fails its own checklist. This is the most defensible finding in the
    list and it is a two-line fix: say what the reads were.
11. **"degree 2 is still the best" is a coin flip.** *Checked.* Degree 2 =
    0.4671, degree 1 = 0.4681 — a margin of **0.0010** against a paired
    per-fold standard deviation of **0.0288**, with degree 2 winning **5 of the
    10 folds**. §2.4 says a result inside its own noise is not a result. The
    `<- still the best` marker printed beside degree 2, and the whole closing
    argument that "the winner is a two-line model", rest on a difference the
    experiment cannot resolve. The conclusion survives — degree 1 *is* also a
    two-line model — but the cell should print the fold spread and the prose
    should say "degree 1 or 2", which is what the notebook's own assert already
    allows.
12. **The penalty table and the degree table are cross-validated with different
    numbers of folds and printed in one column.** *Checked.* The degree sweep
    uses `cv` (10 folds, 640 training rows per fit); the penalty sweep uses
    `cv5` (5 folds, 569 rows per fit). The "read that table twice" cell prints
    `sweep[5]`, three `reg[...]` values and `sweep[2]` as five comparable rows,
    unmarked. §2.1: the comparison is across different training sizes and
    different held-out sets.

### Cross-references that do not resolve (§3.3)

13. **"Every penalty … makes it respectable" — measured, not one of them reaches
    the anchor.** *Checked by running the penalty sweep.* Best cross-validated
    log loss at degree 5: ridge **0.724**, lasso **0.681**, elastic net
    **0.681**, against an anchor of 0.666. All three are still worse than
    predicting the base rate for everybody. The repair is real and enormous
    (from 1.92) and the sentence describing it overstates where it lands. The
    same run makes a second point the notebook never mentions: on the test set
    the tuned models score **worse** than the unpenalised degree-5 model on
    Brier (0.230 and 0.240 against 0.189) and on accuracy (64.8% against 74.9%),
    because the penalty shrinks everything toward the base rate. Log loss
    rewards that; the other two metrics do not.
14. **Two of the three tuned `C` values sit on the boundary of the grid.**
    *Checked.* Ridge's best is `Cs[0]` = 1e-4 and elastic net's is `Cs[-1]` =
    0.3162. The module's docstring justifies truncating the grid on the grounds
    that "the minimum is well inside the range anyway" — for two of the three
    penalties it is not inside the range at all, and a best value at an endpoint
    is the grid running out rather than a hyperparameter being tuned. Those two
    numbers are then carried into the final test-set table as "tuned".
15. **The stated cost reason for truncating the grid does not reproduce.**
    *Checked.* The module says coordinate descent at degree 5 takes "163 s for a
    single cross-validation at C = 1" and "1.6 s at C = 0.01". Measured on
    scikit-learn 1.7.2, one 5-fold cross-validation of that row: **9.0 s at
    C = 0.01, 10.3 s at C = 0.1, 9.6 s at C = 1**. Flat. The whole penalty cell
    runs in **54.7 s** (ridge 5.8 s, lasso 39.3 s, elastic 9.6 s).
16. **"the assert on line 3"** — *checked.* In the rendered `engineer` cell,
    line 3 is `d = d.copy()`. The assert being referred to is on **line 17**.
17. **"It is the third diagnosis, four sections from now"** (code comment) and
    "**four sections from now**" (annotation) — *checked.* The cell is in section
    1; diagnosis 3 is in section **3**. Two sections, not four.
18. **"rank 23 of 29 columns" is imported from a pipeline this notebook does not
    build.** *Checked.* Section 3 offers it as the second symptom of the
    degree-5 failure. Lecture 5's 29-column figure is its *unrepaired* encoder;
    lecture 6's `prep(1)` produces **22** columns, 23 with the intercept, and
    that matrix is **full rank** (`matrix_rank` = 23). The claim as written
    describes a design matrix no cell of lecture 6 constructs.
19. **Diagnosis 3 is demonstrated on a matrix the notebook never fits.**
    *Checked.* The eigenvalue cell builds `Z` from SibSp, Parch, FamilySize, Age
    and Fare — but `NUM` excludes `FamilySize` from the split cell onward, so no
    pipeline in the notebook ever sees the dependence. The prose ("Diagnosis 3
    is Thread 1 returning") presents as a cause of the 1.92 something that was
    already repaired in cell 5. The notebook says this obliquely, once, in a
    prompt box ("that is the whole of the third repair"), and never in the
    section that draws the conclusion.

### Staging (§8)

20. **The assistant failure is announced four times before it runs.** *Checked
    by counting.* (i) the notebook header — "cells marked ⚠ read before running
    contain a defect on purpose"; (ii) the section-5 markdown heading block with
    its own ⚠; (iii) the paragraph immediately above the cell — "It runs, it
    warns about nothing, and it prints a number a reader will quote"; (iv) the
    prompt-box label, "⚠ what the assistant returns", plus its three-bullet
    annotation which gives away the entire answer before the cell has run; and
    (v) the first line of the code itself, `# ⚠ WRONG — this is the failure, not
    the fix`. That is the exact defect §8.1 was written about. Nobody falls in.

### Prompt boxes (§6.1, §6.2)

21. **29 of 29 boxes carry the full three-bullet annotation.** *Checked
    programmatically:* every markdown cell beginning `> **Prompt` contains all
    three of `Left open`, `The usual student version`, `How you would catch it`.
    The budget is five to eight.
22. **Several "usual student version" bullets are invented rather than
    observed** (§6.2) — e.g. "pickling the frame at the end of the last
    notebook", "adding a sixth candidate after seeing the table", "reusing one
    fitted pipeline across the 200 draws". Each is plausible; none names a
    library default or a recorded failure. Meanwhile the genuinely real defaults
    available in the same cells go unmentioned: `return_train_score=False`,
    `rng.choice(replace=True)`, `learning_curve(shuffle=False)`,
    `groupby(observed=False, dropna=True)`, `SGDClassifier(warm_start=False,
    penalty="l2")`, `GridSearchCV(scoring=None)` → accuracy.

### Timing and reproducibility (§7.1, §7.2)

23. **No ⏱ note anywhere states a CPU, and this is not a formality.** *Checked* —
    "about two minutes" appears three times and names no machine, no core count
    and no `n_jobs`. §7.1 requires the CPU figure, and here is why it matters:
    the six timed cells claim 15 s, 30 s, 20 s and 2 min ×3, and on an idle
    16-core M4 Max the ones I measured came in at **6.0 s** (degree sweep),
    **1.9 s** (GridSearchCV), **0.6 s** (the assistant-failure loop) and
    **2.7 s per seed** for the tuning trap. A reader on that hardware who has
    been told "two minutes" has no way to tell a fast machine from a cell that
    silently did nothing. A reader on Colab needs the other number. Both belong
    in the note, and neither is currently there.
24. **A timing measured under load is worthless, and I nearly published one.**
    *Measured, twice.* The same 17-fit loop (cell 24) took **484.5 s** with three
    other Python processes on the box and **0.6 s** with none. The convergence
    cell measured **106 s** under the same contention and **1.0 s** clean. This
    is not a defect in the notebook — it is the reason §7.1 asks for the machine
    as well as the number, and it is worth one sentence in the notebook's own
    timing note, because `NJ = 4` on a two-core runtime is exactly the
    oversubscription that produces it.
25. **Three untimed multi-fit cells** — the assistant-failure loop (17 fits at
    degree 3), the GridSearchCV cell (170 fits plus refit) and the convergence
    cell (6 fits at `max_iter=4000`). All three are fast on a laptop (0.6 s,
    1.9 s, 1.0 s measured) and none was verified on a 2-vCPU runtime, where the
    170-fit one is the one to watch. The notebook times the 6 s degree sweep and
    not the 170-fit grid search, which is the wrong way round.
26. **No exercise lists a re-run order** (§7.2). Section 7 tells the reader to
    "run the equivalent on your neighbour's" notebook and the closing markdown
    asks them to extend the standing constraint, but nothing states which cells
    depend on `RANDOM_STATE`, and `cv`, the anchor and the split are created in
    three different cells.

### Notebook state (§4.1)

27. **`p` is rebound from a float to an array.** *Checked.* The anchor cell sets
    `p = y_train.mean()` (float) and uses it to define `constant_log_loss`; the
    final test-set cell rebinds `p = m.predict_proba(X_test)[:, 1]` (a 179-vector)
    inside its loop. One name, two kinds of object.
28. **`best` is rebound from an int to a float.** *Checked.* The early-stopping
    cell sets `best = int(np.argmin(valid_curve))` — an epoch index, used by the
    plot cell as `ax.axvline(best)` — and the closing summary cell rebinds
    `best = final[winner]["log_loss"]`. Re-running the early-stopping *plot* cell
    after the summary cell draws a vertical line at x = 0.43.
29. **`A` and `B` mean two different things.** *Checked.* The early-stopping cell
    binds them to a 75/25 split of the training rows at module level; `one_seed`
    rebinds them locally to a fresh 80/20 split of the whole frame. The shadowing
    is harmless to execution and confusing to read, and §4.1's remedy (throwaway
    names in throwaway scopes) applies.

### Marking (§8.3)

30. **"Examinable" appears three times and never as a section marker.** *Checked
    programmatically:* twice as "not examinable" in setup code comments, once in
    a prompt annotation. Sections 2 through 7 — including the whole of the
    bias-variance derivation — carry no marking at all.

### Checked and clean

- **§5.1 / §5.2 (markdown that renders):** no line indented four or more spaces
  outside a fence anywhere in the notebook; no fence marker indented at all. The
  two-space continuations in the display-maths blocks are fine. *Checked
  programmatically over all 50 markdown cells.*
- **§3.1 (code quoted in prose):** there are **no** fenced code blocks in any
  markdown cell of this notebook, so nothing can be quoted that does not exist.
  The inline claims I could check are true: `C=1.0` and `penalty="l2"` really
  are `LogisticRegression`'s defaults, `root_mean_squared_error` really did
  arrive in scikit-learn 1.4, and `np.logspace(-4, 4, 17)` really is 17 values.
- **§4.2 (idempotent training cells):** every fitting cell constructs its
  estimator inside itself — `pipeline()` is a factory and the `SGDClassifier` is
  built in the cell that trains it. Re-running any cell twice gives the same
  answer. *Checked by reading all 29 code cells.*
- **§7 (could a student alone at home do this?):** yes — CPU only, one 34 KB
  download, no GPU, no credentials, nothing that needs a lecturer. The only
  obstacle is the untimed six-minute cell in item 20.
- The anchor (**0.666**, exactly 0.665717), the split (**712/179**, rates 0.3834
  and 0.3855), the column counts (**22** at degree 1, **143** at degree 5), the
  title counts, the 891/342 row counts, the group counts on all 891 rows
  (**133 / 102 / 858**, floor **0.121**) and the near-singular spectrum
  (condition number ~1e+16, smallest eigenvalue at machine zero) all reproduce
  exactly. *Checked.*

### Not checked

**What I ran.** The degree sweep, the noise floor on all three row sets, the
convergence table, the eigenvalue cells, the penalty sweep, the assistant-failure
loop, the GridSearchCV cell, one seed of the tuning trap, and the final test-set
table. All numbers attributed to them above are measurements, not estimates.

**What I did not run, and what therefore stands unverified:**

- **The bias-variance decomposition** (800 fits, the two-minute cell). So the
  variance-growth factor, the exactness of the identity, the stackplot, and the
  printed claim that *"the squared bias of the degree-1 model is under 0.01"* are
  unchecked. That last one is a hard-coded string rather than a computed
  quantity, so the notebook does not check it either. Items 4 and 9 bear on this
  cell.
- **The early-stopping cell** (500 epochs). The claim that the validation minimum
  is interior is asserted in the cell itself, which is the right design; I did
  not confirm it runs.
- **The full 8-seed tuning trap.** One seed took 2.7 s and produced a gap of
  0.013 in the expected direction with the two procedures disagreeing on `C`, so
  the mechanism reproduces; the section-5 prose figures — *"about 0.02 of log
  loss, against a seed-to-seed spread of 0.05"* — are 8-seed aggregates I did not
  compute. The single matched split I did measure (0.4317 dishonest against
  0.4501 honest, same 179 rows) gives 0.018, which is consistent with "about
  0.02".
- **Anything about a Colab runtime.** Every timing here is an Apple M4 Max. The
  Colab-scale figures in this script are the lecture module's own, carried
  forward and labelled as such, and nobody has re-measured them on 2 vCPU.
- I could not determine what data or version produces the **4.90** saga intercept
  quoted in the module's comment (item 6). Everything else in that comment
  reproduces.
