# Lecture 2 — *Your RMSE was a lie* — rebuild-by-prompting script

**How to use this file.** Open a blank Colab notebook. For each cell below, type
the prompt into Colab's assistant *verbatim*, keep what comes back, and check it
against **Expect** and **Assert** before moving on. Where the returned code fails
the assert, regenerate — and note in the notebook that you did (§4.4: the claim
is "generated, then regenerated where noted", never "unedited").

Fifteen code cells. Four of them take longer than twenty seconds and say so.
**Six** carry the full three-bullet annotation (cells 4, 6, 7, 9, 11, 14); the
other nine carry a one-line specification only. That ratio is the point — see
GUIDELINES §6.1.

**Two cells contain a defect that is not announced before it** (cells 8 and 13).
Do not read ahead. The reveal is the cell after each, and "would you have caught
it?" only has an answer if you ran the cell first and wrote the number down.

**Every figure below was re-derived** on 2026-08-02, seed 42, scikit-learn 1.7.2
/ numpy 2.3.5 / pandas 2.3.3 / scipy 1.16.3, Apple M4 Max (16 cores). Your
numbers will match: every step is seeded. If one does not, that is a finding,
not a rounding difference.

**Examinable.** §2 (the normal equation), §4 (cross-validation and paired
comparison) and §6 (the interval) are examinable. §1 is engineering hygiene and
is not. §5 (grid search) and §7 (error by group) are examinable for the *method*,
not for the numbers.

**Vocabulary, defined once.** *Fold* — one of the k held-out slices in
cross-validation; a model is fitted k times, each time without one slice.
*Paired difference* — subtracting two models' scores fold by fold rather than
comparing their averages, so the shared difficulty of each fold cancels.
*Edge of the grid* — a winning hyperparameter equal to the largest or smallest
value you searched, which means you found the boundary and not the optimum.
*Percentile bootstrap* — resample the errors with replacement many times, and
take the 2.5th and 97.5th percentiles of the resampled statistic as the interval.
*Red-team* — read your own work looking for the mistake you would make, not for
confirmation that you did not make it.

---

## Cell 1 — setup and versions

**Prompt to type:**

> Setup cell for a teaching notebook. Import numpy as np, pandas as pd and
> matplotlib.pyplot as plt, print the Python, scikit-learn, numpy and pandas
> versions, and set a single constant RANDOM_STATE = 42 that the rest of the
> notebook will use for every split, model and shuffle. `root_mean_squared_error`
> only exists from scikit-learn 1.4, so make the cell fail loudly on anything
> older rather than printing the version and carrying on.

**Expect:** four version lines; `RANDOM_STATE` bound to 42. On current Colab,
scikit-learn ≥ 1.6, so the assert is silent.

**Assert:** `assert tuple(int(p) for p in sklearn.__version__.split(".")[:2]) >= (1, 4)`

**Annotate:** short

---

## Cell 2 — the data

**Prompt to type:**

> Write a function `load_housing()` that downloads
> `https://github.com/ageron/data/raw/main/housing.tgz` into a `datasets` folder
> only if it is not already there, extracts it, and returns
> `datasets/housing/housing.csv` as a DataFrame. Call it, check the shape rather
> than trusting the download, and show the first few rows.

**Expect:** `(20640, 10)`; columns `longitude, latitude, housing_median_age,
total_rooms, total_bedrooms, population, households, median_income,
median_house_value, ocean_proximity`. `total_bedrooms` has 207 missing values —
the only column that does.

**Assert:** `assert housing_full.shape == (20640, 10)`

**Annotate:** short

---

## Cell 3 — every import, and the same split as lecture 1

**Prompt to type:**

> Put every import this notebook needs in this one cell: the scikit-learn
> pipeline, preprocessing, tree, forest, linear model, metrics and model-selection
> pieces, plus `scipy.stats`. Then rebuild lecture 1's split from the seed rather
> than assuming its kernel is still running: cut `median_income` at
> 0, 1.5, 3, 4.5, 6, ∞ into five bands and take a stratified 80/20 split on those
> bands with `random_state=RANDOM_STATE`. Name the halves so the rest of the
> notebook can use them, and check the two sizes.

**Expect:** `16512` training rows and `4128` test rows — 20,640 × 0.8 exactly, no
remainder. `X_train`/`X_test` are DataFrames of 9 columns (the ten minus the
target); `y_train`/`y_test` are Series.

**Assert:** `assert len(X_train) == 16512 and len(X_test) == 4128`

**Annotate:** short — the check is the same-split guarantee. If either number
differs, every comparison in this notebook against lecture 1 is void, and there
is nothing further down that will tell you.

*Note for the author:* `scipy.stats` belongs **here**, not in cell 11. The
current notebook's cell says "Every import this notebook needs, in one place" and
then imports scipy nine cells later. Do not reproduce that.

---

## Cell 4 — what `LinearRegression().fit()` actually computed

**Prompt to type:**

> Take just the numeric columns of `X_train`, impute the missing values with the
> median and standardise them, then add a column of ones. Solve the normal
> equation for theta by hand with numpy, and check numerically that the residual
> is orthogonal to every column of the design matrix — report the largest inner
> product relative to the average size of y, since an absolute tolerance in
> dollars means nothing. Then fit `LinearRegression` on the same matrix and show
> that it found the same coefficients.

**Expect:** design matrix `(16512, 9)` — 8 numeric features plus the constant
column, so **9** parameters. Largest |Xᵀ(Xθ̂ − y)| = `7.106e-06`; mean |y| =
`206,334`; ratio = `3.444e-11`. `theta[0]` is `206333.51865310085` against a
`y.mean()` of `206333.51865310076` — the intercept **is** the mean of `y`, to
8.7e-11, because every other column was standardised to mean zero. That is an
expected answer a reader can state before running the cell. Largest disagreement with scikit-learn's own `[intercept_, coef_]`:
`6.4e-10`. Training RMSE of this fit: **$69,188** — *not* the $68,233 on the
slides, which is the same model with `ocean_proximity` one-hot encoded as well.
Say so in the markdown, or you have printed two numbers for one quantity (§1.5).

**Assert:** `assert np.abs(X_b.T @ residual).max() / np.abs(y).mean() < 1e-6`
and `assert np.allclose(theta, np.r_[lr.intercept_, lr.coef_])`

**Annotate:** full

- **Left open:** *whether the tolerance is doing any work.* `1e-6` against a
  measured `3.4e-11` is five orders of margin, which reads as a check that cannot
  fail. It can. Append `X[:, 3] * 2 - X[:, 6]` as a tenth column — one exact
  linear combination of two existing features — and the ratio goes to `1.07e+04`
  and the assert fires. That is the exercise, and it takes one line. **Re-run
  order: cell 4 only.**
- **The usual student version:** `np.linalg.inv(X.T @ X) @ X.T @ y`, believing
  that a singular `XᵀX` will raise and warn them. On the collinear matrix above
  it does not raise: `cond(XᵀX)` is `1.6e+17`, `inv` returns a θ̂ with
  ‖θ̂‖ = `1.8e6`, and the fit's training RMSE is **$160,835** — worse than
  predicting a constant, with no exception and no warning. `LinearRegression` on
  the same matrix returns **$69,188**, unchanged, because it calls
  `scipy.linalg.lstsq` and takes the minimum-norm solution. The current notebook
  states this backwards: it credits `np.linalg.inv` with the pseudoinverse and
  presents "still returns an answer" as the safe behaviour. It is `inv` that
  returns the wrong answer, and scikit-learn that returns the right one.
- **How you would catch it:** the orthogonality ratio, which is why it is
  computed relative to the scale of `y` rather than in dollars. It is the only
  line in the cell that changes by fifteen orders of magnitude when the matrix
  goes singular.

---

## Cell 5 — why the tree scored zero

**Prompt to type:**

> Build a ColumnTransformer that imputes and standardises the numeric columns and
> one-hot encodes `ocean_proximity`, ignoring categories it has not seen. Put a
> `DecisionTreeRegressor` after it in a pipeline, fit it on the training data and
> print its RMSE on that same training data, plus the number of leaves it ended
> up with.

**Expect:** RMSE exactly `0.0` — the literal float, not a rounded `$0`. Leaves:
**15,830** for 16,512 rows. Those two numbers together are the lesson: not one
leaf per row (rows with identical features and identical targets share a leaf),
but close enough that the tree has memorised the answer sheet. Every one of the
16,512 preprocessed rows is distinct, so nothing prevented it.

**Assert:** `assert root_mean_squared_error(y_train, tree.predict(X_train)) == 0.0`
— exact equality, and it holds.

**Annotate:** short — a zero training error is a statement about capacity, not
about accuracy. Keep this pipeline: everything after it reuses `preprocessing`.

---

## Cell 6 — measure it honestly

**Prompt to type:**

> Cross-validate three models — linear regression, a decision tree and a random
> forest of 100 trees — on the training set with 10-fold CV, scoring RMSE. Use the
> same folds for all three and make the folds reproducible. For each model print
> the mean, the standard deviation and the lowest and highest fold, not just the
> mean.

**Expect:**

| model | mean | sd | folds |
|---|---|---|---|
| Linear regression | $68,282 | $2,659 | $63,623 – $71,279 |
| Decision tree | $68,574 | $2,015 | $65,499 – $70,958 |
| Random forest | $48,687 | $2,911 | $43,329 – $52,189 |

(Standard deviations quoted with `ddof=1`. `np.std` on the fold array defaults to
`ddof=0` and gives $2,523 / $1,911 / $2,762 — state which you printed, they are
different numbers.) The tree that scored `0.0` one cell ago scores **$68,574**
here, which is the whole of section 3 in one comparison. Ten folds of 16,512 rows
means two folds of 1,652 and eight of 1,651: 10 × 1,651 + 2.

**Assert:** `assert len(folds) == 10` for each model, and — worth one extra line —
`assert sorted(np.concatenate([te for _, te in cv.split(X_train)])) == list(range(16512))`,
i.e. the folds are a partition, every row held out exactly once.

**⏱** 7.2 s measured on an Apple M4 Max, 16 cores, `n_jobs=-1` on the forest
(linear 0.1 s, tree 1.0 s, forest 6.1 s). On a free Colab CPU runtime, budget
**2–4 minutes**: the deck's own anchor is "about 25 s" for the tree alone, 25× the
1.0 s measured here. Print the machine you measured on next to the number, as the
deck does — an unattributed wall clock is not a wall clock.

**Annotate:** full

- **Left open:** *how to read the spread.* The forest's folds run from $43,329 to
  $52,189 — a range of **$8,860**, wider than the gap between the linear model and
  the tree by a factor of thirty. Any comparison in this notebook that turns on
  less than a couple of thousand dollars is not a comparison, and this cell is
  where that budget gets set.
- **The usual student version:** `cross_val_score(pipe, X, y, cv=10)`. Passing an
  integer builds `KFold(n_splits=10, shuffle=False)` — scikit-learn's documented
  default is not to shuffle — so the folds become ten contiguous blocks in row
  order. Two people whose frames are sorted differently then get different folds,
  different numbers, and no way to find out why. `KFold(n_splits=10,
  shuffle=True, random_state=RANDOM_STATE)` is nine extra characters and it is
  the difference between a measurement and a coincidence.
- **How you would catch it:** re-run the cell on `X_train.sample(frac=1,
  random_state=0)` and its matching `y`. With `shuffle=True` the three means move
  by nothing at all; with the integer form they move, and the amount they move by
  is the size of the error you were about to publish.

---

## Cell 7 — compare on the same folds

**Prompt to type:**

> Using the per-fold arrays from the last cell, compare the models pairwise fold
> by fold instead of comparing their averages: forest against linear, and tree
> against linear. For each pair print the mean difference, its standard deviation,
> how many of the ten folds each model won, and a paired t-test.

**Expect:**

| pair | mean diff | sd | folds won | p |
|---|---|---|---|---|
| forest − linear | −$19,594 | $1,650 | 10 / 10 to the forest | 0.000 |
| tree − linear | +$292 | $1,634 | **6 / 10 to the tree** | 0.586 |

Read the two rows together. The paired sd of $1,650 is smaller than either
model's own fold sd ($2,659 and $2,911) because both models feel the same
fold-to-fold difficulty and the difference cancels it. And the tree, whose mean
is *worse* than the linear model's, wins the majority of folds.

**Assert:** `assert (forest_minus_linear < 0).sum() == 10`. **No assert on the
tree row** — 6/10 at p = 0.59 is a coin, and an assertion would pin this seed's
coin-flip into the notebook as if it were a result.

**Annotate:** full

- **Left open:** *what to conclude when the pairing does not settle it.* The
  forest row is a result: same sign in all ten folds, a mean nineteen times the
  paired sd. The tree row is not: +$292 against a paired sd of $1,634. The prompt
  asks for both numbers and says nothing about which of the two situations you
  are in, and the printed output looks identical in both.
- **The usual student version:** ranking the two means — "$68,574 > $68,282, so
  the tree is worse" — and writing that sentence into the report. On these exact
  folds the tree is the *better* of the two on six of ten, and the paired t-test
  gives p = 0.586. That is a ranking read out of noise, and the deck calls it the
  single most transferable slide in the lecture.
- **How you would catch it:** the win count, printed next to the mean, every
  time. "10 of 10" and "6 of 10" are two different kinds of sentence, and no mean
  ± sd tells them apart at a glance.

---

## Cell 8 — tune the forest

**Prompt to type:**

> Run a grid search over the whole pipeline for the random forest: `max_features`
> in 4, 6, 8, 10, 12 and `n_estimators` in 30, 100, 200, scoring RMSE with 5-fold
> CV and `n_jobs=-1`. Print the best parameters and the best cross-validated RMSE.
> Also check whether the winning value of either parameter sits on the edge of the
> range we searched, and say so if it does.

**Expect:** 15 combinations. `best_params_` =
`{'model__max_features': 6, 'model__n_estimators': 200}`, best RMSE **$48,613**.
`n_estimators = 200` is the largest value in the grid, so the edge warning fires.
The 15 combinations span only **$1,328** end to end ($48,613 to $49,941).
**Write the winning parameters down on paper before you run cell 9.**

**Assert:** `assert len(search.cv_results_["params"]) == 15` — 5 × 3, an
arithmetic you can do before running anything.

**⏱** 25 s in an idle process on an Apple M4 Max, 16 cores; 60 s in the same
process after other fits had run. On a free Colab CPU runtime, budget **10–20
minutes and do not interrupt it** — 75 fits totalling 8,250 trees. The current
notebook says "2–4 minutes" and names no machine; the deck says 4 minutes for
*twice* this work. Those two claims cannot both be true, so time your own run and
write the machine down beside the number.

**Annotate:** short — the specification is: search the *pipeline*, not the
estimator, so the imputer and the scaler are refitted inside every fold; the
`model__` prefix addresses the step named `model`. The edge check must cover
**both** parameters — the current notebook tests only `n_estimators`, and
`max_features` has edges at 4 and 12 that would go unremarked.

---

## Cell 9 — the same grid, one thing changed

**Prompt to type:**

> Run exactly the same grid search again, but this time pass
> `KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)` as `cv` instead of
> the integer 5. Print both searches' best parameters and best scores side by
> side, and print how many candidate fits each one did.

**Expect:**

| `cv=` | best params | best RMSE | fits |
|---|---|---|---|
| `5` | max_features **6**, n_estimators 200 | $48,613 | 75 |
| `KFold(5, shuffle=True, random_state=42)` | max_features **8**, n_estimators 200 | $48,629 | 75 |

Same grid, same estimator, same seed, same number of fits, same wall clock — and
a **different winner**. `cv=5` selected `max_features=6`; every shuffled search
selects 8, including the 10-fold one on the slides (`$48,180`, and the notebook's
$48,613 is *not* comparable to it — different folds).

**Assert:** `assert len(a.cv_results_["params"]) == len(b.cv_results_["params"]) == 15`
— the assert that pins down "nothing else changed", which is the only claim this
comparison rests on (§2.1).

**⏱** another 25–60 s here / 10–20 minutes on Colab. **Re-run order for this
section: cell 8, then cell 9. Neither depends on anything between cells 6 and 8.**

**Annotate:** full

- **Left open, in cell 8 and on purpose:** `cv=5`. Passing an integer to
  `GridSearchCV` builds `KFold(n_splits=5, shuffle=False)` — the same default
  that cell 6 spent a whole annotation on, two sections earlier, arriving through
  a different door. The rule was stated; nothing restated it here; and a
  competent prompt asking for "5-fold CV" gets the unshuffled version.
- **The usual student version:** this *is* the usual version — `cv=5` is what
  almost everyone types, and it is what the current notebook ships. The
  instructive part is what happens next: the unshuffled search picks
  `max_features=6`, and that model then scores **$48,817** on the test set against
  the shuffled winner's **$49,037**. The invalid selection scored $220 *better*.
  Against a 95% interval half-width of $2,293, that difference is nothing — which
  means the test set will never tell you this happened. There is no downstream
  number that goes wrong.
- **How you would catch it:** only by reading the argument. `cv=5` and
  `cv=KFold(5, shuffle=True, random_state=42)` cost the same, run the same, print
  the same shape of output, and disagree about the answer. Write `KFold(...)`
  explicitly everywhere and the question never arises — which is why the deck's
  code comment is `# not cv=10: the default KFold does not shuffle`.

---

## Cell 10 — was the tuning worth it?

**Prompt to type:**

> Re-score the untuned 100-tree forest and the tuned one on the *same* folds —
> the shuffled 10-fold object from earlier — and print the difference between them
> next to the fold-to-fold spread of the untuned forest.

**Expect:** tuned $48,180 against untuned $48,687 on the same ten shuffled folds:
a gain of **$508**, against a fold sd of $2,762 (`ddof=0`) and a fold range of
$8,860. 150 fits bought an improvement smaller than a fifth of the measurement's
own scatter.

**Assert:** none — the point of the cell is that the difference is inside the
noise, and asserting it would be asserting noise.

**⏱** the tuned model is already fitted; this is ten forest fits of the untuned
one, about 6 s here / 2–4 minutes on Colab. Reuse `results["Random forest"]` from
cell 6 and it is free.

**Annotate:** short — the comparison is only legitimate because both numbers come
from the same fold object. Comparing the cell-8 score ($48,613, five unshuffled
folds) with the cell-6 score ($48,687, ten shuffled folds) is a $74 difference
between two different measurements, and it is the comparison the current notebook
invites.

---

## Cell 11 — the test set. Once.

**Prompt to type:**

> Take the tuned model, predict on the test set, and report the RMSE with a 95%
> bootstrap confidence interval. Bootstrap the squared errors and use the
> percentile method — name it explicitly. Print the cross-validated estimate next
> to it and have the code state whether the two actually agree, rather than
> asserting it in a comment.

**Expect:** test RMSE **$49,037**; 95% percentile interval **$46,752 –
$51,379** (half-width $2,313); cross-validated estimate $48,180; gap $858, which
is inside the interval, so the printed verdict is `True`. 4,128 test districts,
201 of them at the $500,001 cap the model structurally cannot reach.

**Assert:** `assert lo < final_rmse < hi` — a point estimate outside its own
percentile interval means you bootstrapped the wrong array. And print the
agreement as a computed boolean: `print(f"CV estimate inside the interval:
{lo <= cv_estimate <= hi}")`.

**Annotate:** full

- **Left open:** *what to do with the answer.* Nothing in the prompt stops you
  going back to cell 8 and widening the grid because you did not like $49,037.
  The moment you do, this number stops being a held-out estimate and becomes a
  training score with extra steps, and no code will complain. The interval is
  what makes the temptation resistible: $858 against ±$2,313 is not a gap, it is
  the same number measured twice.
- **The usual student version:** omitting `method=`. `scipy.stats.bootstrap`
  defaults to **`method="BCa"`**, not the percentile bootstrap — check the
  signature and you will find `method='BCa'` sitting there. A notebook whose prose
  says "percentile bootstrap" and whose code omits the argument is describing an
  estimator it did not run. On this data the two barely differ — BCa gives
  $46,675 – $51,310 against percentile's $46,752 – $51,379, about $80 at each end
  — and that is exactly why nobody notices for years.
- **How you would catch it:** bootstrap the **squared errors** and take the
  square root of the interval, never bootstrap the RMSE of a resampled *model*.
  Resampling the model is a different and much slower quantity, and if your
  interval comes out at a few hundred dollars wide rather than a few thousand,
  that is what you did.

---

## Cell 12 — one line of the mathematical thread

**Prompt to type:**

> Show numerically that for the percentile bootstrap it makes no difference
> whether you bootstrap the mean squared error and square-root the interval, or
> bootstrap the RMSE directly. Then show the same thing for `method="basic"`.

**Expect:** for `percentile` (and for `BCa`) the two intervals agree to machine
precision — the percentile bootstrap is equivariant under any increasing
transformation, and √ is one. For `method="basic"` they differ by a few tens of
dollars, because the basic bootstrap reflects around the point estimate and
reflection does not commute with √.

**Assert:** `assert np.allclose(np.sqrt(ci_from_mean), ci_from_rmse)` for
`percentile`, and `assert not np.allclose(...)` for `basic`.

**Annotate:** short — this is the cell that retires a piece of folklore. "Take
the mean, then the square root" is presented in the current notebook as a
correctness requirement; for the estimator it insists on, it is a free choice.
The place the ordering genuinely matters is the *t*-interval, which needs the
delta method — which is the reason the deck bootstraps at all.

---

## Cell 13 — where the error actually falls

**Prompt to type:**

> Break the tuned forest's error down by `ocean_proximity` and by income band.
> Give the RMSE and the number of districts in each group, sorted worst first.

**Expect:** five ocean groups and five income bands, each with a count. Read the
overall level of the table against the $49,037 you printed in cell 11 before you
go on, and **write down whether the two look like the same model.**

**Assert:** `assert grouped["n"].sum() == len(<whatever rows you used>)` — and
notice which number that turns out to be.

**Annotate:** short — the specification is: never a group RMSE without its count
beside it. A group of two districts and a group of seven thousand cannot go in
the same column unlabelled.

---

## Cell 14 — the two tables

**Prompt to type:**

> Print that same breakdown twice — once on the training rows and once on the
> test rows — with the counts, side by side, and print the overall RMSE of each.

**Expect:**

| ocean_proximity | n (train) | RMSE (train) | n (test) | RMSE (test) |
|---|---|---|---|---|
| NEAR BAY | 1,846 | $20,355 | 444 | $60,195 |
| NEAR OCEAN | 2,089 | $21,644 | 569 | $57,191 |
| ISLAND | **2** | $67,641 | **3** | $54,754 |
| <1H OCEAN | 7,274 | $18,423 | 1,862 | $48,967 |
| INLAND | 5,301 | $13,318 | 1,250 | $39,829 |
| **overall** | 16,512 | **$17,680** | 4,128 | **$49,037** |

The two RMSE columns differ by a factor of about three, and the ordering of the
groups is not even the same. On the test rows the honest finding is that
**NEAR BAY is 51% worse than INLAND** ($60,195 against $39,829), which is a
different report to the client from a single $49,037.

**Assert:** `assert test_groups["n"].sum() == 4128` and
`assert train_groups["n"].sum() == 16512` — the counts are the whole point, so
assert them.

**Annotate:** full

- **Left open, in cell 13 and on purpose:** *which rows.* "Break the tuned
  forest's error down" does not say what to break it down over, and the rows in
  scope are the training rows. The assistant will use them, the table will print
  cleanly, and nothing in it says "these are the rows the model was fitted to".
  The train column above is a forest quoting its own homework back — the same
  error the tree in cell 5 exists to teach, six cells later, with a better model
  and no zero to give it away.
- **The usual student version:** this is the current notebook's own cell, where
  the variable is even called `pred_train_cv` although nothing about it is
  cross-validated. Its ISLAND row is computed from **2** districts while the prose
  beside it says "five districts in the whole state" — five is the count in all of
  California; the split put 2 in train and 3 in test. The deck states the rule the
  notebook breaks: *report it with the count or do not report it.*
- **How you would catch it:** compare the table's overall figure against the
  headline you printed in cell 11. $17,680 against $49,037 is a factor of 2.8,
  and there is only one thing that makes an error table three times too good.
  Anything you conclude about *fairness* from the left-hand column is a
  conclusion about memorisation.

---

## Cell 15 — what an unseen category encodes to

**Prompt to type:**

> Fit a `OneHotEncoder(handle_unknown="ignore")` on the training
> `ocean_proximity` column with the ISLAND rows removed, then ask it to transform
> a single ISLAND row and print the result and its sum.

**Expect:** `[0. 0. 0. 0.]`, sum `0.0`, four categories in `categories_`, and no
error and no warning about the unknown category. Every other district's row sums
to 1; ISLAND's sums to 0, so the model is told "none of the above" — a prediction
nobody made deliberately.

**Assert:** `assert enc.transform(one_island).sum() == 0` and
`assert enc.transform(one_island).shape == (1, 4)`

**Annotate:** short — pass the row as
`pd.DataFrame({"ocean_proximity": ["ISLAND"]})`, not as `[["ISLAND"]]`. The list
form works but raises `UserWarning: X does not have valid feature names, but
OneHotEncoder was fitted with feature names`, and a cell whose subject is a silent
failure should not ship a warning it never mentions.

*One correction to make while you are here.* The claim "with ten folds, some folds
contain no ISLAND training row at all" is false on this split and takes four lines
to check: under `KFold(10, shuffle=True, random_state=42)` **every** fold's
training part contains at least one of the two ISLAND districts (folds 1 and 7
have one, the other eight have both). The true statement is milder and still
worth the cell: in two of the ten folds the model is trained on a *single* ISLAND
district and asked to predict another, and under the unshuffled `cv=5` of cell 8
the same thing happens in folds 3 and 4.

---

## Closing section — red-team, and what to record

Keep the five questions, but make them performable by a student alone at home.
The current ending opens "Swap notebooks with the team beside you. Ten minutes",
which the reader at 23:00 in their kitchen cannot do; put the pairing in the
lecture slides and give the notebook a version that runs on one person's own
file:

1. What touched the test set, and in which cell? (Answer here: cell 11, once.)
2. What was fitted, and on what? `fit` and `transform` are different verbs.
3. What is the shape here?
4. What was dropped — rows, columns, NaNs? Count them.
5. **What is the default I did not ask for?** Three appear in this notebook:
   `KFold(shuffle=False)`, `stats.bootstrap(method="BCa")`, and
   `LinearRegression`'s least-squares solver. Find all three, then find a fourth.

And one exercise with its re-run order stated, per §7.2:

> **Change the seed.** Set `RANDOM_STATE = 0` in **cell 1**, then re-run **cells
> 3, 6, 7, 8, 9, 11** in that order — cell 3 rebuilds the split, so nothing
> between them can be skipped. Cells 4, 5, 12 and 15 are unaffected. Expect the
> fold means to move by a few hundred dollars and the grid search's
> `max_features` winner to move as well; if the *ordering* of the three models in
> cell 6 changes, that is a finding worth writing down.

---

## Defects found in the current notebook

Every item below was re-derived with python3 against
`notebooks/lecture-02.ipynb`, `tools/make_notebooks.py::lecture_02`,
`tools/make_notebooks.py::lecture_01` and `slides/lecture-02.html`. Numbers are
from a seeded run on scikit-learn 1.7.2 / numpy 2.3.5 / pandas 2.3.3 / scipy
1.16.3, Apple M4 Max, 2026-08-02.

**Verification status: every claim in A–Q was checked by running it.** Nothing
here is asserted from reading alone. Two qualifications, both about method
rather than result:

- To settle A, B, C, D and K, the notebook's two long cells had to be executed —
  the 10-fold cross-validation and the grid search — in a scratchpad copy, never
  in the repository. Without running them there is no way to know that `cv=5`
  changes the winner, and §1 outranks the convenience of not running anything.
  No file outside `tools/prompts/lecture_02.md` was written.
- K's *Colab* figures are the only estimates in this file. The measured numbers
  are from this machine and are labelled as such; the Colab column is scaled from
  the deck's own anchor and is given as a range precisely because the deck's two
  anchors disagree. Anyone with a Colab session should replace them with a
  measurement.

### A. The notebook breaks its own rule one section after teaching it

**§5 passes `cv=5` to `GridSearchCV`.** An integer builds
`KFold(n_splits=5, shuffle=False)` — precisely the default that §4's own markdown
calls out: *"`shuffle=True` is not decoration. The default `KFold` does not
shuffle."* It is not cosmetic. Measured:

| `cv=` | best params | best CV RMSE |
|---|---|---|
| `5` (what the notebook ships) | max_features **6**, n_estimators 200 | $48,613 |
| `KFold(5, shuffle=True, random_state=42)` | max_features **8**, n_estimators 200 | $48,629 |
| `KFold(10, shuffle=True, random_state=42)` (the deck) | max_features **8**, n_estimators 200 | $48,180 |

The shuffle, not the fold count, is what changes the winner. (§2.1, §4.3)

### B. The notebook and its own slide deck disagree on the answer

`slides/lecture-02.html` prints
`{'model__max_features': 8, 'model__n_estimators': 200}` and `48179.64…`, and its
figure highlights "the best cell (8, …)". The notebook produces
`max_features: 6` and `$48,613`. The notebook's markdown explains the difference
away as wall clock — *"The lecture's figure uses `cv=10`, which takes twice as
long; five is enough here"* and *"Five is enough to choose between these
fifteen"* — which is the one thing it is not: five *shuffled* folds also pick 8.
A student who follows the deck and then runs the notebook gets a different model
and no explanation. (§1.5, §3.3)

### C. §6 measures the model on the rows it was fitted to

`pred_train_cv = best.predict(X_train)` — nothing about it is cross-validated,
and the whole group breakdown is the tuned forest grading its own homework, which
is the error §3 of the same notebook exists to teach. Overall training RMSE
**$17,680** against a test RMSE of **$49,037**; the group ordering is different
too (train worst = NEAR OCEAN $21,644, test worst = NEAR BAY $60,195). The deck
does this table on the **test** set. Nothing in the notebook flags the difference.
(§2.1, §10.10)

### D. Group RMSEs printed without counts, and a count that contradicts the prose

The tables give no `n`. The ISLAND row of the training table is computed from
**2** districts; the markdown immediately below says *"`ISLAND` — five districts
in the whole state"*. Five is the whole-California count; the split puts 2 in
train and 3 in test, and `slides/lecture-02.html` has the three-column table
saying exactly that, plus the rule the notebook breaks: *"the ISLAND number is
computed from three districts: report it with the count or do not report it."*
(§1.1, §1.3)

### E. "With ten folds, some folds contain no ISLAND training row at all" — false

Under `KFold(10, shuffle=True, random_state=42)` every fold's training part
contains at least one ISLAND district: folds 1 and 7 contain one, the other eight
contain both. Zero folds train without ISLAND. Four lines to check. (§1.1, §3.2)

### F. An invented library default — the bullet §6.2 exists to forbid

> *"`observed=True` on the groupby. Without it pandas produces a row for every
> unobserved combination of categories, full of NaN, and the table becomes
> unreadable."*

Neither half holds here. `err["ocean"]` is `object` dtype, where `observed` has no
effect at all; `err["income_cat"]` is categorical with all five bands observed.
Run both groupbys with `observed=False`: **five rows each, identical, no NaN.**
"Every unobserved *combination*" would need two or more categorical keys, and
both groupbys have one. (§6.2)

### G. The `np.linalg.inv` claim is backwards, and it appears twice

> *"`np.linalg.inv` is not what scikit-learn uses. It computes the pseudoinverse
> via SVD, which still returns an answer when XᵀX is singular"*

Read literally the subject of "It" is `np.linalg.inv`, which does not compute a
pseudoinverse. Measured on a design matrix with one exactly collinear column
(`cond(XᵀX) = 1.6e+17`): `np.linalg.inv` **does not raise**, returns ‖θ̂‖ = 1.8e6,
and the fit's training RMSE is **$160,835**; `LinearRegression`, via
`scipy.linalg.lstsq`, returns the minimum-norm solution and **$69,188**. So it is
`inv` that silently returns an answer, and a wrong one, and scikit-learn that is
safe — the opposite of what the sentence says. It is written once in the prompt
box (cell 9) and once in the markdown (cell 11), so a reader meets it twice.
(§1.1, §3.2)

### H. A cross-reference to a warning that was never given

> *"…it is why you were warned against engineering a feature as a weighted sum of
> existing ones."*

`lecture_01()` contains no such warning: zero occurrences of "weighted sum",
"collinear", or any equivalent. (Lecture 1 *does* warn about ISLAND, twice, and
that reference resolves — although lecture 1 promises the all-zero encoding "two
sections into the next lecture" and it is in section **6**.) Likewise §6's *"Two
things the previous lecture promised and never did"* — lecture 1 promises the
three training RMSEs will be diagnosed and that ISLAND will come back; it never
promises an error breakdown by group, so the two things are not findable. (§3.3,
§7.3)

### I. A conclusion printed as prose instead of being computed

```
print("The two agree within the interval. A gap smaller than the interval is "
      "not evidence of anything.")
```

Unconditional. It prints whether or not they agree. On this seed they do
($48,613 lies inside $46,567 – $51,152), so the sentence is currently true and
will keep printing when it is not. One boolean fixes it. (§3.2)

### J. A distinction insisted on that does not exist for the chosen method

> *"bootstrap the SQUARED ERRORS and take the square root of the interval, rather
> than bootstrapping the RMSE directly."*

For `method="percentile"` the two are **identical to machine precision** —
percentile intervals are equivariant under increasing transformations, and so, as
it happens, is scipy's BCa. Verified on 4,128 skewed squared errors: percentile
gives the same two endpoints either way; only `method="basic"` differs. The
notebook presents a free choice as a correctness requirement, in a box whose
neighbouring bullet is about naming the estimator you actually ran. (§1.1)

### K. Timings with no machine attached, and mutually inconsistent

The notebook says *"⏱ about 90 seconds — thirty fits in total, ten of them
forests"* and *"⏱ 2–4 minutes. Fifteen combinations × five folds = 75 forest
fits."* Both arithmetic counts are right; neither figure names a machine, and the
deck's convention — *"Timings are for a free Colab CPU runtime and are not
examinable"* — is not carried over. Measured here (Apple M4 Max, 16 cores,
`n_jobs=-1`): the 10-fold cell **7.2 s**, the grid search **25 s** idle / **60 s**
under contention. Scaling from the deck's own anchor (`# about 25 s` for a
tree-only 10-fold, 25× the 1.0 s measured here) puts the 10-fold cell at ~3
minutes and the grid at ~10–25 minutes on Colab — against the deck's separate
claim of 4 minutes for *twice* the grid work. The deck's two anchors disagree with
each other by roughly 5×, and the notebook inherits both. A reader on Colab told
"2–4 minutes" will interrupt a working cell. (§7.1)

### L. The header promises a marker that no cell carries

> *"Cells marked **⚠ read before running** contain a defect on purpose."*

The string "read before running" occurs exactly once in the notebook — in that
sentence. The only other `⚠` is inside the edge-of-grid `print`. Lecture 2 has no
marked cell, and (see A) its one real defect is unmarked. (§3.3, §8.2)

### M. Examinability marked on one section of eight

"examinable" appears twice in the whole notebook, both inside the setup cell's
"not examinable" note. Sections 2–8 carry nothing. (§8.3)

### N. Instructions a student alone cannot carry out

§8 opens *"Swap notebooks with the team beside you. Ten minutes."* — the entire
closing exercise is unperformable by the reader the guidelines are written for.
There is also no exercise anywhere that states a re-run order, and the only
plausible one ("change the seed") would span cells 3, 7, 17, 20, 23, 26 and 32.
(§7.2, §10.9)

### O. The `import` cell's own claim, and a dead import

Cell 7's comment reads *"Every import this notebook needs, in one place"*;
`from scipy import stats` is in cell 32. `matplotlib.pyplot as plt` is imported
and never used anywhere in the notebook. (§3.2)

### P. Smaller, still checkable

- The edge-of-grid check tests `n_estimators` only. `max_features` has edges at 4
  and 12 and is not tested; the prose calls it "the edge-of-grid check" without
  qualification.
- `enc.transform([["ISLAND"]])` emits `UserWarning: X does not have valid feature
  names, but OneHotEncoder was fitted with feature names`, unmentioned; and
  `transform` is called twice on the same input to print two facts about one row.
- §5 never says whether the tuning was worth it. Its $48,613 against the untuned
  $48,687 is a $74 difference — and the two are not on the same folds anyway
  (five unshuffled vs ten shuffled). Scored properly on the same ten shuffled
  folds the gain is $508 against a fold sd of $2,762, which the deck states
  plainly (*"150 fits bought us something we cannot cleanly distinguish from
  noise"*) and the notebook drops. (§2.4)
- The tree's training RMSE prints as `$0` via `:,.0f`. It is exactly `0.0`, and
  the exact-equality assert is available and stronger than a formatted zero.

### Q. Checks that pass

Run and clean, so that a later reviewer does not repeat them: no markdown line is
indented ≥ 4 outside a fence (§5.1); no fence is mis-indented (§5.2); markdown
contains no fenced Python block at all, so §3.1 is vacuous; no name is bound to two
different kinds of object across cells (§4.1) — the only repeated binding is the
`for name, model` loop variable, bound once; the §4 claim that "the folds span
several thousand dollars" is true on all three models ($7,656 / $5,459 / $8,860);
and `assert len(X_train) == 16512 and len(X_test) == 4128` holds.
