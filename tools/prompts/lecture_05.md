# Lecture 5 — Colab prompt script

**Who survives, and how sure are we?** · Titanic · Géron ch. 4 · *Build*

This is the script for rebuilding `notebooks/lecture-05.ipynb` at a Colab
keyboard by prompting. Type the prompt, read what comes back, check it against
**Expect**, then add the **Assert** yourself if the assistant did not.

**Cell N** below means the **N-th code cell**, counting from 1. The mapping to
the current notebook's cell indices is in the table at the end, so every
cross-reference in this file resolves.

**Annotation budget.** 7 of the 29 cells carry the full three-bullet
annotation: **10, 11, 13, 17, 21, 25, 28**. The rest carry the specification
only. The current notebook annotates all 30 of its boxes; that is the defect
this script repairs, and §6.1 is the reason.

---

## Timings — what was actually measured

Every figure below was measured on a 16-core Apple M4 (`Mac16,6`),
scikit-learn 1.7.2, run twice: once unconstrained, once with
`OMP_NUM_THREADS=1` and `n_jobs=2` to approximate a free Colab CPU runtime
(two cores).

| Cell | Notebook currently claims | Measured, unconstrained | Measured, 2 cores / 1 BLAS thread |
|---|---|---|---|
| 13 · 20 seeds | ⏱ "about two minutes" | 7.2 s | 2.2 s |
| 14 · cross_validate ×3 metrics | ⏱ "about 20 seconds" | 0.3 s | 0.1 s |
| 22 · calibration | ⏱ "about 15 seconds" | 0.2 s | 0.1 s |
| 26 · degree sweep | ⏱ "about 30 seconds" | 7.2 s | 7.3 s |
| 28 · convergence table | no marker | 1.2 s | 2.2 s |
| whole notebook | — | 16.5 s | — |

The 2-core run is *faster* on cell 13 because 20 seeds × 5 folds of a 712-row
logistic regression is dominated by joblib process spawn, not by arithmetic.
A free Colab core is slower than an M4 core; allow 3–5× on top of the
unconstrained column and **still** nothing in this notebook reaches 20 s.
§7.1 therefore asks for a ⏱ on **no cell here**. I have kept a ⏱ line on the
two slowest anyway, with the number, because the current notebook's figures
are wrong by 10–100× and a reader who has been told "two minutes" will sit
waiting through a cell that finished while they were reading the box.

I did not measure any of this *on* Colab. Treat the third column as a floor.

---

## Staging — what NOT to say, and when

Two defects are staged in this lecture, and the current notebook gives both of
them away before they happen (§8.1). Type these prompts in the order below and
say nothing extra:

- **The `FamilySize` dependency.** Cell 7 builds it. Do **not** flag it, do not
  write "remember this line", do not mention section 9. The reveal is Cell 18,
  and it lands only if the reader wrote the feature themselves without
  suspicion. The current notebook announces it four times before Cell 18.
- **The 0.5 threshold.** Cell 11 runs and prints an accuracy. Have the reader
  **write the accuracy down**. Only then ask the review question — *what
  cut-off produced that label, and who chose it?* — and go to Cell 12. One ⚠,
  after the fact, not four before it.

---

## Cell 1 — setup, versions, seed

**Prompt to type:**
> print the python, scikit-learn, numpy and pandas versions, assert scikit-learn is at least 1.4, and set RANDOM_STATE = 42 for the whole notebook

**Expect:** four version lines and nothing else. No model, no data.
**Assert:** `tuple(int(p) for p in sklearn.__version__.split(".")[:2]) >= (1, 4)`
**Annotate:** short

> Note: the current notebook justifies the version assert with
> `root_mean_squared_error` arriving in 1.4. That function is never called in
> this notebook. Give the real reason or drop the sentence — the reason to pin
> a version here is `OneHotEncoder`'s `min_frequency` and `infrequent_if_exist`,
> which Cell 10 depends on and which arrived in 1.1.

---

## Cell 2 — every import, in one place

**Prompt to type:**
> put every import this notebook will need in this one cell: matplotlib, pathlib, tarfile, urllib, warnings, and from sklearn the ColumnTransformer, SimpleImputer, LogisticRegression, accuracy / log_loss / brier_score_loss, StratifiedKFold, cross_validate, cross_val_predict, train_test_split, Pipeline, make_pipeline, OneHotEncoder, PolynomialFeatures, StandardScaler

**Expect:** imports only, plus `NJ = 4` for `n_jobs`. No output.
**Assert:** none — the test is Restart-and-run-all, later.
**Annotate:** short

---

## Cell 3 — the data, as a function

**Prompt to type:**
> write load_titanic() that downloads https://github.com/ageron/data/raw/main/titanic.tgz into datasets/ if it isn't already there, extracts it, and returns datasets/titanic/train.csv as a dataframe. call it and show head()

**Expect:** `891 passengers, 12 columns`, then the first five rows.
**Assert:** `full.shape == (891, 12)`, `"Survived" in full.columns`,
`full["Survived"].isin([0, 1]).all()`
**Annotate:** short

---

## Cell 4 — what is missing, counted

**Prompt to type:**
> for every column with at least one missing value, print the count and the percentage of 891

**Expect:** exactly three rows —
`Cabin 687 (77.1%)`, `Age 177 (19.9%)`, `Embarked 2 (0.2%)`.
**Assert:** `missing.to_dict() == {"Cabin": 687, "Age": 177, "Embarked": 2}`
**Annotate:** short

---

## Cell 5 — the base rate

**Prompt to type:**
> how many of the 891 survived, and what fraction

**Expect:** `survived 342 of 891  base rate 0.3838`.
**Assert:** `int(full["Survived"].sum()) == 342`
**Annotate:** short

> Say it here, in the markdown under this cell: 0.3838 is the base rate of the
> **whole file**. The anchor computed in **Cell 9** is 0.3834, the base rate of
> the **training rows only**, and the two are not the same number. The current
> notebook calls the anchor "three cells down" from here; it is thirteen code
> cells down, and it is computed from a different set of rows.

---

## Cell 6 — the rule before the model

**Prompt to type:**
> survival rate by sex, and then by sex crossed with class, with the group sizes

**Expect:** female 0.742 / male 0.189; then six rows, from
`female class 1  0.968  n=94` down to `male class 3  0.135  n=347`.
**Assert:** `tab.loc[("female", 1), "mean"] > 0.95` and
`tab.loc[("male", 3), "mean"] < 0.15`
**Annotate:** short

---

## Cell 7 — features out of a name

**Prompt to type:**
> pull the title out of Name (the bit between the comma and the full stop), fold Mlle and Ms into Miss and Mme into Mrs, keep only Mr / Mrs / Miss / Master and call everything else Rare. also add FamilySize = SibSp + Parch + 1, IsAlone, and Deck from the first letter of Cabin with U for missing

**Expect:** `{'Mr': 517, 'Miss': 185, 'Mrs': 126, 'Master': 40, 'Rare': 23}`,
then survival by title. There are **17** distinct raw titles before the
collapse.
**Assert:** `set(full["Title"]) == {"Mr","Mrs","Miss","Master","Rare"}`; no
nulls in `Title`, `Deck`; `IsAlone` in `{0,1}`.
**Annotate:** short

> **Say nothing about `FamilySize`.** See *Staging*, above.

---

## Cell 8 — split first, stratified on the label

**Prompt to type:**
> split off 20% as a test set, stratified on y, with random_state=RANDOM_STATE

**Expect:** `train rate 0.3834   test rate 0.3855`, 712 and 179 rows.
**Assert:** sizes 712/179; `set(X_train.index).isdisjoint(X_test.index)`;
`abs(y_train.mean() - y_test.mean()) < 0.01`.
**Annotate:** short

> The base-rate assert is the one that has teeth. Sizes are 712/179 whether or
> not `stratify=` was passed.

---

## Cell 9 — what nothing scores

**Prompt to type:**
> without fitting anything: the accuracy of always predicting "did not survive", and the log loss and Brier score of always reporting the training base rate

**Expect:**
`base rate 0.3834`, `accuracy 0.617`, `log loss 0.666`, `Brier 0.236`.
**Assert:** `abs(constant_log_loss - 0.666) < 0.001` and
`abs(majority_accuracy - 0.617) < 0.001`
**Annotate:** short

> 0.666 is the entropy of the label in nats. It is the anchor for the rest of
> the session, and it is **not** 0.693 — that is the anchor for a balanced
> label. Write it on the COMMIT sheet now.

---

## Cell 10 — preprocessing inside the pipeline

**Prompt to type:**
> build a ColumnTransformer: median-impute and standardise the numeric columns, most-frequent-impute and one-hot the categoricals with min_frequency=2, pass IsAlone through. fit it on the training rows and tell me how wide it gets

**Expect:** `11 raw columns become 28 model columns`, 712 rows, no NaN.
**Assert:** `Z.shape[0] == 712` and `not np.isnan(Z).any()`
**Annotate:** full

- **Left open:** what `min_frequency=2` *does to the names*. It does not drop a
  rare level; it merges every level below the threshold into one column called
  `<col>_infrequent_sklearn`. Here that column is Deck alone, and its
  membership is exactly `['T']` — one passenger. Nothing in the prompt asks to
  be told this, and the consequence surfaces eleven cells later in Cell 15.
- **The usual student version:** `OneHotEncoder()`'s default is
  `sparse_output=True`, so it returns a `csr_matrix` and `np.isnan(Z).any()`
  raises `TypeError: ufunc 'isnan' not supported`. It does not raise here only
  because `ColumnTransformer`'s default `sparse_threshold=0.3` densifies the
  stack when the density exceeds 0.3, and this stack lands at **0.379**. Add
  two more categorical levels and the same assert starts throwing. The other
  half of the same default: `SimpleImputer().fit_transform(X)` on its own line
  before the split computes the median over the test rows too.
- **How you would catch it:** print `type(Z)` beside `Z.shape`, and print
  `enc.infrequent_categories_` beside `get_feature_names_out()`. Both are one
  line, and between them they pre-empt the two things this cell hides.

---

## Cell 11 — ⚠ what "build me a classifier" returns

**Prompt to type:**
> build me a classifier for this and tell me how accurate it is

**Expect:** a fitted pipeline and one line — `accuracy: 0.844`.
**Assert:** `y_pred.shape == (179,)` and `set(np.unique(y_pred)) <= {0, 1}`
**Annotate:** full

- **Left open:** the cut-off, and therefore the whole decision. Nothing in that
  sentence mentions a probability, a threshold, or the fact that the two
  mistakes cost different amounts — and nothing in the code that comes back
  mentions them either. It runs, and it prints a number people will quote.
- **The usual student version:** taking `0.844` to the report. `predict()`
  thresholds `predict_proba` at **0.5**, which is correct if and only if a
  false negative and a false positive cost the same. The stakeholder said 10
  to 1. Second real default in the same three lines: `LogisticRegression()`
  is `C=1.0, penalty="l2"` — it is *already regularised*, so "plain logistic
  regression" is not what got fitted.
- **How you would catch it:** whenever you see `.predict()` on a problem with
  unequal costs, ask which threshold it used. If the answer is "the default",
  the threshold is unowned — a decision about human life taken by a library,
  silently, on your behalf.

> **Now, and only now:** ask the reader for the number they wrote down, then
> ask *who chose the cut-off*. This is the one ⚠ this section gets.

---

## Cell 12 — the cost of the rule

**Prompt to type:**
> the stakeholder says a passenger who drowns unescorted costs ten times an escort wasted on someone who would have survived. write total_cost(prob, y_true, t) for the rule "escort everyone whose survival probability is below t", and a grid of cut-offs from 0.02 to 0.98

**Expect:** no output. A function and `GRID` of 97 points.
**Assert:** none. Sanity-check instead: `total_cost(p, y, 0.98)` must flag
nearly everyone, so it should be *dominated by* false positives, not cheap.
**Annotate:** short

---

## Cell 13 — the default cut-off, over 20 seeds

**Prompt to type:**
> do this over 20 different train/test splits: pick the cut-off by minimising that cost on out-of-fold predictions of the training part, then score both 0.5 and the chosen cut-off on the held-out part. report mean cost and mean accuracy for each

**Expect:**

```
cut-off 0.50 (what accuracy chooses)  cost  157.00 ± 32.82   accuracy 0.829
cut-off chosen on cost                cost   55.40 ± 9.21    accuracy 0.746
the default costs 101.60 more, or 183% above the best rule
seeds where 0.50 is the worse rule: 20 of 20
mean chosen cut-off: 0.865
```

**Assert:** `(ch >= cb).all()` and `ab.mean() < ah.mean()` — both directions.
The cost-chosen rule is never beaten on cost **and** it is worse on accuracy.
**⏱** 7.2 s unconstrained, 2.2 s at `n_jobs=2` on an M4. Not the two minutes
the current notebook claims.
**Annotate:** full

- **Left open:** the *direction* of the rule. `flag = prob < t` is one
  character away from `flag = prob > t`, and both produce a curve that looks
  like a cost curve.
- **The usual student version:** writing `flag = prob > t`. I ran it: the
  minimum lands on the **first grid point**, t = 0.02, at cost **371** — which
  is lower than the 708 that 0.5 scores, so it reads as a real saving rather
  than as an error. (The inverted curve is not monotone, which is why "the
  curve went monotone" is not the tell.) The correct rule's minimum is at
  t = 0.83, cost 210, one grid point from neither edge. The other real
  default: `min(GRID, key=...)` returns the *first* minimiser; here exactly one
  grid point achieves 210, so nothing is hidden — but check that, do not
  assume it.
- **How you would catch it:** a cost minimum sitting on the edge of your grid
  is almost always a sign inversion. And report the seed count: **20 of 20** is
  an argument; a mean difference across one seed is not a measurement.

> Both halves matter. The cost-optimal rule is worse on accuracy by 8.3 points
> and better on cost by a factor of 2.83 — the thing the stakeholder actually
> said they cared about.

---

## Cell 14 — fit it properly, three metrics

**Prompt to type:**
> 10-fold stratified CV of that pipeline with C=1e6 to switch the penalty off, scoring log loss, accuracy and Brier, and give me the training scores too. print the per-fold held-out log loss, not just the mean

**Expect:**

```
log loss   train 0.395   held-out 0.496
accuracy   train 0.840   held-out 0.815
Brier      train 0.121   held-out 0.134
per-fold held-out log loss: [0.422 0.43  1.134 0.501 0.337 0.39  0.457 0.471 0.402 0.417]
```

**Assert:** `len(r["test_neg_log_loss"]) == 10` and
`-r["test_neg_log_loss"].mean() < constant_log_loss`
**⏱** 0.3 s. The current notebook says 20 s; delete that marker.
**Annotate:** short

> Two things to say out loud here, both real defaults. `cross_validate` has
> `return_train_score=False` by default — without it, this cell measures half
> the experiment. And `scoring=` with a **list** names its columns
> `test_<scorer>`, while a single **string** names them `test_score`; mixing
> the two conventions is a `KeyError`, and Cell 21 uses the string form.
>
> Look at fold 3: **1.134**, against a mean of 0.496. One fold is 2.7× the
> others. Everything claimed about differences of 0.03 later has to survive
> that spread — the std across these ten folds is **0.217**.

---

## Cell 15 — read the coefficients

**Prompt to type:**
> show me the six largest coefficients by absolute value, with the ColumnTransformer prefixes stripped, as log-odds and as odds multipliers

**Expect:**

```
Deck_infrequent_sklearn -5.678   odds x0.003
Title_Master     +3.484   odds x32.596
Sex_female       +3.308   odds x27.325
Sex_male         -2.833   odds x0.059
Title_Miss       -2.806   odds x0.060
Title_Mrs        -2.020   odds x0.133
```

**Assert:** `len(names_v1) == len(coefs_v1)` — a length mismatch mislabels
every row of the table silently.
**Annotate:** short

> Two things in that list should stop the reader, and **the first one is not
> what the current notebook says it is.**
>
> 1. `Sex_female` and `Sex_male` both carry large weights with opposite signs.
>    There are two sexes; one indicator carries everything the pair can carry.
> 2. The largest weight in the model is `Deck_infrequent_sklearn`, **not**
>    `Deck_T`. There is no `Deck_T` column — `min_frequency=2` merged it (see
>    Cell 10). Ask how many passengers are in that bucket before believing it.

---

## Cell 16 — how many people is that weight fitted to

**Prompt to type:**
> print the passenger count per deck for the whole dataset, and separately how many are on deck T

**Expect:**
`{'U': 687, 'C': 59, 'B': 47, 'D': 33, 'E': 32, 'A': 15, 'F': 13, 'G': 4, 'T': 1}`
then `deck T: 1 passenger(s)`.
**Assert:** `int((full["Deck"] == "T").sum()) == 1`
**Annotate:** short

> Close the loop the current notebook leaves open: print
> `prep.named_transformers_["cat"][-1].infrequent_categories_` here too. It
> returns `['T']` for Deck and `None` for everything else. *That* is what ties
> the −5.678 in Cell 15 to the single passenger in this cell. Without it the
> reader is asked to believe a coefficient whose name they have never seen.
> The one deck-T passenger did not survive, which is why the weight is
> negative.

---

## Cell 17 — count the columns, then count the rank

**Prompt to type:**
> add an intercept column to the transformed training matrix and give me its column count, its rank and its condition number

**Expect:** `columns: 29`, `rank: 23`, `cond: 3.16e+16`, deficiency 6.
**Assert:** `n_cols == 29 and rank == 23`
**Annotate:** full

- **Left open:** where the six come from. Two cells answer it — the five
  one-hot blocks (each sums to 1 on every row, and so does the intercept) and
  one feature engineered by hand in Cell 7. Do not name the second one yet.
- **The usual student version:** never computing the rank at all. Nothing
  errors, `lbfgs` returns, `coef_` is populated, and the coefficients are
  meaningless. The tell people reach for instead is `.corr()`, which reports
  0.98 and reads as "highly correlated" — a warning, not a proof.
- **How you would catch it:** rank against column count, on the design matrix
  **with** the intercept, because the intercept is what each one-hot block is
  dependent *with*. Do not quote the condition number to two figures: the
  matrix is singular, its six smallest singular values are between 1.7e-15 and
  1.5e-14 — floating-point noise — so `cond` has no significant digits at all.
  Say "singular", or quote the rank. The current notebook hard-codes
  `2.6e+16` in three places against a measured 3.16e+16, which is exactly the
  mistake of treating noise as a measurement.

---

## Cell 18 — the dependency you engineered yourself

**Prompt to type:**
> largest absolute value of SibSp + Parch + 1 - FamilySize over the training rows

**Expect:** `max |SibSp + Parch + 1 - FamilySize| = 0`.
**Assert:** `resid == 0.0`, and `5 + 1 == n_cols - rank`.
**Annotate:** short

> **This is the reveal.** Now — not before — go back to the line in Cell 7 and
> quote it:
>
> ```python
> d["FamilySize"] = d["SibSp"] + d["Parch"] + 1
> ```
>
> It looked like good feature engineering. It is a weighted sum of existing
> features with weights 1, 1 and a constant, sitting in `NUM` alongside its own
> parts, and it manufactured the singularity by hand. Exactly zero, not 0.98 —
> an identity, not a correlation. "Highly correlated" would be the wrong word
> and would not tell you the minimiser is non-unique.

---

## Cell 19 — prove the coefficients are not the data

**Prompt to type:**
> take the fitted weights, add 2.5 to both Sex_female and Sex_male and subtract 2.5 from the intercept, and show me the largest difference in the logit across the training rows

**Expect:** `largest difference in the logit over all 712 rows: ~1e-15`.
**Assert:** both of `gap < 1e-9` **and**
`not np.allclose(theta, theta + shift * v)`. Either assert alone proves
nothing.
**Annotate:** short

> Name the vector `shift_dir`, not `v`. In the current notebook `v` is already
> a float in one loop and a pandas row in another, which is the §4.1 defect the
> course warns about, in the cell that carries its most important argument.

---

## Cell 20 — the repair, two lines

**Prompt to type:**
> drop FamilySize from the numeric list and set drop="first" on the encoder, then give me columns, rank and condition number again

**Expect:** `columns: 23   rank: 23   cond: 83.6`.
**Assert:** `cols2 == rank2 == 23`
**Annotate:** short

> Change exactly two things, so that the rank moving is attributable to them.
> Half the repair is not a repair: dropping `FamilySize` alone leaves five
> dependencies, `drop="first"` alone leaves one. The condition number goes from
> singular to 83.6 — quote *that* number, it is the one that means something.

---

## Cell 21 — and it scored better, which was not the reason

**Prompt to type:**
> cross-validate the repaired pipeline on log loss with the same cv, print the held-out score against the anchor, and show me the six largest contrasts

**Expect:** `held-out log loss, repaired model: 0.4681` against
`anchor: 0.6657`. Then the contrasts, largest first: `Title_Miss −4.867`,
`Deck_infrequent_sklearn −4.808`, `Sex_male −4.746`, `Title_Mrs −4.077`,
`Title_Mr −2.950`, `Title_Rare −2.110`. Reference levels dropped:
Pclass 1, female, Embarked C, Title Master, Deck A.
**Assert:** `ll_v2 < constant_log_loss` and `"Sex_female" not in names_v2`.
**Annotate:** full

- **Left open:** whether "it scored better" is a result at all. The prompt asks
  for a mean and gets one; it does not ask whether the mean moved by more than
  the folds disagree.
- **The usual student version:** reporting 0.496 → 0.468 as the justification
  for the repair. I scored both models on the **same ten folds of the same 712
  training rows**, `StratifiedKFold(10, shuffle=True, random_state=42)`. Nine
  of the ten folds are identical to within 0.013; the entire 0.028 improvement
  is **one fold**, which goes 1.134 → 0.843. Drop that fold and the repaired
  model is 0.001 *worse*. The medians are 0.4263 and 0.4257 — a tie. The
  fold-to-fold spread is 0.217 and 0.133, five to eight times the effect.
- **How you would catch it:** print the per-fold vector for both models and
  subtract elementwise. `[0.001, 0.000, 0.291, 0.001, -0.001, 0.001, -0.013,
  0.001, 0.001, -0.000]` is not a distribution shifting; it is one fold moving.
  Then say the true thing: the repair was for **identifiability**, and had the
  score gone slightly down it would still have been the right change. That
  argument does not depend on the effect size, which is why it survives.

---

## Cell 22 — is the probability a probability

**Prompt to type:**
> out-of-fold predicted probabilities for all 712 training rows, then bin them into ten bins and show predicted against observed with the count in each bin, plus the expected calibration error

**Expect:** ten rows; the extremes are well behaved
(`[0.9,1.0) predicted 0.956 observed 0.961 n=76`) and the middle is not
(`[0.6,0.7) predicted 0.643 observed 0.488 n=41`). `ECE 0.050`.
**Assert:** `p_oof.shape == (712,)` and `((p_oof >= 0) & (p_oof <= 1)).all()`
**⏱** 0.2 s. The current notebook says 15 s; delete that marker.
**Annotate:** short

> Print `n` beside each bin, always. `[0.2,0.3)` has n=38 and `[0.3,0.4)` has
> n=28 — the two worst-looking bins hold 66 passengers between them. Out-of-fold
> is not optional here: `predict_proba(X_train)` on a model fitted to X_train
> measures calibration on rows the model has already seen, and calibration is
> the requirement this notebook was given.

---

## Cell 23 — now, and only now, the cut-off

**Prompt to type:**
> using those out-of-fold probabilities, tabulate cost and number of escorts across the grid, and show me 0.5 beside the cost-optimal cut-off

**Expect:**

```
cut-off 0.50   cost   708.0   escorts 443
cut-off 0.83   cost   210.0   escorts 605
saved by choosing the cut-off deliberately: 498
```

**Assert:** `best["cost"] <= half["cost"]` and
`best["flagged"] >= half["flagged"]` — a 10:1 ratio must buy *more* escorts,
not fewer; if it buys fewer, the sign is inverted.
**Annotate:** short

> **Do not comment on 605.** Let the reader read the number themselves. Cell 25
> is what happens when someone finally does.

---

## Cell 24 — the cost curve

**Prompt to type:**
> plot cost against cut-off, mark the chosen point and 0.5

**Expect:** one axis, a broad minimum around 0.83, a dashed line there and a
dotted line at 0.5.
**Assert:** none.
**Annotate:** short

> Mark both. A curve with only its minimum circled hides how far the default
> sits from it, and it is an illustration rather than an argument.

---

## Cell 25 — the policy you can actually staff

**Prompt to type:**
> the safety unit has 80 crew. escort the 80 passengers least likely to survive and give me the cost and composition beside the unconstrained optimum

**Expect:**

```
                        escorts     cost  reached
cost-optimal 0.83           605      210    99.1%
the 80 you have              80     3678    16.4%
439 of 712 passengers did not survive.
the 80 escorts reach 72 of them; 8 go to people who would have lived anyway
implied cut-off: 0.062  (a consequence of the ranking, not a choice)
```

**Assert:** `int(take.sum()) == CREW` exactly, and `unc["flagged"] > CREW` —
the constraint must be *shown* to bind, not assumed to.
**Annotate:** full

- **Left open:** that the rule has changed kind. Nothing in the prompt says
  "ranking", and a threshold and a ranking are different objects: you are no
  longer asking *is this passenger below 0.83*, you are asking *are they in the
  worst 80*. The implied cut-off, 0.062, is whatever the 80th smallest
  probability happened to be. Change the passenger mix and it moves on its own
  with the model untouched.
- **The usual student version:** applying the 0.83 cut-off and truncating the
  list at 80. On this data it returns the same 80 people, and it hides that a
  constraint bound at all. The real library default underneath: `np.argsort`
  defaults to `kind="quicksort"`, which is **not stable**, so the boundary
  passenger is whichever one the sort happened to place first. I checked: this
  `p_oof` has 15 repeated probability values, but the 80th and 81st differ by
  1.1e-4, so `kind="stable"` and the default select the same 80 here. It is
  free insurance, not a fix for an observed bug — say that rather than implying
  the default breaks.
- **How you would catch it:** assert the escort count equals `CREW` exactly. A
  rule that produces 74 escorts because that is where a grid point fell is a
  threshold rule that got lucky, and it wastes six crew. Then check the
  ranking against the best rule you can actually staff: the cheapest *feasible*
  cut-off on the grid costs **3,727**, and the ranking costs **3,678**, so the
  ranking is better than every threshold you could have afforded.

> **99.1% becomes 16.4%.** Same model, same probabilities, same costs, same
> 712 out-of-fold rows — the only thing added was a constraint that was in the
> brief. Report **3,678**, say that 80 crew is what makes it 3,678, and put
> 210 beside it as the answer to *what would more crew buy*. Only **5 of the
> 97** cut-offs on the sweep can be staffed at all (5.2%), so the slide to show
> is the cost curve with the infeasible region greyed out.
>
> And note what capacity does to Cell 22: under a hard cap, any strictly
> increasing relabelling of the probabilities picks the same 80 people at the
> same cost. Calibration is what you need to choose a cut-off from costs;
> **ordering** is what you need to spend a fixed budget.

---

## Cell 26 — push it until it breaks

**Prompt to type:**
> bolt a PolynomialFeatures expansion onto the numeric block only, degrees 1 to 6, and give me a table of columns, training log loss, held-out log loss and held-out accuracy at each degree

**Expect:**

```
deg  cols    train  held-out  accuracy
  1    22    0.395     0.468     81.5%
  2    32    0.381     0.467     82.2%
  3    52    0.355     0.538     79.6%
  4    87    0.310     1.390     77.4%
  5   143    0.286     1.922     76.0%
  6   227    0.284     1.836     76.7%
```

**Assert:** `sweep[1]["cols"] == 22 and sweep[5]["cols"] == 143`;
`sweep[6]["train"] < sweep[1]["train"]`; `sweep[5]["valid"] > sweep[2]["valid"]`.
The last two together *are* the experiment.
**⏱** 7.2 s unconstrained and 7.3 s at `n_jobs=2` on an M4 — the one cell here
that might plausibly approach 20 s on a free Colab core. The current notebook
says 30 s, which is the only one of its four timing claims in the right
neighbourhood.
**Annotate:** short

> Three things to state in the prompt or check afterwards, all real defaults:
>
> - `PolynomialFeatures` defaults to `include_bias=True`. I ran degree 2 both
>   ways: `include_bias=False` gives 33 columns of rank 33 and condition
>   number 253; the default gives 34 columns of rank **33** and condition
>   number 3.7e+15. The default hands back the exact rank deficiency section
>   10 just removed.
> - `return_train_score=False` is the default, and without the training curve
>   this cell measures nothing — one curve cannot show a gap.
> - Expand the **numeric** block only. Squaring a one-hot indicator returns the
>   indicator, and you have just spent twenty minutes deleting exact copies.
>
> 143 columns on 712 rows at degree 5. Count columns against rows before
> interpreting anything.

---

## Cell 27 — both curves, side by side

**Prompt to type:**
> plot training and held-out log loss on the same axis against degree, and held-out accuracy on a second panel

**Expect:** the training curve falling throughout; the held-out curve flat from
1 to 2 and then climbing steeply.
**Assert:** none.
**Annotate:** short

> Say the tie out loud: degrees 1 and 2 differ by **0.0010** in held-out log
> loss, and the fold-to-fold spread at those degrees is 0.133 and 0.121.
> **They are tied**, and naming degree 2 the winner is reporting noise.
>
> Then which metric noticed first. From degree 2 to degree 5, held-out accuracy
> falls **6.2 points** (82.2% → 76.0%) while held-out log loss is multiplied by
> **4.1** (0.467 → 1.922). Accuracy only moves when a prediction crosses the
> cut-off; log loss sees a confident prediction become a confidently wrong one.
> The requirement was probabilities, so the metric that matches the requirement
> is the one that saw it. (The current notebook says "about seven points". It
> is 6.2.)

---

## Cell 28 — the warning worth reading

**Prompt to type:**
> fit each degree once with warnings captured rather than silenced, and tell me whether it converged, how many iterations it took and the largest coefficient

**Expect:**

```
degree 1: converged True    iterations    74   largest |theta|   4.87
degree 2: converged True    iterations   256   largest |theta|   5.43
degree 3: converged True    iterations   935   largest |theta|   6.32
degree 4: converged False   iterations  4000   largest |theta|  18.70
degree 5: converged False   iterations  4000   largest |theta|  11.07
degree 6: converged False   iterations  4000   largest |theta|   3.98
```

**Assert:** `conv[1]["converged"]` and `not conv[4]["converged"]`.
**⏱** 1.2 s unconstrained, 2.2 s at `n_jobs=2`.
**Annotate:** full

- **Left open:** which knob the reader will reach for. The reflex is
  `max_iter`, and it will not help — understanding why is the next lecture.
- **The usual student version:** leaving `max_iter` at scikit-learn's default
  of **100**. Read the iteration column: degree 2 needs 256 and degree 3 needs
  935, so at the default every degree above 1 warns, and "degree 4 is where it
  stops arriving" disappears into a wall of identical warnings. The second real
  default, and this notebook does it to itself seven times: `ConvergenceWarning`
  subclasses `UserWarning`, so the `warnings.simplefilter("ignore")` used in
  Cells 13, 14, 15, 20, 21, 22 and 26 hides the model telling you the minimiser
  does not exist. That is why this cell uses `record=True` with
  `simplefilter("always")`.
- **How you would catch it:** two failures today with the same shape — rank 23
  of 29 means the minimiser is not **unique**; `lbfgs` not converging at degree
  4 means it does not **exist**. Both are statements about the optimisation
  problem, not about the passengers, and both are repaired by the same one-line
  change you have not seen yet. Do **not** tell the reader the largest-θ column
  is the hint: it reads 4.87, 5.43, 6.32, 18.70, 11.07, **3.98** — degree 6 has
  the smallest maximum coefficient in the table, smaller than degree 1. A
  reader who follows that hint finds it contradicting the story.

---

## Cell 29 — the numbers to bring back

**Prompt to type:**
> summarise: best held-out log loss and the degree it came from, held-out accuracy there, the anchor, and the cut-off chosen from the 10:1 costs

**Expect:**

```
held-out log loss, best degree (2)   0.467
held-out accuracy at that degree     82.2%
anchor — report the base rate        0.666
chosen cut-off, from the 10:1 costs  0.83
```

**Assert:** `best_d in (1, 2)` — the two simplest models are tied at the top,
and an assert that accepted only one of them would be asserting noise.
**Annotate:** short

> Add the two numbers section 12b earned and this summary currently omits:
> **3,678** with 80 crew, and 16.4% of at-risk passengers reached. The log loss
> is what the model earned; those two are what the report says.
>
> Fix nothing. The sweep stays at its worst point, the convergence warnings stay
> visible, the model stays unregularised. Lecture 6 has nothing to bite on
> otherwise, and the shape of the failure is the material.

---

## Cell-number mapping

| Script | Notebook cell index | Script | Notebook cell index |
|---|---|---|---|
| 1 | 3 | 16 | 49 |
| 2 | 5 | 17 | 52 |
| 3 | 8 | 18 | 55 |
| 4 | 10 | 19 | 58 |
| 5 | 13 | 20 | 62 |
| 6 | 16 | 21 | 64 |
| 7 | 19 | 22 | 67 |
| 8 | 23 | 23 | 70 |
| 9 | 26 | 24 | 72 |
| 10 | 30 | 25 | 76 |
| 11 | 33 | 26 | 80 |
| 12 | 36 | 27 | 82 |
| 13 | 39 | 28 | 85 |
| 14 | 43 | 29 | 88 |
| 15 | 46 | | |

---

## Defects found in the current notebook

Everything below refers to `notebooks/lecture-05.ipynb` as it stands (89 cells,
29 code cells, 30 prompt boxes). Cell numbers are the notebook's own indices,
0-based, as loaded by `json`. **Every item marked ✅ was executed with
`python3` against `notebooks/datasets/titanic/train.csv` on scikit-learn 1.7.2
before it was written down.** Items marked ⚠ are reasoned, not executed, and
say so.

### 1. `Deck_T` has no coefficient in this model ✅

The worst factual defect in the notebook, and it drives three cells.

Cell 47 states: *"`Deck_T` has the largest weight in the model."* Cell 45's
annotation repeats it. Cell 18's annotation says `Deck_T` is *"precisely that
failure, surviving the collapse"*.

`Deck_T` is not a column. `OneHotEncoder(min_frequency=2)` in cell 30 merges
every level with fewer than two training rows into a single column named
`<col>_infrequent_sklearn`. Verified:

```
enc.infrequent_categories_ for Deck : ['T']
"Deck_T" in get_feature_names_out() : False
largest |coefficient|               : Deck_infrequent_sklearn  -5.678
```

The pedagogical point survives — that bucket contains exactly one passenger,
who did not survive, and it does carry the largest weight — but the name the
reader is told to look for never appears in cell 46's output, and nothing in
the notebook connects the printed name to the deck-T count asserted in cell 49.
It "survives the collapse" is the opposite of what happens.

### 2. The condition number is quoted three times and is wrong all three ✅

Cells 51 and 61 (markdown) and cell 62 (a literal string inside a `print`) all
state **2.6e+16**. Measured: **3.159e+16**. Cell 52 prints the computed value,
so on a real run the notebook's stored output and its own prose disagree by 20%
— §1.5, in the section about numerical trustworthiness.

Worse, the quantity has no significant digits. The six smallest singular values
of the 29-column design matrix are 1.46e-14, 1.01e-14, 6.52e-15, 3.78e-15,
3.18e-15, 1.72e-15 — floating-point zero. `cond` is a ratio against noise and
will differ between BLAS builds. Hard-coding it in a `print` is the error the
guidelines exist to prevent, committed inside the cell that teaches it.

### 3. A prompt box is duplicated verbatim ✅

Cells **74 and 75** are both the prompt box for *"the policy you can actually
staff"*, for the single code cell 76. The source explains it:
`tools/notebooks/lecture_05.py` calls `prompt(label="the policy you can
actually staff", …)` twice, at lines 946–969 and 970–978, with the last two
bullets very slightly reworded. The reader meets the same box, the same
`input`, the same `check`, twice in a row before one code cell. Deleting either
call fixes it. This also means the notebook's box count is **30 for 29 code
cells**.

### 4. Timings overstated by 10× to 100× ✅

Measured on a 16-core M4 (`Mac16,6`), scikit-learn 1.7.2, twice — once
unconstrained, once with `OMP_NUM_THREADS=1` and `n_jobs=2`:

| Marker | Claim | Unconstrained | 2 cores |
|---|---|---|---|
| cell 37 / 38 | "about two minutes" | 7.2 s | 2.2 s |
| cell 41 / 42 | "about 20 seconds" | 0.3 s | 0.1 s |
| cell 65 / 66 | "about 15 seconds" | 0.2 s | 0.1 s |
| cell 78 / 79 | "about 30 seconds" | 7.2 s | 7.3 s |

The whole notebook computes in **16.5 s**. Cell 85 (the convergence table, 1.2
s) carries no marker while three cells that finish in under a second carry one.
A free Colab core is slower than an M4 core, but not by the factor of 16 the
"two minutes" claim would require. §7.1 says mark cells over ~20 s; on this
evidence the honest answer is that **no cell in this notebook needs a ⏱**, and
the header's *"anything that takes more than a few seconds says so"* is
currently false in both directions. I did not measure on Colab itself.

### 5. §2.4 — the repair's improvement is one fold out of ten ✅

Cell 64 prints `held-out log loss, repaired model: 0.4681` under the comment
*"And it scored better, which was not the reason for doing it."* Scored on the
**same ten folds of the same 712 training rows** (`StratifiedKFold(10,
shuffle=True, random_state=42)`, identical `cv` object, so §2.1 is satisfied —
the rows match):

```
v1 folds  [0.422 0.430 1.134 0.501 0.337 0.390 0.457 0.471 0.402 0.417]
v2 folds  [0.421 0.430 0.843 0.500 0.338 0.389 0.470 0.471 0.402 0.417]
per-fold  [0.001 0.000 0.291 0.001 -0.001 0.001 -0.013 0.001 0.001 -0.000]
```

Nine folds are identical to within 0.013. The whole 0.028 mean improvement is
fold 3. Delete fold 3 and the repaired model is **0.0011 worse**; the medians
are 0.4263 and 0.4257, a tie; the fold spreads are 0.217 and 0.133, five to
eight times the effect. The notebook never states the spread beside the claim.
Its own defence — that the repair was for identifiability and would have been
right even had the score fallen — is the correct argument (§2.2) and is present
in cell 63, but the headline sentence still asserts an improvement that is
inside the noise.

### 6. §8.1 — the `FamilySize` trap is announced four times before it fires ✅

By enumeration of the cells between the feature and the reveal:

1. cell 18, `left_open`: *"one of these four lines manufactures an exact linear
   dependence and takes twenty minutes to diagnose in section 9"*
2. cell 20, a ⚠ blockquote quoting the exact line and saying to remember it
3. cell 22, `left_open`: *"FamilySize, SibSp and Parch are all in NUM together.
   That is the trap, sitting in plain sight in the first line"*
4. cell 29, `left_open`: *"some of those 28 are redundant by construction"*

then cells 50–55 spring it. This is the defect §8.1 documents in lecture 19,
here in a stronger form: the trap is not merely flagged, it is *solved* in
cell 20 and again in cell 22.

Cell 18 also contradicts cell 20 directly. It says *"The markdown under this
cell says so and does not say which."* The markdown under it is cell 20, which
says exactly which, in a fenced `python` block.

### 7. §3.3 — cross-references that do not resolve ✅

- **cell 12**: *"the anchor computed three cells down is built out of this
  one."* The anchor (`constant_log_loss`) is computed in **cell 26**, fourteen
  cells later and thirteen code cells later. It is also not built out of this
  one: cell 13 computes `full["Survived"].mean()` = 0.38384, the anchor uses
  `y_train.mean()` = 0.38343.
- **cells 73 and 77**: *"Requirement 3 has been on the board since the first
  fifteen minutes. **It got its own row in the table**: a fixed number of
  crew."* There is no table of requirements anywhere in the notebook, and the
  strings "Requirement 1" and "requirement 3" appear (cells 24, 65, 73, 76, 77)
  without any cell ever enumerating them. A student reading alone cannot
  resolve either the numbering or the table; both live in the slide deck.
- **cell 77**: *"Change the passenger mix and it moves on its own, with the
  model untouched. **Lecture 8 is about exactly that.**"* Lecture 8 is
  *"Retrain it and watch it change"*, Géron chapters 5 and 6 — Gini and
  entropy, bagging, random forests, impurity importance. Word-searched:
  `drift` 0, `distribution shift` 0, `passenger mix` 0, `population` 0. The
  variance-under-resampling material is adjacent but it is not what the
  sentence promises.
- **cell 51**: *"The next two cells answer it."* The five one-hot dependencies
  are answered in the next cell (53); the sixth is proved in cell 55, three
  cells on, with a prompt box (54) in between. Minor, but it is countable and
  it is off.

### 8. Numbers in prose that do not reconcile ✅

- **cells 81 and 83**: *"held-out accuracy falls by about seven points"* from
  degree 2 to degree 5. Measured: 82.17% → 75.99% = **6.18 points**. Round-half-
  up gives 6, not 7.
- The log-loss half of the same sentence checks out: 0.4671 → 1.9217 is ×4.11,
  *"multiplied by more than four"* ✅.
- **cell 35** `student`: *"writing `flag = prob > t`, getting a **monotone**
  cost curve, and choosing the endpoint."* I ran the inverted rule across the
  same 97-point grid: the curve is **not** monotone (it rises from 371 at
  t=0.02 to 4401 at t=0.98 but not step-by-step), though its minimum is indeed
  the first grid point. The diagnosis is right; the tell the reader is told to
  look for is wrong, and a student checking monotonicity would conclude their
  inverted curve was fine.
- Everything else I could check reconciles: 342/891 and 0.384 ✅; Cabin 77.1%,
  Age 19.9%, Embarked 2 ✅; 712/179 ✅; anchor 0.666 and 0.617 ✅; 11 raw → 28
  model columns ✅; 29 columns rank 23 ✅; 23 = 23 after the repair, cond 83.6
  "under a hundred" ✅; accuracy gap 8.35 pts ≈ "about eight" ✅; cost ratio
  157.0/55.4 = 2.83 ≈ "nearly three" ✅; 20 of 20 seeds ✅; 605 escorts, cost
  210, 605−80 = 525 ✅; 99.1% and 16.4% ✅; 3,678 ✅; implied cut-off 0.0624 ≈
  0.062 ✅; 5 feasible cut-offs of 97 = 5.15% ≈ "5.2%" ✅; 22 and 143 columns
  ✅; degrees 1 and 2 differ by 0.0010 ✅.

### 9. The largest-coefficient hint contradicts its own table ✅

Cell 84's `left_open`: *"why raising `max_iter` will not help. That is the next
lecture, and the largest-coefficient column is the hint."* The column cell 85
prints is:

```
degree 1  4.87    degree 3  6.32    degree 5  11.07
degree 2  5.43    degree 4 18.70    degree 6   3.98
```

It is not monotone, and degree 6 — the most over-parameterised model in the
notebook, 227 columns on 712 rows — has the **smallest** maximum coefficient of
the six, below degree 1's. A reader who follows the hint finds evidence against
the story it is pointing at.

### 10. §4.1 — one name, three types ✅

`v` is bound in cell 16 as a float (`for k, v in by_sex.items()`), in cell 19 as
a pandas Series (`for k, v in full.groupby(...).iterrows()`), and in cell 58 as
an `ndarray` (`v = np.zeros_like(theta)`) — the dependency direction in the
identifiability proof, the most load-bearing object in section 9, wearing a name
two loops have already used for something else. This is the exact defect §4.1
cites from lecture 19. Also rebound across cells, less seriously: `Z` (30, 52),
`n` (10, 46, 64), `k` (16, 19), `c` (46, 64), `fig` (72, 82), `d` (80, 85 — and
`d` is the DataFrame parameter of `engineer` in cell 19).

Separately, cell 80 rebinds **`r`**, the `cross_validate` result dict from cell
43, inside its degree loop. Nothing downstream reads it, so nothing breaks, but
after a full run `r` no longer refers to the section-8 experiment.

### 11. An unnamed out-of-order hazard: `prep_v1` is mutated in place ✅

`prep_v1` is a single `ColumnTransformer` object shared by every pipeline built
from it. Verified: `m_v1.named_steps["prep"] is prep_v1` → `True`. It is fitted
in cell 30, refitted by `model_weak.fit` in cell 33, refitted 20 more times
inside cell 39's `m.fit(A, ya)` (on other splits, other imputation medians), and
refitted again in cell 46.

In a straight top-to-bottom run this costs nothing. But a reader who re-runs
cell 39 after cell 46 — an obvious thing to try, it is the cell with the seed
loop — leaves `m_v1` holding coefficients fitted to `X_train` on top of a
preprocessing block fitted to seed 19's split, and cells 52, 55 and 58 then
describe a design matrix that does not correspond to the coefficients they are
about. §4.3 asks for the specific failure to be named. It is not named. The fix
is one word: `clone(prep_v1)` inside `make_model`, or build the ColumnTransformer
in a function.

### 12. §8.3 — nothing is marked examinable ✅

The string "examinable" occurs three times in the whole notebook: cells 2, 3 and
5, all in the setup, all saying *not* examinable. Sections 2 to 14 carry no
marking of any kind. This is the same count lecture 19 was faulted for.

### 13. §6.1 — 30 boxes, 30 full annotations ✅

Counted: 29 code cells, 30 prompt boxes (see defect 3), and every one carries
all three of *Left open* / *The usual student version* / *How you would catch
it*. The guideline is five to eight full annotations, never more than ten.
This notebook is the worst offender in the course.

Several of the *usual student version* bullets are also invention rather than
observation, which §6.2 forbids — cell 74/75's *"gives the WRONG 80 the moment
the cut-off is not monotone in risk"* describes a situation that cannot arise:
a threshold on a probability is monotone in that probability by construction.

### 14. Setup box justified by a function this notebook never calls ✅

Cell 2's `constraint` pins scikit-learn ≥ 1.4 because *"`root_mean_squared_error`
arrived in 1.4"*. Searched: `root_mean_squared_error` appears in no code cell of
lecture 5 — it is boilerplate carried over from the regression lectures. The
notebook does have a real version dependency (`OneHotEncoder(min_frequency=…,
handle_unknown="infrequent_if_exist")`, 1.1+), and cell 30 would fail without
it, which is a better reason and is not the one given.

### 15. §5.1 — one indented line, flagged by the machine check, benign on the page ⚠

Cell 24 line 13 is indented six spaces:

```
      + (1-y^{(i)})\log(1-\hat{p}^{(i)})\Big]$$
```

It is the continuation of the display-math block opened on line 12. The §9
checker (*"no markdown line indented ≥4 outside a fence"*) flags it, because
`$$` is not a fence. It should render correctly regardless — CommonMark does not
let an indented code block interrupt a paragraph, so this is a lazy
continuation, not a code block. **I checked the source, not the rendering.** I
did not open the notebook in Colab. It is the only line ≥4 spaces outside a
fence in the whole file; every other markdown cell is clean.

### 16. §1.2 cannot be satisfied — the notebook stores no outputs ✅

All 29 code cells have `outputs: []` and `execution_count: null`. Nothing in
this file's prose can be reconciled against a stored output, by construction,
which is why every figure in this report was re-derived by running the pipeline
rather than read off the notebook. This appears to be a course-wide convention
rather than a lecture-5 defect, but it does mean the §1.2 machine check
(*"every ≥4-digit prose figure appears in a stored output"*) has nothing to
check against, and the ⏱ check (*"any cell whose stored execution exceeded 20 s"*)
can never fire — which is presumably how defect 4 survived.

### What I checked and found clean

- **§2.1, matched rows.** Every comparison in the notebook is on matched rows,
  and I verified each: section 7's 0.5-vs-chosen comparison scores both cut-offs
  on the same held-out `B` per seed; sections 8 and 10 use the identical `cv`
  object on the same 712 training rows; sections 12 and 12b both score the same
  712 out-of-fold probabilities. This is the defect that sank lecture 19 and
  lecture 5 does not have it. What the prose omits (defect 5) is the *spread*,
  not the alignment.
- **§3.1, quoted code.** The notebook contains exactly one ```` ```python ````
  block in markdown (cell 20). Its line appears verbatim in code cell 19.
- **§3.2, checks offered to the reader.** Every assertion in every code cell
  executes and passes on scikit-learn 1.7.2 — I ran all 29 cells' logic.
- **§4.2, idempotency.** `full = engineer(full)` (cell 19) is idempotent;
  re-running it is safe. No cell continues training an existing fit. The only
  state hazard is defect 11.
- **§7.2, re-run order.** Vacuous: this notebook sets no exercises requiring
  re-runs. The COMMIT box (cell 27) asks for four numbers on paper, which a
  student alone at home can do.
- **§7.4, national calendar.** No holidays or local knowledge assumed.

### What I could not check

- Whether cell 24's LaTeX and cell 20's blockquoted fence render correctly **in
  Colab specifically**. I reasoned from CommonMark; §10.8 asks for the rendered
  page and I did not open one.
- The Colab wall-clock figures. All timings are from a 16-core M4, constrained
  to two cores and one BLAS thread as a proxy. The direction of defect 4 is not
  in doubt — a 16× error cannot be a hardware difference — but the exact Colab
  numbers are unmeasured.
- The download path in cell 8. `datasets/titanic/` was already present, so the
  `urlretrieve` branch never executed and the "~5 s the first time" claim is
  unverified.
