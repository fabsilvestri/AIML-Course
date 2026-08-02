# Lecture 7 — *A model the regulator will accept*, rebuilt by prompting

The script to follow at a Colab keyboard. Twenty-one code cells, in order.

**Application.** CoverType, 581,012 patches of Colorado forest, seven species.
The regulator will not accept a prediction without a human-readable
justification, negotiated into **at most eight conditions per prediction**. That
constraint picks the model before anyone looks at the data.

**The assistant failure this lecture carries.** Ask for "an interpretable
decision tree so I can explain each prediction" and you get
`feature_importances_` — a vector with one entry per **column**, not one per
**prediction**. It is the same 54 numbers for every patch in Colorado. The
regulator asked why *this* parcel was refused; the answer on offer is "elevation
matters a lot, in general".

**Provenance.** These are specifications, not transcripts. Nobody is claiming
this exact string was typed and this exact cell came back.

**Everything below was measured**, not transcribed, on:

| | |
|---|---|
| machine | Apple M4 Max, 16 cores |
| scikit-learn / numpy / pandas | 1.7.2 / 2.3.5 / 2.3.3 |
| seed | `RANDOM_STATE = 42` everywhere |
| end-to-end, warm cache | **16 s** for all 21 cells |
| end-to-end, cold cache | **~6 min 45 s** — the download cell is 391 s of it |

Where a figure would differ on a 2-core Colab runtime, both are given and the
extrapolated one is labelled as extrapolated.

**Annotation budget: 8 of 21 cells get the full three bullets** (cells 2, 4, 9,
10, 14, 17, 19, 21). Every other cell gets the one-line **Box** and nothing
else. That is the point of the budget — three readers of lecture 19 stopped
reading the three-bullet template around cell 30, which is where that notebook
keeps its defect.

---

## Cell 1 — setup and the seed

**Prompt to type:**

> Setup cell. Import sys, sklearn, numpy as np, pandas as pd, matplotlib. Print
> the python, scikit-learn, numpy and pandas versions. Assert scikit-learn is at
> least 1.4 with a message telling me to `%pip install -U scikit-learn`. Set
> `RANDOM_STATE = 42` and `pd.set_option("display.width", 100)`.

**Expect:** four version lines, nothing else. `RANDOM_STATE` bound.
**Assert:** `tuple(int(p) for p in sklearn.__version__.split(".")[:2]) >= (1, 4)`.
**Box:** *input* nothing · *output* every library version this notebook depends
on, and one seed · *constraint* assert the scikit-learn version, do not print it
· *check* the assert fires on an old Colab image, twenty cells before the error
would otherwise appear.
**Annotate:** short

Mark the section **not examinable — engineering**.

---

## Cell 2 — the data

**Prompt to type:**

> Load the covertype dataset from scikit-learn as a dataframe. Print how many
> rows and columns and how many classes. Also make a list `COVER_NAMES` of the
> seven species names in label order: Spruce/Fir, Lodgepole Pine, Ponderosa
> Pine, Cottonwood/Willow, Aspen, Douglas-fir, Krummholz.

**Expect:**

```
581,012 patches, 54 columns, 7 cover types
```

**Assert:** `X_all.shape == (581012, 54)` **and**
`sorted(y_all.unique()) == [1, 2, 3, 4, 5, 6, 7]`. The second one is the one
that matters. Add it even though the prompt did not ask for it.

**⏱ First run: ~6 min 30 s. Measured 391 s** end to end on the M4 Max, of which
**1.0 s** was the 11.2 MB download from figshare and the remaining ~390 s was
scikit-learn decompressing and parsing the CSV — **single-threaded**, so a
2-core Colab runtime will not be faster and probably slower. It caches to 14.3
MB under `~/scikit_learn_data/covertype`; **every later run is 1.6 s**.

Say this in the markdown above the cell before anyone runs it. Six minutes of no
output reads as "it hung", and the cell after it is the one people interrupt.

**Annotate:** full

* **Left open:** the indexing convention. `fetch_covtype` returns labels
  **1 … 7**, not 0 … 6 — it is the odd one out among the scikit-learn loaders,
  and `COVER_NAMES[k - 1]` then appears in five later cells (the label counts,
  the anchor, `plot_tree`, `export_text`, the trace). Each one is a chance to
  drop the minus one.
* **The usual student version:** `COVER_NAMES[k]`. It never raises — the list
  has seven entries and the labels reach 7 only for Krummholz, which is 3.5% of
  the data, so `IndexError` waits until a Krummholz row turns up and everything
  before that is silently shifted by one species. In the anchor cell the same
  slip is worse: `DummyClassifier.predict()` returns `2`, and
  `COVER_NAMES[2]` is "Ponderosa Pine" when the answer is "Lodgepole Pine".
* **How you would catch it:** assert the label **domain**, not just the shape.
  `sorted(y_all.unique()) == [1, 2, 3, 4, 5, 6, 7]` is one line and it tells you
  the convention before anything is built on it. The shape assert would have
  passed either way.

---

## Cell 3 — what the columns are

**Prompt to type:**

> Print the names of the first ten columns and the first three rows of the first
> six columns. Don't print all 54.

**Expect:** `['Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology',
'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways',
'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm',
'Horizontal_Distance_To_Fire_Points']`, then a 3 × 6 frame.
**Assert:** none.
**Box:** *input* the frame · *output* the ten quantitative column names and a
few rows · *constraint* show the quantitative ten **separately** — the other 44
are already one-hot indicators (4 wilderness areas, 40 soil types) and printing
all 54 hides that structure.
**Annotate:** short

In the markdown: nothing to impute, nothing to encode. Unusual, and it is why
this application gets to spend its time on the model rather than the frame.

---

## Cell 4 — a stated compromise, and the split

**Prompt to type:**

> Take a stratified 60,000-row subsample of the data, then split that
> 80/20 into train and test, stratified again, seed `RANDOM_STATE`. Delete the
> full frames afterwards. Print the two sizes.

**Expect:**

```
train 48,000   test 12,000
```

**Assert:** `len(X_train) == 48_000 and len(X_test) == 12_000`, plus
`set(X_train.index).isdisjoint(X_test.index)` and
`X_train.isna().sum().sum() == 0`.

**Write into the comment, out loud:** the subsample is a compromise for speed,
not a silent one. A 200-tree ensemble on all **581,012** rows is minutes per
fit; **every number in this notebook is a 60,000-row number.** The full frame is
**251.0 MB** (measured: `cover.data.memory_usage(deep=True).sum()`, all 54
columns `float64`), which is what `del` is for.

**Annotate:** full

* **Left open:** that `stratify` has to be passed **twice**. It is `None` by
  default in `train_test_split`, and one class — Cottonwood/Willow — is 227 of
  48,000 training rows, 0.5%. An unstratified draw can hand a fold almost none
  of it, and the stratified counts are what make the later confusion matrix
  readable.
* **The usual student version:** subsampling silently and then reporting the
  accuracy as the dataset's. The compromise is fine. Leaving it out of the
  prose is what turns a 60,000-row result into a claim about 581,012 rows.
* **How you would catch it:** re-run this cell. **It raises `NameError`** —
  it deletes `X_all` and `y_all`, which it also reads. That is the correct
  trade (251 MB is how a free Colab runtime dies three cells later, blaming the
  wrong cell) but it makes the cell non-idempotent, so say so in the markdown:
  *to change the subsample size, restart and run all from cell 1.* Cell 3 also
  stops working once this cell has run, for the same reason.

---

## Cell 5 — look at the labels

**Prompt to type:**

> Print how many training patches of each species there are, with the species
> name and the percentage, and the ratio of the commonest to the rarest.

**Expect:**

```
Lodgepole Pine       23,405   48.8%
Spruce/Fir           17,501   36.5%
Ponderosa Pine        2,954    6.2%
Krummholz             1,694    3.5%
Douglas-fir           1,435    3.0%
Aspen                   784    1.6%
Cottonwood/Willow       227    0.5%

commonest / rarest ratio: 103x
```

**Assert:** `counts.sum() == len(y_train)`.
**Box:** *input* the training labels · *output* count and share per species ·
*constraint* name the species — `4: 1728` is not something anyone can think
about · *check* the counts sum to 48,000.
**Annotate:** short

Two numbers in this table are load-bearing and neither is announced yet:
**48.8%** is about to become the baseline, and **1.6%** is why the Aspen row of
the confusion matrix collapses in section 15. Ask the reader to look for thirty
seconds before scrolling; do not tell them what to find.

---

## Cell 6 — the anchor

**Prompt to type:**

> Fit a `DummyClassifier` that always predicts the commonest class, score it on
> the test set, and print which species it predicts and the accuracy.

**Expect:**

```
always 'Lodgepole Pine'  ->  48.8%
species it can ever predict: 1 of 7
```

**Assert:** none.
**Box:** *input* the training labels · *output* the accuracy of always
predicting the commonest species · *constraint* a real `DummyClassifier` fitted
and scored **through the same interface**, not the majority share computed by
hand · *check* it prints one species out of seven.
**Annotate:** short

48.8% is a number, not a verdict. Every accuracy below is read against it.

---

## Cell 7 — one tree, no constraints

**Prompt to type:**

> Fit a `DecisionTreeClassifier` with no constraints on the training set. Print
> its depth, its number of leaves and its training accuracy.

**Expect:**

```
depth           33
leaves          5,699
train accuracy  100.0%
```

**Assert:** `free.get_n_leaves() > 1000` — an unconstrained tree on this data is
enormous, and a small one means something was capped by accident.
**Box:** *input* the 48,000 training patches · *output* depth, leaves, training
accuracy · *constraint* look at its **shape** before its score · *check* more
than a thousand leaves.
**Annotate:** short

5,699 leaves for 48,000 patches is **8.4 patches per leaf** — a lookup table
with a nice diagram. Measure it honestly and move past it: overfitting is not
what this lecture is about.

---

## Cell 8 — measure it honestly

**Prompt to type:**

> Cross-validate that tree, 5 stratified folds, shuffled, seed `RANDOM_STATE`.
> Print the mean and the fold range.

**Expect:**

```
cross-validated 82.1%  (folds 81.4% - 82.8%)
```

**Assert:** none.
**Box:** *input* the unconstrained tree and the training rows · *output*
cross-validated accuracy **with the fold range** · *constraint* stratified
folds; report the range, not only the mean · *check* the mean sits inside the
range and the spread is about a point and a half.
**Annotate:** short

⏱ 1.5 s here (measured, and the same at `n_jobs=1` — five fits of 0.4 s each).
Do **not** write "about 30 seconds" above this cell; it is five fits of an
unconstrained tree on 38,400 rows and it is fast.

Note the number: **82.1% cross-validated**, and the same tree scores 82.6% on
the test set in cell 9. Those are two different quantities and the notebook has
to keep them apart.

---

## Cell 9 — "make it interpretable"

Type this one exactly as a person would, with no warning in the markdown above
it, no ⚠ in the label, and no hint in the box. Run it, read the output, and
**write the five feature names down on paper before scrolling.**

**Prompt to type:**

> Train a decision tree on the covertype data and make it interpretable, so I
> can explain each prediction to a regulator.

**Expect:** a ranked list under a confident heading, and an accuracy:

```
Why the model predicts what it predicts:
Elevation                             0.341
Horizontal_Distance_To_Roadways       0.126
Horizontal_Distance_To_Fire_Points    0.121
Horizontal_Distance_To_Hydrology      0.063
Vertical_Distance_To_Hydrology        0.052
dtype: float64

accuracy: 82.6%
```

**Assert:** none — deliberately. The cell that fails the brief passes every
assert you would think to write.
**Box:** *input* "train a decision tree and make it interpretable, so I can
explain each prediction to a regulator" · *output* the five largest feature
importances · *constraint* print it exactly as returned.
**Annotate:** full

* **Left open:** the **shape** of the answer. The prompt says "explain each
  prediction" and never says how many numbers an explanation has. Nothing in it
  is wrong; nothing in it is checkable either.
* **The usual student version:** shipping this. It runs, it imports nothing
  exotic, the heading says *why the model predicts what it predicts*, and the
  accuracy underneath is real — 82.6% against a 48.8% baseline. There is no
  error message anywhere in this cell.
* **How you would catch it:** count the entries. `len(importances)` is **54** —
  one per **column**. `len(X_test)` is **12,000**. An explanation with as many
  entries as there are *features* rather than as many as there are *predictions*
  is a global summary wearing a local word, and it is byte-identical for every
  patch in Colorado.

**Now — and only now — open the next markdown cell with the ⚠.** Reviewer
question 3: *what is the shape here?* The reader has the five names on paper and
a real answer to "would you have caught it?".

---

## Cell 10 — measure the damage

**Prompt to type:**

> The tree can justify a prediction — the path from root to leaf is a list of
> conditions. Write a function that returns how many conditions each test
> prediction used, and print the mean and the max for the unconstrained tree.

**Expect:**

```
unconstrained tree:  mean 17.88   max 33
the brief allows:    8
```

**Assert:** `free_len.max() == free.get_depth()` — two independent routes to the
same number.
**Annotate:** full

* **Left open:** what counts as a condition. The prompt says "how many
  conditions" and the natural implementation counts **nodes**, which is a
  different number.
* **The usual student version:** `tree.decision_path(X).sum(axis=1)` with no
  `- 1`. `decision_path` returns an indicator matrix over **every node
  visited, leaf included**, so the row sums are 18.88 and 34 — depth **plus
  one**, uniformly, on every row. Nothing raises, the mean is plausible, and the
  brief is then measured against the wrong quantity. The other version of this
  is quoting `get_depth()` as the answer: 33 is the **longest** path, and the
  brief constrains **every** prediction, not the worst one.
* **How you would catch it:** the assert. `free_len.max() == free.get_depth()`
  fails immediately at 34 ≠ 33 if the minus one is missing. That is exactly what
  a check is for: one line, knowable answer, before you build on it.

Then write the corrected specification into markdown — the shape of the output,
the check, and the explicit prohibition:

> *"Fit a `DecisionTreeClassifier` with `max_depth=8` on `X_train`. For a given
> test instance, return the list of `(feature, comparison, threshold, value)`
> tuples along its decision path, and the class distribution of the leaf it
> lands in. Assert that the list has at most eight entries. Do not use
> `feature_importances_`: it is one vector for the whole model."*

Three additions: the **shape**, the **check**, and a prohibition on the
plausible wrong answer. The assistant was obedient, not wrong — "interpretable"
has a common meaning in the literature and it used it.

---

## Cell 11 — what does depth buy?

**Prompt to type:**

> Sweep `max_depth` from 1 to 12. For each, cross-validated accuracy on the
> training set and the number of leaves. Put it in a dataframe and print it.

**Expect:** twelve rows. Spot-check three of them against your own run:

| max_depth | cv_accuracy | leaves |
|---|---|---|
| 1 | 0.6326 | 2 |
| 8 | 0.7375 | 206 |
| 12 | 0.7861 | 1076 |

**Assert:** none.
**Box:** *input* depths 1 to 12 · *output* CV accuracy and leaf count at each
depth · *constraint* sweep **past** the depth we are allowed to use — the rows
we cannot pick are what price the constraint · *check* accuracy rises
monotonically and depth 1 gives exactly 2 leaves.

**⏱ ~10 s measured** (9.8 s, M4 Max, and 9.8 s again at `n_jobs=1` — twelve
`cross_val_score` calls of five fits each do not parallelise usefully because
each call is short). **Extrapolated** to a 2-core Colab CPU runtime: **20–40 s**.
Not measured there.

**Annotate:** short

Cross-validated accuracy is **still climbing at depth 12** (0.7861 and rising)
and so is the leaf count. That gap is the measured price of the constraint —
bring it to the regulator as a conversation, not a `GridSearchCV`.
**Cross-validation does not get a vote on `max_depth` here**, because it is not
optimising the thing the agency is buying.

---

## Cell 12 — the price, drawn

**Prompt to type:**

> Two panels side by side from that table: CV accuracy against depth, and leaves
> against depth on a log y axis. Dashed vertical line at 8 on both.

**Expect:** two panels; the right one spans 2 → 1,076 leaves over three decades.
**Assert:** none.
**Box:** *input* the depth table · *output* accuracy vs depth, leaves vs depth
on a log axis · *constraint* log scale on the leaf count and a marked line at
the depth we are actually allowed · *check* both curves are still rising at the
right edge.
**Annotate:** short

Use `semilogy` because leaf count spans three orders of magnitude. Do **not**
write that a linear axis "shows eleven points at zero" — on a linear axis with a
1,076 maximum, depths 9–12 sit at 33%, 50%, 73% and 100% of the range and are
perfectly visible. The honest reason for the log axis is that growth per level
is roughly **geometric**, and a log axis is where a geometric process is a
straight line. It is not a clean doubling either: the ratios run 2.0, 2.0, 2.0,
2.0, 1.88, 1.88, 1.82, 1.70, 1.54, 1.45, 1.38 — the rate decays, which is
itself the finding.

---

## Cell 13 — tune what is left

**Prompt to type:**

> Grid search `max_depth` in [4, 6, 8, None] and `min_samples_leaf` in
> [1, 5, 20, 50, 200, 500] with the same CV. Print the whole grid as a pivot
> table of mean test score, and the best params.

**Expect:**

```
param_min_samples_leaf    1      5      20     50     200    500
param_max_depth
4.0                    0.6935 0.6935 0.6935 0.6933 0.6922 0.6863
6.0                    0.7125 0.7126 0.7118 0.7110 0.7068 0.6959
8.0                    0.7375 0.7372 0.7335 0.7314 0.7152 0.6969
NaN                    0.8206 0.8088 0.7830 0.7566 0.7185 0.6969

best overall: {'max_depth': None, 'min_samples_leaf': 1}
```

**Assert:** none.
**Box:** *input* a 2-D grid of `max_depth` and `min_samples_leaf` · *output* the
**whole** grid as a pivot table, not `best_params_` alone · *constraint* search
depths we may not use as well · *check* the table has 4 × 6 = 24 finite cells
and the `NaN` row index is `max_depth=None`.

**⏱ 2.2 s measured** with `n_jobs=-1` on 16 cores; **26.7 s** at `n_jobs=1`.
`GridSearchCV` parallelises all 120 fits at once, which is why it beats the
depth sweep above despite doing ten times the work. **Extrapolated** to a 2-core
Colab runtime: **15–30 s**. Not measured there.

**Annotate:** short

Read the interaction, and state the ranges:

* Along `max_depth=8`, `min_samples_leaf` moves the score **0.7375 → 0.6969**,
  4.1 points — but almost all of that is the last step to 500. From 1 to 50 it
  is **0.7375 → 0.7314**, 0.6 points. The depth limit binds first.
* Along `max_depth=None` the same sweep is **0.8206 → 0.6969**, 12.4 points.
  With no depth limit, leaf size **is** the regularisation.

Two hyperparameters that both restrict the tree do not act independently. A 2-D
grid shows that; two 1-D sweeps would find the same winner and show none of it.

---

## Cell 14 — the model we ship, overruling the grid

**Prompt to type:**

> Fit the tree we're actually shipping: `max_depth=8`, `min_samples_leaf=20`.
> Print the number of leaves, how many of the 54 columns it consults, the mean
> and max conditions per prediction, and the training accuracy.

**Expect:**

```
leaves                  163
columns consulted       24 of 54
conditions, mean / max  7.97 / 8
train accuracy          74.4%
```

**Assert:** `lens.max() <= 8, "the brief is violated"` — the brief, expressed as
an assert, in the cell that ships the model.

**Annotate:** full

* **Left open:** why 20 and not 1. The grid's answer under the cap **is** 1
  (0.7375 against 0.7335), and the prompt just asserts 20 without a reason. The
  reason is that this model states its justification as *"91% of the 463
  training patches in this leaf"*, and at `min_samples_leaf=1` that sentence can
  become *"100% of the 1"* — a single surveyed patch wearing the grammar of
  evidence. Auditable is a property of the sentence, not of the score.
  It costs **0.40 points** of cross-validated accuracy (0.7375 − 0.7335), and
  that number goes to the agency with everything else.
* **The usual student version:** taking `best_params_` because it is the best —
  and then, worse, **reporting the numbers of the model you did not ship.** This
  is not hypothetical: the current `lecture-07.ipynb` says in §12 that it
  overrules the grid, and then every headline it quotes afterwards belongs to
  the `min_samples_leaf=1` tree — its §16 table reports 73.3% and 7.99 for a
  model whose real figures are 73.0% and 7.97. The full comparison is below this
  list.
* **How you would catch it:** score exactly the object you named. `tree` is
  bound in this cell; every later number must come from `tree` and from
  `X_test`, and the assert `lens.max() <= 8` is the only thing standing between
  the brief and a model that quietly satisfies a different one. When the brief
  constrains the model, the grid does not get a vote — and then the report has
  to be about the model that was constrained.

Both models, measured on the same 12,000 test rows, same seed — the left column
is the one every later cell must report:

| | shipped, `min_samples_leaf=20` | rejected, `min_samples_leaf=1` |
|---|---|---|
| test accuracy | **73.0%** | 73.3% |
| mean conditions | **7.97** | 7.99 |
| leaves | **163** | 206 |
| smallest leaf | **20** | 1 |
| the traced leaf, cell 17 | **91% of 463** | 90% of 481 |
| Aspen recall | **2.6%** | 3.1% |

Afterwards, in markdown: train 74.4% against 73.35% cross-validated (that second
number is the `8.0 / 20` cell of the pivot table in cell 13 — say so) is a gap
of **1.05 points**, against **17.9 points** for the unconstrained tree. The
depth limit did not only shorten the justification; it removed nearly all the
overfitting as a side effect. Which raises a question this lecture does **not**
answer: if the constrained tree barely overfits, why is it still 9.6 points
worse than the unconstrained one? Lecture 8 opens on it.

---

## Cell 15 — draw it

**Prompt to type:**

> Draw the top two levels of that tree with `plot_tree` into a matplotlib axis.
> Shorten the long feature names first. Filled, rounded, no impurity,
> proportions, small font.

**Expect:** a two-level diagram rooted at `Elevation <= 3046.5`.
**Assert:** none.
**Box:** *input* the shipped tree · *output* the top two levels drawn into a
matplotlib axis · *constraint* `max_depth=2` **in the plot call** — all eight
levels at a legible font is metres of paper · *check* the root split is
Elevation, matching cell 16's text version.
**Annotate:** short

Shorten the names before plotting (`Horizontal_Distance_To_` → `HDist_` and so
on): `Horizontal_Distance_To_Fire_Points` renders as a smear at any font that
fits the tree on a page.

On graphviz, one line and no more: `export_graphviz` writes a `.dot` file that
still needs the `dot` **binary**, which is not a Python package. Mark it *(not
examinable: tooling, not machine learning)* and move on. Do not assert what is
or is not installed on a Colab image you have not checked this term.

---

## Cell 16 — the version you can paste into an email

**Prompt to type:**

> Same tree, same two levels, as indented text with `export_text`, using the
> short names and the species names.

**Expect:**

```
|--- Elevation <= 3046
|   |--- Elevation <= 2510
|   |   |--- Wild_0 <= 0
...
```

**Assert:** none.
**Box:** *input* the same tree · *output* the top two levels as indented text ·
*constraint* no plotting library at all · *check* the root threshold agrees with
the diagram in cell 15.
**Annotate:** short

**Careful with `decimals`.** The real thresholds are half-integers — CART puts a
split **midway between two adjacent observed values**, so nothing in the data
ever sits exactly on one. In this tree **71%** of the 162 thresholds end in
`.5`; the root is **3046.5** and the second split is **2510.5**. But
`export_text(..., decimals=0)` prints `3046` and `2510`, so if you write "the
thresholds sit at half-integers" underneath a `decimals=0` output, the prose
contradicts the only evidence on the page. Either pass `decimals=1`, or make the
point about the *unrounded* value and say where the reader can see it
(`tree.tree_.threshold[0]`).

The remaining 29% are whole numbers, for the same reason: the midpoint of two
observed values two units apart is an integer.

---

## Cell 17 — trace one prediction, all the way down

**Prompt to type:**

> For one test patch, walk the tree by hand using `tree_.children_left`,
> `tree_.feature` and `tree_.threshold`, print each condition it satisfied with
> the patch's own value, and then what the leaf says: the predicted species, the
> proportion, and how many training patches are in that leaf. Use row 27.

**Expect:**

```
1. Elevation    =    3,763  >     3,046
2. Elevation    =    3,763  >     3,306
3. Wild_2       =        1  >         0
4. Soil_31      =        0  <=        0
5. Elevation    =    3,763  >     3,360
6. HDist_Fire_Points =    3,020  >       364
7. Shade_Noon   =      200  <=      244
8. Shade_3pm    =      133  <=      196
-> Krummholz  (91% of the 463 training patches in this leaf)
   true class: Krummholz
```

**Assert:** two of them — `len(conditions) <= 8`, and that the traced class
equals `tree.predict(X_test.iloc[[i]])`. A justification that disagrees with the
model it claims to explain is worse than no justification.

**Annotate:** full

* **Left open:** what `tree_.value` contains. On scikit-learn 1.7.2 it holds
  class **proportions** per node, so `dist.max()` is 0.9093 and `:.0%` prints
  **91%**. Older scikit-learn stored raw **counts** in the same array, where
  that identical line prints a four-figure percentage. The setup cell only
  asserts ≥ 1.4, so if you quote this number in prose, quote it from the output
  in front of you.
* **The usual student version:** reading the leaf as *"this patch is 91% likely
  to be Krummholz"*. The honest sentence is *"of the training patches that
  satisfied these eight conditions, 91% were Krummholz"*, and the difference is
  the whole regulatory argument. `predict_proba` returns exactly these leaf
  proportions, so a tree's "probabilities" are piecewise constant and
  **identical for every patch reaching the same leaf** — a fact you can check in
  one line: `tree.predict_proba(X_test.iloc[[27]]).max()` is 0.9093 too.
* **How you would catch it:** the second assert. Walking `children_left` by hand
  and trusting the walk is how a justification quietly drifts from the model —
  and the drift is silent, because both objects always return *some* species.

Then, in markdown: a leaf built on 20 patches and a leaf built on 6,203 produce
the same kind of sentence and deserve very different amounts of trust. Which is
why the count belongs **in** the justification, not in a footnote.

---

## Cell 18 — how much is each leaf built on

**Prompt to type:**

> Push the training set through the tree and print the smallest, median and
> largest leaf size.

**Expect:**

```
leaves 163   smallest 20   median 48   largest 6,203
```

**Assert:** none needed, but the check is free and worth stating: **the smallest
leaf must equal `AUDITABLE_LEAF`**. It does — 20. If it were smaller,
`min_samples_leaf` is not doing what you think it is.
**Box:** *input* the training patches routed through the tree · *output*
smallest, median, largest leaf · *constraint* drop the zero counts —
`np.bincount(tree.apply(...))` returns a slot for every node id and the internal
nodes are all zeros · *check* `len(sizes) == tree.get_n_leaves()` (163) and
`sizes.min() == 20`.
**Annotate:** short

The largest leaf holds 6,203 of the 48,000 training patches — 12.9% of them in
one rule. The median holds 48. Both sentences look the same in a report.

---

## Cell 19 — the test set. Once.

**Prompt to type:**

> Score the tree on the test set, print the accuracy next to the baseline, and
> print a per-class precision/recall report with the species names.

**Expect:**

```
test accuracy 73.0%   (baseline 48.8%)
```

and seven rows. Three to check against your own run:

| | precision | recall | support |
|---|---|---|---|
| Lodgepole Pine | 0.746 | 0.820 | 5851 |
| Aspen | 0.500 | **0.026** | 196 |
| Douglas-fir | 0.447 | 0.128 | 359 |

**Assert:** none. Everything above this cell used training data and
cross-validated folds — if that is not true of your notebook, this number is not
a test score.

**Annotate:** full

* **Left open:** the order of `target_names`. `classification_report` does not
  look at your names; it pairs them positionally with `np.unique(y_true ∪
  y_pred)`, which here is `[1, 2, 3, 4, 5, 6, 7]` in sorted order. `COVER_NAMES`
  happens to be in exactly that order, which is the only reason this works. Get
  the list order wrong and every row is labelled with the wrong species, with no
  warning of any kind.
* **The usual student version:** reading only the headline. 73.0% against a
  48.8% baseline is a real result and it says nothing whatever about **Aspen**,
  where recall is **2.6%** — 5 of 196 test patches found. Note what you can
  *not* say: "the model never predicts it correctly" is false, and precision
  0.500 means it does predict Aspen, ten times, and is right half the time. Read
  the row before you write the sentence.
* **How you would catch it:** `zero_division=0` is in the call and on this run
  **nothing divides by zero** — every class has a defined precision. It is
  insurance against a class the model never predicts at all, and the
  scikit-learn default is `"warn"`, which prints `0.0` plus a `UserWarning`
  rather than a number. Keep it, but do not explain it as though it fired.

---

## Cell 20 — the confusion matrix, row-normalised

**Prompt to type:**

> Confusion matrix for that tree on the test set, normalised by row, with the
> species names on both axes, rotated labels.

**Expect:** 7 × 7. The Aspen row reads
`[0.005, 0.939, 0.026, 0.000, 0.026, 0.005, 0.000]`.
**Assert:** none.
**Box:** *input* the tree and the test patches · *output* a 7 × 7 matrix
normalised by row · *constraint* `normalize="true"` — each row then reads *"of
the patches that really were this species, where did they go?"* · *check* every
row sums to 1.
**Annotate:** short

**Read the Aspen row.** 93.9% of it is in the Lodgepole Pine column. 196 test
patches are Aspen, 191 of them are misclassified, so **the entire Aspen row is
worth at most 1.6 points of the headline** (191 / 12,000 = 1.59%) — invisible in
a number quoted to one decimal place.

Why: Aspen is 1.6% of the training set, so a split that isolates it improves the
weighted Gini by very little and CART never chooses one. The impurity criterion
is a weighted average, and a rare class carries almost no weight.

Leaving it unnormalised is the failure worth naming here: the majority class
dominates every cell and the rare-class rows are literally too small to read.

---

## Cell 21 — what class weighting buys, and costs

**Prompt to type:**

> Same tree with `class_weight="balanced"`. Print the accuracy and the Aspen
> recall, each next to the value from the unweighted tree.

**Expect:**

```
accuracy      59.4%  (was 73.0%)
Aspen recall  80.1%  (was 2.6%)
```

**Assert:** none.

**Annotate:** full

* **Left open:** which model the agency wants. That is not a machine learning
  decision, and this notebook deliberately does not make it.
* **The usual student version:** printing the improved recall on its own. This
  is the failure in the current `lecture-07.ipynb`: its cell prints
  `accuracy 59.4% (was 73.0%)` — the pair — and then prints
  `Aspen recall 80.1%` with **no unweighted value beside it**, in a cell whose
  own specification says *"report BOTH numbers"* and whose own note says
  *"Show the pair"*. Half the cell obeys the spec and half does not, and it is
  the half that carries the argument. You need `rep0` — the same
  `classification_report(..., output_dict=True)` on the **unweighted** tree — to
  produce the second half.
* **How you would catch it:** any change that improves a rare class will cost
  the common ones. 13.7 points of overall accuracy bought 77.5 points of Aspen
  recall, on the same 12,000 rows. Quote either one alone and you have written
  an argument; quote both and you have written a measurement, and whoever owns
  the decision can make it.

---

## The closing table

Every row on the same **12,000 held-out test patches**, all measured on
scikit-learn 1.7.2 with `RANDOM_STATE = 42`:

| Model | Test accuracy | Conditions per justification (mean) |
|---|---|---|
| always "Lodgepole Pine" | 48.8% | 0 |
| depth-8 tree, `min_samples_leaf=20` — **ours** | **73.0%** | **7.97** |
| unconstrained tree | 82.6% | 17.88 |

Row two meets the brief. Every one of its predictions comes with a reason a
surveyor could check on site.

Then the takeaway question, which the reader is told **not** to look up: *your
neighbour has fitted the same model, with the same hyperparameters, to almost
the same training set. How similar are the two sets of rules?* Lecture 8
("Retrain it and watch it change") opens on the answer.

---

## Exercises, with the re-run order stated

Cell numbers are the ones in this script. **Cell 4 deletes `X_all` and `y_all`,
so it cannot be re-run on its own, and cell 3 stops working after it.** Any
exercise that touches the split is therefore a restart-and-run-all.

1. **Change the auditable leaf size** (`AUDITABLE_LEAF = 20` → 1, 5, 50).
   Re-run **14 → 15 → 16 → 17 → 18 → 19 → 20**, in that order. Skip cell 21: it
   fits its own tree and does not read `AUDITABLE_LEAF`. Expect the shipped
   accuracy to move between 73.0% and 73.3%, and the smallest leaf in cell 18 to
   track whatever you set. Budget ~5 s.
2. **Change the depth cap** to 6 or 12. Re-run **14 → 20** as above. At depth 12
   the assert in cell 14 fails, which is the brief working correctly — that is
   the expected outcome, not a bug to fix.
3. **Re-run cell 11 after cell 19 and watch a number change.** `acc` is a loop
   variable in cell 11 and the shipped test accuracy in cell 19. Re-running 11
   rebinds `acc` to 0.7861, and cell 21 then prints `(was 78.6%)`. Fix it by
   renaming the loop variable to `cv_acc`; this is the lecture-19 `target` bug,
   in a different notebook.
4. **Change the seed.** `RANDOM_STATE` is set in cell 1 and read by cells 4, 7,
   11, 13, 14 and 21. Because of the `del` in cell 4, this is
   **Runtime → Restart, then Run all**, ~20 s warm. Compare the depth-8 test
   accuracy across two seeds before deciding whether 0.40 points of "price of
   auditability" is inside or outside the noise.

---

## Defects found in the current notebook

`notebooks/lecture-07.ipynb`, 69 cells, checked against `GUIDELINES.md`.
Everything marked **verified** was re-derived with `python3` by re-running the
notebook's own pipeline at `RANDOM_STATE = 42` on scikit-learn 1.7.2 / numpy
2.3.5 / pandas 2.3.3.

### 0. The notebook ships with no stored outputs at all — verified

`sum(len(c.get("outputs", [])) for c in nb.cells) == 0` across all 69 cells.
Under §1.2 ("every figure quoted in markdown must appear in a stored cell
output") **no prose figure in this notebook is reconcilable against it**, so
every number below had to be re-derived from scratch. This is upstream of every
other numeric defect here.

### 1. Every headline number belongs to the model the notebook says it rejected — verified

This is the most serious defect. §12 is a whole section arguing that the grid's
answer (`min_samples_leaf=1`) must be overruled in favour of
`min_samples_leaf=20`, and that the overrule "costs 0.40 points". The cell then
fits `min_samples_leaf=20`. **Every figure quoted afterwards is the
`min_samples_leaf=1` tree's.** Both models, same 12,000 test rows, same seed:

| Prose claim | Where | `min_samples_leaf=20` (shipped) | `min_samples_leaf=1` (rejected) |
|---|---|---|---|
| test accuracy "73.3%" | §15 prompt, §16 table | **73.0250%** | 73.3417% |
| "7.99" conditions | §16 table | **7.9661** | 7.9941 |
| "90% of the 481 training patches" | §12, §14 | **90.93% of 463** | 90.02% of **481** |
| Aspen "never once predicts correctly" | §15 prompt | recall **2.55%** (5 of 196) | 3.06% |

No leaf of the shipped tree contains 481 patches (nearest: 463, 487).
§2.1 and §1.5 both, and it inverts the section's own lesson: the notebook
overrules the grid and then reports the grid's model.

### 2. "73.3%" is also the cross-validated number, mislabelled as a test score — verified

`GridSearchCV` gives 0.7335 at `max_depth=8, min_samples_leaf=20`, i.e. 73.35%
→ "73.3%". The §16 table's column header is **Test accuracy**. Two different
quantities collapsed into one figure, §1.5 exactly.

### 3. §8's cross-validation figures are wrong twice in one bullet — verified

The prompt box says *"A mean of 82.6% built from folds spanning four points"*.
Measured: mean **82.06%**, folds **81.42% – 82.84%**, a span of **1.42 points**.
82.6% is that tree's **test** accuracy (82.6333%), printed in a different cell,
three cells later. Both halves of the sentence are wrong and the sentence exists
to teach the reader to report a fold range.

### 4. The first data cell is thirteen times slower than it says — verified

The comment reads *"~30 s and about 11 MB the first time"*. Measured cold, into
an empty `data_home`: **391.4 s** — 6 min 31 s. Of that, **1.0 s** is the 11.2
MB download (so the size claim is right) and the rest is scikit-learn parsing
the decompressed CSV, single-threaded. Warm: **1.6 s**. §7.1, in the cell where
it does most damage: this is the first cell a student runs, six minutes of
silence reads as a hang, and the module's own docstring says the timings exist
precisely to stop that.

### 5. The other three ⏱ markers overstate by 3–12× — verified

| Marker in the notebook | Measured, `n_jobs=-1`, 16 cores | Measured, `n_jobs=1` |
|---|---|---|
| §8 "about 30 seconds" | **1.6 s** | 1.5 s |
| §10 "about 90 seconds" | **9.8 s** | 9.8 s |
| §11 "about 2 minutes" | **2.2 s** | 26.7 s |

The whole notebook runs in **16 s** warm. The ⏱ ordering is also backwards:
§11's grid search is stated as the slowest cell and is in fact the fastest of
the three, because `GridSearchCV` parallelises all 120 fits while §10's twelve
short `cross_val_score` calls do not parallelise at all. Overstated timings are
a §7.1 defect in the same way understated ones are — a reader who is told two
minutes and sees two seconds concludes the cell did not run.

### 6. §5's cross-reference does not resolve — verified

*"Aspen will be invisible in the confusion matrix twelve sections from now"* —
§5 to §15 is **ten** sections. §3.3.

### 7. The half-integer claim contradicts the output above it — verified

§13's markdown: *"The thresholds sit at half-integers"*. The cell immediately
above calls `export_text(..., decimals=0)`, which prints `Elevation <= 3046` and
`Elevation <= 2510`. The true values are 3046.5 and 2510.5, but the reader
cannot see a single half-integer anywhere on the page. Additionally **71.0%** of
the shipped tree's 162 thresholds end in `.5`, not all of them — the rest are
whole numbers, for the same midpoint reason.

### 8. `2959.5` appears nowhere in the notebook — verified

§13's prompt box says *"assuming a threshold of 2959.5 means something about the
terrain"*. 2959.5 is not a threshold of the shipped tree (its Elevation
thresholds are 2286.0, 2307.5, 2329.0, … 2954.5, 3046.5, …). It **is** a
threshold of the unconstrained `free` tree, which is never printed. §1.2: a
prose figure a reader cannot find in any output.

### 9. §10's "usual student version" describes something that does not happen — verified

*"a linear y-axis … shows one point rising and eleven at zero"* and *"it has
been doubling all along"*. Leaf counts are 2, 4, 8, 16, 32, 60, 113, 206, 350,
539, 782, 1076. On a linear axis with a 1,076 maximum, depths 9–12 sit at 33%,
50%, 73% and 100% of the range — four clearly visible points, not eleven at
zero. And the growth ratios are 2.0, 2.0, 2.0, 2.0, 1.88, 1.88, 1.82, 1.70,
1.54, 1.45, 1.38: doubling for four levels, decaying after. §6.2 — an invented
failure rather than an observed one.

### 10. §11's "barely matters" has no stated range — verified

Along `max_depth=8` the score runs 0.7375 → 0.6969 across the printed row, a
**4.06-point** swing — ten times the 0.40 points §12 calls "the price of
auditability". The claim only holds for `min_samples_leaf ≤ 50` (0.6 points).
The deck states the range ("73.8% at 1, 71.5% at 200"); the notebook drops it.
§1.3.

### 11. §15's cell does not satisfy its own specification — verified

The prompt's constraint says *"report BOTH numbers"* and its `catch` says *"Show
the pair"*. The cell prints `accuracy 59.4% (was 73.0%)` — a pair — and then
`Aspen recall 80.1%` with no unweighted counterpart. The missing number is
**2.6%**, and it is the larger half of the argument (77.5 points of recall
bought for 13.7 points of accuracy).

### 12. `ax` is rebound across types — verified

Cell 38: `fig, ax = plt.subplots(1, 2, ...)` binds `ax` to a
`numpy.ndarray` of shape `(2,)`. Cells 50 and 64 bind the same name to a single
`matplotlib.axes.Axes`. Confirmed by construction. §4.1.

### 13. `acc` is a loop variable and a headline result — verified

Cell 36 binds `acc` inside `for d in range(1, 13)` (a mean CV accuracy); cell 62
binds it to the shipped tree's **test** accuracy; cell 67 prints
`f"(was {acc:.1%})"`. Re-running cell 36 after cell 62 — an entirely natural
thing to do while reading the depth sweep — leaves `acc = 0.7861`, and cell 67
then reports the shipped model's accuracy as *"was 78.6%"*, silently. This is
the `target`-clobbered-by-a-loop-variable defect that lecture 19 spends 200
words on. §4.1.

### 14. Cell 13 is not idempotent, and it breaks cell 10 — verified by inspection

`SUBSAMPLE` reads `X_all`/`y_all` and ends `del cover, X_all, y_all`. Re-running
it raises `NameError`; cell 10 (`print(X_all.columns[:10]...)`) also stops
working once it has run. Restart-and-run-all is fine, so this is §4.3 (an
unnamed out-of-order hazard) rather than a broken notebook — but nothing in the
notebook names it, and §7.2 requires any re-run instruction to say so.

### 15. The defect is announced three times before it fires — verified

Before the `feature_importances_` cell runs, the reader has met: the header's
*"Cells marked **⚠ read before running** contain a defect on purpose"*, §9's
*"**⚠ Read before running.**"*, and a prompt box labelled *"⚠ what the assistant
returns"* whose `left_open` bullet already gives away the answer
(*"`feature_importances_` has 54 entries, one per column, not one per
prediction"*) — and then §"Reviewer question 3" repeats it afterwards. §8.1
exactly: nobody falls in, so "would you have caught it?" has no honest answer.

### 16. Fourteen of sixteen sections carry no examinable marker — verified

§8.3 requires every section to be marked *examinable*, *not examinable —
engineering*, or *beyond the book*. The string appears in three places, all
covering §1 and §13.

### Checked and clean

* **§5.1 / §5.2** — no markdown line indented ≥ 4 spaces outside a fence, no
  indented fence marker, in any of the 69 cells. Machine-checked.
* **§3.1** — one fenced block in markdown (§7's commitment form). It is not
  code and does not claim to be. No `python`-tagged block quotes code that is
  absent from a cell.
* **"Reviewer question 3"** — **not** a dangling reference. It is a course-wide
  numbered convention (lecture 1 slide 367 introduces question 1; question 3 is
  used in lectures 5, 15, 17, 21, 23) and `slides/lecture-07.html` uses the same
  label on the matching slide.
* **"Hold that thought until the next lecture"** — resolves. Lecture 8 is
  *"Retrain it and watch it change"*, which is the instability question §16
  asks.
* Everything in the pipeline that the prose gets **right**, re-derived:
  `(581012, 54)`; labels exactly 1–7; 48,000/12,000; the seven class shares
  (48.8 / 36.5 / 6.2 / 3.5 / 3.0 / 1.6 / 0.5%); ratio 103×; baseline 48.7583%;
  unconstrained depth 33, 5,699 leaves, 100.0% train, 17.8835 mean conditions;
  test 82.6333%; 54 importances; leaves 163; 24 of 54 columns; train 74.4062%;
  smallest/median/largest leaf 20 / 48 / 6,203; the 0.40-point price
  (0.7375 − 0.7335); Aspen 93.9% into Lodgepole Pine; "about a point and a
  half" (191/12,000 = 1.59%); `del`'s "250 MB" (measured 251.0 MB).

### Could not verify

* **"not installed on a stock Colab runtime"** (§13, about the `dot` binary). I
  have no Colab runtime here and the claim is about a third-party image that
  changes between terms. It is stated in the notebook as fact and should either
  be re-checked each term or softened to *"may not be installed"*.
* **"about two metres of paper"** (§13, on drawing all eight levels). Rhetorical
  and not derivable from any notebook output.
* **The version at which `tree_.value` changed from counts to proportions.** I
  verified only that on **1.7.2** it holds proportions, which is what makes
  `f"{dist.max():.0%}"` in §14 print `91%`. The setup cell asserts ≥ 1.4; I did
  not establish whether 1.4 is sufficient. Worth pinning, because on a version
  that stores counts that line prints a nonsense percentage rather than failing.
* **The 391 s cold fetch on Colab.** Measured here only. The download portion
  (1.0 s of it) will differ; the ~390 s parse is CPU-bound and single-threaded,
  so a slower core makes it worse, not better.
