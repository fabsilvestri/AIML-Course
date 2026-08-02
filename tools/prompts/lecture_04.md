# Lecture 4 — *It never fires* · a Colab build script

**Applications of Machine Learning** — BSc Mathematics of Artificial Intelligence,
Sapienza. Géron, Chapter 3. Mathematical thread: **imbalance, and the
non-monotonicity of precision**.

This file is what you type. Open a blank Colab notebook, and for each cell below
type the prompt, read what comes back against the **Expect** line, then run it.
The **Assert** line is what you add by hand if the assistant did not add it
itself — it is the part that fails loudly when the answer is wrong.

Lecture 3 ended with a detector that is 96.9% accurate and a sheet of paper with
four numbers on it. This notebook is the Fix half: it finds out what that 96.9%
was made of, and what to report instead.

---

## Before you start

**Hardware.** Everything here runs on CPU. There is no GPU in this notebook and
no cell needs one.

**Wall clock.** Four cells dominate. The figures below come from a 16-core
arm64 laptop that was **heavily loaded at the time** (load average ~190), so
treat the SGD numbers as upper bounds and the ratios as the reliable part:

| Cell | What it does | measured, 16-core (loaded) | Colab, 2 vCPU |
|---|---|---|---|
| 3 | `fetch_openml("mnist_784")`, cached | **2.4 s** | 2–5 s |
| 3 | the same, cold (downloads ~130 MB) | not measured | ~60–90 s, network-bound |
| 5 | `cross_val_predict`, SGD, 3 folds of 40,000 | 3 × **43.8 s** per fold | **3–6 min** |
| 10 | one SGD pipeline fit on all 60,000 rows | **103 s** and **122 s**, two runs | **2–5 min** |
| 14 | `cross_val_predict`, 100-tree forest, 3 folds | ~25 s (extrapolated) | ~60 s |
| 15 | one forest fit on all 60,000 rows | ~10 s (extrapolated) | ~30 s |

Budget **10–20 minutes** of compute for a cold run on Colab free. Nothing else
in the notebook takes more than a second.

The load-independent reason cell 10 is slow: `SGDClassifier(random_state=42)`
on scaled MNIST stops at **`n_iter_ = 746`** — 746 passes over 60,000 × 784
floats (measured). It is single-threaded Cython, so extra cores do not help it.
The forest is the opposite: it scales with cores. The *slow* model in this
notebook is the linear one, which is not what anybody expects.

The forest figures are extrapolated from measured 4-tree and 8-tree fits at
`n_jobs=2` (1.3 s and 2.3 s on 60,000 rows), not measured at 100 trees.

**Memory.** In scikit-learn 1.7, `fetch_openml("mnist_784", as_frame=False)`
returns `int64` pixels: `X_train` alone is **376 MB**, and `StandardScaler`
makes a `float64` copy of the same size. Colab free (12.7 GB) survives this;
a 4 GB machine may not. If you are tight, `X = mnist.data.astype(np.uint8)`
right after loading cuts it to 47 MB and changes nothing downstream, because the
scaler converts to float anyway.

**Vocabulary**, defined once, because this lecture and Lecture 18 share it:

- **base rate** — the fraction of the data that is positive. Here 0.09035.
- **operating point** — a model plus one chosen threshold. Not a model.
- **shift** — a period of work at a desk. "The test shift" is the 10,000 test
  images treated as one day's arrivals for a human reviewer. It has nothing to
  do with `numpy.shift` or a time shift.
- **support** — how many instances are actually flagged at a given threshold.
  A precision of 90% over 12 flagged digits and over 4,000 flagged digits are
  the same number and not the same claim.
- **step down / step up / flat** — what precision does when you accept one more
  instance from the ranking. Lecture 18 §5.2 uses these exact three words on
  detection boxes; keep them the same way round here.
- **red-team** — swap work with another pair and try to break their result.

**Examinability**, marked per section: *examinable*, *not examinable —
engineering*, or *beyond the book*.

---

## §1 · Setup and where we left off — *not examinable, engineering*

## Cell 1 — versions and one seed
**Prompt to type:**
> Print the python, numpy and scikit-learn versions, and set a single
> RANDOM_STATE I can reuse for every split, model and shuffle in this notebook.
> Make it fail loudly if scikit-learn is older than 1.3.

**Expect:** three version lines, an `assert` on the scikit-learn version, and
`RANDOM_STATE = 42`. Not a `print` of the version — a check.
**Assert:** `assert tuple(int(p) for p in sklearn.__version__.split(".")[:2]) >= (1, 3)`
**Annotate:** short

> Nothing in this notebook actually requires a modern scikit-learn — every
> function it calls is a decade old. 1.3 is a defensible floor because it is the
> last release that touched `precision_recall_curve`'s signature (it added
> `drop_intermediate`, default `False` — verified in the installed docstring).
> Do **not** let the assistant pin 1.4 "because of `root_mean_squared_error`":
> that is a regression metric, this is a classification notebook, and it is the
> reason the current notebook gives — see the defect report.

## Cell 2 — every import, in one cell
**Prompt to type:**
> Put every import this notebook will need into one cell at the top: matplotlib,
> `fetch_openml`, `SGDClassifier`, `RandomForestClassifier`, `StratifiedKFold`,
> `cross_val_predict`, `make_pipeline`, `StandardScaler`, and from
> `sklearn.metrics`: accuracy, precision, recall, f1, confusion_matrix,
> `precision_recall_curve`, `roc_auc_score`, `average_precision_score`.
> Nothing below this cell may import anything.

**Expect:** one import block, no `import` statement anywhere below it.
**Assert:** none
**Annotate:** short

> Ask for exactly the names you listed. If the assistant adds `roc_curve` or
> `cross_val_score`, delete them — the current notebook imports both and calls
> neither.

## Cell 3 — MNIST, and the split that already exists
**Prompt to type:**
> Load MNIST from OpenML and build a boolean "is it a 5" target. Split it into
> 60,000 training and 10,000 test images. Print the base rate and the accuracy a
> classifier that always says "not a 5" would get. Then build me a pipeline of
> StandardScaler + SGDClassifier and a 3-fold stratified CV object, both using
> RANDOM_STATE.

**Expect:** `X.shape == (70000, 784)`; `y_train_5.sum()` **5421**;
`y_test_5.sum()` **892**; base rate **0.09035**; never-fires accuracy
**0.90965**. (All four verified against the cached OpenML copy.)
**Assert:**
```python
assert len(X_train) == 60000 and len(X_test) == 10000
assert y_train_5.sum() == 5421 and y_test_5.sum() == 892
```
**⏱** first call downloads ~130 MB and is network-bound — allow **60–90 s** on
Colab free; I did not measure a cold download. Cached, it is **2.4 s** measured.
**Annotate:** full

- **Left open:** *which* 60,000. The prompt says the sizes and not the rows.
  MNIST's canonical partition is positional — training is rows 0…59,999,
  test is rows 60,000…69,999 — and "split it into 60,000 and 10,000" does not
  say that.
- **The usual student version:** `train_test_split(X, y, test_size=10000,
  random_state=42)`. `train_test_split` has **`shuffle=True` as its default**,
  so the two halves it returns are a fresh random partition of all 70,000 rows.
  It runs, the shapes are right, and every MNIST number published since 1998
  stops being comparable with yours. It is also the habit the previous three
  lectures installed.
- **How you would catch it:** the two asserted counts. 5,421 and 892 are
  properties of the canonical partition, not of *a* 60/10 split — a shuffled
  split lands near them and essentially never on them. Add the asserts before
  you look at the output.

---

## §2 · Where 96.90% came from — *examinable*

Write $P$ for the positives, $N$ for the negatives, $m = P+N$, and $p = P/m$
for the base rate. Splitting the confusion counts by true class:

$$\text{accuracy} = \frac{\mathrm{TP}+\mathrm{TN}}{m}
 = \frac{P}{m}\cdot\frac{\mathrm{TP}}{P} + \frac{N}{m}\cdot\frac{\mathrm{TN}}{N}
 = p\,\text{recall} + (1-p)\,\text{specificity}$$

Accuracy is a weighted average of two rates, weighted by the class sizes. Two
identities follow, not estimates:

- a classifier that never fires has recall 0 and specificity 1, so it scores
  exactly $1-p$ — here **0.90965**;
- $\partial\,\text{accuracy}/\partial\,\text{recall} = p$, so at our base rate
  ten points of recall buy **0.90** points of accuracy.

## Cell 4 — what accuracy is worth at four base rates
**Prompt to type:**
> Before we fit anything: print, for base rates 0.5, 0.09035, 0.01 and 0.001,
> the accuracy of a model that always predicts the negative class. Label the
> 0.09035 row as ours.

**Expect:** four lines. Ours reads `0.09035 -> 0.90965`. The 0.001 row reads
0.999, which is the point.
**Assert:** none
**Annotate:** short

> Say "label ours" explicitly. The current notebook prints the list without
> marking which row is this dataset, and the whole argument turns on that row.

## Cell 5 — out-of-fold scores
**Prompt to type:**
> Give me one cross-validated decision-function value per training instance,
> using that pipeline and that cv object, and print the accuracy you get by
> thresholding them at zero.

**Expect:** `y_scores.shape == (60000,)`, dtype float, and an accuracy of about
**0.969** — the number Lecture 3 ended on, reproduced here from scores rather
than from labels.
**Assert:**
```python
assert y_scores.shape == (60000,)
y_pred = (y_scores >= 0)
```
**⏱** three fits on 40,000 rows each. One such fit measured **43.8 s** on a
16-core laptop under heavy load; with `n_jobs=-1` the three folds run in
parallel, so allow **45–90 s** there and **3–6 minutes** on Colab free, where
two vCPUs and a single-threaded `SGDClassifier` make the folds queue.
**Annotate:** full

- **Left open:** that these are the only scores any later cell is allowed to
  choose a threshold on. Nothing in the code stops cell 12 or cell 15 reaching
  for the test set instead, and the whole lecture rests on them not doing so.
- **The usual student version:** leaving `method=` off. `cross_val_predict`'s
  documented default is **`method='predict'`**, so you get 60,000 booleans
  instead of 60,000 scores. Nothing raises — and `precision_recall_curve` will
  happily accept those booleans: fed hard 0/1 labels it returns **3 precisions,
  3 recalls and 2 thresholds** (measured). You get a three-point "curve", plot
  it, and conclude the trade-off is a straight line.
- **How you would catch it:** two checks, both one line. `y_scores.dtype` must
  be a float, not `bool`. And `(y_scores >= 0)` must equal what `predict` would
  return, because `SGDClassifier` predicts positive exactly when the decision
  function is non-negative — if those disagree you have the wrong estimator or
  the wrong method.

### Part two — turn the dial

Every scoring classifier is a **family** of classifiers, one per threshold $t$:
predict positive when $s(x) \ge t$. Lowering $t$ accepts more instances from the
top of the ranking and never releases one, so $\mathrm{TP}(t)$ and
$\mathrm{FP}(t)$ are both non-decreasing as $t$ falls.

**Recall is monotone.** Its denominator is $\mathrm{TP}+\mathrm{FN} = P$, which
counts every positive in the data whatever we predict. A non-decreasing quantity
over a constant.

**Precision is not.** Its denominator is the flagged count, which we choose.

*The one-step lemma.* Accept one more instance. With $T = \mathrm{TP}$ and
$n = T + \mathrm{FP}$ before the step, and $y \in \{0,1\}$ the label of the
instance accepted:

$$\frac{T+y}{n+1} > \frac{T}{n} \iff n(T+y) > T(n+1) \iff y > \frac{T}{n}$$

Since $y$ is 0 or 1: accepting a **negative** is a step **down**; accepting a
**positive** is a step **up**, or **flat** when precision is already 1. That is
the same sentence Lecture 18 §5.2 asserts on detection boxes — *every false
positive is a step down, every true positive but the first is up or flat.*

> **Direction matters, and it is the one thing people get backwards.** Read the
> ranking top-down (accepting instances, lowering the threshold): a negative is
> a step **down**. Read it bottom-up (raising the threshold, dropping
> instances): a negative is a step **up**. Same steps, opposite words. This
> script always reads top-down, so that Lecture 18 can quote it unchanged.

## Cell 6 — classify every step of the ranking
**Prompt to type:**
> Sort all 60,000 instances by score, highest first. Walk down the list and
> compute precision at each cut-off. Then count how many steps go down, how many
> go up and how many are flat, and check those counts against the number of 5s
> and non-5s in the data.

**Expect:** three counts summing to **59,999**. Steps down: exactly **54,579** —
the number of non-5s in the training set. Steps up plus flat: exactly **5,420** —
the 5,421 fives less the one at the top of the ranking, which has no step above
it. Not a tendency: an exact classification of every step.
**Assert:**
```python
order = np.argsort(-y_scores, kind="stable")
lab   = y_train_5[order].astype(np.int64)
tp    = np.cumsum(lab)                       # true positives in the top k
prec  = tp / np.arange(1, len(lab) + 1)
d     = np.diff(prec)
accepts_a_five = (lab[1:] == 1)

assert lab[0] == 1, "the lemma below assumes precision is never 0"
assert (d[~accepts_a_five] < 0).all()      # every non-5 is a step down
assert (d[accepts_a_five] >= 0).all()      # every 5 is a step up, or flat at 1
assert (d < 0).sum() == int((~y_train_5).sum())
assert (d >= 0).sum() == int(y_train_5.sum()) - 1
```
**Annotate:** full

- **Left open:** how ties are broken. Thousands of instances share a score, and
  the arbitrary order inside a tie moves instances between the "up" and "flat"
  buckets. The two bucket *totals* are forced by the labels; the split between
  them is not.
- **The usual student version:** `np.argsort(-y_scores)` with the default.
  `numpy`'s documented default is **`kind='quicksort'`** — an introsort, and
  not stable. Same data, same seed, a different numpy build, a different
  up/flat split. `kind='stable'` costs a few milliseconds and pins it. The
  second usual version is sweeping 100 evenly spaced thresholds and reporting
  the curve as monotone: the steps are one instance wide and a grid steps over
  every one of them.
- **How you would catch it:** 54,579 is not a measurement, it is `(~y_train_5).sum()`.
  I checked this against three different random rankings of the same labels and
  got 54,579 every time — it is forced by the labels alone for any ranking whose
  top instance is a 5. If your "steps down" count is anything else, your sort is
  ascending, or the top of your list is a non-5, and then the first assert above
  is the one that fires.

## Cell 7 — the smallest example, by hand
**Prompt to type:**
> Print precision at the first eight cut-offs as fractions, not decimals. Then
> find the first cut-off where precision goes down and the first where it goes
> up, and print the label of the instance accepted at each.

**Expect:** eight lines like `top-6: 5/6 = 0.8333`. The first step down accepts
a non-5, the first step up accepts a 5. Do the division yourself before you
believe the counts in cell 6.
**Assert:**
```python
first_down = int(np.flatnonzero(d < 0)[0]) + 1
first_up   = int(np.flatnonzero(d > 0)[0]) + 1
assert lab[first_down] == 0 and lab[first_up] == 1
```
**Annotate:** short

> Do **not** let the assistant write `assert prec[4] < prec[5]`. That is what
> the current notebook has, and it is a hard-coded fact about which digit
> happens to sit at rank 6 of one SGD run. The two asserts above are the lemma
> and hold for any ranking.

---

## §3 · Diagnose — what the number was hiding — *examinable*

Nothing new gets fitted. The same out-of-fold predictions, asked a different
question.

## Cell 8 — the four numbers
**Prompt to type:**
> Print the confusion matrix for `y_pred` against `y_train_5`, and then print
> the four cells with their names, because I can never remember scikit-learn's
> ordering.

**Expect:** a 2×2 array; `ravel()` gives **tn, fp, fn, tp in that order**. `tn`
should be by far the largest cell — around 54,000 of 60,000 rows are not 5s. If
your "true negatives" is small, you unpacked it backwards.
**Assert:** `assert tn + fp + fn + tp_ == 60000`
**Annotate:** short

> Name the fourth variable `tp_`, not `tp`. `tp` is the cumulative array from
> cell 6 and is still live.

## Cell 9 — the identity behind the accuracy
**Prompt to type:**
> Print accuracy, precision, recall, F1 and specificity for the same
> predictions. Then check numerically that accuracy equals
> `p*recall + (1-p)*specificity` on our own numbers, and print what fraction of
> the accuracy the specificity term alone accounts for.

**Expect:** accuracy ≈ 0.969, recall ≈ 0.77, specificity ≈ 0.989, and the two
sides of the identity agreeing to floating point. The specificity term
`(1-p)*specificity / accuracy` comes out around **93%** — around, not exactly,
because it depends on the fold split. **It is not 91%.** 91% is $1-p$, the
*weight*; the *share* is the weight times the specificity, over the accuracy.
The current notebook's box confuses the two.
**Assert:** `assert np.isclose(accuracy_score(y_train_5, y_pred), base_rate * recall + (1 - base_rate) * spec)`
**Annotate:** short

Two sentences about the same model, both true:

- *"The detector is 96.9% accurate."* — signed off, deployed.
- *"The detector misses about 23% of the 5s."* — something an audit team can act on.

Accuracy was not wrong. It answered a question nobody in the brief had asked, it
weights the class we care about by 0.09, and it prices two very different errors
identically. **And nobody chose it.** It arrived as a default, from a prompt that
named no metric.

---

## §4 · Repair — the threshold is a dial — *examinable*

`predict()` hides the score and hard-codes the threshold at zero. There is no
`set_threshold()`, and there should not be: you compute the scores and compare
them yourself.

## Cell 10 — one digit, two thresholds
**Prompt to type:**
> Fit the pipeline on all 60,000 training rows. Take `X_train[0]`, which is a 5,
> print its decision-function score, and show what the prediction would be at
> threshold 0 and at threshold 3000.

**Expect:** one score, comfortably positive; `True` at threshold 0 and `False`
at 3000. One digit, one model, two answers.
**Assert:** none
**⏱ the slowest cell in the notebook.** Measured **103 s** and **122 s** on two
runs on a 16-core laptop under heavy load; budget **2–5 minutes** on Colab free
and do not be alarmed if it is more. Extra cores do not help — the classifier is
single-threaded and stops at `n_iter_ = 746` epochs (measured). The scaler is
*not* the cost: converting 376 MB of `int64` to `float64` took **1.6 s**.
**Annotate:** short

> This model was fitted on data including `X_train[0]`, so its score for that
> digit is not out-of-fold. That is fine here — it demonstrates a mechanism and
> no number from it gets reported. The *same fitted object* is reused in cell 15
> to score the **test** set, which is a different thing and is legitimate: fitted
> on train, measured on test.

## Cell 11 — the two curves
**Prompt to type:**
> Compute the precision-recall curve from the out-of-fold scores. Plot precision
> and recall against threshold in one panel, and precision against recall in the
> other.

**Expect:** `precisions` and `recalls` are **one element longer than
`thresholds`**. The threshold panel needs `precisions[:-1]` and `recalls[:-1]`;
the PR panel uses the full arrays. The precision curve is visibly a **sawtooth**,
not a slope — the same shape Lecture 18 zooms into on detection boxes.
**Assert:** `assert len(precisions) == len(recalls) == len(thresholds) + 1`
**Annotate:** full

- **Left open:** which array to slice, and why they differ at all.
  scikit-learn appends a final point where nothing is flagged: recall 0 and
  precision **defined** to be 1. It is a convention, not a measurement, and it
  has no threshold to go with it.
- **The usual student version:** mixing the two lengths in the same notebook.
  This is not hypothetical — the current lecture-4 notebook does it: cell 32
  indexes `precisions[:-1]`, and cell 40, eight cells later, indexes the
  unsliced `precisions` for the same question. They agree here by luck. Where
  they stop agreeing is `thresholds[(precisions >= 0.99).argmax()]`, which
  raises `IndexError` the moment the answer is that final degenerate point —
  and returns a wrong-but-plausible threshold otherwise.
- **How you would catch it:** the assert, and then one rule applied without
  exception: decide whether an index lives in the length-$T$ world
  (`thresholds`) or the length-$T{+}1$ world (`precisions`, `recalls`), and
  slice `[:-1]` at the boundary every time. The assert is three symbols and it
  pins a convention that has moved between library versions.

### Choose a threshold deliberately

`argmax` on a boolean array returns the index of the first `True`. Compact,
not obvious, worth reading once.

## Cell 12 — the 90%-precision operating point
**Prompt to type:**
> Find the threshold where precision first reaches 90%, and print the precision
> and recall you actually get there, plus how many digits are flagged at that
> threshold. Then do the same at 99% precision.

**Expect:** at 90% precision, recall around **0.73** with **a few thousand**
digits flagged. At 99% precision, recall around **2%** with **of order a
hundred** digits flagged. Write both flagged counts down — the second one is the
finding. (Those are the figures the current notebook reports — 4,416 and 116, on
a 21-threshold plateau. I could not re-derive them without running the SGD fit,
so treat them as the shape of the answer rather than the answer.)
**Assert:**
```python
flagged = recalls[:-1] * int(y_train_5.sum()) / precisions[:-1]   # tp/precision = tp+fp
i90 = int((precisions[:-1] >= 0.90).argmax())
i99 = int((precisions[:-1] >= 0.99).argmax())
assert i99 >= i90 and flagged[i99] <= flagged[i90]
print(f"support at 90%: {flagged[i90]:,.0f}   at 99%: {flagged[i99]:,.0f}"
      f"   ratio {flagged[i90] / flagged[i99]:.0f}x")
```
That assert cannot fail, and that is the point of it: $\{p \ge 0.99\} \subseteq
\{p \ge 0.90\}$, so the 99% crossing sits at a threshold at least as high, and
`flagged` is non-increasing in the threshold. The **ratio** is what you are
here to read, and it is not structural — it is a measurement of this model.
**Annotate:** full

- **Left open:** whether one index is enough. §2 has just proved this curve is a
  sawtooth, so "the first index reaching 90%" can in principle be a single lucky
  step held up by a handful of digits. The prompt above does not ask for a guard,
  which is exactly why it asks for the flagged count.
- **The usual student version:** `(precisions >= 0.90).argmax()` and nothing
  else — and then quoting the recall at 99% precision as an operating point. It
  is an operating point only if a hundred flagged digits out of 60,000 is a plan
  anybody can staff. Observed twice: the current lecture-4 notebook's cell 40
  computes its headline recall-at-90%-precision with a bare
  `recalls[(precisions >= 0.90).argmax()]`, eight cells after the section that
  argues against it — and so will **cell 14 of this script**, because the prompt
  I wrote for it does not ask for a guard either.
- **How you would catch it:** add `MIN_SUPPORT` and print whether it changed the
  answer, either way. At 90% precision it will not — the crossing has thousands
  of digits behind it. At 99% it will. **A guard that never fires anywhere has
  not been tested**, and reporting "the guard held" from a notebook where the
  guard could not have moved is worth nothing.

---

## §5 · PR or ROC? — *examinable*

The ROC curve plots recall against the false positive rate. Both denominators
are fixed by the data, which is why it behaves so much better than the PR curve —
and that is a warning, not a recommendation.

The rule: **prefer the PR curve when the positive class is rare.** Do not take it
on trust. Hold the model, the ranking and the scores fixed, and move only the
balance.

## Cell 13 — what rarity does to the metrics
**Prompt to type:**
> Take the same scores and labels. Keep every negative, and randomly subsample
> the positives so the base rate becomes 2% and then 1%. Print ROC AUC and
> average precision at the original base rate and at those two.

**Expect:** three rows. ROC AUC moves by well under a point across all three;
average precision falls by a lot. Same model, same ranking, same scores in all
three rows — only the prevalence moves. That is the argument, and it is made by
construction rather than by assertion.
**Assert:** `assert abs(auc_hi - auc_lo) < 0.01 < (ap_hi - ap_lo)`
**Annotate:** short

> Keep the three results in a **list of tuples**, not a dict keyed on the
> rounded base rate. The current notebook writes `row[round(target, 4)]` and
> then looks it up as `row[0.0904]`. That works only because `base_rate` is a
> `np.float64`: `round(np.float64(5421/60000), 4)` is `0.0904`, while Python's
> built-in `round` on the same value as a plain float gives **`0.0903`**. Both
> verified. Convert the base rate to a Python float anywhere upstream — with
> `.item()`, with `float()`, by round-tripping through pandas — and the cell
> dies with `KeyError: 0.0904`.

The ROC curve cannot see the problem you have. That is the whole reason to learn
two curves rather than one.

---

## §6 · A better classifier — *not examinable, engineering*

A forest has no `decision_function`. It has `predict_proba`, one column per
class; the column for the positive class plays exactly the role of the score,
and every curve above works unchanged.

## Cell 14 — a second model, scored the same way
**Prompt to type:**
> Do the same out-of-fold thing with a 100-tree RandomForestClassifier on the
> same rows and the same folds, using `predict_proba`. Then print ROC AUC,
> average precision, and recall at 90% precision for the SGD and the forest side
> by side.

**Expect:** `y_proba.shape == (60000, 2)`. The forest should be ahead on every
row. ROC AUC and average precision move by a point or two; the
recall-at-90%-precision row moves by tens of points — the current notebook
reports about 24. Same 60,000 rows, same three folds, same seed for both models:
the comparison is on **matched rows**, and that is the only reason it means
anything. Say so when you report it.
**Assert:** `assert y_proba.shape == (60000, 2)`
**⏱** three forest fits on 40,000 rows. **~25 s** on 16 cores, **~60 s** on
Colab free — both **extrapolated** from measured 4- and 8-tree fits, not
measured at 100 trees. Set `n_jobs=-1` on the forest **or** on
`cross_val_predict`, not both — nesting two pools oversubscribes a 2-vCPU
machine and is slower than one.
**Annotate:** full

- **Left open:** which column. Nothing in the prompt says. `predict_proba`
  returns columns in `estimator.classes_` order, and for a boolean target
  `classes_` is `array([False, True])`, so the positive class is **column 1**.
  That is a fact about label ordering, not about probability.
- **The usual student version:** `y_proba[:, 0]`. It is the exact complement, so
  it ranks everything backwards, and the ROC AUC comes out at roughly one minus
  the right answer. A number near 0.03 reads as a catastrophically bad model
  rather than as a sign flip, and people go and tune hyperparameters.
- **How you would catch it:** an AUC below 0.5 from a model that trained without
  complaint is almost always the wrong column or an inverted label. Print
  `forest.classes_` once. If you want it to survive somebody changing the label
  dtype from `bool` to `{0,1}` to `{"5","not5"}`, index with
  `list(forest.classes_).index(True)` instead of a literal 1.

Accuracy would call the forest an improvement of a point or two, which reads as
polish. The last row of that table moves by tens of points, and it is the row
the brief is about. **Which description you report decides whether anyone
approves the change.**

> Do not quote an accuracy for the forest unless you compute one. The current
> notebook's prose quotes "a 1.87-point improvement" and no cell in it computes
> the forest's accuracy at all.

---

## §7 · The operating point, and the test shift — *examinable*

What the client asked for, in our vocabulary: **catch at least 90% of the 5s**
on a shift, and **flag no more than 1,000 items** out of the 10,000 scanned.

Two constraints, one dial. A threshold either satisfies both or it does not, and
the honest answer may be that none does.

> **The trap in the next cell is an index.** The precision constraint in cell 12
> wants the **first** crossing — the lowest threshold that reaches the precision
> floor. The recall constraint here wants the **last** — the highest threshold
> that still clears the recall floor. `argmax` gives you the first,
> `np.where(...)[0][-1]` gives you the last, and getting these the same way
> round produces a threshold that is wrong by a large margin and raises nothing.
> Write down which end you want before you type the prompt.

## Cell 15 — choose, then measure once
**Prompt to type:**
> For each model, take the highest threshold on the cross-validated training
> curve that still gives at least 90% recall, and print both thresholds. Then
> fit both models on the full training set and build the three sets of test-set
> predictions: SGD at threshold 0, SGD at its tuned threshold, forest at its
> tuned threshold. Don't score anything yet.

**Expect:** two thresholds printed, then a dict of three boolean arrays of
length 10,000. **No metric printed in this cell.** The cell must read top to
bottom as choose-then-measure: if any threshold is assigned below a printed test
number, the choice was informed by the answer.
**Assert:** none
**⏱** one forest fit on 60,000 rows: **~10 s** on 16 cores, **~30 s** on Colab.
The SGD is already fitted from cell 10 — reuse it, do not refit, or you pay
another two minutes for nothing.
**Annotate:** short

> **A threshold is a hyperparameter.** Choosing it by looking at test
> performance is the same error as choosing $\alpha$ that way and has the same
> optimistic bias. Choose on cross-validated training scores; measure once.

## Cell 16 — the constraint the metric cannot see
**Prompt to type:**
> For each of those three operating points, print how many test items it flags,
> how many 5s it catches, how many false alarms, how many it misses, and the
> recall. Mark any row that flags more than 1,000 items, because that is what the
> desk can re-check in a shift.

**Expect:** three rows. The SGD row tuned for 90% recall flags **more than
1,000** and gets marked. The forest row tuned for the same 90% recall comes in
**under** capacity, because almost none of what it flags is a false alarm.
**Assert:** `assert any(p.sum() > CAPACITY for p in shift.values())`
**Annotate:** short

> That assert can legitimately fail, and if it does the correct response is to
> write down that the capacity constraint is not binding on your run — not to
> lower `CAPACITY` until it is. It is the one assert in this notebook that is a
> question rather than a check.

Read what you got, in this order:

1. **The SGD cannot satisfy both constraints.** No threshold gives it 90% recall
   inside 1,000 flagged items. That is a finding and it is the correct thing to
   report. It is not a reason to lower the recall target quietly.
2. **The forest's test recall will land slightly off 90%** — the threshold was
   chosen at exactly 90% on cross-validated training scores, so it lands either
   side of the target about half the time. Report it. Re-tuning until the test
   number reaches 90.0 is fitting the test set, and it is a one-line change that
   nobody would notice in review.

---

## §8 · An assistant improves the recall — *examinable*

Here is the request, typed exactly as somebody would type it on a Friday:

> *"My detector is missing too many 5s. Fix it so the recall is high."*

## Cell 17 — the assistant's answer
**Prompt to type:**
> My detector is missing too many 5s. Fix it so the recall is high.

**Expect:** a threshold picked off the training curve at some high recall target
the assistant chose for you (99% in the current notebook — the prompt named no
number), applied to the test shift, and **one** printed line: recall before →
recall after. Both numbers go on your sheet of paper **now**, before you scroll.
If your assistant printed more than recall, note what it added and to what
prompt — that is a finding too.
**Assert:** none
**Annotate:** short

**Stop here.** Write down the two recalls. Then answer, out loud, before the
next cell: *what would you have said to the client?*

---

## §9 · What it cost — *examinable*

⚠ Cell 17 runs, it is correct code, it does exactly what was asked, and the
sentence it printed is true. It printed **one number**. That is the tell.

## Cell 18 — the review question
**Prompt to type:**
> Now put recall, precision and accuracy before and after side by side on the
> test set, print how many items each flags against the 1,000 capacity, and how
> many extra false alarms per shift the change costs.

**Expect:** recall up, precision down, accuracy down — and the accuracy after
the change sitting **below the never-fires baseline of 0.91080** (892 fives in
10,000 test images; verified). The flagged count runs to **several times**
capacity, and the extra false alarms number in the thousands.
**Assert:**
```python
assert accuracy_score(y_test_5, after) < 1 - y_test_5.mean()
```
**Annotate:** full

- **Left open:** which of these the client would have cared about. The table
  does not rank them, because ranking them is the conversation the table exists
  to start.
- **The usual student version:** cell 17, one cell up. That is not a
  hypothetical student — it is what the request produced, and every word of what
  it reported was true. Asking for more of a number is the most natural request
  in machine learning and the most reliable way to get a worse model.
- **How you would catch it:** the never-fires baseline, 0.91080. An accuracy
  below the model with no parameters is the loudest signal available, and no
  single improved metric rescues it. If you only ever add one assert to a
  notebook, add that one.

Nothing was retrained. One number was changed. The assistant did not write a
bug — **we wrote a bad specification.**

Lecture 3's failure — scoring on the rows the model was fitted on — cost about
half a point of accuracy. This one costs tens of points and several times the
desk's capacity, from a prompt that looks just as reasonable.

### The corrected specification

> *"Choose a decision threshold on **cross-validated training scores** that
> reaches **at least 90% recall** while flagging **no more than 1,000 items per
> 10,000**. Report precision, recall and the flagged count together. If no
> threshold satisfies both, say so and show the closest."*

It names the objective *and* the constraint, names the data the choice is made
on, demands that metrics which move together be reported together, and gives
permission to fail — which is what stops it inventing a success.

**Never optimise one metric of a pair.** Precision and recall, bias and variance,
latency and accuracy. Ask for one and you will get it, at the price of the other,
and the price will not be in the reply.

### Forward reference

Lecture 18 reuses this section on object detection: **precision is not
monotone — application 2, on boxes**. It classifies every step of a detection
ranking the same way cell 6 does here, and then introduces the repair this
lecture does not have — average precision, defined on the **maximum precision at
or above each recall level**, which exists for exactly one reason: to replace a
non-monotone quantity by a monotone one. No threshold appears in it. When you get
there, cell 6 is the cell to come back to.

One warning, because the two lectures currently disagree. The `average
precision` that cells 13 and 14 print is
`sklearn.metrics.average_precision_score`, which is
$\sum_n (R_n - R_{n-1})P_n$ on the raw curve — **no envelope**. Lecture 18
defines its own `average_precision` on the enveloped curve. On the five-detection
ranking TP, FP, FP, TP, TP the two give **0.700** and **0.7333** (verified). Same
words, two estimators. Say which one you mean.

---

## §10 · Red-team — *not examinable, engineering*

Swap notebooks with the pair beside you. Eight minutes. Report what you
**found**, not what you would have done differently.

1. What touched the test set? Was the **threshold** chosen on it?
2. What was fitted, and on what rows?
3. What is the shape here — and which column of `predict_proba` did they take?
4. What was dropped? Any rows removed to make a number look better?
5. What is the default nobody asked for — **including the threshold at zero**?

---

## Exercises

Each one lists the cells to re-run, in order (§7.2). Re-running cell 5 or cell 10
costs minutes; the rest cost seconds. Nothing here needs a GPU.

**1 · Move the capacity.** The desk hires a second reviewer: `CAPACITY = 2000`.
Does the SGD now satisfy both constraints? Report the flagged count, not just
yes/no.
→ **Re-run cell 16 only.** Nothing is refitted. ~1 s.

**2 · Make the guard fire.** In cell 12, raise the precision floor from 0.90
until the flagged count drops below 500. At what precision does the support give
way?
→ **Re-run cell 12 only.** ~1 s.

**3 · Break the sort.** In cell 6, change `kind="stable"` to `kind="quicksort"`
and re-run. Do the three counts change? Do the two bucket totals change?
→ **Re-run cell 6, then cell 7.** Cell 7 reads `d` and `lab` from cell 6. ~2 s.
Expect the totals to be identical and the up/flat split to move, or not to move
at all on your numpy build — either outcome is the answer.

**4 · Take the wrong column.** In cell 14, change `y_proba[:, 1]` to
`y_proba[:, 0]` and re-run **only the printing**, not the fit.
→ Split cell 14 into two cells first: the `cross_val_predict` call, and
everything after it. Then this exercise is **re-run the second half only**, ~1 s.
If you did not split it, this exercise costs you the full forest CV again.

**5 · Change the seed.** Set `RANDOM_STATE = 7` and see whether the forest's
test recall still lands within half a point of 90%.
→ **Re-run cells 1, 3, 5, 10, 11, 14, 15, 16, in that order.** Cell 3 rebuilds
`clf` and `cv` from the new seed; cells 5 and 10 are the expensive ones. Budget
**10–15 minutes** on Colab free. Do not skip cell 11 — cells 12 and 15 read
`precisions`, `recalls` and `thresholds` from it, and cell 15 also reads `f_rec`
and `f_thr` from cell 14.

**6 · The honest report.** In four sentences, write what you would send the
client, given the table from cell 16. It must contain the flagged count, the
recall, and the sentence "no threshold satisfies both constraints" if that is
true of your run.
→ No cells. Bring it to the next lecture.

---

## Defects found in the current notebook

Everything below refers to `notebooks/lecture-04.ipynb` as it stands (56 cells,
18 code cells). **Verification status is stated for each item.** MNIST was read
from the local scikit-learn cache; the SGD and forest fits were *not* executed,
per the brief, so anything that depends on those scores is marked as such.

### Verified by execution or by parsing the notebook

**1 · No stored outputs anywhere.** All 18 code cells have `outputs: []`. Every
figure in the prose therefore violates §1.2 — none of them appears in a stored
output and none can be reconciled without re-running the notebook. *Checked:
parsed the `.ipynb`, total stored outputs = 0.* This is the root cause of
items 5, 6 and 7, and of every unverifiable figure in the last section below.

**2 · Four cross-references that do not resolve (§3.3).** *Checked by resolving
each index against the cell list.*
- Cell 4: "cell 10 asserts the current shape". Cell 10 is the base-rate
  loop and contains no assert. The `precision_recall_curve` length assert is
  in **cell 29**.
- Cell 4: "cell 14 works today because cell 4 is still in the kernel".
  Cells 4 and 14 are both **markdown prompt boxes**.
- Cell 6: "the pipeline in cell 4 adds the scaler". Cell 4 is markdown; the
  pipeline is built in **cell 7**, the very cell the box annotates.
- Cell 12: "Nothing in the code stops cell 14 reaching for the test set".
  Cell 14 is markdown; the threshold-choosing cells are **32** and **44**.

All four look like they were written against the source module's *code-cell*
numbering (`tools/notebooks/lecture_04.py`) and never re-counted after
markdown was interleaved.

**3 · `root_mean_squared_error` is the stated reason for the `sklearn >= 1.4`
assert and is never used.** *Checked by string search: the name occurs twice
in the notebook, both times inside the setup cell's comment and its prompt
box.* It is a regression metric; this is a classification notebook. Nothing
here needs anything close to 1.4 — every function called is a decade old — so
a student on a 1.3 Colab image is blocked by an assert citing a function the
notebook does not import. If a floor is wanted, 1.3 is the defensible one:
*checked in the installed docstring*, it is the last release that changed
`precision_recall_curve`'s signature (`drop_intermediate`, `versionadded 1.3`).

**4 · `roc_curve` and `cross_val_score` are imported and never called.**
*Checked by string search: one occurrence each, in the import cell.* Minor,
but the import cell is the one cell whose stated purpose is "every import
this notebook uses".

**5 · "91% of the accuracy is bought by specificity" (cell 22's box) does not
match what cell 23 prints.** Cell 23 prints
`100 * (1 - base_rate) * spec / lhs`. Using only figures the notebook states
in its own prose — accuracy 0.9690 (the §2 heading), 22.8% of 5s missed (cell
24), base rate 0.09035 — the implied specificity is 0.98857 and the printed
share is **92.8%**, not 91%. *Checked: computed from the notebook's own three
prose figures.* 91% is $1-p$, the weight; the box has confused the weight
with the share. §1.4 and §1.5.

**6 · "Accuracy would call that a 1.87-point improvement" (cell 41) is derivable
from nothing in the notebook.** *Checked: no cell computes an accuracy for the
forest — cell 40 prints ROC AUC, average precision and recall-at-90%-precision
only, and `accuracy_score` is never called on `f_scores`.* §1.2.

**7 · "90.39% recall on the cross-validated training scores" (cell 47) is printed
by no cell.** *Checked: cell 44 prints only the two thresholds; the training
recall achieved at `t_forest` is never printed.* The figure is arithmetically
self-consistent (0.9039 × 5,421 = 4,900 exactly), which is why it looks safe,
but the reader cannot reconcile it. §1.2.

**8 · A hard-coded number inside an f-string that is not an f-field.** Cell 32
ends with `print(" -> quote 2.12% as an illustration, not as an operating
point")`. The 2.12% is the recall at the 99%-precision point, which the same
cell computes two lines earlier as `recalls[idx99]`. If the number moves, the
sentence lies and nothing warns you. *Checked by reading the cell.* §1.1.

**9 · `row[0.0904]` in cell 36 survives on a rounding coincidence.** *Checked by
execution:* `round(np.float64(5421/60000), 4)` → `0.0904`, so the lookup works
**today**; Python's built-in `round(0.09035, 4)` on the same value as a plain
float → **`0.0903`**, so the same cell raises `KeyError: 0.0904` the moment
`base_rate` stops being a `np.float64` — `float(...)`, `.item()`, or a
round-trip through pandas all do it. A dict keyed on rounded floats and read
with a transcribed literal.

**10 · `assert (d[~drops_a_five] < 0).all()` in cell 15 fails if the top-ranked
instance is not a 5.** *Checked by execution: I forced a non-5 to the top of
a synthetic ranking of the real labels and the assert evaluated `False`.*
With $T = 0$ at the top of the list, accepting another negative leaves
precision at 0 — a flat step, not a fall. The prose in cell 11 covers this
("precision lies in $(0,1)$"); the code does not, and the assert message
("the lemma says every one of these rises") would send a reader hunting for a
bug in their sort. One extra `assert lab[0] == 1` fixes it.

**11 · "5,417 + 3 = 5,420" (cell 16) is transcribed; only the total is
structural.** *Checked by execution on the real labels with three different
synthetic rankings: the count of steps attributable to non-5s (what cell 15
prints as "precision rises") was **54,579** every time — forced by the labels
alone — while the split of the remaining 5,420 into falls and flats came out
5,184/236, 4,877/543 and 4,654/766.* So 54,579 and 5,420 are properties of the
data and are safe to write in prose; the split
into 5,417 and 3 is a property of one SGD run, and the sentence presents all
three the same way. §1.1.

**12 · `assert prec[4] < prec[5]` (cell 18) is a hard-coded fact about which digit
sits at rank 6.** It holds only if rank 5 is a non-5 and rank 6 is a 5.
*Checked by reasoning from the lemma, which is itself verified in item 11:*
the structural version — find the first index where precision falls and
assert the instance accepted there is a non-5 — cannot fail. The current
assert can, on a different scikit-learn version or a different seed, in a
cell whose whole purpose is a fact the reader can check by hand.

**13 · The unguarded `argmax` the notebook argues against is used by the notebook
eight cells later.** Cell 32 builds a `MIN_SUPPORT` guard and its box calls
the bare `(precisions >= 0.90).argmax()` "the usual student version". Cell 40
then computes the SGD's recall-at-90%-precision with exactly
`recalls[(precisions >= 0.90).argmax()]`, unguarded, and that number is the
one the prose calls "the row the brief is about". *Checked by reading both
cells.* §2.3 — the correction was not propagated.

**14 · The same two cells index arrays of different lengths for the same
question.** Cell 32 uses `precisions[:-1]`; cell 40 uses the unsliced
`precisions`. They agree here, but the notebook is teaching the reader that
the two lengths matter (cell 28's box) and then using both conventions
interchangeably. §4.1 in spirit.

**15 · `idx99` is bound twice to indices with different meanings.** Cell 32:
`idx99 = int((precisions[:-1] >= 0.99).argmax())` — a *precision*-based index
into the length-$T$ arrays. Cell 50: `idx99 = np.where(recalls >= 0.99)[0][-1]`
— a *recall*-based index into the length-$T{+}1$ array, taken from the other
end. Same name, same notebook, two different quantities, no comment. *Checked
by reading both cells.* Both are `int`, so §4.1's letter is not broken; its
purpose is.

**16 · Cell 26's "usual student version" describes what cell 44 does.** The box
warns against "reusing this fitted `clf` for an accuracy figure later — it has
seen everything; every score it produces is optimistic". Cell 44 then writes
`sgd_final = clf  # already fitted above` and reports test metrics from it.
Cell 44 is **correct** — fitted on train, measured on test is exactly right —
which makes cell 26's bullet simply wrong, and wrong in the direction that
teaches a student to distrust a valid procedure. *Checked by reading both
cells.* §6.2.

**17 · "the only threshold that reaches 90% recall flags 1,377 items" (cell 47).**
Every threshold below `t_sgd` also reaches 90% recall, and each flags more.
The intended claim — *no* threshold reaches 90% recall inside capacity — is
true and stronger; "the only threshold" is false as written. *Checked by
reading the code in cell 44, which takes `[0][-1]`, the highest such
threshold.*

**18 · The trap in §7 is announced three times before the cell (§8.1).** The
section heading carries "⚠ Read before running" and describes the outcome;
the prompt box's `constraint` slot repeats "this is the cell to read before
running"; and the box's third bullet gives the answer away in full —
"5,303 extra false alarms and an accuracy below the never-fires baseline" —
*before the reader has run the cell that produces those numbers.* This is the
exact defect §8.1 documents in lecture 19. *Checked by reading cells 48, 49,
50.* The repair is a reorder, not a rewrite: run cell 50 unannounced, have the
reader write the two recalls down, then open §9 with the ⚠.

**19 · Eighteen code cells, eighteen full three-bullet annotations (§6.1).**
*Checked by parsing: 18 code cells, 18 prompt boxes, 18 containing
"Left open".* The budget is five to eight, never more than ten. This notebook
is at 2.25× the ceiling, and its §7 defect — the one thing in it a reader must
not skim — sits at cell 49, well past where all three audit readers stopped
reading the template.

**20 · "examinable" appears twice, both about the setup cell (§8.3).** *Checked by
string search.* Cell 2's box and cell 3's comment both say the setup is not
examinable. No other section carries a mark, including §2, which is the
mathematical thread the lecture is named after.

**21 · Vocabulary undefined on first use (§7.5).** *Checked by reading.* "shift"
(a period of work at a desk — used 20+ times and never defined, and it collides
with the array operation of the same name that lecture 19 uses constantly);
"operating point"; "support"; "plateau"; "red-team"; "flatters".

**22 · This notebook and lecture 18 use the same words for opposite directions.**
Lecture 4 cell 15 prints `drop a non-5, precision rises` — it reads the
ranking bottom-up, raising the threshold. Lecture 18 cell 40 prints
`steps down (precision falls)` and asserts `down == n_fp` — it reads the
ranking top-down, accepting detections. **A non-5 is a "rise" in lecture 4 and
a "fall" in lecture 18, for the same step.** Lecture 18 explicitly tells the
reader it is doing "the way Lecture 4 did for MNIST". *Checked by reading
lecture 18 cells 38–40.* This script fixes it by reading top-down throughout.

**23 · The two lectures also use "average precision" for two different
estimators.** Lecture 4 cells 36 and 40 call
`sklearn.metrics.average_precision_score`, which is the **non-interpolated**
$\sum_n (R_n - R_{n-1})P_n$. Lecture 18 cell 43 defines its own
`average_precision` on the **enveloped** curve, and lecture 18 cell 41 says
"Lecture 4 proved precision has no monotone envelope" — pointing back at a
lecture that never mentions the envelope. *Checked by execution:* on the
5-detection ranking TP, FP, FP, TP, TP the two give **0.700** and **0.7333**.
They are different numbers with the same name across two lectures that
cross-reference each other.

**24 · Timing (§7.1): no CPU figure is given anywhere, and the SGD figure looks
optimistic.** §7.1 requires a CPU number beside every ⏱; none of the four
timing notes in the notebook (cells 11, 26/27, 38, 42) states a core count or
a machine. *Partially checked.* On the SGD fit that cell 27 calls "about 30
s", I measured **103 s** and **122 s** on two separate runs of
`make_pipeline(StandardScaler(), SGDClassifier(random_state=42)).fit(X_train,
y_train_5)` — but the laptop was at load average ~190 both times, so those are
upper bounds and I **cannot** claim the 30 s figure is wrong. What is
load-independent and does support the suspicion: `n_iter_` comes out at
**746**, i.e. the classifier makes 746 passes over 60,000 × 784 floats
single-threaded, and the `StandardScaler` step that people assume is the cost
takes **1.6 s** of it. The forest figures I could only extrapolate (measured
4-tree and 8-tree fits at `n_jobs=2` on 60,000 rows: 1.3 s and 2.3 s → ~30 s
at 100 trees, ~60 s for the 3-fold CV), which makes cell 38's "under a minute"
about right. Cell 42's "about 30 seconds for the two final fits" is right only
because one of the two — the SGD — is already fitted and is silently reused.

### Checked for and *not* found

- **§5.1 / §5.2 — markdown rendering.** *Checked by parsing every markdown cell
  line by line:* no line indented ≥4 spaces outside a fence, and no fence markers
  at all. Clean.
- **§3.1 — code quoted in markdown.** *Checked:* no ```` ```python ```` blocks in
  any markdown cell, so nothing to reconcile. Clean.
- **§4.2 — non-idempotent training.** *Checked by reading:* cells 27 and 44 both
  call `.fit()` on estimators constructed in earlier cells, but scikit-learn's
  `fit` discards prior state, so re-running either is safe. Not a defect here
  (this is a PyTorch hazard, not a scikit-learn one).
- **§2.1 — mismatched comparison windows.** *Checked by reading:* the SGD/forest
  comparison in cell 40 uses the same 60,000 rows and the same `cv` object for
  both models, and the test table in cell 46 scores all three operating points on
  the same 10,000 rows. Matched. The prose does not *say* which rows, which is the
  half of §2.1 it misses, but the comparison itself is sound.

### Claimed and **not verifiable** without executing the training cells

The brief forbids running them, so the following are unchecked. Each is
*arithmetically self-consistent*, which is the most I could establish:

- **1,377 flagged / 377 over capacity** (cell 47). Consistent by subtraction only.
- **811 flagged, 89.91% test recall** (cell 47). Consistent: 0.8991 × 892 = 802
  true positives exactly, leaving 9 false alarms, which matches "almost none of
  them are false alarms".
- **4,416 flagged at 90% precision; 116 at 99%, on a 21-threshold plateau;
  2.12% recall** (cell 31's box and cell 32). Consistent: 116 flagged at ≥99%
  precision implies 115 true positives, and 115 / 5,421 = 0.0212.
- **"5,303 extra false alarms", "51 points of accuracy", "six times the desk
  capacity"** (cells 49 and 54). Mutually consistent: 5,303 extra false alarms
  with 203 fewer misses gives exactly a 0.5100 accuracy drop, and the implied
  flagged count is ~6,300, i.e. 6.3× the 1,000 capacity. All three fit one
  scenario, so if one is wrong all three are.
- **"The previous lecture's failure cost 0.52 points"** (cell 54). *Checked
  against `notebooks/lecture-03.ipynb`: the string "0.52" appears nowhere in it,
  and its own prose calls the same quantity "Half a point".* The figure comes
  from an unstored output of lecture 3 cell 49, so a reader cannot verify a
  cross-lecture claim from either notebook. Not a wrong number, but an
  unreconcilable one — §1.2 across lectures.
- **"a 1.87-point improvement" and "the last row moves by 24 points"** (cell 41).
  The 24 is derivable from cell 40's printed row by one subtraction the sentence
  names, so it is admissible under §1.2. The 1.87 is not derivable from anything
  — see item 6.
