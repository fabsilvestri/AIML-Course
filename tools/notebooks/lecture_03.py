#!/usr/bin/env python3
"""
Lecture 3 — Classification and its metrics. MNIST, Géron Chapter 3.

Exports build() -> list[cell]; tools/make_notebooks.py wraps it.

Merges the old lectures 3 and 4, which built a detector and then diagnosed its
metric. Here the structure is the lecture's: the task and a first detector, the
two results the metrics rest on derived and then verified numerically, the
confusion matrix and the ratios, the curves, and an operating point chosen
against a stated constraint.

Every structural step is followed by an assertion, and anything slower than
about twenty seconds states its wall clock, because "no output" otherwise reads
as "it hung". Runs on CPU throughout.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from make_notebooks import code, header, md, SETUP, SETUP_PROMPT        # noqa: E402
from _prompt import prompt                                # noqa: E402

MNIST_LOADER = code('''
# --- the data ----------------------------------------------------------------
# ~30 s the first time (11 MB from openml), then cached by scikit-learn.
from sklearn.datasets import fetch_openml

mnist = fetch_openml("mnist_784", as_frame=False)
X, y = mnist.data, mnist.target

print(f"X {X.shape} {X.dtype}")
print(f"y {y.shape} {y.dtype}")          # note the dtype of y

assert X.shape == (70000, 784), f"unexpected shape {X.shape}"
''')

def build() -> list:
    cells = header(3, "Classification and its metrics", "", "Chapter 3")

    cells += [
        md("""
One number runs through this notebook: **96.90%**. It is correctly
cross-validated, there is no leakage in it, and its three folds agree to two
tenths of a point. It is also worth almost nothing, and the two results derived
in section 6 are what let you say why.

Runs on free CPU. The slowest cell is about 70 seconds and says so.
"""),
        md("## 1 · Setup"), SETUP_PROMPT, SETUP,
        prompt(
            label="every import, in one place",
            input="nothing",
            output="every name this notebook uses below, imported once",
            constraint="no import anywhere after this cell, so the notebook "
                       "does not depend on a previous one still being in memory",
            check="Runtime -> Restart, then run this cell alone. If anything "
                  "below raises NameError, that import belongs here",
            **{"try": "restart the runtime and run the LAST cell first. It "
                      "fails. A notebook that only runs top to bottom is the "
                      "only kind you can trust."}),
        code('''
# Every name used below, imported once. Later cells repeat the ones that are
# worth seeing beside the code that uses them; Python does not mind.
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, clone
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (accuracy_score, average_precision_score,
                             confusion_matrix, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score,
                             roc_curve)
from sklearn.model_selection import (StratifiedKFold, cross_val_predict,
                                     cross_val_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

print("imports ok")
'''),
        md("""
## 2 · The brief

A postal operator scans handwritten postcode digits. One glyph is misread more
often than the others, and the audit team wants every scanned digit that **is a
5** pulled off the line and sent to a human verification desk.

The desk can re-check about **1,000 items per shift**, and a shift is about
10,000 scanned digits. That figure was given to us; we did not measure it, and
everything we derive from it inherits that status.

Not *read the digit*. **Detect one digit.**
"""),

        md("## 3 · The data"),
        prompt(
            label="the data",
            input="MNIST from OpenML",
            output="X and y, with their shapes and dtypes printed",
            constraint="print the dtype of y as well as its shape — it is not "
                       "what you expect, and the next cell is about that",
            check="assert the shape is (70000, 784) rather than trusting the  download. 784 = 28 x 28. If the second dimension is not a perfect  square you are not holding images, whatever the variable  is called.",
            **{"try": "`as_frame=True`. You get a DataFrame, `X.shape` still works, and every downstream cell that indexes with `X[idx]` breaks. Which is better here, and why?"}),
        MNIST_LOADER,

        md("""
`y` came back as an array of **one-character strings**. This is the first
default nobody asked for, and it is the expensive kind: `y == 5` is `False` for
every image, silently, because a string is never equal to an integer.
"""),
        prompt(
            label="the dtype that silently finds nothing",
            input="y as OpenML delivers it",
            output="what `y == 5` matches before and after casting to uint8",
            constraint="show the count BEFORE the cast — the point is that it is zero and raises nothing",
            check="after the cast, the ten digits 0-9 are all present. Count your positives immediately after building a boolean label, and compare with what you expect. Zero is a number the code will not complain about.",
            **{"try": "delete the cast and re-run the next cell. The label count goes to zero and nothing raises — that is the failure this cell exists to prevent."}),
        code('''
print("before:", y.dtype, repr(y[0]))
print("y == 5 finds", (y == 5).sum(), "images")     # zero!

y = y.astype(np.uint8)

print("after: ", y.dtype, repr(y[0]))
assert y.dtype == np.uint8
assert set(np.unique(y)) == set(range(10))
'''),
        prompt(
            label="look at it",
            input="the first sixteen images and their labels",
            output="a 2x8 grid, each titled with its label",
            constraint="reshape to 28 x 28 — the rows are flat vectors, and imshow of a 784-long vector is not an error, it is a stripe",
            check="read the titles against the pictures. You know what a 5 looks like; that is the entire test and it takes four seconds.",
            **{"try": "plot ten random indices instead of the first ten. Do you still recognise every digit? The ones you cannot are the ones the model will miss."}),
        code('''
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 8, figsize=(10, 3))
for ax, i in zip(axes.ravel(), range(16)):
    ax.imshow(X[i].reshape(28, 28), cmap="binary")
    ax.set_title(str(y[i]))
    ax.axis("off")
plt.tight_layout(); plt.show()
'''),
        md("""
A 784-vector and a 28 × 28 image are the same object in two shapes. The model
only ever sees the 784 numbers, in a fixed order, and has no idea that pixel 0
and pixel 28 are vertical neighbours.
"""),

        md("""
## 4 · The split, which is already made

MNIST arrives partitioned: the first 60,000 rows are the standard training set
and the last 10,000 the standard test set. The corpus is already shuffled, so
the first 60,000 are not sorted by digit or by writer.

Using the standard split is what makes our number comparable to anybody
else's.
"""),
        prompt(
            label="the split that already exists",
            input="the 70,000 rows",
            output="60,000 train and 10,000 test, by position",
            constraint="do NOT call train_test_split — MNIST arrives shuffled and pre-partitioned, and every published result uses this exact cut",
            check="assert the two sizes and that pixels are still 0-255. The pixel range assert. If scaling has crept in above this line, min/max stop being 0/255 and the split is no longer the raw one.",
            **{"try": "shuffle the 70,000 rows and split 60/10 yourself. The accuracy barely moves here — but say why the given split is still the one to use."}),
        code('''
X_train, X_test = X[:60000], X[60000:]
y_train, y_test = y[:60000], y[60000:]

assert len(X_train) == 60000 and len(X_test) == 10000
assert X_train.shape[1] == X_test.shape[1] == 784
assert X_train.min() == 0 and X_train.max() == 255

# From here to the very last cell of the NEXT notebook, X_test is not touched.
print(f"train {len(X_train):,}   test {len(X_test):,}   features {X_train.shape[1]}")
'''),

        md("""
## 5 · From ten classes to two

Two lines, and the imbalance is created. Nothing about MNIST is imbalanced —
**our question is.**
"""),
        prompt(
            label="one class against the rest",
            input="the ten-class labels",
            output="boolean is-it-a-5 labels, with the base rate printed",
            constraint="print the base rate before anything is fitted, so the yardstick exists before there is a result to compare it with",
            check="assert 5,421 positives — which fails loudly if the uint8 cast was skipped. The assert is doing double duty: it checks the positive count and, because a string-vs-int comparison gives zero, it also catches the missing cast two cells up.",
            **{"try": "detect a 1 instead of a 5. The base rate changes; note the new value, because every metric today is read against it."}),
        code('''
y_train_5 = (y_train == 5)
y_test_5  = (y_test  == 5)

assert y_train_5.dtype == bool
assert y_train_5.sum() == 5421, "did the cast to uint8 happen?"

n_pos = int(y_train_5.sum())
n_neg = int((~y_train_5).sum())
base_rate = y_train_5.mean()

print(f"positives {n_pos:,}   negatives {n_neg:,}")
print(f"base rate {base_rate:.5f}  ({100 * base_rate:.2f}%)")
'''),
        prompt(
            label="balanced data, unbalanced task",
            input="the ten-class labels",
            output="how many of each digit, as a bar per class",
            constraint="show all ten, not a summary statistic — the shape of the distribution is the point",
            check="count the positives of the task you are actually solving, never the classes of the dataset you started from.",
            **{"try": "count the positives for each of the ten digits. The rarest and the commonest differ by about 20% — enough to move the anchor by a point."}),
        code('''
counts = np.bincount(y_train, minlength=10)
for d, c in enumerate(counts):
    bar = "#" * (c // 150)
    print(f"{d}  {c:>6,}  {bar}")
print("\\nThe ten classes are balanced. The task we were given is not.")
'''),

        md("""
## 6 · The metric

Classification, so RMSE is gone. The obvious replacement is **accuracy**: the
fraction of instances the classifier gets right.

$$\\text{accuracy} = \\frac{\\#\\{i : \\hat y^{(i)} = y^{(i)}\\}}{m}$$

Name it, out loud, before you have anything to measure. A metric you did not
name is a metric you did not choose.
"""),

        md("""
## 7 · The anchor — what does *nothing* score?

Rule 2 of this course: a metric with nothing to compare it to is decoration.

So before building anything, measure the cheapest possible detector. It looks
at nothing and answers "not a 5" every time.
"""),
        prompt(
            label="the anchor",
            input="nothing but the labels",
            output="the cross-validated accuracy of a detector that always says no",
            constraint="a real estimator run through the real cross-validation, not 1 - base_rate computed by hand — it must be measured the same way the model will be",
            check="its accuracy should equal 1 minus the base rate, to three places. Anchor and 1 - base_rate must agree. If they do not, the fault is in the splitter or the scorer, not in the estimator that does nothing.",
            **{"try": "score the never-fires classifier with `scoring='recall'` instead. It returns 0.0. Which of the two numbers would a client rather be shown?"}),
        code('''
from sklearn.base import BaseEstimator
from sklearn.model_selection import StratifiedKFold, cross_val_score

class NeverFires(BaseEstimator):
    """The dumbest possible detector. It is also 90.96% accurate."""
    def fit(self, X, y=None):
        return self
    def predict(self, X):
        return np.zeros(len(X), dtype=bool)

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

anchor = cross_val_score(NeverFires(), X_train, y_train_5,
                         cv=cv, scoring="accuracy")
print("per fold:", anchor.round(5))
print(f"anchor accuracy {anchor.mean():.5f}")

# it is an identity, not an estimate: accuracy = 1 - base rate, exactly
assert np.isclose(anchor.mean(), 1 - base_rate, atol=1e-4)
'''),
        md("""
**90.96%, with no model, no fit and no features.** So 90.96% is *zero* in the
units that matter, and the only interesting quantity today is the distance
above it.
"""),

        md("""
## 8 · Build the simplest thing that runs

`SGDClassifier` fits a linear model one instance at a time, so it never needs
the whole training set in memory and it handles 60,000 × 784 comfortably.

The standing constraint from the previous lecture applies unchanged: the scaler
goes **inside** a pipeline, so that cross-validation refits it per fold and
leakage is structurally impossible rather than merely avoided.
"""),
        prompt(
            label="the detector",
            input="the 60,000 training rows",
            output="a scaler and an SGD classifier as ONE pipeline object, fitted",
            constraint="scaler inside the pipeline, never a separate fit_transform — the standing constraint from Lecture 2",
            check="assert the coefficient shape is (1, 784), one weight per pixel. (1, 784) is one row of weights over 784 pixels. A shape of (10, 784) means you fitted the ten-class problem by accident.",
            **{"try": "drop the `StandardScaler` from the pipeline. Accuracy falls by about two points — pixels are already on a common 0-255 scale, so why does scaling still help an SGD?"}),
        code('''
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

clf = make_pipeline(StandardScaler(),
                    SGDClassifier(random_state=RANDOM_STATE))

# ⏱ about 30 s: one pass fits the scaler and then the classifier on 60,000 rows.
clf.fit(X_train, y_train_5)

assert clf.named_steps["sgdclassifier"].coef_.shape == (1, 784)
print("fitted:", clf)
'''),
        md("""
### Test it against a case whose answer is known

Step 4 of the working method, and the cheapest twenty seconds in the lecture.
`X_train[0]` is the 5 we plotted above.
"""),
        prompt(
            label="two predictions, and what not to assert",
            input="one known 5 and one known non-5",
            output="the model's prediction for each",
            constraint="assert the SHAPE of the output, never that either prediction is correct",
            check="ask what a broken model would score on your assertion. If it passes, the assertion is decoration.",
            **{"try": "assert the model predicts True for the first 5 in the training set. It passes, and it would pass for a model that predicts True for everything."}),
        code('''
# one known 5 and one known not-5, so the two cases are visible
i_pos = int(np.argmax(y_train))          # first 5
i_neg = int(np.argmin(y_train))          # first non-5

for i in (i_pos, i_neg):
    print(f"index {i:>5}   true: {bool(y_train[i])!s:<5}   "
          f"predicted: {bool(clf.predict([X_train[i]])[0])}")

# Assert the SHAPE, not the answer. A single correct prediction is not evidence
# that the detector works — that is the whole argument of this lecture, and an
# assertion that the model got one example right would contradict it.
assert clf.predict(X_train[:8]).shape == (8,)
assert clf.predict(X_train[:8]).dtype == bool
'''),

        md("""
## 9 · Measure it honestly

⏱ **about 70 seconds** — three complete refits of scaler and classifier.
"""),
        prompt(
            label="cross-validated, against the anchor",
            input="the pipeline, the training rows, three stratified folds",
            output="per-fold accuracy, the mean, and the gap over the never-fires anchor",
            constraint="print the anchor beside it — a lone accuracy is not a result",
            check="the gap in percentage points, stated explicitly rather than left for the reader to subtract. Always subtract. An accuracy without its anchor beside it cannot be read, and the subtraction is what makes 96.9% shrink to 5.9.",
            **{"try": "compute the same subtraction for the digit-1 detector. The accuracy is higher and the margin over its anchor is smaller."}),
        code('''
scores = cross_val_score(clf, X_train, y_train_5, cv=cv,
                         scoring="accuracy", n_jobs=-1)

print("per fold:", scores.round(5))
print(f"mean     {scores.mean():.5f}")
print(f"anchor   {anchor.mean():.5f}")
print(f"gap      {100 * (scores.mean() - anchor.mean()):.2f} percentage points")
'''),
        md("""
Report **every fold**, not just the mean. Here they agree to two tenths of a
point — which is a finding, and one you only have because you printed them.

To see why it is a finding, drop the pipeline and run the same classifier on
raw pixels. ⏱ **about 30 seconds.**
"""),
        prompt(
            label="what the scaler is worth",
            input="the same rows and folds, without the scaler",
            output="scaled and unscaled fold accuracies side by side",
            constraint="change ONE thing — same estimator, same seed, same splits",
            check="compare the WORST unscaled fold with the anchor, not the mean. The worst unscaled fold against the never-fires anchor. If a fold sits near the anchor, that configuration failed completely on that split and the mean is covering for it.",
            **{"try": "raise `max_iter` on the unscaled SGD until it converges. Some of the gap closes; the rest is the scaler doing real work."}),
        code('''
raw = cross_val_score(SGDClassifier(random_state=RANDOM_STATE),
                      X_train, y_train_5, cv=cv, scoring="accuracy", n_jobs=-1)

print("scaled  ", scores.round(5), f"mean {scores.mean():.5f}")
print("unscaled", raw.round(5),    f"mean {raw.mean():.5f}")
print(f"\\nworst unscaled fold: {raw.min():.5f} — "
      f"{100 * (raw.min() - anchor.mean()):.2f} points above doing nothing")
'''),
        md("""
The two means differ by under two points. **The spreads do not compare at
all**: one fold of the unscaled version lands five points below its own
siblings, and reporting only the mean would have hidden that completely.
"""),

        md("""
## 10 · What `cross_val_score` is doing

Write this loop once in your life. After that, use the library — but you will
know what it did.

`clone` copies the hyperparameters and discards anything learned. Reusing `clf`
here would train the same object three times in succession, each fold starting
from the previous fold's parameters.

⏱ **about 70 seconds** — the same three fits, done by hand.
"""),
        prompt(
            label="cross-validation, by hand",
            input="the same folds",
            output="the same three accuracies, computed with an explicit loop",
            constraint="`clone` the estimator each fold — reusing a fitted one carries the previous fold's parameters into the next",
            check="the hand-rolled numbers must match cross_val_score's. If your by-hand numbers do not reproduce the library's, one of the two is wrong and it is worth ten minutes to find out which.",
            **{"try": "pass `clf` itself instead of `clone(clf)`. The three accuracies rise, nothing errors, and the folds have quietly stopped meaning anything."}),
        code('''
from sklearn.base import clone

by_hand = []
for train_idx, test_idx in cv.split(X_train, y_train_5):
    fold_clf = clone(clf)                                  # an unfitted copy
    fold_clf.fit(X_train[train_idx], y_train_5[train_idx])

    pred = fold_clf.predict(X_train[test_idx])
    by_hand.append((pred == y_train_5[test_idx]).mean())

by_hand = np.array(by_hand)
print("by hand:", by_hand.round(5))
print("library:", scores.round(5))
assert np.allclose(by_hand, scores), "the library is doing something else"
'''),

        md("""
## 11 · Where 96.90% came from

Write $P$ for the number of positives, $N$ for the negatives, $m = P + N$, and
$p = P/m$ for the **base rate**. Split the four confusion counts by their true
class:

$$\\text{accuracy} = \\frac{\\mathrm{TP} + \\mathrm{TN}}{m}
 = \\frac{P}{m}\\cdot\\frac{\\mathrm{TP}}{P} + \\frac{N}{m}\\cdot\\frac{\\mathrm{TN}}{N}
 = p\\,\\text{recall} + (1-p)\\,\\text{specificity}$$

Accuracy is a **weighted average of two rates**, weighted by the class sizes.
Two consequences follow immediately, and both are identities rather than
estimates:

- A classifier that never fires has recall $0$ and specificity $1$, so it scores
  exactly $1 - p$.
- $\\partial\\,\\text{accuracy} / \\partial\\,\\text{recall} = p$. At our base
  rate, ten points of recall move accuracy by 0.90 points.

Verify the first, rather than believing it:
"""),
        prompt(
            label="what accuracy is worth here",
            input="a list of base rates",
            output="the accuracy of a model that never fires, at each",
            constraint="print it before any model is fitted, so the number is not a reaction to a result",
            check="ask what accuracy the empty model gets before you ask what yours gets. If you cannot beat it by a margin you would defend, you have no result.",
            **{"try": "work out what the never-fires classifier scores on a problem with a 1-in-10,000 base rate. That is the number a fraud team is up against."}),
        code('''
for p in (0.5, 0.09035, 0.01, 0.001):
    print(f"base rate {p:>7.5f}  ->  never-fires accuracy {1 - p:.5f}")

print("\\nFor any target accuracy below 1, there is a problem rare enough")
print("that the empty model beats it.")
'''),

        md("""
### Part two — what happens when you turn the dial

Every scoring classifier is a *family* of classifiers, one per threshold $t$:
predict positive when $s(x) \\ge t$. Raising $t$ shrinks the flagged set, and it
shrinks it by **losing** instances, never gaining any. So $\\mathrm{TP}(t)$ and
$\\mathrm{FP}(t)$ are both non-increasing.

**Recall is monotone.** Its denominator is $\\mathrm{TP}(t) + \\mathrm{FN}(t) =
P$, which counts every positive in the data whatever we predict. So recall is a
non-increasing quantity divided by a constant.

**Precision is not.** Its denominator is the flagged count, which we choose.

*The one-step lemma.* Raise $t$ just enough to drop the lowest-scoring flagged
instance. With $T = \\mathrm{TP}$, $n = T + \\mathrm{FP}$ before the step and
$y \\in \\{0,1\\}$ the label of the instance dropped:

$$\\frac{T-y}{n-1} > \\frac{T}{n} \\iff n(T-y) > T(n-1) \\iff y < \\frac{T}{n}$$

Since $y$ is 0 or 1 and precision lies in $(0,1)$: dropping a **negative** raises
precision, dropping a **positive** lowers it.

Now count it, on sixty thousand instances. First we need the scores.

⏱ **about 60 seconds** — three refits, asking for the decision function rather
than the labels.
"""),
        prompt(
            label="out-of-fold scores",
            input="the scaled SGD pipeline and the 60,000 training rows",
            output="one decision-function value per training instance, from a model that never saw it",
            constraint="`method='decision_function'`, not `predict` — one call then gives both the labels and the entire threshold sweep",
            check="the array is 60,000 long, and thresholding at 0 reproduces predict(). `(y_scores >= 0)` must equal what `predict` would return. SGD predicts positive exactly when the decision function is non-negative, so if those disagree you have the wrong method or the wrong estimator.",
            **{"try": "use `method='predict'` instead. You get labels, not scores, and every curve below becomes impossible to draw."}),
        code('''
y_scores = cross_val_predict(clf, X_train, y_train_5, cv=cv,
                             method="decision_function", n_jobs=-1)

assert y_scores.shape == (60000,)
# SGDClassifier predicts positive exactly when the decision function is >= 0,
# so one call gives us both the labels and the whole threshold sweep.
y_pred = (y_scores >= 0)
print(f"accuracy {accuracy_score(y_train_5, y_pred):.5f}")
'''),
        prompt(
            label="is precision monotone?",
            input="the out-of-fold scores and their labels",
            output="how often precision FALLS as the threshold rises, counted",
            constraint="walk the ranking one instance at a time rather than sampling a grid of thresholds — a grid can step over every fall there is",
            check="the two counts must sum to the number of steps. You should find thousands of falls, not zero and not a handful — one for essentially every 5 in the data. If you find none, you are sampling rather than walking.",
            **{"try": "count the rises as well as the falls. They should sum to 59,999, and neither is small."}),
        code('''
# Walk the threshold down the ranking, one instance at a time.
order = np.argsort(-y_scores, kind="stable")
lab   = y_train_5[order].astype(np.int64)
tp    = np.cumsum(lab)
n     = np.arange(1, len(lab) + 1)
prec  = tp / n

# step k -> k+1 lowers the threshold; read backwards, d > 0 is a FALL
d = np.diff(prec)
drops_a_five = (lab[1:] == 1)

print(f"steps in total                     {len(d):,}")
print(f"drop a non-5, precision rises      {(d < 0).sum():,}")
print(f"drop a 5,     precision falls      {(d > 0).sum():,}")
print(f"drop a 5,     precision already 1  {((d == 0) & drops_a_five).sum():,}")

assert (d[~drops_a_five] < 0).all(), "the lemma says every one of these rises"
assert (d[drops_a_five] >= 0).all(), "and every one of these falls or is flat"
print("\\nEvery one of the steps is accounted for by the lemma.")
'''),
        md("""
54,579 is the number of non-5s in the training set, and 5,417 + 3 = 5,420 is the
number of 5s less the one at the very top of the ranking, which has no step above
it. This is not a tendency: it is an exact classification of all 59,999 steps.

The textbook's own counterexample is the same thing at the top of the list.
"""),
        prompt(
            label="the smallest possible example",
            input="the top eight of the ranking",
            output="precision at each of the first eight cut-offs",
            constraint="show the arithmetic, not a plot — the point is that 4/5 < 5/6",
            check="assert precision at k=5 is below precision at k=6. Do the division yourself. 5/6 = 0.833, 4/5 = 0.800. Raising the threshold past a real 5 lost you precision, which the word 'threshold' does not prepare you for.",
            **{"try": "extend the list by one more negative at the bottom. Neither precision nor recall moves — dropping an instance you never flagged changes nothing."}),
        code('''
for k in range(1, 9):
    print(f"top-{k}: {tp[k-1]}/{k} = {prec[k-1]:.4f}")

print("\\nRaising the threshold past the 6th-ranked digit — a 5 — takes")
print(f"precision from {prec[5]:.4f} (5/6) down to {prec[4]:.4f} (4/5).")
assert prec[4] < prec[5]
'''),

        md("""
## 12 · What the number was hiding

Nothing new gets fitted here. We ask the *same* out-of-fold predictions a
different question.
"""),
        prompt(
            label="the four numbers",
            input="true labels and predictions at threshold 0",
            output="the confusion matrix, and its four cells named",
            constraint="print the raw matrix as well as the named cells — scikit-learn's cell order is not the one most textbooks draw",
            check="the four cells sum to 60,000. Tn should be the largest cell by far — about 54,000 of 60,000 rows are not 5s. If your 'true negatives' is a small number, your unpacking is the wrong way round.",
            **{"try": "swap the model for the never-fires classifier and print its matrix. Two cells are zero, and accuracy is still 90.96%."}),
        code('''
cm = confusion_matrix(y_train_5, y_pred)
tn, fp, fn, tp_ = cm.ravel()
print(cm)
print()
print(f"true negatives  {tn:>7,}     false positives {fp:>7,}")
print(f"false negatives {fn:>7,}     true positives  {tp_:>7,}")

assert tn + fp + fn + tp_ == 60000
'''),
        prompt(
            label="the identity behind the accuracy",
            input="the same predictions",
            output="accuracy, precision, recall, F1 and specificity, plus the identity accuracy = p*recall + (1-p)*specificity evaluated on our own numbers",
            constraint="verify the identity numerically rather than asserting it in prose",
            check="the two sides agree to floating-point tolerance. Compute the right-hand side yourself and watch it land on the accuracy you already printed. If prose and arithmetic disagree, the prose is the bug.",
            **{"try": "recompute the identity for the digit-1 detector. Both terms change; check that it still closes."}),
        code('''
precision = precision_score(y_train_5, y_pred)
recall    = recall_score(y_train_5, y_pred)
spec      = tn / (tn + fp)

print(f"accuracy    {accuracy_score(y_train_5, y_pred):.5f}")
print(f"precision   {precision:.5f}   {tp_:,} / {tp_ + fp:,}")
print(f"recall      {recall:.5f}   {tp_:,} / {tp_ + fn:,}")
print(f"F1          {f1_score(y_train_5, y_pred):.5f}")
print(f"specificity {spec:.5f}")

# the identity, on our own numbers
lhs = accuracy_score(y_train_5, y_pred)
rhs = base_rate * recall + (1 - base_rate) * spec
print(f"\\np.recall + (1-p).specificity = {rhs:.5f}   vs accuracy {lhs:.5f}")
assert np.isclose(lhs, rhs)

print(f"\\nthe specificity term alone is {100 * (1 - base_rate) * spec / lhs:.1f}%"
      f" of the headline number")
print(f"and {fn:,} fives — {100 * fn / (tp_ + fn):.1f}% of them — went past unflagged")
'''),
        md("""
Two sentences about the same model, both true:

- *"The detector is 96.9% accurate."* — signed off, deployed.
- *"The detector misses 22.8% of the 5s."* — a report the audit team can act on.

Accuracy was not *wrong*. It correctly answered a question nobody in this brief
had asked, it weights the class we care about by 0.09, and it adds two errors
that the client prices completely differently.

**And nobody chose it.** It arrived as a default, from a prompt that did not name
a metric.
"""),

        md("""
## 13 · The threshold is a dial

`predict()` hides the score and hard-codes the threshold at zero. There is no
`set_threshold()` method, and there should not be: you compute the scores and
compare them yourself.
"""),
        prompt(
            label="one digit, two thresholds",
            input="a single training image known to be a 5",
            output="its decision-function score, and the prediction at two thresholds",
            constraint="⏱ this refits on all 60,000 rows, about 30 seconds",
            check="ask of any score in this notebook: was the model that produced it fitted on this row? Here the answer is yes, and it is only used to demonstrate a mechanism.",
            **{"try": "pick a threshold between the two. Precision and recall both land between the printed pairs — the curve is continuous even where it is not monotone."}),
        code('''
some_digit = X_train[0]                       # a 5
clf.fit(X_train, y_train_5)                   # ⏱ about 30 s
score = clf.decision_function([some_digit])
print("score:", score.round(1))

for t in (0, 3000):
    print(f"threshold {t:>5}: predicted {bool(score[0] >= t)}")
'''),
        prompt(
            label="the two curves",
            input="the out-of-fold scores",
            output="precision and recall against threshold, and against each other",
            constraint="plot precision-recall rather than ROC — at a 9% base rate ROC flatters every model, which is the next section",
            check="assert precisions and recalls are exactly one longer than thresholds. The assert. It is three symbols and it pins a convention that has changed between library versions.",
            **{"try": "read the precision at 99% recall off the curve. It is far below the value at 90%, which is the trade-off stated as a number."}),
        code('''
precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)

# precisions and recalls have ONE MORE element than thresholds: the degenerate
# point where nothing is flagged and precision is defined to be 1.
assert len(precisions) == len(recalls) == len(thresholds) + 1

fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 4))
ax.plot(thresholds, precisions[:-1], label="precision")
ax.plot(thresholds, recalls[:-1], label="recall")
ax.set_xlim(-400, 400); ax.set_xlabel("threshold"); ax.legend(); ax.grid(alpha=.3)
ax2.plot(recalls, precisions)
ax2.set_xlabel("recall"); ax2.set_ylabel("precision"); ax2.grid(alpha=.3)
plt.tight_layout(); plt.show()
'''),
        md("""
### Choose a threshold deliberately

`argmax` on a boolean array returns the first `True`. The idiom is compact and
not obvious; read it once and remember it.
"""),
        prompt(
            label="the 90%-precision operating point",
            input="the precision-recall curve",
            output="the lowest threshold reaching 90% precision, with enough flagged digits behind it to be a claim rather than a coincidence",
            constraint="the crossing must hold with at least MIN_SUPPORT instances flagged; report whether that guard changed the answer",
            check="compare against the unguarded first crossing and print which happened. The same guard applied at 99% precision FAILS — only 116 digits are flagged there, on a plateau 21 thresholds wide. A guard that never fires anywhere is not evidence that nothing is wrong.",
            **{"try": "ask for 99% precision instead. Only 116 digits are flagged, on a plateau 21 thresholds wide — the guard that protects the 90% choice does not fire here."}),
        code('''
# Section 11 proved this curve is NOT monotone. So "the
# first index reaching 90%" can in principle be one lucky step held up by a
# handful of flagged digits — a real operating point needs support behind it.
MIN_SUPPORT = 500

n_pos   = int(y_train_5.sum())
flagged = recalls[:-1] * n_pos / precisions[:-1]     # tp / precision = tp + fp
ok      = (precisions[:-1] >= 0.90) & (flagged >= MIN_SUPPORT)
assert ok.any(), "no threshold reaches 90% precision with enough support"

idx = int(np.argmax(ok))
threshold_90 = thresholds[idx]

# Did the guard actually change anything? Say so either way.
naive = int((precisions[:-1] >= 0.90).argmax())
print(f"first crossing        idx {naive}")
print(f"first with support    idx {idx}   "
      f"({'same point — the check held' if naive == idx else 'DIFFERENT point'})")
print(f"digits flagged there  {flagged[idx]:,.0f}")

y_pred_90 = (y_scores >= threshold_90)
print(f"threshold {threshold_90:.2f}")
print(f"precision {precision_score(y_train_5, y_pred_90):.4f}")
print(f"recall    {recall_score(y_train_5, y_pred_90):.4f}")
print(f"\\n90% precision costs "
      f"{100 * (recall - recall_score(y_train_5, y_pred_90)):.2f} points of recall")

# and the other end of the dial, to show what "high precision" really buys.
# Apply the same support test — and watch it fail.
idx99 = int((precisions[:-1] >= 0.99).argmax())
print(f"\\nat 99% precision, recall is {recalls[idx99]:.4f} — one 5 in fifty")
print(f"   but only {flagged[idx99]:,.0f} digits are flagged there, on a "
      f"plateau {int((precisions[:-1] >= 0.99).sum())} thresholds wide")
print(f"   (the 90% point rests on {flagged[idx]:,.0f} digits over "
      f"{int((precisions[:-1] >= 0.90).sum())})")
print("   -> quote 2.12% as an illustration, not as an operating point")
'''),
        md("""
**A threshold is a hyperparameter.** Choosing it by looking at test performance
is the same error as choosing $\\alpha$ that way, and produces the same
optimistic bias. Choose it on cross-validated *training* scores, then measure
once.
"""),

        md("""
### PR or ROC?

The ROC curve plots recall against the false positive rate. Both denominators
are fixed by the data, which is why it behaves so much better than the PR curve
— and that is a warning, not a recommendation.

The rule: **prefer the PR curve when the positive class is rare.** Do not take
it on trust. Take our scores unchanged and thin the positive class, so that the
model, the ranking and the scores are identical and only the balance moves.
"""),
        prompt(
            label="what rarity does to the metrics",
            input="the same scores, with positives subsampled to three base rates",
            output="ROC AUC and average precision at each",
            constraint="subsample the POSITIVES only, leaving every negative in place, so the ranking is untouched and only the prevalence changes",
            check="ROC AUC should barely move; average precision should collapse. ROC AUC moving as much as average precision would mean you resampled the negatives too, and the ranking changed underneath you.",
            **{"try": "thin the positives further, to one in a thousand. Average precision keeps falling; ROC AUC barely moves."}),
        code('''
rng = np.random.default_rng(RANDOM_STATE)
pos, neg = np.where(y_train_5)[0], np.where(~y_train_5)[0]

print(f"{'base rate':>10}  {'positives':>9}  {'ROC AUC':>8}  {'avg prec':>8}")
row = {}
for target in (base_rate, 0.02, 0.01):
    k = int(round(len(neg) * target / (1 - target)))
    keep = np.concatenate([neg, rng.choice(pos, k, replace=False)])
    auc = roc_auc_score(y_train_5[keep], y_scores[keep])
    ap  = average_precision_score(y_train_5[keep], y_scores[keep])
    row[round(target, 4)] = (auc, ap)
    print(f"{y_train_5[keep].mean():>10.4f}  {k:>9,}  {auc:>8.4f}  {ap:>8.4f}")

(auc_hi, ap_hi), (auc_lo, ap_lo) = row[0.0904], row[0.01]
print(f"\\nROC AUC moves by {abs(auc_hi - auc_lo):.4f}")
print(f"average precision falls by {ap_hi - ap_lo:.4f}")
assert abs(auc_hi - auc_lo) < 0.01 < (ap_hi - ap_lo)
'''),
        md("""
The ROC curve cannot see the problem you have. That is the whole reason to learn
two curves rather than one.
"""),

        md("""
## 14 · A better classifier

A forest has no `decision_function`. It has `predict_proba`, which returns one
column per class; the second column — the estimated probability of the positive
class — plays exactly the role of the score, and every curve above works
unchanged.

⏱ **under a minute.**
"""),
        prompt(
            label="a second model, scored the same way",
            input="the same 60,000 rows and the same folds",
            output="out-of-fold probabilities for a random forest, and its curves beside the SGD's",
            constraint="`predict_proba` gives an (n, 2) array — take column 1, the POSITIVE class; column 0 is the exact complement and ranks everything backwards",
            check="assert the shape is (60000, 2) before indexing it. An AUC below 0.5 on a model that trains normally is almost always the wrong column or an inverted label, not a genuine result.",
            **{"try": "compare the two models by ROC AUC alone. They look close. Now compare them by average precision."}),
        code('''
forest = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE,
                                n_jobs=-1)

# the forest is already n_jobs=-1; nesting another pool oversubscribes
y_proba = cross_val_predict(forest, X_train, y_train_5, cv=cv,
                            method="predict_proba")
assert y_proba.shape == (60000, 2)
f_scores = y_proba[:, 1]                       # the POSITIVE class column

f_prec, f_rec, f_thr = precision_recall_curve(y_train_5, f_scores)

print(f"{'':28s}{'SGD':>10s}{'forest':>10s}")
print(f"{'ROC AUC':28s}{roc_auc_score(y_train_5, y_scores):>10.4f}"
      f"{roc_auc_score(y_train_5, f_scores):>10.4f}")
print(f"{'average precision':28s}{average_precision_score(y_train_5, y_scores):>10.4f}"
      f"{average_precision_score(y_train_5, f_scores):>10.4f}")
print(f"{'recall at 90% precision':28s}"
      f"{recalls[(precisions >= 0.90).argmax()]:>10.4f}"
      f"{f_rec[(f_prec >= 0.90).argmax()]:>10.4f}")
'''),
        md("""
Accuracy would call that a 1.87-point improvement, which reads as polish. The
last row moves by 24 points, and it is the row the brief is about.

**Which description you report decides whether anyone approves the change.**
"""),

        md("""
## 15 · The operating point, and the test shift

What the client asked for, in our vocabulary: **catch at least 90% of the 5s**
on a shift, and **flag no more than 1,000 items** out of 10,000 scanned.

Two constraints, one dial. A threshold either satisfies both or it does not, and
the honest answer may be that none does.

Choose on the cross-validated training scores. Then touch the test shift
**once**. ⏱ **about 30 seconds** for the two final fits.
"""),
        prompt(
            label="choose, then measure once",
            input="the training curves for both models, and the untouched test shift",
            output="a threshold per model chosen on training scores, then one pass over the test set",
            constraint="every threshold is chosen before the test set is touched, and the test set is touched exactly once",
            check="the code reads top to bottom as choose-then-measure, with no threshold computed after a test score is printed. Scroll position is the evidence. If a threshold is assigned anywhere below a printed test metric, the choice was informed by the answer.",
            **{"try": "move the threshold by 0.01 either way and re-measure on the test shift. The recall moves by less than the fold-to-fold spread, which is why 89.91% is not a failure."}),
        code('''
# 1. CHOOSE — training scores only
t_sgd    = thresholds[np.where(recalls >= 0.90)[0][-1]]
t_forest = f_thr[np.where(f_rec >= 0.90)[0][-1]]
print(f"SGD threshold    {t_sgd:.2f}")
print(f"forest threshold {t_forest:.2f}")

# what that threshold DELIVERS on the data it was chosen on. Keep this number:
# the test shift below is measured against it, and the gap is the whole point.
cv_recall = recall_score(y_train_5, f_scores >= t_forest)
print()
print(f"forest recall at that threshold, on the scores that chose it: "
      f"{cv_recall:.4f}")

# 2. MEASURE — once
sgd_final    = clf                                  # already fitted above
forest_final = forest.fit(X_train, y_train_5)

shift = {
    "SGD, default threshold":   sgd_final.decision_function(X_test) >= 0,
    "SGD, tuned for 90% recall": sgd_final.decision_function(X_test) >= t_sgd,
    "forest, tuned for 90% recall":
        forest_final.predict_proba(X_test)[:, 1] >= t_forest,
}
'''),
        prompt(
            label="the constraint the metric cannot see",
            input="each operating point's predictions on the test shift",
            output="flagged, caught, false alarms, missed and recall per operating point, with anything over capacity marked",
            constraint="the desk can re-check 1,000 items a shift; an operating point that flags more is not a worse option, it is an unavailable one",
            check="at least one row is marked OVER CAPACITY, or the constraint is not binding and there is nothing to teach here. Multiply the flagged rate by the shift volume and compare it with the staffing. A model nobody can act on has an accuracy and no value.",
            **{"try": "raise the desk capacity to 2,000 and re-solve for the threshold. The recall you can afford roughly doubles, on the same model and the same scores."}),
        code('''
CAPACITY = 1000            # stated by the client, not measured by us

print(f"{'operating point':30s}{'flagged':>8s}{'caught':>8s}"
      f"{'alarms':>8s}{'missed':>8s}{'recall':>9s}")
for name, pred in shift.items():
    tn_, fp_, fn_, tp2 = confusion_matrix(y_test_5, pred).ravel()
    flag = tp2 + fp_
    over = "  OVER CAPACITY" if flag > CAPACITY else ""
    print(f"{name:30s}{flag:>8,}{tp2:>8,}{fp_:>8,}{fn_:>8,}"
          f"{tp2 / (tp2 + fn_):>9.4f}{over}")
'''),
        md("""
The SGD classifier cannot satisfy both constraints: the only threshold that
reaches 90% recall flags 1,377 items, which is 377 over the desk's capacity.
**That is a finding, and it is the correct thing to report.**

The forest, tuned for the same 90% recall, flags 811 — inside capacity, because
almost none of them are false alarms.

### The wrinkle

The forest threshold was chosen because it gave 90.39% recall on the
cross-validated training scores. On the test shift it gives 89.91%. Half a point
below target.

Is that a failure, a bad fold, or noise? It is noise: a threshold chosen at
exactly 90% will land either side of it about half the time. **Report it, do not
tune it away.** Re-tuning to make the test number reach 90% is fitting the test
set.
"""),

        md("""
## 16 · Where we are

- Framing a detection task as binary classification makes the positive class
  rare by construction. One in ten here; one in ten thousand for card fraud.
- $\\text{accuracy} = p\\,\\text{recall} + (1-p)\\,\\text{specificity}$, so the
  never-fires classifier scores exactly $1-p$ and accuracy weights the class you
  care about by how rare it is. You verified that identity to the last decimal.
- Recall is monotone in the threshold; precision is not. You checked the
  one-step lemma at all 59,999 thresholds.
- The confusion matrix separates the two errors accuracy adds together, and the
  client prices them differently.
- An operating point is a choice against a stated constraint, and it has to be
  defended with the curve that matches the class balance — PR here, not ROC.

**Six questions to ask of any reported classification result:**

1. What is the base rate, and what does the never-fires classifier score?
2. Was the metric computed on held-out data?
3. Which of the two errors does the metric add together, and does the client
   price them the same?
4. If a threshold was chosen, on what data, and against what constraint?
5. Is the reported curve PR or ROC, and which one suits this class balance?
6. Was a second metric fixed as a constraint, or was one optimised alone?

**Before the next lecture:** run this notebook top to bottom. Then detect a
different digit — change the `5` in `y_train == 5` and re-run from that cell.
The baseline moves, the accuracy moves with it, and the pair tells you something
neither number does alone.
"""),
    ]
    return cells
