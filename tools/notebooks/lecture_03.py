#!/usr/bin/env python3
"""
Lecture 3 — Finding the rare event. Build, MNIST, Géron Chapter 3.

Exports build() -> list[cell]; tools/make_notebooks.py wraps it.

Structure mirrors the deck: brief -> data -> metric -> anchor -> commitment ->
build -> the worked assistant failure -> record the number. Every structural
step is followed by an assertion, and anything slower than about twenty seconds
states its wall clock, because "no output" otherwise reads as "it hung".
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
    cells = header(3, "Finding the rare event", "build", "Chapter 3")

    cells += [
        md("## 1 · Setup"), SETUP_PROMPT, SETUP,
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
            check="assert the shape is (70000, 784) rather than trusting the "
                  "download",
            left_open="what to do if the download fails or is truncated. A "
                      "short read gives a smaller X and the assert catches it; "
                      "anything subtler it will not.",
            student="`fetch_openml(...)` with no assert, on the reasonable "
                    "grounds that a well-known dataset is a well-known shape. "
                    "It is, until a cache is half-written.",
            catch="784 = 28 x 28. If the second dimension is not a perfect "
                  "square you are not holding images, whatever the variable "
                  "is called."),
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
            check="after the cast, the ten digits 0-9 are all present",
            left_open="that `y == 5` comparing strings to an int is legal Python. It returns all-False, `.sum()` returns 0, and every downstream cell runs happily on a label vector with no positives in it.",
            student="casting without ever printing the before. You then never learn that the comparison you are about to write a hundred times can fail silently, and you meet it again in your own project.",
            catch="count your positives immediately after building a boolean label, and compare with what you expect. Zero is a number the code will not complain about."),
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
            left_open="what you are looking for. 'Look at the data' is advice nobody can act on; here you are checking two specific things, that the images are digits and that the titles match them.",
            student="skipping this because MNIST is famous. The one time the labels are shifted by one relative to the images, every metric downstream is plausible and wrong, and this cell is where it would have shown.",
            catch="read the titles against the pictures. You know what a 5 looks like; that is the entire test and it takes four seconds."),
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
            check="assert the two sizes and that pixels are still 0-255",
            left_open="that this is the last time X_test is mentioned until the end of the NEXT notebook. The comment says so; nothing enforces it.",
            student="`train_test_split(X, y, test_size=1/7, random_state=42)`, which gives the right sizes and the wrong rows, and quietly makes your numbers incomparable with every paper on this dataset.",
            catch="the pixel range assert. If scaling has crept in above this line, min/max stop being 0/255 and the split is no longer the raw one."),
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
            check="assert 5,421 positives — which fails loudly if the uint8 cast was skipped",
            left_open="which way round the imbalance runs. 9% positive is mild by the standards of fraud or fault detection, and it is already enough to make accuracy useless. Nothing here says what 'enough' means.",
            student="going straight to a model. The base rate is one line and it decides whether accuracy is a metric or a decoration.",
            catch="the assert is doing double duty: it checks the positive count and, because a string-vs-int comparison gives zero, it also catches the missing cast two cells up."),
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
            left_open="that two different things are being called balance. The ten classes are near-equal; the binary task built out of them is 9 to 1. A dataset is not imbalanced or not — a TASK is.",
            student="reading 'MNIST is balanced' somewhere and concluding accuracy is safe here. It is safe for ten-way classification and useless for this one, on the same rows.",
            catch="count the positives of the task you are actually solving, never the classes of the dataset you started from."),
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
            check="its accuracy should equal 1 minus the base rate, to three places",
            left_open="why bother measuring what algebra gives you. Because the pipeline measuring it is the thing under test too: if your CV is misconfigured, this anchor comes out wrong and tells you so before a real model hides the problem.",
            student="quoting 90.96% from the base rate and never running it. You then have no evidence your evaluation code works at all.",
            catch="anchor and 1 - base_rate must agree. If they do not, the fault is in the splitter or the scorer, not in the estimator that does nothing."),
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
## 8 · Commit

**Stop. On paper, now** — not in this notebook, where you can quietly revise
it.

```
Metric:                                          ____________
Accuracy a good detector would need:             ____________
Accuracy I expect from the model I build today:  ____________
```

You are estimating, not guessing: doing nothing scores 90.96%, a perfect
detector scores 100%, and your number lives somewhere in between. Saying
*where* is the exercise.
"""),

        md("""
## 9 · Build the simplest thing that runs

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
            check="assert the coefficient shape is (1, 784), one weight per pixel",
            left_open="⏱ about 30 seconds, which the comment gives but the specification does not. A cell with no stated cost is a cell someone interrupts.",
            student="`StandardScaler().fit_transform(X_train)` on its own line, then a classifier. It trains identically and leaks the moment it meets cross-validation, which is the next cell but one.",
            catch="(1, 784) is one row of weights over 784 pixels. A shape of (10, 784) means you fitted the ten-class problem by accident."),
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
            left_open="why that constraint is there, and it is the argument of the whole lecture: a detector that never fires gets one of these two right, so a passing assertion on a single example is compatible with a model that does nothing.",
            student="`assert clf.predict([X_train[i_pos]])[0] == True`, which feels like a test and is satisfied by luck on a class that is 9% of the data.",
            catch="ask what a broken model would score on your assertion. If it passes, the assertion is decoration."),
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
## 10 · Measure it honestly

⏱ **about 70 seconds** — three complete refits of scaler and classifier.
"""),
        prompt(
            label="cross-validated, against the anchor",
            input="the pipeline, the training rows, three stratified folds",
            output="per-fold accuracy, the mean, and the gap over the never-fires anchor",
            constraint="print the anchor beside it — a lone accuracy is not a result",
            check="the gap in percentage points, stated explicitly rather than left for the reader to subtract",
            left_open="how big a gap would be worth having. The cell prints it and does not judge it; six points sounds like a lot until you notice the anchor is at 91.",
            student="reporting 96.9% and stopping. It is true, it is cross-validated, and it is five points from a model with no parameters.",
            catch="always subtract. An accuracy without its anchor beside it cannot be read, and the subtraction is what makes 96.9% shrink to 5.9."),
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
            check="compare the WORST unscaled fold with the anchor, not the mean",
            left_open="that the interesting statistic here is the spread, not the average. Unscaled SGD can land anywhere; a mean over three folds hides a fold that fell over.",
            student="comparing the two means, seeing a point or two, and concluding scaling barely matters. Look at the worst fold instead.",
            catch="the worst unscaled fold against the never-fires anchor. If a fold sits near the anchor, that configuration failed completely on that split and the mean is covering for it."),
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
## 11 · What `cross_val_score` is doing

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
            check="the hand-rolled numbers must match cross_val_score's",
            left_open="what clone actually copies. It takes the hyperparameters and discards the fitted state; passing the same object without it gives you three folds of a model that has already seen everything, and the accuracies come out higher.",
            student="`fold_clf = clf` and a loop. The numbers rise, nothing errors, and the fold structure has quietly stopped meaning anything.",
            catch="if your by-hand numbers do not reproduce the library's, one of the two is wrong and it is worth ten minutes to find out which."),
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
## 12 · An assistant evaluates the classifier

Here is a real request and the code it returns. **⚠ Read before running.** It
runs, it imports nothing exotic, and it prints a number *better* than the one we
just measured.

> *"Evaluate my classifier on the MNIST 5-detector and print the score."*
"""),
        prompt(
            label="the number you must not report",
            input="the fitted model and the rows it was fitted on",
            output="its accuracy on those same rows",
            constraint="this is here to be compared with the cross-validated figure, not to be quoted",
            left_open="how far apart the two should be. This model is barely flexible enough to overfit, so the gap is small — which is itself worth seeing, because it means a small gap is not evidence of honesty.",
            student="reporting this one. It is the first number `.score()` gives you and it is measured on the answers.",
            catch="every accuracy in a notebook should be labelled with what it was measured on. If a number has no such label, assume it was measured on the training rows."),
        code('''
from sklearn.metrics import accuracy_score

y_pred = clf.predict(X_train)
print(f"Accuracy: {accuracy_score(y_train_5, y_pred):.4f}")
'''),
        md("""
### Reviewer question 1: what touched what?

`clf` was **fitted** on `X_train`. It then **predicted** on `X_train`. The score
was then computed on `X_train`. The same 60,000 rows, three times.

### Reviewer question 5: what is the default I did not ask for?

Two of them, in three lines. The assistant chose **accuracy**, because the
prompt said "the score" and accuracy is what a classifier reports when nobody
says otherwise. And it chose to evaluate on the training set, because the prompt
named no evaluation set.

Neither choice was flagged. You asked for a number and you got one.

**Now measure the damage** — and pair the measurements, so that fold-to-fold
variation cancels. ⏱ **about four minutes**, because a single-seed gap of half a
point is not a measurement.
"""),
        prompt(
            label="the gap, over five seeds",
            input="five seeds, three folds each",
            output="fifteen paired (train, held-out) accuracies and the gap between them",
            constraint="pair them — each gap comes from ONE fitted model, so the fold-to-fold variation cancels",
            check="report the spread of the gap, not only its mean",
            left_open="why five seeds rather than one. A single-seed gap of half a point is indistinguishable from noise, and this lecture is about to make an argument that rests on the gap being real.",
            student="one seed, one gap, and a conclusion. The course does this properly here and gets it wrong in an earlier draft of Lecture 14 — which is recorded in that deck rather than hidden.",
            catch="if the spread of the gap is comparable to the gap itself, you have measured nothing. Fifteen pairs is what makes that judgement possible."),
        code('''
from sklearn.model_selection import cross_validate

pairs = []
for seed in (42, 43, 44, 45, 46):
    cv_s = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    r = cross_validate(make_pipeline(StandardScaler(),
                                     SGDClassifier(random_state=seed)),
                       X_train, y_train_5, cv=cv_s, scoring="accuracy",
                       return_train_score=True, n_jobs=-1)
    pairs += list(zip(r["train_score"], r["test_score"]))

gaps = np.array([tr - te for tr, te in pairs])
print(f"{len(pairs)} paired measurements (5 seeds x 3 folds)")
print(f"train mean {np.mean([t for t, _ in pairs]):.5f}")
print(f"cv    mean {np.mean([t for _, t in pairs]):.5f}")
print(f"gap        {100 * gaps.mean():.2f} pp   sd {100 * gaps.std(ddof=1):.2f}")
print(f"positive in {(gaps > 0).sum()}/{len(gaps)} pairs")

assert (gaps > 0).all(), "a model always fits what it was fitted to at least as well"
'''),
        md("""
### Half a point. So why is it a bug?

1. **You did not know it was half a point until you measured.** A decision tree
   in the previous lecture had a gap of the entire number, and nothing in either
   piece of code said which case you were in.
2. **It is this small for a specific reason**: a linear model with 785
   parameters cannot memorise 60,000 instances. Give the same code an
   unconstrained tree and the gap is everything.
3. **The sign is guaranteed and the size is not.** Fifteen out of fifteen — the
   assertion above depends on it — because a model always fits the data it was
   fitted to at least as well as data it has not seen.

The rule is procedural: never score on the rows you fitted. Not because the
damage is always large, but because you cannot tell whether it is.

### The corrected specification

> *"Evaluate this classifier with **stratified 3-fold cross-validation** on the
> training set, seed 42. Report **accuracy per fold** and the mean. In the same
> table, report the same metric for a baseline that always predicts the negative
> class. Do not touch `X_test`."*

Notice what is *not* in it: the metric is still accuracy. We named it, so it is
now ours — and being ours is what makes it something we can be held to.
"""),

        md("""
## 13 · The five reviewer questions, on this notebook

| # | Question | Answer |
|---|---|---|
| 1 | What touched the test set? | Nothing. `X_test` appears in the split and not again |
| 2 | What was fitted, and on what? | Scaler and classifier together, per fold, on that fold's training part |
| 3 | What is the shape here? | 60,000 × 784, asserted |
| 4 | What was dropped? | Nothing — no missing values, no rows or columns removed |
| 5 | What is the default I did not ask for? | The label dtype, the fold count, the loss inside `SGDClassifier`, and the metric |

Four of the five come out clean. Keep the answer to the fifth.
"""),

        md("""
## 14 · Record it

Add one line to your sheet of paper, underneath the three you wrote:

```
Best accuracy I actually obtained:  ____________
```

Bring the sheet to the next lecture. We open by comparing all four numbers out
loud, and the comparison is the lecture.

**Do not tidy this notebook, do not tune anything, and do not delete the version
with the bug in it.** You will need it.
"""),
        prompt(
            label="the five numbers to carry forward",
            input="everything measured above",
            output="one table: positives, base rate, anchor, cross-validated accuracy, and the training-set accuracy",
            constraint="the anchor appears in the table, not in the prose above it",
            left_open="what to conclude. The table is deliberately without a verdict, because the verdict is the next lecture: 96.9% against an anchor of 90.96% is the entire setup for 'it never fires'.",
            student="a summary of the model's numbers with the baseline left out, which is how a five-point improvement gets presented as a 97% success.",
            catch="photograph this table. Every number in the next lecture is measured against one of these five."),
        code('''
summary = {
    "positives in the training set": f"{n_pos:,}",
    "base rate":                     f"{100 * base_rate:.2f}%",
    "never-fires accuracy":          f"{100 * anchor.mean():.2f}%",
    "ours, cross-validated":         f"{100 * scores.mean():.2f}%",
    "ours, scored on its own rows":  f"{100 * accuracy_score(y_train_5, y_pred):.2f}%",
}
for k, v in summary.items():
    print(f"{k:32s} {v:>8s}")
'''),
        md("""
Six numbers, and their **status** matters as much as their value: four measured,
one stated by the client, one discarded.
"""),
    ]
    return cells
