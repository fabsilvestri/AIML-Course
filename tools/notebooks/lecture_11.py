"""
Lecture 11 — The first neural network. Fashion MNIST with Scikit-Learn's
MLPClassifier.

Exports build() -> list[nbformat cell], the contract tools/make_notebooks.py
expects. The deck's headline numbers come from the full 55,000 training images;
this notebook trains on 12,000 so that a free Colab CPU runtime finishes inside
the lecture hour. Every cell says which.
"""

from __future__ import annotations

import nbformat as nbf
from _prompt import prompt                                # noqa: E402


def md(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(text.strip("\n"))


def code(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(text.strip("\n"))


HEADER = """
# The first neural network

**Lecture 11 · Build** · Géron, Chapter 9

Applications of Machine Learning — BSc Mathematics of Artificial Intelligence

---

**How to use this notebook.** You are not expected to type the code. You are
expected to *read* it before you run it, and to be able to say what every line
does and what would break if it changed. Cells marked **⚠ read before running**
contain a defect on purpose.

Run the cells in order. Anything that takes more than a few seconds says so.

**This notebook trains on 12,000 of the 55,000 training images.** The deck
quotes the full run; a free Colab CPU would spend most of the lecture on it.
Where a number here differs from the slide, that is why — and the difference is
itself worth a sentence in your notes.
"""


def build() -> list:
    return [
        md(HEADER),

        md("## 1 · Setup"),
        prompt(
            label="setup",
            input="nothing",
            output="the versions of everything, and a fixed seed",
            constraint="print the versions — a mismatch here produces a confusing error twenty cells later, in a cell that has nothing to do with it",
            left_open="why torch is imported at all in a Scikit-Learn lecture. torchvision is the tidiest loader for Fashion MNIST and nothing else here uses torch — the comment says so, because otherwise the import reads as a promise the notebook does not keep.",
            student="importing torch and reaching for it later, on the reasonable assumption that anything imported is fair game. Today is Scikit-Learn only, and the walls at the end are the reason.",
            catch="an import with no use is a question. Answer it in a comment or delete it."),
        prompt(
            label="setup",
            input="nothing",
            output="the versions of everything, and a fixed seed",
            constraint="print the versions — a mismatch here produces a confusing error twenty cells later, in a cell that has nothing to do with it",
            left_open="why torch is imported at all in a Scikit-Learn lecture. torchvision is the tidiest loader for Fashion MNIST and nothing else here uses torch — the comment says so, because otherwise the import reads as a promise the notebook does not keep.",
            student="importing torch and reaching for it later, on the reasonable assumption that anything imported is fair game. Today is Scikit-Learn only, and the walls at the end are the reason.",
            catch="an import with no use is a question. Answer it in a comment or delete it."),
        code('''
# --- setup -------------------------------------------------------------------
# Not examinable: engineering hygiene. It is here because a version mismatch
# produces a confusing error twenty cells later.
import sys, time, warnings
import numpy as np, sklearn, torch, torchvision
import matplotlib.pyplot as plt

print(f"python       {sys.version.split()[0]}")
print(f"scikit-learn {sklearn.__version__}")
print(f"torch        {torch.__version__}")
print(f"torchvision  {torchvision.__version__}")

RANDOM_STATE = 42          # every split, every model, every shuffle
np.random.seed(RANDOM_STATE)

# Today is Scikit-Learn only. torchvision is imported for the dataset alone —
# it is the tidiest loader for Fashion MNIST and nothing else here uses torch.
'''),

        md("""
## 2 · The brief

A document-processing bureau scans incoming items and has to route each one to
one of ten queues. Nobody will label a training set for you twice, the images
are small and greyscale, and the operator wants to know how often the machine
is right.

We stand in for that corpus with **Fashion MNIST**: 70,000 greyscale 28×28
images in ten classes, which is the dataset Chapter 9 uses. Same shape of
problem, and — unlike the bureau's archive — it is public, so your number is
comparable with everyone else's.
"""),
        prompt(
            label="the data",
            input="Fashion MNIST",
            output="60,000 training and 10,000 test images as uint8 arrays, with the class names",
            constraint="keep them as uint8 for now and print the pixel range — the next sections are about what happens when that range is not what the model expects",
            check="assert both shapes and the dtype",
            left_open="that the pixel range is 0 to 255 and nothing has been divided yet. Section 6 is entirely about a prompt that failed to say so.",
            student="scaling immediately, out of habit, which is correct practice and removes the lecture's central demonstration. The scaling happens two sections down, deliberately.",
            catch="assert the dtype, not just the shape. `uint8` versus `float32` is the difference between an image and a model input, and the failure is silent."),
        prompt(
            label="the data",
            input="Fashion MNIST",
            output="60,000 training and 10,000 test images as uint8 arrays, with the class names",
            constraint="keep them as uint8 for now and print the pixel range — the next sections are about what happens when that range is not what the model expects",
            check="assert both shapes and the dtype",
            left_open="that the pixel range is 0 to 255 and nothing has been divided yet. Section 6 is entirely about a prompt that failed to say so.",
            student="scaling immediately, out of habit, which is correct practice and removes the lecture's central demonstration. The scaling happens two sections down, deliberately.",
            catch="assert the dtype, not just the shape. `uint8` versus `float32` is the difference between an image and a model input, and the failure is silent."),
        code('''
# --- the data ----------------------------------------------------------------
# ~30 MB the first time, a few seconds; instant afterwards.
train_ds = torchvision.datasets.FashionMNIST("datasets", train=True, download=True)
test_ds  = torchvision.datasets.FashionMNIST("datasets", train=False, download=True)

CLASSES = train_ds.classes
X_train_full_u8 = train_ds.data.numpy()          # (60000, 28, 28), uint8
y_train_full    = train_ds.targets.numpy().astype(np.int64)
X_test_u8       = test_ds.data.numpy()
y_test          = test_ds.targets.numpy().astype(np.int64)

assert X_train_full_u8.shape == (60000, 28, 28), X_train_full_u8.shape
assert X_test_u8.shape == (10000, 28, 28), X_test_u8.shape
assert X_train_full_u8.dtype == np.uint8
print(f"{len(X_train_full_u8):,} training images, {len(X_test_u8):,} test images")
print(f"{len(CLASSES)} classes: {', '.join(CLASSES)}")
print(f"pixel values run {X_train_full_u8.min()} to {X_train_full_u8.max()}")
'''),

        md("""
### Look at it before you model it

The rule from Lecture 1 has not changed. Two things to notice: several classes
are garments photographed the same way, and the background is exactly zero.
"""),
        prompt(
            label="look at it before you model it",
            input="three examples of each of the ten classes",
            output="a 3 by 10 grid, each column titled with its class",
            constraint="`vmin=0, vmax=255` — without it every thumbnail is rescaled to its own range and a dark garment looks identical to a bright one",
            left_open="two things to notice: several classes are garments photographed the same way, and the background is exactly zero. Both come back in the error analysis.",
            student="skipping this because the shape assert passed. The confusions the model makes later are exactly the ones you would make from this grid, and knowing that in advance is worth the thirty seconds.",
            catch="`gray_r` rather than `gray`. Fashion MNIST is white-on-black, and reversed it looks like the scanned document the brief is about."),
        prompt(
            label="look at it before you model it",
            input="three examples of each of the ten classes",
            output="a 3 by 10 grid, each column titled with its class",
            constraint="`vmin=0, vmax=255` — without it every thumbnail is rescaled to its own range and a dark garment looks identical to a bright one",
            left_open="two things to notice: several classes are garments photographed the same way, and the background is exactly zero. Both come back in the error analysis.",
            student="skipping this because the shape assert passed. The confusions the model makes later are exactly the ones you would make from this grid, and knowing that in advance is worth the thirty seconds.",
            catch="`gray_r` rather than `gray`. Fashion MNIST is white-on-black, and reversed it looks like the scanned document the brief is about."),
        code('''
fig, axes = plt.subplots(3, 10, figsize=(13, 4.4))
for c in range(10):
    idx = np.where(y_train_full == c)[0][:3]
    for r in range(3):
        ax = axes[r, c]
        ax.imshow(X_train_full_u8[idx[r]], cmap="gray_r", vmin=0, vmax=255)
        ax.set_xticks([]); ax.set_yticks([])
    axes[0, c].set_title(CLASSES[c], fontsize=8)
plt.tight_layout(); plt.show()
'''),
        prompt(
            label="the assertion that decides the metric",
            input="the training labels",
            output="the count per class",
            constraint="assert the classes are EXACTLY balanced rather than observing that they look balanced",
            check="min equals max equals 6,000",
            left_open="what follows from it. The previous application spent an hour on why accuracy is worthless under imbalance; here the classes are balanced by construction, so accuracy is meaningful AND the trivial baseline is trivially computable.",
            student="carrying the habit over uninspected — either using accuracy because it is familiar, or refusing to because the last lecture said not to. Check, then choose.",
            catch="one assert here licenses every accuracy in the notebook. That is a lot of weight for one line, which is why it is an assert and not a print."),
        prompt(
            label="the assertion that decides the metric",
            input="the training labels",
            output="the count per class",
            constraint="assert the classes are EXACTLY balanced rather than observing that they look balanced",
            check="min equals max equals 6,000",
            left_open="what follows from it. The previous application spent an hour on why accuracy is worthless under imbalance; here the classes are balanced by construction, so accuracy is meaningful AND the trivial baseline is trivially computable.",
            student="carrying the habit over uninspected — either using accuracy because it is familiar, or refusing to because the last lecture said not to. Check, then choose.",
            catch="one assert here licenses every accuracy in the notebook. That is a lot of weight for one line, which is why it is an assert and not a print."),
        code('''
counts = np.bincount(y_train_full, minlength=10)
for c, n in enumerate(counts):
    print(f"{CLASSES[c]:12s} {n:,}")
assert counts.min() == counts.max() == 6000, "not balanced after all"
print("\\nperfectly balanced — 6,000 of each")
'''),
        md("""
**That single assertion decides the metric.** Lecture 4 spent an hour on why
accuracy is worthless under imbalance. Here the classes are exactly balanced by
construction, so accuracy is meaningful *and* the trivial baseline is trivially
computable. Do not carry the habit over uninspected: check, then choose.
"""),

        md("""
## 3 · Scale, and split

Two operations, in this order, and both matter.

**Scale.** Pixels arrive as integers 0–255. Neural networks are trained by
gradient descent, and the size of a gradient step is set once for every weight;
inputs two orders of magnitude apart make one step too big for some weights and
too small for others. Dividing by 255 is the whole of it here — every feature is
already on the same scale as every other, so there is nothing to fit and
therefore nothing to leak.

**Split.** The 10,000 test images are the ones the dataset ships as a test set.
We carve a further 5,000 validation images out of the training half, because we
are about to tune hyperparameters by hand and the test set is not for that.
"""),
        prompt(
            label="scale, and split",
            input="the uint8 images",
            output="784-dimensional float32 rows in [0,1], split into fit / validation / test",
            constraint="divide by 255 — every feature is already on the same scale as every other, so there is nothing to FIT and therefore nothing to leak",
            check="assert the sizes sum, that the indices are disjoint, and that the values really are in [0,1]",
            left_open="why scaling matters for a network specifically. The size of a gradient step is set once for every weight, so inputs two orders of magnitude apart make one step too big for some weights and too small for others.",
            student="reaching for StandardScaler, which is a fitted transform and would need to live inside a pipeline. Dividing by a known constant is not the same operation and does not carry the same risk.",
            catch="the disjointness assert on the two index sets. A permutation sliced in two cannot overlap, and asserting it anyway costs nothing and catches the day someone changes the slicing."),
        prompt(
            label="scale, and split",
            input="the uint8 images",
            output="784-dimensional float32 rows in [0,1], split into fit / validation / test",
            constraint="divide by 255 — every feature is already on the same scale as every other, so there is nothing to FIT and therefore nothing to leak",
            check="assert the sizes sum, that the indices are disjoint, and that the values really are in [0,1]",
            left_open="why scaling matters for a network specifically. The size of a gradient step is set once for every weight, so inputs two orders of magnitude apart make one step too big for some weights and too small for others.",
            student="reaching for StandardScaler, which is a fitted transform and would need to live inside a pipeline. Dividing by a known constant is not the same operation and does not carry the same risk.",
            catch="the disjointness assert on the two index sets. A permutation sliced in two cannot overlap, and asserting it anyway costs nothing and catches the day someone changes the slicing."),
        code('''
def flatten_scale(a):
    """(n, 28, 28) uint8  ->  (n, 784) float32 in [0, 1]."""
    return a.reshape(len(a), -1).astype(np.float32) / 255.0

rng = np.random.default_rng(RANDOM_STATE)
order = rng.permutation(len(X_train_full_u8))
val_idx, fit_idx = order[:5_000], order[5_000:]

X_fit_full, y_fit_full = flatten_scale(X_train_full_u8[fit_idx]), y_train_full[fit_idx]
X_val,      y_val      = flatten_scale(X_train_full_u8[val_idx]), y_train_full[val_idx]
X_test,     y_test_    = flatten_scale(X_test_u8), y_test

assert len(X_fit_full) + len(X_val) == 60_000
assert set(fit_idx).isdisjoint(val_idx), "the split overlaps"
assert X_fit_full.shape[1] == 784 and X_val.shape[1] == 784
assert 0.0 <= X_fit_full.min() and X_fit_full.max() <= 1.0
print(f"fit {len(X_fit_full):,}   val {len(X_val):,}   test {len(X_test):,}")

# the subset this notebook actually trains on
SUB = 12_000
X_fit, y_fit = X_fit_full[:SUB], y_fit_full[:SUB]
print(f"\\ntraining on the first {SUB:,} of them, so this finishes in the hour")
'''),

        md("""
## 4 · A number to compare against

Before anything is built: the dumbest model that is still a model. Ten balanced
classes, so predicting the commonest one is right **one time in ten**.
"""),
        prompt(
            label="the anchor",
            input="the majority class",
            output="its accuracy on the test set",
            constraint="assert it is exactly 0.10 — the test set has exactly 1,000 of each class, so anything else means the test set is not what you think it is",
            left_open="that this is also why accuracy is READABLE here. A number is only informative relative to what it must beat.",
            student="skipping the baseline because 'ten classes means 10%'. That is true of a balanced test set and this cell is what establishes the test set is balanced.",
            catch="an exact assert is available here because the dataset is exactly balanced. Take exact asserts when the data allows them; they catch things tolerances do not."),
        prompt(
            label="the anchor",
            input="the majority class",
            output="its accuracy on the test set",
            constraint="assert it is exactly 0.10 — the test set has exactly 1,000 of each class, so anything else means the test set is not what you think it is",
            left_open="that this is also why accuracy is READABLE here. A number is only informative relative to what it must beat.",
            student="skipping the baseline because 'ten classes means 10%'. That is true of a balanced test set and this cell is what establishes the test set is balanced.",
            catch="an exact assert is available here because the dataset is exactly balanced. Take exact asserts when the data allows them; they catch things tolerances do not."),
        code('''
from sklearn.metrics import accuracy_score

majority = np.bincount(y_fit).argmax()
baseline = np.full(len(y_test_), majority)
baseline_acc = accuracy_score(y_test_, baseline)

print(f"always predict '{CLASSES[majority]}'  ->  accuracy {baseline_acc:.4f}")
assert abs(baseline_acc - 0.10) < 1e-9, "the test set is not balanced after all"
print("exactly 10.00% — because the test set has exactly 1,000 of each class")
'''),
        md("""
That is the anchor. It is also the reason accuracy is readable at all here: a
number is only informative relative to what it must beat.
"""),

        md("""
## 5 · Commit

**Stop. On paper, now.** Not in this notebook — on paper, where you cannot
quietly revise it.

```
Metric:                                          ____________
Accuracy a good sorting machine would need:      ____________ %
Accuracy I expect from the model I build today:  ____________ %
```

A prediction you can silently revise is not a prediction.
"""),

        md("""
## 6 · An assistant writes the training code

A real request, and the code it returns. **⚠ Read before running.** It imports
nothing exotic, it raises nothing, and it prints a believable number.

> *"Train a neural network to classify Fashion MNIST images and print the
> accuracy."*

⏱ **about 40 seconds.**
"""),
        prompt(
            label="⏱ 40 s — ⚠ what the assistant returns",
            input="'train a neural network to classify Fashion MNIST and print the accuracy'",
            output="a fitted MLP and its validation accuracy",
            constraint="feed it the RAW uint8 pixels, as the prompt implies — it imports nothing exotic, raises nothing, and prints a believable number",
            left_open="reviewer question 5. Two defaults, and only one is visible: `max_iter=12` is ours, and `learning_rate_init=0.001` is scikit-learn's — the default for inputs of order ONE. We handed the network integers up to 255.",
            student="reading the accuracy and moving on. Nothing in the prompt said 'scale the pixels', nothing in the output said it had not happened, and the number is high enough to look like a result.",
            catch="a library default is chosen for a typical input range. When your inputs are not in that range the default is not a default, it is a mistake with a plausible value."),
        prompt(
            label="⏱ 40 s — ⚠ what the assistant returns",
            input="'train a neural network to classify Fashion MNIST and print the accuracy'",
            output="a fitted MLP and its validation accuracy",
            constraint="feed it the RAW uint8 pixels, as the prompt implies — it imports nothing exotic, raises nothing, and prints a believable number",
            left_open="reviewer question 5. Two defaults, and only one is visible: `max_iter=12` is ours, and `learning_rate_init=0.001` is scikit-learn's — the default for inputs of order ONE. We handed the network integers up to 255.",
            student="reading the accuracy and moving on. Nothing in the prompt said 'scale the pixels', nothing in the output said it had not happened, and the number is high enough to look like a result.",
            catch="a library default is chosen for a typical input range. When your inputs are not in that range the default is not a default, it is a mistake with a plausible value."),
        code('''
from sklearn.neural_network import MLPClassifier

# the assistant's code, unedited
X_raw = X_train_full_u8[fit_idx][:SUB].reshape(SUB, -1).astype(np.float32)
X_raw_val = X_train_full_u8[val_idx].reshape(5_000, -1).astype(np.float32)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")           # it warns; we come back to that
    t0 = time.perf_counter()
    clf_raw = MLPClassifier(hidden_layer_sizes=(300, 100), max_iter=12,
                            random_state=RANDOM_STATE).fit(X_raw, y_fit)
    t_raw = time.perf_counter() - t0

acc_raw = clf_raw.score(X_raw_val, y_val)
print(f"validation accuracy {acc_raw:.4f}   ({t_raw:.0f} s)")
'''),
        md("""
### Reviewer question 5: what is the default I did not ask for?

Two of them, and only one is visible.

`max_iter=12` is ours. `learning_rate_init=0.001` is Scikit-Learn's, and it is
the default *for inputs of order one*. We handed the network integers up to 255.

Nothing in the prompt said "scale the pixels", nothing in the output said it had
not happened, and the accuracy is high enough to look like a result.

**Measure it** — do not guess.
"""),
        prompt(
            label="measure it, do not guess",
            input="the same architecture, same epochs, same seed, scaled pixels",
            output="both accuracies and both final training losses",
            constraint="change ONE thing — only the input scale differs, so the difference is attributable",
            left_open="the loss column. The raw model's training loss is far higher, which says the optimiser never got going rather than that the problem is hard.",
            student="asserting that scaling matters, or quoting a figure from a textbook. The cost is measurable in forty seconds on your own data and it is different for every dataset.",
            catch="report the cost in the units the stakeholder uses. 'The missing division by 255 costs N accuracy points' is a sentence; 'scaling is important' is not."),
        prompt(
            label="measure it, do not guess",
            input="the same architecture, same epochs, same seed, scaled pixels",
            output="both accuracies and both final training losses",
            constraint="change ONE thing — only the input scale differs, so the difference is attributable",
            left_open="the loss column. The raw model's training loss is far higher, which says the optimiser never got going rather than that the problem is hard.",
            student="asserting that scaling matters, or quoting a figure from a textbook. The cost is measurable in forty seconds on your own data and it is different for every dataset.",
            catch="report the cost in the units the stakeholder uses. 'The missing division by 255 costs N accuracy points' is a sentence; 'scaling is important' is not."),
        code('''
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    t0 = time.perf_counter()
    clf_scaled = MLPClassifier(hidden_layer_sizes=(300, 100), max_iter=12,
                               random_state=RANDOM_STATE).fit(X_fit, y_fit)
    t_scaled = time.perf_counter() - t0

acc_scaled = clf_scaled.score(X_val, y_val)
print(f"pixels 0-255   {acc_raw:.4f}")
print(f"pixels 0-1     {acc_scaled:.4f}")
print(f"the missing division by 255 costs "
      f"{100 * (acc_scaled - acc_raw):+.2f} accuracy points")
print(f"final training loss: raw {clf_raw.loss_:.4f}   scaled {clf_scaled.loss_:.4f}")
'''),
        md("""
### The corrected specification

> *"Load Fashion MNIST. Scale the pixels to [0, 1] as float32. Split off 5,000
> validation images with a fixed seed. Train an MLP with hidden layers (300, 100)
> for 12 epochs, seed 42, and report accuracy on the validation set — not on the
> training set. State the wall-clock time."*

Input, output, constraint, check. The vaguer prompt was not wrong about what to
build; it was silent about the conditions under which the thing works.
"""),

        md("""
## 7 · Build it properly, one epoch at a time

`fit()` gives you a number at the end. `partial_fit()` runs a single pass and
hands control back, which is as close to the inside of the loop as
Scikit-Learn will let you get. Remember that; it is the point of the next
lecture.

⏱ **about 60 seconds** for 20 epochs on 12,000 images.
"""),
        prompt(
            label="⏱ 60 s — one epoch at a time",
            input="12,000 images, 20 epochs",
            output="the loss and both accuracies after every epoch, plus the parameter count and the wall clock",
            constraint="`partial_fit` with the full `classes=` list on every call — one pass, control handed back. `fit()` gives you one number at the end and nothing in between",
            check="assert twenty epochs were recorded",
            left_open="that this is as close to the inside of the training loop as scikit-learn will let you get. Remember that; it is the point of the next lecture.",
            student="omitting `classes=` on `partial_fit`, which works on the first call and raises on the second, or worse, silently fits a model that has never seen a class absent from the first batch.",
            catch="record training AND validation accuracy each epoch. The gap between them is Lecture 2's train-versus-cross-validation table drawn as two lines, and nothing about neural networks makes it go away."),
        prompt(
            label="⏱ 60 s — one epoch at a time",
            input="12,000 images, 20 epochs",
            output="the loss and both accuracies after every epoch, plus the parameter count and the wall clock",
            constraint="`partial_fit` with the full `classes=` list on every call — one pass, control handed back. `fit()` gives you one number at the end and nothing in between",
            check="assert twenty epochs were recorded",
            left_open="that this is as close to the inside of the training loop as scikit-learn will let you get. Remember that; it is the point of the next lecture.",
            student="omitting `classes=` on `partial_fit`, which works on the first call and raises on the second, or worse, silently fits a model that has never seen a class absent from the first batch.",
            catch="record training AND validation accuracy each epoch. The gap between them is Lecture 2's train-versus-cross-validation table drawn as two lines, and nothing about neural networks makes it go away."),
        code('''
def train_curve(X, y, hidden=(300, 100), lr=1e-3, epochs=20, batch=128):
    clf = MLPClassifier(hidden_layer_sizes=hidden, activation="relu",
                        solver="adam", learning_rate_init=lr,
                        batch_size=batch, random_state=RANDOM_STATE)
    loss, val_acc, train_acc = [], [], []
    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(epochs):
            clf.partial_fit(X, y, classes=np.arange(10))
            loss.append(clf.loss_)
            val_acc.append(clf.score(X_val, y_val))
            train_acc.append(clf.score(X[:5_000], y[:5_000]))
    return clf, {"loss": loss, "val_acc": val_acc, "train_acc": train_acc,
                 "seconds": time.perf_counter() - t0}

clf, hist = train_curve(X_fit, y_fit)
n_params = sum(w.size for w in clf.coefs_) + sum(b.size for b in clf.intercepts_)

assert len(hist["loss"]) == 20
print(f"{n_params:,} parameters")
print(f"{hist['seconds']:.0f} s for 20 epochs "
      f"({hist['seconds'] / 20:.1f} s per epoch)")
print(f"validation accuracy {hist['val_acc'][-1]:.4f}")
'''),
        prompt(
            label="the two curves",
            input="the recorded history",
            output="loss against epoch, and both accuracies against epoch",
            constraint="both accuracy curves on the SAME axis — the gap is the quantity being shown",
            left_open="that the gap is the overfitting and you have seen it before. It is not a new phenomenon and it does not have a neural-network-specific cure.",
            student="plotting only the loss, which falls smoothly and forever and says nothing about generalisation.",
            catch="print the gap as a number under the plot. A reader should not have to measure a distance on a chart with their eye."),
        prompt(
            label="the two curves",
            input="the recorded history",
            output="loss against epoch, and both accuracies against epoch",
            constraint="both accuracy curves on the SAME axis — the gap is the quantity being shown",
            left_open="that the gap is the overfitting and you have seen it before. It is not a new phenomenon and it does not have a neural-network-specific cure.",
            student="plotting only the loss, which falls smoothly and forever and says nothing about generalisation.",
            catch="print the gap as a number under the plot. A reader should not have to measure a distance on a chart with their eye."),
        code('''
fig, axes = plt.subplots(1, 2, figsize=(12, 3.6))
axes[0].plot(range(1, 21), hist["loss"], marker="o", ms=4)
axes[0].set_xlabel("epoch"); axes[0].set_ylabel("training loss")
axes[1].plot(range(1, 21), hist["train_acc"], label="training")
axes[1].plot(range(1, 21), hist["val_acc"], label="validation")
axes[1].set_xlabel("epoch"); axes[1].set_ylabel("accuracy"); axes[1].legend()
plt.tight_layout(); plt.show()

gap = 100 * (hist["train_acc"][-1] - hist["val_acc"][-1])
print(f"the two curves finish {gap:.1f} accuracy points apart")
'''),
        md("""
The gap is the overfitting, and you have seen it before — it is Lecture 2's
train-versus-cross-validation table drawn as two lines. Nothing new about
neural networks makes it go away.
"""),

        md("""
## 8 · Tune the architecture by hand

Five architectures, then five learning rates, on 6,000 images and 8 epochs so
that ten fits are affordable. This is grid search done with your hands, which is
exactly what it feels like — and exactly why the next lecture automates it.

⏱ **about 2 minutes.**
"""),
        prompt(
            label="⏱ 2 min — five architectures, by hand",
            input="one hidden layer of 30, 100, 300; then two layers; then three",
            output="validation accuracy, parameter count and seconds for each",
            constraint="report the PARAMETER COUNT beside the accuracy — a third layer that buys nothing while costing parameters is a different finding from one that buys nothing while costing nothing",
            left_open="that this is grid search done with your hands, which is exactly what it feels like — and exactly why the next lecture automates it.",
            student="running each architecture on the full 12,000 and spending twenty minutes. 6,000 images and 8 epochs is a stated compromise that makes ten fits affordable, and the comparison is still fair because it is the same for every row.",
            catch="depth buys less than you expected. One hidden layer to two is worth a point or so; a third is worth roughly nothing here."),
        prompt(
            label="⏱ 2 min — five architectures, by hand",
            input="one hidden layer of 30, 100, 300; then two layers; then three",
            output="validation accuracy, parameter count and seconds for each",
            constraint="report the PARAMETER COUNT beside the accuracy — a third layer that buys nothing while costing parameters is a different finding from one that buys nothing while costing nothing",
            left_open="that this is grid search done with your hands, which is exactly what it feels like — and exactly why the next lecture automates it.",
            student="running each architecture on the full 12,000 and spending twenty minutes. 6,000 images and 8 epochs is a stated compromise that makes ten fits affordable, and the comparison is still fair because it is the same for every row.",
            catch="depth buys less than you expected. One hidden layer to two is worth a point or so; a third is worth roughly nothing here."),
        code('''
SMALL = 6_000
archs = [(30,), (100,), (300,), (300, 100), (300, 200, 100)]

arch_rows = []
for h in archs:
    c, r = train_curve(X_fit[:SMALL], y_fit[:SMALL], hidden=h, epochs=8)
    p = sum(w.size for w in c.coefs_) + sum(b.size for b in c.intercepts_)
    arch_rows.append((h, r["val_acc"][-1], p, r["seconds"]))
    print(f"{str(h):18s} val {r['val_acc'][-1]:.4f}   {p:>8,} params   "
          f"{r['seconds']:5.1f} s")

best_arch = max(arch_rows, key=lambda t: t[1])
print(f"\\nbest: {best_arch[0]} at {best_arch[1]:.4f}")
'''),
        prompt(
            label="five learning rates",
            input="1e-4 to 1e-2, same architecture",
            output="validation accuracy at each, and the spread between best and worst",
            constraint="report the WORST as well as the best — the spread is the finding",
            left_open="the reading to write down: the learning rate matters more than the architecture. One badly chosen scalar loses more accuracy than any of the structural choices gains.",
            student="tuning the architecture carefully and leaving the learning rate at its default. The default is fine here and this cell is what tells you how much luck that was.",
            catch="log-spaced, not linear. Learning rates live on a multiplicative scale, and a linear grid from 1e-4 to 1e-2 spends most of its points in a region where nothing changes."),
        prompt(
            label="five learning rates",
            input="1e-4 to 1e-2, same architecture",
            output="validation accuracy at each, and the spread between best and worst",
            constraint="report the WORST as well as the best — the spread is the finding",
            left_open="the reading to write down: the learning rate matters more than the architecture. One badly chosen scalar loses more accuracy than any of the structural choices gains.",
            student="tuning the architecture carefully and leaving the learning rate at its default. The default is fine here and this cell is what tells you how much luck that was.",
            catch="log-spaced, not linear. Learning rates live on a multiplicative scale, and a linear grid from 1e-4 to 1e-2 spends most of its points in a region where nothing changes."),
        code('''
lr_rows = []
for lr in [1e-4, 3e-4, 1e-3, 3e-3, 1e-2]:
    c, r = train_curve(X_fit[:SMALL], y_fit[:SMALL], lr=lr, epochs=8)
    lr_rows.append((lr, r["val_acc"][-1]))
    print(f"lr={lr:<8g} val {r['val_acc'][-1]:.4f}")

best_lr = max(lr_rows, key=lambda t: t[1])
print(f"\\nbest: lr={best_lr[0]:g} at {best_lr[1]:.4f}")
print(f"worst: {min(r[1] for r in lr_rows):.4f} — "
      f"{100 * (best_lr[1] - min(r[1] for r in lr_rows)):.1f} points below")
'''),
        md("""
Two readings, and the second is the one to write down.

1. Depth buys less than you expected. Going from one hidden layer to two is
   worth a point or so; a third layer is worth roughly nothing here.
2. **The learning rate matters more than the architecture.** One badly chosen
   scalar loses more accuracy than any of these structural choices gains.
"""),

        md("""
## 9 · Where does it go wrong?

An accuracy is one number over ten classes. Split it.
"""),
        prompt(
            label="where does it go wrong",
            input="the test predictions",
            output="per-class recall, sorted, and where the worst class goes",
            constraint="sort by recall and show the DESTINATIONS of the worst class — a confusion matrix printed whole is 100 numbers nobody reads",
            left_open="that the confusions are not random. The classes that get mixed up are the ones that are hard for a PERSON looking at a 28×28 thumbnail — which is a useful thing to tell the operator and is invisible in the headline.",
            student="reporting the accuracy and stopping. An accuracy is one number over ten classes, and the operator is going to ask which queue is unreliable.",
            catch="normalise the row before reading it. Raw counts and shares tell different stories, and only one of them answers 'when this class is wrong, where does it go'."),
        prompt(
            label="where does it go wrong",
            input="the test predictions",
            output="per-class recall, sorted, and where the worst class goes",
            constraint="sort by recall and show the DESTINATIONS of the worst class — a confusion matrix printed whole is 100 numbers nobody reads",
            left_open="that the confusions are not random. The classes that get mixed up are the ones that are hard for a PERSON looking at a 28×28 thumbnail — which is a useful thing to tell the operator and is invisible in the headline.",
            student="reporting the accuracy and stopping. An accuracy is one number over ten classes, and the operator is going to ask which queue is unreliable.",
            catch="normalise the row before reading it. Raw counts and shares tell different stories, and only one of them answers 'when this class is wrong, where does it go'."),
        code('''
from sklearn.metrics import confusion_matrix

pred = clf.predict(X_test)
cm = confusion_matrix(y_test_, pred)
recall = cm.diagonal() / cm.sum(axis=1)

for c in np.argsort(recall):
    print(f"{CLASSES[c]:12s} recall {recall[c]:.3f}")

worst = int(np.argmin(recall))
row = cm[worst] / cm[worst].sum()
print(f"\\n'{CLASSES[worst]}' is the hard class. Where does it go?")
for c in np.argsort(-row)[:4]:
    print(f"   -> {CLASSES[c]:12s} {row[c]:.3f}")
'''),
        md("""
The confusions are not random: the classes that get mixed up are the ones that
are hard for a *person* looking at a 28×28 thumbnail. That is a useful thing to
be able to tell the operator, and it is not visible in the headline accuracy.
"""),

        md("""
## 10 · Three things you cannot do

The model works. Now try to change it.
"""),
        prompt(
            label="three things you cannot do",
            input="the fitted model",
            output="the TypeError from changing the loss, the absence of any gradient attribute, and `partial_fit`'s signature",
            constraint="demonstrate each wall by running into it — a list of limitations in prose is an opinion, a caught TypeError is not",
            left_open="that these are not accidents of this library. They are what happens when the training loop is written for you.",
            student="assuming a hidden argument exists and searching the documentation for an hour. `dir(clf)` for anything containing 'grad' answers it in one line.",
            catch="one call to `partial_fit` is one full pass over everything you hand it. There is no smaller unit of control and no hook between forward and backward — that sentence is the whole motivation for PyTorch."),
        code('''
# 1 — change the objective
try:
    MLPClassifier(hidden_layer_sizes=(300, 100), loss="mae")
except TypeError as exc:
    print("changing the loss:", exc)

# 2 — look at a gradient
grads = [a for a in dir(clf) if "grad" in a.lower()]
print(f"\\nattributes containing 'grad': {grads}")

# 3 — stop part-way through an epoch
import inspect
print("\\npartial_fit's signature:",
      inspect.signature(MLPClassifier.partial_fit))
print("one call = one full pass over everything you hand it. There is no")
print("smaller unit of control, and no hook between forward and backward.")
'''),
        prompt(
            label="what the full run would cost",
            input="the measured seconds per epoch",
            output="the extrapolation to the deck's 55,000-image run",
            constraint="extrapolate from a MEASURED number and say it is linear scaling on this CPU — not a figure quoted from anywhere",
            left_open="that there is no `device=` argument to move any of it to a GPU. Scikit-learn is CPU-only by design and says so in its own FAQ.",
            student="assuming the gap between this notebook and the deck is a mistake. The notebook trains on 12,000 of 55,000 so it finishes in the hour, every cell says so, and the difference is itself worth a sentence in your notes.",
            catch="when you subsample for time, state the factor and extrapolate out loud. A reader comparing your number with a published one needs to know which they are holding."),
        prompt(
            label="what the full run would cost",
            input="the measured seconds per epoch",
            output="the extrapolation to the deck's 55,000-image run",
            constraint="extrapolate from a MEASURED number and say it is linear scaling on this CPU — not a figure quoted from anywhere",
            left_open="that there is no `device=` argument to move any of it to a GPU. Scikit-learn is CPU-only by design and says so in its own FAQ.",
            student="assuming the gap between this notebook and the deck is a mistake. The notebook trains on 12,000 of 55,000 so it finishes in the hour, every cell says so, and the difference is itself worth a sentence in your notes.",
            catch="when you subsample for time, state the factor and extrapolate out loud. A reader comparing your number with a published one needs to know which they are holding."),
        code('''
per_epoch = hist["seconds"] / 20
print(f"measured: {per_epoch:.1f} s per epoch on {SUB:,} images, on this CPU")
print(f"the deck's full run is {55_000 / SUB:.1f}x the data")
print(f"scaling linearly, 20 epochs on 55,000 would be about "
      f"{per_epoch * 20 * 55_000 / SUB / 60:.1f} minutes")
print("\\nand there is no device= argument to move any of it to a GPU:")
print("Scikit-Learn is CPU-only by design, and says so in its own FAQ.")
'''),

        md("""
## 11 · Where we are

You have a working image classifier, hand-tuned, measured against a 10%
baseline, with an error analysis.

You also have three walls, and they are not accidents of this library — they are
what happens when the loop is written for you:

| you want to | you cannot |
|---|---|
| change the objective | there is no argument for it |
| see a gradient | nothing is exposed |
| stop mid-epoch, or log per batch | the smallest unit is one pass |
| use a GPU | there is no device to move to |

Write your **best validation accuracy** on the same sheet of paper, next to what
you predicted. Bring it to the next lecture; we open by comparing them.

Do not fix anything yet.
"""),
    ]
