# Lecture 11 — The first neural network

**Fashion MNIST · scikit-learn `MLPClassifier` · Géron Ch 9**

The script for rebuilding this notebook at a Colab keyboard, by prompting. Fifteen
code cells, in order. Type the prompt, read what comes back, check it against
**Expect**, then run.

---

## Before you start

**Runtime.** Runtime → Change runtime type → **CPU**. Selecting a GPU changes
nothing: scikit-learn does not use one, and the last cell of the notebook is
about exactly that. A GPU runtime will produce the same wall-clock numbers.

**Total compute.** About **4½ minutes** of fitting, in four blocks (≈40 s, ≈40 s,
≈60 s, ≈2 min), plus a **~31 MB** download the first time. Those four figures are
the ones the current notebook states; see *What I verified* below — none of them
is backed by a stored output anywhere in the course, so treat them as the
author's estimate and read the seconds your own cells print.

**Seed.** 42 everywhere. Every figure quoted in this script was re-derived with
`python3` against `notebooks/datasets/FashionMNIST` under that seed.

### What I verified, and what I could not

Verified by running: every shape, dtype and pixel range; both class-balance
counts; the 55,000/5,000/10,000 split under `default_rng(42)`; the class
composition of the 12,000-image subsample; the majority class and the exact
0.1000 baseline; that the raw and scaled feature matrices are the *same rows*;
every parameter count; all three "walls"; the exact text of the suppressed
warning; and `partial_fit`'s real behaviour when `classes=` is omitted.

**Not verified:** every accuracy, every training loss, every recall, and every
wall-clock second. Those require running the training cells, which this task
forbids. Where this script names an expected *direction* rather than a number, it
says so and marks it as unconfirmed.

### Reading order for the traps

Two cells in this notebook are meant to catch you. **Cell 7** is one of them.
Nothing before Cell 7 in this script tells you what is wrong with it — that is
deliberate, and it is a repair of the current notebook, which announces the same
defect five times before the reader reaches it. Run Cell 7, write the number it
prints on paper, and only then read its annotation.

---

## Cell 1 — setup

**Prompt to type:**

> Setup cell. Import numpy, sklearn, torch, torchvision, matplotlib.pyplot, and
> sys, time, warnings. Print the python, scikit-learn, torch and torchvision
> versions. Set a seed of 42 in a constant called RANDOM_STATE and seed numpy
> with it.

**Expect:** four version lines. `torch` and `torchvision` are imported but only
`torchvision.datasets` is used all lecture — add a comment saying so, or the
import reads as a promise the notebook never keeps.
**Assert:** none.
**Annotate:** short

---

## Cell 2 — load Fashion MNIST

**Prompt to type:**

> Load Fashion MNIST from torchvision into numpy arrays — train images, train
> labels, test images, test labels — into a folder called `datasets`. Keep the
> pixels as uint8. Print the shapes, the ten class names, and the range of pixel
> values.

**Expect:** `(60000, 28, 28)` and `(10000, 28, 28)`, dtype `uint8`, pixel values
running **0 to 255**, and the ten names in label order:
`T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot`.
**Assert:** both shapes **and** the dtype —
`assert X_train_full_u8.dtype == np.uint8`. The shape assert alone passes on a
float array that has already been divided by something.
**⏱** ~31 MB of gzipped IDX files on first run (26.4 MB train images, 4.4 MB
test images, ~65 kB labels), a few seconds on Colab; instant afterwards. It
expands to 82 MB on disk.
**Annotate:** full

- **Left open:** which of the two loading paths you get. `.data` on the dataset
  object hands you the raw uint8 tensor; passing `transform=` hands you something
  else entirely. The prompt says "uint8", which is the only reason the answer is
  determined.
- **The usual student version:** `FashionMNIST(..., transform=transforms.ToTensor())`
  — the idiom in every torchvision tutorial. `ToTensor()` is documented to convert
  a PIL image in [0, 255] to a FloatTensor in **[0.0, 1.0]** and to reorder to
  (C, H, W). Take that path and the pixels are silently already scaled, the arrays
  are (N, 1, 28, 28) rather than (N, 28, 28), and Cells 7 and 8 of this notebook
  have nothing left to demonstrate. The scaling in this lecture must be visible,
  so it is done by hand in Cell 5.
- **How you would catch it:** print `X.min(), X.max()` and the dtype on the same
  line as the shape. `uint8 0–255` and `float32 0.0–1.0` are the two states this
  whole lecture is about, and a shape assert distinguishes neither.

---

## Cell 3 — look at it before you model it

**Prompt to type:**

> Show three examples of each of the ten classes as a 3 by 10 grid of greyscale
> thumbnails, one class per column, with the class name as the column title. No
> ticks.

**Expect:** a 3×10 grid, columns titled left to right with the ten names in the
order printed by Cell 2. Ask for `vmin=0, vmax=255` if it is not there: without
it every thumbnail is normalised to its own range and a dark garment and a bright
one look identical.
**Assert:** none — this is a looking cell.
**Annotate:** short

Two things are worth writing down here, and only one of them survives checking.
Several classes are the same garment photographed the same way (Pullover, Coat
and Shirt are the ones the error analysis in Cell 13 comes back to). The second —
"the background is exactly zero" — is looser than it sounds: **50.2%** of all
training pixels are exactly 0, and the top-left pixel is 0 in **99.98%** of
images, but only **0.03%** of images have an entirely zero one-pixel border. The
garments are cropped to fill the frame. Say "half the pixels are exactly zero",
which is checkable, rather than "the background is zero", which is not.

---

## Cell 4 — is it balanced?

**Prompt to type:**

> Count how many training images there are of each class, print the count next to
> each class name, and assert that the classes are exactly balanced.

**Expect:** ten lines, **6,000** against every class.
**Assert:** `assert counts.min() == counts.max() == 6000` — exact, not a
tolerance. The data allows an exact assert here, so take it.
**Annotate:** short

That one line decides the metric for the rest of the lecture. Lecture 4 spent an
hour on why accuracy is worthless under imbalance; here the classes are balanced
by construction, so accuracy is meaningful *and* the trivial baseline is
computable in your head. Check, then choose — do not carry either habit over
uninspected.

---

## Cell 5 — scale, and split

**Prompt to type:**

> Flatten the images to 784 columns of float32 and divide by 255. Using a numpy
> Generator seeded with 42, take a permutation of the 60,000 training images, use
> the first 5,000 as validation and the rest to fit on. Do the same to the test
> images. Print the three sizes. Then take the first 12,000 of the fit set as the
> subset we actually train on, and call it SUB.

**Expect:** `fit 55,000   val 5,000   test 10,000`, then a line saying it trains
on the first **12,000** of them. Every column in [0, 1].
**Assert:** four of them, and all four are worth the line —

```python
assert len(X_fit_full) + len(X_val) == 60_000
assert set(fit_idx).isdisjoint(val_idx)
assert X_fit_full.shape[1] == 784
assert 0.0 <= X_fit_full.min() and X_fit_full.max() <= 1.0
```

**Annotate:** full

- **Left open:** why scaling matters for a *network* specifically, as opposed to
  for a tree, which does not care. The size of a gradient step is set once for
  every weight; inputs two orders of magnitude apart make one step too large for
  some weights and too small for others. Nothing in the prompt says this, and
  Cell 7 is what happens when nobody says it.
- **The usual student version:** two of them, both real. `train_test_split(X, y,
  test_size=5000, random_state=42)` — which shuffles by default but does **not**
  stratify by default, so the validation set is balanced only by luck. And
  `StandardScaler`, reached for out of habit: it is a *fitted* transform, it has
  to live inside a `Pipeline` to avoid leaking, and it would centre the zero
  background off zero. Dividing by a known constant is a different operation with
  a different risk profile — there is nothing to fit, so there is nothing to leak.
- **How you would catch it:** the disjointness assert. A permutation sliced in two
  cannot overlap, so the assert can never fire today — and it costs nothing and
  catches the day somebody changes the slicing to `order[:5000]` and
  `order[:55000]`.

**One thing this cell hides.** The 60,000 are exactly balanced; the 12,000 slice
is not. Class counts in `y_fit` run from **1,170** (Bag) to **1,246** (Sandal).
That does not affect the baseline in Cell 6 — the *test* set is exactly balanced,
1,000 per class, and that is what the baseline is scored on — but "perfectly
balanced" in Cell 4 and "the first 12,000" in Cell 5 are not the same claim.

---

## Cell 6 — a number to compare against

**Prompt to type:**

> Predict the commonest class in the training subset for every test image and
> print the accuracy. Assert it is exactly 0.10.

**Expect:** `always predict 'Sandal'  ->  accuracy 0.1000`. **Sandal** is the
majority of the 12,000 subsample (1,246 of them), and the accuracy is exactly
0.1 because the test set holds exactly 1,000 of each of the ten classes — you
can work that out on paper before running it.
**Assert:** `assert abs(baseline_acc - 0.10) < 1e-9`. It passes exactly: 1000 /
10000 is representable.
**Annotate:** short

This is the anchor, and it is also why accuracy is readable at all here. A number
is only informative relative to what it has to beat.

---

## Cell 7 — ★ Commit, then train

**On paper first, before you type anything.** Not in the notebook — on paper,
where you cannot quietly revise it.

```
Metric:                                          ____________
Accuracy a good sorting machine would need:      ____________ %
Accuracy I expect from the model I build today:  ____________ %
```

**Prompt to type** — exactly this, no more:

> Train a neural network to classify Fashion MNIST images and print the accuracy.

Take what comes back. Do not tidy it, do not add arguments to it. Wire it to the
arrays you already have — the same 12,000 fit rows and the same 5,000 validation
rows as Cell 5, but taken from `X_train_full_u8` so they arrive as they came off
disk.

**Expect:** one line — `validation accuracy 0.XXXX   (NN s)`. It imports nothing
exotic, it raises nothing, and it prints a believable number.
**Write the four decimals on paper now, next to what you predicted.** Then read
the annotation.
**Assert:** **none.** Nothing in this cell checks anything, and nothing in the
output says what did not happen. That is the whole exhibit.
**⏱** ≈40 s on a free Colab CPU for 12 epochs on 12,000 images with hidden layers
(300, 100). *(The current notebook's figure; no stored output confirms it.)*
**Annotate:** full

- **Left open:** the input range, and therefore the learning rate. Two defaults
  are running this cell and only one of them is yours. `max_iter=12` you chose.
  `learning_rate_init=0.001` you did not — it is scikit-learn's, verified as the
  `MLPClassifier.__init__` default in 1.7.2, and it is the default **for inputs of
  order one**. You handed the network integers up to 255. A third default is also
  in play: `batch_size='auto'` resolves to `min(200, n_samples)` = **200**, which
  Cell 9 will change to 128 without saying so.
- **The usual student version:** reading the accuracy and moving on. This is not a
  hypothetical — it is what the prompt makes overwhelmingly likely. The prompt
  never said "scale the pixels". The output never said it had not happened. The
  number is high enough to look like a result, and there is no failure to
  investigate. Nothing in the loop from specify to verify is *broken*; the
  specification was silent, and silence is not visible in an output.
- **How you would catch it:** you cannot, from this cell. There is one thing that
  would have caught it and it is one line above the fit:
  `assert X.max() <= 1.0, X.max()`. That is the general shape of it — **a library
  default is chosen for a typical input range, and when your inputs are not in
  that range the default stops being a default and becomes a mistake with a
  plausible value.** Note also that the cell suppresses a warning, and the warning
  is no help: the exact text is `ConvergenceWarning: Stochastic Optimizer:
  Maximum iterations (12) reached and the optimization hasn't converged yet.` It
  is about `max_iter`, which is yours and which you stated. It says nothing about
  the pixel range, which is not yours and which nobody stated. Un-suppressing it
  does not save you.

---

## Cell 8 — ⚠ change one thing

**Prompt to type:**

> Fit the same architecture with the same number of epochs and the same seed, but
> on the scaled pixels this time. Print both validation accuracies, the difference
> in accuracy points, and both models' final training loss.

**Expect:** four lines — `pixels 0-255`, `pixels 0-1`, the signed gap in accuracy
points, and the two `loss_` values side by side. Both accuracies are computed on
**the same 5,000 validation rows** — I checked this: the raw validation matrix is
exactly 255× the scaled one, element for element, with zero maximum absolute
difference, and the same holds for the 12,000 fit rows. So the difference is
attributable to the input scale and to nothing else.
**Assert:** add the one the notebook does not have —
`assert acc_scaled > acc_raw` — and add
`assert clf_raw.loss_ > clf_scaled.loss_`. *(Direction is the notebook's claim;
I could not execute it to confirm. If either assert fires, that is a finding, not
a bug in this script.)*
**⏱** ≈40 s — a second 12-epoch fit on the same 12,000 images, so the same budget
as Cell 7. **The current notebook states no time for this cell at all.**
**Annotate:** short

Read the loss column, not just the accuracy. A far higher final training loss on
the raw model says the optimiser never got going — which is a different diagnosis
from "the problem is hard", and only the loss column separates them.

Now write the corrected specification down, because it is the deliverable of this
section:

> *"Load Fashion MNIST. Scale the pixels to [0, 1] as float32. Split off 5,000
> validation images with a fixed seed. Train an MLP with hidden layers (300, 100)
> for 12 epochs, seed 42, and report accuracy on the validation set — not the
> training set. State the wall-clock time."*

Input, output, constraint, check. The vague prompt in Cell 7 was not *wrong*
about what to build. It was silent about the conditions under which the thing
works, and an assistant fills silence with defaults.

---

## Cell 9 — one epoch at a time

**Prompt to type:**

> Write a function that trains an MLPClassifier one epoch at a time with
> `partial_fit` instead of `fit`, so I can record the training loss, the
> validation accuracy and the training accuracy after every epoch. Arguments for
> the hidden layers, the learning rate, the number of epochs and the batch size.
> Return the model and the history. Run it for 20 epochs on the 12,000, and print
> the parameter count and the seconds.

**Expect:** `266,610 parameters`, then the seconds and the seconds per epoch, then
the final validation accuracy. **Check the parameter count on paper before you
run it:** 784×300 + 300 + 300×100 + 100 + 100×10 + 10 = 235,200 + 300 + 30,000 +
100 + 1,000 + 10 = **266,610**. If the cell prints anything else, the architecture
is not the one you asked for.
**Assert:** `assert len(hist["loss"]) == 20` and `assert n_params == 266_610`.
**⏱** ≈60 s stated. Treat that as low: the loop also calls `score` twice per
epoch, on 5,000 validation and 5,000 training rows, so 20 epochs carry **200,000
extra forward passes** through a 266,610-parameter network on top of the training
itself. Print the measured number; the cell does.
**Annotate:** full

- **Left open:** that this is as close to the inside of the training loop as
  scikit-learn will let you get. One call to `partial_fit` is one complete pass
  over everything you hand it. There is no smaller unit and no hook between
  forward and backward. Remember that — it is the entire motivation for the next
  lecture.
- **The usual student version:** omitting `classes=`. The real behaviour, verified
  against scikit-learn 1.7.2, is **the opposite of what you would guess**: the
  *first* call without `classes=` raises `ValueError: classes must be passed on
  the first call to partial_fit.`, and every call *after* the first is happy
  without it. So the failure is loud and immediate, which is the good case. The
  quiet case is the one worth naming: pass `classes=np.arange(10)` and a first
  batch that happens to be missing class 7, and the model is correctly built with
  ten outputs — I checked, `classes_` comes back as `[0 … 9]` and `n_outputs_` is
  10. Omit it and you never get that far. `classes=` is not insurance against a
  second call; it is what fixes the output layer's width before any data has been
  seen.
- **How you would catch it:** record training **and** validation accuracy every
  epoch, not just the loss. The gap between the two lines is Lecture 2's
  train-versus-cross-validation table drawn as two curves, and nothing about
  neural networks makes it go away. A loss curve alone falls smoothly and forever
  and tells you nothing about generalisation.

---

## Cell 10 — the two curves

**Prompt to type:**

> Two plots side by side from that history: training loss against epoch on the
> left, and training and validation accuracy against epoch on the right, both on
> the same axis with a legend. Then print how far apart the two accuracy curves
> finish, in accuracy points.

**Expect:** two panels, x-axis 1 to 20, the right-hand panel carrying two labelled
lines, and a printed number underneath giving the final gap.
**Assert:** none.
**Annotate:** short

Both accuracy curves must share one axis — the gap *is* the quantity being shown,
and two panels turn it into a task for the reader's eye. Print it as a number
underneath as well: nobody should have to measure a distance on a chart.

---

## Cell 11 — five architectures, by hand

**Prompt to type:**

> Using that function, try five architectures — one hidden layer of 30, then 100,
> then 300, then (300, 100), then (300, 200, 100) — on 6,000 images for 8 epochs
> each. Print the validation accuracy, the parameter count and the seconds for
> each, then the best.

**Expect:** five rows. The parameter counts are fixed by the architecture and you
can check every one of them on paper before running — they are 784→h→…→10 with
biases:

| hidden | parameters |
|---|---|
| (30,) | 23,860 |
| (100,) | 79,510 |
| (300,) | 238,510 |
| (300, 100) | 266,610 |
| (300, 200, 100) | 316,810 |

**Assert:** `assert [row[2] for row in arch_rows] == [23_860, 79_510, 238_510, 266_610, 316_810]`
**⏱** ≈1 min — five fits of 8 epochs on 6,000 images. Half the data and under
half the epochs of Cell 9, deliberately, so that ten fits are affordable inside
the hour. All five rows use the same 6,000 rows and the same 5,000 validation
rows, so the comparison is fair even though every number in it is lower than it
would be on the full set.
**Annotate:** short

Report the parameter count *beside* the accuracy. A third layer that buys nothing
while costing 50,200 parameters is a different finding from one that buys nothing
while costing nothing, and the accuracy column alone cannot tell you which you
have.

---

## Cell 12 — five learning rates

**Prompt to type:**

> Same thing for five learning rates from 1e-4 to 1e-2, log-spaced, on the same
> 6,000 images and 8 epochs, architecture (300, 100). Print the accuracy at each,
> the best, and the worst, and how many points separate them.

**Expect:** five rows for `1e-4, 3e-4, 1e-3, 3e-3, 1e-2`, then a best line and a
worst line with the spread in accuracy points. Same 6,000 rows and same 5,000
validation rows as Cell 11 — one variable moves.
**Assert:** none. Assert instead that the grid is what you asked for:
`assert [r[0] for r in lr_rows] == [1e-4, 3e-4, 1e-3, 3e-3, 1e-2]`.
**⏱** ≈1 min, five more fits at the same size as Cell 11.
**Annotate:** full

- **Left open:** which axis the grid should live on. Log-spaced, not linear.
  Learning rates are multiplicative, and a linear grid from 1e-4 to 1e-2 — 0.0001,
  0.0025, 0.005, 0.0075, 0.01 — spends four of its five points in the top decade
  and never visits the bottom two, where the interesting failure is.
- **The usual student version:** tuning the architecture carefully and never
  touching the learning rate. This is the exact shape of Cell 7's failure and it
  is worth seeing twice: `learning_rate_init=0.001` is the scikit-learn default,
  it is the middle point of this grid, and on scaled pixels it is roughly the
  right answer. Leaving it alone is not wrong here — but until you have run this
  cell you have no idea how much of that was judgement and how much was luck.
  Note also that `MLPClassifier` has a *second* parameter called
  `learning_rate` (default `'constant'`), and it is documented to be used **only
  when `solver='sgd'`**. Set it with `solver='adam'` and it silently does nothing.
- **How you would catch it:** report the **worst** row, not only the best. The
  spread is the finding. A table that prints only its maximum cannot tell you
  whether the hyperparameter matters, and "the learning rate matters more than the
  architecture" is a claim about a spread, which you can only make if you printed
  one.

Write two readings down. Compare the spread here against the spread in Cell 11's
accuracy column: one badly chosen scalar against five structural choices. *(The
current notebook states the outcome of this comparison in prose before the cells
run. Do not read those two paragraphs until you have your own two numbers.)*

---

## Cell 13 — where does it go wrong?

**Prompt to type:**

> Predict the test set with the 20-epoch model, build the confusion matrix, and
> print the per-class recall sorted worst first. Then take the worst class,
> normalise its row, and print the four classes its images most often end up in.

**Expect:** ten lines of `class  recall 0.XXX` ascending, then four destination
lines with shares that are fractions of that class's 1,000 test images.
**Assert:** two you can work out in advance —
`assert cm.sum() == 10_000` and `assert (cm.sum(axis=1) == 1000).all()`. Both hold
because the test set is exactly 1,000 per class, verified.
**Annotate:** short

Normalise the row before reading it. Raw counts and row shares tell different
stories and only the shares answer "when this class is wrong, where does it go" —
which is the question the operator in the brief will actually ask. A confusion
matrix printed whole is 100 numbers nobody reads.

Then go back to the grid in Cell 3 and look at the classes that trade places. The
confusions are not random: they are the ones you would make yourself from a 28×28
thumbnail. That is a sentence you can give the operator, and it is invisible in
the headline accuracy.

---

## Cell 14 — three things you cannot do

**Prompt to type:**

> Show me three limits of MLPClassifier by hitting them, not by describing them:
> try to construct one with a different loss function, list any attribute on the
> fitted model whose name contains "grad", and print `partial_fit`'s signature.

**Expect:** all three verified against scikit-learn 1.7.2 —

1. `TypeError: MLPClassifier.__init__() got an unexpected keyword argument 'loss'`
2. `attributes containing 'grad': []` — an empty list
3. `(self, X, y, sample_weight=None, classes=None)`

**Assert:** none needed; the `TypeError` is the assertion. Wrap it in
`try/except TypeError` so the cell completes and the notebook still runs top to
bottom.
**Annotate:** full

- **Left open:** that none of this is an accident of one library. These are what
  happens when the training loop is written for you: an objective you did not pick
  is compiled in, the intermediate quantities are not surfaced because nothing was
  going to ask for them, and the smallest addressable unit is one whole pass. That
  is the price of `fit()` being one line.
- **The usual student version:** assuming a hidden argument exists and reading the
  documentation for an hour looking for it. All three of these are one line each,
  and the third is the one worth internalising: the signature has four parameters
  and none of them is a callback, a step count, or a batch index.
- **How you would catch it:** `[a for a in dir(clf) if "grad" in a.lower()]`
  returns `[]`, and an empty list is a real answer, not a failed search. The
  reason the question is worth asking at all is Lecture 13: when a twenty-layer
  network refuses to learn, per-layer gradient statistics are the diagnosis, and
  this is the cell that tells you the diagnosis is unavailable here.

---

## Cell 15 — what the full run would cost, and the fourth wall

**Prompt to type:**

> From the measured seconds per epoch in the history, work out what 20 epochs on
> the full 55,000 fit images would cost on this CPU, assuming it scales linearly.
> Print the measured figure, the data ratio, and the extrapolation in minutes.
> Then say whether any of it can be moved to a GPU.

**Expect:** four lines. The data ratio is fixed and checkable: 55,000 / 12,000 =
4.5833, printed as **4.6×**. The minutes figure is `per_epoch × 20 × 4.5833 / 60`
and depends on your machine. Then: there is no `device=` argument anywhere in
scikit-learn. It is CPU-only by design and says so in its own FAQ.
**Assert:** `assert abs(55_000 / SUB - 4.5833) < 1e-3`
**Annotate:** full

- **Left open:** what exactly is being extrapolated. `hist["seconds"]` is wall
  clock for the whole loop, and that loop scored 10,000 rows on every one of its
  20 epochs. Scoring cost is roughly *fixed* as the training set grows; training
  cost scales with it. So multiplying the whole measured time by 4.58
  over-estimates, and the sentence "assuming it scales linearly" is doing more
  work than it looks. Say which component you assumed linear.
- **The usual student version:** assuming the gap between this notebook's number
  and the deck's is a mistake to be reconciled. It is not. This notebook trains on
  **12,000 of 55,000** so that a free CPU runtime finishes inside the lecture hour;
  every cell that does so says so. The gap is the subsampling factor, and knowing
  which of the two numbers you are holding is the point.
- **How you would catch it:** when you subsample for time, state the factor and
  do the extrapolation out loud, in the notebook, from a number you measured. A
  reader comparing your figure against a published one needs to know which they
  have. "About five minutes on this CPU" is a sentence a stakeholder can act on;
  "it is slow" is not.

---

## Where we are

You have a working image classifier, hand-tuned, measured against an exactly
10.00% baseline, with an error analysis. You also have **four** walls, three of
them demonstrated in Cell 14 and the fourth in Cell 15:

| you want to | you cannot |
|---|---|
| change the objective | there is no argument for it |
| see a gradient | nothing is exposed |
| stop mid-epoch, or log per batch | the smallest unit is one pass |
| use a GPU | there is no device to move to |

**Write down one number, and be specific about which.** Write the validation
accuracy from **Cell 9** — 20 epochs, 12,000 images, hidden layers (300, 100) —
next to what you committed to before Cell 7. Not the best row of Cell 11 or 12:
those were fitted on 6,000 images for 8 epochs, and comparing them against a
next-lecture PyTorch model trained on a different budget measures the budget.

Do not fix anything yet.

---

## Re-run map

**Cold start, in order:** 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8 → 9 → 10 → 11 → 12 → 13
→ 14 → 15. Every cell constructs its own model, so re-running any of them is
safe: none of them continues training an existing network.

Specific dependencies, if you want to re-run part of it:

- **Cell 8 alone** needs `acc_raw` and `clf_raw` from Cell 7 still in memory. If
  the kernel restarted, run 1 → 2 → 5 → 7 → 8. Cells 3, 4 and 6 are not required.
- **Cell 10 alone** needs `hist` from Cell 9. Run 1 → 2 → 5 → 9 → 10.
- **Cells 13 and 15** both need `clf` and `hist` from Cell 9. Cells 11 and 12 do
  not touch either, so running the sweeps does not invalidate them — but only
  because the sweeps bind their models to a different name. If you rename anything
  in Cells 11–12, re-run Cell 9 before Cell 13.
- **To change the seed**, edit `RANDOM_STATE` in **Cell 1**, then re-run
  1 → 2 → 5 → 6 → 7 → 8 → 9 → 10 → 11 → 12 → 13 → 15. The split in Cell 5 changes,
  so the baseline's majority class in Cell 6 may stop being Sandal — its accuracy
  will still be exactly 0.1000, because the *test* set is balanced regardless of
  the seed.
- **To train on more than 12,000**, edit `SUB` in **Cell 5** and re-run
  5 → 7 → 8 → 9 → 10 → 13 → 15. Cost scales roughly with `SUB`: at the full 55,000
  Cell 9 alone runs about 4.6× longer. Cells 11 and 12 use their own `SMALL =
  6_000` and are unaffected.

---

## Defects found in the current notebook

`notebooks/lecture-11.ipynb`, 50 cells, 15 code cells. Everything below was
checked with `python3` against the notebook JSON, the dataset on disk, or
scikit-learn 1.7.2, except where marked **unverified**.

### Checked and confirmed

**1. §6.1 — every prompt box carries the full three-bullet annotation.**
Counted: **15 prompt boxes, 15 of them full**. The budget is five to eight and
never more than ten. This is the headline defect the guidelines exist to repair,
and it is the reason all three student readers stopped reading the template
around cell 30 of lecture 19.

**2. §6.2 — the `partial_fit` failure mode is stated backwards.** Cell 29 claims
omitting `classes=` "works on the first call and raises on the second, or worse,
silently fits a model that has never seen a class absent from the first batch."
Verified against scikit-learn 1.7.2: the **first** call without `classes=` raises
`ValueError: classes must be passed on the first call to partial_fit.`, and the
**second** call without it succeeds. The "silently fits" case is unreachable — you
cannot get past call one. Both halves of the bullet are wrong, and it is the
bullet the guidelines single out as the one that must be observed rather than
invented.

**3. §8.1 — the deliberate defect is announced five times before the reader can
fall into it.** Cell 0 ("Cells marked **⚠ read before running** contain a defect
on purpose"), cell 21 ("**⚠ Read before running**"), cell 22's box label ("⚠ what
the assistant returns"), cell 22's `constraint` ("feed it the **RAW** uint8
pixels"), and cell 22's `left_open`, which states the diagnosis in full —
`learning_rate_init=0.001` "is the default *for inputs of order ONE*. We handed
the network integers up to 255" — before the cell has run. Cell 24 then says the
same thing a sixth time, after. Nobody falls in, and "would you have caught it?"
has no honest answer. Grep confirmed ⚠ occurs in exactly cells 0, 21, 22, all
ahead of the defective cell 23.

**4. §3.3 — "The scaling happens two sections down."** Cell 5 sits inside
`## 2 · The brief` (heading at cell 4). Scaling is `## 3 · Scale, and split`
(heading at cell 13). That is **one** section down. Section headings verified by
parsing every `## N ·` in the file.

**5. §3.3 — "The previous application spent an hour on why accuracy is worthless
under imbalance"** (cell 10). Per `LECTURES.md`, that material is **Lecture 4**,
and lecture 11's previous application is the Lectures 9–10 pair on clustering and
dimensionality reduction. Cell 12, two cells later, gets it right and says
"Lecture 4". The notebook contradicts itself about the same reference within
three cells.

**6. §3.3 — "it warns; we come back to that" never comes back.** The comment is
in cell 23. The string "Convergence" appears in **no cell** of the notebook, and
no cell names or discusses the suppressed warning. I ran a 12-epoch fit to get its
exact text: `ConvergenceWarning: Stochastic Optimizer: Maximum iterations (12)
reached and the optimization hasn't converged yet.` This is worth coming back to
precisely because it is *not* about the defect — it warns about `max_iter`, which
the notebook chose and stated, and is silent about the pixel range, which it did
not.

**7. §3.3 — "Both come back in the error analysis" (cell 8) resolves half way.**
The two things named are "several classes are garments photographed the same way"
and "the background is exactly zero". The error analysis (cells 40–43) returns to
the first and never mentions the background at all.

**8. §1.2 / §7.3 — "the background is exactly zero" is not what the data says.**
Measured on the 60,000 training images: 50.21% of all pixels are exactly 0; the
top-left pixel is 0 in 99.98% of images; but only **0.03%** of images have an
entirely zero one-pixel border. The garments are cropped to fill the frame. The
first statistic supports the claim the notebook wants; the third contradicts the
sentence as written.

**9. §4.1 — the name `c` is bound to two different types across cells.** Integer
class index in cells 9, 11 and 42 (`for c in range(10)`, `for c, n in
enumerate(counts)`, `for c in np.argsort(recall)`); a fitted `MLPClassifier` in
cells 36 and 38 (`c, r = train_curve(...)`). The guidelines name this exact
failure and prescribe the fix — throwaway loop variables get throwaway names, and
a one-letter name for a trained model is the opposite of that. It happens to be
harmless today only because cells 36 and 38 never run in the same scope as a
class-index loop that outlives them.

**10. §7.1 — cell 26 is a full 12-epoch fit with no ⏱ marker anywhere above it.**
Cells 24, 25 and 26 contain no timing statement; the ⏱ in cell 21 belongs to
cell 23. The notebook's own header promises "Anything that takes more than a few
seconds says so". By its own estimate for the identical fit in cell 23, this one
costs about 40 seconds. Grep confirmed: no ⏱ between the section 6 heading and
cell 28.

**11. §7.3 — "three walls" over a four-row table.** Cell 49 says "You also have
three walls" and then prints a table with **four** rows. Section 10 is titled
"Three things you cannot do" and contains two code cells demonstrating four
things: cell 46 covers the objective, the gradient and the epoch; cell 48 covers
the GPU. A reader who counts finds four and cannot tell whether they miscounted.

**12. §7.2 — "Write your best validation accuracy" is ambiguous across five
cells, and the next lecture opens by comparing it.** The notebook prints a
validation accuracy in cells 23, 26, 30, 36 (five of them) and 38 (five more) —
twelve candidate numbers. The literal maximum will almost certainly come from a
sweep row fitted on **6,000** images for **8** epochs, while the model actually
carried into the error analysis (`clf`, cell 30) was fitted on **12,000** for
**20**. Lecture 12 then compares that number against a PyTorch model, and the
comparison is on unmatched training budgets — §2.1, one lecture downstream.

**13. §8.3 — "examinable" appears exactly once, in a code comment.** Cell 3's
`# Not examinable: engineering hygiene`. Sections 2 through 11 carry no marker of
any kind. The rule asks for one of *examinable* / *not examinable — engineering*
/ *beyond the book* on every section.

**14. §1.2 — no figure in this notebook can be reconciled with a stored output,
because there are none.** All 15 code cells have zero stored outputs and
`execution_count: null`. This is course-wide, not specific to lecture 11: of the
24 notebooks, only `lecture-19.ipynb` stores any output (21 of them). The
consequence for this file is concrete — the three ⏱ claims (40 s, 60 s, 2 min),
the "worth a point or so" depth claim in cell 39, and "the learning rate matters
more than the architecture" all rest on a run nobody can see.

**15. §9 — none of the six checks the guidelines say "must be added to the
tooling" exists.** `tools/check_all.py` runs decks, provenance, fonts and
notebook-build. There is no check for markdown indentation, prose figures against
stored outputs, quoted code, name-type stability, prompt-box budget, or ⏱ markers.
Defect 1 above is exactly what the missing box-budget check was specified to
catch.

**16. §8.1 — three results are stated in prose before the cells that produce
them.** Cell 35's `catch` bullet: "One hidden layer to two is worth a point or so;
a third is worth roughly nothing here" — above cell 36, which computes it. Cell
37's `left_open`: "the learning rate matters more than the architecture" and "The
default is fine here" — above cell 38. The reader has no measurement left to make.

### Checked and found clean

- **§5.1 / §5.2 — markdown rendering.** Parsed all 35 markdown cells tracking
  fence state: **zero** prose lines indented ≥4 spaces outside a fence, zero
  fences opened or closed at indent ≥4, zero unclosed fences. The `## 5 · Commit`
  form block opens and closes at column 0. The section 11 table is a real
  markdown table, not ASCII art.
- **§2.1 — the raw-versus-scaled comparison is on matched rows.** Verified
  numerically: `X_train_full_u8[fit_idx][:12000].reshape(12000,-1).astype(float32)`
  equals `255 × X_fit` exactly (`np.array_equal` true, max absolute difference
  0.0), and the same holds for the 5,000 validation rows. Same architecture, same
  `max_iter`, same `random_state`, same default `batch_size`. One thing changes.
- **§2.1 — the architecture and learning-rate sweeps are matched.** All ten fits
  use `X_fit[:6000]` and score on the same 5,000-row `X_val`.
- **§4.2 — every training cell re-instantiates its model.** Cells 23 and 26
  construct an `MLPClassifier` inline; `train_curve` constructs one on entry.
  Re-running any of them starts from scratch. No cell continues training an
  existing network.
- **§3.1 / §3.2 — code quoted in prose.** No ```` ```python ```` block appears in
  any markdown cell, so there is nothing to mismatch. The three checks the prose
  offers the reader all execute: `dir(clf)` for "grad" returns `[]`;
  `MLPClassifier(..., loss="mae")` raises
  `TypeError: MLPClassifier.__init__() got an unexpected keyword argument 'loss'`;
  `inspect.signature(MLPClassifier.partial_fit)` returns
  `(self, X, y, sample_weight=None, classes=None)`. I ran all three.
- **Every checkable data figure in the prose is right.** 70,000 images (60,000 +
  10,000) ✓; ten classes ✓; 28×28 ✓; 6,000 per training class ✓; 1,000 per test
  class ✓; pixels 0–255 ✓; 55,000 fit / 5,000 val / 10,000 test under
  `default_rng(42)`, disjoint ✓; the majority class of the 12,000 subsample is
  Sandal and the baseline is exactly 0.1000 ✓; "~30 MB the first time" ✓ (30.9 MB
  of gzipped IDX files).
- **No GPU is required and none is assumed.** Everything runs on a free Colab CPU.
  Note for the course rather than the notebook: `LECTURES.md` marks Part II
  "GPU from Lecture 11 onward", and this notebook is CPU-only by construction —
  which is consistent with its own argument, since its closing point is that
  scikit-learn cannot use a GPU at all.
- **Restart-and-run-all ordering.** Every name is bound before use in a top-to-
  bottom pass: `warnings`/`time` (cell 3) before cell 23; `SUB`, `fit_idx`,
  `val_idx` (cell 15) before cell 23; `clf` and `hist` (cell 30) before cells 42
  and 48; `inspect` imported locally in cell 46.

### Unverified

- **The three ⏱ figures (40 s, 60 s, ~2 min)** — I could not confirm them without
  running the training cells, which this task forbids, and no stored output exists
  to check them against. One is structurally suspect: the 60 s for cell 30 covers
  a loop that also scores 10,000 rows on each of 20 epochs, and nothing in the
  notebook separates training time from scoring time. The same conflation makes
  cell 48's linear extrapolation to 55,000 an over-estimate — scoring cost is
  roughly fixed as the training set grows.
- **Cell 39's two readings** ("one hidden layer to two is worth a point or so; a
  third is worth roughly nothing", "the learning rate matters more than the
  architecture") — unverifiable without running, and unverifiable *by the reader*
  against the file, since no output is stored. Both are also stated before the
  cells that produce them (defect 16).
- **The direction of the raw-versus-scaled result** — cell 26 prints
  `{acc_scaled - acc_raw:+.2f}` and the surrounding prose assumes the sign is
  positive and the raw model's `loss_` higher. Structurally near-certain and
  central to the lecture, but not executed here. If it ever came back negative the
  notebook would print a `-` and say nothing about it; an
  `assert acc_scaled > acc_raw` would turn a silent contradiction into a stop.
