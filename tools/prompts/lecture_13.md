# Lecture 13 — Twenty layers, no learning

**Rebuilding `notebooks/lecture-13.ipynb` in Colab by prompting.**

Géron Chapter 11 · CIFAR-10 · BSc Mathematics of Artificial Intelligence

---

## Before you sit down

**What this notebook is.** You build a network that does not work, and then you
measure why. Nothing is repaired. The measurement is the deliverable; Lecture 14
("Making it train") is the repair.

**Runtime.** A GPU helps but is not needed. Every wall clock below was measured
on a laptop CPU (Apple Silicon, 12 threads, torch 2.13, `numpy` 2.3.5). A free
Colab **CPU** runtime has 2 vCPUs and is roughly 1.5–2.5× slower than that; the
figures in each **⏱** line give both. Total CPU time for the whole notebook,
excluding the one-off download: **about 4 minutes here, 6–10 minutes on a Colab
CPU runtime.**

**Order.** Cells 1–20 run once, top to bottom, and restart-and-run-all passes.
Only three cells train, and each re-instantiates its own model, so any of them
can be re-run alone without continuing a previous fit. The one ordering hazard
is named at cell 19.

**A note on the shape of this script.** Twenty code cells; **seven** carry the
full three-bullet annotation (cells 7, 8, 10, 13, 14, 16, 17) and the other
thirteen carry only the specification. That is deliberate — see `GUIDELINES.md`
§6.1. Every "usual student version" below names a documented library default or
a failure actually observed in this notebook's own output; where neither was
available, the cell is `short`.

**One thing not to read ahead on.** Cell 12 is a transcript of what an assistant
returns for a plausible request. Run it, write its number down, and do not read
cell 13 first.

---

## Section 1 · Setup

Write this markdown first:

> ## 1 · Setup
>
> *Not examinable — engineering hygiene, kept out of the way of the argument.*
>
> Every code cell below is preceded by the specification that produces it.
> These are specifications, not transcripts: the claim is that this is what you
> would have to ask for to get this cell, and that a vaguer prompt gets you
> worse code.

### Cell 1 — setup

**Prompt to type:**

> Setup cell for a teaching notebook. Import math, sys, time, numpy as np,
> torch, torch.nn as nn, torchvision and matplotlib.pyplot as plt. Print the
> python, torch and torchvision versions. Set seed 42 for numpy and torch. Pick
> the device by asking — cuda, then mps, then cpu — print which one won, and if
> it is cpu print a line saying everything still runs and how to turn a GPU on
> in Colab.

**Expect:** three version lines, then `device` on its own line. On a machine
with no accelerator, two extra lines: that everything runs and is slower, and
`Runtime -> Change runtime type -> T4 GPU`.

**Assert:** none.

**Annotate:** short

---

## Section 2 · The data

Write this markdown:

> ## 2 · The data
>
> *Examinable: the balance fact and what it licenses.*
>
> CIFAR-10: 60,000 colour photographs, 32 × 32 pixels, ten classes, split 50,000
> train and 10,000 test by the people who built it. Both published splits are
> exactly balanced — remember that, it decides the metric in section 4.
>
> ⏱ **1–3 minutes the first time**, about 170 MB over the network. Instant
> afterwards, because `download=True` checks before it fetches.

### Cell 2 — load CIFAR-10

**Prompt to type:**

> Load CIFAR-10 train and test with torchvision into a `datasets` folder,
> download only if it is not already there, no transform — I want the raw uint8
> arrays. Call them `Xtr_u8`, `Xte_u8`, `ytr`, `yte` and keep the class names as
> `CLASSES`. Assert both shapes, and assert every class has exactly the same
> number of images in both splits. Print the counts, how many numbers are in one
> image, and the class names.

**Expect:**
`(50000, 32, 32, 3)` and `(10000, 32, 32, 3)`, `uint8`, channels **last**;
`50,000 train + 10,000 test, 10 classes`; `one image is 32x32x3 = 3,072
numbers`; and
`['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse',
'ship', 'truck']`.

**Assert:**
```python
assert Xtr_u8.shape == (50_000, 32, 32, 3)
assert Xte_u8.shape == (10_000, 32, 32, 3)
assert set(np.bincount(ytr)) == {5_000}
assert set(np.bincount(yte)) == {1_000}
```
All four pass (verified against the local copy under `notebooks/datasets/`).

**⏱** 1–3 min on first download, ~1 s thereafter. Identical on CPU and GPU —
this cell does no arithmetic.

**Annotate:** short

---

### Cell 3 — look at it

Write this markdown first:

> ### Look at it
>
> Four examples of each class. Look for two things: whether the objects are
> centred and aligned the way the garments in Fashion MNIST were, and how much
> the backgrounds vary. Both bear on what flattening costs.

**Prompt to type:**

> Show four examples of each CIFAR-10 class in a 4-row by 10-column grid, one
> class per column, titled with the class name. No colormap — these are colour
> images. No axes.

**Expect:** a 4 × 10 grid, column headers reading `airplane … truck`. Objects at
varying scale and position; backgrounds of sky, grass, road, water.

**Assert:** none.

**Annotate:** short

> **If it passes `cmap="gray"`, stop and re-prompt.** Matplotlib ignores `cmap`
> on an `(H, W, 3)` array — it will not raise, and you will not notice.

---

## Section 3 · Split, then scale

Write this markdown:

> ## 3 · Split, then scale — in that order
>
> *Examinable.*
>
> The rule from Lecture 2 ("Your RMSE was a lie") has not been repealed by
> anything in Part II. The scaling statistics come from the **fit** subset only,
> and are then applied to validation and test.
>
> We train on 10,000 of the 50,000 images so a free runtime finishes inside the
> hour, and we flatten each image to 3,072 numbers. Flattening throws away the
> fact that neighbouring pixels are neighbours — the whole subject of Lecture 15
> ("Visual inspection"). Today it is deliberate: we are studying depth, not
> vision.

### Cell 4 — split, then scale

**Prompt to type:**

> Take 5,000 images for validation and 10,000 for fitting from the CIFAR-10
> training set, using a numpy default_rng seeded 42 to permute the indices, and
> assert the two index sets are disjoint. Flatten each image to 3072 floats
> divided by 255, then standardise: compute the per-pixel mean and sd on the fit
> subset only and apply them to fit, validation and test. Add 1e-7 to the sd.
> Assert the three shapes and that the fit subset really is standardised, then
> print the mean and sd of the fit set and of the validation set.

**Expect:**
```
fit 10,000   val 5,000   test 10,000
fit mean -2.87e-06   sd 1.0000
val mean +5.39e-03   sd 1.0003
```
The validation mean is **not** exactly zero, and that is the correct outcome.
(Test, if you print it: mean `+1.06e-02`, sd `0.9977`.)

**Assert:**
```python
assert set(val_idx).isdisjoint(fit_idx)
assert X_fit.shape == (10_000, 3072) and X_val.shape == (5_000, 3072)
assert X_test.shape == (10_000, 3072)
assert abs(X_fit.mean()) < 1e-4 and abs(X_fit.std() - 1) < 1e-2
```

**Annotate:** short

Then write:

> The validation mean is `+5.39e-03`, not zero. It should not be zero: those
> statistics came from a different set of images. A pipeline in which every
> split has mean exactly zero is a pipeline that fitted the scaler on
> everything.
>
> The `+ 1e-7` guard does nothing on this dataset — the smallest per-pixel
> standard deviation across the fit subset is **0.2248**, nowhere near zero. It
> is there for the dataset that has a constant pixel, where the divide-by-zero
> propagates NaN through every layer and surfaces as `loss = nan` five cells
> later, with nothing pointing back here.

---

## Section 4 · The metric, and the number to beat

Write this markdown:

> ## 4 · The metric, and the number to beat
>
> *Examinable.*
>
> Three conditions license plain accuracy: the classes are balanced, there are
> ten of them, and no class is more expensive to get wrong than another. All
> three hold on the CIFAR-10 **test** set. Lecture 4 ("It never fires") was
> about what happens when they do not.
>
> So compute the trivial baseline **before** committing to anything — and
> compute the loss anchor too, because the loss curve is what you will actually
> be staring at.

### Cell 5 — the two anchors

**Prompt to type:**

> Print the per-class counts of the CIFAR-10 test labels, the accuracy you would
> get by always predicting the commonest class, and ln(10) — the cross-entropy
> loss of a model that has learned nothing about ten balanced classes.

**Expect:**
```
test images per class: [1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000]
always predict the commonest class -> accuracy 0.1000
the loss of a model that has learned nothing: ln(10) = 2.3026
```

**Assert:** `assert baseline == 0.1` — exact, because the published test split is
exactly balanced.

**Annotate:** short

Then write this, because it is the one caveat the notebook must not skip:

> **One caveat, and it matters at cell 9.** The *test* split is exactly
> balanced; our 5,000-image *validation* split is a random subsample and is not.
> Its class counts run from 469 to 542, so its majority-class baseline is
> **10.84%**, not 10.00%. When we draw a chance line on a validation-accuracy
> plot, that is the line to draw.

---

## Section 5 · Commit

Markdown only, no cell:

> ## 5 · Commit
>
> **Stop. On paper, now.** Not in this notebook, where you can quietly revise it.
>
> ```
> Metric:                                          ____________
> Accuracy a useful auto-tagger would need:      % ____________
> Accuracy I expect from the model we build today: % ____________
> ```
>
> A prediction you can silently revise is not a prediction. Bring the sheet to
> the next lecture; we score it out loud.

---

## Section 6 · Build the stack

Write this markdown:

> ## 6 · Build the stack
>
> *Examinable.*
>
> Twenty hidden layers of a hundred units, a logistic activation, a linear head.

### Cell 6 — the network

**Prompt to type:**

> Write a `make_net(depth=20, width=100, act=nn.Sigmoid, n_in=3072, n_out=10)`
> that returns an `nn.Sequential`: `depth` blocks of Linear-then-activation,
> then a linear head. Build one, count the parameters, and break the count down
> — the first matrix, the hidden ones, the head — and assert there are depth+1
> Linear modules.

**Expect:**
```
21 weight matrices, 500,210 parameters
  first  3072 x 100 + 100 = 307,300
  each hidden 100 x 100 + 100 = 10,100  (x 19)
  head   100 x 10 + 10 = 1,010
```
Check the arithmetic on paper before you run it: `307,300 + 19 × 10,100 + 1,010
= 500,210`. The first matrix alone is **61.4%** of the parameters — twenty
layers is not twenty times the capacity.

**Assert:** `assert len(lins) == DEPTH + 1`

**Annotate:** short

Then write:

> Note what the specification does **not** say: nothing about how the weights
> start out. That is not an omission being hidden — it is the ordinary case.

---

### Cell 7 — what `nn.Linear` put in those matrices

Write this markdown first:

> ### What did `nn.Linear` put in those matrices?
>
> Nobody said. Look.

**Prompt to type:**

> Take one of the hidden weight matrices and print its shape, min, max, mean and
> standard deviation. Compare the sd against what a uniform on (−b, b) with
> b = 1/sqrt(fan_in) should give, and assert they agree.

**Expect:**
```
hidden weight matrix: (100, 100)
  min -0.1000   max +0.0999
  mean -0.00012   sd 0.05789
that is U(-1/sqrt(fan_in), +1/sqrt(fan_in)) = U(-0.1000, +0.1000)
  a uniform on (-b, b) has sd b/sqrt(3) = 0.05774   <- matches
```
The min and max pin against ±0.1 to four decimals; the measured sd is
`0.057889` against a predicted `0.057735`, a gap of `1.5e-04`.

**Assert:**
```python
assert abs(float(w.std()) - bound / math.sqrt(3)) < 0.002
```
Work the expected value out on paper first: `0.1 / √3 = 0.05774`.

**Annotate:** **full**

- **Left open.** The prompt says "compare against what it should be" and never
  says *which* rule. It happens to be right for the hidden layers and would be
  wrong for the first: layer 1 has fan-in 3,072, so its bound is `0.01804` and
  its sd is `0.010419`, not `0.05774`. A check written against one hardcoded
  bound passes on 19 matrices out of 21 by accident.
- **The usual student version.** Assuming `nn.Linear` uses Xavier/Glorot,
  because that is what the textbook chapter is about. It does not. PyTorch
  initialises the weight with `kaiming_uniform_(weight, a=math.sqrt(5))`, which
  reduces to `U(−1/√fan_in, +1/√fan_in)` — an internal default that matches no
  named scheme in Géron Chapter 11, and it is the direct cause of everything in
  sections 10 onward. It takes four lines to read off and almost nobody does.
- **How you would catch it.** Reviewer question 5 — *what is the default I did
  not ask for?* — applied to a tensor rather than to a function argument. Any
  tensor you did not fill yourself was filled by someone, according to a rule
  you can look up. Turn the rule into an assert, and put the number you expect
  in the assert, not in a comment.

---

## Section 7 · Train it

Write this markdown:

> ## 7 · Train it
>
> *Examinable.*
>
> The loop is Lecture 12's, unchanged, with the three defences that lecture
> ended on: `zero_grad()` inside the batch loop, `eval()` before every
> measurement, and accuracy counted over the set rather than averaged over
> batches.
>
> ⏱ **35–50 s on a laptop CPU, 1–2 minutes on a Colab CPU runtime**, for 20
> epochs. A T4 is not much faster: the matrices are 100 × 100 and the transfers
> cost about what the arithmetic saves.

### Cell 8 — the training loop

**Prompt to type:**

> Move the fit, validation and test arrays to the device as tensors. Write an
> `accuracy(net, X, y)` that puts the net in eval mode, counts hits over the
> whole set in batches of 2000, and puts it back in train mode. Then a
> `train(net, epochs=20, lr=1e-3, batch=128, seed=42, track_grads=False)` using
> Adam and CrossEntropyLoss, re-seeding at the top, shuffling with a seeded
> generator, `zero_grad` inside the batch loop, recording mean training loss and
> validation accuracy each epoch, and — when `track_grads` is on — the norm of
> every Linear weight's gradient at the end of each epoch. Return the net and
> the history including elapsed seconds. Then seed 42, build a fresh 20-layer
> net, train it with `track_grads=True`, and print the epoch-1 and final loss
> beside ln(10), the validation accuracy, the test accuracy and the baseline.

**Expect:** six lines. The two that matter:

```
loss  epoch 1 2.30xx  ->  epoch 20 2.30xx
chance loss ln(10) = 2.3026
validation accuracy 0.1xxx
TEST accuracy       0.1xxx
baseline            0.1000
```

Both losses land within about ±0.02 of **2.3026** and the accuracies within a
point or two of **0.10**. Two independently checkable facts about that number:
the mean per-batch loss of the network *at initialisation*, forward-only, is
**2.3215**; and the lowest loss any model can reach by learning the class priors
alone is the entropy of the fit labels, **2.30230** nats — indistinguishable
from ln(10) at four decimals. There is nowhere below ln(10) for this network to
go.

**Assert:** none in the cell, but write the two anchors down: `ln(10) = 2.3026`
and `baseline = 0.1000`. Everything after this is measured against them.

**⏱** 40 s here (33 s of stepping + ~5 s of per-epoch validation);
**1–2 min on a Colab CPU runtime**. Measured per-step at depth 20: 20.6 ms with
2 threads, 25.4 ms with 12 — the layers are too small for threads to help, which
is why the CPU and GPU numbers are close.

**Annotate:** **full**

- **Left open.** `track_grads=True`, which records the per-layer gradient norm
  every epoch. Nothing uses it until section 10, three sections and eleven cells
  away. It costs one line to collect and it **cannot be collected
  retroactively** — the question "did the profile change over time?" is
  unaskable after the fact, and you will want to ask it.
- **The usual student version.** Leaving the head bare and adding a softmax, or
  adding a `LogSoftmax` and then using `CrossEntropyLoss` anyway.
  `nn.CrossEntropyLoss` applies its own `log_softmax` to whatever you hand it —
  documented, and the reason `make_net` ends in a plain `nn.Linear`. Feed it
  probabilities and it trains, badly, and never raises. Note that this is a
  failure you could not detect from the loss here: a doubly-softmaxed twenty-
  layer sigmoid stack also sits at ln(10).
- **How you would catch it.** Print the loss at epoch 1 and at epoch 20 *beside*
  `ln(10)`, on adjacent lines, in the same cell. Three numbers and the whole
  result is visible before any plot. A loss printed alone, to four decimal
  places, reads as a quantity that is going somewhere.

---

### Cell 9 — the curves that show nothing happening

**Prompt to type:**

> Two plots side by side from the history: training loss per epoch with a
> horizontal line at ln(10), and validation accuracy in percent with a
> horizontal line at the validation majority-class rate. Fix the accuracy y-axis
> from 0 to 30 percent.

**Expect:** a loss curve pinned to the ln(10) line and an accuracy curve
wandering in a narrow band around 10%, both visibly flat because the axes are
fixed.

**Assert:** none.

**Annotate:** short

> **The chance line on the right panel goes at 10.84, not 10.** That is the
> majority-class rate of *our* validation subsample (542 of 5,000), not of the
> published test split. Two different sets, two different baselines; label the
> line with which one it is.
>
> **And fix the y-limits.** Left to autoscale, matplotlib will fit the axis to
> the data — for a series confined to a fraction of a percent that turns pure
> noise into a dramatic trend. Autoscale is for exploring. A fixed axis is for
> claiming, and what we are claiming here is that nothing happened.

---

## Section 8 · Rule out the bugs

Write this markdown:

> ## 8 · Before blaming the architecture, rule out the bugs
>
> *Examinable.*
>
> Lecture 12 ended with three failures that run, produce a plausible number and
> never raise — the three questions it added to the red-team list: is
> `zero_grad()` inside the batch loop, is there an `eval()` before every
> evaluation, and is any metric a plain mean of per-batch values. All three
> would look exactly like this result. Check them rather than assuming.

### Cell 10 — the checks, and the overfit test

**Prompt to type:**

> Add a diagnostic cell. First check the labels are aligned with the images:
> show one training image with its class name as a caption. Then the real test —
> take the same `train` machinery, build a 2-layer net, and try to overfit 200
> images with 200 full-batch Adam steps. Print the training accuracy on those
> 200 images and assert it is above 0.95, so the cell fails loudly if the loop
> cannot memorise.

**Expect:** one small image captioned `ship` (that is `CLASSES[y_fit[3]]`, and
the image is `Xtr_u8[fit_idx[3]]`), then a line reporting training accuracy on
the 200 memorised images. It must come back near 1.000 — a loop that cannot
memorise 200 images is broken, and depth would then be irrelevant.

**Assert:**
```python
assert acc_tiny > 0.95, "the loop cannot memorise 200 images — fix the loop first"
```
**Write this assert even though the assistant will not offer it.** See the third
bullet.

**⏱** ~3 s here, ~6 s on a Colab CPU runtime.

**Annotate:** **full**

- **Left open.** How many failures there are to rule out. The prompt says
  "check the labels" and "try to overfit"; it never says the list is closed.
  Lecture 12 contributes exactly **three** — its red-team questions 6, 7 and 8 —
  and this cell adds two more of its own (label alignment, and can-the-loop-fit-
  anything). If the prose says "all four", count them: neither number is four,
  and a reader who genuinely counts cannot tell whether the missing item is
  theirs or the author's.
- **The usual student version.** Concluding *"deep networks are hard"* and moving
  on. Observed here, in this notebook's own current text: the shipped cell ends
  with `print("The loop can memorise. So the loop is not the bug.")` — an
  unconditional string. It prints that sentence at a training accuracy of 1.00
  and at 0.13 alike, in a notebook whose next cell asserts its premise properly.
  A conclusion printed by a `print` is not a conclusion; it is a comment that
  looks like evidence.
- **How you would catch it.** Turn every printed conclusion into an assert. The
  overfit test is the strongest single diagnostic in deep learning and it is
  worth nothing if its verdict is hard-coded. If the assert fires, you have
  learned something in three seconds; if it never can fire, you have learned
  nothing in three seconds and felt reassured.

---

### Cell 11 — the control

Write this markdown first:

> ### The control: the same code, two layers instead of twenty
>
> One variable changes.
>
> ⏱ **1.5–2.5 minutes on a laptop CPU, 3–6 minutes on a Colab CPU runtime**, for
> the whole sweep.

**Prompt to type:**

> Run the same training at depths 1, 2, 5, 10 and 20, re-seeding before each fit
> so depth is the only variable. Print the final loss and test accuracy at each,
> bar-chart the accuracies with a line at 10 percent, and assert depth 2 beats
> depth 20.

**Expect:** five rows, all five losses near 2.3026, all five accuracies in the
neighbourhood of 0.10 — and depth 2 above depth 20. A bar chart that sits on the
chance line.

**Assert:**
```python
assert sweep[2] > sweep[20], "if this fails, depth is not the variable"
```
This assert encodes the premise of the lecture. If it fires, stop and
re-diagnose rather than continuing to instrument the wrong thing.

**Note, and put it in the prose:** because `train` re-seeds to 42 at its top and
the sweep re-seeds before each `make_net`, the depth-20 row is built from the
identical initial weights and sees the identical shuffle as `deep` in cell 8. It
must reproduce cell 8's test accuracy **exactly**, to all four decimals. If it
does not, something consumed randomness in between and the sweep is not
controlled. Check that equality out loud; it is free.

**⏱** Measured per-step, 2 threads: depth 1 → 8.1 ms, depth 2 → 8.8 ms, depth 5
→ 6.0 ms, depth 10 → 11.7 ms, depth 20 → 20.6 ms. 79 batches × 20 epochs each:
**87 s of stepping**, plus five validation histories, ≈ 1.7 min here.

**Annotate:** short

Then write:

> Adding layers made it **worse**. That is not what capacity is supposed to do,
> and the rest of the notebook is about why.

---

## Section 9 · An assistant writes the network

Write this markdown — and nothing more than this:

> ## 9 · An assistant writes the network
>
> Here is a real request and the code it returns. It runs, it trains without
> error, and it reports a number.
>
> > *"Write me a deep PyTorch classifier for CIFAR-10 with 20 hidden layers and
> > train it for a few epochs."*

**Do not warn the reader here.** No ⚠, no "read before running", no hint in the
prompt box. The whole exercise is whether the number gets written down
uncritically, and `GUIDELINES.md` §8.1 records that a defect announced four
times catches nobody. It is announced once, in section 9's *title only*, and the
contrast opens the next cell.

### Cell 12 — what the assistant returns

**Prompt to type:** (type it exactly this lazily — that is the point)

> write me a deep pytorch classifier for cifar-10 with 20 hidden layers and
> train it for a few epochs

**Expect:** an `nn.Module` subclass wrapping an `nn.Sequential` of
Linear-plus-Sigmoid blocks and a head; Adam at `lr=0.001`; a five-epoch loop
with `optimizer.zero_grad()` inside the batch loop; then `model.eval()`,
`torch.no_grad()`, and accuracy over the whole test set. Five `Epoch n, Loss:
…` lines and one `Test Accuracy: …%`.

Every one of Lecture 12's three defences is present. A review looking for them
finds nothing.

**Write the test accuracy down before you scroll.**

**Assert:** none. That is part of what is being demonstrated.

**⏱** ~10 s here (5 epochs), ~20 s on a Colab CPU runtime.

**Annotate:** short — spec only, deliberately. The full annotation belongs to
cell 13, after the number is on paper.

---

### Cell 13 — the comparison it did not make

Write this markdown first:

> ### ⚠ The review question that catches it
>
> Not *"does it run?"* — it does. Not *"is the loop correct?"* — the
> `zero_grad`, the `eval()` and the whole-set accuracy are all there.
>
> The question is the one this course asks of every number:
>
> > **What would this number be if the model had learned nothing at all?**
>
> Ten balanced classes: **10.00%**. Compare that with what you wrote down.

**Prompt to type:**

> Print the assistant's test accuracy, the majority-class baseline, and the
> difference between them in accuracy points. Then print ln(10) beside the loss
> its loop reported. And recount its loop: how many batches does
> `range(0, len(Xf), 128)` actually produce, and what did it divide the running
> loss by?

**Expect:**
```
the assistant's model:  0.1xxx
a model with no weights at all: 0.1000
difference: ±0.xx accuracy points
```
and then the recount, which is the part nobody asks for:
```
batches actually run:            79
what it divided by (10000//128): 78
```

**Assert:**
```python
assert len(range(0, len(Xf), 128)) == 79 and len(Xf) // 128 == 78
```

**Annotate:** **full**

- **Left open.** Everything about *what to compare the number with*. The prompt
  asked for a classifier and got one; it never asked what the number would be
  under the null, so nothing in 40 lines of correct code computes it. The defect
  is an **absence**, not a mistake — there is no line to point at and say "this
  one is wrong."
- **The usual student version.** Reading `10.24%` as a low but real score. It is
  not a low score, it is *no* score. And there is a second, observed defect in
  that same cell that a Lecture-12 review should have caught and this notebook's
  own prose currently misses: `running / (len(Xf) // 128)` divides by **78**
  when the loop ran **79** batches — `drop_last` semantics done by hand, with
  the short final batch of 16 counted in the numerator and not the denominator.
  Its printed loss is therefore inflated by exactly 79/78 = 1.0128. At the ln(10)
  plateau it prints **2.3321**, not 2.3026 — verified arithmetically, and
  confirmed forward-only at initialisation, where the correct divisor gives
  **2.3215** and the assistant's gives **2.3513**. That is Lecture 12's red-team
  question 8, *"is any metric a plain mean of per-batch values?"*, alive and
  unremarked in the cell the notebook holds up as clean.
- **How you would catch it.** Two habits, both one line. Print every metric next
  to what it would be under the null, in the same `print`. And count the
  denominator of any average you did not let a library compute: `len(range(0, N,
  B))` is `ceil(N/B)`, `N // B` is `floor(N/B)`, and they differ whenever `B`
  does not divide `N`. Here that is a 1.3% error on the headline number of the
  section — small enough to survive review, large enough to make "the loss never
  moved from 2.3026" a sentence the output does not support.

Then write:

> ### The corrected specification
>
> > *"Write a PyTorch classifier for CIFAR-10 with 20 hidden layers. **State
> > explicitly which initialisation each layer gets and why.** After training,
> > report accuracy against the majority-class baseline, and log the gradient
> > norm of the first and last weight matrices at every epoch. If the loss does
> > not fall below ln(10) within three epochs, stop and report that instead of
> > continuing."*
>
> Three additions, and each is something the assistant cannot know unless you
> say it: what a defensible default is, what the number must be compared
> against, and what evidence to produce when the answer is *it did not work*.

---

## Section 10 · Instrument it

Write this markdown:

> ## 10 · Instrument it
>
> *Examinable: what each probe measures, and why one of them lies.*
>
> We wrote the loop, so we can put a probe anywhere in it. Two probes: what each
> layer **outputs** on the way forward, and what each weight matrix
> **receives** on the way back.
>
> Both probes run on a **freshly initialised** network, on the CPU, in float64.
> Matching them matters: the numbers they produce are about to be compared with
> each other, and a forward probe on the trained model against a backward probe
> on a fresh one is a comparison of two different networks.

### Cell 14 — the forward pass

**Prompt to type:**

> Write an `activation_stats(net, X, n=512)` that pushes the first 512 rows
> through the network and, after each Sigmoid, records the mean, the sd over the
> whole tensor, the mean over units of the sd down the batch, and the fraction
> of activations with |h − 0.5| > 0.45. Assert there is one row per hidden
> layer. Run it on a freshly seeded 20-layer net in float64 and again in
> float32, print a table of layers 1, 2, 5, 10, 15, 20 for both, and report the
> layer-20-to-layer-1 ratio of the batch sd and the per-layer factor in each
> precision.

**Expect:** float64 —

| layer | mean | sd (whole tensor) | sd (down the batch) | saturated |
|---|---|---|---|---|
| 1 | 0.4998 | 0.1311 | 1.2916e-01 | 0.000 |
| 2 | 0.4987 | 0.0708 | 1.8519e-02 | 0.000 |
| 5 | 0.4881 | 0.0773 | 5.8696e-05 | 0.000 |
| 10 | 0.5055 | 0.0731 | 2.9748e-09 | 0.000 |
| 15 | 0.4861 | 0.0663 | 1.6030e-13 | 0.000 |
| 20 | 0.5114 | 0.0710 | 2.5070e-17 | 0.000 |

`layer 20 : layer 1 = 1.9411e-16` (**15.7 orders of magnitude**), per layer
**×0.1490**.

And float32, the same network, the same 512 rows: the first three rows are
identical, then the fourth column stops falling — layer 10 reads `1.3965e-08`,
layer 15 `2.6006e-09`, layer 20 `1.9777e-09`, ratio `1.5312e-08`, per layer
`×0.3879`.

**Assert:**
```python
assert len(stats64) == DEPTH
assert stats32[19]["sd"] / stats64[19]["sd"] > 1e7   # the float32 column has floored
```

**Annotate:** **full**

- **Left open.** The dtype. The prompt says "the sd down the batch" and never
  says in what precision, and the answer changes by eight orders of magnitude.
  It also leaves open *which* net — trained or fresh — and the notebook is about
  to compare this number with a backward probe that uses a fresh one.
- **The usual student version.** Reading `h.std()` and concluding the forward
  pass is healthy. It is the natural thing to print and it is flat with depth —
  0.1311 at layer 1, 0.0710 at layer 20 — because `h.std()` over the whole
  tensor is dominated by the spread of *that layer's random biases across the
  hundred units*, and that spread does not care how deep the layer is. What
  carries information is how much one unit's output moves when the **input**
  changes: the sd down the batch, averaged over units. Same tensor, same call,
  one argument different (`dim=0`), and the two columns say opposite things.
  Note also that `Tensor.std()` defaults to `correction=1` while `ndarray.std()`
  defaults to `ddof=0` — irrelevant at n = 512, and not irrelevant if you ever
  probe a batch of 4.
- **How you would catch it.** Two rules, both violated by the obvious version of
  this cell.
  **(a) A check that cannot fail is worse than no check.** The saturated column
  asks whether `|h − 0.5| > 0.45` while the activations sit in a band of sd
  **0.0710** at layer 20 — the threshold is **6.34** standard deviations away,
  and **3.43** away even at layer 1 where the band is widest. Across all
  20 × 512 × 100 activations exactly **one element** crosses it, at layer 1;
  every layer prints `0.000`, and a column of zeros reads as reassurance about
  saturation when it is a statement about the width of the band.
  **(b) Check that your probe can represent what you are asking it to measure.**
  Sigmoid outputs live near 0.5, where the gap between adjacent float32 numbers
  is `5.96e-08`. A batch sd of `2.5e-17` is **4.2e-10 of one ulp**: not small,
  *unrepresentable*. In float32, unit 0 at layer 15 takes **exactly one distinct
  value** across all 512 images, and at layer 20 exactly one as well. The
  `~2e-09` floor the float32 column reports is quantisation noise from the
  handful of units still holding two or three distinct values. It is the dtype,
  not the network — and it turns a per-layer factor of 0.149 into 0.388 and
  sixteen orders of magnitude into eight.

Then write, with the numbers from the cell above it:

> **Read the two sd columns against each other.** The first is flat and the
> second falls off a cliff, and only one of them is about the signal.
>
> `h.std()` over the whole tensor barely moves with depth — 0.1311 at layer 1,
> 0.0710 at layer 20 — because it is dominated by the spread of that layer's
> random **biases** across units, and that spread does not care how deep the
> layer is. Reading it, you would conclude the forward pass is healthy and go
> looking elsewhere. **That conclusion would be wrong.**
>
> The sd **down the batch** collapses by a factor of **0.1490 per layer**, from
> `1.2916e-01` at layer 1 to `2.5070e-17` at layer 20 — a ratio of `1.94e-16`,
> **15.7 orders of magnitude**. By layer 20 every input produces essentially the
> same activation, which is another way of saying the network has stopped being
> a function of its input.
>
> **And in float32 you cannot see any of this.** The same measurement on the same
> network floors out at about `2e-09` from layer 11 onward, because that is where
> the spread drops below one unit in the last place of a float near 0.5. The
> float32 column reports a per-layer factor of 0.388 and eight orders of
> magnitude, and both are properties of the dtype. Section 10's backward probe
> makes exactly this argument about gradient norms; it applies here first.
>
> **So the forward signal does die.** It dies quietly, and every obvious probe —
> the whole-tensor sd, the saturation fraction, the float32 batch sd — says
> otherwise.

---

### Cell 15 — the same layers, on the axis that matters

**Prompt to type:**

> Two plots. First the misleading one: mean and whole-tensor sd against layer,
> on a linear axis with y from 0 to 0.62. Then the float64 batch sd against
> layer on a log axis, with the float32 version on the same axes in a second
> colour so the floor is visible.

**Expect:** the first plot is two near-horizontal lines and supports the wrong
conclusion. The second is a straight descending line from `1.3e-01` to `2.5e-17`
in float64, with the float32 trace peeling away from it around layer 8 and
flattening at `~2e-09`.

**Assert:** none.

**Annotate:** short

> Sixteen orders of magnitude cannot be shown on a linear axis — plotted
> linearly, that quantity is a flat line at zero. When a quantity might span
> orders of magnitude, try a log axis before concluding it is constant. Flat on
> linear and flat on log are very different findings, and *flat on log because
> your dtype ran out* is a third finding again.

---

### Cell 16 — the backward pass, in float64

Write this markdown first:

> ### Now the backward pass
>
> The gradient of the loss with respect to each weight matrix, averaged over
> eight batches. One batch of 128 is a noisy estimate of anything, and a course
> that says so should not then quote one.

**Prompt to type:**

> Write a `grad_profile(net_factory, X, y, n_batches=8, dtype=torch.float64)`
> that builds a fresh net from the factory in the given dtype, runs eight seeded
> random batches of 128 through a backward pass with `net.zero_grad()` between
> them, and returns the mean gradient norm of each Linear weight. Run it in
> float64 and in float32, print the largest relative disagreement between them
> and how many float32 norms came back exactly zero, assert none did, and print
> the norms at layers 1, 5, 10, 15, 20 and the head.

**Expect:**
```
largest relative disagreement between float32 and float64: 5.62e-06
layers whose float32 gradient underflowed to exactly zero: 0
 layer 1   ||dL/dW|| = 4.5673e-17
 layer 5   ||dL/dW|| = 9.2964e-15
layer 10   ||dL/dW|| = 1.8450e-10
layer 15   ||dL/dW|| = 3.0416e-06
layer 20   ||dL/dW|| = 6.6082e-02
    head   ||dL/dW|| = 4.8663e-01
```

**Assert:**
```python
assert (g32 > 0).all(), "if this ever fails, the float32 plot is meaningless"
```

**⏱** 0.22 s float64, 0.10 s float32. Free.

**Annotate:** **full**

- **Left open.** Why float64 at all, and whether it was needed. The prompt asks
  for both precisions and never says what a bad answer would look like, so the
  cell has to be told to check. Here it comes back clean — no underflow, and the
  two precisions agree to `5.6e-06` relative. That is 47× looser than float32
  epsilon (`1.19e-07`), which is itself the fingerprint of the effect: the norm
  is a sum of squares, individual entries of layer 1's gradient are around
  `8e-20`, and `(8e-20)² = 6.8e-39` is below float32's smallest normal
  (`1.18e-38`). The squares go subnormal and lose bits even though the norms do
  not hit zero. **The cell measured that rather than assuming it**, in either
  direction.
- **The usual student version.** Measuring in float32, plotting on a log axis,
  and reading a floor that is the dtype rather than the network — which is
  exactly what happens to the *forward* probe two cells up if you let it stay in
  float32. The other real default in this cell: `net.zero_grad()` has defaulted
  to `set_to_none=True` since torch 2.0, so `.grad` becomes `None`, not a tensor
  of zeros. Gradient-logging code that reads `m.weight.grad.norm()` after a
  zeroing raises `AttributeError` instead of reporting 0.0 — which is why the
  norms here are read *after* `backward()` and before the next `zero_grad()`.
- **How you would catch it.** When you plot something tiny on a log axis, prove
  it did not underflow before you interpret the shape. A log plot of exact zeros
  is blank; a log plot of subnormals is noise that looks like a floor; and a
  floor is precisely the shape you would take as evidence that the gradient
  "levels off". One assert distinguishes all three.

---

### Cell 17 — a straight line on a log axis

**Prompt to type:**

> Plot the float64 gradient norms per layer on a log y axis. Then compute the
> layer-20-to-layer-1 ratio, the per-layer factor as the geometric mean of the
> consecutive ratios, and check it by raising it to the 19th power. Also print
> the geometric mean over layers 2 to 20 only, and the arithmetic mean of the
> ratios for comparison.

**Expect:**
```
layer 20 : layer 1  =  1.4469e+15   (15.2 orders of magnitude)
per layer, going down: x 0.1593
check: 0.1593 ** 19 = 6.9115e-16   and 1/1.4469e+15 = 6.9115e-16
layers 2-20 only:      x 0.1396
arithmetic mean of the ratios: 6.8294  ->  1/6.8294 = 0.1464, ^19 = 1.4019e-16
```

**Assert:**
```python
assert abs((1/gain)**19 / (1/atten) - 1) < 1e-9   # geometric mean is exact by construction
```

**Annotate:** **full**

- **Left open.** *Which* layers count as "the line". Layer 1's gradient is
  `4.5673e-17` and layer 2's is `2.668e-17` — layer 1 is **1.71× larger**, the
  only place in the profile where the norm goes *down* as you move away from the
  loss. It is not an anomaly of the network: layer 1's weight matrix has 3,072
  inputs, so its gradient norm carries a factor the 100-wide matrices do not,
  and the Frobenius norm of a bigger matrix is bigger for the same per-entry
  scale. Fit a line through log₁₀ of all twenty and layer 1 sits **0.90 decades**
  off it; drop it and the remaining nineteen fit to within **0.05 decades**. The
  per-layer factor moves from **0.1593** to **0.1396** depending on which you
  meant, and the prompt does not say.
- **The usual student version.** Taking the arithmetic mean of the ratios. It
  gives 6.8294, so 1/6.8294 = **0.1464** — which looks like a perfectly
  reasonable answer, sits between the two defensible ones, and is wrong for a
  reason you can check in one line: raised to the 19th it gives `1.4019e-16`
  against the measured `6.9115e-16`, off by a factor of **4.93**. The arithmetic
  mean of ratios is dominated by whichever ratio happens to be largest and does
  not reproduce the end-to-end number. The geometric mean does, exactly, by
  construction.
- **How you would catch it.** The self-check in the last print. If the per-layer
  factor to the 19th power does not reproduce the measured end-to-end ratio,
  one of the two is wrong and you have found out in one line, for free, before
  you write it in prose. Then say which layers the factor is over, in the same
  sentence as the number.

Then write, and be careful with the word *exactly*:

> **That is a straight line on a log axis, from layer 2 onward.** Which means
> the attenuation is not an accident of one layer: it is the *same factor,
> applied nineteen times*.
>
> The forward and backward measurements agree, and they agree with theory,
> to within about **7%** — which is what "the same mechanism seen from both
> ends" is worth on eight batches of 128:
>
> | quantity | per-layer factor |
> |---|---|
> | forward, sd down the batch, float64 | 0.1490 |
> | backward, geometric mean over layers 2–20 | 0.1396 |
> | backward, geometric mean over all nineteen | 0.1593 |
> | predicted: σ′(0.5) · √fan_in · sd(W) = 0.25 · 10 · 0.0577 | 0.1443 |
>
> Write down the two numbers — the per-layer factor and the end-to-end ratio.
> They are what the next lecture derives from first principles, and predicting a
> number you have already seen is not the same exercise.

---

### Cell 18 — does training rescue it

Write this markdown first:

> ### Does training rescue it?
>
> We logged the per-layer gradient norms at every epoch. If the first layers
> were merely slow to start, the profile would flatten as the network learns.

**Prompt to type:**

> From the recorded gradient history, plot the per-layer profile at epochs 1, 5,
> 10 and 20 on one log axis, and print layer 1's gradient at epoch 1 and at
> epoch 20.

**Expect:** four curves of the same shape, lying on top of one another. Layer 1
at epoch 20 is within an order of magnitude of layer 1 at epoch 1 and nowhere
near the last layers.

**Assert:** none.

**Annotate:** short

> Plot several epochs on **one** axis. The question is whether the shape changed,
> and one epoch per panel cannot answer it. And note what made this cell possible:
> `track_grads=True`, eleven cells ago, before the first training run. "Did it
> change over time?" cannot be asked retroactively.

---

### Cell 19 — what the weights actually did

Write this markdown first:

> ### What that means for the update
>
> Adam divides by a running estimate of the gradient's own magnitude, so this is
> not simply "small gradient, small step". But the first layers are driven by a
> signal fifteen orders of magnitude below the last ones, and at that level it is
> almost entirely the eight-batch noise. Check what the weights actually did.

**Prompt to type:**

> Reproduce the weights `deep` started from by re-seeding to 42 and calling
> `make_net()` again — `train` does not re-initialise the network it is handed,
> so the same seed gives exactly the tensors it began with. Print the relative
> change in the Frobenius norm of each weight matrix over the twenty epochs, for
> layers 1, 10, 20 and the head.

**Expect:** a monotone-ish increase from layer 1 to the head — the layers nearest
the loss move, the layers nearest the input barely do.

**Assert:** none, but see the hazard below.

**⏱ Ordering hazard — the one in this notebook.** This cell reads `deep`, which
must still be the object trained in **cell 8**. It is safe on a
restart-and-run-all. If you re-run **cell 8** on its own, it rebuilds `deep` from
seed 42 and retrains from scratch, so this cell stays correct. But if you re-run
**cell 11** (the sweep) and then this cell, nothing breaks and nothing is wrong
either — the sweep binds `m`, not `deep`. The genuine trap is inserting any
`torch.manual_seed` or any extra `make_net()` call *between* the re-seed and
`make_net()` inside this cell: the reference weights then belong to a different
network and every relative change becomes meaningless without raising. If you
edit this cell, re-run it whole.

**Annotate:** short

> **Relative change, not absolute.** A matrix whose entries are `U(±0.018)` and
> one whose entries are `U(±0.1)` cannot be compared by the norm of their
> differences. And PyTorch trains **in place**: a name you bound before training
> points at the trained weights afterwards. The only clean copy is one you
> `.clone()` or one you reconstruct from the seed.

---

## Section 11 · Record the number, and stop

Write this markdown:

> ## 11 · Record the number, and stop
>
> You have a network that runs, a loop that is provably correct, and a result
> indistinguishable from guessing. That is today's deliverable.
>
> On the same sheet of paper, next to what you predicted:
>
> ```
> Test accuracy I actually got:            % ____________
> Gradient norm at layer 1:                  ____________
> Gradient norm at layer 20:                 ____________
> Per-layer attenuation factor:              ____________
> ```
>
> Do not repair anything. The next lecture derives that per-layer factor from
> the shape of the weight matrices — and then removes it.

### Cell 20 — the table

**Prompt to type:**

> Print one summary table: the majority-class baseline first, then the 20-layer
> test accuracy, the 2-layer test accuracy, the layer-1 and layer-20 gradient
> norms, and the per-layer attenuation factor, saying which layers it is
> averaged over.

**Expect:**
```
==============================================================
baseline (majority class)              0.1000
20 hidden layers, 20 epochs            0.1xxx
2 hidden layers, same everything else  0.1xxx
gradient, layer 1                      4.5673e-17
gradient, layer 20                     6.6082e-02
per-layer attenuation (layers 2-20)    0.1396
==============================================================
```

**Assert:** none. **The baseline goes first, at the top of the table**, so that
every number below it is read against something.

**Annotate:** short

---

## Section 12 · Red-team

Markdown only:

> ## 12 · Red-team
>
> Swap notebooks with the team beside you. Ten minutes. The five questions, and
> one that is new today:
>
> 1. What touched the test set?
> 2. What was fitted, and on what?
> 3. What is the shape here?
> 4. What was dropped — rows, columns, NaNs? Count them.
> 5. What is the default I did not ask for?
> 6. **What would this number be if the model had learned nothing?** Find the
>    line in your partner's notebook that answers it. If there is no such line,
>    that is the finding.
>
> Report what you **found**, not what you would have done differently.

---
---

# Defects found in the current notebook

Against `notebooks/lecture-13.ipynb` (64 cells, 20 code cells, **no stored
outputs anywhere**) and `GUIDELINES.md`. Every item marked **[verified]** was
re-derived with `python3` against the local CIFAR-10 copy under
`notebooks/datasets/`, seed 42, torch 2.13.0. Items marked **[not verified]**
say why.

## Prose figures that the cell cannot produce — §1.1, §1.2

**1. The three headline instrumentation figures are float64 numbers reported for
a float32 cell. [verified]**

Cell 47 states the sd down the batch "collapses by a factor of **0.149 per
layer** — from 1.29e-01 at layer 1 to **2.42e-17** at layer 20. Sixteen orders
of magnitude." Cell 44 computes `activation_stats(deep, Xf)` where `deep` and
`Xf` are float32.

| quantity | prose says | float64 gives | **float32 gives — what the cell prints** |
|---|---|---|---|
| layer 1, sd down batch | 1.29e-01 | 1.2916e-01 | 1.2916e-01 |
| layer 20, sd down batch | 2.42e-17 | 2.5070e-17 | **1.9777e-09** |
| ratio L20 : L1 | "sixteen orders" | 1.94e-16 (15.7) | **1.53e-08 (7.8)** |
| per-layer factor | 0.149 | 0.148955 | **0.387870** |

The cause is representability, not seed or training. Sigmoid outputs sit near
0.5, where consecutive float32 values differ by `5.96e-08`; a batch sd of
`2.5e-17` is `4.2e-10` of one ulp. Measured directly: in float32, unit 0 takes
**exactly one distinct value** across all 512 images at layer 15 and again at
layer 20 (3 distinct values at layer 10). The `~2e-09` the cell reports is
quantisation noise, not signal. This holds whatever the weights are, so training
the network cannot change it.

*Caveat on my measurement:* I computed on a **freshly initialised** net (seed 42)
because the brief forbids executing training cells; the notebook measures on
`deep` after 20 epochs. The float32 unrepresentability argument is independent of
the weights. The float64 digits (`2.5070e-17`, `0.148955`) are for the fresh net;
the notebook's `2.42e-17` is consistent with the same computation after training
barely moves the early layers, which supports the conclusion that **the prose was
written from a float64 run and the shipped cell was not**.

**2. "Nothing could ever have crossed it" is false. [verified]** Cell 47 says of
the saturation threshold: *"Nothing could ever have crossed it."* Exactly **one
activation out of 20 × 512 × 100 crosses it**, at layer 1 — a fraction of
`1.953e-05`, which the `:.3f` format prints as `0.000`. The surrounding argument
is right and better stated as "one element in 1.02 million"; the absolute is not.

The rest of that paragraph checks out: sd of the band at layer 20 is **0.0710**
("about 0.071" ✓); `0.45 / 0.0710 = 6.34` ("6.3 standard deviations" ✓);
`0.45 / 0.1311 = 3.43` at layer 1 ("3.4 even at layer 1" ✓); whole-tensor sd
"stays near 0.07 at layer 20" ✓ (0.0710).

**3. `0.1013` is not a value the validation accuracy can take. [verified]** Cell
27's annotation says autoscaling gives "a y-axis from 0.0998 to 0.1013". With
5,000 validation images, accuracy is a multiple of `1/5000 = 0.0002`;
`0.0998 = 499/5000` ✓ but `0.1013 / 0.0002 = 506.5` ✗. Separately, cell 28 plots
`100*a`, so the axis is 9.98–10.13 **percent**, not 0.0998–0.1013. Illustrative
figures still have to be achievable in the notebook they illustrate.

**4. `Epoch 5, Loss: 2.3026` cannot be printed by the cell it describes.
[verified]** Cell 36's `catch` bullet quotes it, and cell 40 prints *"The loss it
printed, epoch by epoch, never left ln(10) = 2.3026 either."* The assistant's
cell computes `running / (len(Xf)//128)`. `len(Xf) = 10000`; `range(0, 10000,
128)` yields **79** batches; `10000 // 128` is **78**. The short final batch of 16
is counted in the numerator and not the denominator, inflating the printed loss
by exactly `79/78 = 1.0128`. At the plateau it prints **2.3321**. Verified
forward-only at initialisation: correct divisor **2.3215**, assistant's divisor
**2.3513**.

*Verification limit:* I did not run the 5-epoch loop (brief: no training cells).
The `79 vs 78` count and the `79/78` factor are exact; the plateau value 2.3321
follows from `ln(10) × 79/78` and is exact to the extent that each batch mean is
ln(10).

## A comparison on unmatched objects — §2.1

**5. The forward and backward probes measure two different networks, and the
prose compares them. [verified by reading, arithmetic checked]** Cell 44 profiles
`deep` — trained, 20 epochs. Cell 50 calls `grad_profile(make_net, …)`, which
builds a **fresh** net. Cell 47 then asserts the forward per-layer factor is
*"exactly the rate at which the gradient vanishes on the way back."* Two
networks, and the prose says "exactly" about neither of them:

| | per-layer factor |
|---|---|
| forward (float64, fresh net) | 0.148955 |
| backward, geometric mean over all 19 ratios — **what cell 52 prints** | 0.159251 |
| backward, geometric mean over layers 2–20 | 0.139566 |
| theory: 0.25 · √100 · 0.057735 | 0.144338 |

The prose's 0.149 differs from the number the notebook's own cell 52 prints
(0.1593) by **6.9%**, and the two figures are never reconciled anywhere —
**§1.5**. The physical claim (same mechanism, both directions) is sound and holds
to about ±7%; "exactly" does not.

**6. "That is a straight line on a log axis" is false at layer 1. [verified]**
Cell 53. Layer 1's gradient norm is `4.5673e-17` and layer 2's is `2.668e-17` —
layer 1 is **1.71× larger**, the only descent in the profile, because its matrix
is 3,072-wide. Log₁₀-linear fit over layers 1–20: layer 1 sits **0.90 decades**
off the line. Over layers 2–20: max residual **0.051 decades**. Including layer 1
in the geometric mean is what moves the reported factor from 0.1396 to 0.1593.

**7. A chance line at 10% on a validation set whose baseline is 10.84%.
[verified]** Cell 28 draws `ax[1].axhline(10)` on validation accuracy. The 5,000
validation images are a random subsample with class counts
`[473, 542, 535, 482, 523, 473, 502, 469, 518, 483]`, majority rate
`542/5000 = 10.84%`. The 10% line belongs to the test split. Minor in size,
identical in kind to the §2.1 defect the guidelines were written for.

## A count that does not resolve — §7.3

**8. "all four of the previous lecture's silent failures" — there are three, and
the cell lists five. [verified]** Cell 30's `left_open`. Lecture 12 adds exactly
**three** questions to the red-team list (its cell 60, questions 6, 7 and 8:
`zero_grad` placement, `model.eval()`, per-batch metric means) and has three ⚠
sections (8, 9, 11). Cell 31 enumerates **five** numbered checks. Four matches
neither.

## A conclusion printed regardless of the evidence — §3.2, §6.3

**9. The overfit diagnostic hard-codes its verdict. [verified by reading]** Cell
31 ends with an unconditional
`print("The loop can memorise. So the loop is not the bug.")` and has **no
assert**, in a notebook whose very next code cell asserts its premise
(`assert sweep[2] > sweep[20]`). The strongest diagnostic in deep learning,
rendered incapable of failing. Same category as the saturation column it later
criticises — and the notebook does not notice the symmetry. *I did not run this
cell (it trains), so I cannot report the accuracy it prints; the defect is
structural and visible in the source.*

## A defect the notebook holds up as clean — §2.2, §8.2

**10. The assistant's cell contains Lecture 12 §11's bug, and the notebook says
it does not. [verified]** Cell 38: *"the `zero_grad`, the `eval()` and the
whole-set metric are all there, and a review looking for Lecture 12's failures
finds nothing."* Its *accuracy* is whole-set, but its *loss* is a plain mean of
per-batch values with the wrong denominator (item 4) — which is verbatim Lecture
12's red-team question 8, *"Is any metric a plain mean of per-batch values?"*.
This is the one defect in the notebook that would actually catch the skimmer
(§8.2), and the prose currently tells the reader it is not there.

## Annotation budget — §6.1

**11. Twenty prompt boxes, twenty full three-bullet annotations. [verified]**
Machine-counted: 20 boxes, of which 20 contain all of `Left open`, `The usual
student version` and `How you would catch it`. The guideline is **5–8, never more
than ten**. This is the measured defect that made three readers stop reading
around cell 30 — and cell 30 here is the box for the bug-ruling-out cell,
immediately before the assistant section that the whole lecture turns on.

## Examinability marking — §8.3

**12. The string "examinable" appears exactly once in the notebook, and it is in
a code comment. [verified]** Cell 3, line 2: `# Not examinable: engineering
hygiene…`. No markdown cell in the file carries an examinability marker. §8.3
requires one per section; there are twelve sections. This is character-for-
character the defect §8.3 was written about in lecture 19.

## Staging — §8.1

**13. The assistant defect is announced four times before the cell runs.
[verified by reading]** In order, all *above* code cell 37: the section heading
`## 9 · ⚠ An assistant writes the network`; the line **"Read before running."**;
and then the prompt box, whose three bullets give away the entire answer before
a single line executes — `left_open` states *"what would this number be if the
model had learned nothing at all?"*, `student` states *"the result is
indistinguishable from guessing"*, and `catch` quotes the exact plateau loss. The
preferred shape (§8.1) is to let the cell run unannounced, have the reader write
the number down, and open the next section with the ⚠. Everything needed is
already in the file; it only needs moving below cell 37.

## Timings — §7.1

**14. No CPU figure anywhere; "depending on the runtime" is the only
qualification. [verified by measurement]** Three ⏱ markers exist (cells 4, 24,
32) and none gives a CPU number, while cell 3 tells CPU users only that
everything "is slower". Measured here (laptop CPU, torch 2.13, per-step timings
extrapolated over 79 batches × 20 epochs rather than by running the cells):

| cell | notebook says | measured, 12 threads | measured, 2 threads | Colab CPU (2 vCPU), estimated |
|---|---|---|---|---|
| 26, train depth 20 | "40–90 s" | 40 s + 4 s eval | 33 s | 1–2 min |
| 34, the sweep | "about 2 minutes" | 126 s + 22 s eval | 87 s | 3–6 min |
| 37, the assistant | *(untimed)* | ~10 s | ~8 s | ~20 s |
| 50, `grad_profile` | *(untimed)* | 0.32 s both dtypes | — | ~1 s |

The two stated figures are defensible; the omission is the missing CPU/GPU
split that §7.1 requires. The Colab column is an **extrapolation, not a
measurement** — I have no Colab runtime.

## Checked and clean

Verified with `python3` and **not** defective, listed so the report is not
one-sided:

- **§5.1 / §5.2** — zero markdown lines indented ≥ 4 spaces outside a fence;
  zero fence markers indented ≥ 4. Scanned all 44 markdown cells.
- **§3.1** — the only two fenced blocks in markdown (cells 17 and 60) are
  worksheet blanks with no language tag, not `python` blocks purporting to quote
  code. No code is quoted in prose that does not exist.
- **§3.3, cross-references** — all resolve. `Lecture 2` = "Your RMSE was a lie" ✓;
  `Lecture 4` = "It never fires" ✓; `Lecture 12, section 11` = "⚠ Averaging the
  metric per batch" ✓ (checked in `lecture-12.ipynb`); `Lecture 15` = "Visual
  inspection" ✓; "the next lecture" = 14, "Making it train" ✓; "the metric in
  section 4" ✓; "Nothing uses it until section 10" ✓; "reviewer question 5" =
  "What is the default I did not ask for?" ✓.
- **§4.2, idempotency** — every training cell re-instantiates. Cell 26 re-seeds
  and rebuilds `deep`; cell 31 rebuilds `tiny` and its optimiser; cell 34
  rebuilds per depth; cell 37 rebuilds `model`. All four are re-runnable. This
  notebook is materially better than lecture 19 here.
- **§4.1, type rebinding** — no name is bound to two different types at module
  scope. `acc` is a float at module scope (cell 37) and an `ndarray` inside
  `grad_profile` (cell 50), but the latter is a local; a naive cross-cell
  checker will flag it, a correct one will not. Worth renaming anyway.
- **Cell 59's re-seeding is correct.** `torch.manual_seed(42); before =
  make_net()` does reproduce the tensors `deep` started from, because cell 26
  seeds immediately before its own `make_net()` and `train` does not
  re-initialise the net it is handed.
- **Cell 34's sweep is genuinely controlled**, and stronger than it claims: the
  depth-20 row must reproduce cell 26's test accuracy to all four decimals, since
  both re-seed to 42 before `make_net` and `train` re-seeds again. The notebook
  never points this free consistency check out.
- **The `+ 1e-7` on the scaler is inert on this dataset** — smallest per-pixel sd
  over the fit subset is **0.2248**. The annotation is honest that it is there
  for a dataset that has a constant pixel.
- **The float64 gradient story is sound.** Largest relative float32/float64
  disagreement **5.62e-06**, zero norms underflowed, assert passes. The stated
  rationale is real: layer-1 gradient entries are around `8e-20` and their
  squares (`6.8e-39`) are below float32's smallest normal `1.18e-38`. The irony
  stands — that argument is made carefully for the backward pass and not applied
  to the forward pass two cells earlier, where it decides the headline number.
- **All shapes, counts and parameter arithmetic** — `(50000,32,32,3)`,
  `(10000,32,32,3)`, exact balance in both published splits, `3,072` numbers per
  image, `500,210` parameters `= 307,300 + 19×10,100 + 1,010`, `21` weight
  matrices, hidden-weight sd `0.057889` against `0.1/√3 = 0.057735`, baseline
  exactly `0.1000`, `ln(10) = 2.3026`.

## Could not verify

- **Everything downstream of a training run**: the test accuracy of `deep`, the
  sweep accuracies, the assistant's accuracy, the epoch-1 and epoch-20 losses,
  the epoch-wise gradient history in cell 56, the relative weight changes in cell
  59, and the printed overfit accuracy in cell 31. The brief forbids executing
  training cells. Where the current prose quotes such a number I have said so;
  where my own script quotes one, it is a range with the reasoning attached, not
  a digit.
- **Whether the notebook's `2.42e-17` came from a float64 run or a trained
  float32 run.** It cannot have come from the shipped cell either way (item 1),
  but I cannot say which of the two produced it.
- **Colab wall clocks.** Every ⏱ in this script is measured locally and
  extrapolated; the extrapolation factor (1.5–2.5×) is an estimate.
- **Rendering.** §10.8 asks for a read of the rendered page, not the source. I
  checked the source mechanically for §5.1–5.2 and found it clean, but I have not
  opened the notebook in Colab.
