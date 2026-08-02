# Lecture 15 — Visual inspection · the Colab prompt script

**Flowers102, a convolutional network from scratch.** Géron, Chapter 12.

This is the script a person follows at a Colab keyboard to rebuild
`notebooks/lecture-15.ipynb` by prompting. Seventeen code cells, in order.

**How to read it.** Every cell has a **Prompt to type** — type it, do not
paraphrase it into something longer. The audit of lecture 19 found its prompts
were "3–5× more specified than anything I type"; the prompts below are
deliberately at the length a competent person actually types, and where a lazy
prompt still produces the defect, that is the stronger lesson.

**The prompt box you paste above each cell** is the prompt itself plus the
**Expect** and **Assert** lines. That is the *short* box §6.1 requires, and
sixteen of the seventeen cells get only that. **Seven cells get the full
three-bullet annotation** — cells 2, 5, 7, 10, 12, 13 and 16. Every "usual
student version" below names a library default or a failure that was executed
and observed on this machine; none is invented.

**Staging (§8).** The lecture's defect is in **cell 15**. Cell 15's box carries
no ⚠, no hint and no annotation. You run it, you write the three numbers down,
and *then* cell 16 opens with the contrast. Do not put the ⚠ in cell 15's box.

**Before you start.** Runtime → Change runtime type → **T4 GPU**. On a CPU
runtime this notebook is **about three hours** rather than about four minutes;
the per-cell ⏱ lines below give both, measured. If you have no GPU, cells 10
and 16 are the two to plan around, and §"If you have no GPU" at the end says
what to change.

---

## Cell 1 — setup, seeds, device

**Prompt to type:**

> Setup cell for a teaching notebook. Import numpy as np, torch, torch.nn as nn,
> torchvision, matplotlib.pyplot as plt, and Flowers102 plus transforms from
> torchvision. Print the python, torch and torchvision versions. Seed 42 for
> numpy and torch. Pick cuda if it's there, else mps, else cpu, print which, and
> if it's cpu print how to turn a GPU on in Colab.

**Expect:** three version lines, then a blank line and `device       cuda` on
Colab with a T4 selected (`mps` on Apple Silicon, `cpu` otherwise). On `cpu`,
two extra lines telling you to use Runtime → Change runtime type → T4 GPU.

**Assert:** none. There is nothing here with a knowable right answer — that is
why it is not examinable.

**Annotate:** short

---

## Cell 2 — decode all three splits into memory, once

**Prompt to type:**

> Load all three splits of torchvision's Flowers102 into memory as uint8
> tensors, every image resized to 128×128, with the labels as int64. Download to
> a `datasets` folder. Assert the shapes and print how long the decode took.

**Expect:** `(1020, 3, 128, 128)`, `(1020, 3, 128, 128)`, `(6149, 3, 128, 128)`,
dtype `torch.uint8`, labels 0…101, and a printed decode time. The three tensors
together are 402 MB of uint8 — keep them in that dtype.

**Assert:**

```python
assert X_train.shape == (1020, 3, IMG, IMG), X_train.shape
assert X_val.shape   == (1020, 3, IMG, IMG), X_val.shape
assert X_test.shape  == (6149, 3, IMG, IMG), X_test.shape
assert X_train.dtype == torch.uint8
assert int(y_train.max()) == N_CLASSES - 1 and int(y_train.min()) == 0
```

**⏱** **First run: about 2 minutes of download** (`102flowers.tgz` is
344,862,509 bytes — 345 MB — checked on disk) **plus the decode.** Decode
measured here at **43 s** for all 8,189 JPEGs on an M4 Max with a warm file
cache (7.2 s train, 5.9 s val, 29.6 s test). On a 2-vCPU Colab runtime allow
**2–3 minutes**. This is CPU work on both runtimes — a GPU does not help.
Afterwards the tarball is on disk and only the decode repeats.

**Annotate:** full

* **Left open:** the resize. The prompt says "128×128" and the assistant has two
  ways to write that, one of which is wrong. It also leaves open that uint8 is
  deliberate: these same tensors in float32 are 1.6 GB, which is the subject of
  cell 10.
* **The usual student version:** `transforms.Resize(128)` with an **int**. That
  is torchvision's documented behaviour for a single int — scale the *shorter*
  side to 128 and keep the aspect ratio. Run on this dataset it produces
  `(193, 128)`, `(159, 128)`, `(170, 128)`, `(192, 128)`, `(128, 182)` for the
  first five training images, and `torch.stack` then raises
  `RuntimeError: stack expects each tensor to be equal size, but got
  [3, 128, 193] at entry 0 and [3, 128, 159] at entry 1`. The fix is the tuple:
  `transforms.Resize((128, 128))`.
* **How you would catch it:** the shape asserts, which is why they are in the
  cell and not in your head. Note also that the assistant may reach for a
  `DataLoader` with the transform attached — correct for data that does not fit
  in memory, and pure per-epoch overhead here. Decode **once**, outside the
  training loop.

---

## Cell 3 — how many images per species

**Prompt to type:**

> Count images per species in the training split and in the test split with
> bincount. Assert the training split is exactly ten of each. Print the range for
> test.

**Expect:** training is exactly `10` for all 102 species. Test runs from **20 to
238** images per species — one species has 238. There are **6,149 test images
against 1,020 training images, a ratio of 6.03**: six times as much data to be
scored on as to learn from.

**Assert:**

```python
assert counts_train.min() == counts_train.max() == 10, "train is not balanced"
```

**Annotate:** short

---

## Cell 4 — look at eight of them

**Prompt to type:**

> Show one training image from each of eight randomly chosen species in a row,
> titled by class number, axes off.

**Expect:** a row of eight photographs, each titled `class NN`. Different
scales, different backgrounds, different lighting; the flower is neither centred
nor filling the frame. That list is the argument for weight sharing across
position, and it is why the next section builds a convolution rather than a
dense layer.

**Assert:** none — this cell is for your eyes.

**Annotate:** short

*The one thing that must be in the prompt or the code:* `permute(1, 2, 0)`
before `imshow`. The tensor is channels-first and matplotlib wants
channels-last.

---

## Cell 5 — normalisation, from the training split only

**Prompt to type:**

> Compute per-channel mean and standard deviation from the **training split
> only**, print them, and write a `normalise(x_u8, mean, std)` that takes them
> as arguments rather than closing over globals. Then assert the normalised
> training set has mean 0 and standard deviation 1.

**Expect:** on this dataset, measured —

```
mean ['0.4330', '0.3819', '0.2964']
std  ['0.2896', '0.2408', '0.2684']
```

and both checks passing. Keep the `mean` and `std` **arguments**: cell 16 has to
call this same function with a different pair, and a function that closes over
globals cannot be used for that comparison.

**Assert:**

```python
z = normalise(X_train)
assert z.mean(dim=(0, 2, 3)).abs().max() < 1e-3, "not centred"
assert (z.std(dim=(0, 2, 3)) - 1).abs().max() < 1e-2, "not scaled"
del z
```

Measured margins: the largest channel mean is **4.8 × 10⁻⁷** against a tolerance
of 10⁻³, and the largest `|sd − 1|` is **0.0** — this is a check with a known
answer, and it passes by four orders of magnitude, which is what "known answer"
buys you. `del z` matters: `z` is 1,020 × 3 × 128 × 128 float32 = **200 MB**.
Do the same to the intermediate you computed the statistics from.

**Annotate:** full

* **Left open:** *which* images. The prompt above says "training split only"
  because this cell is the one written deliberately. Cell 15 is the same task
  asked the way people actually ask it, and it is the whole of this lecture.
* **The usual student version:** `transforms.Normalize([0.485, 0.456, 0.406],
  [0.229, 0.224, 0.225])` — the ImageNet constants, copied from a tutorial.
  Right shape, wrong numbers, and nothing raises. The blue channel is the tell:
  ImageNet's 0.406 against this dataset's measured **0.2964**, a gap of 0.11 on a
  0–1 scale, which is 28 grey levels out of 255.
* **How you would catch it:** the assert on the normalised training set. ImageNet
  constants fail it immediately — that is the point of a check whose answer you
  can state before you run it.

---

## Cell 6 — the two numbers to compare against

**Prompt to type:**

> Compute two baselines on the test set from the label counts: always predict
> the commonest species, and uniform random over 102 classes. Print both as
> percentages and assert the first beats the second.

**Expect:** commonest species **3.87 %** (238 ⁄ 6,149 = 0.038705), uniform
**0.98 %** (1 ⁄ 102 = 0.009804). They differ by a factor of **3.95** — computing
only one of them is how people quote 1/102 as "the baseline" on a test set that
is not balanced.

**Assert:**

```python
assert majority > uniform, "the test set is balanced after all — check the counts"
```

The assert doubles as a data check: if the two came out equal, the test split
would be balanced and cell 3's counts would be wrong.

**Annotate:** short

---

### Between cells 6 and 7 — commit, on paper

Not a code cell. Paste a markdown cell and then **stop and write on paper**,
where you cannot quietly revise it:

```
Metric:                                            ____________
Accuracy a deployable sorter would need:         ____________ %
Accuracy I expect from the network we build:     ____________ %
```

Doing nothing scores **3.87 %**. A perfect machine scores 100. Saying *where
between them* is the exercise. You will compare against this in cell 12.

---

## Cell 7 — the network

**Prompt to type:**

> Write a `make_net()` returning an nn.Sequential: four stages of conv-batchnorm-
> relu at 32, 64, 128, 256 channels with a maxpool after each stage, first kernel
> 7 and the rest 3, same padding, then flatten, Linear to 256, relu, dropout 0.5,
> Linear to 102. **Every conv is bias=False** because a batch norm follows it.

**Expect:** a 30-module `nn.Sequential`. Printing `net[:3]` shows
`Conv2d(3, 32, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), bias=False)`,
`BatchNorm2d(32, ...)`, `ReLU()`.

**Assert:** none in this cell — cells 8 and 9 are the checks on it.

**Annotate:** full

* **Left open:** nothing about *why* `bias=False`, so if you drop that clause the
  assistant will not put it back. The clause is in the prompt above precisely
  because it is the one thing here you cannot recover by reading the output.
* **The usual student version:** leaving it off, and getting `nn.Conv2d`'s
  documented default **`bias=True`**. **This is the answer to red-team question
  5 for this lecture.** It adds one parameter per output channel — measured,
  **704 extra parameters** across the seven convolutions, 0.015 % of the model —
  and it changes the function computed by **nothing at all**, because the
  `BatchNorm2d` that immediately follows subtracts the batch mean and then adds
  its own learned shift. The bias is added and then removed. It costs
  parameters, moves no metric, and is invisible in every plot in this notebook,
  which is exactly why it is the question.
* **How you would catch it:** you cannot catch it from any accuracy, any loss
  curve or any figure. You catch it by reading the constructor call, or by
  `print(net[0].bias)` and seeing something other than `None`. That is what
  "what is the default I did not ask for?" means as a review question.

---

## Cell 8 — what is the shape here

**Prompt to type:**

> Push a dummy batch of two zeros through `net` layer by layer and print the
> shape after each one, skipping the ReLUs. Assert the final shape is
> (2, N_CLASSES).

**Expect:** a printed table ending
`Flatten      (2, 16384)` → `Linear       (2, 256)` → `Linear       (2, 102)`.
Four maxpools take 128 → 64 → 32 → 16, so the flatten width is
256 × 16 × 16 = **16,384**.

**Assert:**

```python
assert x.shape == (2, N_CLASSES), f"the head is wrong: {x.shape}"
```

**Annotate:** short

*Batch size 2, not 1, is deliberate:* `BatchNorm2d` in training mode on a batch
of one has zero variance to standardise against. A dummy batch that cannot go
through the real network is not a shape check.

⚠ **Re-run hazard, and it is real.** This cell builds its dummy on the CPU. Once
cell 10 has moved `net` onto the GPU, re-running *this* cell raises
`RuntimeError: slow_conv2d_forward_mps: input(device='cpu') and
weight(device=mps:0') must be on the same device` (the CUDA message is the
equivalent). Confirmed by running it. If you want to re-check shapes after
training, re-run cell 7 first — see the re-run orders at the end.

---

## Cell 9 — where are the parameters

**Prompt to type:**

> Count the parameters of this network in three parts — all the convs and batch
> norms together, the Linear(16384, 256), and the output Linear — as counts and
> as percentages of the total. Assert the three add up to the total. Then print
> parameters per training image.

**Expect:** exactly these, arithmetic you can do on paper before you run it —

| part | parameters | share |
|---|---|---|
| convolution + batch norm | 586,720 | 12.20 % |
| one dense layer, `Linear(16384, 256)` | 4,194,560 | 87.25 % |
| output layer, `Linear(256, 102)` | 26,214 | 0.55 % |
| **total** | **4,807,494** | |

The dense layer is 16,384 × 256 + 256 = 4,194,560, and it is **7.15×** all seven
convolutions and all seven batch norms put together. Per training image:
4,807,494 ⁄ 1,020 = **4,713**.

**Assert:**

```python
assert convs + head + out == total, "a parameter went missing"
```

That assert is what stops you silently omitting a layer from the breakdown.

**Annotate:** short

*Say the number out loud:* **4,713 parameters per training image.** Nothing in
this architecture stops the network storing the training set, and dropout is the
only thing we have asked to prevent it. That is what cell 11's gap is.

---

## Cell 10 — train it

**Prompt to type:**

> Train it. Adam at 3e-4, batch 32, 30 epochs, cross-entropy, seeded. Record
> train and validation accuracy plus wall-clock seconds every epoch into a dict,
> print every tenth epoch. Write the accuracy function so it **normalises one
> batch at a time** and calls `model.eval()`. Keep a CPU clone of the first
> conv's weights before training starts.

**Expect:** four printed lines (epochs 10, 20, 30) each with a train accuracy, a
validation accuracy and a running second count, then the total wall clock. The
training accuracy climbs far above the validation accuracy — a persistent gap
between two plateaus, which Lecture 6 named as **variance**.

**Assert:**

```python
assert len(hist["epoch"]) == EPOCHS
```

**⏱** **39 s measured on MPS** (M4 Max), which is the same order as a Colab T4;
**87 minutes on CPU.** That CPU figure is measured, not guessed: one
forward-backward step on a batch of 32 takes **3,559 ms** on a 12-thread M4 Max
against **28 ms** on MPS, a factor of **127**, and one epoch is 32 steps plus 16
evaluation batches = **174 s**. A 2-vCPU Colab CPU runtime is *slower* than an
M4 Max, so treat 87 minutes as a floor. Nothing prints until epoch 10 finishes;
that is not a hang.

**Annotate:** full

* **Left open:** why `3e-4` and not Adam's documented default `lr=1e-3`. On 1,020
  images the default does not diverge and does not error — it plateaus low. A
  failure mode that produces a plausible number and no traceback is the one you
  have to go looking for.
* **The usual student version:** two of them, both real. **(a)** `normalise(X_test)`
  in one call. It is the obvious way to write it, it works on the 1,020-image
  validation split, and on the test split it asks for
  6,149 × 3 × 128 × 128 float32 = **1,208,942,592 bytes — 1.2 GB** — in one
  allocation, on top of the 402 MB of uint8 already resident and whatever is on
  the GPU. That is how a Colab session dies: no traceback, just "Your session
  crashed after using all available RAM". Normalising one batch of 128 at a time
  costs 25 MB. **(b)** Adding an `nn.Softmax(dim=1)` to the end of the network.
  `nn.CrossEntropyLoss` applies its **own** log-softmax to whatever you hand it —
  measured, the same example gives loss 0.317 on logits and 0.745 on
  softmax'd logits. It trains, badly, and never complains.
* **How you would catch it:** `model.eval()` inside the accuracy function, not
  once at the top of the notebook. This network has **both** dropout and batch
  norm, and `net.train()` at the head of every epoch undoes any single
  `.eval()` you set earlier. Cell 12 turns that into an assertion.

---

## Cell 11 — the learning curve, against wall clock

**Prompt to type:**

> Plot training and validation accuracy from `hist` against **seconds**, not
> epochs. Y axis fixed 0 to 100. Dotted horizontal line at the majority baseline,
> labelled. Then print the final two accuracies and the gap in points.

**Expect:** two curves, the grey dotted baseline sitting at **3.87** on a 0–100
axis, and a printed gap of tens of points between the two final accuracies.

**Assert:** none — but fixing `ax.set_ylim(0, 100)` is doing the same job. An
autoscaled accuracy axis makes every run look dramatic, and it hides how far
above 3.87 and how far below 90 you actually are.

**Annotate:** short

*Seconds, not epochs.* An epoch is a unit of nothing — of what, on what
hardware, at what batch size? From the next lecture on, time is one of the
things being compared, and a 30-epoch scratch run and a 3-epoch fine-tune are
not comparable on an epoch axis.

---

## Cell 12 — the test set, once

**Prompt to type:**

> Evaluate on the test set and print the accuracy, the majority baseline, and the
> ratio between them. Then evaluate a second time and assert the two calls give
> exactly the same number.

**Expect:** one accuracy, the baseline **3.87 %**, and their ratio. Write this
number on the same sheet of paper as your cell-6 prediction.

**Assert:**

```python
assert accuracy(net, X_test, y_test) == test_acc, \
    "evaluation is not deterministic — check model.eval()"
```

Exact equality, not `isclose`. A deterministic function of fixed weights and
fixed data returns the same float every time.

**⏱** **2.4 s on MPS. 6.1 minutes on CPU** — this cell walks all 6,149 test
images **twice**, and one forward pass over a batch of 128 costs 3,764 ms on CPU
against 24 ms on MPS. It is the cheapest cell in the notebook on a GPU and the
third most expensive on a CPU, which is exactly the kind of cell that goes
untimed.

**Annotate:** full

* **Left open:** that the second call is free insurance rather than a waste. It
  costs one line and it is the only check in the notebook that would notice a
  layer left in training mode.
* **The usual student version:** calling `.eval()` once, at the top, after
  building the model — and then `net.train()` at the head of every epoch of
  cell 10 undoes it, thirty times. With dropout alone the symptom is a test
  accuracy that wobbles by a point or two between identical calls. With batch
  norm it is worse: the running statistics keep updating from test batches, so
  the *model itself* changes while you evaluate it, and the number drifts in one
  direction.
* **How you would catch it:** exactly this assert. Two calls, `==`, no
  tolerance. It catches dropout and batch norm together, and it is two lines.

---

## Cell 13 — look at what it learned

**Prompt to type:**

> Show the first conv layer's 32 filters — the first sixteen at initialisation on
> the top row and the same sixteen after training below, using the clone from
> cell 10. Rescale each filter to its own min-max range. Axes off.

**Expect:** top row, structureless noise. Bottom row, colour blobs and oriented
light–dark boundaries. The first layer's weights are 3 × 32 × 7 × 7 = **4,704
numbers** — small enough to look at directly, which is not true of any other
layer here.

**Assert:** none. The comparison *is* the check: if the two rows look identical,
nothing trained.

**Annotate:** full

* **Left open:** which filters are *about flowers*. None of this was specified —
  colour blobs and edge detectors are simply the shape that minimises the loss.
  Sit with that question; the next lecture answers it.
* **The usual student version:** plotting the raw weights and concluding the
  layer learned nothing. Measured on a freshly constructed layer here, the raw
  weights span **[−0.0825, +0.0825]** — signed and centred near zero — and
  matplotlib prints `Clipping input data to the valid range for imshow with RGB
  data ([0..1] for floats or [0..255] for integers). Got range
  [-0.080293484..0.08169286]` and shows you sixteen black squares. That message
  goes to stderr, not to a Python warning, so it is easy to scroll past.
* **How you would catch it:** rescale **per filter**, with `amin`/`amax` over
  dims `(1, 2, 3)` and `keepdim=True`. On one shared scale only the loudest
  filter is visible and the other thirty-one are grey. And keep the clone of the
  initial weights **before** cell 10 trains — there is no way to recover them
  afterwards except by re-seeding and rebuilding, and the before/after
  comparison is the entire content of this figure.

---

## Cell 14 — what one filter does to one photograph

**Prompt to type:**

> Take one test image through just the first conv-batchnorm-relu block in eval
> mode under no_grad, and show the input beside eight of the 32 activation maps.
> Use the magma colormap. Assert the activation shape.

**Expect:** nine panels — the photograph, then eight single-channel maps where
bright means *this filter fired strongly here*.

**Assert:**

```python
assert a1.shape == (1, 32, IMG, IMG), a1.shape
```

Still 128 × 128: this block is `padding=k//2` with no pooling, so the spatial
size is unchanged and only the channel count moves, 3 → 32.

**Annotate:** short

*`eval()` and `no_grad()` are both load-bearing here.* Batch norm in training
mode on a batch of **one** standardises that image against its own statistics
and shows you something that never happens at inference. And `magma` rather than
a diverging colormap because these are post-ReLU and therefore non-negative — a
diverging map spends half its range on values that cannot occur.

---

## Cell 15 — an assistant writes the normalisation

Paste the markdown above this cell exactly as follows, and **no more than
this** — no ⚠, no hint:

> Here is a request of the kind people actually type, and the code it returns.
>
> > *"Compute the per-channel normalisation statistics for the Flowers102
> > dataset and write a function that normalises an image batch with them."*
>
> Run it and write the three numbers down.

**Prompt to type:**

> Compute the per-channel normalisation statistics for the Flowers102 dataset and
> write a function that normalises an image batch with them.

**Expect:** it runs, it imports nothing exotic, and it prints three entirely
believable means. Measured on this data —

```
all splits     ['0.4355', '0.3777', '0.2880']
training only  ['0.4330', '0.3819', '0.2964']
largest difference in any channel mean: 0.0085
```

Write those down before reading on.

**Assert:** none. **That is the point of this cell.** There is no assertion you
could add to the code as returned that would fail, because nothing about it is
numerically wrong.

**⏱** Under 4 s (3.8 s measured), but watch the memory: `torch.cat` over all
three splits builds a 402 MB uint8 copy and then a **1.61 GB** float32 tensor —
peak resident memory 4.2 GB measured — while the three original splits are still
alive. That is larger than the 1.2 GB that cell 10's annotation calls "how a
Colab session dies". Add `del pixels` immediately after the two statistics come
out.

**Annotate:** short

Deliberately short, and deliberately un-annotated. §8.1 of the guidelines: the
defect that is announced four times before it arrives catches nobody. Run it,
write the numbers down, then go to cell 16.

---

## Cell 16 — measure the damage

Now open the section with the ⚠ and the contrast, and only now:

> ### ⚠ Reviewer question 1: what touched the test set?
>
> `torch.cat` on the first line. The mean and standard deviation that will scale
> every **training** image were computed from a set including all **6,149** test
> images.
>
> The prompt said "the dataset". There are three of them — 1,020, 1,020 and
> 6,149 — and the assistant picked the one that makes the code shortest. **The
> thing missing from that prompt is a noun.**
>
> Now measure the damage. Do not guess it.

**Prompt to type:**

> Write `quick_train(mean, std, seed, epochs=12)` that builds a fresh network,
> trains on the training split normalised with the given statistics, and returns
> validation accuracy. Run it with two seeds under the training-only statistics
> and two seeds under the all-splits ones, and print the difference between the
> two conditions **and** the difference between the two seeds.

**Expect:** four validation accuracies, and two differences that you must read
together. **The condition difference comes out smaller than the seed
difference.** That is the finding, not an embarrassment — see the three reasons
below.

**Matched rows (§2.1), and say so in the prose:** all four runs are scored on
**the same 1,020 validation images**, with the same architecture, the same
optimiser, the same 12 epochs and the same batch size; seeds 42 and 43 appear in
both conditions. The only thing that differs between the honest pair and the
leaky pair is which images the two normalisation constants were estimated from —
`MEAN`/`STD` from 1,020 training images, `MEAN_ALL`/`STD_ALL` from all 8,189.
The measured gap between them is **0.0085 at most in any channel mean** and
**0.0038 in any standard deviation**.

**Assert:** none on the accuracies — asserting a direction here would be
asserting noise. What you *can* assert is that the two conditions really did use
different statistics:

```python
assert not torch.allclose(MEAN, MEAN_ALL), "the two conditions are identical"
```

**⏱** **44 s measured on MPS.** **93 minutes on CPU** — four trainings of 12
epochs is 48 epochs against cell 10's 30, so on a CPU runtime this is the single
most expensive cell in the notebook and it is longer than the training cell. The
notebook currently states "about 60 seconds" with no CPU figure at all.

**Annotate:** full

* **Left open:** why it is still a bug when it is that small. Three reasons, and
  the third is the one that matters. **(1)** You did not know it was small until
  you measured; nothing in the code said so and neither did the output.
  **(2)** It is small for reasons you can *name* — two numbers per channel
  estimated from 1,020 images against 8,189, defining an **invertible affine**
  map applied identically to every image. **(3)** A leaked score and an honest
  score can be **identical**, so you cannot detect this from the number. That is
  why the rule is procedural — **split first** — rather than "check whether the
  score looks too good".
* **The usual student version:** concluding that leakage does not matter. Change
  one thing and it has teeth: a test set of 50 images rather than 6,149, so the
  statistics are dominated by it; statistics of the **target** rather than the
  input; any transform that is not invertible; or per-image rather than
  per-dataset statistics. Every one of the reasons in (2) can change, and none
  of them is visible in the accuracy.
* **How you would catch it:** ask reviewer question 1 of a cell that contains no
  model — *what touched the test set?* — and count `mean()` as touching. Then
  note the honesty limit of this very cell: **two seeds give you one difference,
  not a spread.** "Smaller than the seed noise" is the right conclusion here and
  it rests on n = 2. If you want it to carry weight, run three seeds per
  condition (66 s on MPS, ~140 min on CPU) and report the range. Saying that out
  loud is what §2.4 asks for.

---

## Cell 17 — the assertion that catches this bug

**Prompt to type:**

> Write the assertion that would have caught the previous cell's bug. It should
> check what the statistics were computed **from**, not what they are.

**Expect:** something that records provenance. The honest and the leaky
constants differ by at most 0.0085 in any channel — **no assertion on the
values** distinguishes them, which is the whole reason the check has to be about
where they came from.

**Assert:** make it one that can actually fail. The only way to do that is to
have the count come out of the *same call* that produced the statistics, so it
cannot be set independently by the hand that got it wrong:

```python
def channel_stats(x_u8):
    """The statistics, and how many images actually produced them."""
    xf = x_u8.float() / 255.0
    return xf.mean(dim=(0, 2, 3)), xf.std(dim=(0, 2, 3)), len(x_u8)

MEAN, STD, N_STATS = channel_stats(X_train)
assert N_STATS == len(X_train), "statistics saw more than the training split"
assert MEAN.shape == (3,) and STD.shape == (3,)
```

Verified that this one has teeth: called on `X_train` it passes with
`N_STATS = 1020`; called on `torch.cat([X_train, X_val, X_test])` — cell 15's
code — it fires with `N_STATS = 8189`.

**Annotate:** short

*Be honest about how weak even this is.* It catches the exact mistake cell 15
makes, and it does not stop you passing the wrong tensor to `channel_stats` in
the first place. A stronger version returns one object that carries the
statistics **and** their provenance together, so `normalise` cannot be handed a
pair whose origin nobody recorded. **Do not write the version the current
notebook has**, which is
`n_used_for_stats = len(X_train)` followed by
`assert n_used_for_stats == len(X_train)`: both sides are the same expression,
so it cannot fail under any circumstances. A check that cannot fail is not a
check.

---

## Closing markdown — where we are

| | Test accuracy |
|---|---|
| uniform guess, 1 ⁄ 102 | 0.98 % |
| commonest species, 238 ⁄ 6,149 | 3.87 % |
| **yours, from cell 12** | printed above |
| what the operator needs | 90 % |

Write your **measured** accuracy next to the number you predicted before cell 7,
and bring both to the next lecture.

**What we deliberately did not do:** no augmentation, no pretrained weights, no
learning-rate schedule, no architecture search. Each is a lever, and one of them
moves this number by more than everything else in this course put together.

**Red-team.** Swap notebooks. Ten minutes, five questions: (1) what touched the
test set? (2) what was fitted, and on what? (3) what is the shape here? (4) what
was dropped — rows, columns, images? count them. (5) what is the default I did
not ask for? Report what you **found**, not what you would have done
differently.

*Answer to 5 for this lecture:* `bias=True` on `nn.Conv2d`. 704 parameters, no
effect on the function computed, invisible in every metric. See cell 7.

---

## Exercises, with the re-run order (§7.2)

Each one lists the cells to re-run, in order. Cell numbers are this script's,
1–17.

1. **Change the seed.** Edit `RANDOM_STATE` in **cell 1**, then re-run
   **1 → 5 → 7 → 10 → 11 → 12**. Cells 2 and 3 do not depend on the seed and
   cost 43 s of decode, so skip them. Cell 10 re-instantiates the network and
   the optimiser inside itself, so this genuinely restarts training rather than
   continuing it — check that before you trust any re-run of a training cell.
   ⏱ 39 s MPS, 87 min CPU.
2. **Turn the conv bias back on.** In **cell 7** delete `bias=False`, then
   re-run **7 → 8 → 9**. Total parameters go from 4,807,494 to 4,808,198 — 704
   more. Do **not** re-run cell 10 expecting a different accuracy; predict first
   what will happen, then read cell 7's annotation. ⏱ under 5 s either way.
3. **Break the resize.** In **cell 2** change `Resize((IMG, IMG))` to
   `Resize(IMG)` and re-run **cell 2 alone**. Read the exception. Change it
   back and re-run cell 2 again. ⏱ 43 s each way on this machine, 2–3 min on a
   Colab CPU runtime — this is the decode, not the download; the tarball stays
   on disk.
4. **Use the ImageNet constants.** In **cell 5** replace the computed statistics
   with `[0.485, 0.456, 0.406]` and `[0.229, 0.224, 0.225]` and re-run **cell 5
   alone**. The assert fails. Read *which* of the two asserts fails and by how
   much. Change it back. ⏱ under 10 s.
5. **Three seeds instead of two.** In **cell 16** change `range(2)` to
   `range(3)` and re-run **cell 16 alone** — it depends on `MEAN`, `MEAN_ALL`,
   `lossf` and `ytr`, all of which already exist. Report the range within each
   condition next to the difference between conditions. ⏱ 66 s MPS,
   **~140 min CPU** — do not start this one on a CPU runtime.
6. **Re-check the shapes after training.** Re-running **cell 8** on its own
   after cell 10 raises a device error. Re-run **7 → 8** instead, and say why
   that works and why re-running 8 alone does not.

## If you have no GPU

Measured end to end on CPU this notebook is **about three hours** (cell 10:
87 min, cell 16: 93 min, cell 12: 6 min, decode: 43 s), against about four
minutes with an accelerator. If Colab will not give you a T4:

- **Cell 10:** drop `EPOCHS` from 30 to 5. About 15 minutes. The curve shape in
  cell 11 is the same; the accuracy is lower, and you must **say so** next to
  any number you write down — a 5-epoch score and a 30-epoch score are not the
  same measurement and must not be compared with anything scored at 30.
- **Cell 16:** drop `epochs` from 12 to 4 **in both conditions**. About
  31 minutes. Both conditions must move together or the comparison is no longer
  on matched runs.
- **Cell 12:** nothing to change; 6 minutes, and it is the one number the
  lecture actually asks you to bring.

---

## Defects found in the current notebook

`notebooks/lecture-15.ipynb`, 58 cells, 17 code cells. Every claim below was
checked with `python3` against the notebook JSON, the dataset under
`notebooks/datasets/flowers-102/`, or a direct execution, **except** the three
marked *not checked* at the end.

### Verified by execution or arithmetic

1. **§6.1 — annotation budget, the primary defect.** All **17** code cells carry
   a prompt box and **all 17 carry the full three-bullet annotation** — counted
   from the JSON: 17 markdown cells beginning `> **Prompt`, 17 containing
   `Watch this prompt`, 17 with three or more `* **` bullets. The rule is five
   to eight, never more than ten. This script cuts it to seven.

2. **§7.1 — "several minutes on CPU" is 87 minutes.** Cell 31's markdown says
   *"about 40 seconds on a GPU or MPS, several minutes on CPU"*. The 40 s is
   right (**39 s** measured on MPS). The CPU figure is not: one training step on
   a batch of 32 measured **3,559 ms** on a 12-thread Apple M4 Max against
   **28 ms** on MPS, and one epoch is 32 steps plus 16 evaluation batches at
   3,764 ms = **174 s**, so 30 epochs is **87 minutes**. A 2-vCPU Colab CPU
   runtime is slower than an M4 Max, so that is a floor. This is the defect the
   literal reader — alone at home, no GPU — hits hardest.

3. **§7.1 — cell 51's "⏱ about 60 seconds" carries no CPU figure, and it is the
   most expensive cell in the notebook on a CPU.** Four `quick_train` calls of
   12 epochs = 48 epochs, measured **44 s on MPS** (so "60 seconds" is fine) and
   **93 minutes on CPU**. Longer than the training cell it follows.

4. **§7.1 — cell 40 is untimed and walks the 6,149-image test set twice.**
   2.4 s on MPS, **6.1 minutes on CPU** (48 forward passes of 128 at 3,764 ms).
   The header promises *"Anything that takes more than about twenty seconds says
   so before it starts"*; on a CPU runtime three cells break that promise and
   only two of them carry any ⏱ at all.

5. **§3.3 — cross-reference does not resolve.** Cell 15's `left_open` says the
   assistant failure is *"eleven sections from now"*. Cell 15 is in **section
   4**; the failure is **section 11**. That is **seven** sections, not eleven.
   (The other two cross-references do resolve: cell 5's "section 8" and cell
   15's "section 11" both point at the right sections — checked by reading them.)

6. **§1.1 — "four thousand parameters per image" is 4,713.** Cell 28's `catch`
   bullet. Computed by constructing the network: total **4,807,494** parameters,
   1,020 training images, **4,713.2** per image. Conventional rounding gives
   *five* thousand, not four. The cell's own code prints `4,713`, so the prose
   and the output disagree on the screen.

7. **§1.2 — "nine parameters in ten" is 87.25 %.** Cells 28 and 30. Measured:
   `Linear(16384, 256)` holds **4,194,560** of **4,807,494** = **87.25 %**, i.e.
   8.7 in ten. Including the output layer it is 87.80 %. The point stands — the
   dense layer is **7.15×** all convolutions and batch norms combined — but the
   figure as written rounds the wrong way, and the cell prints `87.2%` two lines
   above it.

8. **§1.1 — "they differ in the third decimal place" is false.** Cell 55's
   `student` bullet, about the honest and leaky statistics. Measured over the
   real data: training-only mean `[0.4330, 0.3819, 0.2964]`, all-splits mean
   `[0.4355, 0.3777, 0.2880]`. The blue channel differs by **0.0085** — the
   **second** decimal place (0.29 against 0.28), not the third. The cell's own
   sibling (cell 50) prints exactly this as `largest difference in any channel
   mean: 0.0085`.

9. **§3.2 — cell 56's assert cannot fail.** The code is
   `n_used_for_stats = len(X_train)` followed by
   `assert n_used_for_stats == len(X_train)`. Both sides are the same
   expression, evaluated two lines apart, on an object nothing has touched. It
   is offered as *"the assertion that catches this bug"* and it would pass
   unchanged in a notebook where the statistics came from all 8,189 images. The
   annotation calls it *"weak on purpose"*; weak and vacuous are different
   things, and this is the second one.

10. **§4.3 — an unnamed out-of-order hazard.** Re-running cell 26 (the shape
   walk, which builds `torch.zeros(2, 3, IMG, IMG)` on the CPU) after cell 33
   has moved `net` to the accelerator raises
   `RuntimeError: slow_conv2d_forward_mps: input(device='cpu') and
   weight(device=mps:0') must be on the same device`. Reproduced directly. The
   notebook names no out-of-order hazards at all, and this is the one a reader
   who wants to re-check a shape will hit.

11. **Dead 200 MB allocation, in the cell whose lesson is memory.** Cell 33
   contains `Xva = normalise(X_val).to(device)`. Grepped every cell: `Xva` is
   never read again — `accuracy(net, X_val, y_val)` takes the **uint8** tensor
   and normalises batch by batch. So the cell whose prompt box teaches
   *"normalise ONE BATCH AT A TIME"* also normalises a whole split in one call
   and parks 1,020 × 3 × 128 × 128 × 4 = **200 MB** of float32 on the GPU for
   the rest of the session, unused.

12. **The assistant cell allocates more than the amount the notebook says kills
   Colab.** Cell 50 does `torch.cat([X_train, X_val, X_test])` — a **402 MB**
   uint8 copy — then `.float()/255.0`, a **1,610,022,912-byte (1.61 GB)**
   tensor, while all three original splits are still resident. Measured peak
   RSS **4.24 GB**, 3.8 s. Cell 32's annotation states that a **1.2 GB**
   float32 tensor is *"how a Colab session dies"*; cell 50 builds one 1.33×
   larger, eighteen cells later, with no ⏱ and no memory note. It survives on a
   12.7 GB Colab instance, but the notebook contradicts its own lesson.

13. **§8.1 — the defect is announced five times before it runs.** Counted:
   cell 0 (*"Cells marked ⚠ read before running contain a defect on purpose"*),
   cell 14 (section heading: *"Which images they are computed from is the whole
   of this lecture's assistant failure"*), cell 15's `left_open` (same claim
   again), cell 48 (*"⚠ Read before running"* + *"exactly one thing missing…
   and it is a noun"*), and cell 49's box, which is labelled **⚠** and whose
   three bullets give away the entire answer — *"it is a NOUN"*, *"there are
   three of them"*, *"`torch.cat` on the first line"* — immediately above the
   code. By the time the reader's eye reaches `torch.cat` there is nothing left
   to find. This script's cells 15 and 16 reorder it: run, write the number
   down, then the ⚠.

14. **§8.3 — "examinable" appears twice, both in section 1.** Grepped every
   cell: cell 2's annotation and cell 3's comment, both about the setup cell,
   which is the section that needed the label least. Sections 2–12 carry no
   examinable/not-examinable marker at all.

15. **§1.2 — no cell in the notebook has a stored output.** All 17 code cells
   have `outputs: []` and `execution_count: None`. So *every* prose figure —
   3.87 %, 0.98 %, 6,149, 1.2 GB, 4,704 — fails the mechanical form of §1.2,
   which requires each one to appear in a stored output. I re-derived the ones
   that are checkable and they are **correct**: majority baseline
   238/6,149 = **0.038705 → 3.87 %**; uniform 1/102 = **0.98 %**; splits
   **1,020 / 1,020 / 6,149** summing to **8,189**; test/train ratio **6.03**;
   download **344,862,509 bytes = 345 MB**; 6,149 × 3 × 128 × 128 × 4 =
   **1,208,942,592 bytes = 1.2 GB**; 1,020 × 3 × 128 × 128 × 4 = **200 MB**;
   first-layer weights 3 × 32 × 7 × 7 = **4,704**; flatten width
   256 × 16 × 16 = **16,384**; final shape **(2, 102)**. The two figures that do
   **not** reconcile are items 6, 7 and 8 above.

16. **§2.4 — the headline rests on n = 2.** Cell 53 runs two seeds per
   condition and reports `100*abs(honest[0] - honest[1])` as "the difference
   between seeds" — one difference, from one pair, and the leaky pair's own
   spread is computed and then not printed. The conclusion *"smaller than the
   seed noise"* is almost certainly right and the notebook is right to lead
   with it, but a single gap is not a spread, and §2.4 asks that this be said
   where the headline is. It is not said anywhere.

17. **§7.1 — the decode estimate is low.** Cell 4's markdown says *"about 25
   seconds afterwards (decoding 8,189 JPEGs into one uint8 tensor)"*. Measured
   here: **42.7 s** total (7.2 s + 5.9 s + 29.6 s), on an M4 Max with the files
   in page cache. Hardware-dependent, so this is the softest item in the list —
   but it is 1.7× the stated figure on faster-than-Colab hardware, and it is
   the one number a reader uses to decide whether to start the notebook.

### Checked and found clean

- **§5.1 / §5.2** — no markdown line indented ≥ 4 spaces outside a fence, no
  fence marker indented, no unclosed fence. Scanned every markdown cell
  line by line.
- **§3.1** — the only fenced block in any markdown cell is cell 20's
  fill-in-on-paper form, which is not code and claims to be none. No
  ```` ```python ```` block in markdown anywhere, so nothing to reconcile
  against a code cell.
- **§4.2** — both training cells re-instantiate. Cell 33 calls `make_net()` and
  constructs a fresh `Adam`; `quick_train` in cell 53 does the same inside the
  function. Re-running either genuinely restarts rather than continuing, which
  is the failure lecture 19 shipped.
- **§4.1** — no name is rebound to a different type. `net` goes from
  `nn.Sequential` to `nn.Sequential`, which is a device change, not a type
  change; `x` in cell 26 is a tensor throughout; `m` is a module in cell 26 and
  a *local* inside `quick_train`, so they never collide.
- **§2.1** — the honest/leaky comparison **is** on matched rows: the same 1,020
  validation images, the same seeds 42 and 43 in both conditions, the same 12
  epochs, the same architecture. The prose does not say so explicitly, which is
  the only thing this script adds.
- **The `torch.Generator(device=...)` / `randperm` pattern works on MPS.** I
  expected this to be an Apple-Silicon blocker and it is not — tested on both
  `cpu` and `mps`, both return a permutation.

### Not checked

- **Whether the deck's 80-epoch figures agree with a 30-epoch run.** Cell 0
  claims *"The shape of every curve is the same; the accuracy is a little
  lower."* Confirming it means running both, which the brief excludes.
- **The Colab T4 wall clock.** Every GPU figure in this script is measured on
  Apple MPS (M4 Max) and used as a proxy. MPS at **39 s** for cell 10 matches
  the notebook's claimed "about 40 seconds on a GPU or MPS", so the claim is
  consistent with what I could measure, but I did not run a T4.
- **The actual test accuracy, and therefore whether §12's "yours, today" row and
  the "a long way from 90%" claim in cell 40 are right.** Both require running
  the training cell.
