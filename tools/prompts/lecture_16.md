# Lecture 16 — Don't train from scratch · the Colab prompt script

**Flowers102, transfer learning.** Géron, Chapter 12. *Mathematical thread 8:
weight sharing, equivariance, and where the memory goes.*

This is the script a person follows at a Colab keyboard to rebuild
`notebooks/lecture-16.ipynb` by prompting. **Twenty-three code cells, in order.**

**How to read it.** Every cell has a **Prompt to type** — type it, do not
paraphrase it into something longer. The audit of lecture 19 found its prompts
were "3–5× more specified than anything I type"; the prompts below are at the
length a competent person actually types. Where a lazy prompt still produces the
defect, that is the stronger lesson.

**The box you paste above each cell** is the prompt plus the **Expect** and
**Assert** lines. That is the *short* box §6.1 requires, and sixteen of the
twenty-three cells get only that. **Seven cells get the full three-bullet
annotation** — cells **2, 5, 9, 15, 19, 22 and 23**. Every "usual student
version" below names a documented library default or a failure executed and
observed on this machine; none is invented, and each one says which.

**Staging (§8).** The lecture's assistant failure is in **cell 21**. Cell 21's
box carries no ⚠, no hint and no annotation, and the markdown above it gives
nothing away. You run it, you write four numbers down, and *then* cell 22 opens
with the contrast. **Do not put the ⚠ in cell 21.** The current notebook gives
the answer away six times before the cell runs; that is defect 1 in the list at
the end.

**Before you start.** Runtime → Change runtime type → **T4 GPU**. On a CPU
runtime this notebook is **about 50 minutes** rather than about 3; the per-cell
⏱ lines give both, and §"If you have no GPU" at the end says what to change.
Cells **12, 19, 16 and 21** are the four to plan around, in that order.

**Where the timings come from.** Every ⏱ figure below is measured on an Apple
M4 Max (torch 2.13, torchvision 0.28, Pillow 12): accelerator figures on **MPS**
as a proxy for a T4, CPU figures with `torch.set_num_threads(2)` to stand in for
a 2-vCPU Colab runtime. The long cells are **per-step cost × step count**, taken
as the minimum of repeated measurements — the training cells themselves were not
run end to end. A 2-vCPU Colab CPU is slower than an M4 Max, so every CPU figure
is a **floor**.

---

## Cell 1 — setup, seeds, device

**Prompt to type:**

> Setup cell for a teaching notebook. Import numpy as np, torch, torch.nn as nn,
> torch.nn.functional as F, torchvision, matplotlib.pyplot as plt, transforms
> and Flowers102 from torchvision, and resnet18 with ResNet18_Weights. Print the
> python, torch and torchvision versions. Seed 42 for numpy and torch. Pick
> cuda, else mps, else cpu, and print which one.

**Expect:** three version lines, a blank line, then `device       cuda` on Colab
with a T4 (`mps` on Apple Silicon, `cpu` otherwise).

**Assert:** none.

**Annotate:** short

> Two things this cell does that the previous lecture's did not:
> `torch.nn.functional` is imported, because sections 2–5 measure cosine
> similarities and pooling directly rather than through a module; and the seed is
> **the same 42** as lecture 15. Every headline in this notebook is a difference
> of two accuracies, so the seed has to be held across both runs.

---

## Cell 2 — the same data, at two resolutions

**Prompt to type:**

> Load all three splits of torchvision's Flowers102 into memory as uint8
> tensors, twice: once resized to 128×128 and once to 224×224. Give the 224 ones
> different names. Print how long the decode took. Then compute the per-channel
> mean and std over the **training** split at 128 and write a `normalise(x_uint8)`
> that uses them. Print the majority-class share of the test split.

**Expect:** `(1020, 3, 128, 128)`, `(1020, 3, 128, 128)`, `(6149, 3, 128, 128)`
and the same three at 224. Training-split statistics
`MEAN = [0.4330, 0.3819, 0.2964]`, `STD = [0.2896, 0.2408, 0.2684]`.
`majority baseline 3.87%` — 238 images of the commonest species out of 6,149.
The six tensors together are **1,560 MiB of uint8** (384 MiB at 128 +
1,176 MiB at 224) and they stay resident for the whole notebook.

**Assert:**

```python
assert len(X_train) == len(T_train) == 1020
assert len(X_val)   == len(T_val)   == 1020
assert len(X_test)  == len(T_test)  == 6149
assert T_train.shape[-1] == TRANSFER_IMG
```

**⏱** **44 s of decode** measured at 2 threads with a warm file cache — 17 s for
the three splits at 128, 27 s for the three at 224. On a 2-vCPU Colab runtime
allow **2–4 minutes**. This is CPU work on every runtime; a GPU does not help.
First run also downloads `102flowers.tgz` (345 MB). Under load this laptop took
**3 min 28 s** for the same decode, so treat 44 s as the floor, not the promise.

**Annotate:** full

* **Left open:** *which images the statistics are computed from.* The prompt says
  "the training split" because this script says so; type "compute the mean and
  std" and you will get them over whatever tensor is nearest. It also leaves open
  that 224 is not a preference — it is what the ResNet weights were trained at,
  and it is red-team answer 3 for this application.
* **The usual student version:** statistics over all 8,189 images —
  `torch.cat([X_train, X_val, X_test]).float() / 255`. Not invented: the previous
  notebook (`lecture-15.ipynb`, cell 50) does exactly this, deliberately, and
  builds a **1.50 GiB** float32 tensor to do it. Measured on this data, the
  all-splits mean is `[0.4355, 0.3777, 0.2880]` against the training-only
  `[0.4330, 0.3819, 0.2964]` — the blue channel differs by **0.0085**. Nothing
  raises, nothing crashes, and the test set has touched the preprocessing.
  The second usual version is `transforms.Resize(128)` with an **int**, which is
  documented to scale the *shorter* side and keep the aspect ratio, so
  `torch.stack` raises `RuntimeError: stack expects each tensor to be equal
  size`. The tuple `Resize((128, 128))` is the fix.
* **How you would catch it:** the length asserts, and one print of the two byte
  counts. `X_*` at 128 is 384 MiB and `T_*` at 224 is 1,176 MiB; if either number
  surprises you, you have decoded something at the wrong size. Name the two sets
  of tensors differently — `X_` and `T_` — because passing the wrong one to a
  network raises nothing at all: ResNet is fully convolutional up to the pool, so
  128-pixel input runs and computes features at a scale the weights never saw.

---

## Cell 3 — the previous lecture's architecture, so this notebook stands alone

**Prompt to type:**

> Re-type the network from the previous lecture as `make_net()`: 7×7 conv to 32
> channels then 3×3 convs, batch norm and ReLU throughout, four max pools, then
> flatten, Linear to 256, dropout 0.5, Linear to 102. Assert it has exactly
> 4,807,494 parameters. Then write `accuracy(model, X_uint8, y, norm, bs=64)`
> that normalises **one batch at a time** and returns the fraction correct.

**Expect:** `architecture matches the previous lecture`. Nothing else prints.

**Assert:**

```python
assert sum(p.numel() for p in make_net().parameters()) == 4_807_494
```

**Annotate:** short

> The batching is not an optimisation. `inorm(T_test)` as one tensor is
> 6,149 × 3 × 224 × 224 × 4 bytes = **3.45 GiB**, and that is how a Colab session
> dies. The uint8 tensor stays put and each batch is converted as it is needed.
> The parameter assert is the copy-check: a re-typed architecture with one channel
> count wrong still trains, still prints an accuracy, and invalidates every
> comparison in section 8. A number is a better copy-check than reading.

---

## Cell 4 — thread 8, part one: the parameter count

**Prompt to type:**

> For a 3×128×128 input and a 32×128×128 output, print the number of weights in
> a fully connected layer and in a 7×7 convolution with 32 filters that produces
> the same shaped output, and the ratio. Also print what the dense layer would
> weigh in float32, in gigabytes.

**Expect:**

| | |
|---|---|
| inputs | 49,152 |
| outputs | 524,288 |
| dense layer weights | 25,769,803,776 |
| conv layer weights | 4,704 |
| ratio | 5,478,275 |
| dense layer, float32 | **96 GB** |

**Assert:**

```python
assert conv_weights == 32 * 3 * 7 * 7
assert dense_weights // conv_weights > 5_000_000
```

**Annotate:** short

> `H` and `W` do not appear in the convolutional count — not approximately
> absent, genuinely absent. And the saving is **not** in arithmetic: the
> convolution still computes `C_out · H · W` outputs, each a sum over `C_in · k²`
> terms. What shrank is the number of distinct numbers that must be stored and
> learned. Note the gigabytes: 25.77 **billion parameters** is 96 GB of float32,
> not 25 GB. The current notebook's prose says 25 GB, which is the parameter count
> wearing the wrong unit — see defect 3.

---

## Cell 5 — thread 8, part two: equivariance, measured

**Prompt to type:**

> Check that a convolution commutes with a shift. Take one test image, shift it
> 16 pixels with `torch.roll`, and compare `conv(shifted)` with
> `shift(conv(image))` for a randomly initialised 7×7 conv with padding 3. Drop a
> 24-pixel border before comparing. Print the largest absolute difference, the
> largest activation in the same region, and the ratio.

**Expect:** on this machine, with seed 42 and the first test image:

```
largest |f(Tx) - Tf(x)| on the interior  0.000e+00
largest activation there                2.918
relative                                0.000e+00
```

**Exactly zero, not merely small.** On a CPU backend the two computations perform
the same additions in the same order, so even the floating-point caveat does not
bite. Do not write "that residue is float32 rounding" under a cell that prints
`0.000e+00` — the current notebook does, and it is defect 4.

**Assert:**

```python
assert (a - b).abs().max() / b.abs().max() < 1e-5, "not equivariant"
```

**Annotate:** full

* **Left open:** that this is checked on a **randomly initialised** convolution,
  and deliberately. The property belongs to the operation, not to the training —
  if you check it on a trained network you have proved something weaker and
  learned nothing about why weight sharing is what buys it. Untie the weights and
  the proof's middle step, a change of variable, fails.
* **The usual student version:** comparing the full tensors. Two documented
  defaults collide there and neither is about equivariance: `torch.roll` is
  **cyclic** — it wraps content round the edge — and `nn.Conv2d` has
  `padding_mode='zeros'`, which invents input that was not there. Measured at the
  corner of this same pair of tensors: **1.947e-01**, against 0 on the interior.
  A student who prints the full-tensor maximum reads 0.19, and concludes that
  convolutions are not equivariant.
* **How you would catch it:** report the **relative** figure, not the absolute
  one, and say which region you measured. Then measure the excluded region too
  (cell 6). An identity that holds on part of a domain is only honest if you have
  shown the other part is different.

---

## Cell 6 — and the border, where it fails

**Prompt to type:**

> Now print the same difference at the top-left 4×4 corner of the two tensors,
> to show that the region I excluded really is different.

**Expect:** `largest difference at the border 1.947e-01`. Against exactly 0 in
the interior, so the exclusion was necessary rather than convenient.

**Assert:** none — this cell exists to *fail* the interior identity, so an assert
here would be asserting the bug.

**Annotate:** short

---

## Cell 7 — thread 8, part three: invariance is not equivariance

**Prompt to type:**

> Using the same image and its shift, print the cosine similarity between the
> two feature maps flattened, and between the two globally max-pooled 32-vectors.
> Print 1 − cos as a percentage for both.

**Expect:**

```
cosine similarity, spatial maps   0.6218
cosine similarity, global pooled  1.0000

the map changed by    37.8%
the pooled vector by  0.00%
```

**Assert:**

```python
assert cos_pooled > cos_maps, "pooling did not buy invariance"
```

**Annotate:** short

> The two words are not synonyms. **Equivariant:** the output moves with the
> input. **Invariant:** the output does not change. Invariance is strictly
> stronger and strictly lossier — it is what you get by *discarding* the
> equivariant structure, and pooling is what discards it. Classification wants
> this: "this is a sunflower" is true wherever the sunflower is. Per-pixel
> prediction cannot afford it: segmentation asks, for every pixel, which class is
> it, and the answer **is** the position.

---

## Cell 8 — what pooling costs in resolution

**Prompt to type:**

> Walk `make_net()` with a dummy 1×3×128×128 input and print the size of the
> last max-pool output, how many input pixels one output cell answers for, and
> the factor of spatial resolution lost.

**Expect:** `last feature map 8 x 8`, `one cell answers for 256 input pixels`,
`spatial resolution lost: 256x`. Two flower boundaries fifteen pixels apart land
in the same cell.

**Assert:**

```python
assert last == IMG // 16
```

**Annotate:** short

> Walk the network rather than doing the arithmetic in your head: four
> `MaxPool2d` layers is easy to miscount, and the walk also tells you the shape
> the flatten will see. Say "one cell answers for 256 pixels", not "256× lost" —
> the first is a fact about flowers, the second is a fact about tensors.

---

## Cell 9 — thread 8, part four: where the memory goes

**Markdown above (ask before you run, §8):** *your network has 4,807,494
parameters and your session died with `out of memory`. Which of those two facts
caused the other? Commit to an answer before running the cell.*

**Prompt to type:**

> For a batch of 32, print how much memory this network's parameters take in
> float32, how much the gradients and Adam's state add, and how much the
> activations take — count every module output, because they all stay alive until
> the backward pass reaches them. Print the two ratios.

**Expect:**

| | |
|---|---|
| parameters | 4,807,494 → **18.3 MB** |
| + gradients + Adam state | **73.4 MB** |
| activations, per image | 5,964,646 values → 22.8 MB |
| activations, batch of 32 | **728.1 MB** |
| activations / parameters | **39.7×** |
| activations / everything parameter-shaped | **9.9×** |

**Assert:**

```python
assert act_mb > opt_mb, "the arithmetic says parameters dominate — check it"
```

**Annotate:** full

* **Left open:** the factor of four. The prompt has to name it or you will get
  parameters counted once. Weights, gradients, `exp_avg` and `exp_avg_sq` —
  Adam's two moment buffers are documented state, one float32 array each per
  parameter, allocated lazily on the **first** `opt.step()`, which is why cell 11
  takes a step before it measures anything.
* **The usual student version:** answering the question above with "the
  parameters caused the out-of-memory", and then reducing the parameter count.
  Activation memory is **linear in the batch size**; parameter memory does not
  move at all when you halve the batch, and the 728 MB above is 9.9× everything
  parameter-shaped put together.
* **How you would catch it:** when a run runs out of memory, halve the batch, not
  the model. In order: smaller batch, then smaller input resolution — quadratic,
  so it is the strongest single lever — then gradient checkpointing, then mixed
  precision. Reducing the parameter count is near the bottom of that list.

---

## Cell 10 — and it is not spread evenly

**Prompt to type:**

> Print the activation memory of every convolution in the network at batch 32,
> with its output shape, and what share of the total the first two hold.

**Expect:** seven rows, halving down the stack, and
`first two convolutions: 55% of the convolutional activation memory`.

| conv at index | output shape | MB |
|---|---|---|
| 0 | (32, 128, 128) | 64.0 |
| 3 | (32, 128, 128) | 64.0 |
| 7 | (64, 64, 64) | 32.0 |
| 10 | (64, 64, 64) | 32.0 |
| 14 | (128, 32, 32) | 16.0 |
| 17 | (128, 32, 32) | 16.0 |
| 21 | (256, 16, 16) | 8.0 |

**Assert:** none.

**Annotate:** short

> Activation cost is `C × H × W`. Channels double as the map quarters, so the
> total **halves** at every pooling stage, and the expensive layers are the ones
> nearest the image — which are the ones with almost no parameters. Parameters and
> activations are anti-correlated across a convolutional stack, so any intuition
> carried over from dense networks points the wrong way.

---

## Cell 11 — check the arithmetic against the allocator

**Prompt to type:**

> Check that prediction against the backend. Put the network and a random batch
> of 32 on the device, take one optimiser step first so Adam's state already
> exists, then synchronise, read the allocated memory, run one forward pass,
> synchronise again and print the difference beside the predicted number. If
> there is no accelerator, say so.

**Expect:** on MPS, `predicted 728.1 MB` against `measured 568.1 MB` — **78% of
the prediction**, because the counting is an upper bound: autograd frees the
buffers that no backward needs. Expect the same order, not the same number, and
**say in the notebook what agreement you expected** — the current notebook prints
both and reconciles nothing (defect 5). On CPU it prints `nan` and a line saying
there is no allocator counter.

**Assert:** none. There is no threshold here that is right on every backend.

**⏱** 2–3 s. This cell also defines `lossf`, which cells 12, 17 and 19 all use —
skip it and the next cell raises `NameError`.

**Annotate:** short

---

## Cell 12 — the from-scratch baseline, rebuilt here

**Prompt to type:**

> Train `make_net()` from scratch on the 128-pixel training split for 20 epochs,
> Adam at 3e-4, batch 32, seed 42, and time it. Then print train, validation and
> test accuracy and the gap between train and validation. Build the model and the
> optimiser inside this cell.

**Expect:** four lines — train, val, test, and the gap in points — plus the wall
clock in `SCRATCH_SECONDS`. The deck's 80-epoch run reaches **15.3%** test; 20
epochs reaches less, and the number you get is the one every comparison in this
notebook uses. Both anchors stay on the page: uniform guess **0.98%**, commonest
species **3.87%**.

**Assert:** none here. The assert that matters is in cell 20, where the
comparison is made.

**⏱ 20 epochs = 640 optimiser steps.** **17 s on MPS** (23 ms/step measured,
plus 1.5 s to score all 8,189 images) — the notebook's "about 30 seconds on a GPU
or MPS" is the right order. **On CPU: about 17.5 minutes** (46 ms per image
trained at 2 threads → 15.6 min, plus 1.9 min of scoring). That figure appears
nowhere in the current notebook, and it is the one a reader with no GPU needs.

**Annotate:** short

> Retrain it **in this notebook**. Quoting the previous lecture's number against
> a transfer run measured today compares two sessions, possibly two machines, and
> certainly two epoch counts. A persistent gap between two plateaus is
> **variance** (Lecture 6), and its two cures are more data and more constraint.
> We cannot buy labels, so the constraint has to come from somewhere — and the
> observation that points at where: the first layer learned colour blobs and
> oriented edges, and not one of them is about *flowers*. We spent 1,020 precious
> labels rediscovering something not specific to this problem.

---

## Cell 13 — a backbone somebody else trained

**Prompt to type:**

> Print the preprocessing that `ResNet18_Weights.DEFAULT` declares for itself,
> and how many parameters resnet18 has.

**Expect:**

```
ImageClassification(
    crop_size=[224]
    resize_size=[256]
    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]
    interpolation=InterpolationMode.BILINEAR
)

parameters: 11,689,512
```

**Assert:** none.

**Annotate:** short

> ImageNet: 1.28 million labelled photographs, 1,000 classes, none of them our
> species. Somebody already paid for edges, colours, textures and parts, and those
> weights are a download. **Read the whole repr, not just the two constant
> vectors.** It declares `resize_size=[256]` followed by a centre crop to 224;
> cell 2 squashed straight to 224×224 with `Resize((224, 224))`, which changes the
> aspect ratio and keeps the corners. That is a defensible choice for a
> fixed-in-memory tensor and it is **not** what the metadata says, so say so where
> you make it. The current notebook prints this block and then narrows
> "preprocessing" to the statistics without a word (defect 6).

---

## Cell 14 — ImageNet normalisation, kept separate

**Prompt to type:**

> Define the ImageNet mean and std as constants and a separate `inorm(x_uint8)`
> that uses them. Do not touch the existing `normalise`. Assert the output shape
> on two images.

**Expect:** `ImageNet normalisation ready`.

**Assert:**

```python
assert inorm(T_train[:2]).shape == (2, 3, TRANSFER_IMG, TRANSFER_IMG)
```

**Annotate:** short

> Two incompatible preprocessings now coexist. Give them different names and pass
> them explicitly — `norm=` is a parameter of `accuracy` for exactly this reason.
> Mutate `normalise` to take the ImageNet constants as defaults and every earlier
> cell in the notebook silently changes meaning. These constants are famous and
> they are not universal: they are ImageNet's, and using them on flowers is a
> choice with a reason, not a default.

---

## Cell 15 — freeze everything, then replace the head

**Prompt to type:**

> Load resnet18 with the pretrained weights, freeze every parameter, then
> replace `fc` with a fresh `Linear(512, 102)` and move it to the device. Print
> how many parameters are trainable and how many are frozen, and assert the
> trainable count.

**Expect:** `trainable 52,326   frozen 11,176,512`. 52,326 = 512 × 102 + 102, and
11,176,512 = 11,689,512 − the 513,000 of the discarded 1,000-class head.

**Assert:**

```python
assert n_train_p == 512 * N_CLASSES + N_CLASSES
```

**Annotate:** full

* **Left open:** the order. The prompt says "freeze, **then** replace" because
  this script says so — a prompt that just says "freeze the backbone and put a new
  head on" leaves the assistant free to write it either way round.
* **The usual student version:** replacing `fc` first and freezing afterwards.
  A newly constructed `nn.Linear` has `requires_grad=True` on its parameters —
  that is the documented default for every `nn.Parameter` — so freezing after
  replacing freezes the head too. The result is a model with **zero** trainable
  parameters that trains happily, raises nothing, prints a falling-then-flat loss
  because dropout and batch norm still move, and changes not one weight.
* **How you would catch it:** assert the trainable count against an arithmetic
  expression you can do on paper, `512 · 102 + 102`. It is the only thing that
  distinguishes the two orders, and it takes one second. `print(model)` does not
  distinguish them: the architecture is identical either way.

---

## Cell 16 — if nothing before the head trains, run it once

**Prompt to type:**

> The backbone is frozen, so run it once instead of once per epoch: put the body
> — everything except `fc` — in eval mode, and under `no_grad` push all three
> splits at 224 through it to get 512-dimensional features per image. Time it and
> assert the training features are (1020, 512).

**Expect:** `features (1020, 512)` and a wall clock in `FEATURE_SECONDS`.
`F_train` is 2.0 MiB, `F_test` 12.0 MiB — the 1,176 MiB of images has become
14 MiB of table.

**Assert:**

```python
assert F_train.shape == (1020, 512), F_train.shape
```

**⏱** **about 8 s on MPS** (3.3 s of forward passes for all 8,189 images, the
rest normalising uint8 → float32 and moving 4.9 GB to the device in batches).
**On CPU: about 3.5 minutes** (26 ms per image at 2 threads). The current
notebook says "about 25 seconds" with no device named and no CPU figure.

**Annotate:** short

> `.eval()` on the body and `no_grad()` on the extraction, both. The backbone has
> twenty batch-norm layers whose running statistics are updated in the **forward**
> pass; an extraction pass in `train()` mode would rewrite them with flower
> statistics before the probe has been fitted at all. A frozen prefix is a fixed
> function: computing it once is not an optimisation, it is what "frozen" means.
> Note that `nn.Sequential(*list(net.children())[:-1])` works here only because
> resnet18's forward is a plain chain apart from the flatten — it is not a general
> way to decapitate a network.

---

## Cell 17 — the head, on cached features

**Prompt to type:**

> Train a `Linear(512, 102)` on the cached training features for 60 epochs, Adam
> 1e-3 with weight decay 1e-4, batch 32, seed 42. Print the test accuracy of
> backbone-plus-head and its **total** wall clock including the feature
> extraction, next to the from-scratch number and time.

**Expect:** two lines, the probe and the from-scratch run, accuracy and seconds
each. The deck's figure for this row is **81.2% in 13 s** against 15.3% in 364 s.

**Assert:** none — cell 20 is where the comparison is asserted.

**⏱** the head itself is **0.7 s on MPS, 0.4 s on CPU** — 60 epochs of a linear
model on a 1,020 × 512 table. It is `FEATURE_SECONDS` that costs, which is
exactly why it must be in the total.

**Annotate:** short

> The arithmetic that explains why it works: **52,326** trainable parameters
> fitted with 1,020 examples is about **51 parameters per image**, against
> **4,713** for the from-scratch network. The **11,176,512** frozen ones were
> fitted with 1.28 million examples, by somebody else. The variance problem was
> not solved — it was *moved* to a dataset large enough to absorb it. Report the
> probe's time with the extraction included; it is not free just because the head
> is.

---

## Cell 18 — augmentation, as arithmetic

**Prompt to type:**

> Write `augment(x_uint8, gen)` directly on tensors — random resized crop with
> scale between 0.55 and 1.0, then a horizontal flip half the time, using a
> passed-in `torch.Generator`. Show one training image and seven augmented views
> of it side by side.

**Expect:** eight panels, the original first. The label is the same in all eight
— that is the only thing that makes this legitimate.

**Assert:** none; this is a figure. Look at it before you train on it.

**⏱** 3 s. Note for later: `augment` is a Python loop, **2.6 ms per image at
224** measured, and it runs on the CPU whatever device the model is on.

**Annotate:** short

> Written on tensors rather than as a transform pipeline so the operation is
> visible as arithmetic rather than as a class name. An augmentation that could
> change the label is not an augmentation, it is a mislabelling: vertical flips
> and large rotations are fine for flowers and wrong for digits and for street
> signs. A crop scale that occasionally excludes the subject is a labelled
> photograph of a leaf — which is why 0.55 is written down and not defaulted.

---

## Cell 19 — fine-tune the last block

**Prompt to type:**

> Now fine-tune: load resnet18 pretrained again with a fresh 102-class head,
> freeze everything, then unfreeze `layer4` and `fc`. One Adam with two parameter
> groups — 1e-4 for layer4, 1e-3 for the head. Train 8 epochs at 224 with the
> augmentation, batch 32. Keep the frozen batch-norm layers in eval mode. Record
> the validation accuracy each epoch, both on the clean validation set and on an
> augmented copy of it, using a separate generator for the diagnostic. Print the
> test accuracy and the wall clock.

**Expect:** eight `epoch N  val 0.xxx  (Ns)` lines, then the test accuracy. The
deck's 15-epoch version of this row is **88.0% in 176 s**. Trainable here is
**8,446,054** parameters — `layer4` alone is **8,393,728**, three quarters of
resnet18 — so the fine-tune is fitting **8,280 parameters per training image**,
*more* than the from-scratch network's 4,713, and it still wins. The starting
point, not the parameter count, is what makes that safe.

**Assert:** none inside the loop. Cell 20 asserts the outcome.

**⏱ 8 epochs = 256 optimiser steps.** **About 60 s on MPS** — and read where it
goes: 0.7 s per epoch of gradient work against **6.2 s per epoch of CPU**, of
which 2.7 s is augmenting the training split, 2.7 s is augmenting the validation
split for the diagnostic, and 0.8 s is the two validation passes. On an
accelerator this cell is dominated by the augmentation loop, not by the network.
**On CPU: about 18 minutes** (58 s of training per epoch, 57 s of diagnostics per
epoch, 2.7 min for the final test pass over 6,149 images at 224).

**Annotate:** full

* **Left open:** the second generator, and why. Share one and the augmented
  validation diagnostic consumes draws from the same stream that shuffles the
  training batches — so the *training order* depends on whether the diagnostic ran
  at all. Harmless with a fixed seed and one configuration; it is also exactly the
  coupling this course asks you to find in other people's code.
* **The usual student version:** `for p in ft.parameters(): p.requires_grad =
  False` and then training, expecting the backbone to be frozen. It is not.
  `requires_grad` governs the **backward** pass; batch norm's `running_mean` and
  `running_var` are **buffers**, not parameters, and they are updated in the
  **forward** pass whenever the module is in `train()` mode. Measured directly:
  every parameter set to `requires_grad=False`, one forward pass of 32 images
  inside `torch.no_grad()`, module in `train()` mode — `bn1.running_mean` moved by
  **0.0131**, 10.3% of its largest entry, and `num_batches_tracked` went to 1.
  With the default `momentum=0.1`, after this cell's 256 batches only
  0.9²⁵⁶ ≈ **2 × 10⁻¹²** of the downloaded statistics survives. The backbone you
  are reporting is no longer the one you downloaded, and nothing raised.
* **How you would catch it:** put `bn1`, `layer1`, `layer2`, `layer3` — 15 of
  resnet18's 20 `BatchNorm2d` modules, holding 4,480 running statistics — into
  `.eval()` *inside* the epoch loop, after `ft.train()`, because `train()` resets
  every one of them. Then check it: save `bn1.running_mean.clone()` before the
  loop and assert it is unchanged after. And use two learning rates: 1e-4 on the
  block that is already good, 1e-3 on the head that starts from noise. One rate
  for both damages whichever it is wrong for.

---

## Cell 20 — the comparison this application exists to make

**Prompt to type:**

> Print one table: uniform guess, commonest species, the from-scratch net, the
> frozen probe and the fine-tuned model — accuracy and wall clock in the same
> columns. Then the gain in points over from-scratch, and how many times faster
> the frozen probe was. Assert the fine-tune beat the from-scratch run, and put
> the likely cause in the message if it did not.

**Expect:** five rows. The two anchors stay in the table: **0.98%** uniform,
**3.87%** commonest species. All five accuracies are on the **same 6,149 test
images** — that is what makes the column a comparison.

**Assert:**

```python
assert FT_TEST > SCRATCH_TEST, "transfer learning did not help — check inorm"
```

**Annotate:** short

> **Say what the wall-clock column is measuring, because the three entries are
> not measuring the same thing.** `SCRATCH_SECONDS` is the training loop only, and
> excludes its evaluation. `PROBE_SECONDS` is feature extraction over all three
> splits plus the head. `FT_SECONDS` includes the per-epoch validation passes and
> the augmentation of the validation split — measured above, that is roughly
> **half** of it on either device. A fine-tune timed without its teaching
> diagnostic is nearer 30 s on MPS than 60. If you want the "2.1× faster" claim to
> be a measurement, time the same phases in both runs and say which phases.

---

## Cell 21 — an assistant writes the augmentation

**Markdown above — this exact restraint is the point (§8.1):**

> ## 9 · An assistant writes the augmentation
>
> The fine-tune above wrote its augmentation as tensor arithmetic. Nobody does
> that. Here is the version you get by asking for it, in the ordinary way, and
> the four numbers it produces. **Write them down before you read on.**

No ⚠ here. No hint. No annotation. Do not name the defect.

**Prompt to type:**

> Add data augmentation to the flower classifier: random resized crops and
> horizontal flips, with the ImageNet normalisation. Build the training and
> validation dataloaders. Then score the fine-tuned model `ft` through the
> validation loader ten times and print the mean, the min, the max and the spread
> in accuracy points, and the accuracy on the clean 224 tensors for comparison.

**Expect:** one `transforms.Compose`, two `Flowers102(...)` datasets, two
`DataLoader`s, a scoring loop, and four numbers. The clean number is a fixed
number. **The ten are not**, and the spread is the thing to write down — the deck
measures **1.57 points** between the smallest and the largest for the same fixed
weights on the same 1,020 images. Watch also for
`RandomResizedCrop(224)` written without `scale=`: the documented default is
`scale=(0.08, 1.0)`, a crop of as little as **8% of the image area**, which
widens the spread further.

**Assert:**

```python
assert accuracy(ft, T_val, y_val, inorm) == clean, \
    "even the clean evaluation is not deterministic"
```

That one passes. It is here so that the wobble you are about to see is
attributable to the data and not to a model left in `train()` mode.

**⏱** ten passes over the validation split through a `DataLoader` that re-decodes
and re-crops each JPEG: **about 70 s on MPS**, **about 6 minutes on CPU**
(5–7 ms per image of decode-and-augment measured, plus 26 ms per image of forward
on CPU against 0.4 ms on MPS). The current notebook has no ⏱ on this cell at all.

**Annotate:** short — and keep the short box genuinely short. Specification only:
*input · the fine-tuned model and the assistant's two loaders; output · one fixed
set of weights scored ten times; constraint · score the same weights repeatedly,
the wobble is the whole diagnostic.*

---

## Cell 22 — ⚠ what the default cost you

**Markdown above — now the reveal:**

> ### ⚠ Reviewer question 5: what is the default I did not ask for?
>
> One `tf`, used for both splits:
>
> ```python
> train_ds = Flowers102("datasets", split="train", transform=tf)
> val_ds   = Flowers102("datasets", split="val",   transform=tf)
> ```
>
> **Nothing is leaking.** The model never trains on the validation set. The
> damage is entirely in what the validation number now *means*. Compare the four
> numbers you wrote down: the same weights, the same 1,020 images, several points
> apart. Lecture 12's reviewer question arriving again — **a deterministic
> function of fixed weights and fixed data does not wobble.**

**Prompt to type:**

> Plot the two validation curves from the fine-tuning run — clean and augmented
> — against epoch, and print which epoch early stopping would keep under each,
> and how many points pessimistic the augmented number is at the last epoch.

**Expect:** two curves, the augmented one below the clean one throughout, and two
epoch numbers. **Read this before you read the output:** with only 8 epochs both
curves may still be rising, in which case both rules keep epoch 8 and the
printed comparison shows nothing. That is not a refutation — the deck's 15-epoch
run peaks at epoch **14** clean and epoch **12** augmented, and both of those lie
beyond this notebook's window. The argument that does not depend on the epoch
count is the one to make: the augmented curve's own reproducibility, measured in
cell 21, is **wider than the gaps between its last three epochs**, so whichever
epoch it selects, it selected it by noise. State that next to the plot, and
report the spread from cell 21 beside the two epoch numbers.

**Assert:**

```python
assert len(clean_curve) == len(aug_curve) == FT_EPOCHS
```

Matched rows, matched epochs: both curves are the same 1,020 validation images,
scored after the same optimiser steps, differing only in the transform.

**Annotate:** full

* **Left open:** that a noisy metric is the smaller half of the problem. The
  larger half is that it is also **pessimistic** — the augmented number
  understates the model you shipped, and a number that is too low feels
  conservative and safe, so nobody investigates it. The deck measures 0.86 points
  low on the 15-epoch run.
* **The usual student version:** one `tf` for both splits — one word shorter to
  write and exactly what the prompt's own phrasing ("build the training and
  validation dataloaders") invites. It is not a hypothetical: it is what comes
  back from the prompt in cell 21, it runs, it trains, and the validation accuracy
  climbs over epochs exactly as it should. The second version is treating this as
  cosmetic, a wobbly plot; it changes which checkpoint you keep.
* **How you would catch it:** the assertion in the corrected specification —
  cell 23 — and nothing else on this page. Not by reading the transform, not by
  looking at the curve. **Evaluating the same model on the same data twice must
  give the same number.**

---

## Cell 23 — the one-line check that generalises

**Markdown above — the corrected specification, quoted as a prompt:**

> *"Build **two** transform pipelines: a training one with random resized crop
> and horizontal flip, and a deterministic evaluation one with resize and centre
> crop. Use the second for validation and test. Assert that evaluating the same
> model on the validation set twice gives identical numbers."*

**Prompt to type:**

> Evaluate `ft` on the validation set twice and assert the two numbers are
> **exactly** equal. Print the number once.

**Expect:** `evaluation is deterministic: 0.xxxx twice`, and the assert passes.

**Assert:**

```python
a = accuracy(ft, T_val, y_val, inorm)
b = accuracy(ft, T_val, y_val, inorm)
assert a == b, f"evaluation is not deterministic: {a} vs {b}"
```

**⏱** two passes over 1,020 images at 224: **1 s on MPS**, **55 s on CPU**.

**Annotate:** full

* **Left open:** how many distinct bugs this single line catches. Four, all
  silent: augmentation on the evaluation split; a missing `model.eval()`, which
  leaves dropout sampling and batch norm using batch statistics; a `DataLoader`
  built with `shuffle=True` on the validation split combined with any per-batch
  state; and label misalignment introduced by a re-shuffle between features and
  targets. None of the four raises. All four fail this.
* **The usual student version:** `assert abs(a - b) < 1e-3`. Tolerances are a
  habit carried over from comparing floating-point *computations*, and they are
  wrong here: `accuracy` returns `right / len(X)`, a ratio of two integers, so two
  evaluations of a fixed function on fixed data are **bit-identical**. A 1e-3
  tolerance passes under all four bugs above — the augmented spread in cell 21 is
  a hundred times larger than that and would still slip through a looser one.
* **How you would catch it:** put this line at the bottom of every notebook that
  evaluates a model. It costs a second of wall clock and it is the highest-yield
  assertion in the course.

---

## Closing markdown — red-team, corrected

Swap notebooks with the team beside you. Ten minutes, five questions:

1. What touched the test set?
2. What was fitted, and on what? (`fit` and `transform` are different verbs)
3. What is the shape here?
4. What was dropped — rows, columns, images? Count them.
5. What is the default I did not ask for?

**The answers for *this* notebook** — three of the five differ from the table in
the current notebook, which copies the deck's catalogue of failure modes rather
than describing its own code:

| | |
|---|---|
| 1 | **Nothing was fitted on it.** `MEAN`/`STD` come from the 1,020 training images only — check it, the line is `xf = X_train.float() / 255.0`. Cells 5–7 do *read* one test image, `X_test[:1]`, for the equivariance demonstration; say so, and say that nothing was fitted on it. The failure the deck catalogues here — statistics over all 8,189 images — is the one this notebook **avoided**, and it is worth naming as avoided rather than as done. |
| 2 | the backbone was fitted on ImageNet, not on your data — say so when you report |
| 3 | 224, not 128: the pretrained weights fix the input size |
| 4 | nothing is dropped here — count anyway, and say zero |
| 5 | **one transform for two splits** (cell 21); `RandomResizedCrop`'s `scale=(0.08, 1.0)`; and `train()` mode on frozen batch-norms, which cell 19 handles explicitly. `padding=0`, which the current notebook lists here, is `nn.Conv2d`'s default but appears nowhere in this notebook — every convolution in it names its padding. |

Report what you **found**, not what you would have done differently.

**Mark every section (§8.3).** Sections 2–5 (thread 8) are **examinable**.
Sections 6–8 (the repair) are **examinable**. Section 1 and cell 11 are **not
examinable — engineering**. Section 9 is **examinable**: the assertion in cell 23
is the examinable object, not the anecdote. The current notebook contains the
word "examinable" **zero** times.

### One line to keep

Before you train a vision model from scratch, spend the seconds it takes to
measure what a frozen pretrained backbone and a linear head already give you. You
are entitled to train from scratch. You are not entitled to do it without knowing
what you turned down.

---

## Exercises, with the re-run order (§7.2)

Cell numbers are this script's, 1–23.

1. **Change the seed.** Edit `RANDOM_STATE` in **cell 1**, then re-run
   **1 → 5 → 6 → 7 → 12 → 15 → 16 → 17 → 19 → 20**. Cell 2 does not depend on the
   seed and costs 44 s of decode, so skip it. Cells 12, 17 and 19 each construct
   their model and optimiser inside themselves, so this genuinely restarts
   training rather than continuing it — check that before you trust any re-run of
   a training cell. ⏱ **~2 min MPS, ~40 min CPU**.
2. **The ablation the table is missing.** The comparison in cell 20 attributes
   one gain to two changes at once. Copy **cell 19**, set `FT_EPOCHS = 8` still
   but pass the *un*-augmented batch (`inorm(T_train[idx])` instead of
   `inorm(augment(T_train[idx], gen))`), and re-run **the copy → 20**. You now
   have three transfer rows — frozen probe, layer4 unfrozen, layer4 plus
   augmentation. The deck's 15-epoch numbers are 81.2 / 86.3 / 88.0, i.e. almost
   all of the gain is in the first row. Say which row your gain is in.
   ⏱ **~50 s MPS, ~15 min CPU** (no augmentation, so the epochs are shorter).
3. **Break the freeze on purpose.** In **cell 19**, delete the loop that puts
   `ft.bn1, ft.layer1, ft.layer2, ft.layer3` into `.eval()`, and add
   `before = ft.bn1.running_mean.clone()` before the loop and a print of
   `(ft.bn1.running_mean - before).abs().max()` after it. Re-run **19 → 20**.
   Nothing raises; the statistics moved. Then put the `.eval()` loop back and
   re-run **19 → 20** again, and compare the two test accuracies with the fact
   that neither run reported an error. ⏱ **~2 min MPS, ~36 min CPU for both runs**
   — on a CPU runtime do this one with `FT_EPOCHS = 2` in both halves, and say so
   next to any number you write down.
4. **Feed the pretrained network 128-pixel images.** In **cell 16**, change all
   three calls — `features(T_train)`, `features(T_val)`, `features(T_test)` — to
   the `X_` tensors, and re-run **16 alone**. It does not raise: ResNet is fully
   convolutional up to the pool, so the features come back `(1020, 512)` as
   before. Re-run **17** and read what happened to the accuracy. Then change all
   three back and re-run **16 → 17**. ⏱ **~20 s MPS, ~4 min CPU** each
   way (the 128 pass is roughly a third of the 224 pass).
5. **Widen the crop to torchvision's default.** In **cell 18**, change the scale
   floor from 0.55 to 0.08 — the documented default of `RandomResizedCrop` — and
   re-run **18 alone**. Look at the eight panels and count how many still contain
   the flower. Then re-run **21** with the assistant's pipeline left at its
   default and compare the spread with the one you wrote down. ⏱ **3 s** for cell
   18, **70 s MPS / 6 min CPU** for cell 21.
6. **Make the tolerance fail.** In **cell 23** replace the exact equality with
   `assert abs(a - b) < 1e-3`, then evaluate one of the two calls on
   `augment(T_val, gen)` instead of `T_val`, and re-run **23 alone**. It passes.
   Put the equality back. ⏱ **5 s MPS, 90 s CPU**.

## If you have no GPU

Measured component by component, this notebook is **about 50 minutes on a CPU
runtime** — cell 19: 18 min, cell 12: 17.5 min, cell 21: 6 min, cell 16: 3.5 min,
cell 2: 44 s of decode (2–4 min on Colab), cell 23: 55 s — against **about
3 minutes** with an accelerator. The current notebook gives no runtime advice at
all and states no CPU figure anywhere. If Colab will not give you a T4:

- **Cell 12:** drop `EPOCHS_SCRATCH` from 20 to 5. About 4.5 minutes. The
  from-scratch number will be lower, and you must **say so** next to it — a
  5-epoch score and a 20-epoch score are not the same measurement. The
  comparison in cell 20 still stands, in the direction that matters, and it
  understates the transfer gain rather than flattering it.
- **Cell 19:** drop `FT_EPOCHS` from 8 to 3. About 6 minutes. Cell 22's two
  curves become three points each, so make the noise-band argument from cell 21's
  spread and do not report an early-stopping epoch off three points.
- **Cell 21:** drop the ten scorings to five. About 3 minutes. Report `min` and
  `max` over five and say it is five — a spread from five draws is not the same
  statistic as a spread from ten.
- **Cell 16 and cell 17 are the two you must not cut.** They are the frozen
  probe, they are 3.5 minutes together, and they are the result the lecture
  exists to produce.
- **You cannot skip cell 12 to get to the transfer part.** Cell 12 defines
  `ytr`, which cells 17 and 19 both use, and cell 11 defines `lossf`, which cells
  12, 17 and 19 all use. Skipping either raises `NameError` a long way downstream.

---

## Defects found in the current notebook

`notebooks/lecture-16.ipynb`, 74 cells, 23 code cells. Every claim below was
checked with `python3` against the notebook JSON, `slides/lecture-16.html`,
`LECTURES.md`, the dataset under `notebooks/datasets/flowers-102/`, or by direct
execution, **except** the four marked *not checked* at the end.

### Verified by execution or arithmetic

1. **§6.1 — the annotation budget, the primary defect.** Counted from the JSON:
   **23** markdown cells beginning `> **Prompt`, **23** containing `Watch this
   prompt`, and **23** carrying three or more `* **` bullets. Every one of the
   twenty-three code cells has the full three-bullet annotation. The rule is five
   to eight, never more than ten. All three student readers stopped reading the
   template around cell 30; this notebook's payload — the assistant failure — is
   its cell 21 of 23, deep inside the fatigue zone. This script cuts it to seven.

2. **§8.1 — the defect is announced six times before it runs.** Counted in the
   markdown preceding code cell 21: the header (*"Cells marked ⚠ read before
   running contain a defect on purpose"*); the section-9 heading block, which
   carries *"⚠ Read before running"*; the quoted assistant code, in which the
   offending line is marked `# <--`; the sentence *"Perfectly reasonable, and it
   names both loaders. **That is the hole.**"*; the heading *"Reviewer question 5:
   what is the default I did not ask for?"*; and the paragraph *"One `tf` for both
   splits"* — which states the answer in full, immediately above the cell. The
   prompt box then repeats it in its label and again in its `student` bullet.
   By the time the reader's eye reaches the code there is nothing left to find.
   This script's cells 21 and 22 reorder it: run, write four numbers down, then
   the ⚠.

3. **§1.2 — "25 GB of float32 for one layer" is 96 GB, and the cell prints 96.**
   Cell 11's `catch` bullet (notebook markdown cell 10). The dense layer is
   3·128·128 × 32·128·128 = **25,769,803,776 weights**; at 4 bytes each that is
   103,079,215,104 bytes = **96.0 GiB**. The code cell's own line
   `print(f"dense layer, float32  {dense_weights * 4 / 2**30:>18,.0f} GB")` prints
   **96**. The prose has taken the parameter count, 25.77 billion, and put a
   gigabyte sign on it. The argument survives — 96 GB ends it more firmly — but
   prose and output disagree on the same screen.

4. **§1.1 — "that residue is float32 rounding, not mathematics" describes a
   residue that does not exist.** Executed the equivariance cell exactly as
   written (seed 42, `X_test[:1]`, training-split statistics): the interior
   maximum difference is **0.000e+00** and the relative figure is **0.000e+00**.
   The cell prints that string unconditionally, immediately under a line reading
   `0.000e+00`. The deck is right where the notebook is wrong —
   `slides/lecture-16.html` says *"Bit-for-bit identical, not merely close. On
   this backend the two computations perform the same additions in the same
   order, so even the floating-point caveat does not bite"*. The border figure in
   the same run is **1.947e-01**, so cell 6's "orders of magnitude larger" is
   true, and infinitely so.

5. **§1.5 — two numbers for the same quantity, printed together, never
   reconciled.** Cell 11 (the allocator check) prints `predicted, by counting
   outputs 728.1 MB` beside `measured, by the backend`. Executed on MPS exactly as
   written, including the warm-up optimiser step: **568.1 MB**, 78% of the
   prediction. The gap is real and explicable — the count is an upper bound,
   autograd frees what no backward needs — and the notebook says nothing about it,
   so a reader has no way to tell whether 568 confirms the arithmetic or refutes
   it. §6.3 asks the box to state the expected answer; this box asks for a
   comparison and names no expectation.

6. **§1.2 / §3.2 — the declared preprocessing is printed and then two thirds of
   it is ignored.** Cell 13 prints `weights.transforms()`, which reads
   `resize_size=[256]`, `crop_size=[224]`, then the mean and std — verified by
   running it. The prose under it says *"The weights come with their own
   preprocessing. Use those statistics"*, silently narrowing "preprocessing" to
   the two constant vectors. Cell 2 has already decoded with
   `transforms.Resize((224, 224))`, a direct squash that changes the aspect ratio
   and keeps the corners, which is neither the declared resize nor the declared
   crop. The choice is defensible for fixed in-memory tensors; it is made without
   being named, in the cell whose whole lesson is *read the metadata*.

7. **§3.1 — the quoted assistant code exists in no cell of the notebook.**
   Markdown cell 64 contains a ```` ```python ```` block whose lines
   `transforms.RandomResizedCrop(224, scale=(0.55, 1.0))`,
   `train_ds = Flowers102("datasets", split="train", transform=tf)` and
   `val_ds = Flowers102("datasets", split="val", transform=tf)` appear in **no
   code cell** — string-searched every one. It is the only python fence in the
   notebook, and it is the defect the whole section is about. What actually runs
   is a *simulation* of it: cell 21 re-uses the notebook's own tensor `augment()`
   on the decoded `T_val`. So the reader is told the assistant produced this, is
   shown a consequence produced by something else, and cannot run the quoted code
   to see it fail. I executed the quoted pipeline directly: it works —
   `DataLoader(val_ds, batch_size=32)` yields `(32, 3, 224, 224)` float32 —
   at **5–7 ms per image**, so making it the cell that runs costs about 70 s on
   MPS for ten passes. This script does that instead.

8. **§2.1 — the wall-clock column compares three different phases.** Read from
   the code: `SCRATCH_SECONDS` brackets the training loop and excludes evaluation;
   `PROBE_SECONDS` is `FEATURE_SECONDS + HEAD_SECONDS` and so includes a forward
   pass over all 8,189 images; `FT_SECONDS` brackets a loop that contains, every
   epoch, one clean validation pass, one augmentation of the whole 1,020-image
   validation split and a second validation pass. Measured per-item costs put
   that diagnostic at **3.5 s of the 6.9 s per epoch on MPS** (2.7 s of it is
   augmenting the validation split, 0.8 s the two forward passes) and **57 s of
   the 117 s per epoch on CPU** — roughly **half** of the fine-tune's headline
   number is a teaching diagnostic that the from-scratch number excludes. Cell 20 then
   prints `frozen probe was Nx faster than training from scratch` from two of
   these three quantities. The rows are matched (all five accuracies are the same
   6,149 test images); the *times* are not, and the prose does not say so.

9. **§7.1 — four ⏱ markers, not one CPU figure among them.** The notebook's
   timing lines are *"about 40 seconds"* (decode), *"about 30 seconds on a GPU or
   MPS"* (cell 12), *"about 25 seconds"* (cell 16) and *"about 90 seconds on a GPU
   or MPS"* (cell 19). Measured at 2 threads on an M4 Max, those same cells are
   **17.5 minutes**, **3.5 minutes** and **18 minutes** on CPU. The accelerator
   figures are sound — 23 ms per training step at 128 and per fine-tune step at
   224, 26 ms per 64-image forward at 224, measured on MPS — so the defect is
   entirely the missing CPU column, which is the one the reader alone at home
   needs. The decode's *"about 40 seconds"* is right on fast hardware (**44 s**
   measured, 17 s at 128 plus 27 s at 224) and was **3 min 28 s** on this machine
   under load.

10. **§7.1 — two cells over twenty seconds carry no ⏱ at all.** Code cell 21
   evaluates the 1,020-image validation split **twelve** times (one clean, ten
   augmented, one in the assert) and augments it ten times: **6 minutes on CPU**,
   and the augmentation loop alone is 2.6 ms per image measured. Code cell 23
   evaluates it twice: **55 s on CPU**. Neither has a marker, and cell 21 is in
   the section a reader is most likely to run on its own.

11. **§7 — the notebook gives no runtime advice, and the expensive cell cannot be
   skipped.** The header says nothing about GPU or CPU. End to end the notebook
   is **about 50 minutes on CPU** against about 3 with an accelerator. Worse, a
   reader who wants to skip the from-scratch baseline to reach transfer learning
   cannot: `ytr` is defined only in cell 12 (`Xtr, ytr = normalise(X_train)
   .to(device), y_train.to(device)`) and used in cells 17 and 19, and `lossf` is
   defined only in cell 11 — the accelerator memory check — and used in cells 12,
   17 and 19. Traced by grepping every code cell. `lossf` is used one, six and
   eight cells after it is defined; `ytr` five and seven cells after. Skipping
   either cell raises `NameError` inside a cell that has no visible connection
   to it.

12. **§4.1 — `a` and `b` are tensors in cell 5 and floats in cell 23.** In cell 5
   they are the interior slices of two feature maps, `(1, 32, 80, 80)`; in cell
   23 they are two accuracies. The notebook's own guideline evidence — lecture 19
   rebinding `test_mae` from a float to a loop variable — is this exact shape.
   Throwaway comparisons want throwaway names. Also `rows` is a list of
   `(index, shape, MB)` in cell 10 and a list of `(name, accuracy, seconds)` in
   cell 20, and `gen` is an **MPS** generator in cell 12 and a **CPU** generator
   in cell 19 — the device change is deliberate and correct, since `augment`
   needs a CPU generator, and it is invisible at the call site.

13. **The red-team table describes the deck's catalogue, not this notebook.**
   Answer 1 is *"normalisation statistics computed over all 8,189 images"*. This
   notebook computes them from `X_train` alone — `xf = X_train.float() / 255.0`,
   the only mean/std computation in any cell, verified by executing it: `MEAN =
   [0.4330, 0.3819, 0.2964]`, which is the training-split value and differs from
   the all-splits value by 0.0085 in the blue channel. So the honest answer to
   *"what touched the test set?"* is "nothing was fitted on it, and cells 5–7 read
   one test image for the equivariance demo". Answer 5 lists `padding=0`, which
   appears in no cell — every `nn.Conv2d` in the notebook names its padding
   (`padding=k // 2`, `padding=3`), grepped. Two of five answers do not describe
   the notebook they are printed in.

14. **§1.2 — no cell in the notebook has a stored output.** All 23 code cells
   have `outputs: []` and `execution_count: None`, so every prose figure fails
   the mechanical form of §1.2. I re-derived the checkable ones and they are
   **correct**: parameter count **4,807,494**; ratio **5,478,275**; majority
   baseline 238/6,149 = **3.87%**; uniform 1/102 = **0.98%**; splits
   1,020 / 1,020 / 6,149 = **8,189**; head **52,326** = 512·102 + 102; frozen
   **11,176,512** = 11,689,512 − 513,000; **51** parameters per image
   (52,326/1,020 = 51.3) against **4,713** (4,807,494/1,020 = 4,713.2); last
   feature map **8×8**, one cell per **256** input pixels; parameters **18.3 MB**
   and activations at batch 32 **728.1 MB**; `inorm(T_test)` = **3.45 GiB**,
   written as 3.5 GB. The figures that do **not** reconcile are items 3, 4 and 5.

15. **§8.3 — the word "examinable" appears zero times.** Grepped every markdown
   cell. Lecture 19 used it once and that was cited as a defect; this notebook
   uses it not at all, across ten sections that mix a mathematical thread, an
   engineering measurement and an assistant failure.

16. **§3.3 — "the next cell" is two cells away.** Markdown cell 26 ends *"Commit
   to an answer before running the next cell"*; the next cell is the prompt box
   (markdown cell 27) and the code is cell 28. Because every code cell in this
   notebook is preceded by a prompt box, the phrase is off by one everywhere it
   could be used, which is the literal reader's complaint about lecture 19
   verbatim. The lecture cross-references resolve: Lecture 6 is *Reading a
   learning curve* (variance) ✓; Lecture 12 does contain both the five reviewer
   questions with #5 as *"what is the default I did not ask for?"* and the
   sentence *"A deterministic function of fixed weights and fixed data does not
   change between calls"* ✓ — checked in `notebooks/lecture-12.ipynb`. **Lecture
   18 is the overstatement**: markdown cells 23 and 25 say *"Lecture 18 has to put
   that resolution back, and the whole of its architecture is about how"*, but
   `LECTURES.md` gives Lecture 18 as *Scoring a box, scoring a detector*, whose
   thread is IoU and mAP; per-pixel prediction appears only in its final clause.

### Checked and found clean

- **§5.1 / §5.2** — scanned every markdown cell line by line: no prose line
  indented ≥ 4 spaces outside a fence, no indented fence marker, no unclosed
  fence. The one python fence opens and closes at column 0.
- **§4.2** — all three training cells re-instantiate. Cell 12 calls `make_net()`
  and a fresh `Adam`; cell 17 constructs `head` and its optimiser; cell 19
  constructs `ft` from `resnet18(weights=weights)` with a new head and a new
  optimiser. Re-running any of them genuinely restarts rather than continuing,
  which is the failure lecture 19 shipped.
- **§2.1 for the accuracies** — the five rows of cell 20 are scored on the same
  6,149 test images. The from-scratch net sees them at 128 and the transfer models
  at 224, which is the point of the lecture and not a mismatch of rows. The two
  validation curves in cell 22 are the same 1,020 images at the same epochs.
- **The freeze/replace order in cell 15 is correct**, and its assert is the only
  thing that distinguishes it from the wrong order.
- **The batch-norm claim in cell 19 is true and I reproduced it.** Every
  parameter set to `requires_grad = False`, one forward pass of 32 images in
  `train()` mode inside `no_grad()`: `bn1.running_mean` moved by **0.0131**
  (10.3% of its largest entry) and `num_batches_tracked` went to 1. The notebook's
  repair — putting `bn1`, `layer1`, `layer2`, `layer3` into `.eval()` inside the
  epoch loop, after `ft.train()` — covers 15 of resnet18's 20 `BatchNorm2d`
  modules, correctly leaving `layer4`'s five in training mode.
- **`torch.Generator(device=...)` with `randperm` works on MPS** — tested on both
  `cpu` and `mps`, both return a permutation. So cell 12's device-bound generator
  is not an Apple-Silicon blocker.
- **The cell-21 assert is sound.** `accuracy` returns `right / len(X_u8)`, a
  ratio of Python ints, so `==` between two evaluations is exact and the exact
  comparison in cells 21 and 23 is the right one.
- **`transforms.Normalize` accepts the `(3, 1, 1)` tensors** defined in cell 14,
  and `ToTensor()` is bit-identical to `PILToTensor().float() / 255` — checked, so
  the quoted assistant pipeline and `inorm` agree exactly on the same crop, which
  is what makes the ten scorings comparable with the clean one.

### Not checked

- **Whether the notebook's 20-epoch and 8-epoch runs reproduce the deck's
  ratios.** The header claims the comparison "is internally consistent" and the
  module docstring claims "every ratio the lecture turns on survives the
  shortening". Confirming that means running both training cells, which the brief
  excludes. The deck's figures are 15.3% / 81.2% / 88.0% at 80 and 15 epochs.
- **Whether `assert FT_TEST > SCRATCH_TEST` holds at 8 epochs against 20.** Very
  likely — the deck's gap is 73 points — but it is an assert on an unexecuted
  cell.
- **Whether cell 22's early-stopping contrast survives the shortening. This is
  the one I would check first.** The deck's peaks are epoch **14** clean and epoch
  **12** augmented, from a 15-epoch run; this notebook runs **8**, so both curves
  may still be climbing and both `argmax` values may be 8, in which case the cell
  prints two identical epoch numbers under prose that has already asserted *"it
  selects a different model"*. §2.2 says argue the structural version: the spread
  measured in cell 21 is the reason the criterion is unusable, whatever the two
  argmaxes happen to be. This script's cell 22 does that.
- **The Colab T4 wall clock.** Every accelerator figure here is MPS on an M4 Max
  used as a proxy, and every CPU figure is 2 threads on the same machine, taken as
  the minimum of repeated runs while the machine was under heavy load. I did not
  run a T4, and I did not run the training cells end to end — the long ⏱ figures
  are per-step cost times step count.
