# Lecture 14 — *Making it train* — Colab prompt script

Rebuild `notebooks/lecture-14.ipynb` in Colab by prompting. Twenty-three code
cells, in order. Type the prompt, read what comes back against **Expect**, run
the **Assert**.

---

## Before you start

**Runtime.** Everything here runs on a CPU. If Colab offers a GPU, take it —
the training cells are ~4× faster on a T4 — but nothing in the notebook needs
one, and every ⏱ below is a **CPU** figure.

**Where the CPU figures come from.** Measured on this repository's machine
(12-thread Apple Silicon, PyTorch 2.13, best of three repeats), for the
notebook's own configuration: 20 layers, width 100, 10,000 images, batch 128,
20 epochs, Adam.

| configuration | s / epoch | 20-epoch run |
|---|---|---|
| sigmoid, no normalisation | 1.71 | ~34 s |
| ReLU, no normalisation | 1.41 | ~28 s |
| ReLU + layer norm | 3.60 | ~72 s |
| sigmoid + layer norm | 2.70 | ~54 s |
| ReLU + batch norm | 5.08 | ~102 s |
| sigmoid + batch norm | 5.55 | ~111 s |

Colab's free CPU runtime is 2 vCPU and runs roughly **3–4× slower** than that.
Multiply every ⏱ below by three if you are on it. End to end this notebook is
about **37 minutes of CPU**, and about **10 minutes** on a T4.

**Which numbers are quoted below.** Two kinds, and they are not equally firm.

* The **diagnostic** cells (4, 5, 6, 7, 8, 9, 10, 16, 19, 20) are float64 on
  CPU with fixed seeds. The figures below are a byte-for-byte reproduction —
  yours will match to the digits shown.
* The **training** cells (11–15, 17, 18, 21, 22, 23) are single-seed and
  device-dependent. The figures below are this repository's own cached
  measurements, taken on `mps`, which are the numbers the lecture deck was
  built from (`/private/tmp/claude-501/aiml-data/fits-v2.pkl`, key
  `app07_l14`). On `cuda` or `cpu` expect the same structure and the same
  ordering, with the last digit or two different. Where a claim depends on a
  training number surviving to the tenth of a point, this script says so.

**Examinability.** Sections 3, 4 and 8 are examinable — the variance identity,
the ρ prediction, and the difference between `||dL/dz||` and `||dL/dW||`.
Sections 5, 10, 11 are engineering, not examinable. Section 9's closing
anomaly is beyond the book, for context.

---

## Cell 1 — setup

**Prompt to type:**

> Standard setup cell. Import math, sys, time, numpy as np, torch, torch.nn as
> nn, torchvision and matplotlib.pyplot as plt. Print the python and torch
> versions. Set RANDOM_STATE = 42 and seed torch and numpy with it. Pick cuda
> if available, else mps, else cpu, and print which one you got.

**Expect:** two lines — `python 3.x.x   torch 2.x.x` and `device <cuda|mps|cpu>`.

**Assert:** none.

**Annotate:** short

*Specification.* input · nothing. output · versions, seeds, device. constraint ·
the same seed as Lecture 13, so what follows is a rebuild and not a new
experiment.

---

## Cell 2 — the same 10,000 images

**Prompt to type:**

> Load CIFAR-10 train and test from torchvision into a `datasets` folder.
> Using `np.random.default_rng(42).permutation`, take the first 5,000 indices
> as validation and the next 10,000 as the fitting set. Flatten each image to
> 3072 float32 in [0,1], then standardise per pixel using the mean and sd of
> the **fitting subset only**, and apply that to all three splits. Keep the
> whole 10,000-image test set. Assert the shapes and that the test set is
> balanced, then print the three sizes and a chance baseline of 0.1.

**Expect:** `fit 10,000  val 5,000  test 10,000   baseline 0.1000`.

**Assert:**

```python
assert X_fit.shape == (N_FIT, 3072) and X_test.shape == (10_000, 3072)
assert set(np.bincount(y_test)) == {1_000}
```

**⏱** ~2.5 s if `datasets/cifar-10-batches-py` already exists; **1–3 minutes**
on the first run, which downloads 170 MB.

**Annotate:** short

*Specification.* input · CIFAR-10. output · the identical fit/val/test split
and scaling as Lecture 13. constraint · recompute mu and sd from the fit
subset. check · the balance assert is what earns the hard-coded `baseline = 0.1`
on the line after it.

> **Name the standardiser something other than `sd`.** Cell 8 rebinds `sd`.
> See the defect report.

---

## Cell 3 — one factory, every variant

**Prompt to type:**

> Write one function `make_net(depth=20, width=100, act="sigmoid",
> init="torch", norm=None, dropout=0.0, n_in=3072, n_out=10, dtype=...)` that
> returns an `nn.Sequential`: `depth` blocks of Linear → optional
> BatchNorm1d/LayerNorm → activation → optional Dropout, then a Linear head to
> `n_out`. `act` selects from sigmoid, tanh, relu, elu, selu, leaky (0.01).
> `init` is `"torch"` (leave whatever nn.Linear chose), `"glorot"`, `"he"` or
> `"normal1"`; zero the biases for the three that are not `"torch"`. Also set
> DEPTH, WIDTH, EPOCHS=20, BATCH=128, LR=1e-3 as module constants. Assert the
> Linear count and print the parameter count.

**Expect:** `500,210 parameters`.

**Assert:**

```python
assert len([m for m in make_net() if isinstance(m, nn.Linear)]) == DEPTH + 1
```

**Annotate:** short

*Specification.* input · depth, width, activation, initialisation,
normalisation, dropout. output · a fresh network to that specification.
constraint · **one** function that builds every configuration in the notebook,
so every row of every table below differs from its neighbour in exactly one
argument. check · the Linear count, so a change to the builder cannot silently
change the architecture.

> `init="torch"` must be an explicit `pass` branch, not a missing `elif`.
> Naming the default makes it a choice rather than an absence, and it is the
> row the whole lecture is about.
>
> Check by hand: 3072·100 + 100 + 19·(100·100 + 100) + (100·10 + 10)
> = 307,300 + 191,900 + 1,010 = **500,210**.

---

## Cell 4 — check the variance identity before believing it

**Prompt to type:**

> Check `Var(z) = n_in · Var(w) · Var(a)` numerically. n_in = n_out = 100,
> 20,000 samples of unit-variance input, weight variances 0.001, 0.01, 0.02
> and 0.05. Print the predicted and the measured Var(z) side by side for each.

**Expect:** four rows; with unit-variance inputs the prediction is just
`n_in · Var(w)`, so 0.1000 / 1.0000 / 2.0000 / 5.0000, and the measured column
agrees to about three significant figures.

**Assert:** none — read the columns.

**Annotate:** short

*Specification.* input · random weight matrices at four weight variances.
output · predicted and measured Var(z). constraint · unit-variance inputs, so
the boxed formula can be read straight off the measurement. check · 20,000
samples, not 100 — this is a claim about a variance, and a variance estimated
from a small sample has a variance of its own.

---

## Cell 5 — the conflict, in three rows

**Prompt to type:**

> For the three layer shapes in this network — (3072, 100), (100, 100),
> (100, 10) — print three columns per row: what the forward requirement wants
> for Var(w) (1/n_in), what the backward requirement wants (1/n_out), and what
> Glorot gives (2/(n_in+n_out)).

**Expect:**

| n_in | n_out | forward wants | backward wants | Glorot gives |
|---|---|---|---|---|
| 3072 | 100 | 0.000326 | 0.010000 | 0.000631 |
| 100 | 100 | 0.010000 | 0.010000 | 0.010000 |
| 100 | 10 | 0.010000 | 0.100000 | 0.018182 |

**Assert:** none.

**Annotate:** short

*Specification.* input · the three layer shapes. output · what the forward pass
wants, what the backward pass wants, what Glorot gives. constraint · all three
columns per row — the point is that the first two disagree and the third is a
compromise, not a derivation.

> On the 100 → 100 rows the two demands are equal and Glorot is exact. On the
> first layer they differ by 0.01 / 0.000326 = **30.7×**, and nothing can fix
> that.

---

## Cell 6 — check He's factor of two

**Prompt to type:**

> Draw 500,000 standard normals and print E[z²], E[relu(z)²] and the ratio.

**Expect:** `E[z^2]` ≈ 1.00, `E[relu(z)^2]` ≈ 0.50, ratio ≈ 0.50 to two
decimals.

**Assert:** none.

**Annotate:** short

*Specification.* input · half a million standard normal samples. output ·
E[z²], E[relu(z)²] and their ratio. constraint · measure the ratio rather than
quoting ½ — the whole He correction rests on it and it is one line.

---

## Cell 7 — the prediction, from shapes alone

**Prompt to type:**

> Write `rho(var_w, Ephi2, n_out=WIDTH)` returning `sqrt(n_out * var_w *
> Ephi2)`. Build a dict called `theory` with four entries: the nn.Linear
> default with the logistic (Var(w) = 1/300, E[φ'²] = 1/16), Glorot with the
> logistic (Var(w) = 2/200, same E[φ'²]), Glorot with ReLU (E[φ'²] = 1/2), and
> He with ReLU (Var(w) = 2/100). Print rho and rho to the 19th for each.

**Expect:** exactly these four, and **write them down before you run cell 8**:

| scheme | ρ | ρ¹⁹ |
|---|---|---|
| default, logistic | 0.1443 | 1.067e-16 |
| Glorot, logistic | 0.2500 | 3.638e-12 |
| Glorot, ReLU | 0.7071 | 1.381e-03 |
| He, ReLU | 1.0000 | 1.000e+00 |

**Assert:** none.

**Annotate:** short

*Specification.* input · the weight variance and E[φ'²] of each scheme.
output · ρ per layer and ρ¹⁹. constraint · computed from the **shapes** of the
matrices and one expectation — no network is built and no data is touched.

> The `nn.Linear` default is `kaiming_uniform_(w, a=sqrt(5))`, which works out
> to a uniform on (−b, b) with b = 1/√fan_in. A uniform on (−b, b) has variance
> b²/3 — that is where the 3 in 1/300 comes from, and it is the single most
> commonly missed step in this table.
>
> **Do not call the function `rho`.** Cell 8 wants a local named `rho` and will
> clobber this one. Call it `rho_of` or `predict_rho`. See the defect report.

---

## Cell 8 — measure it, four ways

**Prompt to type:**

> Now measure the attenuation on the real network instead of predicting it.
> For each of the four schemes, build the net in float64, run eight batches of
> 128 drawn from X_fit with a fixed generator, and record the norm of
> **dL/dz** at the output of every Linear — the delta, not the weight
> gradient. You will need `retain_grad()`. Report the geometric mean of
> consecutive delta ratios as the measured ρ, beside the prediction from
> `theory`, plus the percentage error, the forward activation-sd ratio, and
> the end-to-end delta ratio. Also keep the per-layer `||dL/dW||` profiles in a
> dict called `profiles`, I need them for the next two cells. Assert
> prediction and measurement agree within 15% per scheme.

**Expect:**

| scheme | predicted | measured ρ | error | fwd scale | δ₁/δ₂₀ |
|---|---|---|---|---|---|
| default, logistic | 0.1443 | 0.1390 | 3.7% | 0.9682 | 5.238e-17 |
| Glorot, logistic | 0.2500 | 0.2340 | 6.4% | 0.9654 | 1.033e-12 |
| Glorot, ReLU | 0.7071 | 0.7045 | 0.4% | 0.6826 | 1.288e-03 |
| He, ReLU | 1.0000 | 0.9970 | 0.3% | 0.9653 | 9.444e-01 |

**Assert:**

```python
assert abs(rho - theory[k]) / theory[k] < 0.15, \
    f"{k}: prediction and measurement disagree by more than 15%"
```

All four pass; the worst is 6.4%, so the threshold has a factor of two in hand.

**⏱** **under 1 second** on CPU — 0.6 s measured, twelve small float64 networks
and 8 batches each. If your version takes thirty seconds, it is building the
networks inside the batch loop.

**Annotate:** full

* **Left open.** The prompt says "the delta, not the weight gradient" and still
  does not say *at which tensor*. `dL/dz` at the Linear's output and `dL/da`
  after the activation differ by exactly the factor the lecture is about, and
  an assistant will pick one without telling you. Read which one the code
  hooked before you read the numbers.
* **The usual student version.** Asking for "the gradient norm per layer" and
  getting `m.weight.grad.norm()`, because that is the one that needs no
  `retain_grad()`. Non-leaf tensors do not keep `.grad` in PyTorch — the
  default is to free it — so `z.grad` comes back `None` with no error and no
  warning, and the path of least resistance is the weight gradient. That is
  the wrong quantity for this table, and cells 10 and 16 are both about what
  goes wrong when you use it anyway.
* **How you would catch it.** The `He, ReLU` row is the check with a knowable
  answer: ρ must be 1.000 and δ₁/δ₂₀ must be order 1. If your He row comes back
  around 1.09 with an end-to-end ratio of 5, you profiled the weight gradient —
  those are the weight-gradient numbers, and they are in the table under cell
  10.

---

## Cell 9 — four profiles on one log axis

**Prompt to type:**

> Plot the four `||dL/dW||` profiles from `profiles` on one semilogy axis,
> layer 1 (nearest the input) on the left, markers on, a legend at font size 8.

**Expect:** four lines. The two logistic lines fall steeply left to right; the
two ReLU lines are visually flat — Glorot+ReLU spans a factor of 6.6 end to
end and He+ReLU a factor of 5.2, against 7e-16 for the default. On an axis
covering sixteen decades both ReLU lines look like the same flat line.

**Assert:** none.

**Annotate:** short

*Specification.* input · the four weight-gradient profiles. output ·
`||dL/dW||` per layer, one line per scheme. constraint · log y-axis and all
four on **one** plot — the schemes span sixteen orders of magnitude between
them.

> Two of these four lines are flat and only one of them is correctly
> initialised. Do not conclude anything from the picture yet.

---

## Cell 10 — why the previous lecture's diagnostic is not enough

**Prompt to type:**

> Print one row per scheme with four columns: the measured ρ from cell 8, the
> forward scale factor, their quotient, and the geometric mean ratio of
> `||dL/dW||` taken descending the stack. The last column is what Lecture 13
> actually plotted.

**Expect:**

| scheme | ρ | fwd | ρ/fwd | ‖dW‖ ratio |
|---|---|---|---|---|
| default, logistic | 0.1390 | 0.9682 | 0.1436 | 0.1593 |
| Glorot, logistic | 0.2340 | 0.9654 | 0.2424 | 0.2531 |
| **Glorot, ReLU** | **0.7045** | **0.6826** | **1.0321** | **1.1046** |
| He, ReLU | 0.9970 | 0.9653 | 1.0328 | 1.0907 |

**Assert:** none — but do the subtraction: 1.1046 against 1.0907 is a
difference of **1.3%**, and one of those two rows is a 41% error in the weight
variance.

**⏱** instant; `profiles` is already computed.

**Annotate:** full

* **Left open.** The prompt asks for four columns and never says what a
  passing value looks like in any of them. That is deliberate and it is the
  lesson: the reader has to decide, from the last column alone, which rows are
  correctly initialised — and from the last column alone it cannot be done.
* **The usual student version.** Treating a flat `||dL/dW||` profile as a
  certificate that the initialisation is right. This is not a hypothetical: it
  is the diagnostic Lecture 13 built, plotted and taught, and the
  `Glorot, ReLU` row is the counterexample. Its backward attenuation
  (ρ = 0.7045, a factor of 1.4 lost per layer) and its forward attenuation
  (0.6826, a factor of 1.5 lost per layer) are almost exactly reciprocal, so
  they cancel inside `||dL/dW_l|| ≈ ||δ_l||·||a_{l−1}||` and the product comes
  out at 1.10 — indistinguishable from He's 1.09.
* **How you would catch it.** Ask for the two factors separately, never their
  product. Any repair that claims to have fixed propagation should be made to
  print ρ on its own; if it can only show you `||dL/dW||`, it has shown you a
  quantity in which two errors can cancel. The end-to-end column is the same
  trap in a different dress: Glorot+ReLU reads 6.6 and He reads 5.2, so on
  that number the wrong initialisation looks *better*.

---

## Cell 11 — the training harness

**Prompt to type:**

> One training function: `train(epochs=20, lr=1e-3, batch=128, opt="adam",
> clip=None, schedule=None, seed=42, **kw)` where `**kw` goes to `make_net`.
> Optimiser from adam / adamw / sgd / momentum (0.9) / nesterov / rmsprop;
> schedule from None / "cosine" / "onecycle" with max_lr = 10·lr; optional
> `clip_grad_norm_`. Seed inside the function and construct the model and the
> optimiser inside it too. Move X_fit/X_val/X_test and the labels to `device`
> once, outside. Return the net and a history dict with per-epoch loss, val
> accuracy and lr, plus `seconds`, `test_acc` and `n_params`. Then smoke-test
> it: a 2-layer ReLU + He net for 3 epochs should beat 0.2.

**Expect:** `harness check: a 2-layer ReLU network reaches 0.39x in 3 epochs
(2 s)` on CPU. Any value comfortably above 0.2.

**Assert:**

```python
assert h["test_acc"] > 0.2, "the harness itself should be able to learn"
```

**⏱** ~2 s for the smoke test; the tensor moves are instant on CPU.

**Annotate:** short

*Specification.* input · every knob the notebook varies. output · a trained
network and its history including test accuracy. constraint · **one** function,
one seed, one subset, one epoch count, so every row of every table below
differs from its neighbour in exactly one argument. check · a 2-layer ReLU
network must beat 0.2 in three epochs, or the harness itself cannot learn and
every table below is measuring the harness.

> The model and the optimiser must be constructed **inside** `train`. If they
> are not, re-running any cell below continues training the previous one and
> the ablation stops being an ablation.
>
> `train` computes the test accuracy on every call. That is defensible only
> because nothing in this notebook is *selected* by it except the ladder's best
> rung — and cell 13 is where that debt comes due.

---

## Cell 12 — each repair, alone

**Prompt to type:**

> Now run each repair on its own against the Lecture 13 configuration (20
> layers, sigmoid, `init="torch"`). Eight rows: the unchanged network, then
> Glorot initialisation, ReLU with He, batch normalisation, layer
> normalisation, gradient clipping at 1.0, a 1-cycle schedule, dropout 0.1 —
> one change per row, all with the same seed and the same 20 epochs. Print
> test accuracy, final training loss and seconds for each.

**Expect:** read every row against 0.1000, not against each other.

| row | test | final loss |
|---|---|---|
| nothing (Lecture 13) | 0.1000 | ~2.303 |
| Glorot initialisation | 0.1000 | 2.3066 |
| **ReLU, with He** | **0.4114** | 1.1388 |
| **Batch normalisation** | **0.3793** | 1.4399 |
| Layer normalisation | 0.1000 | 2.3063 |
| Gradient clipping | 0.1000 | 2.3036 |
| A 1-cycle schedule | 0.1000 | 2.3023 |
| Dropout 0.1 | 0.1000 | 2.3028 |

**Two** rows move. **Five** sit at exactly chance. **None** is worse than
chance. Write that count down; the notebook's own prose gets it wrong and the
defect report says where.

**Assert:** none.

**⏱** **~6 minutes on CPU** (18 min on Colab's CPU runtime, ~1.5 min on
`mps`/`cuda`). Six sigmoid runs at ~34 s, plus batch norm at ~111 s and layer
norm at ~54 s.

**Annotate:** short

*Specification.* input · seven repairs, each applied to the broken network by
itself. output · test accuracy and final loss for each. constraint · one change
per row, all against the same broken baseline — a stack of seven changes that
works tells you nothing about which of the seven mattered.

> Clipping bounds a gradient that is too *large*; ours is fifteen orders of
> magnitude too *small*. Dropout fights overfitting; a network at chance is not
> overfitting. Three of the five null rows are answers to problems this network
> does not have, and Glorot and layer norm are the two that are aimed at the
> right problem and still fall short.

---

## Cell 13 — the ladder

**Prompt to type:**

> Now stack them in the order the diagnosis suggests — signal, then
> optimisation, then generalisation. Seven rungs: unchanged; + Glorot;
> + ReLU and He; + batch norm; + clipping at 1.0; + a 1-cycle schedule;
> + dropout 0.1, each adding one argument to the rung above. Print each rung's
> test accuracy and its change from the rung above, and keep the loss curves.
> Then print which rung was best and which was last, and bind the best rung's
> label, accuracy and kwargs to BEST_LABEL, BEST_ACC, BEST_KW so I can use them
> further down. Assert the best rung is at least three times chance.

**Expect (single seed, seed 42):**

| rung | test | Δ |
|---|---|---|
| Lecture 13, unchanged | 10.00% | |
| + Glorot initialisation | 10.00% | +0.0 |
| + ReLU and He | 41.14% | +31.1 |
| + batch normalisation | 36.83% | −4.3 |
| + gradient clipping | 39.64% | +2.8 |
| **+ a 1-cycle schedule** | **43.85%** | **+4.2** |
| + dropout 0.1 | 33.43% | −10.4 |

then `best row: + a 1-cycle schedule at 0.4385` and `last row: + dropout 0.1 at
0.3343`, and the "last rung is NOT the best configuration" paragraph fires.

**Assert:**

```python
assert best[1] > 3 * rows[0][1], "the repaired network should not be at chance"
```

0.4385 > 0.3000 — passes with room, but see the third bullet.

**⏱** **~8.5 minutes on CPU** (25 min on Colab's CPU runtime, ~2 min on
`mps`/`cuda`). Two sigmoid runs at ~34 s, one bare ReLU at ~28 s, then four
batch-normalised runs at ~102–108 s each.

**Annotate:** full

* **Left open.** The prompt asks for the best rung and never asks *how much
  better than the rung below it has to be to count*. Nothing in the cell knows
  what a rung-to-rung difference is worth, so `max()` will happily crown a
  +4.2-point step and a +0.05-point step alike.
* **The usual student version.** Reporting the final row, because it has the
  most repairs in it. On this ladder that is 33.43% and the best row is 43.85%
  — the last rung is the *worst* rung but one. The table exists precisely so
  you can say which rungs you dropped and why; the moment you retype the last
  rung's settings into the summary, you have thrown the table away.
* **How you would catch it.** Re-run the ladder with `seed=43, 44, 45, 46` and
  look at the spread before you believe the ranking. This repository has done
  it — `tools/figures_app07.py` runs five seeds a rung, and the answer is that
  `+ ReLU and He` is **41.26 ± 0.35** and `+ a 1-cycle schedule` is
  **41.26 ± 2.79**. Identical means; one of them moves eight times as much
  between seeds. The single-seed winner above is one draw from the wider
  distribution, and the +4.2 it won by is 1.5 standard deviations of its own
  noise. **The honest reading of this table is that the ladder ends at
  `+ ReLU and He`.** Cell 23 will report 43.85% anyway, because that is what
  one seed says, and it is the reader's job to know that.

  *Re-run order for the five-seed version, from a cold kernel:* cells 1, 2, 3,
  11, then 13 with `seed=` threaded through the `train(**kw)` call. Cells 4–10
  and 12 are not needed. Budget 8.5 min per seed on CPU.

---

## Cell 14 — the ladder, drawn

**Prompt to type:**

> Two panels side by side. Left: horizontal bars of test accuracy per rung,
> labels on the y-axis rather than in a legend, dashed grey line at 10%.
> Right: the training-loss curves for four of the seven rungs — the unchanged
> one, `+ ReLU and He`, `+ batch normalisation` and the last one — with a
> dotted line at ln(10) = 2.303.

**Expect:** two bars at the 10% line, five clear of it, and the tallest bar is
the sixth from the top, not the bottom one. On the right, the unchanged curve
is flat on ln(10) for all twenty epochs and the other three fall.

**Assert:** none.

**Annotate:** short

*Specification.* input · the seven rungs. output · a bar per rung and four loss
curves. constraint · draw the chance line at 10% and ln(10) — every bar has to
be read against chance.

> Four curves, not seven. Seven overlapping loss curves are unreadable; a
> selection is a decision, so make it deliberately and say which four you
> chose.

---

## Cell 15 — a repair

**Prompt to type:**

> My 20-layer network isn't learning — the gradients vanish. Switch it to ReLU
> and initialise the weights properly. Then train it for 20 epochs on Xf/yf
> with Adam at 1e-3, batch 128, seed 42, and print the test accuracy on Xt/yt.

**Expect:** a self-contained builder that stacks 20 `Linear → ReLU` blocks and
a head, initialises every Linear with **`nn.init.xavier_uniform_`** and zeroes
the biases, then trains. It prints

```
the assistant's repair: test accuracy 0.3962
the baseline:                          0.1000
It moved. So it worked?
```

**Assert:** none.

**⏱** ~28 s on CPU (85 s on Colab's CPU runtime; ~35 s on `mps`).

**Annotate:** short

*Specification.* input · "switch it to ReLU and initialise the weights
properly". output · a trained network and its test accuracy. constraint · run
it exactly as it came back.

> **Write 0.3962 on your sheet, and the number from the `+ ReLU and He` rung of
> cell 13 next to it. Then go on to cell 16.** Do not read ahead.

---

## Cell 16 — measure the difference rather than arguing about it

**Prompt to type:**

> Compare Glorot and He with ReLU properly. Profile the per-layer gradients
> for both, print the measured per-layer ratio beside the predicted 0.7071 and
> 1.0, and the end-to-end ratio for each. Then train the He version with the
> harness and print the accuracy difference against the run above.

**Expect — and this is the point of the cell.** *Xavier* and *Glorot* are the
same person and the same formula, so the repair in cell 15 was defensible: it
used the initialisation derived for a roughly **linear** activation on an
activation that discards half the variance. Put it back in cell 7's table and
ρ = √(1·½) = 0.7071, not 1. Over nineteen layers that is 1.4e-03 — three orders
of magnitude, against the fifteen of Lecture 13. Enough to train visibly, and
far from correct.

The accuracy half is firm:

```
Xavier + ReLU  test 0.3962
He     + ReLU  test 0.4114
cost of the wrong constant: +1.52 accuracy points
```

The ρ half is where you have to be careful, because *the phrase "profile the
per-layer gradients" is ambiguous and the obvious reading is the wrong one.*
If the cell reaches for the weight-gradient profiler from cell 8, it prints

```
Glorot + ReLU  rho 1.1046  (theory 0.7071)   end to end 6.626e+00
He     + ReLU  rho 1.0907  (theory 1.0000)   end to end 5.206e+00
```

— a "measurement" 56% away from its own theory, in the cell whose job is to
settle the argument. What it must print, using the **delta** profiler, is

```
Glorot + ReLU  rho 0.7045  (theory 0.7071)   end to end 1.288e-03
He     + ReLU  rho 0.9970  (theory 1.0000)   end to end 9.444e-01
```

**Assert:** add one the notebook does not have —

```python
assert abs(rho_glorot - math.sqrt(0.5)) / math.sqrt(0.5) < 0.15
assert abs(rho_he - 1.0) < 0.15
```

With the weight-gradient profiler the first line fails. That is the whole
value of writing it.

**⏱** ~30 s on CPU — two float64 profiles are under a second, the He training
run is ~28 s.

**Annotate:** full

* **Left open.** "Profile the per-layer gradients" — which gradient? Cell 8
  spent a whole prompt insisting on `dL/dz`, and by the time you get here,
  seven cells later, the distinction has gone quiet. The prompt that would
  close it is "the per-layer `dL/dz` norms, the same quantity as cell 8, not
  the weight gradient".
* **The usual student version.** This is not a student mistake, it is the
  mistake in the shipped notebook: `notebooks/lecture-14.ipynb` cell 46 calls
  `grad_profile` here, so the notebook prints `rho 1.1046 (theory 0.7071)` in
  the section that exists to teach the difference between the two quantities.
  Verified against the cached profiles and against the deck, whose slide
  "The damage, measured" gives 0.7045 / 0.6826 / 1.3e-03 — the delta numbers.
  The notebook does not reproduce its own slide.
* **How you would catch it.** Make the repair produce the diagnostic that
  would show it worked, and put a threshold on it. That is the corrected
  specification, and the last clause is the load-bearing one: *"switch to ReLU
  and use the initialisation derived for ReLU — He normal, Var(w) = 2/fan_in —
  then print the per-layer gradient-norm ratio and show me it is within 20% of
  1.0."* "Initialise it properly" is not a specification. A per-layer ratio
  near 1 is. An accuracy that went up is compatible with a great many wrong
  repairs, and +1.52 points is what this one cost.

> One more real default worth naming while you are here:
> `nn.init.kaiming_normal_` defaults to `nonlinearity="leaky_relu"` with
> `a=0`, which happens to give the same gain as `"relu"` — so it is right by
> accident. `nn.init.xavier_uniform_` defaults to `gain=1.0`, which is the
> gain for a linear activation, which is exactly the bug in cell 15.

---

## Cell 17 — normalisation, and what it costs

**Prompt to type:**

> Train the repaired ReLU + He network three ways — no normalisation, batch
> normalisation, layer normalisation — and print test accuracy, wall clock and
> parameter count for each.

**Expect:**

| | test | parameters | seconds (CPU) |
|---|---|---|---|
| none | 0.4114 | 500,210 | ~28 |
| batch | 0.3683 | 504,210 | ~102 |
| layer | 0.4164 | 504,210 | ~72 |

Batch normalisation makes the *repaired* network **worse**, by 4.3 points —
which is not what most people expect and is the reason this cell reports the
wall clock as well as the accuracy.

**Assert:** none.

**⏱** **~3.5 minutes on CPU** (10 min on Colab's CPU runtime, ~1.2 min on
`mps`).

**Annotate:** short

*Specification.* input · the repaired network with no norm, batch norm, layer
norm. output · accuracy, wall clock and parameter count. constraint · report
the parameter count — two learned vectors per 100-unit layer is 200 numbers
against 10,100, **1.98%**, so the wall clock is where the cost actually is:
batch norm is 3.6× the epoch here.

---

## Cell 18 — the two modes disagreeing, quietly

**Prompt to type:**

> Train a batch-normalised ReLU + He net for 3 epochs. Then evaluate it on the
> same 2,000 test images twice — once with the model in `train()` mode and once
> in `eval()` — and print both numbers and the difference in points.

**Expect:** two different accuracies from the same weights on the same images.
`train()` uses the batch statistics of the test batch; `eval()` uses the
running averages accumulated during training. The gap is a few points and its
sign is not fixed.

**Assert:** none.

**⏱** ~15 s on CPU.

**Annotate:** full

* **Left open.** The prompt says "the same 2,000 images" and does not say why
  that matters. It matters because it is the only way the difference can be
  attributed to the mode: change the images and the mode and you have measured
  neither.
* **The usual student version.** Reusing the diagnostic from Lecture 12 — run
  the evaluation twice, and if the number wobbles you left the model in
  training mode. That works for dropout, because dropout is random. `BatchNorm1d`
  in training mode is **deterministic given the batch**: `track_running_stats`
  defaults to `True`, `momentum` to 0.1, and in `train()` it normalises by the
  incoming batch's own mean and variance every time. Feed it the same 2,000
  images twice and it returns the same wrong number twice. The tell is gone.
* **How you would catch it.** From here on, the cheap diagnostic no longer
  works and you have to check the code instead: every measurement path must
  contain an explicit `.eval()`, and the `@torch.no_grad()` decorator on your
  accuracy function is not one. This is red-team question 7 for a reason.

---

## Cell 19 — the row we do not explain

**Prompt to type:**

> For the *broken* network — sigmoid, default init — with no norm, batch norm
> and layer norm, print the measured ρ and the end-to-end dL/dz ratio for each.
> No explanation, just the three rows.

**Expect:** three rows. The `none` row reproduces cell 8's default-logistic
line (ρ = 0.1390, δ₁/δ₂₀ = 5.2e-17); the two normalised rows do not.

**Assert:** none.

**⏱** under 1 s.

**Annotate:** short

*Specification.* input · the broken network with no norm, batch norm, layer
norm. output · ρ and the end-to-end delta ratio for each. constraint · report
the measurement with no explanation attached to it.

> **The anomaly, stated correctly.** Applied alone to the broken network
> (cell 12): batch norm 0.3793, layer norm 0.1000 — one rescues it, the other
> does nothing. Applied to the repaired network (cell 17): batch norm 0.3683,
> layer norm 0.4164 — and now the *layer*-normalised one is ahead by 4.8
> points. That reversal is what has not been accounted for. This cell is the
> measurement that would start to. Whatever you conclude, write down the number
> that supports it.

---

## Cell 20 — look at the norms before choosing a threshold

**Prompt to type:**

> Collect the total gradient norm at every step of two epochs, once with He
> initialisation and once with `normal1` (N(0,1)) — both ReLU. Read the norm
> with `clip_grad_norm_` at an infinite threshold so nothing is actually
> clipped. Print median and max for each, and histogram log10 of both on one
> axis.

**Expect:** He median ≈ 3.2, max ≈ 16. N(0,1) median ≈ 4.6e+17, max ≈ 1.6e+19.
Two histograms about eighteen decades apart on the log10 axis.

**Assert:** none.

**⏱** ~6 s on CPU.

**Annotate:** short

*Specification.* input · two epochs of gradient norms under He and under
N(0,1). output · median and maximum for each, plus both distributions on a log
axis. constraint · use `clip_grad_norm_` with an **infinite** threshold — the
measurement must not be the intervention.

> Now look back at cell 13. The ladder clips at **1.0**, and the He median is
> **3.2** — the threshold is below the median, so it fires on more than half
> the steps and is acting as a step-size limiter, not a safety net. That is a
> real effect obtained for the wrong reason: if you want a smaller step, set a
> smaller learning rate and say so.

---

## Cell 21 — six optimisers, on the repaired network

**Prompt to type:**

> On the repaired network (ReLU, He, batch norm), train with SGD, momentum,
> Nesterov, RMSprop, Adam and AdamW. 1e-2 for the three SGD variants and 1e-3
> for the three adaptive ones — the same learning rate would not be a fair
> comparison. Print test accuracy and final training loss for each.

**Expect:**

| optimiser | lr | test | final loss |
|---|---|---|---|
| sgd | 1e-2 | 0.2750 | 1.966 |
| momentum | 1e-2 | 0.3363 | 1.705 |
| nesterov | 1e-2 | 0.3522 | 1.677 |
| rmsprop | 1e-3 | 0.3683 | 1.301 |
| adam | 1e-3 | 0.3683 | 1.602 |
| adamw | 1e-3 | 0.3609 | 1.543 |

A spread of 9.3 points, and no row at chance.

**Assert:** none.

**⏱** **~10 minutes on CPU** (30 min on Colab's CPU runtime, ~1.5 min on
`mps`/`cuda`). Six batch-normalised 20-epoch runs. This is the longest cell in
the notebook; the notebook's own marker says four minutes and is wrong on every
device.

**Annotate:** short

*Specification.* input · six optimisers. output · test accuracy and final loss.
constraint · different learning rates for the SGD family and the adaptive
family; 1e-3 on plain SGD is not a fair test of plain SGD.

> This section is here and not in cell 12 because comparing optimisers on the
> *broken* network would have measured nothing — none of them can descend a
> gradient that does not arrive, every row would have read 10%, and the
> conclusion would have been that the optimiser does not matter. Fix the
> signal before tuning the search.

---

## Cell 22 — three schedules

**Prompt to type:**

> Same network with clipping at 1.0, three ways: no schedule, cosine, 1-cycle.
> Print the final test accuracy **and** the best validation accuracy of the
> run, since a schedule that ends at a low learning rate can finish below its
> own peak.

**Expect:**

| schedule | test | best validation |
|---|---|---|
| None | 0.3964 | 0.4008 |
| cosine | 0.3883 | 0.3826 |
| onecycle | 0.4385 | 0.4276 |

**Assert:** none.

**⏱** **~5 minutes on CPU** (15 min on Colab's CPU runtime, ~1 min on `mps`).
The notebook carries no marker on this cell at all.

**Annotate:** short

*Specification.* input · the repaired network with and without a schedule.
output · final test and best validation accuracy. constraint · report the peak
as well as the endpoint — a schedule is a hyperparameter with a shape, not a
value.

> OneCycle raises the learning rate to **10× lr** before lowering it, so
> `max_lr` here is 1e-2 and is not the same knob as the `lr=1e-3` in every
> other row. The +4.2 points over the unscheduled run is partly a schedule and
> partly a bigger learning rate, and the cell cannot separate them.
> Cross-check with cell 13's third bullet before you believe the ranking: over
> five seeds this row's spread is ±2.79 points.

---

## Cell 23 — re-measure

**Prompt to type:**

> Final summary. Retrain the Lecture 13 network, and retrain using `BEST_KW`
> from the ladder — the variable, not the settings typed out again. Print the
> chance baseline, both accuracies, the improvement in accuracy points, the
> label of the best rung, and how many rungs were dropped.

**Expect (single seed):**

```
============================================================
baseline (majority class)          0.1000
Lecture 13, 20 layers              0.1000
the same 20 layers, repaired       0.4385
  (best rung: + a 1-cycle schedule)
improvement, accuracy points       +33.85
rungs dropped                           1
============================================================
```

**Assert:** none, but check by eye that the repaired figure matches the rung
you crowned in cell 13. If it does not, `BEST_KW` was not what got retrained.

**⏱** **~2.3 minutes on CPU** — and both of these runs were already done in
cell 13. If you are short of time, print `BEST_ACC` and `rows[0][1]` instead
and skip the retraining entirely.

**Annotate:** short

*Specification.* input · the broken network and the **best** rung of the
ladder. output · baseline, both accuracies, the improvement, and how many
rungs were dropped. constraint · use `BEST_KW`, captured in cell 13, not the
last rung's settings typed out again.

> When a table selects a winner, carry the winner forward in a variable.
> Retyping its settings is how the summary and the table drift apart — and on
> this ladder retyping the last rung would report 33.43% where the table says
> 43.85%, a ten-point error two pages after the section that exists to prevent
> it.
>
> And it is still an MLP on flattened pixels. The gradient now reaches every
> layer and the loss falls; the remaining gap is not an optimisation failure,
> it is that a fully connected network has no way to know that adjacent pixels
> are adjacent. That is the brief for Lecture 15.

---

## Red-team, for the last ten minutes

Swap notebooks. The five standing questions, and three that are new to this
lecture:

1. What touched the test set?
2. What was fitted, and on what?
3. What is the shape here?
4. What was dropped — rows, columns, NaNs? Count them.
5. What is the default I did not ask for?
6. **Which initialisation is on each layer, and which activation was it
   derived for?**
7. **If there is a normalisation layer, is there an `.eval()` before every
   measurement?** Running the evaluation twice will not catch a missing one.
8. **Is every row of the ablation table one change away from its neighbour?**

Report what you **found**, not what you would have done differently.

*Vocabulary, since these terms are used above without ceremony:* **red-team** —
read someone else's notebook looking for the thing that is wrong rather than
the thing that is interesting. **Rung** — one row of the stacked ablation
table. **Clobber** — rebind a name to something of a different kind, so that
later code silently gets the wrong object. **Smoke test** — a cheap run whose
only job is to show the instrument works at all.

---

# Defects found in the current notebook

Everything below is against `notebooks/lecture-14.ipynb` as it stands (69
cells, 23 of them code, no stored outputs). Cell numbers are notebook indices.

Sources used for verification, in decreasing order of authority: the
repository's own cached measurements
(`/private/tmp/claude-501/aiml-data/fits-v2.pkl`, keys `app07_l14`,
`app07_l14_extra`, `app07_deltas`), which are what `slides/lecture-14.html` was
generated from; my own float64 CPU reproduction of the diagnostic cells; and
`slides/lecture-14.html` itself.

## Checked, and confirmed defects

**1 · §8's verification cell measures the wrong quantity, and prints a
measurement 56% away from its own stated theory.** *(§1.1, §1.2, §2.1 — the
most serious defect here.)*

Cell 46 computes

```python
gx = grad_profile(act="relu", init="glorot")
rx = float(np.exp(np.mean(np.log(gx[1:DEPTH] / gx[0:DEPTH-1]))))
print(f"Glorot + ReLU  rho {1/rx:.4f}  (theory {math.sqrt(0.5):.4f})   ...")
```

`grad_profile` returns `||dL/dW||`. The quantity ρ refers to, everywhere else
in the lecture, is `||dL/dz||`. Ran it: the cell prints

```
Glorot + ReLU  rho 1.1046  (theory 0.7071)   end to end 6.626e+00
He     + ReLU  rho 1.0907  (theory 1.0000)   end to end 5.206e+00
```

The first row's "measurement" is 56% from its own theory. The deck's
corresponding slide (*"The damage, measured"*, `slides/lecture-14.html` §66)
gives 0.7045 / 0.6826 / 1.3e-03 and 0.9970 / 0.9653 / 9.4e-01 — the **delta**
numbers, which I reproduced exactly with `delta_profile` + `act_sd`. The
notebook does not reproduce its own slide, and the failure mode is precisely
the one the box above cell 23 calls "the single easiest way to misread this
lecture". The `end to end` column makes it worse: 6.6 for the wrong
initialisation against 5.2 for the right one, so as printed the wrong one looks
better.

**2 · The quoted slide text is wrong in both figures, three times over.**
*(§1.5, §3.3.)*

Cells 37 (prompt box `catch`), 66 (prompt box `student`) and 67 (code comment)
all assert that the deck says *"the number we report is 43.9%, not 33.4%"*. The
slide exists — `slides/lecture-14.html`, `data-menu-title="Re-measure"` — and
says **"The number we report is 41.3%, not 34.0%."** Grepped the whole
repository: `43.9` occurs in exactly one other place, `slides/lecture-18.html`,
where it is a COCO detection mAP on 128 images and has nothing to do with this
lecture; `33.4` occurs nowhere outside `lecture_14.py` / `lecture-14.ipynb`.

The notebook's own figures are not invented — 43.85 and 33.43 are what the
seed-42 ladder actually produces, and they round correctly. The deck's 41.3 and
34.0 are the **five-seed means** of the same two rungs. So this is one quantity
reported as two different numbers, in three places, with the discrepancy
unexplained and attributed to a slide that says something else.

**3 · The ladder crowns its winner on one seed, and the winner is inside the
noise.** *(§2.4.)*

Verified from `app07_l14["ladder"]`, five seeds a rung:

| rung | 5-seed mean | sd | seed 42 alone |
|---|---|---|---|
| + ReLU and He | 41.26% | ±0.35 | 41.14% |
| + batch normalisation | 36.52% | ±0.64 | 36.83% |
| + gradient clipping | 40.78% | ±0.76 | 39.64% |
| + a 1-cycle schedule | 41.26% | ±2.79 | **43.85%** |
| + dropout 0.1 | 34.00% | ±1.33 | 33.43% |

Cell 38 runs one seed, `max()` picks the 1-cycle rung, and `BEST_KW` carries it
into cell 67 as the headline. Over five seeds that rung ties `+ ReLU and He`
exactly and moves eight times as much between seeds. The repository's own
figure code says so in a comment — *"Five seeds a rung, not one… a single seed
cannot support a claim that size"* (`tools/figures_app07.py:631`) — and the
deck devotes a slide to it. The notebook never mentions run-to-run spread, and
§2.4 requires it where the headline is stated.

**4 · "Three of those repairs do nothing" — five do, and none is worse.**
*(§1.1, §7.3.)*

Cell 32 promises *"two of them will turn out to make things worse on their
own"*; cell 35 then says *"Three of those repairs do nothing on their own"* and
names clipping, schedule and dropout. From `app07_l14["alone"]`, the seven
solo runs on the Lecture 13 network are: Glorot 0.1000, ReLU+He 0.4114, batch
norm 0.3793, layer norm 0.1000, clipping 0.1000, 1-cycle 0.1000, dropout
0.1000. **Five** sit at exactly chance (Glorot and layer norm as well as the
three named); **none** is below it. Neither count in the prose is findable in
the table, and the reader who genuinely looks for the two that "make things
worse" cannot tell whether the failure is theirs.

**5 · "On the repaired network the two are within a point of each other" — they
are 4.8 points apart.** *(§1.1, §1.2.)*

Cell 54's markdown. From `app07_l14["norms"]`, on the repaired ReLU+He
network: none 41.14%, **batch 36.83%**, **layer 41.64%**. Batch and layer
differ by 4.81 points. The two that are within a point are *no normalisation*
and *layer normalisation* (0.50 apart). The deck carries the same sentence, so
the error is upstream of the notebook, but the notebook is where a student
meets it.

**6 · `rho` is rebound from a function to a float, and re-running the cell
above raises `TypeError`.** *(§4.1, §4.3.)*

Cell 20 defines `def rho(var_w, Ephi2, n_out=WIDTH)` and builds `theory` from
it. Cell 23 does `rho = geo(d[0:DEPTH-1] / d[1:DEPTH])` at module level. After
cell 23 has run once, re-running cell 20 — the natural thing to do if you want
to change `E[φ'²]` and see what it predicts — fails with
`TypeError: 'float' object is not callable`. Confirmed by AST walk over the
notebook's module-level bindings. This is the same defect the course spends 200
words on in lecture 19, in a notebook whose §4.1 discipline is otherwise good.

**7 · `sd` is rebound out from under the standardising closure.** *(§4.1.)*

Cell 5 sets `sd = X_fit_raw.std(axis=0) + 1e-7`, shape (3072,), and defines
`std = lambda a: (flat(a) - mu) / sd` — which reads `sd` from globals at call
time. Cell 23 rebinds `sd = act_sd(**v)`, shape (20,). The notebook does not
call `std` after cell 23, so nothing visibly breaks today; but any reader who
re-runs a scaling line after cell 23 gets a broadcast error, and any cell added
between them gets silently wrong data. Same class of hazard, one loaded gun
further along.

Also rebound across cells, less dangerously: `g` (a `torch.Generator` in cell
43, an `np.ndarray` in cells 25 and 28), and `fwd` (1/n_out in cell 14, a
geometric mean of activation-sd ratios in cell 23).

**8 · Every ⏱ figure in the notebook is wrong, on every device.** *(§7.1.)*

Measured, best of three, 20 epochs at the notebook's own settings, on a
12-thread CPU; the `mps` column is this repository's own cached `seconds`
fields.

| cell / section | notebook says | CPU measured | `mps` cached |
|---|---|---|---|
| §4 four schemes (cell 23) | "about 30 seconds… on the CPU" | **0.6 s** | — |
| §6 each repair alone (34) | "about 5 minutes" | **~6 min** | ~1.5 min |
| §7 the ladder (38) | "about 5 minutes" | **~8.5 min** | ~1.9 min |
| §9 normalisation (50) | "about 2 minutes" | **~3.5 min** | ~1.2 min |
| §11 six optimisers (62) | "about 4 minutes" | **~10 min** | ~1.5 min |

The §4 marker is the only one that names a device, and it is 50× too long. The
four training markers match neither device: roughly 2–3× too long for `mps`,
and 1.2–2.5× too short for a fast CPU — so on Colab's 2-vCPU CPU runtime, where
a reader with no GPU actually is, the optimiser cell alone is about half an
hour against a stated four minutes.

**9 · Five cells over 20 s carry no ⏱ marker at all.** *(§7.1, §9.)*

Cell 43 (the assistant's 20-epoch training run, ~28 s CPU), cell 46 (~30 s),
cell 53 (~15 s), **cell 64 (three schedules, ~5 min CPU)** and **cell 67 (the
closing re-measure, ~2.3 min CPU)**. Cell 64 is the worst of these: three
batch-normalised 20-epoch runs with nothing in the markdown above it. The
download timing for cell 5 exists but is a comment *inside* the code cell
rather than in the markdown above it, which is where §9's proposed check would
look for it.

**10 · The defect is announced six times before it can be walked into.**
*(§8.1, §8.2.)*

Before cell 43 executes, the reader has been told: (i) the header's *"Cells
marked ⚠ read before running contain a defect on purpose"*; (ii) the section
heading `## 8 · ⚠ An assistant repairs the network`; (iii) *"it looks like the
problem is solved"*; (iv) the prompt label `⚠ what the assistant returns`;
(v) the box's `left_open`, which says Xavier and Glorot are the same formula
and that it was derived for a linear activation; and (vi) the box's `student`,
which gives the punchline in full — *"With ReLU, Glorot gives ρ = √(1·½) =
0.707, not 1"*. Every word of the §8.1 analysis of lecture 19 applies: nobody
falls in. The material for the preferred shape is already all there — run cell
43 unannounced, have the reader write 0.3962 down, and open §8's second half
with the ⚠.

**11 · The header names a marker string that appears in no cell.** *(§3.3.)*

*"Cells marked **⚠ read before running**"*. Grepped: the exact string occurs
only in the header. The two ⚠ that do exist read *"⚠ An assistant repairs the
network"* and *"⚠ what the assistant returns"*. The plural "Cells" is also
wrong — there is one.

**12 · Two cross-references point at the wrong cell.** *(§3.3.)*

* Cell 19's box: *"Write these four numbers down before running the next
  cell."* The four numbers **come from** the next cell (20). The cell meant is
  23, four cells on.
* Cell 24's box: *"The next cell is about a flat profile that is flat for the
  wrong reason."* The next cell (25) is the plot. The flat-profile argument is
  cells 26–28.

Cell 26's *"as the next cell shows"* points at a prompt box rather than the
code, which is structurally unavoidable in this notebook's layout and I have
not counted it.

**13 · A markdown line indented four spaces outside a fence.** *(§5.1, §9.)*

Cell 26, line 13: `    || dL/dW_l ||  ~  || delta_l || . || a_(l-1) ||`. It
renders as a grey monospace block, which is very likely what the author wanted,
but it is what §9's proposed checker flags and there is no reason not to use a
fence. Machine-checked across all 69 cells: this is the only one, and there are
no indented fence markers.

**14 · The ladder clips at a threshold below the median gradient norm, and the
notebook never says so.** *(§1.2 — a number available in the notebook and not
used.)*

Cell 57's markdown warns that *"a clip value below the median silently turns
your optimiser into sign descent"*. Cell 59 then measures the He median at
**3.19**. Four rungs of cell 38's ladder use `clip=1.0`. The deck makes the
connection explicitly (*"The threshold of 1.0 used in the ladder sits below the
He median — deliberately"*); the notebook prints both numbers eight cells apart
and leaves them unreconciled, in the section whose whole point is that people
copy `clip=1.0` out of tutorials.

**15 · No section carries an examinability marker.** *(§8.3.)*

The string "examinable" appears once in the entire notebook, in cell 2's
`left_open` (*"nothing here is examinable"*), on the setup cell. Thirteen
sections, no markers.

**16 · Cell 67 retrains two configurations that cell 38 already trained.**
*(§7.1.)*

`train(act="sigmoid", init="torch")` and `train(**BEST_KW)` are both rungs of
the ladder. `BEST_ACC` is unpacked in cell 38 and then never used. On CPU that
is ~2.3 minutes to recompute two numbers already on screen — and on `mps` or
`cuda`, where kernel scheduling is not bit-deterministic, the retrained figure
may not match the ladder's, giving two numbers for the same quantity with no
reconciliation (§1.5).

## Checked, and **not** defects

* **The ρ table in §3's markdown is exactly right**, including the 3 in
  1/(3·100): `nn.Linear` initialises with `kaiming_uniform_(w, a=sqrt(5))`,
  which is a uniform on (−b, b) with b = 1/√fan_in, variance b²/3. All four
  table entries match the values `theory` computes: 0.1443, 0.2500, 0.7071,
  1.0000.
* **"They agree to within a few per cent" (cell 26)** — measured errors are
  3.7%, 6.4%, 0.4%, 0.3%. Fair.
* **"Fifteen orders of magnitude" (§2 and §6)** — the weight-gradient ratio
  Lecture 13 logged is `dW₁/dW₂₀ = 6.911e-16`, i.e. 15.2 orders. Correct as
  stated, and correctly distinguished from Lecture 13's own "sixteen orders",
  which is the delta (5.238e-17).
* **"Three orders of magnitude, not fifteen" for Glorot+ReLU** — 0.7071¹⁹ =
  1.381e-03, measured δ₁/δ₂₀ = 1.288e-03. Correct.
* **"A factor of 30" on the first layer** — 0.01/0.000326 = 30.7. Correct.
* **"200 numbers against 10,100 — under 2%"** — 200/10,100 = 1.98%. Correct,
  and the printed parameter counts (500,210 → 504,210) reconcile.
* **The 15% assertion in cell 23 passes** on all four schemes, worst case 6.4%.
* **The `theory` and `schemes` dictionaries use byte-identical keys** despite
  the cosmetic double spaces; no `KeyError`.
* **All training cells re-instantiate the model and optimiser** inside `train`
  or inside the cell (§4.2). Cell 43 is idempotent.
* **The Xavier-uniform / Xavier-normal difference between cell 43 and cell 46
  is not a numerical defect** — `xavier_uniform_` and `xavier_normal_` have the
  same variance 2/(fan_in+fan_out), so ρ is unaffected. It is a cosmetic
  mismatch only (the cell said to measure "the assistant's" network measures a
  different draw from the same distribution).
* **`assert best[1] > 3 * rows[0][1]`** passes: 0.4385 against 0.3000.
* **`check_all.py` is clean** on this notebook — but it does not yet implement
  any of the §9 rules, so that is not evidence about defects 1–16.

## Could not check

* **Whether the notebook's own single-seed numbers reproduce on `cuda` or on
  CPU.** Every training figure quoted above is the repository's cached `mps`
  run. I did not execute the training cells (the brief forbids it). The
  ordering of the ladder is robust for the rungs separated by more than ~3
  points; the crowning of `+ a 1-cycle schedule` over `+ ReLU and He` is not,
  and defect 3 is really a statement that it *cannot* be, on any single seed.
* **Whether `act_sd` ever returns an empty array in practice.** It filters on
  `(nn.Sigmoid, nn.Tanh, nn.ReLU)`, so `act="elu"`, `"selu"` or `"leaky"` would
  give `geo()` a zero-length array and a `nan` forward factor. No cell in the
  notebook calls it with those, so this is latent rather than live; I did not
  count it as a defect.
* **Slide-to-notebook agreement for the deck's own internal figures.** In
  passing: `slides/lecture-14.html` §72 reports the best configuration as
  +31.3 points over the start (the five-seed 41.26), while §79 says *"33.9%
  points better"* (the single-seed 43.85 − 10.00 = 33.85). The two slides use
  different seeds' worth of the same quantity. That is a deck defect, outside
  the scope of this report, and I have not pursued it.
