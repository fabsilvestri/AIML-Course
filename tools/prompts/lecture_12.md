# Lecture 12 — Rebuilding it in PyTorch, at a Colab keyboard

Prompt script for `notebooks/lecture-12.ipynb`. Nineteen code cells, in order.
Type the prompt, read what comes back against **Expect**, run the **Assert**.

**Thread 6.** Backpropagation is reverse-mode automatic differentiation. Cells
3–5 establish that; cells 6–19 spend it on the same Fashion-MNIST model
Lecture 11 built with `MLPClassifier`, and on the four ways the rebuilt loop
goes silently wrong.

**Annotation budget.** Seven of the nineteen boxes carry the full three-bullet
annotation (cells 1, 4, 10, 11, 12, 14, 15). The other twelve carry the
specification only. Every "usual student version" below names a documented
library default or a failure observed while preparing this script — three of
them were observed in this repository's own code.

### Where the timings come from

Every ⏱ figure below was measured on **one 16-core Apple-silicon laptop, CPU
device, torch 2.13.0, scikit-learn 1.7.2, seed 42**, while that machine was
under heavy load. They are therefore upper bounds for an idle CPU of that
class. No CUDA device was available, so **no T4 figure in this file is
measured** and none is quoted.

One measured fact worth having before you start: on this model — 266,610
parameters, batch 128 — a training step cost **1.62 ms on the CPU and 2.16 ms
on MPS**. The accelerator is *slower* here, because the batches are too small
to pay back the launch latency. If you run this on a Mac, the "speed-up" the
notebook prints in cell 9 is not a statement about GPUs.

Every accuracy below is from that CPU run at seed 42. Yours will differ in the
third decimal on other hardware and in the second on another seed. The
*orderings* and the *signs* are what you check.

---

## Cell 1 — setup and the device

**Prompt to type:**

> print the python, torch and torchvision versions, set seed 42 for torch and
> numpy, then pick a device — cuda if there is one, otherwise mps, otherwise
> cpu — and print which one it picked

**Expect:** three version lines, then `device` and one of `cuda` / `mps` /
`cpu`. If it prints `cpu`, you also want a line telling you what to do about
it (Runtime → Change runtime type → T4 GPU).

**Assert:** none.

**Annotate:** full

* **Left open** — that this one line decides, from here to the end of the
  course, whether a cell takes one minute or twenty. Nothing else in the
  notebook is allowed to hard-code a device name after it.
* **The usual student version** — `device = "cuda"` copied from a tutorial.
  On a Mac and on a CPU-only Colab runtime `torch.cuda.is_available()` is
  `False` and `.to("cuda")` raises `RuntimeError: Torch not compiled with CUDA
  enabled` or `no CUDA GPUs are available`. The quieter variant is
  `device = "cpu"` left in from debugging, which never raises anything.
* **How you would catch it** — print the device. Then, once, print a per-step
  time on both devices before trusting either. Measured here, the accelerator
  lost: 2.16 ms on MPS against 1.62 ms on CPU for this batch size. "GPU" is
  not a synonym for "faster"; it is a synonym for "more arithmetic per launch".

---

## Cell 2 — the same data, the same split

**Prompt to type:**

> load Fashion-MNIST train and test from torchvision into `datasets`, flatten
> to 784 floats scaled to 0–1, then split the training set with
> `np.random.default_rng(42)`: first 5,000 of the permutation to validation,
> the rest to fit, and take the first 12,000 of the fit set. assert the shapes

**Expect:** `fit 12,000 (of 55,000)   val 5,000   test 10,000`. First run
downloads about 30 MB into `notebooks/datasets/FashionMNIST`; after that it is
instant.

**Assert:**

```python
assert len(X_fit_full) + len(X_val) == 60_000
assert X_fit.shape == (SUB, 784) and X_val.shape == (5_000, 784)
```

**Annotate:** short

This must reproduce Lecture 11's split exactly, and it does: `lecture-11.ipynb`
draws the same permutation from the same seed and slices it the same way
(`val_idx, fit_idx = order[:5_000], order[5_000:]`). Insert one extra call on
`rng` above this cell and every comparison in the notebook silently becomes a
comparison of two different validation sets.

---

## Cell 3 — autograd against a derivative you did by hand

**Prompt to type:**

> in float64, set x=2 and y=3 with requires_grad, compute w = x*y + sin(x) and
> L = w**2, call backward, and compare both grads against the hand
> derivatives. assert they agree to 1e-12

**Expect:**

```
forward:  x*y = 6.0000   sin x = 0.9093
          w   = 6.9093       L = 47.7384
autograd  dL/dx = 35.705220   dL/dy = 27.637190
by hand   dL/dx = 35.705220   dL/dy = 27.637190
```

**Assert:** `abs(x.grad.item() - hand_x) < 1e-12` and the same for `y`.

**⏱** under a second.

**Annotate:** short

float64 is load-bearing. At float32 the two agree to about 1e-7, so a 1e-12
assert would be measuring the mantissa rather than the mathematics.

The point of the cell is the decomposition, and it is worth printing rather
than reading: `x` reaches `L` by two paths, and

| path | value |
|---|---|
| through the product, `2w·y` | 41.455785 |
| through the sine, `2w·cos x` | −5.750565 |
| sum | **35.705220** |

Summing over outgoing edges is the whole bookkeeping of reverse mode. (Those
three numbers are computed here to 6 dp on purpose — see defect 1 at the end
of this file.)

---

## Cell 4 — both modes, timed

**Prompt to type:**

> for hidden sizes 2, 4, 8, 16 build a tiny `Linear-ReLU-Linear` net on 64
> random rows, turn the loss into a function of one flat parameter vector with
> `functional_call`, and time `torch.func.vjp` once against `torch.func.jvp` P
> times, one basis vector each. check the two gradients agree, then print the
> ratio

**Expect:** four rows, P = 51, 99, 195, 387. Reverse mode roughly **constant**
near 0.3–0.4 ms; forward mode roughly **linear** in P, about 10 ms at P=51 and
65 ms at P=387; the ratio column rising monotonically, measured here 24× → 46×
→ 81× → 197×.

**⏱** about 1 second in total — but see the annotation, because what it prints
in a fresh kernel is not what you just read.

**Assert:** `assert torch.allclose(g_rev, g_fwd, atol=1e-5)` — before any
timing is compared. A speed comparison between a right answer and a wrong one
is not a comparison.

**Annotate:** full

* **Left open** — that a timing loop measures whatever the first call has to
  set up. Nothing in the prompt asks for a warm-up, and nothing in the printed
  table says which row paid for one.
* **The usual student version** — this exact cell, with no warm-up call before
  the loop, which is what the notebook ships. Measured in three separate fresh
  processes, the **first row** came back at 408 ms, 828 ms and 884 ms of
  "reverse mode" against 36 ms, 63 ms and 49 ms for all 51 forward passes
  together — reverse mode looking 4× to 15× *slower*, and the ratio column
  printing `0x` because the format string is `{t_fwd/t_rev:6.0f}`. The
  first `torch.func.vjp` call in a process pays a one-time initialisation of
  the functorch machinery, and it lands entirely on row one.
* **How you would catch it** — one throwaway `vjp` and `jvp` before the loop.
  With that added, measured twice: 0.38, 0.36, 0.40, 0.33 ms of reverse mode,
  ratios 26 → 46 → 81 → 197. The rule generalises past this cell: **the first
  timed iteration of anything is a measurement of the import.** Time it twice
  and keep the second.

---

## Cell 5 — the real parameter count, in that ratio

**Prompt to type:**

> same thing for the actual 784-300-100-10 network on 128 rows, but only
> measure one forward direction and multiply — print the projected total in
> minutes and label it as projected, don't run it

**Expect:** `266,610 parameters`, reverse mode ≈ 1.6 ms measured, one forward
direction ≈ 1.4 ms measured, projected total **about 6 minutes** (1.4 ms ×
266,610 / 60 = 6.1 min). The word "projected" must appear in the printed line.

**Assert:** none.

**Annotate:** short

Two measured per-pass costs and one multiplication the reader can check is the
honest form of a claim you cannot afford to run. A reader cannot tell from a
printed float which of your numbers came from a clock, so the float has to say.

---

## Cell 6 — the scikit-learn control

**Prompt to type:**

> fit `MLPClassifier` with hidden layers (300, 100), adam, lr 1e-3, batch 128,
> max_iter 10, seed 42 on the 12,000 fit rows, suppress the convergence
> warning, time it, and score it on the validation set

**Expect:** measured here `7.7 s for 10 epochs, validation accuracy 0.8728`.

**⏱** **7.7 s measured on this 16-core laptop CPU.** The notebook's markdown
says "about 30 seconds", which is a free-Colab-CPU figure that the notebook
does not label as one. Budget for anything from 8 s to a minute and do not be
alarmed either way — but see defect 12.

**Assert:** none here. The comparison assert lives in cell 9.

**Annotate:** short

Time it *here*, on this machine, in this run. The speed-up printed in cell 9 is
a ratio, and a ratio whose halves came from different boxes measures the boxes.

---

## Cell 7 — tensors, and where they live

**Prompt to type:**

> make a 2×2 float tensor, print its dtype, device and requires_grad, move it
> to `device` and show the result is a new tensor, do a matmul, convert back to
> numpy, and then deliberately multiply two 3×4 tensors to show the error

**Expect:** `torch.float32 cpu False`; matmul `[7.0, 10.0, 15.0, 22.0]`; and
`shape error: mat1 and mat2 shapes cannot be multiplied (3x4 and 3x4)...`

**Assert:** none — the `try/except RuntimeError` is the check.

**Annotate:** short

`.to(device)` returns a new tensor. `b.to(device)` on a line by itself, return
value discarded, is a silent no-op and one of the quietest bugs in PyTorch.
Note also that `.numpy()` on a CPU tensor shares memory: it is a view, not a
copy, and writing through one writes through the other.

---

## Cell 8 — what `requires_grad` actually does

**Prompt to type:**

> build a small chain p = u*v, q = p + sin(u), r = q**2 with u and v requiring
> grad, and print each intermediate's `grad_fn` and the parents of r's node.
> then show what `torch.no_grad()` does to the same expression

**Expect:**

```
p = u*v          grad_fn = <MulBackward0 object at 0x...>
q = p + sin u    grad_fn = <AddBackward0 object at 0x...>
r = q**2         grad_fn = <PowBackward0 object at 0x...>

the parents of r's node: ((<AddBackward0 object at 0x...>, 0),)

   inside no_grad: None
```

**Assert:** none.

**Annotate:** short

`requires_grad=True` computes no derivative. It records operations as the
forward pass runs, and `backward()` walks the record. The `None` under
`no_grad()` is the whole reason evaluation code is wrapped in it: no record, no
memory. Everything is differentiable; the flag decides whether anyone wrote
down how you got here.

---

## Cell 9 — the training loop

**Prompt to type:**

> write a `train()` function: build the net fresh inside it, Adam, cross
> entropy, shuffle with a seeded generator each epoch, and the five lines
> zero_grad / forward / loss / backward / step inside the batch loop. track
> epoch loss and validation accuracy. also write an `accuracy()` helper that
> counts hits over the whole set. run it for 10 epochs and print the wall clock
> beside scikit-learn's

**Expect:** `266,610 parameters`; validation accuracy 0.8636 after 10 epochs
(the curve measured here: 0.7696, 0.8170, 0.8406, 0.8496, 0.8584, 0.8532,
0.8568, 0.8624, 0.8680, 0.8636 — note it is not monotone, and the last epoch is
not the best); `1.3 s` against scikit-learn's `7.7 s`, a printed speed-up of
5.9×.

**⏱** **1.3 s measured on this laptop CPU** (940 steps at ~1.4 ms). The
notebook carries no ⏱ marker on this cell although §8 declares 40 s for two
runs of the same function. Do not plan around 20 s until you have seen it on
your own runtime.

**Assert:**

```python
assert abs(hist["val_acc"][-1] - sk_val) < 0.05, \
   "the rebuild should reproduce the model, not replace it"
```

Measured gap here: |0.8636 − 0.8728| = **0.0092**, comfortably inside.

**Annotate:** short

Three properties of this function matter later and none is optional:
it constructs `net` and `opt` **inside** itself, so re-running the cell
retrains rather than continuing training; `zero_grad` is a parameter, which is
what cell 10 flips; and `dropout` is a parameter, which is what cell 12 flips.
The accuracies agreeing is the *result*. The rebuild is not an improvement —
same architecture, same optimiser, same epochs. What you bought is the loop,
and cells 10 to 14 are what you can now get wrong with it.

---

## Cell 10 — train it again with one line deleted

Run this one before reading anything below it. **Write the two final
accuracies on paper**, then read on.

**Prompt to type:**

> run the same training twice, once as normal and once with `zero_grad=False`,
> and plot the loss curves and the validation accuracy curves side by side

**Expect:** two panels. The healthy loss falls 0.93 → 0.31. The other one does
not fall: measured 2.05, 1.63, 1.84, 1.73, 1.68, 1.72, 2.02, 1.67, 1.77, 2.09
— it wanders. Validation accuracy: **0.8636 with, 0.3036 without**, printed as
a cost of **56.00 accuracy points**. No exception, no warning, no NaN, no
message of any kind.

**⏱** **2.2 s measured here** for both runs (the notebook says 40 s).

**Assert:** none — and that is the lesson. There is nothing to assert against,
because nothing went wrong in any sense PyTorch can detect.

**Annotate:** full

* **Left open** — that `backward()` **accumulates into `p.grad`; it does not
  overwrite**. Deleting the clearing line does not mean "no gradient"; it means
  each step uses the sum of every gradient computed since the model was built.
  By the end of epoch 10 that is a sum over 940 batches.
* **The usual student version** — this exact omission, and it is the single
  most common PyTorch bug there is. It survives because Adam's step is bounded
  by the learning rate however large the gradient is: the direction is wrong,
  the *scale* is not, so nothing overflows and nothing prints. Measured at
  three seeds: 42 → 0.3036, 0 → 0.3018, 7 → 0.1872, against 0.86–0.87 for the
  healthy run each time. The model just trains badly, and you look for the
  cause everywhere except the line that is not there.
* **How you would catch it** — predict the curve before you run it, then check
  the prediction. If you cannot predict it you have not understood the
  accumulation, and reading the shape off the plot afterwards will not teach it
  to you. In real code: assert on the loss, not on your memory of the loop —
  a run whose first-epoch loss does not beat `ln 10 = 2.303` on a 10-class
  problem has not started learning.

A note on the plot. Ask for a linear y-axis, not a log one, and look at it.
Measured across seeds 42, 0 and 7, every loss value in both curves lies between
0.31 and 2.34 — a span of 7.6×, which a linear axis shows perfectly. The
notebook's prompt box asks for `set_yscale("log")` on the grounds that the
broken loss "goes somewhere a linear axis cannot show", and that is not what
the data does. Making a defect look dramatic is the opposite of the skill being
taught here: this defect's whole character is that it *looks fine*.

---

## Cell 11 — watch a gradient accumulate, in four lines

**Prompt to type:**

> tiny probe: `nn.Linear(4,1)`, MSE loss, 8 random rows. call backward three
> times without zeroing and print the gradient norm each time. then call
> `zero_grad()` and print the gradient. then show what `set_to_none=False`
> does instead

**Expect:**

```
after 1 backward call(s): |grad| = 4.2099
after 2 backward call(s): |grad| = 8.4199
after 3 backward call(s): |grad| = 12.6298
after zero_grad():          grad is None
after zero_grad(set_to_none=False): |grad| = 0.0000
```

Exactly 1×, 2×, 3× the same number — that is what "accumulates" means, printed.

**Assert:** none, though `assert abs(g2 - 2*g1) < 1e-5` on the first two norms
is one line and makes the claim checkable rather than eyeballable.

**Annotate:** full

* **Left open** — the second, entirely separate way this bug bites.
  `zero_grad()` defaults to **`set_to_none=True`**, so after it `.grad` is
  `None` — not a tensor of zeros. The default is a memory and speed choice, and
  it changes the *type* of the attribute.
* **The usual student version** — gradient-norm logging that reads
  `p.grad.norm()`. It works for the whole of training and then raises
  `AttributeError: 'NoneType' object has no attribute 'norm'` the first time it
  runs after a zeroing — most often at epoch end, or in an evaluation hook,
  i.e. twenty minutes into a run. The same default bites a second way: a
  hand-written optimiser that does `p -= lr * p.grad` for every parameter
  crashes on any parameter that did not receive a gradient this step.
* **How you would catch it** — `if p.grad is not None` before you touch it, or
  `zero_grad(set_to_none=False)` while you are debugging. And note the ordering
  that makes the last line of this cell work at all: `set_to_none=False` on a
  `None` gradient **leaves it `None`** (verified — it only zeroes gradients
  that already exist), so you must run a `backward()` between the two calls.
  If you delete that middle `backward()`, the final print raises the very
  `AttributeError` this bullet is about.

---

## Cell 12 — evaluate a network that was trained with dropout

Again: run it, write the numbers down, then read the annotation.

**Prompt to type:**

> train the same net with dropout 0.2, then measure test accuracy with
> `model.eval()`, and separately take ten readings with the model left in
> `train()` mode. print the mean, the min and the max of the ten

**Expect:**

```
model.eval()   0.8555
model.train()  0.8476  (min 0.8444, max 0.8496)
```

— a cost of 0.79 accuracy points, and a spread of **0.52 points across ten
identical calls on identical data**.

**⏱** about 1.5 s measured here (the notebook says 25 s).

**Assert:** none. Add one if you want the cell to make its own point:
`assert len(set(readings)) > 1` — ten calls to a deterministic function of
fixed weights and fixed data cannot return more than one distinct value.

**Annotate:** full

* **Left open** — that dropout zeroes a random fraction of each hidden layer
  **in training mode only**, and that `nn.Module`'s mode is a flag on the
  module, not a property of the call. `net.eval()` set once at the top of a
  notebook does not stick: every `train()` call flips it back, and the training
  loop in cell 9 calls `net.train()` at the top of every epoch.
* **The usual student version** — evaluating inside the epoch loop right after
  the batches, with no `eval()` between. It is the natural place to put it and
  it is wrong. `torch.no_grad()` does *not* help: it turns off the graph, not
  dropout, and the ten readings above were all taken under `no_grad`.
  `BatchNorm` is the same bug with worse consequences — in training mode it
  normalises by the *current batch*, so your test predictions depend on which
  other images you happened to batch them with.
* **How you would catch it** — **the spread, not the mean.** Measured here the
  cost is 0.79 points and the wobble is 0.52 points, so the cost is only 1.5×
  the noise in the thing measuring it — as a headline it would be inside its
  own error bar. The finding that is not inside anything is that ten calls gave
  ten different answers. Ask any metric twice. If it moves, ask which layer is
  still in training mode.

---

## Cell 13 — `Dataset` and `DataLoader`

**Prompt to type:**

> wrap the fit tensors in a `TensorDataset` and a `DataLoader` with batch 128,
> shuffle on and a seeded generator. print how many batches, the shape of the
> first one, and how short the last one is. assert the loader yields every row

**Expect:** `94 batches of 128`; first batch `(128, 784)`; and the arithmetic
`12000 = 93 x 128 + 96`, i.e. the last batch holds **96** rows, not 128.

**Assert:**

```python
assert sum(len(b) for b, _ in train_dl) == len(Xf), "the loader dropped rows"
```

Measured: 12,000 — nothing dropped.

**Annotate:** short

`drop_last=False` is the default, so nothing is discarded and the last batch is
short. The alternative default is worse than it looks: `drop_last=True` silently
throws away up to 127 training rows *every epoch*, and because the shuffle is
re-drawn each epoch it is a different 96 rows each time — so it is not even a
consistent subset.

Do not skip past that short batch. It is the entire subject of the next cell.

---

## Cell 14 — accuracy, two ways, in one pass

**Prompt to type:**

> score the test set in batches of 384 and return both numbers: the accuracy
> counted over the whole set, and the plain mean of the per-batch accuracies.
> print how many batches there were and how big the last one is

**Expect:**

```
27 batches, the last containing 16 images
accuracy over the set    0.85330
mean of batch accuracies 0.85185
difference -0.1448 accuracy points
```

**Assert:** none — both numbers are printed, and the disagreement is the
finding. If you want it mechanical, `assert abs(mean_batches - over_set) < 1e-9`
fails, which is the honest way to state that these are not the same statistic.

**Annotate:** full

* **Left open** — how small "small" is, and why. 10,000 images in batches of
  384 is 26 full batches and a last batch of **16**. A plain mean gives that
  16-image batch weight 1/27 = 3.70% when it deserves 16/10,000 = 0.16% — an
  over-weighting of 23×. The reason the printed difference is only 0.14 points
  is not that the batch is nearly full; it is that its accuracy happened to
  land near the whole-set figure. The decomposition is exact:
  (0.0370 − 0.0016) × (0.8125 − 0.8533) = −0.00145, i.e. the −0.1448 points
  printed.
* **The usual student version** — `np.mean(batch_accs)`, or the equivalent
  `running_acc / n_batches` in the epoch loop. It is the obvious thing to write
  and it is correct only when every batch is the same size, which — with
  `drop_last=False`, the default you just saw — is never.
* **How you would catch it** — weight by batch size, or count hits over the
  set. And check the sensitivity rather than trusting that it is small: re-run
  the same helper at `batch=3000` (4 batches, last one 1,000 rows, weight 1/4
  against a true 1/10). Measured, the gap moves from −0.1448 to **+0.1950**
  points — larger, but only by 1.35×, and with the **opposite sign**. A
  difference whose sign depends on your batch size is not a small error you can
  reason about; it is an error you cannot see from the number.

---

## Cell 15 — the same model as a subclass

**Prompt to type:**

> rewrite the same 784-300-100-10 network as an `nn.Module` subclass with a
> `forward`, using `nn.ModuleList` for the hidden layers, and assert the
> parameter count matches the Sequential version

**Expect:** the module repr, then `266,610 parameters, registered
automatically`.

**Assert:** `assert sum(p.numel() for p in m.parameters()) == n_params` —
verified equal: 266,610 = 266,610.

**Annotate:** full

* **Left open** — what subclassing actually buys, which is not style.
  `nn.Sequential` runs out of road the moment the forward pass is not a
  straight line: two inputs, a skip connection, a branch. Everything else —
  parameter registration, `.to(device)`, `state_dict()` — comes free either way.
* **The usual student version** — `self.blocks = [nn.Linear(a, b) for ...]`,
  a plain Python list. This is documented `nn.Module` behaviour, not an
  accident: only attributes that are `Parameter`, `Module` or one of the
  container types get registered. A plain list is registered as nothing, so
  those layers are absent from `.parameters()` — the optimiser never sees them,
  `.to(device)` never moves them (so you then get a *device* error, on a line
  far from the cause), and `state_dict()` never saves them. The model runs. It
  trains the head only.
* **How you would catch it** — this assert, against a known-good count. An
  unregistered layer is invisible in the loss curve and in the repr, and
  visible immediately in `sum(p.numel() for p in m.parameters())`. The count is
  also worth doing on paper first: 784×300 + 300 + 300×100 + 100 + 100×10 + 10
  = 235,200 + 300 + 30,000 + 100 + 1,000 + 10 = **266,610**.

---

## Cell 16 — the optional dependency

**Prompt to type:**

> import optuna in a try/except and print the version, and if it is missing
> print the pip command and one sentence saying what it would have done

**Expect:** `optuna 4.6.0` here; on a bare runtime, the fallback branch and
`HAVE_OPTUNA = False`.

**Assert:** none.

**Annotate:** short

A bare `import optuna` at cell one means a runtime without it fails before
anything else can run. If a cell can fail for an environmental reason, catch it
and print the remedy — a traceback is not instructions.

---

## Cell 17 — a search with a sampler that learns

**Prompt to type:**

> if optuna is available, run 8 trials over learning rate 1e-4 to 1e-2 with
> log=True and dropout 0 to 0.5 in steps of 0.1, training 4 epochs each,
> maximising validation accuracy. seed the sampler. print the best config
> beside the 10-epoch baseline

**Expect:** `best <value> at {'lr': ..., 'dropout': ...}` against the baseline
0.8636, plus the printed caveat that the trials ran 4 epochs and the baseline
ran 10.

**⏱** **projected ~4–5 s on this laptop CPU**: 8 trials × 4 epochs = 32 epochs
against the 10 epochs measured at 1.3 s, plus sampler overhead. Not measured
directly — the projection is stated so you can check it. The notebook says
90 s, which is a Colab figure it does not label as one.

**Assert:** none.

**Annotate:** short

`log=True` matters and is easy to leave off: on a uniform sampler over
[1e-4, 1e-2], half the proposals land above 5e-3 and almost none below 1e-3,
which is the half of the range you care about. And seed the sampler, or you
cannot tell an improvement from a resample. The comparison as printed is **not**
like-for-like — 4 epochs against 10, and the best trial's score was measured on
the very validation set that selected it. Both facts inflate it.

---

## Cell 18 — save it

**Prompt to type:**

> save the trained net's `state_dict` to `checkpoints/sorter.pt`, print the
> file size, load it back into a freshly built network and assert the two give
> exactly the same test accuracy

**Expect:** `checkpoints/sorter.pt   1,044 KB` (266,610 float32 parameters plus
the pickle header), then two identical accuracies — 0.8533 and 0.8533 here.

**Assert:** `assert a1 == a2` — **exact** equality, not `allclose`. Loading
weights is deterministic; there is nothing here to be approximate about.

**Annotate:** short

`state_dict()` saves tensors. `torch.save(net, path)` pickles the class
reference with them, and the file stops loading the day the module is renamed
or opened on a machine where your source is not importable. Note also
`weights_only=True` on the load: unpickling executes code, and since torch 2.6
the *effective* default is `True` precisely because of it. Checked on torch
2.13.0 — the signature still reads `weights_only=None`, which torch resolves to
`True` — so passing it explicitly is what documents the reason.

---

## Cell 19 — re-measure

**Prompt to type:**

> print the majority-class baseline, both validation accuracies with their wall
> clocks, and the test accuracy, and label which number is the one we report

**Expect:**

```
baseline (majority class)   0.1000
Scikit-Learn, validation    0.8728   in 7.7 s
PyTorch, validation         0.8636   in 1.3 s on cpu
PyTorch, TEST               0.8533   <- the number you report
```

**Assert:** none.

**Annotate:** short

The baseline is 0.1000 because the Fashion-MNIST test set holds exactly 1,000
images of each of its ten classes (counted: `np.bincount` gives ten 1,000s), so
every class is the majority class and guessing any one of them scores 0.1. Three of these four numbers
were used for tuning and one was not, and printing them in a single column
invites reading them as comparable. The test set has now been touched once.
Every adjustment made after reading it is selection on the test set, whatever
you call it.

---

## Re-run orders, for the things worth checking

Each of these needs cells run in a specific order from the state you are in.

1. **Is the timing table in cell 4 real?** Add a throwaway `vjp` + `jvp` call
   above the `for h in ...` loop, then re-run **cell 4 alone**. Nothing else
   depends on it. Cold, the first row prints `0x`; warmed, it prints ~26×.
2. **Does the missing `zero_grad` depend on the seed?** Change nothing;
   re-run **cell 10** with `train(zero_grad=False, seed=0)` and `seed=7`
   substituted in place. `train()` rebuilds the network and the optimiser
   internally, so cell 10 is idempotent and cell 9 does **not** need re-running.
   Measured: 0.3036, 0.3018, 0.1872.
3. **Does the per-batch averaging error depend on the batch size?** Re-run
   **cell 14** only, with `batch=3000`. It reads `net` from cell 9, which is
   unchanged. Measured: −0.1448 points at 384, +0.1950 at 3,000.
4. **Does dropout's cost survive a bigger rate?** Re-run **cell 12** with
   `train(dropout=0.5)`. Cell 12 rebinds `net_d` and nothing downstream reads
   it, so this is safe. Do not re-run cell 9 in between — it would rebind `net`
   and change cells 14, 18 and 19 underneath you.

---

## Defects found in the current notebook

Everything below was checked against `notebooks/lecture-12.ipynb` and
`tools/notebooks/lecture_12.py` with `python3`. Each item says how it was
checked. Nothing here is asserted from reading alone unless it says so.

**Context first:** no notebook in this course stores cell outputs except
lecture 19 (checked: `lecture-12.ipynb` has 19 code cells, 0 with outputs;
across all 24 notebooks only lecture 19 has any). So GUIDELINES §1.2 — "every
figure quoted in markdown must appear in a stored cell output" — is not
satisfiable for this file as shipped, and the §9 machine checks that read
stored execution times cannot run on it. That is a course-wide property, not a
lecture-12 defect, but it means every number below had to be re-derived by
running the code rather than by reading the file.

### Verified by execution

1. **§1.1, §1.2 — the arithmetic in the markdown after the autograd cell is
   wrong in every digit and does not add up.** The cell (index 10) prints
   `$$41.4557 - 5.7505 = 35.7053$$`. Computed in float64: the two path
   contributions are **41.455785** and **−5.750565**, and the total is
   **35.705220**. Round-half-up to four places gives 41.4558, 5.7506, 35.7052 —
   so all three are wrong, two by truncation rather than rounding (§1.2
   forbids exactly that), and the third by rounding in the wrong direction.
   Worse, the equation is internally inconsistent as printed: 41.4557 − 5.7505
   = 35.7052, not 35.7053. And the code cell immediately above prints
   `dL/dx = {x.grad.item():.6f}` = **35.705220**, so the markdown contradicts
   the output four lines above it. Neither 41.4557 nor 5.7505 is printed
   anywhere in the notebook — the reader cannot reconcile them without deriving
   `2w·y` and `2w·cos x` themselves, and the sentence does not state that
   arithmetic.

2. **§1.1 — "the last batch is not that short" is false, and the notebook's own
   output says so.** Section 11's markdown explains the small difference by
   saying the last batch "is not that short". Checked: `acc_two_ways` defaults
   to `batch=384` on 10,000 test images, giving **27 batches with 16 images in
   the last** — the cell prints that number itself, one line above. 16 of 384
   is 4.2% of a full batch. The real explanation is measurable and different:
   the plain mean over-weights that batch by 23× (3.70% against 0.16%), and the
   difference stays small only because the last batch's accuracy landed near
   the set's (0.8125 against 0.8533). The exact decomposition
   (0.0370 − 0.0016)(0.8125 − 0.8533) = −0.00145 reproduces the printed
   −0.1448 points. So the notebook explains a real effect by the wrong
   mechanism, in the section whose subject is not being able to tell from the
   number whether it mattered.

3. **§1.3 / §2.1 — the follow-up sensitivity claim is directionally
   misleading.** The same markdown says "change either and it grows; change the
   batch size to 3,000 ... and the last batch carries a quarter of the weight
   instead of a tenth". The weights check out (4 batches, last 1,000, 1/4
   against 1/10). Measured, the gap goes from **−0.1448** points at batch 384
   to **+0.1950** at batch 3,000 — it grows by only 1.35× and **changes sign**.
   A reader who follows the instruction gets a positive number where the prose
   led them to expect a bigger negative one.

4. **The first row of the cost-of-autodiff table contradicts the lesson it is
   printed to support, in every fresh kernel.** Ran cell 13's code verbatim in
   three separate cold processes. Row one (P=51) came back:
   `reverse 407.80 ms / forward 35.8 ms`, `reverse 1008 ms / forward 215 ms`,
   `reverse 828 ms / forward 63 ms` — reverse mode 4× to 15× **slower** than
   all 51 forward passes together, and the ratio column printing `0x` because
   the format is `{t_fwd/t_rev:6.0f}`. The markdown directly below then states
   "Reverse mode is flat in $P$; forward mode is linear in it." Diagnosed: the
   first `torch.func.vjp` call in a process pays a one-time initialisation.
   With one throwaway `vjp`+`jvp` before the loop, run twice, reverse mode is
   flat at 0.29–0.41 ms across all four sizes and the ratio column reads
   26 → 46 → 81 → 197 and 24 → 52 → 109 → 205. This is the notebook's central
   quantitative claim and its own first data row refutes it.

5. **The stated reason for the log-scaled loss axis is not what the data
   does.** Section 8's prompt box requires `set_yscale("log")` because "without
   zero_grad the loss goes somewhere a linear axis cannot show alongside the
   healthy run". Ran both trainings at seeds 42, 0 and 7: every value in both
   curves lies in **[0.31, 2.34]**, a total span of 7.0–7.6×. A linear axis
   shows both curves comfortably. The loss cannot run away because Adam's step
   is bounded by the learning rate however large the accumulated gradient is —
   which is *why* this bug is silent, and is the sentence the notebook does not
   say.

6. **§3.3 — the header refers to a marker that appears nowhere.** The header
   says "Cells marked **⚠ read before running** contain a defect on purpose".
   Searched: the string "read before running" occurs exactly **once** in the
   whole notebook — inside that sentence. No cell carries it. The ⚠ symbol
   appears seven times: once in that header sentence, and on three section
   headings and three prompt labels (§8, §9, §11).

7. **§3.3 — the header miscounts its own defects.** "…two of them are the
   defects this lecture exists to teach". Three sections are marked ⚠ (8, 9
   and 11), and section 8 contains two distinct defects (the accumulation and
   the `set_to_none=True` type change), so the honest count is three sections
   and four mechanisms. A reader who stops looking after two stops one section
   early.

8. **§3.3 — the forward reference to Lecture 15 points at a lecture that does
   the opposite.** Section 10 says indexing a tensor by hand "stops working the
   moment [the data does not fit in memory] — which is Lecture 15", and cell
   40's box repeats it as "three lectures from here". Checked `lecture-15.ipynb`:
   it is "Visual inspection" (Géron ch. 12), its sections are setup / data /
   look at it / normalisation / … , and its one mention of `DataLoader` is a
   prompt bullet saying that a DataLoader with a decode transform "is correct
   for data that does not fit in memory and **is pure overhead here**" — it
   decodes once and keeps everything in memory. Grepped all 24 notebooks:
   `num_workers` appears in none, `ImageFolder` in none. The course never
   delivers the topic this cross-reference promises.

9. **§8.1 — the trap is announced five times before the reader reaches it.**
   The missing `zero_grad` is flagged in the header, in the section-8 heading
   (`## 8 · ⚠ The missing zero_grad()`), in the paragraph above the cell, in
   the prompt label (`⏱ 40 s — ⚠ the missing zero_grad()`), and in the
   Left-open bullet — all before the cell runs. The `model.eval()` defect is
   announced four times the same way. This is the exact pattern §8.1 was
   written about, and the notebook's preferred shape ("let the defective cell
   run unannounced, have the reader write the number down") is available here
   at no cost, because `train(zero_grad=False)` is a parameter flip.

10. **§6.1 — the annotation budget is exceeded by a factor of two and a half.**
   Counted programmatically: **19 code cells, 19 prompt boxes, 19 carrying the
   full three-bullet "Watch this prompt" annotation**. The budget is five to
   eight, never more than ten. For scale, the notebook the three readers gave
   up on has 20.

11. **§8.3 — almost nothing is marked examinable.** The string "examinable"
   appears three times: twice about Optuna (section 13) and once about device
   selection. Fourteen of the sixteen sections carry no label, including all
   three ⚠ sections and the whole of the autodiff thread, which is the
   examinable mathematics of the lecture.

12. **§7.1 — the ⏱ markers do not say which machine they describe, and none of
   them matched any machine available here.** Measured on a 16-core
   Apple-silicon laptop, CPU, under load: the scikit-learn control **7.7 s**
   against a marked 30 s; the two `zero_grad` runs **2.2 s** against a marked
   40 s; the dropout section **~1.5 s** against a marked 25 s. Separately,
   section 7's training cell — 10 epochs, measured **1.3 s** here — carries
   **no ⏱ marker at all**, even though section 8 declares 40 s for two runs of
   that same function, i.e. about 20 s each. So the file simultaneously
   over-states three timings and omits the marker on the fourth. Whether the
   markers are right for a free Colab CPU runtime I could not test — there is
   no CUDA device on this machine — which is the point: the reader cannot tell
   what hardware a bare "⏱ about 30 seconds" refers to.

13. **§4.1 — `net` means two different networks, and the toy one comes first.**
   Cell 13's loop binds the module-level name `net` to a 20→h→3 test network
   (`for h in (2,4,8,16): net = nn.Sequential(...)`); cell 29 rebinds `net` to
   the trained 784-300-100-10 model, which cells 44, 56 and 59 then score,
   save and report. Same class, different object, different meaning, sixteen
   cells apart, unremarked — and §4.1's stated remedy is precisely "loop
   variables in throwaway tests get throwaway names". The same pattern applies
   to `f`, `theta`, `P` and `t_rev` (bound in cell 13's loop, rebound at module
   level in cell 16) and to `xb, yb` (cell 35: random `(8,4)` and `(8,1)`
   tensors; cell 41: a `(128, 784)` image batch and its labels). None of these
   breaks the run — checked, restart-and-run-all order is sound — but a reader
   who inspects `net` between cells 13 and 29 gets the wrong network.

14. **§2.4 — the `model.eval()` cost is quoted ahead of the noise that swamps
   it.** Measured (CPU, seed 42, dropout 0.2): eval 0.8555, ten training-mode
   readings mean 0.8476 with min 0.8444 and max 0.8496 — a cost of **0.79
   points** against a spread of **0.52 points** across ten identical calls. The
   cost is 1.5× the run-to-run variation of the instrument measuring it. The
   notebook prints the cost line first and the spread second, then says the
   spread is the tell. It is; the ordering invites the reader to write down
   the 0.79 anyway.

15. **A line that teaches nothing and cannot be explained.** Cell 38 line 3:
   `_, _, Xte_, yte_ = None, None, Xt, yt`. It binds `_` twice to `None` and
   creates two aliases used four times in the same cell and never again. The
   cell works identically with `Xt, yt` written directly. Verified by running
   the cell body both ways.

### Checked and found clean

- **§5.1, §5.2 — markdown rendering.** Scanned every markdown cell: no line
  indented four or more spaces outside a fence, no fence markers at all in the
  markdown (there are no fenced blocks in this notebook's prose), so §3.1 —
  quoted code that exists in no cell — cannot be violated here and is not.
- **§4.2 — training cells are idempotent.** `train()` constructs `net` and
  `opt` inside itself; re-running any training cell retrains from scratch
  rather than continuing. Verified by running `train()` twice and getting
  identical validation curves.
- **The split matches Lecture 11's.** Both draw `np.random.default_rng(42)`,
  permute 60,000, take `order[:5_000]` as validation and `[:12_000]` of the
  rest as the fit set. Compared the source lines directly; they agree.
- **The assert in section 7 holds with room to spare.** Measured |0.8636 −
  0.8728| = 0.0092 against a tolerance of 0.05.
- **Section 12's parameter-count assert holds.** `Sorter()` and the Sequential
  version both give **266,610**. Ran both.
- **Section 14's exact-equality assert holds.** Saving and reloading the
  `state_dict` reproduces the accuracy bit for bit. Serialised size 1,044 KB.
- **Section 10's "the loader dropped rows" assert holds.** 94 batches, 12,000
  rows, last batch 96.
- **The probe cell's ordering is correct and subtle.** `zero_grad(set_to_none=
  False)` on a `None` gradient leaves it `None` — verified — so the
  `backward()` between the two calls is load-bearing, not decorative. The cell
  prints `0.0000` as intended.
- **`optuna` imports here** (4.6.0), so the fallback branch of section 13 was
  not exercised on this machine.

### Not checked

- **Anything GPU-dependent.** No CUDA device was available. Every wall clock in
  this file is a CPU measurement on one laptop, and the notebook's Colab
  figures could be neither confirmed nor refuted.
- **Whether the section-7 assert survives on a T4.** The scikit-learn side is
  CPU-bound either way, so the gap should be stable, but that is reasoning, not
  a measurement.
- **The optuna search cell's real duration.** The ~4–5 s above is 32 epochs
  projected from a measured 10-epoch run, not a timed execution.
- **Whether the machine's load (average 191–260 on 16 cores throughout)
  inflated the timings.** Per-step figures were taken as the minimum over 300
  steps to blunt it, but every absolute second quoted here should be read as an
  upper bound.
