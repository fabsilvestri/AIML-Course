#!/usr/bin/env python3
"""
Lecture 11 — Training deep networks. CIFAR-10, Géron Chapter 11.

Derivation: variance propagation through layers.

Merges the old lectures 13 and 14, which built a twenty-layer network that
would not train and then repaired it across two sessions. Here it is one
lecture: measure the failure, derive why, repair it, and ablate the repairs.

Exports build() -> list[nbformat cell]. Self-contained. Runs on CPU: CIFAR-10 is
subsampled and the stack is narrow, because what has to survive the shrink is
the SHAPE of the gradient curve — a straight line on a log axis — and not any
particular accuracy.
"""

from __future__ import annotations

import nbformat as nbf
from _prompt import prompt                                # noqa: E402


def md(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(text.strip("\n"))


def code(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(text.strip("\n"))


HEADER = """
# Training deep networks

**Lecture 11** · Géron, Chapter 11

Applications of Machine Learning — BSc Mathematics of Artificial Intelligence

---

**How to use this notebook.** Read before you run. You are not expected to type
the code; you are expected to be able to say what every line does and what
would break if it changed.

Every code cell is preceded by the specification that would produce it — input,
output, constraint, check. Read the box, work out what the check should say,
*then* run the cell.

The notebook deliberately builds a network that **does not train**, measures
why, derives the reason, and then repairs it. The broken network is labelled as
such throughout; nothing here is wrong unannounced.

Runs on CPU: 10,000 of the 50,000 images, a narrow stack, and a short run, so a
free Colab session finishes in minutes. What survives the shrink is the *shape*
— the gradient falling by a constant factor per layer — and every comparison
below uses the same subset, the same seed and the same number of epochs, so the
rows can be read against one another.
"""



def build() -> list:
    return [
        md(HEADER),

        md("## 1 · Setup"),
        prompt(
            label="setup",
            input="nothing",
            output="versions, seeds, and the device",
            constraint="pick the device by asking, and say what to do if the answer is cpu",
            check="everything below still runs on cpu; it is slower. Say so, or a student with no GPU will read the wall clocks as a bug.",
            **{"try": "force device = 'cpu' even where an accelerator is "
                      "available, then re-run Section 6. Every number below "
                      "is identical and the wall clock roughly triples. Which "
                      "printouts in this notebook are properties of the "
                      "network, and which are properties of the machine?"}),
        code('''
# Not examinable, and only needed on some machines: PyTorch, numpy and
# torchvision can each end up loading their own OpenMP runtime, and with more
# than one loaded a training cell can deadlock -- no error, no output, and no
# CPU use. These have to be set BEFORE torch is imported, because they are read
# at import time and after that they do nothing.
import os
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# --- setup -------------------------------------------------------------------
# Not examinable: engineering hygiene, not machine learning. It is here because
# a device mismatch produces a confusing error twenty cells later.
import math, sys, time
import numpy as np
import torch
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt

print(f"python       {sys.version.split()[0]}")
print(f"torch        {torch.__version__}")
print(f"torchvision  {torchvision.__version__}")

RANDOM_STATE = 42
torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

if torch.cuda.is_available():
    device = "cuda"                       # NVIDIA, and Colab
elif torch.backends.mps.is_available():
    device = "mps"                        # Apple Silicon
else:
    device = "cpu"
print(f"\\ndevice       {device}")
if device == "cpu":
    print("running on CPU, which is what this notebook is sized for.")
    print("An accelerator, if present, is used and is faster; nothing needs one.")
'''),

        md("""
## 2 · The data

CIFAR-10: 60,000 colour photographs, 32 × 32 pixels, ten classes, split 50,000
train and 10,000 test by the people who built it. Every class has exactly the
same number of images — remember that, it decides the metric in section 4.

⏱ **1–3 minutes the first time** — about 170 MB over the network. Instant
afterwards, because `download=True` checks before it fetches.
"""),
        prompt(
            label="⏱ 1-3 min first time — the data",
            input="CIFAR-10",
            output="50,000 train and 10,000 test images as uint8, and the class names",
            constraint="assert the class counts are EXACTLY balanced in both splits — that fact decides the metric two sections down",
            check="assert both shapes and both balance conditions. `download=True` checks before it fetches, so re-running is instant. A cell that re-downloads 170 MB every run is a cell people stop running.",
            **{"try": "drop the first training image and its label, then re-run. "
                      "The shape assert fires before the balance one. Which of "
                      "the two would you rather have been told about, and does "
                      "the order they are written in matter?"}),
        code('''
train_ds = torchvision.datasets.CIFAR10("datasets", train=True,  download=True)
test_ds  = torchvision.datasets.CIFAR10("datasets", train=False, download=True)
CLASSES  = train_ds.classes

Xtr_u8 = train_ds.data                            # (50000, 32, 32, 3) uint8
Xte_u8 = test_ds.data
ytr = np.asarray(train_ds.targets, dtype=np.int64)
yte = np.asarray(test_ds.targets,  dtype=np.int64)

assert Xtr_u8.shape == (50_000, 32, 32, 3), f"unexpected shape {Xtr_u8.shape}"
assert Xte_u8.shape == (10_000, 32, 32, 3)
assert set(np.bincount(ytr)) == {5_000}, "the training set is not balanced"
assert set(np.bincount(yte)) == {1_000}, "the test set is not balanced"

print(f"{len(Xtr_u8):,} train + {len(Xte_u8):,} test, {len(CLASSES)} classes")
print(f"one image is {Xtr_u8.shape[1]}x{Xtr_u8.shape[2]}x{Xtr_u8.shape[3]} "
      f"= {Xtr_u8[0].size:,} numbers")
print(CLASSES)
'''),

        md("""
### Look at it

Four examples of each class. Two things to notice: the objects are not centred
or aligned the way the garments in Fashion MNIST were, and the backgrounds vary
enormously. A pixel at a fixed position means much less here.
"""),
        prompt(
            label="look at it",
            input="four examples of each class",
            output="a 4 by 10 grid titled by class",
            constraint="no `cmap` — these are colour images, and forcing a greyscale colormap on a three-channel array either errors or silently shows you one channel",
            check="look at the data with the model's assumptions in mind. We are about to flatten to 3,072 numbers, and the grid shows exactly what that throws away.",
            **{"try": "pass cmap='gray' to imshow. On a three-channel array "
                      "matplotlib either raises or shows you something that "
                      "is not the image. Which does it do here, and would you "
                      "have noticed if it were the second?"}),
        code('''
chosen = np.concatenate([np.where(ytr == c)[0][:4] for c in range(10)])
fig, axes = plt.subplots(4, 10, figsize=(13, 5.6))
for k, ax in enumerate(axes.T.ravel()):
    ax.imshow(Xtr_u8[chosen[k]])
    ax.set_xticks([]); ax.set_yticks([])
for c in range(10):
    axes[0, c].set_title(CLASSES[c], fontsize=9)
plt.tight_layout(); plt.show()
'''),

        md("""
## 3 · Split, then scale — in that order

The rule from Lecture 2 has not been repealed by anything in Part II. The
scaling statistics are computed on the **fit** subset only, and then applied to
validation and test.

We flatten each image to a vector of 3,072 numbers. That throws away the fact
that neighbouring pixels are neighbours — which is the whole subject of Lecture
15. Today it is deliberate: we are studying depth, not vision.
"""),
        prompt(
            label="split, THEN scale",
            input="the training images",
            output="standardised fit, validation and test matrices",
            constraint="compute mu and sd on the FIT subset only, then apply them to all three — the rule from Lecture 2 has not been repealed by anything in Part II",
            check="assert the fit subset is standardised, and print the validation mean and sd, which will NOT be exactly 0 and 1. The `+ 1e-7` on the standard deviation. CIFAR has no constant pixels, but a dataset with one gives a divide-by-zero that propagates NaN through every layer and shows up as a loss of nan five cells later.",
            **{"try": "compute mu and sd on all 50,000 images rather than the "
                      "fit subset. The validation mean becomes exactly zero, "
                      "the assert still passes, and you have leaked. The only "
                      "tell was the printed line saying that not-exactly-zero "
                      "is correct."}),
        code('''
N_FIT, N_VAL = 10_000, 5_000

rng   = np.random.default_rng(RANDOM_STATE)
order = rng.permutation(len(Xtr_u8))
val_idx = order[:N_VAL]
fit_idx = order[N_VAL:N_VAL + N_FIT]

assert set(val_idx).isdisjoint(fit_idx), "the split overlaps"

flat = lambda a: a.reshape(len(a), -1).astype(np.float32) / 255.0

X_fit_raw = flat(Xtr_u8[fit_idx])
mu = X_fit_raw.mean(axis=0)                 # fitted on the FIT subset only
sd = X_fit_raw.std(axis=0) + 1e-7

std   = lambda a: (flat(a) - mu) / sd
X_fit, y_fit = std(Xtr_u8[fit_idx]), ytr[fit_idx]
X_val, y_val = std(Xtr_u8[val_idx]), ytr[val_idx]
X_test, y_test = std(Xte_u8), yte

assert X_fit.shape == (N_FIT, 3072) and X_val.shape == (N_VAL, 3072)
assert X_test.shape == (10_000, 3072)
assert abs(X_fit.mean()) < 1e-4 and abs(X_fit.std() - 1) < 1e-2, \\
    "the fit subset should now be standardised"
print(f"fit {len(X_fit):,}   val {len(X_val):,}   test {len(X_test):,}")
print(f"fit mean {X_fit.mean():+.2e}   sd {X_fit.std():.4f}")
print(f"val mean {X_val.mean():+.2e}   sd {X_val.std():.4f}   "
      f"<- not exactly 0 and 1, and that is correct")
'''),
        md("""
The validation mean is not exactly zero. It should not be: those statistics
came from a different set of images. A pipeline in which every split has mean
exactly zero is a pipeline that fitted the scaler on everything.
"""),

        md("""
## 4 · The metric, and the number to beat

Ten classes, exactly balanced, and no class is more expensive to get wrong than
another. Under those three conditions — and only under them — plain accuracy is
defensible. Lecture 4 was about what happens when they do not hold.

So compute the trivial baseline **before** committing to anything.
"""),
        prompt(
            label="the metric, and the number to beat",
            input="the test labels",
            output="the majority-class accuracy and the loss of a model that has learned nothing",
            constraint="print BOTH anchors — the accuracy baseline and ln(10), because the loss curve is what you will actually be staring at",
            check="a loss anchor is as important as an accuracy anchor and almost nobody prints one. ln(k) for k balanced classes, one line, before anything trains.",
            **{"try": "compute ln(k) for k = 2 and for k = 100. A loss of 2.3 "
                      "is chance here and a catastrophe in a binary problem. "
                      "A loss value with no class count beside it cannot be "
                      "read at all."}),
        code('''
counts = np.bincount(y_test, minlength=10)
baseline = counts.max() / counts.sum()
print("test images per class:", counts.tolist())
print(f"\\nalways predict the commonest class -> accuracy {baseline:.4f}")
print(f"the loss of a model that has learned nothing: "
      f"ln(10) = {math.log(10):.4f}")
'''),

        md("""
## 5 · Build the stack

Twenty hidden layers of a hundred units, a logistic activation, and a linear
head. Note what the specification does **not** say: nothing about how the
weights start out. That is not an omission we are hiding — it is the ordinary
case, and it is the subject of the next lecture.
"""),
        prompt(
            label="build the stack",
            input="twenty hidden layers of a hundred units, logistic activation",
            output="the network, its parameter count, and where the parameters are",
            constraint="break the parameter count down by layer — the first matrix is 3,072 × 100 and the other nineteen are 100 × 100, so most of the parameters are in one place",
            check="assert there are DEPTH+1 weight matrices, one per layer plus the head. Count the weight matrices with an assert rather than trusting the loop. An off-by-one in a layer-building loop gives a network that trains and is not the one you described.",
            **{"try": "set WIDTH = 200 and read the breakdown again. Both "
                      "terms grow and not by the same factor: the first "
                      "matrix doubles and each hidden one quadruples. Which "
                      "of depth and width dominates the count here, and which "
                      "of the two does the rest of the lecture blame?"}),
        code('''
DEPTH, WIDTH, N_IN, N_OUT = 20, 100, 3072, 10

def make_net(depth=DEPTH, width=WIDTH, act=nn.Sigmoid, n_in=N_IN, n_out=N_OUT,
             dtype=None):
    # dtype is here because the variance measurements later in this notebook
    # run in float64: the ratios we are checking span many orders of magnitude,
    # and in float32 the smallest of them underflow to zero.
    layers, prev = [], n_in
    for _ in range(depth):
        layers += [nn.Linear(prev, width), act()]
        prev = width
    layers.append(nn.Linear(prev, n_out))
    net = nn.Sequential(*layers)
    return net if dtype is None else net.to(dtype)

net = make_net()
n_params = sum(p.numel() for p in net.parameters())
lins = [m for m in net if isinstance(m, nn.Linear)]

assert len(lins) == DEPTH + 1, "one weight matrix per layer, plus the head"
print(f"{len(lins)} weight matrices, {n_params:,} parameters")
print(f"  first  {N_IN} x {WIDTH} + {WIDTH} = {N_IN*WIDTH + WIDTH:,}")
print(f"  each hidden {WIDTH} x {WIDTH} + {WIDTH} = {WIDTH*WIDTH + WIDTH:,}"
      f"  (x {DEPTH-1})")
print(f"  head   {WIDTH} x {N_OUT} + {N_OUT} = {WIDTH*N_OUT + N_OUT:,}")
'''),
        md("""
### What did `nn.Linear` put in those matrices?

Nobody said. Look.
"""),
        prompt(
            label="what did nn.Linear put in there",
            input="one hidden weight matrix",
            output="its min, max, mean and standard deviation, against the documented default",
            constraint="check the sd against b/√3, the standard deviation of a uniform on (−b, b) — that identity is what turns 'it looks uniform' into a verified claim",
            check="assert the measured sd matches to within 0.002. Reviewer question 5 applied to weights. Any tensor you did not fill yourself was filled by someone, according to a rule you can look up and check.",
            **{"try": "print the same four statistics for lins[0], the "
                      "3,072-input layer. Its bound is 1/sqrt(3072), so its "
                      "weights are about 5.5 times smaller than the hidden "
                      "ones. Same rule, different fan-in — and that "
                      "difference is the entire subject of Section 9."}),
        code('''
w = lins[1].weight.detach()
bound = 1 / math.sqrt(WIDTH)
print(f"hidden weight matrix: {tuple(w.shape)}")
print(f"  min {w.min():+.4f}   max {w.max():+.4f}")
print(f"  mean {w.mean():+.5f}   sd {w.std():.5f}")
print(f"\\nthat is U(-1/sqrt(fan_in), +1/sqrt(fan_in)) = "
      f"U({-bound:.4f}, {bound:+.4f})")
print(f"  a uniform on (-b, b) has sd b/sqrt(3) = "
      f"{bound/math.sqrt(3):.5f}   <- matches")
assert abs(float(w.std()) - bound / math.sqrt(3)) < 0.002
'''),

        md("""
## 6 · Train it

The loop is Lecture 10's, unchanged, with the three defences that lecture
ended on: `zero_grad()` inside the batch loop, `eval()` before every
measurement, and accuracy counted over the set rather than averaged over
batches.

⏱ **about 40–90 seconds** for 20 epochs, depending on the runtime.
"""),
        prompt(
            label="⏱ 40-90 s — train it",
            input="the twenty-layer stack",
            output="the loss curve, validation and test accuracy, against the baseline",
            constraint="the loop is the previous lecture's with all three defences — zero_grad inside the batch loop, eval() before every measurement, accuracy counted over the set",
            check="print the loss at epoch 1 and at epoch 20 beside ln(10). Three numbers, and they say the whole thing before any plot.",
            **{"try": "raise LR from 1e-3 to 1e-1 and re-run. The loss still "
                      "does not move. No learning rate rescues a gradient "
                      "fifteen orders of magnitude too small, and this is the "
                      "ninety-second version of that argument."}),
        code('''
EPOCHS, BATCH, LR = 20, 128, 1e-3

Xf = torch.tensor(X_fit,  device=device); yf = torch.tensor(y_fit,  device=device)
Xv = torch.tensor(X_val,  device=device); yv = torch.tensor(y_val,  device=device)
Xt = torch.tensor(X_test, device=device); yt = torch.tensor(y_test, device=device)

@torch.no_grad()
def accuracy(net, X, y, batch=2000):
    """Counted over the whole set, not averaged per batch — Lecture 10."""
    net.eval()
    hits = sum(int((net(X[i:i+batch]).argmax(1) == y[i:i+batch]).sum())
               for i in range(0, len(X), batch))
    net.train()
    return hits / len(X)

def train(net, epochs=EPOCHS, lr=LR, batch=BATCH, seed=RANDOM_STATE,
          track_grads=False):
    torch.manual_seed(seed)
    net = net.to(device)
    opt   = torch.optim.Adam(net.parameters(), lr=lr)
    lossf = nn.CrossEntropyLoss()
    lins  = [m for m in net if isinstance(m, nn.Linear)]
    g = torch.Generator().manual_seed(seed)
    hist = {"loss": [], "val_acc": [], "grad": []}
    t0 = time.perf_counter()
    for _ in range(epochs):
        net.train()
        perm = torch.randperm(len(Xf), generator=g).to(device)
        total, nb = 0.0, 0
        for i in range(0, len(Xf), batch):
            idx = perm[i:i + batch]
            opt.zero_grad()                                   # 1
            loss = lossf(net(Xf[idx]), yf[idx])               # 2, 3
            loss.backward()                                   # 4
            opt.step()                                        # 5
            total += float(loss.item()); nb += 1
        hist["loss"].append(total / nb)
        hist["val_acc"].append(accuracy(net, Xv, yv))
        if track_grads:
            hist["grad"].append([float(m.weight.grad.norm()) for m in lins])
    hist["seconds"] = time.perf_counter() - t0
    return net, hist

torch.manual_seed(RANDOM_STATE)
deep = make_net()
deep, hist = train(deep, track_grads=True)

print(f"{hist['seconds']:.1f} s on {device}")
print(f"loss  epoch 1 {hist['loss'][0]:.4f}  ->  "
      f"epoch {EPOCHS} {hist['loss'][-1]:.4f}")
print(f"chance loss ln(10) = {math.log(10):.4f}")
print(f"validation accuracy {hist['val_acc'][-1]:.4f}")
print(f"TEST accuracy       {accuracy(deep, Xt, yt):.4f}")
print(f"baseline            {baseline:.4f}")
'''),
        prompt(
            label="the curves that show nothing happening",
            input="the recorded history",
            output="training loss with ln(10) marked, and validation accuracy with 10% marked",
            constraint="draw BOTH anchor lines, and cap the accuracy axis at 30% — autoscaling a flat line at 10% produces a dramatic-looking chart of noise",
            check="fix the axis limits when you are showing that something did NOT happen. Autoscale is for exploring; a fixed axis is for claiming.",
            **{"try": "delete the set_ylim(0, 30) and redraw. Autoscale turns "
                      "a flat line at 10% into a dramatic chart of noise. "
                      "Read the y-axis on the new version and write down its "
                      "full range."}),
        code('''
fig, ax = plt.subplots(1, 2, figsize=(12, 3.6))
ax[0].plot(range(1, EPOCHS+1), hist["loss"], marker="o")
ax[0].axhline(math.log(10), ls=":", color="grey")
ax[0].set_xlabel("epoch"); ax[0].set_ylabel("training loss")
ax[1].plot(range(1, EPOCHS+1), [100*a for a in hist["val_acc"]], marker="o")
ax[1].axhline(10, ls="--", color="grey")
ax[1].set_xlabel("epoch"); ax[1].set_ylabel("validation accuracy, %")
ax[1].set_ylim(0, 30)
plt.tight_layout(); plt.show()
'''),

        md("""
## 7 · Before blaming the architecture, rule out the bugs

Lecture 10 ended with a checklist of failures that run, produce a plausible
number and never raise. All of them would look exactly like this. Check them
rather than assuming.
"""),
        prompt(
            label="rule out the bugs before blaming the architecture",
            input="the checklist from the previous lecture",
            output="a label-image alignment check, and a deliberate overfit of 200 images",
            constraint="check that the SAME loop can memorise 200 images with 2 layers — if it cannot, the loop is the bug and the depth is irrelevant",
            check="the overfit test is the strongest single diagnostic in deep learning. A loop that cannot memorise a tiny sample is broken; a loop that can is not the reason your model does not learn.",
            **{"try": "shuffle the labels before the 200-image overfit. It "
                      "still reaches near-perfect training accuracy, because "
                      "memorising 200 arbitrary labels is exactly what it is "
                      "doing. What does the overfit test prove, then, and "
                      "what does it not?"}),
        code('''
# 1. is zero_grad inside the batch loop?  Read `train` above: yes, line 1.
# 2. is eval() used for every measurement? `accuracy` calls it: yes.
# 3. is the metric counted over the set?   `accuracy` counts hits: yes.
# 4. are the labels aligned with the images?
k = 3
print("label check:", CLASSES[y_fit[k]], "for the image below")
plt.figure(figsize=(1.6, 1.6))
plt.imshow(Xtr_u8[fit_idx[k]]); plt.xticks([]); plt.yticks([]); plt.show()

# 5. can the SAME loop fit anything at all?  Overfit 200 images on purpose:
torch.manual_seed(RANDOM_STATE)
tiny = make_net(depth=2).to(device)
opt, lossf = torch.optim.Adam(tiny.parameters(), lr=1e-3), nn.CrossEntropyLoss()
for _ in range(200):
    opt.zero_grad(); lossf(tiny(Xf[:200]), yf[:200]).backward(); opt.step()
print(f"\\n2 layers, 200 images, 200 steps -> training accuracy "
      f"{accuracy(tiny, Xf[:200], yf[:200]):.3f}")
print("The loop can memorise. So the loop is not the bug.")
'''),

        md("""
### The control: the same code, two layers instead of twenty

One variable changes.

⏱ **about 2 minutes** for the whole sweep.
"""),
        prompt(
            label="⏱ 2 min — the control",
            input="the same code at depths 1, 2, 5, 10 and 20",
            output="final loss and test accuracy at each depth",
            constraint="re-seed before every fit so the only variable is depth",
            check="assert depth 2 beats depth 20 — if it does not, depth is not the variable and the rest of the lecture is about the wrong thing. An assert that encodes the premise of the lecture. If it fires, stop and re-diagnose rather than continuing to instrument the wrong thing.",
            **{"try": "add depth 40 to the sweep. It is no worse than depth "
                      "20, because both sit at chance and chance has a floor. "
                      "Past the point where the signal is gone, more depth "
                      "costs wall clock and measures nothing."}),
        code('''
sweep, params = {}, {}
for k in (1, 2, 5, 10, 20):
    torch.manual_seed(RANDOM_STATE)
    net_k = make_net(depth=k)
    params[k] = sum(p.numel() for p in net_k.parameters())
    m, h = train(net_k)
    sweep[k] = accuracy(m, Xt, yt)
    print(f"depth {k:2d}: {params[k]:>7,} parameters   "
          f"final loss {h['loss'][-1]:.4f}   test accuracy {sweep[k]:.4f}")

# The parameter count is printed because it is the control for the obvious
# rival explanation. The 3,072-input first layer dominates the count, so going
# from one hidden layer to twenty barely changes it -- whatever goes wrong
# below is therefore not a story about capacity.
print(f"\\ndepth 1 -> 20 changes the parameter count by "
      f"{100*(params[20]/params[1] - 1):.0f}%")

plt.figure(figsize=(7, 3.2))
plt.bar([str(k) for k in sweep], [100*v for v in sweep.values()])
plt.axhline(10, ls="--", color="grey")
plt.xlabel("hidden layers"); plt.ylabel("test accuracy, %")
plt.tight_layout(); plt.show()

assert sweep[2] > sweep[20], "if this fails, depth is not the variable"
print("\\nAdding layers made it worse. That is not what capacity is supposed "
      "to do.")
'''),

        md("""
## 8 · Instrument it

We have a loop we wrote, so we can put a probe anywhere in it. Two probes:
what each layer **outputs** on the way forward, and what each weight matrix
**receives** on the way back.

Start with the forward pass.
"""),
        prompt(
            label="instrument the forward pass",
            input="512 training images through the trained network",
            output="mean, sd over the whole tensor, sd down the batch, and a saturation fraction, per layer",
            constraint="report TWO standard deviations — over the whole tensor and down the BATCH — because they say opposite things and only one is about the signal",
            check="assert one row per hidden layer before reading anything off it. The saturated column is worse than useless here, and the notebook says so. It asks whether |h − 0.5| > 0.45 while the activations sit in a band of sd 0.071 — 6.3 standard deviations away, unreachable, and 0.000 at every depth reads as reassurance.",
            **{"try": "change the saturation threshold from 0.45 to 0.15 and "
                      "read the column again. It is still 0.000 everywhere, "
                      "because the band's sd is about 0.071 and 0.15 is still "
                      "two sd out. Find the threshold at which the column "
                      "stops being a statement about the width of the band."}),
        code('''
@torch.no_grad()
def activation_stats(net, X, n=512):
    """Mean, sd and saturated fraction of every hidden layer's output."""
    net.eval()
    h = X[:n]
    rows = []
    for m in net:
        h = m(h)
        if isinstance(m, nn.Sigmoid):
            rows.append({"mean": float(h.mean()),
                         # sd over the WHOLE tensor is dominated by the spread
                         # of this layer's random biases across units, which
                         # does not change with depth — so it stays near 0.07
                         # for twenty layers and the network looks alive.
                         "sd_all": float(h.std()),
                         # What actually carries information: how much a unit's
                         # output moves when the INPUT changes. sd down the
                         # batch, averaged over units.
                         "sd": float(h.std(dim=0).mean()),
                         "saturated": float(((h - 0.5).abs() > 0.45)
                                            .float().mean())})
    net.train()
    return rows

stats = activation_stats(deep, Xf)
assert len(stats) == DEPTH
print(f"{'layer':>6}{'mean':>9}{'sd (all)':>11}{'sd (signal)':>14}"
      f"{'saturated':>11}")
for i in (0, 1, 4, 9, 14, 19):
    s = stats[i]
    print(f"{i+1:>6}{s['mean']:>9.4f}{s['sd_all']:>11.4f}"
          f"{s['sd']:>14.2e}{s['saturated']:>11.3f}")

ratio = stats[19]["sd"] / stats[0]["sd"]
print(f"\\nsignal at layer 20, relative to layer 1: {ratio:.2e}")
print(f"per layer that is a factor of {ratio ** (1/19):.3f}")
'''),
        prompt(
            label="the same layers, on the axis that matters",
            input="the activation statistics",
            output="two plots — the misleading one on a linear axis, then the signal on a log axis",
            constraint="draw BOTH, in that order. The first is what you would have plotted; the second is what is true",
            check="when a quantity might span orders of magnitude, try a log axis before concluding it is constant. Flat on linear and flat on log are very different findings.",
            **{"try": "add the sd down the batch to the first, linear plot. "
                      "It is a flat line at zero for eighteen of the twenty "
                      "layers. The log axis added no information — it made "
                      "sixteen orders of magnitude visible at all."}),
        code('''
plt.figure(figsize=(7, 3.2))
plt.plot(range(1, DEPTH+1), [s["mean"] for s in stats], marker="o", label="mean")
plt.plot(range(1, DEPTH+1), [s["sd_all"] for s in stats], marker="s",
         label="sd over the whole tensor")
plt.xlabel("hidden layer"); plt.ylabel("activation"); plt.ylim(0, 0.62)
plt.legend(); plt.tight_layout(); plt.show()

# The same layers, on the axis that matters. Note the log scale.
plt.figure(figsize=(7, 3.2))
plt.semilogy(range(1, DEPTH+1), [s["sd"] for s in stats], marker="s",
             color="#c0392b", label="sd down the batch (the signal)")
plt.xlabel("hidden layer"); plt.ylabel("input-dependent spread")
plt.legend(); plt.tight_layout(); plt.show()
'''),
        md("""
Read that carefully before going on, because it rules out the answer most
people reach for first.

**Read the two sd columns against each other.** The first is flat and the
second falls off a cliff, and only one of them is about the signal.

`h.std()` over the whole tensor barely moves with depth — it stays near 0.07 at
layer 20 — because it is dominated by the spread of that layer's random
**biases** across units, and that spread does not care how deep the layer is.
Reading it, you would conclude the forward pass is healthy and go looking
elsewhere. **That conclusion would be wrong.**

What carries information is how much a unit's output moves when the *input*
changes: the sd **down the batch**, averaged over units. That is the second
column, and it collapses by a factor of **0.149 per layer** — from
1.29e-01 at layer 1 to **2.42e-17** at layer 20. Sixteen orders of magnitude.
By layer 20 every input produces essentially the same activation, which is
another way of saying the network has stopped being a function of its input.

**And the saturated column is worse than useless.** It asks whether
$|h - 0.5| > 0.45$ — but the activations sit in a band whose sd is about
**0.071**, so that threshold is **6.3 standard deviations** away, and 3.4 even
at layer 1. Nothing could ever have crossed it. `0.000 at every depth` is a
statement about the width of the band, not about saturation, and it read as
reassurance.

**So the forward signal does die.** It dies quietly, and both obvious probes say
otherwise. 0.149 is not a coincidence either — it is exactly the per-layer factor
the mathematical thread predicts from the fan-in, and exactly the rate at which
the gradient vanishes on the way back.
"""),

        md("""
### Now the backward pass

The gradient of the loss with respect to each weight matrix, averaged over
eight batches — one batch of 128 is a noisy estimate of anything, and a course
that says so should not then quote one.

Measured on the CPU in float64. Not superstition: a norm is a sum of squares,
and if the smallest gradients were near 1e-20 then squaring them would underflow
to exactly zero in float32 and the plot would be a lie. We check that it does
not, rather than assuming.
"""),
        prompt(
            label="the backward pass, in float64",
            input="eight batches through a freshly initialised network",
            output="the mean gradient norm per weight matrix, in both precisions",
            constraint="average over EIGHT batches — one batch of 128 is a noisy estimate of anything, and a course that says so should not then quote one",
            check="assert no float32 gradient underflowed to exactly zero, and report the largest disagreement between the two precisions. When you plot something tiny on a log axis, check it did not underflow. A log plot of zeros is blank, and a log plot of denormals is noise.",
            **{"try": "run grad_profile in float16. Several layers underflow "
                      "to exactly zero and the assert fires. Then say what "
                      "the float32 plot would have looked like had this stack "
                      "been thirty layers deep rather than twenty."}),
        code('''
def grad_profile(net_factory, X, y, n_batches=8, dtype=torch.float64):
    torch.manual_seed(RANDOM_STATE)
    net = net_factory().to(dtype)
    lossf = nn.CrossEntropyLoss()
    lins = [m for m in net if isinstance(m, nn.Linear)]
    acc = np.zeros(len(lins))
    g = torch.Generator().manual_seed(RANDOM_STATE)
    for _ in range(n_batches):
        idx = torch.randperm(len(X), generator=g)[:128].numpy()
        net.zero_grad()
        lossf(net(torch.as_tensor(X[idx], dtype=dtype)),
              torch.as_tensor(y[idx])).backward()
        acc += np.array([float(m.weight.grad.norm()) for m in lins])
    return acc / n_batches

g64 = grad_profile(make_net, X_fit, y_fit, dtype=torch.float64)
g32 = grad_profile(make_net, X_fit, y_fit, dtype=torch.float32)

print(f"largest relative disagreement between float32 and float64: "
      f"{np.max(np.abs(g32 - g64) / g64):.2e}")
print(f"layers whose float32 gradient underflowed to exactly zero: "
      f"{int((g32 == 0).sum())}")
assert (g32 > 0).all(), "if this ever fails, the float32 plot is meaningless"

for i in (0, 4, 9, 14, 19, 20):
    name = "head" if i == DEPTH else f"layer {i+1}"
    print(f"{name:>8s}   ||dL/dW|| = {g64[i]:.4e}")
'''),
        prompt(
            label="a straight line on a log axis",
            input="the gradient profile",
            output="the norm per layer, and the per-layer attenuation factor",
            constraint="compute the per-layer factor as a GEOMETRIC mean of the consecutive ratios, and verify it by raising it to the 19th power",
            check="the self-check in the last print. If the per-layer factor to the 19th does not reproduce the measured end-to-end ratio, one of the two is wrong and you have found out in one line.",
            **{"try": "fit a straight line to log10(g64) with np.polyfit and "
                      "compare 10 to the power of its slope with the "
                      "geometric mean printed here. They agree — which is "
                      "what 'a straight line on a log axis' means as a number "
                      "rather than as a picture."}),
        code('''
plt.figure(figsize=(8, 3.4))
plt.semilogy(range(1, len(g64)+1), g64, marker="o")
plt.xlabel("layer  (1 = nearest the input)")
plt.ylabel("||dL/dW||")
plt.title("Gradient norm per weight matrix, at initialisation")
plt.tight_layout(); plt.show()

atten = g64[19] / g64[0]
ratios = g64[1:20] / g64[0:19]
gain = float(np.exp(np.mean(np.log(ratios))))      # geometric mean
print(f"layer 20 : layer 1  =  {atten:.4e}   "
      f"({math.log10(atten):.1f} orders of magnitude)")
print(f"per layer, going down: x {1/gain:.4f}")
print(f"check: {1/gain:.4f} ** 19 = {(1/gain)**19:.4e}   "
      f"and 1/{atten:.4e} = {1/atten:.4e}")
'''),
        md("""
**That is a straight line on a log axis.** Which means the attenuation is not
an accident of one layer: it is the *same factor, applied nineteen times*.

Write down the two numbers — the per-layer factor and the end-to-end ratio.
They are what the next lecture derives from first principles.
"""),

        md("""
### Does training rescue it?

We logged the per-layer gradient norms at every epoch. If the first layers were
merely slow to start, the profile would flatten as the network learns.
"""),
        prompt(
            label="does training rescue it",
            input="the per-layer gradient norms recorded at every epoch",
            output="the profile at epochs 1, 5, 10 and 20",
            constraint="plot several epochs on ONE axis — the question is whether the shape changes, and one epoch per panel cannot answer it",
            check="this is why the instrumentation had to go in before the first training run. The question 'did it change over time' cannot be asked retroactively.",
            **{"try": "re-run the training with 100 epochs and redraw this. "
                      "The profile keeps its shape. The first layer is not "
                      "slow to start; it is receiving a signal no learning "
                      "rate can use, and more epochs of it is more of "
                      "nothing."}),
        code('''
G = np.array(hist["grad"])            # (epochs, layers)
plt.figure(figsize=(8, 3.4))
for ep in (0, 4, 9, 19):
    plt.semilogy(range(1, G.shape[1]+1), G[ep], marker="o",
                 label=f"epoch {ep+1}")
plt.xlabel("layer"); plt.ylabel("||dL/dW||"); plt.legend()
plt.tight_layout(); plt.show()

print(f"layer 1 gradient, epoch 1  {G[0][0]:.3e}")
print(f"layer 1 gradient, epoch {EPOCHS} {G[-1][0]:.3e}")
print("\\nTwenty epochs of Adam did not move the first layer's gradient into a "
      "range where a learning rate of 1e-3 could do anything with it.")
'''),

        md("""
### What that means for the update

Adam divides by a running estimate of the gradient's own magnitude, so it is
not simply "small gradient, small step". But the first layers are being driven
by a signal that is fifteen orders of magnitude below the last ones, and it is
almost entirely noise from the eight-batch spread. Check what the weights
actually did over twenty epochs.
"""),
        prompt(
            label="what the weights actually did",
            input="the initial weights, reproduced, against the trained ones",
            output="the relative change in each layer's weights over twenty epochs",
            constraint="reproduce the initial weights by RE-SEEDING — `train` does not re-initialise the network it is handed, so the same seed gives exactly the tensors `deep` started from",
            check="relative change, not absolute. A weight matrix with small entries and one with large entries cannot be compared by the norm of their differences.",
            **{"try": "print the relative change for all twenty-one matrices "
                      "and plot it against layer index on a log axis. It is "
                      "the same straight line as the gradient profile. That "
                      "is what you would expect, which is exactly why it is "
                      "worth confirming rather than assuming."}),
        code('''
# Re-seeding reproduces exactly the weights `deep` started from, because
# `train` does not re-initialise the network it is handed.
torch.manual_seed(RANDOM_STATE)
before = make_net()
w_before = [m.weight.detach().clone() for m in before if isinstance(m, nn.Linear)]
after = [m.weight.detach().cpu() for m in deep if isinstance(m, nn.Linear)]

for i in (0, 9, 19, 20):
    rel = float((after[i] - w_before[i]).norm() / w_before[i].norm())
    name = "head" if i == DEPTH else f"layer {i+1}"
    print(f"{name:>8s}: relative change in the weights over "
          f"{EPOCHS} epochs = {rel:.4f}")
'''),

        md("""
## 9 · The derivation — variance propagation

One layer, written without the activation for a moment:

$$z_i \\;=\\; \\sum_{j=1}^{n_{\\text{in}}} w_{ij}\\, a_j$$

Assume the weights are drawn independently with mean 0 and variance
$\\operatorname{Var}(w)$, independently of the inputs, and that the inputs have
mean 0. Then every term in the sum has mean 0 and the terms are uncorrelated,
so the variances add:

$$\\boxed{\\;\\operatorname{Var}(z) \\;=\\; n_{\\text{in}}\\cdot
\\operatorname{Var}(w)\\cdot \\operatorname{Var}(a)\\;}$$

That is the whole object. Everything in this lecture is a consequence of it.

**Check it before believing it.**
"""),
        prompt(
            label="check the variance identity before believing it",
            input="random weight matrices at four weight variances",
            output="the predicted and measured variance of the output",
            constraint="unit-variance inputs, so `Var(z) = n_in · Var(w)` can be read directly against the measurement",
            check="20,000 samples, not 100. This is a claim about a variance, and a variance estimated from a small sample has a variance of its own.",
            **{"try": "drop N from 20,000 to 100 and re-run. Prediction and "
                      "measurement now disagree in the second decimal, and "
                      "the identity has not failed: a variance estimated from "
                      "N samples has a variance of its own, of order 2/N."}),
        code('''
torch.manual_seed(RANDOM_STATE)
n_in, n_out, N = 100, 100, 20_000
for var_w in (0.001, 0.01, 0.02, 0.05):
    W = torch.randn(n_out, n_in) * math.sqrt(var_w)
    a = torch.randn(N, n_in)                      # Var(a) = 1
    z = a @ W.T
    print(f"Var(w) = {var_w:.3f}   predicted Var(z) = "
          f"{n_in * var_w:.4f}   measured {z.var():.4f}")
'''),

        md("""
### The forward requirement

We want the signal to arrive at layer 20 with the same spread it had at layer
1 — otherwise, whichever way it drifts, twenty layers of drift is a factor of
$r^{20}$.

Setting $\\operatorname{Var}(z) = \\operatorname{Var}(a)$ gives

$$n_{\\text{in}}\\cdot\\operatorname{Var}(w) = 1
\\qquad\\Longrightarrow\\qquad
\\operatorname{Var}(w) = \\frac{1}{n_{\\text{in}}}$$

### The backward requirement

The backward pass runs the transpose. Ignoring the activation for a moment,

$$\\bar a \\;=\\; W^{\\mathsf T}\\bar z
\\qquad\\Longrightarrow\\qquad
\\operatorname{Var}(\\bar a) = n_{\\text{out}}\\cdot\\operatorname{Var}(w)
\\cdot\\operatorname{Var}(\\bar z)$$

Same algebra, one index swapped — because the transpose sums over the *output*
dimension. Preserving the gradient therefore asks for

$$\\operatorname{Var}(w) = \\frac{1}{n_{\\text{out}}}$$

### The conflict

Those two conditions are the same condition **only when
$n_{\\text{in}} = n_{\\text{out}}$.** A network whose layers change width — every
real network, starting with ours, whose first layer is 3072 → 100 — cannot
satisfy both.

Glorot's answer is not to choose. Take the harmonic mean of the two demands:

$$\\operatorname{Var}(w) = \\frac{2}{n_{\\text{in}} + n_{\\text{out}}}$$

and accept an error in both directions rather than a large error in one.
"""),
        prompt(
            label="the conflict, in three rows",
            input="the three layer shapes in this network",
            output="what the forward pass wants, what the backward pass wants, and what Glorot gives",
            constraint="show all THREE columns per row — the point is that the first two disagree and the third is a compromise, not a derivation",
            check="on the 100 → 100 layers the two demands agree and Glorot is exact. On the first layer they differ by a factor of 30 and nothing can fix that.",
            **{"try": "add a 100 -> 3072 layer, the transpose of the first "
                      "one. Forward and backward swap demands and Glorot "
                      "gives the identical answer for both, because the "
                      "harmonic mean is symmetric in its two arguments. Which "
                      "real architectures have a layer shaped like that?"}),
        code('''
for (nin, nout) in [(3072, 100), (100, 100), (100, 10)]:
    fwd = 1 / nin
    bwd = 1 / nout
    glorot = 2 / (nin + nout)
    print(f"n_in {nin:5d}  n_out {nout:4d}   forward wants Var(w) = {fwd:.6f}"
          f"   backward wants {bwd:.6f}   Glorot gives {glorot:.6f}")
print("\\nOn the 100 -> 100 layers the two demands agree and Glorot is exact.")
print("On the first layer they differ by a factor of 30, and nothing can fix "
      "that; Glorot splits the difference.")
'''),

        md("""
### He's adjustment, for ReLU

Glorot's derivation assumed the activation was roughly linear near zero. ReLU
is not: it sets half of its inputs to zero. For a symmetric, zero-mean $z$,

$$\\mathbb{E}\\!\\left[\\operatorname{ReLU}(z)^2\\right]
= \\tfrac12\\,\\mathbb{E}[z^2]$$

so exactly half the variance is discarded at every layer. Double the weight
variance to put it back:

$$\\operatorname{Var}(w) = \\frac{2}{n_{\\text{in}}}$$

Check the factor of two rather than taking it:
"""),
        prompt(
            label="check He's factor of two",
            input="half a million standard normal samples",
            output="E[z²], E[relu(z)²] and their ratio",
            constraint="measure the ratio rather than quoting 1/2 — it is one line, and the whole He correction rests on it",
            check="the logistic's derivative never exceeds 1/4, which is the other half of the story and the direct cause of the previous lecture's failure.",
            **{"try": "measure the same ratio for a pre-activation with a "
                      "non-zero mean, torch.randn(500_000) + 1. It is no longer "
                      "1/2. He's correction assumes a symmetric zero-mean "
                      "pre-activation, and a drifting bias breaks that assumption "
                      "long before it breaks the network."}),
        code('''
z = torch.randn(500_000)
print(f"E[z^2]            {(z**2).mean():.4f}")
print(f"E[relu(z)^2]      {(torch.relu(z)**2).mean():.4f}")
print(f"ratio             {((torch.relu(z)**2).mean() / (z**2).mean()):.4f}")
print(f"\\nfor the logistic, the derivative never exceeds "
      f"{0.25:.2f}, which is the other half of the story")
'''),

        md("""
### The consequence: it compounds geometrically

Put the activation back. One layer multiplies the *standard deviation* of the
backward signal by

$$\\rho \\;=\\; \\sqrt{\\,n_{\\text{out}}\\cdot\\operatorname{Var}(w)\\cdot
\\mathbb{E}\\!\\left[\\varphi'(z)^2\\right]\\,}$$

and $L$ layers multiply it by $\\rho^{L}$. There is no regime in which a
constant repeated $L$ times is safe: either $\\rho<1$ and the gradient
disappears, or $\\rho>1$ and it explodes. **Only $\\rho = 1$ survives depth.**

Now put our own numbers in. Every hidden layer is 100 → 100.

| | Var(w) | E[φ′²] | ρ |
|---|---|---|---|
| `nn.Linear` default, logistic | 1/(3·100) | (1/4)² | √(100·(1/300)·(1/16)) |
| Glorot, logistic | 2/200 | (1/4)² | √(100·(1/100)·(1/16)) |
| Glorot, ReLU | 2/200 | 1/2 | √(100·(1/100)·(1/2)) |
| He, ReLU | 2/100 | 1/2 | √(100·(2/100)·(1/2)) |

The default is a uniform on $(-b, b)$ with $b = 1/\\sqrt{n_{\\text{in}}}$, and a
uniform on $(-b,b)$ has variance $b^2/3$ — hence the 3.
"""),
        prompt(
            label="the prediction, from shapes alone",
            input="the weight variance and E[φ′²] of each scheme",
            output="ρ per layer, and ρ to the 19th",
            constraint="compute it from the SHAPES of the matrices and one expectation — no network is built and no data is touched",
            check="a prediction made before the measurement is worth ten made after it. Write these four numbers down before running the next cell.",
            **{"try": "add a fifth row: He initialisation with the logistic, "
                      "rho(2 / WIDTH, 0.25 ** 2). Predict from the number "
                      "alone whether it trains, then look for that "
                      "combination in the ladder of Section 13. It is not "
                      "there, and now you can say why."}),
        code('''
def rho(var_w, Ephi2, n_out=WIDTH):
    return math.sqrt(n_out * var_w * Ephi2)

theory = {
    "default, logistic": rho(1 / (3 * WIDTH), 0.25 ** 2),
    "Glorot,  logistic": rho(2 / (2 * WIDTH), 0.25 ** 2),
    "Glorot,  ReLU":     rho(2 / (2 * WIDTH), 0.5),
    "He,      ReLU":     rho(2 / WIDTH,       0.5),
}
for k, v in theory.items():
    print(f"{k:18s}  rho = {v:.4f}   over 19 layers: {v**19:.3e}")
'''),

        md("""
### One builder, extended — and said out loud

Everything from here on varies the initialisation, the activation, the
normalisation and the dropout, so `make_net` needs to take all four. The next
cell **redefines it**.

Redefining a function halfway down a notebook is a genuine hazard: every cell
above still refers to the name, so re-running an early cell after this point
silently uses the new definition. It is done here because the alternative — two
names for one idea — is worse, and it is announced because the dangerous version
of this is the unannounced one. The defaults reproduce the original exactly.
"""),
        prompt(
            label="the builder, taking every knob this lecture turns",
            input="depth, width, activation, initialisation scheme, normalisation, dropout",
            output="the network, on the requested dtype",
            constraint="the DEFAULTS must reproduce the simple version from section 1 exactly — a redefinition that changes the baseline invalidates every table above it",
            check="assert the default build still has DEPTH+1 weight matrices and the same parameter count as the network built in section 1.",
            **{"try": "pass init='normal1' — a normal of standard deviation 1, which is what the vanishing-gradient section uses as its counter-example. Check what rho that predicts before you run it."}),
        code('''
ACTS = {"sigmoid": nn.Sigmoid, "relu": nn.ReLU, "tanh": nn.Tanh}

def init_linear(m, scheme, act):
    """The four schemes this lecture compares, plus PyTorch's own default."""
    if scheme == "torch":
        return                                    # leave nn.Linear as it built it
    fan_in, fan_out = m.weight.shape[1], m.weight.shape[0]
    if scheme == "glorot":
        std = math.sqrt(2.0 / (fan_in + fan_out))
    elif scheme == "he":
        std = math.sqrt(2.0 / fan_in)
    elif scheme == "normal1":
        std = 1.0                                 # the counter-example, deliberately
    else:
        raise ValueError(f"unknown init {scheme!r}")
    nn.init.normal_(m.weight, 0.0, std)
    nn.init.zeros_(m.bias)

def make_net(depth=DEPTH, width=WIDTH, act="sigmoid", init="torch",
             norm=None, dropout=0.0, n_in=N_IN, n_out=N_OUT, dtype=None):
    layers, prev = [], n_in
    for _ in range(depth):
        lin = nn.Linear(prev, width)
        init_linear(lin, init, act)
        layers.append(lin)
        if norm == "batch":
            layers.append(nn.BatchNorm1d(width))
        elif norm == "layer":
            layers.append(nn.LayerNorm(width))
        layers.append(ACTS[act]())
        if dropout:
            layers.append(nn.Dropout(dropout))
        prev = width
    head = nn.Linear(prev, n_out)
    init_linear(head, init, act)
    layers.append(head)
    net = nn.Sequential(*layers)
    return net if dtype is None else net.to(dtype)

# the defaults must still be the network section 1 built
check = make_net()
assert len([m for m in check if isinstance(m, nn.Linear)]) == DEPTH + 1
assert sum(p.numel() for p in check.parameters()) == n_params
print(f"redefined: defaults reproduce section 1 — {n_params:,} parameters")
'''),

        md("""
## 10 · Measure the prediction against the network above

The prediction above used nothing but the shapes of the matrices and one
expectation. If it is right, it should reproduce the attenuation you logged in
the previous lecture — a number that came out of a completely different
calculation.

⏱ **about 30 seconds** — four networks, eight batches each, on the CPU in
float64.
"""),
        prompt(
            label="⏱ 30 s — measure it, four ways",
            input="four initialisation and activation schemes",
            output="predicted ρ, measured ρ, the error, the forward scale, and the end-to-end ratio",
            constraint="measure ||dL/dz|| — the DELTA — not ||dL/dW||. They are different quantities and confusing them is the single easiest way to misread this lecture",
            check="assert prediction and measurement agree within 15%, per scheme. `retain_grad()` on the intermediates. Non-leaf tensors do not keep their gradients by default, and without it `z.grad` is None with no error.",
            **{"try": "delete the retain_grad() call inside delta_profile. "
                      "Every z.grad is None and the cell dies complaining "
                      "about NoneType, never about non-leaf tensors. The "
                      "error names the symptom and never once names the "
                      "cause."}),
        code('''
def grad_profile(n_batches=8, dtype=torch.float64, **kw):
    torch.manual_seed(RANDOM_STATE)
    net = make_net(dtype=dtype, **kw)
    net.train()
    lossf = nn.CrossEntropyLoss()
    lins = [m for m in net if isinstance(m, nn.Linear)]
    acc = np.zeros(len(lins))
    g = torch.Generator().manual_seed(RANDOM_STATE)
    for _ in range(n_batches):
        idx = torch.randperm(len(X_fit), generator=g)[:BATCH].numpy()
        net.zero_grad()
        lossf(net(torch.as_tensor(X_fit[idx], dtype=dtype)),
              torch.as_tensor(y_fit[idx])).backward()
        acc += np.array([float(m.weight.grad.norm()) for m in lins])
    return acc / n_batches

def delta_profile(n_batches=8, dtype=torch.float64, **kw):
    """Norm of dL/dz at every layer -- the quantity the thread is about.

    NOT the same as the weight gradient. See the markdown cell below; getting
    these two confused is the single easiest way to misread this lecture.
    """
    torch.manual_seed(RANDOM_STATE)
    net = make_net(dtype=dtype, **kw); net.train()
    lossf = nn.CrossEntropyLoss()
    g = torch.Generator().manual_seed(RANDOM_STATE)
    acc = None
    for _ in range(n_batches):
        idx = torch.randperm(len(X_fit), generator=g)[:BATCH].numpy()
        h = torch.as_tensor(X_fit[idx], dtype=dtype)
        yb = torch.as_tensor(y_fit[idx])
        zs = []
        for m in net:
            h = m(h)
            if isinstance(m, nn.Linear):
                h.retain_grad(); zs.append(h)
        net.zero_grad()
        lossf(h, yb).backward()
        vals = np.array([float(z.grad.norm()) for z in zs])
        acc = vals if acc is None else acc + vals
    return acc / n_batches

def act_sd(dtype=torch.float64, **kw):
    torch.manual_seed(RANDOM_STATE)
    net = make_net(dtype=dtype, **kw); net.eval()
    h, sds = torch.as_tensor(X_fit[:BATCH], dtype=dtype), []
    with torch.no_grad():
        for m in net:
            h = m(h)
            if isinstance(m, (nn.Sigmoid, nn.Tanh, nn.ReLU)):
                sds.append(float(h.std()))
    return np.array(sds)

def geo(a):
    return float(np.exp(np.mean(np.log(a))))

schemes = {
    "default, logistic": dict(act="sigmoid", init="torch"),
    "Glorot,  logistic": dict(act="sigmoid", init="glorot"),
    "Glorot,  ReLU":     dict(act="relu",    init="glorot"),
    "He,      ReLU":     dict(act="relu",    init="he"),
}
profiles, deltas, forwards = {}, {}, {}
print(f"{'':18s} {'predicted':>10s} {'measured rho':>13s} {'error':>8s} "
      f"{'fwd scale':>10s} {'d1/d20':>11s}")
for k, v in schemes.items():
    profiles[k] = grad_profile(**v)
    d = delta_profile(**v)
    sd = act_sd(**v)
    rho = geo(d[0:DEPTH-1] / d[1:DEPTH])
    fwd = geo(sd[1:DEPTH] / sd[0:DEPTH-1])
    deltas[k], forwards[k] = rho, fwd
    print(f"{k:18s} {theory[k]:10.4f} {rho:13.4f} "
          f"{abs(rho-theory[k])/theory[k]:7.1%} {fwd:10.4f} "
          f"{d[0]/d[DEPTH-1]:11.3e}")
    assert abs(rho - theory[k]) / theory[k] < 0.15, \
        f"{k}: prediction and measurement disagree by more than 15%"
'''),
        prompt(
            label="four profiles on one log axis",
            input="the four weight-gradient profiles",
            output="||dL/dW|| per layer, one line per scheme",
            constraint="log y-axis and all four on ONE plot — the schemes span fifteen orders of magnitude between them",
            check="this is the whole lecture. Everything below it is application.",
            **{"try": "plot the same four profiles on a linear axis. Three of "
                      "the curves become indistinguishable from the x-axis. "
                      "Fifteen orders of magnitude is not a range a linear "
                      "axis can carry, and this is the figure that settles "
                      "it."}),
        code('''
plt.figure(figsize=(8.5, 3.6))
for k, g in profiles.items():
    plt.semilogy(range(1, len(g)+1), g, marker="o", ms=4, label=k)
plt.xlabel("layer  (1 = nearest the input)"); plt.ylabel("||dL/dW||")
plt.legend(fontsize=8); plt.tight_layout(); plt.show()
'''),
        md("""
The prediction came from counting rows and columns of matrices. The measurement
came from a backward pass through twenty layers of a real network on real
photographs. They agree to within a few per cent, and the residual is exactly
where the assumptions are weakest: `E[φ′²]` is not quite (1/4)² because the
pre-activations are not exactly at zero, and successive components of the
gradient are not exactly uncorrelated.

**This is the whole lecture.** Everything below is applying it.

### But that is not the number section 8 logged

Section 8 logged `||dL/dW||`, not `||dL/dz||`. They differ, because

    || dL/dW_l ||  ~  || delta_l || . || a_(l-1) ||

so the weight-gradient ratio carries the backward factor **and** the forward
one. For the logistic the forward pass is scale-stable, so the two nearly
coincide. For an unnormalised ReLU stack they do not — and that is not a
technicality, as the next cell shows.
"""),
        prompt(
            label="why the weight-gradient diagnostic is not enough",
            input="ρ, the forward scale, and the weight-gradient ratio per scheme",
            output="all four columns side by side",
            constraint="show ρ and the forward factor SEPARATELY as well as their product — the product is what the previous lecture measured",
            check="||dL/dW_l|| ≈ ||delta_l||·||a_(l−1)||. Two factors in one number, and they can cancel.",
            **{"try": "construct on paper an initialisation whose rho is 0.5 "
                      "and whose forward factor is 2. Its weight-gradient "
                      "profile would be perfectly flat and the network would "
                      "still not train. That construction is the reason this "
                      "cell exists."}),
        code('''
print(f"{'':18s} {'rho':>8s} {'fwd':>8s} {'rho/fwd':>9s} {'||dW|| ratio':>13s}")
for k in schemes:
    g = profiles[k]
    w = geo(g[0:DEPTH-1] / g[1:DEPTH])       # descending the stack
    print(f"{k:18s} {deltas[k]:8.4f} {forwards[k]:8.4f} "
          f"{deltas[k]/forwards[k]:9.4f} {w:13.4f}")

print()
print("Read the 'Glorot, ReLU' row. Its weight-gradient ratio is close to 1 —")
print("the diagnostic from the previous lecture would pass it — because the")
print("backward attenuation and the forward attenuation cancel in that one")
print("number. A flat gradient profile does NOT certify an initialisation.")
'''),

        md("""
## 11 · The training harness

One function, one seed, one subset, one epoch count. Every row in every table
below differs from its neighbour in exactly one argument.
"""),
        prompt(
            label="the training harness",
            input="every knob the notebook varies",
            output="a trained network and its history, including the test accuracy",
            constraint="ONE function, one seed, one subset, one epoch count — so every row of every table differs from its neighbour in exactly one argument",
            check="a harness check: a 2-layer ReLU network must reach better than 0.2 in three epochs, or the harness itself cannot learn and every table below is measuring the harness. Test the instrument before the experiment. Three epochs on two layers costs seconds and rules out the most expensive possible mistake.",
            **{"try": "lower the harness threshold from 0.2 to 0.1 and re-run. It "
                      "now passes for a network sitting at chance, which is "
                      "precisely what the check was written to rule out. A "
                      "threshold set at the baseline is not a check."}),
        code('''
Xf = torch.tensor(X_fit,  device=device); yf = torch.tensor(y_fit,  device=device)
Xv = torch.tensor(X_val,  device=device); yv = torch.tensor(y_val,  device=device)
Xt = torch.tensor(X_test, device=device); yt = torch.tensor(y_test, device=device)

@torch.no_grad()
def accuracy(net, X, y, batch=2000):
    net.eval()
    hits = sum(int((net(X[i:i+batch]).argmax(1) == y[i:i+batch]).sum())
               for i in range(0, len(X), batch))
    net.train()
    return hits / len(X)

def train(epochs=EPOCHS, lr=LR, batch=BATCH, opt="adam", clip=None,
          schedule=None, seed=RANDOM_STATE, **kw):
    torch.manual_seed(seed)
    net = make_net(**kw).to(device)
    lossf = nn.CrossEntropyLoss()
    p = net.parameters()
    optim = {"adam": lambda: torch.optim.Adam(p, lr=lr),
             "adamw": lambda: torch.optim.AdamW(p, lr=lr),
             "sgd": lambda: torch.optim.SGD(p, lr=lr),
             "momentum": lambda: torch.optim.SGD(p, lr=lr, momentum=0.9),
             "nesterov": lambda: torch.optim.SGD(p, lr=lr, momentum=0.9,
                                                 nesterov=True),
             "rmsprop": lambda: torch.optim.RMSprop(p, lr=lr)}[opt]()

    steps = math.ceil(len(Xf) / batch) * epochs
    sched = None
    if schedule == "onecycle":
        sched = torch.optim.lr_scheduler.OneCycleLR(optim, max_lr=10*lr,
                                                    total_steps=steps)
    elif schedule == "cosine":
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=steps)

    g = torch.Generator().manual_seed(seed)
    hist = {"loss": [], "val_acc": [], "lr": []}
    t0 = time.perf_counter()
    for _ in range(epochs):
        net.train()
        perm = torch.randperm(len(Xf), generator=g).to(device)
        total, nb = 0.0, 0
        for i in range(0, len(Xf), batch):
            idx = perm[i:i + batch]
            optim.zero_grad()
            loss = lossf(net(Xf[idx]), yf[idx])
            loss.backward()
            if clip is not None:
                torch.nn.utils.clip_grad_norm_(net.parameters(), clip)
            optim.step()
            if sched is not None:
                sched.step()
            total += float(loss.item()); nb += 1
        hist["loss"].append(total / nb)
        hist["lr"].append(optim.param_groups[0]["lr"])
        hist["val_acc"].append(accuracy(net, Xv, yv))
    hist["seconds"] = time.perf_counter() - t0
    hist["test_acc"] = accuracy(net, Xt, yt)
    hist["n_params"] = sum(p.numel() for p in net.parameters())
    return net, hist

_, h = train(depth=2, act="relu", init="he", epochs=3)
assert h["test_acc"] > 0.2, "the harness itself should be able to learn"
print(f"harness check: a 2-layer ReLU network reaches "
      f"{h['test_acc']:.3f} in 3 epochs ({h['seconds']:.0f} s)")
'''),

        md("""
## 12 · Each repair, alone

Resist applying everything at once. A stack of seven changes that works tells
you nothing about which of the seven mattered — and two of them will turn out
to make things worse on their own.

⏱ **about 5 minutes** for all seven.
"""),
        prompt(
            label="⏱ 5 min — each repair, alone",
            input="seven repairs, each applied to the broken network by itself",
            output="test accuracy and final loss for each",
            constraint="ONE change per row, all against the same broken baseline — a stack of seven changes that works tells you nothing about which of the seven mattered",
            check="clipping bounds a gradient that is too large; ours is fifteen orders of magnitude too small. Dropout fights overfitting; a network at chance is not overfitting. Applying a fix whose failure mode you have not measured is how a notebook grows to forty cells and stops being explicable.",
            **{"try": "add a row for Glorot with ReLU — the one pairing this "
                      "table skips. It lands between the two schemes it is "
                      "made of, and the rho you computed in Section 9 says "
                      "where. A missing row in an ablation is a claim nobody "
                      "tested."}),
        code('''
alone = [
    ("nothing (as built above)",  dict(act="sigmoid", init="torch")),
    ("Glorot initialisation", dict(act="sigmoid", init="glorot")),
    ("ReLU, with He",         dict(act="relu",    init="he")),
    ("Batch normalisation",   dict(act="sigmoid", init="torch", norm="batch")),
    ("Layer normalisation",   dict(act="sigmoid", init="torch", norm="layer")),
    ("Gradient clipping",     dict(act="sigmoid", init="torch", clip=1.0)),
    ("A 1-cycle schedule",    dict(act="sigmoid", init="torch",
                                   schedule="onecycle")),
    ("Dropout 0.1",           dict(act="sigmoid", init="torch", dropout=0.1)),
]
solo = {}
for label, kw in alone:
    _, h = train(**kw)
    solo[label] = h["test_acc"]
    print(f"{label:24s} test {h['test_acc']:.4f}   loss {h['loss'][-1]:.4f}   "
          f"{h['seconds']:.0f} s")
'''),
        md("""
Read the rows against 0.1000, not against each other.

Three of those repairs do nothing on their own, and that is informative rather
than disappointing: **gradient clipping, a schedule and dropout are all
answers to problems this network does not have.** Clipping bounds a gradient
that is too large; ours is fifteen orders of magnitude too small. Dropout
fights overfitting; a network at chance is not overfitting.

Applying a fix whose failure mode you have not measured is how a notebook grows
to forty cells and stops being explicable.
"""),

        md("""
## 13 · The ladder

Now stack them, in the order the diagnosis suggests: fix the signal first,
then the optimisation, then the generalisation.

⏱ **about 5 minutes.**
"""),
        prompt(
            label="⏱ 5 min — the ladder",
            input="the same repairs, stacked in diagnostic order",
            output="each rung's accuracy and its delta from the rung below",
            constraint="stack in the order the DIAGNOSIS suggests — signal first, then optimisation, then generalisation",
            check="assert the repaired network is at least three times chance, and record which rung was actually best. Capture the best row into a variable and use THAT downstream. Hard-coding the last rung's settings in the summary would report 33.4% where the argument requires 43.9% — the notebook committing the mistake the deck forbids.",
            **{"try": "reverse the ladder: start from the full stack and "
                      "remove one repair at a time. The two orderings "
                      "disagree about which repair mattered, and neither is "
                      "wrong. That disagreement is why one ladder is a "
                      "demonstration and not an attribution."}),
        code('''
ladder = [
    ("as built above, unchanged",   dict(act="sigmoid", init="torch")),
    ("+ Glorot initialisation",     dict(act="sigmoid", init="glorot")),
    ("+ ReLU and He",               dict(act="relu", init="he")),
    ("+ batch normalisation",       dict(act="relu", init="he", norm="batch")),
    ("+ gradient clipping",         dict(act="relu", init="he", norm="batch",
                                         clip=1.0)),
    ("+ a 1-cycle schedule",        dict(act="relu", init="he", norm="batch",
                                         clip=1.0, schedule="onecycle")),
    ("+ dropout 0.1",               dict(act="relu", init="he", norm="batch",
                                         clip=1.0, schedule="onecycle",
                                         dropout=0.1)),
]
rows, prev, curves = [], None, {}
for label, kw in ladder:
    _, h = train(**kw)
    delta = "" if prev is None else f"{100*(h['test_acc'] - prev):+5.1f}"
    print(f"{label:28s} {100*h['test_acc']:6.2f}%  {delta:>6s}   "
          f"{h['seconds']:.0f} s")
    rows.append((label, h["test_acc"], kw))
    curves[label] = h["loss"]
    prev = h["test_acc"]

best = max(rows, key=lambda r: r[1])
BEST_LABEL, BEST_ACC, BEST_KW = best     # the closing summary uses these
print(f"\\nbest row: {best[0]} at {best[1]:.4f}")
print(f"last row: {rows[-1][0]} at {rows[-1][1]:.4f}")
assert best[1] > 3 * rows[0][1], "the repaired network should not be at chance"
if best is not rows[-1]:
    print("\\nThe last rung is NOT the best configuration. Report the best row,")
    print("and say which rungs you dropped and why. That is what the table is")
    print("for; without it you would ship the bottom row by default.")
'''),
        prompt(
            label="the ladder, drawn",
            input="the seven rungs",
            output="a bar per rung, and four loss curves",
            constraint="draw the chance line at 10% on the bar panel and ln(10) on the loss panel — every bar has to be read against chance",
            check="horizontal bars with the labels on the axis, not a legend. Seven long labels in a legend is a puzzle.",
            **{"try": "put the seven labels in a legend instead of on the "
                      "y-axis and redraw. Then decide which version you would "
                      "put on a slide, and notice that the answer has nothing "
                      "to do with matplotlib."}),
        code('''
fig, ax = plt.subplots(1, 2, figsize=(13, 4))
labels = [r[0] for r in rows]
ax[0].barh(range(len(rows))[::-1], [100*r[1] for r in rows])
ax[0].set_yticks(range(len(rows))[::-1]); ax[0].set_yticklabels(labels, fontsize=8)
ax[0].axvline(10, ls="--", color="grey"); ax[0].set_xlabel("test accuracy, %")
for label in (labels[0], labels[2], labels[3], labels[-1]):
    ax[1].plot(range(1, EPOCHS+1), curves[label], label=label)
ax[1].axhline(math.log(10), ls=":", color="grey")
ax[1].set_xlabel("epoch"); ax[1].set_ylabel("training loss")
ax[1].legend(fontsize=8)
plt.tight_layout(); plt.show()
'''),

        md("""
## 14 · Batch normalisation, and layer normalisation

Initialisation fixes the variance **at step zero**. It says nothing about step
five thousand, by which time the weights have moved. Normalisation enforces the
same condition at every step, by standardising each layer's inputs and then
learning a scale and a shift.

⏱ **about 2 minutes.**
"""),
        prompt(
            label="⏱ 2 min — normalisation, and what it costs",
            input="the repaired ReLU network with no norm, batch norm, and layer norm",
            output="accuracy, wall clock and parameter count for each",
            constraint="report the PARAMETER COUNT — two learned vectors per layer is 200 numbers against 10,100, under 2%, and the wall clock is the real cost",
            check="initialisation fixes the variance at step ZERO. It says nothing about step five thousand, by which time the weights have moved — that is what normalisation is for.",
            **{"try": "run the batch-norm row at batch size 8 and again at "
                      "512. The accuracy moves and the parameter count does "
                      "not. Batch normalisation has a hyperparameter that "
                      "never appears in its parameter count, and it is the "
                      "batch size."}),
        code('''
for label, kw in [("none", {}), ("batch", dict(norm="batch")),
                  ("layer", dict(norm="layer"))]:
    _, h = train(act="relu", init="he", **kw)
    print(f"{label:6s} test {h['test_acc']:.4f}   {h['seconds']:5.1f} s   "
          f"{h['n_params']:,} parameters")
'''),
        md("""
Two things to read off, and the second is the one people miss.

* **What it costs.** Two learned vectors per layer, plus two running averages
  that are buffers rather than parameters. On a 100-unit layer that is 200
  numbers against 10,100 — under 2%. The wall clock is the real cost.
* **What batch normalisation depends on that layer normalisation does not.**
  Batch statistics are computed *across the batch*, so the prediction for one
  image depends on the other images in its batch at training time, and on a
  running average at evaluation time. That is why `model.eval()` matters more
  from here on, and why batch normalisation is awkward at batch size 1 or on
  sequences of different lengths.

Layer normalisation standardises across the *features of one example*, so it
has no such dependence. It arrives properly in Chapter 13; it is here for the
contrast.

Watch the two modes disagree, which is the failure Lecture 10 warned about and
which is now much harder to see, because batch normalisation does not
fluctuate:
"""),
        prompt(
            label="the two modes disagreeing, quietly",
            input="a batch-normalised network, evaluated in both modes",
            output="the accuracy each way, and the difference",
            constraint="use the SAME 2,000 images both times — the difference must come from the mode and nothing else",
            check="from here on `model.eval()` matters more, not less, and the cheap diagnostic that used to catch a missing one no longer does.",
            **{"try": "evaluate in train() mode twice on the same 2,000 "
                      "images. The two answers are identical, because the "
                      "batch statistics of a fixed batch are deterministic. "
                      "That is exactly why Lecture 10's run-it-twice "
                      "diagnostic no longer catches a missing eval()."}),
        code('''
net_bn, _ = train(act="relu", init="he", norm="batch", epochs=3)
net_bn.train()
with torch.no_grad():
    train_mode = float((net_bn(Xt[:2000]).argmax(1) == yt[:2000]).float().mean())
net_bn.eval()
with torch.no_grad():
    eval_mode = float((net_bn(Xt[:2000]).argmax(1) == yt[:2000]).float().mean())
print(f"model.train()  {train_mode:.4f}   <- batch statistics of the test batch")
print(f"model.eval()   {eval_mode:.4f}   <- running averages from training")
print(f"difference     {100*(eval_mode - train_mode):+.2f} points")
print("\\nUnlike dropout it does not wobble between calls, so running the "
      "evaluation twice does NOT catch it.")
'''),

        md("""
### The row we do not explain

Applied on their own to the failing network, batch normalisation rescues it
completely and layer normalisation does nothing at all — yet on the *repaired*
network the two are within a point of each other.

We are not going to account for that here, because we have not measured enough
to. Below is the measurement that would start to: run both, on the broken
network, and look at where the per-layer backward factor ends up.
"""),
        prompt(
            label="the row we do not explain",
            input="the broken network with no norm, batch norm, layer norm",
            output="ρ and the end-to-end delta ratio for each",
            constraint="report the measurement without an explanation attached to it",
            check="whatever you conclude, write down the measurement that supports it. An explanation with no number attached is the thing this course is trying to replace.",
            **{"try": "run the same three rows on the repaired ReLU and He "
                      "network instead of the broken one. The three rho "
                      "values move much closer together. That is the "
                      "measurement any explanation of the unexplained row "
                      "would have to account for."}),
        code('''
for label, kw in [("none",  dict(act="sigmoid", init="torch")),
                  ("batch", dict(act="sigmoid", init="torch", norm="batch")),
                  ("layer", dict(act="sigmoid", init="torch", norm="layer"))]:
    d = delta_profile(**kw)
    print(f"{label:6s} rho {geo(d[0:DEPTH-1] / d[1:DEPTH]):.4f}   "
          f"delta_1/delta_20 {d[0]/d[DEPTH-1]:.3e}")
print()
print("Whatever you conclude, write down the measurement that supports it.")
print("An explanation with no number attached is the thing this course is")
print("trying to replace.")
'''),

        md("""
## 15 · Gradient clipping

Clipping rescales the whole gradient vector when its norm exceeds a threshold.
It is a defence against the *other* failure — the one where $\\rho > 1$.

Look at what the norms actually are before choosing a threshold. A clip value
below the median silently turns your optimiser into sign descent.
"""),
        prompt(
            label="look at the norms before choosing a threshold",
            input="two epochs of gradient norms, under He and under N(0,1)",
            output="the median and maximum for each, and both distributions on a log axis",
            constraint="use `clip_grad_norm_` with an INFINITE threshold to read the norm without clipping it — the measurement must not be the intervention",
            check="log10 of the norms, histogrammed. The two initialisations differ by orders of magnitude and a linear histogram shows one bar.",
            **{"try": "set the clip threshold to the median of the He "
                      "distribution and train with it. Almost every step is "
                      "now rescaled to the same length, which is sign descent "
                      "with extra arithmetic. What does the accuracy do?"}),
        code('''
def step_norms(epochs=2, **kw):
    torch.manual_seed(RANDOM_STATE)
    net = make_net(**kw).to(device)
    optim, lossf = torch.optim.Adam(net.parameters(), lr=LR), nn.CrossEntropyLoss()
    g = torch.Generator().manual_seed(RANDOM_STATE)
    out = []
    for _ in range(epochs):
        perm = torch.randperm(len(Xf), generator=g).to(device)
        for i in range(0, len(Xf), BATCH):
            idx = perm[i:i+BATCH]
            optim.zero_grad(); lossf(net(Xf[idx]), yf[idx]).backward()
            out.append(float(torch.nn.utils.clip_grad_norm_(
                net.parameters(), float("inf"))))
            optim.step()
    return np.array(out)

good = step_norms(act="relu", init="he")
bad  = step_norms(act="relu", init="normal1")
print(f"He      : median {np.median(good):8.3f}   max {good.max():10.3f}")
print(f"N(0,1)  : median {np.median(bad):8.3f}   max {bad.max():10.3f}")
plt.figure(figsize=(8, 3))
plt.hist(np.log10(bad + 1e-12), bins=60, alpha=0.7, label="N(0,1)")
plt.hist(np.log10(good + 1e-12), bins=60, alpha=0.7, label="He")
plt.xlabel("log10 total gradient norm, one step"); plt.legend()
plt.tight_layout(); plt.show()
'''),

        md("""
## 16 · Faster optimisers, and schedules

Now that the gradient reaches every layer, the optimiser has something to work
with. Compare six on the *repaired* network — comparing them on the broken one
would have measured nothing, which is why this section is here and not in
section 6.

⏱ **about 4 minutes.**
"""),
        prompt(
            label="⏱ 4 min — six optimisers, on the REPAIRED network",
            input="SGD, momentum, Nesterov, RMSprop, Adam, AdamW",
            output="test accuracy and final loss for each",
            constraint="different learning rates for the SGD family and the adaptive family — 1e-3 on plain SGD is not a fair test of plain SGD",
            check="fix the signal before tuning the search. An optimiser comparison on a network that cannot learn is a comparison of nothing.",
            **{"try": "run plain SGD at 1e-3, the adaptive family's rate. It "
                      "is far worse, and the table would then read 'Adam "
                      "beats SGD' when it means 'that learning rate suits "
                      "Adam'. One learning rate across six optimisers is six "
                      "unfair tests."}),
        code('''
for name, lr in [("sgd", 1e-2), ("momentum", 1e-2), ("nesterov", 1e-2),
                 ("rmsprop", 1e-3), ("adam", 1e-3), ("adamw", 1e-3)]:
    _, h = train(act="relu", init="he", norm="batch", opt=name, lr=lr)
    print(f"{name:9s} lr {lr:g}   test {h['test_acc']:.4f}   "
          f"final loss {h['loss'][-1]:.4f}")
'''),
        prompt(
            label="three schedules",
            input="the repaired network, with and without a schedule",
            output="final test accuracy and BEST validation accuracy for each",
            constraint="report the best validation as well as the final test — a schedule that ends at a low learning rate can finish below its own peak",
            check="a schedule is a hyperparameter with a shape rather than a value. Report the curve or at least its peak, not only its endpoint.",
            **{"try": "print the whole val_acc curve for the one-cycle run "
                      "rather than its maximum. The peak arrives several "
                      "epochs before the end. A schedule finishing at a low "
                      "learning rate can end below its own best, and an "
                      "endpoint alone hides that."}),
        code('''
for name in (None, "cosine", "onecycle"):
    _, h = train(act="relu", init="he", norm="batch", clip=1.0, schedule=name)
    print(f"{str(name):9s} test {h['test_acc']:.4f}   "
          f"best validation {max(h['val_acc']):.4f}")
'''),

        md("""
## 17 · Re-measure

The same twenty layers, the same 10,000 images, the same twenty epochs, the
same seed. Only the four lines that decide how the signal propagates have
changed.
"""),
        prompt(
            label="re-measure",
            input="the broken network and the BEST rung of the ladder",
            output="baseline, both accuracies, the improvement, and how many rungs were dropped",
            constraint="use BEST_KW, captured from the ladder — not the last rung's settings typed out again",
            check="when a table selects a winner, carry the winner forward in a variable. Retyping its settings is how the summary and the table drift apart.",
            **{"try": "replace BEST_KW with ladder[-1][1] and re-run. The "
                      "summary now reports the last rung and the improvement "
                      "drops by about ten points — the notebook committing, "
                      "in one substitution, the mistake the deck spends a "
                      "slide forbidding."}),
        code('''
_, base = train(act="sigmoid", init="torch")
# The BEST rung, not the last one. Hard-coding the last rung's settings here
# would report 33.4% while the slide two pages on says, in as many words, "the
# number we report is 43.9%, not 33.4%" — the notebook demonstrating the very
# mistake the deck forbids.
_, repaired = train(**BEST_KW)

print("=" * 60)
print(f"{'baseline (majority class)':34s} {baseline:.4f}")
print(f"{'as built, 20 layers':34s} {base['test_acc']:.4f}")
print(f"{'the same 20 layers, repaired':34s} {repaired['test_acc']:.4f}")
print(f"{'  (best rung: ' + BEST_LABEL + ')':34s}")
print(f"{'improvement, accuracy points':34s} "
      f"{100*(repaired['test_acc'] - base['test_acc']):+.2f}")
if BEST_KW is not ladder[-1][1]:
    print(f"{'rungs dropped':34s} "
          f"{len(ladder) - 1 - [k for _, k in ladder].index(BEST_KW)}")
print("=" * 60)
print("\\nAnd it is still an MLP on flattened pixels. Lecture 15 changes that,")
print("and the ceiling this network is running into is the subject there.")
'''),

        md("""
## 18 · Where we are

- A twenty-layer stack on default initialisation does not train, and the reason
  is measurable: the gradient shrinks by a constant factor per layer, which is
  a straight line on a log axis.
- The derivation says what that factor is — fan-in and weight variance — and
  that preserving the forward signal and the backward gradient want *different*
  variances. Glorot splits the difference; He corrects for ReLU halving it.
- The error compounds geometrically, so twenty layers multiply it twenty times.
  That is why depth, and not width, is what breaks.
- Seven repairs, each measured alone. Most of the gain is in the first two.

**Four questions to ask of a network that will not train:**

1. Is it the harness? Overfit ten examples first — if it cannot, the bug is not
   in the architecture.
2. What do the per-layer gradient norms look like on a log axis? A straight
   line is initialisation; a cliff is something else.
3. What is the activation's effect on variance, and does the initialisation
   account for it?
4. Is anything downstream — normalisation, clipping — masking the diagnosis
   rather than fixing it?

**Before the next lecture:** run this notebook top to bottom. Then set the
initialisation back to PyTorch's default and re-run the gradient probe alone.
The straight line comes back, and its slope is the per-layer factor the
derivation predicts.
"""),
    ]
    return cells
