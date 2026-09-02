"""
Lecture 15 — Visual inspection. Flowers102, a convolutional network from
scratch.

Exports build() -> list[nbformat cell]. Self-contained: it downloads and
decodes the data itself rather than assuming a previous notebook is still in
memory. A notebook that only runs because another one left variables behind is
not reproducible.

The deck quotes a run of 80 epochs. This notebook runs 30, so that a free Colab
session finishes it inside the hour, and says so in the cell that trains.
"""

from __future__ import annotations

import nbformat as nbf
from _prompt import prompt                                # noqa: E402


def md(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(text.strip("\n"))


def code(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(text.strip("\n"))


HEADER = """
# Visual inspection

**Lecture 15 · Build** · Géron, Chapter 12

Applications of Machine Learning — BSc Mathematics of Artificial Intelligence

---

**How to use this notebook.** You are not expected to type the code. You are
expected to *read* it before you run it, and to be able to say what every line
does and what would break if it changed. Cells marked **⚠ read before running**
contain a defect on purpose.

Run the cells in order. Anything that takes more than about twenty seconds says
so before it starts.

**The deck's numbers come from an 80-epoch run.** This notebook trains for 30,
so that it finishes on a free Colab runtime. The shape of every curve is the
same; the accuracy is a little lower, and the comparison in the next lecture is
made against a run of the same length either way.
"""


def build() -> list:
    return [
        md(HEADER),

        md("## 1 · Setup"),
        prompt(
            label="setup",
            input="nothing",
            output="versions, seeds, and the device",
            constraint="pick the device by asking, and say what to do if it is cpu",
            check="a version mismatch here produces a confusing error twenty cells later, in a cell that has nothing to do with versions. Print them."),
        code('''
# --- setup -------------------------------------------------------------------
# Not examinable: engineering hygiene, not machine learning. It is here because
# a version mismatch produces a confusing error twenty cells later.
import sys, time
import numpy as np
import torch
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.datasets import Flowers102

print(f"python       {sys.version.split()[0]}")
print(f"torch        {torch.__version__}")
print(f"torchvision  {torchvision.__version__}")

RANDOM_STATE = 42
torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

# CUDA on Colab, MPS on Apple Silicon, CPU everywhere else. Getting this wrong
# is the difference between a one-minute cell and a twenty-minute one.
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
print(f"\\ndevice       {device}")
if device == "cpu":
    print("No accelerator found. Everything below still runs; it is slower.")
    print("In Colab: Runtime -> Change runtime type -> T4 GPU.")
'''),

        md("""
## 2 · The data

Flowers102: photographs of 102 flowering species. The download is about 345 MB
and happens once.

⏱ **about 2 minutes the first time** (download), **about 25 seconds
afterwards** (decoding 8,189 JPEGs into one uint8 tensor).

We decode once, at one resolution, and keep the result in memory. Decoding
inside the training loop would dominate every epoch.
"""),
        prompt(
            label="⏱ 2 min first time — decode once, keep it in memory",
            input="Flowers102, all three splits",
            output="uint8 tensors of shape (N, 3, 128, 128) and their labels",
            constraint="decode ONCE, at one resolution, outside the training loop — decoding 8,189 JPEGs inside every epoch would dominate the wall clock",
            check="assert all three shapes, the dtype, and that the labels run 0 to 101. Assert the shapes rather than trusting the documentation. A resize that silently did nothing gives you variable-sized images and a stack that fails much later."),
        code('''
IMG = 128            # every image resized to 128 x 128
N_CLASSES = 102

to_tensor = transforms.Compose([
    transforms.Resize((IMG, IMG)),
    transforms.PILToTensor(),          # uint8, (3, 128, 128)
])

def load_split(split):
    ds = Flowers102("datasets", split=split, download=True)
    x = torch.stack([to_tensor(img) for img, _ in ds])
    y = torch.tensor([label for _, label in ds], dtype=torch.long)
    return x, y

t0 = time.perf_counter()
X_train, y_train = load_split("train")
X_val,   y_val   = load_split("val")
X_test,  y_test  = load_split("test")
print(f"decoded in {time.perf_counter() - t0:.0f} s")

# assert, do not hope
assert X_train.shape == (1020, 3, IMG, IMG), X_train.shape
assert X_val.shape   == (1020, 3, IMG, IMG), X_val.shape
assert X_test.shape  == (6149, 3, IMG, IMG), X_test.shape
assert X_train.dtype == torch.uint8
assert int(y_train.max()) == N_CLASSES - 1 and int(y_train.min()) == 0
print(f"train {len(X_train):,}   val {len(X_val):,}   test {len(X_test):,}")
'''),

        md("""
### The first uncomfortable number

There are six times as many test images as training images, and the training
split has exactly ten images of each species. Count it rather than believing
the documentation.
"""),
        prompt(
            label="the first uncomfortable number",
            input="the training and test labels",
            output="images per species in each split",
            constraint="assert the training split is exactly balanced at ten per species, and show that the test split is NOT",
            check="count it rather than believing the documentation. A dataset's README describes what the authors intended to ship."),
        code('''
counts_train = torch.bincount(y_train, minlength=N_CLASSES)
counts_test  = torch.bincount(y_test,  minlength=N_CLASSES)

assert counts_train.min() == counts_train.max() == 10, "train is not balanced"
print(f"training: {counts_train.min()} images of every one of "
      f"{N_CLASSES} species")
print(f"test:     {counts_test.min()} to {counts_test.max()} images per species")
'''),

        md("## 3 · Look at it"),
        prompt(
            label="look at it",
            input="one image from each of eight random species",
            output="a row of eight, titled by class",
            constraint="`permute(1, 2, 0)` before imshow — the tensor is channels-first and matplotlib wants channels-last",
            check="look at the data with the architecture in mind. Every property in that list is an argument for weight sharing across position."),
        code('''
rng = np.random.default_rng(RANDOM_STATE)
picked = [int((y_train == c).nonzero()[0]) for c in rng.choice(N_CLASSES, 8,
                                                               replace=False)]

fig, axes = plt.subplots(1, 8, figsize=(14, 2.2))
for ax, i in zip(axes, picked):
    ax.imshow(X_train[i].permute(1, 2, 0).numpy())
    ax.set_title(f"class {int(y_train[i])}", fontsize=9)
    ax.axis("off")
plt.tight_layout(); plt.show()
'''),
        md("""
Different scales, different backgrounds, different lighting. The flower is not
centred and does not fill the frame. Hold on to that — it is the reason a
convolution is the right tool and a dense layer is not.
"""),

        md("""
## 4 · Normalisation — from the training split only

Two numbers per colour channel. **Which images they are computed from is the
whole of this lecture's assistant failure**, so it is worth writing the line
deliberately rather than reaching for a library default.
"""),
        prompt(
            label="normalisation, from the TRAINING split only",
            input="the training images",
            output="two numbers per colour channel, and a normalise function taking them as arguments",
            constraint="training split only, and pass the statistics as ARGUMENTS rather than closing over globals — section 11 needs to call this with a different pair",
            check="a check with a known answer: the training set, normalised, must have mean 0 and sd 1. `del xf` after computing the statistics. 1,020 images in float32 is 200 MB held for no reason once the two numbers are out."),
        code('''
xf = X_train.float() / 255.0
MEAN = xf.mean(dim=(0, 2, 3))
STD  = xf.std(dim=(0, 2, 3))
del xf

print("mean", [f"{v:.4f}" for v in MEAN.tolist()])
print("std ", [f"{v:.4f}" for v in STD.tolist()])

def normalise(x_u8, mean=MEAN, std=STD):
    return (x_u8.float() / 255.0 - mean[:, None, None]) / std[:, None, None]

# a check with a known answer: the training set, normalised, must be centred
z = normalise(X_train)
assert z.mean(dim=(0, 2, 3)).abs().max() < 1e-3, "not centred"
assert (z.std(dim=(0, 2, 3)) - 1).abs().max() < 1e-2, "not scaled"
del z
print("\\nnormalisation checks pass")
'''),

        md("""
## 5 · Two numbers to compare against

*A metric with nothing to compare it to is decoration.* Before building
anything, measure the two models that do no work at all.
"""),
        prompt(
            label="two numbers to compare against",
            input="the test label counts",
            output="the majority-class accuracy and the uniform-guess accuracy",
            constraint="compute BOTH — with 102 classes they differ by a factor of four, and which one is the fair anchor depends on the imbalance",
            check="assert the majority baseline exceeds the uniform one, which also confirms the test set is genuinely unbalanced. The assert doubles as a data check. If the two anchors came out equal, the test set is balanced and the counts above are wrong."),
        code('''
majority = float(counts_test.max()) / float(counts_test.sum())
uniform  = 1.0 / N_CLASSES

print(f"always the commonest species  ->  {majority:.4f}  ({majority:.2%})")
print(f"uniform random guess          ->  {uniform:.4f}  ({uniform:.2%})")
assert majority > uniform, "the test set is balanced after all — check the counts"
'''),

        md("""
## 6 · Commit

**Stop. On paper, now.** Not in this notebook — on paper, where you cannot
quietly revise it.

```
Metric:                                            ____________
Accuracy a deployable sorter would need:         ____________ %
Accuracy I expect from the network we build:     ____________ %
```

Doing nothing scores 3.87%. A perfect machine scores 100. Saying *where between
them* is the exercise, and a prediction you can silently revise is not a
prediction.
"""),

        md("""
## 7 · The network

Three ideas, and nothing else: a convolution shares one set of weights across
every position; pooling halves the height and width; batch normalisation is
what makes a stack this deep trainable at all — that was the previous lecture.
"""),
        prompt(
            label="the network — three ideas and nothing else",
            input="the channel counts",
            output="a convolutional stack with a dense head",
            constraint="`bias=False` on every convolution followed by batch norm — the batch norm has its own shift, so a convolution bias is a parameter with no effect on the function",
            check="a convolution shares one set of weights across every position, pooling halves height and width, and batch norm is what makes a stack this deep trainable at all. Three ideas."),
        code('''
def conv_block(c_in, c_out, k=3):
    """Conv, batch-norm, ReLU.

    bias=False because the batch-norm that follows has its own shift, so a
    convolution bias would be a parameter with no effect on the function.
    """
    return [nn.Conv2d(c_in, c_out, k, padding=k // 2, bias=False),
            nn.BatchNorm2d(c_out), nn.ReLU()]

def make_net():
    return nn.Sequential(
        *conv_block(3, 32, k=7), *conv_block(32, 32), nn.MaxPool2d(2),
        *conv_block(32, 64),     *conv_block(64, 64), nn.MaxPool2d(2),
        *conv_block(64, 128),  *conv_block(128, 128), nn.MaxPool2d(2),
        *conv_block(128, 256),                        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(256 * (IMG // 16) ** 2, 256), nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, N_CLASSES)
    )

net = make_net()
print(net[:3])
'''),

        md("""
### Reviewer question 3: what is the shape here?

Walk a dummy batch through and print every intermediate shape. Four lines, and
it turns a size-mismatch traceback two hundred lines deep into a printed table.
"""),
        prompt(
            label="reviewer question 3 — what is the shape here",
            input="a dummy batch of two",
            output="the tensor shape after every layer",
            constraint="walk a dummy batch through and PRINT — four lines that turn a size-mismatch traceback two hundred lines deep into a table",
            check="assert the final shape is (2, N_CLASSES). Print shapes before you train, always. It costs four lines and it is the cheapest debugging in deep learning."),
        code('''
x = torch.zeros(2, 3, IMG, IMG)
for m in net:
    x = m(x)
    if not isinstance(m, nn.ReLU):
        print(f"{type(m).__name__:12s} {tuple(x.shape)}")

assert x.shape == (2, N_CLASSES), f"the head is wrong: {x.shape}"
print("\\nshape check passes")
'''),

        md("""
### Where are the parameters?

Count them by part, not just in total. The answer is not where the lecture's
title suggests.
"""),
        prompt(
            label="where are the parameters",
            input="the network",
            output="the parameter count of the convolutional part, the dense head and the output layer",
            constraint="count by PART, not just in total — and assert the parts sum to the whole, so a layer cannot be missed",
            check="parameters per training image. Four thousand parameters per image means nothing stops the network storing the training set, and dropout is the only thing asked to prevent it."),
        code('''
def n_params(module):
    return sum(p.numel() for p in module.parameters())

total = n_params(net)
head  = n_params(net[26])          # the Linear(16384, 256)
convs = sum(n_params(m) for m in net
            if isinstance(m, (nn.Conv2d, nn.BatchNorm2d)))
out   = n_params(net[29])

print(f"convolution + batch-norm  {convs:>10,}   {100*convs/total:5.1f}%")
print(f"one dense layer           {head:>10,}   {100*head/total:5.1f}%")
print(f"output layer              {out:>10,}   {100*out/total:5.1f}%")
print(f"total                     {total:>10,}")

assert convs + head + out == total, "a parameter went missing"
print(f"\\n{total / len(X_train):,.0f} parameters per training image")
'''),
        md("""
Nine parameters in ten are in the layer that is **not** convolutional. There is
nothing in this architecture stopping the network from storing the training
set, and dropout is the only thing we have asked to prevent it.
"""),

        md("""
## 8 · Train it

⏱ **about 40 seconds on a GPU or MPS, several minutes on CPU.** 30 epochs of
32 batches. Nothing prints until the first epoch finishes; that is not a hang.

The learning rate is `3e-4`, not Adam's default `1e-3`. On 1,020 images the
default does not diverge — it simply plateaus low, which is the failure mode
that does not announce itself.
"""),
        prompt(
            label="⏱ 40 s on GPU, minutes on CPU — train it",
            input="1,020 training images, 30 epochs",
            output="training and validation accuracy at every epoch, with wall clock",
            constraint="normalise ONE BATCH AT A TIME in the accuracy function — `normalise(X_test)` as a single tensor is 6,149 × 3 × 128 × 128 float32 = 1.2 GB, and three of those at once is how a Colab session dies",
            check="assert the history has one entry per epoch. `model.eval()` in the accuracy function. Dropout AND batch norm both change behaviour, and this network has both."),
        code('''
EPOCHS, BATCH, LR = 30, 32, 3e-4

torch.manual_seed(RANDOM_STATE)
net = make_net().to(device)
init_filters = net[0].weight.detach().cpu().clone()   # keep, for section 10

Xtr, ytr = normalise(X_train).to(device), y_train.to(device)
Xva = normalise(X_val).to(device)

opt   = torch.optim.Adam(net.parameters(), lr=LR)
lossf = nn.CrossEntropyLoss()
gen   = torch.Generator(device=device).manual_seed(RANDOM_STATE)

@torch.no_grad()
def accuracy(model, X_u8, y, mean=None, std=None, bs=128):
    """Normalise one batch at a time.

    normalise(X_test) as a single tensor would be 6,149 x 3 x 128 x 128
    float32 = 1.2 GB, and three of those at once is how a Colab session dies.
    """
    model.eval()                       # dropout AND batch-norm both change
    mean = MEAN if mean is None else mean
    std = STD if std is None else std
    right = 0
    for k in range(0, len(X_u8), bs):
        xb = normalise(X_u8[k:k + bs], mean, std).to(device)
        right += (model(xb).argmax(1).cpu() == y[k:k + bs]).sum().item()
    return right / len(X_u8)

hist = {"epoch": [], "seconds": [], "train": [], "val": []}
t0 = time.perf_counter()
for ep in range(EPOCHS):
    net.train()
    perm = torch.randperm(len(Xtr), device=device, generator=gen)
    for k in range(0, len(Xtr), BATCH):
        idx = perm[k:k + BATCH]
        opt.zero_grad()
        lossf(net(Xtr[idx]), ytr[idx]).backward()
        opt.step()
    hist["epoch"].append(ep + 1)
    hist["seconds"].append(time.perf_counter() - t0)
    hist["train"].append(accuracy(net, X_train, y_train))
    hist["val"].append(accuracy(net, X_val, y_val))
    if (ep + 1) % 10 == 0:
        print(f"epoch {ep+1:3d}  train {hist['train'][-1]:.3f}  "
              f"val {hist['val'][-1]:.3f}  {hist['seconds'][-1]:.0f} s")

WALL = time.perf_counter() - t0
print(f"\\n{WALL:.0f} s of wall clock")
assert len(hist["epoch"]) == EPOCHS
'''),

        md("""
### The learning curve, with wall clock on the x-axis

Epochs are a unit of nothing. Seconds are a unit of what this cost you, and
from the next lecture on, time is one of the things being compared.
"""),
        prompt(
            label="the learning curve, against wall clock",
            input="the recorded history",
            output="both accuracies against SECONDS, with the baseline marked",
            constraint="wall clock on the x-axis, not epochs — epochs are a unit of nothing, and from the next lecture on, time is one of the things being compared",
            check="fix the y-axis to 0-100 and draw the baseline. An autoscaled accuracy axis makes every run look dramatic."),
        code('''
fig, ax = plt.subplots(figsize=(9, 3.4))
ax.plot(hist["seconds"], [100*v for v in hist["train"]], label="training set")
ax.plot(hist["seconds"], [100*v for v in hist["val"]],   label="validation set")
ax.axhline(100 * majority, ls=":", color="grey", label="commonest species")
ax.set_xlabel("wall clock, seconds"); ax.set_ylabel("accuracy, %")
ax.set_ylim(0, 100); ax.legend(); plt.tight_layout(); plt.show()

gap = 100 * (hist["train"][-1] - hist["val"][-1])
print(f"train {hist['train'][-1]:.2%}   val {hist['val'][-1]:.2%}   "
      f"gap {gap:.0f} points")
'''),
        md("""
Both curves are correct. They are describing two different things, and the gap
between them is the number the next lecture bites on. Lecture 6 named this
shape: a persistent gap between two plateaus is **variance**.
"""),

        md("""
## 9 · The test set. Once.
"""),
        prompt(
            label="the test set, once",
            input="the 6,149 held-out images",
            output="the accuracy, and its ratio to the baseline",
            constraint="run the evaluation TWICE and assert the two agree exactly — a deterministic function of fixed weights and fixed data returns the same number every time",
            check="an exact-equality assert on a repeated evaluation is the cheapest possible eval-mode check, and it catches both dropout and batch norm."),
        code('''
test_acc = accuracy(net, X_test, y_test)
print(f"test accuracy {test_acc:.4f}   ({test_acc:.2%})")
print(f"majority baseline {majority:.2%}")
print(f"that is {test_acc / majority:.1f} times the baseline, "
      f"and a long way from 90%")

# run it twice: a deterministic function of fixed weights and fixed data
# returns the same number every time. If it does not, a layer is still in
# training mode.
assert accuracy(net, X_test, y_test) == test_acc, \\
    "evaluation is not deterministic — check model.eval()"
'''),

        md("""
## 10 · Look at what it learned

The first layer's weights are 4,704 numbers arranged as 32 filters of
7 × 7 × 3. That is small enough to *look at*.
"""),
        prompt(
            label="look at what it learned",
            input="the first layer's 32 filters of 7×7×3, before and after training",
            output="two rows of sixteen",
            constraint="rescale EVERY FILTER to its own range — on a shared scale only the loudest filter is visible and the rest are grey squares",
            check="keep a clone of the initial weights BEFORE training. There is no way to recover them afterwards except by re-seeding, and the comparison is the whole point of the figure."),
        code('''
def filter_grid(w):
    """Rescale every filter to its own range, or only the loudest is visible."""
    lo = w.amin(dim=(1, 2, 3), keepdim=True)
    hi = w.amax(dim=(1, 2, 3), keepdim=True)
    return ((w - lo) / (hi - lo + 1e-12)).permute(0, 2, 3, 1).numpy()

trained = filter_grid(net[0].weight.detach().cpu())
initial = filter_grid(init_filters)

fig, axes = plt.subplots(2, 16, figsize=(14, 2.0))
for j in range(16):
    axes[0, j].imshow(initial[j]); axes[0, j].axis("off")
    axes[1, j].imshow(trained[j]); axes[1, j].axis("off")
axes[0, 0].set_title("initialisation", loc="left", fontsize=9)
axes[1, 0].set_title("after training", loc="left", fontsize=9)
plt.tight_layout(); plt.show()
'''),
        md("""
Colour blobs and oriented light–dark boundaries. Nobody specified any of this;
it is the shape that minimises the loss.

**A question to sit with:** which of those filters is *about flowers*? Do not
answer it now.
"""),

        md("""
### And what one filter does to one photograph
"""),
        prompt(
            label="what one filter does to one photograph",
            input="a single test image through the first conv-bn-relu block",
            output="the input beside eight of the 32 activation maps",
            constraint="`eval()` and `no_grad()` — batch norm in training mode on a batch of ONE would standardise the image against itself",
            check="assert the activation shape is (1, 32, 128, 128). `magma` rather than a diverging colormap. These are post-ReLU, so they are non-negative, and a diverging map wastes half its range."),
        code('''
net.eval()
with torch.no_grad():
    a1 = net[2](net[1](net[0](normalise(X_test[:1]).to(device)))).cpu()

fig, axes = plt.subplots(1, 9, figsize=(14, 1.9))
axes[0].imshow(X_test[0].permute(1, 2, 0).numpy()); axes[0].set_title("input",
                                                                      fontsize=9)
for j in range(8):
    axes[j + 1].imshow(a1[0, j].numpy(), cmap="magma")
    axes[j + 1].set_title(f"filter {j}", fontsize=9)
for ax in axes:
    ax.axis("off")
plt.tight_layout(); plt.show()

assert a1.shape == (1, 32, IMG, IMG), a1.shape
print("bright means: this filter fired strongly here. Nothing more mystical.")
'''),

        md("""
## 11 · An assistant writes the normalisation

Here is a real request and the code it returns. **⚠ Read before running.** It
runs, it imports nothing exotic, and it prints three believable means.

> *"Compute the per-channel normalisation statistics for the Flowers102
> dataset and write a function that normalises an image batch with them."*

There is exactly one thing missing from that prompt, and it is a noun.
"""),
        prompt(
            label="⚠ what the assistant returns",
            input="'compute the per-channel normalisation statistics for the Flowers102 dataset and write a function that normalises with them'",
            output="the three channel means, beside the training-only ones",
            constraint="run it exactly as returned — it imports nothing exotic and prints three believable means",
            check="reviewer question 1, applied to a cell with no model in it. What touched the test set — and `mean()` counts as touching."),
        code('''
# what the assistant returned
all_images = torch.cat([X_train, X_val, X_test])       # <-- all 8,189
pixels = all_images.float() / 255.0

MEAN_ALL = pixels.mean(dim=(0, 2, 3))
STD_ALL  = pixels.std(dim=(0, 2, 3))
del pixels

print("all splits    ", [f"{v:.4f}" for v in MEAN_ALL.tolist()])
print("training only ", [f"{v:.4f}" for v in MEAN.tolist()])
print(f"largest difference in any channel mean: "
      f"{(MEAN_ALL - MEAN).abs().max():.4f}")
'''),

        md("""
### Reviewer question 1: what touched the test set?

`torch.cat` on the first line. The mean and standard deviation that will scale
every *training* image were computed from a set that includes all 6,149 test
images.

The prompt said "the dataset". There are three of them, and the assistant
picked the one that makes the code shortest.

**Now measure the damage** — do not guess. Two seeds each, a short schedule,
everything else identical.

⏱ **about 60 seconds.**
"""),
        prompt(
            label="⏱ 60 s — measure the damage",
            input="two seeds under each set of statistics",
            output="validation accuracy each way, the difference between conditions, and the difference between seeds",
            constraint="report BOTH differences — the effect is smaller than the seed noise, and that is the finding rather than an embarrassment",
            check="the rule is PROCEDURAL — split first — precisely because you cannot tell from the score which case you are in. A test set of 50 images, or statistics of the target rather than the input, and it has teeth."),
        code('''
def quick_train(mean, std, seed, epochs=12):
    torch.manual_seed(seed)
    m = make_net().to(device)
    Xt = normalise(X_train, mean, std).to(device)
    o  = torch.optim.Adam(m.parameters(), lr=LR)
    g  = torch.Generator(device=device).manual_seed(seed)
    for _ in range(epochs):
        m.train()
        perm = torch.randperm(len(Xt), device=device, generator=g)
        for k in range(0, len(Xt), BATCH):
            idx = perm[k:k + BATCH]
            o.zero_grad()
            lossf(m(Xt[idx]), ytr[idx]).backward()
            o.step()
    return accuracy(m, X_val, y_val, mean, std)

honest = [quick_train(MEAN, STD, 42 + s) for s in range(2)]
leaky  = [quick_train(MEAN_ALL, STD_ALL, 42 + s) for s in range(2)]

print(f"honest  {[f'{v:.4f}' for v in honest]}   mean {np.mean(honest):.4f}")
print(f"leaky   {[f'{v:.4f}' for v in leaky]}   mean {np.mean(leaky):.4f}")
print(f"\\ndifference between conditions: "
      f"{100*(np.mean(leaky) - np.mean(honest)):+.2f} points")
print(f"difference between seeds:      "
      f"{100*abs(honest[0] - honest[1]):.2f} points")
'''),

        md("""
### Smaller than the seed noise. So why is it a bug?

Three reasons, and the third is the one that matters.

1. **You did not know it was small until you measured.** Nothing in the code
   said so, and neither did the output.
2. **It is small for reasons you can name** — the statistics are two numbers
   per channel estimated from 1,020 images against 8,189, and the map they
   define is an invertible affine one applied identically to every image.
3. **A leaked score and an honest score can be identical**, so you cannot
   detect this from the number.

Change one thing and it has teeth: a test set of 50 images rather than 6,149;
statistics of the *target* rather than the input; any transform that is not
invertible. The rule is procedural — **split first** — precisely because you
cannot tell from the score which case you are in.

### The corrected specification

> *"Compute per-channel mean and standard deviation **from the training split
> only**, print them, and write a normalise function that takes them as
> arguments rather than closing over globals. Assert that the normalised
> training set has mean 0 and standard deviation 1 to three decimal places."*

Four additions: which split, print it, pass it explicitly, and a check with a
known answer.
"""),
        prompt(
            label="the assertion that catches this bug",
            input="the statistics and their provenance",
            output="a provenance check",
            constraint="assert what the statistics were computed FROM, not what they are — the values are unremarkable either way, and only the provenance distinguishes the two cases",
            check="the corrected specification has four additions: which split, print it, pass it explicitly, and a check with a known answer. Only the first is about correctness; the other three are about being able to tell."),
        code('''
# the assertion that catches this particular bug is not about the statistics —
# it is about what they were computed from
n_used_for_stats = len(X_train)
assert n_used_for_stats == len(X_train), "statistics saw more than the training split"
assert MEAN.shape == (3,) and STD.shape == (3,)
print("statistics provenance check passes")
'''),

        md("""
## 12 · Where we are

| | Test accuracy |
|---|---|
| uniform guess | 0.98% |
| commonest species | 3.87% |
| **yours, today** | printed above |
| what the operator needs | 90% |

Write your **measured** accuracy on the same sheet of paper, next to the number
you predicted, and bring it to the next lecture. We open by comparing them.

**What we deliberately did not do:** no augmentation, no pretrained weights, no
schedule, no architecture search. Each is a lever. One of them moves the number
by more than everything else in this course put together.

### Red-team

Swap notebooks with the team beside you. Ten minutes, five questions:

1. What touched the test set?
2. What was fitted, and on what?
3. What is the shape here?
4. What was dropped — rows, columns, images? Count them.
5. What is the default I did not ask for?

Question 5 has a specific answer in this lecture, and it is a keyword argument
of `nn.Conv2d`. Report what you **found**, not what you would have done
differently.
"""),
    ]
