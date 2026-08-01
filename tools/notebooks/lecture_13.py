"""
Lecture 13 — Twenty layers, no learning.

Build. CIFAR-10, Géron Chapter 11.

Exports build() -> list[nbformat cell]. Self-contained: it downloads and splits
the data itself rather than assuming Lecture 12's kernel is still alive. A
notebook that only runs because a previous one left variables in memory is not
reproducible.

The notebook deliberately builds a network that does not work, and then
instruments it. Nothing here is repaired — that is the next lecture.
"""

from __future__ import annotations

import nbformat as nbf
from _prompt import prompt                                # noqa: E402


def md(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(text.strip("\n"))


def code(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(text.strip("\n"))


HEADER = """
# Twenty layers, no learning

**Lecture 13 · Build** · Géron, Chapter 11

Applications of Machine Learning — BSc Mathematics of Artificial Intelligence

---

**How to use this notebook.** Read before you run. You are not expected to type
the code; you are expected to be able to say what every line does and what
would break if it changed.

Today you build a network that **does not work**, and then you measure why. Do
not fix anything. The measurement is the deliverable, and the repair is the
next lecture.

We train on 10,000 of the 50,000 images so that a free Colab runtime finishes
inside the hour. Every comparison below uses the same 10,000, the same seed and
the same number of epochs, so the rows can be read against one another.
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
            left_open="that a device mismatch produces a confusing error twenty cells later, in a cell that has nothing to do with devices.",
            student="hard-coding a device from a tutorial. It raises on the wrong hardware, or silently leaves an accelerator idle for the whole lecture.",
            catch="everything below still runs on cpu; it is slower. Say so, or a student with no GPU will read the wall clocks as a bug."),
        code('''
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
    print("no accelerator found. Everything runs; it is slower.")
    print("In Colab: Runtime -> Change runtime type -> T4 GPU.")
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
            check="assert both shapes and both balance conditions",
            left_open="that one image is 32×32×3 = 3,072 numbers. That count is the reason the first weight matrix dwarfs the other twenty.",
            student="assuming CIFAR-10 is shaped like MNIST. It is (N, 32, 32, 3), channels LAST, and a reshape that assumes channels-first produces 3,072 numbers in the wrong order — which trains, badly, and never raises.",
            catch="`download=True` checks before it fetches, so re-running is instant. A cell that re-downloads 170 MB every run is a cell people stop running."),
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
            left_open="two things to notice: the objects are not centred or aligned the way the garments in the previous application were, and the backgrounds vary enormously. A pixel at a fixed position means much less here.",
            student="skipping the picture. This is the cell that tells you why flattening hurts more on CIFAR than on Fashion MNIST, before any model runs.",
            catch="look at the data with the model's assumptions in mind. We are about to flatten to 3,072 numbers, and the grid shows exactly what that throws away."),
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
            check="assert the fit subset is standardised, and print the validation mean and sd, which will NOT be exactly 0 and 1",
            left_open="why the validation mean should not be exactly zero. Those statistics came from a different set of images — a pipeline in which every split has mean exactly zero is a pipeline that fitted the scaler on everything.",
            student="standardising before splitting, which makes all three means exactly zero and looks tidier. That tidiness is the leak.",
            catch="the `+ 1e-7` on the standard deviation. CIFAR has no constant pixels, but a dataset with one gives a divide-by-zero that propagates NaN through every layer and shows up as a loss of nan five cells later."),
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
            left_open="the three conditions under which plain accuracy is defensible: balanced classes, ten of them, and no class more expensive to get wrong than another. All three hold here and the previous application was about what happens when they do not.",
            student="not computing ln(10) = 2.3026. The training loss in this notebook sits at that value for twenty epochs, formatted to four decimal places, and without the anchor it reads as a number that is going down.",
            catch="a loss anchor is as important as an accuracy anchor and almost nobody prints one. ln(k) for k balanced classes, one line, before anything trains."),
        code('''
counts = np.bincount(y_test, minlength=10)
baseline = counts.max() / counts.sum()
print("test images per class:", counts.tolist())
print(f"\\nalways predict the commonest class -> accuracy {baseline:.4f}")
print(f"the loss of a model that has learned nothing: "
      f"ln(10) = {math.log(10):.4f}")
'''),

        md("""
## 5 · Commit

**Stop. On paper, now.** Not in this notebook, where you can quietly revise it.

```
Metric:                                          ____________
Accuracy a useful auto-tagger would need:      % ____________
Accuracy I expect from the model we build today: % ____________
```

A prediction you can silently revise is not a prediction. Bring the sheet to
the next lecture; we score it out loud.
"""),

        md("""
## 6 · Build the stack

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
            check="assert there are DEPTH+1 weight matrices, one per layer plus the head",
            left_open="what the specification does NOT say: anything about how the weights start out. That is not an omission being hidden — it is the ordinary case, and it is the subject of the next lecture.",
            student="assuming twenty layers means twenty times the capacity. The parameter count says otherwise, and the accuracy says otherwise again.",
            catch="count the weight matrices with an assert rather than trusting the loop. An off-by-one in a layer-building loop gives a network that trains and is not the one you described."),
        code('''
DEPTH, WIDTH, N_IN, N_OUT = 20, 100, 3072, 10

def make_net(depth=DEPTH, width=WIDTH, act=nn.Sigmoid, n_in=N_IN, n_out=N_OUT):
    layers, prev = [], n_in
    for _ in range(depth):
        layers += [nn.Linear(prev, width), act()]
        prev = width
    layers.append(nn.Linear(prev, n_out))
    return nn.Sequential(*layers)

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
            check="assert the measured sd matches to within 0.002",
            left_open="that nobody chose this. It is U(−1/√fan_in, +1/√fan_in), scikit-learn's PyTorch equivalent of a default nobody asked for, and it is the direct cause of everything that follows.",
            student="never looking. The initialisation is the single most important unstated choice in this notebook and it takes four lines to read it off.",
            catch="reviewer question 5 applied to weights. Any tensor you did not fill yourself was filled by someone, according to a rule you can look up and check."),
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
## 7 · Train it

The loop is Lecture 12's, unchanged, with the three defences that lecture
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
            left_open="`track_grads=True`, which records the per-layer gradient norm every epoch. Nothing uses it until section 10, and it costs one line to collect and cannot be collected retroactively.",
            student="not instrumenting the first run, then having to retrain to diagnose. Log the cheap thing while you have the chance.",
            catch="print the loss at epoch 1 and at epoch 20 beside ln(10). Three numbers, and they say the whole thing before any plot."),
        code('''
EPOCHS, BATCH, LR = 20, 128, 1e-3

Xf = torch.tensor(X_fit,  device=device); yf = torch.tensor(y_fit,  device=device)
Xv = torch.tensor(X_val,  device=device); yv = torch.tensor(y_val,  device=device)
Xt = torch.tensor(X_test, device=device); yt = torch.tensor(y_test, device=device)

@torch.no_grad()
def accuracy(net, X, y, batch=2000):
    """Counted over the whole set. Lecture 12, section 11."""
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
            left_open="that a flat loss at ln(10) is the network telling you it has learned nothing, and it looks identical to a network that is learning slowly.",
            student="letting matplotlib autoscale. A y-axis from 0.0998 to 0.1013 turns pure noise into a trend, and it is the most common way a true plot lies.",
            catch="fix the axis limits when you are showing that something did NOT happen. Autoscale is for exploring; a fixed axis is for claiming."),
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
## 8 · Before blaming the architecture, rule out the bugs

Lecture 12 ended with a checklist of failures that run, produce a plausible
number and never raise. All of them would look exactly like this. Check them
rather than assuming.
"""),
        prompt(
            label="rule out the bugs before blaming the architecture",
            input="the checklist from the previous lecture",
            output="a label-image alignment check, and a deliberate overfit of 200 images",
            constraint="check that the SAME loop can memorise 200 images with 2 layers — if it cannot, the loop is the bug and the depth is irrelevant",
            left_open="that all four of the previous lecture's silent failures would look exactly like this result. Checking them rather than assuming is what separates a diagnosis from a guess.",
            student="concluding 'deep networks are hard' and moving on. Every failure mode on that checklist produces a plausible number and never raises, and three of them are one line each to rule out.",
            catch="the overfit test is the strongest single diagnostic in deep learning. A loop that cannot memorise a tiny sample is broken; a loop that can is not the reason your model does not learn."),
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
            check="assert depth 2 beats depth 20 — if it does not, depth is not the variable and the rest of the lecture is about the wrong thing",
            left_open="that adding layers made it WORSE. That is not what capacity is supposed to do, and the rest of the notebook is about why.",
            student="testing only the failing configuration. One number is a result; two numbers that differ in one variable is a finding.",
            catch="an assert that encodes the premise of the lecture. If it fires, stop and re-diagnose rather than continuing to instrument the wrong thing."),
        code('''
sweep = {}
for k in (1, 2, 5, 10, 20):
    torch.manual_seed(RANDOM_STATE)
    m, h = train(make_net(depth=k))
    sweep[k] = accuracy(m, Xt, yt)
    print(f"depth {k:2d}: final loss {h['loss'][-1]:.4f}   "
          f"test accuracy {sweep[k]:.4f}")

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
## 9 · ⚠ An assistant writes the network

Here is a real request and the code it returns. It runs. It trains without
error. It reports a number.

> *"Write me a deep PyTorch classifier for CIFAR-10 with 20 hidden layers and
> train it for a few epochs."*

**Read before running.**
"""),
        prompt(
            label="⚠ what the assistant returns",
            input="'write me a deep PyTorch classifier for CIFAR-10 with 20 hidden layers and train it for a few epochs'",
            output="a trained model and its test accuracy",
            constraint="run it exactly as returned — the zero_grad, the eval() and the whole-set metric are ALL there, and a review looking for the previous lecture's failures finds nothing",
            left_open="the review question that does catch it, and it is not 'does it run'. It is: what would this number be if the model had learned nothing at all?",
            student="accepting it, because it is correct code. Every line is defensible and the result is indistinguishable from guessing — the defect is an absence, not a mistake.",
            catch="'Epoch 5, Loss: 2.3026' is five decimal places of nothing, formatted to look like progress. Anchor every loss you print."),
        code('''
# --- what the assistant returned ---------------------------------------------
class DeepClassifier(nn.Module):
    def __init__(self, input_dim=3072, hidden_dim=100, num_layers=20,
                 num_classes=10):
        super().__init__()
        layers = []
        for i in range(num_layers):
            layers.append(nn.Linear(input_dim if i == 0 else hidden_dim,
                                    hidden_dim))
            layers.append(nn.Sigmoid())
        layers.append(nn.Linear(hidden_dim, num_classes))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

torch.manual_seed(RANDOM_STATE)
model = DeepClassifier().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

for epoch in range(5):
    perm = torch.randperm(len(Xf), device=device)
    running = 0.0
    for i in range(0, len(Xf), 128):
        idx = perm[i:i+128]
        optimizer.zero_grad()
        loss = criterion(model(Xf[idx]), yf[idx])
        loss.backward()
        optimizer.step()
        running += loss.item()
    print(f"Epoch {epoch+1}, Loss: {running/(len(Xf)//128):.4f}")

model.eval()
with torch.no_grad():
    acc = (model(Xt).argmax(1) == yt).float().mean().item()
print(f"Test Accuracy: {acc*100:.2f}%")
'''),
        md("""
### The review question that catches it

Not *"does it run?"* — it does. Not *"is the loop correct?"* — it is; the
`zero_grad`, the `eval()` and the whole-set metric are all there, and a review
looking for Lecture 12's failures finds nothing.

The question is the one this course asks of every number:

> **What would this number be if the model had learned nothing at all?**

Ten balanced classes: 10%. Compare that with what it printed.
"""),
        prompt(
            label="the comparison the assistant did not make",
            input="its accuracy and the baseline",
            output="the difference, in accuracy points",
            constraint="print the difference explicitly rather than leaving the reader to subtract",
            left_open="that the loss never left ln(10) either, epoch after epoch. Two independent signals, both available, neither compared to anything.",
            student="reading '10.24%' as a low but real score. Against a baseline of 10.00% it is not a low score, it is no score.",
            catch="the corrected specification has three additions and each is something the assistant cannot know unless you say it: what a defensible default is, what the number must be compared against, and what evidence to produce when the answer is 'it did not work'."),
        code('''
print(f"the assistant's model:  {acc:.4f}")
print(f"a model with no weights at all: {baseline:.4f}")
print(f"difference: {100*(acc - baseline):+.2f} accuracy points")
print()
print("The loss it printed, epoch by epoch, never left ln(10) = "
      f"{math.log(10):.4f} either.")
print("Five decimal places of nothing, formatted to look like progress.")
'''),
        md("""
### The corrected specification

> *"Write a PyTorch classifier for CIFAR-10 with 20 hidden layers. **State
> explicitly which initialisation each layer gets and why.** After training,
> report accuracy against the majority-class baseline, and log the gradient
> norm of the first and last weight matrices at every epoch. If the loss does
> not fall below ln(10) within three epochs, stop and report that instead of
> continuing."*

Three additions, and each of them is a thing the assistant cannot know unless
you say it: what a defensible default is, what the number must be compared
against, and what evidence to produce when the answer is *it did not work*.
"""),

        md("""
## 10 · Instrument it

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
            check="assert one row per hidden layer before reading anything off it",
            left_open="that the whole-tensor sd is dominated by the spread of that layer's random BIASES across units, which does not change with depth. It stays near 0.07 at layer 20 and the network looks alive.",
            student="reading `h.std()` and concluding the forward pass is healthy. What carries information is how much a unit's output moves when the INPUT changes — sd down the batch, averaged over units — and that collapses by sixteen orders of magnitude.",
            catch="the saturated column is worse than useless here, and the notebook says so. It asks whether |h − 0.5| > 0.45 while the activations sit in a band of sd 0.071 — 6.3 standard deviations away, unreachable, and 0.000 at every depth reads as reassurance."),
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
            left_open="that sixteen orders of magnitude cannot be shown on a linear axis. A quantity that spans that range and is plotted linearly is a flat line at zero.",
            student="plotting only the first. It is the natural choice, it is not wrong as a plot, and it supports exactly the wrong conclusion.",
            catch="when a quantity might span orders of magnitude, try a log axis before concluding it is constant. Flat on linear and flat on log are very different findings."),
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
            check="assert no float32 gradient underflowed to exactly zero, and report the largest disagreement between the two precisions",
            left_open="why float64 is not superstition here. A norm is a sum of squares, so gradients near 1e-20 would square to exactly zero in float32 and the plot would be a lie. The cell CHECKS that they do not rather than assuming.",
            student="measuring in float32 and plotting a floor that is the dtype rather than the network. The assert is what distinguishes the two.",
            catch="when you plot something tiny on a log axis, check it did not underflow. A log plot of zeros is blank, and a log plot of denormals is noise."),
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
            left_open="what a straight line on a log axis means: the attenuation is not an accident of one layer, it is the SAME factor applied nineteen times.",
            student="taking an arithmetic mean of the ratios, which is dominated by whichever layer happens to be largest and does not reproduce the end-to-end number.",
            catch="the self-check in the last print. If the per-layer factor to the 19th does not reproduce the measured end-to-end ratio, one of the two is wrong and you have found out in one line."),
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
            left_open="what it would look like if the first layers were merely slow to start: the profile would flatten as the network learns. It does not.",
            student="assuming more epochs will fix it. Twenty epochs of Adam did not move the first layer's gradient into a range where a learning rate of 1e-3 could do anything with it.",
            catch="this is why the instrumentation had to go in before the first training run. The question 'did it change over time' cannot be asked retroactively."),
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
            left_open="why this is not simply 'small gradient, small step'. Adam divides by a running estimate of the gradient's own magnitude, so the step size is not proportional to the gradient — but the first layers are driven by a signal fifteen orders of magnitude below the last ones, and it is almost entirely noise.",
            student="keeping a reference to the network before training and being surprised it changed. PyTorch trains in place; the only clean copy is one you clone or one you reconstruct from the seed.",
            catch="relative change, not absolute. A weight matrix with small entries and one with large entries cannot be compared by the norm of their differences."),
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
## 11 · Record the number, and stop

You have a network that runs, a loop that is provably correct, and a result
indistinguishable from guessing. That is today's deliverable.

On the same sheet of paper, next to what you predicted:

```
Test accuracy I actually got:            % ____________
Gradient norm at layer 1:                  ____________
Gradient norm at layer 20:                 ____________
Per-layer attenuation factor:              ____________
```

Do not repair anything. The next lecture derives that per-layer factor from
the shape of the weight matrices — and then removes it.
"""),
        prompt(
            label="record the number, and stop",
            input="everything measured",
            output="baseline, both depths, both gradient norms, and the attenuation factor",
            constraint="the baseline goes in the table, at the top",
            left_open="that nothing is repaired. You have a network that runs, a loop that is provably correct, and a result indistinguishable from guessing — and that is the deliverable.",
            student="fixing it. The next lecture derives the per-layer factor from the shape of the weight matrices and then removes it, and it needs the measurement to derive it against.",
            catch="photograph this table. The per-layer attenuation is the number the next lecture predicts from first principles, and predicting a number you have already seen is not the same exercise."),
        code('''
print("=" * 62)
print(f"{'baseline (majority class)':38s} {baseline:.4f}")
print(f"{'20 hidden layers, ' + str(EPOCHS) + ' epochs':38s} "
      f"{accuracy(deep, Xt, yt):.4f}")
print(f"{'2 hidden layers, same everything else':38s} {sweep[2]:.4f}")
print(f"{'gradient, layer 1':38s} {g64[0]:.4e}")
print(f"{'gradient, layer 20':38s} {g64[19]:.4e}")
print(f"{'per-layer attenuation':38s} {1/gain:.4f}")
print("=" * 62)
'''),

        md("""
## 12 · Red-team

Swap notebooks with the team beside you. Ten minutes. The five questions, and
one that is new today:

1. What touched the test set?
2. What was fitted, and on what?
3. What is the shape here?
4. What was dropped — rows, columns, NaNs? Count them.
5. What is the default I did not ask for?
6. **What would this number be if the model had learned nothing?** Find the
   line in your partner's notebook that answers it. If there is no such line,
   that is the finding.

Report what you **found**, not what you would have done differently.
"""),
    ]
