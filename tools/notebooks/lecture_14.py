"""
Lecture 14 — Making it train.

Fix. Géron Chapter 11. Mathematical thread 7: variance propagation through
layers.

Exports build() -> list[nbformat cell]. Self-contained: it reloads and re-splits
the data rather than assuming Lecture 13's kernel is still alive.

The thread is derived first and then *checked against the previous lecture's
own logged numbers*, which is the only reason it is in the course. Every repair
is measured on its own before any of them are combined.
"""

from __future__ import annotations

import nbformat as nbf


def md(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(text.strip("\n"))


def code(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(text.strip("\n"))


HEADER = """
# Making it train

**Lecture 14 · Fix** · Géron, Chapter 11 · *Mathematical thread: variance
propagation through layers*

Applications of Machine Learning — BSc Mathematics of Artificial Intelligence

---

**How to use this notebook.** Read before you run. Cells marked
**⚠ read before running** contain a defect on purpose.

Have the previous lecture's sheet of paper beside you. Section 3 predicts a
number you already measured, from the shapes of the weight matrices alone, and
the comparison is the point of the whole lecture.
"""


def build() -> list:
    return [
        md(HEADER),

        md("## 1 · Setup, and the same data as last time"),
        code('''
# --- setup -------------------------------------------------------------------
import math, sys, time
import numpy as np
import torch
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt

print(f"python  {sys.version.split()[0]}   torch {torch.__version__}")

RANDOM_STATE = 42
torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
print(f"device  {device}")
'''),
        code('''
# The same 10,000 images, the same seed, the same scaling. If your split
# differs from the previous lecture's by one image, none of the comparisons
# below mean anything.
# ⏱ instant if you ran Lecture 13 in this runtime; 1-3 minutes otherwise.
train_ds = torchvision.datasets.CIFAR10("datasets", train=True,  download=True)
test_ds  = torchvision.datasets.CIFAR10("datasets", train=False, download=True)
CLASSES  = train_ds.classes

Xtr_u8, Xte_u8 = train_ds.data, test_ds.data
ytr = np.asarray(train_ds.targets, dtype=np.int64)
yte = np.asarray(test_ds.targets,  dtype=np.int64)

N_FIT, N_VAL = 10_000, 5_000
rng   = np.random.default_rng(RANDOM_STATE)
order = rng.permutation(len(Xtr_u8))
val_idx, fit_idx = order[:N_VAL], order[N_VAL:N_VAL + N_FIT]

flat = lambda a: a.reshape(len(a), -1).astype(np.float32) / 255.0
X_fit_raw = flat(Xtr_u8[fit_idx])
mu, sd = X_fit_raw.mean(axis=0), X_fit_raw.std(axis=0) + 1e-7
std = lambda a: (flat(a) - mu) / sd

X_fit,  y_fit  = std(Xtr_u8[fit_idx]), ytr[fit_idx]
X_val,  y_val  = std(Xtr_u8[val_idx]), ytr[val_idx]
X_test, y_test = std(Xte_u8), yte

assert X_fit.shape == (N_FIT, 3072) and X_test.shape == (10_000, 3072)
assert set(np.bincount(y_test)) == {1_000}
baseline = 0.1
print(f"fit {len(X_fit):,}  val {len(X_val):,}  test {len(X_test):,}   "
      f"baseline {baseline:.4f}")
'''),

        md("""
## 2 · Where we left off

Twenty hidden layers, a logistic activation, whatever `nn.Linear` chose for the
weights, and a training loop with no bugs in it. The result was
indistinguishable from guessing, and the gradient norm fell by fifteen orders
of magnitude between the last layer and the first.

Rebuild exactly that network, so the repairs have something to be measured
against.
"""),
        code('''
DEPTH, WIDTH, N_IN, N_OUT = 20, 100, 3072, 10
EPOCHS, BATCH, LR = 20, 128, 1e-3

ACTS = {"sigmoid": nn.Sigmoid, "tanh": nn.Tanh, "relu": nn.ReLU,
        "elu": nn.ELU, "selu": nn.SELU,
        "leaky": lambda: nn.LeakyReLU(0.01)}

def make_net(depth=DEPTH, width=WIDTH, act="sigmoid", init="torch",
             norm=None, dropout=0.0, n_in=N_IN, n_out=N_OUT,
             dtype=torch.float32):
    layers, prev = [], n_in
    for _ in range(depth):
        layers.append(nn.Linear(prev, width))
        if norm == "batch":
            layers.append(nn.BatchNorm1d(width))
        elif norm == "layer":
            layers.append(nn.LayerNorm(width))
        layers.append(ACTS[act]())
        if dropout:
            layers.append(nn.Dropout(dropout))
        prev = width
    layers.append(nn.Linear(prev, n_out))

    net = nn.Sequential(*layers).to(dtype)
    for m in net:
        if not isinstance(m, nn.Linear):
            continue
        if init == "torch":
            pass                                  # whatever the default is
        elif init == "glorot":
            nn.init.xavier_normal_(m.weight); nn.init.zeros_(m.bias)
        elif init == "he":
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            nn.init.zeros_(m.bias)
        elif init == "normal1":
            nn.init.normal_(m.weight, 0.0, 1.0); nn.init.zeros_(m.bias)
        else:
            raise ValueError(init)
    return net

assert len([m for m in make_net() if isinstance(m, nn.Linear)]) == DEPTH + 1
print(f"{sum(p.numel() for p in make_net().parameters()):,} parameters")
'''),

        md("""
## 3 · Thread 7 — variance propagation

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
## 4 · Now measure it, on the network you built last time

The prediction above used nothing but the shapes of the matrices and one
expectation. If it is right, it should reproduce the attenuation you logged in
the previous lecture — a number that came out of a completely different
calculation.

⏱ **about 30 seconds** — four networks, eight batches each, on the CPU in
float64.
"""),
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

### But that is not the number on your sheet

The previous lecture logged `||dL/dW||`, not `||dL/dz||`. They differ, because

    || dL/dW_l ||  ~  || delta_l || . || a_(l-1) ||

so the weight-gradient ratio carries the backward factor **and** the forward
one. For the logistic the forward pass is scale-stable, so the two nearly
coincide. For an unnormalised ReLU stack they do not — and that is not a
technicality, as the next cell shows.
"""),
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
## 5 · The training harness

One function, one seed, one subset, one epoch count. Every row in every table
below differs from its neighbour in exactly one argument.
"""),
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
## 6 · Each repair, alone

Resist applying everything at once. A stack of seven changes that works tells
you nothing about which of the seven mattered — and two of them will turn out
to make things worse on their own.

⏱ **about 5 minutes** for all seven.
"""),
        code('''
alone = [
    ("nothing (Lecture 13)",  dict(act="sigmoid", init="torch")),
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
## 7 · The ladder

Now stack them, in the order the diagnosis suggests: fix the signal first,
then the optimisation, then the generalisation.

⏱ **about 5 minutes.**
"""),
        code('''
ladder = [
    ("Lecture 13, unchanged",       dict(act="sigmoid", init="torch")),
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
    rows.append((label, h["test_acc"]))
    curves[label] = h["loss"]
    prev = h["test_acc"]

best = max(rows, key=lambda r: r[1])
print(f"\\nbest row: {best[0]} at {best[1]:.4f}")
print(f"last row: {rows[-1][0]} at {rows[-1][1]:.4f}")
assert best[1] > 3 * rows[0][1], "the repaired network should not be at chance"
if best is not rows[-1]:
    print("\\nThe last rung is NOT the best configuration. Report the best row,")
    print("and say which rungs you dropped and why. That is what the table is")
    print("for; without it you would ship the bottom row by default.")
'''),
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
## 8 · ⚠ An assistant repairs the network

> *"My 20-layer network isn't learning — the gradients vanish. Switch it to
> ReLU and initialise the weights properly."*

Here is what comes back. It runs, the loss falls, the accuracy is several times
the baseline, and it looks like the problem is solved.
"""),
        code('''
def assistant_fix(depth=20, width=100, n_in=3072, n_out=10):
    layers, prev = [], n_in
    for _ in range(depth):
        layers += [nn.Linear(prev, width), nn.ReLU()]
        prev = width
    layers.append(nn.Linear(prev, n_out))
    model = nn.Sequential(*layers)
    for m in model:
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)      # "proper initialisation"
            nn.init.zeros_(m.bias)
    return model

torch.manual_seed(RANDOM_STATE)
model = assistant_fix().to(device)
optim, lossf = torch.optim.Adam(model.parameters(), lr=1e-3), nn.CrossEntropyLoss()
g = torch.Generator().manual_seed(RANDOM_STATE)
for ep in range(EPOCHS):
    model.train()
    perm = torch.randperm(len(Xf), generator=g).to(device)
    for i in range(0, len(Xf), BATCH):
        idx = perm[i:i+BATCH]
        optim.zero_grad(); lossf(model(Xf[idx]), yf[idx]).backward(); optim.step()
xavier_relu = accuracy(model, Xt, yt)
print(f"the assistant's repair: test accuracy {xavier_relu:.4f}")
print(f"the baseline:                          {baseline:.4f}")
print("It moved. So it worked?")
'''),
        md("""
### The review question that catches it

*Xavier* and *Glorot* are the same person and the same formula. So the
assistant did do something defensible — but it used the initialisation derived
**for a roughly linear activation** on an activation that throws away half the
variance.

Put it back in the table from section 3: with ReLU, Glorot gives
$\\rho = \\sqrt{1\\cdot\\tfrac12} = 0.707$, not 1. Over nineteen layers that is
$0.707^{19}$ — three orders of magnitude, not fifteen. Enough to train
visibly, and far from correct.

Measure the difference rather than arguing about it.
"""),
        code('''
gx = grad_profile(act="relu", init="glorot")
gh = grad_profile(act="relu", init="he")
rx = float(np.exp(np.mean(np.log(gx[1:DEPTH] / gx[0:DEPTH-1]))))
rh = float(np.exp(np.mean(np.log(gh[1:DEPTH] / gh[0:DEPTH-1]))))
print(f"Glorot + ReLU  rho {1/rx:.4f}  (theory {math.sqrt(0.5):.4f})   "
      f"end to end {gx[0]/gx[DEPTH-1]:.3e}")
print(f"He     + ReLU  rho {1/rh:.4f}  (theory {1.0:.4f})   "
      f"end to end {gh[0]/gh[DEPTH-1]:.3e}")

_, h_he = train(act="relu", init="he")
print(f"\\nXavier + ReLU  test {xavier_relu:.4f}")
print(f"He     + ReLU  test {h_he['test_acc']:.4f}")
print(f"cost of the wrong constant: "
      f"{100*(h_he['test_acc'] - xavier_relu):+.2f} accuracy points")
'''),
        md("""
### The corrected specification

> *"Switch to ReLU and use the initialisation derived for ReLU — He normal,
> `Var(w) = 2/fan_in`. **Then print the per-layer gradient norm ratio and show
> me it is within 20% of 1.0**, so I can see the initialisation did what it
> claims."*

The last sentence is the important one. "Initialise it properly" is not a
specification; a per-layer ratio near 1 is.
"""),

        md("""
## 9 · Batch normalisation, and layer normalisation

Initialisation fixes the variance **at step zero**. It says nothing about step
five thousand, by which time the weights have moved. Normalisation enforces the
same condition at every step, by standardising each layer's inputs and then
learning a scale and a shift.

⏱ **about 2 minutes.**
"""),
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

Watch the two modes disagree, which is the failure Lecture 12 warned about and
which is now much harder to see, because batch normalisation does not
fluctuate:
"""),
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

Applied on their own to the Lecture 13 network, batch normalisation rescues it
completely and layer normalisation does nothing at all — yet on the *repaired*
network the two are within a point of each other.

We are not going to account for that here, because we have not measured enough
to. Below is the measurement that would start to: run both, on the broken
network, and look at where the per-layer backward factor ends up.
"""),
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
## 10 · Gradient clipping

Clipping rescales the whole gradient vector when its norm exceeds a threshold.
It is a defence against the *other* failure — the one where $\\rho > 1$.

Look at what the norms actually are before choosing a threshold. A clip value
below the median silently turns your optimiser into sign descent.
"""),
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
## 11 · Faster optimisers, and schedules

Now that the gradient reaches every layer, the optimiser has something to work
with. Compare six on the *repaired* network — comparing them on the broken one
would have measured nothing, which is why this section is here and not in
section 6.

⏱ **about 4 minutes.**
"""),
        code('''
for name, lr in [("sgd", 1e-2), ("momentum", 1e-2), ("nesterov", 1e-2),
                 ("rmsprop", 1e-3), ("adam", 1e-3), ("adamw", 1e-3)]:
    _, h = train(act="relu", init="he", norm="batch", opt=name, lr=lr)
    print(f"{name:9s} lr {lr:g}   test {h['test_acc']:.4f}   "
          f"final loss {h['loss'][-1]:.4f}")
'''),
        code('''
for name in (None, "cosine", "onecycle"):
    _, h = train(act="relu", init="he", norm="batch", clip=1.0, schedule=name)
    print(f"{str(name):9s} test {h['test_acc']:.4f}   "
          f"best validation {max(h['val_acc']):.4f}")
'''),

        md("""
## 12 · Re-measure

The same twenty layers, the same 10,000 images, the same twenty epochs, the
same seed. Only the four lines that decide how the signal propagates have
changed.
"""),
        code('''
_, base = train(act="sigmoid", init="torch")
_, best = train(act="relu", init="he", norm="batch", clip=1.0,
                schedule="onecycle", dropout=0.1)

print("=" * 60)
print(f"{'baseline (majority class)':34s} {baseline:.4f}")
print(f"{'Lecture 13, 20 layers':34s} {base['test_acc']:.4f}")
print(f"{'the same 20 layers, repaired':34s} {best['test_acc']:.4f}")
print(f"{'improvement, accuracy points':34s} "
      f"{100*(best['test_acc'] - base['test_acc']):+.2f}")
print("=" * 60)
print("\\nAnd it is still an MLP on flattened pixels. Lecture 15 changes that,")
print("and the ceiling this network is running into is the subject there.")
'''),

        md("""
## 13 · Red-team

Swap notebooks. Ten minutes. The five questions, and three that are new:

1. What touched the test set?
2. What was fitted, and on what?
3. What is the shape here?
4. What was dropped — rows, columns, NaNs? Count them.
5. What is the default I did not ask for?
6. **Which initialisation is on each layer, and which activation was it
   derived for?**
7. **If there is a normalisation layer, is there a `model.eval()` before every
   measurement?** Running the evaluation twice will not catch a missing one.
8. **Is every row of the ablation table one change away from its neighbour?**
   Two changes in one row is not a measurement.

Report what you **found**, not what you would have done differently.
"""),
    ]
