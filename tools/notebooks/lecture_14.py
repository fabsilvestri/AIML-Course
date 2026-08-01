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
from _prompt import prompt                                # noqa: E402


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
        prompt(
            label="setup",
            input="nothing",
            output="versions, seeds, device",
            constraint="same seeds as the previous lecture, so the rebuild below is a rebuild and not a new experiment",
            left_open="nothing here is examinable. It is the same four lines every notebook in Part II opens with, and they are worth being bored by.",
            student="changing the seed between the broken and the repaired run. Every comparison in this notebook is a difference of two numbers, and a changed seed puts noise in that difference.",
            catch="if a notebook exists to measure a repair, its setup must be byte-identical to the notebook that measured the fault."),
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
        prompt(
            label="the same 10,000 images",
            input="CIFAR-10",
            output="the identical fit / validation / test split and scaling",
            constraint="recompute mu and sd from the FIT subset, exactly as before — if your split differs from the previous lecture's by one image, none of the comparisons below mean anything",
            check="assert the shapes and that the test set is still balanced",
            left_open="that `baseline = 0.1` is hard-coded here rather than recomputed. The balance assert on the line above is what earns that.",
            student="reusing variables from the previous notebook's kernel. It works in the lecture room and nowhere else.",
            catch="an assert immediately before a hard-coded constant is how you make a shortcut safe. Without it, 0.1 is a number someone remembered."),
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
        prompt(
            label="one factory, every variant",
            input="depth, width, activation, initialisation, normalisation, dropout",
            output="a fresh network built to that specification",
            constraint="ONE function that can build every configuration in the notebook — every row of every table below differs from its neighbour in exactly one argument to this call",
            check="assert the layer count, so a change to the builder cannot silently change the architecture",
            left_open="that `init='torch'` is a deliberate no-op branch. It means 'whatever nn.Linear chose', and naming it makes the default a choice rather than an absence.",
            student="writing a separate model class per experiment. Two classes that differ in three places cannot support a claim about one of them.",
            catch="if your ablation table is built from more than one constructor, it is not an ablation table."),
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
        prompt(
            label="check the variance identity before believing it",
            input="random weight matrices at four weight variances",
            output="the predicted and measured variance of the output",
            constraint="unit-variance inputs, so `Var(z) = n_in · Var(w)` can be read directly against the measurement",
            left_open="the assumptions that make the identity true: weights drawn independently with mean zero, independent of the inputs, and inputs with mean zero. Every one of them is approximately true in a real network, and the residual error later is exactly where they are weakest.",
            student="taking the boxed formula on faith. It is one line to check and every result in the lecture is a consequence of it.",
            catch="20,000 samples, not 100. This is a claim about a variance, and a variance estimated from a small sample has a variance of its own."),
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
            left_open="that the two conditions are the same condition ONLY when n_in = n_out. Every real network changes width somewhere, and ours starts at 3072 → 100.",
            student="learning 'Glorot is 2/(fan_in+fan_out)' as a formula. It is a harmonic mean of two irreconcilable demands, and knowing that tells you when it will be close and when it will not.",
            catch="on the 100 → 100 layers the two demands agree and Glorot is exact. On the first layer they differ by a factor of 30 and nothing can fix that."),
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
            left_open="why it is exactly a half. ReLU zeroes the negative half of a symmetric zero-mean distribution, so exactly half the second moment survives.",
            student="memorising `2/fan_in` without the reason. When you meet an activation that is not ReLU — ELU, SELU, GELU — you then have no way to work out its constant.",
            catch="the logistic's derivative never exceeds 1/4, which is the other half of the story and the direct cause of the previous lecture's failure."),
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
            left_open="that only ρ = 1 survives depth. If ρ < 1 the gradient disappears and if ρ > 1 it explodes; there is no regime in which a constant repeated L times is safe.",
            student="not noticing where the 3 comes from. The nn.Linear default is a uniform on (−b, b) with b = 1/√fan_in, and a uniform on (−b, b) has variance b²/3.",
            catch="a prediction made before the measurement is worth ten made after it. Write these four numbers down before running the next cell."),
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
        prompt(
            label="⏱ 30 s — measure it, four ways",
            input="four initialisation and activation schemes",
            output="predicted ρ, measured ρ, the error, the forward scale, and the end-to-end ratio",
            constraint="measure ||dL/dz|| — the DELTA — not ||dL/dW||. They are different quantities and confusing them is the single easiest way to misread this lecture",
            check="assert prediction and measurement agree within 15%, per scheme",
            left_open="where the residual comes from, and the cell says: E[φ′²] is not quite (1/4)² because the pre-activations are not exactly at zero, and successive gradient components are not exactly uncorrelated.",
            student="profiling the weight gradient and comparing it with a theory about the delta. The two differ by the forward factor, and for an unnormalised ReLU stack that is not a technicality.",
            catch="`retain_grad()` on the intermediates. Non-leaf tensors do not keep their gradients by default, and without it `z.grad` is None with no error."),
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
            left_open="that the prediction came from counting rows and columns of matrices, and the measurement came from a backward pass through twenty layers on real photographs. They agree to a few per cent.",
            student="reading the flat line as the good one without checking which quantity is plotted. The next cell is about a flat profile that is flat for the wrong reason.",
            catch="this is the whole lecture. Everything below it is application."),
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
        prompt(
            label="why the previous lecture's diagnostic is not enough",
            input="ρ, the forward scale, and the weight-gradient ratio per scheme",
            output="all four columns side by side",
            constraint="show ρ and the forward factor SEPARATELY as well as their product — the product is what the previous lecture measured",
            left_open="the 'Glorot, ReLU' row. Its weight-gradient ratio is close to 1, so the previous lecture's diagnostic would pass it — because the backward attenuation and the forward attenuation cancel in that one number.",
            student="using a flat ||dL/dW|| profile as a certificate that the initialisation is right. A flat gradient profile does NOT certify an initialisation, and this row is the counterexample.",
            catch="||dL/dW_l|| ≈ ||delta_l||·||a_(l−1)||. Two factors in one number, and they can cancel."),
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
        prompt(
            label="the training harness",
            input="every knob the notebook varies",
            output="a trained network and its history, including the test accuracy",
            constraint="ONE function, one seed, one subset, one epoch count — so every row of every table differs from its neighbour in exactly one argument",
            check="a harness check: a 2-layer ReLU network must reach better than 0.2 in three epochs, or the harness itself cannot learn and every table below is measuring the harness",
            left_open="that the harness computes the test accuracy on every call. That is defensible only because no configuration in this notebook is chosen by it — the ladder is fixed in advance.",
            student="skipping the harness check. Twenty rows of 10% would then read as twenty failed repairs rather than one broken loop.",
            catch="test the instrument before the experiment. Three epochs on two layers costs seconds and rules out the most expensive possible mistake."),
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
        prompt(
            label="⏱ 5 min — each repair, alone",
            input="seven repairs, each applied to the broken network by itself",
            output="test accuracy and final loss for each",
            constraint="ONE change per row, all against the same broken baseline — a stack of seven changes that works tells you nothing about which of the seven mattered",
            left_open="that three of them do nothing on their own, and that is informative rather than disappointing. Clipping, a schedule and dropout are all answers to problems this network does not have.",
            student="applying everything at once because it is faster. Two of these repairs make things WORSE alone, and a combined run hides both facts.",
            catch="clipping bounds a gradient that is too large; ours is fifteen orders of magnitude too small. Dropout fights overfitting; a network at chance is not overfitting. Applying a fix whose failure mode you have not measured is how a notebook grows to forty cells and stops being explicable."),
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
        prompt(
            label="⏱ 5 min — the ladder",
            input="the same repairs, stacked in diagnostic order",
            output="each rung's accuracy and its delta from the rung below",
            constraint="stack in the order the DIAGNOSIS suggests — signal first, then optimisation, then generalisation",
            check="assert the repaired network is at least three times chance, and record which rung was actually best",
            left_open="that the LAST rung is not the best one. The cell detects that and says so, because otherwise you would ship the bottom row by default.",
            student="reporting the final row because it has the most repairs in it. The table exists precisely so you can say which rungs you dropped and why.",
            catch="capture the best row into a variable and use THAT downstream. Hard-coding the last rung's settings in the summary would report 33.4% where the argument requires 43.9% — the notebook committing the mistake the deck forbids."),
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
            left_open="why only four curves are plotted. Seven overlapping loss curves are unreadable, and the four chosen are the ones that differ in kind rather than in degree.",
            student="plotting all seven and producing a figure nobody can read. A selection is a decision; make it deliberately and say what you selected.",
            catch="horizontal bars with the labels on the axis, not a legend. Seven long labels in a legend is a puzzle."),
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
        prompt(
            label="⚠ what the assistant returns",
            input="'my 20-layer network isn't learning — switch it to ReLU and initialise the weights properly'",
            output="a trained network and its test accuracy",
            constraint="run it exactly as returned. The loss falls, the accuracy is several times the baseline, and it looks like the problem is solved",
            left_open="that Xavier and Glorot are the same person and the same formula, so the assistant did do something defensible — it used the initialisation derived for a roughly LINEAR activation on an activation that throws away half the variance.",
            student="accepting it because the number moved. With ReLU, Glorot gives ρ = √(1·½) = 0.707, not 1 — over nineteen layers that is three orders of magnitude rather than fifteen. Enough to train visibly, and far from correct.",
            catch="a repair that improves the number is not thereby the right repair. Put the constant back into the ρ table and see what it predicts."),
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
        prompt(
            label="measure the difference rather than arguing about it",
            input="both initialisations with ReLU",
            output="the measured ρ against theory for each, and the accuracy cost",
            constraint="report the measured ρ beside the PREDICTED one — 0.707 and 1.0 were both derived before this cell ran",
            left_open="the corrected specification, whose last sentence is the important one: 'print the per-layer gradient norm ratio and show me it is within 20% of 1.0'. 'Initialise it properly' is not a specification; a per-layer ratio near 1 is.",
            student="arguing about which initialisation is correct. Both are correct for the activation they were derived for, and the measurement settles it in thirty seconds.",
            catch="ask any repair to produce the diagnostic that would show it worked. An accuracy that went up is compatible with a great many wrong repairs."),
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
        prompt(
            label="⏱ 2 min — normalisation, and what it costs",
            input="the repaired ReLU network with no norm, batch norm, and layer norm",
            output="accuracy, wall clock and parameter count for each",
            constraint="report the PARAMETER COUNT — two learned vectors per layer is 200 numbers against 10,100, under 2%, and the wall clock is the real cost",
            left_open="what batch normalisation depends on that layer normalisation does not. Batch statistics are computed ACROSS THE BATCH, so the prediction for one image depends on the other images in its batch at training time and on a running average at evaluation time.",
            student="assuming normalisation is free because the parameter count barely moves. The wall clock is where it is paid, and the batch dependence is where it bites.",
            catch="initialisation fixes the variance at step ZERO. It says nothing about step five thousand, by which time the weights have moved — that is what normalisation is for."),
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
        prompt(
            label="the two modes disagreeing, quietly",
            input="a batch-normalised network, evaluated in both modes",
            output="the accuracy each way, and the difference",
            constraint="use the SAME 2,000 images both times — the difference must come from the mode and nothing else",
            left_open="why this is harder to catch than the dropout version. Batch normalisation does not FLUCTUATE, so running the evaluation twice gives the same wrong number twice.",
            student="relying on the wobble test from Lecture 12. It catches dropout because dropout is random; batch norm in training mode is deterministic given the batch, and the tell is gone.",
            catch="from here on `model.eval()` matters more, not less, and the cheap diagnostic that used to catch a missing one no longer does."),
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
        prompt(
            label="the row we do not explain",
            input="the broken network with no norm, batch norm, layer norm",
            output="ρ and the end-to-end delta ratio for each",
            constraint="report the measurement without an explanation attached to it",
            left_open="the anomaly itself: applied alone to the broken network, batch normalisation rescues it completely and layer normalisation does nothing — yet on the REPAIRED network the two are within a point of each other.",
            student="inventing an explanation. We have not measured enough to account for it, and this cell is the measurement that would start to.",
            catch="whatever you conclude, write down the measurement that supports it. An explanation with no number attached is the thing this course is trying to replace."),
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
        prompt(
            label="look at the norms before choosing a threshold",
            input="two epochs of gradient norms, under He and under N(0,1)",
            output="the median and maximum for each, and both distributions on a log axis",
            constraint="use `clip_grad_norm_` with an INFINITE threshold to read the norm without clipping it — the measurement must not be the intervention",
            left_open="that clipping defends against the OTHER failure, the one where ρ > 1. It is in this notebook to be measured, not because this network needs it.",
            student="picking `clip=1.0` from a tutorial. A clip value below the median silently turns your optimiser into sign descent, and nothing warns you.",
            catch="log10 of the norms, histogrammed. The two initialisations differ by orders of magnitude and a linear histogram shows one bar."),
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
        prompt(
            label="⏱ 4 min — six optimisers, on the REPAIRED network",
            input="SGD, momentum, Nesterov, RMSprop, Adam, AdamW",
            output="test accuracy and final loss for each",
            constraint="different learning rates for the SGD family and the adaptive family — 1e-3 on plain SGD is not a fair test of plain SGD",
            left_open="why this section is here and not in section 6. Comparing optimisers on the BROKEN network would have measured nothing: none of them can descend a gradient that does not arrive.",
            student="benchmarking optimisers first, because it is the most obviously tunable thing. Every row would have read 10% and the conclusion would have been that the optimiser does not matter.",
            catch="fix the signal before tuning the search. An optimiser comparison on a network that cannot learn is a comparison of nothing."),
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
            left_open="that OneCycle raises the learning rate to 10× before lowering it, so the `max_lr` here is not the same knob as `lr` elsewhere in the notebook.",
            student="comparing a scheduled run against an unscheduled one at the same nominal learning rate and concluding the schedule is what helped. The schedule changed the learning rate; that IS the comparison, and it needs saying.",
            catch="a schedule is a hyperparameter with a shape rather than a value. Report the curve or at least its peak, not only its endpoint."),
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
        prompt(
            label="re-measure",
            input="the broken network and the BEST rung of the ladder",
            output="baseline, both accuracies, the improvement, and how many rungs were dropped",
            constraint="use BEST_KW, captured from the ladder — not the last rung's settings typed out again",
            left_open="that the ceiling this network is running into is the subject of the next lecture. It is still an MLP on flattened pixels, and no amount of initialisation repairs that.",
            student="hard-coding the final rung. That would report 33.4% where the argument requires 43.9% — the notebook committing, two pages later, the exact mistake the deck spends a slide forbidding.",
            catch="when a table selects a winner, carry the winner forward in a variable. Retyping its settings is how the summary and the table drift apart."),
        code('''
_, base = train(act="sigmoid", init="torch")
# The BEST rung, not the last one. Hard-coding the last rung's settings here
# would report 33.4% while the slide two pages on says, in as many words, "the
# number we report is 43.9%, not 33.4%" — the notebook demonstrating the very
# mistake the deck forbids.
_, repaired = train(**BEST_KW)

print("=" * 60)
print(f"{'baseline (majority class)':34s} {baseline:.4f}")
print(f"{'Lecture 13, 20 layers':34s} {base['test_acc']:.4f}")
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
