"""
Lecture 10 — PyTorch. Géron Chapter 10.

Derivation: backpropagation as reverse-mode automatic differentiation.

Exports build() -> list[nbformat cell]. Self-contained: it reloads and re-splits
the data rather than assuming Lecture 9's kernel is still alive. A notebook that
only runs because a previous one left variables in memory is not reproducible.

Runs on CPU throughout. The device check is kept because knowing how to ask is
part of the lecture, but nothing here needs an accelerator.
"""

from __future__ import annotations

import nbformat as nbf
from _prompt import prompt                                # noqa: E402


def md(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(text.strip("\n"))


def code(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(text.strip("\n"))


HEADER = """
# PyTorch

**Lecture 10** · Géron, Chapter 10

Applications of Machine Learning — BSc Mathematics of Artificial Intelligence

---

**How to use this notebook.** Read before you run. Every code cell is preceded
by the specification that would produce it — input, output, constraint, check.

Three sections marked **⚠** deliberately run *broken* code, and say so in their
heading before you reach them. They are not traps: they are the only way to see
what a bug that raises no exception actually does to a loss curve. Each one runs
the broken version and the correct version side by side and prices the
difference.

Trains on 12,000 images so that a free Colab CPU runtime finishes in a few
minutes.
"""


def build() -> list:
    return [
        md(HEADER),

        md("## 1 · Setup, and the hardware question"),
        prompt(
            label="setup, and the hardware question",
            input="nothing",
            output="versions, seeds, and the device this notebook will use",
            constraint="pick the device by ASKING — cuda, then mps, then cpu — and print which one won",
            check="print the device and say what to do if it is cpu. A student whose runtime silently has no accelerator will otherwise conclude PyTorch is slow.",
            **{"try": "force `device = \"cpu\"` and re-run the notebook end to end. Everything works and the wall clock barely moves — which is the measured claim behind this course running on a free tier."}),
        code('''
# --- setup -------------------------------------------------------------------
import sys, time, warnings
import numpy as np, sklearn, torch, torchvision
import torch.nn as nn
import matplotlib.pyplot as plt

print(f"python       {sys.version.split()[0]}")
print(f"torch        {torch.__version__}")
print(f"torchvision  {torchvision.__version__}")

RANDOM_STATE = 42
torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

# Not examinable. PyTorch will use an accelerator if one is present -- CUDA on
# Colab and most Linux boxes, MPS on Apple Silicon -- and fall back to the CPU
# everywhere else. Nothing in this course needs the accelerator; Lecture 10
# measures where the crossover actually falls, and it is above this scale.
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
print(f"\\ndevice       {device}")
if device == "cpu":
    print("no accelerator found. Everything below still runs; it is slower.")
    print("CPU is expected here: nothing in this notebook needs more.")
'''),

        md("""
## 2 · The same data, the same split

Reloaded here rather than inherited. If your split differs from the previous
lecture's by one image, none of the comparisons below mean anything.
"""),
        prompt(
            label="the same data, the same split",
            input="Fashion MNIST again",
            output="the identical fit / validation / test split as the previous lecture",
            constraint="reload and re-split from the same seed rather than inheriting — if your split differs from the previous lecture's by one image, none of the comparisons below mean anything",
            check="assert the sizes and the shapes. Two notebooks that compare numbers must rebuild the same split from the same seed, and assert the sizes. That assert is the only thing standing between you and an invalid comparison.",
            **{"try": "raise the subsample from 12,000 to 20,000 images. Every accuracy rises a little and every timing rises a lot; decide which of the two you would trade."}),
        code('''
train_ds = torchvision.datasets.FashionMNIST("datasets", train=True, download=True)
test_ds  = torchvision.datasets.FashionMNIST("datasets", train=False, download=True)
CLASSES  = train_ds.classes

def flatten_scale(a):
    return a.reshape(len(a), -1).astype(np.float32) / 255.0

Xtr_u8, ytr = train_ds.data.numpy(), train_ds.targets.numpy().astype(np.int64)
Xte_u8, yte = test_ds.data.numpy(),  test_ds.targets.numpy().astype(np.int64)

rng = np.random.default_rng(RANDOM_STATE)
order = rng.permutation(len(Xtr_u8))
val_idx, fit_idx = order[:5_000], order[5_000:]

X_fit_full, y_fit_full = flatten_scale(Xtr_u8[fit_idx]), ytr[fit_idx]
X_val,      y_val      = flatten_scale(Xtr_u8[val_idx]), ytr[val_idx]
X_test,     y_test     = flatten_scale(Xte_u8),          yte

SUB = 12_000
X_fit, y_fit = X_fit_full[:SUB], y_fit_full[:SUB]

assert len(X_fit_full) + len(X_val) == 60_000
assert X_fit.shape == (SUB, 784) and X_val.shape == (5_000, 784)
print(f"fit {len(X_fit):,} (of {len(X_fit_full):,})   "
      f"val {len(X_val):,}   test {len(X_test):,}")
'''),

        md("""
## 3 · The chain rule has two directions

Take a composition $L = f_3(f_2(f_1(\\theta)))$. The chain rule gives

$$\\frac{\\partial L}{\\partial \\theta} = J_3 J_2 J_1$$

a product of Jacobians. Matrix multiplication is associative, so you may
bracket it either way — and the two bracketings are two different algorithms
with two different costs.

* $\\left(J_3 J_2\\right) J_1$ — **reverse mode**: start at the output.
* $J_3 \\left(J_2 J_1\\right)$ — **forward mode**: start at the input.

Same answer. Verify that first, on a function you can differentiate by hand.
"""),
        prompt(
            label="autograd against a hand derivative",
            input="L = w², w = xy + sin x, at x=2, y=3",
            output="both partial derivatives, from autograd and by hand",
            constraint="float64 — at float32 the agreement is only to about 1e-7 and the 1e-12 assert would be measuring precision rather than correctness",
            check="assert agreement to 1e-12, which is machine precision for doubles. That summation over outgoing edges is the entire bookkeeping of reverse-mode autodiff. There is nothing else in it.",
            **{"try": "change the function and redo the hand derivative. If autograd and your algebra disagree, one of you is wrong and it is worth ten minutes to find out which."}),
        code('''
# L = w^2,  w = x*y + sin(x),  at x = 2, y = 3
x = torch.tensor(2.0, dtype=torch.float64, requires_grad=True)
y = torch.tensor(3.0, dtype=torch.float64, requires_grad=True)

w = x * y + torch.sin(x)
L = w ** 2
print(f"forward:  x*y = {(x*y).item():.4f}   sin x = {torch.sin(x).item():.4f}")
print(f"          w   = {w.item():.4f}       L = {L.item():.4f}")

L.backward()

# by hand: dL/dx = 2w (y + cos x),  dL/dy = 2w * x
hand_x = 2 * w.item() * (3 + np.cos(2))
hand_y = 2 * w.item() * 2
print(f"\\nautograd  dL/dx = {x.grad.item():.6f}   dL/dy = {y.grad.item():.6f}")
print(f"by hand   dL/dx = {hand_x:.6f}   dL/dy = {hand_y:.6f}")
assert abs(x.grad.item() - hand_x) < 1e-12
assert abs(y.grad.item() - hand_y) < 1e-12
print("\\nagree to machine precision")
'''),
        md("""
Note where the arithmetic went. `x` is used **twice** — once by the product and
once by the sine — so its derivative is a *sum* over the two paths:

$$41.4558 - 5.7506 = 35.7052$$

That summation over outgoing edges is the entire bookkeeping of reverse-mode
autodiff. There is nothing else in it.
"""),

        md("""
### The cost argument

Forward mode propagates a **tangent**: pick a direction in input space, push it
through, read the directional derivative of every output. One pass per *input*.

Reverse mode propagates an **adjoint**: pick a direction in output space, pull
it back, read the sensitivity of that output to every input. One pass per
*output*.

So for $n$ inputs and $m$ outputs:

| | passes | good when |
|---|---|---|
| forward | $n$ | few inputs |
| reverse | $m$ | few outputs |

**Training a network has $n$ = millions of parameters and $m$ = 1**, because the
loss is a single scalar. That is the whole argument, and it is why every deep
learning framework you will ever use is built around the reverse mode.

Measure it. `torch.func` exposes both directly.
"""),
        prompt(
            label="both modes, timed",
            input="tiny networks with 2 to 16 hidden units",
            output="the wall clock for the whole gradient in reverse mode against P passes in forward mode",
            constraint="assert the two modes AGREE before comparing their costs — a speed comparison between a right answer and a wrong one is not a comparison",
            check="reverse mode is flat in P and forward mode is linear in it. The ratio column is the whole reason every deep learning framework you will ever use is built around the reverse mode.",
            **{"try": "swap the input and output dimensions. The two modes trade places, which is the whole content of the cost table."}),
        code('''
from torch.func import jvp, vjp, functional_call

def flat_loss(net, X, y):
    """Turn a module into a pure function of one flat parameter vector."""
    names  = [n for n, _ in net.named_parameters()]
    shapes = [p.shape  for _, p in net.named_parameters()]
    sizes  = [p.numel() for _, p in net.named_parameters()]
    lossf  = nn.CrossEntropyLoss()
    def f(theta):
        out, i = {}, 0
        for n, s, k in zip(names, shapes, sizes):
            out[n] = theta[i:i + k].view(s); i += k
        return lossf(functional_call(net, out, (X,)), y)
    theta0 = torch.cat([p.detach().reshape(-1) for _, p in net.named_parameters()])
    return f, theta0

torch.manual_seed(RANDOM_STATE)
rows = []
for h in (2, 4, 8, 16):
    net = nn.Sequential(nn.Linear(20, h), nn.ReLU(), nn.Linear(h, 3))
    Xs, ys = torch.randn(64, 20), torch.randint(0, 3, (64,))
    f, theta = flat_loss(net, Xs, ys)
    P = theta.numel()

    t0 = time.perf_counter()
    _, pull = vjp(f, theta)
    g_rev = pull(torch.tensor(1.0))[0]
    t_rev = time.perf_counter() - t0

    basis = torch.eye(P)
    t0 = time.perf_counter()
    g_fwd = torch.stack([jvp(f, (theta,), (basis[i],))[1] for i in range(P)])
    t_fwd = time.perf_counter() - t0

    assert torch.allclose(g_rev, g_fwd, atol=1e-5), "the two modes disagree"
    rows.append((P, t_rev, t_fwd))
    print(f"P={P:4d}   reverse {t_rev*1e3:7.2f} ms (1 pass)   "
          f"forward {t_fwd*1e3:8.1f} ms ({P} passes)   ratio {t_fwd/t_rev:6.0f}x")
'''),
        md("""
The two columns compute **the same numbers** — the assertion says so. Reverse
mode is flat in $P$; forward mode is linear in it. Now put the real network's
parameter count into that ratio.
"""),
        prompt(
            label="the real parameter count, in that ratio",
            input="the actual 784-300-100-10 network",
            output="the measured reverse-mode gradient time, the measured cost of ONE forward direction, and the projected cost of all of them",
            constraint="do NOT run the projected figure — measure two per-pass costs and multiply, and say in the output that it is projected",
            check="label projected numbers as projected, in the output itself. A reader cannot tell from a printed float which of your numbers came from a clock.",
            **{"try": "work the ratio out for a network with a hundred outputs instead of one. Forward mode is still hopeless, and now you can say by how much."}),
        code('''
big = nn.Sequential(nn.Linear(784, 300), nn.ReLU(),
                    nn.Linear(300, 100), nn.ReLU(), nn.Linear(100, 10))
f, theta = flat_loss(big, torch.randn(128, 784), torch.randint(0, 10, (128,)))
P = theta.numel()

t0 = time.perf_counter()
_, pull = vjp(f, theta); pull(torch.tensor(1.0))
t_rev = time.perf_counter() - t0

tan = torch.randn_like(theta)
t0 = time.perf_counter()
jvp(f, (theta,), (tan,))
t_one_fwd = time.perf_counter() - t0

print(f"{P:,} parameters")
print(f"reverse mode, whole gradient:  {t_rev*1e3:.1f} ms   (measured)")
print(f"forward mode, one direction:   {t_one_fwd*1e3:.1f} ms   (measured)")
print(f"forward mode, all {P:,}:  about "
      f"{t_one_fwd * P / 60:,.0f} minutes   (projected, not run)")
'''),
        md("""
We did not run the projected figure, and the notebook says so. That is the
honest form of the claim: two measured per-pass costs and one multiplication
you can check.
"""),

        md("""
## 4 · What the previous lecture could not do

Time the Scikit-Learn model on this machine, as the control.

⏱ **about 30 seconds.**
"""),
        prompt(
            label="⏱ 30 s — the control",
            input="the same architecture, optimiser, batch size, learning rate and epochs",
            output="scikit-learn's wall clock and validation accuracy on THIS machine",
            constraint="time it here rather than quoting the previous lecture — the speed-up claimed below is a ratio, and both halves have to come from the same box",
            check="when you claim a speed-up, measure both halves in the same cell run, on the same data, with the same hyperparameters.",
            **{"try": "run it twice without changing the seed. Identical. Then change the seed and see how much of any later difference was ever real."}),
        code('''
from sklearn.neural_network import MLPClassifier

EPOCHS, BATCH, LR = 10, 128, 1e-3

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    sk = MLPClassifier(hidden_layer_sizes=(300, 100), solver="adam",
                       learning_rate_init=LR, batch_size=BATCH,
                       max_iter=EPOCHS, random_state=RANDOM_STATE)
    t0 = time.perf_counter(); sk.fit(X_fit, y_fit)
    sk_seconds = time.perf_counter() - t0

sk_val = sk.score(X_val, y_val)
print(f"Scikit-Learn: {sk_seconds:.1f} s for {EPOCHS} epochs, "
      f"validation accuracy {sk_val:.4f}")
'''),

        md("""
## 5 · Tensors, and where they live

A tensor is an array that also knows two things a NumPy array does not: which
device it is on, and whether to record what is done to it.
"""),
        prompt(
            label="tensors, and where they live",
            input="a small tensor",
            output="its dtype, device and requires_grad, then the same tensor moved to the accelerator, and a deliberate shape error",
            constraint="show `.to(device)` returning a NEW tensor — it does not move in place, and a discarded return value is a silent no-op",
            check="a shape error in PyTorch is loud, unlike almost everything else this course diagnoses. Take the ones you are given.",
            **{"try": "call `.numpy()` on a tensor that still requires grad. It refuses, and the error names exactly the reason."}),
        code('''
a = torch.tensor([[1., 2.], [3., 4.]])
print(a, a.dtype, a.device, a.requires_grad, sep="\\n")

b = a.to(device)
print(f"\\nmoved to {b.device}")
print("matmul:", (b @ b).flatten().tolist())

# NumPy interop is free and shares memory on CPU
print("\\nback to numpy:", (a.numpy() * 2).tolist())

# a shape error is loud, unlike most of what this course diagnoses
try:
    torch.randn(3, 4) @ torch.randn(3, 4)
except RuntimeError as exc:
    print(f"\\nshape error: {str(exc)[:80]}...")
'''),

        md("""
## 6 · Why `requires_grad` builds a graph

Setting `requires_grad=True` does not compute a derivative. It tells PyTorch to
**record every operation** applied to the tensor, building the computational
graph as the forward pass runs. `backward()` then walks that record backwards.

You can look at the record.
"""),
        prompt(
            label="what requires_grad actually does",
            input="a small chain of operations",
            output="the `grad_fn` of each intermediate, and the parents of the last one",
            constraint="print `grad_fn`, not the values — the record IS the thing being demonstrated",
            check="`torch.no_grad()` builds no record, which is why evaluation code is wrapped in it: less memory and no graph. Show the `None` rather than asserting the saving.",
            **{"try": "set `requires_grad=False` on one input and re-run. Its `.grad` stays None — the graph simply does not reach it."}),
        code('''
u = torch.tensor(2.0, requires_grad=True)
v = torch.tensor(3.0, requires_grad=True)
p = u * v
q = p + torch.sin(u)
r = q ** 2

for name, t in [("p = u*v", p), ("q = p + sin u", q), ("r = q**2", r)]:
    print(f"{name:16s} grad_fn = {t.grad_fn}")

print("\\nthe parents of r's node:", r.grad_fn.next_functions)
print("\\nno graph is built when you do not ask for one:")
with torch.no_grad():
    print("   inside no_grad:", (u * v).grad_fn)
print("   ...which is why evaluation code is wrapped in it: less memory, no record")
'''),

        md("""
## 7 · The training loop

Five lines, and every one of them is yours.
"""),
        prompt(
            label="the training loop, five lines, all yours",
            input="the data on the device, and the architecture",
            output="a trained network, its history, and the wall clock beside scikit-learn's",
            constraint="zero_grad, forward, loss, backward, step — in that order, inside the BATCH loop, with the permutation drawn from a seeded generator",
            check="assert the rebuild lands within a few points of scikit-learn — it should reproduce the model, not replace it. The assert comparing the two implementations. A rebuild that scores very differently has a bug, and without the assert you will read the difference as a finding.",
            **{"try": "reorder the five lines so the step comes before backward. Nothing raises, and the model does not learn."}),
        code('''
def make_net(dropout=0.0):
    layers = [nn.Linear(784, 300), nn.ReLU()]
    if dropout: layers.append(nn.Dropout(dropout))
    layers += [nn.Linear(300, 100), nn.ReLU()]
    if dropout: layers.append(nn.Dropout(dropout))
    layers.append(nn.Linear(100, 10))
    return nn.Sequential(*layers)

Xf = torch.tensor(X_fit, device=device);   yf = torch.tensor(y_fit, device=device)
Xv = torch.tensor(X_val, device=device);   yv = torch.tensor(y_val, device=device)
Xt = torch.tensor(X_test, device=device);  yt = torch.tensor(y_test, device=device)

@torch.no_grad()
def accuracy(net, X, y, batch=1000):
    """Counted over the WHOLE set, not averaged over batches. See section 11."""
    net.eval()
    hits = sum((net(X[i:i+batch]).argmax(1) == y[i:i+batch]).sum().item()
               for i in range(0, len(X), batch))
    return hits / len(X)

def train(epochs=EPOCHS, lr=LR, batch=BATCH, dropout=0.0,
          zero_grad=True, seed=RANDOM_STATE, track=True):
    torch.manual_seed(seed)
    net  = make_net(dropout).to(device)
    opt  = torch.optim.Adam(net.parameters(), lr=lr)
    lossf = nn.CrossEntropyLoss()
    g = torch.Generator().manual_seed(seed)
    hist = {"loss": [], "val_acc": []}
    t0 = time.perf_counter()
    for _ in range(epochs):
        net.train()
        perm = torch.randperm(len(Xf), generator=g).to(device)
        total, nb = 0.0, 0
        for i in range(0, len(Xf), batch):
            idx = perm[i:i + batch]
            if zero_grad:
                opt.zero_grad()                 # 1
            out  = net(Xf[idx])                 # 2
            loss = lossf(out, yf[idx])          # 3
            loss.backward()                     # 4
            opt.step()                          # 5
            total += loss.item(); nb += 1
        if track:
            hist["loss"].append(total / nb)
            hist["val_acc"].append(accuracy(net, Xv, yv))
    return net, hist, time.perf_counter() - t0

net, hist, pt_seconds = train()
n_params = sum(p.numel() for p in net.parameters())
print(f"{n_params:,} parameters")
print(f"PyTorch on {device}: {pt_seconds:.1f} s for {EPOCHS} epochs, "
      f"validation accuracy {hist['val_acc'][-1]:.4f}")
print(f"Scikit-Learn on CPU: {sk_seconds:.1f} s, {sk_val:.4f}")

# The deck's crossover table quotes a parameter count per architecture. The
# timings in it are durations and belong to whichever machine measured them
# (AUTHORING 3.2a); the parameter counts are arithmetic and belong to everyone,
# so compute them here rather than reading them off a slide.
print()
for hidden in ((100,), (300, 100), (1000, 1000), (2000, 2000)):
    sizes = (784, *hidden, 10)
    total = sum(sizes[i] * sizes[i + 1] + sizes[i + 1] for i in range(len(sizes) - 1))
    print(f"hidden {str(hidden):<14s} {total:>10,} parameters")
print(f"\\nspeed-up {sk_seconds / pt_seconds:.1f}x for the same architecture, "
      f"same optimiser, same epochs")
assert abs(hist["val_acc"][-1] - sk_val) < 0.05, \\
    "the rebuild should reproduce the model, not replace it"
'''),
        md("""
The accuracies agree to within a few points — as they must, because it is the
same model. **The rebuild is not an improvement.** What we bought is the loop.
"""),

        md("""
## 8 · ⚠ The missing `zero_grad()`

`backward()` **accumulates** into `p.grad`; it does not overwrite. Leave the
clearing line out and each step uses the sum of every gradient computed so far.

There is no exception, no warning, and no NaN. Read the code, then predict what
the curves will do, then run it.

⏱ **about 40 seconds** for the two runs.
"""),
        prompt(
            label="⏱ 40 s — ⚠ the missing zero_grad()",
            input="two runs, identical except for one line",
            output="both loss curves and both accuracy curves",
            constraint="log scale on the loss panel — without zero_grad the loss goes somewhere a linear axis cannot show alongside the healthy run",
            check="predict what the curves will do before you run it. If you cannot, you have not understood the accumulation, and reading the answer off the plot will not teach it to you.",
            **{"try": "move `zero_grad()` outside the batch loop, into the epoch loop. It is a much quieter bug than removing it — the curves look nearly right."}),
        code('''
net_ok,  hist_ok,  _ = train(zero_grad=True)
net_bad, hist_bad, _ = train(zero_grad=False)

fig, axes = plt.subplots(1, 2, figsize=(12, 3.6))
axes[0].plot(hist_ok["loss"],  label="with zero_grad()")
axes[0].plot(hist_bad["loss"], label="without", ls="--")
axes[0].set_yscale("log"); axes[0].set_xlabel("epoch")
axes[0].set_ylabel("training loss"); axes[0].legend()
axes[1].plot(hist_ok["val_acc"],  label="with zero_grad()")
axes[1].plot(hist_bad["val_acc"], label="without", ls="--")
axes[1].set_xlabel("epoch"); axes[1].set_ylabel("validation accuracy")
axes[1].set_ylim(0, 1); axes[1].legend()
plt.tight_layout(); plt.show()

cost = 100 * (hist_ok["val_acc"][-1] - hist_bad["val_acc"][-1])
print(f"with    {hist_ok['val_acc'][-1]:.4f}")
print(f"without {hist_bad['val_acc'][-1]:.4f}")
print(f"cost of one missing line: {cost:.2f} accuracy points, silently")
'''),
        prompt(
            label="watch it accumulate",
            input="three backward passes with no zeroing anywhere",
            output="the gradient norm after each, then after zero_grad, then with set_to_none=False",
            constraint="use a tiny probe module — four inputs and one output, so the norm is a number you can follow",
            check="three lines and a print is enough to make an accumulation bug visible. Build the smallest thing that shows the mechanism rather than reasoning about the big one.",
            **{"try": "print the gradient norm each step rather than the loss. The accumulation is arithmetic, and the norm shows it growing linearly."}),
        code('''
# why: watch the gradient of one weight accumulate across three backward passes
probe = nn.Linear(4, 1)
lossf = nn.MSELoss()
xb, yb = torch.randn(8, 4), torch.randn(8, 1)
for k in range(3):
    lossf(probe(xb), yb).backward()          # no zero_grad anywhere
    print(f"after {k+1} backward call(s): |grad| = {probe.weight.grad.norm():.4f}")
# zero_grad() defaults to set_to_none=True, so .grad becomes None rather than a
# tensor of zeros — reading .norm() on it raises AttributeError. That default is
# a performance choice, and it is the second way this bug bites: code that
# inspects gradients after zeroing them breaks rather than reporting zero.
probe.zero_grad()
print(f"after zero_grad():          grad is {probe.weight.grad}")

probe.zero_grad(set_to_none=False)
lossf(probe(xb), yb).backward()
probe.zero_grad(set_to_none=False)
print(f"after zero_grad(set_to_none=False): |grad| = "
      f"{probe.weight.grad.norm():.4f}")
'''),

        md("""
## 9 · ⚠ The missing `model.eval()`

Dropout zeroes a random fraction of each hidden layer **during training only**.
`model.eval()` switches it off. Forget it and you evaluate a randomly crippled
network — and get a different answer every time you ask.

⏱ **about 25 seconds.**
"""),
        prompt(
            label="⏱ 25 s — ⚠ the missing model.eval()",
            input="a network trained with dropout, evaluated both ways",
            output="the accuracy under eval(), and ten readings under train()",
            constraint="take TEN readings in training mode — one wrong number looks like a wrong number, ten different wrong numbers is a diagnosis",
            check="the SPREAD is the tell. A deterministic function of fixed weights and fixed data does not change between calls — if your metric wobbles, ask which layer is still in training mode.",
            **{"try": "set dropout=0.0, keep the model in train() mode, and "
                      "take the ten readings again. All ten agree, because "
                      "there is no longer a stochastic layer in the network. "
                      "The wobble diagnoses dropout specifically; a batch "
                      "norm left in train() mode fails this test only when "
                      "the batching changes between calls."}),
        code('''
net_d, hist_d, _ = train(dropout=0.2)

_, _, Xte_, yte_ = None, None, Xt, yt

net_d.eval()
with torch.no_grad():
    acc_eval = (net_d(Xte_).argmax(1) == yte_).float().mean().item()

net_d.train()                                   # the bug
torch.manual_seed(RANDOM_STATE)
with torch.no_grad():
    readings = [(net_d(Xte_).argmax(1) == yte_).float().mean().item()
                for _ in range(10)]

print(f"model.eval()   {acc_eval:.4f}")
print(f"model.train()  {np.mean(readings):.4f}  "
      f"(min {min(readings):.4f}, max {max(readings):.4f})")
print(f"\\ncost {100 * (acc_eval - np.mean(readings)):.2f} accuracy points")
print(f"and a spread of {100 * (max(readings) - min(readings)):.2f} points "
      f"across ten identical calls")
print("\\nThe spread is the tell. A deterministic function of fixed weights and")
print("fixed data does not change between calls. If your metric wobbles, ask")
print("which layer is still in training mode.")
net_d.eval()
'''),

        md("""
## 10 · `Dataset` and `DataLoader`

Indexing a big tensor by hand works while the data fits in memory. It stops
working the moment it does not — which is Lecture 15. `DataLoader` does the
shuffling, batching and (optionally) parallel loading.
"""),
        prompt(
            label="Dataset and DataLoader",
            input="the training tensors",
            output="a shuffling loader, its batch count, and the size of the last batch",
            constraint="seed the loader's generator — an unseeded shuffle makes every run of the notebook a different experiment",
            check="assert the loader yields every row, so nothing was silently dropped. Indexing a big tensor by hand works while the data fits in memory. It stops working the moment it does not, which is three lectures from here.",
            **{"try": "set `shuffle=False`. On a corpus ordered by class the loss curve becomes a sawtooth, one tooth per class."}),
        code('''
from torch.utils.data import TensorDataset, DataLoader

train_dl = DataLoader(TensorDataset(Xf.cpu(), yf.cpu()),
                      batch_size=BATCH, shuffle=True,
                      generator=torch.Generator().manual_seed(RANDOM_STATE))

xb, yb = next(iter(train_dl))
print(f"{len(train_dl)} batches of {BATCH}")
print(f"first batch: {tuple(xb.shape)}  labels {tuple(yb.shape)}")
print(f"last batch is short: {len(Xf)} = {len(Xf)//BATCH} x {BATCH} "
      f"+ {len(Xf) % BATCH}")
assert sum(len(b) for b, _ in train_dl) == len(Xf), "the loader dropped rows"
print("\\ndrop_last=False by default, so nothing is silently discarded —")
print("but the short last batch is about to matter. See the next section.")
'''),

        md("""
## 11 · ⚠ Averaging the metric per batch

The obvious way to compute validation accuracy is to average the per-batch
accuracies. That is **not** the accuracy over the set unless every batch is the
same size — and the last one never is.
"""),
        prompt(
            label="⚠ averaging the metric per batch",
            input="the test set, batched",
            output="the accuracy counted over the set, and the mean of the per-batch accuracies",
            constraint="compute BOTH in the same pass, so the difference cannot be attributed to anything else",
            check="weight by the batch size, or count over the set. Never take a plain mean of per-batch metrics — you cannot tell from the number whether it mattered.",
            **{"try": "make the batch size divide the corpus exactly. The bug vanishes, which is why it survives so long in code that was tested on a round number."}),
        code('''
def acc_two_ways(net, X, y, batch=384):
    net.eval(); per_batch, hits = [], 0
    with torch.no_grad():
        for i in range(0, len(X), batch):
            xb, yb = X[i:i+batch], y[i:i+batch]
            h = (net(xb).argmax(1) == yb).sum().item()
            hits += h; per_batch.append(h / len(xb))
    return hits / len(X), float(np.mean(per_batch)), len(per_batch), len(xb)

over_set, mean_batches, nb, last = acc_two_ways(net, Xt, yt)
print(f"{nb} batches, the last containing {last} images")
print(f"accuracy over the set    {over_set:.5f}")
print(f"mean of batch accuracies {mean_batches:.5f}")
print(f"difference {100*(mean_batches - over_set):+.4f} accuracy points")
'''),
        md("""
Small — here. It is small because the last batch is not that short and the model
is not much better or worse on it than on any other. Change either and it grows;
change the batch size to 3,000 on a 10,000-image set and the last batch carries
a quarter of the weight instead of a tenth.

The decision rule, which is the point: **weight by the batch size, or count over
the set.** Never take a plain mean of per-batch metrics, because you cannot tell
from the number whether it mattered.
"""),

        md("""
## 12 · A custom module

`nn.Sequential` runs out of road as soon as the forward pass is not a straight
line. Subclass `nn.Module` and write `forward` yourself; everything else —
parameter registration, `.to(device)`, `state_dict` — comes for free.
"""),
        prompt(
            label="a custom module",
            input="the same architecture, written as a subclass",
            output="the module's repr and its parameter count",
            constraint="`nn.ModuleList`, not a plain Python list — a list of layers is not registered, so its parameters never reach the optimiser and never move with `.to(device)`",
            check="assert the parameter count matches the Sequential version exactly — same model, written differently. Assert the parameter count against a known-good implementation. An unregistered layer is invisible in the repr and in the loss, and visible immediately in the count.",
            **{"try": "add a second hidden layer by editing the module rather than a constructor argument. That edit is the thing Lecture 9 could not do."}),
        code('''
class Sorter(nn.Module):
    def __init__(self, widths=(300, 100), n_in=784, n_out=10, dropout=0.0):
        super().__init__()
        sizes = (n_in,) + tuple(widths)
        self.blocks = nn.ModuleList(
            nn.Linear(a, b) for a, b in zip(sizes, sizes[1:]))
        self.drop = nn.Dropout(dropout) if dropout else nn.Identity()
        self.head = nn.Linear(sizes[-1], n_out)

    def forward(self, x):
        for blk in self.blocks:
            x = self.drop(torch.relu(blk(x)))
        return self.head(x)

m = Sorter().to(device)
print(m)
print(f"\\n{sum(p.numel() for p in m.parameters()):,} parameters, registered "
      f"automatically")
assert sum(p.numel() for p in m.parameters()) == n_params
print("same count as the Sequential version — same model, written differently")
'''),

        md("""
## 13 · Hyperparameter search with Optuna

Lecture 11 tuned by hand. Optuna does the same search with a sampler that
learns from the trials it has already run. **Not examinable** — it is not in
the book — but it is what you will actually use.

⏱ **about 90 seconds** for 8 short trials.
"""),
        prompt(
            label="the optional dependency",
            input="nothing",
            output="the optuna version, or an explanation and the install command",
            constraint="degrade gracefully — a missing optional package must not stop the notebook, and the fallback branch has to say what the thing WOULD have done",
            check="if a cell can fail for an environmental reason, catch it and print the remedy. A traceback is not instructions.",
            **{"try": "skip the install and run the cell anyway. It says what is missing and why, rather than failing sixty lines later."}),
        code('''
try:
    import optuna
    HAVE_OPTUNA = True
    print(f"optuna {optuna.__version__}")
except ImportError:
    HAVE_OPTUNA = False
    print("optuna is not installed here.  %pip install optuna")
    print("The idea, in one sentence: propose a configuration, train it briefly,")
    print("score it, and let a sampler propose the next one from what it learned.")
'''),
        prompt(
            label="⏱ 90 s — search, with a sampler that learns",
            input="learning rate on a log scale and dropout on a grid, 8 trials",
            output="the best configuration found, beside the hand-tuned baseline",
            constraint="`log=True` on the learning rate — it lives on a multiplicative scale, and a uniform sampler spends most of its trials between 0.005 and 0.01",
            check="seed the sampler. An unseeded search gives a different answer every time and you cannot tell improvement from resampling.",
            **{"try": "run it again with `sampler=optuna.samplers.RandomSampler()`. Same budget, worse best value — the sampler is doing real work."}),
        code('''
if HAVE_OPTUNA:
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        lr   = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        drop = trial.suggest_float("dropout", 0.0, 0.5, step=0.1)
        _, h, _ = train(epochs=4, lr=lr, dropout=drop)
        return h["val_acc"][-1]

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=8)

    print(f"best {study.best_value:.4f} at {study.best_params}")
    print(f"hand-tuned baseline was {hist['val_acc'][-1]:.4f} "
          f"after {EPOCHS} epochs")
    print("\\nThe trials ran 4 epochs each, so this is not a like-for-like")
    print("comparison — it is a search over configurations, not a final model.")
'''),

        md("""
## 14 · Save it

`state_dict()` — the tensors, not the class. Saving the whole object pickles
your source file with it, and the file will not load next term when the class
has moved.
"""),
        prompt(
            label="save it",
            input="the trained network",
            output="a checkpoint on disk, reloaded into a fresh network and checked",
            constraint="`state_dict()` — the TENSORS, not the class. Saving the whole object pickles your source file with it, and the file will not load next term when the class has moved",
            check="assert the reloaded model gives EXACTLY the same accuracy — identical, not close. An exact equality assert is available here because loading weights is deterministic. Assert it, do not eyeball it.",
            **{"try": "reload into a model whose architecture differs by one layer. It refuses, and the message names the mismatched keys."}),
        code('''
from pathlib import Path

Path("checkpoints").mkdir(exist_ok=True)
torch.save(net.state_dict(), "checkpoints/sorter.pt")
size_kb = Path("checkpoints/sorter.pt").stat().st_size / 1024
print(f"checkpoints/sorter.pt   {size_kb:,.0f} KB")

# reload into a fresh network and check it is the same function
fresh = make_net().to(device)
fresh.load_state_dict(torch.load("checkpoints/sorter.pt", weights_only=True))
a1, a2 = accuracy(net, Xt, yt), accuracy(fresh, Xt, yt)
print(f"original {a1:.4f}   reloaded {a2:.4f}")
assert a1 == a2, "the reloaded model is not the same function"
print("identical — assert it, do not eyeball it")
'''),

        md("""
## 15 · Re-measure

Compare with the sheet of paper from the previous lecture.
"""),
        prompt(
            label="re-measure",
            input="everything measured",
            output="baseline, both validation numbers with their wall clocks, and the test accuracy",
            constraint="label the test number as the one you report, and say that the test set has now been touched once",
            check="compare with the sheet of paper from the previous lecture. A prediction you can silently revise is not a prediction.",
            **{"try": "compare against the Lecture 9 accuracy. The gain is real but small; the point of this lecture was never the accuracy."}),
        code('''
final_test = accuracy(net, Xt, yt)
print(f"baseline (majority class)   0.1000")
print(f"Scikit-Learn, validation    {sk_val:.4f}   in {sk_seconds:.1f} s")
print(f"PyTorch, validation         {hist['val_acc'][-1]:.4f}   "
      f"in {pt_seconds:.1f} s on {device}")
print(f"PyTorch, TEST               {final_test:.4f}   <- the number you report")
print("\\nThe test set has now been touched once. Do not tune against it.")
'''),

        md("""
## 16 · Where we are

- Reverse-mode differentiation costs one sweep per **output**, and training has
  one output — a scalar loss — and millions of inputs. That asymmetry is the
  whole reason training is affordable.
- You wrote the loop: forward, loss, `backward()`, step. Nothing in it is
  hidden any more.
- Then you replaced its parts with `DataLoader`, a custom `Module` and a proper
  evaluation path — knowing, this time, what each replaced.

**Three questions to ask of any training loop**, yours or anyone else's. All
three failures are silent, and you have now watched each one happen:

1. Is `opt.zero_grad()` inside the **batch** loop? Inside the *epoch* loop is a
   different and much quieter bug.
2. Is there a `model.eval()` before every evaluation, and a `model.train()`
   before training resumes?
3. Is any reported metric a plain mean of per-batch values? The last batch is
   usually short, so that mean is weighted wrongly.

**Before the next lecture:** run this notebook top to bottom. Then make the
network four times deeper and re-run the training loop. It trains far worse than
its size suggests it should — and Lecture 11 derives why, and repairs it.
"""),
    ]
