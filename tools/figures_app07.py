#!/usr/bin/env python3
"""
Application 7 — Lectures 13 and 14. CIFAR-10, a twenty-layer stack that will not
train, and the variance argument that explains why.

    python3 tools/figures_app07.py

Everything printed on slides/lecture-13.html and slides/lecture-14.html comes
from here, via figkit.export() into assets/figures/figures.json. Expensive runs
are cached (figkit.cached, which shares
/private/tmp/claude-501/aiml-data/fits-v2.pkl with the other applications —
hence the `app07_` prefix on every key) so a cosmetic re-run takes seconds.

Two measurement decisions worth stating, because both were nearly got wrong.

  * **The gradient probe runs on the CPU, in float64 as well as float32.** The
    deepest layers' gradients land at 1e-17, which is comfortably inside
    float32's range — but that had to be *checked*, not assumed, because a norm
    is a sum of squares and 1e-20 squared underflows to exactly zero. The two
    dtypes are compared and the largest relative disagreement is exported.

  * **Every training run uses the same seed, the same subset and the same
    number of epochs**, so the ablation ladder is a comparison of one change at
    a time rather than of one lucky run against another.

Timings are wall-clock on the machine that generated the figures — an Apple
Silicon laptop with an MPS backend and no CUDA. They are measurements, not
guarantees, and the decks say so.
"""

from __future__ import annotations

import math
import sys
import time
import xml.dom.minidom
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))

from figkit import (setup, save, cached, load_cache, export, OUT, SEED,   # noqa: E402
                    PRIMARY, ACCENT, SUCCESS, MATH, MUTED, RULE, AXIS,
                    BODY, SMALL, TICK, check_text_floor, plain_log)


class _Pow10:
    """A format-string stand-in for figkit.plain_log's `fmt`.

    Every log axis in this application spans between four and eighteen decades,
    so a digit format ("{:,.0f}") is useless: the labels would be 0 and
    100000000000000000000. plain_log calls `fmt.format(v)`, and anything with a
    `.format` method satisfies that, so this writes the decade with unicode
    superscripts instead. The three static Source Sans 3 cuts all carry U+207B
    and U+2070-2079; checked, not assumed.
    """

    _SUP = str.maketrans("-0123456789", "\u207b\u2070\u00b9\u00b2\u00b3"
                                        "\u2074\u2075\u2076\u2077\u2078"
                                        "\u2079")

    def format(self, v: float) -> str:
        e = int(round(math.log10(v)))
        return "10" + str(e).translate(self._SUP)


POW10 = _Pow10()


def decades(values, step=4):
    """Decade ticks spanning `values`, thinned so the labels do not collide."""
    a = np.asarray([v for v in np.ravel(values) if v > 0])
    lo, hi = int(np.floor(np.log10(a.min()))), int(np.ceil(np.log10(a.max())))
    ticks = list(range(lo, hi + 1, step))
    if ticks[-1] != hi:
        ticks.append(hi)
    return [10.0 ** e for e in ticks]

import torch                                                    # noqa: E402
import torch.nn as nn                                           # noqa: E402
import torchvision                                              # noqa: E402

DATA = Path("/private/tmp/claude-501/aiml-data/cifar")

CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]

# The architecture the whole application is built around. Twenty hidden layers
# is a deliberate choice, not a good one.
DEPTH = 20            # hidden layers
WIDTH = 100           # units in each
N_IN = 32 * 32 * 3    # 3,072 — colour, so three times MNIST's pixel count
N_OUT = 10

N_FIT = 10_000        # a subset, so a free Colab runtime finishes in the hour
N_VAL = 5_000
EPOCHS = 20
BATCH = 128
LR = 1e-3             # Adam, exactly as Lecture 12 left it

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

ACTS = {"sigmoid": nn.Sigmoid, "tanh": nn.Tanh, "relu": nn.ReLU,
        "leaky": lambda: nn.LeakyReLU(0.01), "elu": nn.ELU, "selu": nn.SELU}


# ------------------------------------------------------------------ the data

def load_cifar() -> dict:
    """The 60,000 images, flattened to 3,072 floats and standardised.

    The scaling statistics are computed on the FIT subset only. That is the
    Lecture 2 rule, restated: nothing derived from data we then score on.
    """
    tr = torchvision.datasets.CIFAR10(DATA, train=True, download=True)
    te = torchvision.datasets.CIFAR10(DATA, train=False, download=True)

    Xtr_u8 = tr.data                                  # (50000, 32, 32, 3) uint8
    Xte_u8 = te.data
    ytr = np.asarray(tr.targets, dtype=np.int64)
    yte = np.asarray(te.targets, dtype=np.int64)

    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(Xtr_u8))
    val_idx = order[:N_VAL]
    fit_idx = order[N_VAL:N_VAL + N_FIT]

    flat = lambda a: a.reshape(len(a), -1).astype(np.float32) / 255.0
    X_fit_raw = flat(Xtr_u8[fit_idx])
    mu = X_fit_raw.mean(axis=0)
    sd = X_fit_raw.std(axis=0) + 1e-7

    std = lambda a: (flat(a) - mu) / sd
    return {
        "X_fit": std(Xtr_u8[fit_idx]), "y_fit": ytr[fit_idx],
        "X_val": std(Xtr_u8[val_idx]), "y_val": ytr[val_idx],
        "X_test": std(Xte_u8), "y_test": yte,
        "grid_images": Xtr_u8, "grid_labels": ytr,
        "train_counts": np.bincount(ytr, minlength=10),
        "test_counts": np.bincount(yte, minlength=10),
        "fit_counts": np.bincount(ytr[fit_idx], minlength=10),
        "one_image": Xtr_u8[7],
        "pixel_mean": float(X_fit_raw.mean()),
        "pixel_sd": float(X_fit_raw.std()),
    }


# ------------------------------------------------------------------ the model

def make_net(depth=DEPTH, width=WIDTH, act="sigmoid", init="torch",
             norm=None, dropout=0.0, n_in=N_IN, n_out=N_OUT,
             dtype=torch.float32) -> nn.Sequential:
    """The stack the lecture builds, with every knob the next lecture turns.

    `init="torch"` is the point of Lecture 13: it is not a choice anyone made.
    It is what `nn.Linear` does when the specification says nothing, namely
    U(-1/sqrt(fan_in), +1/sqrt(fan_in)).
    """
    layers: list[nn.Module] = []
    prev = n_in
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
            pass                                   # whatever the default is
        elif init == "normal1":
            nn.init.normal_(m.weight, 0.0, 1.0); nn.init.zeros_(m.bias)
        elif init == "glorot":
            nn.init.xavier_normal_(m.weight); nn.init.zeros_(m.bias)
        elif init == "he":
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            nn.init.zeros_(m.bias)
        else:
            raise ValueError(init)
    return net


def n_params(net) -> int:
    return sum(p.numel() for p in net.parameters())


def linear_layers(net) -> list[nn.Linear]:
    return [m for m in net if isinstance(m, nn.Linear)]


# --------------------------------------------------------- instrumentation

def grad_profile(X, y, *, n_batches=8, dtype=torch.float64, **kw) -> np.ndarray:
    """Mean gradient norm of every weight matrix, at initialisation.

    Averaged over several batches because a single batch of 128 is a noisy
    estimate of anything, and a course that says so should not then quote one.
    Run on the CPU: MPS has no float64, and the whole point is to see how small
    these get before the arithmetic stops being able to represent it.
    """
    torch.manual_seed(SEED)
    net = make_net(dtype=dtype, **kw)
    net.train()
    lossf = nn.CrossEntropyLoss()
    lins = linear_layers(net)
    acc = np.zeros(len(lins))
    g = torch.Generator().manual_seed(SEED)
    for b in range(n_batches):
        idx = torch.randperm(len(X), generator=g)[:BATCH]
        xb = torch.as_tensor(X[idx.numpy()], dtype=dtype)
        yb = torch.as_tensor(y[idx.numpy()])
        net.zero_grad()
        lossf(net(xb), yb).backward()
        acc += np.array([float(m.weight.grad.norm()) for m in lins])
    return acc / n_batches


def delta_profile(X, y, *, n_batches=8, dtype=torch.float64, **kw) -> np.ndarray:
    """Norm of the gradient with respect to each layer's OUTPUT, not its weights.

    This distinction cost an afternoon and is worth the extra function. The
    thread predicts the per-layer factor for the *backward signal*

        delta_l = dL/dz_l ,

    but what a practitioner logs — and what Lecture 13 plots — is the gradient
    of the WEIGHTS, and

        || dL/dW_l ||  ~  || delta_l || . || a_(l-1) || .

    So the weight-gradient ratio is the backward factor divided by the forward
    activation-scale factor. Those two are equal only when the forward pass is
    scale-stable, which the logistic is (it settles at a fixed point) and an
    unnormalised ReLU stack is not. Measure both, and the discrepancy stops
    being a mystery and becomes the second half of the thread.
    """
    torch.manual_seed(SEED)
    net = make_net(dtype=dtype, **kw)
    net.train()
    lossf = nn.CrossEntropyLoss()
    g = torch.Generator().manual_seed(SEED)
    acc = None
    for _ in range(n_batches):
        idx = torch.randperm(len(X), generator=g)[:BATCH]
        h = torch.as_tensor(X[idx.numpy()], dtype=dtype)
        yb = torch.as_tensor(y[idx.numpy()])
        zs = []
        for m in net:
            h = m(h)
            if isinstance(m, nn.Linear):
                h.retain_grad()
                zs.append(h)
        net.zero_grad()
        lossf(h, yb).backward()
        vals = np.array([float(z.grad.norm()) for z in zs])
        acc = vals if acc is None else acc + vals
    return acc / n_batches


def act_profile(X, *, dtype=torch.float64, **kw) -> dict:
    """Mean, sd and saturated fraction of every hidden layer's output."""
    torch.manual_seed(SEED)
    net = make_net(dtype=dtype, **kw)
    net.eval()
    h = torch.as_tensor(X[:BATCH], dtype=dtype)
    means, sds, signal, sat = [], [], [], []
    with torch.no_grad():
        for m in net:
            h = m(h)
            if isinstance(m, tuple(t for t in (nn.Sigmoid, nn.Tanh, nn.ReLU,
                                               nn.LeakyReLU, nn.ELU, nn.SELU))):
                means.append(float(h.mean()))
                # Two different questions, and only one of them is the one we
                # are asking. h.std() over the whole tensor is dominated by the
                # spread of the layer's random BIASES across units, which does
                # not change with depth — so it sits near 0.07 at layer 20 and
                # the network looks alive. What carries information is how much
                # a unit's output moves when the INPUT changes, which is the sd
                # down the batch, averaged over units. That falls by a factor of
                # 0.14 a layer here: exactly the rate Thread 7 predicts from the
                # fan-in, and exactly the rate the gradient falls at.
                sds.append(float(h.std()))
                signal.append(float(h.std(dim=0).mean()))
                # "saturated" means the activation's own derivative has
                # collapsed: for the logistic that is |a - 1/2| > 0.45.
                if isinstance(m, nn.Sigmoid):
                    sat.append(float(((h - 0.5).abs() > 0.45).double().mean()))
                elif isinstance(m, nn.Tanh):
                    sat.append(float((h.abs() > 0.9).double().mean()))
                else:
                    sat.append(float((h == 0).double().mean()))
    return {"mean": means, "sd": sds, "sd_signal": signal,
            "saturated": sat}


def layer_hist(X, layers=(1, 10, 20), **kw) -> dict:
    """The distribution of one layer's activations, for three depths."""
    torch.manual_seed(SEED)
    net = make_net(dtype=torch.float64, **kw)
    net.eval()
    h = torch.as_tensor(X[:512], dtype=torch.float64)
    out, k = {}, 0
    with torch.no_grad():
        for m in net:
            h = m(h)
            if isinstance(m, (nn.Sigmoid, nn.Tanh, nn.ReLU)):
                k += 1
                if k in layers:
                    out[k] = h.flatten().numpy().copy()
    return out


# --------------------------------------------------------------- training

def train(d, *, epochs=EPOCHS, lr=LR, batch=BATCH, opt="adam", clip=None,
          schedule=None, track_grads=False, seed=SEED, **kw) -> dict:
    """One run. Same seed, same subset, same epochs — so rows are comparable."""
    torch.manual_seed(seed)
    net = make_net(**kw).to(DEVICE)
    lossf = nn.CrossEntropyLoss()

    params = net.parameters()
    if opt == "adam":
        optim = torch.optim.Adam(params, lr=lr)
    elif opt == "adamw":
        optim = torch.optim.AdamW(params, lr=lr)
    elif opt == "sgd":
        optim = torch.optim.SGD(params, lr=lr)
    elif opt == "momentum":
        optim = torch.optim.SGD(params, lr=lr, momentum=0.9)
    elif opt == "nesterov":
        optim = torch.optim.SGD(params, lr=lr, momentum=0.9, nesterov=True)
    elif opt == "rmsprop":
        optim = torch.optim.RMSprop(params, lr=lr)
    else:
        raise ValueError(opt)

    Xf = torch.tensor(d["X_fit"], device=DEVICE)
    yf = torch.tensor(d["y_fit"], device=DEVICE)
    n_steps = math.ceil(len(Xf) / batch) * epochs
    sched = None
    if schedule == "onecycle":
        sched = torch.optim.lr_scheduler.OneCycleLR(
            optim, max_lr=10 * lr, total_steps=n_steps)
    elif schedule == "cosine":
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=n_steps)
    elif schedule == "exp":
        sched = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=0.97)

    lins = linear_layers(net)
    hist = {"loss": [], "val_acc": [], "lr": [], "grad": []}
    g = torch.Generator().manual_seed(seed)
    t0 = time.perf_counter()
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(len(Xf), generator=g).to(DEVICE)
        total, nb = 0.0, 0
        for i in range(0, len(Xf), batch):
            idx = perm[i:i + batch]
            optim.zero_grad()
            loss = lossf(net(Xf[idx]), yf[idx])
            loss.backward()
            if clip is not None:
                torch.nn.utils.clip_grad_norm_(net.parameters(), clip)
            optim.step()
            if sched is not None and schedule != "exp":
                sched.step()
            total += float(loss.item()); nb += 1
        if sched is not None and schedule == "exp":
            sched.step()
        hist["loss"].append(total / nb)
        hist["lr"].append(optim.param_groups[0]["lr"])
        hist["val_acc"].append(accuracy(net, d["X_val"], d["y_val"]))
        if track_grads:
            hist["grad"].append([float(m.weight.grad.norm()) for m in lins])
    seconds = time.perf_counter() - t0

    return {
        "loss": hist["loss"], "val_acc": hist["val_acc"], "lr": hist["lr"],
        "grad": hist["grad"],
        "first_loss": hist["loss"][0], "last_loss": hist["loss"][-1],
        "best_val": max(hist["val_acc"]), "final_val": hist["val_acc"][-1],
        "test_acc": accuracy(net, d["X_test"], d["y_test"]),
        "seconds": seconds, "n_params": n_params(net),
        "epochs": epochs, "device": DEVICE,
    }


@torch.no_grad()
def accuracy(net, X, y, batch=2000) -> float:
    """Counted over the whole set, never a mean of per-batch means. Lecture 12."""
    net.eval()
    Xt = torch.tensor(X, device=DEVICE)
    yt = torch.tensor(y, device=DEVICE)
    hits = 0
    for i in range(0, len(Xt), batch):
        hits += int((net(Xt[i:i + batch]).argmax(1) == yt[i:i + batch]).sum())
    net.train()
    return hits / len(Xt)


def grad_norm_distribution(d, **kw) -> list:
    """Every per-step total gradient norm in one epoch, for the clipping slide."""
    torch.manual_seed(SEED)
    net = make_net(**kw).to(DEVICE)
    lossf = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(net.parameters(), lr=LR)
    Xf = torch.tensor(d["X_fit"], device=DEVICE)
    yf = torch.tensor(d["y_fit"], device=DEVICE)
    g = torch.Generator().manual_seed(SEED)
    norms = []
    net.train()
    for _ in range(3):
        perm = torch.randperm(len(Xf), generator=g).to(DEVICE)
        for i in range(0, len(Xf), BATCH):
            idx = perm[i:i + BATCH]
            optim.zero_grad()
            lossf(net(Xf[idx]), yf[idx]).backward()
            norms.append(float(torch.nn.utils.clip_grad_norm_(
                net.parameters(), float("inf"))))
            optim.step()
    return norms


# ----------------------------------------------------------- the experiments

def run_l13(d) -> dict:
    """Everything Lecture 13 measures."""
    out = {}
    print("    the twenty-layer stack, as specified")
    out["deep"] = _strip(train(d, act="sigmoid", init="torch", track_grads=True))
    print("    a two-layer control, same code, same everything else")
    out["shallow"] = _strip(train(d, depth=2, act="sigmoid", init="torch"))
    print("    the depth sweep")
    out["depth_sweep"] = {}
    for k in (1, 2, 5, 10, 20):
        r = _strip(train(d, depth=k, act="sigmoid", init="torch"))
        out["depth_sweep"][str(k)] = {"test_acc": r["test_acc"],
                                      "last_loss": r["last_loss"],
                                      "n_params": r["n_params"],
                                      "seconds": r["seconds"]}
        print(f"      depth {k:2d}: test {r['test_acc']:.4f}  "
              f"loss {r['last_loss']:.4f}")
    print("    the gradient profile, float64 and float32")
    out["grad64"] = grad_profile(d["X_fit"], d["y_fit"],
                                 dtype=torch.float64).tolist()
    out["grad32"] = grad_profile(d["X_fit"], d["y_fit"],
                                 dtype=torch.float32).tolist()
    print("    the activation profile")
    out["acts"] = act_profile(d["X_fit"])
    return out


def run_l13_extra(d) -> dict:
    """How far each layer's weights actually moved over the whole run.

    A gradient norm is a rate; this is the distance travelled. They are not the
    same claim and the slide makes the second one, so it is measured
    separately rather than inferred.
    """
    torch.manual_seed(SEED)
    before = [m.weight.detach().clone()
              for m in linear_layers(make_net(act="sigmoid", init="torch"))]

    torch.manual_seed(SEED)
    net = make_net(act="sigmoid", init="torch").to(DEVICE)
    lossf = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(net.parameters(), lr=LR)
    Xf = torch.tensor(d["X_fit"], device=DEVICE)
    yf = torch.tensor(d["y_fit"], device=DEVICE)
    g = torch.Generator().manual_seed(SEED)
    for _ in range(EPOCHS):
        net.train()
        perm = torch.randperm(len(Xf), generator=g).to(DEVICE)
        for i in range(0, len(Xf), BATCH):
            idx = perm[i:i + BATCH]
            optim.zero_grad()
            lossf(net(Xf[idx]), yf[idx]).backward()
            optim.step()
    after = [m.weight.detach().cpu() for m in linear_layers(net)]

    # The assistant's version of the same network, run for the five epochs its
    # snippet asks for. The deck prints its console output verbatim, so it has
    # to be produced here rather than typed from memory (TRICKS section 4).
    assistant = _strip(train(d, epochs=5, act="sigmoid", init="torch"))

    # Can the loop memorise a handful of examples? If not, the bug is the loop.
    # The slide claims a number for this, so measure it.
    n_mem, steps_mem = 200, 200
    torch.manual_seed(SEED)
    tiny = make_net(depth=2, act="sigmoid", init="torch").to(DEVICE)
    opt2 = torch.optim.Adam(tiny.parameters(), lr=LR)
    Xm = torch.tensor(d["X_fit"][:n_mem], device=DEVICE)
    ym = torch.tensor(d["y_fit"][:n_mem], device=DEVICE)
    tiny.train()
    for _ in range(steps_mem):
        opt2.zero_grad()
        lossf(tiny(Xm), ym).backward()
        opt2.step()
    memorise = accuracy(tiny, d["X_fit"][:n_mem], d["y_fit"][:n_mem])

    return {"wchange": [float((a - b).norm() / b.norm())
                        for a, b in zip(after, before)],
            "assistant": {"loss": assistant["loss"],
                          "test_acc": assistant["test_acc"],
                          "seconds": assistant["seconds"],
                          "epochs": 5},
            "memorise": {"n": n_mem, "steps": steps_mem, "acc": memorise}}


def run_deltas(d) -> dict:
    """The true backward factor rho, and the forward scale factor beside it."""
    schemes = {
        "default_sigmoid": dict(act="sigmoid", init="torch"),
        "normal1_sigmoid": dict(act="sigmoid", init="normal1"),
        "glorot_sigmoid":  dict(act="sigmoid", init="glorot"),
        "glorot_relu":     dict(act="relu",    init="glorot"),
        "he_relu":         dict(act="relu",    init="he"),
    }
    out = {}
    for name, kw in schemes.items():
        delta = delta_profile(d["X_fit"], d["y_fit"], **kw)
        sd = np.array(act_profile(d["X_fit"], **kw)["sd"])
        # delta[l] is the gradient at layer l's output; descending the stack
        # means going from index l to index l-1.
        r_back = delta[0:DEPTH - 1] / delta[1:DEPTH]
        r_fwd = sd[1:DEPTH] / sd[0:DEPTH - 1]
        out[name] = {
            "delta": [float(v) for v in delta],
            "rho": float(np.exp(np.mean(np.log(r_back)))),
            "forward_ratio": float(np.exp(np.mean(np.log(r_fwd)))),
            "delta_attenuation": float(delta[0] / delta[DEPTH - 1]),
        }
        print(f"      {name:16s} rho {out[name]['rho']:.4f}   "
              f"forward scale {out[name]['forward_ratio']:.4f}   "
              f"delta_1/delta_20 {out[name]['delta_attenuation']:.3e}")
    return out


def run_timing(d, repeats: int = 3, epochs: int = 3) -> dict:
    """What normalisation costs on the clock, measured robustly.

    The figures for this application were generated on a heavily shared
    machine, where a single wall-clock reading says more about what else was
    running than about the configuration. The minimum of several repeats is the
    standard defence: contention can only ever make a run slower, so the
    smallest observation is the closest to the uncontended cost.

    Three epochs rather than twenty, because this measures the per-epoch cost
    and nothing else.
    """
    out = {"repeats": repeats, "epochs": epochs}
    for label, kw in [("none", {}), ("batch", dict(norm="batch")),
                      ("layer", dict(norm="layer"))]:
        times = [train(d, epochs=epochs, act="relu", init="he", **kw)["seconds"]
                 for _ in range(repeats)]
        out[label] = {"min": float(min(times)), "all": [float(t) for t in times]}
        print(f"      {label:6s} {min(times):.2f} s for {epochs} epochs "
              f"(min of {repeats}; saw {[round(t, 2) for t in times]})")
    out["batch_overhead"] = out["batch"]["min"] / out["none"]["min"]
    out["layer_overhead"] = out["layer"]["min"] / out["none"]["min"]
    return out


def run_l14_extra(d) -> dict:
    """The Lecture 14 assistant failure: Glorot's constant, on ReLU.

    Xavier and Glorot are the same person and the same formula, so a request to
    'initialise it properly' with ReLU very often returns xavier_uniform_. It
    is not nothing — it is exactly a factor of sqrt(2) per layer wrong, which
    is three orders of magnitude over nineteen layers rather than fifteen.
    """
    out = {}
    out["grad_glorot_relu"] = grad_profile(d["X_fit"], d["y_fit"],
                                           act="relu", init="glorot").tolist()
    out["glorot_relu"] = _strip(train(d, act="relu", init="glorot"))
    out["he_relu"] = _strip(train(d, act="relu", init="he"))
    print(f"      Xavier + ReLU test {out['glorot_relu']['test_acc']:.4f}   "
          f"He + ReLU test {out['he_relu']['test_acc']:.4f}")
    return out


def run_l14(d) -> dict:
    """Everything Lecture 14 measures."""
    out = {}

    print("    gradient profiles for four initialisations")
    schemes = {
        "default_sigmoid": dict(act="sigmoid", init="torch"),
        "normal1_sigmoid": dict(act="sigmoid", init="normal1"),
        "glorot_sigmoid":  dict(act="sigmoid", init="glorot"),
        "he_relu":         dict(act="relu",    init="he"),
    }
    out["grads"] = {k: grad_profile(d["X_fit"], d["y_fit"], **v).tolist()
                    for k, v in schemes.items()}
    out["acts"] = {k: act_profile(d["X_fit"], **v) for k, v in schemes.items()}

    print("    activation variance against depth, for five weight scales")
    out["var_sweep"] = _var_sweep(d)

    print("    the ablation ladder")
    ladder = [
        ("Lecture 13, unchanged", dict(act="sigmoid", init="torch")),
        ("+ Glorot initialisation", dict(act="sigmoid", init="glorot")),
        ("+ ReLU and He initialisation", dict(act="relu", init="he")),
        ("+ batch normalisation", dict(act="relu", init="he", norm="batch")),
        ("+ gradient clipping", dict(act="relu", init="he", norm="batch",
                                     clip=1.0)),
        ("+ a 1-cycle schedule", dict(act="relu", init="he", norm="batch",
                                      clip=1.0, schedule="onecycle")),
        ("+ dropout 0.1", dict(act="relu", init="he", norm="batch", clip=1.0,
                               schedule="onecycle", dropout=0.1)),
    ]
    # Five seeds a rung, not one. The slide tells students this table is the
    # deliverable and asks them to act on 2.8- and 4.2-point steps; a single
    # seed cannot support a claim that size, and the course says so itself two
    # lectures earlier (Lecture 15 reports sd 2.34 points for a run of the same
    # shape) and again in Lecture 21, where it refuses to rank a 0.94-point gap
    # measured one seed each. One pass is about 200 s, so this costs 17 minutes
    # and buys the difference between a result and an anecdote.
    LADDER_SEEDS = [SEED + k for k in range(5)]
    rows, prev = [], None
    for label, kw in ladder:
        runs = [_strip(train(d, seed=sd, **kw)) for sd in LADDER_SEEDS]
        accs = [r["test_acc"] for r in runs]
        mean = float(np.mean(accs))
        sd = float(np.std(accs, ddof=1))
        rows.append({"label": label, "test_acc": mean, "sd": sd,
                     "sd_pts": 100 * sd, "n_seeds": len(LADDER_SEEDS),
                     "accs": accs,
                     "final_val": float(np.mean([r["final_val"] for r in runs])),
                     "best_val": float(np.mean([r["best_val"] for r in runs])),
                     "last_loss": float(np.mean([r["last_loss"] for r in runs])),
                     "seconds": float(np.mean([r["seconds"] for r in runs])),
                     "delta": 0.0 if prev is None else mean - prev})
        prev = mean
        print(f"      {label:32s} test {mean:.4f} +/- {100 * sd:.2f} pts  "
              f"({np.mean([r['seconds'] for r in runs]):.0f} s a seed)")
    # the typical seed-to-seed spread, for judging every step in the table
    out["ladder_sd_pts"] = float(np.mean([r["sd_pts"] for r in rows]))
    out["ladder"] = rows

    print("    the same repairs, one at a time on the Lecture 13 network")
    alone = [
        ("Glorot initialisation", dict(act="sigmoid", init="glorot")),
        ("ReLU (with He)", dict(act="relu", init="he")),
        ("Batch normalisation", dict(act="sigmoid", init="torch",
                                     norm="batch")),
        ("Layer normalisation", dict(act="sigmoid", init="torch",
                                     norm="layer")),
        ("Gradient clipping", dict(act="sigmoid", init="torch", clip=1.0)),
        ("A 1-cycle schedule", dict(act="sigmoid", init="torch",
                                    schedule="onecycle")),
        ("Dropout 0.1", dict(act="sigmoid", init="torch", dropout=0.1)),
    ]
    solo = []
    for label, kw in alone:
        r = _strip(train(d, **kw))
        solo.append({"label": label, "test_acc": r["test_acc"],
                     "last_loss": r["last_loss"], "seconds": r["seconds"]})
        print(f"      {label:26s} alone: test {r['test_acc']:.4f}")
    out["alone"] = solo

    print("    layer norm against batch norm, on the repaired network")
    out["norms"] = {}
    for label, kw in [("none", {}), ("batch", dict(norm="batch")),
                      ("layer", dict(norm="layer"))]:
        r = _strip(train(d, act="relu", init="he", **kw))
        out["norms"][label] = {"test_acc": r["test_acc"],
                               "seconds": r["seconds"],
                               "n_params": r["n_params"],
                               "last_loss": r["last_loss"]}
        print(f"      {label:6s} test {r['test_acc']:.4f}  "
              f"{r['seconds']:.0f} s  {r['n_params']:,} params")

    print("    optimisers, on the repaired network")
    out["optims"] = {}
    for name, lr in [("sgd", 1e-2), ("momentum", 1e-2), ("nesterov", 1e-2),
                     ("rmsprop", 1e-3), ("adam", 1e-3), ("adamw", 1e-3)]:
        r = _strip(train(d, act="relu", init="he", norm="batch",
                         opt=name, lr=lr))
        out["optims"][name] = {"test_acc": r["test_acc"], "lr": lr,
                               "loss": r["loss"], "seconds": r["seconds"]}
        print(f"      {name:9s} lr {lr:g}  test {r['test_acc']:.4f}")

    print("    schedules, on the repaired network")
    out["schedules"] = {}
    for name in (None, "exp", "cosine", "onecycle"):
        r = _strip(train(d, act="relu", init="he", norm="batch",
                         clip=1.0, schedule=name))
        out["schedules"][str(name)] = {"test_acc": r["test_acc"],
                                       "lr": r["lr"], "loss": r["loss"],
                                       "val_acc": r["val_acc"]}
        print(f"      {str(name):9s} test {r['test_acc']:.4f}")

    print("    activation functions, on the repaired network")
    out["activations"] = {}
    for name in ("sigmoid", "tanh", "relu", "leaky", "elu", "selu"):
        init = "he" if name in ("relu", "leaky", "elu") else "glorot"
        r = _strip(train(d, act=name, init=init, norm="batch"))
        out["activations"][name] = {"test_acc": r["test_acc"], "init": init,
                                    "seconds": r["seconds"]}
        print(f"      {name:8s} ({init}) test {r['test_acc']:.4f}")

    print("    the gradient-norm distribution, for clipping")
    out["norm_dist_deep"] = grad_norm_distribution(d, act="relu", init="he")
    out["norm_dist_bad"] = grad_norm_distribution(d, act="relu", init="normal1")

    print("    the repaired network, tracked")
    out["fixed"] = _strip(train(d, act="relu", init="he", norm="batch",
                                clip=1.0, schedule="onecycle", dropout=0.1,
                                track_grads=True))
    out["grads_fixed"] = grad_profile(d["X_fit"], d["y_fit"], act="relu",
                                      init="he", norm="batch").tolist()
    return out


def var_check(n_rows: int = 20_000) -> dict:
    """The thread's central identity, checked on one random layer.

    The deck prints this block verbatim, so it is produced here rather than
    typed from memory. Cheap enough not to be cached.
    """
    torch.manual_seed(SEED)
    out = {"var_w": [], "predicted": [], "measured": [], "n_rows": n_rows,
           "fan_in": WIDTH}
    for var_w in (0.001, 0.01, 0.02, 0.05):
        W = torch.randn(WIDTH, WIDTH, dtype=torch.float64) * math.sqrt(var_w)
        a = torch.randn(n_rows, WIDTH, dtype=torch.float64)   # Var(a) = 1
        z = a @ W.T
        out["var_w"].append(var_w)
        out["predicted"].append(float(WIDTH * var_w))
        out["measured"].append(float(z.var()))
    return out


def _var_sweep(d) -> dict:
    """Activation sd against layer index, for a range of weight standard
    deviations, with the theory's prediction beside it.

    This is the thread's central claim made falsifiable: the forward variance
    is multiplied by fan_in * Var(w) at every layer, so it is geometric in the
    depth, and the exponent is under our control.
    """
    out = {}
    X = torch.as_tensor(d["X_fit"][:BATCH], dtype=torch.float64)
    for label, sd in [("0.05", 0.05), ("0.10", 0.10),
                      ("0.1414", math.sqrt(2 / WIDTH)), ("0.20", 0.20)]:
        torch.manual_seed(SEED)
        net = make_net(act="relu", init="torch", dtype=torch.float64)
        for m in net:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0.0, sd)
                nn.init.zeros_(m.bias)
        h, sds = X, []
        with torch.no_grad():
            for m in net:
                h = m(h)
                if isinstance(m, nn.ReLU):
                    sds.append(float(h.std()))
        # theory: Var multiplied by fan_in * Var(w) / 2 per ReLU layer
        r = WIDTH * sd ** 2 / 2
        out[label] = {"sd": sds, "ratio": r,
                      "predicted": [sds[0] * r ** (0.5 * k)
                                    for k in range(len(sds))]}
    return out


def _strip(r: dict) -> dict:
    return r


# ------------------------------------------------------------- the figures

def fig_grid(d):
    """Forty images, four per class, with the class name over each column."""
    imgs, labs = d["grid_images"], d["grid_labels"]
    chosen = np.concatenate([np.where(labs == c)[0][:4] for c in range(10)])
    fig, axes = plt.subplots(4, 10, figsize=(11.0, 5.0))
    for k, ax in enumerate(axes.T.ravel()):
        ax.imshow(imgs[chosen[k]])
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
    for c in range(10):
        axes[0, c].set_title(CLASSES[c], fontsize=SMALL, color=MUTED, pad=6)
    fig.suptitle("Four examples of each of the ten classes, 32 × 32, colour",
                 fontsize=SMALL, color=MUTED, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, "l13-grid", raster=True)


def fig_class_balance(d):
    counts = d["train_counts"]
    fig, ax = plt.subplots(figsize=(11.0, 3.3))
    ax.bar(range(10), counts, color=PRIMARY, width=0.68)
    ax.axhline(counts.mean(), color=ACCENT, lw=2, ls="--")
    ax.annotate(f"every class: {counts[0]:,} images\n"
                f"so the majority-class baseline is exactly 10%",
                xy=(4.5, counts.mean()), xytext=(0.6, counts.mean() * 0.42),
                color=ACCENT,
                bbox=dict(boxstyle="round,pad=0.5", fc="white", ec=ACCENT,
                          lw=1.2),
                arrowprops=dict(arrowstyle="->", color=ACCENT, lw=1.8))
    ax.set_xticks(range(10))
    ax.set_xticklabels(CLASSES, rotation=30, ha="right")
    ax.set_ylabel("training images")
    ax.set_ylim(0, counts.max() * 1.25)
    ax.set_title("CIFAR-10 is exactly balanced — by construction, not by luck")
    fig.tight_layout()
    save(fig, "l13-class-balance")


def fig_loss_flat(l13):
    deep, shallow = l13["deep"], l13["shallow"]
    fig, ax = plt.subplots(figsize=(11.0, 3.4))
    ep = range(1, len(deep["loss"]) + 1)
    ax.plot(ep, deep["loss"], color=ACCENT, lw=2.5, marker="o", ms=4,
            label=f"{DEPTH} hidden layers")
    ax.plot(ep, shallow["loss"], color=PRIMARY, lw=2.5, marker="s", ms=4,
            label="2 hidden layers")
    chance = math.log(10)
    ax.axhline(chance, color=MUTED, ls=":", lw=1.6)
    ax.annotate(f"ln 10 = {chance:.4f}\nthe loss of a model that has learned "
                f"nothing",
                xy=(len(deep["loss"]) * 0.62, chance),
                xytext=(len(deep["loss"]) * 0.30, chance - 0.55),
                color=MUTED,
                bbox=dict(boxstyle="round,pad=0.5", fc="white", ec=RULE),
                arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.6))
    ax.set_xlabel("epoch"); ax.set_ylabel("training loss")
    ax.set_title("Same data, same optimiser, same twenty epochs")
    ax.legend(loc="upper right")
    fig.tight_layout()
    save(fig, "l13-loss-flat")


def fig_grad_by_depth(l13):
    """The figure the whole application turns on."""
    g = np.array(l13["grad64"])
    fig, ax = plt.subplots(figsize=(11.0, 3.5))
    x = np.arange(1, len(g) + 1)
    ax.semilogy(x, g, color=ACCENT, lw=2.5, marker="o", ms=5)
    ax.set_yscale("log")
    plain_log(ax, "y", decades(g, 3), fmt=POW10)
    ax.set_xticks([1, 5, 10, 15, 20, 21])
    ax.set_xticklabels(["1", "5", "10", "15", "20", "out"])
    ax.set_xlabel("layer  (1 = nearest the input)")
    ax.set_ylabel("‖∂L/∂W‖")
    ratio = g[19] / g[0]
    ax.annotate(f"layer 20 : layer 1  =  {ratio:.1e}",
                xy=(1.4, g[0] * 3), xytext=(4.0, g[0] * 3),
                color=ACCENT,
                bbox=dict(boxstyle="round,pad=0.5", fc="white", ec=ACCENT,
                          lw=1.2),
                arrowprops=dict(arrowstyle="->", color=ACCENT, lw=1.8))
    ax.set_title("Gradient norm per weight matrix, at initialisation "
                 "(log scale)")
    fig.tight_layout()
    save(fig, "l13-grad-by-depth")


def fig_act_by_depth(l13):
    a = l13["acts"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 3.4))
    x = np.arange(1, len(a["sd"]) + 1)
    ax1.plot(x, a["mean"], color=PRIMARY, lw=2.5, marker="o", ms=4,
             label="mean")
    ax1.plot(x, a["sd"], color=ACCENT, lw=2.5, marker="s", ms=4,
             label="standard deviation")
    ax1.set_xlabel("hidden layer"); ax1.set_ylabel("activation")
    ax1.set_ylim(0, 0.62)
    ax1.legend(loc="center right")
    ax1.set_title("Every layer says almost the same thing")
    ax2.plot(x, np.array(a["sd"]) / a["sd"][0], color=MATH, lw=2.5,
             marker="o", ms=4)
    ax2.axhline(1.0, color=RULE, lw=1.2)
    ax2.set_xlabel("hidden layer")
    ax2.set_ylabel("sd, relative to layer 1")
    ax2.set_ylim(0, 1.15)
    ax2.set_title("The forward signal does not vanish")
    fig.tight_layout()
    save(fig, "l13-act-by-depth")


def fig_act_hist(hists):
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.2), sharey=True)
    for ax, (k, v) in zip(axes, sorted(hists.items())):
        ax.hist(v, bins=60, range=(0, 1), color=PRIMARY)
        ax.set_title(f"hidden layer {k}")
        ax.set_xlabel("activation")
        ax.set_xlim(0, 1)
    axes[0].set_ylabel("count")
    fig.suptitle("The logistic output of three layers, over 512 images",
                 fontsize=SMALL, color=MUTED, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, "l13-act-hist")


def fig_depth_sweep(l13):
    sw = l13["depth_sweep"]
    ks = sorted(int(k) for k in sw)
    acc = [100 * sw[str(k)]["test_acc"] for k in ks]
    fig, ax = plt.subplots(figsize=(11.0, 3.3))
    bars = ax.bar([str(k) for k in ks], acc, color=PRIMARY, width=0.55)
    bars[-1].set_color(ACCENT)
    ax.axhline(10, color=MUTED, ls="--", lw=1.8)
    ax.annotate("the 10% baseline", xy=(0.15, 10), xytext=(0.15, 15.5),
                color=MUTED,
                bbox=dict(boxstyle="round,pad=0.5", fc="white", ec=RULE))
    for b, v in zip(bars, acc):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.8, f"{v:.1f}%",
                ha="center", color=MUTED, fontsize=TICK)
    ax.set_xlabel("hidden layers")
    ax.set_ylabel("test accuracy, %")
    ax.set_ylim(0, max(acc) * 1.28)
    ax.set_title("Everything else identical; only the depth changes")
    fig.tight_layout()
    save(fig, "l13-depth-sweep")


def fig_init_grads(l14):
    g = l14["grads"]
    fig, ax = plt.subplots(figsize=(11.0, 3.6))
    style = [("normal1_sigmoid", "N(0, 1), sigmoid", MUTED, "-."),
             ("default_sigmoid", "PyTorch default, sigmoid", ACCENT, "-"),
             ("glorot_sigmoid", "Glorot, sigmoid", MATH, "--"),
             ("he_relu", "He, ReLU", SUCCESS, "-")]
    x = np.arange(1, len(g["he_relu"]) + 1)
    for key, label, colour, ls in style:
        ax.semilogy(x, g[key], color=colour, lw=2.4, ls=ls, marker="o", ms=3.5,
                    label=label)
    plain_log(ax, "y", decades([v for vs in g.values() for v in vs], 4),
              fmt=POW10)
    ax.set_xticks([1, 5, 10, 15, 20, 21])
    ax.set_xticklabels(["1", "5", "10", "15", "20", "out"])
    ax.set_xlabel("layer  (1 = nearest the input)")
    ax.set_ylabel("‖∂L/∂W‖")
    ax.legend(loc="lower right", ncol=2)
    ax.set_title("The same twenty layers, four initialisations (log scale)")
    fig.tight_layout()
    save(fig, "l14-init-grads")


def fig_var_vs_depth(l14):
    sw = l14["var_sweep"]
    fig, ax = plt.subplots(figsize=(11.0, 3.5))
    colours = {"0.05": MUTED, "0.10": PRIMARY, "0.1414": SUCCESS,
               "0.20": ACCENT}
    for label in ("0.05", "0.10", "0.1414", "0.20"):
        v = sw[label]
        x = np.arange(1, len(v["sd"]) + 1)
        name = ("√(2/100) = 0.1414" if label == "0.1414"
                else f"sd(w) = {label}")
        ax.semilogy(x, v["sd"], color=colours[label], lw=2.4, marker="o", ms=4,
                    label=f"{name}   (r = {v['ratio']:.2f})")
        ax.semilogy(x, v["predicted"], color=colours[label], lw=1.2, ls=":")
    plain_log(ax, "y",
              decades([v for s_ in sw.values() for v in s_["sd"]], 3),
              fmt=POW10)
    ax.set_xlabel("hidden layer")
    ax.set_ylabel("activation standard deviation")
    ax.legend(loc="lower left", fontsize=TICK)
    ax.set_title("Measured (solid) against  sd₁ · r^(l/2)  (dotted), "
                 "ReLU, 100 units")
    fig.tight_layout()
    save(fig, "l14-var-vs-depth")


def fig_ladder(l14):
    rows = l14["ladder"]
    labels = [r["label"] for r in rows]
    acc = [100 * r["test_acc"] for r in rows]
    fig, ax = plt.subplots(figsize=(11.0, 4.3))
    y = np.arange(len(rows))[::-1]
    # Green is the BEST row, not the last one. On this ladder they are not the
    # same row, and colouring the bottom bar green would assert the thing the
    # table exists to disprove.
    best = int(np.argmax(acc))
    colours = [SUCCESS if i == best else
               ACCENT if r["delta"] < 0 or i == 0 else PRIMARY
               for i, r in enumerate(rows)]
    ax.barh(y, acc, color=colours, height=0.62)
    ax.axvline(10, color=MUTED, ls="--", lw=1.8)
    for yy, v, r in zip(y, acc, rows):
        note = f"{v:.1f}%"
        if r["delta"]:
            note += f"   ({100 * r['delta']:+.1f})"
        ax.text(v + 0.7, yy, note, va="center", color=MUTED, fontsize=TICK)
    ax.set_yticks(y); ax.set_yticklabels(labels)
    ax.set_xlabel("test accuracy, %")
    ax.set_xlim(0, max(acc) * 1.3)
    ax.grid(axis="y", visible=False)
    ax.set_title("Each row adds one change to the row above it")
    fig.tight_layout()
    save(fig, "l14-ladder")


def fig_curves(l14, l13):
    fig, ax = plt.subplots(figsize=(11.0, 3.4))
    ep = range(1, len(l13["deep"]["loss"]) + 1)
    ax.plot(ep, l13["deep"]["loss"], color=ACCENT, lw=2.5, marker="o", ms=4,
            label="Lecture 13's network")
    ax.plot(ep, l14["fixed"]["loss"], color=SUCCESS, lw=2.5, marker="s", ms=4,
            label="after every repair")
    ax.axhline(math.log(10), color=MUTED, ls=":", lw=1.6)
    ax.set_xlabel("epoch"); ax.set_ylabel("training loss")
    ax.legend()
    ax.set_title("The same twenty layers, before and after")
    fig.tight_layout()
    save(fig, "l14-curves")


def fig_clip(l14):
    bad = np.asarray(l14["norm_dist_bad"])
    good = np.asarray(l14["norm_dist_deep"])
    # Log-spaced bins: the two distributions span different orders of
    # magnitude, and linear bins on a log axis pile everything into one column.
    lo = min(bad.min(), good.min()) * 0.8
    hi = max(bad.max(), good.max()) * 1.2
    bins = np.logspace(np.log10(lo), np.log10(hi), 70)

    fig, ax = plt.subplots(figsize=(11.0, 3.2))
    ax.hist(bad, bins=bins, color=ACCENT, alpha=0.85,
            label="N(0, 1) initialisation")
    ax.hist(good, bins=bins, color=SUCCESS, alpha=0.85,
            label="He initialisation")
    ax.set_xscale("log")
    plain_log(ax, "x", decades(np.concatenate([bad, good]), 4), fmt=POW10)
    ax.axvline(1.0, color=PRIMARY, lw=2.2, ls="--")
    top = ax.get_ylim()[1]
    ax.annotate("clip at 1.0", xy=(1.0, top * 0.55),
                xytext=(8.0, top * 0.72), color=PRIMARY,
                bbox=dict(boxstyle="round,pad=0.5", fc="white", ec=PRIMARY),
                arrowprops=dict(arrowstyle="->", color=PRIMARY, lw=1.6))
    ax.set_xlabel("total gradient norm, one optimiser step")
    ax.set_ylabel("steps")
    ax.legend()
    ax.set_title("Three epochs of steps, ReLU, twenty layers")
    fig.tight_layout()
    save(fig, "l14-clip")


def fig_schedules(l14):
    sch = l14["schedules"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 3.3))
    names = [("None", "constant", MUTED, "-"), ("exp", "exponential", PRIMARY, "--"),
             ("cosine", "cosine", MATH, "-."), ("onecycle", "1-cycle", SUCCESS, "-")]
    for key, label, colour, ls in names:
        v = sch[key]
        ep = range(1, len(v["lr"]) + 1)
        ax1.plot(ep, v["lr"], color=colour, lw=2.3, ls=ls, label=label)
        ax2.plot(ep, [100 * a for a in v["val_acc"]], color=colour, lw=2.3,
                 ls=ls, label=label)
    ax1.set_xlabel("epoch"); ax1.set_ylabel("learning rate")
    ax1.set_title("What the schedule does")
    ax1.legend(fontsize=TICK)
    ax2.set_xlabel("epoch"); ax2.set_ylabel("validation accuracy, %")
    ax2.set_title("What it buys")
    fig.tight_layout()
    save(fig, "l14-schedules")


def fig_optims(l14):
    op = l14["optims"]
    fig, ax = plt.subplots(figsize=(11.0, 3.4))
    style = [("sgd", "SGD", MUTED, ":"), ("momentum", "momentum", PRIMARY, "--"),
             ("nesterov", "Nesterov", MATH, "-."),
             ("rmsprop", "RMSProp", ACCENT, "--"),
             ("adam", "Adam", SUCCESS, "-"), ("adamw", "AdamW", "#7a5195", "-")]
    for key, label, colour, ls in style:
        v = op[key]
        ax.plot(range(1, len(v["loss"]) + 1), v["loss"], color=colour, lw=2.2,
                ls=ls, label=f"{label}  ({100 * v['test_acc']:.1f}%)")
    ax.set_xlabel("epoch"); ax.set_ylabel("training loss")
    ax.legend(ncol=2, fontsize=TICK)
    ax.set_title("Six optimisers on the repaired network; test accuracy in "
                 "the legend")
    fig.tight_layout()
    save(fig, "l14-optims")


def fig_activations(l14):
    """The activation functions themselves, and their derivatives."""
    z = np.linspace(-5, 5, 400)
    fns = {
        "logistic": (1 / (1 + np.exp(-z)), ACCENT),
        "tanh": (np.tanh(z), MATH),
        "ReLU": (np.maximum(0, z), SUCCESS),
        "ELU": (np.where(z > 0, z, np.exp(np.minimum(z, 0)) - 1), PRIMARY),
    }
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 3.3))
    for name, (v, c) in fns.items():
        ax1.plot(z, v, color=c, lw=2.3, label=name)
        ax2.plot(z[:-1], np.diff(v) / np.diff(z), color=c, lw=2.3, label=name)
    ax1.set_title("The activation"); ax2.set_title("Its derivative")
    ax1.set_xlabel("z"); ax2.set_xlabel("z")
    ax2.axhline(0.25, color=ACCENT, ls=":", lw=1.6)
    ax2.annotate("the logistic never exceeds 1/4", xy=(-4.6, 0.25),
                 xytext=(-4.6, 0.55), color=ACCENT, fontsize=TICK,
                 bbox=dict(boxstyle="round,pad=0.5", fc="white", ec=ACCENT))
    ax1.legend(fontsize=TICK, loc="upper left")
    ax2.set_ylim(-0.1, 1.35)
    fig.tight_layout()
    save(fig, "l14-activations")


def fig_alone(l14):
    rows = l14["alone"]
    labels = [r["label"] for r in rows]
    acc = [100 * r["test_acc"] for r in rows]
    fig, ax = plt.subplots(figsize=(11.0, 3.8))
    y = np.arange(len(rows))[::-1]
    colours = [SUCCESS if v > 15 else ACCENT for v in acc]
    ax.barh(y, acc, color=colours, height=0.6)
    ax.axvline(10, color=MUTED, ls="--", lw=1.8)
    for yy, v in zip(y, acc):
        ax.text(v + 0.5, yy, f"{v:.1f}%", va="center", color=MUTED,
                fontsize=TICK)
    ax.set_yticks(y); ax.set_yticklabels(labels)
    ax.set_xlabel("test accuracy, %")
    ax.set_xlim(0, max(acc) * 1.25)
    ax.grid(axis="y", visible=False)
    ax.set_title("Each repair applied alone to the Lecture 13 network")
    fig.tight_layout()
    save(fig, "l14-alone")


def fig_grad_fixed(l14, l13):
    fig, ax = plt.subplots(figsize=(11.0, 3.4))
    x = np.arange(1, len(l13["grad64"]) + 1)
    ax.semilogy(x, l13["grad64"], color=ACCENT, lw=2.5, marker="o", ms=4,
                label="Lecture 13's network")
    ax.semilogy(x, l14["grads_fixed"], color=SUCCESS, lw=2.5, marker="s", ms=4,
                label="He, ReLU, batch normalisation")
    plain_log(ax, "y", decades(list(l13["grad64"]) + list(l14["grads_fixed"]), 4),
              fmt=POW10)
    ax.set_xticks([1, 5, 10, 15, 20, 21])
    ax.set_xticklabels(["1", "5", "10", "15", "20", "out"])
    ax.set_xlabel("layer  (1 = nearest the input)")
    ax.set_ylabel("‖∂L/∂W‖")
    ax.legend(loc="lower right")
    ax.set_title("The figure Lecture 13 ended on, repaired (log scale)")
    fig.tight_layout()
    save(fig, "l14-grad-fixed")


# ------------------------------------------------------------- the diagrams

def validate_diagrams() -> list[str]:
    bad = []
    for name in ("d-deepstack.svg", "d-varprop.svg"):
        path = OUT / name
        if not path.is_file():
            bad.append(f"{name}: missing")
            continue
        try:
            xml.dom.minidom.parse(str(path))
        except Exception as exc:                          # noqa: BLE001
            bad.append(f"{name}: {type(exc).__name__}: {exc}")
    return bad


# ------------------------------------------------------------------- driver

def main() -> int:
    setup()
    load_cache()

    print("Loading CIFAR-10…")
    d = load_cifar()
    print(f"  fit {len(d['X_fit']):,}  val {len(d['X_val']):,}  "
          f"test {len(d['X_test']):,}   device {DEVICE}")

    facts: dict = {}

    # ---- the brief, the data, the baseline --------------------------------
    ref = make_net()
    facts["l13_depth"] = DEPTH
    facts["l13_width"] = WIDTH
    facts["l13_n_features"] = N_IN
    facts["l13_n_classes"] = N_OUT
    facts["l13_n_train_full"] = 50_000
    facts["l13_n_total"] = 60_000
    facts["l13_n_hidden_minus1"] = DEPTH - 1
    facts["l13_n_test"] = int(len(d["X_test"]))
    facts["l13_n_fit"] = int(len(d["X_fit"]))
    facts["l13_n_val"] = int(len(d["X_val"]))
    facts["l13_n_params"] = int(n_params(ref))
    facts["l13_n_weight_matrices"] = len(linear_layers(ref))
    facts["l13_per_class_train"] = int(d["train_counts"][0])
    facts["l13_per_class_test"] = int(d["test_counts"][0])
    facts["l13_baseline_acc"] = 1.0 / N_OUT
    facts["l13_chance_loss"] = math.log(N_OUT)
    facts["l13_epochs"] = EPOCHS
    facts["l13_batch"] = BATCH
    facts["l13_lr"] = LR
    facts["l13_device"] = DEVICE
    facts["l13_first_layer_params"] = N_IN * WIDTH + WIDTH
    facts["l13_hidden_layer_params"] = WIDTH * WIDTH + WIDTH
    facts["l13_head_params"] = WIDTH * N_OUT + N_OUT
    facts["l13_default_bound"] = 1.0 / math.sqrt(WIDTH)
    facts["l13_default_bound_in"] = 1.0 / math.sqrt(N_IN)
    facts["l13_steps_per_epoch"] = math.ceil(N_FIT / BATCH)
    facts["l13_steps"] = math.ceil(N_FIT / BATCH) * EPOCHS

    # What nn.Linear actually put in a hidden weight matrix, measured rather
    # than quoted from the documentation. The slide prints all four numbers.
    torch.manual_seed(SEED)
    _w = linear_layers(make_net())[1].weight.detach()
    facts["l13_init"] = {
        "min": float(_w.min()), "max": float(_w.max()),
        "mean": float(_w.mean()), "sd": float(_w.std()),
        "sd_predicted": (1.0 / math.sqrt(WIDTH)) / math.sqrt(3.0),
        "var": float(_w.var()),
    }

    print("Lecture 13 — the build:")
    l13 = cached("app07_l13", lambda: run_l13(d))
    hists = cached("app07_hists",
                   lambda: layer_hist(d["X_fit"], act="sigmoid", init="torch"))

    g = np.array(l13["grad64"])
    g32 = np.array(l13["grad32"])
    ratios = g[1:DEPTH] / g[0:DEPTH - 1]
    geo = float(np.exp(np.mean(np.log(ratios))))
    facts["l13_grad_norms"] = [float(v) for v in g]
    facts["l13_grad_norms_f32"] = [float(v) for v in g32]
    facts["l13_grad_dtype_max_rel_diff"] = float(
        np.max(np.abs(g32 - g) / g))
    facts["l13_grad_l1"] = float(g[0])
    facts["l13_grad_l10"] = float(g[9])
    facts["l13_grad_l20"] = float(g[19])
    facts["l13_grad_head"] = float(g[20])
    facts["l13_attenuation"] = float(g[19] / g[0])
    facts["l13_attenuation_log10"] = float(np.log10(g[19] / g[0]))
    facts["l13_per_layer_gain"] = geo
    facts["l13_per_layer_attenuation"] = 1.0 / geo
    facts["l13_n_ratios"] = int(len(ratios))
    facts["l13_gain_check"] = float(geo ** len(ratios))
    # The same thing read downwards, which is how the slide states it:
    # rho^19, where rho is the per-layer attenuation.
    facts["l13_attenuation_down"] = float(g[0] / g[19])
    facts["l13_per_layer_check"] = float((1.0 / geo) ** len(ratios))
    facts["l13_f32_zero_layers"] = int((g32 == 0).sum())
    print(f"    gradient at layer 1  {g[0]:.3e}")
    print(f"    gradient at layer 20 {g[19]:.3e}")
    print(f"    attenuation {g[19] / g[0]:.4e}  "
          f"= {1 / geo:.4f} per layer, {len(ratios)} times over")

    facts["l13_acts"] = l13["acts"]
    facts["l13_act_sd_l1"] = float(l13["acts"]["sd"][0])
    facts["l13_act_sd_l20"] = float(l13["acts"]["sd"][19])
    facts["l13_act_mean_l20"] = float(l13["acts"]["mean"][19])
    facts["l13_act_sat_l20"] = float(l13["acts"]["saturated"][19])

    for name in ("deep", "shallow"):
        r = l13[name]
        facts[f"l13_{name}"] = {
            "first_loss": r["first_loss"], "last_loss": r["last_loss"],
            "test_acc": r["test_acc"], "final_val": r["final_val"],
            "best_val": r["best_val"], "seconds": r["seconds"],
            "n_params": r["n_params"], "loss": r["loss"],
            "val_acc": r["val_acc"],
        }
    facts["l13_loss_drop"] = float(l13["deep"]["first_loss"]
                                   - l13["deep"]["last_loss"])
    facts["l13_depth_sweep"] = l13["depth_sweep"]
    facts["l13_deep_grad_ep1_l1"] = float(l13["deep"]["grad"][0][0])
    facts["l13_deep_grad_last_l1"] = float(l13["deep"]["grad"][-1][0])
    facts["l13_deep_grad_ep1_head"] = float(l13["deep"]["grad"][0][-1])
    facts["l13_deep_grad_last_head"] = float(l13["deep"]["grad"][-1][-1])

    extra = cached("app07_l13_extra", lambda: run_l13_extra(d))
    facts["l13_wchange"] = extra["wchange"]
    facts["l13_assistant"] = extra["assistant"]
    facts["l13_memorise"] = extra["memorise"]
    print(f"    weights moved: layer 1 {extra['wchange'][0]:.4f}   "
          f"head {extra['wchange'][-1]:.4f}")
    print(f"    deep: loss {l13['deep']['first_loss']:.4f} -> "
          f"{l13['deep']['last_loss']:.4f}, test "
          f"{l13['deep']['test_acc']:.4f}")
    print(f"    shallow: test {l13['shallow']['test_acc']:.4f}")

    print("Lecture 14 — the repair:")
    l14 = cached("app07_l14", lambda: run_l14(d))

    facts["l14_grads"] = {k: [float(x) for x in v]
                          for k, v in l14["grads"].items()}
    facts["l14_grads_fixed"] = [float(x) for x in l14["grads_fixed"]]
    per_layer = {}
    for k, v in l14["grads"].items():
        a = np.array(v)
        r = a[1:DEPTH] / a[0:DEPTH - 1]
        per_layer[k] = {
            "gain": float(np.exp(np.mean(np.log(r)))),
            "attenuation": float(a[19] / a[0]),
        }
        per_layer[k]["per_layer"] = 1.0 / per_layer[k]["gain"]
    facts["l14_per_layer"] = per_layer
    for k, v in per_layer.items():
        print(f"    {k:16s} per-layer factor {v['per_layer']:.4f}   "
              f"end to end {v['attenuation']:.3e}")

    # what the theory says the per-layer backward factor should be
    facts["l14_theory"] = {
        "default_sigmoid": math.sqrt(WIDTH * (1 / (3 * WIDTH)) * 0.0625),
        "glorot_sigmoid": math.sqrt(WIDTH * (2 / (2 * WIDTH)) * 0.0625),
        "glorot_relu": math.sqrt(WIDTH * (2 / (2 * WIDTH)) * 0.5),
        "he_relu": math.sqrt(WIDTH * (2 / WIDTH) * 0.5),
    }

    print("    the backward factor rho, measured directly")
    deltas = cached("app07_deltas", lambda: run_deltas(d))
    facts["l14_delta"] = {
        k: {"rho": v["rho"], "forward_ratio": v["forward_ratio"],
            "delta_attenuation": v["delta_attenuation"]}
        for k, v in deltas.items()}
    facts["l14_theory_fwd"] = {
        "glorot_relu": math.sqrt(WIDTH * (2 / (2 * WIDTH)) / 2),
        "he_relu": math.sqrt(WIDTH * (2 / WIDTH) / 2),
    }
    facts["l14_glorot_relu_fwd_over19"] = (
        facts["l14_theory_fwd"]["glorot_relu"] ** (DEPTH - 1))

    # How far the prediction is from the measurement, as a relative error, and
    # what the theory says the weight-gradient ratio should therefore be.
    facts["l14_err"] = {}
    facts["l14_wratio_pred"] = {}
    for k, th in facts["l14_theory"].items():
        m = deltas[k]["rho"]
        facts["l14_err"][k] = abs(m - th) / th
        facts["l14_wratio_pred"][k] = m / deltas[k]["forward_ratio"]
    facts["l14_err_max"] = max(facts["l14_err"].values())
    print("    prediction error: "
          + ", ".join(f"{k} {100 * v:.1f}%"
                      for k, v in facts["l14_err"].items()))

    xtra = cached("app07_l14_extra", lambda: run_l14_extra(d))
    gx = np.array(xtra["grad_glorot_relu"])
    rx = gx[1:DEPTH] / gx[0:DEPTH - 1]
    facts["l14_glorot_relu"] = {
        "gain": float(np.exp(np.mean(np.log(rx)))),
        "per_layer": float(1 / np.exp(np.mean(np.log(rx)))),
        "attenuation": float(gx[19] / gx[0]),
        "test_acc": xtra["glorot_relu"]["test_acc"],
        "last_loss": xtra["glorot_relu"]["last_loss"],
    }
    facts["l14_he_relu_test"] = xtra["he_relu"]["test_acc"]
    facts["l14_xavier_cost"] = (xtra["he_relu"]["test_acc"]
                                - xtra["glorot_relu"]["test_acc"])
    facts["l14_grad_glorot_relu"] = [float(v) for v in gx]
    print(f"    Xavier+ReLU per-layer {facts['l14_glorot_relu']['per_layer']:.4f} "
          f"(theory {facts['l14_theory']['glorot_relu']:.4f}), costs "
          f"{100 * facts['l14_xavier_cost']:+.2f} points")
    facts["l14_sigmoid_dmax"] = 0.25
    facts["l14_sigmoid_dmax_sq"] = 0.0625
    facts["l14_var_check"] = var_check()
    print("    variance identity: predicted "
          f"{facts['l14_var_check']['predicted']} against measured "
          f"{[round(v, 4) for v in facts['l14_var_check']['measured']]}")
    facts["l14_glorot_var"] = 2.0 / (WIDTH + WIDTH)
    facts["l14_he_var"] = 2.0 / WIDTH
    facts["l14_he_sd"] = math.sqrt(2.0 / WIDTH)
    facts["l14_glorot_sd"] = math.sqrt(2.0 / (WIDTH + WIDTH))
    facts["l14_default_var"] = 1.0 / (3.0 * WIDTH)
    facts["l14_default_sd"] = math.sqrt(1.0 / (3.0 * WIDTH))

    facts["l14_var_sweep"] = {
        k: {"sd": [float(x) for x in v["sd"]], "ratio": float(v["ratio"])}
        for k, v in l14["var_sweep"].items()}

    facts["l14_ladder"] = l14["ladder"]
    facts["l14_ladder_sd_pts"] = l14["ladder_sd_pts"]
    facts["l14_ladder_seeds"] = l14["ladder"][0]["n_seeds"]
    # which steps clear the seed spread, and which are noise wearing a sign
    facts["l14_steps_real"] = sum(
        1 for r in l14["ladder"][1:]
        if abs(100 * r["delta"]) > 2 * l14["ladder_sd_pts"])
    facts["l14_steps_noise"] = sum(
        1 for r in l14["ladder"][1:]
        if abs(100 * r["delta"]) <= 2 * l14["ladder_sd_pts"])
    facts["l14_alone"] = l14["alone"]
    facts["l14_norms"] = l14["norms"]
    facts["l14_optims"] = {k: {"test_acc": v["test_acc"], "lr": v["lr"],
                               "seconds": v["seconds"],
                               "last_loss": v["loss"][-1]}
                           for k, v in l14["optims"].items()}
    facts["l14_schedules"] = {k: {"test_acc": v["test_acc"]}
                              for k, v in l14["schedules"].items()}
    facts["l14_activations"] = l14["activations"]
    facts["l14_fixed"] = {
        "test_acc": l14["fixed"]["test_acc"],
        "final_val": l14["fixed"]["final_val"],
        "best_val": l14["fixed"]["best_val"],
        "last_loss": l14["fixed"]["last_loss"],
        "first_loss": l14["fixed"]["first_loss"],
        "seconds": l14["fixed"]["seconds"],
        "n_params": l14["fixed"]["n_params"],
        "loss": l14["fixed"]["loss"],
        "val_acc": l14["fixed"]["val_acc"],
    }
    facts["l14_total_gain"] = (l14["ladder"][-1]["test_acc"]
                               - l14["ladder"][0]["test_acc"])
    facts["l14_best_row"] = max(r["test_acc"] for r in l14["ladder"])
    facts["l14_biggest_step"] = max(r["delta"] for r in l14["ladder"])
    facts["l14_clip_median"] = float(np.median(l14["norm_dist_deep"]))
    facts["l14_clip_max"] = float(np.max(l14["norm_dist_deep"]))
    facts["l14_clip_bad_max"] = float(np.max(l14["norm_dist_bad"]))
    facts["l14_clip_bad_median"] = float(np.median(l14["norm_dist_bad"]))
    facts["l14_bn_params"] = (l14["norms"]["batch"]["n_params"]
                              - l14["norms"]["none"]["n_params"])
    facts["l14_bn_params_per_layer"] = 2 * WIDTH
    facts["l14_bn_cost"] = (l14["norms"]["batch"]["test_acc"]
                            - l14["norms"]["none"]["test_acc"])
    facts["l14_ln_gain"] = (l14["norms"]["layer"]["test_acc"]
                            - l14["norms"]["none"]["test_acc"])

    print("    the clock cost of normalisation, min of several repeats")
    timing = cached("app07_timing", lambda: run_timing(d))
    facts["l14_timing"] = timing
    facts["l14_bn_overhead"] = timing["batch_overhead"]
    facts["l14_ln_overhead"] = timing["layer_overhead"]

    # The best row of the ladder is not always the last one. Say which.
    best = max(l14["ladder"], key=lambda r: r["test_acc"])
    facts["l14_best_label"] = best["label"]
    facts["l14_best_gain"] = best["test_acc"] - l14["ladder"][0]["test_acc"]
    facts["l14_last_row_gain"] = (l14["ladder"][-1]["test_acc"]
                                  - best["test_acc"])
    facts["l14_worst_step"] = min(r["delta"] for r in l14["ladder"])
    facts["l14_act_spread"] = (max(v["test_acc"] for v in
                                   l14["activations"].values())
                               - min(v["test_acc"] for v in
                                     l14["activations"].values()))
    print(f"    best rung: {best['label']} at {best['test_acc']:.4f} "
          f"(+{100 * facts['l14_best_gain']:.1f} points); the last rung is "
          f"{100 * facts['l14_last_row_gain']:+.1f}")

    print("Figures:")
    fig_grid(d)
    fig_class_balance(d)
    fig_loss_flat(l13)
    fig_grad_by_depth(l13)
    fig_act_by_depth(l13)
    fig_act_hist(hists)
    fig_depth_sweep(l13)
    fig_init_grads(l14)
    fig_var_vs_depth(l14)
    fig_ladder(l14)
    fig_curves(l14, l13)
    fig_clip(l14)
    fig_schedules(l14)
    fig_optims(l14)
    fig_activations(l14)
    fig_alone(l14)
    fig_grad_fixed(l14, l13)

    export(**facts)

    bad = validate_diagrams()
    if bad:
        print("\nmalformed diagrams:")
        for b in bad:
            print("  " + b)
        return 1

    problems = check_text_floor()
    if problems:
        print("\ntext floor:")
        for p in problems:
            print("  " + p)
        return 1
    print("\nall figures clear the 15px text floor; all d-*.svg parse as XML")
    return 0


if __name__ == "__main__":
    sys.exit(main())
