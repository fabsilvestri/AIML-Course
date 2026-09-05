#!/usr/bin/env python3
"""
Application 6 — Lectures 11 and 12. Fashion MNIST, the first neural network,
and the rebuild in PyTorch.

    python3 tools/figures_app06.py

Everything printed on slides/lecture-11.html and slides/lecture-12.html comes
from here, via figkit.export() into assets/figures/figures.json. Expensive fits
are cached (figkit.cached) so a cosmetic re-run takes seconds; delete
/private/tmp/claude-501/aiml-data/fits-v2.pkl to refit from scratch.

Timings are wall-clock on the machine that generated the figures — an Apple
Silicon laptop with an MPS backend and no CUDA. They are measurements, not
guarantees, and the decks say so.
"""

from __future__ import annotations

import sys
import time
import warnings
import xml.dom.minidom
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))

from figkit import (setup, save, cached, load_cache, plain_log, export, OUT, SEED,   # noqa: E402
                    PRIMARY, ACCENT, SUCCESS, MATH, MUTED, RULE, AXIS,
                    BODY, SMALL, TICK, check_text_floor)

import torch                                                    # noqa: E402
import torch.nn as nn                                           # noqa: E402
import torchvision                                              # noqa: E402
from sklearn.neural_network import MLPClassifier                # noqa: E402
from sklearn.exceptions import ConvergenceWarning               # noqa: E402

DATA = Path("/private/tmp/claude-501/aiml-data/fashion")

CLASSES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
           "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

# The architecture the whole application is built around: 784 -> 300 -> 100 -> 10.
HIDDEN = (300, 100)
N_VAL = 5_000                 # carved out of the 60,000 training images
EPOCHS = 20                   # the benchmark both frameworks run
BATCH = 128

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


# ------------------------------------------------------------------ the data

def load_fashion():
    """The 70,000 images, flattened to 784 floats in [0, 1], plus the raw uint8.

    Returns a dict; the split is the official one shipped with the dataset, so
    every course in the world reports on the same 10,000 test images.
    """
    tr = torchvision.datasets.FashionMNIST(DATA, train=True, download=True)
    te = torchvision.datasets.FashionMNIST(DATA, train=False, download=True)

    Xtr_u8 = tr.data.numpy()                       # (60000, 28, 28) uint8
    Xte_u8 = te.data.numpy()
    ytr = tr.targets.numpy().astype(np.int64)
    yte = te.targets.numpy().astype(np.int64)

    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(Xtr_u8))
    val_idx, fit_idx = order[:N_VAL], order[N_VAL:]

    flat = lambda a: a.reshape(len(a), -1).astype(np.float32) / 255.0
    return {
        "X_fit": flat(Xtr_u8[fit_idx]), "y_fit": ytr[fit_idx],
        "X_val": flat(Xtr_u8[val_idx]), "y_val": ytr[val_idx],
        "X_test": flat(Xte_u8), "y_test": yte,
        "X_fit_raw": Xtr_u8[fit_idx].reshape(len(fit_idx), -1).astype(np.float32),
        "X_val_raw": Xtr_u8[val_idx].reshape(N_VAL, -1).astype(np.float32),
        "grid_images": Xtr_u8[:40], "grid_labels": ytr[:40],
        "train_counts": np.bincount(ytr, minlength=10),
        "test_counts": np.bincount(yte, minlength=10),
        "pixel_sample": Xtr_u8[0],
    }


def n_params(hidden=HIDDEN, n_in=784, n_out=10) -> int:
    sizes = (n_in,) + tuple(hidden) + (n_out,)
    return sum(sizes[i] * sizes[i + 1] + sizes[i + 1] for i in range(len(sizes) - 1))


def layer_params(hidden=HIDDEN, n_in=784, n_out=10) -> dict:
    """The per-layer breakdown the slide shows as a worked sum.

    The slide prints each line of the arithmetic, so every one of those numbers
    is a quantity check_provenance.py must be able to trace — not only the
    total. The middle layer was missing while its neighbours were exported,
    which is exactly the kind of gap a per-line check catches and a per-total
    one does not."""
    sizes = (n_in,) + tuple(hidden) + (n_out,)
    rows = [{"in": sizes[i], "out": sizes[i + 1],
             "params": sizes[i] * sizes[i + 1] + sizes[i + 1]}
            for i in range(len(sizes) - 1)]
    return {"sizes": list(sizes), "layers": rows,
            "total": sum(r["params"] for r in rows)}


# --------------------------------------------------- Lecture 11 · the figures

def fig_grid(d):
    """Forty garments, four per class, with the class name over each column."""
    tr = torchvision.datasets.FashionMNIST(DATA, train=True, download=False)
    imgs, labs = tr.data.numpy(), tr.targets.numpy()
    chosen = np.concatenate([np.where(labs == c)[0][:4] for c in range(10)])

    fig, axes = plt.subplots(4, 10, figsize=(11.0, 5.0))
    for k, ax in enumerate(axes.T.ravel()):
        ax.imshow(imgs[chosen[k]], cmap="gray_r", vmin=0, vmax=255)
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
    for c in range(10):
        axes[0, c].set_title(CLASSES[c], fontsize=SMALL, color=MUTED,
                             loc="center", pad=6)
    fig.suptitle("Four examples of each of the ten classes, 28 × 28, greyscale",
                 fontsize=SMALL, color=MUTED, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, "l11-grid", raster=True)


def fig_class_balance(d):
    counts = d["train_counts"]
    fig, ax = plt.subplots(figsize=(11.0, 3.3))
    ax.bar(range(10), counts, color=PRIMARY, width=0.68)
    ax.axhline(counts.mean(), color=ACCENT, lw=2, ls="--")
    ax.annotate(f"every class: {counts[0]:,} images\n"
                f"so the majority-class baseline is exactly 10%",
                xy=(4.5, counts.mean()), xytext=(1.2, counts.mean() * 0.45),
                color=ACCENT,
                bbox=dict(boxstyle="round,pad=0.5", fc="white", ec=ACCENT, lw=1.2),
                arrowprops=dict(arrowstyle="->", color=ACCENT, lw=2))
    ax.set_xticks(range(10))
    ax.set_xticklabels(CLASSES, rotation=30, ha="right")
    ax.set_ylabel("training images")
    ax.set_ylim(0, counts.max() * 1.18)
    ax.set_title("Fashion MNIST is perfectly balanced — unlike the rare-event detector")
    fig.tight_layout()
    save(fig, "l11-class-balance")


def fig_activations():
    z = np.linspace(-4, 4, 400)
    step = (z >= 0).astype(float)
    sig = 1 / (1 + np.exp(-z))
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 3.4))

    a = axes[0]
    a.plot(z[z < 0], step[z < 0], color=ACCENT, lw=2.5)
    a.plot(z[z >= 0], step[z >= 0], color=ACCENT, lw=2.5)
    a.plot(z, np.zeros_like(z), color=MUTED, lw=2, ls=":")
    a.set_title("Heaviside step — the perceptron's")
    a.set_ylim(-0.35, 1.35)
    a.annotate("derivative is 0 everywhere it exists",
               xy=(1.6, 0.0), xytext=(-3.8, -0.28), color=MUTED,
               bbox=dict(boxstyle="round,pad=0.5", fc="white", ec=RULE, lw=1))
    a.set_xlabel("z")

    b = axes[1]
    b.plot(z, sig, color=PRIMARY, lw=2.5, label="sigmoid")
    b.plot(z, np.tanh(z), color=MATH, lw=2.5, label="tanh")
    b.plot(z, np.maximum(z, 0), color=SUCCESS, lw=2.5, label="ReLU")
    b.set_ylim(-1.2, 2.4)
    b.set_xlabel("z")
    b.legend(loc="upper left")
    b.set_title("The three that have a usable gradient")
    fig.tight_layout()
    save(fig, "l11-activations")


# ------------------------------------------- Lecture 11 · scikit-learn fitting

def sk_epoch_curve(d, hidden=HIDDEN, lr=1e-3, epochs=EPOCHS, batch=BATCH,
                   n_fit=None, raw=False):
    """Fit MLPClassifier one epoch at a time, recording loss and accuracy.

    partial_fit is the only per-epoch handle scikit-learn gives us, and it is
    exactly as far as the control goes: no per-batch loss, no gradients, no
    hook between the forward and the backward pass. That limit is the point of
    the lecture, so it is measured rather than asserted.
    """
    X = d["X_fit_raw"] if raw else d["X_fit"]
    Xv = d["X_val_raw"] if raw else d["X_val"]
    y, yv = d["y_fit"], d["y_val"]
    if n_fit:
        X, y = X[:n_fit], y[:n_fit]

    clf = MLPClassifier(hidden_layer_sizes=hidden, activation="relu",
                        solver="adam", learning_rate_init=lr,
                        batch_size=batch, random_state=SEED)
    loss, val_acc, train_acc = [], [], []
    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        for _ in range(epochs):
            clf.partial_fit(X, y, classes=np.arange(10))
            loss.append(float(clf.loss_))
            val_acc.append(float(clf.score(Xv, yv)))
            train_acc.append(float(clf.score(X[:10_000], y[:10_000])))
    secs = time.perf_counter() - t0
    return {"loss": loss, "val_acc": val_acc, "train_acc": train_acc,
            "seconds": secs, "n_fit": int(len(X)),
            "test_acc": float(clf.score(d["X_test"], d["y_test"])),
            "n_params": int(sum(w.size for w in clf.coefs_)
                            + sum(b.size for b in clf.intercepts_))}


def sk_sweeps(d):
    """Hand-tuning, as the students do it: architectures, then learning rates.

    On a 10,000-image subset so that ten fits fit inside the lecture hour.
    """
    sub = 10_000
    archs = [(30,), (100,), (300,), (300, 100), (300, 200, 100)]
    lrs = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2]

    arch_rows = []
    for h in archs:
        r = sk_epoch_curve(d, hidden=h, lr=1e-3, epochs=12, n_fit=sub)
        arch_rows.append({"hidden": list(h), "val_acc": r["val_acc"][-1],
                          "seconds": r["seconds"], "n_params": r["n_params"]})
        print(f"      hidden={h}  val {r['val_acc'][-1]:.4f}  "
              f"{r['seconds']:.1f}s  {r['n_params']:,} params")

    lr_rows = []
    for lr in lrs:
        r = sk_epoch_curve(d, hidden=HIDDEN, lr=lr, epochs=12, n_fit=sub)
        lr_rows.append({"lr": lr, "val_acc": r["val_acc"][-1],
                        "curve": r["val_acc"], "loss": r["loss"]})
        print(f"      lr={lr:<7g}  val {r['val_acc'][-1]:.4f}")
    return {"arch": arch_rows, "lr": lr_rows, "subset": sub, "epochs": 12}


def fig_sweeps(sw):
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 3.6))

    a = axes[0]
    labels = ["×".join(str(x) for x in r["hidden"]) for r in sw["arch"]]
    accs = [100 * r["val_acc"] for r in sw["arch"]]
    bars = a.barh(range(len(accs)), accs, color=PRIMARY, height=0.6)
    best = int(np.argmax(accs))
    bars[best].set_color(SUCCESS)
    a.set_yticks(range(len(accs))); a.set_yticklabels(labels)
    a.set_xlim(80, max(accs) + 2.2)
    a.set_xlabel("validation accuracy, %")
    a.set_title("Hidden layers")
    for i, v in enumerate(accs):
        a.text(v + 0.12, i, f"{v:.1f}", va="center", color=MUTED, fontsize=TICK)
    a.invert_yaxis()

    b = axes[1]
    best_lr = max(sw["lr"], key=lambda r: r["val_acc"])["lr"]
    for r in sw["lr"]:
        c = SUCCESS if r["lr"] == best_lr else RULE
        w = 2.6 if r["lr"] == best_lr else 1.8
        b.plot(range(1, len(r["curve"]) + 1), [100 * v for v in r["curve"]],
               color=c, lw=w)
        b.text(len(r["curve"]) + 0.15, 100 * r["curve"][-1],
               f"{r['lr']:g}", color=c if c != RULE else MUTED,
               fontsize=TICK, va="center")
    b.set_xlabel("epoch"); b.set_ylabel("validation accuracy, %")
    b.set_xlim(1, len(sw["lr"][0]["curve"]) + 3.2)
    b.set_title("Learning rate")
    fig.tight_layout()
    save(fig, "l11-sweeps")


def fig_loss_curve(run):
    ep = range(1, len(run["loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 3.4))
    axes[0].plot(ep, run["loss"], color=PRIMARY, lw=2.5, marker="o", ms=4)
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("training loss")
    axes[0].set_title("Cross-entropy on the training set")
    axes[1].plot(ep, [100 * v for v in run["train_acc"]], color=RULE, lw=2.5,
                 label="training")
    axes[1].plot(ep, [100 * v for v in run["val_acc"]], color=SUCCESS, lw=2.5,
                 label="validation")
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("accuracy, %")
    axes[1].legend(loc="lower right")
    axes[1].set_title("The gap that opens is the overfitting")
    gap = 100 * (run["train_acc"][-1] - run["val_acc"][-1])
    axes[1].annotate(f"{gap:.1f} points apart\nby epoch {len(run['loss'])}",
                     xy=(len(run["loss"]), 100 * run["val_acc"][-1]),
                     xytext=(len(run["loss"]) * 0.30,
                             100 * run["val_acc"][-1] - 4.0),
                     color=ACCENT,
                     bbox=dict(boxstyle="round,pad=0.5", fc="white", ec=ACCENT,
                               lw=1.2),
                     arrowprops=dict(arrowstyle="->", color=ACCENT, lw=2))
    fig.tight_layout()
    save(fig, "l11-loss-curve")


def fig_confusion(cm):
    cm = np.asarray(cm, dtype=float)
    pct = 100 * cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(6.6, 5.6))
    im = ax.imshow(pct, cmap="Blues", vmin=0, vmax=100)
    ax.set_xticks(range(10)); ax.set_yticks(range(10))
    ax.set_xticklabels(CLASSES, rotation=45, ha="right")
    ax.set_yticklabels(CLASSES)
    ax.grid(False)
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    worst = int(np.argmin(np.diag(pct)))
    ax.add_patch(plt.Rectangle((worst - 0.5, worst - 0.5), 1, 1,
                               fill=False, ec=ACCENT, lw=3))
    ax.set_title(f"Row-normalised, %. {CLASSES[worst]} is the hard one")
    fig.colorbar(im, ax=ax, shrink=0.82, label="% of the true class")
    fig.tight_layout()
    save(fig, "l11-confusion")
    return worst


# ---------------------------------------------- Lecture 12 · thread, autodiff

def autodiff_worked():
    """The scalar example the thread walks through, both ways, checked.

        w = x*y + sin(x)      L = w^2      at x = 2, y = 3
    """
    x = torch.tensor(2.0, dtype=torch.float64, requires_grad=True)
    y = torch.tensor(3.0, dtype=torch.float64, requires_grad=True)
    w = x * y + torch.sin(x)
    L = w ** 2
    L.backward()

    by_hand_x = 2 * (2 * 3 + np.sin(2)) * (3 + np.cos(2))
    by_hand_y = 2 * (2 * 3 + np.sin(2)) * 2
    return {"x": 2.0, "y": 3.0,
            "prod": 6.0, "sin_x": float(np.sin(2)), "w": float(w),
            "L": float(L),
            "dL_dw": float(2 * w),
            "dL_dx": float(x.grad), "dL_dy": float(y.grad),
            "hand_dL_dx": float(by_hand_x), "hand_dL_dy": float(by_hand_y),
            "agree": float(abs(float(x.grad) - by_hand_x))}


def _flat_loss_fn(net, X, y):
    """A closure taking a flat parameter vector to the scalar loss.

    torch.func wants a pure function; this is the standard way to get one.
    """
    from torch.func import functional_call
    names = [n for n, _ in net.named_parameters()]
    shapes = [p.shape for _, p in net.named_parameters()]
    sizes = [p.numel() for _, p in net.named_parameters()]
    lossf = nn.CrossEntropyLoss()

    def f(theta):
        out, i = {}, 0
        for n, s, k in zip(names, shapes, sizes):
            out[n] = theta[i:i + k].view(s)
            i += k
        return lossf(functional_call(net, out, (X,)), y)

    theta0 = torch.cat([p.detach().reshape(-1) for _, p in net.named_parameters()])
    return f, theta0


def forward_vs_reverse():
    """Time both modes on the same scalar loss, at four parameter counts.

    Forward mode needs one pass per *input* direction, so getting the whole
    gradient means P passes. Reverse mode needs one pass per *output*, and the
    loss is one number. Both are run in full at the small sizes; for the real
    266,610-parameter network only the per-pass costs are measured and the
    forward-mode total is stated as a projection, because running it would take
    hours.
    """
    from torch.func import jvp, vjp
    torch.manual_seed(SEED)
    rows = []
    for h in (2, 4, 8, 16, 32):
        net = nn.Sequential(nn.Linear(20, h), nn.ReLU(), nn.Linear(h, 3))
        X = torch.randn(64, 20)
        y = torch.randint(0, 3, (64,))
        f, theta = _flat_loss_fn(net, X, y)
        P = int(theta.numel())

        # warm up both paths: the first call through torch.func pays a tracing
        # cost that would otherwise land entirely on the smallest network and
        # make reverse mode look like it grows cheaper with P
        for _ in range(5):
            _, _pull = vjp(f, theta)
            _pull(torch.tensor(1.0))
            jvp(f, (theta,), (torch.zeros_like(theta),))

        # --- reverse: one backward pass gives every partial derivative
        t0 = time.perf_counter()
        for _ in range(20):
            _, pull = vjp(f, theta)
            g_rev = pull(torch.tensor(1.0))[0]
        t_rev = (time.perf_counter() - t0) / 20

        # --- forward: one jvp per coordinate direction, P of them
        basis = torch.eye(P)
        t0 = time.perf_counter()
        g_fwd = torch.stack([jvp(f, (theta,), (basis[i],))[1] for i in range(P)])
        t_fwd = time.perf_counter() - t0

        rows.append({"P": P, "t_reverse": t_rev, "t_forward_total": t_fwd,
                     "t_forward_per_pass": t_fwd / P,
                     "max_abs_diff": float((g_rev - g_fwd).abs().max())})
        print(f"      P={P:5d}  reverse {t_rev*1e3:7.2f} ms   "
              f"forward {t_fwd*1e3:9.1f} ms ({P} passes)   "
              f"agree to {rows[-1]['max_abs_diff']:.2e}")

    # --- and the per-pass cost on the network the course actually trains
    big = nn.Sequential(nn.Flatten(), nn.Linear(784, 300), nn.ReLU(),
                        nn.Linear(300, 100), nn.ReLU(), nn.Linear(100, 10))
    X = torch.randn(128, 784)
    y = torch.randint(0, 10, (128,))
    f, theta = _flat_loss_fn(big, X, y)
    P = int(theta.numel())
    t0 = time.perf_counter()
    for _ in range(20):
        _, pull = vjp(f, theta)
        pull(torch.tensor(1.0))
    t_rev = (time.perf_counter() - t0) / 20
    tan = torch.randn_like(theta)
    t0 = time.perf_counter()
    for _ in range(20):
        jvp(f, (theta,), (tan,))
    t_fwd1 = (time.perf_counter() - t0) / 20

    return {"small": rows,
            "big": {"P": P, "t_reverse": t_rev, "t_forward_per_pass": t_fwd1,
                    "t_forward_projected_s": t_fwd1 * P,
                    "t_forward_projected_min": t_fwd1 * P / 60.0,
                    "t_forward_projected_h": t_fwd1 * P / 3600.0,
                    "speedup": t_fwd1 * P / t_rev}}


def device_crossover(d, widths=((100,), (300, 100), (1000, 1000),
                                (2000, 2000)), epochs=2, n_fit=12_000):
    """When does the accelerator actually start to pay?

    The headline benchmark finds almost no difference between CPU and MPS,
    which is a real result and worth explaining rather than hiding: at 266,610
    parameters and a batch of 128 there is not enough arithmetic per batch to
    cover the cost of dispatching it. Widen the layers and the answer changes.
    """
    rows = []
    for w in widths:
        cpu = torch_train(d, "cpu", epochs=epochs, n_fit=n_fit, track=False,
                          widths=w)
        gpu = torch_train(d, DEVICE, epochs=epochs, n_fit=n_fit, track=False,
                          widths=w)
        rows.append({"widths": list(w), "n_params": cpu["n_params"],
                     "cpu_s": cpu["seconds"], "dev_s": gpu["seconds"],
                     "ratio": cpu["seconds"] / gpu["seconds"]})
        print(f"      {str(w):16s} {cpu['n_params']:>9,} params   "
              f"cpu {cpu['seconds']:6.1f}s   {DEVICE} {gpu['seconds']:6.1f}s   "
              f"{rows[-1]['ratio']:.2f}x")
    return {"rows": rows, "epochs": epochs, "n_fit": n_fit, "batch": BATCH}


def fig_crossover(cx):
    rows = cx["rows"]
    P = np.array([r["n_params"] for r in rows], dtype=float)
    ratio = np.array([r["ratio"] for r in rows])
    fig, ax = plt.subplots(figsize=(11.0, 3.3))
    ax.plot(P, ratio, color=PRIMARY, lw=2.5, marker="o", ms=7)
    ax.axhline(1.0, color=ACCENT, lw=2, ls="--")
    ax.set_xscale("log"); plain_log(ax, "x", fmt="{:,.0f}")
    ax.set_xlabel("parameters in the network")
    ax.set_ylabel(f"CPU seconds / {DEVICE.upper()} seconds")
    ax.set_ylim(0, max(ratio.max() * 1.25, 1.6))
    ax.annotate("below this line the\naccelerator is no help",
                xy=(P[0], 1.0), xytext=(P[0] * 1.4, max(ratio.max() * 0.55, 0.5)),
                color=ACCENT,
                bbox=dict(boxstyle="round,pad=0.5", fc="white", ec=ACCENT, lw=1.2),
                arrowprops=dict(arrowstyle="->", color=ACCENT, lw=2))
    for p, r, row in zip(P, ratio, rows):
        ax.annotate("×".join(str(x) for x in row["widths"]), xy=(p, r),
                    xytext=(0, 12), textcoords="offset points",
                    ha="center", color=MUTED, fontsize=TICK)
    ax.set_title(f"{cx['epochs']} epochs, {cx['n_fit']:,} images, "
                 f"batch {cx['batch']}, same code on both devices")
    fig.tight_layout()
    save(fig, "l12-device-crossover")


def fig_fwd_vs_rev(fr):
    rows = fr["small"]
    P = np.array([r["P"] for r in rows], dtype=float)
    rev = np.array([1e3 * r["t_reverse"] for r in rows])
    fwd = np.array([1e3 * r["t_forward_total"] for r in rows])

    fig, ax = plt.subplots(figsize=(11.0, 3.5))
    ax.plot(P, fwd, color=ACCENT, lw=2.5, marker="o", ms=6,
            label="forward mode — one pass per parameter")
    ax.plot(P, rev, color=SUCCESS, lw=2.5, marker="s", ms=6,
            label="reverse mode — one pass, total")
    ax.set_yscale("log"); plain_log(ax, "y", fmt="{:g}")
    ax.set_xlabel("number of parameters  P")
    ax.set_ylabel("time for the full gradient, ms")
    ax.legend(loc="center right")
    ratio = fwd[-1] / rev[-1]
    ax.annotate(f"at P = {int(P[-1])} the gap is already {ratio:,.0f}×\n"
                f"and it grows linearly in P",
                xy=(P[-1], fwd[-1]), xytext=(P[0] + 40, fwd[-1] * 0.10),
                color=ACCENT,
                bbox=dict(boxstyle="round,pad=0.5", fc="white", ec=ACCENT, lw=1.2),
                arrowprops=dict(arrowstyle="->", color=ACCENT, lw=2))
    ax.set_title("Same gradient, same values, two orders of association")
    fig.tight_layout()
    save(fig, "l12-fwd-vs-rev")
    return float(ratio)


# ------------------------------------------------- Lecture 12 · PyTorch build

def make_net(dropout: float = 0.0, widths=HIDDEN) -> nn.Module:
    sizes = (784,) + tuple(widths)
    layers: list = [nn.Flatten()]
    for a, b in zip(sizes, sizes[1:]):
        layers += [nn.Linear(a, b), nn.ReLU()]
        if dropout:
            layers.append(nn.Dropout(dropout))
    layers.append(nn.Linear(sizes[-1], 10))
    return nn.Sequential(*layers)


def _tensors(d, device):
    t = lambda a, dt: torch.tensor(a, dtype=dt, device=device)
    return (t(d["X_fit"], torch.float32), t(d["y_fit"], torch.long),
            t(d["X_val"], torch.float32), t(d["y_val"], torch.long),
            t(d["X_test"], torch.float32), t(d["y_test"], torch.long))


@torch.no_grad()
def accuracy(net, X, y, batch=1000) -> float:
    hits = 0
    for i in range(0, len(X), batch):
        hits += (net(X[i:i + batch]).argmax(1) == y[i:i + batch]).sum().item()
    return hits / len(X)


def torch_train(d, device, epochs=EPOCHS, batch=BATCH, lr=1e-3,
                dropout=0.0, zero_grad=True, n_fit=None, seed=SEED,
                track=True, widths=HIDDEN):
    """The training loop exactly as the lecture writes it on the slide."""
    torch.manual_seed(seed)
    Xf, yf, Xv, yv, Xt, yt = _tensors(d, device)
    if n_fit:
        Xf, yf = Xf[:n_fit], yf[:n_fit]

    net = make_net(dropout, widths).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    lossf = nn.CrossEntropyLoss()
    g = torch.Generator().manual_seed(seed)

    hist = {"loss": [], "val_acc": [], "train_acc": []}
    t0 = time.perf_counter()
    for _ in range(epochs):
        net.train()
        perm = torch.randperm(len(Xf), generator=g).to(device)
        running, nb = 0.0, 0
        for i in range(0, len(Xf), batch):
            idx = perm[i:i + batch]
            if zero_grad:
                opt.zero_grad()
            out = net(Xf[idx])
            loss = lossf(out, yf[idx])
            loss.backward()
            opt.step()
            running += loss.item(); nb += 1
        if track:
            net.eval()
            hist["loss"].append(running / nb)
            hist["val_acc"].append(accuracy(net, Xv, yv))
            hist["train_acc"].append(accuracy(net, Xf[:10_000], yf[:10_000]))
    if device == "mps":
        torch.mps.synchronize()
    secs = time.perf_counter() - t0

    net.eval()
    return {"hist": hist, "seconds": secs,
            "val_acc": accuracy(net, Xv, yv),
            "test_acc": accuracy(net, Xt, yt),
            "n_params": int(sum(p.numel() for p in net.parameters())),
            "state": {k: v.detach().cpu() for k, v in net.state_dict().items()}}


def fig_walltime(bench):
    names = list(bench.keys())
    secs = [bench[n]["seconds"] for n in names]
    colours = [ACCENT if "Scikit" in n else
               (SUCCESS if "MPS" in n else PRIMARY) for n in names]
    fig, ax = plt.subplots(figsize=(11.0, 3.2))
    bars = ax.barh(range(len(names)), secs, color=colours, height=0.6)
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel(f"wall-clock seconds for {EPOCHS} epochs on "
                  f"{bench[names[0]]['n_fit']:,} images")
    ax.set_xlim(0, max(secs) * 1.30)
    for b, s, n in zip(bars, secs, names):
        ax.text(s + max(secs) * 0.015, b.get_y() + b.get_height() / 2,
                f"{s:,.0f} s   ({bench[n]['test_acc'] * 100:.1f}% test)",
                va="center", color=MUTED, fontsize=TICK)
    ax.set_title("Identical architecture, identical optimiser, identical epochs")
    fig.tight_layout()
    save(fig, "l12-walltime")


def fig_zero_grad(with_zg, without_zg):
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 3.5))
    ep = range(1, len(with_zg["hist"]["loss"]) + 1)
    axes[0].plot(ep, with_zg["hist"]["loss"], color=SUCCESS, lw=2.5,
                 label="with opt.zero_grad()")
    axes[0].plot(ep, without_zg["hist"]["loss"], color=ACCENT, lw=2.5,
                 ls="--", label="without it")
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("training loss")
    axes[0].set_yscale("log"); plain_log(axes[0], "y", fmt="{:g}")
    axes[0].legend(loc="upper right")
    axes[0].set_title("No exception, no warning, no NaN")

    axes[1].plot(ep, [100 * v for v in with_zg["hist"]["val_acc"]],
                 color=SUCCESS, lw=2.5)
    axes[1].plot(ep, [100 * v for v in without_zg["hist"]["val_acc"]],
                 color=ACCENT, lw=2.5, ls="--")
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("validation accuracy, %")
    axes[1].set_ylim(0, 100)
    gap = 100 * (with_zg["val_acc"] - without_zg["val_acc"])
    axes[1].annotate(f"{gap:.1f} accuracy points,\nsilently",
                     xy=(len(ep), 100 * without_zg["val_acc"]),
                     xytext=(len(ep) * 0.28, 100 * without_zg["val_acc"] + 18),
                     color=ACCENT,
                     bbox=dict(boxstyle="round,pad=0.5", fc="white", ec=ACCENT,
                               lw=1.2),
                     arrowprops=dict(arrowstyle="->", color=ACCENT, lw=2))
    axes[1].set_title("One missing line")
    fig.tight_layout()
    save(fig, "l12-zerograd")


def measure_zero_grad(d, epochs=12, n_fit=20_000):
    """Train twice, identically seeded, with and without zero_grad()."""
    a = torch_train(d, DEVICE, epochs=epochs, n_fit=n_fit, zero_grad=True)
    b = torch_train(d, DEVICE, epochs=epochs, n_fit=n_fit, zero_grad=False)
    for r in (a, b):
        r.pop("state")
    return {"with": a, "without": b,
            "acc_cost_pts": 100 * (a["val_acc"] - b["val_acc"]),
            "loss_ratio": b["hist"]["loss"][-1] / a["hist"]["loss"][-1]}


def measure_eval_mode(d, epochs=12, n_fit=20_000, dropout=0.2, repeats=10):
    """Train with dropout, then evaluate in eval() mode and in train() mode.

    The second reading is what a missing model.eval() gives you: dropout still
    zeroing a fifth of every hidden layer at inference. It does not raise; it
    is not even deterministic, which is the tell if you look for it.
    """
    r = torch_train(d, DEVICE, epochs=epochs, n_fit=n_fit, dropout=dropout)
    net = make_net(dropout).to(DEVICE)
    net.load_state_dict({k: v.to(DEVICE) for k, v in r["state"].items()})
    _, _, Xv, yv, Xt, yt = _tensors(d, DEVICE)

    net.eval()
    correct = accuracy(net, Xt, yt)

    net.train()                                   # the bug
    torch.manual_seed(SEED)
    wrong = [accuracy(net, Xt, yt) for _ in range(repeats)]
    return {"eval_acc": correct,
            "train_mode_acc_mean": float(np.mean(wrong)),
            "train_mode_acc_min": float(np.min(wrong)),
            "train_mode_acc_max": float(np.max(wrong)),
            "train_mode_spread_pts": float(100 * (np.max(wrong) - np.min(wrong))),
            "cost_pts": float(100 * (correct - np.mean(wrong))),
            "dropout": dropout, "repeats": repeats,
            "val_acc": r["val_acc"], "seconds": r["seconds"]}


def batch_mean_bias(d, net_state, batch=384):
    """Averaging a per-batch metric is not the metric over the set.

    The last batch is short, so a plain mean of the per-batch accuracies
    over-weights it. Measured on the 10,000 test images, because that is the
    number the lecture reports.
    """
    net = make_net().to(DEVICE)
    net.load_state_dict({k: v.to(DEVICE) for k, v in net_state.items()})
    net.eval()
    _, _, _, _, Xt, yt = _tensors(d, DEVICE)

    per_batch, sizes, hits = [], [], 0
    with torch.no_grad():
        for i in range(0, len(Xt), batch):
            xb, yb = Xt[i:i + batch], yt[i:i + batch]
            h = (net(xb).argmax(1) == yb).sum().item()
            hits += h
            per_batch.append(h / len(xb)); sizes.append(len(xb))
    over_set = hits / len(Xt)
    naive = float(np.mean(per_batch))
    return {"batch": batch, "n_batches": len(per_batch),
            "last_batch_size": sizes[-1],
            "over_the_set": over_set, "mean_of_batches": naive,
            "bias_pts": 100 * (naive - over_set),
            "last_batch_acc": per_batch[-1]}


def fig_learning_curves(sk, pt):
    ep = range(1, len(pt["hist"]["val_acc"]) + 1)
    fig, ax = plt.subplots(figsize=(11.0, 3.4))
    ax.plot(range(1, len(sk["val_acc"]) + 1), [100 * v for v in sk["val_acc"]],
            color=ACCENT, lw=2.5, marker="o", ms=4,
            label="Scikit-Learn MLPClassifier (CPU)")
    ax.plot(ep, [100 * v for v in pt["hist"]["val_acc"]], color=SUCCESS,
            lw=2.5, marker="s", ms=4, label=f"PyTorch ({DEVICE.upper()})")
    ax.set_xlabel("epoch"); ax.set_ylabel("validation accuracy, %")
    ax.legend(loc="lower right")
    ax.set_title("The rebuild is not an improvement — it is the same model")
    d_end = abs(100 * (sk["val_acc"][-1] - pt["hist"]["val_acc"][-1]))
    ax.annotate(f"{d_end:.1f} points apart after {len(ep)} epochs",
                xy=(len(ep), 100 * pt["hist"]["val_acc"][-1]),
                xytext=(len(ep) * 0.30, 100 * pt["hist"]["val_acc"][-1] - 5),
                color=MUTED,
                bbox=dict(boxstyle="round,pad=0.5", fc="white", ec=RULE, lw=1),
                arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.8))
    fig.tight_layout()
    save(fig, "l12-learning-curves")


def run_optuna(d, n_trials=12, epochs=6, n_fit=12_000):
    """A small Optuna study. Non-examinable engineering, flagged as such."""
    try:
        import optuna
    except ImportError:
        return {"available": False}
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        drop = trial.suggest_float("dropout", 0.0, 0.5, step=0.1)
        r = torch_train(d, DEVICE, epochs=epochs, n_fit=n_fit, lr=lr,
                        dropout=drop, track=False, seed=SEED)
        return r["val_acc"]

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=SEED))
    t0 = time.perf_counter()
    study.optimize(objective, n_trials=n_trials)
    return {"available": True, "version": optuna.__version__,
            "n_trials": n_trials, "epochs": epochs, "n_fit": n_fit,
            "best_value": float(study.best_value),
            "best_lr": float(study.best_params["lr"]),
            "best_dropout": float(study.best_params["dropout"]),
            "seconds": time.perf_counter() - t0,
            "values": [float(t.value) for t in study.trials]}


def fig_optuna(op):
    if not op.get("available"):
        return
    v = [100 * x for x in op["values"]]
    running = np.maximum.accumulate(v)
    fig, ax = plt.subplots(figsize=(11.0, 3.2))
    ax.plot(range(1, len(v) + 1), v, "o", color=RULE, ms=8, label="trial")
    ax.plot(range(1, len(v) + 1), running, color=SUCCESS, lw=2.5,
            label="best so far")
    ax.set_xlabel("trial"); ax.set_ylabel("validation accuracy, %")
    ax.legend(loc="lower right")
    ax.set_title(f"{op['n_trials']} trials, {op['epochs']} epochs each, "
                 f"{op['n_fit']:,} images")
    fig.tight_layout()
    save(fig, "l12-optuna")


# ---------------------------------------------------------------- diagram XML

def validate_diagrams() -> list[str]:
    """A malformed hand-authored SVG fails silently to an empty box."""
    bad = []
    for p in sorted(OUT.glob("d-*.svg")):
        try:
            xml.dom.minidom.parse(str(p))
        except Exception as exc:                              # noqa: BLE001
            bad.append(f"{p.name}: {exc}")
    return bad


# ---------------------------------------------------------------------- main

def main() -> int:
    setup()
    load_cache()
    print("Fashion MNIST…")
    d = load_fashion()

    facts: dict = {
        "l11_n_train_total": int(len(d["X_fit"]) + N_VAL),
        "l11_n_fit": int(len(d["X_fit"])),
        "l11_n_val": N_VAL,
        "l11_n_test": int(len(d["X_test"])),
        "l11_n_total": int(len(d["X_fit"]) + N_VAL + len(d["X_test"])),
        "l11_n_pixels": 784,
        "l11_n_classes": 10,
        "l11_per_class_train": int(d["train_counts"][0]),
        "l11_per_class_test": int(d["test_counts"][0]),
        "l11_class_counts": {CLASSES[i]: int(d["train_counts"][i])
                             for i in range(10)},
        "l11_pixel_min": int(d["pixel_sample"].min()),
        "l11_pixel_max": 255,
        "l11_baseline_majority_acc": float(d["test_counts"].max()
                                           / d["test_counts"].sum()),
        "l11_n_params": n_params(),
        "l11_n_params_first_layer": 784 * 300 + 300,
        "l11_hidden": list(HIDDEN),
        "l11_epochs": EPOCHS,
        "l11_batch": BATCH,
        "l11_steps_per_epoch": -(-int(len(d["X_fit"])) // BATCH),
        "l11_total_steps": EPOCHS * (-(-int(len(d["X_fit"])) // BATCH)),
        "device": DEVICE,
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
    }

    print("Lecture 11 figures — the data:")
    fig_grid(d)
    fig_class_balance(d)
    fig_activations()

    print("Lecture 11 — Scikit-Learn, hand tuning on a 10,000-image subset:")
    sw = cached("app06_sk_sweeps", lambda: sk_sweeps(d))
    fig_sweeps(sw)
    facts["l11_sweep"] = sw
    facts["l11_best_arch"] = max(sw["arch"], key=lambda r: r["val_acc"])
    facts["l11_best_lr"] = max(sw["lr"], key=lambda r: r["val_acc"])["lr"]
    facts["l11_worst_lr_acc"] = min(r["val_acc"] for r in sw["lr"])

    print("Lecture 11 — the headline fit, 55,000 images, 20 epochs:")
    sk = cached("app06_sk_main", lambda: sk_epoch_curve(d))
    fig_loss_curve(sk)
    facts["l11_sk"] = {k: v for k, v in sk.items()}
    print(f"      test accuracy {sk['test_acc'] * 100:.2f}%  in "
          f"{sk['seconds']:.0f} s")

    print("Lecture 11 — the assistant's unscaled pixels:")
    raw = cached("app06_sk_raw",
                 lambda: sk_epoch_curve(d, epochs=12, n_fit=10_000, raw=True))
    scaled = cached("app06_sk_scaled_ref",
                    lambda: sk_epoch_curve(d, epochs=12, n_fit=10_000))
    facts["l11_unscaled"] = {
        "raw_val_acc": raw["val_acc"][-1], "scaled_val_acc": scaled["val_acc"][-1],
        "cost_pts": 100 * (scaled["val_acc"][-1] - raw["val_acc"][-1]),
        "raw_final_loss": raw["loss"][-1], "scaled_final_loss": scaled["loss"][-1],
        "n_fit": raw["n_fit"], "epochs": 12}
    print(f"      raw 0-255 {raw['val_acc'][-1]*100:.2f}%   "
          f"scaled {scaled['val_acc'][-1]*100:.2f}%")

    print("Lecture 11 — where the errors are:")
    cm = cached("app06_confusion", lambda: _confusion(d, sk))
    worst = fig_confusion(cm["matrix"])
    facts["l11_confusion"] = {"per_class_recall": cm["recall"],
                              "worst_class": CLASSES[worst],
                              "worst_recall": cm["recall"][CLASSES[worst]],
                              "shirt_as_tshirt": cm["shirt_as_tshirt"],
                              "shirt_as_pullover": cm["shirt_as_pullover"],
                              "shirt_as_coat": cm["shirt_as_coat"]}

    print("Thread 6 — the worked scalar example:")
    facts["l12_autodiff"] = autodiff_worked()
    print(f"      dL/dx = {facts['l12_autodiff']['dL_dx']:.6f}  "
          f"dL/dy = {facts['l12_autodiff']['dL_dy']:.6f}")

    print("Thread 6 — forward mode against reverse mode:")
    fr = cached("app06_fwd_rev_v2", forward_vs_reverse)
    # milliseconds and ratios, so a slide can quote one field rather than do
    # arithmetic that would then be unprovenanced
    for r in fr["small"]:
        r["t_reverse_ms"] = 1e3 * r["t_reverse"]
        r["t_forward_ms"] = 1e3 * r["t_forward_total"]
        r["ratio"] = r["t_forward_total"] / r["t_reverse"]
    fr["big"]["t_reverse_ms"] = 1e3 * fr["big"]["t_reverse"]
    fr["big"]["t_forward_per_pass_ms"] = 1e3 * fr["big"]["t_forward_per_pass"]
    # what a whole training run would cost in forward mode, at the measured
    # per-pass cost. Stated on the slide as a projection, because it is one.
    fr["big"]["full_training_days"] = (fr["big"]["t_forward_projected_s"]
                                       * facts["l11_total_steps"]
                                       / (3600 * 24))
    fr["big"]["full_training_years"] = fr["big"]["full_training_days"] / 365.25
    facts["l12_modes"] = fr
    facts["l12_modes"]["small_ratio_at_max_P"] = fig_fwd_vs_rev(fr)

    print("Lecture 12 — the same benchmark in PyTorch:")
    pt_cpu = cached("app06_torch_cpu",
                    lambda: _strip(torch_train(d, "cpu")))
    pt_dev = cached(f"app06_torch_{DEVICE}", lambda: torch_train(d, DEVICE))
    bench = {
        "Scikit-Learn · CPU": {"seconds": sk["seconds"],
                               "test_acc": sk["test_acc"],
                               "n_fit": sk["n_fit"]},
        "PyTorch · CPU": {"seconds": pt_cpu["seconds"],
                          "test_acc": pt_cpu["test_acc"],
                          "n_fit": int(len(d["X_fit"]))},
        f"PyTorch · {DEVICE.upper()}": {"seconds": pt_dev["seconds"],
                                        "test_acc": pt_dev["test_acc"],
                                        "n_fit": int(len(d["X_fit"]))},
    }
    fig_walltime(bench)
    fig_learning_curves(sk, pt_dev)
    facts["l12_bench"] = bench
    # flat aliases, so a slide can quote one number without a nested lookup
    facts["l12_sk_seconds"] = sk["seconds"]
    facts["l12_sk_test_acc"] = sk["test_acc"]
    facts["l12_pt_cpu_seconds"] = pt_cpu["seconds"]
    facts["l12_pt_cpu_test_acc"] = pt_cpu["test_acc"]
    facts["l12_pt_dev_seconds"] = pt_dev["seconds"]
    facts["l12_pt_dev_test_acc"] = pt_dev["test_acc"]
    facts["l12_pt_dev_val_acc"] = pt_dev["val_acc"]
    facts["l12_speedup_vs_sklearn"] = sk["seconds"] / pt_dev["seconds"]
    facts["l12_speedup_cpu_vs_mps"] = pt_cpu["seconds"] / pt_dev["seconds"]

    print(f"Lecture 12 — when does the accelerator start to pay ({DEVICE})?")
    cx = cached("app06_crossover", lambda: device_crossover(d))
    fig_crossover(cx)
    facts["l12_crossover"] = cx
    facts["l12_torch"] = {"val_acc": pt_dev["val_acc"],
                          "test_acc": pt_dev["test_acc"],
                          "n_params": pt_dev["n_params"],
                          "hist": pt_dev["hist"]}

    print("Lecture 12 — the cost of a missing zero_grad():")
    zg = cached("app06_zero_grad", lambda: measure_zero_grad(d))
    fig_zero_grad(zg["with"], zg["without"])
    facts["l12_zero_grad"] = {
        "with_val_acc": zg["with"]["val_acc"],
        "without_val_acc": zg["without"]["val_acc"],
        "acc_cost_pts": zg["acc_cost_pts"],
        "with_final_loss": zg["with"]["hist"]["loss"][-1],
        "without_final_loss": zg["without"]["hist"]["loss"][-1],
        "loss_ratio": zg["loss_ratio"],
        "epochs": len(zg["with"]["hist"]["loss"]),
        "n_fit": 20_000}
    print(f"      with {zg['with']['val_acc']*100:.2f}%   "
          f"without {zg['without']['val_acc']*100:.2f}%   "
          f"cost {zg['acc_cost_pts']:.2f} points")

    print("Lecture 12 — the cost of a missing model.eval():")
    em = cached("app06_eval_mode", lambda: measure_eval_mode(d))
    facts["l12_eval_mode"] = em
    print(f"      eval() {em['eval_acc']*100:.2f}%   "
          f"train() {em['train_mode_acc_mean']*100:.2f}% "
          f"(spread {em['train_mode_spread_pts']:.2f} pts)   "
          f"cost {em['cost_pts']:.2f} points")

    print("Lecture 12 — averaging the metric per batch:")
    bm = batch_mean_bias(d, pt_dev["state"])
    facts["l12_batch_mean"] = bm
    print(f"      over the set {bm['over_the_set']*100:.3f}%   "
          f"mean of batches {bm['mean_of_batches']*100:.3f}%   "
          f"bias {bm['bias_pts']:+.3f} points")

    print("Lecture 12 — Optuna:")
    op = cached("app06_optuna", lambda: run_optuna(d))
    fig_optuna(op)
    facts["l12_optuna"] = op

    print("Saving the model:")
    ckpt = OUT.parent.parent / "notebooks"          # not committed; size only
    size_kb = sum(v.numel() * v.element_size()
                  for v in pt_dev["state"].values()) / 1024
    facts["l12_checkpoint_kb"] = float(size_kb)
    print(f"      state_dict is {size_kb:,.0f} KB")

    facts["l11_layer_params"] = layer_params()
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


def _strip(r):
    r.pop("state", None)
    return r


def _confusion(d, sk_run):
    """Refit the headline model once more to get its confusion matrix."""
    clf = MLPClassifier(hidden_layer_sizes=HIDDEN, activation="relu",
                        solver="adam", learning_rate_init=1e-3,
                        batch_size=BATCH, max_iter=EPOCHS, random_state=SEED)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        clf.fit(d["X_fit"], d["y_fit"])
    from sklearn.metrics import confusion_matrix
    m = confusion_matrix(d["y_test"], clf.predict(d["X_test"]))
    recall = {CLASSES[i]: float(m[i, i] / m[i].sum()) for i in range(10)}
    return {"matrix": m.tolist(), "recall": recall,
            "shirt_as_tshirt": float(m[6, 0] / m[6].sum()),
            "shirt_as_pullover": float(m[6, 2] / m[6].sum()),
            "shirt_as_coat": float(m[6, 4] / m[6].sum())}


if __name__ == "__main__":
    sys.exit(main())
