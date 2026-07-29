#!/usr/bin/env python3
"""
Application 2 — rare-event detection on MNIST. Figures and numbers for
Lectures 3 and 4.

    python3 tools/figures_app02.py

Everything here is measured. The decks quote nothing this script has not
written to assets/figures/figures.json through figkit.export(), which merges
rather than overwrites so that each application can own its own script.

The expensive parts — three cross-validated fits of an SGD classifier on
60,000 x 784 pixels, and a five-seed repeat of the same — are cached, so a
cosmetic re-run takes seconds. Delete the cache file
(/private/tmp/claude-501/aiml-data/app02-fits.pkl) to refit from scratch.

The cache is this application's own file rather than figkit's shared one.
figkit.cached rewrites the whole shared dict on every miss, so two authors
generating figures at the same time silently overwrite each other's entries —
which happened once here and cost a seven-minute refit. Same mechanism, one
file per application.

Wall clock from cold: about seven minutes. From the cache: about twenty
seconds.
"""

from __future__ import annotations

import pickle
import time
import urllib.error
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from figkit import (setup, save, export, check_text_floor,
                    PRIMARY, ACCENT, SUCCESS, MATH, MUTED, RULE, AXIS,
                    BODY, SMALL, TICK, SEED)

ROOT = Path(__file__).resolve().parent.parent
CACHE = Path("/private/tmp/claude-501/aiml-data")

# The one split the whole application uses. MNIST arrives already shuffled and
# already partitioned: the first 60,000 rows are the training set and the last
# 10,000 the test set, so no train_test_split call is needed or wanted.
N_TRAIN = 60_000
N_FOLDS = 3
DIGIT = 5

# One shift on the sorting line is exactly the test set: 10,000 scanned digits.
# Every per-shift count on a Lecture 4 slide is therefore a literal count, not
# a rate multiplied by an invented volume.
DESK_CAPACITY = 1_000        # items the verification desk can re-check, per shift
RECALL_TARGET = 0.90         # the audit contract's stated floor

SOFT = "#9fb8ca"

# ---------------------------------------------------------------- the cache

_CACHE_FILE = CACHE / "app02-fits.pkl"
_cache: dict = {}


def load_cache() -> None:
    global _cache
    _cache = (pickle.loads(_CACHE_FILE.read_bytes())
              if _CACHE_FILE.is_file() else {})


def cached(key, fn):
    """Run fn() once and remember the result across runs. See the module note."""
    if key in _cache:
        print(f"    [cached] {key}")
        return _cache[key]
    print(f"    [computing] {key}")
    t0 = time.time()
    value = fn()
    _cache[key] = value
    CACHE.mkdir(parents=True, exist_ok=True)
    _CACHE_FILE.write_bytes(pickle.dumps(_cache))
    print(f"    [done] {key} in {time.time() - t0:.0f} s")
    return value


# ------------------------------------------------------------------ the data

def load_mnist():
    """fetch_openml once, then a local .npz. ~30 s cold, ~1 s afterwards.

    Stored as uint8: the same 70,000 x 784 array is 55 MB rather than 440 MB,
    and every downstream call casts to float anyway.
    """
    npz = CACHE / "mnist784.npz"
    if not npz.is_file():
        CACHE.mkdir(parents=True, exist_ok=True)
        from sklearn.datasets import fetch_openml
        print("  fetching mnist_784 from openml (about 30 s)…")
        try:
            m = fetch_openml("mnist_784", as_frame=False, parser="auto")
        except urllib.error.URLError as exc:            # pragma: no cover
            raise SystemExit(f"cannot reach openml: {exc}")
        np.savez_compressed(npz, X=m.data.astype(np.uint8),
                            y=m.target.astype(np.uint8))
    d = np.load(npz)
    return d["X"], d["y"]


def pipeline(seed=SEED):
    """The classifier the lecture builds, as one estimator.

    The standing constraint from Lecture 2 — all preprocessing inside a
    Pipeline passed to cross-validation — applies here too, which is why the
    scaler is not a separate fit_transform call.
    """
    from sklearn.linear_model import SGDClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(StandardScaler(), SGDClassifier(random_state=seed))


def folds(seed=SEED):
    from sklearn.model_selection import StratifiedKFold
    return StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)


# ------------------------------------------------------- measurement helpers

def measure_cv_accuracy(Xtr, y5):
    """Accuracy of the built classifier, and of the same thing unscaled.

    Both are reported per fold. The unscaled variant is not a straw man: it is
    what you get if you drop the pipeline, and its point is the spread.
    """
    from sklearn.linear_model import SGDClassifier
    from sklearn.model_selection import cross_val_score
    out = {}
    for name, est in (("scaled", pipeline()),
                      ("raw", SGDClassifier(random_state=SEED))):
        t0 = time.time()
        s = cross_val_score(est, Xtr, y5, cv=folds(), scoring="accuracy",
                            n_jobs=-1)
        out[name] = {"folds": [float(v) for v in s],
                     "mean": float(s.mean()), "std": float(s.std()),
                     "min": float(s.min()), "max": float(s.max()),
                     "seconds": round(time.time() - t0, 1)}
    return out


def measure_cv_scores(Xtr, y5):
    """Out-of-fold decision-function values for every training instance.

    SGDClassifier predicts the positive class exactly when the decision
    function is >= 0, so one cross_val_predict call gives both the labels and
    the whole threshold sweep — half the fits of asking for each separately.
    """
    from sklearn.model_selection import cross_val_predict
    s = cross_val_predict(pipeline(), Xtr, y5, cv=folds(),
                          method="decision_function", n_jobs=-1)
    return np.asarray(s, dtype=np.float64)


def measure_train_vs_cv(Xtr, y5, seeds=(42, 43, 44, 45, 46)):
    """The damage done by scoring on the data the model was fitted to.

    Paired: each fold contributes one (training accuracy, held-out accuracy)
    from the *same* fitted model, so the fold-to-fold variation cancels. Five
    seeds x three folds = fifteen pairs, because a single-seed gap of half a
    point is not a measurement.
    """
    from sklearn.model_selection import cross_validate
    pairs = []
    for s in seeds:
        r = cross_validate(pipeline(s), Xtr, y5, cv=folds(s),
                           scoring="accuracy", return_train_score=True,
                           n_jobs=-1)
        pairs += [(float(a), float(b))
                  for a, b in zip(r["train_score"], r["test_score"])]
    gaps = np.array([a - b for a, b in pairs])
    return {"pairs": [[a, b] for a, b in pairs],
            "n_seeds": len(seeds), "n_pairs": len(pairs),
            "mean_train": float(np.mean([a for a, _ in pairs])),
            "mean_cv": float(np.mean([b for _, b in pairs])),
            "gap_mean_pp": float(100 * gaps.mean()),
            "gap_std_pp": float(100 * gaps.std(ddof=1)),
            "gap_min_pp": float(100 * gaps.min()),
            "gap_max_pp": float(100 * gaps.max()),
            "gap_positive": int((gaps > 0).sum())}


def measure_forest(Xtr, y5):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_predict
    p = cross_val_predict(
        RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1),
        Xtr, y5, cv=folds(), method="predict_proba")
    return np.asarray(p[:, 1], dtype=np.float64)


def measure_test(Xtr, y5, Xte, y5te, thresholds):
    """Fit once on all 60,000 training rows, then score the test shift once.

    `thresholds` maps a name to (kind, value); kind is "sgd" or "forest".
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import confusion_matrix
    sgd = pipeline().fit(Xtr, y5)
    rf = RandomForestClassifier(n_estimators=100, random_state=SEED,
                                n_jobs=-1).fit(Xtr, y5)
    s_sgd = sgd.decision_function(Xte)
    s_rf = rf.predict_proba(Xte)[:, 1]

    out = {"sgd_train_accuracy": float(sgd.score(Xtr, y5))}
    for name, (kind, thr) in thresholds.items():
        pred = (s_sgd if kind == "sgd" else s_rf) >= thr
        tn, fp, fn, tp = confusion_matrix(y5te, pred).ravel()
        out[name] = {
            "threshold": float(thr),
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
            "flagged": int(tp + fp),
            "accuracy": float((pred == y5te).mean()),
            "precision": float(tp / (tp + fp)) if tp + fp else 0.0,
            "recall": float(tp / (tp + fn)),
            "within_capacity": bool(tp + fp <= DESK_CAPACITY),
        }
    return out


def at_precision(prec, rec, thr, target):
    """The lowest threshold whose precision reaches `target`."""
    i = int((prec >= target).argmax())
    return {"threshold": float(thr[min(i, len(thr) - 1)]),
            "precision": float(prec[i]), "recall": float(rec[i])}


def at_recall(prec, rec, thr, target):
    """The highest threshold whose recall is still at least `target`."""
    i = int(np.where(rec >= target)[0][-1])
    return {"threshold": float(thr[min(i, len(thr) - 1)]),
            "precision": float(prec[i]), "recall": float(rec[i])}


def rank_walk(scores, y5):
    """Precision and recall as the threshold descends one instance at a time.

    This is the whole of thread 2, computed rather than asserted. Sorting by
    score and walking down the list is the same sweep `precision_recall_curve`
    performs; doing it by hand is what makes the counting argument visible.
    """
    order = np.argsort(-scores, kind="stable")
    lab = y5[order].astype(np.int64)
    tp = np.cumsum(lab)
    n = np.arange(1, len(lab) + 1)
    prec = tp / n
    rec = tp / lab.sum()
    # Walking the threshold UP means going from n+1 predicted positives to n,
    # which drops the instance ranked n+1. d = prec[n] - prec[n-1] compares the
    # two, so d > 0 is a fall when read in the rising-threshold direction.
    d = np.diff(prec)
    dropped_positive = lab[1:] == 1      # the instance each step removes
    return {
        "tp": tp, "prec": prec, "rec": rec, "n": n, "lab": lab,
        "n_steps": int(len(prec) - 1),
        "n_positives": int(lab.sum()),
        "n_negatives": int(len(lab) - lab.sum()),
        # of the steps that drop a 5, precision falls at all but the handful
        # where it was already 1 and stays there
        "n_steps_dropping_a_five": int(dropped_positive.sum()),
        "precision_falls_when_threshold_rises": int((d > 0).sum()),
        "precision_flat_when_threshold_rises": int(
            ((d == 0) & dropped_positive).sum()),
        "precision_rises_when_threshold_rises": int((d < 0).sum()),
        "top10": [[int(k + 1), int(tp[k]), float(prec[k])] for k in range(10)],
    }


# ------------------------------------------------------------------- L3 figs

def fig_digits(X, y):
    """A strip of the real images, with every 5 ringed."""
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(y), size=48, replace=False)
    fig, axes = plt.subplots(3, 16, figsize=(10.6, 2.35))
    for ax, i in zip(axes.ravel(), idx):
        ax.imshow(X[i].reshape(28, 28), cmap="binary", interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(y[i] == DIGIT)
            sp.set_color(ACCENT); sp.set_linewidth(2.5)
        ax.grid(False)
    n5 = int((y[idx] == DIGIT).sum())
    fig.suptitle(f"48 digits drawn at random — {n5} of them are 5s, ringed in red",
                 x=0.012, ha="left", fontsize=BODY, color=MUTED)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    return save(fig, "l03-digits", raster=True), n5


def fig_class_counts(ytr):
    counts = np.bincount(ytr, minlength=10)
    fig, ax = plt.subplots(figsize=(10.6, 3.3))
    colours = [ACCENT if d == DIGIT else SOFT for d in range(10)]
    bars = ax.bar(np.arange(10), counts, color=colours, width=0.72)
    for d, (b, c) in enumerate(zip(bars, counts)):
        ax.text(b.get_x() + b.get_width() / 2, c + 120, f"{c:,}", ha="center",
                fontsize=SMALL, color=ACCENT if d == DIGIT else MUTED,
                fontweight="bold" if d == DIGIT else "normal")
    ax.set_xticks(np.arange(10))
    ax.set_xlabel("digit")
    ax.set_ylabel("images in the training set")
    ax.set_title("The ten classes are balanced. The task we were given is not.")
    ax.set_ylim(0, counts.max() * 1.42)
    ax.grid(axis="x", alpha=0)
    p = 100 * counts[DIGIT] / counts.sum()
    ax.annotate(f"our positive class:\n{counts[DIGIT]:,} of {counts.sum():,} = {p:.1f}%",
                xy=(DIGIT, counts[DIGIT]), xytext=(6.4, counts.max() * 1.20),
                fontsize=SMALL, color=ACCENT, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.5", fc="white", ec=ACCENT, lw=1.2),
                arrowprops=dict(arrowstyle="->", color=ACCENT, lw=1.8))
    fig.tight_layout()
    save(fig, "l03-class-counts")
    return {str(d): int(c) for d, c in enumerate(counts)}


def fig_baseline(ytr):
    """What a detector that never fires scores, for each of the ten digits."""
    counts = np.bincount(ytr, minlength=10)
    acc = 100 * (1 - counts / counts.sum())
    fig, ax = plt.subplots(figsize=(10.6, 3.3))
    colours = [ACCENT if d == DIGIT else SOFT for d in range(10)]
    bars = ax.barh(np.arange(10), acc, color=colours, height=0.70)
    for d, (b, a) in enumerate(zip(bars, acc)):
        # outside the bar: white on #9fb8ca is about 2:1 and unreadable
        ax.text(a + 0.06, b.get_y() + b.get_height() / 2,
                f"{a:.2f}%" + ("   ← ours" if d == DIGIT else ""),
                va="center", ha="left", fontsize=SMALL,
                color=ACCENT if d == DIGIT else MUTED,
                fontweight="bold" if d == DIGIT else "normal")
    ax.set_yticks(np.arange(10)); ax.set_ylabel("detector for digit")
    ax.set_xlabel("accuracy of a detector that always answers “no”")
    ax.set_xlim(85, 92.6)
    ax.invert_yaxis()
    ax.grid(axis="y", alpha=0)
    ax.set_title("The cheapest possible model, for each of the ten tasks")
    fig.tight_layout()
    save(fig, "l03-baseline")
    return {str(d): float(a) for d, a in enumerate(acc)}


def fig_folds(cv, base_acc):
    """Three folds, twice: with the pipeline and without it."""
    fig, ax = plt.subplots(figsize=(10.6, 3.4))
    rows = [("with the scaler, inside the pipeline", cv["scaled"], PRIMARY),
            ("the same classifier, unscaled", cv["raw"], ACCENT)]
    for i, (label, r, colour) in enumerate(rows):
        f = np.array(r["folds"]) * 100
        ax.scatter(f, [i] * len(f), s=150, color=colour, zorder=3,
                   label=None, clip_on=False)
        ax.plot([f.min(), f.max()], [i, i], color=colour, lw=2.5, alpha=0.35,
                zorder=2)
        ax.text(f.max() + 0.14, i, f"mean {f.mean():.2f}%", ha="left",
                va="center", fontsize=SMALL, color=colour, fontweight="bold")
        ax.text(f.min(), i + 0.26, f"worst fold {f.min():.2f}%", ha="center",
                va="bottom", fontsize=SMALL, color=MUTED)
    ax.axvline(base_acc * 100, color=MUTED, lw=2, ls="--")
    ax.text(base_acc * 100 + 0.16, -0.42,
            f"a detector that never fires — {base_acc * 100:.2f}%", ha="left",
            va="center", fontsize=SMALL, color=MUTED, fontweight="bold")
    ax.set_yticks([0, 1], [r[0] for r in rows], fontsize=TICK)
    ax.set_ylim(-0.62, 1.70)
    ax.set_xlim(90.0, 98.0)
    ax.set_xlabel("accuracy on the held-out fold (%)")
    ax.set_title("Three folds each. Report all of them, not the average.")
    ax.grid(axis="y", alpha=0)
    fig.tight_layout()
    return save(fig, "l03-folds")


def fig_train_vs_cv(tvc):
    pairs = np.array(tvc["pairs"]) * 100
    fig, ax = plt.subplots(figsize=(10.6, 3.4))
    x = np.arange(len(pairs))
    for i, (tr, cvv) in enumerate(pairs):
        ax.plot([i, i], [cvv, tr], color=RULE, lw=1.6, zorder=1)
    ax.scatter(x, pairs[:, 0], s=90, color=ACCENT, zorder=3,
               label="scored on the rows it was fitted to")
    ax.scatter(x, pairs[:, 1], s=90, color=PRIMARY, zorder=3,
               label="scored on the held-out fold")
    ax.set_xticks([])
    ax.set_xlabel(f"{tvc['n_pairs']} paired measurements "
                  f"({tvc['n_seeds']} seeds × {N_FOLDS} folds)")
    ax.set_ylabel("accuracy (%)")
    ax.set_title("The same fitted model, scored twice")
    ax.legend(loc="lower left", ncols=1)
    ax.set_ylim(96.15, 97.9)
    ax.grid(axis="x", alpha=0)
    ax.annotate(f"gap {tvc['gap_mean_pp']:.2f} pp\n"
                f"(sd {tvc['gap_std_pp']:.2f}, same sign "
                f"{tvc['gap_positive']}/{tvc['n_pairs']})",
                xy=(x[9], pairs[9].mean()), xytext=(x[10], 96.42),
                fontsize=SMALL, color=ACCENT, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.5", fc="white", ec=ACCENT, lw=1.2),
                arrowprops=dict(arrowstyle="->", color=ACCENT, lw=1.8))
    fig.tight_layout()
    return save(fig, "l03-train-vs-cv")


# ------------------------------------------------------------------- L4 figs

def fig_never_fires(base_rate, our_acc):
    p = np.linspace(0.001, 0.5, 400)
    fig, ax = plt.subplots(figsize=(10.6, 3.4))
    ax.plot(100 * p, 100 * (1 - p), color=ACCENT, lw=3,
            label="a classifier that never fires")
    ax.axhline(100 * our_acc, color=PRIMARY, lw=2, ls="--")
    ax.text(48, 100 * our_acc + 0.9, f"our classifier — {100 * our_acc:.2f}%",
            ha="right", fontsize=SMALL, color=PRIMARY, fontweight="bold")
    ax.scatter([100 * base_rate], [100 * (1 - base_rate)], s=170, color=ACCENT,
               zorder=4)
    ax.annotate(f"detecting a 5\n{100 * base_rate:.2f}% positive\n"
                f"→ {100 * (1 - base_rate):.2f}% accuracy",
                xy=(100 * base_rate, 100 * (1 - base_rate)),
                xytext=(15, 62), fontsize=SMALL, color=ACCENT, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.5", fc="white", ec=ACCENT, lw=1.2),
                arrowprops=dict(arrowstyle="->", color=ACCENT, lw=1.8))
    ax.set_xlabel("base rate — positives as a percentage of the data")
    ax.set_ylabel("accuracy (%)")
    ax.set_ylim(48, 102); ax.set_xlim(0, 50)
    ax.set_title("Accuracy of the model that does nothing, against how rare the event is")
    ax.legend(loc="lower left")
    fig.tight_layout()
    return save(fig, "l04-never-fires")


def fig_accuracy_weights(base_rate, spec):
    r = np.linspace(0, 1, 200)
    fig, ax = plt.subplots(figsize=(10.6, 3.4))
    for p, colour, lw in ((0.5, SOFT, 2.2), (base_rate, ACCENT, 3.0),
                          (0.01, MATH, 2.2)):
        ax.plot(100 * r, 100 * ((1 - p) * spec + p * r), color=colour, lw=lw,
                label=f"base rate {100 * p:.2f}%")
    ax.set_xlabel("recall (%)")
    ax.set_ylabel("accuracy (%)")
    ax.set_title("Accuracy is a weighted average, and recall carries the small weight")
    ax.legend(loc="lower right")
    ax.set_ylim(40, 105)
    ax.annotate("at our base rate, ten points of recall buy\n"
                f"{base_rate:.5f} × 10 = {100 * base_rate * 0.10:.2f} points of accuracy",
                xy=(50, 100 * ((1 - base_rate) * spec + base_rate * 0.5)),
                xytext=(14, 46), fontsize=SMALL, color=ACCENT, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.5", fc="white", ec=ACCENT, lw=1.2),
                arrowprops=dict(arrowstyle="->", color=ACCENT, lw=1.8))
    fig.tight_layout()
    return save(fig, "l04-accuracy-weights")


def fig_confusion(cm):
    tn, fp, fn, tp = cm
    m = np.array([[tn, fp], [fn, tp]], dtype=float)
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    ax.imshow(np.log10(m + 1), cmap="Blues", vmin=0, vmax=np.log10(m.max()) * 1.25)
    labels = [["true negatives", "false positives"],
              ["false negatives", "true positives"]]
    for i in range(2):
        for j in range(2):
            colour = "white" if i == j == 0 else "#16212b"
            ax.text(j, i - 0.12, f"{int(m[i, j]):,}", ha="center", va="center",
                    fontsize=25, fontweight="bold", color=colour)
            ax.text(j, i + 0.20, labels[i][j], ha="center", va="center",
                    fontsize=SMALL, color=colour)
    ax.set_xticks([0, 1], ["predicted “not a 5”", "predicted “a 5”"])
    ax.set_yticks([0, 1], ["actually not a 5", "actually a 5"])
    ax.set_title("Cross-validated predictions for all 60,000 training digits")
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.add_patch(Rectangle((0.5, 0.5), 1, 1, fill=False, ec=SUCCESS, lw=3))
    ax.add_patch(Rectangle((-0.5, 0.5), 1, 1, fill=False, ec=ACCENT, lw=3, ls="--"))
    fig.tight_layout()
    return save(fig, "l04-confusion")


def fig_nonmonotone(walk):
    prec, rec, n = walk["prec"], walk["rec"], walk["n"]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10.6, 3.7),
                                  gridspec_kw={"width_ratios": [1.35, 1]})
    keep = n <= 20_000
    ax.plot(n[keep], 100 * prec[keep], color=MATH, lw=2.5, label="precision")
    ax.plot(n[keep], 100 * rec[keep], color=PRIMARY, lw=2.5, label="recall")
    ax.set_xlabel("instances predicted positive — the threshold falls to the right")
    ax.set_ylabel("%")
    ax.set_title("Recall only ever goes up. Precision does not.")
    ax.legend(loc="center right")
    ax.set_ylim(0, 104)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:,.0f}")

    k = np.arange(1, 13)
    ax2.step(k, 100 * prec[:12], where="mid", color=MATH, lw=2.5)
    ax2.scatter(k, 100 * prec[:12], s=70, color=MATH, zorder=3)
    for i, dy in ((4, -4.5), (5, +3.0)):
        ax2.text(i + 1, 100 * prec[i] + dy, f"{walk['tp'][i]}/{i + 1}",
                 ha="center", va="top" if dy < 0 else "bottom",
                 fontsize=SMALL, color=ACCENT, fontweight="bold")
    ax2.annotate("raise the threshold past\nthe 6th-ranked digit — a 5 —\n"
                 "and precision falls",
                 xy=(5.5, 81.7), xytext=(6.7, 59),
                 fontsize=SMALL, color=ACCENT, fontweight="bold",
                 bbox=dict(boxstyle="round,pad=0.5", fc="white", ec=ACCENT, lw=1.2),
                 arrowprops=dict(arrowstyle="->", color=ACCENT, lw=1.8))
    ax2.set_xlabel("the top of our own ranking")
    ax2.set_ylabel("precision (%)")
    ax2.set_title("Zoomed to the first twelve")
    ax2.set_ylim(55, 108); ax2.set_xticks(k[::2]); ax2.set_xlim(0.4, 12.6)
    fig.tight_layout()
    return save(fig, "l04-nonmonotone")


def fig_pr_curve(prec, rec, thr, op):
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10.6, 3.7))
    ax.plot(thr[:-1] if len(thr) == len(prec) else thr, 100 * prec[:len(thr)],
            color=MATH, lw=2.5, label="precision")
    ax.plot(thr, 100 * rec[:len(thr)], color=PRIMARY, lw=2.5, label="recall")
    ax.axvline(op["threshold"], color=ACCENT, lw=2, ls="--")
    ax.text(op["threshold"] - 25, 8, f"threshold {op['threshold']:.0f}",
            ha="right", fontsize=SMALL, color=ACCENT, fontweight="bold")
    ax.set_xlim(-400, 400)
    ax.set_xlabel("decision threshold")
    ax.set_ylabel("%")
    ax.set_title("Both metrics against the threshold")
    ax.legend(loc="center left")

    ax2.plot(100 * rec, 100 * prec, color=MATH, lw=2.5)
    ax2.scatter([100 * op["recall"]], [100 * op["precision"]], s=170,
                color=ACCENT, zorder=4)
    ax2.annotate(f"90% precision\n→ {100 * op['recall']:.1f}% recall",
                 xy=(100 * op["recall"], 100 * op["precision"]),
                 xytext=(24, 34), fontsize=SMALL, color=ACCENT, fontweight="bold",
                 bbox=dict(boxstyle="round,pad=0.5", fc="white", ec=ACCENT, lw=1.2),
                 arrowprops=dict(arrowstyle="->", color=ACCENT, lw=1.8))
    ax2.set_xlabel("recall (%)"); ax2.set_ylabel("precision (%)")
    ax2.set_title("The same information, plotted against itself")
    ax2.set_xlim(0, 101); ax2.set_ylim(0, 104)
    fig.tight_layout()
    return save(fig, "l04-pr-curve")


def fig_roc(sgd_roc, rf_roc):
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    ax.plot([0, 100], [0, 100], color=MUTED, lw=1.6, ls=":",
            label="a coin, weighted any way you like")
    ax.plot(100 * sgd_roc["fpr"], 100 * sgd_roc["tpr"], color=PRIMARY, lw=2.8,
            label=f"SGD — AUC {sgd_roc['auc']:.4f}")
    ax.plot(100 * rf_roc["fpr"], 100 * rf_roc["tpr"], color=SUCCESS, lw=2.8,
            label=f"random forest — AUC {rf_roc['auc']:.4f}")
    ax.set_xlabel("false positive rate (%) — non-5s we flag")
    ax.set_ylabel("recall (%) — 5s we catch")
    ax.set_title("The ROC curve of both classifiers")
    ax.legend(loc="lower right")
    ax.set_xlim(0, 100); ax.set_ylim(0, 101)
    fig.tight_layout()
    return save(fig, "l04-roc")


def fig_pr_vs_roc(rarity):
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10.6, 3.7))
    colours = [PRIMARY, MATH, ACCENT]
    for (row, colour) in zip(rarity, colours):
        ax.plot(100 * np.array(row["fpr"]), 100 * np.array(row["tpr"]),
                color=colour, lw=2.5,
                label=f"{100 * row['base_rate']:.2f}% positive — AUC {row['auc']:.3f}")
        ax2.plot(100 * np.array(row["rec"]), 100 * np.array(row["prec"]),
                 color=colour, lw=2.5,
                 label=f"{100 * row['base_rate']:.2f}% — AP {row['ap']:.3f}")
    ax.set_xlabel("false positive rate (%)"); ax.set_ylabel("recall (%)")
    ax.set_title("ROC — barely moves")
    ax.legend(loc="lower right", fontsize=TICK - 1)
    ax.set_xlim(0, 100); ax.set_ylim(0, 101)
    ax2.set_xlabel("recall (%)"); ax2.set_ylabel("precision (%)")
    ax2.set_title("Precision–recall — collapses")
    ax2.legend(loc="lower left", fontsize=TICK - 1)
    ax2.set_xlim(0, 101); ax2.set_ylim(0, 104)
    fig.tight_layout()
    return save(fig, "l04-pr-vs-roc")


def fig_operating(test, order):
    names = [n for n in order]
    flagged = [test[n]["flagged"] for n in names]
    tp = [test[n]["tp"] for n in names]
    fp = [test[n]["fp"] for n in names]
    fn = [test[n]["fn"] for n in names]
    y = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(10.6, 3.7))
    ax.barh(y, tp, color=SUCCESS, height=0.62, label="5s caught")
    ax.barh(y, fp, left=tp, color=ACCENT, height=0.62, label="false alarms")
    ax.barh(y, fn, left=np.array(tp) + np.array(fp), color=SOFT, height=0.62,
            label="5s missed")
    for i, n in enumerate(names):
        over = "" if test[n]["within_capacity"] else " · over capacity"
        ax.text(flagged[i] + fn[i] + 40, i,
                f"{flagged[i]:,} flagged · {100 * test[n]['recall']:.1f}% caught{over}",
                va="center", fontsize=SMALL,
                color=ACCENT if over else MUTED,
                fontweight="bold" if over else "normal")
    ax.axvline(DESK_CAPACITY, color="#16212b", lw=2.5)
    ax.text(DESK_CAPACITY + 30, -0.62, f"desk capacity {DESK_CAPACITY:,}",
            fontsize=SMALL, color="#16212b", fontweight="bold", va="bottom")
    ax.set_yticks(y, [n.replace("_", " ") for n in names], fontsize=TICK)
    ax.invert_yaxis()
    ax.set_xlabel("items in one shift of 10,000 scanned digits")
    ax.set_xlim(0, 2750); ax.set_ylim(3.6, -1.05)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:,.0f}")
    ax.grid(axis="y", alpha=0)
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), ncols=3)
    fig.tight_layout()
    return save(fig, "l04-operating")


# ---------------------------------------------------------------------- main

def main():
    setup()
    load_cache()

    print("Loading MNIST…")
    X, y = load_mnist()
    Xtr_u8, ytr = X[:N_TRAIN], y[:N_TRAIN]
    Xte_u8, yte = X[N_TRAIN:], y[N_TRAIN:]
    Xtr = Xtr_u8.astype(np.float64)
    Xte = Xte_u8.astype(np.float64)
    y5 = (ytr == DIGIT)
    y5te = (yte == DIGIT)

    facts = {
        "n_images": int(len(y)),
        "n_pixels": int(X.shape[1]),
        "n_train": int(len(ytr)),
        "n_test": int(len(yte)),
        "pixel_min": int(X.min()), "pixel_max": int(X.max()),
        "n_fives_total": int((y == DIGIT).sum()),
        "n_fives_train": int(y5.sum()),
        "n_not_five_train": int((~y5).sum()),
        "n_fives_test": int(y5te.sum()),
        "base_rate_train": float(y5.mean()),
        "base_rate_test": float(y5te.mean()),
        "never_fires_accuracy_train": float(1 - y5.mean()),
        "never_fires_accuracy_test": float(1 - y5te.mean()),
        "desk_capacity": DESK_CAPACITY,
        "recall_target": RECALL_TARGET,
        "n_folds": N_FOLDS,
    }

    print("Lecture 3 figures:")
    _, facts["n_fives_in_sample"] = fig_digits(X, y)
    facts["digit_counts_train"] = fig_class_counts(ytr)
    facts["never_fires_accuracy_by_digit"] = fig_baseline(ytr)

    print("Cross-validated accuracy of the classifier they build:")
    cv = cached("l03_cv_accuracy", lambda: measure_cv_accuracy(Xtr, y5))
    facts["cv_accuracy"] = cv
    for k in ("scaled", "raw"):
        print(f"    {k:7s} {cv[k]['mean']:.5f}  folds "
              f"{[round(f, 5) for f in cv[k]['folds']]}")
    fig_folds(cv, facts["never_fires_accuracy_train"])

    print("What evaluating on the training rows is worth, over five seeds:")
    tvc = cached("l03_train_vs_cv", lambda: measure_train_vs_cv(Xtr, y5))
    facts["train_vs_cv"] = tvc
    print(f"    gap {tvc['gap_mean_pp']:.3f} pp ± {tvc['gap_std_pp']:.3f}, "
          f"positive in {tvc['gap_positive']}/{tvc['n_pairs']}")
    fig_train_vs_cv(tvc)

    # ---- Lecture 4 -------------------------------------------------------
    from sklearn.metrics import (average_precision_score, confusion_matrix,
                                 f1_score, precision_recall_curve,
                                 precision_score, recall_score, roc_auc_score,
                                 roc_curve)

    print("Out-of-fold decision scores:")
    scores = cached("l04_cv_scores", lambda: measure_cv_scores(Xtr, y5))
    pred = scores >= 0.0

    tn, fp, fn, tp = confusion_matrix(y5, pred).ravel()
    facts["confusion"] = {"tn": int(tn), "fp": int(fp),
                          "fn": int(fn), "tp": int(tp)}
    facts["headline"] = {
        "accuracy": float((pred == y5).mean()),
        "precision": float(precision_score(y5, pred)),
        "recall": float(recall_score(y5, pred)),
        "f1": float(f1_score(y5, pred)),
        "specificity": float(tn / (tn + fp)),
        "missed_fives_pct": float(100 * fn / (tp + fn)),
    }
    print(f"    accuracy {facts['headline']['accuracy']:.5f}  "
          f"precision {facts['headline']['precision']:.4f}  "
          f"recall {facts['headline']['recall']:.4f}")

    # the identity that makes accuracy readable
    p = facts["base_rate_train"]
    facts["accuracy_identity"] = {
        "base_rate": p,
        "specificity_term": float((1 - p) * facts["headline"]["specificity"]),
        "recall_term": float(p * facts["headline"]["recall"]),
        "sum": float((1 - p) * facts["headline"]["specificity"]
                     + p * facts["headline"]["recall"]),
        "accuracy_per_10pp_recall": float(10 * p),
        # the same identity read as a decomposition of the gap over the
        # never-fires baseline: recall earns points, lost specificity spends them
        "recall_term_pp": float(100 * p * facts["headline"]["recall"]),
        "specificity_drop_pp": float(
            100 * (1 - facts["headline"]["specificity"])),
        "specificity_loss_pp": float(
            100 * (1 - p) * (1 - facts["headline"]["specificity"])),
        "specificity_share_of_accuracy_pct": float(
            100 * (1 - p) * facts["headline"]["specificity"]
            / facts["headline"]["accuracy"]),
    }

    print("Thread 2 — walking the threshold down the ranking:")
    walk = rank_walk(scores, y5)
    facts["threshold_walk"] = {
        k: walk[k] for k in
        ("n_steps", "n_positives", "n_negatives", "n_steps_dropping_a_five",
         "precision_falls_when_threshold_rises",
         "precision_flat_when_threshold_rises",
         "precision_rises_when_threshold_rises", "top10")}
    print(f"    precision falls at "
          f"{walk['precision_falls_when_threshold_rises']:,} of "
          f"{walk['n_steps']:,} steps; there are {walk['n_positives']:,} 5s")
    fig_nonmonotone(walk)

    prec, rec, thr = precision_recall_curve(y5, scores)
    facts["average_precision"] = float(average_precision_score(y5, scores))
    facts["roc_auc"] = float(roc_auc_score(y5, scores))
    facts["operating_points_cv"] = {
        "precision_90": at_precision(prec, rec, thr, 0.90),
        "precision_99": at_precision(prec, rec, thr, 0.99),
        "recall_90": at_recall(prec, rec, thr, 0.90),
        "recall_99": at_recall(prec, rec, thr, 0.99),
    }
    op = facts["operating_points_cv"]["precision_90"]
    print(f"    90% precision at threshold {op['threshold']:.1f} "
          f"→ recall {op['recall']:.4f}")
    fig_pr_curve(prec, rec, thr, op)
    fig_never_fires(p, facts["headline"]["accuracy"])
    fig_accuracy_weights(p, facts["headline"]["specificity"])
    fig_confusion((tn, fp, fn, tp))

    print("A random forest, for comparison:")
    rf_scores = cached("l04_forest_scores", lambda: measure_forest(Xtr, y5))
    rf_prec, rf_rec, rf_thr = precision_recall_curve(y5, rf_scores)
    rf_pred = rf_scores >= 0.5
    rtn, rfp, rfn, rtp = confusion_matrix(y5, rf_pred).ravel()
    facts["forest"] = {
        "roc_auc": float(roc_auc_score(y5, rf_scores)),
        "average_precision": float(average_precision_score(y5, rf_scores)),
        "accuracy": float((rf_pred == y5).mean()),
        "precision": float(precision_score(y5, rf_pred)),
        "recall": float(recall_score(y5, rf_pred)),
        "f1": float(f1_score(y5, rf_pred)),
        "confusion": {"tn": int(rtn), "fp": int(rfp),
                      "fn": int(rfn), "tp": int(rtp)},
        "recall_at_90_precision": float(rf_rec[int((rf_prec >= 0.90).argmax())]),
        "operating_points_cv": {
            "recall_90": at_recall(rf_prec, rf_rec, rf_thr, 0.90),
            "precision_99": at_precision(rf_prec, rf_rec, rf_thr, 0.99),
        },
    }
    print(f"    forest AUC {facts['forest']['roc_auc']:.4f}  "
          f"AP {facts['forest']['average_precision']:.4f}")

    fpr, tpr, _ = roc_curve(y5, scores)
    rfpr, rtpr, _ = roc_curve(y5, rf_scores)
    sub = slice(None, None, max(1, len(fpr) // 3000))
    rsub = slice(None, None, max(1, len(rfpr) // 3000))
    fig_roc({"fpr": fpr[sub], "tpr": tpr[sub], "auc": facts["roc_auc"]},
            {"fpr": rfpr[rsub], "tpr": rtpr[rsub],
             "auc": facts["forest"]["roc_auc"]})

    print("The same classifier at three base rates:")
    rng = np.random.default_rng(SEED)
    pos = np.where(y5)[0]
    neg = np.where(~y5)[0]
    rarity, rarity_facts = [], []
    for target in (facts["base_rate_train"], 0.02, 0.01):
        k = int(round(len(neg) * target / (1 - target)))
        keep = np.concatenate([neg, rng.choice(pos, k, replace=False)])
        yk, sk = y5[keep], scores[keep]
        f_, t_, _ = roc_curve(yk, sk)
        pk, rk, _ = precision_recall_curve(yk, sk)
        s2 = slice(None, None, max(1, len(f_) // 2000))
        s3 = slice(None, None, max(1, len(pk) // 2000))
        row = {"base_rate": float(yk.mean()), "n_positives": int(k),
               "auc": float(roc_auc_score(yk, sk)),
               "ap": float(average_precision_score(yk, sk))}
        rarity.append(dict(row, fpr=f_[s2], tpr=t_[s2], prec=pk[s3], rec=rk[s3]))
        rarity_facts.append(row)
        print(f"    base rate {row['base_rate']:.4f}  AUC {row['auc']:.4f}  "
              f"AP {row['ap']:.4f}")
    facts["rarity"] = rarity_facts
    fig_pr_vs_roc(rarity)

    print("The test shift — fitted once, scored once:")
    thresholds = {
        "default_threshold": ("sgd", 0.0),
        "tuned_for_90_precision": ("sgd", op["threshold"]),
        "tuned_for_90_recall": ("sgd",
                                facts["operating_points_cv"]["recall_90"]["threshold"]),
        "assistant_max_recall": ("sgd",
                                 facts["operating_points_cv"]["recall_99"]["threshold"]),
        "forest_default": ("forest", 0.5),
        "forest_tuned_for_90_recall": (
            "forest", facts["forest"]["operating_points_cv"]["recall_90"]["threshold"]),
    }
    test = cached("l04_test_shift",
                  lambda: measure_test(Xtr, y5, Xte, y5te, thresholds))
    facts["test_shift"] = test
    facts["sgd_train_accuracy"] = test["sgd_train_accuracy"]
    for n in thresholds:
        t = test[n]
        print(f"    {n:28s} flagged {t['flagged']:5,}  recall {t['recall']:.4f}  "
              f"precision {t['precision']:.4f}  acc {t['accuracy']:.4f}")
    fig_operating(test, ["default_threshold", "tuned_for_90_precision",
                         "tuned_for_90_recall", "forest_tuned_for_90_recall"])

    # Differences the slides quote in their own right.
    h = facts["headline"]
    facts["gaps"] = {
        "accuracy_over_never_fires_pp": float(
            100 * (h["accuracy"] - facts["never_fires_accuracy_train"])),
        "missed_fives": int(fn),
        "false_alarms": int(fp),
        "misclassified": int(fn + fp),
        "misclassified_pct": float(100 * (fn + fp) / len(y5)),
        "n_flagged": int(tp + fp),
        "forest_minus_sgd_recall_at_90_precision_pp": float(
            100 * (facts["forest"]["recall_at_90_precision"] - op["recall"])),
        "ap_drop_1pct_base_rate": float(
            facts["rarity"][0]["ap"] - facts["rarity"][-1]["ap"]),
        "auc_drop_1pct_base_rate": float(
            facts["rarity"][0]["auc"] - facts["rarity"][-1]["auc"]),
        "flagged_over_capacity_at_90_recall": int(
            test["tuned_for_90_recall"]["flagged"] - DESK_CAPACITY),
        "assistant_recall_gain_pp": float(
            100 * (test["assistant_max_recall"]["recall"]
                   - test["default_threshold"]["recall"])),
        "assistant_accuracy_loss_pp": float(
            100 * (test["default_threshold"]["accuracy"]
                   - test["assistant_max_recall"]["accuracy"])),
        "assistant_extra_false_alarms": int(
            test["assistant_max_recall"]["fp"] - test["default_threshold"]["fp"]),
        "below_perfect_pp": float(100 * (1 - h["accuracy"])),
        "anchor_errors_removed_pct": float(
            100 * (h["accuracy"] - facts["never_fires_accuracy_train"])
            / facts["base_rate_train"]),
        "forest_minus_sgd_accuracy_pp": float(
            100 * (facts["forest"]["accuracy"] - h["accuracy"])),
        "forest_minus_sgd_recall_pp": float(
            100 * (facts["forest"]["recall"] - h["recall"])),
        "recall_cost_of_90_precision_pp": float(100 * (h["recall"] - op["recall"])),
    }

    if (problems := check_text_floor()):
        print("\nfigures whose text lands under the on-slide floor:")
        for p_ in problems:
            print("  " + p_)
        raise SystemExit(1)

    # Everything under one key. figures.json is shared and figkit.export()
    # merges at the top level only, so a bare "n_train" here silently replaces
    # Application 1's — which is exactly what happened, and check_provenance.py
    # is what caught it. check_provenance flattens to any depth, so nesting
    # costs nothing.
    export(app02=facts)
    print("\nApplication 2 done.")


if __name__ == "__main__":
    main()
