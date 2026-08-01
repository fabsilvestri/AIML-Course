#!/usr/bin/env python3
"""
Application 5 — grouping faces with almost no labels. Lectures 9 and 10.

    python3 tools/figures_app05.py

Every quantity printed in slides/lecture-09.html and slides/lecture-10.html is
produced here and written to assets/figures/figures.json through
`figkit.export()`, which merges rather than overwrites so that twenty-four
lectures can own twenty-four scripts.

The dataset is `sklearn.datasets.fetch_olivetti_faces`: 400 photographs of 40
people, 64x64 greyscale, 4,096 features, about 6.5 MB in memory. It is small
enough that nothing here needs a GPU and large enough that the thing Lecture 9
ends on — a model-selection sweep over k that takes a quarter of an hour — is
real rather than staged.

Read TRICKS section 6 and section 11.6 before adding a figure. Two rules that
are not obvious:

  * a grid of face images is a PNG. As SVG it embeds one <image> per face and
    the browser decodes forty base64 blobs on every slide transition.
  * matplotlib sizes are POINTS; check_text_floor() refuses to ship a figure
    whose smallest label lands under 15px on the slide.

**On the timings.** Wall-clock is a property of a machine, not of an algorithm,
and this one was measured with the thread count pinned below so that a run
competing with other work still produces comparable numbers. Every timing on
the slides is quoted as a ratio against another timing measured in the same
run; the absolute seconds are given for scale and labelled as such.
"""

from __future__ import annotations

import os

# OpenMP reads these when the library loads, so they must be set before numpy,
# scipy or scikit-learn are imported. Two threads: enough that the numbers are
# not silly, few enough that they are the same on a busy machine.
N_THREADS = "2"
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, N_THREADS)

import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parent))

from figkit import (ACCENT, BODY, MATH, MUTED, PRIMARY, RULE, SMALL, SUCCESS,
                    cached, check_text_floor, export, load_cache, save, setup)

SEED = 42
N_SEEDS = 20              # never quote a single-seed number — TRICKS section 4
LABEL_BUDGET = 40         # the annotator's budget: 40 of the 400 photographs
K_MAX = 60                # the sweep Lecture 9 runs, and Lecture 10 repairs
N_INIT = 10


# --------------------------------------------------------------- housekeeping

def jsonable(obj):
    """numpy scalars and arrays are not JSON; figures.json has to be."""
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [jsonable(v) for v in obj.tolist()]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    return obj


# --------------------------------------------------------------------- data

def load_faces():
    """400 x 4096, float32 in [0, 1], plus the 40 identities we mostly hide."""
    from sklearn.datasets import fetch_olivetti_faces
    d = fetch_olivetti_faces(shuffle=False)
    return d.data, d.target, d.images


def split(X, y, seed=SEED):
    """Stratified by identity: 7 photographs of each person train, 3 test.

    Clustering needs no split — there is no held-out quantity to protect. Label
    propagation, the reconstruction error of an unseen face, and the leak
    Lecture 10 measures all do.
    """
    from sklearn.model_selection import train_test_split
    return train_test_split(X, y, test_size=0.3, stratify=y, random_state=seed)


def audit_index(rng):
    """Which 40 of the 400 photographs the annotator was paid to label."""
    return np.sort(rng.choice(400, size=LABEL_BUDGET, replace=False))


# ------------------------------------------------- face montages (helpers)

def montage(ax, images, ncol, *, gap=2):
    """Tile square images into one array so the slide gets one <image>, not n."""
    images = np.asarray(images)
    n, h, w = images.shape
    nrow = int(np.ceil(n / ncol))
    canvas = np.ones((nrow * (h + gap) - gap, ncol * (w + gap) - gap))
    for i, im in enumerate(images):
        r, c = divmod(i, ncol)
        canvas[r * (h + gap):r * (h + gap) + h,
               c * (w + gap):c * (w + gap) + w] = im
    ax.imshow(canvas, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    return nrow


def box_cell(ax, i, ncol, h=64, w=64, gap=2, color=ACCENT, lw=2.4):
    r, c = divmod(i, ncol)
    ax.add_patch(Rectangle((c * (w + gap) - 0.5, r * (h + gap) - 0.5), w, h,
                           fill=False, edgecolor=color, linewidth=lw))


def callout(ax, text, xy, xytext, colour):
    ax.annotate(text, xy=xy, xytext=xytext, textcoords="axes fraction",
                fontsize=SMALL, color=colour,
                bbox=dict(boxstyle="round,pad=0.5", fc="white", ec=RULE),
                arrowprops=dict(arrowstyle="->", color=colour, lw=2))


def loglabels(ax, which="x", ticks=None):
    """Plain decimal labels on a log axis.

    `setup()` sets text.parse_math=False course-wide — TRICKS section 9.1 —
    which is right for every string we write and wrong for the one matplotlib
    writes itself: the default LogFormatter emits "$\\mathdefault{10^{2}}$" and
    with mathtext off that appears on the slide verbatim. Every log axis in
    this file therefore gets its ticks named explicitly.
    """
    from matplotlib.ticker import FuncFormatter, NullFormatter
    axis = ax.xaxis if which == "x" else ax.yaxis
    if ticks is not None:
        (ax.set_xticks if which == "x" else ax.set_yticks)(ticks)
    fmt = FuncFormatter(lambda v, _: (f"{v:,.0f}" if v >= 1 else
                                      f"{v:g}".lstrip("0") or "0"))
    axis.set_major_formatter(fmt)
    axis.set_minor_formatter(NullFormatter())


# =========================================================== LECTURE 9 ======

def fig_corpus(images, y):
    first = np.array([images[y == p][0] for p in range(40)])
    fig, ax = plt.subplots(figsize=(10.0, 4.4))
    montage(ax, first, ncol=10)
    ax.set_title("One photograph of each of the 40 people. The corpus holds "
                 "400, and none of them arrives labelled")
    save(fig, "l09-corpus", raster=True)


def fig_one_person(images, y):
    rows = np.concatenate([images[y == 0], images[y == 22]])
    fig, ax = plt.subplots(figsize=(10.0, 2.6))
    montage(ax, rows, ncol=10)
    ax.set_title("Ten photographs of one person, then ten of another: glasses "
                 "on and off, lighting, expression, head angle")
    save(fig, "l09-one-person", raster=True)


def distance_structure(X, y):
    from sklearn.metrics import pairwise_distances
    D = pairwise_distances(X)
    iu = np.triu_indices(len(X), k=1)
    same = y[iu[0]] == y[iu[1]]
    within, between = D[iu][same], D[iu][~same]
    return {"within_mean": float(within.mean()),
            "within_sd": float(within.std()),
            "between_mean": float(between.mean()),
            "between_sd": float(between.std()),
            "within_median": float(np.median(within)),
            "between_median": float(np.median(between)),
            "n_within": int(same.sum()), "n_between": int((~same).sum()),
            "overlap": float((within > np.median(between)).mean()),
            "_within": within, "_between": between}


def fig_distances(ds):
    fig, ax = plt.subplots(figsize=(9.6, 3.4))
    hi = max(ds["_between"].max(), ds["_within"].max())
    bins = np.linspace(0, hi, 70)
    ax.hist(ds["_between"], bins=bins, density=True, color=RULE,
            label=f"different people ({ds['n_between']:,} pairs)")
    ax.hist(ds["_within"], bins=bins, density=True, color=ACCENT, alpha=0.72,
            label=f"same person ({ds['n_within']:,} pairs)")
    ax.axvline(ds["between_median"], color=PRIMARY, lw=2.2, ls="--")
    ax.set_xlabel("Euclidean distance between two faces, raw pixels")
    ax.set_ylabel("density")
    ax.set_title("Distance carries identity, and not cleanly")
    ax.legend(loc="upper right")
    callout(ax, f"{100 * ds['overlap']:.0f}% of same-person pairs\n"
                f"lie right of the dashed median",
            (ds["between_median"], ax.get_ylim()[1] * 0.28), (0.03, 0.42),
            PRIMARY)
    fig.tight_layout()
    save(fig, "l09-distances")


def kmeans_sweep(X, y, audit, k_max=K_MAX):
    """The sweep Lecture 9 runs: k-means at every k, inertia and silhouette.

    This is the slow thing. n_init=10 means every k is ten independent runs of
    Lloyd's algorithm over 400 x 4,096 floats, and every silhouette needs the
    full 400 x 400 distance matrix in 4,096 dimensions.
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score, silhouette_score

    ks = list(range(2, k_max + 1))
    inertia, sil, ari_audit, ari_full, labels = [], [], [], [], {}
    t_fit = t_sil = 0.0
    for k in ks:
        t0 = time.perf_counter()
        km = KMeans(n_clusters=k, n_init=N_INIT, random_state=SEED).fit(X)
        t_fit += time.perf_counter() - t0
        t0 = time.perf_counter()
        s = silhouette_score(X, km.labels_)
        t_sil += time.perf_counter() - t0
        inertia.append(float(km.inertia_))
        sil.append(float(s))
        ari_audit.append(float(adjusted_rand_score(y[audit], km.labels_[audit])))
        ari_full.append(float(adjusted_rand_score(y, km.labels_)))
        if k in (2, 10, 40, 60):
            labels[k] = km.labels_.copy()
        print(f"      k={k:3d}  inertia={km.inertia_:9.1f}  sil={s:+.4f}",
              flush=True)
    return {"ks": ks, "inertia": inertia, "silhouette": sil,
            "ari_audit": ari_audit, "ari_full": ari_full,
            "t_fit": t_fit, "t_sil": t_sil, "t_total": t_fit + t_sil,
            "labels": labels}


def random_baseline(X, y, audit, ks=(2, 10, 40, 60), n_seeds=N_SEEDS):
    """The trivial anchor: put every face in a uniformly random cluster."""
    from sklearn.metrics import adjusted_rand_score, silhouette_score
    out = {}
    for k in ks:
        s, a = [], []
        for seed in range(n_seeds):
            rng = np.random.default_rng(1000 + seed)
            lab = rng.integers(0, k, size=len(X))
            while len(np.unique(lab)) < 2:
                lab = rng.integers(0, k, size=len(X))
            s.append(float(silhouette_score(X, lab)))
            a.append(float(adjusted_rand_score(y[audit], lab[audit])))
        out[k] = {"sil_mean": float(np.mean(s)), "sil_sd": float(np.std(s)),
                  "sil_max": float(np.max(s)),
                  "ari_mean": float(np.mean(a)), "ari_sd": float(np.std(a)),
                  "ari_max": float(np.max(a))}
    return out


def elbow_evidence(sweep):
    """Quantify how badly the elbow rule does here. Not a rhetorical claim."""
    ks = np.array(sweep["ks"], float)
    inertia = np.array(sweep["inertia"], float)
    # the kneedle construction: the point furthest below the chord joining the
    # two ends of the curve, after rescaling both axes to [0, 1]
    kn = (ks - ks.min()) / (ks.max() - ks.min())
    inn = (inertia - inertia.min()) / (inertia.max() - inertia.min())
    drop = (1 - kn) - inn
    i40 = list(sweep["ks"]).index(40)
    d1 = -np.diff(inertia)
    return {"k_knee": int(ks[int(np.argmax(drop))]),
            "knee_prominence": float(drop.max()),
            "inertia_k2": float(inertia[0]),
            "inertia_k40": float(inertia[i40]),
            "inertia_kmax": float(inertia[-1]),
            "frac_remaining_at_40": float(inertia[i40] / inertia[0]),
            "step_at_10": float(d1[8]), "step_at_40": float(d1[i40]),
            "step_ratio_10_40": float(d1[8] / d1[i40])}


def fig_elbow(sweep, ev):
    fig, ax = plt.subplots(figsize=(9.6, 3.4))
    ax.plot(sweep["ks"], sweep["inertia"], color=PRIMARY, lw=2.6)
    ax.axvline(40, color=SUCCESS, lw=2.2, ls="--")
    i40 = list(sweep["ks"]).index(40)
    callout(ax, "k = 40, the number of people —\nwhich we are not supposed to know",
            (40, sweep["inertia"][i40]), (0.46, 0.68), SUCCESS)
    callout(ax, f"the sharpest bend the kneedle rule finds\nis k = {ev['k_knee']}, "
                f"and it is barely a bend",
            (ev["k_knee"], sweep["inertia"][ev["k_knee"] - 2]), (0.12, 0.26),
            ACCENT)
    ax.set_xlabel("k, number of clusters")
    ax.set_ylabel("inertia")
    ax.set_title("Inertia falls smoothly and never stops falling")
    fig.tight_layout()
    save(fig, "l09-elbow")


def fig_silhouette_vs_k(sweep, base):
    fig, ax = plt.subplots(figsize=(9.6, 3.5))
    ks, sil = sweep["ks"], sweep["silhouette"]
    ax.plot(ks, sil, color=PRIMARY, lw=2.6, label="k-means on raw pixels")
    best_i = int(np.argmax(sil))
    best = int(ks[best_i])
    ax.plot([best], [sil[best_i]], "o", color=ACCENT, ms=11, zorder=5)
    xs = sorted(base)
    ax.plot(xs, [base[k]["sil_mean"] for k in xs], color=MUTED, lw=2.2, ls=":",
            marker="s", ms=7, label=f"random assignment, {N_SEEDS} seeds")
    ax.axvline(40, color=SUCCESS, lw=2.2, ls="--")
    ax.set_xlim(0, max(ks) + 10)
    callout(ax, f"best {sil[best_i]:.3f} at k = {best} — and\nstill rising at "
                f"the edge of the grid",
            (best, sil[best_i]), (0.30, 0.60), ACCENT)
    ax.annotate("k = 40", xy=(40, min(sil)), xytext=(41.5, min(sil) - 0.005),
                fontsize=SMALL, color=SUCCESS, va="bottom")
    ax.set_xlabel("k, number of clusters")
    ax.set_ylabel("mean silhouette")
    ax.set_title("The winner sits on the edge of the range we searched")
    ax.legend(loc="lower left")
    fig.tight_layout()
    save(fig, "l09-silhouette-vs-k")


def fig_silhouette_diagram(X, sweep):
    from sklearn.metrics import silhouette_samples
    shown = [2, 10, 40, 60]
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.0), sharex=True)
    lo, hi = 0.0, 0.0
    for ax, k in zip(axes.ravel(), shown):
        lab = sweep["labels"][k]
        vals = silhouette_samples(X, lab)
        lo, hi = min(lo, vals.min()), max(hi, vals.max())
        pos = 0
        order = sorted(range(k), key=lambda c: -vals[lab == c].mean())
        cmap = plt.get_cmap("viridis")
        for j, c in enumerate(order):
            v = np.sort(vals[lab == c])
            ax.fill_betweenx(np.arange(pos, pos + len(v)), 0, v,
                             color=cmap(j / max(k - 1, 1)), lw=0)
            pos += len(v) + 3
        ax.axvline(vals.mean(), color=ACCENT, lw=2.2, ls="--")
        ax.set_title(f"k = {k}    mean {vals.mean():.3f}", fontsize=BODY)
        ax.set_yticks([]); ax.grid(False)
    for ax in axes.ravel():
        ax.set_xlim(lo - 0.03, hi + 0.03)
    for ax in axes[1]:
        ax.set_xlabel("silhouette coefficient")
    fig.tight_layout()
    save(fig, "l09-silhouette-diagram")


def cluster_quality(labels, y):
    out = []
    for c in np.unique(labels):
        members = np.where(labels == c)[0]
        _, counts = np.unique(y[members], return_counts=True)
        out.append({"cluster": int(c), "size": int(len(members)),
                    "n_identities": int(len(counts)),
                    "purity": float(counts.max() / len(members)),
                    "members": members})
    return out


def fig_clusters(images, y, labels, k):
    """The point of using a face dataset: look at the faces in a cluster."""
    q = cluster_quality(labels, y)
    big = [c for c in q if c["size"] >= 5]
    big.sort(key=lambda c: (-c["purity"], -c["size"]))
    best, worst = big[0], big[-1]

    fig, axes = plt.subplots(2, 1, figsize=(10.0, 3.2))
    for ax, c, tag, colour in ((axes[0], best, "cleanest", SUCCESS),
                               (axes[1], worst, "worst", ACCENT)):
        ims = images[c["members"]][:10]
        montage(ax, ims, ncol=10)
        plural = "y" if c["n_identities"] == 1 else "ies"
        ax.set_title(f"{tag} cluster at k = {k}: {c['size']} photographs, "
                     f"{c['n_identities']} identit{plural}, "
                     f"purity {c['purity']:.2f}",
                     fontsize=SMALL, color=colour)
        if tag == "worst":
            majority = np.bincount(y[c["members"]]).argmax()
            for i, idx in enumerate(c["members"][:10]):
                if y[idx] != majority:
                    box_cell(ax, i, 10)
    fig.tight_layout()
    save(fig, "l09-clusters", raster=True)
    purities = [c["purity"] for c in q]
    return {"best_purity": best["purity"], "best_size": best["size"],
            "worst_purity": worst["purity"], "worst_size": worst["size"],
            "worst_identities": worst["n_identities"],
            "mean_purity": float(np.mean(purities)),
            "n_pure": int(sum(c["n_identities"] == 1 for c in q)),
            "n_singleton": int(sum(c["size"] == 1 for c in q)),
            "k": int(k)}


def fig_cluster_sizes(labels, y, k):
    q = cluster_quality(labels, y)
    sizes = np.array([c["size"] for c in q])
    ids = np.array([c["n_identities"] for c in q])
    order = np.argsort(-sizes)
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 3.2))
    axes[0].bar(range(len(sizes)), sizes[order], color=PRIMARY, width=0.85)
    axes[0].axhline(10, color=SUCCESS, lw=2.2, ls="--")
    callout(axes[0], "10 = the true size of every group",
            (len(sizes) * 0.45, 10), (0.26, 0.62), SUCCESS)
    axes[0].set_xlabel(f"cluster, sorted by size (k = {k})")
    axes[0].set_ylabel("photographs")
    axes[0].set_title("Sizes are not 10")
    vals, counts = np.unique(ids, return_counts=True)
    axes[1].bar(vals, counts, color=ACCENT, width=0.7)
    axes[1].set_xlabel("distinct people inside one cluster")
    axes[1].set_ylabel("clusters")
    axes[1].set_title("Few clusters hold one person")
    axes[1].set_xticks(vals)
    fig.tight_layout()
    save(fig, "l09-cluster-sizes")
    return {"largest": int(sizes.max()), "smallest": int(sizes.min()),
            "n_size_10": int((sizes == 10).sum())}


def fig_timing_l09(sweep, k_max=K_MAX):
    fig, ax = plt.subplots(figsize=(9.6, 2.6))
    left = 0.0
    for name, val, colour in (("k-means fits", sweep["t_fit"], PRIMARY),
                              ("silhouette scores", sweep["t_sil"], MATH)):
        ax.barh([0], [val], left=left, color=colour, height=0.5, label=name)
        ax.text(left + val / 2, 0, f"{val:.0f}s", ha="center", va="center",
                color="white", fontsize=BODY)
        left += val
    ax.set_yticks([]); ax.grid(False)
    ax.set_xlim(0, left * 1.02)
    ax.set_xlabel("wall-clock seconds, two threads")
    ax.set_title(f"One sweep over k = 2 to {k_max}, n_init = {N_INIT}, "
                 f"on 400 faces")
    ax.legend(loc="lower right", ncol=2)
    # Below two minutes, say seconds: rounding to whole minutes printed
    # "1 minutes" on the slide, and "2" would round an 89-second sweep up.
    spent = (f"{left:.0f} seconds" if left < 120 else f"{left / 60:.0f} minutes")
    ax.text(left * 0.5, 0.42, f"{spent} — and the answer it "
                              f"gives is still ambiguous",
            ha="center", fontsize=SMALL, color=ACCENT,
            bbox=dict(boxstyle="round,pad=0.5", fc="white", ec=RULE))
    fig.tight_layout()
    save(fig, "l09-timing")


def selection_optimism(X, n_seeds=N_SEEDS):
    """Lecture 9's assistant failure, measured.

    The prompt asks for "the best k by silhouette" and the code returns the
    silhouette at that k. That number is a maximum over a noisy criterion
    evaluated on the same faces it was chosen with. Split the corpus, choose on
    one half, score on the other, and the difference is the optimism.
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    rows = []
    for seed in range(n_seeds):
        rng = np.random.default_rng(2000 + seed)
        perm = rng.permutation(len(X))
        a, b = perm[:200], perm[200:]
        best = (-2.0, None, None)
        for k in (5, 10, 20, 40, 60):
            km = KMeans(n_clusters=k, n_init=3, random_state=SEED).fit(X[a])
            s = silhouette_score(X[a], km.labels_)
            if s > best[0]:
                best = (float(s), k, km)
        held = float(silhouette_score(X[b], best[2].predict(X[b])))
        rows.append((best[1], best[0], held))
    ks, chosen, held = (np.array(v) for v in zip(*rows))
    return {"k_mode": int(np.bincount(ks).argmax()),
            "selected_mean": float(chosen.mean()),
            "selected_sd": float(chosen.std()),
            "heldout_mean": float(held.mean()),
            "heldout_sd": float(held.std()),
            "optimism_mean": float((chosen - held).mean()),
            "optimism_sd": float((chosen - held).std()),
            "optimism_pct": float((100 * (chosen - held) / chosen).mean()),
            "n_worse": int((held < chosen).sum()),
            "n_seeds": n_seeds}


# ========================================================== LECTURE 10 ======

def fig_eigenfaces(X):
    from sklearn.decomposition import PCA
    pca = PCA(n_components=15, random_state=SEED).fit(X)
    tiles = [X.mean(axis=0).reshape(64, 64)]
    for comp in pca.components_:
        c = comp.reshape(64, 64)
        tiles.append((c - c.min()) / (c.max() - c.min()))
    fig, ax = plt.subplots(figsize=(10.0, 2.9))
    montage(ax, np.array(tiles), ncol=8)
    ax.set_title("The mean face, then the first 15 principal components. "
                 "Each component is itself an image")
    save(fig, "l10-eigenfaces", raster=True)


def pca_curves(X):
    from sklearn.decomposition import PCA
    pca = PCA(random_state=SEED).fit(X)
    evr = pca.explained_variance_ratio_
    cum = np.cumsum(evr)
    return {"evr": evr.tolist(), "cum": cum.tolist(),
            "d95": int(np.searchsorted(cum, 0.95) + 1),
            "d99": int(np.searchsorted(cum, 0.99) + 1),
            "d90": int(np.searchsorted(cum, 0.90) + 1),
            "n_components_max": int(len(evr)),
            "evr_1": float(evr[0]), "evr_2": float(evr[1]),
            "evr_top10": float(cum[9]), "evr_top50": float(cum[49])}


def fig_scree(pc):
    evr = np.array(pc["evr"]); cum = np.array(pc["cum"])
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 3.2))
    axes[0].plot(range(1, 51), evr[:50] * 100, color=PRIMARY, lw=2.4,
                 marker="o", ms=4)
    axes[0].set_xlabel("component")
    axes[0].set_ylabel("variance explained, %")
    axes[0].set_title("The first few carry most of it")
    callout(axes[0], f"component 1 alone: {100 * pc['evr_1']:.1f}%",
            (1, 100 * pc["evr_1"]), (0.30, 0.66), ACCENT)
    axes[1].plot(range(1, len(cum) + 1), cum * 100, color=PRIMARY, lw=2.4)
    axes[1].axhline(95, color=SUCCESS, lw=2.2, ls="--")
    axes[1].axvline(pc["d95"], color=SUCCESS, lw=2.2, ls=":")
    callout(axes[1], f"95% of the variance in {pc['d95']} components",
            (pc["d95"], 95), (0.26, 0.30), SUCCESS)
    axes[1].set_xlabel("components kept")
    axes[1].set_ylabel("cumulative, %")
    axes[1].set_title("How many you actually need")
    fig.tight_layout()
    save(fig, "l10-scree")


def fig_reconstruction(X_tr, X_te, ds=(1, 2, 5, 10, 25, 50, 100, 200)):
    from sklearn.decomposition import PCA
    face = X_te[0]
    tiles, labels = [], []
    for d in ds:
        p = PCA(n_components=d, random_state=SEED).fit(X_tr)
        tiles.append(p.inverse_transform(p.transform(face[None]))[0].reshape(64, 64))
        labels.append(str(d))
    tiles.append(face.reshape(64, 64)); labels.append("4096")
    fig, ax = plt.subplots(figsize=(10.0, 1.85))
    montage(ax, np.array(tiles), ncol=9)
    for i, lab in enumerate(labels):
        ax.text(i * 66 + 32, 70, lab, ha="center", va="top", fontsize=SMALL,
                color=ACCENT if lab == "4096" else PRIMARY)
    ax.set_ylim(88, -6)
    ax.set_title("One held-out face, rebuilt from this many components",
                 fontsize=BODY)
    save(fig, "l10-reconstruction", raster=True)


def reduction_quality(X_tr, X_te, ds):
    """Reconstruction error on faces the reducer has never seen."""
    from sklearn.decomposition import PCA
    from sklearn.random_projection import GaussianRandomProjection
    pca_err, rp_err = [], []
    for d in ds:
        p = PCA(n_components=d, random_state=SEED).fit(X_tr)
        rec = p.inverse_transform(p.transform(X_te))
        pca_err.append(float(((rec - X_te) ** 2).mean()))
        errs = []
        for seed in range(5):
            g = GaussianRandomProjection(n_components=d, random_state=seed)
            g.fit(X_tr)
            back = np.linalg.pinv(g.components_.T)   # least-squares inverse
            errs.append(float(((g.transform(X_te) @ back - X_te) ** 2).mean()))
        rp_err.append(float(np.mean(errs)))
    return {"ds": [int(d) for d in ds], "pca": pca_err, "rp": rp_err,
            "test_var": float(X_te.var())}


def fig_recon_error(rq, d95):
    fig, ax = plt.subplots(figsize=(9.6, 3.3))
    ax.plot(rq["ds"], rq["pca"], color=PRIMARY, lw=2.6, marker="o", ms=6,
            label="PCA — the optimal linear subspace")
    ax.plot(rq["ds"], rq["rp"], color=ACCENT, lw=2.6, marker="s", ms=6,
            ls="--", label="Gaussian random projection")
    ax.set_xscale("log"); ax.set_yscale("log")
    loglabels(ax, "x", ticks=[2, 5, 10, 25, 50, 100, d95, 279])
    loglabels(ax, "y", ticks=[0.001, 0.01, 0.1, 1.0])
    ax.set_xlabel("components kept, d")
    ax.set_ylabel("squared reconstruction error,\nheld-out faces")
    ax.set_title("PCA minimises exactly this quantity; a random subspace does not")
    ax.legend(loc="center left")
    i = rq["ds"].index(d95)
    callout(ax, f"at d = {d95} the random subspace\n"
                f"costs {rq['rp'][i] / rq['pca'][i]:.0f} times the error",
            (rq["ds"][i], rq["rp"][i]), (0.30, 0.22), ACCENT)
    fig.tight_layout()
    save(fig, "l10-recon-error")


def jl_numbers():
    from sklearn.random_projection import johnson_lindenstrauss_min_dim
    eps = [0.1, 0.2, 0.3, 0.5]
    return {"eps": eps,
            "n400": {str(e): int(johnson_lindenstrauss_min_dim(400, eps=e))
                     for e in eps},
            "n1m": {str(e): int(johnson_lindenstrauss_min_dim(1_000_000, eps=e))
                    for e in eps},
            "d_ambient": 4096}


def fig_jl(jl):
    from sklearn.random_projection import johnson_lindenstrauss_min_dim
    ns = np.unique(np.logspace(2, 7, 60).astype(int))
    fig, ax = plt.subplots(figsize=(9.6, 3.4))
    for e, colour in zip([0.1, 0.2, 0.3], [MATH, PRIMARY, SUCCESS]):
        ax.plot(ns, johnson_lindenstrauss_min_dim(ns, eps=e), lw=2.6,
                color=colour, label=f"tolerance {e:g}")
    ax.axhline(4096, color=ACCENT, lw=2.4, ls="--")
    ax.axvline(400, color=MUTED, lw=1.6, ls=":")
    ax.set_xscale("log")
    loglabels(ax, "x", ticks=[100, 400, 1_000, 10_000, 100_000, 1_000_000,
                              10_000_000])
    ax.tick_params(axis="x", labelrotation=30)
    ax.set_xlabel("n, the number of points")
    ax.set_ylabel("dimensions the bound requires")
    ax.set_title("The dimension you start in appears on neither axis")
    ax.legend(loc="upper left")
    callout(ax, "4096 — the dimension we have,\nwhich the bound never asks about",
            (3e5, 4096), (0.36, 0.56), ACCENT)
    callout(ax, f"our 400 faces at tolerance 0.2:\n{jl['n400']['0.2']:,} dimensions",
            (400, jl["n400"]["0.2"]), (0.05, 0.24), PRIMARY)
    fig.tight_layout()
    save(fig, "l10-jl")


def jl_measured(X, ds=(50, 100, 200, 400, 800, 1382, 1600), n_seeds=5):
    """What a random projection really does to all 79,800 pairwise distances.

    The lemma bounds the distortion of **squared** distances, so that is what is
    measured here. Comparing a distance ratio against the same epsilon would
    understate the distortion by roughly a factor of two, which on a slide
    beside the theorem would be a lie by a factor of two.
    """
    from sklearn.metrics import pairwise_distances
    from sklearn.random_projection import GaussianRandomProjection
    iu = np.triu_indices(len(X), k=1)
    D0 = pairwise_distances(X)[iu] ** 2
    out = {"ds": [int(d) for d in ds], "max": [], "p95": [], "mean": [],
           "n_pairs": int(len(D0))}
    for d in ds:
        m, p, a = [], [], []
        for seed in range(n_seeds):
            g = GaussianRandomProjection(n_components=d, random_state=seed)
            D1 = pairwise_distances(g.fit_transform(X))[iu] ** 2
            dist = np.abs(D1 / D0 - 1)
            m.append(float(dist.max()))
            p.append(float(np.quantile(dist, 0.95)))
            a.append(float(dist.mean()))
        out["max"].append(float(np.mean(m)))
        out["p95"].append(float(np.mean(p)))
        out["mean"].append(float(np.mean(a)))
    # the smallest d in the grid at which every pair is already inside 0.2
    under = [d for d, v in zip(out["ds"], out["max"]) if v <= 0.2]
    out["d_under_02"] = int(min(under)) if under else -1
    out["max_at_1382"] = out["max"][out["ds"].index(1382)]
    return out


def fig_jl_measured(jm, jl):
    fig, ax = plt.subplots(figsize=(9.6, 3.3))
    ax.plot(jm["ds"], jm["max"], color=ACCENT, lw=2.6, marker="o", ms=6,
            label="worst pair")
    ax.plot(jm["ds"], jm["p95"], color=PRIMARY, lw=2.6, marker="s", ms=6,
            label="95th percentile")
    ax.plot(jm["ds"], jm["mean"], color=SUCCESS, lw=2.6, marker="^", ms=6,
            label="mean")
    ax.axhline(0.2, color=MATH, lw=2.2, ls="--")
    ax.axvline(jl["n400"]["0.2"], color=MATH, lw=2.2, ls=":")
    ax.set_xscale("log")
    loglabels(ax, "x", ticks=jm["ds"])
    ax.tick_params(axis="x", labelrotation=30)
    ax.set_xlabel("projected dimension, d")
    ax.set_ylabel("distortion of a\nsquared distance")
    ax.set_title(f"Measured on all {jm['n_pairs']:,} pairs of faces")
    ax.legend(loc="upper right")
    i = jm["ds"].index(jl["n400"]["0.2"])
    callout(ax, f"at the bound's own d = {jl['n400']['0.2']:,}, the worst\n"
                f"of the {jm['n_pairs']:,} pairs is {jm['max'][i]:.2f} — inside "
                f"the 0.2\nit promises, and not by much",
            (jl["n400"]["0.2"], jm["max"][i]), (0.22, 0.42), MATH)
    fig.tight_layout()
    save(fig, "l10-jl-measured")


def reducer_bench(X_tr, X_te, d):
    from sklearn.decomposition import PCA, IncrementalPCA
    from sklearn.random_projection import GaussianRandomProjection

    def timeit(make, repeats=3):
        best, obj = np.inf, None
        for _ in range(repeats):
            t0 = time.perf_counter()
            obj = make()
            best = min(best, time.perf_counter() - t0)
        return best, obj

    def pca_err(p):
        return float(((p.inverse_transform(p.transform(X_te)) - X_te) ** 2).mean())

    rows = []
    t, p = timeit(lambda: PCA(n_components=d, svd_solver="full",
                              random_state=SEED).fit(X_tr))
    rows.append(("PCA, full SVD", t, pca_err(p)))
    t, p = timeit(lambda: PCA(n_components=d, svd_solver="randomized",
                              random_state=SEED).fit(X_tr))
    rows.append(("PCA, randomised", t, pca_err(p)))
    # a batch has to be at least n_components wide, or partial_fit refuses
    batch = int(max(d + 10, 70))
    t, p = timeit(lambda: IncrementalPCA(n_components=d, batch_size=batch).fit(X_tr))
    rows.append((f"Incremental PCA, {len(X_tr) // batch + 1} batches", t,
                 pca_err(p)))

    def rp():
        return GaussianRandomProjection(n_components=d,
                                        random_state=SEED).fit(X_tr)
    t, g = timeit(rp)
    back = np.linalg.pinv(g.components_.T)
    rows.append(("Random projection", t,
                 float(((g.transform(X_te) @ back - X_te) ** 2).mean())))
    return {"d": int(d), "names": [r[0] for r in rows],
            "time": [float(r[1]) for r in rows],
            "error": [float(r[2]) for r in rows]}


def fig_reducers(rb):
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 3.2))
    cols = [PRIMARY, PRIMARY, PRIMARY, ACCENT]
    y = np.arange(len(rb["names"]))[::-1]
    axes[0].barh(y, rb["time"], color=cols, height=0.6)
    for yy, v in zip(y, rb["time"]):
        axes[0].text(v, yy, f"  {v * 1000:.0f} ms", va="center", fontsize=SMALL,
                     color=MUTED)
    axes[0].set_yticks(y); axes[0].set_yticklabels(rb["names"])
    axes[0].set_xlabel("fit time, seconds")
    axes[0].set_title(f"Cost at d = {rb['d']}")
    axes[0].set_xlim(0, max(rb["time"]) * 1.55)
    axes[1].barh(y, rb["error"], color=cols, height=0.6)
    axes[1].set_yticks(y); axes[1].set_yticklabels([])
    for yy, v in zip(y, rb["error"]):
        axes[1].text(v, yy, f"  {v:.4f}", va="center", fontsize=SMALL,
                     color=MUTED)
    axes[1].set_xlabel("held-out reconstruction error")
    axes[1].set_title("Quality at the same d")
    axes[1].set_xlim(0, max(rb["error"]) * 1.45)
    fig.tight_layout()
    save(fig, "l10-reducers")


def sweep_speed(X, y, audit, ds, k_max=K_MAX):
    """Lecture 9's sweep, run again inside a PCA subspace, and timed."""
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.metrics import adjusted_rand_score, silhouette_score
    out = {"ds": [], "time": [], "best_k": [], "best_sil": [], "ari_full": [],
           "ari_audit": []}
    for d in ds:
        t0 = time.perf_counter()
        Z = PCA(n_components=d, random_state=SEED).fit_transform(X)
        best = (-2.0, None, None)
        for k in range(2, k_max + 1):
            km = KMeans(n_clusters=k, n_init=N_INIT, random_state=SEED).fit(Z)
            s = silhouette_score(Z, km.labels_)
            if s > best[0]:
                best = (float(s), k, km.labels_.copy())
        elapsed = time.perf_counter() - t0
        out["ds"].append(int(d)); out["time"].append(float(elapsed))
        out["best_sil"].append(best[0]); out["best_k"].append(int(best[1]))
        out["ari_full"].append(float(adjusted_rand_score(y, best[2])))
        out["ari_audit"].append(
            float(adjusted_rand_score(y[audit], best[2][audit])))
        print(f"      d={d:4d}  {elapsed:6.1f}s  best k={best[1]:3d}  "
              f"sil={best[0]:.3f}  ARI={out['ari_full'][-1]:.3f}", flush=True)
    return out


def lighting_repair(X, y, audit, d, k_max=K_MAX):
    """Lecture 9 diagnosed lighting; this measures the two one-line repairs.

    L9 concludes that Euclidean distance on raw pixels groups by illumination
    rather than by identity, and L10 then lists "discard the first few
    components" under *what we did not do* — a crisp diagnosis with no measured
    remedy, which is the least useful place to leave a student. Both candidate
    fixes are one line and take seconds on 400 x 4,096:

      * drop the leading principal components, which is where a global
        lighting direction lands
      * scale each image to unit norm first, which removes overall brightness
        but not its direction

    Identical protocol to `sweep_speed`: choose k by silhouette, then report
    ARI. Choosing k by ARI instead would be selecting on the labels we are
    pretending not to have, and would flatter every row equally.
    """
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.metrics import adjusted_rand_score, silhouette_score

    def run(Z, tag):
        t0 = time.perf_counter()
        best = (-2.0, None, None)
        for k in range(2, k_max + 1):
            km = KMeans(n_clusters=k, n_init=N_INIT, random_state=SEED).fit(Z)
            sc = silhouette_score(Z, km.labels_)
            if sc > best[0]:
                best = (float(sc), k, km.labels_.copy())
        r = {"tag": tag, "dims": int(Z.shape[1]), "best_k": int(best[1]),
             "best_sil": best[0],
             "ari_full": float(adjusted_rand_score(y, best[2])),
             "ari_audit": float(adjusted_rand_score(y[audit], best[2][audit])),
             "seconds": float(time.perf_counter() - t0)}
        print(f"      {tag:34s} k={r['best_k']:3d}  sil={r['best_sil']:.3f}  "
              f"ARI={r['ari_full']:.3f}  ({r['seconds']:.0f}s)", flush=True)
        return r

    P = PCA(n_components=d, random_state=SEED).fit_transform(X)
    Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
    Pn = PCA(n_components=d, random_state=SEED).fit_transform(Xn)
    rows = [run(P, f"PCA {d} (what the deck did)"),
            run(P[:, 3:], f"PCA {d}, first 3 discarded"),
            run(Pn, f"unit-norm images, then PCA {d}")]
    out = {"rows": rows, "d": int(d)}
    out["ari_gain_drop3"] = rows[1]["ari_full"] - rows[0]["ari_full"]
    out["ari_gain_unitnorm"] = rows[2]["ari_full"] - rows[0]["ari_full"]
    return out


def all_sweeps(X, y, audit, ds, k_max=K_MAX):
    """The 4,096-dimensional sweep and every reduced one, in a single run.

    These have to be cached together. Wall-clock depends on what else the
    machine is doing, so a raw timing cached from one run and a reduced timing
    measured in another give a speed-up that is a fact about the afternoon
    rather than about the method. One cache entry, one process, one comparison.
    """
    return {"raw": kmeans_sweep(X, y, audit, k_max),
            "reduced": sweep_speed(X, y, audit, ds, k_max)}


def fig_speed_quality(ss, sweep, ari_full_at_best):
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 3.3))
    axes[0].plot(ss["ds"], ss["time"], color=PRIMARY, lw=2.6, marker="o", ms=6)
    axes[0].axhline(sweep["t_total"], color=ACCENT, lw=2.4, ls="--")
    axes[0].set_xscale("log"); axes[0].set_yscale("log")
    loglabels(axes[0], "x", ticks=ss["ds"])
    loglabels(axes[0], "y", ticks=[10, 30, 100, 300])
    axes[0].tick_params(axis="x", labelrotation=30)
    axes[0].set_xlabel("dimensions the sweep runs in")
    axes[0].set_ylabel("seconds for the sweep")
    axes[0].set_title("Time")
    callout(axes[0], f"4096 dims: {sweep['t_total']:.0f}s",
            (ss["ds"][-1], sweep["t_total"]), (0.10, 0.72), ACCENT)
    axes[1].plot(ss["ds"], ss["ari_full"], color=SUCCESS, lw=2.6, marker="o",
                 ms=6)
    axes[1].axhline(ari_full_at_best, color=ACCENT, lw=2.4, ls="--")
    axes[1].set_xscale("log")
    loglabels(axes[1], "x", ticks=ss["ds"])
    axes[1].tick_params(axis="x", labelrotation=30)
    axes[1].set_ylim(0, 0.75)
    axes[1].set_xlabel("dimensions the sweep runs in")
    axes[1].set_ylabel("ARI against the identities")
    axes[1].set_title("Quality")
    callout(axes[1], f"4096 dims: ARI {ari_full_at_best:.2f}",
            (ss["ds"][-1], ari_full_at_best), (0.12, 0.20), ACCENT)
    fig.tight_layout()
    save(fig, "l10-speed-quality")


def dbscan_sweep(Z, y, eps_grid):
    from sklearn.cluster import DBSCAN
    from sklearn.metrics import adjusted_rand_score
    n_clusters, n_noise, ari = [], [], []
    for eps in eps_grid:
        lab = DBSCAN(eps=float(eps), min_samples=3).fit_predict(Z)
        n_clusters.append(int(len(set(lab.tolist()) - {-1})))
        n_noise.append(int((lab == -1).sum()))
        ari.append(float(adjusted_rand_score(y, lab)))
    i = int(np.argmax(ari))
    return {"eps": [float(e) for e in eps_grid], "n_clusters": n_clusters,
            "n_noise": n_noise, "ari": ari, "best_ari": float(ari[i]),
            "best_eps": float(eps_grid[i]),
            "clusters_at_best": n_clusters[i], "noise_at_best": n_noise[i],
            "max_clusters": int(max(n_clusters))}


def fig_dbscan(db):
    fig, ax = plt.subplots(figsize=(9.6, 3.2))
    ax.plot(db["eps"], db["n_clusters"], color=PRIMARY, lw=2.6, marker="o",
            ms=5)
    ax.set_xlabel("eps, the neighbourhood radius")
    ax.set_ylabel("clusters found", color=PRIMARY)
    ax.set_title("DBSCAN needs a density scale. These faces do not have one")
    ax2 = ax.twinx()
    ax2.plot(db["eps"], db["n_noise"], color=ACCENT, lw=2.6, marker="s", ms=5,
             ls="--")
    ax2.set_ylabel("faces called noise", color=ACCENT)
    ax2.grid(False)
    ax.axhline(40, color=SUCCESS, lw=2.2, ls=":")
    i = db["eps"].index(db["best_eps"])
    callout(ax, f"the best eps finds {db['clusters_at_best']} clusters — and\n"
                f"discards {db['noise_at_best']} faces as noise, for ARI "
                f"{db['best_ari']:.2f}",
            (db["best_eps"], db["n_clusters"][i]), (0.36, 0.52), SUCCESS)
    fig.tight_layout()
    save(fig, "l10-dbscan")


def gmm_sweep(Z, y, ks):
    from sklearn.metrics import adjusted_rand_score
    from sklearn.mixture import GaussianMixture
    bic, aic, ari = [], [], []
    for k in ks:
        g = GaussianMixture(n_components=k, covariance_type="diag", n_init=3,
                            random_state=SEED, reg_covar=1e-4).fit(Z)
        bic.append(float(g.bic(Z))); aic.append(float(g.aic(Z)))
        ari.append(float(adjusted_rand_score(y, g.predict(Z))))
    return {"ks": [int(k) for k in ks], "bic": bic, "aic": aic, "ari": ari,
            "k_bic": int(ks[int(np.argmin(bic))]),
            "k_aic": int(ks[int(np.argmin(aic))]),
            "best_ari": float(max(ari)),
            "k_best_ari": int(ks[int(np.argmax(ari))]),
            "ari_at_40": float(ari[list(ks).index(40)]) if 40 in list(ks)
            else float("nan")}


def fig_gmm(gm):
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 3.2))
    axes[0].plot(gm["ks"], gm["bic"], color=PRIMARY, lw=2.6, marker="o", ms=5,
                 label="BIC")
    axes[0].plot(gm["ks"], gm["aic"], color=MUTED, lw=2.2, marker="s", ms=5,
                 ls="--", label="AIC")
    axes[0].axvline(40, color=SUCCESS, lw=2.2, ls=":")
    axes[0].set_xlabel("components")
    axes[0].set_ylabel("information criterion")
    axes[0].set_title("What BIC and AIC choose")
    axes[0].legend(loc="upper left")
    axes[1].plot(gm["ks"], gm["ari"], color=SUCCESS, lw=2.6, marker="o", ms=5)
    axes[1].axvline(40, color=SUCCESS, lw=2.2, ls=":")
    axes[1].set_xlabel("components")
    axes[1].set_ylabel("ARI against the identities")
    axes[1].set_title("What recovers the people")
    callout(axes[1], f"BIC picks {gm['k_bic']}; ARI peaks at {gm['k_best_ari']}",
            (gm["k_best_ari"], gm["best_ari"]), (0.08, 0.20), ACCENT)
    fig.tight_layout()
    save(fig, "l10-gmm")


def corrupt(images, rng, n=12):
    """Faces that are not faces: rotated, mirrored and dimmed, double-exposed."""
    idx = rng.choice(len(images), size=n, replace=False)
    out, kind = [], []
    for j, i in enumerate(idx):
        im = images[i].copy()
        if j % 3 == 0:
            im = np.rot90(im); kind.append("rotated")
        elif j % 3 == 1:
            im = im[:, ::-1] * 0.35; kind.append("dimmed")
        else:
            im = 0.5 * im + 0.5 * im[::-1]; kind.append("double-exposed")
        out.append(np.ascontiguousarray(im))
    return np.array(out), kind


def anomaly_experiment(X, images):
    """Two detectors, the same twelve planted anomalies."""
    from sklearn.decomposition import PCA
    from sklearn.mixture import GaussianMixture
    rng = np.random.default_rng(SEED)
    bad_im, kinds = corrupt(images, rng, n=12)
    bad = bad_im.reshape(len(bad_im), -1).astype(np.float32)
    Xa = np.vstack([X, bad])
    is_bad = np.zeros(len(Xa), bool); is_bad[len(X):] = True

    pca = PCA(n_components=0.99, random_state=SEED).fit(X)
    err = ((pca.inverse_transform(pca.transform(Xa)) - Xa) ** 2).mean(axis=1)
    gm = GaussianMixture(n_components=40, covariance_type="diag", n_init=3,
                         random_state=SEED, reg_covar=1e-4).fit(pca.transform(X))
    dens = gm.score_samples(pca.transform(Xa))

    def caught(score, descending, n=12):
        order = np.argsort(-score if descending else score)[:n]
        return int(is_bad[order].sum())

    return {"kinds": kinds, "err": err, "dens": dens, "is_bad": is_bad,
            "images": np.vstack([images, bad_im]),
            "n_components": int(pca.n_components_),
            "n_planted": 12,
            "caught_recon": caught(err, True),
            "caught_density": caught(dens, False),
            "median_err_clean": float(np.median(err[~is_bad])),
            "median_err_bad": float(np.median(err[is_bad]))}


def fig_anomalies(an):
    top_err = np.argsort(-an["err"])[:10]
    top_dens = np.argsort(an["dens"])[:10]
    fig, axes = plt.subplots(2, 1, figsize=(10.0, 3.2))
    for ax, idx, name, hits in (
            (axes[0], top_err, "reconstruction error", an["caught_recon"]),
            (axes[1], top_dens, "lowest mixture density", an["caught_density"])):
        montage(ax, an["images"][idx], ncol=10)
        ax.set_title(f"the ten most anomalous by {name} — "
                     f"{int(an['is_bad'][idx].sum())} of these ten are planted "
                     f"({hits} of 12 caught in the top twelve)",
                     fontsize=SMALL, color=PRIMARY)
        for i, j in enumerate(idx):
            if an["is_bad"][j]:
                box_cell(ax, i, 10, color=SUCCESS)
    fig.tight_layout()
    save(fig, "l10-anomalies", raster=True)


def label_propagation(X, y, d, budget=LABEL_BUDGET, n_seeds=N_SEEDS):
    """Five ways to spend forty labels, on the same held-out faces."""
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression

    def score(Ztr, ytr, Zte, yte):
        clf = LogisticRegression(max_iter=3000, random_state=SEED)
        return float(clf.fit(Ztr, ytr).score(Zte, yte))

    arms = {"40 at random": [], "40, one per cluster": [],
            "propagated to whole cluster": [],
            "propagated to closest 75%": [], "all 280 labels": []}
    for seed in range(n_seeds):
        X_tr, X_te, y_tr, y_te = split(X, y, seed=seed)
        pca = PCA(n_components=d, random_state=SEED).fit(X_tr)
        Ztr, Zte = pca.transform(X_tr), pca.transform(X_te)
        rng = np.random.default_rng(3000 + seed)

        pick = rng.choice(len(Ztr), size=budget, replace=False)
        arms["40 at random"].append(score(Ztr[pick], y_tr[pick], Zte, y_te))

        km = KMeans(n_clusters=budget, n_init=10, random_state=seed).fit(Ztr)
        dist = km.transform(Ztr)
        rep = np.argmin(dist, axis=0)          # the face nearest each centroid
        arms["40, one per cluster"].append(score(Ztr[rep], y_tr[rep], Zte, y_te))

        prop = y_tr[rep][km.labels_]           # the representative's label
        arms["propagated to whole cluster"].append(score(Ztr, prop, Zte, y_te))

        own = dist[np.arange(len(Ztr)), km.labels_]
        keep = np.zeros(len(Ztr), bool)
        for c in range(budget):
            members = np.where(km.labels_ == c)[0]
            cut = np.percentile(own[members], 75)
            keep[members[own[members] <= cut]] = True
        arms["propagated to closest 75%"].append(
            score(Ztr[keep], prop[keep], Zte, y_te))

        arms["all 280 labels"].append(score(Ztr, y_tr, Zte, y_te))
        print(f"      seed {seed:2d}  " + "  ".join(
            f"{v[-1]:.3f}" for v in arms.values()), flush=True)

    out = {k: {"mean": float(np.mean(v)), "sd": float(np.std(v)),
               "min": float(np.min(v)), "max": float(np.max(v))}
           for k, v in arms.items()}
    out["_order"] = list(arms)
    return out


def fig_labelprop(lp):
    names = lp["_order"]
    means = [lp[n]["mean"] * 100 for n in names]
    sds = [lp[n]["sd"] * 100 for n in names]
    cols = [ACCENT, PRIMARY, PRIMARY, PRIMARY, SUCCESS]
    fig, ax = plt.subplots(figsize=(9.6, 3.4))
    yy = np.arange(len(names))[::-1]
    ax.barh(yy, means, xerr=sds, color=cols, height=0.62,
            error_kw=dict(ecolor=MUTED, capsize=5, lw=1.8))
    for y_, m, s in zip(yy, means, sds):
        ax.text(m + s + 1.5, y_, f"{m:.1f}%", va="center", fontsize=SMALL,
                color=MUTED)
    ax.set_yticks(yy); ax.set_yticklabels(names)
    ax.set_xlabel(f"accuracy on the 120 held-out faces, mean of {N_SEEDS} splits")
    ax.set_xlim(0, 112)
    ax.set_title("What forty labels buy, depending on which forty")
    fig.tight_layout()
    save(fig, "l10-labelprop")


def pca_leak(X, y, d, n_seeds=N_SEEDS):
    """Lecture 10's assistant failure: PCA fitted before the split.

    Two numbers, because they behave differently. The reconstruction error of a
    held-out face measured on a subspace that face helped define is inflated by
    construction. Downstream accuracy is a separate question and is measured
    separately — reporting only the one that moves would be the same dishonesty
    the course exists to remove.
    """
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    rows = []
    for seed in range(n_seeds):
        X_tr, X_te, y_tr, y_te = split(X, y, seed=seed)
        honest = PCA(n_components=d, random_state=SEED).fit(X_tr)
        leaky = PCA(n_components=d, random_state=SEED).fit(X)

        def err(p):
            return float(((p.inverse_transform(p.transform(X_te)) - X_te) ** 2
                          ).mean())

        def acc(p):
            clf = LogisticRegression(max_iter=3000, random_state=SEED)
            return float(clf.fit(p.transform(X_tr), y_tr)
                         .score(p.transform(X_te), y_te))

        rows.append((err(honest), err(leaky), acc(honest), acc(leaky)))
        print(f"      seed {seed:2d}  err {rows[-1][0]:.5f} -> {rows[-1][1]:.5f}"
              f"   acc {rows[-1][2]:.3f} -> {rows[-1][3]:.3f}", flush=True)
    e_h, e_l, a_h, a_l = (np.array(v) for v in zip(*rows))
    drop = 100 * (1 - e_l / e_h)
    return {"d": int(d), "n_seeds": n_seeds,
            "err_honest": float(e_h.mean()), "err_leaky": float(e_l.mean()),
            "err_drop_pct": float(drop.mean()), "err_drop_sd": float(drop.std()),
            "err_wins": int((e_l < e_h).sum()),
            "acc_honest": float(a_h.mean()), "acc_honest_sd": float(a_h.std()),
            "acc_leaky": float(a_l.mean()), "acc_leaky_sd": float(a_l.std()),
            "acc_gap_mean": float((a_l - a_h).mean()),
            "acc_gap_sd": float((a_l - a_h).std()),
            "acc_wins": int((a_l > a_h).sum())}


def fig_leak(lk):
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 3.2))
    axes[0].bar([0, 1], [lk["err_honest"], lk["err_leaky"]],
                color=[SUCCESS, ACCENT], width=0.5)
    axes[0].set_xticks([0, 1])
    axes[0].set_xticklabels(["fitted on train", "fitted on everything"])
    axes[0].set_ylabel("reconstruction error,\nheld-out faces")
    axes[0].set_ylim(0, max(lk["err_honest"], lk["err_leaky"]) * 1.55)
    axes[0].set_title("The number the leak flatters")
    callout(axes[0], f"{lk['err_drop_pct']:.0f}% lower,\nand entirely fictional",
            (1, lk["err_leaky"]), (0.10, 0.62), ACCENT)
    axes[1].bar([0, 1], [lk["acc_honest"] * 100, lk["acc_leaky"] * 100],
                yerr=[lk["acc_honest_sd"] * 100, lk["acc_leaky_sd"] * 100],
                color=[SUCCESS, ACCENT], width=0.5,
                error_kw=dict(ecolor=MUTED, capsize=6, lw=1.8))
    axes[1].set_xticks([0, 1])
    axes[1].set_xticklabels(["fitted on train", "fitted on everything"])
    axes[1].set_ylabel("classification accuracy, %")
    axes[1].set_ylim(0, 118)
    axes[1].set_title("The number it barely moves")
    callout(axes[1], f"{lk['acc_gap_mean'] * 100:+.1f} points,\n"
                     f"sd {lk['acc_gap_sd'] * 100:.1f}",
            (1, lk["acc_leaky"] * 100), (0.08, 0.18), MUTED)
    fig.tight_layout()
    save(fig, "l10-leak")


# ------------------------------------------------------------------- driver

def main():
    setup()
    load_cache()

    print("Loading Olivetti faces…")
    X, y, images = load_faces()
    audit = audit_index(np.random.default_rng(SEED))
    X_tr, X_te, y_tr, y_te = split(X, y)

    same_pairs = int(sum(y[audit][i] == y[audit][j]
                         for i in range(LABEL_BUDGET)
                         for j in range(i + 1, LABEL_BUDGET)))
    facts = {
        "l09_n_images": int(len(X)),
        "l09_n_people": int(len(np.unique(y))),
        "l09_per_person": int(np.bincount(y).min()),
        "l09_side": 64,
        "l09_n_features": int(X.shape[1]),
        "l09_bytes_mb": float(X.nbytes / 1e6),
        "l09_n_pairs": int(len(X) * (len(X) - 1) // 2),
        "l09_label_budget": LABEL_BUDGET,
        "l09_labelled_pct": 100.0 * LABEL_BUDGET / len(X),
        "l09_audit_pairs": LABEL_BUDGET * (LABEL_BUDGET - 1) // 2,
        "l09_audit_same_pairs": same_pairs,
        "l09_threads": int(N_THREADS),
        "l10_n_train": int(len(X_tr)),
        "l10_n_test": int(len(X_te)),
    }
    print(f"  {facts['l09_n_images']} images, {facts['l09_n_features']} features, "
          f"{facts['l09_bytes_mb']:.2f} MB")

    print("Lecture 9 — the corpus:")
    fig_corpus(images, y)
    fig_one_person(images, y)

    print("Lecture 9 — distances in pixel space:")
    ds = cached("app05/distances", lambda: distance_structure(X, y))
    fig_distances(ds)
    facts.update({f"l09_dist_{k}": v for k, v in ds.items()
                  if not k.startswith("_")})
    # the mirror statistic: different-person pairs that are closer than the
    # median same-person pair. Both directions, because one of them alone
    # always flatters whichever conclusion you already wanted.
    facts["l09_dist_overlap_rev"] = float(
        (ds["_between"] < np.median(ds["_within"])).mean())

    # d95 is needed before the sweeps, because the reduced sweeps have to be
    # timed in the same process as the 4,096-dimensional one.
    pc = cached("app05/pca_curves", lambda: pca_curves(X))
    d95 = pc["d95"]

    print(f"Lecture 9 and 10 — every sweep, k = 2 to {K_MAX}, in one run:")
    sweeps = cached("app05/sweeps_v2",
                    lambda: all_sweeps(X, y, audit, [20, 50, 100, d95, 399]))
    sweep, ss = sweeps["raw"], sweeps["reduced"]
    print(f"    {sweep['t_total']:.0f}s total "
          f"({sweep['t_fit']:.0f}s fitting, {sweep['t_sil']:.0f}s scoring)")
    best_i = int(np.argmax(sweep["silhouette"]))
    best_k = int(sweep["ks"][best_i])
    shown_k = best_k if best_k in sweep["labels"] else 40

    print("Lecture 9 — the trivial baseline, random assignment:")
    base = cached("app05/random_baseline", lambda: random_baseline(X, y, audit))

    ev = elbow_evidence(sweep)
    fig_elbow(sweep, ev)
    fig_silhouette_vs_k(sweep, base)
    fig_silhouette_diagram(X, sweep)
    cq = fig_clusters(images, y, sweep["labels"][shown_k], shown_k)
    sizes = fig_cluster_sizes(sweep["labels"][shown_k], y, shown_k)
    fig_timing_l09(sweep)

    print("Lecture 9 — the assistant failure, over 20 seeds:")
    opt = cached("app05/selection_optimism", lambda: selection_optimism(X))

    i40 = list(sweep["ks"]).index(40)
    facts.update({
        "l09_sweep_seconds": sweep["t_total"],
        "l09_sweep_fit_seconds": sweep["t_fit"],
        "l09_sweep_sil_seconds": sweep["t_sil"],
        "l09_sweep_minutes": sweep["t_total"] / 60.0,
        "l09_sweep_n_fits": (K_MAX - 1) * N_INIT,
        "l09_k_max": K_MAX, "l09_n_init": N_INIT,
        "l09_best_k": best_k,
        "l09_best_silhouette": float(sweep["silhouette"][best_i]),
        "l09_sil_at_40": float(sweep["silhouette"][i40]),
        "l09_ari_full_at_best": float(sweep["ari_full"][best_i]),
        "l09_ari_full_at_40": float(sweep["ari_full"][i40]),
        "l09_ari_audit_at_best": float(sweep["ari_audit"][best_i]),
        "l09_ari_audit_at_40": float(sweep["ari_audit"][i40]),
        "l09_elbow": ev,
        "l09_random_baseline": base,
        "l09_clusters": cq,
        "l09_cluster_sizes": sizes,
        "l09_selection_optimism": opt,
    })

    print("Lecture 10 — PCA:")
    fig_eigenfaces(X)
    fig_scree(pc)
    fig_reconstruction(X_tr, X_te)

    rq = cached("app05/reduction_quality",
                lambda: reduction_quality(X_tr, X_te,
                                          [2, 5, 10, 25, 50, 100, d95, 279]))
    fig_recon_error(rq, d95)

    print("Lecture 10 — Johnson–Lindenstrauss:")
    jl = jl_numbers()
    fig_jl(jl)
    jm = cached("app05/jl_measured_sq", lambda: jl_measured(X))
    fig_jl_measured(jm, jl)

    print("Lecture 10 — the four reducers:")
    rb = cached("app05/reducers", lambda: reducer_bench(X_tr, X_te, d95))
    fig_reducers(rb)

    print("Lecture 10 — the same sweep, compressed first:")
    fig_speed_quality(ss, sweep, sweep["ari_full"][best_i])

    from sklearn.decomposition import PCA
    Z = PCA(n_components=d95, random_state=SEED).fit_transform(X)

    print("Lecture 10 — DBSCAN and Gaussian mixtures:")
    db = cached("app05/dbscan",
                lambda: dbscan_sweep(Z, y, np.round(np.linspace(2, 14, 25), 2)))
    fig_dbscan(db)
    gm = cached("app05/gmm", lambda: gmm_sweep(Z, y, list(range(5, 81, 5))))
    fig_gmm(gm)

    print("Lecture 10 — anomalies:")
    an = cached("app05/anomalies", lambda: anomaly_experiment(X, images))
    fig_anomalies(an)

    print("Lecture 10 — label propagation:")
    lp = cached("app05/labelprop", lambda: label_propagation(X, y, d=d95))
    fig_labelprop(lp)

    print("Lecture 10 — the assistant failure, over 20 seeds:")
    lk = cached("app05/pca_leak", lambda: pca_leak(X, y, d=d95))
    fig_leak(lk)

    gmm_params = 4096 * 4097 // 2
    i95 = ss["ds"].index(d95)
    facts.update({
        "l10_d90": pc["d90"], "l10_d95": d95, "l10_d99": pc["d99"],
        "l10_evr_1": pc["evr_1"], "l10_evr_2": pc["evr_2"],
        "l10_evr_top10": pc["evr_top10"], "l10_evr_top50": pc["evr_top50"],
        "l10_max_components": pc["n_components_max"],
        "l10_compression_95": float(4096 / d95),
        # a random d-dimensional subspace of R^4096 keeps d/4096 of a generic
        # vector's squared norm. That is why random projection reconstructs so
        # badly while still preserving distances.
        "l10_random_energy_pct": 100.0 * d95 / 4096,
        "l10_rp_error_ratio": float(rq["rp"][rq["ds"].index(d95)]
                                    / rq["pca"][rq["ds"].index(d95)]),
        "l10_recon": rq,
        "l10_jl": jl,
        "l10_jl_measured": jm,
        "l10_reducers": rb,
        "l10_sweep_speed": ss,
        "l10_sweep_seconds_at_95": ss["time"][i95],
        "l10_speedup": float(sweep["t_total"] / ss["time"][i95]),
        "l10_ari_at_95": ss["ari_full"][i95],
        "l10_lighting_repair": cached(
            "l10_lighting_repair_v1",
            lambda: lighting_repair(X, y, audit, pc["d95"])),
        "l10_best_k_at_95": ss["best_k"][i95],
        "l10_dbscan": db,
        "l10_gmm": gm,
        "l10_gmm_full_params_per_component": gmm_params,
        "l10_gmm_full_params_40": gmm_params * 40,
        "l10_gmm_diag_params_40": (2 * d95 + 1) * 40,
        "l10_anomaly": {"kinds": an["kinds"], "n_components": an["n_components"],
                        "n_planted": an["n_planted"],
                        "caught_recon": an["caught_recon"],
                        "caught_density": an["caught_density"],
                        "median_err_clean": an["median_err_clean"],
                        "median_err_bad": an["median_err_bad"]},
        "l10_labelprop": {k: v for k, v in lp.items() if k != "_order"},
        "l10_pca_leak": lk,
    })

    export(**jsonable(facts))

    problems = [p for p in check_text_floor()
                if p.startswith("l09-") or p.startswith("l10-")]
    if problems:
        print("\nfigures under the text floor:")
        for p in problems:
            print("  " + p)
        return 1
    print("\nlecture 9 and 10 figures clear the 15px text floor")
    return 0


if __name__ == "__main__":
    sys.exit(main())
