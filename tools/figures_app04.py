#!/usr/bin/env python3
"""
Figures and measured numbers for Application 4 — Lectures 7 and 8.

    python3 tools/figures_app04.py

Land cover classification on the Forest CoverType dataset, under an
interpretability constraint (Lecture 7), and the variance of an average of
correlated predictors (Lecture 8).

Everything a slide in either deck prints is produced here and merged into
`assets/figures/figures.json` through `figkit.export()`. Nothing is illustrative.

The script also prints, under `=== BLOCKS FOR THE SLIDES ===`, the exact text of
every code/output block the two decks show. Those are pasted verbatim; if this
script and a slide disagree, the slide is the bug (TRICKS §4).

Expensive fits go through `figkit.cached()`. A cold run is roughly four minutes
on a laptop CPU (the CoverType download is another thirty seconds); a warm run
is about twenty seconds. Delete /private/tmp/claude-501/aiml-data/fits-v2.pkl to
refit from scratch.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

import figkit as fk
from figkit import (ACCENT, AXIS, BODY, MATH, MUTED, PRIMARY, RULE, SMALL,
                    SUCCESS, TICK, cached, check_text_floor, export, load_cache,
                    save, setup)

CACHE = fk._mf.CACHE
SOFT = "#f4f7f9"

SEED = 42
N_SAMPLE = 60_000          # stratified subsample of the 581,012 rows
TEST_FRAC = 0.2
LEGIBLE_DEPTH = 8          # the cap the brief imposes: 8 conditions per rule
N_TREES = 200              # pool size for every ensemble in Lecture 8

# bootstrap? feature subsampling? random thresholds? — one arm per combination
KINDS = ["bagging", "forest", "extra", "extra_bs"]

COVER_TYPES = ["Spruce/Fir", "Lodgepole Pine", "Ponderosa Pine",
               "Cottonwood/Willow", "Aspen", "Douglas-fir", "Krummholz"]

facts: dict = {}


def plain_log(ax, which="y", ticks=None, fmt="{:,.0f}"):
    """Log ticks without mathtext.

    setup() sets text.parse_math=False course-wide (TRICKS §9.1), and
    matplotlib's log formatter emits "$\\mathdefault{10^{3}}$" — which then
    renders as that literal string. Any log axis in this course must set its own
    ticks and a plain formatter.
    """
    from matplotlib.ticker import FuncFormatter, NullFormatter
    axis = ax.yaxis if which == "y" else ax.xaxis
    if ticks is not None:
        (ax.set_yticks if which == "y" else ax.set_xticks)(ticks)
    axis.set_major_formatter(FuncFormatter(lambda v, _: fmt.format(v)))
    axis.set_minor_formatter(NullFormatter())


# --------------------------------------------------------------------- data

def short_names(names: list[str]) -> list[str]:
    """Feature names that fit inside a plotted tree node."""
    out = []
    for n in names:
        n = (n.replace("Horizontal_Distance_To_", "HDist_")
              .replace("Vertical_Distance_To_", "VDist_")
              .replace("Hillshade_", "Shade_")
              .replace("Wilderness_Area_", "Wild_")
              .replace("Soil_Type_", "Soil_")
              .replace("Hydrology", "Hydro")
              .replace("Roadways", "Road")
              .replace("Fire_Points", "Fire"))
        out.append(n)
    return out


def load_cover():
    """The 581,012-row CoverType set, stratified down to N_SAMPLE and split.

    Subsampling is a deliberate, stated compromise: the full set trains a tree in
    about half a minute and a 200-tree forest in several, which does not fit in a
    ninety-minute session. The class proportions are preserved exactly.
    """
    from sklearn.datasets import fetch_covtype
    from sklearn.model_selection import train_test_split

    d = fetch_covtype(data_home=str(CACHE), download_if_missing=True)
    X_all, y_all = d.data, d.target
    names = short_names(list(d.feature_names))

    X, _, y, _ = train_test_split(X_all, y_all, train_size=N_SAMPLE,
                                  stratify=y_all, random_state=SEED)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_FRAC, stratify=y, random_state=SEED)
    return dict(X_all_shape=X_all.shape, y_all=y_all, names=names,
                X_tr=X_tr, X_te=X_te, y_tr=y_tr, y_te=y_te)


# ------------------------------------------------------------- lecture 7 fits

def sweep_depth(D):
    """Training and 5-fold cross-validated accuracy against max_depth."""
    from sklearn.model_selection import StratifiedKFold, cross_validate
    from sklearn.tree import DecisionTreeClassifier

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    depths = list(range(1, 19))
    rows = []
    for d in depths:
        clf = DecisionTreeClassifier(max_depth=d, random_state=SEED)
        r = cross_validate(clf, D["X_tr"], D["y_tr"], cv=cv,
                           return_train_score=True, n_jobs=-1)
        full = clf.fit(D["X_tr"], D["y_tr"])
        rows.append(dict(depth=d,
                         train=float(r["train_score"].mean()),
                         cv=float(r["test_score"].mean()),
                         cv_std=float(r["test_score"].std()),
                         leaves=int(full.get_n_leaves())))
    return depths, rows


def grid_2d(D):
    """5-fold accuracy over max_depth x min_samples_leaf."""
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.tree import DecisionTreeClassifier

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    depths = [4, 6, 8, 10, 12, None]
    leaves = [1, 5, 20, 50, 200]
    grid = np.zeros((len(depths), len(leaves)))
    for i, d in enumerate(depths):
        for j, m in enumerate(leaves):
            clf = DecisionTreeClassifier(max_depth=d, min_samples_leaf=m,
                                         random_state=SEED)
            grid[i, j] = cross_val_score(clf, D["X_tr"], D["y_tr"],
                                         cv=cv, n_jobs=-1).mean()
    return depths, leaves, grid


def path_lengths(D):
    """How many conditions each test prediction actually rests on."""
    from sklearn.tree import DecisionTreeClassifier

    out = {}
    for tag, kw in [("free", dict()),
                    ("legible", dict(max_depth=LEGIBLE_DEPTH,
                                     min_samples_leaf=BEST_LEAF))]:
        clf = DecisionTreeClassifier(random_state=SEED, **kw).fit(
            D["X_tr"], D["y_tr"])
        # decision_path has one entry per node visited, root and leaf included;
        # the number of *tests* applied is that minus one.
        lens = np.asarray(clf.decision_path(D["X_te"]).sum(axis=1)).ravel() - 1
        out[tag] = dict(lens=lens.astype(int),
                        depth=int(clf.get_depth()),
                        leaves=int(clf.get_n_leaves()),
                        nodes=int(clf.tree_.node_count),
                        train=float(clf.score(D["X_tr"], D["y_tr"])),
                        test=float(clf.score(D["X_te"], D["y_te"])))
    return out


def tree_features(D, depth, leaf):
    """Which of the 54 columns a constrained tree actually consults."""
    from sklearn.tree import DecisionTreeClassifier
    t = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=leaf,
                               random_state=SEED).fit(D["X_tr"], D["y_tr"])
    return [f for f in t.tree_.feature if f >= 0]


# ------------------------------------------------------------- lecture 8 fits

def criterion_pairs(D):
    """Gini against entropy over 20 resamples: do they disagree?"""
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeClassifier

    rows = []
    for s in range(20):
        Xs, _, ys, _ = train_test_split(D["X_tr"], D["y_tr"], train_size=0.8,
                                        stratify=D["y_tr"], random_state=s)
        rec = {}
        for crit in ("gini", "entropy"):
            t = DecisionTreeClassifier(criterion=crit, max_depth=LEGIBLE_DEPTH,
                                       random_state=SEED).fit(Xs, ys)
            rec[crit] = dict(
                acc=float(accuracy_score(D["y_te"], t.predict(D["X_te"]))),
                root_feat=int(t.tree_.feature[0]),
                root_thr=float(t.tree_.threshold[0]),
                pred=t.predict(D["X_te"]))
        rec["agree"] = float((rec["gini"]["pred"] == rec["entropy"]["pred"]).mean())
        rec["same_root"] = bool(rec["gini"]["root_feat"] == rec["entropy"]["root_feat"])
        rows.append(rec)
    return rows


def instability(D):
    """Twenty trees on twenty 90% subsamples of the SAME training set."""
    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeClassifier

    trees, preds, roots, accs, second = [], [], [], [], []
    leaves, nfeat = [], []
    for s in range(20):
        Xs, _, ys, _ = train_test_split(D["X_tr"], D["y_tr"], train_size=0.9,
                                        stratify=D["y_tr"], random_state=1000 + s)
        t = DecisionTreeClassifier(max_depth=LEGIBLE_DEPTH,
                                   min_samples_leaf=BEST_LEAF,
                                   random_state=SEED).fit(Xs, ys)
        preds.append(t.predict(D["X_te"]))
        roots.append((int(t.tree_.feature[0]), float(t.tree_.threshold[0])))
        accs.append(float(t.score(D["X_te"], D["y_te"])))
        # the signature of the top three levels: which questions, in which
        # order. Two trees with the same signature ask the same first seven
        # questions of every patch that reaches them.
        tt, sig, frontier = t.tree_, [], [0]
        for _ in range(3):
            nxt = []
            for nd in frontier:
                if nd == -1 or tt.children_left[nd] == -1:
                    sig.append(("leaf", 0.0))
                    nxt += [-1, -1]
                else:
                    sig.append((int(tt.feature[nd]), round(float(tt.threshold[nd]), 1)))
                    nxt += [int(tt.children_left[nd]), int(tt.children_right[nd])]
            frontier = nxt
        second.append(tuple(sig))
        leaves.append(int(t.get_n_leaves()))
        nfeat.append(len({int(f) for f in t.tree_.feature if f >= 0}))
        trees.append(t)
    P = np.array(preds)
    n = len(P)
    dis = [float((P[i] != P[j]).mean()) for i in range(n) for j in range(i + 1, n)]
    unanimous = float((P == P[0]).all(axis=0).mean())
    return dict(preds=P, roots=roots, accs=np.array(accs), second=second,
                disagree=np.array(dis), unanimous=unanimous,
                n_signatures=len(set(second)), leaves=leaves, nfeat=nfeat)


def ensemble_pool(D, kind):
    """Fit a 200-member pool and keep every member's test predictions.

    Fitting once and slicing the pool is what makes the n_estimators sweep and
    the correlation measurement the *same* experiment rather than two.
    """
    from sklearn.ensemble import (BaggingClassifier, ExtraTreesClassifier,
                                  RandomForestClassifier)
    from sklearn.tree import DecisionTreeClassifier

    if kind == "bagging":
        m = BaggingClassifier(DecisionTreeClassifier(random_state=SEED),
                              n_estimators=N_TREES, max_features=1.0,
                              bootstrap=True, random_state=SEED, n_jobs=-1)
    elif kind == "forest":
        m = RandomForestClassifier(n_estimators=N_TREES, max_features="sqrt",
                                   random_state=SEED, n_jobs=-1)
    else:
        m = ExtraTreesClassifier(n_estimators=N_TREES, max_features="sqrt",
                                 bootstrap=(kind == "extra_bs"),
                                 random_state=SEED, n_jobs=-1)
    m.fit(D["X_tr"], D["y_tr"])

    classes = m.classes_
    n_te = len(D["y_te"])
    prob_sum = np.zeros((n_te, len(classes)))
    correct = np.zeros((N_TREES, n_te), dtype=np.float32)
    checkpoints = [1, 2, 3, 5, 10, 20, 30, 50, 75, 100, 150, 200]
    acc_at = {}
    # Every ensemble in Scikit-Learn re-encodes y as positions 0..k-1 before
    # handing it to its members, so a member's predict() returns POSITIONS, not
    # cover types. Comparing them with y_test directly scores about 8% and looks
    # like a bad model rather than a bug. A bootstrap can also drop a rare class
    # from one member, so its probability columns are a subset.
    assert set(np.asarray(m.estimators_[0].classes_).astype(int)) <= set(
        range(len(classes))), "member classes are not positions — check the map"
    for t, est in enumerate(m.estimators_):
        p = est.predict_proba(D["X_te"])
        cols = np.asarray(est.classes_).astype(int)
        prob_sum[:, cols] += p
        member = classes[est.predict(D["X_te"]).astype(int)]
        correct[t] = (member == D["y_te"]).astype(np.float32)
        if t + 1 in checkpoints:
            acc_at[t + 1] = float(
                (classes[prob_sum.argmax(axis=1)] == D["y_te"]).mean())
    return dict(correct=correct, acc_at=acc_at,
                acc=float(m.score(D["X_te"], D["y_te"])),
                importances=getattr(m, "feature_importances_", None))


def make_ensemble(kind, n, seed):
    from sklearn.ensemble import (BaggingClassifier, ExtraTreesClassifier,
                                  RandomForestClassifier)
    from sklearn.tree import DecisionTreeClassifier
    if kind == "bagging":
        return BaggingClassifier(DecisionTreeClassifier(random_state=seed),
                                 n_estimators=n, max_features=1.0,
                                 bootstrap=True, random_state=seed, n_jobs=-1)
    if kind == "forest":
        return RandomForestClassifier(n_estimators=n, max_features="sqrt",
                                      random_state=seed, n_jobs=-1)
    # sklearn's default for extra-trees is bootstrap=False: the random
    # thresholds REPLACE the bootstrap rather than joining it. That default is
    # measured, not assumed — "extra_bs" turns it back on, and the pair is what
    # separates the effect of the bootstrap from the effect of the thresholds.
    return ExtraTreesClassifier(n_estimators=n, max_features="sqrt",
                                bootstrap=(kind == "extra_bs"),
                                random_state=seed, n_jobs=-1)


def variance_experiment(kind, K=20, m=20, n_z=20_000, n_test=12_000):
    """The thread's experiment: K independent training sets, m members each.

    rho in the formula is the correlation between two members of the pool over
    the randomness of the whole procedure — which includes *which training set
    you got*. Two trees grown on the same data are correlated through that data;
    that is the entire content of the correlated term. So we need more than one
    training set, and CoverType's 581,012 rows are enough to cut 20 disjoint
    ones without ever reusing a row.
    """
    from sklearn.datasets import fetch_covtype
    d = fetch_covtype(data_home=str(CACHE), download_if_missing=True)
    X, y = d.data, d.target
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(y))
    te = perm[:n_test]
    pool = perm[n_test:n_test + K * n_z]
    assert len(pool) == K * n_z, "not enough rows for disjoint training sets"
    Xte, yte = X[te], y[te]

    S = np.zeros((K, m, n_test), dtype=np.float32)
    for k in range(K):
        idx = pool[k * n_z:(k + 1) * n_z]
        model = make_ensemble(kind, m, seed=1000 + k).fit(X[idx], y[idx])
        cls = model.classes_
        for j, est in enumerate(model.estimators_):
            S[k, j] = (cls[est.predict(Xte).astype(int)] == yte)
    return S


def decompose(S: np.ndarray) -> dict:
    """Split one member's variance into the part the training set explains.

    tau^2  the variance carried by *which training set you drew*  -> shared by
           every member of the pool, so averaging cannot touch it
    within the variance a member has left once the training set is fixed
    rho    tau^2 / (tau^2 + within)
    """
    K, m, N = S.shape
    within = S.var(axis=1, ddof=1).mean(axis=0)          # per test point
    between = S.mean(axis=1).var(axis=0, ddof=1)         # per test point
    tau2 = np.maximum(between - within / m, 0.0)         # ANOVA estimator
    sigma2 = tau2 + within
    rho = float(tau2.mean() / sigma2.mean())
    s2 = float(sigma2.mean())
    curve, pred = {}, {}
    for n in (1, 2, 3, 5, 10, 20):
        curve[n] = float(S[:, :n].mean(axis=1).var(axis=0, ddof=1).mean())
        pred[n] = float((tau2 + within / n).mean())
    return dict(rho=rho, sigma2=s2, tau2=float(tau2.mean()),
                within=float(within.mean()), curve=curve, predicted=pred,
                floor=float(tau2.mean()), K=K, m=m,
                member_acc=float(S.mean()))


def importance_check(D):
    """The assistant's 'explanation': impurity importance, and a decoy column."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.inspection import permutation_importance

    rng = np.random.default_rng(SEED)
    Xtr = np.hstack([D["X_tr"], rng.random((len(D["X_tr"]), 1))])
    Xte = np.hstack([D["X_te"], rng.random((len(D["X_te"]), 1))])
    rf = RandomForestClassifier(n_estimators=100, max_features="sqrt",
                                random_state=SEED, n_jobs=-1).fit(Xtr, D["y_tr"])
    imp = rf.feature_importances_
    order = np.argsort(imp)[::-1]
    decoy = len(imp) - 1
    rank = int(np.where(order == decoy)[0][0]) + 1

    sub = slice(0, 4000)
    perm = permutation_importance(rf, Xte[sub], D["y_te"][sub], n_repeats=5,
                                  random_state=SEED, n_jobs=-1)
    return dict(imp=imp, rank=rank, n_cols=len(imp),
                decoy_imp=float(imp[decoy]),
                perm_mean=perm.importances_mean,
                perm_std=perm.importances_std,
                decoy_perm=float(perm.importances_mean[decoy]),
                acc=float(rf.score(Xte, D["y_te"])))


# ------------------------------------------------------------------- figures

def fig_class_balance(D):
    y = D["y_tr"]
    counts = np.array([(y == k).sum() for k in range(1, 8)])
    share = counts / counts.sum()
    order = np.argsort(share)[::-1]

    fig, ax = plt.subplots(figsize=(11.0, 3.4))
    colours = [ACCENT if i == order[0] else PRIMARY for i in order]
    ax.bar(range(7), share[order] * 100, color=colours, width=0.68)
    ax.set_xticks(range(7))
    ax.set_xticklabels([COVER_TYPES[i] for i in order], rotation=18,
                       ha="right")
    ax.set_ylabel("share of the training set  (%)")
    ax.set_title("Seven cover types, 48,000 training patches")
    ax.set_ylim(0, 56)
    for k, i in enumerate(order):
        ax.text(k, share[i] * 100 + 1.2, f"{share[i]*100:.1f}", ha="center",
                fontsize=TICK, color=MUTED)
    ax.annotate(f"always predict this and you are\nright {share[order[0]]*100:.1f}% of the time",
                xy=(0.28, share[order[0]] * 100), xytext=(1.9, 43),
                fontsize=SMALL, color=ACCENT,
                bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=ACCENT,
                          lw=1.2),
                arrowprops=dict(arrowstyle="->", color=ACCENT, lw=1.8))
    save(fig, "l07-class-balance")
    return counts, share


def fig_tree(D, tree):
    """The top three levels, plotted. No graphviz binary on this machine, so
    sklearn.tree.plot_tree into matplotlib rather than export_graphviz."""
    from sklearn.tree import plot_tree
    fig, ax = plt.subplots(figsize=(11.5, 3.7))
    anns = plot_tree(tree, max_depth=2, feature_names=D["names"],
                     class_names=COVER_TYPES, filled=True, rounded=True,
                     impurity=False, proportion=True, precision=1,
                     fontsize=11, ax=ax)
    # the seven-entry class-count vector is wider than a node box at any size
    # a projector can read; the split, the share and the majority class are
    # what a justification actually needs
    for a in anns:
        a.set_text("\n".join(l for l in a.get_text().split("\n")
                              if not l.startswith("value")))
    save(fig, "l07-tree")


def fig_depth(depths, rows):
    tr = np.array([r["train"] for r in rows]) * 100
    cv = np.array([r["cv"] for r in rows]) * 100
    sd = np.array([r["cv_std"] for r in rows]) * 100
    lv = np.array([r["leaves"] for r in rows])

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 3.9))
    ax = axes[0]
    ax.plot(depths, tr, "-o", ms=4, color=ACCENT, label="training accuracy")
    ax.plot(depths, cv, "-o", ms=4, color=PRIMARY, label="5-fold CV accuracy")
    ax.fill_between(depths, cv - sd, cv + sd, color=PRIMARY, alpha=0.16, lw=0)
    ax.axvline(LEGIBLE_DEPTH, color=SUCCESS, lw=2, ls="--")
    ax.text(LEGIBLE_DEPTH + 0.4, 96.5, "the brief stops here",
            color=SUCCESS, fontsize=SMALL, va="top", ha="left")
    ax.set_xticks(range(2, 19, 2))
    ax.set_xlabel("max_depth")
    ax.set_ylabel("accuracy  (%)")
    ax.set_title("Accuracy keeps climbing past the cap")
    ax.set_ylim(55, 102)
    ax.legend(loc="lower right")

    ax = axes[1]
    ax.semilogy(depths, lv, "-o", ms=4, color=MATH)
    ax.axvline(LEGIBLE_DEPTH, color=SUCCESS, lw=2, ls="--")
    ax.set_xticks(range(2, 19, 2))
    ax.set_xlabel("max_depth")
    ax.set_ylabel("leaves  (log scale)")
    plain_log(ax, "y", [1, 10, 100, 1_000, 10_000])
    ax.set_ylim(1.5, 12_000)
    ax.set_title("So does the number of rules to read")
    ax.annotate(f"{lv[LEGIBLE_DEPTH-1]:,} rules", xy=(LEGIBLE_DEPTH, lv[LEGIBLE_DEPTH-1]),
                xytext=(2.2, 3_000), fontsize=SMALL, color=SUCCESS,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=SUCCESS, lw=1.2),
                arrowprops=dict(arrowstyle="->", color=SUCCESS, lw=1.8))
    fig.tight_layout()
    save(fig, "l07-depth")


def fig_grid(depths, leaves, grid):
    fig, ax = plt.subplots(figsize=(10.2, 3.2))
    im = ax.imshow(grid * 100, cmap="Blues", aspect="auto")
    ax.set_xticks(range(len(leaves)), [str(m) for m in leaves])
    ax.set_yticks(range(len(depths)), ["None" if d is None else str(d)
                                       for d in depths])
    ax.set_xlabel("min_samples_leaf")
    ax.set_ylabel("max_depth")
    ax.set_title("5-fold accuracy (%) on the training set")
    ax.grid(False)
    best = np.unravel_index(grid.argmax(), grid.shape)
    for i in range(len(depths)):
        for j in range(len(leaves)):
            v = grid[i, j] * 100
            ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=TICK,
                    color="white" if v > (grid.min() + np.ptp(grid) * 0.6) * 100
                    else "#16212b")
    ax.add_patch(plt.Rectangle((best[1] - .5, best[0] - .5), 1, 1, fill=False,
                               ec=ACCENT, lw=3))
    row = list(depths).index(LEGIBLE_DEPTH)
    ax.add_patch(plt.Rectangle((-.5, row - .5), len(leaves), 1, fill=False,
                               ec=SUCCESS, lw=3, ls="--"))
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    save(fig, "l07-grid")
    return best


def fig_paths(pl):
    free, leg = pl["free"]["lens"], pl["legible"]["lens"]
    fig, ax = plt.subplots(figsize=(11.0, 3.4))
    bins = np.arange(0, max(free.max(), leg.max()) + 2) - 0.5
    ax.hist(free, bins=bins, color=ACCENT, alpha=0.85,
            label=f"unconstrained tree  (mean {free.mean():.1f})")
    ax.hist(leg, bins=bins, color=SUCCESS, alpha=0.85,
            label=f"depth-{LEGIBLE_DEPTH} tree  (mean {leg.mean():.1f})")
    ax.set_yscale("log")
    plain_log(ax, "y", [1, 10, 100, 1_000, 10_000])
    ax.set_ylim(0.7, 40_000)
    ax.set_xlabel("conditions the regulator must read to justify one prediction")
    ax.set_ylabel("test patches  (log scale)")
    ax.set_title("Length of the decision path, 12,000 test predictions")
    ax.legend(loc="upper right")
    ax.annotate("nobody reads this", xy=(free.mean(), 400), xytext=(24, 22),
                fontsize=SMALL, color=ACCENT,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=ACCENT, lw=1.2),
                arrowprops=dict(arrowstyle="->", color=ACCENT, lw=1.8))
    save(fig, "l07-paths")


def fig_confusion(D, tree):
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(D["y_te"], tree.predict(D["X_te"]),
                          labels=range(1, 8), normalize="true")
    fig, ax = plt.subplots(figsize=(7.4, 6.2))
    im = ax.imshow(cm * 100, cmap="Blues", vmin=0, vmax=100)
    ax.set_xticks(range(7), COVER_TYPES, rotation=40, ha="right")
    ax.set_yticks(range(7), COVER_TYPES)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true cover type")
    ax.set_title("Row-normalised, %")
    ax.grid(False)
    for i in range(7):
        for j in range(7):
            v = cm[i, j] * 100
            if v >= 0.5:
                ax.text(j, i, f"{v:.0f}", ha="center", va="center",
                        fontsize=TICK, color="white" if v > 55 else "#16212b")
    fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    save(fig, "l07-confusion")
    return cm


def fig_impurity():
    p = np.linspace(1e-6, 1 - 1e-6, 400)
    gini = 2 * p * (1 - p)
    ent = -(p * np.log2(p) + (1 - p) * np.log2(1 - p))
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 3.7))

    ax = axes[0]
    ax.plot(p, gini, color=PRIMARY, lw=2.4, label="Gini,  2p(1−p)")
    ax.plot(p, ent, color=MATH, lw=2.4, label="entropy,  H(p), in bits")
    ax.plot(p, ent / 2, color=MATH, lw=1.8, ls="--", label="entropy / 2")
    ax.set_xlabel("proportion of the positive class in a node,  p")
    ax.set_ylabel("impurity")
    ax.set_title("Two measures of the same thing")
    ax.legend(loc="lower center", fontsize=TICK)

    ax = axes[1]
    ax.plot(p, ent / 2 - gini, color=ACCENT, lw=2.4)
    ax.axhline(0, color=AXIS, lw=1)
    ax.set_xlabel("p")
    ax.set_ylabel("entropy/2  −  Gini")
    ax.set_title("The gap: same zeros, same maximum, wider shoulders")
    ax.annotate(f"largest gap {np.max(ent/2 - gini):.3f}\nat p = "
                f"{p[np.argmax(ent/2 - gini)]:.2f}",
                xy=(p[np.argmax(ent / 2 - gini)], np.max(ent / 2 - gini)),
                xytext=(0.30, 0.020), fontsize=SMALL, color=ACCENT,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=ACCENT, lw=1.2),
                arrowprops=dict(arrowstyle="->", color=ACCENT, lw=1.8))
    fig.tight_layout()
    save(fig, "l08-impurity")
    return float(np.max(ent / 2 - gini)), float(p[np.argmax(ent / 2 - gini)])


def fig_criteria(pairs):
    g = np.array([r["gini"]["acc"] for r in pairs]) * 100
    e = np.array([r["entropy"]["acc"] for r in pairs]) * 100
    agree = np.array([r["agree"] for r in pairs]) * 100

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 3.7))
    ax = axes[0]
    ax.plot(range(1, 21), g, "-o", ms=4, color=PRIMARY, label="gini")
    ax.plot(range(1, 21), e, "-o", ms=4, color=MATH, label="entropy")
    ax.set_xlabel("resample")
    ax.set_ylabel("test accuracy  (%)")
    ax.set_title("Twenty resamples, both criteria, depth 8")
    ax.set_ylim(72.2, 74.6)
    ax.legend(loc="upper left", ncols=2)

    ax = axes[1]
    ax.hist(e - g, bins=12, color=SUCCESS)
    ax.axvline(0, color=AXIS, lw=1.5)
    ax.axvline((e - g).mean(), color=ACCENT, lw=2.4)
    ax.set_xlabel("entropy − gini, per resample  (percentage points)")
    ax.set_ylabel("resamples")
    ax.set_title(f"Paired difference: mean {(e-g).mean():+.2f} pp, "
                 f"sd {(e-g).std():.2f}")
    fig.tight_layout()
    save(fig, "l08-criteria")
    return g, e, agree


def fig_instability(inst, names):
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.1))

    # left: which question each of the 20 trees asks at each of the seven
    # nodes in its top three levels
    sigs = inst["second"]
    feats = [f for sig in sigs for f, _ in sig]
    seen = sorted({f for f in feats if f != "leaf"},
                  key=lambda f: -feats.count(f))
    palette = [PRIMARY, ACCENT, SUCCESS, MATH, "#b8860b", "#0f7b8a",
               "#8c5a2b", "#7b8794", "#a63a6d", "#3f6d3a", "#5b5bb0"]
    idx = {f: i for i, f in enumerate(seen)}
    M = np.zeros((len(sigs), 7))
    for r, sig in enumerate(sigs):
        for c, (f, _) in enumerate(sig):
            M[r, c] = idx.get(f, len(seen))
    cmap = ListedColormap(
        [palette[i % len(palette)] for i in range(len(seen))] + [RULE])
    ax = axes[0]
    ax.imshow(M, cmap=cmap, aspect="auto", interpolation="nearest",
              vmin=-0.5, vmax=len(seen) + 0.5)
    ax.set_xticks(range(7), ["root", "L", "R", "LL", "LR", "RL", "RR"])
    ax.set_yticks([0, 4, 9, 14, 19], ["1", "5", "10", "15", "20"])
    ax.set_ylabel("tree, one per subsample")
    ax.set_xlabel("node in the top three levels")
    ax.set_title(f"{inst['n_signatures']} distinct structures among 20 trees")
    ax.grid(False)
    # the questions barely move; the thresholds do, and that is where the
    # instability lives
    for c in range(7):
        k = len({round(sig[c][1], 1) for sig in sigs if sig[c][0] != "leaf"})
        ax.text(c, len(sigs) - 0.32, f"{k}", ha="center", va="top",
                fontsize=TICK, color=ACCENT)
    ax.text(-0.62, len(sigs) - 0.32, "distinct\nthresholds:", ha="right",
            va="top", fontsize=TICK, color=ACCENT)
    ax.legend(handles=[Patch(facecolor=palette[i % len(palette)],
                             label=names[f]) for i, f in enumerate(seen[:6])],
              loc="upper left", bbox_to_anchor=(1.005, 1.0), fontsize=TICK,
              frameon=False)

    ax = axes[1]
    d = inst["disagree"] * 100
    ax.hist(d, bins=16, color=ACCENT)
    ax.axvline(d.mean(), color=PRIMARY, lw=2.4)
    ax.set_xlabel("test patches where the two trees disagree  (%)")
    ax.set_ylabel("pairs of trees")
    ax.set_title(f"All 190 pairs: mean {d.mean():.1f}%, "
                 f"range {d.min():.1f}–{d.max():.1f}%")
    fig.tight_layout()
    save(fig, "l08-instability")
    from collections import Counter
    cnt = Counter(f for f, _ in inst["roots"])
    thr = {}
    for f, t in inst["roots"]:
        thr.setdefault(f, []).append(t)
    return cnt, thr


def fig_variance_law():
    n = np.arange(1, 101)
    fig, ax = plt.subplots(figsize=(11.0, 3.4))
    sigma2 = 1.0
    for rho, col in [(0.0, SUCCESS), (0.2, PRIMARY), (0.5, MATH), (0.8, ACCENT)]:
        v = rho * sigma2 + (1 - rho) * sigma2 / n
        ax.plot(n, v, lw=2.4, color=col, label=f"ρ = {rho}")
        ax.axhline(rho * sigma2, color=col, lw=1.1, ls=":")
    ax.set_xlabel("n, the number of averaged predictors")
    ax.set_ylabel("variance of the average  (σ² = 1)")
    ax.set_title("The floor is ρσ². No amount of averaging goes below it")
    ax.set_ylim(0, 1.02)
    ax.legend(loc="upper right", ncols=4, fontsize=TICK)
    ax.annotate("this part vanishes\nlike 1/n", xy=(12, 0.55), xytext=(30, 0.76),
                fontsize=SMALL, color=MUTED,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=RULE, lw=1.2),
                arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.8))
    save(fig, "l08-variance-law")


def fig_rho(dec: dict):
    kinds = KINDS
    label = {"bagging": "Bagging\nbootstrap only",
             "forest": "Random forest\nbootstrap + features",
             "extra": "Extra-trees\nfeatures + thresholds",
             "extra_bs": "Extra-trees, bootstrap=True\nall three"}
    cols = [ACCENT, PRIMARY, MATH, SUCCESS]
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.0),
                             gridspec_kw=dict(wspace=0.34))

    ax = axes[0]
    rho = [dec[k]["rho"] for k in kinds]
    y = np.arange(len(kinds))[::-1]
    ax.barh(y, rho, color=cols, height=0.6)
    ax.set_yticks(y, [label[k] for k in kinds], fontsize=SMALL)
    ax.set_xlabel("measured ρ")
    ax.set_title("Every variant attacks the same term")
    ax.set_xlim(0, max(rho) * 1.30)
    for yy, r in zip(y, rho):
        ax.text(r + max(rho) * 0.02, yy, f"{r:.3f}", va="center",
                fontsize=TICK, color=MUTED)

    ax = axes[1]
    for k, col in zip(kinds, cols):
        c = dec[k]["curve"]
        ns = sorted(c)
        ax.plot(ns, [c[n] for n in ns], "o", ms=7, color=col,
                label=label[k].split("\n")[0])
        nn = np.linspace(1, 22, 150)
        ax.plot(nn, dec[k]["tau2"] + dec[k]["within"] / nn, lw=1.8, color=col,
                alpha=0.7)
        ax.axhline(dec[k]["tau2"], color=col, lw=1.0, ls=":")
    ax.set_yscale("log")
    plain_log(ax, "y", [0.005, 0.01, 0.02, 0.05, 0.1, 0.2], fmt="{:.3f}")
    ax.set_ylim(0.005, 0.28)
    ax.set_xlim(0, 22)
    ax.set_xlabel("n, members averaged")
    ax.set_ylabel("variance of the average  (log)")
    ax.set_title("Points measured, curves are ρσ² + (1−ρ)σ²/n")
    ax.legend(loc="upper right", fontsize=TICK, frameon=True,
              facecolor="white", edgecolor=RULE)
    fig.tight_layout()
    save(fig, "l08-rho")


def fig_ensembles(pools, tree_acc):
    fig, ax = plt.subplots(figsize=(11.0, 3.4))
    label = {"bagging": "Bagging", "forest": "Random forest",
             "extra": "Extra-trees", "extra_bs": "Extra-trees, bootstrap=True"}
    for k, col in zip(KINDS, [ACCENT, PRIMARY, MATH, SUCCESS]):
        a = pools[k]["acc_at"]
        ns = sorted(a)
        ax.plot(ns, [a[n] * 100 for n in ns], "-o", ms=4, color=col,
                label=label[k])
    ax.axhline(tree_acc * 100, color=MUTED, lw=2, ls="--")
    ax.text(3.2, tree_acc * 100 + 0.5, "the legible depth-8 tree, 73.3%",
            color=MUTED, fontsize=SMALL)
    ax.set_ylim(71.5, 91.5)
    ax.set_xscale("log")
    plain_log(ax, "x", [1, 3, 10, 30, 100, 200])
    ax.set_xlim(0.85, 260)
    ax.set_xlabel("n_estimators  (log scale)")
    ax.set_ylabel("test accuracy  (%)")
    ax.set_title("Accuracy against pool size, 12,000 held-out patches")
    ax.legend(loc="center right")
    save(fig, "l08-ensembles")


def fig_importance(ic, names):
    lab = names + ["random_decoy"]
    imp, perm = ic["imp"], ic["perm_mean"]
    order = np.argsort(imp)[::-1][:14]
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.0), sharey=True)

    y = np.arange(len(order))[::-1]
    ax = axes[0]
    cols = [ACCENT if lab[i] == "random_decoy" else PRIMARY for i in order]
    ax.barh(y, imp[order], color=cols, height=0.66)
    ax.set_yticks(y, [lab[i] for i in order])
    ax.set_xlabel("impurity importance")
    ax.set_title("What the forest reports")

    ax = axes[1]
    ax.barh(y, perm[order], color=cols, height=0.66,
            xerr=ic["perm_std"][order], error_kw=dict(ecolor=AXIS, lw=1.2))
    ax.set_xlabel("permutation importance, held-out")
    ax.set_title("What each column is actually worth")
    ax.axvline(0, color=AXIS, lw=1.2)
    dy = int(np.where(np.array([lab[i] for i in order]) == "random_decoy")[0][0])
    ax.annotate("the decoy: +0.001 ± 0.002", xy=(0.004, y[dy]),
                xytext=(0.10, y[dy] - 2.4), fontsize=SMALL, color=ACCENT,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=ACCENT,
                          lw=1.2),
                arrowprops=dict(arrowstyle="->", color=ACCENT, lw=1.8))
    fig.tight_layout()
    save(fig, "l08-importance")


def fig_tradeoff(rows):
    """Accuracy against the number of rules a human would have to read."""
    fig, ax = plt.subplots(figsize=(11.0, 3.5))
    for name, rules, acc, col in rows:
        ax.scatter([rules], [acc * 100], s=150, color=col, zorder=3)
        ax.annotate(f"{name}\n{acc*100:.1f}%", xy=(rules, acc * 100),
                    xytext=(0, 16), textcoords="offset points", ha="center",
                    fontsize=SMALL, color=col)
    ax.set_xscale("log")
    plain_log(ax, "x", [10, 1_000, 100_000, 10_000_000])
    ax.set_xlabel("leaves in the model  —  rules a regulator would have to read "
                  "(log scale)")
    ax.set_ylabel("test accuracy  (%)")
    ax.set_title("The trade the regulator was not offered")
    ax.set_ylim(60, 100)
    ax.set_xlim(3, 3e7)
    save(fig, "l08-tradeoff")


# ---------------------------------------------------------------------- main

BEST_LEAF = 1          # overwritten by the grid before path_lengths() runs


def main():
    global BEST_LEAF
    setup()
    load_cache()

    print("Loading CoverType…")
    D = cached("l07-data", lambda: load_cover())
    from sklearn.tree import DecisionTreeClassifier, export_text

    names = D["names"]
    # Prefixed. These were bare (n_train, n_test, ...) and collided with the
    # housing values make_figures.py writes under the same names — figures.json
    # ended up holding 16,512/4,128 while Lectures 7 and 8 quote 48,000/12,000
    # thirteen times between them. Provenance passed only because those two
    # numbers happened to be reachable by another route.
    facts.update(
        l07_n_rows_total=int(D["X_all_shape"][0]),
        l07_n_features=int(D["X_all_shape"][1]),
        l07_n_classes=7,
        l07_n_sample=int(N_SAMPLE),
        l07_n_train=int(len(D["y_tr"])),
        l07_n_test=int(len(D["y_te"])),
        l07_sample_fraction=round(N_SAMPLE / D["X_all_shape"][0], 4),
    )

    print("\nLecture 7 — class balance and the trivial baseline")
    counts, share = fig_class_balance(D)
    maj = int(np.argmax(counts)) + 1
    baseline = float((D["y_te"] == maj).mean())
    facts.update(
        class_counts={COVER_TYPES[i]: int(counts[i]) for i in range(7)},
        class_share={COVER_TYPES[i]: round(float(share[i]), 4) for i in range(7)},
        majority_class=COVER_TYPES[maj - 1],
        baseline_accuracy=round(baseline, 4),
        rarest_share=round(float(share.min()), 4),
        rarest_count=int(counts.min()),
    )

    print("\nLecture 7 — depth sweep")
    depths, rows = cached("l07-depth", lambda: sweep_depth(D))
    fig_depth(depths, rows)
    by_depth = {r["depth"]: r for r in rows}
    facts.update(
        depth_curve={str(r["depth"]): dict(train=round(r["train"], 4),
                                           cv=round(r["cv"], 4),
                                           leaves=r["leaves"]) for r in rows},
        cv_at_legible=round(by_depth[LEGIBLE_DEPTH]["cv"], 4),
        leaves_at_legible=int(by_depth[LEGIBLE_DEPTH]["leaves"]),
        cv_at_depth_18=round(by_depth[18]["cv"], 4),
        leaves_at_depth_18=int(by_depth[18]["leaves"]),
        cv_at_depth_1=round(by_depth[1]["cv"], 4),
        cv_at_depth_3=round(by_depth[3]["cv"], 4),
        leaves_at_depth_3=int(by_depth[3]["leaves"]),
    )

    print("\nLecture 7 — the 2-D grid")
    gd, gl, grid = cached("l07-grid", lambda: grid_2d(D))
    best = fig_grid(gd, gl, grid)
    row = gd.index(LEGIBLE_DEPTH)
    BEST_LEAF = gl[int(grid[row].argmax())]
    facts.update(
        grid_best_depth="None" if gd[best[0]] is None else int(gd[best[0]]),
        grid_best_leaf=int(gl[best[1]]),
        grid_best_cv=round(float(grid.max()), 4),
        legible_best_leaf=int(BEST_LEAF),
        legible_best_cv=round(float(grid[row].max()), 4),
        grid_worst_cv=round(float(grid.min()), 4),
    )
    print(f"    best under the cap: min_samples_leaf={BEST_LEAF}")

    print("\nLecture 7 — the two trees, and their decision paths")
    pl = cached(f"l07-paths-{BEST_LEAF}", lambda: path_lengths(D))
    fig_paths(pl)
    facts.update(
        free_depth=pl["free"]["depth"], free_leaves=pl["free"]["leaves"],
        free_nodes=pl["free"]["nodes"],
        free_train_acc=round(pl["free"]["train"], 4),
        free_test_acc=round(pl["free"]["test"], 4),
        free_path_mean=round(float(pl["free"]["lens"].mean()), 2),
        free_path_max=int(pl["free"]["lens"].max()),
        legible_depth=LEGIBLE_DEPTH,
        legible_leaves=pl["legible"]["leaves"],
        legible_nodes=pl["legible"]["nodes"],
        legible_train_acc=round(pl["legible"]["train"], 4),
        legible_test_acc=round(pl["legible"]["test"], 4),
        legible_path_mean=round(float(pl["legible"]["lens"].mean()), 2),
        legible_path_max=int(pl["legible"]["lens"].max()),
    )
    facts["legible_features_used"] = int(len(
        {int(f) for f in tree_features(D, LEGIBLE_DEPTH, BEST_LEAF)}))

    tree = DecisionTreeClassifier(max_depth=LEGIBLE_DEPTH,
                                  min_samples_leaf=BEST_LEAF,
                                  random_state=SEED).fit(D["X_tr"], D["y_tr"])
    fig_tree(D, tree)
    cm = fig_confusion(D, tree)
    facts.update(
        recall_by_class={COVER_TYPES[i]: round(float(cm[i, i]), 4)
                         for i in range(7)},
        worst_class=COVER_TYPES[int(np.argmin(np.diag(cm)))],
        worst_class_recall=round(float(np.diag(cm).min()), 4),
    )

    # ---- the traced prediction -------------------------------------------
    t_ = tree.tree_
    node, trace = 0, []
    idx = int(np.where(D["y_te"] == 7)[0][0])       # a Krummholz patch
    x = D["X_te"][idx]
    while t_.children_left[node] != -1:
        f, thr = int(t_.feature[node]), float(t_.threshold[node])
        go_left = x[f] <= thr
        trace.append((names[f], thr, float(x[f]), go_left))
        node = t_.children_left[node] if go_left else t_.children_right[node]
    # every integer a slide prints with a thousands separator has to be
    # reachable from figures.json, code blocks included — check_provenance.py
    # does not blank <pre>
    facts.update(
        quoted_thresholds=[int(round(t)) for _, t, _, _ in trace],
        quoted_values=sorted({int(round(v)) for _, _, v, _ in trace}),
        tree_root_threshold=int(round(float(t_.threshold[0]))),
        tree_level2_thresholds=[
            int(round(float(t_.threshold[t_.children_left[0]]))),
            int(round(float(t_.threshold[t_.children_right[0]])))],
        var_experiment_K=20, var_experiment_m=20,
        var_experiment_rows_per_set=20_000,
        var_experiment_rows_total=400_000,
        var_experiment_test=12_000,
        trace_n_conditions=len(trace),
        trace_leaf_samples=int(round(t_.n_node_samples[node])),
        trace_leaf_purity=round(float(t_.value[node][0].max()), 4),
        trace_true_class=COVER_TYPES[6],
    )

    print("\nLecture 8 — impurity")
    gap, gap_at = fig_impurity()
    facts.update(impurity_gap=round(gap, 4), impurity_gap_at=round(gap_at, 3))

    print("\nLecture 8 — gini against entropy")
    pairs = cached("l08-criteria", lambda: criterion_pairs(D))
    g, e, agree = fig_criteria(pairs)
    facts.update(
        gini_acc_mean=round(float(g.mean()) / 100, 4),
        entropy_acc_mean=round(float(e.mean()) / 100, 4),
        criterion_diff_mean=round(float((e - g).mean()) / 100, 5),
        criterion_diff_sd=round(float((e - g).std()) / 100, 5),
        criterion_agreement=round(float(agree.mean()) / 100, 4),
        criterion_same_root=int(sum(r["same_root"] for r in pairs)),
        criterion_entropy_wins=int((e > g).sum()),
    )

    print("\nLecture 8 — instability")
    inst = cached(f"l08-instability-{BEST_LEAF}", lambda: instability(D))
    cnt, thr = fig_instability(inst, names)
    top_feat, top_n = cnt.most_common(1)[0]
    facts.update(
        instability_root_variants=len(cnt),
        instability_root_top=names[top_feat],
        instability_root_top_n=int(top_n),
        instability_thr_lo=round(float(min(thr[top_feat])), 1),
        instability_thr_hi=round(float(max(thr[top_feat])), 1),
        instability_thr_span=round(float(max(thr[top_feat]) - min(thr[top_feat])), 1),
        instability_disagree_mean=round(float(inst["disagree"].mean()), 4),
        instability_disagree_max=round(float(inst["disagree"].max()), 4),
        instability_disagree_min=round(float(inst["disagree"].min()), 4),
        instability_acc_mean=round(float(inst["accs"].mean()), 4),
        instability_acc_sd=round(float(inst["accs"].std()), 4),
        instability_acc_lo=round(float(inst["accs"].min()), 4),
        instability_acc_hi=round(float(inst["accs"].max()), 4),
        instability_unanimous=round(float(inst["unanimous"]), 4),
        instability_signatures=int(inst["n_signatures"]),
        instability_n_thresholds=len({round(t, 1) for _, t in inst["roots"]}),
        instability_pair_worst=round(float(inst["disagree"].max()), 4),
        instability_leaves_lo=int(min(inst["leaves"])),
        instability_leaves_hi=int(max(inst["leaves"])),
        instability_nfeat_lo=int(min(inst["nfeat"])),
        instability_nfeat_hi=int(max(inst["nfeat"])),
        instability_thresholds_per_node=[
            len({round(sig[c][1], 1) for sig in inst["second"]
                 if sig[c][0] != "leaf"}) for c in range(7)],
    )

    print("\nLecture 8 — the three ensembles")
    fig_variance_law()
    pools, dec = {}, {}
    for kind in KINDS:
        pools[kind] = cached(f"l08-pool-{kind}", lambda k=kind: ensemble_pool(D, k))
        S = cached(f"l08-var-{kind}", lambda k=kind: variance_experiment(k))
        dec[kind] = decompose(S)
    fig_rho(dec)
    fig_ensembles(pools, pl["legible"]["test"])
    for kind in pools:
        facts[f"{kind}_acc"] = round(pools[kind]["acc"], 4)
        facts[f"{kind}_acc_at_10"] = round(pools[kind]["acc_at"][10], 4)
        facts[f"{kind}_rho"] = round(dec[kind]["rho"], 4)
        facts[f"{kind}_sigma2"] = round(dec[kind]["sigma2"], 4)
        facts[f"{kind}_tau2"] = round(dec[kind]["tau2"], 5)
        facts[f"{kind}_within"] = round(dec[kind]["within"], 5)
        facts[f"{kind}_var_1"] = round(dec[kind]["curve"][1], 5)
        facts[f"{kind}_var_20"] = round(dec[kind]["curve"][20], 5)
        facts[f"{kind}_var_20_pred"] = round(dec[kind]["predicted"][20], 5)
        facts[f"{kind}_var_floor"] = round(dec[kind]["floor"], 5)
        facts[f"{kind}_var_drop"] = round(
            1 - dec[kind]["curve"][20] / dec[kind]["curve"][1], 4)
        facts[f"{kind}_member_acc"] = round(dec[kind]["member_acc"], 4)
        facts[f"{kind}_single_tree_acc"] = round(
            float(pools[kind]["correct"][0].mean()), 4)

    print("\nLecture 8 — feature importance and the decoy column")
    ic = cached("l08-importance", lambda: importance_check(D))
    fig_importance(ic, names)
    facts.update(
        decoy_rank=int(ic["rank"]), decoy_of=int(ic["n_cols"]),
        decoy_impurity_importance=round(ic["decoy_imp"], 4),
        decoy_permutation_importance=round(ic["decoy_perm"], 5),
        decoy_forest_acc=round(ic["acc"], 4),
        top_importance_feature=names[int(np.argmax(ic["imp"][:len(names)]))],
        top_importance_value=round(float(ic["imp"][:len(names)].max()), 4),
        n_below_decoy=int((ic["imp"] < ic["decoy_imp"]).sum()),
    )

    # forest leaves, for the trade-off figure
    from sklearn.ensemble import RandomForestClassifier
    rf_leaves = cached("l08-rf-leaves", lambda: int(sum(
        t.get_n_leaves() for t in RandomForestClassifier(
            n_estimators=N_TREES, max_features="sqrt", random_state=SEED,
            n_jobs=-1).fit(D["X_tr"], D["y_tr"]).estimators_)))
    facts["forest_total_leaves"] = rf_leaves
    d3 = DecisionTreeClassifier(max_depth=3, random_state=SEED).fit(
        D["X_tr"], D["y_tr"])
    facts["depth3_test_acc"] = round(float(d3.score(D["X_te"], D["y_te"])), 4)
    fig_tradeoff([
        ("depth 3", by_depth[3]["leaves"], facts["depth3_test_acc"], SUCCESS),
        (f"depth {LEGIBLE_DEPTH}", pl["legible"]["leaves"],
         pl["legible"]["test"], SUCCESS),
        ("unconstrained tree", pl["free"]["leaves"], pl["free"]["test"], MATH),
        ("random forest, 200 trees", rf_leaves, pools["forest"]["acc"], ACCENT),
    ])

    # ------------------------------------------------------------- report
    print("\n" + "=" * 70)
    print("=== BLOCKS FOR THE SLIDES ===")
    print("=" * 70)

    print("\n--- L7: the tree, as text (depth 3) ---")
    txt = export_text(tree, feature_names=names, class_names=COVER_TYPES,
                      max_depth=2, show_weights=False, decimals=0)
    for line in txt.splitlines()[:16]:
        print(line[:78])

    print("\n--- L7: a traced prediction ---")
    print(f">>> justify(tree, X_test[{idx}])")
    for i, (f, t, v, left) in enumerate(trace, 1):
        rel = "<=" if left else ">"
        print(f"{i}. {f} = {v:,.0f}  {rel} {t:,.0f}")
    print(f"-> {COVER_TYPES[int(np.argmax(t_.value[node][0]))]} "
          f"({t_.value[node][0].max()*100:.0f}% of "
          f"{int(round(t_.n_node_samples[node]))} training patches in this leaf)")
    print(f"   true class: {COVER_TYPES[int(D['y_te'][idx]) - 1]}")

    print("\n--- L7: the two trees compared ---")
    for tag in ("free", "legible"):
        p = pl[tag]
        print(f"{tag:8s} depth {p['depth']:3d}  leaves {p['leaves']:6,d}  "
              f"train {p['train']*100:5.1f}%  test {p['test']*100:5.1f}%  "
              f"mean path {p['lens'].mean():5.2f}")

    print("\n--- L8: variance decomposition ---")
    print(f"{'':10s} {'rho':>7s} {'sigma^2':>9s} {'tau^2':>9s} "
          f"{'V(1)':>9s} {'V(20)':>9s} {'pred(20)':>9s} {'member':>8s} "
          f"{'200-tree':>9s}")
    for k in KINDS:
        d = dec[k]
        print(f"{k:10s} {d['rho']:7.3f} {d['sigma2']:9.5f} {d['tau2']:9.5f} "
              f"{d['curve'][1]:9.5f} {d['curve'][20]:9.5f} "
              f"{d['predicted'][20]:9.5f} {d['member_acc']*100:7.1f}% "
              f"{pools[k]['acc']*100:8.1f}%")

    print("\n--- L8: instability ---")
    print(f"root feature over 20 subsamples: "
          f"{ {names[f]: c for f, c in cnt.most_common()} }")
    print(f"pairwise disagreement: mean {inst['disagree'].mean()*100:.1f}%  "
          f"range {inst['disagree'].min()*100:.1f}–"
          f"{inst['disagree'].max()*100:.1f}%")
    print(f"accuracy: {inst['accs'].mean()*100:.2f}% ± "
          f"{inst['accs'].std()*100:.2f}%  "
          f"({inst['accs'].min()*100:.2f}–{inst['accs'].max()*100:.2f}%)")
    print(f"distinct top-3-level structures: {inst['n_signatures']} of 20; "
          f"distinct root thresholds: "
          f"{len({round(t,1) for _, t in inst['roots']})}")
    print(f"all 20 trees agree on {inst['unanimous']*100:.1f}% of test patches")
    print(f"leaves per tree {min(inst['leaves'])}–{max(inst['leaves'])}; "
          f"columns consulted {min(inst['nfeat'])}–{max(inst['nfeat'])}")
    print(f"single-tree accuracies inside the pools: bagging "
          f"{facts['bagging_single_tree_acc']*100:.1f}%, forest "
          f"{facts['forest_single_tree_acc']*100:.1f}%, extra "
          f"{facts['extra_single_tree_acc']*100:.1f}%")

    print("\n--- L8: the decoy column ---")
    print(f"random_decoy ranks {ic['rank']} of {ic['n_cols']} by impurity "
          f"importance ({ic['decoy_imp']:.4f}); "
          f"{int((ic['imp'] < ic['decoy_imp']).sum())} real columns rank below it")
    print(f"its permutation importance on held-out data: "
          f"{ic['decoy_perm']:+.5f} ± {ic['perm_std'][-1]:.5f}")

    export(**facts)

    if (problems := check_text_floor()):
        print("\nfigures whose text lands under the on-slide floor:")
        for p_ in problems:
            print("  " + p_)
        raise SystemExit(1)
    print("\ntext floor: clean")


if __name__ == "__main__":
    main()
