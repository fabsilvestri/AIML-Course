#!/usr/bin/env python3
"""
Application 3 — Titanic survival with calibrated probabilities. Lectures 5 & 6.

    python3 tools/figures_app03.py

Every figure named l05-* or l06-*, and every number those two decks print, is
produced here. `figkit` re-exports the shared machinery from make_figures.py so
that two authors never edit one script; `figkit.export()` merges into
assets/figures/figures.json rather than overwriting it.

The modelling path is the one the slides and the notebooks describe, in the same
order, with the same seed:

    train.csv (891 labelled rows)  ->  stratified 80/20 split  ->  712 / 179
    engineer Title, FamilySize, IsAlone, Deck
    numeric   : median impute -> standardise -> PolynomialFeatures(degree)
    categorical: most-frequent impute -> one-hot
    model     : LogisticRegression(C=1e6)   # "as close to unregularised as lbfgs gets"

C is set high on purpose in Lecture 5: the brief asks for coefficients that can
be read as odds ratios, and shrunk coefficients cannot be. That decision is what
Lecture 6 then has to undo.
"""

from __future__ import annotations

import tarfile
import urllib.request
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from figkit import (ACCENT, AXIS, BODY, MATH, MUTED, OUT, PRIMARY, RULE, SEED,
                    SMALL, SUCCESS, TICK, cached, check_text_floor, export,
                    load_cache, plain_log, save, setup)

CACHE = Path("/private/tmp/claude-501/aiml-data")
SOFT = "#f4f7f9"

# The two operating costs the brief states, in escort-units. Deliberately not
# money: this deck has enough dollar signs to lose already (TRICKS §9.1).
COST_FN = 10.0      # a passenger who did not survive and was never flagged
COST_FP = 1.0       # an escort assigned to a passenger who survived anyway

DEGREES = [1, 2, 3, 4, 5, 6]
N_SPLITS = 10
N_BOOT = 200        # training sets drawn for the bias/variance measurement
BOOT_N = 400        # rows in each, drawn without replacement from the 712


# ------------------------------------------------------------------- the data

def load_titanic() -> pd.DataFrame:
    """Exactly the loader shown on the slides, with a cache outside the repo."""
    tarball = CACHE / "titanic.tgz"
    if not tarball.is_file():
        url = "https://github.com/ageron/data/raw/main/titanic.tgz"
        print(f"  downloading {url}")
        CACHE.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, tarball)
        with tarfile.open(tarball) as t:
            t.extractall(path=CACHE, filter="data")
    return pd.read_csv(CACHE / "titanic" / "train.csv")


def engineer(d: pd.DataFrame) -> pd.DataFrame:
    """Four engineered columns. Each one is defended on a slide."""
    d = d.copy()
    d["Title"] = (d["Name"].str.extract(r",\s*([^\.]+)\.", expand=False)
                  .str.strip()
                  .replace({"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"}))
    d["Title"] = d["Title"].where(
        d["Title"].isin(["Mr", "Mrs", "Miss", "Master"]), "Rare")
    d["FamilySize"] = d["SibSp"] + d["Parch"] + 1
    d["IsAlone"] = (d["FamilySize"] == 1).astype(int)
    d["Deck"] = d["Cabin"].str[0].fillna("U")
    return d


# Two column sets, because the deck builds the wrong one first and then
# repairs it. v1 is what an assistant writes; v2 is what survives the review.
NUM_V1 = ["Age", "Fare", "FamilySize", "SibSp", "Parch"]
NUM = ["Age", "Fare", "SibSp", "Parch"]      # FamilySize = SibSp + Parch + 1
CAT = ["Pclass", "Sex", "Embarked", "Title", "Deck"]
BIN = ["IsAlone"]


def split(full: pd.DataFrame):
    from sklearn.model_selection import train_test_split
    X = full[NUM_V1 + CAT + BIN]
    y = full["Survived"]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y)
    return X_tr, X_te, y_tr, y_te


def prep_v1():
    """The obvious preprocessing — and the one with two exact dependencies.

    Every one-hot block sums to 1, which is the intercept column; and
    FamilySize = SibSp + Parch + 1. Both make X rank-deficient, which is
    Thread 1's failure condition on the students' own matrix.
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    return ColumnTransformer([
        ("num", make_pipeline(SimpleImputer(strategy="median"),
                              StandardScaler()), NUM_V1),
        ("cat", make_pipeline(SimpleImputer(strategy="most_frequent"),
                              OneHotEncoder(handle_unknown="ignore")), CAT),
        ("bin", "passthrough", BIN)])


def prep(degree: int = 1):
    """The repaired preprocessing, and the one everything after it uses.

    `drop="first"` removes one level per categorical column; FamilySize leaves
    the numeric block. The polynomial expansion acts on the four numeric
    columns only — the one-hot columns are already their own indicators, and
    squaring an indicator returns the indicator.
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import (OneHotEncoder, PolynomialFeatures,
                                       StandardScaler)
    num = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                        PolynomialFeatures(degree=degree, include_bias=False))
    cat = make_pipeline(SimpleImputer(strategy="most_frequent"),
                        OneHotEncoder(drop="first", handle_unknown="ignore"))
    return ColumnTransformer([("num", num, NUM), ("cat", cat, CAT),
                              ("bin", "passthrough", BIN)])


def _logreg(C, penalty, solver, l1_ratio, max_iter):
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(C=C, penalty=penalty, solver=solver,
                              l1_ratio=l1_ratio, max_iter=max_iter,
                              random_state=SEED)


def model_v1(C=1e6):
    from sklearn.pipeline import Pipeline
    return Pipeline([("prep", prep_v1()),
                     ("clf", _logreg(C, "l2", "lbfgs", None, 4000))])


def model(degree=1, C=1e6, penalty="l2", solver="lbfgs", l1_ratio=None,
          max_iter=4000):
    from sklearn.pipeline import Pipeline
    return Pipeline([("prep", prep(degree)),
                     ("clf", _logreg(C, penalty, solver, l1_ratio, max_iter))])


def cv_splitter():
    from sklearn.model_selection import StratifiedKFold
    return StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)


def quiet(fn):
    """Run fn with sklearn's convergence chatter silenced.

    Where the warning is itself the lesson — the unregularised fit at high
    degree — it is *counted* rather than suppressed; see `separation_check`.
    """
    def inner(*a, **k):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return fn(*a, **k)
    return inner


# =========================================================== LECTURE 5 · BUILD

def describe(full, X_tr, X_te, y_tr, y_te) -> dict:
    n = len(full)
    miss = full[["Age", "Cabin", "Embarked"]].isna().sum()
    base = float(y_tr.mean())
    p = base
    return {
        "n_rows": n,
        "n_train": int(len(X_tr)),
        "n_test": int(len(X_te)),
        "n_columns": int(full.shape[1]),
        "missing": {"Age": int(miss["Age"]), "Cabin": int(miss["Cabin"]),
                    "Embarked": int(miss["Embarked"])},
        "missing_pct": {"Age": 100.0 * miss["Age"] / n,
                        "Cabin": 100.0 * miss["Cabin"] / n,
                        "Embarked": 100.0 * miss["Embarked"] / n},
        "n_survived": int(full["Survived"].sum()),
        "n_died": int((1 - full["Survived"]).sum()),
        "base_rate": float(full["Survived"].mean()),
        "base_rate_train": base,
        "base_rate_test": float(y_te.mean()),
        "majority_accuracy_train": 1.0 - base,
        "majority_accuracy_test": 1.0 - float(y_te.mean()),
        # the constant-probability model: predict the training base rate for
        # everyone. This is the anchor the commitment is estimated against.
        "constant_log_loss": float(-(p * np.log(p) + (1 - p) * np.log(1 - p))),
        "constant_brier": float(p * (1 - p)),
    }


def survival_slices(full) -> dict:
    g = full.groupby(["Sex", "Pclass"])["Survived"].agg(["mean", "count"])
    t = full.groupby("Title")["Survived"].agg(["mean", "count"])
    return {
        "by_sex_class": {f"{s}-{c}": {"rate": float(r["mean"]),
                                      "n": int(r["count"])}
                         for (s, c), r in g.iterrows()},
        "by_title": {k: {"rate": float(r["mean"]), "n": int(r["count"])}
                     for k, r in t.iterrows()},
        "by_sex": {k: float(v) for k, v in
                   full.groupby("Sex")["Survived"].mean().items()},
    }


def fig_missing(full):
    counts = full.isna().sum().sort_values(ascending=False)
    counts = counts[counts > 0]
    fig, ax = plt.subplots(figsize=(10.6, 3.6))
    bars = ax.barh(counts.index[::-1], counts.values[::-1], color=PRIMARY,
                   height=0.6)
    n = len(full)
    for rect, v in zip(bars, counts.values[::-1]):
        ax.text(v + n * 0.012, rect.get_y() + rect.get_height() / 2,
                f"{v}  ({100 * v / n:.0f}%)", va="center", fontsize=SMALL,
                fontweight="bold", color=PRIMARY)
    ax.set_xlim(0, n)
    ax.axvline(n, color=RULE, lw=1.2)
    ax.text(n, -0.75, f"all {n} passengers", ha="right", va="bottom",
            fontsize=SMALL, color=MUTED)
    ax.set_xlabel("passengers with the value missing")
    ax.set_title("Three columns have holes; one of them is mostly hole")
    ax.grid(axis="y", alpha=0)
    fig.tight_layout()
    return save(fig, "l05-missing")


def fig_survival_rates(full, facts):
    order = [("female", 1), ("female", 2), ("female", 3),
             ("male", 1), ("male", 2), ("male", 3)]
    rates = [facts["by_sex_class"][f"{s}-{c}"]["rate"] for s, c in order]
    ns = [facts["by_sex_class"][f"{s}-{c}"]["n"] for s, c in order]
    labels = [f"{s}\nclass {c}" for s, c in order]
    colours = [SUCCESS] * 3 + [ACCENT] * 3
    fig, ax = plt.subplots(figsize=(10.8, 4.2))
    bars = ax.bar(labels, rates, color=colours, width=0.62)
    for rect, r, k in zip(bars, rates, ns):
        ax.text(rect.get_x() + rect.get_width() / 2, r + 0.028,
                f"{100 * r:.0f}%", ha="center", fontsize=SMALL,
                fontweight="bold", color="#16212b")
        ax.text(rect.get_x() + rect.get_width() / 2, 0.035, f"n = {k}",
                ha="center", fontsize=SMALL, color="white", fontweight="bold")
    b = full["Survived"].mean()
    ax.axhline(b, color=MATH, lw=2, ls="--")
    ax.text(5.48, b + 0.02, f"everyone: {100 * b:.0f}%", ha="right",
            fontsize=SMALL, color=MATH, fontweight="bold")
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("survived")
    ax.yaxis.set_major_formatter(lambda v, _: f"{100 * v:.0f}%")
    ax.set_title("Two columns already explain most of it")
    ax.grid(axis="x", alpha=0)
    fig.tight_layout()
    return save(fig, "l05-survival-rates")


def fig_sigmoid():
    t = np.linspace(-8, 8, 400)
    s = 1 / (1 + np.exp(-t))
    fig, ax = plt.subplots(figsize=(10.6, 4.0))
    ax.plot(t, s, color=PRIMARY, lw=3)
    ax.axhline(0.5, color=RULE, lw=1.2, ls=":")
    ax.axvline(0, color=RULE, lw=1.2, ls=":")
    ax.plot([0], [0.5], "o", color=ACCENT, ms=10, zorder=5)
    ax.annotate("t = 0  ⇒  p = 0.5\nequal odds", xy=(0, 0.5),
                xytext=(-7.6, 0.74), fontsize=SMALL, color=ACCENT,
                fontweight="bold",
                bbox=dict(fc="white", ec=ACCENT, lw=1.2, boxstyle="round,pad=0.4"),
                arrowprops=dict(arrowstyle="->", color=ACCENT, lw=1.8))
    ax.annotate("saturates: a large change in t\nbarely moves p",
                xy=(6.0, 1 / (1 + np.exp(-6.0))), xytext=(1.1, 0.30),
                fontsize=SMALL, color=MUTED,
                bbox=dict(fc="white", ec=RULE, lw=1.2, boxstyle="round,pad=0.4"),
                arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.8))
    ax.set_xlabel("t  =  θ·x   (the log-odds)")
    ax.set_ylabel("σ(t)  =  estimated probability")
    ax.set_ylim(-0.04, 1.1)
    ax.set_title("The logistic function turns a real number into a probability")
    fig.tight_layout()
    return save(fig, "l05-sigmoid")


def fit_base(X_tr, y_tr):
    """The first fit of the session: the naive encoding, degree 1."""
    return quiet(lambda: model_v1().fit(X_tr, y_tr))()


def coefficients_v1(m) -> dict:
    """The weights as first fitted — before anyone checks the rank."""
    names = [n.split("__")[-1] for n in m[:-1].get_feature_names_out()]
    coefs = m[-1].coef_[0]
    order = np.argsort(-np.abs(coefs))
    return {"names": [names[i] for i in order],
            "coef": [float(coefs[i]) for i in order],
            "odds": [float(np.exp(coefs[i])) for i in order],
            "intercept": float(m[-1].intercept_[0]),
            "n_features": int(len(coefs))}


def rank_check(X_tr, y_tr) -> dict:
    """Thread 1's failure condition, measured on the students' own matrix."""
    def design(m):
        Z = np.asarray(m[:-1].transform(X_tr), dtype=float)
        return np.c_[np.ones(len(Z)), Z]          # with the intercept column

    m1 = quiet(lambda: model_v1().fit(X_tr, y_tr))()
    m2 = quiet(lambda: model(degree=1).fit(X_tr, y_tr))()
    A, B = design(m1), design(m2)
    sa = np.linalg.svd(A, compute_uv=False)
    sb = np.linalg.svd(B, compute_uv=False)
    return {
        "v1_columns": int(A.shape[1]),
        "v1_rank": int(np.linalg.matrix_rank(A)),
        "v1_deficiency": int(A.shape[1] - np.linalg.matrix_rank(A)),
        "v1_smallest_singular_value": float(sa.min()),
        "v1_condition_number": float(sa.max() / sa.min()),
        "v2_columns": int(B.shape[1]),
        "v2_rank": int(np.linalg.matrix_rank(B)),
        "v2_condition_number": float(sb.max() / sb.min()),
        "family_residual": float(
            np.abs(X_tr["SibSp"] + X_tr["Parch"] + 1 - X_tr["FamilySize"]).max()),
        "n_categorical_columns": len(CAT),
    }


def coefficients_v2(X_tr, y_tr) -> dict:
    """The coefficients that can actually be read, and their odds ratios."""
    from sklearn.model_selection import cross_val_score
    m = quiet(lambda: model(degree=1).fit(X_tr, y_tr))()
    names = [n.split("__")[-1] for n in m[:-1].get_feature_names_out()]
    coefs = m[-1].coef_[0]
    order = np.argsort(-np.abs(coefs))
    ll = quiet(cross_val_score)(model(degree=1), X_tr, y_tr, cv=cv_splitter(),
                                scoring="neg_log_loss", n_jobs=-1)
    return {"names": [names[i] for i in order],
            "coef": [float(coefs[i]) for i in order],
            "odds": [float(np.exp(coefs[i])) for i in order],
            "intercept": float(m[-1].intercept_[0]),
            "n_features": int(len(coefs)),
            "cv_log_loss": float(-ll.mean()),
            "reference_levels": {"Pclass": 1, "Sex": "female",
                                 "Embarked": "C", "Title": "Master",
                                 "Deck": "A"}}


def coef_ambiguity(X_tr, y_tr, shift=2.5) -> dict:
    """Move along the null space and nothing observable changes.

    Sex_female + Sex_male = 1 for every row, and the intercept column is also 1
    - so the vector (intercept -1, Sex_female +1, Sex_male +1) is annihilated by
    the design matrix. Add any multiple of it to the fitted weights and every
    logit, and therefore every predicted probability, is unchanged **exactly**.
    """
    m = quiet(lambda: model_v1().fit(X_tr, y_tr))()
    names = [n.split("__")[-1] for n in m[:-1].get_feature_names_out()]
    Z = np.asarray(m[:-1].transform(X_tr), dtype=float)
    theta = m[-1].coef_[0].copy()
    b0 = float(m[-1].intercept_[0])

    v = np.zeros_like(theta)
    v[names.index("Sex_female")] = 1.0
    v[names.index("Sex_male")] = 1.0
    theta2 = theta + shift * v
    b02 = b0 - shift

    logit1 = Z @ theta + b0
    logit2 = Z @ theta2 + b02
    p1 = 1 / (1 + np.exp(-logit1))
    p2 = 1 / (1 + np.exp(-logit2))
    return {
        "shift": shift,
        "names": ["intercept", "Sex_female", "Sex_male"],
        "coef_a": [b0, float(theta[names.index("Sex_female")]),
                   float(theta[names.index("Sex_male")])],
        "coef_b": [b02, float(theta2[names.index("Sex_female")]),
                   float(theta2[names.index("Sex_male")])],
        "max_abs_logit_difference": float(np.abs(logit1 - logit2).max()),
        "max_abs_prob_difference": float(np.abs(p1 - p2).max()),
        "penalty_a": float(np.sum(theta ** 2)),
        "penalty_b": float(np.sum(theta2 ** 2)),
    }


def fig_coef_ambiguity(amb):
    keys, a, b = amb["names"], amb["coef_a"], amb["coef_b"]
    x = np.arange(len(keys)); w = 0.36
    fig, ax = plt.subplots(figsize=(10.4, 4.0))
    r1 = ax.bar(x - w / 2, a, w, color=PRIMARY, label="the fit you got")
    r2 = ax.bar(x + w / 2, b, w, color=ACCENT,
                label=f"the same fit, shifted by {amb['shift']:.1f} along the "
                      f"null space")
    for rects, vals in ((r1, a), (r2, b)):
        for rect, v in zip(rects, vals):
            ax.text(rect.get_x() + rect.get_width() / 2,
                    v + (0.20 if v >= 0 else -0.20), f"{v:+.2f}", ha="center",
                    va="bottom" if v >= 0 else "top", fontsize=SMALL,
                    fontweight="bold", color="#16212b")
    ax.axhline(0, color=AXIS, lw=1.4)
    ax.set_xticks(x, keys)
    ax.set_ylabel("coefficient (log-odds)")
    lo, hi = min(min(a), min(b)), max(max(a), max(b))
    ax.set_ylim(lo - 1.6, hi + 2.2)
    ax.set_title("Every predicted probability agrees to "
                 f"{amb['max_abs_prob_difference']:.0e}")
    ax.grid(axis="x", alpha=0)
    ax.legend(loc="upper left")
    fig.tight_layout()
    return save(fig, "l05-coef-ambiguity")


def fig_coefficients(co):
    keep = ["Sex_male", "Title_Mr", "Title_Mrs", "Title_Miss", "Title_Rare",
            "Pclass_2", "Pclass_3", "Age", "Fare", "SibSp", "Parch",
            "IsAlone", "Deck_U", "Embarked_S"]
    idx = [co["names"].index(k) for k in keep if k in co["names"]]
    names = [co["names"][i] for i in idx]
    vals = [co["coef"][i] for i in idx]
    order = np.argsort(vals)
    names = [names[i] for i in order]
    vals = [vals[i] for i in order]
    colours = [ACCENT if v < 0 else SUCCESS for v in vals]
    fig, ax = plt.subplots(figsize=(10.6, 5.0))
    ax.barh(names, vals, color=colours, height=0.66)
    span = max(abs(min(vals)), abs(max(vals)))
    for y_, v in zip(range(len(vals)), vals):
        off = span * 0.03
        ax.text(v + (off if v > 0 else -off), y_,
                f"×{np.exp(v):.2f}", va="center",
                ha="left" if v > 0 else "right", fontsize=SMALL,
                fontweight="bold", color="#16212b")
    ax.axvline(0, color=AXIS, lw=1.4)
    ax.set_xlim(-span * 1.42, span * 1.42)
    ax.set_xlabel("coefficient (log-odds)   ·   label: multiplier on the odds")
    ax.set_title("Each weight is a multiplier on the odds, not on the probability")
    ax.grid(axis="y", alpha=0)
    fig.tight_layout()
    return save(fig, "l05-coefficients")


def boundary_model(X_tr, y_tr):
    """A two-feature model, purely so the boundary can be drawn honestly."""
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    cols = ["Age", "Fare"]
    m = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                      LogisticRegression(C=1e6, max_iter=4000))
    m.fit(X_tr[cols], y_tr)
    return m, cols


def fig_boundary(X_tr, y_tr):
    m, cols = boundary_model(X_tr, y_tr)
    a = np.linspace(0, 80, 260)
    f = np.linspace(0, 300, 260)
    A, F = np.meshgrid(a, f)
    grid = pd.DataFrame({"Age": A.ravel(), "Fare": F.ravel()})
    P = m.predict_proba(grid)[:, 1].reshape(A.shape)

    fig, ax = plt.subplots(figsize=(10.4, 4.9))
    cs = ax.contourf(A, F, P, levels=np.linspace(0, 1, 11), cmap="RdYlGn",
                     alpha=0.55)
    line = ax.contour(A, F, P, levels=[0.5], colors=[PRIMARY], linewidths=3)
    ax.clabel(line, fmt={0.5: "p = 0.5"}, fontsize=SMALL, inline=True)
    d = X_tr.assign(y=y_tr.values).dropna(subset=["Age"])
    ax.scatter(d.loc[d.y == 0, "Age"], d.loc[d.y == 0, "Fare"], s=26,
               facecolor="none", edgecolor="#16212b", linewidths=1.0,
               label="died", zorder=3)
    ax.scatter(d.loc[d.y == 1, "Age"], d.loc[d.y == 1, "Fare"], s=30,
               color="#16212b", marker="^", label="survived", zorder=3)
    ax.set_xlim(0, 80); ax.set_ylim(0, 300)
    ax.set_xlabel("Age (years)"); ax.set_ylabel("Fare (1912 pounds)")
    ax.set_title("Two features only — the boundary is a straight line, and it "
                 "has to be")
    ax.legend(loc="upper right", framealpha=0.95, frameon=True)
    fig.colorbar(cs, ax=ax, label="estimated P(survived)", pad=0.015)
    fig.tight_layout()
    return save(fig, "l05-boundary", raster=True)


def threshold_sweep(p, y) -> dict:
    """Sweep the cut-off on OUT-OF-FOLD probabilities.

    Choosing a cut-off on predictions the model has already seen is the same
    mistake as scoring a tree on its training rows, one level up. `p` here is
    `cross_val_predict`, so every probability was produced by a fit that had
    never seen that passenger.
    """
    ts = np.linspace(0.02, 0.98, 97)
    rows = []
    for t in ts:
        # the operational rule: escort everyone whose survival probability is
        # BELOW the cut-off
        flag = p < t
        died = (y.values == 0)
        fn = int((died & ~flag).sum())       # not flagged, did not survive
        fp = int((~died & flag).sum())       # flagged, survived anyway
        tp = int((died & flag).sum())
        cost = COST_FN * fn + COST_FP * fp
        rows.append({"t": float(t), "flagged": int(flag.sum()), "fn": fn,
                     "fp": fp, "tp": tp, "cost": float(cost),
                     "recall": tp / max(died.sum(), 1),
                     "precision": tp / max(flag.sum(), 1)})
    best = min(rows, key=lambda r: r["cost"])
    half = min(rows, key=lambda r: abs(r["t"] - 0.5))
    return {"rows": rows, "best": best, "at_half": half,
            "cost_saved": half["cost"] - best["cost"],
            "cost_ratio": COST_FN / COST_FP}


def fig_threshold(sw):
    ts = [r["t"] for r in sw["rows"]]
    cost = [r["cost"] for r in sw["rows"]]
    rec = [r["recall"] for r in sw["rows"]]
    prec = [r["precision"] for r in sw["rows"]]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11.4, 4.2))

    ax.plot(ts, cost, color=PRIMARY, lw=3)
    b, h = sw["best"], sw["at_half"]
    ax.plot([b["t"]], [b["cost"]], "o", color=SUCCESS, ms=11, zorder=5)
    ax.plot([h["t"]], [h["cost"]], "o", color=ACCENT, ms=11, zorder=5)
    ax.annotate(f"cut-off {h['t']:.2f}\ncost {h['cost']:.0f}",
                xy=(h["t"], h["cost"]), xytext=(0.10, max(cost) * 0.86),
                fontsize=SMALL, color=ACCENT, fontweight="bold",
                bbox=dict(fc="white", ec=ACCENT, lw=1.2,
                          boxstyle="round,pad=0.35"),
                arrowprops=dict(arrowstyle="->", color=ACCENT, lw=1.8))
    ax.annotate(f"cut-off {b['t']:.2f}\ncost {b['cost']:.0f}",
                xy=(b["t"], b["cost"]), xytext=(0.44, max(cost) * 0.28),
                fontsize=SMALL, color=SUCCESS, fontweight="bold",
                bbox=dict(fc="white", ec=SUCCESS, lw=1.2,
                          boxstyle="round,pad=0.35"),
                arrowprops=dict(arrowstyle="->", color=SUCCESS, lw=1.8))
    ax.set_xlabel("cut-off on P(survived)")
    ax.set_ylabel(f"expected cost   ({COST_FN:.0f} : {COST_FP:.0f})")
    ax.set_title("The cut-off is a decision, not a default")

    ax2.plot(ts, rec, color=SUCCESS, lw=3, label="recall on the at-risk class")
    ax2.plot(ts, prec, color=ACCENT, lw=3, label="precision")
    ax2.axvline(b["t"], color=PRIMARY, lw=2, ls="--")
    ax2.text(b["t"] + 0.02, 0.06, f"chosen {b['t']:.2f}", fontsize=SMALL,
             color=PRIMARY, fontweight="bold")
    ax2.set_xlabel("cut-off on P(survived)")
    ax2.set_ylim(0, 1.05)
    ax2.set_title("What the choice buys and what it costs")
    ax2.legend(loc="lower right")
    fig.tight_layout()
    return save(fig, "l05-threshold")


def calibration(m, X, y, bins=10) -> dict:
    p = m.predict_proba(X)[:, 1]
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    rows = []
    for b in range(bins):
        sel = idx == b
        if sel.sum() == 0:
            continue
        rows.append({"lo": float(edges[b]), "hi": float(edges[b + 1]),
                     "mean_p": float(p[sel].mean()),
                     "observed": float(y.values[sel].mean()),
                     "n": int(sel.sum())})
    ece = sum(r["n"] * abs(r["mean_p"] - r["observed"]) for r in rows) / len(p)
    return {"bins": rows, "ece": float(ece), "n": int(len(p))}


def fig_calibration(cal_in, cal_out):
    fig, ax = plt.subplots(figsize=(7.6, 5.4))
    ax.plot([0, 1], [0, 1], color=RULE, lw=2, ls="--")
    ax.text(0.62, 0.55, "perfect calibration", rotation=34, fontsize=SMALL,
            color=MUTED, rotation_mode="anchor")
    for cal, colour, label, marker in ((cal_in, PRIMARY, "training folds", "o"),
                                       (cal_out, ACCENT, "held-out folds", "s")):
        xs = [r["mean_p"] for r in cal["bins"]]
        ys = [r["observed"] for r in cal["bins"]]
        ns = [r["n"] for r in cal["bins"]]
        ax.plot(xs, ys, color=colour, lw=2.4, marker=marker,
                ms=0, zorder=3, label=f"{label} (ECE {cal['ece']:.3f})")
        ax.scatter(xs, ys, s=[18 + 2.4 * n for n in ns], color=colour,
                   zorder=4, alpha=0.85, linewidths=0)
    ax.set_xlabel("predicted probability of survival")
    ax.set_ylabel("observed survival rate in the bin")
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    ax.set_title("Marker area is the number of passengers in the bin")
    ax.legend(loc="upper left")
    fig.tight_layout()
    return save(fig, "l05-calibration")


def degree_sweep(X_tr, y_tr) -> dict:
    from sklearn.model_selection import cross_validate
    out = {}
    for deg in DEGREES:
        m = model(degree=deg)
        r = quiet(cross_validate)(
            m, X_tr, y_tr, cv=cv_splitter(), return_train_score=True, n_jobs=-1,
            scoring=["neg_log_loss", "accuracy", "neg_brier_score"])
        fitted = quiet(lambda: model(degree=deg).fit(X_tr, y_tr))()
        n_feat = int(fitted[:-1].transform(X_tr).shape[1])
        # Lecture 5 shows an "iterations used" row to make the convergence
        # warning concrete. It was hand-typed and one of its six numbers was
        # not a measurement of anything; lbfgs reports the real count.
        n_iter = fitted[-1].n_iter_
        out[deg] = {
            "n_features": n_feat,
            "n_iter": int(n_iter[0] if hasattr(n_iter, "__len__") else n_iter),
            "converged": bool((n_iter[0] if hasattr(n_iter, "__len__")
                               else n_iter) < 4000),
            "train_log_loss": float(-r["train_neg_log_loss"].mean()),
            "cv_log_loss": float(-r["test_neg_log_loss"].mean()),
            "cv_log_loss_std": float(r["test_neg_log_loss"].std()),
            "train_accuracy": float(r["train_accuracy"].mean()),
            "cv_accuracy": float(r["test_accuracy"].mean()),
            "cv_accuracy_std": float(r["test_accuracy"].std()),
            "train_brier": float(-r["train_neg_brier_score"].mean()),
            "cv_brier": float(-r["test_neg_brier_score"].mean()),
            "cv_log_loss_folds": [float(-v) for v in r["test_neg_log_loss"]],
        }
    best = min(DEGREES, key=lambda d: out[d]["cv_log_loss"])
    return {"by_degree": out, "best_degree": best,
            "best_cv_log_loss": out[best]["cv_log_loss"],
            "best_cv_accuracy": out[best]["cv_accuracy"],
            "worst_degree": max(DEGREES, key=lambda d: out[d]["cv_log_loss"])}


def fig_degree_curves(ds):
    d = ds["by_degree"]
    xs = DEGREES
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11.6, 4.3))

    tr = [d[k]["train_log_loss"] for k in xs]
    cv = [d[k]["cv_log_loss"] for k in xs]
    ax.plot(xs, tr, color=PRIMARY, lw=3, marker="o", ms=7,
            label="training folds")
    ax.plot(xs, cv, color=ACCENT, lw=3, marker="s", ms=7,
            label="held-out folds")
    b = ds["best_degree"]
    ax.plot([b], [d[b]["cv_log_loss"]], "o", color=SUCCESS, ms=14,
            markerfacecolor="none", markeredgewidth=3, zorder=5)
    ax.annotate(f"best held-out\ndegree {b}: {d[b]['cv_log_loss']:.3f}",
                xy=(b, d[b]["cv_log_loss"]), xytext=(2.5, 1.55),
                fontsize=SMALL, color=SUCCESS, fontweight="bold",
                bbox=dict(fc="white", ec=SUCCESS, lw=1.2,
                          boxstyle="round,pad=0.35"),
                arrowprops=dict(arrowstyle="->", color=SUCCESS, lw=1.8))
    ax.set_xlabel("polynomial degree on the five numeric columns")
    ax.set_ylabel("log loss (lower is better)")
    ax.set_title("Log loss")
    ax.legend(loc="upper left")

    tra = [100 * d[k]["train_accuracy"] for k in xs]
    cva = [100 * d[k]["cv_accuracy"] for k in xs]
    ax2.plot(xs, tra, color=PRIMARY, lw=3, marker="o", ms=7,
             label="training folds")
    ax2.plot(xs, cva, color=ACCENT, lw=3, marker="s", ms=7,
             label="held-out folds")
    ax2.set_xlabel("polynomial degree on the five numeric columns")
    ax2.set_ylabel("accuracy (%)")
    ax2.set_ylim(70, 92)
    ax2.set_title("Accuracy — the same story, much quieter")
    ax2.legend(loc="lower left")
    fig.tight_layout()
    return save(fig, "l05-degree-curves")


def fig_degree_features(ds):
    d = ds["by_degree"]
    xs = DEGREES
    nf = [d[k]["n_features"] for k in xs]
    fig, ax = plt.subplots(figsize=(10.6, 3.5))
    bars = ax.bar([str(x) for x in xs], nf, color=PRIMARY, width=0.6)
    for rect, v in zip(bars, nf):
        ax.text(rect.get_x() + rect.get_width() / 2, v + max(nf) * 0.03,
                f"{v}", ha="center", fontsize=SMALL, fontweight="bold",
                color=PRIMARY)
    ax.axhline(712, color=ACCENT, lw=2, ls="--")
    ax.text(-0.42, 712 + max(nf) * 0.04, "712 training rows", fontsize=SMALL,
            color=ACCENT, fontweight="bold")
    ax.set_xlabel("polynomial degree")
    ax.set_ylabel("columns entering the model")
    # the 712-row reference line has to be INSIDE the limits, or it is invisible
    # and its label floats above the axes — which bbox_inches="tight" then
    # expands the figure to include, wrecking the aspect ratio and shrinking
    # every label to 10.6px on the slide
    ax.set_ylim(0, max(max(nf), 712) * 1.18)
    ax.set_title("Degree 6 asks 712 rows to determine 484 weights")
    ax.grid(axis="x", alpha=0)
    fig.tight_layout()
    return save(fig, "l05-degree-features")


# --- the worked assistant failure, Lecture 5 --------------------------------

def accuracy_trap(X_tr, y_tr, X_te, y_te, n_seeds=20) -> dict:
    """The prompt asked for accuracy. Accuracy picks the cut-off at 0.5.

    Measured over 20 splits, because one subtraction of two noisy numbers is
    not a measurement.
    """
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import cross_val_predict, train_test_split
    X = pd.concat([X_tr, X_te]); y = pd.concat([y_tr, y_te])
    grid = np.linspace(0.02, 0.98, 97)

    def cost(p, yv, t):
        flag = p < t
        died = (yv == 0)
        return float(COST_FN * (died & ~flag).sum()
                     + COST_FP * (~died & flag).sum())

    acc_half, acc_best, cost_half, cost_best, chosen = [], [], [], [], []
    for s in range(n_seeds):
        A, B, ya, yb = train_test_split(X, y, test_size=0.2, random_state=s,
                                        stratify=y)
        # the cut-off is chosen on out-of-fold training predictions, never on B
        p_oof = quiet(cross_val_predict)(model(degree=1), A, ya, cv=5,
                                         method="predict_proba", n_jobs=-1)[:, 1]
        chosen_t = float(min(grid, key=lambda t: cost(p_oof, ya.values, t)))
        m = quiet(lambda: model(degree=1).fit(A, ya))()
        p = m.predict_proba(B)[:, 1]
        acc_half.append(accuracy_score(yb, (p >= 0.5).astype(int)))
        acc_best.append(accuracy_score(yb, (p >= chosen_t).astype(int)))
        cost_half.append(cost(p, yb.values, 0.5))
        cost_best.append(cost(p, yb.values, chosen_t))
        chosen.append(chosen_t)
    ch, cb = np.array(cost_half), np.array(cost_best)
    return {
        "n_seeds": n_seeds,
        "accuracy_at_half": float(np.mean(acc_half)),
        "accuracy_at_half_sd": float(np.std(acc_half)),
        "accuracy_at_best_cost": float(np.mean(acc_best)),
        "accuracy_at_best_cost_sd": float(np.std(acc_best)),
        "cost_at_half": float(ch.mean()), "cost_at_half_sd": float(ch.std()),
        "cost_at_best": float(cb.mean()), "cost_at_best_sd": float(cb.std()),
        "cost_penalty": float((ch - cb).mean()),
        "cost_penalty_sd": float((ch - cb).std()),
        "cost_penalty_pct": float(100 * (ch - cb).mean() / cb.mean()),
        "chosen_threshold": float(np.mean(chosen)),
        "chosen_threshold_sd": float(np.std(chosen)),
        "seeds_where_half_is_worse": int((ch > cb).sum()),
        "accuracy_drop": float(np.mean(acc_half) - np.mean(acc_best)),
    }


def fig_cost_vs_accuracy(trap, sw):
    fig, ax = plt.subplots(figsize=(10.2, 4.0))
    labels = ["cut-off 0.50\n(what accuracy chooses)",
              f"cut-off {sw['best']['t']:.2f}\n(what the brief chooses)"]
    costs = [trap["cost_at_half"], trap["cost_at_best"]]
    accs = [100 * trap["accuracy_at_half"], 100 * trap["accuracy_at_best_cost"]]
    x = np.arange(2)
    b1 = ax.bar(x - 0.19, costs, 0.36, color=ACCENT, label="expected cost")
    ax.set_ylabel("expected cost (escort-units)")
    ax.set_xticks(x, labels)
    ax.set_ylim(0, max(costs) * 1.32)
    for rect, v in zip(b1, costs):
        ax.text(rect.get_x() + rect.get_width() / 2, v + max(costs) * 0.03,
                f"{v:.0f}", ha="center", fontsize=SMALL, fontweight="bold",
                color=ACCENT)
    ax2 = ax.twinx()
    b2 = ax2.bar(x + 0.19, accs, 0.36, color=PRIMARY, label="accuracy")
    ax2.set_ylabel("accuracy (%)")
    ax2.set_ylim(0, 100)
    ax2.grid(False)
    for rect, v in zip(b2, accs):
        ax2.text(rect.get_x() + rect.get_width() / 2, v + 2.4, f"{v:.1f}%",
                 ha="center", fontsize=SMALL, fontweight="bold", color=PRIMARY)
    ax.set_title("Accuracy prefers the left pair. The stakeholder pays for the "
                 "right one.")
    ax.grid(axis="x", alpha=0)
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper center", ncols=2)
    fig.tight_layout()
    return save(fig, "l05-cost-vs-accuracy")


# ============================================================= LECTURE 6 · FIX

def noise_cells(full) -> dict:
    """Irreducible error, measured: passengers whose recorded inputs match.

    Within a cell every model of these features must predict the same number,
    so whatever variation in outcome remains inside the cell is a floor no
    model of these features can go below. k(m-k)/(m(m-1)) is the unbiased
    estimator of p(1-p) from m Bernoulli draws.
    """
    d = full.copy()
    d["AgeBand"] = pd.cut(d["Age"], [0, 12, 25, 40, 60, 100],
                          labels=["0-12", "13-25", "26-40", "41-60", "60+"])
    d["AgeBand"] = d["AgeBand"].cat.add_categories(["missing"]).fillna("missing")
    d["FamBand"] = pd.cut(d["FamilySize"], [0, 1, 4, 20],
                          labels=["alone", "2-4", "5+"])
    keys = ["Sex", "Pclass", "AgeBand", "FamBand", "Embarked"]
    g = d.groupby(keys, observed=True)["Survived"].agg(["sum", "count"])
    g = g.rename(columns={"sum": "k", "count": "m"})
    multi = g[g["m"] >= 2]
    unbiased = multi["k"] * (multi["m"] - multi["k"]) / (
        multi["m"] * (multi["m"] - 1))
    # weight by passengers, since the expectation is over the population
    noise = float((unbiased * multi["m"]).sum() / multi["m"].sum())
    mixed = multi[(multi["k"] > 0) & (multi["k"] < multi["m"])]
    biggest = mixed.assign(mix=lambda t: t["k"] * (t["m"] - t["k"])).nlargest(
        4, "mix")
    return {
        "n_cells": int(len(g)),
        "n_cells_multi": int(len(multi)),
        "n_passengers_multi": int(multi["m"].sum()),
        "n_cells_mixed": int(len(mixed)),
        "n_passengers_mixed": int(mixed["m"].sum()),
        "noise_brier": noise,
        "examples": [{"key": " · ".join(str(v) for v in k), "k": int(r["k"]),
                      "m": int(r["m"])}
                     for k, r in biggest.iterrows()],
        "keys": keys,
    }


def _boot_fit(deg, X, y, sel, X_te):
    m = quiet(lambda: model(degree=deg).fit(X.iloc[sel], y.iloc[sel]))()
    return m.predict_proba(X_te)[:, 1]


def bias_variance(X_tr, y_tr, X_te, y_te) -> dict:
    """The decomposition, measured on the students' own pipeline.

    For each degree, draw N_BOOT training sets of BOOT_N rows, fit, and predict
    on the held-out set. For a fixed test point,

        E_D[(y - p)^2]  =  (y - pbar)^2  +  E_D[(p - pbar)^2],   pbar = E_D[p]

    exactly. The first term averages, over the label draw, to bias^2 + noise;
    the second is variance. Both are measured directly here. Splitting the
    first needs p*(x), which is never observed - `noise_cells` bounds it.
    """
    from joblib import Parallel, delayed
    rng = np.random.default_rng(SEED)
    idx = np.arange(len(X_tr))
    draws = [rng.choice(idx, size=BOOT_N, replace=False) for _ in range(N_BOOT)]
    yv = y_te.values.astype(float)
    out = {}
    for deg in DEGREES:
        rows = Parallel(n_jobs=-1)(
            delayed(_boot_fit)(deg, X_tr, y_tr, sel, X_te) for sel in draws)
        P = np.array(rows)
        pbar = P.mean(axis=0)
        var = ((P - pbar) ** 2).mean(axis=0)
        bias_plus_noise = (yv - pbar) ** 2
        total = ((yv - P) ** 2).mean(axis=0)
        out[deg] = {
            "total": float(total.mean()),
            "variance": float(var.mean()),
            "bias2_plus_noise": float(bias_plus_noise.mean()),
            "identity_residual": float(
                abs(total.mean() - var.mean() - bias_plus_noise.mean())),
        }
        print(f"    degree {deg}: total {out[deg]['total']:.4f} "
              f"= bias2+noise {out[deg]['bias2_plus_noise']:.4f} "
              f"+ var {out[deg]['variance']:.4f}", flush=True)
    best = min(DEGREES, key=lambda d: out[d]["total"])
    return {"by_degree": out, "n_boot": N_BOOT, "boot_n": BOOT_N,
            "best_degree": best,
            "variance_growth": out[max(DEGREES)]["variance"] / out[1]["variance"]}


def fig_bias_variance(bv, noise):
    d = bv["by_degree"]
    xs = DEGREES
    bpn = [d[k]["bias2_plus_noise"] for k in xs]
    var = [d[k]["variance"] for k in xs]
    tot = [d[k]["total"] for k in xs]
    fig, ax = plt.subplots(figsize=(10.9, 4.6))
    ax.stackplot(xs, bpn, var, colors=["#9fb8ca", "#e6b0a8"],
                 labels=["bias² + noise", "variance"], edgecolor="white",
                 linewidth=1.4)
    ax.plot(xs, tot, color="#16212b", lw=3, marker="o", ms=7,
            label="total (expected Brier score)")
    nf = noise["noise_brier"]
    ax.axhline(nf, color=MATH, lw=2.4, ls="--")
    ax.text(6.0, nf - 0.012, f"measured noise floor  {nf:.3f}", ha="right",
            va="top", fontsize=SMALL, color=MATH, fontweight="bold",
            bbox=dict(fc="white", ec="none", pad=1.6))
    b = bv["best_degree"]
    ax.axvline(b, color=SUCCESS, lw=2, ls=":")
    ax.annotate(f"degree {b}", xy=(b, tot[b - 1]), xytext=(b + 0.5, 0.30),
                fontsize=SMALL, color=SUCCESS, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=SUCCESS, lw=1.8))
    ax.set_xlabel("polynomial degree")
    ax.set_ylabel("expected squared error on a held-out passenger")
    ax.set_xlim(1, 6)
    ax.set_ylim(0, max(tot) * 1.12)
    ax.set_title(f"{bv['n_boot']} training sets of {bv['boot_n']} rows each, "
                 f"measured — not asserted")
    ax.legend(loc="upper left")
    fig.tight_layout()
    return save(fig, "l06-bias-variance")


def fig_noise_cells(noise):
    ex = noise["examples"]
    fig, ax = plt.subplots(figsize=(10.8, 3.6))
    ys = np.arange(len(ex))[::-1]
    for y_, e in zip(ys, ex):
        k, m = e["k"], e["m"]
        ax.barh(y_, m - k, color=ACCENT, height=0.55)
        ax.barh(y_, k, left=m - k, color=SUCCESS, height=0.55)
        ax.text(m + 0.35, y_, f"{m - k} died, {k} survived", va="center",
                fontsize=SMALL, fontweight="bold", color="#16212b")
    ax.set_yticks(ys, [e["key"] for e in ex], fontsize=SMALL)
    ax.set_xlim(0, max(e["m"] for e in ex) * 1.75)
    ax.set_xlabel("passengers sharing exactly these recorded values")
    ax.set_title("Identical inputs, different outcomes — the floor, drawn")
    ax.grid(axis="y", alpha=0)
    fig.tight_layout()
    return save(fig, "l06-noise-cells")


def learning_curves(X_tr, y_tr) -> dict:
    from sklearn.model_selection import learning_curve
    sizes = np.linspace(0.12, 1.0, 8)
    out = {}
    for deg in (1, 5):
        n, tr, te = quiet(learning_curve)(
            model(degree=deg), X_tr, y_tr, cv=cv_splitter(), train_sizes=sizes,
            scoring="neg_log_loss", n_jobs=-1, shuffle=True, random_state=SEED)
        out[deg] = {"sizes": [int(v) for v in n],
                    "train": [float(-v) for v in tr.mean(axis=1)],
                    "valid": [float(-v) for v in te.mean(axis=1)]}
        out[deg]["final_gap"] = out[deg]["valid"][-1] - out[deg]["train"][-1]
    return out


def fig_learning_curve(lc):
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.3), sharey=True)
    top = max(max(lc[d]["valid"]) for d in (1, 5))
    titles = {1: "Degree 1 — high bias", 5: "Degree 5 — high variance"}
    for ax, deg in zip(axes, (1, 5)):
        c = lc[deg]
        ax.plot(c["sizes"], c["train"], color=PRIMARY, lw=3, marker="o", ms=6,
                label="training folds")
        ax.plot(c["sizes"], c["valid"], color=ACCENT, lw=3, marker="s", ms=6,
                label="held-out folds")
        ax.fill_between(c["sizes"], c["train"], c["valid"], color=ACCENT,
                        alpha=0.12)
        ax.set_xlabel("training passengers")
        ax.set_title(titles[deg])
        ax.set_ylim(0, top * 1.08)
        ax.annotate(f"final gap {c['final_gap']:.3f}",
                    xy=(c["sizes"][-1], (c["train"][-1] + c["valid"][-1]) / 2),
                    xytext=(c["sizes"][0] + 30, top * 0.86), fontsize=SMALL,
                    color=ACCENT, fontweight="bold",
                    bbox=dict(fc="white", ec=ACCENT, lw=1.2,
                              boxstyle="round,pad=0.32"),
                    arrowprops=dict(arrowstyle="->", color=ACCENT, lw=1.8))
    axes[0].set_ylabel("log loss")
    axes[0].legend(loc="lower right")
    fig.tight_layout()
    return save(fig, "l06-learning-curve")


def regularisation_sweep(X_tr, y_tr) -> dict:
    from sklearn.model_selection import cross_val_score
    Cs = np.logspace(-4, 4, 17)
    deg = 5          # the overfitting one, so the repair has something to do
    out = {}
    for name, kw in (("ridge", dict(penalty="l2", solver="lbfgs")),
                     ("lasso", dict(penalty="l1", solver="liblinear")),
                     ("elastic", dict(penalty="elasticnet", solver="saga",
                                      l1_ratio=0.5, max_iter=3000))):
        scores, nz = [], []
        for C in Cs:
            m = model(degree=deg, C=float(C), **kw)
            s = quiet(cross_val_score)(m, X_tr, y_tr, cv=cv_splitter(),
                                       scoring="neg_log_loss", n_jobs=-1)
            scores.append(float(-s.mean()))
            fitted = quiet(lambda: model(degree=deg, C=float(C), **kw)
                           .fit(X_tr, y_tr))()
            nz.append(int((np.abs(fitted[-1].coef_[0]) > 1e-8).sum()))
        j = int(np.argmin(scores))
        out[name] = {"C": [float(c) for c in Cs], "log_loss": scores,
                     "n_nonzero": nz, "best_C": float(Cs[j]),
                     "best_log_loss": scores[j], "best_n_nonzero": nz[j]}
    out["degree"] = deg
    out["n_features"] = int(
        quiet(lambda: model(degree=deg).fit(X_tr, y_tr))()[:-1]
        .transform(X_tr).shape[1])
    return out


def fig_alpha_curve(rs, ds):
    fig, ax = plt.subplots(figsize=(10.9, 4.4))
    styles = {"ridge": (PRIMARY, "ridge  (ℓ₂)"),
              "lasso": (ACCENT, "lasso  (ℓ₁)"),
              "elastic": (MATH, "elastic net  (ℓ₁ ratio 0.5)")}
    for name, (colour, label) in styles.items():
        r = rs[name]
        ax.semilogx(r["C"], r["log_loss"], color=colour, lw=3, label=label)
        plain_log(ax, "x", fmt="{:g}")
        ax.plot([r["best_C"]], [r["best_log_loss"]], "o", color=colour, ms=10,
                zorder=5)
    unreg = ds["by_degree"][rs["degree"]]["cv_log_loss"]
    ax.axhline(unreg, color="#16212b", lw=2, ls="--")
    ax.text(1e-4, unreg - 0.06, f"what you built: {unreg:.2f}", fontsize=SMALL,
            color="#16212b", fontweight="bold", va="top")
    base = ds["by_degree"][2]["cv_log_loss"]
    ax.axhline(base, color=SUCCESS, lw=2, ls=":")
    ax.text(1e-4, base + 0.05, f"best unregularised degree: {base:.3f}",
            fontsize=SMALL, color=SUCCESS, fontweight="bold")
    ax.set_xlabel("C  =  1/α   (larger C means weaker regularisation)")
    ax.set_ylabel("cross-validated log loss")
    ax.set_ylim(0.38, min(2.6, max(rs["ridge"]["log_loss"]) * 1.05))
    ax.set_title(f"Degree {rs['degree']}, {rs['n_features']} columns, "
                 f"712 rows — with the penalty turned on")
    ax.legend(loc="upper left")
    fig.tight_layout()
    return save(fig, "l06-alpha-curve")


def coef_paths(X_tr, y_tr) -> dict:
    Cs = np.logspace(-3, 3, 25)
    deg = 5
    out = {}
    for name, kw in (("ridge", dict(penalty="l2", solver="lbfgs")),
                     ("lasso", dict(penalty="l1", solver="liblinear"))):
        paths = []
        for C in Cs:
            m = quiet(lambda: model(degree=deg, C=float(C), **kw)
                      .fit(X_tr, y_tr))()
            paths.append(m[-1].coef_[0].copy())
        A = np.array(paths)
        out[name] = {"C": [float(c) for c in Cs], "paths": A.tolist(),
                     "n_nonzero": [int((np.abs(row) > 1e-8).sum()) for row in A],
                     "max_abs": [float(np.abs(row).max()) for row in A]}
    out["degree"] = deg
    return out


def fig_paths(cp):
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.3), sharey=True)
    rng = np.random.default_rng(SEED)
    for ax, name, title in ((axes[0], "ridge", "Ridge — every weight shrinks, "
                                               "none reaches zero"),
                            (axes[1], "lasso", "Lasso — weights arrive at "
                                               "exactly zero and stay")):
        A = np.array(cp[name]["paths"])
        Cs = np.array(cp[name]["C"])
        keep = rng.choice(A.shape[1], size=min(45, A.shape[1]), replace=False)
        for j in keep:
            ax.semilogx(Cs, A[:, j], color=PRIMARY if name == "ridge" else ACCENT,
                        lw=1.3, alpha=0.55)
        # after plotting: semilogx re-applies the log scale, which would reset
        # the formatter and put mathtext back on the axis
        plain_log(ax, "x", fmt="{:g}")
        ax.axhline(0, color=AXIS, lw=1.4)
        ax.set_xlabel("C  =  1/α")
        ax.set_title(title)
        ax.set_ylim(-6, 6)
    axes[0].set_ylabel("coefficient")
    ax2 = axes[1].twinx()
    ax2.semilogx(cp["lasso"]["C"], cp["lasso"]["n_nonzero"], color=SUCCESS,
                 lw=3)
    plain_log(ax2, "x", fmt="{:g}")
    ax2.set_ylabel("non-zero weights", color=SUCCESS)
    ax2.tick_params(axis="y", colors=SUCCESS)
    ax2.grid(False)
    fig.tight_layout()
    return save(fig, "l06-paths")


def condition_numbers(X_tr) -> dict:
    """Thread 1, returning. FamilySize = SibSp + Parch + 1, exactly.

    So the design matrix the students engineered themselves has an exact linear
    dependence in it, and XᵀX is singular. Adding αI moves every eigenvalue up
    by α, so the ridge system is invertible for every α > 0 — and the condition
    number is bounded by (λ_max + α)/α whatever the data do.
    """
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    cols = ["SibSp", "Parch", "FamilySize", "Age", "Fare"]
    Z = make_pipeline(SimpleImputer(strategy="median"),
                      StandardScaler()).fit_transform(X_tr[cols])
    G = Z.T @ Z
    eig = np.linalg.eigvalsh(G)
    alphas = np.logspace(-12, 3, 61)
    conds = [float((eig.max() + a) / (eig.min() + a)) for a in alphas]
    # the exact dependence, verified rather than asserted
    resid = float(np.abs(X_tr["SibSp"] + X_tr["Parch"] + 1
                         - X_tr["FamilySize"]).max())
    return {
        "columns": cols,
        "eigenvalues": [float(v) for v in eig],
        "smallest_eigenvalue": float(eig.min()),
        "largest_eigenvalue": float(eig.max()),
        "cond_alpha_zero": float(np.linalg.cond(G)),
        "alphas": [float(a) for a in alphas],
        "cond": conds,
        "dependence_residual": resid,
        "cond_at_1": float((eig.max() + 1) / (eig.min() + 1)),
    }


def fig_condition(cn):
    fig, ax = plt.subplots(figsize=(10.6, 4.2))
    ax.loglog(cn["alphas"], cn["cond"], color=MATH, lw=3)
    plain_log(ax, "x", fmt="{:g}")
    plain_log(ax, "y", fmt="{:,.0f}")
    ax.axhline(cn["cond_alpha_zero"], color=ACCENT, lw=2, ls="--")
    ax.text(1e-12, cn["cond_alpha_zero"] * 0.35,
            f"α = 0:  condition number {cn['cond_alpha_zero']:.1e}",
            fontsize=SMALL, color=ACCENT, fontweight="bold")
    ax.plot([1.0], [cn["cond_at_1"]], "o", color=SUCCESS, ms=11, zorder=5)
    ax.annotate(f"α = 1:  {cn['cond_at_1']:.0f}", xy=(1.0, cn["cond_at_1"]),
                xytext=(1e-6, 8), fontsize=SMALL, color=SUCCESS,
                fontweight="bold",
                bbox=dict(fc="white", ec=SUCCESS, lw=1.2,
                          boxstyle="round,pad=0.35"),
                arrowprops=dict(arrowstyle="->", color=SUCCESS, lw=1.8))
    ax.set_xlabel("α")
    ax.set_ylabel("condition number of  XᵀX + αI")
    ax.set_title("FamilySize = SibSp + Parch + 1 — the dependence they "
                 "engineered themselves")
    fig.tight_layout()
    return save(fig, "l06-condition")


def early_stopping(X_tr, y_tr) -> dict:
    from sklearn.linear_model import SGDClassifier
    from sklearn.metrics import log_loss
    from sklearn.model_selection import train_test_split
    A, B, ya, yb = train_test_split(X_tr, y_tr, test_size=0.25,
                                    random_state=SEED, stratify=y_tr)
    pre = prep(degree=5)
    Za = quiet(lambda: pre.fit_transform(A))()
    Zb = pre.transform(B)
    clf = SGDClassifier(loss="log_loss", penalty=None, learning_rate="constant",
                        eta0=0.0015, random_state=SEED, warm_start=True,
                        max_iter=1, tol=None)
    tr, va = [], []
    n_epochs = 500
    for _ in range(n_epochs):
        quiet(lambda: clf.fit(Za, ya))()
        tr.append(float(log_loss(ya, clf.predict_proba(Za)[:, 1],
                                 labels=[0, 1])))
        va.append(float(log_loss(yb, clf.predict_proba(Zb)[:, 1],
                                 labels=[0, 1])))
    best = int(np.argmin(va))
    return {"epochs": n_epochs, "train": tr, "valid": va, "best_epoch": best + 1,
            "best_valid": va[best], "final_valid": va[-1],
            "train_at_best": tr[best], "final_train": tr[-1],
            "regret": va[-1] - va[best]}


def fig_early_stopping(es):
    xs = np.arange(1, es["epochs"] + 1)
    fig, ax = plt.subplots(figsize=(10.8, 4.2))
    ax.plot(xs, es["train"], color=PRIMARY, lw=2.6, label="training subset")
    ax.plot(xs, es["valid"], color=ACCENT, lw=2.6, label="validation subset")
    ax.axvline(es["best_epoch"], color=SUCCESS, lw=2.4, ls="--")
    ax.plot([es["best_epoch"]], [es["best_valid"]], "o", color=SUCCESS, ms=11,
            zorder=5)
    ax.annotate(f"stop here — epoch {es['best_epoch']}\n"
                f"validation {es['best_valid']:.3f}",
                xy=(es["best_epoch"], es["best_valid"]),
                xytext=(es["epochs"] * 0.30, max(es["valid"]) * 0.90),
                fontsize=SMALL, color=SUCCESS, fontweight="bold",
                bbox=dict(fc="white", ec=SUCCESS, lw=1.2,
                          boxstyle="round,pad=0.35"),
                arrowprops=dict(arrowstyle="->", color=SUCCESS, lw=1.8))
    ax.annotate("keep going and the training loss\nfalls while the real one rises",
                xy=(es["epochs"] * 0.95, es["final_valid"]),
                xytext=(es["epochs"] * 0.34, max(es["valid"]) * 0.60),
                fontsize=SMALL, color=ACCENT,
                bbox=dict(fc="white", ec=RULE, lw=1.2,
                          boxstyle="round,pad=0.35"),
                arrowprops=dict(arrowstyle="->", color=ACCENT, lw=1.8))
    ax.set_xlabel("epoch")
    ax.set_ylabel("log loss")
    ax.set_title("One model, two curves, and a point where they disagree")
    ax.legend(loc="upper right")
    fig.tight_layout()
    return save(fig, "l06-early-stopping")


# --- the worked assistant failure, Lecture 6 --------------------------------

TRAP_DEG = 3          # enough columns to overfit; cheap enough for 20 seeds
TRAP_CS = np.logspace(-4, 4, 17)


def _one_tuning_seed(X, y, s):
    from sklearn.metrics import log_loss
    from sklearn.model_selection import cross_val_score, train_test_split
    A, B, ya, yb = train_test_split(X, y, test_size=0.2, random_state=s,
                                    stratify=y)
    test_scores, cv_scores = [], []
    for C in TRAP_CS:
        m = quiet(lambda: model(degree=TRAP_DEG, C=float(C)).fit(A, ya))()
        test_scores.append(float(log_loss(yb, m.predict_proba(B)[:, 1],
                                          labels=[0, 1])))
        cv_scores.append(float(-quiet(cross_val_score)(
            model(degree=TRAP_DEG, C=float(C)), A, ya, cv=5,
            scoring="neg_log_loss").mean()))
    j = int(np.argmin(test_scores))          # what the assistant wrote
    k = int(np.argmin(cv_scores))            # what it should have been
    return test_scores[j], test_scores[k], int(j == k)


def tuning_trap(X_tr, y_tr, X_te, y_te, n_seeds=20) -> dict:
    """The prompt said "find the best C and tell me how well it does".

    The plausible code loops over C, scores each on the test set, keeps the
    winner, and reports the winner's test score. Every C got a look at the test
    set; the minimum of 17 noisy numbers is biased downward, and that bias is
    what is measured here.
    """
    from joblib import Parallel, delayed
    X = pd.concat([X_tr, X_te]); y = pd.concat([y_tr, y_te])
    rows = Parallel(n_jobs=-1)(
        delayed(_one_tuning_seed)(X, y, s) for s in range(n_seeds))
    r = np.array([x[0] for x in rows])
    h = np.array([x[1] for x in rows])
    same = int(sum(x[2] for x in rows))
    return {"n_seeds": n_seeds, "n_values_tried": len(TRAP_CS),
            "degree": TRAP_DEG,
            "reported": float(r.mean()), "reported_sd": float(r.std()),
            "honest": float(h.mean()), "honest_sd": float(h.std()),
            "optimism": float((h - r).mean()),
            "optimism_sd": float((h - r).std()),
            "optimism_pct": float(100 * (h - r).mean() / h.mean()),
            "seeds_where_reported_is_better": int((r < h).sum()),
            "same_C_chosen": same}


def final_scores(X_tr, y_tr, X_te, y_te, rs, ds) -> dict:
    """The test set. Once."""
    from sklearn.metrics import (accuracy_score, brier_score_loss, log_loss,
                                 confusion_matrix)
    out = {}
    variants = {
        "committed_degree5_unregularised": model(degree=5),
        "best_unregularised_degree2": model(degree=2),
        "ridge_degree5_tuned": model(degree=5, C=rs["ridge"]["best_C"]),
        "lasso_degree5_tuned": model(degree=5, C=rs["lasso"]["best_C"],
                                     penalty="l1", solver="liblinear"),
        "sklearn_default_degree1": model(degree=1, C=1.0),
    }
    for name, m in variants.items():
        quiet(lambda: m.fit(X_tr, y_tr))()
        p = m.predict_proba(X_te)[:, 1]
        out[name] = {
            "log_loss": float(log_loss(y_te, p, labels=[0, 1])),
            "brier": float(brier_score_loss(y_te, p)),
            "accuracy": float(accuracy_score(y_te, (p >= 0.5).astype(int))),
        }
    winner = min(out, key=lambda k: out[k]["log_loss"])
    out["winner"] = winner
    out["improvement_vs_committed"] = (
        out["committed_degree5_unregularised"]["log_loss"]
        - out[winner]["log_loss"])
    return out


def separation_check(X_tr, y_tr) -> dict:
    """At high degree the unregularised fit does not converge, and should not.

    In a space where the classes are linearly separable the likelihood has no
    interior maximum: pushing ‖θ‖ up always improves it. lbfgs runs out of
    iterations rather than finding an optimum that does not exist.
    """
    out = {}
    for deg in DEGREES:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            m = model(degree=deg).fit(X_tr, y_tr)
        conv = [x for x in w if "converge" in str(x.message).lower()]
        out[deg] = {"converged": len(conv) == 0,
                    "n_iter": int(np.max(m[-1].n_iter_)),
                    "max_abs_coef": float(np.abs(m[-1].coef_).max())}
    first_bad = next((d for d in DEGREES if not out[d]["converged"]), None)
    return {"by_degree": out, "first_nonconverging_degree": first_bad}


# ------------------------------------------------------------------- driver

def main():
    setup()
    load_cache()
    print("Loading Titanic…")
    full = engineer(load_titanic())
    X_tr, X_te, y_tr, y_te = split(full)

    facts: dict = {}
    print("Lecture 5 — the data:")
    facts["l05_data"] = describe(full, X_tr, X_te, y_tr, y_te)
    facts["l05_slices"] = survival_slices(full)
    fig_missing(full)
    fig_survival_rates(full, facts["l05_slices"])
    fig_sigmoid()

    print("Lecture 5 — the model:")
    base = fit_base(X_tr, y_tr)
    facts["l05_coefficients_v1"] = coefficients_v1(base)
    facts["l05_rank"] = rank_check(X_tr, y_tr)
    facts["l05_ambiguity"] = coef_ambiguity(X_tr, y_tr)
    facts["l05_coefficients"] = coefficients_v2(X_tr, y_tr)
    fig_coefficients(facts["l05_coefficients"])
    fig_coef_ambiguity(facts["l05_ambiguity"])
    fig_boundary(X_tr, y_tr)

    print("Lecture 5 — calibration and the cut-off:")
    from sklearn.model_selection import cross_val_predict

    def _cal():
        return quiet(cross_val_predict)(model(degree=1), X_tr, y_tr,
                                        cv=cv_splitter(), method="predict_proba",
                                        n_jobs=-1)[:, 1]
    p_out = cached("l05_cv_proba_v2", _cal)

    facts["l05_threshold"] = threshold_sweep(p_out, y_tr)
    fig_threshold(facts["l05_threshold"])

    fitted_v2 = quiet(lambda: model(degree=1).fit(X_tr, y_tr))()
    cal_in = calibration(fitted_v2, X_tr, y_tr)

    class _Const:
        def __init__(self, p): self.p = p
        def predict_proba(self, X): return np.c_[1 - self.p, self.p]
    cal_out = calibration(_Const(p_out), X_tr, y_tr)
    facts["l05_calibration"] = {"train": cal_in, "heldout": cal_out}
    fig_calibration(cal_in, cal_out)

    print("Lecture 5 — pushing the degree up:")
    facts["l05_degrees"] = cached("l05_degrees_v2",
                                  lambda: degree_sweep(X_tr, y_tr))
    fig_degree_curves(facts["l05_degrees"])
    fig_degree_features(facts["l05_degrees"])
    facts["l05_separation"] = cached("l05_separation_v2",
                                     lambda: separation_check(X_tr, y_tr))

    print("Lecture 5 — the assistant failure:")
    facts["l05_accuracy_trap"] = cached(
        "l05_accuracy_trap_v2", lambda: accuracy_trap(X_tr, y_tr, X_te, y_te))
    fig_cost_vs_accuracy(facts["l05_accuracy_trap"], facts["l05_threshold"])

    print("Lecture 6 — the noise floor:")
    facts["l06_noise"] = noise_cells(full)
    fig_noise_cells(facts["l06_noise"])

    print("Lecture 6 — the decomposition:")
    facts["l06_bias_variance"] = cached(
        "l06_bias_variance_v2", lambda: bias_variance(X_tr, y_tr, X_te, y_te))
    fig_bias_variance(facts["l06_bias_variance"], facts["l06_noise"])

    print("Lecture 6 — learning curves:")
    facts["l06_learning_curves"] = cached(
        "l06_learning_curves_v2", lambda: learning_curves(X_tr, y_tr))
    fig_learning_curve(facts["l06_learning_curves"])

    print("Lecture 6 — regularisation:")
    facts["l06_regularisation"] = cached(
        "l06_regularisation_v2", lambda: regularisation_sweep(X_tr, y_tr))
    fig_alpha_curve(facts["l06_regularisation"], facts["l05_degrees"])
    facts["l06_paths"] = cached("l06_paths_v2", lambda: coef_paths(X_tr, y_tr))
    fig_paths(facts["l06_paths"])

    print("Lecture 6 — Thread 1 returning:")
    facts["l06_condition"] = condition_numbers(X_tr)
    fig_condition(facts["l06_condition"])

    print("Lecture 6 — early stopping:")
    facts["l06_early_stopping"] = cached(
        "l06_early_stopping_v2", lambda: early_stopping(X_tr, y_tr))
    fig_early_stopping(facts["l06_early_stopping"])

    print("Lecture 6 — the assistant failure:")
    facts["l06_tuning_trap"] = cached(
        "l06_tuning_trap_v2", lambda: tuning_trap(X_tr, y_tr, X_te, y_te))

    print("Lecture 6 — the test set, once:")
    facts["l06_final"] = cached(
        "l06_final_v2",
        lambda: final_scores(X_tr, y_tr, X_te, y_te,
                             facts["l06_regularisation"], facts["l05_degrees"]))

    # differences the slides quote in their own right
    d = facts["l05_degrees"]["by_degree"]
    facts["l06_gaps"] = {
        "deg2_to_deg5_cv_log_loss": d[5]["cv_log_loss"] - d[2]["cv_log_loss"],
        "deg5_train_minus_cv": d[5]["cv_log_loss"] - d[5]["train_log_loss"],
        "deg1_train_minus_cv": d[1]["cv_log_loss"] - d[1]["train_log_loss"],
        "ridge_gain_at_deg5": (d[5]["cv_log_loss"]
                               - facts["l06_regularisation"]["ridge"]["best_log_loss"]),
        "accuracy_over_majority": (facts["l05_degrees"]["best_cv_accuracy"]
                                   - facts["l05_data"]["majority_accuracy_train"]),
        "log_loss_over_constant": (facts["l05_data"]["constant_log_loss"]
                                   - facts["l05_degrees"]["best_cv_log_loss"]),
    }

    if (problems := check_text_floor()):
        print("\nfigures whose text lands under the on-slide floor:")
        for p_ in problems:
            print("  " + p_)
        raise SystemExit(1)

    export(**facts)
    print("\nfigures.json updated (merged).")


if __name__ == "__main__":
    main()
