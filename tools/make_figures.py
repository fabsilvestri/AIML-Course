#!/usr/bin/env python3
"""
Generate every data figure used in the slide decks, and every number quoted in
them.

Figures are produced from the real datasets, with the same code paths the
lectures describe, so a number printed on a slide can always be reproduced by
re-running this script. Nothing here is illustrative-only.

    python3 tools/make_figures.py

Output: assets/figures/*.svg (vector, for anything sparse)
        assets/figures/*.png (raster @160dpi, for dense scatter plots where
                              SVG would embed 20k+ path elements)
        assets/figures/figures.json — every number the slides quote

Expensive fits are cached in CACHE/fits-v2.pkl, keyed by name, so adding one
measurement does not re-run the others. Delete the file to refit everything.
"""

from __future__ import annotations

import json
import re
import struct
from functools import reduce
from math import gcd
import pickle
import tarfile
import urllib.request
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.ticker import FuncFormatter

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "assets" / "figures"
FONTS = ROOT / "assets" / "fonts"
CACHE = Path("/private/tmp/claude-501/aiml-data")

# Palette shared with assets/css/custom.css — keep in sync with :root there.
PRIMARY = "#0b3d62"
ACCENT = "#c0392b"
SUCCESS = "#14663a"     # was #1e8449 — 4.35:1 on the fix panel failed WCAG AA
MATH = "#6c3483"
MUTED = "#4b5563"       # was #6b7280 — the weakest readable text in the deck
RULE = "#b0bcc7"        # was #d5dbe1 — 1.40:1 is simply absent on a projector
SOFT = "#f4f7f9"
AXIS = "#7b8794"        # 3.66:1, clears the 3:1 minimum for graphics

# The plot SVGs display at roughly 1:1 on the slide, so every size below is an
# on-slide pixel. See TRICKS §11.6.
BODY = 17
SMALL = 15              # the floor for anything a student has to read
TICK = 15

SEED = 42
N_FOLDS = 10
CAP = 500_000

# The lectures use a shuffled KFold rather than the unshuffled default, so that
# two students whose frames are ordered differently get identical folds.
def cv_splitter():
    from sklearn.model_selection import KFold
    return KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)


def setup():
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)

    # The plot SVGs are authored at 677-927pt and capped at 420-560px on the
    # slide, so they display at roughly 1:1 — these numbers ARE on-slide pixels.
    # At the old font.size=13 a tick label was 13px against 30px slide text.
    #
    # matplotlib cannot read woff2 and a variable ttf resolves to its default
    # instance (ExtraLight for Source Sans 3), hence the three static cuts.
    for _f in ("SourceSans3-Regular.ttf", "SourceSans3-SemiBold.ttf",
               "SourceSans3-Bold.ttf"):
        font_manager.fontManager.addfont(ROOT / "assets" / "fonts" / _f)

    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "font.family": "Source Sans 3",
        "font.size": BODY,
        "axes.titlesize": 19,
        "axes.titleweight": "normal",   # the slide's h2 is the title; this is a
        "axes.titlecolor": MUTED,       # subtitle, and must not compete with it
        "axes.titlelocation": "left",
        "axes.labelsize": BODY,
        "axes.labelcolor": "#16212b",
        "axes.edgecolor": AXIS,         # 3.66:1; was #98a4b0 at 2.54:1
        "axes.linewidth": 1.2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,         # gridlines under the bars, not through
        "grid.color": RULE,             # 1.93:1; was #d5dbe1 at 1.40:1
        "grid.linewidth": 0.9,
        "xtick.color": "#33414d",
        "ytick.color": "#33414d",
        "xtick.labelsize": TICK,
        "ytick.labelsize": TICK,
        "legend.frameon": False,
        "legend.fontsize": TICK,
        "figure.autolayout": False,
        "svg.hashsalt": "aiml-course",
        # TRICKS §9.1, in the other renderer: two dollar signs in one string
        # are mathtext to matplotlib exactly as they are maths to KaTeX, so
        # "$137,500 (101)  $162,500 (91)" renders as an italic equation. No
        # figure in this course uses mathtext; turn the trap off entirely.
        "text.parse_math": False,
    })
    # a missing font is a silent revert to DejaVu, so fail loudly instead
    resolved = font_manager.findfont("Source Sans 3", fallback_to_default=False)
    if "SourceSans3" not in Path(resolved).name:
        raise RuntimeError(f"Source Sans 3 did not resolve: {resolved}")


# ------------------------------------------------------------------- caching

_CACHE_FILE = CACHE / "fits-v2.pkl"
_cache: dict = {}


def load_cache():
    global _cache
    if _CACHE_FILE.is_file():
        _cache = pickle.loads(_CACHE_FILE.read_bytes())
    else:
        _cache = {}


def cached(key, fn):
    """Run fn() once and remember the result across runs."""
    if key in _cache:
        print(f"    [cached] {key}")
        return _cache[key]
    print(f"    [computing] {key}")
    value = fn()
    _cache[key] = value
    _CACHE_FILE.write_bytes(pickle.dumps(_cache))
    return value


def save(fig, name, *, raster=False, dpi=160):
    ext = "png" if raster else "svg"
    path = OUT / f"{name}.{ext}"
    # no creation date in the file, for the same reason as svg.hashsalt
    meta = {"Date": None} if not raster else {}
    fig.savefig(path, format=ext, dpi=dpi, bbox_inches="tight",
                pad_inches=0.15, metadata=meta)
    plt.close(fig)
    kb = path.stat().st_size / 1024
    print(f"  {path.relative_to(ROOT)}  ({kb:.0f} KB)")
    return path


def load_housing() -> pd.DataFrame:
    """Exactly the loader shown on the slides, with a cache outside the repo."""
    tarball = CACHE / "housing.tgz"
    if not tarball.is_file():
        url = "https://github.com/ageron/data/raw/main/housing.tgz"
        print(f"  downloading {url}")
        urllib.request.urlretrieve(url, tarball)
        with tarfile.open(tarball) as t:
            t.extractall(path=CACHE, filter="data")
    return pd.read_csv(CACHE / "housing" / "housing.csv")


usd = FuncFormatter(lambda v, _: f"${v/1000:,.0f}k" if v else "$0")


def income_cats(h):
    return pd.cut(h["median_income"], bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                  labels=[1, 2, 3, 4, 5])


def split(h):
    """The one stratified split the whole course uses. Named as on the slides."""
    from sklearn.model_selection import train_test_split
    cats = income_cats(h)
    strat_train_set, strat_test_set = train_test_split(
        h, test_size=0.2, random_state=SEED, stratify=cats)
    return {
        "train": strat_train_set,
        "test": strat_test_set,
        "X_train": strat_train_set.drop("median_house_value", axis=1),
        "y_train": strat_train_set["median_house_value"].copy(),
        "X_test": strat_test_set.drop("median_house_value", axis=1),
        "y_test": strat_test_set["median_house_value"].copy(),
    }


def preprocessing(X_train):
    """The ColumnTransformer the lectures build, and nothing else."""
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    num = X_train.select_dtypes(include=[np.number]).columns.tolist()
    cat = ["ocean_proximity"]
    prep = ColumnTransformer([
        ("num", make_pipeline(SimpleImputer(strategy="median"),
                              StandardScaler()), num),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
    ])
    return prep, num, cat


# ---------------------------------------------------------------- lecture 1

def fig_histograms(h):
    cols = ["longitude", "latitude", "housing_median_age", "total_rooms",
            "total_bedrooms", "population", "households", "median_income",
            "median_house_value"]
    fig, axes = plt.subplots(3, 3, figsize=(10.8, 6.6))
    for ax, c in zip(axes.ravel(), cols):
        ax.hist(h[c].dropna(), bins=50, color=PRIMARY, edgecolor="white",
                linewidth=0.25)
        ax.set_title(c, fontsize=SMALL)
        ax.tick_params(labelsize=9)
        ax.grid(alpha=0.55)
    # the two features the slides call out
    axes.ravel()[7].set_title("median_income", color=ACCENT, fontsize=SMALL)
    axes.ravel()[8].set_title("median_house_value", color=ACCENT, fontsize=SMALL)
    fig.suptitle("housing_full.hist(bins=50)", fontsize=BODY, color=MUTED,
                 y=1.005)
    fig.tight_layout()
    return save(fig, "l1-histograms")


def fig_hist_annotated(h):
    """The same two histograms, enlarged, with the cap and the scaling marked."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.8, 3.8))

    a1.hist(h["median_income"], bins=50, color=PRIMARY, edgecolor="white",
            linewidth=0.3)
    a1.set_title("median_income — not dollars, and capped")
    a1.set_xlabel("median_income")
    a1.axvline(15.0001, color=ACCENT, lw=2)
    a1.annotate("capped at 15.0001", xy=(15.0001, a1.get_ylim()[1] * 0.55),
                xytext=(10.2, a1.get_ylim()[1] * 0.78), color=ACCENT,
                fontsize=SMALL, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=ACCENT, lw=1.6))
    a1.text(0.97, 0.44, "≈ tens of thousands of dollars:\n3 means about $30,000",
            transform=a1.transAxes, ha="right", fontsize=SMALL, color="#33414d",
            bbox=dict(boxstyle="round,pad=0.45", facecolor="white",
                      edgecolor=RULE))

    a2.hist(h["median_house_value"], bins=50, color=PRIMARY, edgecolor="white",
            linewidth=0.3)
    a2.set_title("median_house_value — the label is capped")
    a2.set_xlabel("median_house_value")
    a2.xaxis.set_major_formatter(usd)
    n_cap = int((h["median_house_value"] >= CAP).sum())
    a2.axvline(500_001, color=ACCENT, lw=2)
    a2.annotate(f"{n_cap:,} districts pile up\nat the $500,001 cap",
                xy=(500_001, n_cap * 0.9),
                xytext=(250_000, a2.get_ylim()[1] * 0.72), color=ACCENT,
                fontsize=SMALL, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=ACCENT, lw=1.6))
    fig.tight_layout()
    return save(fig, "l1-hist-annotated")


def fig_geo(train):
    """Exploration plots — on the training set, after the split."""
    # plain
    fig, ax = plt.subplots(figsize=(6.4, 6.7))
    ax.scatter(train["longitude"], train["latitude"], s=6, color=PRIMARY)
    ax.set_xlabel("longitude"); ax.set_ylabel("latitude")
    ax.set_title("alpha = 1.0  —  it looks like California, and that is all")
    fig.tight_layout()
    save(fig, "l1-geo-plain", raster=True)

    # alpha, annotated
    fig, ax = plt.subplots(figsize=(6.6, 6.8))
    ax.scatter(train["longitude"], train["latitude"], s=8, color=PRIMARY,
               alpha=0.2, linewidths=0)
    ax.set_xlabel("longitude"); ax.set_ylabel("latitude")
    ax.set_title("alpha = 0.2  —  the density structure appears")
    for name, (lon, lat), off in [
        ("Bay Area", (-122.3, 37.8), (-124.6, 39.3)),
        ("Central Valley", (-120.6, 36.9), (-119.2, 39.0)),
        ("Los Angeles", (-118.3, 34.05), (-120.9, 33.3)),
        ("San Diego", (-117.15, 32.75), (-119.4, 32.2)),
    ]:
        ax.annotate(name, xy=(lon, lat), xytext=off, fontsize=SMALL,
                    color=ACCENT, fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color=ACCENT, lw=1.4))
    fig.tight_layout()
    save(fig, "l1-geo-alpha", raster=True)

    # price + population
    fig, ax = plt.subplots(figsize=(7.4, 6.4))
    sc = ax.scatter(train["longitude"], train["latitude"],
                    s=train["population"] / 100,
                    c=train["median_house_value"],
                    cmap="jet", alpha=0.45, linewidths=0)
    ax.set_xlabel("longitude"); ax.set_ylabel("latitude")
    ax.set_title("radius = population,  colour = median house value")
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("median_house_value")
    cb.ax.yaxis.set_major_formatter(usd)
    fig.tight_layout()
    return save(fig, "l1-geo-price", raster=True)


def fig_corr(train):
    """The correlation column — computed on the training set, as the slide says."""
    corr = train.corr(numeric_only=True)["median_house_value"].drop(
        "median_house_value").sort_values()
    fig, ax = plt.subplots(figsize=(10.8, 4.2))
    colors = [ACCENT if v < 0 else PRIMARY for v in corr]
    ax.barh(corr.index, corr.values, color=colors, height=0.62)
    ax.axvline(0, color="#98a4b0", lw=1)
    ax.set_xlabel("Pearson correlation with median_house_value")
    ax.set_title("One predictor dominates  (training set)")
    ax.grid(axis="y", alpha=0)
    for i, v in enumerate(corr.values):
        ax.text(v + (0.012 if v >= 0 else -0.012), i, f"{v:+.3f}",
                va="center", ha="left" if v >= 0 else "right", fontsize=SMALL,
                color=PRIMARY if v >= 0 else ACCENT, fontweight="bold")
    ax.set_xlim(-0.40, 0.92)          # room for the signed labels at both ends
    fig.tight_layout()
    save(fig, "l1-corr")
    return {k: float(v) for k, v in
            corr.sort_values(ascending=False).items()}


def attribute_combinations(train):
    """The three engineered ratios, and their correlation with the target."""
    t = train.copy()
    t["rooms_per_house"] = t["total_rooms"] / t["households"]
    t["bedrooms_ratio"] = t["total_bedrooms"] / t["total_rooms"]
    t["people_per_house"] = t["population"] / t["households"]
    corr = t.corr(numeric_only=True)["median_house_value"]
    return {k: float(corr[k]) for k in
            ["median_income", "rooms_per_house", "total_rooms",
             "bedrooms_ratio", "people_per_house", "latitude"]}


def label_clusters(train, n=5):
    """Which label values do districts actually pile up on? Count them.

    The deck used to name $450,000, $350,000 and $280,000, following the book's
    prose. Measured on our training split those are 31, 62 and 3 districts, and
    the median count across all distinct label values is 3 — so one of the three
    is a real stripe, one is marginal, and one is indistinguishable from the
    background. The stripes that are actually there sit on multiples of $12,500.
    """
    counts = train["median_house_value"].value_counts()
    background = float(counts.median())
    cap = float(counts.index.max())
    top = counts.drop(cap).head(n)
    # every stripe sits on a multiple of this — measured, not asserted
    modulus = reduce(gcd, [int(v) for v in top.index])
    return {
        "cap": cap,
        "cap_count": int(counts.loc[cap]),
        "background_count": background,
        "top": [[float(v), int(c)] for v, c in top.items()],
        "modulus": float(modulus),
        # as pairs, not a dict: dict keys are strings and would not be seen by
        # tools/check_provenance.py, so the slide quoting them would look
        # unsourced even though the script computes them
        "asserted_by_the_book": [[float(v), int(counts.get(v, 0))]
                                 for v in (450_000, 350_000, 280_000)],
    }


def fig_income_value(train, clusters):
    fig, ax = plt.subplots(figsize=(9.0, 5.4))
    ax.scatter(train["median_income"], train["median_house_value"], s=7,
               alpha=0.1, color=PRIMARY, linewidths=0)
    ax.set_xlabel("median_income"); ax.set_ylabel("median_house_value")
    ax.yaxis.set_major_formatter(usd)
    ax.set_title("The strongest predictor — and the artefacts in the label")
    ax.set_xlim(0, 15.5)
    cap, n_cap = clusters["cap"], clusters["cap_count"]
    ax.axhline(cap, color=ACCENT, lw=2.2)
    ax.text(15.3, cap - 14000, f"${cap:,.0f} cap — {n_cap} districts",
            va="top", ha="right", fontsize=SMALL, color=ACCENT,
            fontweight="bold", zorder=5,
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                      edgecolor="none"))
    for value, _ in clusters["top"]:
        ax.axhline(value, color=ACCENT, lw=1.2, ls="--", alpha=0.75)
    # the five stripes are $25,000 apart, so a label on each one overlaps its
    # neighbours: list them once instead, in the empty low-price/high-income
    # corner
    listing = "   ".join(f"${v:,.0f} ({c})" for v, c in clusters["top"][:3])
    listing2 = "   ".join(f"${v:,.0f} ({c})" for v, c in clusters["top"][3:])
    ax.text(0.985, 0.03,
            f"dashed — commonest label values, districts on each:\n"
            f"{listing}\n{listing2}"
            f"   — a typical value carries "
            f"{clusters['background_count']:.0f}",
            transform=ax.transAxes, va="bottom", ha="right", fontsize=SMALL,
            color="#33414d", linespacing=1.5,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor=RULE))
    fig.tight_layout()
    return save(fig, "l1-income-value", raster=True)


def fig_income_cat(h):
    cats = income_cats(h)
    counts = cats.value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(10.0, 3.6))
    ax.bar([str(c) for c in counts.index], counts.values, color=PRIMARY,
           width=0.62)
    ax.set_xlabel("income category")
    ax.set_ylabel("number of districts")
    ax.set_title("Stratify on the strongest predictor, not on a raw float")
    ax.grid(axis="x", alpha=0)
    for i, v in enumerate(counts.values):
        ax.text(i, v + 90, f"{v:,}", ha="center", fontsize=SMALL,
                color=PRIMARY, fontweight="bold")
    ax.set_ylim(0, counts.max() * 1.15)
    fig.tight_layout()
    save(fig, "l1-income-cat")
    return {str(k): int(v) for k, v in counts.items()}


def fig_strat_bias(h):
    """Sampling bias of a random split vs a stratified one — measured."""
    from sklearn.model_selection import train_test_split

    cats = income_cats(h)
    overall = cats.value_counts(normalize=True).sort_index()

    rand_tr, rand_te = train_test_split(h.assign(cat=cats), test_size=0.2,
                                        random_state=SEED)
    strat_tr, strat_te = train_test_split(h.assign(cat=cats), test_size=0.2,
                                          random_state=SEED, stratify=cats)
    rand = rand_te["cat"].value_counts(normalize=True).sort_index()
    strat = strat_te["cat"].value_counts(normalize=True).sort_index()

    err_rand = 100 * (rand / overall - 1)
    err_strat = 100 * (strat / overall - 1)

    x = np.arange(len(overall)); w = 0.38
    fig, ax = plt.subplots(figsize=(10.4, 3.9))
    ax.bar(x - w/2, err_rand, w, label="random split", color=ACCENT)
    ax.bar(x + w/2, err_strat, w, label="stratified split", color=SUCCESS)
    ax.axhline(0, color="#98a4b0", lw=1)
    ax.set_xticks(x, [str(c) for c in overall.index])
    ax.set_xlabel("income category")
    ax.set_ylabel("% error vs the full data")
    ax.set_title("Test-set composition error, by sampling method")
    ax.legend(loc="lower right", ncols=2)
    ax.grid(axis="x", alpha=0)
    fig.tight_layout()
    save(fig, "l1-strat-bias")
    return {"random_max_abs_pct": float(err_rand.abs().max()),
            "strat_max_abs_pct": float(err_strat.abs().max())}


def baseline(sp):
    """The trivial baseline: predict one constant, forever.

    Nothing in either lecture is interpretable without it. Reported three ways,
    because the deck quotes training RMSE in Lecture 1 and cross-validated and
    test RMSE in Lecture 2.
    """
    from sklearn.dummy import DummyRegressor
    from sklearn.metrics import root_mean_squared_error
    from sklearn.model_selection import cross_val_score

    y_tr, y_te = sp["y_train"], sp["y_test"]
    X_tr, X_te = sp["X_train"], sp["X_test"]
    mean_, median_ = float(y_tr.mean()), float(y_tr.median())

    out = {
        "train_mean": mean_,
        "train_median": median_,
        "train_q1": float(y_tr.quantile(0.25)),
        "train_q3": float(y_tr.quantile(0.75)),
        "mean_train_rmse": float(root_mean_squared_error(
            y_tr, np.full(len(y_tr), mean_))),
        "median_train_rmse": float(root_mean_squared_error(
            y_tr, np.full(len(y_tr), median_))),
        "mean_test_rmse": float(root_mean_squared_error(
            y_te, np.full(len(y_te), mean_))),
        "median_test_rmse": float(root_mean_squared_error(
            y_te, np.full(len(y_te), median_))),
    }
    folds = -cross_val_score(DummyRegressor(strategy="mean"), X_tr, y_tr,
                             scoring="neg_root_mean_squared_error",
                             cv=cv_splitter())
    out["mean_cv_rmse"] = float(folds.mean())
    out["mean_cv_std"] = float(folds.std())
    print(f"    constant = train mean   train={out['mean_train_rmse']:,.0f}  "
          f"cv={out['mean_cv_rmse']:,.0f}  test={out['mean_test_rmse']:,.0f}")
    print(f"    constant = train median train={out['median_train_rmse']:,.0f}  "
          f"                 test={out['median_test_rmse']:,.0f}")
    return out


def measure_leak(h, n_seeds=20):
    """How much does 'scale before split' actually cost? Measure, over seeds.

    A single split gives a single subtraction, and a single subtraction of two
    noisy numbers is not a measurement. Twenty splits give the distribution the
    difference is drawn from — which on this dataset is centred on zero.

    The reason is stated on the slide: standardisation is an invertible affine
    map, OLS is equivariant under one, and with 20,640 rows the training and
    full-data statistics nearly coincide. Remove any of the three and the leak
    has teeth.
    """
    from sklearn.decomposition import PCA
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import root_mean_squared_error
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    X = h.select_dtypes(include=[np.number]).drop(columns=["median_house_value"])
    y = h["median_house_value"].values

    def run(make_prep, make_model, seed):
        # LEAKY: preprocessing fitted on everything, then split
        Xall = make_prep().fit_transform(X)
        a_tr, a_te, ya_tr, ya_te = train_test_split(Xall, y, test_size=0.2,
                                                    random_state=seed)
        leaky = root_mean_squared_error(
            ya_te, make_model().fit(a_tr, ya_tr).predict(a_te))
        # CORRECT: split, then fit preprocessing on the training part only
        b_tr, b_te, yb_tr, yb_te = train_test_split(X, y, test_size=0.2,
                                                    random_state=seed)
        prep = make_prep()
        correct = root_mean_squared_error(
            yb_te, make_model().fit(prep.fit_transform(b_tr), yb_tr)
                              .predict(prep.transform(b_te)))
        return float(leaky), float(correct)

    base = lambda: make_pipeline(SimpleImputer(strategy="median"),
                                 StandardScaler())
    cases = {
        "linear_regression": (base, LinearRegression),
        "random_forest": (base, lambda: RandomForestRegressor(
            n_estimators=50, random_state=SEED, n_jobs=-1)),
        "pca_4": (lambda: make_pipeline(SimpleImputer(strategy="median"),
                                        StandardScaler(), PCA(n_components=4)),
                  LinearRegression),
    }
    # seed 42 first, so the number printed on the code slide is one of these
    seeds = [SEED] + list(range(1, n_seeds))
    out = {"n_seeds": n_seeds, "seeds": seeds}
    for name, (mp, mm) in cases.items():
        leaky, correct = zip(*(run(mp, mm, s) for s in seeds))
        leaky, correct = np.array(leaky), np.array(correct)
        d = leaky - correct                       # negative = leak looks better
        out[name] = {
            "leaky_seed42": float(leaky[0]),
            "correct_seed42": float(correct[0]),
            # what a single split would have let you report, which is the point
            "diff_seed42": float(abs(leaky[0] - correct[0])),
            "diff_abs_mean": float(np.abs(d.mean())),
            "diff_abs_std": float(d.std(ddof=1)),
            "correct_mean": float(correct.mean()),
            "correct_std": float(correct.std(ddof=1)),
            "diff_mean": float(d.mean()),
            "diff_std": float(d.std(ddof=1)),
            "diff_sign_flips": bool((d > 0).any() and (d < 0).any()),
            "diff_abs_max": float(np.abs(d).max()),
        }
        print(f"    {name:20s} honest={correct.mean():9,.0f} "
              f"± {correct.std(ddof=1):6,.0f}   "
              f"leak cost={d.mean():+8.2f} ± {d.std(ddof=1):,.2f}"
              f"{'  (sign flips)' if out[name]['diff_sign_flips'] else ''}")
    return out


def scaling_ablation(sp):
    """Does dropping the scaler change the models in this lecture? Measure.

    'Without scaling the model will ignore income and obsess over room counts'
    is false for every model in Lectures 1 and 2: least squares by pseudoinverse
    is equivariant under an invertible affine map of the features, and trees
    are invariant under any monotone rescaling of a feature.
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import cross_val_score
    from sklearn.pipeline import Pipeline, make_pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.tree import DecisionTreeRegressor

    X_tr, y_tr = sp["X_train"], sp["y_train"]
    num = X_tr.select_dtypes(include=[np.number]).columns.tolist()
    cat = ["ocean_proximity"]

    def prep(scaled):
        steps = [SimpleImputer(strategy="median")]
        if scaled:
            steps.append(StandardScaler())
        return ColumnTransformer([
            ("num", make_pipeline(*steps), num),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat)])

    out = {}
    for name, est in [("linear_regression", LinearRegression()),
                      ("decision_tree", DecisionTreeRegressor(
                          random_state=SEED))]:
        row = {}
        for tag, scaled in [("scaled", True), ("unscaled", False)]:
            folds = -cross_val_score(
                Pipeline([("prep", prep(scaled)), ("model", est)]),
                X_tr, y_tr, scoring="neg_root_mean_squared_error",
                cv=cv_splitter())
            row[tag] = float(folds.mean())
        row["diff"] = abs(row["scaled"] - row["unscaled"])
        out[name] = row
        print(f"    {name:20s} scaled={row['scaled']:9,.2f}  "
              f"unscaled={row['unscaled']:9,.2f}  diff={row['diff']:,.2f}")
    return out


# ---------------------------------------------------------------- lecture 2

def build_models(sp):
    """Train the three models the lectures use, honestly, and return numbers."""
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import cross_val_score
    from sklearn.pipeline import Pipeline
    from sklearn.tree import DecisionTreeRegressor

    X_tr, y_tr = sp["X_train"], sp["y_train"]
    prep, num, cat = preprocessing(X_tr)

    models = {
        "Linear regression": LinearRegression(),
        "Decision tree": DecisionTreeRegressor(random_state=SEED),
        "Random forest": RandomForestRegressor(n_estimators=100,
                                               random_state=SEED, n_jobs=-1),
    }

    res = {}
    for name, est in models.items():
        pipe = Pipeline([("prep", prep), ("model", est)])
        pipe.fit(X_tr, y_tr)
        train_rmse = float(np.sqrt(np.mean((pipe.predict(X_tr) - y_tr) ** 2)))
        folds = -cross_val_score(pipe, X_tr, y_tr,
                                 scoring="neg_root_mean_squared_error",
                                 cv=cv_splitter())
        res[name] = {"train_rmse": train_rmse,
                     "cv_folds": folds.tolist(),
                     "cv_mean": float(folds.mean()),
                     "cv_std": float(folds.std())}
        print(f"    {name:20s} train={train_rmse:9,.0f}  "
              f"cv={folds.mean():9,.0f} ± {folds.std():,.0f}")
    # the exact block the slide prints
    res["tree_describe"] = pd.Series(
        res["Decision tree"]["cv_folds"]).describe().to_string()
    return res


def paired_folds(res):
    """Compare models fold by fold, not mean against mean.

    The three models are evaluated on the *same* ten folds, so the honest
    comparison is the paired difference. Its spread is far smaller than the
    spread of either model's fold scores, because the fold-to-fold variation is
    mostly the composition of the held-out fold, which both models share.
    """
    from scipy import stats
    names = ["Linear regression", "Decision tree", "Random forest"]
    out = {}
    for a, b in [("Decision tree", "Linear regression"),
                 ("Random forest", "Linear regression")]:
        d = np.array(res[a]["cv_folds"]) - np.array(res[b]["cv_folds"])
        t = stats.ttest_rel(res[a]["cv_folds"], res[b]["cv_folds"])
        out[f"{a} - {b}"] = {"mean": float(d.mean()),
                             "abs_mean": float(abs(d.mean())),
                             "std": float(d.std(ddof=1)),
                             "p_value": float(t.pvalue),
                             "folds_where_first_is_better": int((d < 0).sum()),
                             "all_same_sign": bool((d > 0).all() or (d < 0).all())}
        print(f"    {a} - {b}: {d.mean():+,.0f} ± {d.std(ddof=1):,.0f} per fold, "
              f"p={t.pvalue:.4f}, same sign in every fold="
              f"{out[f'{a} - {b}']['all_same_sign']}")
    out["fold_std"] = {n: float(np.std(res[n]["cv_folds"], ddof=1))
                       for n in names}
    return out


def normal_equation(sp):
    """Thread 1, on our own data: the closed form against Scikit-Learn's."""
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LinearRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler, add_dummy_feature

    housing_num = sp["X_train"].select_dtypes(include=[np.number])
    X = make_pipeline(SimpleImputer(strategy="median"),
                      StandardScaler()).fit_transform(housing_num)
    y = sp["y_train"].values

    X_b = add_dummy_feature(X)
    theta_best = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y

    lin_reg = LinearRegression().fit(X, y)
    sk = np.r_[lin_reg.intercept_, lin_reg.coef_]

    residual = X_b @ theta_best - y
    orth = float(np.abs(X_b.T @ residual).max())
    scale = float(np.abs(X_b.T @ y).max())
    print(f"    n features={X.shape[1]}  max|theta - sklearn|="
          f"{np.abs(theta_best - sk).max():.3e}  max|X.T r|={orth:.3e}"
          f"  (scale of X.T y: {scale:.3e}, ratio {orth / scale:.1e})")
    return {"features": housing_num.columns.tolist(),
            "orthogonality_scale": scale,
            "orthogonality_ratio": orth / scale,
            "theta": [float(v) for v in theta_best],
            "sklearn_intercept": float(lin_reg.intercept_),
            "sklearn_coef": [float(v) for v in lin_reg.coef_],
            "max_abs_diff": float(np.abs(theta_best - sk).max()),
            "orthogonality_max": orth,
            "n_instances": int(X.shape[0]), "n_features": int(X.shape[1])}


def fig_train_vs_cv(res, base):
    names = ["Linear regression", "Decision tree", "Random forest"]
    tr = [res[n]["train_rmse"] for n in names]
    cv = [res[n]["cv_mean"] for n in names]
    # wider than the 0.8 rule of thumb: six value labels need horizontal room
    x = np.arange(3); w = 0.34
    fig, ax = plt.subplots(figsize=(10.8, 4.4))
    b1 = ax.bar(x - w/2, tr, w, label="RMSE on training data", color="#9fb8ca")
    b2 = ax.bar(x + w/2, cv, w, label="RMSE, 10-fold cross-validation",
                color=PRIMARY)
    ax.set_xticks(x, names)
    ax.yaxis.set_major_formatter(usd)
    ax.set_ylabel("RMSE")
    ax.set_title("The training number and the honest number")
    ax.grid(axis="x", alpha=0)
    b0 = base["mean_cv_rmse"]
    # the training label goes inside its bar: for linear regression the two
    # bars are the same height to within $49, so two labels above them touch
    for rect, v in zip(b1, tr):
        if v > b0 * 0.10:
            ax.text(rect.get_x() + rect.get_width()/2, v - b0 * 0.022,
                    f"${v:,.0f}", ha="center", va="top", fontsize=SMALL,
                    fontweight="bold", color="#16212b")
        else:
            ax.text(rect.get_x() + rect.get_width()/2, v + b0 * 0.022,
                    f"${v:,.0f}", ha="center", fontsize=SMALL,
                    fontweight="bold", color=PRIMARY)
    for rect, v in zip(b2, cv):
        ax.text(rect.get_x() + rect.get_width()/2, v + b0 * 0.022,
                f"${v:,.0f}", ha="center", fontsize=SMALL, fontweight="bold",
                color=PRIMARY)
    ax.annotate("a tree that memorises\nits training set\nscores exactly 0",
                xy=(1 - w/2, 1200), xytext=(1 - w/2 - 0.12, 26_000),
                color=ACCENT,
                fontsize=SMALL, fontweight="bold", ha="center",
                arrowprops=dict(arrowstyle="->", color=ACCENT, lw=1.8))
    # the anchor: predicting one constant
    ax.axhline(b0, color=ACCENT, lw=2, ls="--")
    ax.text(-0.52, b0 * 0.975, f"predict the mean — ${b0:,.0f}", va="top",
            ha="left", fontsize=SMALL, color=ACCENT, fontweight="bold")
    ax.legend(loc="upper right", bbox_to_anchor=(1.0, 0.88))
    ax.set_xlim(-0.62, 2.6)
    ax.set_ylim(0, b0 * 1.10)
    fig.tight_layout()
    return save(fig, "l2-train-vs-cv")


def fig_cv_spread(res):
    names = ["Linear regression", "Decision tree", "Random forest"]
    fig, ax = plt.subplots(figsize=(10.4, 4.2))
    rng = np.random.default_rng(SEED)
    for i, n in enumerate(names):
        f = np.array(res[n]["cv_folds"])
        ax.scatter(rng.normal(i, 0.055, len(f)), f, s=52, color=PRIMARY,
                   alpha=0.7, zorder=3, linewidths=0)
        ax.hlines(f.mean(), i - 0.24, i + 0.24, color=ACCENT, lw=2.6, zorder=4)
        ax.text(i + 0.30, f.mean(), f"mean ${f.mean():,.0f}\nstd ${f.std():,.0f}",
                va="center", fontsize=SMALL, color=ACCENT, fontweight="bold")
    ax.set_xticks(range(3), names)
    ax.set_xlim(-0.45, 2.85)
    ax.yaxis.set_major_formatter(usd)
    ax.set_ylabel("RMSE per fold")
    ax.set_title("Ten folds each — report the spread, not only the mean")
    ax.grid(axis="x", alpha=0)
    fig.tight_layout()
    return save(fig, "l2-cv-spread")


GRID = {"model__max_features": [4, 6, 8, 10, 12],
        "model__n_estimators": [30, 100, 200]}


def run_grid(sp):
    """The grid search the slides show: max_features x n_estimators, 10 folds."""
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import GridSearchCV
    from sklearn.pipeline import Pipeline

    X_tr, y_tr = sp["X_train"], sp["y_train"]
    prep, *_ = preprocessing(X_tr)
    gs = GridSearchCV(
        Pipeline([("prep", prep),
                  ("model", RandomForestRegressor(random_state=SEED,
                                                  n_jobs=-1))]),
        GRID, cv=cv_splitter(), scoring="neg_root_mean_squared_error",
        n_jobs=-1)
    gs.fit(X_tr, y_tr)
    return gs


def fig_grid(gs):
    cv = pd.DataFrame(gs.cv_results_)
    piv = cv.pivot_table(index="param_model__max_features",
                         columns="param_model__n_estimators",
                         values="mean_test_score")
    Z = -piv.values
    fig, ax = plt.subplots(figsize=(9.4, 4.6))
    im = ax.imshow(Z, cmap="viridis_r", aspect="auto")
    ax.set_xticks(range(len(piv.columns)), [str(c) for c in piv.columns])
    ax.set_yticks(range(len(piv.index)), [str(i) for i in piv.index])
    ax.set_xlabel("n_estimators"); ax.set_ylabel("max_features")
    n = len(GRID["model__max_features"]) * len(GRID["model__n_estimators"])
    ax.set_title(f"Grid search: {n} combinations × {N_FOLDS} folds "
                 f"= {n * N_FOLDS} fits")
    ax.grid(False)
    best = np.unravel_index(np.argmin(Z), Z.shape)
    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            # pick text colour from the cell's own luminance, not by guesswork
            lum = np.dot(im.cmap(im.norm(Z[i, j]))[:3], [0.299, 0.587, 0.114])
            ax.text(j, i, f"${Z[i, j]:,.0f}", ha="center", va="center",
                    fontsize=SMALL, fontweight="bold",
                    color=ACCENT if (i, j) == best
                    else ("#16212b" if lum > 0.6 else "white"))
    ax.add_patch(plt.Rectangle((best[1] - .5, best[0] - .5), 1, 1,
                               fill=False, edgecolor=ACCENT, lw=3.5))
    fig.colorbar(im, ax=ax, label="CV RMSE")
    fig.tight_layout()
    save(fig, "l2-grid")
    bp = {k: int(v) for k, v in gs.best_params_.items()}
    on_edge = [k for k, v in bp.items() if v in (min(GRID[k]), max(GRID[k]))]
    return {"best_params": bp, "best_cv_rmse": float(-gs.best_score_),
            "best_on_grid_boundary": on_edge,
            "n_combinations": int(n), "n_fits": int(n * N_FOLDS)}


def tune_preprocessing(sp, best_params):
    """A preprocessing hyperparameter, tuned the same way — a real one.

    The imputation strategy is a hyperparameter of a step inside the
    ColumnTransformer inside the Pipeline; the double-underscore path reaches
    it. This is the honest version of 'you can tune preprocessing too'.
    """
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import GridSearchCV
    from sklearn.pipeline import Pipeline

    X_tr, y_tr = sp["X_train"], sp["y_train"]
    prep, *_ = preprocessing(X_tr)
    pipe = Pipeline([("prep", prep),
                     ("model", RandomForestRegressor(
                         random_state=SEED, n_jobs=-1,
                         **{k.split("__", 1)[1]: v
                            for k, v in best_params.items()}))])
    grid = {"prep__num__simpleimputer__strategy":
            ["median", "mean", "most_frequent"]}
    gs = GridSearchCV(pipe, grid, cv=cv_splitter(),
                      scoring="neg_root_mean_squared_error", n_jobs=-1)
    gs.fit(X_tr, y_tr)
    scores = {str(p["prep__num__simpleimputer__strategy"]): float(-s)
              for p, s in zip(gs.cv_results_["params"],
                              gs.cv_results_["mean_test_score"])}
    print(f"    imputer strategy: " +
          "  ".join(f"{k}={v:,.0f}" for k, v in scores.items()))
    return {"scores": scores,
            "best": str(gs.best_params_["prep__num__simpleimputer__strategy"]),
            "spread": float(max(scores.values()) - min(scores.values()))}


def fig_importance(gs):
    prep_fitted = gs.best_estimator_.named_steps["prep"]
    imp = gs.best_estimator_.named_steps["model"].feature_importances_
    names = [n.split("__", 1)[-1]
             for n in prep_fitted.get_feature_names_out()]
    order = np.argsort(imp)            # all thirteen: the smallest is
                                       # ocean_proximity_ISLAND, and the
                                       # next slide is about exactly that
    fig, ax = plt.subplots(figsize=(10.8, 4.4))
    # the rare category is drawn in the accent colour: the next slide is about
    # what those five districts do, and the figure should already point at it
    colors = [ACCENT if names[i].endswith("ISLAND") else PRIMARY for i in order]
    ax.barh([names[i] for i in order], imp[order], color=colors, height=0.66)
    for lbl, c in zip(ax.get_yticklabels(), colors):
        lbl.set_color(c)
    ax.set_xlabel("feature importance")
    ax.set_title("What the forest actually used")
    ax.grid(axis="y", alpha=0)
    for i, v in enumerate(imp[order]):
        # three decimals hides the one that matters: ISLAND is 0.0003
        ax.text(v + 0.004, i, f"{v:.4f}" if v < 0.01 else f"{v:.3f}",
                va="center", fontsize=SMALL, color=colors[i])
    ax.set_xlim(0, imp.max() * 1.18)
    fig.tight_layout()
    save(fig, "l2-importance")
    ranked = sorted(zip(imp.tolist(), names), reverse=True)
    return [[round(float(v), 4), n] for v, n in ranked]


def fig_test_ci(gs, sp):
    """Bootstrap the test RMSE.

    The squared errors are severely right-skewed — a capped, heavy-tailed
    target — so the t-interval on their mean is the wrong tool. The percentile
    bootstrap makes no distributional assumption.
    """
    from scipy import stats
    X_te, y_te = sp["X_test"], sp["y_test"]
    pred = gs.best_estimator_.predict(X_te)

    def rmse(squared_errors):
        return np.sqrt(np.mean(squared_errors))

    squared_errors = (pred - y_te) ** 2
    point = float(rmse(squared_errors))
    boot = stats.bootstrap([squared_errors], rmse, confidence_level=0.95,
                           random_state=SEED)
    lo, hi = (float(v) for v in boot.confidence_interval)

    fig, ax = plt.subplots(figsize=(10.4, 2.6))
    ax.errorbar([point], [0], xerr=[[point - lo], [hi - point]], fmt="o",
                color=PRIMARY, markersize=13, capsize=10, lw=2.6,
                capthick=2.6, zorder=3)
    ax.set_yticks([])
    ax.xaxis.set_major_formatter(usd)
    ax.set_xlim(lo - 6000, hi + 6000)
    ax.set_ylim(-1, 1)
    ax.set_title("Test RMSE with a 95% bootstrap confidence interval")
    ax.text(point, 0.30, f"${point:,.0f}", ha="center", fontsize=24.7,
            fontweight="bold", color=PRIMARY)
    ax.text(lo, -0.36, f"${lo:,.0f}", ha="center", fontsize=SMALL, color=MUTED)
    ax.text(hi, -0.36, f"${hi:,.0f}", ha="center", fontsize=SMALL, color=MUTED)
    ax.text(point, -0.72, "report the interval, not the point estimate",
            ha="center", fontsize=SMALL, color=ACCENT, fontweight="bold")
    ax.grid(axis="y", alpha=0)
    fig.tight_layout()
    save(fig, "l2-test-ci")
    return {"test_rmse": point, "ci_lo": lo, "ci_hi": hi,
            "ci_half_width": (hi - lo) / 2,
            "skew_squared_errors": float(stats.skew(squared_errors))}


def fig_residuals(gs, sp):
    X_te, y_te = sp["X_test"], sp["y_test"]
    pred = gs.best_estimator_.predict(X_te)
    fig, ax = plt.subplots(figsize=(6.8, 6.2))
    ax.scatter(y_te, pred, s=8, alpha=0.18, color=PRIMARY, linewidths=0)
    lim = [0, 520_000]
    ax.plot(lim, lim, color=ACCENT, lw=2, label="perfect prediction")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("actual median_house_value")
    ax.set_ylabel("predicted")
    ax.xaxis.set_major_formatter(usd); ax.yaxis.set_major_formatter(usd)
    ax.set_title("Where the model fails, and why the cap matters")
    ax.axvline(500_001, color=ACCENT, ls="--", lw=1.4, alpha=0.8)
    ax.annotate("every capped district sits on this line —\n"
                "the model cannot predict above the cap\nit was trained on",
                # the tip must land where the stripe is DENSE (y ~ 440-500k);
                # at y=300k it sat in the sparse part and read as pointing at
                # empty space
                xy=(500_500, 468_000), xytext=(14_000, 300_000), fontsize=SMALL,
                color=ACCENT, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.45", facecolor="white",
                          edgecolor=ACCENT, alpha=0.95),
                arrowprops=dict(arrowstyle="->", color=ACCENT, lw=1.8))
    # lower right: the upper left is where the callout goes, and at 15px the
    # two collided
    ax.legend(loc="lower right")
    fig.tight_layout()
    return save(fig, "l2-residuals", raster=True)


def error_analysis(gs, sp, base):
    """The error analysis the deck promises: worst cases, and slices.

    Géron, Chapter 2, 'Analyze the Best Models and Their Errors'.
    """
    from sklearn.metrics import root_mean_squared_error

    X_te, y_te = sp["X_test"], sp["y_test"]
    pred = gs.best_estimator_.predict(X_te)
    err = pred - y_te

    df = X_te.copy()
    df["actual"] = y_te
    df["predicted"] = pred
    df["error"] = err
    df["abs_error"] = err.abs()
    df["income_cat"] = income_cats(X_te)

    worst = df.nlargest(10, "abs_error")
    worst_rows = [{
        "actual": float(r.actual), "predicted": float(r.predicted),
        "error": float(r.error), "abs_error": float(r.abs_error),
        "income_cat": int(r.income_cat),
        "ocean": str(r.ocean_proximity),
        "median_income": float(r.median_income),
    } for r in worst.itertuples()]

    def slice_rmse(by):
        out = {}
        for key, g in df.groupby(by, observed=True):
            out[str(key)] = {
                "n": int(len(g)),
                "rmse": float(root_mean_squared_error(g["actual"],
                                                      g["predicted"])),
                "median_actual": float(g["actual"].median()),
            }
        return out

    by_income = slice_rmse("income_cat")
    by_ocean = slice_rmse("ocean_proximity")

    capped = df["actual"] >= CAP
    uncapped_rmse = float(root_mean_squared_error(
        df.loc[~capped, "actual"], df.loc[~capped, "predicted"]))
    # the honest comparison for the reduced set: the same constant baseline,
    # re-measured on the same reduced set
    uncapped_base = float(root_mean_squared_error(
        df.loc[~capped, "actual"],
        np.full(int((~capped).sum()), base["train_mean"])))

    out = {
        "worst10": worst_rows,
        "worst10_capped": int(sum(r["actual"] >= CAP for r in worst_rows)),
        "by_income_cat": by_income,
        "by_ocean_proximity": by_ocean,
        "n_capped_test": int(capped.sum()),
        "n_capped_train": int((sp["y_train"] >= CAP).sum()),
        "rmse_excluding_capped": uncapped_rmse,
        "baseline_excluding_capped": uncapped_base,
        "worst_income_cat": max(by_income, key=lambda k: by_income[k]["rmse"]),
        "worst_ocean": max(by_ocean, key=lambda k: by_ocean[k]["rmse"]),
    }
    print(f"    worst single error ${worst_rows[0]['error']:,.0f} on a district "
          f"worth ${worst_rows[0]['actual']:,.0f}")
    for k, v in by_income.items():
        print(f"      income_cat {k}: n={v['n']:5d}  rmse=${v['rmse']:,.0f}")
    for k, v in by_ocean.items():
        print(f"      {k:12s}: n={v['n']:5d}  rmse=${v['rmse']:,.0f}")
    print(f"    excluding the {out['n_capped_test']} capped test districts: "
          f"${uncapped_rmse:,.0f}  (baseline on the same rows "
          f"${uncapped_base:,.0f})")
    return out


def fig_error_slices(ea, overall):
    """RMSE by income category and by ocean proximity, with counts.

    Horizontal bars: at the 15px legibility floor, five vertical bars per panel
    cannot carry a value label each without colliding, and the category names
    ("<1H OCEAN") do not fit under a tick either. Laid on their side both fit
    with room to spare.

    The two panels share an x-axis. They must: the figure exists to compare
    these groups with each other and with the one reported number, so a bar's
    length has to mean the same thing in both.
    """
    inc = ea["by_income_cat"]
    oce = ea["by_ocean_proximity"]
    ks = sorted(inc, key=int, reverse=True)
    ks2 = sorted(oce, key=lambda k: oce[k]["rmse"])
    top = max([v["rmse"] for v in inc.values()] +
              [v["rmse"] for v in oce.values()]) * 1.30

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.8, 3.8), sharex=True)

    def panel(ax, keys, data, colors, title, label):
        y = range(len(keys))
        ax.barh(y, [data[k]["rmse"] for k in keys], color=colors, height=0.66,
                zorder=2)
        ax.set_yticks(y, [label(k) for k in keys])
        for lbl, c in zip(ax.get_yticklabels(), colors):
            lbl.set_color(c)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0)
        ax.set_axisbelow(True)
        ax.xaxis.set_major_formatter(usd)
        ax.set_xlim(0, top)
        for i, k in enumerate(keys):
            ax.text(data[k]["rmse"] + top * 0.015, i,
                    f"${data[k]['rmse']:,.0f}", va="center", fontsize=SMALL,
                    color=colors[i], fontweight="bold", zorder=5,
                    # white ground: the reference line must not strike a label
                    bbox=dict(boxstyle="round,pad=0.12", facecolor="white",
                              edgecolor="none"))
        ax.axvline(overall, color=ACCENT, lw=1.8, ls="--", zorder=3)

    panel(a1, ks, inc, [PRIMARY] * len(ks), "RMSE by income category",
          lambda k: f"{k}  ({inc[k]['n']:,})")
    # a group too small to trust is drawn in the accent colour, not hidden
    oce_colors = [ACCENT if oce[k]["n"] < 50 else PRIMARY for k in ks2]
    panel(a2, ks2, oce, oce_colors, "RMSE by ocean_proximity",
          lambda k: f"{k}  ({oce[k]['n']:,})")
    a1.set_xlabel("test RMSE        (districts in the group in brackets)")
    a2.set_xlabel(f"dashed: the one reported number, ${overall:,.0f}")
    a2.xaxis.label.set_color(ACCENT)
    fig.tight_layout()
    return save(fig, "l2-error-slices")


def absolute_vs_relative(gs, sp, base):
    """The brief states a relative criterion; RMSE is absolute. Measure both."""
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import root_mean_squared_error
    from sklearn.pipeline import Pipeline

    X_tr, y_tr = sp["X_train"], sp["y_train"]
    X_te, y_te = sp["X_test"], sp["y_test"]

    pred = gs.best_estimator_.predict(X_te)
    ape = np.abs(pred - y_te) / y_te
    out = {
        "rmse": float(root_mean_squared_error(y_te, pred)),
        "mae": float(np.mean(np.abs(pred - y_te))),
        "median_ape_pct": float(100 * np.median(ape)),
        "within_30pct": float(100 * np.mean(ape <= 0.30)),
        "rmse_cheapest_decile": float(root_mean_squared_error(
            y_te[y_te <= y_te.quantile(0.1)],
            pred[(y_te <= y_te.quantile(0.1)).values])),
        "rmse_dearest_decile": float(root_mean_squared_error(
            y_te[y_te >= y_te.quantile(0.9)],
            pred[(y_te >= y_te.quantile(0.9)).values])),
        "target_median": float(y_tr.median()),
        "thirty_pct_of_median": float(0.30 * y_tr.median()),
    }

    # the alternative the slide names: regress the log of the target
    prep, *_ = preprocessing(X_tr)
    best = {k.split("__", 1)[1]: v for k, v in gs.best_params_.items()}
    logm = Pipeline([("prep", prep),
                     ("model", RandomForestRegressor(random_state=SEED,
                                                     n_jobs=-1, **best))])
    logm.fit(X_tr, np.log(y_tr))
    log_pred = np.exp(logm.predict(X_te))
    log_ape = np.abs(log_pred - y_te) / y_te
    out["log_target"] = {
        "rmse": float(root_mean_squared_error(y_te, log_pred)),
        "median_ape_pct": float(100 * np.median(log_ape)),
        "within_30pct": float(100 * np.mean(log_ape <= 0.30)),
    }
    print(f"    RMSE ${out['rmse']:,.0f}   MAE ${out['mae']:,.0f}   "
          f"median APE {out['median_ape_pct']:.1f}%   "
          f"within 30%: {out['within_30pct']:.1f}%")
    print(f"    cheapest decile RMSE ${out['rmse_cheapest_decile']:,.0f}   "
          f"dearest decile RMSE ${out['rmse_dearest_decile']:,.0f}")
    print(f"    log target: RMSE ${out['log_target']['rmse']:,.0f}  "
          f"median APE {out['log_target']['median_ape_pct']:.1f}%  "
          f"within 30%: {out['log_target']['within_30pct']:.1f}%")
    return out


def island_check(h, sp):
    """Pay off the ISLAND warning: what a rare category actually does.

    Measured, not asserted. The silent failure is real, but on this dataset it
    needs help to fire — so we say how rare it is, and then show the mechanism
    directly.
    """
    from sklearn.model_selection import KFold
    from sklearn.preprocessing import OneHotEncoder

    X_tr = sp["X_train"]
    n_total = int((h["ocean_proximity"] == "ISLAND").sum())
    n_train = int((X_tr["ocean_proximity"] == "ISLAND").sum())
    n_test = int((sp["X_test"]["ocean_proximity"] == "ISLAND").sum())

    # how the folds actually fall, with the splitter the lectures use
    is_island = (X_tr["ocean_proximity"] == "ISLAND").values
    per_fold = []
    for tr_idx, va_idx in cv_splitter().split(X_tr):
        per_fold.append({"train": int(is_island[tr_idx].sum()),
                         "val": int(is_island[va_idx].sum())})
    starved = [f for f in per_fold if f["train"] == 0 and f["val"] > 0]

    # and how often that happens at all, over many shufflings
    hits = 0
    trials = 500
    for s in range(trials):
        kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=s)
        if any(is_island[tr].sum() == 0 and is_island[va].sum() > 0
               for tr, va in kf.split(X_tr)):
            hits += 1

    # the mechanism, shown directly: fit without ISLAND, transform an ISLAND row
    enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    seen = X_tr.loc[~is_island, ["ocean_proximity"]]
    enc.fit(seen)
    row = enc.transform(pd.DataFrame({"ocean_proximity": ["ISLAND"]}))
    out = {
        "n_total": n_total, "n_train": n_train, "n_test": n_test,
        "categories_seen": [str(c) for c in enc.categories_[0]],
        "encoded_island": [float(v) for v in row[0]],
        "encoded_sum": float(row.sum()),
        "folds_without_island_in_training": len(starved),
        "pct_of_shufflings_with_a_starved_fold": 100.0 * hits / trials,
        "per_fold": per_fold,
    }
    print(f"    ISLAND: {n_total} districts — {n_train} train, {n_test} test")
    print(f"    folds whose training part has no ISLAND: {len(starved)}/"
          f"{N_FOLDS};  over {trials} shufflings: "
          f"{out['pct_of_shufflings_with_a_starved_fold']:.1f}%")
    print(f"    an unseen category encodes to {row[0].tolist()} — sum "
          f"{row.sum():.0f}, and no warning")
    return out


# ---------------------------------------------------------------- driver

# --------------------------------------------------- the on-slide text floor

# Mirrors .fig-wide / .fig-tall in assets/css/custom.css. If those caps change,
# change these.
CANVAS_W, CAP_WIDE, CAP_TALL, FLOOR_PX = 1280, 420, 528, 15.0


def _natural_css_px(path: Path) -> tuple[float, float]:
    """The size the browser gives a figure before any CSS scaling."""
    if path.suffix == ".png":
        w, h = struct.unpack(">II", path.read_bytes()[16:24])
        return float(w), float(h)          # one image pixel is one CSS pixel
    head = path.read_text()[:600]
    # matplotlib writes points; the hand-authored d-*.svg diagrams are unitless
    # (user units, which are CSS pixels already).
    pt = re.search(r'width="([\d.]+)pt"\s+height="([\d.]+)pt"', head)
    if pt:
        w, h = float(pt.group(1)), float(pt.group(2))
        return w * 96 / 72, h * 96 / 72    # the browser converts pt at 96/72
    plain = re.search(r'width="([\d.]+)"\s+height="([\d.]+)"', head)
    if plain:
        return float(plain.group(1)), float(plain.group(2))
    # NO viewBox fallback. I added one, reasoning that viewBox gives the
    # intrinsic size per the SVG spec — it does not, for an SVG loaded through
    # <img>. Measured in Chrome: a diagram with only a viewBox reports
    # naturalWidth 300 (the CSS replaced-element default) and renders at 0x0,
    # completely invisible on the slide. The fallback made this check pass
    # against dimensions the browser never uses, which is worse than no check.
    raise RuntimeError(
        f"{path.name}: no width/height on the <svg> tag. A viewBox alone is not "
        f"an intrinsic size for an <img>, and the figure renders blank. Add "
        f"width and height matching the viewBox.")


def check_text_floor() -> list[str]:
    """Refuse to ship a figure whose smallest label lands under the floor.

    Two conversions have to line up and neither is obvious, which is why this is
    a check rather than a paragraph in TRICKS:

      * matplotlib font sizes are POINTS. A point becomes 96/72 = 1.33 CSS px in
        an SVG, but dpi/72 = 2.22 px in a PNG rasterised at 160 — so a raster is
        authored 1.67x larger for the same `fontsize`.
      * the slide then scales the image uniformly (uniformly only since the
        .fig-wide aspect bug was fixed) by min(1280/w, cap/h).

    Today nothing is under the floor, but only because the ~0.5x downscale of
    the near-square rasters happens to cancel the 1.67x from dpi=160. Change the
    dpi, the caps, or save a square figure as SVG instead of PNG, and text drops
    under the floor with nothing to say so.
    """
    used: dict[str, str] = {}          # figure filename -> the figure's class
    for deck in sorted(ROOT.glob("slides/lecture-[0-9][0-9].html")):
        html = deck.read_text()
        for m in re.finditer(r'<figure class="([^"]*)"[^>]*>\s*<img\s+src="([^"]+)"',
                             html):
            used[Path(m.group(2)).name] = m.group(1)

    problems = []
    for name, classes in sorted(used.items()):
        path = OUT / name
        if not path.is_file():
            continue
        nw, nh = _natural_css_px(path)
        cap = CAP_TALL if "fig-tall" in classes else CAP_WIDE
        scale = min(CANVAS_W / nw, cap / nh)
        px_per_pt = (160 if path.suffix == ".png" else 96) / 72
        on_slide = SMALL * px_per_pt * scale
        if on_slide < FLOOR_PX:
            problems.append(
                f"{name}: smallest text renders at {on_slide:.1f}px on the "
                f"slide (floor {FLOOR_PX:.0f}). Widen it, or raise its sizes.")
    return problems


def main():
    setup()
    load_cache()
    print("Loading California housing…")
    h = load_housing()
    sp = split(h)

    facts = {
        "n_rows": int(len(h)),
        "n_missing_bedrooms": int(h["total_bedrooms"].isna().sum()),
        "n_train": int(len(sp["train"])),
        "n_test": int(len(sp["test"])),
        "n_capped": int((h["median_house_value"] >= CAP).sum()),
        "cap_value": float(h["median_house_value"].max()),
        "target_min": float(h["median_house_value"].min()),
        "target_median": float(h["median_house_value"].median()),
        "target_q1": float(h["median_house_value"].quantile(0.25)),
        "target_q3": float(h["median_house_value"].quantile(0.75)),
        "income_min": float(h["median_income"].min()),
        "income_max": float(h["median_income"].max()),
        "total_rooms_min": float(h["total_rooms"].min()),
        "total_rooms_max": float(h["total_rooms"].max()),
        "ocean_counts": {str(k): int(v) for k, v in
                         h["ocean_proximity"].value_counts().items()},
    }
    facts["pct_capped"] = 100.0 * facts["n_capped"] / facts["n_rows"]

    print("Lecture 1 figures:")
    fig_histograms(h)
    fig_hist_annotated(h)
    facts["income_cat_counts"] = fig_income_cat(h)
    facts.update(fig_strat_bias(h))
    # exploration happens on the training set, after the split
    fig_geo(sp["train"])
    facts["corr_with_target"] = fig_corr(sp["train"])
    facts["corr_with_combinations"] = attribute_combinations(sp["train"])
    facts["label_clusters"] = label_clusters(sp["train"])
    fig_income_value(sp["train"], facts["label_clusters"])

    print("The trivial baseline (Lecture 1, before the commitment):")
    facts["baseline"] = cached("baseline", lambda: baseline(sp))

    print("Cost of the scale-before-split leak, over 20 seeds:")
    facts["leak_cost"] = cached("leak_multiseed", lambda: measure_leak(h))

    print("Does scaling change these models at all?")
    facts["scaling"] = cached("scaling_ablation", lambda: scaling_ablation(sp))

    print("Lecture 2 — the three models:")
    res = cached("models", lambda: build_models(sp))
    for n in ["Linear regression", "Decision tree", "Random forest"]:
        print(f"    {n:20s} train={res[n]['train_rmse']:9,.0f}  "
              f"cv={res[n]['cv_mean']:9,.0f} ± {res[n]['cv_std']:,.0f}")

    print("The same folds, compared pairwise:")
    facts["paired_folds"] = paired_folds(res)

    print("Thread 1 on our own data:")
    facts["normal_equation"] = cached("normal_equation",
                                      lambda: normal_equation(sp))

    print("Grid search:")
    gs = cached("grid", lambda: run_grid(sp))

    print("Lecture 2 figures:")
    fig_train_vs_cv(res, facts["baseline"])
    fig_cv_spread(res)
    facts.update(fig_grid(gs))
    facts["importance"] = fig_importance(gs)
    facts["preprocessing_tuning"] = cached(
        "prep_tuning", lambda: tune_preprocessing(sp, gs.best_params_))
    facts.update(fig_test_ci(gs, sp))
    fig_residuals(gs, sp)

    print("Error analysis:")
    ea = cached("error_analysis",
                lambda: error_analysis(gs, sp, facts["baseline"]))
    fig_error_slices(ea, facts["test_rmse"])
    facts["error_analysis"] = ea

    print("Absolute error against the relative criterion in the brief:")
    facts["relative"] = cached(
        "absolute_vs_relative",
        lambda: absolute_vs_relative(gs, sp, facts["baseline"]))

    print("The ISLAND category:")
    facts["island"] = cached("island", lambda: island_check(h, sp))

    facts["tree_describe"] = res["tree_describe"]
    facts["models"] = {n: {"train_rmse": res[n]["train_rmse"],
                           "cv_mean": res[n]["cv_mean"],
                           "cv_std": res[n]["cv_std"],
                           "cv_folds": res[n]["cv_folds"]}
                       for n in ["Linear regression", "Decision tree",
                                 "Random forest"]}
    # how much of the baseline's error each model removes
    b = facts["baseline"]["mean_cv_rmse"]
    facts["reduction_vs_baseline_pct"] = {
        n: 100.0 * (1 - facts["models"][n]["cv_mean"] / b)
        for n in facts["models"]}
    facts["reduction_vs_baseline_pct"]["Tuned forest (test)"] = 100.0 * (
        1 - facts["test_rmse"] / facts["baseline"]["mean_test_rmse"])

    # Differences the slides quote in their own right. A gap between two
    # exported numbers is still a number on a slide, so it is exported too.
    m = facts["models"]
    facts["gaps"] = {
        "linear_cv_minus_train": m["Linear regression"]["cv_mean"]
                                 - m["Linear regression"]["train_rmse"],
        "tuning_gain": m["Random forest"]["cv_mean"] - facts["best_cv_rmse"],
        "test_minus_best_cv": facts["test_rmse"] - facts["best_cv_rmse"],
        "log_target_rmse_penalty": (facts["relative"]["log_target"]["rmse"]
                                    - facts["relative"]["rmse"]),
        "n_test_uncapped": facts["n_test"]
                           - facts["error_analysis"]["n_capped_test"],
        "n_bedrooms_nonnull": facts["n_rows"] - facts["n_missing_bedrooms"],
        "island_importance_rank_from_bottom": 1,
    }

    if (floor_problems := check_text_floor()):
        print("\nfigures whose text lands under the on-slide floor:")
        for p_ in floor_problems:
            print("  " + p_)
        raise SystemExit(1)

    # MERGE, do not overwrite. figures.json is shared with every
    # tools/figures_appNN.py, so writing it wholesale here would silently
    # delete several hundred values belonging to other lectures, and the only
    # symptom would be a flood of provenance failures with no obvious cause.
    out = OUT / "figures.json"
    existing = json.loads(out.read_text()) if out.is_file() else {}
    clobbered = {k for k in facts if k in existing and existing[k] != facts[k]}
    existing.update(facts)
    out.write_text(json.dumps(existing, indent=2))
    print(f"\nNumbers merged into {out.relative_to(ROOT)} "
          f"({len(facts)} from Lectures 1-2, {len(existing)} total)")
    if clobbered:
        raise SystemExit(
            f"\nfigures.json collision: {len(clobbered)} key(s) already held a "
            f"different value —\n    {', '.join(sorted(clobbered))}\n"
            f"Another script wrote them, and merging here destroys its numbers "
            f"silently. This happened once: figures_app04.py exported a bare "
            f"n_train/n_test for CoverType, this file overwrote them with "
            f"housing's, and Lectures 7-8 quoted values figures.json no longer "
            f"contained.\nPrefix the other script's keys with its lecture "
            f"(l07_n_train) and re-run it.")


if __name__ == "__main__":
    main()
