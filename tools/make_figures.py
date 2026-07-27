#!/usr/bin/env python3
"""
Generate every data figure used in the slide decks.

Figures are produced from the real datasets, with the same code paths the
lectures describe, so a number printed on a slide can always be reproduced by
re-running this script. Nothing here is illustrative-only.

    python3 tools/make_figures.py

Output: assets/figures/*.svg (vector, for anything sparse)
        assets/figures/*.png (raster @160dpi, for dense scatter plots where
                              SVG would embed 20k+ path elements)
"""

from __future__ import annotations

import json
import tarfile
import urllib.request
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "assets" / "figures"
CACHE = Path("/private/tmp/claude-501/aiml-data")

# Palette shared with assets/css/custom.css
PRIMARY = "#0b3d62"
ACCENT = "#c0392b"
SUCCESS = "#1e8449"
MATH = "#6c3483"
MUTED = "#6b7280"
RULE = "#d5dbe1"
SOFT = "#f4f7f9"

SEED = 42


def setup():
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "font.size": 13,
        "axes.titlesize": 15,
        "axes.titleweight": "bold",
        "axes.titlecolor": PRIMARY,
        "axes.labelsize": 13,
        "axes.labelcolor": "#16212b",
        "axes.edgecolor": "#98a4b0",
        "axes.linewidth": 1.0,
        "axes.grid": True,
        "grid.color": RULE,
        "grid.linewidth": 0.8,
        "xtick.color": "#33414d",
        "ytick.color": "#33414d",
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.frameon": False,
        "legend.fontsize": 11,
        "figure.autolayout": False,
    })


def save(fig, name, *, raster=False, dpi=160):
    ext = "png" if raster else "svg"
    path = OUT / f"{name}.{ext}"
    fig.savefig(path, format=ext, dpi=dpi, bbox_inches="tight",
                pad_inches=0.15)
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


# ---------------------------------------------------------------- lecture 1

def fig_histograms(h):
    cols = ["longitude", "latitude", "housing_median_age", "total_rooms",
            "total_bedrooms", "population", "households", "median_income",
            "median_house_value"]
    fig, axes = plt.subplots(3, 3, figsize=(13, 8))
    for ax, c in zip(axes.ravel(), cols):
        ax.hist(h[c].dropna(), bins=50, color=PRIMARY, edgecolor="white",
                linewidth=0.25)
        ax.set_title(c, fontsize=12)
        ax.tick_params(labelsize=9)
        ax.grid(alpha=0.55)
    # the two features the slides call out
    axes.ravel()[7].set_title("median_income", color=ACCENT, fontsize=12)
    axes.ravel()[8].set_title("median_house_value", color=ACCENT, fontsize=12)
    fig.suptitle("housing.hist(bins=50)", fontsize=13, color=MUTED, y=1.005)
    fig.tight_layout()
    return save(fig, "l1-histograms")


def fig_hist_annotated(h):
    """The same two histograms, enlarged, with the cap and the scaling marked."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.6))

    a1.hist(h["median_income"], bins=50, color=PRIMARY, edgecolor="white",
            linewidth=0.3)
    a1.set_title("median_income — not dollars, and capped")
    a1.set_xlabel("median_income")
    a1.axvline(15.0001, color=ACCENT, lw=2)
    a1.annotate("capped at 15.0001", xy=(15.0001, a1.get_ylim()[1] * 0.55),
                xytext=(10.2, a1.get_ylim()[1] * 0.78), color=ACCENT,
                fontsize=11, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=ACCENT, lw=1.6))
    a1.text(0.97, 0.44, "≈ tens of thousands of dollars:\n3 means about $30,000",
            transform=a1.transAxes, ha="right", fontsize=10.5, color="#33414d",
            bbox=dict(boxstyle="round,pad=0.45", facecolor="white",
                      edgecolor=RULE))

    a2.hist(h["median_house_value"], bins=50, color=PRIMARY, edgecolor="white",
            linewidth=0.3)
    a2.set_title("median_house_value — the label is capped")
    a2.set_xlabel("median_house_value")
    a2.xaxis.set_major_formatter(usd)
    n_cap = int((h["median_house_value"] >= 500_000).sum())
    a2.axvline(500_001, color=ACCENT, lw=2)
    a2.annotate(f"{n_cap:,} districts pile up\nat the $500,001 cap",
                xy=(500_001, n_cap * 0.9),
                xytext=(250_000, a2.get_ylim()[1] * 0.72), color=ACCENT,
                fontsize=11, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=ACCENT, lw=1.6))
    fig.tight_layout()
    return save(fig, "l1-hist-annotated")


def fig_geo(h):
    # plain
    fig, ax = plt.subplots(figsize=(7.2, 7.6))
    ax.scatter(h["longitude"], h["latitude"], s=6, color=PRIMARY)
    ax.set_xlabel("longitude"); ax.set_ylabel("latitude")
    ax.set_title("alpha = 1.0  —  it looks like California, and that is all")
    fig.tight_layout()
    save(fig, "l1-geo-plain", raster=True)

    # alpha, annotated
    fig, ax = plt.subplots(figsize=(7.6, 7.8))
    ax.scatter(h["longitude"], h["latitude"], s=8, color=PRIMARY, alpha=0.2,
               linewidths=0)
    ax.set_xlabel("longitude"); ax.set_ylabel("latitude")
    ax.set_title("alpha = 0.2  —  the density structure appears")
    for name, (lon, lat), off in [
        ("Bay Area", (-122.3, 37.8), (-124.6, 39.3)),
        ("Central Valley", (-120.6, 36.9), (-119.2, 39.0)),
        ("Los Angeles", (-118.3, 34.05), (-120.9, 33.3)),
        ("San Diego", (-117.15, 32.75), (-119.4, 32.2)),
    ]:
        ax.annotate(name, xy=(lon, lat), xytext=off, fontsize=11,
                    color=ACCENT, fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color=ACCENT, lw=1.4))
    fig.tight_layout()
    save(fig, "l1-geo-alpha", raster=True)

    # price + population
    fig, ax = plt.subplots(figsize=(8.6, 7.4))
    sc = ax.scatter(h["longitude"], h["latitude"],
                    s=h["population"] / 100, c=h["median_house_value"],
                    cmap="jet", alpha=0.45, linewidths=0)
    ax.set_xlabel("longitude"); ax.set_ylabel("latitude")
    ax.set_title("radius = population,  colour = median house value")
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("median_house_value")
    cb.ax.yaxis.set_major_formatter(usd)
    fig.tight_layout()
    return save(fig, "l1-geo-price", raster=True)


def fig_corr(h):
    from sklearn.model_selection import train_test_split
    cats = pd.cut(h["median_income"], bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                  labels=[1, 2, 3, 4, 5])
    train, _ = train_test_split(h, test_size=0.2, random_state=SEED,
                                stratify=cats)
    corr = train.corr(numeric_only=True)["median_house_value"].drop(
        "median_house_value").sort_values()
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    colors = [ACCENT if v < 0 else PRIMARY for v in corr]
    ax.barh(corr.index, corr.values, color=colors, height=0.62)
    ax.axvline(0, color="#98a4b0", lw=1)
    ax.set_xlabel("Pearson correlation with median_house_value")
    ax.set_title("One predictor dominates  (training set)")
    ax.grid(axis="y", alpha=0)
    for i, v in enumerate(corr.values):
        ax.text(v + (0.012 if v >= 0 else -0.012), i, f"{v:+.3f}",
                va="center", ha="left" if v >= 0 else "right", fontsize=10.5,
                color=PRIMARY if v >= 0 else ACCENT, fontweight="bold")
    ax.set_xlim(-0.28, 0.83)
    fig.tight_layout()
    return save(fig, "l1-corr")


def fig_income_value(h):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(h["median_income"], h["median_house_value"], s=7, alpha=0.1,
               color=PRIMARY, linewidths=0)
    ax.set_xlabel("median_income"); ax.set_ylabel("median_house_value")
    ax.yaxis.set_major_formatter(usd)
    ax.set_title("The strongest predictor — and the artefacts in the label")
    ax.set_xlim(0, 15.5)
    for y, label, style in [
        (500_001, "$500,001 cap", dict(color=ACCENT, lw=2.2)),
        (450_000, "$450,000", dict(color=ACCENT, lw=1.2, ls="--", alpha=0.75)),
        (350_000, "$350,000", dict(color=ACCENT, lw=1.2, ls="--", alpha=0.75)),
        (280_000, "$280,000", dict(color=ACCENT, lw=1.2, ls="--", alpha=0.75)),
    ]:
        ax.axhline(y, **style)
        ax.text(15.3, y, label, va="center", ha="left", fontsize=10.5,
                color=ACCENT, fontweight="bold" if y > 500_000 else "normal")
    fig.tight_layout()
    return save(fig, "l1-income-value", raster=True)


def fig_income_cat(h):
    cats = pd.cut(h["median_income"],
                  bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                  labels=[1, 2, 3, 4, 5])
    counts = cats.value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    ax.bar([str(c) for c in counts.index], counts.values, color=PRIMARY,
           width=0.62)
    ax.set_xlabel("income category")
    ax.set_ylabel("number of districts")
    ax.set_title("Stratify on the strongest predictor, not on a raw float")
    ax.grid(axis="x", alpha=0)
    for i, v in enumerate(counts.values):
        ax.text(i, v + 90, f"{v:,}", ha="center", fontsize=11,
                color=PRIMARY, fontweight="bold")
    ax.set_ylim(0, counts.max() * 1.15)
    fig.tight_layout()
    return save(fig, "l1-income-cat")


def fig_strat_bias(h):
    """Sampling bias of a random split vs a stratified one — measured."""
    from sklearn.model_selection import train_test_split

    cats = pd.cut(h["median_income"], bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                  labels=[1, 2, 3, 4, 5])
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
    fig, ax = plt.subplots(figsize=(9.6, 4.6))
    ax.bar(x - w/2, err_rand, w, label="random split", color=ACCENT)
    ax.bar(x + w/2, err_strat, w, label="stratified split", color=SUCCESS)
    ax.axhline(0, color="#98a4b0", lw=1)
    ax.set_xticks(x, [str(c) for c in overall.index])
    ax.set_xlabel("income category")
    ax.set_ylabel("% error vs the full dataset")
    ax.set_title("Test-set composition error, by sampling method")
    ax.legend()
    ax.grid(axis="x", alpha=0)
    fig.tight_layout()
    save(fig, "l1-strat-bias")
    return {"random_max_abs_pct": float(err_rand.abs().max()),
            "strat_max_abs_pct": float(err_strat.abs().max())}


# ---------------------------------------------------------------- lecture 2

def build_models(h):
    """Train the three models the lectures use, honestly, and return numbers."""
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import cross_val_score, train_test_split
    from sklearn.pipeline import Pipeline, make_pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.tree import DecisionTreeRegressor

    cats = pd.cut(h["median_income"], bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                  labels=[1, 2, 3, 4, 5])
    train, test = train_test_split(h, test_size=0.2, random_state=SEED,
                                   stratify=cats)
    X_tr = train.drop("median_house_value", axis=1)
    y_tr = train["median_house_value"].copy()
    X_te = test.drop("median_house_value", axis=1)
    y_te = test["median_house_value"].copy()

    num = X_tr.select_dtypes(include=[np.number]).columns.tolist()
    cat = ["ocean_proximity"]
    prep = ColumnTransformer([
        ("num", make_pipeline(SimpleImputer(strategy="median"),
                              StandardScaler()), num),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
    ])

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
                                 scoring="neg_root_mean_squared_error", cv=10)
        res[name] = {"train_rmse": train_rmse,
                     "cv_folds": folds.tolist(),
                     "cv_mean": float(folds.mean()),
                     "cv_std": float(folds.std())}
        print(f"    {name:20s} train={train_rmse:9,.0f}  "
              f"cv={folds.mean():9,.0f} ± {folds.std():,.0f}")

    res["_split"] = (X_tr, y_tr, X_te, y_te, prep, num, cat)
    return res


def fig_train_vs_cv(res):
    names = ["Linear regression", "Decision tree", "Random forest"]
    tr = [res[n]["train_rmse"] for n in names]
    cv = [res[n]["cv_mean"] for n in names]
    x = np.arange(3); w = 0.36
    fig, ax = plt.subplots(figsize=(10, 5.2))
    b1 = ax.bar(x - w/2, tr, w, label="RMSE on training data", color="#9fb8ca")
    b2 = ax.bar(x + w/2, cv, w, label="RMSE, 10-fold cross-validation",
                color=PRIMARY)
    ax.set_xticks(x, names)
    ax.yaxis.set_major_formatter(usd)
    ax.set_ylabel("RMSE")
    ax.set_title("The training number and the honest number")
    ax.legend(loc="upper left")
    ax.grid(axis="x", alpha=0)
    for rect, v in list(zip(b1, tr)) + list(zip(b2, cv)):
        ax.text(rect.get_x() + rect.get_width()/2, v + 2200,
                f"${v:,.0f}", ha="center", fontsize=10.5, fontweight="bold",
                color=PRIMARY)
    ax.annotate("a tree that memorises\nits training set\nscores exactly 0",
                xy=(1 - w/2, 900), xytext=(1 - w/2, 26_000), color=ACCENT,
                fontsize=11, fontweight="bold", ha="center",
                arrowprops=dict(arrowstyle="->", color=ACCENT, lw=1.8))
    ax.set_ylim(0, max(cv) * 1.32)
    fig.tight_layout()
    return save(fig, "l2-train-vs-cv")


def fig_cv_spread(res):
    names = ["Linear regression", "Decision tree", "Random forest"]
    fig, ax = plt.subplots(figsize=(9.6, 5.0))
    rng = np.random.default_rng(SEED)
    for i, n in enumerate(names):
        f = np.array(res[n]["cv_folds"])
        ax.scatter(rng.normal(i, 0.055, len(f)), f, s=52, color=PRIMARY,
                   alpha=0.7, zorder=3, linewidths=0)
        ax.hlines(f.mean(), i - 0.24, i + 0.24, color=ACCENT, lw=2.6, zorder=4)
        ax.text(i + 0.30, f.mean(), f"mean ${f.mean():,.0f}\nstd ${f.std():,.0f}",
                va="center", fontsize=10.5, color=ACCENT, fontweight="bold")
    ax.set_xticks(range(3), names)
    ax.set_xlim(-0.45, 2.85)
    ax.yaxis.set_major_formatter(usd)
    ax.set_ylabel("RMSE per fold")
    ax.set_title("Ten folds each — report the spread, not only the mean")
    ax.grid(axis="x", alpha=0)
    fig.tight_layout()
    return save(fig, "l2-cv-spread")


def fig_grid(res):
    """Grid search over the forest — a real surface, not a sketch."""
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import GridSearchCV
    from sklearn.pipeline import Pipeline

    X_tr, y_tr, *_rest = res["_split"]
    prep = res["_split"][4]
    grid = {"model__max_features": [4, 6, 8, 10, 12],
            "model__n_estimators": [30, 100, 200]}
    gs = GridSearchCV(
        Pipeline([("prep", prep),
                  ("model", RandomForestRegressor(random_state=SEED,
                                                  n_jobs=-1))]),
        grid, cv=10, scoring="neg_root_mean_squared_error", n_jobs=-1)
    gs.fit(X_tr, y_tr)
    cv = pd.DataFrame(gs.cv_results_)
    piv = cv.pivot_table(index="param_model__max_features",
                         columns="param_model__n_estimators",
                         values="mean_test_score")
    Z = -piv.values

    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    im = ax.imshow(Z, cmap="viridis_r", aspect="auto")
    ax.set_xticks(range(len(piv.columns)), [str(c) for c in piv.columns])
    ax.set_yticks(range(len(piv.index)), [str(i) for i in piv.index])
    ax.set_xlabel("n_estimators"); ax.set_ylabel("max_features")
    ax.set_title("Grid search: 15 combinations × 10 folds = 150 fits")
    ax.grid(False)
    best = np.unravel_index(np.argmin(Z), Z.shape)
    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            is_best = (i, j) == best
            # pick text colour from the cell's own luminance, not by guesswork
            lum = np.dot(im.cmap(im.norm(Z[i, j]))[:3], [0.299, 0.587, 0.114])
            ax.text(j, i, f"${Z[i, j]:,.0f}", ha="center", va="center",
                    fontsize=11, fontweight="bold",
                    color=ACCENT if is_best
                    else ("#16212b" if lum > 0.6 else "white"))
    ax.add_patch(plt.Rectangle((best[1] - .5, best[0] - .5), 1, 1,
                               fill=False, edgecolor=ACCENT, lw=3.5))
    fig.colorbar(im, ax=ax, label="CV RMSE")
    fig.tight_layout()
    save(fig, "l2-grid")
    bp = {k: int(v) for k, v in gs.best_params_.items()}
    on_edge = [k for k, v in bp.items()
               if v in (min(grid[k]), max(grid[k]))]
    return {"best_params": bp, "best_cv_rmse": float(-gs.best_score_),
            "best_on_grid_boundary": on_edge, "gs": gs}


def fig_grid_from_cached(gs):
    """Redraw the grid heatmap from a completed search, without refitting."""
    cv = pd.DataFrame(gs.cv_results_)
    piv = cv.pivot_table(index="param_model__max_features",
                         columns="param_model__n_estimators",
                         values="mean_test_score")
    Z = -piv.values
    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    im = ax.imshow(Z, cmap="viridis_r", aspect="auto")
    ax.set_xticks(range(len(piv.columns)), [str(c) for c in piv.columns])
    ax.set_yticks(range(len(piv.index)), [str(i) for i in piv.index])
    ax.set_xlabel("n_estimators"); ax.set_ylabel("max_features")
    ax.set_title("Grid search: 15 combinations × 10 folds = 150 fits")
    ax.grid(False)
    best = np.unravel_index(np.argmin(Z), Z.shape)
    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            lum = np.dot(im.cmap(im.norm(Z[i, j]))[:3], [0.299, 0.587, 0.114])
            ax.text(j, i, f"${Z[i, j]:,.0f}", ha="center", va="center",
                    fontsize=11, fontweight="bold",
                    color=ACCENT if (i, j) == best
                    else ("#16212b" if lum > 0.6 else "white"))
    ax.add_patch(plt.Rectangle((best[1] - .5, best[0] - .5), 1, 1,
                               fill=False, edgecolor=ACCENT, lw=3.5))
    fig.colorbar(im, ax=ax, label="CV RMSE")
    fig.tight_layout()
    return save(fig, "l2-grid")


def fig_importance(gs, res):
    prep_fitted = gs.best_estimator_.named_steps["prep"]
    imp = gs.best_estimator_.named_steps["model"].feature_importances_
    names = list(prep_fitted.get_feature_names_out())
    names = [n.split("__", 1)[-1] for n in names]
    order = np.argsort(imp)[-12:]
    fig, ax = plt.subplots(figsize=(9.6, 5.4))
    ax.barh([names[i] for i in order], imp[order], color=PRIMARY, height=0.66)
    ax.set_xlabel("feature importance")
    ax.set_title("What the forest actually used")
    ax.grid(axis="y", alpha=0)
    for i, v in enumerate(imp[order]):
        ax.text(v + 0.004, i, f"{v:.3f}", va="center", fontsize=10,
                color=PRIMARY)
    ax.set_xlim(0, imp.max() * 1.18)
    fig.tight_layout()
    return save(fig, "l2-importance")


def fig_test_ci(gs, res):
    from scipy import stats
    X_tr, y_tr, X_te, y_te, *_ = res["_split"]
    pred = gs.best_estimator_.predict(X_te)
    err = (pred - y_te) ** 2
    rmse = float(np.sqrt(err.mean()))
    lo, hi = np.sqrt(stats.t.interval(0.95, len(err) - 1,
                                      loc=err.mean(),
                                      scale=stats.sem(err)))

    fig, ax = plt.subplots(figsize=(9.6, 3.4))
    ax.errorbar([rmse], [0], xerr=[[rmse - lo], [hi - rmse]], fmt="o",
                color=PRIMARY, markersize=13, capsize=10, lw=2.6,
                capthick=2.6, zorder=3)
    ax.set_yticks([])
    ax.xaxis.set_major_formatter(usd)
    ax.set_xlim(lo - 6000, hi + 6000)
    ax.set_ylim(-1, 1)
    ax.set_title("Test RMSE with a 95% confidence interval")
    ax.text(rmse, 0.30, f"${rmse:,.0f}", ha="center", fontsize=14,
            fontweight="bold", color=PRIMARY)
    ax.text(lo, -0.36, f"${lo:,.0f}", ha="center", fontsize=11, color=MUTED)
    ax.text(hi, -0.36, f"${hi:,.0f}", ha="center", fontsize=11, color=MUTED)
    ax.text(rmse, -0.72, "report the interval, not the point estimate",
            ha="center", fontsize=11, color=ACCENT, fontweight="bold")
    ax.grid(axis="y", alpha=0)
    fig.tight_layout()
    save(fig, "l2-test-ci")
    return {"test_rmse": rmse, "ci_lo": float(lo), "ci_hi": float(hi)}


def fig_residuals(gs, res):
    X_tr, y_tr, X_te, y_te, *_ = res["_split"]
    pred = gs.best_estimator_.predict(X_te)
    fig, ax = plt.subplots(figsize=(7.6, 7.0))
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
                xy=(498_000, 300_000), xytext=(20_000, 430_000), fontsize=10.5,
                color=ACCENT, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.45", facecolor="white",
                          edgecolor=ACCENT, alpha=0.95),
                arrowprops=dict(arrowstyle="->", color=ACCENT, lw=1.8))
    ax.legend(loc="upper left")
    fig.tight_layout()
    return save(fig, "l2-residuals", raster=True)


# ---------------------------------------------------------------- driver

def main():
    setup()
    import pickle
    cache = CACHE / "fits.pkl"
    print("Loading California housing…")
    h = load_housing()
    facts = {"n_rows": int(len(h)),
             "n_missing_bedrooms": int(h["total_bedrooms"].isna().sum())}

    print("Lecture 1 figures:")
    fig_histograms(h)
    fig_hist_annotated(h)
    fig_geo(h)
    fig_corr(h)
    fig_income_value(h)
    fig_income_cat(h)
    facts.update(fig_strat_bias(h))

    if cache.is_file():
        print("Lecture 2 — reusing cached fits (delete "
              f"{cache} to refit):")
        res, gs, grid = pickle.loads(cache.read_bytes())
        for n in ["Linear regression", "Decision tree", "Random forest"]:
            print(f"    {n:20s} train={res[n]['train_rmse']:9,.0f}  "
                  f"cv={res[n]['cv_mean']:9,.0f}")
        refit = False
    else:
        print("Lecture 2 — training models (this takes a few minutes):")
        res = build_models(h)
        refit = True
    print("Lecture 2 figures:")
    fig_train_vs_cv(res)
    fig_cv_spread(res)
    if refit:
        grid = fig_grid(res)
        gs = grid.pop("gs")
        cache.write_bytes(pickle.dumps((res, gs, grid)))
    else:
        fig_grid_from_cached(gs)
    fig_importance(gs, res)
    facts.update(fig_test_ci(gs, res))
    fig_residuals(gs, res)

    facts.update({k: v for k, v in grid.items()})
    facts["models"] = {n: {"train_rmse": res[n]["train_rmse"],
                           "cv_mean": res[n]["cv_mean"],
                           "cv_std": res[n]["cv_std"]}
                       for n in ["Linear regression", "Decision tree",
                                 "Random forest"]}
    out = OUT / "figures.json"
    out.write_text(json.dumps(facts, indent=2))
    print(f"\nNumbers written to {out.relative_to(ROOT)}:")
    print(json.dumps(facts, indent=2))


if __name__ == "__main__":
    main()
