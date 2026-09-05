#!/usr/bin/env python3
"""
Application 10 — demand forecasting. Every figure and every number quoted in
Lectures 19 and 20.

    python3 tools/figures_app10.py

Dataset: Chicago Transit Authority daily boarding totals, 2001-01-01 to
2021-11-30, as distributed with the textbook.

    https://github.com/ageron/data/raw/main/ridership.tgz

Two lectures:

  * Lecture 19 (Build) plots the series, finds the weekly and yearly
    seasonality, measures the naive forecasts, and evaluates a linear model and
    a small recurrent network **with a random cross-validation split**.
  * Lecture 20 (Fix) develops stationarity, differencing and autocorrelation,
    then re-measures the same models with time-based backtesting and reports
    the gap.

Everything expensive is cached in CACHE/fits-app10.pkl — this file keeps its
own cache rather than sharing make_figures.py's, so that two authors running
their scripts at the same time cannot corrupt each other's fits.

Read TRICKS §6 and §11.6 before adding a figure.
"""

from __future__ import annotations

import pickle
import tarfile
import time
import urllib.request
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

from figkit import (ACCENT, AXIS, MATH, MUTED, PRIMARY, RULE, SMALL, SUCCESS,
                    check_text_floor, export, save, setup)

ROOT = Path(__file__).resolve().parent.parent
CACHE = Path("/private/tmp/claude-501/aiml-data")
CACHE_FILE = CACHE / "fits-app10.pkl"

SEED = 42
WINDOW = 56                 # eight weeks of history, as the book uses
K = 5                       # folds, everywhere in this application

# The modelling pool. The book trains on 2016-2018 and validates on the first
# five months of 2019; we pool the two so that a random split and a time split
# see exactly the same rows, which is the whole point of Lecture 20.
POOL_FROM, POOL_TO = "2016-01", "2019-05"

# riders, in thousands, on an axis
riders = FuncFormatter(lambda v, _: f"{v/1000:,.0f}k" if v else "0")


# --------------------------------------------------------------------- cache

_cache: dict = {}


def load_cache() -> None:
    global _cache
    _cache = pickle.loads(CACHE_FILE.read_bytes()) if CACHE_FILE.is_file() else {}


def cached(key, fn):
    if key in _cache:
        print(f"    [cached] {key}")
        return _cache[key]
    print(f"    [computing] {key}")
    t0 = time.time()
    value = fn()
    _cache[key] = value
    CACHE_FILE.write_bytes(pickle.dumps(_cache))
    print(f"    [done] {key} in {time.time() - t0:.0f}s")
    return value


# ---------------------------------------------------------------------- data

def load_ridership() -> pd.DataFrame:
    """Exactly the loader shown on the slides, with a cache outside the repo."""
    tarball = CACHE / "ridership.tgz"
    if not tarball.is_file():
        CACHE.mkdir(parents=True, exist_ok=True)
        url = "https://github.com/ageron/data/raw/main/ridership.tgz"
        print(f"  downloading {url}")
        urllib.request.urlretrieve(url, tarball)
        with tarfile.open(tarball) as t:
            t.extractall(path=CACHE, filter="data")
    path = CACHE / "ridership" / "CTA_-_Ridership_-_Daily_Boarding_Totals.csv"
    df = pd.read_csv(path, parse_dates=["service_date"])
    df.columns = ["date", "day_type", "bus", "rail", "total"]
    df = df.sort_values("date").set_index("date")
    df = df.drop("total", axis=1).drop_duplicates()
    return df


def windows(series: pd.Series, w: int = WINDOW):
    """Every w-day window, with the value that follows it as the target.

    Returns X scaled by 1e6 (the book's scaling: it puts the values near the
    0-1 range, which suits the default weight initialisation), y likewise, and
    the date each target falls on.
    """
    v = series.values.astype(np.float64)
    n = len(v) - w
    X = np.stack([v[i:i + w] for i in range(n)]) / 1e6
    return X, v[w:] / 1e6, series.index[w:]


def mae(pred, true) -> float:
    """Mean absolute error, back in riders."""
    return float(np.abs(np.asarray(pred) - np.asarray(true)).mean() * 1e6)


def mape(pred, true) -> float:
    p, t = np.asarray(pred), np.asarray(true)
    return float((np.abs(p - t) / t).mean() * 100)


# ============================================================ LECTURE 19 ====

def fig_full_series(df):
    """Twenty-one years of it, monthly, so the whole shape is on one slide."""
    monthly = df[["bus", "rail"]].resample("ME").mean()
    fig, ax = plt.subplots(figsize=(10.6, 3.9))
    ax.plot(monthly.index, monthly["bus"], color=PRIMARY, lw=1.6, label="bus")
    ax.plot(monthly.index, monthly["rail"], color=ACCENT, lw=1.6, label="rail")
    ax.yaxis.set_major_formatter(riders)
    ax.set_ylabel("boardings per day")
    ax.set_title("Monthly mean daily boardings, 2001–2021")
    ax.legend(loc="lower left")
    ax.annotate("March 2020",
                xy=(pd.Timestamp("2020-04-15"), 120_000),
                xytext=(pd.Timestamp("2013-06-01"), 200_000),
                fontsize=SMALL, fontweight="bold", color=MUTED,
                bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                          edgecolor=RULE),
                arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.6))
    ax.axvspan(pd.Timestamp("2016-01-01"), pd.Timestamp("2019-06-01"),
               color=SUCCESS, alpha=0.10)
    ax.text(pd.Timestamp("2017-08-01"), 1_030_000, "the years we model",
            ha="center", fontsize=SMALL, color=SUCCESS, fontweight="bold")
    ax.set_ylim(0, 1_150_000)
    fig.tight_layout()
    return save(fig, "l19-series")


def fig_weekly(df):
    """Ten weeks of daily rail and bus: the sawtooth is the whole slide."""
    part = df[["bus", "rail"]]["2019-03":"2019-05"]
    fig, ax = plt.subplots(figsize=(10.6, 3.9))
    ax.plot(part.index, part["bus"], color=PRIMARY, lw=1.7, label="bus")
    ax.plot(part.index, part["rail"], color=ACCENT, lw=1.7, label="rail")
    # shade every weekend, so the reader does not have to count
    for d in pd.date_range(part.index.min(), part.index.max()):
        if d.dayofweek == 5:
            ax.axvspan(d, d + pd.Timedelta(days=2), color=AXIS, alpha=0.12,
                       linewidth=0)
    ax.yaxis.set_major_formatter(riders)
    ax.set_ylabel("boardings")
    ax.set_title("Daily boardings, March to May 2019 — shaded bands are weekends")
    ax.legend(loc="lower left", ncols=2)
    ax.set_ylim(0, 1_150_000)
    ax.annotate("Memorial Day",
                xy=(pd.Timestamp("2019-05-27"), 250_000),
                xytext=(pd.Timestamp("2019-04-14"), 90_000),
                fontsize=SMALL, fontweight="bold", color=MUTED,
                bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                          edgecolor=RULE),
                arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.6))
    fig.tight_layout()
    return save(fig, "l19-weekly")


def fig_yearly(df):
    """Monthly means and a 12-month rolling average: seasonality plus trend."""
    monthly = df[["bus", "rail"]].resample("ME").mean()["2001":"2019"]
    roll = monthly.rolling(window=12).mean()
    fig, ax = plt.subplots(figsize=(10.6, 3.9))
    ax.plot(monthly.index, monthly["rail"], color=ACCENT, lw=1.0, alpha=0.55,
            marker=".", ms=3, label="rail, monthly mean")
    ax.plot(roll.index, roll["rail"], color=ACCENT, lw=2.6,
            label="rail, 12-month rolling mean")
    ax.plot(monthly.index, monthly["bus"], color=PRIMARY, lw=1.0, alpha=0.45,
            marker=".", ms=3, label="bus, monthly mean")
    ax.plot(roll.index, roll["bus"], color=PRIMARY, lw=2.6,
            label="bus, 12-month rolling mean")
    ax.yaxis.set_major_formatter(riders)
    ax.set_ylabel("boardings per day")
    ax.set_title("The wobble is yearly seasonality; the smooth line is the trend")
    ax.legend(loc="lower left", fontsize=SMALL, ncols=2)
    ax.set_ylim(250_000, 1_050_000)
    fig.tight_layout()
    return save(fig, "l19-yearly")


def fig_naive(df):
    """The naive forecast, overlaid, with its errors underneath."""
    part = df["rail"]["2019-03":"2019-05"]
    naive = df["rail"].shift(7)["2019-03":"2019-05"]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(10.6, 5.0), sharex=True,
                                 height_ratios=[2.1, 1])
    a1.plot(part.index, part, color=PRIMARY, lw=1.9, label="actual")
    a1.plot(naive.index, naive, color=ACCENT, lw=1.7, ls="--",
            label="forecast = value seven days earlier")
    a1.yaxis.set_major_formatter(riders)
    a1.set_ylabel("rail boardings")
    a1.set_title("Copy last week — and it is nearly right")
    a1.legend(loc="lower left", fontsize=SMALL)
    a1.set_ylim(0, 900_000)

    err = (part - naive)
    a2.bar(err.index, err.values, color=AXIS, width=1.0)
    worst = err.abs().idxmax()
    a2.bar([worst], [err.loc[worst]], color=ACCENT, width=1.0)
    a2.yaxis.set_major_formatter(riders)
    a2.set_ylabel("error")
    a2.axhline(0, color=MUTED, lw=1)
    a2.annotate(f"{err.loc[worst]:,.0f} on the holiday",
                xy=(worst, err.loc[worst]),
                xytext=(pd.Timestamp("2019-03-20"), err.min() * 0.78),
                fontsize=SMALL, fontweight="bold", color=ACCENT,
                bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                          edgecolor=RULE),
                arrowprops=dict(arrowstyle="->", color=ACCENT, lw=1.6))
    fig.tight_layout()
    return save(fig, "l19-naive")


def baselines(df) -> dict:
    """Every forecast that requires no model at all, on the modelling pool."""
    pool = df["rail"][POOL_FROM:POOL_TO]
    X, y, dates = windows(pool)
    out = {
        "n_days": int(len(pool)),
        "n_windows": int(len(X)),
        "first_target": str(dates[0].date()),
        "last_target": str(dates[-1].date()),
        "pool_mean": float(pool.mean()),
        # a constant: the mean of everything before the last year
        "constant": mae(np.full(len(y), pool[:"2018-12"].mean() / 1e6), y),
        "naive_1": mae(X[:, -1], y),
        "naive_7": mae(X[:, -7], y),
        "naive_7_mape": mape(X[:, -7], y),
        "naive_14": mae(X[:, -14], y),
        "naive_1_mape": mape(X[:, -1], y),
    }
    # the book's own window, so the slides can quote a number a reader can check
    part = df["rail"]["2019-03":"2019-05"]
    naive = df["rail"].shift(7)["2019-03":"2019-05"]
    out["naive_7_book_window"] = float((part - naive).abs().mean())
    out["naive_7_book_window_mape"] = float(
        ((part - naive).abs() / part).mean() * 100)
    bus = df["bus"]["2019-03":"2019-05"]
    bus_naive = df["bus"].shift(7)["2019-03":"2019-05"]
    out["naive_7_bus_book_window"] = float((bus - bus_naive).abs().mean())
    out["naive_7_bus_book_window_mape"] = float(
        ((bus - bus_naive).abs() / bus).mean() * 100)
    # the worst single day
    err = (part - naive)
    out["worst_naive_day"] = str(err.abs().idxmax().date())
    out["worst_naive_error"] = float(err.loc[err.abs().idxmax()])
    out["worst_naive_error_abs"] = float(abs(err.loc[err.abs().idxmax()]))

    # Lecture 19's worked assistant failure: the same baseline written with the
    # shift pointing the wrong way, so that it reads the future. Scored on the
    # same rows, it is indistinguishable from the correct one.
    backwards = df["rail"].shift(-7).reindex(dates)
    out["naive_minus7"] = mae(backwards.values / 1e6, y)
    out["naive_shift_difference"] = abs(out["naive_minus7"] - out["naive_7"])

    # what a longer window would cost in rows
    out["rows_at_365"] = int(len(pool) - 365)
    return out


def fig_baselines(base):
    names = ["Predict a constant", "Copy the day before", "Copy two weeks ago",
             "Copy last week"]
    vals = [base["constant"], base["naive_1"], base["naive_14"], base["naive_7"]]
    fig, ax = plt.subplots(figsize=(10.6, 3.5))
    colours = [AXIS, AXIS, AXIS, SUCCESS]
    bars = ax.barh(names, vals, color=colours, height=0.62)
    for r, v in zip(bars, vals):
        ax.text(v + 3_000, r.get_y() + r.get_height() / 2, f"{v:,.0f}",
                va="center", fontsize=SMALL, fontweight="bold", color="#16212b")
    ax.xaxis.set_major_formatter(riders)
    ax.set_xlabel("MAE, riders per day")
    ax.set_title("Four forecasts that need no model at all")
    ax.set_xlim(0, max(vals) * 1.18)
    ax.grid(axis="y", alpha=0)
    ax.invert_yaxis()
    fig.tight_layout()
    return save(fig, "l19-baselines")


# ============================================================ LECTURE 20 ====

def linear_weights(df) -> dict:
    """What the fifty-six weights actually learned. Column j is lag 56 - j."""
    from sklearn.linear_model import LinearRegression
    X, y, _ = windows(df["rail"][POOL_FROM:POOL_TO])
    m = LinearRegression().fit(X, y)
    w = m.coef_
    lag = {56 - j: float(w[j]) for j in range(56)}
    top = sorted(lag.items(), key=lambda kv: -abs(kv[1]))[:6]
    return {
        "by_lag": lag,
        "top": [[int(k), round(v, 3)] for k, v in top],
        "sum": float(w.sum()),
        "intercept_riders": float(m.intercept_ * 1e6),
        "weekly_lags": [round(lag[k], 3) for k in (7, 14, 21, 28, 35, 42, 49)],
        "n_weekly_positive": int(sum(lag[k] > 0 for k in
                                     (7, 14, 21, 28, 35, 42, 49))),
    }


def fig_weights(lw):
    lags = np.arange(1, 57)
    vals = np.array([lw["by_lag"][k] for k in lags])
    weekly = (lags % 7 == 0)
    fig, ax = plt.subplots(figsize=(10.6, 3.6))
    ax.bar(lags[~weekly], vals[~weekly], color="#9fb8ca", width=0.72)
    ax.bar(lags[weekly], vals[weekly], color=ACCENT, width=0.72)
    ax.axhline(0, color=MUTED, lw=1)
    ax.set_xlabel("lag, in days before the day being predicted")
    ax.set_ylabel("weight")
    ax.set_title("Nobody told it that a week has seven days")
    ax.set_xticks([1, 7, 14, 21, 28, 35, 42, 49, 56])
    ax.annotate(f"lag 1: {lw['by_lag'][1]:+.2f}", xy=(1.4, lw["by_lag"][1]),
                xytext=(6, lw["by_lag"][1] * 0.80), fontsize=SMALL,
                fontweight="bold", color=MUTED,
                bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                          edgecolor=RULE),
                arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.6))
    ax.text(0.99, 0.06, "red: every multiple of seven", transform=ax.transAxes,
            ha="right", fontsize=SMALL, color=ACCENT, fontweight="bold")
    fig.tight_layout()
    return save(fig, "l19-weights")


def fig_stationarity(df):
    """Rolling mean and rolling standard deviation, which is the definition."""
    s = df["rail"]["2001":"2019"]
    roll = s.rolling(365, min_periods=365)
    fig, ax = plt.subplots(figsize=(10.6, 3.9))
    ax.plot(s.index, roll.mean(), color=PRIMARY, lw=2.4,
            label="rolling mean, 365 days")
    ax.plot(s.index, roll.std(), color=MATH, lw=2.4,
            label="rolling standard deviation, 365 days")
    ax.yaxis.set_major_formatter(riders)
    ax.set_ylabel("rail boardings")
    ax.set_title("Neither of these is constant — so the series is not stationary")
    ax.legend(loc="center left", fontsize=SMALL)
    ax.set_ylim(0, 800_000)
    lo, hi = roll.mean().min(), roll.mean().max()
    ax.annotate(f"the mean moves by {hi - lo:,.0f}",
                xy=(pd.Timestamp("2012-06-01"), 640_000),
                xytext=(pd.Timestamp("2003-06-01"), 700_000),
                fontsize=SMALL, fontweight="bold", color=PRIMARY,
                bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                          edgecolor=RULE),
                arrowprops=dict(arrowstyle="->", color=PRIMARY, lw=1.6))
    fig.tight_layout()
    return save(fig, "l20-stationarity")


def fig_differencing(df):
    """The level, the one-step difference and the seven-step difference."""
    s = df["rail"]["2019-03":"2019-05"]
    d1 = df["rail"].diff(1)["2019-03":"2019-05"]
    d7 = df["rail"].diff(7)["2019-03":"2019-05"]
    fig, axes = plt.subplots(3, 1, figsize=(10.6, 5.4), sharex=True)
    for ax, y_, c, t in [
            (axes[0], s, PRIMARY, "the series"),
            (axes[1], d1, ACCENT, "differenced once — the weekly cycle survives"),
            (axes[2], d7, SUCCESS, "differenced at lag 7 — the weekly cycle is gone")]:
        ax.plot(y_.index, y_.values, color=c, lw=1.8)
        ax.set_title(t, fontsize=SMALL)
        ax.yaxis.set_major_formatter(riders)
        ax.axhline(0, color=MUTED, lw=1)
    # panels 2 and 3 show the same quantity, so they share a scale (TRICKS 11.6)
    lim = max(d1.abs().max(), d7.abs().max()) * 1.1
    axes[1].set_ylim(-lim, lim)
    axes[2].set_ylim(-lim, lim)
    axes[0].set_ylim(0, 900_000)
    axes[2].set_xlabel("March to May 2019")
    fig.tight_layout()
    return save(fig, "l20-differencing")


def acf_values(df) -> dict:
    from statsmodels.tsa.stattools import acf, adfuller
    s = df["rail"][POOL_FROM:POOL_TO]
    a_level = acf(s.values, nlags=60, fft=False)
    a_d7 = acf(s.diff(7).dropna().values, nlags=60, fft=False)
    out = {
        "acf_level": [float(x) for x in a_level],
        "acf_diff7": [float(x) for x in a_d7],
        "rho_1": float(a_level[1]),
        "rho_7": float(a_level[7]),
        "rho_14": float(a_level[14]),
        "rho_56": float(a_level[56]),
        "rho_1_diff7": float(a_d7[1]),
        "rho_7_diff7": float(a_d7[7]),
    }
    for name, series in [("level", s), ("diff1", s.diff().dropna()),
                         ("diff7", s.diff(7).dropna())]:
        stat, p = adfuller(series.values, autolag="AIC")[:2]
        out[f"adf_{name}"] = float(stat)
        out[f"adf_{name}_p"] = float(p)
        out[f"std_{name}"] = float(series.std())
        out[f"mean_{name}"] = float(series.mean())
    return out


def naive_identity(df, af) -> dict:
    """Thread 10's payoff, checked against the data.

    For a weakly stationary series with variance s^2 and autocorrelation rho_k,
    the lag-k difference has variance 2 s^2 (1 - rho_k). The naive forecast's
    error *is* that difference, so the strength of the naive forecast is a
    statement about rho_k and nothing else.
    """
    s = df["rail"][POOL_FROM:POOL_TO]
    sd = float(s.std())
    out = {"sd": sd}
    for k in (1, 7):
        rho = af[f"rho_{k}"]
        out[f"predicted_sd_diff{k}"] = float(np.sqrt(2 * sd ** 2 * (1 - rho)))
        out[f"measured_sd_diff{k}"] = float(s.diff(k).std())
        out[f"measured_mae_diff{k}"] = float(s.diff(k).abs().mean())
    # a Gaussian of this width would give this MAE; ours is far smaller, so the
    # error distribution is not Gaussian
    out["gaussian_mae_diff7"] = float(
        np.sqrt(2 / np.pi) * out["measured_sd_diff7"])
    out["kurtosis_diff7"] = float(s.diff(7).kurtosis())
    return out


def regime_check(df) -> dict:
    """What each protocol says when the series changes regime.

    Not a leak: a regime change is a fact about the world. It is here because it
    is the one case where the two protocols disagree by a factor rather than by
    a percentage, and because a random split cannot see it at all.
    """
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import KFold, cross_val_score

    pool = df["rail"]["2016-01":"2021-11"]
    X, y, dates = windows(pool)
    shuffled = -cross_val_score(
        LinearRegression(), X, y, scoring="neg_mean_absolute_error",
        cv=KFold(K, shuffle=True, random_state=SEED)).mean() * 1e6

    tr = dates < "2020-01-01"
    te = (dates >= "2020-03-01") & (dates < "2020-07-01")
    m = LinearRegression().fit(X[tr], y[tr])
    return {
        "shuffled_cv_2016_2021": float(shuffled),
        "forward_2020": mae(m.predict(X[te]), y[te]),
        "naive_2020": mae(X[te, -7], y[te]),
        "level_2020": float(y[te].mean() * 1e6),
        "level_2019": float(df["rail"]["2019-03":"2019-06"].mean()),
        "n_test_2020": int(te.sum()),
    }


def fig_acf(af):
    lags = np.arange(0, 41)
    lvl = np.array(af["acf_level"])[:41]
    d7 = np.array(af["acf_diff7"])[:41]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(10.6, 5.0), sharex=True,
                                 sharey=True)
    for ax, vals, c, t in [
            (a1, lvl, PRIMARY, "autocorrelation of the series"),
            (a2, d7, SUCCESS, "autocorrelation after differencing at lag 7")]:
        ax.bar(lags, vals, color=c, width=0.62)
        ax.axhline(0, color=MUTED, lw=1)
        ax.set_title(t, fontsize=SMALL)
        ax.set_ylabel("correlation")
    for lag in (7, 14, 21, 28, 35):
        a1.bar([lag], [lvl[lag]], color=ACCENT, width=0.62)
    a1.annotate(f"lag 7: {lvl[7]:.2f}", xy=(7, lvl[7]),
                xytext=(11, 1.02), fontsize=SMALL, fontweight="bold",
                color=ACCENT,
                bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                          edgecolor=RULE),
                arrowprops=dict(arrowstyle="->", color=ACCENT, lw=1.6))
    a1.annotate(f"lag 1: {lvl[1]:.2f}", xy=(1, lvl[1]),
                xytext=(2.2, -0.85), fontsize=SMALL, fontweight="bold",
                color=PRIMARY,
                bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                          edgecolor=RULE),
                arrowprops=dict(arrowstyle="->", color=PRIMARY, lw=1.6))
    a2.set_xlabel("lag, in days")
    a1.set_ylim(-1.15, 1.35)
    fig.tight_layout()
    return save(fig, "l20-acf")


# --------------------------------------------------------------- the models

def linear_scores(df) -> dict:
    """The linear model under every protocol we discuss. Cheap; no cache."""
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import KFold, TimeSeriesSplit, cross_val_score

    pool = df["rail"][POOL_FROM:POOL_TO]
    X, y, dates = windows(pool)
    out: dict = {}

    # 1 · what the assistant's cross_val_score returns, over 20 seeds
    seeds = []
    for seed in range(20):
        cv = KFold(K, shuffle=True, random_state=seed)
        seeds.append(-cross_val_score(LinearRegression(), X, y, cv=cv,
                                      scoring="neg_mean_absolute_error").mean()
                     * 1e6)
    out["random_cv_seeds"] = [float(s) for s in seeds]
    out["random_cv"] = float(np.mean(seeds))
    out["random_cv_std"] = float(np.std(seeds))

    cv = KFold(K, shuffle=True, random_state=SEED)
    folds = -cross_val_score(LinearRegression(), X, y, cv=cv,
                             scoring="neg_mean_absolute_error") * 1e6
    out["random_cv_folds"] = [float(f) for f in folds]

    # 2 · the same call with a time-aware splitter
    ts = -cross_val_score(LinearRegression(), X, y, cv=TimeSeriesSplit(K),
                          scoring="neg_mean_absolute_error") * 1e6
    out["timeseries_cv_folds"] = [float(f) for f in ts]
    out["timeseries_cv"] = float(ts.mean())

    # 3 · the split a deployed system actually faces
    tr = dates < "2019-01-01"
    m = LinearRegression().fit(X[tr], y[tr])
    out["honest_split"] = mae(m.predict(X[~tr]), y[~tr])
    out["honest_split_mape"] = mape(m.predict(X[~tr]), y[~tr])
    out["honest_n_train"] = int(tr.sum())
    out["honest_n_test"] = int((~tr).sum())
    out["honest_naive"] = mae(X[~tr, -7], y[~tr])
    out["pool_naive"] = mae(X[:, -7], y)

    # 4 · the confound audit: is the gap the protocol, or the period?
    #     the naive forecast needs no fitting, so its MAE measures only how hard
    #     the rows are. Skill = model MAE / naive MAE on the SAME rows.
    naive_random = []
    for tr_i, te_i in KFold(K, shuffle=True, random_state=SEED).split(X):
        naive_random.append(mae(X[te_i, -7], y[te_i]))
    out["random_cv_naive"] = float(np.mean(naive_random))
    out["random_cv_skill"] = out["random_cv"] / out["random_cv_naive"]
    out["honest_skill"] = out["honest_split"] / out["honest_naive"]

    # 5 · rolling-origin backtest at a FIXED training size, which removes the
    #     other confound: TimeSeriesSplit's first fold trains on a fifth of the
    #     data and is therefore not comparable with anything.
    L, T = 700, 98
    starts = [700, 798, 896, 994, 1092]
    fm, fn_, spans = [], [], []
    for s in starts:
        te = np.arange(s, min(s + T, len(X)))
        trn = np.arange(s - L, s)
        m = LinearRegression().fit(X[trn], y[trn])
        fm.append(mae(m.predict(X[te]), y[te]))
        fn_.append(mae(X[te, -7], y[te]))
        spans.append(f"{dates[te[0]].date()} to {dates[te[-1]].date()}")
    out["rolling_folds"] = [float(f) for f in fm]
    out["rolling_naive_folds"] = [float(f) for f in fn_]
    out["rolling_spans"] = spans
    out["rolling"] = float(np.mean(fm))
    out["rolling_naive"] = float(np.mean(fn_))
    out["rolling_skill"] = out["rolling"] / out["rolling_naive"]
    out["rolling_train_size"] = L

    # the matched random split: same training size, same test size, 20 draws
    from sklearn.model_selection import ShuffleSplit
    rm, rn = [], []
    for tr_i, te_i in ShuffleSplit(20, train_size=L, test_size=T,
                                   random_state=SEED).split(X):
        m = LinearRegression().fit(X[tr_i], y[tr_i])
        rm.append(mae(m.predict(X[te_i]), y[te_i]))
        rn.append(mae(X[te_i, -7], y[te_i]))
    out["matched_random"] = float(np.mean(rm))
    out["matched_random_naive"] = float(np.mean(rn))
    out["matched_random_skill"] = out["matched_random"] / out["matched_random_naive"]
    out["skill_gap"] = out["rolling_skill"] - out["matched_random_skill"]
    out["skill_gap_pct"] = 100 * out["skill_gap"] / out["matched_random_skill"]

    # 6 · the four-protocol table, each protocol scored against a baseline
    #     measured on ITS OWN test rows. The four protocols score four
    #     different sets of days, so one shared denominator would compare four
    #     models against four different periods -- the exact mistake this
    #     lecture exists to teach against.
    def naive_on_folds(splitter) -> float:
        return float(np.mean([mae(X[te, -7], y[te])
                              for _, te in splitter.split(X)]))

    cut = int(len(X) * 0.8)
    m_ho = LinearRegression().fit(X[:cut], y[:cut])
    gap_folds = -cross_val_score(LinearRegression(), X, y,
                                 cv=TimeSeriesSplit(K, gap=WINDOW),
                                 scoring="neg_mean_absolute_error") * 1e6
    rows = [
        ("random 5-fold", float(folds.mean()),
         naive_on_folds(KFold(K, shuffle=True, random_state=SEED))),
        ("one forward hold-out", mae(m_ho.predict(X[cut:]), y[cut:]),
         mae(X[cut:, -7], y[cut:])),
        ("forward 5-fold", out["timeseries_cv"],
         naive_on_folds(TimeSeriesSplit(K))),
        ("forward 5-fold + purge", float(gap_folds.mean()),
         naive_on_folds(TimeSeriesSplit(K, gap=WINDOW))),
    ]
    out["protocols"] = [{"name": n, "mae": s, "naive": v,
                         "margin_pct": 100.0 * (v - s) / v} for n, s, v in rows]
    out["protocol_naive_all"] = mae(X[:, -7], y)
    _claimed = out["protocols"][0]["margin_pct"]
    _real = out["protocols"][-1]["margin_pct"]
    out["protocol_share_pct"] = 100.0 * (_claimed - _real) / _claimed

    # 7 · Lecture 16 scores its models on the forward test slice, and its
    #     six-model ladder on the rows from 2019-01-01. Two more row sets, two
    #     more baselines, for the same reason.
    out["rnn_cut"] = int(cut)
    out["rnn_n_test"] = int(len(X) - cut)
    out["rnn_test_naive"] = mae(X[cut:, -7], y[cut:])
    out["rnn_test_target"] = out["rnn_test_naive"] * 0.9
    _lad = dates >= "2019-01-01"
    out["ladder_n_test"] = int(_lad.sum())
    out["ladder_naive"] = mae(X[_lad, -7], y[_lad])
    # ...and its MAPE, for the same reason the MAE needed one: a percentage
    # quoted against a baseline measured over different days is the same
    # defect wearing a percent sign.
    out["ladder_naive_mape"] = mape(X[_lad, -7], y[_lad])
    return out


# ------------------------------------------------------- the recurrent models
#
# Everything below runs on the CPU. That is not an oversight: this model is
# tiny and the sequences are short, so the run is dominated by per-step kernel
# launches rather than by arithmetic, and Apple's MPS backend is measurably
# slower than the CPU here. The script measures both and exports the pair, so
# the slide can quote it instead of asserting it.

def _torch():
    import torch
    import torch.nn as nn
    return torch, nn


def make_arrays(frame: pd.DataFrame, w: int, target: int = 0, horizon: int = 1):
    """Windows of `w` rows, labelled with the next `horizon` target values."""
    V = frame.values.astype(np.float32)
    n = len(V) - w - horizon + 1
    X = np.stack([V[i:i + w] for i in range(n)])
    y = np.stack([V[i + w:i + w + horizon, target] for i in range(n)])
    return X, y, frame.index[w:w + n]


def build_model(kind: str, input_size: int, output_size: int = 1):
    torch, nn = _torch()

    class Rnn(nn.Module):
        def __init__(self, cell, layers=1, hidden=32):
            super().__init__()
            self.rnn = cell(input_size, hidden, num_layers=layers,
                            batch_first=True)
            self.head = nn.Linear(hidden, output_size)

        def forward(self, x):
            o, _ = self.rnn(x)
            return self.head(o[:, -1])

    class ConvGru(nn.Module):
        """A 1D convolution halves the sequence before the recurrent layer."""

        def __init__(self, hidden=32):
            super().__init__()
            self.conv = nn.Conv1d(input_size, hidden, kernel_size=4, stride=2)
            self.gru = nn.GRU(hidden, hidden, batch_first=True)
            self.head = nn.Linear(hidden, output_size)

        def forward(self, x):
            z = torch.relu(self.conv(x.permute(0, 2, 1))).permute(0, 2, 1)
            o, _ = self.gru(z)
            return self.head(o[:, -1])

    if kind == "linear":
        return nn.Sequential(nn.Flatten(), nn.LazyLinear(output_size))
    if kind == "rnn":
        return Rnn(nn.RNN)
    if kind == "deep":
        return Rnn(nn.RNN, layers=3)
    if kind == "lstm":
        return Rnn(nn.LSTM)
    if kind == "gru":
        return Rnn(nn.GRU)
    if kind == "convgru":
        return ConvGru()
    raise ValueError(kind)


RECIPES = {"sgd": ("sgd", 0.05, 200), "adam": ("adam", 0.005, 120)}


def fit_torch(kind, Xtr, ytr, recipe, seed=SEED, device="cpu"):
    torch, nn = _torch()
    torch.manual_seed(seed)
    model = build_model(kind, Xtr.shape[2], ytr.shape[1]).to(device)
    opt_name, lr, epochs = RECIPES[recipe]
    Xt = torch.tensor(Xtr).to(device)
    yt = torch.tensor(ytr).to(device)
    if kind == "linear":                       # LazyLinear needs one pass
        model(Xt[:2])
    opt = (torch.optim.Adam(model.parameters(), lr=lr) if opt_name == "adam"
           else torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9))
    loss_fn = nn.HuberLoss(delta=0.05)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(Xt, yt), batch_size=32, shuffle=True,
        generator=torch.Generator().manual_seed(seed))
    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            opt.step()
    model.eval()
    return model


def predict_torch(model, X, device="cpu"):
    torch, _ = _torch()
    with torch.no_grad():
        return model(torch.tensor(X).to(device)).cpu().numpy()


def select_and_fit(kind, X, y, dates, cut_fit, cut_end, device="cpu"):
    """Choose the recipe on a slice of the TRAINING period, then refit on all
    of it. Nothing after `cut_end` is touched here."""
    sel = dates < cut_fit
    trn = dates < cut_end
    scores = {}
    for recipe in RECIPES:
        m = fit_torch(kind, X[sel], y[sel], recipe, device=device)
        held = trn & ~sel
        scores[recipe] = mae(predict_torch(m, X[held], device)[:, 0], y[held, 0])
    best = min(scores, key=scores.get)
    model = fit_torch(kind, X[trn], y[trn], best, device=device)
    return model, best, {k: float(v) for k, v in scores.items()}


def torch_models(df) -> dict:
    """Every recurrent model in the two lectures, measured honestly."""
    torch, _ = _torch()
    torch.set_num_threads(4)

    rail = df["rail"][POOL_FROM:POOL_TO].to_frame() / 1e6
    mul = df[["rail", "bus"]][POOL_FROM:POOL_TO] / 1e6
    mul["next_day_type"] = df["day_type"].shift(-1)[POOL_FROM:POOL_TO]
    mul = pd.get_dummies(mul, dtype=float)

    CUT_SEL, CUT_END = "2018-07-01", "2019-01-01"
    out: dict = {"recipes": {k: list(v) for k, v in RECIPES.items()},
                 "mulvar_columns": list(mul.columns)}

    Xu, yu, du = make_arrays(rail, WINDOW)
    Xm, ym, dm = make_arrays(mul, WINDOW)
    out["n_windows"] = int(len(Xu))
    out["mulvar_input_size"] = int(Xm.shape[2])

    ladder = [("linear", Xu, yu, du, "Linear, 56 lags"),
              ("rnn", Xu, yu, du, "Simple RNN, 32 units"),
              ("deep", Xu, yu, du, "Deep RNN, 3 layers"),
              ("rnn", Xm, ym, dm, "Simple RNN, multivariate"),
              ("gru", Xm, ym, dm, "GRU, multivariate"),
              ("lstm", Xm, ym, dm, "LSTM, multivariate")]

    out["ladder"] = []
    for kind, X, y, dates, label in ladder:
        t0 = time.time()
        model, best, sel_scores = select_and_fit(kind, X, y, dates,
                                                 CUT_SEL, CUT_END)
        te = dates >= CUT_END
        pred = predict_torch(model, X[te])[:, 0]
        tr = dates < CUT_END
        row = {
            "label": label, "kind": kind, "recipe": best,
            "selection": sel_scores,
            "inputs": int(X.shape[2]),
            "test_mae": mae(pred, y[te, 0]),
            "test_mape": mape(pred, y[te, 0]),
            "train_mae": mae(predict_torch(model, X[tr])[:, 0], y[tr, 0]),
            "seconds": round(time.time() - t0, 1),
        }
        out["ladder"].append(row)
        print(f"      {label:28s} {row['test_mae']:9,.0f}  "
              f"({best}, {row['seconds']}s)")

    # --- the simple RNN under the two protocols, which is the Lecture 19/20 pair
    from sklearn.model_selection import KFold
    folds = []
    for tr_i, te_i in KFold(K, shuffle=True, random_state=SEED).split(Xu):
        m = fit_torch("rnn", Xu[tr_i], yu[tr_i], "sgd")
        folds.append(mae(predict_torch(m, Xu[te_i])[:, 0], yu[te_i, 0]))
    out["rnn_random_cv_folds"] = [float(f) for f in folds]
    out["rnn_random_cv"] = float(np.mean(folds))
    out["rnn_random_cv_naive"] = float(np.mean(
        [mae(Xu[te_i, -7, 0], yu[te_i, 0])
         for _, te_i in KFold(K, shuffle=True, random_state=SEED).split(Xu)]))
    print(f"      RNN random 5-fold CV        {out['rnn_random_cv']:9,.0f}")

    # rolling-origin backtest for the same model, at a fixed training size
    L, T = 700, 98
    roll, roll_naive = [], []
    for s in (700, 798, 896, 994, 1092):
        te = np.arange(s, min(s + T, len(Xu)))
        trn = np.arange(s - L, s)
        m = fit_torch("rnn", Xu[trn], yu[trn], "sgd")
        roll.append(mae(predict_torch(m, Xu[te])[:, 0], yu[te, 0]))
        roll_naive.append(mae(Xu[te, -7, 0], yu[te, 0]))
    out["rnn_rolling_folds"] = [float(f) for f in roll]
    out["rnn_rolling"] = float(np.mean(roll))
    out["rnn_rolling_naive"] = float(np.mean(roll_naive))
    print(f"      RNN rolling backtest        {out['rnn_rolling']:9,.0f}")

    # the linear model, same two protocols, for the Lecture 19 table
    lin_folds = []
    for tr_i, te_i in KFold(K, shuffle=True, random_state=SEED).split(Xu):
        m = fit_torch("linear", Xu[tr_i], yu[tr_i], "sgd")
        lin_folds.append(mae(predict_torch(m, Xu[te_i])[:, 0], yu[te_i, 0]))
    out["torch_linear_random_cv"] = float(np.mean(lin_folds))

    # --- several steps ahead: one model, fourteen outputs
    Xa, ya, da = make_arrays(mul, WINDOW, horizon=14)
    model, best, _ = select_and_fit("gru", Xa, ya, da, CUT_SEL, CUT_END)
    te = da >= CUT_END
    P = predict_torch(model, Xa[te])
    # The naive reference at horizon h: the last value of the same day of the
    # week that is still inside the window. Window position 55 is day t, so day
    # t+h-7 sits at 48+h for h <= 7, and t+h-14 at 41+h for h <= 14.
    naive_pos = [48 + h if h <= 7 else 41 + h for h in range(1, 15)]
    out["ahead"] = {
        "recipe": best,
        "by_horizon": [mae(P[:, h], ya[te, h]) for h in range(14)],
        "naive_by_horizon": [mae(Xa[te, naive_pos[h], 0], ya[te, h])
                             for h in range(14)],
    }
    print(f"      14-step GRU: t+1 {out['ahead']['by_horizon'][0]:,.0f}  "
          f"t+14 {out['ahead']['by_horizon'][13]:,.0f}")

    # a convolutional alternative on the same 14-step task
    Xc, yc, dc = make_arrays(mul, 112, horizon=14)
    model, best, _ = select_and_fit("convgru", Xc, yc, dc, CUT_SEL, CUT_END)
    te = dc >= CUT_END
    P = predict_torch(model, Xc[te])
    out["conv"] = {"recipe": best, "window": 112,
                   "by_horizon": [mae(P[:, h], yc[te, h]) for h in range(14)]}
    print(f"      conv+GRU:    t+1 {out['conv']['by_horizon'][0]:,.0f}  "
          f"t+14 {out['conv']['by_horizon'][13]:,.0f}")

    # --- device timing, measured rather than assumed
    timing = {}
    for device in ("cpu", "mps"):
        if device == "mps" and not torch.backends.mps.is_available():
            continue
        t0 = time.time()
        fit_torch("rnn", Xu[:700], yu[:700], "adam", device=device)
        timing[device] = round(time.time() - t0, 1)
    out["device_seconds"] = timing
    print(f"      device timing {timing}")
    return out


def fig_ladder(tm, naive):
    # `naive` must be copy-last-week on the SAME rows the ladder scores -- the
    # 151 days from CUT_END -- not the pool-wide figure. Every bar in this
    # chart is a forecast on those days, and a baseline bar measured over all
    # 1,191 would be the one thing here that is not comparable with the rest.
    rows = [("Copy last week", naive, AXIS)]
    rows += [(r["label"], r["test_mae"],
              SUCCESS if r["test_mae"] < naive else ACCENT)
             for r in tm["ladder"]]
    fig, ax = plt.subplots(figsize=(10.6, 4.2))
    names = [r[0] for r in rows]
    vals = [r[1] for r in rows]
    bars = ax.barh(names, vals, color=[r[2] for r in rows], height=0.62)
    for r, v in zip(bars, vals):
        ax.text(v + 900, r.get_y() + r.get_height() / 2, f"{v:,.0f}",
                va="center", fontsize=SMALL, fontweight="bold", color="#16212b")
    ax.xaxis.set_major_formatter(riders)
    ax.set_xlabel("MAE forecasting the five months after the cut")
    ax.set_title("Every score here is a forecast, not an interpolation")
    ax.set_xlim(0, max(vals) * 1.28)
    ax.grid(axis="y", alpha=0)
    ax.invert_yaxis()
    fig.tight_layout()
    return save(fig, "l20-ladder")


def fig_horizon(tm, base):
    h = np.arange(1, 15)
    seq = np.array(tm["ahead"]["by_horizon"])
    conv = np.array(tm["conv"]["by_horizon"])
    fig, ax = plt.subplots(figsize=(10.6, 3.6))
    ax.plot(h, seq, color=PRIMARY, lw=2.4, marker="o", ms=5,
            label="GRU, 56-day window")
    ax.plot(h, conv, color=MATH, lw=2.4, marker="s", ms=5,
            label="1D convolution then GRU, 112-day window")
    ax.plot(h, tm["ahead"]["naive_by_horizon"], color=AXIS, lw=2, ls="--",
            label="copy the same day of the week")
    ax.set_xlabel("days ahead")
    ax.set_ylabel("MAE")
    ax.yaxis.set_major_formatter(riders)
    ax.set_title("One model, fourteen outputs — and the error grows with distance")
    ax.legend(loc="lower right", fontsize=SMALL)
    ax.set_xticks(h)
    fig.tight_layout()
    return save(fig, "l20-horizon")


def fig_protocols(lin):
    """The headline of Lecture 20: four protocols, and what each reports.

    The naive forecast needs no fitting, so its MAE on the same rows measures
    only how hard those rows are. Drawing it beside every protocol is what
    stops the audience reading a harder test period as a bigger leak.
    """
    rows = [
        ("Random 5-fold CV\nwhat we committed", lin["random_cv"],
         lin["random_cv_naive"], ACCENT),
        ("Rolling-origin backtest\nfixed training size", lin["rolling"],
         lin["rolling_naive"], SUCCESS),
        ("Train to end of 2018,\nforecast the next five months", lin["honest_split"],
         lin["honest_naive"], SUCCESS),
    ]
    fig, ax = plt.subplots(figsize=(10.6, 4.3))
    ypos = np.arange(len(rows))[::-1]
    h = 0.32
    for yp, (name, model, naive, colour) in zip(ypos, rows):
        ax.barh(yp + h / 2 + 0.03, naive, h, color="#cbd8e2")
        ax.barh(yp - h / 2 - 0.03, model, h, color=colour)
        ax.text(naive + 1_500, yp + h / 2 + 0.03, f"{naive:,.0f}", va="center",
                fontsize=SMALL, color=MUTED)
        ax.text(model + 1_500, yp - h / 2 - 0.03, f"{model:,.0f}", va="center",
                fontsize=SMALL, fontweight="bold", color=colour)
        ax.text(78_000, yp, f"{100 * (naive - model) / naive:.1f}% better\nthan copying",
                va="center", fontsize=SMALL, fontweight="bold", color=colour)
    ax.set_yticks(ypos, [r[0] for r in rows], fontsize=SMALL)
    ax.xaxis.set_major_formatter(riders)
    ax.set_xlabel("MAE, riders per day")
    ax.set_title("Grey: copy last week, on the same rows.  Colour: the linear model")
    ax.set_xlim(0, 95_000)
    ax.grid(axis="y", alpha=0)
    fig.tight_layout()
    return save(fig, "l20-protocols")


def margins(facts) -> dict:
    """How much of each model's advantage over copying last week survives.

    The margin is the right thing to compare across protocols, because it is
    measured against a forecast that needs no fitting and is therefore scored on
    exactly the same rows with exactly the same difficulty.
    """
    lin, tm = facts["linear"], facts["torch"]
    rnn_honest = [r for r in tm["ladder"] if r["label"] == "Simple RNN, 32 units"][0]
    out = {}
    for name, random_mae, random_naive, honest_mae, honest_naive in [
            ("linear", lin["random_cv"], lin["pool_naive"],
             lin["honest_split"], lin["honest_naive"]),
            ("rnn", tm["rnn_random_cv"], tm["rnn_random_cv_naive"],
             rnn_honest["test_mae"], lin["honest_naive"])]:
        claimed = 100 * (random_naive - random_mae) / random_naive
        real = 100 * (honest_naive - honest_mae) / honest_naive
        out[name] = {
            "random_mae": float(random_mae), "honest_mae": float(honest_mae),
            "gap": float(honest_mae - random_mae),
            "gap_pct": float(100 * (honest_mae - random_mae) / random_mae),
            "claimed_margin_pct": float(claimed),
            "real_margin_pct": float(real),
            "margin_lost_pct": float(100 * (claimed - real) / claimed),
            "random_skill": float(random_mae / random_naive),
            "honest_skill": float(honest_mae / honest_naive),
        }
        out[name]["skill_overstated_pct"] = float(
            100 * (out[name]["honest_skill"] - out[name]["random_skill"])
            / out[name]["random_skill"])
    return out


def fig_margin(mg):
    """The punchline: how much of the margin over the naive forecast survives."""
    claimed = mg["rnn"]["claimed_margin_pct"]
    real = mg["rnn"]["real_margin_pct"]
    fig, ax = plt.subplots(figsize=(10.6, 3.2))
    bars = ax.barh(["Reported under a\nrandom split", "Measured by\nforecasting forward"],
                   [claimed, real], color=[ACCENT, SUCCESS], height=0.55)
    for r, v in zip(bars, [claimed, real]):
        ax.text(v + 0.4, r.get_y() + r.get_height() / 2, f"{v:.1f}%", va="center",
                fontsize=SMALL, fontweight="bold", color="#16212b")
    ax.set_xlabel("improvement over copying last week, %")
    ax.set_title("How much of the recurrent model's advantage survives")
    ax.set_xlim(0, claimed * 1.35)
    ax.grid(axis="y", alpha=0)
    ax.invert_yaxis()
    ax.vlines([claimed, real], -0.45, 1.45, color=RULE, lw=1.5, ls="--")
    ax.annotate("", xy=(real, 0.5), xytext=(claimed, 0.5),
                arrowprops=dict(arrowstyle="<->", color=ACCENT, lw=2))
    ax.text((claimed + real) / 2, 0.5,
            f"{mg['rnn']['margin_lost_pct']:.0f}% of the margin was the split",
            ha="center", va="center", fontsize=SMALL, fontweight="bold",
            color=ACCENT,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                      edgecolor=RULE))
    fig.tight_layout()
    return save(fig, "l20-margin")


def fig_folds(lin):
    """Every fold of both protocols, so the gap can be read against the spread."""
    rnd = np.array(lin["random_cv_folds"])
    roll = np.array(lin["rolling_folds"])
    fig, ax = plt.subplots(figsize=(10.6, 3.6))
    rng = np.random.default_rng(SEED)
    for i, (vals, colour, label) in enumerate(
            [(rnd, ACCENT, "random 5-fold CV"),
             (roll, SUCCESS, "rolling-origin backtest")]):
        ax.scatter(vals, rng.normal(i, 0.045, len(vals)), s=64, color=colour,
                   alpha=0.8, zorder=3, linewidths=0)
        ax.vlines(vals.mean(), i - 0.20, i + 0.20, color=colour, lw=2.8, zorder=4)
        ax.text(vals.mean(), i + 0.30, f"mean {vals.mean():,.0f}", ha="center",
                fontsize=SMALL, fontweight="bold", color=colour)
    ax.set_yticks([0, 1], ["random\n5-fold CV", "rolling-origin\nbacktest"],
                  fontsize=SMALL)
    ax.set_ylim(-0.5, 1.6)
    ax.xaxis.set_major_formatter(riders)
    ax.set_xlabel("MAE per fold, riders per day")
    ax.set_title("Five folds each. Read the gap against the scatter, not instead of it")
    ax.grid(axis="y", alpha=0)
    fig.tight_layout()
    return save(fig, "l20-folds")


def fig_committed(facts):
    """Lecture 19's closing figure: the three models under the split we used."""
    base, tm, lin = facts["baselines"], facts["torch"], facts["linear"]
    rows = [("Copy last week", base["naive_7"], AXIS),
            ("Linear model, 56 lags", lin["random_cv"], PRIMARY),
            ("Simple RNN, 32 units", tm["rnn_random_cv"], PRIMARY)]
    fig, ax = plt.subplots(figsize=(10.6, 3.2))
    bars = ax.barh([r[0] for r in rows], [r[1] for r in rows],
                   color=[r[2] for r in rows], height=0.58)
    for r, (_, v, _) in zip(bars, rows):
        ax.text(v + 900, r.get_y() + r.get_height() / 2, f"{v:,.0f}",
                va="center", fontsize=SMALL, fontweight="bold", color="#16212b")
    ax.axvline(base["naive_7"], color=AXIS, lw=2, ls="--")
    ax.xaxis.set_major_formatter(riders)
    ax.set_xlabel("MAE, five-fold cross-validation")
    ax.set_title("Where we finish today")
    ax.set_xlim(0, base["naive_7"] * 1.25)
    ax.grid(axis="y", alpha=0)
    ax.invert_yaxis()
    fig.tight_layout()
    return save(fig, "l19-committed")


def main():
    setup()
    load_cache()
    CACHE.mkdir(parents=True, exist_ok=True)
    print("Loading Chicago transit ridership…")
    df = load_ridership()
    print(f"  {len(df):,} days, {df.index.min().date()} to {df.index.max().date()}")

    raw = pd.read_csv(CACHE / "ridership" /
                      "CTA_-_Ridership_-_Daily_Boarding_Totals.csv")
    facts: dict = {
        "n_rows_csv": int(len(raw)),
        "n_duplicate_rows": int(len(raw) - len(df)),
        "n_days_total": int(len(df)),
        "first_day": str(df.index.min().date()),
        "last_day": str(df.index.max().date()),
        "bus_day_one": int(df["bus"].iloc[0]),
        "rail_day_one": int(df["rail"].iloc[0]),
        "day_type_counts": {str(k): int(v) for k, v in
                            df["day_type"].value_counts().items()},
    }

    print("Lecture 19 figures:")
    fig_full_series(df)
    fig_weekly(df)
    fig_yearly(df)
    fig_naive(df)
    facts["baselines"] = baselines(df)
    fig_baselines(facts["baselines"])
    facts["weights"] = linear_weights(df)
    fig_weights(facts["weights"])
    print(f"    largest weights by lag: {facts['weights']['top']}")

    print("Lecture 20 — the thread:")
    fig_stationarity(df)
    fig_differencing(df)
    facts["acf"] = cached("acf", lambda: acf_values(df))
    fig_acf(facts["acf"])
    facts["identity"] = naive_identity(df, facts["acf"])
    print(f"    lag-7 difference: predicted sd "
          f"{facts['identity']['predicted_sd_diff7']:,.0f}, measured "
          f"{facts['identity']['measured_sd_diff7']:,.0f}")
    facts["regime"] = cached("regime", lambda: regime_check(df))

    print("Lecture 20 — the linear model under every protocol:")
    facts["linear"] = cached("linear_scores", lambda: linear_scores(df))
    for k in ("random_cv", "timeseries_cv", "honest_split", "rolling",
              "matched_random"):
        print(f"    {k:16s} {facts['linear'][k]:10,.0f}")

    print("Lecture 19/20 — the recurrent models (slow; cached):")
    facts["torch"] = cached("torch_models", lambda: torch_models(df))
    facts["margins"] = margins(facts)
    for k, v in facts["margins"].items():
        print(f"    {k:8s} {v['random_mae']:9,.0f} -> {v['honest_mae']:9,.0f}"
              f"   margin {v['claimed_margin_pct']:.1f}% -> {v['real_margin_pct']:.1f}%"
              f"   skill overstated {v['skill_overstated_pct']:.1f}%")

    print("Lecture 20 figures:")
    fig_protocols(facts["linear"])
    fig_margin(facts["margins"])
    fig_folds(facts["linear"])
    fig_ladder(facts["torch"], facts["linear"]["ladder_naive"])
    fig_horizon(facts["torch"], facts["baselines"])
    fig_committed(facts)

    export(**{"app10": facts})

    problems = check_text_floor()
    if problems:
        print("\ntext floor:")
        for p in problems:
            print("  " + p)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
