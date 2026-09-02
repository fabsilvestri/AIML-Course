#!/usr/bin/env python3
"""
Lecture 15 — Time series. Chicago transit ridership, Géron Chapter 13.

Derivation: stationarity, differencing and autocorrelation.

Old lecture 19's data and baselines with old lecture 20's derivation and
backtesting argument, because the new plan puts the mathematics and the honest
protocol in the same lecture as the series they are about. The recurrent
architectures move to Lecture 16.

Runs on CPU throughout; the series is small.
"""

from __future__ import annotations

import nbformat as nbf

from _prompt import prompt


def md(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(text.strip("\n"))


def code(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(text.strip("\n"))


HEADER = """
# Time series

**Lecture 15** · Géron, Chapter 15

Applications of Machine Learning — BSc Mathematics of Artificial Intelligence

---

**How to use this notebook.** Read before you run. Every code cell is preceded
by the specification that would produce it — input, output, constraint, check.

Cells marked **⚠** deliberately run code that is wrong, and say so in the
heading before you reach them. They are the failures this lecture is about;
each runs the broken version beside the correct one and prices the difference.

Runs on CPU. Nothing here needs an accelerator.

**A CPU runtime is enough.** The largest model here has a few thousand
parameters. If Colab offers you a GPU, decline it — for sequences this short the
transfers cost more than the arithmetic saves.
"""

SETUP = '''
# Not examinable, and only needed on some machines: PyTorch, numpy and
# torchvision can each end up loading their own OpenMP runtime, and with more
# than one loaded a training cell can deadlock -- no error, no output, and no
# CPU use. These have to be set BEFORE torch is imported, because they are read
# at import time and after that they do nothing.
import os
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# --- setup -------------------------------------------------------------------
# Not examinable: this is engineering hygiene, not machine learning.
import sys
from pathlib import Path
import tarfile, urllib.request

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

print(f"python  {sys.version.split()[0]}")
print(f"pandas  {pd.__version__}")
print(f"torch   {torch.__version__}")

RANDOM_STATE = 42                  # one seed, used everywhere
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)

# A time series is plotted more often than anything else in this lecture, so it
# is worth setting the defaults once.
plt.rcParams.update({"figure.figsize": (11, 3.2), "axes.grid": True,
                     "grid.alpha": 0.3, "font.size": 11})
'''

LOADER = '''
# --- the data ----------------------------------------------------------------
def load_ridership():
    """Daily boarding totals for the Chicago Transit Authority."""
    tarball = Path("datasets/ridership.tgz")
    if not tarball.is_file():
        Path("datasets").mkdir(parents=True, exist_ok=True)
        url = "https://github.com/ageron/data/raw/main/ridership.tgz"
        urllib.request.urlretrieve(url, tarball)
        with tarfile.open(tarball) as t:
            t.extractall(path="datasets", filter="data")
    return pd.read_csv(
        "datasets/ridership/CTA_-_Ridership_-_Daily_Boarding_Totals.csv",
        parse_dates=["service_date"])

raw = load_ridership()
print(f"{len(raw):,} rows as published")
'''


def build() -> list:
    cells: list = [
        md(HEADER), md("## 1 · Setup"),
        prompt(output="library versions printed, one seed set everywhere, plot "
                      "defaults fixed once",
               constraint="not examinable — engineering hygiene, kept out of "
                          "the way of the argument"),
        code(SETUP)]

    cells += [
        md("""
## 2 · The data

The Chicago Transit Authority publishes one row per day: how many people
boarded a bus, how many boarded a train, and what kind of day it was.
"""),
        prompt(input="the CTA daily-boardings CSV, fetched once and cached",
               output="one row a day: date, day type, bus and rail counts",
               constraint="download only if the file is not already on disk, so "
                          "the notebook is re-runnable offline",
               check="print the row count and compare it with what the "
                     "publisher states"),
        code(LOADER),
        md("""
### Tidying, and one line that matters

Four operations. Three are housekeeping and one is a decision:

* rename the columns, because `service_date` is long and we will type it often
* sort by date and index by it — a series with an unsorted index is a trap
* drop `total`, which is exactly `bus + rail` and so carries nothing
* **drop duplicate rows.** The published file repeats some days.

Count the duplicates rather than assuming there are none.
"""),
        prompt(
               input="the raw frame, one row a day, columns as published",
               output="a frame indexed by date with bus and rail only",
               constraint="drop `total` — it is exactly bus + rail — and drop duplicate rows, reporting how many",
               check="print the count removed and the date range that survives"),
        code('''
df = raw.copy()
df.columns = ["date", "day_type", "bus", "rail", "total"]
df = df.sort_values("date").set_index("date")
df = df.drop("total", axis=1)          # it is just bus + rail

before = len(df)
df = df.drop_duplicates()
print(f"{before - len(df)} duplicate rows removed, {len(df):,} days left")
print(f"from {df.index.min().date()} to {df.index.max().date()}")
df.head(3)
'''),
        md("""
### `day_type`

Three values: `W` for a weekday, `A` for a Saturday, `U` for a Sunday **or a
public holiday**. That last word is why the column is worth having — the model
cannot see a calendar, and the fourth of July behaves like a Sunday whatever the
date says.
"""),
        prompt(
               input="the tidied frame",
               output="how many days of each type, and the three days around Memorial Day 2019",
               constraint="do not assume U means Sunday — show a case where it is a public holiday on a Monday",
               check="the three printed values are A, U, U"),
        code('''
print(df["day_type"].value_counts().to_string())

# Memorial Day 2019 fell on a Monday, and the data knows it
print(list(df.loc["2019-05-25":"2019-05-27"]["day_type"]))
'''),
    ]
    cells += [
        md("""
## 3 · Look at it before you model it

Twenty-one years, then ten weeks, then a single year. Three time scales, three
different facts, and none of them is visible at the other two scales.
"""),
        prompt(
               input="the rail series, all twenty-one years",
               output="one line plot, rail boardings against date",
               constraint="thin line, no markers — the shape is the point, not the individual days",
               check="the 2020 cliff is visible without being pointed at"),
        code('''
fig, ax = plt.subplots(figsize=(11, 3.2))
df["rail"].plot(ax=ax, lw=0.6, color="#0b3d62")
ax.set_ylabel("rail boardings")
ax.set_title("Twenty-one years — and one event that ends the series we know")
plt.show()
'''),
        md("""
**What the plot says.** A slow decline through the 2010s, and then in March 2020
a cliff. Everything after that cliff is a different process. We are not going to
model across it: this notebook works on **2016-01 to 2019-05**, and Lecture 16
returns to the cliff to ask what a model owes you when the world changes.
"""),
        prompt(
               input="bus and rail over ten weeks of 2019",
               output="both series on one axis, with markers",
               constraint="a window short enough that individual days are distinguishable",
               check="every week shows a weekend trough; some weeks show a third"),
        code('''
fig, ax = plt.subplots(figsize=(11, 3.2))
df.loc["2019-03":"2019-05", ["bus", "rail"]].plot(ax=ax, marker=".", ms=4)
ax.set_ylabel("boardings")
ax.set_title("Ten weeks — the shape that actually matters")
plt.show()
'''),
        md("""
**Weekly seasonality, and a second trough.** Every week has a deep weekend
trough. Some weeks have a third dip in the middle, and those are the public
holidays — which is exactly what `day_type` marks and what a bare day-of-week
feature would miss.

Whatever we build has to know what day of the week it is. The cheapest way to
tell it is to show it the last week, or the last eight.
"""),
    ]
    cells += [
        prompt(
               label="the window we will work on, fixed before any model",
               input="the tidied frame",
               output="the rail pool and the days that can be scored",
               constraint="fix the span and the window length HERE, before the derivation and before any baseline — a window chosen after seeing a result is a hyperparameter fitted on the test set",
               check="assert the first WINDOW days are excluded from the targets. They are consumed by the history every model needs, so they cannot be scored."),
        code('''
# The pool we will work on, and the days we will be scored on. The first 56 days
# are consumed by the window the models need, so they cannot be targets.
WINDOW = 56
pool = df["rail"]["2016-01":"2019-05"]
target = pool[WINDOW:]

assert len(target) == len(pool) - WINDOW
print(f"pool   {len(pool):,} days, {pool.index.min().date()} to {pool.index.max().date()}")
print(f"scored {len(target):,} days, from {target.index.min().date()}")
'''),

        md("""
## 4 · The derivation — stationarity, differencing, autocorrelation

A series is **strictly stationary** if the joint distribution of
$(X_{t_1},\\dots,X_{t_k})$ is unchanged when every index is shifted by the same
$h$. That is far more than anyone can check, so in practice we ask for **weak
stationarity**: constant mean, constant variance, and an autocovariance
$\\gamma(h) = \\operatorname{Cov}(X_t, X_{t+h})$ that depends on the lag $h$ and
not on $t$.

**Why a model needs it.** Fitting one set of weights to all of history assumes
that what a value meant in 2016 is what it means in 2019. If the mean drifts,
the model is averaging two different worlds and is right about neither.

Ridership is not stationary: there is a downward trend and a hard weekly cycle.
"""),
prompt(
       input="the rail pool, and the same series differenced at lag 1 and lag 7",
       output="an augmented Dickey-Fuller p-value for each, with a verdict",
       constraint="install statsmodels only if it is missing, so the notebook runs on a bare environment",
       check="the level series should fail to look stationary where the differenced ones do not. Decide before running which of the three you expect to fail. The raw  series should not look stationary; the seasonal difference should. If all  three pass, you have read the sign backwards — which is the single most  common ADF error and it never raises anything."),
        code('''
try:
    from statsmodels.tsa.stattools import adfuller
except ImportError:                         # Colab has it; a bare venv may not
    !pip -q install statsmodels
    from statsmodels.tsa.stattools import adfuller

def adf(series, label):
    stat, p, *_ = adfuller(series.dropna())
    verdict = "stationary" if p < 0.05 else "NOT stationary"
    print(f"{label:28s} ADF p = {p:7.4f}   {verdict}")

adf(pool,             "rail, as it is")
adf(pool.diff(),      "first difference")
adf(pool.diff(7),     "seasonal difference (7)")
'''),
        md("""
### Differencing

$\\nabla X_t = X_t - X_{t-1}$ removes a linear trend; applying it twice removes a
quadratic one. A series that is stationary after $d$ differences is *integrated
of order $d$*. And $\\nabla_7 X_t = X_t - X_{t-7}$ removes a weekly cycle, which
is what this series most needs.

**Differencing is not free.** For a stationary series,

$$\\operatorname{Var}(X_t - X_{t-h}) = 2\\gamma(0)\\,(1 - \\rho(h))$$

so differencing at a lag where the autocorrelation $\\rho(h)$ is *below* $1/2$
makes the variance **larger**, not smaller. Check it against the data rather than
believing it.
"""),
prompt(
       input="the pool, and its autocorrelation at lags 1, 7 and 14",
       output="predicted and measured standard deviation of each difference",
       constraint="predict from the identity Var(X_t - X_t-h) = 2 gamma(0)(1 - rho(h)) BEFORE measuring, so the theory is exposed to the data",
       check="flag any lag where differencing makes the spread larger, which is the point of the cell. Print predicted/measured as a ratio and expect 1.00 to within a per cent  at every lag. If lag 7 agrees and lag 1 does not, suspect your estimate of  rho, not the identity."),
        code('''
from statsmodels.tsa.stattools import acf

# Two standard estimators of the same rho disagree here, and which one you used
# has to be stated. statsmodels' acf divides by n at every lag (the "biased"
# estimator, which keeps the autocovariance sequence positive semi-definite);
# pandas' Series.autocorr computes a Pearson correlation on the overlapping
# pairs, dividing by n - h. At lag 7 they differ in the third decimal, which is
# enough to move the predicted spread by about two thousand boardings. We use
# statsmodels, and the slides quote the same.
rho_all = acf(pool.values, nlags=60, fft=False)

sd = pool.std()
print(f"{'series':28s} sd {sd:>10,.0f}")
for h in (1, 7, 14):
    rho = rho_all[h]
    predicted = np.sqrt(2 * sd**2 * (1 - rho))
    measured = pool.diff(h).std()
    flag = "  <- WORSE than not differencing" if measured > sd else ""
    print(f"lag {h:>2d}: rho {rho:+.3f}   predicted sd {predicted:>10,.0f}"
          f"   measured {measured:>10,.0f}{flag}")

print(f"\\npandas Series.autocorr at lag 7 would give "
      f"{pool.autocorr(lag=7):+.3f} instead of {rho_all[7]:+.3f}")

# A standard deviation is not a mean absolute error, and converting one into
# the other assumes a shape. For a Gaussian, E|X| = sqrt(2/pi) * sigma.
d7 = pool.diff(7).dropna()
gaussian_mae = float(np.sqrt(2 / np.pi) * d7.std())
print(f"\\nif the 7-day difference were Gaussian, its MAE would be "
      f"{gaussian_mae:>10,.0f}")
print(f"the measured MAE is                                 "
      f"{d7.abs().mean():>10,.0f}")
print(f"excess kurtosis of the difference: {d7.kurtosis():.1f} — most days are")
print("far calmer than a Gaussian of that spread, and a few are far worse.")
'''),
        md("""
**Read the first row.** At lag 1 the autocorrelation is well under a half, so
first-differencing this series *increases* its spread. The textbook reflex —
"it is not stationary, difference it" — makes the problem harder here. At lag 7
the autocorrelation is high and the difference is genuinely smaller.

That is the whole explanation of why "copy last week" was so hard to beat in the
previous lecture: $\\nabla_7$ is close to white noise, and a naive forecast is
exactly the model that assumes it *is*.
"""),
prompt(
       input="five months of the rail series, raw and differenced at lags 1 and 7",
       output="three stacked panels sharing an x axis",
       constraint="a zero line on each, so 'bigger swings' is visible rather than asserted",
       check="the first difference should look wilder than the series it came from. The zero line should be visibly crossed on the differenced panels and  nowhere near the raw one. If all three look alike, the y-limits are shared  and the figure is showing you nothing."),
        code('''
fig, axes = plt.subplots(3, 1, figsize=(11, 6), sharex=True)
recent = pool["2019-01":"2019-05"]
for ax, (s, title) in zip(axes, [
        (recent,          "the series"),
        (recent.diff(),   "first difference — bigger swings, not smaller"),
        (recent.diff(7),  "seasonal difference at lag 7 — nearly noise")]):
    ax.plot(s.index, s.values, lw=1, color="#0b3d62")
    ax.axhline(0, color="#c0392b", lw=1)
    ax.set_title(title, fontsize=11, loc="left")
plt.tight_layout(); plt.show()
'''),
        md("""
### The autocorrelation function

One picture that contains everything above: correlation against lag. Spikes at
7, 14, 21 and a slow decay elsewhere.
"""),
prompt(
       input="autocorrelation of the pool and of its seasonal difference, lags 0 to 42",
       output="both on one bar chart",
       constraint="plot them side by side at each lag, not on two charts, so the collapse at lag 7 is directly comparable",
       check="spikes at 7, 14 and 21 in the raw series; nothing much left after differencing. Read lag 7 and lag 14 specifically and say the two numbers out loud. Those  two bars are the claim. The other forty-one are scenery, and scanning all  forty-three is how you talk yourself into a pattern."),
        code('''
lags = np.arange(0, 43)
acf_level = [pool.autocorr(lag=int(k)) if k else 1.0 for k in lags]
acf_diff7 = [pool.diff(7).autocorr(lag=int(k)) if k else 1.0 for k in lags]

fig, ax = plt.subplots(figsize=(11, 3.2))
ax.bar(lags - 0.2, acf_level, width=0.4, label="the series", color="#0b3d62")
ax.bar(lags + 0.2, acf_diff7, width=0.4, label="after seasonal differencing",
       color="#14663a")
ax.axhline(0, color="#33414d", lw=1)
ax.set_xlabel("lag, in days"); ax.set_ylabel("autocorrelation")
ax.legend(); ax.set_title("Where the structure is")
plt.show()
'''),
    ]
    cells += [
        md("""
## 5 · Choose a metric, and the baseline to beat

Before a single model. Three candidates:

| Metric | What it says | Why not |
|---|---|---|
| **MAE** | average error in boardings | — |
| RMSE | punishes large errors more | one strike day dominates the score |
| MAPE | average error as a percentage | division by a small denominator |

**The MAPE trap.** MAPE divides by the truth. Christmas Day has about a tenth of
a Tuesday's ridership, so a 20,000-boarding error on Christmas counts for as much
as a 200,000-boarding error on a Tuesday. The metric would spend the model's
capacity on the days nobody is planning staffing for.

We use **MAE, in boardings**. It is in the units the operations team already
thinks in, and one unit of it is one person.
"""),
        prompt(
               input="two aligned series, either of which may have gaps",
               output="mean absolute error, in boardings",
               constraint="ignore days where either side is missing rather than filling them",
               check="mae(x, x) is 0, and a series against a shifted copy of itself is not"),
        code('''
def mae(truth, forecast):
    """Mean absolute error, ignoring days where either side is missing."""
    truth, forecast = pd.Series(truth), pd.Series(forecast)
    mask = truth.notna() & forecast.notna()
    return float((truth[mask] - forecast[mask]).abs().mean())

print(f"mae against itself      {mae(pool, pool):.1f}")
print(f"mae against a shift     {mae(pool, pool.shift(7)):,.0f} boardings")
'''),
        prompt(
            label="the same baseline on the window the book plots",
            input="March to May 2019, rail and bus",
            output="MAE and MAPE for copy-last-week on each",
            constraint="quote the window in the same sentence as the number — the same forecast scores very differently over three spring months than over three years",
            check="also report the single worst day. One holiday costs more than eight ordinary days put together, and a mean absolute error hides that entirely."),
        code('''
for name, col in (("rail", "rail"), ("bus", "bus")):
    part  = df[col]["2019-03":"2019-05"]
    naive = df[col].shift(7)["2019-03":"2019-05"]
    err   = (part - naive)
    mae_w  = float(err.abs().mean())
    mape_w = float((err.abs() / part).mean() * 100)
    print(f"{name:5s} 2019-03 to 2019-05:  MAE {mae_w:>9,.0f}   MAPE {mape_w:>5.1f}%")

part  = df["rail"]["2019-03":"2019-05"]
err   = part - df["rail"].shift(7)["2019-03":"2019-05"]
worst = err.abs().idxmax()
print(f"\\nworst single day: {worst.date()}, error {abs(err.loc[worst]):,.0f}")
print(f"that one day is {abs(err.loc[worst]) / err.abs().mean():.0f} times the "
      f"mean absolute error over the window")
'''),

        md("""
### The baselines

A number is only good or bad next to another number. Three forecasts that
require no fitting at all:
"""),
        prompt(
               input="the 2016-01 to 2019-05 rail pool",
               output="MAE for four forecasts that require no fitting",
               constraint="every forecast scored on exactly the same days, so they are comparable",
               check="copying last week beats copying yesterday — if not, the alignment is wrong"),
        code('''
baselines = {
    "a constant (the mean)":  pd.Series(pool[:"2018-12"].mean(), index=target.index),
    "copy the day before":    pool.shift(1)[WINDOW:],
    "copy last week":         pool.shift(7)[WINDOW:],
    "copy two weeks back":    pool.shift(14)[WINDOW:],
}
for name, forecast in baselines.items():
    print(f"{name:24s} MAE {mae(target, forecast):>10,.0f}")
'''),
        md("""
**Read that table before going on.** Copying the same weekday one week ago is
about **2.4 times better** than copying yesterday, and roughly three times better
than the mean. It costs nothing, it has no parameters, it cannot overfit, and it
already knows about weekends and — by accident — about most public holidays.

That is the number to beat. Not the mean. Not zero.

### The number to beat

Everything below is measured against it, including the parts that disappoint.
"""),
        prompt(
               input="the naive-forecast MAE",
               output="the target MAE, stated before any model is fitted",
               constraint="state the target BEFORE fitting, so it cannot be chosen afterwards to fit the result",
               check="the target is printed, so it cannot be quietly revised"),
        code('''
NAIVE_MAE = mae(target, pool.shift(7)[WINDOW:])

# ---------------------------------------------------------------------------
# MY COMMITMENT
# ---------------------------------------------------------------------------
# Beating the naive forecast by less than this is not worth deploying:
TARGET_IMPROVEMENT = 0.10          # ← change it if you disagree, but change it NOW

TARGET_MAE = NAIVE_MAE * (1 - TARGET_IMPROVEMENT)
print(f"naive  MAE {NAIVE_MAE:>10,.0f}")
print(f"target MAE {TARGET_MAE:>10,.0f}   ({TARGET_IMPROVEMENT:.0%} better)")
'''),
    ]
    cells += [
        md("""
## 6 · Turning a series into a table

Supervised learning wants rows. A series is not rows, so we cut it into
overlapping windows: 56 days in, the 57th day out.

Why 56? Eight whole weeks. Any multiple of seven lets a model line up the same
weekday; eight of them give it enough repetitions to average over. And why divide
by a million — because the numbers are around 600,000 and a network initialised
in the usual way expects inputs near 1.
"""),
        prompt(
               input="a series as a tensor, and a window length",
               output="a Dataset yielding (window, next step) pairs",
               constraint="the target must be the step AFTER the window and never inside it — an off-by-one leaks one day and nothing later complains",
               check="length is len(series) - window_length"),
        code('''
class TimeSeriesDataset(torch.utils.data.Dataset):
    """Every window of `window_length` steps, and the step that follows it."""

    def __init__(self, series, window_length):
        self.series, self.window_length = series, window_length

    def __len__(self):
        return len(self.series) - self.window_length

    def __getitem__(self, idx):
        end = idx + self.window_length     # first index after the window
        return self.series[idx:end], self.series[end]
'''),
        md("""
### Test the class before trusting it

Six numbers, a window of three. Read the output and check by eye that the target
is the step *after* the window and never inside it. Off-by-one here would be a
leak of exactly one day, and no later cell would complain.
"""),
        prompt(
               input="six integers and a window of three",
               output="every window printed with the value that follows it",
               constraint="test the class on data small enough to check by eye before trusting it",
               check="read the output and confirm the target is never inside the window"),
        code('''
toy = torch.tensor([[0], [1], [2], [3], [4], [5]])
for window, t in TimeSeriesDataset(toy, window_length=3):
    print(window.flatten().tolist(), "->", t.item())
'''),
        prompt(
               input="the rail pool, scaled by a million",
               output="X of shape (rows, 56, 1) and y of shape (rows, 1)",
               constraint="assert the shapes rather than printing them, so a wrong shape stops the notebook",
               check="rows equals len(pool) - 56"),
        code('''
series = torch.tensor(pool.values / 1e6, dtype=torch.float32).unsqueeze(1)
dataset = TimeSeriesDataset(series, WINDOW)

X = torch.stack([w for w, _ in dataset])       # [rows, time, series]
y = torch.stack([t for _, t in dataset])

assert X.shape == (len(pool) - WINDOW, WINDOW, 1), X.shape
assert y.shape == (len(pool) - WINDOW, 1), y.shape
print(f"X {tuple(X.shape)}   y {tuple(y.shape)}")
print("rows =", len(pool), "days -", WINDOW, "consumed by the first window")
'''),
    ]
    cells += [
        md("""
## 7 · What the linear model learned

56 lags in, one number out — 57 parameters. Plot the coefficient on each lag
against the lag, and the model tells you what it found, in a language you can
check against the data.
"""),
        prompt(
               input="the fitted linear model's 56 coefficients",
               output="coefficient against lag, with whole weeks marked",
               constraint="lag 1 on the right, so the axis reads as time running backwards from today",
               check="print the mean weight on weekly lags against all others rather than eyeballing"),
        code('''
from sklearn.linear_model import LinearRegression

# Fitted here rather than inherited from a cell above, so this section stands on
# its own: the same windows, the same time split, nothing shuffled.
values = pool.values / 1e6
Xw = np.stack([values[i:i + WINDOW] for i in range(len(values) - WINDOW)])
yw = values[WINDOW:]
cut = int(len(Xw) * 0.8)
lin = LinearRegression().fit(Xw[:cut], yw[:cut])

lags = np.arange(WINDOW, 0, -1)             # lag 56 first, lag 1 last
fig, ax = plt.subplots(figsize=(11, 3.2))
ax.bar(lags, lin.coef_.ravel(), color="#0b3d62", width=0.8)
for k in (7, 14, 21, 28, 35, 42, 49, 56):
    ax.axvline(k, color="#c0392b", lw=1, ls="--", alpha=0.6)
ax.invert_xaxis()
ax.set_xlabel("lag, in days")
ax.set_ylabel("coefficient")
ax.set_title("Dashed lines mark whole weeks — nobody told it about weeks")
plt.show()

week = lin.coef_.ravel()[[WINDOW - k for k in (7, 14, 21, 28)]]
print(f"mean coefficient on the weekly lags   {week.mean():+.4f}")
print(f"mean coefficient on all other lags    "
      f"{np.delete(lin.coef_.ravel(), [WINDOW - k for k in (7,14,21,28)]).mean():+.4f}")
'''),
        md("""
The weight lands on lags 7, 14, 21 and 28. We never mentioned weeks; the model
found them because they are in the data. This is the pleasant version of a linear
model being interpretable — the coefficients say something you can go and check.
"""),
    ]
    cells += [
        md("""
## 8 · A random split on a series, measured

Exactly the cell from the previous lecture. Run it and look at the folds: they
agree with one another, which is what a stable measurement looks like.
"""),
prompt(
       input="the pool as 56-lag windows",
       output="a cross-validated MAE from a shuffled five-fold split",
       constraint="this is the previous lecture's broken cell, reproduced exactly",
       check="look at the fold spread — they agree with each other, which is what a stable measurement of the wrong quantity looks like. It must reproduce Lecture 19's figure to the boarding. If it does not, one  of the two notebooks has drifted and every comparison below is void — go  back and fix that before reading on."),
        code('''
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, TimeSeriesSplit, cross_val_score

values = pool.values / 1e6
X = np.stack([values[i:i + WINDOW] for i in range(len(values) - WINDOW)])
y = values[WINDOW:]
print(f"X {X.shape}   y {y.shape}")

model = LinearRegression()
cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
folds_random = -cross_val_score(model, X, y, cv=cv,
                                scoring="neg_mean_absolute_error") * 1e6
print(f"random 5-fold   MAE {folds_random.mean():>10,.0f}   "
      f"folds {np.round(folds_random).astype(int)}")

# One shuffled split is one draw. The slides quote the mean over twenty seeds,
# because a single seed of this splitter moves by more than the effect we are
# about to attribute to the splitter itself.
seed_means = np.array([
    -cross_val_score(model, X, y, cv=KFold(5, shuffle=True, random_state=s),
                     scoring="neg_mean_absolute_error").mean() * 1e6
    for s in range(20)])
RANDOM_CV = float(seed_means.mean())
print(f"over 20 seeds   MAE {RANDOM_CV:>10,.0f}   "
      f"sd across seeds {seed_means.std():>7,.0f}")
'''),
        prompt(
            label="the split a deployed system actually faces",
            input="the same windows, fit on everything up to the end of 2018",
            output="MAE on the five months that follow, and the naive forecast on the same rows",
            constraint="score the naive baseline on the SAME test rows — otherwise a gap between protocols could be the rows being harder rather than the protocol being honest",
            check="report both, and their ratio. The ratio is a skill score, and it is the only thing comparable between two protocols that scored different days."),
        code('''
dates = pool.index[WINDOW:]
tr = dates < "2019-01-01"

fit = LinearRegression().fit(X[tr], y[tr])
HONEST_SPLIT = 1e6 * float(np.abs(fit.predict(X[~tr]) - y[~tr]).mean())
HONEST_NAIVE = 1e6 * float(np.abs(X[~tr, -7] - y[~tr]).mean())

print(f"train {tr.sum():,} days, test {(~tr).sum():,} days")
print(f"linear, forecasting forward   MAE {HONEST_SPLIT:>10,.0f}")
print(f"copy last week, same rows     MAE {HONEST_NAIVE:>10,.0f}")
print(f"skill (model / naive)             {HONEST_SPLIT / HONEST_NAIVE:>10.2f}")
print(f"\\nrandom split, same model      MAE {RANDOM_CV:>10,.0f}")
print("The naive forecast needs no fitting, so its MAE measures only how hard")
print("the rows are. That is what makes the two protocols comparable at all.")
'''),

        prompt(
            label="the matched random split",
            input="twenty random splits with the SAME training and test sizes as the rolling backtest",
            output="the mean MAE",
            constraint="match the sizes — TimeSeriesSplit's first fold trains on a fifth of the data, so a comparison against it confounds the protocol with the training-set size",
            check="this is the row that isolates the protocol. Same model, same sizes, same metric; only the ordering differs."),
        code('''
from sklearn.model_selection import ShuffleSplit

L, T = 700, 98          # the rolling backtest's training and test sizes
matched = np.array([
    1e6 * np.abs(LinearRegression().fit(X[a], y[a]).predict(X[b]) - y[b]).mean()
    for a, b in ShuffleSplit(20, train_size=L, test_size=T,
                             random_state=RANDOM_STATE).split(X)])
MATCHED_RANDOM = float(matched.mean())
print(f"matched random split, 20 draws  MAE {MATCHED_RANDOM:>10,.0f}")
print(f"(train {L}, test {T} — the same sizes the forward backtest uses)")
'''),

        md("""
### The same model, split by time

`TimeSeriesSplit` never lets a training row come after a test row. One call
changes, nothing else.
"""),
prompt(
       input="the same X and y",
       output="the same five-fold MAE, split by time instead",
       constraint="no training row may come after a test row; change one call and nothing else",
       check="report the gap against the shuffled number in boardings and per cent. Compare the largest training index with the smallest test index and assert  the gap. One line, and it fails loudly the day someone sorts the frame  upstream."),
        code('''
cv = TimeSeriesSplit(n_splits=5)          # was KFold(shuffle=True)
folds_time = -cross_val_score(model, X, y, cv=cv,
                              scoring="neg_mean_absolute_error") * 1e6
print(f"forward 5-fold  MAE {folds_time.mean():>10,.0f}   "
      f"folds {np.round(folds_time).astype(int)}")
print()
print(f"the shuffle flattered the model by "
      f"{folds_time.mean() - folds_random.mean():,.0f} boardings "
      f"({100 * (folds_time.mean() - folds_random.mean()) / folds_time.mean():.0f}%)")
'''),
        md("""
**Now look at the spread.** The forward folds disagree with each other far more
than the shuffled ones did — and that disagreement is real information, not
noise to be averaged away. The first fold trains on a couple of hundred days and
scores badly; the later folds train on years.

Two conditions have to hold for a split to be honest here, and the shuffle broke
both:

1. **No training row may come after a test row.** Otherwise the model has seen
   the future.
2. **No training row may be adjacent to a test row.** Two consecutive days are
   nearly the same number, so a neighbour in the training set is very close to
   giving away the answer.

`TimeSeriesSplit` fixes the first. The second needs a **gap**.
"""),
prompt(
       input="the same X and y again",
       output="MAE with a gap of one window between train and test",
       constraint="no training row may be ADJACENT to a test row either — two consecutive days are nearly the same number",
       check="print mean and fold spread for all three protocols together. The gapped number must be **worse** again than the plain time split. Each  time you remove a route for information to travel, the score gets worse  and more honest. A protocol that improves the score is a protocol you  should distrust."),
        code('''
# Condition 2, made explicit: leave a gap the width of one window between the
# end of training and the start of testing, so no test target can be predicted
# from a day that is effectively in the training set.
cv = TimeSeriesSplit(n_splits=5, gap=WINDOW)
folds_gap = -cross_val_score(model, X, y, cv=cv,
                             scoring="neg_mean_absolute_error") * 1e6
print(f"forward + gap   MAE {folds_gap.mean():>10,.0f}   "
      f"folds {np.round(folds_gap).astype(int)}")

for label, f in [("random 5-fold", folds_random),
                 ("forward", folds_time),
                 ("forward + purge", folds_gap)]:
    print(f"{label:18s} {f.mean():>10,.0f}   sd across folds {f.std():>9,.0f}")
'''),
        md("""
### The margin, recomputed

The number that matters is not the MAE. It is how much of the naive baseline's
score the model actually takes off — and that is what the split was inflating.
"""),
prompt(
       input="the naive baseline and the four protocols' MAEs",
       output="the margin over the baseline that each protocol reports",
       constraint="quote the margin, not the MAE — the margin is what the shuffle was inflating",
       check="state what share of the claimed margin was protocol rather than model. The margin should shrink monotonically as the protocol gets stricter. If  it does not, either a protocol is mis-implemented or the baseline is being  measured over a different set of days than the models."),
        code('''
target = pool[WINDOW:]
naive = pool.shift(7)[WINDOW:]
mask = target.notna() & naive.notna()
NAIVE_MAE = float((target[mask] - naive[mask]).abs().mean())

# Four protocols, one model, one dataset. The only thing that changes is which
# rows are allowed to train on which other rows.
cut = int(len(X) * 0.8)
holdout = 1e6 * np.abs(
    LinearRegression().fit(X[:cut], y[:cut]).predict(X[cut:]) - y[cut:]).mean()

protocols = [("random 5-fold",           folds_random.mean()),
             ("one forward hold-out",    holdout),
             ("forward 5-fold",          folds_time.mean()),
             ("forward 5-fold + purge",  folds_gap.mean())]

print(f"naive baseline {NAIVE_MAE:,.0f}")
print()
print(f"{'protocol':26s}{'MAE':>10s}{'margin':>10s}")
for name, score in protocols:
    print(f"{name:26s}{score:>10,.0f}{(NAIVE_MAE - score) / NAIVE_MAE:>9.1%}")

claimed = 100 * (NAIVE_MAE - folds_random.mean()) / NAIVE_MAE
real    = 100 * (NAIVE_MAE - folds_gap.mean())    / NAIVE_MAE
print()
print(f"{100 * (claimed - real) / claimed:.0f}% of the claimed margin "
      f"was the protocol, not the model")
'''),
    ]
    cells += [
        md("""
## 9 · Regime change

Everything above stops at May 2019. The series does not — in March 2020 the
level falls by roughly three quarters and never returns to where it was.

This is worth being precise about, because it is **not a leak and not a bug**.
The protocol was correct, the measurement was honest, and the model is still
useless afterwards. No split protects you from the world changing.
"""),
prompt(
       input="the whole series, through 2021",
       output="the mean level before and after March 2020, and a plot spanning both",
       constraint="this is not a leak and not a bug — say so plainly; the protocol was right and the model still stopped working",
       check="the ratio of the two levels is printed, not described. Print the mean level either side of March 2020. If the two differ by more  than the model's entire error budget, the question is not 'why is the  model wrong' but 'when did this model stop being about the world'."),
        code('''
level_2019 = df["rail"]["2019-01":"2019-05"].mean()
level_2020 = df["rail"]["2020-04":"2020-08"].mean()
print(f"mean daily rail boardings, early 2019   {level_2019:>10,.0f}")
print(f"mean daily rail boardings, mid 2020     {level_2020:>10,.0f}")
print(f"                                        {level_2020 / level_2019:>10.1%} "
      f"of the earlier level")

fig, ax = plt.subplots(figsize=(11, 3.2))
df["rail"]["2019-01":"2021-06"].plot(ax=ax, lw=0.8, color="#0b3d62")
ax.axvspan(pd.Timestamp("2020-03-15"), pd.Timestamp("2020-06-01"),
           color="#c0392b", alpha=0.15)
ax.set_title("A correct protocol, an honest number, and a model that stopped working")
plt.show()
'''),
        md("""
**What to do about it** is a monitoring question, not a modelling one: measure
the live error against the target, and have a rule that says when to
stop trusting the model. A model that is never re-measured after deployment is
an assumption wearing a number's clothes.
"""),
    ]
    cells += [
        md("""
## 10 · The temporal checklist

Take this to any dataset with a timestamp in it:

1. **Is there a time column?** If yes, no shuffled split — ever, including
   inside `cross_val_score`, `train_test_split` and any tuner's own CV.
2. **Is there a gap between train and test?** Adjacent rows leak.
3. **Would I know every feature at prediction time?** Say it out loud for each
   one. `shift(-1)` on a calendar is fine; on the target it is the answer.
4. **Is the baseline seasonal?** Compare against copying the same weekday, not
   against the mean.
5. **Did I difference reflexively?** Check $\\rho(h) > 1/2$ first, or the
   variance goes up.
6. **Is the score a single number when the decision needs a curve?**
7. **What would tell me the regime has changed?** Write the trigger down before
   deployment, not after.

### ★ Record your numbers

The claimed margin, the honest margin, and the difference between them. That
difference is the most useful number in this application: it is the size of the
mistake that nothing in the output complained about.

### Five questions to ask of any forecast

* Set `gap=0` in `TimeSeriesSplit`. How much of the margin comes back? That
  amount was adjacency.
* Give the model `df["day_type"]` **without** the `shift(-1)`. The score barely
  moves — explain why that is worse, not better.
* Train on 2016–2019, test on 2020. Then argue, in two sentences, whether the
  model was wrong or the question was.
"""),
    ]
    cells += [
        md("""
## 11 · Where we are

- Ordered, autocorrelated rows break the assumption every earlier lecture
  rested on: a random split puts a point's own near-future in the training set.
- Differencing once removes a linear trend, `d` times a degree-`d` polynomial,
  and seasonal differencing removes a period.
- The autocorrelation predicts, before any model is fitted, that the
  seasonal-naive forecast will be hard to beat — and it was.
- Backtesting on a rolling origin is the protocol; a random split is optimistic
  by a margin you measured here.

**Five questions to ask of any forecasting result:**

1. At what horizon? One step ahead is a different problem from seven.
2. Against which baseline? Naive and seasonal-naive, not the mean.
3. On what protocol — rolling origin, how many folds, anything purged at the
   boundary?
4. In what units — differenced or not, scaled or not?
5. With what spread across origins? One origin is one sample.

**Before the next lecture:** run this notebook top to bottom. Then shuffle the
rows before splitting and re-run the evaluation. The score improves and nothing
warns you — that improvement is the whole argument of this lecture.
"""),
    ]
    return cells
