"""
Lecture 20 — When the past does not repeat  (Break → Fix)

Application 10, second half. It reproduces Lecture 19's broken measurement
first, so the two numbers sit side by side, then repairs the protocol, then
spends what is left of the honest margin on models that earn it: multivariate
input, gates, and a fourteen-day horizon.

Exports build() -> list[nbformat cell]. Self-contained.

The thread is stationarity, and it is not decoration: it is what explains why
copying last week is so hard to beat, and why the first difference of this
series has a *larger* standard deviation than the series itself.
"""

from __future__ import annotations

import nbformat as nbf

from _prompt import prompt


def md(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(text.strip("\n"))


def code(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(text.strip("\n"))


HEADER = """
# When the past does not repeat

**Lecture 20 · Break → Fix** · Géron, Chapter 15

Applications of Machine Learning — BSc Mathematics of Artificial Intelligence

---

**This notebook reproduces the broken measurement first.** Both numbers are
printed side by side before anything is repaired, because the point is not that
one of them is wrong — it is that nothing in the output of the wrong one tells
you so.

**A CPU runtime is enough**, and for sequences this short it is faster than the
GPU: the models are small enough that moving the data costs more than the
arithmetic saves.
"""

SETUP = '''
# --- setup -------------------------------------------------------------------
import sys
from pathlib import Path
import tarfile, urllib.request

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

print(f"python  {sys.version.split()[0]}")
print(f"torch   {torch.__version__}")

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)
plt.rcParams.update({"figure.figsize": (11, 3.2), "axes.grid": True,
                     "grid.alpha": 0.3, "font.size": 11})

def load_ridership():
    tarball = Path("datasets/ridership.tgz")
    if not tarball.is_file():
        Path("datasets").mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(
            "https://github.com/ageron/data/raw/main/ridership.tgz", tarball)
        with tarfile.open(tarball) as t:
            t.extractall(path="datasets", filter="data")
    return pd.read_csv(
        "datasets/ridership/CTA_-_Ridership_-_Daily_Boarding_Totals.csv",
        parse_dates=["service_date"])

df = load_ridership()
df.columns = ["date", "day_type", "bus", "rail", "total"]
df = df.sort_values("date").set_index("date").drop("total", axis=1)
df = df.drop_duplicates()

WINDOW = 56
pool = df["rail"]["2016-01":"2019-05"]
print(f"{len(df):,} days on file; {len(pool):,} in the pool we model")
'''


def build() -> list:
    cells: list = [md(HEADER), md("## 1 · Setup and the same data"),prompt(
                                                                           input="the same CTA file as the previous lecture",
                                                                           output="the tidied frame and the 2016-2019 pool, ready to model",
                                                                           constraint="identical preparation to Lecture 19, so any difference in the numbers is the protocol and not the data",
                                                                           check="print the day count and confirm it matches"),
                                                                     code(SETUP)]

    cells += [
        md("""
## 2 · Thread 10 — stationarity

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
       check="the level series should fail to look stationary where the differenced ones do not"),
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
       check="flag any lag where differencing makes the spread larger, which is the point of the cell"),
        code('''
sd = pool.std()
print(f"{'series':28s} sd {sd:>10,.0f}")
for h in (1, 7, 14):
    rho = pool.autocorr(lag=h)
    predicted = np.sqrt(2 * sd**2 * (1 - rho))
    measured = pool.diff(h).std()
    flag = "  <- WORSE than not differencing" if measured > sd else ""
    print(f"lag {h:>2d}: rho {rho:+.3f}   predicted sd {predicted:>10,.0f}"
          f"   measured {measured:>10,.0f}{flag}")
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
       check="the first difference should look wilder than the series it came from"),
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
       check="spikes at 7, 14 and 21 in the raw series; nothing much left after differencing"),
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
## 3 · The broken measurement, reproduced

Exactly the cell from the previous lecture. Run it and look at the folds: they
agree with one another, which is what a stable measurement looks like.
"""),
prompt(
       input="the pool as 56-lag windows",
       output="a cross-validated MAE from a shuffled five-fold split",
       constraint="this is the previous lecture's broken cell, reproduced exactly",
       check="look at the fold spread — they agree with each other, which is what a stable measurement of the wrong quantity looks like"),
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
       check="report the gap against the shuffled number in boardings and per cent"),
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
       check="print mean and fold spread for all three protocols together"),
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
       check="state what share of the claimed margin was protocol rather than model"),
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
### Which of those four is "the" number?

The lecture quotes the **single forward hold-out**, because it is the protocol
that matches how the model would actually be used: fit once on everything up to
a date, forecast forward from there. The purged five-fold is stricter still, and
its margin is smaller again — mostly because its first fold trains on a couple of
hundred days and is scored anyway.

The useful discipline is not picking the smallest number. It is **saying which
protocol produced the one you quote**, so that a reader can reproduce it and a
colleague can disagree with it. A margin without a protocol attached is not a
result.

## 4 · Spending what is left

The honest margin is smaller, so the improvements have to be real. Three, in
order of how much they buy.

### Improvement 1 — more series, not more layers

Stacking three recurrent layers is the reflex, and on 1,191 windows it overfits.
What actually helps is giving the model something it does not already have: bus
ridership, and **tomorrow's day type**, which is known in advance from a calendar
and is therefore not a leak.
"""),
prompt(
       input="rail, bus, and tomorrow's day type from the calendar",
       output="a five-column frame, day type one-hot encoded",
       constraint="shift(-1) on the CALENDAR is legitimate and shift(-1) on the target is a leak — the test is whether the value is knowable at prediction time",
       check="assert the column names, so a silent change in encoding stops the notebook"),
        code('''
mulvar = df[["rail", "bus"]] / 1e6
mulvar["next_day_type"] = df["day_type"].shift(-1)   # known in advance
mulvar = pd.get_dummies(mulvar, dtype=float)         # 5 columns

assert list(mulvar.columns) == ["rail", "bus", "next_day_type_A",
                                "next_day_type_U", "next_day_type_W"], \\
    list(mulvar.columns)
mulvar = mulvar["2016-01":"2019-05"].dropna()
print(mulvar.shape, list(mulvar.columns))
mulvar.head(3)
'''),
        md("""
**`shift(-1)` again — and this time it is legitimate.** In Lecture 19 a
`shift(-1)` on the *target* was a leak. Here it is applied to the calendar, and
the difference is not the sign: it is whether the value would be available at
the moment of the forecast. Tomorrow's day type is on a wall planner. Tomorrow's
ridership is not.

State the rule you are using, every time, in one line: *would I know this number
when I have to make the prediction?*
"""),
prompt(
       input="the multivariate frame, a window and a horizon",
       output="windows over all series, with rail alone as the target",
       constraint="one windowing function used for every model below, so the comparison is like for like",
       check="print the shapes and the train/test split sizes"),
        code('''
from torch.utils.data import DataLoader, TensorDataset

def make_windows(frame, window=WINDOW, horizon=1):
    """Windows over several series; the target is `rail` only."""
    arr = torch.tensor(frame.values, dtype=torch.float32)
    rail = arr[:, 0]
    n = len(arr) - window - horizon + 1
    Xs = torch.stack([arr[i:i + window] for i in range(n)])
    ys = torch.stack([rail[i + window:i + window + horizon] for i in range(n)])
    return Xs, ys

Xm, ym = make_windows(mulvar)
cut = int(len(Xm) * 0.8)
print(f"X {tuple(Xm.shape)}   y {tuple(ym.shape)}   train {cut}, test {len(Xm) - cut}")
'''),
        md("""
### Improvement 2 — gates

A simple RNN multiplies by the same recurrent matrix at every step, so gradients
over 56 steps either vanish or explode — Lecture 13's thread, in a new place. A
**GRU** adds two gates: an update gate that decides how much of the old state to
keep, and a reset gate that decides how much of it to use. Keeping is now
addition rather than repeated multiplication, so a gradient can travel.
"""),
prompt(
       input="windows of five series",
       output="a trained GRU and its held-out MAE",
       constraint="gates rather than a plain RNN, because a 56-step recurrence multiplies by the same matrix every step",
       check="watch the held-out MAE as it trains rather than reading only the final number"),
        code('''
class GruModel(nn.Module):
    def __init__(self, input_size=5, hidden_size=32, output_size=1):
        super().__init__()
        self.rnn = nn.GRU(input_size, hidden_size, batch_first=True)
        self.head = nn.Linear(hidden_size, output_size)

    def forward(self, X):
        outputs, _state = self.rnn(X)
        return self.head(outputs[:, -1])

def train(model, Xtr, ytr, Xte, yte, epochs=120, lr=0.005, quiet=False):
    loss_fn = nn.HuberLoss(delta=0.05)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=32, shuffle=True)
    for epoch in range(epochs):
        model.train()
        for Xb, yb in loader:
            opt.zero_grad()
            loss_fn(model(Xb), yb).backward()
            opt.step()
        if not quiet and ((epoch + 1) % 30 == 0 or epoch == 0):
            model.eval()
            with torch.no_grad():
                v = 1e6 * (model(Xte) - yte).abs().mean().item()
            print(f"  epoch {epoch + 1:>3d}   held-out MAE {v:>10,.0f}")
    model.eval()
    with torch.no_grad():
        return 1e6 * (model(Xte) - yte).abs().mean().item()

torch.manual_seed(RANDOM_STATE)
print("GRU on five series:")
gru_mae = train(GruModel(input_size=5), Xm[:cut], ym[:cut], Xm[cut:], ym[cut:])
'''),
prompt(
       input="the same GRU, on rail alone",
       output="its held-out MAE, beside the five-series version",
       constraint="change ONE thing — two changes at once is not an experiment",
       check="the table separates 'gates helped' from 'more series helped'"),
        code('''
# The same model on rail alone, to separate "gates helped" from "more series
# helped". Two changes at once is not an experiment.
torch.manual_seed(RANDOM_STATE)
Xr, yr = make_windows(mulvar[["rail"]])
print("GRU on rail alone:")
gru_rail = train(GruModel(input_size=1), Xr[:cut], yr[:cut], Xr[cut:], yr[cut:],
                 quiet=True)

print()
print(f"{'model':34s}{'MAE':>12s}{'vs naive':>11s}")
for name, score in [("copy last week", NAIVE_MAE),
                    ("linear, forward + purge", folds_gap.mean()),
                    ("GRU, rail only", gru_rail),
                    ("GRU, five series", gru_mae)]:
    print(f"{name:34s}{score:>12,.0f}{(NAIVE_MAE - score) / NAIVE_MAE:>10.1%}")
'''),
    ]

    cells += [
        md("""
### Improvement 3 — forecast a fortnight, not a day

A staffing decision needs more than tomorrow. Two changes: the target becomes 14
values instead of 1, and the head produces 14 numbers.
"""),
prompt(
       input="the same windows, with a fourteen-day target",
       output="MAE at each horizon, plotted against the naive baseline",
       constraint="one head producing fourteen numbers, not fourteen models",
       check="read the SHAPE — a single average would hide that day 14 is no better than copying last week"),
        code('''
HORIZON = 14
Xh, yh = make_windows(mulvar, horizon=HORIZON)
cut_h = int(len(Xh) * 0.8)

torch.manual_seed(RANDOM_STATE)
horizon_model = GruModel(input_size=5, output_size=HORIZON)
h_mae = train(horizon_model, Xh[:cut_h], yh[:cut_h], Xh[cut_h:], yh[cut_h:],
              quiet=True)

horizon_model.eval()
with torch.no_grad():
    pred = horizon_model(Xh[cut_h:])
per_step = 1e6 * (pred - yh[cut_h:]).abs().mean(dim=0).numpy()

fig, ax = plt.subplots(figsize=(11, 3.2))
ax.plot(range(1, HORIZON + 1), per_step, marker="o", color="#0b3d62",
        label="the model")
ax.axhline(NAIVE_MAE, color="#c0392b", ls="--", label="copy last week")
ax.set_xlabel("days ahead"); ax.set_ylabel("MAE")
ax.set_title("Error against horizon — the margin is spent by about day 7")
ax.legend(); plt.show()

for k in (1, 7, 14):
    print(f"{k:>2d} days ahead   MAE {per_step[k - 1]:>10,.0f}"
          f"   vs naive {(NAIVE_MAE - per_step[k - 1]) / NAIVE_MAE:>7.1%}")
'''),
        md("""
**Read the shape, not the average.** One number for "the fourteen-day forecast"
would hide that day 1 is good and day 14 is no better than copying last week.
Report the curve. If a decision only needs three days, say so and be judged on
three.
"""),
    ]

    cells += [
        md("""
## 5 · Regime change

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
       check="the ratio of the two levels is printed, not described"),
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
the live error against the committed number, and have a rule that says when to
stop trusting the model. A model that is never re-measured after deployment is
an assumption wearing a number's clothes.
"""),
    ]

    cells += [
        md("""
## 6 · The temporal checklist

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

### Red-team your own notebook

* Set `gap=0` in `TimeSeriesSplit`. How much of the margin comes back? That
  amount was adjacency.
* Give the model `df["day_type"]` **without** the `shift(-1)`. The score barely
  moves — explain why that is worse, not better.
* Train on 2016–2019, test on 2020. Then argue, in two sentences, whether the
  model was wrong or the question was.
"""),
    ]
    return cells
