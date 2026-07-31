"""
Lecture 19 — Tomorrow's number  (Build)

Application 10: forecast Chicago rail ridership one day ahead. A linear model
on 56 lags, then a recurrent network, scored against the one baseline that is
genuinely hard to beat — copy the same weekday last week.

Exports build() -> list[nbformat cell]. Self-contained: it downloads its own
data and imports everything it uses.

The defect this notebook plants on purpose is the one every time series invites:
a random K-fold. It runs, it is the same call scikit-learn is always used with,
and it reports a number about a fifth better than the truth — because with a
shuffled split, the day before and the day after the held-out day are both in
the training set. The repair is Lecture 20's whole subject; here we only have to
notice that the number is too good.
"""

from __future__ import annotations

import nbformat as nbf

from _prompt import prompt


def md(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(text.strip("\n"))


def code(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(text.strip("\n"))


HEADER = """
# Tomorrow's number

**Lecture 19 · Build** · Géron, Chapter 15

Applications of Machine Learning — BSc Mathematics of Artificial Intelligence

---

**How to use this notebook.** Read before you run. The cell marked
**⚠ read before running** contains a defect on purpose, and it is the defect
this application is about: it runs, it prints a believable number, and the
number is better than the model can actually do.

**A CPU runtime is enough.** The largest model here has a few thousand
parameters. If Colab offers you a GPU, decline it — for sequences this short the
transfers cost more than the arithmetic saves.

**One number to commit to.** Before you fit anything you will write down a
target. Everything after that is measured against it, including the parts that
disappoint.
"""

SETUP = '''
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
model across it: this notebook works on **2016-01 to 2019-05**, and Lecture 20
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
        md("""
## 4 · Choose a metric, commit a number

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

# The pool we will work on, and the days we will be scored on. The first 56 days
# are consumed by the window the models need, so they cannot be targets.
WINDOW = 56
pool = df["rail"]["2016-01":"2019-05"]
target = pool[WINDOW:]

print(f"pool   {len(pool):,} days, {pool.index.min().date()} to {pool.index.max().date()}")
print(f"scored {len(target):,} days, from {target.index.min().date()}")
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

### ★ Commit

Write it down now, before you have seen a model's score. It is much harder to
call a result disappointing after you know what it is.
"""),
        prompt(
               input="the naive-forecast MAE",
               output="a target MAE written down before any model is fitted",
               constraint="the improvement is chosen and recorded now, not chosen later to fit the result",
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
## 5 · Turning a series into a table

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
## 6 · ⚠ Read before running

Here is the cell to write a specification for. What you want is "score a linear
model on 56 lags". Write the specification, generate the code, and read what
comes back before you run it.

```
# YOUR SPECIFICATION HERE
#
#   input:      X of shape (1191, 56, 1), y of shape (1191, 1)
#   output:     mean absolute error in boardings
#   constraint: ...
#   check:      ...
```

The two lines left blank are the whole exercise. Below is what an assistant
returns when you leave them blank — and it is the most ordinary code in this
course.
"""),
        prompt(
               input="X flattened to (rows, 56), y as a vector",
               output="a cross-validated MAE in boardings",
               constraint="none stated — this is the under-specified prompt, and the missing constraint is the whole cell",
               check="none stated either, which is why the number goes unchallenged"),
        code('''
# ⚠ read before running — this is the assistant's answer to an under-specified
# prompt. It runs. Its number is wrong.
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score

Xflat = X.reshape(len(X), -1).numpy()          # (1191, 56)
yflat = y.ravel().numpy()

cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
folds = -cross_val_score(LinearRegression(), Xflat, yflat, cv=cv,
                         scoring="neg_mean_absolute_error")
print(f"cross-validated MAE {1e6 * folds.mean():,.0f}")
print(f"per fold            {np.round(1e6 * folds).astype(int)}")
print(f"naive baseline      {NAIVE_MAE:,.0f}")
'''),
        md("""
### Why the number hid the defect

It is not absurd. It is about 20% better than the naive forecast, the five folds
agree with each other to within a couple of percent, and the standard deviation
across folds is small — which reads as *stable*, and stability is what we are
usually taught to look for.

Look at what `shuffle=True` does to a series. Held-out day *t* has days *t−1* and
*t+1* in the training set. Yesterday's ridership predicts today's very well, so
the model is not forecasting: it is **interpolating between two days it has
already seen**. Nothing you will ever deploy has that.

Score it the way it will be used — train on the past, predict the future — and
the same model on the same data gives a materially worse number:
"""),
        prompt(
               input="the same X and y",
               output="MAE from a single split at 80% of the way through time",
               constraint="every training row must come before every test row",
               check="compare against the shuffled number and state the gap in boardings and per cent"),
        code('''
# The same model, split by time instead of at random.
cut = int(len(Xflat) * 0.8)
lin = LinearRegression().fit(Xflat[:cut], yflat[:cut])
honest = 1e6 * np.abs(lin.predict(Xflat[cut:]) - yflat[cut:]).mean()

shuffled = 1e6 * folds.mean()
print(f"random 5-fold   MAE {shuffled:>10,.0f}   <- what the assistant reported")
print(f"train past,     MAE {honest:>10,.0f}   <- what it can actually do")
print(f"predict future")
print()
print(f"the split flattered it by {honest - shuffled:,.0f} boardings "
      f"({100 * (honest - shuffled) / honest:.0f}%)")
'''),
        md("""
**The five folds agreeing did not mean the number was right.** They agreed
because they were all wrong in the same way. A stable measurement of the wrong
quantity is stable.

Keep both numbers. In the next lecture we take the honest one apart and find
that it is still not honest enough.
"""),
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
## 8 · A recurrent network

The linear model treats lag 3 and lag 45 as two unrelated columns. A recurrent
layer instead reads the window one step at a time, carrying a hidden state:

$$\\mathbf{h}_t = \\tanh(W_x \\mathbf{x}_t + W_h \\mathbf{h}_{t-1} + \\mathbf{b})$$

Same weights at every step. That is the parameter saving, and it is also the
assumption: *what a value means does not depend on where in the window it sits.*
"""),
        prompt(
               input="a batch of windows, [batch, time, series]",
               output="one number per window",
               constraint="a linear head on the final step — the hidden state is bounded by tanh and ridership is not",
               check="print the parameter count, split between recurrent and head"),
        code('''
class SimpleRnn(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, output_size=1):
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.head = nn.Linear(hidden_size, output_size)

    def forward(self, X):                      # X: [batch, time, series]
        outputs, last_state = self.rnn(X)
        return self.head(outputs[:, -1])       # only the final step

model = SimpleRnn()
n_params = sum(p.numel() for p in model.parameters())
print(f"{n_params:,} parameters "
      f"({sum(p.numel() for p in model.rnn.parameters()):,} recurrent, "
      f"{sum(p.numel() for p in model.head.parameters()):,} in the head)")
'''),
        md("""
**Why the head?** `nn.RNN` gives you a hidden state of 32 numbers, in [−1, 1]
because of the tanh. Ridership is not in [−1, 1]. The linear head is what turns a
state into a quantity, and leaving it out is the most common way this model is
got wrong.

### The training loop

Three choices worth stating:

* **Huber loss** with δ = 0.05 (that is 50,000 boardings). Quadratic near zero,
  linear far away — so a strike day steers the fit less than it would under MSE.
* **Shuffle the windows.** The *series* must not be shuffled; the *windows* must
  be, or every batch is one contiguous stretch and the gradients are correlated.
* **Split by time.** Same 80/20 cut as above, so the two models are comparable.
"""),
        prompt(
               input="the training windows",
               output="a trained network, with held-out MAE printed as it goes",
               constraint="shuffle the WINDOWS but never the series; Huber loss so a strike day does not steer the fit; the same time split as the linear model",
               check="held-out MAE should fall then flatten; if it rises, say so rather than picking the best epoch"),
        code('''
from torch.utils.data import DataLoader, TensorDataset

train_set = TensorDataset(X[:cut], y[:cut])
loader = DataLoader(train_set, batch_size=32, shuffle=True)

model = SimpleRnn()
loss_fn = nn.HuberLoss(delta=0.05)
opt = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)

EPOCHS = 200
history = []
for epoch in range(EPOCHS):
    model.train()
    for Xb, yb in loader:
        opt.zero_grad()
        loss_fn(model(Xb), yb).backward()
        opt.step()
    if (epoch + 1) % 20 == 0 or epoch == 0:
        model.eval()
        with torch.no_grad():
            val = 1e6 * (model(X[cut:]) - y[cut:]).abs().mean().item()
        history.append((epoch + 1, val))
        print(f"epoch {epoch + 1:>3d}   held-out MAE {val:>10,.0f}")
'''),
        prompt(
               input="the trained network and the held-out windows",
               output="one table: baseline, linear, RNN, against the committed target",
               constraint="report the committed number beside the results, whether or not they beat it",
               check="the naive baseline appears in the table, not only in the prose"),
        code('''
model.eval()
with torch.no_grad():
    rnn_mae = 1e6 * (model(X[cut:]) - y[cut:]).abs().mean().item()

rows = [("copy last week (no fitting)", NAIVE_MAE),
        ("linear, 56 lags",             honest),
        ("simple RNN, 32 units",        rnn_mae)]
print(f"{'model':32s}{'MAE':>12s}{'vs naive':>12s}")
for name, score in rows:
    print(f"{name:32s}{score:>12,.0f}{(NAIVE_MAE - score) / NAIVE_MAE:>11.1%}")
print()
print(f"committed target                 {TARGET_MAE:>12,.0f}")
'''),
    ]

    cells += [
        md("""
## 9 · Where we finish

Read the table honestly, against the number you committed to in section 4 —
not against zero, and not against the mean.

Three things this notebook did **not** do, and the next lecture does:

1. **The split is still wrong.** An 80/20 cut by time is better than a shuffle,
   but the model still chose nothing on a held-out period — and we compared two
   models on the same test set, which is a selection we have not paid for.
2. **Nothing was differenced.** Lecture 20's mathematical thread is
   stationarity, and it explains why the naive forecast is so strong here.
3. **One step ahead only.** A staffing decision needs a week.

### ★ Record your number

Whatever the RNN scored, write it down with the date, the split, and the seed.
It is the number Lecture 20 will attack.

### Red-team your own notebook

* Change `RANDOM_STATE`. How much does the RNN's score move? If it moves more
  than the margin over the baseline, you do not have a margin — you have a seed.
* Set `WINDOW` to 7, then to 112. Does the linear model improve? Does the RNN?
* Replace `shift(7)` with `shift(-7)` in the baseline. The score improves. Say,
  in one sentence, exactly why that number is worthless.
"""),
    ]
    return cells
