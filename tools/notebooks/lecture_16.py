#!/usr/bin/env python3
"""
Lecture 16 — Recurrent networks. Chicago transit ridership, Géron Chapter 13.

No derivation of its own: Lecture 11's variance argument is the one that
applies, unrolled through time rather than down layers, and Lecture 15 owns the
stationarity mathematics.

Old lecture 19's recurrent half with old lecture 20's improvements. Every model
is scored on the SAME rolling-origin protocol Lecture 15 fixed, so the ladder is
a comparison rather than a collection.

Runs on CPU; the series is short and the networks are small.
"""

from __future__ import annotations

import nbformat as nbf

from _prompt import prompt


def md(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(text.strip("\n"))


def code(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(text.strip("\n"))


HEADER = """
# Recurrent networks

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
## 3 · Windows, and the protocol from Lecture 15

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
## 4 · A recurrent network

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
               output="one table: baseline, linear, RNN, against the target target",
               constraint="report the target beside the results, whether or not they beat it",
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
print(f"target                 {TARGET_MAE:>12,.0f}")
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

## 5 · Deeper, multivariate, and further ahead

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
       check="assert the column names, so a silent change in encoding stops the notebook. For each shifted column ask: would I know this value at 6pm the day  before? If yes it is a feature, if no it is the answer. That question is  the whole of leak detection and it takes five seconds a column."),
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
**`shift(-1)` again — and this time it is legitimate.** In Lecture 15 a
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
       check="print the shapes and the train/test split sizes. Run it on a tiny array with distinct values and read the pairs by eye, as  Lecture 19 did with six integers. Shapes agreeing is not the same as  contents aligning, and only one of the two is checkable at scale."),
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
       check="watch the held-out MAE as it trains rather than reading only the final number. The held-out MAE must fall and then flatten. If it rises, say so and  report the final epoch, not the best one you saw. Lecture 19's RNN rose at  epoch 160 and the honest number was the one at 200."),
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
       check="the table separates 'gates helped' from 'more series helped'. You should be able to name, in one sentence, the single difference between  this run and the previous one. If the sentence needs an 'and', it is two  experiments and it answers neither."),
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
## 6 · Where we are

- A recurrent cell replaces a fixed window with a state, and unrolling shows it
  is a deep network with shared weights.
- Backpropagation through time is Lecture 11's problem with one aggravation:
  the same matrix multiplies the gradient at every step, so its powers vanish
  or explode.
- A gate is a path along which the gradient is *added to* rather than
  multiplied. Everything else about LSTM and GRU is arrangement.
- Every model here was scored on Lecture 15's rolling origin, so the rows can
  be read against one another.

**Before the next lecture:** run this notebook, then lengthen the window from
56 steps to 200 and re-run the plain RNN. It gets worse, not better, and the
gradient probe says why.
"""),
    ]
    return cells
