#!/usr/bin/env python3
"""
Lecture 16 — Recurrent networks. Chicago transit ridership, Géron Chapter 13.

No derivation of its own: Lecture 11's variance argument is the one that
applies, unrolled through time rather than down layers, and Lecture 15 owns the
stationarity mathematics.

Old lecture 19's recurrent half with old lecture 20's improvements. Every model
is scored on the single forward hold-out Lecture 15 named, so the ladder is
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

**Time.** Most of this notebook is seconds a cell. Section 6 runs the whole
experimental programme the slides report — the six-model ladder, the same model
under a random split, fourteen horizons, a convolutional alternative, and the
2020 regime change — under one protocol. Expect **five to fifteen minutes** for
that section, depending on the machine. It is the lecture's result rather than a
detour, which is why it is run here rather than quoted.
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
                          "the way of the argument",
        **{"try": "change the seed here and re-run the ladder in Section 15. "
                  "The MAEs move by hundreds of boardings. Hold that spread "
                  "in mind when the ladder reports the differences between "
                  "six architectures."}),
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
                     "publisher states",
        **{"try": "re-download the CSV and compare the printed row count with "
                  "Lecture 15's. The two lectures have to be scored on the "
                  "same days, and this is the one cell where that could "
                  "quietly stop being true without anything raising."}),
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
               check="print the count removed and the date range that survives",
               **{"try": "keep the duplicate rows and re-run the notebook. "
                         "Every MAE moves a little and no assert fires. "
                         "Duplicates in a time series are not a leak, they "
                         "are a re-weighting: some days now count twice in "
                         "the fit and once in the score."}),
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
               check="the three printed values are A, U, U",
               **{"try": "count the U days that fall on a Monday to Friday. "
                         "Those are the public holidays, and they are the "
                         "only reason this column beats a plain day-of-week "
                         "feature. How many are in the pool, and is that "
                         "enough to learn anything from?"}),
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
               check="length is len(series) - window_length",
               **{"try": "return self.series[idx:end + 1] as the window. "
                         "__len__ is unchanged and the target is now the last "
                         "element of its own window. Which assert two cells "
                         "down catches it, and how many cells would have run "
                         "before it fired?"}),
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
               check="read the output and confirm the target is never inside the window",
               **{"try": "make the toy values 10, 20, 30, 40, 50, 60 instead "
                         "of 0 to 5. The pairs become unambiguous at a glance "
                         "— with 0 to 5 a window of [0, 1, 2] could be read "
                         "as positions rather than values. Choose test data "
                         "whose values cannot be mistaken for its indices."}),
        code('''
toy = torch.tensor([[0], [1], [2], [3], [4], [5]])
for window, t in TimeSeriesDataset(toy, window_length=3):
    print(window.flatten().tolist(), "->", t.item())
'''),
        prompt(
               input="the tidied frame; rebuild the same pool and window as Lecture 15",
               output="X of shape (rows, 56, 1) and y of shape (rows, 1)",
               constraint="assert the shapes rather than printing them, so a wrong shape stops the notebook",
               check="rows equals len(pool) - 56",
               **{"try": "change WINDOW to 28 here and leave Lecture 15 "
                         "alone. Both notebooks still run, both print an MAE, "
                         "and the two are no longer comparable. Which printed "
                         "line would have told you, and does any assert?"}),
        code('''
# The same pool and the same window as Lecture 15, rebuilt here rather than
# inherited: a notebook that only runs because a previous one is still in memory
# is not reproducible, and the two lectures must be scored on the same days.
WINDOW = 56
pool = df["rail"]["2016-01":"2019-05"]
target = pool[WINDOW:]
assert len(target) == len(pool) - WINDOW

series = torch.tensor(pool.values / 1e6, dtype=torch.float32).unsqueeze(1)
dataset = TimeSeriesDataset(series, WINDOW)

X = torch.stack([w for w, _ in dataset])       # [rows, time, series]
y = torch.stack([t for _, t in dataset])

# The split, fixed here and used by every model below. By TIME, not at random:
# the windows overlap, so a shuffled split puts almost every test window's
# neighbours in the training set.
cut = int(len(X) * 0.8)

# Lecture 15's reference numbers, recomputed here rather than inherited from a
# kernel that may not still be running: the naive forecast every model has to
# beat, the target committed to before anything was fitted, and the linear
# model under the honest forward split. A notebook that only runs because a
# previous one is still in memory is not reproducible, and these decide whether
# anything below is an improvement.
from sklearn.linear_model import LinearRegression

_naive = pool.shift(7)[WINDOW:]
_mask  = target.notna() & _naive.notna()

# The baseline has to be measured on the rows the models are scored on. Every
# model below is scored on the forward test slice only, so a baseline averaged
# over all 1,191 days would be a different set of days -- and Lecture 15 spent
# an hour on why that comparison is not a comparison. The all-row figure is
# kept, but only to show how far apart the two are.
_test = _mask.copy()
_test.iloc[:cut] = False
NAIVE_ALL  = float((target[_mask] - _naive[_mask]).abs().mean())
NAIVE_MAE  = float((target[_test] - _naive[_test]).abs().mean())

# The commitment does NOT move. Lecture 15 fixed it at 10% better than the
# naive forecast, and printed the number precisely so it could not be quietly
# revised afterwards. It was computed from the pool-wide baseline, so it is
# 0.9 * NAIVE_ALL -- and we inherit that number rather than recomputing it.
#
# Recomputing it here would be worse than a rounding change. The forward rows
# are harder, so the row-matched baseline is larger, so 10% of it is a LAXER
# bar: 60,503 instead of 49,859, and models that miss the commitment would
# appear to clear it. A correction that moves the bar in your own favour is
# exactly the one to distrust.
COMMITTED_MAE = NAIVE_ALL * (1 - 0.10)       # 49,859 -- Lecture 15's, unchanged
BAR_ON_TEST   = NAIVE_MAE * (1 - 0.10)       # what the same rule asks on these rows

_values = pool.values / 1e6
_Xl = np.stack([_values[i:i + WINDOW] for i in range(len(_values) - WINDOW)])
_yl = _values[WINDOW:]
honest = 1e6 * float(np.abs(
    LinearRegression().fit(_Xl[:cut], _yl[:cut]).predict(_Xl[cut:]) - _yl[cut:]).mean())

print(f"train {cut}, test {len(X) - cut}, split by time")
print(f"naive, test rows only   MAE {NAIVE_MAE:>10,.0f}")
print(f"naive, all 1,191 days   MAE {NAIVE_ALL:>10,.0f}  <- not comparable")
print(f"committed target        MAE {COMMITTED_MAE:>10,.0f}   Lecture 15's, fixed")
print(f"  the same 10% here     MAE {BAR_ON_TEST:>10,.0f}   laxer, because these rows are harder")
print(f"linear, forward split   MAE {honest:>10,.0f}")

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
               check="print the parameter count, split between recurrent and head",
               **{"try": "delete the head and return outputs[:, -1] directly. "
                         "tanh bounds the hidden state to (-1, 1) and the "
                         "scaled targets sit near 0.6, so it half works — "
                         "which is worse than failing. Now try the same thing "
                         "on the raw boardings, where it cannot."}),
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
               check="held-out MAE should fall then flatten; if it rises, say so rather than picking the best epoch",
               **{"try": "swap HuberLoss for MSELoss and re-run. The held-out "
                         "MAE gets worse, and one strike day is the reason: "
                         "squared error spends the model's capacity on the "
                         "largest residual, which is the day nobody was "
                         "staffing for anyway."}),
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
               check="the naive baseline appears in the table, not only in the prose",
               **{"try": "delete the target line from the printout. The table "
                         "still reads as a result — three models, three "
                         "numbers, one of them best. Committing to a target "
                         "before fitting is the only thing that turns that "
                         "into a decision."}),
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
print(f"committed target       {COMMITTED_MAE:>12,.0f}   (Lecture 15, fixed before fitting)")
'''),
    ]
    cells += [
        md("""
### Which of these is "the" number?

This lecture quotes the **single forward hold-out**, because it is the protocol
that matches how the model would actually be used: fit once on everything up to
a date, forecast forward from there. Lecture 15 measured three others on the
same series — a random five-fold, a forward five-fold, and a forward five-fold
with a purge — and it is worth going back to that table before accepting this
one.

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
       check="assert the column names, so a silent change in encoding stops the notebook. For each shifted column ask: would I know this value at 6pm the day  before? If yes it is a feature, if no it is the answer. That question is  the whole of leak detection and it takes five seconds a column.",
       **{"try": "use shift(+1) rather than shift(-1) on day_type: "
                 "yesterday's, not tomorrow's. The column-name assert still "
                 "passes and the model still trains, but the feature is now "
                 "useless rather than leaky. Which of those two mistakes "
                 "would you notice sooner?"}),
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
       check="print the shapes and the train/test split sizes. Run it on a tiny array with distinct values and read the pairs by eye, as  Lecture 19 did with six integers. Shapes agreeing is not the same as  contents aligning, and only one of the two is checkable at scale.",
       **{"try": "run make_windows on a five-row frame of distinct integers "
                 "and read every pair by eye, as the check asks. Then check "
                 "your reading against the printed shapes. Only one of the "
                 "two would catch a target taken from the wrong column."}),
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
       check="watch the held-out MAE as it trains rather than reading only the final number. The held-out MAE must fall and then flatten. If it rises, say so and  report the final epoch, not the best one you saw. Lecture 19's RNN rose at  epoch 160 and the honest number was the one at 200.",
       **{"try": "replace nn.GRU with nn.RNN in GruModel and change nothing "
                 "else. Over a 56-step recurrence the same matrix is applied "
                 "fifty-six times, and the held-out MAE is what the gates "
                 "were buying."}),
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
       check="the table separates 'gates helped' from 'more series helped'. You should be able to name, in one sentence, the single difference between  this run and the previous one. If the sentence needs an 'and', it is two  experiments and it answers neither.",
       **{"try": "add a fourth row: the GRU on rail and bus but WITHOUT the "
                 "day-type columns. It separates 'more series helped' from "
                 "'the calendar helped', which this table still confounds in "
                 "one row."}),
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
                    ("linear, forward split", honest),
                    ("GRU, rail only", gru_rail),
                    ("GRU, five series", gru_mae)]:
    print(f"{name:34s}{score:>12,.0f}{(NAIVE_MAE - score) / NAIVE_MAE:>10.1%}")
'''),
    ]
    cells += [
        md("""
## 6 · The ladder, under the protocol the deck reports

Everything above built one model at a time to explain a mechanism. This section
runs the whole comparison at once, under the protocol the slides quote — because
a table of six models is only a table if every row was produced the same way.

Two things are held fixed and stated. The **recipe** — optimiser, learning rate
and epoch count — is chosen for each architecture on a slice of the *training*
period, never on the test period; an architecture that loses only because it was
given the wrong learning rate has not been shown to lose. And the **split** is
by time: everything before 2019 trains, everything from 2019 is scored.

⏱ **about ten minutes on CPU** for all six. It is the longest cell in the
course, and it is the lecture's result rather than a detour.
"""),
        prompt(
            label="⏱ 10 min — six architectures, one protocol",
            input="the univariate and multivariate window sets",
            output="test MAE, train MAE and the chosen recipe for each of six models",
            constraint="select the recipe on a held-out slice of the TRAINING period and then refit on all of it — selecting on the test period is the failure this whole part of the course is about, and it is one line away at every step",
            check="print the recipe each model chose beside its score. If every architecture chose the same one, the selection step is doing nothing and should be removed rather than reported.",
            **{"try": "fix every model to the sgd recipe and re-run. At least one architecture gets substantially worse, which is what the selection step exists to prevent."}),
        code('''
import time
from torch.utils.data import DataLoader, TensorDataset


def make_arrays(frame, w=WINDOW, target=0, horizon=1):
    """Windows of w rows, labelled with the next value of column `target`."""
    V = frame.values.astype(np.float32)
    n = len(V) - w - horizon + 1
    X = np.stack([V[i:i + w] for i in range(n)])
    y = np.stack([V[i + w:i + w + horizon, target] for i in range(n)])
    return X, y, frame.index[w:w + n]


def build_model(kind, input_size, output_size=1):
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
    return {"rnn":     lambda: Rnn(nn.RNN),
            "deep":    lambda: Rnn(nn.RNN, layers=3),
            "gru":     lambda: Rnn(nn.GRU),
            "lstm":    lambda: Rnn(nn.LSTM),
            "convgru": ConvGru}[kind]()


# optimiser, learning rate, epochs. Two recipes, and every architecture is
# offered both, so no model loses for want of a learning rate.
RECIPES = {"sgd": ("sgd", 0.05, 200), "adam": ("adam", 0.005, 120)}


def fit_torch(kind, Xtr, ytr, recipe, seed=RANDOM_STATE):
    torch.manual_seed(seed)
    model = build_model(kind, Xtr.shape[2], ytr.shape[1])
    name, lr, epochs = RECIPES[recipe]
    Xt, yt = torch.tensor(Xtr), torch.tensor(ytr)
    if kind == "linear":
        model(Xt[:2])                     # LazyLinear needs one pass to build
    opt = (torch.optim.Adam(model.parameters(), lr=lr) if name == "adam"
           else torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9))
    loss_fn = nn.HuberLoss(delta=0.05)    # a strike day must not steer the fit
    loader = DataLoader(TensorDataset(Xt, yt), batch_size=32, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            opt.step()
    return model.eval()


def predict_torch(model, X):
    with torch.no_grad():
        return model(torch.tensor(X)).numpy()


def mae_1e6(pred, truth):
    return 1e6 * float(np.abs(np.asarray(pred) - np.asarray(truth)).mean())
'''),
        prompt(
            label="the selection step, and the two window sets",
            input="the tidied frame",
            output="univariate rail windows, multivariate windows, and a recipe-selection function",
            constraint="the selection slice comes out of the TRAINING period — nothing at or after the test cut is read here, and the function says so in its docstring",
            check="print the multivariate columns. The next day's day-type is a legitimate feature because it is known in advance; the same column shifted the other way would be a leak.",
            **{"try": "move CUT_SEL to 2019-03-01, inside the test period. "
                      "Every recipe is now selected on days the model is "
                      "scored on, nothing raises, and every number in the "
                      "ladder improves. That improvement is the failure, not "
                      "the result."}),
        code('''
CUT_SEL, CUT_END = "2018-07-01", "2019-01-01"

# The ladder scores from CUT_END, which is a third set of days again -- so it
# needs its own baseline for the same reason.
_lad = _mask & (target.index >= CUT_END)
NAIVE_LADDER = float((target[_lad] - _naive[_lad]).abs().mean())
NAIVE_LADDER_MAPE = float((100 * (target[_lad] - _naive[_lad]).abs()
                           / target[_lad].abs()).mean())
LADDER_MEAN = float(target[_lad].mean())
print(f"copy last week, ladder rows only  MAE {NAIVE_LADDER:>10,.0f}"
      f"   MAPE {NAIVE_LADDER_MAPE:5.1f}%")
print(f"mean level on those same days     {LADDER_MEAN:>10,.0f}"
      f"   <- the denominator for any 'share of a day'")

def select_and_fit(kind, X, y, dates):
    """Choose the recipe inside the training period, then refit on all of it.

    Nothing at or after CUT_END is touched here. That is the whole point: the
    selection slice comes out of the training period, not out of the test set.
    """
    sel, trn = dates < CUT_SEL, dates < CUT_END
    held = trn & ~sel
    scores = {}
    for recipe in RECIPES:
        m = fit_torch(kind, X[sel], y[sel], recipe)
        scores[recipe] = mae_1e6(predict_torch(m, X[held])[:, 0], y[held, 0])
    best = min(scores, key=scores.get)
    return fit_torch(kind, X[trn], y[trn], best), best, scores


POOL_FROM, POOL_TO = "2016-01", "2019-05"   # the same window as Lecture 15
rail_f = df["rail"][POOL_FROM:POOL_TO].to_frame() / 1e6
mul_f  = df[["rail", "bus"]][POOL_FROM:POOL_TO] / 1e6
mul_f["next_day_type"] = df["day_type"].shift(-1)[POOL_FROM:POOL_TO]
mul_f = pd.get_dummies(mul_f, dtype=float)

Xu, yu, du = make_arrays(rail_f)
Xm, ym, dm = make_arrays(mul_f)
print(f"univariate   {Xu.shape}   multivariate {Xm.shape}")
print(f"multivariate columns: {list(mul_f.columns)}")
'''),
        prompt(
            label="⏱ 10 min — run the ladder",
            input="six (architecture, window set) pairs",
            output="one row each: test MAE, train MAE, chosen recipe, wall clock",
            constraint="report the TRAIN MAE beside the test MAE — without it, a model that is worse on both is indistinguishable from one that overfits, and those have opposite fixes",
            check="the naive baseline goes in the same table. Six architectures that all fail to beat copying last week is a finding, and it is one the table has to be able to show.",
            **{"try": "read the train MAE column on its own. Which rows are "
                      "worse on both, and which are far better on train than "
                      "on test? Those two failures have opposite fixes, and "
                      "the test column alone cannot tell them apart."}),
        code('''
ladder = [("linear", Xu, yu, du, "Linear, 56 lags"),
          ("rnn",    Xu, yu, du, "Simple RNN, 32 units"),
          ("deep",   Xu, yu, du, "Deep RNN, 3 layers"),
          ("rnn",    Xm, ym, dm, "Simple RNN, multivariate"),
          ("gru",    Xm, ym, dm, "GRU, multivariate"),
          ("lstm",   Xm, ym, dm, "LSTM, multivariate")]

print(f"{'model':28s}{'test MAE':>10s}{'train MAE':>11s}{'recipe':>8s}")
rows = []
for kind, X, y, dates, label in ladder:
    t0 = time.perf_counter()
    model, best, scores = select_and_fit(kind, X, y, dates)
    te, tr = dates >= CUT_END, dates < CUT_END
    test_mae  = mae_1e6(predict_torch(model, X[te])[:, 0], y[te, 0])
    train_mae = mae_1e6(predict_torch(model, X[tr])[:, 0], y[tr, 0])
    rows.append((label, test_mae, train_mae, best))
    print(f"{label:28s}{test_mae:>10,.0f}{train_mae:>11,.0f}{best:>8s}"
          f"   ({time.perf_counter() - t0:.0f}s)")

print(f"\\n{'copy last week':28s}{NAIVE_LADDER:>10,.0f}")
'''),
        prompt(
            label="the recurrent model under a random split",
            input="the univariate windows, five shuffled folds",
            output="the RNN's MAE under the protocol Lecture 15 showed was wrong",
            constraint="the same architecture and the same recipe as the ladder's row — only the splitter changes, or the comparison is not about the splitter",
            check="compare it with the ladder's forward-split row. The raw gap is the protocol PLUS the rows -- the two protocols score different days, and the forward ones are harder. Put each number beside its own naive baseline: the skill ratio is what isolates the protocol, and it is still larger than every architectural difference in the table above.",
            **{"try": "run the same five folds with KFold(5, shuffle=False). "
                      "It is still not a forward split — fold 1 trains on the "
                      "future of fold 5 — but it does keep neighbouring "
                      "windows together. Where does its number fall between "
                      "the two printed here, and which of the two leaks does "
                      "that isolate?"}),
        code('''
from sklearn.model_selection import KFold

folds = []
for tr_i, te_i in KFold(5, shuffle=True, random_state=RANDOM_STATE).split(Xu):
    m = fit_torch("rnn", Xu[tr_i], yu[tr_i], "sgd")
    folds.append(mae_1e6(predict_torch(m, Xu[te_i])[:, 0], yu[te_i, 0]))
RNN_RANDOM_CV = float(np.mean(folds))

# The two protocols score DIFFERENT rows, so the raw difference mixes the
# protocol with how hard those rows are. Each number needs its own baseline,
# and the skill ratio -- model over naive on the same rows -- is what isolates
# the protocol. This is Lecture 15's own warning applied to our own comparison.
naive_random = float(np.mean([
    mae_1e6(Xu[te_i][:, -7, 0], yu[te_i, 0])
    for _, te_i in KFold(5, shuffle=True, random_state=RANDOM_STATE).split(Xu)]))

forward = [r for r in rows if r[0] == "Simple RNN, 32 units"][0][1]
print(f"{'':28s}{'MAE':>10s}{'naive':>10s}{'skill':>8s}")
print(f"{'random 5-fold':28s}{RNN_RANDOM_CV:>10,.0f}{naive_random:>10,.0f}"
      f"{RNN_RANDOM_CV / naive_random:>8.3f}")
print(f"{'forward split':28s}{forward:>10,.0f}{NAIVE_LADDER:>10,.0f}"
      f"{forward / NAIVE_LADDER:>8.3f}")
print()
print(f"raw difference  {forward - RNN_RANDOM_CV:>10,.0f}  <- protocol AND rows")
print(f"skill worsens by{forward / NAIVE_LADDER - RNN_RANDOM_CV / naive_random:>10.3f}"
      f"  <- the protocol alone")
'''),

        md("""
### Several steps ahead

One output was a choice. A system that plans staffing needs a fortnight, not a
day — so give the model fourteen outputs and read the error at each horizon.
"""),
        prompt(
            label="fourteen horizons from one model",
            input="the multivariate windows, labelled with the next fourteen days",
            output="MAE at each horizon, beside a naive forecast at the same horizon",
            constraint="the naive reference has to be honest at every horizon — the last value of the SAME weekday still inside the window, which is a different window position for each h",
            check="the error should rise with the horizon and then flatten. If it is flat from the start, the model is predicting the weekly pattern and nothing else.",
            **{"try": "replace naive_pos with a constant 55 — yesterday, at "
                      "every horizon. The naive column stops rising with h "
                      "and the GRU suddenly looks excellent at t+14. A "
                      "baseline that does not get harder with the horizon is "
                      "not a baseline for that horizon."}),
        code('''
Xa, ya, da = make_arrays(mul_f, horizon=14)
model, best, _ = select_and_fit("gru", Xa, ya, da)
te = da >= CUT_END
P = predict_torch(model, Xa[te])

# Window position 55 is day t, so the same weekday before t+h sits at 48+h for
# h <= 7 and at 41+h for h <= 14.
naive_pos = [48 + h if h <= 7 else 41 + h for h in range(1, 15)]
ahead = [mae_1e6(P[:, h], ya[te, h]) for h in range(14)]
ahead_naive = [mae_1e6(Xa[te, naive_pos[h], 0], ya[te, h]) for h in range(14)]

print(f"recipe chosen: {best}")
print(f"{'horizon':>8s}{'GRU':>10s}{'naive':>10s}")
for h in (0, 6, 13):
    print(f"{'t+' + str(h + 1):>8s}{ahead[h]:>10,.0f}{ahead_naive[h]:>10,.0f}")
print(f"\\nt+1 {ahead[0]:,.0f} rises to t+14 {ahead[13]:,.0f}")
'''),

        prompt(
            label="a convolutional alternative on the same task",
            input="the same fourteen-horizon task, on a 112-day window",
            output="MAE at each horizon",
            constraint="give it a LONGER window — the convolution halves the sequence before the recurrent layer, so it can afford one, and a fair comparison lets each architecture have the input it is built for",
            check="report where it wins and where it loses rather than a single mean. It is better at the near horizons and worse at the far ones, and a mean would hide both.",
            **{"try": "give the conv+GRU the same 56-day window as the "
                      "others. It loses its near-horizon advantage, because "
                      "the convolution halves the sequence and it is now "
                      "working from 28 effective steps. Which of the two "
                      "comparisons is fair, and can both be?"}),
        code('''
Xc, yc, dc = make_arrays(mul_f, w=112, horizon=14)
model_c, best_c, _ = select_and_fit("convgru", Xc, yc, dc)
tec = dc >= CUT_END
Pc = predict_torch(model_c, Xc[tec])
conv = [mae_1e6(Pc[:, h], yc[tec, h]) for h in range(14)]

print(f"recipe chosen: {best_c}, window 112 days")
print(f"{'horizon':>8s}{'conv+GRU':>11s}{'GRU':>10s}")
for h in (0, 6, 13):
    print(f"{'t+' + str(h + 1):>8s}{conv[h]:>11,.0f}{ahead[h]:>10,.0f}")
'''),

        md("""
### Regime change, and the one case where the protocols disagree by a factor

A regime change is not a leak — it is a fact about the world. It is here because
it is the case where a random split cannot see the problem at all.
"""),
        prompt(
            label="what each protocol says when the world changes",
            input="the series extended through 2021, and the spring of 2020",
            output="a shuffled cross-validation over the whole span, and a forward fit scored on March to June 2020",
            constraint="score the naive forecast on the same 2020 rows — it needs no fitting, so it separates 'the model was wrong' from 'the days were unlike anything before them'",
            check="report the level as well as the error. A model can be wrong by 84,000 on days whose mean is 141,000, and that ratio is the whole story.",
            **{"try": "score the same forward fit on March to June 2019 "
                      "instead of 2020, changing nothing else. The MAE falls "
                      "by most of the gap. Neither the model nor the protocol "
                      "changed between those two runs."}),
        code('''
from sklearn.model_selection import cross_val_score

pool21 = df["rail"]["2016-01":"2021-11"]
v21 = pool21.values / 1e6
X21 = np.stack([v21[i:i + WINDOW] for i in range(len(v21) - WINDOW)])
y21 = v21[WINDOW:]
d21 = pool21.index[WINDOW:]

shuffled = -cross_val_score(
    LinearRegression(), X21, y21, scoring="neg_mean_absolute_error",
    cv=KFold(5, shuffle=True, random_state=RANDOM_STATE)).mean() * 1e6

tr21 = d21 < "2020-01-01"
te21 = (d21 >= "2020-03-01") & (d21 < "2020-07-01")
fit21 = LinearRegression().fit(X21[tr21], y21[tr21])

print(f"shuffled 5-fold over 2016-2021   MAE {shuffled:>10,.0f}")
print(f"forward fit, scored on 2020      MAE "
      f"{mae_1e6(fit21.predict(X21[te21]), y21[te21]):>10,.0f}")
print(f"copy last week, same 2020 rows   MAE "
      f"{mae_1e6(X21[te21, -7], y21[te21]):>10,.0f}")
print(f"the actual level on those days       {1e6 * y21[te21].mean():>10,.0f}")
print(f"\\nThe shuffled protocol reports a number close to the ladder's and")
print("sees nothing at all. Every fold of it contains 2020 in its training set.")
'''),

        md("""
**Read the train column beside the test column.** The deep RNN is worse on
*both* than the single-layer one, so this is not overfitting — it is a harder
optimisation problem that the recipe search did not solve. Depth is not free,
and nothing about the test score alone would have told you which of the two it
was.

The multivariate rows are the interesting ones: adding bus ridership and the
*next* day's day-type is worth more than any architectural change on this
series. The gates help, but the extra series helps more — and knowing which is
which is why every row here changed exactly one thing.
"""),

        md("""
## 7 · Where we are

- A recurrent cell replaces a fixed window with a state, and unrolling shows it
  is a deep network with shared weights.
- Backpropagation through time is Lecture 11's problem with one aggravation:
  the same matrix multiplies the gradient at every step, so its powers vanish
  or explode.
- A gate is a path along which the gradient is *added to* rather than
  multiplied. Everything else about LSTM and GRU is arrangement.
- Every model here was scored on the one forward hold-out of Lecture 15, so the rows can
  be read against one another.

**Before the next lecture:** run this notebook, then lengthen the window from
56 steps to 200 and re-run the plain RNN. It gets worse, not better, and the
gradient probe says why.
"""),
    ]
    return cells
