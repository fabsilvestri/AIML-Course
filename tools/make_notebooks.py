#!/usr/bin/env python3
"""
Build the Colab notebooks the slides link to.

    python3 tools/make_notebooks.py            # write notebooks/lecture-NN.ipynb
    python3 tools/make_notebooks.py --check    # ...then execute each one and fail
                                               #    on the first error

Why generated rather than hand-edited JSON: the same reason the figures are.
Twenty-four notebooks have to stay consistent with each other and with the
decks, and a diff of .ipynb JSON is unreadable, so drift would be invisible.

The notebooks are **ours**. No cell is taken from the textbook's own notebooks.
Each one follows TRICKS §10: it mirrors the lecture, carries the lecture's
worked assistant failure with code that actually runs, asserts after every
structural step, and states an expected wall-clock next to anything slow — for
an audience of mathematicians rather than engineers, "no output for four
minutes" otherwise reads as "it hung" and gets interrupted.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import nbformat as nbf
# _prompt lives in tools/notebooks/, beside the per-lecture modules. That
# directory goes on sys.path in _discover() so those modules can import their
# siblings — but this import runs at module load, long before _discover() is
# called, so it needs the path itself.
sys.path.insert(0, str(Path(__file__).resolve().parent / "notebooks"))
from _prompt import prompt                                # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "notebooks"

SEED = 42


# ------------------------------------------------------------------ helpers

def md(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(text.strip("\n"))


def code(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(text.strip("\n"))


def header(n: int, title: str, kind: str, chapters: str, thread: str = "") -> list:
    badge = "Build" if kind == "build" else "Fix"
    line = f"**Lecture {n} · {badge}** · Géron, {chapters}"
    if thread:
        line += f" · *Mathematical thread: {thread}*"
    return [md(f"""
# {title}

{line}

Applications of Machine Learning — BSc Mathematics of Artificial Intelligence

---

**How to use this notebook.** You are not expected to type the code. You are
expected to *read* it before you run it, and to be able to say what every line
does and what would break if it changed. Cells marked **⚠ read before running**
contain a defect on purpose.

Run the cells in order. Anything that takes more than a few seconds says so.
""")]


# The honesty note that every notebook's opening cell must carry. It is applied
# at BUILD time rather than written into each header, because the headers are
# produced three different ways — `header()` above, a `HEADER` constant in
# lectures 11-18 and 20-24, and an inline `md(...)` first cell in lectures 9 and
# 10. Patching three mechanisms means the fourth one somebody adds next month
# quietly ships without the note, and the note is the thing that keeps the
# prompt boxes honest.
PROMPT_NOTE = """
**About the prompt boxes.** Every code cell in this notebook is preceded by a
quoted prompt, and three lines follow it: what the prompt leaves open, the
version a student typically writes instead, and how you would catch a wrong
answer. Those three lines are the part worth reading twice.

The prompts here are **specifications, not transcripts** — this is what you
would have to ask for in order to get this cell, not a recording of somebody
asking for it. If your own prompt is vaguer than the box, expect worse code than
the cell below it.

*Lecture 19 is the one exception in this course.* It was built cell by cell
against Colab's Gemini 3.1 Pro, and its prompts are verbatim. It says so itself.
"""

# Lecture 19 makes the stronger claim in its own words and must not be given the
# weaker one as well.
NOTE_EXEMPT = {19}


def _ensure_prompt_note(nb: nbf.NotebookNode, n: int) -> None:
    """Append the honesty note to the opening markdown cell, once.

    Only where there is something to be honest ABOUT. The first version of this
    added the note unconditionally, so twenty-two notebooks with no prompt boxes
    at all shipped a header announcing prompt boxes — a claim in the one place a
    student reads before anything else, and false. That is the failure this
    course spends twenty-four lectures on, committed by its own build script, so
    the note is now gated on the boxes actually existing and worded to stay true
    at partial coverage.
    """
    if n in NOTE_EXEMPT or not nb.cells:
        return
    # The note now claims EVERY code cell has a box, so verify that before
    # writing it. The first version of this claimed boxes existed when none
    # did; the fix was to gate on their existing at all, and this is the same
    # fix again one step stronger, because the claim itself got stronger.
    for i, c in enumerate(nb.cells):
        if c.cell_type != "code":
            continue
        prev = nb.cells[i - 1] if i else None
        if not (prev is not None and prev.cell_type == "markdown"
                and prev.source.lstrip().startswith("> **Prompt")):
            raise RuntimeError(
                f"lecture {n}: the code cell at index {i} has no prompt box, "
                f"but the header note about to be added says every cell does. "
                f"Add the box or weaken the note — do not ship the claim.")
    first = nb.cells[0]
    if first.cell_type != "markdown":
        raise RuntimeError(f"lecture {n}: first cell is {first.cell_type}, "
                           f"expected the markdown header")
    if "specifications, not transcripts" in first.source:
        return
    first.source = first.source.rstrip() + "\n" + PROMPT_NOTE.rstrip() + "\n"


SETUP = code('''
# --- setup -------------------------------------------------------------------
# Not examinable: this is engineering hygiene, not machine learning. It is here
# because a version mismatch produces a confusing error twenty cells later.
import sys, sklearn, numpy as np, pandas as pd, matplotlib

print(f"python       {sys.version.split()[0]}")
print(f"scikit-learn {sklearn.__version__}")
print(f"numpy        {np.__version__}")
print(f"pandas       {pd.__version__}")

# root_mean_squared_error arrived in scikit-learn 1.4
assert tuple(int(p) for p in sklearn.__version__.split(".")[:2]) >= (1, 4), \\
    "This notebook needs scikit-learn >= 1.4.  In Colab: %pip install -U scikit-learn"

RANDOM_STATE = 42          # every split, every model, every shuffle
pd.set_option("display.width", 100)
''')


LOADER = code('''
# --- the data ----------------------------------------------------------------
# A function, not a manual download: the data will change, and you will need
# this on another machine.  ~5 s the first time, instant afterwards.
from pathlib import Path
import tarfile, urllib.request

def load_housing():
    tarball = Path("datasets/housing.tgz")
    if not tarball.is_file():
        Path("datasets").mkdir(parents=True, exist_ok=True)
        url = "https://github.com/ageron/data/raw/main/housing.tgz"
        urllib.request.urlretrieve(url, tarball)
        with tarfile.open(tarball) as t:
            t.extractall(path="datasets", filter="data")
    return pd.read_csv("datasets/housing/housing.csv")

housing_full = load_housing()

assert housing_full.shape == (20640, 10), f"unexpected shape {housing_full.shape}"
print(f"{len(housing_full):,} districts, {housing_full.shape[1]} columns")
housing_full.head()
''')


SETUP_PROMPT = prompt(
    label="setup",
    input="nothing",
    output="the version of every library this notebook depends on, and one "
           "seed",
    constraint="ASSERT the scikit-learn version rather than printing it — "
               "`root_mean_squared_error` arrived in 1.4, and on an older "
               "Colab image the failure is an ImportError twenty cells from "
               "here",
    left_open="that RANDOM_STATE is defined once and used for every split, "
              "every model and every shuffle. A notebook with three different "
              "seeds in it cannot be reproduced by reading it.",
    student="printing the versions and not checking them, so the notebook "
            "reports its own incompatibility as information rather than as an "
            "error.",
    catch="not examinable, and it is here because a version mismatch produces "
          "a confusing error in a cell that has nothing to do with versions.")

LOADER_PROMPT = prompt(
    label="the data",
    input="the California housing tarball",
    output="20,640 districts and 10 columns",
    constraint="a FUNCTION that downloads if absent and reads if present — the "
               "data will change, and you will need this on another machine",
    check="assert the shape, rather than trusting the download",
    left_open="what to do if the download is truncated. A short read gives a "
              "smaller frame and the assert catches it; anything subtler it "
              "will not.",
    student="downloading by hand and reading a path under ~/Downloads. It "
            "works on your machine and nowhere else, which you discover at the "
            "demo.",
    catch="delete `datasets/` and re-run. If the cell cannot rebuild its own "
          "input from nothing, it is not reproducible, it is cached.")


# ------------------------------------------------------------------ lecture 1

def lecture_01() -> nbf.NotebookNode:
    cells = header(
        1, "Welcome, and a price you can't trust", "build", "Chapters 1–2")

    cells += [
        md("## 1 · Setup"), SETUP_PROMPT, SETUP,
        md("## 2 · The data"), LOADER_PROMPT, LOADER,

        md("""
### What is in it

Ten attributes per district. One of them is not numeric, and one column has
holes in it. Find both before reading on.
"""),
prompt(
       label="what is in it",
       input="the loaded frame",
       output="every column, its type and its non-null count",
       constraint="`.info()`, not `.head()` — the two things worth finding here are a non-numeric column and a column with holes in it, and neither is visible in five rows",
       left_open="which column is which. Find both before reading on; the next cell names them.",
       student="`.head()` and an impression. A column that is 99% present looks complete in the first five rows, and the dtype of a mostly-numeric-looking column is not visible at all.",
       catch="non-null counts against the row count. That subtraction is the missing-value audit, and it is free."),
        code('''
housing_full.info()
'''),
prompt(
       label="count the holes, and the categories",
       input="the frame",
       output="how many districts are missing total_bedrooms, and the counts of every category level",
       constraint="print the missing count as a PERCENTAGE as well as a count — 207 sounds like a lot and 1% does not",
       left_open="that ISLAND has five districts in the whole of California. Remember it: it comes back in the next lecture and it does not announce itself when it breaks.",
       student="dropping the rows with missing bedrooms, which throws away 207 districts to avoid writing one imputer.",
       catch="`value_counts()` on every categorical, always. A level with n=5 is a level that will be absent from some cross-validation folds."),
        code('''
n_missing = housing_full["total_bedrooms"].isna().sum()
print(f"total_bedrooms is missing in {n_missing} districts "
      f"({100 * n_missing / len(housing_full):.1f}%)")
print()
print(housing_full["ocean_proximity"].value_counts())
'''),
        md("""
`ISLAND` has five districts in the whole of California. Remember that; it comes
back in the next lecture and it does not announce itself when it breaks.
"""),

        md("""
## 3 · Split before you look

This is the first rule and the easiest one to break. Everything you learn from
the data *before* the split leaks into the choices you make afterwards — through
you, not through the code. There is no library that prevents this.

We stratify on income because the experts told us income predicts price. A
random split gets the income mix wrong by up to 6.4%; stratifying gets it wrong
by 0.36%.
"""),
prompt(
       label="split before you look",
       input="the whole frame",
       output="a stratified 80/20 split",
       constraint="stratify on the INCOME BAND, because the experts said income predicts price — a random split gets the income mix wrong by up to 6.4% and stratifying gets it wrong by 0.36%",
       check="assert the two halves sum to the whole and that their indices are disjoint",
       left_open="why this is the first rule and the easiest to break. Everything you learn from the data BEFORE the split leaks into the choices you make afterwards — through you, not through the code. There is no library that prevents this.",
       student="exploring first and splitting later, because exploring is the interesting part. By then you have chosen which features to engineer using the test rows.",
       catch="the comment on the last line — from here to the final cell, `test_set` is not touched again. Write it down, in the code, where it will be read."),
        code('''
from sklearn.model_selection import train_test_split

income_cat = pd.cut(housing_full["median_income"],
                    bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                    labels=[1, 2, 3, 4, 5])

train_set, test_set = train_test_split(
    housing_full, test_size=0.2, random_state=RANDOM_STATE, stratify=income_cat)

# assert, do not hope
assert len(train_set) + len(test_set) == len(housing_full)
assert set(train_set.index).isdisjoint(test_set.index), "the split overlaps"
print(f"train {len(train_set):,}   test {len(test_set):,}")

# From here to the very last cell, `test_set` is not touched again.
housing = train_set.copy()
'''),

        md("""
## 4 · Look — at the training set only

Two things should jump out of the histograms. Take thirty seconds before you
scroll.
"""),
prompt(
       label="look — at the TRAINING set only",
       input="the training half",
       output="a histogram of every numeric column",
       constraint="`housing`, the training copy, not `housing_full` — the whole point of the previous cell was to make this cell safe",
       left_open="two things that should jump out. Take thirty seconds before scrolling: the income is not in dollars, and the TARGET is capped.",
       student="plotting the full frame out of habit. It is one word different and it undoes the split.",
       catch="50 bins, not the default 10. A cap at the top of a distribution is one bar, and at 10 bins it is inside a bar with everything else."),
        code('''
import matplotlib.pyplot as plt

housing.hist(bins=50, figsize=(12, 8))
plt.tight_layout(); plt.show()
'''),
        md("""
**The income is not in dollars** — it is scaled, and capped at 15.0001.

**The target is capped too**, and the target is our label. Count it rather than
squinting at it:
"""),
prompt(
       label="count the cap rather than squinting at it",
       input="the target column",
       output="how many districts sit at the cap, and the commonest values below it",
       constraint="count it — a histogram shows you a spike and a count tells you whether it is 5% of your labels or 0.5%",
       left_open="what the commonest values have in common. Every one of them is a multiple of $12,500 — artefacts of how the survey recorded prices, not facts about California.",
       student="noticing the cap and moving on. The target is the LABEL, so a cap on it means 5% of your training rows have a label that is not the answer, and no model can be right about them.",
       catch="`value_counts()` on a continuous target should be almost flat. Where it is not, the recording process is visible, and that is a fact about the survey rather than the world."),
        code('''
capped = (housing["median_house_value"] >= 500_000).sum()
print(f"{capped} districts sit at the cap "
      f"({100 * capped / len(housing):.1f}% of the training set)")

# which values do districts actually pile up on?
counts = housing["median_house_value"].value_counts()
print(f"\\na typical price is shared by {counts.median():.0f} districts")
print("\\nthe five commonest values below the cap:")
print(counts.drop(counts.index.max()).head(5))
'''),
        md("""
Every one of those is a multiple of **$12,500**. They are artefacts of how the
survey recorded prices, not facts about California.

A well-known description of this dataset names fainter lines at \\$450,000,
\\$350,000 and \\$280,000. Check that claim against the counts above before you
believe it — one of the three is real, one is marginal, and one is
indistinguishable from the background.
"""),
prompt(
       label="check the famous claim",
       input="three values named in a well-known description of this dataset",
       output="how many districts sit at each",
       constraint="check the claim against the counts you just computed rather than repeating it",
       left_open="the answer: one of the three is real, one is marginal, and one is indistinguishable from the background. The cell does not say which.",
       student="repeating 'there are also lines at 450,000, 350,000 and 280,000' because it is in the book. Two of the three do not survive a count.",
       catch="when a source names specific numbers about your data, the numbers are checkable. Three lines, and you either confirm it or you have found something."),
        code('''
for value in (450_000, 350_000, 280_000):
    print(f"${value:>9,}  {counts.get(value, 0):>4d} districts")
'''),

        md("""
## 5 · A number to compare against

Rule 2 of this course: *a metric with nothing to compare it to is decoration.*

So before building anything, measure the dumbest possible model — predict the
same number for every district. Everything you build today has to beat this, and
by how much is the only thing that will make your RMSE mean anything.
"""),
prompt(
       label="a number to compare against",
       input="the training mean",
       output="the RMSE of predicting it for every district, and what the human experts cost",
       constraint="compute the dumbest possible model BEFORE building anything — everything today has to beat it, and by how much is the only thing that will make your RMSE mean anything",
       left_open="that the expert figure is quoted, not measured. About 30% off on a typical $200,000 district, which the notebook converts to dollars so the two numbers are on one scale.",
       student="reporting an RMSE of $68,000 with nothing beside it. Is that good? The question is unanswerable without this cell.",
       catch="rule 2 of this course: a metric with nothing to compare it to is decoration. This is the cheapest possible comparison and it takes four lines."),
        code('''
from sklearn.metrics import root_mean_squared_error

y_train = housing["median_house_value"]
y_test  = test_set["median_house_value"]

baseline = np.full(len(y_test), y_train.mean())
baseline_rmse = root_mean_squared_error(y_test, baseline)
print(f"predict the training mean  ->  RMSE ${baseline_rmse:,.0f}")
print(f"the human experts are off by about 30%, i.e. roughly  ${0.30 * 200_000:,.0f}")
'''),

        md("""
## 6 · Commit

**Stop. On paper, now.** Not in this notebook — on paper, where you cannot
quietly revise it.

```
Metric:                                        ____________
Target RMSE for a good system:               $ ____________
RMSE I expect from the model I build today:  $ ____________
```

A prediction you can silently revise is not a prediction.
"""),

        md("""
## 7 · An assistant writes the preprocessing

Here is a real request and the code it returns. **⚠ Read before running.** It
runs, it imports nothing exotic, and it prints a believable number.

> *"Load the housing data, scale the features and split it into training and
> test sets."*
"""),
prompt(
       label="⚠ what the assistant returns",
       input="'load the housing data, scale the features and split it into training and test sets'",
       output="a fitted linear model and its RMSE",
       constraint="run it exactly as returned — it imports nothing exotic and prints a believable number",
       left_open="reviewer question 1. `fit_transform` ran on ALL the rows: the median that fills the missing values, and the mean and standard deviation that scale every column, were computed from a set that includes the rows we then call the test set.",
       student="this exact code. The prompt asked for scaling and splitting and did not say in which order, and one order is a leak.",
       catch="the model is evaluated on rows whose own values helped define the transformation applied to them. That sentence is the bug, and no part of the output says it."),
        code('''
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

X_all = housing_full.select_dtypes(include=[np.number]).drop(
    columns=["median_house_value"])
y_all = housing_full["median_house_value"]

prep = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
X_scaled = prep.fit_transform(X_all)          # <-- all 20,640 rows

X_tr, X_te, y_tr, y_te = train_test_split(
    X_scaled, y_all, test_size=0.2, random_state=RANDOM_STATE)

leaky = LinearRegression().fit(X_tr, y_tr)
print(f"RMSE ${root_mean_squared_error(y_te, leaky.predict(X_te)):,.0f}   looks fine")
'''),
        md("""
### Reviewer question 1: what touched the test set?

`fit_transform` ran on **all** the rows. The median that fills the missing
values, and the mean and standard deviation that scale every column, were all
computed from a set that includes the rows we then call the test set.

So the model is evaluated on rows whose own values helped define the
transformation applied to them.

**Now measure the damage** — do not guess:
"""),
prompt(
       label="measure the damage, do not guess",
       input="the same data, split first and preprocessed second",
       output="both RMSEs and the difference between them",
       constraint="change ONE thing — the order of the split and the fit — so the difference is attributable",
       left_open="that the answer is about a dollar. That is the finding, and the markdown below it is why it is still a bug.",
       student="assuming the leak must be large because it is called a leak, or assuming it must be small because the number came out small. Neither was knowable before this cell.",
       catch="three reasons it is small here, all nameable: centring and scaling is an invertible affine map, ordinary least squares is equivariant under one, and with 20,640 rows the training and test statistics nearly coincide. Remove any one and the leak has teeth."),
        code('''
# the same thing, done correctly: split first, fit the preprocessing on train
Xc_tr, Xc_te, yc_tr, yc_te = train_test_split(
    X_all, y_all, test_size=0.2, random_state=RANDOM_STATE)

prep_ok = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
honest = LinearRegression().fit(prep_ok.fit_transform(Xc_tr), yc_tr)
honest_rmse = root_mean_squared_error(yc_te, honest.predict(prep_ok.transform(Xc_te)))
leaky_rmse  = root_mean_squared_error(y_te, leaky.predict(X_te))

print(f"leaky   ${leaky_rmse:,.2f}")
print(f"correct ${honest_rmse:,.2f}")
print(f"the leak is worth ${abs(honest_rmse - leaky_rmse):,.2f}")
'''),
        md("""
### About a dollar. So why is it a bug?

Three reasons, and the third is the one that matters:

1. **You did not know it was a dollar until you measured.** Nothing in the code
   said so, and neither did the output.
2. **It is this small for three specific reasons** — centring and scaling is an
   invertible affine map, ordinary least squares is equivariant under one, and
   with 20,640 rows the training and test statistics nearly coincide. Remove any
   one of those and the leak has teeth.
3. **A leaked score and an honest score can be identical**, so you cannot detect
   it from the number. That is why the rule is procedural: *split first* — not
   because the damage is always large, but because you cannot tell whether it is.

You will meet the same error worth far more than a dollar in about an hour.
"""),

        md("""
## 8 · Build it properly

One `Pipeline`, so that cross-validation refits *all* of it on each fold and the
leak becomes structurally impossible rather than merely avoided.
"""),
prompt(
       label="build it properly",
       input="the training features",
       output="one ColumnTransformer handling numeric and categorical columns",
       constraint="one Pipeline, so cross-validation refits ALL of it on each fold and the leak becomes structurally impossible rather than merely avoided",
       check="assert the numeric and categorical column lists together account for every column — a column silently dropped here is a feature you never notice you are not using",
       left_open="`handle_unknown='ignore'` on the encoder. It is the right choice and it is also how ISLAND becomes an all-zero row with no warning, two sections into the next lecture.",
       student="listing the numeric columns by hand and forgetting one. The set assert is what catches it.",
       catch="the difference between avoided and impossible. A pipeline does not make you more careful; it removes the option."),
        code('''
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

X_train = housing.drop(columns=["median_house_value"])

num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = ["ocean_proximity"]

preprocessing = ColumnTransformer([
    ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
])

assert set(num_cols) | set(cat_cols) == set(X_train.columns), "a column was dropped"
print(f"{len(num_cols)} numeric + {len(cat_cols)} categorical")
'''),
prompt(
       label="⏱ 20 s — three models, scored on their own training data",
       input="the three model families",
       output="each one's RMSE on the rows it was fitted to",
       constraint="score on the TRAINING data, deliberately — this is the setup for the next lecture and not a result",
       left_open="that one of the three numbers is zero, and that two of the three are meaningless. The notebook does not say which.",
       student="reporting these. A tree with no depth limit puts every training row in its own leaf, and its zero is not a model that is perfect, it is a model that has memorised.",
       catch="write your best RMSE on paper next to what you predicted, and do not fix anything. Being wrong is the point and the diagnosis is the next ninety minutes."),
        code('''
# ~20 s: the forest is 100 trees on 16,512 rows.
models = {
    "Linear regression": LinearRegression(),
    "Decision tree":     DecisionTreeRegressor(random_state=RANDOM_STATE),
    "Random forest":     RandomForestRegressor(n_estimators=100,
                                               random_state=RANDOM_STATE, n_jobs=-1),
}

for name, model in models.items():
    pipe = Pipeline([("prep", preprocessing), ("model", model)]).fit(X_train, y_train)
    rmse = root_mean_squared_error(y_train, pipe.predict(X_train))
    print(f"{name:20s} RMSE on training data  ${rmse:>10,.0f}")
'''),

        md("""
## 9 · Where we are

Three numbers. One of them is zero.

Write your **best RMSE** on the same sheet of paper, next to what you predicted.
Bring it to the next lecture — we open by comparing them, and two of these
numbers are meaningless.

Do not fix anything yet. Being wrong is the point, and the diagnosis is the next
ninety minutes.
"""),
    ]
    return build(cells)


# ------------------------------------------------------------------ lecture 2

def lecture_02() -> nbf.NotebookNode:
    cells = header(
        2, "Your RMSE was a lie", "fix", "Chapters 2 & 4",
        thread="least squares and the normal equation")

    cells += [
        md("## 1 · Setup and where we left off"),
        SETUP_PROMPT, SETUP, LOADER_PROMPT, LOADER,
prompt(
       label="every import, and the same split",
       input="the same data and the same seed",
       output="the identical 16,512 / 4,128 split as the previous lecture",
       constraint="every import this notebook needs in ONE place, and the split rebuilt from the seed rather than inherited",
       check="assert the two sizes exactly — if they differ, every comparison against the previous lecture is void",
       left_open="that the stratification bins are repeated here verbatim. Change them and you get a different split from the same seed.",
       student="continuing in the previous notebook's kernel, where all of this already exists. It works in the room and nowhere else.",
       catch="a notebook that only runs because a previous one is still in memory is not reproducible. Restart-and-run-all is the only test of that."),
        code('''
# Every import this notebook needs, in one place — a notebook that only runs
# because a previous one is still in memory is not reproducible.
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import (GridSearchCV, KFold, cross_val_score,
                                     train_test_split)
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, add_dummy_feature
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt

income_cat = pd.cut(housing_full["median_income"],
                    bins=[0., 1.5, 3.0, 4.5, 6., np.inf], labels=[1, 2, 3, 4, 5])
train_set, test_set = train_test_split(
    housing_full, test_size=0.2, random_state=RANDOM_STATE, stratify=income_cat)

housing  = train_set.copy()
X_train  = housing.drop(columns=["median_house_value"])
y_train  = housing["median_house_value"]
X_test   = test_set.drop(columns=["median_house_value"])
y_test   = test_set["median_house_value"]

assert len(X_train) == 16512 and len(X_test) == 4128
print("same split as the previous lecture — the seed guarantees it")
'''),

        md("""
## 2 · Thread 1 — what `LinearRegression().fit()` actually computed

Minimising $\\lVert X\\theta - y\\rVert^2$ gives the normal equation

$$X^{\\mathsf T}(X\\hat\\theta - y) = 0$$

Read row by row, that says the residual is orthogonal to **every column of X**.
Least squares is not an algebraic trick — it is a projection onto the column
space of $X$. Verify it rather than believing it:
"""),
prompt(
       label="what LinearRegression().fit() actually computed",
       input="the numeric features with an intercept column added",
       output="the normal-equation solution, and the residual's inner product with every column of X",
       constraint="add the intercept column with `add_dummy_feature` BEFORE solving — without it the residual is not orthogonal to the constant and the assert fails for the wrong reason",
       check="assert the largest |Xᵀ(Xθ̂ − y)| is negligible RELATIVE to the scale of y — an absolute tolerance on dollars is meaningless",
       left_open="what the orthogonality means. Read row by row, Xᵀ(Xθ̂ − y) = 0 says the residual is orthogonal to EVERY COLUMN of X: least squares is not an algebraic trick, it is a projection onto the column space.",
       student="taking the normal equation on faith. It is four lines to verify and the verification is the thread.",
       catch="`np.linalg.inv` is not what scikit-learn uses. It computes the pseudoinverse via SVD, which still returns an answer when XᵀX is singular — more features than instances, or two collinear columns. That is the whole failure condition."),
        code('''
num = X_train.select_dtypes(include=[np.number])
X = make_pipeline(SimpleImputer(strategy="median"), StandardScaler()).fit_transform(num)
X_b = add_dummy_feature(X)                     # the x0 = 1 column, for the intercept
y = y_train.values

theta = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y
residual = X_b @ theta - y

# every column of X is orthogonal to the residual, to numerical precision
orth = X_b.T @ residual
print(f"largest |Xᵀ(Xθ̂ − y)| = {np.abs(orth).max():.3e}")
print(f"relative to the scale of y ({np.abs(y).mean():,.0f}): "
      f"{np.abs(orth).max() / np.abs(y).mean():.2e}")
assert np.abs(orth).max() / np.abs(y).mean() < 1e-6, "not orthogonal — check X"
'''),
        md("""
`np.linalg.inv` is not what scikit-learn uses. It computes the pseudoinverse via
SVD, which still returns an answer when $X^{\\mathsf T}X$ is singular — when you
have more features than instances, or when two columns are collinear. That is
the whole failure condition, and it is why you were warned against engineering a
feature as a weighted sum of existing ones.
"""),

        md("""
## 3 · Why the tree scored zero

Not because it is perfect. Because we asked it to grade its own homework.
"""),
prompt(
       label="why the tree scored zero",
       input="the same pipeline and the same training rows",
       output="its RMSE on the data it was fitted to",
       constraint="score it on the training rows again, and say in the output what that means",
       left_open="that it is not zero because the tree is perfect. It is zero because we asked it to grade its own homework, and an unconstrained tree can put every training row in its own leaf.",
       student="concluding the tree is the best model, or concluding it is broken. Neither: it is a correct answer to a question nobody should have asked.",
       catch="any model flexible enough to memorise will score perfectly on its own training rows. A zero training error is a statement about capacity, not about accuracy."),
        code('''
num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
preprocessing = ColumnTransformer([
    ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), ["ocean_proximity"]),
])

tree = Pipeline([("prep", preprocessing),
                 ("model", DecisionTreeRegressor(random_state=RANDOM_STATE))])
tree.fit(X_train, y_train)
print(f"RMSE on the data it was fitted to: "
      f"${root_mean_squared_error(y_train, tree.predict(X_train)):,.0f}")
print("An unconstrained tree can put every training row in its own leaf.")
'''),

        md("""
## 4 · Measure it honestly

`shuffle=True` is not decoration. The default `KFold` does **not** shuffle, so
two students whose dataframes are in different row orders get different folds
and cannot work out why their numbers disagree.

⏱ **about 90 seconds** — thirty fits in total, ten of them forests.
"""),
prompt(
       label="⏱ 90 s — measure it honestly",
       input="the three models, ten folds each",
       output="mean, standard deviation and range of the fold RMSEs",
       constraint="`shuffle=True` is NOT decoration — the default KFold does not shuffle, so two students whose dataframes are in different row orders get different folds and cannot work out why their numbers disagree",
       left_open="how to read the spread. The folds span several thousand dollars, so any comparison that turns on less than a couple of thousand is not a comparison.",
       student="reporting only the mean. A mean of $50,000 built from folds spanning $8,000 supports very different claims from one built from folds spanning $500.",
       catch="print the fold minimum and maximum beside the mean, every time. It is one f-string and it decides which differences you are allowed to talk about."),
        code('''
cv = KFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)

models = {
    "Linear regression": LinearRegression(),
    "Decision tree":     DecisionTreeRegressor(random_state=RANDOM_STATE),
    "Random forest":     RandomForestRegressor(n_estimators=100,
                                               random_state=RANDOM_STATE, n_jobs=-1),
}

results = {}
for name, model in models.items():
    pipe = Pipeline([("prep", preprocessing), ("model", model)])
    folds = -cross_val_score(pipe, X_train, y_train, cv=cv,
                             scoring="neg_root_mean_squared_error")
    results[name] = folds
    print(f"{name:20s} ${folds.mean():>9,.0f}  ± ${folds.std():>6,.0f}"
          f"   (folds ${folds.min():,.0f} – ${folds.max():,.0f})")
'''),
        md("""
Report the spread, not just the mean. The folds span several thousand dollars,
so **any comparison that turns on less than a couple of thousand is not a
comparison.**

Compare the models on the *same* folds rather than comparing two averages —
paired differences remove the fold-to-fold variation that both models share:
"""),
prompt(
       label="compare on the SAME folds",
       input="the two arrays of per-fold scores",
       output="the paired difference, and how many folds each model wins",
       constraint="subtract PER FOLD rather than comparing two averages — paired differences remove the fold-to-fold variation that both models share",
       left_open="why the paired standard deviation is so much smaller than either model's own. The folds differ in difficulty and both models feel it identically, so the difference cancels it.",
       student="comparing two means with their own standard deviations, concluding the intervals overlap, and declaring the comparison inconclusive. The paired version usually is not.",
       catch="report the win count. '10 of 10 folds' is an argument that a mean difference with a large standard deviation is not."),
        code('''
diff = results["Random forest"] - results["Linear regression"]
print(f"forest − linear, per fold:  mean ${diff.mean():,.0f}  sd ${diff.std():,.0f}")
print(f"folds where the forest wins: {(diff < 0).sum()}/10")
'''),

        md("""
## 5 · Tune — on validation folds, never on the test set

⏱ **2–4 minutes.** Fifteen combinations × five folds = 75 forest fits. The
lecture's figure uses `cv=10`, which takes twice as long; five is enough here.
"""),
prompt(
       label="⏱ 2-4 min — tune on validation folds, never on the test set",
       input="fifteen combinations, five folds each",
       output="the best parameters and the best cross-validated RMSE",
       constraint="the grid searches the WHOLE PIPELINE, so the preprocessing is refitted inside every fold — `model__` prefixes because the parameters belong to a step",
       check="detect whether the winner sits on the EDGE of the grid and say so — an optimum at the boundary means the optimum may lie outside it",
       left_open="that `cv=5` here where the deck uses 10. Five is enough to choose between these fifteen and it halves the wall clock; the choice is stated rather than silent.",
       student="reading `best_score_` as the model's accuracy. It is the score of the winner of a fifteen-way selection, measured on the folds that selected it, and it is optimistic by construction.",
       catch="the edge-of-grid check is four lines and it is the difference between a search and a shrug. If the largest value wins, the search was too small."),
        code('''
grid = {"model__max_features": [4, 6, 8, 10, 12],
        "model__n_estimators": [30, 100, 200]}

search = GridSearchCV(
    Pipeline([("prep", preprocessing),
              ("model", RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1))]),
    grid, cv=5, scoring="neg_root_mean_squared_error", n_jobs=-1)
search.fit(X_train, y_train)

print(f"best {search.best_params_}")
print(f"best cross-validated RMSE ${-search.best_score_:,.0f}")

best_n = search.best_params_["model__n_estimators"]
if best_n == max(grid["model__n_estimators"]):
    print("\\n⚠ the winner sits on the EDGE of the grid — the optimum may lie "
          "outside it. Search again with larger values.")
'''),

        md("""
## 6 · Look at what it gets wrong

Two things the previous lecture promised and never did. Both take one cell.
"""),
prompt(
       label="look at what it gets wrong",
       input="the tuned model's training predictions",
       output="RMSE broken down by ocean proximity and by income band",
       constraint="break the error down by GROUP — a single RMSE is an average over districts that are not alike",
       left_open="which group is about to matter. ISLAND — five districts in the whole state — is the category the first lecture warned you about.",
       student="reporting the headline RMSE and stopping. The stakeholder will ask 'is it worse anywhere in particular', and this is the cell that answers it.",
       catch="`observed=True` on the groupby. Without it pandas produces a row for every unobserved combination of categories, full of NaN, and the table becomes unreadable."),
        code('''
best = search.best_estimator_
pred_train_cv = best.predict(X_train)

err = pd.DataFrame({
    "actual": y_train,
    "predicted": pred_train_cv,
    "error": pred_train_cv - y_train,
    "ocean": housing["ocean_proximity"],
    "income_cat": pd.cut(housing["median_income"],
                         bins=[0., 1.5, 3.0, 4.5, 6., np.inf], labels=[1, 2, 3, 4, 5]),
})

print("RMSE by ocean_proximity:")
print(err.groupby("ocean", observed=True)["error"]
        .apply(lambda e: np.sqrt((e ** 2).mean())).sort_values(ascending=False)
        .apply(lambda v: f"${v:,.0f}"))

print("\\nRMSE by income category:")
print(err.groupby("income_cat", observed=True)["error"]
        .apply(lambda e: np.sqrt((e ** 2).mean()))
        .apply(lambda v: f"${v:,.0f}"))
'''),
        md("""
`ISLAND` — five districts in the whole state — is the category the first lecture
warned you about. With ten folds, some folds contain no ISLAND training row at
all, and `handle_unknown="ignore"` then encodes it as an all-zero column and
says nothing.
"""),
prompt(
       label="what an unseen category actually encodes to",
       input="an encoder fitted without ISLAND, asked to transform ISLAND",
       output="the resulting row, and its sum",
       constraint="DEMONSTRATE it rather than describing it — fit without the category and transform with it",
       left_open="the consequence for cross-validation. With ten folds, some folds contain no ISLAND training row at all, and `handle_unknown='ignore'` then encodes it as an all-zero column and says nothing.",
       student="setting `handle_unknown='ignore'` because the alternative raises, and never finding out what it does instead. It is the right choice and it is silent.",
       catch="the sum of the encoded row is zero. Every other district's row sums to one, so the model sees ISLAND as 'none of the above' — which is a prediction, and nobody made it deliberately."),
        code('''
enc = OneHotEncoder(handle_unknown="ignore").fit(
    housing[["ocean_proximity"]].query("ocean_proximity != 'ISLAND'"))
print("an unseen category encodes to:", enc.transform([["ISLAND"]]).toarray()[0])
print("sum:", enc.transform([["ISLAND"]]).toarray().sum(), "— and no warning")
'''),

        md("""
## 7 · The test set. Once.

Everything so far used only training data. This is the first and last time the
test set is touched.
"""),
prompt(
       label="the test set, once",
       input="the 4,128 held-out districts",
       output="the test RMSE with a 95% bootstrap interval, beside the cross-validated estimate",
       constraint="`method='percentile'` is NOT optional — scipy defaults to BCa, and we are claiming the percentile bootstrap. Name the estimator you mean",
       left_open="how to read the two numbers together. They agree within the interval, and a gap smaller than the interval is not evidence of anything.",
       student="tuning after seeing this number. If you adjust hyperparameters to improve it you are fitting the test set, and the improvement will not generalise.",
       catch="bootstrap the SQUARED ERRORS and take the square root of the interval, rather than bootstrapping the RMSE directly. The mean is what the bootstrap is good at; the square root comes after."),
        code('''
from scipy import stats

final_pred = best.predict(X_test)
final_rmse = root_mean_squared_error(y_test, final_pred)

squared = (final_pred - y_test.values) ** 2
# method= is not optional: scipy defaults to BCa, and we are claiming the
# PERCENTILE bootstrap. Name the estimator you mean.
lo, hi = np.sqrt(stats.bootstrap([squared], np.mean,
                                 confidence_level=0.95, method="percentile",
                                 random_state=RANDOM_STATE).confidence_interval)

print(f"test RMSE  ${final_rmse:,.0f}")
print(f"95% interval  ${lo:,.0f} – ${hi:,.0f}")
print(f"\\ncross-validated estimate was ${-search.best_score_:,.0f}")
print("The two agree within the interval. A gap smaller than the interval is "
      "not evidence of anything.")
'''),
        md("""
**Do not tune now.** If you adjust hyperparameters to improve that number you
are fitting the test set, and the improvement will not generalise. The number
you have is the number you report.
"""),

        md("""
## 8 · Red-team

Swap notebooks with the team beside you. Ten minutes. Five questions:

1. What touched the test set?
2. What was fitted, and on what? (`fit` and `transform` are different verbs)
3. What is the shape here?
4. What was dropped — rows, columns, NaNs? Count them.
5. What is the default I did not ask for?

Report what you **found**, not what you would have done differently.
"""),
    ]
    return build(cells)


# ------------------------------------------------------------------ driver

def build(cells: list) -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook(cells=cells)
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python",
                       "name": "python3"},
        "language_info": {"name": "python"},
        "colab": {"provenance": [], "toc_visible": True},
    }
    return nb


LECTURES = {1: lecture_01, 2: lecture_02}

# Lectures 3+ live in tools/notebooks/lecture_NN.py, each exporting build() ->
# list[cell]. One file per lecture so that authors working on different
# applications never touch the same file.
def _discover() -> None:
    import importlib.util
    # The lecture modules are loaded by FILE PATH, which leaves them with no
    # package — so `from ._prompt import prompt` cannot resolve. Putting their
    # own directory on sys.path lets them import siblings plainly instead.
    mod_dir = Path(__file__).parent / "notebooks"
    if str(mod_dir) not in sys.path:
        sys.path.insert(0, str(mod_dir))
    for path in sorted(mod_dir.glob("lecture_*.py")):
        n = int(path.stem.split("_")[1])
        if n in LECTURES:
            continue
        spec = importlib.util.spec_from_file_location(path.stem, path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        LECTURES[n] = (lambda m: lambda: build(m.build()))(mod)


_discover()


# Notebooks that are NOT generated from a module, and must not be overwritten.
#
# Lecture 19 was rebuilt cell by cell in Colab: each code cell was produced by
# prompting Colab's Gemini 3.1 Pro and keeping what came back, and each is
# preceded by the prompt that produced it plus three lines on what the prompt
# leaves open, what a student typically writes instead, and how you would catch
# a wrong answer. The shipped .ipynb therefore carries real generated code and
# real outputs from a real session — including the planted `shuffle=True`, which
# emerged from an under-specified prompt rather than being written in by hand.
#
# `tools/notebooks/lecture_19.py` is kept because it still documents the arc,
# but regenerating from it would silently replace all of the above with the
# hand-written version. So this script refuses, loudly, rather than quietly
# undoing a day's work the way `figures_app08.py` once undid the diagram fonts.
COLAB_AUTHORED = {19}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="execute each notebook and fail on the first error")
    ap.add_argument("--only", default="",
                    help="comma-separated lecture numbers, e.g. --only 19,20. "
                         "Executing all 24 takes hours, so a notebook under "
                         "development needs a way to be checked on its own.")
    args = ap.parse_args()
    wanted = {int(x) for x in args.only.split(",") if x.strip()}

    OUT.mkdir(parents=True, exist_ok=True)
    written = []
    for n, fn in sorted(LECTURES.items()):
        if wanted and n not in wanted:
            continue
        if n in COLAB_AUTHORED:
            print(f"  lecture-{n:02d}  SKIPPED — authored in Colab, not generated. "
                  f"See COLAB_AUTHORED in this file.")
            continue
        path = OUT / f"lecture-{n:02d}.ipynb"
        nb = fn()
        _ensure_prompt_note(nb, n)
        nbf.write(nb, path)
        cells = nbf.read(path, as_version=4).cells
        n_code = sum(c.cell_type == "code" for c in cells)
        # Compile every code cell before claiming the notebook was written. The
        # modules hold cell bodies as ''' literals, so a single
        # backslash-n in one becomes a REAL newline in the generated cell and
        # cuts an f-string in half — valid Python in the module, a SyntaxError
        # in the notebook, and invisible until someone runs it. Executing all 24
        # to find that out takes hours; compiling them takes milliseconds.
        for i, c in enumerate(cells):
            if c.cell_type != "code":
                continue
            src = c.source
            if any(line.lstrip().startswith(("!", "%")) for line in src.splitlines()):
                continue                      # shell and magic lines are not Python
            try:
                compile(src, f"{path.name}:cell{i}", "exec")
            except SyntaxError as exc:
                print(f"  {path.name}: cell {i} does not compile — "
                      f"{exc.msg} at line {exc.lineno}")
                print(f"    {(exc.text or '').strip()[:90]}")
                return 1
        print(f"  {path.relative_to(ROOT)}  {n_code} code cells")
        written.append(path)

    missing = [] if wanted else [n for n in range(1, 25) if n not in LECTURES]
    if missing:
        print(f"\nnot yet written: lectures {missing[0]}–{missing[-1]} "
              f"({len(missing)} of 24)")

    if args.check:
        from nbclient import NotebookClient
        print("\nexecuting:")
        for path in written:
            nb = nbf.read(path, as_version=4)
            client = NotebookClient(nb, timeout=1800, kernel_name="python3",
                                    resources={"metadata": {"path": str(OUT)}})
            try:
                client.execute()
            except Exception as exc:                      # noqa: BLE001
                print(f"  {path.name}: FAILED — {type(exc).__name__}: "
                      f"{str(exc)[:300]}")
                return 1
            print(f"  {path.name}: ran clean")
    return 0


if __name__ == "__main__":
    sys.exit(main())
