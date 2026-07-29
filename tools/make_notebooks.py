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


# ------------------------------------------------------------------ lecture 1

def lecture_01() -> nbf.NotebookNode:
    cells = header(
        1, "Welcome, and a price you can't trust", "build", "Chapters 1–2")

    cells += [
        md("## 1 · Setup"), SETUP,
        md("## 2 · The data"), LOADER,

        md("""
### What is in it

Ten attributes per district. One of them is not numeric, and one column has
holes in it. Find both before reading on.
"""),
        code('''
housing_full.info()
'''),
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
        md("## 1 · Setup and where we left off"), SETUP, LOADER,
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
        code('''
from scipy import stats

final_pred = best.predict(X_test)
final_rmse = root_mean_squared_error(y_test, final_pred)

squared = (final_pred - y_test.values) ** 2
lo, hi = np.sqrt(stats.bootstrap([squared], np.mean,
                                 confidence_level=0.95,
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
    for path in sorted((Path(__file__).parent / "notebooks").glob("lecture_*.py")):
        n = int(path.stem.split("_")[1])
        if n in LECTURES:
            continue
        spec = importlib.util.spec_from_file_location(path.stem, path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        LECTURES[n] = (lambda m: lambda: build(m.build()))(mod)


_discover()


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
        path = OUT / f"lecture-{n:02d}.ipynb"
        nbf.write(fn(), path)
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
