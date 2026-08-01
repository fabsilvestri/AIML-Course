#!/usr/bin/env python3
"""
Lecture 6 — Reading a learning curve. Fix, Titanic, Géron Chapter 4.

Exports build() -> list[cell]; tools/make_notebooks.py wraps it.

Structure mirrors the deck: thread -> diagnosis -> repair -> re-measure ->
red team. It rebuilds the previous lecture's degree sweep from scratch,
imports included: a notebook that only runs because another notebook is still
in memory is not reproducible.

Three places where the notebook deliberately runs a smaller experiment than the
deck, each stated in the cell that does it:

  * the decomposition is measured at degrees 1, 2, 3 and 5 rather than all six
    -- 4 and 6 are the two most expensive and neither carries the argument;
  * the penalty sweep stops at C = 0.316 rather than 1e4, because the
    coordinate descent takes 163 s for a single weak-penalty value at degree 5
    and the minimum is well inside the range anyway;
  * the tuning trap uses 8 seeds rather than 20.

Everything else reproduces the slide numbers exactly.

`n_jobs` is capped rather than set to -1 throughout. A free Colab CPU runtime
has two cores, so -1 buys nothing there, and on a shared machine it turns a
two-minute cell into a twenty-minute one by oversubscribing the box. `NJ` is
defined in the notebook's own setup cell, not here -- it has to exist in the
kernel, where the cells run.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from make_notebooks import code, header, md, SETUP        # noqa: E402
from _prompt import prompt                                # noqa: E402


def build() -> list:
    cells = header(
        6, "Reading a learning curve", "fix", "Chapter 4",
        thread="the bias-variance decomposition")

    # ---------------------------------------------------------------- setup
    cells += [
        md("## 1 · Setup and where we left off"), SETUP,
        prompt(
            label="the imports, again, in full",
            input="nothing",
            output="every import this notebook needs",
            constraint="repeat them — do NOT rely on the previous notebook's kernel",
            left_open="that this looks like duplication and is not. Two notebooks that share a kernel are one notebook with a confusing file layout.",
            student="opening this straight after the build session, running from section 2, and having everything work. It then fails for the person you send it to, who did not run the other one first.",
            catch="Restart-and-run-all. If a notebook cannot start from a cold kernel it is not reproducible, however many times it has worked for you."),
        code('''
# Every import this notebook needs, in one place. A notebook that only runs
# because a previous one is still in memory is not reproducible.
import tarfile, urllib.request, warnings
from pathlib import Path

import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from sklearn.model_selection import (GridSearchCV, StratifiedKFold,
                                     cross_val_score, cross_validate,
                                     learning_curve, train_test_split)
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import (OneHotEncoder, PolynomialFeatures,
                                   StandardScaler)

# Not examinable. A free Colab CPU runtime has two cores, so n_jobs=-1 buys
# nothing there, and on a shared machine it makes everything slower by
# oversubscribing. Raise it if you have cores to spare.
NJ = 4
'''),
        prompt(
            label="the data, rebuilt from scratch",
            input="the same tarball",
            output="891 rows, with the shape and the label domain asserted",
            constraint="download-if-absent again, so this notebook stands alone",
            check="assert the shape and that Survived holds only 0 and 1",
            left_open="that this is byte-identical to the build session's loader. It is deliberate: the repair has to start from exactly the state the build session ended in, or the comparison means nothing.",
            student="pickling the frame at the end of the last notebook and loading it here. It works until the pickle is stale, and a stale pickle looks exactly like a fresh one.",
            catch="if two notebooks must agree on a dataset, they should agree by rebuilding it, not by passing it."),
        code('''
# --- the data ----------------------------------------------------------------
# A function, not a manual download. ~5 s the first time, instant afterwards.
def load_titanic():
    tarball = Path("datasets/titanic.tgz")
    if not tarball.is_file():
        Path("datasets").mkdir(parents=True, exist_ok=True)
        url = "https://github.com/ageron/data/raw/main/titanic.tgz"
        urllib.request.urlretrieve(url, tarball)
        with tarfile.open(tarball) as t:
            t.extractall(path="datasets", filter="data")
    return pd.read_csv("datasets/titanic/train.csv")

full = load_titanic()

assert full.shape == (891, 12), f"unexpected shape {full.shape}"
assert full["Survived"].isin([0, 1]).all()
print(f"{len(full)} passengers, {full['Survived'].sum()} of whom survived")
'''),
        prompt(
            label="the same four engineered columns",
            input="the raw frame",
            output="Title, FamilySize, IsAlone, Deck — unchanged from the build session",
            constraint="keep FamilySize even though we know it is the fault, so the diagnosis in section 3 has something to find",
            check="assert the dependence is still exactly zero, and that the title set is the expected five",
            left_open="that the assert on line 3 is not a health check, it is a load-bearing part of the argument four sections from now. The comment above it says so.",
            student="quietly fixing the bug here, because they remember it from last time. The notebook then diagnoses a fault that is no longer present and every downstream number changes.",
            catch="when a notebook exists to repair something, resist repairing it in the setup. Reproduce the broken state first, deliberately."),
        code('''
# --- the four engineered columns, exactly as in the build session ------------
def engineer(d):
    d = d.copy()
    d["Title"] = (d["Name"].str.extract(r",\\s*([^\\.]+)\\.", expand=False)
                  .str.strip()
                  .replace({"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"}))
    d["Title"] = d["Title"].where(
        d["Title"].isin(["Mr", "Mrs", "Miss", "Master"]), "Rare")
    d["FamilySize"] = d["SibSp"] + d["Parch"] + 1
    d["IsAlone"] = (d["FamilySize"] == 1).astype(int)
    d["Deck"] = d["Cabin"].str[0].fillna("U")
    return d

full = engineer(full)

# Keep this line in view. It is the third diagnosis, four sections from now.
assert (full["SibSp"] + full["Parch"] + 1 - full["FamilySize"]).abs().max() == 0
assert set(full["Title"]) == {"Mr", "Mrs", "Miss", "Master", "Rare"}
print(full["Title"].value_counts().to_dict())
'''),
        prompt(
            label="split first, same seed, same 712 rows",
            input="the engineered frame",
            output="the same 712/179 stratified split as the build session",
            constraint="FamilySize stays in the FRAME (in ALL) but out of the numeric block — section 3 needs the column to demonstrate the dependence, and the pipeline needs it excluded",
            check="assert the sizes, the disjointness, and the identical column sets",
            left_open="the difference between a column existing and a column being modelled. ALL and NUM differ by exactly one name, and that is the whole of the third repair.",
            student="dropping FamilySize from the frame entirely, which makes the dependence undemonstrable, or leaving it in NUM, which reproduces the singular design matrix.",
            catch="the same seed and the same stratification as the notebook you are comparing against. A repair measured on a different split is not a measured repair."),
        code('''
# --- split first, stratified on the label ------------------------------------
NUM = ["Age", "Fare", "SibSp", "Parch"]     # FamilySize is left out on purpose
CAT = ["Pclass", "Sex", "Embarked", "Title", "Deck"]
BIN = ["IsAlone"]
ALL = ["Age", "Fare", "SibSp", "Parch", "FamilySize"] + CAT + BIN

X = full[ALL]
y = full["Survived"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)

assert len(X_train) == 712 and len(X_test) == 179
assert len(X_train) + len(X_test) == len(full)
assert set(X_train.index).isdisjoint(X_test.index)
assert set(X_train.columns) == set(X_test.columns)
print(f"train rate {y_train.mean():.4f}   test rate {y_test.mean():.4f}")
'''),
        prompt(
            label="one pipeline factory, and the anchor",
            input="a degree and a penalty setting",
            output="a fresh unfitted pipeline, plus the 0.666 anchor recomputed",
            constraint="a FUNCTION returning a new pipeline each call — a single shared object refitted in a loop carries the previous iteration's state into the next",
            check="assert the anchor is still 0.666, so this notebook and the build session are provably on the same footing",
            left_open="that `C=1e6` is the default here. That switches the penalty off, which is the state we are about to repair, and it is stated in the signature rather than hidden.",
            student="building `m = pipeline()` once at the top and reusing it everywhere. Scikit-learn refits in place, so the tenth cross-validation is not measuring what the first one measured.",
            catch="recompute the anchor rather than copying 0.666 across from the other notebook. If it comes out different, the two notebooks are not on the same data and everything below is incomparable."),
        code('''
# --- the pipeline, and the anchor --------------------------------------------
def prep(degree=1):
    num = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                        PolynomialFeatures(degree=degree, include_bias=False))
    cat = make_pipeline(SimpleImputer(strategy="most_frequent"),
                        OneHotEncoder(drop="first", handle_unknown="infrequent_if_exist",
                                        min_frequency=2))
    return ColumnTransformer([("num", num, NUM), ("cat", cat, CAT),
                              ("bin", "passthrough", BIN)])

def pipeline(degree=1, C=1e6, penalty="l2", solver="lbfgs", l1_ratio=None,
             max_iter=4000):
    return Pipeline([
        ("prep", prep(degree)),
        ("clf", LogisticRegression(C=C, penalty=penalty, solver=solver,
                                   l1_ratio=l1_ratio, max_iter=max_iter,
                                   random_state=RANDOM_STATE))])

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)

p = y_train.mean()
constant_log_loss = -(p * np.log(p) + (1 - p) * np.log(1 - p))
print(f"anchor — report the base rate to everyone: {constant_log_loss:.3f}")
assert abs(constant_log_loss - 0.666) < 0.001
'''),

        md("""
### The degree sweep, rebuilt

`return_train_score=True` is not on by default and it is the whole experiment.
One curve tells you nothing.

⏱ **about 15 seconds.** The convergence warnings at degree 4 and above are
suppressed here because we count them deliberately later — they are the second
diagnosis, not noise.
"""),
        prompt(
            label="⏱ 15 s — the degree sweep, rebuilt",
            input="degrees 1 to 6",
            output="columns, training log loss and held-out log loss at each degree",
            constraint="`return_train_score=True` — it is off by default and it is the whole experiment",
            check="assert the ARGUMENT, not three decimals of it: that a low degree wins on held-out score, and that degree 5 is far worse than the anchor",
            left_open="why the asserts are loose. Pinning the exact number makes the cell fail whenever the pipeline legitimately changes — which it just did, when the encoder was repaired.",
            student="`assert sweep[5]['valid'] == 1.957`, which is true today and turns every future improvement into a red cell. An assert should encode the claim, not the current output.",
            catch="ask what your assert would do if someone improved the code. If it would break, it is pinning a number rather than checking a fact."),
        code('''
DEGREES = [1, 2, 3, 4, 5, 6]
sweep = {}

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    for d in DEGREES:
        r = cross_validate(pipeline(degree=d), X_train, y_train, cv=cv,
                           scoring="neg_log_loss", return_train_score=True,
                           n_jobs=NJ)
        n_cols = prep(d).fit(X_train).transform(X_train).shape[1]
        sweep[d] = dict(cols=n_cols,
                        train=float(-r["train_score"].mean()),
                        valid=float(-r["test_score"].mean()))

for d in DEGREES:
    s = sweep[d]
    print(f"degree {d}  {s['cols']:4d} cols   train {s['train']:.3f}   "
          f"held-out {s['valid']:.3f}")

assert sweep[1]["cols"] == 22 and sweep[5]["cols"] == 143
# Assert the argument, not three decimals of it: a low degree is the best
# held-out score, and degree 5 is far worse than the constant anchor. Pinning
# the exact number makes the cell fail whenever the pipeline legitimately
# changes -- which it just did, when the encoder was repaired.
assert min(sweep, key=lambda d: sweep[d]["valid"]) in (1, 2), \
    "a low degree should still win on held-out log loss"
assert sweep[5]["valid"] > 2 * constant_log_loss, \
    "degree 5 should still be far worse than saying nothing"
'''),

        # ------------------------------------------------------------ thread
        md("""
## 2 · Thread 3 — the bias-variance decomposition

Fix one passenger $x$. Two things are random: the **training set** $D$ you
happened to draw, and the **label** $y$ of that passenger. Write
$\\hat{p}_D(x)$ for what your model predicts,
$\\bar{p}(x) = \\mathbb{E}_D[\\hat{p}_D(x)]$ for the average prediction over
training sets, and $p^{*}(x)$ for the truth.

Add and subtract $\\bar{p}$, and the cross term dies because $\\bar{p}$ is
*defined* as $\\mathbb{E}_D[\\hat{p}_D]$:

$$\\mathbb{E}_{D}\\left[(y - \\hat{p}_D)^{2}\\right]
  = (y - \\bar{p})^{2} + \\mathbb{E}_{D}\\left[(\\hat{p}_D - \\bar{p})^{2}\\right]$$

Do it again on the first term, this time inserting $p^{*}$, and average over
$y$ as well. $\\mathbb{E}_y[y] = p^{*}$ kills the new cross term and
$\\operatorname{Var}(y) = p^{*}(1-p^{*})$ because $y$ is Bernoulli:

$$\\mathbb{E}\\left[(y - \\hat{p}_D)^{2}\\right]
  = \\underbrace{(p^{*} - \\bar{p})^{2}}_{\\text{squared bias}}
  + \\underbrace{\\mathbb{E}_{D}\\left[(\\hat{p}_D - \\bar{p})^{2}\\right]}_{\\text{variance}}
  + \\underbrace{p^{*}(1 - p^{*})}_{\\text{noise}}$$

**One caveat, stated before any number appears.** That identity is exact for
*squared error*. Log loss does not decompose this way, so everything measured
in this section is a **Brier** score. The shape of the answer transfers; the
numbers are Brier numbers.

### The third term first — the one you cannot fix

$p^{*}$ is never observed. But where several passengers share *exactly* the
same recorded values, every model of those columns must give them the same
number, so whatever their outcomes do inside that cell is a floor.
"""),
        prompt(
            label="the term you cannot fix, measured",
            input="passengers grouped by five banded columns",
            output="the noise floor, as a Brier score",
            constraint="use the UNBIASED estimator k(m−k)/(m(m−1)), not the plain variance — with cells of size 2 the biased version understates the floor by half",
            check="assert 133 cells, 102 with two or more passengers, and a floor of 0.121",
            left_open="that this is a lower bound on the achievable error and not the true noise. Passengers sharing these five banded columns may still differ in ways the columns do not record.",
            student="estimating noise as the residual of their best model, which measures the model rather than the problem. This estimate uses no model at all — where inputs are identical, every model must agree.",
            catch="a floor computed without fitting anything is the only one you can compare a fitted model against. If your 'irreducible error' fell when you improved the model, it was never irreducible."),
        code('''
d = full.copy()
d["AgeBand"] = pd.cut(d["Age"], [0, 12, 25, 40, 60, 100],
                      labels=["0-12", "13-25", "26-40", "41-60", "60+"])
d["AgeBand"] = d["AgeBand"].cat.add_categories(["missing"]).fillna("missing")
d["FamBand"] = pd.cut(d["FamilySize"], [0, 1, 4, 20],
                      labels=["alone", "2-4", "5+"])

keys  = ["Sex", "Pclass", "AgeBand", "FamBand", "Embarked"]
cells = d.groupby(keys, observed=True)["Survived"].agg(["sum", "count"])
cells = cells.rename(columns={"sum": "k", "count": "m"})

# k(m-k) / (m(m-1)) is the unbiased estimator of p(1-p) from m Bernoulli draws
multi    = cells[cells["m"] >= 2]
unbiased = multi["k"] * (multi["m"] - multi["k"]) / (multi["m"] * (multi["m"] - 1))
noise    = float((unbiased * multi["m"]).sum() / multi["m"].sum())

mixed = multi[(multi["k"] > 0) & (multi["k"] < multi["m"])]

assert len(cells) == 133 and len(multi) == 102
assert int(multi["m"].sum()) == 858
print(f"{len(cells)} cells; {len(multi)} hold two or more passengers "
      f"({int(multi['m'].sum())} people)")
print(f"{len(mixed)} of those cells are mixed, covering "
      f"{int(mixed['m'].sum())} passengers")
print(f"\\nmeasured noise floor (Brier): {noise:.3f}")
assert abs(noise - 0.121) < 0.001
'''),
        prompt(
            label="look at the cells that disagree",
            input="the mixed cells",
            output="the four where identical inputs disagree most loudly",
            constraint="rank by k(m−k), the count of disagreeing pairs — not by cell size, which would just find the biggest groups",
            left_open="what to conclude. Nothing here is fixable by modelling: these passengers are identical in every recorded column and some lived and some died.",
            student="treating the noise floor as an abstraction. Printing four actual groups of people is what makes it a fact about the ship rather than a term in an equation.",
            catch="when a bound looks suspiciously convenient, print the rows it came from. A floor you can read is a floor you can defend."),
        code('''
# The four cells where identical inputs disagree most loudly.
worst = mixed.assign(mix=lambda t: t["k"] * (t["m"] - t["k"])).nlargest(4, "mix")
for key, row in worst.iterrows():
    print(f"{' · '.join(str(v) for v in key):42s}  "
          f"{int(row['m'] - row['k']):2d} died, {int(row['k']):2d} survived")

print("\\nNo model of these five columns can separate the people inside a row.")
'''),

        md("""
### The other two terms, on your own pipeline

200 training sets of 400 rows each, drawn without replacement from the 712 you
hold. Every fit uses the pipeline above, unchanged.

The deck decomposes all six degrees; we do 1, 2, 3 and 5. Degrees 4 and 6 are
the two most expensive fits in the sweep and neither carries the argument —
degree 1 is the low-variance end and degree 5 is the model the build session
committed to.

⏱ **about two minutes.** That is the cell working, not the cell hanging.
"""),
        prompt(
            label="⏱ 2 min — the other two terms",
            input="200 training sets of 400 rows, drawn without replacement",
            output="variance, bias²+noise and total expected Brier at degrees 1, 2, 3, 5",
            constraint="draw WITHOUT replacement — a bootstrap with replacement gives each fit ~253 distinct rows, and the variance you measure is then partly the variance of the resampling scheme",
            check="assert the prediction matrix is exactly (200, 179) before anything is averaged",
            left_open="why four degrees and not six. Degrees 4 and 6 are the two most expensive fits and neither carries the argument; the docstring says so rather than the reader guessing.",
            student="reusing one fitted pipeline across the 200 draws, or fitting on X_train and predicting on X_train. Both give a variance term near zero, which is the answer you wanted and not the one that is true.",
            catch="the shape assert. A silently broadcast array here produces a beautifully wrong stackplot and no error."),
        code('''
def boot_fit(deg, sel):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = pipeline(degree=deg).fit(X_train.iloc[sel], y_train.iloc[sel])
    return m.predict_proba(X_test)[:, 1]

BV_DEGREES = [1, 2, 3, 5]
N_BOOT, BOOT_N = 200, 400
rng   = np.random.default_rng(RANDOM_STATE)
draws = [rng.choice(len(X_train), size=BOOT_N, replace=False)
         for _ in range(N_BOOT)]

yv, bv = y_test.values.astype(float), {}
for deg in BV_DEGREES:
    P = np.array(Parallel(n_jobs=NJ)(
        delayed(boot_fit)(deg, sel) for sel in draws))
    assert P.shape == (N_BOOT, len(X_test))
    pbar = P.mean(axis=0)
    bv[deg] = dict(variance=float(((P - pbar) ** 2).mean(axis=0).mean()),
                   bias2_noise=float(((yv - pbar) ** 2).mean()),
                   total=float(((yv - P) ** 2).mean(axis=0).mean()))
    print(f"degree {deg}: total {bv[deg]['total']:.3f} = "
          f"bias²+noise {bv[deg]['bias2_noise']:.3f} + "
          f"variance {bv[deg]['variance']:.3f}")
'''),
        md("""
**Check the identity before believing the plot.** The three columns are not a
*model* of the error — they are the error, rearranged. If the residual is not
at floating-point noise you have a bug, and the picture will still look
plausible.
"""),
        prompt(
            label="check the identity before believing the plot",
            input="the three measured terms",
            output="the residual of total − variance − (bias²+noise), per degree",
            constraint="assert it is below 1e-12 — these three columns are not a MODEL of the error, they are the error rearranged, so anything above floating-point noise is a bug",
            check="also assert the variance grows by more than 10x from degree 1 to 5",
            left_open="that the decomposition is exact for squared error and not for log loss. Everything in this section is therefore a Brier number, stated before the first number appeared.",
            student="plotting the three terms and interpreting them without checking they sum. A broken decomposition still produces a plausible stacked area chart, which is the problem.",
            catch="whenever you decompose a quantity, assert the parts add up. It is one line and it is the difference between a measurement and an illustration."),
        code('''
for deg in BV_DEGREES:
    resid = abs(bv[deg]["total"] - bv[deg]["variance"] - bv[deg]["bias2_noise"])
    assert resid < 1e-12, f"degree {deg}: identity residual {resid:g}"
print("identity holds at machine precision for every degree")

growth = bv[5]["variance"] / bv[1]["variance"]
print(f"\\nvariance is multiplied by {growth:.1f} from degree 1 to degree 5")
print(f"bias² + noise at degree 1: {bv[1]['bias2_noise']:.3f}, "
      f"against a measured noise floor of {noise:.3f}")
print("=> the squared bias of the degree-1 model is under 0.01. There was "
      "never much bias to buy back.")
assert growth > 10
'''),
        prompt(
            label="the decomposition, stacked",
            input="the three terms at four degrees",
            output="a stackplot with the total overlaid and the measured floor drawn in",
            constraint="draw the noise floor as a horizontal line — the point of the picture is how much of the bottom band is unreachable",
            left_open="that the bias band barely moves. The squared bias at degree 1 is under 0.01, so almost the whole bottom band is the floor, and there was never much bias for extra capacity to buy back.",
            student="reading the growing total as 'the model got worse' without seeing WHICH term grew. The stack is what turns that into a diagnosis.",
            catch="if a stacked plot has a term you cannot separately measure, it is a diagram, not data. Every band here was measured independently and checked to sum."),
        code('''
fig, ax = plt.subplots(figsize=(7.5, 3.6))
ax.stackplot(BV_DEGREES,
             [bv[k]["bias2_noise"] for k in BV_DEGREES],
             [bv[k]["variance"] for k in BV_DEGREES],
             labels=["bias² + noise", "variance"], colors=["#9fb8ca", "#e6b0a8"])
ax.plot(BV_DEGREES, [bv[k]["total"] for k in BV_DEGREES], "o-", color="#16212b",
        label="total (expected Brier)")
ax.axhline(noise, ls="--", color="#6c3483")
ax.text(5, noise - 0.006, f"measured noise floor {noise:.3f}", ha="right",
        va="top", color="#6c3483")
ax.set_xlabel("polynomial degree")
ax.set_ylabel("expected squared error")
ax.legend(loc="upper left")
plt.show()
'''),

        md("""
### Putting it back on the two curves

The training curve is measured on the rows that were fitted, so it sees no
variance term at all. The held-out curve is the sum of all three. **The
vertical gap between them is the variance term.**
"""),
        prompt(
            label="put it back on the two curves",
            input="the degree sweep",
            output="held-out minus training log loss, at degree 1 and degree 5",
            constraint="report the GAP, not the two numbers separately",
            check="assert the gap at degree 5 is more than ten times the gap at degree 1",
            left_open="why the gap is the variance term. The training curve is measured on the rows that were fitted, so it carries no variance at all; the held-out curve carries all three terms.",
            student="looking at the two curves and saying 'it overfits' without quantifying. The factor of ten is what connects the picture back to the decomposition you just measured.",
            catch="the vertical gap between a training and a held-out curve IS the variance term. That is the sentence to remember from this lecture."),
        code('''
for deg in (1, 5):
    gap = sweep[deg]["valid"] - sweep[deg]["train"]
    print(f"degree {deg}: held-out − training = {gap:.3f}")

g1 = sweep[1]["valid"] - sweep[1]["train"]
g5 = sweep[5]["valid"] - sweep[5]["train"]
print(f"\\nthe gap grew by a factor of {g5 / g1:.0f} while the model family "
      f"stayed the same size in every other respect")
assert g5 > 10 * g1, \
    "the gap at degree 5 should dwarf the gap at degree 1"
'''),

        md("""
### Learning curves — the same two curves against a different knob

Now the x-axis is the number of **training rows**. That answers a question the
degree sweep cannot: *would more data help?*

Note `shuffle=True`. Without it the sub-samples are the first $n$ rows of the
frame, in whatever order your frame happens to be in.

⏱ **about 30 seconds.**
"""),
        prompt(
            label="⏱ 30 s — would more data help?",
            input="training subsets from 12% to 100% of the 712 rows",
            output="training and held-out log loss against the number of rows, at degrees 1 and 5",
            constraint="`shuffle=True` — without it the sub-samples are the first n rows of the frame, in whatever order the frame happens to be in",
            check="assert eight sizes and that the largest is 640",
            left_open="that this answers a question the degree sweep cannot. The degree sweep varies capacity; this varies data, and only one of those is something you can go and buy.",
            student="forgetting shuffle and getting a learning curve of a sorted prefix. The curve is smooth, the shape is wrong, and nothing warns you.",
            catch="check what the first point is measured on. At 12% of 712 that is 85 passengers, and a held-out log loss from 85 rows is noisy enough to read as a trend."),
        code('''
lc = {}
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    for deg in (1, 5):
        n, tr, va = learning_curve(
            pipeline(degree=deg), X_train, y_train, cv=cv,
            train_sizes=np.linspace(0.12, 1.0, 8), scoring="neg_log_loss",
            n_jobs=NJ, shuffle=True, random_state=RANDOM_STATE)
        lc[deg] = dict(n=n, train=-tr.mean(axis=1), valid=-va.mean(axis=1))
        assert len(n) == 8 and n[-1] == 640

for deg in (1, 5):
    c = lc[deg]
    print(f"degree {deg}: at {c['n'][0]} rows held-out {c['valid'][0]:6.3f}; "
          f"at {c['n'][-1]} rows held-out {c['valid'][-1]:.3f}, "
          f"gap {c['valid'][-1] - c['train'][-1]:.3f}")
'''),
        prompt(
            label="two learning curves, one y-axis",
            input="the learning curves at degrees 1 and 5",
            output="both panels, sharing a y-axis",
            constraint="`sharey=True` — the panels are being compared, and independent y-axes would make a gap of 1.6 and a gap of 0.05 look alike",
            left_open="what to do about the right panel. It is still falling at 640 rows, which normally means get more data. There are 891 passengers on the Titanic, and there will never be more.",
            student="reading two separately-scaled panels as if they were comparable. It is the most common way a true plot tells a false story.",
            catch="curves that have met mean more data changes nothing. Curves still far apart and falling mean starved of rows, not of capacity — and the two diagnoses have opposite remedies."),
        code('''
fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
for ax, deg, title in zip(axes, (1, 5),
                          ("Degree 1 — high bias", "Degree 5 — high variance")):
    c = lc[deg]
    ax.plot(c["n"], c["train"], "o-", color="#0b3d62", label="training folds")
    ax.plot(c["n"], c["valid"], "s-", color="#c0392b", label="held-out folds")
    ax.set_title(title)
    ax.set_xlabel("training passengers")
axes[0].set_ylabel("log loss")
axes[0].legend(loc="upper right")
plt.show()

print("Left: the curves have met — more data changes nothing.")
print("Right: still 1.6 apart and falling — starved of rows, not of capacity.")
print("There are 891 passengers on the Titanic. There will never be more.")
'''),

        # --------------------------------------------------------- diagnosis
        md("""
## 3 · Diagnose — three faults, one shape

Degree 5 scored 1.957 against an anchor of 0.666. It is **three times worse
than saying nothing at all**. Accuracy at that degree was 75.4%, which is poor
but not absurd — the two metrics disagree because $-\\log(0.01) \\approx 4.6$,
so one confident mistake contributes more than twenty ordinary ones.

**Diagnosis 1** is the decomposition you just measured: 143 columns from 712
rows is five rows per weight.

**Diagnosis 2** is the warning the build session told you to read.
"""),
        prompt(
            label="the warning, read rather than silenced",
            input="each degree, fitted with warnings captured",
            output="convergence, iterations, and the largest absolute coefficient",
            constraint="`record=True` with `simplefilter('always')` — the cell exists to READ a warning, so suppressing it defeats the purpose",
            check="assert degrees 1 and 3 converge and degree 4 does not",
            left_open="why raising max_iter will not help. If some θ separates the classes then scaling it up drives the training loss towards zero, so the infimum is never attained: the optimiser is not slow, it is looking for something that does not exist.",
            student="`max_iter=100000` and a coffee. The largest-coefficient column is the evidence — it grows without bound, which is what non-existence looks like numerically.",
            catch="two symptoms, one shape: the minimiser does not EXIST (separation) and the minimiser is not UNIQUE (rank 23 of 29). Both are about the optimisation problem, neither is about the passengers."),
        code('''
sep = {}
for deg in DEGREES:
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        m = pipeline(degree=deg).fit(X_train, y_train)
    converged = not any("converge" in str(x.message).lower() for x in w)
    sep[deg] = dict(converged=converged,
                    n_iter=int(np.max(m[-1].n_iter_)),
                    max_coef=float(np.abs(m[-1].coef_).max()))
    print(f"degree {deg}: converged {str(converged):5s}  "
          f"iterations {sep[deg]['n_iter']:5d}  "
          f"largest |θ| {sep[deg]['max_coef']:6.2f}")

assert sep[1]["converged"] and sep[3]["converged"]
assert not sep[4]["converged"], "degree 4 is where it stops arriving"
'''),
        md("""
The reflex is `max_iter=100000`. It will not help, because the optimiser is not
converging slowly — **it is looking for something that does not exist.**

If some $\\boldsymbol\\theta$ separates the two classes in the expanded space,
then $\\sigma(c\\,\\boldsymbol\\theta^{\\intercal}\\mathbf{x}) \\to \\{0, 1\\}$ as
$c \\to \\infty$, so the training log loss falls monotonically towards zero as
$\\lVert\\boldsymbol\\theta\\rVert \\to \\infty$. The infimum is never attained.

Two symptoms of one thing: the minimiser does not **exist** (separation), and
from the build session, the minimiser is not **unique** (rank 23 of 29
columns). Both are statements about the optimisation problem, not about the
passengers.

**Diagnosis 3 is Thread 1 returning.** You engineered a column that is an exact
linear combination of two others.
"""),
        prompt(
            label="the dependence, as an eigenvalue",
            input="SibSp, Parch, FamilySize, Age and Fare, scaled",
            output="the eigenvalues of XᵀX and its condition number",
            constraint="use `eigvalsh`, not `eigvals` — XᵀX is symmetric, and the general routine returns complex numbers with tiny imaginary parts that then have to be explained away",
            check="assert the smallest eigenvalue is below 1e-9 — the dependence must show up as a zero, not merely as a small number",
            left_open="that an exact zero eigenvalue and 'highly correlated columns' are different claims. This one is an identity, by construction, from a line of feature engineering.",
            student="reporting the condition number without the eigenvalues. 1e+15 tells you something is wrong; the zero eigenvalue tells you what.",
            catch="a zero eigenvalue of XᵀX is a direction in which your coefficients can move without changing a single prediction."),
        code('''
print("FamilySize − (SibSp + Parch + 1), largest absolute value over 712 rows:")
print((X_train["SibSp"] + X_train["Parch"] + 1
       - X_train["FamilySize"]).abs().max())

cols = ["SibSp", "Parch", "FamilySize", "Age", "Fare"]
Z = make_pipeline(SimpleImputer(strategy="median"),
                  StandardScaler()).fit_transform(X_train[cols])
G   = Z.T @ Z
eig = np.linalg.eigvalsh(G)

print(f"\\neigenvalues of XᵀX: {np.array2string(eig, precision=2)}")
print(f"condition number at α = 0: {np.linalg.cond(G):.2e}")

assert eig.min() < 1e-9, "the dependence should show as a zero eigenvalue"
'''),
        md("""
If $\\mathbf{X}^{\\intercal}\\mathbf{X}\\,\\mathbf{v} = \\lambda\\mathbf{v}$ then
$(\\mathbf{X}^{\\intercal}\\mathbf{X} + \\alpha\\mathbf{I})\\mathbf{v} =
(\\lambda + \\alpha)\\mathbf{v}$: same eigenvectors, every eigenvalue moved up
by exactly $\\alpha$. $\\mathbf{X}^{\\intercal}\\mathbf{X}$ is positive
semi-definite, so $\\lambda + \\alpha \\ge \\alpha > 0$ and **the matrix is
invertible for every $\\alpha > 0$**. That is the promise Lecture 2 made about
ridge, and this is its two-line proof.

It also bounds the conditioning by $(\\lambda_{\\max} + \\alpha)/\\alpha$, which
does not mention $\\lambda_{\\min}$ at all.
"""),
        prompt(
            label="what α does to the eigenvalues",
            input="the same eigenvalues",
            output="the condition number at four values of α, beside its bound",
            constraint="compute it from the SHIFTED eigenvalues (λ+α), not by refitting — the point is that ridge moves every eigenvalue up by exactly α and leaves the eigenvectors alone",
            check="assert α=1 brings a 1e15 condition number under 2000",
            left_open="that this is a two-line proof, not an empirical finding. XᵀX is positive semi-definite, so λ+α ≥ α > 0 and the matrix is invertible for every α > 0.",
            student="believing ridge 'removes multicollinearity'. It does not. The dependence is still exactly there; it is dominated, and the printout says so in as many words.",
            catch="the bound (λmax+α)/α does not mention λmin at all. That is why ridge works on a singular design and no amount of column-dropping is needed first."),
        code('''
for alpha in (0.0, 1e-6, 1e-3, 1.0):
    if alpha == 0.0:
        print(f"α = {alpha:<8g} condition number {np.linalg.cond(G):.3e}")
    else:
        k = (eig.max() + alpha) / (eig.min() + alpha)
        print(f"α = {alpha:<8g} condition number {k:.3e}   "
              f"(bound {(eig.max() + alpha) / alpha:.3e})")

k1 = (eig.max() + 1.0) / (eig.min() + 1.0)
assert k1 < 2000, "α = 1 should tame a 1e15 condition number"
print(f"\\nAt α = 1 the condition number falls from ~1e15 to {k1:.0f}.")
print("The dependence is not repaired. It is dominated.")
'''),

        # ------------------------------------------------------------- fix
        md("""
## 4 · Fix — ridge, lasso, elastic net, early stopping

Stop asking for the best fit. Ask for the best fit *among small models*:

$$J(\\boldsymbol\\theta) = L(\\boldsymbol\\theta)
  + \\alpha\\,\\Omega(\\boldsymbol\\theta)$$

Ridge takes $\\Omega = \\tfrac12\\lVert\\mathbf{w}\\rVert_2^2$ and has the same
closed form as Thread 1 with one term added,
$\\hat{\\boldsymbol\\theta} = (\\mathbf{X}^{\\intercal}\\mathbf{X} +
\\alpha\\mathbf{A})^{-1}\\mathbf{X}^{\\intercal}y$. Lasso takes
$\\Omega = \\lVert\\mathbf{w}\\rVert_1$, whose subgradient at zero is the whole
interval $[-1, 1]$ — which is why weights arrive at exactly zero and stay.

**Two traps in the API.** Scikit-Learn takes `C = 1/α`, so *small* `C` is
*strong* regularisation; and its default is `C=1.0, penalty="l2"`, so logistic
regression is regularised unless you say otherwise. The build session set
`C=1e6` on purpose, to have something to repair.
"""),
        prompt(
            label="the penalty grid, and a solver trap",
            input="eight values of C, and three penalty configurations",
            output="the grid and the penalty dictionary, nothing fitted yet",
            constraint="C = 1/α in scikit-learn, so SMALL C is STRONG regularisation — and the default is C=1.0 with an L2 penalty, so logistic regression is regularised unless you say otherwise",
            left_open="the solver choice on the lasso row. liblinear is the obvious pick for an L1 penalty and it is wrong here: it implements the intercept as a synthetic constant column and penalises it like any other weight, fitting an intercept of exactly 0.0000 at C=0.001 where saga fits 4.90.",
            student="reading a coefficient table without checking which solver produced it. This lecture states that the bias term is not penalised, and with the wrong solver that statement is false in your own output.",
            catch="check the intercept. If it is suspiciously near zero under a strong penalty, your solver is penalising it, and every coefficient in the table is compensating for that."),
        code('''
# The deck sweeps 17 values from 1e-4 to 1e4 with 10 folds. We stop at
# C = 0.316 and use 5 folds here, for a reason worth knowing: coordinate
# descent at degree 5 takes 163 s for a SINGLE cross-validation at C = 1, and
# 1.6 s at C = 0.01. The cost of a solver is not uniform over its
# hyperparameter, and the minimum is well inside the range either way.
#
# Note the solver on the lasso row. liblinear is the obvious choice for an L1
# penalty and it is the wrong one here: Scikit-Learn implements its intercept
# as a synthetic constant column and penalises that column's weight like any
# other, so at C = 0.001 it fits an intercept of exactly 0.0000 where saga fits
# 4.90. This lecture states that the bias term is not penalised, and its own
# red-team checklist asks you to check exactly that. Check the solver before
# you read a coefficient table.
Cs = np.logspace(-4, -0.5, 8)
cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

PENALTIES = {
    "ridge":   dict(penalty="l2", solver="lbfgs"),
    "lasso":   dict(penalty="l1", solver="saga", max_iter=5000),
    "elastic": dict(penalty="elasticnet", solver="saga", l1_ratio=0.5,
                    max_iter=1000),
}
'''),
        md("⏱ **about two minutes.**"),
        prompt(
            label="⏱ 2 min — three penalties, eight values each",
            input="degree 5, the eight C values, ridge / lasso / elastic net",
            output="the best C per penalty, its log loss, and how many weights survive",
            constraint="count the NON-ZERO weights as well as the score — the difference between ridge and lasso is not visible in the score alone",
            check="assert lasso zeroes something and ridge zeroes nothing",
            left_open="why the grid stops at C = 0.316. Coordinate descent at degree 5 takes 163 s for a single value at C = 1 and 1.6 s at C = 0.01 — the cost of a solver is not uniform over its hyperparameter.",
            student="sweeping to C = 1e4 because the deck does, and waiting twenty minutes for values that are all worse than the minimum already found. Grid ranges have costs as well as coverage.",
            catch="the nnz column is the check that the penalty you asked for is the penalty you got. An L1 run with 143 of 143 weights non-zero did not apply an L1 penalty."),
        code('''
reg = {}
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    for name, kw in PENALTIES.items():
        scores, nnz = [], []
        for C in Cs:
            s = cross_val_score(pipeline(degree=5, C=float(C), **kw),
                                X_train, y_train, cv=cv5,
                                scoring="neg_log_loss", n_jobs=NJ)
            fitted = pipeline(degree=5, C=float(C), **kw).fit(X_train, y_train)
            scores.append(float(-s.mean()))
            nnz.append(int((np.abs(fitted[-1].coef_[0]) > 1e-8).sum()))
        j = int(np.argmin(scores))
        reg[name] = dict(C=Cs[j], log_loss=scores[j], nnz=nnz[j],
                         all_scores=scores)
        print(f"{name:8s} best C = {Cs[j]:.4g}   log loss {scores[j]:.3f}   "
              f"non-zero weights {nnz[j]:3d} of 143")

assert reg["lasso"]["nnz"] < 143, "lasso must zero something out"
assert reg["ridge"]["nnz"] == 143, "ridge shrinks everything and zeroes nothing"
'''),
        md("""
### Read that table twice

Every penalty takes the worst model in the sweep — 1.957, worse than the anchor
— and makes it respectable. **And none of them beats the plain degree-2 model
you already had at 0.468.**

That is not a disappointing result to be explained away. The decomposition said
so an hour ago: the squared bias at degree 1 was under 0.01, so there was never
much bias for the extra capacity to buy back. Regularisation is a **repair**,
not an upgrade.
"""),
        prompt(
            label="read that table twice",
            input="the sweep and the three tuned penalties",
            output="all four numbers in one column, with degree 2 at the bottom",
            constraint="put the unregularised degree-2 model in the SAME list — a comparison between three repairs and no baseline is not a comparison",
            check="assert no penalty beats degree 2, and say in the message that if one ever does, the story changes and you must say so",
            left_open="that this is the intended result. Every penalty takes the worst model in the sweep and makes it respectable, and none of them beats the plain model you already had.",
            student="presenting the ridge improvement as the finding. The improvement is real and it is a repair, not an upgrade — the decomposition predicted exactly this an hour earlier.",
            catch="an assert that encodes the ARGUMENT, with a message saying what to do if it fires. That is the difference between a test and a tripwire."),
        code('''
print(f"degree 5, no penalty      {sweep[5]['valid']:.3f}")
for name in PENALTIES:
    print(f"degree 5, {name:8s}        {reg[name]['log_loss']:.3f}")
print(f"degree 2, no penalty      {sweep[2]['valid']:.3f}   <- still the best")

assert min(reg[n]["log_loss"] for n in PENALTIES) > sweep[2]["valid"], \\
    "if a penalty ever beats degree 2 here, the story changes and you must say so"
'''),

        md("""
### Early stopping — keeping the weights small by not going far

Gradient descent starts at $\\boldsymbol\\theta = \\mathbf{0}$, the smallest
possible model, and moves outward. Held-out error falls, reaches a minimum, and
then rises.

`warm_start=True` with `max_iter=1` is what makes each `fit` call one epoch of
the *same* fit. `penalty=None` is deliberate — we want early stopping to be the
only regularisation in the room. The transform is fitted on the training part
only, **outside** the loop; refitting it inside would be the first
application's leak, 500 times over.

⏱ **about 20 seconds.**
"""),
        prompt(
            label="⏱ 20 s — early stopping",
            input="a 75/25 split of the training rows, degree 5, no penalty at all",
            output="training and validation log loss at each of 500 epochs, and the epoch that minimises the validation curve",
            constraint="fit the transform on the training part ONLY and OUTSIDE the loop — refitting it inside would be the first application's leak, five hundred times over",
            check="assert the minimum is interior, since a minimum at epoch 500 would mean there was nothing to stop early",
            left_open="that `warm_start=True` with `max_iter=1` is what makes each fit call one epoch of the SAME fit rather than 500 independent one-epoch models.",
            student="`max_iter=1` without `warm_start`, which silently restarts from zero every epoch and produces a flat curve that looks like a model that cannot learn.",
            catch="`penalty=None` is deliberate. If you leave the default L2 on, early stopping is not the only regulariser in the room and the experiment measures two things at once."),
        code('''
A, B, y_a, y_b = train_test_split(X_train, y_train, test_size=0.25,
                                  random_state=RANDOM_STATE, stratify=y_train)
pre = prep(degree=5)
Z_a = pre.fit_transform(A)          # fitted on the training part only
Z_b = pre.transform(B)

assert Z_a.shape[1] == Z_b.shape[1] == 143
assert len(A) + len(B) == 712

clf = SGDClassifier(loss="log_loss", penalty=None, learning_rate="constant",
                    eta0=0.0015, random_state=RANDOM_STATE,
                    warm_start=True, max_iter=1, tol=None)

train_curve, valid_curve = [], []
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    for epoch in range(500):
        clf.fit(Z_a, y_a)
        train_curve.append(log_loss(y_a, clf.predict_proba(Z_a)[:, 1],
                                    labels=[0, 1]))
        valid_curve.append(log_loss(y_b, clf.predict_proba(Z_b)[:, 1],
                                    labels=[0, 1]))

best = int(np.argmin(valid_curve))
print(f"best epoch {best + 1}: validation {valid_curve[best]:.3f}")
print(f"epoch 500:      validation {valid_curve[-1]:.3f}")
print(f"regret for not stopping: {valid_curve[-1] - valid_curve[best]:.3f}")
assert best + 1 < 500, "the minimum must be interior or there is nothing to see"
'''),
        prompt(
            label="the shape, not the number",
            input="the two epoch curves",
            output="both curves with the chosen epoch marked",
            constraint="say out loud that these are bad numbers in absolute terms — plain SGD at a constant learning rate on 143 correlated columns is a poor optimiser",
            left_open="that the stopping epoch is a hyperparameter chosen on held-out rows, exactly like C. It is not free regularisation; it is regularisation whose knob happens to be time.",
            student="quoting this validation log loss beside the tuned ridge number, as if the two were competing. This cell is showing a shape.",
            catch="when a demonstration is deliberately not competitive, print that in the cell. A number left unqualified on a slide will be quoted qualified."),
        code('''
fig, ax = plt.subplots(figsize=(7.5, 3.2))
ax.plot(train_curve, color="#0b3d62", label="training subset")
ax.plot(valid_curve, color="#c0392b", label="validation subset")
ax.axvline(best, ls="--", color="#14663a")
ax.set_xlabel("epoch"); ax.set_ylabel("log loss"); ax.legend()
plt.show()

print("These are bad numbers in absolute terms: plain SGD at a constant")
print("learning rate on 143 correlated columns is a poor optimiser. We are")
print("showing the SHAPE, not a competitive model — and the stopping epoch is")
print("a hyperparameter chosen on held-out rows like any other.")
'''),

        # ------------------------------------------- the assistant failure
        md("""
## 5 · The worked assistant failure

> ⚠ **read before running**

The prompt:

> *"Find the best value of `C` for my logistic regression and tell me how well
> it does."*

Every word is reasonable. Nothing in it says **on what data** the best `C` is to
be found, or **on what data** "how well it does" is to be measured. Here is the
plausible code it returns. It runs, it warns about nothing, and it prints a
number a reader will quote.
"""),
        prompt(
            label="⚠ what the assistant returns",
            input="'find the best value of C and tell me how well it does'",
            output="the best C and its log loss",
            constraint="score each candidate on the test set and report the best — which is what the prompt literally asks for",
            left_open="what touched the test set. Seventeen models did, and then we reported the score of whichever one it liked best. The code never WRITES to the test set; it reads it eighteen times, and reading is enough.",
            student="exactly this. It runs, it warns about nothing, and it prints a number a reader will quote — and the number is a minimum over seventeen noisy estimates, which is biased downward even when every estimate is individually unbiased.",
            catch="count reads of the test set, not writes. One read at the very end is the budget; anything that selects between models is a read."),
        code('''
# ⚠ WRONG — this is the failure, not the fix
best_C, best_score = None, np.inf

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    for C in np.logspace(-4, 4, 17):
        m = pipeline(degree=3, C=float(C)).fit(X_train, y_train)
        score = log_loss(y_test, m.predict_proba(X_test)[:, 1], labels=[0, 1])
        if score < best_score:
            best_C, best_score = float(C), score

print(f"best C = {best_C:g}, log loss = {best_score:.4f}")
'''),
        md("""
**The review question:** *what touched the test set?*

Seventeen models did, and then we reported the score of whichever one it liked
best. `best_score` is a **minimum over seventeen noisy estimates**, and the
minimum of noisy estimates is biased downward even when every estimate is
individually unbiased. The code never *writes* to the test set. It reads it
eighteen times, and reading is enough.

**Now measure the damage rather than asserting it.** For each of 8 seeds:
split, run the loop above, and also run the honest version that picks `C` by
5-fold cross-validation inside the training part. Score both on the same
held-out rows.

⏱ **about two minutes.** The deck uses 20 seeds; 8 gives the same conclusion
with a wider interval, and the interval is the point.
"""),
        prompt(
            label="⏱ 2 min — measure the optimism",
            input="8 seeds, 17 candidates each",
            output="the dishonest score and the honest score on the same held-out rows",
            constraint="score BOTH choices on the same rows — the difference has to come from the selection procedure and nothing else",
            check="assert the bias is positive, since it is one-sided by construction",
            left_open="how the effect scales. About 0.02 of log loss against a seed-to-seed spread of 0.05: on a single split you would never see it. Seventeen candidates is a small search, and a randomised search over a thousand does the same thing much louder.",
            student="running this once, seeing a difference smaller than the noise, and concluding the leak does not matter. It never averages away and it grows with how hard you looked.",
            catch="report how often the two procedures picked the SAME C. When they agree the leak costs nothing on that seed, and the count tells you how much of the effect is selection rather than scoring."),
        code('''
def one_seed(seed):
    A, B, ya, yb = train_test_split(X, y, test_size=0.2, random_state=seed,
                                    stratify=y)
    test_scores, cv_scores = [], []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for C in np.logspace(-4, 4, 17):
            m = pipeline(degree=3, C=float(C)).fit(A, ya)
            test_scores.append(log_loss(yb, m.predict_proba(B)[:, 1],
                                        labels=[0, 1]))
            cv_scores.append(-cross_val_score(
                pipeline(degree=3, C=float(C)), A, ya, cv=5,
                scoring="neg_log_loss").mean())
    # what the assistant reported, and what an honest choice would have scored
    return (test_scores[int(np.argmin(test_scores))],
            test_scores[int(np.argmin(cv_scores))],
            int(np.argmin(test_scores) == np.argmin(cv_scores)))

rows = Parallel(n_jobs=NJ)(delayed(one_seed)(s) for s in range(8))
reported = np.array([r[0] for r in rows])
honest   = np.array([r[1] for r in rows])
same_C   = sum(r[2] for r in rows)

print(f"chosen on the test set, scored on it: {reported.mean():.3f} "
      f"± {reported.std():.3f}")
print(f"chosen by cross-validation:           {honest.mean():.3f} "
      f"± {honest.std():.3f}")
print(f"optimism: {(honest - reported).mean():+.3f} of log loss "
      f"({100 * (honest - reported).mean() / honest.mean():.1f}%)")
print(f"seeds where the dishonest number flatters: "
      f"{int((reported < honest).sum())} of {len(rows)}")
print(f"seeds where both procedures picked the same C: {same_C} of {len(rows)}")

assert (honest - reported).mean() > 0, "the bias is one-sided by construction"
'''),
        md("""
**Read both halves of that.**

*It is small.* About 0.02 of log loss, against a seed-to-seed spread of 0.05.
On a single split you would never see it. Seventeen candidates is a small
search.

*It is real, and it is one-sided.* It never averages away, and **it grows with
the number of candidates you try.** Search a thousand configurations — which a
randomised search does in an afternoon — and the same procedure hands you a
number that is confidently wrong. The size of the lie scales with how hard you
looked.

**The corrected specification:**

> *"Choose `C` by 10-fold cross-validation **on the training set only**, using
> `GridSearchCV` over a pipeline. Refit the winner on the full training set.
> Then evaluate once on the test set and report both numbers separately, with
> the fold spread."*
"""),
        prompt(
            label="the honest version",
            input="the same 17 candidates",
            output="the cross-validated estimate with its fold spread, and one test score",
            constraint="GridSearchCV over the whole PIPELINE, so the preprocessing is refitted inside every fold, and the test set is touched exactly once at the end",
            left_open="that the test number is allowed to be worse than the CV number. The cell says so, because the instinct when it is worse is to go back and adjust something.",
            student="grid-searching the classifier alone on pre-transformed data. The search is then honest about C and dishonest about the scaler.",
            catch="report both numbers separately with the fold spread beside them. One number that has been through a selection is not the same kind of object as one that has not."),
        code('''
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    gs = GridSearchCV(pipeline(degree=3),
                      {"clf__C": np.logspace(-4, 4, 17)},
                      scoring="neg_log_loss", cv=cv,
                      n_jobs=NJ).fit(X_train, y_train)

honest_cv   = -gs.best_score_
honest_test = log_loss(y_test, gs.predict_proba(X_test)[:, 1], labels=[0, 1])
fold_spread = gs.cv_results_["std_test_score"][gs.best_index_]

print(f"chosen C:                {gs.best_params_['clf__C']:g}")
print(f"honest CV estimate:      {honest_cv:.3f}  (fold sd {fold_spread:.3f})")
print(f"the test set, once:      {honest_test:.3f}")
print("\\nThe second number is allowed to be worse than the first.")

assert gs.best_index_ is not None
'''),

        # ------------------------------------------------------- re-measure
        md("""
## 6 · Re-measure — the test set, once

179 passengers, untouched since the split at the top of this notebook. Five
candidates, every one of whose hyperparameters was fixed before this cell ran.
"""),
        prompt(
            label="the test set, once",
            input="179 passengers untouched since the split, and five fixed candidates",
            output="log loss, Brier and accuracy for each",
            constraint="every hyperparameter must have been fixed BEFORE this cell ran — no selection happens here, only measurement",
            check="assert the winner beats the anchor",
            left_open="that the winner is the model with no repair in it. Degree 2, no penalty, chosen by reading a held-out curve.",
            student="adding a sixth candidate after seeing the table. That is the assistant failure from section 5, committed by hand and one cell later.",
            catch="if you find yourself editing this cell after reading its output, stop. The test set has now been read twice."),
        code('''
candidates = {
    "degree 5, no penalty":   pipeline(degree=5),
    "degree 5, ridge tuned":  pipeline(degree=5, C=float(reg["ridge"]["C"])),
    "degree 5, lasso tuned":  pipeline(degree=5, C=float(reg["lasso"]["C"]),
                                       penalty="l1", solver="saga", max_iter=5000),
    "degree 1, sklearn defaults": pipeline(degree=1, C=1.0),
    "degree 2, no penalty":   pipeline(degree=2),
}

final = {}
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    for name, m in candidates.items():
        m.fit(X_train, y_train)
        p = m.predict_proba(X_test)[:, 1]
        final[name] = dict(log_loss=log_loss(y_test, p, labels=[0, 1]),
                           brier=brier_score_loss(y_test, p),
                           accuracy=accuracy_score(y_test, p >= 0.5))
        print(f"{name:30s} log loss {final[name]['log_loss']:.3f}   "
              f"Brier {final[name]['brier']:.3f}   "
              f"accuracy {final[name]['accuracy']:.1%}")

winner = min(final, key=lambda k: final[k]["log_loss"])
print(f"\\nwinner: {winner}")
assert final[winner]["log_loss"] < constant_log_loss, \
    "the winner must at least beat the anchor"
'''),
        prompt(
            label="the five numbers, and the floor",
            input="everything measured",
            output="anchor, best cross-validated, final test, majority accuracy, and the Brier score against the measured floor",
            constraint="put the noise floor beside the final Brier — the gap between them is what is actually left to win",
            check="assert the final model beats the anchor",
            left_open="what the ninety minutes bought. Not a better number: the winner is a two-line model you could have written before the build session started. It bought knowing WHY it wins, in three terms.",
            student="concluding the session was wasted because the simple model won. The alternative is choosing the interesting model over the better one, which is the failure this course exists to prevent.",
            catch="0.013 of Brier left on the table, in total, for any model of these columns. A remaining-headroom number is the most honest thing you can put at the end of a report."),
        code('''
committed = final["degree 5, no penalty"]["log_loss"]
best      = final[winner]["log_loss"]

print(f"anchor — report the base rate to everyone   {constant_log_loss:.3f}")
print(f"best cross-validated log loss               {sweep[2]['valid']:.3f}")
print(f"final, on the test set                      {best:.3f}")
print(f"majority-class accuracy on the test set     {1 - y_test.mean():.1%}")
print(f"\\nimprovement over where the build session ended: {committed - best:.3f}")
print(f"Brier {final[winner]['brier']:.3f} against a measured floor of {noise:.3f}")
print("=> about 0.013 of Brier score is left on the table, in total, for any")
print("   model of these columns.")

assert best < constant_log_loss, "we must at least beat the anchor"
'''),
        md("""
**The winner is a model with no repair in it.** Degree 2, no penalty, chosen by
reading a held-out curve — and `LogisticRegression()` with every default left
alone is statistically the same model.

Ninety minutes of ridge, lasso, elastic net and early stopping, and the best
system on the test set is a two-line model you could have written before the
build session started. Report it anyway. The alternative is choosing the
interesting model over the better one, which is the failure this course exists
to prevent.

What the work bought you is not a better number. It is knowing **why** degree 2
wins, in three terms rather than as a preference; knowing the floor is 0.121
Brier; and knowing that the degree-5 model was not merely worse but *ill-posed*,
and exactly which term repairs that.

---

## 7 · Red-team

Swap notebooks. Eight minutes. Report what you found, not what you would have
done differently.

1. **What touched the test set?** Count every *read*, not every write.
2. **What was fitted, and on what?** Is the scaler inside the cross-validated
   pipeline, or fitted once outside it?
3. **What is the shape here?** How many columns does the degree-5 pipeline
   actually produce, and how many rows are there?
4. **What was dropped?** Which passengers have an imputed `Age`, and how many?
5. **What is the default you did not ask for?** Find every
   `LogisticRegression` whose `C` was never stated.

The cell below answers questions 3 and 4 on this notebook. Run the equivalent
on your neighbour's.
"""),
        prompt(
            label="red-team questions 3 and 4",
            input="the degree-5 pipeline and the two splits",
            output="columns against rows, rows per weight, and how many ages were imputed",
            constraint="answer them on THIS notebook first, then run the equivalent on your neighbour's",
            check="assert 143 columns and that no passenger was dropped — the imputer fills, and it is fitted per fold",
            left_open="questions 1, 2 and 5 of the checklist, which no cell can answer for you: what touched the test set, what was fitted on what, and which `LogisticRegression` had its C left unstated.",
            student="running the red team on someone else's notebook and not their own. Five rows per weight is the kind of thing that is obvious in another person's code and invisible in yours.",
            catch="rows per weight, printed. 712 rows over 143 columns is 5.0, and no amount of tuning repairs a ratio like that."),
        code('''
n_cols = prep(5).fit(X_train).transform(X_train).shape[1]
print(f"degree-5 columns: {n_cols}   training rows: {len(X_train)}   "
      f"rows per weight: {len(X_train) / n_cols:.1f}")

print(f"\\nimputed Age, training set: {int(X_train['Age'].isna().sum())} of "
      f"{len(X_train)}")
print(f"imputed Age, test set:     {int(X_test['Age'].isna().sum())} of "
      f"{len(X_test)}")
print(f"missing Embarked:          {int(full['Embarked'].isna().sum())}")

# Nothing was dropped: the imputer fills, and it is fitted per fold.
assert len(X_train) + len(X_test) == 891
assert n_cols == 143
'''),
        md("""
### The standing constraint, extended

Add one clause to what you wrote down in Lecture 2:

> *"Split before anything is fitted. All preprocessing lives inside a
> `Pipeline` passed to cross-validation. Nothing derived from the test set may
> appear in the training path — **including the choice of any
> hyperparameter**. Fixed random seed. Print the fold scores, not just the
> mean."*

Eleven added words. They are the whole of today's assistant failure.
"""),
    ]
    return cells
