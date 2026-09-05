#!/usr/bin/env python3
"""
Lecture 5 — Regularisation and the bias-variance trade-off. Titanic, Géron Ch 4.

Exports build() -> list[cell]; tools/make_notebooks.py wraps it.

Structure mirrors the deck: the decomposition derived and then measured, the
curves read, the three faults of the degree-5 model diagnosed, the four repairs
applied, and the test set touched once.

It rebuilds the previous lecture's degree sweep from scratch, imports included:
a notebook that only runs because another notebook is still in memory is not
reproducible.

Every quantity this notebook prints that also appears on a slide is computed the
way `tools/figures_app03.py` computes it, since that script wrote
`assets/figures/figures.json` and the slides are read off it — the same split,
the same `prep()`, the same `StratifiedKFold(10)`, the same `C=1e6`, the same
200 draws of 400 rows, the same 20 resplits in the tuning measurement.

Two places run a smaller experiment than the script, each stated in the cell
that does it, and neither changes a number the deck reports:

  * the lasso and elastic-net sweeps stop at C = 0.1 rather than 1e4. Coordinate
    descent at degree 5 costs about the same at every weak penalty and both
    curves are flat above C = 0.01, so the reported minimum — at C = 3.2e-4 and
    C = 1e-4 — is unaffected. Ridge is cheap and runs the full grid, so one
    curve on the plot still shows the weak-penalty tail.
  * the coefficient paths use 13 values of C rather than 25, with the same
    endpoints, so the non-zero counts the deck quotes (5 and 142) are the same
    two numbers.

`n_jobs` is capped rather than set to -1. A free Colab CPU runtime has two
cores, so -1 buys nothing there and on a shared machine it oversubscribes.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from make_notebooks import code, header, md, SETUP, SETUP_PROMPT        # noqa: E402
from _prompt import prompt                                # noqa: E402


def build() -> list:
    cells = header(5, "Regularisation and the bias–variance trade-off", "",
                   "Chapter 4")

    cells += [
        md("""
The previous lecture ended with two curves on one axis: a training curve that
falls forever and a held-out curve that turns at degree 2 and climbs to 1.922 —
**worse than reporting the base rate to everybody**.

This notebook says what those curves are made of. First the decomposition,
derived on the slides and measured here on your own pipeline, with the identity
checked to floating-point noise before any picture is drawn. Then the three
separate faults in the degree-5 model, and the one term that repairs all three.

Runs on free CPU. It is the heaviest notebook in Part I — a few minutes on a
laptop and two to three times that on a two-core Colab runtime — and the four
slow cells each say what they cost.
"""),

        # ---------------------------------------------------------- setup
        md("## 1 · Setup, and the sweep rebuilt"), SETUP_PROMPT, SETUP,
        prompt(
            label="every import, in one place",
            input="nothing",
            output="every name this notebook uses below, imported once",
            constraint="repeat them here — do NOT rely on the previous "
                       "notebook's kernel still being alive",
            check="Runtime → Restart, then run this cell alone. If anything "
                  "below raises NameError, that import belongs here",
            **{"try": "run the last cell of the notebook first, from a cold "
                      "kernel. It fails. A notebook that only runs top to "
                      "bottom is the only kind you can trust."}),
        code('''
# Every name used below, imported once. A notebook that only runs because a
# previous one is still in memory is not reproducible.
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
# nothing there and on a shared machine it oversubscribes the box.
NJ = 4
'''),
        prompt(
            label="the data, rebuilt from scratch",
            input="the Titanic tarball at github.com/ageron/data",
            output="891 rows with the four engineered columns added",
            constraint="download-if-absent again, and keep FamilySize even "
                       "though we already know it is one of the faults — "
                       "section 4 needs the column in order to demonstrate the "
                       "dependence",
            check="891 rows, Survived holding only 0 and 1, and "
                  "SibSp + Parch + 1 - FamilySize exactly zero on every row. "
                  "When a notebook exists to repair something, reproduce the "
                  "broken state first, deliberately",
            **{"try": "delete `datasets/` and re-run. If the cell cannot "
                      "rebuild its own input from nothing it is cached, not "
                      "reproducible."}),
        code('''
# ~5 s the first time, instant afterwards.
def load_titanic():
    tarball = Path("datasets/titanic.tgz")
    if not tarball.is_file():
        Path("datasets").mkdir(parents=True, exist_ok=True)
        url = "https://github.com/ageron/data/raw/main/titanic.tgz"
        urllib.request.urlretrieve(url, tarball)
        with tarfile.open(tarball) as t:
            t.extractall(path="datasets", filter="data")
    return pd.read_csv("datasets/titanic/train.csv")


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


full = engineer(load_titanic())

assert full.shape[0] == 891
assert full["Survived"].isin([0, 1]).all()
# Keep this line in view. It is the third diagnosis, three sections from now.
assert (full["SibSp"] + full["Parch"] + 1 - full["FamilySize"]).abs().max() == 0
print(f"{len(full)} passengers, {int(full['Survived'].sum())} of whom survived")
'''),
        prompt(
            label="the same split, and one pipeline factory",
            input="the engineered frame",
            output="the same 712/179 stratified split as the previous lecture, "
                   "and a function returning a fresh unfitted pipeline",
            constraint="a FUNCTION returning a new pipeline on each call — one "
                       "shared object refitted in a loop carries the previous "
                       "iteration's state into the next; and FamilySize stays "
                       "in the frame but out of the numeric block",
            check="712 and 179 rows, and the anchor recomputed to 0.666. "
                  "Recompute the anchor rather than copying it across: if it "
                  "comes out different, the two notebooks are not on the same "
                  "data and nothing below is comparable",
            **{"try": "add FamilySize back to NUM. The degree-1 column count "
                      "rises from 22 to 23, and section 4's condition number "
                      "goes with it."}),
        code('''
NUM = ["Age", "Fare", "SibSp", "Parch"]     # FamilySize is left out on purpose
CAT = ["Pclass", "Sex", "Embarked", "Title", "Deck"]
BIN = ["IsAlone"]
ALL = NUM + ["FamilySize"] + CAT + BIN

X, y = full[ALL], full["Survived"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)

assert len(X_train) == 712 and len(X_test) == 179
assert set(X_train.index).isdisjoint(X_test.index)


def prep(degree=1):
    num = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                        PolynomialFeatures(degree=degree, include_bias=False))
    # min_frequency rather than handle_unknown="ignore": with drop="first" an
    # ignored unknown level is encoded all-zeros, which is the encoding of the
    # reference level, so an unseen deck would be scored as though it were A.
    cat = make_pipeline(SimpleImputer(strategy="most_frequent"),
                        OneHotEncoder(drop="first", min_frequency=2,
                                      handle_unknown="infrequent_if_exist"))
    return ColumnTransformer([("num", num, NUM), ("cat", cat, CAT),
                              ("bin", "passthrough", BIN)])


def pipeline(degree=1, C=1e6, penalty="l2", solver="lbfgs", l1_ratio=None,
             max_iter=4000):
    """C=1e6 is the previous lecture's setting: the penalty effectively off."""
    return Pipeline([
        ("prep", prep(degree)),
        ("clf", LogisticRegression(C=C, penalty=penalty, solver=solver,
                                   l1_ratio=l1_ratio, max_iter=max_iter,
                                   random_state=RANDOM_STATE))])


cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)

p = y_train.mean()
constant_log_loss = -(p * np.log(p) + (1 - p) * np.log(1 - p))
print(f"train rate {y_train.mean():.4f}   test rate {y_test.mean():.4f}")
print(f"anchor — report the base rate to everyone: {constant_log_loss:.3f}")
assert abs(constant_log_loss - 0.666) < 0.001
'''),
        md("""
### The degree sweep, rebuilt

⏱ **about 10 seconds.** The convergence warnings at degree 4 and above are
suppressed here and *counted* in section 4 — they are the second diagnosis, not
noise.
"""),
        prompt(
            label="⏱ 10 s — the degree sweep, rebuilt",
            input="degrees 1 to 6",
            output="columns, training log loss and held-out log loss at each "
                   "degree",
            constraint="`return_train_score=True` — it is off by default and "
                       "it is the whole experiment: one curve tells you nothing",
            check="22 columns at degree 1 and 143 at degree 5, on 712 rows; a "
                  "low degree wins on held-out score; and degree 5 is worse "
                  "than the constant anchor of 0.666, which is the fact the "
                  "rest of the notebook explains",
            **{"try": "assert the exact held-out score at degree 5 instead of "
                      "asserting the argument. Ask what that assert would do "
                      "if someone legitimately improved the encoder — it would "
                      "fail, and it would be pinning a number rather than "
                      "checking a fact."}),
        code('''
DEGREES = [1, 2, 3, 4, 5, 6]
sweep = {}

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    for d in DEGREES:
        r = cross_validate(pipeline(degree=d), X_train, y_train, cv=cv,
                           scoring=["neg_log_loss", "accuracy"],
                           return_train_score=True, n_jobs=NJ)
        n_cols = prep(d).fit(X_train).transform(X_train).shape[1]
        sweep[d] = dict(cols=n_cols,
                        train=float(-r["train_neg_log_loss"].mean()),
                        valid=float(-r["test_neg_log_loss"].mean()),
                        acc=float(r["test_accuracy"].mean()))

print(f"{'deg':>3} {'cols':>5} {'train':>8} {'held-out':>9} {'gap':>8} "
      f"{'accuracy':>9}")
for d in DEGREES:
    s = sweep[d]
    print(f"{d:>3} {s['cols']:>5} {s['train']:>8.3f} {s['valid']:>9.3f} "
          f"{s['valid'] - s['train']:>8.3f} {s['acc']:>9.1%}")

assert sweep[1]["cols"] == 22 and sweep[5]["cols"] == 143
# Assert the argument, not three decimals of it.
assert min(sweep, key=lambda d: sweep[d]["valid"]) in (1, 2), \\
    "a low degree should win on held-out log loss"
assert sweep[5]["valid"] > 2 * constant_log_loss, \\
    "degree 5 should be far worse than saying nothing"
'''),

        # ================================================ THE DECOMPOSITION
        md("""
## 2 · The decomposition, measured

Fix one passenger $x$. Two things are random: the **training set** $D$ you
happened to draw, and the **label** $y$ of that passenger. Write
$\\hat{p}_D(x)$ for what your model predicts,
$\\bar{p}(x) = \\mathbb{E}_D[\\hat{p}_D(x)]$ for the average prediction over
training sets, and $p^{*}(x)$ for the truth.

Add and subtract $\\bar{p}$; the cross term dies because $\\bar{p}$ is *defined*
as $\\mathbb{E}_D[\\hat{p}_D]$. Do it again on the first term with $p^{*}$
inserted, and average over $y$ as well:

$$\\mathbb{E}\\left[(y - \\hat{p}_D)^{2}\\right]
  = \\underbrace{(p^{*} - \\bar{p})^{2}}_{\\text{squared bias}}
  + \\underbrace{\\mathbb{E}_{D}\\left[(\\hat{p}_D - \\bar{p})^{2}\\right]}_{\\text{variance}}
  + \\underbrace{p^{*}(1 - p^{*})}_{\\text{noise}}$$

**One caveat, before any number appears.** That identity is exact for *squared
error*. Log loss does not decompose this way, so everything in this section is a
**Brier** score. The shape of the answer transfers; the numbers are Brier
numbers.

### The third term first — the one you cannot fix

$p^{*}$ is never observed. But where several passengers share *exactly* the same
recorded values, every model of those columns must give them the same number, so
whatever their outcomes do inside that cell is a floor.
"""),
        prompt(
            label="the term you cannot fix, measured",
            input="passengers grouped by five banded columns",
            output="the noise floor, as a Brier score",
            constraint="use the UNBIASED estimator k(m−k)/(m(m−1)), not the "
                       "plain sample variance — with cells of size 2 the biased "
                       "version understates the floor by half",
            check="133 cells, 102 of them holding two or more passengers, and "
                  "a floor of 0.121. If your 'irreducible error' falls when you "
                  "improve the model, it was never irreducible",
            **{"try": "drop `Embarked` from the key list. Cells get larger and "
                      "the measured floor rises — coarser inputs mean more "
                      "genuinely identical passengers, and a higher floor."}),
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
mixed    = multi[(multi["k"] > 0) & (multi["k"] < multi["m"])]

print(f"{len(cells)} cells; {len(multi)} hold two or more passengers "
      f"({int(multi['m'].sum())} people)")
print(f"{len(mixed)} of those cells are mixed, covering "
      f"{int(mixed['m'].sum())} passengers "
      f"({mixed['m'].sum() / len(full):.0%} of everyone on board)")
print(f"\\nmeasured noise floor (Brier): {noise:.3f}")

assert len(cells) == 133 and len(multi) == 102
assert int(multi["m"].sum()) == 858
assert abs(noise - 0.121) < 0.001
'''),
        prompt(
            label="look at the cells that disagree",
            input="the mixed cells",
            output="the four where identical inputs disagree most loudly",
            constraint="rank by k(m−k), the number of disagreeing pairs — not "
                       "by cell size, which would just find the biggest groups",
            check="the loudest cell holds 62 third-class men aged 26 to 40 "
                  "travelling alone from Southampton, 13 of whom survived. A "
                  "floor you can read is a floor you can defend",
            **{"try": "rank by cell size instead. The largest cell is not the "
                      "most informative one: a cell of 80 where 79 died says "
                      "almost nothing about the floor."}),
        code('''
worst = mixed.assign(mix=lambda t: t["k"] * (t["m"] - t["k"])).nlargest(4, "mix")
for key, row in worst.iterrows():
    print(f"{' · '.join(str(v) for v in key):42s}  "
          f"{int(row['m']):3d} passengers: {int(row['m'] - row['k']):2d} died, "
          f"{int(row['k']):2d} survived")
print("\\nNo model of these five columns can separate the people inside a row.")

top = worst.iloc[0]
assert int(top["m"]) == 62 and int(top["k"]) == 13
'''),
        md("""
### The other two terms, on your own pipeline

200 training sets of 400 rows each, drawn **without replacement** from the 712
you hold. A bootstrap *with* replacement gives each fit about 253 distinct rows,
and the variance you would then measure is partly the variance of the resampling
scheme.

⏱ **under a minute on a laptop, two to three minutes on Colab.** That is the
cell working, not the cell hanging.
"""),
        prompt(
            label="⏱ 30 s — the other two terms",
            input="200 training sets of 400 rows, drawn without replacement, at "
                  "each of six degrees",
            output="variance, bias²+noise and total expected Brier per degree",
            constraint="draw WITHOUT replacement, and predict on the same "
                       "held-out rows every time — the expectation is over the "
                       "training set alone, so the evaluation points must not "
                       "move",
            check="the prediction matrix is exactly (200, 179) before anything "
                  "is averaged. A silently broadcast array here produces a "
                  "beautifully wrong stackplot and no error",
            **{"try": "`replace=True`. The variance term rises at every degree, "
                      "and none of the rise is the model's."}),
        code('''
def boot_fit(deg, sel):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = pipeline(degree=deg).fit(X_train.iloc[sel], y_train.iloc[sel])
    return m.predict_proba(X_test)[:, 1]


N_BOOT, BOOT_N = 200, 400
rng   = np.random.default_rng(RANDOM_STATE)
draws = [rng.choice(len(X_train), size=BOOT_N, replace=False)
         for _ in range(N_BOOT)]

yv, bv = y_test.values.astype(float), {}
for deg in DEGREES:
    P = np.array(Parallel(n_jobs=NJ)(
        delayed(boot_fit)(deg, sel) for sel in draws))
    assert P.shape == (N_BOOT, len(X_test))
    pbar = P.mean(axis=0)
    bv[deg] = dict(variance=float(((P - pbar) ** 2).mean(axis=0).mean()),
                   bias2_noise=float(((yv - pbar) ** 2).mean()),
                   total=float(((yv - P) ** 2).mean(axis=0).mean()))

print(f"{'deg':>3} {'bias2+noise':>12} {'variance':>10} {'total':>8}")
for deg in DEGREES:
    b = bv[deg]
    print(f"{deg:>3} {b['bias2_noise']:>12.3f} {b['variance']:>10.3f} "
          f"{b['total']:>8.3f}")
'''),
        prompt(
            label="check the identity before believing the plot",
            input="the three measured terms",
            output="the residual of total − variance − (bias²+noise) at every "
                   "degree, and the growth of the variance term",
            constraint="assert the residual is below 1e-12 — these three "
                       "columns are not a MODEL of the error, they are the "
                       "error rearranged, so anything above floating-point "
                       "noise is a bug",
            check="the residual is at machine epsilon, and the variance term is "
                  "multiplied by about twelve from degree 1 to degree 6 while "
                  "bias²+noise moves by a fifth. Whenever you decompose a "
                  "quantity, assert the parts add up",
            **{"try": "subtract the measured floor from the first column to "
                      "get the squared bias. At degree 1 it is under 0.01 — "
                      "there was never much bias for extra capacity to buy "
                      "back."}),
        code('''
for deg in DEGREES:
    resid = abs(bv[deg]["total"] - bv[deg]["variance"] - bv[deg]["bias2_noise"])
    assert resid < 1e-12, f"degree {deg}: identity residual {resid:g}"
print("identity holds at machine precision for every degree")

growth = bv[6]["variance"] / bv[1]["variance"]
print(f"\\nvariance multiplied by {growth:.1f} from degree 1 to degree 6")
print(f"bias2 + noise at degree 1: {bv[1]['bias2_noise']:.3f}, against a "
      f"measured noise floor of {noise:.3f}")
print(f"=> squared bias at degree 1 is about "
      f"{bv[1]['bias2_noise'] - noise:.3f}. There was never much bias to buy "
      f"back.")
assert growth > 10
'''),
        prompt(
            label="the decomposition, stacked",
            input="the three terms at six degrees",
            output="a stackplot with the total overlaid and the measured floor "
                   "drawn in",
            constraint="draw the noise floor as a horizontal line — the point "
                       "of the picture is how much of the bottom band is "
                       "unreachable",
            check="the black total line lies exactly on top of the stack, "
                  "because the identity closed in the cell above. If a stacked "
                  "plot has a band you cannot separately measure, it is a "
                  "diagram, not data",
            **{"try": "remove the floor line. The bottom band now reads as "
                      "something to attack, and almost none of it is."}),
        code('''
fig, ax = plt.subplots(figsize=(7.5, 3.6))
ax.stackplot(DEGREES,
             [bv[k]["bias2_noise"] for k in DEGREES],
             [bv[k]["variance"] for k in DEGREES],
             labels=["bias2 + noise", "variance"], colors=["#9fb8ca", "#e6b0a8"])
ax.plot(DEGREES, [bv[k]["total"] for k in DEGREES], "o-", color="#16212b",
        label="total (expected Brier)")
ax.axhline(noise, ls="--", color="#6c3483")
ax.text(6, noise - 0.006, f"measured noise floor {noise:.3f}", ha="right",
        va="top", color="#6c3483")
ax.set_xlabel("polynomial degree")
ax.set_ylabel("expected squared error")
ax.legend(loc="upper left")
plt.show()
'''),
        md("""
### Putting it back on the two curves

The training curve is measured on the rows that were fitted, so it sees no
variance term at all. The held-out curve is the sum of all three. **The vertical
gap between them is the variance term.**
"""),
        prompt(
            label="the gap is the variance term",
            input="the degree sweep",
            output="held-out minus training log loss at degree 1 and degree 5, "
                   "and the ratio",
            constraint="report the GAP, not the two numbers separately — the "
                       "gap is the quantity the decomposition named",
            check="0.073 at degree 1 and 1.636 at degree 5, so the gap grew by "
                  "a factor of about twenty-two while the data did not change "
                  "at all",
            **{"try": "compute the same two gaps in Brier rather than log loss "
                      "and compare them with the variance column above. Same "
                      "story, different units — and only the Brier one is the "
                      "quantity the identity is about."}),
        code('''
for deg in (1, 5):
    gap = sweep[deg]["valid"] - sweep[deg]["train"]
    print(f"degree {deg}: held-out - training = {gap:.3f}")

g1 = sweep[1]["valid"] - sweep[1]["train"]
g5 = sweep[5]["valid"] - sweep[5]["train"]
print(f"\\nthe gap grew by a factor of {g5 / g1:.0f} while the model family "
      f"stayed the same size in every other respect")
assert g5 > 10 * g1
'''),

        # ================================================ CURVES
        md("""
## 3 · Learning curves, and validation curves

A **learning curve** puts the number of training *rows* on the x-axis, and
answers a question the degree sweep cannot: *would more data help?*

Note `shuffle=True`. Without it the sub-samples are the first $n$ rows of the
frame, in whatever order the frame happens to be in.

⏱ **about 15 seconds.**
"""),
        prompt(
            label="⏱ 15 s — would more data help?",
            input="training subsets from 12% to 100% of the 712 rows, at "
                  "degrees 1 and 5",
            output="training and held-out log loss against the number of rows",
            constraint="`shuffle=True` — without it the sub-samples are the "
                       "first n rows of the frame, in whatever order the frame "
                       "happens to be in",
            check="eight sizes, the largest 640; degree 1 ends with the two "
                  "curves 0.078 apart and degree 5 with them 1.625 apart. "
                  "Curves that have met mean more data changes nothing; curves "
                  "still far apart and falling mean starved of rows",
            **{"try": "read the first point. At 12% of 712 that is 76 "
                      "passengers, and a held-out log loss of 9.9 from 76 rows "
                      "is noisy enough to read as a trend when it is not."}),
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
    print(f"degree {deg}: at {c['n'][0]:3d} rows held-out {c['valid'][0]:6.3f};"
          f"  at {c['n'][-1]} rows train {c['train'][-1]:.3f} "
          f"held-out {c['valid'][-1]:.3f}, "
          f"gap {c['valid'][-1] - c['train'][-1]:.3f}")
'''),
        prompt(
            label="two learning curves, one y-axis",
            input="the learning curves at degrees 1 and 5",
            output="both panels, sharing a y-axis",
            constraint="`sharey=True` — the panels are being compared, and "
                       "independent y-axes would make a gap of 1.6 and a gap of "
                       "0.08 look alike",
            check="the left panel's two curves have visibly met and the right "
                  "panel's have not. Those are the only two shapes there are, "
                  "and their remedies are opposites",
            **{"try": "`sharey=False`. Both panels now look like the same "
                      "picture, which is the mistake the constraint exists to "
                      "prevent."}),
        code('''
fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
for ax, deg, title in zip(axes, (1, 5),
                          ("Degree 1 - the curves have met",
                           "Degree 5 - the gap is still open")):
    c = lc[deg]
    ax.plot(c["n"], c["train"], "o-", color="#0b3d62", label="training folds")
    ax.plot(c["n"], c["valid"], "s-", color="#c0392b", label="held-out folds")
    ax.set_title(title)
    ax.set_xlabel("training passengers")
axes[0].set_ylabel("log loss")
axes[0].legend(loc="upper right")
plt.show()

print("There are 891 passengers on the Titanic. There will never be more,")
print("so rows are not a knob we can turn and the only repair is to shrink")
print("the model. That is section 5.")
'''),

        # ================================================ THREE FAULTS
        md("""
## 4 · Three faults in one model

Degree 5 scored worse than the anchor. Held-out accuracy at that degree was
about 76%, which is poor but not absurd — the two metrics disagree because
$-\\log(0.01) \\approx 4.6$, so one confident mistake contributes more than
twenty ordinary ones.

**Fault 1** is the variance you just measured: 143 columns from 712 rows is five
rows per weight.

**Fault 2** is the convergence warning the previous lecture told you to read.
"""),
        prompt(
            label="the warning, read rather than silenced",
            input="each degree, fitted once with warnings captured",
            output="convergence, iterations used, and the largest absolute "
                   "coefficient",
            constraint="`record=True` with `simplefilter('always')` — the cell "
                       "exists to READ a warning, so suppressing it defeats the "
                       "purpose",
            check="degrees 1 to 3 converge and 4 to 6 do not, and the largest "
                  "weight jumps from about 6 to about 19 at the transition. A "
                  "weight of 19 in log-odds is a probability indistinguishable "
                  "from 0 or 1",
            **{"try": "refit degree 4 with `max_iter=8000`. It still does not "
                      "converge and the largest weight is larger. The optimiser "
                      "is not slow — it is looking for something that does not "
                      "exist."}),
        code('''
sep = {}
for deg in DEGREES:
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        m = pipeline(degree=deg).fit(X_train, y_train)
    sep[deg] = dict(converged=not any("converge" in str(x.message).lower()
                                      for x in w),
                    n_iter=int(np.max(m[-1].n_iter_)),
                    max_coef=float(np.abs(m[-1].coef_).max()))
    print(f"degree {deg}: converged {str(sep[deg]['converged']):5s}  "
          f"iterations {sep[deg]['n_iter']:5d}  "
          f"largest |theta| {sep[deg]['max_coef']:6.2f}")

assert sep[1]["converged"] and sep[3]["converged"]
assert not sep[4]["converged"], "degree 4 is where it stops arriving"
'''),
        md("""
If some $\\boldsymbol\\theta$ separates the two classes in the expanded space
then $\\sigma(c\\,\\boldsymbol\\theta^{\\intercal}\\mathbf{x})$ tends to 0 or 1
as $c \\to \\infty$, so the training log loss falls monotonically towards zero as
$\\lVert\\boldsymbol\\theta\\rVert \\to \\infty$. **The infimum is never
attained**, so there is no minimum for `lbfgs` to find and raising `max_iter`
cannot help.

**Fault 3** is the column you engineered yourself:
`FamilySize = SibSp + Parch + 1`, exactly, on every row.
"""),
        prompt(
            label="the dependence, as an eigenvalue",
            input="SibSp, Parch, FamilySize, Age and Fare, imputed and scaled",
            output="the eigenvalues of XᵀX and its condition number",
            constraint="use `eigvalsh`, not `eigvals` — XᵀX is symmetric, and "
                       "the general routine returns complex numbers with tiny "
                       "imaginary parts that then have to be explained away",
            check="the smallest eigenvalue is below 1e-9: the dependence shows "
                  "up as a zero, not merely as a small number. A zero "
                  "eigenvalue of XᵀX is a direction in which the coefficients "
                  "can move without changing a single prediction",
            **{"try": "drop FamilySize from `cols`. The smallest eigenvalue "
                      "jumps by twelve orders of magnitude, and the condition "
                      "number with it."}),
        code('''
cols = ["SibSp", "Parch", "FamilySize", "Age", "Fare"]
Z = make_pipeline(SimpleImputer(strategy="median"),
                  StandardScaler()).fit_transform(X_train[cols])
G   = Z.T @ Z
eig = np.linalg.eigvalsh(G)

print(f"eigenvalues of XtX: {np.array2string(eig, precision=2)}")
print(f"condition number at alpha = 0: {np.linalg.cond(G):.2e}")
assert eig.min() < 1e-9, "the dependence should show as a zero eigenvalue"
'''),
        md("""
If $\\mathbf{X}^{\\intercal}\\mathbf{X}\\,\\mathbf{v} = \\lambda\\mathbf{v}$ then
$(\\mathbf{X}^{\\intercal}\\mathbf{X} + \\alpha\\mathbf{I})\\mathbf{v} =
(\\lambda + \\alpha)\\mathbf{v}$: same eigenvectors, every eigenvalue moved up by
exactly $\\alpha$. $\\mathbf{X}^{\\intercal}\\mathbf{X}$ is positive
semi-definite, so $\\lambda + \\alpha \\ge \\alpha > 0$ and **the matrix is
invertible for every $\\alpha > 0$**.

That is the question Lecture 2 left open when it derived
$\\hat{\\boldsymbol\\theta} = (\\mathbf{X}^{\\intercal}\\mathbf{X})^{-1}
\\mathbf{X}^{\\intercal}y$ and asked what to do when the inverse does not exist.
Those three lines are the whole answer.
"""),
        prompt(
            label="what alpha does to the spectrum",
            input="the same eigenvalues",
            output="the condition number at four values of alpha, beside its "
                   "bound",
            constraint="compute it from the SHIFTED eigenvalues, not by "
                       "refitting — the claim is that ridge moves every "
                       "eigenvalue up by exactly alpha and leaves the "
                       "eigenvectors alone, and refitting would hide that",
            check="alpha = 1 brings a condition number of 9e14 down to about "
                  "1800. The bound (lmax+alpha)/alpha does not mention lmin at "
                  "all, which is why ridge works on a singular design with no "
                  "column-dropping first",
            **{"try": "alpha = 1e-6. The condition number falls by six orders "
                      "of magnitude and the fit is barely penalised — the "
                      "conditioning repair is far cheaper than the statistical "
                      "one."}),
        code('''
for alpha in (0.0, 1e-6, 1e-3, 1.0):
    if alpha == 0.0:
        print(f"alpha = {alpha:<8g} condition number {np.linalg.cond(G):.3e}")
    else:
        k = (eig.max() + alpha) / (eig.min() + alpha)
        print(f"alpha = {alpha:<8g} condition number {k:.3e}   "
              f"(bound {(eig.max() + alpha) / alpha:.3e})")

k1 = (eig.max() + 1.0) / (eig.min() + 1.0)
print(f"\\nAt alpha = 1 the condition number falls to {k1:.0f}.")
print("The dependence is not repaired. It is dominated.")
assert k1 < 2000
'''),

        # ================================================ REGULARISATION
        md("""
## 5 · Four repairs

$$J(\\boldsymbol\\theta) = \\underbrace{L(\\boldsymbol\\theta)}_{\\text{fit}}
  + \\underbrace{\\alpha\\,\\Omega(\\boldsymbol\\theta)}_{\\text{stay small}}$$

Ridge takes $\\Omega = \\tfrac12\\lVert\\mathbf{w}\\rVert_2^2$ and has the same
closed form as Lecture 2 with one term added. Lasso takes
$\\Omega = \\lVert\\mathbf{w}\\rVert_1$, whose subgradient at zero is the whole
interval $[-1, 1]$ — which is why weights arrive at exactly zero and stay.
Elastic net mixes them.

**Two traps in the API.** Scikit-learn takes `C = 1/α`, so *small* `C` is
*strong* regularisation; and its default is `C=1.0, penalty="l2"`, so logistic
regression is regularised unless you say otherwise. The previous lecture set
`C=1e6` on purpose, to have something to repair.
"""),
        prompt(
            label="the penalty grid, and a solver trap",
            input="a log-spaced grid of C, and three penalty configurations",
            output="the grids and the penalty dictionary, nothing fitted yet",
            constraint="saga for the L1 rows, not liblinear — liblinear "
                       "implements the intercept as a synthetic constant column "
                       "and penalises its weight like any other, so at C = 0.001 "
                       "it fits an intercept of exactly 0.0000 where saga fits "
                       "4.90, and this lecture states that the bias term is not "
                       "penalised",
            check="C = 1/alpha, so the grid runs from strong regularisation on "
                  "the left to none on the right. Before fitting anything, say "
                  "which end you expect the minimum at: at 143 columns and 712 "
                  "rows it is the strong end",
            **{"try": "swap saga for liblinear on the lasso row and print "
                      "`clf.intercept_` at C = 0.001. If it is suspiciously "
                      "near zero, the solver is penalising the bias term and "
                      "every coefficient is compensating for it."}),
        code('''
CS_FULL  = np.logspace(-4, 4, 17)      # what the deck sweeps
CS_SHORT = CS_FULL[:7]                 # up to C = 0.1

# Ridge is cheap (lbfgs), so it runs the whole grid and shows the weak-penalty
# tail. Coordinate descent at degree 5 costs about the same at every weak
# penalty, and both saga curves are flat above C = 0.01, so stopping at C = 0.1
# leaves their minima exactly where they were.
PENALTIES = {
    "ridge":   (CS_FULL,  dict(penalty="l2", solver="lbfgs")),
    "lasso":   (CS_SHORT, dict(penalty="l1", solver="saga", max_iter=5000)),
    "elastic": (CS_SHORT, dict(penalty="elasticnet", solver="saga",
                               l1_ratio=0.5, max_iter=3000)),
}
print(f"ridge grid   {len(CS_FULL)} values, {CS_FULL[0]:g} to {CS_FULL[-1]:g}")
print(f"saga grids   {len(CS_SHORT)} values, {CS_SHORT[0]:g} to {CS_SHORT[-1]:g}")
'''),
        md("""
⏱ **a minute or two on a laptop, three to four minutes on Colab.** This is the
slowest cell in the notebook.
"""),
        prompt(
            label="⏱ 90 s — three penalties, on the degree-5 model that failed",
            input="degree 5, the C grids, ridge / lasso / elastic net",
            output="the best C per penalty, its cross-validated log loss, and "
                   "how many weights survive",
            constraint="count the NON-ZERO weights as well as the score — the "
                       "difference between ridge and lasso is not visible in "
                       "the score alone",
            check="lasso zeroes something and ridge zeroes nothing: an L1 run "
                  "that leaves 143 of 143 weights non-zero did not apply an L1 "
                  "penalty. Ridge should land near 0.71 and both saga penalties "
                  "near 0.68, all far below the unregularised 1.92",
            **{"try": "widen `CS_SHORT` to `np.logspace(-8, 4, 25)`. Does the "
                      "lasso minimum move? If it does, the number on the slide "
                      "was an artefact of where we stopped looking — which is "
                      "item four of the checklist."}),
        code('''
reg = {}
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    for name, (grid, kw) in PENALTIES.items():
        scores = [float(-cross_val_score(pipeline(degree=5, C=float(C), **kw),
                                         X_train, y_train, cv=cv,
                                         scoring="neg_log_loss",
                                         n_jobs=NJ).mean())
                  for C in grid]
        j = int(np.argmin(scores))
        # The non-zero count is only reported at the winner, so fit once here
        # rather than at every C: a saga fit at degree 5 costs as much as a
        # whole cross-validation, and 24 of them would double this cell.
        best_fit = pipeline(degree=5, C=float(grid[j]), **kw).fit(X_train, y_train)
        nnz = int((np.abs(best_fit[-1].coef_[0]) > 1e-8).sum())
        reg[name] = dict(C=float(grid[j]), log_loss=scores[j], nnz=nnz,
                         grid=np.asarray(grid), scores=scores)
        print(f"{name:8s} best C = {grid[j]:.5g}   log loss {scores[j]:.3f}   "
              f"non-zero weights {nnz:3d} of 143   "
              f"(removes {sweep[5]['valid'] - scores[j]:.3f})")

assert reg["lasso"]["nnz"] < 143, "lasso must zero something out"
assert reg["ridge"]["nnz"] == 143, "ridge shrinks everything and zeroes nothing"
'''),
        prompt(
            label="the three validation curves, on one axis",
            input="the sweep",
            output="cross-validated log loss against C for each penalty, with "
                   "the unregularised degree-5 and degree-2 scores drawn as "
                   "horizontal lines",
            constraint="put BOTH baselines on the plot — a comparison between "
                       "three repairs and no baseline is not a comparison",
            check="every curve is far below the degree-5 line and none of them "
                  "reaches the degree-2 line. Read the ENDS of a validation "
                  "curve first: a minimum on the edge of the range means the "
                  "range was too narrow",
            **{"try": "drop the degree-2 line. The picture now says "
                      "regularisation was a triumph, and it was not."}),
        code('''
fig, ax = plt.subplots(figsize=(7.8, 3.6))
for name, colour in (("ridge", "#0b3d62"), ("lasso", "#c0392b"),
                     ("elastic", "#6c3483")):
    ax.semilogx(reg[name]["grid"], reg[name]["scores"], "o-", color=colour,
                label=name)
ax.axhline(sweep[5]["valid"], ls="--", color="#16212b")
ax.text(1e-4, sweep[5]["valid"],
        f" degree 5, no penalty: {sweep[5]['valid']:.2f}", va="bottom",
        color="#16212b")
ax.axhline(sweep[2]["valid"], ls=":", color="#14663a")
ax.text(1e-4, sweep[2]["valid"],
        f" best unregularised degree: {sweep[2]['valid']:.3f}", va="bottom",
        color="#14663a")
ax.set_xlabel("C = 1/alpha   (larger C means weaker regularisation)")
ax.set_ylabel("cross-validated log loss")
ax.legend()
plt.show()
'''),
        prompt(
            label="read that table twice",
            input="the sweep and the three tuned penalties",
            output="all four numbers in one column, with degree 2 at the bottom",
            constraint="put the unregularised degree-2 model in the SAME list "
                       "— it is the baseline the three repairs have to beat",
            check="no penalty beats degree 2. The decomposition predicted this "
                  "twenty minutes ago: squared bias at degree 1 was under 0.01, "
                  "so there was never much bias for extra capacity to buy back",
            **{"try": "read the assert message. It says what to do if the "
                      "assert ever fires, which is what separates a tripwire "
                      "from a test."}),
        code('''
print(f"degree 5, no penalty      {sweep[5]['valid']:.3f}")
for name in PENALTIES:
    print(f"degree 5, {name:8s}        {reg[name]['log_loss']:.3f}")
print(f"degree 2, no penalty      {sweep[2]['valid']:.3f}   <- still the best")
print("\\nRegularisation is a repair, not an upgrade.")

assert min(reg[n]["log_loss"] for n in PENALTIES) > sweep[2]["valid"], \\
    "if a penalty ever beats degree 2 here, the story changes and you must say so"
'''),
        md("""
### Coefficient paths — ridge shrinks, lasso selects

⏱ **about 30 seconds.** Thirteen values of `C` from $10^{-3}$ to $10^{3}$, which
are the endpoints the slides use.
"""),
        prompt(
            label="⏱ 30 s — the coefficient paths",
            input="degree 5, ridge and lasso, 13 values of C from 1e-3 to 1e3",
            output="every weight against C for both penalties, and the count of "
                   "non-zero weights for lasso",
            constraint="a log x-axis and a shared y-range, so the two panels "
                       "are comparable — the whole claim is about the "
                       "difference in shape",
            check="ridge keeps 143 non-zero weights at every C, while lasso "
                  "rises from 5 at the strong end to 142 at the weak end. If "
                  "your lasso count never changes, check the solver",
            **{"try": "count how many lasso weights are below 1e-3 rather than "
                      "exactly zero. Ridge now looks sparse too — 'exactly "
                      "zero' is the whole difference, and it is a property of "
                      "the subgradient at the origin, not of the size of the "
                      "numbers."}),
        code('''
PATH_CS = np.logspace(-3, 3, 13)
paths = {}
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    for name, kw in (("ridge", dict(penalty="l2", solver="lbfgs")),
                     ("lasso", dict(penalty="l1", solver="saga",
                                    max_iter=5000))):
        A = np.array([pipeline(degree=5, C=float(C), **kw)
                      .fit(X_train, y_train)[-1].coef_[0].copy()
                      for C in PATH_CS])
        paths[name] = dict(coef=A,
                           nnz=[int((np.abs(r) > 1e-8).sum()) for r in A])

print(f"ridge non-zero weights: {paths['ridge']['nnz'][0]} at C = 1e-3, "
      f"{paths['ridge']['nnz'][-1]} at C = 1e3")
print(f"lasso non-zero weights: {paths['lasso']['nnz'][0]} at C = 1e-3, "
      f"{paths['lasso']['nnz'][-1]} at C = 1e3")

fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
for ax, name, colour in ((axes[0], "ridge", "#0b3d62"),
                         (axes[1], "lasso", "#c0392b")):
    for j in range(paths[name]["coef"].shape[1]):
        ax.semilogx(PATH_CS, paths[name]["coef"][:, j], color=colour, lw=0.9,
                    alpha=0.5)
    ax.axhline(0, color="#16212b", lw=1.2)
    ax.set_xlabel("C = 1/alpha"); ax.set_title(name); ax.set_ylim(-6, 6)
axes[0].set_ylabel("coefficient")
plt.show()

assert paths["ridge"]["nnz"] == [143] * len(PATH_CS)
assert paths["lasso"]["nnz"][0] < paths["lasso"]["nnz"][-1]
'''),
        md("""
### Early stopping — keeping the weights small by not going far

Gradient descent starts at $\\boldsymbol\\theta = \\mathbf{0}$, the smallest
possible model, and moves outward. Held-out error falls, reaches a minimum, and
then rises.

`warm_start=True` with `max_iter=1` is what makes each `fit` call one epoch of
the *same* fit. `penalty=None` is deliberate — early stopping should be the only
regularisation in the room. The transform is fitted on the training part only,
**outside** the loop; refitting it inside would leak, 500 times over.
"""),
        prompt(
            label="early stopping, one epoch at a time",
            input="a 75/25 split of the training rows, degree 5, no penalty at "
                  "all",
            output="training and validation log loss at each of 500 epochs, and "
                   "the epoch that minimises the validation curve",
            constraint="fit the transform on the training part ONLY and OUTSIDE "
                       "the loop — refitting it inside would leak five hundred "
                       "times over",
            check="the minimum is interior, near epoch 105, and the validation "
                  "curve at epoch 500 is about 1.27 worse than at the minimum. "
                  "A minimum at epoch 500 would mean there was nothing to stop "
                  "early",
            **{"try": "leave `penalty` at its default instead of None. Early "
                      "stopping is then not the only regulariser in the room "
                      "and the experiment measures two things at once."}),
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
print(f"best epoch {best + 1}: train {train_curve[best]:.3f}   "
      f"validation {valid_curve[best]:.3f}")
print(f"epoch 500:      train {train_curve[-1]:.3f}   "
      f"validation {valid_curve[-1]:.3f}")
print(f"regret for not stopping: {valid_curve[-1] - valid_curve[best]:.3f}")

fig, ax = plt.subplots(figsize=(7.5, 3.2))
ax.plot(train_curve, color="#0b3d62", label="training subset")
ax.plot(valid_curve, color="#c0392b", label="validation subset")
ax.axvline(best, ls="--", color="#14663a")
ax.set_xlabel("epoch"); ax.set_ylabel("log loss"); ax.legend()
plt.show()

print("\\nThese are bad numbers in absolute terms: plain SGD at a constant")
print("learning rate on 143 correlated columns is a poor optimiser. This is")
print("the SHAPE, not a competitive model - and the stopping epoch is a")
print("hyperparameter chosen on held-out rows like any other.")
assert best + 1 < 500, "the minimum must be interior or there is nothing to see"
'''),

        # ============================== CHOOSING IT HONESTLY
        md("""
## 6 · Choosing $\\alpha$ is choosing a model

Every repair above needs a value of `C`, and choosing one **consumes whatever
data it looks at**. Here is the version of that loop which does not notice:
"""),
        prompt(
            label="the loop that selects on the rows it then reports",
            input="17 values of C at degree 3",
            output="the best C and its test log loss",
            constraint="score each candidate on the test set and keep the "
                       "winner — this is the failure being measured, not the "
                       "recommendation",
            check="count reads of the test set, not writes. Seventeen reads "
                  "select a model, and the reported score is then a minimum "
                  "over seventeen noisy estimates — biased downward even when "
                  "every estimate is individually unbiased",
            **{"try": "print all seventeen test scores. Their spread is the "
                      "noise the minimum is being taken over, and it is far "
                      "larger than the difference between neighbouring C "
                      "values."}),
        code('''
best_C, best_score = None, np.inf
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    for C in CS_FULL:
        m = pipeline(degree=3, C=float(C)).fit(X_train, y_train)
        score = log_loss(y_test, m.predict_proba(X_test)[:, 1], labels=[0, 1])
        if score < best_score:
            best_C, best_score = float(C), score

print(f"best C = {best_C:g}, log loss = {best_score:.4f}")
print("Seventeen models read the test set. Reading is enough.")
'''),
        md("""
⏱ **about 20 seconds.** Twenty resplits. On each: run the loop above, and also
the honest version that picks `C` by 5-fold cross-validation *inside* the
training part. Score both on the same held-out rows, so the difference comes
from the selection procedure and nothing else.
"""),
        prompt(
            label="⏱ 20 s — measure the optimism over 20 resplits",
            input="20 resplits of the whole labelled frame, 17 candidates each",
            output="the dishonest score and the honest score on the same "
                   "held-out rows, with the spread and the sign count",
            constraint="score BOTH choices on the SAME rows — the difference "
                       "has to come from the selection and nothing else",
            check="the optimism is positive, because it is one-sided by "
                  "construction: about 0.019 of log loss, roughly 4%, against a "
                  "split-to-split spread of 0.045. Report how often the two "
                  "procedures picked the same C — when they agree the leak "
                  "costs nothing on that split",
            **{"try": "cut the grid to three values of C. The optimism "
                      "shrinks. The size of the error scales with how hard you "
                      "looked, which is why a randomised search over a thousand "
                      "configurations is where this becomes dangerous."}),
        code('''
# Resplit the frame in the row order the split above produced, so these 20
# splits are the ones the slides report. train_test_split is order-sensitive.
X_all = pd.concat([X_train, X_test])
y_all = pd.concat([y_train, y_test])


def one_split(seed):
    A2, B2, ya, yb = train_test_split(X_all, y_all, test_size=0.2,
                                      random_state=seed, stratify=y_all)
    test_scores, cv_scores = [], []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for C in CS_FULL:
            m = pipeline(degree=3, C=float(C)).fit(A2, ya)
            test_scores.append(log_loss(yb, m.predict_proba(B2)[:, 1],
                                        labels=[0, 1]))
            cv_scores.append(-cross_val_score(
                pipeline(degree=3, C=float(C)), A2, ya, cv=5,
                scoring="neg_log_loss").mean())
    j, k = int(np.argmin(test_scores)), int(np.argmin(cv_scores))
    return test_scores[j], test_scores[k], int(j == k)


rows     = Parallel(n_jobs=NJ)(delayed(one_split)(s) for s in range(20))
reported = np.array([r[0] for r in rows])
honest   = np.array([r[1] for r in rows])
same_C   = sum(r[2] for r in rows)

print(f"chosen on the test set, scored on it: {reported.mean():.3f} "
      f"+/- {reported.std():.3f}")
print(f"chosen by cross-validation:           {honest.mean():.3f} "
      f"+/- {honest.std():.3f}")
print(f"optimism: {(honest - reported).mean():.3f} of log loss "
      f"({100 * (honest - reported).mean() / honest.mean():.1f}%)")
print(f"splits where the dishonest number flatters: "
      f"{int((reported < honest).sum())} of {len(rows)}")
print(f"splits where both procedures picked the same C: {same_C} of {len(rows)}")

assert (honest - reported).mean() > 0, "the bias is one-sided by construction"
'''),
        md("""
**Read both halves.** *It is small* — about 0.02 of log loss against a
split-to-split spread of 0.045, so on one split you would never see it. *It is
real and one-sided* — it never averages away, and it grows with the number of
candidates you try.

The procedure that does not have the problem costs the same:
"""),
        prompt(
            label="the honest version",
            input="the same 17 candidates",
            output="the cross-validated estimate with its fold spread, and one "
                   "test score",
            constraint="GridSearchCV over the whole PIPELINE, so the "
                       "preprocessing is refitted inside every fold and the "
                       "test set is touched exactly once at the end",
            check="two numbers, reported separately, with the fold spread "
                  "beside the first. The second is allowed to be worse than "
                  "the first — a number that has been through a selection is "
                  "not the same kind of object as one that has not",
            **{"try": "pass the bare classifier instead of the pipeline. The "
                      "preprocessing is then fitted once, outside the folds, "
                      "and the cross-validated estimate improves. That "
                      "improvement is the leak."}),
        code('''
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    gs = GridSearchCV(pipeline(degree=3), {"clf__C": CS_FULL},
                      scoring="neg_log_loss", cv=cv,
                      n_jobs=NJ).fit(X_train, y_train)

print(f"chosen C:           {gs.best_params_['clf__C']:g}")
print(f"honest CV estimate: {-gs.best_score_:.3f}  "
      f"(fold sd {gs.cv_results_['std_test_score'][gs.best_index_]:.3f})")
print(f"the test set, once: "
      f"{log_loss(y_test, gs.predict_proba(X_test)[:, 1], labels=[0, 1]):.3f}")
'''),

        # ============================== THE TEST SET
        md("""
## 7 · The test set. Once

179 passengers, and they have been read a great deal: 200 bootstrap draws at six
degrees in section 2, seventeen models in section 6 on purpose, and five more
below. What has *not* happened is any setting being chosen on them and then
kept — section 6's `best_C` is computed in order to be thrown away, and the five
candidates below arrive with their hyperparameters already fixed by
cross-validation. Naming the best of five reported measurements is a report; add
a sixth candidate to improve the winner and it becomes a tuning loop. Five
candidates, every one of whose hyperparameters was fixed before this cell ran.
"""),
        prompt(
            label="the test set, once",
            input="179 test passengers and five candidates with fixed "
                  "hyperparameters",
            output="log loss, Brier and accuracy for each",
            constraint="every hyperparameter must have been fixed BEFORE this "
                       "cell ran — no selection happens here, only measurement",
            check="the winner is degree 2 with no penalty, and scikit-learn's "
                  "own defaults at degree 1 are within a hundredth of it. If "
                  "you find yourself editing this cell after reading its "
                  "output, stop: the test set has now been read twice",
            **{"try": "add a sixth candidate — elastic net at degree 5, tuned. "
                      "You are now selecting among six on the test set, which "
                      "is the failure section 6 measured."}),
        code('''
candidates = {
    "degree 5, no penalty":       pipeline(degree=5),
    "degree 5, ridge tuned":      pipeline(degree=5, C=reg["ridge"]["C"]),
    "degree 5, lasso tuned":      pipeline(degree=5, C=reg["lasso"]["C"],
                                           penalty="l1", solver="saga",
                                           max_iter=5000),
    "degree 1, sklearn defaults": pipeline(degree=1, C=1.0),
    "degree 2, no penalty":       pipeline(degree=2),
}

final = {}
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    for name, m in candidates.items():
        m.fit(X_train, y_train)
        pr = m.predict_proba(X_test)[:, 1]
        final[name] = dict(log_loss=log_loss(y_test, pr, labels=[0, 1]),
                           brier=brier_score_loss(y_test, pr),
                           accuracy=accuracy_score(y_test, pr >= 0.5))
        print(f"{name:30s} log loss {final[name]['log_loss']:.3f}   "
              f"Brier {final[name]['brier']:.3f}   "
              f"accuracy {final[name]['accuracy']:.1%}")

winner = min(final, key=lambda k: final[k]["log_loss"])
print(f"\\nwinner: {winner}")
assert final[winner]["log_loss"] < constant_log_loss
'''),
        prompt(
            label="the numbers to keep, and the headroom",
            input="everything measured",
            output="the anchor, the best cross-validated score, the final test "
                   "score, the improvement over degree 5, and the Brier score "
                   "against the measured floor",
            constraint="put the noise floor beside the final Brier — the gap "
                       "between them is what is actually left to win, and it is "
                       "the most honest thing a report can end with",
            check="the improvement over the degree-5 model is about 1.3 of log "
                  "loss, and about 0.013 of Brier is left on the table in total "
                  "for any model of these columns",
            **{"try": "work out how many test passengers one point of accuracy "
                      "is. 179 rows means it is under two people, which is why "
                      "the accuracy column of the table above supports no "
                      "comparison at all."}),
        code('''
committed = final["degree 5, no penalty"]["log_loss"]
best_ll   = final[winner]["log_loss"]

print(f"anchor - report the base rate to everyone   {constant_log_loss:.3f}")
print(f"best cross-validated log loss (degree 2)    {sweep[2]['valid']:.3f}")
print(f"final, on the test set                      {best_ll:.3f}")
print(f"test accuracy at a 0.5 cut-off              "
      f"{final[winner]['accuracy']:.1%}")
print(f"majority-class accuracy on the test set     {1 - y_test.mean():.1%}")
print(f"\\nimprovement over where the previous lecture ended: "
      f"{committed - best_ll:.3f}")
print(f"cross-validated, we beat the anchor by "
      f"{constant_log_loss - sweep[2]['valid']:.3f} of log loss")
print(f"and the majority class by "
      f"{100 * (sweep[2]['acc'] - (1 - p)):.1f} points of accuracy")
print(f"Brier {final[winner]['brier']:.3f} against a measured floor of "
      f"{noise:.3f}")
print(f"=> about {final[winner]['brier'] - noise:.3f} of Brier score is left "
      f"on the table, in total, for any model of these columns.")

assert best_ll < constant_log_loss, "we must at least beat the anchor"
'''),
        md("""
## 8 · Where we are

**The winner is a model with no repair in it.** Degree 2, no penalty, chosen by
reading a held-out curve — and `LogisticRegression()` with every default left
alone is statistically the same model. Ninety minutes of ridge, lasso, elastic
net and early stopping, and the best system on the test set is two lines anyone
could have written before the previous lecture started.

Report it anyway. The alternative is choosing the interesting model over the
better one.

### The six things to take away

1. Expected squared error splits **exactly** into squared bias, variance and
   irreducible noise — over the training set and the label, never over test
   points. You checked the residual at floating-point noise.
2. The vertical gap between a training curve and a held-out curve **is** the
   variance term, and you watched it grow by a factor of twenty-two across the
   degree sweep.
3. The noise floor is measurable without fitting anything: 0.121 Brier, from
   passengers whose recorded values are identical.
4. Degree 5 failed three ways at once — high variance, no minimiser (separation),
   and no unique minimiser (an exact column dependence).
5. Adding $\\alpha\\mathbf{I}$ repairs the last two for every $\\alpha > 0$,
   which is the question Lecture 2 left open. Ridge shrinks, lasso selects,
   elastic net does both, early stopping does it by not arriving.
6. A hyperparameter chosen on the rows you then report is optimistic,
   one-sided, and the cost grows with the size of the search.

**Before the next lecture:** widen the penalty grid as the notebook slide asks,
and see whether the lasso minimum moves. Then answer the checklist on this
notebook: is the scaler inside the cross-validated pipeline, was every `C`
stated, and how many columns does the degree-5 pipeline produce against how many
rows?
"""),
    ]
    return cells
