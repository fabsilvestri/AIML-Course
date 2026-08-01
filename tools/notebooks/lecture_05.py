#!/usr/bin/env python3
"""
Lecture 5 — Who survives, and how sure are we? Build, Titanic, Géron Ch. 4.

Exports build() -> list[cell]; tools/make_notebooks.py wraps it.

Structure mirrors the deck: brief -> data -> metric -> commitment -> build.
Every import the notebook needs is in its own setup cell; a notebook that only
runs because the previous one is still in the kernel is not reproducible.

The build half deliberately ends somewhere bad. The degree sweep is left at its
worst point, the convergence warnings are left visible, and nothing is
regularised -- Lecture 6 is what repairs all three, and it has nothing to bite
on if this notebook tidies up after itself.

Assertions here are about shapes, counts, ranks and identities. There is
deliberately no assertion that the model classifies any particular passenger
correctly: that is fragile, and in a lecture whose whole argument is that a
hard label hides the decision, it would also be arguing the wrong side.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from make_notebooks import code, header, md, SETUP, SETUP_PROMPT        # noqa: E402
from _prompt import prompt                                # noqa: E402


def build() -> list:
    cells = header(
        5, "Who survives, and how sure are we?", "build", "Chapter 4")

    # ---------------------------------------------------------------- setup
    cells += [
        md("""
## 1 · Setup

The brief: a maritime-safety group wants, for each passenger, a **calibrated
probability** of survival and a **defensible cut-off** for assigning a limited
number of escorts — not a bare label. That one sentence chooses the model
family before we have looked at a single row.
"""),
        SETUP_PROMPT, SETUP,
        prompt(
            label="the imports, all of them, here",
            input="nothing",
            output="every import this notebook uses, in one cell",
            constraint="everything in ONE place at the top — not scattered where first needed",
            left_open="why it matters. A notebook that runs only because the previous one is still in the kernel is not reproducible, and you discover that on the machine of whoever you sent it to.",
            student="importing where needed, which works perfectly until cells are run out of order or the kernel is restarted halfway.",
            catch="Restart-and-run-all before you believe any notebook, including your own. It is the only test of this that exists."),
        code('''
# Every import this notebook needs, in one place. A notebook that only runs
# because a previous one is still in memory is not reproducible.
import tarfile, urllib.request, warnings
from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, brier_score_loss, log_loss)
from sklearn.model_selection import (StratifiedKFold, cross_val_predict,
                                     cross_validate, train_test_split)
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import (OneHotEncoder, PolynomialFeatures,
                                   StandardScaler)

# Not examinable. A free Colab CPU runtime has two cores, so n_jobs=-1 buys
# nothing there, and on a shared machine it makes everything slower.
NJ = 4
'''),

        # ----------------------------------------------------------- the data
        md("## 2 · The data"),
        prompt(
            label="the data, as a function",
            input="the Titanic tarball at github.com/ageron/data",
            output="891 rows and 12 columns, with the shape asserted",
            constraint="a FUNCTION that downloads if absent and reads if present — not a manual download and a hard-coded path",
            check="assert the shape, that Survived exists, and that it holds only 0/1",
            left_open="what happens on a partial download. The tarball is either there or it is not; a truncated one is there, and the shape assert is the only thing between it and forty minutes of confusion.",
            student="downloading by hand and reading `~/Downloads/train.csv`. It works on your machine and nowhere else, which you find out at the demo.",
            catch="delete `datasets/` and re-run. If the cell cannot rebuild its own input from nothing, it is not reproducible, it is just cached."),
        code('''
# A function, not a manual download: the data will change, and you will need
# this on another machine. ~5 s the first time, instant afterwards.
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
assert "Survived" in full.columns
assert full["Survived"].isin([0, 1]).all()
print(f"{len(full)} passengers, {full.shape[1]} columns")
full.head()
'''),
        prompt(
            label="what is missing, counted",
            input="the 891 rows",
            output="every column with at least one missing value, with a percentage",
            constraint="count them — do not eyeball `.head()` and form an impression",
            check="assert the exact dictionary, so the numbers below are pinned to a known file",
            left_open="what to DO about each one. That is the next markdown cell, and the three columns get three different answers because 77% missing, 20% missing and 2 rows missing are not one problem.",
            student="`full.dropna()`, which here throws away 708 of 891 passengers because almost nobody has a cabin recorded. The model then trains on the 183 richest people on the ship.",
            catch="compare rows before and after any drop. If a one-line clean removes most of your data, the line is the problem, not the data."),
        code('''
# --- what is missing, counted rather than eyeballed --------------------------
missing = full.isna().sum()
missing = missing[missing > 0].sort_values(ascending=False)
for col, n in missing.items():
    print(f"{col:10s} {n:4d} missing   ({n / len(full):.1%})")

assert missing.to_dict() == {"Cabin": 687, "Age": 177, "Embarked": 2}
'''),
        md("""
Three different problems, three different answers:

- **`Cabin`** is 77% missing. Do not impute it. But *which deck* a cabin is on
  is the first letter, and "no cabin recorded" is itself informative — third
  class largely had no cabin number. Turn it into a `Deck` column with `U` for
  unknown.
- **`Age`** is 20% missing. Impute the median, **inside the pipeline**, so the
  median is computed per fold and never from the test set.
- **`Embarked`** is 2 rows. Impute the most frequent value and move on.

**The label.** 342 survived of 891 — a base rate of 0.384. Not balanced, not
badly imbalanced either, so accuracy will not collapse the way it did in the
previous application. It will fail for a different reason.
"""),
        prompt(
            label="the base rate",
            input="the label column",
            output="how many survived, and the rate",
            constraint="print it before anything is fitted",
            check="assert 342 survivors",
            left_open="what 0.384 implies. It is not the collapse of the previous application — accuracy will not be useless here — and the lecture is about it failing a different way.",
            student="skipping it because 'Titanic is roughly balanced'. Roughly is not a number, and the anchor computed three cells down is built out of this one.",
            catch="every classification notebook should print its base rate above the first model. It costs one line and it is the denominator of every claim below it."),
        code('''
p_all = full["Survived"].mean()
print(f"survived {int(full['Survived'].sum())} of {len(full)}  "
      f"base rate {p_all:.4f}")
assert int(full["Survived"].sum()) == 342
'''),

        # --------------------------------------------------- what the data says
        md("""
### Before any model: what does the data already say?

Two cross-tabulations you could hand to the stakeholder without fitting
anything. They matter because they are the rule the model has to beat.
"""),
        prompt(
            label="the rule before the model",
            input="sex, class, and the label",
            output="survival rate by sex, and by sex crossed with class",
            constraint="cross the two — the marginal rates alone hide that first-class women and third-class men are the two extremes",
            check="assert the two extremes, above 0.95 and below 0.15",
            left_open="that this table IS a model. It is one you could hand over on paper, it needs no library, and it is the thing anything you fit has to beat.",
            student="going straight to `.fit()`. You then have no idea whether your 0.80 accuracy is skill or whether a two-row lookup table gets there too.",
            catch="if a hand-written rule matches your model, you have not built a model, you have built an expensive way to write down that rule."),
        code('''
by_sex = full.groupby("Sex")["Survived"].mean()
print("survival rate by sex")
for k, v in by_sex.items():
    print(f"  {k:8s} {v:.3f}")

print("\\nsurvival rate by sex and class")
tab = full.groupby(["Sex", "Pclass"])["Survived"].agg(["mean", "size"])
for (sex, cls), row in tab.iterrows():
    print(f"  {sex:8s} class {cls}  {row['mean']:.3f}   n={int(row['size'])}")

# Women in first class and men in third are the two extremes, and the gap
# between them is larger than anything a model will add on top.
assert tab.loc[("female", 1), "mean"] > 0.95
assert tab.loc[("male", 3), "mean"] < 0.15
'''),

        # ------------------------------------------------ feature engineering
        md("""
## 3 · Feature engineering

`Name` looks like free text. It is not — it carries a title, and the title
encodes age, sex and marital status in one token.

```
Braund, Mr. Owen Harris
Cumings, Mrs. John Bradley (Florence Briggs Thayer)
Heikkinen, Miss. Laina
```

The rare titles (Dr, Rev, Col, Countess, …) are individually too small to fit a
weight to, so they are collapsed into `Rare`. `Mlle`/`Ms`/`Mme` are spelling
variants of `Miss`/`Mrs`.
"""),
        prompt(
            label="features out of a name",
            input="the raw columns",
            output="Title, FamilySize, IsAlone and Deck",
            constraint="collapse the rare titles into one level — Dr, Rev, Col and Countess are individually too small to fit a weight to",
            check="assert the title set is exactly the five expected, and that no engineered column is null",
            left_open="that one of these four lines manufactures an exact linear dependence and takes twenty minutes to diagnose in section 9. The markdown under this cell says so and does not say which.",
            student="one-hot encoding all seventeen raw titles, giving the model several columns that are 1 for exactly one passenger. `Deck_T` further down is precisely that failure, surviving the collapse.",
            catch="value_counts on every categorical you create, before encoding it. A level with n=1 is a memorised passenger wearing a coefficient."),
        code('''
def engineer(d):
    d = d.copy()
    d["Title"] = (d["Name"].str.extract(r",\\s*([^\\.]+)\\.", expand=False)
                  .str.strip()
                  .replace({"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"}))
    d["Title"] = d["Title"].where(
        d["Title"].isin(["Mr", "Mrs", "Miss", "Master"]), "Rare")
    d["FamilySize"] = d["SibSp"] + d["Parch"] + 1
    d["IsAlone"]    = (d["FamilySize"] == 1).astype(int)
    d["Deck"]       = d["Cabin"].str[0].fillna("U")
    return d

full = engineer(full)

assert set(full["Title"]) == {"Mr", "Mrs", "Miss", "Master", "Rare"}
assert full["Title"].isna().sum() == 0
assert full["Deck"].isna().sum() == 0
assert full["IsAlone"].isin([0, 1]).all()

print(full["Title"].value_counts().to_dict())
print("\\nsurvival rate by title")
for k, v in full.groupby("Title")["Survived"].agg(["mean", "size"]).iterrows():
    print(f"  {k:8s} {v['mean']:.3f}   n={int(v['size'])}")
'''),
        md("""
> ⚠ **Look at this line and remember it.**
>
> ```python
> d["FamilySize"] = d["SibSp"] + d["Parch"] + 1
> ```
>
> It is defensible feature engineering, it will improve nothing, and it is
> going to break the model in a way that takes twenty minutes to diagnose. We
> come back to it later in this session.
"""),

        # -------------------------------------------------------- split first
        md("""
## 4 · Split first — still

Stratified on the **label** this time, not on a predictor. With 179 test rows
an unstratified draw can move the base rate by several points, and every
downstream number moves with it.
"""),
        prompt(
            label="split first, stratified on the label",
            input="the engineered frame",
            output="712 training and 179 test rows",
            constraint="stratify on y — with 179 test rows an unstratified draw moves the base rate by several points and every number below moves with it",
            check="assert the sizes, that the indices are disjoint, and that the two base rates agree to within a point",
            left_open="that FamilySize, SibSp and Parch are all in NUM together. That is the trap, sitting in plain sight in the first line.",
            student="stratifying on a predictor rather than the label, or forgetting `stratify=` entirely and never checking. The disjointness assert is cheap; the base-rate assert is the one that catches it.",
            catch="assert on the two base rates, not just the two sizes. Sizes are right even when the split is badly unbalanced."),
        code('''
NUM = ["Age", "Fare", "FamilySize", "SibSp", "Parch"]
CAT = ["Pclass", "Sex", "Embarked", "Title", "Deck"]
BIN = ["IsAlone"]

X = full[NUM + CAT + BIN]
y = full["Survived"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)

assert len(X_train) == 712 and len(X_test) == 179
assert len(X_train) + len(X_test) == len(full)
assert set(X_train.index).isdisjoint(X_test.index)
assert set(X_train.columns) == set(X_test.columns)
assert abs(y_train.mean() - y_test.mean()) < 0.01, "stratification failed"

print(f"train rate {y_train.mean():.4f}   test rate {y_test.mean():.4f}")
print("712 training passengers. That is a small dataset, and by the end of")
print("this session the size will be doing visible damage.")
'''),

        # ------------------------------------------------------- the metric
        md("""
## 5 · The metric, and the commitment

**Why accuracy cannot be the headline here.**

- It reads a probability of 0.51 and one of 0.99 as the same answer.
- It cannot be computed at all until a cut-off is chosen — so quoting it hides
  the very decision the brief asked us to make explicit.
- It weighs the two mistakes equally, and the stakeholder does not.

**Log loss** is the metric that matches requirement 1. For labels
$y \\in \\{0,1\\}$ and predicted probabilities $\\hat{p}$:

$$L = -\\frac{1}{m}\\sum_{i=1}^{m}\\Big[y^{(i)}\\log \\hat{p}^{(i)}
      + (1-y^{(i)})\\log(1-\\hat{p}^{(i)})\\Big]$$

It is a **proper scoring rule**: it is minimised, in expectation, by reporting
your true belief. Any attempt to game it by shading probabilities towards 0 or
1 makes it worse. The **Brier score**, the mean squared error of a probability,
is proper too, and bounded — we report both.

### Before committing: what does *nothing* score?
"""),
        prompt(
            label="what nothing scores",
            input="the training labels only",
            output="the accuracy of always predicting 'died', and the log loss and Brier score of always reporting the base rate",
            constraint="compute the constant-predictor log loss from the base rate, not by fitting anything",
            check="assert it equals 0.666 — this number is quoted for the rest of the session",
            left_open="that 0.666 is the entropy of the label in nats. The formula is the same one; the interpretation is what makes it an anchor and not an arbitrary constant.",
            student="anchoring on 0.5 because 'log loss of a coin flip is 0.693'. That is the anchor for a balanced label. This one is 0.384 positive, and its anchor is lower.",
            catch="always report the anchor beside the model. A log loss of 0.45 sounds like nothing until you see that doing no work at all scores 0.666."),
        code('''
p = y_train.mean()

majority_accuracy = 1 - p                  # predict "did not survive" always
constant_log_loss = -(p * np.log(p) + (1 - p) * np.log(1 - p))
constant_brier    = p * (1 - p)

print(f"base rate on the training set   {p:.4f}")
print(f"always 'did not survive'        accuracy {majority_accuracy:.3f}")
print(f"always report p = {p:.3f}          log loss {constant_log_loss:.3f}   "
      f"Brier {constant_brier:.3f}")

assert abs(constant_log_loss - 0.666) < 0.001
assert abs(majority_accuracy - 0.617) < 0.001
'''),
        md("""
### Your anchor: **0.666**

The log loss of a model that has learned **nothing except the base rate**. It
is the entropy of the label, in nats. Anything above it is worse than knowing
only how many people died; the distance below it is the only part you actually
earned.

---

## ★ COMMIT — write this down, on paper, now

```
Metric:
Log loss for a system I would sign off:
Log loss I expect from the model we build today:
Accuracy I expect at my chosen cut-off:
```

You are not guessing in the dark. Reporting the base rate to everyone scores
0.666, and the sex-and-class table above is a rule you could apply by hand.

**Do not run the next cell until you have written four numbers down.** The
whole of the next lecture is scored against them.
"""),

        # ---------------------------------------------------------- the build
        md("""
## 6 · Build the simplest thing that runs

Preprocessing goes **inside** the pipeline, so that cross-validation refits the
imputer, the scaler and the encoder on each training fold and never sees the
held-out one.
"""),
        prompt(
            label="preprocessing inside the pipeline",
            input="the five numeric, five categorical and one binary column",
            output="a fitted ColumnTransformer, and the width it produces",
            constraint="impute and scale INSIDE the pipeline, so cross-validation refits the median per fold and never sees the held-out one",
            check="assert no NaN survives, and print how many raw columns became how many model columns",
            left_open="that 11 raw columns become 28, and that some of those 28 are redundant by construction. Section 9 counts the rank; this cell only counts the columns.",
            student="`SimpleImputer().fit_transform(X)` on its own line before splitting. The median is then computed over the test rows too, and it leaks — silently, and by about the amount that makes your numbers look good.",
            catch="if any preprocessing step is fitted outside a Pipeline, your cross-validation score is optimistic. There is no version of that which is fine."),
        code('''
prep_v1 = ColumnTransformer([
    ("num", make_pipeline(SimpleImputer(strategy="median"),
                          StandardScaler()), NUM),
    ("cat", make_pipeline(SimpleImputer(strategy="most_frequent"),
                          OneHotEncoder(handle_unknown="infrequent_if_exist",
                                        min_frequency=2)), CAT),
    ("bin", "passthrough", BIN),
])

Z = prep_v1.fit(X_train).transform(X_train)
print(f"{X_train.shape[1]} raw columns become {Z.shape[1]} model columns")
assert Z.shape[0] == 712
assert not np.isnan(Z).any(), "the imputer should have left no NaN"
'''),

        # ------------------------------------------- the assistant failure
        md("""
## 7 · The worked assistant failure

> ⚠ **read before running**

The prompt:

> *"Build me a classifier for this and tell me how accurate it is."*

Nothing in that sentence is wrong. Nothing in it mentions a probability, a
cut-off, or the fact that the two mistakes cost different amounts. Here is the
plausible code it returns — it runs, and it prints a number people will quote.
"""),
        prompt(
            label="⚠ what the assistant returns",
            input="'build me a classifier for this and tell me how accurate it is'",
            output="a fitted pipeline and its test accuracy",
            constraint="use `.predict()` — the hard label — because that is what the prompt asked for and the point is what it costs",
            check="assert the output is hard 0/1 labels, which is the defect made visible rather than a test of correctness",
            left_open="which cut-off produced those labels, and who chose it. Nobody did: `predict()` thresholds at 0.5, correct only if a false negative and a false positive cost the same. The stakeholder said 10 to 1.",
            student="reading the accuracy, writing it in the report, and shipping. The prompt is not wrong, the code is not wrong, and the number is a decision about human life taken silently by a library.",
            catch="whenever you see `.predict()` on a problem with unequal costs, ask what threshold it used. If the answer is 'the default', the threshold is unowned."),
        code('''
# ⚠ this is the failure, not the fix
model_weak = Pipeline([("prep", prep_v1),
                       ("clf", LogisticRegression(max_iter=1000))])
model_weak.fit(X_train, y_train)

y_pred = model_weak.predict(X_test)
print(f"accuracy: {accuracy_score(y_test, y_pred):.3f}")

assert y_pred.shape == (179,)
assert set(np.unique(y_pred)) <= {0, 1}, "predict() returns hard labels"
'''),
        md("""
**The review question:** *`predict()` returned a label. What cut-off produced
it, and who chose that cut-off?*

Nobody chose it. `predict()` thresholds the probability at **0.5**, which is
the correct choice if and only if a false negative and a false positive cost
the same. The stakeholder told us they do not: a passenger who drowns unescorted
costs **ten times** an escort assigned to someone who would have survived
anyway.

So `0.5` is not a default we forgot to change. It is a **decision about human
life**, taken silently, by a library, on our behalf.

**Now measure the damage.** The operational rule is: assign an escort to
everyone whose survival probability is *below* the cut-off. Choose the cut-off
on out-of-fold predictions — choosing it on predictions the model has already
seen is scoring a tree on its training rows, one level up.
"""),
        prompt(
            label="the cost of the rule",
            input="the stated 10 : 1 cost ratio",
            output="a function scoring the rule 'escort everyone below probability t'",
            constraint="the rule flags LOW survival probability — the direction is easy to invert and the resulting curve looks plausible upside down",
            left_open="the grid resolution. 97 points across 0.02 to 0.98 is fine here and would not be if the optimum sat in the tail.",
            student="writing `flag = prob > t`, getting a monotone cost curve, and choosing the endpoint. A cost curve with its minimum at the edge of the grid is almost always a sign inversion.",
            catch="sanity-check with a degenerate threshold. At t=0.98 nearly everyone is flagged; if your cost is LOWEST there, the comparison is backwards."),
        code('''
COST_FN, COST_FP = 10.0, 1.0          # stated by the stakeholder

def total_cost(prob, y_true, t):
    """Cost of the escort rule `flag everyone with prob < t`."""
    flag = prob < t
    died = (np.asarray(y_true) == 0)
    return float(COST_FN * (died & ~flag).sum()      # drowned, not escorted
                 + COST_FP * (~died & flag).sum())   # escorted, survived anyway

GRID = np.linspace(0.02, 0.98, 97)
'''),
        md("""
⏱ **about two minutes.** 20 seeds, because one subtraction of two noisy
numbers is not a measurement. For each seed: split, choose the cut-off by cost
on out-of-fold training predictions, then score *both* cut-offs on the held-out
part.
"""),
        prompt(
            label="⏱ 2 min — the default cut-off, measured over 20 seeds",
            input="20 different train/test splits",
            output="accuracy and cost at 0.5 and at the cost-chosen cut-off",
            constraint="choose the cut-off on OUT-OF-FOLD training predictions and score it on the held-out part — choosing it on predictions the model has already seen is training-set scoring one level up",
            check="assert the cost-chosen rule is never beaten on cost, and that it is WORSE on accuracy — both directions matter",
            left_open="why twenty seeds. One subtraction of two noisy numbers is not a measurement, and the claim being made here is a comparison.",
            student="one seed, one comparison, and a conclusion. Here the effect is large enough to survive it — 20 of 20 seeds — but you would not know that from one.",
            catch="count the seeds where the sign goes your way and report that count. '20 of 20' is an argument; a mean difference on its own is not."),
        code('''
def one_seed(seed):
    A, B, ya, yb = train_test_split(X, y, test_size=0.2, random_state=seed,
                                    stratify=y)
    m = Pipeline([("prep", prep_v1),
                  ("clf", LogisticRegression(C=1e6, max_iter=4000,
                                             random_state=RANDOM_STATE))])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p_oof = cross_val_predict(m, A, ya, cv=5,
                                  method="predict_proba", n_jobs=NJ)[:, 1]
        chosen = float(min(GRID, key=lambda t: total_cost(p_oof, ya, t)))
        pb = m.fit(A, ya).predict_proba(B)[:, 1]
    return dict(chosen=chosen,
                acc_half=accuracy_score(yb, (pb >= 0.5).astype(int)),
                acc_best=accuracy_score(yb, (pb >= chosen).astype(int)),
                cost_half=total_cost(pb, yb, 0.5),
                cost_best=total_cost(pb, yb, chosen))

trap = [one_seed(s) for s in range(20)]

ch = np.array([r["cost_half"] for r in trap])
cb = np.array([r["cost_best"] for r in trap])
ah = np.array([r["acc_half"] for r in trap])
ab = np.array([r["acc_best"] for r in trap])

print(f"cut-off 0.50 (what accuracy chooses)  cost {ch.mean():7.2f} "
      f"± {ch.std():.2f}   accuracy {ah.mean():.3f}")
print(f"cut-off chosen on cost                cost {cb.mean():7.2f} "
      f"± {cb.std():.2f}   accuracy {ab.mean():.3f}")
print(f"\\nthe default costs {(ch - cb).mean():.2f} more, "
      f"or {100 * (ch - cb).mean() / cb.mean():.0f}% above the best rule")
print(f"seeds where 0.50 is the worse rule: {int((ch > cb).sum())} of 20")
print(f"mean chosen cut-off: {np.mean([r['chosen'] for r in trap]):.3f}")

assert (ch >= cb).all(), "the cost-chosen rule cannot be beaten on cost"
assert ab.mean() < ah.mean(), "and it is bought with accuracy"
'''),
        md("""
**Read both halves.** The cost-optimal rule is worse on accuracy — by about
eight points — and better on the thing the stakeholder actually said they
cared about, by a factor of nearly three. The default cut-off did not merely
differ; it lost on 20 seeds out of 20.

**The corrected specification:**

> *"Fit a model that outputs calibrated probabilities. Report log loss and
> Brier score, both cross-validated. Do **not** threshold. Then, separately,
> choose a cut-off by minimising 10·FN + 1·FP on out-of-fold predictions, and
> report the accuracy at that cut-off as a consequence rather than as the
> objective."*
"""),

        # --------------------------------------------------- fit and measure
        md("""
## 8 · Fit it properly, and measure three things

`C=1e6` switches the penalty **off**. Scikit-Learn's default is `C=1.0` with
`penalty="l2"` — logistic regression is regularised unless you say otherwise —
and we want the unregularised model here, because repairing it is the whole of
the next lecture.

⏱ **about 20 seconds.**
"""),
        prompt(
            label="⏱ 20 s — fit it properly, three metrics",
            input="the preprocessing block and the training rows",
            output="cross-validated log loss, accuracy and Brier, train and held-out",
            constraint="C=1e6 switches the penalty OFF — scikit-learn regularises logistic regression by default, and the unregularised model is what the NEXT lecture repairs",
            check="assert ten folds were returned and that the model beats the 0.666 anchor",
            left_open="that `LogisticRegression()` with no arguments is already a regularised model. Most people meet that fact for the first time when a coefficient they expected to be huge comes back small.",
            student="fitting the default and calling it 'plain logistic regression'. It is ridge-penalised logistic regression, and the whole diagnosis in section 9 would be partly masked by it.",
            catch="print the per-fold numbers, not only the mean. Ten folds whose spread is 0.08 do not support a claim about a difference of 0.01."),
        code('''
def make_model(prep, C=1e6):
    """A preprocessing block plus an UNregularised logistic regression."""
    return Pipeline([("prep", prep),
                     ("clf", LogisticRegression(C=C, max_iter=4000,
                                                random_state=RANDOM_STATE))])

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    r = cross_validate(make_model(prep_v1), X_train, y_train, cv=cv,
                       scoring=["neg_log_loss", "accuracy", "neg_brier_score"],
                       return_train_score=True, n_jobs=NJ)

print(f"log loss   train {-r['train_neg_log_loss'].mean():.3f}   "
      f"held-out {-r['test_neg_log_loss'].mean():.3f}")
print(f"accuracy   train {r['train_accuracy'].mean():.3f}   "
      f"held-out {r['test_accuracy'].mean():.3f}")
print(f"Brier      train {-r['train_neg_brier_score'].mean():.3f}   "
      f"held-out {-r['test_neg_brier_score'].mean():.3f}")
print(f"\\nper-fold held-out log loss: "
      f"{np.array2string(-r['test_neg_log_loss'], precision=3)}")

assert len(r["test_neg_log_loss"]) == 10, "report folds, not just the mean"
assert -r["test_neg_log_loss"].mean() < constant_log_loss, \\
    "the model must at least beat the anchor"
'''),

        # ------------------------------------------------ the coefficients
        md("""
## 9 · Read the coefficients — and notice something wrong

This is why the brief chose logistic regression. Each weight is a change in
**log-odds**, so $e^{\\theta_j}$ is a multiplicative effect on the odds.
"""),
        prompt(
            label="read the coefficients",
            input="the fitted unregularised model",
            output="the six largest weights, as log-odds and as odds multipliers",
            constraint="strip the ColumnTransformer's `num__`/`cat__` prefixes so the names are readable, and sort by ABSOLUTE weight",
            check="assert the name list and the coefficient vector are the same length — a mismatch here silently mislabels every row of the table",
            left_open="that two entries in this table are wrong in different ways: Sex_female and Sex_male are redundant by construction, and Deck_T is fitted to one passenger.",
            student="reading the table as 'the effect of being female'. It is a solution among infinitely many, and the cell in section 9 proves the coefficients can be shifted arbitrarily without changing a single prediction.",
            catch="`get_feature_names_out()` beside `coef_`, always zipped, never assumed to line up by memory of the column order."),
        code('''
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    m_v1 = make_model(prep_v1).fit(X_train, y_train)

names_v1 = [n.split("__")[-1] for n in m_v1[:-1].get_feature_names_out()]
coefs_v1 = m_v1[-1].coef_[0]

assert len(names_v1) == len(coefs_v1)
for n, c in sorted(zip(names_v1, coefs_v1), key=lambda t: -abs(t[1]))[:6]:
    print(f"{n:16s} {c:+.3f}   odds x{np.exp(c):.3f}")
'''),
        md("""
Two things in that list should stop you.

1. **`Sex_female` and `Sex_male` both have large weights, with opposite
   signs.** There are only two sexes in this dataset. One indicator carries all
   the information the pair can carry; the second is redundant by construction.
2. **`Deck_T` has the largest weight in the model.** Ask how many passengers
   are on deck T before you believe it.
"""),
        prompt(
            label="how many people is that weight fitted to",
            input="the Deck column",
            output="passengers per deck, and the count for deck T specifically",
            constraint="print the whole distribution, not just the suspicious level — the point is that you should have looked before believing any of them",
            check="assert deck T has exactly one passenger",
            left_open="what to do about it. `min_frequency=2` on the encoder would fold it away; the notebook leaves it visible because seeing the largest weight in the model belong to one person is the lesson.",
            student="quoting 'deck T is the strongest predictor of survival' in a report. It is the strongest coefficient, fitted to n=1, and it is noise with a confident sign.",
            catch="for every large coefficient on a categorical level, look up how many rows carry that level. Large weight plus tiny n is memorisation."),
        code('''
print("passengers per deck, whole dataset:")
print(full["Deck"].value_counts().to_dict())
print(f"\\ndeck T: {int((full['Deck'] == 'T').sum())} passenger(s)")
print("The largest weight in the model is fitted to one person.")

assert int((full["Deck"] == "T").sum()) == 1
'''),

        # ------------------------------------------------- what is the shape
        md("""
### Reviewer question 3 — *what is the shape here?*

Count the columns, then count the **rank**. They should be equal.
"""),
        prompt(
            label="count the columns, then count the rank",
            input="the transformed design matrix with an intercept column added",
            output="the number of columns, the matrix rank, and the condition number",
            constraint="add the intercept column before taking the rank — the intercept is what each one-hot block is dependent WITH",
            check="assert 29 columns and rank 23",
            left_open="where the six come from. The next two cells answer it: five one-hot blocks and one feature engineered by hand.",
            student="never computing the rank at all. Nothing errors, the model fits, and the coefficients are quietly meaningless — the condition number is 2.6e+16, which is another way of saying singular.",
            catch="rank against column count, on the design matrix WITH the intercept. The gap is the number of directions your coefficients can move in without changing a prediction."),
        code('''
Z   = m_v1[:-1].transform(X_train)
Z_b = np.c_[np.ones(len(Z)), Z]              # add the intercept column

n_cols = Z_b.shape[1]
rank   = int(np.linalg.matrix_rank(Z_b))
print(f"columns: {n_cols}")
print(f"rank:    {rank}")
print(f"cond:    {np.linalg.cond(Z_b):.2e}")

assert n_cols == 29 and rank == 23, "29 columns carrying 23 of information"
print(f"\\nrank deficiency: {n_cols - rank}")
'''),
        md("""
### Where the six come from

**Five of them are the one-hot blocks.** Every one-hot block sums to 1 on every
row — and so does the intercept column. Five categorical columns, five exact
dependencies.

**The sixth is the one you engineered yourself.**
"""),
        prompt(
            label="the dependency you engineered yourself",
            input="SibSp, Parch and FamilySize",
            output="the largest absolute residual of SibSp + Parch + 1 - FamilySize",
            constraint="assert it is EXACTLY zero — this is not a correlation, it is an identity, and 'highly correlated' would be the wrong word",
            check="5 one-hot blocks + 1 engineered sum must account for the whole deficiency",
            left_open="that the line responsible looked like good practice when it was written, six sections ago, and the markdown at the time said to remember it.",
            student="checking `.corr()` instead. A correlation of 0.98 is a warning; a residual of exactly 0.0 is a proof, and only the second one tells you the minimiser is not unique.",
            catch="when you build a feature as a sum of others, you have created an exact dependence. Combining features is fine; combining them ADDITIVELY alongside their own parts is not."),
        code('''
resid = (X_train["SibSp"] + X_train["Parch"] + 1
         - X_train["FamilySize"]).abs().max()
print(f"max |SibSp + Parch + 1 - FamilySize| = {resid}")
assert resid == 0.0, "this is an exact linear dependence, by construction"

print("\\n5 one-hot blocks + 1 engineered sum = 6. That is the deficiency.")
assert 5 + 1 == n_cols - rank
'''),
        md("""
### Thread 1 said this would happen

> *"If one column is a linear combination of others, the columns are dependent
> and $\\mathbf{X}^{\\intercal}\\mathbf{X}$ is singular."* — and *"when creating
> new combined features, avoid simple weighted sums of existing features."*

A weighted sum with weights $1, 1$ and a constant is still a weighted sum. The
singularity was manufactured by hand, in a feature-engineering cell, and it
looked like good practice at the time.

**The consequence: the minimiser is not unique.** Add any multiple of a
dependency direction to $\\boldsymbol\\theta$ and every prediction is
unchanged. Prove it rather than believing it.
"""),
        prompt(
            label="prove the coefficients are not the data",
            input="the fitted weights and the transformed rows",
            output="the largest difference in the logit between two different coefficient vectors",
            constraint="shift along the dependency direction — add the same constant to Sex_female and Sex_male and subtract it from the intercept",
            check="assert the logits agree to under 1e-9 AND that the vectors genuinely differ — one assert without the other proves nothing",
            left_open="how large the shift can be. Any multiple works, which is the point: the solution set is a line, not a point.",
            student="believing the coefficient table because it came out of a fitted model. The vector you got is an accident of lbfgs, and any story told about 'the effect of being female' is a story about the solver.",
            catch="the pair of asserts. Numerically identical predictions from provably different parameters is what non-identifiability looks like in code."),
        code('''
theta = m_v1[-1].coef_[0].copy()
b0    = float(m_v1[-1].intercept_[0])

v = np.zeros_like(theta)
v[names_v1.index("Sex_female")] = 1.0
v[names_v1.index("Sex_male")]   = 1.0     # the block sums to 1 on every row

shift   = 2.5
logit_a = Z @ theta                + b0
logit_b = Z @ (theta + shift * v) + (b0 - shift)

gap = float(np.abs(logit_a - logit_b).max())
print(f"largest difference in the logit over all 712 rows: {gap:.2e}")
print("Two completely different coefficient vectors. The same model.")

assert gap < 1e-9, "the two parameterisations must be numerically identical"
assert not np.allclose(theta, theta + shift * v), "but the vectors differ"
'''),
        md("""
**So the coefficients are not the data.** They are *a* solution among
infinitely many, and the one you got is an accident of the solver. Any story
you tell about "the effect of being female" reading off this table is a story
about `lbfgs`.
"""),

        # ------------------------------------------------------- the repair
        md("""
## 10 · The repair

Two one-line changes:

- drop `FamilySize` from the numeric block — `SibSp` and `Parch` already carry it;
- `drop="first"` on the encoder, so each categorical block loses one level and
  becomes a set of contrasts against a **reference level**.
"""),
        prompt(
            label="the repair, in two lines",
            input="the same columns, minus FamilySize, with drop='first' on the encoder",
            output="the new column count, rank and condition number",
            constraint="change exactly TWO things, so that the rank moving is attributable",
            check="assert columns equals rank equals 23 — full rank",
            left_open="what drop='first' costs. Every coefficient is now a contrast against a reference level, so the table below reads differently and the reference levels have to be stated.",
            student="dropping FamilySize and stopping, which removes one of the six dependencies and leaves five. Or setting drop='first' and stopping, which leaves one. Half the repair is not a repair.",
            catch="the condition number, before and after: 2.6e+16 to under a hundred. It moves by fourteen orders of magnitude, which is not something you can misread."),
        code('''
NUM2 = ["Age", "Fare", "SibSp", "Parch"]          # FamilySize removed

prep_v2 = ColumnTransformer([
    ("num", make_pipeline(SimpleImputer(strategy="median"),
                          StandardScaler()), NUM2),
    ("cat", make_pipeline(SimpleImputer(strategy="most_frequent"),
                          OneHotEncoder(drop="first",
                                        handle_unknown="infrequent_if_exist",
                                        min_frequency=2)), CAT),
    ("bin", "passthrough", BIN),
])

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    m_v2 = make_model(prep_v2).fit(X_train, y_train)

Z2   = m_v2[:-1].transform(X_train)
Z2_b = np.c_[np.ones(len(Z2)), Z2]
cols2, rank2 = Z2_b.shape[1], int(np.linalg.matrix_rank(Z2_b))

print(f"columns: {cols2}   rank: {rank2}   cond: {np.linalg.cond(Z2_b):.1f}")
assert cols2 == rank2 == 23, "full rank now"
print("\\nFull rank. The minimiser is unique, and the condition number fell")
print("from 2.6e+16 to under a hundred.")
'''),
        prompt(
            label="and it scored better, which was not the reason",
            input="the repaired preprocessing",
            output="held-out log loss, and the six largest contrasts",
            constraint="scoring with a single STRING names its columns test_score, a LIST names them test_<scorer>. Mixing the conventions is a KeyError",
            check="assert Sex_female is gone — one level of each block must have been dropped",
            left_open="that scoring better is a side effect. The repair was for identifiability; had the score gone slightly down it would still have been the right change.",
            student="reporting the improvement as the justification. Next time the score does not improve, the same student concludes the repair was not worth making.",
            catch="state the reference levels out loud. A coefficient table with drop='first' and no statement of what was dropped cannot be read by anyone, including you in a month."),
        code('''
# And it scored better, which was not the reason for doing it.
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    r2 = cross_validate(make_model(prep_v2), X_train, y_train, cv=cv,
                        scoring="neg_log_loss", return_train_score=True,
                        n_jobs=NJ)
# A single string scorer names its columns test_score/train_score; a LIST of
# scorers names them test_<scorer>. Mixing the two conventions up is a
# KeyError, and it is the fifth reviewer question wearing a disguise.
ll_v2 = -r2["test_score"].mean()
print(f"held-out log loss, repaired model: {ll_v2:.4f}")
print(f"anchor:                            {constant_log_loss:.4f}")
assert ll_v2 < constant_log_loss

names_v2 = [n.split("__")[-1] for n in m_v2[:-1].get_feature_names_out()]
print("\\nreference levels (the dropped ones): Pclass 1, female, Embarked C,")
print("Title Master, Deck A. Every coefficient below is relative to those.")
for n, c in sorted(zip(names_v2, m_v2[-1].coef_[0]),
                   key=lambda t: -abs(t[1]))[:6]:
    print(f"  {n:16s} {c:+.3f}   odds x{np.exp(c):.3f}")

assert "Sex_female" not in names_v2, "one level of each block must be gone"
'''),

        # --------------------------------------------------- calibration
        md("""
## 11 · Is it calibrated?

Requirement 1 was a *probability*, so check that the number means what it
claims. Of the passengers given a probability near 0.3, about 30% should have
survived. Use **out-of-fold** predictions: calibration measured on training
rows is measured on data the model has already fitted.

⏱ **about 15 seconds.**
"""),
        prompt(
            label="⏱ 15 s — is the probability a probability",
            input="out-of-fold predicted probabilities for all 712 training rows",
            output="ten bins, each with its mean prediction against its observed rate, and the expected calibration error",
            constraint="OUT-OF-FOLD — calibration measured on the training rows is measured on data the model has already fitted",
            check="assert the shape is 712 and that every value lies in [0, 1]",
            left_open="how many bins. Ten is conventional; with 712 rows the extreme bins hold very few passengers, and their observed rates are correspondingly noisy.",
            student="calling `predict_proba` on X_train directly. The model has seen every one of those rows, so the calibration comes out better than it is, and this is the metric the brief asked for.",
            catch="print n for each bin beside the rates. A bin with n=6 whose observed rate is 0.17 is one passenger, not a calibration failure."),
        code('''
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    p_oof = cross_val_predict(make_model(prep_v2), X_train, y_train, cv=cv,
                              method="predict_proba", n_jobs=NJ)[:, 1]

assert p_oof.shape == (712,)
assert ((p_oof >= 0) & (p_oof <= 1)).all(), "these must be probabilities"

edges = np.linspace(0, 1, 11)
idx   = np.clip(np.digitize(p_oof, edges) - 1, 0, 9)
yv    = y_train.values
ece   = 0.0
for b in range(10):
    sel = idx == b
    if sel.sum():
        pred, obs = p_oof[sel].mean(), yv[sel].mean()
        ece += sel.sum() / len(yv) * abs(pred - obs)
        print(f"[{edges[b]:.1f},{edges[b+1]:.1f})  predicted {pred:.3f}   "
              f"observed {obs:.3f}   n={int(sel.sum())}")
print(f"\\nexpected calibration error: {ece:.3f}")
'''),

        # ------------------------------------------------------- the cut-off
        md("""
## 12 · Now, and only now, the cut-off

The probabilities are the model. The cut-off is a **business decision** taken
on top of them, and it is chosen against the stated 10 : 1 costs — not at 0.5,
and not on the test set.
"""),
        prompt(
            label="now, and only now, the cut-off",
            input="the out-of-fold probabilities and the 10 : 1 costs",
            output="the cost and escort count at 0.5 and at the cost-optimal cut-off",
            constraint="chosen on out-of-fold predictions, never on the test set",
            check="assert the cost-optimal rule flags MORE people than 0.5 — a 10:1 ratio should buy more escorts, and if it buys fewer the sign is inverted somewhere",
            left_open="that the answer is 605 escorts. The number is printed and not commented on; section 12b is what happens when someone finally reads it.",
            student="reading the saving, circling the minimum, and moving on. The minimum is correct and unstaffable, and nothing in this cell says so.",
            catch="whenever a rule outputs a count of PEOPLE, read the count. 605 out of 712 should stop you regardless of what the cost column says."),
        code('''
rows = [dict(t=float(t), cost=total_cost(p_oof, yv, t),
             flagged=int((p_oof < t).sum())) for t in GRID]
best = min(rows, key=lambda r: r["cost"])
half = min(rows, key=lambda r: abs(r["t"] - 0.5))

print(f"cut-off 0.50   cost {half['cost']:7.1f}   escorts {half['flagged']}")
print(f"cut-off {best['t']:.2f}   cost {best['cost']:7.1f}   "
      f"escorts {best['flagged']}")
print(f"\\nsaved by choosing the cut-off deliberately: "
      f"{half['cost'] - best['cost']:.0f}")

assert best["cost"] <= half["cost"]
assert best["flagged"] >= half["flagged"], \\
    "a 10:1 cost ratio should buy more escorts, not fewer"
'''),
        prompt(
            label="the cost curve",
            input="the grid of cut-offs and their costs",
            output="cost against cut-off, with the chosen point and 0.5 both marked",
            constraint="mark BOTH — a curve with only its minimum marked hides how far the default sits from it",
            left_open="that most of this curve is unreachable. Section 12b works out that only 5.2% of these cut-offs can be staffed at all, and a later slide greys the rest out.",
            student="plotting the curve, circling the minimum, and putting it on a slide. It draws the stakeholder's eye to the one region of the curve they cannot buy.",
            catch="a decision plot should show the alternative you are arguing against. Otherwise it is an illustration, not an argument."),
        code('''
fig, ax = plt.subplots(figsize=(7.5, 3.2))
ax.plot([r["t"] for r in rows], [r["cost"] for r in rows], color="#0b3d62")
ax.axvline(best["t"], ls="--", color="#14663a")
ax.axvline(0.5, ls=":", color="#c0392b")
ax.set_xlabel("cut-off"); ax.set_ylabel("total cost, 10·FN + 1·FP")
ax.set_title("The cut-off is a decision, and it has a cost curve")
plt.show()
'''),

        # --------------------------------------- the constraint nobody used
        md("""
## 12b · The constraint the costs cannot see

Read the last two lines of that output again.

The cost-optimal cut-off assigns an escort to **605 of 712 passengers**. The
safety unit has **80 crew**. We have just recommended a policy that needs 525
more people than exist, and reported its cost as 210 — a number nobody can ever
pay, for a plan nobody can ever staff.

Requirement 3 has been on the board since the first fifteen minutes. It got its
own row in the table: *a fixed number of crew*. Every cell since has optimised
against the **cost ratio** and not one has referred to the **capacity**. They
are different constraints. The costs say what a mistake is worth; the capacity
says how many escorts there are to spend, and no price makes a person appear.

This is the most ordinary failure in applied work, and it does not look like a
failure. Nothing errored. The cost curve is correct. The cut-off really is
optimal — for a question the stakeholder did not ask.
"""),
        prompt(
            label="the policy you can actually staff",
            input="the same out-of-fold probabilities, and CREW = 80",
            output="the cost and composition of escorting the 80 passengers "
                   "least likely to survive, beside the unconstrained optimum",
            constraint="the rule must be a RANKING, not a cut-off — take the "
                       "80 lowest probabilities, whatever value the 80th "
                       "happens to have",
            check="exactly 80 escorts are assigned, and the unconstrained rule "
                  "asks for more than 80, so the constraint is shown to bind "
                  "rather than assumed to",
            left_open="what to do if the ranking ties at the boundary. With 712 "
                      "continuous probabilities it will not, but on a coarser "
                      "model it would, and 'take the lowest 80' silently becomes "
                      "'take whichever 80 numpy happened to sort first'.",
            student="applying the cost-optimal cut-off and then truncating the "
                    "list at 80 — which gives the same 80 people here, and gives "
                    "the WRONG 80 the moment the cut-off is not monotone in risk. "
                    "The truncation also hides the fact that a constraint bound "
                    "at all.",
            catch="assert the escort count equals CREW exactly. A rule that "
                  "produces 74 escorts because that is where a grid point fell "
                  "is not a capacity rule, it is a threshold rule that got "
                  "lucky — and it wastes six crew.",
        ),
        prompt(
            label="the policy you can actually staff",
            input="the same out-of-fold probabilities, and CREW = 80",
            output="the cost and composition of escorting the 80 passengers least likely to survive, beside the unconstrained optimum",
            constraint="the rule must be a RANKING, not a cut-off — take the 80 lowest probabilities, whatever value the 80th happens to have",
            check="exactly 80 escorts are assigned, and the unconstrained rule asks for more than 80, so the constraint is shown to bind rather than assumed to",
            left_open="what to do if the ranking ties at the boundary. With 712 continuous probabilities it will not, but on a coarser model it would, and 'take the lowest 80' silently becomes 'take whichever 80 numpy happened to sort first'.",
            student="applying the cost-optimal cut-off and then truncating the list at 80 — which gives the same 80 people here, and the WRONG 80 the moment the cut-off is not monotone in risk.",
            catch="assert the escort count equals CREW exactly. A rule producing 74 escorts because that is where a grid point fell is a threshold rule that got lucky, and it wastes six crew."),
        code('''
CREW = 80          # requirement 3, stated in the brief and unused until now

order = np.argsort(p_oof, kind="stable")   # least likely to survive first
take  = np.zeros(len(p_oof), dtype=bool)
take[order[:CREW]] = True

died = (yv == 0)
cap_tp = int((died & take).sum())           # escorted, would have died
cap_fp = int((~died & take).sum())          # escorted, would have lived
cap_cost = COST_FN * int((died & ~take).sum()) + COST_FP * cap_fp

unc = best                                   # from the cell above
print(f"{'':22s}{'escorts':>9}{'cost':>9}{'reached':>9}")
print(f"{'cost-optimal 0.83':22s}{unc['flagged']:>9}{unc['cost']:>9.0f}"
      f"{(p_oof < unc['t'])[died].sum() / died.sum():>8.1%}")
print(f"{'the 80 you have':22s}{int(take.sum()):>9}{cap_cost:>9.0f}"
      f"{cap_tp / died.sum():>8.1%}")
print(f"\\n{died.sum()} of {len(yv)} passengers did not survive.")
print(f"the 80 escorts reach {cap_tp} of them; {cap_fp} go to people "
      f"who would have lived anyway")
print(f"implied cut-off: {np.sort(p_oof)[CREW - 1]:.3f}  "
      f"(a consequence of the ranking, not a choice)")

assert int(take.sum()) == CREW, "a capacity rule assigns exactly CREW escorts"
assert unc["flagged"] > CREW, \\
    "if the cost-optimal rule fits within capacity there is nothing to teach here"
'''),
        md("""
### 99.1% becomes 16.4%

That is the whole finding, in two numbers.

The unconstrained analysis reported **99.1%** of at-risk passengers reached. The
policy you can actually staff reaches **16.4%**. Same model, same probabilities,
same costs — the only thing added was the constraint that was in the brief all
along. The difference between those two numbers is not a modelling result. It is
the distance between a report and a plan.

**Three things change once capacity binds.**

**The rule stops being a threshold and becomes a ranking.** You are not asking
*is this passenger below 0.83*, you are asking *are they in the worst 80*. Note
what that does to the cut-off: it lands at **0.062**, and nobody chose it — it is
whatever the 80th smallest probability happened to be. Change the passenger mix
and it moves on its own, with the model untouched. Lecture 8 is about exactly
that.

**Calibration stops mattering and ordering starts.** Section 11 worked hard to
check that a probability of 0.3 means 30%. Under a hard cap, any strictly
increasing relabelling of the probabilities gives you the same 80 people and the
same cost. Calibration is what you need to choose a cut-off from costs; ranking
is what you need to spend a fixed budget. **Requirement 1 and requirement 3 want
different things from the same model**, and the brief asked for both.

**Most of the cost curve is scenery.** Only **5.2%** of the cut-offs on that
sweep can be staffed at all. Plotting the full curve and circling its minimum
draws the eye to the one region of it that is unreachable.

### What to report

Not 210. Report **3,678**, say that 80 crew is what makes it 3,678, and put the
unconstrained number beside it as the answer to *what would more crew buy* —
which is a budget question and a good one. A cost curve with the infeasible
region greyed out is a better slide than a cost curve with a circle on its
minimum, because it tells the stakeholder what to go and change.
"""),

        # ------------------------------------------------- push the capacity
        md("""
## 13 · Push it until it breaks

The model is linear in the features. Give it curvature by adding polynomial
terms of the four numeric columns, and record **both** curves — training and
held-out — at every degree.

Squaring a one-hot indicator returns the indicator, so the categorical block is
left alone; expanding it would add exact copies, and we have just spent twenty
minutes removing exact copies.

⏱ **about 30 seconds.** `return_train_score=True` is not on by default and it
is the whole experiment: one curve tells you nothing.
"""),
        prompt(
            label="⏱ 30 s — push it until it breaks",
            input="polynomial expansions of the four numeric columns, degrees 1 to 6",
            output="columns, training log loss, held-out log loss and accuracy at each degree",
            constraint="expand the NUMERIC block only — squaring a one-hot indicator returns the indicator, and we have just spent twenty minutes removing exact copies",
            check="assert training error falls at every degree while held-out error does not — the two asserts together are the experiment",
            left_open="`return_train_score=True` is off by default, and without it this cell measures nothing. One curve cannot show a gap.",
            student="recording only the held-out curve, seeing it wobble, and concluding the model is fine. The training curve is what makes the wobble legible as overfitting.",
            catch="22 columns at degree 1 becomes 143 at degree 5, on 712 rows. Count your columns against your rows before you interpret anything."),
        code('''
# The repaired preprocessing of section 10, with a polynomial expansion of the
# four numeric columns bolted on. NUM2 and prep_v2 already exist.
def poly_model(degree):
    prep = ColumnTransformer([
        ("num", make_pipeline(
            SimpleImputer(strategy="median"), StandardScaler(),
            PolynomialFeatures(degree=degree, include_bias=False)), NUM2),
        ("cat", make_pipeline(SimpleImputer(strategy="most_frequent"),
                              OneHotEncoder(drop="first",
                                            handle_unknown="infrequent_if_exist",
                                        min_frequency=2)), CAT),
        ("bin", "passthrough", BIN)])
    return make_model(prep)

DEGREES = [1, 2, 3, 4, 5, 6]
sweep = {}

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    for d in DEGREES:
        m = poly_model(d)
        r = cross_validate(m, X_train, y_train, cv=cv,
                           scoring=["neg_log_loss", "accuracy"],
                           return_train_score=True, n_jobs=NJ)
        n_cols_d = m[:-1].fit(X_train).transform(X_train).shape[1]
        sweep[d] = dict(cols=n_cols_d,
                        train=float(-r["train_neg_log_loss"].mean()),
                        valid=float(-r["test_neg_log_loss"].mean()),
                        acc=float(r["test_accuracy"].mean()))

print(f"{'deg':>3} {'cols':>5} {'train':>8} {'held-out':>9} {'accuracy':>9}")
for d in DEGREES:
    s = sweep[d]
    print(f"{d:>3} {s['cols']:>5} {s['train']:>8.3f} {s['valid']:>9.3f} "
          f"{s['acc']:>9.1%}")

assert sweep[1]["cols"] == 22 and sweep[5]["cols"] == 143
assert sweep[6]["train"] < sweep[1]["train"], "training error falls forever"
assert sweep[5]["valid"] > sweep[2]["valid"], "held-out error does not"
'''),
        prompt(
            label="both curves, side by side",
            input="the sweep",
            output="train and held-out log loss on one axis, held-out accuracy on the other",
            constraint="both log-loss curves on the SAME axis — the gap between them is the quantity being shown, and it is invisible across two panels",
            left_open="which metric noticed first. Accuracy falls seven points from degree 2 to 5; log loss multiplies by four. Accuracy only sees a prediction cross the cut-off.",
            student="reading the accuracy panel and calling the damage mild. The requirement was for probabilities, and the metric matching the requirement is the one that saw it.",
            catch="degrees 1 and 2 differ by under a hundredth, and the fold-to-fold spread is far larger. They are tied, and reporting 2 as the winner is reporting noise."),
        code('''
fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
axes[0].plot(DEGREES, [sweep[d]["train"] for d in DEGREES], "o-",
             color="#0b3d62", label="training folds")
axes[0].plot(DEGREES, [sweep[d]["valid"] for d in DEGREES], "s-",
             color="#c0392b", label="held-out folds")
axes[0].set_xlabel("polynomial degree"); axes[0].set_ylabel("log loss")
axes[0].legend()
axes[1].plot(DEGREES, [sweep[d]["acc"] for d in DEGREES], "s-", color="#c0392b")
axes[1].set_xlabel("polynomial degree"); axes[1].set_ylabel("held-out accuracy")
plt.tight_layout(); plt.show()

print("The training curve never stops improving.")
print("The held-out curve turns at degree 2 and then falls off a cliff.")
'''),
        md("""
Degrees 1 and 2 differ by well under a hundredth in held-out log loss, and the
fold-to-fold spread is far larger than that. **Those two are a tie**, and
reporting degree 2 as the winner would be reporting noise.

Note which metric saw the damage first. From degree 2 to degree 5, held-out
accuracy falls by about seven points; held-out log loss is multiplied by more
than four. Accuracy only notices when a prediction crosses the cut-off. Log
loss notices a confident prediction becoming a *confidently wrong* one — and
the requirement was for probabilities, so the metric that matches the
requirement is the metric that saw it.

### A warning worth reading rather than silencing
"""),
        prompt(
            label="the warning worth reading",
            input="each degree, fitted once with warnings captured",
            output="whether it converged, how many iterations it took, and the largest coefficient",
            constraint="capture the warnings with `record=True` rather than silencing them — the whole cell exists to read one",
            check="assert degree 1 converges and degree 4 does not, so the transition is pinned",
            left_open="why raising max_iter will not help. That is the next lecture, and the largest-coefficient column is the hint.",
            student="`warnings.simplefilter('ignore')` at the top of the notebook, which makes this entire diagnosis invisible. The convergence warning is the model telling you the minimiser does not exist.",
            catch="two failures with the same shape: rank 23 of 29 means the minimiser is not UNIQUE, non-convergence at degree 4 means it does not EXIST. Both are about the optimisation problem, not the passengers."),
        code('''
conv = {}
for d in DEGREES:
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        mm = poly_model(d).fit(X_train, y_train)
    converged = not any("converge" in str(x.message).lower() for x in w)
    conv[d] = dict(converged=converged,
                   n_iter=int(np.max(mm[-1].n_iter_)),
                   max_coef=float(np.abs(mm[-1].coef_).max()))
    print(f"degree {d}: converged {str(converged):5s}  "
          f"iterations {conv[d]['n_iter']:5d}  "
          f"largest |theta| {conv[d]['max_coef']:6.2f}")

assert conv[1]["converged"], "the simple model converges"
assert not conv[4]["converged"], "degree 4 is where it stops arriving"
'''),
        md("""
The reflex is to raise `max_iter`. **It will not help, and understanding why is
the point** — that is the next lecture.

Two failures today, and they have the same shape:

| Symptom | What it means |
|---|---|
| rank 23 of 29 columns | the minimiser is not **unique** |
| `lbfgs` will not converge at degree 4 | the minimiser does not **exist** |

Both are statements about the optimisation problem, not about the passengers.
Both are repaired by the same one-line change, and you have not seen it yet.

---

## 14 · Where we are

**The test set has not been touched, and will not be until the end of the next
lecture.**
"""),
        prompt(
            label="the numbers to bring back",
            input="everything measured above",
            output="best held-out log loss and accuracy, the anchor, and the chosen cut-off",
            constraint="the anchor goes in the summary, not in the prose above it",
            check="assert the best degree is 1 or 2 — the two simplest models are tied at the top, and an assert that accepted only one of them would be asserting noise",
            left_open="the repair. Nothing here is fixed: the sweep is left at its worst point and the convergence warnings are left visible, because the next lecture has nothing to bite on otherwise.",
            student="tidying up before handing in — regularising, silencing the warnings, dropping the bad degrees. The shape of the failure is the material.",
            catch="photograph this output beside the four numbers you wrote down in section 5. The next lecture is scored against both."),
        code('''
best_d = min(DEGREES, key=lambda d: sweep[d]["valid"])
print(f"held-out log loss, best degree ({best_d})   {sweep[best_d]['valid']:.3f}")
print(f"held-out accuracy at that degree         {sweep[best_d]['acc']:.1%}")
print(f"anchor — report the base rate            {constant_log_loss:.3f}")
print(f"chosen cut-off, from the 10:1 costs      {best['t']:.2f}")

assert best_d in (1, 2), "the two simplest models are tied at the top"
print("\\n★ Write your best log loss on the same sheet of paper, next to what")
print("  you predicted. Bring it to the next lecture, with both curves.")
print("\\n  Do not fix anything. The shape of the failure is the material.")
'''),
    ]
    return cells
