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

from make_notebooks import code, header, md, SETUP        # noqa: E402


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
        SETUP,
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
        code('''
fig, ax = plt.subplots(figsize=(7.5, 3.2))
ax.plot([r["t"] for r in rows], [r["cost"] for r in rows], color="#0b3d62")
ax.axvline(best["t"], ls="--", color="#14663a")
ax.axvline(0.5, ls=":", color="#c0392b")
ax.set_xlabel("cut-off"); ax.set_ylabel("total cost, 10·FN + 1·FP")
ax.set_title("The cut-off is a decision, and it has a cost curve")
plt.show()
'''),

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
