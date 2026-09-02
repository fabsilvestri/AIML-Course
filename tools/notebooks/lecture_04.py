#!/usr/bin/env python3
"""
Lecture 4 — Training models. Titanic, Géron Chapter 4.

Exports build() -> list[cell]; tools/make_notebooks.py wraps it.

Structure mirrors the deck: gradient descent implemented from the derivation and
checked against Lecture 2's closed form, then logistic regression on the Titanic
manifest — the coefficients, the two conditions under which the fit is not well
defined, the calibration check, the polynomial sweep, and softmax.

Every quantity this notebook prints that also appears on a slide is computed the
way `tools/figures_app03.py` computes it, because that script is what wrote
`assets/figures/figures.json` and the slides are read off it: the same split
(random_state=42, stratified on the label), the same two preprocessing blocks,
the same `StratifiedKFold(10)`, the same `C=1e6`. A notebook that reaches the
same conclusion by a different route is a notebook whose numbers cannot be
diffed against the deck.

The one deliberate departure from the deck: the gradient-descent section fits a
synthetic one-feature regression rather than the Titanic columns, because the
point of it is that the iterative answer and the closed-form answer agree to
eight decimals, and that is only a *check* when both are available.

Runs on CPU throughout, in about two minutes.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from make_notebooks import code, header, md, SETUP, SETUP_PROMPT        # noqa: E402
from _prompt import prompt                                # noqa: E402


def build() -> list:
    cells = header(4, "Training models", "", "Chapter 4")

    cells += [
        md("""
Two halves, and the first one is short.

**Gradient descent, implemented from the derivation** — twelve lines — and then
checked against the closed form of Lecture 2, which it must agree with to eight
decimals. Then the same optimiser fits a model that *has* no closed form:
logistic regression on 891 Titanic passengers, where the brief asks for a
probability rather than a label.

Runs on free CPU in about two minutes. The slowest cell says so.
"""),

        # ---------------------------------------------------------- setup
        md("## 1 · Setup"), SETUP_PROMPT, SETUP,
        prompt(
            label="every import, in one place",
            input="nothing",
            output="every name this notebook uses below, imported once",
            constraint="no import anywhere after this cell, so the notebook does "
                       "not depend on a previous one still being in memory",
            check="Runtime → Restart, then run this cell alone. If anything "
                  "below raises NameError, that import belongs here",
            **{"try": "restart the runtime and run the LAST cell first. It "
                      "fails. A notebook that only runs top to bottom is the "
                      "only kind you can trust."}),
        code('''
# Every name used below, imported once. A notebook that only runs because a
# previous one is still in memory is not reproducible.
import tarfile, urllib.request, warnings
from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression, SGDRegressor
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import (StratifiedKFold, cross_val_predict,
                                     cross_val_score, cross_validate,
                                     train_test_split)
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import (OneHotEncoder, PolynomialFeatures,
                                   StandardScaler)

# Not examinable. A free Colab CPU runtime has two cores, so n_jobs=-1 buys
# little there, and on a shared machine it can make everything slower.
NJ = 4
'''),

        # ================================================ PART 1 — DESCENT
        md("""
## 2 · Gradient descent, from the derivation

The derivation gave five things. This section runs all five.

$$\\nabla J(\\boldsymbol\\theta) = \\frac{2}{m}\\mathbf{X}^{\\intercal}
  (\\mathbf{X}\\boldsymbol\\theta - y),
  \\qquad
  \\boldsymbol\\theta \\leftarrow \\boldsymbol\\theta - \\eta\\,\\nabla J$$

and the claims to be checked: the fixed point is the normal equation, the cost
is convex, the iteration converges **iff** $\\eta < 2/\\lambda_{\\max}$ where
$\\lambda_{\\max}$ is the largest eigenvalue of
$\\frac{2}{m}\\mathbf{X}^{\\intercal}\\mathbf{X}$, and one row gives an
*unbiased* estimate of the full gradient.

A one-feature problem with a known answer, so that "did it work" has a
definition rather than a vibe.
"""),
        prompt(
            label="a regression problem whose answer is known",
            input="a seed",
            output="100 instances of y = 4 + 3x + noise, and the design matrix "
                   "with its leading column of ones",
            constraint="build X_b explicitly rather than letting an estimator "
                       "add the intercept — the derivation is about a matrix "
                       "with a ones column in it, and hiding that column hides "
                       "half of step 4",
            check="X_b is (100, 2) and its first column is all ones. The true "
                  "parameters are (4, 3), so anything the cells below produce "
                  "has a number to be near",
            **{"try": "raise the noise from 1.0 to 5.0. The fitted parameters "
                      "move away from (4, 3) — but the iterative and the "
                      "closed-form answers still agree with EACH OTHER to "
                      "eight decimals, because they are solving the same "
                      "system, not estimating the truth."}),
        code('''
rng = np.random.default_rng(RANDOM_STATE)

m_gd = 100
x = rng.uniform(0, 2, size=(m_gd, 1))
y_gd = 4 + 3 * x + rng.standard_normal((m_gd, 1))   # true theta = (4, 3)

X_b = np.c_[np.ones((m_gd, 1)), x]                  # the leading ones column

assert X_b.shape == (100, 2)
assert (X_b[:, 0] == 1).all(), "the intercept column is part of the matrix"
print(f"X_b {X_b.shape}   y {y_gd.shape}")
'''),
        prompt(
            label="the closed form, for something to check against",
            input="X_b and y",
            output="the least-squares parameters from the normal equation",
            constraint="use `lstsq`, not an explicit inverse — the answer is the "
                       "same here and the explicit inverse is what fails first "
                       "on a near-singular design",
            check="two parameters, both within about 0.5 of (4, 3) at this "
                  "noise level; and scikit-learn's LinearRegression must agree "
                  "with them to 1e-10, because it is solving the same system",
            **{"try": "`np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y_gd` instead. "
                      "Identical here. Now imagine a design whose condition "
                      "number is 2.6e+16 — the one this notebook builds in "
                      "section 8."}),
        code('''
# Lecture 2's result. lstsq solves the normal equations through a decomposition
# rather than by forming an inverse, which is what every library does.
theta_closed = np.linalg.lstsq(X_b, y_gd, rcond=None)[0]
print(f"normal equation      theta = {theta_closed.ravel()}")

# The same thing, from the library, as a cross-check on our own algebra.
lin = LinearRegression().fit(x, y_gd)
theta_sklearn = np.r_[lin.intercept_, lin.coef_.ravel()]
print(f"LinearRegression     theta = {theta_sklearn}")

assert np.allclose(theta_closed.ravel(), theta_sklearn, atol=1e-10)
'''),
        prompt(
            label="the batch step, written out",
            input="X_b, y and a learning rate",
            output="the parameter path and the cost at every step",
            constraint="record the COST at each step, not only the final "
                       "parameters — a diverging fit returns numbers, and the "
                       "cost curve is the only cheap way to see which of the "
                       "three regimes you are in",
            check="with eta=0.1 the final parameters agree with the closed form "
                  "to eight decimals. They are solutions of the same linear "
                  "system, so the agreement is exact arithmetic, not luck",
            **{"try": "return after 10 steps instead of 1000. The cost has "
                      "already fallen most of the way; the last 900 steps buy "
                      "the last three decimals."}),
        code('''
def mse(theta):
    """J(theta) = (1/m) ||X theta - y||^2, the cost the derivation minimises."""
    r = X_b @ theta - y_gd
    return float(np.sum(r * r) / m_gd)


def batch_gd(eta, n_steps=1000, theta0=None):
    """Batch gradient descent, exactly steps 4 and 5 of the derivation."""
    theta = np.zeros((2, 1)) if theta0 is None else theta0.copy()
    costs = [mse(theta)]
    for _ in range(n_steps):
        grad  = 2 / m_gd * X_b.T @ (X_b @ theta - y_gd)   # step 4
        theta = theta - eta * grad                        # step 5
        costs.append(mse(theta))
    return theta, np.array(costs)


theta_gd, costs_gd = batch_gd(eta=0.1)
print(f"gradient descent     theta = {theta_gd.ravel()}")
print(f"normal equation      theta = {theta_closed.ravel()}")
print(f"largest disagreement       = {np.abs(theta_gd - theta_closed).max():.2e}")

# Step 6: the fixed point of the iteration IS the normal equation, so the two
# answers are the same answer reached two ways.
assert np.allclose(theta_gd, theta_closed, atol=1e-8)
'''),
        md("""
### The learning rate, and the bound the derivation gives for it

Step 8 said the error contracts in every eigen-direction if and only if

$$0 < \\eta < \\frac{2}{\\lambda_{\\max}},
  \\qquad \\lambda_{\\max} = \\lambda_{\\max}\\!\\left(\\tfrac{2}{m}\\mathbf{X}^{\\intercal}\\mathbf{X}\\right)$$

That is a *prediction*, not a rule of thumb: it names the exact value at which
descent stops working. So compute it, then step over it.
""")]

    cells += [
        prompt(
            label="three learning rates, one of them past the bound",
            input="the Hessian of the cost",
            output="the divergence threshold 2/lambda_max, and the cost curve at "
                   "a rate below it, near it, and just above it",
            constraint="derive the third rate FROM the bound rather than "
                       "picking a large number — the claim being tested is that "
                       "the bound is exact, and a rate of 10 would test nothing",
            check="the first two curves end below their starting cost and the "
                  "third does not. Work out the sign of 1 - eta*lambda_max in "
                  "each case before running: it is above 1 in absolute value "
                  "only for the third",
            **{"try": "set the last rate to 0.99 * eta_max instead of 1.05 *. "
                      "It converges, slowly and by oscillating — the boundary "
                      "is where the derivation says it is."}),
        code('''
# Step 7's Hessian, which for a quadratic cost is constant.
H = 2 / m_gd * X_b.T @ X_b
eigenvalues = np.linalg.eigvalsh(H)

assert (eigenvalues >= 0).all(), "positive semi-definite, so the cost is convex"

lam_max, lam_min = eigenvalues.max(), eigenvalues.min()
eta_max = 2 / lam_max
kappa   = lam_max / lam_min

print(f"eigenvalues of the Hessian  {eigenvalues}")
print(f"divergence threshold 2/lam  {eta_max:.4f}")
print(f"condition number kappa      {kappa:.1f}")
print(f"predicted contraction/step  {(kappa - 1) / (kappa + 1):.4f}")

ETA_SMALL   = 0.02
ETA_GOOD    = 0.10
ETA_DIVERGE = 1.05 * eta_max        # just past the bound, on purpose

for name, eta in [("too small", ETA_SMALL), ("good", ETA_GOOD),
                  ("past the bound", ETA_DIVERGE)]:
    with np.errstate(over="ignore", invalid="ignore"):
        _, c = batch_gd(eta, n_steps=60)
    end = c[-1]
    print(f"{name:16s} eta={eta:6.3f}   cost after 60 steps "
          f"{end:12.4f}" + ("   <-- diverged" if not np.isfinite(end)
                            or end > c[0] else ""))

with np.errstate(over="ignore", invalid="ignore"):
    _, c_div = batch_gd(ETA_DIVERGE, n_steps=60)
assert not np.isfinite(c_div[-1]) or c_div[-1] > c_div[0], \\
    "past 2/lambda_max the cost must not fall"
'''),
        prompt(
            label="the three cost curves, on a log scale",
            input="the three learning rates",
            output="cost against step number for each",
            constraint="log scale on the cost — the diverging curve is otherwise "
                       "several orders of magnitude tall and flattens the other "
                       "two into the axis",
            check="the good rate is a straight line on a log scale, because the "
                  "error contracts by a constant factor per step. That "
                  "straightness is the derivation's contraction factor, drawn",
            **{"try": "plot on a linear scale. You can no longer see that the "
                      "slow rate is still improving, which is the whole "
                      "difference between it and the good one."}),
        code('''
fig, ax = plt.subplots(figsize=(7.5, 3.4))
for name, eta in [("eta = 0.02, too small", ETA_SMALL),
                  ("eta = 0.10, good", ETA_GOOD),
                  (f"eta = {ETA_DIVERGE:.2f} > 2/lam_max", ETA_DIVERGE)]:
    with np.errstate(over="ignore", invalid="ignore"):
        _, c = batch_gd(eta, n_steps=60)
    ax.plot(np.clip(c, 1e-3, 1e12), label=name)
ax.set_yscale("log")
ax.set_xlabel("step"); ax.set_ylabel("cost J (log scale)")
ax.set_title("The learning rate decides which of three things happens")
ax.legend(); plt.show()
'''),
        prompt(
            label="one row is an unbiased gradient",
            input="the design matrix and a parameter vector",
            output="the average over all m single-instance gradients, beside "
                   "the batch gradient",
            constraint="average over EVERY row rather than sampling — the claim "
                       "is about the expectation, and an expectation over a "
                       "uniform draw on m items is exactly the mean of the m "
                       "values",
            check="the two vectors agree to floating-point exactly, not "
                  "approximately. This is an identity, not an approximation: "
                  "step 10 is one line of algebra",
            **{"try": "average over a random 10 rows instead of all 100. Now "
                      "the agreement is only approximate — that gap is the "
                      "variance which forces a learning schedule."}),
        code('''
theta_probe = np.array([[1.0], [1.0]])            # any point will do

grad_batch = 2 / m_gd * X_b.T @ (X_b @ theta_probe - y_gd)

# g^(i) = 2 (theta.x^(i) - y^(i)) x^(i), one row at a time
per_row = np.stack([2 * (X_b[i:i+1] @ theta_probe - y_gd[i:i+1]) * X_b[i:i+1].T
                    for i in range(m_gd)])
grad_mean = per_row.mean(axis=0)

print(f"batch gradient            {grad_batch.ravel()}")
print(f"mean of the m row-wise    {grad_mean.ravel()}")
print(f"largest disagreement      {np.abs(grad_batch - grad_mean).max():.2e}")

assert np.allclose(grad_batch, grad_mean, atol=1e-12), \\
    "E_i[g^(i)] = grad J is an identity, so this must hold to machine precision"
'''),
        prompt(
            label="the library's version of the same loop",
            input="the same 100 instances",
            output="SGDRegressor's parameters beside the closed form",
            constraint="scale the feature inside a pipeline — SGDRegressor does "
                       "descent, and the step count is set by the condition "
                       "number, which scaling is what controls",
            check="the two parameter vectors agree to about a tenth. They do "
                  "NOT agree to eight decimals, because a stochastic solver "
                  "with a schedule stops near the minimum rather than at it",
            **{"try": "`max_iter=10`. The gap widens. Stochastic descent trades "
                      "exactness for cost per step, and that trade is the "
                      "entire reason it exists."}),
        code('''
sgd = make_pipeline(
    StandardScaler(),
    SGDRegressor(max_iter=2000, tol=1e-5, eta0=0.01,
                 random_state=RANDOM_STATE)).fit(x, y_gd.ravel())

# undo the scaling by hand, so the two parameter vectors are comparable
scaler = sgd[0]
slope_scaled = sgd[-1].coef_[0]
slope = slope_scaled / scaler.scale_[0]
intercept = sgd[-1].intercept_[0] - slope * scaler.mean_[0]

print(f"SGDRegressor         theta = [{intercept:.4f} {slope:.4f}]")
print(f"normal equation      theta = {theta_closed.ravel()}")
print("Close, not identical: a stochastic solver stops NEAR the minimum.")

assert abs(slope - theta_closed[1, 0]) < 0.25
'''),
        md("""
### What that section established

| Claim from the derivation | How it was checked |
|---|---|
| the fixed point is the normal equation | the two parameter vectors agree to 1e-8 |
| the cost is convex | every eigenvalue of the Hessian is $\\ge 0$ |
| $\\eta < 2/\\lambda_{\\max}$ | the cost falls below the bound and rises above it |
| one row is an unbiased gradient | the identity holds to machine precision |

Everything from here on uses the same optimiser on a cost that has **no closed
form to check it against**. That is the point of doing it in this order.
"""),

        # ================================================ PART 2 — TITANIC
        md("""
## 3 · The worked example: a probability, not a label

The brief: for each passenger profile, *how likely is this person to get off
unaided* — and the answer has to be explainable afterwards. Three consequences,
and they choose the model before we have seen a row:

- "how likely" is a **probability**, not a label;
- a probability used to allocate must be **calibrated**;
- "explain" means **weights a human can read**.
"""),
        prompt(
            label="the data, as a function",
            input="the Titanic tarball at github.com/ageron/data",
            output="891 rows and 12 columns, with the shape asserted",
            constraint="a FUNCTION that downloads if absent and reads if present "
                       "— not a manual download and a hard-coded path",
            check="891 rows, 12 columns, and Survived holding only 0 and 1",
            **{"try": "delete the `datasets/` directory and re-run. If the cell "
                      "cannot rebuild its own input from nothing it is not "
                      "reproducible, it is cached."}),
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

full = load_titanic()

assert full.shape == (891, 12), f"unexpected shape {full.shape}"
assert full["Survived"].isin([0, 1]).all()
print(f"{len(full)} passengers, {full.shape[1]} columns")
full.head()
'''),
        prompt(
            label="what is missing, and what the label does",
            input="the 891 rows",
            output="every column with at least one missing value as a count and "
                   "a percentage, and the survival counts",
            constraint="count them — do not eyeball `.head()` and form an "
                       "impression",
            check="assert the exact dictionary. 687 of 891 is 77%, so Cabin is "
                  "not a column with gaps, it is a gap with a column; and 342 "
                  "of 891 is a base rate of 0.384",
            **{"try": "`full.dropna()`. 891 rows become 183. A one-line clean "
                      "that removes four fifths of your data is the problem, "
                      "not the data."}),
        code('''
missing = full.isna().sum()
missing = missing[missing > 0].sort_values(ascending=False)
for col, n in missing.items():
    print(f"{col:10s} {n:4d} missing   ({n / len(full):.1%})")

assert missing.to_dict() == {"Cabin": 687, "Age": 177, "Embarked": 2}

n_surv = int(full["Survived"].sum())
n_died = len(full) - n_surv
print(f"\\nsurvived {n_surv} of {len(full)}   did not survive {n_died}")
print(f"base rate {full['Survived'].mean():.4f}   "
      f"({full['Survived'].mean():.1%} survived, "
      f"{1 - full['Survived'].mean():.1%} did not)")
assert (n_surv, n_died) == (342, 549)
'''),
        md("""
Three different problems, three different answers:

- **`Cabin`**, 687 missing — do not impute it; you would be inventing 687 cabin
  numbers. Take the deck letter where there is one and give the rest their own
  level `U`. Note as you do it that `U` will largely mean *third class*: the
  cabin number was recorded for passengers whose ticket carried one, so the
  column is partly a proxy for `Pclass`.
- **`Age`**, 177 missing — impute the median **inside the pipeline**, so it is
  recomputed on every training fold and never sees a held-out row.
- **`Embarked`**, 2 rows — impute the most frequent port. It cannot matter, and
  you should still be able to say what you did.
"""),
        prompt(
            label="the rule before the model",
            input="sex, class, title and the label",
            output="survival rate by sex crossed with class, and by title",
            constraint="cross sex with class — the marginal rates alone hide "
                       "that first-class women and third-class men are the two "
                       "extremes",
            check="first-class women above 0.95 and third-class men below 0.15. "
                  "If a hand-written rule matches your model you have not built "
                  "a model, you have built an expensive way to write that rule "
                  "down",
            **{"try": "group by class alone. The seven-fold gap collapses to a "
                      "three-fold one, because sex is doing most of the work "
                      "and the marginal hides it."}),
        code('''
tab = full.groupby(["Sex", "Pclass"])["Survived"].agg(["mean", "size"])
print("survival rate by sex and class")
for (sex, cls), row in tab.iterrows():
    print(f"  {sex:8s} class {cls}  {row['mean']:.3f}   n={int(row['size'])}")

assert tab.loc[("female", 1), "mean"] > 0.95
assert tab.loc[("male", 3), "mean"] < 0.15
'''),
        md("""
## 4 · Feature engineering

`Name` looks like free text. It is not — it carries a **title**, and the title
carries sex, marital status and, for `Master`, being a boy.

```
Braund, Mr. Owen Harris
Cumings, Mrs. John Bradley (Florence Briggs Thayer)
Heikkinen, Miss. Laina
```

The rare titles (Dr, Rev, Col, Countess, …) are individually too small to fit a
weight to, so they collapse into `Rare`; `Mlle`/`Ms`/`Mme` are spelling variants.
"""),
        prompt(
            label="features out of a name",
            input="the raw columns",
            output="Title, FamilySize, IsAlone and Deck, and the survival rate "
                   "of each title",
            constraint="collapse the rare titles into one level — Dr, Rev, Col "
                       "and Countess are individually too small to fit a weight "
                       "to",
            check="exactly five title levels, none null; and `Master` — 40 boys "
                  "— survives at about 0.575, nearly four times the rate of "
                  "`Mr`, which is the only reason the column is not a "
                  "re-encoding of Sex",
            **{"try": "drop the `.replace({...})` line. `Mlle` becomes its own "
                      "level with n=2, and a level with n=2 is two memorised "
                      "passengers wearing a coefficient."}),
        code('''
def engineer(d):
    d = d.copy()
    d["Title"] = (d["Name"].str.extract(r",\\s*([^\\.]+)\\.", expand=False)
                  .str.strip()
                  .replace({"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"}))
    d["Title"] = d["Title"].where(
        d["Title"].isin(["Mr", "Mrs", "Miss", "Master"]), "Rare")
    # An exact linear function of two columns we already have. Deliberate, and
    # section 8 is where it is paid for.
    d["FamilySize"] = d["SibSp"] + d["Parch"] + 1
    d["IsAlone"]    = (d["FamilySize"] == 1).astype(int)
    d["Deck"]       = d["Cabin"].str[0].fillna("U")
    return d

full = engineer(full)

assert set(full["Title"]) == {"Mr", "Mrs", "Miss", "Master", "Rare"}
assert full[["Title", "Deck", "FamilySize", "IsAlone"]].isna().sum().sum() == 0

print("survival rate by title")
for k, v in full.groupby("Title")["Survived"].agg(["mean", "size"]).iterrows():
    print(f"  {k:8s} {v['mean']:.3f}   n={int(v['size'])}")
'''),
        prompt(
            label="split first, stratified on the label",
            input="the engineered frame",
            output="712 training and 179 test rows",
            constraint="stratify on y — with 179 test rows an unstratified draw "
                       "moves the base rate by several points and every number "
                       "below moves with it",
            check="712 and 179, disjoint indices, and the two base rates within "
                  "a point of each other. Assert on the RATES as well as the "
                  "sizes: the sizes are right even when the split is badly "
                  "unbalanced",
            **{"try": "drop `stratify=y` and print the two rates. They separate "
                      "by about a point on this seed — and by three on some "
                      "others."}),
        code('''
NUM_V1 = ["Age", "Fare", "FamilySize", "SibSp", "Parch"]
NUM    = ["Age", "Fare", "SibSp", "Parch"]        # FamilySize = SibSp+Parch+1
CAT    = ["Pclass", "Sex", "Embarked", "Title", "Deck"]
BIN    = ["IsAlone"]

X = full[NUM_V1 + CAT + BIN]
y = full["Survived"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)

assert len(X_train) == 712 and len(X_test) == 179
assert set(X_train.index).isdisjoint(X_test.index)
assert abs(y_train.mean() - y_test.mean()) < 0.01, "stratification failed"

print(f"train rate {y_train.mean():.4f}   test rate {y_test.mean():.4f}")
print("712 training passengers. That is a small dataset, and by section 10 the")
print("size is doing visible damage.")
'''),
        md("""
## 5 · The metric, and the anchor

Accuracy cannot be the headline here. It reads a probability of 0.51 and one of
0.99 as the same answer; it cannot be computed at all until a cut-off is chosen,
so quoting it *hides* the decision the brief asked to be explicit; and it weighs
the two mistakes equally.

**Log loss** is the metric that matches the requirement:

$$L = -\\frac{1}{m}\\sum_{i=1}^{m}\\Big[y^{(i)}\\log \\hat{p}^{(i)}
      + (1-y^{(i)})\\log(1-\\hat{p}^{(i)})\\Big]$$

It is a **proper scoring rule**: minimised, in expectation, by reporting your
true belief. Accuracy is not, which is why optimising accuracy destroys
calibration.
"""),
        prompt(
            label="what nothing scores",
            input="the training labels only",
            output="the accuracy of always predicting 'did not survive', and the "
                   "log loss of always reporting the base rate",
            constraint="compute the constant-predictor log loss from the base "
                       "rate arithmetically, not by fitting anything",
            check="it is the binary entropy of p = 0.3834, so you can predict "
                  "it before running: -[p ln p + (1-p) ln(1-p)] = 0.666 nats. "
                  "Every number below has to be read against it",
            **{"try": "report p = 0.5 to everyone instead. The log loss rises "
                      "to ln 2 = 0.693 — the maximum, and the cost of knowing "
                      "nothing at all, including the base rate."}),
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
### The anchor: **0.666**

The log loss of a model that has learned nothing except the base rate. It is
the entropy of the label, in nats. Anything above it is worse than knowing only
how many people died; the distance below it is the only part you earned.

---

## 6 · Logistic regression

$$\\hat p = \\sigma(\\boldsymbol\\theta^{\\intercal}\\mathbf{x}),
  \\qquad \\sigma(t) = \\frac{1}{1+e^{-t}},
  \\qquad
  \\log\\frac{\\hat p}{1-\\hat p} = \\boldsymbol\\theta^{\\intercal}\\mathbf{x}$$

The linear model predicts the **log-odds**. Setting the gradient of the log loss
to zero gives
$\\mathbf{X}^{\\intercal}(\\sigma(\\mathbf{X}\\boldsymbol\\theta) - y) = 0$, in
which $\\boldsymbol\\theta$ sits inside a transcendental function: there is no
closed form. The cost is convex, so descent finds the global minimum anyway.

Preprocessing goes **inside** the pipeline, so cross-validation refits the
imputer, the scaler and the encoder on each training fold.
"""),
        prompt(
            label="preprocessing inside the pipeline",
            input="the five numeric, five categorical and one binary column",
            output="a fitted ColumnTransformer and the width it produces",
            constraint="impute and scale INSIDE the pipeline, so cross-validation "
                       "refits the median per fold and never sees the held-out "
                       "one",
            check="no NaN survives, and 11 raw columns become 28 model columns "
                  "— five one-hot blocks expand. Count the levels of each "
                  "categorical and add them up before running",
            **{"try": "fit the scaler on all of X and then split. The scores "
                      "below improve slightly. That improvement is the leak, "
                      "and there is no version of it that is fine."}),
        code('''
# The obvious preprocessing, and the one section 8 takes apart.
prep_v1 = ColumnTransformer([
    ("num", make_pipeline(SimpleImputer(strategy="median"),
                          StandardScaler()), NUM_V1),
    ("cat", make_pipeline(SimpleImputer(strategy="most_frequent"),
                          OneHotEncoder(handle_unknown="ignore")), CAT),
    ("bin", "passthrough", BIN),
])

Z = prep_v1.fit(X_train).transform(X_train)
print(f"{X_train.shape[1]} raw columns become {Z.shape[1]} model columns")
assert Z.shape == (712, 28)
assert not np.isnan(Z).any(), "the imputer should have left no NaN"
'''),
        prompt(
            label="⏱ 15 s — fit it, and measure it honestly",
            input="the preprocessing block and the 712 training rows",
            output="cross-validated log loss, accuracy and Brier, train and "
                   "held-out, with the per-fold spread",
            constraint="C=1e6 switches the penalty OFF — scikit-learn "
                       "regularises logistic regression by default, and today's "
                       "coefficients have to be the data's rather than a "
                       "compromise with a penalty",
            check="ten folds, and a held-out log loss below the 0.666 anchor. "
                  "Print the per-fold numbers: ten folds whose spread is 0.13 "
                  "do not support a claim about a difference of 0.01",
            **{"try": "`KFold(10)` instead of `StratifiedKFold(10)`. With 71 "
                      "passengers per fold an unstratified split hands some "
                      "fold a base rate several points from the truth, and the "
                      "spread widens."}),
        code('''
def make_model(prep, C=1e6, max_iter=4000):
    """Preprocessing plus an effectively UNregularised logistic regression."""
    return Pipeline([("prep", prep),
                     ("clf", LogisticRegression(C=C, max_iter=max_iter,
                                                random_state=RANDOM_STATE))])

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    r = cross_validate(make_model(prep_v1), X_train, y_train, cv=cv,
                       scoring=["neg_log_loss", "accuracy", "neg_brier_score"],
                       return_train_score=True, n_jobs=NJ)

ll_v1 = -r["test_neg_log_loss"].mean()
print(f"log loss   train {-r['train_neg_log_loss'].mean():.3f}   "
      f"held-out {ll_v1:.4f}   (anchor {constant_log_loss:.3f})")
print(f"accuracy   train {r['train_accuracy'].mean():.3f}   "
      f"held-out {r['test_accuracy'].mean():.3f}")
print(f"Brier      train {-r['train_neg_brier_score'].mean():.3f}   "
      f"held-out {-r['test_neg_brier_score'].mean():.3f}")
print(f"\\nper-fold held-out log loss: "
      f"{np.array2string(-r['test_neg_log_loss'], precision=3)}")
print(f"fold-to-fold spread (sd):   {r['test_neg_log_loss'].std():.3f}")

assert len(r["test_neg_log_loss"]) == 10, "report folds, not just the mean"
assert ll_v1 < constant_log_loss, "the model must at least beat the anchor"
'''),

        # -------------------------------------------------- coefficients
        md("""
## 7 · Read the coefficients

This is why the brief chose logistic regression: each weight is a change in
**log-odds**, so $e^{\\theta_j}$ is a multiplicative effect on the odds.
"""),
        prompt(
            label="the six largest weights",
            input="the fitted unregularised model",
            output="the six largest weights as log-odds and as odds multipliers",
            constraint="strip the ColumnTransformer's `num__`/`cat__` prefixes "
                       "and sort by ABSOLUTE weight, so a large negative "
                       "coefficient is not hidden at the far end of the list",
            check="the name list and the coefficient vector are the same length "
                  "— a mismatch here silently mislabels every row. Then read "
                  "the signs against the survival rates printed in section 4: "
                  "at least two of them are impossible",
            **{"try": "sort by signed value instead. `Deck_T` moves to the top "
                      "and `Title_Master` to the bottom, and the pairing that "
                      "makes the problem obvious is broken up."}),
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
Three of those rows cannot be right.

- `Title_Miss` is strongly **negative** — but 70.3% of them survived, against
  38.4% overall.
- `Title_Mrs` is negative too, and it is the highest-surviving group in the data.
- `Deck_T` is the largest weight in the model. Ask how many passengers are on
  deck T before believing it.

None of these is a weak effect or a noisy estimate. They have the wrong sign,
and the model's *predictions* are nonetheless good. Before reaching for a story
about confounding, ask the cheaper question: what is the shape of the design
matrix?
"""),
        prompt(
            label="how many people is that weight fitted to",
            input="the Deck column",
            output="passengers per deck, and the count for deck T specifically",
            constraint="print the whole distribution, not only the suspicious "
                       "level — the point is that you should have looked before "
                       "believing any of them",
            check="deck T has exactly one passenger, so the largest weight in "
                  "the model is fitted to one person. Large weight plus tiny n "
                  "is memorisation",
            **{"try": "the same count on `Title`. `Rare` has 23, which is small "
                      "but not absurd — the check is a scale, not a switch."}),
        code('''
print("passengers per deck, whole dataset:")
print(full["Deck"].value_counts().to_dict())
print(f"\\ndeck T: {int((full['Deck'] == 'T').sum())} passenger(s)")
assert int((full["Deck"] == "T").sum()) == 1
'''),

        # ------------------------------------------------------ the rank
        md("""
## 8 · Failure condition one: the minimiser is not unique

Lecture 2 established that $\\mathbf{X}^{\\intercal}\\mathbf{X}$ is invertible
exactly when $\\mathbf{X}$ has full column rank. Logistic regression has no
normal equation, but the same condition governs it: if
$\\mathbf{X}\\mathbf{v} = \\mathbf{0}$ for some $\\mathbf{v} \\neq \\mathbf{0}$
then $\\boldsymbol\\theta$ and $\\boldsymbol\\theta + c\\,\\mathbf{v}$ give
*identical* logits for every passenger and every $c$.

Count the columns, then count the rank. They should be equal.
"""),
        prompt(
            label="count the columns, then count the rank",
            input="the transformed design matrix with an intercept column added",
            output="the column count, the matrix rank, and the condition number",
            constraint="add the intercept column BEFORE taking the rank — the "
                       "intercept is what each one-hot block is dependent WITH, "
                       "and without it the deficiency is five rather than six",
            check="29 columns and rank 23. Work the six out on paper first: "
                  "five one-hot blocks each sum to the intercept column, and "
                  "one engineered column is a sum of two others",
            **{"try": "take the rank of Z without the intercept column. It is "
                      "24 of 28 — the deficiency drops by one, because one of "
                      "the six dependencies involved the intercept."}),
        code('''
Z   = m_v1[:-1].transform(X_train)
Z_b = np.c_[np.ones(len(Z)), Z]              # add the intercept column

n_cols = Z_b.shape[1]
rank   = int(np.linalg.matrix_rank(Z_b))
cond   = np.linalg.cond(Z_b)
print(f"columns: {n_cols}")
print(f"rank:    {rank}")
print(f"cond:    {cond:.2e}")
print(f"\\nrank deficiency: {n_cols - rank}")

assert (n_cols, rank) == (29, 23), "29 columns carrying 23 columns of information"
'''),
        prompt(
            label="the dependency you engineered yourself",
            input="SibSp, Parch and FamilySize",
            output="the largest absolute residual of SibSp + Parch + 1 - "
                   "FamilySize",
            constraint="assert it is EXACTLY zero — this is not a correlation, "
                       "it is an identity, and 'highly correlated' would be the "
                       "wrong words for it",
            check="0.0, and then 5 one-hot blocks + 1 engineered sum = 6, which "
                  "is the whole deficiency. Combining features is fine; "
                  "combining them additively alongside their own parts is not",
            **{"try": "`FamilySize = SibSp + Parch + 1 + IsAlone`. Still an "
                      "exact dependence, because IsAlone is itself a function "
                      "of the other two."}),
        code('''
resid = (X_train["SibSp"] + X_train["Parch"] + 1
         - X_train["FamilySize"]).abs().max()
print(f"max |SibSp + Parch + 1 - FamilySize| = {resid}")
assert resid == 0.0, "an exact linear dependence, by construction"

print("\\n5 one-hot blocks + 1 engineered sum = 6. That is the deficiency.")
assert 5 + 1 == n_cols - rank
'''),
        prompt(
            label="prove the coefficients are not the data",
            input="the fitted weights and the transformed rows",
            output="the largest difference in the logit between two different "
                   "coefficient vectors",
            constraint="shift along a dependency direction — add the same "
                       "constant to Sex_female and to Sex_male and subtract it "
                       "from the intercept, since that block sums to 1 on every "
                       "row",
            check="the logits agree to under 1e-9 AND the two vectors genuinely "
                  "differ. Either assert alone proves nothing; the pair is what "
                  "non-identifiability looks like in code",
            **{"try": "shift by 100 instead of 2.5. The logits still agree. "
                      "There is no size of shift that the data can see."}),
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
print(f"largest difference in the logit over all 712 rows: {gap:.1e}")
print(f"Sex_male is {theta[names_v1.index('Sex_male')]:+.2f} in one fit "
      f"and {theta[names_v1.index('Sex_male')] + shift:+.2f} in the other.")
print("Two completely different coefficient vectors. The same model.")

assert gap < 1e-9, "the two parameterisations must be numerically identical"
assert not np.allclose(theta, theta + shift * v), "but the vectors differ"
'''),
        md("""
**So the coefficients are not the data.** They are *a* solution among infinitely
many, and the one you got is an accident of the solver. Any story about "the
effect of being female" read off that table is a story about `lbfgs`.

### The repair

Two changes, and only two, so that the rank moving is attributable:

- drop `FamilySize` — `SibSp` and `Parch` already carry it;
- `drop="first"` on the encoder, so each block loses one level and becomes a set
  of contrasts against a **reference level**.

`min_frequency=2` rather than `handle_unknown="ignore"`: with `drop="first"` an
ignored unknown level is encoded all-zeros, which is exactly the encoding of the
*reference* level — so the deck-T passenger would be scored as though they were
on deck A whenever a fold holds them out.
"""),
        prompt(
            label="the repair, in two lines",
            input="the same columns minus FamilySize, with drop='first' on the "
                  "encoder",
            output="the new column count, rank and condition number, and the "
                   "cross-validated log loss beside the old one",
            constraint="change exactly TWO things, so that the rank moving is "
                       "attributable to them",
            check="columns = rank = 23, full rank; and the condition number "
                  "falls from 2.6e+16 to under a hundred — fourteen orders of "
                  "magnitude, which is not something you can misread",
            **{"try": "make only the `drop='first'` change and leave FamilySize "
                      "in. Rank 24 of 24? No — 23 of 24. One dependency "
                      "remains, and it is the one you wrote yourself."}),
        code('''
prep_v2 = ColumnTransformer([
    ("num", make_pipeline(SimpleImputer(strategy="median"),
                          StandardScaler()), NUM),          # FamilySize gone
    ("cat", make_pipeline(SimpleImputer(strategy="most_frequent"),
                          OneHotEncoder(drop="first", min_frequency=2,
                                        handle_unknown="infrequent_if_exist")),
     CAT),
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

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    folds_v2 = -cross_val_score(make_model(prep_v2), X_train, y_train, cv=cv,
                                scoring="neg_log_loss", n_jobs=NJ)
ll_v2 = folds_v2.mean()

print(f"\\nheld-out log loss, rank-deficient encoding {ll_v1:.4f}   "
      f"fold spread {r['test_neg_log_loss'].std():.2f}")
print(f"held-out log loss, full-rank encoding      {ll_v2:.4f}   "
      f"fold spread {folds_v2.std():.2f}")
print("Not a large move, and not the reason for doing it: the first row was")
print("produced by a fit that had no unique answer, and nothing said so.")
assert ll_v2 < ll_v1
'''),
        prompt(
            label="the coefficients, now that they mean something",
            input="the repaired model",
            output="the six largest contrasts, and Pclass_3 read out in words",
            constraint="state the reference levels out loud — a coefficient "
                       "table with drop='first' and no statement of what was "
                       "dropped cannot be read by anybody, including you in a "
                       "month",
            check="`Sex_female` is gone, because one level of each block was "
                  "dropped; and Pclass_3 is about -1.17, whose exponential is "
                  "0.31 — travelling third class multiplies the ODDS of "
                  "survival by about a third relative to first",
            **{"try": "exponentiate a coefficient and read it as a probability "
                      "ratio. It is wrong: odds of 1 scaled by 0.31 give "
                      "p = 0.24, but odds of 9 scaled by 0.31 give p = 0.74."}),
        code('''
names_v2 = [n.split("__")[-1] for n in m_v2[:-1].get_feature_names_out()]
coefs_v2 = m_v2[-1].coef_[0]

print("reference levels (the dropped ones): Pclass 1, Sex female, Embarked C,")
print("Title Master, Deck A. Every coefficient below is relative to those.\\n")
for n, c in sorted(zip(names_v2, coefs_v2), key=lambda t: -abs(t[1]))[:6]:
    print(f"  {n:24s} {c:+.3f}   odds x{np.exp(c):.3f}")

c3 = float(coefs_v2[names_v2.index("Pclass_3")])
print(f"\\nPclass_3 = {c3:+.3f}: holding every other recorded feature fixed,")
print(f"travelling third class multiplies the ODDS of survival by "
      f"{np.exp(c3):.2f},")
print("relative to first class.")

assert "Sex_female" not in names_v2, "one level of each block must be gone"
'''),

        # ------------------------------------------------- calibration
        md("""
## 9 · Is the probability a probability?

Requirement two was **calibration**:

$$P\\bigl(y = 1 \\mid \\hat p(\\mathbf{x}) = q\\bigr) = q
  \\quad\\text{for every } q \\in (0,1)$$

Take every passenger scored near $q$; about $q$ of them should have survived.
Use **out-of-fold** predictions — a reliability diagram drawn on training rows
always looks excellent, because the model has already fitted them.
"""),
        prompt(
            label="⏱ 15 s — is the probability a probability",
            input="out-of-fold predicted probabilities for all 712 training rows",
            output="ten bins, each with its mean prediction against its observed "
                   "rate, and the expected calibration error",
            constraint="OUT-OF-FOLD — calibration measured on the training rows "
                       "is measured on data the model has already fitted",
            check="the shape is 712 and every value lies in [0, 1]. Print n for "
                  "each bin beside the rates: a bin with n=6 whose observed "
                  "rate is 0.17 is one passenger, not a calibration failure",
            **{"try": "swap `cross_val_predict` for `m_v2.predict_proba`. The "
                      "ECE falls. That improvement is not the model getting "
                      "better, it is the measurement getting worse."}),
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
print(f"\\nexpected calibration error, out of fold: {ece:.3f}")
'''),
        md("""
Roughly calibrated, and not by luck: log loss is a proper scoring rule and we
minimised it directly, so calibration is what the objective was asking for.

## 10 · A bare label hides a cut-off nobody chose

`.predict()` is `.predict_proba() >= 0.5`. Nothing in the prompt says 0.5,
nothing in the code says 0.5, and 0.5 is cost-optimal only when the two mistakes
cost the same. Suppose a stakeholder states that a passenger who drowns
unescorted costs **ten times** an escort assigned to someone who would have
survived anyway. Then measure the difference rather than asserting it.
"""),
        prompt(
            label="the cost of a rule",
            input="the stated 10 : 1 cost ratio",
            output="a function scoring the rule 'escort everyone below "
                   "probability t'",
            constraint="the rule flags LOW survival probability — the direction "
                       "is easy to invert and the resulting curve looks entirely "
                       "plausible upside down",
            check="at t = 0.02 almost nobody is flagged, so the cost is just "
                  "under 10 x the number who died — 439 of the 712, so a little "
                  "under 4390, which you can work out before running. If the "
                  "cost is LOWEST there, the direction of the rule is inverted",
            **{"try": "set COST_FN = COST_FP = 1. The cost-minimising cut-off "
                      "moves to about 0.5, which is what `.predict()` assumed "
                      "all along."}),
        code('''
COST_FN, COST_FP = 10.0, 1.0          # stated by the stakeholder

def total_cost(prob, y_true, t):
    """Cost of the rule `flag everyone with predicted probability < t`."""
    flag = prob < t
    died = (np.asarray(y_true) == 0)
    return float(COST_FN * (died & ~flag).sum()      # drowned, not escorted
                 + COST_FP * (~died & flag).sum())   # escorted, survived anyway

GRID = np.linspace(0.02, 0.98, 97)

n_died_train = int((yv == 0).sum())
print(f"{n_died_train} of {len(yv)} training passengers did not survive")
print(f"cost at t=0.02 (flag almost nobody): {total_cost(p_oof, yv, 0.02):8.1f}"
      f"   ~ 10 x {n_died_train} = {10 * n_died_train}")
print(f"cost at t=0.50 (what .predict() does): {total_cost(p_oof, yv, 0.50):6.1f}")
print(f"cost at t=0.98 (flag almost everyone): {total_cost(p_oof, yv, 0.98):6.1f}")
print("\\nAt 10:1 the cost falls as MORE people are flagged, up to a point.")
print("That point is what the next cell measures, and it is not 0.5.")

assert total_cost(p_oof, yv, 0.02) > total_cost(p_oof, yv, 0.98), \\
    "flagging nobody must cost more than flagging everyone at a 10:1 ratio"
'''),
        md("""
⏱ **about 30 seconds.** Twenty splits, because one subtraction of two noisy
numbers is not a measurement. On each: fit, choose the cut-off by cost on
out-of-fold *training* predictions, then score *both* cut-offs on the held-out
part.
"""),
        prompt(
            label="⏱ 30 s — the default cut-off, measured over 20 splits",
            input="20 different train/test splits of the whole labelled frame",
            output="accuracy and cost at 0.5 and at the cost-chosen cut-off",
            constraint="choose the cut-off on OUT-OF-FOLD training predictions "
                       "and score it on the held-out part — choosing it on "
                       "predictions the model has already seen is training-set "
                       "scoring one level up",
            check="the cost-chosen rule is never beaten on cost, and it is "
                  "WORSE on accuracy. Both directions matter: if the better "
                  "rule also won on accuracy there would be nothing to teach",
            **{"try": "count the splits where the sign goes your way and report "
                      "the count. '20 of 20' is an argument; a mean difference "
                      "on its own is not."}),
        code('''
# The whole labelled frame, in the row order the split of section 4 produced.
# The 20 resplits below are of THIS frame, not of `full` — same rows, different
# order, and train_test_split is order-sensitive, so shuffling here would give
# 20 different splits and 20 different numbers from the ones on the slides.
X_all = pd.concat([X_train, X_test])
y_all = pd.concat([y_train, y_test])


def one_split(seed):
    A, B, ya, yb = train_test_split(X_all, y_all, test_size=0.2,
                                    random_state=seed, stratify=y_all)
    m = make_model(prep_v2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # the cut-off is chosen on out-of-fold training predictions, never on B
        p_in   = cross_val_predict(m, A, ya, cv=5, method="predict_proba",
                                   n_jobs=NJ)[:, 1]
        chosen = float(min(GRID, key=lambda t: total_cost(p_in, ya.values, t)))
        pb     = m.fit(A, ya).predict_proba(B)[:, 1]
    return dict(chosen=chosen,
                acc_half=accuracy_score(yb, (pb >= 0.5).astype(int)),
                acc_best=accuracy_score(yb, (pb >= chosen).astype(int)),
                cost_half=total_cost(pb, yb.values, 0.5),
                cost_best=total_cost(pb, yb.values, chosen))

trap = [one_split(s) for s in range(20)]

ch = np.array([t["cost_half"] for t in trap])
cb = np.array([t["cost_best"] for t in trap])
ah = np.array([t["acc_half"]  for t in trap])
ab = np.array([t["acc_best"]  for t in trap])

print(f"cut-off 0.50 (what .predict() chooses)  cost {ch.mean():7.2f}   "
      f"accuracy {ah.mean():.3f}")
print(f"cut-off chosen from the costs           cost {cb.mean():7.2f}   "
      f"accuracy {ab.mean():.3f}")
print(f"\\nthe default costs {(ch - cb).mean():.2f} more per evacuation")
print(f"splits where 0.50 is the worse rule: {int((ch > cb).sum())} of 20")
print(f"mean chosen cut-off: {np.mean([t['chosen'] for t in trap]):.3f}")

assert (ch >= cb).all(), "the cost-chosen rule cannot be beaten on cost"
assert ab.mean() < ah.mean(), "and it is bought with accuracy"
'''),
        md("""
Read **both** halves. The cost-optimal rule is worse on accuracy by about eight
points, and better on the thing the stakeholder said they cared about by a
factor of nearly three. Accuracy went *down* when we did the right thing.

For a perfectly calibrated model the cost-minimising cut-off is
$t^{\\ast} = c_{\\text{FN}}/(c_{\\text{FN}} + c_{\\text{FP}}) = 10/11 = 0.909$.
The measured optimum sits below that, and the gap **is** the calibration error,
expressed in a unit somebody cares about.

**So: fit and report on a proper scoring rule; choose any cut-off separately, on
out-of-fold predictions, against a constraint somebody stated; report accuracy
at that cut-off as a consequence, never as the objective.**
"""),

        # -------------------------------------------------- the sweep
        md("""
## 11 · Failure condition two: the minimiser does not exist

The model is linear in the features, so give it curvature the way section 2's
polynomial regression did — powers and cross-products of the four numeric
columns. Squaring a one-hot indicator returns the indicator, so the categorical
block is left alone.

⏱ **about 20 seconds.** `return_train_score=True` is not on by default and it
is the whole experiment: one curve tells you nothing.
"""),
        prompt(
            label="⏱ 20 s — push the capacity until it breaks",
            input="polynomial expansions of the four numeric columns, degrees 1 "
                  "to 6",
            output="columns, training log loss, held-out log loss and held-out "
                   "accuracy at each degree",
            constraint="expand the NUMERIC block only — squaring a one-hot "
                       "indicator returns the indicator, and section 8 was "
                       "about removing exact copies",
            check="training error falls at every degree while held-out error "
                  "does not; the two asserts together are the experiment. And "
                  "22 columns at degree 1 becomes 143 at degree 5, on 712 rows "
                  "— count columns against rows before interpreting anything",
            **{"try": "degree 7. The column count passes 700, and the model has "
                      "about one weight per training passenger."}),
        code('''
def poly_model(degree, C=1e6):
    """The repaired preprocessing with a polynomial expansion bolted on."""
    prep = ColumnTransformer([
        ("num", make_pipeline(
            SimpleImputer(strategy="median"), StandardScaler(),
            PolynomialFeatures(degree=degree, include_bias=False)), NUM),
        ("cat", make_pipeline(
            SimpleImputer(strategy="most_frequent"),
            OneHotEncoder(drop="first", min_frequency=2,
                          handle_unknown="infrequent_if_exist")), CAT),
        ("bin", "passthrough", BIN)])
    return make_model(prep, C=C)

DEGREES = [1, 2, 3, 4, 5, 6]
sweep = {}

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    for d in DEGREES:
        r = cross_validate(poly_model(d), X_train, y_train, cv=cv,
                           scoring=["neg_log_loss", "accuracy"],
                           return_train_score=True, n_jobs=NJ)
        fitted = poly_model(d).fit(X_train, y_train)
        sweep[d] = dict(cols=int(fitted[:-1].transform(X_train).shape[1]),
                        train=float(-r["train_neg_log_loss"].mean()),
                        valid=float(-r["test_neg_log_loss"].mean()),
                        acc=float(r["test_accuracy"].mean()),
                        n_iter=int(np.max(fitted[-1].n_iter_)),
                        max_coef=float(np.abs(fitted[-1].coef_).max()))

print(f"{'deg':>3} {'cols':>5} {'train':>8} {'held-out':>9} {'accuracy':>9} "
      f"{'iters':>6} {'converged':>10}")
for d in DEGREES:
    s = sweep[d]
    print(f"{d:>3} {s['cols']:>5} {s['train']:>8.3f} {s['valid']:>9.3f} "
          f"{s['acc']:>9.1%} {s['n_iter']:>6} "
          f"{str(s['n_iter'] < 4000):>10}")

assert sweep[1]["cols"] == 22 and sweep[5]["cols"] == 143
assert sweep[6]["train"] < sweep[1]["train"], "training error falls forever"
assert sweep[5]["valid"] > sweep[2]["valid"], "held-out error does not"
assert sweep[1]["n_iter"] < 4000 and sweep[4]["n_iter"] == 4000, \\
    "degree 4 is where lbfgs stops arriving"
'''),
        prompt(
            label="both curves on one axis",
            input="the sweep",
            output="train and held-out log loss on one axis, held-out accuracy "
                   "on the other",
            constraint="both log-loss curves on the SAME axis — the GAP between "
                       "them is the quantity being shown, and a gap is "
                       "invisible across two panels",
            check="degrees 1 and 2 differ by well under a hundredth in held-out "
                  "log loss, and the fold-to-fold spread printed in section 6 "
                  "was 0.13. They are tied, and reporting 2 as the winner is "
                  "reporting noise",
            **{"try": "plot accuracy on the same axis as log loss. It is flat "
                      "by comparison — accuracy only moves when a prediction "
                      "crosses the cut-off, and log loss moves when a "
                      "confident prediction becomes a confidently wrong one."}),
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

lo, hi = sweep[2]["valid"], sweep[5]["valid"]
print(f"held-out log loss   degree 2 {lo:.3f}   degree 5 {hi:.3f}   "
      f"x{hi / lo:.1f}")
print(f"held-out accuracy   degree 2 {sweep[2]['acc']:.1%}   "
      f"degree 5 {sweep[5]['acc']:.1%}   "
      f"{100 * (sweep[2]['acc'] - sweep[5]['acc']):.1f} points")
print("\\nThe training curve never stops improving. The held-out curve turns")
print("at degree 2 and then falls off a cliff — and log loss saw it first.")
'''),
        md("""
### Why more iterations cannot help

`lbfgs` stops converging at degree 4 and the reflex is to raise `max_iter`. It
cannot help. With 87 columns and 712 rows the two classes become **linearly
separable**: if some $\\boldsymbol\\theta$ classifies every training row
correctly then $J(c\\,\\boldsymbol\\theta) \\to 0$ as $c \\to \\infty$, so the
likelihood has no maximum and the weights run away.

The warning is not a numerical nuisance. It is the solver reporting that the
estimate you asked for **does not exist**.
"""),
        prompt(
            label="watch the weights run away",
            input="the largest fitted weight at each degree",
            output="max |theta| against degree, beside whether the fit converged",
            constraint="read the size of the weights, not only the convergence "
                       "flag — 'did not converge' and 'the weights are growing "
                       "without bound' are the same fact, and the second is the "
                       "one that explains the first",
            check="the largest weight at a non-converging degree is several "
                  "times the largest at degree 1, and it would keep growing "
                  "with max_iter. A weight of 18 in log-odds is a predicted "
                  "probability indistinguishable from 0 or 1",
            **{"try": "refit degree 4 with max_iter=8000. It still does not "
                      "converge, and the largest weight is larger. That is the "
                      "proof that iterations are not the missing ingredient."}),
        code('''
print(f"{'deg':>3} {'converged':>10} {'iters':>6} {'max |theta|':>12}")
for d in DEGREES:
    s = sweep[d]
    print(f"{d:>3} {str(s['n_iter'] < 4000):>10} {s['n_iter']:>6} "
          f"{s['max_coef']:>12.2f}")

print("\\nrank 23 of 29 columns          -> the minimiser is not UNIQUE")
print("lbfgs will not converge at deg 4 -> the minimiser does not EXIST")
print("Both are statements about the optimisation problem, not the passengers.")
print("Both are repaired by the same one-line change: the next lecture.")

assert sweep[4]["max_coef"] > sweep[1]["max_coef"]
'''),

        # ------------------------------------------------------- softmax
        md("""
## 12 · More than two classes: softmax regression

$$s_k(\\mathbf{x}) = \\boldsymbol\\theta_k^{\\intercal}\\mathbf{x},
  \\qquad
  \\hat p_k = \\frac{e^{s_k}}{\\sum_{j=1}^{K} e^{s_j}},
  \\qquad
  J = -\\frac{1}{m}\\sum_i \\sum_k y_k^{(i)} \\log \\hat p_k^{(i)}$$

and the gradient is
$\\nabla_{\\boldsymbol\\theta_k} J = \\frac{1}{m}\\sum_i
(\\hat p_k^{(i)} - y_k^{(i)})\\mathbf{x}^{(i)}$ — error times feature, for the
third time in this notebook.

A real three-class target on the same manifest: predict the **ticket class**
from the other columns.
"""),
        prompt(
            label="softmax on three classes",
            input="the Titanic columns other than Pclass, and Pclass as a "
                  "three-class target",
            output="held-out accuracy and cross-entropy, and the shape of the "
                   "predicted probability matrix",
            constraint="drop Pclass from the FEATURES — it is the target here, "
                       "and leaving it in gives a perfect score that measures "
                       "nothing",
            check="the probability matrix is (712, 3) and every row sums to 1, "
                  "by construction rather than by luck. Predict the accuracy "
                  "roughly first: the majority class is third at 55%, so "
                  "anything near 0.55 has learned nothing",
            **{"try": "put Fare back on its raw scale by removing the "
                      "StandardScaler. Accuracy barely moves and the solver "
                      "takes many more iterations — the condition number "
                      "again."}),
        code('''
CAT_S = ["Sex", "Embarked", "Title", "Deck"]      # Pclass is the target now
prep_s = ColumnTransformer([
    ("num", make_pipeline(SimpleImputer(strategy="median"),
                          StandardScaler()), NUM),
    ("cat", make_pipeline(SimpleImputer(strategy="most_frequent"),
                          OneHotEncoder(drop="first", min_frequency=2,
                                        handle_unknown="infrequent_if_exist")),
     CAT_S),
    ("bin", "passthrough", BIN),
])

y_class = full.loc[X_train.index, "Pclass"]
soft = Pipeline([("prep", prep_s),
                 ("clf", LogisticRegression(C=1.0, max_iter=4000,
                                            random_state=RANDOM_STATE))])

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    r_s = cross_validate(soft, X_train, y_class, cv=5,
                         scoring=["accuracy", "neg_log_loss"], n_jobs=NJ)
    P = soft.fit(X_train, y_class).predict_proba(X_train)

print(f"classes: {soft[-1].classes_}")
print(f"predicted probability matrix: {P.shape}")
print(f"every row sums to 1: {np.allclose(P.sum(axis=1), 1.0)}")
print(f"majority class share:      {y_class.value_counts(normalize=True).max():.3f}")
print(f"held-out accuracy:         {r_s['test_accuracy'].mean():.3f}")
print(f"held-out cross-entropy:    {-r_s['test_neg_log_loss'].mean():.3f}")

assert P.shape == (712, 3)
assert np.allclose(P.sum(axis=1), 1.0), "softmax normalises by construction"
assert r_s["test_accuracy"].mean() > y_class.value_counts(normalize=True).max()
'''),
        prompt(
            label="the softmax gradient, checked at the fitted point",
            input="the fitted weights, the design matrix and the one-hot labels",
            output="the largest entry of (1/m) X^T (P - Y), which the derivation "
                   "says is the gradient",
            constraint="fit with a WEAK penalty so the stationarity condition of "
                       "the unpenalised cost is nearly satisfied — with the "
                       "default C=1 the gradient at the optimum is not zero, it "
                       "is the penalty's gradient with the sign flipped",
            check="the largest entry is small — of order 1e-3 or below. If your "
                  "formula had the sign or the transpose wrong it would be of "
                  "order 1, not of order 1e-3",
            **{"try": "refit with C=0.01. The residual grows by orders of "
                      "magnitude, and the size of it is exactly the penalty "
                      "term the next lecture adds."}),
        code('''
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    soft_free = Pipeline([("prep", prep_s),
                          ("clf", LogisticRegression(C=1e6, max_iter=20000,
                                                     random_state=RANDOM_STATE))
                          ]).fit(X_train, y_class)

Zs = np.asarray(soft_free[:-1].transform(X_train), dtype=float)
Ps = soft_free.predict_proba(X_train)
Ys = np.zeros_like(Ps)
for j, c in enumerate(soft_free[-1].classes_):
    Ys[:, j] = (y_class.values == c)

grad = Zs.T @ (Ps - Ys) / len(Zs)      # the derivation's formula, all K at once
print(f"largest entry of (1/m) X^T (P - Y): {np.abs(grad).max():.2e}")
print("At the minimum of an unpenalised cost this is zero. It is not exactly")
print("zero here because C=1e6 is a very weak penalty rather than none at all.")

assert np.abs(grad).max() < 1e-2
'''),

        # --------------------------------------------------------- close
        md("""
## 13 · Where we are

The test set has not been touched, and will not be until the end of the next
lecture.
"""),
        prompt(
            label="the numbers to keep",
            input="everything measured above",
            output="the anchor, the best held-out log loss and accuracy, the "
                   "rank of both designs, and the cut-off chosen from the costs",
            constraint="print the anchor beside every score, because a log loss "
                       "of 0.47 means nothing until you know that doing no work "
                       "at all scores 0.666",
            check="the best degree is 1 or 2 — the two simplest models are tied "
                  "at the top, and an assert that accepted only one of them "
                  "would be asserting noise",
            **{"try": "re-run the whole notebook with RANDOM_STATE = 7. Every "
                      "number moves a little; which conclusions move with "
                      "them, and which do not?"}),
        code('''
best_d = min(DEGREES, key=lambda d: sweep[d]["valid"])

print(f"anchor, report the base rate to everyone   {constant_log_loss:.3f}")
print(f"held-out log loss, rank-deficient design   {ll_v1:.3f}")
print(f"held-out log loss, full-rank design        {ll_v2:.3f}")
print(f"held-out log loss, best degree ({best_d})         "
      f"{sweep[best_d]['valid']:.3f}")
print(f"held-out accuracy at that degree           {sweep[best_d]['acc']:.1%}")
print(f"expected calibration error, out of fold    {ece:.3f}")
print(f"cut-off chosen from the 10:1 costs         "
      f"{np.mean([t['chosen'] for t in trap]):.2f}")

assert best_d in (1, 2), "the two simplest models are tied at the top"
'''),
        md("""
### The six things to take away

1. Descent moves along $-\\nabla J$; on the MSE that gradient is
   $\\frac{2}{m}\\mathbf{X}^{\\intercal}(\\mathbf{X}\\boldsymbol\\theta - y)$
   and its zero is the normal equation. You checked the two agree to 1e-8.
2. The cost is convex, the iteration converges **iff**
   $\\eta < 2/\\lambda_{\\max}$, and the step *count* is set by the condition
   number — which is why scaling is part of the algorithm.
3. One row is an unbiased gradient. You checked that identity to machine
   precision; it is what makes mini-batch descent legitimate rather than merely
   cheap.
4. Logistic regression predicts the **log-odds** linearly. A coefficient
   multiplies the odds, always relative to a stated reference level.
5. Its fit is undefined in two ways, and you produced both: not **unique** under
   rank deficiency, not **existent** under separation.
6. A probability is checkable and a label is not — and `.predict()` is a cut-off
   nobody chose.

**Before the next lecture:** raise `ETA_DIVERGE` in section 2 and find where the
cost first becomes `inf`, then check that against $2/\\lambda_{\\max}$ for that
design matrix. The next lecture repairs points 5 and 6 with one line, and
explains the shape of the two curves in section 11 with a decomposition into
three terms.
"""),
    ]
    return cells
