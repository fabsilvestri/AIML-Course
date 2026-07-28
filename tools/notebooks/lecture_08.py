#!/usr/bin/env python3
"""
Lecture 8 — Retrain it and watch it change. Fix, CoverType, Géron Ch. 5–6.

Exports build() -> list[cell]; tools/make_notebooks.py wraps it.

Structure mirrors the deck: thread -> diagnosis -> repair -> re-measure ->
red-team. Every import this notebook needs is in its own setup cell: a notebook
that only runs because the previous one is still in the kernel is not
reproducible.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from make_notebooks import code, header, md, SETUP        # noqa: E402


REBUILD = code('''
# --- everything this notebook needs, in one place ----------------------------
from sklearn.datasets import fetch_covtype
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (BaggingClassifier, ExtraTreesClassifier,
                              RandomForestClassifier)
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

COVER_NAMES = ["Spruce/Fir", "Lodgepole Pine", "Ponderosa Pine",
               "Cottonwood/Willow", "Aspen", "Douglas-fir", "Krummholz"]

cover = fetch_covtype(as_frame=True)              # ~5 s from the local cache
X, _, y, _ = train_test_split(cover.data, cover.target, train_size=60_000,
                              stratify=cover.target, random_state=RANDOM_STATE)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
del cover

tree = DecisionTreeClassifier(max_depth=8, min_samples_leaf=1,
                              random_state=RANDOM_STATE).fit(X_train, y_train)
tree_acc = tree.score(X_test, y_test)
baseline = DummyClassifier(strategy="most_frequent").fit(
    X_train, y_train).score(X_test, y_test)

assert len(X_train) == 48_000 and len(X_test) == 12_000
print("same split as the previous lecture — the seed guarantees it")
print(f"depth-8 tree {tree_acc:.1%}   constant baseline {baseline:.1%}")
''')


def build() -> list:
    cells = header(
        8, "Retrain it and watch it change", "fix", "Chapters 5 & 6",
        thread="impurity, and why averaging reduces variance")

    cells += [
        md("## 1 · Setup and where we left off"), SETUP, REBUILD,

        md("""
## 2 · Thread 4, part one — Gini and entropy

CART minimises **Gini impurity**, $G = 1 - \\sum_k p_k^2$. Scikit-Learn offers
**entropy**, $H = -\\sum_k p_k \\log_2 p_k$, instead. Both are zero exactly at
purity and maximal at uniformity. Géron says the choice usually makes little
difference.

That is a claim about behaviour, and we have a dataset. Plot the two first.
"""),
        code('''
p = np.linspace(1e-9, 1 - 1e-9, 400)
gini = 2 * p * (1 - p)
ent = -(p * np.log2(p) + (1 - p) * np.log2(1 - p))

fig, ax = plt.subplots(1, 2, figsize=(11, 3.4))
ax[0].plot(p, gini, label="Gini  2p(1-p)")
ax[0].plot(p, ent, label="entropy  H(p), bits")
ax[0].plot(p, ent / 2, "--", label="entropy / 2")
ax[0].set_xlabel("p"); ax[0].set_ylabel("impurity"); ax[0].legend(fontsize=8)
ax[1].plot(p, ent / 2 - gini, color="firebrick")
ax[1].axhline(0, lw=1, color="grey")
ax[1].set_xlabel("p"); ax[1].set_ylabel("entropy/2 - Gini")
plt.tight_layout(); plt.show()

gap = (ent / 2 - gini)
print(f"largest gap {gap.max():.4f} at p = {p[gap.argmax()]:.3f}")
'''),
        md("""
Same zeros, same maximum, different shoulders: entropy penalises a nearly-pure
node more heavily. Now measure whether that changes anything, **paired** — both
criteria on the same resample each time, so the resample-to-resample noise
cancels.

⏱ **about 40 seconds** — twenty fits.
"""),
        code('''
rows = []
for seed in range(10):
    Xs, _, ys, _ = train_test_split(X_train, y_train, train_size=0.8,
                                    stratify=y_train, random_state=seed)
    rec = {}
    for crit in ("gini", "entropy"):
        t = DecisionTreeClassifier(criterion=crit, max_depth=8,
                                   random_state=RANDOM_STATE).fit(Xs, ys)
        rec[crit] = t.score(X_test, y_test)
        rec[crit + "_root"] = int(t.tree_.feature[0])
        rec[crit + "_pred"] = t.predict(X_test)
    rec["agree"] = float((rec["gini_pred"] == rec["entropy_pred"]).mean())
    rows.append(rec)

diff = np.array([r["entropy"] - r["gini"] for r in rows]) * 100
same_root = sum(r["gini_root"] == r["entropy_root"] for r in rows)

print(f"same root feature      {same_root} of {len(rows)} resamples")
print(f"predictions agreeing   {np.mean([r['agree'] for r in rows]):.1%}")
print(f"entropy - gini         {diff.mean():+.2f} +/- {diff.std():.2f} points")
print(f"resamples entropy won  {(diff > 0).sum()} of {len(rows)}")
'''),
        md("""
Three statements, all true, and only the third is a recommendation:

1. The effect is **real** — the sign is consistent, and gini wins on 8 of the
   10 resamples.
2. The effect is **tiny** — a fraction of a point, against the several points
   `max_depth` was worth in the previous lecture.
3. So **do not spend your tuning budget here.** Leave the default.

"Statistically detectable" and "worth acting on" are different claims and need
different evidence.
"""),

        md("""
## 3 · Thread 4, part two — the variance of an average

Forget trees for ten minutes. You have $n$ predictors of the same quantity, each
with variance $\\sigma^2$, and you average them. Almost everyone answers
$\\sigma^2/n$, and that is only valid when they are uncorrelated.

For **identically distributed** predictors with common pairwise correlation
$\\rho$:

$$\\operatorname{Var}\\!\\left(\\frac{1}{n}\\sum_t f_t\\right)
  = \\frac{1}{n^2}\\Big[n\\sigma^2 + n(n-1)\\rho\\sigma^2\\Big]
  = \\rho\\sigma^2 + \\frac{(1-\\rho)\\sigma^2}{n}$$

There are $n$ diagonal terms of $\\sigma^2$ and $n(n-1)$ off-diagonal terms of
$\\rho\\sigma^2$ in the double sum; that is the whole derivation.

**Averaging destroys the independent component and leaves the correlated one
completely untouched.** Check it numerically before believing it.
"""),
        code('''
def correlated(n, rho, draws, rng):
    """n predictors, unit variance, common pairwise correlation rho."""
    shared = rng.standard_normal((draws, 1))
    own = rng.standard_normal((draws, n))
    return np.sqrt(rho) * shared + np.sqrt(1 - rho) * own


rng = np.random.default_rng(RANDOM_STATE)
print(f"{'rho':>5s} {'n':>4s} {'measured':>10s} {'formula':>10s}")
for rho in (0.0, 0.3, 0.8):
    for n in (1, 5, 50):
        f = correlated(n, rho, 200_000, rng)
        measured = f.mean(axis=1).var()
        formula = rho + (1 - rho) / n
        print(f"{rho:5.1f} {n:4d} {measured:10.4f} {formula:10.4f}")
        assert abs(measured - formula) < 0.02
'''),
        md("""
Three limits worth memorising:

| case | variance of the average |
|---|---|
| $\\rho = 0$ | $\\sigma^2/n$ — the classical result |
| $\\rho = 1$ | $\\sigma^2$ — averaging is a no-op |
| $n \\to \\infty$ | $\\rho\\sigma^2$ — the floor |

So the way to improve an ensemble is not more members. It is **less correlated**
members. And anything that lowers $\\rho$ by making individual members worse is a
trade, not a free lunch: $\\sigma^2$ appears in both terms.
"""),

        md("""
## 4 · Diagnose — the question from the end of the last lecture

Your neighbour fitted the same model, same hyperparameters, same seed, to
*almost* the same training set. How similar are the two sets of rules?

Twenty trees, each on 90% of the same training set, so any two share about 80%
of their rows.

⏱ **about 30 seconds.**
"""),
        code('''
trees, preds, roots, accs = [], [], [], []
for seed in range(20):
    Xs, _, ys, _ = train_test_split(X_train, y_train, train_size=0.9,
                                    stratify=y_train, random_state=1000 + seed)
    t = DecisionTreeClassifier(max_depth=8, min_samples_leaf=1,
                               random_state=RANDOM_STATE).fit(Xs, ys)
    trees.append(t)
    preds.append(t.predict(X_test))
    roots.append((int(t.tree_.feature[0]), round(float(t.tree_.threshold[0]), 1)))
    accs.append(t.score(X_test, y_test))

accs = np.array(accs)
assert len(preds) == 20 and all(len(p) == 12_000 for p in preds)
print(f"accuracy  mean {accs.mean():.2%}  sd {accs.std():.2%}  "
      f"({accs.min():.2%} - {accs.max():.2%})")
'''),
        md("""
A spread of about a point across twenty refits. Watching only the headline
number, you would conclude the model is completely stable.

That is the flattering half. Now the other one.
"""),
        code('''
P = np.array(preds)
disagree = np.array([(P[i] != P[j]).mean()
                     for i in range(20) for j in range(i + 1, 20)])
unanimous = (P == P[0]).all(axis=0).mean()

print(f"pairwise disagreement  {disagree.mean():.1%}  "
      f"({disagree.min():.1%} - {disagree.max():.1%}) over "
      f"{len(disagree)} pairs")
print(f"patches all 20 agree on  {unanimous:.1%}")
'''),
        code('''
from collections import Counter

print("root feature:", {X_train.columns[f]: c
                        for (f, _), c in Counter(roots).items()})
print("distinct root thresholds:", len({thr for _, thr in roots}))
print("leaves per tree:", min(t.get_n_leaves() for t in trees), "-",
      max(t.get_n_leaves() for t in trees))
print("columns consulted:",
      min(len({int(f) for f in t.tree_.feature if f >= 0}) for t in trees), "-",
      max(len({int(f) for f in t.tree_.feature if f >= 0}) for t in trees))
'''),
        md("""
### The diagnosis

The part you put on a slide is stable — the root is the same feature every time.
The part that decides is not: two trees with the same accuracy disagree about one
prediction in eleven.

Both results are consistent. A tree is a **hierarchy**: the root split is chosen
from 48,000 patches and wins by a wide margin, but a node eight levels down was
chosen from a few hundred, where two candidates are often separated by a hair.
Change one row and everything below that node is a different tree. Accuracy is an
average over 12,000 patches, and averages hide substitutions.

**A stable metric is not a stable model. It is evidence that you measured the
wrong thing.**

This is *variance* in the sense of the bias–variance decomposition, and the
thread tells us exactly what to do about variance: average.
"""),

        md("""
## 5 · Fix — bagging

Draw rows with replacement, fit a tree, repeat. Each bootstrap sample contains
about 63% of the distinct rows. The members are **unconstrained** trees: we want
low bias from each and we are about to average the variance away.

⏱ **about 40 seconds** — 100 unconstrained trees on 48,000 rows.
"""),
        code('''
bag = BaggingClassifier(DecisionTreeClassifier(random_state=RANDOM_STATE),
                        n_estimators=100, bootstrap=True, oob_score=True,
                        random_state=RANDOM_STATE, n_jobs=-1).fit(X_train, y_train)

print(f"out-of-bag  {bag.oob_score_:.1%}")
print(f"test        {bag.score(X_test, y_test):.1%}")
'''),
        md("""
A given row is missed by one draw with probability $1 - 1/m$, so by all $m$ draws
with probability $(1-1/m)^m \\to e^{-1} \\approx 0.368$. Those out-of-bag rows give
a generalisation estimate with no validation split and no extra fits. It replaces
the *validation* set, not the test set.

### One trap, worth ten minutes of your life

⚠ **Read before running.** This is the shape question, and it costs an afternoon
the first time.
"""),
        code('''
member = bag.estimators_[0]
print("ensemble labels:", bag.classes_)
print("member labels:  ", member.classes_)

Xte_arr = X_test.to_numpy()
naive = (member.predict(Xte_arr) == y_test).mean()
mapped = (bag.classes_[member.predict(Xte_arr).astype(int)] == y_test).mean()

print(f"\\ncompared directly with y_test: {naive:.1%}   <- looks like a bad model")
print(f"mapped through classes_:       {mapped:.1%}   <- the truth")
assert mapped > naive, "the mapping should rescue the score"
'''),
        md("""
Every ensemble in Scikit-Learn re-encodes `y` as positions `0..k-1` before
handing it to its members, so a member's `predict` returns **positions**, not
cover types. Comparing them with `y_test` directly gives a plausible-looking
small number rather than an exception.

Reviewer question 5: *what is the default I did not ask for?* This one.
"""),

        md("""
## 6 · Random forests and extra-trees

Bagging randomises the **rows**. A random forest also randomises the
**columns**: at every node, only $\\lfloor\\sqrt{54}\\rfloor = 7$ features are
considered. Extra-trees also randomises the **thresholds** — drawn at random
rather than searched for.

All three exist to reduce one quantity: $\\rho$.

⏱ **about 40 seconds for the three.**
"""),
        code('''
ensembles = {
    "bagging": bag,
    "forest": RandomForestClassifier(n_estimators=100, max_features="sqrt",
                                     random_state=RANDOM_STATE, n_jobs=-1),
    "extra": ExtraTreesClassifier(n_estimators=100, max_features="sqrt",
                                  bootstrap=False, random_state=RANDOM_STATE,
                                  n_jobs=-1),
    "extra_bs": ExtraTreesClassifier(n_estimators=100, max_features="sqrt",
                                     bootstrap=True, random_state=RANDOM_STATE,
                                     n_jobs=-1),
}
for name, m in ensembles.items():
    if name != "bagging":
        m.fit(X_train, y_train)
    member_acc = (m.classes_[m.estimators_[0].predict(X_test.to_numpy()).astype(int)]
                  == y_test).mean()
    print(f"{name:10s} ensemble {m.score(X_test, y_test):.1%}   "
          f"one member {member_acc:.1%}")
'''),
        md("""
Note the second column. Every mechanism that makes the members less alike also
makes each of them **worse**. That is the trade the formula warned about.

Scikit-Learn's default for extra-trees is `bootstrap=False` — the random
thresholds *replace* the bootstrap rather than joining it. `extra_bs` turns it
back on, and the pair is what separates the two effects.
"""),

        md("""
## 7 · Measuring $\\rho$ needs something we do not have

$\\rho$ is the correlation between two members over the randomness of the **whole
procedure**, which includes which training set you were handed. Conditional on
one dataset the members are independent by construction, and $\\rho$ would
measure as zero.

So the experiment needs several *independent* training sets. CoverType has
581,012 rows, so we can cut disjoint ones and never reuse a row.

⏱ **about 90 seconds.** The lecture's figure uses 20 training sets of 20,000
rows and 20 members; this is a smaller version of the same experiment, so the
numbers will be close but not identical.
"""),
        code('''
K, M, N_Z, N_TE = 10, 10, 15_000, 6_000

full = fetch_covtype(as_frame=False)
rng = np.random.default_rng(RANDOM_STATE)
perm = rng.permutation(len(full.target))
te, pool = perm[:N_TE], perm[N_TE:N_TE + K * N_Z]
Xte, yte = full.data[te], full.target[te]

assert len(pool) == K * N_Z and len(set(pool) & set(te)) == 0


def make(kind, n, seed):
    if kind == "bagging":
        return BaggingClassifier(DecisionTreeClassifier(random_state=seed),
                                 n_estimators=n, bootstrap=True,
                                 random_state=seed, n_jobs=-1)
    if kind == "forest":
        return RandomForestClassifier(n_estimators=n, max_features="sqrt",
                                      random_state=seed, n_jobs=-1)
    return ExtraTreesClassifier(n_estimators=n, max_features="sqrt",
                                bootstrap=(kind == "extra_bs"),
                                random_state=seed, n_jobs=-1)


def experiment(kind):
    S = np.zeros((K, M, N_TE), dtype=np.float32)
    for k in range(K):
        idx = pool[k * N_Z:(k + 1) * N_Z]
        m = make(kind, M, 1000 + k).fit(full.data[idx], full.target[idx])
        for j, est in enumerate(m.estimators_):
            S[k, j] = (m.classes_[est.predict(Xte).astype(int)] == yte)
    return S
'''),
        code('''
def decompose(S):
    """Split one member's variance into the part its training set explains."""
    K_, M_, _ = S.shape
    within = S.var(axis=1, ddof=1).mean(axis=0)        # per test patch
    between = S.mean(axis=1).var(axis=0, ddof=1)       # per test patch
    tau2 = np.maximum(between - within / M_, 0.0)      # ANOVA estimator
    sigma2 = tau2 + within
    curve = {n: float(S[:, :n].mean(axis=1).var(axis=0, ddof=1).mean())
             for n in (1, 2, 5, 10)}
    return dict(rho=float(tau2.mean() / sigma2.mean()),
                sigma2=float(sigma2.mean()), tau2=float(tau2.mean()),
                within=float(within.mean()), curve=curve)


dec = {}
for kind in ("bagging", "forest", "extra", "extra_bs"):
    dec[kind] = decompose(experiment(kind))

print(f"{'':10s} {'rho':>7s} {'sigma^2':>9s} {'V(1)':>9s} {'V(10)':>9s} "
      f"{'predicted':>10s}")
for kind, d in dec.items():
    pred = d["tau2"] + d["within"] / 10
    print(f"{kind:10s} {d['rho']:7.3f} {d['sigma2']:9.4f} "
          f"{d['curve'][1]:9.4f} {d['curve'][10]:9.4f} {pred:10.4f}")
'''),
        md("""
The last two columns are the point of the whole lecture: the variance of an
average of ten members, measured, against
$\\rho\\sigma^2 + (1-\\rho)\\sigma^2/n$ evaluated at $n = 10$. Nothing was fitted
to make them agree.

Read the $\\rho$ column carefully. Every variant is below bagging, so the claim
survives — but the mechanisms are **not additive**. Feature subsampling does most
of the work; random thresholds are largely a substitute for the bootstrap rather
than an addition to it.

And there is a floor to the floor: nothing gets $\\rho$ near zero, because all
four ensembles ultimately saw the same rows. $\\rho$ is not a property of the
algorithm — it is the share of the variance that comes from *which data you were
given*.
"""),
        code('''
fig, ax = plt.subplots(figsize=(7, 3.6))
for kind, d in dec.items():
    ns = sorted(d["curve"])
    ax.plot(ns, [d["curve"][n] for n in ns], "o", label=kind)
    nn = np.linspace(1, 10, 60)
    ax.plot(nn, d["tau2"] + d["within"] / nn, lw=1.5, alpha=0.7)
    ax.axhline(d["tau2"], ls=":", lw=1)
ax.set_xlabel("n, members averaged"); ax.set_ylabel("variance of the average")
ax.set_yscale("log"); ax.legend(fontsize=8)
plt.tight_layout(); plt.show()
'''),

        md("""
## 8 · Now the bill

We fixed the variance. Ask what happened to the thing we were hired to deliver.
"""),
        code('''
rnd = ensembles["forest"]
total_leaves = sum(t.get_n_leaves() for t in rnd.estimators_)

print(f"depth-8 tree     {tree.get_n_leaves():>10,d} leaves   "
      f"{tree_acc:.1%}")
print(f"100-tree forest  {total_leaves:>10,d} leaves   "
      f"{rnd.score(X_test, y_test):.1%}")
print("\\nThe justification for one prediction is now 100 decision paths "
      "and a vote.")
'''),
        md("""
There is no technical fix for this. The regulator asked for a model whose every
prediction comes with a human-readable justification, and we have built one that
is far more accurate and **cannot supply one**. What follows recovers something,
and it is genuinely less than what was lost.
"""),

        md("""
## 9 · An assistant explains the forest

> *"The forest replaced our decision tree. Show me which features it relies on,
> so I can put that in the report to the regulator."*

**⚠ Read before running.** It runs, it is fast, and the top of the list is
entirely sensible.
"""),
        code('''
imp = pd.Series(rnd.feature_importances_, index=X_train.columns)
print("The measurements the model relies on:")
print(imp.sort_values(ascending=False).head(8).round(3))
'''),
        md("""
### The review question: what would the answer look like if it were wrong?

Add a **control**. A column of uniform random numbers cannot possibly carry
information about which trees grow where. Guess where it ranks among the 55
before running the cell.

⏱ **about 20 seconds.**
"""),
        code('''
rng = np.random.default_rng(RANDOM_STATE)
X_tr2 = X_train.assign(random_decoy=rng.random(len(X_train)))
X_te2 = X_test.assign(random_decoy=rng.random(len(X_test)))

rnd2 = RandomForestClassifier(n_estimators=100, max_features="sqrt",
                              random_state=RANDOM_STATE,
                              n_jobs=-1).fit(X_tr2, y_train)

imp2 = pd.Series(rnd2.feature_importances_, index=X_tr2.columns)
order = imp2.sort_values(ascending=False)
rank = list(order.index).index("random_decoy") + 1

print(f"random_decoy ranks {rank} of {len(imp2)}  "
      f"(importance {imp2['random_decoy']:.4f})")
print(f"{(imp2 < imp2['random_decoy']).sum()} real columns rank below it")
assert rank < 30, "expected the decoy to rank absurdly high"
'''),
        md("""
### Why impurity importance does that

`feature_importances_` sums, over every node that split on a feature, the
weighted impurity reduction that split achieved. Two biases follow directly:

- **It is measured on the training data.** A split that reduces impurity on the
  rows that chose it will do so whether or not the feature is informative.
- **It favours high-cardinality features.** A continuous column offers thousands
  of candidate thresholds; a 0/1 soil-type column offers one. More chances to
  find a lucky split means more accumulated impurity reduction.

Our decoy is continuous and forty of the fifty-four real columns are binary.

The repair: shuffle one column of the **held-out** set, re-score, and see how
much accuracy falls.

⏱ **about 60 seconds.** *(`permutation_importance` goes beyond Chapter 6 —
**not examinable**. It is here because the alternative is to ship the biased
measurement.)*
"""),
        code('''
sub = slice(0, 3000)
perm = permutation_importance(rnd2, X_te2.iloc[sub], y_test.iloc[sub],
                              n_repeats=3, random_state=RANDOM_STATE, n_jobs=-1)

pi = pd.DataFrame({"mean": perm.importances_mean, "sd": perm.importances_std},
                  index=X_te2.columns).sort_values("mean", ascending=False)
print(pi.head(8).round(4))
print()
print("random_decoy:")
print(pi.loc[["random_decoy"]].round(5))
'''),
        code('''
top = order.head(12).index
fig, ax = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
colours = ["firebrick" if c == "random_decoy" else "steelblue" for c in top]
ax[0].barh(range(len(top))[::-1], imp2[top], color=colours)
ax[0].set_yticks(range(len(top))[::-1], top, fontsize=7)
ax[0].set_xlabel("impurity importance")
ax[1].barh(range(len(top))[::-1], pi.loc[top, "mean"], color=colours,
           xerr=pi.loc[top, "sd"])
ax[1].axvline(0, lw=1, color="grey")
ax[1].set_xlabel("permutation importance, held out")
plt.tight_layout(); plt.show()
'''),
        md("""
Same model, same columns, same row order. The decoy is near the top on the left
and indistinguishable from zero on the right.

### What importance can and cannot tell a regulator

| Question | Can importance answer it? |
|---|---|
| Which measurements should the survey keep collecting? | yes — this is what it is for |
| Is the model using a variable it legally must not? | yes, as a screen |
| Why was *this* parcel refused? | **no** |
| Would removing this column hurt? | no — remove it, refit, and measure |

Row three is the regulator's actual question.
"""),

        md("""
## 10 · The rest of Chapter 6, briefly

Everything above trains members **in parallel** and averages them. **Boosting**
trains them in sequence, each correcting its predecessor — so today's formula
does not apply to it, and it reduces *bias* rather than variance.

⏱ **about 60 seconds.**
"""),
        code('''
from sklearn.ensemble import HistGradientBoostingClassifier

hgb = HistGradientBoostingClassifier(max_iter=100, learning_rate=0.2,
                                     early_stopping=True, n_iter_no_change=10,
                                     random_state=RANDOM_STATE)
hgb.fit(X_train, y_train)

print(f"histogram gradient boosting  {hgb.score(X_test, y_test):.1%}  "
      f"({hgb.n_iter_} iterations, early stopping)")
print(f"bagging                      {bag.score(X_test, y_test):.1%}")
print(f"the legible depth-8 tree     {tree_acc:.1%}")
'''),
        md("""
`AdaBoost` reweights the misclassified instances; gradient boosting fits each
new member to the ensemble's residual errors; `StackingClassifier` trains a
*blender* on cross-validated out-of-fold predictions — and that cross-validation
is the only reason stacking is safe. Build it by hand and reviewer question 1
becomes the whole difficulty: what touched the data the blender learns from?
"""),

        md("""
## 11 · Re-measure, and score your sheet

| Reality | Test accuracy |
|---|---|
| always "Lodgepole Pine" | 48.8% |
| the legible depth-8 tree | 73.3% |
| bagged, 200 unconstrained trees | 89.8% |

The diagnosis was instability, not inaccuracy — so test the thing we diagnosed.
Repeat the twenty-subsample experiment on a forest and predict the answer before
you run it.

⏱ **about 2 minutes** — twenty forests of 30 trees.
"""),
        code('''
fpreds = []
for seed in range(20):
    Xs, _, ys, _ = train_test_split(X_train, y_train, train_size=0.9,
                                    stratify=y_train, random_state=1000 + seed)
    f = RandomForestClassifier(n_estimators=30, max_features="sqrt",
                               random_state=RANDOM_STATE,
                               n_jobs=-1).fit(Xs, ys)
    fpreds.append(f.predict(X_test))

F = np.array(fpreds)
fdis = np.array([(F[i] != F[j]).mean()
                 for i in range(20) for j in range(i + 1, 20)])

print(f"single tree, pairwise disagreement  {disagree.mean():.1%}")
print(f"30-tree forest                      {fdis.mean():.1%}")
print(f"patches all 20 forests agree on     {(F == F[0]).all(axis=0).mean():.1%}")
'''),

        md("""
## 12 · Red-team

Swap notebooks with the team beside you. Eight minutes, five questions:

1. What touched the test set? Was any hyperparameter chosen by looking at it?
2. What was fitted, and on what? Check where `permutation_importance` was
   computed.
3. What is the shape here? Check every comparison between a member's output and
   `y_test`.
4. What was dropped — rows, columns, classes? Count them.
5. What is the default you did not ask for? Find `bootstrap`, `max_features` and
   `oob_score` in their code and say what each is set to.

Report what you **found**, not what you would have done differently.

### The standing constraint, extended

> "… Split before anything is fitted. All preprocessing inside a `Pipeline`
> passed to cross-validation. Nothing derived from the test set in the training
> path. Fixed seeds. Per-fold scores, not just the mean. **Any importance or
> explanation is computed on held-out data and reported with a control. Any
> per-instance claim is checked for stability under refitting.**"
"""),
    ]
    return cells
