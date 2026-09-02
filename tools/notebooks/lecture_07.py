#!/usr/bin/env python3
"""
Lecture 7 — Ensembles and random forests. CoverType, Géron Ch. 6.

Exports build() -> list[cell]; tools/make_notebooks.py wraps it.

Structure mirrors the deck: rebuild the previous lecture's tree from the seed ->
check the variance formula on synthetic predictors -> build the four ensembles
-> measure rho on twenty disjoint training sets and hold the formula to it ->
count what it cost -> feature importance, with a control -> boosting and
stacking.

Every quantity this notebook prints that also appears on a slide is computed on
the same rows, with the same seed, at the same scale as tools/figures_app04.py:
200-member pools on the 48,000 training rows, and 20 disjoint training sets of
20,000 rows with 20 members each against a shared 12,000-row test set. A smaller
version of that experiment would land near the slide numbers without matching
them, which is worse than either.

Every import this notebook needs is in its own setup cell: a notebook that only
runs because the previous one is still in the kernel is not reproducible.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from make_notebooks import code, header, md, SETUP, SETUP_PROMPT        # noqa: E402
from _prompt import prompt                                # noqa: E402


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
AUDITABLE_LEAF = 20            # the previous lecture's brief, not a tuned value

cover = fetch_covtype(as_frame=True)              # ~5 s from the local cache
X, _, y, _ = train_test_split(cover.data, cover.target, train_size=60_000,
                              stratify=cover.target, random_state=RANDOM_STATE)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
del cover, X, y

tree = DecisionTreeClassifier(max_depth=8, min_samples_leaf=AUDITABLE_LEAF,
                              random_state=RANDOM_STATE).fit(X_train, y_train)
tree_acc = tree.score(X_test, y_test)
baseline = DummyClassifier(strategy="most_frequent").fit(
    X_train, y_train).score(X_test, y_test)

assert len(X_train) == 48_000 and len(X_test) == 12_000
print("same split as the previous lecture — the seed guarantees it")
print(f"depth-8 tree {tree_acc:.1%}   constant baseline {baseline:.1%}")
''')


def build() -> list:
    cells = header(7, "Ensembles and random forests", "", "Chapter 6")

    cells += [
        md("""
Everything on Lecture 7's slides, on the same rows and with the same seed as the
figures: four 200-member ensembles on the 48,000 training patches, twenty
disjoint training sets of 20,000 rows for the correlation experiment, and the
decoy column that exposes what impurity importance is measuring.

The derivation is checked twice — once on synthetic predictors with a $\\rho$ you
choose, and once against the four real ensembles, where the formula has to
predict a variance it was never fitted to.

Runs on free CPU. It rebuilds the previous lecture's split from the seed, so
nothing has to be carried across. The whole notebook is roughly eight minutes on
Colab's two cores; the slow cells say so, and the numbers they report are one
machine's and will not be yours.
"""),

        md("## 1 · Setup and where we left off"), SETUP_PROMPT, SETUP,
        prompt(
            label="rebuild the previous lecture's state",
            input="the same dataset, the same seed",
            output="the identical 48,000 / 12,000 split, the depth-8 tree with a "
                   "minimum leaf of 20, and the constant baseline",
            constraint="reproduce the split from the SEED, not by loading "
                       "anything the previous notebook saved",
            check="assert the two sizes — if the split differs, every comparison "
                  "in this notebook is against a different model. The tree "
                  "accuracy must reproduce the previous lecture's to the last "
                  "decimal; if it does not, stop here",
            **{"try": "change `RANDOM_STATE` to 0. Every number in this notebook "
                      "moves by a few tenths of a point, and none of them can be "
                      "compared with the slides any more"}),
        REBUILD,

        md("""
## 2 · The result we are carrying forward

The previous lecture ended by refitting that tree on twenty different 90%
subsamples of the same training rows. Reproduce it here in one cell, because the
whole of today is a response to it — and because we will run the same
measurement on an ensemble at the end.

⏱ **about 30 seconds** — twenty fits on 43,200 rows each.
"""),
        prompt(
            label="⏱ 30 s — the instability, reproduced",
            input="twenty 90% subsamples of the training rows",
            output="the spread of the accuracy, and the pairwise disagreement "
                   "over all 190 pairs of prediction vectors",
            constraint="the SAME twenty seeds the previous lecture used, so the "
                       "two notebooks are measuring the same twenty trees",
            check="20 choose 2 is 190 pairs. Expect the accuracy to move by a "
                  "fraction of a point and the predictions by an order of "
                  "magnitude more — a stable metric over an unstable model",
            **{"try": "compare the accuracies pairwise instead of the "
                      "predictions. The spread is under a point and tells you "
                      "nothing about the substitutions underneath it"}),
        code('''
tree_preds = []
for seed in range(20):
    Xs, _, ys, _ = train_test_split(X_train, y_train, train_size=0.9,
                                    stratify=y_train, random_state=1000 + seed)
    t20 = DecisionTreeClassifier(max_depth=8, min_samples_leaf=AUDITABLE_LEAF,
                                 random_state=RANDOM_STATE).fit(Xs, ys)
    tree_preds.append(t20.predict(X_test))

P = np.array(tree_preds)
tree_disagree = np.array([(P[i] != P[j]).mean()
                          for i in range(20) for j in range(i + 1, 20)])
assert len(tree_disagree) == 190

print(f"single tree, pairwise disagreement  {tree_disagree.mean():.1%}")
print(f"patches all 20 trees agree on       {(P == P[0]).all(axis=0).mean():.1%}")
'''),
        md("""
That is **variance** in the sense of the bias–variance decomposition: sensitivity
of the fitted function to the particular training sample. The rest of this
notebook is one question — what does averaging do to variance? — answered first
on paper and then on this dataset.
"""),

        md("""
## 3 · The mathematics — the variance of an average

You have $n$ predictors of the same quantity, each with variance $\\sigma^2$, and
you average them. Almost everyone answers $\\sigma^2/n$, and that step is valid
only when they are uncorrelated.

For **identically distributed** predictors with common pairwise correlation
$\\rho$, expand the variance of the sum as a double sum of covariances. It has
$n$ diagonal terms of $\\sigma^2$ and $n(n-1)$ off-diagonal terms of
$\\rho\\sigma^2$, so

$$\\operatorname{Var}\\!\\left(\\frac{1}{n}\\sum_t f_t\\right)
  = \\frac{1}{n^2}\\Big[n\\sigma^2 + n(n-1)\\rho\\sigma^2\\Big]
  = \\rho\\sigma^2 + \\frac{(1-\\rho)\\sigma^2}{n}.$$

**Averaging destroys the independent component and leaves the correlated one
completely untouched.** Check it numerically before believing it.
"""),
        prompt(
            label="the variance of an average, checked numerically",
            input="n predictors with unit variance and common pairwise "
                  "correlation rho",
            output="the measured variance of their average, beside "
                   "rho + (1-rho)/n",
            constraint="build the correlation from a SHARED component plus an "
                       "independent one — sqrt(rho)*shared + sqrt(1-rho)*own has "
                       "exactly the covariance structure the formula assumes, "
                       "and nothing else does",
            check="at rho = 0 the measured variance must fall like 1/n; at any "
                  "rho > 0 it must flatten onto rho. Work out the n = 1 column "
                  "before running: it is 1.0 for every rho",
            **{"try": "rho = 1.0 with n = 50. The measured variance stays at 1 — "
                      "fifty identical predictors are one predictor"}),
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
## 4 · Bagging, and a validation set for nothing

Draw rows with replacement, fit a tree, repeat. Each bootstrap sample contains
about 63% of the distinct rows, because a given row is missed by all $m$ draws
with probability $(1-1/m)^m \\to e^{-1} \\approx 0.368$.

The members are **unconstrained** trees. Averaging cannot touch bias, so the
members must not have any to spare — which is the opposite of the previous
lecture's advice and follows from the same formula.

⏱ **about 40 seconds** — 200 unconstrained trees on 48,000 rows.
"""),
        prompt(
            label="⏱ 40 s — bagging, with out-of-bag scoring",
            input="200 unconstrained trees, each on a bootstrap sample",
            output="the out-of-bag score and the test score",
            constraint="the members are UNCONSTRAINED — no `max_depth`, no "
                       "`min_samples_leaf` — because we want low bias from each "
                       "and are about to average the variance away",
            check="out-of-bag and test should agree to within a few tenths of a "
                  "point. If out-of-bag is much BETTER, something in the "
                  "pipeline saw the out-of-bag rows anyway, and the gap is a "
                  "leakage detector you got for free",
            **{"try": "`bootstrap=False`. `oob_score=True` then raises, because "
                      "with no bootstrap there are no out-of-bag rows at all"}),
        code('''
bag = BaggingClassifier(DecisionTreeClassifier(random_state=RANDOM_STATE),
                        n_estimators=200, max_features=1.0, bootstrap=True,
                        oob_score=True, random_state=RANDOM_STATE,
                        n_jobs=-1).fit(X_train, y_train)

print(f"out-of-bag  {bag.oob_score_:.1%}")
print(f"test        {bag.score(X_test, y_test):.1%}")
print(f"one tree    {tree_acc:.1%}   constant {baseline:.1%}")
'''),

        md("""
### One trap, worth ten minutes of your life

Every ensemble in scikit-learn re-encodes `y` as positions `0..k-1` before handing
it to its members, so a member's `predict` returns **positions**, not cover types.
Comparing them with `y_test` directly gives a plausible-looking small number
rather than an exception.
"""),
        prompt(
            label="the trap that costs an afternoon",
            input="one member of the ensemble and the test set",
            output="that member's accuracy computed naively, and again mapped "
                   "through `classes_`",
            constraint="show the WRONG number first — it is plausible and small "
                       "rather than an exception, which is what makes it "
                       "expensive",
            check="the labels are 1..7 and the positions 0..6, so a naive "
                  "comparison is right only where a patch's class happens to sit "
                  "one below its own index — about one time in seven by chance. "
                  "Predict roughly 8% before running",
            **{"try": "print `bag.classes_[bag.estimators_[0].predict(...)]` "
                      "beside the raw output for ten patches. The two differ by "
                      "exactly one, everywhere"}),
        code('''
member = bag.estimators_[0]
Xte_arr = X_test.to_numpy()          # members were fitted on arrays, not frames

print("ensemble labels:", bag.classes_)
print("member labels:  ", member.classes_)

naive = (member.predict(Xte_arr) == y_test).mean()
mapped = (bag.classes_[member.predict(Xte_arr).astype(int)] == y_test).mean()

print(f"\\ncompared directly with y_test: {naive:.1%}   <- looks like a bad model")
print(f"mapped through classes_:       {mapped:.1%}   <- the truth")
assert mapped > naive, "the mapping should rescue the score"
'''),

        md("""
## 5 · Three ways to decorrelate the members

Bagging randomises the **rows**. A random forest also randomises the
**columns**: at every node only $\\lfloor\\sqrt{54}\\rfloor = 7$ features are
considered. Extra-trees also randomises the **thresholds**, drawn at random
rather than searched for — and scikit-learn's default turns the bootstrap
*off* when it does, so the thresholds replace it rather than joining it.

All three exist to reduce one quantity: $\\rho$.

⏱ **about 90 seconds** for the four pools together.
"""),
        prompt(
            label="⏱ 90 s — four pools of 200 members",
            input="bagging, random forest, extra-trees with and without the "
                  "bootstrap",
            output="each ensemble's accuracy at 1, 10 and 200 members, and the "
                   "accuracy of ONE of its members",
            constraint="accumulate the members' predicted probabilities so the "
                       "accuracy at 10 and at 200 come from the SAME fitted "
                       "pool — refitting a 10-member ensemble separately would "
                       "be a different experiment",
            check="a member's probability columns are a subset of the "
                  "ensemble's, because a bootstrap can drop a rare class "
                  "entirely from one member. Index by `est.classes_`, not by "
                  "position, or the columns silently misalign",
            **{"try": "read the member column against the ensemble column. Every "
                      "mechanism that makes members less alike also makes each "
                      "one worse — that is the trade the formula warned about"}),
        code('''
def make_ensemble(kind, n, seed):
    """One ensemble of `n` members. `kind` names what it randomises."""
    if kind == "bagging":
        return BaggingClassifier(DecisionTreeClassifier(random_state=seed),
                                 n_estimators=n, max_features=1.0,
                                 bootstrap=True, random_state=seed, n_jobs=-1)
    if kind == "forest":
        return RandomForestClassifier(n_estimators=n, max_features="sqrt",
                                      random_state=seed, n_jobs=-1)
    # sklearn's extra-trees default is bootstrap=False; "extra_bs" turns it back
    # on, and the pair separates the thresholds from the bootstrap
    return ExtraTreesClassifier(n_estimators=n, max_features="sqrt",
                                bootstrap=(kind == "extra_bs"),
                                random_state=seed, n_jobs=-1)


KINDS = ["bagging", "forest", "extra", "extra_bs"]
pools, acc_at = {}, {}

for kind in KINDS:
    m = bag if kind == "bagging" else make_ensemble(kind, 200,
                                                    RANDOM_STATE).fit(X_train,
                                                                      y_train)
    classes = m.classes_
    votes = np.zeros((len(y_test), len(classes)))
    at = {}
    for i, est in enumerate(m.estimators_):
        cols = np.asarray(est.classes_).astype(int)   # a subset, not a range
        votes[:, cols] += est.predict_proba(Xte_arr)
        if i + 1 in (1, 10, 200):
            at[i + 1] = float((classes[votes.argmax(axis=1)] == y_test).mean())
    pools[kind], acc_at[kind] = m, at
    print(f"{kind:9s} one member {at[1]:.1%}   10 members {at[10]:.1%}   "
          f"200 members {at[200]:.1%}")
'''),

        md("""
## 6 · Measuring $\\rho$ needs something we do not have

$\\rho$ is the correlation between two members over the randomness of the **whole
procedure**, which includes which training set you were handed. Conditional on
one dataset the members are independent by construction, and $\\rho$ would
measure as zero.

So the experiment needs several *independent* training sets. CoverType has
581,012 rows, so we can cut **20 disjoint sets of 20,000 rows** — 400,000 rows,
no row reused — and keep 12,000 aside as a shared test set.

⏱ **about 3 minutes** — 4 ensemble kinds x 20 training sets x 20 members.
"""),
        prompt(
            label="⏱ 3 min — twenty disjoint training sets",
            input="twenty DISJOINT training sets of 20,000 rows and one shared "
                  "12,000-row test set",
            output="for each ensemble kind, an array recording whether every "
                   "member of every ensemble was right about every test patch",
            constraint="the training sets must be disjoint from each other AND "
                       "from the test set — rho is a correlation over the "
                       "randomness of the whole procedure, including which "
                       "training set you were handed",
            check="assert the pool is exactly 20 x 20,000 rows and shares no "
                  "index with the test set, and print both against the 581,012 "
                  "rows CoverType has. If the pools overlap, the between-group "
                  "variance is contaminated and rho comes out too high for a "
                  "reason no amount of reading the formula will find",
            **{"try": "reuse one training set for all twenty ensembles. tau^2 "
                      "collapses towards zero and rho with it — which is exactly "
                      "the mistake this design exists to avoid"}),
        code('''
K, M, N_Z, N_TE = 20, 20, 20_000, 12_000

full = fetch_covtype(as_frame=False)               # arrays: 581,012 x 54
rng = np.random.default_rng(RANDOM_STATE)
perm = rng.permutation(len(full.target))
te, pool = perm[:N_TE], perm[N_TE:N_TE + K * N_Z]
Xte_big, yte_big = full.data[te], full.target[te]

assert len(pool) == K * N_Z
assert len(set(pool.tolist()) & set(te.tolist())) == 0, "train and test overlap"

print(f"{len(full.target):,} rows available")
print(f"{K} disjoint training sets x {N_Z:,} rows = {K * N_Z:,}, "
      f"plus {N_TE:,} held out")


def experiment(kind):
    """S[k, j, i] = 1 when member j of ensemble k is right about patch i."""
    S = np.zeros((K, M, N_TE), dtype=np.float32)
    for k in range(K):
        idx = pool[k * N_Z:(k + 1) * N_Z]
        m = make_ensemble(kind, M, 1000 + k).fit(full.data[idx],
                                                 full.target[idx])
        for j, est in enumerate(m.estimators_):
            S[k, j] = (m.classes_[est.predict(Xte_big).astype(int)] == yte_big)
    return S
'''),
        prompt(
            label="rho, sigma squared, and the prediction",
            input="the correctness array, 20 training sets by 20 members by "
                  "12,000 test patches",
            output="rho, sigma^2, the measured variance at n = 1 and n = 20, and "
                   "the formula's prediction at n = 20",
            constraint="use the ANOVA estimator — between-group variance minus "
                       "within-group over M — and clamp it at zero, since an "
                       "unbiased variance estimate can come out negative on "
                       "small samples",
            check="measured and predicted must agree in the third decimal place, "
                  "on all four ensembles, with nothing fitted to make them. That "
                  "agreement is the whole lecture; if it fails, the design is "
                  "wrong before the formula is",
            **{"try": "drop the `- within / M_` correction. tau^2 comes out "
                      "systematically too large, because the group means are "
                      "themselves noisy estimates"}),
        code('''
def decompose(S):
    """Split one member's variance into the part its training set explains."""
    K_, M_, _ = S.shape
    within = S.var(axis=1, ddof=1).mean(axis=0)        # per test patch
    between = S.mean(axis=1).var(axis=0, ddof=1)       # per test patch
    tau2 = np.maximum(between - within / M_, 0.0)      # ANOVA estimator
    sigma2 = tau2 + within
    curve = {n: float(S[:, :n].mean(axis=1).var(axis=0, ddof=1).mean())
             for n in (1, 2, 3, 5, 10, 20)}
    return dict(rho=float(tau2.mean() / sigma2.mean()),
                sigma2=float(sigma2.mean()), tau2=float(tau2.mean()),
                within=float(within.mean()), curve=curve)


dec = {kind: decompose(experiment(kind)) for kind in KINDS}

print(f"{'':10s} {'rho':>7s} {'sigma^2':>9s} {'V(1)':>9s} {'V(20)':>9s} "
      f"{'predicted':>10s} {'removed':>9s} {'floor':>9s}")
for kind, d in dec.items():
    pred = d["tau2"] + d["within"] / 20
    removed = 1 - d["curve"][20] / d["curve"][1]
    print(f"{kind:10s} {d['rho']:7.3f} {d['sigma2']:9.4f} "
          f"{d['curve'][1]:9.5f} {d['curve'][20]:9.5f} {pred:10.5f} "
          f"{removed:9.1%} {d['tau2']:9.5f}")

del full                                   # 250 MB we no longer need
'''),
        md("""
The last three columns are the point of the whole lecture: the variance of an
average of twenty members, measured, against
$\\rho\\sigma^2 + (1-\\rho)\\sigma^2/n$ evaluated at $n = 20$, and the floor
$\\rho\\sigma^2$ that no amount of averaging reaches below. Nothing was fitted to
make them agree.

Read the $\\rho$ column carefully. Every variant is below bagging, so the claim
survives — but the mechanisms are **not additive**. Feature subsampling does most
of the work; random thresholds are largely a substitute for the bootstrap rather
than an addition to it.

And there is a floor to the floor: nothing gets $\\rho$ near zero, because all
four ensembles ultimately saw the same rows. $\\rho$ is not a property of the
algorithm — it is the share of the variance that comes from *which data you were
given*.
"""),
        prompt(
            label="the curve and its floor",
            input="the four decompositions",
            output="measured variance against n, the predicted curve, and each "
                   "rho*sigma^2 floor",
            constraint="log y-axis, and draw the floor as a horizontal line per "
                       "variant — the point of the picture is that the curves "
                       "flatten onto different floors rather than towards zero",
            check="plot the measured points and the predicted curve on the same "
                  "axes. Agreement between two independently computed things is "
                  "the strongest evidence a notebook can offer, and it only "
                  "counts if both are visible",
            **{"try": "a linear y-axis. The four floors become indistinguishable "
                      "from zero and the figure stops making its point"}),
        code('''
fig, ax = plt.subplots(figsize=(7, 3.6))
for kind, d in dec.items():
    ns = sorted(d["curve"])
    ax.plot(ns, [d["curve"][n] for n in ns], "o", label=kind)
    nn = np.linspace(1, 20, 80)
    ax.plot(nn, d["tau2"] + d["within"] / nn, lw=1.5, alpha=0.7)
    ax.axhline(d["tau2"], ls=":", lw=1)
ax.set_xlabel("n, members averaged"); ax.set_ylabel("variance of the average")
ax.set_yscale("log"); ax.legend(fontsize=8)
plt.tight_layout(); plt.show()
'''),
        prompt(
            label="the uncomfortable table",
            input="the four decompositions and the four pools",
            output="rho, one member's accuracy and the ensemble's accuracy, in "
                   "one table",
            constraint="sort by rho so the ordering is visible, and put the "
                       "member accuracy beside it — the point is a "
                       "correlation the formula does NOT predict",
            check="the lowest-rho ensemble should NOT be the most accurate. The "
                  "formula is about variance and says nothing about bias, and "
                  "every mechanism that lowers rho does so by handicapping the "
                  "members",
            **{"try": "add a column of `1 - member_accuracy`. It tracks rho "
                      "almost exactly, which is the trade stated as a number"}),
        code('''
print(f"{'':10s} {'rho':>7s} {'one member':>12s} {'ensemble':>10s}")
for kind in sorted(KINDS, key=lambda k: -dec[k]["rho"]):
    print(f"{kind:10s} {dec[kind]['rho']:7.3f} {acc_at[kind][1]:12.1%} "
          f"{acc_at[kind][200]:10.1%}")

print("\\nThe ensemble with the HIGHEST rho is the most accurate. That is not a")
print("contradiction: the formula is about variance, accuracy depends on bias")
print("too, and sigma^2 appears in both of its terms.")
'''),

        md("""
## 7 · Now the bill

We removed the variance. Ask what happened to the thing the agency asked for: a
human-readable justification for every individual prediction.
"""),
        prompt(
            label="now the bill",
            input="the 200-tree forest and the depth-8 tree",
            output="total leaves and accuracy for each",
            constraint="count leaves across ALL 200 members — the justification "
                       "for one prediction is now 200 decision paths and a vote",
            check="the previous lecture's tree has 163 leaves. Divide the "
                  "forest's total by that before running the cell and you have "
                  "predicted the order of magnitude: four",
            **{"try": "count the DISTINCT questions asked across the forest "
                      "instead. It is smaller than the leaf count and still far "
                      "beyond anything a person reads"}),
        code('''
rnd = pools["forest"]
total_leaves = sum(t.get_n_leaves() for t in rnd.estimators_)

print(f"depth-8 tree     {tree.get_n_leaves():>10,d} leaves   {tree_acc:.1%}")
print(f"200-tree forest  {total_leaves:>10,d} leaves   "
      f"{acc_at['forest'][200]:.1%}")
print(f"\\nbought {100 * (acc_at['forest'][200] - tree_acc):.1f} points of "
      f"accuracy with {total_leaves // tree.get_n_leaves():,}x the model")
'''),
        md("""
There is no technical fix for this. The agency asked for a model whose every
prediction comes with a human-readable justification, and we have built one that
is far more accurate and **cannot supply one**. What follows recovers something,
and it is genuinely less than what was lost.
"""),

        md("""
## 8 · Feature importance, and the control that exposes it

`feature_importances_` sums, over every node that split on a feature, the
weighted impurity reduction that split achieved — the $\\Delta I$ of the previous
lecture, accumulated. It runs, it is fast, and the top of the list is entirely
sensible.

Before trusting any ranking, ask what it would look like on a variable you *know*
carries no information. Add a column of uniform random numbers and find out.
Guess where it ranks among the 55 before running the cell.

⏱ **about 30 seconds.**
"""),
        prompt(
            label="⏱ 30 s — the control column",
            input="the same training and test data plus one column of uniform "
                  "random numbers",
            output="the top of the impurity ranking, and where the decoy lands "
                   "among the 55 columns",
            constraint="add the decoy to BOTH train and test, and refit — a "
                       "control added to only one side is not a control, it is a "
                       "distribution shift",
            check="the decoy carries no information, so the honest answer is "
                  "55th. Predict that, then read what impurity importance "
                  "actually says",
            **{"try": "make the decoy a 0/1 coin flip instead of a uniform draw. "
                      "It falls a long way down the ranking — the bias is about "
                      "how many candidate thresholds a column offers"}),
        code('''
decoy_rng = np.random.default_rng(RANDOM_STATE)
X_tr2 = X_train.assign(random_decoy=decoy_rng.random(len(X_train)))
X_te2 = X_test.assign(random_decoy=decoy_rng.random(len(X_test)))

rnd2 = RandomForestClassifier(n_estimators=100, max_features="sqrt",
                              random_state=RANDOM_STATE,
                              n_jobs=-1).fit(X_tr2, y_train)

imp = pd.Series(rnd2.feature_importances_, index=X_tr2.columns)
order = imp.sort_values(ascending=False)
rank = list(order.index).index("random_decoy") + 1

print(order.head(6).round(4).to_string())
print(f"\\nrandom_decoy ranks {rank} of {len(imp)}  "
      f"(importance {imp['random_decoy']:.4f})")
print(f"{(imp < imp['random_decoy']).sum()} real columns rank below it")
print(f"forest accuracy with the decoy: {rnd2.score(X_te2, y_test):.1%}")
assert rank < 30, "expected the decoy to rank absurdly high"
'''),
        md("""
### Why impurity importance does that

Two biases, both structural:

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
        prompt(
            label="⏱ 60 s — the repair",
            input="the fitted forest and 4,000 HELD-OUT rows",
            output="permutation importance with its standard deviation, for the "
                   "decoy and for the strongest real column",
            constraint="permute on HELD-OUT data — the whole defect of the "
                       "impurity version is that it is measured on the rows that "
                       "chose the splits",
            check="report the standard deviation. An importance of 0.001 +/- "
                  "0.002 is zero, and without the second number it reads as a "
                  "small positive effect. The decoy must land within about half "
                  "a standard deviation of zero",
            **{"try": "`n_repeats=1`. The standard deviation is no longer "
                      "defined and the decoy's single draw is as likely to be "
                      "positive as negative — the repeats are the measurement, "
                      "not a refinement of it"}),
        code('''
sub = slice(0, 4000)
perm = permutation_importance(rnd2, X_te2.iloc[sub], y_test.iloc[sub],
                              n_repeats=5, random_state=RANDOM_STATE, n_jobs=-1)

pi = pd.DataFrame({"mean": perm.importances_mean, "sd": perm.importances_std},
                  index=X_te2.columns).sort_values("mean", ascending=False)
print(pi.head(6).round(4).to_string())
print()
print("random_decoy, both ways:")
print(f"  impurity     {imp['random_decoy']:.4f}   (rank {rank} of {len(imp)})")
print(f"  permutation  {pi.loc['random_decoy', 'mean']:+.5f} +/- "
      f"{pi.loc['random_decoy', 'sd']:.5f}   (rank "
      f"{list(pi.index).index('random_decoy') + 1} of {len(pi)})")
'''),
        prompt(
            label="the two rankings, side by side",
            input="both importance measures on the same twelve columns",
            output="two horizontal bar charts, the decoy coloured differently",
            constraint="same columns, same order, shared y-axis — the comparison "
                       "IS the figure, and re-sorting each panel independently "
                       "would destroy it",
            check="error bars on the right panel. They are what let a reader see "
                  "that the decoy's bar is not small but absent",
            **{"try": "sort the right panel by its own values. The decoy moves "
                      "and the two panels can no longer be read across, which is "
                      "the whole content of the figure"}),
        code('''
top = order.head(12).index
fig, ax = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
colours = ["firebrick" if c == "random_decoy" else "steelblue" for c in top]
ax[0].barh(range(len(top))[::-1], imp[top], color=colours)
ax[0].set_yticks(range(len(top))[::-1], top, fontsize=7)
ax[0].set_xlabel("impurity importance (training data)")
ax[1].barh(range(len(top))[::-1], pi.loc[top, "mean"], color=colours,
           xerr=pi.loc[top, "sd"])
ax[1].axvline(0, lw=1, color="grey")
ax[1].set_xlabel("permutation importance (held out)")
plt.tight_layout(); plt.show()
'''),
        md("""
Same model, same columns, same row order. The decoy is near the top on the left
and indistinguishable from zero on the right.

### What importance can and cannot tell the agency

| Question | Can importance answer it? |
|---|---|
| Which measurements should the survey keep collecting? | yes — this is what it is for |
| Is the model using a variable it legally must not? | yes, as a screen |
| Why was *this* parcel refused? | **no** |
| Would removing this column hurt? | no — remove it, refit, and measure |

Row three is the agency's actual question.
"""),

        md("""
## 9 · Boosting, and what practitioners reach for

Everything above trains members **in parallel** and averages them. **Boosting**
trains them in sequence, each correcting its predecessor — so the members are
neither identically distributed nor exchangeable, today's formula does not apply
to it, and it reduces *bias* rather than variance.

`AdaBoost` reweights the misclassified instances. Gradient boosting fits each new
member to the ensemble's residual errors. `HistGradientBoostingClassifier` bins
each feature into at most 255 buckets first, which makes a split search cost
$O(\\text{bins})$ rather than $O(m)$ — it is the algorithm LightGBM and XGBoost
implement, and on tabular data it is the usual first choice.

On *this* dataset it is not the winner, and the cell below is where you find that
out rather than assume it either way.

⏱ **about 2 minutes** — 100 boosting rounds on 48,000 rows and seven classes.
"""),
        prompt(
            label="⏱ 2 min — histogram gradient boosting",
            input="the same training rows, 100 boosting rounds at a learning "
                  "rate of 0.05",
            output="its training and test accuracy, beside bagging, the forest "
                   "and the legible tree",
            constraint="report the TRAINING score too — boosting drives it up "
                       "monotonically whether or not the test score follows, "
                       "which is the whole difference from bagging",
            check="adding members to a BAGGED ensemble can only help, because "
                  "(1-rho)sigma^2/n falls monotonically and nothing else moves. "
                  "Adding members to a BOOSTED one eventually hurts. So expect a "
                  "train-test gap here that the ensembles above did not have",
            **{"try": "`learning_rate=0.2` with the same 100 rounds. The test "
                      "score falls by several points — at a high rate the later "
                      "rounds overshoot, and more boosting makes the model "
                      "worse rather than better"}),
        code('''
from sklearn.ensemble import HistGradientBoostingClassifier

hgb = HistGradientBoostingClassifier(max_iter=100, learning_rate=0.05,
                                     early_stopping=False,
                                     random_state=RANDOM_STATE).fit(X_train,
                                                                    y_train)

print(f"histogram gradient boosting  train {hgb.score(X_train, y_train):.1%}   "
      f"test {hgb.score(X_test, y_test):.1%}")
print(f"bagging, 200 trees           {'':11s}test {acc_at['bagging'][200]:.1%}")
print(f"random forest, 200 trees     {'':11s}test {acc_at['forest'][200]:.1%}")
print(f"the legible depth-8 tree     {'':11s}test {tree_acc:.1%}")
print("\\nThe usual first choice for tabular data, and on this dataset it loses")
print("to a bag of unconstrained trees. Usual is not always — which is why the")
print("comparison is a cell rather than a sentence.")
'''),
        md("""
`StackingClassifier` trains a *blender* on cross-validated out-of-fold
predictions, and that cross-validation is the only reason stacking is safe. Build
it by hand and the blender sees in-sample predictions, learns to trust whichever
base learner overfits hardest, and the ensemble ends up worse than its best
member — with no error and no warning.
"""),

        md("""
## 10 · Did the averaging fix what we diagnosed?

The diagnosis was *instability*, not inaccuracy. So test the thing that was
diagnosed. Repeat section 2's experiment on forests and **predict the answer
before you run it**.

⏱ **about 2 minutes** — twenty forests of 30 trees.
"""),
        prompt(
            label="⏱ 2 min — test the thing we diagnosed",
            input="the same twenty 90% subsamples, forests of 30 trees",
            output="pairwise disagreement and unanimity, beside the single "
                   "tree's",
            constraint="the SAME twenty seeds as section 2 — a stability "
                       "comparison across different subsamples is not a "
                       "comparison",
            check="the formula says twenty members remove about nine tenths of a "
                  "member's variance, so predict a disagreement several times "
                  "smaller than the tree's. A repair whose effect you can "
                  "predict is a repair you understood",
            **{"try": "`n_estimators=3`. The disagreement lands between the two "
                      "— most of the available reduction really does arrive in "
                      "the first few members"}),
        code('''
fpreds = []
for seed in range(20):
    Xs, _, ys, _ = train_test_split(X_train, y_train, train_size=0.9,
                                    stratify=y_train, random_state=1000 + seed)
    f30 = RandomForestClassifier(n_estimators=30, max_features="sqrt",
                                 random_state=RANDOM_STATE,
                                 n_jobs=-1).fit(Xs, ys)
    fpreds.append(f30.predict(X_test))

F = np.array(fpreds)
fdis = np.array([(F[i] != F[j]).mean()
                 for i in range(20) for j in range(i + 1, 20)])

print(f"single tree, pairwise disagreement  {tree_disagree.mean():.1%}")
print(f"30-tree forest                      {fdis.mean():.1%}")
print(f"patches all 20 trees agree on       {(P == P[0]).all(axis=0).mean():.1%}")
print(f"patches all 20 forests agree on     {(F == F[0]).all(axis=0).mean():.1%}")
'''),

        md("""
## 11 · Where we are

The report the agency needs says three things, and the third is the one that is
not a machine learning judgement:

1. The legible tree is 73.0% accurate, and its rules reproduce only to within
   about one prediction in eleven.
2. The ensemble is far more accurate and cannot justify a single decision.
3. **The choice between them is the agency's**, and here are both numbers with
   the experiments that produced them.

### The standing constraint, extended

> "… Split before anything is fitted. All preprocessing inside a `Pipeline`
> passed to cross-validation. Nothing derived from the test set in the training
> path. Fixed seeds. Per-fold scores, not just the mean. **Any importance or
> explanation is computed on held-out data and reported with a control. Any
> per-instance claim is checked for stability under refitting.**"

Both clauses come from a measurement in this notebook rather than from a
principle: the decoy's rank among the 55 columns, and the disagreement in
section 2.

**What to change before the next lecture.** Set `max_features=None` on the random
forest in section 5, which turns it back into bagging. Watch $\\rho$ rise in
section 6, each member get better, and the ensemble accuracy rise with it — the
trade of the whole lecture, in one keyword.
"""),
    ]
    return cells
