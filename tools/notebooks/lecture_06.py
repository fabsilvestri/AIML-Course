#!/usr/bin/env python3
"""
Lecture 6 — Decision trees. CoverType, Géron Ch. 5.

Exports build() -> list[cell]; tools/make_notebooks.py wraps it.

Structure mirrors the deck: the task and the data -> the impurity derivation,
checked numerically -> growing, reading and tracing a tree -> the two failure
conditions. Every quantity this notebook prints that also appears on a slide is
computed on the same rows, with the same seed, by the same call — the split, the
depth sweep, the 2-D grid, the paired criterion comparison, the traced
prediction and the stability experiment all reproduce tools/figures_app04.py
exactly rather than approximating it.

Anything slower than about twenty seconds states its wall clock, because "no
output" otherwise reads as "it hung" to an audience of mathematicians rather
than engineers.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from make_notebooks import code, header, md, SETUP, SETUP_PROMPT        # noqa: E402
from _prompt import prompt                                # noqa: E402


COVER_LOADER = code('''
# --- the data ----------------------------------------------------------------
# ~30 s and about 11 MB the first time; cached by scikit-learn afterwards.
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split

COVER_NAMES = ["Spruce/Fir", "Lodgepole Pine", "Ponderosa Pine",
               "Cottonwood/Willow", "Aspen", "Douglas-fir", "Krummholz"]

cover = fetch_covtype(as_frame=True)
X_all, y_all = cover.data, cover.target

print(f"{len(X_all):,} patches, {X_all.shape[1]} columns, "
      f"{y_all.nunique()} cover types")
assert X_all.shape == (581012, 54), f"unexpected shape {X_all.shape}"
assert sorted(y_all.unique()) == [1, 2, 3, 4, 5, 6, 7]   # 1-based, not 0-based
''')


SUBSAMPLE = code('''
# A tenth of the data, stratified so the class proportions are preserved
# exactly. A stated compromise for speed, not a silent one: the next lecture
# fits 200-tree ensembles, and on all 581,012 rows that is minutes per fit.
X, _, y, _ = train_test_split(X_all, y_all, train_size=60_000,
                              stratify=y_all, random_state=RANDOM_STATE)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

del cover, X_all, y_all          # 250 MB we no longer need

assert len(X_train) == 48_000 and len(X_test) == 12_000
assert set(X_train.index).isdisjoint(X_test.index), "the split overlaps"
assert X_train.isna().sum().sum() == 0, "unexpected missing values"
print(f"train {len(X_train):,}   test {len(X_test):,}")
''')


def build() -> list:
    cells = header(6, "Decision trees", "", "Chapter 5")

    cells += [
        md("""
Everything on Lecture 6's slides, on the same rows and with the same seed: the
split, the baseline, the two impurity measures, the paired comparison between
them, the depth sweep, the 2-D grid, the tree we ship, one traced prediction,
the per-class errors, and the two failure conditions.

The mathematics is not only stated here, it is **checked**: that both impurity
measures are maximal exactly at the uniform distribution, and that the impurity
decrease is non-negative at every node of a fitted tree.

Runs on free CPU. The download is about 11 MB; the whole notebook is roughly
five minutes on Colab's two cores, and the four slow cells say so. Every timing
here is one machine's and will not be yours — treat them as orders of magnitude,
not as figures.
"""),

        md("## 1 · Setup"), SETUP_PROMPT, SETUP,

        md("""
## 2 · The brief

An agency that manages a national forest publishes a map of **forest cover
type** — which of seven species dominates each 30 by 30 metre patch — over an
area far too large to survey on foot. The map decides where logging is
permitted, which parcels qualify for habitat protection, and how fire risk is
modelled, so it goes into the public record.

The requirement that decides the model, stated before any data is seen:

> Every individual prediction must be accompanied by a **human-readable
> justification**: a statement, in terms of the measured quantities, of why
> *this* patch was classified as *that* species.

Negotiated into something testable:

| Requirement | Testable form |
|---|---|
| readable by a non-specialist | conditions on the measured columns, not transformed ones |
| short enough to check | at most **8** conditions per prediction |
| auditable | same patch, same justification; applicable by hand |

That rules out every model whose explanation is a weighted sum of 54 terms or a
surface with no finite description. It leaves a decision tree, whose path from
root to leaf *is* the justification.
"""),

        md("## 3 · The data"),
        prompt(
            label="the data",
            input="the CoverType dataset from scikit-learn",
            output="581,012 patches by 54 columns, with the seven species named",
            constraint="print the shape and the label domain — the labels are "
                       "1-based, not 0-based, and every index into COVER_NAMES "
                       "below depends on that",
            check="`sorted(y.unique()) == [1..7]`. Assert the label domain, not "
                  "just the shape: it is what tells you the indexing convention "
                  "before anything is built on it",
            **{"try": "index `COVER_NAMES[k]` instead of `COVER_NAMES[k - 1]` in "
                      "the count loop below. It raises IndexError — but only "
                      "because class 7 exists and runs off the end of a "
                      "seven-element list. Now make the same change where only "
                      "the majority class is indexed: nothing raises at all, and "
                      "you quietly get 'Ponderosa Pine' where 'Lodgepole Pine' "
                      "was meant. Which of the two sites would you rather the bug "
                      "had landed in?"}),
        COVER_LOADER,

        md("""
Ten quantitative columns — elevation, aspect, slope, hillshade at three times of
day, and four distances — plus four wilderness-area indicators and forty
soil-type indicators, both already one-hot. Nothing to impute, nothing to encode.
"""),
        prompt(
            label="what the columns are",
            input="the frame",
            output="the ten quantitative column names and the first few rows",
            constraint="show the QUANTITATIVE ten separately — the other 44 are "
                       "already one-hot indicators, and printing all 54 hides "
                       "that structure",
            check="a mean of 0.03 means something quite different for elevation "
                  "and for Soil_Type_23, so separate the measured columns from "
                  "the indicators before looking at any summary",
            **{"try": "`X_all.describe()` on all 54. Count how many rows of that "
                      "table are about a 0/1 indicator"}),
        code('''
print(X_all.columns[:10].tolist())
print()
print(X_all.iloc[:3, :6])
'''),

        md("""
## 4 · Split before you look

Stratified on the label this time — with one class at half a per cent of the
data, an unstratified draw can hand a fold almost none of it.
"""),
        prompt(
            label="a stated compromise, and the split",
            input="all 581,012 rows",
            output="a stratified tenth, then a stratified 48,000 / 12,000 split "
                   "of that",
            constraint="stratify BOTH times — one class is half a per cent of "
                       "the data, and an unstratified draw can hand a fold "
                       "almost none of it",
            check="assert the two sizes, that the indices are disjoint, and that "
                  "nothing is missing. `del` the full frames afterwards: 250 MB "
                  "held for no reason is how a free Colab runtime dies three "
                  "cells later, and the traceback blames the wrong cell",
            **{"try": "drop `stratify=y_all` from the first split and re-run "
                      "cell 5. Cottonwood/Willow's count moves by tens of "
                      "patches out of 227"}),
        SUBSAMPLE,

        md("""
## 5 · Look at the labels

Take thirty seconds over these counts before scrolling. One number here decides
what the baseline is.
"""),
        prompt(
            label="look at the labels",
            input="the training labels",
            output="how many patches of each species, as a count and a share",
            constraint="name the species — `4: 227` is not something anyone can "
                       "think about",
            check="the counts sum to the training size. Print shares beside "
                  "counts: a 48.8% majority is a baseline, and 1.6% is a class "
                  "the impurity criterion will decline to split for",
            **{"try": "compute the same counts on `y_test`. Stratification means "
                      "the shares agree to a tenth of a point — that is what "
                      "`stratify=` bought"}),
        code('''
counts = y_train.value_counts()
for k, n in counts.items():
    print(f"{COVER_NAMES[k - 1]:20s} {n:6,d}   {n / len(y_train):6.1%}")

assert counts.sum() == len(y_train)
print(f"\\ncommonest / rarest ratio: {counts.max() / counts.min():.0f}x")
'''),

        md("""
## 6 · A number to compare against

A metric with nothing to compare it to is decoration. The cheapest possible
classifier predicts the commonest species for every patch in Colorado, forever.
"""),
        prompt(
            label="the anchor",
            input="the training labels",
            output="the test accuracy of always predicting the commonest species",
            constraint="use a real `DummyClassifier` fitted and scored through "
                       "the same interface, not the majority share computed by "
                       "hand",
            check="the answer must equal Lodgepole Pine's share of the TEST set "
                  "to the last decimal, because that is what the classifier "
                  "does. Work it out from the shares above before running",
            **{"try": "`strategy='stratified'` instead. It predicts each "
                      "species with its own frequency and scores far worse — "
                      "guessing in proportion is not the same as guessing well"}),
        code('''
from sklearn.dummy import DummyClassifier

dummy = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
baseline = dummy.score(X_test, y_test)

print(f"always '{COVER_NAMES[dummy.predict(X_test.iloc[:1])[0] - 1]}'"
      f"  ->  {baseline:.1%}")
print("species it can ever predict: 1 of 7")
'''),

        md("""
## 7 · The mathematics — impurity

A node holds $m$ training instances; $p_k$ is the fraction of them in class $k$,
and $\\mathbf{p} = (p_1, \\dots, p_K)$ is the only thing an impurity measure sees.

Both of scikit-learn's criteria have one shape: sum a single strictly concave
$\\varphi$ with $\\varphi(0) = \\varphi(1) = 0$ over the class shares.

$$I(\\mathbf{p}) = \\sum_{k=1}^{K} \\varphi(p_k)
  \\qquad
  \\varphi(p) = p(1-p) \\;\\Rightarrow\\; G = 1 - \\sum_k p_k^2
  \\qquad
  \\varphi(p) = -p\\log_2 p \\;\\Rightarrow\\; H = -\\sum_k p_k \\log_2 p_k$$

$G$ is the probability that two instances drawn from the node with replacement
belong to different classes. $H$ is the number of bits needed to transmit the
class of one instance. Draw them before comparing them.
"""),
        prompt(
            label="Gini and entropy, drawn before they are compared",
            input="p from 0 to 1 on a two-class node",
            output="both impurity curves, entropy halved to put them on one "
                   "scale, and their difference",
            constraint="stop short of 0 and 1 — `log2(0)` is a warning and a "
                       "NaN, and the NaN then propagates silently into the "
                       "difference plot",
            check="both curves are 0 at the ends and both agree at p = 1/2 once "
                  "entropy is halved. So the largest gap is somewhere strictly "
                  "between, and by symmetry there are two of them",
            **{"try": "plot `ent - gini` instead of `ent / 2 - gini`. The "
                      "difference no longer returns to zero at p = 1/2, and the "
                      "shapes are no longer comparable at all"}),
        code('''
import matplotlib.pyplot as plt

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

gap = ent / 2 - gini
print(f"largest gap {gap.max():.4f} at p = {p[gap.argmax()]:.3f}")
'''),

        md("""
### Check the maximum, on our seven classes

Jensen's inequality on a concave $\\varphi$ gives
$I(\\mathbf{p}) \\le K\\,\\varphi(1/K)$, with equality only at the uniform
distribution. So $G_{\\max} = 1 - 1/K$ and $H_{\\max} = \\log_2 K$.

Work both out for $K = 7$ **before** running the cell.
"""),
        prompt(
            label="the two maxima, verified",
            input="K = 7, and a few thousand random class distributions",
            output="the largest G and H any of them reaches, beside the "
                   "predicted maxima at the uniform distribution",
            constraint="draw the distributions from a Dirichlet so they are "
                       "spread over the simplex, not from normalised uniforms, "
                       "which cluster near the centre and would make the bound "
                       "look tighter than it is",
            check="6/7 = 0.857 and log2(7) = 2.807. No random distribution may "
                  "exceed either, and the uniform one must attain both exactly",
            **{"try": "raise the Dirichlet concentration from 0.5 to 50. The "
                      "random draws crowd towards uniform and the measured "
                      "maxima climb to meet the bound"}),
        code('''
def gini_of(p):
    return 1.0 - np.sum(p ** 2, axis=-1)


def entropy_of(p):
    # 0 * log(0) is 0 by the limit, but numpy calls it nan, so mask it out
    q = np.where(p > 0, p, 1.0)
    return -np.sum(np.where(p > 0, p * np.log2(q), 0.0), axis=-1)


K = 7
rng = np.random.default_rng(RANDOM_STATE)
P = rng.dirichlet(np.full(K, 0.5), size=5000)          # spread over the simplex
uniform = np.full(K, 1 / K)

print(f"G  max over 5,000 draws {gini_of(P).max():.4f}   "
      f"at uniform {gini_of(uniform):.4f}   predicted {1 - 1 / K:.4f}")
print(f"H  max over 5,000 draws {entropy_of(P).max():.4f}   "
      f"at uniform {entropy_of(uniform):.4f}   predicted {np.log2(K):.4f}")

assert gini_of(P).max() <= gini_of(uniform) + 1e-12
assert entropy_of(P).max() <= entropy_of(uniform) + 1e-12
'''),

        md("""
### Check that a split never increases impurity

Every instance goes left or right, so for each class the parent share is the
size-weighted mixture of the children's shares. Concavity of $\\varphi$ then
gives, for the whole node,

$$\\Delta I \\;=\\; I_{\\text{parent}} - \\left[\\frac{m_L}{m}I_L + \\frac{m_R}{m}I_R\\right] \\;\\ge\\; 0 .$$

That is a claim about **every internal node of every fitted tree**, so test it on
one.
"""),
        prompt(
            label="the impurity decrease, at every node",
            input="a fitted tree's `tree_` arrays",
            output="the smallest impurity decrease anywhere in the tree, and how "
                   "many nodes were checked",
            constraint="weight each child by `n_node_samples`, not by 1/2 — the "
                       "inequality is about the size-weighted mixture, and with "
                       "equal weights it is simply false",
            check="the minimum must be >= 0 at every internal node, to floating "
                  "point. It should also be > 0 at almost all of them, since a "
                  "split with no gain is one CART would not have made",
            **{"try": "replace the weights by 0.5 and 0.5. Negative decreases "
                      "appear immediately — the weighting is the content of the "
                      "inequality, not bookkeeping"}),
        code('''
from sklearn.tree import DecisionTreeClassifier

probe = DecisionTreeClassifier(max_depth=8, min_samples_leaf=20,
                               random_state=RANDOM_STATE).fit(X_train, y_train)
t = probe.tree_

deltas = []
for node in range(t.node_count):
    left, right = t.children_left[node], t.children_right[node]
    if left == -1:                                   # a leaf splits nothing
        continue
    # tree_.value holds class PROPORTIONS per node in current scikit-learn
    p, pl, pr = (t.value[n][0] for n in (node, left, right))
    m, ml, mr = (t.n_node_samples[n] for n in (node, left, right))
    deltas.append(gini_of(p) - (ml / m) * gini_of(pl) - (mr / m) * gini_of(pr))

deltas = np.array(deltas)
print(f"internal nodes checked {len(deltas)}")
print(f"smallest impurity decrease {deltas.min():.3e}")
assert deltas.min() >= -1e-12, "a split increased impurity — check the weights"
'''),

        md("""
### Does the choice of criterion matter?

Both criteria are Schur-concave, so whenever one split's children are uniformly
more concentrated than another's, the two **must** rank them the same way. They
are free to differ only on incomparable pairs — and in the two-class case by at
most 0.055.

Measure it, **paired**: both criteria on the same resample each time, so the
resample-to-resample noise cancels.

⏱ **about 40 seconds** — forty fits on 38,400 rows each.
"""),
        prompt(
            label="⏱ 40 s — does the criterion matter",
            input="twenty 80% resamples of the training set",
            output="how often the root feature agrees, how often the 12,000 "
                   "predictions agree, and the paired accuracy difference",
            constraint="PAIRED — both criteria on the same resample each time, "
                       "and both scored on the same held-out patches",
            check="the root split is chosen from 48,000 patches and wins by a "
                  "wide margin, so the derivation predicts the root feature "
                  "agrees every time. Predict that before running, then read "
                  "the effect size beside the sign count",
            **{"try": "raise `max_depth` from 8 to 16. The agreement between the "
                      "two criteria falls, because deeper nodes are decided by "
                      "smaller samples and closer calls"}),
        code('''
rows = []
for seed in range(20):
    Xs, _, ys, _ = train_test_split(X_train, y_train, train_size=0.8,
                                    stratify=y_train, random_state=seed)
    rec = {}
    for crit in ("gini", "entropy"):
        clf = DecisionTreeClassifier(criterion=crit, max_depth=8,
                                     random_state=RANDOM_STATE).fit(Xs, ys)
        rec[crit] = float((clf.predict(X_test) == y_test).mean())
        rec[crit + "_root"] = int(clf.tree_.feature[0])
        rec[crit + "_pred"] = clf.predict(X_test)
    rec["agree"] = float((rec["gini_pred"] == rec["entropy_pred"]).mean())
    rows.append(rec)

g = np.array([r["gini"] for r in rows])
e = np.array([r["entropy"] for r in rows])
diff = (e - g) * 100

print(f"same root feature       {sum(r['gini_root'] == r['entropy_root'] for r in rows)} of 20")
print(f"predictions agreeing    {np.mean([r['agree'] for r in rows]):.1%}")
print(f"mean accuracy, gini     {g.mean():.1%}")
print(f"mean accuracy, entropy  {e.mean():.1%}")
print(f"entropy - gini          {diff.mean():+.2f} +/- {diff.std():.2f} points")
print(f"resamples gini won      {(diff < 0).sum()} of 20")
'''),
        md("""
Three statements, all true, and only the third is a recommendation:

1. The effect is **real** — the mean sits about five standard errors from zero.
2. The effect is **tiny** — a fraction of a point, against the eight points
   `max_depth` is worth two sections from now.
3. So **do not spend your tuning budget here.** Leave the default.

"Statistically detectable" and "worth acting on" are different claims and need
different evidence. Twenty paired resamples give you both; one run gives neither.
"""),

        md("""
## 8 · One tree, with no constraints at all

A decision tree asks a sequence of questions of the form `x[k] <= t`, and the
leaf it reaches predicts the majority class of the training patches that reached
the same leaf. Look at its **shape** before its score.
"""),
        prompt(
            label="one tree, unconstrained",
            input="the 48,000 training patches",
            output="its depth, its leaf count, and its training accuracy",
            constraint="no `max_depth` at all — this is the tree the algorithm "
                       "grows when nothing stops it",
            check="leaves against rows. With thousands of leaves for 48,000 "
                  "patches the tree is a lookup table, and its training "
                  "accuracy is therefore 100% by construction rather than by "
                  "merit",
            **{"try": "`min_samples_leaf=5`. The leaf count falls by more than "
                      "half and the training accuracy stops being 100% — the "
                      "lookup table needs leaves of one to work"}),
        code('''
free = DecisionTreeClassifier(random_state=RANDOM_STATE).fit(X_train, y_train)

print(f"depth           {free.get_depth()}")
print(f"leaves          {free.get_n_leaves():,}")
print(f"train accuracy  {free.score(X_train, y_train):.1%}")

assert free.get_n_leaves() > 1000, "expected a large, unconstrained tree"
'''),
        md("""
Measure it honestly before drawing any conclusion from that 100%.

⏱ **about 30 seconds** — five fits on 38,400 rows each.
"""),
        prompt(
            label="⏱ 30 s — measure it honestly",
            input="the unconstrained tree and the training rows",
            output="cross-validated accuracy with the fold range, and the test "
                   "score",
            constraint="stratified folds, and report the RANGE as well as the "
                       "mean",
            check="a mean built from folds spanning four points is a different "
                  "object from one built from folds spanning half a point, so "
                  "print the fold minimum and maximum beside it",
            **{"try": "`shuffle=False` on the splitter. The rows arrive from a "
                      "stratified split so the effect is small here — on a "
                      "sorted frame it would not be"}),
        code('''
from sklearn.model_selection import StratifiedKFold, cross_val_score

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
free_cv = cross_val_score(free, X_train, y_train, cv=cv, n_jobs=-1)

print(f"cross-validated {free_cv.mean():.1%}  (folds {free_cv.min():.1%} - "
      f"{free_cv.max():.1%})")
print(f"test            {free.score(X_test, y_test):.1%}")
'''),

        md("""
## 9 · How long a justification actually is

The tree *can* justify a prediction — the path from root to leaf is a list of
conditions. Count them, and compare with the eight the brief allows.
"""),
        prompt(
            label="count the conditions per prediction",
            input="a fitted tree and the 12,000 test patches",
            output="the mean and maximum number of conditions applied per "
                   "prediction",
            constraint="count nodes VISITED minus one — the leaf is not a "
                       "condition, and an off-by-one here silently reports "
                       "depth + 1",
            check="assert the maximum path length equals the tree's own "
                  "`get_depth()`. Two independent routes to the same number is "
                  "how you find out that `decision_path` counts the leaf",
            **{"try": "drop the `- 1`. The assert fires, which is the whole "
                      "reason it is there"}),
        code('''
def path_lengths(tree, X):
    """Conditions applied per prediction: nodes visited, minus the leaf."""
    visited = np.asarray(tree.decision_path(X).sum(axis=1)).ravel()
    return visited - 1


free_len = path_lengths(free, X_test)
print(f"unconstrained tree:  mean {free_len.mean():.2f}   max {free_len.max()}")
print(f"the brief allows:    8")

assert free_len.max() == free.get_depth()
'''),

        md("""
## 10 · What does depth actually buy?

The brief fixes `max_depth`. Before accepting that, measure what it costs,
because the number belongs in the report to the agency.

⏱ **about 90 seconds** — eighteen depths, five folds each, plus one full fit per
depth for the leaf count.
"""),
        prompt(
            label="⏱ 90 s — what does depth buy",
            input="depths 1 to 18",
            output="cross-validated accuracy and leaf count at each depth",
            constraint="sweep PAST the depth we are allowed to use — the rows we "
                       "cannot pick are what tell the agency what its "
                       "requirement costs",
            check="the leaf count roughly doubles per level while the depth "
                  "limit binds, so it should be near 2**d at small d and fall "
                  "away from it once the data runs out. Check that at d = 1, 2, 3",
            **{"try": "print the training score beside the CV score. The two "
                      "separate steadily with depth, and the gap is the "
                      "overfitting the depth limit is removing"}),
        code('''
rows = []
for d in range(1, 19):
    clf = DecisionTreeClassifier(max_depth=d, random_state=RANDOM_STATE)
    acc = cross_val_score(clf, X_train, y_train, cv=cv, n_jobs=-1).mean()
    leaves = clf.fit(X_train, y_train).get_n_leaves()
    rows.append({"max_depth": d, "cv_accuracy": acc, "leaves": leaves})

depth_table = pd.DataFrame(rows).set_index("max_depth")
print(depth_table.to_string(float_format=lambda v: f"{v:.4f}"))

cost = 100 * (depth_table.loc[18, "cv_accuracy"] - depth_table.loc[8, "cv_accuracy"])
print(f"\\nprice of the eight-condition requirement: {cost:.1f} points")
'''),
        prompt(
            label="the price, drawn",
            input="the depth table",
            output="accuracy against depth, and leaves against depth on a log "
                   "axis",
            constraint="log scale on the leaf count — it spans three orders of "
                       "magnitude, and on a linear axis every depth below 12 is "
                       "flat on the floor",
            check="mark the constrained value on both panels. A sweep with no "
                  "line at the value you actually chose makes the reader do the "
                  "lookup",
            **{"try": "a linear leaf axis. Everything below depth 12 collapses "
                      "onto the x-axis and the doubling is invisible"}),
        code('''
fig, ax = plt.subplots(1, 2, figsize=(11, 3.4))
ax[0].plot(depth_table.index, depth_table["cv_accuracy"] * 100, "-o")
ax[0].axvline(8, ls="--", color="green")
ax[0].set_xlabel("max_depth"); ax[0].set_ylabel("CV accuracy (%)")
ax[1].semilogy(depth_table.index, depth_table["leaves"], "-o", color="purple")
ax[1].axvline(8, ls="--", color="green")
ax[1].set_xlabel("max_depth"); ax[1].set_ylabel("leaves")
plt.tight_layout(); plt.show()
'''),
        md("""
Cross-validated accuracy is still climbing at depth 18, and the number of rules
is climbing with it. That difference is the measured price of the requirement.
Bring it to the agency — perhaps they will trade a condition or two for it. That
is a conversation, not a `GridSearchCV`.

**Cross-validation does not get a vote on `max_depth` here**, because it is not
optimising the thing the agency is buying.
"""),

        md("""
## 11 · Tune what is left, inside the constraint

`min_samples_leaf` is still ours. Search the whole grid anyway — including the
depths we may not use — because the rows we cannot pick are what tell the agency
what its requirement costs.

⏱ **about 2 minutes** — 30 combinations, five folds each.
"""),
        prompt(
            label="⏱ 2 min — tune what is left",
            input="a 2-D grid of max_depth and min_samples_leaf",
            output="the full grid as a pivot table, not just the winner",
            constraint="search the WHOLE grid including depths we may not use, "
                       "and print the table rather than `best_params_` alone",
            check="along the depth-8 row the leaf size should barely matter, "
                  "because the depth limit binds first; down the unlimited row "
                  "it should matter enormously. Predict both before reading the "
                  "table",
            **{"try": "read the two rows as separate 1-D sweeps. They disagree "
                      "about how much `min_samples_leaf` is worth, which is why "
                      "the grid is two-dimensional"}),
        code('''
from sklearn.model_selection import GridSearchCV

grid = {"max_depth": [4, 6, 8, 10, 12, None],
        "min_samples_leaf": [1, 5, 20, 50, 200]}

search = GridSearchCV(DecisionTreeClassifier(random_state=RANDOM_STATE),
                      grid, cv=cv, n_jobs=-1).fit(X_train, y_train)

res = pd.DataFrame(search.cv_results_)
pivot = res.pivot_table(index="param_max_depth", columns="param_min_samples_leaf",
                        values="mean_test_score", dropna=False)
print(pivot.to_string(float_format=lambda v: f"{v:.4f}"))
print(f"\\nbest overall: {search.best_params_}  {search.best_score_:.4f}")
'''),

        md("""
## 12 · The model we ship — overruling the grid

The grid's answer under the cap is `min_samples_leaf=1`. **We are not going to
ship it.**

The model states its justification as *"91% of the 463 training patches in this
leaf"*. With a minimum leaf of 1 that sentence can become *"100% of the 1"* — a
single surveyed patch wearing the grammar of evidence. The brief asks for a
justification that can be audited, and that is not one.

So we overrule the grid, for exactly the reason we overruled it on depth: **when
the brief constrains the model, the grid does not get a vote.** The cell below
prints what that costs, and the number goes to the agency with everything else.
"""),
        prompt(
            label="overruling the grid, deliberately",
            input="the depth-8 row of the grid, and max_depth 8 with a minimum "
                  "leaf of 20",
            output="leaves, columns consulted, conditions per prediction, "
                   "training accuracy, and the cross-validated points the choice "
                   "costs",
            constraint="`min_samples_leaf=20` comes from the BRIEF, not from the "
                       "grid — the grid's answer under the cap is 1",
            check="assert no prediction uses more than 8 conditions. That assert "
                  "is the brief written as code, and it belongs in the cell that "
                  "ships the model",
            **{"try": "raise the leaf minimum to 200. The smallest leaf follows "
                      "it, the tree loses a point and a half of accuracy, and "
                      "the depth cap still binds first"}),
        code('''
AUDITABLE_LEAF = 20        # the brief, not the grid — see the note above

tree = DecisionTreeClassifier(max_depth=8, min_samples_leaf=AUDITABLE_LEAF,
                              random_state=RANDOM_STATE).fit(X_train, y_train)

used = {int(f) for f in tree.tree_.feature if f >= 0}
lens = path_lengths(tree, X_test)
cv_here = pivot.loc[8, AUDITABLE_LEAF]
cv_pref = pivot.loc[8].max()

print(f"leaves                  {tree.get_n_leaves()}")
print(f"columns consulted       {len(used)} of {X_train.shape[1]}")
print(f"conditions, mean / max  {lens.mean():.2f} / {lens.max()}")
print(f"train accuracy          {tree.score(X_train, y_train):.1%}")
print(f"cross-validated         {cv_here:.4f}   "
      f"(the grid preferred {cv_pref:.4f})")
print(f"price of auditability   {100 * (cv_pref - cv_here):.2f} points")

assert lens.max() <= 8, "the brief is violated"
'''),
        md("""
The gap between training and cross-validated accuracy has almost vanished: the
depth limit did not only shorten the justification, it removed nearly all of the
overfitting as a side effect.

Which raises a question this lecture answers only at the end: if the constrained
tree barely overfits, why is it still nearly nine points worse than the
unconstrained one?
"""),

        md("""
## 13 · Read the tree

Two routes. `export_graphviz` writes a `.dot` file, which then needs the `dot`
binary — not a Python package, and not installed on a stock Colab runtime. The
call succeeds, writes a file, and nothing renders.

*(Not examinable: this is tooling, not machine learning.)*

`plot_tree` draws into a matplotlib axis and works anywhere matplotlib does.
`max_depth=2` in the drawing call is essential — all eight levels at a readable
size is about two metres of paper.
"""),
        prompt(
            label="draw it",
            input="the shipped tree",
            output="the top three levels, drawn into a matplotlib axis",
            constraint="`max_depth=2` in the PLOT call, not in the estimator — "
                       "we are drawing less of the tree we shipped, not "
                       "shipping a smaller tree",
            check="the root threshold should land on a half-integer. CART puts "
                  "a split midway between two adjacent observed values, so "
                  "nothing in the data sits exactly on one",
            **{"try": "`max_depth=4` in the plot call. The node text becomes "
                      "unreadable at any figure size that fits on a slide, "
                      "which is the practical limit this argument runs into"}),
        code('''
from sklearn.tree import export_text, plot_tree

short = [c.replace("Horizontal_Distance_To_", "HDist_")
          .replace("Vertical_Distance_To_", "VDist_")
          .replace("Hillshade_", "Shade_")
          .replace("Wilderness_Area_", "Wild_")
          .replace("Soil_Type_", "Soil_")
          .replace("Hydrology", "Hydro")
          .replace("Roadways", "Road")
          .replace("Fire_Points", "Fire") for c in X_train.columns]

fig, ax = plt.subplots(figsize=(13, 4.5))
plot_tree(tree, max_depth=2, feature_names=short, class_names=COVER_NAMES,
          filled=True, rounded=True, impurity=False, proportion=True,
          precision=1, fontsize=8, ax=ax)
plt.show()

print(f"root question: {short[tree.tree_.feature[0]]} "
      f"<= {tree.tree_.threshold[0]:.1f}")
'''),
        prompt(
            label="the version you can paste into an email",
            input="the same tree",
            output="the top three levels as indented text",
            constraint="no plotting library at all — for a model whose selling "
                       "point is that a person can read it, that matters more "
                       "than it sounds",
            check="the text and the drawing must ask the same root question. If "
                  "they differ you are looking at two different trees",
            **{"try": "`show_weights=True`. Each line gains the class counts, "
                      "which is what turns the text from a picture of the model "
                      "into an auditable record of it"}),
        code('''
print(export_text(tree, feature_names=short, class_names=COVER_NAMES,
                  max_depth=2, decimals=0))
'''),

        md("""
## 14 · Trace one prediction, all the way down

The entire justification mechanism is three arrays: `tree_.children_left`,
`tree_.feature` and `tree_.threshold`.
"""),
        prompt(
            label="trace one prediction all the way down",
            input="one test patch and the fitted tree",
            output="the conditions it satisfied, and the class distribution of "
                   "its leaf",
            constraint="walk `children_left` / `feature` / `threshold` by hand "
                       "— the whole justification mechanism is those three "
                       "arrays, and reimplementing it is how you learn that",
            check="assert at most 8 conditions AND that the traced class agrees "
                  "with `predict()`. A justification that disagrees with the "
                  "model it claims to explain is worse than no justification",
            **{"try": "another patch — change `i`. The number of conditions is "
                      "8 for almost every one, because the depth cap binds "
                      "before the purity does"}),
        code('''
def justify(tree, x, names):
    """The conditions one instance satisfied on its way to a leaf."""
    t, node, out = tree.tree_, 0, []
    while t.children_left[node] != -1:
        f, thr = int(t.feature[node]), float(t.threshold[node])
        left = x[f] <= thr
        out.append((names[f], "<=" if left else ">", thr, float(x[f])))
        node = t.children_left[node] if left else t.children_right[node]
    return out, node


i = 27
conditions, leaf = justify(tree, X_test.iloc[i].values, short)

for n, (name, op, thr, val) in enumerate(conditions, 1):
    print(f"{n}. {name:12s} = {val:8,.0f}  {op:2s} {thr:8,.0f}")

dist = tree.tree_.value[leaf][0]
print(f"-> {COVER_NAMES[int(dist.argmax())]}  ({dist.max():.0%} of the "
      f"{int(tree.tree_.n_node_samples[leaf])} training patches in this leaf)")
print(f"   true class: {COVER_NAMES[int(y_test.iloc[i]) - 1]}")

assert len(conditions) <= 8
assert int(dist.argmax()) + 1 == tree.predict(X_test.iloc[[i]])[0], \\
    "the trace disagrees with predict()"
'''),
        md("""
### Read the leaf carefully

That is a statement about the **training patches in the leaf**, not a probability
that this patch is that species. `predict_proba` returns exactly those leaf
proportions, so a tree's "probabilities" are piecewise constant and identical for
every patch reaching the same leaf.

The honest sentence: *"Of the training patches that satisfied these eight
conditions, 91% were Krummholz."* Not *"this patch is 91% likely to be
Krummholz."*

A leaf built on twenty patches and a leaf built on four thousand produce the same
kind of sentence and deserve very different amounts of trust — which is why the
count belongs in the justification.
"""),
        prompt(
            label="how much is each leaf built on",
            input="the training patches routed through the tree",
            output="the smallest, median and largest leaf",
            constraint="drop the zero counts — `bincount` returns a slot for "
                       "every node id, and the internal nodes are all zeros",
            check="the minimum leaf size must equal `AUDITABLE_LEAF` exactly. If "
                  "it is smaller, `min_samples_leaf` is not doing what you think "
                  "it is",
            **{"try": "keep the zeros. The reported minimum becomes 0 and the "
                      "median collapses, because half the slots are internal "
                      "nodes"}),
        code('''
sizes = np.bincount(tree.apply(X_train))
sizes = sizes[sizes > 0]
print(f"leaves {len(sizes)}   smallest {sizes.min()}   "
      f"median {int(np.median(sizes))}   largest {sizes.max():,}")

assert sizes.min() == AUDITABLE_LEAF
'''),

        md("""
## 15 · The test set. Once.

Everything above used only training data and cross-validated folds.
"""),
        prompt(
            label="the test set, once",
            input="the 12,000 held-out patches",
            output="accuracy against the baseline, and per-class precision and "
                   "recall",
            constraint="per-class numbers, not just the headline — the headline "
                       "is an average over seven very differently sized classes",
            check="the per-class recalls, weighted by class share, must "
                  "reproduce the overall accuracy. That identity is what makes "
                  "'accuracy hides the rare classes' precise rather than "
                  "rhetorical",
            **{"try": "`digits=1`. Aspen's recall rounds to 0.0, which is a "
                      "different claim from 0.026 and would be repeated by "
                      "anyone reading the table"}),
        code('''
from sklearn.metrics import (ConfusionMatrixDisplay, classification_report,
                             confusion_matrix)

acc = tree.score(X_test, y_test)
print(f"test accuracy {acc:.1%}   (baseline {baseline:.1%})")
print()
print(classification_report(y_test, tree.predict(X_test),
                            target_names=COVER_NAMES, digits=3, zero_division=0))
'''),
        prompt(
            label="the confusion matrix, row-normalised",
            input="the tree and the test patches",
            output="a seven by seven matrix normalised by row",
            constraint="`normalize='true'` — each row then reads as 'of the "
                       "patches that really were this species, where did they "
                       "go?'",
            check="every row sums to 1, and the diagonal is the recall column of "
                  "the report above. Read the Aspen row: nearly all of it should "
                  "be in one off-diagonal cell",
            **{"try": "`normalize='pred'`. Each COLUMN now sums to 1 and the "
                      "diagonal becomes precision instead of recall — the same "
                      "matrix answering a different question"}),
        code('''
fig, ax = plt.subplots(figsize=(7, 6))
ConfusionMatrixDisplay.from_estimator(
    tree, X_test, y_test, display_labels=COVER_NAMES, normalize="true",
    cmap="Blues", values_format=".2f", xticks_rotation=45, ax=ax, colorbar=False)
plt.tight_layout(); plt.show()

cm = confusion_matrix(y_test, tree.predict(X_test), normalize="true")
aspen = COVER_NAMES.index("Aspen")
print(f"Aspen recall {cm[aspen, aspen]:.1%};  "
      f"{cm[aspen].max():.1%} of Aspen goes to "
      f"{COVER_NAMES[int(cm[aspen].argmax())]}")
'''),
        md("""
**Why the tree cannot see Aspen.** $J$ weights each child's impurity by how many
instances it holds. A split that isolates a class at 1.6% of the node moves the
weighted average by almost nothing, so the greedy search never chooses one. With
`max_depth=8` there are at most 256 leaves to allocate across seven species, and
the algorithm spends them where the instances are.

That is a property of the objective, not a bug in the implementation. Read the
definition of $J$ again and you could have predicted it.
"""),
        prompt(
            label="what class weighting buys, and costs",
            input="the same tree with `class_weight='balanced'`",
            output="overall accuracy and Aspen recall, each beside its "
                   "unweighted value",
            constraint="report BOTH numbers — one goes up and one goes down, and "
                       "quoting either alone is an argument rather than a "
                       "measurement",
            check="`balanced` makes every class carry the same total weight, so "
                  "Aspen's 1.6% share is scaled up by about 1/(7 x 0.016) ~ 9. "
                  "Predict the direction of both numbers before running",
            **{"try": "pass an explicit dict weighting only Aspen. Its recall "
                      "rises and the other six move much less than under "
                      "`balanced` — targeted reweighting is cheaper than "
                      "uniform reweighting"}),
        code('''
balanced = DecisionTreeClassifier(max_depth=8, min_samples_leaf=AUDITABLE_LEAF,
                                  class_weight="balanced",
                                  random_state=RANDOM_STATE).fit(X_train, y_train)
rep = classification_report(y_test, balanced.predict(X_test),
                            target_names=COVER_NAMES, output_dict=True,
                            zero_division=0)

print("                accuracy   Aspen recall")
print(f"default weights {acc:8.1%}   {cm[aspen, aspen]:8.1%}")
print(f"balanced        {balanced.score(X_test, y_test):8.1%}   "
      f"{rep['Aspen']['recall']:8.1%}")
print("\\nA different model, answering a different question. Which one the "
      "agency wants is not a machine learning decision.")
'''),

        md("""
## 16 · Failure condition 1 — every boundary is axis-aligned

Every split is $x_k \\le t_k$, so every decision boundary is perpendicular to an
axis. Scaling a column is harmless, because a split is invariant under any
strictly increasing transformation of that column alone. **Rotating** the space
is not: it mixes columns, and the family of candidate splits is not
rotation-invariant.

Two identical problems, one rotated by 45 degrees, is the cheapest way to see it.
"""),
        prompt(
            label="rotate the problem and refit",
            input="a two-class problem separable by a single vertical line, and "
                  "the same points rotated by 45 degrees",
            output="the depth and leaf count of a tree fitted to each",
            constraint="rotate BOTH the training and the test points by the same "
                       "matrix — this is the same problem in new coordinates, "
                       "not a harder one",
            check="a vertical boundary needs exactly ONE split, so depth 1 and "
                  "two leaves. After rotation the same boundary is a diagonal "
                  "and has to be built out of a staircase, so both numbers must "
                  "grow. Predict roughly how much before running",
            **{"try": "rotate by 90 degrees instead of 45. Nothing changes at "
                      "all — a right angle maps axes onto axes, so the split "
                      "family is preserved"}),
        code('''
rot_rng = np.random.default_rng(RANDOM_STATE)
pts = rot_rng.uniform(-1, 1, size=(2000, 2))
lab = (pts[:, 0] > 0).astype(int)                 # one vertical boundary

theta = np.pi / 4
R = np.array([[np.cos(theta), -np.sin(theta)],
              [np.sin(theta), np.cos(theta)]])

for name, P in [("axis-aligned", pts), ("rotated 45 degrees", pts @ R.T)]:
    t45 = DecisionTreeClassifier(random_state=RANDOM_STATE).fit(P, lab)
    print(f"{name:20s} depth {t45.get_depth():3d}   "
          f"leaves {t45.get_n_leaves():4d}")
'''),

        md("""
## 17 · Failure condition 2 — the rules are not stable

You have a model whose every prediction comes with a rule a person can read. Your
neighbour fits the same model, the same hyperparameters, the same seed — to
*almost* the same training set.

**How similar are the two sets of rules?** Write down a guess before you run the
next cell.

⏱ **about 30 seconds** — twenty fits on 43,200 rows each.
"""),
        prompt(
            label="⏱ 30 s — twenty nearly-identical training sets",
            input="twenty 90% subsamples of the same training rows",
            output="each tree's accuracy, and each tree's 12,000 predictions",
            constraint="same hyperparameters and the same estimator seed every "
                       "time — only the ROWS change, so any difference is "
                       "attributable to the data",
            check="assert twenty prediction vectors of 12,000 each before "
                  "anything is compared. Any two subsamples share about 80% of "
                  "their rows, so the accuracies should barely move",
            **{"try": "`train_size=0.5`. The accuracy still barely moves and the "
                      "disagreement in the next cell rises sharply — which is "
                      "the whole point"}),
        code('''
trees, preds, roots, accs = [], [], [], []
for seed in range(20):
    Xs, _, ys, _ = train_test_split(X_train, y_train, train_size=0.9,
                                    stratify=y_train, random_state=1000 + seed)
    t20 = DecisionTreeClassifier(max_depth=8, min_samples_leaf=AUDITABLE_LEAF,
                                 random_state=RANDOM_STATE).fit(Xs, ys)
    trees.append(t20)
    preds.append(t20.predict(X_test))
    roots.append((int(t20.tree_.feature[0]),
                  round(float(t20.tree_.threshold[0]), 1)))
    accs.append(t20.score(X_test, y_test))

accs = np.array(accs)
assert len(preds) == 20 and all(len(p) == 12_000 for p in preds)
print(f"accuracy  mean {accs.mean():.2%}  sd {accs.std():.2%}  "
      f"({accs.min():.2%} - {accs.max():.2%})")
'''),
        md("""
A spread of a point across twenty refits. Watching only the headline number, you
would conclude the model is completely stable. That is the flattering half.
"""),
        prompt(
            label="the other half of the answer",
            input="the twenty prediction vectors",
            output="pairwise disagreement over all 190 pairs, and the share of "
                   "patches all twenty agree on",
            constraint="compare PREDICTIONS, not accuracies — two models with "
                       "identical accuracy can disagree on one patch in eleven",
            check="20 choose 2 is 190 pairs, so the disagreement array must have "
                  "exactly that length. A stable metric is not a stable model; "
                  "it is evidence you measured the wrong thing",
            **{"try": "compare the accuracies pairwise instead. The spread is "
                      "under a point and tells you nothing about the "
                      "substitutions underneath it"}),
        code('''
P = np.array(preds)
disagree = np.array([(P[i] != P[j]).mean()
                     for i in range(20) for j in range(i + 1, 20)])
unanimous = (P == P[0]).all(axis=0).mean()

assert len(disagree) == 190
print(f"pairwise disagreement    {disagree.mean():.1%}  "
      f"({disagree.min():.1%} - {disagree.max():.1%}) over 190 pairs")
print(f"patches all 20 agree on  {unanimous:.1%}")
'''),
        prompt(
            label="what is stable and what is not",
            input="the twenty trees",
            output="the root feature, the number of distinct root thresholds, "
                   "the leaf counts and the columns consulted",
            constraint="separate the root FEATURE from the root THRESHOLD — the "
                       "feature is the same every time and the threshold is not",
            check="the root is chosen from 43,200 patches and wins by a wide "
                  "margin, so the feature should be identical 20 times out of "
                  "20 — the same fact that made gini and entropy agree about it",
            **{"try": "look at the SECOND level instead of the root. The "
                      "agreement drops, because those nodes are decided by "
                      "smaller samples"}),
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

Both results are consistent. A tree is a **hierarchy**. The root split is chosen
from tens of thousands of patches and wins by a wide margin; a node eight levels
down was chosen from a few hundred, where two candidates are often separated by a
hair. Change one row and everything below that node is a different tree. Accuracy
is an average over 12,000 patches, and averages hide substitutions.

**A stable metric is not a stable model. It is evidence that you measured the
wrong thing.**

This is *variance* in the sense of Lecture 5's decomposition, and failure
condition 1 was the same fact in a different hat: rotating the data changes which
axis-aligned split wins by a hair, and the subtree below it changes completely.

The next lecture derives what averaging does to variance, builds the models that
exploit it — and shows what the repair costs: the justification this notebook
spent fifteen sections earning.
"""),

        md("""
## 18 · Where we are

| Model | Test accuracy | Conditions per justification |
|---|---|---|
| always "Lodgepole Pine" | 48.8% | 0 |
| depth-8 tree, leaf 20 — **ours** | 73.0% | 7.97 |
| unconstrained tree | 82.6% | 17.88 |

Row two meets the brief, and every one of its predictions comes with a reason a
surveyor could check on site.

**What to change before the next lecture.** Raise `max_depth` from 8 to 12 in
section 12 and re-run sections 12 to 15. Accuracy rises by about four points, the
justification roughly doubles in length, and the pair is the trade this whole
lecture is about.
"""),
    ]
    return cells
