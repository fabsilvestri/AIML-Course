#!/usr/bin/env python3
"""
Lecture 7 — A model the regulator will accept. Build, CoverType, Géron Ch. 5.

Exports build() -> list[cell]; tools/make_notebooks.py wraps it.

Structure mirrors the deck: brief -> data -> metric -> anchor -> commitment ->
build -> the worked assistant failure -> read the tree -> record the number.
Every structural step is followed by an assertion, and anything slower than
about twenty seconds states its wall clock, because "no output" otherwise reads
as "it hung" to an audience of mathematicians rather than engineers.
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
# exactly. This is a stated compromise for speed, not a silent one: a 200-tree
# ensemble on all 581,012 rows takes minutes per fit.
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
    cells = header(7, "A model the regulator will accept", "build", "Chapter 5")

    cells += [
        md("## 1 · Setup"), SETUP_PROMPT, SETUP,

        md("""
## 2 · The brief

You work for the agency that manages a national forest. It has to publish a map
of **forest cover type** — which of seven species dominates each 30 by 30 metre
patch — over an area far too large to survey on foot.

The map goes into the public record. It decides where logging is permitted,
which parcels qualify for habitat protection, and how fire risk is modelled.

The regulator's constraint, and it is not negotiable:

> Every individual prediction must be accompanied by a **human-readable
> justification**: a statement, in terms of the measured quantities, of why
> *this* patch was classified as *that* species.

Negotiated into something testable:

| Requirement | Testable form |
|---|---|
| readable by a non-specialist | conditions on the measured columns, not transformed ones |
| short enough to check | at most **8** conditions per prediction |
| auditable | same patch, same justification; applicable by hand |

Note what that does to the model choice. It is made before we look at the data.
"""),

        md("## 3 · The data"), prompt(
                                      label="the data",
                                      input="the CoverType dataset from scikit-learn",
                                      output="581,012 patches by 54 columns, with the seven species named",
                                      constraint="print the shape and the label domain — the labels are 1-based, not 0-based, and every index into COVER_NAMES below depends on that",
                                      check="assert the shape and that the classes are exactly 1 through 7",
                                      left_open="that 1-based labels are a trap the whole notebook has to work around. `COVER_NAMES[k - 1]` appears in five later cells and each one is a chance to drop the minus one.",
                                      student="`COVER_NAMES[y]`, which is off by one everywhere and never raises until a 7 shows up. Six of the seven species then have the wrong name in every table and plot.",
                                      catch="assert the label domain, not just the shape. `sorted(y.unique()) == [1..7]` is what tells you the indexing convention before you build anything on it."),
                               COVER_LOADER,

        md("""
Ten quantitative columns — elevation, aspect, slope, hillshade at three times of
day, and four distances — plus four wilderness-area indicators and forty soil
type indicators, both already one-hot. Nothing to impute, nothing to encode.
"""),
        prompt(
            label="what the columns are",
            input="the frame",
            output="the ten quantitative column names and the first few rows",
            constraint="show the QUANTITATIVE ten separately — the other 44 are already one-hot indicators and printing all 54 hides that structure",
            left_open="that there is nothing to impute and nothing to encode here. That is unusual, and it is why this application can spend its time on the model rather than the frame.",
            student="running `.describe()` on all 54 columns and reading a wall of numbers in which the four wilderness indicators and forty soil indicators look exactly like measurements.",
            catch="separate the measured columns from the indicator columns before you look at anything. A mean of 0.03 means something quite different for elevation and for Soil_Type_23."),
        code('''
print(X_all.columns[:10].tolist())
print()
print(X_all.iloc[:3, :6])
'''),

        md("""
## 4 · Split before you look

Same rule as the first application, and stratified on the label this time —
with one class at half a per cent of the data, an unstratified split can hand a
fold almost none of it.
"""),
        prompt(
               label="a stated compromise, and the split",
               input="all 581,012 rows",
               output="a stratified tenth, then a stratified 48,000 / 12,000 split of that",
               constraint="stratify BOTH times — one class is half a per cent of the data, and an unstratified draw can hand a fold almost none of it",
               check="assert the two sizes, that the indices are disjoint, and that nothing is missing",
               left_open="that subsampling is a compromise for speed and the comment says so out loud. A 200-tree ensemble on all 581,012 rows takes minutes per fit; every number in this notebook is a 60,000-row number.",
               student="subsampling silently and reporting the accuracy as if it were the full dataset's. The compromise is fine; leaving it unstated is not.",
               catch="`del` the full frames afterwards. 250 MB held for no reason is how a free Colab runtime dies three cells later, and the traceback blames the wrong cell."),
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
            constraint="name the species — `4: 1728` is not something anyone can think about",
            check="assert the counts sum to the training size, and print the commonest-to-rarest ratio",
            left_open="what the ratio implies. One number in this table decides what the baseline is, and another decides that Aspen will be invisible in the confusion matrix twelve sections from now.",
            student="scrolling past. The imbalance here is the cause of the Aspen row collapsing later, and this is the cell where it was visible.",
            catch="always print class shares beside class counts. A 48.8% majority is a baseline; 1.6% is a class your impurity criterion will decline to split for."),
        code('''
counts = y_train.value_counts()
for k, n in counts.items():
    print(f"{COVER_NAMES[k - 1]:20s} {n:6,d}   {n / len(y_train):6.1%}")

assert counts.sum() == len(y_train)
print(f"\\ncommonest / rarest ratio: {counts.max() / counts.min():.0f}x")
'''),

        md("""
## 6 · A number to compare against

Rule 2 of this course: *a metric with nothing to compare it to is decoration.*
The cheapest possible classifier predicts the commonest species for every patch
in Colorado, forever.
"""),
        prompt(
            label="the anchor",
            input="the training labels",
            output="the accuracy of always predicting the commonest species",
            constraint="use a real DummyClassifier fitted and scored through the same interface, not the majority share computed by hand",
            left_open="how bad 48.8% actually is. It is a number, not a verdict — the point is that every accuracy below has to be read against it.",
            student="reporting 73% as good without ever computing this. The distance from 48.8 to 73.3 is what was earned; the 48.8 was free.",
            catch="print what the dummy can ever predict: one species of seven. An anchor that scores well while being obviously useless is exactly what makes it useful as an anchor."),
        code('''
from sklearn.dummy import DummyClassifier

dummy = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
baseline = dummy.score(X_test, y_test)

print(f"always '{COVER_NAMES[dummy.predict(X_test.iloc[:1])[0] - 1]}'"
      f"  ->  {baseline:.1%}")
print("species it can ever predict: 1 of 7")
'''),

        md("""
## 7 · Commit

**Stop. On paper, now.** Not in this notebook — on paper, where you cannot
quietly revise it.

```
Metric:                                        ____________
Accuracy a good system would need:             ____________
Accuracy I expect from the model I build today: ___________
```

You are not guessing in the dark: a constant scores 48.8%, and the model has to
justify every prediction in at most eight conditions.
"""),

        md("""
## 8 · One tree, with no constraints at all

A decision tree asks a sequence of questions of the form `x[k] <= t`, and the
leaf it reaches predicts the majority class of the training patches that reached
the same leaf. Fit one and look at its shape before looking at its score.
"""),
        prompt(
            label="one tree, unconstrained",
            input="the 48,000 training patches",
            output="its depth, its leaf count, and its training accuracy",
            constraint="look at its SHAPE before its score",
            check="assert it grew past a thousand leaves — an unconstrained tree on this data should be enormous, and a small one means something was capped by accident",
            left_open="that 100% training accuracy is not the subject of this lecture. It is measured honestly and moved past; the lecture is about justification, not overfitting.",
            student="reporting the training accuracy. With 5,699 leaves for 48,000 patches the tree can put a handful in each leaf and look them up, which is memorisation with a nice diagram.",
            catch="leaves against rows. Eight patches per leaf is a lookup table; the number is the diagnosis and it takes one line."),
        code('''
from sklearn.tree import DecisionTreeClassifier

free = DecisionTreeClassifier(random_state=RANDOM_STATE).fit(X_train, y_train)

print(f"depth           {free.get_depth()}")
print(f"leaves          {free.get_n_leaves():,}")
print(f"train accuracy  {free.score(X_train, y_train):.1%}")

assert free.get_n_leaves() > 1000, "expected a large, unconstrained tree"
'''),
        md("""
100% on the data it was fitted to. You met that number in the second lecture and
you know what it means: with 5,699 leaves for 48,000 patches, the tree can put a
handful of patches in each leaf and look them up.

Measure it honestly, then move on — overfitting is *not* what this lecture is
about.

⏱ **about 30 seconds** — five fits on 38,400 rows each.
"""),
        prompt(
            label="⏱ 30 s — measure it honestly",
            input="the unconstrained tree and the training rows",
            output="cross-validated accuracy with the fold range",
            constraint="stratified folds, and report the RANGE as well as the mean",
            left_open="that this number exists only to be moved past. It is the unconstrained model's honest score and the brief forbids the model that produced it.",
            student="stopping here, because the number is good. The brief is not about accuracy, and a model that cannot justify a prediction fails it at any score.",
            catch="fold minimum and maximum beside the mean. A mean of 82.6% built from folds spanning four points is a different object from one built from folds spanning half a point."),
        code('''
from sklearn.model_selection import StratifiedKFold, cross_val_score

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
free_cv = cross_val_score(free, X_train, y_train, cv=cv, n_jobs=-1)

print(f"cross-validated {free_cv.mean():.1%}  (folds {free_cv.min():.1%} - "
      f"{free_cv.max():.1%})")
'''),

        md("""
## 9 · An assistant writes the interpretable model

Here is a real request and the code it returns. **⚠ Read before running.** It
runs, it imports nothing exotic, and it prints a ranked list of features under a
heading that says *why the model predicts what it predicts*.

> *"Train a decision tree on the covertype data and make it interpretable, so I
> can explain each prediction to a regulator."*
"""),
        prompt(
            label="⚠ what the assistant returns",
            input="'train a decision tree and make it interpretable, so I can explain each prediction to a regulator'",
            output="the five largest feature importances, under a heading saying why the model predicts what it predicts",
            constraint="print it exactly as returned — it runs, it imports nothing exotic, and it answers a different question than the one asked",
            left_open="the SHAPE of the answer. `feature_importances_` has 54 entries, one per column, not one per prediction. It is the same vector for every patch in Colorado.",
            student="shipping this. The regulator asked why THIS parcel was refused; the answer on offer is 'elevation matters a lot, in general', which describes the training run rather than the decision.",
            catch="count the entries in any explanation you are offered. If there are as many as there are FEATURES rather than as many as there are PREDICTIONS, it is a global summary wearing a local word."),
        code('''
importances = pd.Series(free.feature_importances_, index=X_train.columns)

print("Why the model predicts what it predicts:")
print(importances.sort_values(ascending=False).head(5).round(3))
print(f"\\naccuracy: {free.score(X_test, y_test):.1%}")
'''),

        md("""
### Reviewer question 3: what is the shape here?

`feature_importances_` has **54 entries** — one per column, not one per
prediction. It is the same vector for every patch in Colorado.

The regulator asked why *this* parcel was refused. The answer on offer is
"elevation matters a lot, in general". That is a description of the training
run, not a justification.

**Now measure the damage.** The tree *can* justify a prediction — the path from
root to leaf is a list of conditions. Count them.
"""),
        prompt(
            label="measure the damage — count the conditions",
            input="a fitted tree and the test patches",
            output="the mean and maximum number of conditions applied per prediction",
            constraint="count nodes VISITED minus one — the leaf is not a condition, and an off-by-one here silently reports depth+1",
            check="assert the maximum path length equals the tree's own reported depth",
            left_open="that the tree CAN justify a prediction, and always could. The path from root to leaf is a list of conditions; the assistant reached for a global summary instead.",
            student="trusting `get_depth()` as the answer. Depth is the longest path, not the typical one, and the brief constrains every prediction rather than the worst.",
            catch="the assert tying path length back to `get_depth()`. Two independent routes to the same number is how you find out `decision_path` counts the leaf."),
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
### The corrected specification

> *"Fit a `DecisionTreeClassifier` with `max_depth=8` on `X_train`. For a given
> test instance, return the list of `(feature, comparison, threshold, value)`
> tuples along its decision path, and the class distribution of the leaf it lands
> in. Assert that the list has at most eight entries. Do not use
> `feature_importances_`: it is one vector for the whole model."*

Three additions: the **shape** of the output, the **check**, and an explicit
prohibition on the plausible wrong answer. The assistant was obedient, not wrong
— "interpretable" has a common meaning in the literature and it used it.
"""),

        md("""
## 10 · What does depth actually buy?

The constraint fixes `max_depth`. Before accepting that, measure what it costs,
because the number belongs in the report to the agency.

⏱ **about 90 seconds** — twelve depths, five folds each.
"""),
        prompt(
            label="⏱ 90 s — what does depth buy",
            input="depths 1 to 12",
            output="cross-validated accuracy and leaf count at each depth",
            constraint="sweep PAST the depth we are allowed to use — the rows we cannot pick are what tell the agency what its constraint costs",
            left_open="that cross-validation does not get a vote on max_depth here. It is not optimising the thing the agency is buying, and the sweep exists to price the constraint rather than to choose it.",
            student="running the sweep, finding depth 12 best, and using it. The constraint is not negotiable and the sweep was never a search.",
            catch="when a hyperparameter is fixed by the brief, still measure the alternatives — and report the difference as a price, not as a missed opportunity."),
        code('''
rows = []
for d in range(1, 13):
    clf = DecisionTreeClassifier(max_depth=d, random_state=RANDOM_STATE)
    acc = cross_val_score(clf, X_train, y_train, cv=cv, n_jobs=-1).mean()
    leaves = clf.fit(X_train, y_train).get_n_leaves()
    rows.append({"max_depth": d, "cv_accuracy": acc, "leaves": leaves})

depth_table = pd.DataFrame(rows).set_index("max_depth")
print(depth_table.to_string(float_format=lambda v: f"{v:.4f}"))
'''),
        prompt(
            label="the price, drawn",
            input="the depth table",
            output="accuracy against depth, and leaves against depth on a log axis",
            constraint="log scale on the leaf count — it spans three orders of magnitude, and on a linear axis every depth below 10 is flat on the floor",
            left_open="that both curves are still climbing at depth 12. That is the measured price of the constraint, and it is the kind of thing to bring to the regulator as a conversation.",
            student="a linear y-axis on the right panel, which shows one point rising and eleven at zero, and concludes leaf count 'explodes at depth 12' when it has been doubling all along.",
            catch="mark the constrained value on both panels. A sweep with no line at the value you actually chose makes the reader do the lookup."),
        code('''
import matplotlib.pyplot as plt

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
Cross-validated accuracy is still climbing at depth 12 and the number of rules is
climbing with it. That difference is the measured price of the constraint. Bring
it to the regulator — perhaps they will trade a condition or two for it. That is
a conversation, not a `GridSearchCV`.

**Cross-validation does not get a vote on `max_depth` here**, because it is not
optimising the thing the agency is buying.
"""),

        md("""
## 11 · Tune what is left, inside the constraint

`min_samples_leaf` is still ours. Search the whole grid anyway — including the
depths we may not use — because the rows we cannot pick are what tell the agency
what its constraint costs.

⏱ **about 2 minutes** — 24 combinations, five folds each.
"""),
        prompt(
            label="⏱ 2 min — tune what is left",
            input="a 2-D grid of max_depth and min_samples_leaf",
            output="the full grid as a pivot table, not just the winner",
            constraint="search the WHOLE grid including depths we may not use, and print the table rather than `best_params_` alone",
            left_open="that the two hyperparameters interact. Along max_depth=8 the leaf size barely matters because the depth limit binds first; along max_depth=None it matters enormously, because leaf size is then the only regularisation.",
            student="two separate 1-D sweeps, one per hyperparameter. They would find the same best value and show none of the interaction, and the interaction is the finding.",
            catch="print the grid as a table whenever two hyperparameters both restrict the same thing. `best_params_` is one cell of it and the shape of the rest is the result."),
        code('''
from sklearn.model_selection import GridSearchCV

grid = {"max_depth": [4, 6, 8, None],
        "min_samples_leaf": [1, 5, 20, 50, 200, 500]}

search = GridSearchCV(DecisionTreeClassifier(random_state=RANDOM_STATE),
                      grid, cv=cv, n_jobs=-1).fit(X_train, y_train)

res = pd.DataFrame(search.cv_results_)
pivot = res.pivot_table(index="param_max_depth", columns="param_min_samples_leaf",
                        values="mean_test_score", dropna=False)
print(pivot.to_string(float_format=lambda v: f"{v:.4f}"))
print(f"\\nbest overall: {search.best_params_}")
'''),
        md("""
Read two things off that table.

**Along the `max_depth=8` row, `min_samples_leaf` barely matters** — the depth
limit binds first, so the leaf-size limit has almost nothing left to do.

**Along the `max_depth=None` row it matters enormously** — with no depth limit,
the leaf size *is* the regularisation.

Two hyperparameters that both restrict the tree do not act independently. A 2-D
grid shows that; two separate 1-D sweeps would not have.
"""),

        md("""
## 12 · The model we are going to ship — overruling the grid

The grid's answer under the cap is `min_samples_leaf=1`. **We are not going to
ship it.**

The model states its justification as *"90% of the 481 training patches in this
leaf"*. With a minimum leaf of 1 that sentence can become *"100% of the 1"* — a
single surveyed patch wearing the grammar of evidence. The brief asks for a
justification a regulator can audit, and that is not one.

So we overrule the grid, for exactly the reason we overruled it on depth: **when
the brief constrains the model, the grid does not get a vote.** It costs 0.40
points of cross-validated accuracy, and that number goes to the agency with
everything else.
"""),
        prompt(
            label="overruling the grid, deliberately",
            input="max_depth 8 and a minimum leaf of 20",
            output="leaves, columns consulted, conditions per prediction, and training accuracy",
            constraint="min_samples_leaf=20 comes from the BRIEF, not from the grid — the grid's answer under the cap is 1",
            check="assert no prediction uses more than 8 conditions, which is the brief expressed as an assert",
            left_open="why 20 and not 1. The model's justification reads '90% of the 481 training patches in this leaf'. With a minimum leaf of 1 that becomes '100% of the 1' — a single surveyed patch wearing the grammar of evidence.",
            student="taking `best_params_` because it is the best. It costs 0.40 points of cross-validated accuracy to overrule it, and that number goes to the agency with everything else rather than being hidden.",
            catch="when the brief constrains the model, the grid does not get a vote — and the assert that encodes the brief belongs in the cell that ships the model."),
        code('''
AUDITABLE_LEAF = 20        # the brief, not the grid — see the note above

tree = DecisionTreeClassifier(max_depth=8, min_samples_leaf=AUDITABLE_LEAF,
                              random_state=RANDOM_STATE).fit(X_train, y_train)

used = {int(f) for f in tree.tree_.feature if f >= 0}
lens = path_lengths(tree, X_test)

print(f"leaves                  {tree.get_n_leaves()}")
print(f"columns consulted       {len(used)} of {X_train.shape[1]}")
print(f"conditions, mean / max  {lens.mean():.2f} / {lens.max()}")
print(f"train accuracy          {tree.score(X_train, y_train):.1%}")

assert lens.max() <= 8, "the brief is violated"
'''),
        md("""
The gap between training and cross-validated accuracy has almost vanished. The
depth limit did not only shorten the justification; it removed nearly all of the
overfitting as a side effect.

Which raises a question we are **not** answering today: if the constrained tree
barely overfits, why is it still several points worse than the unconstrained
one? Hold that thought until the next lecture.
"""),

        md("""
## 13 · Read the tree

Two routes. `export_graphviz` writes a `.dot` file, which then needs the `dot`
binary — not a Python package, and not installed on a stock Colab runtime. The
call succeeds, writes a file, and nothing renders.

*(Not examinable: this is tooling, not machine learning.)*

`plot_tree` draws into a matplotlib axis and works anywhere matplotlib does.
`max_depth=2` is essential — all eight levels at a readable size is about two
metres of paper.
"""),
        prompt(
            label="draw it",
            input="the shipped tree",
            output="the top two levels, drawn into a matplotlib axis",
            constraint="`max_depth=2` in the PLOT call — all eight levels at a readable font is about two metres of paper",
            left_open="why not graphviz. `export_graphviz` writes a .dot file that needs the `dot` binary, which is not a Python package and is not on a stock Colab runtime: the call succeeds, writes a file, and nothing renders.",
            student="following the first tutorial hit to `export_graphviz`, getting no error and no picture, and losing twenty minutes to a missing system package.",
            catch="shorten the feature names before plotting. `Horizontal_Distance_To_Fire_Points` renders as a smear at any font size that fits eight levels on a page."),
        code('''
from sklearn.tree import export_text, plot_tree

short = [c.replace("Horizontal_Distance_To_", "HDist_")
          .replace("Vertical_Distance_To_", "VDist_")
          .replace("Hillshade_", "Shade_")
          .replace("Wilderness_Area_", "Wild_")
          .replace("Soil_Type_", "Soil_") for c in X_train.columns]

fig, ax = plt.subplots(figsize=(13, 4.5))
plot_tree(tree, max_depth=2, feature_names=short, class_names=COVER_NAMES,
          filled=True, rounded=True, impurity=False, proportion=True,
          precision=1, fontsize=8, ax=ax)
plt.show()
'''),
        prompt(
            label="the version you can paste into an email",
            input="the same tree",
            output="the top two levels as indented text",
            constraint="no plotting library at all — for a model whose selling point is that a person can read it, that matters more than it sounds",
            left_open="why the thresholds sit at half-integers. CART puts a split midway between two adjacent observed values, so nothing in the data ever sits exactly on a threshold.",
            student="assuming a threshold of 2959.5 means something about the terrain. It means two training patches were at 2959 and 2960.",
            catch="if a model is sold as interpretable, check that its explanation survives being pasted into an email. A PNG does not."),
        code('''
print(export_text(tree, feature_names=short, class_names=COVER_NAMES,
                  max_depth=2, decimals=0))
'''),
        md("""
`export_text` needs no plotting library at all, and it is what you paste into an
email. For a model whose selling point is that a person can read it, that
matters more than it sounds.

The thresholds sit at half-integers — CART puts a split **midway between two
adjacent observed values**, so nothing in the data sits exactly on one.
"""),

        md("""
## 14 · Trace one prediction, all the way down

The entire justification mechanism is three arrays: `tree_.children_left`,
`tree_.feature` and `tree_.threshold`.
"""),
        prompt(
            label="trace one prediction all the way down",
            input="one test patch and the fitted tree",
            output="the conditions it satisfied, and the class distribution of its leaf",
            constraint="walk `children_left` / `feature` / `threshold` by hand — the whole justification mechanism is those three arrays",
            check="assert at most 8 conditions AND that the traced class agrees with `predict()`",
            left_open="how to read the leaf. `predict_proba` returns exactly these leaf proportions, so a tree's probabilities are piecewise constant and identical for every patch reaching the same leaf.",
            student="reporting '90% likely to be Krummholz'. The honest sentence is 'of the training patches that satisfied these eight conditions, 90% were Krummholz', and the difference is the whole regulatory argument.",
            catch="the assert that the hand-walk agrees with `predict()`. A justification that disagrees with the model it claims to explain is worse than no justification."),
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
x = X_test.iloc[i].values
conditions, leaf = justify(tree, x, short)

for n, (name, op, thr, val) in enumerate(conditions, 1):
    print(f"{n}. {name:12s} = {val:8,.0f}  {op:2s} {thr:8,.0f}")

dist = tree.tree_.value[leaf][0]
print(f"-> {COVER_NAMES[int(dist.argmax())]}  ({dist.max():.0%} of the "
      f"{int(tree.tree_.n_node_samples[leaf])} training patches in this leaf)")
print(f"   true class: {COVER_NAMES[int(y_test.iloc[i]) - 1]}")

assert len(conditions) <= 8
assert COVER_NAMES[int(dist.argmax())] == COVER_NAMES[tree.predict(
    X_test.iloc[[i]])[0] - 1], "the trace disagrees with predict()"
'''),
        md("""
### Read the leaf carefully

That is a statement about the **training patches in the leaf**, not a probability
that this patch is that species. `predict_proba` returns exactly those leaf
proportions, so a tree's "probabilities" are piecewise constant and identical
for every patch reaching the same leaf.

The honest sentence: *"Of the training patches that satisfied these eight
conditions, 90% were Krummholz."* Not *"this patch is 90% likely to be
Krummholz."*

A leaf built on four patches and a leaf built on four thousand produce the same
kind of sentence and deserve very different amounts of trust — which is why the
count belongs in the justification.
"""),
        prompt(
            label="how much is each leaf built on",
            input="the training patches routed through the tree",
            output="the smallest, median and largest leaf",
            constraint="drop the zero counts — `bincount` returns a slot for every node id, and the internal nodes are all zeros",
            left_open="what to do with the number. A leaf built on four patches and one built on four thousand produce the same kind of sentence and deserve very different amounts of trust.",
            student="quoting leaf proportions without the count. That is why the count belongs in the justification itself, not in a footnote.",
            catch="the minimum leaf size should equal what you set. If it is smaller, `min_samples_leaf` is not doing what you think it is."),
        code('''
sizes = np.bincount(tree.apply(X_train))
sizes = sizes[sizes > 0]
print(f"leaves {len(sizes)}   smallest {sizes.min()}   "
      f"median {int(np.median(sizes))}   largest {sizes.max():,}")
'''),

        md("""
## 15 · The test set. Once.

Everything above used only training data and cross-validated folds.
"""),
        prompt(
            label="the test set, once",
            input="the 12,000 held-out patches",
            output="accuracy against the baseline, and per-class precision and recall",
            constraint="per-class numbers, not just the headline — the headline is an average over seven very differently sized classes",
            left_open="`zero_division=0`. It is there because a class the model never predicts has an undefined precision, and the default would print a warning instead of a number.",
            student="reading only the accuracy. 73.3% against a baseline of 48.8% is a real result and it says nothing about the class that the model never once predicts correctly.",
            catch="everything above this cell used training data and cross-validated folds. If that is not true of your notebook, this number is not a test score."),
        code('''
from sklearn.metrics import ConfusionMatrixDisplay, classification_report

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
            constraint="`normalize='true'` — each row then reads as 'of the patches that really were this species, where did they go?'",
            left_open="the Aspen row, which is almost entirely in the Lodgepole Pine column. On the headline number that costs about a point and a half and is invisible.",
            student="leaving it unnormalised, where the majority class dominates every cell and the rare-class failures are literally too small to see.",
            catch="why it happens: Aspen is 1.6% of the training set, so a split that isolates it improves the weighted Gini by very little and CART never chooses one. The impurity criterion is a weighted average and a rare class carries almost no weight."),
        code('''
fig, ax = plt.subplots(figsize=(7, 6))
ConfusionMatrixDisplay.from_estimator(
    tree, X_test, y_test, display_labels=COVER_NAMES, normalize="true",
    cmap="Blues", values_format=".2f", xticks_rotation=45, ax=ax, colorbar=False)
plt.tight_layout(); plt.show()
'''),
        md("""
`normalize="true"` divides by the row total, so each row reads as *"of the
patches that really were this species, where did they go?"*

**Read the Aspen row.** Almost all of it is in the Lodgepole Pine column. The
model finds a handful of Aspen patches in every hundred, and on the headline
number that costs about a point and a half and is invisible.

Why: Aspen is 1.6% of the training set, so a split that isolates it improves the
weighted Gini by very little and CART never chooses one. The impurity criterion
is a weighted average, and a rare class carries almost no weight.

Try `class_weight="balanced"` below and see what it does to both numbers.
"""),
        prompt(
            label="what class weighting buys, and costs",
            input="the same tree with class_weight='balanced'",
            output="overall accuracy and Aspen recall, each beside its unweighted value",
            constraint="report BOTH numbers — one goes up and one goes down, and quoting either alone is an argument rather than a measurement",
            left_open="which model the agency wants. That is not a machine learning decision, and the notebook deliberately does not make it.",
            student="turning on class_weight because the confusion matrix looked bad, and reporting the improved recall without the lost accuracy. It is a different model answering a different question.",
            catch="any change that improves a rare class will cost the common ones. Show the pair, and let whoever owns the decision make it."),
        code('''
balanced = DecisionTreeClassifier(max_depth=8, class_weight="balanced",
                                  random_state=RANDOM_STATE).fit(X_train, y_train)
rep = classification_report(y_test, balanced.predict(X_test),
                            target_names=COVER_NAMES, output_dict=True,
                            zero_division=0)

print(f"accuracy      {balanced.score(X_test, y_test):.1%}  "
      f"(was {acc:.1%})")
print(f"Aspen recall  {rep['Aspen']['recall']:.1%}")
print("\\nA different model, answering a different question. Which one the "
      "agency wants is not a machine learning decision.")
'''),

        md("""
## 16 · Where we are

Write your **best accuracy** on the same sheet of paper, next to what you
predicted. Bring it to the next lecture — we open by comparing them.

| Model | Test accuracy | Conditions per justification |
|---|---|---|
| always "Lodgepole Pine" | 48.8% | 0 |
| depth-8 tree — **ours** | 73.3% | 7.99 |
| unconstrained tree | 82.6% | 17.88 |

Row two meets the brief. Every one of its predictions comes with a reason a
surveyor could check on site.

**Do not fix anything.** One question to take away, and do not look it up:

> Your neighbour has fitted the same model, with the same hyperparameters, to
> *almost* the same training set. How similar are the two sets of rules?
"""),
    ]
    return cells
