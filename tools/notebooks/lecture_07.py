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

from make_notebooks import code, header, md, SETUP        # noqa: E402


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
        md("## 1 · Setup"), SETUP,

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

        md("## 3 · The data"), COVER_LOADER,

        md("""
Ten quantitative columns — elevation, aspect, slope, hillshade at three times of
day, and four distances — plus four wilderness-area indicators and forty soil
type indicators, both already one-hot. Nothing to impute, nothing to encode.
"""),
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
        SUBSAMPLE,

        md("""
## 5 · Look at the labels

Take thirty seconds over these counts before scrolling. One number here decides
what the baseline is.
"""),
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
        code('''
from sklearn.metrics import ConfusionMatrixDisplay, classification_report

acc = tree.score(X_test, y_test)
print(f"test accuracy {acc:.1%}   (baseline {baseline:.1%})")
print()
print(classification_report(y_test, tree.predict(X_test),
                            target_names=COVER_NAMES, digits=3, zero_division=0))
'''),
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
