#!/usr/bin/env python3
"""
Lecture 4 — It never fires. Fix, MNIST, Géron Chapter 3.

Exports build() -> list[cell]; tools/make_notebooks.py wraps it.

Structure mirrors the deck: thread -> diagnosis -> repair -> re-measure ->
red team. It rebuilds the previous lecture's detector from scratch, imports
included: a notebook that only runs because another notebook is still in memory
is not reproducible.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from make_notebooks import code, header, md, SETUP        # noqa: E402


def build() -> list:
    cells = header(
        4, "It never fires", "fix", "Chapter 3",
        thread="imbalance, and the non-monotonicity of precision")

    cells += [
        md("## 1 · Setup and where we left off"), SETUP,
        code('''
# Every import this notebook needs, in one place. A notebook that only runs
# because a previous one is still in memory is not reproducible.
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (accuracy_score, average_precision_score,
                             confusion_matrix, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score,
                             roc_curve)
from sklearn.model_selection import (StratifiedKFold, cross_val_predict,
                                     cross_val_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
'''),
        code('''
# ~30 s the first time; scikit-learn has it cached from the previous notebook.
mnist = fetch_openml("mnist_784", as_frame=False)
X, y = mnist.data, mnist.target.astype(np.uint8)

X_train, X_test = X[:60000], X[60000:]
y_train, y_test = y[:60000], y[60000:]
y_train_5 = (y_train == 5)
y_test_5  = (y_test  == 5)

assert len(X_train) == 60000 and len(X_test) == 10000
assert y_train_5.sum() == 5421 and y_test_5.sum() == 892

base_rate = y_train_5.mean()
print(f"base rate {base_rate:.5f}   never-fires accuracy {1 - base_rate:.5f}")

clf = make_pipeline(StandardScaler(), SGDClassifier(random_state=RANDOM_STATE))
cv  = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
'''),

        md("""
## 2 · Thread 2 — where 96.90% came from

Write $P$ for the number of positives, $N$ for the negatives, $m = P + N$, and
$p = P/m$ for the **base rate**. Split the four confusion counts by their true
class:

$$\\text{accuracy} = \\frac{\\mathrm{TP} + \\mathrm{TN}}{m}
 = \\frac{P}{m}\\cdot\\frac{\\mathrm{TP}}{P} + \\frac{N}{m}\\cdot\\frac{\\mathrm{TN}}{N}
 = p\\,\\text{recall} + (1-p)\\,\\text{specificity}$$

Accuracy is a **weighted average of two rates**, weighted by the class sizes.
Two consequences follow immediately, and both are identities rather than
estimates:

- A classifier that never fires has recall $0$ and specificity $1$, so it scores
  exactly $1 - p$.
- $\\partial\\,\\text{accuracy} / \\partial\\,\\text{recall} = p$. At our base
  rate, ten points of recall move accuracy by 0.90 points.

Verify the first, rather than believing it:
"""),
        code('''
for p in (0.5, 0.09035, 0.01, 0.001):
    print(f"base rate {p:>7.5f}  ->  never-fires accuracy {1 - p:.5f}")

print("\\nFor any target accuracy below 1, there is a problem rare enough")
print("that the empty model beats it.")
'''),

        md("""
### Part two — what happens when you turn the dial

Every scoring classifier is a *family* of classifiers, one per threshold $t$:
predict positive when $s(x) \\ge t$. Raising $t$ shrinks the flagged set, and it
shrinks it by **losing** instances, never gaining any. So $\\mathrm{TP}(t)$ and
$\\mathrm{FP}(t)$ are both non-increasing.

**Recall is monotone.** Its denominator is $\\mathrm{TP}(t) + \\mathrm{FN}(t) =
P$, which counts every positive in the data whatever we predict. So recall is a
non-increasing quantity divided by a constant.

**Precision is not.** Its denominator is the flagged count, which we choose.

*The one-step lemma.* Raise $t$ just enough to drop the lowest-scoring flagged
instance. With $T = \\mathrm{TP}$, $n = T + \\mathrm{FP}$ before the step and
$y \\in \\{0,1\\}$ the label of the instance dropped:

$$\\frac{T-y}{n-1} > \\frac{T}{n} \\iff n(T-y) > T(n-1) \\iff y < \\frac{T}{n}$$

Since $y$ is 0 or 1 and precision lies in $(0,1)$: dropping a **negative** raises
precision, dropping a **positive** lowers it.

Now count it, on sixty thousand instances. First we need the scores.

⏱ **about 60 seconds** — three refits, asking for the decision function rather
than the labels.
"""),
        code('''
y_scores = cross_val_predict(clf, X_train, y_train_5, cv=cv,
                             method="decision_function", n_jobs=-1)

assert y_scores.shape == (60000,)
# SGDClassifier predicts positive exactly when the decision function is >= 0,
# so one call gives us both the labels and the whole threshold sweep.
y_pred = (y_scores >= 0)
print(f"accuracy {accuracy_score(y_train_5, y_pred):.5f}")
'''),
        code('''
# Walk the threshold down the ranking, one instance at a time.
order = np.argsort(-y_scores, kind="stable")
lab   = y_train_5[order].astype(np.int64)
tp    = np.cumsum(lab)
n     = np.arange(1, len(lab) + 1)
prec  = tp / n

# step k -> k+1 lowers the threshold; read backwards, d > 0 is a FALL
d = np.diff(prec)
drops_a_five = (lab[1:] == 1)

print(f"steps in total                     {len(d):,}")
print(f"drop a non-5, precision rises      {(d < 0).sum():,}")
print(f"drop a 5,     precision falls      {(d > 0).sum():,}")
print(f"drop a 5,     precision already 1  {((d == 0) & drops_a_five).sum():,}")

assert (d[~drops_a_five] < 0).all(), "the lemma says every one of these rises"
assert (d[drops_a_five] >= 0).all(), "and every one of these falls or is flat"
print("\\nEvery one of the steps is accounted for by the lemma.")
'''),
        md("""
54,579 is the number of non-5s in the training set, and 5,417 + 3 = 5,420 is the
number of 5s less the one at the very top of the ranking, which has no step above
it. This is not a tendency: it is an exact classification of all 59,999 steps.

The textbook's own counterexample is the same thing at the top of the list.
"""),
        code('''
for k in range(1, 9):
    print(f"top-{k}: {tp[k-1]}/{k} = {prec[k-1]:.4f}")

print("\\nRaising the threshold past the 6th-ranked digit — a 5 — takes")
print(f"precision from {prec[5]:.4f} (5/6) down to {prec[4]:.4f} (4/5).")
assert prec[4] < prec[5]
'''),

        md("""
## 3 · Diagnose — what the number was hiding

Nothing new gets fitted here. We ask the *same* out-of-fold predictions a
different question.
"""),
        code('''
cm = confusion_matrix(y_train_5, y_pred)
tn, fp, fn, tp_ = cm.ravel()
print(cm)
print()
print(f"true negatives  {tn:>7,}     false positives {fp:>7,}")
print(f"false negatives {fn:>7,}     true positives  {tp_:>7,}")

assert tn + fp + fn + tp_ == 60000
'''),
        code('''
precision = precision_score(y_train_5, y_pred)
recall    = recall_score(y_train_5, y_pred)
spec      = tn / (tn + fp)

print(f"accuracy    {accuracy_score(y_train_5, y_pred):.5f}")
print(f"precision   {precision:.5f}   {tp_:,} / {tp_ + fp:,}")
print(f"recall      {recall:.5f}   {tp_:,} / {tp_ + fn:,}")
print(f"F1          {f1_score(y_train_5, y_pred):.5f}")
print(f"specificity {spec:.5f}")

# the identity, on our own numbers
lhs = accuracy_score(y_train_5, y_pred)
rhs = base_rate * recall + (1 - base_rate) * spec
print(f"\\np.recall + (1-p).specificity = {rhs:.5f}   vs accuracy {lhs:.5f}")
assert np.isclose(lhs, rhs)

print(f"\\nthe specificity term alone is {100 * (1 - base_rate) * spec / lhs:.1f}%"
      f" of the headline number")
print(f"and {fn:,} fives — {100 * fn / (tp_ + fn):.1f}% of them — went past unflagged")
'''),
        md("""
Two sentences about the same model, both true:

- *"The detector is 96.9% accurate."* — signed off, deployed.
- *"The detector misses 22.8% of the 5s."* — a report the audit team can act on.

Accuracy was not *wrong*. It correctly answered a question nobody in this brief
had asked, it weights the class we care about by 0.09, and it adds two errors
that the client prices completely differently.

**And nobody chose it.** It arrived as a default, from a prompt that did not name
a metric.
"""),

        md("""
## 4 · Repair — the threshold is a dial

`predict()` hides the score and hard-codes the threshold at zero. There is no
`set_threshold()` method, and there should not be: you compute the scores and
compare them yourself.
"""),
        code('''
some_digit = X_train[0]                       # a 5
clf.fit(X_train, y_train_5)                   # ⏱ about 30 s
score = clf.decision_function([some_digit])
print("score:", score.round(1))

for t in (0, 3000):
    print(f"threshold {t:>5}: predicted {bool(score[0] >= t)}")
'''),
        code('''
precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)

# precisions and recalls have ONE MORE element than thresholds: the degenerate
# point where nothing is flagged and precision is defined to be 1.
assert len(precisions) == len(recalls) == len(thresholds) + 1

fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 4))
ax.plot(thresholds, precisions[:-1], label="precision")
ax.plot(thresholds, recalls[:-1], label="recall")
ax.set_xlim(-400, 400); ax.set_xlabel("threshold"); ax.legend(); ax.grid(alpha=.3)
ax2.plot(recalls, precisions)
ax2.set_xlabel("recall"); ax2.set_ylabel("precision"); ax2.grid(alpha=.3)
plt.tight_layout(); plt.show()
'''),
        md("""
### Choose a threshold deliberately

`argmax` on a boolean array returns the first `True`. The idiom is compact and
not obvious; read it once and remember it.
"""),
        code('''
# Thread 2 spent twenty minutes proving this curve is NOT monotone. So "the
# first index reaching 90%" can in principle be one lucky step held up by a
# handful of flagged digits — a real operating point needs support behind it.
MIN_SUPPORT = 500

n_pos   = int(y_train_5.sum())
flagged = recalls[:-1] * n_pos / precisions[:-1]     # tp / precision = tp + fp
ok      = (precisions[:-1] >= 0.90) & (flagged >= MIN_SUPPORT)
assert ok.any(), "no threshold reaches 90% precision with enough support"

idx = int(np.argmax(ok))
threshold_90 = thresholds[idx]

# Did the guard actually change anything? Say so either way.
naive = int((precisions[:-1] >= 0.90).argmax())
print(f"first crossing        idx {naive}")
print(f"first with support    idx {idx}   "
      f"({'same point — the check held' if naive == idx else 'DIFFERENT point'})")
print(f"digits flagged there  {flagged[idx]:,.0f}")

y_pred_90 = (y_scores >= threshold_90)
print(f"threshold {threshold_90:.2f}")
print(f"precision {precision_score(y_train_5, y_pred_90):.4f}")
print(f"recall    {recall_score(y_train_5, y_pred_90):.4f}")
print(f"\\n90% precision costs "
      f"{100 * (recall - recall_score(y_train_5, y_pred_90)):.2f} points of recall")

# and the other end of the dial, to show what "high precision" really buys.
# Apply the same support test — and watch it fail.
idx99 = int((precisions[:-1] >= 0.99).argmax())
print(f"\\nat 99% precision, recall is {recalls[idx99]:.4f} — one 5 in fifty")
print(f"   but only {flagged[idx99]:,.0f} digits are flagged there, on a "
      f"plateau {int((precisions[:-1] >= 0.99).sum())} thresholds wide")
print(f"   (the 90% point rests on {flagged[idx]:,.0f} digits over "
      f"{int((precisions[:-1] >= 0.90).sum())})")
print("   -> quote 2.12% as an illustration, not as an operating point")
'''),
        md("""
**A threshold is a hyperparameter.** Choosing it by looking at test performance
is the same error as choosing $\\alpha$ that way, and produces the same
optimistic bias. Choose it on cross-validated *training* scores, then measure
once.
"""),

        md("""
### PR or ROC?

The ROC curve plots recall against the false positive rate. Both denominators
are fixed by the data, which is why it behaves so much better than the PR curve
— and that is a warning, not a recommendation.

The rule: **prefer the PR curve when the positive class is rare.** Do not take
it on trust. Take our scores unchanged and thin the positive class, so that the
model, the ranking and the scores are identical and only the balance moves.
"""),
        code('''
rng = np.random.default_rng(RANDOM_STATE)
pos, neg = np.where(y_train_5)[0], np.where(~y_train_5)[0]

print(f"{'base rate':>10}  {'positives':>9}  {'ROC AUC':>8}  {'avg prec':>8}")
row = {}
for target in (base_rate, 0.02, 0.01):
    k = int(round(len(neg) * target / (1 - target)))
    keep = np.concatenate([neg, rng.choice(pos, k, replace=False)])
    auc = roc_auc_score(y_train_5[keep], y_scores[keep])
    ap  = average_precision_score(y_train_5[keep], y_scores[keep])
    row[round(target, 4)] = (auc, ap)
    print(f"{y_train_5[keep].mean():>10.4f}  {k:>9,}  {auc:>8.4f}  {ap:>8.4f}")

(auc_hi, ap_hi), (auc_lo, ap_lo) = row[0.0904], row[0.01]
print(f"\\nROC AUC moves by {abs(auc_hi - auc_lo):.4f}")
print(f"average precision falls by {ap_hi - ap_lo:.4f}")
assert abs(auc_hi - auc_lo) < 0.01 < (ap_hi - ap_lo)
'''),
        md("""
The ROC curve cannot see the problem you have. That is the whole reason to learn
two curves rather than one.
"""),

        md("""
## 5 · A better classifier

A forest has no `decision_function`. It has `predict_proba`, which returns one
column per class; the second column — the estimated probability of the positive
class — plays exactly the role of the score, and every curve above works
unchanged.

⏱ **under a minute.**
"""),
        code('''
forest = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE,
                                n_jobs=-1)

# the forest is already n_jobs=-1; nesting another pool oversubscribes
y_proba = cross_val_predict(forest, X_train, y_train_5, cv=cv,
                            method="predict_proba")
assert y_proba.shape == (60000, 2)
f_scores = y_proba[:, 1]                       # the POSITIVE class column

f_prec, f_rec, f_thr = precision_recall_curve(y_train_5, f_scores)

print(f"{'':28s}{'SGD':>10s}{'forest':>10s}")
print(f"{'ROC AUC':28s}{roc_auc_score(y_train_5, y_scores):>10.4f}"
      f"{roc_auc_score(y_train_5, f_scores):>10.4f}")
print(f"{'average precision':28s}{average_precision_score(y_train_5, y_scores):>10.4f}"
      f"{average_precision_score(y_train_5, f_scores):>10.4f}")
print(f"{'recall at 90% precision':28s}"
      f"{recalls[(precisions >= 0.90).argmax()]:>10.4f}"
      f"{f_rec[(f_prec >= 0.90).argmax()]:>10.4f}")
'''),
        md("""
Accuracy would call that a 1.87-point improvement, which reads as polish. The
last row moves by 24 points, and it is the row the brief is about.

**Which description you report decides whether anyone approves the change.**
"""),

        md("""
## 6 · The operating point, and the test shift

What the client asked for, in our vocabulary: **catch at least 90% of the 5s**
on a shift, and **flag no more than 1,000 items** out of 10,000 scanned.

Two constraints, one dial. A threshold either satisfies both or it does not, and
the honest answer may be that none does.

Choose on the cross-validated training scores. Then touch the test shift
**once**. ⏱ **about 30 seconds** for the two final fits.
"""),
        code('''
# 1. CHOOSE — training scores only
t_sgd    = thresholds[np.where(recalls >= 0.90)[0][-1]]
t_forest = f_thr[np.where(f_rec >= 0.90)[0][-1]]
print(f"SGD threshold    {t_sgd:.2f}")
print(f"forest threshold {t_forest:.2f}")

# 2. MEASURE — once
sgd_final    = clf                                  # already fitted above
forest_final = forest.fit(X_train, y_train_5)

shift = {
    "SGD, default threshold":   sgd_final.decision_function(X_test) >= 0,
    "SGD, tuned for 90% recall": sgd_final.decision_function(X_test) >= t_sgd,
    "forest, tuned for 90% recall":
        forest_final.predict_proba(X_test)[:, 1] >= t_forest,
}
'''),
        code('''
CAPACITY = 1000            # stated by the client, not measured by us

print(f"{'operating point':30s}{'flagged':>8s}{'caught':>8s}"
      f"{'alarms':>8s}{'missed':>8s}{'recall':>9s}")
for name, pred in shift.items():
    tn_, fp_, fn_, tp2 = confusion_matrix(y_test_5, pred).ravel()
    flag = tp2 + fp_
    over = "  OVER CAPACITY" if flag > CAPACITY else ""
    print(f"{name:30s}{flag:>8,}{tp2:>8,}{fp_:>8,}{fn_:>8,}"
          f"{tp2 / (tp2 + fn_):>9.4f}{over}")
'''),
        md("""
The SGD classifier cannot satisfy both constraints: the only threshold that
reaches 90% recall flags 1,377 items, which is 377 over the desk's capacity.
**That is a finding, and it is the correct thing to report.**

The forest, tuned for the same 90% recall, flags 811 — inside capacity, because
almost none of them are false alarms.

### The wrinkle

The forest threshold was chosen because it gave 90.39% recall on the
cross-validated training scores. On the test shift it gives 89.91%. Half a point
below target.

Is that a failure, a bad fold, or noise? It is noise: a threshold chosen at
exactly 90% will land either side of it about half the time. **Report it, do not
tune it away.** Re-tuning to make the test number reach 90% is fitting the test
set.
"""),

        md("""
## 7 · An assistant improves the recall

**⚠ Read before running.** This one runs, it is correct code, it does exactly
what was asked, and the sentence it reports is true.

> *"My detector is missing too many 5s. Fix it so the recall is high."*
"""),
        code('''
idx99 = np.where(recalls >= 0.99)[0][-1]
threshold_high_recall = thresholds[idx99]

shift_scores = sgd_final.decision_function(X_test)
y_high_recall = (shift_scores >= threshold_high_recall)

print(f"Recall improved from "
      f"{recall_score(y_test_5, shift_scores >= 0):.4f} to "
      f"{recall_score(y_test_5, y_high_recall):.4f}")
'''),
        md("""
### The review question: *what happened to everything you did not ask about?*

It printed one number. That is the tell.
"""),
        code('''
before, after = (shift_scores >= 0), y_high_recall
print(f"{'':22s}{'before':>10s}{'after':>10s}")
for label, metric in (("recall", recall_score), ("precision", precision_score),
                      ("accuracy", accuracy_score)):
    print(f"{label:22s}{metric(y_test_5, before):>10.4f}"
          f"{metric(y_test_5, after):>10.4f}")

flag_before, flag_after = before.sum(), after.sum()
print(f"{'items flagged':22s}{flag_before:>10,}{flag_after:>10,}"
      f"     (capacity {CAPACITY:,})")
print(f"\\nextra false alarms per shift: "
      f"{(after & ~y_test_5).sum() - (before & ~y_test_5).sum():,}")
print(f"accuracy is now BELOW the never-fires baseline of "
      f"{1 - y_test_5.mean():.4f}")

assert accuracy_score(y_test_5, after) < 1 - y_test_5.mean()
'''),
        md("""
Nothing was retrained. One number was changed. The assistant did not write a
bug — **we wrote a bad specification.**

The previous lecture's failure cost 0.52 points. This one costs 51 points of
accuracy and six times the desk capacity, from a prompt that is just as
reasonable-looking.

### The corrected specification

> *"Choose a decision threshold on **cross-validated training scores** that
> reaches **at least 90% recall** while flagging **no more than 1,000 items per
> 10,000**. Report precision, recall and the flagged count together. If no
> threshold satisfies both, say so and show the closest."*

It names the objective *and* the constraint, names the data the choice is made
on, demands that metrics which move together be reported together, and gives
permission to fail — which is what stops it inventing a success.

**Never optimise one metric of a pair.** Precision and recall, bias and
variance, latency and accuracy. Ask for one and you will get it, at the price of
the other, and the price will not be in the reply.
"""),

        md("""
## 8 · Red-team

Swap notebooks with the team beside you. Eight minutes.

1. What touched the test set? Was the **threshold** chosen on it?
2. What was fitted, and on what?
3. What is the shape here — and which column of `predict_proba` did they take?
4. What was dropped? Any rows removed to make a number look better?
5. What is the default I did not ask for — **including the threshold at zero**?

Report what you **found**, not what you would have done differently.
"""),
    ]
    return cells
