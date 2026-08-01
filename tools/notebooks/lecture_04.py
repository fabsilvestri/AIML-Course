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

from make_notebooks import code, header, md, SETUP, SETUP_PROMPT        # noqa: E402
from _prompt import prompt                                # noqa: E402


def build() -> list:
    cells = header(
        4, "It never fires", "fix", "Chapter 3",
        thread="imbalance, and the non-monotonicity of precision")

    cells += [
        md("## 1 · Setup and where we left off"), SETUP_PROMPT, SETUP,
        prompt(
            label="setup",
            input="nothing",
            output="every import this notebook uses, in one cell",
            constraint="no import anywhere below this cell, so the notebook does not depend on a previous one still being in memory",
            left_open="it does not say to pin versions. `precision_recall_curve` returned one more threshold than precision in older scikit-learns, and cell 10 asserts the current shape — on the wrong version that assert is the only thing between you and an off-by-one you would never see.",
            student="importing as you go, so cell 14 works today because cell 4 is still in the kernel. Restart the runtime and the notebook is a stranger.",
            catch="Runtime -> Restart, then run this cell alone. If anything below it raises NameError, the import belongs here."),
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
        prompt(
            label="the data, and the split that already exists",
            input="MNIST from OpenML",
            output="X_train/X_test split 60,000 / 10,000, and boolean is-it-a-5 labels",
            constraint="do NOT call train_test_split — MNIST arrives shuffled and already partitioned, and re-splitting it silently mixes the two halves",
            check="assert the counts: 5,421 fives in train, 892 in test",
            left_open="nothing about scaling, because there is none here. Pixels are already 0-255 on a common scale; the pipeline in cell 4 adds the scaler where it belongs, inside cross-validation.",
            student="`train_test_split(X, y, test_size=0.2)` out of habit. It runs, it looks tidier, and it destroys the only reason this dataset is comparable across every paper that has used it.",
            catch="the two asserted counts. They are properties of the canonical split, so they fail the moment somebody reshuffles it."),
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
        prompt(
            label="what accuracy is worth here",
            input="a list of base rates",
            output="the accuracy of a model that never fires, at each",
            constraint="print it before any model is fitted, so the number is not a reaction to a result",
            left_open="which base rate is ours. 0.09035 is in the list without being labelled as ours, and the whole argument turns on it: 90.965% accuracy is what you get for predicting 'not a 5' forever.",
            student="skipping this and computing accuracy after fitting. The number then arrives with no yardstick beside it and 97% sounds excellent.",
            catch="ask what accuracy the empty model gets before you ask what yours gets. If you cannot beat it by a margin you would defend, you have no result."),
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
        prompt(
            label="out-of-fold scores",
            input="the scaled SGD pipeline and the 60,000 training rows",
            output="one decision-function value per training instance, from a model that never saw it",
            constraint="`method='decision_function'`, not `predict` — one call then gives both the labels and the entire threshold sweep",
            check="the array is 60,000 long, and thresholding at 0 reproduces predict()",
            left_open="that these scores are the ONLY ones any later cell may choose a threshold on. Nothing in the code stops cell 14 reaching for the test set instead, and the whole lecture rests on it not doing so.",
            student="`cross_val_predict(..., method='predict')`, then refitting to get scores when the threshold section arrives — two fits, and the scores no longer correspond to the labels already reasoned about.",
            catch="`(y_scores >= 0)` must equal what `predict` would return. SGD predicts positive exactly when the decision function is non-negative, so if those disagree you have the wrong method or the wrong estimator."),
        code('''
y_scores = cross_val_predict(clf, X_train, y_train_5, cv=cv,
                             method="decision_function", n_jobs=-1)

assert y_scores.shape == (60000,)
# SGDClassifier predicts positive exactly when the decision function is >= 0,
# so one call gives us both the labels and the whole threshold sweep.
y_pred = (y_scores >= 0)
print(f"accuracy {accuracy_score(y_train_5, y_pred):.5f}")
'''),
        prompt(
            label="is precision monotone?",
            input="the out-of-fold scores and their labels",
            output="how often precision FALLS as the threshold rises, counted",
            constraint="walk the ranking one instance at a time rather than sampling a grid of thresholds — a grid can step over every fall there is",
            check="the two counts must sum to the number of steps",
            left_open="`kind='stable'` is doing real work and the prompt does not say why. Ties in the score are common; an unstable sort orders them arbitrarily, so the count of falls would change between numpy versions on identical data.",
            student="sweeping 100 evenly spaced thresholds and concluding the curve is monotone because their 100 samples happened to be. The falls are one instance wide; a grid does not see them.",
            catch="you should find thousands of falls, not zero and not a handful — one for essentially every 5 in the data. If you find none, you are sampling rather than walking."),
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
        prompt(
            label="the smallest possible example",
            input="the top eight of the ranking",
            output="precision at each of the first eight cut-offs",
            constraint="show the arithmetic, not a plot — the point is that 4/5 < 5/6",
            check="assert precision at k=5 is below precision at k=6",
            left_open="it does not say what makes this convincing. Eight rows is not evidence about the whole curve; it is evidence about the mechanism, and the mechanism is what generalises.",
            student="believing the previous cell's counts without ever seeing one. A number you cannot reconstruct by hand is a number you will not argue with when it is wrong.",
            catch="do the division yourself. 5/6 = 0.833, 4/5 = 0.800. Raising the threshold past a real 5 lost you precision, which the word 'threshold' does not prepare you for."),
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
        prompt(
            label="the four numbers",
            input="true labels and predictions at threshold 0",
            output="the confusion matrix, and its four cells named",
            constraint="print the raw matrix as well as the named cells — scikit-learn's cell order is not the one most textbooks draw",
            check="the four cells sum to 60,000",
            left_open="which corner is which. `ravel()` gives tn, fp, fn, tp in that order, and half the diagrams in circulation put them somewhere else. Nothing warns you if you unpack them wrongly.",
            student="reading the matrix off the printout by eye and mislabelling a corner, then reporting a recall that is actually specificity.",
            catch="tn should be the largest cell by far — about 54,000 of 60,000 rows are not 5s. If your 'true negatives' is a small number, your unpacking is the wrong way round."),
        code('''
cm = confusion_matrix(y_train_5, y_pred)
tn, fp, fn, tp_ = cm.ravel()
print(cm)
print()
print(f"true negatives  {tn:>7,}     false positives {fp:>7,}")
print(f"false negatives {fn:>7,}     true positives  {tp_:>7,}")

assert tn + fp + fn + tp_ == 60000
'''),
        prompt(
            label="the identity behind the accuracy",
            input="the same predictions",
            output="accuracy, precision, recall, F1 and specificity, plus the identity accuracy = p*recall + (1-p)*specificity evaluated on our own numbers",
            constraint="verify the identity numerically rather than asserting it in prose",
            check="the two sides agree to floating-point tolerance",
            left_open="what the identity is FOR. It shows that at a base rate of 0.09, 91% of the accuracy is bought by specificity — so accuracy is mostly a report on the negatives, which nobody asked about.",
            student="quoting F1 as 'the balanced one' and stopping. F1 balances precision against recall at a fixed threshold; it says nothing about whether that threshold is the one the client wants.",
            catch="compute the right-hand side yourself and watch it land on the accuracy you already printed. If prose and arithmetic disagree, the prose is the bug."),
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
        prompt(
            label="one digit, two thresholds",
            input="a single training image known to be a 5",
            output="its decision-function score, and the prediction at two thresholds",
            constraint="⏱ this refits on all 60,000 rows, about 30 seconds",
            left_open="that the model is being fitted on data including this very digit, so the score is not an out-of-fold one. That is fine for showing what a threshold does and fatal for any number you would report.",
            student="reusing this fitted `clf` for an accuracy figure later. It has seen everything; every score it produces is optimistic.",
            catch="ask of any score in this notebook: was the model that produced it fitted on this row? Here the answer is yes, and it is only used to demonstrate a mechanism."),
        code('''
some_digit = X_train[0]                       # a 5
clf.fit(X_train, y_train_5)                   # ⏱ about 30 s
score = clf.decision_function([some_digit])
print("score:", score.round(1))

for t in (0, 3000):
    print(f"threshold {t:>5}: predicted {bool(score[0] >= t)}")
'''),
        prompt(
            label="the two curves",
            input="the out-of-fold scores",
            output="precision and recall against threshold, and against each other",
            constraint="plot precision-recall rather than ROC — at a 9% base rate ROC flatters every model, which is the next section",
            check="assert precisions and recalls are exactly one longer than thresholds",
            left_open="why the lengths differ. scikit-learn appends the degenerate point where nothing is flagged and precision is DEFINED as 1 — it is a convention, not a measurement, and plotting it against thresholds without the [:-1] silently shifts every point by one.",
            student="`ax.plot(thresholds, precisions)` and a broadcast error, or worse, equal lengths on some other version and a curve quietly off by one.",
            catch="the assert. It is three symbols and it pins a convention that has changed between library versions."),
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
        prompt(
            label="the 90%-precision operating point",
            input="the precision-recall curve",
            output="the lowest threshold reaching 90% precision, with enough flagged digits behind it to be a claim rather than a coincidence",
            constraint="the crossing must hold with at least MIN_SUPPORT instances flagged; report whether that guard changed the answer",
            check="compare against the unguarded first crossing and print which happened",
            left_open="what MIN_SUPPORT should be. 500 is a judgement, not a measurement, and the prompt does not defend it — so the cell prints the support it actually got (4,416) and lets you decide the guard was slack.",
            student="`(precisions >= 0.90).argmax()` alone. On a curve this lecture has just proved is not monotone, that is the first lucky step, and it may be held up by a dozen digits.",
            catch="the same guard applied at 99% precision FAILS — only 116 digits are flagged there, on a plateau 21 thresholds wide. A guard that never fires anywhere is not evidence that nothing is wrong."),
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
        prompt(
            label="what rarity does to the metrics",
            input="the same scores, with positives subsampled to three base rates",
            output="ROC AUC and average precision at each",
            constraint="subsample the POSITIVES only, leaving every negative in place, so the ranking is untouched and only the prevalence changes",
            check="ROC AUC should barely move; average precision should collapse",
            left_open="that this is the whole argument for preferring PR to ROC here, and it is made by construction rather than by assertion — the model is literally identical across the three rows.",
            student="comparing two different models at two different base rates and concluding something about the metrics. Change one thing.",
            catch="ROC AUC moving as much as average precision would mean you resampled the negatives too, and the ranking changed underneath you."),
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
        prompt(
            label="a second model, scored the same way",
            input="the same 60,000 rows and the same folds",
            output="out-of-fold probabilities for a random forest, and its curves beside the SGD's",
            constraint="`predict_proba` gives an (n, 2) array — take column 1, the POSITIVE class; column 0 is the exact complement and ranks everything backwards",
            check="assert the shape is (60000, 2) before indexing it",
            left_open="that `n_jobs=-1` appears twice — on the forest and on cross_val_predict. Nesting two pools oversubscribes the machine and can be slower than one. The code sets it in one place only.",
            student="`y_proba[:, 0]`, which produces a model that looks catastrophically worse than chance rather than one that is broken. An AUC of 0.03 is not a bad model, it is a sign flip.",
            catch="an AUC below 0.5 on a model that trains normally is almost always the wrong column or an inverted label, not a genuine result."),
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
        prompt(
            label="choose, then measure once",
            input="the training curves for both models, and the untouched test shift",
            output="a threshold per model chosen on training scores, then one pass over the test set",
            constraint="every threshold is chosen before the test set is touched, and the test set is touched exactly once",
            check="the code reads top to bottom as choose-then-measure, with no threshold computed after a test score is printed",
            left_open="that `np.where(...)[0][-1]` takes the LAST index, the highest threshold still clearing 90% recall — the opposite end from the precision constraint two cells ago, which wants the first. Getting these the same way round is a silent, common error.",
            student="tuning the threshold until the test recall reads 0.90. It will, and the number will mean nothing.",
            catch="scroll position is the evidence. If a threshold is assigned anywhere below a printed test metric, the choice was informed by the answer."),
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
        prompt(
            label="the constraint the metric cannot see",
            input="each operating point's predictions on the test shift",
            output="flagged, caught, false alarms, missed and recall per operating point, with anything over capacity marked",
            constraint="the desk can re-check 1,000 items a shift; an operating point that flags more is not a worse option, it is an unavailable one",
            check="at least one row is marked OVER CAPACITY, or the constraint is not binding and there is nothing to teach here",
            left_open="what to do about a row that exceeds capacity. Marking it is not deciding; Lecture 5 turns the same situation into a ranking, because when capacity binds you stop thresholding and start choosing the worst K.",
            student="reporting recall alone and letting the reader assume the plan is deliverable. Recall says nothing about how many people it takes.",
            catch="multiply the flagged rate by the shift volume and compare it with the staffing. A model nobody can act on has an accuracy and no value."),
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
        prompt(
            label="the assistant's answer",
            input="the same curve",
            output="the threshold giving 99% recall, applied to the test shift",
            constraint="this is the cell to read before running — it is what an assistant returns when asked to 'improve recall', and it does exactly that",
            left_open="everything except recall. The request named one metric, so one metric is what improved; nothing in the prompt mentions alarms, capacity, or the baseline.",
            student="this IS the student version, and it is also the assistant version. Asking for more of a number is the most natural request in machine learning and the most reliable way to get a worse model.",
            catch="recall going up is not the finding. Ask what went down, and the next cell answers: 5,303 extra false alarms and an accuracy below the never-fires baseline."),
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
        prompt(
            label="what it cost",
            input="the before and after predictions on the test shift",
            output="recall, precision and accuracy side by side, the flagged counts against capacity, and the extra false alarms",
            constraint="report every metric that moved, not the one that was asked for",
            check="accuracy after the change should sit BELOW the never-fires baseline",
            left_open="which of these the client would have cared about. The table does not rank them, because that is the conversation the table exists to start.",
            student="reporting the improved recall to the client and letting them discover the alarm volume in production.",
            catch="an accuracy below the empty model is the loudest signal in this notebook. If your change puts you there, no single improved metric rescues it."),
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
