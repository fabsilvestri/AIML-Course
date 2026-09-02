"""
Lecture 22 — Recommender systems: neural, and evaluated honestly.

Two-tower models, sampled softmax and the logQ correction, then the four
evaluation protocols measured against each other on one unchanged model.

Exports build() -> list[nbformat cell].

The four-cell table is the whole notebook. Everything before it exists so that
the table is reproduced rather than quoted, and everything after it is about
what the table does not say.

Runs on CPU in a few minutes. MovieLens 1M, binarised at 4.
"""

from __future__ import annotations

import nbformat as nbf
from _prompt import prompt                                # noqa: E402


def md(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(text.strip("\n"))


def code(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(text.strip("\n"))


HEADER = """
# Recommender systems: neural, and evaluated honestly

**Lecture 22** · *Lecture notes — outside the textbook, and examinable*

Applications of Machine Learning — BSc Mathematics of Artificial Intelligence

---

**How to use this notebook.** Read before you run. Every code cell is preceded
by the specification that would produce it — input, output, constraint, check.

Cells marked **⚠** deliberately run code that is wrong, and say so in the
heading before you reach them. They are the failures this lecture is about;
each runs the broken version beside the correct one and prices the difference.

Runs on CPU in a few minutes.

**Scale.** Identical to the deck: the whole of MovieLens 1M binarised at 4, one
model, four protocols. The full-catalogue protocols score every one of the 3,706
films for every test user — which is affordable, and is the point.
"""


def build() -> list:
    cells: list = [md(HEADER)]

    # ------------------------------------------------------------------ 1
    cells += [
        md("## 1 · Setup, and the implicit view of MovieLens"),
        prompt(
            label="setup",
            input="nothing",
            output="versions and the seed",
            constraint="cap the BLAS threads before numpy is imported",
            check="print the versions. The protocol comparison below is exact arithmetic, but the SVD is not, and its thread count changes the last digits."),
        code('''
import os
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

import sys, zipfile, urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

print(f"python  {sys.version.split()[0]}")
print(f"numpy   {np.__version__}")

SEED = 42
rng = np.random.default_rng(SEED)
'''),
        prompt(
            label="MovieLens 1M, binarised",
            input="the ratings file",
            output="the positive interactions, with timestamps",
            constraint="keep the timestamps — half of this notebook is the difference between splitting a history at random and splitting it in time, and without them that comparison cannot be made",
            check="assert the release's counts, then report how many ratings survive the threshold. A film rated 3 is now indistinguishable from one never seen, which is deliberate."),
        code('''
URL   = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
CACHE = Path("datasets/movielens")
CACHE.mkdir(parents=True, exist_ok=True)
z = CACHE / "ml-1m.zip"

if not z.is_file():
    print("downloading MovieLens 1M ...", flush=True)
    req = urllib.request.Request(URL, headers={"User-Agent": "curl/8"})
    z.write_bytes(urllib.request.urlopen(req, timeout=180).read())
if not (CACHE / "ml-1m" / "ratings.dat").is_file():
    with zipfile.ZipFile(z) as fz:
        fz.extractall(CACHE)

r = pd.read_csv(CACHE / "ml-1m" / "ratings.dat", sep="::", engine="python",
                names=["user", "item", "rating", "ts"], encoding="latin-1")
r["u"] = pd.factorize(r["user"])[0]
r["i"] = pd.factorize(r["item"])[0]
n_u, n_i = r["u"].nunique(), r["i"].nunique()
assert (len(r), n_u, n_i) == (1000209, 6040, 3706)

imp = r[r["rating"] >= 4].sort_values("ts").copy()
print(f"ratings                    {len(r):,}")
print(f"positive interactions      {len(imp):,}  ({100*len(imp)/len(r):.1f}%)")
print(f"cells in the grid          {n_u*n_i:,}")
print(f"cells with no interaction  {100*(1-len(imp)/(n_u*n_i)):.2f}% — meaning unknown")
'''),
    ]

    # ------------------------------------------------------------------ 2
    cells += [
        md("""
## 2 · Two held-out sets, from the same data

Decision 1 of the protocol: which interaction is held out.

* **Temporal** — the user's *most recent* interaction. Training contains only
  their past, which is what a deployed system has.
* **Random** — one interaction chosen uniformly. Training then contains the
  user's *future*, which no deployed system ever has.

The second is a leak by Lecture 2's definition: information in the training data
that will not be available at prediction time. No column was duplicated and
nothing was fitted before the split — the leak is in the *shape* of the
evaluation, which is why it survived as a default.
"""),
        prompt(
            label="the two held-out sets",
            input="the interaction table, sorted by time",
            output="one held-out interaction per user, under each rule",
            constraint="hold out exactly one per user under both rules, so the two protocols differ in WHICH interaction and in nothing else",
            check="assert one row per user WITH POSITIVES, not per user — a user who never gave a 4 or a 5 has nothing to hold out, and asserting against the full user count fails on exactly that."),
        code('''
last   = imp.groupby("u").tail(1)[["u", "i"]]
rand   = (imp.groupby("u")
             .apply(lambda g: g.iloc[rng.integers(len(g))], include_groups=False)
             .reset_index()[["u", "i"]])

# Not every user survives the threshold: a user who never gave a 4 or a 5 has
# no positive interactions and so cannot be evaluated at all. Assert against
# the users we actually have, and report how many the binarisation dropped --
# that is a property of the filtering step, and it belongs in the protocol.
n_eval = imp["u"].nunique()
assert len(last) == len(rand) == n_eval

overlap = len(set(map(tuple, last.values)) & set(map(tuple, rand.values)))
print(f"users in the dataset        {n_u:,}")
print(f"users with any positive     {n_eval:,}")
print(f"dropped by the threshold    {n_u - n_eval:,}")
print(f"held out per user           1")
print(f"same interaction in both    {overlap:,}  ({100*overlap/n_eval:.1f}%)")
'''),
        prompt(
            label="build the training matrix for a protocol",
            input="a held-out set",
            output="the binary interaction matrix with those interactions removed",
            constraint="remove exactly the held-out pairs and nothing else — removing all of a user's interactions with an item, when they interacted twice, quietly changes the training set between protocols",
            check="assert the training matrix has exactly len(imp) - n_u ones. If it has fewer, more was removed than was held out."),
        code('''
def build(hold):
    held = set(map(tuple, hold.values))
    R = np.zeros((n_u, n_i), dtype=np.float32)
    for uu, ii in zip(imp["u"].values, imp["i"].values):
        if (uu, ii) in held:
            held.discard((uu, ii))         # remove one occurrence, not all
            continue
        R[uu, ii] = 1.0
    return R

R_temporal = build(last)
R_random   = build(rand)
for name, R in (("temporal", R_temporal), ("random", R_random)):
    print(f"{name:<10} training interactions {int(R.sum()):,}")
'''),
    ]

    # ------------------------------------------------------------------ 3
    cells += [
        md("""
## 3 · One model, and the baseline that must be in every table

The model is a rank-32 factorisation of the binary matrix — the simplest
two-tower model there is, with both towers as lookup tables. The claim of this
lecture is not that it is good; it is that the protocol moves the number more
than the model does, and that argument is cleanest with a plain model.

The baseline is: **recommend the ten most popular items, to everybody**, minus
what the user has already interacted with. No personalisation, no parameters.
A recommender that does not beat it has not been shown to personalise anything.
"""),
        prompt(
            label="the two scorers",
            input="a training matrix",
            output="a scoring function per user, for the factorisation and for popularity",
            constraint="fit the factorisation once per split and reuse it for both candidate protocols — refitting per protocol would confound the two decisions",
            check="the popularity scorer must not depend on the user at all. If it does, it is not the baseline it claims to be."),
        code('''
from numpy.linalg import svd

def fit(R, k=32):
    U, S, Vt = svd(R, full_matrices=False)
    P, Q = U[:, :k] * S[:k], Vt[:k].T
    popularity = R.sum(0)                  # identical for every user
    return P, Q, popularity

def scorers(R):
    P, Q, popularity = fit(R)
    return {"factorisation": lambda u: P[u] @ Q.T,
            "popularity":    lambda u: popularity}

print("fitting both splits ...", flush=True)
sc = {"temporal": scorers(R_temporal), "random": scorers(R_random)}
print("done")
'''),
        md("""
## 4 · Decision 2: what the held-out item is ranked against

* **Full catalogue** — score all 3,706 films. This is what a deployed system
  does.
* **Sampled 100** — score the held-out item against 100 items the user did not
  interact with. A shortcut from when scoring a catalogue was expensive.

Ranking one item against 100 random others is a far easier task, and the 100 are
drawn uniformly, so they are mostly obscure films nobody would recommend. Note
what that implies before you see the numbers: the sample almost never contains a
*popular* item, which is the only real competitor for a top-ten slot.
"""),
        prompt(
            label="the two evaluators",
            input="a scorer, a training matrix, a held-out set",
            output="HR@10 and NDCG@10",
            constraint="remove already-seen items from the candidates in BOTH evaluators — leaving them in flatters popularity and the factorisation by different amounts, so the choice is not neutral between them",
            check="a random ranker should score about 10/101 under sampling and 10/3706 against the catalogue. Compute both floors and print them; they are most of the explanation for what follows.",
            **{"try": "drop the seen-item exclusion from the full-catalogue evaluator. Popularity falls furthest, because the films it recommends are the ones heavy users have already watched."}),
        code('''
def eval_full(score, R, hold):
    hr, nd = [], []
    for uu, gold in zip(hold["u"].values, hold["i"].values):
        s = score(uu).copy()
        s[R[uu] > 0] = -np.inf              # never re-recommend a seen item
        top = np.argpartition(-s, 10)[:10]
        top = top[np.argsort(-s[top])].tolist()
        hit = gold in top
        hr.append(int(hit))
        nd.append(1.0 / np.log2(top.index(gold) + 2) if hit else 0.0)
    return float(np.mean(hr)), float(np.mean(nd))

def eval_sampled(score, R, hold, n_neg=100, seed=SEED):
    g = np.random.default_rng(seed)
    hr, nd = [], []
    for uu, gold in zip(hold["u"].values, hold["i"].values):
        s = score(uu)
        seen = R[uu] > 0
        pool = g.choice(n_i, size=n_neg * 3, replace=False)
        negs = [c for c in pool if not seen[c] and c != gold][:n_neg]
        cand = np.array([gold] + negs)
        rank = int(np.argsort(-s[cand]).tolist().index(0)) + 1
        hr.append(int(rank <= 10))
        nd.append(1.0 / np.log2(rank + 1) if rank <= 10 else 0.0)
    return float(np.mean(hr)), float(np.mean(nd))

print(f"a random ranker scores about {10/101:.4f} under sampling")
print(f"and about {10/n_i:.4f} against the full catalogue")
print(f"the floor moves by a factor of {(10/101)/(10/n_i):.0f}")
'''),
    ]

    # ------------------------------------------------------------------ 5
    cells += [
        md("""
## 5 · The table

One model. One dataset. Four protocols. Predict the spread before you run it.
"""),
        prompt(
            label="⏱ 2 min — the four protocols",
            input="both splits, both candidate rules, both methods",
            output="HR@10 and NDCG@10 for each of the eight cells",
            constraint="change one thing at a time and hold everything else fixed — the same fitted model is scored under both candidate rules, and the same candidate rule is applied to both splits",
            check="assert the model beats popularity under every protocol. If the ordering flips somewhere, say so rather than hiding it — a reversal is a finding, not a bug."),
        code('''
results = {}
for split, hold, R in (("temporal", last, R_temporal), ("random", rand, R_random)):
    for method in ("factorisation", "popularity"):
        s = sc[split][method]
        hr_f, nd_f = eval_full(s, R, hold)
        hr_s, nd_s = eval_sampled(s, R, hold)
        results[(split, method)] = dict(hr_full=hr_f, ndcg_full=nd_f,
                                        hr_samp=hr_s, ndcg_samp=nd_s)
        print(f"  {split:<9} {method:<14} HR@10 full {hr_f:.4f}   sampled {hr_s:.4f}")

for split in ("temporal", "random"):
    for key in ("hr_full", "hr_samp"):
        assert results[(split, "factorisation")][key] > results[(split, "popularity")][key], \\
            f"popularity beat the model under {split}/{key} — report it, do not hide it"
'''),
        prompt(
            label="the four cells, as a table",
            input="the results dictionary",
            output="HR@10 for each protocol, and the ratio between the extremes",
            constraint="print the ratio explicitly — the whole lecture is that number, and reading it off two decimals in different rows is how it gets missed",
            check="the same model under two protocols should span close to an order of magnitude. If it does not, one of the four evaluations is not doing what it claims."),
        code('''
print(f"{'HR@10':<26} {'full':>8} {'sampled':>9}")
for split in ("temporal", "random"):
    for method in ("factorisation", "popularity"):
        v = results[(split, method)]
        print(f"{split + ' · ' + method:<26} {v['hr_full']:>8.4f} {v['hr_samp']:>9.4f}")

honest = results[("temporal", "factorisation")]["hr_full"]
best   = results[("random", "factorisation")]["hr_samp"]
print()
print(f"same model, same data: {honest:.4f} to {best:.4f} — a factor of {best/honest:.1f}")
'''),
        md("""
Nothing about the model changed. Only the two sentences describing how it was
scored.
"""),
        prompt(
            label="take it apart",
            input="the same results",
            output="the effect of each decision separately",
            constraint="isolate one decision at a time, holding the other fixed, and report the effect on BOTH methods — if a change lifts the unpersonalised baseline by the same factor, it is the task getting easier and not the model learning",
            check="the sampling effect should be much larger than the split effect. Both should apply to popularity too."),
        code('''
print("effect of the split (full catalogue):")
for method in ("factorisation", "popularity"):
    t = results[("temporal", method)]["hr_full"]
    q = results[("random", method)]["hr_full"]
    print(f"  {method:<14} {t:.4f} -> {q:.4f}   x{q/t:.2f}")

print("\\neffect of sampling (temporal split):")
for method in ("factorisation", "popularity"):
    t = results[("temporal", method)]["hr_full"]
    q = results[("temporal", method)]["hr_samp"]
    print(f"  {method:<14} {t:.4f} -> {q:.4f}   x{q/t:.1f}")
'''),
        prompt(
            label="the comparison that ends careers",
            input="two cells from opposite corners",
            output="popularity under the generous protocol, against the model under the honest one",
            constraint="print them side by side, because that is how they appear in a paper that quotes a baseline from another paper",
            check="both numbers are arithmetically correct. That is what makes the comparison dangerous rather than merely wrong."),
        code('''
generous = results[("random", "popularity")]["hr_samp"]
honest_m = results[("temporal", "factorisation")]["hr_full"]
print(f"popularity, random split, sampled 100      HR@10 {generous:.4f}")
print(f"factorisation, temporal split, full        HR@10 {honest_m:.4f}")
print()
print(f"the unpersonalised method reports {generous/honest_m:.1f}x the real one")
'''),
    ]

    # ------------------------------------------------------------------ 6
    cells += [
        md("""
## 6 · What is stable, and what the honest number is

The absolute numbers spanned nearly an order of magnitude. The **ratio to the
baseline**, under a fixed protocol, is far steadier — which is the practical
rule: report the baseline in the same table, under the same protocol. It does
not repair the protocol, and it does make your number readable by someone whose
protocol differs, which is everyone.
"""),
        prompt(
            label="the ratio to the baseline, per protocol",
            input="the results",
            output="the factorisation divided by popularity, under each of the four",
            constraint="four ratios, one per protocol, and never a ratio across protocols",
            check="the spread of the ratios should be far smaller than the spread of the absolutes. That gap is the argument for always reporting a baseline."),
        code('''
ratios = []
for split in ("temporal", "random"):
    for key, label in (("hr_full", "full"), ("hr_samp", "sampled")):
        rr = results[(split, "factorisation")][key] / results[(split, "popularity")][key]
        ratios.append(rr)
        print(f"  {split:<9} {label:<8} model / popularity = x{rr:.2f}")

print()
print(f"absolute HR@10 spanned  x{max(r[k] for r in [results[('random','factorisation')]] for k in ['hr_samp']) / results[('temporal','factorisation')]['hr_full']:.1f}")
print(f"the ratio spanned only  x{max(ratios)/min(ratios):.2f}")
'''),
        prompt(
            label="the honest headline",
            input="the temporal, full-catalogue cells",
            output="one sentence with every clause that makes it true",
            constraint="state the protocol in the same sentence as the number — a number without it is not a measurement",
            check="also print the chance floor, so the small absolute number is legible as a hard task rather than a failure."),
        code('''
f_hr = results[("temporal", "factorisation")]["hr_full"]
p_hr = results[("temporal", "popularity")]["hr_full"]
f_nd = results[("temporal", "factorisation")]["ndcg_full"]
print("A rank-32 factorisation places the user's next film in the top ten")
print(f"{100*f_hr:.1f}% of the time, against {100*p_hr:.1f}% for a popularity list,")
print(f"under a temporal split with all {n_i:,} films ranked.")
print()
print(f"  NDCG@10                        {f_nd:.4f}")
print(f"  chance                         {10/n_i:.4f}")
print(f"  the model, against chance      x{f_hr/(10/n_i):.0f}")
print(f"  the model, against popularity  x{f_hr/p_hr:.1f}")
'''),
        md("""
Small absolute numbers are what an honest protocol on a hard task looks like:
place *one specific film* in the top ten of 3,706, from a history that ends
before it.

A field that finds small numbers embarrassing will drift toward protocols that
produce large ones. That drift is not fraud, and it does the same damage.
"""),
        prompt(
            label="what the metric cannot see",
            input="what each method recommends",
            output="catalogue coverage — the fraction of films that ever appear in anyone's top ten",
            constraint="count distinct items across all users' recommendations, not per user — a model that gives everyone a different ordering of the same fifty films has high per-user diversity and no coverage",
            check="popularity's coverage should be near zero by construction. If the factorisation's is also near zero, it is a popularity list with extra steps."),
        code('''
def coverage(score, R, hold):
    shown = set()
    for uu in hold["u"].values:
        s = score(uu).copy()
        s[R[uu] > 0] = -np.inf
        top = np.argpartition(-s, 10)[:10]
        shown.update(int(t) for t in top)
    return len(shown) / n_i

for method in ("popularity", "factorisation"):
    c = coverage(sc["temporal"][method], R_temporal, last)
    print(f"  {method:<14} catalogue coverage {100*c:.1f}%")
print()
print("HR@10 gives full credit for a correct popular recommendation and the")
print("same credit for a correct obscure one. Coverage is one thing it cannot see.")
'''),
        md("""
## 7 · Five questions for a recommender paper

1. **Full catalogue or sampled?** If sampled, the number is not comparable to
   anything.
2. **Random split or temporal?** A random split leaks the user's future.
3. **Is popularity in the table**, under the same protocol?
4. **Were the baselines tuned** as carefully as the proposed model, or quoted
   from elsewhere?
5. **Is there an online result?** If not, the claim is about a log, not about
   users.

---

**Part V closed.** Four lectures, two datasets, and not one method you had not
already met. What they needed was the discipline of Part I applied to problems
whose ground truth is itself a measurement.

**Next.** Lecture 23 takes Lecture 20's contrastive objective and puts an image
on one side of it.
"""),
    ]

    return cells
