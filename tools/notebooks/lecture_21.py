"""
Lecture 21 — Recommender systems: from ratings to factors.

Derivation 21: the low-rank model, why the SVD of Lecture 8 cannot be taken,
the observed-entry objective, ALS and SGD, and bias terms. Then item-item
neighbourhoods and implicit feedback.

Exports build() -> list[nbformat cell].

MovieLens 1M rather than the smaller release, because Lecture 22 needs the
timestamps and because the two lectures must share one split.

Runs on CPU in a few minutes. ALS is a d x d solve per user and there are only
6,040 users; the dense SVD of the filled matrix is the slowest cell, and it is
there because seeing that number is the point of the derivation.
"""

from __future__ import annotations

import nbformat as nbf
from _prompt import prompt                                # noqa: E402


def md(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(text.strip("\n"))


def code(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(text.strip("\n"))


HEADER = """
# Recommender systems: from ratings to factors

**Lecture 21** · *Lecture notes — outside the textbook, and examinable* ·
*Derivation: matrix factorisation, and the SVD it is not*

Applications of Machine Learning — BSc Mathematics of Artificial Intelligence

---

**These notes are the primary source for this lecture.** Lectures 19–22 sit outside the textbook, so the extended notes — [lecture-21.pdf](https://fabsilvestri.github.io/AIML-Course/notes/lecture-21.pdf) — are what you are examined from, and this notebook is where their figures come from.

**How to use this notebook.** Read before you run. Every code cell is preceded
by the specification that would produce it — input, output, constraint, check.

Cells marked **⚠** deliberately run code that is wrong, and say so in the
heading before you reach them. They are the failures this lecture is about;
each runs the broken version beside the correct one and prices the difference.

Runs on CPU in a few minutes.

**Scale.** Identical to the deck: the whole of MovieLens 1M, one 90/10 split
fixed before any model is fitted, and every model measured on the same held-out
ratings.
"""


def build() -> list:
    cells: list = [md(HEADER)]

    # ------------------------------------------------------------------ 1
    cells += [
        md("## 1 · Setup and the data"),
        prompt(
            label="setup",
            input="nothing",
            output="versions and the seed",
            constraint="cap the BLAS threads before numpy is imported — the dense SVD later in this notebook otherwise saturates every core and the notebook becomes a benchmark of the machine",
            check="print the versions. MovieLens has several releases and they give different numbers.",
            **{"try": "unset the two thread caps and time the dense SVD in "
                      "Section 4. The RMSE is identical and the wall clock is "
                      "not. Both are numbers this notebook prints; only one "
                      "of them is a property of the method."}),
        code('''
import os
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

import sys, ssl, zipfile, urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

print(f"python  {sys.version.split()[0]}")
print(f"numpy   {np.__version__}")
print(f"pandas  {pd.__version__}")

SEED = 42
rng = np.random.default_rng(SEED)
'''),
        prompt(
            label="MovieLens 1M",
            input="the GroupLens zip",
            output="a ratings table with contiguous user and item indices",
            constraint="factorise the ids into 0..n-1 — the raw ids are not contiguous, and using them as array indices silently allocates a matrix with empty rows",
            check="assert the counts against the release's published figures: 1,000,209 ratings, 6,040 users, 3,706 films. If they differ you have a different release and every number below will differ too.",
            **{"try": "use r['item'] directly as the column index rather than "
                      "factorising it. The raw film ids run to 3,952 with "
                      "gaps, so the matrix gains 246 all-zero columns and the "
                      "printed density falls. The assert in this cell is the "
                      "only thing that would have told you."}),
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
r["u"] = pd.factorize(r["user"])[0]      # contiguous, so they can index arrays
r["i"] = pd.factorize(r["item"])[0]

n_u, n_i = r["u"].nunique(), r["i"].nunique()
assert (len(r), n_u, n_i) == (1000209, 6040, 3706), "not the ml-1m release"

density = len(r) / (n_u * n_i)
print(f"{len(r):,} ratings, {n_u:,} users, {n_i:,} films")
print(f"cells in the matrix        {n_u*n_i:,}")
print(f"density                    {100*density:.2f}%")
print(f"missing                    {100*(1-density):.1f}%")
'''),
        md("""
**Missing is not zero.** A cell is empty because the user never saw the film, or
saw it and did not rate it, or was never shown it by whatever system was running
at the time. None of those is "disliked", and the whole derivation below turns
on refusing to pretend otherwise.
"""),
        prompt(
            label="what people actually give",
            input="the rating column",
            output="the distribution, and the mean",
            constraint="report shares rather than counts — the shape is the point, and it is not the uniform 1-5 the scale suggests",
            check="the distribution should be heavily skewed upward. People mostly rate things they liked.",
            **{"try": "compute the mean rating per user and then average those. "
                      "It is not the overall mean printed here, because heavy "
                      "raters carry different weight in the two. Which of them is "
                      "the right anchor for the user-mean baseline in the next "
                      "section?"}),
        code('''
dist = r["rating"].value_counts().sort_index()
for k, v in dist.items():
    print(f"  {k} stars   {v:>7,}   {100*v/len(r):>5.1f}%")
print(f"\\nmean rating   {r['rating'].mean():.2f}")
print(f"4 or 5        {100*(r['rating'] >= 4).mean():.1f}%")
print(f"\\nratings per user, median   {r.groupby('u').size().median():.0f}")
print(f"ratings per film, median   {r.groupby('i').size().median():.0f}")
'''),
        prompt(
            label="the long tail",
            input="the rating counts per film",
            output="the share of ratings held by the most-rated films",
            constraint="report it at several cut points, because a single one can be chosen to say anything",
            check="also count the films with fewer than twenty ratings. Those are where a recommendation would be valuable and where every model here has least to work with.",
            **{"try": "recompute these shares over the TRAINING half alone. "
                      "They barely move, which is what makes the long tail a "
                      "property of the corpus rather than of the split — "
                      "worth checking rather than assuming, since every later "
                      "model is fitted on that half."}),
        code('''
pop = r.groupby("i").size().sort_values(ascending=False).values
for frac in (0.01, 0.05, 0.10, 0.20):
    share = pop[:max(1, int(frac * n_i))].sum() / len(r)
    print(f"top {100*frac:>4.0f}% of films hold {100*share:>5.1f}% of ratings")
print(f"\\nfilms with fewer than 20 ratings   {100*np.mean(pop < 20):.1f}%")
'''),
    ]

    # ------------------------------------------------------------------ 2
    cells += [
        md("""
## 2 · The split, and the baselines

One 90/10 split of the ratings, fixed here and used by every model below.

A random split of ratings puts a user's future in the training set and their
past in the test set, which no deployed system ever gets. It is the standard
protocol for rating prediction and it is optimistic; Lecture 22 measures how
optimistic.
"""),
        prompt(
            label="the split",
            input="all the ratings",
            output="train and test index arrays",
            constraint="one seed, fixed before any model is fitted, and reused by everything — a table whose rows were scored on different splits is not a table",
            check="assert the two halves are disjoint and cover everything.",
            **{"try": "split by time instead: the last 10% of ratings by "
                      "timestamp. Every RMSE in the notebook rises. A random "
                      "split over ratings puts a user's later opinions into "
                      "the training set, and no deployed model is ever "
                      "allowed to know them."}),
        code('''
perm   = rng.permutation(len(r))
n_test = int(0.1 * len(r))
te, tr = perm[:n_test], perm[n_test:]
assert len(set(te) & set(tr)) == 0 and len(te) + len(tr) == len(r)

u_tr, i_tr, y_tr = r["u"].values[tr], r["i"].values[tr], r["rating"].values[tr].astype(float)
u_te, i_te, y_te = r["u"].values[te], r["i"].values[te], r["rating"].values[te].astype(float)

def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))

print(f"train  {len(tr):,} ratings")
print(f"test   {len(te):,} ratings")
'''),
        prompt(
            label="the baselines",
            input="the training ratings",
            output="RMSE for the global mean, the user mean and the film mean",
            constraint="compute the means on the TRAINING half only — a mean that has seen the test ratings is a leak, and it is the easiest one in this whole course to commit by accident",
            check="a user or film absent from the training half must fall back to the global mean, not to a division by zero. Assert there are no NaNs in the predictions.",
            **{"try": "compute the two means over all the ratings rather than "
                      "the training half. Every baseline improves and the "
                      "factorisation's advantage shrinks. The leak is one "
                      "subscript, and nothing in this cell would catch it."}),
        code('''
mu = float(y_tr.mean())
cu = np.bincount(u_tr, minlength=n_u)
ci = np.bincount(i_tr, minlength=n_i)
user_mean = np.where(cu > 0, np.bincount(u_tr, weights=y_tr, minlength=n_u) / np.maximum(cu, 1), mu)
item_mean = np.where(ci > 0, np.bincount(i_tr, weights=y_tr, minlength=n_i) / np.maximum(ci, 1), mu)

baselines = {
    "global mean": rmse(y_te, np.full(len(y_te), mu)),
    "user mean":   rmse(y_te, user_mean[u_te]),
    "item mean":   rmse(y_te, item_mean[i_te]),
}
for k, v in baselines.items():
    print(f"  {k:<14} rmse {v:.4f}")
'''),
        prompt(
            label="the bias model",
            input="the training ratings",
            output="mu + b_u + b_i, fitted by alternating closed forms",
            constraint="shrink each offset by the count: (count + lambda) in the denominator, so a user with three ratings is not given a large offset on that evidence",
            check="print the extremes of both bias vectors. The spread should be larger than most of what the factorisation adds later, which is the point of the slide.",
            **{"try": "set lambda to 0 and look at the extremes again. The largest offsets will belong to users and films with almost no ratings."}),
        code('''
LAM = 10.0
bu = np.zeros(n_u)
bi = np.zeros(n_i)
for _ in range(15):
    bu = np.bincount(u_tr, weights=y_tr - mu - bi[i_tr], minlength=n_u) / (cu + LAM)
    bi = np.bincount(i_tr, weights=y_tr - mu - bu[u_tr], minlength=n_i) / (ci + LAM)

baselines["bias model"] = rmse(y_te, np.clip(mu + bu[u_te] + bi[i_te], 1, 5))
print(f"  bias model     rmse {baselines['bias model']:.4f}")
print()
print(f"user offset   {bu.min():+.4f} to {bu.max():+.4f}")
print(f"film offset   {bi.min():+.4f} to {bi.max():+.4f}")
'''),
    ]

    # ------------------------------------------------------------------ 3
    cells += [
        md("""
## 3 · ⚠ Derivation 21, steps 3–5: taking the SVD anyway

The best rank-$d$ approximation of a matrix in the Frobenius norm is its
truncated SVD. The theorem is about **one given matrix**, and its norm sums over
**every** entry — so to use it we must invent values for the 95.5% we do not
have.

**The next cell is deliberately wrong.** It fills the matrix and takes the SVD,
which is the mistake the derivation exists to rule out. Predict the RMSE before
you run it.
"""),
        prompt(
            label="⚠ fill the missing entries and factorise",
            input="the training ratings, as a dense matrix with the gaps filled",
            output="the rank-32 truncated SVD, scored on the same held-out ratings",
            constraint="try both fills — zeros and the global mean — because the difference between them is itself the lesson",
            check="compare against the global-mean predictor. If the optimal rank-32 approximation is worse than predicting one number for everybody, the objective was wrong, not the arithmetic.",
            **{"try": "raise the rank to 128. It gets WORSE, because more capacity means a better fit to the invented entries."}),
        code('''
from numpy.linalg import svd

def svd_fill(fill_value, k=32):
    M = np.full((n_u, n_i), fill_value, dtype=np.float64)
    M[u_tr, i_tr] = y_tr                     # observed entries overwrite the fill
    U, S, Vt = svd(M, full_matrices=False)
    R = (U[:, :k] * S[:k]) @ Vt[:k]
    return rmse(y_te, np.clip(R[u_te, i_te], 1, 5))

fills = {"filled with zeros": svd_fill(0.0),
         "filled with the mean": svd_fill(mu)}
for name, v in fills.items():
    print(f"  rank-32 SVD, {name:<22} rmse {v:.4f}")
print(f"  {'predicting the global mean':<36} rmse {baselines['global mean']:.4f}")
print(f"  {'the bias model':<36} rmse {baselines['bias model']:.4f}")
'''),
        md("""
The zero-filled SVD is **worse than predicting one number for everybody**, from
the mathematically optimal rank-32 approximation.

The SVD did exactly what it promised: the best rank-32 approximation *to the
matrix it was given*. That matrix is 95.5% fill, so the best approximation is
mostly a good approximation of the fill. Ratings run 1 to 5, so a zero is not a
low rating — it is off the scale, and the whole approximation is dragged toward
it.

Filling with the mean is less absurd and still worse than the bias model, which
has no factors at all.

**Imputing missing values to make a method applicable makes the method answer a
different question.**
"""),
    ]

    # ------------------------------------------------------------------ 4
    cells += [
        md("""
## 4 · Steps 6–11: change the objective, not the data

$$\\min_{P,Q} \\sum_{(u,i) \\in \\Omega} (r_{ui} - p_u \\cdot q_i)^2
+ \\lambda(\\|P\\|_F^2 + \\|Q\\|_F^2)$$

The sum runs over observed entries only. Nothing is imputed — and we have left
the world where the SVD applies: no closed form, no uniqueness, no orthogonal
factors, and a non-convex problem.

Fix $Q$, though, and each $p_u$ is a ridge regression over that user's own
ratings, with the closed form of step 10.
"""),
        prompt(
            label="group the ratings once",
            input="the training index arrays",
            output="a row-grouped view: which ratings belong to each user, and to each film",
            constraint="one argsort, not a boolean mask per user — a mask over a million ratings, 6,040 times, is the difference between seconds and minutes for identical arithmetic",
            check="assert the groups partition the ratings and that a spot-checked user's group contains exactly their ratings.",
            **{"try": "replace the grouping with a boolean mask, u_tr == k, "
                      "inside the ALS loop, and time one sweep. It is the "
                      "identical arithmetic and takes minutes instead of "
                      "seconds. Where else in this notebook is an O(n) scan "
                      "hiding inside a loop over n?"}),
        code('''
def group(idx, n):
    """order[start[k]:start[k+1]] are the entries with idx == k."""
    order = np.argsort(idx, kind="stable")
    start = np.concatenate([[0], np.cumsum(np.bincount(idx, minlength=n))])
    return order, start

ou, su = group(u_tr, n_u)
oi, si = group(i_tr, n_i)

assert su[-1] == len(u_tr) and si[-1] == len(i_tr)
k = 7
assert set(u_tr[ou[su[k]:su[k+1]]]) == {k}
print(f"user {k} rated {su[k+1]-su[k]} films in the training half")
'''),
        prompt(
            label="ALS, from step 10",
            input="the training ratings, a rank, a regularisation",
            output="P and Q, plus the fitted biases",
            constraint="solve (Q_u^T Q_u + lambda I) p_u = Q_u^T r_u for each user — Q_u contains ONLY the films that user rated, which is where the missing entries go: they never enter the system",
            check="assert the training error does not increase between sweeps. Each half-step solves its subproblem exactly, so it cannot; if it does, the systems are being built wrong.",
            **{"try": "set reg to 0. Users with a handful of ratings get enormous factor vectors, and the test RMSE rises even though the training error falls."}),
        code('''
def als(d=16, iters=12, reg=10.0, biases=True, verbose=False):
    if biases:
        resid = y_tr - mu - bu[u_tr] - bi[i_tr]
    else:
        resid = y_tr - mu

    g = np.random.default_rng(SEED)
    P = g.normal(0, 0.1, (n_u, d))
    Q = g.normal(0, 0.1, (n_i, d))
    I = np.eye(d)

    def half(F, G, other, order, start, n):
        for k in range(n):
            sl = order[start[k]:start[k+1]]
            if sl.size == 0:
                F[k] = 0.0
                continue
            Gk = G[other[sl]]                  # only the entries this row observed
            F[k] = np.linalg.solve(Gk.T @ Gk + reg * I, Gk.T @ resid[sl])

    prev = np.inf
    for it in range(iters):
        half(P, Q, i_tr, ou, su, n_u)
        half(Q, P, u_tr, oi, si, n_i)
        train = rmse(resid, (P[u_tr] * Q[i_tr]).sum(1))
        assert train <= prev + 1e-9, "an ALS sweep increased the training error"
        prev = train
        if verbose:
            print(f"  sweep {it+1:>2}   train rmse on the residual {train:.4f}")
    return P, Q

def predict(P, Q, u, i, biases=True):
    p = (P[u] * Q[i]).sum(1) + mu
    if biases:
        p = p + bu[u] + bi[i]
    return np.clip(p, 1.0, 5.0)

P, Q = als(d=16, verbose=True)
print(f"\\nrank 16, with biases   test rmse {rmse(y_te, predict(P, Q, u_te, i_te)):.4f}")
'''),
        prompt(
            label="how many factors?",
            input="a range of ranks",
            output="test RMSE for each",
            constraint="hold the regularisation, the sweeps and the split fixed, so the rank is the only thing that changed between rows",
            check="expect a U-curve. If the error falls monotonically, the largest rank tried is not yet large enough to overfit and the sweep should be extended.",
            **{"try": "extend the sweep to ranks 128 and 256. Does the "
                      "U-curve turn, and where? A sweep that stops before the "
                      "minimum will always report the largest rank you tried "
                      "as the best one."}),
        code('''
sweep = []
for d in (1, 2, 4, 8, 16, 32, 64):
    Pd, Qd = als(d=d)
    sweep.append((d, rmse(y_te, predict(Pd, Qd, u_te, i_te))))
    print(f"  rank {d:>2}   rmse {sweep[-1][1]:.4f}")

best = min(sweep, key=lambda s: s[1])
print(f"\\nbest rank {best[0]} at rmse {best[1]:.4f}")
print(f"bias model alone         {baselines['bias model']:.4f}")
print(f"even rank 1              {sweep[0][1]:.4f}")
'''),
        md("""
A U-curve, which is Lecture 5 exactly: more capacity fits the training ratings
better and the held-out ratings worse. The rank is a capacity parameter and it
is chosen the way every capacity parameter in this course is chosen.
"""),
        prompt(
            label="are the biases earning their place?",
            input="the best rank, with and without explicit bias terms",
            output="both test RMSEs",
            constraint="change only the bias terms between the two runs",
            check="report whichever way it comes out. The textbook claim is that biases help; at this rank they may not, because a factor dimension can hold a constant and represent them itself.",
            **{"try": "run the same comparison at rank 1 and at rank 64. At "
                      "low rank the explicit biases should matter more, "
                      "because a single factor cannot hold a constant and a "
                      "taste at once. Does it come out that way, and if not, "
                      "what else could hold the constant?"}),
        code('''
Pb, Qb = als(d=best[0], biases=True)
Pn, Qn = als(d=best[0], biases=False)
with_b    = rmse(y_te, predict(Pb, Qb, u_te, i_te, biases=True))
without_b = rmse(y_te, predict(Pn, Qn, u_te, i_te, biases=False))
print(f"rank {best[0]}, with explicit biases      {with_b:.4f}")
print(f"rank {best[0]}, without                   {without_b:.4f}")
print(f"difference                          {abs(with_b - without_b):.4f}")
'''),
        prompt(
            label="the rotation nobody can see",
            input="a fitted P and Q, and a random invertible matrix M",
            output="the predictions from (PM, QM^-T), beside the originals",
            constraint="use a random M, not a permutation — the claim is about every invertible matrix, not a relabelling",
            check="assert the two prediction vectors agree to floating-point tolerance. This is why an individual latent factor cannot be interpreted.",
            **{"try": "make M orthogonal — the Q of a QR decomposition of a "
                      "random matrix. The predictions are still identical, "
                      "and now the factor norms are preserved as well. Which "
                      "of those two invariances would a paper's factor plot "
                      "have to defend itself against?"}),
        code('''
M = rng.normal(size=(best[0], best[0]))
Pm = Pb @ M
Qm = Qb @ np.linalg.inv(M).T
a = (Pb[u_te] * Qb[i_te]).sum(1)
b = (Pm[u_te] * Qm[i_te]).sum(1)
assert np.allclose(a, b, atol=1e-8)
print("predictions identical under P -> PM, Q -> QM^-T")
print(f"but the factors are not:  ||P - PM||_F = {np.linalg.norm(Pb - Pm):.1f}")
print("\\nSo a latent dimension is not a genre, and a factor plot is not evidence.")
'''),
    ]

    # ------------------------------------------------------------------ 5
    cells += [
        md("""
## 5 · Step 12: the same objective, one rating at a time

ALS solves each side exactly. SGD takes one rating at a time and steps both
vectors. Same objective, different route — and SGD is the one that suits
implicit data and streams, which is where Lecture 22 goes.
"""),
        prompt(
            label="⏱ 1 min — SGD, from the update rule",
            input="the training ratings, shuffled once per epoch",
            output="the factors, and the training RMSE per epoch",
            constraint="copy p_u BEFORE updating it — using the already-updated p_u in the q_i step is a different algorithm, it still converges, and nothing warns you",
            check="the training curve must fall monotonically. Compare the final test RMSE against ALS at the same rank: two routes to one objective should arrive in comparable places.",
            **{"try": "use Ps[uu] rather than the saved pu in the Qs update. "
                      "It still converges and the training curve still falls, "
                      "and it is a different algorithm. How far apart do the "
                      "two test RMSEs end up, and would you have caught the "
                      "substitution in a code review?"}),
        code('''
def sgd(d=32, epochs=15, lr=0.01, reg=0.05, seed=SEED):
    g = np.random.default_rng(seed)
    Ps = g.normal(0, 0.1, (n_u, d))
    Qs = g.normal(0, 0.1, (n_i, d))
    bus, bis = np.zeros(n_u), np.zeros(n_i)
    hist = []
    for _ in range(epochs):
        for n in g.permutation(len(y_tr)):
            uu, ii = u_tr[n], i_tr[n]
            e = y_tr[n] - (mu + bus[uu] + bis[ii] + Ps[uu] @ Qs[ii])
            bus[uu] += lr * (e - reg * bus[uu])
            bis[ii] += lr * (e - reg * bis[ii])
            pu = Ps[uu].copy()               # before the update, not after
            Ps[uu] += lr * (e * Qs[ii] - reg * pu)
            Qs[ii] += lr * (e * pu - reg * Qs[ii])
        hist.append(rmse(y_tr, np.clip(mu + bus[u_tr] + bis[i_tr]
                                       + (Ps[u_tr] * Qs[i_tr]).sum(1), 1, 5)))
    return Ps, Qs, bus, bis, hist

Ps, Qs, bus, bis, hist = sgd()
for e, v in enumerate(hist, 1):
    print(f"  epoch {e:>2}   train rmse {v:.4f}")

sgd_test = rmse(y_te, np.clip(mu + bus[u_te] + bis[i_te]
                              + (Ps[u_te] * Qs[i_te]).sum(1), 1, 5))
P32, Q32 = als(d=32)
print(f"\\nSGD rank 32   test rmse {sgd_test:.4f}")
print(f"ALS rank 32   test rmse {rmse(y_te, predict(P32, Q32, u_te, i_te)):.4f}")
'''),
    ]

    # ------------------------------------------------------------------ 6
    cells += [
        md("""
## 6 · The other family: item–item neighbourhoods

Predict from similar films. The columns are **bias-adjusted first** — without
that, every popular film is similar to every other popular film, because both
columns are full of fours. Same trap as term frequency without idf in Lecture 19.
"""),
        prompt(
            label="item–item cosine",
            input="the bias-adjusted training matrix",
            output="a film-by-film similarity matrix",
            constraint="subtract mu, b_u and b_i before taking the cosine, and zero the diagonal so a film is not its own neighbour",
            check="assert the diagonal is zero and the matrix is symmetric. A film that is its own nearest neighbour predicts itself perfectly and the RMSE looks wonderful.",
            **{"try": "skip the np.fill_diagonal and run the next cell. Every "
                      "film is now its own nearest neighbour at similarity 1, "
                      "and the RMSE looks superb. That is why this is an "
                      "assert and not a sentence of prose."}),
        code('''
R_tr = np.zeros((n_u, n_i), dtype=np.float32)
R_tr[u_tr, i_tr] = (y_tr - mu - bu[u_tr] - bi[i_tr]).astype(np.float32)

Rn = R_tr / (np.linalg.norm(R_tr, axis=0) + 1e-9)
S_ii = Rn.T @ Rn
np.fill_diagonal(S_ii, 0.0)
assert np.allclose(np.diag(S_ii), 0) and np.allclose(S_ii, S_ii.T, atol=1e-5)
print(f"similarity matrix {S_ii.shape}, diagonal zeroed")
'''),
        prompt(
            label="predict from the k nearest films",
            input="a neighbourhood size",
            output="test RMSE",
            constraint="use only neighbours this user actually rated, and only positive similarities — averaging in a negative weight inverts the contribution",
            check="fall back to the bias prediction when a user has rated none of the neighbours. Without the fallback those cases become zeros and the RMSE is meaningless.",
            **{"try": "drop the w > 0 condition and keep the negative "
                      "similarities. A film that is anti-correlated with this "
                      "one now pushes the prediction the wrong way in "
                      "proportion to how anti-correlated it is. Does the RMSE "
                      "rise or fall, and can you say which before you run it?"}),
        code('''
idx_all = np.argsort(-S_ii, axis=1)
for k in (10, 20, 50):
    idx = idx_all[:, :k]
    preds = np.empty(len(y_te))
    for n, (uu, ii) in enumerate(zip(u_te, i_te)):
        nb = idx[ii]
        w, v = S_ii[ii, nb], R_tr[uu, nb]
        m = (v != 0) & (w > 0)
        adj = float(w[m] @ v[m]) / float(w[m].sum()) if m.any() else 0.0
        preds[n] = mu + bu[uu] + bi[ii] + adj
    print(f"  item-item k={k:<3} rmse {rmse(y_te, np.clip(preds, 1, 5)):.4f}")
'''),
        md("""
## 7 · Implicit feedback has no negatives

Everything above assumed a number the user typed. Most systems have none — only
clicks, plays and purchases. A user watched film A; what does that say about
film B, which they did not watch? That they disliked it, or have not seen it, or
were never shown it. Only the first is a negative.

So an objective over observed entries alone is trivially satisfied by predicting
"yes" for everything, and the two ways out are to weight all cells by confidence,
or to model only **pairwise** preference (BPR). Lecture 22 takes the second route.
"""),
        prompt(
            label="the implicit view of this data",
            input="the ratings, binarised at 4",
            output="how many positives, and how many cells of unknown meaning",
            constraint="say 'cells of unknown meaning', not 'negatives' — the whole section is about refusing that word",
            check="the positive count should be a small fraction of the grid. That ratio is what makes sampling necessary.",
            **{"try": "binarise at 3 rather than 4. The positive count "
                      "roughly doubles and 'cells of unknown meaning' covers "
                      "less of the grid. That threshold is a definition of "
                      "what a positive is, and the next lecture evaluates "
                      "against whichever one you pick."}),
        code('''
pos = (r["rating"] >= 4).sum()
grid = n_u * n_i
print(f"explicit ratings           {len(r):,}")
print(f"of which positive (>= 4)   {pos:,}  ({100*pos/len(r):.1f}%)")
print(f"cells in the grid          {grid:,}")
print(f"cells with no interaction  {100*(1 - pos/grid):.2f}% — meaning unknown")
'''),
        md("""
## 8 · Everything, on one split

Every row below was scored on the same held-out ratings, from the same split,
with the same seed.
"""),
        prompt(
            label="the summary table",
            input="every result above",
            output="one row per model",
            constraint="include the models that lost, including both SVD fills",
            check="the bias model should be within striking distance of the factorisation. Knowing what fraction of your result is arithmetic is the point of fitting baselines first.",
            **{"try": "add the item-item model at k=10, which lost to k=50. A "
                      "summary table keeping only each family's best row "
                      "makes every family look as though it had one idea, and "
                      "hides how sensitive each was to a setting somebody "
                      "chose."}),
        code('''
rows = [
    ("global mean",              baselines["global mean"]),
    ("user mean",                baselines["user mean"]),
    ("item mean",                baselines["item mean"]),
    ("bias model",               baselines["bias model"]),
    ("rank-32 SVD, zero-filled", fills["filled with zeros"]),
    ("rank-32 SVD, mean-filled", fills["filled with the mean"]),
    ("item-item, k=50",          rmse(y_te, np.clip(preds, 1, 5))),
    (f"factorisation, d={best[0]}",  best[1]),
]
print(f"{'model':<26} {'rmse':>8}")
for name, v in rows:
    print(f"{name:<26} {v:>8.4f}")

span = baselines["global mean"] - best[1]
print()
print(f"the bias model closes {100*(baselines['global mean']-baselines['bias model'])/span:.1f}%"
      " of the distance from the global mean to the full model")
'''),
        md("""
## 9 · Five questions for a recommender result

1. **Rating prediction or ranking?** RMSE and NDCG answer different questions.
2. **What is the baseline?** The bias model is four lines and is often close.
3. **How was it split?** Randomly over ratings, or in time?
4. **What happened to the missing entries** — excluded, imputed, or weighted?
5. **Explicit or implicit**, and if implicit, where did the negatives come from?

---

**Next.** Lecture 22 keeps the model and changes the question: from "what would
they rate it" to "what should we show". Question 3 becomes a measurement, and it
is a large one.
"""),
    ]

    return cells
