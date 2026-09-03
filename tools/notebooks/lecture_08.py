#!/usr/bin/env python3
"""
Lecture 8 — Dimensionality reduction and unsupervised learning.
Olivetti faces, Géron Chapters 7–8.

Exports build() -> list[cell]; tools/make_notebooks.py wraps it.

DRAFT PATH. The final module is `lecture_08.py`; that name is still held by
old lecture 8 (ensembles, CoverType) while it is converted into new Lecture 7,
so this file carries a suffix until the integrator renames it. Because
`make_notebooks._discover()` keys modules on the number in the filename and
skips a number it has already seen, `lecture_08.py` wins and this file is
silently ignored by the build — which is the safe failure. Test it by calling
`build()` directly.

Merges the old lectures 9 and 10, which clustered 4,096-dimensional faces and
then repaired the cost. Here there is no Build/Fix arc: the cost of working in
4,096 dimensions is a stated property of the problem, measured in section 9 and
removed in section 14, and nothing is wrong on purpose.

EVERY quantity this notebook prints that also appears on a slide is computed
the way `tools/figures_app05.py` computes it — the same grid (k = 2..60), the
same `n_init=10`, the same seeds, the same twenty splits, and the same
*squared*-distance convention in the Johnson–Lindenstrauss measurement. The two
diverged in the old pair and the numbers on the slides were unreachable from
the notebook; keep them together.

Runs on CPU throughout: ~85 s measured at two threads on a 2026 laptop, and
three to five minutes on Colab's two cores. Anything that might exceed twenty
seconds on Colab carries a ⏱ marker and a wall-clock estimate.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from make_notebooks import code, header, md                            # noqa: E402
from _prompt import prompt                                             # noqa: E402


def build() -> list:
    cells = header(8, "Dimensionality reduction and unsupervised learning",
                   "", "Chapters 7–8")

    cells += [
        md("""
Four hundred photographs, forty people, no labels — except forty the project
paid an annotator for. Two chapters in one notebook, because the second one
does not work on this data until the first has been applied to it.

The whole notebook runs on free CPU. It takes about **90 seconds** on a modern
laptop with the BLAS threads capped at two, and **three to five minutes** on
Colab's two cores. The slowest cell is the 4,096-dimensional model-selection
sweep, and it says so.

Two rules hold throughout, and both are on the slides:

- the full `y` is **not** available to the method. It is used twice, both times
  marked **AUDIT**: as the forty labels we paid for, and to find out afterwards
  what those forty could not have told us;
- the number that *chooses* a model may not be the number that *reports* it.
  With no labels anywhere in sight, that rule still applies — section 13
  measures what it is worth here.
"""),

        # ------------------------------------------------------- 1 · setup
        md("## 1 · Setup"),
        prompt(
            label="setup, and one line that is not hygiene",
            input="nothing",
            output="library versions, one seed, and a thread cap",
            constraint="cap the BLAS threads BEFORE importing numpy — the "
                       "environment variables are read at import time, and "
                       "setting them afterwards does nothing at all",
            check="this notebook reports several durations and one speed-up. A "
                  "duration is only repeatable if what it depends on is "
                  "controlled; with the default 'all cores' the number is "
                  "about the machine's mood, not about the method",
            **{"try": "raise the cap to 8 and re-run section 9. Every absolute "
                      "second changes; the 4,096-to-123 ratio in section 14 "
                      "barely does. Which of the two belongs on a slide?"}),
        code('''
# --- setup -------------------------------------------------------------------
# Not examinable: engineering hygiene. The thread limit is here so a timing you
# measure is a timing you can repeat, and it must run before numpy is imported.
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import sys, time
import numpy as np
import sklearn
import matplotlib.pyplot as plt

print(f"python       {sys.version.split()[0]}")
print(f"scikit-learn {sklearn.__version__}")
print(f"numpy        {np.__version__}")

RANDOM_STATE = 42          # every split, every model, every shuffle
K_MAX = 60                 # the sweep runs k = 2 .. K_MAX
N_INIT = 10                # restarts of Lloyd's algorithm at every k
N_SEEDS = 20               # never quote a single-seed number
'''),
        prompt(
            label="every import, in one place",
            input="nothing",
            output="every name used below, imported once",
            constraint="no import anywhere after this cell, so the notebook "
                       "does not depend on a previous one still being in memory",
            check="Runtime → Restart, then run the setup cell and this one "
                  "alone. If anything below raises NameError, that import "
                  "belongs here",
            **{"try": "restart the runtime and run the LAST cell first. It "
                      "fails. A notebook that only runs top to bottom is the "
                      "only kind you can trust."}),
        code('''
from sklearn.cluster import DBSCAN, KMeans
from sklearn.datasets import fetch_olivetti_faces
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (adjusted_rand_score, pairwise_distances,
                             silhouette_samples, silhouette_score)
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.random_projection import (GaussianRandomProjection,
                                       johnson_lindenstrauss_min_dim)

print("imports ok")
'''),

        # -------------------------------------------------------- 2 · data
        md("""
## 2 · The corpus

An institution has a photographic archive. Nobody knows how many distinct
people are in it, and nobody has the budget to look at every picture. The task
is to **group the photographs by identity**, so a human can name a *group*
rather than a photograph.

The constraint that decides everything: **labelling is expensive**. We may pay
an annotator for forty labels. That is 10% of the corpus.
"""),
        prompt(
            label="the corpus",
            input="Olivetti faces",
            output="400 photographs as 4,096-dimensional vectors, plus the "
                   "64×64 images",
            constraint="`shuffle=False` — the ten photographs of each person "
                       "stay adjacent, which every montage below relies on",
            check="assert the shape, the pixel range, forty people, and exactly "
                  "ten photographs each. Assert the RANGE in particular: "
                  "Olivetti arrives in [0,1] and many face datasets arrive in "
                  "[0,255], and every distance in this notebook is a factor of "
                  "255 different if you assume wrong",
            **{"try": "`shuffle=True`. Every assert still passes and every "
                      "montage below becomes meaningless. Which assert would "
                      "have caught it, and why is there no such assert?"}),
        code('''
faces = fetch_olivetti_faces(shuffle=False)      # ~4 MB, a few seconds
X, y, images = faces.data, faces.target, faces.images

assert X.shape == (400, 4096), f"unexpected shape {X.shape}"
assert images.shape == (400, 64, 64)
assert X.min() >= 0.0 and X.max() <= 1.0, "pixels are not in [0, 1]"
assert len(np.unique(y)) == 40
assert np.bincount(y).min() == np.bincount(y).max() == 10

print(f"{len(X)} photographs of {len(np.unique(y))} people, "
      f"{np.bincount(y)[0]} each")
print(f"each is {images.shape[1]}x{images.shape[2]} = {X.shape[1]} features")
print(f"the whole corpus is {X.nbytes / 1e6:.4f} MB in memory")
print(f"pairs of photographs: {len(X) * (len(X) - 1) // 2:,}")
'''),
        md("""
`y` exists because Olivetti is a benchmark. **In the brief it does not.** From
here on the only labels the method may use are the forty we pay for; every
other use of `y` is flagged **AUDIT** and is a result the project itself could
not have computed.
"""),
        prompt(
            label="the split, used from section 6 onwards",
            input="the corpus and the identities",
            output="280 training and 120 held-out photographs, stratified by "
                   "identity",
            constraint="clustering needs no split — there is no held-out "
                       "quantity to protect. Reconstruction error, label "
                       "propagation and the leak in section 20 all do, so the "
                       "split is made once, here, with a fixed seed",
            check="assert the sizes AND the stratification: with forty people "
                  "and 120 held-out rows, an unstratified split can hand a "
                  "person zero test photographs, and the accuracy is then "
                  "measured on 39 people while claiming 40",
            **{"try": "drop `stratify=y` and re-run this cell. Which of the "
                      "three asserts fails first?"}),
        code('''
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=RANDOM_STATE)

assert len(X_tr) == 280 and len(X_te) == 120
assert np.bincount(y_tr).min() == 7 and np.bincount(y_te).min() == 3
print(f"{len(X_tr)} train, {len(X_te)} test, "
      f"{np.bincount(y_tr).min()}+{np.bincount(y_te).min()} per person")
'''),

        # ------------------------------------------------ 3 · look at them
        md("""
## 3 · It is a face dataset. Look at faces
"""),
        prompt(
            label="look at the data",
            input="one photograph of each of the forty people, then ten of one "
                  "person and ten of another",
            output="tiled images, ten to a row",
            constraint="tile into ONE array rather than making forty subplots "
                       "— forty axes at this size is slow and each carries its "
                       "own margins",
            check="`vmin=0, vmax=1` on the imshow. Without it matplotlib "
                  "rescales each montage to its own range, and two montages "
                  "become incomparable in brightness for no stated reason",
            **{"try": "show two people rather than one in the second montage — "
                      "it already does. Cover the second row and ask whether "
                      "you could tell the two apart from the first row alone."}),
        code('''
def montage(ax, ims, ncol, gap=2):
    """Tile square images into one array, so the figure holds one image."""
    ims = np.asarray(ims)
    n, h, w = ims.shape
    nrow = int(np.ceil(n / ncol))
    canvas = np.ones((nrow * (h + gap) - gap, ncol * (w + gap) - gap))
    for i, im in enumerate(ims):
        r, c = divmod(i, ncol)
        canvas[r * (h + gap):r * (h + gap) + h,
               c * (w + gap):c * (w + gap) + w] = im
    ax.imshow(canvas, cmap="gray", vmin=0, vmax=1)
    ax.set_xticks([]); ax.set_yticks([])
    return ax

fig, ax = plt.subplots(figsize=(11, 5))
montage(ax, np.array([images[y == p][0] for p in range(40)]), ncol=10)
ax.set_title("one photograph of each of the 40 people")
plt.show()

fig, ax = plt.subplots(figsize=(11, 2.6))
montage(ax, np.concatenate([images[y == 0], images[y == 22]]), ncol=10)
ax.set_title("ten photographs of one person, then ten of another")
plt.show()
'''),
        md("""
Glasses on and off, lighting from either side, eyes shut, head turned. Any
method that groups those ten together has to be insensitive to all of it — and
still separate them from the other 390.
"""),

        # ----------------------------------------------- 4 · is this hard?
        md("""
## 4 · Is this possible at all? Measure before assuming

A photograph is a point in $\\mathbb{R}^{4096}$, and k-means groups points that
are close in Euclidean distance. So one measurement decides whether the hour is
worth spending: **are two photographs of the same person closer than two
photographs of different people?**
"""),
        prompt(
            label="AUDIT — the distance structure",
            input="all 79,800 pairs of photographs, and the hidden identities",
            output="within-person against between-person distances, and the "
                   "fraction of same-person pairs that lie beyond the median "
                   "different-person pair",
            constraint="this uses the hidden labels and is marked AUDIT — the "
                       "stakeholder could not afford it, and it is here to tell "
                       "US whether the representation can work at all",
            check="assert 1,800 within-person pairs, and that the two counts "
                  "sum to 400·399/2 = 79,800. Ten photographs of each of forty "
                  "people gives 40 × C(10,2) = 40 × 45 = 1,800, and you can "
                  "write that down before running the cell",
            **{"try": "report the mirror statistic as well — different-person "
                      "pairs closer than the median same-person pair. It is "
                      "smaller, and quoting only whichever direction flatters "
                      "your conclusion is the failure this cell exists to "
                      "prevent."}),
        code('''
D = pairwise_distances(X)
iu = np.triu_indices(len(X), k=1)
same = y[iu[0]] == y[iu[1]]                      # AUDIT — the hidden labels
within, between = D[iu][same], D[iu][~same]

assert len(within) == 40 * (10 * 9 // 2) == 1800
assert len(within) + len(between) == 400 * 399 // 2 == 79800

print(f"same person      {len(within):>6,} pairs   "
      f"mean {within.mean():.4f}   median {np.median(within):.4f}")
print(f"different people {len(between):>6,} pairs   "
      f"mean {between.mean():.4f}   median {np.median(between):.4f}")

overlap = (within > np.median(between)).mean()
overlap_rev = (between < np.median(within)).mean()
print(f"\\n{100 * overlap:.4f}% of same-person pairs lie beyond the median "
      f"different-person pair")
print(f"{100 * overlap_rev:.4f}% of different-person pairs lie inside the "
      f"median same-person pair")

fig, ax = plt.subplots(figsize=(9, 3))
bins = np.linspace(0, max(between.max(), within.max()), 70)
ax.hist(between, bins=bins, density=True, color="#b0bcc7", label="different")
ax.hist(within, bins=bins, density=True, color="#c0392b", alpha=0.7, label="same")
ax.axvline(np.median(between), color="#0b3d62", ls="--", lw=2)
ax.set_xlabel("Euclidean distance, raw pixels"); ax.legend(); plt.show()
'''),
        md("""
The two distributions overlap over most of their range. Pixel distance is
*partly* identity and partly lighting and pose, and that is the ceiling on what
any distance-based method can do in this representation. It is a property of
the representation, not of the algorithm — which is why the first half of this
notebook is about changing the representation.
"""),

        # ------------------------------------------- 5 · the forty labels
        md("""
## 5 · The forty labels

The annotator is handed forty photographs, chosen at random, and asked which of
them show the same person. That is all the supervision the project has.
"""),
        prompt(
            label="the forty labels, and all the supervision there is",
            input="the corpus",
            output="forty randomly chosen indices and the annotator's answers",
            constraint="draw them from a generator created HERE rather than "
                       "from a shared one — a `default_rng` is stateful, so an "
                       "extra call inserted above this line would silently "
                       "change which forty you get, and every ARI below with it",
            check="assert forty indices and forty DISTINCT ones. Then print how "
                  "many pairs among the forty are the same person: C(40,2) = "
                  "780 pairs, and with ten photographs per person you should "
                  "expect roughly 780 × 9/399 ≈ 18 of them",
            **{"try": "change the seed to 43. The ARI numbers everywhere below "
                      "move by a few hundredths. That is the width of the "
                      "interval on a metric measured on forty photographs."}),
        code('''
# a generator of its own, so no earlier cell can shift this draw
audit = np.sort(np.random.default_rng(RANDOM_STATE).choice(
    400, size=40, replace=False))
y_audit = y[audit]                               # the annotator's answers

assert len(audit) == 40 and len(np.unique(audit)) == 40

same_pairs = int(sum(y_audit[i] == y_audit[j]
                     for i in range(40) for j in range(i + 1, 40)))
print(f"{40 * 39 // 2} pairs among the labelled forty, "
      f"{same_pairs} of which are the same person")
print(f"distinct people seen at least once: {len(np.unique(y_audit))}")
print(f"fraction of the corpus labelled: {100 * 40 / len(X):.1f}%")
'''),

        # ------------------------------------------------- 6 · PCA by hand
        md("""
## 6 · The mathematics, verified

Centre the data and stack the faces as the rows of $\\mathbf{X}$. The singular
value decomposition writes

$$\\tilde{\\mathbf{X}} = \\mathbf{U}\\,\\boldsymbol\\Sigma\\,\\mathbf{V}^{\\mathsf T},$$

with $\\mathbf{U}$ and $\\mathbf{V}$ orthogonal and $\\boldsymbol\\Sigma$
diagonal with non-negative decreasing entries. The columns of $\\mathbf{V}$ are
the **principal components**, and they are the eigenvectors of
$\\mathbf{C} = \\tilde{\\mathbf{X}}^{\\mathsf T}\\tilde{\\mathbf{X}}$ with
eigenvalues $\\sigma_j^2$ — the matrix Lecture 2's normal equation needed to be
invertible.

Compute it by hand and check that scikit-learn agrees. A full SVD of a
400 × 4,096 matrix takes a couple of seconds.
"""),
        prompt(
            label="PCA by hand, then checked",
            input="the centred data matrix",
            output="the SVD, compared against scikit-learn's PCA",
            constraint="CENTRE before the SVD — `PCA` centres internally, and "
                       "an uncentred SVD gives a first component that is "
                       "essentially the mean face and agrees with nothing",
            check="assert the components agree to 1e-4, comparing ABSOLUTE "
                  "values: singular vectors are defined only up to a sign, so "
                  "an exact comparison fails on a correct implementation about "
                  "half the time",
            **{"try": "drop the `- X.mean(axis=0)` and re-run. The assert fires — "
                      "then look at the per-component differences before deciding "
                      "why. ALL FIVE disagree, not just the first: without "
                      "centring the leading singular vector points at the mean "
                      "face, and every later one is orthogonal to that rather "
                      "than to the centred data's directions."}),
        code('''
X_centred = X - X.mean(axis=0)
# full_matrices=False: with 400 rows and 4,096 columns the full U would be
# 4096 x 4096 — 134 MB of mostly zero — and it is never needed.
U, S, Vt = np.linalg.svd(X_centred, full_matrices=False)

pca = PCA(random_state=RANDOM_STATE).fit(X)

assert Vt.shape == (400, 4096), Vt.shape
agree = np.abs(np.abs(Vt[:5]) - np.abs(pca.components_[:5])).max()
print(f"largest disagreement over the first five components: {agree:.2e}")
assert agree < 1e-4

# the explained variance ratio IS the normalised squared singular values
evr = S ** 2 / (S ** 2).sum()
print(f"largest disagreement over the variance ratios:      "
      f"{np.abs(evr - pca.explained_variance_ratio_).max():.2e}")
print(f"non-zero components available: {pca.n_components_}")
'''),
        md("""
### Step 7 of the derivation, to machine precision

Keep the first $d$ components and project. The derivation ends at

$$\\min_{\\mathbf{W}} E(\\mathbf{W}) \\;=\\; \\bigl\\lVert \\tilde{\\mathbf{X}} - \\tilde{\\mathbf{X}}_d \\bigr\\rVert_F^2 \\;=\\; \\sum_{j>d}\\sigma_j^2 .$$

That is an identity, so assert it.
"""),
        prompt(
            label="Eckart-Young, to machine precision",
            input="the rank-100 truncation",
            output="the Frobenius error of the truncation, beside the tail of "
                   "the spectrum",
            constraint="rebuild $\\mathbf{X}_d$ from the truncated factors, not "
                       "from `inverse_transform` — the point is that the two "
                       "sides are computed by different routes and still agree",
            check="assert the RELATIVE difference is below 1e-5. 'Small' is "
                  "meaningless until you divide by something: the two sides are "
                  "each of order 10^0 here, and would be of order 10^6 on "
                  "pixel values in [0, 255]",
            **{"try": "change d to 399. Both sides go to almost zero and the "
                      "relative test still passes — which is the point of "
                      "dividing rather than subtracting."}),
        code('''
d = 100
Xd = U[:, :d] * S[:d] @ Vt[:d]                   # the rank-d truncation
lhs = ((X_centred - Xd) ** 2).sum()
rhs = (S[d:] ** 2).sum()
print(f"||X - X_d||^2_F = {lhs:.6f}")
print(f"sum_(j>d) s_j^2 = {rhs:.6f}")
print(f"relative difference {abs(lhs - rhs) / rhs:.2e}")
assert abs(lhs - rhs) / rhs < 1e-5
'''),

        # ---------------------------------------------------- 7 · choosing d
        md("""
## 7 · Choosing $d$ from the explained variance ratio

By step 7, the share of squared error *removed* by keeping $d$ components is
exactly $\\sum_{j \\le d} r_j$ with $r_j = \\sigma_j^2 / \\sum_i \\sigma_i^2$.
The ratio is not a heuristic; it is the theorem, normalised.
"""),
        prompt(
            label="how many components do we need",
            input="the explained variance ratios",
            output="the counts reaching 90%, 95% and 99%, and the cumulative "
                   "curve",
            constraint="`searchsorted` plus one — the index at which the "
                       "cumulative sum first exceeds the threshold is one less "
                       "than the number of components kept",
            check="print the reduction factor beside the count. '95% needs 123 "
                  "components, 33x fewer than 4,096' is the sentence; the raw "
                  "count alone is not",
            **{"try": "read off d99 and compare it with d95. The last four "
                      "points of variance cost more components than the first "
                      "ninety-five. That shape is what the scree plot is."}),
        code('''
cum = np.cumsum(pca.explained_variance_ratio_)
d90 = int(np.searchsorted(cum, 0.90) + 1)
d95 = int(np.searchsorted(cum, 0.95) + 1)
d99 = int(np.searchsorted(cum, 0.99) + 1)

print(f"component 1 alone explains {100 * pca.explained_variance_ratio_[0]:.4f}%")
print(f"component 2 explains       {100 * pca.explained_variance_ratio_[1]:.4f}%")
print(f"the first 10 explain       {100 * cum[9]:.4f}%")
print(f"the first 50 explain       {100 * cum[49]:.4f}%")
print(f"\\n90% of the variance needs {d90} components")
print(f"95% needs {d95}  ({4096 / d95:.4f}x fewer numbers than 4096)")
print(f"99% needs {d99}")

fig, ax = plt.subplots(figsize=(8, 3))
ax.plot(range(1, len(cum) + 1), 100 * cum, color="#0b3d62", lw=2)
ax.axhline(95, color="#14663a", ls="--", lw=2)
ax.axvline(d95, color="#14663a", ls=":", lw=2)
ax.set_xlabel("components kept"); ax.set_ylabel("cumulative variance, %")
plt.show()
'''),
        prompt(
            label="the eigenfaces",
            input="the mean face and the first fifteen components",
            output="one tiled image",
            constraint="rescale each component to [0,1] INDIVIDUALLY — "
                       "components have negative entries and no common scale, "
                       "and `vmin=0, vmax=1` would clip half of every one of "
                       "them to black",
            check="show the mean face first. The components are directions "
                  "away from it, and without it in the frame they are hard to "
                  "read as faces at all",
            **{"try": "look at components 1 to 4 and say what varies across "
                      "them. It is the direction of the lamp, not the person "
                      "— which is the failure condition of section 4 showing "
                      "up in the basis itself."}),
        code('''
def rescale(v):
    v = v.reshape(64, 64)
    return (v - v.min()) / (v.max() - v.min())

tiles = [X.mean(axis=0).reshape(64, 64)] + [rescale(c) for c in pca.components_[:15]]
fig, ax = plt.subplots(figsize=(11, 3))
montage(ax, np.array(tiles), ncol=8)
ax.set_title("the mean face, then the first 15 principal components")
plt.show()
'''),
        prompt(
            label="a face with components taken away",
            input="one held-out face, reconstructed at eight dimensionalities",
            output="the sequence from 1 component to the original 4,096",
            constraint="fit each PCA on the TRAINING faces only and reconstruct "
                       "a held-out one — reconstructing a face the subspace was "
                       "fitted on flatters every column of this figure",
            check="the identity should become recognisable well before d95 = "
                  "123. If it does not, you are reconstructing the wrong face "
                  "or you forgot `inverse_transform`",
            **{"try": "reconstruct `X_tr[0]` instead of `X_te[0]`. It is "
                      "sharper at every d, and the difference is exactly the "
                      "reason the constraint above exists."}),
        code('''
face = X_te[0]
tiles, names = [], []
for k in (1, 2, 5, 10, 25, 50, 100, 200):
    p = PCA(n_components=k, random_state=RANDOM_STATE).fit(X_tr)
    tiles.append(p.inverse_transform(p.transform(face[None]))[0].reshape(64, 64))
    names.append(str(k))
tiles.append(face.reshape(64, 64)); names.append("4096")

fig, ax = plt.subplots(figsize=(11, 1.8))
montage(ax, np.array(tiles), ncol=9)
ax.set_title("  ".join(f"{n:>6s}" for n in names))
plt.show()
'''),

        # ------------------------------------------------------- 8 · JL
        md("""
## 8 · Johnson–Lindenstrauss

> For any $0 < \\varepsilon < 1$ and any set of $m$ points in
> $\\mathbb{R}^{n}$ there is a linear map into $\\mathbb{R}^{d}$ with
> $$d \\;\\ge\\; \\frac{4\\log m}{\\varepsilon^{2}/2 - \\varepsilon^{3}/3}$$
> preserving every pairwise **squared** distance to within a factor
> $1 \\pm \\varepsilon$.

**Read the formula again and say what is missing.** $m$ is there.
$\\varepsilon$ is there. $n$ — the dimension you start in — is not. Four
hundred points need the same target dimension whether they live in 4,096
dimensions or in four million.
"""),
        prompt(
            label="the bound, evaluated",
            input="400 points and a million points, at four tolerances",
            output="the guaranteed target dimension in each case",
            constraint="print the two population sizes side by side, so the "
                       "log m growth is visible rather than asserted",
            check="a factor of 2,500 more points should cost about TWICE the "
                  "dimension, not 2,500 times. That ratio is what log m means, "
                  "and you can predict it before running the cell",
            **{"try": "add eps=0.05. The bound exceeds 4,096 by a wide margin "
                      "— on 400 points, for a tight enough tolerance, the "
                      "theorem asks for more dimensions than you started with."}),
        code('''
for eps in (0.1, 0.2, 0.3, 0.5):
    print(f"eps={eps}:  400 points -> "
          f"{johnson_lindenstrauss_min_dim(400, eps=eps):>7,}"
          f"   1,000,000 points -> "
          f"{johnson_lindenstrauss_min_dim(1_000_000, eps=eps):>7,}")
print(f"\\nthe dimension we actually have: {X.shape[1]}")
'''),
        md("""
At $\\varepsilon = 0.1$ the bound for our 400 faces is **larger than 4,096**.
The theorem is a worst-case guarantee over all possible point sets; a
photographic archive is not the worst case. So measure what actually happens.

⏱ **10–40 s on Colab** — seven target dimensions, five random draws each.
"""),
        prompt(
            label="⏱ what actually happens to our distances",
            input="all 79,800 pairwise distances, before and after projection, "
                  "at seven target dimensions",
            output="the worst, 95th-percentile and mean relative distortion",
            constraint="measure the distortion of SQUARED distances — that is "
                       "what the lemma bounds. Comparing plain distance ratios "
                       "against the same epsilon understates the distortion by "
                       "roughly a factor of two, which beside the theorem is a "
                       "lie by a factor of two",
            check="report the WORST pair as well as the percentile: the theorem "
                  "bounds the worst case, so a percentile alone does not test "
                  "it. At d = 1,382 — the bound's own answer for eps = 0.2 — "
                  "the worst pair should come in under 0.2",
            **{"try": "drop the `** 2` from both distance arrays. Every number "
                      "roughly halves and the bound suddenly looks generous. "
                      "That is the factor-of-two error, made visible."}),
        code('''
iu = np.triu_indices(len(X), k=1)
D0sq = pairwise_distances(X)[iu] ** 2            # SQUARED — what the lemma bounds
print(f"{len(D0sq):,} pairwise distances\\n")

for dd in (50, 100, 200, 400, 800, 1382, 1600):
    worst, p95, mean = [], [], []
    for seed in range(5):                        # one draw is one sample
        g = GaussianRandomProjection(n_components=dd, random_state=seed)
        D1sq = pairwise_distances(g.fit_transform(X))[iu] ** 2
        rel = np.abs(D1sq / D0sq - 1)
        worst.append(rel.max()); p95.append(np.quantile(rel, 0.95))
        mean.append(rel.mean())
    print(f"d={dd:5d}   worst pair {np.mean(worst):.4f}   "
          f"95th pct {np.mean(p95):.4f}   mean {np.mean(mean):.4f}", flush=True)
'''),
        md("""
At the bound's own $d = 1{,}382$ the worst of the 79,800 pairs is inside the
0.2 it promises — and not by much. The bound is *sufficient*, not *necessary*,
it is loose but not absurdly so, and the reason to teach it is not the
constant. It is that the constant does not contain 4,096.
"""),

        # ------------------------------------------- 9 · the four reducers
        md("""
## 9 · Four ways to reduce, timed and scored

Randomised PCA approximates the top components without the full SVD.
Incremental PCA never holds the whole matrix. Random projection does not look
at the data at all.
"""),
        prompt(
            label="four ways to reduce, timed and scored",
            input="full SVD, randomised SVD, incremental PCA, random "
                  "projection, all at d = 123",
            output="the fit time and the held-out reconstruction error for each",
            constraint="take the BEST of three timings, not one — a single "
                       "wall-clock on a shared machine measures the machine. "
                       "And `batch_size` at least `n_components` for "
                       "IncrementalPCA: it fits each batch, and a batch "
                       "narrower than the target dimensionality cannot "
                       "determine it",
            check="the three PCA variants should land on the same "
                  "reconstruction error to three significant figures — they "
                  "are three routes to the same subspace. If randomised "
                  "disagrees, `n_oversamples` is too small",
            **{"try": "set `batch_size=100` on the IncrementalPCA, below the 123 "
                      "components asked for. It raises — and the message is "
                      "'Number of input features has changed from 100 to 123', "
                      "which is about neither the batch size nor n_components as "
                      "you set them. An error naming the wrong constraint costs "
                      "more than no error, because it sends you to the wrong "
                      "line."}),
        code('''
D95 = d95            # the dimension every experiment below runs in

def timeit(make, repeats=3):
    """Best of three: one wall-clock on a shared machine measures the machine."""
    best, obj = np.inf, None
    for _ in range(repeats):
        t0 = time.perf_counter()
        obj = make()
        best = min(best, time.perf_counter() - t0)
    return best, obj

def pca_err(p):
    return float(((p.inverse_transform(p.transform(X_te)) - X_te) ** 2).mean())

batch = int(max(D95 + 10, 70))
rows = []
t, p = timeit(lambda: PCA(D95, svd_solver="full",
                          random_state=RANDOM_STATE).fit(X_tr))
rows.append(("PCA, full SVD", t, pca_err(p)))
t, p = timeit(lambda: PCA(D95, svd_solver="randomized",
                          random_state=RANDOM_STATE).fit(X_tr))
rows.append(("PCA, randomised", t, pca_err(p)))
t, p = timeit(lambda: IncrementalPCA(D95, batch_size=batch).fit(X_tr))
rows.append((f"Incremental PCA, {len(X_tr) // batch + 1} batches", t, pca_err(p)))

t, g = timeit(lambda: GaussianRandomProjection(
    D95, random_state=RANDOM_STATE).fit(X_tr))
back = np.linalg.pinv(g.components_.T)           # least-squares re-embedding
rows.append(("Random projection", t,
             float(((g.transform(X_te) @ back - X_te) ** 2).mean())))

for name, t, err in rows:
    print(f"{name:30s} {1000 * t:8.4f} ms   held-out error {err:.7f}")
print(f"\\nrandom projection is {rows[3][2] / rows[0][2]:.2f}x worse than "
      f"full PCA at the same d")
print(f"a random {D95}-dimensional subspace of R^4096 keeps about "
      f"{100 * D95 / 4096:.4f}% of a generic vector's squared norm")
'''),
        prompt(
            label="the same comparison, across every d",
            input="PCA and Gaussian random projection at eight values of d",
            output="held-out squared reconstruction error for each",
            constraint="average the random projection over five draws — a "
                       "single random matrix is one sample from the "
                       "distribution the comparison is about",
            check="PCA at d = 2 should already beat random projection at d = "
                  "279. If it does not, the pseudo-inverse re-embedding is "
                  "wrong: `pinv(components_.T)`, not `components_`",
            **{"try": "print `rq_rp` beside `X_te.var()`. The random-projection "
                      "error is of the same order as simply predicting the mean "
                      "face — which is what 'preserves distances, not "
                      "positions' costs."}),
        code('''
ds = [2, 5, 10, 25, 50, 100, D95, 279]
rq_pca, rq_rp = [], []
for dd in ds:
    p = PCA(n_components=dd, random_state=RANDOM_STATE).fit(X_tr)
    rq_pca.append(float(((p.inverse_transform(p.transform(X_te)) - X_te) ** 2).mean()))
    errs = []
    for seed in range(5):
        g = GaussianRandomProjection(n_components=dd, random_state=seed).fit(X_tr)
        bk = np.linalg.pinv(g.components_.T)
        errs.append(float(((g.transform(X_te) @ bk - X_te) ** 2).mean()))
    rq_rp.append(float(np.mean(errs)))
    print(f"d={dd:4d}   PCA {rq_pca[-1]:.7f}   random projection {rq_rp[-1]:.7f}")

i95 = ds.index(D95)
print(f"\\nat d={D95}: random projection is "
      f"{rq_rp[i95] / rq_pca[i95]:.4f}x the PCA error")
print(f"variance of the held-out faces, for scale: {X_te.var():.7f}")

fig, ax = plt.subplots(figsize=(8, 3))
ax.loglog(ds, rq_pca, "o-", color="#0b3d62", lw=2, label="PCA")
ax.loglog(ds, rq_rp, "s--", color="#c0392b", lw=2, label="Gaussian random projection")
ax.set_xlabel("components kept, d"); ax.set_ylabel("held-out squared error")
ax.legend(); plt.show()
'''),

        # ------------------------------------------ 10 · the raw sweep
        md("""
## 10 · Clustering in 4,096 dimensions, and what it costs

k-means at every $k$ from 2 to 60, ten restarts each, scored by silhouette.
Both cost models are linear in the number of features: Lloyd's algorithm is
$O(\\text{iterations} \\times m\\,k\\,n)$ and the silhouette is $O(m^2 n)$.

⏱ **45 s on a fast laptop, 2–4 minutes on Colab's two cores.** This is the
slowest cell in the notebook, and the number it prints is the one section 14
divides into.
"""),
        prompt(
            label="⏱ the sweep in raw pixels",
            input="the 400 × 4,096 matrix",
            output="inertia, silhouette, ARI on the forty and ARI on all 400 "
                   "at every k from 2 to 60, plus the wall clock split between "
                   "fitting and scoring",
            constraint="record the two timings SEPARATELY — they scale "
                       "differently in m, and the whole argument of section 14 "
                       "rests on both being linear in n",
            check="count the steps at which inertia RISES. For the optimal "
                  "partition it cannot: a finer partition cannot have larger "
                  "inertia, so the true J(k) is non-increasing and J(m) = 0. "
                  "Lloyd's algorithm only reaches a local minimum, so a rise "
                  "means the restarts at that k landed worse than at k−1. "
                  "Predict that the count is small and every rise is tiny "
                  "against the curve's range — then check it",
            **{"try": "drop `n_init` to 1. The sweep is ten times faster and "
                      "the number of rising steps jumps. That count is what "
                      "the restarts were buying."}),
        code('''
ks = list(range(2, K_MAX + 1))
inertia, sil, ari_audit, ari_full, labels = [], [], [], [], {}
t_fit = t_sil = 0.0

for k in ks:
    t0 = time.perf_counter()
    km = KMeans(n_clusters=k, n_init=N_INIT, random_state=RANDOM_STATE).fit(X)
    t_fit += time.perf_counter() - t0
    t0 = time.perf_counter()
    s = silhouette_score(X, km.labels_)
    t_sil += time.perf_counter() - t0
    inertia.append(float(km.inertia_)); sil.append(float(s))
    ari_audit.append(float(adjusted_rand_score(y_audit, km.labels_[audit])))
    ari_full.append(float(adjusted_rand_score(y, km.labels_)))   # AUDIT
    if k in (2, 10, 40, 60):
        labels[k] = km.labels_.copy()

t_total = t_fit + t_sil

# J(k) is non-increasing for the OPTIMAL partition; Lloyd's algorithm only
# reaches a local minimum, so a rise is the restarts at that k landing worse
# than at k-1. Report it rather than asserting it away — asserting a property
# the algorithm does not guarantee is how a correct notebook stops running.
steps = np.diff(inertia)
rises = int((steps > 0).sum())
span = inertia[0] - inertia[-1]
print(f"inertia rises at {rises} of {len(steps)} steps "
      f"(largest rise {max(steps.max(), 0.0):.4f}, "
      f"{100 * max(steps.max(), 0.0) / span:.4f}% of the curve's range)")

print(f"{(K_MAX - 1) * N_INIT} runs of Lloyd's algorithm: {t_fit:.4f} s")
print(f"{K_MAX - 1} silhouette scores:            {t_sil:.4f} s")
print(f"one sweep, total:                {t_total:.4f} s "
      f"({t_total / 60:.4f} minutes)")
'''),
        prompt(
            label="the elbow, and the kneedle rule",
            input="inertia against k",
            output="the curve, and the k the kneedle rule picks beside the truth",
            constraint="rescale BOTH axes to [0,1] before applying the rule — "
                       "it is about the furthest point below the chord, and "
                       "that is meaningless while one axis runs to 60 and the "
                       "other to tens of thousands",
            check="inertia at k = 40 divided by inertia at k = 2 tells you how "
                  "much of the spread is still unexplained at the true k. If "
                  "the elbow were real, that fraction would be small. Predict "
                  "whether it is before running",
            **{"try": "run the rule on k = 2..20 only. It picks a different k. "
                      "A selection rule whose answer depends on where you "
                      "stopped the sweep is not a selection rule."}),
        code('''
ka = np.array(ks, float); ia = np.array(inertia, float)
kn = (ka - ka.min()) / (ka.max() - ka.min())
inn = (ia - ia.min()) / (ia.max() - ia.min())
drop = (1 - kn) - inn
i40 = ks.index(40)

print(f"the kneedle rule picks k = {int(ka[int(np.argmax(drop))])}")
print(f"the truth, which we are not supposed to know, is k = 40")
print(f"inertia at k=2  {inertia[0]:.4f}")
print(f"inertia at k=40 {inertia[i40]:.4f}   "
      f"({100 * inertia[i40] / inertia[0]:.4f}% of it still there)")
print(f"inertia at k=60 {inertia[-1]:.4f}")

fig, ax = plt.subplots(figsize=(8, 3))
ax.plot(ks, inertia, color="#0b3d62", lw=2)
ax.axvline(40, color="#14663a", ls="--", lw=2)
ax.set_xlabel("k"); ax.set_ylabel("inertia"); ax.set_title("find the elbow")
plt.show()
'''),

        # ------------------------------------------- 11 · anchor and choice
        md("""
## 11 · The trivial baseline, and what the silhouette chose

*A metric with nothing to compare it to is decoration.* Neither the silhouette
nor the ARI has a natural scale, so before reading either one, measure what
**no method at all** scores: throw every face into a uniformly random group,
twenty times. Eighty silhouettes, each needing the full 400 × 400 distance
matrix in 4,096 dimensions — a few seconds.
"""),
        prompt(
            label="what nothing scores",
            input="uniformly random cluster assignments at four values of k, "
                  "twenty seeds each",
            output="silhouette and ARI for random assignment, with the "
                   "standard deviation",
            constraint="report the SPREAD, not just the mean — the anchor is "
                       "not a point, it is a range, and the range is what tells "
                       "you whether 0.02 is a discovery",
            check="the ARI of a random grouping should sit at zero at every k, "
                  "by construction: that is what 'adjusted' means. The "
                  "silhouette should not — predict its sign before running, "
                  "and say why it gets more negative as k grows",
            **{"try": "compare the silhouette anchors at k=2 and k=60. The "
                      "anchor moves by 0.18 across the sweep, which is more "
                      "than the whole signal we are about to find."}),
        code('''
def random_assignment_scores(k, n_seeds=N_SEEDS):
    s, a = [], []
    for seed in range(n_seeds):
        r = np.random.default_rng(1000 + seed)
        lab = r.integers(0, k, size=len(X))
        while len(np.unique(lab)) < 2:           # a one-cluster draw has no silhouette
            lab = r.integers(0, k, size=len(X))
        s.append(float(silhouette_score(X, lab)))
        a.append(float(adjusted_rand_score(y_audit, lab[audit])))
    return np.array(s), np.array(a)

for k in (2, 10, 40, 60):
    s, a = random_assignment_scores(k)
    print(f"k={k:3d}  silhouette {s.mean():+.4f} +/- {s.std():.4f}"
          f"   ARI on the 40 {a.mean():+.4f} +/- {a.std():.4f}", flush=True)
'''),
        prompt(
            label="what the silhouette chose",
            input="the silhouette at every k from the sweep",
            output="the best k, its silhouette and its two ARIs, beside the "
                   "same three quantities at the true k = 40",
            constraint="choose k by the silhouette alone. Choosing it by ARI "
                       "would be selecting a hyperparameter on the answer key — "
                       "the labels this whole notebook is pretending not to have",
            check="the silhouette at the TRUE k should be lower than at the "
                  "chosen k, or the criterion would have found the truth. "
                  "Check both ARIs too: they disagree in size because one is "
                  "measured on 780 pairs and one on 79,800",
            **{"try": "pick k by `np.argmax(ari_full)` instead and note how "
                      "much better everything looks. That is the number you "
                      "are not allowed to have."}),
        code('''
best_i = int(np.argmax(sil))
best_k = ks[best_i]

print(f"best silhouette {sil[best_i]:.4f} at k = {best_k}")
print(f"  ARI on the forty labels : {ari_audit[best_i]:.4f}")
print(f"  ARI on all 400 (AUDIT)  : {ari_full[best_i]:.4f}")
print(f"\\nat the true k = 40:")
print(f"  silhouette              : {sil[i40]:.4f}")
print(f"  ARI on the forty labels : {ari_audit[i40]:.4f}")
print(f"  ARI on all 400 (AUDIT)  : {ari_full[i40]:.4f}")

fig, ax = plt.subplots(figsize=(8, 3))
ax.plot(ks, sil, color="#0b3d62", lw=2, label="k-means")
ax.axhline(0, color="#4b5563", ls=":", lw=2, label="random assignment")
ax.axvline(40, color="#14663a", ls="--", lw=2, label="the true 40")
ax.set_xlabel("k"); ax.set_ylabel("mean silhouette"); ax.legend()
plt.show()
'''),

        # ------------------------------------------ 12 · diagram and faces
        md("""
## 12 · The mean hides the shape, and the shape hides the people

Draw every point's coefficient, sorted, grouped by cluster: one knife per
cluster, its width the cluster's size and its length the quality.
"""),
        prompt(
            label="the silhouette diagram",
            input="the per-point silhouette at four values of k",
            output="one knife per cluster, sorted, with the mean drawn across",
            constraint="sort the clusters by their own mean and sort the points "
                       "inside each — an unsorted diagram is noise with a "
                       "colour map",
            check="`sharex=True` across the panels. Four silhouette diagrams on "
                  "independent x-axes cannot be compared, which is the only "
                  "reason to draw four",
            **{"try": "look for a knife that crosses into negative s_i. Those "
                      "points are closer to another cluster than to their own "
                      "— and the mean silhouette says nothing about how many "
                      "there are."}),
        code('''
fig, axes = plt.subplots(1, 4, figsize=(13, 4), sharex=True)
for ax, k in zip(axes, [2, 10, 40, 60]):
    vals = silhouette_samples(X, labels[k])
    pos = 0
    order = sorted(range(k), key=lambda c: -vals[labels[k] == c].mean())
    for j, c in enumerate(order):
        v = np.sort(vals[labels[k] == c])
        ax.fill_betweenx(np.arange(pos, pos + len(v)), 0, v,
                         color=plt.get_cmap("viridis")(j / max(k - 1, 1)))
        pos += len(v) + 3
    ax.axvline(vals.mean(), color="#c0392b", ls="--", lw=2)
    ax.set_title(f"k = {k}, mean {vals.mean():.3f}")
    ax.set_yticks([])
plt.tight_layout(); plt.show()
'''),
        prompt(
            label="AUDIT — now look at the faces",
            input="the k = 40 clustering and the hidden identities",
            output="cluster sizes, purity, and montages of the cleanest and "
                   "worst clusters",
            constraint="show the WORST cluster, not only the best — a montage "
                       "of the cleanest cluster is a marketing image",
            check="restrict the best/worst search to clusters with at least "
                  "five members, and break a tie in purity by SIZE. A cluster "
                  "of one has purity 1.00 and tells you nothing, and among "
                  "perfectly pure clusters the largest is the one worth "
                  "showing. Every true group has exactly ten members, so the "
                  "spread of sizes is itself a measurement",
            **{"try": "run the same block on `labels[60]`, the k the "
                      "silhouette chose. Mean purity rises and the number of "
                      "clusters holding exactly one person rises with it — "
                      "which is why 59 scored better than 40."}),
        code('''
lab = labels[40]
sizes = np.bincount(lab, minlength=40)
purity = np.array([np.bincount(y[lab == c]).max() / max((lab == c).sum(), 1)
                   for c in range(40)])          # AUDIT — the hidden labels
n_ids = np.array([len(np.unique(y[lab == c])) for c in range(40)])

# Sort by purity, then by size, so a tie among perfectly pure clusters is
# broken by the LARGEST one. `max(..., key=purity)` would return whichever
# happened to come first, which is a different cluster and a different slide.
big = sorted((c for c in range(40) if sizes[c] >= 5),
             key=lambda c: (-purity[c], -sizes[c]))
best, worst = big[0], big[-1]

print(f"cluster sizes: smallest {sizes.min()}, largest {sizes.max()}, "
      f"{(sizes == 10).sum()} of 40 have exactly 10 members")
print(f"clusters holding exactly one person: {(n_ids == 1).sum()} of 40")
print(f"clusters with a single member:       {(sizes == 1).sum()}")
print(f"mean purity across the 40 clusters:  {purity.mean():.4f}")
print(f"cleanest: {sizes[best]} photographs, purity {purity[best]:.4f}")
print(f"worst:    {sizes[worst]} photographs of {n_ids[worst]} people, "
      f"purity {purity[worst]:.4f}")

fig, axes = plt.subplots(2, 1, figsize=(11, 3))
for ax, c, tag in ((axes[0], best, "cleanest"), (axes[1], worst, "worst")):
    montage(ax, images[lab == c][:10], ncol=10)
    ax.set_title(f"{tag}: {sizes[c]} photographs, {n_ids[c]} people, "
                 f"purity {purity[c]:.2f}")
plt.tight_layout(); plt.show()
'''),
        md("""
Look at the worst cluster before reading on. The faces in it are not similar
*people* — they are similar **photographs**: same lighting, same head angle. In
4,096 raw pixels a lamp on the left is a bigger vector than a different nose,
and section 4 predicted exactly this.
"""),

        # ---------------------------------------------- 13 · the optimism
        md("""
## 13 · Which number chose the model, and which number reports it

Sweep $k$, keep the best silhouette, print it — and the number that *chose* the
model is also the number that *reports* it. The maximum of several noisy
estimates is biased upwards, and the bias grows with how many candidates you
tried. It needs no labels to exist.

Measure it: choose $k$ on one half of the corpus, score the chosen model on the
other half, which had no vote.

⏱ **10–45 s.**
"""),
        prompt(
            label="⏱ measure the optimism",
            input="twenty random halves of the corpus",
            output="the selection silhouette and the held-out silhouette per "
                   "seed, and the gap",
            constraint="score the CHOSEN model with `predict` on the other "
                       "half, not a refit — otherwise you are comparing two "
                       "different models rather than one model on two samples",
            check="twenty seeds, and report how many of them the held-out score "
                  "was worse in. If the effect were noise, that count would sit "
                  "near ten",
            **{"try": "shrink the candidate list to `[40]`. The optimism "
                      "collapses, because with one candidate there is no "
                      "maximum to take. The bias is a property of SELECTING, "
                      "not of the silhouette."}),
        code('''
rows = []
for seed in range(N_SEEDS):
    r = np.random.default_rng(2000 + seed)
    perm = r.permutation(len(X))
    a, b = perm[:200], perm[200:]

    chosen, chosen_k, chosen_km = -2.0, None, None
    for k in (5, 10, 20, 40, 60):
        km = KMeans(n_clusters=k, n_init=3, random_state=RANDOM_STATE).fit(X[a])
        s = float(silhouette_score(X[a], km.labels_))
        if s > chosen:
            chosen, chosen_k, chosen_km = s, k, km
    held = float(silhouette_score(X[b], chosen_km.predict(X[b])))
    rows.append((chosen_k, chosen, held))

kk, sel, hel = (np.array(v) for v in zip(*rows))
print(f"k chosen most often: {int(np.bincount(kk).argmax())}")
print(f"reported (chose AND scored on the same half) {sel.mean():.4f} "
      f"+/- {sel.std():.4f}")
print(f"held out (had no vote)                       {hel.mean():.4f} "
      f"+/- {hel.std():.4f}")
print(f"optimism                                     {(sel - hel).mean():+.4f} "
      f"+/- {(sel - hel).std():.4f}")
print(f"that is {(100 * (sel - hel) / sel).mean():.4f}% of the reported score, "
      f"worse in {(hel < sel).sum()}/{N_SEEDS} splits")
'''),

        # ------------------------------------------ 14 · compress and sweep
        md("""
## 14 · Compress first, then cluster

Same k grid, same `n_init`, same seed, same silhouette. The only thing that
changes is **what k-means is looking at**.

⏱ **10–45 s** for all five reduced sweeps together.
"""),
        prompt(
            label="⏱ the same sweep, compressed first",
            input="the corpus, reduced to five different dimensionalities",
            output="wall clock, best k, silhouette and both ARIs for each",
            constraint="change ONE thing. The PCA fit is INSIDE the timer, "
                       "because it is part of what the pipeline costs, and "
                       "leaving it out would flatter the speed-up",
            check="the ARI column is the one to read across rows. The "
                  "silhouette is computed in whatever space you hand it, so it "
                  "is NOT comparable between rows — a different representation "
                  "is a different evaluation set",
            **{"try": "change 0.95 to 0.99 in section 7 and re-run from there. "
                      "d goes from 123 to 260, the sweep slows by roughly the "
                      "ratio of the two, and the ARI barely moves."}),
        code('''
def sweep(Z, k_max=K_MAX):
    """The sweep of section 10, in whatever space Z lives in."""
    best = (-2.0, None, None)
    for k in range(2, k_max + 1):
        km = KMeans(n_clusters=k, n_init=N_INIT, random_state=RANDOM_STATE).fit(Z)
        s = float(silhouette_score(Z, km.labels_))
        if s > best[0]:
            best = (s, k, km.labels_.copy())
    return best

print(f"{'dims':>6}  {'seconds':>8}  {'best k':>6}  {'silhouette':>10}  "
      f"{'ARI all':>8}  {'ARI 40':>8}")
print(f"{4096:>6}  {t_total:>8.4f}  {best_k:>6}  {sil[best_i]:>10.4f}  "
      f"{ari_full[best_i]:>8.4f}  {ari_audit[best_i]:>8.4f}")

speed = {}
for dd in (20, 50, 100, D95, 399):
    t0 = time.perf_counter()
    Z = PCA(n_components=dd, random_state=RANDOM_STATE).fit_transform(X)
    s, k, lb = sweep(Z)
    elapsed = time.perf_counter() - t0
    speed[dd] = dict(time=elapsed, best_k=k, best_sil=s,
                     ari_full=float(adjusted_rand_score(y, lb)),
                     ari_audit=float(adjusted_rand_score(y_audit, lb[audit])))
    print(f"{dd:>6}  {elapsed:>8.4f}  {k:>6}  {s:>10.4f}  "
          f"{speed[dd]['ari_full']:>8.4f}  {speed[dd]['ari_audit']:>8.4f}",
          flush=True)

print(f"\\nspeed-up at d={D95}: {t_total / speed[D95]['time']:.4f}x")
'''),
        md("""
The silhouette **rises** after compression. Not because the clustering got
cleverer — because the criterion is computed in a space where the distances
mean something different. Compressing the data changes every distance, so the
silhouette is not comparable across those rows. The ARI is, because it is
measured against the identities and does not depend on the representation.

Generalise it: *never compare two scores measured on different evaluation
sets*, and a different representation is a different evaluation set.
"""),
        prompt(
            label="does removing the lighting direction help?",
            input="three representations: PCA to 123, PCA with the leading "
                  "three components discarded, and unit-norm images then PCA",
            output="best k, silhouette and ARI for each, on the identical "
                  "protocol",
            constraint="choose k by SILHOUETTE in all three, then report ARI. "
                       "Sweeping k and keeping the best ARI selects a "
                       "hyperparameter on the answer key and flatters whichever "
                       "variant has a lucky k somewhere in the sweep",
            check="a correct diagnosis does not guarantee that the obvious "
                  "repair works. Predict the sign of the change before running, "
                  "then read what actually happened",
            **{"try": "re-run choosing k by `ari_full` instead. Both repairs "
                      "flip to looking like gains. Same data, same code, two "
                      "protocols, opposite conclusions — the protocol was the "
                      "finding."}),
        code('''
P = PCA(n_components=D95, random_state=RANDOM_STATE).fit_transform(X)
Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
Pn = PCA(n_components=D95, random_state=RANDOM_STATE).fit_transform(Xn)

base_ari = None
for tag, Z in ((f"PCA {D95}", P),
               (f"PCA {D95}, first 3 discarded", P[:, 3:]),
               (f"unit-norm images, then PCA {D95}", Pn)):
    s, k, lb = sweep(Z)
    af = float(adjusted_rand_score(y, lb))
    if base_ari is None:
        base_ari = af
    print(f"{tag:34s} dims={Z.shape[1]:4d}  k={k:3d}  sil={s:.4f}  "
          f"ARI={af:.4f}  delta={af - base_ari:+.4f}", flush=True)
'''),
        md("""
Neither repair helps. The diagnosis was right — lighting *does* dominate the
distance — and the leading components carry pose and face shape as well as
illumination, so removing them throws away signal with the nuisance. The remedy
that does work is a face-specific representation, which is Lecture 13.
"""),

        # ------------------------------------- 15 · shapes k-means can't see
        md("""
## 15 · Shapes k-means cannot see

Compression fixed the clock. It did not fix the geometry: every k-means cluster
is still a convex Voronoi cell, every point still belongs to exactly one, and
every group is still assumed to be the same size.
"""),
        prompt(
            label="DBSCAN, and what it cannot see",
            input="the 123-dimensional faces, over a grid of eps",
            output="clusters found, faces called noise and ARI at each eps, and "
                   "the best row",
            constraint="count clusters EXCLUDING the noise label — DBSCAN uses "
                       "−1 for noise, and counting it as a cluster inflates "
                       "every k you report by one",
            check="report the noise count beside the ARI. A method that scores "
                  "well by declaring a quarter of the corpus unclassifiable has "
                  "not solved the brief, and the ARI alone will not tell you "
                  "that happened",
            **{"try": "raise `min_samples` to 5. The peak cluster count falls "
                      "and the noise count rises at every eps — the two "
                      "parameters trade against each other and neither one "
                      "alone is 'the' density scale."}),
        code('''
Z95 = PCA(n_components=D95, random_state=RANDOM_STATE).fit_transform(X)
assert Z95.shape == (400, D95)

eps_grid = np.round(np.linspace(2, 14, 25), 2)
best = (-2.0, None, None, None)
n_cl_all = []
for eps in eps_grid:
    lb = DBSCAN(eps=float(eps), min_samples=3).fit_predict(Z95)
    n_cl = len(set(lb.tolist()) - {-1})
    n_noise = int((lb == -1).sum())
    ari = float(adjusted_rand_score(y, lb))          # AUDIT
    n_cl_all.append(n_cl)
    if ari > best[0]:
        best = (ari, float(eps), n_cl, n_noise)

print(f"best over the eps grid: ARI {best[0]:.4f} at eps={best[1]:.2f}, "
      f"{best[2]} clusters, {best[3]} of 400 faces called noise")
print(f"the most clusters DBSCAN ever finds on this grid: {max(n_cl_all)}")
print("The right answer is 40 clusters and 0 noise. DBSCAN never gets there:")
print("these faces have no single density scale that separates people.")
'''),
        prompt(
            label="why the full covariance is not slow but undefined",
            input="the two dimension counts",
            output="the free parameters of one full covariance, of forty of "
                   "them, and of forty diagonal ones at d = 123",
            constraint="compute it BEFORE fitting anything — the point is that "
                       "arithmetic rules this out in one line, where a timeout "
                       "would take an afternoon and prove nothing",
            check="n(n+1)/2 for n = 4096 is 8,390,656, and forty of them is "
                  "335,626,240 parameters estimated from 400 photographs. You "
                  "can do that arithmetic on paper",
            **{"try": "work out what d would make forty full covariances "
                      "estimable from 400 rows at all. There is no such d "
                      "worth having, which is why `covariance_type='diag'` is "
                      "a decision rather than a default."}),
        code('''
full_params = 4096 * 4097 // 2
print(f"one full 4096-dimensional covariance: {full_params:,} free parameters")
print(f"forty of them:                        {40 * full_params:,}")
print(f"estimated from:                       {len(X)} photographs")
print(f"forty DIAGONAL covariances at d={D95}: {(2 * D95 + 1) * 40:,}")
print("\\nThe first is not slow. It is undefined: every estimate is singular.")
'''),
        md("""
### Gaussian mixtures

Sixteen values of $k$, three restarts each — a few seconds.
"""),
        prompt(
            label="Gaussian mixtures, BIC and ARI",
            input="the 123-dimensional faces, at k = 5, 10, ... 80",
            output="BIC, AIC and ARI at every k, and what each criterion picks",
            constraint="`covariance_type='diag'` and a non-zero `reg_covar` — "
                       "with 400 points in 123 dimensions an unregularised "
                       "covariance goes singular and the likelihood diverges",
            check="BIC and ARI should NOT pick the same k. BIC asks which model "
                  "explains the data as a density; we are asking which grouping "
                  "matches the people, and those coincide only when the people "
                  "really are Gaussian blobs",
            **{"try": "switch to `covariance_type='spherical'`. It is faster, "
                      "and it makes the mixture very nearly k-means — which is "
                      "worth seeing, because it says what the extra parameters "
                      "of 'diag' were buying."}),
        code('''
gks = list(range(5, 81, 5))
bic, aic, gari = [], [], []
for k in gks:
    g = GaussianMixture(n_components=k, covariance_type="diag", n_init=3,
                        random_state=RANDOM_STATE, reg_covar=1e-4).fit(Z95)
    bic.append(float(g.bic(Z95))); aic.append(float(g.aic(Z95)))
    gari.append(float(adjusted_rand_score(y, g.predict(Z95))))     # AUDIT
    print(f"k={k:3d}  BIC {bic[-1]:12.1f}  AIC {aic[-1]:12.1f}  "
          f"ARI {gari[-1]:.4f}", flush=True)

print(f"\\nBIC picks k={gks[int(np.argmin(bic))]}, "
      f"AIC picks k={gks[int(np.argmin(aic))]}, "
      f"ARI peaks at k={gks[int(np.argmax(gari))]} with {max(gari):.4f}")
print(f"ARI at the true k=40: {gari[gks.index(40)]:.4f}")
'''),

        # -------------------------------------------------- 16 · anomalies
        md("""
## 16 · Anomaly detection, two ways

Plant twelve corrupted images in the corpus and see which detector finds them.
One uses the mixture's density; one uses the PCA reconstruction error — a face
the subspace cannot rebuild is a face unlike the ones that built the subspace.
"""),
        prompt(
            label="anomaly detection, two ways",
            input="twelve deliberately corrupted images planted in the corpus",
            output="how many of the twelve each detector puts in its top twelve",
            constraint="fit the PCA and the mixture on the CLEAN corpus and "
                       "score the contaminated one — a detector fitted on the "
                       "anomalies has already learned them",
            check="assert the contaminated matrix is 412 rows with exactly 12 "
                  "flagged. And use THREE kinds of corruption: a detector that "
                  "finds rotations and misses dimming has been measured on one "
                  "failure mode and reported as general",
            **{"try": "fit `p99` on `Xa` instead of `X`. The reconstruction "
                      "detector gets worse, because the subspace now spans the "
                      "corruptions it was supposed to find."}),
        code('''
def corrupt(ims, r, n=12):
    """Faces that are not faces: rotated, dimmed-and-mirrored, double-exposed."""
    idx = r.choice(len(ims), size=n, replace=False)
    out, kinds = [], []
    for j, i in enumerate(idx):
        im = ims[i].copy()
        if j % 3 == 0:
            im = np.rot90(im); kinds.append("rotated")
        elif j % 3 == 1:
            im = im[:, ::-1] * 0.35; kinds.append("dimmed")
        else:
            im = 0.5 * im + 0.5 * im[::-1]; kinds.append("double-exposed")
        out.append(np.ascontiguousarray(im))
    return np.array(out), kinds

bad_im, kinds = corrupt(images, np.random.default_rng(RANDOM_STATE))
Xa = np.vstack([X, bad_im.reshape(12, -1).astype(np.float32)])
is_bad = np.zeros(len(Xa), bool); is_bad[400:] = True
assert Xa.shape == (412, 4096) and is_bad.sum() == 12

p99 = PCA(n_components=0.99, random_state=RANDOM_STATE).fit(X)   # clean only
err = ((p99.inverse_transform(p99.transform(Xa)) - Xa) ** 2).mean(axis=1)

gmm = GaussianMixture(n_components=40, covariance_type="diag", n_init=3,
                      random_state=RANDOM_STATE, reg_covar=1e-4
                      ).fit(p99.transform(X))
dens = gmm.score_samples(p99.transform(Xa))

print(f"the 99% subspace has {p99.n_components_} components")
print(f"reconstruction error: {is_bad[np.argsort(-err)[:12]].sum()} of 12 "
      f"planted images in the top twelve")
print(f"lowest density:       {is_bad[np.argsort(dens)[:12]].sum()} of 12")
print(f"median reconstruction error, genuine faces  {np.median(err[~is_bad]):.8f}")
print(f"median reconstruction error, planted images {np.median(err[is_bad]):.8f}")

fig, axes = plt.subplots(2, 1, figsize=(11, 3))
allims = np.vstack([images, bad_im])
montage(axes[0], allims[np.argsort(-err)[:10]], ncol=10)
axes[0].set_title("most anomalous by reconstruction error")
montage(axes[1], allims[np.argsort(dens)[:10]], ncol=10)
axes[1].set_title("most anomalous by lowest mixture density")
plt.tight_layout(); plt.show()
'''),
        md("""
A rotated or double-exposed face lies **outside** the subspace the eigenfaces
span, so its reconstruction error is large. The mixture density is estimated
*inside* that subspace, so a corruption that projects onto an ordinary-looking
point is invisible to it.

**The decision rule.** Use the reconstruction error when the anomaly is a
different *kind* of object. Use the density when it is an ordinary object in an
unusual place.
"""),

        # ------------------------------------------- 17 · the forty labels
        md("""
## 17 · Spending the forty labels

A budget of forty labels and 280 training photographs. Where the budget is
spent turns out to matter more than what is done with it.

⏱ **15–45 s** — twenty stratified splits, five strategies each.
"""),
        prompt(
            label="⏱ five ways to spend forty labels",
            input="forty labels, spent five different ways, over twenty splits",
            output="held-out accuracy for each strategy, with its spread",
            constraint="the same classifier, the same features and the same "
                       "held-out faces throughout — the only thing that varies "
                       "is WHICH forty were labelled",
            check="assert the forty representatives are forty DISTINCT faces: "
                  "`argmin` down the 280 × 40 distance matrix can in principle "
                  "return the same face for two centroids. Report the all-280 "
                  "row as a ceiling — a semi-supervised result with no "
                  "fully-supervised comparison cannot be read",
            **{"try": "raise the percentile from 75 to 100. That is the "
                      "propagate-to-the-whole-cluster row, so the two should "
                      "agree — a cheap check that the masking is doing what "
                      "you think."}),
        code('''
def accuracy(Ztr, ytr, Zte, yte):
    clf = LogisticRegression(max_iter=3000, random_state=RANDOM_STATE)
    return float(clf.fit(Ztr, ytr).score(Zte, yte))

arms = {"40 at random": [], "40, one per cluster": [],
        "propagated to whole cluster": [], "propagated to closest 75%": [],
        "all 280 labels": []}

for seed in range(N_SEEDS):
    Xa_tr, Xa_te, ya_tr, ya_te = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=seed)
    p = PCA(n_components=D95, random_state=RANDOM_STATE).fit(Xa_tr)
    Ztr, Zte = p.transform(Xa_tr), p.transform(Xa_te)
    r = np.random.default_rng(3000 + seed)

    pick = r.choice(len(Ztr), size=40, replace=False)
    arms["40 at random"].append(accuracy(Ztr[pick], ya_tr[pick], Zte, ya_te))

    km = KMeans(n_clusters=40, n_init=10, random_state=seed).fit(Ztr)
    dist = km.transform(Ztr)                       # (280, 40)
    rep = np.argmin(dist, axis=0)                  # nearest face to each centroid
    assert len(rep) == 40 and len(np.unique(rep)) == 40
    arms["40, one per cluster"].append(accuracy(Ztr[rep], ya_tr[rep], Zte, ya_te))

    prop = ya_tr[rep][km.labels_]                  # 280 labels from 40
    arms["propagated to whole cluster"].append(accuracy(Ztr, prop, Zte, ya_te))

    own = dist[np.arange(len(Ztr)), km.labels_]
    keep = np.zeros(len(Ztr), bool)
    for c in range(40):
        m = np.where(km.labels_ == c)[0]
        keep[m[own[m] <= np.percentile(own[m], 75)]] = True
    arms["propagated to closest 75%"].append(
        accuracy(Ztr[keep], prop[keep], Zte, ya_te))

    arms["all 280 labels"].append(accuracy(Ztr, ya_tr, Zte, ya_te))

for name, v in arms.items():
    v = np.array(v)
    print(f"{name:30s} {100 * v.mean():6.4f}%  sd {100 * v.std():.4f}  "
          f"min {100 * v.min():.4f}  max {100 * v.max():.4f}")
'''),
        md("""
Same budget, same classifier, same features — and forty labels spent on cluster
representatives are worth about a quarter more accuracy than forty spent at
random. Nothing here is a better classifier; the difference is entirely in the
sampling.

Note also what propagation does. It turns forty labels into 280 and scores
*worse* than the forty alone, because a wrongly propagated label is worse than
no label. Report that; do not bury it.
"""),

        # ---------------------------------------------------- 18 · the leak
        md("""
## 18 · Fitting a reducer before the split

A reducer is an estimator: it is **fitted**. Fit it on everything and the
held-out faces have helped choose the subspace they are then judged in. With
400 points in 4,096 dimensions this is not a rounding error — the 280 training
faces span at most a 280-dimensional subspace, so adding 120 more points
genuinely changes which directions survive.

Two consequences, measured separately over twenty splits, because they do not
behave the same way — and reporting only the one that moves would be the
dishonesty this course exists to remove.

⏱ **15–45 s.**
"""),
        prompt(
            label="⏱ measure both consequences",
            input="twenty stratified splits, each with the PCA fitted honestly "
                  "and then on everything",
            output="held-out reconstruction error and accuracy both ways, with "
                  "the win counts",
            constraint="report BOTH even though only one of them moves — the "
                       "null result is half the finding, and a rule justified "
                       "by only the half that moved is a rule nobody will keep",
            check="predict the SIGN of each effect before running. The "
                  "reconstruction error is the objective PCA optimises, so "
                  "letting the held-out faces vote should lower it in nearly "
                  "every split; accuracy is several steps downstream and the "
                  "classifier is refitted honestly either way",
            **{"try": "raise d to 279, the rank of the training set. The "
                      "reconstruction gap widens, because there are more "
                      "directions for the extra 120 faces to influence."}),
        code('''
rows = []
for seed in range(N_SEEDS):
    Xa_tr, Xa_te, ya_tr, ya_te = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=seed)
    honest = PCA(n_components=D95, random_state=RANDOM_STATE).fit(Xa_tr)
    leaky = PCA(n_components=D95, random_state=RANDOM_STATE).fit(X)   # all 400

    def err(p):
        return float(((p.inverse_transform(p.transform(Xa_te)) - Xa_te) ** 2).mean())

    def acc(p):
        c = LogisticRegression(max_iter=3000, random_state=RANDOM_STATE)
        return float(c.fit(p.transform(Xa_tr), ya_tr)
                     .score(p.transform(Xa_te), ya_te))

    rows.append((err(honest), err(leaky), acc(honest), acc(leaky)))

eh, el, ah, al = (np.array(v) for v in zip(*rows))
drop = 100 * (1 - el / eh)
print(f"reconstruction error  honest {eh.mean():.7f}   leaky {el.mean():.7f}")
print(f"  the leak makes it look {drop.mean():.4f}% better "
      f"(sd {drop.std():.4f}), in {(el < eh).sum()}/{N_SEEDS} splits")
print(f"accuracy              honest {ah.mean():.6f} (sd {ah.std():.6f})   "
      f"leaky {al.mean():.6f} (sd {al.std():.6f})")
print(f"  difference {100 * (al - ah).mean():+.4f} points "
      f"(sd {100 * (al - ah).std():.4f}), leaky wins {(al > ah).sum()}/{N_SEEDS}")
'''),
        md("""
**The decision rule.** Fitting an unsupervised step on everything costs you
*exactly when that step's own objective is the number you report*.
Reconstruction error is that objective; downstream accuracy is not — and you
cannot tell which case you are in without doing the split, which is the
argument for always splitting first.
"""),
        prompt(
            label="the structural fix",
            input="the reducer and the classifier as one object",
            output="the pipeline's held-out accuracy",
            constraint="both steps in ONE `Pipeline`, so cross-validation "
                       "refits the PCA inside every fold",
            check="the accuracy should match the 'honest' row above at the same "
                  "seed, because it is the same computation — the Pipeline "
                  "changes who can get it wrong, not what the answer is",
            **{"try": "list the unsupervised steps in your own notebooks that "
                      "are safe to fit outside a Pipeline. The list is empty; "
                      "scaler, imputer, encoder and PCA all belong inside."}),
        code('''
pipe = Pipeline([("pca", PCA(n_components=D95, random_state=RANDOM_STATE)),
                 ("clf", LogisticRegression(max_iter=3000,
                                            random_state=RANDOM_STATE))])
pipe.fit(X_tr, y_tr)
print(f"pipeline accuracy on the held-out 120: {pipe.score(X_te, y_te):.6f}")
print("PCA is refitted inside every fold, so the leak is structurally "
      "impossible rather than merely avoided.")
'''),

        # ------------------------------------------------------ 19 · close
        md("""
## 19 · What to take away

1. **PCA is the SVD of the centred data.** The principal subspace minimises
   squared reconstruction error exactly, and the error it leaves is
   $\\sum_{j>d}\\sigma_j^2$ — checked to machine precision in section 6.
2. **The explained variance ratio is that theorem, normalised**, and it is how
   $d$ is chosen against a stated target rather than by eye.
3. **Johnson–Lindenstrauss** bounds the target dimension by the number of
   points and the tolerance, never by the dimension you started in — and it is
   a loose worst case, so measure the distortion you actually get.
4. **Inertia cannot choose $k$**: it is monotone in $k$ and zero at $k = m$.
   The silhouette can, at $O(m^2 n)$, and must be read against a random anchor.
5. **The score that chose a model is optimistic**, by about half its own value
   here, and that holds with no labels anywhere in sight.
6. **k-means assumes convex, equal-sized clusters and a known $k$.** DBSCAN and
   Gaussian mixtures drop the first two, and on this data DBSCAN fails for a
   stateable reason: there is no single density scale.
7. **Reconstruction error finds anomalies of a different kind; density finds
   ordinary objects in unusual places.**
8. **Which forty labels mattered more than what was done with them.**

The failure condition that runs through the whole notebook is section 4's: in
raw pixels a lighting condition is a larger vector than a face, and no amount
of compression repairs it. The representation is the thing to change, and doing
that properly is Lecture 13.
"""),
    ]
    return cells
