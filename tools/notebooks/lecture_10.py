#!/usr/bin/env python3
"""
Lecture 10 — Four thousand dimensions is too many. Fix, Chapters 7–8.

`tools/make_notebooks.py` discovers this file and calls `build()`, which returns
the list of cells. Self-contained: it reloads the data and rebuilds the state
Lecture 9 ended in, because a notebook that only runs because another one left
variables in the kernel is not reproducible.

Structure mirrors the lecture — thread, diagnosis, repair, re-measure, red team.
"""

from __future__ import annotations

import nbformat as nbf


def md(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(text.strip("\n"))


def code(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(text.strip("\n"))


def build() -> list:
    return [
        md("""
# Four thousand dimensions is too many

**Lecture 10 · Fix** · Géron, Chapters 7 & 8 ·
*Mathematical thread: SVD, PCA and Johnson–Lindenstrauss*

Applications of Machine Learning — BSc Mathematics of Artificial Intelligence

---

**How to use this notebook.** You are not expected to type the code. You are
expected to *read* it before you run it, and to be able to say what every line
does and what would break if it changed. Cells marked **⚠ read before running**
contain a defect on purpose.

Run the cells in order. Anything that takes more than a few seconds says so.
"""),

        md("## 1 · Setup, and where we left off"),
        code('''
# --- setup -------------------------------------------------------------------
# Not examinable. The thread limit makes a measured time repeatable; the default
# is "all cores", which on a shared machine means "whatever is left".
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import sys, time
import numpy as np
import sklearn
import matplotlib.pyplot as plt

# Every import this notebook needs, in one place.
from sklearn.cluster import DBSCAN, KMeans
from sklearn.datasets import fetch_olivetti_faces
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (adjusted_rand_score, pairwise_distances,
                             silhouette_score)
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.random_projection import (GaussianRandomProjection,
                                       johnson_lindenstrauss_min_dim)

print(f"python       {sys.version.split()[0]}")
print(f"scikit-learn {sklearn.__version__}")

RANDOM_STATE = 42
rng = np.random.default_rng(RANDOM_STATE)

faces = fetch_olivetti_faces(shuffle=False)
X, y, images = faces.data, faces.target, faces.images
audit = np.sort(rng.choice(400, size=40, replace=False))   # the same forty
y_audit = y[audit]

assert X.shape == (400, 4096)
assert len(audit) == 40
print(f"{len(X)} faces, {X.shape[1]} features — the same corpus as last time")
'''),

        md("""
## 2 · Thread 5 — PCA via the SVD

Centre the data and stack the faces as the rows of $\\mathbf{X}$. The singular
value decomposition writes

$$\\mathbf{X} = \\mathbf{U}\\,\\boldsymbol\\Sigma\\,\\mathbf{V}^{\\mathsf T},$$

with $\\mathbf{U}$ and $\\mathbf{V}$ orthogonal and $\\boldsymbol\\Sigma$
diagonal with non-negative, decreasing entries. The columns of $\\mathbf{V}$ are
the **principal components**, and they are also the eigenvectors of
$\\mathbf{X}^{\\mathsf T}\\mathbf{X}$ — which is the matrix Lecture 2 needed to
be invertible.

Compute it by hand and check that scikit-learn agrees.

⏱ **about 20 seconds** — a full SVD of a 400 × 4,096 matrix.
"""),
        code('''
X_centred = X - X.mean(axis=0)
U, S, Vt = np.linalg.svd(X_centred, full_matrices=False)

pca = PCA(random_state=RANDOM_STATE).fit(X)

assert Vt.shape == (400, 4096), Vt.shape
# components are defined up to a sign, so compare absolute values
agree = np.abs(np.abs(Vt[:5]) - np.abs(pca.components_[:5])).max()
print(f"largest disagreement over the first five components: {agree:.2e}")
assert agree < 1e-4

# the explained variance ratio IS the normalised squared singular values
evr = S ** 2 / (S ** 2).sum()
print(f"largest disagreement over the variance ratios: "
      f"{np.abs(evr - pca.explained_variance_ratio_).max():.2e}")
'''),

        md("""
### In what sense does PCA minimise anything?

Keep the first $d$ components and project. The Eckart–Young theorem says this
is the best rank-$d$ approximation of $\\mathbf{X}$ in Frobenius norm, and the
error it leaves is exactly the tail of the spectrum:

$$\\lVert \\mathbf{X} - \\mathbf{X}_d \\rVert_F^2 = \\sum_{j>d}\\sigma_j^2 .$$

That is a statement you can check to machine precision, so check it.
"""),
        code('''
d = 100
Xd = U[:, :d] * S[:d] @ Vt[:d]
lhs = ((X_centred - Xd) ** 2).sum()
rhs = (S[d:] ** 2).sum()
print(f"‖X − X_d‖²_F = {lhs:.6f}")
print(f"Σ_{{j>d}} σ_j²  = {rhs:.6f}")
print(f"relative difference {abs(lhs - rhs) / rhs:.2e}")
assert abs(lhs - rhs) / rhs < 1e-5
'''),

        md("""
### The eigenfaces

The components live in the same space as the data, so each one is a 64 × 64
image. That is why the principal components of a face dataset have a name.
"""),
        code('''
def montage(ax, ims, ncol, gap=2):
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

def rescale(v):
    v = v.reshape(64, 64)
    return (v - v.min()) / (v.max() - v.min())

tiles = [X.mean(axis=0).reshape(64, 64)] + [rescale(c) for c in pca.components_[:15]]
fig, ax = plt.subplots(figsize=(11, 3))
montage(ax, np.array(tiles), ncol=8)
ax.set_title("the mean face, then the first 15 principal components")
plt.show()
'''),

        md("""
### How many components do we need?
"""),
        code('''
cum = np.cumsum(pca.explained_variance_ratio_)
d95 = int(np.searchsorted(cum, 0.95) + 1)
d99 = int(np.searchsorted(cum, 0.99) + 1)

print(f"component 1 alone explains {100 * pca.explained_variance_ratio_[0]:.1f}%")
print(f"95% of the variance needs {d95} components  "
      f"({4096 / d95:.0f}x fewer than 4096)")
print(f"99% needs {d99}")

fig, ax = plt.subplots(figsize=(8, 3))
ax.plot(range(1, len(cum) + 1), 100 * cum, color="#0b3d62", lw=2)
ax.axhline(95, color="#14663a", ls="--", lw=2)
ax.axvline(d95, color="#14663a", ls=":", lw=2)
ax.set_xlabel("components kept"); ax.set_ylabel("cumulative variance, %")
plt.show()
'''),
        code('''
# what does a face look like as you take components away?
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=RANDOM_STATE)
assert len(X_tr) == 280 and len(X_te) == 120
assert np.bincount(y_tr).min() == 7 and np.bincount(y_te).min() == 3

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

        md("""
## 3 · Johnson–Lindenstrauss, as the textbook states it

> For any $0 < \\varepsilon < 1$ and any set of $n$ points in
> $\\mathbb{R}^{D}$, there is a linear map into $\\mathbb{R}^{d}$ with
> $$d \\;\\ge\\; \\frac{4\\log n}{\\varepsilon^{2}/2 - \\varepsilon^{3}/3}$$
> that preserves every pairwise distance to within a factor $1 \\pm \\varepsilon$.

**Read the formula again and say what is missing.** $n$ is there. $\\varepsilon$
is there. $D$ — the dimension you start in — is not. Four hundred points need
the same target dimension whether they live in 4,096 dimensions or in four
million.
"""),
        code('''
for eps in (0.1, 0.2, 0.3, 0.5):
    print(f"eps={eps}:  400 points -> {johnson_lindenstrauss_min_dim(400, eps=eps):>7,}"
          f"   1,000,000 points -> "
          f"{johnson_lindenstrauss_min_dim(1_000_000, eps=eps):>7,}")
print(f"\\nthe dimension we actually have: {X.shape[1]}")
'''),
        md("""
Two things students usually miss and one that surprises everybody:

1. the bound grows like $\\log n$ — a factor of 2,500 more points costs about
   twice the dimension;
2. it does not mention $D$ at all;
3. at $\\varepsilon = 0.1$ the bound for our 400 faces is **larger than 4,096**.
   The theorem is a worst-case guarantee over all possible point sets. Ours is
   not the worst case, so measure what actually happens.

⏱ **about 1 minute.**
"""),
        code('''
iu = np.triu_indices(len(X), k=1)
D0 = pairwise_distances(X)[iu]
print(f"{len(D0):,} pairwise distances\\n")

for dd in (50, 100, 200, 400, 800):
    worst, p95 = [], []
    for seed in range(3):
        g = GaussianRandomProjection(n_components=dd, random_state=seed)
        D1 = pairwise_distances(g.fit_transform(X))[iu]
        rel = np.abs(D1 / D0 - 1)
        worst.append(rel.max()); p95.append(np.quantile(rel, 0.95))
    print(f"d={dd:5d}   worst pair {np.mean(worst):.3f}   "
          f"95th percentile {np.mean(p95):.3f}")
'''),
        md("""
At $d = 200$ the worst distance in the whole corpus is distorted by far less
than the $\\varepsilon = 0.2$ the bound would only guarantee at 1,382
dimensions. The bound is *sufficient*, not *necessary* — and the reason to teach
it is not the constant. It is that the constant does not contain 4,096.
"""),

        md("""
## 4 · The repair: compress, then cluster

Same sweep as the previous lecture, same k-means, same silhouette. The only
change is what k-means is looking at.

⏱ **1–2 minutes** for the reduced sweep. Compare it with what you wrote down
last time.
"""),
        code('''
def sweep(data, ks=(2, 5, 10, 15, 20, 30, 40, 50, 60), n_init=5):
    t0 = time.perf_counter()
    best = (-2.0, None, None)
    for k in ks:
        km = KMeans(n_clusters=k, n_init=n_init, random_state=RANDOM_STATE).fit(data)
        s = silhouette_score(data, km.labels_)
        if s > best[0]:
            best = (s, k, km.labels_.copy())
    return best[0], best[1], best[2], time.perf_counter() - t0

Z95 = PCA(n_components=d95, random_state=RANDOM_STATE).fit_transform(X)
assert Z95.shape == (400, d95)

s_raw, k_raw, lab_raw, t_raw = sweep(X)
s_red, k_red, lab_red, t_red = sweep(Z95)

print(f"4096 dims: {t_raw:6.1f}s   best k={k_raw:3d}  silhouette {s_raw:.4f}  "
      f"ARI(all) {adjusted_rand_score(y, lab_raw):.3f}")
print(f"{d95:4d} dims: {t_red:6.1f}s   best k={k_red:3d}  silhouette {s_red:.4f}  "
      f"ARI(all) {adjusted_rand_score(y, lab_red):.3f}")
print(f"\\nspeed-up {t_raw / t_red:.1f}x")
'''),
        md("""
The silhouette *rises* after compression. Not because the clustering got
cleverer — because the criterion is computed in a space where the distances
mean more. Report both numbers; the comparison is only meaningful because the
ARI, which is measured against the identities and does not depend on the
representation, moved in the same direction.
"""),

        md("""
## 5 · Four ways to reduce, timed and scored

Randomised PCA approximates the top components without the full SVD.
Incremental PCA never holds the whole matrix. Random projection does not look at
the data at all.
"""),
        code('''
def bench(make, name):
    t0 = time.perf_counter(); obj = make(); t = time.perf_counter() - t0
    if hasattr(obj, "inverse_transform"):
        err = ((obj.inverse_transform(obj.transform(X_te)) - X_te) ** 2).mean()
    else:
        back = np.linalg.pinv(obj.components_.T)
        err = ((obj.transform(X_te) @ back - X_te) ** 2).mean()
    print(f"{name:22s} {1000 * t:8.0f} ms   held-out error {err:.5f}")
    return t, err

bench(lambda: PCA(d95, svd_solver="full", random_state=RANDOM_STATE).fit(X_tr),
      "PCA, full SVD")
bench(lambda: PCA(d95, svd_solver="randomized", random_state=RANDOM_STATE).fit(X_tr),
      "PCA, randomised")
bench(lambda: IncrementalPCA(d95, batch_size=70).fit(X_tr), "Incremental PCA")
bench(lambda: GaussianRandomProjection(d95, random_state=RANDOM_STATE).fit(X_tr),
      "Random projection")
'''),
        md("""
Random projection is essentially free — it draws a Gaussian matrix — and it pays
for that in reconstruction error, because it is not looking for the subspace the
faces occupy. Which you prefer depends on whether you need the representation or
the reconstruction.
"""),

        md("""
## 6 · Shapes that k-means cannot see

k-means partitions space with a Voronoi diagram: every cluster is convex, and
every point belongs to one. Two methods that do not.
"""),
        code('''
best = (-2, None)
for eps in np.linspace(2, 14, 25):
    lab = DBSCAN(eps=float(eps), min_samples=3).fit_predict(Z95)
    n_clusters = len(set(lab.tolist()) - {-1})
    ari = adjusted_rand_score(y, lab)
    if ari > best[0]:
        best = (ari, eps, n_clusters, (lab == -1).sum())
print(f"DBSCAN, best over the eps grid: ARI {best[0]:.3f} at eps={best[1]:.2f} "
      f"with {best[2]} clusters and {best[3]} faces called noise")
print("The right answer is 40 clusters and 0 noise. DBSCAN never gets there:")
print("faces in this subspace have no density scale that separates people.")
'''),
        code('''
# Gaussian mixtures. NOTE the covariance_type — the full version is impossible
# here, and it is worth seeing why before running the diagonal one.
full_params = 4096 * 4097 // 2
print(f"a full 4096-dimensional covariance has {full_params:,} free parameters")
print(f"forty of them: {40 * full_params:,}, estimated from 400 photographs")
print("This is not slow. It is undefined.\\n")

bic, ari = [], []
ks = list(range(5, 81, 5))
for k in ks:
    g = GaussianMixture(n_components=k, covariance_type="diag", n_init=3,
                        random_state=RANDOM_STATE, reg_covar=1e-4).fit(Z95)
    bic.append(g.bic(Z95)); ari.append(adjusted_rand_score(y, g.predict(Z95)))
    print(f"k={k:3d}  BIC {g.bic(Z95):12.0f}  ARI {ari[-1]:.3f}", flush=True)

print(f"\\nBIC picks k={ks[int(np.argmin(bic))]}; "
      f"ARI peaks at k={ks[int(np.argmax(ari))]}")
'''),

        md("""
## 7 · Anomaly detection, two ways

Plant twelve corrupted images in the corpus and see which detector finds them.
One uses the mixture's density, one uses the PCA reconstruction error — a face
the subspace cannot rebuild is a face unlike the ones that built the subspace.
"""),
        code('''
def corrupt(ims, r, n=12):
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

p99 = PCA(n_components=0.99, random_state=RANDOM_STATE).fit(X)
err = ((p99.inverse_transform(p99.transform(Xa)) - Xa) ** 2).mean(axis=1)

gmm = GaussianMixture(n_components=40, covariance_type="diag", n_init=3,
                      random_state=RANDOM_STATE, reg_covar=1e-4
                      ).fit(p99.transform(X))
dens = gmm.score_samples(p99.transform(Xa))

print(f"reconstruction error: {is_bad[np.argsort(-err)[:12]].sum()} of 12 "
      f"planted images in the top twelve")
print(f"lowest mixture density: {is_bad[np.argsort(dens)[:12]].sum()} of 12")

fig, axes = plt.subplots(2, 1, figsize=(11, 3))
allims = np.vstack([images, bad_im])
montage(axes[0], allims[np.argsort(-err)[:10]], ncol=10)
axes[0].set_title("most anomalous by reconstruction error")
montage(axes[1], allims[np.argsort(dens)[:10]], ncol=10)
axes[1].set_title("most anomalous by lowest density")
plt.tight_layout(); plt.show()
'''),

        md("""
## 8 · Spending the forty labels

Now the payoff. We have a budget of forty labels and 280 training photographs.
Where you spend the budget matters more than what you do with it.

⏱ **about 1 minute.**
"""),
        code('''
def accuracy(Ztr, ytr, Zte, yte):
    clf = LogisticRegression(max_iter=3000, random_state=RANDOM_STATE)
    return clf.fit(Ztr, ytr).score(Zte, yte)

p = PCA(n_components=d95, random_state=RANDOM_STATE).fit(X_tr)
Ztr, Zte = p.transform(X_tr), p.transform(X_te)

r = np.random.default_rng(RANDOM_STATE)
pick = r.choice(len(Ztr), size=40, replace=False)
print(f"40 at random               {accuracy(Ztr[pick], y_tr[pick], Zte, y_te):.3f}")

km = KMeans(n_clusters=40, n_init=10, random_state=RANDOM_STATE).fit(Ztr)
dist = km.transform(Ztr)
rep = np.argmin(dist, axis=0)                 # the face nearest each centroid
assert len(rep) == 40 and len(np.unique(rep)) == 40
print(f"40, one per cluster        {accuracy(Ztr[rep], y_tr[rep], Zte, y_te):.3f}")

prop = y_tr[rep][km.labels_]                  # the representative's label
print(f"propagated to the cluster  {accuracy(Ztr, prop, Zte, y_te):.3f}")

own = dist[np.arange(len(Ztr)), km.labels_]
keep = np.zeros(len(Ztr), bool)
for c in range(40):
    m = np.where(km.labels_ == c)[0]
    keep[m[own[m] <= np.percentile(own[m], 75)]] = True
print(f"propagated to closest 75%  {accuracy(Ztr[keep], prop[keep], Zte, y_te):.3f}")

print(f"all 280 true labels        {accuracy(Ztr, y_tr, Zte, y_te):.3f}")
'''),
        md("""
Same forty labels, several times the accuracy, purely because the clustering
chose *which* forty. Nothing here is a better classifier; the difference is
entirely in the sampling.
"""),

        md("""
## 9 · An assistant reduces the dimension for us

**⚠ Read before running.** This is today's failure, and unlike the scaling leak
in the first application it costs something you can see.

> *"Reduce the faces to 95% of the variance with PCA and train a classifier;
> report the held-out accuracy and the reconstruction error."*
"""),
        code('''
pca_all = PCA(n_components=d95, random_state=RANDOM_STATE).fit(X)   # <-- all 400
Z_all = pca_all.transform(X)

Ztr_l, Zte_l, ytr_l, yte_l = train_test_split(
    Z_all, y, test_size=0.3, stratify=y, random_state=RANDOM_STATE)

clf = LogisticRegression(max_iter=3000, random_state=RANDOM_STATE).fit(Ztr_l, ytr_l)
rec = pca_all.inverse_transform(pca_all.transform(X_te))
print(f"accuracy {clf.score(Zte_l, yte_l):.3f}")
print(f"reconstruction error on held-out faces {((rec - X_te) ** 2).mean():.5f}")
'''),
        md("""
### Reviewer question 2: what was fitted, and on what?

`PCA(...).fit(X)` — all four hundred photographs, including the hundred and
twenty we then call held out. The subspace those faces are projected onto was
chosen partly *by* those faces, and with 400 points in 4,096 dimensions that is
not a rounding error: the training set alone spans at most a 279-dimensional
subspace, and adding the test faces changes which directions survive.

Measure both consequences over twenty splits, and report both even though only
one of them moves.

⏱ **2–4 minutes.**
"""),
        code('''
rows = []
for seed in range(20):
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=seed)
    honest = PCA(n_components=d95, random_state=RANDOM_STATE).fit(Xtr)
    leaky = PCA(n_components=d95, random_state=RANDOM_STATE).fit(X)

    def err(pp):
        return ((pp.inverse_transform(pp.transform(Xte)) - Xte) ** 2).mean()

    def acc(pp):
        c = LogisticRegression(max_iter=3000, random_state=RANDOM_STATE)
        return c.fit(pp.transform(Xtr), ytr).score(pp.transform(Xte), yte)

    rows.append((err(honest), err(leaky), acc(honest), acc(leaky)))

eh, el, ah, al = (np.array(v) for v in zip(*rows))
print(f"reconstruction error  honest {eh.mean():.5f}   leaky {el.mean():.5f}")
print(f"  the leak makes it look {100 * (1 - el / eh).mean():.0f}% better, "
      f"in {(el < eh).sum()}/20 splits")
print(f"accuracy              honest {ah.mean():.3f}   leaky {al.mean():.3f}")
print(f"  difference {100 * (al - ah).mean():+.2f} points, "
      f"sd {100 * (al - ah).std():.2f}, leaky wins {(al > ah).sum()}/20")
'''),
        md("""
### The corrected specification

> *"Split first, stratified by identity, fixed seed. Fit PCA on the training
> faces only and `transform` the held-out ones — put it in a `Pipeline` so
> cross-validation refits it per fold. Report the held-out reconstruction error
> and the accuracy, both with their spread over seeds."*

And the decision rule, which is the point of measuring rather than asserting:
**the damage from fitting an unsupervised step on everything is large exactly
when that step's own output is the thing you report.** Reconstruction error is
that thing. Downstream accuracy usually is not — but you cannot tell which case
you are in without the split.
"""),
        code('''
from sklearn.pipeline import Pipeline

pipe = Pipeline([("pca", PCA(n_components=d95, random_state=RANDOM_STATE)),
                 ("clf", LogisticRegression(max_iter=3000,
                                            random_state=RANDOM_STATE))])
pipe.fit(X_tr, y_tr)
print(f"pipeline accuracy {pipe.score(X_te, y_te):.3f}")
print("PCA is now refitted inside every fold, so the leak is structurally "
      "impossible rather than merely avoided.")
'''),

        md("""
## 10 · Red-team

Swap notebooks with the team beside you. Fifteen minutes. Five questions:

1. What touched the test set?
2. What was fitted, and on what? (`fit` and `transform` are different verbs)
3. What is the shape here?
4. What was dropped — rows, columns, NaNs? Count them.
5. What is the default I did not ask for?

Two extra ones that belong to this application specifically:

6. **Which number chose the model, and which number reports it?** If they are
   the same number, it is optimistic.
7. **Was the cluster looked at?** A silhouette without a montage is a number
   about a geometry, not about people.

Report what you **found**, not what you would have done differently.
"""),
    ]
