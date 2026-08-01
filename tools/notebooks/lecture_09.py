#!/usr/bin/env python3
"""
Lecture 9 — Forty labels for four hundred faces. Build, Olivetti, Chapter 8.

`tools/make_notebooks.py` discovers this file and calls `build()`, which returns
the list of cells. The notebook is self-contained: it imports everything it
needs and re-derives everything it uses, because a notebook that only runs
because another one left variables in the kernel is not reproducible.

Structure mirrors the lecture — brief, data, metric, commitment, build.
"""

from __future__ import annotations

import nbformat as nbf
from _prompt import prompt                                # noqa: E402


def md(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(text.strip("\n"))


def code(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(text.strip("\n"))


def build() -> list:
    return [
        md("""
# Forty labels for four hundred faces

**Lecture 9 · Build** · Géron, Chapter 8

Applications of Machine Learning — BSc Mathematics of Artificial Intelligence

---

**How to use this notebook.** You are not expected to type the code. You are
expected to *read* it before you run it, and to be able to say what every line
does and what would break if it changed. Cells marked **⚠ read before running**
contain a defect on purpose.

Run the cells in order. Anything that takes more than a few seconds says so.
"""),

        md("## 1 · Setup"),
        prompt(
            label="setup, and one line that is not hygiene",
            input="nothing",
            output="versions, a fixed seed, and a thread cap",
            constraint="cap the BLAS threads before importing numpy — the environment variables are read at import time, and setting them afterwards does nothing at all",
            left_open="why a thread cap belongs in a clustering notebook. This lecture ends by quoting a wall clock, and the default is 'all cores', which on a shared machine means 'whatever is left' — a timing you cannot repeat.",
            student="setting `OMP_NUM_THREADS` in a later cell and wondering why the timings still vary by a factor of three between runs.",
            catch="if a notebook reports a duration, it has to control what the duration depends on. Otherwise the number is about the machine's mood."),
        code('''
# --- setup -------------------------------------------------------------------
# Not examinable: this is engineering hygiene, not machine learning. The thread
# limit is here so that a timing you measure is a timing you can repeat — the
# default is "all cores", which on a shared machine means "whatever is left".
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import sys, time
import numpy as np
import sklearn
import matplotlib
import matplotlib.pyplot as plt

print(f"python       {sys.version.split()[0]}")
print(f"scikit-learn {sklearn.__version__}")
print(f"numpy        {np.__version__}")

RANDOM_STATE = 42          # every split, every model, every shuffle
rng = np.random.default_rng(RANDOM_STATE)
'''),

        md("""
## 2 · The brief

An institution has a photographic archive. Nobody knows how many distinct
people are in it, and nobody has the budget to look at every picture. The task
is to **group the photographs by identity** so that a human can then name a
group rather than a photograph.

The constraint that decides everything: **labelling is expensive**. We are
allowed to pay an annotator for forty labels. That is 10% of the corpus.

The stand-in corpus is Olivetti — small, in the textbook, and downloadable in
seconds.
"""),
        prompt(
            label="the corpus",
            input="Olivetti faces",
            output="400 photographs as 4,096-dimensional vectors, plus the 64×64 images",
            constraint="`shuffle=False` — the ten photographs of each person stay adjacent, which every montage below relies on",
            check="assert the shape, the pixel range, forty people, and exactly ten photographs each",
            left_open="that `y` exists only because Olivetti is a benchmark. In the brief it does not exist: the only labels this project may use are the forty it pays for, and every other use of `y` below is marked AUDIT.",
            student="using `y` to evaluate, tune, or choose k, on the grounds that it is right there. The entire application is about what you can do without it.",
            catch="assert the pixel range. Olivetti arrives in [0,1] and many face datasets arrive in [0,255]; every distance in this notebook is a factor of 255 different if you assume wrong."),
        code('''
from sklearn.datasets import fetch_olivetti_faces

faces = fetch_olivetti_faces(shuffle=False)      # ~4 MB, a few seconds
X, y, images = faces.data, faces.target, faces.images

assert X.shape == (400, 4096), f"unexpected shape {X.shape}"
assert images.shape == (400, 64, 64)
assert X.min() >= 0.0 and X.max() <= 1.0, "pixels are not in [0, 1]"
assert len(np.unique(y)) == 40
assert np.bincount(y).min() == np.bincount(y).max() == 10

print(f"{len(X)} photographs of {len(np.unique(y))} people, "
      f"{np.bincount(y)[0]} each")
print(f"each photograph is {images.shape[1]}x{images.shape[2]} = "
      f"{X.shape[1]} features")
print(f"the whole corpus is {X.nbytes / 1e6:.2f} MB in memory")
'''),

        md("""
`y` exists because Olivetti is a benchmark. **In the brief it does not.** From
here on, the only labels we are allowed to use are the forty we pay for; `y` is
used exactly twice, both times flagged, both times as an audit the stakeholder
could not afford.
"""),

        md("## 3 · Look at the data. It is a face dataset — look at faces"),
        prompt(
            label="look at the data — it is faces, so look at faces",
            input="one photograph of each of the forty people",
            output="a single tiled image, ten to a row",
            constraint="tile into ONE array rather than making forty subplots — forty axes at this size is slow and each one carries its own margins",
            left_open="what you are looking for. Not 'do these look like faces' but 'what varies within a person', which the next cell answers.",
            student="skipping the picture because the shape assert passed. This is a face dataset and the single cheapest check available is whether it contains faces.",
            catch="`vmin=0, vmax=1` on the imshow. Without it matplotlib rescales each montage to its own range, and two montages become incomparable in brightness for no stated reason."),
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
'''),
        prompt(
            label="ten photographs of one person",
            input="all ten images of two different people",
            output="two rows of ten",
            constraint="show two people, not one — a single row shows variation but not whether that variation is smaller than the between-person variation",
            left_open="what the method is up against: glasses on and off, lighting from either side, eyes shut, head turned. Any method that groups these ten has to be insensitive to all of it while still separating them from the other 390.",
            student="looking at the grid of forty, seeing forty distinct faces, and concluding the task is easy. The difficulty is entirely within-person, and the forty-face montage cannot show it.",
            catch="when you look at data, look at the variation you need the model to ignore, not only at the variation you need it to see."),
        code('''
fig, ax = plt.subplots(figsize=(11, 2.6))
montage(ax, np.concatenate([images[y == 0], images[y == 22]]), ncol=10)
ax.set_title("ten photographs of one person, then ten of another")
plt.show()
'''),
        md("""
Glasses on and off, lighting from either side, eyes shut, head turned. Any
method that groups these ten together has to be insensitive to all of it —
while still separating them from the other 390.
"""),

        md("""
## 4 · Is this hard? Measure it before assuming

A photograph is a point in $\\mathbb{R}^{4096}$. k-means will group points that
are close in Euclidean distance. So the question is whether two photographs of
the same person are closer than two photographs of different people — in raw
pixels, which is the only representation we have.
"""),
        prompt(
            label="AUDIT — is this hard? measure before assuming",
            input="all 79,800 pairs of photographs",
            output="the distance distribution within a person against between people",
            constraint="this uses the hidden labels and is marked AUDIT — the stakeholder could not afford it, and it is here to tell US whether the representation can work at all",
            check="assert 1,800 within-person pairs and that the two counts sum to 400·399/2",
            left_open="the ceiling. The overlap fraction is the limit on what any distance-based method can do in this representation, and no amount of tuning k moves it.",
            student="going straight to k-means. If same-person pairs are routinely further apart than different-person pairs, the clustering result was decided before k-means ran.",
            catch="a histogram of within against between distances, before any model. It costs one cell and tells you whether you are tuning or hoping."),
        code('''
from sklearn.metrics import pairwise_distances

D = pairwise_distances(X)
iu = np.triu_indices(len(X), k=1)
same = y[iu[0]] == y[iu[1]]                      # AUDIT — uses the hidden labels
within, between = D[iu][same], D[iu][~same]

assert len(within) == 40 * (10 * 9 // 2) == 1800
assert len(within) + len(between) == 400 * 399 // 2

print(f"same person      mean {within.mean():.2f}  median {np.median(within):.2f}")
print(f"different people mean {between.mean():.2f}  median {np.median(between):.2f}")
overlap = (within > np.median(between)).mean()
print(f"\\n{100 * overlap:.0f}% of same-person pairs are further apart than the "
      f"median different-person pair")

fig, ax = plt.subplots(figsize=(9, 3))
bins = np.linspace(0, max(between.max(), within.max()), 70)
ax.hist(between, bins=bins, density=True, color="#b0bcc7", label="different")
ax.hist(within, bins=bins, density=True, color="#c0392b", alpha=0.7, label="same")
ax.set_xlabel("Euclidean distance, raw pixels"); ax.legend(); plt.show()
'''),
        md("""
The two distributions overlap heavily. Pixel distance is **partly** identity and
partly lighting and pose. Keep that number in mind: it is the ceiling on what
any distance-based method can do in this representation.
"""),

        md("""
## 5 · The forty labels

The annotator is given forty photographs, chosen at random, and asked which of
them show the same person. That is all the supervision the project has.
"""),
        prompt(
            label="the forty labels, and all the supervision there is",
            input="the corpus",
            output="forty randomly chosen indices and the annotator's answers for them",
            constraint="choose WITHOUT replacement and sort — a repeated index means paying twice for one answer",
            check="assert forty indices and forty distinct ones",
            left_open="how few pairs that is. Forty photographs give 780 pairs, and only a handful are the same person — every ARI in this notebook is computed from that handful.",
            student="drawing the forty stratified by person, which is a much better sample and requires knowing the answer in advance. The annotator is handed random photographs because nobody knows who is in them.",
            catch="print how many distinct people the forty happen to cover. If the sample misses a person entirely, no clustering can be rewarded for finding them."),
        code('''
audit = np.sort(rng.choice(400, size=40, replace=False))
y_audit = y[audit]                               # the annotator's answers

assert len(audit) == 40 and len(np.unique(audit)) == 40

same_pairs = sum(y_audit[i] == y_audit[j]
                 for i in range(40) for j in range(i + 1, 40))
print(f"{40 * 39 // 2} pairs among the labelled forty, "
      f"{same_pairs} of which are the same person")
print(f"distinct people seen at least once: {len(np.unique(y_audit))}")
'''),

        md("""
## 6 · Two metrics, and what each is for

**Silhouette** needs no labels at all. For a point $i$ with mean distance $a$ to
its own cluster and mean distance $b$ to the nearest other cluster,

$$s_i = \\frac{b - a}{\\max(a, b)} \\in [-1, 1].$$

It is a *model-selection* criterion: it can be computed on the whole corpus and
so can choose $k$.

**Adjusted Rand index** compares two groupings pair by pair, corrected so that a
random grouping scores 0. We can only compute it on the forty labelled
photographs — which is the number the stakeholder signs off on.
"""),

        md("""
## 7 · The trivial baseline

*Rule 2 of this course: a metric with nothing to compare it to is decoration.*

Before committing to anything, measure what **no method at all** scores: throw
every face into a uniformly random cluster.

⏱ **about 1 minute** — twenty seeds, and each silhouette needs the full
400 × 400 distance matrix in 4,096 dimensions.
"""),
        prompt(
            label="⏱ 1 min — what nothing scores",
            input="uniformly random cluster assignments, twenty seeds",
            output="silhouette and ARI for random assignment, with their spread",
            constraint="twenty seeds, and report the STANDARD DEVIATION — the anchor is not a point, it is a range, and the range is what tells you whether 0.02 is a discovery",
            left_open="that ARI is corrected for chance by construction, so its zero is structural, while the silhouette's zero here is measured. Both land at zero and only one of them had to.",
            student="assuming a silhouette of 0.15 must be meaningful because it is positive. Positive against what — this cell is the 'what'.",
            catch="every unsupervised metric needs an empirical null. They have no natural scale, and a number with no null is a number with no units."),
        code('''
from sklearn.metrics import adjusted_rand_score, silhouette_score

def random_assignment_scores(k, n_seeds=20):
    sil, ari = [], []
    for seed in range(n_seeds):
        r = np.random.default_rng(1000 + seed)
        lab = r.integers(0, k, size=len(X))
        sil.append(silhouette_score(X, lab))
        ari.append(adjusted_rand_score(y_audit, lab[audit]))
    return np.array(sil), np.array(ari)

for k in (10, 40):
    s, a = random_assignment_scores(k)
    print(f"k={k:3d}  silhouette {s.mean():+.4f} ± {s.std():.4f}"
          f"   ARI on the 40 {a.mean():+.4f} ± {a.std():.4f}")
'''),
        md("""
Both anchors sit at zero, and now you know how far from zero is noise. A
silhouette of 0.02 is not a discovery.
"""),

        md("""
## 8 · Commit

**Stop. On paper, now.** Not in this notebook — on paper, where you cannot
quietly revise it.

```
Metric the stakeholder signs off on:  ARI on the 40 labelled photographs
ARI a good system would reach:        ____________
ARI I expect from what we build today: ____________
Silhouette I expect at the k we pick:  ____________
```

Random assignment scores 0.00 on both. Perfect agreement scores 1.00. Your
number belongs somewhere in between, and saying *where* is the exercise.
"""),

        md("""
## 9 · k-means, and choosing k by the elbow

The textbook's first suggestion: plot inertia — the sum of squared distances
from each point to its centroid — against $k$, and look for the bend.

⏱ **3–6 minutes.** Every $k$ is `n_init` independent runs of Lloyd's algorithm
on 400 × 4,096 floats. This is the cost the next lecture removes.
"""),
        prompt(
            label="⏱ 3-6 min — the sweep",
            input="nine values of k",
            output="inertia, silhouette and ARI at each, and the wall clock",
            constraint="record the elapsed time — this notebook ends by quoting it, and it is the reason the next lecture exists",
            check="assert inertia is non-increasing in k — a finer partition cannot have larger inertia, so if it does, `n_init` is too small and Lloyd's algorithm landed in a bad local minimum",
            left_open="that `n_init=5` is a compromise. Each k is five independent runs on 400×4,096 floats, and the monotonicity assert is what tells you whether five was enough.",
            student="`n_init=1` for speed, getting a jagged inertia curve, and looking for an elbow in what is mostly optimiser noise.",
            catch="the monotonicity assert is a property of the OBJECTIVE, not of the algorithm. When it fails, the algorithm failed to optimise, and that is worth knowing before you read the curve."),
        code('''
from sklearn.cluster import KMeans

ks = [2, 5, 10, 15, 20, 30, 40, 50, 60]
inertia, silhouette, ari40, labels = [], [], [], {}

t0 = time.perf_counter()
for k in ks:
    km = KMeans(n_clusters=k, n_init=5, random_state=RANDOM_STATE).fit(X)
    inertia.append(km.inertia_)
    silhouette.append(silhouette_score(X, km.labels_))
    ari40.append(adjusted_rand_score(y_audit, km.labels_[audit]))
    labels[k] = km.labels_
    print(f"k={k:3d}  inertia {km.inertia_:8.1f}  "
          f"silhouette {silhouette[-1]:+.4f}  ARI40 {ari40[-1]:+.4f}", flush=True)
sweep_seconds = time.perf_counter() - t0
print(f"\\n{sweep_seconds:.0f} seconds for {len(ks)} values of k")

assert len(inertia) == len(ks)
assert all(inertia[i] >= inertia[i + 1] for i in range(len(ks) - 1)), \\
    "inertia must be non-increasing in k — if it is not, n_init is too small"
'''),
        prompt(
            label="find the elbow",
            input="inertia against k",
            output="the curve",
            constraint="plot it and say nothing — the next cells are about what you cannot read off it",
            left_open="where the elbow is. Ask three people and get three answers, and that is a statement about the curve rather than about people.",
            student="picking the bend by eye, confidently. Inertia is monotonically decreasing by construction, reaches exactly zero at k = n, and nothing in it knows how many people are in the archive.",
            catch="a criterion with no interior optimum cannot select a model. If your selection rule is 'look for the bend', you are the selection rule."),
        code('''
fig, ax = plt.subplots(figsize=(8, 3))
ax.plot(ks, inertia, "o-", color="#0b3d62", lw=2)
ax.set_xlabel("k"); ax.set_ylabel("inertia")
ax.set_title("find the elbow")
plt.show()
'''),
        md("""
### Where is the elbow?

Ask three people and get three answers. That is not a joke about human
disagreement — it is a statement about the curve. Inertia is monotonically
decreasing in $k$ by construction (a finer partition cannot have larger inertia),
it reaches exactly zero at $k = n$, and nothing in it knows how many people are
in the archive.

Quantify the disagreement rather than asserting it:
"""),
        prompt(
            label="quantify the disagreement",
            input="the inertia curve",
            output="the k that the kneedle rule picks, beside the truth",
            constraint="rescale BOTH axes to [0,1] first — the rule is about the furthest point below the chord, and that is meaningless while one axis runs to 60 and the other to thousands",
            left_open="that the truth is k = 40 and we are not supposed to know it. The cell prints it as a scoreboard, not as an input.",
            student="implementing kneedle on raw axes, where the answer is decided entirely by the units of inertia.",
            catch="when a heuristic disagrees with the truth by this much, the lesson is not to find a better heuristic. It is that inertia does not contain the answer."),
        code('''
# the "kneedle" rule: rescale both axes to [0, 1], then take the point furthest
# below the straight line joining the two ends
kn = (np.array(ks) - min(ks)) / (max(ks) - min(ks))
inn = (np.array(inertia) - min(inertia)) / (max(inertia) - min(inertia))
drop = (1 - kn) - inn
print(f"the kneedle rule picks k = {ks[int(np.argmax(drop))]}")
print(f"the truth, which we are not supposed to know, is k = 40")
'''),

        md("""
## 10 · Silhouette, which has a maximum

Inertia has no interior optimum. The silhouette does: it is penalised both for
splitting a group and for merging two.
"""),
        prompt(
            label="silhouette, which has a maximum",
            input="the silhouette at each k",
            output="the curve, with the random-assignment level and the true k marked",
            constraint="draw the zero line — it is the anchor from section 7, and without it the curve has no scale",
            left_open="why it has an interior optimum at all. The silhouette is penalised both for splitting a group and for merging two; inertia is only penalised for one of those.",
            student="reading the peak as the answer. It is a better criterion than inertia and it still peaks some distance from 40, which is the honest result.",
            catch="mark the truth on the plot when you have it, even though the method may not use it. A criterion that peaks in the wrong place is a finding worth seeing."),
        code('''
fig, ax = plt.subplots(figsize=(8, 3))
ax.plot(ks, silhouette, "o-", color="#0b3d62", lw=2, label="k-means")
ax.axhline(0, color="#4b5563", ls=":", lw=2, label="random assignment")
ax.axvline(40, color="#14663a", ls="--", lw=2, label="the true 40")
ax.set_xlabel("k"); ax.set_ylabel("mean silhouette"); ax.legend()
plt.show()

best_k = ks[int(np.argmax(silhouette))]
print(f"best silhouette {max(silhouette):.4f} at k = {best_k}")
print(f"silhouette at the true k = 40: {silhouette[ks.index(40)]:.4f}")
'''),

        md("""
## 11 · The silhouette diagram

The mean hides the shape. Draw every point's coefficient, sorted, grouped by
cluster: one knife per cluster. A healthy clustering has knives of similar
length, all of them reaching past the dashed mean.
"""),
        prompt(
            label="the diagram, because the mean hides the shape",
            input="the per-point silhouette at three values of k",
            output="one knife per cluster, sorted, with the mean drawn across",
            constraint="sort the clusters by their own mean and sort the points inside each — an unsorted diagram is noise with a colour map",
            left_open="what healthy looks like: knives of similar length, all of them reaching past the dashed mean. A cluster entirely to the left of the mean is a cluster that should not exist.",
            student="reading only the mean silhouette. Two clusterings with the same mean can be one good partition and one that merged half the corpus into a single blob.",
            catch="`sharex=True` across the panels. Three silhouette diagrams on independent x-axes cannot be compared, which is the only reason to draw three."),
        code('''
from sklearn.metrics import silhouette_samples

fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharex=True)
for ax, k in zip(axes, [2, 10, 40]):
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

        md("""
## 12 · Now look at the faces

A silhouette of 0.15 means nothing until you see what it grouped. This is the
whole advantage of a face dataset, and it is the step that gets skipped.
"""),
        prompt(
            label="AUDIT — now look at the faces",
            input="the k=40 clustering and the hidden labels",
            output="cluster sizes, purity, and montages of the cleanest and worst clusters",
            constraint="show the WORST cluster, not only the best — a montage of the cleanest cluster is a marketing image",
            left_open="what the worst cluster is made of. The faces in it are not similar PEOPLE — they are similar PHOTOGRAPHS: same lighting, same head angle. In 4,096 raw pixels a lamp on the left is a bigger vector than a different nose.",
            student="reporting the silhouette and stopping. A silhouette of 0.15 means nothing until you see what it grouped, and this is the step that gets skipped.",
            catch="restrict the best/worst search to clusters with at least a few members. A cluster of one has purity 1.00 and tells you nothing."),
        code('''
lab = labels[40]
sizes = np.bincount(lab, minlength=40)
purity = np.array([np.bincount(y[lab == c]).max() / max((lab == c).sum(), 1)
                   for c in range(40)])          # AUDIT — uses the hidden labels

print(f"cluster sizes: min {sizes.min()}, max {sizes.max()}, "
      f"{(sizes == 10).sum()} of 40 have exactly 10 members")
print(f"clusters containing exactly one person: "
      f"{sum(len(np.unique(y[lab == c])) == 1 for c in range(40))} of 40")

big = [c for c in range(40) if sizes[c] >= 5]
best, worst = max(big, key=lambda c: purity[c]), min(big, key=lambda c: purity[c])

fig, axes = plt.subplots(2, 1, figsize=(11, 3))
for ax, c, tag in ((axes[0], best, "cleanest"), (axes[1], worst, "worst")):
    montage(ax, images[lab == c][:10], ncol=10)
    ax.set_title(f"{tag} cluster: {sizes[c]} photographs, "
                 f"{len(np.unique(y[lab == c]))} people, purity {purity[c]:.2f}")
plt.tight_layout(); plt.show()
'''),
        md("""
Look at the worst cluster before you read on. The faces in it are not similar
*people* — they are similar **photographs**: same lighting, same head angle. In
4,096 raw pixels, a lamp on the left is a bigger vector than a different nose.
"""),

        md("""
## 13 · An assistant chooses k for us

Here is a real request and the code it returns. **⚠ Read before running.** It
runs, it uses no exotic import, and it prints a believable number.

> *"Cluster the faces with k-means, pick the best number of clusters using the
> silhouette score, and report the score."*
"""),
        prompt(
            label="⚠ what the assistant returns",
            input="'cluster the faces, pick the best k by silhouette, and report the score'",
            output="the winning k and its silhouette",
            constraint="run it exactly as written — no test set is touched, because there is no test set, and the defect is subtler than that",
            left_open="that the score reported is a MAXIMUM over a noisy criterion, evaluated on the same data that chose it. Five candidate values of k, five noisy estimates, and we print the largest.",
            student="reporting that number. It is an optimistically biased estimate of the silhouette the chosen model would obtain on new photographs — for exactly the reason the previous application's grid search could not report its own best score.",
            catch="the rule from application 1 in a new costume: the number that chose the model cannot also be the number that reports it. It applies without any labels anywhere in sight."),
        code('''
best_score, best_k_reported, best_model = -2, None, None
for k in [5, 10, 20, 40, 60]:
    km = KMeans(n_clusters=k, n_init=3, random_state=RANDOM_STATE).fit(X)
    s = silhouette_score(X, km.labels_)
    if s > best_score:
        best_score, best_k_reported, best_model = s, k, km

print(f"best k = {best_k_reported}, silhouette = {best_score:.4f}")
'''),
        md("""
### Reviewer question 5: what is the default I did not ask for?

Nothing here touched a test set, because there is no test set. The defect is
subtler and it is the one that matters for every unsupervised model selection:

**the score reported is a maximum over a noisy criterion, evaluated on the same
data that chose it.** Five candidate values of $k$, five noisy estimates, and we
print the largest. That is an optimistically biased estimate of the silhouette
the chosen model would obtain on new photographs — for exactly the reason the
previous application's grid search could not report its own best score.

Measure it. Choose $k$ on one half of the corpus, then score the chosen model on
the other half, which had no vote.

⏱ **about 2 minutes.**
"""),
        prompt(
            label="⏱ 2 min — measure the optimism",
            input="five random halves of the corpus",
            output="the selection silhouette and the held-out silhouette, per seed",
            constraint="choose k on one half and score the CHOSEN model on the other, which had no vote — and use `predict`, not a refit, so it is the same model being scored",
            left_open="that both halves are the same corpus, so this measures selection optimism and not generalisation to a new archive.",
            student="refitting on the held-out half, which measures something else entirely and usually shows no gap at all.",
            catch="five seeds and the individual numbers printed, not just the mean. The optimism is small, and a single seed cannot distinguish it from noise."),
        code('''
gaps = []
for seed in range(5):
    r = np.random.default_rng(2000 + seed)
    perm = r.permutation(len(X))
    a, b = perm[:200], perm[200:]

    chosen, chosen_km = -2, None
    for k in [5, 10, 20, 40, 60]:
        km = KMeans(n_clusters=k, n_init=3, random_state=RANDOM_STATE).fit(X[a])
        s = silhouette_score(X[a], km.labels_)
        if s > chosen:
            chosen, chosen_km = s, km
    held = silhouette_score(X[b], chosen_km.predict(X[b]))
    gaps.append(chosen - held)
    print(f"seed {seed}: selected {chosen:.4f}   held out {held:.4f}   "
          f"optimism {chosen - held:+.4f}")

print(f"\\nmean optimism {np.mean(gaps):+.4f}")
'''),
        md("""
### The corrected specification

> *"Cluster the faces with k-means over k in a stated range. Choose k by mean
> silhouette computed on a randomly held-out half of the corpus, refit the
> chosen k on everything, and report both the selection score and the held-out
> score. Fix the seed and print every k, not only the winner."*

The rule is the same one as in the first application, in a new costume: **the
number that chose the model cannot also be the number that reports it.**
"""),

        md("""
## 14 · How long did all this take?

Write it down. It is the reason the next lecture exists.
"""),
        prompt(
            label="how long did all this take",
            input="the recorded wall clock",
            output="the measured sweep time, and an extrapolation to a full sweep",
            constraint="extrapolate honestly — state that it is an estimate for THIS machine, scaled from a measured number rather than guessed",
            left_open="what to do about it. Nothing, today. This number is the reason the next lecture exists, and it is meant to be uncomfortable.",
            student="not timing anything, then being surprised when the same sweep at finer resolution runs overnight.",
            catch="every second of it was spent on 4,096 numbers per face, most of which are describing a lamp. That sentence is the whole setup for dimensionality reduction."),
        code('''
print(f"the coarse sweep over {len(ks)} values of k took "
      f"{sweep_seconds:.0f} seconds")
print(f"a full sweep over k = 2..60 at n_init=10 is roughly "
      f"{sweep_seconds * (sum(range(2, 61)) / sum(ks)) * 2 / 60:.0f} minutes "
      f"on this machine")
print("\\nand every one of those seconds was spent on 4,096 numbers per face,")
print("most of which are describing a lamp.")
'''),

        md("""
## 15 · Where we are

Add to your sheet:

```
Best k we chose:                    ____________
Silhouette at that k:               ____________
ARI on the 40 labelled photographs: ____________
Minutes the sweep took:             ____________
```

Bring the sheet to the next lecture. We open by scoring your commitment, out
loud, and then we take four thousand dimensions away and do it again.

Do not fix anything yet.
"""),
    ]
