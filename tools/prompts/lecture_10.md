# Lecture 10 — Four thousand dimensions is too many

**Rebuild-by-prompting script.** Source module: `tools/notebooks/lecture_10.py`.
Current render: `notebooks/lecture-10.ipynb`. Géron, chapters 7 & 8.
Mathematical thread: the SVD, PCA, Eckart–Young, Johnson–Lindenstrauss.

This is the **Fix** half of the Olivetti application. Lecture 9 established that
k-means on raw 4,096-dimensional faces is weak. This notebook compresses first,
then re-measures, then breaks the compression on purpose.

---

## Before you sit down

**Environment.** Everything below was executed on **Apple M4 Max, macOS 25.5,
Python 3.13.5, numpy 2.3.5, scikit-learn 1.7.2**, with
`OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=2` — the same thread cap
the setup cell sets. Every wall clock quoted as **measured** is from that
machine. Colab's free 2-vCPU box is slower; I have **not** measured it, so where
I give a Colab figure it is labelled *estimate* and you should treat it as a
budget, not a promise.

**Only one cell in this notebook takes more than 20 seconds** on the reference
machine: the k-means sweep, cell 9, at 27 s. That is worth knowing before you
start, because the current notebook's ⏱ markers say otherwise on four cells and
every one of them is wrong in the same direction (see the defect report).

**Dataset.** `fetch_olivetti_faces` downloads ~4 MB the first time and caches to
`~/scikit_learn_data/olivetti_py3.pkz`. No GPU anywhere in this notebook.

**Cold-kernel order.** Cells 1 → 17 in order. Three hard dependencies are worth
naming now, because each one produces a `NameError` rather than a wrong answer:

- `montage()` is defined in **cell 4** and used again in cells 6 and 13.
- `X_tr / X_te / y_tr / y_te` are created in **cell 6** and used in cells 10, 14,
  15 and 17.
- `d95` is created in **cell 5** and used in cells 9, 10, 14, 15, 16, 17.

**Notation.** *Held-out* = the 120 photographs in `X_te`. *ARI(all)* = adjusted
Rand index against all 400 identities. *ARI(40)* = the same index against the
forty audited labels only — lecture 9 reports **ARI(40)**, this notebook reports
**ARI(all)**, and they are different numbers on the same clustering. Say which
one you mean every time.

---

## Cell 1 — setup and the corpus

**Prompt to type:**

> Load the Olivetti faces with scikit-learn, unshuffled. Print the python and
> sklearn versions, cap the BLAS threads at 2 so timings are repeatable, and set
> a seed. Also draw 40 random indices out of the 400 with a seeded Generator —
> those are the only labels we're allowed to look at.

**Expect:** `X.shape == (400, 4096)`, `y.shape == (400,)`,
`images.shape == (400, 64, 64)`, dtype `float32`, pixel range [0, 1]. Version
lines. Forty sorted indices.

**Assert:** `X.shape == (400, 4096)` and `len(audit) == 40`.

**⏱** Under a second after the first download.

**Annotate:** short

> **Note.** The forty audit indices exist so that this notebook's ARI is
> comparable with lecture 9's. Draw them from `np.random.default_rng(42)` as the
> *first* call on that generator — the generator is stateful, so an extra `rng`
> call inserted above this line silently changes which forty you get. **Then
> actually use them.** See cell 9.

---

## Cell 2 — PCA by hand, checked against scikit-learn

**Prompt to type:**

> Centre the faces, take the SVD with numpy, and check that the right singular
> vectors match `PCA().fit(X).components_`. Also check that the explained
> variance ratio is the normalised squared singular values.

**Expect:** `U (400, 400)`, `S (400,)`, `Vt (400, 4096)` with
`full_matrices=False`. A printed disagreement of order **1e-07** on the
components and **1e-07** on the variance ratios.

**Assert:** `np.abs(np.abs(Vt[:5]) - np.abs(pca.components_[:5])).max() < 1e-4`
— measured **6.07e-07**. And
`np.abs(evr - pca.explained_variance_ratio_).max()` — measured **1.94e-07**.

**⏱** Measured **0.21 s** (SVD 0.14 s, `PCA().fit` 0.07 s) on M4 Max. Colab
estimate: 1–2 s. The current notebook says "about 20 seconds"; it is not.

**Annotate:** full

- **Left open** — that the two sides of the check are the *same computation*.
  `PCA` centres the data and calls the same LAPACK driver; the point of the cell
  is not that two libraries agree, it is that the principal components **are**
  the right singular vectors of the centred matrix, and the explained variance
  ratio **is** $\sigma_j^2 / \sum_k \sigma_k^2$. Neither is a coincidence to be
  verified; both are definitions to be recognised.
- **The usual student version** — writing the natural assertion,
  `np.abs(Vt[:5] - pca.components_[:5]).max() < 1e-4`, and watching it fail.
  Singular vectors are defined only up to a **sign**: if $(u_j, v_j)$ is a
  singular pair then so is $(-u_j, -v_j)$, and LAPACK's sign convention is
  whatever the driver produced. Measured on this data: **224 of the 400 rows**
  of `Vt` are the negation of the corresponding row of `pca.components_`, and
  the naive difference maxes out at **8.96e-02** — six orders of magnitude above
  the tolerance. The observed conclusion is "numpy and sklearn disagree", which
  is wrong; they agree exactly, up to the only ambiguity the decomposition has.
  Compare absolute values, or fix the sign yourself before comparing.
- **How you would catch it** — `full_matrices=False`, and check the shape you
  got. With 400 rows and 4,096 columns the full `U` is 4,096 × 4,096 — 134 MB of
  mostly zeros — and it is never needed. If `U.shape[1]` is not 400 you asked
  for the wrong decomposition. Then, when a sign-invariant object disagrees by
  1e-01 and a sign-invariant comparison of the same object agrees by 1e-07, the
  difference between the two numbers *is* the diagnosis.

---

## Cell 3 — Eckart–Young, to machine precision

**Prompt to type:**

> Truncate to rank 100 by rebuilding X from U, S, Vt (not with
> `inverse_transform`), and check that the squared Frobenius error equals the sum
> of the discarded squared singular values.

**Expect:** two numbers that agree to about six significant figures. Measured:
$\lVert X - X_d\rVert_F^2 =$ **2040.778442**, $\sum_{j>100}\sigma_j^2 =$
**2040.778320**, relative difference **5.98e-08**.

**Assert:** `abs(lhs - rhs) / rhs < 1e-5`.

**Annotate:** short

> **Note.** Rebuild `Xd` from the truncated factors rather than from
> `inverse_transform`, so the two sides come by different routes. The residual
> 6e-08 is float32 accumulation over 400 × 4,096 entries, not a flaw in the
> theorem. Eckart–Young says this is the **best** rank-$d$ approximation in
> Frobenius norm, not merely a good one — an identity you can assert in three
> lines, so assert it.

---

## Cell 4 — the eigenfaces

**Prompt to type:**

> Show the mean face and the first 15 principal components as 64×64 images in one
> tiled figure. Write a small `montage(ax, ims, ncol)` helper — I'll want it
> again later.

**Expect:** one figure, 16 tiles, 8 per row. Ghostly symmetric face-shaped
patterns; the first few dominated by broad left-right lighting.

**Assert:** none.

**Annotate:** short

> **Note.** Rescale each component to [0, 1] **individually** before plotting.
> Components have negative entries and no common scale, so a shared
> `vmin=0, vmax=1` clips half of every one of them to black and the figure looks
> like PCA found nothing. Plot the mean face first: the components are directions
> *away* from it and are hard to read as faces without it in the frame. Keep
> `montage` in this cell — cells 6 and 13 call it.

---

## Cell 5 — how many components do we need

**Prompt to type:**

> How many components to reach 95% and 99% of the variance? Print both, print how
> much the first component alone explains, and plot the cumulative curve with a
> line at 95%.

**Expect:** first component **23.8%**. **95% needs `d95 = 123` components**
(33× fewer than 4,096); **99% needs `d99 = 260`**. A concave curve rising
steeply then flattening.

**Assert:** `0.95 <= cum[d95-1]` and `cum[d95-2] < 0.95` — measured 0.95039 and
0.94984.

**Annotate:** short

> **Note.** `np.searchsorted(cum, 0.95) + 1`. The index where the cumulative sum
> first reaches the threshold is one less than the number of components kept;
> dropping the `+ 1` here gives 122 components at **94.98%** of the variance,
> which is quietly *below* the threshold you asked for and invisible in every
> number downstream. Print the reduction factor beside the count — "123
> components, 33× fewer than 4,096" is the sentence; the bare count is not.
> **95% and 99% are conventions, not findings**: the gap between 123 and 260 is
> more than a factor of two.

---

## Cell 6 — split, then a face with components taken away

**Prompt to type:**

> Split 70/30 with a fixed seed, then show one held-out face reconstructed at
> 1, 2, 5, 10, 25, 50, 100, 200 components and finally the original, in one row.

**Expect:** `len(X_tr) == 280`, `len(X_te) == 120`. Nine tiles. At 1–2
components a generic face; identity readable somewhere around 10–25; lighting
and background detail arriving last.

**Assert:** `len(X_tr) == 280 and len(X_te) == 120`, and
`np.bincount(y_tr).min() == 7 and np.bincount(y_te).min() == 3` — every one of
the forty people keeps at least 7 training and 3 test photographs.

**⏱** Measured **0.3 s** for all eight PCA fits on M4 Max.

**Annotate:** full

- **Left open** — *what to look for*. The prompt asks for a picture and gets
  one; it does not say what the picture is evidence for. Identity survives
  compression long before lighting does, and that is the whole answer to why
  clustering improves after PCA in cell 9. Without that sentence the figure is
  decorative.
- **The usual student version** — omitting `stratify=y`. `stratify` defaults to
  `None` and `shuffle` defaults to `True`, so the library default is a plain
  random split: nothing in scikit-learn guarantees that all forty people appear
  in both halves. With 40 classes and 10 photographs each, an unstratified
  70/30 split routinely hands somebody 0 or 1 test rows, and every accuracy in
  cells 14–17 is then measured on 39 people while claiming 40. The other half of
  the same default: `random_state` is `None`, so re-running the cell re-splits
  and every number below it moves.
- **How you would catch it** — the `bincount` assert, which costs one line and
  has an answer you can work out on paper before running it: 10 photographs per
  person, `test_size=0.3` stratified, so 7 train and 3 test each, exactly, for
  everyone. If either minimum comes back lower, the split is not the split you
  described. Also fit the PCA on `X_tr` and reconstruct a face from `X_te`:
  reconstructing a face the subspace was fitted on flatters every column of the
  figure.

---

## Cell 7 — the Johnson–Lindenstrauss bound

**Prompt to type:**

> Use `johnson_lindenstrauss_min_dim` to print the guaranteed target dimension for
> 400 points and for a million points, at eps = 0.1, 0.2, 0.3, 0.5. Print our
> actual dimension underneath.

**Expect:** exactly these, from sklearn 1.7.2 —

| eps | 400 points | 1,000,000 points | ratio |
|---|---|---|---|
| 0.1 | 5,135 | 11,841 | 2.31 |
| 0.2 | 1,382 | 3,188 | 2.31 |
| 0.3 | 665 | 1,535 | 2.31 |
| 0.5 | 287 | 663 | 2.31 |

and `4096` underneath.

**Assert:** none.

**Annotate:** short

> **Note.** Read the formula
> $d \ge 4\log n / (\varepsilon^2/2 - \varepsilon^3/3)$ and say what is missing:
> $n$ is there, $\varepsilon$ is there, and $D$ — the dimension you start in — is
> **not**. Four hundred points need the same target dimension whether they live
> in 4,096 dimensions or four million. Second: 2,500× more points costs **2.31×**
> the dimension, identically at all four eps, because the bound is linear in
> $\log n$ and $\log(10^6)/\log(400) = 2.31$. Third, and this is the one that
> surprises everybody: at eps = 0.1 the bound for our 400 faces is **5,135 —
> larger than the 4,096 we started with**. It is a worst-case guarantee over all
> point sets, and ours is not the worst case. So measure.

---

## Cell 8 — what the distortion actually is

**Prompt to type:**

> Take all pairwise distances between the 400 faces, project with
> `GaussianRandomProjection` to d = 50, 100, 200, 400, 800, and report the worst
> and the 95th-percentile relative distortion at each d. Average over 3 seeds.

**Expect:** **79,800** pairs. Measured, mean over seeds 0/1/2 —

| d | worst pair | 95th pct | mean |
|---|---|---|---|
| 50 | 0.438 | 0.200 | 0.082 |
| 100 | 0.306 | 0.137 | 0.056 |
| 200 | **0.252** | 0.105 | 0.043 |
| 400 | 0.147 | 0.070 | 0.029 |
| 800 | 0.107 | 0.047 | 0.020 |

**Assert:** `len(D0) == 400 * 399 // 2 == 79800`.

**⏱** Measured **0.6 s** on M4 Max — 15 projections plus 16 distance matrices of
400 × 400. The current notebook says "about 1 minute"; it is not.

**Annotate:** full

- **Left open** — *which statistic answers the theorem*. The prompt asks for two
  columns and does not say which one carries the argument. Johnson–Lindenstrauss
  bounds the **worst** pair; a percentile is a different claim about a different
  quantity, and the numbers above show the gap is a factor of 2.4 at d = 200.
- **The usual student version** — reading the percentile and writing the
  conclusion about the worst case. This is not hypothetical: **the current
  lecture-10 notebook does exactly this.** Its prose says "at $d = 200$ the worst
  distance in the whole corpus is distorted by far less than the
  $\varepsilon = 0.2$ the bound would only guarantee at 1,382 dimensions." The
  measured worst pair at d = 200 is **0.252, which is larger than 0.20** — the
  claim is false as written and true of the 95th percentile (0.105), which is
  the column beside it. The sentence becomes true at **d = 400** (worst 0.147).
  The prompt box directly above it says, correctly, "report the WORST pair as
  well as the percentile — a percentile alone does not test it", and then the
  prose reads the percentile anyway.
- **How you would catch it** — three seeds, not one. A single draw of a random
  projection is one sample from the distribution the theorem quantifies over, and
  a max over 79,800 pairs is exactly the statistic with the heaviest seed-to-seed
  tail. Then, before writing any sentence containing the word "worst", put your
  finger on the *worst* column. Here the two columns disagree about the
  conclusion and the prose picked the wrong one.

---

## Cell 9 — compress, then cluster

**Prompt to type:**

> Run the same k-means sweep as last week — k in 2, 5, 10, 15, 20, 30, 40, 50, 60,
> `n_init=5`, same seed — on the raw 4,096-dim faces and on the d95 PCA
> projection. For each, report the best silhouette, its k, the ARI, and how long
> the sweep took.

**Expect:** `Z95.shape == (400, 123)`. Measured —

| space | wall clock | best k | silhouette | ARI(all) | ARI(40) |
|---|---|---|---|---|---|
| 4,096 dims | 27.3 s | 60 | 0.1695 | 0.454 | 0.336 |
| 123 dims | 0.1 s | 60 | 0.2061 | 0.521 | 0.436 |

Speed-up **187×**.

**Assert:** `Z95.shape == (400, d95)` before sweeping it.

**⏱** Measured **27.4 s total** on M4 Max — and essentially all of it is the
**raw** sweep (27.3 s), not the reduced one (0.1 s). Colab estimate: 2–4 minutes
for the raw half, still under a second for the reduced half. This is the only
cell in the notebook that needs a ⏱ marker at all.

**Annotate:** full

- **Left open** — *which rows the ARI is computed on*. This is the one that
  matters, because the prompt says "the same sweep as last week" and invites a
  before-and-after comparison. Lecture 9's sweep prints
  `adjusted_rand_score(y_audit, km.labels_[audit])` — **40 rows**, the audited
  labels, because forty labels is all the supervision the project bought. This
  notebook prints `adjusted_rand_score(y, lab)` — **400 rows**, every identity.
  On the identical raw clustering those are **0.336 and 0.454**. Comparing this
  notebook's 0.454 against last week's number is a comparison of denominators.
  Print both columns, or print ARI(40) and say so.
- **The usual student version** — reporting the silhouette rise, 0.1695 → 0.2061,
  as the result. The silhouette is computed **in the space you hand it**: before
  compression it measures distances in 4,096 dimensions, after compression in
  123. Those are not the same function, so the two numbers are not comparable and
  the rise is not evidence that the clustering improved. The ARI is measured
  against identities and does not depend on the representation — it moved the
  same way (0.454 → 0.521 on all rows, 0.336 → 0.436 on the forty), and *that*
  is the result. The second observed failure here: **`best k = 60` is the top of
  the grid.** Extend it and the silhouette keeps climbing — 0.2167 at k = 70,
  0.2385 at k = 80, 0.2404 at k = 100 — so 60 was not selected, it was where the
  loop stopped. ARI(all), meanwhile, peaks at k = 60 (0.521) and falls to 0.472
  at k = 70. The grid is concealing a disagreement, not resolving one.
- **How you would catch it** — change exactly one thing. Same k grid, same
  `n_init`, same seed, same metric; only the matrix differs. Then, whenever a
  metric moves because its **input space** changed, report a second metric that
  does not depend on the representation. And whenever an argmax lands on the
  first or last element of a grid, extend the grid before you report it as a
  choice.

---

## Cell 10 — four ways to reduce, timed and scored

**Prompt to type:**

> Compare four reducers to d95 dimensions on the training faces: PCA with the full
> SVD, PCA randomised, `IncrementalPCA`, and `GaussianRandomProjection`. Print the
> fit time and the held-out reconstruction MSE for each.

**Expect:** measured on M4 Max, fit on the 280 training faces, error on the 120
held-out ones —

| reducer | fit | held-out MSE |
|---|---|---|
| PCA, full SVD | 48 ms | 0.00240 |
| PCA, randomised | 43 ms | 0.00241 |
| Incremental PCA | 51 ms | 0.00240 |
| Random projection | 6 ms | **0.32093** |

**Assert:** none, but check the shapes: every `transform(X_te)` is (120, 123).

**Annotate:** full

- **Left open** — *what the reconstruction column costs you*. Random projection
  is ~8× faster to fit and its held-out error is **134× larger**, because it
  never looks at the data and so cannot be looking for the subspace faces
  actually occupy. Which reducer you want depends on whether you need the
  **representation** (all four are usable) or the **reconstruction** (only the
  PCAs are). The table exists so you choose deliberately; the timing column
  alone says "random projection wins", and it does not.
- **The usual student version** — setting `IncrementalPCA(n_components=123)` with
  a small `batch_size` and being unable to read the error. `IncrementalPCA` fits
  each batch, and a batch with fewer rows than `n_components` cannot determine
  123 directions. What you actually get, measured with `batch_size=64` on the 280
  training faces, is **not** a message about batches:

  `ValueError: Number of input features has changed from 64 to 123 between calls to partial_fit! Try setting n_components to a fixed value.`

  The first batch silently reduced `n_components_` to 64, and the failure
  surfaces on the *second* batch as a phantom complaint about the number of
  features — which the reader has not changed. The rule is `batch_size >=
  n_components`; `max(2 * d95, 256)` satisfies it here. Note that the *default*
  is safe: `batch_size=None` becomes `5 * n_features = 20,480`, one batch for all
  280 rows. The trap is only reachable by setting `batch_size` yourself, which is
  the entire reason anyone uses `IncrementalPCA`.
- **How you would catch it** — assert `batch_size >= n_components` before you
  fit, and check `obj.n_components_` after: if it does not equal what you asked
  for, the estimator quietly gave you something smaller. Separately, do not
  assume the error columns are computed the same way for all four rows — check
  which branch runs. In sklearn 1.7.2 `GaussianRandomProjection` **does** have
  `inverse_transform` (it has since 1.1), so a `hasattr` branch that expects to
  fall through to a manual pseudo-inverse never fires. Both routes give 0.32093
  here, so nothing is wrong with the number — but the branch is dead code and the
  current notebook's prose describes the branch that does not run.

---

## Cell 11 — DBSCAN over a grid of eps

**Prompt to type:**

> Run DBSCAN on the reduced faces over 25 values of eps from 2 to 14 with
> `min_samples=3`. Report the best ARI, the eps that gave it, the number of
> clusters excluding noise, and how many faces were called noise.

**Expect:** best **ARI 0.133 at eps = 6.00**, with **41 clusters** and **97 of
400 faces labelled noise**.

**Assert:** none. Count clusters as `len(set(lab) - {-1})` — DBSCAN uses `-1`
for noise and counting it inflates every k you report.

**⏱** Measured 0.3 s.

**Annotate:** short

> **Note.** The right answer is 40 clusters and 0 noise. Report the **noise
> count** next to the ARI: a method that scores by declaring a quarter of the
> corpus unclassifiable has not solved the brief. Careful with the sentence you
> write — at the best-ARI eps DBSCAN returns **41** clusters, which is nearly
> right, so "it never finds 40 clusters" is not the finding. The finding is ARI
> 0.133 against k-means' 0.521 on the same rows, with 97 faces discarded: faces
> in this subspace have no density scale that separates people.

---

## Cell 12 — the full covariance, ruled out by arithmetic

**Prompt to type:**

> Before fitting any Gaussian mixture: how many free parameters does one full
> 4096-dimensional covariance have, and forty of them? Then sweep a diagonal
> mixture over k = 5 to 80 in steps of 5 on the reduced faces and print BIC and
> ARI for each.

**Expect:** one full covariance = $4096 \times 4097 / 2 =$ **8,390,656**
parameters; forty of them = **335,626,240**, estimated from 400 photographs.
Then 16 rows of BIC and ARI. Measured: BIC rises **monotonically** from 56,468
at k = 5 to 119,175 at k = 80; ARI rises to **0.548 at k = 55** (0.458 at
k = 40).

**Assert:** none.

**⏱** Measured 1.1 s.

**Annotate:** short

> **Note.** Do the parameter count *before* fitting: 335 million parameters from
> 400 photographs is not a resource problem you can wait out, it is an
> underdetermined estimate, and one line of arithmetic settles what an afternoon
> of `covariance_type="full"` cannot. Be precise about which number is which —
> **8.39 million is one covariance, 335 million is forty**; the notebook's prompt
> box attaches "335 million" to a single component. And read the BIC column
> before quoting `ks[argmin(bic)]`: BIC here is strictly increasing over the whole
> grid, so the argmin is always **k = 5, the first point on the grid**. BIC is not
> "peaking somewhere else"; it is not peaking. A criterion with no interior
> optimum has not selected anything.

---

## Cell 13 — anomaly detection, two ways

**Prompt to type:**

> Plant 12 corrupted faces in the corpus — four rotated, four dimmed and mirrored,
> four double-exposed. Fit PCA at 99% variance and a 40-component diagonal
> mixture on the clean 400, then score all 412. For each detector, how many of the
> 12 land in its top 12? Show both top-10 rows as images.

**Expect:** `Xa.shape == (412, 4096)`, `is_bad.sum() == 12`, and
`p99.n_components_ == 260`. Measured: **reconstruction error 12 of 12**;
**lowest mixture density 6 of 12**.

**Assert:** `Xa.shape == (412, 4096) and is_bad.sum() == 12`.

**⏱** Measured 0.2 s.

**Annotate:** full

- **Left open** — *why reconstruction error is a detector at all*. It is a
  different signal from low density: a face the subspace cannot rebuild is a face
  unlike the ones that built the subspace. Density asks "is this where the faces
  are?"; reconstruction asks "is this in the span of the faces?" The cell shows
  they are not the same question, and the prompt does not say so.
- **The usual student version** — fitting on `Xa`, the contaminated array,
  because `Xa` is the thing being scored. The twelve corrupted faces then help
  choose the subspace that is supposed to fail on them, and the detector's own
  training data has taught it that rotated faces are normal. Fit on the clean 400,
  score the 412. The same discipline as cell 6, in a costume where it is easier to
  miss because there is no `train_test_split` in the cell to remind you.
- **How you would catch it** — break the score down **by kind of corruption**,
  not just by count, and the reason is visible in the numbers. Reconstruction
  error finds 4 rotated, 4 dimmed, 4 double-exposed — all three. Mixture density
  finds 4 rotated and 2 double-exposed and **0 of the 4 dimmed**: multiplying a
  face by 0.35 moves it towards the origin, which the diagonal mixture reads as
  unremarkable rather than as unlike a face. "6 of 12" hides a detector that is
  blind to an entire failure mode. Aggregate counts over heterogeneous
  contamination are averages over things you are trying to tell apart.

---

## Cell 14 — spending the forty labels

**Prompt to type:**

> Budget: 40 labels out of 280 training faces. Compare four ways to spend them,
> all with the same logistic regression on the d95 PCA of the training set, all
> scored on the same held-out 120: (a) 40 at random, (b) 40 chosen as the face
> nearest each of 40 k-means centroids, (c) propagate each representative's label
> to its whole cluster, (d) propagate only to the closest 75% of each cluster.
> Print the accuracy with all 280 true labels as a ceiling.

**Expect:** measured, same classifier and same 120 held-out rows throughout —

| labels spent | held-out accuracy |
|---|---|
| 40 at random | 0.433 |
| 40, one per cluster | **0.633** |
| propagated to the cluster (280 rows) | 0.525 |
| propagated to closest 75% (206 rows) | 0.558 |
| all 280 true labels | 0.975 |

**Assert:** `len(rep) == 40 and len(np.unique(rep)) == 40` — forty distinct
faces.

**⏱** Measured 0.3 s.

**Annotate:** short

> **Note.** Nothing here is a better classifier: same estimator, same features,
> same held-out rows. The whole difference is **which** forty were labelled, and
> the ratio is **0.633 / 0.433 = 1.46×** — a real gain for zero extra annotation
> budget, but *not* "several times", which is what the current notebook's prose
> claims. Two things worth printing that the prompt does not ask for: the random
> forty cover only **26 of the 40 people**, the cluster representatives cover
> **33**, which is most of the mechanism; and propagation actually **loses**
> accuracy here (0.525 < 0.633) because the propagated labels are only **62.9%**
> correct. Always print the all-280 ceiling — a semi-supervised number with no
> fully-supervised comparison cannot be read.

---

## Cell 15 — what the assistant returns

**Prompt to type:**

> Reduce the faces to 95% of the variance with PCA and train a classifier; report
> the held-out accuracy and the reconstruction error.

**Expect:** two numbers. **Write them both down before you go on.**

**Assert:** none.

**⏱** Measured 0.1 s.

**Annotate:** short

> **Note.** This is the specification as a person would actually type it: one
> sentence, two deliverables, no methodology. Run it, record both numbers, and
> move to cell 16.

---

## Cell 16 — ⚠ measure both consequences

Now open the trap. The cell you just ran was
`PCA(n_components=d95).fit(X)` — **all four hundred photographs**, including the
hundred and twenty it then calls held out. The subspace those faces are
projected onto was chosen partly *by* those faces. With 400 points in 4,096
dimensions that is not a rounding error: the 280 training faces span at most a
**279**-dimensional subspace, and adding the test faces changes which directions
survive.

Reviewer question 2 — *what was fitted, and on what?* — catches it in one
reading. `fit` and `transform` are different verbs.

**Prompt to type:**

> Over 20 different stratified splits, fit PCA two ways — on the training faces
> only, and on all 400 — and for each report the held-out reconstruction error and
> the held-out classifier accuracy. Print the means and how often the leaky one
> wins.

**Expect:** measured over seeds 0–19 —

| quantity | honest | leaky | verdict |
|---|---|---|---|
| held-out reconstruction MSE | 0.00243 | 0.00096 | leaky looks **61%** better, in **20/20** splits |
| held-out accuracy | 0.974 | 0.974 | difference **−0.00 points**, sd 0.46, leaky wins **3/20**, ties **14/20** |

**Assert:** `(el < eh).all()` — the leak improves the reconstruction error on
every single split. Nothing analogous holds for accuracy.

**⏱** Measured **6.2 s** on M4 Max (20 splits × 2 PCA fits × 2 logistic
regressions). Colab estimate: 30–60 s. The current notebook says "2–4 minutes";
it is not.

**Annotate:** full

- **Left open** — *the decision rule*, which is the entire payoff and which no
  prompt of this shape will produce on its own. The damage from fitting an
  unsupervised step on everything is large exactly when **that step's own output
  is the thing you report**. Reconstruction error is that thing: it is 61% better
  under the leak, one-sided, on 20 splits out of 20. Downstream accuracy usually
  is not: here the leaky arm is *not even better on average*, it wins 3 splits
  out of 20 and ties 14. Both results are real and they point opposite ways.
- **The usual student version** — measuring only the accuracy, finding −0.00
  points against a split-to-split spread of **1.64 points** (honest accuracy
  ranges 91.7% to 100.0% across the 20 seeds), and concluding the leak is
  harmless. It is harmless *for that number on this dataset*. The same code with
  reconstruction error as the deliverable overstates the result by a factor of
  2.5. **You cannot tell which case you are in without splitting** — which is the
  argument for splitting even when you expect the leak not to matter, and it is
  an argument that does not depend on the effect size going your way.
- **How you would catch it** — go through the notebook and list every
  unsupervised step — scaler, imputer, PCA, encoder, vectoriser — and for each
  one name the rows it was fitted on. Then check the hyperparameters too, not
  just the fits. **This cell's own "honest" arm fails that second check:** it
  passes `n_components=d95`, and `d95 = 123` came from a PCA fitted on all 400
  faces in cell 5. Recomputed honestly from the 280 training faces alone, 95% of
  the variance needs **104** components (102–105 across the 20 seeds), not 123.
  The honest arm is fitted on training rows and *sized* by the test rows. That is
  a smaller leak than the one being demonstrated, in the cell that demonstrates
  it, and no ⚠ marks it.

---

## Cell 17 — the structural fix

**Prompt to type:**

> Put the PCA and the logistic regression in one `Pipeline` and score it on the
> held-out faces.

**Expect:** **0.975** — identical to the single-split numbers in cells 14 and 15,
because on this dataset the leak does not move the accuracy. That is the point:
the pipeline is not buying you a better number, it is buying you a guarantee.

**Assert:** none.

**Annotate:** short

> **Note.** The corrected specification is: *"split first, stratified by
> identity, fixed seed; fit PCA on the training faces only and `transform` the
> held-out ones; put it in a `Pipeline` so cross-validation refits it per fold;
> report both the reconstruction error and the accuracy with their spread over
> seeds."* The difference between **avoided** and **impossible** is the whole
> lesson: a pipeline does not make you less likely to leak, it makes leaking
> structurally unavailable, because `cross_val_score` refits every step inside
> every fold. The discipline that depends on remembering is the discipline that
> fails under deadline. The list of unsupervised steps that are safe to fit
> outside the pipeline is empty.

---

## Exercises, with the cells to re-run

**§7.2 — every exercise below names the cells and the order.** All timings are
M4 Max measured; on Colab budget several times each.

1. **Fix the honest arm of the leak experiment.** In cell 16, replace
   `n_components=d95` in the honest PCA with a `d95` recomputed inside the loop
   from `Xtr` alone. Re-run **cell 16 only** (it re-splits internally and depends
   on nothing above it except `X`, `y` and `d95`). ~15 s. Does the honest
   reconstruction error rise or fall, and by how much? Say why before you run it.
2. **Make the sweep's `best k` mean something.** In cell 9, extend `ks` to
   include 70, 80, 100. Re-run **cell 9 only** (~60 s, the raw sweep dominates).
   Report where the silhouette peaks and where ARI(all) peaks. They are not the
   same k; which one would you have used to choose the model, and which to report
   it?
3. **Match the denominators.** Add an ARI(40) column to cell 9 using `audit` from
   cell 1: `adjusted_rand_score(y_audit, lab[audit])`. Re-run **cell 1, then cell
   9** — cell 1 must come first because `audit` must be the first draw from the
   seeded generator. ~30 s. Compare with the ARI you wrote down in lecture 9.
4. **Find the ε where the notebook's sentence becomes true.** Re-run **cell 8
   only** (~1 s) with `d` extended to 300. At which d does the *worst* pair first
   fall below 0.20? (Measured here: not at 200, where it is 0.252; it is below by
   400, where it is 0.147.)
5. **Break `IncrementalPCA` on purpose.** In cell 10, change `batch_size` to 64.
   Re-run **cell 10 only** (~1 s). Write down the error message verbatim, then
   explain why it mentions features rather than batches.
6. **Which corruption does density miss?** In cell 13, print the detector hits
   broken down by `kinds` rather than as a total. Re-run **cell 13 only** (~1 s).
   One detector is blind to one of the three corruptions — say which, and why
   that corruption in particular.

---

## What is examinable

- §2 SVD, PCA, Eckart–Young — **examinable**.
- §3 Johnson–Lindenstrauss, the statement and what it omits — **examinable**.
- §5 the four reducers, and randomised vs incremental SVD — **beyond the book,
  for context**; the trade-off (representation vs reconstruction) is examinable,
  the implementations are not.
- §6 DBSCAN, Gaussian mixtures, BIC — **examinable**.
- §7 anomaly detection by reconstruction error — **examinable**.
- §8 label propagation — **examinable**.
- §9 the leak, and the decision rule — **examinable, and the point of the
  lecture**.
- Setup, thread pinning, `montage()`, timing harnesses — **not examinable,
  engineering**.

---

## Defects found in the current notebook

`notebooks/lecture-10.ipynb`, 55 cells (17 code, 38 markdown), rendered from
`tools/notebooks/lecture_10.py`. Everything below was **executed** against the
notebook's own code and data on the reference machine unless marked otherwise.
Cell numbers are zero-based notebook indices.

### Verified by execution

**D1 — `d95` is 123, not 118 (§1.1).** The prompt box above cell 15 instructs:
*"'95% needs 118 components, 35x fewer than 4,096' is the sentence"*. Running the
notebook's own two lines gives **`d95 = 123`** and **33×**, not 118 and 35×. Both
figures in the same quoted sentence are wrong, and the sentence is presented to
the student as the model of how to report the result. `d99 = 260`.

**D2 — the section 3 conclusion is contradicted by the cell it concludes
(§1.1, §2.1).** The markdown after cell 23, and the `left_open` bullet in the
prompt box before it, both state: *"At $d = 200$ the worst distance in the whole
corpus is distorted by far less than the $\varepsilon = 0.2$ the bound would only
guarantee at 1,382 dimensions."* Measured from that cell, mean over its own three
seeds: **worst pair at d = 200 is 0.252**, which is *greater* than 0.20. The
95th percentile is 0.105, and that is the column the sentence actually describes.
The claim first becomes true at d = 400 (worst 0.147). This is the notebook's
headline for the Johnson–Lindenstrauss section, it is stated twice, and the
prompt box immediately above it warns *"a percentile alone does not test it"*.
(1,382 is correct: `johnson_lindenstrauss_min_dim(400, eps=0.2) == 1382`.)

**D3 — "several times the accuracy" is 1.46× (§1.1, §1.4).** The markdown after
cell 43 says *"Same forty labels, several times the accuracy"*, and the prompt
box's `left_open` says *"it is several times the accuracy for the same annotation
budget"*. Measured: 40 at random **0.433**, 40 one-per-cluster **0.633**. The
ratio is **1.46×**. The gain is real and worth teaching; "several times" is not
what it is.

**D4 — "335 million parameters per component" is off by 40× (§1.1).** The prompt
box before cell 37 says *"335 million parameters per component, estimated from
400 photographs"*. The cell's own arithmetic prints **8,390,656** for one full
4,096-dimensional covariance and **335,626,240** for forty. 335 million is the
total, not the per-component figure.

**D5 — four ⏱ markers are wrong, all in the same direction (§7.1).** Measured on
M4 Max with the notebook's own 2-thread cap:

| markdown claims | measured | ratio |
|---|---|---|
| cell 4/5, "about 20 seconds" for the SVD | **0.21 s** | 95× over |
| cell 22, "about 1 minute" for the projections | **0.6 s** | 100× over |
| cell 26, "1–2 minutes" for the reduced sweep | **0.1 s** reduced (27.3 s raw) | see D6 |
| cell 48, "2–4 minutes" for twenty splits | **6.2 s** | 25× over |

Over-statement is the safer direction, but §7.1 exists because readers budget
their evening from these markers, and a notebook whose every ⏱ is wrong by
one to two orders of magnitude trains the reader to ignore all of them —
including the one that is real.

**D6 — the ⏱ marker on the sweep names the wrong half (§7.1).** The markdown
before cell 27 says *"⏱ 1–2 minutes for the reduced sweep"*. Measured: the
**reduced** sweep is **0.1 s**; the **raw** 4,096-dimensional sweep in the same
cell is **27.3 s** and is the only computation in the notebook exceeding 20 s. A
reader who skips the raw sweep to save time will skip the fast half.

**D7 — the before/after ARI comparison is on mismatched rows (§2.1).** The
markdown before cell 27 says *"Same sweep as the previous lecture... Compare it
with what you wrote down last time."* Lecture 9 (`tools/notebooks/lecture_09.py`
lines 353 and 294) reports `adjusted_rand_score(y_audit, km.labels_[audit])` —
**40 rows**. Lecture 10 cell 27 reports `adjusted_rand_score(y, lab_raw)` —
**400 rows**. On the identical raw clustering these are **0.336** and **0.454**.
The column is honestly labelled `ARI(all)`, so the cell is not lying; the *prose
instruction* to compare it with last week's number is the defect.

**D8 — `audit` and `y_audit` are computed and never used (§4, dead state).** Cell
3 computes both, and the prompt box above it claims reproducing the forty *"is
the only reason the two notebooks are comparable at all"*. `grep` over
`tools/notebooks/lecture_10.py`: `audit` appears on lines 90, 91 and 94 only —
the assignment and its own assert. `y_audit` appears once. Neither is read again
in any of the 17 code cells. The variable that the prose says makes the
comparison possible is the variable that makes D7 unfixable as written.

**D9 — `err` is rebound from an array to a function (§4.1).** Cell 40:
`err = ((p99.inverse_transform(...) - Xa) ** 2).mean(axis=1)`, an `ndarray` of
length 412. Cell 50: `def err(pp):`, a function. Verified by AST walk over all
17 code cells. §4.1 is the rule lecture 19 spent 200 words teaching.

**D10 — `g` is rebound from a projector to a mixture (§4.1).** Cell 23:
`g = GaussianRandomProjection(...)`. Cell 37: `g = GaussianMixture(...)`. Same
AST walk. Also `ari` goes from a float (cell 35, inside the DBSCAN loop) to a
list (cell 37, `bic, ari = [], []`), and `p` is a `PCA(n_components=200)` fitted
on `X_tr` after cell 17 and a `PCA(n_components=d95)` after cell 43 — same type,
different meaning, 26 cells apart.

**D11 — the pseudo-inverse branch in cell 31 is unreachable, and the prose
describes it as the branch that runs (§3.2).** The prompt box's `left_open` says
*"random projection has no `inverse_transform`, so its reconstruction goes
through a pseudo-inverse. The two error columns are therefore not computed
identically, and the cell does it rather than hiding it."* In scikit-learn 1.7.2,
`hasattr(GaussianRandomProjection(...).fit(X_tr), "inverse_transform")` is
**True** — the method was added in 1.1 — so `bench()` takes the `if` branch for
all four reducers and the `else` branch never executes. Both routes happen to
give **0.32093**, so no printed number is wrong; the teaching claim and the dead
code are.

**D12 — the "honest" arm of the leak experiment leaks its own hyperparameter
(§2.1, unlabelled).** Cell 50's honest PCA is `PCA(n_components=d95).fit(Xtr)`,
but `d95 = 123` was computed in cell 15 from `PCA().fit(X)` — all 400 faces.
Recomputing 95%-of-variance from the 280 training faces alone gives **104**
components for the seed-42 split and **102–105** across the twenty seeds of cell
50. So the honest arm is fitted on training rows and *sized* by test rows, in the
cell whose subject is exactly that mistake. This is the notebook's best
unlabelled trap (§8.2) and nothing marks it.

**D13 — cell 35 prints a conclusion its own output contradicts.** The cell prints
*"...with 41 clusters..."* and then, on the next line, *"The right answer is 40
clusters and 0 noise. DBSCAN never gets there."* At the best-ARI eps DBSCAN
returns **41** clusters — it does very nearly get there on cluster count. The
real failure is ARI **0.133** with **97 of 400** faces called noise. Relatedly,
the prompt box claims cluster count and ARI *"peak in completely different
places"*; the cluster count over the grid peaks at 50, but at the ARI peak it is
41, so the two are not as far apart as claimed.

**D14 — `argmin(bic)` selects the first point on the grid (§1.3).** Cell 37
prints `BIC picks k={ks[argmin(bic)]}`. Measured, BIC is **strictly increasing**
across the entire grid, 56,468 at k = 5 to 119,175 at k = 80, so the argmin is
always **k = 5**, the grid's left edge, for any grid starting there. The prompt
box describes BIC and ARI as *"peaking in different places"*; BIC does not peak.
A criterion with no interior optimum has not selected a model — which is the
lesson lecture 9 taught about the inertia elbow, repeated here undetected.

**D15 — `best k = 60` is the grid's right edge (§1.3).** Cell 27 reports best
k = 60 for both spaces; 60 is the largest k in `ks`. Extending the grid on the
reduced data: silhouette **0.2061** at k = 60, **0.2167** at 70, **0.2385** at
80, **0.2404** at 100 — still climbing. ARI(all) peaks at k = 60 (0.521) and
falls to 0.472 at 70. The reported k is where the loop stopped, not where the
criterion turned.

**D16 — the trap is announced six times before it (§8.1).** Before cell 47 the
reader has been told, in order: the header's *"Cells marked ⚠ read before running
contain a defect on purpose"*; the prompt box before cell 17, *"fitting on all
400 because it is one line shorter — that is exactly the failure this notebook
builds to in section 9"*; the §9 heading *"An assistant reduces the dimension for
us"*; the *"⚠ Read before running. This is today's failure"* paragraph; the
prompt box label *"⚠ what the assistant returns"*; and that box's `constraint`
(*"fit the PCA on X, the whole corpus, as written"*) and `left_open` (*"All four
hundred photographs, including the hundred and twenty we then call held out"*),
which give away the full diagnosis before the cell has run. Lecture 19's four
announcements were judged sufficient to guarantee nobody falls in; this is six.

**D17 — every prompt box carries the full three-bullet annotation (§6.1).** All
**17** code cells have a `prompt()` box and all 17 supply `left_open`, `student`
and `catch`. The budget is five to eight. The two boxes that most need reading —
cell 31's `IncrementalPCA` constraint and cell 50's decision rule — sit at
positions 10 and 16 of 17, past where all three audit readers stopped.

**D18 — the notebook ships with no stored outputs (§1.2).** All 17 code cells
have `execution_count: null` and zero outputs. Every figure quoted in the
markdown is therefore unreconcilable against a stored output by construction,
which is how D1–D4 survived. It also means `tools/check_notebook_numbers.py`
cannot check this notebook at all.

### Checked and clean

- **§5.1 / §5.2 markdown rendering.** Scanned all 38 markdown cells: no prose
  line indented ≥ 4 spaces outside a fence, and no fence marker indented at all.
  Clean.
- **§3.1 code quoted in prose.** No ```` ```python ```` blocks appear in any
  markdown cell, so there is nothing to mismatch. The two inline spans that do
  not appear verbatim in a code cell are `` `covariance_type='full'` `` (naming
  an option the notebook deliberately does not run) and `` `PCA(...).fit(X)` ``
  (an ellipsis paraphrase of cell 47's real line); both are legitimate.
- **§3.3 cross-references.** *"section 9"* in cell 16 resolves — §9 is the leak
  section. *"the cell below"* (cell 0), *"the previous lecture"* / *"last time"*
  (cell 25), *"the build session"* (cells 2, 26) and *"the first application"*
  (cell 45) all resolve to real things. The §9 reference is correct as a pointer;
  its problem is D16, not the pointer.
- **§4.2 idempotence.** No cell trains a stateful model incrementally. Every
  estimator is constructed inside the cell that fits it, so re-running any cell
  in place gives the same answer. `IncrementalPCA` is constructed fresh inside a
  `lambda` each time.
- **§7 feasibility.** Nothing here needs a GPU, a download beyond the 4 MB
  Olivetti cache, or more than a few hundred MB of RAM. `full_matrices=False`
  keeps the SVD to 400 × 4,096. A student alone at home can run the whole
  notebook in well under two minutes of compute.
- **The asserts all pass.** Every `assert` in all 17 code cells was executed:
  shapes (400, 4096) and (400, 123) and (412, 4096); 40 audit indices; 280/120
  split with minima 7 and 3; 40 distinct cluster representatives;
  Eckart–Young at 5.98e-08; components agreeing at 6.07e-07.

### Not checked

- **Colab wall clocks.** Every timing above is Apple M4 Max. I did not run this
  on Colab's 2-vCPU box, so the Colab estimates in this script are extrapolations
  and are labelled as such. The *ratios* in D5 would shrink on slower hardware,
  but the SVD cell would have to be ~100× slower than measured for "about 20
  seconds" to be right, which is not plausible for a 400 × 4,096 thin SVD.
- **Cross-lecture claims.** *"unlike the scaling leak in the first application"*
  (cell 45) refers to material in lectures 3–7; I verified that lectures 6, 7 and
  9 all reference "the first application" consistently, but I did not re-derive
  that lecture's numbers.
- **Rendered appearance in Colab.** §10.8 asks for the rendered page to be read,
  not the source. I checked the markdown source mechanically and found it clean,
  but I did not open the notebook in Colab.
- **Figure legibility.** Whether identity is actually readable at 10–25
  components in cell 17's montage, and whether the anomaly montages in cell 40
  show what the captions say, are visual judgements I did not make — I verified
  the numbers behind them, not the images.
