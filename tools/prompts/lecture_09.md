# Lecture 9 — Forty labels for four hundred faces

**Build · Olivetti · Géron chapter 8.** Rebuild this notebook in Colab by
prompting, one cell at a time, in the order below.

Sixteen code cells. Seven carry the full three-bullet annotation; the other nine
carry the specification only. Everything a cell must print is stated before you
type it, so you can tell a right answer from a plausible one.

**Where the numbers come from.** Every figure quoted in this script was
re-derived by running the cell, on an idle **Apple M4 Max**, macOS 15,
Python 3.13.5, scikit-learn 1.7.2, NumPy 2.3.5, BLAS capped at 2 threads.
Anything I could not measure is marked *estimate*. I did not measure a Colab
runtime; where I give a Colab figure it is an assumption, stated as one.

**A note on timings before you start.** Every cell in this notebook is fast —
the whole thing is under fifteen seconds of compute on a modern laptop. That is
not what the shipped notebook says, and the difference matters: see cell 16 and
the defect report.

---

## Cell 1 — setup and the thread cap

**Prompt to type:**

> set OMP_NUM_THREADS, OPENBLAS_NUM_THREADS and MKL_NUM_THREADS to 2 before
> importing numpy — they're read at import time. then import numpy, sklearn,
> matplotlib, print the versions, and fix a seed of 42.

**Expect:** three version lines (`python 3.x`, `scikit-learn 1.x`, `numpy 2.x`);
`RANDOM_STATE = 42` and an `rng = np.random.default_rng(RANDOM_STATE)` left in
the namespace. Nothing else printed.

**Assert:** none — but check by eye that the `os.environ` block sits *above* the
`import numpy`. If it does not, the cap does nothing and you will not be told.

**Annotate:** short

---

## Cell 2 — the corpus

**Prompt to type:**

> load the olivetti faces from sklearn with shuffle=False. print how many
> photographs, how many people, how many photographs per person, and how many MB
> the array is. assert the shapes and that the pixels are in [0,1].

**Expect:** `X` of shape `(400, 4096)`, `float32`; `images` of shape
`(400, 64, 64)`; `y` with 40 distinct values, exactly 10 of each; **6.55 MB** in
memory (`X.nbytes / 1e6 = 6.5536`).

**Assert:**

```python
assert X.shape == (400, 4096)
assert images.shape == (400, 64, 64)
assert X.min() >= 0.0 and X.max() <= 1.0
assert len(np.unique(y)) == 40
assert np.bincount(y).min() == np.bincount(y).max() == 10
```

**⏱** First call only: `fetch_olivetti_faces` downloads ~4 MB from a remote
host. Under a second from cache, 5–30 s on a slow link. It caches to
`~/scikit_learn_data/olivetti_py3.pkz`; a second run does not re-download.

**Annotate:** short

**Then write, in a text cell, before you go on:**

> `y` exists because Olivetti is a benchmark. In the brief it does not. From
> here on the only labels we may use are the forty we pay for. `y` is used
> twice more, both times marked **AUDIT**, both times as something the
> stakeholder could not have afforded.

---

## Cell 3 — look at the faces

**Prompt to type:**

> write a helper that tiles a stack of 64x64 images into one array with a 2 pixel
> gap between them and shows it with imshow in grey, and use it to show one
> photograph of each of the 40 people, ten to a row.

**Expect:** a single figure, one image, 4 rows × 10 columns, greyscale, no axis
ticks. Forty visibly different faces.

**Assert:** none. This cell is checked with your eyes, which is the point of it.

**Annotate:** full

- **Left open:** what you are looking for. Not "are these faces" but "what
  varies *within* one person", which cell 4 answers and this montage structurally
  cannot — it shows one photograph each.
- **The usual student version:** leaving `vmin` and `vmax` off the `imshow`.
  Matplotlib's documented default is `vmin=None, vmax=None`, which normalises
  **each array to its own min and max**. Every montage in this notebook is then
  drawn on a different brightness scale, and cell 4's two rows become
  incomparable for a reason nothing on the page states. Ask for
  `vmin=0, vmax=1` explicitly, or check that it came back with them.
- **How you would catch it:** re-run the helper on `images[0:1]` alone. If the
  single face still spans full black to full white, the scale is per-array and
  no two of your figures mean the same thing.

---

## Cell 4 — ten photographs of one person, then ten of another

**Prompt to type:**

> now show all ten photographs of person 0 and all ten of person 22, as two rows
> of ten, using the same helper.

**Expect:** one figure, two rows of ten. Row 1 and row 2 are different people;
within each row you should be able to see glasses on and off, lighting from
either side, eyes shut, head turned.

**Assert:** none.

**Annotate:** short

**Then, in a text cell:** *any method that groups those ten has to be
insensitive to all of that, while still separating them from the other 390.*

---

## Cell 5 — AUDIT: is this actually hard?

**Prompt to type:**

> compute all pairwise euclidean distances between the 400 photographs in raw
> pixels. using the true labels, split the pairs into same-person and
> different-person, print the mean and median of each, and what fraction of
> same-person pairs are further apart than the median different-person pair.
> histogram both on the same axes.

**Expect:** **79,800** pairs, of which **1,800** are same-person.
same person mean **8.38**, median **8.15**; different people mean **12.41**,
median **12.15**; **6%** of same-person pairs are further apart than the median
different-person pair (0.0644, so it prints `6%`). Two overlapping histograms,
the same-person one shifted left.

**Assert:**

```python
assert len(within) == 40 * (10 * 9 // 2) == 1800
assert len(within) + len(between) == 400 * 399 // 2
```

**Annotate:** short

**Then, in a text cell:** the distributions are separated in the mean and
overlap in the tails. That 6% is a floor, not a ceiling: it is the fraction of
same-person pairs that a distance-based method starts out already having lost.
Write it down.

---

## Cell 6 — the forty labels, and all the supervision there is

**Prompt to type:**

> pick 40 of the 400 photographs at random without replacement, sort the
> indices, and take their true labels as the annotator's answers. print how many
> pairs that gives, how many of those pairs are the same person, and how many
> distinct people the forty happen to cover.

**Expect:** with `rng` seeded at 42 and untouched since cell 1: **780** pairs,
**22** of them same-person, **25** distinct people covered. Fifteen of the forty
people appear nowhere in the labelled sample, so no clustering can be rewarded
for finding them.

**Assert:**

```python
assert len(audit) == 40 and len(np.unique(audit)) == 40
```

**Annotate:** short

> **Re-run order (§7.2).** `audit` depends on the state of `rng`, which cell 1
> creates and only this cell consumes. If you re-run this cell alone you get a
> *different* forty and every ARI below shifts. To reproduce the numbers in this
> script, re-run **cell 1, then cell 6**, in that order, before re-running
> anything downstream.

---

## Cell 7 — what nothing at all scores

**Prompt to type:**

> for each k in [2,5,10,15,20,30,40,50,60], throw all 400 faces into uniformly
> random clusters, 20 seeds each, and print the mean and standard deviation of
> the silhouette and of the ARI against my 40 labelled photos. keep the arrays,
> I want to plot the silhouette one later.

**Expect:** nine rows. The **ARI** null sits at zero at every k — the largest
mean in magnitude is −0.0089 at k=40, against standard deviations of 0.010–0.039.
The **silhouette** null does not:

| k | null silhouette | null ARI |
|---|---|---|
| 2 | +0.0000 ± 0.0009 | +0.0008 ± 0.0103 |
| 5 | −0.0197 ± 0.0028 | +0.0025 ± 0.0161 |
| 10 | −0.0422 ± 0.0045 | −0.0040 ± 0.0205 |
| 15 | −0.0587 ± 0.0079 | +0.0018 ± 0.0318 |
| 20 | −0.0718 ± 0.0068 | −0.0071 ± 0.0186 |
| 30 | −0.1028 ± 0.0096 | −0.0022 ± 0.0390 |
| 40 | −0.1288 ± 0.0107 | −0.0089 ± 0.0244 |
| 50 | −0.1576 ± 0.0152 | −0.0062 ± 0.0271 |
| 60 | −0.1805 ± 0.0124 | −0.0075 ± 0.0261 |

**Assert:**

```python
assert abs(null_ari_mean).max() < 0.05      # chance-corrected by construction
```

Do **not** write the same assert for the silhouette. Work out on paper first
what you expect it to be, then read the table.

**⏱** 180 silhouette evaluations on 400 × 4,096 floats. **3 s** measured idle on
the machine above. *Estimate:* 15–30 s on a free Colab CPU runtime. Measured
83 s on the same laptop while it was running twenty other jobs (load average
394 on 16 cores) — which is what the thread cap in cell 1 is for, and it still
was not enough.

**Annotate:** full

- **Left open:** that the two nulls are different *kinds* of object. ARI is
  corrected for chance in its definition, so its zero is structural and holds at
  every k. The silhouette has no such correction, so its null had to be
  measured — and it turns out to depend on k, falling from 0.000 at k=2 to
  −0.181 at k=60. One horizontal line cannot represent it.
- **The usual student version:** drawing a line at 0 and labelling it "random
  assignment", on the reasoning that s = (b−a)/max(a,b) is zero when a ≈ b. The
  shipped notebook does exactly this (`ax.axhline(0, ..., label="random
  assignment")`). At k=40 the measured value is **−0.1288 ± 0.0107** — twelve
  standard deviations from the line it draws.
- **How you would catch it:** any unsupervised score needs an empirical null at
  the *same* k you are comparing. It costs one loop. Without it, "silhouette
  0.15" is a number with no units.

---

## Cell 8 — the sweep

**Prompt to type:**

> fit kmeans for k in [2,5,10,15,20,30,40,50,60] with n_init=5 and
> random_state=42. for each k record the inertia, the silhouette over all 400
> photos, and the ARI against my 40 labelled ones, print them as you go, keep the
> labels in a dict, and time the whole loop. assert inertia never goes up as k
> goes up.

**Expect:** nine rows printed as they arrive, then the elapsed time.

| k | inertia | silhouette | ARI on the 40 |
|---|---|---|---|
| 2 | 26654.7 | +0.1516 | +0.0272 |
| 5 | 21548.2 | +0.1045 | +0.0597 |
| 10 | 19142.3 | +0.0850 | +0.1007 |
| 15 | 17242.7 | +0.1018 | +0.1596 |
| 20 | 15811.4 | +0.1139 | +0.1960 |
| 30 | 13871.4 | +0.1250 | +0.2185 |
| 40 | 12139.7 | +0.1465 | +0.2978 |
| 50 | 10796.9 | +0.1581 | +0.3639 |
| 60 | 9744.2 | +0.1695 | +0.3364 |

Read that table before you plot anything. The silhouette **falls** to k=10 and
then **rises to the end of the grid**. It does not have an interior maximum
here.

**Assert:**

```python
assert all(inertia[i] >= inertia[i + 1] for i in range(len(ks) - 1)), \
    "inertia must be non-increasing in k"
```

**⏱** **3.4 s** measured idle on the machine above. *Estimate:* 15–25 s on a
free Colab CPU runtime. The cell prints its own elapsed time — use that number,
not mine, in cell 16.

**Annotate:** full

- **Left open:** what `n_init=5` buys. Five independent Lloyd runs per k, best
  inertia kept. The assert is a property of the *objective* (a finer partition
  cannot have larger inertia), not of the algorithm, so when it fails it is
  telling you the optimiser failed, not that the maths did.
- **The usual student version:** omitting `n_init` entirely. Since
  scikit-learn 1.4 the default is `n_init="auto"`, and with the default
  `init="k-means++"` that resolves to **exactly one** initialisation
  (`_BaseKMeans._check_params_vs_input`). Measured on this data: with the
  default, inertia at k=40 is **12539.2** against **12139.7** with `n_init=5` —
  3.3% worse at the one k that matters, 2.2% worse at k=50. And the monotonicity
  assert **still passes**, so the assert does not protect you from this.
- **How you would catch it:** re-run the same grid at `n_init=1` and at
  `n_init=5` and diff the inertias. If they differ at any k, the curve you are
  about to read an elbow off is partly optimiser noise. That is four lines and
  it costs seconds.

---

## Cell 9 — plot inertia against k

**Prompt to type:**

> plot inertia against k. no annotations on it.

**Expect:** one monotone decreasing curve, 26654.7 down to 9744.2, no bend that
survives being looked at twice.

**Assert:** none.

**Annotate:** short

---

## Cell 10 — quantify the disagreement about the elbow

**Prompt to type:**

> implement the kneedle rule on this curve: rescale k and inertia both to [0,1],
> then take the point furthest below the straight line joining the two ends.
> print the k it picks, and print that the truth is 40.

**Expect:** the rule picks **k = 15**. The truth, which the brief does not let us
use, is 40.

**Assert:** none.

**Annotate:** short

**Then, in a text cell:** the lesson is not "find a better elbow rule". Inertia
is monotone by construction and reaches exactly zero at k = n. It does not
contain the answer, so no rule reading it can.

---

## Cell 11 — the silhouette against its own null

**Prompt to type:**

> plot the silhouette from the sweep against k, and on the same axes plot the
> random-assignment silhouette from cell 7 at the same nine k values, with a band
> at ± one standard deviation. mark k=40 with a vertical line. then print the
> best k by silhouette and the silhouette at k=40.

**Expect:** two curves on nine matched k values (§2.1: the same nine k, not a
horizontal line). k-means runs 0.085–0.170; the null runs 0.000 down to −0.181,
so the *gap* widens with k throughout. Printed: best silhouette **0.1695 at
k = 60**, silhouette at the true k=40 **0.1465**.

**Assert:**

```python
assert best_k == max(ks), "the winner is at the edge of the grid"
```

Yes — that assertion is written to pass. Write down why it passes before you go
on.

**Annotate:** short

**Then, in a text cell:** the criterion that was supposed to have an interior
optimum has put its maximum on the boundary of the grid we searched. Whatever it
is measuring, it is not "how many people are in the archive".

---

## Cell 12 — the silhouette diagram

**Prompt to type:**

> draw the silhouette diagram at k=2, k=10 and k=40 side by side, one shape per
> cluster, points sorted within each cluster, clusters sorted by their own mean,
> a dashed line at the overall mean, and share the x axis across the three
> panels.

**Expect:** three panels, means printed in the titles: **0.152** at k=2,
**0.085** at k=10, **0.146** at k=40. At k=2 one enormous shape and one small
one; at k=40, forty knives of visibly unequal length, several of them entirely
left of the dashed mean.

**Assert:** none — but check `sharex=True` actually came back. Three silhouette
diagrams on independent x-axes cannot be compared, which is the only reason to
draw three.

**Annotate:** short

---

## Cell 13 — AUDIT: now look at what it grouped

**Prompt to type:**

> take the k=40 labels from the sweep. using the true labels, print the cluster
> sizes, how many clusters have exactly 10 members, and how many contain exactly
> one person. then show the cleanest and the worst cluster as two montage rows,
> ignoring clusters with fewer than 5 members, and put the size, the number of
> people and the purity in each title.

**Expect:** sizes from **4** to **25**; only **7 of 40** clusters have exactly
ten members; **16 of 40** contain exactly one person. The cleanest qualifying
cluster has 9 photographs, 1 person, purity 1.00. The worst has **24
photographs, 10 people, purity 0.21**.

**Assert:**

```python
assert sizes.sum() == 400
assert min(sizes[c] for c in big) >= 5      # a cluster of one has purity 1.00
```

**Annotate:** full

- **Left open:** what the worst cluster is made of. Look at it before reading
  on. Those 24 photographs are not similar *people* — they are similar
  *photographs*: same lighting, same head angle. In 4,096 raw pixels a lamp on
  the left is a bigger vector than a different nose.
- **The usual student version:** `montage(ax, images[lab == c][:10], ...)` under
  a title that says `f"{sizes[c]} photographs"`. That is what the shipped
  notebook does, and on this seed it draws **ten** faces under a caption that
  says **24** — the annotation does not sit on what it annotates. Either drop
  the slice or say "first 10 of 24" in the title.
- **How you would catch it:** count the faces in the picture and compare with
  the number in the caption. It is the cheapest check in the notebook and it
  catches a whole family of montage bugs.

---

## Cell 14 — an assistant chooses k

Type this one exactly as it comes back. Run it. **Write the two numbers it
prints on your sheet before you read the next section.**

**Prompt to type:**

> cluster the faces with k-means, pick the best number of clusters using the
> silhouette score, and report the score.

**Expect:** a loop over a handful of candidate k, `n_init=3`, and one line:
`best k = 60, silhouette = 0.1695`. Nothing warns. Nothing crashes. The
candidates it scores are 5: 0.1045, 10: 0.0850, 20: 0.1139, 40: 0.1465,
60: 0.1695.

**Assert:** none. Deliberately none — write down what you would have asserted.

**⏱** 1.5 s measured idle. *Estimate:* under 10 s on Colab.

**Annotate:** full

- **Left open:** the range of k. The prompt says "the best number of clusters"
  and never says out of what, so the assistant invented a candidate list. The
  answer it returns is the largest entry on that invented list.
- **The usual student version:** reporting 0.1695 as the score of the chosen
  model. Two things are wrong with it and only one is famous. Application 1
  already gave you the check for the other: *"detect whether the winner sits on
  the EDGE of the grid and say so — an optimum at the boundary means the optimum
  may lie outside it."* Sixty is the largest candidate. Nothing in this cell
  looks.
- **How you would catch it:** ask of any selection, *what did it choose between,
  and did it choose an end?* Then ask the question of cell 15.

**Then, in a text cell — and not before:**

> ### ⚠ Reviewer question 5: what is the default I did not ask for?
>
> Nothing here touched a test set, because there is no test set. The defect is
> subtler, and it is the one that matters for every unsupervised model
> selection: **the score reported is a maximum over a noisy criterion, evaluated
> on the same data that chose it.** Five candidate values of k, five noisy
> estimates, and we print the largest. It is the rule from application 1 in a
> new costume — *the number that chose the model cannot also be the number that
> reports it* — and it applies with no labels anywhere in sight.
>
> There is a second thing wrong with the cell, and this notebook did not label
> it. The winner is the largest candidate on the list. Application 1's grid
> search printed a warning for exactly that.
>
> Now measure the first one. Carefully — because measuring it wrong is easier
> than measuring it right.

---

## Cell 15 — measure the optimism, with a control

**Prompt to type:**

> split the corpus into two random halves. choose k on the first half by
> silhouette, then score that same fitted model on the second half with predict
> — no refitting. do it for 5 seeds and print the selected and held-out
> silhouette each time. then do the same thing again with k fixed at the value
> the selection keeps choosing, so I can see how much of the gap is actually
> from selecting k.

**Expect:** two blocks. The selection block picks **k = 60 on all five seeds**:

| seed | selected | held out | gap |
|---|---|---|---|
| 0 | 0.1744 | 0.0947 | +0.0797 |
| 1 | 0.1697 | 0.0909 | +0.0787 |
| 2 | 0.1847 | 0.0758 | +0.1089 |
| 3 | 0.1772 | 0.0473 | +0.1299 |
| 4 | 0.1850 | 0.0990 | +0.0860 |

mean gap **+0.0967**. The control block, k fixed at 60 with no selection at all,
gives **+0.0797, +0.0787, +0.1089, +0.1299, +0.0860** — mean **+0.0967**.
Identical, seed for seed.

**Assert:**

```python
assert np.allclose(sel_gap, fixed_gap)   # selection contributed nothing
```

**⏱** 6 s measured idle for the selection block, roughly the same again for the
control. *Estimate:* under a minute on Colab.

**Annotate:** full

- **Left open:** what the gap is a gap *of*. Both halves come from the same 400
  photographs of the same 40 people, so this measures optimism within one
  corpus, not generalisation to a new archive.
- **The usual student version:** running the selection block only, reporting
  mean optimism +0.097, and calling it the price of choosing k. Measured above:
  **none of it is.** The selection picks k=60 every time, so the comparison is
  a model scored on the points it was fitted on against the points it was not —
  which is the gap you get with k fixed in advance and no selection anywhere.
  The lesson is true; this experiment is not evidence for it. (For contrast:
  with k fixed at 40 the same gap is **+0.0664**, so the number moves with k,
  which is another thing the selection block confounds.)
- **How you would catch it:** whenever you attribute a gap to a cause, run the
  experiment again with the cause removed. If the number does not move, you
  measured something else. Here it did not move at all, to four decimal places.

**Then, in a text cell:**

> ### The corrected specification
>
> *"Cluster the faces with k-means over k in a stated range, and say what the
> range is. Choose k by mean silhouette on a randomly held-out half, refit the
> chosen k on everything, and report both the selection score and the held-out
> score. Warn if the winner is at either end of the range. Fix the seed and
> print every k, not only the winner."*
>
> And the honest report of what cell 15 measured: on this corpus the
> fitted-versus-held-out silhouette gap is about **+0.10 at k=60**, of which
> **0.00** is attributable to having selected k, because the selection is
> degenerate here. To measure selection optimism you need a selection that
> selects.

---

## Cell 16 — what this costs, and why the next lecture exists

**Prompt to type:**

> using the sweep time I already measured, and timing one silhouette_score call
> on the 400 faces, work out what the same sweep would cost on an archive of
> 100,000 photographs at 4,096 dimensions. kmeans is linear in n per iteration,
> silhouette is quadratic in n. print the cost model and the assumptions, not
> just the answer.

**Expect:** the measured sweep (3.4 s idle here — use *your* number), a measured
`silhouette_score` at n=400 of about **0.017 s**, then the extrapolation:
(100000/400)² = **62,500**, so **one** silhouette evaluation on the archive is
0.017 × 62,500 ≈ **1,060 s**, about **18 minutes** — and the sweep does nine of
them. The k-means fits scale by 250, not 62,500. Also worth printing: a dense
100,000 × 100,000 distance matrix in float64 is **80 GB**.

If you want the other half of the argument measured rather than assumed, re-run
the cell-8 sweep on 256 randomly chosen pixel columns instead of all 4,096. I
measured **38.3 s at d=4,096 against 2.40 s at d=256** — a 16× cut in d bought a
16× cut in time, linear as the cost model says. (Both figures were taken while
this laptop was loaded, so read the *ratio*, not the seconds.)

**Assert:**

```python
assert sweep_seconds > 0        # you measured it, you did not type it
```

**Annotate:** full

- **Left open:** what to do about it. Nothing, today. Both terms are linear in
  d = 4,096, and most of those 4,096 numbers are describing a lamp. That
  sentence is the whole setup for the next lecture.
- **The usual student version:** quoting the absolute wall clock as the
  motivation — "the sweep took N minutes, therefore we need fewer dimensions".
  The shipped notebook does this, and on the machine above the sweep takes
  **3.4 seconds**, so its closing argument reads as an argument for doing
  nothing. Its extrapolation cell prints `roughly 1 minutes` — wrong in the
  grammar and wrong in the point. A wall clock measured on one idle laptop is
  not a reason; a cost model that is quadratic in n is.
- **How you would catch it:** state the exponent, not the seconds. Any claim of
  the form "this is too slow" that does not name what it is slow *in* cannot be
  checked, and will be false on somebody's machine within a year.

---

## Closing text cell

> ```
> Best k the silhouette chose:              ____________
> Silhouette at that k:                     ____________
> Silhouette of random assignment at that k: ___________
> ARI on the 40 labelled photographs:       ____________
> Seconds the sweep took on your machine:   ____________
> ```
>
> Bring the sheet. We open the next lecture by scoring your commitment out loud,
> and then we take four thousand dimensions away and do it again.
>
> Do not fix anything yet.

Note the third line. The commitment sheet in the shipped notebook has no such
row, and a silhouette written down without the null it is being compared to is
the thing this whole lecture is about.

---

## Defects found in the current notebook

`notebooks/lecture-09.ipynb`, 56 cells, 16 of them code. **The notebook ships
with zero stored outputs** (`execution_count` is `None` on all sixteen code
cells), so §1.2 — "every prose figure must appear in a stored cell output" —
cannot be satisfied by it as shipped, and the §9 machine check for ⏱ markers
("any cell whose stored execution exceeded 20 s") has nothing to read. Verified
with `nbformat`.

I verified everything below by running it. Method: the notebook's own code,
extracted from `tools/notebooks/lecture_09.py`, on the machine named at the top
of this file, with the same seeds. **The one thing I could not check is Colab**
— every timing claim below is for the machine named above, and Colab could be
several times slower. It cannot be a hundred times slower, which is what the
notebook's claims would require.

### 1. Every ⏱ figure in the notebook is wrong by one to two orders of magnitude — *checked*

| Notebook says | Measured, idle | Ratio |
|---|---|---|
| §7 "⏱ **about 1 minute**" | **0.7 s** | 85× |
| §9 "⏱ **3–6 minutes**" | **3.35 s** | 55–110× |
| §13 "⏱ **about 2 minutes**" | **6.0 s** | 20× |

This is the most consequential defect, because §14 quotes the wall clock as the
reason lecture 10 exists ("Write it down. It is the reason the next lecture
exists"). Measured, that reason is three seconds long. A student who follows
§7.2 and budgets six minutes for the sweep learns that the notebook's timings
are decorative, in a course whose §7.1 exists because untimed cells blocked four
of six exercises in lecture 19.

Second-order: cell 54's extrapolation is
`sweep_seconds * (sum(range(2,61)) / sum(ks)) * 2 / 60`, i.e.
`sweep_seconds × 0.2633` minutes. At the measured 3.35 s it prints **`roughly 1
minutes`** — an f-string with `:.0f` and a hard-coded plural.

I also measured the opposite failure, which is worth stating because it is the
notebook's own subject: re-running the identical `k=60, n_init=1` fit while this
laptop was under load average 394 on 16 cores took **17.9 s and 47.1 s** on two
consecutive calls, against **0.155 s** idle. The thread cap in §1 does not
control for that, and §1's annotation claims it does ("a timing you measure is a
timing you can repeat").

### 2. "Random assignment scores 0.00 on both" is false for the silhouette — *checked*

Three places state it: cell 25 ("Both anchors sit at zero"), cell 26 the commit
sheet ("Random assignment scores 0.00 on both"), and cell 23's `left_open`
("Both land at zero and only one of them had to").

Measured, 20 seeds, the notebook's own `random_assignment_scores`:

```
k=10  silhouette -0.0422 ± 0.0045   ARI on the 40 -0.0040 ± 0.0205
k=40  silhouette -0.1288 ± 0.0107   ARI on the 40 -0.0089 ± 0.0244
```

ARI lands at zero. The silhouette does not, at either k the cell prints, and it
is **twelve standard deviations** from zero at k=40. Extending the same
measurement across the sweep's nine k values, the null silhouette falls
monotonically from +0.0000 ± 0.0009 at k=2 to **−0.1805 ± 0.0124** at k=60,
while the null ARI stays within ±0.009 of zero throughout.

The notebook's intended sentence — *"only one of them had to land at zero"* — is
exactly right and its numbers are backwards. It is the one place in the notebook
where the measurement it took was discarded in favour of the answer it expected.

Consequence: cell 25's "a silhouette of 0.02 is not a discovery" is also wrong
as stated. At k=40, 0.02 sits **13.9 standard deviations above** the measured
null, which is as far from "not a discovery" as this notebook gets.

### 3. §2.1 — the silhouette curve is compared against an anchor that is not the measured one — *checked*

Cell 37 draws `ax.axhline(0, color="#4b5563", ls=":", lw=2, label="random
assignment")`, and cell 36's `constraint` calls it "the anchor from section 7".
Section 7's measured anchor at the k values plotted is −0.042 to −0.181, not
zero. The plotted line is not the anchor, and because the null is k-dependent no
horizontal line can be. This is the lecture-19 §6 shape: the notebook measured
the right thing and then compared against something else.

### 4. §10's premise — "silhouette, which has a maximum" — does not hold on this data — *checked*

Cell 35: *"Inertia has no interior optimum. The silhouette does."* Cell 36's
`student` bullet: *"it still peaks some distance from 40."*

Measured over the notebook's own grid (`n_init=5`, `random_state=42`):

```
k    2      5      10     15     20     30     40     50     60
sil  .1516  .1045  .0850  .1018  .1139  .1250  .1465  .1581  .1695
```

The maximum over the grid is at **k = 60, the last point searched** — the
boundary, not an interior optimum. The curve falls to a minimum at k=10 and then
rises monotonically to the edge. Cell 37 prints `best silhouette 0.1695 at
k = 60`, directly under a section heading asserting the opposite. I confirmed
the peak is not an `n_init` artefact as far as k=80 (silhouette +0.1873, still
rising); I could **not** complete the sweep past k=80 under machine load, so I
cannot say where or whether it turns over.

### 5. §13's optimism experiment does not measure what it says — *checked, and this is the second serious one*

Cell 50 selects k on half the corpus and scores the winner on the other half.
Measured: **the selection chooses k=60 on all five seeds.** Re-running the
identical experiment with k **fixed at 60 in advance** — no selection at all:

```
selected-k gap  mean +0.0967  per seed [0.0797 0.0787 0.1089 0.1299 0.0860]
fixed k=60 gap  mean +0.0967  per seed [0.0797 0.0787 0.1089 0.1299 0.0860]
excess attributable to selecting k: +0.0000
```

Identical, seed for seed, to four decimal places. **Zero** of the reported
optimism comes from selecting k; all of it is the ordinary fitted-versus-held-out
gap, which for a fixed k=40 is +0.0664. The section's thesis — the number that
chose the model cannot report it — is true, and lecture 2 already establishes it
properly. The evidence offered here does not support it. This is §2.1/§2.2
repeated: a confound left uncontrolled, in the section whose subject is
uncontrolled comparisons.

Relatedly, cell 49's `catch` says *"the optimism is small, and a single seed
cannot distinguish it from noise."* Measured: mean +0.0967 against a per-seed
spread of 0.079–0.130, all five positive, and more than half of the selected
score. It is the largest effect in the notebook, and every single seed shows it.

### 6. §13 misses the defect its own course already taught, and the one it names points at the wrong lecture — *checked*

The assistant cell returns `best k = 60` from candidates `[5, 10, 20, 40, 60]` —
the largest. Lecture 2 (`tools/make_notebooks.py`) ships the check verbatim:
*"detect whether the winner sits on the EDGE of the grid and say so"*, with an
`if best_n == max(...)` warning in its code. Lecture 9 does not mention it. By
§8.2 this unlabelled second defect is the better trap of the two and it is
simply absent.

Cross-references in the same passage disagree with each other (§3.3). Cell 46's
`student` bullet and cell 48 both say *"the previous application's grid search
could not report its own best score"*; cell 46's `catch` says *"the rule from
application 1"* and cell 51 says *"the first application"*. Checked by grep:
`GridSearchCV` appears in `lecture_06.py` and `lecture_07.py` only, and lecture
7's grid search is explicitly **not** used to choose (*"cross-validation does
not get a vote on `max_depth` here"*). The `best_score_`-is-optimistic rule is in
**lecture 2**, i.e. application 1. Two of the four references are wrong.

### 7. §8.1 — the defect is announced four times before the reader reaches it — *checked by reading, not measurement*

Cell 0 ("Cells marked **⚠ read before running** contain a defect on purpose"),
cell 45 (section heading plus "**⚠ Read before running**"), cell 46's label
("⚠ what the assistant returns"), and cell 46's `constraint` ("the defect is
subtler than that"). Worse, cell 46's `left_open` and `student` bullets state
the answer in full — *"the score reported is a MAXIMUM over a noisy criterion,
evaluated on the same data that chose it"* — in the box immediately **above** the
cell. Cell 48 then re-states it as a discovery. Nobody falls in, which is the
exact finding §8.1 records from lecture 19.

### 8. §6.1 — sixteen code cells, sixteen full three-bullet annotations — *checked*

Every one of the sixteen `prompt()` calls in `lecture_09.py` supplies all three
of `left_open`, `student`, `catch`. The budget is five to eight. Measured by
counting the keyword arguments in the module.

### 9. §6.2 — a `constraint` that names a dependency the notebook does not have — *checked*

Cell 5: *"`shuffle=False` — the ten photographs of each person stay adjacent,
**which every montage below relies on**"*. No montage relies on it. Checked by
reading all three: cell 10 uses `images[y == p][0]`, cell 12 uses
`images[y == 0]` and `images[y == 22]`, cell 43 uses `images[lab == c][:10]` —
all boolean masks over `y` or `lab`, none an index range. Under `shuffle=True`
`y` is permuted with `X`, so every one of them still returns the right faces.
The stated reason for the argument is fiction; `shuffle=False` is also
scikit-learn's own default, so the argument is not even doing the work of
overriding one.

### 10. §5.4 — a caption that does not describe its picture — *checked*

Cell 43: `montage(ax, images[lab == c][:10], ncol=10)` under
`ax.set_title(f"{tag} cluster: {sizes[c]} photographs, ...")`. On the shipped
seed the worst cluster has **24** photographs; the montage draws **ten** of them
under a title reading "24 photographs, 10 people, purity 0.21". This is
`<- lowest` beside the wrong number again, in a cell whose whole purpose is
looking carefully at a picture.

Same cell, minor: `plt.subplots(2, 1, figsize=(11, 3))` gives each montage row
about 1.4 inches of height for a strip that is 10 × 64 pixels wide, so the faces
render squashed.

### 11. §3.3 — "the next cell" points two cells away — *checked*

Cell 9's `left_open`: *"what you are looking for. Not 'do these look like faces'
but 'what varies within a person', **which the next cell answers**"*. The next
cell is 10, the montage of forty different people, which cannot show
within-person variation. The cell that answers it is **12**. The identical
phrase failed the same way in lecture 19 (§3.3).

### 12. §4.1 — a name bound to two kinds of object — *checked*

`s` is an `ndarray` of twenty silhouettes in cell 24
(`s, a = random_assignment_scores(k)`) and a scalar `np.float64` in cells 47 and
50 (`s = silhouette_score(...)`). `a` is an `ndarray` of twenty ARI values in
cell 24 and an `ndarray` of 200 row indices in cell 50 (`a, b = perm[:200],
perm[200:]`). Both are module-level bindings and neither is remarked on, in a
course that spends 200 words on `target` being clobbered. Also dead: `best_model`
is assigned in cell 47 and never read.

### 13. §1.1 — figures I checked and found *correct* — *checked*

For balance, these reconcile: 79,800 pairs; 1,800 within-person; the pixel-range
and per-person-count asserts; "400 photographs of 40 people, 10 each"; 6.55 MB;
the audit sample under seed 42 covering 25 distinct people with 22 same-person
pairs among 780; the inertia monotonicity assert (holds, and still holds at the
library default `n_init`); the kneedle rule picking k=15 against a truth of 40;
`sizes` at k=40 running 4–25 with 7 clusters of exactly ten and 16 single-person
clusters; silhouette-diagram means of 0.152, 0.085 and 0.146.

One judgement call rather than an error: cell 17 says the two distance
distributions "overlap **heavily**". The means are 8.38 and 12.41 and the
notebook's own printed statistic is that **6%** of same-person pairs exceed the
median different-person pair. "Heavily" is doing more work than 6% supports; the
printed number is the better sentence.

### 14. Checks that came back clean — *checked mechanically*

- §5.1/§5.2: no markdown line indented ≥4 spaces outside a fence, and no fence
  marker indented at all, across all 40 markdown cells.
- §3.1: no ` ```python ` block appears in any markdown cell, so there is no
  quoted code that could fail to exist. The two fenced blocks (cells 26 and 55)
  are commitment sheets, not code.
- §4.2 idempotence: every cell that fits a model constructs it in the same cell.
  Re-running any single cell is safe **except** cell 20, which consumes `rng` and
  silently redraws the forty labels — noted as a re-run hazard in cell 6 of this
  script; the notebook itself does not name it (§4.3).
- §7 carry-out-ability: nothing here needs a GPU, the dataset is a 4 MB
  download, and peak memory is a 400 × 400 distance matrix. A student alone at
  home can run all of it — faster than the notebook promises.
