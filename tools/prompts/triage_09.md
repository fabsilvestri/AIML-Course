# Triage — lecture 9 defect report

Claims triaged: **14** (the task message said 4; the Phase A report's
`Defects found in the current notebook` section contains fourteen numbered
entries, and all fourteen are triaged below).

**Method.** Every number was re-derived by running the notebook's own code,
extracted verbatim from `tools/notebooks/lecture_09.py`, with the same seeds and
the same `OMP_NUM_THREADS=2` cap the notebook sets. Machine: Darwin 25.5.0,
16 cores, Python 3.13.5, scikit-learn 1.7.2, numpy 2.3.5, load average 3.8–10.7
during the runs. Olivetti was already cached at `~/scikit_learn_data/olivetti_py3.pkz`.
Scripts: `…/scratchpad/verify09.py` and `…/scratchpad/l09/verify09b.py`.

**Noted once, not repeated per claim (per the brief):** `notebooks/lecture-09.ipynb`
stores **zero** outputs — `execution_count is None` and `len(outputs) == 0` on all
sixteen code cells — so no prose figure in this notebook can be reconciled
against a stored output, and the §9 ⏱ machine check has nothing to read. Phase A
states this correctly.

---

### Claim 1 — every ⏱ figure is wrong by one to two orders of magnitude
**Verdict:** CONFIRMED

**Evidence:** the three timed cells, run with the notebook's own thread cap:

```
=== cell 24 (markdown cell 22 says "⏱ about 1 minute") ===
CELL24 WALL CLOCK 0.68 s        loadavg (7.84, 4.39, 3.29)
=== cell 29 (markdown cell 27 says "⏱ 3–6 minutes") ===
CELL29 WALL CLOCK 4.58 s        loadavg (10.73, 5.04, 3.53)
=== cell 50 (markdown cell 48 says "⏱ about 2 minutes") ===
CELL50 WALL CLOCK 3.29 s        loadavg (10.03, 4.99, 3.52)
```

Ratios: **88×**, **39–79×**, **36×**. My cell-29 and cell-50 numbers differ from
Phase A's (3.35 s and 6.0 s) because of machine load; the direction and the order
of magnitude reproduce, which is what the claim asserts.

The Colab caveat is real but does not rescue the figures: the notebook caps BLAS
at **two** threads, so this is a two-thread run on both machines. A Colab CPU
runtime is a small number of vCPUs and is plausibly 3–8× slower here, not 36–88×.

Second-order sub-claim, also confirmed. Cell 54's extrapolation ran and printed,
verbatim:

```
the coarse sweep over 9 values of k took 5 seconds
a full sweep over k = 2..60 at n_init=10 is roughly 1 minutes on this machine
```

`roughly 1 minutes` — `:.0f` against a hard-coded plural. One correction to
Phase A's arithmetic: the multiplier is `(1829/232) * 2/60 =` **0.26279**, not
0.2633. Immaterial to the conclusion.

The load-average sub-claim (17.9 s / 47.1 s under load 394 vs 0.155 s idle) is a
statement about Phase A's machine at a moment I cannot recreate; I did observe
the same effect in weaker form — the identical `k=60, n_init=5` fit took **30.3 s**
in one run and is part of a 4.58 s nine-value sweep in another, a 6× swing under
the cap. The point that the thread cap does not control for external load stands.

**Severity:** misleads a student
**Origin:** hand-written prose (the ⏱ figures and the `⏱ N min` prompt labels);
the `roughly 1 minutes` plural is generated code
**Fix:** re-measure all three cells and write the measured seconds; make cell 54
pluralise, or report seconds throughout.

---

### Claim 2 — "random assignment scores 0.00 on both" is false for the silhouette
**Verdict:** CONFIRMED

**Evidence:** the notebook's own `random_assignment_scores`, twenty seeds:

```
k= 10  silhouette -0.0422 ± 0.0045   ARI40 -0.0040 ± 0.0205   [9.5 sigma from 0]
k= 40  silhouette -0.1288 ± 0.0107   ARI40 -0.0089 ± 0.0244   [12.1 sigma from 0]
```

Identical to Phase A to four decimal places. Extended across the sweep grid:

```
k=  2  null sil +0.0000 ± 0.0009   null ARI +0.0008 ± 0.0103
k=  5  null sil -0.0197 ± 0.0028   null ARI +0.0025 ± 0.0161
k= 10  null sil -0.0422 ± 0.0045   null ARI -0.0040 ± 0.0205
k= 15  null sil -0.0587 ± 0.0079   null ARI +0.0018 ± 0.0318
k= 20  null sil -0.0718 ± 0.0068   null ARI -0.0071 ± 0.0186
k= 30  null sil -0.1028 ± 0.0096   null ARI -0.0022 ± 0.0390
k= 40  null sil -0.1288 ± 0.0107   null ARI -0.0089 ± 0.0244
k= 50  null sil -0.1576 ± 0.0152   null ARI -0.0062 ± 0.0271
k= 60  null sil -0.1805 ± 0.0124   null ARI -0.0075 ± 0.0261
```

The null silhouette falls monotonically and is k-dependent; the null ARI stays
within ±0.009 of zero throughout. The three prose statements are present as
quoted (cell 23 `left_open` "Both land at zero and only one of them had to";
cell 25 "Both anchors sit at zero"; cell 26 "Random assignment scores 0.00 on
both").

The consequence sub-claim also holds: at k=40, `(0.02 − (−0.1288)) / 0.0107 =`
**13.9** standard deviations above the measured null, so cell 25's "a silhouette
of 0.02 is not a discovery" is false as stated.

**Severity:** misleads a student
**Origin:** hand-written prose
**Fix:** print the measured null and say "the ARI null is zero by construction;
the silhouette null is not, and it depends on k" — the sentence the cell already
half-writes.

---

### Claim 3 — the silhouette curve is drawn against an anchor that is not the measured one
**Verdict:** CONFIRMED

**Evidence:** both halves are present verbatim.

```
cell 36: > **constraint** · draw the zero line — it is the anchor from section 7, …
cell 37: ax.axhline(0, color="#4b5563", ls=":", lw=2, label="random assignment")
```

Section 7's measured anchor over the nine plotted k values is **−0.0000 to
−0.1805** (table under claim 2), never 0 except at k=2. The line labelled
"random assignment" is not the random-assignment level at any k the plot shows,
and since the null is k-dependent no horizontal line can be.

**Severity:** misleads a student
**Origin:** hand-written prose (the `constraint` specifies the wrong line; the
code obeys it)
**Fix:** plot the measured null curve, or a ±1 sd band from
`random_assignment_scores`, instead of `axhline(0)`.

---

### Claim 4 — §10's premise "silhouette, which has a maximum" does not hold on this data
**Verdict:** CONFIRMED, with one correction to the claim's framing

**Evidence:** the notebook's own grid, `n_init=5`, `random_state=42`:

```
k    2      5      10     15     20     30     40     50     60
sil  .1516  .1045  .0850  .1018  .1139  .1250  .1465  .1581  .1695
best silhouette 0.1695 at k = 60      interior optimum? False
```

The maximum over the grid is at **k = 60, the last point searched**. Cell 37
prints `best silhouette 0.1695 at k = 60` directly under the heading
"Silhouette, which has a maximum", and cell 36's `student` bullet — "it still
peaks some distance from 40" — is false as printed: on the grid it does not peak
at all, it hits the edge.

**Correction to Phase A.** Phase A could not extend past k=80 and said so. I
completed it:

```
k=  60  +0.1695    k= 150  +0.2013    k= 300  +0.1143
k=  70  +0.1848    k= 200  +0.1759    k= 350  +0.0659
k=  80  +0.1873    k= 250  +0.1469    k= 390  +0.0245
k= 100  +0.1931
k= 120  +0.1919
```

So the silhouette **does** have an interior maximum on this data — at **k ≈ 150**,
almost four times the true 40. The claim as worded ("does not hold on this data")
is too strong; what is confirmed, and is the defect that reaches the reader, is
that the maximum lies **outside the notebook's own grid**, so the section's
printed evidence contradicts its heading and its `student` bullet.

**Severity:** misleads a student
**Origin:** hand-written prose (heading and `student` bullet), against generated
code that prints the contradiction
**Fix:** extend the grid past the turning point, or rewrite the section around
what the printed curve actually shows — a criterion that keeps rewarding finer
partitions until k ≈ 150.

---

### Claim 5 — §13's optimism experiment does not measure what it says
**Verdict:** CONFIRMED

**Evidence:** cell 50 as written, then the same experiment with k fixed at 60 in
advance (no selection step at all):

```
seed 0: selected k=60 0.1744   held out 0.0947   optimism +0.0797
seed 1: selected k=60 0.1697   held out 0.0909   optimism +0.0787
seed 2: selected k=60 0.1847   held out 0.0758   optimism +0.1089
seed 3: selected k=60 0.1772   held out 0.0473   optimism +0.1299
seed 4: selected k=60 0.1850   held out 0.0990   optimism +0.0860
mean optimism +0.0967   chosen ks [60, 60, 60, 60, 60]

selected-k gap  mean +0.0967  per seed ['0.0797','0.0787','0.1089','0.1299','0.0860']
fixed k=60 gap  mean +0.0967  per seed ['0.0797','0.0787','0.1089','0.1299','0.0860']
excess attributable to selecting k: +0.0000
fixed k=40 gap  mean +0.0664  per seed ['0.0484','0.0531','0.0819','0.0785','0.0699']
```

Identical seed for seed to four decimals. The selection picks k=60 on all five
seeds, so **zero** of the reported optimism comes from selecting k; the whole
+0.0967 is the ordinary fitted-versus-held-out gap at that k. The section's
thesis is true and its evidence is confounded — the §2.1/§2.2 shape, in the
section about uncontrolled comparisons.

Sub-claim also confirmed: cell 49's `catch` says "the optimism is small, and a
single seed cannot distinguish it from noise". Measured, it is +0.0967 against a
per-seed range of 0.0787–0.1299, all five positive, and more than half of the
selected score (0.0967 / 0.178). Every single seed shows it.

**Severity:** misleads a student
**Origin:** generated code (the experiment design), with the `catch` bullet's
"small … cannot distinguish it from noise" being prose contradicted by that code
**Fix:** widen the candidate grid so the selection is not pinned to the edge, and
subtract a fixed-k control so the printed number is the *excess* attributable to
selecting.

---

### Claim 6 — §13 misses the edge-of-grid defect, and its cross-references point at the wrong lecture
**Verdict:** CONFIRMED

**Evidence, part 1 (the missing check).** Cell 47's candidates are
`[5, 10, 20, 40, 60]` and the winner is the largest:

```
  k=5 sil +0.1045   k=10 sil +0.0850   k=20 sil +0.1139   k=40 sil +0.1465   k=60 sil +0.1695
best k = 60, silhouette = 0.1695   (grid max = 60)
```

Lecture 2 ships the check verbatim in `tools/make_notebooks.py`:

```
766:  check="detect whether the winner sits on the EDGE of the grid and say so — an
        optimum at the boundary means the optimum may lie outside it",
784:  if best_n == max(grid["model__n_estimators"]):
785:      print("\n⚠ the winner sits on the EDGE of the grid — the optimum may lie "
```

Lecture 9 does not mention it. Given claim 4 (the true peak is at k ≈ 150), the
boundary warning would have been *correct* here, which makes the omission worse.

**Evidence, part 2 (the cross-references).** The four references are present as
quoted. Applications in this course are dataset-paired:
L1–L2 California housing, L3–L4 MNIST, L5–L6 Titanic, L7–L8 CoverType,
L9–L10 Olivetti. So "application 1" and "the first application" = lectures 1–2,
and "the previous application" = lectures 7–8 (CoverType).

- cell 46 `student`: "the **previous** application's grid search could not report its own best score" — **wrong**
- cell 48: same phrase — **wrong**
- cell 46 `catch`: "the rule from **application 1**" — correct (L2 line 768: "reading `best_score_` as the model's accuracy … optimistic by construction")
- cell 51: "the **first** application" — correct

Two of four wrong, as claimed. The previous application is lecture 7, whose grid
search is explicitly *not* used to choose (`lecture_07.py:352` "cross-validation
does not get a vote on max_depth here").

**One inaccuracy in Phase A's own evidence, which does not change the verdict:**
it says `GridSearchCV` "appears in `lecture_06.py` and `lecture_07.py` only".
It also appears in `tools/make_notebooks.py` at lines 603 and 774 — i.e. in
lecture 2, which is the very lecture Phase A goes on to name as the right target.
The grep line is sloppy; the conclusion it supports is right.

**Severity:** misleads a student
**Origin:** hand-written prose (the cross-references); the missing edge check is
a code omission
**Fix:** change "the previous application" to "the first application" in cells 46
and 48, and add the `best_k_reported == max(candidates)` warning to cell 47.

---

### Claim 7 — §8.1, the defect is announced four times before the reader reaches it
**Verdict:** CONFIRMED

**Evidence:** grep for `defect` and `⚠` across the notebook returns exactly the
four sites claimed, in order:

```
cell  0 (md): Cells marked **⚠ read before running** contain a defect on purpose.
cell 45 (md): Here is a real request and the code it returns. **⚠ Read before running.**
cell 46 (md): > **Prompt · ⚠ what the assistant returns**
cell 46 (md): > **constraint** · … and the defect is subtler than that
```

And cell 46's own bullets give the answer away above the cell:

```
> * **Left open:** that the score reported is a MAXIMUM over a noisy criterion,
    evaluated on the same data that chose it. Five candidate values of k, five
    noisy estimates, and we print the largest.
```

Cell 48 then restates it as a discovery. This is §8.1's lecture-19 finding
reproduced exactly.

**Severity:** wrong but harmless (nothing false reaches the reader; the section
simply cannot do its job)
**Origin:** notebook structure
**Fix:** run cell 47 unannounced, have the reader write the number down, and open
§13's post-mortem with the ⚠ — §8.1's "preferred shape".

---

### Claim 8 — §6.1, sixteen code cells and sixteen full three-bullet annotations
**Verdict:** CONFIRMED

**Evidence:** counted by regex over `tools/notebooks/lecture_09.py`:

```
prompt calls 16
left_open 16   student 16   catch 16
input 16   output 16   constraint 16   check 4
```

Sixteen boxes, every one carrying all three notes. §6.1's budget is five to eight,
never more than ten.

**Severity:** wrong but harmless
**Origin:** notebook structure
**Fix:** keep the short `input · output · constraint · check` spec on all sixteen;
reserve the three bullets for the five to eight cells where the prompt genuinely
failed — cells 46, 49 and 53 are the obvious keepers.

---

### Claim 9 — a `constraint` naming a dependency the notebook does not have
**Verdict:** CONFIRMED

**Evidence:** cell 5's constraint reads "`shuffle=False` — the ten photographs of
each person stay adjacent, **which every montage below relies on**". I re-fetched
the corpus with `shuffle=True, random_state=0` and re-ran all three montages'
indexing:

```
shuffled: is y sorted? False   first 20 targets: [13 30 34 19 24  6 15 26 14 21 …]
cell10 still yields 40 images, one per person: (40, 64, 64)   all 40 people present: True
cell12 shape unshuffled (20, 64, 64)   shuffled (20, 64, 64)
cell12 shuffled set of person-0 images == unshuffled set: True
cell43 uses a boolean mask over `lab` (row-aligned with images) — ordering irrelevant
fetch_olivetti_faces default shuffle = False
```

All three montages are boolean masks over `y` or `lab`, never index ranges, and
`y` is permuted with `X`, so every one still returns the right faces. The stated
reason is fiction, and `shuffle=False` is scikit-learn's own default, so the
argument overrides nothing either.

**Severity:** misleads a student (§6.2 — a reason invented rather than observed,
in the field whose purpose is to state what must be true of the method)
**Origin:** hand-written prose
**Fix:** replace the reason with a true one — e.g. "`shuffle=False` is the
default and is stated so the row order is reproducible" — or drop the clause.

---

### Claim 10 — a caption that does not describe its picture
**Verdict:** CONFIRMED

**Evidence:** cell 43 on the shipped seed:

```
sizes min 4, max 25, exactly-10 count 7
single-person clusters 16 of 40
best  cluster  3: size  9, people  1, purity 1.00, montage draws  9
worst cluster  5: size 24, people 10, purity 0.21, montage draws 10
```

The worst panel's title renders as "worst cluster: **24** photographs, 10 people,
purity 0.21" above a montage of **ten** — `montage(ax, images[lab == c][:10], ncol=10)`.
The best panel happens to be consistent (9 and 9), so only one of the two rows is
wrong, which makes it harder to spot. This is §5.4 in a cell whose whole purpose
is looking carefully at a picture.

Minor sub-claim also confirmed by reading: `plt.subplots(2, 1, figsize=(11, 3))`
gives each row ~1.4 inches of height for a strip 10 × 64 px wide, so the faces
render squashed.

**Severity:** misleads a student
**Origin:** generated code
**Fix:** either drop `[:10]` and let `montage` wrap, or say "10 of {sizes[c]}
photographs" in the title.

---

### Claim 11 — "the next cell" points two cells away
**Verdict:** CONFIRMED

**Evidence:**

```
cell  9: * **Left open:** … 'what varies within a person', which the next cell answers.
cell 10: montage(ax, np.array([images[y == p][0] for p in range(40)]), ncol=10)
         title "one photograph of each of the 40 people"
cell 12: montage(ax, np.concatenate([images[y == 0], images[y == 22]]), ncol=10)
         title "ten photographs of one person, then ten of another"
```

Cell 10 is one photograph each of forty *different* people and structurally
cannot show within-person variation. Cell 12 answers it. Cell 11's own `student`
bullet says so: "looking at the grid of forty … the forty-face montage cannot
show it."

**Severity:** wrong but harmless
**Origin:** hand-written prose
**Fix:** "which the montage two cells below answers", or move the sentence to
cell 11.

---

### Claim 12 — a name bound to two kinds of object
**Verdict:** CONFIRMED

**Evidence:** every module-level binding of `s`, `a`, `b` in the notebook:

```
cell 24: s, a = random_assignment_scores(k)       # both ndarray, shape (20,)
cell 47: s = silhouette_score(X, km.labels_)      # np.float64 scalar
cell 50: a, b = perm[:200], perm[200:]            # ndarray, 200 row indices
cell 50: s = silhouette_score(X[a], km.labels_)   # np.float64 scalar
```

`s`: ndarray → scalar. `a`: ndarray of 20 ARI values → ndarray of 200 row
indices — same type, different meaning, which §4.1 covers ("one name, one
meaning"). Neither is remarked on. `best_model` is dead as claimed:

```
cell 47: best_score, best_k_reported, best_model = -2, None, None
cell 47: best_score, best_k_reported, best_model = s, k, km
```

— assigned twice, read nowhere.

**Severity:** wrong but harmless
**Origin:** generated code
**Fix:** rename to `sil_null, ari_null` in cell 24, `sil` in cells 47/50, and
`half_a, half_b` in cell 50; drop `best_model` or use it.

---

### Claim 13 — figures Phase A checked and found correct
**Verdict:** CONFIRMED (they do reconcile)

**Evidence:** every figure in Phase A's list, re-derived independently:

```
400 photographs of 40 people, 10 each        ✓  (all five asserts in cell 6 pass)
corpus MB = 6.55                             ✓
total pairs 79800  within 1800  between 78000 ✓  (both cell-16 asserts pass)
780 pairs among the forty, 22 same-person    ✓
distinct people seen at least once: 25       ✓
inertia monotone assert                      PASS
  … and at the library-default n_init:       PASS
kneedle picks k = 15   (truth 40)            ✓
sizes min 4, max 25; 7 clusters of exactly 10; 16 single-person clusters  ✓
silhouette-diagram panel means: k=2 0.152, k=10 0.085, k=40 0.146         ✓
```

The judgement call also holds: cell 17's "overlap **heavily**" sits above a cell
whose printed statistic is `6% of same-person pairs are further apart than the
median different-person pair` (raw 6.44%), with means 8.38 and 12.41. 6% is a
modest overlap and the printed number is the better sentence — §1.4's "name the
operation" applied to an adverb.

**Severity:** cosmetic (the "heavily" wording only)
**Origin:** hand-written prose
**Fix:** "the two distributions overlap — 6% of same-person pairs are further
apart than the median different-person pair."

---

### Claim 14 — checks that came back clean
**Verdict:** CONFIRMED

**Evidence:** re-run mechanically over the shipped `.ipynb`:

```
§5.1/§5.2 violations: NONE   (all 40 markdown cells; no prose line indented ≥4
                              outside a fence, no fence marker indented at all)
```python blocks in markdown: NONE
markdown cells containing any fence: [26, 55]   — both commitment sheets
```

§4.2 idempotence, by reading: every fitting cell constructs its estimator inside
itself (cells 29, 47, 50). The cell-20 hazard is real — `rng` is module-level and
consumed:

```
first draw  : [25 31 32 34 36 48 69 74] …
second draw : [ 2 14 35 53 55 58 76 90] …
identical on re-run? False
```

Re-running cell 20 alone silently redraws the forty labels, changing every ARI
downstream. The notebook does not name this (§4.3).

§7 carry-out-ability: no GPU needed, 4 MB download, peak memory a 400 × 400
distance matrix. A student alone at home can run all of it — much faster than the
notebook promises, which is claim 1.

**Severity:** n/a (this entry reports clean checks); the unnamed cell-20 re-run
hazard within it is *wrong but harmless* until a reader re-runs it, then it
misleads
**Origin:** notebook structure
**Fix:** add "⚠ re-running this cell alone redraws the forty labels — restart and
run all instead" to cell 19, or seed the draw locally with
`np.random.default_rng(RANDOM_STATE)` inside cell 20.

---

## Summary

```
confirmed: 14   false positive: 0   unverifiable: 0
of the confirmed, 8 mislead a student
origin split — prose: 8   code: 4   structure: 3
```

(Origins sum to 15, not 14: claim 6 is genuinely split — the cross-references are
prose, the missing edge-of-grid check is a code omission. Claims 13 and 14 are
"clean" reports; their residual sub-findings are counted as prose and structure
respectively.)

**Duplicates — same underlying defect counted more than once:**

- **Claims 2 and 3** are one root cause: the silhouette's null is not zero and is
  k-dependent. Claim 2 is the prose saying otherwise (cells 23, 25, 26); claim 3
  is the plot drawing `axhline(0)` (cell 37). One fix — measure the null and use
  it — closes both. Two distinct artefacts, so both are worth listing, but a
  rebuild should treat them as one item.
- **Claims 4, 5 and 6** share a root cause: **k = 60 is the top of every grid in
  this notebook, and the silhouette is still rising there.** Claim 4 is the §10
  grid `[2…60]` peaking at its edge; claim 5 is §13's selection pinning to k=60 on
  all five seeds, which is *why* the selection contributes zero optimism; claim 6
  is the absence of the edge-of-grid warning lecture 2 already ships. Widening the
  grid past k ≈ 150 changes all three at once.
- **Claim 1's** "roughly 1 minutes" is second-order to claim 1 and Phase A already
  nests it there.
- **Claims 7 and 8** are both §6/§8 annotation-budget problems and would be fixed
  by the same pass over the prompt boxes, but they are separate rules.

**Note on scope:** the calibration cases in the brief (lectures 3 and 6) do not
appear in lecture 9, so this triage offers no independent read on them.

---

## Additional defects not in the Phase A report

All four were re-derived to the same evidence standard.

### A1 — §7 / carry-out-ability: cell 24 emits 120 lines of red `UserWarning` before its two result lines
**Verdict:** CONFIRMED

`adjusted_rand_score(y_audit, lab[audit])` is called with 40 samples and up to 40
distinct labels, which trips a scikit-learn guard on every call. Captured stderr
from cell 24 as written:

```
CELL 24: stderr lines emitted = 120
/opt/anaconda3/lib/python3.13/site-packages/sklearn/metrics/cluster/_supervised.py:49:
  UserWarning: The number of unique classes is greater than 50% of the number of
  samples. `y` could represent a regression problem, not a classification problem.
```

Forty calls × three lines each. Cell 29 emits the same warning once per k (nine
more), and every `adjusted_rand_score` call in the notebook triggers it. In Colab
these render as a red stderr block, so the notebook's *first* scored cell — the
one establishing the baseline that everything else is measured against — presents
as a wall of red warnings above two lines of result. A student alone at home
cannot tell whether the cell failed. This is exactly the reader §7 protects.

**Severity:** misleads a student
**Origin:** generated code
**Fix:** `warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.metrics.cluster")`
in the setup cell, with one sentence saying which warning and why it is expected.

### A2 — §8.3: "examinable" appears once in fifteen sections, and it is a *negative* marking on the setup cell
**Verdict:** CONFIRMED

```
=== 'examinable' occurrences ===
  cell 3 (code): # Not examinable: this is engineering hygiene, not machine learning. …
  total: 1   /   section headings matching '## N ·': 15
```

§8.3 requires every section to carry one of *examinable*, *not examinable —
engineering*, or *beyond the book, for context*. Fourteen of fifteen carry none,
and the one marking present is on the one section that plainly is not examinable.
This is precisely the count §8.3 records from lecture 19 ("appears **once** in the
whole of lecture 19, on the section that needed it least") — reproduced, section
count and all.

**Severity:** wrong but harmless
**Origin:** notebook structure
**Fix:** one marking per `## N ·` heading.

### A3 — §6.3/§6.4: only 4 of 16 prompt boxes carry a `check ·` clause
**Verdict:** CONFIRMED

```
prompt boxes: 16   with a `check ·` line: 4   (cells 5, 15, 19, 28)
```

§6.4 names `input · output · constraint · check` as the course standard
"because the `check ·` slot structurally forces an expected answer", and §6.3
asks for checks "whose answer can be worked out on paper before running". Twelve
of sixteen boxes have no check at all — including cell 36 (the silhouette curve),
cell 46 (the ⚠ defect cell) and cell 49 (the optimism experiment), which are the
three places in this notebook where a stated expected answer would have caught
claims 4, 6 and 5 respectively.

Phase A counted the three *notes* (claim 8) and did not count the four *spec*
fields; the two are separate rules and this one is unreported.

**Severity:** wrong but harmless
**Origin:** notebook structure
**Fix:** add a `check ·` to every box; at minimum to cells 36, 46 and 49 — e.g.
cell 46's should read "check · whether the winner sits at the top of the
candidate list".

### A4 — §8.2: the "defect" cell prints the identical number the honest cell already printed
**Verdict:** CONFIRMED

```
cell 29 (the honest sweep, n_init=5):  k= 60 … silhouette +0.1695
cell 37 (the honest report):           best silhouette 0.1695 at k = 60
cell 47 (the ⚠ defect cell, n_init=3): best k = 60, silhouette = 0.1695
```

Identical to four decimal places, because both land on the same k=60 partition.
So a reader who follows cell 45's instruction to read the ⚠ cell carefully sees a
number they have already written down twice as a legitimate result. There is
nothing in the output to catch — the defect is entirely in the *provenance* of the
number, and the notebook's own §12 has just spent a section teaching the reader to
interrogate outputs. Combined with claim 7 (the answer is stated in the box above
the cell), §13's trap has no surface on which a reader can be caught, which is the
opposite of §8.2's "the best trap is the one that is not labelled".

**Severity:** wrong but harmless
**Origin:** notebook structure
**Fix:** give cell 47 a candidate grid that does not coincide with the sweep
(e.g. `[3, 8, 18, 35, 55]`), so its reported number is one the reader has not
seen and cannot reconcile — which is the thing they are meant to notice.
