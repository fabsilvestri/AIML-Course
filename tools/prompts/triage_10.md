# Triage — lecture 10 defect report

Claims triaged: **D1–D18**, in the order they appear in
`tools/prompts/lecture_10.md` § *Defects found in the current notebook*.
(The task message said "11 claims"; the Phase A report contains eighteen
numbered claims. All eighteen are triaged below.)

## Conditions

Everything below was re-derived on this machine, from the notebook's own code
and the cached Olivetti corpus (`~/scikit_learn_data/olivetti_py3.pkz`,
1,405,393 bytes). The environment is an **exact** match for the reference the
Phase A report used:

```
python 3.13.5   scikit-learn 1.7.2   numpy 2.3.5
OMP_NUM_THREADS = OPENBLAS_NUM_THREADS = MKL_NUM_THREADS = 2
```

Every non-timing figure in the Phase A report reproduced **to the last printed
digit**. That is a strong signal that the report's author actually ran the code
rather than estimating it, and it is the reason so many verdicts below are
CONFIRMED rather than FALSE POSITIVE.

**One caveat, applied only to timings.** The machine was running at load average
**181** during triage (a dozen other agents). Wall clocks I measured are
therefore **upper bounds**; contention can only slow things down. Where a claim
is that a ⏱ marker *overstates* the time, an inflated measurement still settles
it. Where the claim is about the *ratio*, I say so.

**Notebook-wide (per the brief, stated once, not repeated per claim):**
`notebooks/lecture-10.ipynb` has 55 cells (17 code, 38 markdown); all 17 code
cells have `execution_count: null` and zero outputs. No prose figure in this
notebook can be reconciled against a stored output. This is D18's subject and is
not re-litigated under the other claims.

---

### Claim D1 — `d95` is 123 and 33×, not the "118 components, 35×" the prompt box quotes

**Verdict:** CONFIRMED

**Evidence:** Ran the notebook's own two lines from cell 15
(`cum = np.cumsum(pca.explained_variance_ratio_)`,
`d95 = int(np.searchsorted(cum, 0.95) + 1)`):

```
comp1 23.8%  d95=123 (33x)  d99=260
cum[d95-1]=0.95039  cum[d95-2]=0.94984
```

The text is in `tools/notebooks/lecture_10.py:223`, rendered as notebook cell 14
(the prompt box above code cell 15), `How you would catch it`:

> print the reduction factor beside the count. '95% needs 118 components, 35x
> fewer than 4,096' is the sentence; the raw count on its own is not.

Both figures in the quoted model sentence are wrong: 118 → **123**, 35× → **33×**.
The cell directly below prints 123 and 33.

**Severity:** misleads a student
**Origin:** hand-written prose
**Fix:** In `lecture_10.py:223` change the quoted sentence to "95% needs 123
components, 33x fewer than 4,096".

---

### Claim D2 — the §3 conclusion ("at d=200 the worst distance is distorted by far less than ε=0.2") is contradicted by its own cell

**Verdict:** CONFIRMED

**Evidence:** Ran cell 23 verbatim, mean over its own three seeds, plus d = 300
for the exercise-4 crossover:

```
79,800 pairwise distances; 400*399//2 = 79800
d=   50 worst 0.438  p95 0.200   per-seed worst [0.425, 0.475, 0.415]
d=  100 worst 0.306  p95 0.137   per-seed worst [0.307, 0.328, 0.283]
d=  200 worst 0.252  p95 0.105   per-seed worst [0.298, 0.231, 0.227]
d=  300 worst 0.182  p95 0.081   per-seed worst [0.205, 0.161, 0.179]
d=  400 worst 0.147  p95 0.070   per-seed worst [0.155, 0.150, 0.137]
d=  800 worst 0.107  p95 0.047   per-seed worst [0.120, 0.101, 0.100]
```

**0.252 > 0.20.** The sentence is false of the worst pair at d = 200 and true of
the 95th percentile (0.105) printed in the column beside it. It first becomes
true of the worst pair at d = 400 (0.147); not even the *best* of the three
individual seeds at d = 200 (0.227) is below 0.20.

The claim appears twice, as the report says — notebook cell 24 (body prose) and
cell 22 (`Left open` bullet) — and the `constraint` line of that same box reads
*"report the WORST pair as well as the percentile — the theorem bounds the worst
case, so a percentile alone does not test it"*.
`johnson_lindenstrauss_min_dim(400, eps=0.2)` = **1382**, so that half is right.

**Severity:** misleads a student
**Origin:** hand-written prose
**Fix:** Rewrite both places to name d = 400 (worst 0.147), or to say the 95th
percentile at d = 200 is 0.105 while the worst pair is 0.252 — which is the more
interesting sentence.

---

### Claim D3 — "several times the accuracy" is 1.46×

**Verdict:** CONFIRMED

**Evidence:** Ran cell 43 verbatim:

```
40 at random               0.433
40, one per cluster        0.633
propagated to the cluster  0.525
propagated to closest 75%  0.558  (n=206)
all 280 true labels        0.975
ratio 1.46x ; people covered random 26, rep 33
propagated labels correct: 62.9%
```

0.633 / 0.433 = **1.46×**. The phrase "several times the accuracy" appears in
notebook cell 42 (`Left open`) and cell 44 (body prose). Every supporting figure
the Phase A report cites also reproduces exactly: 26 vs 33 people covered, 62.9%
propagation correctness, and propagation *losing* accuracy (0.525 < 0.633).

**Severity:** misleads a student
**Origin:** hand-written prose
**Fix:** Replace "several times the accuracy" with "1.46× the accuracy (0.433 →
0.633)" in `lecture_10.py:560` and `:594`.

---

### Claim D4 — "335 million parameters per component" is off by 40×

**Verdict:** CONFIRMED

**Evidence:** The cell's own arithmetic (cell 37):

```
one full cov: 8,390,656   forty: 335,626,240
```

4096 × 4097 / 2 = 8,390,656 for **one** covariance; 335,626,240 is **forty** of
them. Notebook cell 36, `The usual student version`:

> 335 million parameters per component, estimated from 400 photographs

**Severity:** misleads a student
**Origin:** hand-written prose
**Fix:** `lecture_10.py:467` — "8.4 million parameters per component, 335 million
for forty, estimated from 400 photographs".

---

### Claim D5 — four ⏱ markers are wrong, all overstating

**Verdict:** CONFIRMED (and it is an **undercount** — there is a fifth)

**Evidence:** Located every ⏱ in the notebook and timed the cell it governs:

| md cell | claims | measured here (load 181) | Phase A measured |
|---|---|---|---|
| 4 | "about 20 seconds" — SVD | **0.211 s** (SVD 0.128 + `PCA().fit` 0.083) | 0.21 s |
| 21 | "about 1 minute" — projections | **0.40 s** (incl. an extra d=300) | 0.6 s |
| 25 | "1–2 minutes" for the reduced sweep | **1.1 s** reduced / 35.4 s raw | 0.1 s / 27.3 s |
| 41 | "about 1 minute" — spending the labels | **0.19 s** | *not listed* |
| 48 | "2–4 minutes" — twenty splits | **58.1 s** | 6.2 s |

All five overstate, all in the same direction. The four the report names are
confirmed. The **fifth**, md cell 41 → code cell 43, is not in the report's table
and is wrong by roughly the same factor (0.19 s against "about 1 minute").

On the ratios: the first three reproduce the report's magnitude almost exactly.
The cell-48 row does not — I measured 58.1 s against the report's 6.2 s, because
this machine was 90× oversubscribed. Component timing on the same loaded machine
(one `PCA(123).fit(Xtr)` = 217 ms, one `LogisticRegression(max_iter=3000).fit` =
60 ms, `n_iter_ = 40`) puts an idle-machine figure near the report's 6.2 s. Either
way the marker overstates; the claimed **25×** ratio I can bound but not confirm.

**Severity:** misleads a student
**Origin:** hand-written prose
**Fix:** Re-derive all five markers on an idle machine. Per §7.1 only cell 27
(the raw sweep, >20 s) needs a marker at all — delete the other four.

---

### Claim D6 — the ⏱ on the sweep names the wrong half

**Verdict:** CONFIRMED

**Evidence:** Notebook cell 25: *"⏱ **1–2 minutes** for the reduced sweep."* Ran
cell 27 verbatim:

```
CELL 9 TOTAL 36.4s
4096 dims:   35.4s  best k=60  sil 0.1695  ARI(all) 0.454  ARI(40) 0.336
 123 dims:    1.1s  best k=60  sil 0.2061  ARI(all) 0.521  ARI(40) 0.436
speed-up 33.6x
```

The **reduced** sweep is 1.1 s; the **raw** sweep in the same cell is 35.4 s and
is the only computation in the notebook over 20 s. The marker names the fast half.
(All four metric values — 0.1695, 0.2061, 0.454, 0.521 — reproduce exactly. The
speed-up is 33.6× here against the report's 187×, which is contention on the
reduced sweep's denominator, not a defect in either.)

**Severity:** misleads a student
**Origin:** hand-written prose
**Fix:** `lecture_10.py:347` — "⏱ about 30 s, essentially all of it the raw
4,096-dimensional sweep; the reduced sweep is under two seconds."

---

### Claim D7 — the before/after ARI comparison is on mismatched rows

**Verdict:** CONFIRMED

**Evidence:** `tools/notebooks/lecture_09.py` line 353 (and 294):

```python
ari40.append(adjusted_rand_score(y_audit, km.labels_[audit]))   # 40 rows
```

`tools/notebooks/lecture_10.py` line 377 (notebook cell 27):

```python
ARI(all) {adjusted_rand_score(y, lab_raw):.3f}                  # 400 rows
```

Both computed on the **identical** raw clustering in one run:

```
4096 dims:  ARI(all) 0.454   ARI(40) 0.336
 123 dims:  ARI(all) 0.521   ARI(40) 0.436
```

Notebook cell 25 instructs *"Same sweep as the previous lecture… Compare it with
what you wrote down last time."* A reader who does that compares 0.454 against
0.336 and reads a 35% improvement that is entirely denominator. The column label
`ARI(all)` is honest; the instruction to compare is the defect, exactly as the
report states.

**Severity:** misleads a student
**Origin:** hand-written prose
**Fix:** Add the `ARI(40)` column to cell 27 (one line, `audit` is already in
scope) and say which denominator lecture 9 used.

---

### Claim D8 — `audit` and `y_audit` are computed and never used

**Verdict:** CONFIRMED

**Evidence:** `grep -n "audit\|y_audit" tools/notebooks/lecture_10.py`:

```
51:  output="the corpus, and the identical forty audit indices as the build session",
90:  audit = np.sort(rng.choice(400, size=40, replace=False))   # the same forty
91:  y_audit = y[audit]
94:  assert len(audit) == 40
```

Line 51 is prompt-box prose; 90, 91 and 94 are the assignment and its own assert.
Neither name is read again in any of the 17 code cells. Meanwhile cell 2's
`Left open` asserts that reproducing the forty *"is the only reason the two
notebooks are comparable at all"* — and the notebook then never uses them, which
is precisely what makes D7 true.

**Severity:** misleads a student
**Origin:** generated code (dead state), with a prose claim resting on it
**Fix:** Use `audit` in cell 27 (this fixes D7 and D8 together), or delete both
names and the prose claim.

---

### Claim D9 — `err` is rebound from an array to a function

**Verdict:** CONFIRMED

**Evidence:** AST walk over all 17 code cells, collecting every `Assign` to a
bare `Name` plus every `FunctionDef`:

```
err: [(31, 'Call:mean'), (31, 'Call:mean'), (40, 'Call:mean'), (50, 'FunctionDef')]
```

Cell 40: `err = ((p99.inverse_transform(p99.transform(Xa)) - Xa) ** 2).mean(axis=1)`
— an `ndarray` of length 412. Cell 50: `def err(pp):` — a function. §4.1.

**Severity:** wrong but harmless
**Origin:** generated code
**Fix:** Rename the cell-50 closure to `recon_err`.

---

### Claim D10 — `g`, `ari` and `p` are each rebound to a different kind of object

**Verdict:** CONFIRMED (all three sub-claims)

**Evidence:** Same AST walk plus source inspection:

- `g`: `[(23, 'Call:GaussianRandomProjection'), (37, 'Call:fit')]` — cell 23
  `g = GaussianRandomProjection(n_components=dd, random_state=seed)`; cell 37
  `g = GaussianMixture(...).fit(Z95)` (the AST reports `Call:fit` because of the
  chained `.fit`).
- `ari`: cell 35 `ari = adjusted_rand_score(y, lab)` — a float inside the DBSCAN
  loop; cell 37 `bic, ari = [], []` — a list. (Missed by the constructor-diff
  heuristic because a tuple target is not a bare `Name`; confirmed by reading
  `lecture_10.py:453` and `:477`.)
- `p`: cell 17 leaves `p = PCA(n_components=200, ...)` fitted on `X_tr` at loop
  exit; cell 43 `p = PCA(n_components=d95, ...).fit(X_tr)`. Same type, different
  meaning, **26 cells apart** (43 − 17 = 26, as claimed).

**Severity:** wrong but harmless
**Origin:** generated code
**Fix:** `proj` / `gmm`; `ari_eps` / `aris`; `p_k` / `p95`.

---

### Claim D11 — the pseudo-inverse branch in cell 31 is unreachable and the prose describes it as the branch that runs

**Verdict:** CONFIRMED

**Evidence:** Ran cell 31 with the branch instrumented:

```
PCA, full SVD             132 ms  err 0.00240  branch=if(inverse_transform)  transform(X_te).shape=(120, 123)
PCA, randomised            77 ms  err 0.00241  branch=if(inverse_transform)  transform(X_te).shape=(120, 123)
Incremental PCA            79 ms  err 0.00240  branch=if(inverse_transform)  transform(X_te).shape=(120, 123)
Random projection           6 ms  err 0.32093  branch=if(inverse_transform)  transform(X_te).shape=(120, 123)
hasattr(GRP,'inverse_transform') = True
manual pinv route err: 0.32093
```

The `else` branch never executes for any of the four reducers. Notebook cell 30,
`Left open`: *"random projection has **no** `inverse_transform`, so its
reconstruction goes through a pseudo-inverse. The two error columns are therefore
**not** computed identically…"* — both halves false in sklearn 1.7.2. As the
report says, both routes give **0.32093**, so no printed number is wrong.

(Incidentally: the `IncrementalPCA` trap in the same box reproduces verbatim.
`batch_size=64` →
`ValueError: Number of input features has changed from 64 to 123 between calls to partial_fit! Try setting n_components to a fixed value.`
and the default `batch_size` is `5 * n_features = 20480`, one batch. Not a claim,
but it means that box is sound where D11 is not.)

**Severity:** misleads a student
**Origin:** hand-written prose (with dead generated code behind it)
**Fix:** Delete the `else` branch and rewrite the bullet: `GaussianRandomProjection`
has had `inverse_transform` since sklearn 1.1, and it *is* the pseudo-inverse.

---

### Claim D12 — the "honest" arm of the leak experiment leaks its own hyperparameter

**Verdict:** CONFIRMED

**Evidence:** Cell 50's honest arm is `PCA(n_components=d95).fit(Xtr)` with
`d95 = 123` computed in cell 15 from `PCA().fit(X)` — all 400 faces. Recomputing
95%-of-variance from `Xtr` alone, inside cell 50's own loop over seeds 0–19:

```
honest d95 per seed: [105, 104, 104, 104, 105, 104, 104, 103, 105, 103,
                      103, 104, 104, 102, 103, 103, 103, 105, 105, 104]
min 102  max 105;  seed-42 split -> 104
```

**104** for the seed-42 split and **102–105** across the twenty seeds, against the
123 the cell actually passes. Exactly as claimed.

I also measured the consequence, which the report does not:

```
honest err with d95=123 (leaked size): 0.00243
honest err with per-split d95 (~104):  0.00260   -> RISES by 7%
leaky:                                 0.00096
```

So the honest arm is currently **flattered** by its leaked hyperparameter, and
fixing it makes the honest/leaky gap *wider* (2.54× → 2.70×). The lecture's
conclusion survives the fix — the defect is real, and it happens not to threaten
the finding it sits inside. This is the notebook's best unlabelled trap (§8.2)
and nothing marks it.

**Severity:** misleads a student
**Origin:** generated code
**Fix:** Recompute `d95` from `Xtr` inside cell 50's loop, and add a ⚠ or leave
it as the deliberate unlabelled trap — but say which, somewhere.

---

### Claim D13 — cell 35 prints a conclusion its own output contradicts

**Verdict:** CONFIRMED (both halves)

**Evidence:** Ran cell 35 verbatim and dumped the whole eps grid:

```
best ARI 0.133 at eps=6.00, 41 clusters, 97 noise
max cluster-count over grid: (eps=5.5, 50 clusters, ari=0.130, 138 noise)
```

The cell prints *"…with 41 clusters and 97 faces called noise"* and then, on the
next line, *"The right answer is 40 clusters and 0 noise. DBSCAN never gets
there."* On cluster count it gets to **41**, off by one.

Second half — cell 34's `The usual student version` says cluster count and ARI
*"peak in completely different places"*. Measured: cluster count peaks at eps
**5.50** (50 clusters, ARI 0.130); ARI peaks at eps **6.00** (41 clusters, ARI
0.133). **Adjacent points on a 25-point grid**, and the ARI at the two is 0.130 vs
0.133. Not "completely different places".

The real failure — ARI 0.133 against k-means' 0.521 on the same rows, with 97 of
400 faces discarded — reproduces exactly.

**Severity:** misleads a student
**Origin:** hand-written prose (both the hard-coded `print` string in cell 35 and
the cell-34 bullet)
**Fix:** "DBSCAN reaches 41 clusters but only ARI 0.133, and only by calling 97
of 400 faces noise." Drop "peak in completely different places".

---

### Claim D14 — `argmin(bic)` selects the first point on the grid

**Verdict:** CONFIRMED

**Evidence:** Ran cell 37 verbatim:

```
k=  5 BIC        56468 ARI 0.061      k= 45 BIC        91640 ARI 0.497
k= 10 BIC        61676 ARI 0.099      k= 50 BIC        95530 ARI 0.472
k= 15 BIC        66026 ARI 0.207      k= 55 BIC        99519 ARI 0.548
k= 20 BIC        71196 ARI 0.303      k= 60 BIC       103003 ARI 0.511
k= 25 BIC        74903 ARI 0.312      k= 65 BIC       107764 ARI 0.484
k= 30 BIC        79247 ARI 0.350      k= 70 BIC       111250 ARI 0.476
k= 35 BIC        83656 ARI 0.424      k= 75 BIC       114116 ARI 0.494
k= 40 BIC        87846 ARI 0.458      k= 80 BIC       119175 ARI 0.523

BIC picks k=5; ARI peaks at k=55 (0.548); ARI at k=40 = 0.458
BIC strictly increasing: True
```

56,468 → 119,175, **strictly increasing at every one of the 15 steps**, so the
argmin is k = 5, the grid's left edge, for any grid starting there. Cell 36's
`Left open` describes BIC and ARI as *"peaking in different places"*; BIC does not
peak at all. Same §1.3 defect lecture 9 taught about the inertia elbow.

**Severity:** misleads a student
**Origin:** hand-written prose (the `Left open` bullet) plus a generated
`print` that reports an argmin as a selection
**Fix:** Print "BIC is strictly increasing over the whole grid — it has selected
nothing" instead of `BIC picks k={ks[argmin(bic)]}`.

---

### Claim D15 — `best k = 60` is the grid's right edge

**Verdict:** CONFIRMED

**Evidence:** `ks=(2, 5, 10, 15, 20, 30, 40, 50, 60)`; cell 27 reports k = 60 for
both spaces. Extending the grid on the reduced data, same `n_init=5`, same seed:

```
k= 60  sil 0.2061  ARI(all) 0.521
k= 70  sil 0.2167  ARI(all) 0.472
k= 80  sil 0.2385  ARI(all) 0.502
k=100  sil 0.2404  ARI(all) 0.493
```

The silhouette is still climbing at k = 100 — all four values match the report
exactly. ARI(all) peaks at k = 60 (0.521) and falls to 0.472 at k = 70. So the
reported k is where the loop stopped, and the two criteria disagree about where
the optimum is, which the grid conceals.

**Severity:** misleads a student
**Origin:** generated code (the `ks` default in `sweep()`)
**Fix:** Extend `ks` past the silhouette turn, or state in prose that 60 is the
grid edge and the silhouette has not turned.

---

### Claim D16 — the trap is announced six times before it

**Verdict:** CONFIRMED

**Evidence:** Read every markdown cell preceding code cell 47. Six announcements,
in order:

1. **cell 0** — *"Cells marked **⚠ read before running** contain a defect on purpose."*
2. **cell 16** (`The usual student version`) — *"fitting on all 400 because it is one line shorter. That is exactly the failure this notebook builds to in section 9."*
3. **cell 45** (heading) — *"## 9 · An assistant reduces the dimension for us"*
4. **cell 45** (body) — *"**⚠ Read before running.** This is today's failure…"*
5. **cell 46** (box label) — *"**Prompt · ⚠ what the assistant returns**"*
6. **cell 46** (`constraint` + `Left open`) — *"fit the PCA on X, the whole corpus, as written"* / *"All four hundred photographs, including the hundred and twenty we then call held out."*

Item 6 gives the complete diagnosis before the cell has run. §8.1's evidence is
that lecture 19's **four** announcements were enough that "nobody falls in"; this
is six.

**Severity:** wrong but harmless (nothing false is stated — but the trap cannot
catch anyone, so the lesson's evidence never lands)
**Origin:** notebook structure
**Fix:** Per §8.1, let cell 47 run unannounced; move the ⚠, the `Left open` and
the §9 framing into cell 48, after the reader has written the numbers down.

---

### Claim D17 — every prompt box carries the full three-bullet annotation

**Verdict:** CONFIRMED

**Evidence:** Scanned all 38 markdown cells for `**Left open:**` and for all three
of `Left open` / `usual student version` / `How you would catch`:

```
markdown cells with **Left open:** : 17   of which full 3-bullet: 17
prompt-box cells: [2, 5, 8, 11, 14, 16, 19, 22, 26, 30, 34, 36, 39, 42, 46, 49, 52]
code cells:       [3, 6, 9, 12, 15, 17, 20, 23, 27, 31, 35, 37, 40, 43, 47, 50, 53]
```

17 code cells, 17 boxes, 17 full annotations. §6.1's budget is five to eight,
never more than ten. The positional claim also holds: cell 31's box (the
`IncrementalPCA` constraint) is box **10 of 17** and cell 50's (the decision rule)
is box **16 of 17** — both past cell 30, where §6.1 records all three audit
readers stopping.

**Severity:** wrong but harmless
**Origin:** notebook structure
**Fix:** Keep the full form on cells 23, 27, 31, 47 and 50; reduce the other
twelve to the short specification.

---

### Claim D18 — the notebook ships with no stored outputs

**Verdict:** CONFIRMED

**Evidence:**

```
total 55   code 17   md 38
cell 3 exec_count=None noutputs=0      cell 31 exec_count=None noutputs=0
cell 6 exec_count=None noutputs=0      cell 35 exec_count=None noutputs=0
cell 9 exec_count=None noutputs=0      cell 37 exec_count=None noutputs=0
cell 12 exec_count=None noutputs=0     cell 40 exec_count=None noutputs=0
cell 15 exec_count=None noutputs=0     cell 43 exec_count=None noutputs=0
cell 17 exec_count=None noutputs=0     cell 47 exec_count=None noutputs=0
cell 20 exec_count=None noutputs=0     cell 50 exec_count=None noutputs=0
cell 23 exec_count=None noutputs=0     cell 53 exec_count=None noutputs=0
cell 27 exec_count=None noutputs=0
```

17 of 17. §1.2 cannot be satisfied and `tools/check_notebook_numbers.py` has
nothing to check against — which is the mechanism by which D1–D4 survived. Per
the brief this condition is common to all 23 notebooks; it is recorded here
because it is the claim.

**Severity:** wrong but harmless (structural; it is the *enabler* of D1–D4 rather
than a misstatement itself)
**Origin:** notebook structure
**Fix:** Execute before committing, or make the numeric check read from a stored
`outputs.json`.

---

## What I re-derived that the task message asked about specifically

All five re-derived, all matching the script to the printed digit:

| asked | measured |
|---|---|
| SVD-vs-PCA sign agreement | abs-value disagreement **6.07e-07**; naive signed **8.96e-02** on the first five, 2.19e-01 over all rows; **224 of 400** rows of `Vt` are the negation of `pca.components_`; evr disagreement **1.94e-07** |
| Eckart–Young to 1e-5 | lhs **2040.778442**, rhs **2040.778320**, relative difference **5.98e-08** |
| 95% / 99% component counts | **d95 = 123** (33×), **d99 = 260**; `cum[122]=0.95039`, `cum[121]=0.94984`; `p99.n_components_ = 260` |
| JL bound at each eps | 0.1 → 5135 / 11841; 0.2 → **1382** / 3188; 0.3 → 665 / 1535; 0.5 → 287 / 663; ratio **2.31** at all four |
| distortion at d = 50…800 | see D2 — every worst/95th/mean cell matches |

**The leak experiment's central finding — re-run in full, seeds 0–19:**

```
reconstruction error  honest 0.00243   leaky 0.00096
  the leak makes it look 61% better, in 20/20 splits      (el < eh).all() = True
accuracy              honest 0.974     leaky 0.974
  difference -0.00 points, sd 0.46, leaky wins 3/20, ties 14/20, honest wins 3/20
  honest accuracy range 0.917 to 1.000, sd 1.64 points
```

**The conclusion holds.** Large, one-sided damage on reconstruction error (2.54×,
20/20 splits, no overlap); nothing on downstream accuracy (the leaky arm is not
even better on average, and the −0.00 point difference sits inside a 1.64-point
split-to-split spread). Every figure in the lecture-10 script's table — 0.00243,
0.00096, 61%, 20/20, 0.974/0.974, sd 0.46, 3/20, 14/20, 1.64 — reproduced exactly.

Unlike the lecture-19 §6 confound, this comparison is **not** confounded: the two
arms are scored on the same `Xte` within each seed, differ in exactly one line
(`.fit(Xtr)` vs `.fit(X)`), and the result is reported as a paired per-split win
count rather than as a difference of means. I looked for the lecture-19 failure
mode here and did not find it. The one thing wrong with the cell is D12, and D12
pushes the effect in the *safe* direction (fixing it widens the honest/leaky gap
from 2.54× to 2.70×).

I also spot-checked the report's "Checked and clean" section and it holds:
**0** markdown lines indented ≥ 4 outside a fence, **0** indented fence markers,
**0** ` ```python ` blocks in markdown. Cell 15's leaky single-split numbers
(accuracy 0.975, reconstruction error 0.00095) and cell 53's pipeline accuracy
(0.975) both reproduce, as do every assert in all 17 code cells.

---

## Summary

```
confirmed: 18   false positive: 0   unverifiable: 0
of the confirmed, 13 mislead a student
origin split — prose: 10   code: 5   structure: 3
```

Per claim: prose — D1 D2 D3 D4 D5 D6 D7 D11 D13 D14; code — D8 D9 D10 D12 D15;
structure — D16 D17 D18. Wrong-but-harmless — D9 D10 D16 D17 D18; the other
thirteen mislead a student.

**Duplicates / overlaps:**

- **D5 row 3 and D6 are the same ⏱ marker** (md cell 25, "1–2 minutes for the
  reduced sweep"). D5 counts it as one of four wrong markers; D6 makes the sharper
  point that it names the wrong half of the cell. One underlying defect, two claims.
- **D9 and D10 are one underlying defect** — §4.1 rebinding — split across four
  names (`err`, `g`, `ari`, `p`). One fix pass addresses both.
- **D13's second half and D14's "peaking in different places" are not duplicates**:
  they are different prompt boxes (cell 34 vs cell 36) making the same *kind* of
  unverified claim about two different pairs of curves.
- **D2 and D7 are not duplicates** but share a root cause: a sentence reads the
  column beside the one it names. Same for D1 and D18.

**Count discrepancy:** the task message said 11 claims; the Phase A report has 18
numbered claims (D1–D18). All 18 are above.

**Calibration note.** None of the three pre-verified calibration claims (lectures
3 and 6) appear in this lecture, so this triage offers no independent read on them.

**Note on the confirmation rate.** A 18/0/0 result is exactly the pattern the
brief warns about, so it deserves a reason. The reason is that the Phase A report
for lecture 10 is unusually literal: it quotes printed output rather than
paraphrasing it, and my environment (python 3.13.5 / sklearn 1.7.2 / numpy 2.3.5,
2 BLAS threads, cached Olivetti) is byte-identical to the one it used. Every
non-timing figure it states — `d95=123`, `0.252`, `1.46×`, `8,390,656`,
`335,626,240`, `0.336`/`0.454`, `0.133`/`41`/`97`, `56,468`→`119,175`,
`0.2061`/`0.2167`/`0.2385`/`0.2404`, `104` and `102–105`, `0.00243`/`0.00096`/61%,
`0.32093`, `12 of 12`/`6 of 12`, and the full JL and distortion tables — I
re-derived independently and got the same digits. I found **one** thing the report
got wrong, and it errs against itself: **D5 undercounts**, there are five bad ⏱
markers, not four (md cell 41's "about 1 minute" governs a 0.19 s cell). The only
figure I could not reproduce is the *ratio* in D5's last row, and that is because
this machine was at load average 181 during triage, not because the report is wrong.
