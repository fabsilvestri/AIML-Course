# Triage — lecture 8, `notebooks/lecture-08.ipynb`

29 numbered claims in the Phase A report's *Defects found in the current
notebook*, plus its *Checked clean* and *Not checked* sections (triaged at the
end). Verdicts below are in report order.

**Verification machine.** Apple M4 Max, 16 cores (12P + 4E), 128 GB, Python
3.13.5, scikit-learn 1.7.2, numpy 2.3.5, pandas 2.3.3 — the same machine class
the Phase A report used. CoverType was already cached under
`~/scikit_learn_data/covertype`; nothing was downloaded. Every figure below was
re-derived by running the notebook's own code verbatim in three standalone
scripts (split + criterion + twenty-tree stability; the four ensembles,
importances and boosting; the ρ decomposition).

**Timing caveat, stated once.** Roughly a dozen other agents were running on
this machine while I measured, so every wall clock below is an *upper* bound
under contention. Where my figure differs from Phase A's (cell 28: 15.4 s here
vs 5.7 s there) contention is the reason, and it never changes a verdict — the
notebook's estimates are wrong by factors far larger than the contention.

**Stored outputs, stated once (per the brief).** All 21 code cells have
`outputs: []` and `execution_count: null`. No prose figure in this notebook can
be reconciled against a stored output; every numeric verdict below therefore
comes from re-running the cell, not from reading one.

---

### Claim 1 — cell 30 says the naive member comparison gives "20%"; it gives 7.7%
**Verdict:** CONFIRMED
**Evidence:** running cells 28 and 31 verbatim:
```
bag.classes_ [1 2 3 4 5 6 7]   member.classes_ [0 1 2 3 4 5 6]
naive 0.0765   mapped 0.7883
```
Cell 30's bullet: *"comparing `member.predict(X)` with `y_test` directly,
getting 20%"*. The number is 7.7%, not 20%.
**Severity:** misleads a student
**Origin:** hand-written prose
**Fix:** "getting 7.7%" — and say it is below the 14.3% of uniform guessing,
which is the tell the bullet is trying to teach.

### Claim 2 — §11's table quotes "bagged, 200 unconstrained trees | 89.8%"
**Verdict:** CONFIRMED
**Evidence:** cell 28 is `n_estimators=100`. Measured:
`oob 0.8946  test 0.8968` → **89.7%** for a **100**-tree ensemble. No 200-tree
ensemble is fitted anywhere in the notebook (grep for `n_estimators`: 100, 100,
100, 100, 100, 30, and `M=10`). Both the count and the accuracy in cell 65 are
wrong.
**Severity:** misleads a student
**Origin:** hand-written prose
**Fix:** "bagged, 100 unconstrained trees | 89.7%".

### Claim 3 — "forty of the fifty-four real columns are binary" — it is forty-four
**Verdict:** CONFIRMED
**Evidence:** `X_train.nunique()` →
`cols with <=2 distinct: 44   >2: 10`; the 44 are 4 `Wilderness_Area_*` + 40
`Soil_Type_*`. The sentence appears in cell 53's bullet and again in cell 55.
**Severity:** wrong but harmless (the cardinality-bias argument gets stronger,
not weaker)
**Origin:** hand-written prose
**Fix:** "forty-four of the fifty-four", both occurrences.

### Claim 4 — "the root split is chosen from 48,000 patches"
**Verdict:** CONFIRMED
**Evidence:** cell 19 fits on `train_size=0.9` subsamples; the loop's `Xs` has
**43,200** rows (printed: `CELL19 ... subsample rows 43200`). The sentence
appears in cell 21's "Left open" bullet and again in cell 25. 48,000 is the full
training set, which no tree in that experiment sees.
**Severity:** wrong but harmless (the hierarchy argument is unaffected)
**Origin:** hand-written prose
**Fix:** "43,200 patches", both occurrences.

### Claim 5 — "the sign is consistent" contradicts "gini wins on 8 of the 10 resamples"
**Verdict:** CONFIRMED
**Evidence:** re-running cell 11, the per-resample `entropy − gini` differences
in points are
```
[+0.31, -0.47, -0.82, -0.93, +0.19, -0.26, -0.07, -0.58, -0.43, -1.02]
```
and the cell itself prints `resamples entropy won  2 of 10`. Two of ten have the
opposite sign, so "the sign is consistent" is false in the same sentence of cell
12 that gives the correct 8-of-10 count.
**Severity:** misleads a student
**Origin:** hand-written prose
**Fix:** "the sign favours gini on 8 of the 10 resamples" — drop "consistent".

### Claim 6 — a result stated as real that is inside its own noise (§2.4)
**Verdict:** CONFIRMED
**Evidence:** cell 11 prints `entropy - gini  -0.41 +/- 0.43 points`
(re-derived exactly). Cell 12's headline is *"The effect is **real**"*. The sd
exceeds the magnitude of the mean; §2.4 requires that to be said where the
headline is stated, and cell 12 states the headline without it.
**Severity:** misleads a student
**Origin:** hand-written prose
**Fix:** replace "real" with "small and inside its own spread — what survives is
the 8-of-10 sign count and the identical root feature".

### Claim 7 — the ρ table contradicts the mechanism the prose attributes it to
**Verdict:** FALSE POSITIVE
**Evidence:** I reproduced cell 41 exactly:
```
               rho   sigma^2      V(1)     V(10)  predicted
bagging      0.078    0.1616    0.1622    0.0266     0.0275
forest       0.052    0.1740    0.1730    0.0240     0.0255
extra        0.072    0.1710    0.1701    0.0270     0.0281
extra_bs     0.051    0.1792    0.1757    0.0244     0.0262
```
The claim's arithmetic ("the bootstrap does ~3.5× more than feature subsampling
plus random thresholds combined") comes from `bagging → extra`, which changes
**three** things at once: it removes the bootstrap *and* adds feature
subsampling *and* adds random thresholds. That is the very confound §2.1 forbids.
The one-variable contrasts available in this table are:

| contrast | what changes | Δρ |
|---|---|---|
| bagging → forest | feature subsampling (bootstrap on in both) | **−0.026** |
| extra → extra_bs | the bootstrap (features + thresholds on in both) | −0.021 |
| forest → extra_bs | random thresholds (bootstrap + features on in both) | −0.001 |

Read on matched pairs, cell 42's sentence — *"Feature subsampling does most of
the work; random thresholds are largely a substitute for the bootstrap rather
than an addition to it"* — is **supported**: feature subsampling is the largest
single effect (0.026), and random thresholds add essentially nothing on top of a
forest (0.001) while recovering only 0.006 of the 0.021 the bootstrap is worth
when they replace it. The claim inverts the conclusion by using a confounded
contrast.
**Severity:** n/a
**Origin:** n/a
**Fix:** none needed. (If anything, cell 42 could quote the three matched deltas
above, which would make the point unarguable.)

### Claim 8 — "nothing gets ρ near zero, because all four ensembles ultimately saw the same rows"
**Verdict:** FALSE POSITIVE
**Evidence:** (b) first, because it decides the claim. "All four ensembles" is
the four **variants** of the table — bagging, forest, extra, extra_bs — and
those *do* all see the same rows: `experiment()` walks the same `pool` slices
for every `kind`, so variant-to-variant the training data is identical. The
disjointness cell 39 asserts is between the **K=10 training sets inside one
variant**, which is a different axis and is not what the sentence is about. The
next sentence in the same cell removes the ambiguity: *"ρ is not a property of
the algorithm — it is the share of the variance that comes from which data you
were given."* That is the correct statement of why τ² > 0 for all four.
(a) "near zero": at ρ = 0.052 the floor ρσ² = 0.0090 sits well above zero
against V(10) = 0.0240 — i.e. more members could still buy a further 2.7×, and
the achieved reduction is 6.1–7.2× out of 10, not 10×. The floor being visible
in the measurement *is* the point of cell 44. Calling 0.05 "near zero" is a
reading, not a re-derivation, and the notebook's reading is the one its own
figure supports.
**Severity:** n/a
**Origin:** n/a
**Fix:** none needed.

### Claim 9 — cell 36's summary is refuted by the table cell 35 prints
**Verdict:** CONFIRMED
**Evidence:** cell 35, re-run verbatim:
```
bagging    ensemble 0.8968   one member 0.7883
forest     ensemble 0.8847   one member 0.7360
extra      ensemble 0.8879   one member 0.7778
extra_bs   ensemble 0.8790   one member 0.7277
```
Cell 36: *"Every mechanism that makes the members less alike also makes each of
them **worse**."* Extra-trees members are **4.2 points better** than forest
members (77.8 vs 73.6). The second column is not monotone, and cell 36's own
next paragraph supplies the reason (`bootstrap=False`, so extra's member trains
on all 48,000 distinct rows).
**Severity:** misleads a student
**Origin:** hand-written prose
**Fix:** "every mechanism *that subsamples the rows* makes each member worse —
compare only the pairs that differ in one thing: bagging→forest, extra→extra_bs."

### Claim 10 — the notebook stores no cell outputs at all
**Verdict:** CONFIRMED
**Evidence:** `all(len(c["outputs"]) == 0 for code cells)` → `True`;
`execution_count` is `None` for all 21.
**Severity:** wrong but harmless *for this lecture* — it is a course-wide
condition, not a lecture-8 defect, and it is why every other numeric verdict
here had to be produced by re-running rather than by reading.
**Origin:** notebook structure
**Fix:** none at lecture level; it is a decision for the rebuild.

### Claim 11 — cell 19's two panels are on different rows and the caption says they are the same
**Verdict:** FALSE POSITIVE
**Evidence:** the caption (cell 60) reads *"Same model, same columns, same row
order."* In context "row order" is the order of the **bars** — cell 58's
constraint spells it out: *"same columns, same order, shared y-axis"*. The
caption makes no claim about which data rows each panel is measured on, and the
notebook states that difference explicitly, twice, in the two cells immediately
before: cell 55 — *"**It is measured on the training data.**"* — and cell 56's
prompt — *"input · the fitted forest and 3,000 **HELD-OUT** rows"*,
*"constraint · permute on **HELD-OUT** data"*. The row difference is the lesson
and the notebook says so; nothing denies it.
**Severity:** n/a
**Origin:** n/a
**Fix:** none needed. A one-clause improvement is still available — "left:
48,000 training rows; right: 3,000 held-out rows" — but it repairs an omission
in a caption, not a false statement.

### Claim 12 — the four-row ensemble table varies more than one thing per row
**Verdict:** CONFIRMED (as a fact; as a *defect* it reduces to claim 9)
**Evidence:** from cell 35's constructors: bagging = all 54 features per node +
bootstrap; forest = 7 features + bootstrap; extra = 7 features + random
thresholds + **no** bootstrap; extra_bs = 7 features + random thresholds +
bootstrap. So only `bagging → forest` and `extra → extra_bs` are one-variable
contrasts. Note what is *not* wrong: all four rows are scored on the same 12,000
test rows, so this is not a §2.1 mismatched-window defect; and cell 36 supplies
the isolating pair itself (*"`extra_bs` turns it back on, and the pair is what
separates the two effects"*). The only sentence that actually reads the table as
a progression is cell 36's monotone claim — claim 9.
**Severity:** wrong but harmless
**Origin:** generated code
**Fix:** none beyond claim 9's; optionally label each row with its two or three
active mechanisms.

### Claim 13 — "the legible depth-8 tree" is not the tree lecture 7 called that
**Verdict:** CONFIRMED, with one correction to the claim
**Evidence:** both trees fitted on the identical split:
```
L8 tree, min_samples_leaf=1 : acc 0.7334  leaves 206
L7 tree, min_samples_leaf=20: acc 0.7302  leaves 163
set(X_train.index).isdisjoint(X_test.index) -> True   (the split is identical)
```
`tools/notebooks/lecture_07.py:467` ships `AUDITABLE_LEAF = 20` with the
comment *"the brief, not the grid"*, and lecture 7 explicitly refuses
`min_samples_leaf=1` — *"'100% of the 1' — a single surveyed patch wearing the
grammar of evidence"*. Lecture 8 cell 5 rebuilds with `min_samples_leaf=1` and
then calls the result *"the legible depth-8 tree"* in cell 63 and *"the legible
depth-8 tree | 73.3%"* in cell 65. **Correction:** cell 65's 73.3% is lecture
8's *own* correctly rounded figure (73.34%), not a transcription of lecture 7's
— lecture 7's tree scores 73.02%, and it is *lecture 7's* summary table that
misprints it as 73.3%. So there is one defect here, not two: a label that
lecture 7 attached to a 163-leaf auditable tree is reused for a 206-leaf tree
whose leaves can hold one patch. Cell 4's "Left open" bullet does disclose the
parameter change, and cell 5 prints *"same split as the previous lecture"* —
true, and it is what makes the substitution easy to miss.
**Severity:** misleads a student
**Origin:** hand-written prose
**Fix:** in cells 63 and 65 write "the depth-8 tree, `min_samples_leaf=1`
(lecture 7 shipped 20)"; reserve "legible" for lecture 7's tree.

### Claim 14 — no ⏱ figure in the notebook names a machine
**Verdict:** CONFIRMED
**Evidence:** nine ⏱ estimates (cells 9, 17, 26, 33, 37, 52, 55, 61, 65); a
string search over all 69 cells for `CPU`, `cores`, `vCPU`, `M4`, `laptop`
returns nothing, and the only `Colab` mentions are about the sklearn version.
Measured against them on 16 cores (under contention):

| cell | notebook says | measured |
|---|---|---|
| 11 | 40 s | 2.9 s |
| 19 | 30 s | 3.9 s |
| 28 | 40 s | 15.4 s (5.7 s uncontended per Phase A) |
| 35 | 40 s | 3.8 s |
| 39 | 90 s | 1.1 s |
| 54 | 20 s | 1.8 s |
| 57 | 60 s | 57.6 s |
| 63 | 60 s | 5.6 s |
| 67 | 2 min | 9.6 s |

Eight of the nine are overstated by 3–80×; the ninth (cell 57) is the one
discussed under claims 16–17.
**Severity:** misleads a student (§7.1 — a reader cannot tell whether a slow run
is their machine or the estimate)
**Origin:** hand-written prose
**Fix:** state the reference machine once in the header and re-measure all nine.

### Claim 15 — the 90-second marker is on the wrong cell; the expensive cell has none
**Verdict:** CONFIRMED
**Evidence:** cell 37's *"⏱ about 90 seconds"* sits above cell 39, which loads
the cached array and defines two functions: **1.1 s** measured. The work is cell
41, which calls `experiment()` four times: **11.6 s** measured, of which
**8.6 s** is the bagging variant alone (forest 0.7 s, extra 1.0 s, extra_bs
1.2 s). Cell 40, the markdown immediately above cell 41, carries no ⏱ marker.
**Severity:** misleads a student (the untimed cell is the long one — the exact
§7.1 failure mode)
**Origin:** hand-written prose
**Fix:** move the marker to cell 40 and size it to cell 41's work.

### Claim 16 — `permutation_importance` is the one cell where the estimate is too *low*
**Verdict:** FALSE POSITIVE (as stated) — the residual true part is claim 17
**Evidence:** cell 55 says *"⏱ about 60 seconds"*. Measured, same call, same
data, `n_jobs=-1` as the notebook writes it: **57.6 s** — i.e. the estimate is
approximately *right*, not too low. Phase A measured 68.1 s and 140.9 s on a
second call in the same process; with a run-to-run spread of 57.6 / 68.1 / 140.9
on one machine, "too low" is not demonstrable. What *is* demonstrable, and worth
keeping: this is the only one of the nine ⏱ estimates that is not a large
overstatement, and it is the cell that will strand a reader on 2 vCPU. That is a
consequence of `n_jobs=-1`, which is claim 17.
**Severity:** n/a as stated
**Origin:** n/a
**Fix:** none needed here; fix claim 17 and the 60 s becomes an overstatement
like the other eight.

### Claim 17 — `n_jobs=-1` in cell 57 makes that cell ten to twenty times slower
**Verdict:** CONFIRMED
**Evidence:** identical call, identical data, same process:
```
PERM n_jobs=1   5.9 s
PERM n_jobs=-1 57.6 s          -> 9.8x slower
identical numbers both n_jobs: True   (np.allclose on importances_mean)
```
The forest being shipped to each worker is 100 trees / **755,163** leaves
(measured), one task per column, 55 columns. `permutation_importance`'s `n_jobs`
defaults to `None` (one job), so this is an override the notebook added, and it
is the single largest wall-clock cost in the notebook. Cells 28, 35, 39, 54 and
67 also pass `n_jobs=-1`, and there the parallel unit is a whole tree fit, so it
genuinely helps.
**Severity:** misleads a student (and blocks the reader §7 protects)
**Origin:** generated code
**Fix:** `n_jobs=1` in cell 57, and nowhere else.

### Claim 18 — peak memory is never stated and the notebook's own advice is reversed
**Verdict:** CONFIRMED
**Evidence:** cell 4's bullet: *"`del cover` after splitting. 250 MB held for no
reason is how a free runtime dies…"*, and cell 5 does `del cover`. Cell 39 then
binds `full = fetch_covtype(as_frame=False)`:
```
full.data nbytes 251 MB (239 MiB)  float64 (581012, 54)
```
alongside the still-live `X_train`/`X_test`, and never frees it. A search over
cells 42–68 finds no further use of `full`; its last use is inside
`experiment()`, called from cell 41.
**Severity:** misleads a student (the reader who followed the cell-4 advice is
handed the opposite in cell 39)
**Origin:** generated code
**Fix:** `del full` at the end of cell 41, and state the peak in cell 37.

### Claim 19 — `perm` is bound to two different types
**Verdict:** CONFIRMED
**Evidence:** binding scan of the code cells:
```
cell 39 | perm = rng.permutation(len(full.target))     -> ndarray, size 581012
cell 57 | perm = permutation_importance(rnd2, ...)     -> sklearn.utils.Bunch
```
§4.1 forbids exactly this, and cell 68's red-team question 2 sends the reader to
*"check where `permutation_importance` was computed"* — i.e. to a name that
means two things. (`rng` is rebound in cells 15, 39 and 54 but always to the
same type and always re-seeded; that one is harmless, as the claim says.)
**Severity:** misleads a student
**Origin:** generated code
**Fix:** rename cell 57's binding to `pimp` (`pi` is already the DataFrame).

### Claim 20 — a vacuous assert presented as a check
**Verdict:** CONFIRMED
**Evidence:** cell 39:
`assert len(pool) == K * N_Z and len(set(pool) & set(te)) == 0`. `te` and `pool`
are `perm[:6000]` and `perm[6000:156000]` — disjoint slices of a single
permutation, whose elements are unique by construction. The second conjunct
cannot fail for any seed, any `K`, any `N_Z`. Only `len(pool) == K * N_Z` is a
real check (and it fails only if `K*N_Z + N_TE > 581,012`).
**Severity:** misleads a student — this is the audit's "check that passes for
the wrong reason", in a cell whose prompt box advertises the disjointness assert
as the thing that catches contamination
**Origin:** generated code
**Fix:** assert something that can fail — e.g. that the K training slices are
pairwise disjoint *and* that `K*N_Z + N_TE <= len(perm)` — or drop the second
conjunct and say why it is free.

### Claim 21 — the `classes_` trap is announced four times before the reader reaches it
**Verdict:** CONFIRMED
**Evidence:** in reading order before cell 31 runs: cell 29 heading *"### One
trap, worth ten minutes of your life"*; cell 29 *"⚠ **Read before running.**
This is the shape question, and it costs an afternoon the first time"*; cell 30
prompt label *"⚠ the trap that costs an afternoon"*; cell 30 constraint *"show
the WRONG number first"*; cell 30 "Left open" — *"a member's `predict` returns
POSITIONS, not cover types"* — which states the answer outright. Five
announcements, the last of them the solution.
**Severity:** wrong but harmless numerically, but it defeats the cell's purpose
(§8.1: nobody falls into a trap flagged four times)
**Origin:** notebook structure
**Fix:** run cell 31 unannounced, have the reader write the number down, and
open the *next* markdown cell with the ⚠ and the explanation.

### Claim 22 — same pattern on the importance defect
**Verdict:** CONFIRMED
**Evidence:** cell 49 *"**⚠ Read before running.** It runs, it is fast, and the
top of the list is entirely sensible"*; cell 50 prompt label *"⚠ what the
assistant returns"*; cell 50 constraint *"print it as returned — it runs, it is
fast, and the top of the list is entirely sensible"* (the same sentence, twice).
Cell 52's *"Guess where it ranks among the 55 before running the cell"* is the
good version and arrives after all three.
**Severity:** wrong but harmless (same §8.1 cost as claim 21)
**Origin:** notebook structure
**Fix:** keep cell 52's guess-first framing; delete the two ⚠ pre-announcements.

### Claim 23 — §8.3, "examinable" appears nowhere as a section label
**Verdict:** CONFIRMED, with a correction to the count
**Evidence:** the string occurs **four** times, not two, and all four are the
negative form: cell 2 (*"not examinable"*, in a "How you would catch it"
bullet), cell 3 (code comment *"Not examinable: this is engineering hygiene"*),
cell 55 (*"**not examinable**"* about `permutation_importance`) and cell 56
(same, in a bullet). None of the twelve `##` sections carries one of §8.3's
three labels.
**Severity:** wrong but harmless
**Origin:** notebook structure
**Fix:** label all twelve sections; lecture_08.md already states which are
examinable (cells 1–11, 15–17, 19, 21).

### Claim 24 — all 21 prompt boxes carry the full three-bullet annotation
**Verdict:** CONFIRMED
**Evidence:** 21 cells begin `> **Prompt`; 21 of 21 contain all three of
`Left open:`, `usual student version`, `How you would catch it`. §6.1 caps this
at five to eight per notebook, never more than ten. The two boxes that carry the
lecture — the `classes_` trap (cell 30) and the decoy (cell 53) — sit at
positions 10 and 17 of 21.
**Severity:** misleads a student (by attrition — the measured effect §6.1 was
written for)
**Origin:** notebook structure
**Fix:** keep the full form on cells 30, 50/53, 56 and 40; reduce the rest to
the specification.

### Claim 25 — two "How you would catch it" bullets are not catches
**Verdict:** CONFIRMED
**Evidence:** cell 2: *"How you would catch it: **not examinable, and it is here
because a version mismatch produces a confusing error in a cell that has nothing
to do with versions.**"* — a justification. Cell 4: *"How you would catch it:
**`del cover` after splitting. 250 MB held for no reason…**"* — a memory tip
about a different line. Neither states how a reader would detect a wrong answer.
**Severity:** wrong but harmless
**Origin:** hand-written prose
**Fix:** cell 2 → "the assert fires on import, before any modelling"; cell 4 →
"the two printed sizes must be 48,000 and 12,000 or the split is not last
lecture's".

### Claim 26 — the setup box cites an API this notebook never uses
**Verdict:** CONFIRMED
**Evidence:** `root_mean_squared_error` occurs in exactly two cells — cell 2's
constraint and cell 3's comment — and in neither is it called; it appears in no
import and no expression anywhere in the notebook. It is a **regression** metric
and lecture 8 is a classification notebook. Both strings come verbatim from the
shared template: `tools/make_notebooks.py:148` (the `SETUP` comment) and
`:188` (the `SETUP_PROMPT` constraint); the metric is actually used in lecture 2
(`make_notebooks.py:388`). Nothing in lecture 8 requires scikit-learn 1.4 —
cell 28 passes the base estimator positionally, so even the
`base_estimator`→`estimator` rename does not bite here.
**Severity:** wrong but harmless (the version floor is still worth asserting;
the reason given is false for this notebook)
**Origin:** hand-written prose (shared template)
**Fix:** give lecture 8 its own reason, e.g. *"`BaggingClassifier`'s
`base_estimator` was removed in 1.4"* — which is true and is the trap cell 27's
box already discusses.

### Claim 27 — "the next cell is the control that answers it" points at the biased table
**Verdict:** CONFIRMED
**Evidence:** the bullet is in cell 50 (markdown). Cell **51** is
`imp = pd.Series(rnd.feature_importances_, …)` — the impurity table being
criticised. The control (`random_decoy`) is cell **54**, four cells later.
**Severity:** wrong but harmless (§3.3)
**Origin:** hand-written prose
**Fix:** "the control that answers it is three cells further on".

### Claim 28 — "a free runtime dies three cells later"
**Verdict:** FALSE POSITIVE
**Evidence:** the full sentence in cell 4 is *"`del cover` after splitting.
250 MB held for no reason is how a free runtime dies three cells later, blaming
the wrong cell."* It describes a counterfactual — what happens in a notebook
that does *not* delete `cover` — and this notebook deletes it in cell 5. There
is no cell being pointed at, so there is nothing for a reader to look up and
fail to find, which is what §3.3's evidence is about. Counting to cell 7 tests a
reference the sentence does not make.
**Severity:** n/a
**Origin:** n/a
**Fix:** none needed. ("a few cells later" would remove even the temptation to
count.)

### Claim 29 — "a confusing error twenty cells later" / "twenty cells from here"
**Verdict:** CONFIRMED (duplicate of claim 26)
**Evidence:** unlike claim 28 this *is* an assertion about this notebook — cell
2: *"on an older Colab image the failure is an ImportError twenty cells from
here"*; cell 3: *"a version mismatch produces a confusing error twenty cells
later."* Cell 3 + 20 = cell 23, a markdown prompt box; cell 2 + 20 = cell 22,
which imports nothing. And no cell in the notebook imports or calls anything
that needs scikit-learn 1.4 (see claim 26), so no ImportError occurs anywhere at
any distance. The count is idiomatic; the claim about this notebook is false.
**Severity:** cosmetic (it is fixed by the same edit as claim 26)
**Origin:** hand-written prose (shared template)
**Fix:** as claim 26; drop "twenty cells".

---

## Additional finding — not in the Phase A list

**Cell 24 prints `root feature: {'Elevation': 1}`, not `{'Elevation': 20}`.**
Verified by running the cell's own line:
```
Counter(roots) = Counter({(0, 3046.5): 13, (0, 3043.5): 4, (0, 3049.5): 2, (0, 3034.5): 1})
{X_train.columns[f]: c for (f, _), c in Counter(roots).items()}  ->  {'Elevation': 1}
```
`roots` holds `(feature, threshold)` pairs, so `Counter` has four keys — one per
distinct threshold — and the dict comprehension collapses them onto one key,
keeping the **last** count. The reader is shown `1` directly beneath prose
(cells 23 and 25) claiming the root feature is identical across all twenty
trees, and directly above *"the root is the same feature every time"*. The
number that would support the sentence, 20, never appears.

This also means the rebuild script is written against a wrong expectation:
`tools/prompts/lecture_08.md` cell 8's **Expect** block states
`root feature: {'Elevation': 20}`. Both need fixing, e.g.
`Counter(f for f, _ in roots)` mapped through `X_train.columns`.
**Severity:** misleads a student · **Origin:** generated code

---

## The report's own "Checked clean" and "Not checked" sections

Re-checked, since they are claims too:

- **§5.1/§5.2 markdown rendering — upheld.** Parsing all 48 markdown cells: no
  line indented ≥ 4 spaces outside a fence, no indented fence marker, and no
  fenced block at all. All tables are real markdown tables.
- **§3.1 code quoted in prose — upheld.** Zero ``` fences in any markdown cell,
  so nothing is quoted that could fail to exist.
- **§4.2 idempotency — upheld by reading.** Every training cell constructs its
  estimators inside itself; cell 35 reuses the already-fitted `bag` by design
  and refits the other three; cells 15, 39, 54 re-seed `rng`. I did not execute
  a restart-and-run-all.
- **Reproducibility of the split — upheld and re-derived.** Lecture 7
  (`lecture_07.py:48-51`) and lecture 8 (`lecture_08.py:39-42`) pass the same
  objects, sizes, stratification and seed; `len(X_train), len(X_test) =
  48000, 12000` and `set(X_train.index).isdisjoint(X_test.index)` is `True`.
- **Arithmetic in prose — upheld and re-derived.** ⌊√54⌋ = 7; C(20,2) = 190
  pairs (printed); 1 − e⁻¹ = 0.368; 581,012 × 54, labels 1..7; majority class 2
  at 48.76% (`baseline 0.4876`); depth-8 tree **206** leaves, forest **749,170**
  leaves, ratio **3,636.7 → 3,637**; pairwise disagreement **9.0%**; decoy rank
  **10 of 55** at **0.0399** with **45** columns below it, and `-0.0 ± 0.00223`
  on permutation; `assert rank < 30` passes. Also re-derived and matching the
  report: cell 8's `largest gap 0.0545 at p = 0.095`; cell 19's
  `mean 73.33% sd 0.26% (72.52% - 73.71%)`; cell 22's `9.0% (5.0% - 15.3%)` and
  `71.5%` unanimity; cell 24's `4` distinct root thresholds, `200 - 216` leaves,
  `26 - 31` columns; cell 63's `78.2%` at **11** iterations; cell 67's `6.5%`
  and `78.2%`.
- **"The published ⏱ figures" (not checked) — still not checked on 2 vCPU.**
  Same limitation here; my figures are 16-core and contended. Claims 14, 15 and
  17 do not depend on the target machine.
- **"The lecture's own ρ figure" (not checked) — still not checked.** Cell 37's
  *"20 training sets of 20,000 rows and 20 members … close but not identical"*
  is UNVERIFIABLE without running a ~4× larger experiment; I ran only the
  notebook's K=10, M=10, N_Z=15,000 configuration.
- **The three figures (cells 8, 44, 59) — still not rendered.** I ran the code
  paths but did not read the rendered page, so axes, legends, error bars and the
  `set_yticks(..., fontsize=7)` call remain unchecked by eye. UNVERIFIABLE here.
- **Reproducibility across runs — now checked, and it holds.** Every figure I
  re-derived matches the Phase A report to the digit, on a separate process and
  a separate day, including all four ρ values and τ² = 0.0126 / 0.0090 / 0.0122
  / 0.0092. Only wall clocks moved.

---

## Summary

```
confirmed: 24   false positive: 5   unverifiable: 0
of the confirmed, 13 mislead a student
origin split — prose: 14   code: 5   structure: 5
duplicates:
  5 & 6    one sentence in cell 12; one edit fixes both
  9 & 12   the same defect — cell 36's monotone claim against cell 35's table;
           12 restates it as a table-design point
  21 & 22  the same §8.1 pattern, two instances (classes_ trap, decoy)
  26 & 29  the same inherited SETUP_PROMPT sentence; 29 adds only the cell count
  16 & 17  the same cell; 16 falls once 17 is fixed
  2 & 13   two rows of one table (cell 65), fixed in one edit
  14 & 15  the same ⏱ family; 15 is the placement half of 14
```

Five claims are false positives, and two of them (7, 8) would have sent the
rebuild to *change a sentence that is correct*: claim 7's counter-argument uses a
three-variable contrast to overturn a conclusion that holds on every matched
pair, and claim 8 rests on reading "all four ensembles" as "all forty ensembles".
Claim 11 and 28 read a caption and an idiom as assertions they do not make;
claim 16's headline does not reproduce (57.6 s against a 60 s estimate).

The confirmed set is dominated by prose (14 of 24), which matches the audit's
prior. The five code defects are all in cells the reader is least able to check:
`n_jobs=-1` (cell 57), the unfreed 251 MB (cell 39), the `perm` rebinding
(cells 39/57), the assert that cannot fail (cell 39), and — new here — the root
counter that prints `1` where the prose says twenty (cell 24).
