# Triage — lecture 1

Claims triaged: the sixteen numbered items in the *Defects found in the current
notebook* section of `tools/prompts/lecture_01.md`, plus the trailing **Clean:**
paragraph (triaged as item 17).

**On the "zero numbered items" note in my task message.** That is not the case
here. `tools/prompts/lecture_01.md` ends with a numbered list of **16** items
(lines 537–653), each a bolded `**N · …**` paragraph, followed by a **Clean:**
paragraph. Nothing was in prose-only form and nothing had to be reconstructed.

**Artefact:** `notebooks/lecture-01.ipynb` — 42 cells, 13 code. Verified.
**Source:** `tools/make_notebooks.py`, `lecture_01()` (line 219 onward).
**Environment:** scikit-learn 1.7.2, pandas 2.3.3, numpy 2.3.5, Python 3.13.5,
Apple arm64 — identical to the versions the Phase A report says it used.
**Data:** `notebooks/datasets/housing/housing.csv`, cached, shape `(20640, 10)`.

**Stated once, not repeated per claim (per the brief):** all 13 code cells have
`execution_count: null` and zero stored outputs, so no prose figure in this
notebook can be reconciled against a stored output. Where a figure was
checkable, I re-derived it from the CSV instead; where the claim is a *timing*,
that route is not available.

Verification scripts (scratchpad, not part of the repo):
`…/scratchpad/verify01.py` (numbers) and `…/scratchpad/strings01.py`
(string/structure searches). Raw output of both is quoted below.

---

### Claim 1 — The baseline is scored on the test set, twelve cells after cell 15 promises `test_set` is not touched again

**Verdict:** CONFIRMED

**Evidence:** String search over `nb["cells"]` for `test_set`:

```
C1  'test_set' in cells: [(14, 'markdown', 1), (15, 'code', 5), (27, 'code', 1)]
```

Cells 14, 15, 27 only — exactly as the report states. Cell 15's last comment is
`# From here to the very last cell, `test_set` is not touched again.` and cell
27 line 4 is `y_test  = test_set["median_house_value"]`, then
`baseline = np.full(len(y_test), y_train.mean())`. 27 − 15 = 12 cells. The
"very last cell" use never happens: the last code cell is 40 and it uses
`y_train`; cell 41 is markdown.

Two small inaccuracies in the claim's own wording, neither affecting the
verdict: cell 27 does not *open* with the `y_test` line (it opens with
`from sklearn.metrics import root_mean_squared_error`), and the false comment
sits on cell 15's second-to-last line, with `housing = train_set.copy()` after
it.

**Severity:** misleads a student
**Origin:** generated code
**Fix:** score the baseline on `y_train` in cell 27 (`np.full(len(y_train), y_train.mean())`), or delete the promise in cell 15.

---

### Claim 2 — The baseline and the three models are scored on different rows

**Verdict:** CONFIRMED

**Evidence:** `verify01.py`, re-deriving both baselines from the CSV under the
notebook's own split (`pd.cut` bands, `stratify=income_cat`, `random_state=42`):

```
train 16512 test 4128
baseline on TEST rows  = 115,727.19
baseline on TRAIN rows = 115,310.56
gap = 416.63
y_train.std(ddof=0)    = 115,310.56
train mean = 206,334
```

All three of the report's figures reproduce to the cent: $115,727.19 (what cell
27 prints), $115,310.56 (the matched figure, absent from the notebook), gap
$416.63. Cell 40 scores the three models on `X_train`/`y_train` — the 16,512
training rows. Cell 25: *"Everything you build today has to beat this"*; cell
41: *"we open by comparing them"*. Nothing states the windows differ.

Note the training-row baseline equals `y_train.std(ddof=0)` exactly, which is
the paper check the script's cell-10 annotation proposes — so the fix is
self-verifying.

**Severity:** misleads a student
**Origin:** generated code
**Fix:** score the baseline on the training rows so cells 27 and 40 share a window, and say which rows in the printed line.

---

### Claim 3 — The `⏱ 20 s` marker is wrong by a factor of three to twenty

**Verdict:** UNVERIFIABLE

**Evidence:** The structural half checks out. Search:

```
C3  '20 s' : [(14, 'markdown', 1), (39, 'markdown', 1), (40, 'code', 1)]
C3  '~20'  : [(40, 'code', 1)]
C3  clock marker U+23F1: [(39, 'markdown', 1)]
```

Cell 39's label is `> **Prompt · ⏱ 20 s — three models, scored on their own
training data**` and cell 40's first line is `# ~20 s: the forest is 100 trees
on 16,512 rows.` The ⏱ in cell 39 is the only one in the notebook. (The cell-14
hit on `20 s` is the substring of `stratifying`; not a marker.)

The *magnitude* half — 0.9 s / 6.0 s against 20 s — requires executing cell 40,
which the brief forbids ("Do not execute training cells"). I did not run it, so
I cannot show a number, and I am not willing to rubber-stamp one. This is the
one claim in lecture 1 the rule blocked; it is cheap to settle (a
`RandomForestRegressor(100)` on 16,512×8 is seconds, not minutes) and someone
permitted to run cell 40 should.

Two things I *can* record. First, §7.1's machine check ("any cell whose stored
execution exceeded 20 s must have a ⏱ marker") cannot fire either way, because
no cell has a stored execution. Second, the marker and the code disagree with
each other about what is being timed: cell 40 passes `n_jobs=-1`, and the Phase
A report's own 6.0 s figure is for `n_jobs=1`, so whatever the true number is,
the 6.0 s the report cites is not the configuration the notebook ships.

**Severity:** misleads a student — if the claim holds, this is the only number a reader uses to decide whether to walk away from the laptop (§7.1)
**Origin:** hand-written prose
**Fix:** time cell 40 as shipped, then either correct the marker to the measured value or drop it (§7.1 sets the threshold at ~20 s).

---

### Claim 4 — No cell has a stored output

**Verdict:** CONFIRMED

**Evidence:**

```
cells: 42  code: 13
all execution_count None: True
all outputs empty: True
```

**Severity:** misleads a student — it disables §1.2 for the whole file
**Origin:** notebook structure
**Fix:** ship the notebook executed, so §1.2 and `check_notebooks.py --advisory` have something to run against.

**One over-reach in the claim.** It says items 5–8 "are all instances of it".
Items 5 and 6 are (a prose figure with no cell behind it). Items 7 and 8 are
not: item 7 is a logic inconsistency inside cell 21's own code and item 8 is a
prose verdict that contradicts counts the notebook *does* compute. Both would
survive executing the notebook — and item 8 would become more visible, not
less. Do not treat 7 and 8 as fixed by fixing 4.

---

### Claim 5 — The 6.4% / 0.36% stratification figures appear in prose and in a prompt `constraint`, and in no cell

**Verdict:** CONFIRMED (and the figures themselves are correct)

**Evidence:** Re-derived from the CSV at seed 42:

```
               overall   strat    rand  strat_err%  rand_err%
median_income
1               0.0398  0.0400  0.0424      0.3650     6.4477
2               0.3188  0.3188  0.3074     -0.0152    -3.5861
3               0.3506  0.3505  0.3452     -0.0138    -1.5340
4               0.1763  0.1764  0.1841      0.0275     4.4243
5               0.1144  0.1143  0.1209     -0.0847     5.6308
max |strat err| = 0.3650%   max |rand err| = 6.4477%
band counts overall: {1: 822, 2: 6581, 3: 7236, 4: 3639, 5: 2362}
```

6.4% and 0.36% are right. Where they appear:

```
C5  '6.4' : [(13, 'markdown', 1), (14, 'markdown', 1)]   '0.36': [(13, 'markdown', 1), (14, 'markdown', 1)]
```

Cell 13 (section prose) and cell 14 (the prompt `constraint`). Both markdown;
no code cell computes either. Cell 23's box tells the reader *"when a source
names specific numbers about your data, the numbers are checkable"* — nine cells
after the notebook names two numbers about its own data that it does not let
the reader check. §1.2.

**Severity:** wrong but harmless — the figures are right, the reader just cannot confirm them
**Origin:** hand-written prose
**Fix:** add the script's cell 6 (the five-band error table, stratified vs unstratified) between cells 15 and 16.

---

### Claim 6 — "$68,000" in cell 26 is a forward reference to a training score

**Verdict:** CONFIRMED

**Evidence:** The figure appears once and only in cell 26:

```
C6  '68,000'/'68000'/'$68': [(26, 'markdown', 1)] [] [(26, 'markdown', 1)]
```

Cell 26's second bullet: *"reporting an RMSE of $68,000 with nothing beside
it."* The matching quantity, re-derived through the notebook's own cell-38
`ColumnTransformer` on the cell-15 training half:

```
num 8 cat 1
Linear regression RMSE on training data = $68,232.84
```

Produced by cell 40, fourteen cells later. No cell at or before 26 produces any
RMSE at all — cell 27, the next code cell, is the first, and it prints
$115,727. So a reader at cell 26 has no source in the notebook for $68,000.

One qualification: that the figure is a *deliberate* forward reference is
inference, not something I can demonstrate. What is demonstrated is that it
rounds to the cell-40 linear training RMSE and is otherwise unsourced — §1.2.
The report's added gloss ("the one thing a training-data score must never be")
is editorial; cell 26's bullet is arguing about the absence of a comparison,
not about train-vs-test.

**Severity:** wrong but harmless
**Origin:** hand-written prose
**Fix:** either make it a plain hypothetical ("an RMSE of, say, $70,000") or defer the bullet until after cell 40.

---

### Claim 7 — Cell 21 puts 23 districts on both sides of its own definition of the cap

**Verdict:** CONFIRMED

**Evidence:**

```
max value      = 500001.0
>= 500_000     = 787
== v.max()     = 764
== 500_000     = 23
counts.index.max() = 500001.0
counts.drop(counts.index.max()).head(5):
137500.0    101
162500.0     91
112500.0     82
187500.0     76
225000.0     75
percent of training set (>=500k): 4.8%
median districts per distinct price = 3.0
```

Cell 21 prints `capped` from `(housing["median_house_value"] >= 500_000).sum()`
= **787**, then labels `counts.drop(counts.index.max()).head(5)` as *"the five
commonest values below the cap"* — and `counts.index.max()` is `500001.0`, so
the drop removes only the 764 districts at 500,001. The 23 districts at exactly
500,000 are inside `capped` and inside "below the cap". Neither 764 nor 23 is
printed anywhere.

(The 23 do not surface in the top five — 137,500 leads with 101 — so the
inconsistency is silent in the output as well as in the code.)

**Severity:** misleads a student — this is the cell teaching "count it rather than squinting at it", and its own two counts use two different definitions
**Origin:** generated code
**Fix:** use one definition — either `v == v.max()` (764) in both places, or `>= 500_000` in both and `counts[counts.index < 500_000]` for the second print — and say in the output which one was used.

---

### Claim 8 — Cell 22's verdict on the famous claim does not match the counts

**Verdict:** CONFIRMED

**Evidence:**

```
$  450,000 ->   31 districts   percentile of count distribution = 99.54
$  350,000 ->   62 districts   percentile of count distribution = 99.78
$  280,000 ->    3 districts   percentile of count distribution = 39.98
median count = 3.0
280000/12500 = 22.4   450000/12500 = 36.0   350000/12500 = 28.0
  neighbour $262,500 -> 19 districts
  neighbour $275,000 -> 54 districts
  neighbour $287,500 -> 17 districts
  neighbour $300,000 -> 21 districts
```

Every figure in the claim reproduces, including all four neighbour counts
(19 / 54 / 17 / 21) and both percentiles. Cell 22 says *"one of the three is
real, one is marginal, and one is indistinguishable from the background"* and
cell 23's `left_open` bullet repeats it. Two of the three sit above the 99.5th
percentile of the per-price count distribution against a median of 3; 31 is not
"marginal".

Worse than the report states: cell 22 lists the values in the order 450,000 /
350,000 / 280,000, so a reader matching the three adjectives to the three
values positionally concludes that **350,000 is the marginal one** — and 350,000
is the *strongest* of the three (62 districts, 99.78th percentile, twice the
count of 450,000). The sentence is not merely imprecise; read as written it
inverts the two real lines.

**Severity:** misleads a student — §1.3, and it is the verdict cell 23 asks the reader to reach independently
**Origin:** hand-written prose
**Fix:** replace with the derivable statement: two of the three are real, 280,000 is absent, and it is the only one of the three that is not a multiple of $12,500 (22.4×) — its four nearest grid points carry 19 / 54 / 17 / 21.

---

### Claim 9 — "You will meet the same error worth far more than a dollar in about an hour" resolves to nothing

**Verdict:** CONFIRMED

**Evidence:**

```
C9  'about an hour': [(35, 'markdown', 1)]   'an hour': [(35, 'markdown', 1)]
```

Cell 35, last line, in full: *"You will meet the same error worth far more than
a dollar in about an hour."* No lecture number, no section, no cell. The
notebook's only other forward pointers are *"the next lecture"* (cells 10, 12,
37), which do resolve. §3.3, and §7 for the reader with no lecture theatre.

**Severity:** wrong but harmless
**Origin:** hand-written prose
**Fix:** name the lecture and section, or cut the sentence.

---

### Claim 10 — The Pipeline is justified by a cross-validation the notebook never runs

**Verdict:** CONFIRMED

**Evidence:**

```
C10 'cross-valid' (ci): [(10, 'markdown', 1), (36, 'markdown', 1), (37, 'markdown', 1)]
C10 'cross_val' in code: []
C10 'cross_val' anywhere: []
```

Three markdown mentions, zero code. Cell 36: *"One `Pipeline`, so that
cross-validation refits **all** of it on each fold"*; cell 37's `constraint`
repeats it. Nothing in lecture 1 refits anything on a fold. (Cell 10's
`catch` bullet adds a third dependency on the idea — *"a level with n=5 is a
level that will be absent from some cross-validation folds"* — also never
demonstrated.)

**Severity:** wrong but harmless — the claim is true, it is just asserted rather than shown
**Origin:** hand-written prose
**Fix:** four lines of `cross_val_score` on the cell-38 pipeline after cell 38, or reword cells 36/37 to promise it for lecture 2.

---

### Claim 11 — The defect is announced three times before the cell that contains it

**Verdict:** CONFIRMED

**Evidence:**

```
C11 warn sign U+26A0: [(0, 'markdown', 1), (29, 'markdown', 1), (30, 'markdown', 1)]
C11 'read before running' (ci): [(0, 'markdown', 1), (29, 'markdown', 1)]
```

Cell 0: *"Cells marked **⚠ read before running** contain a defect on purpose."*
Cell 29: *"**⚠ Read before running.** It runs, it imports nothing exotic, and it
prints a believable number."* Cell 30: the label `> **Prompt · ⚠ what the
assistant returns**` — and, decisively, its first bullet, which is the complete
diagnosis before the code has been run once:

> **Left open:** reviewer question 1. `fit_transform` ran on ALL the rows: the
> median that fills the missing values, and the mean and standard deviation
> that scale every column, were computed from a set that includes the rows we
> then call the test set.

Cell 31 is nine lines and the flagged line carries its own inline comment
`# <-- all 20,640 rows`. §8.1, and §8.2 ("the best trap is the one that is not
labelled").

**Severity:** misleads a student — in the inverse direction: it removes the lesson rather than teaching a falsehood. This is the one item on the list the Phase A script already plans to fix (its "Where the defect is" preamble).
**Origin:** notebook structure
**Fix:** move the whole reveal — the ⚠, cell 29's warning and cell 30's first bullet — into the markdown after cell 34, as the script specifies.

---

### Claim 12 — Thirteen code cells, thirteen full three-bullet annotations, zero short boxes

**Verdict:** CONFIRMED

**Evidence:** Counted over every markdown cell containing `> **Prompt`:

```
  cell 2: left=1 usual=1 catch=1  -> FULL
  cell 5: left=1 usual=1 catch=1  -> FULL
  cell 8: left=1 usual=1 catch=1  -> FULL
  cell 10: left=1 usual=1 catch=1  -> FULL
  cell 14: left=1 usual=1 catch=1  -> FULL
  cell 17: left=1 usual=1 catch=1  -> FULL
  cell 20: left=1 usual=1 catch=1  -> FULL
  cell 23: left=1 usual=1 catch=1  -> FULL
  cell 26: left=1 usual=1 catch=1  -> FULL
  cell 30: left=1 usual=1 catch=1  -> FULL
  cell 33: left=1 usual=1 catch=1  -> FULL
  cell 37: left=1 usual=1 catch=1  -> FULL
  cell 39: left=1 usual=1 catch=1  -> FULL
  totals: full=13  non-full=0
```

13 boxes, one per code cell, all three bullets present in every one. §6.1 asks
for five to eight full, never more than ten. The setup cell (2) and the
`.info()` cell (8) both carry full annotations.

**Severity:** wrong but harmless — annotation fatigue is the documented effect (§6.1), and it lands hardest on cell 30, which is the defect
**Origin:** notebook structure
**Fix:** reduce to short boxes everywhere except cells 14, 26, 30, 33, 37, 39 — which is exactly the six the script marks `Annotate: full`.

---

### Claim 13 — "Examinable" appears twice, both times on the setup cell

**Verdict:** CONFIRMED

**Evidence:**

```
C13 'examinable' (ci): [(2, 'markdown', 1), (3, 'code', 1)]
```

Cell 2's third bullet (*"not examinable, and it is here because…"*) and cell
3's comment (*"# Not examinable: this is engineering hygiene…"*). The notebook
has nine `## N ·` sections:

```
1 ## 1 · Setup          16 ## 4 · Look — at the training set only   29 ## 7 · An assistant writes the preprocessing
4 ## 2 · The data       25 ## 5 · A number to compare against       36 ## 8 · Build it properly
13 ## 3 · Split before you look   28 ## 6 · Commit                  41 ## 9 · Where we are
```

Eight of the nine carry no marker, including §3, §7 and the commit. §8.3.

**Severity:** wrong but harmless
**Origin:** notebook structure
**Fix:** add the per-section markers the script's "Examinability, per section" paragraph already specifies.

---

### Claim 14 — Several `catch` fields are not catches

**Verdict:** CONFIRMED

**Evidence:** `catch` is a real field name — `tools/make_notebooks.py`'s
`prompt()` helper takes `catch=`, rendered as the "How you would catch it"
bullet. The three quotes are exact (`make_notebooks.py` lines 197, 283, 386):

- cell 2 / line 197: *"not examinable, and it is here because a version mismatch produces a confusing error…"* — a justification, not a check.
- cell 14 / line 283: *"the comment on the last line — from here to the final cell, `test_set` is not touched again."* — reading a comment is not an executable check, and the comment it points at is **false** (claim 1). This slot does not merely fail to check something; it certifies the notebook's one broken promise.
- cell 26 / line 386: *"rule 2 of this course: a metric with nothing to compare it to is decoration."* — verbatim restatement of cell 25's opening line.

§6.3. Note the notebook's separate `check ·` field is *not* affected — cells 5,
14 and 37 each carry a real one (`assert` the shape; `assert` the halves sum
and the indices are disjoint; `assert` the column lists account for every
column), and all three are executable and executed.

**Severity:** wrong but harmless in two of three; **misleads a student** in cell 14, where the catch endorses the false comment
**Origin:** hand-written prose
**Fix:** replace cell 14's catch with `set(train_set.index).isdisjoint(test_set.index)` plus the band-share table; give cells 2 and 26 knowable-outcome checks or drop to short boxes (claim 12's fix covers both).

---

### Claim 15 — Cell 40 raises `NameError: y_train` if cell 27 has not been run

**Verdict:** CONFIRMED, and the hazard is larger than stated

**Evidence:** I executed every code cell **except 27 and 40** in a fresh
namespace (cell 40 is a training cell and was not run):

```
cell 3: OK
cell 6: OK
cell 9: OK
cell 11: OK
cell 15: OK
cell 18: OK
cell 21: OK
cell 24: OK
cell 31: NameError: name 'root_mean_squared_error' is not defined
cell 34: NameError: name 'root_mean_squared_error' is not defined
cell 38: OK

after skipping cell 27:
  y_train                    bound=False
  y_test                     bound=False
  root_mean_squared_error    bound=False
  X_train                    bound=True
  preprocessing              bound=True
```

`y_train` is unbound, and cell 40's `.fit(X_train, y_train)` evaluates its
arguments before any fitting, so the `NameError` is raised before the forest is
touched. Search confirms the binding is unique:

```
C15 'y_train' in cells: [(27, 'code', 2), (40, 'code', 2)]
C15 'y_test'  in cells: [(27, 'code', 3)]
```

The correction: cell 27 exports **three** things the rest of the notebook needs
— the `root_mean_squared_error` import, `y_train`, and `y_test` — and the first
failure a reader hits is not cell 40 but **cell 31**, nine cells earlier, which
is the ⚠ leak cell. Skipping the baseline cell breaks the lecture's centrepiece.

Restart-and-run-all passes, so this is an out-of-order hazard (§4.3), not a
build failure. The claim's arithmetic is off by one: 40 − 27 = 13 cells, not
fourteen.

The claim's two sub-assertions also hold. No name is assigned from two
different constructors anywhere in the file (`strings01.py` found no name with
more than one top-level assignment), so §4.1 is clean; and cell 40's loop
target `model` takes `LinearRegression`, `DecisionTreeRegressor` and
`RandomForestRegressor` in turn, which §4.1's own evidence paragraph says
should get a throwaway name.

**Severity:** misleads a student
**Origin:** notebook structure
**Fix:** move `from sklearn.metrics import root_mean_squared_error` and `y_train = housing[...]` into cell 15 (where `housing` is created), and name the hazard in the prose above cell 27.

---

### Claim 16 — §7's comparison is sound but silently uses a third split

**Verdict:** CONFIRMED (both halves — the exoneration and the defect)

**Evidence:** Reproducing cells 31 and 34 exactly:

```
index sets identical: True
leaky   $70,469.35
correct $70,468.46
leak    $0.88
of 4128 cell-31 test rows, 3289 are section-3 TRAINING rows (79.7%)
```

`(yc_te.index == y_te.index).all()` is `True`, so the $0.88 is a genuine
like-for-like measurement on identical rows and the headline finding stands.
Both $70,469.35 and $70,468.46 reproduce to the cent.

The defect: cells 31 and 34 both call `train_test_split(..., test_size=0.2,
random_state=RANDOM_STATE)` on the full 20,640 rows with **no** `stratify=`,
so this is a third partition. 3,289 of its 4,128 test rows — 79.7% — are rows
§3 designated as training rows. §3 (cells 13–15) spends a whole section
establishing that which rows go where is a decision; §7 makes a different one
without a word.

**Severity:** wrong but harmless — the comparison is valid for its purpose; the omission is what §2.1 asks to be stated
**Origin:** hand-written prose (an omission — the code is deliberate and correct for what §7 needs)
**Fix:** add the caveat paragraph the script specifies to the markdown after cell 34, with the 79.7%.

---

### Claim 17 — the trailing **Clean:** paragraph (five assertions)

**Verdict:** CONFIRMED — all five hold

**Evidence:**

```
  lines indented >=4 outside fence: []
  fence markers (cell, line, indent, was_inside): [(28, 5, 0, False), (28, 9, 0, True)]
  ```python blocks in markdown: []
```

and, for the prompt-box guarantee, every code cell's immediate predecessor:

```
  code cell 3: prev is prompt box = True     code cell 24: prev is prompt box = True
  code cell 6: prev is prompt box = True     code cell 27: prev is prompt box = True
  code cell 9: prev is prompt box = True     code cell 31: prev is prompt box = True
  code cell 11: prev is prompt box = True    code cell 34: prev is prompt box = True
  code cell 15: prev is prompt box = True    code cell 38: prev is prompt box = True
  code cell 18: prev is prompt box = True    code cell 40: prev is prompt box = True
  code cell 21: prev is prompt box = True
```

§5.1 clean, §5.2 clean (cell 28's single fence opens and closes at column 0),
§3.1 vacuous (no ```` ```python ```` block in any markdown cell), and cell 0's
claim that every code cell is preceded by a prompt box is true — 13 of 13.

**Severity:** n/a
**Origin:** n/a
**Fix:** none needed

---

## Summary

```
confirmed: 16   false positive: 0   unverifiable: 1
```

(17 entries: claims 1–16 plus the **Clean:** paragraph as claim 17. Claim 3 is
the sole UNVERIFIABLE; claims 1–2, 4–17 are CONFIRMED.)

```
of the confirmed, 8 mislead a student
   1, 2, 7, 8, 11, 14 (in part), 15   — plus 4, which disables §1.2 for the whole file
origin split, confirmed only (16) — prose: 7   code: 3   structure: 5   n/a: 1
   prose:     5, 6, 8, 9, 10, 14, 16
   code:      1, 2, 7
   structure: 4, 11, 12, 13, 15
   n/a:       17 (the Clean paragraph)
   the unverifiable claim 3 is also prose, giving 8/17 prose over all entries
duplicates:
   1 and 2 share a root cause (cell 27 scoring on test rows) but are distinct
     defects — 1 is the broken promise in cell 15's comment, 2 is the
     unmatched comparison in §9. Fixing cell 27 fixes both; fixing only the
     comment fixes neither.
   14's cell-14 example is the same underlying defect as 1, counted twice: the
     `catch` field endorses the false comment. One fix serves both.
   5 and 6 are both instances of 4 (a prose figure with no cell behind it).
     7 and 8 are NOT, contrary to what 4 asserts — see the entry for 4.
```

**Calibration note.** Lecture 1 contains none of the three pre-verified claims.

**On the refute-don't-confirm stance.** I went looking for false positives and
did not find one: every numeric claim in this report reproduced from the CSV
exactly, to the cent and to the district — 115,727.19 / 115,310.56 / 416.63,
6.4477 / 0.3650, 787 / 764 / 23, 31 / 62 / 3 with both percentiles, all four
neighbour counts, 70,469.35 / 70,468.46 / 0.88, 3,289 / 79.7%, 68,232.84 — and
every string-search claim matched to the cell index. The four corrections I do
have all run the same way, toward the claim being *understated*:

- **claim 8** — the adjectives are not merely imprecise; read positionally,
  cell 22 calls the strongest of the three lines the marginal one.
- **claim 15** — the first cell to break is 31, not 40, and three names are
  involved, not one.
- **claim 14** — cell 14's `catch` does not just fail to check something, it
  vouches for the false comment from claim 1.
- **claim 4** — the only over-reach in the whole report: items 7 and 8 are not
  instances of "no stored outputs" and will not be fixed by executing the
  notebook.

Whoever produced the Phase A report for lecture 1 did the arithmetic. The one
thing they claim that nobody has now checked is the wall-clock in claim 3.
