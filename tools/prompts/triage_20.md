# Triage — lecture 20

Claims from the `Defects found in the current notebook` section of
`tools/prompts/lecture_20.md`, in the order they appear there (16 numbered
claims, not 10).

Everything below was re-derived from
`notebooks/datasets/ridership/CTA_-_Ridership_-_Daily_Boarding_Totals.csv`
with `python3` (pandas 2.3.3, scikit-learn 1.7.2, statsmodels 0.14.5,
torch 2.13.0). No training cell was executed.

**Stated once, not repeated per claim:** `notebooks/lecture-20.ipynb` has 48
cells, 15 of them code, and **every one of the 15 has an empty `outputs`
list**. No prose figure in this notebook can be reconciled against a stored
output, so every numeric claim below is reconciled against the data instead, by
re-running the cell's code.

**Baseline reproduction** (all four protocol numbers, exactly as the notebook
would print them):

```
df len 7639   pool len 1247   2016-01-01 .. 2019-05-31
X (1191, 56)  y (1191,)
random 5-fold   44,761   folds [51096 43678 39538 43738 45754]
forward 5-fold  51,880   folds [80506 38448 44838 37991 57615]
forward + gap   53,696   folds [88277 39123 45156 38268 57655]
NAIVE_MAE       55,398.56  (1,191 rows)
cut 952   test 239   holdout 52,451
protocol      MAE      margin
random     44,761       19.2%
holdout    52,451        5.3%
forward    51,880        6.4%
gap        53,696        3.1%
84% of claimed margin was protocol
```

---

### Claim 1 — the §2.1 margin table (cell 27) subtracts one 1,191-row baseline from four scores measured on four different windows

**Verdict:** CONFIRMED

**Evidence:** `NAIVE_MAE` is `float((target[mask] - naive[mask]).abs().mean())`
over all 1,191 rows = **55,398.56**. The four protocols score on:

```
protocol                rows scored            matched naive MAE   matched margin   notebook prints
random 5-fold           all 1,191               55,398.6 (55,386 fold-avg)   19.2%      19.2%
one forward hold-out    952..1190  (239 rows)   67,225.4                     22.0%       5.3%
forward 5-fold          201..1190  (990 rows)   57,248.9 (fold-avg)           9.4%       6.4%
forward 5-fold + purge  201..1190  (990 rows)   57,248.9 (fold-avg)           6.2%       3.1%
```

(fold-wise matched naive, forward splits: `[60357 45790 60307 44007 75782]`.)
The hold-out row is wrong by a factor of 22.0/5.3 = **4.2**, in the direction
that makes the honest protocol look worst.

Two independent corroborations that this is a defect and not a reading of mine:

* the notebook's own `catch` bullet, one cell above (cell 26), says *"the margin
  should shrink monotonically as the protocol gets stricter. If it does not,
  either a protocol is mis-implemented or **the baseline is being measured over
  a different set of days than the models**."* The printed column is
  19.2 → **5.3 → 6.4** → 3.1. It is not monotonic. The notebook prints the
  failure of its own check and continues.
* `notebooks/lecture-19.ipynb` cell 42's stored output already carries the
  matched figure: `Copying last week | 67,225 | -21.3%`. Lecture 19 computed
  67,225 on the last 239 days; lecture 20 reverts to 55,399 for the same window.

One correction to the claim's wording: the `left_open` bullet is in cell 26,
**one** cell above the code, not three.

**Severity:** misleads a student

**Origin:** generated code

**Fix:** compute the baseline per protocol on that protocol's own test rows
(`err[te].mean()` inside each split; `err[cut:].mean()` for the hold-out) and
print a `baseline` column beside `MAE` and `margin`.

---

### Claim 2 — the same error in the model-comparison table (cell 38); matched baseline for the two GRU rows is 67,225

**Verdict:** CONFIRMED

**Evidence:** cell 38 prints four rows against one `NAIVE_MAE`:
`copy last week` (rows 0–1,190), `folds_gap.mean()` (fold-averaged over rows
201–1,190), `gru_rail` and `gru_mae` (rows 952–1,190). Reproduced the shapes
without training:

```
mulvar shape (1247, 5)   Xm (1191, 56, 5)   cut 952   test 239
naive MAE, rows 952..1190 : 67,225.43
naive MAE, rows   0..1190 : 55,398.56
```

`mulvar` and `pool` share an identical 1,247-row index, so the GRU test rows are
the same 239 days as the hold-out. The `vs naive` column therefore subtracts a
1,191-day baseline from 239-day scores. Any GRU MAE between 55,399 and 67,225
will be printed as *worse than copying last week* when on its own rows it is
better.

**Severity:** misleads a student

**Origin:** generated code

**Fix:** replace `NAIVE_MAE` in this table with the per-row-set baseline
(67,225 for both GRU rows; 57,249 for the purged-CV row) and label the row count.

---

### Claim 3 — `ax.axhline(NAIVE_MAE)` in the horizon chart (cell 41) draws a line 10,000+ boardings too low, and lag-7 is unavailable past horizon 7

**Verdict:** CONFIRMED

**Evidence:** the horizon model is scored on `Xh[cut_h:]` — `Xh (1178, 56, 5)`,
`cut_h = 942`, so **236** test windows. Naive baselines recomputed on exactly
those 236 windows:

```
h= 1  lag7 65,493 (available)   lag14 75,785
h= 7  lag7 65,218 (available)   lag14 76,110
h= 8  lag7 65,372 (NOT available)  lag14 76,136
h=10  lag7 67,425 (NOT available)  lag14 77,888
h=14  lag7 67,165 (NOT available)  lag14 77,315
```

"NOT available" is arithmetic, not opinion: at horizon `h` the target sits at
input index `WINDOW + h - 1`, so its lag-7 predecessor is at `WINDOW + h - 8`,
which leaves the 0..`WINDOW`−1 input window as soon as `h ≥ 8`. Using it would
be reading a day the forecaster has not seen.

The line is drawn at 55,399 against an honest 65,146–65,493 (h ≤ 7) and
76,136–77,888 (h ≥ 8) — 9,700 to 22,500 too low. The chart title *"the margin is
spent by about day 7"* is read off that line. Whether the conclusion survives a
correct baseline needs the training run and I did not do it; what is wrong is
the line.

Minor correction to the claim: the lag-14 range is 76,136–**77,888** (h = 10),
not 77,873.

**Severity:** misleads a student

**Origin:** generated code

**Fix:** draw a step baseline — lag-7 for horizons 1–7, lag-14 for 8–14 — both
computed on `Xh[cut_h:]`, and say in the caption which is which.

---

### Claim 4 — the stationarity section contradicts its own output (cells 4, 5, 6)

**Verdict:** CONFIRMED

**Evidence:** ran cell 6 verbatim:

```
rail, as it is               ADF p =  0.0001   stationary   stat=-4.6129 lags=22 nobs=1224
first difference             ADF p =  0.0000   stationary   stat=-11.4460 lags=20 nobs=1225
seasonal difference (7)      ADF p =  0.0000   stationary   stat=-10.9321 lags=21 nobs=1218
```

All three print `stationary`. Against that:

* cell 4 markdown: *"Ridership is not stationary: there is a downward trend and
  a hard weekly cycle."*
* cell 5 `check`: *"the level series should fail to look stationary where the
  differenced ones do not."*
* cell 5 `catch`: *"If all three pass, you have read the sign backwards — which
  is the single most common ADF error and it never raises anything."*

A reader who follows the `catch` instruction literally is told they inverted the
sign when they did not. This is GUIDELINES §3.2 ("any check offered to the
reader must have been executed") in the section whose subject is reading a
verdict correctly.

The underlying maths in cell 4 is fine — the level series is not weakly
stationary; Jan–May means fall 639,234 → 617,790 → 602,579 → 583,030 across
2016–2019 and weekday means run 326,415 (Sun) to 743,081 (Thu) — but ADF tests
only for a unit root, and deterministic trend plus deterministic seasonality do
not produce one. The notebook never draws that distinction.

**Severity:** misleads a student

**Origin:** hand-written prose

**Fix:** say ADF tests for a *unit root*, predict the actual outcome (all three
reject), and move the non-stationarity evidence to the mean/variance figures
that do show it — or add a KPSS test, whose null is the other way round.

---

### Claim 5 — "the shuffle flattered the model by 7,119 boardings" (cell 21) compares scores on different rows from models trained on different amounts of data

**Verdict:** CONFIRMED

**Evidence:** cell 21 computes `folds_time.mean() - folds_random.mean()` =
51,879.65 − 44,760.78 = **7,118.87**. The two sides are not comparable:

```
shuffled KFold : scored on all 1,191 rows, train sizes [952, 953, 953, 953, 953]
TimeSeriesSplit: scored on rows 201..1190, train sizes [201, 399, 597, 795, 993]
```

Matched on both — shuffled out-of-fold predictions (`cross_val_predict`) versus
the 80/20 hold-out model, both scored on the same last 239 days:

```
shuffled OOF, last 239 : 52,705.2
time-split model, same : 52,450.7
difference             :    +254.5   (the leaky protocol is 0.5% WORSE)
```

This is GUIDELINES §2.1's confound reproduced in a new notebook. Lecture 19 got
7,689.93 from `holdout − shuffled`; lecture 20 gets 7,118.87 from
`forwardCV − shuffled`. Different pairing, same structure, same confound — so
lecture 20 does **inherit** the framing lecture 19 §6 was faulted for.

Two notes on the claim's wording, neither affecting the verdict: the notebook
prints `14%`, not `13.7%` (the format is `:.0f`); and the shuffled folds train
on 952–953 rows whose *composition* necessarily differs — matching the size is
the most that is available, and the composition difference is the leak itself.

**Severity:** misleads a student

**Origin:** generated code

**Fix:** delete the subtraction, or replace it with the matched one above and
argue the point structurally (rows *t* and *t*+1 share 55 of 56 coordinates;
fold spread as cell 24 already prints it is **3,758** shuffled versus **15,970**
forward and **18,627** gapped — the shuffle manufactured stability, not a better
mean), which is true whichever way the MAE moves.

---

### Claim 6 — "Set `gap=0`. How much of the margin comes back? That amount was adjacency" (cell 47) attributes to adjacency an effect confounded with training-set size

**Verdict:** CONFIRMED

**Evidence:** `TimeSeriesSplit(gap=56)` removes 56 rows from *every* training
fold, not just the adjacency:

```
plain  train sizes [201, 399, 597, 795, 993]
gapped train sizes [145, 343, 541, 739, 937]
```

so it changes adjacency and training size together. Size-only control — `gap=0`
with the **first** 56 rows of each training fold dropped, adjacency intact:

```
gap=0                          51,880
gap=56 (the notebook's)        53,696   (+1,816, what the exercise calls adjacency)
gap=0, first-56-dropped        58,334   folds [111100 38995 45232 38443 57899]
```

Removing 56 rows without touching adjacency costs 6,454; removing the 56
*adjacent* rows costs 1,816. Relative to a size-matched control the gap is
4,638 boardings **better**, i.e. opposite in sign to the exercise's premise. A
student who does exactly what the bullet says will attribute 1,816 to adjacency
and be wrong about both the size and the sign.

Caveat I record honestly: the control's mean is dominated by fold 1, whose
training set drops 201 → 145 rows (MAE 111,100). Over folds 2–5 the control is
only ~92 boardings above the gapped run. That weakens the size of the effect,
not the existence of the confound — which is visible in the train sizes above
without any modelling.

**Severity:** misleads a student

**Origin:** hand-written prose

**Fix:** ask for both controls and have the student conclude that neither
isolates adjacency, then point at the structural argument (55 of 56 shared
coordinates) as the one that survives.

---

### Claim 7 — "in March 2020 the level falls by roughly three quarters" (cell 43) against a printed 17.9%

**Verdict:** CONFIRMED

**Evidence:** cell 45, run verbatim:

```
level 2019 (Jan–May) 583,030
level 2020 (Apr–Aug) 104,395
ratio 17.9%  ->  a fall of 82.1%
```

Cell 43 says "roughly three quarters" (75%). The cell three cells later prints
the number that contradicts it. Mild, and self-correcting for a reader who runs
the cell — but it is a §1.1 transcription, and the honest word is "four fifths".

**Severity:** wrong but harmless

**Origin:** hand-written prose

**Fix:** "falls by more than four fifths".

---

### Claim 8 — "on 1,191 windows it overfits" (cell 28); the training set is 952, and no stacked model is built

**Verdict:** CONFIRMED

**Evidence:** cell 28 markdown reads *"Stacking three recurrent layers is the
reflex, and on 1,191 windows it overfits."* 1,191 is `len(Xm)`, the total; the
models train on `Xm[:cut]` with `cut = int(1191 * 0.8) = 952`, and are scored on
239. And `num_layers` occurs **0 times** in the whole `.ipynb` — every `nn.GRU`
is single-layer — so no stacked recurrent model exists anywhere in the notebook
and the overfitting claim is asserted, never shown.

**Severity:** wrong but harmless

**Origin:** hand-written prose

**Fix:** "on 952 training windows"; and either demonstrate the stacked model or
soften to "is known to overfit at this data size".

---

### Claim 9 — 15 code cells, 15 full three-bullet annotations, against a §6.1 budget of five to eight

**Verdict:** CONFIRMED

**Evidence:** counted directly from `notebooks/lecture-20.ipynb`:

```
markdown cells beginning "> **Prompt" : 15
cells containing "Watch this prompt"  : 15
code cells                            : 15
```

GUIDELINES §6.1: *"aim for five to eight per notebook, never more than ten."*
15 > 10. This is the course-wide pattern (465 boxes, all fully annotated), not
a lecture-20 invention — `tools/notebooks/_prompt.py` emits the three bullets
whenever `left_open`/`student`/`catch` are supplied, and `lecture_20.py`
supplies them for every box.

**Severity:** wrong but harmless — the measured cost is annotation fatigue, and
in this notebook the box that most needs reading (cell 26, the monotonic-margin
`catch`) is the 7th of 15

**Origin:** notebook structure

**Fix:** keep all 15 four-field specs; drop the three bullets on the ~8 cells
where the prompt did not actually fail, starting with cells 8, 11, 14, 32, 37.

---

### Claim 10 — no timing markers anywhere, on a notebook with three 120-epoch training cells

**Verdict:** CONFIRMED (the checkable part; see caveat)

**Evidence:** over the whole `.ipynb`: `⏱` occurs **0** times, `"minute"`
occurs **0** times. Cells 36, 38 and 41 all call `train(...)`, whose signature
is `epochs=120`, and none of the markdown above them states a duration.

Caveat, recorded rather than glossed: I did not execute the training cells, so
the report's *≈15 s per cell on an Apple M4 Max* is unverified here. Note that
GUIDELINES §7.1 triggers at *"over ~20 seconds"*, so on the report's own
measurement the mechanical rule is not crossed on that machine. The header of
this notebook recommends a **CPU Colab runtime**, where 120 epochs × 30
minibatches is very likely past 20 s — but that is an argument, not an output.

**Severity:** wrong but harmless — it does not misinform, it blocks; §7 records
4 of 6 exercises blocked in lecture 19 on exactly this

**Origin:** hand-written prose

**Fix:** time cells 36, 38, 41 once on a Colab CPU runtime and put a ⏱ with
both figures (laptop and Colab) in the markdown above each.

---

### Claim 11 — the string "examinable" occurs 0 times

**Verdict:** CONFIRMED

**Evidence:** `"examinable"` occurs **0** times across all 48 cells. GUIDELINES
§8.3 requires every section to carry one of *examinable* / *not examinable —
engineering* / *beyond the book, for context*. Lecture 20 has six numbered
sections and none is marked.

**Severity:** cosmetic

**Origin:** hand-written prose

**Fix:** add one marker per `##` heading.

---

### Claim 12 — `day_type` is one-hot encoded as A / U / W and never decoded

**Verdict:** CONFIRMED

**Evidence:** cell 30 asserts the column names
`["rail", "bus", "next_day_type_A", "next_day_type_U", "next_day_type_W"]` and
no cell explains them. Over the whole `.ipynb`: `"Saturday"` 0 occurrences,
`"Sunday"` 0, `"holiday"` 0. Decoded from the data:

```
day_type counts:  W 5336   U 1216   A 1087
A: 1,087 days, dayofweek {5: 1087}   -> every A day is a Saturday
U: 1,216 days, 1,091 Sundays + 125 non-Sundays
W: 5,336 days, all Mon–Fri
the 125 non-Sunday U days fall on: 1 Jan, 25–31 May, 3–5 Jul, 1–7 Sep,
                                   22–28 Nov, 25–26 Dec
```

i.e. New Year's Day, Memorial Day, Independence Day, Labor Day, Thanksgiving,
Christmas. GUIDELINES §7.4 exists for exactly this: those are not common
knowledge in Rome, and the notebook names none of them.

**Severity:** misleads a student — "A" is guessable as *weekend* and is in fact
Saturday only; a reader who guesses will mis-read the model's one calendar feature

**Origin:** hand-written prose

**Fix:** one line under cell 30: *W = weekday, A = Saturday, U = Sunday or US
public holiday (New Year's Day, Memorial Day, 4 July, Labor Day, Thanksgiving,
Christmas)*.

---

### Claim 13 — no run-to-run spread is ever measured, yet single-seed GRU results are tabulated as a comparison

**Verdict:** CONFIRMED

**Evidence:** `gru_mae` (cell 36) and `gru_rail` (cell 38) are each produced by
one call to `train()` after a single `torch.manual_seed(RANDOM_STATE)`; there is
no loop over seeds anywhere (`seed` occurs 5 times, all as
`RANDOM_STATE`/`manual_seed` set-up). Cell 38 then prints them in one table as
"gates helped" versus "more series helped". The comparable evidence from the
partner lecture, read out of `lecture-19.ipynb` cell 40's **stored** output:

```
Epoch 140  44,316.50
Epoch 160  52,307.17
Epoch 180  49,127.35
Epoch 200  48,964.94      spread across the last four readings: 7,990.67
```

against a margin over target of 894. Nothing in lecture 20 lets a reader tell
whether the gru_mae − gru_rail difference exceeds that.

**Severity:** misleads a student

**Origin:** notebook structure

**Fix:** run each GRU at three seeds and print mean ± spread; say in the same
sentence whether the difference clears it.

---

### Claim 14 — cell 35's "which is exactly what the next cell is for" points to the wrong cell

**Verdict:** CONFIRMED

**Evidence:** cell 35 (the prompt box for code cell 36) ends its *usual student
version* bullet with *"Two changes, one number, nothing learned — which is
exactly what the next cell is for."* The next cell is 36, which trains the
five-series GRU — the run with two changes in it. The separation of "gates
helped" from "more series helped" happens in cell **38**, two cells later.

Weaker than claims 1–6: read from cell 36's own position, cell 38 *is* the next
code cell, so the reference is ambiguous rather than plainly false. GUIDELINES
§3.3 asks for references that are counted, so it still fails.

**Severity:** cosmetic

**Origin:** hand-written prose

**Fix:** "which is what the rail-only run two cells below is for".

---

### Claim 15 — `cut` is rebound and "becomes a wrong split the moment `WINDOW` changes"

**Verdict:** FALSE POSITIVE

**Evidence:** the rebinding is real — cell 27 has `cut = int(len(X) * 0.8)` for
the numpy array `X`, cell 33 has `cut = int(len(Xm) * 0.8)` for the torch tensor
`Xm`. The asserted consequence is not. Both `X` and `Xm` derive their length
from the same 1,247-row date range with the same `WINDOW`:
`len(X) = 1247 − WINDOW`, and
`len(Xm) = len(mulvar) − WINDOW − horizon + 1 = 1247 − WINDOW` at `horizon = 1`.
They are equal identically, not coincidentally:

```
WINDOW=  7  len(X)=1240  len(Xm)=1240  cut_X=992  cut_Xm=992  equal=True
WINDOW= 28  len(X)=1219  len(Xm)=1219  cut_X=975  cut_Xm=975  equal=True
WINDOW= 56  len(X)=1191  len(Xm)=1191  cut_X=952  cut_Xm=952  equal=True
WINDOW= 91  len(X)=1156  len(Xm)=1156  cut_X=924  cut_Xm=924  equal=True
WINDOW=120  len(X)=1127  len(Xm)=1127  cut_X=901  cut_Xm=901  equal=True
```

No value of `WINDOW` produces the "wrong split" the claim predicts. Nor is this
a GUIDELINES §4.1 violation as §4.1 is written — §4.1 forbids rebinding a name
to *a different kind of object* (`model`: LinearRegression → SimpleRnn), and
`cut` is an `int` meaning "the 80% split point" both times. The advisory machine
check (§9, "a name assigned from two different constructors") would not fire
either: the constructor is `int(...)` in both cells.

The horizon cell correctly uses a distinct name (`cut_h = int(len(Xh) * 0.8)` =
942, since `Xh` has 1,178 rows at `horizon=14`), which is the one place the
lengths genuinely differ.

**Note for the rebuild:** the same false belief is load-bearing in the Phase A
report's *proposed* notebook — red-team exercises 4 and 5 (lines 1134–1142 of
`lecture_20.md`) tell the student the two bindings "happen to be equal at 952
and the bug is invisible until you change `WINDOW`". If that text ships, the
rebuilt notebook sends students to hunt a bug that cannot occur.

**Severity:** n/a (wrong but harmless as a claim; the proposed *fix* text would
mislead a student)

**Origin:** generated code

**Fix:** none needed in the notebook. Renaming cell 33's binding to `cut_m` is
a cosmetic tidy at most; the red-team text asserting the WINDOW hazard must not
ship.

---

### Claim 16 — none of the three red-team exercises names the cells to re-run, and one asserts an outcome no cell produces

**Verdict:** CONFIRMED

**Evidence:** cell 47's red-team section reads in full:

```
* Set `gap=0` in `TimeSeriesSplit`. How much of the margin comes back? That
  amount was adjacency.
* Give the model `df["day_type"]` **without** the `shift(-1)`. The score barely
  moves — explain why that is worse, not better.
* Train on 2016–2019, test on 2020. Then argue, in two sentences, whether the
  model was wrong or the question was.
```

No cell number appears. Verified against the notebook:

* bullet 2 requires rebuilding `mulvar` (cell 30), re-windowing (cell 33) and
  retraining (cell 36) in that order — three cells spanning a training run;
* bullet 3 requires editing two separate `"2016-01":"2019-05"` slices, at cell 3
  (`pool`) and cell 30 (`mulvar`), 27 cells apart. Editing one and not the other
  silently desynchronises the two data paths.

And *"The score barely moves"* is asserted with no cell in the notebook
producing it — the un-shifted variant is never run. GUIDELINES §7.2.

**Severity:** wrong but harmless — it blocks the exercise rather than
misinforming, except for the asserted outcome, which does misinform

**Origin:** hand-written prose

**Fix:** name the cells and their order in each bullet; delete "the score barely
moves" or run the variant and quote the number.

---

## Summary

```
confirmed: 15   false positive: 1   unverifiable: 0
of the confirmed, 8 mislead a student (1, 2, 3, 4, 5, 6, 12, 13)
origin split — prose: 9 (4, 6, 7, 8, 10, 11, 12, 14, 16)
               code:  5 (1, 2, 3, 5, and 15, the false positive)
               structure: 2 (9, 13)
duplicates: claims 1, 2 and 3 are one underlying defect — a single global
  NAIVE_MAE (55,399, 1,191 rows) used as the reference for scores measured on
  239 rows (cells 27, 38, 41) and 990 rows (cell 27) — counted three times, at
  three sites. Fixing it means one change of principle and three edits.
  Claim 5 is the same *class* of error (§2.1) but a distinct instance: it is a
  model-versus-model comparison across windows with no baseline in it, and it
  needs its own fix.
```

### Two questions the task asked directly

**Does lecture 20 resolve what lecture 19 left open, on matched rows?** No.
Lecture 19's cell 42 had already computed the matched 239-day baseline
(`Copying last week | 67,225`). Lecture 20's cell 27 re-derives 55,399 over
1,191 rows and prints the hold-out margin as **5.3%** — numerically identical to
the row lecture 19 labelled wrong — where the matched figure is **22.0%**. The
defect is not merely unresolved; it is re-committed in three cells (27, 38, 41),
one cell after a prompt box that names it (cell 26's `left_open`: *"does not say
the baseline must be scored on each protocol's own test days. Lecture 19 got
this wrong and said so in its section 9"*) and whose `catch` bullet gives the
exact diagnostic the printed table then fails.

**Does it inherit lecture 19 §6's confounded 7,690 framing?** Yes — claim 5.
Cell 21 prints `folds_time.mean() − folds_random.mean()` = **7,119** as "the
shuffle flattered the model", comparing 990 rows against 1,191 with training
sets of 201–993 rows against 952–953. On matched rows and matched training size
the leaky estimator is 254 boardings **worse** (52,705 vs 52,451), reproducing
GUIDELINES §2.1's finding in a new notebook.

### One item outside the numbered claims

The Phase A report's closing note is correct and I confirm it: GUIDELINES §6.4
says lecture 20 uses *"annotations with no boxes"*. It does not — all 15 code
cells carry the structured `input · output · constraint · check` box, which §6.4
itself names as the standard. Lecture 20's §6 defect is the budget (§6.1: 15
full annotations against a ceiling of 10), not the convention. §6.4's example
list needs correcting.

### Checked and clean (spot-checks of the report's own "clean" list)

* Cell 18 reproduces lecture 19's shuffled five-fold to the boarding:
  **44,760.77**, folds `[51096 43678 39538 43738 45754]`, against lecture 19
  cell 31's stored `Mean CV MAE: 44,760.77` and identical per-fold values. Its
  `catch` bullet demands exactly this and it holds.
* Cell 9's variance identity holds: lag 1 ρ=+0.419 predicted 198,145 measured
  198,098 (ratio 1.000, flagged WORSE, series sd 183,841); lag 7 ρ=+0.837
  predicted 104,821 measured 104,730; lag 14 ρ=+0.811 predicted 113,157 measured
  113,068. Cell 10's prose ("well under a half at lag 1") is correct.
* Data preparation matches lecture 19: 7,639 rows after `drop_duplicates()`,
  pool 1,247 days, 2016-01-01 to 2019-05-31 — the figure cell 2's `catch` tells
  the reader to check against.
