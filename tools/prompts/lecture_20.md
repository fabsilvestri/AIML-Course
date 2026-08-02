# Lecture 20 — *When the past does not repeat*

**Colab prompt script.** Application 10, second half. Follow it top to bottom at
a Colab keyboard; every code cell in the notebook comes from one numbered entry
below.

**Runtime: CPU.** Choose *Runtime → Change runtime type → CPU*. The three
training cells are GRUs over 56 steps on ~950 windows; they are small enough
that moving data to a GPU costs more than the arithmetic saves. Total training
time for the whole notebook is under two minutes on a fast laptop and under
twelve on a Colab CPU runtime. Every figure is stated per cell.

---

## Where this lecture starts

Lecture 19 ends holding a table it has itself labelled wrong. Its last code cell
prints three forecasts on the held-out period and an `Improvement vs NAIVE_MAE`
column computed against `NAIVE_MAE = 55,399` — a baseline measured over all
1,191 days, while the three models were measured over the last 239. Its §9 says
so, corrects the table by hand, and stops with the sentence *"Lecture 20 begins
here."*

So this notebook begins by finishing that correction properly, and then applies
the same test to every comparison it makes afterwards. **Whenever this script
asks you to compare two numbers, it names the rows both were scored on.** That
is the whole discipline; the rest is arithmetic.

Two figures inherited from Lecture 19, both re-derived from the ridership file
rather than transcribed:

| From Lecture 19 | Value | Rows it was scored on |
|---|---|---|
| naive baseline, "copy last week" | 55,399 | all 1,191 scorable days |
| the same baseline on the held-out window | 67,225 | the last 239 days, 2018-10-05 to 2019-05-31 |
| committed target (10% under 55,399) | 49,859 | — |
| RNN, final epoch | 48,965 | the last 239 days |

**One caveat to carry, and to repeat wherever it matters.** Lecture 19's
headline is that its RNN met the committed target. The margin is
`49,859 − 48,965 = 894` boardings. Its own last four held-out readings were
44,317 / 52,307 / 49,127 / 48,965 — a spread of **7,990**, which is **8.9 times**
the margin. That result is inside its own noise. Nothing below leans on it as
evidence; where it is quoted, it is quoted as a number to beat by more than
7,990, not as a result. Cell 16 exists precisely to stop this lecture repeating
the mistake with its own GRUs.

---

## Conventions used below

Each entry gives the prompt to type, in the course's standard four-part form
(**input · output · constraint · check**), then what must come back, then the
assertion that must pass. `Annotate: short` means the notebook keeps only the
prompt box above the cell. `Annotate: full` means the box is followed by the
three lines — *Left open* / *The usual student version* / *How you would catch
it*. **Six of the eighteen cells get the full annotation.** The rest do not, and
that is deliberate: an annotation on every cell is an annotation nobody reads by
cell 30.

Section markers, per house rule: every section is marked *examinable*,
*not examinable — engineering*, or *beyond the book, for context*.

---

# § 1 · Setup and the same data

*Not examinable — engineering hygiene.*

Markdown to put above the section: this notebook re-derives Lecture 19's data
exactly, so that any number that differs afterwards is the protocol and not the
data. If the assertion in cell 1 fails, stop: nothing downstream is comparable.

## Cell 1 — setup, load, and the modelling pool

**Prompt to type:**

> **input** · the CTA daily boardings CSV from `ageron/data`, downloading and
> extracting the tarball if it is not already in `datasets/`
> **output** · a frame indexed by date with `day_type`, `bus`, `rail`, and a
> `pool` series of rail boardings from 2016-01-01 to 2019-05-31
> **constraint** · sort by date, drop the redundant `total` column, drop exact
> duplicate rows; set numpy and torch seeds to 42
> **check** · print the number of days on file and the pool length, and assert
> the pool has 1,247 days running from 2016-01-01 to 2019-05-31

**Expect:** `7,639 days on file; 1,247 in the pool we model`. The raw CSV has
7,701 rows; 62 are exact repeats of a date already present, and dropping them
loses nothing (Lecture 19 checked that every repeated date carried identical
numbers, and this preparation reproduces its index exactly).

**Assert:**

```python
assert len(pool) == 1247, len(pool)
assert str(pool.index[0].date()) == "2016-01-01"
assert str(pool.index[-1].date()) == "2019-05-31"
```

**⏱** First run only: the tarball download is about 1 MB, a few seconds. After
that, well under a second.

**Annotate:** short

---

# § 2 · Stationarity, and a test that answers a different question

*Examinable.*

Markdown to put above the section, **before** cell 2 and containing no warning
about what cell 2 will print:

> A series is **strictly stationary** if the joint distribution of
> $(X_{t_1},\dots,X_{t_k})$ is unchanged when every index is shifted by the same
> $h$. That is more than anyone can check, so in practice we ask for **weak
> stationarity**: constant mean, constant variance, and an autocovariance
> $\gamma(h) = \operatorname{Cov}(X_t, X_{t+h})$ that depends on the lag $h$ and
> not on $t$.
>
> **Why a model needs it.** Fitting one set of weights to all of history assumes
> that what a value meant in 2016 is what it means in 2019. If the mean drifts,
> the model is averaging two different worlds and is right about neither.
>
> The standard instrument is the **augmented Dickey–Fuller test**. Run it on the
> level series and on two differences of it, and write the three verdicts down
> before reading on.

That last sentence is the whole staging. Do not say what the answer will be.

## Cell 2 — augmented Dickey–Fuller, three ways

**Prompt to type:**

> **input** · the rail pool, its first difference, and its seasonal difference
> at lag 7
> **output** · for each, the ADF statistic, the p-value, and a one-word verdict
> **constraint** · import `statsmodels` and `pip install` it only if the import
> fails, so the notebook runs on a bare environment; print which regression and
> lag-selection settings were used
> **check** · state in the printed line what the null hypothesis is, so the
> direction of the p-value is on the page and not in your head

**Expect:** three lines. With `statsmodels` 0.14.x and the defaults
(`regression="c"`, `autolag="AIC"`, 22 lags selected, 1,224 observations):

```
rail, as it is           ADF stat  -4.613   p = 0.000122   reject unit root
first difference         ADF stat -11.446   p = 0.000000   reject unit root
seasonal difference (7)  ADF stat -10.932   p = 0.000000   reject unit root
```

**Write the three p-values down now.** All three reject. Including the one the
section just told you is not stationary.

**Assert:**

```python
assert adfuller(pool.dropna())[1] < 0.05      # yes, the LEVEL series
```

That assertion is not a formality — it is the finding. If it fails, your
`statsmodels` is selecting lags differently and the numbers below will not match.

**⏱** Under a second (0.03 s for all three on an Apple M4 Max).

**Annotate:** full

- **Left open:** the prompt asks for a verdict but never says what the test is a
  verdict *about*. ADF's null hypothesis is that the series has a **unit root**.
  Rejecting it says the series is not a random walk. It does not say the mean is
  constant, and it is close to blind to a deterministic weekly cycle. The
  prompt's `check ·` slot forces the null onto the page, which is the only
  reason the contradiction on the next cell is legible rather than confusing.
- **The usual student version:** `adfuller(series)` and read `p < 0.05` as
  "stationary, next". The library makes this easy: `adfuller` returns a
  seven-element tuple with the p-value at index 1 and no verdict at all, and its
  default `regression="c"` fits a constant but **no trend term**. This is not a
  hypothetical — the version of this notebook shipped with the course prints
  `"stationary" if p < 0.05 else "NOT stationary"` for exactly this series, two
  cells below a markdown paragraph asserting the series is not stationary. The
  notebook contradicts itself in its own stored output and nothing complains.
- **How you would catch it:** the check is not another test, it is the
  definition. Weak stationarity requires a constant mean. Go and compute the
  mean — that is cell 3 — and see whether the number the test just gave you
  survives contact with it.

## Cell 3 — the definition, computed

**Prompt to type:**

> **input** · the pool
> **output** · the mean and standard deviation of the January-to-May window of
> each year 2016–2019, and the mean by day of the week
> **constraint** · compare like months across years, because 2019 stops at the
> end of May and a full-year mean would be comparing five months against twelve
> **check** · the mean should move materially across years and enormously across
> weekdays; quote both as a fall in boardings and as a per cent

**Expect:**

```
Jan-May   2016  152 days   mean 639,234
Jan-May   2017  151 days   mean 617,790
Jan-May   2018  151 days   mean 602,579
Jan-May   2019  151 days   mean 583,030
   like-for-like fall 2016 to 2019:  56,204  (8.8%)

mean by weekday: Mon 681,783  Tue 736,531  Wed 739,950  Thu 743,081
                 Fri 731,001  Sat 434,787  Sun 326,415
```

**Assert:**

```python
janmay = lambda y: pool[f"{y}-01":f"{y}-05"].mean()
assert janmay(2016) > janmay(2017) > janmay(2018) > janmay(2019)
```

**⏱** Under a second.

**Annotate:** full

Markdown to put **after** this cell — this is where the ⚠ goes, and it is the
only place it goes:

> ⚠ **Look at the two cells together.** The test said *reject the unit root* for
> the level series. The definition says the mean falls by 56,204 boardings —
> 8.8% — over three like-for-like windows, and moves by a factor of 2.3 between
> a Wednesday (739,950) and a Sunday (326,415). A quantity whose mean depends on
> $t$ is not weakly stationary, whatever the test returned.
>
> Both are correct. They answer different questions. ADF asks *is this a random
> walk?*, and the answer is no — the series is mean-reverting around a slowly
> falling level with a hard weekly cycle. The section asked *may I fit one set
> of weights to all of it?*, and the answer is still no.
>
> **This is what a test used as an oracle looks like.** The output was a
> p-value, the p-value was small, and the small p-value was pointed at a
> question nobody had checked it could answer. Naming the null in the printed
> line is what makes that visible on the page rather than in a viva.

- **Left open:** the prompt says "compare like months" but does not say why the
  full-year comparison would be wrong. It would be wrong in a specific,
  quantified way: 2019 contributes 151 days of January-to-May, so a full-year
  mean for 2019 is a winter-and-spring mean set against three whole-year means.
  The drift is real either way — 652,038 to 583,030 on full years, 639,234 to
  583,030 like-for-like — but only one of those two comparisons could be
  defended if the fall had been smaller.
- **The usual student version:** `pool.groupby(pool.index.year).mean()` and quote
  the four numbers. It runs, it looks like a table, and the last row is five
  months of a different season. The identical error shipped in Lecture 19 §7,
  where "weekly lags" were bucketed as {7, 14, 21, 28} while the figure drew
  lines at all eight multiples of seven — lag 35 carries +0.23, the second
  largest coefficient in the model, and was counted as evidence against weekly
  structure. A statistic that depends on an arbitrary partition has to survive a
  different partition or be replaced.
- **How you would catch it:** re-run the summary under a second defensible
  partition — here, restrict every year to Jan–May. If the story changes, the
  story was the partition. It costs one line.

## Cell 4 — differencing is not free

Markdown above the cell:

> $\nabla X_t = X_t - X_{t-1}$ removes a linear trend; twice removes a quadratic
> one. A series stationary after $d$ differences is *integrated of order $d$*.
> And $\nabla_7 X_t = X_t - X_{t-7}$ removes a weekly cycle.
>
> For a stationary series,
> $\operatorname{Var}(X_t - X_{t-h}) = 2\gamma(0)\,(1 - \rho(h))$, so
> differencing at a lag where $\rho(h) < 1/2$ makes the variance **larger**.
> Predict the three numbers from the identity before you measure them.

**Prompt to type:**

> **input** · the pool and its autocorrelation at lags 1, 7 and 14
> **output** · for each lag: rho, the standard deviation predicted by
> `sqrt(2 * sd**2 * (1 - rho))`, the measured standard deviation of the
> difference, and the ratio predicted/measured
> **constraint** · compute the prediction from the identity before the
> measurement, and print both, so the theory is exposed to the data
> **check** · the ratio must be 1.00 to within one per cent at every lag; flag
> any lag where differencing makes the spread larger than the series itself

**Expect:**

```
series                    sd  183,841
lag  1: rho +0.419   predicted  198,145   measured  198,098   ratio 1.000   <- WORSE
lag  7: rho +0.837   predicted  104,821   measured  104,730   ratio 1.001
lag 14: rho +0.811   predicted  113,157   measured  113,068   ratio 1.001
```

**Assert:**

```python
for h in (1, 7, 14):
    rho = pool.autocorr(lag=h)
    ratio = np.sqrt(2 * pool.std()**2 * (1 - rho)) / pool.diff(h).std()
    assert 0.99 < ratio < 1.01, (h, ratio)
assert pool.diff(1).std() > pool.std()      # the point of the cell
```

**⏱** Under a second.

**Annotate:** short

Markdown after the cell:

> At lag 1 the autocorrelation is **0.419**, below the half the identity needs,
> so first-differencing raises the spread from 183,841 to 198,098. The reflex —
> *it is not stationary, difference it* — makes this problem harder. At lag 7,
> $\rho = 0.837$ and the difference is genuinely smaller, 104,730.
>
> That is the whole explanation of why "copy last week" was so hard to beat in
> Lecture 19: $\nabla_7$ is close to noise, and the naive forecast is exactly
> the model that assumes it is.

## Cell 5 — the same three series, drawn

**Prompt to type:**

> **input** · January to May 2019 of the rail series, raw and differenced at
> lags 1 and 7
> **output** · three stacked panels sharing the x axis, each with a zero line
> **constraint** · do **not** share the y axis — the panels have different
> scales and that is the point
> **check** · the zero line should be crossed constantly on the two differenced
> panels and nowhere near the raw one; if all three look alike the y-limits got
> shared

**Expect:** a 3×1 figure, 151 days wide. Top panel oscillating between roughly
150,000 and 800,000 with the weekly saw-tooth; middle panel swinging either side
of zero by several hundred thousand; bottom panel a tight band around zero with
occasional holiday spikes.

**Assert:** none — it is a figure. The claim it supports was asserted
numerically in cell 4.

**⏱** Under a second.

**Annotate:** short

## Cell 6 — where the structure is

**Prompt to type:**

> **input** · autocorrelation of the pool and of its lag-7 difference, for lags
> 0 to 42
> **output** · both as paired bars on one chart, one pair per lag
> **constraint** · one chart, not two — a collapse is a comparison and cannot be
> shown by a single series
> **check** · before running, name what you expect at lag 7 and lag 14; read
> those two pairs specifically afterwards

**Expect:** the raw series stays high at every lag — 0.837 at 7, 0.811 at 14,
0.799 at 21, and **0.818 at 35**, which is higher than lag 14. The seasonal
difference collapses: −0.412 at lag 7, −0.045 at lag 14.

**Assert:**

```python
assert pool.autocorr(7) > 0.8 and abs(pool.diff(7).autocorr(14)) < 0.1
```

**⏱** Under a second.

**Annotate:** short

Note for the markdown after this cell: say *lag 35 is 0.818, higher than lag
14*. The weekly structure does not decay over five weeks, and a summary that
buckets "weekly lags" as {7, 14, 21, 28} and stops is choosing its own answer.

---

# § 3 · The measurement, broken and then repaired

*Examinable. This is the section the application exists for.*

Markdown above the section, with no warning in it:

> The next cell is Lecture 19's, reproduced without changes. Run it, look at the
> five folds, and write the mean down.

## Cell 7 — a cross-validated linear model

**Prompt to type:**

> **input** · the pool cut into 56-day windows, one row per day, target the next
> day
> **output** · a five-fold cross-validated MAE in boardings, plus the per-fold
> values
> **constraint** · scale by 1e6 before fitting and convert back afterwards;
> fixed `random_state` so the number reproduces
> **check** · print `X.shape` and `y.shape`; the folds should agree with one
> another

**Expect:**

```
X (1191, 56)   y (1191,)
random 5-fold   MAE     44,761   folds [51096 43678 39538 43738 45754]
```

**Assert:**

```python
assert X.shape == (1191, 56)
assert round(folds_random.mean()) == 44761, folds_random.mean()
```

That second assertion is the point of reproducing the cell at all: it must match
Lecture 19's stored output to the boarding, or the two notebooks have drifted
and every comparison below is void.

**⏱** Under a second (0.01 s).

**Annotate:** short

## Cell 8 — the same model, split by time

**Prompt to type:**

> **input** · the same `X` and `y`
> **output** · the same five-fold MAE with `TimeSeriesSplit` instead of shuffled
> `KFold`
> **constraint** · change one call and nothing else
> **check** · print the fold values as well as the mean, and print the standard
> deviation across folds for both protocols

**Expect:**

```
forward 5-fold  MAE     51,880   folds [80506 38448 44838 37991 57615]
   fold sd:  random 3,758   forward 15,970
```

**Assert:**

```python
assert round(folds_time.mean()) == 51880, folds_time.mean()
assert folds_time.std() > 4 * folds_random.std()
```

**⏱** Under a second.

**Annotate:** short

Markdown after the cell — note what it does **not** claim:

> The forward mean is higher, and the folds disagree with each other four times
> as much: a spread of 15,970 against 3,758. Resist the obvious sentence. The
> two protocols are not scored on the same rows — the shuffled folds cover all
> 1,191 days, the forward folds cover days 201 to 1,190 — and the forward folds
> train on 201, 399, 597, 795 and 993 rows against a flat 953 for the shuffle.
> Two things changed. Cells 10 and 11 take them apart.

## Cell 9 — and a gap

**Prompt to type:**

> **input** · the same `X` and `y` again
> **output** · the five-fold MAE with `TimeSeriesSplit(n_splits=5, gap=WINDOW)`
> **constraint** · leave a gap the width of one window between the end of
> training and the start of testing
> **check** · print the training and test index ranges of each fold, so what
> `gap` actually removed is visible rather than assumed

**Expect:**

```
forward + gap   MAE     53,696   folds [88277 39123 45156 38268 57655]
   fold 1: train[0:144] n=145   test[201:398] n=198
   fold 2: train[0:342] n=343   test[399:596] n=198
   ...
```

**Assert:**

```python
assert round(folds_gap.mean()) == 53696, folds_gap.mean()
```

**⏱** Under a second.

**Annotate:** short

Markdown after the cell — **do not** say the 1,816-boarding difference measures
adjacency:

> `gap=WINDOW` removes the last 56 training rows of every fold. It therefore
> changes two things at once: it breaks adjacency **and** it shrinks every
> training set by 56 rows. The 1,816 boardings between 51,880 and 53,696 are
> both of those together, and cell 12 is where the case against adjacency gets
> made in a way that does not depend on the size of that number.

## Cell 10 — the same four protocols, scored against a baseline on their own rows

Markdown above the cell:

> Lecture 19 finished by printing an `Improvement vs NAIVE_MAE` column in which
> the models were scored on the last 239 days and the baseline on all 1,191. Its
> §9 caught it and corrected that one table by hand. This cell does the
> correction as arithmetic, for every protocol at once, so it cannot drift back.

**Prompt to type:**

> **input** · the per-row absolute error of the naive "copy last week" forecast,
> aligned to the rows of `X`, and the four protocols from cells 7–9 plus a
> single 80/20 forward hold-out
> **output** · for each protocol: its MAE, the naive baseline **restricted to
> exactly the rows that protocol scored on**, and the margin between them
> **constraint** · the baseline must be recomputed per fold and averaged the
> same way the model's score is; do not reuse one global `NAIVE_MAE`
> **check** · print the number of rows each protocol scored on next to its
> margin, and print the mismatched margin beside the matched one so the size of
> the error is on the page

**Expect:**

```
protocol                       MAE   naive, same rows   margin   rows
random 5-fold             44,761             55,386    19.2%    1191
one forward hold-out      52,451             67,225    22.0%     239
forward 5-fold            51,880             57,249     9.4%     990
forward 5-fold + purge    53,696             57,249     6.2%     990

  the same margins against one global naive of 55,399:
  19.2%   5.3%   6.4%   3.1%
```

**Assert:**

```python
assert round(naive_matched["one forward hold-out"]) == 67225
assert round(naive_matched["random 5-fold"]) == 55386
assert abs(margin["one forward hold-out"] - 0.220) < 0.001
```

**⏱** Under a second.

**Annotate:** full

- **Left open:** the prompt says "restricted to exactly the rows that protocol
  scored on" but not how to average across folds. `cross_val_score` returns one
  number per fold and averages them with **equal weight**, regardless of fold
  size, so the baseline has to be averaged the same way or the two columns are
  built differently. Here it barely matters — the pooled shuffled MAE is 44,766
  against a fold-mean of 44,761, five boardings apart, because the folds are
  238–239 rows each. Name the convention anyway; the day the folds are uneven it
  will matter and nothing will say so.
- **The usual student version:** compute `NAIVE_MAE` once at the top and
  subtract it from everything. This is not invented: it is what Lecture 19's
  final cell does, it is what the shipped version of *this* notebook does in its
  own margin table, and it is what produces the second row of the block above —
  **5.3% instead of 22.0%**, an error four times the size of the quantity being
  reported. It runs, it prints a tidy table, and the column header is a lie
  about which days were compared.
- **How you would catch it:** print the row count beside every margin. Four
  different row counts in one table is the whole diagnosis, and it fits in one
  extra column.

Markdown after the cell:

> Read the second row. Scored against the baseline **on its own 239 days**, the
> single forward hold-out gives the linear model a **22.0%** margin — the figure
> Lecture 19's §9 arrived at by hand, now derived. Against a global baseline it
> reads 5.3%. Same model, same predictions, same test days; only the baseline's
> window moved.
>
> And notice which way the error went. The mismatched table made the honest
> protocols look **worse** than they are, because the held-out window is harder
> for everybody: 67,225 against 55,399, 21% harder for the naive forecast alone.
> A mismatched comparison does not have a direction you can correct for. It
> measures the rows.
>
> The margins that survive matching: 19.2% shuffled, 9.4% forward, 6.2% forward
> and purged. Roughly **68%** of the shuffled protocol's apparent margin does not
> survive the strictest protocol. That is still not a clean attribution — cell 11
> is.

## Cell 11 — the clean head-to-head

Markdown above the cell:

> Cell 10 matched the rows. It did not match the training sets: the shuffled
> folds each trained on 953 rows, the forward folds on 201 to 993. One
> comparison holds both fixed.

**Prompt to type:**

> **input** · out-of-fold predictions from the shuffled `KFold` via
> `cross_val_predict`, and the 80/20 forward hold-out model
> **output** · both scored on the same last 239 days
> **constraint** · both models must have been fitted on 952 training rows, so
> the only difference left is whether the training rows were allowed to sit
> either side of a test row
> **check** · print the difference in boardings and say which way it goes before
> you interpret it

**Expect:**

```
shuffled out-of-fold, last 239 days :  52,705
forward hold-out,     last 239 days :  52,451
difference                          :    +254   (the leaky one is 0.5% WORSE)
```

**Assert:**

```python
assert round(oof_last239) == 52705
assert round(holdout_mae) == 52451
assert oof_last239 > holdout_mae      # yes, that way round
```

**⏱** Under a second.

**Annotate:** full

- **Left open:** the prompt does not say that `cross_val_predict` is only a
  legitimate thing to score this way because MAE decomposes over samples —
  scikit-learn's own documentation warns that out-of-fold predictions are
  inappropriate for metrics that do not. Mean absolute error is a mean of
  per-row terms, so restricting it to a subset of rows is meaningful. R² is not,
  and the same cell written with `scoring="r2"` would be quietly meaningless.
- **The usual student version:** stopping at cell 8 and writing *"the shuffle
  flattered the model by 7,119 boardings, 13.7%. Nothing else changed."*
  Something else changed — two things did. Lecture 19 shipped that sentence
  almost verbatim ("flattered the model by 7,690 boardings — 17.2%. Nothing else
  changed. Same rows, same columns, same estimator, same seed") and the rows
  were not the same. The version of this notebook in the repository prints the
  same claim with the 7,119 figure.
- **How you would catch it:** the arithmetic here is four lines and it reverses
  the sign of the headline. Any time a sentence contains "nothing else changed",
  write down the list of things that would have to be equal for that to be true,
  and check them one at a time. The list for a cross-validated comparison is:
  the rows, the training-set sizes, the estimator, the seed.

Markdown after the cell:

> **The leakage bought nothing.** On matched rows with matched training-set
> sizes, the shuffled protocol scores 254 boardings *worse*. Everything in cell
> 8's 7,119-boarding gap was window difficulty and training-set size.
>
> The lesson — a shuffled split is invalid on a windowed series — is still true.
> It is now unsupported by the effect size, so it has to be argued another way,
> and the other way is stronger because it does not depend on this dataset at
> all. That is cell 12.

## Cell 12 — the argument that does not depend on the number

**Prompt to type:**

> **input** · consecutive rows of `X`
> **output** · how many of the 56 input coordinates row *t* and row *t+1* share
> **constraint** · assert it rather than print it, for every consecutive pair,
> not a sampled one
> **check** · the answer is knowable on paper before running: windows step by
> one day, so it must be 55

**Expect:** `every consecutive pair of rows shares 55 of 56 coordinates (98.2%)`.

**Assert:**

```python
assert (X[:-1, 1:] == X[1:, :-1]).all()
```

**⏱** Under a second.

**Annotate:** short

Markdown after the cell:

> Under a shuffle, the training set contains rows that are **98% identical** to
> test rows. That is true of any sliding-window dataset with any window length,
> on any series, whatever the MAE happens to do. It was true of cell 7 even
> though cell 11 showed the leakage cost nothing measurable here.
>
> A second unconfounded observation, already on the page: the shuffle's fold
> spread was **3,758** against **15,970** forward. The shuffle manufactured
> *stability*, not a better mean. **A stable measurement of the wrong quantity
> is stable.**

Markdown closing the section — **which number to quote**:

> Four protocols, four defensible numbers. This lecture quotes the **single
> forward hold-out, 52,451, against a baseline of 67,225 on the same 239 days —
> a 22.0% margin**, because that protocol matches how the model would be used:
> fit once on everything up to a date, forecast forward. The purged five-fold is
> stricter and reads 6.2%, mostly because its first fold trains on 145 days and
> is scored anyway.
>
> The discipline is not picking the smallest number. It is naming the protocol
> and the rows beside the number you quote. A margin with neither attached is
> not a result.

---

# § 4 · Spending what is left

*Examinable, except cell 14 which is engineering.*

## Cell 13 — more series, and a calendar

Markdown above the cell:

> The honest margin is 22.0%, so an improvement has to be real to show. The
> first one is not architectural: give the model something it does not have.
> Bus ridership, and **tomorrow's day type**, which is printed on a wall planner
> and is therefore knowable at the moment of the forecast.
>
> The CTA's `day_type` column has three values, and the notebook has to say what
> they are rather than one-hot encoding three letters: **W** is a weekday, **A**
> is a Saturday, **U** is a Sunday *or* a public holiday. Verify it rather than
> take it: all 1,087 `A` days are Saturdays, and of 1,216 `U` days, 1,091 are
> Sundays and the remaining 125 are US federal holidays — 1 January, the last
> Monday in May (Memorial Day), 4 July (Independence Day), the first Monday in
> September (Labor Day), the fourth Thursday in November (Thanksgiving) and 25
> December. Those dates are not common knowledge outside the United States and
> the column will not tell you.

**Prompt to type:**

> **input** · `rail`, `bus`, and `day_type` shifted by −1 so it describes
> tomorrow
> **output** · a five-column frame over 2016-01 to 2019-05, day type one-hot
> encoded as floats
> **constraint** · `shift(-1)` on the calendar is legitimate and `shift(-1)` on
> the target is a leak — the test is whether the value is knowable at prediction
> time, not the sign of the shift
> **check** · cross-tabulate `day_type` against the weekday name and assert the
> five column names, so a silent change in the encoding stops the notebook

**Expect:**

```
day_type  Mon  Tue  Wed  Thu  Fri   Sat   Sun
A           0    0    0    0    0  1087     0
U          59   10   10   30   12     4  1091
W        1033 1082 1081 1061 1079     0     0

(1247, 5) ['rail', 'bus', 'next_day_type_A', 'next_day_type_U', 'next_day_type_W']
```

**Assert:**

```python
assert list(mulvar.columns) == ["rail", "bus", "next_day_type_A",
                                "next_day_type_U", "next_day_type_W"]
assert mulvar.shape == (1247, 5)
assert (df.loc[df.day_type == "A"].index.day_name() == "Saturday").all()
```

**⏱** Under a second.

**Annotate:** short

Markdown after the cell:

> **`shift(-1)` again, and this time it is not a leak.** In Lecture 19 a
> `shift(-1)` on the target was one. Here it is on the calendar. The difference
> is not the sign — it is whether the value is available when the forecast has to
> be made. Tomorrow's day type is; tomorrow's ridership is not.
>
> Say the rule out loud for every column, every time: *would I know this number
> at 6pm the day before?* If yes it is a feature. If no it is the answer.

## Cell 14 — one windowing function, tested on six integers

*Not examinable — engineering, but the cell everything below depends on.*

**Prompt to type:**

> **input** · a frame of several series, a window length and a horizon
> **output** · a function returning `(windows, targets)` where the target is
> `rail` only, taken from the days immediately **after** each window
> **constraint** · one function used by every model below, so the comparisons
> are like for like
> **check** · run it on the integers 0–5 with `window=3, horizon=1` and read the
> three pairs by eye before using it on anything real

**Expect:**

```
[0.0, 1.0, 2.0] -> [3.0]
[1.0, 2.0, 3.0] -> [4.0]
[2.0, 3.0, 4.0] -> [5.0]

X (1191, 56, 5)   y (1191, 1)   train 952, test 239
```

**Assert:**

```python
Xt, yt = make_windows(pd.DataFrame({"v": range(6)}), window=3, horizon=1)
assert Xt.squeeze(-1).tolist() == [[0, 1, 2], [1, 2, 3], [2, 3, 4]]
assert yt.tolist() == [[3], [4], [5]]
assert Xm.shape == (1191, 56, 5) and cut == 952
```

**⏱** Under a second.

**Annotate:** short

Note: `cut` here equals 952, the same value cell 10 used for the numpy `X`. That
is a coincidence of the two arrays having 1,191 rows each. Give this one a
distinct name — `cut_m` — rather than rebinding `cut`, or a later reader cannot
tell which array a `cut` in scope belongs to.

## Cell 15 — a GRU on five series

Markdown above the cell:

> A simple RNN multiplies by the same recurrent matrix at every step, so a
> gradient over 56 steps either vanishes or explodes — Lecture 13's thread
> (*Twenty layers, no learning*) in a new place. A **GRU** adds an update gate,
> deciding how much of the old state to keep, and a reset gate, deciding how
> much of it to use. Keeping becomes addition rather than repeated
> multiplication, so a gradient can travel.

**Prompt to type:**

> **input** · the five-series windows, split 80/20 by position
> **output** · a trained GRU with 32 hidden units and its held-out MAE, printed
> every 30 epochs as well as at the end
> **constraint** · construct the model and the optimiser inside this cell, so
> re-running it retrains from scratch rather than continuing
> **check** · the held-out MAE must fall and then flatten; if it rises, report
> the final epoch, not the best one seen

**Expect:** a per-epoch trace and a final number. The trace starts high — the
untrained head predicts near zero, so epoch 1 is in the hundreds of thousands —
and should be in the 45,000–60,000 region by epoch 120. Anything under 30,000 or
over 100,000 at the end means something is wrong. Reference points on the **same
239 test days**: naive 67,225, linear hold-out 52,451, Lecture 19's RNN 48,965.

**Assert:**

```python
assert 20_000 < gru_mae < 120_000, gru_mae
```

Deliberately loose. A tight assertion here would be an assertion about a
particular seed on a particular BLAS, and this cell has no reproducible target —
which is the subject of cell 16.

**⏱** About **15 s** for 120 epochs (30 minibatches each) measured on an Apple
M4 Max; budget **45–90 s** on a Colab CPU runtime, whose 2 vCPUs are three to
five times slower on this workload. The Colab figure is extrapolated from the
local measurement, not measured on Colab.

**Annotate:** short

Note for §4.2 compliance: the model **must** be constructed inside the cell.
Lecture 19's training cell was not, so its own "change the seed and re-run"
exercise trained the existing network for a further 200 epochs. Anyone
re-running this cell to try a different learning rate needs it to start from
initialisation.

## Cell 16 — gates, or more series? and is either of them a result?

Markdown above the cell:

> Two things changed between the linear model and cell 15: the architecture and
> the inputs. Separating them is one extra run. Deciding whether the difference
> means anything is three more.

**Prompt to type:**

> **input** · the five-series windows and the rail-only windows
> **output** · both configurations trained at seeds 0, 1 and 2, reporting the
> mean and the min-to-max spread of the three held-out MAEs for each
> **constraint** · change exactly one thing between the two configurations —
> `input_size` — and re-seed and reconstruct the model before every run
> **check** · print the gap between the two means beside the larger of the two
> spreads, and state which is bigger

**Expect:** six numbers and a summary block of the form

```
GRU, rail only     mean  <a>   spread over 3 seeds  <sa>
GRU, five series   mean  <b>   spread over 3 seeds  <sb>
   gap between means      <|a-b|>
   larger spread          <max(sa,sb)>
   VERDICT: the gap is / is not larger than the run-to-run spread
```

The verdict is not pre-computed here and you must record your own. What the
verdict has to be measured against is: Lecture 19's comparable RNN moved by
**7,990** boardings across four readings of the same run. Treat a gap under
about 8,000 as unproven until these three seeds say otherwise.

**Assert:**

```python
assert len(rail_scores) == 3 and len(five_scores) == 3
assert all(20_000 < s < 120_000 for s in rail_scores + five_scores)
```

**⏱** Six training runs. About **90 s** on an Apple M4 Max; budget **5–9
minutes** on a Colab CPU runtime. This is the longest cell in the notebook. If
you are short of time, reduce to two seeds and say in your write-up that you did
— a spread from two readings is weak, and saying so is the point.

**Annotate:** full

- **Left open:** the prompt says "re-seed before every run" without saying what
  a seed controls here. `torch.manual_seed` set once at the top does **not**
  make a later run reproducible if a `DataLoader` with `shuffle=True` has
  already consumed draws from the global generator. The seed must be set
  immediately before the model is constructed *and* before the loader is built,
  every time. Otherwise "three seeds" is three points from one drifting stream
  and the spread you measure is not the spread you think.
- **The usual student version:** train each configuration once, put the two
  numbers in a table, and write "gates helped" or "the extra series helped". The
  precedent is in the previous lecture: its RNN cleared the committed target by
  **894** boardings and its own printed epoch trace moved by **7,990** across the
  last four readings — 8.9 times the margin. Both numbers are in the notebook.
  The headline stands anyway. One run per configuration cannot distinguish a
  real effect from that.
- **How you would catch it:** ask of any headline *what would this number be if
  the model had learned nothing?* and *is this difference larger than the
  noise?* The first is answered by the matched baseline, 67,225. The second
  needs more than one run, and three cheap ones cost 90 seconds.

## Cell 17 — a fortnight, not a day

Markdown above the cell:

> A staffing decision needs more than tomorrow. Two changes: the target becomes
> 14 values, and the head produces 14 numbers. One head, not fourteen models.
>
> The baseline needs a decision too, and this is where the comparison quietly
> changes shape if you let it. "Copy last week" predicts day $t+k$ from day
> $t+k-7$. For $k \le 7$ that source day is inside the input window and known.
> For $k > 7$ it is **not yet observed**, so the honest naive forecast at
> horizons 8–14 has to reach back fourteen days. Draw two baseline segments, not
> one flat line.

**Prompt to type:**

> **input** · the five-series windows with `horizon=14`, split 80/20 by position
> **output** · the model's MAE at each of the 14 horizons, and the naive
> baseline at each horizon computed **on the same 236 test windows**, both on one
> chart
> **constraint** · one head producing fourteen numbers; the naive baseline uses
> lag 7 for horizons 1–7 and lag 14 for horizons 8–14, because a lag-7 source
> day is unobserved beyond a week
> **check** · read the shape, not the average — a single mean would hide that
> day 1 and day 14 are different problems; error must grow with horizon

**Expect:** the naive curve is fully determined by the data and can be checked
before the model finishes training:

```
horizon   1..7  (lag 7)  naive  65,493  65,433  65,286  65,146  65,239  65,309  65,218
horizon  8..14  (lag 14) naive  76,136  76,172  77,888  77,869  77,793  77,873  77,315
```

The model curve should rise from horizon 1 and sit below both baseline segments
at short horizons. **Where it crosses is your finding — do not assume day 7.**

**Assert:**

```python
assert Xh.shape == (1178, 56, 5) and cut_h == 942 and len(Xh) - cut_h == 236
assert round(naive_h[0]) == 65493 and round(naive_h[13]) == 77315
assert per_step.shape == (14,)
```

**⏱** About **15 s** on an Apple M4 Max; budget **45–90 s** on a Colab CPU
runtime.

**Annotate:** full

- **Left open:** the prompt fixes the baseline's lag rule but not what to do at
  the boundary. Horizon 7 uses lag 7 and horizon 8 uses lag 14, so the baseline
  jumps by about 11,000 boardings between two adjacent points on the x axis.
  That step is real and has to be drawn as a step, with both segments labelled —
  a smooth interpolation between them would be a forecast nobody could make.
- **The usual student version:** `ax.axhline(NAIVE_MAE)` at 55,399 and a flat red
  line across all fourteen horizons. This is exactly what the version of this
  notebook in the repository draws, and it is wrong twice over: 55,399 is the
  whole-window baseline where these 236 test windows give 65,493, and it is a
  lag-7 baseline extended to horizons where lag 7 is unobservable. Its chart
  title asserts "the margin is spent by about day 7" — a claim read off a line
  that is roughly 10,000 boardings too low for the first seven horizons and
  12,000 too low for the last seven. The second common variant is
  `(pred - y).abs().mean()` with no `dim=0`, which collapses the horizon axis
  into one number and deletes the finding.
- **How you would catch it:** the naive curve costs nothing to compute and does
  not need the model, so compute it first and sanity-check it against the
  matched hold-out baseline you already have: horizon 1 with lag 7 over 236
  windows should land near the 67,225 you computed over 239 windows in cell 10.
  It lands at 65,493 — close, and the small difference is three fewer windows
  and a slightly different span. If it landed at 55,399 you have used the whole
  window again.

Markdown after the cell:

> **Report the curve, not its average.** One number for "the fourteen-day
> forecast" hides that day 1 and day 14 are different problems. If a decision
> only needs three days, say so and be judged on three.

---

# § 5 · Regime change

*Beyond the book, for context. Not examinable.*

Markdown above the cell:

> Everything above stops at May 2019. The series does not.
>
> This section is here because it is **neither a leak nor a bug**. The protocol
> was correct, the measurement was honest, and the model is useless afterwards
> anyway. No split protects you from the world changing.

## Cell 18 — before, during, and after

**Prompt to type:**

> **input** · the whole rail series through November 2021
> **output** · the mean daily level in Jan–May 2019, Apr–Aug 2020 and Sep–Nov
> 2021, each as a percentage of the 2019 level, and a plot spanning all of it
> **constraint** · print the ratios rather than describing them, and shade March
> to June 2020 on the plot
> **check** · compare the size of the change against the model's entire error
> budget — the whole margin argued about above is a few thousand boardings

**Expect:**

```
Jan-May 2019    583,030    100.0%
Apr-Aug 2020    104,395     17.9%
Sep-Nov 2021    286,134     49.1%
```

**Assert:**

```python
assert 0.17 < level_2020 / level_2019 < 0.19
assert df.index.max().year == 2021
```

**⏱** Under a second.

**Annotate:** short

Markdown after the cell — and note the arithmetic, because "roughly three
quarters" is the phrase that suggests itself and it is wrong:

> The level fell to **17.9%** of its early-2019 value: a fall of **82%**, not of
> three quarters. By late 2021 it had recovered to **49.1%** — half, and still
> nothing like where it started.
>
> Set that against the argument of §3. The entire dispute there was over a few
> thousand boardings of margin on a baseline of 67,225. This is a drop of
> **478,635**. Every protocol in this notebook was correct and every one of them
> was answering a question about a world that stopped existing in March 2020.
>
> What to do about it is a **monitoring** question, not a modelling one: measure
> the live error against the committed number and write down, before deployment,
> the rule that says when to stop trusting the model. A model that is never
> re-measured after deployment is an assumption wearing a number's clothes.

---

# § 6 · The temporal checklist

*Examinable.*

Markdown, no cell:

> 1. **Is there a time column?** Then no shuffled split — including inside
>    `cross_val_score`, `train_test_split`, and any tuner's own CV.
> 2. **Is there a gap between train and test?** Adjacent rows share 55 of 56
>    coordinates here. Whether removing them changes the score is a separate
>    question from whether they should be there.
> 3. **Would I know every feature at prediction time?** Say it out loud, column
>    by column. `shift(-1)` on a calendar is fine; on the target it is the
>    answer.
> 4. **Is the baseline seasonal, and is it scored on the same rows as the
>    model?** Copying the same weekday, on the model's own test days. Both
>    halves of that sentence are load-bearing.
> 5. **Did I difference reflexively?** Check $\rho(h) > 1/2$ first. At lag 1 here
>    it is 0.419 and differencing raises the spread from 183,841 to 198,098.
> 6. **Is the score a single number when the decision needs a curve?**
> 7. **Is the difference bigger than the run-to-run spread?** Three seeds cost 90
>    seconds.
> 8. **What would tell me the regime has changed?** Write the trigger down before
>    deployment, not after.

## ★ Record your numbers

Three, with the protocol and the row count beside each:

- the shuffled margin, **19.2% on 1,191 days**;
- the hold-out margin against its own baseline, **22.0% on 239 days**;
- the same hold-out margin against the global baseline, **5.3%**.

The third is not a result. It is the size of the mistake that nothing in the
output complained about, and it is four times the quantity being reported.

## Red-team your own notebook

Each of these names the cells to re-run and the order. Re-running a training
cell retrains from scratch, because every training cell constructs its model
inside itself.

1. **Does the gap cost adjacency or training rows?** Run cell 9 again with
   `gap=0` (that is cell 8's number, 51,880) and then a third variant that keeps
   `gap=0` but drops the **first** 56 rows of each training fold — same training
   size, adjacency preserved. That third variant scores **58,334**, which is
   *worse* than the gapped 53,696 by more than the 1,816 the gap cost. Neither
   control isolates adjacency, because dropping rows from either end also
   changes which era you train on. Write two sentences on why the structural
   argument in cell 12 is the one that survives this. *Cells to re-run: 9 only.*
2. **Is tomorrow's calendar worth anything?** Rebuild `mulvar` using
   `df["day_type"]` with no `shift(-1)` — today's day type, still knowable at
   prediction time, so still not a leak — and rerun the comparison. *Cells to
   re-run, in order: 13, then 14, then 16.* About 90 s on a laptop, 5–9 minutes
   on Colab. Predict first whether the score moves, then check whether whatever
   move you see exceeds the three-seed spread cell 16 measured. If it does not,
   you have not learned that the shift is worthless — you have learned that this
   experiment cannot tell.
3. **Train on 2016–2019, test on 2020.** The pool is sliced to 2019-05 in cell
   1 and again in cell 13, so both slices have to change. *Cells to re-run, in
   order: 1 (change the pool end date to 2020-12-31), 7, 13, 14, 15.* Then argue
   in two sentences whether the model was wrong or the question was.
4. **Change `WINDOW` from 56 to 7, and then to 112.** *Cells to re-run, in
   order: 1, 7, 8, 9, 10, 12, 14, 15.* The number of scorable rows changes with
   `WINDOW`, so cell 10's matched baselines change too — check that they did. If
   your margins moved but your baseline column did not, the baseline is being
   computed somewhere it should not be.
5. **Re-run from a fresh runtime, top to bottom.** Not a formality: cell 14 binds
   `cut_m` and cell 10 binds `cut`, and if you follow the shipped notebook's
   habit of calling both of them `cut`, the two happen to be equal at 952 and
   the bug is invisible until you change `WINDOW`.

---

# Defects found in the current notebook

Checked against `notebooks/lecture-20.ipynb` (48 cells, 15 of them code) and, where
the defect is a join with the previous lecture, against `notebooks/lecture-19.ipynb`.
**Every numeric claim below was recomputed from
`notebooks/datasets/ridership/CTA_-_Ridership_-_Daily_Boarding_Totals.csv` with
`python3`.** Where I could not check something, I say so.

## Verified by recomputation

**1 · §2.1 — the margin table compares four protocols on four different windows
against one baseline (cell 27).** `NAIVE_MAE` is computed over all 1,191 rows
(55,399) and subtracted from four scores measured on different rows: the
shuffled CV over 1,191, the hold-out over the last 239, and both forward CVs
over rows 201–1,190. Scored against the baseline on their own rows the margins
are 19.2% / 22.0% / 9.4% / 6.2%; the notebook prints 19.2% / 5.3% / 6.4% / 3.1%.
The hold-out row is wrong by a factor of four, and it is wrong in the direction
that makes the honest protocol look worst. **This is the same error Lecture 19
§9 confesses to and the one its `left_open` bullet in cell 26 explicitly warns
against — three cells above the code that commits it.**

**2 · §2.1 — the same error in the model-comparison table (cell 38).** Four rows,
four scoring windows: `NAIVE_MAE` (rows 0–1,190), `folds_gap.mean()` (rows
201–1,190, fold-averaged), `gru_rail` and `gru_mae` (rows 952–1,190). The `vs
naive` column subtracts a 1,191-day baseline from 239-day scores. The matched
baseline for the two GRU rows is 67,225, not 55,399.

**3 · §2.1 — the same error again in the horizon chart (cell 41).**
`ax.axhline(NAIVE_MAE)` draws a flat line at 55,399 against a model scored on
236 test windows. On those windows the lag-7 naive forecast is 65,493 at horizon
1 (10,094 higher), and at horizons 8–14 a lag-7 forecast is not available at all
— the honest lag-14 baseline is 76,136 to 77,873. The chart title, *"the margin
is spent by about day 7"*, is read off a line that is 10,000–12,000 boardings too
low across the whole axis. The stated conclusion may or may not survive a correct
baseline; it is not supported by the one drawn.

**4 · §1.1 / §3.2 — the stationarity section contradicts its own output (cells 4,
5, 6).** The markdown states *"Ridership is not stationary"*; the prompt box
states *"the level series should fail to look stationary where the differenced
ones do not"*; the `catch` bullet states *"If all three pass, you have read the
sign backwards."* Running the cell, all three pass: ADF p = 0.000122 on the level
series (statistic −4.613, `regression="c"`, 22 lags, 1,224 observations), 0.000000
on both differences. The reader who follows the `catch` instruction is told they
made an error they did not make. The mathematics is not in dispute — the level
series really is non-stationary, its Jan–May mean falling 639,234 → 583,030 over
2016–2019 and its weekday means ranging 326,415 (Sunday) to 743,081 (Thursday) —
but ADF tests for a unit root, which is a narrower question, and the notebook
never says so.

**5 · §2.1 — "the shuffle flattered the model by 7,119 boardings (13.7%)" (cell
21).** The two numbers are scored on different rows (1,191 vs 990) by models
trained on different amounts of data (953 rows vs 201/399/597/795/993). Matched
on both — shuffled out-of-fold predictions against the 80/20 hold-out model, same
952 training rows, same last 239 days — the figures are 52,705 and 52,451: the
leaky protocol is **254 boardings worse**. The claim as printed is not supported.

**6 · §2.2 — "Set `gap=0`. How much of the margin comes back? That amount was
adjacency" (cell 47, red-team).** It was not only adjacency. `TimeSeriesSplit`'s
`gap` removes the last 56 rows of every training fold, so it changes adjacency
and training-set size together. A same-size control — `gap=0` with the first 56
rows of each training fold dropped — scores 58,334 against the gapped 53,696, a
4,638-boarding move in the opposite direction, larger than the 1,816 the exercise
attributes to adjacency. The exercise asks the student to draw a conclusion the
data does not support.

**7 · §1.1 — "in March 2020 the level falls by roughly three quarters" (cell
43).** The cell's own output prints 17.9%, i.e. a fall of **82.1%**. Mean daily
rail boardings: 583,030 in Jan–May 2019, 104,395 in Apr–Aug 2020.

**8 · §1.1 — "on 1,191 windows it overfits" (cell 28).** The training set is
**952** windows; 1,191 is the total including the 239 held out. Separately, no
stacked recurrent model is built anywhere in the notebook, so the overfitting
claim is asserted and never demonstrated.

**9 · §6.1 — 15 code cells, 15 full three-bullet annotations.** The budget is
five to eight per notebook and never more than ten. Measured directly from the
`.ipynb`: 15 markdown cells begin `> **Prompt`, and all 15 contain
`**Watch this prompt.**`.

**10 · §7.1 — no timing markers anywhere.** The string `⏱` occurs 0 times, and so
do "minute" and any wall-clock figure. Three cells train GRUs for 120 epochs
(cells 36, 38, 41). Measured here: 0.12 s per epoch, 30 minibatches of 32 over
952 training windows, so ≈15 s per cell on an Apple M4 Max — modest, and
precisely the fact a reader alone at home cannot know without being told.
*(Timings were taken on a machine also running other jobs; the figures quoted are
best-of-four to remove contention. The Colab CPU figures in the script above are
extrapolations from these, not measurements on Colab.)*

**11 · §8.3 — the string "examinable" occurs 0 times.** Every section is supposed
to carry one of *examinable* / *not examinable — engineering* / *beyond the book,
for context*.

**12 · §7.5 / §7.4 — `day_type` is one-hot encoded and never decoded.** The cell
asserts the column names `next_day_type_A`, `next_day_type_U`, `next_day_type_W`
and no cell or paragraph says what A, U and W mean. Verified from the data: all
1,087 `A` days are Saturdays; of 1,216 `U` days, 1,091 are Sundays and 125 are US
federal holidays (New Year's Day, Memorial Day, Independence Day, Labor Day,
Thanksgiving, Christmas). Those holidays are not common knowledge in Rome and
the notebook never names one.

**13 · §2.4 — no run-to-run spread is ever measured.** `gru_rail` and `gru_mae`
are single runs at one seed and are tabulated as a comparison. The comparable
result in the previous lecture moved by 7,990 boardings across four readings of a
single run, against a margin of 894. Nothing in lecture 20 lets a reader tell
whether the GRU difference is larger than that.

**14 · §3.3 — one cross-reference does not resolve.** Cell 35's bullet ends
*"which is exactly what the next cell is for"*, referring to separating "gates
helped" from "more series helped". The next cell (36) trains the five-series GRU;
the separation happens in cell 38, two cells later.

**15 · §4.1 — `cut` is rebound.** Cell 27 sets `cut = int(len(X) * 0.8)` for the
numpy array `X`; cell 33 sets `cut = int(len(Xm) * 0.8)` for the torch tensor
`Xm`. Both evaluate to 952 only because both arrays happen to have 1,191 rows, so
the collision is silent today and becomes a wrong split the moment `WINDOW`
changes — which is exactly what red-team exercise 4 above asks a reader to do.

**16 · §7.2 — none of the three red-team exercises names the cells to re-run.**
"Give the model `df['day_type']` without the `shift(-1)`" requires cells 30, 33
and 36 in that order, spanning six cells and a training run; "train on 2016–2019,
test on 2020" requires editing two separate date slices (cells 3 and 30) that are
27 cells apart. Neither is stated. The same bullet also asserts an outcome — *"the
score barely moves"* — that no cell in the notebook produces.

## Checked and clean

Reported so the list is not read as exhaustive-by-omission:

- **§5.1 / §5.2** — no markdown line is indented ≥4 spaces outside a fence, and
  no fence marker is indented at all. Checked programmatically over all 33
  markdown cells. Lecture 19's cell 41 defect does not recur here.
- **§3.1** — there are no ```` ```python ```` blocks in any markdown cell, so
  nothing is quoted that could fail to exist.
- **§4.2** — all three training cells construct the model inside themselves
  (`GruModel(input_size=5)`, `GruModel(input_size=1)`, `horizon_model = ...`), so
  re-running one retrains from scratch. Lecture 19's non-idempotent cell 40
  defect does not recur.
- **§4.4** — the header correctly says the boxes are "specifications, not
  transcripts" and names Lecture 19 as the single exception. No blanket
  provenance claim.
- **Cross-references** — "Lecture 19 ... said so in its section 9" resolves
  (lecture 19 §9 is *"Where we finish — and the number in this table that is
  wrong"*). "Lecture 13's thread" resolves (*Twenty layers, no learning*).
  "Lecture 19's RNN rose at epoch 160 and the honest number was the one at 200"
  is correct against lecture 19's stored output (44,316.50 at epoch 140, 52,307.17
  at 160, 48,964.94 at 200). "Lecture 19 did with six integers" resolves to
  lecture 19 cell 27.
- **§1 data preparation** — the claim that this notebook's preparation is
  identical to Lecture 19's is true, despite using a different call:
  `drop_duplicates()` over all columns and `drop_duplicates(subset=["date"])`
  produce the same 7,639-row index here, because all 62 duplicated dates carry
  identical values. Pool length 1,247, 2016-01-01 to 2019-05-31, matching the
  figure the prompt box tells the reader to check against.
- **The reproduction cell works.** Cell 18 reproduces Lecture 19's shuffled
  five-fold to the boarding: 44,760.77, folds 51,096 / 43,678 / 39,538 / 43,738 /
  45,754. Its `catch` bullet demands exactly this and it holds.

## Could not check

- **The three GRU results.** I did not execute the training cells, per the brief.
  Everything I say about cells 36, 38 and 41 concerns the baselines they are
  compared against and the protocol around them, all of which are computable
  without training. The MAEs themselves are unverified, and the notebook stores
  no outputs for them.
- **§1.2 more generally.** The notebook has **no stored outputs at all** — all 15
  code cells have an empty `outputs` list. So no prose figure in it can be
  reconciled against a stored output; I reconciled them against the data instead.
  This also means the machine check in §9 of the guidelines ("every ≥4-digit
  prose figure appears in a stored output") cannot pass on this file as shipped.
- **Whether the horizon model's crossing point is day 7.** Determining that needs
  the training run. What is checkable, and wrong, is the baseline it is measured
  against.

## One correction to the guidelines

`GUIDELINES.md` §6.4 states that lecture 20 uses "annotations with no boxes".
That is not what the file contains. All 15 code cells are preceded by a prompt
box in the structured `input · output · constraint · check` form the same section
names as the standard — lecture 20 already uses it. The defect is §6.1, not
§6.4: every one of those 15 boxes also carries the full three-bullet annotation,
against a budget of five to eight. Verified by counting markdown cells beginning
`> **Prompt` (15) and containing `**Watch this prompt.**` (15).
