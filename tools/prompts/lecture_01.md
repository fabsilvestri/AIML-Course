# Lecture 1 — *Welcome, and a price you can't trust*

**Build it in Colab by prompting.** This file is the script you follow at the
keyboard. Fourteen code cells, in order. For each one: the prompt you type, what
must come back, and the assertion that has to pass before you move on.

Everything below was re-derived on 2 August 2026 against
`https://github.com/ageron/data/raw/main/housing.tgz` — scikit-learn 1.7.2,
pandas 2.3.3, numpy 2.3.5, Python 3.13, Apple arm64. Every figure in this file
came out of a run, not out of the existing notebook.

**Runtime.** Nothing here needs a ⏱ marker. The slowest cell is cell 14, the
three models: **6.0 s** with `n_jobs=1` on one arm64 core, **0.9 s** with
`n_jobs=-1` on sixteen. The whole notebook runs end to end in about **3 s** plus
one ~5 s download the first time. There is no GPU anywhere in this lecture and
Colab's free CPU runtime is the right choice.

**Vocabulary, defined once.** *Stratify* — force the training and test halves to
carry the same mix of some variable. *Leak* — let the test rows influence
anything the model sees, including the mean a scaler subtracts. *Baseline* — a
prediction so stupid it needs no fitting, whose only job is to make your real
score mean something.

**Where the defect is.** One cell in this notebook is wrong on purpose (cell 11)
and one is wrong on purpose without being marked (cell 12). Do not put a ⚠ on
either. The reveal for cell 11 goes in the markdown **after cell 12**, and it is
the only place in the notebook where the leak may be named. If you write "read
before running" above cell 11, the lesson is gone — the reader will find
`fit_transform(X_all)` in four seconds and learn nothing about how it feels to
believe a number.

**Examinability, per section.** §1–2 setup and loading: *not examinable —
engineering*. §3 split before you look: **examinable**. §4 what is in the data:
**examinable**. §5 baseline: **examinable**. §6 commit: **examinable**. §7 the
leak: **examinable, and it is the lecture**. §8 pipeline: **examinable**. §9
three models: *examinable in lecture 2, where the numbers are diagnosed*.

---

## Cell 1 — setup and versions

**Prompt to type:**

> Setup cell for a teaching notebook. Import numpy, pandas, matplotlib and
> sklearn and print their versions. Set one seed I can reuse everywhere. I need
> `root_mean_squared_error`, which arrived in sklearn 1.4 — make the cell fail
> loudly right here if the runtime is older, don't just print the version.

**Expect:** four version lines, an `assert` on the parsed scikit-learn version
with a message naming `%pip install -U scikit-learn`, and one constant —
`RANDOM_STATE = 42` or similar — defined once.

**Assert:** `assert tuple(int(p) for p in sklearn.__version__.split(".")[:2]) >= (1, 4)`

**Annotate:** short

---

## Cell 2 — the loader

**Prompt to type:**

> Load the California housing data from
> `https://github.com/ageron/data/raw/main/housing.tgz`. Write it as a function
> that downloads the tarball only if it isn't already on disk, extracts it, and
> returns the CSV as a dataframe. Assert the shape rather than trusting the
> download, then show me the first few rows.

**Expect:** `20,640 districts, 10 columns` and a five-row `head()`. Second run of
the same cell is instantaneous — that is the test of "only if it isn't already
on disk".

**Assert:** `assert housing_full.shape == (20640, 10)`

**Annotate:** short

---

## Cell 3 — what is in it

**Prompt to type:**

> Show me what's actually in it — dtypes and non-null counts, not the first five
> rows.

**Expect:** `.info()`. Ten columns: nine `float64` and one `object`
(`ocean_proximity`). Every column has 20,640 non-null values except
`total_bedrooms`, which has **20,433**. Both facts you are looking for are
invisible in a `head()`: a `head()` of five rows shows no missing bedrooms at
all, and shows `ocean_proximity` as text without telling you it is not a number.

**Assert:** none — this cell is a display. The subtraction 20,640 − 20,433 is
cell 4.

**Annotate:** short

---

## Cell 4 — the holes and the categories

**Prompt to type:**

> How many districts are missing `total_bedrooms`? Give me the count and the
> percentage. Then the counts of every `ocean_proximity` value.

**Expect:** `207` missing, `1.0%`. Categories: `<1H OCEAN` 9,136, `INLAND`
6,551, `NEAR OCEAN` 2,658, `NEAR BAY` 2,290, **`ISLAND` 5**.

Five. In the whole of California. Write that number down — it is the input to
cell 13's annotation and to the first thing lecture 2 goes looking for.

**Assert:** `assert housing_full["ocean_proximity"].value_counts().min() == 5`

**Annotate:** short

---

## Cell 5 — split before you look

**Prompt to type:**

> Split `housing_full` 80/20 into a training and a test set, seed 42. The domain
> experts say median income is what predicts price, so I need the income
> distribution to come out the same in both halves. Print the two sizes and
> check the halves don't overlap.

**Expect:** `train 16,512   test 4,128`. Whatever it does about income should be
visible as a banding step (`pd.cut` at 1.5 / 3.0 / 4.5 / 6.0, or `pd.qcut` into
five) followed by `stratify=` on that band. If `stratify` does not appear in the
call, the prompt was not obeyed however good the prose around it looks.

**Assert:**

```python
assert len(train_set) + len(test_set) == len(housing_full)
assert set(train_set.index).isdisjoint(test_set.index)
```

**Annotate:** full

* **Left open:** which bands. `pd.cut` at [1.5, 3.0, 4.5, 6.0] and `pd.qcut` into
  five quantiles are different strata and both are defensible; only one of them
  is a decision you made. The cut points matter: they put 822 districts in band
  1 and 7,236 in band 3, so band 1 is the one a random split will get wrong.
* **The usual student version:** `train_test_split(housing_full, test_size=0.2,
  random_state=42)` and nothing else. **`stratify` defaults to `None`**, so the
  split is unstratified whatever your sentence about income said, and no warning
  is issued. Measured on this data at seed 42: the unstratified test set
  over-represents income band 1 by **+6.4%** and band 5 by **+5.6%**; the
  stratified one is wrong by at most **0.36%**.
* **How you would catch it:** cell 6. Do not take the word `stratify` on trust —
  the whole content of this cell is a claim about proportions, and proportions
  are three lines to check.

---

## Cell 6 — check that the split did what you asked

**Prompt to type:**

> For each of the five income bands, print its share of the whole frame, its
> share of my test set, and the percentage error between the two. Do the same
> for a plain unstratified split with the same seed so I can see the difference
> side by side.

**Expect:** a five-row table. Stratified errors: +0.36, −0.02, −0.01, +0.03,
−0.08 (per cent). Unstratified: **+6.45**, −3.59, −1.53, +4.42, +5.63. The
largest stratified error is on band 1 — the smallest band, 822 districts — which
is exactly where you would predict it, and it is eighteen times smaller than the
unstratified error on the same band.

**Assert:** `assert strat_err.abs().max() < 1.0 < rand_err.abs().max()`

**Annotate:** short

---

## Cell 7 — look, at the training half only

**Prompt to type:**

> Histogram every numeric column of the training half. Enough bins to see the
> tails, and make the figure big enough to read.

**Expect:** **nine** panels — `ocean_proximity` is text, so pandas silently
leaves it out, and nine is the number that tells you it did. Two things should
be visible and both are about a hard right edge: `median_income` stops dead at
15.0001 and is plainly not in dollars, and `median_house_value` has a spike at
its own maximum. `DataFrame.hist` **defaults to `bins=10`**; at ten bins across
$500,000 the spike is inside a bar 50,000 dollars wide and disappears. Ask for
50.

**Assert:** none. The counting happens in cell 8 — a spike you can see is not a
number you can write down.

**Annotate:** short

---

## Cell 8 — count the cap instead of squinting at it

**Prompt to type:**

> How many training districts sit at the top of the `median_house_value` range,
> and what percentage of the training set is that? Then show me the commonest
> values below it.

**Expect:** the maximum is **500,001** and **764** districts are on it. Ask for
`>= 500_000` instead and you get **787**, because a further **23** districts sit
on a round 500,000 — the two numbers are both defensible and they are not the
same number, so say in the output which one you printed. Either way it is
**4.8%** of the 16,512 training rows. The five commonest values below the cap:
137,500 (101 districts), 162,500 (91), 112,500 (82), 187,500 (76), 225,000 (75)
— every one a multiple of **$12,500**, against a median of **3** districts per
distinct price.

**Assert:**

```python
v = housing["median_house_value"]
assert (v == v.max()).sum() == 764 and (v >= 500_000).sum() == 787
assert all(x % 12_500 == 0 for x in counts.drop(v.max()).head(5).index)
```

**Annotate:** short

---

## Cell 9 — check the famous claim

**Prompt to type:**

> Géron's description of this dataset mentions extra horizontal lines at
> 450,000, 350,000 and 280,000. Print how many training districts sit on each,
> and print the median number of districts per distinct price so I have
> something to compare them against.

**Expect:** **450,000 → 31**, **350,000 → 62**, **280,000 → 3**, against a median
of 3. Two of the three are real: 31 and 62 sit at the 99.5th and 99.8th
percentile of the count distribution. The third is not marginal, it is absent —
3 districts is the median, i.e. exactly the background. And cell 8 already told
you why: 450,000 and 350,000 are multiples of $12,500 (36× and 28×), and 280,000
is not (22.4×). The nearest four grid points to it carry 19, 54, 17 and 21
districts — 275,000 alone has eighteen times more than the line the book names.

**Assert:** `assert counts.get(280_000, 0) <= counts.median() < counts.get(350_000, 0)`

**Annotate:** short

---

## Cell 10 — a number to compare against

**Prompt to type:**

> What RMSE do I get if I ignore the features completely and predict the
> training mean for every district? Print it next to the mean itself.

**Expect:** mean **$206,334**, RMSE **$115,311**, scored on the 16,512 training
rows. If your number is **$115,727** you scored it on the test set — see below.

**Assert:** `assert abs(baseline_rmse - y_train.std(ddof=0)) < 1e-6`

**Annotate:** full

* **Left open:** *which districts.* The prompt says "for every district" and
  never says which set of districts, and the RMSE of a constant is a property of
  the rows you score it on, not of the constant.
* **The usual student version:** scoring the baseline on the test set. This is
  not a hypothetical — `notebooks/lecture-01.ipynb` as currently shipped does
  exactly this. Its cell 15 carries the comment *"From here to the very last
  cell, `test_set` is not touched again"*; its cell 27, twelve cells later,
  opens with `y_test = test_set["median_house_value"]` and scores the baseline
  on those rows. The number it prints is $115,727 and the three models it is
  later compared against are scored on the other 16,512 rows.
* **How you would catch it:** the RMSE of a constant equal to the mean of a
  sample, scored on that same sample, **is** that sample's standard deviation.
  Not approximately — to the last cent, because both are the same square root of
  the same sum. So `assert abs(rmse - y_train.std(ddof=0)) < 1e-6` passes only if
  you scored on the rows the mean came from. You can work the answer out on
  paper before you run it, which is the only kind of check worth writing.

---

### Markdown between cell 10 and cell 11 — §6 · Commit

No code cell. A fenced block, opened and closed at column 0, with three blanks
for metric, target RMSE, and the RMSE the reader expects from what they build
today. Insist on paper, not on a cell: a prediction you can silently revise is
not a prediction, and every remaining cell in this notebook is measured against
what the reader writes here.

---

## Cell 11 — the assistant writes the preprocessing

This is the shortest prompt in the notebook and it is the point of the lecture.
Type it exactly as written — do not improve it, do not add a constraint, do not
mention the order of operations.

**Prompt to type:**

> Load the housing data, scale the features and split it into training and test
> sets. Then fit a linear regression and print its RMSE.

**Expect:** a `make_pipeline(SimpleImputer(...), StandardScaler())`, a
`fit_transform` over the whole frame, a `train_test_split` of the result, a
`LinearRegression`, and **RMSE ≈ $70,469**. It runs first time, imports nothing
exotic, and the number is comfortably better than the $115,311 baseline. Nothing
in the output is a warning.

**Assert:** none, deliberately. There is nothing here to assert that would fail.

**Annotate:** full

* **Left open:** the order of the two operations. The sentence names scaling and
  splitting and does not say which comes first; English does not carry an order
  across an "and", and the assistant has to pick one.
* **The usual student version:** this is it. Fourteen words, unedited — the
  shortest and most representative prompt in the course. There is no sloppier
  variant to warn you about, and that is why this cell exists.
* **How you would catch it:** not from this cell. Write **$70,469** down on the
  same sheet of paper as your commit, next to the baseline, and run the next
  cell.

---

## Cell 12 — the same thing, the other way round

**Prompt to type:**

> Now do it the other way round: split first, fit the imputer and the scaler on
> the training half only, transform the test half with the objects you already
> fitted, and print both RMSEs and the difference between them.

**Expect:** leaky **$70,469.35**, correct **$70,468.46**, difference **$0.88**.
Both must be on the same 4,128 rows — if the assistant made a fresh split, they
are not, and the difference you are reading is not the leak.

**Assert:** `assert (yc_te.index == y_te.index).all()`

**Annotate:** full

* **Left open:** the seed on the second split. The prompt says "split first" and
  never says *split the same way*, so nothing in it requires the second
  `train_test_split` to reproduce the first.
* **The usual student version:** a bare `train_test_split(X_all, y_all,
  test_size=0.2)` on the second call. **`random_state` defaults to `None`**, so
  it draws a different 20% and the two RMSEs are computed on two different
  samples of 4,128 districts. Measured: across twenty such splits the honest
  RMSE ranges from **$67,951 to $72,833**, a spread of **$4,882**, and the gap
  you would then report as "the leak" has a median of **$911** and reaches
  **$2,518**. That is a thousand times the real answer, in the right direction,
  with a plausible size. It is the most convincing wrong result in this notebook
  and there is no ⚠ on it.
* **How you would catch it:** the assert above, and only the assert above. Two
  RMSEs printed one under the other look comparable whether or not they are —
  that is what a tidy column of numbers does to a reader. Compare the index sets,
  not the numbers.

---

### Markdown after cell 12 — the reveal

This is where the ⚠ goes, and the **only** place. Say it once: `fit_transform`
in cell 11 ran on all 20,640 rows, so the median that filled the missing
bedrooms and the mean and standard deviation that scaled every column were all
computed from a set that includes the rows we then called the test set. The
model was evaluated on rows whose own values helped define the transformation
applied to them.

Then the three reasons it is only worth $0.88, in this order:

1. You did not know it was $0.88 until cell 12. Nothing in cell 11's code said
   so and nothing in its output did.
2. It is small for three specific reasons — centring and scaling is an
   invertible affine map, ordinary least squares is equivariant under one, and
   with 20,640 rows the train and test statistics nearly coincide. The medians
   the two fits actually chose for `total_bedrooms` were 435 and 437. Remove any
   one of those three and the leak has teeth.
3. A leaked score and an honest score can be identical, so you cannot detect this
   from the number. The rule is procedural — *split first* — not because the
   damage is always large but because you cannot tell whether it is.

One caveat that belongs here and is missing from the current notebook: cells 11
and 12 split the full frame **unstratified**, so this is a third split, not the
one §3 built. 3,289 of its 4,128 test rows — **79.7%** — are rows §3 designated
as training rows. That is fine for the purpose (the two numbers being compared
are on identical rows, which is the only thing this comparison needs) and it
must be said, because the notebook spent a whole section telling the reader that
which rows go where is a decision.

---

## Cell 13 — build it properly

**Prompt to type:**

> Build one preprocessing object I can drop into a `Pipeline`: median-impute and
> standardise the numeric columns, one-hot encode `ocean_proximity`. Print how
> many columns went into each branch.

**Expect:** `8 numeric + 1 categorical`. Eight, not nine: `median_house_value`
is the label and must not be in `X_train`. If you see 9, the target is a feature
and every score after this point is meaningless.

**Assert:** `assert set(num_cols) | set(cat_cols) == set(X_train.columns)`

**Annotate:** full

* **Left open:** what the encoder should do with a category it has not seen.
  `ISLAND` has 5 districts in the whole state (cell 4) and the stratified split
  put only **2** of them in the training half, so the training part of a
  cross-validation fold can easily contain none.
* **The usual student version:** a bare `OneHotEncoder()`. **`handle_unknown`
  defaults to `"error"`.** Measured: over 30 seeds of shuffled 5-fold CV on this
  training half, **7 of 30 seeds hit at least one fold** where `ISLAND` is in the
  validation part and absent from the training part. What happens then is worse
  than a crash — `cross_val_score` catches the `ValueError`, emits a
  `UserWarning`, and writes **`nan`** into that fold. Your mean CV score becomes
  `nan`, or, if you reached for `np.nanmean`, a mean over four folds that you
  will report as five. `handle_unknown="ignore"` is the fix and it has its own
  cost: `ISLAND` becomes an all-zero row with no warning, which is the first
  thing lecture 2 goes looking for.
* **How you would catch it:** `ColumnTransformer` **defaults to
  `remainder="drop"`**, so any column you forgot to list is silently gone and
  the model trains perfectly well without it. The set assert above is the only
  thing standing between you and a feature you never notice you are not using.

---

## Cell 14 — three models, scored on their own training data

**Prompt to type:**

> Put a linear regression, a decision tree and a random forest through that
> pipeline, fit each on the training set, and print each one's RMSE **on the
> training data**. I know that is not a validation score — I want the training
> score.

**Expect:** linear **$68,233**, decision tree **$0**, random forest **$18,058**.
Exactly zero, not "about zero" — if it prints `1.2e-11` you are looking at a
different estimator or a different split.

**Assert:** `assert rmse_tree == 0.0`

**Annotate:** full

* **Left open:** nothing about the split, because there is none — and that is
  the one thing worth saying out loud in the prompt, which is why the second
  sentence is there. Without it the assistant will insert a validation split you
  did not ask for, or a comment telling you this is wrong, and the three numbers
  lecture 2 diagnoses will not be on the page.
* **The usual student version:** a bare `DecisionTreeRegressor()`, which is also
  what you want here. **`max_depth` defaults to `None`**, so the tree grows until
  every leaf is pure; on 16,512 rows with eight continuous features every
  training row ends up in a leaf of its own and the training RMSE is exactly
  `0.0`. It is the most convincing wrong number in the course, and the reason it
  convinces is that it is not approximately anything.
* **How you would catch it:** `assert rmse == 0.0`, exactly, with no tolerance.
  A model that is genuinely good on training data gives you 18,058 like the
  forest. A hard zero is not a model that is perfect, it is a model that has
  memorised, and the exactness is the diagnosis rather than a rounding artefact.

---

### Markdown after cell 14 — §9 · Where we are

Three numbers, and the reader writes their best one on the same sheet of paper
next to what they predicted in §6. State plainly which comparison is legitimate
and which is not: **68,233 / 0 / 18,058 are all training scores and the baseline
115,311 is a training score, so those four are on the same 16,512 rows and can
be compared.** The $70,469 and $70,468 from cells 11 and 12 are on a different
split and belong to the leak argument, not to this table. Do not put all six in
one table.

Do not fix anything. Being wrong is the point and the diagnosis is lecture 2.

---

## Exercises

Each one lists the cells to re-run, in order. Nothing in this notebook is
non-idempotent — every cell that fits a model constructs it inside itself — so
re-running is safe, and the only hazard is running a cell whose inputs a later
cell has not yet rebuilt.

1. **Change the seed.** Set `RANDOM_STATE = 7` in cell 1, then re-run **1 → 5 →
   6 → 7 → 8 → 9 → 10 → 13 → 14**. Before you re-run 11 and 12, read them: your
   prompt for cell 11 never mentioned a seed, so check whether the code you were
   given picked up your constant, hardcoded 42, or left `random_state` out
   altogether — all three are things an assistant returns for that prompt, and
   only the first will follow you. Which of the numbers in this file move, and by
   how much? The tree's zero does not. Ask why that one is invariant.
2. **Break the stratification.** Delete `stratify=` from cell 5, re-run **5 →
   6**, and read the second column of cell 6's table. Then re-run **10** and say
   how much of the change in the baseline is the different rows and how much is
   anything else. (There is nothing else.)
3. **Give the leak teeth.** Three variants, each removing exactly one of the
   three reasons the reveal gives. Edit **both** cell 11 and cell 12 each time,
   then re-run **11 → 12**.
   *(a)* Swap `StandardScaler` for `MinMaxScaler`. The leak stays at **$0.88** to
   the cent. It is fitted on the two extreme values rather than on the mean, so
   it looks far more exposed to a single test row — and it is still an affine
   map, which is the only property that mattered.
   *(b)* Swap it for `QuantileTransformer(n_quantiles=100, random_state=42)`,
   which is not affine. The leak becomes **$204.92** — 233 times larger, on the
   same rows, with the same estimator and the same seed.
   *(c)* Keep `StandardScaler`, replace `LinearRegression` with
   `KNeighborsRegressor(5)`, and run it on `housing_full.iloc[:500]`. Ordinary
   least squares is equivariant under an affine map; nearest neighbours is not,
   and 500 rows is small enough that the train and test statistics genuinely
   differ. The leaky score comes out **$2,658 better** than the honest one.
   Reason 1 is (b), reasons 2 and 3 together are (c), and (a) is the one that
   changes nothing. Say which of the three you would have guessed.
4. **Drop the seed on purpose.** Remove `random_state` from the second
   `train_test_split` in cell 12 and re-run **12** five times. Record the five
   "leak" figures. They will not agree, and none of them will be $0.88.
5. **Find the missing feature.** Delete one name from `num_cols` in cell 13,
   re-run **13 → 14**, and see what fails. The set assert should stop you. Now
   delete the assert as well and re-run the same two cells: nothing complains,
   `remainder="drop"` throws the column away in silence, and the forest still
   fits. Measured over all eight numeric columns, the forest's training RMSE
   lands anywhere from **$17,826** to **$21,017** against **$18,058** with all
   eight — and dropping `housing_median_age` or `total_bedrooms` makes the number
   go **down**, which is to say a feature you lost by accident can make your
   score look better. That is what cell 13's third bullet is about.

---

## Defects found in the current notebook

Against `notebooks/lecture-01.ipynb` as it stands (42 cells, 13 of them code).
Each item was checked by running the code, not by reading it. Cell numbers are
zero-based indices into `nb["cells"]`, matching the file.

**1 · The baseline is scored on the test set, twelve cells after the notebook
promises it will not be.** §2.1, §3.2. Cell 15 ends with the comment `# From
here to the very last cell, test_set is not touched again`, and cell 14's
annotation makes that comment the *catch* for the whole section. Cell 27 opens
`y_test = test_set["median_house_value"]` and scores the baseline on those rows.
The promise is broken at cell 27 and the "very last cell" use it promises never
happens — cell 40 uses `y_train`. Verified by string search: `test_set` appears
in cells 14, 15 and 27 only.

**2 · The baseline and the three models are scored on different rows.** §2.1, the
most serious item. Cell 27 scores the constant on the 4,128 test rows
(**$115,727.19**); cell 40 scores the three models on the 16,512 training rows.
Cell 25 says *"Everything you build today has to beat this"* and cell 41 says
*"we open by comparing them"*. The correct matched figure is **$115,310.56** —
the training-row baseline — and it is not in the notebook. The gap is $416.63,
small, but the comparison as written is between two windows and nothing says so.

**3 · The `⏱ 20 s` marker is wrong by a factor of three to twenty.** §1.1, §7.1.
Cell 39's prompt label reads `⏱ 20 s` and cell 40's first line reads `# ~20 s:
the forest is 100 trees on 16,512 rows`. Measured: all three models together
take **0.9 s** with `n_jobs=-1` on 16 arm64 cores and the forest alone takes
**6.0 s** with `n_jobs=1`. It is the only ⏱ in the notebook and it is the one
number a reader uses to decide whether to walk away from the laptop.

**4 · No cell has a stored output.** §1.2, §9. All 13 code cells have
`execution_count: null` and zero outputs. Every figure in the prose is therefore
unreconcilable against the artefact as shipped, and the check in §9's table
("every ≥4-digit prose figure appears in a stored output") has nothing to run
against. This is upstream of items 5–8: they are all instances of it.

**5 · The 6.4% / 0.36% stratification figures appear in prose and in a prompt
`constraint`, and in no cell.** §1.2. I re-derived them and **they are correct** —
at seed 42 the unstratified test set's worst band error is **+6.45%** and the
stratified one's is **+0.365%** — but no cell computes them, so a reader cannot
check them. This is in the notebook that two sections later (cell 22) tells the
reader that when a source names specific numbers about your data, the numbers are
checkable. Cell 6 of the script above is the fix.

**6 · "$68,000" in cell 26 is a forward reference to a training score.** §1.2.
The figure is real — the linear regression's RMSE on the training data is
**$68,232.84** — but it is produced by cell 40, fourteen cells later, and the
bullet presents it as the RMSE somebody would "report", which is the one thing a
training-data score must never be.

**7 · Cell 21 puts 23 districts on both sides of its own definition of the cap.**
`capped = (housing["median_house_value"] >= 500_000).sum()` prints **787**, and
the same cell then lists "the five commonest values below the cap" after
`counts.drop(counts.index.max())`, which drops only the value **500,001**. The
training half has **764** districts at 500,001 and **23** at exactly 500,000, so
those 23 are counted as at the cap in the first print and as below it in the
second. Neither the 764 nor the 23 appears anywhere.

**8 · Cell 22's verdict on the famous claim does not match the counts.** §1.1,
§1.3. The prose says *"one of the three is real, one is marginal, and one is
indistinguishable from the background"*. Counted on the training half: **350,000
→ 62 districts** (99.78th percentile of the per-price count distribution),
**450,000 → 31** (99.54th percentile), **280,000 → 3** (the median is 3). Two are
real and one is absent; 31 districts against a median of 3 is not "marginal". The
stronger and fully-derivable statement is one the cell above already sets up:
280,000 is the only one of the three that is not a multiple of $12,500 (22.4×),
and its four nearest grid neighbours — 262,500 / 275,000 / 287,500 / 300,000 —
carry 19 / 54 / 17 / 21 districts.

**9 · "You will meet the same error worth far more than a dollar in about an
hour" resolves to nothing.** §3.3, §7.5. Cell 35. No cell, section or later
lecture is named, and the reader working alone at 23:00 has no hour of lecture
to sit through. Name the lecture and the section, or cut it.

**10 · The Pipeline is justified by a cross-validation the notebook never runs.**
§3.2. Cell 36 and cell 37's `constraint` both argue for one `Pipeline` "so that
cross-validation refits *all* of it on each fold". `cross-valid` appears three
times in the markdown; `cross_val` appears **zero** times in any code cell.
Nothing in lecture 1 ever refits anything on a fold, so the claim the reader is
asked to accept is never demonstrated — and it is demonstrable in four lines.

**11 · The defect is announced three times before the cell that contains it.**
§8.1. Cell 0: *"Cells marked ⚠ read before running contain a defect on purpose"*.
Cell 29: *"⚠ Read before running. It runs, it imports nothing exotic, and it
prints a believable number."* Cell 30: a prompt label beginning `⚠`, and a **Left
open** bullet that states the entire diagnosis — *"`fit_transform` ran on ALL the
rows"* — before the cell has been run once. By the time the reader's eye reaches
the code, `fit_transform(X_all)` is the only line they are looking at. Nobody
falls in, so nobody learns what it is like to believe a number.

**12 · Thirteen code cells, thirteen full three-bullet annotations, zero short
boxes.** §6.1. The rule is five to eight full ones and never more than ten. Every
box in the notebook is full, including the setup cell and the `.info()` cell.

**13 · "Examinable" appears twice, both times on the setup cell.** §8.3. Cell 2's
third bullet and cell 3's comment, both saying the setup is *not* examinable.
Sections 2 through 9 — including §3, §7 and the commit — carry no marker at all.

**14 · Several `catch` fields are not catches.** §6.3. The slot exists to force a
check with a knowable outcome, and in cell 2 it reads *"not examinable, and it is
here because a version mismatch produces a confusing error"*, in cell 14 *"the
comment on the last line"*, in cell 26 *"rule 2 of this course: a metric with
nothing to compare it to is decoration"*. Three restatements of the section's
thesis in the one slot that is supposed to be executable.

**15 · Cell 40 raises `NameError: y_train` if cell 27 has not been run.** §4.3.
`y_train` is bound in the *baseline* cell — the cell whose subject is a constant
model — and consumed fourteen cells later as the training label for all three
real models. Restart-and-run-all passes, so this is not a build failure; it is
the specific out-of-order hazard §4.3 asks to be named, and it is not named. No
name in the notebook is rebound to a different type, so §4.1 is clean; the loop
variable `model` in cell 40 takes three different estimator classes, which is
what §4.1's own evidence paragraph suggests giving a throwaway name.

**16 · §7's comparison is sound but silently uses a third split.** §2.1, in the
notebook's favour on the first point. I verified that cells 31 and 34 score the
leaky and honest models on **identical** test rows (`(yc_te.index ==
y_te.index).all()` is `True`), so the $0.88 is a real like-for-like measurement
and the headline finding stands. But both cells split the full 20,640 rows
**unstratified**, so this is a third partition: **3,289 of its 4,128 test rows,
79.7%, are rows §3 designated as training rows.** The notebook spends a whole
section establishing that which rows go where is a decision, and then makes a
different one without a word.

**Clean:** no markdown line is indented four or more spaces outside a fence
(§5.1), no fence marker is indented (§5.2), the single fenced block in cell 28
opens and closes at column 0, no ```` ```python ```` block appears in any
markdown cell so §3.1 has nothing to violate, and the build script's own
`_ensure_prompt_note` guarantee holds — every one of the 13 code cells is
preceded by a prompt box, so the header's claim in cell 0 is true.
