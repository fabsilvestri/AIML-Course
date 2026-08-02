# Lecture 3 — Finding the rare event · Colab prompt script

Rebuild `notebooks/lecture-03.ipynb` in Colab by prompting. Source of truth for
the content: `tools/notebooks/lecture_03.py`. Géron, Chapter 3.

**Who this is for.** One person at a Colab keyboard, typing prompts into the
notebook's assistant, reading what comes back, and keeping a sheet of paper
beside the laptop. No GPU. Everything here runs on the free CPU runtime.

**Machine assumed.** Colab's free CPU runtime, **2 vCPU**. Every `⏱` figure
below is the one the module states; the module names no hardware and the
notebook stores no outputs, so **none of them has been re-measured** — see the
defect report. Put `%%time` on the first line of every `⏱` cell and write down
what yours actually says. If a cell is running at more than about 3× the stated
figure you are on a throttled runtime, not stuck.

**Seventeen cells.** The current notebook has sixteen. Cell 3 here is the
module's dtype cell split in two, so that the trap fires *before* it is
explained (GUIDELINES §8.1). Everything else is one-for-one.

**Annotation budget (§6.1).** Seven of seventeen cells carry the full
three-bullet annotation: **3, 6, 9, 10, 11, 14, 16**. The rest carry the short
specification only. Every "usual student version" bullet below names a
scikit-learn / numpy / matplotlib default that was checked against the installed
libraries (scikit-learn 1.7.2, numpy 2.3.5, python 3.13.5), or a failure that is
sitting in the shipped notebook right now.

**Cell 15 is deliberately un-annotated.** It is the lecture's centrepiece
defect. Run it, write the number down, and only then read §12's opening. Do not
read ahead.

---

## Cell 1 — setup and one seed

**Prompt to type:**

> print the versions of python, sklearn, numpy, pandas and matplotlib, assert
> sklearn is at least 1.4, and set RANDOM_STATE = 42

**Expect:** five version lines, then nothing. `RANDOM_STATE` bound once and used
by every split, every model and every shuffle from here on.
**Assert:** `tuple(int(p) for p in sklearn.__version__.split(".")[:2]) >= (1, 4)`
**Annotate:** short

Not examinable — engineering hygiene. It is here because a version mismatch
surfaces twenty cells away as an `ImportError` in a cell that has nothing to do
with versions.

---

## Cell 2 — load MNIST

**Prompt to type:**

> load mnist_784 from openml with sklearn, as numpy arrays not a dataframe.
> print the shape and dtype of X and y

**Expect:**

```
X (70000, 784) int64
y (70000,) object
```

`X` is `int64` on scikit-learn 1.7 (pandas parser) and `float64` on older images
that use the liac-arff parser — either is fine. `y`'s dtype is the one to look
at. 784 = 28 × 28; if the second dimension is not a perfect square you are not
holding images, whatever the variable is called.

**Assert:** `X.shape == (70000, 784)` — assert it rather than trusting the
download. A truncated read gives a smaller `X` and this catches it.
**⏱** ~30 s on a cold runtime (a 15 MB gzip from openml.org), then cached by
scikit-learn under `~/scikit_learn_data`; about a second on every later run.
**Annotate:** short

---

## Cell 3 — how many fives are there?

Ask the obvious question. Do not read the next section until you have run this
and written the answer on your sheet.

**Prompt to type:**

> how many of these images are 5s?

**Expect:** one number, printed without complaint:

```
0
```

**Assert:** none — and that is the point. There is nothing here for an assertion
to catch, because nothing went wrong at the level the language can see.
**Annotate:** full

* **Left open:** what type the labels are. "How many are 5s" is a question about
  the data, and both `y == 5` and `y == "5"` are honest answers to it. Nothing
  in the question forces the answer to check which one this array wants.
* **The usual student version:** the code you just got. `fetch_openml` with
  `as_frame=False` returns the target as an **object array of one-character
  strings** — `y.dtype` prints `object` and `repr(y[0])` is `'5'`. `y == 5`
  compares a `str` with an `int` element by element, every comparison returns
  `False`, `.sum()` returns `0`, and numpy 2.3 raises **no warning** (checked
  for both the `object` array OpenML actually returns and for a `<U1` array).
  This is a library default nobody asked for, and it is the expensive kind: the
  comparison you are about to write a hundred times fails silently.
* **How you would catch it:** count the positives immediately after building any
  boolean mask, and know the number you expect *before* you look. `0` is a
  perfectly good answer to "there are none". The code cannot tell that apart
  from "the comparison is broken". You can.

---

## Cell 4 — fix the dtype and re-count

**Prompt to type:**

> the labels are strings. cast y to uint8, print the dtype before and after,
> and count the 5s again

**Expect:**

```
before: object '5'
y == 5 finds 0 images
after:  uint8 5
```

and the count is now **6,313** across all 70,000 rows (5,421 of them in the
first 60,000 — you will meet that number again at cell 7).

**Assert:** `y.dtype == np.uint8` and `set(np.unique(y)) == set(range(10))` —
all ten digits present after the cast.
**Annotate:** short

⚠ **This cell is not idempotent, and you must not re-run it.** Once `y` is
`uint8`, running it again prints `before: uint8 5` and `y == 5 finds 6313
images`, and the demonstration reverses itself without erroring. If you need to
redo it, re-run cell 2 first.

---

## Cell 5 — look at sixteen of them

**Prompt to type:**

> show the first 16 images in a 2x8 grid with their labels as titles

**Expect:** sixteen recognisable digits, titled

```
5 0 4 1 9 2 1 3 1 4 3 5 3 6 1 7
```

Read the titles against the pictures. You know what a 5 looks like; that is the
whole test and it takes four seconds. What you are checking is not "are these
digits" but "are these titles on *these* digits" — a labels-shifted-by-one bug
is plausible and silent everywhere downstream, and this is the only cell where
it would show.

**Assert:** none — this one is for your eyes. But note that the reshape is not
optional: `plt.imshow(X[0])` on the flat 784-vector raises
`TypeError: Invalid shape (784,) for image data` (checked). It does not draw a
stripe; it stops.
**Annotate:** short

A 784-vector and a 28 × 28 image are the same object in two shapes. The model
only ever sees the 784 numbers in a fixed order, and has no idea that pixel 0
and pixel 28 are vertical neighbours.

---

## Cell 6 — split into train and test

Type this exactly as written. Do not help it.

**Prompt to type:**

> split X and y into a training set and a test set

**Expect:** almost certainly `train_test_split(X, y, test_size=..., random_state=42)`
and a pair of sizes that look right. Before you accept it, print `y_train[0]`.
Under the standard MNIST split that is **5** — the digit in the top-left tile of
the grid you just plotted. After a shuffled split it is 5 about 9 times in 100.

**Then re-prompt:**

> no — MNIST is already partitioned and already shuffled: the first 60000 rows
> are the standard training set and the last 10000 the standard test set. split
> it by position instead

**Expect:** `X_train, X_test = X[:60000], X[60000:]` and the same for `y`, then
`train 60,000   test 10,000   features 784`.

**Assert:** `len(X_train) == 60000 and len(X_test) == 10000`,
`X_train.shape[1] == X_test.shape[1] == 784`, and
`X_train.min() == 0 and X_train.max() == 255` — the pixel range check is the one
that earns its keep: if scaling has crept in above this line, min/max stop being
0/255 and this is no longer the raw split.
**Annotate:** full

* **Left open:** which rows go where. "Split into train and test" has two right
  answers, and this dataset accepts only one of them. The prompt asks for sizes
  and gets sizes.
* **The usual student version:** `train_test_split`. Its `shuffle` parameter
  defaults to `True` (checked in the signature), so you get the right *sizes*
  and a different set of *rows* from every published MNIST result — and nothing,
  anywhere, reports that. Using the standard cut is the only thing that makes
  your number comparable with anybody else's.
* **How you would catch it:** `y_train[0] == 5`. One line, and you can work out
  the expected answer on paper before you run it, because you plotted that exact
  image two cells ago.

Write in a comment, on the split cell: **from here to the very last cell of the
next notebook, `X_test` is not touched.** Nothing enforces that. It is a promise
you keep by hand.

---

## Cell 7 — one class against the rest

**Prompt to type:**

> make boolean labels for is-it-a-5 on train and test, and print how many
> positives and the base rate in the training set

**Expect:**

```
positives 5,421   negatives 54,579
base rate 0.09035  (9.04%)
```

The base rate before anything is fitted, so the yardstick exists before there is
a result to compare it against.

**Assert:** `y_train_5.dtype == bool` and
`y_train_5.sum() == 5421, "did the cast to uint8 happen?"` — this assertion does
double duty. A skipped cast leaves `y_train` as strings, `y_train == 5` returns
an all-`False` bool array with no warning, the sum is 0, and this is where it
stops.
**Annotate:** short

Two lines, and the imbalance is created. Nothing about MNIST is imbalanced. Our
*question* is.

---

## Cell 8 — how many of each digit

**Prompt to type:**

> print a text bar chart of how many of each digit are in the training set

**Expect:** ten rows, counts
`[5923, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 5949]`. The rarest digit
is **5** at 5,421 and the commonest is **1** at 6,742 — a ratio of **1.24**.

**Assert:** none.
**Annotate:** short

Two different things are being called balance. The ten classes are near-equal;
the binary task built out of them is 9 to 1. A dataset is not imbalanced or not
— a **task** is. Count the positives of the task you are solving, never the
classes of the dataset you started from.

---

## Cell 9 — the anchor: what does *nothing* score?

Rule 2 of this course: a metric with nothing to compare it to is decoration. So
before building anything, measure the cheapest possible detector — one that
looks at nothing and answers "not a 5" every time.

**Prompt to type:**

> write a baseline estimator that always predicts False, and score it with
> stratified 3-fold cross-validation on the training set, seed 42, accuracy per
> fold and the mean

**Expect:**

```
per fold: [0.90965 0.90965 0.90965]
anchor accuracy 0.90965
```

Identical in all three folds, and exactly `1 - base_rate`. That is not luck —
see the third bullet.

**Assert:** `np.isclose(anchor.mean(), 1 - base_rate, atol=1e-4)`. Anchor and
`1 - base_rate` must agree; if they do not, the fault is in the splitter or the
scorer, not in an estimator that does nothing.
**Annotate:** full

* **Left open:** how the baseline is to be **measured**. "A baseline that always
  says no" names an estimator, not an evaluation — and `1 - base_rate` gives the
  same number in one line without any of this. The reason to push a do-nothing
  estimator through the real splitter and the real scorer is that the splitter
  and the scorer are also under test. If your CV is misconfigured, this is where
  it says so, while a real model would still hide it.
* **The usual student version:** `DummyClassifier()` and quoting whatever it
  prints. Its `strategy` defaults to `"prior"` (checked in the signature) — for
  `predict` that is the majority class, so it is the right answer here and the
  wrong habit, because the same default is a probability vector the moment you
  ask for one in the next lecture. And `StratifiedKFold` defaults to
  `n_splits=5, shuffle=False` (checked): a prompt that does not name a fold
  count gives you five folds, not the three every later cell in this notebook
  compares against.
* **How you would catch it:** the arithmetic is exact and you can do it on
  paper before running. 5,421 / 3 = 1,807 and 60,000 / 3 = 20,000, both without
  remainder, so a stratified 3-fold split puts exactly 1,807 positives in each
  held-out fold, and never-fires accuracy is 1 − 1807/20000 = **0.90965** in
  *every* fold. Verified: all three folds come out at 0.909650. If your folds
  differ from each other, your splitter is not stratifying.

**90.96%, with no model, no fit and no features.** So 90.96% is *zero* in the
units that matter, and the only interesting quantity today is the distance
above it.

---

## Stop here. On paper, not in the notebook.

```
Metric:                                          ____________
Accuracy a good detector would need:             ____________
Accuracy I expect from the model I build today:  ____________
```

You are estimating, not guessing: doing nothing scores 90.96%, a perfect
detector scores 100%, and your number lives between them. Saying *where* is the
exercise. Write it in ink, in this notebook you could quietly revise it.

---

## Cell 10 — the detector

**Prompt to type:**

> fit an SGDClassifier on the training set with a StandardScaler, both in one
> pipeline, seed 42

**Expect:** a `Pipeline` printed as fitted, with steps `standardscaler` and
`sgdclassifier`.
**Assert:** `clf.named_steps["sgdclassifier"].coef_.shape == (1, 784)` — one
weight per pixel, one row.
**⏱** ~30 s stated by the module (2 vCPU assumed; not re-measured). One pass
fits the scaler and then the classifier over 60,000 × 784.
**Annotate:** full

* **Left open:** which linear model. `SGDClassifier` is a *solver*, not a model,
  and its `loss` defaults to `"hinge"` (checked) — that is a linear SVM, not
  logistic regression. The consequence arrives later and hard: this object has
  no `predict_proba` at all
  (`AttributeError: This 'SGDClassifier' has no attribute 'predict_proba'`,
  checked), which is the wall you hit the moment you want to move a threshold.
* **The usual student version:** `X_scaled = StandardScaler().fit_transform(X_train)`
  on its own line, then a classifier on `X_scaled`. It trains identically today
  and leaks the moment it meets cross-validation, because the scaler has already
  seen the held-out rows. The scaler goes **inside** the pipeline so that CV
  refits it per fold and leakage is structurally impossible rather than merely
  avoided — the standing constraint from Lecture 2.
* **How you would catch it:** `(1, 784)` is one row of weights over 784 pixels;
  a shape of `(10, 784)` means you passed `y_train` instead of `y_train_5` and
  fitted the ten-class problem by accident. Second check, and nobody does it:
  67 pixels of `X_train` are constant across all 60,000 rows (verified).
  `StandardScaler` sets `scale_` to 1.0 for a zero-variance column and says
  nothing — no warning, no error. `(clf.named_steps["standardscaler"].scale_ == 1).sum()`
  should be 67.

---

## Cell 11 — two predictions, and what not to assert

Step 4 of the working method, and the cheapest twenty seconds in the lecture.

**Prompt to type:**

> predict on one image that is a 5 and one that is not, print both predictions
> with the true label, and assert the shape and dtype of predict on the first 8
> rows — do not assert that either prediction is right

**Expect:** two lines. `np.argmax(y_train_5)` is index **0** (the 5 in the
top-left tile of the grid) and `np.argmin(y_train_5)` is index **1** (a 0).
Whether the model gets them right is not the output you are here for.

**Assert:** `clf.predict(X_train[:8]).shape == (8,)` and
`clf.predict(X_train[:8]).dtype == bool`. The shape and the dtype — never the
answer.
**Annotate:** full

* **Left open:** how "an image that is a 5" is to be found. There is an index of
  a five (`np.argmax(y_train_5)` → 0) and an index of the *largest label*
  (`np.argmax(y_train)` → 4), and English does not distinguish them.
* **The usual student version:** the one sitting in the shipped notebook. Cell
  33 of `notebooks/lecture-03.ipynb` reads
  `i_pos = int(np.argmax(y_train))          # first 5`. Verified against the
  data: that is index **4**, and index 4 is a **nine**. The same cell prints
  `bool(y_train[i])` as its truth column, so `bool(9)` prints `True` and the
  notebook displays a nine as its known positive. Nothing raises; the assertions
  below it still pass, because they are on the shape.
* **How you would catch it:** ask what a *broken* model would score on your
  assertion. A detector that never fires gets one of these two examples right,
  so `assert clf.predict([X_train[i_pos]])[0] == True` feels like a test and is
  satisfied by luck on a class that is 9% of the data. If a broken model passes
  your assertion, the assertion is decoration. That is why this one is on
  `.shape` and `.dtype` — and it is the argument of the whole lecture.

---

## Cell 12 — cross-validated, against the anchor

**Prompt to type:**

> cross-validate the pipeline with the same 3 folds, accuracy per fold and the
> mean, and print the anchor and the gap in percentage points next to it

**Expect:** three fold accuracies, their mean, the anchor `0.90965`, and the gap
printed as a subtraction you did not have to do yourself. **Write the mean and
the gap on your sheet.** The module's prose puts the mean near 96.9% and the gap
near 5.9 points; those figures are not re-derived here (training cell, not run —
see the defect report), so the number that counts is yours.

**Assert:** none. Report **every fold**, not just the mean; fold agreement is a
finding and you only have it because you printed them.
**⏱** ~70 s stated by the module (2 vCPU assumed; not re-measured) — three
complete refits of scaler and classifier. Pass `n_jobs=-1`; on 2 vCPU that buys
less than you expect.
**Annotate:** short

Note the default you have just avoided: `cross_val_score`'s `cv` defaults to
`None`, which is 5-fold. Passing the `StratifiedKFold` object from cell 9 is
what makes this number comparable with the anchor — same splitter, same seed,
same rows.

---

## Cell 13 — what the scaler is worth

**Prompt to type:**

> run the same cross-validation on a bare SGDClassifier with no scaler, same
> folds and seed, and print the scaled and unscaled fold accuracies side by side

**Expect:** two rows of three folds. Compare the **worst unscaled fold** with
the anchor, not the mean: `raw.min() - 0.90965`, in points. The module reports
the two means within about two points of each other and one unscaled fold about
five points below its siblings — **not re-derived here** (training cell). Your
own numbers are the evidence.

**Assert:** none. Change exactly **one** thing — same estimator, same seed, same
splits, same rows — or you are measuring the change you did not make.
**⏱** ~30 s stated by the module (2 vCPU assumed; not re-measured).
**Annotate:** short

The interesting statistic here is the **spread**, not the average. If a fold
sits near the anchor, that configuration failed completely on that split and the
mean is covering for it.

---

## Cell 14 — cross-validation, by hand

Write this loop once in your life. After that use the library, but you will know
what it did.

**Prompt to type:**

> write the same 3-fold cross-validation as an explicit loop over cv.split, and
> check the numbers match cross_val_score

**Expect:** two rows of three numbers that agree to every printed digit.
**Assert:** `np.allclose(by_hand, scores), "the library is doing something else"`
**⏱** ~70 s stated by the module (2 vCPU assumed; not re-measured) — the same
three fits, done by hand.
**Annotate:** full

* **Left open:** what has to be fresh each fold. The prompt says "the same
  cross-validation" and says nothing about the estimator's state, which is the
  only thing that can go wrong in a nine-line loop.
* **The usual student version:** `fold_clf = clf` instead of `clone(clf)` — and
  here the notebook's own explanation of the damage is wrong, which is worth
  more than the bug. `SGDClassifier.warm_start` defaults to `False`, and the
  docstring is explicit: fit will "just erase the previous solution". Verified —
  refitting a fitted `SGDClassifier` on new data gives coefficients identical to
  a fresh estimator fitted on that data. So the three fold accuracies come out
  **the same**, not higher, and folds do not contaminate each other. The real
  damage is elsewhere and entirely silent: without `clone`, the loop refits
  `clf` itself, and after it `clf` is a model fitted on 40,000 rows instead of
  60,000. Every later cell that uses `clf` — the training-set accuracy at cell
  15, the summary table at cell 17 — is then quietly measuring a different
  model.
* **How you would catch it:** two assertions. `np.allclose(by_hand, scores)`
  first; then check that the object you fitted at cell 10 is still that object,
  with `clf.named_steps["standardscaler"].n_samples_seen_ == 60000`. Under
  `clone` it is 60000; under `fold_clf = clf` it is 40000, and nothing else in
  the notebook will ever tell you.

---

## Cell 15 — evaluate the classifier

Type the prompt. Run the cell. Write the number on your sheet. Then read the
next section — not before.

**Prompt to type:**

> evaluate my classifier on the MNIST 5-detector and print the score

**Expect:** one line, `Accuracy: 0.9xxx`, and a number **higher** than the mean
you wrote down at cell 12.
**Assert:** none.
**Annotate:** short

---

## ⚠ Now open it up

**Reviewer question 1 — what touched what?** `clf` was **fitted** on `X_train`.
It then **predicted** on `X_train`. The score was computed on `X_train`. The
same 60,000 rows, three times.

**Reviewer question 5 — what is the default you did not ask for?** Two of them,
in three lines. The assistant chose **accuracy**, because you said "the score"
and accuracy is what a classifier reports when nobody says otherwise. And it
chose to evaluate on the training set, because your prompt named no evaluation
set. Neither choice was flagged. You asked for a number and you got one.

Now measure the damage.

---

## Cell 16 — the gap, over five seeds

**Prompt to type:**

> repeat the cross-validation for seeds 42 to 46 keeping both the train and the
> test score of each fold, then print the mean gap between them, its standard
> deviation, and how many of the pairs are positive

**Expect:** `15 paired measurements (5 seeds x 3 folds)`, a train mean, a CV
mean, the gap in percentage points with an sd beside it, and `positive in
15/15`. The module reports the gap at about half a point — **not re-derived
here** (training cell).

**Assert:** `(gaps > 0).all(), "a model always fits what it was fitted to at
least as well"` — the sign is guaranteed; the size is not.
**⏱** ~4 minutes stated by the module (2 vCPU assumed; not re-measured) —
fifteen fits. This is the longest cell in the notebook. Start it, then read the
paragraph below while it runs.
**Annotate:** full

* **Left open:** what is to be compared with what. "Measure the overfitting"
  names no estimator of it, and a single-seed gap of half a point is
  indistinguishable from noise — while the argument this lecture is about to
  make rests on the gap being real.
* **The usual student version:** `cross_validate(...)` and reading `test_score`
  only. `return_train_score` defaults to **`False`** (checked in the signature),
  so the train scores you need are simply absent from the returned dict, and the
  natural conclusion is that there is nothing here to compare. You have to ask
  for the other half of the measurement by name.
* **How you would catch it:** pair them and count signs. Each gap must come from
  **one** fitted model — same seed, same fold — so that fold-to-fold variation
  cancels; if any pair comes out negative you have crossed seed *i*'s train
  score with seed *j*'s test score. Then print the sd next to the mean: if the
  spread of the gap is comparable to the gap, you have measured nothing, and
  fifteen pairs is what makes that judgement possible.

### Half a point. So why is it a bug?

1. **You did not know it was half a point until you measured.** A decision tree
in the previous lecture had a gap of the entire number, and nothing in either
piece of code said which case you were in.
2. **It is this small for a specific reason.** A linear model with 785
parameters — 784 weights and one intercept — cannot memorise 60,000 instances.
Give the same code an unconstrained tree and the gap is everything.
3. **The sign is guaranteed and the size is not.** Fifteen out of fifteen, and
the assertion depends on it, because a model always fits the data it was fitted
to at least as well as data it has not seen.

The rule is procedural: never score on the rows you fitted. Not because the
damage is always large, but because you cannot tell whether it is.

### The corrected prompt

> evaluate this classifier with stratified 3-fold cross-validation on the
> training set, seed 42. report accuracy per fold and the mean, and in the same
> table the same metric for a baseline that always predicts the negative class.
> do not touch X_test

Notice what is *not* in it: the metric is still accuracy. We named it, so it is
ours — and being ours is what makes it something we can be held to.

---

## Cell 17 — the five numbers to carry forward

**Prompt to type:**

> print a table with the number of positives, the base rate, the never-fires
> accuracy, our cross-validated accuracy and our accuracy on the training rows

**Expect:** five rows, in that order, the anchor sitting *inside* the table
rather than in the prose above it. Four of the five you have already written
down; the fifth is the one you must not report.

**Assert:** none. The table is deliberately without a verdict — the verdict is
the next lecture.
**Annotate:** short

Photograph it. Every number in Lecture 4 is measured against one of these five.
And their **status** matters as much as their value: four measured, one (the
1,000 items per shift in the brief) stated by the client and never verified, one
discarded.

---

## Before you close the tab

Add one line under the three you committed to at §9:

```
Best accuracy I actually obtained:  ____________
```

Bring the sheet. **Do not tidy this notebook, do not tune anything, and do not
delete the version with the bug in it.** You will need it.

### Re-run order, if you have to redo anything (§7.2)

- Changed the seed at cell 1 → re-run **1, 9, 10, 12, 13, 14, 15, 16, 17** in
  that order. Cells 2–8 do not depend on it. Budget about 7 minutes.
- Re-ran cell 2 for any reason → re-run **2, 4, 6, 7, 8** before anything else,
  or `y` is strings again and cell 7's assertion stops you.
- Never re-run cell 4 on its own. See the warning on that cell.

---

## Defects found in the current notebook

`notebooks/lecture-03.ipynb`, 56 cells, 16 of them code. Each item below says
how it was checked. Everything marked **verified** was re-derived with `python3`
against the notebook JSON or against the MNIST data itself (scikit-learn 1.7.2,
numpy 2.3.5, matplotlib, python 3.13.5).

### Wrong code, wrong claims

1. **Cell 33 finds a nine and calls it a five.** `i_pos = int(np.argmax(y_train))
   # first 5` returns index **4**, and `y_train[4]` is **9**. The truth column
   is printed as `bool(y_train[i])`, so `bool(9)` renders as `True` and the cell
   presents a nine as its known positive. `i_neg = int(np.argmin(y_train))` is
   index 1, a zero — a non-5 by accident, not by construction. The correct
   expressions are `np.argmax(y_train_5)` → 0 and `np.argmin(y_train_5)` → 1.
   The two `assert`s below still pass, because they are on the shape. **Verified
   against the data.** This is the most serious defect found: the cell whose
   whole subject is "assert the shape, not the answer" is displaying the wrong
   answer.
2. **Cell 31 points at code that is not there.** "`X_train[0]` is the 5 we
   plotted above" — true of the data (`y_train[0] == 5`, verified), but cell 33
   never uses index 0. §3.1/§3.3. **Verified.**
3. **`imshow` of a flat vector is an error, not a stripe.** Cell 11's prompt box
   claims "imshow of a 784-long vector is not an error, it is a stripe". It
   raises `TypeError: Invalid shape (784,) for image data`. **Verified by
   running it.** §3.2 — a claim offered to the reader that does not execute as
   described.
4. **The `clone` explanation is false.** Cell 41: "Reusing `clf` here would
   train the same object three times in succession, each fold starting from the
   previous fold's parameters", repeated in cell 42's bullets as "the accuracies
   come out higher" and "The numbers rise". `SGDClassifier.warm_start` defaults
   to `False` and its docstring says `fit` will "just erase the previous
   solution"; refitting a fitted estimator gives coefficients identical to a
   fresh one. **Verified from the signature, the docstring and a refit
   comparison.** The accuracies do *not* rise. The real consequence — `clf` left
   fitted on 40,000 rows, which changes what cells 46 and 54 measure — is not
   stated anywhere. §6.2: the "usual student version" bullet here is invented,
   and wrong.
5. **"11 MB from openml"** (cell 7 comment). The cached artifact is
   `mnist_784.arff.gz`, **15,469,256 bytes** (14.8 MiB). **Verified by `stat`.**
   §1.1.

### Numbers and comparisons

6. **The notebook stores no outputs at all.** All 56 cells have
   `execution_count: null` and zero outputs. **Verified from the JSON.** Every
   prose figure in the file — 90.96%, 96.9%, "two tenths of a point", "five
   points below its own siblings", "under two points", "Half a point" — is
   therefore unreconcilable against the notebook's own data (§1.2), and §9's
   machine check for `⏱` markers, which keys off stored execution time, has
   nothing to run against.
7. **The same gap is given three different sizes in one prompt box.** Cell 35:
   "six points sounds like a lot", "five points from a model with no
   parameters", "makes 96.9% shrink to 5.9". Under the notebook's own figures
   96.9 − 90.96 = **5.94**, so "five points" is wrong. **Verified
   arithmetically** (the underlying 96.9% was not re-derived — training cell).
8. **"Six numbers" under a five-row table.** Cell 55 says "Six numbers, and
   their status matters as much as their value: four measured, one stated by the
   client, one discarded" — the table above it prints exactly five rows, and the
   client's figure (1,000 items per shift) is not one of them. **Verified by
   counting the dict.** §1.2.
9. **An understated check.** Cell 24's box asks for agreement "to three places"
   and the code asserts `atol=1e-4`. The identity is **exact**: 5,421/3 = 1,807
   and 60,000/3 = 20,000, so all three folds score 0.909650 and the mean is
   exactly `1 - base_rate`. **Verified by running the split.** §6.3 — the box
   had a check with an exactly knowable answer available and asked for a weaker
   one.

### Cross-references that do not resolve (§3.3)

10. Cell 6: "the next cell is about that" (the dtype of `y`). The next cell is
the loader itself, cell 7; the dtype cell is **cell 10**. **Verified by
counting.**
11. Cell 18: "it also catches the missing cast **two cells up**". The cast is at
cell 10; the assertion is at cell 19 — **nine cells**, or three code cells,
above. **Verified by counting.**
12. Cell 29: "it leaks the moment it meets cross-validation, which is **the next
cell but one**". `cross_val_score` is at **cell 36**, six cells after the fit at
30. **Verified by counting.**

### Staging and budget

13. **§8.1 — the trap is announced five times before it fires.** Cell 0 ("Cells
marked ⚠ contain a defect on purpose"), cell 44 (the heading, the "⚠ Read before
running", and "it prints a number *better* than the one we just measured"), cell
45 (the box titled "the number you must not report", the constraint "not to be
quoted", and the student bullet "reporting this one"). Lecture 19's **four**
announcements is the defect GUIDELINES cites; this is worse. **Verified by
reading the cells.** Preferred shape: run cell 46 unannounced, write the number
down, *then* open with the ⚠ — which is what the script above does.
14. **§6.1 — 16 code cells, 16 full three-bullet prompt boxes.** Every single
box carries "Left open / The usual student version / How you would catch it".
The budget is five to eight, never more than ten. **Verified by counting
`**Watch this prompt.**` in the JSON.**
15. **§8.3 — "examinable" appears twice in the whole file**, once in a markdown
box and once in a code comment, both on the setup cell, across fourteen
sections. **Verified by string count.**

### State and idempotency (§4)

16. **Cell 10 is not idempotent, and nothing says so.** `y` is rebound from an
object array of strings to `uint8` (§4.1: one name, two types). Re-running the
cell after the cast prints `before: uint8 5` and `y == 5 finds **6313** images`
— the demonstration reverses itself, silently, in the cell that carries the
lecture's headline lesson. **Verified**: 6,313 fives across all 70,000 rows.
17. **`y_test_5` is computed and never used.** Cell 19 builds it from `y_test`;
the name occurs **once** in all sixteen code cells. **Verified by string count.**
Meanwhile cell 51 answers "What touched the test set?" with "Nothing". `X_test`
itself is clean — it occurs in code only inside cell 16 (and in three markdown
cells) — but the test *labels* are read, and a dead binding built from held-out
data is exactly what reviewer question 1 exists to find.

### Instructions a reader alone at home cannot carry out (§7)

18. **No timing anywhere names hardware.** "~30 s", "about 70 seconds", "about
30 seconds", "about four minutes" — §7.1 requires the CPU figure, and a reader
on a throttled Colab runtime cannot tell "slow" from "hung". **Verified by
reading; the timings themselves could not be re-measured** (see below).
19. **Two `⏱` figures are in the wrong place.** Cell 7's "~30 s" lives only in a
code comment, and cell 30's "about 30 s" lives in a code comment *and* in the
box's **Left open** bullet — which is the slot for what the specification omits,
and the bullet says as much ("which the comment gives but the specification does
not"). §9's rule wants the marker in the markdown above the cell. **Verified.**
20. **No restart-and-run-all evidence.** With no stored outputs and a
non-idempotent cell 10, §10.1 cannot be shown to hold. **Not checked** — it
requires running the training cells.

### Checked and clean

- **§5.1 / §5.2** — no markdown line indented ≥ 4 spaces outside a fence, and no
  fence marker indented at all, across all 40 markdown cells. **Verified.**
- **§3.1** — no ```` ```python ```` fence appears in any markdown cell, so there
  is no quoted block that could fail to exist in a code cell. **Verified.**
- The `5,421 → 9.04% → 90.96%` chain is exact and internally consistent.
  **Verified against the data.**
- Cell 19's `assert y_train_5.sum() == 5421` really does catch a skipped cast:
  an object array of strings compared with `5` yields an all-`False` bool array,
  sum 0, no warning. **Verified.**
- `class NeverFires(BaseEstimator)` — no `ClassifierMixin`, no `_estimator_type`
  — runs under scikit-learn 1.7.2 with `cross_val_score(..., cv=StratifiedKFold,
  scoring="accuracy")` and emits **no warning**. **Verified by running it**
  (`fit` is a no-op, so this is not a training cell).
- Cell 51's "X_test appears in the split and not again" is true of the code.
  **Verified by string count.**

### Could not check

Everything that requires fitting on 60,000 rows, because the brief forbids
running training cells: **96.9%**, the 5.9-point gap, "they agree to two tenths
of a point", "one fold of the unscaled version lands five points below its own
siblings", "the two means differ by under two points", "Half a point",
"positive in 15/15", and all five wall-clock figures. A reader following the
script above measures every one of them and writes them down, which is the only
place those numbers can honestly come from.
