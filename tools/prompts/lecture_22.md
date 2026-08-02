# Lecture 22 — *Reusing what someone else learned*, rebuilt by prompting

A script for one person at a Colab keyboard. Twenty-six code cells, in order.
Type the prompt, read what comes back against **Expect**, run the **Assert**.

*Vocabulary, once.* A **logit** is an unnormalised score, before any softmax. A
**leak** is information from the scoring rows reaching the fitting step. A
**smoke test** is a one-line assertion run before the real work, whose expected
answer you know on paper. To **red-team** a notebook is to read someone else's
and go looking for the leak on purpose.

---

## Before you start

**Runtime.** Choose a GPU runtime if Colab offers one. Everything runs on CPU
too; the CPU figures below are measured, and they are the ones that decide
whether you finish.

**Downloads: 423 MB in total, measured.** IMDb tarball 80 MB, then
`distilbert-base-uncased` 256 MB, then `all-MiniLM-L6-v2` 87 MB. Colab caches
none of it across sessions.

**Where the wall clock goes.** Everything below was measured on a 16-core
laptop with `OMP_NUM_THREADS=1` — the value cell 1 sets — except where marked
MPS. A free Colab CPU runtime has 2 vCPUs, so budget roughly three to five
times these figures on the torch cells; that multiplier is an estimate, the
numbers it multiplies are not.

| cell | what | measured here |
|---|---|---|
| 2 | download, extract 100,011 files, read 50,000 | 80 MB + 15 s extract + 3 s read |
| 11 | two GRU runs, 158 steps each | 0.183 s/step → ≈ 1 min for both |
| 14 | DistilBERT download | 256 MB |
| 15, 17 | one scoring pass over 3,000 reviews | 0.48 s per batch of 32 → ≈ 45 s each |
| 16 | fine-tune, 125 steps of batch 16 | 1.07 s/step → ≈ 2.2 min |
| 18 | embed 2,000 reviews | 2.3 s on MPS, ≈ 22 s on CPU |
| 23 | leak experiment, 400 docs × 20 seeds | 21 s |
| 24 | leak experiment, 25,000 docs × 3 seeds | 141 s |
| 26 | grouped split, 6 seeds | 10 s |

**Re-run order (§7.2).** Cell 16 fine-tunes `model` **in place** and does not
re-create it. Running cell 16 twice trains for two epochs, not one, and cell
15's zero-shot number can no longer be reproduced. To redo the fine-tune from
scratch: **cell 14, then 15, then 16, then 17** — in that order, all four. To
change `RANDOM_STATE`: **cell 1, then 2, then 10, then 11** (cells 11 and 26
construct their own models and are safe to repeat on their own).

---

## Cell 1 — setup, and the line that has to be first

**Prompt to type:**

> Setup cell for a teaching notebook. Import numpy, torch, torch.nn and
> matplotlib. Seed torch and numpy with 42. Pick cuda, else mps, else cpu, and
> print the device and the python and torch versions.

**Expect:** ten-odd lines of imports, then `python 3.x.x`, `torch 2.x.x`,
`device cuda` or `mps` or `cpu`. What you will **not** get is the three
`os.environ` lines this notebook needs. Add them yourself, **above the imports**:

```python
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
```

with a comment saying they must precede `import torch`, because nothing else in
the file will ever say so.

**Assert:** none.

**Annotate:** full

* **Left open:** the prompt says nothing about *order*, because on any other
  notebook order does not matter. Here it does. On macOS PyTorch and
  scikit-learn each ship their own OpenMP runtime; with both loaded, cell 21's
  KMeans deadlocks — and a deadlock is not an error message, it is a cell that
  never finishes.
* **The usual student version:** moving the `os.environ` block down to the other
  imports on a later tidy-up. `OMP_NUM_THREADS` is read by OpenMP **at import
  time**: set afterwards it is a string in a dictionary and nothing more. The
  same tidy-up also un-caps the thread count, and that is measurable rather than
  hypothetical — one of cell 23's logistic regressions took **0.01 s** with
  `OMP_NUM_THREADS=1` and **2.33 s** without, back to back on the same 300-row
  fit, because 16 threads fighting over a 300×60,000 sparse matrix is all
  overhead. Cell 23 does forty of those fits: 21 seconds capped, and uncapped I
  killed it at six minutes without a printed line.
* **How you would catch it:** when an environment variable must precede an
  import, put the reason in a comment on the line above it. Then check the value
  actually took: `print(torch.get_num_threads())` should say `1`, not `12`.

---

## Cell 2 — the same corpus, the same split

**Prompt to type:**

> Download the IMDb tarball from
> `https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz` into
> `datasets/` only if it is not already there, and read the train and test
> splits into lists of strings with labels 1 for pos and 0 for neg. Then take a
> 5,000 / 2,000 fit and validation split out of train with a numpy Generator
> seeded at 42, and print the three sizes.

**Expect:** `fit 5,000   val 2,000   test 25,000`. Both halves are exactly
25,000 rows, exactly half positive. The reader is rebuilding lecture 21's split
from the seed rather than inheriting it, which is the only thing that makes the
two lectures comparable.

**Assert:** `len(train_x) == len(test_x) == 25_000` and
`set(fit_i).isdisjoint(val_i)`.

**⏱** 80 MB download, then **15 s** to extract 100,011 files and **3 s** to read
50,000 of them, measured on an SSD; on Colab's disk budget one to three minutes.
Re-running is instant because of the `if not DATA.is_dir()` guard — check that
it is there.

**Annotate:** short

---

## Cell 3 — softmax, and the shift that changes nothing

**Prompt to type:**

> Take `z = [1000, 1001, 1002]` as float32 and print `exp(z)`, then
> `exp(z)/exp(z).sum()`, then the same thing with `z.max()` subtracted first.
> Wrap the first two in `np.errstate(over="ignore")` so the overflow is
> deliberate rather than noise, and print the x at which `exp` overflows float32.

**Expect:** `exp(z)` is `[inf inf inf]`, the ratio is `[nan nan nan]`, and with
the max subtracted it is `[0.09003057 0.24472848 0.66524094]`. The overflow
threshold prints as **88.72** — `log(3.4028235e+38)`, and much smaller than
anyone guesses.

**Assert:** none.

**Annotate:** short

---

## Cell 4 — the invariance itself, on values that do not overflow

**Prompt to type:**

> Now show the shift invariance on values that do not overflow: `a = [1,2,3]`
> float32, softmax of `a` and of `a + 50`, print both and assert the largest
> difference is under 1e-6.

**Expect:** both rows print `[0.09003057 0.24472848 0.66524094]` — the same three
numbers as cell 3's stable line, which is the point. Largest difference
**5.96e-08**, i.e. one float32 ulp.

**Assert:** `np.abs(p1 - p2).max() < 1e-6`.

**Annotate:** short

*Two claims, two examples.* Cell 3 is about float32, cell 4 is about algebra.
Demonstrating both on the overflowing input would let you conclude that softmax
is not shift-invariant, which is false.

---

## Cell 5 — the gradient, derived and verified

**Prompt to type:**

> Check that the derivative of cross-entropy with respect to the logits is
> `p − y`. Random 7×5 float64 logits with `requires_grad`, random targets, use
> `nn.CrossEntropyLoss(reduction="sum")`, and compare `z.grad` against
> `softmax(z) − onehot(y)`. Assert they agree to 1e-10 and that each row of the
> gradient sums to zero.

**Expect:** `|autograd - (p - y)| = 5.551e-17`; row sums come out at
**2.22e-16**, which is zero in float64. Largest single component 0.961, inside
[−1, 1] as it must be. `reduction="sum"` is in the prompt on purpose: the
default is `reduction="mean"`, and with the default every gradient is divided by
7 and the 1e-10 assertion fails for a reason that has nothing to do with the
mathematics.

**Assert:** `err < 1e-10`, `analytic.sum(1).abs().max() < 1e-10`,
`analytic.abs().max() <= 1.0`.

**Annotate:** short

*The row-sum assertion is the interesting one:* the gradient has no component
along the all-ones direction, which is the derivative form of cell 4's shift
invariance.

---

## Cell 6 — why the loss consumes logits

**Prompt to type:**

> Compare two ways of getting cross-entropy in float32, both scored against
> float64: the naive one, softmax then log; and `-(z_c - logsumexp(z))`. Sweep
> how wrong the row is rather than the scale of the logits — put the true class
> 1, 5, 10, 20, 40, 60, 80, 90, 100 and 110 nats below the largest logit, 2,000
> rows of 10 classes at each level — and print the fraction of non-finite
> results and the median relative error for both forms at each level.

**Expect:** ten lines. Both forms sit at a median relative error near **2e-8**
for losses 1 through 90 — indistinguishable. At loss 100 the naive form jumps to
**2.57e-04** while the stable one stays at 2.17e-08. At loss 110 the naive form
is **100% non-finite** and its median error prints as `nan`. The stable form is
0.0% non-finite at every level.

**Assert:** `rows[0][3] == 0.0 and rows[-1][3] == 0.0` — the stable form never
fails, at either end.

**Annotate:** short

*Why sweep the loss and not the logit scale:* the naive form has to represent
e^(−loss) in float32, so the loss is the quantity the failure depends on.
Sweeping the standard deviation instead puts most rows near zero loss where both
forms agree trivially, and you conclude the whole argument is folklore.

---

## Cell 7 — the two failure modes, drawn

**Prompt to type:**

> Two panels from that sweep: median relative error on a log y axis on the left,
> percentage of rows returning inf or nan on the right, naive and combined on
> both, x axis the true loss of the row in nats.

**Expect:** left panel — the two curves lie on top of each other until 90, then
the naive one lifts four orders of magnitude at 100; right panel — the naive
line is flat at 0 and steps to 100 between 100 and 110. **The naive curve on the
left panel ends at 100 and no marker is drawn at 110**: its value there is `nan`,
matplotlib drops non-finite points from a line silently, and `np.maximum(x, 1e-9)`
does not repair a `nan`. If you want the point, plot the non-finite rate
separately — which is what the right panel is for.

**Assert:** none.

**Annotate:** short

---

## Cell 8 — two rows, small enough to check by hand

**Prompt to type:**

> Two rows by hand, target class 2: `[100, 0, -100]` and `[0, 0, -100]`. For
> each print the naive float32 value, the `-(z_c - logsumexp)` value, the float64
> value, and what `nn.CrossEntropyLoss` returns. Assert the first is not finite
> and that the naive value on the second is off by more than 1e-3.

**Expect:**

| row | naive | combined | float64 | PyTorch |
|---|---|---|---|---|
| `[100, 0, -100]` | `inf` | 200.0 | 200.0 | 200.0 |
| `[0, 0, -100]` | 100.6399 | 100.69314575 | 100.69314718 | 100.69314575 |

The second row's `p(true class)` prints as **1.9618e-44** — a float32 subnormal,
which is where the 0.0533 of missing accuracy went. PyTorch agrees with the
combined form to the last digit, which is evidence about what the library does
internally.

**Assert:** `not np.isfinite(n1)`; `abs(n2 - e2) > 1e-3`; `abs(s2 - e2) < 1e-4`.

**Annotate:** short

*The loud failure stops your run. The quiet one — finite, plausible, wrong in
the second decimal place — is the one that ships.*

---

## Cell 9 — cross-entropy and KL divergence

**Prompt to type:**

> Show that cross-entropy equals entropy plus KL divergence. Take a random
> Dirichlet q over 5 classes and two targets: a one-hot on class 2, and the same
> label-smoothed to 0.9 with 0.025 elsewhere. Print H, KL, H+KL and CE for both
> and assert they agree to 1e-12. Mask out the zeros before taking logs.

**Expect:**

```
one-hot         H -0.0000   KL 1.1447   H+KL 1.1447   CE 1.1447
label-smoothed  H 0.4637    KL 0.8178   H+KL 1.2815   CE 1.2815
```

Yes, the one-hot entropy prints with a minus sign: `-(0 * log 1)` is negative
zero in IEEE 754. `H(onehot) == 0.0` is still `True`, so the assertion passes;
if that bothers you, print `H(p) + 0.0`.

**Assert:** `abs(CE - H - KL) < 1e-12` for both targets, and `H(onehot) == 0.0`.

**Annotate:** short

*The identity holds up to the entropy of the target, and that term is zero only
for hard labels — which is exactly why both targets are in the cell.*

---

## Cell 10 — the previous lecture's model, rebuilt

**Prompt to type:**

> Rebuild lecture 21's classifier here, with the fixes: a regex word tokeniser,
> a 20,000-word vocabulary counted on the fit split only with 0 for pad and 1
> for unknown, ids padded to 192, `nn.Embedding(vocab, 128, padding_idx=0)`, a
> bidirectional GRU with 64 hidden units over a **packed** sequence, and a
> linear head to 2 logits. Encode fit and val and print the batch shape.

**Expect:** `fit batch (5000, 192)`. Under the hood: 42,370 distinct tokens in
the fit split, of which the top 19,998 get ids 2…19,999. **42.9% of the fit
reviews are longer than 192 tokens** and are truncated — that is a modelling
decision the prompt made silently and it is worth knowing you made it.

**Assert:** `Xf.shape == (N_FIT, MAXLEN)`.

**Annotate:** short

*Carry the fixes forward with the architecture.* `padding_idx=0` and the packing
are both lessons from the previous lecture, and neither is visible in a shape
check.

---

## Cell 11 — an improvement nobody would refuse

**Prompt to type:**

> The model returns raw numbers. Add a softmax to the output so it returns
> probabilities, and keep the training loop working. Train the original and the
> new one from the same seed — two epochs, batch 64, Adam at 1e-3 — and print
> the epoch loss and the validation accuracy for each.

**Expect:** four lines, two per model, of the form
`epoch 2: loss 0.xxxx  val 0.xxxx`. Both losses fall over the two epochs. Both
models train without a warning, an exception or a shape error.

**Write down all four numbers before you run cell 12.** Then decide, from those
four numbers alone, whether you would have shipped this.

**Assert:** none.

**⏱** 158 optimiser steps per run, 0.183 s each measured → **about a minute for
both runs** on this CPU, a few seconds on a GPU. On a 2-vCPU Colab CPU runtime
budget five minutes.

**Annotate:** short

---

## Cell 12 — look at the loss column first

**Prompt to type:**

> Print both final losses and both validation accuracies side by side, the
> difference in points, and `-log(e/(e+1))` beside them.

**Expect:** the accuracy gap is a few points — a number that could mean anything.
The loss column is the one that identifies the cause. `-log(e/(e+1))` prints as
**0.3133**, and the second model's loss **cannot** go below it, ever, at any
learning rate, for any amount of training.

**Assert:** none in the notebook. Add one, because it holds by algebra rather
than by luck: `assert loss_bad[-1] >= 0.3132`.

**Annotate:** full

* **Left open:** cell 11's prompt never said what `forward` is *for*. It returns
  the number the loss consumes and the number a user reads, and those are two
  different objects. Nothing in "return probabilities" says which one moved.
* **The usual student version:** exactly the prompt in cell 11, which is a
  request no reviewer would refuse. `nn.CrossEntropyLoss` documents its input as
  *unnormalised logits* and applies its own `log_softmax`; feeding it
  probabilities gives a softmax of a softmax. On two classes the inner softmax
  lands in [0, 1], so the outer one can never exceed e/(e+1) = **0.7311**, and
  the loss can never drop below **0.3133** — a maximally confident row scores
  exactly 0.31326, which I checked. Nothing raises. The loss still goes down.
* **How you would catch it:** when a loss plateaus at a value you can compute in
  closed form, compute it — a match names the bug rather than merely detecting
  it. But do not rely on seeing the plateau here: two epochs on 5,000 reviews may
  stop the *correct* model well above 0.3133 too, in which case the two columns
  merely look disappointing. The check that does not depend on how long you
  trained is on the output, not the loss: a row of logits does not sum to 1.
  `out.sum(1)` on the correct model averages −0.016 over a batch; on the
  softmaxed one it is 1.000000 to seven decimals. One line, no training needed.

---

## Cell 13 — the assertion that would have caught it

**Prompt to type:**

> Before any training: build a fresh untrained model, run it on 512 of the fit
> reviews, and assert its cross-entropy is within 0.05 of log 2, with a message
> saying the head or the targets are wrong.

**Expect:** `untrained loss 0.6972   log 2 = 0.6931`. Across four seeds I
measured 0.6944, 0.6972, 0.7036 and 0.7081 — the band is 0.05 wide because the
untrained logits have a standard deviation of about 0.21, not because 0.05 is a
round number.

**Assert:** `abs(l0 - np.log(2)) < 0.05`.

**Annotate:** full

* **Left open:** *when* it runs. An untrained two-class model on balanced data
  has no information, so it must score the entropy of a coin flip; run before
  training it costs nothing and pins the head width, the target dtype and the
  label encoding at once. Run after training it says nothing at all.
* **The usual student version:** no smoke test whatsoever, so the first evidence
  that something is wrong is a slightly disappointing accuracy forty minutes
  later, by which time five other things have changed too.
* **How you would catch it — and what this assertion does *not* catch.** I ran
  it against cell 11's softmaxed model: **it passes, at 0.6919.** It has to.
  Before training, the logits are tiny, so softmax of them is very near
  [0.5, 0.5], and the cross-entropy of a near-uniform row is log 2 whether or not
  you softmaxed it first — the double softmax pushes the loss *closer* to log 2,
  not further. So this assertion catches a wrong head size, transposed targets
  and mislabelled data, and it is worth every second it costs; it does not catch
  today's defect. For that one add the row-sum check from cell 12. Two
  assertions, three lines, and between them they cover the four bugs above.

---

## Cell 14 — borrow the whole model

**Prompt to type:**

> Load `distilbert-base-uncased` with a 2-class classification head and its
> tokenizer, encode the first 2,000 fit reviews at max_length 192 with padding
> to max_length, and take 3,000 test reviews to score everything on. Print the
> parameter count.

**Expect:** `66,955,010 parameters`, and a red warning you should read rather
than scroll past:

```
Some weights of DistilBertForSequenceClassification were not initialized from
the model checkpoint at distilbert-base-uncased and are newly initialized:
['classifier.bias', 'classifier.weight', 'pre_classifier.bias',
'pre_classifier.weight']
```

That warning is the subject of cell 15: the body is pretrained, the head is
noise. Encoded shape `(2000, 192)`. The scoring subset prints **50.1% positive**
— but only if you shuffled.

**Assert:** `ids.shape == (N_FT, MAXLEN)` and
`abs(score_y.mean() - 0.5) < 0.03`.

**⏱** 256 MB download. The tokenising itself is 0.6 s for 2,000 reviews — the
fast tokenizer is not where your time goes, the download is.

**Annotate:** full

* **Left open:** *which* 3,000 test reviews. The prompt says "take 3,000", the
  obvious code is `test_x[:3000]`, and the prompt gives no reason to prefer
  anything else.
* **The usual student version:** `test_x[:3000]`. The IMDb loader in cell 2
  reads `pos` before `neg`, so the test list is 12,500 positives followed by
  12,500 negatives. I checked: `test_y[:3000].mean()` is **1.0**. Every review
  in that prefix is positive, a model that answers "positive" always scores
  100%, and what you have measured and called accuracy is recall.
* **How you would catch it:** one line, on every subset you ever score on —
  `assert abs(y.mean() - 0.5) < 0.03`. It costs nothing and it catches the most
  embarrassing result available in this notebook. The shuffled subset lands at
  0.5007, which is inside the band by a factor of forty.

---

## Cell 15 — the floor, before any training

**Prompt to type:**

> Score that model on the 3,000 before any fine-tuning, in batches of 32, and
> print the accuracy.

**Expect:** something near 50%. The body is informative and the head is the noise
the warning in cell 14 named, so this number measures the pair, not the
pretrained model — a pretrained encoder with a fresh head is **not** a zero-shot
classifier. Whatever it prints is the floor the fine-tune has to beat, and it is
worth having on the page rather than assumed.

**Assert:** none.

**⏱** 94 batches, 0.48 s each measured → **about 45 s** on this CPU.

**Annotate:** short

---

## Cell 16 — fine-tune, at a hundredth of the learning rate

**Prompt to type:**

> Fine-tune it for one epoch on the 2,000 encoded reviews, batch 16, AdamW at
> 2e-5, feeding logits to `nn.CrossEntropyLoss`. Print the loss every 25 steps
> with the elapsed seconds, and the total at the end.

**Expect:** six progress lines (steps 0, 25, 50, 75, 100, 125) starting near
0.69 and falling. `2e-5` is about a hundredth of the `1e-3` used for the GRU,
and it is in the prompt on purpose.

**Assert:** none here; the floor is asserted in cell 17.

**⏱** 125 steps at 1.07 s each measured → **about 2.2 minutes** on this CPU,
roughly 10 minutes on a 2-vCPU Colab CPU runtime, well under a minute on a GPU.

**Annotate:** full

* **Left open:** nothing about *why* the rate is small — so the next person to
  edit the cell has no reason not to raise it. The body already encodes
  something; a large step overwrites it before the head has learned anything to
  overwrite it for.
* **The usual student version:** dropping the rate from the prompt, or reusing
  `LR` from cell 10 because it worked. `torch.optim.AdamW` defaults to
  `lr=1e-3` — I checked the signature — which is the same 1e-3 the GRU used, so
  both roads lead to a rate fifty times too large. What you get is not an error;
  it is a loss that sits near 0.69 and stops moving, because the pretrained
  weights are gone.
* **How you would catch it:** print the loss *during* training, not only at the
  end. A fine-tune going wrong is legible in the first twenty steps and costs two
  minutes to confirm at the end. And note that this cell mutates `model` in
  place: to run it again honestly you must re-run cell 14 first, then 15.

---

## Cell 17 — the comparison, on the same rows

**Prompt to type:**

> Score the fine-tuned DistilBERT and the from-scratch GRU on the same 3,000
> shuffled test reviews. Print those two with the always-one-class baseline and
> the untrained-head number, then the gain in points and the fraction of errors
> removed.

**Expect:** four lines. The baseline prints **50.0%** exactly — the IMDb test set
is 12,500 positive and 12,500 negative, so `max(mean, 1-mean)` is exactly 0.5.
The GRU is scored here on the **same 3,000 shuffled test reviews** as DistilBERT,
not on its own validation split; two accuracies on two different subsets are not
a comparison. Report errors removed as well as points: going from 85% to 90%
removes a third of the mistakes, and the desk counts mistakes.

**Assert:** `ft_acc > 0.75`, with a message naming the learning rate. This is a
floor, not the headline: one epoch on 2,000 reviews should land far above the
majority class, and if it does not, cell 16 is broken rather than disappointing.

**⏱** two scoring passes, about 45 s each on CPU, plus a few seconds to re-encode
the 3,000 reviews for the GRU.

**Annotate:** short

---

## Cell 18 — one vector per review

**Prompt to type:**

> Load `sentence-transformers/all-MiniLM-L6-v2`, and write a function that
> embeds a list of texts: truncate at 256 tokens, batch 64, mean-pool the last
> hidden state, L2-normalise, return a numpy array. Run it on 2,000 negative
> test reviews and assert the norms are 1.

**Expect:** `2,000 reviews, 384 dimensions, on the unit sphere`. On the unit
sphere the dot product *is* the cosine, which makes cell 19 a single matrix
product.

**Assert:** `V.shape[0] == len(neg)` and
`np.allclose(np.linalg.norm(V, axis=1), 1.0, atol=1e-5)`.

**⏱** 2.3 s on MPS, **22 s** on this CPU, a minute or two on Colab CPU.

**Annotate:** full

* **Left open:** "mean-pool" — over *what*. The prompt does not say, the shapes
  are identical either way, and every assertion in the cell passes either way.
* **The usual student version:** `out.last_hidden_state.mean(1)`. That averages
  over the padding as well as the words. The model's own card ships a
  `mean_pooling` helper that multiplies by the attention mask first, and it ships
  it for this reason. I ran both poolings over the same 512 reviews: the median
  cosine between them is **0.9006** and the worst is **0.3371** — a third of a
  vector, on a review that shared its batch with a much longer one. Because
  `padding=True` pads per batch, the size of the damage depends on which reviews
  happen to be batched together, so it is not even stable between runs of the
  same corpus in a different order.
* **How you would catch it:** embed one short text alone, then again in a batch
  with a very long one, and check the two vectors are equal. Masked pooling gives
  the same vector; the plain mean does not. That is a two-line test and it is the
  same padding bug as lecture 21's, in a new costume.

---

## Cell 19 — search the corpus

**Prompt to type:**

> Three complaint-shaped queries — wooden acting, unintelligible sound mix, far
> too long — embedded with the same function, dot them against the 2,000, and
> print the nearest review to each with its cosine.

**Expect:** a `(3, 2000)` similarity matrix, and three hits with cosines of
**0.555**, **0.534** and **0.424**. Print the cosine beside every result: the
median similarity over all 6,000 pairs is 0.212 and the maximum anywhere is
0.555, so a "nearest neighbour" here is nearer than average and not close in any
absolute sense. A ranking on its own hides that.

**Assert:** `sims.shape == (len(queries), len(neg))`.

**Annotate:** short

*Worth measuring on your own queries before you deploy it:* cosine similarity has
no notion of negation, so *the sound was perfect* and *the sound was not perfect*
sit close together. Three hand-written queries will never show you that.

---

## Cell 20 — what keyword search returns instead

**Prompt to type:**

> Same three queries through a plain tf-idf vectoriser fitted on the same 2,000
> reviews, and print how many of the top three overlap with the semantic top
> three for each query.

**Expect:** `0/3`, `0/3`, `1/3`. Two of the three queries return a completely
disjoint set of documents, and the tf-idf top similarities are 0.194, 0.153 and
0.134 — far lower than the embedding's, because a complaint rarely reuses the
desk's vocabulary. Compare the top-*k* sets rather than the top-1: a single
disagreement could be a tie.

**Assert:** none.

**Annotate:** short

*Zero overlap and total overlap are both worth knowing.* Semantic search is
better at this and worse at exact-match retrieval, which is what the desk asks
for next.

---

## Cell 21 — grouping the complaints

**Prompt to type:**

> Cluster the 2,000 vectors with k-means for k in 3, 4, 5, 6, 8, 10, score each
> with silhouette on a sample of 1,500 with a fixed random_state, print the
> scores, then refit at the best k.

**Expect:**

```
k= 3  silhouette 0.0130     k= 6  silhouette 0.0163
k= 4  silhouette 0.0118     k= 8  silhouette 0.0167
k= 5  silhouette 0.0153     k=10  silhouette 0.0144
best k = 8
```

**Read the size of those numbers, not just their ranking.** A silhouette of 0.017
is not weak structure, it is no structure: these reviews do not fall into
separated groups, they fill a region. And the winner is not stable — I re-scored
the same six clusterings under ten different silhouette sample seeds and the
argmax was k=8 eight times, k=6 once and k=5 once. The subsampling standard
deviation is 0.0005 and the gap between k=8 and k=6 is 0.0004. `sample_size=1500`
is seeded, so your run reproduces; it is still an approximation, and here it is
the approximation that picks k.

**Assert:** `len(np.unique(km.labels_)) == best_k`.

**⏱** about 1.5 s for the whole sweep.

**Annotate:** short

*k-means on unit vectors is spherical k-means in all but name.* That is
appropriate here and worth knowing you chose it.

---

## Cell 22 — name each group by what makes it different

**Prompt to type:**

> Name each cluster by the terms whose mean tf-idf weight inside it most exceeds
> their mean outside it — min_df 5, max_df 0.4, English stop words, 1–2 grams —
> five terms per cluster, printed with the cluster size, largest first.

**Expect:** eight rows over a 7,684-term vocabulary, sizes 438, 373, 346, 255,
223, 219, 106 and 40:

```
  438  bad, worst, terrible, seen, awful
  373  flynn, actor, ed, bishop, gein
  346  horror, zombie, cave, zombies, scary
  ...
   40  christian, church, ufo, believe, religious
```

Note what came back. One genuine theme (horror), one cluster that is just the
word *bad*, and several named after proper nouns — the desk asked for recurring
complaints and got recurring *films*. Ranking by lift rather than by frequency is
what stopped every cluster being called "film, movie, one, like"; it does not
make a cluster a theme. Print the size beside the terms: a theme covering 40 of
2,000 reviews is not one the desk needs to hear about.

**Assert:** none.

**Annotate:** short

---

## Cell 23 — build me a pipeline

**Prompt to type:**

> Build a scikit-learn pipeline that vectorises the reviews with tf-idf and
> classifies them with logistic regression, and report the test accuracy.

Read what came back before you run it. Then, in the same cell:

> Now measure it: 400 documents sampled from the full 50,000, 20 seeds. Fit the
> vectoriser the way you just wrote it, and also the other way — split first,
> fit on the training half only, transform the test half — and print the mean
> accuracy difference, its standard deviation, and how many seeds the first way
> wins.

**Expect:**
`400 docs, 20 seeds:  +0.30 points (sd 2.53), leak wins on 10/20`.

Read that line honestly. The mean says the leak buys three tenths of a point;
the standard deviation says the seed-to-seed spread is eight times that, the win
count is exactly ten out of twenty, and the standard error on the mean is 0.57.
**This is not a measured effect**, and I checked how unmeasured it is: re-run at
60 seeds and the mean is **−0.32** — the leak *loses*. The sign is not stable.

What *is* stable is deterministic and prints in the same cell: the leaky
vectoriser sees a vocabulary of **60,000** columns against the honest one's
**54,074**, on identical documents. Roughly six thousand of those columns exist
because a test document used them.

**Assert:** `Z.shape[1] >= Ztr.shape[1]`.

**⏱** 21 s with `OMP_NUM_THREADS=1`. Uncapped I killed it at six minutes with
nothing printed. If this cell is slow, cell 1 is wrong.

**Annotate:** full

* **Left open:** *which* documents get vectorised. "Vectorise the reviews and
  classify them" reads as one operation on one corpus, and the split is a detail
  inside the classifier — which is exactly backwards.
* **The usual student version:** `fit_transform` on everything, then
  `train_test_split` on the matrix. `TfidfVectorizer.fit` learns two things from
  every document you hand it: which columns exist, and the inverse document
  frequency weighting each one. Both are fitted quantities and both are supposed
  to come from the training half only.
* **How you would catch it:** twenty seeds, and then look at the spread before
  the mean. A result inside its own noise is not a result, and this one is —
  which is why the argument for the rule has to be the structural one (columns
  that exist because of a test document) rather than the accuracy one. Argue the
  mechanism, and the rule survives the day your dataset does not cooperate.

---

## Cell 24 — the same leak at full size

**Prompt to type:**

> Run the same experiment at 25,000 documents, three seeds, and strip-plot the
> individual seeds at both sizes on one axis, with a tick at each mean.

**Expect:** `25,000 docs, 3 seeds: +0.10 points (sd 0.01)`. Seed by seed:
0.9024 vs 0.9013, 0.9002 vs 0.8992, 0.9083 vs 0.9075. Three tenths of a point at
400 documents against one tenth here — but read it beside cell 23's standard
deviation of 2.53 before calling that a trend, because the full-size sd is 0.01.
**The only size at which this effect is cleanly measurable is the largest one.**

One thing the plot will not tell you: `max_features=60_000` binds at this size,
so both arms have exactly 60,000 columns and the "columns that exist because a
test document used them" mechanism is gone. What is left is the idf weighting,
averaged over 25,000 draws, where removing a quarter of the documents moves
almost nothing. That is the real reason the effect shrinks.

Plot the individual seeds rather than error bars: three points at one size and
twenty at the other would imply a precision neither has.

**Assert:** none.

**⏱ 141 s**, measured, with `OMP_NUM_THREADS=1`. Not "about a minute".

**Annotate:** short

**The decision rule, and it does not rest on the effect size:** fit the
vectoriser inside the pipeline, always. Not because the damage is always large —
today it is a tenth of a point and unmeasurable at 400 documents — but because
the mechanism scales with the reciprocal of the corpus size, and the corpus is
smallest exactly when the project starts.

---

## Cell 25 — the leak no pipeline protects you from

**Prompt to type:**

> Lower-case and whitespace-normalise every review, and count how many test
> reviews appear verbatim in training, and how many duplicates there are inside
> training.

**Expect:** `test reviews also present in training: 123` and
`duplicate reviews within training: 96`. Three lines, and you know rather than
assume.

Two things to notice. The normalisation buys nothing here: matching the raw
strings finds the *same* 123 and 96 — I ran both. And 123 is not zero. IMDb is
usually described as deduplicated, and it is *nearly* so — 0.49% of the test set
is a verbatim copy of something in training. Small enough that the cost of
duplication cannot be measured on this corpus, which is why cell 26 builds one
where it can.

**Assert:** none.

**⏱** about 4 s to normalise 50,000 documents.

**Annotate:** short

---

## Cell 26 — rows split wrongly, against objects fitted wrongly

**Prompt to type:**

> Build a corpus that is not clean: 1,500 unique reviews with a third of them
> added a second time, keeping a group id for each original. Score a correct
> pipeline on it twice — once with `train_test_split`, once with
> `GroupShuffleSplit` on the group ids — six seeds, and print both accuracies,
> the gap, and what fraction of test rows had a copy of themselves in training.

**Expect:**

```
random split  0.9081 (sd 0.0154)
grouped split 0.8239 (sd 0.0349)
the duplicate leak is worth +8.42 points
34% of test rows had a copy of themselves in training
```

**Nothing was fitted on the test set in either arm.** The vectoriser is fitted
inside the split both times; the estimator, the seeds and the hyperparameters are
identical. The whole difference is which rows the split happened to separate.

**And now the comparison that matters, on matched corpus sizes.** The duplicated
corpus is 1,950 rows. I re-ran cell 23's experiment at **1,950 documents, 20
seeds**: the vectoriser leak is worth **+0.16 points (sd 0.52)**. Same row count,
same estimator, same vectoriser settings, both measured over 20 and 6 seeds
respectively:

| what was wrong | corpus | accuracy it buys |
|---|---|---|
| the object was fitted wrongly | 1,950 rows | **+0.16** points |
| the rows were split wrongly | 1,950 rows | **+8.42** points |

Fifty times more damage from the split than from the leak everyone teaches. A
correct pipeline is not sufficient, and no amount of care about `fit` versus
`transform` touches this one — it is fixed by grouping.

**Assert:** `set(g[tr2]).isdisjoint(g[te2])` — no group straddles the grouped
split.

**⏱** 10 s for all six seeds.

**Annotate:** full

* **Left open:** what a "row" is. The corpus has 1,950 rows and 1,500 entries,
  and every splitter in scikit-learn splits rows.
* **The usual student version:** `train_test_split(X, y, test_size=0.25)` and
  nothing else, because it is correct for every dataset where a row is an
  independent draw. It shuffles by default (`shuffle=True`), it has no `groups`
  argument at all, and a duplicated entry is simply two rows that may land on
  opposite sides. Here 34% of test rows had their twin in training.
* **How you would catch it:** report the twin fraction — `np.isin(g[te], g[tr]).mean()`
  — rather than the accuracy. That number explains the gap, and it is the one to
  go looking for in a corpus you did not build. One honest caveat when you quote
  the 8.42: `GroupShuffleSplit` cannot hit exactly 25% of rows, so its test sets
  ran 477–493 rows against the random split's fixed 488. Six seeds is what makes
  that difference small compared to the effect, not something to skip.

---

## Closing section — red-team a peer's notebook

Not a code cell. Nine questions, ten minutes, swap with the team beside you:

1. What touched the test set?
2. What was fitted, and on what? (`fit` and `transform` are different verbs)
3. What is the shape here?
4. What was dropped — rows, columns, NaNs? Count them.
5. What is the default I did not ask for?
6. Was the tokenizer or vectoriser fitted before the split?
7. Does `forward` return logits, or probabilities?
8. Is padding excluded from every pooling and every mean?
9. Are there duplicate documents across the split? Count them.

Report what you **found**, not what you would have done differently.

---

## Annotation budget

Twenty-six cells, **eight** full annotations: cells 1, 12, 13, 14, 16, 18, 23
and 26. The other eighteen carry the specification only. The current notebook
carries the full three-bullet block on all twenty-six, which is the defect
`GUIDELINES.md` §6.1 exists to repair.

---

# Defects found in the current notebook

`notebooks/lecture-22.ipynb`, against `GUIDELINES.md`. Everything below was
checked with `python3` against the notebook's own data unless it says otherwise;
the four unverifiable items are listed last and say why.

### Checked, and wrong

1. **§6.1 — annotation budget: 26 of 26.** Every prompt box carries the full
   three-bullet block. `python3 tools/check_notebooks.py 22` fails the notebook
   on exactly this: *"26 full annotations, budget is 10"*. This is the largest
   count in the course and the notebook is 72 cells long, so the reader meets
   the twelfth full annotation around cell 30 — where the audit says all three
   readers stopped reading, and where this notebook's own defect sits.

2. **§3.2 — the corrected specification does not do what it claims.** Cell 33
   and cell 34's annotation both offer *"assert that an untrained model's loss
   on balanced classes is within 0.05 of log 2"* as the fix for the double
   softmax, and cell 34 says it catches *"a wrong head size, transposed targets,
   **a double softmax**, and labels that are not what you think they are"*. I ran
   the assertion against the softmaxed model: **it passes, at 0.6919**, against
   log 2 = 0.6931. It cannot fail — before training the logits are near zero, so
   the extra softmax produces a near-uniform row whose cross-entropy is log 2,
   and it lands *closer* to the target than the correct model's 0.6972. The
   check offered for independent verification does not verify the thing it is
   offered for. (The check that does: the softmaxed model's `out.sum(1)` is
   1.000000; the correct model's averages −0.016.)

3. **§7.1 — no CPU figure anywhere, and the two that exist say "on a GPU".**
   The string "CPU" appears **zero** times in the notebook; "GPU" appears twice,
   both attached to timings (cell 30: *"about 1-3 minutes … on a GPU"*; cell 36:
   *"about 1–3 minutes to fine-tune, on a GPU"*). Measured on a 16-core CPU with
   the notebook's own `OMP_NUM_THREADS=1`: the fine-tune is 1.07 s/step × 125
   steps = **2.2 minutes**, each 3,000-review scoring pass is **45 s**, and there
   are two of them. On a 2-vCPU Colab CPU runtime those become roughly ten and
   three minutes. This is the exact failure §7.1 was written for.

4. **§1.1 / §2.4 — the 400-document leak result is inside its own noise, and the
   notebook's conclusion rests on it.** Cell 61 prints
   `+0.30 points (sd 2.53), leak wins on 10/20` — an effect eight times smaller
   than its own spread, winning on exactly half the seeds, standard error 0.57.
   Cell 64 nevertheless concludes *"At 400 the leaky vocabulary has columns that
   exist because a test document used them"* and generalises to *"it scales with
   the reciprocal of your corpus size"*. Re-running the same code at **60 seeds
   gives −0.32 points** — the sign flips. The prompt box's own bullet says a
   single split proves nothing; the section's conclusion then leans on twenty.
   Fix per §2.2: argue the mechanism (60,000 leaky columns against 54,074 honest
   ones, deterministic) rather than the accuracy.

5. **§1.1 — the mechanism the closing paragraph names is absent at full size.**
   Cell 64: *"At 25,000 documents the inverse document frequencies are an average
   over 25,000 draws"* — true — implying that at 25,000 the vocabulary effect is
   merely diluted. It is not diluted, it is **gone**: `max_features=60_000` binds
   in both arms, and I measured both vocabularies at exactly 60,000 columns for
   all three seeds. The contrast the section draws is with a mechanism its own
   parameters have switched off.

6. **§3.1-adjacent — an assertion whose message contradicts what it tests.**
   Cell 61: `assert Z.shape[1] >= Ztr.shape[1], "the leaky vocabulary must be
   larger"`. At 400 documents it is larger (60,000 vs 54,074); at 25,000 it is
   **equal**, so the message is false exactly where the notebook wants the reader
   to reason about vocabularies.

7. **§7.1 / §1.1 — a comment wrong in both of its numbers.** Cell 63:
   `# ⏱ about a minute: five seeds at the full corpus size.` The code on the next
   line is `leak_experiment(25_000, 3)` — **three** seeds, not five — and it took
   **141 s** measured, not about a minute.

8. **§3.3 — "Note line 4 below" points at the wrong line.** Cell 45: *"Note line
   4 below: the mean is over the real tokens, using the attention mask."* Line 4
   of the next code cell (47) is
   `stk = AutoTokenizer.from_pretrained(MINILM)`. The masked mean is on **line
   15**. This is lecture 19's *"ten cells earlier was fifteen"* defect, one file
   later.

9. **§3.3 — three "next cell" references that are not the next cell.** Cell 27
   (*"added in the NEXT cell"*) → the code is two cells later; cell 46 (*"makes
   the next cell one matrix product"*) → three cells later; cell 65 (*"the next
   cell builds a corpus that is not clean"*) → three cells later. Each points at
   the next *code* cell with a prompt box in between. Individually venial;
   together they train the literal reader to stop trusting the phrase, which is
   what the audit recorded.

10. **§8.1 — the primary defect is announced four times before it runs.** Header
    cell 0 (*"neither of today's two defects raises an exception"*), heading cell
    26 (*"⚠ Read before running — an assistant improves the model"*), the body of
    cell 26 which gives the whole diagnosis including e/(e+1) ≈ 0.731 and the
    0.313 floor, and the prompt box at cell 29 titled *"⚠ two runs, one extra
    softmax"* whose three bullets state the answer again. By the time the reader
    reaches `if double_softmax:` there is nothing left to catch. Same count for
    §10's leak: cells 0, 59, 60 and the code comment *"# what the assistant
    wrote: fit on everything, then split"*. §8's preferred shape — run it
    unannounced, write the number down, *then* open the next section with the ⚠ —
    is available here at the cost of moving two paragraphs.

11. **§8.3 — "examinable" appears once in the whole notebook**, in a code comment
    (*"Not examinable, and only needed on macOS"*). Eleven sections, one marker.

12. **§4.1 — six names rebound across cells.** Verified by AST walk over the
    top-level assignments of every code cell:
    - `z` — float32 numpy array (cell 8) → `torch.randn` tensor with
      `requires_grad` (cell 13);
    - `q` — a Dirichlet probability vector (cell 25) → a **query string**, as a
      loop variable (cells 50 and 52). This is the `target` clobbering of lecture
      19, unremarked;
    - `g` — the array of loss levels used by the plot (cell 19) → the per-seed
      accuracy gaps, as a loop variable (cell 63);
    - `p` and `onehot` — torch tensors (cell 13) → numpy arrays (cell 25);
    - `m` — a float32 scalar `z.max()` (cell 8) → a boolean cluster mask
      (cell 58);
    - `naive` — per-row naive losses (cell 17) → per-seed random-split accuracies
      (cell 69). Same dtype, opposite meaning, which §4.1's heading — *one name,
      one meaning, for the whole notebook* — is precisely about.

13. **§4.2 — cell 42 is not idempotent, and the notebook never says so.** It
    fine-tunes `model` and builds `opt` without re-creating `model`, which was
    loaded in cell 38. Re-running it trains a second epoch on the same data, and
    silently invalidates `zero_shot` from cell 40, which is still printed by cell
    44. This is lecture 19's cell-40 defect, and §4.2 was written for it.

14. **§1.2 — not one cell has a stored output.** All 26 code cells have
    `execution_count: null` and zero outputs, so every figure in the prose —
    0.731, 0.313, 88.7, log 2, the parameter count — reconciles with nothing.
    `check_notebooks.py --advisory` lists **47** prose figures matching no stored
    output for this notebook. Whatever else is true, the §1.2 machine check
    cannot pass on a file with no outputs, and §10's pre-flight item 1
    (restart-and-run-all, it passes) has left no evidence behind.

15. **§7.1, the plot that drops a point.** Cell 19 clamps with
    `np.maximum([...], 1e-9)` and its annotation warns that *"a median error of
    exactly 0.0 is not plottable on a log scale"*. I ran the sweep: no median
    error is ever 0.0 — the ten values run 1.8e-08 to 2.6e-04. The value that
    **is** unplottable is the `nan` at loss 110, where the naive form is 100%
    non-finite, and `np.maximum` does not repair a `nan`. The naive curve
    therefore stops at 100 and matplotlib says nothing. The annotation warns
    about the right hazard for the wrong reason, and the clamp does not fix the
    real one.

### Checked, and defensible — recorded so the next reader does not re-check

16. **§5.1 indentation.** Exactly one markdown line in the notebook is indented
    ≥ 4 spaces outside a fence: cell 11's `    = -z_c + \log\sum_j e^{z_j}$$`,
    the continuation of a `$$…$$` block. I rendered the cell through a CommonMark
    implementation: it stays inside the paragraph and does **not** become a code
    block, because an indented block cannot interrupt a paragraph. The repo's own
    `check_notebooks.py` exempts exactly this case. No fence marker anywhere in
    the notebook is indented. No violation.

17. **§3.1 quoted code.** No ```` ```python ```` block appears in any markdown
    cell of this notebook, so there is nothing to fail to match. Checked by
    regex over all 46 markdown cells.

18. **Cross-references that do resolve:** "Lecture 21's padding bug" (lecture 21
    does contain `padding_idx`, `pack_padded` and the phrase "last-of-padding");
    "Lecture 9, unchanged: k-means … by silhouette" (lecture 9 mentions
    silhouette 40 times); "the same rule as application 5" (application 5 spans
    lectures 9–10). All three checked.

19. **The `norm` claim in cell 65's box** — *"an exact string match finds fewer
    duplicates than there are"* — is **false on this corpus**, but harmlessly so:
    normalised matching finds 123 cross-split and 96 within-train duplicates, and
    raw string matching finds the identical 123 and 96. The constraint costs
    nothing and buys nothing here; it would be worth keeping only with the word
    "may".

20. **Cell 67's *"IMDb is clean"*** sits directly under a cell that prints 123
    test reviews present verbatim in training. "Clean" is defensible at 0.49% and
    the sentence's real claim — that the cost cannot be *measured* here — is
    correct. Worth one clause acknowledging the 123 rather than rounding it to
    clean.

### Not checked, and why

21. **Every accuracy and every training loss.** The brief forbids executing
    training cells, so I did not run cells 30, 42 or anything downstream of them.
    Unverified as a consequence: *"the extra softmax costs N points"*, the four
    numbers in cell 44's table, and `assert ft_acc > 0.75`.

22. **§1.1, the notebook's central claim about the floor.** Cell 33 states *"It
    stops above 0.3, exactly where the algebra said it would."* The floor itself
    I did verify — 0.31326, and a maximally confident double-softmax row hits it
    exactly — but *whether the softmaxed model's loss reaches it in two epochs on
    5,000 reviews*, and whether the correct model gets meaningfully below it in
    the same two epochs, I could not test. If the correct model also ends near
    0.4, the two loss columns differ by less than the prose implies and the
    "tell" is not visible. This is the one claim in the notebook I would want run
    before the lecture, and it is one cell.

23. **The deadlock in cell 1's comment.** I did not reproduce the macOS KMeans
    deadlock — doing so means hanging a kernel on purpose. What I did verify is
    the *other* consequence of the same line: `OMP_NUM_THREADS=1` makes cell 61's
    logistic regression more than 200× faster on this machine (0.01 s against
    2.33 s per fit, back to back), so the variable is load-bearing whether or not
    the deadlock reproduces on your hardware.

24. **Colab wall clocks.** Every timing above is from a 16-core laptop with an
    MPS device available. The Colab CPU multiplier of three to five is an
    estimate from the core count, not a measurement.
