# Lecture 21 — Reading the customers

**Rebuilding this notebook in Colab by prompting.** IMDb film reviews:
tokenisation, embeddings, a bidirectional GRU, and one swap that changes nothing
but the embedding table.

23 code cells. Follow them in order. You type the prompt, read what comes back,
run it, and check it against **Expect** before you move on.

---

## Before you touch the keyboard

**Runtime.** *Runtime → Change runtime type → T4 GPU* if you can get one. The
notebook runs without it — the GRU cells here are the slowest CPU material in
the course, and the honest budget is below.

**Wall clock, measured.** Every ⏱ in this file was measured on an Apple-silicon
laptop CPU, torch 2.13, on the real corpus: once with all 12 threads, and once
pinned to 2 threads (`torch.set_num_threads(2)`) as a stand-in for a free Colab
CPU runtime, which gives you 2 vCPUs. Where a cell shows two figures they are
*12-thread* / *2-thread*, in that order.

**No GPU figure in this file was measured.** I did not have one. Where the
current notebook quotes GPU seconds, this script marks them unverified rather
than repeating them.

| | 12-thread laptop CPU | 2-vCPU (Colab CPU) |
|---|---|---|
| whole notebook, end to end | about 12 min | **35–45 min** |
| the single worst cell (cell 7, tf-idf) | 3 min 7 s | 4–6 min |
| one GRU training run (cell 16, 18, 20) | 35 s | **3 min 20 s** |

Five GRU runs happen in this notebook. On 2 vCPUs that alone is about 14
minutes. Plan for it, or get the T4.

**Downloads.** The IMDb tarball is 84,125,825 bytes — 80 MiB, 30–90 s the first
time. DistilBERT's weights (cell 19) are another ~250 MB.

**Every number in the Expect lines below was re-derived from the corpus in
`notebooks/datasets/aclImdb` with `python3` before this file was written.** The
two exceptions are marked *not measured*: the accuracies of the trained GRUs,
which I did not train, and anything about a GPU.

**One deliberate departure from the current notebook.** The current version
announces its padding defect in five cells before the defective cell runs. Here
the defective model is written by the natural prompt at cell 13, trained
unannounced at cell 15, and only revealed at cell 16 — after you have written
its number down. Same cells, same lesson, reordered. See §8.1 of `GUIDELINES.md`.

---

## Cell 1 — setup

**Prompt to type:**

> Standard setup cell: import numpy, torch, matplotlib, sklearn; print the
> versions; set a seed of 42 everywhere; pick cuda, then mps, then cpu, and
> print which one. If it lands on cpu, print how to switch Colab to a GPU.

**Expect:** three version lines and one `device` line. Nothing else.
**Assert:** none.
**Annotate:** short

---

## Cell 2 — the corpus

**Prompt to type:**

> Download `https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz`
> into `datasets/` if it isn't already there, untar it, and read the train and
> test halves that ship with it into lists of strings plus 0/1 label arrays
> (`pos` = 1). Print the sizes and the positive share of each half. Don't
> re-split anything.

**Expect:** `25,000 train, 25,000 test`, and `positive share: train 0.500, test
0.500` — exactly 12,500 of each class in each half, so both shares are 0.500 to
three decimals.
**Assert:**

```python
assert len(train_x) == 25_000 and len(test_x) == 25_000
assert set(train_y) == {0, 1}
```

**⏱** 30–90 s the first time (80 MiB), then about 8 s to read 50,000 files off
disk. Instant on re-run only if you keep `datasets/`.
**Annotate:** short

Balanced by construction, in both halves. That is what makes accuracy a
defensible metric here — the condition application 2 (lectures 3–4) spent ninety
minutes on.

---

## Cell 3 — carve the validation set out of TRAINING

**Prompt to type:**

> From the 25,000 training reviews, take 5,000 for fitting and 2,000 for
> validation. Leave the test half alone. Print the three sizes.

**Expect:** `fit 5,000   val 2,000   test 25,000 (untouched)`. Then print
`fit_y.mean()` and `val_y.mean()` yourself: with a seed-42 permutation they are
**0.4988** and **0.4900**. If either prints **1.000**, read the annotation.
**Assert:**

```python
assert set(fit_i).isdisjoint(val_i), "the split overlaps"
assert len(fit_x) == N_FIT and len(val_x) == N_VAL
assert 0.45 < fit_y.mean() < 0.55, "the fit split is not balanced"
```

**Annotate:** full

* **Left open:** the word *shuffle* is nowhere in that prompt, and nothing in it
  says the corpus is ordered. It is: `load_imdb` reads `pos` before `neg`, so
  **the first 12,500 training labels are all 1** — verified. The permutation is
  the only thing making the slice legitimate, and you did not ask for one.
* **The usual student version:** `train_x[:5000]`, which is 5,000 positive
  reviews and zero negative ones. The model learns to answer "positive", the
  validation set agrees with it, and the "accuracy" you print is recall on one
  class. The two disjointness asserts both pass.
* **How you would catch it:** print the positive share of every split you cut,
  every time. Three sizes tell you nothing; `fit_y.mean()` tells you everything,
  and it is the same one line that catches this in half the classic text
  corpora, which are nearly all sorted by label.

---

## Cell 4 — how long is a review, and what does a cut cost

**Prompt to type:**

> Write a word tokenizer: lowercase, replace `<br />` with a space, and take
> `[a-z0-9']+`. Tokenise all 25,000 training reviews and print mean, median,
> 90th percentile and max length. Then set MAXLEN = 192 and print what fraction
> of reviews it truncates and what fraction of all words it keeps.

**Expect:** `mean 234   median 174   90th pct 458   max 2473`, then `cutting at
192: truncates 43.8% of reviews, keeps 66.7% of all words`. The two percentages
are meant to be far apart — that is the point of printing both.
**Assert:** none.
**⏱** about 12 s to tokenise 25,000 reviews.
**Annotate:** short

Three decisions are already made inside those four lines of tokenizer:
lowercasing, the `<br />` replacement, and the character class that defines a
word. None of them is neutral, and none was in your prompt.

---

## Cell 5 — the vocabulary, and the histogram

**Prompt to type:**

> Count every token in the training half. Print how many distinct words there
> are and how many appear exactly once, as a share. Then a histogram of review
> lengths with a dashed line at MAXLEN — clip the lengths first so one outlier
> doesn't flatten it.

**Expect:** `87,171 distinct words`, `35,344 of them appear exactly once
(40.6%)`, and a histogram whose bulk sits left of the dashed line. Clip at
1,200: exactly **15** of 25,000 reviews are longer than that, and the longest is
2,473 words.
**Assert:** none.
**Annotate:** short

Two words in five are seen once and never again.

**A caution the current notebook does not give you.** Cell 8 keeps the top
19,998 words, and the least frequent word that survives that cut appears
**twice** — verified. So no once-seen word ever gets a row in the embedding
table; they all become `[UNK]`. The problem the hapax count actually diagnoses
is *coverage*, not "learning a 128-dimensional vector from one example". 8.7% of
the words that do get a row appear exactly twice, and those are the ones that
sentence describes.

---

## Cell 6 — the trivial anchor

**Prompt to type:**

> Print the accuracy of always predicting the commoner class on the test set.

**Expect:** `always predict one class: 50.0%`.
**Assert:** none.
**Annotate:** short

---

## Cell 7 — the anchor that is actually hard to beat

**Prompt to type:**

> tf-idf on the 25,000 training reviews with unigrams and bigrams, `min_df=2`,
> capped at 200,000 features, then logistic regression, and score it on the test
> half. Print the accuracy, the number of features and how long it took.

**Expect:** `tf-idf + logistic regression: 90.1%  (200,000 features)`. The
feature cap binds exactly — you get 200,000, not fewer. **This is the number
your recurrent network has to beat, and it is the one to write on the commitment
sheet.**
**Assert:**

```python
assert X_bow_train.shape[0] == 25_000
assert X_bow_train.shape[1] == X_bow_test.shape[1], "different feature spaces"
```

**⏱ Measured: 3 min 7 s** on a 12-thread laptop CPU — 2 min 41 s of it inside
`fit_transform`, 27 s in the logistic regression. Budget **4–6 min on a 2-vCPU
Colab CPU**; the bigram vectoriser is the single slowest non-GRU cell in the
notebook and it does not use the GPU at all, so a T4 does not help here. The
current notebook says "about 15 seconds". It is not.
**Annotate:** full

* **Left open:** `ngram_range` and `min_df`. `TfidfVectorizer` defaults to
  `ngram_range=(1, 1)` and `min_df=1` — a prompt that says only "tf-idf and
  logistic regression" gets you unigrams, a much weaker and much faster anchor,
  and you will not notice because it still prints a plausible number.
* **The usual student version:** `vec.fit_transform(test_x)` on the second line.
  `fit` and `transform` are different verbs — reviewer question 2 — and
  `fit_transform` on test builds a *different* feature space. With
  `max_features=200_000` binding on both halves the column counts match, so the
  shape assert above passes and nothing raises; you get a wrong number rather
  than an error.
* **How you would catch it:** the shape assert is necessary and not sufficient.
  The sufficient check is `vec.vocabulary_` — it is a fitted attribute like any
  other, and it must have been fitted once. If in doubt, `assert
  X_bow_test.shape[1] == len(vec.vocabulary_)` after a single fit.

---

**Stop here. On paper, not in the notebook.**

```
Metric:                                            ____________
Accuracy the desk would need to trust it:        ____________ %
Accuracy I expect from the model I build today:  ____________ %
Best accuracy I actually obtain:                 ____________ %
```

You have two anchors: 50.0% and 90.1%. Estimate against the second.

---

## Cell 8 — the vocabulary, fitted on the fit split only

**Prompt to type:**

> Build a word→index map from the 5,000 fit reviews only: the 19,998 commonest
> words, with 0 reserved for padding and 1 for `[UNK]`. Then print, on the test
> half, the share of tokens that are out of vocabulary and the share of
> *distinct* words that are.

**Expect:** `19,998 words in the vocabulary` (the fit split has 42,370 distinct
words, so the cap binds), then `unseen test tokens: 3.9%` and `unseen DISTINCT
test words: 77.9%` over 85,994 distinct test words.
**Assert:**

```python
assert 0 not in w2i.values() and 1 not in w2i.values()
assert len(w2i) <= VOCAB - 2
```

**⏱** about 15 s — it tokenises the whole test half to compute the two rates.
**Annotate:** short

3.9% looks survivable. 77.9% never does, and the distinct words are the
informative ones. The problem is not the *size* of the vocabulary: a word
vocabulary is closed and language is not, so no size fixes it.

---

## Cell 9 — the same sentence, under subwords

**Prompt to type:**

> Load `distilbert-base-uncased`'s tokenizer. Take the sentence "The plot was
> unwatchable and utterly discombobulating." and print it three ways: our word
> tokens with anything outside `w2i` shown as `[UNK]`, WordPiece's pieces, and
> the pieces-per-word ratio with both vocabulary sizes.

**Expect:** exactly this, verified:

```
word tokenizer : ['the', 'plot', 'was', 'unwatchable', 'and', 'utterly', '[UNK]']
WordPiece      : ['the', 'plot', 'was', 'un', '##watch', '##able', 'and', 'utterly',
                  'disco', '##mbo', '##bula', '##ting', '.']
7 words -> 13 pieces (1.86 per word)
vocabulary: ours 20,000   WordPiece 30,522
```

**Assert:** none.
**Annotate:** short

**Read that output carefully — it does not say what the current notebook says it
says.** Only *one* word becomes `[UNK]`, and it is `discombobulating`.
`unwatchable` — the word that actually carries the sentiment — is in our
vocabulary, at rank under 19,998. So this cell demonstrates coverage of a rare
word, not loss of the sentiment-bearing one. Say that, or pick a sentence where
the sentiment word really is out of vocabulary.

The 1.86 pieces per word is this sentence, not the corpus. Over 500 random test
reviews the real ratio is **1.34** — that number matters at cell 18.

---

## Cell 10 — how often does WordPiece give up

**Prompt to type:**

> On 2,000 randomly chosen test reviews, count the `[UNK]` tokens WordPiece
> emits as a share of all its tokens.

**Expect:** `[UNK] rate over ~552,000 subword tokens: 0.0000%` — with
`truncation=True, max_length=512` on a seeded random sample I measured
**exactly zero** `[UNK]`s in 551,797 tokens.
**Assert:**

```python
assert n_unk / n_tok < 0.001, "a subword tokenizer should almost never miss"
```

**⏱** about 5 s.
**Annotate:** short

The word *randomly* is load-bearing and it is in the prompt on purpose:
`test_x[:2000]` is 2,000 positive reviews (cell 3's annotation). It happens not
to matter for a rate that ignores the label. Take the habit anyway.

---

## Cell 11 — from integers to vectors

**Prompt to type:**

> `nn.Embedding` with 20,000 rows and 128 columns, index 0 reserved for padding.
> Feed it a random batch of 32 by MAXLEN ids and print the output shape and the
> table's parameter count.

**Expect:** `embedding table: 2,560,000 parameters` and `(32, 192) -> (32, 192,
128)`.
**Assert:**

```python
assert emb(x).shape == (32, MAXLEN, 128)
assert torch.equal(emb.weight[0], torch.zeros(128)), "padding row is not zero"
```

**Annotate:** full

* **Left open:** what "reserved" buys you. `padding_idx=0` does exactly two
  things: it zeroes row 0 at construction, and it zeroes row 0's gradient. It
  does **not** protect the row from being written to. I verified both halves:
  after `emb.weight.data.copy_(init)` the row is whatever `init` had there, and
  a subsequent Adam step leaves it exactly where the copy put it, because the
  gradient really is zero. Remember this at cell 19.
* **The usual student version:** omitting `padding_idx` entirely — its default
  is `None`. Row 0 is then an ordinary trainable row initialised from N(0, 1);
  its measured standard deviation is **1.00**, not something small. Nothing
  errors, and padding slowly acquires a meaning.
* **How you would catch it:** assert the padding row is zero *after* training,
  not only at construction. It is one line, it is the only way to know the flag
  did what it claims, and in this notebook it would fail for three of the five
  configurations you are about to train.

---

## Cell 12 — padding, and the length you must keep

**Prompt to type:**

> Write `pad_batch(seqs)`: pad or truncate a list of id-sequences to a
> `(n, MAXLEN)` int64 array with zeros, and return it together with an array of
> the true lengths. Then `encode_words(texts)` that tokenises with `w2i`
> (unknown → 1) and calls it. Encode the fit and validation splits.

**Expect:** `fit batch (5000, 192)   lengths 10–192`. The minimum true length in
the fit split is 10 and 43.1% of its rows sit at exactly 192.
**Assert:**

```python
assert Xf_w.shape == (N_FIT, MAXLEN)
assert (Lf_w >= 1).all() and (Lf_w <= MAXLEN).all()
assert (Xf_w[0, Lf_w[0]:] == 0).all(), "something is written past the true length"
```

**Annotate:** short

Note that the third assert is vacuously true for any row whose length is
MAXLEN — the slice is empty — and 43.1% of these rows are. Pick a row you know
is short, or assert it over the whole batch with a mask.

---

## Cell 13 — the classifier

**Prompt to type:**

> Write a PyTorch model that classifies a padded batch of token ids: an
> embedding, a bidirectional GRU, and a linear head returning two logits.
> Embedding 128, hidden 64, `batch_first`.

**Expect:** a small `nn.Module`, and `2,634,754 parameters` for a 20,000-word
vocabulary — 2,560,000 of them in the embedding table, i.e. 97% of the model is
the lookup table.
**Assert:**

```python
assert _m(Xf_w[:4], Lf_w[:4]).shape == (4, 2), "the head must return two logits"
```

**Annotate:** short

Read what came back before you run it, and be able to say what every line does.
Two things are worth checking on sight: with `bidirectional=True` the hidden
state `h` has shape `(2, N, hidden)` — `h[-1]` is the backward direction alone,
and you want both — and if you are given `pack_padded_sequence`, its `lengths`
argument must be a **CPU** tensor even when everything else is on the GPU. The
error message for that names neither the argument nor the fix.

---

## Cell 14 — the loop, with early stopping

**Prompt to type:**

> Training loop: Adam at 1e-3, batch 64, 2 epochs, cross-entropy. After each
> epoch measure validation accuracy over the whole set and print the epoch loss
> and accuracy with elapsed seconds. Keep the weights from the best validation
> epoch and reload them at the end. Only pass parameters with `requires_grad` to
> the optimiser.

**Expect:** two function definitions and no output. The accuracy helper must
count hits over the set and divide once — not average per-batch accuracies,
which is a different number whenever the last batch is short.
**Assert:** none — this cell defines, it does not run.
**Annotate:** short

Two details to check in what comes back. `net.state_dict()` returns references
to live tensors, so the "best" weights keep training unless each one is
`.clone()`d. And once the loop reloads the best epoch, **`curve[-1]` is no
longer the accuracy of the model you are holding** — `max(curve)` is. Use
`max(curve)` in every comparison below.

---

## Cell 15 — train it

**Prompt to type:**

> Seed 42, build the model, train it on the word-tokenised fit split, tag it
> "our words / random".

**Expect:** two epoch lines and a "keeping epoch N" line. **Write the best
validation accuracy on the paper sheet before you go on.** Compare it with 50.0%
and with 90.1%.

*Not measured by this script — I did not train these models, and there is no
number here for you to check yours against. That is deliberate: the only thing
you can verify is your own run.*
**Assert:** none.
**⏱** 10 s / **44 s**. This is the fast one; the runs from cell 16 on are five
times slower, for a reason you will find out.
**Annotate:** short

---

## Cell 16 — ⚠ what the summary was

Look at the forward pass you were given at cell 13. It ends

```python
out, _ = self.rnn(e)
return self.head(out[:, -1, :])
```

`out[:, -1, :]` is position `MAXLEN - 1` — **the last position of the padded
batch**. For every review shorter than 192 words that position is padding, and
**56.9%** of the fit rows are shorter than 192. The head is reading the GRU's
state after it has consumed a run of zeros.

Nothing in your prompt was wrong. It said "a padded batch" and did not say what
to do about the padding, and the model that came back took the last *row* rather
than the last *word*.

**Prompt to type:**

> That takes the last position of the padded batch, which is padding for most
> reviews. Rewrite it to summarise each review at its true end: pack the
> sequences with the lengths, take the final hidden state of both directions and
> concatenate them. Keep everything else identical. Call it `PackedGRU`. Then
> train it with the same seed and print both validation accuracies.

**Expect:** the same 2,634,754 parameters — the fix costs nothing — and two
accuracies. The packed model should be the better of the two; how much better is
your measurement, not mine.
**Assert:**

```python
assert packed_m(Xf_w[:4], Lf_w[:4]).shape == (4, 2)
```

**⏱ 35 s / 3 min 20 s** for the training run — measured. Packing makes this
model **3.4× slower per batch than the buggy one on a CPU** (0.220 s vs 0.065 s
at 12 threads, 1.27 s vs 0.275 s at 2 threads), because a packed sequence gives
the CPU a different batch size at every timestep instead of one clean rectangle.
The correct model is the expensive one. GPU figures unverified.
**Annotate:** full

* **Left open:** the word "padded" in the cell 13 prompt described the *input*
  and said nothing about the *output*. A specification that mentions padding has
  to say how the summary avoids it — that is the whole content of the fix, and
  it is one clause.
* **The usual student version:** exactly what you just ran. `out[:, -1, :]` is
  the obvious reading of "the last output", the shapes are right, the loss goes
  down, and nothing raises. The second most common is `h[-1]` after
  `bidirectional=True`, which is the backward direction's final state on its
  own — half the model, silently.
* **How you would catch it:** compare the two accuracies on **the same
  validation rows, from the same seed, with only the summary changed** — which
  is what these two runs are. And then cell 17, which catches it without
  training anything at all.

---

## Cell 17 — the test that catches it

**Prompt to type:**

> Write a test that would have caught that without training anything: take one
> short review, run it through padded to its own length and padded to MAXLEN,
> and check the logits don't move. Run it on both models and print how far each
> one moves.

**Expect:** the packed model's two outputs are identical — I measured a maximum
absolute difference of **exactly 0.0**, not merely small. The last-of-padding
model moves: on an untrained pair, **0.28** for a 155-token review and **0.46**
for a 10-token one. Shorter review, more padding, bigger error.
**Assert:**

```python
n = int(Lf_w[short])
assert n < MAXLEN, "pick a review that is actually shorter than MAXLEN"
one = packed_m(Xf_w[short:short+1, :n].to(device), torch.tensor([n]))
two = packed_m(Xf_w[short:short+1, :].to(device),  torch.tensor([n]))
assert torch.allclose(one, two, atol=1e-4), "padding is being read"
```

**Annotate:** full

* **Left open:** *which* review. The current notebook uses row 0 and does not
  check its length. **43.1% of the fit rows are exactly MAXLEN long**, and for
  any of those the two inputs are the same tensor, both models pass, and the
  buggy one prints that it moves by 0.000. With seed 42 row 0 happens to be 155
  tokens, so the test works — by luck. The `assert n < MAXLEN` above is what
  makes it work on purpose.
* **The usual student version:** asserting `out.shape == (N, 2)`. **Both models
  pass it.** A padding bug is invisible to a shape test, and "reviewer question
  3 — what is the shape here?" is the question that finds it by hand, not an
  assertion on shape that finds it automatically. Those are two different
  things and the notebook uses one phrase for both.
* **How you would catch it:** feed the same content two ways and require the
  same answer. Any model consuming variable-length input should be invariant to
  how that input was padded. It is four lines, it needs no labels, no training
  and no GPU, and it generalises to every masked model in the rest of the
  course.

---

## Cell 18 — change one thing: the tokenizer

**Prompt to type:**

> Encode the fit and validation splits with the DistilBERT tokenizer instead —
> truncate and pad to MAXLEN, and take the true lengths from the attention mask.
> Then train the same `PackedGRU` on them, same seed, same epochs, vocabulary
> 30,522. Tag it "subword / random".

**Expect:** `subword batch (5000, 192)   vocabulary 30,522`, then a model with
**3,981,570** parameters (the wider table is the whole difference) and an
accuracy close to cell 16's.
**Assert:**

```python
assert Xf_s.shape == (N_FIT, MAXLEN)
assert (Lf_s >= 1).all()
```

**⏱ 35 s / 3 min 20 s** for the run, plus about 2 s to encode 7,000 reviews.
**Annotate:** full

* **Left open:** what result would count as success. **At the deck's scale these
  two land within a few hundredths of a point of each other. That is a null
  result, not a ranking** — two single-seed numbers that close say only that the
  effect is smaller than the seed-to-seed spread, and nothing about which
  tokenizer is better. §2.4. Do not write a sentence with "better" in it here
  unless you have run several seeds and can quote the spread.
* **The usual student version:** expecting the subword tokenizer to help by
  itself. Three measured reasons it does not. The token-level OOV rate you were
  fixing was only **3.9%** (cell 8), so there was little to win. WordPiece emits
  **1.34 pieces per word** over the corpus (cell 9), so the same 192 positions
  now hold about **143 words instead of 192** — you bought coverage and paid for
  it in sequence length. And `un`, `##watch`, `##able` arrive as three random
  vectors whose *composition* the model must learn from 5,000 labels.
* **How you would catch it:** hold everything else fixed and say so — same
  architecture, same seed, same epochs, same 2,000 validation rows — and then
  report the difference next to something that tells you how big a difference is
  meaningful. A subword tokenizer is not a better tokenizer by itself. It is a
  vocabulary that someone else's pretraining can be poured into, which is the
  next cell.

---

## Cell 19 — pour the pretraining in

**Prompt to type:**

> Load `distilbert-base-uncased`'s word embedding table, project it from 768
> down to 128 with PCA, and rescale the result so it can be dropped into our
> `nn.Embedding`. Print how much variance the 128 components keep.

**Expect:** `pretrained table: (30522, 768)` and `128 components keep 47.0% of
the variance`. PCA itself takes under a second; the model download is ~250 MB.
**Assert:**

```python
assert Z.shape == (tk.vocab_size, EMB_DIM)
```

**⏱** dominated by the download, not the arithmetic.
**Annotate:** full

* **Left open:** what "so it can be dropped in" means numerically. The projected
  table has the variance of the data — measured std **0.0698** — and the
  architecture it is going into initialises from something else entirely.
  Dropping a table in at the wrong scale changes the effective learning rate of
  everything downstream, and no error is raised.
* **The usual student version:** `Z / Z.std() * 0.1`, with a comment saying it
  matches `nn.Embedding`'s own scale. **It does not.** `nn.Embedding`
  initialises from N(0, 1) — I measured the default table's std at **1.00**. The
  line as written puts the pretrained vectors in at one tenth of the scale the
  architecture actually uses. That is a defensible choice; it is not the choice
  the comment claims, and the comment is in the current notebook.
* **How you would catch it:** print the number you are claiming to match.
  `nn.Embedding(30522, 128).weight.std()` is one line and it settles it. The
  same line catches the second problem: `copy_` writes **every** row including
  row 0, so after this table is installed the padding row is no longer zero — I
  measured its norm at **1.09** — and `padding_idx` will never move it back,
  because all `padding_idx` does after construction is zero the gradient. Zero
  row 0 explicitly after the copy.

---

## Cell 20 — frozen, then tuned

**Prompt to type:**

> Two more runs on the subword batches with that table as the initial
> embedding: one with the table frozen, one with it trainable. Same seed before
> each. Tag them "subword / frozen" and "subword / tuned".

**Expect:** four epoch lines and two "keeping epoch" lines. Frozen isolates what
the pretrained vectors are worth; tuned shows what adapting them adds. Neither
number is predicted here.
**Assert:** none.
**⏱ 70 s / 6 min 40 s** — two runs.
**Annotate:** full

* **Left open:** what "frozen" has to mean in the optimiser, not just in the
  tensor. Setting `requires_grad = False` stops the updates; it does not stop
  Adam from being handed the parameter.
* **The usual student version:** `torch.optim.Adam(net.parameters())` with a
  frozen table. Adam then allocates and carries two moment buffers for
  **3,906,816** embedding parameters it will never update — 98% of this model —
  and on a small GPU that is exactly where the memory goes. The cell 14 loop
  filters on `requires_grad` for this reason.
* **How you would catch it:** `sum(p.numel() for g in opt.param_groups for p in
  g['params'])` against `sum(p.numel() for p in net.parameters() if
  p.requires_grad)`. They must be equal. Two runs, one variable, is the cheapest
  possible ablation and it answers a question that the tuned run alone cannot.

---

## Cell 21 — the test set. Once.

**Prompt to type:**

> Encode the full test half both ways, then score every model on it and print a
> table with the two baselines: majority, tf-idf, and all four GRU
> configurations. Word models on the word encoding, subword models on the
> subword one.

**Expect:** six rows. Two of them are known before you run: `always one class
50.0%` and `tf-idf + logistic regression 90.1%`. The four GRU rows are yours.
**Every model must be scored on the same 25,000 test reviews** — the encodings
differ, the rows do not.
**Assert:**

```python
assert Xt_w.shape[0] == 25_000 and Xt_s.shape[0] == 25_000
```

**⏱** about 1 min / 3 min: two encodings of 25,000 reviews (1.6 s word, 5 s
WordPiece — the tokenising is not the cost) plus one GRU forward pass each,
measured at **16.7 s / ~40 s** per pass over the full test half.
**Annotate:** short

Keep the anchors in the table. The bag of words is in that column for a reason
and it is not there to be flattered. And keep all four GRU configurations —
dropping the two that did not win is how a null result turns into a ranking.

---

## Cell 22 — four curves

**Prompt to type:**

> Plot the four validation curves on one axis, markers and lines, integer epoch
> ticks, accuracy in per cent.

**Expect:** four two-point lines. Force `plt.xticks(range(1, EPOCHS + 1))` or
matplotlib offers you epoch 1.5.
**Assert:** none.
**Annotate:** short

Two points is not a trend. This figure shows where the four configurations sit,
not where they are going — which is why markers matter: a bare line between two
points invites an extrapolation the data cannot support.

---

## Cell 23 — the requirement the metric cannot see

**Prompt to type:**

> Time each of the three deployable configurations end to end on the full test
> set — raw strings in, labels out, tokenising included — three passes each, and
> print the median with the min and max and the milliseconds per review.

**Expect:** measured on a 12-thread laptop CPU: `tf-idf + logistic regression`
about **26.5 s** per pass (1.06 ms/review), `GRU, our words` about **18 s** (1.6
s to tokenise, 16.7 s to run). tf-idf wins on CPU, but only by about 1.4× — much
less than the ordering the notebook asserts.
**Assert:** the current notebook writes

```python
assert cheapest.startswith("tf-idf"), f"expected counting words to win, got {cheapest}"
```

**Do not copy that assert without reading this.** The margin on CPU is 1.4×, and
the two sides scale differently: `vec.transform` is CPU-bound bigram counting
that a GPU does not touch, while the GRU forward pass is the part a T4
accelerates by an order of magnitude. On the GPU runtime the setup cell tells
students to select, this assertion is likely to fail and stop the notebook at
the last cell. **I could not test that — I had no GPU.** If you are on a T4,
expect it to fire, and turn it into a `print` of the ordering.
**⏱** measured 3 min 20 s at 12 threads; **8–10 min on 2 vCPUs**. Nine full
passes over 25,000 reviews.
**Annotate:** short

None of these is anywhere near the brief's hour, so the cost column does not
decide anything here. A requirement that turns out not to bind is still worth
measuring — you did not know it did not bind until you measured. It starts to
bind at lecture 22, where the model is **25.2×** larger: DistilBERT is 66,362,880
parameters against this notebook's 2,634,754.

---

# Defects found in the current notebook

`notebooks/lecture-21.ipynb`, checked against `GUIDELINES.md`. Everything marked
**verified** was re-derived with `python3` against the real corpus in
`notebooks/datasets/aclImdb` and a real torch 2.13 / transformers 4.57.3
install. Everything marked **not checked** says why.

### Numbers that do not reconcile (§1.1, §1.2)

1. **The notebook stores no outputs at all — 23 code cells, zero stored
   outputs.** Verified with `nbformat`. §1.2 requires every prose figure to
   appear in a stored output; none can, so every figure in the file is
   unverifiable from the file itself. This is the root cause of most of what
   follows.
2. **"One review of 2,470 words stretches the axis"** (cell 13's `catch`). The
   longest training review is **2,473** words. Verified. §1.1 — a transcribed
   figure, and off by three in a sentence whose whole job is precision about an
   outlier.
3. **"seeing 100,000 distinct words"** (cell 13's `student`). The actual count
   is **87,171**. Verified. It is framed as a hypothetical, but it is the
   notebook's own quantity and it is 15% high.
4. **"Thirteen lines, and eleven of them are Lecture 12"** (cell 36). The
   shipped `GRUClassifier` is **19 non-blank lines**. Verified by counting the
   cell. Thirteen is roughly the count of the class *without* the deliberate
   defect branch and *without* the `init`/`freeze` transfer machinery — i.e. of
   a class that is not in the notebook. §1.1.
5. **"⏱ about 15 seconds: 25,000 documents, unigrams and bigrams"** (cell 20,
   and the prompt label "⏱ 15 s"). Measured: **187 seconds** — 161 s in
   `fit_transform`, 27 s in `LogisticRegression` — on a 12-thread Apple-silicon
   CPU. Verified. That is 12× the stated figure on hardware faster than a Colab
   CPU runtime. §7.1, and the most consequential timing error in the notebook.

### Comparisons on the wrong quantity (§2.1, §1.5)

6. **The headline "the bug costs N points" compares two numbers that describe
   models nobody is holding.** Cell 46 prints `curve_scratch[-1]` against
   `curve_padbug[-1]` — the *last* epoch's validation accuracy — while `train()`
   has already reloaded the weights of the *best* epoch and printed `best_val`.
   Verified by reading `train()`: `net.load_state_dict(best_state)` runs before
   it returns, and `curve[-1]` is untouched by that. With `EPOCHS = 2` the two
   agree only when epoch 2 won both runs. §1.5 — two numbers for the same
   quantity, never reconciled. The fix is `max(curve_scratch)`.
7. **The prompt box at cell 59 promises "every configuration's test accuracy"
   and the cell scores three of five.** Verified: `curve_wp_random` and
   `curve_frozen` are produced by calls that discard the model with `_`, so
   "subword / random" and "subword / frozen" have no test number. The figure at
   cell 62 plots four curves; the table has three GRU rows. §2.1.
8. **The header claims the ordering of four configurations is preserved at this
   scale, and §11 says two of them land within a few hundredths of a point.**
   Both are in the notebook: cell 0 — *"the ordering of the four configurations
   is the same, and the ordering is the point"* — and cell 53 — *"a null result,
   not a ranking"*. An ordering between two configurations inside their own
   seed-to-seed spread is not an ordering. §2.4, and the correction is not
   propagated back to the header (§2.3). Verified by reading both cells; the
   accuracies themselves I did **not** measure.
9. **The hapax diagnosis is stated on rows the model never sees.** Cell 14
   computes 87,171 distinct words and 40.6% hapax over all **25,000** training
   reviews; the vocabulary at cell 24 is fitted on the **5,000** fit reviews,
   where the figures are 42,370 and 42.5%. Verified. Close enough that the
   sentence survives, but §2.1 asks that the rows be stated and they are not.
10. **"You cannot learn a 128-dimensional vector for a word from one example"
   (cell 15) describes words that never get a vector.** Verified: the least
   frequent word surviving the top-19,998 cut appears **twice**, so every
   once-seen word maps to `[UNK]` and has no row at all. 8.7% of the kept
   vocabulary appears exactly twice — those are the words that sentence is
   really about. The diagnosis is coverage, not one-shot estimation.

### Claims about library behaviour that are false (§3.2, §6.2)

11. **`Z = (Z / Z.std() * 0.1)  # match nn.Embedding's own scale` (cell 55) does
   not match `nn.Embedding`'s scale.** Measured: `nn.Embedding(30522,
   128).weight.std()` is **1.00**; the projected table's std is 0.0698 and the
   line sets it to 0.100. Verified. The prompt box repeats the claim in its
   `catch`. A tenth of the initialisation scale may well be the right choice —
   the justification given for it is simply not true.
12. **`padding_idx=0` does not keep the padding row at zero in three of the five
   configurations.** Cells 30 and 32 assert and assert again that row 0 is
   zero; cell 38's `self.emb.weight.data.copy_(torch.from_numpy(init))` then
   overwrites every row including row 0 for the frozen and tuned runs.
   Measured: row 0's norm after the copy is **1.09**, and it stays there —
   `padding_idx` zeroes the *gradient*, which I also verified, so an Adam step
   does not move it back. Cell 31's own `catch` says *"assert the padding row
   is zero AFTER training too"*. Take that advice on this notebook and it
   fails. §3.2.
13. **The `[UNK]` demonstration at cell 27 does not show what the prose says.**
   The constraint is *"show the word tokenizer producing `[UNK]` for exactly
   the words that carry the sentiment"* and the input is *"one sentence with
   two rare words in it"*. Measured output: **one** `[UNK]`, and it is
   `discombobulating`. `unwatchable` — the sentiment-bearing word — is inside
   the 19,998-word vocabulary. Verified.

### Tests that can silently stop testing (§3.2)

14. **The padding-invariance test is vacuous for 43.1% of possible rows.** Cell
   49 takes `n = int(Lf_w[0])` and compares `Xf_w[:1, :n]` with `Xf_w[:1, :]`.
   Measured: **43.1%** of fit rows have length exactly MAXLEN, and for any of
   those the two tensors are identical, `torch.allclose` passes for *both*
   models, and the buggy model prints that it moves by 0.000. With seed 42 row
   0 is 155 tokens long, so it works — verified, and by luck. There is no
   `assert n < MAXLEN`. This is the notebook's flagship test.
15. **The last assert in the notebook is likely to fail on the recommended
   runtime.** Cell 65 ends `assert cheapest.startswith("tf-idf")`. Measured on
   CPU: tf-idf 26.5 s per full pass, word GRU 18.3 s of which only 1.6 s is
   tokenising — so tf-idf wins by 1.4×, and the GRU side is the side a GPU
   accelerates while `vec.transform` is not. The setup cell tells students to
   select a T4. **Not checked on a GPU — I had none.** Flagged as at-risk, not
   as proven.
16. **Cell 35's `assert (Xf_w[0, Lf_w[0]:] == 0).all()`** is vacuously true
   whenever row 0 is MAXLEN long — same 43.1%, same missing guard. Verified.

### Staging the defect (§8.1)

17. **Five cells announce the padding defect before it runs.** Verified by
   search: cell 0 (*"Cells marked ⚠ read before running contain a defect on
   purpose"*), cell 34 (*"the setup for the assistant failure two sections
   down"*), cell 38 (the class body carries `if self.last_of_padding:  # the
   assistant's version`, **eight cells** before it is used), cell 44 (a section
   heading plus four paragraphs explaining the bug in full), cell 45 (the
   prompt box, whose `student` field says *"exactly this"*). §8.1's evidence is
   that lecture 19 did this four times and *"nobody falls in"*. This does it
   five times, and the third is inside the code.
18. **The defect is not what the notebook says an assistant returned.** It is a
   `last_of_padding=True` branch inside a class that also contains the correct
   path. Cell 38's prompt box asks for the correct model and the code contains
   both, so the box is not a specification of its own cell. §4.4.

### Cross-references and vocabulary (§3.3, §7.5)

19. **Two numbering systems for the same referent, adjacent, neither defined.**
   Cell 5's prompt says *"the condition application 2 spent ninety minutes on"*
   and cell 7, immediately after, says *"the condition Lecture 4 spent ninety
   minutes on"*. **Both are correct** — `README.md` line 17 says each
   application spans two lectures, so application 2 is lectures 3–4; I checked
   all three such references and application 5 → lecture 10 (PCA/SVD) and
   application 6 → lectures 11–12 (PyTorch) also resolve. Verified. The defect
   is §7.5, not §3.3: nothing in this notebook tells the reader the two
   systems are the same thing.
20. **"Reviewer question 3" (cell 44) is never defined in this notebook.** It
   resolves — `TRICKS.md` line 110: *"3. What is the shape here?"*. Verified.
   But it is invoked here as the question that catches the padding bug, and
   three cells later the notebook says *"a test on output shape does not"*
   catch it. Two different meanings of "shape", one phrase, three cells apart.

### Instructions a reader alone at home cannot carry out (§7.1)

21. **Every ⏱ in the notebook is a GPU figure, and the CPU numbers are given as
   adjectives.** Cell 39: *"about 40–90 seconds per run on a GPU, several
   minutes on a CPU"*. The prompt labels are worse — *"⏱ 40-90 s — our words,
   random table"* with no CPU figure at all. Measured on 2 threads: **3 min 20
   s per packed run**, five runs, ≈ 14 minutes of training. §7.1 asks for the
   CPU number and it is nowhere in the file.
22. **No total is stated anywhere.** Summing my measurements: **35–45 minutes**
   end to end on a 2-vCPU CPU runtime, about 12 on a fast laptop. The header
   says *"the whole notebook finishes in a few minutes"*. On a GPU that may be
   true — **not checked** — but the setup cell explicitly supports the CPU
   path, and on it the claim is out by an order of magnitude.
23. **Cell 63 says the timing cell takes "two to three minutes".** Measured 3
   min 20 s at 12 threads and **8–10 min at 2**. §7.1.

### Checked and clean

I checked these and found no violation, which is worth saying:

- **§5.1 / §5.2** — no markdown line is indented ≥4 spaces outside a fence, and
  no fence marker is indented. Verified by parsing all 45 markdown cells. The
  one fence in the file (cell 21, the commitment form) opens and closes at
  column 0.
- **§3.1** — there are no ```` ```python ```` blocks in any markdown cell, so
  nothing can be quoted that does not exist.
- **§4.2** — every training cell constructs its model inside the call and
  re-seeds first, so re-running any of cells 43, 46, 52, 57 alone is idempotent
  and reproducible. This is better than lecture 19.
- **§4.1** — I found no name rebound to a different type across cells.
- **§7.3** — cell 10's *"Two things to notice"* delivers exactly two.
- **The download size.** *"About 80 MB"* — the tarball is 84,125,825 bytes, 80.2
  MiB. Verified.
- **"25 times larger"** (cell 66). DistilBERT is 66,362,880 parameters against
  the word-vocabulary GRU's 2,634,754 — **25.2×**. Verified exactly. Note it is
  16.7× against the subword model, and the sentence does not say which.

### Not checked

- Every GRU accuracy, every claim about which configuration wins, and the
  header's ordering claim. Training five models was out of scope and would
  produce single-seed numbers that, by the notebook's own §11, cannot settle the
  question anyway.
- All GPU timings, including whether the assert at cell 65 fires on a T4.
- Whether `pip install transformers` is needed in Colab. The setup cell installs
  nothing and cell 27 imports `transformers`; it is preinstalled in the current
  Colab image, which is a fact about Colab I could not verify from here.

### One dead parameter, for completeness

`train(..., double_softmax=False)` is defined and branched on and **never passed
`True` by any cell** — verified, two occurrences in the whole notebook, both in
the definition. It is a hook for lecture 22's first open question. Harmless, and
it is 4 lines a reader will try to account for.
