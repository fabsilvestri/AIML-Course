# Lecture 23 — One catalogue, two modalities

**Rebuilding this notebook in Colab, by prompting.**

Sixteen code cells. You type sixteen prompts and read sixteen outputs. The
lecture is not the code — it is the moment in cell 9 when a number arrives that
looks like evidence and is not.

---

## Before you sit down

**Runtime.** A GPU runtime if Colab offers you one; a CPU runtime works and this
script gives you both figures. Nothing here is trained, so the GPU only buys
inference speed.

**What lands on disk, first run** — measured, not estimated:

| item | size |
|---|---|
| COCO split index (CSV) | 5.0 MB |
| 200 catalogue images | 32.6 MB |
| `google/vit-base-patch16-224` | 330 MB |
| `sentence-transformers/all-MiniLM-L6-v2` | 87 MB |
| `openai/clip-vit-base-patch32` | 577 MB |
| CIFAR-10 test tarball | 170.5 MB |
| **total** | **≈ 1.20 GB** |

It does **not** download COCO, which is about 20 GB.

**Total compute**, with everything already cached: about **45 s** measured on a
12-thread Apple-Silicon CPU (torch 2.13, transformers 4.57). A Colab CPU runtime
has 2 vCPUs; budget **3–5 minutes** of compute there, plus however long 1.2 GB
takes to arrive. On a Colab T4 the compute falls under 30 s and the download
dominates completely.

**One constant governs the whole notebook.** `N_CATALOGUE = 200`. Every recall
below is over 200 candidates, and it must appear in the same sentence as the
recall, every time. A recall without its candidate-set size is not a number.

**Numbers in this script.** Every figure quoted was produced by running the real
pipeline over the real 200 images on CPU, with `SEED = 42`. Float32 on a GPU can
flip a near-tie, so read the recalls as ±1 entry (±0.5 percentage points) and the
means as good to the third decimal.

---

## Cell 1 — setup

**Prompt to type:**

> Setup cell. Import numpy, pandas, torch, transformers, PIL, pathlib. Print the
> python, torch and transformers versions. Seed numpy and torch with 42. Pick
> cuda if available, else mps, else cpu, and print which. Define
> `N_CATALOGUE = 200` with a comment saying every recall in this notebook is over
> that many candidates.

**Expect:** four version lines, one `device` line, and the constant defined. No
computation.

**Assert:** none.

**Annotate:** short

---

## Cell 2 — the catalogue

Markdown to put above it: two downloads, and neither of them is COCO. A 5 MB CSV
listing 5,000 COCO-2014 validation images with their human-written captions, then
the first 200 of those images by COCO id.

**Prompt to type:**

> Download
> `https://huggingface.co/datasets/nlphuji/mscoco_2014_5k_test_image_text_retrieval/resolve/main/test_5k_mscoco_2014.csv`
> into `datasets/app12/` if it is not already there, and read it. Sort by
> `cocoid`, take the first `N_CATALOGUE` rows, and for each one download
> `http://images.cocodataset.org/val2014/<filename>` into `datasets/app12/images/`
> if it is not already there. The `raw` column is a Python list literal of
> captions — parse it and collapse each caption's whitespace. Build a list of
> dicts with `sku` (`CAT-` plus the zero-padded 6-digit cocoid), `file` and
> `captions`. Print how many images the index lists and how many MB the
> catalogue occupies.

**Expect:** `the split index lists 5,000 images`, then
`catalogue: 200 entries, 32.6 MB on disk`.

**Assert:**

```python
assert len(catalogue) == N_CATALOGUE
assert all(len(e["captions"]) >= 2 for e in catalogue)
assert len({e["sku"] for e in catalogue}) == N_CATALOGUE
```

**⏱** About **1 minute** the first time on any runtime — it is 200 sequential
HTTP requests to `images.cocodataset.org`, so it is network-bound and a GPU buys
nothing. Instant on every later run. If the loop stalls, re-run it: it skips
files already on disk.

**Annotate:** full

* **Left open:** the prompt says *sort by `cocoid`, take the first 200* and never
  says why. It is a deterministic **rule**, not a sample. Nobody chose which 200
  images make retrieval look good, and that is the only defence you have against
  the accusation.
* **The usual student version:** `split.head(200)` on the frame as it was read,
  or `.sample(200)` with no seed. Both return 200 images. Measured on this file:
  the unsorted `head(200)` and the sorted `head(200)` share **9 images of 200**.
  Two students then report recalls over two different catalogues and spend an
  afternoon arguing about the models.
* **How you would catch it:** the whitespace clause is the other half. Of the
  1,000 captions attached to these 200 images, **154 change** under
  `" ".join(c.split())` — 143 carry a leading or trailing space, 16 carry a
  double space, and 5 contain a literal newline. A query that differs from a
  description only in whitespace is a bug you will chase in section 11 and not
  find.

---

## Cell 3 — one entry, and the two sides

Markdown above it: the five captions are **evidence**, not part of the product.
Caption 1 becomes the entry's written description, caption 2 becomes the
customer's query — two different people's sentences about the same picture.

**Prompt to type:**

> Print the first entry's sku and its first three captions. Then build three
> parallel lists: `images` (the PIL images, converted to RGB), `descriptions`
> (caption 1 of each entry) and `queries` (caption 2 of each entry).

**Expect:** `CAT-000042`, then three captions about a metal rack of shoes — the
second of which mentions a dog asleep on it, which is the point: different people
noticed different things.

**Assert:**

```python
assert len(images) == len(descriptions) == len(queries) == N_CATALOGUE
assert descriptions[0] != queries[0], "query and description must differ"
```

**⏱** Decoding 200 JPEGs takes **4.5 s** on CPU. Under the 20 s line, but it is
not free.

**Annotate:** short

> The second assert is one line and it is the difference between measuring
> retrieval and measuring string equality. Use the same caption on both sides and
> a hash table scores 100%.

---

## Cell 4 — the metric, and the baseline you do not have to simulate

**Prompt to type:**

> Write `ranks_of_truth(sim)` for a square similarity matrix whose correct answer
> for row *i* is column *i*: return the rank of the correct entry, 1 being best,
> with **ties counted against the model**. Write `report(name, sim)` printing
> Recall@1, @5, @10 and the median rank, always with the candidate count. Then
> print the random-ranking values for 200 candidates computed **arithmetically**
> as k/n — no simulation — and the expected rank. Finally show what a model that
> outputs a constant score for everything reports under a strict `>` and under
> `>=`.

**Expect:**

```
random ranking over 200 candidates, computed exactly:
  R@1     0.5%
  R@5     2.5%
  R@10    5.0%
  expected rank 100.5
  MRR 0.029

with a strict '>' a constant scorer reports R@1 = 100%
with '>=' it reports R@1 = 0.0%
```

**Assert:** inside `ranks_of_truth`, `assert sim.shape[0] == sim.shape[1]` before
reading a diagonal out of it.

**Annotate:** full

* **Left open:** the prompt asks for the tie rule but not for the demonstration.
  Ask for the demonstration. `(sim >= correct).sum(axis=1)` on an all-zeros
  matrix gives every query rank **200**, so R@1 = 0.0% — worse than the 0.5% of
  shuffling, which is correct, because a scorer that has learned nothing should
  not be rewarded for it. Swap to a strict `>` and the same matrix reports
  **R@1 = 100%**.
* **The usual student version:** never writing a tie rule at all, and inheriting
  one from `argsort`. `np.argsort` is not stable by default — it is introsort —
  so ties resolve on memory layout, and on a degenerate score matrix the diagonal
  frequently wins by accident. That is exactly the failure mode of a badly
  initialised encoder, which is the one you most need the metric to catch.
* **How you would catch it:** compute the baseline in closed form. One relevant
  entry among *n*, ranked uniformly, gives P(rank ≤ k) = k/n **exactly** — no
  seed, no noise, no run-to-run spread to argue about. A simulated baseline over
  200 candidates has a standard error you then have to reason about, for a number
  you could have written down.

### ✍️ Commit, before cell 5

On paper, before running another cell: the metric, the candidate-set size, the
Recall@1 you would call a working system, and the Recall@1 you expect from what
you are about to build. Shuffling scores **0.5%** at rank 1 and **5.0%** in the
top ten over **200** candidates. Your number belongs above those, and saying
*where* is the exercise.

---

## Cell 5 — encode the images

Markdown above it: an image-only encoder. A vision transformer trained on
ImageNet labels. No text appears anywhere in its training.

**Prompt to type:**

> Load `google/vit-base-patch16-224` with `AutoImageProcessor` and `ViTModel`,
> pass `add_pooling_layer=False`, move it to `device` and put it in eval mode.
> Encode the 200 catalogue images in batches of 32 under `torch.no_grad()`,
> taking the CLS token from `last_hidden_state`. Stack into a float64 array `V`
> and print the shape and the elapsed time.

**Expect:** `images encoded: (200, 768) in 13.0s` on a 12-thread CPU. On a T4,
about 2 s.

**Assert:** `assert V.shape == (N_CATALOGUE, 768), V.shape`

**⏱** Model download **20–60 s** the first time (330 MB). Inference: **13 s**
measured on a 12-thread CPU for 200 images; budget **60–90 s** on a 2-vCPU Colab
CPU runtime, about 2 s on a T4.

**Annotate:** full

* **Left open:** that no text appears anywhere in this model's training. It is a
  vision transformer fitted to ImageNet labels. That single fact is what cells 7,
  9 and 10 are about, and the prompt never mentions it.
* **The usual student version:** dropping `add_pooling_layer=False` — it is the
  default — and then reaching for `outputs.pooler_output` because it sounds like
  the summary vector. **Checked against the checkpoint:** `model.safetensors` for
  `google/vit-base-patch16-224` holds 200 tensors and **not one of them is named
  `pooler`**. Asking for the pooler builds `pooler.dense.weight` and
  `pooler.dense.bias` fresh, drawn N(0, 0.02²). Your image embedding is then a
  random projection of the CLS token, and every shape check passes.
* **How you would catch it:** it is not silent. transformers prints
  `Some weights of ViTModel were not initialized from the model checkpoint at
  google/vit-base-patch16-224 and are newly initialized: ['pooler.dense.bias',
  'pooler.dense.weight']`. It is the single most-scrolled-past message in the
  library. And the trap has a second floor: because cell 1 called
  `torch.manual_seed(42)`, that random layer is **reproducible** — two loads
  after the same seed give bit-identical weights (checked). So the wrong number
  is stable across re-runs, which is precisely how it survives review.

---

## Cell 6 — encode the queries

Markdown above it: a text-only encoder. Mean-pooled sentence embeddings. No image
appears anywhere in its training either. Two encoders, two worlds, no contact.

**Prompt to type:**

> Load `sentence-transformers/all-MiniLM-L6-v2` with `AutoTokenizer` and
> `AutoModel` onto `device` in eval mode. Write `minilm(sentences)` that
> tokenises in batches of 64 with padding and truncation at 128, runs the model
> under `no_grad`, and mean-pools `last_hidden_state` **using the attention
> mask** as the divisor. Return float64. Apply it to `queries` as `T`.

**Expect:** `queries encoded: (200, 384)`, in about **0.6 s** on CPU.

**Assert:** `assert T.shape == (N_CATALOGUE, 384), T.shape`

**Annotate:** short

> The mask is the whole cell. `last_hidden_state.mean(1)` averages the padding
> too, so a 6-word caption padded to 20 is diluted by 70% and a 20-word one is
> not. Every shape check still passes. This is Lecture 22's pooling bug in its
> third costume — Lecture 21 had it as `padding_idx`, Lecture 22 as the pooled
> review. If your pooling line has no `attention_mask` in it, it is wrong.

---

## Cell 7 — now take the cosine

Markdown above it, one line: **reviewer question 3 — what is the shape here?**

**Prompt to type:**

> Print the shapes of `V` and `T`, then try `V @ T.T` inside a try/except and
> print the ValueError instead of letting the notebook stop.

**Expect:**

```
V (200, 768)   T (200, 384)

ValueError: matmul: Input operand 1 has a mismatch in its core dimension 0,
with gufunc signature (n?,k),(k,m?)->(n?,m?) (size 384 is different from 768)
```

**Assert:** none — the exception *is* the output.

**Annotate:** short

> Say the sentence the error is really saying: there is no cosine between them,
> because there is no inner product between them. 768 against 384 is a loud
> failure. The quiet version is two encoders that happen to share a dimension,
> where the multiplication succeeds and means nothing.

---

## Cell 8 — force the dimensions to agree, three ways

Markdown above it: three ways, so nobody can blame the particular fix — a random
Gaussian projection 768 → 384 (Johnson–Lindenstrauss, thread 5, Lecture 10);
keeping the first 384 image coordinates; zero-padding the text to 768.

**Prompt to type:**

> Write `unit(x)` that L2-normalises rows. With
> `rng = np.random.default_rng(SEED)`, draw `R` of shape (768, 384) from
> N(0, 1/384). Build three 200×200 cosine matrices between the images and the
> queries: one via `R`, one by truncating the image vectors to their first 384
> coordinates, one by zero-padding the text to 768. Report each with `report`,
> rows being the text queries, and print the arithmetic random baseline
> underneath.

**Expect:** all three flat on the baseline, and the median rank near 100.5:

```
JL projection 768->384       R@1   1.0%  R@5   4.0%  R@10   6.0%   median rank  96 of 200
truncate image to 384        R@1   0.5%  R@5   2.0%  R@10   5.5%   median rank  89 of 200
zero-pad text to 768         R@1   0.0%  R@5   2.5%  R@10   4.5%   median rank  88 of 200

random ranking over 200 candidates: R@1 0.5%  R@5 2.5%  R@10 5.0%
```

**Assert:** `assert S.shape == (N_CATALOGUE, N_CATALOGUE), (name, S.shape)` for
each of the three, before reporting.

**Annotate:** short

> Two things to say out loud. **One:** transpose before reporting. These matrices
> are built image-major and `report` reads rows as queries — a silent transpose
> measures image-to-text and prints the label *text-to-image*. **Two:** the JL
> theorem promised that a random projection nearly preserves the geometry **of
> the image space**. It promised nothing about aligning that geometry with
> anything else, and the three lines above are what that distinction costs.

---

## Cell 9 — a colleague's prompt

> **Staging note — do not skip this.** Put **no warning** above this cell. No ⚠,
> no "read before running", no "under-specified". The markdown says only: *a
> colleague was asked to check whether the images and their captions look
> similar, and sent back this.* Run it, and **write the number down on paper**
> before you read cell 10. If you flag the trap here, nobody falls in and the
> section teaches nothing.

**Prompt to type** — this is the whole of what your colleague was given, typed as
they would have typed it:

> Encode the catalogue images with a ViT and the captions with a sentence
> transformer, then check whether an image and its caption are similar.

Reconstructing what came back: project the image vectors with `R`, normalise
both sides, take the cosine of each **matched** pair, print the mean and the
share above 0.05.

**Expect:**

```
mean similarity of an image and its caption: 0.007
18% of pairs score above 0.05
```

**Assert:** none. That is the finding.

**Annotate:** full

* **Left open:** everything that would make 0.007 mean something. The cell
  computed the **200 numbers on the diagonal** and never touched the other
  200 × 200 − 200 = **39,800**. It reports a **level** where only a
  **difference** means anything.
* **The usual student version:** the version above *is* the usual version — it is
  what the prompt literally asks for, and it runs, and it prints. The student
  addition is reading `0.007` as small and concluding the models are broken, or
  reading `18% above 0.05` as encouraging. Neither reading is available, because
  no unrelated pair was ever scored. Note also the second silent default: **0.05
  was never in the prompt.** A threshold nobody asked for, applied to a quantity
  nobody calibrated.
* **How you would catch it:** reviewer question 5 — *what is the default I did
  not ask for?* — in an unusual form. Here the default nobody asked for is the
  **missing control group**. Any time a cell reports a similarity, a distance or
  a score as a bare level, the next question is what that number is for two
  things you know are unrelated.

> **On honesty:** the reconstructed cell quietly uses `R` from cell 8. The
> original prompt asked for no projection, and as cell 7 proved it could not have
> produced any number at all without one. Say so in the markdown. "What the weak
> prompt returns" is really "what the weak prompt returns once somebody has
> silently patched the one part that crashed" — and the silent patch is itself
> worth a sentence.

---

## Cell 10 — the control that was missing

Markdown above it — **now** the ⚠, and the contrast, and the reader's number
still on paper: *what does an unrelated pair score?*

**Prompt to type:**

> Take the full 200 × 200 JL similarity matrix. Split it into the diagonal
> (matched pairs) and the off-diagonal (unrelated pairs). Print the mean and sd
> of each with `ddof=1` and the count beside each, then their difference and
> Cohen's d using the pooled sd.

**Expect:**

```
matched pairs    mean +0.0070   sd 0.0503   (n = 200)
unrelated pairs  mean +0.0036   sd 0.0515   (n = 39,800)
difference       +0.0034   Cohen's d +0.066
```

The diagonal mean is the number from cell 9. An image and its own caption score
**0.0034** above an image and a stranger's caption, on a scale whose sd is
**0.05**. Write **d = 0.066** on your sheet; cell 15 asks for it.

**Assert:** none.

**Annotate:** full

* **Left open:** why it happened, which is the part worth the lecture. Both
  encoders work. The ViT separates images from images; MiniLM separates sentences
  from sentences. Neither ever saw a single (image, sentence) pair, so neither has
  any reason to place them anywhere in particular relative to each other.
* **The usual student version:** concluding the models are broken and going to
  look for a better checkpoint. They are not broken. Apply any rotation to the
  image space: every image–image cosine is unchanged, so the ViT's loss is
  unchanged — and every image–text cosine changes. The loss **cannot tell those
  worlds apart**, so it did not pick one. There is nothing to fix at the
  checkpoint; the missing thing is an objective that compared the two towers.
* **How you would catch it:** `ddof=1` on both variances, and **n printed beside
  each**. 200 against 39,800 is a wildly asymmetric comparison and the reader must
  see it — the off-diagonal mean is pinned to three decimal places and the
  diagonal mean is not, and that asymmetry is why d, not the raw gap, is the
  quantity to quote.

**The corrected specification, for the markdown underneath:**

> Build the full 200 × 200 cosine matrix. Report the mean and sd of the diagonal
> **and** of the off-diagonal with their counts, their standardised difference,
> and Recall@1, 5 and 10 for text-to-image ranking over 200 candidates. Print the
> exact random-ranking values k/n beside them. Break ties against the model.

Every clause is there because leaving it out produced a number somebody believed.

---

## Cell 11 — a jointly trained pair

Markdown above it: same architecture family, one difference — both towers were
trained by a single objective that compared them.

**Prompt to type:**

> Load `openai/clip-vit-base-patch32` with `CLIPModel` and `CLIPProcessor` onto
> `device` in eval mode. Print the projection dim and the parameter count in
> millions. Write batched `clip_images` and `clip_text` helpers using
> `get_image_features` and `get_text_features` under `no_grad`, returning float64.
> Encode the 200 images and the 200 queries, keep the raw outputs, and normalise
> both with `unit()`.

**Expect:** `projection dim 512`, `parameters 151M`, then
`image features (200, 512)   text features (200, 512)`.

**Assert:**
`assert I_raw.shape == Q_raw.shape == (N_CATALOGUE, clip.config.projection_dim)`

**⏱** Download **1–2 min** the first time (577 MB). Inference: **9.4 s** measured
on a 12-thread CPU for 200 images plus 200 sentences; budget **45–60 s** on a
2-vCPU Colab CPU runtime, a couple of seconds on a T4.

**Annotate:** short

> `get_image_features` and `get_text_features` do **not** return unit vectors —
> measured on these 200 items, the image feature norms run **8.66 to 11.44** and
> the text norms **6.28 to 11.25**. The cosine of unnormalised vectors is a dot
> product weighted by two arbitrary magnitudes. Skipping `unit()` still runs,
> still returns a matrix, still produces a ranking: **R@1 72.5% instead of
> 75.0%** over 200 candidates, checked. That is Lecture 24's assistant failure,
> and 2.5 points is more than you would guess for a line that looks cosmetic.

---

## Cell 12 — a known-answer test, before the real one

Markdown above it: step 4 of the working loop — *test against a case whose answer
you know*. The catalogue carries no labels, so nothing in it can tell you the
model is loaded correctly and preprocessing its inputs the way it was trained to.
CIFAR-10 can.

**Prompt to type:**

> Load the CIFAR-10 test split with torchvision into `datasets`, take the first
> 500 images as `cifar_images` with labels `y` and the class-name list, and encode
> them with `clip_images`, normalised. Print the count and the feature shape, and
> say "500" in the printout.

**Expect:** `500 CIFAR-10 images encoded: (500, 512)`, with
`classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog',
'horse', 'ship', 'truck']`.

**Assert:**

```python
assert len(cifar_images) == len(y) == N_CIFAR
assert len(classes) == 10, classes
```

**⏱** Download **1–2 min** the first time (170.5 MB). Inference: **14.8 s**
measured on a 12-thread CPU for 500 images; budget **60–90 s** on a 2-vCPU Colab
CPU runtime.

**Annotate:** short

> `N_CIFAR = 500` is a named constant for the same reason `N_CATALOGUE` is: a
> subset is a subset even when the point of the cell is not the number, and the
> accuracy must be quoted with it. Going straight to the retrieval numbers is the
> mistake here — if the processor were mismatched to the checkpoint, retrieval
> would degrade gracefully and silently and nothing in the catalogue could catch
> it.

---

## Cell 13 — the classifier is ten sentences

**Prompt to type:**

> For each of the templates `"{}"`, `"a photo of a {}"` and
> `"a low-resolution photo of a {}"`, build the ten class sentences, encode them
> with `clip_text`, normalise, take the argmax of the image-text cosine and print
> the accuracy over the 500 images with chance beside it. Print all three, not
> just the best.

**Expect:**

```
'dog'                                  accuracy  88.0%   over 500 images   (chance 10.0%)
'a photo of a dog'                     accuracy  90.6%   over 500 images   (chance 10.0%)
'a low-resolution photo of a dog'      accuracy  90.6%   over 500 images   (chance 10.0%)
```

**Assert:** none.

**Annotate:** full

* **Left open:** that there is no fitted parameter anywhere in this cell. Nothing
  was trained, nothing was tuned, and the classifier is ten English sentences.
  Which sentences is a **choice you made**, and the prompt never says who made it
  or on what evidence.
* **The usual student version:** running one template, or running three and
  reporting the best. Both are the same error wearing different clothes. The best
  template here is chosen by looking at its accuracy on the 500 images you then
  report the accuracy on — a hyperparameter selected on the test set. It is the
  failure from application 3 (Lecture 6), where seventeen models were scored on
  the test set and the best one reported. Note also that **two templates tie at
  90.6%**, so "the best template" is not even a well-defined object here.
* **How you would catch it:** quote the **spread**, not the maximum. Across these
  three templates the range is **2.6 percentage points** over 500 images. A
  "zero-shot" number is a number for **one particular sentence**; if you report
  90.6% without saying which sentence produced it and what the others gave, you
  have reported a selection, not a measurement.

---

## Cell 14 — back to the catalogue

Markdown above it: same 200 images, same 200 queries, same metric, same tie rule,
same baseline. Only the encoders changed.

**Prompt to type:**

> Build `sim_clip = Q @ I.T`, rows being the text queries. Report three rows with
> the existing `report` function: the ViT+MiniLM JL route (transposed so rows are
> queries), CLIP text-to-image, and CLIP image-to-text. Print the arithmetic
> random baseline and the expected rank underneath, and say "over 200 candidates,
> ties against the model" above the table.

**Expect:**

```
over 200 candidates, ties against the model:

ViT + MiniLM, JL projection        R@1   1.0%  R@5   4.0%  R@10   6.0%   median rank  96 of 200
joint dual encoder, text->image    R@1  75.0%  R@5  97.0%  R@10  99.0%   median rank   1 of 200
joint dual encoder, image->text    R@1  73.5%  R@5  97.0%  R@10 100.0%   median rank   1 of 200

random ranking (arithmetic)        R@1   0.5%  R@5   2.5%  R@10   5.0%   expected rank   100 of 200
```

**Assert:** `assert sim_clip.shape == (N_CATALOGUE, N_CATALOGUE)`

**Annotate:** short

> Three things the table has to carry, and they are all in the format string.
> **One variable changed** — same rows, same columns, same metric, same tie rule,
> same baseline, only the encoders. **Both directions are printed**, because
> text-to-image and image-to-text are different numbers on the same matrix
> (75.0% and 73.5% here) and reporting one as "retrieval accuracy" hides which
> was measured. **The baseline sits in the table**, because 75.0% is only legible
> against 0.5%.

---

## Cell 15 — the same control, on the joint model

**Prompt to type:**

> Run exactly the cell-10 analysis on `sim_clip`, unchanged — same split, same
> `ddof=1`, same counts printed beside each mean, same number of decimals on
> Cohen's d.

**Expect:**

```
matched pairs    mean +0.3037   sd 0.0346   (n = 200)
unrelated pairs  mean +0.1516   sd 0.0395   (n = 39,800)
difference       +0.1522   Cohen's d +4.100
```

Against cell 10's **+0.066**. Same computation, same catalogue, same 39,800
unrelated pairs: **d goes from 0.066 to 4.10**. That gap is the entire
application in one number.

**Assert:** none.

**Annotate:** short

> "Unchanged" is a real instruction and it is easy to fail. The version in the
> current notebook drops the `(n = ...)` annotations and prints d as `.2f` where
> cell 10 printed `+.3f` — so the two numbers the reader is asked to compare
> arrive in different formats, from a cell whose own specification says
> *identical*. When you fix something, re-run the diagnostic that detected it,
> byte for byte. A new diagnostic on the fixed version proves nothing about the
> old one.

---

## Cell 16 — the other route, and where it breaks

Markdown above it: some entries carry a written description, and for those the
previous application's semantic search applies directly — same space on both
sides, because both sides are sentences. Then model the catalogue as it really
arrives.

**Prompt to type:**

> Embed `descriptions` and `queries` with `minilm`, normalise both, and report
> text-query-to-text-description retrieval. Then delete the descriptions of the
> entries where `i % 10 < 3` by setting those columns to `-inf` so they are not in
> the index at all, and report again. Finally print Recall@1 restricted to those
> 60 entries, with and without their description, and Recall@1 on the same 60
> entries via `sim_clip`.

**Expect:**

```
text query -> text description     R@1  61.5%  R@5  89.0%  R@10  96.0%   median rank   1 of 200
...with 60 descriptions deleted    R@1  46.0%  R@5  65.5%  R@10  68.5%   median rank   2 of 200

R@1 on those 60 entries: with a description 66.7%, with none 0.0%
R@1 on the same 60 via the joint image route: 80.0%
```

**Assert:** `assert len(blanked) == 60, len(blanked)`

**⏱** Re-encoding 400 sentences with MiniLM is **about 1 s** on CPU. Under the
line.

**Annotate:** full

* **Left open:** that the answer is not "worse". It is **zero, structurally**. An
  entry with no text is not in a text index; `-inf` makes its rank 200 for
  certain, and no amount of better embedding moves it. 66.7% → 0.0%, and the
  image route on the *same 60 rows* is untouched at 80.0%, because it never read
  a description. That contrast is the number Lecture 24 repairs.
* **The usual student version:** imputing `""` for the missing descriptions
  instead of `-inf`. MiniLM happily embeds the empty string — it returns the
  mean-pooled `[CLS]`/`[SEP]` representation, a perfectly ordinary unit vector —
  so the blanked entries land at some arbitrary point in the index and a
  structural zero becomes a small positive number that reads like a modelling
  result.
* **How you would catch it:** the aggregate line is a trap and it is worth
  building deliberately. `46.0%` is **not** a recall over 200 candidates — 60
  candidates were removed, so it is 200 rows scored against 140 reachable
  entries, and `report` still prints *of 200*. Check the two halves separately
  and the direction is obvious: on the 60 blanked rows R@1 falls 66.7% → 0.0%,
  and on the **other 140 rows it rises, 59.3% → 65.7%**, because deleting 60
  competitors made their job easier. The headline `61.5% → 46.0%` averages a
  structural zero against a free gift. Matched rows, or nothing.

---

## Section 12 — red-team, unchanged

Fifteen minutes, swap notebooks. The five reviewer questions, then the six ways to
leak a *retrieval* evaluation specifically. Two of them now have a number attached
from this notebook and should be quoted with it:

* *ties broken in the model's favour* — a constant scorer reports **R@1 = 100%**
  under a strict `>` and **0.0%** under `>=` (cell 4);
* *recall reported without the candidate-set size* — cell 16's `46.0%` is over
  **140** reachable candidates while the printout says **200**.

Before leaving: the best Recall@1 obtained, **over how many candidates**, and one
query the system got wrong with an explanation.

---

## Annotation budget, as built

Sixteen code cells. Sixteen short specification boxes. **Seven** full three-bullet
annotations — cells **2, 4, 5, 9, 10, 13, 16** — inside the five-to-eight budget
of GUIDELINES §6.1. The current notebook carries **sixteen**, one on every box.

Every "usual student version" bullet above names a checked library default or a
measured failure: `add_pooling_layer` defaulting to `True` against a checkpoint
with no pooler tensors (checked); `np.argsort`'s unstable default (documented);
`last_hidden_state.mean(1)` over padding; the 9-of-200 overlap between sorted and
unsorted `head(200)` (measured); the 2.6-point template spread (measured); the
59.3% → 65.7% rise on unblanked rows (measured).

---

## Defects found in the current notebook

`notebooks/lecture-23.ipynb`, 50 cells, 16 of them code. Everything below marked
**checked** was verified with `python3` against the notebook's own data — the
cached split index and the 200 images under `notebooks/datasets/app12/`, and the
three checkpoints in the local HuggingFace cache — by running the real pipeline
on CPU with `SEED = 42`.

### 1. A prose figure that its own cell cannot produce — **checked**

Cell 27 (the ⚠ prompt box) says:

> **The usual student version:** reading 'mean similarity 0.14' as evidence of
> anything.

Cell 28 prints `mean similarity of an image and its caption: 0.007`. Measured on
the real 200 images: **0.007**, with 18% of pairs above 0.05. **0.14 is 20× the
value the cell produces**, and it is the only concrete number offered to the
reader in the section the whole lecture turns on. It cannot be numerical noise:
the quantity is a cosine between a random projection of a ViT CLS token and a
MiniLM embedding, which concentrates at zero by construction. GUIDELINES §1.1.

### 2. Zero stored outputs, so §1.2 cannot be satisfied anywhere — **checked**

All 16 code cells have `execution_count: null` and `outputs: []`. Every figure in
the markdown — `0.14`, "Section 7's d was near zero", "all three land at chance",
"zero, structurally" — is therefore unreconcilable against a stored output, which
is what §1.2 requires and what `tools/check_notebook_numbers.py` checks. The
claims that *are* true (checked: d = 0.066 and 4.10; all three routes at chance;
the structural zero) are true by luck of authorship rather than by verification.

### 3. "No warning" is false, and contradicted three lines later — **checked**

Cell 15's constraint reads: *"asking for one gives you a RANDOMLY INITIALISED
layer with no error and no warning."* Loaded with `add_pooling_layer=True`,
transformers 4.57 prints:

```
Some weights of ViTModel were not initialized from the model checkpoint at
google/vit-base-patch16-224 and are newly initialized: ['pooler.dense.bias',
'pooler.dense.weight']
```

The same box's third bullet then says *"when a checkpoint warns that weights were
newly initialised, read it"* — advice that is unusable if, as the constraint
states, there is no warning. The source module's own code comment (line 284) says
only "with no error", which is correct; the prompt box added "and no warning".
Confirmed separately: `model.safetensors` holds 200 tensors and none is named
`pooler`. GUIDELINES §1.1, §3.2.

### 4. A mismatched comparison, unflagged — **checked**

Cell 48 reports two rows the reader is invited to compare:

```
text query -> text description     R@1  61.5%   median rank 1 of 200
...with 60 descriptions deleted    R@1  46.0%   median rank 2 of 200
```

The second row is not scored against 200 candidates. Sixty columns were set to
`-inf`, so 200 queries are ranked against **140 reachable entries**, and `report`
prints *of 200* regardless because `n = len(r)` counts rows. Splitting the rows:
the 60 blanked entries fall **66.7% → 0.0%** and the other 140 **rise, 59.3% →
65.7%**, because deleting 60 competitors made their task easier. The headline
15.5-point drop averages a structural zero against a free gift, and the notebook's
own standing rule — *a recall without its candidate-set size is not a number* — is
broken by its own cell. GUIDELINES §2.1, and §8.2's observation that the unlabelled
defect is the one that catches people.

### 5. "Under-specified in exactly one place" — **checked, and it is at least four**

Cell 26. By the notebook's own content the quoted weak prompt is under-specified
in at least four places: no control group (section 7); which caption goes on which
side (section 2 spends a whole cell and an assert on it); how 768 and 384 are to
be made comparable (section 6 proves it cannot be done without a choice); and what
"similar" means, since cell 28 silently introduces a `0.05` threshold that appears
nowhere in the quoted prompt. GUIDELINES §7.3 — "notice N things" must have exactly
N findable things.

### 6. "What the weak prompt returns" is not what the weak prompt returns — **checked**

Cell 28 opens `Vp = unit(unit(V) @ R)`, using the JL matrix `R` defined in cell 25.
The quoted prompt asks for no projection, and cell 22 has just demonstrated that
without one the multiplication raises `ValueError`. The cell therefore shows what
the weak prompt returns *after* an unacknowledged repair. GUIDELINES §4.4 — do not
make blanket provenance claims.

### 7. The trap is announced four times before it fires — **checked by counting**

Cell 0 ("Cells marked **⚠ read before running** contain a defect on purpose"),
cell 26's heading ("## 7 · ⚠ Read before running — the assistant failure"), cell
26's body ("Under-specified in exactly one place. Find it before you run the
cell"), and cell 27's prompt label ("⚠ what the weak prompt returns"). Four flags
above a four-line cell. GUIDELINES §8.1 — the preferred shape is to let it run
unannounced, have the reader write the number down, then open the next section
with the ⚠.

### 8. "Run the IDENTICAL analysis" — and it is not identical — **checked**

Cell 44's constraint demands the identical computation as section 7, and its own
section-7 counterpart (cell 30) insists on *"n printed beside each"*. Cell 45 drops
both `(n = ...)` annotations and formats Cohen's d as `.2f` where cell 31 uses
`+.3f`. The reader compares `+0.066` against `4.10` — different precision,
different sign convention, missing counts. GUIDELINES §2.1, §3.2.

### 9. Two cross-references off by two cells — **checked by counting**

* Cell 10: *"you will see why three cells from now"* — the tie demonstration is at
  the foot of cell 12, **two** cells later.
* Cell 21: *"reaching for a projection immediately, which is the next cell"* — the
  next cell is 22 (the `ValueError`); the projection is cell **25**, four later.

GUIDELINES §3.3.

### 10. Two naming systems for the same pointer, two cells apart — **checked**

Cell 39 says *"a hyperparameter chosen on the test set — application 3, in a new
costume"*; cell 41 says *"— Lecture 6, in a new costume"*. Both resolve to the
same content (`LECTURES.md`: applications run in consecutive pairs, so application
3 is Lectures 5–6, and Lecture 6 §5 is the choose-on-the-test-set demonstration),
but the reader meets two vocabularies for one reference within three cells. The
course's house style elsewhere is "application N" (Lectures 9, 17, 18, 21, 22, 24).
Cell 23's *"thread 5"* has the same problem in reverse: it resolves to Lecture 10
(`LECTURES.md`, thread 5 = SVD, PCA and Johnson–Lindenstrauss), but Lecture 21
writes the fuller "thread 5, from Lecture 10" and this notebook does not.

### 11. Forward reference to a list twenty cells away — **checked by counting**

Cell 29: *"In the five reviewer questions this is number 5."* The five questions
are first listed in cell 49, the last cell of the notebook. A reader at cell 29
cannot check the claim. (It is correct: #5 is *"What is the default I did not ask
for?"*)

### 12. The header's download budget omits 170.5 MB — **checked**

Cell 0 lists "A 5 MB index …, then 200 individual images (about 30 MB), then three
model checkpoints (about 1 GB in total)". Measured on disk: index 5.0 MB, images
32.6 MB, ViT 330 MB + MiniLM 87 MB + CLIP 577 MB = 994 MB — all three accurate.
But CIFAR-10 is a fourth download of **170.5 MB**, stated in cell 36 and missing
from the header. Actual first-run total **1,202 MB** against a stated ~1,035 MB, a
16% understatement, in the one paragraph a reader on a metered connection reads.

### 13. The header's wall clock excludes its own upper bound — **checked**

Cell 0: *"three to five minutes end to end."* Summing the notebook's own
per-section budgets — catalogue ~1 min (cell 4); ViT 20–60 s + ~15 s (cell 14);
CLIP 1–2 min + a few seconds (cell 33); CIFAR 1–2 min + ~10 s (cell 36) — gives
**3.8 to 6.5 minutes**, before the untimed MiniLM download and the untimed
re-encode in section 11. The stated range does not contain its own maximum.

### 14. No CPU figure anywhere — **checked**

GUIDELINES §7.1 requires the CPU number for every cell over ~20 s. Cell 0 gives
"on a Colab GPU runtime" and no other runtime is costed. Cell 3's device selection
explicitly supports `cpu`, so the notebook expects CPU readers and never budgets
for them. Measured on a 12-thread Apple-Silicon CPU: decode 200 JPEGs 4.5 s, ViT
200 images 12.6 s, MiniLM 200 sentences 0.6 s, CLIP 200 images + 200 texts 9.4 s,
CLIP 500 CIFAR images 14.8 s — about 45 s total, which on a 2-vCPU Colab CPU
runtime is a 3–5 minute compute budget the notebook never states. This is the
defect class that cost the literal reader 4.5 of 6 exercises in Lecture 19.

### 15. Sixteen full annotations where the budget is five to eight — **checked**

All 16 prompt boxes (cells 2, 5, 8, 11, 15, 18, 21, 24, 27, 30, 34, 37, 39, 42,
44, 47) carry the complete three-bullet "Watch this prompt" block. GUIDELINES
§6.1. Also §6.3: **six** of the 16 boxes have no `check ·` clause at all (cells 2,
21, 27, 30, 39, 44), and the `check` slot is the one that structurally forces an
expected answer.

### 16. Nothing is marked examinable — **checked**

GUIDELINES §8.3 requires every section to carry one of *examinable*, *not
examinable — engineering*, or *beyond the book, for context*. The string
"examinable" occurs exactly once in the whole notebook, inside a code comment in
cell 3 (`# Not examinable: version hygiene`), where no reader looking for section
labels will find it. Twelve sections, zero labels.

### 17. "The five human-written captions each" is wrong for 10 rows — **checked**

Cell 4. Parsing the cached CSV: 4,990 of the 5,000 rows carry five captions and
**10 carry six**. Harmless for this notebook — all 200 catalogue entries have
exactly five (checked) — and the code's assert is the tolerant `>= 2`, but the
prose states a property of the file that the file does not have. GUIDELINES §1.1.

### Checked clean

* **§5.1 / §5.2** — no markdown line indented ≥ 4 spaces outside a fence, and no
  fence marker indented at all, across all 34 markdown cells. Verified by script.
* **§3.1** — the notebook contains no ```` ```python ```` block in markdown, so
  there is no quoted code that could fail to exist in a cell.
* **§4.1** — no name is bound to two different kinds of object across cells.
  Verified by walking the AST of all 16 code cells: the only names bound in more
  than one cell are `S`, `matched`, `unrelated`, `pooled` (ndarray/float in both
  places) and the function-local `b`, `i`, `out`.
* **§4.2** — no training cell exists; every cell is inference or arithmetic, and
  all are idempotent, including cell 25, whose `rng` is re-seeded inside the cell.
* Arithmetic in prose: `200 × 200 − 200 = 39,800` ✓; `1/200 = 0.5%` and
  `10/200 = 5.0%` ✓; `i % 10 < 3` gives exactly 60 of 200 ✓; expected rank
  `(n+1)/2 = 100.5` ✓; a constant scorer reports 100% under `>` and 0.0% under
  `>=` ✓; the split index has exactly 5,000 rows ✓; the 200 images occupy 32.6 MB
  ✓ ("about 30 MB"); the CSV is 5.01 MB ✓ ("5 MB"); CLIP is 577 MB ✓ ("about
  600 MB"); CIFAR-10 is 170.5 MB ✓ ("170 MB"); `clip.config.projection_dim` is
  512 ✓; `get_image_features` returns non-unit rows ✓ (norms 8.66–11.44).
* Cross-lecture references that **do** resolve: *"the next lecture's assistant
  failure"* → `lecture_24.py` §6, which is indeed the unnormalised-features
  failure ✓; *"the padding bug from the previous application in a third costume"*
  → Lecture 21 (`padding_idx`) then Lecture 22 (mask-less mean pool) ✓;
  *"the previous application's semantic search"* → Lecture 22 ✓; *"thread 5"* →
  Lecture 10 ✓; *"see section 3 — a constant scorer then reports 100%"* ✓.

### Not checked

* Whether the two encoder claims about training data are exactly true as stated
  ("no text appears anywhere in this model's training" for
  `google/vit-base-patch16-224`, "no image" for MiniLM). Both are correct as far
  as the model cards go, but I did not audit the pretraining corpora.
* Wall-clock on an actual Colab runtime, GPU or CPU. Every timing above was
  measured on a 12-thread Apple-Silicon CPU with the checkpoints already cached;
  the Colab CPU figures in this script are scaled estimates and are labelled as
  such.
* Download times for the checkpoints and the COCO images. Sizes are measured from
  disk; durations depend on the network.
* Whether the notebook passes restart-and-run-all in Colab. I read the dependency
  order cell by cell and found no forward reference — every name is defined before
  use — but I did not execute the notebook itself, only a faithful reconstruction
  of its pipeline as a script.
