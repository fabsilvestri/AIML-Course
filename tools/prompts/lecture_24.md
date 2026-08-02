# Lecture 24 — Closing the loop, and closing the course

**Prompt script.** Sixteen code cells, in order. Type the prompt, read the
output against **Expect**, run **Assert**, move on. Seven cells carry the full
three-bullet annotation; the other nine carry the specification only.

Source module: `tools/notebooks/lecture_24.py`. Current render:
`notebooks/lecture-24.ipynb`.

---

## Before you start

**What this downloads, cold.** CLIP 577 MB, the BLIP captioner 944 MB, the Qwen
instruct model 942 MB, MiniLM 87 MB — **2.49 GB of checkpoints** — plus 31 MB of
images and a 4.8 MB index. It does not download COCO. If you ran lecture 23 in
this runtime, CLIP and the images are already on disk and only the two 1 GB
checkpoints are new.

**Wall clock, measured CPU-only** on an Apple-silicon laptop, torch 2.13.0,
transformers 4.57.3, with a **warm** checkpoint cache:

| Cell | What | Measured |
|---|---|---|
| 3 | CLIP load + encode 200 images + 200 queries | 2.3 s + 3.8 s + 0.6 s |
| 12 | MiniLM load + encode 400 sentences | 1.7 s + 1.7 s |
| 13 | BLIP load + 60 captions, `num_beams=3` | 7 s + **98 s** |
| 15–16 | Qwen load + 12 generations of ≤140 tokens | 2 s + **19 s** |

Everything else is numpy on a 200×200 matrix and is instantaneous. **On a cold
cache add the download time for 2.49 GB**, which on a free Colab CPU runtime is
usually the larger of the two costs. Nothing here trains, so there is no GPU
requirement — a laptop CPU finishes the whole notebook in roughly three minutes
of compute once the checkpoints are down.

**One state hazard, stated once.** `rng` is created in cell 5 and *consumed* by
it. Cell 9 draws from the same `rng`. So cell 9 is **not idempotent**: run it
four times and the B = 2 loss reads 0.015, 0.008, 0.008, 0.024 — measured. The
only reproducible way to read cell 9's table is restart-and-run-all. See §7.2 at
the end of this script for the two exercises this affects.

---

# Section 1 · Setup and the catalogue

*Examinable: no — engineering.*

## Cell 1 — setup
**Prompt to type:**
> print the python, torch and transformers versions, set numpy and torch seeds
> to 42, pick cuda / mps / cpu and print which, and set N_CATALOGUE = 200

**Expect:** four version/device lines and nothing else. On a laptop the device
line says `mps` or `cpu`; the notebook needs no GPU.
**Assert:** none.
**Annotate:** short

## Cell 2 — the same catalogue, rebuilt
**Prompt to type:**
> download the mscoco 5k test index from huggingface into `datasets/app12`,
> sort by cocoid, take the first 200 rows, download each image from
> `images.cocodataset.org/val2014/` if it isn't already on disk, and build a
> list of dicts with a SKU `CAT-{cocoid:06d}`, the file path and the five
> captions whitespace-normalised. Caption 0 is the description, caption 1 is
> the query.

**Expect:** `catalogue: 200 entries`. Under a minute cold, instant warm.
**Assert:** `len(catalogue) == 200` and `len({e["sku"] for e in catalogue}) == 200`.
**Annotate:** short

Sorting by `cocoid` and slicing is what makes this notebook and lecture 23 the
same corpus. Entry 0 is `CAT-000042`; if yours is not, the rule changed.

## Cell 3 — the embeddings, raw ones kept
**Prompt to type:**
> encode the 200 images and the 200 queries with
> `openai/clip-vit-base-patch32`, batched, and give me float64 numpy arrays

**Expect:** two `(200, 512)` arrays. **Their rows are not unit vectors** — image
lengths run 8.66 to 11.44, text lengths 6.28 to 11.25. Keep the raw arrays as
`I_raw`, `Q_raw` and add `I, Q = unit(I_raw), unit(Q_raw)` yourself.
**Assert:** run `assert np.allclose(np.linalg.norm(I_raw, axis=1), 1.0)` **on
what came back**. It fails. Write down that it failed. Then assert unit rows on
`I` and `Q`, where it passes.
**Annotate:** full

* **Left open:** the prompt says "encode", and normalisation is not encoding.
  Nothing in it, and nothing in the function names, decides whether the output
  lives on the sphere.
* **The usual student version:** trusting the name. `CLIPModel.get_image_features`
  and `get_text_features` return the projection output **unnormalised** — that
  is the documented behaviour, and `CLIPModel.forward` normalises internally
  before computing `logits_per_image` while the two `get_*_features` methods do
  not. A student who has only ever called `forward` has never seen an
  unnormalised CLIP feature and assumes there is no such thing.
* **How you would catch it:** the assert above, on the array the assistant
  handed you, before you write a second line. It is one line and it is the whole
  lecture.

---

# Section 2 · Why the sphere

*Examinable: yes.*

The unnormalised inner product factorises as `<a,b> = ||a|| ||b|| cos(theta)`.
Two quantities are mixed: which direction the encoder chose, and how loudly it
said it. Only the direction carries the semantics.

## Cell 4 — how much do the lengths vary
**Prompt to type:**
> print the min, max and max/min ratio of the row norms of `I_raw`, and the
> same for `Q_raw`

**Expect:** images `8.66 / 11.44 / 1.32x`; text `6.28 / 11.25 / 1.79x`.
**Assert:** none — but note which side varies more. The text side does, by a
wide margin, and section 6 is about which side that puts at risk.
**Annotate:** short

`check ·` you can do on paper: the cosines between an image and a caption sit
around 0.15 (cell 6 measures it). A 1.79× spread in length multiplies a 0.15
cosine into a range that comfortably overlaps the gap between a right answer and
a wrong one. Decide before cell 6 whether you think that is enough to reorder a
ranking.

---

# Section 3 · What should an unrelated pair score?

*Examinable: yes.*

Answer this before you run anything, out loud, and write your answer down.

> Two captions that have nothing to do with each other. What cosine do you
> expect?

## Cell 5 — random unit vectors, six dimensions
**Prompt to type:**
> draw 4000 pairs of random gaussian unit vectors at d = 2, 8, 32, 128, 512 and
> 2048, and print the sd of their cosine against 1/sqrt(d) and the fraction with
> |cos| > 0.5. then do 20000 pairs at d = 512 and print the mean, sd and the
> most negative value.

**Expect:** the sd tracks `1/sqrt(d)` at every dimension —

| d | sd measured | 1/√d | \|cos\| > 0.5 |
|---:|---:|---:|---:|
| 2 | 0.7061 | 0.7071 | 66.8% |
| 8 | 0.3575 | 0.3536 | 18.0% |
| 32 | 0.1739 | 0.1768 | 0.1% |
| 128 | 0.0896 | 0.0884 | 0.0% |
| 512 | 0.0440 | 0.0442 | 0.0% |
| 2048 | 0.0226 | 0.0221 | 0.0% |

and at d = 512 over 20,000 pairs: mean `-0.00013`, sd `0.0439`, **most negative
`-0.169`**.
**Assert:** `abs(c.std(ddof=1) - 1/np.sqrt(512)) < 0.005`.
**Annotate:** full

* **Left open:** the derivation, which the prompt does not ask for and which is
  three lines. Fix `u = e1` by rotational symmetry and write `v = z/||z||` with
  `z ~ N(0, I_d)`. Then `u.v = z1/||z||`, whose expectation is zero by the
  symmetry `v -> -v`, and whose second moment is `1/d` because the d coordinates
  share the total equally. So the sd is exactly `1/sqrt(d)`.
* **The usual student version:** answering **−1** to the question above. Most
  rooms do. It is wrong for a reason that is not about machine learning at all:
  **for a fixed unit vector there is exactly one point at cosine −1.** Asking
  200 unrelated captions all to sit there is asking for a configuration that
  does not exist. Two hundred mutually near-orthogonal directions, on the other
  hand, is what 512 dimensions are for.
* **How you would catch it:** the `most negative` line. Over twenty thousand
  random pairs the extreme is `-0.169` — nothing is anywhere near −1, and one
  printed number settles an argument the room was going to have anyway. **In
  high dimensions two unrelated things are orthogonal, not opposite.** That is
  why the contrastive loss targets zero for a non-matching pair, and it is the
  concentration result from application 5 in a new costume.

## Cell 6 — the same measurement on the trained embeddings
**Prompt to type:**
> on the unit-normalised CLIP features, print the mean, sd and min cosine for
> three groups: image–image off-diagonal, image–caption off-diagonal, and the
> matched diagonal. also print the distance between the image centroid and the
> text centroid, and the fraction of unrelated image–caption pairs below zero.

**Expect:**

| pairs | mean | sd | min |
|---|---:|---:|---:|
| two unrelated images | +0.471 | 0.086 | +0.124 |
| an image and an unrelated caption | +0.152 | 0.039 | −0.005 |
| an image and its own caption | +0.304 | 0.035 | +0.201 |

centroid distance **0.831**; fraction of unrelated image–caption pairs below
zero **0.0%**.
**Assert:** `it[off].min() > -0.1` — the geometry says nothing goes near −1, and
only the minimum tests it.
**Annotate:** short

Read two things off this. Unrelated pairs sit near zero and nowhere near −1,
exactly as cell 5 predicts. But they are not *at* zero either, and the two
modalities are centred 0.831 apart: **an absolute cosine threshold tuned on
image–image pairs is meaningless for image–text pairs**, because +0.30 is a
*match* across modalities and *below average* within the image cloud.

Why the gap exists is an open research question — *beyond the book, for
context*. That it exists is four lines of measurement and is examinable.

---

# Section 4 · The temperature

*Examinable: yes.*

Row *i* of the similarity matrix is a *B*-class problem whose correct answer is
column *i*:

`L = -mean_i log( exp(S_ii / tau) / sum_j exp(S_ij / tau) )`

Everything is familiar from thread 11 except `tau`.

## Cell 7 — six temperatures
**Prompt to type:**
> write an infonce function that takes a matrix of cosines and a temperature and
> returns the symmetric loss, the mean probability on the diagonal, the mean
> probability on the hardest off-diagonal entry of each row, and the top-1
> accuracy. run it on `I @ Q.T` at tau = 1, 0.3, 0.1, 0.03, 0.01 and 0.003 and
> print log 200 above the table.

**Expect:** `log 200 = 5.298`, and

| tau | loss | p(correct) | p(hardest wrong) | top-1 |
|---:|---:|---:|---:|---:|
| 1.000 | 5.148 | 0.0058 | 0.0056 | 73.5% |
| 0.300 | 4.802 | 0.0083 | 0.0075 | 73.5% |
| 0.100 | 3.865 | 0.0223 | 0.0160 | 73.5% |
| 0.030 | 1.579 | 0.2834 | 0.0964 | 73.5% |
| 0.010 | 0.786 | 0.6949 | 0.1932 | 73.5% |
| 0.003 | 1.766 | 0.7372 | 0.2439 | 73.5% |

**Assert:** `len(set(r["accuracy"] for r in rows)) == 1`. The whole top-1 column
is one number.
**Annotate:** full

* **Left open:** why `log B` belongs above the table. A contrastive loss is
  scored against a batch-dependent ceiling, and 5.148 at `tau = 1` is not "a
  bad loss" — it is *chance*, to three decimals, for a model that is in fact
  73.5% correct. At `tau = 1` the logits are cosines, a row spans at most 2,
  `exp` of a range of 2 is a ratio of at most 7.4 spread across 200 competitors,
  and the softmax is nearly uniform whatever the model says.
* **The usual student version:** tuning `tau` to improve accuracy. Look at the
  top-1 column: **73.5% at every temperature.** Dividing every entry of a row by
  the same positive constant cannot change which entry is largest, so `tau`
  cannot change the accuracy of a fixed model, and any sweep that appears to is
  measuring something else.
* **How you would catch it:** the identity from thread 11,
  `dL/dS_ij = (p_ij - 1[j=i]) / tau`. A small `tau` concentrates the push on the
  few hardest negatives — the `p(hardest wrong)` column climbs from 0.0056 to
  0.2439 while top-1 does not move. **Temperature changes where the gradient
  goes, not what the model currently gets right.**

## Cell 8 — the temperature the model learned
**Prompt to type:**
> read `clip.logit_scale`, exponentiate it, and print 1/tau, tau, and infonce at
> that temperature

**Expect:** `1/tau = 100.00`, `tau = 0.01000`, loss `0.786`, p(correct) `0.695`,
top-1 `73.5%`.
**Assert:** `abs(clip.logit_scale.exp().item() - 100.0) < 1e-4`.
**Annotate:** short

Note the coincidence and keep it: **CLIP's learned temperature is exactly
0.01** — the number section 6's prompt is about to name. So when section 6 goes
wrong, the temperature will not be what went wrong. `tau` is not a
hyperparameter anybody tunes by hand; the model stores `log(1/tau)` and learns
it by gradient descent, clamped from above because the loss would otherwise be
minimised by driving `tau` to zero.

---

# Section 5 · The batch is the label set

*Examinable: yes.*

The loss needs, for each image, a set of captions it should *not* match. Nobody
labels those — they are the other members of the batch, free and correct with
high probability on a large corpus.

## Cell 9 — batch size is not a memory setting
**Prompt to type:**
> for B in 2, 8, 32, 128 and 200, draw random sub-batches without replacement,
> average infonce over 200 draws (one draw at B = 200), and print top-1, 1/B,
> the loss and log B

**Expect:**

| B | top-1 | chance 1/B | loss | log B |
|---:|---:|---:|---:|---:|
| 2 | 99.5% | 50.0% | 0.015 | 0.693 |
| 8 | 98.4% | 12.5% | 0.054 | 2.079 |
| 32 | 92.2% | 3.1% | 0.216 | 3.466 |
| 128 | 78.9% | 0.8% | 0.598 | 4.852 |
| 200 | 73.5% | 0.5% | 0.786 | 5.298 |

**Assert:** none — see the warning below. Assert nothing on the small-B rows.
**⏱** under 20 s, but **not idempotent**: it consumes `rng`, which cell 5 also
consumes. On a fresh restart-and-run-all the B = 2 loss is 0.015; re-running
this cell alone three more times gave 0.008, 0.008, 0.024 — measured. The top-1
column is stable to about a point; the loss column at B = 2 varies by 3×.
**Annotate:** short

Two things to carry away. **Accuracy falls with B and chance falls faster**, so
the gap — the learning signal — widens: 99.5% against 50.0% is a 49.5-point edge,
73.5% against 0.5% is a 73.0-point one. That is the whole argument for a large
batch, and it is why these models are trained with batches in the tens of
thousands. And **a contrastive loss value is not comparable across papers**: the
same model scores 0.015 and 0.786 in this one table, and the only thing that
changed is `B`. A loss without its `log B` beside it is not a number. Almost
nobody prints it.

---

# Section 6 · Ask for the loss

*Examinable: yes.*

You now have everything you need to write this yourself. Ask for it instead, the
way you would if you were in a hurry.

## Cell 10 — the symmetric contrastive loss
**Prompt to type:**
> write the symmetric contrastive loss for a batch of image and text embeddings,
> with a temperature of 0.01

**Expect:** a short function — `logits = img @ txt.T / tau`, a target of
`torch.arange(len(img))`, and one half times the sum of `cross_entropy(logits,
target)` and `cross_entropy(logits.T, target)`. It runs, the docstring is
accurate, and the shapes, the target and the factor of one half are all correct.
Call it on `torch.tensor(I_raw)` and `torch.tensor(Q_raw)` — the arrays as
`get_*_features` returned them.

**Write the number down before you read on: `loss on raw features: 82.907`.**

**Assert:** none. Nothing raises. That is the point.
**Annotate:** full

*Read these three bullets only after you have run the cell and written 82.907
down.*

* **Left open:** the review question — **is `img @ txt.T` a cosine?** Only if
  both sides are unit vectors, and cell 3 established that they are not. So the
  entries are `||a|| ||b|| cos(theta)`; `tau = 0.01` is dividing a quantity with
  no fixed scale, and the row-wise softmax is comparing lengths as much as
  directions. Compare against section 4: the same model on the sphere at the
  same `tau` scores **0.786**, against a `log B` ceiling of **5.298**. A loss of
  82.907 is 15.6× *worse than knowing nothing* (82.907 / 5.298).
* **The usual student version:** this, verbatim, shipped. It is not a bad answer
  to the question that was asked — the question did not mention normalisation,
  and neither does `get_image_features`. Reviewer question 5 again: the default
  nobody asked for.
* **How you would catch it:** two lines in the specification —
  `assert torch.allclose(img.norm(dim=1), torch.ones(len(img)), atol=1e-5)` and
  the same for `txt`. The bug becomes a crash instead of a number. Everything
  else about this function was right, which is exactly why reading it is not
  enough.

## Cell 11 — what it costs, and by what mechanism
**Prompt to type:**
> compare infonce on `I @ Q.T` against `I_raw @ Q_raw.T` at the learned tau, both
> loss and top-1. then, for each direction of the matrix, count how many
> opposite-side items each row wins, and rank-correlate that with the row's
> embedding length. do it for the raw features and for the normalised ones so I
> have a control.

**Expect:** the headline, and then the mechanism.

| | loss | top-1 (image → query) |
|---|---:|---:|
| on the unit sphere | 0.786 | 73.5% |
| raw dot products | 82.907 | 56.5% |

Seventeen points of top-1, gone. Now which side did it, measured **on the same
200 rows both ways**:

| direction | ratio of lengths on the ranked side | top-1 unit → raw |
|---|---:|---:|
| query → image (image length ranks) | 1.32× | 75.0% → 72.5% |
| image → query (text length ranks) | 1.79× | 73.5% → **56.5%** |

and, in the image → query direction:

| | rank corr (length, items won) | most won by one caption | captions never first |
|---|---:|---:|---:|
| unit sphere | −0.25 | 4 of 200 | 35 |
| raw | **+0.39** | **27 of 200** | **81** |

**The single longest caption in the catalogue** — index 75, `||q|| = 11.25`, the
maximum — is *"He is dressed in the times that his tours encompass."* It wins
**27 of the 200 images** on raw dot products and **0 of 200** on the sphere. A
sentence with no concrete noun in it becomes the most-returned answer in the
catalogue, because its vector is long.
**Assert:** `np.corrcoef` on the raw side exceeds the normalised side —
`corr_raw > corr_unit + 0.5`. Without the control row this table proves nothing.
**Annotate:** full

* **Left open:** *which direction*. `infonce`'s `accuracy` uses `argmax(1)` —
  for each image, the best caption — so the quantity that reorders it is the
  **text** length, not the image length. Correlating image length with queries
  won measures the other direction, where the spread is 1.32× instead of 1.79×
  and the damage is 2.5 points instead of 17. Pick the direction your metric
  actually scores, and say which one it is.
* **The usual student version:** reading the loss, seeing a big number, and
  stopping. 82.907 tells you something broke; it does not tell you that the
  breakage has a *shape* — that it concentrates the catalogue's answers on
  whichever items the encoder happened to speak loudly about, and that 81 of 200
  captions become unreachable. In a product that is not "a worse loss", it is
  "the same six results for every query".
* **How you would catch it:** always print the control. `+0.39` means nothing
  until you have seen that the same statistic on the normalised features is
  `−0.25`, and `27 of 200` means nothing until you have seen that the healthy
  maximum is `4`. One extra line, and a number becomes a comparison.

**The corrected specification:**

> Symmetric InfoNCE. **L2-normalise both embedding sets along the feature axis
> before the matrix product**, so the logits are cosines divided by `tau`.
> Assert that every row of both matrices has unit norm to within 1e-5. Report
> the loss *and* the in-batch top-1 accuracy, and print `log B` beside the loss.

---

# Section 7 · Repair 1 — entries with no description

*Examinable: partly. The retrieval measurement is; the captioner is engineering.*

## Cell 12 — the text route and its hole
**Prompt to type:**
> mean-pool `sentence-transformers/all-MiniLM-L6-v2` over the attention mask to
> embed the 200 descriptions and the 200 queries, normalise, and score
> query-to-description R@1 with a `>=` tie rule. then blank every entry with
> `i % 10 < 3` by setting its column to -inf and report R@1 on those 60 entries
> with and without their descriptions.

**Expect:** R@1 on the 60 blanked entries — described **66.7%**, deleted
**0.0%**. Deleted is structurally zero, not merely small: an entry with no text
is not in a text index at all.
**Assert:** `len(blanked) == 60`.
**Annotate:** short

Use the *same* blanking rule as lecture 23. A repair measured against a
differently-broken baseline is not measured.

## Cell 13 — write the missing descriptions
**Prompt to type:**
> caption the 60 blanked images with
> `Salesforce/blip-image-captioning-base`, batches of 8, `max_new_tokens=30`,
> `num_beams=3`, whitespace-normalise the output, and print four of them beside
> the human description

**Expect:** 60 captions. They are shorter than the human ones — **8.1 words
against 10.6** — and blander. Four examples as they came back:

| human | generated |
|---|---|
| This wire metal rack holds several pairs of shoes and sandals | a dog laying on top of a pair of shoes |
| A traffic light over a street surrounded by tall buildings. | a black and white photo of a city street |
| A toilet seat sits on top of a hole in the ground. | a white toilet bowl |
| A green vase filed with red roses sitting on top of table. | a vase of flowers sitting on a window sie |

**Assert:** `len(generated) == len(blanked)`.
**⏱** **98 s** on an Apple-silicon laptop CPU with the checkpoint already
cached, plus 7 s to load. Cold, add the 944 MB download. Caption only the
blanked 60 — captioning all 200 replaces descriptions that already exist and
destroys the control group.
**Annotate:** short

Look at row 1 before you decide the generated captions are worse. The human
`descriptions[0]` for `CAT-000042` never mentions the dog; **four of that
entry's five human captions do**, and so does the auto-caption. Row 4 is a
different story — `window sie` is a decoding artefact, not a word.

## Cell 14 — what the repair recovers
**Prompt to type:**
> fill the blanked rows of the description matrix with the embedded generated
> captions and re-score R@1 on the same 60 entries, four ways: human present,
> deleted, auto-captioned, and via the CLIP image route. also print the overall
> figure over all 200.

**Expect:**

| R@1, on the same 60 entries | |
|---|---:|
| human description present | 66.7% (40/60) |
| description deleted | 0.0% |
| auto-caption | 61.7% (37/60) |
| joint image route (unused) | 80.0% (48/60) |

and over all 200 queries: **61.5% → 46.0% → 62.0%**.
**Assert:** none.
**Annotate:** full

* **Left open:** what the overall chain measures. It ends *above* where it
  started — 62.0% against 61.5% — and it is tempting to read that as
  auto-captions beating human descriptions. Decompose it and the two row sets
  move in opposite directions:

| rows | full | deleted | filled |
|---|---:|---:|---:|
| the 60 blanked | 66.7% | 0.0% | 61.7% |
| the 140 untouched | 59.3% | 65.7% | 62.1% |
| all 200 | 61.5% | 46.0% | 62.0% |

  The 140 untouched entries *gain* 6.4 points when 60 competitors are deleted,
  because deleting a competitor makes every other query easier — their own
  descriptions never change. The overall line is a mixture of a real repair and
  an artefact of a shrinking index. **Report the 60.**
* **The usual student version:** quoting the overall improvement. It attributes
  to all 200 entries a change that touched 60, and here it does not merely
  dilute the effect — it reverses its sign.
* **How you would catch it:** score the rows you changed, separately, and say so
  in the sentence. And put an interval on it: 40/60 against 37/60 is 66.7% ±11.9
  against 61.7% ±12.3 at 95% normal approximation. **Those intervals overlap.**
  The honest claim is "an auto-caption recovers most of the loss on these sixty
  entries, and sixty entries cannot resolve the remainder."

Three things not to claim. A generated caption is not evidence about the
product — it describes the photograph, and the photograph is not the
specification. We are scoring generated captions against human captions of the
*same* image, which is a friendly test. And the control row is the loudest thing
in the table: **the image route alone gets 80.0% on these 60**, beating the human
descriptions, so the repair we just performed is on the weaker of two routes we
already have.

The failure mode to watch for is an auto-caption that is *wrong*, making an entry
findable under the wrong query — worse than unfindable, and **nothing measured
here detects it**.

---

# Section 8 · Repair 2 — queries with no single right answer

*Examinable: no — the grounding metric is; the generation is beyond the book.*

Customers do not type captions. They type "something to sit on". A ranked list is
a poor answer because the customer wants a shortlist **with reasons**, and a
ranking has no place to put a reason.

Decide how to check it **before** generating anything. "The answer is good" is
not measurable in a lecture, so require cited stock numbers: a SKU either exists
in the catalogue or it does not.

## Cell 15 — the model and the shortlist
**Prompt to type:**
> load `Qwen/Qwen2.5-0.5B-Instruct`, write an `ask(prompt)` helper that applies
> the chat template with a system message telling it to cite catalogue SKUs
> shaped `CAT-123456`, and generate greedily with `do_sample=False`. then embed
> six ambiguous queries with CLIP text and take the top-5 images for each.

**Expect:** `top5` of shape `(6, 5)`. Nothing is generated yet.
**Assert:** `top5.shape == (6, 5)`.
**⏱** 2 s to load warm; cold, add the 942 MB download.
**Annotate:** short

The six queries: *something to sit on · somewhere to eat outdoors · a way to get
across town without a car · gear for bad weather · something to put flowers in ·
a machine that heats food*. Six of the twelve on the slides, so the cell finishes
inside the hour — the deck quotes the full run.

## Cell 16 — closed book against retrieval-augmented
**Prompt to type:**
> for each of the six queries, ask the model twice: once telling it only that
> the catalogue has 200 entries, and once giving it the five retrieved SKUs and
> their descriptions and telling it to cite only from that list. regex out every
> `CAT-\d{6}`, and print how many cited SKUs actually exist under each
> condition.

**Expect:**

```
closed book              0 of  21 cited SKUs exist (  0.0%)
retrieval-augmented     12 of  12 cited SKUs exist (100.0%)
```

**Assert:** `set(grounded_cited) <= valid_skus`.
**⏱** **19 s** for all twelve generations on an Apple-silicon laptop CPU.
**Annotate:** full

* **Left open:** what the denominator is. Those 21 closed-book citations are
  **three distinct strings** — `CAT-123456`, `CAT-789012`, `CAT-234567` — repeated
  seven times over. And `CAT-123456` is the example **we put in the system
  prompt**. The metric as written counts one echoed format example seven times
  and reports it as seven hallucinations. Deduplicate and the honest line is *0
  of 3 distinct invented numbers, one of which we supplied*. The
  retrieval-augmented side has 11 unique of 12. Decide which denominator you
  meant before you print a percentage.
* **The usual student version:** omitting `do_sample`. The Qwen2.5-Instruct
  checkpoint ships `generation_config.json` with **`do_sample: true`,
  `temperature: 0.7`, `top_p: 0.8`, `top_k: 20`** — so `model.generate()` samples
  unless you say otherwise, and the grounding rate becomes a random variable you
  have not characterised. `do_sample=False` is not a stylistic preference here;
  it is what makes 0% and 100% numbers rather than draws. You will see
  transformers warn `generation flags are not valid and may be ignored:
  ['temperature', 'top_p', 'top_k']` — that warning is the confirmation that your
  override took.
* **How you would catch it:** read one closed-book answer in full. It is fluent,
  correctly formatted, confident, and every SKU in it is invented. **The
  closed-book failure is not that the model refuses — it is that it does not
  refuse**, and nothing in the output distinguishes an invented stock number from
  a real one. Only the set membership test does.

Then read one grounded answer, because 100% is not the win it looks like. The
shortlist retrieved for *"something to sit on"* was: *a set of park benches near
a lamp post · a white toilet bowl with an electronic brown seat · a toilet seat
sits on top of a hole in the ground · a black and white pic of a man and a horse
· a young man bending next to a toilet.* The model cited the benches and the
toilet bowl. **Grounding is not correctness.** Both citations exist; one of them
is a toilet. We measured the cheap half, and we should say which half we
measured.

**The retriever is now the ceiling.** If the right entry is not in the top five,
no amount of generation recovers it — which is why lecture 23's R@5 is the number
that matters here, not its R@1.

---

# Section 9 · Red-team, and the end

Swap notebooks. Four questions for *this* one:

1. Are the embeddings on both sides unit vectors? **Assert it, do not read it.**
2. Is every recall reported with its candidate-set size?
3. Were the auto-captions generated for entries that are also in the evaluation
   queries — and does that matter here?
4. Does the grounding metric count a SKU cited twice as two citations? What would
   that do to the percentage? *(Cell 16 answers this one: 21 citations, 3 distinct
   strings.)*

### The course, in five rules

1. Split before anything is fitted.
2. All preprocessing inside the object that is cross-validated.
3. Nothing derived from the test set in the training path.
4. Fixed seeds; report per-fold scores, not only the mean.
5. Every number gets a baseline, and every baseline gets stated.

Twelve applications and twenty-four lectures, and those five never needed
extending — not for images, not for sequences, and not for two modalities at
once.

You will forget most of the syntax. Keep the **five** reviewer questions, the
**five** rules above, and the habit of writing the number down first.

---

## §7.2 — exercises, with their re-run order

Cell numbers are the code-cell numbers in this script.

**E1 — does the temperature really not move the accuracy?** Edit cell 7's list
to `[3.0, 1.0, 0.1, 0.001]`. Re-run **cell 7 only**. The top-1 column is 73.5%
at all four. *(Cell 7 is idempotent — it re-derives `sim` from `I` and `Q`.)*

**E2 — how much of the batch-size effect is chance?** Re-run **cell 9** four
times without restarting and record the B = 2 row each time. Observed:
0.015, 0.008, 0.008, 0.024. Then restart and run cells 1 → 9 in order; you get
0.015 again. **Cell 9 is not idempotent because `rng` is created in cell 5.**
This is the exercise the state hazard at the top of the script refers to.

**E3 — normalise only one side.** In cell 11, score `infonce(I_raw @ Q.T, tau)`
and `infonce(I @ Q_raw.T, tau)`. Predict which one is worse *before* you run it,
using the 1.32× and 1.79× from cell 4. Re-run **cell 11 only** — cells 3 and 8
must already have run, for `I_raw`, `Q_raw` and `tau`. Run and verified:

| | loss | top-1 |
|---|---:|---:|
| both unit | 0.786 | 73.5% |
| images raw only | 6.125 | 73.5% |
| text raw only | 7.306 | 56.5% |
| both raw | 82.907 | 56.5% |

**Leaving the images unnormalised costs zero points of top-1. The entire
17-point collapse comes from the text side.** The reason is one line of algebra
and is the best single check in this notebook: `I_raw @ Q.T` scales row *i* by
`‖I_i‖`, a constant *across* that row, so `argmax(1)` cannot move; `I @ Q_raw.T`
scales column *j* by `‖Q_j‖`, which varies across the row, so it can and does.
Work that out on paper before running and you will have predicted all four rows.

**E4 — is 66.7% vs 61.7% a result?** Take the interval from cell 14's annotation
and work out how many entries you would need for the difference to clear it. No
re-runs.

**E5 — deduplicate the grounding metric.** In cell 16, replace
`SKU_RE.findall(...)` accumulation with a set per query. Re-run **cell 16 only**
(cells 15 and 2 must have run, for `ask`, `top5` and `valid_skus`). The
closed-book denominator falls from 21 to 3. ⏱ 19 s.

**E6 — caption all 200 instead of 60.** Change `blanked` in cell 13's loop to
`range(200)` and re-run **cells 13 → 14** in that order. ⏱ **about 5.5 minutes**
on a laptop CPU (98 s for 60, scaled). Then explain why cell 14's table is no
longer interpretable.

---

## Defects found in the current notebook

Checked against `notebooks/lecture-24.ipynb` (46 cells, 16 code cells, **no
stored outputs and no execution counts** — so §1.2's "prose figures must
reconcile with stored outputs" cannot be machine-checked on this file at all;
every figure below I re-derived by running the notebook's own code on the cached
catalogue in `notebooks/datasets/app12`, CPU, torch 2.13.0, transformers 4.57.3).

### Verified by running the code

**1. Cell 32 prints `inf`, and it contradicts cell 29 three cells earlier.**
§1.5. Cell 29 prints `loss on raw features: 82.907` — torch's `cross_entropy`
uses a numerically stable log-softmax. Cell 32 computes the same quantity via
`torch.softmax(...)` followed by `np.log(...)`; the raw logits reach ±3,600 at
`tau = 0.01`, two diagonal entries of `p` and one of `q` underflow to exactly
0.0, and the cell prints

```
raw dot products                 inf    56.5%
```

with a `RuntimeWarning: divide by zero encountered in log`. Two numbers for the
same quantity, three cells apart, unreconciled — and one of them is not a number.
Verified: I ran the notebook's `infonce` verbatim and got
`{'loss': inf, 'n_zero_diag_p': 2, 'n_zero_diag_q': 1}`.

**2. The annotation on cell 32 describes an output the cell cannot produce.**
§1.1. Its "usual student version" bullet reads *"seeing a small loss difference
and concluding it does not matter."* The printed difference is `0.786` against
`inf`. There is no small difference to see.

**3. Cell 32 measures the mechanism in the direction where it barely happens.**
§2.1, §2.2. `wins = np.bincount((I_raw @ Q_raw.T).argmax(0))` is the
query→image direction, ranked by **image** length (spread 1.32×). The
`accuracy` printed two lines above it is `argmax(1)`, the image→query direction,
ranked by **text** length (spread 1.79×). Measured both ways on the same 200
rows:

```
query -> image (what `wins` measures):  75.0% -> 72.5%   (-2.5 points)
image -> query (what `accuracy` scores): 73.5% -> 56.5%   (-17.0 points)
```

So the cell prints a 17-point collapse and then explains it with a statistic
taken in the direction that moved 2.5 points.

**4. Cell 32's mechanism numbers have no control, and against one they are
nearly null.** §2.2. It prints rank correlation `+0.26`, *"one image took 6 of
200 queries"* and *"49 images were never ranked first"*. The same three
statistics computed on the **normalised** features are `+0.03`, `5 of 200`, and
`41`. Six against a healthy five is not evidence of anything. Measured in the
direction the metric actually scores, the contrast is real and large: rank
correlation `+0.39` against `−0.25`, most-won `27 of 200` against `4`, and
never-first `81` against `35` — and the 27 all go to caption 75, *"He is dressed
in the times that his tours encompass."*, which has the single longest text
embedding in the catalogue (11.25, the maximum).

**5. The modality-gap argument in cell 18 is false, and falsifiable from the
notebook's own data.** §1.1, §3.2. The prose reads:

> "Only the ranking within a row is trained, so adding a constant offset to
> every image embedding changes no ranking and no loss — the objective has no
> reason to remove the gap, and it does not."

Adding a constant `c` to every image embedding sends `S_ij` to
`S_ij + c·Q_j`, which varies with `j` and therefore *does* change every row's
ranking. Measured: shifting the images by exactly the gap vector
(`c = Q̄ − Ī`, `‖c‖ = 0.831`) and renormalising takes the loss from **0.786 to
1.821** and top-1 from **73.5% to 54.0%**. Even a small random offset of norm
0.30 moves the loss to 0.820. The *conclusion* — that the gap survives training —
is correct; the reason given for it is the opposite of the truth, and the
objective in fact penalises closing the gap by 1.03 nats.

**6. Cell 39's "over all 200" chain is a mixture of two row sets moving in
opposite directions.** §2.1. The notebook prints `61.5% -> 46.0% -> 62.0%` and
the prose above it says *"A generated caption recovers **part** of the loss, not
all of it."* The overall chain ends **above** where it started, which is the
opposite claim. The prompt box then warns against *"reporting the overall
improvement, which is smaller"* — it is not smaller, it is larger. Decomposed:

```
                 full   deleted  filled
the 60 blanked   66.7%    0.0%   61.7%
the 140 untouched 59.3%   65.7%   62.1%
all 200          61.5%   46.0%   62.0%
```

The 140 untouched entries gain 6.4 points from the *deletion* alone, because
removing 60 competitors makes every remaining query easier; their own
descriptions never change. Ten of those 140 flip R@1 status between `full` and
`filled`.

**7. The grounding percentage in cell 44 has an inflated denominator, and the
notebook's own red-team question 4 asks about exactly this and never answers
it.** §1.4, §2.4. Measured with `do_sample=False`: closed book **0 of 21**,
retrieval-augmented **12 of 12**. But the 21 closed-book citations are three
distinct strings — `CAT-123456`, `CAT-789012`, `CAT-234567` — and `CAT-123456` is
the format example the notebook itself puts in the system prompt. Deduplicated:
0 of 3, one of them supplied by us. Section 9 asks *"Does the grounding metric
count a SKU cited twice as two citations? What would that do to the
percentage?"* — the headline number is the one that question invalidates.

**8. `B` is bound to two different types at global scope.** §4.1, and it is on
the list of things `check_all.py` is supposed to catch. Cell 14:
`B = unit(rng.normal(size=(20000, d)))`, an ndarray of shape (20000, 512). Cell
26: `for B in [2, 8, 32, 128, N_CATALOGUE]`, an int. This is the exact defect
lecture 19 spends 200 words on and then commits.

**9. Cell 26 is not idempotent.** §4.2, §4.3. `rng` is created in cell 14 and
consumed by it; cell 26 draws 600 more sub-batches from the same generator.
Measured — restart-and-run-all, then re-run cell 26 alone three times:

```
B=2 loss:  0.015  ->  0.008  ->  0.008  ->  0.024
```

A 3× range on a figure printed to three decimals, and the notebook gives no
re-run order. The top-1 column is stable to about one point.

**10. `tau` is a loop variable before it is a constant.** §4.1. Cell 20 ends with
`tau` bound to `0.003`, the last value of `for tau in [...]`. Cell 26 rebinds it
to `1/scale = 0.01`. Cell 29's headline number depends on which of those is
live: run cell 29 after cell 20 but before cell 26 and the "loss on raw
features" reads a different number. Nothing in the notebook names the hazard.

**11. Qwen's own `generation_config.json` sets `do_sample: true`.** Confirmed by
loading it: `do_sample=True, temperature=0.7, top_p=0.8, top_k=20,
repetition_penalty=1.1`. Cell 42 correctly passes `do_sample=False`, and the
prompt box's `catch` bullet correctly flags it — but the notebook never says
*why* it is necessary here specifically, i.e. that this checkpoint ships sampling
on. Not a defect in the code; a missed §6.2 grounding in the only place in the
notebook where a real library default was available for free.

### Verified by inspection

**12. The defect is announced four times before the cell.** §8.1 — the precise
defect the guidelines were written to stop. Cell 0 (header): *"The cell marked
⚠ read before running contains the defect this lecture is about."* Cell 7
(prompt box for the embeddings cell): *"the assistant failure in section 6 is
about what happens when you use them."* Cell 27 (section heading): *"## 6 · ⚠
Read before running — the assistant failure."* Cell 27 again, immediately above
the cell: *"One clause missing — the clause section 2 spent five minutes on."*
Plus the prompt label itself, *"⚠ what the assistant returns"*. By the time the
reader's eye reaches `img @ txt.T`, nobody falls in.

**13. "The four rules" resolves to nothing.** §3.3. Cell 45's last paragraph:
*"Keep the five reviewer questions, the four rules, and the habit of writing the
number down first."* The section immediately above it is headed **"The course,
in five rules"** and lists five numbered items. I grepped all 24 source modules
for a set of four rules and found none. The five reviewer questions do resolve —
lecture 3 §13 tabulates them.

**14. The header understates the download for a fresh runtime.** §7.1. It says
*"the same 200-entry catalogue as the previous lecture (cached), plus two more
checkpoints: a captioner (about 1 GB) and a small instruction-tuned language
model (about 1 GB)."* Measured from the HF cache: BLIP 944 MB, Qwen 942 MB —
both accurate — but CLIP is a further **577 MB** and MiniLM **87 MB**, and a
student opening lecture 24 without having run lecture 23 in the same runtime
downloads **2.49 GB**, not 2 GB.

**15. No CPU figure anywhere, and the only wall clock given is for a GPU.**
§7.1. Header: *"Expected wall clock on a Colab GPU runtime: five to eight minutes
end to end."* Sections 7 and 8 say *"1–3 min"* and *"2–4 min"*, both implicitly
GPU. Nothing trains in this notebook, so a CPU runtime is entirely adequate and
is what most students will have — measured, the two generative cells are 98 s and
19 s on a laptop CPU. A reader on CPU has no way to tell from the notebook
whether they are two minutes or forty from the end. This is the failure mode that
blocked 4 of 6 exercises in the lecture 19 audit.

**16. §8.3 is unmet.** *"Every section gets one of: examinable, not examinable —
engineering, or beyond the book, for context."* The string "examinable" appears
**three times** in a nine-section notebook (cells 16, 18 and 45), all three
negative — *"not examinable"* — and six sections carry no marking at all.

**17. Cell 7's justification for `Fn` refers to a namespace that does not
exist.** §3.1-adjacent. The prompt box says `torch.nn.functional as Fn` is used
*"rather than `F` — `F` is already the feature matrix in the previous notebook's
namespace"*. This notebook's feature matrices are `I`, `Q`, `D` and `Qt`; `F`
appears nowhere in any cell of lecture 24, and the notebook opens by insisting it
does not rely on the previous notebook's kernel. The convention is a good one;
the reason given for it contradicts the notebook's own header.

**18. Section 7's opening sentence presents a synthetic hole as a property of
the data.** *"Sixty of the two hundred entries have no description, so they score
exactly zero on the text route."* The 60 are chosen by `i % 10 < 3`, an arbitrary
rule applied by the notebook itself; and they do not "score zero" — their column
is set to `-inf`, so `ranks_of_truth` returns rank 200 for each, giving R@1 of
0.0%. The rank and the score are different quantities and the sentence conflates
them.

### Checked and found clean

- **§5.1 / §5.2** — no markdown line indented ≥4 outside a fence, no fence marker
  indented ≥4, no unclosed fence. Scanned all 30 markdown cells.
- **§3.1** — no ```` ```python ```` block appears in any markdown cell, so there
  is no quoted code to fail to match a cell.
- **§3.3, the resolving references** — *"not used until section 8"* (`valid_skus`,
  cell 5 → used in cell 44, section 8) ✓; *"the assistant failure in section 6"*
  ✓; *"the clause section 2 spent five minutes on"* ✓ (section 2 is the
  factorisation); *"the concentration result from Lecture 10"* and *"application
  5"* are the same thing — lectures 9 and 10 are Build and Fix of application 5 ✓;
  *"the R@5 from the previous lecture"* ✓ — lecture 23 line 227 computes
  `{f"R@{k}": (r <= k).mean() for k in (1, 5, 10)}`.
- **§1** — the catalogue is genuinely deterministic. Sorting the index by
  `cocoid` and taking the first 200 gives entry 0 = `CAT-000042` on a clean
  re-download, and `len({sku}) == 200`.
- **§4.2** — no training cell exists in this notebook, so the
  re-instantiation rule has nothing to bite on. All model loads are
  `from_pretrained(...).eval()` and are idempotent.
- MiniLM and CLIP are bit-deterministic on CPU here: two calls in one process
  agree exactly, and batch size 8 vs 64 vs 200 agrees to 1.3e-7. The R@1 figures
  are reproducible; only cell 26's `rng`-dependent table is not.

### Could not check

- Whether the deck's "full run" figures (twelve ambiguous queries, longer beam)
  reconcile with the six-query / beam-3 subset the notebook uses. The slides are
  not in this repository, so the module docstring's claim that *"the deck quotes
  the full run"* is unverifiable from here.
- Colab-specific rendering (§10.8). I checked the markdown source mechanically
  and read it, but I did not open the notebook in Colab, which is the only
  environment `_prompt.py` says matters.
- GPU wall clock. Every timing above is CPU on Apple silicon; I have no CUDA
  device, so the header's "five to eight minutes on a Colab GPU runtime" is
  neither confirmed nor contradicted.
