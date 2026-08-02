# Triage — lecture 24

Claims triaged from the `Defects found in the current notebook` section of
`tools/prompts/lecture_24.md`, against `notebooks/lecture-24.ipynb` (46 cells,
16 code cells) and `tools/notebooks/lecture_24.py`.

**Scope note.** The task message said 9 claims. The Phase A report contains
**18** numbered claims (1–11 "Verified by running the code", 12–18 "Verified by
inspection"). All 18 are triaged below.

**Stated once, not repeated per claim (per the brief):** `lecture-24.ipynb`
stores **no outputs and no execution counts** (all 16 `execution_count` are
`None`, all `outputs` empty), so §1.2 cannot be machine-reconciled on this file.
Every figure below was re-derived by running the notebook's own code verbatim.

**Environment.** torch 2.13.0, transformers 4.57.3, CPU, Python 3.13.
`notebooks/datasets/app12` holds the CSV and all 200 images. All four
checkpoints were **already in the HuggingFace cache** — CLIP
`model.safetensors` 577 MB, BLIP 944 MB, Qwen2.5-0.5B-Instruct 942 MB, MiniLM
87 MB — so nothing was downloaded. Per the task instruction, **no generation was
run**: BLIP captioning (cell 37) and Qwen generation (cell 44) were not
executed, and the figures that depend on them are marked so.

**Baseline reproduction.** Cells 3, 6, 8, 11, 14, 17, 20, 23, 26 reproduce the
prompt script exactly: entry 0 = `CAT-000042`; image lengths 8.66/11.44/1.32×,
text 6.28/11.25/1.79×; the d-sweep sd column 0.7061/0.3575/0.1739/0.0896/
0.0440/0.0226; most negative cosine −0.169; modality gap 0.831; `log 200 =
5.298`; the full tau table including top-1 73.5% at all six; `1/tau = 100.00`.
The notebook's numeric spine is sound — the defects are elsewhere.

---

### Claim 1 — cell 32 prints `inf`, contradicting cell 29's `82.907` three cells earlier
**Verdict:** CONFIRMED
**Evidence:** ran the notebook's `infonce` verbatim on `I_raw @ Q_raw.T` at
`tau = 1/scale`:

```
raw:  {'loss': inf, 'accuracy': 0.565, 'n_zero_diag_p': 2, 'n_zero_diag_q': 1}
unit: {'loss': 0.786, 'accuracy': 0.735}
warnings: ['divide by zero encountered in log', 'divide by zero encountered in log']
raw logit range at tau=0.01: -35 .. 3327
cell 29 (torch cross_entropy, tau=0.01): 82.907
```

Two diagonal entries of `p` and one of `q` underflow to exactly 0.0, so
`np.log` returns `-inf` and the mean is `inf`. Cell 29 computes the same
quantity through `Fn.cross_entropy`, which is a stable log-softmax, and prints
`82.907`. Both figures are exactly as Phase A reported. Nothing in the notebook
reconciles them.
**Severity:** misleads a student
**Origin:** generated code
**Fix:** compute the loss in `infonce` with `log_softmax` rather than
`softmax` then `np.log`; then cell 32 prints `82.907` and matches cell 29.

---

### Claim 2 — cell 32's "usual student version" bullet describes an output the cell cannot produce
**Verdict:** CONFIRMED
**Evidence:** the bullet is in cell **31** (the prompt box above the code cell;
Phase A's "cell 32" is off by one, the substance is right). Cell 31, verbatim:

> * **The usual student version:** seeing a small loss difference and concluding
>   it does not matter.

The cell below prints `0.786` against `inf` (claim 1). Even with claim 1 fixed
it would print `0.786` against `82.907` — a 105× difference. There is no small
loss difference to see under either reading, so this bullet is wrong
independently of claim 1 and is not a duplicate of it.
**Severity:** misleads a student
**Origin:** hand-written prose
**Fix:** rewrite the bullet to the failure the cell actually shows — reading a
large loss and stopping, without asking what shape the damage has.

---

### Claim 3 — cell 32 measures the mechanism in the direction where it barely happens
**Verdict:** CONFIRMED
**Evidence:** cell 32 computes
`wins = np.bincount((I_raw @ Q_raw.T).argmax(0), ...)` and correlates it with
`norms`, which cell 11 bound to the **image** lengths. `infonce`'s `accuracy`
uses `logits.argmax(1)`. Measured on the same 200 rows:

```
image->query (argmax(1), what `accuracy` scores): unit 73.5% -> raw 56.5%
query->image (argmax(0), what `wins` measures) : unit 75.0% -> raw 72.5%
```

Exactly the numbers Phase A gives. The cell prints a 17.0-point collapse two
lines above a statistic taken in the direction that moved 2.5 points. Confirmed
independently by the E3 decomposition in the prompt script, which I also ran:
`I_raw @ Q.T` (images raw only) loss 6.125, top-1 **73.5%** — unchanged;
`I @ Q_raw.T` (text raw only) loss 7.306, top-1 **56.5%** — the whole collapse.
Scaling row *i* by `‖I_i‖` is constant across the row and cannot move
`argmax(1)`; scaling column *j* by `‖Q_j‖` varies across the row and does.
**Severity:** misleads a student
**Origin:** generated code
**Fix:** change cell 32 to `argmax(1)` and correlate against
`np.linalg.norm(Q_raw, axis=1)`, and say which direction is being scored.

---

### Claim 4 — cell 32's mechanism numbers have no control, and against one they are nearly null
**Verdict:** CONFIRMED
**Evidence:** all four rows re-derived:

```
notebook (raw,  argmax(0), image len): corr +0.26  max  6 of 200  never-first 49
CONTROL  (unit, argmax(0), image len): corr +0.03  max  5 of 200  never-first 41
raw      (argmax(1), text len)       : corr +0.39  max 27 of 200  never-first 81
unit     (argmax(1), text len)       : corr -0.25  max  4 of 200  never-first 35
most-won caption idx 75 (||q||=11.25) wins 27 raw, 0 unit
longest caption idx 75, ||q||=11.25: 'He is dressed in the times that his tours encompass.'
```

Every figure Phase A quotes reproduces exactly. The notebook's printed `6 of
200` against a healthy control of `5 of 200` is not evidence of anything, and
the notebook prints no control at all — which is the §2.2 failure. The
statistic in the scored direction (+0.39 vs −0.25, 27 vs 4, 81 vs 35) is real
and large, and index 75 is both the maximum text norm and the 27-win caption.
**Severity:** misleads a student
**Origin:** generated code
**Fix:** print the same three statistics on `I @ Q.T` as a control row, in the
`argmax(1)` direction.

---

### Claim 5 — the modality-gap argument in cell 18 is false and falsifiable from the notebook's own data
**Verdict:** CONFIRMED
**Evidence:** cell 18 reads *"Only the ranking within a row is trained, so
adding a constant offset to every image embedding changes no ranking and no
loss."* Adding `c` to every image sends `S_ij = I_i·Q_j` to `S_ij + c·Q_j`,
and `c·Q_j` varies with *j*. Measured with `c = Q̄ − Ī`:

```
||c|| = 0.831
shift by gap, NOT renormalised: loss 1.610  top-1 54.0%   (baseline 0.786 / 73.5%)
rows whose argmax moved:        91 of 200
shift by gap + renormalise:     loss 1.821  top-1 54.0%
small random offset ||v||=0.30: loss 0.821  top-1 71.0%
sd over j of c·Q_j:             0.0543
```

91 of 200 rows change their argmax under the offset the prose says changes no
ranking. Both figures Phase A quotes (1.821, 54.0%, and 0.820 for the small
offset — I get 0.821 with my own random `v`, seed-dependent) reproduce. The
*conclusion* (the gap survives training) is correct; the reason given is the
opposite of the truth — the objective penalises closing the gap by
1.821 − 0.786 ≈ 1.04 nats.
**Severity:** misleads a student
**Origin:** hand-written prose
**Fix:** replace the sentence. A constant offset *does* change every row's
ranking; the honest statement is that the loss is invariant to a *rotation*, and
that nothing in the objective rewards closing the gap.

---

### Claim 6 — cell 39's "over all 200" chain mixes two row sets moving in opposite directions
**Verdict:** CONFIRMED (mechanism); the `filled` column is not re-derived — see below
**Evidence:** ran cell 35 verbatim (MiniLM, cached, no generation):

```
R@1 on the 60  — human present 66.7% (40/60)   deleted 0.0%
R@1 on the 140 — full 59.3%   deleted 65.7%    (gain +6.4 pts)
R@1 all 200    — full 61.5%   deleted 46.0%
joint image route on the 60: 80.0% (48/60)
```

Every non-generative figure in Phase A's decomposition table reproduces exactly,
including the +6.4-point gain on the 140 untouched rows from the *deletion
alone* — the entries whose own descriptions never change. That is the confound,
and it is fully verified. The prompt box (cell 38) says the student error is
*"reporting the overall improvement, which is smaller"*; cell 40 says a
generated caption *"recovers **part** of the loss"*.

**Not re-derived:** the `filled` column (61.7% on the 60, 62.1% on the 140,
62.0% overall) and "ten of those 140 flip R@1 status" require BLIP captioning,
which I was instructed not to run. So Phase A's headline — that the overall
chain ends *above* 61.5% and therefore reverses the prose's sign — rests on one
figure I did not reproduce. The structural defect (an overall number confounded
by a shrinking index) is confirmed on its own.
**Severity:** misleads a student
**Origin:** hand-written prose
**Fix:** delete the overall chain or print the 60/140/200 decomposition beside
it, and state that the 140 gain is an artefact of deleting competitors.

---

### Claim 7 — the grounding percentage in cell 44 has an inflated denominator, and red-team question 4 asks about it and never answers it
**Verdict:** CONFIRMED (structure); the counts 0-of-21 / 12-of-12 are not re-derived
**Evidence:** by inspection of cell 44, the metric accumulates with `+=` and
never deduplicates:

```
closed_cited += SKU_RE.findall(closed)
ok  = sum(s in valid_skus for s in cited)
pct = 100 * ok / len(cited) if cited else 0.0
```

so a SKU cited twice is two citations and `len(cited)` is a token count, not a
distinct count. Cell 42's system prompt supplies the format example:
`"Cite catalogue SKUs, which always look like CAT-123456."` — so any echo of
`CAT-123456` is counted as a hallucination the notebook itself planted. Cell 45
asks *"Does the grounding metric count a SKU cited twice as two citations? What
would that do to the percentage?"* and the notebook never answers it — I read
every cell after 45; there are none.

**Not re-derived:** the specific counts (21 citations, 3 distinct strings, 12 of
12) require Qwen generation, which I did not run.
**Severity:** misleads a student
**Origin:** generated code
**Fix:** use `set(SKU_RE.findall(...))` per query, print both denominators, and
exclude the system-prompt example string.

---

### Claim 8 — `B` is bound to two different types at global scope
**Verdict:** CONFIRMED
**Evidence:** cell 14, line 12: `B = unit(rng.normal(size=(20000, d)))` —
verified at runtime, `type(B).__name__ == 'ndarray'`, shape `(20000, 512)`.
Cell 26, line 3: `for B in [2, 8, 32, 128, N_CATALOGUE]` — an `int`. A §4.1
violation, and the notebook thread that lectures on exactly this
(`target` clobbered by a loop variable, lecture 19) is the one this lecture
closes.
**Severity:** wrong but harmless — in restart-and-run-all order the array `B` is
dead before cell 26 rebinds it, so no printed number changes
**Origin:** generated code
**Fix:** rename cell 14's second sample matrix to `V` (or `B2`); leave the loop
variable alone.

---

### Claim 9 — cell 26 is not idempotent
**Verdict:** CONFIRMED
**Evidence:** `rng` is created in cell 14 and consumed by it; cell 26 draws 600
more sub-batches from the same generator. Simulated restart-and-run-all (cell 14
then cell 26), then cell 26 three more times:

```
B=2 loss  : 0.015  ->  0.008  ->  0.008  ->  0.024
B=2 top-1 : 99.5%  ->  99.8%  ->  100.0% ->  99.0%
fresh restart-and-run-all B=2 loss: 0.015   (matches run 1)
```

Phase A's four values reproduce digit-for-digit. A 3× range on a figure printed
to three decimals; the top-1 column is stable to about a point, as claimed. The
notebook states no re-run order anywhere — the hazard note exists only in
`tools/prompts/lecture_24.md`, not in the artefact.
**Severity:** misleads a student
**Origin:** notebook structure
**Fix:** re-seed inside cell 26 (`rng = np.random.default_rng(SEED + 1)`), or
state the restart-and-run-all requirement in the markdown above it.

---

### Claim 10 — `tau` is a loop variable before it is a constant
**Verdict:** CONFIRMED
**Evidence:** cell 20 ends with `for tau in [1.0, 0.3, 0.1, 0.03, 0.01, 0.003]`,
leaving `tau = 0.003`. Cell 26 opens `tau = 1 / scale` = `0.01`. Cell 29 calls
`contrastive_loss(img_t, txt_t, tau)`. Measured:

```
cell 29 with tau = 0.01  (cell 26 has run): 82.907
cell 29 with tau = 0.003 (cell 20 still live): 276.352
```

The headline number of the lecture's central defect cell changes by 3.3× on
cell execution order, and nothing in the notebook names the hazard. §4.1 and
§4.3.
**Severity:** misleads a student
**Origin:** generated code
**Fix:** rename cell 20's loop variable to `t`, and pass `1/scale` explicitly at
cell 29.

---

### Claim 11 — Qwen's `generation_config.json` sets `do_sample: true`, and the notebook never says so
**Verdict:** FALSE POSITIVE (as a guidelines defect); the underlying fact is
confirmed
**Evidence:** the fact is exactly as stated —
`~/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/*/generation_config.json`:

```json
{"do_sample": true, "repetition_penalty": 1.1, "temperature": 0.7,
 "top_p": 0.8, "top_k": 20, ...}
```

But cell 42 passes `do_sample=False`, and cell 41's catch bullet reads
*"`do_sample=False`. A sampled generation gives a different answer every run and
the grounding rate becomes a random variable you have not characterised."*
§6.2 requires the bullet be *grounded in* a real failure or a documented library
default — this one is; it just does not cite the file. Phase A concedes
*"Not a defect in the code"*. No rule is violated, so this is a missed
improvement rather than a defect.
**Severity:** cosmetic
**Origin:** hand-written prose
**Fix:** none needed; naming the checkpoint's shipped defaults in the bullet
would strengthen it.

---

### Claim 12 — the defect is announced four times before the cell
**Verdict:** CONFIRMED
**Evidence:** all five announcements found by string search, in order:

- cell 0: *"The cell marked **⚠ read before running** contains the defect this
  lecture is about, and it does not raise an exception."*
- cell 7: *"the assistant failure in section 6 is about what happens when you
  use them"*
- cell 27: *"## 6 · ⚠ Read before running — the assistant failure"*
- cell 27: *"One clause missing — the clause section 2 spent five minutes on."*
- cell 28 (the box label): *"**Prompt · ⚠ what the assistant returns**"*

That is §8.1's failure verbatim, and the guideline names it as the precise
defect it was written to stop. §8.1's preferred shape — run the cell
unannounced, write the number down, *then* open with the ⚠ — is available here
at zero cost, since cell 29 does not raise.
**Severity:** misleads a student — "would you have caught it?" has no honest
answer after five flags
**Origin:** notebook structure
**Fix:** drop the ⚠ from cells 0, 7 and 28 and move section 6's heading below
cell 29.

---

### Claim 13 — "the four rules" resolves to nothing
**Verdict:** CONFIRMED
**Evidence:** cell 45, final paragraph (the phrase wraps across a line, which is
why a naive grep misses it):

```
You will forget most of the syntax. Keep the five reviewer questions, the four
rules, and the habit of writing the number down first.
```

The heading immediately above it is `### The course, in five rules` and lists
five numbered items. `"four rules"` appears in no other cell of this notebook
and in no other source module (grepped all 24). The five reviewer questions do
resolve. §3.3.
**Severity:** wrong but harmless
**Origin:** hand-written prose
**Fix:** `the four rules` → `the five rules`.

---

### Claim 14 — the header understates the download for a fresh runtime
**Verdict:** CONFIRMED
**Evidence:** cell 0: *"The same 200-entry catalogue as the previous lecture
(cached), plus two more checkpoints: a captioner (about 1 GB) and a small
instruction-tuned language model (about 1 GB)."* Measured from the HF cache
(`model.safetensors` per repo):

```
CLIP  openai/clip-vit-base-patch32          577 MB
BLIP  Salesforce/blip-image-captioning-base 944 MB
Qwen  Qwen2.5-0.5B-Instruct                 942 MB
MiniLM sentence-transformers/all-MiniLM-L6-v2 87 MB
                                    total  2550 MB = 2.49 GB
```

The two "about 1 GB" figures are accurate. CLIP and MiniLM are not mentioned,
and cell 8 and cell 35 load them unconditionally — a student who did not run
lecture 23 in the same runtime downloads 2.49 GB, not 2 GB. §7.1.
**Severity:** misleads a student
**Origin:** hand-written prose
**Fix:** state 2.49 GB cold, and note that CLIP and the images are already
present if lecture 23 ran in this runtime.

---

### Claim 15 — no CPU wall-clock figure anywhere; the only one given is for a GPU
**Verdict:** CONFIRMED
**Evidence:** cell 0: *"**Expected wall clock on a Colab GPU runtime:** five to
eight minutes end to end."* Cell 33: *"**Expected wall clock: 1–3 min**, most of
it the 1 GB download."* Cell 40: *"**Expected wall clock: 2–4 min**, most of it
the 1 GB download."* Neither of the latter two names a device, and both sit
under a header that has established GPU as the reference. Nothing in this
notebook trains — every model load is `from_pretrained(...).eval()` — so a CPU
runtime is adequate and is what most students have. §7.1, whose evidence is
that an untimed cell blocked 4 of 6 exercises in the lecture 19 audit.
**Severity:** misleads a student
**Origin:** hand-written prose
**Fix:** give the CPU number beside every ⏱ marker and in the header.

---

### Claim 16 — §8.3 is unmet
**Verdict:** CONFIRMED
**Evidence:** `"examinable"` appears three times in the whole notebook, and all
three are negative:

```
cell 16: ...research question, outside the book and not examinable. THAT it exists...
cell 18: ...research question, outside Chapters 1-16 and not examinable. That it exists...
cell 45: ...live research area, outside Chapters 1-16 and not examinable.
```

The notebook has nine `##` sections (cells 1, 9, 12, 18, 24, 27, 33, 40, 45).
None carries a positive *examinable* marking; six carry no marking at all. §8.3
requires one of the three labels on **every** section. `tools/prompts/lecture_24.md`
marks all nine correctly — the markings were simply never carried into the
artefact.
**Severity:** wrong but harmless
**Origin:** notebook structure
**Fix:** emit the script's `*Examinable: …*` line into each section heading cell.

---

### Claim 17 — cell 7's justification for `Fn` refers to a namespace that does not exist
**Verdict:** FALSE POSITIVE
**Evidence:** two errors in the claim.

First, the bullet is in cell **2** (the `setup` prompt box), not cell 7.

Second, and decisively, `F` **does** exist in the previous notebook's namespace.
`notebooks/lecture-23.ipynb` cell 38 and `tools/notebooks/lecture_23.py:576`:

```python
F = unit(clip_images(cifar_images))
print(f"{N_CIFAR} CIFAR-10 images encoded: {F.shape}")
```

and line 591 uses `(F @ W.T)`. So *"`F` is already the feature matrix in the
previous notebook's namespace"* is literally true. Phase A's observation that
`F` appears nowhere in lecture 24 is correct but does not bear on the claim —
the box is about lecture 23's `F`, and the whole point of `Fn` is that it
avoids the collision.

**What does survive, weakly:** if lecture 24 is self-contained (cell 0 and the
module docstring both insist it is), the two notebooks never share a kernel, so
a collision cannot occur and the stated reason is inapplicable. That is a mild
§3.3 inconsistency, not "a namespace that does not exist".
**Severity:** cosmetic
**Origin:** hand-written prose
**Fix:** none needed; optionally soften the reason to "`F` is the feature matrix
in lecture 23, so `Fn` keeps the convention readable across both".

---

### Claim 18 — section 7's opening presents a synthetic hole as a property of the data, and conflates score with rank
**Verdict:** CONFIRMED
**Evidence:** cell 33: *"Sixty of the two hundred entries have no description,
so they score exactly zero on the text route."* Cell 35 creates the hole
itself:

```python
blanked = np.array([i for i in range(N_CATALOGUE) if i % 10 < 3])
sim_missing[:, blanked] = -np.inf
```

`i % 10 < 3` is a rule the notebook applies, not a property of COCO. And the
blanked score is `-inf`, not zero. Measured:

```
rank of a blanked entry under -inf: unique values = {200}
R@1 on the 60 - human present 66.7%   deleted 0.0%
```

`ranks_of_truth` returns rank 200 for every blanked entry (since
`-inf >= -inf` is `True` for all 200 columns), which is what makes R@1 exactly
0.0%. Score and rank are different quantities and the sentence uses one for the
other, in a section whose whole subject is that "not in the index" and "scores
badly" are different failures.
**Severity:** misleads a student
**Origin:** hand-written prose
**Fix:** *"We blank sixty of the two hundred entries — the rule is `i % 10 < 3`
— by setting their column to `-inf`, so they are not in the text index at all
and their rank is 200 by construction."*

---

## Additional defects not in the Phase A report

### A1 — §6.1 is violated by a factor of 1.6, and the repo's own checker says so
**Verdict:** CONFIRMED
**Evidence:** `python3 tools/check_notebooks.py --advisory`:

```
FAIL  lecture-24.ipynb
        16 full annotations, budget is 10 (§6.1) — every reader in the audit
        stopped reading the template around cell 30
```

Re-derived independently: 16 prompt boxes (cells 2, 5, 7, 10, 13, 16, 19, 22,
25, 28, 31, 34, 36, 38, 41, 43) and **all 16** carry all three of
`Left open:` / `usual student version:` / `How you would catch it:`. §6.1 asks
for five to eight, never more than ten. The notebook makes this explicit in
cell 0: *"Every code cell in this notebook is preceded by a quoted prompt, and
three lines follow it"* — the budget is broken by design, not by drift.

This matters more here than elsewhere: §6.1's evidence is that all three audit
readers stopped reading the template *around cell 30*, and cell 29 is this
notebook's defect cell. It also compounds claim 12 — the fifth announcement of
the trap sits in the annotation block readers have already stopped reading.

Note that `tools/prompts/lecture_24.md` claims *"Seven cells carry the full
three-bullet annotation; the other nine carry the specification only."* The
artefact has sixteen. The script's own plan was never implemented.
**Severity:** misleads a student
**Origin:** notebook structure
**Fix:** cut nine annotation blocks to the specification only, keeping the full
form on cells 7, 19, 28, 31, 38, 43 (the six where the prompt genuinely fails).

### A2 — cell 27's "chance falls faster, so the gap widens" is contradicted by the table directly above it
**Verdict:** CONFIRMED
**Evidence:** cell 27 reads *"Accuracy falls with *B* and chance falls faster,
so the gap — the learning signal — widens."* Computed from cell 26's own printed
columns (restart-and-run-all state):

```
B=  2  acc  99.5%  chance  50.0%  gap  49.5%
B=  8  acc  98.4%  chance  12.5%  gap  85.9%
B= 32  acc  92.2%  chance   3.1%  gap  89.0%
B=128  acc  78.9%  chance   0.8%  gap  78.1%
B=200  acc  73.5%  chance   0.5%  gap  73.0%
```

The gap **peaks at B = 32 and then falls** — 89.0 → 78.1 → 73.0. And "chance
falls faster" is false over the top half of the table: from B = 128 to B = 200,
accuracy falls 5.4 points while chance falls 0.28 points, so accuracy falls
about 19× faster. The claim survives only between the two endpoints the prose
happens to quote (`tools/prompts/lecture_24.md`: *"99.5% against 50.0% is a
49.5-point edge, 73.5% against 0.5% is a 73.0-point one"*), which is §1.3 —
a statistic that does not survive a defensible re-partition of its own buckets.

The structural argument is available and unconfounded (§2.2): the *ratio*
accuracy / chance rises monotonically (1.99 → 7.9 → 29.5 → 101 → 147), and
`log B` — the ceiling the notebook already prints — rises without bound. Argue
the ratio, not the difference.
**Severity:** misleads a student — it is the notebook's stated "whole argument
for a large batch"
**Origin:** hand-written prose
**Fix:** *"Accuracy falls with B, but chance falls faster in ratio: the model is
2× chance at B = 2 and 147× chance at B = 200. That ratio, not the difference,
is the learning signal."*

---

## Corroborated Phase A "clean" findings

Re-checked independently and agree:

- **§5.1 / §5.2** — scanned all 30 markdown cells: 0 lines indented ≥4 outside a
  fence, 0 fence markers indented ≥4, 0 unclosed fences.
- **§3.1** — no ```` ```python ```` block in any markdown cell.
- **§3.3** — *"the concentration result from Lecture 10"* resolves: lecture 10 is
  *"Four thousand dimensions is too many"*, and `LECTURES.md:222` states the
  cross-reference explicitly. Lecture 23 line 227 computes R@5. Application 5 →
  application 12 is "seven applications later" ✓.
- **§1** — the catalogue is deterministic: entry 0 = `CAT-000042`, 200 unique
  SKUs, reproduced from the cached CSV.
- **§4.2** — no training cell exists; all loads are `from_pretrained(...).eval()`.
- CLIP and MiniLM are bit-reproducible on CPU here — every figure in the "Baseline
  reproduction" note above matched the script to the printed precision.

---

## Summary

```
confirmed: 16   false positive: 2   unverifiable: 0
of the confirmed, 13 mislead a student (3 wrong but harmless, 0 cosmetic)
origin split, the 16 confirmed — prose: 7   code: 6   structure: 3
  prose      2, 5, 6, 13, 14, 15, 18
  code       1, 3, 4, 7, 8, 10
  structure  9, 12, 16
origin split, all 18 real defects (incl. A1 structure, A2 prose)
           — prose: 8   code: 6   structure: 4
duplicates: none. Claims 1 and 2 share cell 32 but are independent defects —
  claim 2's bullet is wrong even with claim 1 fixed (0.786 vs 82.907 is not a
  small difference). Claims 3 and 4 are two faces of one cell but are separately
  fixable (wrong direction; missing control). Claim 12 and additional defect A1
  compound each other but are different guidelines (§8.1 vs §6.1).
partially verified: claim 6 (the `filled` column, 61.7/62.1/62.0%, needs BLIP)
  and claim 7 (the counts 0-of-21 / 12-of-12 need Qwen) — both confirmed on
  their structural substance without running generation.
```

Counting the two additional defects, the lecture-24 defect list stands at **18
real defects** (16 of Phase A's 18, plus A1 and A2).

The audit's prior — that defects concentrate in hand-written prose — holds only
weakly here: 7 of 16 confirmed defects are prose against 6 in generated code.
Both of Phase A's false positives are prose claims. The six code-origin defects
are not trivial ones:
claim 1 (`inf`), claim 3 (wrong direction) and claim 10 (`tau` rebound) each
change or destroy a printed headline number.
