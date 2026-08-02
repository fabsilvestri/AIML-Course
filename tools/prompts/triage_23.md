# Triage — lecture 23 defect report

`notebooks/lecture-23.ipynb` (50 cells, 16 code) against
`tools/prompts/lecture_23.md` § *Defects found in the current notebook*.

**Scope note.** The task message said 12 claims; the Phase A report contains
**17** numbered claims. All 17 are triaged below.

**Note once, not repeated per claim (per the brief).** All 16 code cells have
`execution_count: null` and `outputs: []`. No prose figure in this notebook can
be reconciled against a stored output. This is the subject of claim 2 and is the
enabling condition for claim 1; it is not re-litigated elsewhere.

## How the verification was done

Every checkpoint and dataset the notebook needs was already cached, so the real
pipeline was run rather than estimated:

| artefact | location | status |
|---|---|---|
| `google/vit-base-patch16-224` | HF hub cache | cached (346 MB `model.safetensors`) |
| `sentence-transformers/all-MiniLM-L6-v2` | HF hub cache | cached (90 MB) |
| `openai/clip-vit-base-patch32` | HF hub cache | cached (605 MB) |
| COCO split index + 200 images | `notebooks/datasets/app12/` | cached (5.0 MB + 200 files, 32.6 MB) |
| CIFAR-10 test tarball | `notebooks/datasets/` | cached (170,498,071 B) |

Script: a faithful cell-by-cell reconstruction of all 16 code cells, run on CPU
with `SEED = 42`, `torch 2.13.0`, `transformers 4.57.3`, `numpy 2.3.5`,
`HF_HUB_OFFLINE=1`. **Nothing was downloaded.** Full log reproduced inline below
where each claim needs it.

Every figure in `tools/prompts/lecture_23.md` reproduced exactly — 0.007 / 18%,
d = 0.066 and 4.10, R@1 75.0% / 73.5%, 61.5% → 46.0%, 66.7% → 0.0%, 80.0%,
59.3% → 65.7%, 88.0/90.6/90.6%, norms 8.66–11.44 and 6.28–11.25, 151M
parameters, 9-of-200 head overlap, 154 whitespace-changing captions. The Phase A
numbers are trustworthy; the disagreements below are with the **notebook**, not
with the report.

### Mathematical spine, checked independently

The four items the task brief asked for, all pure numpy:

```
CONCENTRATION  (4,000 random unit-vector pairs per d)
  d=    2 mean +0.00880 sd 0.70813  1/sqrt(d) 0.70711  min -1.0000
  d=    8 mean +0.00449 sd 0.35252  1/sqrt(d) 0.35355  min -0.9329
  d=   32 mean +0.00406 sd 0.17555  1/sqrt(d) 0.17678  min -0.5176
  d=  128 mean +0.00108 sd 0.08793  1/sqrt(d) 0.08839  min -0.3301
  d=  512 mean +0.00098 sd 0.04424  1/sqrt(d) 0.04419  min -0.1790
  d= 2048 mean -0.00079 sd 0.02247  1/sqrt(d) 0.02210  min -0.0778
```

Mean 0 and sd exactly 1/√d, to three significant figures at every d. One
correction to the brief's phrasing: "nothing is anywhere near −1" holds from
d = 8 up, but **at d = 2 the cosine does reach −1.0000** — in the plane two
random unit vectors are antipodal with positive density. The concentration
argument is a large-d argument and should be stated as one.

```
arithmetic baseline n=200 : R@1 0.5%  R@5 2.5%  R@10 5.0%  expected rank 100.5
simulated over 20 runs    : R@1 mean 0.575%  min 0.0%  max 1.0%  (se ~0.50%)
```

The simulated baseline's standard error (0.50%) is the same size as the quantity
(0.5%) — the notebook's "compute it in closed form" argument is correct and the
margin is not marginal.

```
zero matrix, strict '>'  R@1 = 100.0%
zero matrix, with '>='   R@1 =   0.0%   (every rank = 200)
ranks_of_truth(np.zeros((200,384))) -> AssertionError: (200, 384)
ranks_of_truth(np.zeros((200,200))) -> ok
joint dual encoder, text->image  R@1 75.0%   (sim_clip)
joint dual encoder, image->text  R@1 73.5%   (sim_clip.T, same matrix)
```

All four hold: the tie rule flips 100% ↔ 0%, the square assert fires, and
transposing the same matrix measures the other direction and gives a different
number.

---

### Claim 1 — cell 27 quotes "mean similarity 0.14"; cell 28 prints 0.007

**Verdict:** CONFIRMED

**Evidence:** cell 27's bullet reads *"**The usual student version:** reading
'mean similarity 0.14' as evidence of anything."* Cell 28, run on the real 200
cached images:

```
CELL28 mean similarity of an image and its caption: 0.007
CELL28 18% of pairs score above 0.05
CELL31 matched  mean +0.0070 sd 0.0503 (n=200)
```

0.14 is 20× the value the cell it annotates produces. It cannot be run-to-run
noise: the sd of the diagonal is 0.0503, so 0.14 sits 2.6 sd from the mean of a
distribution of 200 values whose own mean is pinned at 0.0070. Scanning
section 7's markdown (cells 26–32) for numeric literals, `0.14` is the **only**
concrete measured figure offered anywhere in the section:
cell 27 → `['0.05', '200', '39,800', '0.14', '5']`; the rest are the
candidate-set size, the 39,800 arithmetic, and the threshold.

**Severity:** misleads a student

**Origin:** hand-written prose (`student=` field, `tools/notebooks/lecture_23.py:420`)

**Fix:** replace `0.14` with `0.007` in cell 27's bullet, and re-derive it
whenever cell 28 changes.

---

### Claim 2 — zero stored outputs, so §1.2 cannot be satisfied anywhere

**Verdict:** CONFIRMED

**Evidence:**

```
execution_count set across the 16 code cells : {None}
total stored outputs                          : 0
```

The figures the report names are all present in markdown and none is
reconcilable: `0.14` (cell 27), *"Section 7's d was near zero"* (cell 44),
*"zero, structurally"* (cell 49). One correction to the report: *"all three land
at chance"* is in **cell 24**, not quoted as a figure — it is a qualitative
claim, and it happens to be true (1.0% / 0.5% / 0.0% against a 0.5% baseline).

The claim's closing point is the important one and it is correct: d = 0.066 and
4.10, the three chance routes, and the structural zero all reproduce exactly, so
they are right — but nothing in the artefact demonstrates that, and `0.14` sat
in the same markdown with the same authority.

**Severity:** misleads a student

**Origin:** notebook structure

**Fix:** execute and store outputs before shipping; §1.2 and
`tools/check_notebook_numbers.py` are inoperative without them.

---

### Claim 3 — cell 15 says "no error and no warning"; transformers warns

**Verdict:** CONFIRMED

**Evidence:** two clean subprocess runs, stderr captured at fd level:

```
===== add_pooling_layer=True =====
  Some weights of ViTModel were not initialized from the model checkpoint at
  google/vit-base-patch16-224 and are newly initialized: ['pooler.dense.bias',
  'pooler.dense.weight']
  You should probably TRAIN this model on a down-stream task ...
  [end]
===== add_pooling_layer=False =====
  [end]
```

The warning is printed on `True` and absent on `False`. Supporting checks:

```
tensors in checkpoint: 200
any named pooler: []
bit-identical pooler weights after torch.manual_seed(42): True
pooler weight std: 0.01999   (i.e. N(0, 0.02^2))
```

So every factual sub-claim in the lecture script holds — 200 tensors, no
`pooler`, freshly drawn N(0, 0.02²), reproducible under the seed — and the
notebook's constraint (cell 15) is the one thing that is false. It is also
self-contradicting: the same box's third bullet says *"when a checkpoint warns
that weights were newly initialised, read it"*, advice that is unusable if there
is no warning. The source module's code comment
(`tools/notebooks/lecture_23.py:285`) says only *"with no error"*, which is
correct; the prompt box added *"and no warning"*.

**Severity:** misleads a student

**Origin:** hand-written prose (prompt-box `constraint=`, `lecture_23.py:273`)

**Fix:** delete "and no warning" from cell 15's constraint.

---

### Claim 4 — cell 48's "46.0%" is over 140 reachable candidates, printed "of 200"

**Verdict:** CONFIRMED

**Evidence:**

```
text query -> text description     R@1  61.5%  R@5 89.0%  R@10 96.0%  median rank 1 of 200
...with 60 descriptions deleted    R@1  46.0%  R@5 65.5%  R@10 68.5%  median rank 2 of 200
R@1 on those 60: with description 66.7%, with none 0.0%
CLAIM4 other 140 rows: full 59.3% -> missing 65.7%
CLAIM4 reachable columns after blanking: 140
```

`report` computes `n = len(r)`, which counts **rows** (200 queries), while
`sim_missing[:, blanked] = -np.inf` removed 60 **columns**. So the second row
ranks 200 queries against 140 reachable entries and prints *of 200*.

Splitting the rows confirms the report's decomposition exactly: the 60 blanked
entries go 66.7% → 0.0%, the other 140 **rise** 59.3% → 65.7%. The headline
15.5-point drop is a structural zero averaged against a free gift. The
notebook's own standing rule — *"a recall without its candidate-set size is not
a number"* (cell 3's comment, cell 2's constraint) — is broken by its own cell.

**Severity:** misleads a student

**Origin:** generated code (`report`'s `of {n}` where `n = len(r)`)

**Fix:** print the reachable-candidate count, not the row count, and show the
two row-halves separately instead of the aggregate.

---

### Claim 5 — "under-specified in exactly one place" — it is at least four

**Verdict:** CONFIRMED

**Evidence:** cell 26 reads *"Under-specified in exactly one place. Find it
before you run the cell."* The quoted prompt is *"Encode the catalogue images
with a ViT and the captions with a sentence transformer, then check whether an
image and its caption are similar."* Four omissions, each one the notebook
itself treats as substantive:

1. **no control group** — cells 29–31 exist entirely for this.
2. **which caption goes on which side** — cell 8's constraint and cell 9's
   `assert descriptions[0] != queries[0], "query and description must differ"`.
3. **how 768 and 384 are made comparable** — cell 22 prints
   `ValueError: ... size 384 is different from 768` (reproduced in my run),
   proving it cannot be done without a choice; cell 25 makes three of them.
4. **what "similar" means** — cell 28 applies a `0.05` threshold. Grepping the
   quoted prompt in cells 26 and 27: `0.05` appears in the box's *output* line
   and in the code, and **nowhere in the quoted prompt**.

GUIDELINES §7.3: a reader who genuinely looks and finds four cannot tell whether
the failure is theirs.

**Severity:** misleads a student

**Origin:** hand-written prose (cell 26)

**Fix:** "under-specified in at least four places — find them", or drop the count.

---

### Claim 6 — "what the weak prompt returns" silently uses `R` from cell 25

**Verdict:** CONFIRMED

**Evidence:** cell 28 in full (6 lines):

```python
# --- what the weak prompt returns --------------------------------------------
Vp = unit(unit(V) @ R)
sim_matched = (Vp * unit(T)).sum(axis=1)      # cosine of each matched pair

print(f"mean similarity of an image and its caption: {sim_matched.mean():.3f}")
print(f"{(sim_matched > 0.05).mean():.0%} of pairs score above 0.05")
```

`R` is `rng.normal(0, 1/np.sqrt(384), size=(768, 384))`, defined in cell 25. The
quoted prompt asks for no projection, and cell 22 — two code cells earlier —
demonstrates that without one the multiplication raises `ValueError`. So the
cell labelled *"⚠ what the weak prompt returns"* shows what it returns after an
unacknowledged repair. GUIDELINES §4.4. The lecture script already identifies
this and prescribes the fix ("Say so in the markdown"); the notebook does not.

**Severity:** wrong but harmless — the number produced is right and the section's
lesson survives; the provenance label is what is false.

**Origin:** hand-written prose (the cell label and section framing)

**Fix:** one sentence in cell 26 saying the reconstruction borrows `R`, because
the literal prompt could not have produced any number at all.

---

### Claim 7 — the trap is announced four times before it fires

**Verdict:** CONFIRMED

**Evidence:** counted in the notebook, above a **6-line** code cell (cell 28):

1. cell 0 — *"Cells marked **⚠ read before running** contain a defect on purpose."*
2. cell 26 heading — *"## 7 · ⚠ Read before running — the assistant failure"*
3. cell 26 body — *"Under-specified in exactly one place. Find it before you run the cell."*
4. cell 27 label — *"> **Prompt · ⚠ what the weak prompt returns**"*

This is the same count GUIDELINES §8.1 makes against lecture 19 (header,
how-to-use note, section heading, paragraph above), and §8.1's preferred shape —
run it unannounced, have the reader write the number down, *then* open the next
section with the ⚠ — is exactly what the lecture script's own staging note for
cell 9 prescribes.

**Severity:** wrong but harmless — nothing stated is false, but the section's
whole purpose is defeated.

**Origin:** notebook structure

**Fix:** move all four flags after cell 28, into cell 29.

---

### Claim 8 — "run the IDENTICAL analysis" — cell 45 is not identical to cell 31

**Verdict:** CONFIRMED

**Evidence:** cell 44's constraint: *"run the IDENTICAL analysis as section 7 —
… it is only a comparison if the computation is the same"*. Cell 30's own catch
bullet: *"`ddof=1` on both variances, and **n printed beside each**."* The two
cells differ:

| | cell 31 | cell 45 |
|---|---|---|
| `(n = …)` beside each mean | yes, both | **absent, both** |
| Cohen's d format | `{…:+.3f}` | `{…:.2f}` |

Printed output from my run:

```
CELL31 difference +0.0034   Cohen's d +0.066
CELL45 difference +0.1522   Cohen's d 4.10
```

The two numbers the reader is instructed to compare arrive with different
precision, different sign convention, and only one carrying the 200-vs-39,800
asymmetry that cell 30 says the reader "should see".

**Severity:** misleads a student

**Origin:** generated code (cell 45's format strings)

**Fix:** copy cell 31's three `print` statements into cell 45 verbatim, changing
only the input matrix.

---

### Claim 9 — two cross-references off by two and four cells

**Verdict:** CONFIRMED

**Evidence:** counted against the cell list.

* Cell 10: *"you will see why three cells from now."* The tie demonstration is
  the last four lines of **cell 12** — `zeros = np.zeros((n, n))` … `with a
  strict '>' a constant scorer reports R@1 = …`. Cells 11, 12, 13 are
  markdown / code / markdown, so cell 12 is **two** cells later (and the *first*
  code cell after, under either counting convention). "Three cells from now" is
  cell 13, the ✍️ Commit box.
* Cell 21: *"reaching for a projection immediately, which is the next cell."*
  The next cell is 22, which is the `try/except` printing the `ValueError`. Code
  cells after 21 are `[22, 25, 28, 31]`; `R` is defined in **cell 25**, four
  cells later.

GUIDELINES §3.3.

**Severity:** wrong but harmless

**Origin:** hand-written prose

**Fix:** "two cells from now" and "which is two code cells further on".

---

### Claim 10 — "application 3" and "Lecture 6" for the same pointer

**Verdict:** CONFIRMED (as an accurate observation; see caveat)

**Evidence:** cell 39 — *"a hyperparameter chosen on the test set — application
3, in a new costume"*; cell 41 — *"a hyperparameter chosen on the test set —
Lecture 6, in a new costume"*. Two cells apart, same content.

Both resolve. `LECTURES.md` maps chapter 4 (Training models) to Lectures 5–6,
applications run in consecutive pairs (this notebook's own cache dir is
`datasets/app12` for Lectures 23–24), so application 3 = Lectures 5–6; and
`tools/notebooks/lecture_06.py:870` is the choose-on-the-test-set demonstration
(*"score each candidate on the test set and report the best"*, seventeen models).

House style, by grep over `tools/notebooks/*.py`: "application N" is used in
lectures 9, 17, 18, 21, 22. `lecture_23.py:602` is the only place in the course
that writes "Lecture 6, in a new costume".

Cell 23's *"thread 5"* likewise resolves — `LECTURES.md` thread 5 = *SVD, PCA
and Johnson–Lindenstrauss (L10)* — and lecture 21 writes the fuller form.

**Caveat.** No GUIDELINES rule is actually broken: §3.3 requires cross-references
to *resolve*, and all three do. This is a house-style inconsistency, not a
defect. Ranking it with the others would overstate it.

**Severity:** cosmetic

**Origin:** hand-written prose

**Fix:** use "application 3" in both places, and "thread 5, from Lecture 10".

---

### Claim 11 — cell 29 forward-references a list in cell 49

**Verdict:** CONFIRMED

**Evidence:** cell 29 — *"In the five reviewer questions this is number 5 — the
default nobody asked for is the missing control group."* The five questions are
listed once, in **cell 49**, which is the last cell of a 50-cell notebook —
twenty cells later. A reader at cell 29 cannot check it. The claim is correct on
the merits: cell 49's item 5 is *"What is the default I did not ask for?"*

**Severity:** wrong but harmless

**Origin:** notebook structure

**Fix:** name the question inline, or move the five-question list to the front.

---

### Claim 12 — the header's download budget omits CIFAR-10's 170.5 MB

**Verdict:** CONFIRMED

**Evidence:** cell 0: *"A 5 MB index of the COCO validation split, then 200
individual images (about 30 MB), then three model checkpoints (about 1 GB in
total)."* Measured on disk:

```
COCO index      5,011,921 B  = 5.0 MB   ("5 MB")        ok
200 images                   = 32.6 MB  ("about 30 MB") ok
ViT   model.safetensors      = 346 MB (330 MiB)
MiniLM model.safetensors     =  90 MB  (87 MiB)
CLIP  model.safetensors      = 605 MB (577 MiB)   -> 994 MiB ("about 1 GB") ok
CIFAR-10 tarball   170,498,071 B = 170.5 MB       -> NOT IN THE HEADER
```

All three stated items are accurate. CIFAR-10 is a **fourth** download, stated
in cell 36 (*"1–2 min for the 170 MB download"*) and fetched by cell 38's
`CIFAR10(root="datasets", train=False, download=True)`, and it is missing from
the header. Stated ≈ 1,035 MB against an actual ≈ 1,202 MB, a 16%
understatement, in the one paragraph a reader on a metered connection reads.

Minor caveat on the report's arithmetic: it mixes MiB (models) with MB (files).
Consistently decimal the actual total is ≈ 1,249 MB and the understatement ≈ 21%.
The substance — a 170 MB download missing from the header — is unaffected.

**Severity:** misleads a student

**Origin:** hand-written prose (cell 0)

**Fix:** add CIFAR-10 to the header list and restate the total as ≈ 1.2 GB.

---

### Claim 13 — "three to five minutes" excludes its own upper bound

**Verdict:** CONFIRMED

**Evidence:** cell 0 states *"**Expected wall clock on a Colab GPU runtime:**
three to five minutes end to end."* Summing the notebook's own per-section
budgets, all located by grep:

| cell | stated | min | max |
|---|---|---|---|
| 4 | catalogue, "about one minute" | 1.00 | 1.00 |
| 14 | ViT, "20–60 s" download + "about 15 s" | 0.58 | 1.25 |
| 33 | CLIP, "1–2 min" + "a few seconds" | 1.05 | 2.10 |
| 36 | CIFAR, "1–2 min" + "about 10 s" | 1.17 | 2.17 |
| | **total (minutes)** | **3.80** | **6.52** |

3.8–6.5 minutes against a stated 3–5. The lower bound is inside the range; the
**upper bound is not**. And this is before the MiniLM download (untimed
anywhere) and section 11's re-encode of 400 sentences, which I measured at 2.9 s
and which carries no ⏱ marker.

**Severity:** misleads a student

**Origin:** hand-written prose (cell 0)

**Fix:** state 4–7 minutes, or state that the range is compute-only.

---

### Claim 14 — no CPU figure anywhere

**Verdict:** CONFIRMED

**Evidence:** grep for `CPU|GPU|Colab` across all 50 cells returns three hits,
all in cell 0, and none is a CPU budget:

```
cell 0: **Expected wall clock on a Colab GPU runtime:** three to five minutes
cell 0: (same sentence, second match)
cell 0: ...built cell by cell against Colab's Gemini 3.1 Pro...
```

The string `CPU` does not occur in the notebook. Cell 3 nonetheless selects
`cuda` → `mps` → **`cpu`**, so the notebook expects CPU readers and never
budgets for them. Measured on this 12-thread Apple-Silicon CPU, checkpoints
cached:

```
ViT, 200 images            16.6 s
MiniLM, 200 queries         0.9 s
CLIP, 200 images + 200 texts 12.5 s
CLIP, 500 CIFAR images     17.7 s
MiniLM re-encode, 400 sents  2.9 s
```

≈ 51 s of compute plus JPEG decode — on a 2-vCPU Colab CPU runtime that is the
3–5 minute compute budget the notebook never states.

One honest qualification to the report's §7.1 framing: on *this* CPU no single
cell crosses the 20 s threshold (the largest is 17.7 s), so §7.1's mechanical
trigger would not fire here. The defect is the absent runtime, not a missing ⏱
on a specific cell. GUIDELINES §7's headline finding stands regardless: the
literal reader lost 4.5 of 6 exercises to exactly this.

**Severity:** misleads a student

**Origin:** hand-written prose (cell 0)

**Fix:** add a CPU column to the header budget.

---

### Claim 15 — sixteen full annotations where the budget is five to eight

**Verdict:** CONFIRMED

**Evidence:** scripted over the notebook:

```
prompt-box cells        : [2, 5, 8, 11, 15, 18, 21, 24, 27, 30, 34, 37, 39, 42, 44, 47]  (16)
carrying "Watch this prompt" 3-bullet block : 16
boxes with NO "check ·" clause              : [2, 21, 27, 30, 39, 44]  (6)
```

Every one of the 16 boxes carries the full three-bullet block, against
GUIDELINES §6.1's five-to-eight target and never-more-than-ten cap. The §6.3
sub-claim also holds: six boxes have no `check ·` slot, and §6.4 identifies that
slot as the one that structurally forces an expected answer.

The lecture script's proposed budget — full annotations on cells 2, 4, 5, 9, 10,
13, 16 (seven) — is inside §6.1.

**Severity:** wrong but harmless — nothing stated is false, but §6.1's evidence
is that fatigue makes readers stop around the cell where the defect lives, and
here that cell is 28.

**Origin:** notebook structure

**Fix:** reduce to the seven full annotations the lecture script names; the rest
keep the short specification box.

---

### Claim 16 — nothing is marked examinable

**Verdict:** CONFIRMED

**Evidence:** the string "examinable" occurs exactly once in all 50 cells:

```
cell 3 (code): # Not examinable: version hygiene. It is here because a ...
```

It is inside a Python comment, in the setup cell, where no reader looking for
section labels will find it. The notebook has twelve numbered sections and zero
carry *examinable* / *not examinable — engineering* / *beyond the book, for
context*. GUIDELINES §8.3.

**Severity:** wrong but harmless

**Origin:** notebook structure

**Fix:** add one label per section heading.

---

### Claim 17 — "the five human-written captions each" is wrong for 10 rows

**Verdict:** CONFIRMED

**Evidence:** cell 4 says the index lists *"5,000 COCO 2014 validation images
with the five human-written captions each"*. Parsing the cached CSV:

```
CLAIM17 caption-count distribution over 5000 rows: {5: 4990, 6: 10}
CLAIM17 catalogue-200 caption counts:              {5: 200}
```

4,990 rows carry five captions and **10 carry six**. Harmless for this notebook —
all 200 catalogue entries do have exactly five, and the code's assert is the
tolerant `len(e["captions"]) >= 2` — but the prose asserts a property of the file
that the file does not have. GUIDELINES §1.1.

**Severity:** wrong but harmless

**Origin:** hand-written prose (cell 4)

**Fix:** "five captions each (ten rows carry six)".

---

## Summary

```
confirmed: 17   false positive: 0   unverifiable: 0
of the confirmed, 9 mislead a student
origin split — prose: 10   code: 2   structure: 5
duplicates: none are the same underlying defect, but three clusters share a location —
  · 12, 13, 14 are three distinct errors in one paragraph (cell 0): a missing
    download, a wall clock whose range excludes its own maximum, and no CPU
    figure. One rewrite of cell 0 fixes all three.
  · 5 and 7 both concern the staging of section 7 (cell 26) but cite different
    rules (§7.3 "exactly N findable things" vs §8.1 "do not announce the trap").
  · 1 and 2 are causally linked — 0.14 survived because there is no stored
    output to contradict it — but are separately actionable.
```

### Confidence and its limits

Unusually for this triage, **nothing was refuted**. That is a real result rather
than a rubber stamp, and the reason is that all three checkpoints and both
datasets were already cached, so every claim was settled by running the real
pipeline rather than by argument: 12 of the 17 are arithmetic or counting claims
I re-derived from scratch, and the remaining 5 are behavioural claims I executed.
The strongest evidence that the Phase A report was careful is that every one of
the ~25 independent figures in `lecture_23.md` reproduced to the stated
precision on a different machine.

Two claims I would rank lower than the report does, and said so above:

* **Claim 10** breaks no GUIDELINES rule — §3.3 requires cross-references to
  resolve, and "application 3", "Lecture 6" and "thread 5" all do. Cosmetic.
* **Claim 14**'s §7.1 framing is slightly off: no individual cell exceeds 20 s on
  a fast CPU, so §7.1's mechanical trigger would not fire. The underlying defect
  (a CPU-capable notebook with no CPU budget) is real.

### Not checked

* Whether the two encoder training-data claims are exactly true as stated ("no
  text anywhere in `google/vit-base-patch16-224`'s training", "no image" for
  MiniLM). Correct per the model cards; I did not audit pretraining corpora.
* Wall clock on an actual Colab runtime. All timings above are a 12-thread
  Apple-Silicon CPU with everything cached.
* Download durations. Sizes are measured from disk; durations depend on network.
* Restart-and-run-all in Colab. I executed a faithful cell-by-cell
  reconstruction as a script, in notebook order, with no errors and no forward
  references — not the `.ipynb` itself.
