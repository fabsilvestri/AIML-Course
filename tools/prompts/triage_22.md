# Triage — lecture 22

Claims from `tools/prompts/lecture_22.md` §*Defects found in the current
notebook*, against `notebooks/lecture-22.ipynb`.

**Count.** The task message says 31 claims; the Phase A report contains **24**
numbered entries (15 "checked and wrong", 5 "defensible", 4 "not checked").
All 24 are triaged below.

**Stated once, not repeated per claim (per the brief).** The notebook stores no
outputs: all 26 code cells have `execution_count: null` and zero outputs, so no
prose figure in this notebook can be reconciled against a stored output. This is
the subject of claim 14 and is the standing reason several figures had to be
re-derived from scratch rather than read off the file.

**Environment.** macOS, 16 cores, `OMP_NUM_THREADS=1` unless stated. IMDb read
from `notebooks/datasets/aclImdb` (25,000 train / 25,000 test). DistilBERT and
MiniLM were already in `~/.cache/huggingface`. Per the brief, **no training cell
was executed** (cells 30 and 42).

---

### Claim 1 — §6.1: 26 full annotations against a budget of 10

**Verdict:** CONFIRMED

**Evidence:**

```
$ python3 tools/check_notebooks.py 22
FAIL  lecture-22.ipynb
        26 full annotations, budget is 10 (§6.1) — every reader in the audit
        stopped reading the template around cell 30
1 violation(s) of GUIDELINES.md
```

72 cells total (46 markdown, 26 code); every one of the 26 code cells carries a
`Watch this prompt` block. Boxes sit at cells
`[2, 4, 7, 9, 12, 16, 18, 21, 24, 27, 29, 31, 34, 37, 39, 41, 43, 46, 49, 51,
55, 57, 60, 62, 65, 68]`. The eleventh is at cell 29 and the twelfth at cell 31
— the audit's "around cell 30" — and **cell 30 is literally this notebook's
primary defect cell**. GUIDELINES §6.1 names lecture 22's count (26) explicitly.

**Severity:** misleads a student
**Origin:** notebook structure
**Fix:** keep the short `input · output · constraint · check` box on every code
cell; reserve the three-bullet block for five to eight cells, cell 29 among them.

---

### Claim 2 — §3.2: the corrected specification does not catch the double softmax

**Verdict:** CONFIRMED

**Evidence:** cell 33 offers *"assert that an untrained model's loss on balanced
classes is within 0.05 of log 2"* as the fix, and cell 34's `Left open` bullet
says it catches *"a wrong head size, transposed targets, **a double softmax**,
and labels that are not what you think they are"*. I rebuilt cells 5, 28 and 35
verbatim and ran the assertion on the softmaxed model over six seeds
(`fit_y[:512].mean() = 0.5059`, so the batch is balanced):

```
seed  plain loss   softmaxed  |plain-log2|  |sm-log2|  plain_pass sm_pass  out.sum(1)
   0      0.6972      0.6919        0.0040     0.0012    True  True   plain 0.1992 / sm 1.000000
   1      0.6944      0.6907        0.0012     0.0024    True  True   plain 0.0334 / sm 1.000000
   2      0.7081      0.6985        0.0150     0.0053    True  True   plain 0.0132 / sm 1.000000
   3      0.7036      0.6964        0.0105     0.0032    True  True   plain 0.0419 / sm 1.000000
   4      0.6929      0.6892        0.0003     0.0040    True  True   plain 0.0653 / sm 1.000000
   5      0.7045      0.6963        0.0114     0.0031    True  True   plain -0.2417 / sm 1.000000
log 2 = 0.6931471805599453
```

The assertion **passes on the softmaxed model at every seed**, and at five of
six it lands *closer* to log 2 than the correct model does — seed 0 reproduces
the report's figures exactly (0.6919 against 0.6972). The mechanism is
structural, not luck: before training the logits are near zero, the extra
softmax squashes them towards a uniform row, and the cross-entropy of a uniform
two-class row *is* log 2. The check offered for independent verification cannot
fail on the bug it is offered for.

One correction to the report: its proposed replacement check reports the correct
model's `out.sum(1)` as averaging −0.016; I measure 0.199 at seed 0 and values
from −0.24 to +0.20 across seeds. The *check* is still sound — the softmaxed
model's rows sum to exactly 1.000000 and the correct model's sum to an arbitrary
number — but the −0.016 figure does not reproduce.

**Severity:** misleads a student
**Origin:** hand-written prose
**Fix:** replace the log-2 assertion with `assert not torch.allclose(out.sum(1),
torch.ones(len(out)))` (or assert `out` has values outside [0, 1]) and drop
"a double softmax" from cell 34's list.

---

### Claim 3 — §7.1: no CPU figure anywhere; the only two timings say "on a GPU"

**Verdict:** CONFIRMED (textual part measured; the fine-tune timing is
extrapolated, not run)

**Evidence:** the string `CPU` appears **0** times in the notebook. `GPU`
appears **twice**, both on timings:

```
cell 30 (code):     # ⏱ about 1-3 minutes for the two runs together, on a GPU.
cell 36 (markdown): ⏱ **about 1–3 minutes** to fine-tune, on a GPU.
```

Lowercase `cpu` occurs five times, all as `.cpu()` calls or `device = "cpu"` —
never as a timing. I could not run cell 42 (training, forbidden), so I measured
the forward pass only, DistilBERT on CPU with the notebook's own
`OMP_NUM_THREADS=1` (`torch.get_num_threads() == 1`), `max_length=192`,
batch 32:

```
256 reviews forward-only on CPU, 1 thread: 6.3s  -> 3,000 reviews ~ 74s
66,955,010 parameters
```

So each of the two 3,000-review scoring passes (cells 40 and 44) costs about
**75 s** on this machine — the report says 45 s; the order of magnitude is
right, the exact figure did not reproduce. A fine-tune step is forward plus
backward, so cell 42's 125 steps (2,000 / 16) cost roughly three times the
forward cost of 2,000 reviews, i.e. **two to three minutes** — consistent with
the report's 2.2 minutes, but not independently timed. The notebook's own
"1–3 minutes … on a GPU" is the only number a reader gets, and on a two-vCPU
Colab CPU runtime it is wrong by a large factor.

**Severity:** misleads a student
**Origin:** hand-written prose
**Fix:** give both numbers on cells 30, 42 and 47 — "about 1–3 min on a GPU,
about 10 min on a Colab CPU runtime".

---

### Claim 4 — §1.1 / §2.4: the 400-document leak result is inside its own noise

**Verdict:** CONFIRMED

**Evidence:** cell 61's code run verbatim (data from the notebook's own corpus):

```
400 docs, 20 seeds:  +0.30 points (sd 2.53), leak wins on 10/20   [22s]
stderr of mean: 0.566
400 docs, 60 seeds:  -0.32 points (sd 3.16), leak wins on 22/60   [65s]
```

Every figure in the claim reproduces exactly: `+0.30 (sd 2.53), 10/20`, standard
error 0.57, and **the sign flips at 60 seeds**. Per-seed gaps at 20 seeds range
from −6.00 to +5.00 points. Cell 64 nevertheless states *"At 400 the leaky
vocabulary has columns that exist because a test document used them"* and
generalises to *"it scales with the reciprocal of your corpus size"* — a
conclusion the measurement does not support. Cell 60's own bullet says *"a
single split proves nothing in either direction"*, and twenty is not enough
either. This is GUIDELINES §2.4 verbatim.

**Severity:** misleads a student
**Origin:** hand-written prose
**Fix:** per §2.2, argue the mechanism (60,000 leaky columns against 54,074
honest ones, deterministic and seed-independent) and report the accuracy as
"inside the noise at this size".

---

### Claim 5 — §1.1: at 25,000 documents the named mechanism is absent, not diluted

**Verdict:** CONFIRMED

**Evidence:** `leak_experiment(25_000, 3)` run verbatim, printing both
vocabulary sizes:

```
seed 0: leaky vocab 60000  honest vocab 60000  equal? True  gap +0.11  [69s]
seed 1: leaky vocab 60000  honest vocab 60000  equal? True  gap +0.10  [116s]
seed 2: leaky vocab 60000  honest vocab 60000  equal? True  gap +0.08  [164s]
25,000 docs, 3 seeds: +0.10 points (sd 0.01)
```

`max_features=60_000` binds in **both** arms at full size, so the two
vocabularies are byte-for-byte the same size and there are no columns that exist
"because a test document used them" — the mechanism is switched off, not
diluted. Only the idf weights differ. At 400 documents the honest vocabulary is
50,466–54,667 against a leaky 60,000, so the mechanism exists there and only
there. Cell 64's contrast ("an average over 25,000 draws … removing a quarter of
them barely moves any of them") describes idf dilution, which is real, but the
sentence next to it about columns is not what its own parameters do.

**Severity:** misleads a student
**Origin:** hand-written prose
**Fix:** say what actually differs at 25,000 — identical 60,000-column
vocabularies, different idf weights — or drop `max_features` so the vocabulary
mechanism is live at both sizes.

---

### Claim 6 — §3.1-adjacent: an assertion whose message is false where it passes

**Verdict:** CONFIRMED

**Evidence:** cell 61 line: `assert Z.shape[1] >= Ztr.shape[1], "the leaky
vocabulary must be larger"`. Measured (same runs as claims 4 and 5):

| corpus | leaky vocab | honest vocab | assertion |
|---|---|---|---|
| 400 | 60,000 | 50,466–54,667 | passes, message true |
| 25,000 | 60,000 | 60,000 | passes, message **false** |

Cell 60's `check ·` slot repeats the false framing: *"assert the leaky
vocabulary is at least as large as the honest one, **which is what makes it the
leaky one**"*. At full size it is not larger, and it is still the leaky one.

**Severity:** misleads a student
**Origin:** generated code (assert message) — the framing is repeated in prose
in cell 60
**Fix:** `"the leaky vocabulary can never be smaller"`, and state in cell 60
that at 25,000 the cap makes them equal. Same underlying defect as claim 5.

---

### Claim 7 — §7.1 / §1.1: cell 63's comment is wrong in both of its numbers

**Verdict:** CONFIRMED

**Evidence:** cell 63, first two lines:

```python
# ⏱ about a minute: five seeds at the full corpus size.
full = leak_experiment(25_000, 3)
```

`leak_experiment(n_docs, seeds)` — the second positional argument is `3`, so
**three** seeds, contradicted by the line directly above it. Cell 62's own
prompt box says "25,000 documents, three seeds" and its constraint says "three
points at one size and twenty at the other", so the code and the box agree and
the comment is the outlier. Timing, measured end to end on this machine:

```
TOTAL WALL CLOCK for leak_experiment(25_000, 3): 164s
```

164 s, not "about a minute" (the report measured 141 s; either way the comment
is off by roughly a factor of three).

**Severity:** misleads a student
**Origin:** generated code (a hand-written English comment inside a code cell)
**Fix:** `# ⏱ about three minutes: three seeds at the full corpus size.`

---

### Claim 8 — §3.3: "Note line 4 below" points at the wrong line

**Verdict:** CONFIRMED

**Evidence:** cell 45 (markdown) says *"Note line 4 below: the mean is over the
**real** tokens, using the attention mask."* Cell 47, numbered:

```
  4 | stk = AutoTokenizer.from_pretrained(MINILM)
 ...
 14 |         mask = enc["attention_mask"].unsqueeze(-1).float()
 15 |         pooled = (out * mask).sum(1) / mask.sum(1)      # real tokens only
```

Line 4 loads a tokenizer. The masked mean is line 15. This is GUIDELINES §3.3's
"ten cells earlier was fifteen" defect in a new file.

**Severity:** misleads a student
**Origin:** hand-written prose
**Fix:** "Note line 15 below", or drop the line number and quote the expression.

---

### Claim 9 — §3.3: three "next cell" references that are not the next cell

**Verdict:** CONFIRMED (the substance; the report's "two/three cells later"
distances are measured from the code cell each box annotates, not from the box)

**Evidence:**

| box | phrase | what the next cell actually is | where the thing is |
|---|---|---|---|
| cell 27 | "The defect this lecture is about is added in the NEXT cell, as an argument" | cell 28 — the `GRUClassifier` definition, defect-free by design | `double_softmax=True` is in cell 30 |
| cell 46 | "which makes the next cell one matrix product" | cell 47 — the `embed` function | `sims = Q @ V.T` is in cell 50 |
| cell 65 | "The next cell builds a corpus that is not clean, and says so" | cell 66 — the duplicate counter on clean IMDb | the duplicated corpus is built in cell 69 |

In each case a prompt box (and in two cases a prose cell) sits between. A reader
who follows the instruction literally at cell 27 looks at the model definition
for a defect that is not there.

**Severity:** misleads a student
**Origin:** hand-written prose
**Fix:** name the cell by its content — "in the training cell below", "in the
search cell", "in the last cell of this section".

---

### Claim 10 — §8.1: the primary defect is announced before it runs

**Verdict:** CONFIRMED as to the facts; the "four times" tally counts cell 26
twice

**Evidence:** everything before cell 30 that gives the answer away:

```
cell 0  (header):  "Cells marked **⚠ read before running** contain a defect on
                    purpose, and neither of today's two defects raises an exception."
cell 26 (heading): "## 6 · ⚠ Read before running — an assistant "improves" the model"
cell 26 (body):    "...on two classes the output can never exceed $e/(e+1) \approx 0.731$,
                    and the loss is floored near $-\log 0.731 \approx 0.313$."
cell 29 (box):     "⚠ two runs, one extra softmax" + "the loss is FLOORED near
                    −log(0.731) ≈ 0.313. That is the tell"
```

That is three cells and four statements, the last two of which give the exact
numeric answer. The same holds for the §10 leak: cell 0, cell 59's heading and
body ("Reviewer question 2: what was fitted, and on what?"), cell 60's box, and
cell 61's comment `# what the assistant wrote: fit on everything, then split`.
GUIDELINES §8.1 is explicit that this is the shape to avoid, and §8.2 that the
unlabelled trap is the one that catches readers. Nothing here is *false* — this
is a staging judgement, not a factual error.

**Severity:** wrong but harmless (pedagogically costly, factually correct)
**Origin:** notebook structure
**Fix:** move cell 26's body and cell 29's third bullet to *after* cell 32, per
§8.1's preferred shape. Two paragraphs move; no cell changes.

---

### Claim 11 — §8.3: "examinable" appears once in the whole notebook

**Verdict:** CONFIRMED

**Evidence:**

```
$ grep -oi "examinable\|beyond the book" notebooks/lecture-22.ipynb | sort | uniq -c
   1 examinable
```

The single occurrence is in cell 3's code comment: *"Not examinable, and only
needed on macOS"*. There are eleven `##` sections (cells 1, 6, 11, 14, 23, 26,
36, 45, 54, 59, 71), and none of the other ten carries any of §8.3's three
markers.

**Severity:** wrong but harmless
**Origin:** notebook structure
**Fix:** one marker per `##` heading.

---

### Claim 12 — §4.1: six names rebound to a different kind of object

**Verdict:** CONFIRMED, and understated — `p` has three meanings, not two

**Evidence:** AST walk over every binding (assignments and loop targets,
including nested tuple targets) in all 26 code cells:

```
z       [(8, assign), (13, assign), (22, assign)]
q       [(25, assign), (50, loop),  (52, loop)]
g       [(19, assign), (63, loop),  (69, assign)]
p       [(5, loop),    (13, assign),(22, assign), (25, loop), (38, loop)]
onehot  [(13, assign), (25, assign)]
m       [(8, assign),  (58, assign)]
naive   [(17, assign), (22, assign), (69, assign)]
```

Reading the objects off the cells: `z` float32 numpy array (8) → float64
`requires_grad` torch tensor (13); `q` Dirichlet probability vector (25) → query
**string** as a loop variable (50, 52) — the `target` clobbering of lecture 19,
unremarked; `g` the loss-level x-axis array (19) → per-seed accuracy gaps as a
loop variable (63) → a group-id array (69), three meanings; `p` a `Path` in the
loader loop (5) → a torch softmax tensor (13) → a numpy distribution (25);
`onehot` torch tensor (13) → numpy array (25); `m` a float32 scalar `z.max()`
(8) → a boolean cluster mask (58); `naive` per-row losses (17) → per-seed
random-split accuracies (69) — same dtype, opposite meaning.

`check_notebooks.py --advisory` independently flags nine constructor collisions
(`a`, `e`, `g`, `idx`, `opt`, `out`, `y`, `z`, `z64`), a partly different set,
so both lists understate the total.

**Severity:** wrong but harmless (`naive` and `q` are the dangerous two)
**Origin:** generated code
**Fix:** throwaway names for throwaway loops — `qs`, `gap`, `nv`.

---

### Claim 13 — §4.2: cell 42 is not idempotent, and the notebook never says so

**Verdict:** CONFIRMED

**Evidence:** cell 38 loads `model = AutoModelForSequenceClassification
.from_pretrained(BERT, num_labels=2)`. Cell 42 begins:

```python
torch.manual_seed(RANDOM_STATE)
opt   = torch.optim.AdamW(model.parameters(), lr=2e-5)
...
model.train()
```

It constructs the optimiser but not the model, so re-running it trains a second
epoch on the same 2,000 reviews. `zero_shot`, computed from `model` in cell 40,
is still printed by cell 44's table — after cell 42 has mutated the object it
was measured from. Contrast cell 30, which *is* idempotent because the model is
constructed at the call site (`train(GRUClassifier(VOCAB))`).

```
$ grep -oi "re-run\|rerun\|idempot\|restart\|run it again" notebooks/lecture-22.ipynb
(no output)
```

Nothing in the notebook warns about it. GUIDELINES §4.2 was written for lecture
19's cell 40, which is the same defect.

**Severity:** misleads a student
**Origin:** generated code
**Fix:** re-load the model inside cell 42, or add
`# re-run this cell only after re-running cell 38`.

---

### Claim 14 — §1.2: not one cell has a stored output

**Verdict:** CONFIRMED on the substance; the "47 prose figures" figure is wrong

**Evidence:**

```
total cells 72   Counter({'markdown': 46, 'code': 26})
exec counts Counter({'None': 26})
outputs nonempty 0
```

All 26 code cells have `execution_count: null` and zero outputs, so 0.731,
0.313, 88.7, log 2 and the parameter count reconcile with nothing, the §1.2
machine check cannot pass, and §10 pre-flight item 1 has left no evidence.

The report's "47" is the total advisory-note count, which mixes two checks.
Broken out by calling the checker's functions directly:

```
numbers advisory (§1.2):     38
quoted-code advisory (§3.1):  0
rebinding advisory (§4.1):    9      # 38 + 0 + 9 = 47
```

So **38** prose figures match no stored output, not 47.

**Severity:** misleads a student (the reader can check nothing)
**Origin:** notebook structure
**Fix:** execute and commit with outputs; correct the count to 38 wherever it is
quoted.

---

### Claim 15 — §7.1: the plot's clamp warns about the wrong hazard

**Verdict:** CONFIRMED

**Evidence:** cell 17's sweep run verbatim:

```
loss   1  naive:   0.0% non-finite, median err 2.860e-08   stable: 0.0%, 3.902e-08
loss   5  naive:   0.0% non-finite, median err 1.971e-08   stable: 0.0%, 2.354e-08
loss  10  naive:   0.0% non-finite, median err 2.462e-08   stable: 0.0%, 2.629e-08
loss  20  naive:   0.0% non-finite, median err 2.577e-08   stable: 0.0%, 2.594e-08
loss  40  naive:   0.0% non-finite, median err 2.770e-08   stable: 0.0%, 2.774e-08
loss  60  naive:   0.0% non-finite, median err 1.829e-08   stable: 0.0%, 1.844e-08
loss  80  naive:   0.0% non-finite, median err 2.757e-08   stable: 0.0%, 2.785e-08
loss  90  naive:   0.0% non-finite, median err 2.758e-08   stable: 0.0%, 2.498e-08
loss 100  naive:   0.0% non-finite, median err 2.569e-04   stable: 0.0%, 2.173e-08
loss 110  naive: 100.0% non-finite, median err nan         stable: 0.0%, 2.017e-08

any exactly 0.0 ? False
finite naive median errors: 1.829e-08 .. 2.569e-04
```

No median error is ever 0.0, so cell 18's warning — *"A median error of exactly
0.0 is not plottable on a log scale and matplotlib's response is to drop the
point silently"* — describes a case that does not occur. The case that *does*
occur is the `nan` at loss 110, and `np.maximum` does not repair it:

```
np.maximum(med, 1e-9) -> [2.86e-08 ... 2.57e-04 nan]
finite drawn points: 9 of 10
no warning raised
```

The naive curve stops at loss 100 and matplotlib says nothing. The annotation
names the right hazard (a silently dropped point) for the wrong reason, and the
clamp it recommends does not prevent the drop that actually happens.

**Severity:** misleads a student
**Origin:** hand-written prose (the clamp itself is harmless code)
**Fix:** replace the clamp with `np.nan_to_num(..., nan=<floor>)` or plot the
non-finite rate as the reason the point is missing, and reword the bullet to
name `nan`, not 0.0.

---

### Claim 16 — §5.1: the one indented markdown line is not a violation

**Verdict:** CONFIRMED (the report's finding of *no violation* is correct)

**Evidence:** exactly one markdown line in the notebook is indented ≥ 4 spaces
outside a fence:

```
cell 11 line 5: '    = -z_c + \\log\\sum_j e^{z_j}$$'
```

Rendered through `markdown_it` in strict CommonMark mode:

```html
<p>$$L = -\log p_c = -\log \frac{e^{z_c}}{\sum_j e^{z_j}}
= -z_c + \log\sum_j e^{z_j}$$</p>
--- contains <pre> or <code>?  False
```

It stays inside the paragraph, because an indented code block cannot interrupt a
paragraph. `check_notebooks.py`'s `check_indentation` returns an empty list for
this notebook. No fence marker anywhere is indented (there are no fences in any
markdown cell at all).

**Severity:** not a defect — this entry records a non-violation
**Origin:** hand-written prose
**Fix:** none needed

---

### Claim 17 — §3.1: no quoted `python` block, so nothing to mismatch

**Verdict:** CONFIRMED (the report's finding of *no violation* is correct)

**Evidence:**

```
md cells 46
python fences in md: 0
any fence in md: 0
$ grep -c '```python' notebooks/lecture-22.ipynb
0
```

`check_quoted_code` returns zero notes.

**Severity:** not a defect — this entry records a non-violation
**Origin:** hand-written prose
**Fix:** none needed

---

### Claim 18 — cross-references to other lectures resolve

**Verdict:** CONFIRMED (the report's finding is correct)

**Evidence:**

```
L21 'padding_idx':      5
L21 'pack_padded':      2
L21 'last-of-padding':  1
L09 'silhouette':      46 occurrences
```

(The report says 40 for silhouette; `grep -c` counts *lines*, `grep -o | wc -l`
counts occurrences — 46. Either way the reference resolves.) "The same rule as
application 5": `LECTURES.md` states *"Applications are covered in pairs of
consecutive lectures"*, so application 5 is lectures 9–10, which is the
clustering application (chapter 8 · Unsupervised learning → lectures 9, 10).
Correct.

**Severity:** not a defect — this entry records three resolved references
**Origin:** hand-written prose
**Fix:** none needed

---

### Claim 19 — cell 65's `norm` constraint is false on this corpus

**Verdict:** CONFIRMED

**Evidence:** cell 65's constraint says *"normalise whitespace and case before
comparing — an exact string match finds fewer duplicates than there are"*. Run
both ways over the notebook's own corpus:

```
normalised: cross-split 123  within-train 96
raw exact : cross-split 123  within-train 96
```

Identical. The normalisation costs nothing and buys nothing on IMDb. Unlike the
other prose defects this one is a claim that is false rather than a number that
is wrong, and it would be true on most other corpora — hence "may".

**Severity:** wrong but harmless
**Origin:** hand-written prose
**Fix:** "an exact string match **may** find fewer duplicates than there are".

---

### Claim 20 — cell 67's "IMDb is clean" sits above a cell printing 123 duplicates

**Verdict:** CONFIRMED as to the facts; whether "clean" is defensible is a
judgement I agree with

**Evidence:** cell 66 prints, on this corpus, 123 test reviews present verbatim
in training and 96 duplicates within training. 123 / 25,000 = **0.492%**. Cell
67 opens *"IMDb is clean, because its authors deduplicated it — which also means
its cost cannot be measured here."* The second half of that sentence is the load
-bearing claim and it is correct: 0.49% is far too small to move a measurement.

**Severity:** cosmetic
**Origin:** hand-written prose
**Fix:** "IMDb is nearly clean — 123 of 25,000 test reviews, 0.5%, appear
verbatim in training — because its authors deduplicated it".

---

### Claim 21 — every accuracy and every training loss is unverified

**Verdict:** UNVERIFIABLE (correctly declared)

**Evidence:** the brief forbids executing training cells. Cells 30 and 42, and
everything downstream (cells 32, 44), were not run. Unverified as a consequence:
*"the extra softmax costs N points"*, the four accuracies in cell 44's table,
the error-reduction figure, and `assert ft_acc > 0.75`. Note that the notebook
stores no outputs either, so nothing in the file can substitute.

**Severity:** n/a
**Origin:** n/a
**Fix:** run cells 30, 32, 42 and 44 once before the lecture and commit the
outputs.

---

### Claim 22 — whether the softmaxed model's loss actually reaches the 0.313 floor

**Verdict:** UNVERIFIABLE for the trained model; the floor itself CONFIRMED

**Evidence:** the algebra checks out exactly:

```
e/(e+1) = 0.7310585786300049   -log = 0.3132616875182228
CE(softmax([1e6, -1e6]), class 0) = 0.31326165795326233
```

A maximally confident double-softmax row hits the floor to seven digits. What I
could not test is cell 33's actual claim — *"It stops above 0.3, exactly where
the algebra said it would"* — because that requires running cell 30. If the
correct model also finishes near 0.4 after two epochs on 5,000 reviews, the two
loss columns are close and the "tell" the section is built on is not visible in
the output. This is the one claim in the notebook worth running before the
lecture, and it is one cell.

**Severity:** n/a — but this is the notebook's central claim
**Origin:** hand-written prose
**Fix:** run cell 30 and quote the two measured final losses in cell 33.

---

### Claim 23 — the macOS KMeans deadlock in cell 3's comment

**Verdict:** UNVERIFIABLE for the deadlock; the report's *other* measurement
CONFIRMED

**Evidence:** reproducing a deadlock means hanging a kernel on purpose, so I did
not. The load-bearing side effect of the same line is measurable, and it
reproduces:

```
OMP_NUM_THREADS=1   fit times ['0.011', '0.009', '0.009']  best 0.009s
OMP_NUM_THREADS=16  fit times ['3.744', '3.524', '2.290']  best 2.290s
(16 cores)
```

A ~250× difference on cell 61's `LogisticRegression(max_iter=2000, C=4.0)` fit
— the report measured 0.01 s against 2.33 s, "more than 200×", which reproduces.
`os.environ.setdefault("OMP_NUM_THREADS", "1")` is load-bearing on this hardware
whether or not the deadlock reproduces, so the comment's *instruction* (set it
before importing torch) is right even where its stated *reason* is untested.

**Severity:** n/a
**Origin:** generated code (comment)
**Fix:** none needed; optionally add the measured speed-up as a second reason.

---

### Claim 24 — the Colab wall-clock multiplier is an estimate

**Verdict:** UNVERIFIABLE (correctly declared)

**Evidence:** every timing in the report and in this triage is from a 16-core
macOS laptop. No Colab runtime was used. The "three to five times slower on a
two-vCPU Colab CPU runtime" figure is inference from core count, not
measurement. My own CPU measurement (75 s per 3,000-review DistilBERT forward
pass at one thread) is consistent with it but does not confirm it.

**Severity:** n/a
**Origin:** hand-written prose
**Fix:** measure once on a free Colab CPU runtime before quoting a CPU number.

---

## Summary

```
confirmed: 20   false positive: 0   unverifiable: 4
```

Of the 20 confirmed, **three are verified non-defects** — claims 16, 17 and 18,
where Phase A said "checked, and defensible" and I agree. That leaves **17
confirmed defects**:

```
of the 17 confirmed defects, 12 mislead a student
  claims 1, 2, 3, 4, 5, 6, 7, 8, 9, 13, 14, 15
wrong but harmless: 4   claims 10, 11, 12, 19
cosmetic:           1   claim 20

origin split (all 24 entries) — prose: 14   code: 5   structure: 4   n/a: 1
  prose:     2, 3, 4, 5, 8, 9, 15, 16, 17, 18, 19, 20, 22, 24
  code:      6, 7, 12, 13, 23
  structure: 1, 10, 11, 14
  n/a:       21

origin split (17 confirmed defects only) — prose: 9   code: 4   structure: 4
  prose:     2, 3, 4, 5, 8, 9, 15, 19, 20
  code:      6, 7, 12, 13
  structure: 1, 10, 11, 14

The audit's hypothesis — defects concentrate in hand-written prose — holds here
but less sharply than in lecture 19: 9 of 17 are prose, and the four structural
ones (annotation budget, defect staging, missing examinable markers, no stored
outputs) are properties of how the notebook is assembled rather than of anything
written in it.

duplicates:
  claims 5 and 6 are the same underlying fact — `max_features=60_000` binds in
    both arms at 25,000 documents, so the two vocabularies are equal. Claim 5
    calls it a false prose contrast; claim 6 calls it a false assert message.
    One fix settles both.
  claims 4 and 5 are adjacent but distinct: 4 is about the 400-document result
    being inside its noise, 5 about the 25,000-document mechanism being absent.
  claim 3 and claim 7 both fail §7.1 (missing / wrong wall-clock) but on
    different cells and for different reasons.
  claim 10's second half (the §10 leak announced in cells 0, 59, 60, 61)
    overlaps nothing else.
```

**Corrections to the Phase A report** (every claim below still stands; five of
the numbers offered in support of them do not reproduce):

| Report says | Measured |
|---|---|
| claim 2: correct model's `out.sum(1)` averages −0.016 | 0.199 at seed 0; −0.24 to +0.20 across seeds. The check still works — the softmaxed model's rows sum to exactly 1.000000 |
| claim 3: each 3,000-review scoring pass is 45 s | ~75 s (extrapolated from 6.3 s / 256 reviews, CPU, 1 thread) |
| claim 7: `leak_experiment(25_000, 3)` took 141 s | 164 s here. "About a minute" is wrong either way |
| claim 14: 47 prose figures match no stored output | 38. The 47 is 38 numeric notes + 9 §4.1 rebinding notes |
| claim 18: lecture 9 mentions silhouette 40 times | 46 occurrences (40 is the line count) |

**Not a correction, an addition:** claim 12 understates. `p` is bound to three
different kinds of object, not two — a `Path` in cell 5's loader loop, a torch
tensor in cell 13, a numpy distribution in cell 25 — and `g` likewise has three
(loss levels, accuracy gaps, group ids).

**Calibration note.** This lecture contains none of the three independently
verified calibration claims (lectures 3 and 6).
