# Triage — lecture 13 defect report

Against `tools/prompts/lecture_13.md` (§ *Defects found in the current
notebook*, items 1–14), `notebooks/lecture-13.ipynb` and
`tools/notebooks/lecture_13.py`.

**Stated once, not repeated per claim (per TRIAGE_BRIEF):** `lecture-13.ipynb`
has 64 cells, 20 code cells and **zero stored outputs** (`execution_count` is
`null` everywhere). No prose figure in this notebook can be reconciled against a
stored output; every number below was re-derived by running the notebook's own
code.

**How the re-derivation was done.** CIFAR-10 is cached at
`notebooks/datasets/cifar-10-batches-py`. Environment: python 3.13.5, torch
**2.13.0**, torchvision 0.28.0, numpy 2.3.5, CPU, 12 threads — the same torch
version the Phase A report used. Cells 3, 6, 12, 16, 20, 23, 44, 50 and 52 were
reproduced verbatim in
`…/scratchpad/tri13/{v13_base,v13_fwd,v13_bwd,v13_misc,v13_loss}.py`.
**No training cell was executed.** Where the notebook measures a *trained*
network (`deep`, cell 44) my measurement is on the freshly initialised net —
seed 42, `make_net()`, which is bit-identical to what `deep` starts from, since
cell 26 seeds immediately before `make_net()` and `train` does not
re-initialise. That substitution is flagged in every claim it affects.

---

### Claim 1 — the headline forward-collapse figures (2.42e-17, ×0.149, "sixteen orders") are float64 numbers, and the cell that produces them is float32
**Verdict:** CONFIRMED

**Evidence:** Cell 44 runs `activation_stats(deep, Xf)`. `deep` comes from
`make_net()` (default float32) and `Xf = torch.tensor(X_fit)` where `X_fit` is
`np.float32` — so the cell computes `h.std(dim=0).mean()` in float32. Running
cell 44's function verbatim on the seed-42 net, in each dtype
(`v13_fwd.py`):

```
                       layer 1 sd     layer 20 sd    L20:L1        per-layer
prose (cell 47)        1.29e-01       2.42e-17       "sixteen"     0.149
float64                1.2916e-01     2.5070e-17     1.9411e-16    0.148955
float32 — the cell     1.2916e-01     1.9777e-09     1.5312e-08    0.387870
                                                     (7.8 orders)
```

The float32 column is not a seed or training artefact, and the argument does not
depend on the weights. In float32 `h.std(dim=0)` for one unit is either
**exactly 0** (all 512 values identical) or at least one ulp apart; with
activations in [0.25, 1) the smallest ulp is 2.98e-08, so a single 1-ulp
disagreement in 512 values gives a column sd of **2.637e-09**
(measured: `torch.tensor(col).std()` = 2.637e-09), and even if only one of the
100 units is non-degenerate the mean-of-sds floor is **~2.6e-11** — six orders
of magnitude above the 2.42e-17 the prose quotes. In float32 the printed number
can be 0.0 or ≥ ~1e-11; **2.42e-17 is unrepresentable as an outcome of this
cell whatever the weights are.** Corroborating: at layer 15 and again at layer
20, unit 0 takes **exactly one distinct value** across all 512 images in
float32 (2 distinct values in float64 at layer 20).

The consequence for the reader: the lecture's headline sentence is *"Sixteen
orders of magnitude"* and *"a factor of 0.149 per layer"*, and the cell they run
prints **7.8 orders** and **0.388**.

**Severity:** misleads a student
**Origin:** hand-written prose (cell 47's figures; the underlying dtype choice
is in the generated cell 44)
**Fix:** compute `activation_stats` in float64 (`net.double()`, `X.double()`),
as cell 50 already does for the backward pass — then the cell prints the
numbers the prose quotes; the notebook's own float64 argument in cell 48 is
already the justification.

---

### Claim 2 — "Nothing could ever have crossed it" is false; exactly one activation crosses the saturation threshold
**Verdict:** CONFIRMED (at initialisation; see caveat)

**Evidence:** Counting `((h - 0.5).abs() > 0.45)` over all 20 hidden layers,
512 images × 100 units, seed-42 net (`v13_fwd.py`), identically in float32 and
float64:

```
total crossings over 20 layers: 1 of 1,024,000   (all of it at layer 1)
layer 1 column value           : 1/51,200 = 1.953e-05  -> ':.3f' prints 0.000
layers 2-20                    : 0
```

The rest of that paragraph is correct and I re-derived it:
`sd_all` at layer 20 = **0.0710** ("about 0.071" ✓), `0.45 / 0.0710 = 6.34`
("6.3 standard deviations" ✓), `sd_all` at layer 1 = **0.1311** and
`0.45 / 0.1311 = 3.43` ("3.4 even at layer 1" ✓), and `sd_all` stays in
0.058–0.080 across all twenty layers ("stays near 0.07" ✓). At 3.4 sd the
threshold is plainly reachable, which is why one element reaches it.

*Caveat:* the notebook measures `deep` after 20 epochs; I measured at
initialisation. Layer-1 weights barely move (that is the lecture's own thesis,
cell 59), so the count is unlikely to change, but I cannot demonstrate the
trained figure without training. The prose absolute is refuted at
initialisation.

Note also that the Phase A report quotes the crossing fraction as `1.953e-05`
while describing it as "out of 20 × 512 × 100"; 1.953e-05 is the **per-layer**
fraction at layer 1 (1/51,200). Out of 1.02 million it is 9.77e-07. Both
numbers are right, the sentence pairs the wrong two.

**Severity:** wrong but harmless
**Origin:** hand-written prose
**Fix:** replace "Nothing could ever have crossed it" with "One element in 1.02
million crossed it, all of it at layer 1, where the threshold is only 3.4 sd
away."

---

### Claim 3 — `0.1013` is not a validation accuracy the notebook can produce, and the axis is in percent
**Verdict:** CONFIRMED

**Evidence:** Cell 27's *usual student version* bullet: *"A y-axis from 0.0998
to 0.1013 turns pure noise into a trend."* The validation set is 5,000 images
(`N_VAL = 5_000`, cell 12), so any accuracy is a multiple of 1/5000 = 0.0002:

```
0.0998 * 5000 = 499.0    <- attainable
0.1013 * 5000 = 506.5    <- not an integer, unattainable
```

Separately, cell 28 plots `[100*a for a in hist["val_acc"]]`, so an autoscaled
axis on this notebook's data would run 9.98–10.13 **percent**, not 0.0998–0.1013.

**Severity:** wrong but harmless
**Origin:** hand-written prose
**Fix:** `"A y-axis from 9.98% to 10.14% turns pure noise into a trend"`
(506/5000 = 0.1012, 507/5000 = 0.1014).

---

### Claim 4 — `Epoch 5, Loss: 2.3026` cannot be printed by cell 37; the divisor is wrong by 79/78
**Verdict:** CONFIRMED

**Evidence:** Cell 37 prints `running/(len(Xf)//128)`. `len(Xf) = 10_000`
(`N_FIT`), and:

```
len(range(0, 10000, 128)) = 79       <- batches actually summed into `running`
10000 // 128              = 78       <- the divisor
last batch size           = 16
79 / 78                   = 1.0128205
```

Re-derived forward-only at initialisation, no optimiser step (`v13_loss.py`):

```
batches counted : 79
running / 79 = 2.3218     <- correct mean of per-batch means
running / 78 = 2.3516     <- what the cell prints
ratio        = 1.012821
```

At the plateau, where every batch mean is ln(10) = 2.302585, the cell prints
`2.302585 × 79/78` = **2.3321**, not 2.3026. So cell 36's `catch` bullet
(*"'Epoch 5, Loss: 2.3026' is five decimal places of nothing"*) quotes a string
the cell cannot emit, and cell 40's `print("The loss it printed, epoch by epoch,
never left ln(10) = " f"{math.log(10):.4f} either.")` asserts an equality the
reader's own output will contradict in the third decimal.

*Verification limit:* the 5-epoch loop is a training cell and was not run. The
79-vs-78 count, the 79/78 factor and the two forward-only figures above are
exact; 2.3321 follows from ln(10) × 79/78 and is exact to the extent that each
batch mean sits at ln(10).

(Aside, not a Phase A claim: 2.3026 is **four** decimal places, not five.)

**Severity:** misleads a student — the lecture's punchline is "anchor every loss
you print", and the anchor it hands the reader does not match what prints.
**Origin:** generated code (cell 37's `len(Xf)//128`), repeated as fact in prose
(cells 36, 40)
**Fix:** either divide by a counted `nb`, or — better, since cell 37 is
deliberately the assistant's code — leave the code and change cells 36/40 to
quote **2.3321** and say why it is not 2.3026. That turns the bug into the
lesson.

---

### Claim 5 — the forward probe and the backward probe are two different networks, and "exactly" holds for neither
**Verdict:** CONFIRMED

**Evidence:** By reading: cell 44 is `stats = activation_stats(deep, Xf)` —
`deep` is the network returned by `train(...)` in cell 26, after 20 epochs. Cell
50 is `g64 = grad_profile(make_net, X_fit, y_fit, …)`, and `grad_profile` does
`net = net_factory().to(dtype)` — a **fresh** net; cell 52's own plot title says
*"at initialisation"*. Cell 47 then claims 0.149 is *"exactly the rate at which
the gradient vanishes on the way back."*

Re-derived (`v13_bwd.py`, cell 50/52 verbatim, float64, 8 batches, seed 42):

| quantity | value |
|---|---|
| forward per-layer factor, float64, fresh net | **0.148955** |
| backward geometric mean over all 19 ratios — **what cell 52 prints** | **0.1593** |
| backward geometric mean over layers 2–20 only | 0.139566 |
| theory `0.25 · √100 · 0.057735` | 0.144338 |

Cell 52's full printed output, reproduced:

```
layer 20 : layer 1  =  1.4469e+15   (15.2 orders of magnitude)
per layer, going down: x 0.1593
check: 0.1593 ** 19 = 6.9115e-16   and 1/1.4469e+15 = 6.9115e-16
```

So the prose's 0.149 differs from the notebook's own printed 0.1593 by **6.9%**,
and from the theory value by 3.2%. The two figures for the same quantity are
never reconciled anywhere in the file — GUIDELINES §1.5 — and cell 60's
worksheet asks the reader to write down "Per-layer attenuation factor" while
cell 62's summary table prints `1/gain` = 0.1593 and cell 47 has told them
0.149. The physical claim (same mechanism both directions) is sound to about
±7%; *exactly* is not.

**Severity:** misleads a student — cell 53 tells the reader these are the numbers
the next lecture will predict from first principles, and the notebook gives two.
**Origin:** hand-written prose ("exactly"), compounded by notebook structure
(trained net vs fresh net)
**Fix:** run the forward probe on a fresh net too, or say "0.149 forward against
0.159 backward — the same mechanism, measured on two networks, agreeing to 7%".

---

### Claim 6 — "That is a straight line on a log axis" is false at layer 1
**Verdict:** CONFIRMED

**Evidence:** `g64` re-derived from cell 50 verbatim (`v13_bwd.py`):

```
 layer 1  4.5673e-17
 layer 2  2.6676e-17     <- SMALLER than layer 1
 layer 3  1.6671e-16
 …
layer 20  6.6082e-02
    head  4.8663e-01

ratio 1->2 : 0.5841   (layer 1 is 1.7121x LARGER than layer 2)
ratios 2->3 … 19->20 : 6.25, 7.70, 7.25, 6.93, 7.51, 7.19, 7.28, 7.29, 7.65,
                       7.16, 6.64, 6.67, 6.80, 6.98, 7.14, 7.77, 7.40, 7.59
```

Layer 1 is the only descent in the whole profile — its matrix is 3,072 wide, so
its fan-in differs from every other layer's. Log₁₀-linear fit:

```
fit over layers 1-20 : layer 1 residual +0.899 decades, max |residual| 0.899
fit over layers 2-20 : max |residual| 0.051 decades
```

Layers 2–20 are a straight line to within 0.05 decades; layer 1 sits **0.9
decades** off it and is visible as a kink on cell 52's own plot. Including it is
also what moves the reported factor from 0.1396 to 0.1593 (claim 5).

**Severity:** misleads a student — the reader is looking at the plot while the
prose denies what it shows, and cannot tell whether the kink is their error.
**Origin:** hand-written prose (cell 53)
**Fix:** "From layer 2 onward that is a straight line on a log axis. Layer 1
sits above it because its fan-in is 3,072, not 100 — which is the next lecture's
whole subject."

---

### Claim 7 — a 10% chance line drawn on a validation split whose majority class is 10.84%
**Verdict:** CONFIRMED (minor; a defence exists — stated below)

**Evidence:** Cell 28 draws `ax[1].axhline(10)` on the validation-accuracy
panel. Reproducing cell 12's split (`np.random.default_rng(42).permutation(50000)`,
`val_idx = order[:5000]`):

```
val class counts : [473, 542, 535, 482, 523, 473, 502, 469, 518, 483]
val majority     : 542 / 5000 = 0.108400
test counts      : [1000]*10, majority = 0.100000
fit  counts      : [992, 1027, 975, 998, 1033, 1028, 953, 1009, 993, 992], majority 0.1033
```

The notebook defines its baseline as majority-class accuracy (cell 16,
`baseline = counts.max()/counts.sum()`, computed on `y_test`). Applying the
notebook's own definition to the plotted split gives **10.84%**, not 10.00%; the
10% line belongs to the test split. Same kind as the §2.1 defect GUIDELINES was
written for, 0.84 points in size.

*The defence:* 10% is also the chance rate of a uniform guesser on ten classes,
which is defensible on any split. The defect is that the notebook never says
which of the two it means, and its own stated definition gives the other number.

**Severity:** cosmetic
**Origin:** generated code (cell 28), specified by the prompt box in cell 27
**Fix:** draw `100*y_val_majority` on the validation panel, or label the line
"chance, 10 classes" so it is not read as this split's baseline.

---

### Claim 8 — "all four of the previous lecture's silent failures": there are three, and the cell lists five
**Verdict:** CONFIRMED

**Evidence:** Cell 30's `Left open` bullet: *"that **all four** of the previous
lecture's silent failures would look exactly like this result."*

`lecture-12.ipynb` cell 60 (its red-team section) adds exactly **three**
questions to the standing five:

```
6. Is `opt.zero_grad()` inside the batch loop? …
7. Is there a `model.eval()` before every evaluation …
8. Is any metric a plain mean of per-batch values?
```

and lecture 12 has exactly **three** ⚠ sections: `## 8 · ⚠ The missing
zero_grad()`, `## 9 · ⚠ The missing model.eval()`, `## 11 · ⚠ Averaging the
metric per batch`.

Cell 31 enumerates **five** numbered checks (`# 1.` … `# 5.`), of which 1–3 are
lecture 12's three and 4–5 (label alignment, the overfit test) are new today.

The notebook contradicts itself inside the same bullet list: cell 30's
*usual student version* bullet says *"**three** of them are one line each to
rule out"*. Four matches neither three nor five.

**Severity:** misleads a student — GUIDELINES §7.3; a reader who counts finds
three or five and cannot tell whether the miss is theirs.
**Origin:** hand-written prose
**Fix:** "all three of the previous lecture's silent failures", and note that
checks 4 and 5 are new today.

---

### Claim 9 — the overfit diagnostic prints its verdict unconditionally
**Verdict:** CONFIRMED (structural; the printed accuracy itself is unverifiable)

**Evidence:** Cell 31, last three lines, verbatim:

```python
print(f"\n2 layers, 200 images, 200 steps -> training accuracy "
      f"{accuracy(tiny, Xf[:200], yf[:200]):.3f}")
print("The loop can memorise. So the loop is not the bug.")
```

`grep -c assert` on cell 31: **0**. The cell prints the conclusion whatever the
accuracy is. The contrast is inside the notebook: the very next code cell,
cell 34, ends with `assert sweep[2] > sweep[20], "if this fails, depth is not
the variable"`, and its prompt box (cell 33) explicitly praises *"an assert that
encodes the premise of the lecture"*. Cell 30's own box calls the overfit test
*"the strongest single diagnostic in deep learning"* — and then ships it as a
check that cannot fail.

I did not run cell 31 (it trains), so I cannot report the accuracy it prints;
the defect is structural and settled by the source.

**Severity:** misleads a student — this is precisely "a check that passed for the
wrong reason", the failure mode GUIDELINES was written around.
**Origin:** generated code
**Fix:** `assert accuracy(tiny, Xf[:200], yf[:200]) > 0.9, "the loop cannot
memorise 200 images — the loop is the bug"` before the print.

---

### Claim 10 — cell 38 tells the reader a Lecture-12 review of the assistant's cell finds nothing; question 8 catches it
**Verdict:** CONFIRMED (with a sharpening — see below)

**Evidence:** Cell 38: *"Not 'is the loop correct?' — it is; the `zero_grad`,
the `eval()` and the whole-set metric are all there, and a review looking for
Lecture 12's failures finds nothing."* Lecture 12's question 8 is *"Is any
metric a plain mean of per-batch values?"* Cell 37's loss is
`running/(len(Xf)//128)` — a sum of 79 per-batch means divided by 78 (claim 4,
measured 2.3516 against a correct 2.3218 at initialisation). Answering question
8 honestly on cell 37 gives *yes*, and the divisor makes it worse than a plain
mean.

**Sharpening.** The notebook's own `train` (cell 26) also averages the loss per
batch — `hist["loss"].append(total / nb)` — so "plain mean of per-batch values"
alone does not separate the assistant's cell from the notebook's. Cell 24 is
careful to claim only that *accuracy* is counted over the set. The defect that
does separate them, and that cell 38 misses, is the **wrong denominator**: 79
batches summed, divided by 78.

**Severity:** misleads a student — GUIDELINES §8.2 says the best trap is the
unlabelled one, and this is the one unlabelled defect in the file that would
catch a skimmer. The prose actively tells them it is not there.
**Origin:** hand-written prose (cell 38)
**Fix:** change cell 38 to "…finds one: question 8. Count the batches
`range(0, 10000, 128)` produces, then look at the divisor."

---

### Claim 11 — twenty prompt boxes, twenty full three-bullet annotations
**Verdict:** CONFIRMED

**Evidence:** Machine-counted over `lecture-13.ipynb`:

```
prompt boxes (markdown cells containing '**Prompt ·'): 20
  at cells [2, 5, 8, 11, 15, 19, 22, 25, 27, 30, 33, 36, 39, 43, 45, 49, 51, 55, 58, 61]
of those, containing ALL of 'Left open' + 'The usual student version'
  + 'How you would catch it'                          : 20
code cells                                            : 20
```

GUIDELINES §6.1: *"aim for five to eight per notebook, never more than ten"*
full annotations, with a **short** box on every other code cell. 20 is 2.5× the
ceiling. Cell 30 — the box immediately before the bug-ruling-out cell, and the
last one before the section the lecture turns on — is exactly where §6.1 says
readers have already stopped.

**Severity:** misleads a student (the reader stops reading the boxes before the
box that matters — the effect §6.1 records as measured, not hypothetical)
**Origin:** notebook structure
**Fix:** keep full three-bullet annotations on cells 30, 36, 39, 43, 49 (five)
and reduce the other fifteen to `input · output · constraint · check` only.

---

### Claim 12 — "examinable" appears exactly once, in a code comment; no markdown cell carries a marker
**Verdict:** CONFIRMED

**Evidence:** Case-insensitive search for `examinable` across all 64 cells
returns exactly one hit:

```
cell 3 [code]: # Not examinable: engineering hygiene, not machine learning. …
```

Zero hits in any markdown cell. The notebook has **twelve** `## `-level
sections (cells 1, 4, 10, 14, 17, 18, 24, 29, 35, 42, 60, 63). GUIDELINES §8.3
requires every section to carry one of *examinable* / *not examinable —
engineering* / *beyond the book, for context*. Twelve required, zero present —
and §8.3 was written about lecture 19, where the string appeared once.

**Severity:** wrong but harmless (an omission; nothing stated is false)
**Origin:** notebook structure
**Fix:** add an examinability marker to each of the twelve section headers.

---

### Claim 13 — the assistant defect is announced repeatedly before cell 37 runs
**Verdict:** CONFIRMED — but the count "four times" is not what is there

**Evidence:** Everything above code cell 37, in order:

* cell 35 heading: `## 9 · ⚠ An assistant writes the network`
* cell 35 body: *"It runs. It trains without error. It reports a number."*
* cell 35 last line: **"Read before running."**
* cell 36 box title: `**Prompt · ⚠ what the assistant returns**`
* cell 36 `Left open`: *"what would this number be if the model had learned
  nothing at all?"*
* cell 36 `The usual student version`: *"the result is indistinguishable from
  guessing"*
* cell 36 `How you would catch it`: *"'Epoch 5, Loss: 2.3026' is five decimal
  places of nothing"* — quoting the answer

That is **seven** flags, not four, and the Phase A report's own enumeration
lists three items while calling it four. The §8.1 violation is real and larger
than claimed; the number in the claim is wrong. Note the giveaway in cell 36's
`catch` bullet is also the string cell 37 cannot print (claim 4), so the
pre-announcement is both premature and inaccurate.

**Severity:** wrong but harmless — the pedagogy is blunted, nothing stated about
the data is false
**Origin:** notebook structure (cell ordering)
**Fix:** move cell 36's three bullets below cell 37, and open cell 38 with the ⚠
after the reader has written the number down. §8.1's "same words, same cells,
reordered".

---

### Claim 14 — no CPU wall-clock anywhere; "depending on the runtime" is the only qualification
**Verdict:** CONFIRMED for the §7.1 gap; the report's measured table is
UNVERIFIABLE here (it requires training runs)

**Evidence, by reading:** exactly three ⏱ markers exist, in markdown cells 4, 24
and 32 (a further three are echoes inside prompt-box titles at cells 5, 25, 33):

```
cell  4: ⏱ **1–3 minutes the first time** — about 170 MB over the network.
cell 24: ⏱ **about 40–90 seconds** for 20 epochs, depending on the runtime.
cell 32: ⏱ **about 2 minutes** for the whole sweep.
```

None gives a CPU figure. The only thing a CPU reader is told is cell 3's
`print("no accelerator found. Everything runs; it is slower.")`. GUIDELINES §7.1
requires the wall clock **and the CPU number** for every cell over ~20 s.

**Evidence, by measurement (no training cell executed):** timing 10 isolated
forward+backward+Adam steps on the seed-42 net, batch 128, 12 CPU threads
(`v13_misc.py`) gives **22.0 ms/step**, hence by extrapolation:

```
cell 26  20 epochs x 79 steps  ~ 35 s  + 20 validation passes   (notebook: "40-90 s")
cell 37   5 epochs x 79 steps  ~  9 s                           (notebook: untimed)
cell 50  measured directly, both dtypes: 0.08 s                 (notebook: untimed)
```

So the two stated figures are defensible on a 12-thread laptop CPU, and cell 37
at ~9 s sits under §7.1's 20 s threshold — the omission is the missing CPU/GPU
split, not a wrong number. The Phase A report's Colab column is its own
extrapolation and I have no Colab runtime either; that column stays
unverifiable.

**Severity:** wrong but harmless
**Origin:** hand-written prose
**Fix:** "⏱ about 40 s on a CPU, 15 s on a T4" on cell 24 and the equivalent on
cell 32, and drop "depending on the runtime".

---

## Independently re-derived and NOT defective

Checked because the claims lean on them; all clean, so the report is not
one-sided:

- **`nn.Linear` default init** (cell 23). Hidden matrix (100, 100):
  min −0.099996, max +0.099994 against `1/√100 = 0.100000`; measured sd
  **0.057888** against `b/√3 = 0.057735`, so cell 23's `assert abs(sd - b/√3) <
  0.002` passes with margin 1.5e-04. First layer: bound `1/√3072 = 0.0180422`,
  min −0.0180422, max +0.0180421. The prose claim U(−1/√fan_in, +1/√fan_in) is
  exactly right.
- **Parameter arithmetic** (cell 20). `3072×100 + 100 = 307,300`;
  `19 × (100×100 + 100) = 191,900`; `100×10 + 10 = 1,010`; total **500,210**,
  matching `sum(p.numel() …)` = 500210. 21 `nn.Linear` modules, so
  `assert len(lins) == DEPTH + 1` passes.
- **Shapes and balance** (cell 6). `(50000, 32, 32, 3)`, `(10000, 32, 32, 3)`,
  `set(np.bincount(ytr)) == {5000}`, `set(np.bincount(yte)) == {1000}`, one
  image = 3,072 numbers. All four asserts pass.
- **`ln(10) = 2.3026`** (cell 16) and **test baseline exactly 0.1000**.
- **The float64 gradient story** (cell 50). Largest relative float32/float64
  disagreement **5.62e-06**; zero float32 norms underflowed to exactly 0; the
  assert passes. The stated rationale is real — layer-1 gradient *entries* are
  around 1e-19 and their squares are below float32's smallest normal 1.18e-38.
  (The irony noted in claim 1 stands: that argument is made carefully here and
  not applied to cell 44.)
- **`+ 1e-7` on the scaler is inert on this data.** Smallest per-pixel sd over
  the fit subset: **0.22482**.
- **§5.1 / §5.2.** Zero markdown lines indented ≥4 spaces outside a fence across
  all 44 markdown cells; zero fence markers indented at all.
- **Cell 52's self-check reconciles.** `0.1593 ** 19 = 6.9115e-16` and
  `1 / 1.4469e+15 = 6.9115e-16` — the geometric mean does reproduce the
  end-to-end ratio, exactly as its prompt box promises.

## Could not verify

- Anything downstream of a training run: `deep`'s test accuracy, the sweep
  accuracies, the assistant's accuracy, epoch-1 and epoch-20 losses, cell 56's
  epoch-wise gradient history, cell 59's relative weight changes, cell 31's
  printed overfit accuracy. Training cells were not executed.
- Whether the *trained* net still shows exactly one saturation crossing (claim
  2) and a forward per-layer factor of 0.149 (claims 1, 5). Measured at
  initialisation instead; the float32-representability half of claim 1 is
  weight-independent and therefore unaffected.
- Colab wall clocks (claim 14).
- Rendered appearance in Colab. Source scanned mechanically for §5.1–5.2 and
  clean; the page itself not opened.

---

## Summary

```
confirmed: 14   false positive: 0   unverifiable: 0
of the confirmed, 8 mislead a student (1, 4, 5, 6, 8, 9, 10, 11)
origin split — prose: 8 (1, 2, 3, 5, 6, 8, 10, 14)   code: 3 (4, 7, 9)   structure: 3 (11, 12, 13)
duplicates:
  4 and 10 are one underlying defect — cell 37's `len(Xf)//128` — seen from two
    sides (4: the printed loss is not 2.3026; 10: cell 38 says a review finds
    nothing). Fixing the denominator, or documenting it, settles both.
  5 and 6 overlap on one number: including layer 1 in the geometric mean is
    what moves the backward factor from 0.1396 to 0.1593, which is also the
    size of the gap claim 5 reports. Distinct prose sentences, one cause.
  1 and 5 both concern the figure 0.149 but for different reasons (1: the cell
    is float32 and cannot produce it; 5: it is never reconciled with cell 52's
    0.1593). Both need fixing.
```

**Calibration note.** Lecture 13 contains none of the three pre-verified
calibration claims (they are in lectures 3 and 6). Fourteen of fourteen
confirmed is a higher rate than the brief's refute-by-default stance expects, so
it is worth stating plainly what the confirmations rest on: **eight of the
fourteen were settled by re-running the notebook's own cells and printing the
number** (1, 2, 3, 4, 5, 6, 7 and the timing arithmetic under 14), and the
remaining six (8, 9, 10, 11, 12, 13) by string-counting the file, not by
judgement. Where a claim's *wording* is inaccurate I have said so rather than
waving it through — claim 13's "four times" is wrong (there are seven flags),
claim 2's fraction is quoted against the wrong denominator, and claim 10's
diagnosis needed sharpening because the notebook's own training loop shares the
property it accuses the assistant of. The Phase A report for this lecture is
unusually careful; every figure in it that I re-derived came back identical to
four significant figures.
