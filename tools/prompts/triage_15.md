# Triage — lecture 15 defect report

Artefact: `notebooks/lecture-15.ipynb` (58 cells, 17 code cells).
Claims triaged: **26** — items 1–17 of *Verified by execution or arithmetic*,
the six *Checked and found clean* bullets (18–23), the three *Not checked*
bullets (24–26).

**Verification environment.** Apple M4 Max, macOS 26.5.2, Python 3.13.5,
torch 2.13.0, torchvision 0.28.0, MPS available, no CUDA. Flowers102 was
**already cached** under `notebooks/datasets/flowers-102/` (tarball + extracted
`jpg/` + `setid.mat` + `imagelabels.mat`), so nothing was downloaded and every
data claim is re-derivable. **No training cell was executed.** Timing figures
come from a 3-step benchmark on random tensors, not from a training run.

**Two caveats that apply to every entry, stated once.**

1. **No stored outputs.** All 17 code cells carry `outputs: []` and
   `execution_count: null`. No prose figure in this notebook can be reconciled
   against a stored output; every number below was re-derived from the cached
   dataset or from a network constructed from cell 23's source.
2. **The machine was under heavy load during verification** — load average
   **172** on 16 cores, from ~20 sibling triage agents. Absolute wall-clock
   figures below are therefore upper bounds. Where this matters (claims 2, 3, 4,
   17) it is said in the entry. The per-step benchmarks nevertheless landed
   within 10 % of the report's own figures, so the load did not distort them.

---

### Claim 1 — §6.1: all 17 code cells carry a prompt box *and* the full three-bullet annotation

**Verdict:** CONFIRMED

**Evidence:** Counted from the JSON — 17 markdown cells begin `> **Prompt`
(cells 2, 5, 8, 11, 15, 18, 22, 25, 28, 32, 35, 39, 42, 46, 49, 52, 55), the
same 17 contain `Watch this prompt`, and every one of the 17 has exactly three
`* **` bullets. Not one short box exists. The repo's own checker agrees:

```
$ python3 tools/check_notebooks.py 15
FAIL  lecture-15.ipynb
        17 full annotations, budget is 10 (§6.1) — every reader in the audit
        stopped reading the template around cell 30
1 violation(s) of GUIDELINES.md
```

**Severity:** misleads a student — the audit's readers stopped reading the
template around cell 30, which in this notebook is the parameter-count lesson
(cells 28–30), the one place the annotation is load-bearing.

**Origin:** notebook structure

**Fix:** keep the short box everywhere; reserve the three bullets for the seven
cells the script nominates (2, 5, 7, 10, 12, 13, 16).

---

### Claim 2 — §7.1: cell 31's "several minutes on CPU" is about 87 minutes

**Verdict:** CONFIRMED

**Evidence:** Cell 31 reads *"⏱ **about 40 seconds on a GPU or MPS, several
minutes on CPU.** 30 epochs of 32 batches."* Benchmarked the exact architecture
(3 steps after a warm-up, batch 32, 128×128, Adam, cross-entropy):

```
mps: fwd+bwd batch32   26 ms | fwd batch128   24 ms
cpu: fwd+bwd batch32 3238 ms | fwd batch128 3851 ms
```

One epoch of cell 33 is 32 training steps plus 16 evaluation batches of 128
(8 for `X_train`, 8 for `X_val`) = **165 s on CPU**, so 30 epochs = **82.6
minutes**. The report says 87 minutes; the difference is machine load, and both
are ~20× "several minutes". The GPU half of the sentence is right: 30 × 1.2 s =
**36 s on MPS**, consistent with "about 40 seconds".

**Severity:** misleads a student — this is the sentence the no-GPU reader plans
their evening around.

**Origin:** hand-written prose

**Fix:** "about 40 s on a GPU or MPS, **about 90 minutes on CPU**".

---

### Claim 3 — §7.1: cell 51's "⏱ about 60 seconds" carries no CPU figure and is the most expensive cell on a CPU

**Verdict:** CONFIRMED

**Evidence:** Cell 51 says *"⏱ **about 60 seconds.**"* and cell 52's box header
repeats *"⏱ 60 s — measure the damage"*; neither mentions CPU. Cell 53 runs
`quick_train` four times at 12 epochs: 4 × (12 × 32) = 1,536 training steps plus
4 validation passes (32 batches of 128). At the measured CPU rates that is
1,536 × 3.238 s + 32 × 3.851 s = **85 minutes** (report: 93 minutes) against
1,536 × 0.026 s ≈ **41 s on MPS**, so "about 60 seconds" is fine for the GPU
path. 48 epochs here against cell 33's 30 — on a CPU runtime this cell is longer
than the training cell it follows.

**Severity:** misleads a student

**Origin:** hand-written prose

**Fix:** add "about 90 minutes on CPU — longer than the training cell".

---

### Claim 4 — §7.1: cell 40 is untimed and walks the 6,149-image test set twice

**Verdict:** CONFIRMED

**Evidence:** The only cells containing `⏱` are 4, 5, 31, 32, 51 and 52 —
cells 38 and 39, which precede the test cell, carry none. Cell 40 calls
`accuracy(net, X_test, y_test)` twice, i.e. 2 × ⌈6149/128⌉ = **98** forward
passes of 128. At the measured rates: 98 × 3.851 s = **6.3 minutes on CPU**,
98 × 0.024 s = **2.4 s on MPS**. Cell 0 promises *"Anything that takes more than
about twenty seconds says so before it starts."*

*Correction to the claim:* its parenthetical says "48 forward passes of 128",
which is half the true count and does not yield its own 6.1-minute figure; 98
passes does. The minute figure is right, the arithmetic beside it is not.

**Severity:** misleads a student

**Origin:** notebook structure (a missing marker)

**Fix:** `⏱ 2.4 s on GPU/MPS, about 6 minutes on CPU — this cell evaluates twice.`

---

### Claim 5 — §3.3: cell 15's "eleven sections from now" is seven

**Verdict:** CONFIRMED

**Evidence:** `## ` headings sit at cells 1, 4, 10, 14, 17, 20, 21, 31, 38, 41,
48, 57 — sections 1 to 12. Cell 15 lies between the section-4 heading (cell 14)
and the section-5 heading (cell 17), so it is **in section 4**. The assistant
failure is section 11 (cell 48). 11 − 4 = **7**. There are only twelve sections,
so counting eleven forward from four runs off the end of the notebook. The same
cell's `constraint` line says *"section 11 needs to call this with a different
pair"* — correct — so cell 15 contradicts itself four lines apart. Cell 5's
"section 8" also resolves (section 8 is cell 31, and cell 32's box carries the
1.2 GB memory point it promises).

**Severity:** wrong but harmless

**Origin:** hand-written prose

**Fix:** "in section 11".

---

### Claim 6 — §1.1: "Four thousand parameters per image" is 4,713

**Verdict:** CONFIRMED

**Evidence:** Constructed the network from cell 23's source and counted:

```
convolution + batch-norm     586,720   12.20%
one dense layer            4,194,560   87.25%
output layer                  26,214    0.55%
total                      4,807,494
4,807,494 / 1,020 = 4713.23  ->  "4,713"
```

Cell 29's own last line prints `{total / len(X_train):,.0f}` = **4,713**, while
cell 28's `catch` bullet says *"Four thousand parameters per image"*. §1.2
requires conventional round-half-up: 4,713 to the nearest thousand is **five**
thousand. Prose and printed output disagree on the same screen.

**Severity:** cosmetic — the argument the sentence makes is unaffected.

**Origin:** hand-written prose

**Fix:** "4,713 parameters per training image".

---

### Claim 7 — §1.2: "nine parameters in ten" should be 8.7 in ten

**Verdict:** FALSE POSITIVE

**Evidence:** `Linear(16384, 256)` holds 4,194,560 of 4,807,494 parameters =
**87.2496 %**. Rounded conventionally (round-half-up, the rule §1.2 names),
8.72 in ten is **9 in ten**. Taking both non-convolutional layers together it is
(4,194,560 + 26,214)/4,807,494 = **87.80 %**, which also rounds to 9 in ten. The
claim's own replacement figure — "8.7 in ten" — is a more precise statement of
the same number, not a correction of a wrong one, and the assertion that the
figure "rounds the wrong way" is not supported by the arithmetic. Contrast
claim 6, where 4,713 genuinely rounds to five thousand and the prose says four:
that one is a real rounding error and this one is not.

**Severity:** n/a

**Origin:** n/a

**Fix:** none needed. (Optionally quote 87.2 %, which is what cell 29 prints.)

---

### Claim 8 — §1.1: "they differ in the third decimal place" (cell 55)

**Verdict:** CONFIRMED (imprecise as written)

**Evidence:** Re-derived from the cached dataset with the notebook's own
transform:

```
MEAN train ['0.4330', '0.3819', '0.2964']
MEAN all   ['0.4355', '0.3777', '0.2880']
per-channel difference  +0.00258  -0.00419  -0.00845
max |dmean| 0.008454   max |dstd| 0.003818
```

The first *differing digit* is the third decimal for red only; for green
(0.38 vs 0.37) and blue (0.29 vs 0.28) it is the **second**. Cell 50 prints
`largest difference in any channel mean: 0.0085` a few lines above the sentence,
so the two sit visibly side by side.

*In fairness to the notebook:* under the other natural reading — "the difference
is of order 10⁻³" — all three channels qualify (0.0026, 0.0042, 0.0085 are all
below 0.01). The sentence is loose rather than false, and the point it is making
(no assertion on the values distinguishes the two conditions) is exactly right.

**Severity:** cosmetic

**Origin:** hand-written prose

**Fix:** "they differ by at most 0.0085 in any channel".

---

### Claim 9 — §3.2: cell 56's assert cannot fail

**Verdict:** CONFIRMED — and it is the most serious defect in this notebook

**Evidence:** The cell is

```python
n_used_for_stats = len(X_train)
assert n_used_for_stats == len(X_train), "statistics saw more than the training split"
```

Both sides are the same expression. Ran the cell verbatim after substituting the
**leaky** statistics for `MEAN`/`STD` — the exact situation it claims to catch:

```
statistics provenance check passes
...with MEAN = ['0.4355', '0.3777', '0.2880'] = the ALL-SPLITS statistics
n_used_for_stats = 1020 but the statistics actually saw 8189
```

It passes, unchanged, in a notebook where the statistics came from all 8,189
images. Cell 55 calls it *"weak on purpose"*; it is not weak, it is vacuous, and
it is offered under the heading *"the assertion that catches this bug"*. This is
the audit's canonical failure — a check that passes for the wrong reason —
sitting in the cell whose subject is checks.

**Severity:** misleads a student

**Origin:** generated code

**Fix:** the script's `channel_stats(x_u8)` returning `(mean, std, len(x_u8))`
from one call, so the count cannot be set by the hand that got it wrong.

---

### Claim 10 — §4.3: unnamed out-of-order hazard, cell 26 after cell 33

**Verdict:** CONFIRMED

**Evidence:** Reproduced directly — a `Sequential` moved to `mps`, then cell 26's
`x = torch.zeros(2, 3, IMG, IMG)` (built on the CPU) pushed through it:

```
RuntimeError: slow_conv2d_forward_mps: input(device='cpu') and weight(device=mps:0')  must be on the same device
```

character for character what the report quotes — including torch's own missing
opening quote in `device=mps:0')`, which is how you can tell the report ran it
rather than paraphrasing it. The notebook names no
out-of-order hazard anywhere, and re-checking a shape after training is the
obvious thing a reader does.

**Severity:** misleads a student — it blocks the literal reader with an error
that looks like a bug in the notebook.

**Origin:** notebook structure

**Fix:** name it above cell 26 and give the re-run order "23 → 26".

---

### Claim 11 — dead 200 MB allocation in the cell whose lesson is memory

**Verdict:** CONFIRMED

**Evidence:** `Xva` appears **exactly once** in all 58 cells — the assignment
`Xva = normalise(X_val).to(device)` in cell 33 — and is never read. Validation
accuracy goes through `accuracy(net, X_val, y_val)`, which takes the uint8
tensor and normalises batch by batch. Size: 1,020 × 3 × 128 × 128 × 4 =
**200,540,160 bytes = 200 MB (191 MiB)**, parked on the accelerator for the rest
of the session. (`Xtr` by contrast is read four times and is legitimate.)

**Severity:** wrong but harmless — 200 MB on a 15 GB T4 — but it is the cell
whose own box says *"normalise ONE BATCH AT A TIME"*.

**Origin:** generated code

**Fix:** delete the line.

---

### Claim 12 — the assistant cell allocates more than the notebook's own memory warning

**Verdict:** CONFIRMED, with one correction to the claim's paraphrase

**Evidence:** Ran cell 50's pattern on the real cached splits, measuring peak RSS:

```
peak RSS with three uint8 splits resident: 0.62 GB
  cat -> 402505728 bytes uint8 = 402.5 MB ; peak 1.02 GB
  float32 tensor = 1,610,022,912 bytes = 1.61 GB ; peak 4.24 GB
cell-50 pattern: 2.5 s (machine under load), PEAK RSS 4.24 GB
```

Peak **4.24 GB**, matching the report exactly. 1.61 GB / 1.209 GB = **1.33×** the
tensor cell 32 flags. Cell 50 has no ⏱ and no memory note.

*Correction:* cell 32 does **not** say a 1.2 GB tensor is how a Colab session
dies. It says *"`normalise(X_test)` as a single tensor is 6,149 × 3 × 128 × 128
float32 = 1.2 GB, and **three of those at once** is how a Colab session dies"* —
a ~3.6 GB threshold. The size comparison, the absent note and the internal
tension are all real; the quoted threshold in the claim is not what the notebook
says.

*Also:* the report's implied remedy is already present — cell 50 does
`del pixels`. What survives is `all_images`, the 402 MB uint8 copy, which is
never deleted.

**Severity:** misleads a student — the notebook teaches a memory rule and then
breaks it, unremarked, eighteen cells later.

**Origin:** notebook structure — the code is deliberately the assistant's
unedited output and must not be "fixed"; the missing note is the defect.

**Fix:** add a ⏱/memory line above cell 50 stating the 1.61 GB peak, and make
the point that the assistant's shortest code is also the most expensive.

---

### Claim 13 — §8.1: the defect is announced five times before it runs

**Verdict:** CONFIRMED

**Evidence:** All five locations are present, in this order:

| cell | text |
|---|---|
| 0 | "Cells marked **⚠ read before running** contain a defect on purpose" |
| 14 | "**Which images they are computed from is the whole of this lecture's assistant failure**" |
| 15 | "that which images these come from is the whole of this lecture's assistant failure" |
| 48 | "**⚠ Read before running.**" + "There is exactly one thing missing from that prompt, and it is a noun." |
| 49 | box labelled **⚠**, three bullets: "it is a NOUN", "there are three of them", "`torch.cat` on the first line means the statistics … were computed from a set including all 6,149 test images" |

Cell 49 sits immediately above cell 50 and gives the complete answer — the
mechanism, the line, and the count — before the reader's eye reaches the code.
The `⚠` character appears in cells 0, 48 and 49 only. One nuance: cell 0's
announcement is generic, so **four** are specific to this defect; the count of
five is right only if the generic header is included.

**Severity:** misleads a student — the reader is invited to believe they would
have caught a trap that was fully disclosed to them.

**Origin:** notebook structure

**Fix:** the script's ordering — run cell 50 unannounced, write the numbers down,
then open with the ⚠ and the contrast.

---

### Claim 14 — §8.3: "examinable" appears twice, both in section 1

**Verdict:** CONFIRMED

**Evidence:** Grepped all 58 cells for `xaminable`; exactly two hits:

```
cell 2 (markdown): "... and it is not examinable either way."
cell 3 (code):     "# Not examinable: engineering hygiene, not machine learning."
```

Both are inside section 1 (cells 1–3), the setup section. Sections 2–12 carry no
*examinable* / *not examinable* / *beyond the book* marker at all.

**Severity:** wrong but harmless

**Origin:** notebook structure

**Fix:** one marker per section heading.

---

### Claim 15 — §1.2: no cell has a stored output; the checkable figures re-derive correctly

**Verdict:** CONFIRMED for the structural finding. The re-derivations are correct
**except one arithmetic slip inside the claim itself.**

**Evidence:** All 17 code cells: `outputs: []`, `execution_count: null`.
Independently re-derived, from the cached dataset and a constructed network:

| figure | re-derived | ✓ |
|---|---|---|
| majority baseline | test counts 20…238, sum 6,149; 238/6,149 = 0.038705 → 3.87 % | ✓ |
| uniform | 1/102 = 0.009804 → 0.98 % | ✓ |
| splits | 1,020 / 1,020 / 6,149 = 8,189; train exactly 10 per species across 102 | ✓ |
| test/train ratio | 6,149/1,020 = 6.028 | ✓ |
| download | `102flowers.tgz` = 344,862,509 bytes = 345 MB | ✓ |
| 1.2 GB | 6,149 × 3 × 128 × 128 × 4 = 1,208,942,592 | ✓ |
| 200 MB | 1,020 × 3 × 128 × 128 × 4 = 200,540,160 | ✓ |
| first-layer weights | 3 × 32 × 7 × 7 = 4,704 | ✓ |
| final shape | `(2, 102)` from the shape walk | ✓ |
| **flatten width** | claim writes "**256 × 16 × 16 = 16,384**" — that product is **65,536** | ✗ |

The *value* 16,384 is right; the arithmetic offered for it is not. Four maxpools
take 128 → 64 → 32 → 16 → **8**, and the printed walk ends
`MaxPool2d (2, 256, 8, 8)` → `Flatten (2, 16384)`; the code computes
`256 * (IMG // 16) ** 2` = 256 × 8² = 16,384. The notebook never states this
arithmetic, so it is a defect in the report (and in the script's own cell-8
"Expect" line), not in `lecture-15.ipynb` — **it must not be copied into the
rebuild.**

Also: the claim says *"The two figures that do not reconcile are items 6, 7 and
8"* — three items are listed, and item 7 is a false positive.

**Severity:** wrong but harmless (the §1.2 finding itself; all 23 notebooks share
it)

**Origin:** notebook structure

**Fix:** execute and store outputs before shipping; correct the flatten
arithmetic wherever the script repeats it.

---

### Claim 16 — §2.4: the headline rests on n = 2

**Verdict:** CONFIRMED

**Evidence:** Cell 53 prints `100*abs(honest[0] - honest[1])` as *"difference
between seeds"* — one gap from one pair. `leaky` is a two-element list whose own
spread is computed and never printed. Cell 52's box states the conclusion
(*"the effect is smaller than the seed noise, and that is the finding rather
than an embarrassment"*) and cell 54's three reasons never mention that it rests
on two runs per condition. Grepped every cell: no caveat about the number of
seeds appears anywhere in the notebook.

**Severity:** wrong but harmless — the conclusion is very probably right; §2.4
asks that the weakness be stated where the headline is.

**Origin:** hand-written prose (the omission), with the single-gap print in
generated code

**Fix:** print both spreads, and say "n = 2 per condition" next to the headline.

---

### Claim 17 — §7.1: cell 4's "about 25 seconds" decode estimate is low

**Verdict:** CONFIRMED in direction; the exact figure is hardware-dependent and I
could not measure on an idle machine

**Evidence:** Re-decoded all three splits from the cached JPEGs with the
notebook's exact `Resize((128,128)) + PILToTensor` pipeline:

```
  train:  19.7 s  (1020, 3, 128, 128) torch.uint8
  val:    21.7 s  (1020, 3, 128, 128) torch.uint8
  test:  121.3 s  (6149, 3, 128, 128) torch.uint8
decoded in 162.6 s TOTAL
```

162.6 s, but at load average 172 — treat that as an upper bound and the report's
**42.7 s** on an idle machine as the better figure. I cannot separate the two
cleanly. What both settle: the decode is nowhere near 25 s on hardware faster
than Colab's, and a 2-vCPU runtime is slower again. This remains the softest
item in the list, and it is the number a reader uses to decide whether to start.

**Severity:** wrong but harmless

**Origin:** hand-written prose

**Fix:** "about 45 seconds on a fast laptop, 2–3 minutes on a Colab CPU runtime".

---

### Claim 18 — §5.1 / §5.2: no bad markdown indentation, no unclosed fence

**Verdict:** CONFIRMED (the clean assessment is accurate)

**Evidence:** Scanned all 41 markdown cells line by line tracking fence state:
zero lines indented ≥ 4 spaces outside a fence, zero fence markers indented ≥ 4,
zero unclosed fences. `tools/check_notebooks.py 15` reports no §5 violation — its
only complaint is §6.1 (claim 1).

**Severity:** n/a — no defect asserted

**Origin:** n/a

**Fix:** none needed

---

### Claim 19 — §3.1: the only fenced block in markdown is cell 20's paper form

**Verdict:** CONFIRMED (the clean assessment is accurate)

**Evidence:** Exactly one fenced block exists in any markdown cell — cell 20,
empty language tag, body beginning
`Metric:                                            ____________`. No
```` ```python ```` block appears in markdown anywhere, so there is nothing to
reconcile against a code cell.

**Severity:** n/a — no defect asserted

**Origin:** n/a

**Fix:** none needed

---

### Claim 20 — §4.2: both training cells re-instantiate the model and optimiser

**Verdict:** CONFIRMED (the clean assessment is accurate)

**Evidence:** Cell 33 contains `net = make_net().to(device)` and
`opt = torch.optim.Adam(net.parameters(), lr=LR)` inside the cell, and reseeds
with `torch.manual_seed(RANDOM_STATE)` first. Cell 53's `quick_train` does
`torch.manual_seed(seed)`, `m = make_net().to(device)` and
`o = torch.optim.Adam(m.parameters(), lr=LR)` inside the function body. Both are
idempotent: re-running restarts rather than continuing, which is the failure
lecture 19 shipped.

**Severity:** n/a — no defect asserted

**Origin:** n/a

**Fix:** none needed

---

### Claim 21 — §4.1: no name is rebound to a different kind of object

**Verdict:** CONFIRMED (the clean assessment is accurate)

**Evidence:** Scanned every assignment in all 17 code cells. Names assigned in
more than one place, with their bindings:

```
device : "cuda" / "mps" / "cpu"           (three str literals, one cell)
x      : local inside load_split (cell 6, tensor); module-level tensor (cell 26)
t0     : float (cell 6), float (cell 33)
net    : make_net() -> make_net().to(device)   — Sequential to Sequential
perm   : local in cell 33 loop; local inside quick_train (cell 53)
idx    : same, both locals
```

No name changes kind. `m` is a loop variable over modules in cell 26 and a local
model inside `quick_train`; they never collide.

**Severity:** n/a — no defect asserted

**Origin:** n/a

**Fix:** none needed

---

### Claim 22 — §2.1: the honest/leaky comparison is on matched rows

**Verdict:** CONFIRMED (the clean assessment is accurate)

**Evidence:** Cell 53:

```python
honest = [quick_train(MEAN, STD, 42 + s) for s in range(2)]
leaky  = [quick_train(MEAN_ALL, STD_ALL, 42 + s) for s in range(2)]
```

Seeds 42 and 43 appear in both arms; `epochs=12` is the default in both; both
build `make_net()` and use the same `BATCH` and `LR`; both are scored by
`accuracy(m, X_val, y_val, mean, std)` on the same 1,020 validation images. The
only difference is which statistics normalise the images. Confirmed also that
the difference between the two constant sets is real and small — measured
max |Δmean| = 0.008454, max |Δsd| = 0.003818 — so
`assert not torch.allclose(MEAN, MEAN_ALL)` would hold. The prose indeed never
says the rows are matched, which is what the script adds.

**Severity:** n/a — no defect asserted

**Origin:** n/a

**Fix:** none needed (say so in prose)

---

### Claim 23 — `torch.Generator(device=…)` / `randperm` works on MPS

**Verdict:** CONFIRMED (the clean assessment is accurate)

**Evidence:**

```
cpu True [2, 6, 1, 8, 4]
mps True [6, 8, 9, 4, 0]
```

`torch.Generator(device=d).manual_seed(42)` followed by
`torch.randperm(10, device=d, generator=g)` returns a full permutation on both
devices (torch 2.13.0, M4 Max). No Apple-Silicon blocker. Note the permutations
differ between devices, so a "change the seed" exercise is not reproducible
across CPU and MPS — worth a sentence, but not a defect.

**Severity:** n/a — no defect asserted

**Origin:** n/a

**Fix:** none needed

---

### Claim 24 — whether the deck's 80-epoch figures agree with a 30-epoch run

**Verdict:** UNVERIFIABLE

**Evidence:** Cell 0 claims *"The shape of every curve is the same; the accuracy
is a little lower."* Settling it requires two training runs (80 epochs ≈ 96 s on
MPS, ≈ 3.7 hours on this CPU). The brief excludes training. Nobody has tested
this claim.

**Severity:** n/a — untested

**Origin:** hand-written prose (untested)

**Fix:** run both once and record the two curves, or drop the claim to "the
accuracy is lower; we have not re-plotted the deck's curves at 30 epochs".

---

### Claim 25 — the Colab T4 wall clock

**Verdict:** UNVERIFIABLE

**Evidence:** No CUDA device on this machine (`torch.cuda.is_available()` is
`False`). Every GPU figure is an Apple-MPS proxy. The proxy is at least sound at
the step level — I measured 26 ms per training step and 24 ms per 128-image
forward on MPS, against the report's 28 ms and 24 ms — but a T4 is a different
device and no T4 was run.

**Severity:** n/a — untested

**Origin:** n/a

**Fix:** run the notebook once on a T4 and replace the proxy figures.

---

### Claim 26 — the actual test accuracy, and whether "a long way from 90 %" is right

**Verdict:** UNVERIFIABLE

**Evidence:** Cell 40 prints `test_acc / majority` and asserts *"a long way from
90%"*; cell 57's table has a "**yours, today** — printed above" row. Both require
executing cell 33 (30 epochs). Not run. What *is* checkable is that the sentence
is hard-coded prose inside an f-string — the notebook will print "a long way
from 90%" whatever the accuracy turns out to be, including if it were 92 %. That
is worth fixing independently of what the number is.

**Severity:** n/a — untested (but see the note above)

**Origin:** generated code

**Fix:** derive the phrase from `test_acc`, or drop it.

---

## Summary

```
confirmed: 22   false positive: 1   unverifiable: 3
of the confirmed, 8 mislead a student
   (claims 1, 2, 3, 4, 9, 10, 12, 13)
origin split, over the 16 confirmed defect claims —
   prose: 7 (2, 3, 5, 6, 8, 16, 17)
   code: 2 (9, 11)
   structure: 7 (1, 4, 10, 12, 13, 14, 15)
   (claims 18–23 assert no defect; 24–26 untested)
duplicates:
   claims 6 and 8 are counted a second time inside claim 15, which lists them
      as "the figures that do not reconcile"
   claims 2, 3 and 4 are three sites of one root cause — §7.1 CPU wall clock
      absent or wrong throughout — and should be fixed as one pass, not three
   claims 11 and 12 are both "a float32 tensor built where the notebook's own
      lesson says not to", in two different cells; distinct defects, one theme
```

**The one claim I overturned.** Claim 7 ("nine parameters in ten" is wrong):
87.25 % rounds half-up to 9 in ten, which is the rule §1.2 states. Note that the
adjacent claim 6, about the *same* annotation bullet, is a genuine rounding error
in the other direction — 4,713 to "four thousand" — so the two must not be
treated as one finding.

**Severity weighting, if the rebuild can only do part of it.** Claim 9 first: an
assert that cannot fail, offered as *"the assertion that catches this bug"*, in
the section about checks. Then claim 13 (the trap is fully disclosed before it
runs, so the lecture's central exercise catches nobody), then claims 2/3/4 as one
pass (the no-GPU reader is currently told 30 epochs takes "several minutes" when
it takes about ninety), then claim 1.

---

## Appendix — three things the report did not catch

Not part of the 26 claims; recorded because they bear on the rebuild.

1. **Cell 31: "Nothing prints until the first epoch finishes; that is not a
   hang."** Cell 33 prints under `if (ep + 1) % 10 == 0`, so nothing prints
   until **epoch 10** — about 28 minutes into a CPU run, not 165 seconds. This
   is the same reader claim 2 protects, and it is a second reason they will
   think the cell has hung. The script's cell 10 already says "epoch 10", so the
   correction exists; it is simply not listed as a defect. Demonstrable from the
   source alone.

2. **The script's cell-15 ⏱ note says "Add `del pixels` immediately after the
   two statistics come out."** Cell 50 already does `del pixels`. What is never
   released is `all_images`, the 402 MB uint8 copy from `torch.cat`. The advice
   as written changes nothing.

3. **`256 × 16 × 16 = 16,384` appears in the script's cell-8 "Expect" line as
   well as in claim 15.** The product is 65,536; the flatten width is
   256 × 8 × 8. Fixing it in the report is not enough — it is in the prompt
   script the rebuild will follow.

---

## Appendix — the checks the brief asked for, in full

Run without downloading anything (Flowers102 was already cached) and without
training.

**Parameter split — ~90 % of parameters are in one `Linear`, not the convolutions.**
Confirmed. 7 convolutions + 7 batch norms = 586,720 (12.20 %); `Linear(16384,
256)` = 16,384 × 256 + 256 = 4,194,560 (87.25 %); `Linear(256, 102)` = 26,214
(0.55 %); total 4,807,494, and the three parts sum to it exactly. The dense layer
is **7.149×** all convolutions and batch norms combined.

**Memory arithmetic.** 6,149 × 3 × 128 × 128 × 4 = **1,208,942,592 bytes =
1.209 GB**. Confirmed. Also 1,020 × 3 × 128 × 128 × 4 = 200,540,160 (200 MB) and
8,189 × 3 × 128 × 128 × 4 = 1,610,022,912 (1.61 GB), the last measured at a
4.24 GB peak RSS.

**The shape walk.** Printed in full from a constructed network:

```
Conv2d       (2, 32, 128, 128)   BatchNorm2d (2, 32, 128, 128)
Conv2d       (2, 32, 128, 128)   BatchNorm2d (2, 32, 128, 128)
MaxPool2d    (2, 32, 64, 64)
Conv2d/BN    (2, 64, 64, 64) ×2
MaxPool2d    (2, 64, 32, 32)
Conv2d/BN    (2, 128, 32, 32) ×2
MaxPool2d    (2, 128, 16, 16)
Conv2d/BN    (2, 256, 16, 16)
MaxPool2d    (2, 256, 8, 8)
Flatten      (2, 16384)
Linear       (2, 256)
Dropout      (2, 256)
Linear       (2, 102)
```

Four maxpools take 128 → 64 → 32 → 16 → **8**. `net[:3]` prints
`Conv2d(3, 32, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), bias=False)`,
`BatchNorm2d(32, …)`, `ReLU()` as the script's cell 7 expects, and `net[26]` /
`net[29]` are the two `Linear`s the notebook indexes by hand — both correct.

**`bias=True` on a Conv2d followed by BatchNorm2d is a parameter with no effect
on the function.** Confirmed by construction. Two identical convolutions, one
with `bias=False` and one with an arbitrary non-zero bias, each followed by its
own batch norm:

```
conv bias values: [-3.517, 2.242, -2.693, -2.97, -1.086, -5.73, 2.24, -2.401]
max |diff|, TRAIN mode:                       3.7e-06
max |diff|, EVAL mode (running stats converged): 1.2e-05
output scale for reference, max|y0|:          4.51
```

Differences are float32 round-off on a signal of magnitude 4.5. The cost is
32+32+64+64+128+128+256 = **704 parameters** across the seven convolutions —
total goes 4,807,494 → 4,808,198, confirmed by building both. `net[0].bias` is
`None` with `bias=False` and a `(32,)` tensor with `bias=True`, which is the only
way to see it.

**Dataset claims (cache present, nothing downloaded).** Train 1,020 images,
`bincount` min = max = **10** across all 102 species. Val 1,020, also 10 each.
Test **6,149**, from **20** to **238** per species — unbalanced. Labels 0…101.
Totals 8,189. Majority baseline 238/6,149 = 3.87 %, uniform 0.98 %, ratio 3.95.
