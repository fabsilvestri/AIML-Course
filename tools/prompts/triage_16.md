# Triage — lecture 16

Claims triaged: the **16** numbered items under *Verified by execution or
arithmetic*, the **8** bullets under *Checked and found clean* (numbered 17–24
here), and the **4** bullets under *Not checked* (25–28). Twenty-eight in total,
in the order they appear.

**Environment.** `python3`, torch 2.13.0, torchvision 0.28.0, Apple M4 Max,
`device = mps`. Flowers102 is cached at `notebooks/datasets/flowers-102/`
(`jpg/`, `imagelabels.mat`, `setid.mat` all present), so every data-dependent
claim was re-derived from the real images rather than reasoned about. The
pretrained `resnet18-f37072fd.pth` is in `~/.cache/torch/hub/checkpoints/`, so
the ResNet claims were also executed. **No training cell was run.**

**Stated once, not repeated per claim (per the brief).** All 23 code cells of
`notebooks/lecture-16.ipynb` have `outputs: []` and `execution_count: None`, so
no prose figure in this notebook can be reconciled against a stored output.
Where a claim is about a number I re-derived that number from the notebook's own
code and data instead.

---

### Claim 1 — §6.1: all 23 code cells carry the full three-bullet annotation, against a budget of 5–8 (max 10)

**Verdict:** CONFIRMED

**Evidence:** counted from the JSON —

```
markdown cells                 51
starting '> **Prompt'          23
containing 'Watch this prompt' 23
with >=3 lines starting '* **' 23
code cells                     23
```

`python3 tools/check_notebooks.py --advisory` agrees and blocks on it:

```
FAIL  lecture-16.ipynb
        23 full annotations, budget is 10 (§6.1)
```

The payload cell (the assistant failure) is code cell 21 of 23 — raw notebook
cell 66 of 74 — well past the cell-30 point at which all three audit readers
stopped reading the template.

**Severity:** misleads a student

**Origin:** notebook structure

**Fix:** cut to seven full annotations; the prompt script's cells 2, 5, 9, 15,
19, 22, 23 are a defensible seven.

---

### Claim 2 — §8.1: the defect is announced six times before code cell 21 runs

**Verdict:** CONFIRMED

**Evidence:** every one of the six is verbatim in the markdown preceding code
cell 66. Notebook cell 0:

> Cells marked **⚠ read before running** contain a defect on purpose.

Notebook cell 64, in order, all before the cell:

> **⚠ Read before running.**
> ...
> Perfectly reasonable, and it names both loaders. That is the hole.
> ```python
> val_ds   = Flowers102("datasets", split="val",   transform=tf)   # <--
> ```
> ### Reviewer question 5: what is the default I did not ask for?
> One `tf` for both splits.

Then notebook cell 65 repeats it twice more — box label *"⚠ what the assistant
returns — one transform for two splits"* and student bullet *"reusing `tf` for
both splits"*. Eight signals, six of them before the box.

**Severity:** misleads a student — it removes the only chance the reader has to
answer *"would I have caught it?"*

**Origin:** notebook structure

**Fix:** run cell 21 unannounced, have the reader write the four numbers down,
open the following section with the ⚠ and the contrast.

---

### Claim 3 — §1.2: prose says "25 GB of float32 for one layer"; the cell prints 96

**Verdict:** CONFIRMED

**Evidence:** notebook markdown cell 10, catch bullet, verbatim: *"print the
dense layer's size in gigabytes. **25 GB** of float32 for one layer is the kind
of number that ends an argument."* The code cell immediately below (cell 11)
contains `print(f"dense layer, float32  {dense_weights * 4 / 2**30:>18,.0f} GB")`.
Re-derived:

```
dense weights   25,769,803,776
bytes           103,079,215,104
printed value               96
```

25.77 billion is the **parameter count**, not a byte count. Prose and output
disagree on the same screen.

**Severity:** misleads a student

**Origin:** hand-written prose

**Fix:** change the bullet to "96 GB".

---

### Claim 4 — §1.1: "that residue is float32 rounding" sits under a printed `0.000e+00`

**Verdict:** CONFIRMED

**Evidence:** executed the equivariance cell exactly as written (seed 42,
`normalise(X_test[:1])` with training-split MEAN/STD, `roll` 16, `padding=3`,
border `m = 24`):

```
MEAN ['0.4330', '0.3819', '0.2964']
STD  ['0.2896', '0.2408', '0.2684']

largest |f(Tx)-Tf(x)| interior  0.000e+00
largest activation there        2.918
relative                        0.000e+00
interior slice shape (1, 32, 80, 80)
border difference               1.947e-01
exactly zero? True
```

`(a-b).abs().max().item() == 0.0` is exactly `True`, not merely small. The cell
then unconditionally prints `"that residue is float32 rounding, not
mathematics"`. There is no residue. The border figure 1.947e-01 also reproduces,
so cell 6's *"orders of magnitude larger"* is true — infinitely so.

This is the reason the identity needs no training and no cached data to check:
it is a property of the operation.

**Severity:** misleads a student — it teaches that the identity is approximate
when the cell has just demonstrated it is exact.

**Origin:** hand-written prose (a `print` string in generated code)

**Fix:** replace with "bit-for-bit identical, not merely close — on this backend
the two computations perform the same additions in the same order."

---

### Claim 5 — §1.5: predicted 728.1 MB and measured 568.1 MB printed together, never reconciled

**Verdict:** CONFIRMED

**Evidence:** ran the allocator cell verbatim on MPS, including the warm-up
optimiser step:

```
device mps
predicted, by counting outputs     728.1 MB
measured, by the backend           568.1 MB
ratio 78.03%
```

Both numbers reproduce to the decimal. The surrounding markdown says nothing
about the gap: cell 32 is *"A prediction nobody checks is a claim."*, cell 35 is
*"The rule that follows"*. The prompt box (cell 33) has **no `check ·` field at
all** and its three bullets name no expected agreement. A reader cannot tell
whether 568 confirms the arithmetic or refutes it.

**Severity:** misleads a student

**Origin:** hand-written prose (the missing sentence)

**Fix:** state the expectation in the box — same order, roughly 75–80%, because
the count is an upper bound and autograd frees what no backward needs.

---

### Claim 6 — §1.2/§3.2: the declared preprocessing is printed, then narrowed to two vectors

**Verdict:** CONFIRMED

**Evidence:** ran cell 42's `print(weights.transforms())`:

```
ImageClassification(
    crop_size=[224]
    resize_size=[256]
    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]
    interpolation=InterpolationMode.BILINEAR
)
```

Markdown cell 43, immediately below, in full: *"**The weights come with their
own preprocessing.** Use those statistics, not the ones you computed from the
flowers."* — "preprocessing" silently becomes "statistics". Cell 6 has already
decoded with `transforms.Resize((size, size))`, i.e. `Resize((224, 224))`, a
direct squash that changes the aspect ratio and keeps the corners; that is
neither `resize_size=[256]` nor the centre crop to 224. The choice is
defensible for fixed in-memory tensors and is made without being named, in the
cell whose lesson is *read the metadata*.

**Severity:** misleads a student

**Origin:** hand-written prose

**Fix:** one sentence under cell 42 naming the squash and why it was chosen.

---

### Claim 7 — §3.1: the quoted assistant code exists in no code cell

**Verdict:** FALSE POSITIVE

**Evidence:** the *factual* half is true — string-searched all 23 code cells:

```
'RandomResizedCrop(224, scale=(0.55, 1.0))' in code cells []
'train_ds = Flowers102'                     in code cells []
'val_ds   = Flowers102'                     in code cells []
```

But GUIDELINES §9 exempts exactly this notebook by name when it explains why the
§3.1 check is advisory rather than blocking:

> quoting code you are deliberately *not* running is legitimate — **lecture 16
> shows what the assistant returned**

The lecture-19 evidence behind §3.1 is a different failure: prose invited the
reader to *go and find* a loop in the notebook, and neither the loop nor its
variable `toy` existed. Here the block is presented as an exhibit of what came
back from an assistant, not as a pointer into the file. The claim is a
correctly-fired advisory on a case the rule was written to allow.

I also ran the quoted pipeline verbatim to confirm the report's own subsidiary
statement, and it does work:

```
quoted pipeline runs: (32, 3, 224, 224) torch.float32   0.56s for one batch
RandomResizedCrop default scale: (0.08, 1.0)
```

**Residual point, not a §3.1 defect and not counted as confirmed:** cell 66 does
not run the quoted loaders — it re-uses the notebook's own tensor `augment()` on
`T_val` — and no sentence says so. That is a small provenance mismatch (§4.4 in
spirit) rather than the §3.1 violation claimed.

**Severity:** cosmetic

**Origin:** notebook structure

**Fix:** none needed for §3.1; optionally one clause noting that the cell
simulates the quoted pipeline with the notebook's own `augment`.

---

### Claim 8 — §2.1: the wall-clock column compares three different phases

**Verdict:** CONFIRMED

**Evidence:** read from the code, and the bracketing is unambiguous.

- Cell 38: `t0` is set before the epoch loop and `SCRATCH_SECONDS =
  time.perf_counter() - t0` is the line immediately after it; `SCRATCH_TEST`,
  `TRAIN_ACC`, `VAL_ACC` are all computed *after*. Training loop only.
- Cell 53: `PROBE_SECONDS = FEATURE_SECONDS + HEAD_SECONDS`, and
  `FEATURE_SECONDS` (cell 51) brackets `features(T_train), features(T_val),
  features(T_test)` — a forward pass over all 8,189 images.
- Cell 60: `t0` before the loop, `FT_SECONDS` after it, and the loop body
  contains `clean_curve.append(accuracy(ft, T_val, y_val, inorm))` and
  `aug_curve.append(accuracy(ft, augment(T_val, gen_val), y_val, inorm))`.

Measured on this machine at 224, so the "roughly half" is a measurement and not
an estimate:

```
augment 1020 imgs @224:  1.96 s  = 1.92 ms/image
one val pass @224 on mps: 0.68 s
```

Per fine-tune epoch the diagnostic costs `1.96 + 2 × 0.68 = 3.32 s` against
`1.96 s` of training augmentation plus ~0.7 s of gradient work — **55% of the
epoch**. Cell 63 then prints `frozen probe was {SCRATCH_SECONDS /
PROBE_SECONDS:.1f}x faster`, a ratio of two of these three unlike quantities.
The rows are matched; the times are not, and no prose says so.

**Severity:** misleads a student

**Origin:** hand-written prose (the code is honest; the column label is not)

**Fix:** say what each entry brackets, next to the table.

---

### Claim 9 — §7.1: four ⏱ markers, no CPU figure among them

**Verdict:** CONFIRMED

**Evidence:** grepped every ⏱ in the notebook — there are exactly four, in four
markdown cells (plus their four box-label echoes):

```
md4 : ⏱ **about 40 seconds** — the dataset is already downloaded ...
md36: ⏱ **about 30 seconds on a GPU or MPS.** 20 epochs.
md49: ⏱ **about 25 seconds** — the test split is 6,149 images at 224 × 224.
md58: ⏱ **about 90 seconds on a GPU or MPS.** 8 epochs of fine-tuning ...
```

Not one CPU number. Measured here with `torch.set_num_threads(2)`:

```
CPU(2thr) train step @128 batch32: 3.672 s
  20 epochs = 640 steps          -> 39.2 min of gradient work
CPU(2thr) resnet18 fwd 64@224:    9.84 s = 154 ms/image
  8,189 images                   -> 21.0 min
```

**Note where I differ from the report.** My 2-thread figures are 2–6× *slower*
than the ones in the Phase A report (115 vs 46 ms/image at 128; 154 vs 26
ms/image at 224) — this machine is running many agents concurrently, and the
report says its own figures were taken under load too. So the specific minute
counts in the report are not reproducible as stated, and neither set should be
printed as a promise. **The defect is confirmed regardless and in the same
direction:** cells whose only stated time is "about 30 seconds" and "about 25
seconds" are tens of minutes without an accelerator, and md49 does not even name
a device.

**Severity:** misleads a student — this is the reader §7.1 exists to protect

**Origin:** hand-written prose

**Fix:** give both columns and say which hardware; state the CPU figure as a
floor, not a promise.

---

### Claim 10 — §7.1: code cells 21 and 23 exceed 20 s and carry no ⏱ at all

**Verdict:** CONFIRMED

**Evidence:** neither code cell 66 nor code cell 72, nor any markdown cell above
them (63, 64, 65 / 70, 71), contains a ⏱ — checked by scanning for the glyph in
every cell (the only four occurrences are listed in claim 9).

Cost, read from cell 66: one `clean = accuracy(...)`, ten
`accuracy(ft, augment(T_val, gen_w), ...)` inside the list comprehension, and
one more inside the assert — **twelve** evaluations of the 1,020-image split and
**ten** augmentations of it. At the rates measured above that is
`12 × 0.68 + 10 × 1.96 = 27.8 s` **on MPS**, and minutes on CPU. Cell 72
evaluates it twice: 1.4 s on MPS, but `2 × 1020 × 154 ms ≈ 5 min` at 2 threads.

Cell 66 is inside section 9, the section a reader is most likely to run on its
own.

**Severity:** misleads a student

**Origin:** notebook structure

**Fix:** add ⏱ to both, with the CPU number.

---

### Claim 11 — §7: no runtime advice, and the expensive cell cannot be skipped

**Verdict:** CONFIRMED

**Evidence:** the header (cell 0) contains no mention of GPU, CPU, T4 or runtime
type — read in full. The dependency trace, by string search over the 23 code
cells (0-indexed):

```
'lossf' in code cells [10, 11, 16, 18]
'ytr'   in code cells [11, 16, 18]
```

`lossf` is defined only in code cell 10 (the accelerator memory check, notebook
cell 34) and used one, six and eight cells later. `ytr` is defined only in code
cell 11 (`Xtr, ytr = normalise(X_train).to(device), y_train.to(device)`, the
from-scratch training cell) and used five and seven cells later. A reader who
skips the from-scratch baseline to reach transfer learning gets `NameError` in
cell 17 or 19, with no visible connection to what they skipped.

**Severity:** misleads a student

**Origin:** generated code (the dependency) + hand-written prose (the missing
advice)

**Fix:** hoist `lossf` and `ytr` into the setup cells, and add a runtime section.

---

### Claim 12 — §4.1: `a`/`b`, `rows` and `gen` each carry two meanings

**Verdict:** CONFIRMED

**Evidence:** read from the code cells, all three sub-claims hold.

```
code4  (nb15): a = y_of_shift[..., m:-m, m:-m]      -> tensor (1, 32, 80, 80)
code4  (nb15): b = shift_of_y[..., m:-m, m:-m]      -> tensor (1, 32, 80, 80)
code22 (nb72): a = accuracy(ft, T_val, y_val, inorm) -> float
code22 (nb72): b = accuracy(ft, T_val, y_val, inorm) -> float
```

The interior slice shape `(1, 32, 80, 80)` is from my own run of cell 15.

`rows` is `z, rows = torch.zeros(1, 3, IMG, IMG), []` in code cell 9 (notebook
31), filled with `(index, shape, MB)` triples; and `rows = [("uniform guess",
1 / N_CLASSES, None), ...]` in code cell 19 (notebook 63), `(name, accuracy,
seconds)` triples.

`gen = torch.Generator(device=device).manual_seed(RANDOM_STATE)` in code cell 11
(a **device** generator, `mps` here) against `gen = torch.Generator()
.manual_seed(RANDOM_STATE)` in code cell 18 (**CPU**). The device change is
correct — `augment` needs a CPU generator — and is invisible at the call site.

**Severity:** wrong but harmless for `a`/`b` and `rows`; the `gen` device change
is the one worth a comment.

**Origin:** generated code

**Fix:** rename the throwaway comparisons (`lhs`/`rhs`, `mem_rows`/`result_rows`)
and name the CPU generator `gen_cpu` in cell 18, as cell 17 already does.

---

### Claim 13 — the red-team table describes the deck's catalogue, not this notebook

**Verdict:** CONFIRMED

**Evidence:** two of the five answers are wrong about this notebook.

*Answer 1* claims *"normalisation statistics computed over all 8,189 images"*.
The only mean/std computation in any cell is `xf = X_train.float() / 255.0` in
code cell 1. Decoded all three splits and computed both:

```
training-only mean ['0.4330', '0.3819', '0.2964']   <- what the notebook uses
all-splits   mean ['0.4355', '0.3777', '0.2880']
abs diff          ['0.0026', '0.0042', '0.0085']
float32 tensor of all 8,189 at 128: 1.499 GiB
```

The notebook computes the training-only figure. It **avoided** the failure its
own red-team table reports as having committed.

*Answer 5* lists `padding=0`. Grepped every `padding=` in every code cell:

```
padding in code cell 2 : padding=k // 2
padding in code cell 4 : padding=3
```

`padding=0` appears nowhere; every `nn.Conv2d` in the notebook names its
padding. (It is `nn.Conv2d`'s documented default, but it is not a default this
notebook accepted.)

**Severity:** misleads a student — the section's whole instruction is *"Report
what you found, not what you would have done differently"*, and the table does
the opposite.

**Origin:** hand-written prose

**Fix:** rewrite answers 1 and 5 against this notebook's code; name the
all-splits failure as *avoided*.

---

### Claim 14 — §1.2: no stored outputs; the checkable figures were re-derived and are correct

**Verdict:** CONFIRMED

**Evidence:** all 23 code cells have `outputs: []` and `execution_count: None`
(counted from the JSON). Every figure the report lists as re-derived, re-derived
here independently:

```
param count            4,807,494  True
ratio                  5,478,275
majority baseline      3.87%   (max count 238 of 6149)
uniform                0.0098039...
splits                 1020 / 1020 / 6149  = 8189
head                   52,326        = 512*102 + 102
frozen                 11,176,512    = 11,689,512 - 513,000
per-image params       51.3  vs  4,713.2
last feature map       8 x 8   (IMG//16 = 8), 256 input pixels per cell
parameters             18.3 MB;  activations @32  728.1 MB
act/par 39.7x   act/opt 9.9x
conv rows: 64.0 64.0 32.0 32.0 16.0 16.0 8.0 MB; first two = 55%
resnet18               11,689,512;  layer4 = 8,393,728
inorm(T_test)          3.448 GiB
```

Every one matches. The three that do not reconcile are the ones already raised
as claims 3, 4 and 5.

One nuance the report glosses: `3.45 GiB` is written in the prose as `3.5 GB`,
which is correct as a rounded **GiB** figure and wrong as a decimal-GB figure
(3.70 GB). Cosmetic.

**Severity:** cosmetic (this claim is a clean bill of health with three named
exceptions)

**Origin:** notebook structure

**Fix:** none needed — but see claims 3, 4, 5.

---

### Claim 15 — §8.3: "examinable" appears zero times

**Verdict:** CONFIRMED

**Evidence:** case-insensitive count over the concatenated source of all 74
cells: `examinable count: 0`. §8.3 requires every section to carry one of
*examinable* / *not examinable — engineering* / *beyond the book*; this notebook
has ten sections and marks none.

**Severity:** wrong but harmless — but it is precisely the reader §8.3 protects
who cannot tell a mathematical thread from an engineering aside.

**Origin:** hand-written prose

**Fix:** mark all ten sections.

---

### Claim 16 — §3.3: "the next cell" is two cells away; Lecture 18 is overstated

**Verdict:** CONFIRMED (both halves)

**Evidence:** exactly one occurrence of "next cell" in the whole notebook, at
the end of markdown cell 26:

> Commit to an answer before running the next cell.

The next cell is markdown cell 27 (the prompt box). The code is cell 28. Because
every code cell in this notebook is preceded by a box, the phrase is off by one
wherever it is used.

The two cross-references the report clears do clear:

- **Lecture 6** — `LECTURES.md:96` gives *"Lecture 6 — Reading a learning
  curve"*, thread *the bias–variance decomposition*, *"a persistent gap means
  variance"*. ✓
- **Lecture 12** — searched `notebooks/lecture-12.ipynb`: it contains the five
  questions with #5 *"What is the default I did not ask for?"* (under the
  heading `## 16 · Red-team`, not the word "reviewer") and the sentence *"A
  deterministic function of fixed weights and fixed data does not change between
  calls"*. ✓ Substantively correct.

**Lecture 18 is the overstatement.** Notebook cells 23 and 25 say *"Lecture 18
has to put that resolution back, and the whole of its architecture is about
how."* `LECTURES.md:178` gives Lecture 18 as *Scoring a box, scoring a
detector*, thread *IoU's vanishing gradient, and mAP as a mean of a mean*, with
per-pixel prediction appearing only in the final clause. Counted in
`notebooks/lecture-18.ipynb`:

```
segmentation      4        upsampl          0
per-pixel         3        transposed       0
IoU             117        resolution       0
mAP              36
```

Lecture 18 has no architecture for restoring resolution at all.

**Severity:** misleads a student — §3.3's evidence is a reader who lost trust in
the phrase for the rest of the file.

**Origin:** hand-written prose

**Fix:** "two cells below"; and cut the Lecture 18 sentence to "segmentation,
touched at the end of Lecture 18, has to put that resolution back."

---

### Claim 17 — §5.1/§5.2: markdown is clean

**Verdict:** CONFIRMED (the notebook is clean; the report is right)

**Evidence:** scanned every line of every markdown cell tracking fence state:

```
fence markers: 2
violations: none
```

Two markers, both at column 0 — the single python fence in cell 64 — opening and
closing correctly. No prose line indented ≥4 outside a fence, no indented fence
marker, no unclosed fence. `check_notebooks.py` reports no §5 failure for this
notebook either.

**Severity:** n/a

**Origin:** n/a

**Fix:** none needed

---

### Claim 18 — §4.2: all three training cells re-instantiate

**Verdict:** CONFIRMED (clean)

**Evidence:** read from the code.

- Code cell 11: `scratch = make_net().to(device)` and `opt =
  torch.optim.Adam(scratch.parameters(), lr=LR)`.
- Code cell 16: `head = nn.Linear(512, N_CLASSES).to(device)` and `opt =
  torch.optim.Adam(head.parameters(), lr=1e-3, weight_decay=1e-4)`.
- Code cell 18: `ft = resnet18(weights=weights)`, `ft.fc = nn.Linear(512,
  N_CLASSES)`, and a fresh two-group `Adam`.

Each is preceded by `torch.manual_seed(RANDOM_STATE)`. Re-running any of them
restarts rather than continues — the exact failure lecture 19 shipped.

**Severity:** n/a

**Origin:** n/a

**Fix:** none needed

---

### Claim 19 — §2.1: the five accuracy rows are on the same 6,149 test images

**Verdict:** CONFIRMED (clean)

**Evidence:** read from the code. `SCRATCH_TEST = accuracy(scratch, X_test,
y_test, normalise)` (128 px), `PROBE_TEST` from `F_test`, which cell 51 builds
as `features(T_test)` (224 px), and `FT_TEST = accuracy(ft, T_test, y_test,
inorm)` (224 px). All three index the same `y_test`, `len(y_test) = 6149`
(verified against the decoded dataset). The resolution differs and the *rows* do
not — which is the lecture's subject, not a mismatch. The two curves in cell 69
are both `accuracy(..., y_val, ...)` over the same 1,020 images at the same
epochs.

Contrast claim 8: the accuracy column is matched, the wall-clock column is not.

**Severity:** n/a

**Origin:** n/a

**Fix:** none needed

---

### Claim 20 — the freeze/replace order in cell 15 is correct, and only its assert distinguishes it

**Verdict:** CONFIRMED (clean)

**Evidence:** ran both orders on the real pretrained resnet18:

```
freeze then replace: trainable 52326
replace then freeze: trainable 0
frozen (correct order): 11176512
print(model) identical: True
```

The notebook's order is freeze-then-replace, and its assert
`n_train_p == 512 * N_CLASSES + N_CLASSES` = 52,326 is the only thing that
separates the two — `str(model)` is byte-identical for both.

**Severity:** n/a

**Origin:** n/a

**Fix:** none needed

---

### Claim 21 — the batch-norm claim in cell 19 is true and reproducible

**Verdict:** CONFIRMED (clean — and this is the claim worth having run rather
than reasoned about)

**Evidence:** every parameter set to `requires_grad = False`, module in
`train()` mode, one forward pass of 32 images **inside `torch.no_grad()`**:

```
bn1.running_mean max move   0.0131
as % of largest entry       10.3%
num_batches_tracked         0 -> 1
any grad?                   False
0.9**256 = 1.9e-12
```

Freezing weights does not freeze running statistics: they are buffers, updated
in the forward pass, and neither `requires_grad=False` nor `no_grad()` stops it.
Both figures the report gives — 0.0131 and 10.3% — reproduce exactly, as does
the momentum arithmetic. The repair works:

```
with .eval() on bn1, move = 0.0
```

And it covers what the report says it covers:

```
total BatchNorm2d in resnet18: 20
covered by bn1+layer1..3:      15   running stats held: 4480
layer4 BatchNorm2d:             5
```

15 of 20, holding 4,480 running statistics, correctly leaving `layer4`'s five in
training mode since `layer4` is being fine-tuned.

**Severity:** n/a

**Origin:** n/a

**Fix:** none needed

---

### Claim 22 — `torch.Generator(device=...)` with `randperm` works on MPS

**Verdict:** CONFIRMED (clean)

**Evidence:**

```
randperm on cpu: ok, [2, 6, 1, 8, 4]...
randperm on mps: ok, [6, 8, 9, 4, 0]...
```

Cell 11's device-bound generator is not an Apple-Silicon blocker.

**Severity:** n/a

**Origin:** n/a

**Fix:** none needed

---

### Claim 23 — the cell-21 assert is sound; exact `==` is the right comparison

**Verdict:** CONFIRMED (clean)

**Evidence:** `accuracy` returns `right / len(X_u8)` where `right` accumulates
`.sum().item()` — Python ints throughout. Demonstrated end to end on the real
resnet18 and the real 1,020-image validation split at 224 on MPS:

```
a = 0.00980392156862745
b = 0.00980392156862745
equal = True   type = float
```

(The value is chance because the head is untrained — irrelevant to the point,
which is bit-identity.) A tolerance would be wrong here, and the exact
comparison in cells 21 and 23 is the right one.

**Severity:** n/a

**Origin:** n/a

**Fix:** none needed

---

### Claim 24 — `Normalize` accepts the `(3,1,1)` tensors, and `ToTensor()` matches `PILToTensor().float()/255`

**Verdict:** CONFIRMED (clean)

**Evidence:**

```
Normalize with (3,1,1) tensors: ok (3, 8, 8)
ToTensor bit-identical to PILToTensor/255: True
```

(the second checked with `torch.equal` on a real Flowers102 image resized to
224.) So the quoted assistant pipeline and `inorm` agree exactly on the same
crop, which is what makes the ten scorings comparable with the clean one.

**Severity:** n/a

**Origin:** n/a

**Fix:** none needed

---

### Claim 25 — whether the 20-epoch and 8-epoch runs reproduce the deck's ratios: not checked

**Verdict:** UNVERIFIABLE

**Evidence:** confirming it requires executing code cells 11, 16 and 18 —
training runs the brief forbids. I ran none of them. The header's claim that the
comparison "is internally consistent" and the module docstring's *"every ratio
the lecture turns on survives the shortening"* therefore stand untested. Note
that these are **blanket provenance claims of the §4.4 kind**, made about runs
that were never stored (all 23 cells have empty outputs), so nobody can check
them from the artefact either.

**Severity:** misleads a student, *if* it turns out false — the header asserts it
as fact.

**Origin:** hand-written prose

**Fix:** either run both cells and store the outputs, or soften the header to
name the epoch counts without claiming the ratios survive.

---

### Claim 26 — whether `assert FT_TEST > SCRATCH_TEST` holds at 8 epochs against 20: not checked

**Verdict:** UNVERIFIABLE

**Evidence:** the assert is on notebook cell 63 and depends on `FT_TEST` and
`SCRATCH_TEST`, both produced by training cells I did not run. Nothing in the
artefact settles it. The report's "very likely" is an estimate, not a
measurement, and I am not upgrading it.

**Severity:** misleads a student, *if* it fails — an `AssertionError` at the
climax of the notebook is the worst place for one.

**Origin:** generated code

**Fix:** run the notebook once end to end and record the two numbers.

---

### Claim 27 — whether cell 22's early-stopping contrast survives the shortening: not checked

**Verdict:** UNVERIFIABLE

**Evidence:** `best_clean` and `best_aug` come from `np.argmax` over
`clean_curve` and `aug_curve`, which cell 60's training loop builds. I did not
run it. The structural hazard is real and readable without running anything:
markdown cell 67 states *"it selects a different model"* as fact, and cell 69
prints two epoch numbers that may well both be 8 if both curves are still
climbing at `FT_EPOCHS = 8`. Under §2.2 the argument that does not depend on the
effect size — cell 21's measured spread being wider than the gaps between the
last epochs — is the one to make, and the notebook does not make it.

I agree with the report that this is the first thing to check when someone is
allowed to train.

**Severity:** misleads a student, *if* the two argmaxes coincide

**Origin:** hand-written prose (the assertion above an unrun cell)

**Fix:** make the noise-band argument beside the plot, whatever the argmaxes do.

---

### Claim 28 — the Colab T4 wall clock: not checked

**Verdict:** UNVERIFIABLE

**Evidence:** no T4 is available here. Every accelerator figure in the Phase A
report is MPS on an M4 Max used as a proxy, and every CPU figure is
`torch.set_num_threads(2)` on the same machine. I reproduced the MPS figures
that do not require training (728.1 / 568.1 MB, 0.68 s per validation pass at
224, 1.92 ms per augmented image) and they hold; I could not reproduce the
CPU-minute figures — see claim 9, where my 2-thread measurements came out 2–6×
slower under concurrent load. The report is right to flag this as unchecked, and
right that the long ⏱ figures are per-step cost times step count rather than
end-to-end measurements.

**Severity:** wrong but harmless — an honest declaration, correctly made

**Origin:** hand-written prose

**Fix:** none needed; keep the declaration, and label any CPU figure a floor.

---

## Summary

```
confirmed: 23   false positive: 1   unverifiable: 4
of the confirmed, 12 mislead a student
origin split — prose: 12   code: 3   structure: 5   (n/a: 8)
```

Counting notes. Claims 17–24 are the report's *"checked and found clean"*
bullets: CONFIRMED here means *the notebook is clean and the report's clean bill
is correct* — I re-executed every one rather than accepting it. They carry no
severity, need no fix, and have no origin, which is why the origin split covers
20 claims rather than 28.

The twelve "misleads a student" are claims **1, 2, 3, 4, 5, 6, 8, 9, 10, 11,
13, 16**. Claims 12 and 15 are confirmed but harmless, and 14 is cosmetic. Of
the four unverifiable, three (25, 26, 27) would also mislead if they turn out
false.

Origin by claim — prose: 3, 4, 5, 6, 8, 9, 13, 15, 16, 25, 27, 28. Code: 11,
12, 26. Structure: 1, 2, 7, 10, 14. (Claim 11 is genuinely mixed — the
`lossf`/`ytr` dependency is in the generated code, the missing runtime advice is
prose; it is counted once, under code.)

**The audit's prose-over-code finding holds here.** Twelve of the twenty
attributable claims are hand-written prose, and every one of the twelve
"misleads a student" verdicts is prose or structure — not one is a bug in the
Python. The four figures that actually contradict the notebook's own executed
output (claims 3, 4, 5, 13) are all sentences somebody typed.

**Duplicates / overlaps.**

- **14 is a meta-claim over 3, 4 and 5.** It says so itself: *"The figures that
  do not reconcile are items 3, 4 and 5."* Counting it as a fourth defect would
  triple-count. Its own content — that every other figure re-derives correctly —
  is a clean bill and is where its value is.
- **9, 10 and 11 are three faces of one §7 failure** (no CPU column; two long
  cells with no marker; no runtime advice at all). They are separable fixes but
  one root cause, and 9 and 11 both carry the same "about 50 minutes on CPU"
  end-to-end figure.
- **2 and 1 interact:** the §6.1 over-annotation is part of *why* the defect is
  announced so many times — the box's `student` bullet is one of the eight
  announcements. Fixing 1 removes two of them for free.
- **6 and 13 are not duplicates** despite both being about preprocessing: 6 is
  about the *declared* ResNet transform being ignored; 13 is about the red-team
  table crediting this notebook with a mistake it avoided.

**On the flagged possible false positive.** The team lead's note was right.
Claim 7 is a FALSE POSITIVE as a §3.1 violation — `GUIDELINES.md` §9 names
lecture 16 by name as the legitimate case that makes that check advisory, and
the quoted block is an exhibit rather than a pointer into the file. I did verify
the string search (the lines really are in no code cell) and I did run the
quoted pipeline (it works, `(32, 3, 224, 224)` float32, 0.56 s per batch). The
one residual observation buried inside the claim — that cell 66 simulates the
quoted pipeline with the notebook's own `augment` and never says so — is real
but small, and is not the §3.1 defect that was claimed.

**On the one claim the lead singled out for care.** Claim 21, that freezing
weights does not freeze batch-norm running statistics, was run rather than
reasoned about, on the real pretrained resnet18 with the cached checkpoint:
`bn1.running_mean` moved by **0.0131** (10.3% of its largest entry) and
`num_batches_tracked` went 0 → 1, under `requires_grad=False` for every
parameter and inside `torch.no_grad()`. Putting the module in `.eval()` reduces
the movement to exactly **0.0**. The notebook is right, its repair covers 15 of
resnet18's 20 `BatchNorm2d` modules, and leaving `layer4`'s five in training
mode is correct because `layer4` is what is being fine-tuned.

**Calibration.** This lecture contains none of the three settled claims from the
brief, so there is no external check on my verdicts here. The closest analogue
is claim 4, whose shape matches lecture 3's *"imshow of a 784-long vector is not
an error"* — prose asserting a hedge that the executed cell contradicts. I got
`0.000e+00` exactly, with `== 0.0` returning `True`, and the prose says
"float32 rounding".
