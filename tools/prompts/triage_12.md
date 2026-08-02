# Triage — lecture 12

Twenty-eight claims from the `Defects found in the current notebook` section of
`tools/prompts/lecture_12.md`, in order: fifteen under *Verified by execution*
(1–15), nine under *Checked and found clean* (16–24), four under *Not checked*
(25–28).

**Environment.** `python3` 3.13.5, torch 2.13.0, torchvision present, optuna
4.6.0, 16-core Apple Silicon, **no CUDA** (`torch.cuda.is_available()` →
`False`), MPS available. `uptime` during this triage: load averages
`135.13 103.89 54.65` on 16 cores — the machine is under heavy load from other
agents, so every absolute wall clock below is an **upper bound**, and I say so
where it matters.

**Noted once, per the brief.** `notebooks/lecture-12.ipynb` stores **no cell
outputs** (checked: 61 cells, 19 code cells, 0 with outputs, every
`execution_count` is `None`). So no prose figure in this file can be reconciled
against a stored output, and the §9 machine check that reads stored execution
times cannot run. This is not repeated per claim.

**Scope limit.** I did not run the Fashion-MNIST training, the scikit-learn
control, or the Optuna search. Claims whose headline number is a function of a
trained model's accuracy are marked UNVERIFIABLE, and where a claim mixes a
structural half with an accuracy-dependent half I say which half I settled.
Verification scripts live in `/private/tmp/claude-501/triage12/`.

---

### Claim 1 — the arithmetic in the markdown after the autograd cell is wrong in every digit and does not add up
**Verdict:** CONFIRMED
**Evidence:** Re-derived in float64 exactly as the cell does:

```
$ python3 -c "import torch,numpy as np; ..."
w        = 6.909297426825682
2w*y     = 41.45578456095409   -> .6f 41.455785   -> .4f 41.4558
2w*cos x = -5.7505645338736375 -> .6f -5.750565   -> .4f -5.7506
sum      = 35.70522002708046   -> .6f 35.705220   -> .4f 35.7052
x.grad   = 35.705220
41.4557 - 5.7505 = 35.7052
```

Cell 10 (markdown) prints `$$41.4557 - 5.7505 = 35.7053$$`, and cell 8 (the
prompt box) repeats the same three figures. Round-half-up to 4 dp gives
**41.4558**, **−5.7506**, **35.7052**: all three printed figures are wrong, the
first two by truncation (§1.2 forbids exactly that), the third by rounding in
the wrong direction. The equation is also internally inconsistent as printed —
`41.4557 − 5.7505` is `35.7052`, not `35.7053`. Cell 9, four lines above, prints
`dL/dx = {x.grad.item():.6f}` = **35.705220**, which the markdown then
contradicts. String search: `41.4557` and `5.7505` occur only in cells 8 and 10,
both markdown — neither path contribution is printed by any code cell, so the
reader cannot reconcile them without deriving `2w·y` and `2w·cos x` themselves,
and the sentence does not state that arithmetic.
**Severity:** misleads a student
**Origin:** hand-written prose
**Fix:** print the two path contributions from the cell and quote them at the
printed precision: `41.455785 − 5.750565 = 35.705220`.

---

### Claim 2 — "the last batch is not that short" is false, and the cell's own output says so
**Verdict:** CONFIRMED
**Evidence:** The phrase occurs twice — cell 43 (prompt box, "The last batch is
not that short") and cell 45 (markdown, same words). Cell 44 defines
`acc_two_ways(net, X, y, batch=384)` and is called on `Xt` (10,000 test rows).
Batch arithmetic, run:

```
batch=384: 27 batches, last 16; plain-mean weight 3.7037% vs true 0.1600%
           over-weight 23.15x   multiplier (1/nb - last/n) = +0.035437
```

16 of 384 is 4.2% of a full batch, and the cell prints `27 batches, the last
containing 16 images` one line above the prose that calls it "not that short".
The claimed mechanism is also exactly right: I derived the identity by hand and
it is exact — with `A` the pooled accuracy over the 26 full batches and `a` the
last batch's,

```
mean - over_set = (1/27 - 0.0016) * (a - A)   [exact; both weightings differ
                                               only on the last batch]
```

so the difference is the 23× over-weighting multiplied by how far the last
batch's accuracy sits from the rest — not by the batch being nearly full.
*Not settled here:* the specific figures 0.8125, 0.8533 and −0.1448 come from
the trained model and were not reproduced (no training run).
**Severity:** misleads a student
**Origin:** hand-written prose
**Fix:** replace "not that short" with the over-weighting: 16 rows get 1/27 of
the weight instead of 16/10,000 — 23× too much — and the effect is small only
because that batch's accuracy landed near the set's.

---

### Claim 3 — the follow-up sensitivity claim ("change either and it grows") is directionally misleading
**Verdict:** UNVERIFIABLE
**Evidence:** The weight arithmetic in the prose checks out exactly:

```
batch=3000: 4 batches, last 1000; plain-mean weight 25.0000% vs true 10.0000%
            over-weight 2.50x   multiplier (1/nb - last/n) = +0.150000
```

— "a quarter of the weight instead of a tenth" is correct. But the claim's
headline is that the measured gap moves from **−0.1448** to **+0.1950** points,
i.e. changes sign, and both numbers are functions of the trained model's
accuracy on two *different* subsets of test images (the last 16 rows at
batch 384, the last 1,000 at batch 3,000). Reproducing them requires the
training run I was told not to execute, so I have not tested the sign flip.
What I can say structurally: by the exact identity in claim 2, the gap is
`multiplier × (a_last − A_rest)`; the multiplier grows 4.23× (0.035437 →
0.150000), but `a_last` is measured on an entirely different set of rows, so
nothing in the arithmetic guarantees that the gap grows or that it keeps its
sign. The notebook's "change either and it grows" is therefore not derivable
from the mechanism it invokes, whichever way the measurement lands.
**Severity:** misleads a student *(if the reported sign flip holds)*
**Origin:** hand-written prose
**Fix:** state the multiplier, not the outcome — at batch 3,000 the weighting
error is 4.2× larger and lands on different rows, so the gap's size *and sign*
are not predictable from the number you just read.

---

### Claim 4 — the first row of the cost-of-autodiff table contradicts the lesson it is printed to support, in every fresh kernel
**Verdict:** CONFIRMED
**Evidence:** Ran cell 13's code verbatim in three separate cold processes
(`/private/tmp/claude-501/triage12/cell13.py`):

```
--- cold 1 ---
P=  51   reverse 2619.64 ms (1 pass)   forward     44.3 ms (51 passes)   ratio      0x
P=  99   reverse    0.59 ms (1 pass)   forward     25.4 ms (99 passes)   ratio     43x
P= 195   reverse    0.51 ms (1 pass)   forward     48.4 ms (195 passes)  ratio     95x
P= 387   reverse    0.65 ms (1 pass)   forward    396.1 ms (387 passes)  ratio    606x
--- cold 2 ---
P=  51   reverse 6353.59 ms   forward    402.7 ms   ratio      0x
--- cold 3 ---
P=  51   reverse 3713.81 ms   forward    322.4 ms   ratio      0x
```

Three out of three cold processes: row one shows reverse mode **4× to 15×
slower** than all 51 forward passes together, and the ratio column prints `0x`
because the format string is `{t_fwd/t_rev:6.0f}`. Cell 14, immediately below,
asserts "Reverse mode is flat in $P$; forward mode is linear in it." With one
throwaway `vjp`+`jvp` before the loop (`--warm`), rows 2–4 hold the claim
(reverse 0.38–0.82 ms, flat) and the `0x` disappears. Absolute times here are
inflated by machine load; the *sign* of the row-one comparison is not.
**Severity:** misleads a student
**Origin:** generated code
**Fix:** add one throwaway `vjp(f, theta)` + `jvp(...)` above the `for h in ...`
loop, and widen the ratio format so a sub-unit ratio cannot print as `0x`.

---

### Claim 5 — the stated reason for the log-scaled loss axis is not what the data does
**Verdict:** CONFIRMED (mechanism); the specific range `[0.31, 2.34]` not reproduced
**Evidence:** Cell 32's prompt box requires `set_yscale("log")` "because
without zero_grad the loss goes somewhere a linear axis cannot show alongside
the healthy run", and cell 33 applies it. The premise is that the broken loss
runs away. It cannot: Adam's step is bounded by the learning rate no matter how
large the accumulated gradient is —

```
Adam step with gradient 1:     |delta| = 0.001
Adam step with gradient 1000:  |delta| = 0.001
Adam step with gradient 1e+06: |delta| = 0.001      (lr = 1e-3)
```

and a surrogate run of the notebook's `train()` loop verbatim on random 784-d
data with 10 classes (not Fashion-MNIST), seeds 42/0/7, keeps the
`zero_grad=False` curve pinned at 2.30–2.33 for all ten epochs — it does not
escape upward at any seed. So the upward half of the stated justification is
refuted on mechanism and on a same-shaped experiment. I did not reproduce the
lower bound 0.31, which needs the real training run.
**Severity:** misleads a student
**Origin:** hand-written prose
**Fix:** ask for a linear axis and say why the loss *cannot* run away — Adam's
step is lr-bounded, which is precisely why this bug is silent.

---

### Claim 6 — §3.3, the header refers to a marker that appears nowhere
**Verdict:** CONFIRMED
**Evidence:** String search over all 61 cells:

```
'read before running': total 1 in 1 cells -> [(0, 'markdown', 1)]
'⚠'                  : total 7 in 7 cells -> [(0,md),(31,md),(32,md),(36,md),(37,md),(42,md),(43,md)]
```

The phrase occurs exactly once, inside the header sentence that promises it. No
cell carries it. The ⚠ symbol appears seven times: once in that sentence, then
on three section headings (cells 31, 36, 42 = §8, §9, §11) and their three
prompt labels (32, 37, 43).
**Severity:** cosmetic
**Origin:** notebook structure
**Fix:** either mark the three cells `⚠ read before running` or drop the phrase
and refer to the ⚠ sections.

---

### Claim 7 — §3.3, the header miscounts its own defects
**Verdict:** CONFIRMED
**Evidence:** The header (cell 0) reads: "Cells marked **⚠ read before running**
contain a defect on purpose — **two of them** are the defects this lecture
exists to teach, and **neither** raises an exception." Three sections carry ⚠:
§8 (`The missing zero_grad()`), §9 (`The missing model.eval()`), §11
(`Averaging the metric per batch`) — all three stage a deliberate defect, and §8
carries two distinct mechanisms (accumulation into `.grad`, and
`zero_grad()`'s `set_to_none=True` changing the *type* of the attribute — both
verified under claim 23). So the honest count is three sections and four
mechanisms against a header that says two and reinforces it with "neither".
Caveat on my own verdict: "two of them" is grammatically a subset claim and can
be read as "two of the three are the headline ones", so this is a weaker finding
than the others; the countable facts (3 marked sections, 4 mechanisms) are what
I verified. It is the same underlying defect as claim 6 — a header describing a
marking convention the notebook does not implement.
**Severity:** wrong but harmless
**Origin:** notebook structure
**Fix:** "three of them are the defects this lecture exists to teach, and none
raises an exception."

---

### Claim 8 — §3.3, the forward reference to Lecture 15 points at a lecture that does the opposite
**Verdict:** CONFIRMED
**Evidence:** Cell 39 says indexing a tensor by hand "stops working the moment
it does not — which is Lecture 15"; cell 40's box repeats it as "three lectures
from here". Checked `notebooks/lecture-15.ipynb`:

```
# Visual inspection
**Lecture 15 · Build** · Géron, Chapter 12
sections: 1 Setup / 2 The data / 3 Look at it / 4 Normalisation — from the
training split only / 5 Two numbers to compare against / 6 Commit / 7 The
network / 8 Train it / 9 The test set. Once. / 10 Look at what it learned /
11 An assistant writes the normalisation / 12 Where we are
```

Its single mention of `DataLoader` (cell 5, a prompt bullet) reads: "using a
DataLoader with a decode transform, which is correct for data that does not fit
in memory and **is pure overhead here**." Grepped all 24 notebooks:
`num_workers` → `{}` (none), `ImageFolder` → `{}` (none), `does not fit in
memory` → only lecture-15, in that same bullet. The course never delivers the
topic this cross-reference promises.
**Severity:** misleads a student
**Origin:** hand-written prose
**Fix:** drop the forward reference, or point at the general problem rather than
a lecture that argues the opposite.

---

### Claim 9 — §8.1, the trap is announced five times before the reader reaches it
**Verdict:** CONFIRMED
**Evidence:** The defective cell is 33. Before it, `zero_grad` as *the defect*
is flagged in:

```
cell  0 (header)   '⚠ read before running ... contain a defect on purpose'
cell 31 (heading)  '## 8 · ⚠ The missing `zero_grad()`'
cell 31 (para)     '`backward()` **accumulates** into `p.grad`; it does not overwrite...'
cell 32 (label)    '> **Prompt · ⏱ 40 s — ⚠ the missing zero_grad()**'
cell 32 (bullets)  'Left open: that `backward()` ACCUMULATES...' +
                   'The usual student version: this exact omission.'
```

Five announcements, all before the cell runs. The same for `model.eval()` before
cell 38: cell 36 heading, cell 36 paragraph, cell 37 label, cell 37 Left-open
bullet, cell 37 student bullet. GUIDELINES §8.1's preferred shape (run it
unannounced, write the number down, *then* open with the ⚠) costs nothing here
because both defects are parameter flips — `train(zero_grad=False)` and
`train(dropout=0.2)` plus a missing `eval()`.
**Severity:** misleads a student
**Origin:** notebook structure
**Fix:** move §8's and §9's headings, ⚠ and annotation bullets to *after* the
cell, keeping the prompt box's specification before it.

---

### Claim 10 — §6.1, the annotation budget is exceeded
**Verdict:** CONFIRMED
**Evidence:** Counted programmatically over `lecture-12.ipynb`:

```
code cells: 19
prompt boxes: 19
full three-bullet annotations: 19   (all of "**Watch this prompt.**" +
                                     "Left open" + "usual student version" +
                                     "How you would catch it")
```

§6.1's budget is five to eight, never more than ten. 19 is 2.4× the top of the
target range and 1.9× the hard cap (the claim's "a factor of two and a half"
matches the target range, not the cap). This also contradicts the plan in
`tools/prompts/lecture_12.md` itself, which states "Seven of the nineteen boxes
carry the full three-bullet annotation (cells 1, 4, 10, 11, 12, 14, 15)" — the
shipped notebook carries nineteen.
**Severity:** wrong but harmless
**Origin:** notebook structure
**Fix:** reduce to the seven the prompt script already nominates; the other
twelve keep the specification only.

---

### Claim 11 — §8.3, almost nothing is marked examinable
**Verdict:** CONFIRMED (the claim's own count is off by one)
**Evidence:** The string occurs **four** times, not three:

```
cell  2 (markdown, prompt box §1) '...It is not examinable and it is not optional.'
cell  3 (code comment, §1)        '# Not examinable, but from here to the end...'
cell 49 (markdown, §13)           '**Not examinable** — it is not in the book'
cell 50 (markdown, prompt box §13)'that this is not examinable — it is not in the book'
```

Two are about Optuna and two about device selection (one of those inside a code
comment, which is probably why the claim counted three). The substance is exact:
the notebook has **16 sections** and only §1 and §13 carry any label, so
**fourteen of sixteen** carry none — including all three ⚠ sections (§8, §9,
§11) and the whole autodiff thread (§3), which is the examinable mathematics of
the lecture. §8.3 requires every section to carry one of *examinable* / *not
examinable — engineering* / *beyond the book*.
**Severity:** wrong but harmless
**Origin:** notebook structure
**Fix:** add one label per section heading; mark §3 and §7–§11 examinable.

---

### Claim 12 — §7.1, the ⏱ markers do not say which machine they describe, and one cell has none
**Verdict:** CONFIRMED (structural half); the quoted wall clocks not re-measured
**Evidence:** All eight ⏱ occurrences, i.e. four distinct markers, each stated
twice (section heading + prompt label):

```
cell 18/19  '⏱ **about 30 seconds.**'                  §4  scikit-learn control
cell 31/32  '⏱ **about 40 seconds** for the two runs.' §8  zero_grad, two runs
cell 36/37  '⏱ **about 25 seconds.**'                  §9  dropout / eval
cell 49/52  '⏱ **about 90 seconds** for 8 short trials.' §13 Optuna
```

None names a device, a runtime or a CPU class. And §7 — the cell that actually
trains the model for 10 epochs — carries no marker at all:

```
cell 27 [markdown] has clock: False | ## 7 · The training loop
cell 28 [markdown] has clock: False | > **Prompt · the training loop, ...
cell 29 [code]     has clock: False | def make_net(dropout=0.0):
```

which is internally inconsistent with §8's "about 40 seconds for the two runs"
of that same `train()` function, i.e. ~20 s each — over §7.1's 20-second
threshold. The claim's measured counter-figures (7.7 s, 2.2 s, ~1.5 s, 1.3 s)
require training runs I did not execute, so I confirm the marker defects, not
those numbers. No CUDA device here either, so the Colab figures remain untested
— which is the claim's point.
**Severity:** misleads a student
**Origin:** hand-written prose
**Fix:** name the machine in every ⏱ ("~30 s on a free Colab CPU runtime") and
add one to §7.

---

### Claim 13 — §4.1, `net` means two different networks, and the toy one comes first
**Verdict:** CONFIRMED
**Evidence:** Module-level bindings, by regex over code cells:

```
net    : cell 13  net = nn.Sequential(nn.Linear(20, h), nn.ReLU(), nn.Linear(h, 3))   # in `for h in (2,4,8,16)`
         cell 29  net, hist, pt_seconds = train()                                     # 784-300-100-10
`net` appears in code cells: [13, 29, 44, 56, 59]
f, theta : cell 13 (toy) ; cell 16 (784-300-100-10)
t_rev    : cell 13 (toy) ; cell 16
xb, yb   : cell 35 torch.randn(8, 4), torch.randn(8, 1)
           cell 41 next(iter(train_dl))        # (128, 784) images + labels
```

Cell 13 leaves the module-level name `net` bound to the h=16 toy 20→16→3
network; cell 29, sixteen cells later, rebinds it to the trained model, which
cells 44, 56 and 59 then score, save and report. Same class, different object,
different meaning, unremarked — and §4.1's own remedy is "loop variables in
throwaway tests get throwaway names". Restart-and-run-all ordering is sound
(nothing reads the stale binding), so this breaks nothing; a reader inspecting
`net` between cells 13 and 29 gets the wrong network.
**Severity:** misleads a student
**Origin:** generated code
**Fix:** rename the loop's objects `toy`, `f_toy`, `theta_toy`, `t_toy` and the
probe's tensors `xp, yp`.

---

### Claim 14 — §2.4, the `model.eval()` cost is quoted ahead of the noise
**Verdict:** CONFIRMED (print order); the 0.79 / 0.52 figures not re-measured
**Evidence:** Cell 38's print order is settled by the source:

```python
print(f"model.eval()   {acc_eval:.4f}")
print(f"model.train()  {np.mean(readings):.4f}  (min ..., max ...)")
print(f"\ncost {100 * (acc_eval - np.mean(readings)):.2f} accuracy points")
print(f"and a spread of {100 * (max(readings) - min(readings)):.2f} points ...")
print("\nThe spread is the tell. ...")
```

The cost line is printed before the spread line, and the "the spread is the
tell" sentence comes last — so the number a reader writes down is the cost. The
underlying phenomenon is real and I verified it directly on an untrained
784-300-100-10 net with dropout 0.2 on 10,000 random rows: `eval` gives one
reading; ten `no_grad` readings in `train()` mode gave **10 distinct values out
of 10**. The specific cost 0.79 and spread 0.52 come from the trained model and
were not reproduced, so the "1.5× the noise" ratio is untested here.
**Severity:** misleads a student
**Origin:** generated code
**Fix:** print the spread first, then the cost, and state the ratio between them
on the same line.

---

### Claim 15 — a line that teaches nothing and cannot be explained
**Verdict:** CONFIRMED
**Evidence:** Cell 38, line 3, verbatim:

```python
net_d, hist_d, _ = train(dropout=0.2)

_, _, Xte_, yte_ = None, None, Xt, yt
```

`_` is bound to `None` twice (silently discarding the elapsed time returned on
line 1), and `Xte_`/`yte_` are plain aliases of `Xt`/`yt` — the same objects, no
copy, no device move, no slicing. They are used exactly four times, all inside
this cell (twice in the `acc_eval` expression, twice in the `readings`
comprehension) and never again in the notebook. Substituting `Xt, yt` directly
is a pure textual substitution of identical bindings.
**Severity:** wrong but harmless
**Origin:** generated code
**Fix:** delete the line and use `Xt, yt`.

---

### Claim 16 — §5.1/§5.2, markdown rendering is clean and §3.1 cannot be violated
**Verdict:** CONFIRMED (the claim of cleanliness is correct)
**Evidence:**

```
markdown lines indented >=4 outside fence: 0  []
markdown cells containing ``` : []
```

No prose line is indented four or more spaces outside a fence, and there are no
fenced blocks in any markdown cell — so §3.1 (quoted code existing in no cell)
is vacuous here, as claimed.
**Severity:** n/a — the claim asserts the absence of a defect
**Origin:** hand-written prose
**Fix:** none needed

---

### Claim 17 — §4.2, training cells are idempotent
**Verdict:** CONFIRMED (the claim of cleanliness is correct)
**Evidence:** `tools/notebooks/lecture_12.py:432-437` (notebook cell 29):

```python
def train(epochs=EPOCHS, lr=LR, batch=BATCH, dropout=0.0,
          zero_grad=True, seed=RANDOM_STATE, track=True):
    torch.manual_seed(seed)
    net  = make_net(dropout).to(device)
    opt  = torch.optim.Adam(net.parameters(), lr=lr)
```

The network, the optimiser, the loss and the shuffling generator are all
constructed inside the function from a seed argument, so re-running any training
cell retrains from scratch — the §4.2 failure mode (silently continuing
training) cannot occur. I confirmed this by reading the construction, not by
running `train()` twice on Fashion-MNIST; my surrogate run of the same loop on
random data gave byte-identical loss curves across repeated calls at a fixed
seed.
**Severity:** n/a — the claim asserts the absence of a defect
**Origin:** generated code
**Fix:** none needed

---

### Claim 18 — the split matches Lecture 11's
**Verdict:** CONFIRMED (the claim of cleanliness is correct)
**Evidence:** `lecture-11.ipynb` cell 15 against `lecture-12.ipynb` cell 6 —
the four load-bearing lines are identical:

```python
rng = np.random.default_rng(RANDOM_STATE)            # both, RANDOM_STATE = 42 in both files
order = rng.permutation(len(...))                    # both, over 60,000
val_idx, fit_idx = order[:5_000], order[5_000:]      # both
SUB = 12_000; X_fit, y_fit = X_fit_full[:SUB], ...   # both
```

Both draw the generator fresh immediately before the permutation, so nothing
else consumes it first. Checked the generator directly:
`np.random.default_rng(42).permutation(60000)` is reproducible across calls
(first five indices `[3493 57546 8815 19332 15566]`), so the two notebooks
select the same 5,000 validation and the same first 12,000 fit rows.
**Severity:** n/a — the claim asserts the absence of a defect
**Origin:** generated code
**Fix:** none needed

---

### Claim 19 — the assert in section 7 holds with room to spare
**Verdict:** UNVERIFIABLE
**Evidence:** `assert abs(hist["val_acc"][-1] - sk_val) < 0.05` compares the
PyTorch validation accuracy against `MLPClassifier`'s. Both sides require the
training runs I was told not to execute, so the claimed gap |0.8636 − 0.8728| =
0.0092 is untested here. Nobody has re-checked it in this triage.
**Severity:** n/a — untested
**Origin:** generated code
**Fix:** none proposed — needs a run

---

### Claim 20 — section 12's parameter-count assert holds
**Verdict:** CONFIRMED (the claim of cleanliness is correct)
**Evidence:** Built both models and counted, no training involved:

```
Sequential params: 266,610   Sorter params: 266,610   equal: True
hand arithmetic 784*300+300+300*100+100+100*10+10 = 266610
```

`assert sum(p.numel() for p in m.parameters()) == n_params` passes, and the
on-paper arithmetic in the prompt script reproduces it.
**Severity:** n/a — the claim asserts the absence of a defect
**Origin:** generated code
**Fix:** none needed

---

### Claim 21 — section 14's exact-equality assert holds, serialised size 1,044 KB
**Verdict:** CONFIRMED (the claim of cleanliness is correct)
**Evidence:** Ran the save/reload cell against a freshly built (untrained)
`make_net()` and scored both models on 10,000 fixed random rows — the mechanism
under test is determinism of `state_dict` round-tripping, which does not need a
trained model:

```
checkpoints/sorter.pt 1,044 KB  (raw bytes 1069193)
original 0.1048   reloaded 0.1048   exactly equal: True
torch.load signature weights_only default: None
```

`a1 == a2` holds exactly. The 1,044 KB figure is reproduced exactly. The prompt
script's claim about `weights_only` is also right as stated: on torch 2.13.0 the
signature still reads `weights_only=None`, which torch resolves to `True`.
**Severity:** n/a — the claim asserts the absence of a defect
**Origin:** generated code
**Fix:** none needed

---

### Claim 22 — section 10's "the loader dropped rows" assert holds
**Verdict:** CONFIRMED (the claim of cleanliness is correct)
**Evidence:** Ran the loader on a (12,000 × 784) zero tensor — the counts depend
only on shapes, not on the data:

```
94 batches of 128; first (128, 784); 12000 = 93 x 128 + 96
rows yielded: 12000
last batch len: 96
```

`assert sum(len(b) for b, _ in train_dl) == len(Xf)` passes; `drop_last=False`
is the default and nothing is discarded.
**Severity:** n/a — the claim asserts the absence of a defect
**Origin:** generated code
**Fix:** none needed

---

### Claim 23 — the probe cell's ordering is correct and subtle
**Verdict:** CONFIRMED (the claim of cleanliness is correct)
**Evidence:** Ran cell 35's body verbatim:

```
after 1 backward: |grad| = 4.2099
after 2 backward: |grad| = 8.4199
after 3 backward: |grad| = 12.6298
after zero_grad(): None
after zero_grad(set_to_none=False) on a None grad: None      <- leaves it None
after backward + zero_grad(set_to_none=False): |grad| = 0.0000
no middle backward -> AttributeError: 'NoneType' object has no attribute 'norm'
```

Exactly 1×, 2×, 3× the same norm — `backward()` accumulates into `.grad`, it
does not overwrite. `zero_grad()` sets `.grad` to `None`, not to zeros.
`zero_grad(set_to_none=False)` on a `None` gradient leaves it `None`, so the
`backward()` between the two calls is load-bearing: deleting it makes the final
print raise the very `AttributeError` the annotation is about. The cell prints
`0.0000` as intended.
**Severity:** n/a — the claim asserts the absence of a defect
**Origin:** generated code
**Fix:** none needed

---

### Claim 24 — optuna imports here, so the fallback branch was not exercised
**Verdict:** CONFIRMED (the claim of cleanliness is correct)
**Evidence:** `python3 -c "import optuna; print(optuna.__version__)"` → `4.6.0`.
The `except ImportError` branch of cell 51 is therefore dead on this machine and
its printed text has not been seen executing.
**Severity:** n/a — the claim asserts the absence of a defect
**Origin:** generated code
**Fix:** none needed

---

### Claim 25 — nothing GPU-dependent was checked; no CUDA device available
**Verdict:** CONFIRMED (the environmental claim is accurate)
**Evidence:** `torch.cuda.is_available()` → `False`; `torch.backends.mps
.is_available()` → `True`. So on this machine the notebook's cell 3 selects
`mps`, no Colab/T4 figure in the file is testable, and every timing quoted in
the Phase A report is a CPU/MPS measurement on one laptop. One consequence worth
flagging for the rebuild: the notebook's ⏱ markers are all Colab-shaped numbers
that no available machine can confirm or refute (claim 12).
**Severity:** n/a — environment note
**Origin:** n/a
**Fix:** none needed

---

### Claim 26 — whether the section-7 assert survives on a T4 was not checked
**Verdict:** UNVERIFIABLE
**Evidence:** No CUDA device (see claim 25), and the assert requires both
training runs regardless. The report's own reasoning — that the scikit-learn
half is CPU-bound either way so the gap should be stable — is reasoning, not a
measurement, and it stays that way after this triage.
**Severity:** n/a — untested
**Origin:** n/a
**Fix:** none proposed — needs a CUDA runtime

---

### Claim 27 — the Optuna search cell's real duration was not measured
**Verdict:** UNVERIFIABLE
**Evidence:** Cell 53 runs 8 trials × 4 epochs of `train()` on the Fashion-MNIST
fit set. I did not execute it, so the ~4–5 s projection stands as a projection,
and the notebook's "⏱ about 90 seconds" (cells 49/52) remains untested against
any machine.
**Severity:** n/a — untested
**Origin:** n/a
**Fix:** none proposed — needs a run

---

### Claim 28 — machine load may have inflated the timings
**Verdict:** CONFIRMED (the caveat is accurate and applies to this triage too)
**Evidence:** `uptime` during this triage: `load averages: 135.13 103.89 54.65`
on `hw.ncpu = 16`. The effect is visible in my own cell-13 measurements, whose
absolute forward-mode times vary by an order of magnitude between cold runs
(44 ms / 402 ms / 322 ms at P=51) while the qualitative finding — reverse mode
slower than 51 forward passes on the first row, ratio printing `0x` — reproduces
3/3. So every absolute second in the Phase A report and in this file should be
read as an upper bound; the orderings and signs are what survive.
**Severity:** n/a — environment note
**Origin:** n/a
**Fix:** none needed

---

## Summary

```
confirmed: 24   false positive: 0   unverifiable: 4
of the confirmed, 9 mislead a student
origin split — prose: 7   code: 12   structure: 5   (n/a: 4 environment notes)
duplicates: claims 6 and 7 are the same underlying defect — a header that
            describes an ⚠ marking convention the notebook never applies;
            claims 2 and 3 are two halves of one paragraph (cells 43 and 45)
            about the same per-batch-averaging gap.
```

**Notes on these verdicts.**

- Zero false positives is an unusual result and I want it read with its
  qualifications, not as a rubber stamp. Nine of the 24 confirmations (16–24)
  are *claims that nothing is broken*, which are cheap to confirm. Of the
  fifteen defect claims, **eleven** were settled outright by execution or exact
  string/count search, **three** (2, 12, 14) were confirmed on their structural
  half while their quoted wall clocks or accuracies went untested, and **one**
  (3) I could not settle at all.
- Two claims contain small errors of their own, which I confirmed anyway
  because their substance holds: claim 11 says "examinable" appears three times
  when it appears four (the fourth is inside a code comment), and claim 10's
  "factor of two and a half" is 19/8 against §6.1's target range — against the
  hard cap of ten it is 1.9×.
- Claim 7 is the weakest confirmation in this file. "Two of them" is
  grammatically a subset claim and survives a charitable reading; what I
  verified is the count (three ⚠ sections, four mechanisms) and the fact that
  its referent — cells marked "⚠ read before running" — does not exist.
- The origin split over the fifteen *defect* claims is prose 6 / code 4 /
  structure 5. That is a weaker concentration in prose than the audit's
  hypothesis predicts, but the two most damaging defects here split one each:
  claim 1 (prose arithmetic contradicting the output four lines above it) and
  claim 4 (a missing warm-up in generated code that makes the lecture's central
  quantitative table refute its own caption on every cold kernel).
- Calibration items from the brief (lectures 3 and 6) do not appear in
  lecture 12, so this file offers no independent check against them.
