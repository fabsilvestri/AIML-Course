# Triage — Lecture 14 defect report

Against `notebooks/lecture-14.ipynb` (69 cells, 23 code, **no stored outputs** —
per the brief this is noted once and not repeated per claim; every prose figure
in this notebook is reconciled against a re-derivation, not against the file).

**How I verified.** I rebuilt cells 3, 5, 8, 20 and 23 verbatim into a module
(`scratchpad/l14/env14.py`) reading the cached CIFAR-10 at
`notebooks/datasets/cifar-10-batches-py`, and ran the notebook's own
`grad_profile`, `delta_profile`, `act_sd`, `geo`, `rho` and `theory` on it, in
float64 on CPU, seed 42. **No training cell was executed.** Where a figure
depends on training I used the repository's own cached run
(`/private/tmp/claude-501/aiml-data/fits-v2.pkl`, keys `app07_l14`,
`app07_l14_extra`), which is `mps`, and said so.

The report contains **16 numbered claims** (the task brief said 17 — if the
count comes from splitting claim 12's two bullets or claim 7's second paragraph,
both sub-parts are given their own verdict below).

---

### Claim 1 — §8's cell 46 profiles `||dL/dW||` but labels and compares it as ρ (`||dL/dz||`), printing a number 56% from its own stated theory

**Verdict:** CONFIRMED

**Evidence:** cell 46 run verbatim (`grad_profile` = the weight-gradient norm):

```
Glorot + ReLU  rho 1.1046  (theory 0.7071)   end to end 6.626e+00
He     + ReLU  rho 1.0907  (theory 1.0000)   end to end 5.206e+00
```

1.1046 / 0.7071 = 1.562, i.e. 56.2% above its own printed theory. The quantity
the theory is about is the delta, and running the notebook's own
`delta_profile` + `act_sd` on the same two configurations gives

```
                    predicted  measured rho    error  fwd scale      d1/d20
Glorot,  ReLU          0.7071        0.7045    0.4%     0.6826   1.288e-03
He,      ReLU          1.0000        0.9970    0.3%     0.9653   9.444e-01
```

which is byte-for-byte the deck's own slide — `slides/lecture-14.html:1080-1081`
(*"The damage, measured"*) reads `0.7071 / 0.7045 / 0.6826 / 1.3e-03` and
`1.0000 / 0.9970 / 0.9653 / 9.4e-01`. So the notebook does not reproduce its own
slide, and it fails in exactly the way `delta_profile`'s own docstring and the
cell-27 box call *"the single easiest way to misread this lecture"*. The
`end to end` column compounds it: 6.626 for the wrong initialisation against
5.206 for the right one, so as printed the wrong one is the one further from 1
in the direction that reads as "more gradient".

**Severity:** misleads a student
**Origin:** generated code
**Fix:** `gx = delta_profile(act="relu", init="glorot")` / `gh = delta_profile(...)`,
and take the ratio in the same direction cell 23 does (`d[0:DEPTH-1]/d[1:DEPTH]`,
no `1/rx`).

---

### Claim 2 — the quoted slide text is wrong in both figures, in three places

**Verdict:** CONFIRMED

**Evidence:** `slides/lecture-14.html:1261` reads

```
  <p class="lead">The number we report is
  41.3%, not 34.0%.</p>
```

The notebook says 43.9 / 33.4 in three cells:

```
cell 37 [markdown]: ... would report 33.4% where the argument requires 43.9% ...
cell 66 [markdown]: ... That would report 33.4% where the argument requires 43.9% ...
cell 67 [code]:     # number we report is 43.9%, not 33.4%" — the notebook ...
```

Repo grep: `43.9` occurs elsewhere only in `slides/lecture-18.html:865,877`
(a COCO mAP, unrelated). The notebook's own figures are not invented —
`app07_l14["ladder"]` gives seed-42 `accs[0]` of 0.4385 and 0.3343 — but the
deck's 41.3 / 34.0 are the five-seed means (`test_acc` 0.41262 and 0.34004) of
the same two rungs. One quantity, two numbers, unreconciled (§1.5), and only
cell 67 falsely attributes its pair to the slide as a quotation.

**Severity:** misleads a student
**Origin:** hand-written prose
**Fix:** quote the slide's actual 41.3 / 34.0, or drop the quotation and say
"the best rung, not the last" without a number.

---

### Claim 3 — the ladder crowns its winner on one seed and the winner is inside the noise

**Verdict:** CONFIRMED

**Evidence:** `app07_l14["ladder"]`, five seeds a rung, `accs[0]` is seed 42:

| rung | 5-seed mean | sd (pts) | seed 42 |
|---|---|---|---|
| + ReLU and He | 0.41262 | 0.351 | 0.4114 |
| + batch normalisation | 0.36522 | 0.636 | 0.3683 |
| + gradient clipping | 0.40776 | 0.759 | 0.3964 |
| + a 1-cycle schedule | 0.41262 | 2.788 | **0.4385** |
| + dropout 0.1 | 0.34004 | 1.329 | 0.3343 |

The two top rungs tie to five decimal places (0.41262 both) and the crowned one
has eight times the spread. Cell 38 runs one seed, `max()` picks it, `BEST_KW`
carries it into cell 67 as the headline. `tools/figures_app07.py:624-631`:
*"Five seeds a rung, not one… a single seed cannot support a claim that size"*.
Grepped the notebook: no cell mentions run-to-run spread.

**Severity:** misleads a student
**Origin:** notebook structure
**Fix:** run the ladder over ≥3 seeds and print mean ± sd, or state in cell 39's
markdown that the top two rungs are not separated by this experiment.

---

### Claim 4 — "Three of those repairs do nothing" — five do, and none is worse

**Verdict:** CONFIRMED

**Evidence:** `app07_l14["alone"]` (`test_acc`), each repair applied alone to
the Lecture 13 network:

```
Glorot initialisation 0.1000   ReLU (with He) 0.4114   Batch normalisation 0.3793
Layer normalisation   0.1000   Gradient clipping 0.1000
A 1-cycle schedule    0.1000   Dropout 0.1       0.1000
```

Five are exactly 0.1000 (Glorot and layer norm as well as the three cell 35
names); none is below 0.1000, so cell 32's *"two of them will turn out to make
things worse on their own"* has no referent at all. §7.3: the reader who
genuinely looks for the two cannot tell whether the failure is theirs.

**Severity:** misleads a student
**Origin:** hand-written prose
**Fix:** cell 32 → "none of them makes things worse; several do nothing";
cell 35 → "five of those repairs do nothing on their own", and name all five.

---

### Claim 5 — "on the repaired network the two are within a point of each other" — they are 4.8 points apart

**Verdict:** CONFIRMED

**Evidence:** `app07_l14["norms"]`, all on `act="relu", init="he"`:

```
none  test_acc 0.4114   n_params 500,210
batch test_acc 0.3683   n_params 504,210
layer test_acc 0.4164   n_params 504,210
```

batch − layer = 4.81 points. The pair within a point is *none* and *layer*
(0.50 apart). Cell 54's markdown asserts the opposite pairing.

**Severity:** misleads a student
**Origin:** hand-written prose
**Fix:** "on the repaired network layer normalisation is indistinguishable from
no normalisation, and batch normalisation costs 4.8 points."

---

### Claim 6 — `rho` is rebound from function to float, and re-running cell 20 raises `TypeError`

**Verdict:** FALSE POSITIVE

**Evidence:** the rebinding is real (AST walk: `rho` bound in cells 20 and 23;
cell 23 line 69 is `    rho = geo(d[0:DEPTH-1] / d[1:DEPTH])` at module level).
But the stated failure does not occur, because cell 20's **first statement is
the `def`**, which rebinds `rho` back to the function before `theory` is built:

```
after cell 20, rho is <function rho at 0x10310f100>
after cell 23, rho is 0.7045 <class 'float'>
--- now re-run cell 20 ---
default, logistic   rho = 0.1443   over 19 layers: 1.067e-16
Glorot,  logistic   rho = 0.2500   over 19 layers: 3.638e-12
Glorot,  ReLU       rho = 0.7071   over 19 layers: 1.381e-03
He,      ReLU       rho = 1.0000   over 19 layers: 1.000e+00
RE-RUN SUCCEEDED; rho is now <function rho at 0x10310f240>
```

(cell 20's real source exec'd twice with `rho` set to a float in between).
No `TypeError` is reachable by re-running any cell of this notebook; it needs a
*new* cell that calls `rho(...)` after cell 23. The §4.1 one-name-one-meaning
violation stands as a naming hazard, but the demonstration offered is wrong and
"confirmed by AST walk over module-level bindings" does not establish it — an
AST walk finds the rebinding, not the exception.

**Severity:** cosmetic
**Origin:** generated code
**Fix:** none needed for correctness; rename cell 23's local to `rho_meas` if
the §4.1 rule is to be enforced literally.

---

### Claim 7 — `sd` is rebound out from under the standardising closure

**Verdict:** CONFIRMED (latent, as the claim itself states)

**Evidence:** cell 5 binds `sd` shape (3072,) and `std = lambda a: (flat(a) - mu) / sd`,
which resolves `sd` from globals at call time; cell 23 line 68 rebinds
`sd = act_sd(**v)`, shape (20,). Verified both shapes live:
`cell 5 sd shape: (3072,)`, `cell 23 sd shape: (20,)`. Minimal single-namespace
reproduction of the two bindings:

```
before rebind: (8, 3072)
after rebind -> ValueError: operands could not be broadcast together with shapes (8,3072) (20,)
```

Grepped: no cell calls the `std` lambda after cell 23 (the only later matches
are `h.std()` method calls inside cell 23 itself), so nothing breaks in a clean
run — the hazard is for a reader who adds or re-runs a scaling line.

*Second paragraph of the claim:* AST walk confirms `g` bound in cells 25, 28
(loop variable over `profiles.items()`, an `ndarray`) and 43 (`torch.Generator`),
and `fwd` bound in cells 14 and 23. One inaccuracy in the report: cell 14 sets
`fwd = 1 / nin`, not `1/n_out`.

**Severity:** wrong but harmless
**Origin:** generated code
**Fix:** rename cell 23's `sd` to `sds` (and `fwd` to `fwd_meas`).

---

### Claim 8 — every ⏱ figure is wrong

**Verdict:** CONFIRMED (see scope note)

**Evidence:** the §4 marker, the only one naming a device, I measured directly.
Cell 23 in full — four schemes × (`grad_profile` + `delta_profile` + `act_sd`),
float64, CPU:

```
[cell 23 wall clock: 1.37 s]
```

against cell 21's *"⏱ about 30 seconds — four networks, eight batches each, on
the CPU in float64"*. Roughly 20× too long on this machine (the report's own
0.6 s is the same conclusion on a faster box).

For the four training markers I did not execute training. From the repository's
own cached `seconds` (`mps`, 20 epochs, the notebook's settings):

| cell | notebook says | cached `mps` total |
|---|---|---|
| §6 each repair alone (34) | "about 5 minutes" | 85.5 s + baseline ≈ **1.6 min** |
| §7 the ladder (38) | "about 5 minutes" | 110.3 s = **1.8 min** |
| §9 normalisation (50) | "about 2 minutes" | 70.8 s = **1.2 min** |
| §11 six optimisers (62) | "about 4 minutes" | 92.3 s = **1.5 min** |

So on `mps` all four markers overstate by 1.7–3.2×, and none matches. **Scope
note:** the report's CPU column for these four rows is its own measurement and I
did not reproduce it; §7.1 requires a CPU number, and the notebook gives one
only for §4, where it is wrong by ~20×.

**Severity:** misleads a student
**Origin:** hand-written prose
**Fix:** re-time on both devices and state both, e.g. "⏱ 2 s CPU" for §4.

---

### Claim 9 — five cells over 20 s carry no ⏱ marker

**Verdict:** CONFIRMED for the structural half; the 20-s threshold holds
unconditionally for two of the five and is device-dependent for three

**Evidence:** every `⏱` in the notebook, grepped:

```
cell 5 [code], cell 21/22 (§4), cell 32/33 (§6), cell 36/37 (§7),
cell 48/49 (§9), cell 60/61 (§11)
```

Cells 43, 46, 53, 64 and 67 all contain 20-epoch (or 3-epoch) training runs and
have no marker in the markdown above them. From cached `mps` per-run seconds:
cell 64 is three runs of the clipped/batch-normed config ≈ 47 s and cell 67 is
7.5 + 17.1 ≈ 25 s — over 20 s on the *fastest* device the notebook supports, so
those two are unconditional. Cells 43 (≈9 s `mps`), 46 (0.12 s of profiling plus
one ≈9 s run — I measured the profiling part directly) and 53 (3 epochs, ≈5 s
`mps`) exceed 20 s only on CPU, which is the report's own measurement and not
independently reproduced here. The cell-5 marker being a code comment rather
than markdown is confirmed by the grep above.

**Severity:** misleads a student
**Origin:** notebook structure
**Fix:** add markdown ⏱ markers above cells 43, 46, 53, 64, 67; move cell 5's
into the markdown above it.

---

### Claim 10 — the defect is announced six times before it can be walked into

**Verdict:** CONFIRMED

**Evidence:** all six, read in order, all strictly before cell 43:

1. cell 0: `**⚠ read before running** contain a defect on purpose.`
2. cell 41: `## 8 · ⚠ An assistant repairs the network`
3. cell 41: `…and it looks like the problem is solved.`
4. cell 42: `> **Prompt · ⚠ what the assistant returns**`
5. cell 42 Left open: `…Xavier and Glorot are the same person and the same formula… derived for a roughly LINEAR activation…`
6. cell 42 student: `With ReLU, Glorot gives ρ = √(1·½) = 0.707, not 1 — over nineteen layers that is three orders of magnitude rather than fifteen.`

(6) is the punchline of §8 in full, given before the cell that is supposed to
produce it. GUIDELINES §8.1 is about lecture 19 flagging its defect four times;
this is six.

**Severity:** misleads a student
**Origin:** notebook structure
**Fix:** move the §8.1 preferred shape in — run cell 43 unannounced, have the
reader write the number down, open the second half of §8 with the ⚠ and the
contrast; strip (v) and (vi) from cell 42.

---

### Claim 11 — the header names a marker string that appears in no cell

**Verdict:** CONFIRMED (with one sub-point imprecise)

**Evidence:** grep for the exact string across all 69 cells returns one hit,
cell 0 itself:

```
cell 0 [ma]: **⚠ read before running** contain a defect on purpose.
```

Every `⚠` in the notebook:

```
cell 0  [ma]: **⚠ read before running** contain a defect on purpose.
cell 41 [ma]: ## 8 · ⚠ An assistant repairs the network
cell 42 [ma]: > **Prompt · ⚠ what the assistant returns**
```

so no cell is marked with the string the header tells the reader to look for.
The sub-point *"The plural 'Cells' is also wrong — there is one"* is imprecise:
two cells do carry a ⚠ (41 and 42), though they are one section and one defect.

**Severity:** cosmetic
**Origin:** hand-written prose
**Fix:** header → "One section is marked ⚠", or put the literal string
`⚠ read before running` on cell 42.

---

### Claim 12 — two cross-references point at the wrong cell

**Verdict:** CONFIRMED (both bullets)

**Evidence:** cell 19's box is the prompt for cell 20 and closes with
*"Write these four numbers down before running the next cell."* The next cell
**is** cell 20, and it is the cell that produces the four numbers — they do not
exist to be written down until it has run. The cell that must not be run first
is 23 (the measurement), four cells on.

Cell 24's box closes with *"The next cell is about a flat profile that is flat
for the wrong reason."* Cell 25 is `plt.figure(...)` — the plot. The flat-profile
argument is cell 26's markdown and cells 27–28.

**Severity:** wrong but harmless
**Origin:** hand-written prose
**Fix:** "before running the measurement in section 4" and "section 4 ends with
a flat profile that is flat for the wrong reason".

---

### Claim 13 — a markdown line indented four spaces outside a fence

**Verdict:** CONFIRMED

**Evidence:** fence-aware scan of all 69 cells, exactly one hit, and no indented
fence markers anywhere:

```
=== indented-4 prose lines outside fences (all 69 cells) ===
  cell 26 line 13: '    || dL/dW_l ||  ~  || delta_l || . || a_(l-1) ||'
=== indented fence markers ===
(none)
```

**Severity:** cosmetic
**Origin:** hand-written prose
**Fix:** wrap it in a ``` fence.

---

### Claim 14 — the ladder clips below the median gradient norm and the notebook never says so

**Verdict:** CONFIRMED, and understated

**Evidence:** cell 57's markdown warns *"A clip value below the median silently
turns your optimiser into sign descent."* Cell 59 measures it; the cached
distribution (`app07_l14["norm_dist_deep"]`, He, same builder and seed):

```
3 epochs                            median 3.192  min 1.755  max 16.073  frac<1.0 0.000
first 2 epochs (= cell 59's call)   median 3.466  min 2.383  max 16.073  frac<1.0 0.000
```

Not one step in the run falls below 1.0, so `clip=1.0` — used by four of cell
38's seven rungs — rescales **every** step, not merely the lower half. The deck
says so at `slides/lecture-14.html:893-895`: *"Median 3.19 … the ladder sits
below the He median — deliberately"*. Grepped the notebook: cells 38 and 59 are
eight cells apart and no markdown connects them.

**Severity:** misleads a student
**Origin:** hand-written prose
**Fix:** one sentence after cell 59 — "the ladder's `clip=1.0` is below every
norm you just measured; that rung is sign descent, deliberately."

---

### Claim 15 — no section carries an examinability marker

**Verdict:** CONFIRMED

**Evidence:** grep for "examinable" across all 69 cells returns one hit, in a
prompt box on the setup cell:

```
cell 2 [ma]: * **Left open:** nothing here is examinable. It is the same four lines every notebook in Part II opens with…
```

Thirteen `## ` sections, no markers. This is the §8.3 finding about lecture 19
reproduced exactly ("appears once… on the section that needed it least").

**Severity:** wrong but harmless
**Origin:** notebook structure
**Fix:** one of *examinable* / *not examinable — engineering* / *beyond the
book, for context* on each of the thirteen section headings.

---

### Claim 16 — cell 67 retrains two configurations cell 38 already trained

**Verdict:** CONFIRMED

**Evidence:** cell 67 runs `train(act="sigmoid", init="torch")` (ladder rung 0)
and `train(**BEST_KW)` (whichever rung `max()` picked). Grep for the three
unpacked names:

```
cell 38: BEST_LABEL, BEST_ACC, BEST_KW = best     # the closing summary uses these
cell 67: _, repaired = train(**BEST_KW)
cell 67: if BEST_KW is not ladder[-1][1]:
cell 67: print(f"{'  (best rung: ' + BEST_LABEL + ')':34s}")
```

`BEST_ACC` is bound in cell 38 and never read again — the number is already on
screen and is recomputed instead. Cached `mps` cost of the two re-runs: 7.5 +
17.1 ≈ 25 s (the report's ~2.3 min is CPU and not reproduced here). The §1.5
risk is real on non-deterministic backends, though the ladder rung and the
re-measure use the same seed and settings, so on CPU they agree.

**Severity:** wrong but harmless
**Origin:** notebook structure
**Fix:** `print` `BEST_ACC` and `rows[0][1]` instead of retraining.

---

## Independent check of the report's "not defects" list

Not required, but the lecture's own sharpest claim was worth settling, and it is
the one the report leaves implicit. **A flat `||dL/dW||` profile does not
certify an initialisation — verified.** Per-layer `||dL/dW||`, layers 1…21,
float64, seed 42, eight batches:

```
Glorot,ReLU  5.91e-03 1.40e-03 1.37e-03 1.24e-03 ... 9.08e-04 8.58e-04 8.92e-04 6.25e-04
He,ReLU      4.70e+00 8.30e-01 8.46e-01 8.08e-01 ... 7.18e-01 7.80e-01 9.02e-01 8.82e-01
Glorot,logistic 1.98e-12 ... 4.29e-01 1.38e+00      (spread 7.0e+11)
default,logistic 4.57e-17 ... 6.61e-02 4.87e-01     (spread 1.8e+16)
```

Across layers 2–20 the Glorot+ReLU profile varies by a factor of 1.6; the whole
end-to-end 6.63 is the 3072→100 first layer. He+ReLU behaves identically (5.21,
same first-layer step). The two ReLU rows are indistinguishable in shape while
their ρ differ by 0.7045 against 0.9970 — so Lecture 13's diagnostic passes the
wrong initialisation, exactly as cell 28 claims, and the cancellation is
`rho/fwd` = 1.0321 against 1.0328. Also re-derived and correct: the variance
identity (predicted 0.1000/1.0000/2.0000/5.0000 vs measured
0.1008/0.9971/2.0034/4.9614); `E[relu(z)²]/E[z²]` = 0.5035; the four ρ
predictions 0.1443 / 0.2500 / 0.7071 / 1.0000; √(1/2) = 0.70711 and
0.70711¹⁹ = 1.381e-03 (three orders) against the logged `dW₁/dW₂₀` = 6.9115e-16
(15.2 orders); the factor of 30 (0.01/0.000326 = 30.67); the 15% assert passing
on all four schemes, worst case 6.4%; and 200/10,100 = 1.98% with the printed
counts 500,210 → 504,210. Every item on the report's "not defects" list that I
touched holds.

---

## Summary

```
confirmed: 15   false positive: 1   unverifiable: 0
of the confirmed, 9 mislead a student
origin split — prose: 8   code: 3   structure: 5
```

Confirmed and misleading: 1, 2, 3, 4, 5, 8, 9, 10, 14.
Confirmed, lower severity: 7, 11, 12, 13, 15, 16.
False positive: 6.

**Scope notes rather than verdict changes.** Claim 8's CPU column for the four
training cells and claim 9's CPU durations for cells 43/46/53 are the report's
own measurements; I confirmed both headlines by other means (direct measurement
of §4, cached `mps` totals) without executing training. Claim 3's ordering is
established on the cached `mps` seeds only; the report says as much itself.

**duplicates:** claims 6 and 7 are the same underlying defect — §4.1 name reuse
across cells in the cell 20/23 block — counted twice, and claim 7's second
paragraph counts `g` and `fwd` a third time. Claims 2 and 3 have a common root
(a single-seed ladder reported where the deck reports five-seed means) but are
distinct defects: one is a misquotation, the other is an unqualified winner.
Claims 8 and 9 are both §7.1 but are not duplicates — wrong markers versus
missing ones.
