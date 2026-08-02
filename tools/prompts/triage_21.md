# Triage — lecture 21

Artefact: `notebooks/lecture-21.ipynb` (68 cells: 45 markdown, 23 code).
Source: `tools/notebooks/lecture_21.py`.
Claims triaged: the 23 numbered items, the 7 "checked and clean" bullets, the
3 "not checked" bullets (claims 24–33), plus the dead-parameter note (34).

**Environment.** python 3.13.5, torch 2.13.0, transformers 4.57.3,
scikit-learn 1.7.2, Apple silicon, 16 cores. Corpus read from the cached
`notebooks/datasets/aclImdb`; `distilbert-base-uncased` read from the local HF
cache with `HF_HUB_OFFLINE=1`. **No GRU was trained.**

**Stated once, not repeated per claim (brief §"What counts as UNVERIFIABLE"):**
the notebook stores zero outputs, so no prose figure can be reconciled against
the file itself. Every numeric verdict below is a re-derivation from the corpus,
not a comparison against a stored output.

**A caveat that bears on every wall-clock number below.** The machine was under
`load average 135` while these timings ran (many concurrent agents). All my
timings are therefore upper bounds on an idle machine of the same class, and
they are not directly comparable to the Phase A report's timings, which were
probably taken under a different load. Where a timing verdict turns on a ratio
rather than an absolute, I say so.

---

### Claim 1 — the notebook stores no outputs at all
**Verdict:** CONFIRMED
**Evidence:**
```
$ python3 -c "import nbformat; nb=nbformat.read('notebooks/lecture-21.ipynb',as_version=4); ..."
total cells 68
markdown cells 45   code cells 23
TOTAL OUTPUTS 0
```
Every one of the 23 code cells has an empty `outputs` list.
**Severity:** misleads a student
**Origin:** notebook structure
**Fix:** execute the notebook once and commit it with outputs, or drop the prose
figures that cannot be checked.

---

### Claim 2 — "one review of 2,470 words" is 2,473
**Verdict:** CONFIRMED
**Evidence:** re-derived cell 12's tokenizer over all 25,000 training reviews:
```
CLAIM2 max train review length = 2473
  mean 234 median 174 p90 458
  cutting at 192: truncates 43.8% of reviews, keeps 66.7% of all words
```
Cell 13's `catch` says *"One review of 2,470 words stretches the axis"*.
**Severity:** cosmetic
**Origin:** hand-written prose
**Fix:** 2,470 → 2,473.

---

### Claim 3 — "seeing 100,000 distinct words"; the real count is 87,171
**Verdict:** CONFIRMED
**Evidence:**
```
CLAIM3/9 distinct over 25000 train = 87171
```
Cell 13's `student` field says *"seeing 100,000 distinct words and concluding the
model needs a bigger vocabulary"*. The figure the reader will actually see
printed by cell 14 is 87,171 — 15% below the number the prompt box attributes to
them.
**Severity:** wrong but harmless
**Origin:** hand-written prose
**Fix:** 100,000 → 87,171, or write "nearly ninety thousand".

---

### Claim 4 — "thirteen lines" describes a class that is not in the notebook
**Verdict:** CONFIRMED
**Evidence:** counted cell 38's `GRUClassifier` as shipped — 19 non-blank lines,
none of them comments:
```
CLAIM4 non-blank lines in class: 19
CLAIM4 non-blank non-comment:    19
```
Stripping the `last_of_padding` branch and the `init`/`freeze` transfer
machinery leaves 10 lines of substance (13 once `def __init__`, `super()` and the
`class` line are all counted) — i.e. thirteen describes a class the notebook does
not contain. Cell 36 says *"Thirteen lines, and eleven of them are Lecture 12."*
**Severity:** wrong but harmless
**Origin:** hand-written prose
**Fix:** state the shipped count, or say "thirteen lines of substance once the
deliberate defect branch and the transfer machinery are set aside".

---

### Claim 5 — "⏱ about 15 seconds" for cell 20
**Verdict:** CONFIRMED (the 15 s figure is wrong), with a correction to the
report's own number
**Evidence:** cell 20 run verbatim on the cached corpus:
```
CLAIM5 cell 20 total 86 s  (fit_transform 30 s, transform 19 s, LogisticRegression 37 s)
CLAIM5 tf-idf accuracy 90.1%  features 200000
```
86 s, not the report's 187 s and not the notebook's 15 s. That is **5.7×** the
stated figure, not 12× — and on a loaded 12-thread machine, i.e. faster hardware
than a free Colab CPU runtime. The direction and the §7.1 violation are real;
the report's "187 seconds / 161 s in `fit_transform`" is not reproducible here
and looks like a heavily-contended measurement.
**Severity:** misleads a student
**Origin:** hand-written prose (the `# ⏱ about 15 seconds` annotation and the
prompt label duplicate the same wrong figure)
**Fix:** re-time on an idle machine and state a range with the hardware, e.g.
"⏱ 1–3 minutes on a CPU runtime".

---

### Claim 6 — the headline "the bug costs N points" compares last-epoch numbers while `train()` returns best-epoch weights
**Verdict:** CONFIRMED
**Evidence:** cell 41, the last three statements of `train()`:
```python
    net.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    print(f"  {tag}: keeping epoch {best_epoch} (val {best_val:.4f})")
    return net, curve, losses
```
`curve` is appended once per epoch and is never touched by the reload. Cell 46
then prints:
```python
print(f"\ncorrect        {curve_scratch[-1]:.1%}")
print(f"last of padding {curve_padbug[-1]:.1%}")
print(f"the bug costs   {100 * (curve_scratch[-1] - curve_padbug[-1]):.2f} points")
```
So the cell prints the *last* epoch for both models while both models in memory,
and both `best_val` lines already printed above it, are the *best* epoch. With
`EPOCHS = 2` these coincide only when epoch 2 won both runs. §1.5.
**Severity:** misleads a student
**Origin:** generated code
**Fix:** `max(curve_scratch)` and `max(curve_padbug)`.

---

### Claim 7 — the prompt promises "every configuration's test accuracy"; the cell scores three of five
**Verdict:** CONFIRMED
**Evidence:** cell 52 and cell 57 discard their models:
```python
_, curve_wp_random, _ = train(GRUClassifier(tk.vocab_size), ...)   # cell 52
_, curve_frozen, _    = train(GRUClassifier(tk.vocab_size, init=Z, freeze=True), ...)  # cell 57
```
Cell 60's `results` dict has exactly three GRU rows — `scratch`, `tuned`,
`padbug`. Cell 62 plots four curves. Cell 59's prompt `output` field says *"every
configuration's test accuracy, beside both anchors"*.
**Severity:** misleads a student
**Origin:** generated code
**Fix:** bind the two discarded models (`wp_random, curve_wp_random, _ = …`) and
add their rows to `results`.

---

### Claim 8 — the header claims a preserved ordering that §11 calls a null result
**Verdict:** CONFIRMED (the contradiction), UNVERIFIABLE (the accuracies)
**Evidence:** both texts are in the file. Cell 0:
> "The accuracies are lower than the deck's; the *ordering* of the four
> configurations is the same, and the ordering is the point."

Cell 53:
> "At the deck's scale these two land within a few hundredths of a point of each
> other — a **null result, not a ranking** … At this notebook's smaller scale it
> is usually worse."

A header that makes the ordering "the point" and a section that retracts the
ordering for two of the four configurations cannot both stand. §2.4 requires the
spread to be stated where the headline is; §2.3 requires the correction to
propagate back. Neither happens. I did **not** measure any accuracy.
**Severity:** misleads a student
**Origin:** hand-written prose
**Fix:** in cell 0, say which part of the ordering survives — "the pretrained
runs sit above the random ones; the two middle configurations are within noise".

---

### Claim 9 — the hapax diagnosis is computed on 25,000 rows, the model is fitted on 5,000
**Verdict:** CONFIRMED
**Evidence:** cell 12 builds `train_tok` from all of `train_x`; cell 14 counts
over it; cell 24 fits `w2i` from `fit_x` only. Both computed:
```
CLAIM3/9 distinct over 25000 train = 87171
CLAIM9  hapax over 25000 = 35344 (40.5%)
CLAIM9  distinct over 5000 fit = 42370 ; hapax 18014 (42.5%)
```
The report's "40.6%" is 40.5%; its 42,370 / 42.5% for the fit split is exact.
The diagnosis "two words in five are seen once" survives either way, but the
rows it is stated on are not the rows the vocabulary is built from and the
notebook never says so. §2.1.
**Severity:** misleads a student
**Origin:** hand-written prose
**Fix:** name the rows — "over the 25,000 training reviews" at cell 14, and add
the fit-split figure where the vocabulary is built.

---

### Claim 10 — "you cannot learn a vector from one example" describes words that get no vector at all
**Verdict:** CONFIRMED
**Evidence:**
```
CLAIM10 vocab kept = 19998   min count among kept = 2   last word: embarrasment
CLAIM10 kept words with count==2: 1746 (8.7% of kept)
CLAIM10 any kept word with count==1? False
```
Not one once-seen word survives the `most_common(19_998)` cut, so every hapax
maps to `[UNK]` (index 1) and has no row of its own. The words the sentence is
actually about are the 1,746 seen exactly twice. Cell 15 says *"You cannot learn
a 128-dimensional vector for a word from one example."*
**Severity:** misleads a student
**Origin:** hand-written prose
**Fix:** "Two words in five never enter the table at all — they become `[UNK]`.
The ones that do enter it are barely better off: 8.7% of the vocabulary is seen
exactly twice."

---

### Claim 11 — `Z / Z.std() * 0.1  # match nn.Embedding's own scale` does not match it
**Verdict:** CONFIRMED
**Evidence:**
```
CLAIM11 projected std BEFORE rescale = 0.0698
CLAIM11 std AFTER rescale            = 0.1000
CLAIM11 nn.Embedding(30522,128).weight.std() = 1.0001
CLAIM11 with padding_idx=0            std = 0.9997
CLAIM11 explained variance 46.9%
```
`nn.Embedding` initialises from `N(0, 1)`, so its own scale is 1.00. The line
sets the transplanted table to a **tenth** of that. The comment at cell 55 and
the `catch` at cell 54 both give the same false justification.
**Severity:** misleads a student
**Origin:** hand-written prose
**Fix:** "rescale to 0.1 — a tenth of `nn.Embedding`'s N(0,1) default, so the
transplanted table does not dominate the randomly-initialised GRU".

---

### Claim 12 — `padding_idx=0` does not keep row 0 at zero once `init` is copied in
**Verdict:** CONFIRMED (the mechanism), with a correction to the count
**Evidence:**
```
CLAIM12 no-init:      row0 norm = 0.0000
CLAIM12 tuned  init:  row0 norm after copy = 1.0890
CLAIM12 frozen init:  row0 norm after copy = 1.0890
CLAIM12 emb.weight.grad[0] abs max = 0.000e+00   (padding_idx zeroes the gradient)
CLAIM12 tuned row0 norm after 5 Adam steps = 1.0890
CLAIM12 cell32 assert holds at construction: True
```
`self.emb.weight.data.copy_(torch.from_numpy(init))` in cell 38 overwrites all
30,522 rows including row 0, and because `padding_idx` zeroes the *gradient*,
Adam never moves it back. Cell 30 ("pins row 0 at zero **and keeps it there**"),
cell 31's constraint and cell 32's assert all describe a state that two of the
runs leave behind. Cell 31's own `catch` — *"assert the padding row is zero AFTER
training too"* — fails on this notebook.
**Correction:** the report says "three of the five configurations". `init=` is
passed in exactly **two** of the five GRU runs (subword/frozen at cell 57 and
subword/tuned at cell 57). The other three (`scratch`, `padbug`, `wp_random`)
keep a zero row 0.
**Severity:** misleads a student
**Origin:** generated code
**Fix:** `init[0] = 0` before the copy, or `self.emb.weight.data[0].zero_()`
after it — and then the `catch` becomes a check the reader can actually run.

---

### Claim 13 — the `[UNK]` demonstration at cell 27 does not show what the prose says
**Verdict:** CONFIRMED
**Evidence:** cell 27's first print line, re-derived exactly:
```
CLAIM13 word tokenizer : ['the', 'plot', 'was', 'unwatchable', 'and', 'utterly', '[UNK]']
CLAIM13 'unwatchable' in vocab: True   count in fit: 24
CLAIM13 'discombobulating' in vocab: False   count in fit: None
```
One `[UNK]`, not two. The prompt at cell 26 says `input · one sentence with two
rare words in it` and `constraint · show the word tokenizer producing [UNK] for
exactly the words that carry the sentiment` — and `unwatchable`, the word that
carries the sentiment, appears 24 times in the fit split and is comfortably
inside the vocabulary.
**Severity:** misleads a student
**Origin:** hand-written prose
**Fix:** pick a sentiment word that is genuinely rare (or lower `VOCAB` for the
demo), or rewrite the constraint to match: one rare word, one common one.

---

### Claim 14 — the padding-invariance test is vacuous for 43.1% of possible rows
**Verdict:** CONFIRMED
**Evidence:**
```
CLAIM14/16 share of fit rows with length == MAXLEN: 43.1%
CLAIM14/16 Lf_w[0] = 155
CLAIM14 using a full-length row r=5 n=192
CLAIM14 correct model identical inputs: True   allclose: True
CLAIM14 BUGGY model moves by 0.000 on a full-length row
CLAIM14 buggy model on row 0 (n=155) moves by 0.234
```
When `n == MAXLEN`, `Xf_w[:1, :n]` and `Xf_w[:1, :]` are the same tensor, so
`torch.allclose` passes for the buggy model too and the "how far it moves" line
prints 0.000 — the flagship test would silently endorse the defect. With seed 42
row 0 happens to be 155 tokens, so the shipped notebook works, by luck. There is
no `assert n < MAXLEN`. This is precisely the "check that passes for the wrong
reason" the audit names as the most dangerous defect in the course.
**Severity:** misleads a student
**Origin:** generated code
**Fix:** `n = int(Lf_w[(Lf_w < MAXLEN).nonzero()[0]])`, or simply
`assert n < MAXLEN, "pick a row that is actually padded"`.

---

### Claim 15 — the last assert is likely to fail on the recommended runtime
**Verdict:** UNVERIFIABLE on GPU; on CPU the report's own numbers do not
reproduce
**Evidence:** cell 65 re-run with untrained GRUs (a forward pass costs the same
whatever the weights are, so no training was needed):
```
CLAIM15 tf-idf + logistic regression    13.1 s  (0.52 ms/review, min 12.9 max 15.9)
CLAIM15 GRU, our words                  44.3 s  (1.77 ms/review, min 44.0 max 49.8)
CLAIM15 GRU, subword                    56.0 s  (2.24 ms/review, min 44.9 max 66.6)
CLAIM15 cheapest = tf-idf … assert passes: True
CLAIM15 encode_words(test_x) alone:  1.8 s
CLAIM15 vec.transform(test_x) alone: 6.8 s
```
On CPU tf-idf wins by **3.4×**, and the assert passes comfortably — the report's
"tf-idf 26.5 s, word GRU 18.3 s" is the opposite ordering and does not reproduce
here. (Its own sentence is also self-contradictory: with those numbers the GRU
wins, so tf-idf would *lose* by 1.4×, not win by it.) The risk it flags is
nonetheless real in direction: 42.5 s of the word GRU's 44.3 s is compute a T4
accelerates, while `vec.transform`'s 6.8 s is not — so a large enough speed-up
does flip the assert. I have no GPU and cannot settle it.
**Severity:** wrong but harmless (as a CPU claim); at-risk on GPU
**Origin:** generated code
**Fix:** none needed until measured on a T4; if it does flip, replace the assert
with a printed ordering and a comment about which side the accelerator helps.

---

### Claim 16 — `assert (Xf_w[0, Lf_w[0]:] == 0).all()` is vacuous when row 0 is MAXLEN long
**Verdict:** CONFIRMED
**Evidence:**
```
CLAIM16 slice length for row 0 (L0=155): 37
CLAIM16 slice length for a full row:      0   (empty -> vacuously True)
CLAIM16 empty-slice .all() is True
```
Same 43.1% of rows, same missing guard as claim 14. **This is the same
underlying defect as claim 14** (no `assert n < MAXLEN` anywhere), counted twice
because it appears in two cells.
**Severity:** misleads a student
**Origin:** generated code
**Fix:** as claim 14 — one guard, applied at cell 35 and cell 49.

---

### Claim 17 — five cells announce the padding defect before it runs
**Verdict:** CONFIRMED
**Evidence:** all five located in the file:
- cell 0 — *"Cells marked **⚠ read before running** contain a defect on purpose."*
- cell 34 (`student`) — *"…which is the setup for the assistant failure two sections down."*
- cell 38 — `if self.last_of_padding:                 # the assistant's version`, in the class body, **eight cells** before cell 46 uses it
- cell 44 — a ⚠ section heading plus four paragraphs naming `out[:, -1, :]` outright
- cell 45 — the prompt box, `student` field: *"exactly this"*

§8.1's evidence is that lecture 19 announced its trap four times and *"nobody
falls in"*. This is five, and the third is inside the code the reader is asked
to review.
**Severity:** wrong but harmless
**Origin:** notebook structure
**Fix:** §8.1's preferred shape — let cell 46 run unannounced, have the reader
write the number down, then open §10 with the ⚠ and the contrast.

---

### Claim 18 — the defect is not what the notebook says an assistant returned
**Verdict:** CONFIRMED
**Evidence:** cell 44 presents the defect as *"the code it returns"* from a
single under-specified prompt. What cell 46 actually runs is
`GRUClassifier(VOCAB, last_of_padding=True)` — a flag on a class that also
contains the correct path. Cell 37's prompt box specifies the *correct* model
(`constraint · pack_padded_sequence with enforce_sorted=False, and take the
final hidden state of BOTH directions`) while cell 38 contains both models, so
that box is not a specification of its own cell. §4.4.
**Severity:** wrong but harmless
**Origin:** notebook structure
**Fix:** either say plainly that the defect is staged as a flag, or move the
`last_of_padding` path into its own class defined at cell 46.

---

### Claim 19 — two numbering systems for the same referent, adjacent, neither defined
**Verdict:** CONFIRMED
**Evidence:** cell 5's `left_open` — *"the condition **application 2** spent
ninety minutes on"*; cell 7, the next markdown cell — *"the condition **Lecture
4** spent ninety minutes on"*. Both resolve: `README.md` line 17, *"Each
application spans two lectures"*, and `LECTURES.md` line 14, *"Applications are
covered in pairs of consecutive lectures"* — so application 2 = lectures 3–4,
application 5 = lectures 9–10 (cell 54 "application 5" / cell 55 "Lecture 10",
PCA/SVD), application 6 = lectures 11–12. The cross-references are correct;
nothing in the notebook tells the reader the two systems are the same thing.
§7.5, not §3.3, exactly as the report says.
**Severity:** wrong but harmless
**Origin:** hand-written prose
**Fix:** one clause on first use — "application 2 (lectures 3–4)".

---

### Claim 20 — "Reviewer question 3" is never defined in this notebook
**Verdict:** FALSE POSITIVE
**Evidence:** cell 44 defines it in the same sentence that names it:
> **Reviewer question 3: what is the shape here?** Position `MAXLEN - 1` is
> padding for every review shorter than `MAXLEN` — which is most of them.

That is verbatim `TRICKS.md` line 111 (`3. What is the shape here?` — the report
cites line 110, which is question 2). A reader who has never seen the list loses
nothing: the question is spelled out where it is used. The same pattern is used
across the course (lecture-05 cell "Reviewer question 3 — *what is the shape
here?*", lecture-07, lecture-15).

**Residual worth keeping.** The claim's *second* half is a real observation and
should not be lost with the false headline: cell 44 uses "shape" to mean *which
positions hold real tokens*, and cell 47, three cells later, says *"a test on
output shape does not"* catch the bug — "shape" there meaning `(N, 2)`. Two
meanings, one word, three cells apart. That is a genuine §7.5 friction and is
worth a one-word edit.
**Severity:** cosmetic
**Origin:** hand-written prose
**Fix:** cell 47 — "a test on the output's *dimensions* does not".

---

### Claim 21 — every ⏱ is a GPU figure and the CPU numbers are adjectives
**Verdict:** CONFIRMED (headline), with a correction to its closing sentence
**Evidence:** cell 39 — *"⏱ **about 40–90 seconds per run** on a GPU, several
minutes on a CPU."* Cell 42's prompt label — *"⏱ 40-90 s — our words, random
table"* — carries no CPU figure at all. So the CPU cost is given exactly once,
as the adjective "several minutes", which is what §7.1 asks not to do; the
report's stronger "it is nowhere in the file" is an overstatement.
Micro-benchmark of the loop (12 optimiser steps on random labels, **not** the
training cell; no accuracy produced, no model kept):
```
torch threads 12 : 0.687 s/train-batch x 79 batches x 2 epochs + val = 113 s (1.9 min)
torch threads 2  : 0.944 s/train-batch x 79 batches x 2 epochs + val = 153 s (2.6 min)
```
So ≈2–3 minutes per packed run under heavy load — the same order as the report's
"3 min 20 s", and consistent with "several minutes". Five runs ≈ 10–13 minutes.
**Severity:** wrong but harmless
**Origin:** hand-written prose
**Fix:** put the CPU number in the prompt label too: "⏱ 40–90 s on a GPU, ≈3 min
on a CPU runtime".

---

### Claim 22 — no total is stated, and "the whole notebook finishes in a few minutes" is out by an order of magnitude
**Verdict:** CONFIRMED — and it can be settled from the notebook's own figures,
with no hardware at all
**Evidence:** summing only the ⏱ the notebook itself states:
```
corpus download        30–90 s   (cell 4)
tf-idf baseline           15 s   (cell 20 label)
five training runs   40–90 s x 5 = 3.3–7.5 min  (cell 39, GPU figures)
test encoding      "about a minute"  (cell 58)
timing cell          "two to three minutes"  (cell 63)
                     ------------------------
                     ~8–14 minutes, on a GPU
```
Cell 0 says *"so the whole notebook finishes in a few minutes"*. It does not add
up even on the notebook's own optimistic GPU numbers. My CPU measurements
(claim 5: 86 s; claim 21: ≈2–3 min per run; claim 23: 5.8 min) put the CPU path
at ~25 minutes on a loaded 12-thread laptop — the report's "35–45 minutes on a
2-vCPU runtime" is plausible but I did not measure a full end-to-end run.
**Severity:** misleads a student
**Origin:** hand-written prose
**Fix:** replace "a few minutes" with a stated total for both paths, and add a
line to the setup cell.

---

### Claim 23 — cell 63's "two to three minutes" for the timing cell
**Verdict:** CONFIRMED
**Evidence:** cell 65 run verbatim (untrained models; forward-pass cost is
weight-independent):
```
CLAIM23 whole cell 65 wall time: 348 s (5.8 min)
```
5.8 minutes at 12 threads on a loaded machine, against a stated "two to three
minutes"; the 2-thread case is worse. The cell does three full passes per
configuration over 25,000 reviews and re-tokenises every time, which is exactly
what makes it expensive.
**Severity:** wrong but harmless
**Origin:** hand-written prose
**Fix:** re-time it and state a range, or drop `repeats` to 2 and say so.

---

## The "checked and clean" findings (claims 24–30)

These are negative claims. CONFIRMED below means *the report is right that there
is no violation* — I re-ran each check rather than taking it on trust.

### Claim 24 — §5.1 / §5.2: no bad markdown indentation, the one fence is at column 0
**Verdict:** CONFIRMED
**Evidence:** parsed all 45 markdown cells:
```
CLAIM24 markdown lines indented >=4 outside a fence: 0 []
CLAIM24 fences: [(21, 0, '```', 'open'), (21, 0, '```', 'close')]
```
One fence, cell 21 (the commitment form), opening and closing at column 0.
**Severity:** n/a
**Origin:** n/a
**Fix:** none needed

### Claim 25 — §3.1: no ```` ```python ```` blocks in any markdown cell
**Verdict:** CONFIRMED
**Evidence:** `CLAIM25 markdown cells with ```python: []`
**Severity:** n/a
**Origin:** n/a
**Fix:** none needed

### Claim 26 — §4.2: every training cell re-instantiates and re-seeds
**Verdict:** CONFIRMED
**Evidence:** cells 43, 46, 52 and 57 each open with `torch.manual_seed(RANDOM_STATE)`
and construct `GRUClassifier(...)` inside the `train(...)` call, so each is
idempotent on re-run. Cell 41's `train()` also clones the state dict
(`{k: v.detach().cpu().clone() …}`), which is the hazard its own `catch` names.
**Severity:** n/a
**Origin:** n/a
**Fix:** none needed

### Claim 27 — §4.1: no name rebound to a different type
**Verdict:** CONFIRMED
**Evidence:** every top-level assignment in the 23 code cells is unique to one
cell: `CLAIM27 top-level names assigned in more than one cell: {}`
**Severity:** n/a
**Origin:** n/a
**Fix:** none needed

### Claim 28 — §7.3: cell 10's "Two things to notice" delivers exactly two
**Verdict:** CONFIRMED
**Evidence:** cell 10 — *"Two things to notice: how long a review is, and how
many distinct words there are."* Cell 12 delivers the lengths, cell 14 the
distinct count. Two named, two delivered.
**Severity:** n/a
**Origin:** n/a
**Fix:** none needed

### Claim 29 — the download size, "about 80 MB"
**Verdict:** CONFIRMED
**Evidence:** `notebooks/datasets/aclImdb_v1.tar.gz` is 84,125,825 bytes =
80.23 MiB.
**Severity:** n/a
**Origin:** n/a
**Fix:** none needed

### Claim 30 — "25 times larger" (cell 66)
**Verdict:** CONFIRMED, and the report's caveat is right
**Evidence:**
```
PARAMS word GRU 2,634,754   subword GRU 3,981,570   distilbert 66,362,880
PARAMS ratio vs word 25.19   vs subword 16.67
```
25.2× against the word-vocabulary GRU. Cell 66 does not say which GRU it means,
and against the subword model — which is the one §11 spends its effort on — the
factor is 16.7×.
**Severity:** cosmetic
**Origin:** hand-written prose
**Fix:** "25 times larger than the word-vocabulary model".

---

## The "not checked" declarations (claims 31–33)

### Claim 31 — every GRU accuracy, every "which configuration wins", the header's ordering
**Verdict:** UNVERIFIABLE
**Evidence:** training five models was out of scope by instruction. The report's
reason — that single-seed numbers cannot settle the question the notebook itself
calls a null result (cell 53) — is sound, and is the same point as claim 8.
**Severity:** n/a
**Origin:** n/a
**Fix:** none needed — but note this leaves claim 8's *numeric* half untested.

### Claim 32 — all GPU timings, including whether cell 65's assert fires on a T4
**Verdict:** UNVERIFIABLE
**Evidence:** no CUDA device available; `torch.cuda.is_available()` is False on
this machine. See claim 15 for the CPU half.
**Severity:** n/a
**Origin:** n/a
**Fix:** none needed

### Claim 33 — whether `pip install transformers` is needed in Colab
**Verdict:** UNVERIFIABLE
**Evidence:** the notebook half is checkable and true: cell 3 installs nothing,
and cell 27 does `from transformers import AutoTokenizer` (cell 55 also imports
`AutoModel`). Whether the current Colab image preinstalls `transformers` is a
fact about a runtime I cannot reach from here.
**Severity:** n/a
**Origin:** n/a
**Fix:** none needed — but a one-line `!pip install -q transformers` in the setup
cell would cost nothing and remove the question.

---

## One extra, outside the 33

### Claim 34 — `double_softmax` is defined, branched on, and never passed `True`
**Verdict:** CONFIRMED
**Evidence:** two occurrences in the whole notebook, both inside cell 41:
```
cell 41 | def train(net, Xf, Lf, yf, Xv, Lv, yv, double_softmax=False, tag=""):
cell 41 | if double_softmax:
```
No call site passes it. It is 3 lines (the parameter, the `if`, and the
`torch.softmax` under it), not the 4 the report says.
**Severity:** cosmetic
**Origin:** generated code
**Fix:** delete it, or add a one-line comment saying it is a hook for lecture 22.

---

## Summary

```
confirmed: 29   false positive: 1   unverifiable: 4        (34 entries)
confirmed: 28   false positive: 1   unverifiable: 4        (the 33 in scope)
```
The two lines differ only by claim 34, the dead-parameter note, which sits
outside the report's 33.

- **Confirmed:** 1–14, 16–19, 21–30, 34
- **False positive:** 20
- **Unverifiable:** 15 (GPU half only — the CPU half is confirmed *against* the
  report), 31, 32, 33
- Claim 8 is confirmed on its structural half (the two texts contradict each
  other) and unverifiable on its numeric half; counted as confirmed, since the
  report itself does not claim to have measured the accuracies.

```
of the confirmed, 13 mislead a student
```
Claims 1, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 22.
The remaining confirmed: 4, 15, 17, 18, 19, 21, 23 wrong but harmless;
2, 3, 20, 30, 34 and the seven clean checks cosmetic or n/a.

```
origin split — prose: 15   code: 7   structure: 3   (9 carry no origin)
```
- **prose (15):** 2, 3, 4, 5, 8, 9, 10, 11, 13, 19, 20, 21, 22, 23, 30
- **code (7):** 6, 7, 12, 14, 15, 16, 34
- **structure (3):** 1, 17, 18
- **no origin (9):** 24, 25, 26, 27, 28, 29 (clean checks) and 31, 32, 33 (untested)

The audit's hypothesis holds here: **twice as many defects live in the
hand-written prose as in the generated Python**, and the prose ones are where
the misleading figures are. The three code defects that do mislead (6, 12, 14/16)
share one shape — a check or a headline that is *silently* computing a different
quantity from the one the prose names.

```
duplicates:
```
- **14 and 16** are one defect — the missing `n < MAXLEN` / row-0-length guard —
  counted twice because it surfaces at cell 35 and cell 49. One fix closes both.
- **3 and 9** both hang on the same re-derivation (87,171 distinct words over
  25,000 rows) but are distinct defects: 3 is a wrong figure in a prompt box,
  9 is a diagnosis attributed to the wrong split.
- **8 and 31** are the same unresolved question from two directions: the header's
  ordering claim, which claim 8 shows is contradicted internally and claim 31
  declines to measure.
- **21, 22, 23** are three instances of one §7.1 failure (no CPU wall clock
  anywhere), at three different cells. **5** is a fourth, and the largest.

## Corrections to the Phase A report itself

Recorded because a triage that only grades the notebook misses where the report
would mislead the rebuild:

1. **Claim 5** — cell 20 measures **86 s** here, not 187 s; the overshoot is
   **5.7×**, not 12×. The `fit_transform` split (30 s / 19 s / 37 s) does not
   match the reported 161 s / 27 s at all.
2. **Claim 15** — on CPU tf-idf is **3.4× faster** than the word GRU and the
   assert passes; the report's ordering is reversed, and its sentence
   ("tf-idf wins by 1.4×") contradicts the numbers printed beside it.
3. **Claim 12** — `init=` reaches **two** of five configurations, not three.
4. **Claim 20** — the reviewer question *is* defined in cell 44, verbatim; and
   `TRICKS.md` line **111**, not 110.
5. **Claim 9** — hapax over 25,000 rows is **40.5%**, not 40.6%.
6. **Claim 34** — the dead parameter is 3 lines, not 4.
