# Notebook guidelines

Rules for authoring and revising the twenty-four Colab notebooks of
*Applications of Machine Learning*.

Every rule below exists because something failed. Three subagents read
`lecture-19.ipynb` as students — a mathematician with weak programming, a
confident skimmer who reads code before prose, and a literal-minded reader
working alone at home with no GPU — and each finding was then re-derived
against the raw data before being written down here. Where a rule cites
evidence, that evidence is a real defect that shipped.

The notebooks are the course's most-used artefact. A student reads them alone,
at night, in a second language, without the lecturer in the room. That reader
is the one these rules protect.

---

## 0 · The one-sentence version

**Every number in the prose must be re-derived from the notebook's own data
before it is written, every comparison must be on matched data, and every
instruction must be one the reader can actually carry out.**

Everything below is that sentence, made checkable.

---

## 1 · Numbers

### 1.1 Re-derive every figure. Never transcribe one.

Any number that appears in markdown must be computed from the notebook's own
data at the moment it is written, and re-checked whenever the cell above it
changes.

*Evidence.* Four separate figures in lecture 19 were wrong, and all four were
checkable in under a minute from data already in the notebook:

| Claim | Reality |
|---|---|
| weekly lags carry "twenty-two times the weight of all the others **put together**" | 22.4 is a ratio of **means**; the sums are 0.53 vs 0.31, a ratio of **1.7** |
| "Christmas Day has about a tenth of a Tuesday's ridership" | 126,000 / 745,000 = **0.169**, about a sixth |
| "Replace `shift(7)` with `shift(-7)`. **The score improves.**" | it gets **worse**, 55,646 vs 55,399 |
| "Some weeks have a *third* dip in the middle of them" | exactly **one** non-Sunday holiday in the plotted window, and it is a Monday fused to the weekend at the right edge |

### 1.2 Prose figures must reconcile with stored outputs.

Every figure quoted in markdown must appear in a stored cell output, or be
derivable from stored outputs by arithmetic stated in the same sentence.
Rounding is fine and must be conventional round-half-up (`44,316.50` →
`44,317`); silent re-rounding in a different direction is not.

*How to check.* `tools/check_notebook_numbers.py` (see §9). Run it before every
commit that touches markdown.

### 1.3 A statistic must survive a defensible re-partition.

If a summary depends on an arbitrary choice of buckets, either justify the
choice in the sentence or report the statistic that does not depend on it.

*Evidence.* Lecture 19 §7 bucketed "weekly lags" as {7, 14, 21, 28} while its
own figure drew lines at all eight multiples of seven. **Lag 35 carries +0.23 —
the second-largest coefficient in the model, larger than lag 7 — and was
counted as evidence against weekly structure.** Under all eight the ratio
becomes −64.5, i.e. meaningless.

### 1.4 Say mean, sum, or median. Never "the weight of all of them".

A ratio of means and a ratio of sums are different numbers and English does not
distinguish them. Name the operation.

### 1.5 Two numbers for the same quantity must be reconciled where they appear.

*Evidence.* Lecture 19 prints "copying yesterday" as **130,844.66** in one cell
and **130,745.60** four cells later. The cause is real and defensible
(`target.shift(1)` loses the first day; `pool.shift(1)` reaches back for it) and
is **never stated**. The notebook had, two cells earlier, taught the reader to
treat exactly this as an alignment bug.

---

## 2 · Comparisons

### 2.1 A comparison must hold the scoring window constant.

When claiming that A beats B, A and B must be scored on the **same rows**.
State which rows. If they differ, the comparison measures the rows.

*Evidence — the most serious defect found.* Lecture 19 §6 claims:

> "The split flattered the model by 7,690 boardings — 17.2%. **Nothing else
> changed.** Same rows, same columns, same estimator, same seed."

Something else did change. The shuffled CV averages over all **1,191** days;
the time split over the last **239**, which §9 itself later proves are 21%
harder. Scoring the shuffled out-of-fold predictions on the same 239 days:

```
shuffled-CV predictions, last 239 days : 52,705
time-split model,        same 239 days : 52,451
difference                             :   +255  (the leaky one is 0.5% WORSE)
```

The entire 7,690 gap is window difficulty. The lesson — shuffled CV is invalid
on a windowed series — is **true**; the evidence offered for it does not
support it.

### 2.2 Prefer the argument that does not depend on the effect size.

When a methodological point is structural, argue it structurally. Then it holds
whether or not the number happens to move on your dataset.

*Evidence.* The valid argument for §6 was available and unconfounded: rows *t*
and *t+1* share **55 of 56 input coordinates**, so under a shuffle the training
set contains rows 98% identical to test rows. That is true regardless of what
the MAE does. A second unconfounded option: fold **spread** was 11,600 shuffled
against 42,000 under `TimeSeriesSplit` — the shuffle manufactured *stability*,
not a better mean, which is what the notebook's own best sentence ("a stable
measurement of the wrong quantity is stable") actually describes.

### 2.3 When you correct an error, propagate the correction backwards.

*Evidence.* Lecture 19 §9 confesses that a model and its baseline must be
scored on the same days — the identical error as §6, one section later — and
corrects only the final table. §6's conclusion stands unretracted, and §9's own
red-team bullet then reasons from a target it has just retracted.

### 2.4 A result inside its own noise is not a result.

If the run-to-run spread exceeds the margin, say so where the headline is
stated, not in an exercise at the end.

*Evidence.* Lecture 19's headline is "the RNN met the committed target". Margin
over target: **894** boardings. Observed spread across the last four readings:
**7,990** — 8.9× the margin. The notebook prints both numbers and lets the
headline stand.

---

## 3 · Code quoted in prose

### 3.1 Quoted code must exist verbatim in a cell of the same notebook.

Copy it from the cell. Verify by string search.

*Evidence.* Lecture 19 §9's post-mortem quotes

```python
for window, target in TimeSeriesDataset(toy, 3):   # <- rebinds `target`
```

That loop appears in no cell, and `toy` exists nowhere in the file. The
diagnosis was correct; the evidence the reader was invited to go and find did
not exist. Two of three readers went looking for it.

### 3.2 Any check offered to the reader must have been executed.

If the prose says "one line settles it", run that line first.

*Evidence.* Lecture 19 §2 offered
`(raw["total"] == raw["bus"] + raw["rail"]).all()` as a check the reader could
run *"without asking anyone"*. It raises `KeyError`: `raw` still carries the
published names, and the rename happens on `df`. The one check offered for
independent verification did not execute — in the section whose entire subject
is the difference between knowing and assuming.

### 3.3 Cross-references must resolve.

"The next cell", "ten cells earlier", "section 4" must be correct. Count them.

*Evidence.* "That is the next cell" pointed two cells away; "ten cells earlier"
was fifteen. A literal reader lost trust in the phrase for the rest of the file.

---

## 4 · Notebook state

### 4.1 One name, one meaning, for the whole notebook.

No variable may be rebound to a different kind of object.

*Evidence.* Lecture 19 devotes 200 words to `target` being clobbered by a loop
variable — and then rebinds `model` from `LinearRegression` to `SimpleRnn`, and
`test_mae` from a float to a loop variable, both unremarked. Use `t`, `m2`,
`fold_mae`. Loop variables in throwaway tests get throwaway names.

### 4.2 Training cells must re-instantiate the model.

A cell that trains must construct the model and the optimiser inside itself, or
it is not idempotent and re-running it silently continues training.

*Evidence.* Lecture 19's cell 40 does not, so any exercise that says "change the
seed and re-run" trains the existing network for a further 200 epochs. The
notebook asks the reader to do exactly that.

### 4.3 Restart-and-run-all must pass, and out-of-order hazards must be named.

State the specific failure a reader will hit, not the general advice.

### 4.4 Do not make blanket provenance claims.

*Evidence.* Lecture 19's header says every code cell is "what came back,
unedited". Cell 42 contains a comment about a bug in a different cell that its
own displayed prompt never mentions — so at least one displayed prompt is not
the prompt that produced the code. The notebook admits this for one cell and
not the other. Say "generated, then regenerated where noted".

---

## 5 · Markdown that renders

### 5.1 No prose line may be indented four or more spaces outside a fence.

Markdown turns it into a code block.

### 5.2 Fences open and close at column 0.

CommonMark permits at most three spaces on a closing fence. More and it does
not close.

### 5.3 ASCII tables need a header row and aligned columns.

### 5.4 An annotation must sit on the row it annotates.

*Evidence — one cell, all four rules broken.* Lecture 19's cell 41 opened a
fence, staircased its table through 1, 2, 3, 4, 22 and 40 spaces of indent,
closed the fence at 40 spaces so it never closed, and left every subsequent line
at 40 spaces. The result: the *"44,317 is not this model's score"* argument, the
final prompt box and all three annotation bullets rendered as grey monospace.

And `<- lowest` sat visually beside **65,723** rather than 44,317. Read as
printed, the notebook said the lowest MAE was 65,723. The literal reader's
honest answer to "which number do I write down?" was **44,317** — which is
precisely the mistake the cell exists to prevent.

**This was the cell teaching "do not select an epoch by its test score", and its
rendering pushed the reader into doing exactly that.** Prefer a real markdown
table to ASCII art wherever the content allows it.

---

## 6 · Prompt boxes

### 6.1 Budget them. Annotation fatigue is measured, not hypothetical.

All three readers independently stopped reading the three-bullet template by
around cell 30 — which in lecture 19 is exactly where the notebook needs them,
because cell 30 is the defect. The notebook's own text admits it: *"This one is
different from the others. Do not skip to the next cell."*

**Current counts, measured: 465 prompt boxes across the course, and every
single one carries the full three-bullet annotation — there is no short form in
use anywhere. Lecture 5 has 30, lecture 6 has 29, lecture 22 has 26, lecture 18
has 24. Lecture 19, the notebook the readers gave up on around cell 30, has
20 — fewer than eight other notebooks.**

Rule: every code cell keeps a **short** box — the specification, so a reader who
cannot read the code can still check intent against output. The full
three-bullet annotation is reserved for the places where the prompt genuinely
failed: aim for **five to eight per notebook**, never more than ten.

### 6.2 "The usual student version" must be observed, not invented.

Ground it in a real failure — one you saw, one from the assistant's actual
output, or a documented library default. If you are inventing a plausible
mistake, cut the bullet.

*Evidence.* Two of the three readers called this the weakest bullet. The
mathematician's objection is decisive: *"I do not write the usual student
version because I do not write PyTorch at all."* Roughly a third of the
annotation wordcount was dead weight for that reader. In lectures 1–18 and
20–24 these boxes are **specifications rather than transcripts**, so the bullet
is invention by construction unless it names a real default.

### 6.3 The specification must state the expected answer, not just the request.

The strongest boxes name a check with a knowable outcome — the parameter-count
arithmetic (`1×32 + 32×32 + 2×32 = 1,120`) was, for the weakest programmer of
the three, *"the only moment this term I felt I checked something rather than
watched it happen"*. Prefer `check ·` clauses whose answer can be worked out on
paper before running.

### 6.4 Use one convention across the course.

Lecture 18 uses the structured `input · output · constraint · check` form,
lecture 19 free prose, lecture 20 annotations with no boxes. Three adjacent
lectures, three conventions. **The structured form is the standard**, because
the `check ·` slot structurally forces an expected answer.

---

## 7 · Instructions the reader can carry out

The literal reader could complete **1.5 of 6** exercises. Every failure traced
to an untimed cell or an unstated re-run order — none to the mathematics.

### 7.1 State wall-clock for every cell over ~20 seconds, and give the CPU number.

*Evidence.* Lecture 19's training cell states no duration. Four of six exercises
were blocked on not knowing whether it takes two minutes or forty on a laptop.

### 7.2 An exercise that requires re-running cells must list which, in order.

"Change the seed" spanned cells 2, 38 and 40, thirty-six cells apart, with a
non-idempotent training cell in the middle.

### 7.3 "Notice N things" must have exactly N findable things.

Or say "look for anything that surprises you". The reader who genuinely looks
and finds N−1 cannot tell whether the failure is theirs.

### 7.4 Do not assume a national calendar.

Memorial Day and the Fourth of July are not common knowledge in Rome. Name the
holiday and the date.

### 7.5 Define vocabulary on first use.

Observed as friction for a second-language reader: *strike day*, *red-team*,
*clobbered*, *load-bearing*, *smoke test*, *flattered the model*.

---

## 8 · Staging the defect

### 8.1 Do not announce the trap four times before it.

*Evidence.* Lecture 19 flags its defect in the header, the how-to-use note, the
section heading, and the paragraph immediately above the cell. The skimmer's
verdict: *"By the time my eye reaches `shuffle=True` I am hunting for one flag
in a nine-line cell. Nobody falls in."*

**Preferred shape:** let the defective cell run unannounced, have the reader
write the number down, *then* open the next section with the ⚠ and the
contrast. Same words, same cells, reordered — and "would you have caught it?"
acquires a real answer.

### 8.2 The best trap is the one that is not labelled.

The skimmer was caught by the *unlabelled* second defect (a baseline scored on a
different window from the models) and not by the labelled one. That is the
model to follow.

### 8.3 Mark what is examinable, everywhere.

The string "examinable" appears **once** in the whole of lecture 19, on the
section that needed it least. Every section gets one of: *examinable*, *not
examinable — engineering*, or *beyond the book, for context*.

---

## 9 · What is machine-checked

`tools/check_all.py` must pass before every commit. Rules enforceable
mechanically, and which must be added to the tooling:

| Rule | Check |
|---|---|
| §5.1–5.2 | no markdown line indented ≥4 outside a fence; no fence marker indented ≥4 |
| §1.2 | every ≥4-digit prose figure appears in a stored output or is flagged for review |
| §3.1 | every ```` ```python ```` block in markdown appears verbatim in a code cell of the same notebook |
| §4.1 | no name bound to two different types across cells |
| §6.1 | count boxes per notebook; warn above ten full annotations |
| §7.1 | any cell whose stored execution exceeded 20 s must have a ⏱ marker in the markdown above it |

Everything else needs a human, and most of it needs a human who is not the
author.

---

## 10 · Pre-flight checklist

Before committing a notebook:

1. Restart-and-run-all from a cold kernel. It passes.
2. Every prose figure re-derived from the notebook's own data. **Not**
   transcribed from a previous run.
3. Every comparison: same rows, and the prose says which.
4. Every quoted code block found by string search in a cell of this notebook.
5. Every check offered to the reader executed, output seen.
6. Every cross-reference counted.
7. `check_all.py` clean.
8. Read the rendered markdown, not the source. Cell 41 is invisible in source
   review and obvious on the page.
9. Every instruction to the reader attempted as if you were a student with a
   laptop, no GPU, and no lecturer.
10. Ask of every headline: *what would this number be if the model had learned
    nothing?* and *is this difference larger than the noise?*

---

## Appendix · Why three readers

One reader finds the defects they are equipped to find. The three profiles
disagreed usefully, and each caught something the others missed:

- **The mathematician** (strong proofs, weak code) found the §6 confound by
  re-running the comparison on matched days. Neither of the others tested it,
  and neither did the author.
- **The skimmer** (reads code and outputs, skips prose) found that the
  *unlabelled* defect caught them and the labelled one did not — a fact only
  visible to someone who reads the way most students actually read.
- **The literal reader** (follows instructions exactly) found that `<- lowest`
  annotates the wrong number, that the suggested check crashes, and that 4.5 of
  6 exercises were impossible alone at home. Every one of those is invisible to
  a reader who skims past an instruction they cannot perform.

Where they disagreed — the engineer attributed the hard held-out window to the
2019 polar vortex, the mathematician to seasonal composition — the disagreement
was itself the finding, and neither matched the notebook's own shrug.
