# Authoring guide

The working spec for *Applicazioni Informatiche del Machine Learning*. Replaces
the earlier `TRICKS.md` and `GUIDELINES.md`, both of which described a course
structure this one no longer uses.

`LECTURES.md` is the authoritative plan. This file says how to build a lecture
that matches it.

---

## 1 · The organising principle

**One topic per lecture, in the order of the primary text, each lecture
self-contained.**

The test for any slide: a student who missed the previous lecture must still be
able to follow it. Where a lecture genuinely depends on an earlier result, state
the result on the slide rather than referring to it — *"recall from Lecture 8
that the principal subspace minimises reconstruction error"*, with the statement
written out, not a pointer to it.

Nothing in this course is wrong on purpose. No planted defect, no
under-specified prompt shown failing, no number committed in order to be
falsified later. If a method has a failure condition, state it as a property of
the method.

---

## 2 · Deck anatomy

90 real minutes = 2 academic hours × 45. Roughly 75 minutes on slides, the last
15 on the notebook.

| min | block | slide classes |
|---|---|---|
| 10 | where we are · what the method is for | `.divider`, prose |
| 20 | **the mathematics** | `.eqn-box`, `.panel-math`, `.derivation` |
| 40 | **the method** — mechanics, hyperparameters, failure conditions, worked example | code, `.figslide`, tables |
| 15 | further ground · variants · what is used in practice | tables, prose |
| 5 | the notebook | `.notebook-slide` |

**The clock is a constraint, not a decoration.** Count slides against it before
adding any. A deck runs 70–90 slides at this pace; below 60 the lecture will end
early, above 100 it will not finish.

Every deck opens with a title slide and a *where we are* slide placing the
lecture in the arc, and closes with a summary slide of the three or four things
to remember.

---

## 3 · Recurring devices

Four, and each appears in every lecture.

### 3.1 The mathematics block

One object, derived — not stated. The derivation is examinable (40% of the
written paper), so it must be complete enough to reproduce from the slides
alone. Each step gets its own slide or fragment; no slide carries two
non-obvious steps.

End the block with **what the result buys you**: one slide connecting the object
to the method about to be taught.

### 3.2 The worked example

Real numbers from a real dataset, computed by `tools/make_figures.py`, never
invented. If a slide states a figure, the script that produced it writes it into
`assets/figures/figures.json`, and the two must agree. **If a slide and the
script disagree, the script is right and the slide is a bug.**

### 3.2a A wall-clock second is not a figure.

`figures.json` is the authority for every number on a slide — except a
duration, which is a property of a machine and cannot be reproduced by anyone
else's. Three rules follow, and they are not optional in the GPU lectures,
which are full of timings:

1. **A duration on a slide is stated to one significant figure and labelled**
   as one machine's measurement. "About 80 seconds" is honest; "78.3 s" claims
   a precision the number does not have.
2. **Never put a duration in a column that invites a comparison the timing
   cannot support.** Lecture 8 shipped a table where full SVD beat randomised
   PCA by 0.007 s; on another machine the order reverses. A reader takes an
   ordering from a table whether or not the prose claims one. If two timings
   are within noise of each other, **say that** — it is the finding.
3. **A ratio is safer than a second, and still not safe.** The same sweep
   measured 37× faster on one machine and 22× on another. Quote the order of
   magnitude, and make the *shape* the claim: a large saving bought at no cost
   in quality.

The notebook must agree with whatever the slide says, which in practice means
the notebook prints its own timing and says in the surrounding prose that the
reader's number will differ. A notebook that reports durations should cap BLAS
threads before importing numpy, so at least its own numbers are repeatable on
one machine:

```python
import os
os.environ["OMP_NUM_THREADS"] = "2"      # read at import time; setting this
os.environ["OPENBLAS_NUM_THREADS"] = "2" # after `import numpy` does nothing
```

### 3.3 The failure conditions

Every method gets a slide naming what it cannot do and what breaks it —
stated as a property, with the condition under which it bites. This is where the
old course's planted defects go: leakage, imbalance, vanishing gradients,
temporal correlation, popularity bias. They are taught, not sprung.

### 3.4 The notebook slide

The last slide of every deck: what the notebook contains, what to run first,
what needs a GPU, what to change and see what happens. Links to the Colab badge.

---

## 4 · Notebooks

One per lecture, `notebooks/lecture-NN.ipynb`, complete and correct.

### 4.1 Every code cell is preceded by its prompt.

A markdown cell in this exact form:

```
> **Prompt · short name**
>
> **input** · what it gets
>
> **output** · what it must produce
>
> **constraint** · the decision the prompt must make explicit
>
> **check** · a test whose expected answer can be worked out before running
```

The prompt is a **specification**: what you would have to ask an assistant for
in order to get the cell below it. It is not a transcript, and it is never
followed by an annotation explaining how it fails.

Prefer `check ·` clauses whose answer is knowable on paper — a shape, a count, a
parameter-count arithmetic, a value a formula predicts.

### 4.2 Comments explain why, not what.

`# fit on train only, so the test statistics never enter the training path` —
not `# fit the scaler`. A comment that restates the line is noise.

### 4.3 Every number in the prose is computed by the notebook.

Any figure in a markdown cell must appear in the stored output of a cell of the
same notebook. Never transcribed from a previous run.

### 4.4 One name, one meaning, for the whole notebook.

If `X_train` is a DataFrame in cell 10 it is a DataFrame in cell 40.

### 4.5 State the cost.

Any cell over ~20 seconds carries a ⏱ marker and a wall-clock estimate in the
markdown above it, and says whether it needs a GPU.

### 4.6 It runs cold.

Restart-and-run-all from a fresh Colab kernel, top to bottom, no manual steps,
no cell that must be run twice. Data is downloaded by a function that works on
any machine.

---

## 5 · Slides

### 5.1 Six type steps, and no inline font sizes.

The scale is declared in `assets/css/custom.css`. If none of the steps fits, add
one there with the reason written on the line — do not write `1.12em` into a
slide.

### 5.2 One idea per slide.

If a slide needs two sentences of explanation to be readable, it is two slides.

### 5.3 Figures are generated, not drawn by hand.

Plots come from `tools/make_figures.py`. Diagrams (`d-*.svg`) are hand-authored
in the theme palette, and are for structure a plot cannot show.

### 5.3a Escape `<` and `>` in prose and in inline maths.

`$t_1 < \dots < t_k$` written with bare angle brackets is invalid HTML that
browsers happen to recover from, and it also truncates the deck checkers'
notion of a text run at the `<`, which hides other faults on the same line.
Write `&lt;` and `&gt;`, as `$z_i &gt; z_j$` already does.

### 5.4 Nothing overflows.

`python3 tools/check_overflow.py` before committing. A slide that overflows on
the projector is a slide nobody reads the bottom of.

### 5.5 Never name a weekday.

Lectures refer to one another by number or relatively, so the material is
independent of the timetable.

---

## 6 · Marking scope

Every section gets one of: **examinable**, **not examinable — engineering**, or
**beyond the syllabus, for context**. A student must never have to guess.

The examinable surface is Géron Chapters 1–16 plus the lecture notes for
Lectures 19–22.

---

## 7 · Checks

```bash
python3 tools/check_all.py          # everything below
python3 tools/check_notebooks.py    # notebook rules in §4
python3 tools/check_decks.py        # deck structure
python3 tools/check_overflow.py     # §5.4
python3 tools/check_provenance.py   # §3.2 — slide figures against figures.json
python3 tools/check_diagrams.py     # d-*.svg palette and fonts
```

## 8 · Pre-flight

Before committing a lecture:

1. Restart-and-run-all from a cold kernel. It passes.
2. Every prose figure re-derived from the notebook's own output.
3. Every slide figure agrees with `figures.json`.
4. Every cross-reference to another lecture states the result, not just a pointer.
5. Slide count against the clock.
6. `check_all.py` clean.
7. Read the rendered deck and the rendered notebook, not the source.
8. Every instruction to the reader attempted as a student with a laptop and no GPU.
