# How these decks are made

The working spec for *Applicazioni Informatiche del Machine Learning*. Lectures 1
and 2 are the template; everything here is a rule the remaining twenty-two decks
must follow, or a trap that has already cost us once.

Read this before writing a slide.

---

## 1. The organising principle

**Chapters never introduce themselves. Every method enters the course at the
moment an application breaks without it.**

Concretely, this means a Fix lecture may only teach a method that repairs a
number the students themselves produced in the preceding Build lecture. If you
cannot name the broken measurement, the method does not belong there yet.

The test: for every method taught, you must be able to complete the sentence
*"we are doing this because in the last lecture, X measured Y and that was
wrong."* If you cannot, cut it or move it.

---

## 2. Deck anatomy

Every application is two lectures of 90 real minutes (2 academic hours × 45 min).

### A · Build

| min | block | slide classes |
|---|---|---|
| 15 | the problem, the data, the stakeholder | divider, prose, tables |
| 15 | choose a metric, **commit a number in writing** | `.commit-slide` |
| 60 | build the simplest thing that runs | code, figures |

Ends with everyone holding a number they are pleased with, and which is wrong.

### B · Break → Fix

| min | block | slide classes |
|---|---|---|
| 20 | the mathematical thread | `.badge-math`, `.eqn-box`, `.panel-math` |
| 20 | diagnose why the number was wrong | `.panel-fail` |
| 35 | the method that repairs it | `.panel-fix`, code, figures |
| 15 | re-measure, red-team | `.commit`, tables |

**The clock is a constraint, not a decoration.** Lecture 1 shipped with 25
minutes allotted to a 60-minute build; if the build does not finish, no number is
committed and the following lecture has nothing to bite on. Count your slides
against the clock before you add any.

---

## 3. The five recurring devices

These are what make the course a course rather than twenty-four talks. Each
appears in every application.

### 3.1 The commitment

At the end of every Build lecture, students write on paper: the metric, the
target for a good system, and the number they expect today's model to reach.

Two rules learned the hard way:

- **Anchor it.** A committed number is worthless without a baseline to estimate
  against. Compute the trivial baseline (predict the mean / the majority class)
  *before* the commitment slide, so the commitment is an estimate rather than a
  wish.
- **Do not spoil it.** Never pre-announce that the number will be wrong before
  they have written it down. A ritual whose punchline is given away in advance is
  theatre.

Score it in the following lecture, out loud, against the measured result.

### 3.2 The mathematical thread

Twenty minutes at the top of each Fix lecture, on **exactly one** mathematical
object — the one the application just built depends on. Not a parallel theory
course: the thread must do visible work on a system the students wrote.

Threads are cross-referential and ordered; see `LECTURES.md`. Thread *n* may
assume threads 1…*n*−1 and nothing else.

### 3.3 The worked assistant failure

Every lecture carries at least one, in four beats:

1. an under-specified prompt
2. the plausible code it returns — **which must actually run**
3. the review question that catches it
4. the corrected specification

Choose the failure so that it is *the failure that lecture is about*. A
demonstration of a leak in a lecture that is not about leaks is decoration.

**Measure the damage, and report it honestly even when it is nil.** Lecture 1's
scale-before-split leak costs about a dollar on this dataset. Saying so is the
lesson — but say *why* (an invertible affine map, an equivariant estimator, and
20,640 rows), so students get a decision rule rather than a shrug.

### 3.4 The red team

Fifteen minutes at the end of every Fix lecture: swap notebooks, find the leak.
The five reviewer questions, in this order, every time:

1. What touched the test set?
2. What was fitted, and on what? (`fit` and `transform` are different verbs)
3. What is the shape here?
4. What was dropped — rows, columns, NaNs? Count them.
5. What is the default I did not ask for?

### 3.5 The silent-failure catalogue

The course's spine. Each of these runs, produces a plausible number, and is
diagnosed somewhere in the twenty-four lectures:

| failure | diagnosed in |
|---|---|
| scaler fitted before the split | L2 |
| evaluating on training data | L2 |
| accuracy under class imbalance | L4 |
| hyperparameters chosen on the test set | L2, L6 |
| missing `model.eval()` | L12 |
| missing `optimizer.zero_grad()` | L12 |
| metric averaged per batch, not over the set | L12 |
| random split on autocorrelated time series | L20 |

---

## 4. The provenance contract

**Every number printed on a slide is produced by `tools/make_figures.py`.**

This is the course's main claim to authority — it is the same standard the course
demands of students — and it is the first thing to rot. Lecture 2 shipped with
three code/output blocks copied from the textbook's pipeline, contradicting the
prose beside them.

Rules:

- No number reaches a slide that the script has not written to
  `assets/figures/figures.json`. Prose numbers too — counts, percentages, ranges.
- No pasted REPL output that the script did not generate. If a slide shows
  `>>> something.describe()`, the script must produce that exact block.
- If a slide and the script disagree, **the script is right and the slide is a
  bug.**
- Expensive fits are cached (`fits.pkl` under the scratch dir) so a cosmetic
  re-run takes seconds. Delete the cache to refit from scratch.

Never quote a single-seed number as if it were a measurement. If an effect is
small, run it over ~20 seeds and report the spread — otherwise you are teaching
the exact error the course exists to eliminate.

---

## 5. Scope discipline

Everything taught comes from Géron, *Hands-On Machine Learning with Scikit-Learn
and PyTorch* (2025), **Chapters 1–16**. Nothing else is examinable.

- Datasets must be ones the book uses or names.
- Where the field goes beyond the book, say so on the slide and move on.
- Engineering practice that is genuinely needed but outside the book (Colab
  mechanics, version pinning, notebook hygiene) may be taught **flagged on the
  slide as non-examinable**.
- Prerequisites are by definition outside the book; the site says so.

---

## 6. Figures

Two kinds. Both live in `assets/figures/` and both are committed.

### 6.1 Plots — generated, never illustrative

`tools/make_figures.py`, from the real dataset, with the same code path the
lecture describes.

- **SVG** for anything sparse (bars, lines, heatmaps, small scatters).
- **PNG at 160 dpi** where point count is high — a 20,640-point scatter as SVG
  embeds one path element per district and reaches megabytes.
- Naming: `l<NN>-<slug>.svg|png`, e.g. `l2-train-vs-cv.svg`.
- rcParams live in `setup()` and are shared, so every plot in the course looks
  like the same document.
- Annotate the plot with the point the slide makes — an arrow to the artefact, a
  labelled line at the cap. A plot that needs the caption to be legible is not
  finished.
- Put callouts on empty regions with a white `bbox`. Check: annotations over
  data are unreadable on a projector.

### 6.2 Diagrams — hand-authored SVG

`d-<slug>.svg`. Written by hand, not generated, using the slide palette.

Visual language, consistent across all of them:

| element | treatment |
|---|---|
| box | `rx="7"`, fill `#eef4f8` or `#ffffff`, stroke 2–2.5px |
| the wrong path | `#c0392b`, dashed for the leaking edge |
| the right path | `#1e8449` |
| the mathematical object | `#6c3483` |
| arrows | `marker-end` triangle, `stroke-width: 2.5` |
| body text | 15–17px in a ~1080-wide viewBox |

Every diagram must be legible at 1280×720 with no zooming. Validate as XML
(`xml.dom.minidom.parse`) before committing — a malformed SVG fails silently to
an empty box.

### 6.3 On the slide

```html
<section class="figslide" data-menu-title="◆ What it shows">
  <h2>The claim the figure supports</h2>
  <figure class="fig fig-wide">
    <img src="../assets/figures/l2-train-vs-cv.svg"
         alt="Describe what is visible, not what it means.">
  </figure>
  <p class="takeaway muted">One sentence. What should they see?</p>
</section>
```

Modifiers: `.fig-wide` (short, full width), `.fig-tall` (square plots),
`.fig-inset` (sharing a slide with text). Every `<img>` gets real alt text.

---

## 7. Slide vocabulary

Defined in `assets/css/custom.css`; use these rather than inventing markup.

| class | use |
|---|---|
| `.title-slide` | deck opener |
| `.divider` | part break — dark, with `.kicker` and `.clock` |
| `.figslide` | a slide whose subject is a figure |
| `.commit-slide` / `.commit` | the written commitment |
| `.panel` | neutral callout |
| `.panel-fail` | the broken thing |
| `.panel-fix` | the repair |
| `.panel-math` | the thread |
| `.panel-book` | quoting the textbook |
| `.badge-build` `.badge-fix` `.badge-math` `.badge-ch` | lecture and chapter tags |
| `.eqn-box` / `.eqn-label` | display maths |
| `.cols` (+`.cols-40-60`, `.cols-60-40`) | two columns |
| `.big-num` (+`.bad`, `.good`) | a single dominant number |
| `.takeaway` | the one-line reading of a figure |
| `.usd` | **currency in prose — mandatory, see §9.1** |
| `.fail` `.fix` `.muted` `.small` `.smaller` `.tight` | inline utilities |

Colour is semantic and load-bearing across all 24 decks: blue = structure,
red = failure, green = repair, purple = the mathematical thread, grey =
secondary. Do not use them decoratively.

---

## 8. Writing conventions

- **Never name a weekday.** Lectures refer to one another relatively: *in the
  next lecture*, *in the previous lecture*, *in two lectures*. The material must
  not depend on the timetable. Enforced by the linter.
- **Never use a third-party notebook**, including the textbook author's. Every
  notebook is ours, written from a specification during the lecture.
- Sentence case in headings. No terminal full stop in a heading.
- Speaker notes (`<aside class="notes">`) carry the things you would say and not
  write: what to ask the room, where they will get stuck, what not to give away.
- Prefer a measured number to an adjective. "Roughly eighteen times better" beats
  "much better", and can be checked.

---

## 9. Traps that have already cost us

Each of these produced a bug that looked fine until someone read the slide
carefully. All are checked by `tools/check_decks.py` where checking is possible.

### 9.1 KaTeX eats currency

`$120,000 and $265,000` is read as inline maths and renders as italic
*120,000and265,000* — both dollar signs and the spaces gone.

**Always** wrap prose currency: `<span class="usd">$120,000</span>`. Table cells
happen to survive (each `<td>` is its own text node) but wrap them anyway for
consistency. Never write `$49{,}037$` intending a dollar sign — that is maths,
and renders without the symbol.

### 9.2 Content-table CSS bleeds into code blocks

reveal renders line-numbered code as `<table class="hljs-ln">`, whose parent is
the `<code>` element — so a `:not(pre) >` guard does **not** exclude it. Scope
content-table rules with `:not(.hljs-ln)`.

### 9.3 `:nth-child` outranks a plain `td` selector

A `tr:nth-child(even) td` rule beats `td.hljs-ln-numbers`, so it applied the
gutter padding to alternate lines only and every other line of source code sat a
few pixels out of true. Keep code-table rules free of `:nth-child`.

### 9.4 Stepped line highlighting gives away the answer

`data-line-numbers="1-8|4|5-6"` on an exam question points straight at the
defect. Use a bare `data-line-numbers` on anything the student is meant to
diagnose; save the stepping for the answer.

### 9.5 Code on a slide must run

Lecture 1 shipped a snippet that raised `ValueError: Input X contains NaN` —
`housing_num` has 207 missing values and `StandardScaler` passes them through.
A slide whose point is careful reading must not show code that crashes.

Related: keep dataframe names consistent across a deck. `housing` used before it
is defined, or redefined to mean something narrower, gives the audience least
equipped to debug a `NameError` exactly that.

### 9.6 Slides overflow silently

reveal scales content down rather than complaining. Measure: content height must
be ≤ 720 in reveal's coordinate space. The check is scriptable — walk the slides,
compare `offsetTop + offsetHeight` against `Reveal.getConfig().height`.

### 9.7 Footers on dark slides

`.deck-footer` sits outside `.reveal`, so no sibling selector reaches it. It is
toggled from JS on `slidechanged` via `.is-hidden` for `.divider` and
`.title-slide`.

---

## 10. Notebooks

One per lecture, `notebooks/lecture-NN.ipynb`, opened from the slides via

```
https://colab.research.google.com/github/fabsilvestri/AIML-Course/blob/main/notebooks/lecture-NN.ipynb
```

- **Ours, always.** Written from a specification, showing the loop rather than
  describing it.
- Structure mirrors the lecture: brief → data → metric → commitment → build, or
  thread → diagnosis → repair → re-measure.
- Include the worked assistant failure: the weak prompt in a markdown cell, the
  plausible code it returns in a code cell **that runs**, the review question,
  and the fix.
- **Assertions after every structural step.** This is the scaffolding the
  audience most needs, being mathematicians rather than engineers:
  ```python
  assert len(X_train) + len(X_test) == 20640
  assert set(X_train.columns) == set(X_test.columns)
  assert not X_train.isna().any().any()
  ```
- State expected wall-clock next to any cell that runs longer than ~20 seconds,
  so "no output" is not read as "it hung" and interrupted.
- Fixed seeds everywhere; `KFold(shuffle=True, random_state=42)` rather than the
  default unshuffled split, so two students with differently ordered frames get
  the same folds.
- Every notebook linked from the slides that use it, and from `index.html`.

---

## 11. Before committing

```bash
python3 tools/check_decks.py     # currency, weekdays, third-party notebooks,
                                 # missing figures, deck length
python3 tools/make_figures.py    # regenerate every figure and figures.json
```

Then look at the deck in a browser. The linter catches what is mechanical; it
cannot tell you a slide is ugly, overfull, or wrong.
