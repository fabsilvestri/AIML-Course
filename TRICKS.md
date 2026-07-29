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

### 9.1 A dollar sign is a maths delimiter in three different renderers

This is one hazard, not three, and it has now been rediscovered three times in
three files. **A course about money, taught with maths, in three renderers that
all treat `$…$` as an equation.** Read this once and you will not meet it again.

| where | what happens | the defence |
|---|---|---|
| **KaTeX**, in slides and the site | `$120,000 and $265,000` renders as italic *120,000and265,000* — both signs and the spaces gone | wrap every prose amount: `<span class="usd">$120,000</span>` |
| **matplotlib**, in any plot string | mathtext turns a cluster listing into an italic equation reading *187,500(76)225,000* | `"text.parse_math": False` in `setup()`, set script-wide |
| **the checkers**, reading the decks | must tell `$1.0$` (maths) from `$120,000 and ` (currency) or they cry wolf on every slide | the backslash-plus-shape test in `check_decks.py` and `check_provenance.py` |

Two corollaries worth stating outright:

- Never write `$49{,}037$` meaning a dollar amount. That is maths, and renders as
  an italic number **with no currency symbol at all** — which is worse than the
  visible breakage, because it looks deliberate.
- Table cells survive by accident (each `<td>` is its own text node, so there is
  nothing to pair with). Wrap them anyway. The next edit that merges two cells
  into a sentence will silently reintroduce the bug.

If you add a fourth renderer, add a fourth row here rather than a fourth
defence somewhere else.

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

### 9.4 Mathtext on a log axis

`setup()` sets `text.parse_math: False` course-wide (§9.1). matplotlib's own log
formatter then loses: it emits `$\mathdefault{10^{3}}$` and, with parsing off,
draws that string **literally** on the axis. Thirteen figures across Lectures 6,
12 and 16 shipped this way.

It survived every check we had, because the string is legal text, the figure is
a legal SVG, and it clears the 15px floor comfortably — it is just wrong. Use
`figkit.plain_log()`, and note the two things that make a wrong fix look right:

* **Order matters.** `semilogx` re-applies the log scale, and applying a scale
  installs that scale's default formatter. `plain_log` above the plotting line
  is undone by it. Call it *after*.
* **Comments are not renders.** matplotlib's SVG backend writes each label's
  source string as an XML comment above its glyphs, so `grep mathdefault` finds
  both the broken figures and, in principle, harmless ones. Confirm by looking
  for the `$` glyph (`SourceSans3-Regular-24`) in the drawn output before you
  believe either a defect report or a fix.

`tools/check_decks.py` now fails the build on any of it.

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

## 11. Typography and legibility

The deck is projected in a hall, sometimes badly. **Legibility from the back row
beats refinement**, and every number below was measured in the live page rather
than estimated.

### 11.1 The type scale — six steps, no inline sizes

Nineteen sizes accumulated across two decks, thirty-four of them written inline
as `style="font-size:1.15em"`. Most steps sat within 5% of a neighbour: invisible
as a distinction, but a decision to re-make on every future slide.

```css
--t-xs: 0.62em;  --t-sm: 0.78em;  --t-md: 1em;   --t-lg: 1.18em;
--t-h2: 1.62em;  --t-h1: 2.10em;  --t-num: 2.60em;
```

**Never write an inline `font-size`.** If a paragraph needs to be larger, it is
`.lead`; smaller, `.small` or `.smaller`. If none fits, add a step to the scale
deliberately — do not invent `1.12em` on one slide.

`h4` must be an eyebrow (uppercase, tracked, `--t-sm`), not a shrunken heading.
It was smaller than the body text it introduced.

### 11.2 Spacing is canvas pixels, not `em`

Margins in `em` mean a `.small` paragraph is spaced 22% tighter than a body one
and a `.smaller` list 36% tighter — there is no baseline. Use the four-step
rhythm, which is independent of the element's own size:

```css
--sp-1: 8px;  --sp-2: 14px;  --sp-3: 22px;  --sp-4: 34px;
```

### 11.3 Anchor the layout

reveal centres both axes, so the h2 baseline swung **16 px → 241 px** across
Lecture 1 — a title moving a third of the screen between consecutive slides is
the most visible defect from the back of a hall. `center: false`, content flush
left, and only `.title-slide`, `.divider`, `.figslide` and `.commit-slide`
centred by flex.

Consequence worth keeping: with centring off, **overflow becomes visible instead
of being silently re-centred**.

Reserve a band for the footer rather than trimming slides one at a time — a
per-slide trim is a fix that has to be re-made every time content changes:

```css
.reveal .slides > section:not(.divider):not(.title-slide) { padding-bottom: 46px; }
```

**Ordering.** `center: false` changes the measured height of exactly the slides
most likely to be near the limit, so run `check_overflow.py` *after* the layout
change, never before. Trims made against the centred layout are provisional and
may turn out to be unnecessary — or insufficient.

### 11.4 Floors

| element | floor | note |
|---|---|---|
| any text | **18 px** (1/40 of canvas) | 24 px is comfortable |
| code | **18.6 px** | `pre { font-size: 0.62em }` |
| plot tick labels | **15 px** | these are on-slide pixels, see §11.6 |
| text contrast | **4.5:1** | WCAG AA |
| graphics contrast | **3:1** | axes, rules, borders |

Code lines are capped at **80 characters** — measured, an 85-char line at 18.6 px
is 921 px against ~1103 px usable, leaving no headroom for a narrower fallback
mono. `pre { width: fit-content }` so a 22-character snippet is not painted as a
1237 px slab.

### 11.5 Colour, corrected

`--aiml-success` moves from `#1e8449` to **`#14663a`**: the old green was 4.35:1
on the fix panel, failing AA. Same green, rescued.

Two greys, not one. `--aiml-muted` `#6b7280` is **chrome only** — footer, slide
number. Anything a student must read uses `--aiml-muted-strong` `#4b5563`
(7.56:1).

`--aiml-rule` moves to `#b0bcc7`; `#d5dbe1` is 1.4:1 and simply absent on a
projector. Keep the old value as `--aiml-rule-faint` for table cell rules only.

**Red is overloaded.** It means broken, urgent, write-this-down, look-here — and
was also the colour of every inline `<code>`, so ten identifiers in a row read as
a warning stripe. Inline code is `--aiml-primary`; the chip background carries
the distinction.

Panel titles take their panel's semantic colour rather than grey — they are doing
semantic work anyway, and grey-on-tint failed AA in all four variants.

### 11.6 Plots must belong to the same document

The SVGs are authored at 677–927 pt and capped at 420–560 px on the slide, so
**they display at roughly 1:1 — the rcParams numbers are on-slide pixels.** A
`font.size: 13` tick label is 13 px against 30 px slide text.

**matplotlib cannot read `.woff2`.** Its FreeType raises `Can not load face
(unknown file format; error code 0x2)` — verified. Worse, anyone who wraps the
call defensively gets a silent revert to DejaVu with no error at all. And a
*variable* `.ttf` resolves to its default instance, which for Source Sans 3 is
ExtraLight.

So `assets/fonts/` carries both: woff2 variable fonts for the browser, and three
**static** cuts — `SourceSans3-{Regular,SemiBold,Bold}.ttf` — for matplotlib,
renamed so `font_manager` sees one family with three weights. Register those, and
the plots and the slides are in the same typeface:

```python
from matplotlib import font_manager
for f in ("SourceSans3-Regular.ttf", "SourceSans3-SemiBold.ttf",
          "SourceSans3-Bold.ttf"):
    font_manager.fontManager.addfont(ROOT / "assets/fonts" / f)

"font.family": "Source Sans 3",
"font.size": 17, "axes.titlesize": 19, "axes.labelsize": 17,
"xtick.labelsize": 15, "ytick.labelsize": 15, "legend.fontsize": 15,
"axes.titleweight": "normal",          # the slide's h2 is the title
"axes.titlecolor": "#4b5563",
"axes.titlelocation": "left",
"axes.spines.top": False, "axes.spines.right": False,
"axes.edgecolor": "#7b8794",           # 3.66:1, was 2.54:1
"grid.color": "#b0bcc7",               # 1.93:1, was 1.40:1 — invisible
"axes.axisbelow": True,                # gridlines under the bars, not through
```

Then shrink each `figsize` ~20%, and delete the per-call `fontsize=` arguments
that currently defeat the global scale.

**`width: 100%` next to `max-height` distorts the aspect ratio.** The two axes
are then set independently and the browser squashes the image into the box.
`.fig-wide` did exactly this, stretching seven figures horizontally by 10–48%
— a heatmap whose cells should be square rendered at 1.48 : 1, and the
letterforms in those figures were wider than the same typeface on the slide
beside them. Use `max-width` with `max-height`, never `width` with `max-height`.
To fill the full slide width, author at aspect ≤ 0.328.

**How big a plot's text ends up, in three steps.** This is not obvious and was
got wrong twice.

1. matplotlib writes SVG in **points**; the browser converts at 96/72, so 1pt
   becomes **1.33px**. PNG is rasterised at `dpi=160`, so 1pt becomes **2.22px**
   in the file — rasters are authored 1.67× larger than vectors for the same
   `fontsize`.
2. The slide then scales the image. `fig-wide` is `width: 100%` against a usable
   **1280px** (sections have no horizontal padding), so wide figures scale *up*
   and their authored width barely matters. What binds is `max-height`: 420px
   for `fig-wide`, 528px for `fig-tall`.
3. So a figure taller than about 0.44 of its width gets clamped by height, and
   all its text shrinks by that ratio.

**You do not have to do this arithmetic — `make_figures.py` does it and refuses
to ship a figure under the floor.** It reads each figure's intrinsic size, finds
which `.fig-*` class the decks use it with, applies the right px-per-point, and
raises if the smallest label would land below 15px. Current minimum across all
22 figures is 17.4px.

That check exists because the margin was an accident: the ~0.5× downscale of the
near-square rasters happens to cancel the 1.67× from `dpi=160`, and nothing said
so. Change the dpi, change the caps, or save a square figure as SVG instead of
PNG, and text drops under the floor silently.

Two ways to get this wrong, both of which happened:

- **Do not scale a raster's `fontsize` by the clamp ratio.** Multiplying
  `◆ Residuals` by ~1.7 would take its labels to ~31px — larger than the 30px
  slide body text.
- **Measure the vertical scale, not the horizontal one.** Text height follows
  `sy`. Reading `sx` off a stretched figure overstates it.

**The palette lives in two places.** `make_figures.py` carries its own copy of
the hex values. Any change to a `--aiml-*` token in `custom.css` must be mirrored
there in the same commit, and all figures regenerated — or the decks ship with
two greens.

**Panels of the same quantity share a y-axis.** A two-panel figure drawn at
different limits makes a smaller bar look taller, which is precisely the reading
error Lecture 2 teaches students to catch. Set the limit explicitly on both.

`jet` is perceptually non-monotonic and imports yellow/cyan/magenta from outside
the palette. Where a slide *prints* `cmap="jet"` for fidelity to the book, keep
the printed code and plot with `turbo`, and spend one sentence on why — a free
teachable moment in a course about not trusting defaults.

### 11.6a A diagram must carry its own typeface

Every `d-*.svg` names a font and, until late in the build, shipped no glyphs:

```svg
<svg ... font-family="'Source Sans 3','Source Sans Pro',...,sans-serif">
```

The decks embed them as `<img src="../assets/figures/d-kfold.svg">`, and **an
SVG inside an `<img>` is a separate document in secure static mode**. It cannot
see the page's `@font-face`, and it may not fetch anything external — not a
stylesheet, not a font file, not even same-origin. So it gets Source Sans 3 only
if the *machine* has Source Sans 3 installed. This one does not. A lecture
theatre's will not.

All forty diagrams were therefore rendering in Helvetica, beside matplotlib
figures rendering in Source Sans 3, on the same slide. matplotlib is immune
because `svg.fonttype` defaults to `path`: it ships outlines.
`l24-concentration.svg` embeds 365 of them; `d-course-arc.svg` embedded none.

**It also silently invalidated the measurement.** `check_diagrams.py` originally
inlined each file into a page that linked the stylesheet, so it measured in a
typeface the projector never showed — and Helvetica is 5–12% wider here, so
every "that label fits" was optimistic about the wrong font.

Both halves are fixed and both matter:

* `tools/embed_diagram_fonts.py` subsets the vendored variable font to exactly
  the characters each diagram draws, at the weights it uses, and inlines it as a
  base64 `@font-face`. About 10 KB a face, because a diagram uses forty
  characters rather than a hundred and twenty. `--check` fails the build on any
  diagram that carries none.
* `check_diagrams.py` now opens the diagram as the top-level document — no
  stylesheet, nothing injected. Verified equivalent to the `<img>` path by
  screenshotting one both ways: 0 differing pixels of 1,874,080.

Confirm a fix of this kind by measuring, not by looking. Embedded labels match
the `@font-face` widths to 0.08px; the fallback ran 5–12% wide, and gave "part
1" through "part 4" four different widths where Source Sans 3 gives one.

### 11.7 One header for every hand-drawn diagram

Seven diagrams used ten corner radii, six stroke widths and fourteen font sizes.
With 22 decks still to draw, fix it now. Start every `d-*.svg` with:

```xml
<!-- width and height are REQUIRED and must match the viewBox. Without them the
     figure renders at 0x0 through an <img> and the slide shows nothing — see
     the warning below this block. -->
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1080 420"
     width="1080" height="420"
     font-family="'Source Sans 3','Source Sans Pro',Helvetica,sans-serif">
  <style>
    .box   { fill:#fff; stroke:#0b3d62; stroke-width:2; }
    .box-r { fill:#fff; stroke:#c0392b; stroke-width:2; }
    .box-g { fill:#fff; stroke:#14663a; stroke-width:2; }
    .box-m { fill:#fff; stroke:#6c3483; stroke-width:2; }
    .t     { font-size:19px; fill:#16212b; }
    .t-sub { font-size:16px; fill:#4b5563; }
    .t-hd  { font-size:23px; font-weight:700; }
    .flow  { stroke-width:2.5; fill:none; }
  </style>
```

`rx="7"` everywhere; strokes 2 / 2.5 / 3 only; those three text classes only.

**A `viewBox` alone is not an intrinsic size.** This snippet originally omitted
`width` and `height`, and fourteen diagrams inherited the omission. Measured in
Chrome: an SVG loaded through `<img>` with only a `viewBox` reports
`naturalWidth` 300 — the CSS replaced-element default — and renders at **0×0**.
The slide shows nothing.

It survived because every check agreed with it. The file exists, so the
missing-figure check passed. And the text-floor check read the same wrong
intrinsic size, so it computed the floor against 300px of width and passed
trivially — a check passing for the wrong reason, which is worse than no check.
Both now reject an SVG with no `width`.

**Do not re-implement `.panel` inside an SVG.** Explanatory boxes belong in HTML
above or below the figure, or the same idea exists in two visual languages.

**Compose diacritics with `<tspan>`, not combining marks.** `θ̂` written as
U+03B8 + U+0302 renders with a detached, offset hat in every non-Mac fallback —
it did, four times, in `d-projection.svg`.

### 11.8 Restraint

- **At most one `.panel` per slide.** At 65 panels across 199 slides, a callout
  every third slide stops meaning "stop, this matters".
- **A panel is never smaller than the text it interrupts** (`font-size: 0.94em`
  floor). The most consequential sentence on the Assessment slide was set smaller
  than the routine bullets above it.
- **Every figure gets exactly one `.takeaway`** — one sentence saying what to
  look at.
- If a trailing note is not readable from row 20, it is a **speaker note**, not a
  `<p class="small muted">`. Roughly half of the 123 `.small` uses belong in
  `<aside class="notes">`.

### 11.9 Unverified — do not ship blind

`-webkit-text-stroke: 0.011em currentColor` on `.reveal .katex` was suggested to
thicken Computer Modern's hairlines, which are the first casualty of ambient
light. **Not tested on the hall projector.** Try it there before adopting it.

---

## 12. Before committing

```bash
python3 tools/make_figures.py       # regenerate every figure and figures.json
python3 tools/check_decks.py        # currency, weekdays, third-party notebooks,
                                    # missing figures, deck length
python3 tools/check_provenance.py   # every quantity on a slide is traceable
python3 tools/check_overflow.py     # nothing exceeds the 720px canvas
```

`check_provenance.py` is the one that enforces §4. It pulls every money amount
and thousands-separated integer out of the decks and requires each to be
reachable from `figures.json` — exactly, or at a rounding a lecturer would
plausibly write. Anything that is genuinely not a measurement (durations, marks,
chapter numbers, constants quoted from the book) goes in its `ALLOWED` table
**with the reason attached**. Adding to that table without a reason is how the
contract rots again.

Then look at the deck in a browser. The linter catches what is mechanical; it
cannot tell you a slide is ugly, overfull, or wrong.
