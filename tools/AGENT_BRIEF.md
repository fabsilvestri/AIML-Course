# Brief for a lecture-rebuild agent

You are rebuilding one or two lectures of *Applicazioni Informatiche del Machine
Learning* onto the 2026-09 design. Read these three files first, in order:

1. `REBUILD.md` — the resumable state, the decisions already fixed, and the
   per-lecture routine. **Do not relitigate anything in "Decisions already
   fixed".**
2. `LECTURES.md` — the plan. Your lecture's entry is the specification.
3. `AUTHORING.md` — §2 deck anatomy, §3 recurring devices, §4 notebooks.

Then read `slides/lecture-01.html`, `slides/lecture-02.html` and
`slides/lecture-03.html`, which are already converted and are the template.

---

## What you are changing

The course dropped its Build → Break → Fix pairing. Every lecture is now one
topic, self-contained: ~10 min context, ~20 min **the mathematics derived**,
~40 min the method with a worked example, ~15 min further ground, ~5 min the
notebook. **Nothing is wrong on purpose** — the old planted defects become
*failure conditions*, stated as properties of the method.

Cut on sight: planted defects · "commit a number in writing" · the peer
red-team · weak-prompt-versus-usable-prompt demonstrations · "Thread N"
numbering · `badge-build` / `badge-fix` / `badge-math` / `commit-slide`.

Use instead: `badge-lec` on the title slide, `.scope-exam` / `.scope-eng` /
`.scope-context` to mark examinable scope, `.panel-when` for a failure
condition, `.derivation` for a derivation, `.notebook-slide` for the last slide.

---

## Files you may write

    slides/lecture-NN.html          your lecture's deck
    tools/notebooks/lecture_NN.py   your lecture's notebook generator

Renumbering: your source is an OLD lecture number. `git mv` both files into
their new number before editing.

## Files you must NOT touch

    tools/make_site.py   REBUILD.md   index.html   tools/make_nb_index.py
    any lecture that is not yours   assets/figures/*   figures.json

The integrator updates those at merge. Editing them concurrently loses writes.

---

## Method

`tools/deckkit.py` does slide-level surgery — `python3 tools/deckkit.py
slides/lecture-NN.html` lists the slides with their indices. Rebuilding a deck
is selecting, reordering and rewriting a list, not retyping HTML. Most slides
survive; the framing around them does not.

Deck target **70–90 slides**, counted against the 90-minute clock. Above 100 it
will not finish.

Notebook: every code cell preceded by a `prompt()` box with `input` / `output` /
`constraint` / `check`, plus `**{"try": ...}` — one modification and what should
happen to the output. `check` must have an answer workable on paper *before*
running. The kwargs `left_open`, `student` and `catch` are retired; do not use
them.

Regenerate with `python3 tools/make_notebooks.py --only NN` (never without
`--only`, which would rewrite all 24).

---

## Three classes of error that are invisible to grep

Every lecture converted so far has shipped at least one. Look for all three.

1. **Stale claims about how the course runs.** Lecture 1 shipped four, none of
   which contained the word Build or Fix: "you will not type most of the code in
   this course", "a loop, run out loud, every lecture", "rule of the room", and
   four rules written for submitted work when nothing is submitted. Ask of every
   slide: *does this describe something that still happens?*
2. **Lecture-number references broken by renumbering.** Check every one against
   `LECTURES.md`. Ridge moved from 6 to 5, PyTorch from 12 to 10, mAP from 18 to
   14, the estimator API from Lecture 1 to Lecture 2.
3. **Deck and notebook computing the same quantity differently.** Lecture 2's
   notebook searched `cv=5` where every slide figure used `KFold(10)`, and
   sliced errors on the training set where the deck says plainly it does so on
   the test set. Both were internally consistent; they disagreed with each
   other. **Any quantity that appears on a slide and in the notebook must be
   computed the same way, on the same rows, with the same seed.**

The slide figures come from `assets/figures/figures.json`, which is the
authority. If a slide and the script disagree, the script is right.

**Except durations.** A wall-clock second is a property of a machine, so no
amount of matching makes a slide and a notebook agree on one. See AUTHORING.md
§3.2a: one significant figure, labelled as one machine's measurement, never in
a column that invites an unstable comparison, and the notebook says the
reader's number will differ. Cap BLAS threads before importing numpy if your
notebook reports any timing at all.

---

## Deliverable

Report back, in prose, no more than 30 lines:

- deck: slide count, what you kept, what you cut, what you wrote new
- notebook: cell count, whether it executes, wall-clock
- **every quantity your notebook prints that also appears on a slide**, with
  both values, so the integrator can diff them
- every stale claim you found, by class (1/2/3 above)
- anything you could not resolve

Do not update the site, `REBUILD.md`, or push. The integrator does that.
