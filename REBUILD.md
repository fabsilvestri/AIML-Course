# Rebuild status

Resumable progress log for the redesign. **If you are picking this up cold —
a new session, a restarted machine — read this file first, then `LECTURES.md`
(the plan) and `AUTHORING.md` (how to build one lecture).**

Last updated: 2026-09-05 (three rounds of colleague review; ~190 defects
found and repaired, and eight new checks so the classes cannot return) ·
Term starts: ~2026-09-22

---

## THE GOAL

**All 24 lectures rebuilt, and `tools/check_consistency.py` clean on every one
of them.** Set by the lecturer, 2026-09-01. Nothing is finished until both
halves hold: a lecture whose deck is beautiful and whose notebook does not
reproduce its figures is not done.

Definition of done, per lecture, all of it:

1. deck on the new design, 70-105 slides, no overflow
2. notebook executes cold on CPU and prints every figure the deck states
3. `check_consistency.py N` clean
4. `check_decks`, `check_notebooks`, `check_provenance` clean
5. `index.html` shows it published, `REBUILD.md` row says done
6. committed and pushed

## Where this stands

**Done.** All 24 decks and all 24 notebooks are on the new design, all 24 are
published on the site, and `check_consistency.py` is clean on every one of them.
Lectures 19-22 — Part V, search and recommendation — were written from nothing,
with three new figure scripts behind them.

`tools/check_names.py` was added along the way and belongs in the fast set: it
joins a notebook's code cells and asks pyflakes for undefined names, which found
twelve real defects in six notebooks in 0.3 seconds — the class that splitting
and reordering older modules keeps producing, and that compiling each cell in
isolation cannot see.

## Status snapshot — 2026-09-03

Everything below was true and verified when it was written, and is pushed to
`origin/main`.

| | |
|---|---|
| Decks | 24 / 24 on the new design, 70–105 slides each |
| Notebooks | 24 / 24, 540 code cells, every one behind a specification box, sections numbered uniquely and in order |
| Runs on CPU | all 24, cold, from a clean kernel |
| `check_consistency` | **24 / 24 clean** — **859** deck figures verified against notebook output. The count has moved twice for the same reason and both are worth knowing: 826 was over 22 lectures (21 and 22 reported green while checking nothing), and the later 932 included lecture 24, which was *also* reporting green while comparing zero figures — every number it quotes is rounder than the four-significant-digit bar. Fixed in round 3 by having notebook 24 print R@1 to two decimals; the guard now fails when a deck states none of its own figures, not only when a namespace matches no key |
| `check_all` (8 fast checks) | clean |
| Browser checks | no slide over the canvas on any of 25 pages; 39 diagrams clean |
| Published on the site | 24 / 24 |
| `try` fields | **540 / 540** — every prompt box, enforced by `check_notebooks §4.1a` |
| Part V extended notes | 4 / 4 written, built and linked from `index.html` |

Two documented exception lists, both printed in the checker's output rather than
skipped silently — these are the first thing to be sceptical of:
`SCALE_ONLY` (3 figures, Lecture 18 runs at a scale its CPU notebook does not)
and `CROSS_LECTURE` (5 figures a deck quotes from another lecture's experiment).

### The open items — none

**Closed 2026-09-03.** The `try` fields are done: 538 of 538 boxes, every
lecture. `check_notebooks.py` now has a hard rule (`§4.1a`) that fails on a
prompt box without one, so the omission cannot recur; `AUTHORING.md §4.1a`
specifies the field and records why the rule is blocking rather than advisory.

The census that found the gap is still the right way to check it, and now
answers 0:

```sh
python3 - <<'EOF'
import json, pathlib
tot = twt = 0
for p in sorted(pathlib.Path('notebooks').glob('lecture-*.ipynb')):
    md = [''.join(c['source']) for c in json.load(open(p))['cells']
          if c['cell_type'] == 'markdown']
    boxes = [m for m in md if '**Prompt' in m]
    t = sum('**try**' in m for m in boxes)
    tot += len(boxes); twt += t
print(f"{tot} boxes, {twt} with try, {tot - twt} missing")   # 538 538 0
EOF
```

### Part V extended notes — `notes/`

Lectures 19–22 are taught from notes rather than from Géron, so for those four
**the notes are the primary source**, not a supplement. They now exist as
LaTeX, in `notes/lecture-{19,20,21,22}.tex` on a shared `preamble.tex`, and
build to PDFs of 18, 17, 16 and 16 pages:

```sh
make -C notes            # rebuilds any PDF whose .tex or preamble is newer
make -C notes clean      # intermediates only; the PDFs stay
```

The PDFs are **tracked**, because the site links to them; the `.aux/.log/
.out/.toc` are gitignored. `check_decks.py` fails if `index.html` links a notes
PDF that is not on disk.

Every figure in the notes is one the corresponding notebook prints, and that is
**verified rather than asserted**: `tools/check_notes.py` is `check_consistency`
pointed at `notes/*.tex`, sharing its machinery so there is one rule about
numbers in this repo instead of two. It anchors on `figures.json` — a number in
the notes is considered only when it is quoting one of that lecture's
measurements — then requires the notebook to have printed it. It strips
`verbatim` blocks and mathematics first, for the same reason the deck check
strips `<pre>`: neither is a claim about a measurement.

```sh
python3 tools/check_notes.py          # all four
python3 tools/check_notes.py 21       # one of them
```

Verified 2026-09-03: **216 stated figures across the four sets of notes — 86,
68, 43 and 19 — every one of them printed by its own notebook.** Lecture 20's
notebook takes about ten minutes to execute cold; the other three are quick.

On the site they appear in two places: a third button on each of the four
Part V lecture cards (`btn-notes`, emitted by `make_site.py` for any lecture
whose chapter field is empty), and a table in *Textbook and scope*.

## The three review rounds — 2026-09-04 to 05

Five colleagues read the whole course three times: mathematics, code, teaching,
cross-artefact consistency, and assessment. Roughly 190 defects were found and
repaired. Read the reports under the session scratchpad if they survive; what
follows is what a future session actually needs.

**The counts fell but the kind changed, and that is the real result.** Round 1
found original defects. Round 2 found mostly round 1's repair residue. Round 3
found almost nothing in the course prose and almost everything in round 2's
repairs and in the checks themselves. A repair is a change, and a change is a
defect until something verifies it.

**Four classes were invisible to every check that existed, and each now has one:**

* *A repair applied to generated output and not its generator.* The Part V
  heading was fixed in `index.html` while `make_site.py` still wrote the old
  one; `figures_app08.py` still emitted `thread 6` into an SVG that had been
  cleaned by hand. Always fix the generator, then regenerate.
* *Text inside a figure.* The closing course-arc diagram carried the whole
  abandoned Build/Break-Fix pairing, and two more diagrams sent students to
  pre-renumbering lectures. `check_decks` now rejects a figure that names a
  lecture later than the deck showing it.
* *Text rendered as paths.* matplotlib draws its labels as glyph outlines, so
  160 of the 199 figures contain no readable text at all — three stale lecture
  numbers sat in `set_title()` calls, visible to students and invisible to
  every grep. No plot label may name a lecture now.
* *A check that passes by examining nothing.* Twice: a namespace matching no
  key, and a deck stating none of its own figures. Both guarded, both verified
  by planting the defect.

**Every guard is probe-tested.** Each was confirmed to reject the exact defect
it was written for by planting it and watching the check fail. Two guards
shipped in a state where they passed their own defect — the section-numbering
check ordered only the digits, so `15 -> 12b -> 16` slipped through, and the
hardware check matched "needs a GPU" but not "everything runs on a GPU". Write
the probe before trusting the guard.

**Do not use `git checkout --` to undo a probe** on a file with uncommitted
work. It cost three files of edits in this session. Save and restore the bytes
in the probe itself.

## Where the two 2026-09-03 jobs stand

### Job 2 — exam-style exercises: **DONE 2026-09-04**

Five questions at the end of every deck, 1 to 24, and the order on a deck is
**this lecture's new exercises first, last lecture's solutions after them**.
The solutions are not part of lecture N: they answer the set given out at the
end of lecture N-1, and they sit at the very end where a student revising will
find them. Putting them before the new exercises reads as though the lecture
were about them, which it is not. Lecture 24 carries its own five, then 23's
solutions, then its own, because there is no lecture 25. Generated by `tools/make_exercises.py` from one table, injected
between `<!-- BEGIN EXERCISES -->` markers, and re-runnable:

```sh
python3 tools/make_exercises.py          # every deck
python3 tools/make_exercises.py 7 8      # just these
```

It extends a pattern the decks already had: Lecture 1 set a specimen Part B
question whose last two parts cannot be answered until cross-validation exists,
and Lecture 2 answers them.

Four things learned the hard way, all now guarded:

* `\text{...}` in a non-raw Python string **is a tab**. Thirteen reached the
  slides before `check_decks` caught the KaTeX breakage.
* Raw-ifying the module to protect the LaTeX also raw-ified the layout code, so
  every `\n` printed as a literal backslash-n. The tokenizer pass is bounded to
  the exercise data now — and it must be a tokenizer, because a regex
  desynchronises on the literals that were already raw.
* Truncating a question to a reminder cut through inline maths, leaving an odd
  number of `$`. `short()` backs up past an unclosed delimiter.
* A bare `<` inside `$...$` starts an HTML tag (AUTHORING §5.3a), and a named
  weekday breaks §5.5. Both were caught by `check_decks`, not by reading.

#### The correctness audit — all 120 read, 2026-09-04

Every question and every solution was read against its own deck. Six defects,
all now fixed, and each is a class worth knowing about:

| Where | Defect | Class |
|---|---|---|
| L17 Q1 | quoted 709, the float64 overflow point, which is on no deck | figure the student never saw |
| L18 Q2 | premise (the double-softmax floor) lives in the notebook only | question unanswerable from the slides |
| L04 Q2 | said all six rank deficiencies come from the encoder; five do, and the sixth is the engineered `FamilySize` | answer thinner than the deck |
| L17 Q5 | stated 40% as *the* type-OOV rate; it is the **floor**, and at the deck's 20,000-word vocabulary the rate is 77% | a real number in the wrong role |
| L18 Q1 | "the variance grows with $d_k$" where the deck derives $\operatorname{Var}(s) = d_k$ exactly | vaguer than what was asked for |
| L24 Q5 | listed five rules that were **not** the deck's five | confidently wrong, and the worst of them |

The last one is the reason a mechanical check cannot close this job: every one
of those five rules is a true, on-topic, in-scope sentence about this course.
`check_exercises` had nothing to object to. Only reading the slide catches it.

What *is* mechanical now: `tools/exercise_claims_test.py` recomputes every
number an answer asserts — the parameter counts, the AP and NDCG sums, the
harmonic mean behind Glorot, the JL bound at $n = 400$, the $\rho < 1/2$
crossover, `bootstrap=False` with `oob_score=True` raising. It and
`try_claims_test.py` are both in `check_all.py` now; neither was, which is how
a check nobody runs becomes a check nobody wrote.

Prefer quadrature to sampling in that file. The ReLU half-moment check was
Monte-Carlo first and failed at 0.5034 against 0.5 — noise, not a wrong answer,
and a test that cries wolf gets deleted rather than fixed.

### The examination rule changed — 2026-09-04

The old rule was written plus a **compulsory** oral, 50/50, each passed
independently at 18/30. The lecturer's constraint is the number of orals: at
twenty-five minutes each, a full cohort is several working days, every session.

The rule now:

| | |
|---|---|
| Written | out of 30, pass at 18, **capped at 27 on its own** |
| Oral | **optional** — three questions drawn in front of the candidate from the 120 published exercises, seven to ten minutes |
| Arithmetic | the oral moves the written mark by **at most ±3**; it cannot turn a passing written into a fail |
| Timing | decided after the written mark is seen, and **binding once registered** |

Three things make it work, and all three are load-bearing:

* **The cap creates the only reason to sit it.** 28, 29 and 30 exist nowhere
  else, so the candidates who turn up are the ones with something to show.
* **The downside is real.** A free option is one everybody exercises, which is
  the queue the change exists to remove. Three marks down is what makes it a
  decision rather than a lottery ticket.
* **The bank is published.** All 120 questions and their solutions are already
  on the decks and in `notes/exercises.pdf`, so a risky oral is still a fair
  one. This is why the exercise audit had to come first: the bank became the
  examination the moment it became the oral.

Note the thing that limits the deterrent: a student who dislikes the outcome
can normally refuse the grade and resit. The cost that actually bites is the
lost appello, not the lost marks. Worth confirming against the regolamento
before the syllabus is filed — teaching starts around 2026-09-22.

Stated in exactly three places (site, deck 1, deck 24), specified in
AUTHORING §6.1, and enforced by `tools/check_assessment.py`, which is in
`check_all.py`. Its three arms were each verified to fire by mutating a file
and watching it fail — including the one that catches two pages disagreeing on
a number rather than merely omitting it.

Two edits elsewhere follow from the same change: Lectures 19 and 21 claimed
some question was "the most-asked oral question", which cannot be true of a
bank the students can read. Both now point at the written paper.

**Do not use `git checkout --` to undo a probe on a file with uncommitted
work.** Verifying the new check that way cost three files' worth of edits in
this session and they had to be retyped. Save and restore the bytes in the
probe itself.

### Job 1 — the try-field audit: **32 claims verified, six found false**

`tools/try_claims_test.py` executes every claim that can be reproduced
standalone; it is at 43/43 and is where a newly verified claim goes.
`tools/try_audit.py` extracts and triages all 539, and
`--check-numbers` audits every stated figure against the notebook's own output
using `check_consistency`'s cache — 67 figures checked, and the three it cannot match are legitimately derived rather than printed.

**Nine false claims found so far, all corrected in the notebook rather than in
the test:**

| lecture | the claim | what is actually true |
|---|---|---|
| 8 | "the assert fails on component 1 and the rest still agree" | all five components disagree |
| 8 | "the message names the constraint you just violated" | it names the wrong one |
| 20 | "the assert reproducing Lecture 19's NDCG@10 fires" | there was no such assert; one was added |
| 3 | "delete the cast … nothing raises" | the cell's own dtype assert fires |
| 6 | "index `COVER_NAMES[k]` … nothing raises" | class 7 runs off a seven-element list |
| 19 (exercise) | worked NDCG@10 of 0.8455 | 0.8396 |
| 15 | "`end = idx + w - 1`&nbsp;… the target is now the last day of the window, every MAE collapses" | it shortens the window to 55 days; the target still follows it, and nothing leaks |
| 19 | "keep the score == 0 rows as relevant … every metric rises" | SciFact's test qrels contain **no** zero rows: 339 judgements, all positive. The exercise was a no-op, and the constraint above it made the same claim |
| 19 | "change the discount to log2(i) and watch rank 1 become **infinite**; change the AP denominator to the hit count and watch **every score rise**" | it raises ZeroDivisionError, and nothing rises: the loop runs to the end of the corpus, so the hit count always reaches \|R\|. The two denominators differ only on a truncated ranking |

Nine of forty-three is a rate worth taking seriously, and it is why the audit
continues rather than being declared finished.

**To resume:** work the `--assert` class first, then `--number`, adding each
verified claim to `try_claims_test.py` rather than checking it once in a shell.

## How to work on it now

The rebuild is finished, so this file stops being a plan and becomes a manual.
Before changing anything:

1. `python3 tools/check_all.py` — five checks, a few seconds. Run it after every
   edit, not at the end.
2. `python3 tools/check_names.py` if you touched a notebook module. It answers
   in 0.3 seconds the question an execution answers in half an hour.
3. `python3 tools/check_consistency.py N` if you touched a deck or a notebook.
   It executes the notebook, cached by content hash, so only what you changed
   is slow. **Fix what it reports by adding the missing computation to the
   notebook**, not by changing the slide — the slide's number came from a real
   experiment, and a student who cannot reproduce it has been told to take it
   on trust.
4. `python3 tools/check_all.py --full` before a release: the browser checks for
   overflow and diagram labels, and the consistency sweep over all 24.
5. Read `AUTHORING.md` §2 (deck anatomy) and §4 (notebooks) before writing.

**A note on diagnosing a slow check.** `nbconvert`'s own process sits at 0% CPU
while the kernel it spawned does the work. Do not read that as a deadlock, as I
did: check the child, or watch the output file grow.

**A note on this machine's filesystem, which cost hours.** Reads under
`notebooks/datasets/` are intercepted at roughly **one second per file** — `cat`
alone on the 8,189 Flowers102 JPEGs takes many minutes. Lecture 12's execution
spent 31 minutes in blocking `read()` having used 17 seconds of CPU, and the
`sample` output showed 2,274 of 2,285 samples inside `read()` rather than inside
the JPEG decoder.

It is not the notebooks and it is not OpenMP, which is what I assumed first. The
symptom is a process at 0% CPU that never finishes, on a dataset directory that
`ls` says is fully present.

The fix is to **re-extract from the local archive**, which writes fresh files
that read at full speed:

    mv notebooks/datasets/aclImdb notebooks/datasets/_evicted-aclImdb
    python3 -c "import tarfile; tarfile.open('notebooks/datasets/aclImdb_v1.tar.gz').extractall('notebooks/datasets', filter='data')"

IMDb went from ~1 s/file to 20 files in 0.00 s, after a 15-second extraction.
The same works for `flowers-102/102flowers.tgz`. Move rather than delete: the
old directory is gitignored and regenerable, but it is the user's disk.

None of this affects a student. It affects only how long verification takes on
this machine, and it is worth knowing before spending an hour on the wrong
hypothesis.

Never leave this file stale. A row that says `wip` with no commit behind it is
worse than no row.

### STANDING INSTRUCTION — slide/notebook consistency is not optional

From the lecturer, 2026-09-01, emphatically: **at the end of the rebuild, every
slide and its notebook must agree, and a verification pass must be run and
re-run until everything checks out.**

This is the step that has found a real defect in every single lecture converted
so far, and no other check catches any of them:

| lecture | what only the diff found |
|---|---|
| 1 | deck and notebook counted the price stripes on different row sets |
| 2 | notebook searched `cv=5` where every slide figure used `KFold(10)`; error slices on the training half where the deck says test |
| 3 | deck quoted a 90.39% recall the notebook never printed |
| 8 | Johnson-Lindenstrauss measured on **unsquared** distances — the deck right, the notebook wrong by a factor of two; and a k-grid that did not contain the k the deck's headline names |

So it is now a tool, `tools/check_consistency.py`, not a habit. Run it, fix,
run it again, until it is clean. See §7 of AUTHORING.md.

**Verification loop, state at last checkpoint:**

| lecture | consistency | deck | notebook |
|---|---|---|---|
| **1–24** | **clean** | **done** | **done** |

All twenty-four. Every figure stated on a slide is a number its own notebook
prints, with two kinds of documented exception, both listed in the checker's
output rather than skipped in silence:

- `SCALE_ONLY` — three figures on Lecture 18, whose deck fine-tunes on 20,000
  reviews and clusters 12,500 while the notebook uses 2,000 of each so it
  finishes on a CPU. Its header names both numbers.
- `CROSS_LECTURE` — five figures a deck quotes from another lecture's
  experiment, each entry naming the lecture whose notebook reproduces it.

**Lectures 4-8 passed the consistency check on the first run, with no
intervention.** All five were agent-drafted against `tools/AGENT_BRIEF.md`.
Lectures 1-3, which I wrote before the brief existed, needed 33 fixes between
them. The brief is doing the work: it names the three invisible error classes
and tells the drafter to run the checker before reporting.

The fixes are almost never "correct a wrong number". They are "the notebook
never computed this at all" — a slide asserting something no cell produces. So
the repair is to add the computation, which is also the right thing
pedagogically: a student running the notebook can now check the claim in front
of them. Lecture 2 gained six cells this way (all four training RMSEs, both
paired comparisons, the three imputation strategies, the bootstrap half-width,
the ten worst predictions, the capped-districts trap and the
absolute-versus-relative arms).

**`NAMESPACES` must come from the Source column, never from inference.** It is
tempting to work out which figures.json prefix a deck belongs to by seeing which
one it matches most. Lecture 7 (CoverType) scores 15 hits against `l06_*`, the
Titanic keys, because accuracies and class shares both live in [0, 1] and
collide at four significant figures. An inferred map blesses the wrong namespace
and then passes.

**A trap in the loop.** Some deck numbers are there to show a procedure that is
*wrong* — Lecture 2 quotes a test-set pair to say it decides nothing. Computing
those in the notebook models the bad practice. Compute them, but label the cell
with what the deck says about them; do not quietly drop the number from either
side.

### The per-lecture routine — every lecture, without being asked

Standing instruction from the lecturer. All five steps, in order, before
starting the next lecture:

1. **Build** the deck and the notebook.
2. **Check** — `check_consistency.py` FIRST (it executes the notebook and
   diffs every figure the deck states against what the notebook prints; this is
   the check that finds things), then `check_decks.py`, `check_overflow.py`,
   `check_notebooks.py`. Add the lecture to `NAMESPACES` in
   `check_consistency.py` or it is skipped rather than checked.
3. **Update `index.html`** so the site matches what the lecture now says. The
   site is student-facing: a lecture that is done and a page that still
   describes the old one is worse than neither.
4. **Update this file** — the lecture's row, and any new debt discovered.
5. **Commit, push to `main`, and print the status** — what is done, what is
   next, and anything that needs a decision.

Steps 3 and 5 are the ones easily forgotten. They are not optional.

### Three classes of stale claim, all invisible to grep

Auditing Lecture 1 and the site turned up fifteen false statements. Only four
mentioned the old structure. The other eleven were:

1. **Lecture-number references broken by renumbering.** "again in Lecture 6 when
   ridge repairs it" (ridge is L5), "Lecture 8. This single calculation explains
   bagging" (L7), "PyTorch ... taught in Lecture 12" (L10), "taught from scratch
   in Lecture 1" of the estimator API (L2, and L1 now fits nothing at all).
   **Check every lecture number against LECTURES.md mechanically** — the audit
   loop that does this is worth rewriting each time.
2. **Counts that moved.** "each of the twelve threads is a derivation" — there
   are eighteen.
3. **Slide and notebook computing the same quantity on different rows.** The
   deck counted the price stripes on the training split; the rebuilt notebook
   counted them on all 20,640 rows. Both were internally consistent, both passed
   check_provenance, and they disagreed: 62 against 79 districts at $350,000.
   **Whenever a notebook and its deck report the same quantity, run the notebook
   and diff the numbers against the slide.**

### Tools that look orphaned and are not

`compress_diagram.py`, `trim_diagram.py`, `fix_label_clearance.py`,
`embed_diagram_fonts.py` and every `figures_appNN.py` are **run by hand**, not
imported. A "which module imports this?" sweep reports all of them as dead. They
are not. Each has a `__main__` and a usage line in its docstring; regenerating
one lecture's figures means running its `figures_appNN.py` directly.

`figures.json` is shared: `make_figures.py` **merges** into it rather than
overwriting, and raises on a key collision. A script that writes it wholesale
silently deletes several hundred values belonging to other lectures.

### Regenerating a notebook is idempotent

`make_notebooks.py` reuses the cell ids already on disk wherever the cell
sequence is unchanged (`_keep_cell_ids`). Before that, nbformat minted a fresh
random id per cell per build, so regenerating an unchanged notebook still
produced a diff touching every cell — which buried the one line that had
actually changed. If you see id-only churn again, that helper has regressed.

### Parallel drafting (in progress, 2026-09-01)

Three agents are drafting the CPU block in parallel, working from
`tools/AGENT_BRIEF.md`. Units are **coupled pairs**, not single lectures: the
Titanic pair cross-references itself 11 and 9 times, CoverType 6 and 5, so
splitting a pair means each half writes references to content it cannot see.

| agent | drafts | from | status |
|---|---|---|---|
| L4-L5-Titanic | L4, L5 | old 05, 06 | drafted |
| L6-L7-CoverType | L6, L7 | old 07, 08 | drafted |
| L8-Olivetti | L8 (merge) | old 09 + 10 | **integrated**, old 9/10 deleted |
| L9-L10-Networks | L9, L10 | old 11, 12 | running |
| L11-DeepTraining | L11 (merge) | old 13 + 14 | running, at `lecture-11-NEW` |

| L12-L13-Vision | L12, L13 | old 15, 16 | **died at the usage limit** |
| L14-Detection | L14 (merge) | old 17 + 18 | **died at the usage limit**, nothing written |

### The 2026-09-01 usage-limit crash — state left behind

All five running agents failed at once on a shared session limit. What survived,
verified against `git ls-files` rather than trusted:

* **Lectures 4, 5, 6, 7, 8 are fully converted and committed.** No old badges,
  no commit slides, `badge-lec` present, titles correct. L4 also passes
  `check_consistency`.
* **Lectures 9, 10 and 12 were RENAMED but NOT converted.** `git mv` had run —
  old 11 → 09, old 12 → 10, old 15 → 12 — so the modules and decks sit at their
  new numbers holding their old content and old titles. The renames are the
  fiddly part and they are done; the conversion is not started.
* **Lectures 11, 13, 14 were not begun.** Old 13, 14, 16, 17, 18 are untouched
  at their old numbers.
* No `-NEW` draft was left behind by any of the three merge agents.

Resuming from here means: convert 9, 10 and 12 in place (no rename needed), then
carry on. Do not re-run the renames — they are committed.

Still unassigned: {L15, L16} from old 19, 20 · {L17, L18} from old 21, 22 ·
**{L19-L22} from nothing** · {L23, L24} from old 23, 24.

**Part V (L19-L22) needs figures that do not exist.** Every other lecture reads
`figures.json`; these four must ADD to it, and `make_figures.py` merges rather
than overwrites and raises on a key collision. Two agents writing it at once
would corrupt it, so the IR and RecSys agents must be sequenced, not run in
parallel — or write their facts and let the integrator merge.

**The collision pattern, every time.** A merge target's number is still occupied
by the old lecture that another agent is converting. Build at `-NEW` and let the
integrator rename once the slot frees. That is how L8 was done and how L11 is
being done.

**L8 builds at a temporary path on purpose:** `slides/lecture-08.html` still
holds old lecture 8 until the CoverType agent moves it to `lecture-07.html`.
The integrator renames L8 into place afterwards.

Agents draft deck and notebook only. They do not touch `make_site.py`,
`REBUILD.md`, `index.html` or `make_nb_index.py`, and they do not commit. The
integrator runs the full routine on every draft — execute cold, diff every
printed figure against `figures.json`, audit lecture numbers against
`LECTURES.md`, check overflow — before it lands. Realistic gain is ~2-2.5x, not
3x: notebook execution is CPU-bound on 16 shared cores and does not
parallelise, only the authoring does.

If this file is being read after a crash and the table above still says "in
progress", check `git status` for drafted-but-uncommitted decks and modules.

### Renumbering: the source modules are keyed by OLD lecture numbers

`tools/notebooks/lecture_NN.py` and `slides/lecture-NN.html` still carry the old
numbering for everything not yet rebuilt. The "Source" column in the table below
is the authority on which old lecture feeds which new one. When a new lecture
consumes an old module, **`git mv` the module into place or delete it** — do not
leave two files claiming the same number.

Consumed so far: old lecture 4 (*It never fires*) is merged into new Lecture 3.
Its generator `lecture_04.py`, its notebook `notebooks/lecture-04.ipynb` and its
deck `slides/lecture-04.html` are all deleted — an unlinked file still answers a
guessed URL, and all three would have served the old course's content. New L4
is built from old L5, so the deck count is 24 until then.

### Reading a deck for stale claims

Converting a deck is **not** a grep for "Build" and "Fix". Lecture 1 shipped
four false claims that contained neither word: "you will not type most of the
code in this course", "a loop, run out loud, every lecture", "rule of the room",
and four rules written for submitted work when nothing is submitted. Read every
slide asking *does this describe something that still happens?*

---

## Decisions already fixed

Settled with the lecturer; do not relitigate them without asking.

- **Structure:** one topic per lecture, book order, each lecture self-contained.
  No Build/Fix pairing, no planted defects, no "commit a number", no red-team.
- **Nothing is wrong on purpose.** Old planted defects become *failure
  conditions* slides — taught as properties of the method.
- **Lecture shape:** ~75 min slides, last ~15 min touring the notebook.
  Notebooks are for self-study; no live coding in class.
- **Notebooks** ship complete and correct, heavily commented, every code cell
  preceded by the specification that would generate it.
- **Mathematics** lives inside the lecture whose method rests on it — eighteen
  derivations, not a separate numbered device.
- **Part V** (L19–22, IR and RecSys) is taught from lecture notes and is
  examinable. Datasets: **SciFact (BEIR)** for retrieval, **MovieLens** for
  recommendation. *Confirmed by the lecturer.*
- **Assessment:** written + oral, 50/50, each passed independently. The paper
  becomes A mathematics 40% · B method choice applied to a scenario 35% ·
  C reading results 25%.
- **Slides:** reveal.js 5.2.1 kept, vendored and offline. Theme refreshed, not
  replaced. Lecture 1 is the template.
- **Palette: unchanged.** Confirmed by the lecturer — deep blue, brick red,
  green, purple, tuned for projector contrast. Effort goes into consistency
  across the 24 decks, not repainting.
- **Prompt boxes carry a fifth field, `try`** — one modification and what should
  happen to the output — set below a rule, because it addresses the reader
  rather than the assistant. *Confirmed by the lecturer.*
- **Derivations: one step per slide**, with the reason beside the step. Slow and
  unmissable; the normal equation is ~8 slides. *Confirmed by the lecturer.*
- **Every notebook must run on CPU.** L11-L24 were written for a GPU, which
  means they cannot be executed — and therefore cannot be number-diffed — by
  whoever is building them. Cut epochs, subsample, use smaller backbones until
  each runs on Colab's free CPU in a few minutes. *Confirmed by the lecturer.*
  Consequences, all of which are part of the job and not optional:
  its figures must be regenerated at the smaller scale (`tools/figures_app06`,
  `07`, `08`, `09`, `10`, `11`, `12` import torch), the slide numbers change to
  match, and `figures.json` is rewritten. A lecture is not done until the
  notebook has been executed and every figure it prints traces to
  `figures.json`.
- **Order of work: teaching order, each lecture finished properly** before the
  next is started. The tail may still be moving when term begins; nothing that
  is taught is half-built. *Confirmed by the lecturer.*

---

## Status

Legend: `done` · `wip` · `todo`

### Infrastructure

| Item | Status | Notes |
|---|---|---|
| `LECTURES.md` — the plan | **done** | 379df6c |
| `AUTHORING.md` — the spec | **done** | 379df6c |
| `README.md` | **done** | 379df6c |
| Repo cleanup (prompts toolchain, TRICKS, GUIDELINES, caches) | **done** | 379df6c |
| `assets/css/custom.css` — theme refresh | **done** | additive: `.badge-lec`, `.scope-*`, `.derivation`, `.panel-when`, `.notebook-slide`. Old `.badge-build/-fix/-fail` and `.commit` kept defined until the last deck is converted |
| `index.html` — site rebuild | **done** | now generated by `tools/make_site.py` from a table kept in step with LECTURES.md; unbuilt lectures show *In preparation* |
| `tools/make_site.py` — generates the site's lecture and derivation lists | **done** | flip a lecture's `published` flag to publish it |
| `tools/make_nb_index.py` — generates the 24-entry notebook index in 14 decks | **done** | titles come from make_site.py, so site and decks cannot disagree |
| `tools/make_notebooks.py` / `_prompt.py` — retargeted to AUTHORING §4 | **done** | three-line annotation dropped everywhere; `COLAB_AUTHORED` emptied so L19 generates like the rest |
| `tools/check_notebooks.py` — rules retargeted to AUTHORING §4 | **done** | passes on all 24 notebooks |
| `tools/check_decks.py` — site check rewritten | **done** | was "every lecture must be linked", which is false mid-rebuild; now checks each lecture is on the page, that a linked one has its files, and that nothing is marked *In preparation* while a converted deck exists |
| `tools/make_figures.py` — figures for reassigned lectures | **accepted** | figure filenames keep the old numbering. Renaming would touch every slide that cites one, for no gain to a student; the mapping lives in `check_consistency.NAMESPACES` |
| Part V figures | **done** | `figures_ir.py` (L19), `figures_dense.py` (L20), `figures_recsys.py` (L21–22); keys `l19_`, `l20_`, `rec21_`, `rec22_` |

### Lectures

Deck = `slides/lecture-NN.html`, Notebook = `notebooks/lecture-NN.ipynb`.
"Source" names the old lecture whose material is reused.

| # | Topic | Ch | Dataset | Source | Deck | Notebook |
|---|---|---|---|---|---|---|
| 1 | What ML is, and how we will work | 1–2 | housing | old L1 | **done** | **done** |

*Lectures 1-3 are complete through all five steps of the routine, site included.*
| 2 | The end-to-end project | 2 | housing | old L1+L2 | **done** | **done** |
| 3 | Classification and its metrics | 3 | MNIST | old L3+L4 | **done** | **done** |
| 4 | Training models | 4 | Titanic | old L5 | **done** | **done** |
| 5 | Regularisation and bias–variance | 4 | Titanic | old L6 | **done** | **done** |
| 6 | Decision trees | 5 | CoverType | old L7 | **done** | **done** |
| 7 | Ensembles and random forests | 6 | CoverType | old L8 | **done** | **done** |
| 8 | Dimensionality reduction and unsupervised | 7–8 | Olivetti | old L9+L10 | **done** | **done** |
| 9 | Neural networks, from the perceptron up | 9 | Fashion-MNIST | old L11 | **done** | **done** |
| 10 | PyTorch | 10 | Fashion-MNIST | old L12 | **done** | **done** |
| 11 | Training deep networks | 11 | CIFAR-10 | old L13+L14 | **done** | **done** |
| 12 | Convolutional networks | 12 | Flowers102 | old L15 | **done** | **done** |
| 13 | Transfer learning | 12 | Flowers102 | old L16 | **done** | **done** |
| 14 | Detection and segmentation | 12 | COCO | old L17+L18 | **done** | **done** |
| 15 | Time series | 13 | Chicago transit | old L19+L20 | **done** | **done** |
| 16 | Recurrent networks | 13 | Chicago transit | old L20 | **done** | **done** |
| 17 | Text | 14 | IMDb | old L21 | **done** | **done** |
| 18 | Attention and transformers | 14–15 | IMDb | old L22 | **done** | **done** |
| 19 | IR: the lexical foundation | notes | SciFact | **new** | **done** | **done** |
| 20 | IR: dense retrieval | notes | SciFact | **new** | **done** | **done** |
| 21 | RecSys: from ratings to factors | notes | MovieLens | **new** | **done** | **done** |
| 22 | RecSys: neural, evaluated honestly | notes | MovieLens | **new** | **done** | **done** |
| 23 | Vision transformers and multimodal retrieval | 15–16 | COCO | old L23 | **done** | **done** |
| 24 | Generation, RAG, and closing | 15–16 | COCO + V | old L24 | **done** | **done** |

### Carried-over debts

Things noticed during the rebuild that are not yet fixed.

- ~~the `try` field is missing from ten lectures~~ — **CLOSED 2026-09-03.**
  All 289 missing boxes were written, so the course now stands at 538 of 538.
  Both halves of the debt are done: the fields, and the rule that stops it
  recurring. `check_notebooks.check_every_box_has_a_try` is **hard**, not
  advisory, and the reasoning for that choice is in its docstring and in
  `AUTHORING.md §4.1a` — a rule that cannot fire on anything that exists is
  exactly the kind to make blocking, since from there the only way to violate
  it is to write a new box without one. Verified by stripping a `try` from a
  copy of `lecture-24.ipynb` and watching the checker fail with the cell index
  and the box name.

- ~~`check_consistency` reported green on two lectures it never checked~~ —
  **found and CLOSED 2026-09-03.** `facts()` extracted a lecture's
  `figures.json` namespace with `re.match(r"(l\d\d|app\d\d)", ...)`, and
  lectures 21 and 22 are the only two whose keys are prefixed `rec21_`/`rec22_`.
  So `facts()` returned nothing for them, `stated_facts` had nothing to anchor
  on, the failure list came back empty, and the run printed *"ok — every stated
  figure is printed by its notebook"* for two lectures it had not compared at
  all. The headline "826 figures verified" was 22 lectures, not 24; the honest
  figure over all 24 is **932**.

  Two changes: `facts()` accepts `rec\d\d`, and `main()` now FAILS when a
  namespace matches no key rather than reporting ok. The general lesson is the
  second one — *a check that cannot fail is worse than no check, because it
  occupies the line where the real one would have been.* Ask it of every
  checker in `tools/`.

  With the fix, lecture 22 failed on five figures immediately: the in-batch
  negatives table at batches 128, 512 and 2,048, and the sampled NDCG@10 for
  the temporal factorisation. Repaired by adding the computation, per the
  standing instruction, not by editing the slide.

- Old decks and notebooks 2–24 still describe Build/Fix, planted defects,
  "commit a number", the twelve threads and the weak-prompt device. Every one is
  rewritten as its row above is worked.
- ~~`notebooks/checkpoints/sorter.pt`~~ — resolved. It is not tracked:
  `notebooks/checkpoints/` is in `.gitignore`, and the file is an *output* of
  Lecture 10's notebook rather than a source artefact. Nothing to delete.
- **Check every deck for claims that describe the old delivery model**, not just
  the old structure. Lecture 1 shipped with four: "you will not type most of the
  code in this course", "a loop, run out loud, every lecture", "rule of the
  room", and four rules written for submitted work when nothing is submitted.
  None of them mentions Build or Fix, so no grep finds them — they have to be
  read for. The question to ask of a slide is *does this describe something that
  still happens?*
- All 24 notebooks were regenerated, so the three-line prompt annotation is
  already gone from every one of them. Their **prose** still describes Build/Fix
  and planted defects; that goes lecture by lecture.
- ~~`assets/figures/d-buildfix.svg`~~ — resolved. No deck cites it and the file
  is gone.
- `tools/deckkit.py` — new: slide-level surgery on a deck, which is how a
  lecture is converted without retyping the slides that survive.
- `tools/figures_app02.py` … `figures_app12.py` are named by the old
  twelve-application scheme, and the figures they emit are named by OLD lecture
  numbers (`l03-*`, `l04-*`). New Lecture 3 legitimately uses both `l03-*` and
  `l04-*` files. Renaming would break every slide that cites one, and
  `check_provenance` with it, so leave it until a lecture is touched anyway.
- ~~dead prompt kwargs~~ **cleared.** All 1,149 are gone from all 24 modules:
  `left_open` and `student` dropped (they described the retired device), and
  383 `catch=` lines folded into their box's `check=`, since they were
  verification instructions and that is what `check` now means. `_prompt.py`
  still accepts the three names silently — tighten it to reject them once no
  branch can reintroduce one.
- **A merge is roughly twice a remap.** L3 fused two 580-line generator modules
  and two decks (72 + 87 slides) into one 92-slide deck and one 27-cell
  notebook. The mechanical parts — stripping the dead annotation kwargs,
  renumbering `## N ·` headings, re-adding the imports the dropped setup cell
  carried — are where the time went, not the writing.
- **Lecture 3 takes ~2.5 min to execute.** The random forest on 60,000 rows
  dominates.
- **Lecture 2 took ~70 s to execute** (150 grid fits at 10 folds). On Colab's
  two cores expect several minutes; the cell says 3-6 min.
- **No notebook ships with stored outputs** (`execution_count: null`
  throughout), so GUIDELINES rule §1.2 — prose figures must appear in a stored
  output — cannot fire. The number-diff against `figures.json` is doing that
  job instead; keep doing it.
- **The measured cost of a lecture**, for planning: L2 = 1 unit. L1 was ~1.5
  (it carried the theme, the site and the tooling). Estimated remaining:
  tier A (L4,5,6,7,9,10) ~7 units, tier B merges (L3, L8) ~4, tier C
  (L11-18, 23, 24, now including the CPU shrink) ~13, tier D (L19-22, from
  nothing) ~12.
