# Rebuild status

Resumable progress log for the redesign. **If you are picking this up cold —
a new session, a restarted machine — read this file first, then `LECTURES.md`
(the plan) and `AUTHORING.md` (how to build one lecture).**

Last updated: 2026-09-01 (Lectures 1-3 built) · Term starts: ~2026-09-22

---

## How to resume

1. `git log --oneline -15` — every lecture is committed on its own, so the last
   commit says exactly where work stopped.
2. Find the first row below that is not `done`. That is the next task.
3. Read `AUTHORING.md` §2 (deck anatomy) and §4 (notebooks) before writing.

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

| lecture | first run | after fixes | status |
|---|---|---|---|
| 1 | 4 | **0** | clean |
| 2 | 28 → 24 → 9 | in progress | fixing |
| 3 | not yet run | | |
| 8 | not yet run (draft at `lecture-08-NEW`) | | |

The fixes are almost never "correct a wrong number". They are "the notebook
never computed this at all" — a slide asserting something no cell produces. So
the repair is to add the computation, which is also the right thing
pedagogically: a student running the notebook can now check the claim in front
of them. Lecture 2 gained six cells this way (all four training RMSEs, both
paired comparisons, the three imputation strategies, the bootstrap half-width,
the ten worst predictions, the capped-districts trap and the
absolute-versus-relative arms).

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

| agent | drafts | from | target files |
|---|---|---|---|
| L4-L5-Titanic | L4, L5 | old 05, 06 | `lecture-04/05.html`, `lecture_04/05.py` |
| L6-L7-CoverType | L6, L7 | old 07, 08 | `lecture-06/07.html`, `lecture_06/07.py` |
| L8-Olivetti | L8 (merge) | old 09 + 10 | `lecture-08-NEW.html`, `lecture_08_NEW.py` |

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
| `tools/check_notebooks.py` — rules retargeted to AUTHORING §4 | todo | §6.1 box budget no longer fires; the 4 blocking rules still apply |
| `tools/check_decks.py` — site check rewritten | **done** | was "every lecture must be linked", which is false mid-rebuild; now checks each lecture is on the page, that a linked one has its files, and that nothing is marked *In preparation* while a converted deck exists |
| `tools/make_figures.py` — figures for reassigned lectures | todo | figure names are `l03-*` etc. by old numbering |
| Part V figures (`figures_ir.py`, `figures_recsys.py`) | todo | nothing exists yet |

### Lectures

Deck = `slides/lecture-NN.html`, Notebook = `notebooks/lecture-NN.ipynb`.
"Source" names the old lecture whose material is reused.

| # | Topic | Ch | Dataset | Source | Deck | Notebook |
|---|---|---|---|---|---|---|
| 1 | What ML is, and how we will work | 1–2 | housing | old L1 | **done** | **done** |

*Lectures 1-3 are complete through all five steps of the routine, site included.*
| 2 | The end-to-end project | 2 | housing | old L1+L2 | **done** | **done** |
| 3 | Classification and its metrics | 3 | MNIST | old L3+L4 | **done** | **done** |
| 4 | Training models | 4 | Titanic | old L5 | todo | todo |
| 5 | Regularisation and bias–variance | 4 | Titanic | old L6 | todo | todo |
| 6 | Decision trees | 5 | CoverType | old L7 | todo | todo |
| 7 | Ensembles and random forests | 6 | CoverType | old L8 | todo | todo |
| 8 | Dimensionality reduction and unsupervised | 7–8 | Olivetti | old L9+L10 | todo | todo |
| 9 | Neural networks, from the perceptron up | 9 | Fashion-MNIST | old L11 | todo | todo |
| 10 | PyTorch | 10 | Fashion-MNIST | old L12 | todo | todo |
| 11 | Training deep networks | 11 | CIFAR-10 | old L13+L14 | todo | todo |
| 12 | Convolutional networks | 12 | Flowers102 | old L15 | todo | todo |
| 13 | Transfer learning | 12 | Flowers102 | old L16 | todo | todo |
| 14 | Detection and segmentation | 12 | COCO | old L17+L18 | todo | todo |
| 15 | Time series | 13 | Chicago transit | old L19+L20 | todo | todo |
| 16 | Recurrent networks | 13 | Chicago transit | old L20 | todo | todo |
| 17 | Text | 14 | IMDb | old L21 | todo | todo |
| 18 | Attention and transformers | 14–15 | IMDb | old L22 | todo | todo |
| 19 | IR: the lexical foundation | notes | SciFact | **new** | todo | todo |
| 20 | IR: dense retrieval | notes | SciFact | **new** | todo | todo |
| 21 | RecSys: from ratings to factors | notes | MovieLens | **new** | todo | todo |
| 22 | RecSys: neural, evaluated honestly | notes | MovieLens | **new** | todo | todo |
| 23 | Vision transformers and multimodal retrieval | 15–16 | COCO | old L23 | todo | todo |
| 24 | Generation, RAG, and closing | 15–16 | COCO + V | old L24 | todo | todo |

### Carried-over debts

Things noticed during the rebuild that are not yet fixed.

- Old decks and notebooks 2–24 still describe Build/Fix, planted defects,
  "commit a number", the twelve threads and the weak-prompt device. Every one is
  rewritten as its row above is worked.
- `notebooks/checkpoints/sorter.pt` — tracked 1 MB model from an old notebook.
  Delete when the notebook that produced it is rebuilt.
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
- `assets/figures/d-buildfix.svg` is now unreferenced. Delete once no deck
  cites it.
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
