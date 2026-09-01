# Rebuild status

Resumable progress log for the redesign. **If you are picking this up cold —
a new session, a restarted machine — read this file first, then `LECTURES.md`
(the plan) and `AUTHORING.md` (how to build one lecture).**

Last updated: 2026-09-01 (Lecture 1 built) · Term starts: ~2026-09-22

---

## How to resume

1. `git log --oneline -15` — every lecture is committed on its own, so the last
   commit says exactly where work stopped.
2. Find the first row below that is not `done`. That is the next task.
3. Read `AUTHORING.md` §2 (deck anatomy) and §4 (notebooks) before writing.
4. Build deck and notebook for that lecture, run the checks in §7, commit with
   the message form `slides+notebooks/lecture-NN — <topic>`, and update the row
   here **in the same commit**.

Never leave this file stale. A row that says `wip` with no commit behind it is
worse than no row.

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
  replaced. Lecture 1 is the template; it gets signed off before the other 23.

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
| `index.html` — site rebuild | todo | after L1 sign-off |
| `tools/make_notebooks.py` / `_prompt.py` — retargeted to AUTHORING §4 | **done** | three-line annotation dropped everywhere; `COLAB_AUTHORED` emptied so L19 generates like the rest |
| `tools/check_notebooks.py` — rules retargeted to AUTHORING §4 | todo | §6.1 box budget no longer fires; the 4 blocking rules still apply |
| `tools/make_figures.py` — figures for reassigned lectures | todo | figure names are `l03-*` etc. by old numbering |
| Part V figures (`figures_ir.py`, `figures_recsys.py`) | todo | nothing exists yet |

### Lectures

Deck = `slides/lecture-NN.html`, Notebook = `notebooks/lecture-NN.ipynb`.
"Source" names the old lecture whose material is reused.

| # | Topic | Ch | Dataset | Source | Deck | Notebook |
|---|---|---|---|---|---|---|
| 1 | What ML is, and how we will work | 1–2 | housing | old L1 | **done** | **done** |
| 2 | The end-to-end project | 2 | housing | old L1+L2 | todo | todo |
| 3 | Classification and its metrics | 3 | MNIST | old L3+L4 | todo | todo |
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
- All 24 notebooks were regenerated, so the three-line prompt annotation is
  already gone from every one of them. Their **prose** still describes Build/Fix
  and planted defects; that goes lecture by lecture.
- `assets/figures/d-buildfix.svg` is now unreferenced. Delete once no deck
  cites it.
- `tools/deckkit.py` — new: slide-level surgery on a deck, which is how a
  lecture is converted without retyping the slides that survive.
- `tools/figures_app02.py` … `figures_app12.py` are named by the old
  twelve-application scheme. Rename to lecture numbers as each is touched.
