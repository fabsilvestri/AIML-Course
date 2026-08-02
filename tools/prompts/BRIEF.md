# Brief — converting a lecture into a Colab prompt script

You are preparing ONE lecture of *Applications of Machine Learning* (BSc
Mathematics of Artificial Intelligence, Sapienza) to be **rebuilt in Colab by
prompting**, the way lecture 19 was. Your lecture number is in your task
message.

Repo root: `/Users/fabriziosilvestri/Documents/Codice/AIML-Course`

## Read first, in this order

1. **`GUIDELINES.md`** — binding. Every rule applies to your output.
2. **`notebooks/lecture-19.ipynb`** — the exemplar. It is the only notebook
   actually built by prompting, and its boxes are verbatim transcripts of what
   was typed. Study its shape — then improve on it, because the audit that
   produced `GUIDELINES.md` found real defects in it.
3. **Your source module**: `tools/notebooks/lecture_NN.py`.
   *Exception:* lectures 1 and 2 live in `tools/make_notebooks.py`, as the
   functions `lecture_01()` and `lecture_02()`.
4. **`notebooks/lecture-NN.ipynb`** — how your lecture currently renders.

## Deliverable

Exactly one file: **`tools/prompts/lecture_NN.md`**

It is the script a person will follow at a Colab keyboard. For every code cell,
in order:

```
## Cell N — <short label>
**Prompt to type:**
> <the exact prompt, as a person would actually type it>
**Expect:** <what must come back: shapes, columns, the form of the output>
**Assert:** <the assertion that must pass, or "none">
**⏱** <wall clock if over ~20 s, including a CPU figure; omit otherwise>
**Annotate:** full | short
<if full: three bullets — Left open / The usual student version / How you would catch it>
```

## Hard requirements

- **§6.1 — BUDGET THE ANNOTATIONS.** Every cell gets a *short* specification
  box. Only **5–8 cells in the whole notebook** get the full three-bullet
  annotation; choose the ones where the prompt genuinely fails. Annotating
  every cell is the defect being repaired — all three student readers stopped
  reading the template around cell 30, which is exactly where lecture 19 keeps
  its defect. Measured: the course currently has 465 boxes and every one
  carries the full annotation.

- **§6.2 — "The usual student version" must be real.** It must name an actual
  library default or an actual observed failure. If you would be inventing a
  plausible-sounding mistake, mark that cell `short` instead. Good examples:
  `train_test_split` shuffles by default; scipy's `bootstrap` defaults to BCa
  not percentile; `nn.CrossEntropyLoss` applies its own log-softmax;
  `zero_grad()` defaults to `set_to_none=True`; scikit-learn ensembles re-encode
  `y` to positions 0..k−1.

- **Prompts must be REALISTIC.** The audit found lecture 19's prompts were
  "3–5× more specified than anything I type", and that several were
  teacher-prompts that hand over the answer (*"work out which one and drop
  it"*). Write what a competent person would actually type at the keyboard. If
  a defect survives a lazier prompt, that is the **stronger** lesson, not the
  weaker one.

- **§7.1** State wall clock for anything over ~20 s, and give the CPU figure —
  the literal reader could complete only 1.5 of 6 exercises, every failure
  traced to an untimed cell or an unstated re-run order.
  **§7.2** Any exercise requiring re-runs must list which cells, in what order.

- **§1** Every number you put in prose must be derivable from the notebook's
  own data. Verify it with `python3`. Never transcribe a figure.

- **§8** Do not announce a deliberate defect more than once before it. Prefer:
  let the defective cell run unannounced, have the reader write the number
  down, *then* open the next section with the ⚠ and the contrast.

- **§2.1** Any comparison you script must be on matched rows, and the prose must
  say which rows.

## Also required — defect report

End your file with:

```
## Defects found in the current notebook
```

List everything in the CURRENT `notebooks/lecture-NN.ipynb` that violates
`GUIDELINES.md`:

- figures in prose that do not reconcile with the data
- comparisons on mismatched rows or windows
- code quoted in markdown that exists in no cell of that notebook
- cross-references that do not resolve ("the next cell", "ten cells earlier")
- markdown lines indented ≥4 spaces outside a fence, or fences closing indented
- names rebound to a different type across cells; training cells that are not
  idempotent
- instructions a student alone at home with no GPU could not carry out

**Verify each one with `python3` rather than asserting it.** A claim you have
not checked is worth less than no claim. Say explicitly which ones you checked
and which you could not.

## Rules

- Create **only** `tools/prompts/lecture_NN.md`. Modify nothing else — no
  notebooks, no modules, no `git` commands that write.
- `python3` for verification is fine. Datasets already present under
  `notebooks/datasets/` may be read.
- Do **not** execute training cells.
- Your file **is** the deliverable. Write it before you finish.
