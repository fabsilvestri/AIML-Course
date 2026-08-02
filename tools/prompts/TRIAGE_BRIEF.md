# Brief — triaging a notebook's defect report

Phase A produced `tools/prompts/lecture_NN.md`, each ending in a
`Defects found in the current notebook` section. Across 23 notebooks those
sections contain **413 claims**. Your job is to establish which are real.

Your lecture number is in your task message.
Repo root: `/Users/fabriziosilvestri/Documents/Codice/AIML-Course`

## Your stance: refute, do not confirm

**Default to FALSE POSITIVE.** A claim is real only if you can *demonstrate*
it — run the code, print the number, show the mismatch. "It looks wrong to me"
is a false positive. "I re-derived it and got X where the notebook says Y" is
confirmed.

This matters because a triage that rubber-stamps is worse than no triage: it
launders 413 guesses into 413 facts, and the rebuild then spends its effort on
things that were never broken. The audit that produced `GUIDELINES.md` found
that the most dangerous defect in this course was **a check that passed for the
wrong reason**. Do not become one.

Where a claim is about a *number*, re-derive the number. Where it is about
*behaviour* (raises / does not raise, renders / does not render), run it.
Datasets already cached under `notebooks/datasets/` may be read.

## Read first

1. `GUIDELINES.md` — the rules the claims are made against.
2. `tools/prompts/lecture_NN.md` — the claims you are triaging, at the end.
3. `notebooks/lecture-NN.ipynb` — the artefact they are about.
4. `tools/notebooks/lecture_NN.py` — its source (lectures 1 and 2:
   `tools/make_notebooks.py`, functions `lecture_01` / `lecture_02`).

## Deliverable

Exactly one file: **`tools/prompts/triage_NN.md`**

One entry per claim, in the order they appear in the Phase A report:

```
### Claim N — <one-line restatement>
**Verdict:** CONFIRMED | FALSE POSITIVE | UNVERIFIABLE
**Evidence:** <the command you ran and what it printed, or the exact lines of
the notebook that settle it. Not an argument — an output.>
**Severity:** misleads a student | wrong but harmless | cosmetic
**Origin:** hand-written prose | generated code | notebook structure
**Fix:** <one line, or "none needed">
```

Then a summary block:

```
## Summary
confirmed: N   false positive: N   unverifiable: N
of the confirmed, N mislead a student
origin split — prose: N   code: N   structure: N
duplicates: <claims that are the same underlying defect counted twice>
```

## What counts as UNVERIFIABLE

Say so plainly rather than guessing. Legitimate cases:

- the claim concerns a training run you must not execute
- the notebook stores no outputs, so a prose figure cannot be reconciled
  against anything (this is true of all 23 — note it once, do not repeat it per
  claim)
- the dataset is not cached and downloading it is out of scope

An UNVERIFIABLE claim is not a false positive. It is a claim nobody has tested,
which is a different and sometimes worse thing.

## The `Origin` field matters

The audit found that in this course the defects concentrate in **hand-written
prose**, not in generated code — one reader put it as *"every genuine defect I
found is in the markdown, not the Python."* Three claims have already been
verified by hand and all three were prose errors. Recording origin per claim
lets us test whether that holds at scale, so be accurate about it.

## Calibration

Some claims are already settled. If your lecture contains one of these, your
verdict on it tells us whether to trust your other verdicts:

- **Lecture 3** — `i_pos = int(np.argmax(y_train))  # first 5` returns the
  first **nine**, not the first five, and `bool(9)` prints as `True`.
  **CONFIRMED**, verified independently.
- **Lecture 3** — the prompt box claim *"imshow of a 784-long vector is not an
  error, it is a stripe"*. It raises `TypeError: Invalid shape (784,) for image
  data`. **CONFIRMED**, verified independently.
- **Lecture 6** — the prompt box claim that liblinear *"fits an intercept of
  exactly 0.0000 where saga fits 4.90"*. liblinear gives 0.0000; saga gives
  −0.0001. **CONFIRMED**, verified independently.

Reach your own verdict on these anyway. If you disagree with one, say why and
show the evidence — the independent verification could itself be wrong.

## Rules

- Create **only** `tools/prompts/triage_NN.md`. Modify nothing else — no
  notebooks, no modules, no `git` commands that write.
- `python3` for verification is expected, not optional.
- Do **not** execute training cells.
- Your file **is** the deliverable. Write it before you finish.
