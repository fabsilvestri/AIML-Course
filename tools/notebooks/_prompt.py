"""
The prompt box that precedes every code cell.

Lecture 1 teaches a loop — specify, generate, read, test, verify — and draws it
in `d-loop.svg`. Four of its five steps are the student's; the one an assistant
does for them is the typing. So every code cell is preceded by the specification
that produces it, in the four-part vocabulary the lecture uses:

    input       what goes in, with its shape or type
    output      what comes out, and in what units
    constraint  what must be true of the method, not just the answer
    check       how you would know the answer is wrong
    try         one modification, and what should happen to the output

Prefer a `check` whose answer can be worked out on paper before the cell is
run — a shape, a count, a parameter-count arithmetic, a value a formula
predicts. A box whose check is "it runs" is a box that taught nothing.

`try` is what turns a cell from something read into something done. It names
ONE change — a parameter, a column, an argument — and asks what the output
should do, so a student who runs it has a prediction to be wrong about. It is
set apart from the four specification fields because it is not part of the
specification: it is the exercise the specification makes possible. Omit it
where there is genuinely nothing to vary (an import cell), never because
nothing came to mind.

Two honesty rules, both load-bearing:

**These are specifications, not transcripts.** No claim is made that this exact
string was typed into an assistant and returned this exact code. The claim is
narrower and more useful: this is what you would have to ask for to get this
cell, and if your own prompt is vaguer than this, the code you get back will be
worse.

**Blockquote, not styled HTML.** The first version of this file emitted a
`<div>` with inline `style`. It looked right in every local viewer and rendered
as flat undifferentiated text in **Colab**, which strips inline style
attributes, and Colab is the only place students ever open these. Markdown
blockquotes survive Colab, GitHub and nbviewer alike, so that is what this
emits now.

**The three-line annotation is gone.** `left_open` / `student` / `catch` were
the reading-a-weak-prompt device of the old course design. They are still
accepted, and silently dropped, so that a not-yet-rewritten lecture module
still builds — regenerating any notebook removes the annotation immediately.
Delete the kwargs as you rewrite each lecture; the argument will be rejected
once the last one is converted.
"""

from __future__ import annotations

import nbformat as nbf

_SPEC_FIELDS = ("input", "output", "constraint", "check")

# Not a specification field. Rendered under the box, after a rule, because it
# addresses the reader rather than the assistant.
_TRY_FIELD = "try"

# Accepted and dropped; see the module docstring. Not a rendering vocabulary
# any more — just the set of names a stale lecture module may still pass.
_DEAD_FIELDS = ("left_open", "student", "catch")


def prompt(*, label: str = "", **fields: str) -> nbf.NotebookNode:
    """A prompt box, as a markdown cell.

        prompt(label="tidy",
               input="the raw CTA CSV, one row a day",
               output="a DataFrame indexed by date, bus and rail columns",
               constraint="drop duplicate rows and say how many there were",
               check="the index is sorted and unique, and "
                     "(raw.bus + raw.rail == raw.total).all()",
               **{"try": "drop the sort. Which of the two asserts fails, "
                         "and which one silently still passes?"})

    `try` is a Python keyword, so it can only be passed as **{"try": ...}.
    That is ugly enough to be worth the trade: the field reads as `try ·` in
    the notebook, which is the word that belongs there.

    Any field may be omitted where it genuinely does not apply — a setup cell
    has no meaningful `check` — but omitting `constraint` on a cell that fits
    or scores a model is almost always a specification too vague to be worth
    writing down.
    """
    unknown = set(fields) - set(_SPEC_FIELDS) - {_TRY_FIELD} - set(_DEAD_FIELDS)
    if unknown:
        raise ValueError(
            f"prompt() got {sorted(unknown)}; the vocabulary is "
            f"{list(_SPEC_FIELDS)} plus {_TRY_FIELD!r}")

    spec = [f"> **{k}** · {_flat(fields[k])}"
            for k in _SPEC_FIELDS if fields.get(k)]
    if not spec:
        raise ValueError("prompt() needs at least one of "
                         + ", ".join(_SPEC_FIELDS))

    head = f"> **Prompt · {label}**" if label else "> **Prompt**"
    parts = [head, ">", "\n>\n".join(spec)]

    if fields.get(_TRY_FIELD):
        parts += [">", "> ---", ">",
                  f"> **try** · {_flat(fields[_TRY_FIELD])}"]

    return nbf.v4.new_markdown_cell("\n".join(parts))


def _flat(text: str) -> str:
    """One line, however the caller wrapped it in the source."""
    return " ".join(str(text).split())
