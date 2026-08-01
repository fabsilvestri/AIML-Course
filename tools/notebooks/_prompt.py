"""
The prompt box that precedes every code cell, and the three lines under it.

Lecture 1 teaches a loop — specify, generate, read, test, verify — and draws it
in `d-loop.svg`. Four of its five steps are the student's; the one an assistant
does for them is the typing. A notebook that shows only the typing teaches the
one step that was never the hard part.

So every code cell is preceded by the specification that produces it, in the
four-part vocabulary the lecture uses:

    input       what goes in, with its shape or type
    output      what comes out, and in what units
    constraint  what must be true of the method, not just the answer
    check       how you would know the answer is wrong

and then by three lines about the specification itself:

    left_open   what it fails to pin down, and what that costs
    student     the version a student actually writes, and what it gets them
    catch       how you would catch a wrong answer, concretely

**The three lines are the ones that teach.** The four fields describe a good
prompt; the three lines describe how to *read* one, which is the transferable
skill. Write them about this cell — naming the actual variable, the actual
number you expect — or leave them out. A generic caution is worse than silence,
because it trains students to skim exactly the block they should read.

Two honesty rules, both load-bearing:

**These are specifications, not transcripts.** No claim is made that this exact
string was typed into an assistant and returned this exact code. The claim is
narrower and more useful: this is what you would have to ask for to get this
cell, and if your own prompt is vaguer than this, the code you get back will be
worse. `lecture-19.ipynb` is the single exception in this course — it was built
cell by cell in Colab against Gemini 3.1 Pro, and its boxes are verbatim. It
says so in its own header. Nothing else may claim that.

**Blockquote, not styled HTML.** The first version of this file emitted a
`<div>` with inline `style` — a purple rule, a tinted background, the lot. It
looked right in every local viewer and rendered as flat undifferentiated text
in **Colab**, which strips inline style attributes, and Colab is the only place
students ever open these. The formatting was checked in the one environment
that did not matter. Markdown blockquotes survive Colab, GitHub and nbviewer
alike, so that is what this emits now.
"""

from __future__ import annotations

import nbformat as nbf

_SPEC_FIELDS = ("input", "output", "constraint", "check")

_NOTE_FIELDS = (
    ("left_open", "Left open"),
    ("student", "The usual student version"),
    ("catch", "How you would catch it"),
)


def prompt(*, label: str = "", **fields: str) -> nbf.NotebookNode:
    """A prompt box, as a markdown cell.

        prompt(label="tidy",
               input="the raw CTA CSV, one row a day",
               output="a DataFrame indexed by date, bus and rail columns",
               constraint="drop duplicate rows and say how many there were",
               check="the index is sorted and unique",
               left_open="which column is redundant — deliberately",
               student="'clean the dataframe', which gets you a dropna() "
                       "nobody asked for and no count of what went",
               catch="(raw.bus + raw.rail == raw.total).all() settles it "
                     "in one line, without asking anyone")

    Any field may be omitted where it genuinely does not apply — a setup cell
    has no meaningful `check` — but omitting `constraint` on a cell that fits
    or scores a model is almost always a specification too vague to be worth
    writing down.
    """
    unknown = set(fields) - set(_SPEC_FIELDS) - {k for k, _ in _NOTE_FIELDS}
    if unknown:
        raise ValueError(
            f"prompt() got {sorted(unknown)}; the vocabulary is "
            f"{list(_SPEC_FIELDS)} plus {[k for k, _ in _NOTE_FIELDS]}")

    spec = [f"> **{k}** · {_flat(fields[k])}"
            for k in _SPEC_FIELDS if fields.get(k)]
    if not spec:
        raise ValueError("prompt() needs at least one of "
                         + ", ".join(_SPEC_FIELDS))

    head = f"> **Prompt · {label}**" if label else "> **Prompt**"
    parts = [head, ">", "\n>\n".join(spec)]

    notes = [f"* **{title}:** {_flat(fields[k])}"
             for k, title in _NOTE_FIELDS if fields.get(k)]
    if notes:
        parts += ["", "**Watch this prompt.**", ""] + notes

    return nbf.v4.new_markdown_cell("\n".join(parts))


def _flat(text: str) -> str:
    """One line, however the caller wrapped it in the source."""
    return " ".join(str(text).split())
