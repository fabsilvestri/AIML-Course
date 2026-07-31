"""
The specification box that precedes every code cell.

Lecture 1 teaches a loop — specify, generate, read, test, verify — and draws it
in `d-loop.svg`. Four of its five steps are the student's; the one an assistant
does for them is the typing. A notebook that shows only the typing teaches the
one step that was never the hard part.

So every code cell in every notebook is preceded by the specification that
produces it, in the four-part vocabulary the lecture uses:

    input       what goes in, with its shape or type
    output      what comes out, and in what units
    constraint  what must be true of the method, not just the answer
    check       how you would know the answer is wrong

**These are specifications, not transcripts.** No claim is made that this exact
string was typed into an assistant and returned this exact code — the course
does not get to invent provenance it did not record. The claim is narrower and
more useful: this is what you would have to ask for to get this cell, and if
your own prompt is vaguer than this, the code you get back will be worse.

The `constraint` and `check` lines are the ones that matter. Lecture 19's
planted defect is a random K-fold on a time series, and it survives precisely
because a prompt with no constraint line cannot forbid it.
"""

from __future__ import annotations

import html

import nbformat as nbf

# Purple is the algebra hue in this course's palette (TRICKS section 7), which
# is the right register: a specification is the reasoning, not the result.
_BOX = (
    '<div style="border-left:4px solid #6c3483;background:#f6f0f9;'
    'padding:0.55em 0.9em;margin:0.3em 0 0.2em 0;border-radius:3px;'
    'font-size:0.95em">\n'
    '<b style="color:#6c3483">Prompt</b> '
    '<span style="color:#4b5563">&mdash; read this before the code, and ask '
    'whether the code below actually satisfies it</span>\n'
    '<div style="margin-top:0.45em;line-height:1.55">\n{body}\n</div>\n'
    "</div>"
)

_FIELDS = ("input", "output", "constraint", "check")


def prompt(**fields: str) -> nbf.NotebookNode:
    """A specification box, as a markdown cell.

        prompt(input="the raw CTA CSV, one row a day",
               output="a DataFrame indexed by date, bus and rail columns",
               constraint="drop duplicate rows and say how many there were",
               check="the index is sorted and unique")

    Any of the four may be omitted where it genuinely does not apply — a setup
    cell has no meaningful `check` — but omitting `constraint` on a cell that
    fits or scores a model is almost always a specification that is too vague to
    be worth writing down.
    """
    unknown = set(fields) - set(_FIELDS)
    if unknown:
        raise ValueError(f"prompt() got {sorted(unknown)}; "
                         f"the vocabulary is {list(_FIELDS)}")
    rows = []
    for key in _FIELDS:
        if key not in fields:
            continue
        text = " ".join(str(fields[key]).split())
        rows.append(f'<b style="color:#16212b">{key}</b> '
                    f'<span style="color:#33414d">&middot; {html.escape(text)}'
                    f"</span><br>")
    if not rows:
        raise ValueError("prompt() needs at least one of " + ", ".join(_FIELDS))
    return nbf.v4.new_markdown_cell(_BOX.format(body="\n".join(rows)))
