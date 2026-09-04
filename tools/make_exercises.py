#!/usr/bin/env python3
"""
Five exam-style exercises at the end of every deck, answered one lecture later.

    python3 tools/make_exercises.py          # rewrite every deck
    python3 tools/make_exercises.py 7 8      # just these

THE DESIGN, set by the lecturer 2026-09-03.

  * Every deck, 1 to 24, ends with five exercises in the style of the written
    paper. They are NOT presented -- they are there for a student reading the
    deck afterwards.
  * Lecture N's solutions appear on lecture N+1's deck, so a student has to
    attempt them before the answer is available.
  * Lecture 24 is the exception: no lecture 25 exists, so it carries lecture
    23's solutions, its own five, AND its own solutions.

This extends a pattern the decks already had. Lecture 1 sets a specimen Part B
question whose last two parts are deliberately unanswerable until
cross-validation exists, and Lecture 2 answers them -- "◇ Answer — 3" and
"◇ Answer — 4". The exercises below use the same slide vocabulary, so a
student cannot tell the new ones from the ones that were always there.

WHY GENERATED. The same reason index.html is: 24 decks x (5 questions + 5
solutions) is 240 near-identical blocks, hand-editing them is how a solution
ends up on the wrong deck, and a solution on the wrong deck is worse than no
solution at all. Everything lives in EXERCISES below, and the injection is
idempotent between BEGIN/END markers.
"""
from __future__ import annotations

import html
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BEGIN, END = "<!-- BEGIN EXERCISES -->", "<!-- END EXERCISES -->"

# ----------------------------------------------------------------------
# One entry per exercise:
#   q     the question, as HTML. Keep it to a sentence or two -- the paper does.
#   marks what it is worth, so the length of the expected answer is obvious.
#   a     the answer, in as few words as the question allows.
#   why   the reasoning, one bullet per step. Two or three, never five.
# ----------------------------------------------------------------------
EXERCISES: dict[int, list[dict]] = {}


def ex(n, q, marks, a, why):
    EXERCISES.setdefault(n, []).append(
        dict(q=q, marks=marks, a=a, why=why))


# ------------------------------------------------------------------ L01
ex(1, "A colleague reports that their model reaches an RMSE of $48{,}000$ on "
      "the California housing data, and that they chose the model by trying "
      "six of them and keeping the best. State, in one sentence, what that "
      "number is an estimate of &mdash; and what it is <em>not</em> an "
      "estimate of.", 4,
   "It estimates the best-of-six score on that particular split; it does not "
   "estimate the error on new data.",
   ["Choosing the minimum of six numbers measured on the same data makes that "
    "minimum a <em>selection</em>, not a measurement",
    "The quantity a client cares about &mdash; error on districts nobody has "
    "seen &mdash; needs data that took part in no choice"])
ex(1, "The median income column is capped at 15. Give one consequence for the "
      "<em>model</em> and one for the <em>evaluation</em>, and say which of "
      "the two you would raise with the client first.", 5,
   "Model: it can never learn what happens above the cap. Evaluation: the test "
   "set is capped too, so the error is silent about exactly those districts.",
   ["The cap is in both halves of the split, so no held-out number reveals it",
    "Raise the evaluation one first: the model can be retrained, but a metric "
    "that cannot see the failure will not tell you to"])
ex(1, "You are given a stratified split on income category and a naive random "
      "split of the same data. Both report similar RMSE. Does that mean the "
      "stratification was unnecessary? Answer yes or no, with a reason.", 4,
   "No.",
   ["Similar RMSE on <em>one</em> draw says nothing about the spread over "
    "draws, and the stratification is there to reduce that spread",
    "The failure a naive split causes is occasional and large, which is "
    "precisely what a single comparison cannot show"])
ex(1, "The brief says the output feeds a downstream system that expects a "
      "price. Name the one thing this fact settles about the problem framing, "
      "and one thing it leaves open.", 4,
   "It settles that this is supervised regression, not classification. It "
   "leaves the metric open.",
   ["'A price' fixes the type of the output and nothing about how error is "
    "weighed",
    "Whether being wrong by $50{,}000$ on a cheap district matters as much as "
    "on an expensive one is a question for the client, not the data"])
ex(1, "In the working loop &mdash; specify, generate, read, test, verify "
      "&mdash; exactly one step is done by the assistant. Name it, and say "
      "what goes wrong if a student treats a second step as the "
      "assistant&rsquo;s.", 3,
   "Generate. If reading is also delegated, nothing checks that the code does "
   "what the specification asked for.",
   ["The specification is the student's claim about what the code must do",
    "Only a reader who holds that claim can notice the code meeting a "
    "different one"])


def slide(title, body, menu=None, cls=""):
    menu = menu or title
    c = f' class="{cls}"' if cls else ""
    return (f'<section{c} data-menu-title="{menu}">\n{body}\n</section>\n')


def question_slides(n, items):
    """Two slides: three questions then two, so neither runs over the canvas."""
    out = []
    out.append(slide(
        "",
        '  <p class="kicker">Not presented &mdash; for afterwards</p>\n'
        f'  <h1>Exercises<br>Lecture {n}</h1>\n'
        '  <p class="clock">five, in the style of the written paper</p>',
        menu=f"◆ Exercises · Lecture {n}", cls="divider"))
    for part, (lo, hi) in enumerate(((0, 3), (3, 5))):
        lis = "\n".join(
            f'    <li>{it["q"]} <span class="ex-marks">[{it["marks"]}]</span></li>'
            for it in items[lo:hi])
        cnt = f' style="counter-reset: ex {lo}"' if lo else ""
        out.append(slide(
            "", f'  <h2>Exercises &mdash; Lecture {n}'
                f'{" (continued)" if lo else ""}</h2>\n'
                f'  <ul class="ex-list"{cnt}>\n{lis}\n  </ul>\n'
                f'  <p class="ex-note">Solutions on Lecture {n + 1}&rsquo;s deck.</p>'
            if n < 24 else
                f'  <h2>Exercises &mdash; Lecture {n}'
                f'{" (continued)" if lo else ""}</h2>\n'
                f'  <ul class="ex-list"{cnt}>\n{lis}\n  </ul>\n'
                f'  <p class="ex-note">Solutions overleaf &mdash; there is no '
                f'Lecture 25.</p>',
            menu=f"Exercises · {n}{' (2)' if lo else ''}"))
    return out


def solution_slides(n, items):
    """Two answers a slide, side by side.

    Three to a slide ran 706px on Lecture 2 against a footer at 674 -- the
    check_overflow warning that made this layout two-up. It is also the layout
    the deck's own specimen answers already use, so the pages match.
    """
    out = []
    out.append(slide(
        "",
        '  <p class="kicker">Set last time</p>\n'
        f'  <h1>Solutions<br>Lecture {n}</h1>\n'
        '  <p class="clock">not presented</p>',
        menu=f"◇ Solutions · Lecture {n}", cls="divider"))
    for part, (lo, hi) in enumerate(((0, 2), (2, 4), (4, 5))):
        chunk = items[lo:hi]
        if not chunk:
            continue
        blocks = []
        for k, it in enumerate(chunk, start=lo + 1):
            why = "\n".join(f'        <li>{w}</li>' for w in it["why"])
            blocks.append(
                f'    <div>\n'
                f'      <p class="smaller"><strong>{k}.</strong> '
                f'{short(it["q"])}</p>\n'
                f'      <p class="lead"><span class="fix">{it["a"]}</span></p>\n'
                f'      <ul class="tight smaller">\n{why}\n      </ul>\n'
                f'    </div>')
        out.append(slide(
            "", f'  <h2>Solutions &mdash; Lecture {n}'
                f'{f" ({part + 1})" if part else ""}</h2>\n'
                f'  <div class="cols">\n' + "\n".join(blocks) + '\n  </div>',
            menu=f"Solutions · {n}{f' ({part + 1})' if part else ''}"))
    return out


def short(q, limit=110):
    """The question, trimmed to a reminder rather than repeated in full."""
    plain = re.sub(r"<[^>]+>", "", q)
    plain = " ".join(plain.split())
    return plain if len(plain) <= limit else plain[:limit].rsplit(" ", 1)[0] + "&hellip;"


def build(n):
    """Every exercise/solution slide lecture n's deck should carry."""
    out = []
    if n - 1 in EXERCISES:
        out += solution_slides(n - 1, EXERCISES[n - 1])
    if n in EXERCISES:
        out += question_slides(n, EXERCISES[n])
        if n == 24:
            out += solution_slides(24, EXERCISES[24])
    return out


def inject(n) -> str | None:
    path = ROOT / f"slides/lecture-{n:02d}.html"
    src = path.read_text(encoding="utf-8")
    body = "\n".join(build(n))
    if not body:
        return None
    block = f"{BEGIN}\n{body}{END}\n"

    if BEGIN in src:
        src = re.sub(re.escape(BEGIN) + r".*?" + re.escape(END) + r"\n",
                     block, src, flags=re.S)
    else:
        # before the closing "Next" divider, which is always the last slide
        i = src.rfind('<section class="divider"')
        if i < 0:
            i = src.rfind("</section>")
            i = src.rfind("<section", 0, i)
        src = src[:i] + block + src[i:]
    path.write_text(src, encoding="utf-8")
    return path.name


def main() -> int:
    want = [int(a) for a in sys.argv[1:]] or list(range(1, 25))
    done = [r for n in want if (r := inject(n))]
    have = sorted(EXERCISES)
    print(f"{len(done)} deck(s) rewritten; exercises defined for "
          f"{len(have)} lecture(s): {have}")
    missing = [n for n in range(1, 25) if n not in EXERCISES]
    if missing:
        print(f"still to write: {missing}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
