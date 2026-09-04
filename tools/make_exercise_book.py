#!/usr/bin/env python3
"""
Every exercise in the course, each followed by its solution, as one LaTeX book.

    python3 tools/make_exercise_book.py     # writes notes/exercises.tex
    make -C notes                           # builds notes/exercises.pdf

The questions and answers are NOT retyped here. They are imported from
tools/make_exercises.py, the same table the decks are generated from, so the
book and the slides cannot disagree: fix a solution in one place and both
change. That is the whole reason this is a generator rather than a document.

The only real work is turning the slide markup into LaTeX. The exercise text
is HTML with inline mathematics in it, so the conversion has to protect the
maths and the code spans before escaping anything -- an unprotected `_` inside
$p_u$ becomes a subscript error, and an unprotected `&` anywhere ends the
document.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

from make_exercises import EXERCISES                        # noqa: E402
from make_site import LECTURES                              # noqa: E402

TITLE = {n: t for n, t, *_ in LECTURES}
SOURCE = {n: (s or "Lecture notes") for n, _, s, *_ in LECTURES}

ENTITIES = {
    "&mdash;": "---", "&ndash;": "--", "&hellip;": r"\ldots{}",
    "&rsquo;": "'", "&lsquo;": "`", "&ldquo;": "``", "&rdquo;": "''",
    "&nbsp;": "~", "&times;": r"$\times$", "&le;": r"$\le$",
    "&ge;": r"$\ge$", "&lt;": "<", "&gt;": ">", "&amp;": r"\&",
}


def tex(s: str) -> str:
    """HTML from the slide data -> LaTeX.

    The order matters and cost two build failures to get right. Anything this
    function GENERATES is LaTeX and must not be escaped afterwards -- the first
    version escaped its own output and produced `\\textbackslash{}emph{...}`.
    So every generated fragment, including the ones that replace a tag or an
    entity, is parked in `kept` and only the literal prose is escaped.
    """
    kept: list[str] = []

    def keep(payload: str) -> str:
        kept.append(payload)
        return f"\0{len(kept) - 1}\0"

    # 1. inline maths, verbatim: it is TeX already, and escaping it would break
    #    every subscript in the course.
    # Entities can appear INSIDE the maths -- the decks must write `&lt;` there,
    # because a bare `<` would open an HTML tag and swallow the closing `$`
    # (AUTHORING 5.3a). LaTeX wants the character itself, so decode before
    # protecting, or `$\rho(h) &lt; 1/2$` reaches TeX with a live `&` in it.
    MATH_ENTITIES = {"&lt;": "<", "&gt;": ">", "&le;": r"\le ",
                     "&ge;": r"\ge ", "&nbsp;": r"\,", "&amp;": r"\&"}

    def math(m):
        body = m.group(0)
        for k, v in MATH_ENTITIES.items():
            body = body.replace(k, v)
        return keep(body)

    s = re.sub(r"\$[^$]*\$", math, s)

    # 2. code spans, escaped by their own rules and then protected whole
    def code(m):
        inner = m.group(1)
        for ch, rep in (("\\", r"\textbackslash{}"), ("{", r"\{"), ("}", r"\}"),
                        ("_", r"\_"), ("&", r"\&"), ("%", r"\%"), ("#", r"\#"),
                        ("$", r"\$")):
            inner = inner.replace(ch, rep)
        return keep(r"\code{" + inner + "}")
    s = re.sub(r"<code>(.*?)</code>", code, s, flags=re.S)

    # 3. tags become protected LaTeX, but the text BETWEEN them stays literal
    #    so that it still gets escaped in step 5.
    for tag, opener in (("em", r"\emph{"), ("strong", r"\textbf{")):
        s = re.sub(rf"<{tag}>", lambda _m, o=opener: keep(o), s)
        s = re.sub(rf"</{tag}>", lambda _m: keep("}"), s)
    s = re.sub(r"<[^>]+>", "", s)

    # 4. entities, likewise protected
    for k, v in ENTITIES.items():
        s = s.replace(k, keep(v))

    # 5. only literal prose is left; escape it
    for ch, rep in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"),
                    ("#", r"\#"), ("_", r"\_"), ("^", r"\^{}"),
                    ("{", r"\{"), ("}", r"\}"), ("~", r"\textasciitilde{}")):
        s = s.replace(ch, rep)

    # 6. restore
    s = re.sub("\0(\\d+)\0", lambda m: kept[int(m.group(1))], s)
    return " ".join(s.split())


HEAD = r"""\documentclass[11pt,a4paper]{article}
\input{preamble}

% One exercise, then its solution. The solution is set immediately after the
% question rather than at the back: this is a revision document, not an exam
% paper, and a reader who has to flip to check an answer stops checking.
\newtcolorbox{solution}[1][]{
  breakable, enhanced, colback=white, colframe=aimlsuccess,
  boxrule=0.5pt, left=8pt, right=8pt, top=5pt, bottom=5pt,
  fonttitle=\bfseries\small, coltitle=white, colbacktitle=aimlsuccess,
  attach boxed title to top left={yshift=-2mm, xshift=4mm},
  boxed title style={boxrule=0pt, arc=1pt}, #1}

\newcounter{exq}[section]
\newcommand{\question}[2]{%
  \refstepcounter{exq}%
  \medskip
  \noindent
  \begin{minipage}{\textwidth}
    \textbf{\theexq.}~#1 \hfill{\color{aimlmuted}\textbf{[#2]}}
  \end{minipage}\par}

\begin{document}
\thispagestyle{empty}
\begin{flushleft}
  {\color{aimlmuted}\small\sffamily
   APPLICATIONS OF MACHINE LEARNING \quad$\cdot$\quad
   BSc Mathematics of Artificial Intelligence}\\[2mm]
  {\color{aimlrule}\rule{\textwidth}{0.8pt}}\\[4mm]
  {\Huge\bfseries\color{aimlprimary} Exercises and solutions\par}\vspace{3mm}
  {\color{aimlmuted}\large All twenty-four lectures \quad$\cdot$\quad
   120 questions, each answered}\\[3mm]
  {\color{aimlrule}\rule{\textwidth}{0.8pt}}
\end{flushleft}
\vspace{2mm}

\begin{keybox}[title={How to use this}]
These are the five questions set at the end of each lecture's deck, in the
style of the written paper, with the solutions that appear on the following
lecture's deck. Everything here is on the slides already; this is the same
material in one place, for revision.

\textbf{They are also the oral.} If you sit the optional oral, three of these
120 are drawn in front of you and you answer them aloud, marked against the
same scheme printed here with each solution. There is no second, hidden bank:
this is it.

Each question is answerable from \textbf{its own lecture and the ones before
it}, and from nothing else --- that is a rule the course checks mechanically,
not an aspiration. So if a question seems to need something you have not met,
the fault is ours: say so.

The mark in brackets is what the question would be worth on the paper, and it
is the best guide to how long an answer should be. Four marks is a short
paragraph, not a page.
\end{keybox}

\vspace{1mm}
{\small\tableofcontents}
\newpage
"""


def main() -> int:
    out = [HEAD]
    total = 0
    for n in sorted(EXERCISES):
        src = SOURCE[n]
        out.append(f"\n\\section*{{Lecture {n} \\quad {tex(TITLE[n])}}}")
        out.append(f"\\addcontentsline{{toc}}{{section}}"
                   f"{{Lecture {n} \\quad {tex(TITLE[n])}}}")
        out.append(f"\\setcounter{{exq}}{{0}}")
        out.append(f"{{\\color{{aimlmuted}}\\small {tex(src)}}}\\par\\medskip")
        for it in EXERCISES[n]:
            total += 1
            out.append(f"\\question{{{tex(it['q'])}}}{{{it['marks']}}}")
            why = "\n".join(f"  \\item {tex(w)}" for w in it["why"])
            out.append(
                "\\begin{solution}[title={Solution}]\n"
                f"{tex(it['a'])}\n"
                "\\begin{itemize}[leftmargin=*,nosep]\n" + why + "\n"
                "\\end{itemize}\n"
                "\\end{solution}")
        out.append("")
    out.append("\\end{document}\n")
    dest = ROOT / "notes" / "exercises.tex"
    dest.write_text("\n".join(out), encoding="utf-8")
    print(f"notes/exercises.tex — {total} questions across "
          f"{len(EXERCISES)} lectures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
