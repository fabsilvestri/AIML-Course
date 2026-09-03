#!/usr/bin/env python3
r"""
Lint the slide decks and the course site for the mistakes that do not announce
themselves in a browser.

    python3 tools/check_decks.py          # exits non-zero if anything is found

Checks
------
1. Currency eaten by KaTeX. Two "$" in one text run are read as inline maths:
   "$120,000 and $265,000" renders as italic "120,000and265,000", losing both
   dollar signs and the spaces. Wrap each amount: <span class="usd">$120,000</span>.
2. Weekday names. Lectures refer to one another relatively, never by day.
3. Third-party notebooks. The course uses no notebook it did not write.
4. Referenced figures that do not exist, or SVG diagrams with no intrinsic
   size — a <svg> carrying only a viewBox renders at 0x0 through an <img>, so
   the slide shows nothing and every other check still passes.
5. Decks under the 70-slide minimum.
6. Raw mathtext drawn on a figure. matplotlib's log formatter emits
   `$\mathdefault{10^{3}}$`, and with `text.parse_math: False` that is drawn
   literally as an axis label. Thirteen figures shipped this way, unnoticed by
   every other check. Use `figkit.plain_log()`.
7. Leftover drafting placeholders. Three different templating syntaxes have been
   used by different authors; 432 tokens once reached the repository unreplaced,
   and would have shown as literal `@@l09_best_k@@` or `«int:l11_n_params»` on a
   projector. Whatever syntax you draft in, `tools/substitute.py` must resolve it
   before the deck ships.
"""

from __future__ import annotations

import html
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PAGES = sorted(ROOT.glob("slides/lecture-[0-9][0-9].html")) + [ROOT / "index.html"]

PLACEHOLDERS = re.compile(r"@@[^@\s]{1,80}?@@|«[^»\n]{1,80}?»")
WEEKDAYS = re.compile(r"\b(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|"
                      r"Sunday|yesterday|tomorrow)\b", re.I)
THIRD_PARTY_NB = re.compile(r"homl\.info|github\.com/ageron/handson", re.I)


def text_runs(src: str):
    """Yield (line, text) for each run of rendered text between tags.

    <pre>, <script>, <style> and comments are blanked — none of them is prose,
    and the KaTeX delimiter config inside <script> otherwise trips check 1.

    Attributes are blanked rather than removed so line numbers stay true.
    """
    blank = lambda m: " " * len(m.group())
    body = re.sub(r"<pre.*?</pre>", blank, src, flags=re.S)
    body = re.sub(r"<script.*?</script>", blank, body, flags=re.S)
    body = re.sub(r"<style.*?</style>", blank, body, flags=re.S)
    body = re.sub(r"<!--.*?-->", blank, body, flags=re.S)
    body = re.sub(r"<(\w+)([^>]*)>",
                  lambda m: "<" + m.group(1) + " " * len(m.group(2)) + ">", body)
    for m in re.finditer(r">([^<]+)<", body):
        # decode entities: "&lt;" would otherwise read as the word "lt"
        yield body[: m.start()].count("\n") + 1, html.unescape(m.group(1))


def eaten_currency(run: str) -> str | None:
    """Return the offending $...$ pair if KaTeX would swallow prose currency.

    A pair is maths if it contains a backslash or is a short symbolic token
    ("r", "-1", "1.0", "[0,1]", "O(n^{3})"). It is currency-eaten-as-maths when
    the captured text carries a thousands separator, ends mid-sentence, or
    contains a real word — none of which appear in inline maths.
    """
    for content in re.findall(r"(?<!\$)\$([^$\n]{1,160}?)\$(?!\$)", run):
        if "\\" in content:
            continue
        thousands = re.search(r"\d{1,3}(,\d{3})+", content)
        word = re.search(r"[A-Za-z]{3,}", content)
        dangling = content != content.strip()
        # A relational or superscript operator means maths, whatever else is
        # there: "$2wx = $" is an expression, not $2 followed by prose. Dashes
        # and parentheses are excluded — they are common in prose.
        operator = re.search(r"[=^_<>≤≥≠]", content)
        if operator:
            continue
        if thousands or word or dangling:
            return content
    return None


def unbalanced_math(run: str) -> str | None:
    """Return the offending text if `$` opens inline maths and never closes it.

    `eaten_currency` above asks whether a *closed* pair should have been
    currency. It cannot see the opposite mistake -- a `$` that was never closed
    -- and it skips the very case that hides one: a run containing `=` is taken
    for maths and returned early. Lecture 10 shipped
    `<li>The kneedle rule picked $k = 19</li>` on that path, and it rendered as
    a literal dollar and `k = 19` on the projector for a whole lecture.

    A leftover `$` is not itself the bug -- `<span class="usd">$500,001</span>`
    is one on purpose, and KaTeX leaves an unpaired delimiter alone. So pair
    the delimiters the way KaTeX does, then look at what is left: a leftover
    followed by an amount is currency, and a leftover followed by anything else
    is maths that will not render.
    """
    rest = re.sub(r"\$\$[^$]*\$\$", "", run)          # display pairs first
    # Inline pairs may span source lines -- the newline is just whitespace to
    # KaTeX -- so this must NOT exclude \n, or every multi-line $...$ reads as
    # two loose delimiters. Non-greedy, so "$12,500 and $350,000" pairs at the
    # first closer and leaves nothing dangling.
    rest = re.sub(r"\$[^$]*?\$", "", rest, flags=re.S)  # then inline pairs
    for m in re.finditer(r"\$(.*)", rest):
        tail = m.group(1)
        if re.match(r"\s?\d[\d,]*(?:\.\d+)?", tail):  # an amount: currency
            continue
        return ("$" + tail).strip()[:70]
    return None


def check(path: Path) -> list[str]:
    src = path.read_text()
    rel = path.relative_to(ROOT)
    out: list[str] = []

    for line, run in text_runs(src):
        # 1. currency pairing
        if (bad := eaten_currency(run)):
            out.append(f"{rel}:{line}: KaTeX will eat this as maths — wrap the "
                       f"amount in <span class=\"usd\">: ${' '.join(bad.split())[:60]}$")
        # 1b. an inline $ that was never closed
        if (bad := unbalanced_math(run)):
            out.append(f"{rel}:{line}: inline $ never closed — KaTeX renders "
                       f"the rest as literal text: {bad}")
        # 2. weekdays
        if (m := WEEKDAYS.search(run)):
            out.append(f"{rel}:{line}: names a weekday ({m.group()}) — "
                       f"use 'in the next lecture' / 'in two lectures'")

    # 3. leftover placeholders
    for m in PLACEHOLDERS.finditer(src):
        out.append(f"{rel}:{src[:m.start()].count(chr(10)) + 1}: unsubstituted "
                   f"placeholder {m.group()} — run tools/substitute.py")

    # 3b. third-party notebooks
    for m in THIRD_PARTY_NB.finditer(src):
        out.append(f"{rel}:{src[:m.start()].count(chr(10)) + 1}: "
                   f"third-party notebook reference ({m.group()})")

    # 4b. raw mathtext baked into a figure
    for m in re.finditer(r'src="([^"]*assets/figures/[^"]+\.svg)"', src):
        target = (path.parent / m.group(1)).resolve()
        if target.exists() and "mathdefault" in target.read_text():
            out.append(f"{rel}: {target.name} draws raw LaTeX as an axis label "
                       f"($\\mathdefault{{...}}$) — use figkit.plain_log()")

    # 4. missing figures, and figures the browser cannot size
    for m in re.finditer(r'src="([^"]*assets/figures/[^"]+)"', src):
        target = (path.parent / m.group(1)).resolve()
        if not target.exists():
            out.append(f"{rel}: missing figure {m.group(1)}")
        elif target.suffix == ".svg":
            tag = re.search(r"<svg\b[^>]*>", target.read_text()[:800])
            if tag and not re.search(r'\bwidth="', tag.group()):
                out.append(f"{rel}: {target.name} has no width/height — a "
                           f"viewBox alone is not an intrinsic size for an "
                           f"<img>, and it renders blank")

    # 5. deck length
    if path.parent.name == "slides":
        n = len(re.findall(r"<section", src))
        if n < 70:
            out.append(f"{rel}: {n} slides — below the 70-slide minimum")

    return out


def check_site_index() -> list[str]:
    """Every lecture must appear on the site, and its state must be honest.

    index.html was written while the course was half-built, so eleven cards said
    "in preparation" long after the lecture existed — and nothing noticed,
    because a card with no link is not a broken link. The absence of a link is
    exactly what has to be checked.

    Since the 2026-09 redesign the page is generated by tools/make_site.py, and
    a lecture legitimately shows "In preparation" until its deck is converted.
    So the check is now three things, all of which can fail silently:

      * every lecture appears on the page at all;
      * a lecture the page LINKS to has the files it links to; and
      * a lecture the page calls "In preparation" does not, in fact, already
        have a converted deck sitting in the repo — which is the failure mode
        that lost eleven lectures last time, in the other direction.
    """
    out: list[str] = []
    src = (ROOT / "index.html").read_text()
    published = {n for n in range(1, 25)
                 if f'href="slides/lecture-{n:02d}.html"' in src}

    for n in range(1, 25):
        if f'<span class="n">{n:02d}</span>' not in src:
            out.append(f"index.html: lecture {n:02d} is not on the page at all")
            continue
        if n in published:
            if not (ROOT / f"slides/lecture-{n:02d}.html").exists():
                out.append(f"index.html links to lecture {n:02d}'s slides, "
                           f"which do not exist")
            if f"notebooks/lecture-{n:02d}.ipynb" not in src:
                out.append(f"index.html links lecture {n:02d}'s slides but not "
                           f"its notebook")
        # Part V's extended notes. The page links a PDF for lectures 19-22 and
        # those PDFs are build artefacts of notes/*.tex, so they can go missing
        # exactly the way a converted deck can -- and a dead link to the
        # PRIMARY SOURCE of an examinable lecture is worse than a dead link to
        # a supplement.
        if f'href="notes/lecture-{n:02d}.pdf"' in src:
            if not (ROOT / f"notes/lecture-{n:02d}.pdf").exists():
                out.append(f"index.html links lecture {n:02d}'s notes PDF, "
                           f"which does not exist — run make -C notes")

    if 'class="pending"' in src:
        out.append('index.html: still says "in preparation" somewhere')
    return out


def check_type_scale() -> list[str]:
    """No bare font-size in the stylesheet.

    The deck declared a seven-step type scale and then set size with a bare
    number in thirteen more rules, so "what sizes does this deck use?" had no
    answer you could read anywhere — you had to grep, and grep found 0.86em
    next to 0.88em next to 0.9em with nothing to say whether that was three
    decisions or one accident.

    Every size is a named token now. The two absolute pixel values are the
    30px root that every em is relative to, and the footer, which is chrome
    outside .reveal and must not scale with the slide. Both say so where they
    are declared.
    """
    import re
    css = (ROOT / "assets" / "css" / "custom.css").read_text()
    bad = []
    for m in re.finditer(r"font-size:\s*([^;\n]+)", css):
        v = m.group(1).strip()
        if v.startswith("var(--t-") or v == "inherit":
            continue
        if v in ("30px", "15px"):          # documented absolutes; see the CSS
            continue
        line = css[:m.start()].count("\n") + 1
        bad.append(f"assets/css/custom.css:{line}: bare font-size {v} — "
                   f"use a --t-* token, or declare one with its reason")
    return bad


def main() -> int:
    problems: list[str] = []
    for page in PAGES:
        if page.exists():
            problems += check(page)
    problems += check_site_index()
    problems += check_type_scale()

    if problems:
        print(f"{len(problems)} problem(s):\n")
        for p in problems:
            print("  " + p)
        return 1

    print(f"clean — {len(PAGES)} page(s) checked")
    return 0


if __name__ == "__main__":
    sys.exit(main())
