#!/usr/bin/env python3
"""
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
6. Leftover drafting placeholders. Three different templating syntaxes have been
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
PAGES = sorted(ROOT.glob("slides/*.html")) + [ROOT / "index.html"]

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


def check(path: Path) -> list[str]:
    src = path.read_text()
    rel = path.relative_to(ROOT)
    out: list[str] = []

    for line, run in text_runs(src):
        # 1. currency pairing
        if (bad := eaten_currency(run)):
            out.append(f"{rel}:{line}: KaTeX will eat this as maths — wrap the "
                       f"amount in <span class=\"usd\">: ${' '.join(bad.split())[:60]}$")
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


def main() -> int:
    problems: list[str] = []
    for page in PAGES:
        if page.exists():
            problems += check(page)

    if problems:
        print(f"{len(problems)} problem(s):\n")
        for p in problems:
            print("  " + p)
        return 1

    print(f"clean — {len(PAGES)} page(s) checked")
    return 0


if __name__ == "__main__":
    sys.exit(main())
