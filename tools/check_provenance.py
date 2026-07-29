#!/usr/bin/env python3
"""
Enforce the provenance contract: every quantity on a slide comes from the data.

    python3 tools/check_provenance.py           # exits non-zero on an orphan
    python3 tools/check_provenance.py -v        # also list what matched

The course's central claim — the same standard it demands of students — is that
every number printed on a slide can be reproduced by re-running
`tools/make_figures.py`. That claim rotted once already: three code blocks and a
results table were carried over from the textbook's pipeline and sat for weeks
contradicting the prose beside them, because nothing checked.

This checks. It pulls every money amount and every thousands-separated integer
out of the decks and the site, and requires each to be reachable from
`assets/figures/figures.json` — exactly, or at a rounding a lecturer would
plausibly write ("about $68,000" for 68,573.73).

Numbers that are legitimately not measurements — durations, marks, chapter and
lecture numbers, dataset constants quoted from the book — live in ALLOWED below,
each with the reason it is there. Adding to that list is fine; adding to it
without a reason is how the contract rots again.
"""

from __future__ import annotations

import html
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FIGURES = ROOT / "assets" / "figures" / "figures.json"
# The hand-drawn diagrams too. They carry quantities — "all 20,640 rows" — and
# for most of this course's life nothing read them, so d-course-arc.svg was free
# to claim the rare-event detector "missed one five in four" when it missed
# 22.8%. This catches the numerals; a claim written out in words is still only
# caught by a person reading the diagram, which is worth knowing about.
PAGES = (sorted(ROOT.glob("slides/lecture-[0-9][0-9].html"))
         + sorted(ROOT.glob("assets/figures/d-*.svg"))
         + [ROOT / "index.html"])

# Quantities that are not measurements. Keep the reason attached.
ALLOWED: dict[float, str] = {
    # course structure
    24: "lectures in the course",
    12: "applications / mathematical threads",
    48: "academic hours",
    90: "minutes per lecture",
    45: "minutes in an academic hour",
    70: "minimum slides per deck",
    # the clock
    10: "minutes, lecture block", 15: "minutes, lecture block",
    20: "minutes, lecture block", 25: "minutes, lecture block",
    35: "minutes, lecture block", 60: "minutes, lecture block",
    # assessment
    18: "pass mark out of 30", 30: "marks available",
    40: "Part A weight, %", 25: "Part C weight, %",
    50: "written/oral split, %", 95: "confidence level, %",
    # the book
    16: "last chapter in scope",
    # dataset constants stated in the book, not derived by us
    500_001: "the price cap, quoted from the book",
    15.0001: "median_income cap, quoted from the book",
    0.4999: "median_income floor, quoted from the book",
    # Lecture 14's variance table shows what happens either side of rho = 1. The
    # row above one is marked "illustrative" on the slide itself: no network in
    # this course was initialised that badly, and running one to NaN to fill in
    # a table cell would teach nothing the row does not already say.
    1.4: "rho > 1, the illustrative row of Lecture 14's variance table",
    # Lecture 22 tabulates -log(p) at four probabilities to show why a confident
    # mistake is expensive. These are the logarithm, not a measurement of
    # anything: 0.69, 2.30 and 4.61 are -ln(1/2), -ln(1/10) and -ln(1/100).
    4.61: "-ln(0.01), Lecture 22's cross-entropy table",
}

# Tolerated roundings, as (divisor, label). The sub-dollar steps are needed for
# the quantities whose whole point is that they are tiny — the per-seed cost of
# the scale-before-split leak is -$0.02, and rounding that to the nearest dollar
# is zero.
ROUNDINGS = [(0.01, "to a cent"), (0.1, "to 10c"), (1, "exact"),
             (10, "to $10"), (100, "to $100"),
             (1_000, "to $1,000"), (10_000, "to $10,000")]


def flatten(obj) -> set[float]:
    """Every number reachable in figures.json, at any nesting depth."""
    out: set[float] = set()
    if isinstance(obj, bool):
        return out
    if isinstance(obj, (int, float)):
        out.add(float(obj))
    elif isinstance(obj, dict):
        for v in obj.values():
            out |= flatten(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            out |= flatten(v)
    return out


def derived(values: set[float]) -> set[float]:
    """Values a lecturer may legitimately write: rounded, and as percentages."""
    out = set(values)
    for v in values:
        for div, _ in ROUNDINGS:
            out.add(round(round(v / div) * div, 2))
        out.add(round(v))
        if 0 < v <= 1:                      # a ratio quoted as a percentage
            out.add(round(v * 100, 1))
            out.add(round(v * 100))
    return out


def only_text_elements(src: str) -> str:
    """Blank everything in an SVG that is not drawn as words.

    Geometry is full of pairs that look exactly like thousands-separated
    integers: `points="120,290 620,290 780,410"` is four of them. Only a `<text>`
    element is read by anybody in the room, so only a `<text>` element is held to
    the provenance contract. Blanked rather than removed, so reported line
    numbers still point at the right line.
    """
    out = list(" " * len(src))
    for m in re.finditer(r"<text\b[^>]*>(.*?)</text>", src, re.S):
        for i in range(m.start(1), m.end(1)):
            out[i] = src[i]
    for i, ch in enumerate(src):
        if ch == "\n":
            out[i] = "\n"
    return "".join(out)


def quantities(src: str, svg: bool = False):
    """Yield (line, text, value) for each money amount or 000-separated integer."""
    blank = lambda m: " " * len(m.group())
    if svg:
        src = only_text_elements(src)
    body = re.sub(r"<script.*?</script>", blank, src, flags=re.S)
    body = re.sub(r"<style.*?</style>", blank, body, flags=re.S)
    body = re.sub(r"<!--.*?-->", blank, body, flags=re.S)
    body = re.sub(r'\b(?:alt|title|content)="[^"]*"', blank, body)
    body = html.unescape(body)
    # Strip KaTeX before looking for money. A $...$ pair is maths if it holds a
    # backslash ("$7.1\times10^{-6}$") OR is a bare symbolic token with neither
    # a thousands separator nor a real word — "$0.909$", "$t^2$", "$[0,1]$".
    # Currency reads the other way: "$120,000 and " has both.
    def _is_maths(text: str) -> bool:
        inner = text[1:-1]
        if "\\" in inner:
            return True
        return not (re.search(r"\d{1,3}(,\d{3})+", inner)
                    or re.search(r"[A-Za-z]{2,}", inner))

    body = re.sub(r"(?<!\$)\$[^$\n]{1,160}?\$(?!\$)",
                  lambda m: blank(m) if _is_maths(m.group()) else m.group(), body)

    pattern = re.compile(r"\$\s?(\d[\d,]*(?:\.\d+)?)|(?<![\d.])(\d{1,3}(?:,\d{3})+)")
    for m in pattern.finditer(body):
        raw = m.group(1) or m.group(2)
        try:
            val = float(raw.replace(",", ""))
        except ValueError:
            continue
        yield body[: m.start()].count("\n") + 1, m.group().strip(), val


def flatten_keyed(obj, path: str = ""):
    """Every number in figures.json, with the dotted path that reaches it.

    `flatten` throws the keys away, which is all the money rule ever needed.
    Scoping a cell to its own lecture needs to know who owns each number.
    """
    if isinstance(obj, bool):
        return
    if isinstance(obj, (int, float)):
        yield path, float(obj)
    elif isinstance(obj, dict):
        for k, v in obj.items():
            yield from flatten_keyed(v, f"{path}.{k}" if path else str(k))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from flatten_keyed(v, f"{path}.{i}" if path else str(i))


def own_values(keyed, lecture: int) -> list:
    """The values this lecture is entitled to quote.

    Matching a cell against all 700-odd measurements in the course is how a
    check passes for the wrong reason: `353 s` found *something* within half a
    unit of 353 and sailed through, which is the same coincidence that let an
    invented `32,768` sit in Lecture 24 for days. A number on a slide belongs to
    the application that measured it, so that is the pool it is checked against.

    Lectures 1-4 predate the namespace and own the unprefixed keys.
    """
    app = (lecture + 1) // 2
    pair = lecture - 1 if lecture % 2 == 0 else lecture + 1
    own = (f"l{lecture:02d}_", f"l{pair:02d}_", f"app{app:02d}")
    out = []
    for key, v in keyed:
        head = key.split(".")[0]
        prefixed = re.match(r"l\d\d_|app\d\d", head)
        if head.startswith(own) or (not prefixed and lecture <= 4):
            out.append(v)
    return out


def cell_matches(text: str, val: float, measured) -> bool:
    """Is a displayed cell value one of the measured ones, at its own precision?

    A slide shows `96.84%` for a measurement stored as 0.9684, and `364 s` for
    364.32452. So compare at the precision the slide chose — half a unit in the
    last displayed place — and try the percentage reading as well as the bare
    one. Matching on exact equality instead reported 370 false failures, nearly
    all of them numbers that were in figures.json all along.
    """
    # Decimal places of the NUMBER, not of the cell. Counting them across the
    # whole string made "77.18 points" eight decimal places and the tolerance
    # 5e-9, so every cell carrying a unit failed.
    num = NUMERIC.search(text)
    frac = num.group().split(".")
    dp = len(frac[1]) if len(frac) > 1 else 0
    # A hair of slack: 0.96785 displays as "96.79%", and the difference is
    # exactly half a unit in the last place — which binary floating point puts
    # a whisker over the limit, failing a cell that is perfectly correct.
    tol = 0.5 * 10 ** (-dp) + 1e-9
    for v in measured:
        if not isinstance(v, (int, float)) or isinstance(v, bool):
            continue
        if abs(v - val) <= tol or abs(v * 100 - val) <= tol:
            return True
    return False


NUM_CELL = re.compile(r'<t[dh] class="[^"]*\bnum\b[^"]*">(.*?)</t[dh]>', re.S)
NUMERIC = re.compile(r"[\u2212+-]?[\d,]+(?:\.\d+)?")


def table_numbers(src: str):
    """Yield (line, text, value) for every purely numeric `class="num"` cell.

    The money-and-thousands rule reads `$120,000` and `20,640`, and is blind to
    every measurement under a thousand written without a separator — which is
    almost all of them: accuracies, seconds, points, counts. Lecture 16 quoted
    `353 s` for a wall clock that figures.json records as 364.3, in three tables,
    with two speed-ups hand-computed from it, and provenance passed.

    A `class="num"` cell is the one place a number cannot be prose. If it is in
    that column it is a measurement, so it has to be reachable — which makes
    this the cheapest strong rule available.
    """
    for m in NUM_CELL.finditer(src):
        inner = html.unescape(re.sub(r"<[^>]+>", " ", m.group(1))).strip()
        # a bare unit, a dash, or a range is not a single measurement
        if not inner or len(NUMERIC.findall(inner)) != 1:
            continue
        raw = NUMERIC.search(inner).group()
        try:
            val = float(raw.replace(",", "").replace("\u2212", "-"))
        except ValueError:
            continue
        yield src[: m.start()].count("\n") + 1, inner, val


def main() -> int:
    verbose = "-v" in sys.argv
    if not FIGURES.is_file():
        print(f"missing {FIGURES.relative_to(ROOT)} — run tools/make_figures.py")
        return 1

    measured = flatten(json.loads(FIGURES.read_text()))
    keyed = list(flatten_keyed(json.loads(FIGURES.read_text())))
    measured_values = [v for _k, v in keyed]
    reachable = derived(measured)
    allowed = set(ALLOWED)

    orphans, cell_orphans, borrowed, matched = [], [], [], 0
    for page in PAGES:
        if not page.is_file():
            continue
        rel = page.relative_to(ROOT)
        src = page.read_text()
        cells = [] if page.suffix == ".svg" else list(table_numbers(src))
        m_lec = re.search(r"lecture-(\d\d)\.html$", page.name)
        page_values = (own_values(keyed, int(m_lec.group(1)))
                       if m_lec else measured_values)
        for line, text, val in list(quantities(src, svg=page.suffix == ".svg")):
            if val in allowed or val in reachable:
                matched += 1
                if verbose:
                    why = "structural" if val in allowed else "measured"
                    print(f"  ok  {rel}:{line}  {text:>14s}  ({why})")
                continue
            orphans.append(f"{rel}:{line}: {text} is not in figures.json "
                           f"and is not a declared constant")

        for line, text, val in cells:
            if val in allowed or cell_matches(text, val, page_values):
                matched += 1
                continue
            # Quoting another lecture's measurement is legitimate and the course
            # does it deliberately — Lecture 24 revisits all twelve applications,
            # and a thread's summary table cites the lecture it came from. So a
            # number found outside this lecture's own namespace is reported, not
            # failed; only a number found nowhere at all is a failure.
            if cell_matches(text, val, measured_values):
                borrowed.append(f"{rel}:{line}: {text!r} is another lecture's "
                                f"measurement — check it is still the right one")
                matched += 1
                continue
            cell_orphans.append(f"{rel}:{line}: table cell {text!r} is not in "
                                f"figures.json and is not a declared constant")

    if borrowed and "-v" in sys.argv:
        print(f"{len(borrowed)} table cell(s) borrowed from another lecture:\n")
        for bq in borrowed:
            print("  " + bq)
        print()

    if cell_orphans:
        print(f"{len(cell_orphans)} numeric table cell(s) not traceable:\n")
        for o in cell_orphans[:60]:
            print("  " + o)
        if len(cell_orphans) > 60:
            print(f"  ... and {len(cell_orphans) - 60} more")
        print()

    if orphans:
        print(f"{len(orphans)} unprovenanced quantity(ies); {matched} verified:\n")
        for o in orphans:
            print("  " + o)
        print("\nEither export it from tools/make_figures.py, or add it to "
              "ALLOWED with the reason it is not a measurement.")
        return 1

    if cell_orphans:
        return 1
    print(f"provenance clean — {matched} quantities, all traceable")
    return 0


if __name__ == "__main__":
    sys.exit(main())
