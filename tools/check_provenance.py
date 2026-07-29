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
PAGES = sorted(ROOT.glob("slides/lecture-[0-9][0-9].html")) + [ROOT / "index.html"]

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


def quantities(src: str):
    """Yield (line, text, value) for each money amount or 000-separated integer."""
    blank = lambda m: " " * len(m.group())
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


def main() -> int:
    verbose = "-v" in sys.argv
    if not FIGURES.is_file():
        print(f"missing {FIGURES.relative_to(ROOT)} — run tools/make_figures.py")
        return 1

    measured = flatten(json.loads(FIGURES.read_text()))
    reachable = derived(measured)
    allowed = set(ALLOWED)

    orphans, matched = [], 0
    for page in PAGES:
        if not page.is_file():
            continue
        rel = page.relative_to(ROOT)
        for line, text, val in quantities(page.read_text()):
            if val in allowed or val in reachable:
                matched += 1
                if verbose:
                    why = "structural" if val in allowed else "measured"
                    print(f"  ok  {rel}:{line}  {text:>14s}  ({why})")
                continue
            orphans.append(f"{rel}:{line}: {text} is not in figures.json "
                           f"and is not a declared constant")

    if orphans:
        print(f"{len(orphans)} unprovenanced quantity(ies); {matched} verified:\n")
        for o in orphans:
            print("  " + o)
        print("\nEither export it from tools/make_figures.py, or add it to "
              "ALLOWED with the reason it is not a measurement.")
        return 1

    print(f"provenance clean — {matched} quantities, all traceable")
    return 0


if __name__ == "__main__":
    sys.exit(main())
