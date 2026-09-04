#!/usr/bin/env python3
"""The examination rule is stated in five places. Keep them the same one.

    python3 tools/check_assessment.py

The rule as of 2026-09-04:

    written    marked out of 30, pass at 18, capped at 27 on its own
    oral       optional, three questions drawn from the published bank
    arithmetic the oral moves the written mark by at most +/-3, floor 18
    binding    registered after the written mark is seen, and final

It appears on the site, on deck 1, on deck 24, in the exercise book's preface,
and in the notes. A rule restated in five places is a rule that drifts, and the
failure is silent: every one of those pages renders perfectly while telling a
student something different about their own grade.

Three things are checked.

1. No sentence anywhere still states the old rule (50/50, both parts passed
   independently, an oral on any topic).
2. Every page that states the rule states all of it -- the cap, the swing, the
   pass mark, and that the oral is optional. Half the rule is worse than none:
   "the oral can lower your mark" without "27 is the cap" reads as a threat
   with no upside.
3. The size of the bank quoted in the prose is the size of the actual bank.
   Add a lecture and the number moves; this is the only check that would say so.
"""
from __future__ import annotations

import html
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RED, GREEN, BOLD, OFF = "\033[31m", "\033[32m", "\033[1m", "\033[0m"

# Phrases that can only mean the superseded rule. Deliberately specific: deck
# 17 says "under a 50/50 base rate" about class balance, which is not this.
STALE = [
    (r"final mark is (?:the|their) average",   "the 50/50 average"),
    (r"passed independently",                  "both parts passed independently"),
    (r"18/30 on each",                         "18/30 on each part"),
    (r"half the final mark",                   "the paper as half the mark"),
    (r"strong written cannot rescue",          "the old compensation rule"),
    (r"cannot compensate for a failing oral",  "the old compensation rule"),
    (r"any topic from the course",             "the unbounded oral"),
    (r"most-asked oral question",              "a frequency claim about the oral"),
    (r"at the oral I will put a cell",         "the promise of a universal oral"),
    (r"twenty to twenty-five minutes",         "the old oral length"),
]

# Each page that states the rule must state all four parts of it.
REQUIRED = {
    "index.html": None,
    "slides/lecture-01.html": None,
    "slides/lecture-24.html": None,
}
PARTS = [
    (r"\b27\b",                     "the cap at 27"),
    (r"(?:&plusmn;|±)\s*3",    "the +/-3 swing"),
    (r"\b18\b",                     "the pass mark of 18"),
    (r"\boptional\b",               "that the oral is optional"),
]

# Files that quote the size of the bank, and must quote it correctly.
QUOTES_BANK = [
    "index.html",
    "slides/lecture-01.html",
    "slides/lecture-24.html",
    "tools/make_exercise_book.py",
]


def prose(path: Path) -> str:
    """Rendered text: tags stripped, entities kept (the rule is written with
    &plusmn; and &mdash;, and stripping those would hide the swing)."""
    s = path.read_text(encoding="utf-8")
    if path.suffix in {".html"}:
        s = re.sub(r"<[^>]+>", " ", s)
    return " ".join(s.split())


def assessment_text(path: Path) -> str:
    """The part of a page that talks about the examination."""
    s = prose(path)
    keys = ("Assessment", "How the mark is made", "Written examination",
            "Written &mdash; two hours", "Written — two hours")
    for k in keys:
        i = s.find(k)
        if i > -1:
            return s[i:]
    return s


def bank_size() -> int:
    sys.path.insert(0, str(ROOT / "tools"))
    from make_exercises import EXERCISES
    return sum(len(v) for v in EXERCISES.values())


def main() -> int:
    fails = 0
    n = bank_size()

    scan = ([ROOT / "index.html", ROOT / "README.md", ROOT / "LECTURES.md",
             ROOT / "AUTHORING.md"]
            + sorted((ROOT / "slides").glob("lecture-*.html"))
            + sorted((ROOT / "notes").glob("*.tex")))
    for path in scan:
        text = prose(path)
        for pattern, what in STALE:
            if re.search(pattern, text, re.I):
                rel = path.relative_to(ROOT)
                print(f"{RED}FAIL{OFF}  {rel} still states {what}")
                fails += 1

    for rel in REQUIRED:
        text = assessment_text(ROOT / rel)
        missing = [what for pattern, what in PARTS
                   if not re.search(pattern, text, re.I)]
        if missing:
            print(f"{RED}FAIL{OFF}  {rel} states the rule without "
                  f"{', '.join(missing)}")
            fails += 1
        else:
            print(f"{GREEN}ok{OFF}    {rel} — cap, swing, pass mark, optional")

    # Presence is not agreement. These pull the actual numbers out of the
    # prose and require every page to say the same one -- the drift that
    # matters is not a missing cap, it is one page saying 27 and another 28.
    VALUES = [
        # Anchored to exam vocabulary on purpose: "capped at 15" alone is the
        # median-income column of the housing data, in three other places.
        (r"(?:written|mark)[^.]{0,60}?capped at (?:a )?"
         r"(?:<[^>]+>\s*)?(\d+)|"
         r"capped at (?:<[^>]+>\s*)?(\d+)[^.]{0,40}?on its own|"
         r"up to (\d+) on its own", 27, "the cap"),
        (r"at most (?:&plusmn;|±)\s*(\d+)", 3, "the swing"),
        (r"pass at (\d+)", 18, "the pass mark"),
    ]
    for path in scan:
        text = prose(path)
        for pattern, want, what in VALUES:
            got = {int(g) for m in re.finditer(pattern, text, re.I)
                   for g in m.groups() if g}
            if got - {want}:
                rel = path.relative_to(ROOT)
                print(f"{RED}FAIL{OFF}  {rel} states {what} as "
                      f"{sorted(got - {want})}, not {want}")
                fails += 1

    for rel in QUOTES_BANK:
        text = html.unescape(prose(ROOT / rel))
        quoted = {int(m) for m in
                  re.findall(r"\b(\d{2,4})\b(?=[^.]{0,60}?exercises)", text)}
        quoted |= {int(m) for m in
                   re.findall(r"(?:these|of the)\s+(\d{2,4})\b", text)}
        wrong = {q for q in quoted if 20 <= q <= 999 and q != n}
        if n not in quoted:
            print(f"{RED}FAIL{OFF}  {rel} never quotes the bank size ({n})")
            fails += 1
        elif wrong:
            print(f"{RED}FAIL{OFF}  {rel} quotes {sorted(wrong)} for a bank "
                  f"of {n}")
            fails += 1
        else:
            print(f"{GREEN}ok{OFF}    {rel} — bank quoted as {n}")

    print()
    if fails:
        print(f"{BOLD}{RED}{fails} assessment inconsistenc"
              f"{'y' if fails == 1 else 'ies'}{OFF}")
        return 1
    print(f"{BOLD}{GREEN}every page states the same examination rule, "
          f"and the bank is {n}{OFF}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
