#!/usr/bin/env python3
"""
Run every check, in the order that fails cheapest first.

    python3 tools/check_all.py            # the fast ones (seconds)
    python3 tools/check_all.py --full     # plus the browser and notebook checks

There are now seven, written at seven different times, and nobody was going to
remember all seven. Two defects shipped in this course precisely because a check
existed and was not run: the fonts embedded in the diagrams, and the eleven
lecture cards on the site that still said "in preparation".

Ordering is deliberate — grep-speed checks first, then the ones that start
Chrome, then the one that executes 24 notebooks. A failure should cost you the
least time it can.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

FAST = [
    ("decks",       ["tools/check_decks.py"],
     "weekdays, currency eaten by KaTeX, missing figures, leftover placeholders, "
     "raw mathtext, and every link the site owes"),
    ("provenance",  ["tools/check_provenance.py"],
     "every quoted quantity traceable to figures.json"),
    ("fonts",       ["tools/embed_diagram_fonts.py", "--check"],
     "each hand-drawn diagram carries the typeface it names"),
    ("notebooks",   ["tools/make_notebooks.py"],
     "all 24 build and every code cell compiles"),
    # Cheap, and it catches what compiling each cell cannot: a cell that reads a
    # global no earlier cell defines. Six of those in one pass, all from
    # splitting or reordering older modules.
    ("names",       ["tools/check_names.py"],
     "every name a notebook uses is defined before it is used"),
]

FULL = [
    # The one that finds things. Every FAST check verifies an artefact against
    # itself; this is the only one that compares a deck with its notebook, and
    # it has caught a real defect in every lecture converted so far. It is here
    # rather than in FAST only because it executes notebooks -- though it caches
    # them by content hash, so a repeat run costs seconds.
    ("consistency", ["tools/check_consistency.py"],
     "every slide figure is one its own notebook prints  [executes notebooks]"),
    ("overflow",    ["tools/check_overflow.py"],
     "nothing taller than the canvas or off its side  [Chrome]"),
    ("diagrams",    ["tools/check_diagrams.py"],
     "every label inside its frame, none overlapping or struck  [Chrome]"),
    ("execution",   ["tools/make_notebooks.py", "--check"],
     "all 24 notebooks execute clean  [slow: hours]"),
]


def run(name: str, cmd: list[str], blurb: str) -> bool:
    print(f"\n\033[1m{name}\033[0m — {blurb}")
    t0 = time.time()
    r = subprocess.run([sys.executable, *cmd], cwd=ROOT,
                       capture_output=True, text=True)
    took = time.time() - t0
    tail = [ln for ln in r.stdout.splitlines() if ln.strip()]
    if r.returncode == 0:
        print(f"  ok ({took:.1f}s)  {tail[-1].strip() if tail else ''}")
        return True
    print(f"  FAILED ({took:.1f}s)")
    for ln in tail[-25:]:
        print("  " + ln)
    if r.stderr.strip():
        print("  " + r.stderr.strip().splitlines()[-1])
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true",
                    help="also run the browser checks and execute the notebooks")
    args = ap.parse_args()

    checks = FAST + (FULL if args.full else [])
    failed = [name for name, cmd, blurb in checks if not run(name, cmd, blurb)]

    print()
    if failed:
        print(f"\033[1m{len(failed)} of {len(checks)} failed:\033[0m "
              f"{', '.join(failed)}")
        return 1
    print(f"\033[1mall {len(checks)} checks clean\033[0m"
          + ("" if args.full else "  —  run with --full for the browser checks"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
