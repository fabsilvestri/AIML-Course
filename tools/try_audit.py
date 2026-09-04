#!/usr/bin/env python3
"""
Extract and triage every `try` field in the course.

    python3 tools/try_audit.py               # counts by class
    python3 tools/try_audit.py --assert      # the ones predicting an assert
    python3 tools/try_audit.py --number      # the ones stating a figure
    python3 tools/try_audit.py --lecture 14  # one lecture, with its code

WHY. `check_notebooks §4.1a` verifies that a prompt box HAS a `try`. Nothing
verifies that what the `try` predicts is TRUE. A `try` saying "the assert on
52,326 fires" is a claim a student will test, and if it is wrong they lose an
afternoon and their trust in the notebook.

The audit that uses this is recorded in REBUILD.md § "The try-field audit".
Two false claims were found in the first seventeen tested (both in Lecture 8),
so the base rate is not negligible and the audit is not optional.

Classes, in rising order of how badly a wrong one hurts:

  question     ends in a question or asks the reader to work something out.
               Cannot be false; nothing to verify.
  qualitative  states a direction ("accuracy falls") with no figure. Falsifiable
               but cheap to be roughly right about.
  number       states a specific figure. A student checks these.
  assert       predicts that an assertion fires, or that something raises.
               The highest-value class: if it is wrong, the student sees
               silence where they were promised a failure.
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

ROOT = pathlib.Path(__file__).resolve().parent.parent

NUM = re.compile(r"\d[\d,]*(?:\.\d+)?")
ASSERTY = re.compile(r"\bassert\b|\bfires\b|\braises\b|\bcrash", re.I)
QUESTION = re.compile(r"\?\s*$|which |what |why |how |does |where |say why|"
                      r"work out|find the", re.I)


def tries():
    """(lecture, label, try text, the code cell it talks about)."""
    for p in sorted((ROOT / "notebooks").glob("lecture-*.ipynb")):
        n = int(re.search(r"(\d+)", p.name).group(1))
        cells = json.loads(p.read_text())["cells"]
        for i, c in enumerate(cells):
            s = "".join(c["source"])
            if c["cell_type"] != "markdown" or "**try**" not in s:
                continue
            m = re.search(r"\*\*Prompt(?: · ([^*]+))?\*\*", s)
            label = (m.group(1) or "").strip() if m else ""
            text = s.split("**try** ·", 1)[1].strip().lstrip("· ").strip()
            code = ""
            for j in range(i + 1, min(i + 3, len(cells))):
                if cells[j]["cell_type"] == "code":
                    code = "".join(cells[j]["source"])
                    break
            yield n, label, text, code


def classify(text: str) -> list[str]:
    out = []
    if ASSERTY.search(text):
        out.append("assert")
    if [m for m in NUM.finditer(text)
            if len(m.group().replace(",", "")) >= 3]:
        out.append("number")
    if QUESTION.search(text):
        out.append("question")
    return out or ["qualitative"]



# --------------------------------------------------------------- numbers
# Every `try` that states a figure is a claim a student will check. Most of
# those figures should already appear in the lecture's own executed output --
# "the assert on 52,326 fires", "it costs 3,840 weights". Where one does not,
# it is either arithmetic derived from the cell (fine, and listed so a human
# can confirm it) or a number nobody has ever checked (the thing this finds).
#
# Deliberately the same rule as check_consistency: reuse its executed-notebook
# cache and its rounding tolerance, so there is one notion of "the notebook
# printed this" in the repo rather than two.
def check_numbers(lectures=None, execute=False):
    import check_consistency as cc
    NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")
    rows = [r for r in tries() if not lectures or r[0] in lectures]
    by_lec = {}
    for n, label, text, code in rows:
        by_lec.setdefault(n, []).append((label, text, code))

    unmatched, total = [], 0
    for n in sorted(by_lec):
        nb = ROOT / f"notebooks/lecture-{n:02d}.ipynb"
        run = cc.executed(nb, execute)
        if run is None:
            print(f"  lecture {n:02d}: notebook did not execute; skipped")
            continue
        printed = cc.printed_numbers(run)
        for label, text, code in by_lec[n]:
            for m in NUMBER.finditer(text):
                raw = m.group(0).rstrip(".")
                if cc.significant(raw) < 3:
                    continue                     # too round to be a quotation
                v = float(raw.replace(",", ""))
                if 1900 <= v <= 2100 and float(v).is_integer():
                    continue                     # a year
                total += 1
                # in the notebook's output, or written in the cell it talks about
                if cc.matches(v, printed) or raw in code or raw.replace(",", "") in code:
                    continue
                ctx = " ".join(text.split())
                i = ctx.find(raw)
                unmatched.append((n, label, raw, ctx[max(0, i-40):i+45]))
    return total, unmatched


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--assert", dest="asserty", action="store_true")
    ap.add_argument("--number", action="store_true")
    ap.add_argument("--lecture", type=int)
    ap.add_argument("--code", type=int, default=0,
                    help="lines of the code cell to print beside each")
    ap.add_argument("--check-numbers", action="store_true",
                    help="every figure a try states, against the notebook's output")
    ap.add_argument("--execute", action="store_true")
    a = ap.parse_args()

    if a.check_numbers:
        lec = [a.lecture] if a.lecture else None
        total, bad = check_numbers(lec, a.execute)
        for n, label, raw, ctx in bad:
            print(f"L{n:02d}  {raw:>12}   …{ctx}…")
            print(f"          in: {label[:70]}")
        print(f"\n{len(bad)} of {total} stated figures are not in the "
              f"notebook's output nor in the cell")
        return 1 if bad else 0

    rows = list(tries())
    sel = rows
    if a.lecture:
        sel = [r for r in rows if r[0] == a.lecture]
        a.code = a.code or 14
    if a.asserty:
        sel = [r for r in sel if "assert" in classify(r[2])]
    if a.number:
        sel = [r for r in sel if "number" in classify(r[2])]

    if a.asserty or a.number or a.lecture:
        for n, label, text, code in sel:
            print(f"\n--- L{n:02d} · {label[:60]}")
            print(f"    {text}")
            for ln in code.splitlines()[:a.code]:
                print(f"      | {ln}")
        print(f"\n{len(sel)} of {len(rows)}")
        return 0

    k = collections.Counter(tuple(sorted(classify(t))) for _, _, t, _ in rows)
    print(f"{len(rows)} try fields")
    for key, v in k.most_common():
        print(f"  {'+'.join(key):28s} {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
