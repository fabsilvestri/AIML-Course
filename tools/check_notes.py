#!/usr/bin/env python3
"""
The Part V extended notes, checked against the notebooks that produce them.

    python3 tools/check_notes.py            # lectures 19-22
    python3 tools/check_notes.py 21         # one of them
    python3 tools/check_notes.py --execute  # ignore the cached runs

WHY THIS EXISTS. Lectures 19-22 are taught from `notes/lecture-NN.tex` rather
than from Geron, so for those four the notes are the PRIMARY SOURCE. They were
written from the decks, and `check_consistency` had already verified the decks
against the notebooks -- so at the moment of writing all three agreed. Nothing
kept them agreeing. Change a number in a deck and re-run every check in this
repo and they all pass while the notes, which the students are examined from,
quietly say something else.

This closes that gap, and it is deliberately the SAME check as
`check_consistency`, pointed at a different artefact: anchor on figures.json,
so that a number in the notes is considered only when it is quoting one of that
lecture's measurements, then require the notebook to have printed it. Course
metadata, page numbers, cutoffs and dataset constants are not in figures.json
and are never considered, with no exception list to maintain.

WHAT IT STRIPS BEFORE LOOKING, and why each would otherwise be noise:

  * `verbatim` blocks -- code shown in the notes is not a claim about a
    measurement, exactly as `<pre>` is not on a slide;
  * inline and display mathematics -- `\\log_2(i+1)`, `2^{rel}-1`, `d = 16` are
    structure, and the claims live in the prose and the tables;
  * LaTeX lengths and options -- `p{0.28\\textwidth}`, `[11pt,a4paper]`,
    `boxrule=0.4pt`, `\\lecture{19}`. Every one of these is a typesetting
    decision, and none is a result.

Everything after that is prose or a table cell, which is where a figure a
student would quote actually appears.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_consistency as cc                                  # noqa: E402

ROOT = cc.ROOT
NOTES = ROOT / "notes"
LECTURES = (19, 20, 21, 22)

BOLD, RED, GREEN, YELLOW, OFF = cc.BOLD, cc.RED, cc.GREEN, cc.YELLOW, cc.OFF


def prose(tex: str) -> str:
    """The parts of a .tex a student reads as a claim.

    Blanked rather than deleted, so a reported line number still points at the
    line it came from -- the same lesson `check_consistency` learned about
    <pre> blocks, and the same fix.
    """
    def blank(m):
        return "\n" * m.group(0).count("\n")

    tex = re.sub(r"\\begin\{verbatim\}.*?\\end\{verbatim\}", blank, tex, flags=re.S)
    tex = re.sub(r"\\\[.*?\\\]", blank, tex, flags=re.S)          # display maths
    tex = re.sub(r"\$[^$]*\$", " ", tex)                          # inline maths
    tex = re.sub(r"\\documentclass\[[^\]]*\]", " ", tex)
    tex = re.sub(r"\\lecture\{\d+\}", " lecture ", tex)
    tex = re.sub(r"\\notestitle\{\d+\}", " ", tex)
    tex = re.sub(r"p\{[\d.]+\\textwidth\}", " ", tex)             # column specs
    tex = re.sub(r"\\begin\{tabular\}\{[^}]*\}", " ", tex)
    tex = re.sub(r"[a-z]+=[\d.]+(pt|mm|cm|em)?", " ", tex)        # boxrule=0.4pt
    tex = re.sub(r"\\[a-zA-Z@]+", " ", tex)                       # command names
    return tex


def stated(tex_path: Path, own) -> list[tuple[int, float, str, str]]:
    """Which of this lecture's figures.json values the notes state, and where."""
    src = prose(tex_path.read_text(encoding="utf-8"))
    hits: list[tuple[int, float, str, str]] = []
    for line_no, line in enumerate(src.splitlines(), start=1):
        if not line.strip():
            continue
        for nm in cc.NUM.finditer(line):
            raw = nm.group(0).rstrip(".")
            try:
                v = float(raw.replace(",", ""))
            except ValueError:
                continue
            if cc.DURATION.match(line[nm.end(): nm.end() + 12].strip()):
                continue                              # AUTHORING 3.2a
            if cc.significant(raw) < 4:
                continue                              # too round to be a quotation
            if 1900 <= v <= 2100 and float(v).is_integer():
                continue                              # a year
            key = cc.fact_for(v, own)
            if key:
                hits.append((line_no, v, key, " ".join(line.split())[:88]))
    return hits


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("lectures", nargs="*", type=int)
    ap.add_argument("--execute", action="store_true",
                    help="re-execute even if a cached run exists")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()

    wanted = [n for n in (a.lectures or LECTURES) if n in LECTURES]
    if a.lectures and not wanted:
        print(f"{YELLOW}nothing to do{OFF} — notes exist only for "
              f"lectures {', '.join(map(str, LECTURES))}")
        return 0

    bad = 0
    for n in wanted:
        tex = NOTES / f"lecture-{n:02d}.tex"
        nb = ROOT / f"notebooks/lecture-{n:02d}.ipynb"
        if not tex.is_file():
            print(f"{RED}FAIL{OFF}  lecture {n:02d} — notes/{tex.name} missing")
            bad += 1
            continue

        own = cc.facts(n)
        if not own:
            print(f"{YELLOW}skip{OFF}  lecture {n:02d} — no figures.json "
                  f"namespace, nothing to anchor on")
            continue

        run = cc.executed(nb, a.execute)
        if run is None:
            print(f"{RED}FAIL{OFF}  lecture {n:02d} — the notebook did not run")
            bad += 1
            continue

        printed = cc.printed_numbers(run)
        hits = stated(tex, own)
        wrong = [h for h in hits
                 if not cc.matches(h[1], printed)
                 and h[2] not in cc.SCALE_ONLY
                 and f"l{n}:{h[2]}" not in cc.CROSS_LECTURE]

        if wrong:
            bad += len(wrong)
            print(f"{RED}FAIL{OFF}  lecture {n:02d} — "
                  f"{len(wrong)} of {len(hits)} stated figures are not printed "
                  f"by notebooks/lecture-{n:02d}.ipynb")
            for line, v, key, ctx in wrong[:10]:
                print(f"        notes/{tex.name}:{line}  {v:g}  ({key})")
                print(f"            …{ctx}…")
            if len(wrong) > 10:
                print(f"        … and {len(wrong) - 10} more")
        else:
            print(f"{GREEN}ok{OFF}    lecture {n:02d} — "
                  f"{len(hits)} stated figures, every one printed by its notebook")
            if a.verbose:
                for line, v, key, _ in hits:
                    print(f"        {v:g}  {key}  (line {line})")

    print()
    if bad:
        print(f"{BOLD}{RED}{bad} figure(s) in the Part V notes that no notebook "
              f"prints{OFF}")
        return 1
    print(f"{BOLD}{GREEN}the Part V notes agree with their notebooks{OFF}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
