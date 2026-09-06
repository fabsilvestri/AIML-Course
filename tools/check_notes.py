#!/usr/bin/env python3
"""
The extended lecture notes, checked against the notebooks that produce them.

    python3 tools/check_notes.py            # every lecture that has notes
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
import html
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_consistency as cc                                  # noqa: E402

ROOT = cc.ROOT
NOTES = ROOT / "notes"
LECTURES = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22)

BOLD, RED, GREEN, YELLOW, OFF = cc.BOLD, cc.RED, cc.GREEN, cc.YELLOW, cc.OFF


def prose(tex: str) -> str:
    """The parts of a .tex a student reads as a claim.

    Blanked rather than deleted, so a reported line number still points at the
    line it came from -- the same lesson `check_consistency` learned about
    <pre> blocks, and the same fix.
    """
    def blank(m):
        return "\n" * m.group(0).count("\n")

    # LaTeX writes a thousands separator as {,} -- 48{,}180 -- so that the
    # comma does not take sentence spacing. Left alone, cc.NUM sees "48" and
    # "180" and every grouped integer in every set of notes is invisible to
    # this check. Lecture 2's notes quote forty-odd of them and scored "0
    # stated figures, every one printed": a pass with nothing behind it.
    tex = tex.replace("{,}", ",").replace("\\,", "")

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


def deck_numbers(n: int) -> list[float]:
    """Every number the lecture's deck states, as floats."""
    src = (ROOT / "slides" / f"lecture-{n:02d}.html").read_text(encoding="utf-8")
    body = src.split('<div class="slides">', 1)[-1]
    body = re.sub(r"<aside class=\"notes\">.*?</aside>", " ", body, flags=re.S)
    text = html.unescape(re.sub(r"<[^>]+>", " ", body))
    out = []
    for tok in re.findall(r"\d+\.\d+|\d{2,}", text.replace(",", "")):
        try:
            out.append(float(tok))
        except ValueError:
            pass
    return out


def untranscribed(tex_path: Path, n: int) -> list[tuple[int, str, str]]:
    """Numbers in the notes that this lecture's deck does not state.

    stated() only considers a number when it is a value in figures.json AND
    carries four significant digits -- cc.significant() drops rounder ones as
    too coarse to be a quotation. On a lecture whose figures are mostly quoted
    to three (lecture 4: 0.496, 0.468, 83.6) that leaves nearly all of the
    prose unexamined, and a mistyped digit there would reach the students.

    The decks are already verified against the notebooks by check_consistency,
    so they are a sound second anchor: a number in the notes that appears on no
    slide of its own lecture was typed, not measured.
    """
    deck = deck_numbers(n)
    bad: list[tuple[int, str, str]] = []
    for line_no, line in enumerate(prose(tex_path.read_text(encoding="utf-8"))
                                   .splitlines(), start=1):
        flat = line.replace("{,}", "").replace(",", "")
        for raw in re.findall(r"\d+\.\d+|\d{2,}", flat):
            if len(raw.replace(".", "").lstrip("0")) < 3:
                continue                       # too short to be a quotation
            if re.fullmatch(rf"{n}\.\d+", raw):
                continue                       # "Exercise 19.2" is a label
            dec = len(raw.split(".")[1]) if "." in raw else 0
            want = float(raw)
            if any(round(d, dec) == want for d in deck):
                continue
            bad.append((line_no, raw, " ".join(line.split())[:76]))
    return bad


def stated(tex_path: Path, own) -> list[tuple[int, float, str, str, int]]:
    """Which of this lecture's figures.json values the notes state, and where.

    The decimal count travels with each hit. check_consistency learned in round
    5 that a fixed precision is the wrong instrument -- a note that writes
    "6.9733" is claiming four decimals of a printed 6.973289, and checking it at
    two would accept a notebook printing 6.97 and nothing more. This check was
    still passing everything through cc.matches() at the default, so it was the
    laxer of the two on exactly the artefact students are examined from.
    """
    src = prose(tex_path.read_text(encoding="utf-8"))
    hits: list[tuple[int, float, str, str, int]] = []
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
                dec = len(raw.split(".")[1]) if "." in raw else 0
                hits.append((line_no, v, key, " ".join(line.split())[:88], dec))
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
        if not hits:
            # The same failure check_consistency guards against: figures are
            # available, the notes quote none of them precisely enough to
            # check, and the check reports ok for having compared nothing.
            # An empty selection must fail, not pass.
            print(f"{RED}FAIL{OFF}  lecture {n:02d} — {len(own)} figures "
                  f"available in {cc.NAMESPACES[n]} and the notes quote none "
                  f"of them precisely enough to check, so nothing was checked.")
            bad += 1
            continue
        def excused(h) -> bool:
            """SCALE_ONLY and CROSS_LECTURE, addressed the way check_consistency
            addresses them.

            This built the lookup key as f"l{n}:{h[2]}" -- unpadded, and with
            the leading slash figures.json keys carry. check_consistency uses
            f"l{n:02d}:" against a path with the slash stripped, which is the
            convention every entry in the dict is written in. So no
            CROSS_LECTURE entry could ever match here, and the exemption has
            been silently dead for as long as this check has existed. Notes
            19-22 never needed one, which is why nothing said so.
            """
            path = h[2].lstrip("/")
            root = path.split("/")[0].split("[")[0]
            if root in cc.SCALE_ONLY:
                return True
            return any(c.startswith(f"l{n:02d}:")
                       and path.startswith(c.split(":", 1)[1])
                       for c in cc.CROSS_LECTURE)

        wrong = [h for h in hits
                 if not cc.matches(h[1], printed, h[4]) and not excused(h)]

        if wrong:
            bad += len(wrong)
            print(f"{RED}FAIL{OFF}  lecture {n:02d} — "
                  f"{len(wrong)} of {len(hits)} stated figures are not printed "
                  f"by notebooks/lecture-{n:02d}.ipynb")
            for line, v, key, ctx, _dec in wrong[:10]:
                print(f"        notes/{tex.name}:{line}  {v:g}  ({key})")
                print(f"            …{ctx}…")
            if len(wrong) > 10:
                print(f"        … and {len(wrong) - 10} more")
        loose = untranscribed(tex, n)
        if loose:
            bad += len(loose)
            print(f"{RED}FAIL{OFF}  lecture {n:02d} — {len(loose)} number(s) in "
                  f"the notes that deck {n:02d} does not state")
            for line, raw, ctx in loose[:8]:
                print(f"        notes/{tex.name}:{line}  {raw}")
                print(f"            …{ctx}…")
            if len(loose) > 8:
                print(f"        … and {len(loose) - 8} more")
        elif not wrong:
            print(f"{GREEN}ok{OFF}    lecture {n:02d} — "
                  f"{len(hits)} stated figures, every one printed by its notebook"
                  f"; every number also appears on the deck")
            if a.verbose:
                for line, v, key, _, _dec in hits:
                    print(f"        {v:g}  {key}  (line {line})")

    print()
    if bad:
        print(f"{BOLD}{RED}{bad} figure(s) in the lecture notes that no notebook "
              f"prints{OFF}")
        return 1
    print(f"{BOLD}{GREEN}the lecture notes agree with their notebooks{OFF}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
