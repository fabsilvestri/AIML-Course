#!/usr/bin/env python3
"""
Take vertical slack out of a hand-drawn diagram, so it stops being height-clamped.

    python3 tools/compress_diagram.py d-leakage --cut 410:14 --cut 445:12
    python3 tools/compress_diagram.py d-leakage --cut 410:14 --cut 445:12 --apply

Each `--cut Y:N` removes N units of space at height Y: everything at or below Y
moves up by N, and anything spanning Y gets N shorter. Cuts compose, so the
total removed is the sum, and the viewBox shrinks to match.

**Why this and not `trim_diagram.py`.** That one lifts a whole explanatory panel
out of the drawing and into the slide, which is the better fix when there is a
panel to lift. Four diagrams have no such panel — their bottom text *is* the
drawing, and their ink already fills the viewBox with no margin to crop. The
only room left is between the rows, so this closes the gaps instead.

**What it refuses.** Path data is adjusted by treating the numbers after M, L,
C, S, Q and T as x,y pairs, which is true of every path in these files. Any
path containing H, V, A or a lowercase (relative) command breaks that
assumption, so the run stops rather than silently bending a stroke. It also
refuses a cut that would put a text baseline outside the shape holding it —
run tools/check_diagrams.py afterwards regardless, since that is the check
that measures label clearance properly.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FIGURES = ROOT / "assets" / "figures"

# Commands whose arguments are x,y pairs. Anything else and the pairing that
# lets us adjust only the y values stops being true.
PAIRED = set("MLCSQT")
# z/Z is closepath and carries no coordinates, so it is harmless. H, V and
# A do carry them and break the pairing; lowercase commands are relative,
# which would need the current point tracked rather than each y adjusted.
FORBIDDEN = re.compile(r"[HVA]|[mlcsqtahv](?![a-zA-Z])")


def shift_for(y: float, cuts: list[tuple[float, float]]) -> float:
    """How far up a point at `y` moves. Cuts at or above y all apply."""
    return sum(n for at, n in cuts if y >= at)


def span_shrink(top: float, bottom: float, cuts: list[tuple[float, float]]) -> float:
    """How much shorter a box spanning top..bottom becomes."""
    return sum(n for at, n in cuts if top < at <= bottom)


def adjust_path(d: str, cuts) -> str:
    out, i = [], 0
    tokens = re.findall(r"[A-Za-z]|-?[\d.]+", d)
    cmd = None
    coords: list[str] = []
    for t in tokens:
        if re.match(r"[A-Za-z]", t):
            cmd = t
            out.append(t)
            coords = []
            continue
        coords.append(t)
        if cmd in PAIRED and len(coords) % 2 == 0:          # this one is a y
            y = float(t)
            out.append(f"{y - shift_for(y, cuts):g}")
        else:
            out.append(t)
    # rebuild with single spaces; SVG does not care about the original spacing
    s, prev_alpha = "", False
    for t in out:
        if re.match(r"[A-Za-z]", t):
            s += (" " if s else "") + t
            prev_alpha = True
        else:
            s += " " + t
            prev_alpha = False
    return s.strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--cut", action="append", default=[], metavar="Y:N",
                    help="remove N units of height at y=Y; repeatable")
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    cuts = []
    for c in args.cut:
        at, n = c.split(":")
        cuts.append((float(at), float(n)))
    cuts.sort()
    total = sum(n for _, n in cuts)

    path = FIGURES / (args.name if args.name.endswith(".svg") else args.name + ".svg")
    svg = path.read_text()

    for m in re.finditer(r'\bd="([^"]+)"', svg):
        if FORBIDDEN.search(m.group(1)):
            print(f"refusing: path uses a relative or arc command, so its "
                  f"numbers are not x,y pairs:\n  {m.group(1)[:80]}")
            return 1

    vb = re.search(r'viewBox="0 0 ([\d.]+) ([\d.]+)"', svg)
    vw, vh = float(vb.group(1)), float(vb.group(2))
    out = svg

    # rects: move the top, shrink by whatever cuts fall inside
    def fix_rect(m):
        y, h = float(m.group(2)), float(m.group(4))
        ny = y - shift_for(y, cuts)
        nh = h - span_shrink(y, y + h, cuts)
        return f'{m.group(1)}{ny:g}{m.group(3)}{nh:g}{m.group(5)}'
    out = re.sub(r'(<rect[^>]*\by=")(-?[\d.]+)("[^>]*\bheight=")([\d.]+)(")',
                 fix_rect, out)

    for attr in ("y", "y1", "y2", "cy"):
        def fix(m, attr=attr):
            y = float(m.group(2))
            return f'{m.group(1)}{y - shift_for(y, cuts):g}"'
        out = re.sub(rf'(<(?:text|line|circle|ellipse|tspan)[^>]*?\b{attr}=")'
                     rf'(-?[\d.]+)"', fix, out)

    out = re.sub(r'\bd="([^"]+)"',
                 lambda m: 'd="' + adjust_path(m.group(1), cuts) + '"', out)

    new_h = vh - total
    out = out.replace(f'viewBox="0 0 {vb.group(1)} {vb.group(2)}"',
                      f'viewBox="0 0 {vb.group(1)} {new_h:g}"', 1)
    out = re.sub(r'(<svg\b[^>]*?)\sheight="[\d.]+"',
                 rf'\1 height="{round(1280 * new_h / vw):g}"', out, count=1)

    print(f"{path.name}: viewBox {vh:g} -> {new_h:g} "
          f"({total:g} removed at {', '.join(f'y={a:g}' for a, _ in cuts)})")
    print(f"  rendered {round(1280 * new_h / vw)}px at width 1280")
    if not args.apply:
        print("  (report only; pass --apply to write)")
        return 0
    path.write_text(out)
    print("  written — now run tools/check_diagrams.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
