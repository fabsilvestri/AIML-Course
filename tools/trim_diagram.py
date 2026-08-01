#!/usr/bin/env python3
"""
Lift an explanatory panel out of a hand-drawn diagram, so its text stops being
shrunk by the figure cap.

    python3 tools/trim_diagram.py d-coltransformer --at 400          # report
    python3 tools/trim_diagram.py d-coltransformer --at 400 --apply

**Why.** `.fig-wide` caps a figure's rendered height, so a diagram taller than
the cap is scaled down and *all* of its text with it — `check_text_floor`
reports these as 1.17-1.26 on-slide px per authored point against the 1.33 an
unclamped figure gets. Six of the seven offenders are tall for the same reason:
a paragraph of prose sits in a band across the bottom of the artwork.

That prose does not want to be in an SVG. Moved into the slide's HTML it is
selectable, searchable, styled by the deck's own type scale, and no longer
forces the drawing above it to shrink. This tool does the SVG half: it removes
every element lying wholly below `--at` and shortens the viewBox to match.

**It refuses rather than guesses.** Anything crossing the cut line, and any
`<path>` below it (whose extent cannot be read off its attributes without
parsing the geometry), stops the run and is listed. A diagram is a drawing, and
silently deleting half a stroke is worse than doing nothing.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FIGURES = ROOT / "assets" / "figures"

# Elements whose vertical extent can be read straight off their attributes.
# <path> is deliberately absent: its extent lives inside the `d` string.
EXTENT = {
    "rect":   lambda a: (num(a, "y"), num(a, "y") + num(a, "height")),
    "text":   lambda a: (num(a, "y") - 20, num(a, "y") + 6),   # ascent/descent
    "line":   lambda a: (min(num(a, "y1"), num(a, "y2")),
                         max(num(a, "y1"), num(a, "y2"))),
    "circle": lambda a: (num(a, "cy") - num(a, "r"), num(a, "cy") + num(a, "r")),
    "ellipse": lambda a: (num(a, "cy") - num(a, "ry"), num(a, "cy") + num(a, "ry")),
}


def num(attrs: str, name: str) -> float:
    m = re.search(rf'\b{name}="(-?[\d.]+)"', attrs)
    return float(m.group(1)) if m else 0.0


def elements(svg: str):
    """(start, end, tag, attrs) for every drawable element, in document order."""
    for m in re.finditer(r"<(rect|text|line|circle|ellipse|path|polyline|polygon)\b"
                         r"([^>]*?)(/>|>)", svg):
        tag = m.group(1)
        end = m.end()
        if m.group(3) == ">":                      # find the closing tag
            close = svg.find(f"</{tag}>", end)
            if close != -1:
                end = close + len(tag) + 3
        yield m.start(), end, tag, m.group(2)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--at", type=float, required=True,
                    help="y in viewBox units; everything wholly below goes")
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    path = FIGURES / (args.name if args.name.endswith(".svg")
                      else args.name + ".svg")
    svg = path.read_text()
    vb = re.search(r'viewBox="0 0 ([\d.]+) ([\d.]+)"', svg)
    if not vb:
        print(f"{path.name}: no viewBox")
        return 1
    vw, vh = float(vb.group(1)), float(vb.group(2))

    doomed, straddling, opaque = [], [], []
    for start, end, tag, attrs in elements(svg):
        if tag in ("path", "polyline", "polygon"):
            # cannot be measured from attributes; only complain if it might be
            # in the way, which we cannot know — so report and let a person look
            opaque.append((tag, svg[start:end][:70]))
            continue
        top, bottom = EXTENT[tag](attrs)
        if top >= args.at:
            doomed.append((start, end, tag, svg[start:end]))
        elif bottom > args.at:
            straddling.append((tag, round(top, 1), round(bottom, 1),
                               svg[start:end][:70]))

    print(f"{path.name}  viewBox {vw:g}x{vh:g}  cut at y={args.at:g}")
    print(f"  {len(doomed)} element(s) wholly below the cut")
    for _, _, tag, raw in doomed:
        flat = " ".join(re.sub(r"<[^>]+>", " ", raw).split())
        print(f"    {tag:8s} {flat[:70]}")
    if straddling:
        print(f"  {len(straddling)} element(s) CROSS the cut — refusing:")
        for tag, t, b, raw in straddling:
            print(f"    {tag:8s} y {t}..{b}   {' '.join(raw.split())[:60]}")
    if opaque:
        print(f"  {len(opaque)} path/polyline/polygon not measurable from "
              f"attributes — check by eye:")
        for tag, raw in opaque[:6]:
            print(f"    {tag:8s} {' '.join(raw.split())[:66]}")

    if not args.apply:
        print("\n  (report only; pass --apply to write)")
        return 0
    if straddling:
        print("\n  refusing to apply: move the cut, or edit those by hand")
        return 1
    if not doomed:
        print("\n  nothing to remove")
        return 1

    out = svg
    for start, end, _, _ in sorted(doomed, reverse=True):
        out = out[:start] + out[end:]
    new_h = args.at
    out = out.replace(f'viewBox="0 0 {vb.group(1)} {vb.group(2)}"',
                      f'viewBox="0 0 {vb.group(1)} {new_h:g}"', 1)
    out = re.sub(r'(<svg\b[^>]*?)\sheight="[\d.]+"',
                 rf'\1 height="{round(1280 * new_h / vw):g}"', out, count=1)
    path.write_text(out)
    print(f"\n  written: viewBox height {vh:g} -> {new_h:g}, "
          f"rendered {round(1280 * new_h / vw)}px at width 1280")
    return 0


if __name__ == "__main__":
    sys.exit(main())
