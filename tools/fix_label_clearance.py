#!/usr/bin/env python3
"""
Grow the boxes that were sized to a baseline instead of to the ink.

    python3 tools/fix_label_clearance.py --dry-run
    python3 tools/fix_label_clearance.py

`check_diagrams.py` reports 39 labels with under 6px of clear space between
their ink and the border of the shape holding them, and **28 of them are the
bottom edge**. That is not scatter, it is one habit: a box is sized so that the
last line's BASELINE sits comfortably inside it, and the descenders of g, y, p
and the comma then land on the border. `d-course-arc` does it on 12 of 12 cards.

So the repair is systematic rather than 39 individual nudges: find the shape
each tight label belongs to, work out how much clearance it lacks, and add that
to the shape's height (or width, for the handful that are tight at the side).
Nothing moves; boxes get taller by a few units.

Matching a rendered shape back to the file is by geometry: for a `rect`,
`getBBox()` returns exactly its `x`/`y`/`width`/`height` attributes, so the
attribute string is a reliable key. Shapes that are not rects are reported and
left alone — an ellipse or polygon needs a judgement about which way to grow.

Re-run `check_diagrams.py` afterwards. Growing a box can push it into its
neighbour or off the frame, and the overlap and out-of-frame tests are what
catch that.
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location(
    "_cd", Path(__file__).with_name("check_diagrams.py"))
_cd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_cd)

TARGET = 9.0          # on-slide px to aim for: clear of the 6px floor, and
                      # just under the 10px the course's own authors average


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--only", default="")
    args = ap.parse_args()

    from playwright.sync_api import sync_playwright

    files = sorted(p for p in (ROOT / "assets" / "figures").glob("d-*.svg")
                   if args.only in p.name)
    grown, skipped = 0, []

    with sync_playwright() as pw:
        browser = pw.chromium.launch(channel="chrome")
        page = browser.new_page(viewport={"width": 1400, "height": 900},
                                device_scale_factor=_cd.SHOT_SCALE)
        for f in files:
            page.goto(f.as_uri())
            page.wait_for_timeout(180)
            r = page.evaluate(_cd.MEASURE)
            vb, texts, shapes = r["vb"], r["texts"], r.get("shapes", [])
            on_slide = 1280.0 / vb["w"]

            # worst deficit per shape, per side
            need: dict[tuple, dict] = defaultdict(lambda: defaultdict(float))
            for t in texts:
                sh = _cd.owner_of(t, shapes)
                if sh is None:
                    continue
                gap, side = _cd.clearance(t, sh)
                if gap * on_slide >= _cd.CLEAR_FAIL:
                    continue
                key = (round(sh["x"], 2), round(sh["y"], 2),
                       round(sh["w"], 2), round(sh["h"], 2))
                deficit = (TARGET / on_slide) - gap
                need[key][side] = max(need[key][side], deficit)

            if not need:
                continue
            src = f.read_text()
            before = src
            for (x, y, w, h), sides in need.items():
                dh = max(sides.get("bottom", 0), sides.get("top", 0))
                dw = max(sides.get("left", 0), sides.get("right", 0))
                pat = re.compile(
                    r'(<rect\b[^>]*?\bx="%s"[^>]*?\by="%s"[^>]*?\bwidth="%s"[^>]*?\bheight=")%s(")'
                    % (_n(x), _n(y), _n(w), _n(h)))
                m = pat.search(src)
                if not m:
                    skipped.append(f"{f.name}: no rect at "
                                   f"({x:g},{y:g},{w:g}x{h:g}) — "
                                   f"not a rect, or attributes out of order")
                    continue
                if dh:
                    # whole units: these files are hand-authored and a height of
                    # 80.7858 is noise in a diff nobody wants to read
                    new_h = math.ceil(h + dh)
                    src = pat.sub(lambda mm: f"{mm.group(1)}"
                                             f"{new_h:g}{mm.group(2)}", src, 1)
                    grown += 1
                    print(f"  {f.name:26s} height {h:g} -> {new_h:g}"
                          f"   (+{(new_h - h) * on_slide:.1f}px of clear space)")
                if dw:
                    skipped.append(f"{f.name}: ({x:g},{y:g}) is tight at the "
                                   f"side by {dw * on_slide:.1f}px — widening "
                                   f"moves content, so do it by hand")
            if src != before and not args.dry_run:
                f.write_text(src)
        browser.close()

    if skipped:
        print(f"\n{len(skipped)} left for a person:")
        for s in skipped:
            print("  " + s)
    print(f"\n{grown} box(es) grown{' (dry run)' if args.dry_run else ''}. "
          f"Re-run tools/check_diagrams.py — a taller box can reach its "
          f"neighbour.")
    return 0


def _n(v: float) -> str:
    """A number as it is most likely written in the file: 68 not 68.0."""
    return re.escape(f"{v:g}")


if __name__ == "__main__":
    sys.exit(main())
