#!/usr/bin/env python3
"""
Measure the hand-drawn diagrams, instead of trusting the hand that drew them.

Every `assets/figures/d-*.svg` is written by hand, and the one thing hand-drawn
SVG cannot do is know how wide its own text is. You place a label at x=540 with
`text-anchor="middle"`, it reads fine in the editor, and at 19px in Source Sans
3 it is 1,233px wide inside a 1,080px viewBox — so both ends are simply gone.
Nothing clips, nothing warns, and the SVG is perfectly valid.

Three failures are measured, by rendering the file in Chrome the way a slide
does and asking the browser for each label's box.

**"The way a slide does" is load-bearing.** The first version of this file
inlined the SVG into a page that linked the course stylesheet, so the browser
resolved Source Sans 3 from the page's `@font-face` and measured every label in
it. The decks embed these files as `<img src="...">`, and an SVG inside an
`<img>` is a separate document in secure static mode: it cannot see the page's
fonts and may not fetch anything. Unless the font is installed on the machine —
it is not, on this one or on a lecture-theatre one — the projector shows
Helvetica. So this file spent its first run measuring a typeface nobody would
see, and every "that label fits" was an answer about the wrong font.
`tools/embed_diagram_fonts.py` puts the real font inside each diagram; this
renders through an `<img>` so it can never again disagree with the projector.

1. **Out of frame** — a `<text>` whose box crosses the viewBox edge. Fails.
2. **Overlap** — two labels whose *ink* covers the same ground. An annotation
   printed across a caption is unreadable at the back of a lecture theatre and
   invisible in a code review. Fails.
3. **Crowded, or escaped.** A label's ink must keep clear of the border of the
   shape that holds it. A caption wider than its own plate crosses that border;
   far more often, a box was sized to its last line's BASELINE and the
   descenders of g, y, p and the comma land on the bottom edge — 28 of the 38
   tight cases in this course were that one habit. Attribution is by how much of
   the LABEL a shape covers, never by containment, because a caption that
   escapes its plate is not contained by it and a containment join can never
   find the thing it is looking for. Fails.

4. **Struck through** — a label with something drawn underneath its ink. In
   d-projection the projection arrow ran through the circumflex of `Xθ̂`, so a
   diagram whose whole subject is the distinction between θ and θ̂ displayed it
   as `Xθ`.

   This is decided on pixels, by rendering the file twice — once as it is, once
   with every label hidden — and asking what was already on the canvas where the
   ink landed. The obvious geometric test, `isPointInStroke` on each path, was
   tried first and is useless here: Chrome ignores `stroke-dasharray`, so it
   reports a hit through the gap of a dashed line, and half these diagrams route
   a dashed rule around a label on purpose. Comparing renders costs a second
   screenshot and has no such blind spot — a label on a shaded panel is fine, a
   label over a line is not.

Estimating the width in Python is not good enough. A first pass at this used an
average advance of 0.50 em and reported twenty-two overflowing diagrams; the
browser found a different set. Kerning, the tspan-built glyphs and the fallback
chain all move the number, so the only trustworthy width is the rendered one.

    python3 tools/check_diagrams.py             # exits non-zero on any failure
    python3 tools/check_diagrams.py --list      # print every box, for drawing

Needs Playwright driving the installed Chrome, exactly as check_overflow.py
does; without it the check reports that and exits 0.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FIGURES = ROOT / "assets" / "figures"

# A label may sit a little proud of the frame without being a defect — a
# descender or an italic overhang is not what we are hunting.
EDGE_SLACK = 3.0
# Bounding boxes are the wrong yardstick for stacked text and it took two wrong
# rules to see it. A `<text>` box includes the font's full ascent and descent,
# so two lines of a caption ALWAYS have overlapping boxes — the first rule, on
# shared area, called every two-line caption in the course a collision. Raising
# the bar to 45% of the shorter box then cleared a caption in d-projection whose
# baselines are 16px apart carrying 19px text, which really does print one line
# into the other.
#
# Baseline separation against font size was the third rule and also wrong: it
# called a 23px heading with its 19px caption 24px below "too tight", which is
# ordinary typesetting. Leading is a matter of taste; ink touching ink is not.
#
# So measure the ink. Canvas `measureText` returns the real ascent and descent
# of the actual string in the actual font — a line of lower-case with no
# descenders occupies far less than its em box, which is exactly the slack that
# makes normal captions legal and d-projection's 16px-apart pair illegal.
OVERLAP_MIN_DX = 0.25      # shared width, as a fraction of the narrower label
INK_SLACK = 1.0            # px of ink overlap to forgive
SHOT_SCALE = 2             # device pixels per user unit in the two renders
# Clearance between a label's ink and the border of the shape holding it, in
# ON-SLIDE pixels. 6 is defensible on acuity alone: at 3 m from a 1280-wide
# canvas one canvas pixel is 2.34 mm, and the 1-arcmin limit at the back of a
# 15 m room is 1.9 px — which is where two edges stop merging, not where a gap
# reads as deliberate space. It is also roughly twice the 2.4-3.0 px weight of
# the borders themselves. 10 is where this course's own authors already sit at
# the lower quartile, so it is "match your best work" rather than a legibility
# floor, and it warns rather than fails.
CLEAR_FAIL = 6.0
CLEAR_WARN = 10.0
# Six device pixels, which sounds absurdly low until you measure the case this
# was written for: the circumflex of `Xθ̂` crossing a 3.5px arrow puts exactly
# 7px of ink on top of the line. A thin mark over a thin line IS a small number
# of pixels, and it still renders the label as `Xθ` on a projector. The pixel
# has to be ink (the two renders differ there) AND on top of something (the
# hidden-text render is not the local background there), so noise does not
# accumulate the way a bare threshold suggests.
STRIKE_PX = 6              # device pixels of ink over foreground before it counts
COLOUR_EPS = 26            # per-channel difference that counts as "a different colour"

# The file is opened as the top-level document: no stylesheet, nothing injected,
# anything the diagram needs it has to carry. That is the same font resolution an
# <img> gets — verified by screenshotting one diagram both ways and diffing:
# 0 differing pixels of 1,874,080 — and unlike an <img> it leaves a DOM to
# measure. Measuring what the deck actually shows is the whole point.

HIDE_TEXT = "() => { document.querySelectorAll('svg text')" \
            ".forEach(t => t.style.visibility = 'hidden'); }"

MEASURE = """() => {
  const svg = document.querySelector('svg');
  // createElement in an SVG document makes an SVG-namespaced element, and an
  // SVG <canvas> has no getContext. Name the HTML namespace explicitly.
  const ctx = document.createElementNS('http://www.w3.org/1999/xhtml', 'canvas')
                      .getContext('2d');
  const vb  = svg.viewBox.baseVal;
  const out = [];
  svg.querySelectorAll('text').forEach((t, i) => {
    const b = t.getBBox();
    const base = t.y && t.y.baseVal.numberOfItems
                 ? t.y.baseVal.getItem(0).value : b.y + b.height;
    const cs = getComputedStyle(t);
    // A label painted with `paint-order: stroke` and a white stroke knocks out
    // whatever is behind it before drawing the glyphs — it carries its own
    // clearing, which is the fix for a guide line running behind a reading, not
    // a defect. Without this the check flags the halo itself, since the halo is
    // by definition ink laid over something already drawn.
    const haloed = (cs.paintOrder || '').includes('stroke') &&
                   cs.stroke && cs.stroke !== 'none' &&
                   parseFloat(cs.strokeWidth) > 0;
    const fs = parseFloat(cs.fontSize) || 19;
    // real ink extents of this exact string in this exact font
    ctx.font = `${cs.fontStyle} ${cs.fontWeight} ${cs.fontSize} ${cs.fontFamily}`;
    const m = ctx.measureText(t.textContent || '');
    // clamp to the element's own box: a tspan lifted by dy (a hat on a theta)
    // puts ink where measureText, which knows nothing of tspans, will not.
    const inkTop = Math.min(base - (m.actualBoundingBoxAscent || 0), b.y + b.height);
    const inkBot = Math.max(base + (m.actualBoundingBoxDescent || 0), b.y);
    out.push({i, x: b.x, y: b.y, w: b.width, h: b.height, base, fs, haloed,
              inkTop: Math.max(inkTop, b.y),
              inkBot: Math.min(inkBot, b.y + b.height),
              text: (t.textContent || '').replace(/\\s+/g, ' ').trim()});
  });
  // Shapes a label might live inside. A stroke or a fill makes a shape
  // visible; under 14 units it is an ornament, not a container; and anything
  // over 80% of the frame is a background, which the viewBox test already owns.
  const shapes = [];
  const frameArea = vb.width * vb.height;
  svg.querySelectorAll('rect, circle, ellipse, polygon').forEach(el => {
    const cs = getComputedStyle(el);
    const stroked = cs.stroke && cs.stroke !== 'none' &&
                    parseFloat(cs.strokeWidth) > 0;
    const filled = cs.fill && cs.fill !== 'none';
    if (!stroked && !filled) return;
    let b; try { b = el.getBBox(); } catch (e) { return; }
    if (b.width < 14 || b.height < 14) return;
    if (b.width * b.height > 0.8 * frameArea) return;
    shapes.push({x: b.x, y: b.y, w: b.width, h: b.height,
                 sw: parseFloat(cs.strokeWidth) || 0,
                 cls: el.getAttribute('class') || el.tagName});
  });
  return {vb: {x: vb.x, y: vb.y, w: vb.width, h: vb.height},
          texts: out, shapes};
}"""


def clash(a: dict, b: dict) -> float:
    """How far two labels' ink overlaps vertically, where they share width."""
    dx = min(a["x"] + a["w"], b["x"] + b["w"]) - max(a["x"], b["x"])
    if dx <= 0 or dx / (min(a["w"], b["w"]) or 1) < OVERLAP_MIN_DX:
        return 0.0                        # no shared ground: cannot clash
    dy = min(a["inkBot"], b["inkBot"]) - max(a["inkTop"], b["inkTop"])
    return dy if dy > INK_SLACK else 0.0


def struck_labels(page, wrapper, texts, out_dir, stem, vb_w):
    """Which labels have something already drawn where their ink lands.

    Two renders of the same file: one as it is, one with every label hidden.
    Wherever the two differ, ink was laid down; if what the second render has at
    that pixel is not the local background, the ink went on top of something.
    """
    from PIL import Image

    shot_a = out_dir / f"{stem}-a.png"
    shot_b = out_dir / f"{stem}-b.png"
    page.locator("svg").screenshot(path=str(shot_a))
    page.evaluate(HIDE_TEXT)
    page.locator("svg").screenshot(path=str(shot_b))
    page.reload()
    page.wait_for_timeout(150)

    # The screenshot is in CSS pixels, the boxes are in viewBox units, and since
    # the diagrams are authored at viewBox 1080 but presented at width 1280 the
    # two differ by 1.185. Assuming they were equal put the search window in the
    # wrong place and invented four defects the moment the files were resized.
    a = Image.open(shot_a).convert("RGB")
    b = Image.open(shot_b).convert("RGB")
    if a.size != b.size:
        return []
    ap, bp = a.load(), b.load()
    W, H = a.size
    px_per_unit = (W / SHOT_SCALE) / vb_w * SHOT_SCALE

    struck = []
    for t in texts:
        if t.get("haloed"):
            continue
        x0 = max(0, int(t["x"] * px_per_unit) - 1)
        x1 = min(W, int((t["x"] + t["w"]) * px_per_unit) + 2)
        # the FULL box, not the tight ink box. The tight box comes from canvas
        # metrics, which know nothing of a tspan raised by dy — so searching it
        # skipped the circumflex of `Xθ̂`, which is the one piece of ink that was
        # actually struck through, and the check passed on the very file it was
        # written for. Being generous here is free: only pixels where the two
        # renders differ are counted, and those are text ink by construction.
        y0 = max(0, int(t["y"] * px_per_unit) - 1)
        y1 = min(H, int((t["y"] + t["h"]) * px_per_unit) + 2)
        if x1 <= x0 or y1 <= y0:
            continue
        # the local background is whatever colour dominates the region with the
        # labels hidden — the panel fill, not the page white
        counts: dict[tuple, int] = {}
        for y in range(y0, y1):
            for x in range(x0, x1):
                c = bp[x, y]
                counts[c] = counts.get(c, 0) + 1
        bg = max(counts, key=counts.get)

        n = 0
        for y in range(y0, y1):
            for x in range(x0, x1):
                pa, pb = ap[x, y], bp[x, y]
                inked = max(abs(pa[k] - pb[k]) for k in range(3)) > COLOUR_EPS
                if not inked:
                    continue
                if max(abs(pb[k] - bg[k]) for k in range(3)) > COLOUR_EPS:
                    n += 1
        if n >= STRIKE_PX:
            struck.append((n, t))
    return struck


def owner_of(t: dict, shapes: list) -> dict | None:
    """Which shape a label belongs to, for the purpose of measuring its border.

    Attribution is by the fraction of the LABEL that a shape covers, never by
    containment — because the defect being hunted destroys the evidence for
    containment. A caption that runs out of its plate is, by definition, not
    inside that plate, so a containment join can never attribute the label you
    are looking for. Ties go to the smallest shape, so a node box wins over the
    panel drawn around it.

    d-nms is why the tie-break matters: three identical 170x150 rects all
    enclose the label 0.98, because overlapping regions are the diagram's whole
    subject. A join that cannot express that produced a confident false report
    about that file.
    """
    best, best_f = None, 0.0
    for sh in shapes:
        dx = min(t["x"] + t["w"], sh["x"] + sh["w"]) - max(t["x"], sh["x"])
        dy = min(t["y"] + t["h"], sh["y"] + sh["h"]) - max(t["y"], sh["y"])
        if dx <= 0 or dy <= 0:
            continue
        f = (dx * dy) / ((t["w"] * t["h"]) or 1)
        if f > best_f + 0.02 or (abs(f - best_f) <= 0.02 and best is not None
                                 and sh["w"] * sh["h"] < best["w"] * best["h"]):
            best, best_f = sh, max(f, best_f)
    return best if best_f >= 0.50 else None


def clearance(t: dict, sh: dict) -> tuple[float, str]:
    """Smallest gap between a label's ink and its shape's border, and which side.

    The stroke is centred on the path, so the inner edge is half a stroke in.
    Vertical extents use the measured ink rather than the em box, because the
    habit this catches is a box fitted to the last line's BASELINE, which leaves
    the descenders of g, y, p and the comma sitting on the border.
    """
    half = sh["sw"] / 2
    gaps = {"left":   t["x"] - (sh["x"] + half),
            "right":  (sh["x"] + sh["w"] - half) - (t["x"] + t["w"]),
            "top":    t["inkTop"] - (sh["y"] + half),
            "bottom": (sh["y"] + sh["h"] - half) - t["inkBot"]}
    side = min(gaps, key=gaps.get)
    return gaps[side], side


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true",
                    help="print every measured box, not only the failures")
    ap.add_argument("--only", default="", help="substring of the filename")
    args = ap.parse_args()

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("playwright not installed — skipping the diagram check")
        return 0

    files = sorted(p for p in FIGURES.glob("d-*.svg") if args.only in p.name)
    scratch = Path("/private/tmp/claude-501") / "diagcheck"
    scratch.mkdir(parents=True, exist_ok=True)

    problems = 0
    with sync_playwright() as pw:
        browser = pw.chromium.launch(channel="chrome")
        page = browser.new_page(viewport={"width": 1400, "height": 900},
                                device_scale_factor=SHOT_SCALE)
        for f in files:
            wrapper = f
            page.goto(f.as_uri())
            page.wait_for_timeout(220)               # let the webfont land
            r = page.evaluate(MEASURE)
            vb, texts = r["vb"], r["texts"]
            struck = struck_labels(page, wrapper, texts, scratch, f.stem, vb["w"])

            out_of_frame, collisions, tight = [], [], []
            for t in texts:
                over = max(vb["x"] - t["x"],
                           (t["x"] + t["w"]) - (vb["x"] + vb["w"]),
                           vb["y"] - t["y"],
                           (t["y"] + t["h"]) - (vb["y"] + vb["h"]))
                if over > EDGE_SLACK:
                    out_of_frame.append((over, t))
            on_slide = 1280.0 / vb["w"]
            for t in texts:
                sh = owner_of(t, r.get("shapes", []))
                if sh is None:
                    continue                    # free-floating: test B only
                gap, side = clearance(t, sh)
                if gap * on_slide < CLEAR_FAIL:
                    tight.append((gap * on_slide, side, t, sh))
            for i, a in enumerate(texts):
                for b in texts[i + 1:]:
                    ink = clash(a, b)
                    if ink:
                        collisions.append((ink, a, b))

            if args.list:
                print(f"\n{f.name}  viewBox {vb['w']:.0f}x{vb['h']:.0f}")
                for t in texts:
                    print(f"   [{t['x']:7.1f},{t['y']:7.1f}] "
                          f"{t['w']:6.1f}x{t['h']:5.1f}  {t['text'][:52]!r}")
                continue

            if out_of_frame or collisions or struck or tight:
                print(f"\n{f.name}  (viewBox {vb['w']:.0f}x{vb['h']:.0f})")
                for over, t in sorted(out_of_frame, reverse=True,
                                      key=lambda p: p[0]):
                    print(f"   OUT   +{over:6.1f}px  w={t['w']:6.1f}  "
                          f"{t['text'][:58]!r}")
                for ink, a, b in sorted(collisions, reverse=True,
                                        key=lambda p: p[0]):
                    print(f"   OVER  ink overlaps by {ink:5.1f}px "
                          f"(baselines {abs(a['base'] - b['base']):.0f}px apart)")
                    print(f"           {a['text'][:52]!r}")
                    print(f"           {b['text'][:52]!r}")
                for n, t in sorted(struck, reverse=True, key=lambda p: p[0]):
                    print(f"   STRUCK {n:5d}px of ink landed on something "
                          f"already drawn")
                    print(f"           {t['text'][:52]!r}")
                for gap, side, t, sh in sorted(tight, key=lambda p: p[0]):
                    verdict = "TIGHT" if gap >= 0 else "ESCAPES"
                    print(f"   {verdict:7s} {gap:6.2f}px of clear space at the "
                          f"{side} of its {sh['cls']}")
                    print(f"           {t['text'][:52]!r}")
                problems += (len(out_of_frame) + len(collisions) + len(struck)
                             + len(tight))
        browser.close()

    if args.list:
        return 0
    if problems:
        print(f"\n{problems} diagram defect(s). Widen the viewBox, split the "
              f"label over two lines, or move it.")
        return 1
    print(f"{len(files)} diagrams: every label inside its frame, none overlapping")
    return 0


if __name__ == "__main__":
    sys.exit(main())
