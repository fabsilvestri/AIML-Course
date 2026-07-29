#!/usr/bin/env python3
"""
Find slides whose content is taller than reveal's coordinate space.

reveal scales overfull content down rather than complaining, so a slide that
does not fit does not look broken — it looks slightly smaller, and then
illegible on a projector at the back of the room. TRICKS section 9.6.

Three things are measured:

1. **Vertical overflow** — content taller than reveal's 720px canvas. Fails.
2b. **Horizontal overflow of anything else** — a display equation, table or
   figure wider than the 1280px canvas. KaTeX lays a long `$$…$$` out at its
   natural width and simply overhangs both edges; nothing clips, nothing warns,
   and the two ends of the identity are off the projector. Fails.

2. **Horizontal overflow of code** — a `pre` whose scrollWidth exceeds its
   clientWidth. This is the one that shows up on the podium machine and never on
   yours, because it depends on the fallback mono. Fails.
3. **The wrong typeface.** Every visible run of text must resolve to Source
   Sans 3, Source Code Pro, or one of KaTeX's own. The deck footer sits outside
   `.reveal` — deliberately, so it survives slide changes — and the theme sets
   the family on `.reveal`, so the footer inherited from `body`, which no rule
   had ever touched. It rendered in the browser's default serif on every slide
   of all 24 decks: Times, under a Source Sans 3 heading, for the one piece of
   text on screen the whole lecture. No check could see it, because it is not an
   overflow, a contrast failure or a missing asset. Fails.

4. **Footer collision** — `.deck-footer` is fixed 46px up from the bottom, so a
   slide measuring 700px is under the limit and still runs underneath it.
   Warns, because the fix is in the CSS rather than the deck.

    python3 tools/check_overflow.py            # exits non-zero if anything fails
    python3 tools/check_overflow.py --top 10   # or just list the tallest slides

Needs Playwright driving the installed Chrome:

    python3 -m pip install playwright

If Playwright is not available the check reports that and exits 0, so it can sit
in front of a commit without becoming a hard dependency.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DECKS = sorted(ROOT.glob("slides/lecture-[0-9][0-9].html"))

# Walk the direct children of every slide and take the lowest edge any of them
# reaches. Slides that are not the current one are display:none, so each is
# forced visible for the measurement and then put back.
# Which typefaces a deck is allowed to resolve to. KaTeX ships its own and must
# use them; code is Source Code Pro; everything else is Source Sans 3.
FONTS = """() => {
  const ok = /^(Source Sans 3|Source Sans Pro|Source Code Pro|KaTeX)/;
  const bad = new Map();
  // Show everything first. A deck opens on its title slide, where JS hides the
  // footer — so a scan of the visible page checks every slide except the ones
  // that matter and the element that was wrong. The first version of this check
  // did exactly that and reported the footer clean while it was set in Times.
  const restore = [];
  document.querySelectorAll('.slides > section').forEach(sec => {
    if (!sec.classList.contains('present')) {
      restore.push([sec, sec.style.display, sec.style.visibility]);
      sec.style.display = 'block';
      sec.style.visibility = 'visible';
    }
  });
  const footer = document.querySelector('.deck-footer');
  const wasHidden = footer && footer.classList.contains('is-hidden');
  if (wasHidden) footer.classList.remove('is-hidden');
  const walk = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
  let n;
  while ((n = walk.nextNode())) {
    if (!n.textContent.trim()) continue;
    const el = n.parentElement;
    if (!el || el.closest('script, style')) continue;
    // Only text a person can see. KaTeX emits a parallel MathML tree for screen
    // readers, clipped to a pixel and in the UA's 'math' font, and reveal keeps
    // a hidden "Resume presentation" button in Arial. Both are correct as they
    // are; neither is ever projected.
    if (el.closest('.katex-mathml, annotation, .resume-button')) continue;
    if (!el.getClientRects().length) continue;
    const cs0 = getComputedStyle(el);
    if (cs0.visibility === 'hidden' || cs0.display === 'none') continue;
    const fam = cs0.fontFamily.replace(/["']/g, '');
    if (ok.test(fam)) continue;
    const key = fam.split(',')[0].trim();
    if (!bad.has(key)) {
      bad.set(key, {family: key, text: n.textContent.trim().slice(0, 46),
                    where: el.className || el.tagName});
    }
  }
  restore.forEach(([sec, d, v]) => { sec.style.display = d; sec.style.visibility = v; });
  if (wasHidden) footer.classList.add('is-hidden');
  return [...bad.values()];
}"""

MEASURE = """() => {
  const H = Reveal.getConfig().height;
  const out = [];
  document.querySelectorAll('.slides > section').forEach((s, i) => {
    const hidden = !s.classList.contains('present');
    if (hidden) { s.style.display = 'block'; s.style.visibility = 'visible'; }
    let bottom = 0;
    s.querySelectorAll(':scope > *').forEach(el => {
      const edge = el.offsetTop + el.offsetHeight;
      if (edge > bottom) bottom = edge;
    });
    // widest code overrun, in px — depends on the mono actually resolved
    let codeOver = 0;
    s.querySelectorAll('pre').forEach(el => {
      const over = el.scrollWidth - el.clientWidth;
      if (over > codeOver) codeOver = over;
    });
    // anything else that runs off the side. A long display equation is the
    // usual culprit: KaTeX lays it out at its natural width and simply
    // overhangs, silently, off both edges of the canvas. Measured against the
    // slide's own box rather than the canvas, so padding counts.
    const W = Reveal.getConfig().width;
    let sideOver = 0, sideWhat = '';
    s.querySelectorAll('.katex-display > .katex, table, figure, blockquote')
     .forEach(el => {
      const w = Math.max(el.scrollWidth, el.getBoundingClientRect().width /
                         (Reveal.getScale() || 1));
      const over = w - W;
      if (over > sideOver) { sideOver = over; sideWhat = el.tagName.toLowerCase(); }
    });
    if (hidden) { s.style.display = ''; s.style.visibility = ''; }
    out.push({n: i + 1, title: s.dataset.menuTitle || '(untitled)',
              height: Math.round(bottom), limit: H,
              code_over: Math.round(codeOver),
              side_over: Math.round(sideOver), side_what: sideWhat,
              // the footer is suppressed on these two, so they may use the full
              // canvas; every other slide has to stop short of it
              footered: !(s.classList.contains('divider') ||
                          s.classList.contains('title-slide'))});
  });
  return out;
}"""

# .deck-footer is position: fixed, bottom: 14px, ~15px tall — measured, it eats
# the bottom 46px of the canvas on any slide that shows it.
FOOTER_PX = 46


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=0,
                    help="list the N tallest slides per deck instead of only "
                         "the ones that overflow")
    args = ap.parse_args()

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("playwright not installed — skipping the overflow check")
        return 0

    problems = 0
    with sync_playwright() as pw:
        browser = pw.chromium.launch(channel="chrome")
        page = browser.new_page(viewport={"width": 1600, "height": 1000})
        for deck in DECKS:
            page.goto(deck.as_uri())
            page.wait_for_function("() => window.Reveal && Reveal.isReady()",
                                   timeout=30_000)
            page.wait_for_timeout(1500)          # let KaTeX and highlight settle
            fonts = page.evaluate(FONTS)
            rows = page.evaluate(MEASURE)
            limit = rows[0]["limit"]
            over = [r for r in rows if r["height"] > limit]
            wide = [r for r in rows if r["code_over"] > 0]
            # 8px of slack: KaTeX's own reported width rounds a little high on
            # equations that in fact sit inside the canvas.
            side = [r for r in rows if r["side_over"] > 8]
            footer = [r for r in rows if r["footered"]
                      and limit - FOOTER_PX < r["height"] <= limit]

            print(f"\n{deck.relative_to(ROOT)}: {len(rows)} slides, "
                  f"{len(over)} over {limit}px, {len(wide)} with code too wide, "
                  f"{len(side)} running off the side, "
                  f"{len(footer)} under the footer"
                  + (f", {len(fonts)} IN THE WRONG TYPEFACE" if fonts else ""))
            for r in sorted(over, key=lambda r: -r["height"]):
                print(f"  OVER  {r['height']:5d}px  #{r['n']:3d}  {r['title']}")
            for r in sorted(wide, key=lambda r: -r["code_over"]):
                print(f"  WIDE  +{r['code_over']:4d}px  #{r['n']:3d}  {r['title']}")
            for r in sorted(side, key=lambda r: -r["side_over"]):
                print(f"  SIDE  +{r['side_over']:4d}px  #{r['n']:3d}  {r['title']}"
                      f"   ({r['side_what']} wider than the canvas)")
            for r in sorted(footer, key=lambda r: -r["height"]):
                print(f"  warn  {r['height']:5d}px  #{r['n']:3d}  {r['title']}"
                      f"   (footer starts at {limit - FOOTER_PX})")
            if args.top:
                print("  --- tallest ---")
                for r in sorted(rows, key=lambda r: -r["height"])[:args.top]:
                    print(f"        {r['height']:5d}px  #{r['n']:3d}  {r['title']}")
            for f in fonts:
                print(f"  FONT  resolves to {f['family']!r} — {f['where']}")
                print(f"                    {f['text']!r}")
            problems += len(over) + len(wide) + len(side) + len(fonts)
        browser.close()

    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
