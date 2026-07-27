#!/usr/bin/env python3
"""
Find slides whose content is taller than reveal's coordinate space.

reveal scales overfull content down rather than complaining, so a slide that
does not fit does not look broken — it looks slightly smaller, and then
illegible on a projector at the back of the room. TRICKS section 9.6.

    python3 tools/check_overflow.py            # exits non-zero if anything is found
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
DECKS = sorted(ROOT.glob("slides/*.html"))

# Walk the direct children of every slide and take the lowest edge any of them
# reaches. Slides that are not the current one are display:none, so each is
# forced visible for the measurement and then put back.
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
    if (hidden) { s.style.display = ''; s.style.visibility = ''; }
    out.push({n: i + 1, title: s.dataset.menuTitle || '(untitled)',
              height: Math.round(bottom), limit: H});
  });
  return out;
}"""


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
            rows = page.evaluate(MEASURE)
            limit = rows[0]["limit"]
            over = [r for r in rows if r["height"] > limit]
            shown = (sorted(rows, key=lambda r: -r["height"])[:args.top]
                     if args.top else sorted(over, key=lambda r: -r["height"]))
            print(f"\n{deck.relative_to(ROOT)}: {len(rows)} slides, "
                  f"{len(over)} over {limit}px")
            for r in shown:
                flag = "OVER" if r["height"] > limit else "    "
                print(f"  {flag} {r['height']:5d}px  #{r['n']:3d}  {r['title']}")
            problems += len(over)
        browser.close()

    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
