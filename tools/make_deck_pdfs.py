#!/usr/bin/env python3
"""
A PDF handout for every deck, printed from the deck itself.

    python3 tools/make_deck_pdfs.py          # all 24
    python3 tools/make_deck_pdfs.py 7 8      # just these
    python3 tools/make_deck_pdfs.py --port 8731

Output: slides/pdf/lecture-NN.pdf, one page per slide at 1280x720.

HOW, AND WHY NOT THE OBVIOUS WAY. reveal.js paginates for print by adding
`.reveal-print` to <html> and wrapping each slide in a `.pdf-page`, from
JavaScript, when the URL carries `?print-pdf`. On the vendored reveal 5.2.1
that never happens: the config reports view="print" and isPrintView() true,
<html> keeps `reveal-full-page`, no `.pdf-page` appears, and the result is a
single page. The stock reveal demo in lib/ fails identically, so it is
reveal's own path rather than anything in these decks -- which also means the
site's advice to "append ?print-pdf and print to PDF" had never worked.

So pagination lives in assets/css/print.css instead, in pure CSS, and this
script just drives headless Chrome. The deck is loaded WITHOUT any query
string: the print stylesheet is `media="print"`, so it applies to any print,
including a student pressing Cmd+P.

The trailing blank page. Chrome emits one more page than there are slides.
The last slide's break is switched off in CSS by two selectors and the page
survives both, so it is trimmed here instead -- and only ever when it really
is blank, which is asserted rather than assumed.
"""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "slides" / "pdf"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"


def slide_count(deck: Path) -> int:
    return len(re.findall(r"^<section", deck.read_text(encoding="utf-8"), re.M))


def serve(port: int):
    """A local server, because Chrome will not print a file:// deck's assets."""
    try:
        urllib.request.urlopen(f"http://localhost:{port}/index.html", timeout=2)
        return None                       # already up; leave it alone
    except Exception:
        pass
    p = subprocess.Popen([sys.executable, "-m", "http.server", str(port)],
                         cwd=ROOT, stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL)
    for _ in range(40):
        try:
            urllib.request.urlopen(f"http://localhost:{port}/index.html", timeout=1)
            return p
        except Exception:
            time.sleep(0.25)
    p.terminate()
    raise SystemExit(f"could not start a server on port {port}")


def trim_trailing_blank(pdf: Path, expected: int) -> int:
    """Drop the final page if, and only if, it carries no text."""
    from pypdf import PdfReader, PdfWriter
    r = PdfReader(str(pdf))
    if len(r.pages) == expected:
        return len(r.pages)
    if len(r.pages) != expected + 1:
        return len(r.pages)               # something else is wrong; say so upstream
    if r.pages[-1].extract_text().strip():
        return len(r.pages)               # not blank — keep it, and report
    w = PdfWriter()
    for page in r.pages[:-1]:
        w.add_page(page)
    with open(pdf, "wb") as fh:
        w.write(fh)
    return expected


def render(n: int, port: int) -> tuple[int, int]:
    deck = ROOT / f"slides/lecture-{n:02d}.html"
    dest = OUT / f"lecture-{n:02d}.pdf"
    want = slide_count(deck)
    subprocess.run(
        [CHROME, "--headless=new", "--disable-gpu", "--no-sandbox",
         "--virtual-time-budget=45000", "--no-pdf-header-footer",
         f"--print-to-pdf={dest}",
         f"http://localhost:{port}/slides/lecture-{n:02d}.html"],
        check=True, capture_output=True)
    return want, trim_trailing_blank(dest, want)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("lectures", nargs="*", type=int)
    ap.add_argument("--port", type=int, default=8731)
    a = ap.parse_args()

    if not Path(CHROME).exists():
        raise SystemExit(f"Chrome not found at {CHROME}")
    OUT.mkdir(parents=True, exist_ok=True)
    proc = serve(a.port)
    bad = 0
    try:
        for n in (a.lectures or range(1, 25)):
            want, got = render(n, a.port)
            ok = want == got
            bad += not ok
            size = (OUT / f"lecture-{n:02d}.pdf").stat().st_size / 1e6
            print(f"{'ok  ' if ok else 'FAIL'}  lecture-{n:02d}.pdf  "
                  f"{got:>3} pages (deck has {want})  {size:5.1f} MB")
    finally:
        if proc:
            proc.terminate()
    print()
    if bad:
        print(f"{bad} deck(s) whose page count does not match their slide count")
        return 1
    print("every deck exported, one page per slide")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
