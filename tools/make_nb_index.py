#!/usr/bin/env python3
"""Regenerate the "every notebook in the course" block in every deck.

    python3 tools/make_nb_index.py

Fourteen decks carry a 24-entry index of all the notebooks. Kept by hand it
drifts the moment a lecture is renamed or renumbered, in fourteen places at
once, and nothing catches it because a wrong title is not a broken link.

Titles come from tools/make_site.py, which is the same table the course site
uses, so the site and every deck cannot disagree.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from make_site import LECTURES, COLAB                       # noqa: E402

BLOCK = re.compile(r'(<div class="nb-index">\n).*?(\s*</div>)', re.S)


def entries() -> str:
    out = []
    for n, title, *_rest, published in LECTURES:
        cls = "done" if published else "todo"
        out.append(f'    <p class="{cls}"><a target="_blank" '
                   f'href="{COLAB.format(n)}">{n:02d} &middot; {title}</a></p>')
    return "\n".join(out)


def main() -> int:
    body = entries()
    touched = 0
    for deck in sorted(ROOT.glob("slides/lecture-[0-9][0-9].html")):
        src = deck.read_text(encoding="utf-8")
        if not BLOCK.search(src):
            continue
        new = BLOCK.sub(lambda m: m.group(1) + body + m.group(2), src)
        if new != src:
            deck.write_text(new, encoding="utf-8")
            touched += 1
    print(f"notebook index rewritten in {touched} deck(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
