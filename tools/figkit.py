#!/usr/bin/env python3
"""
Shared figure machinery, so each application can own its own script.

`make_figures.py` holds Lectures 1–2 and remains the reference implementation.
Everything reusable in it is re-exported here, so a new application writes

    tools/figures_app03.py          # Titanic, Lectures 5–6

without touching a file another author is editing. Twenty-two lectures cannot
share one script.

    from figkit import setup, save, cached, PRIMARY, ACCENT, SUCCESS, MATH, \\
                       MUTED, RULE, AXIS, BODY, SMALL, TICK, usd, OUT, export

Everything here is already used by Lectures 1–2; nothing is new. In particular
`setup()` registers the vendored Source Sans 3 cuts and raises if they fail to
resolve, because a missing font is otherwise a silent revert to DejaVu.

Read TRICKS §6 and §11.6 before writing a figure. The two rules that are not
obvious:

  * `.fig-wide` renders at aspect 0.328 or narrower; anything taller is capped
    by height. Never set `width` next to `max-height` — see §11.6.
  * matplotlib sizes are POINTS. `make_figures.py` refuses to ship a figure
    whose smallest label lands under 15px on the slide; run its check before
    you commit.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Import the reference implementation by path rather than by name, so this works
# whether or not tools/ is on sys.path.
_spec = importlib.util.spec_from_file_location(
    "_make_figures", Path(__file__).with_name("make_figures.py"))
_mf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mf)

# --- palette (mirrors assets/css/custom.css; change both together) ----------
PRIMARY = _mf.PRIMARY
ACCENT = _mf.ACCENT
SUCCESS = _mf.SUCCESS
MATH = _mf.MATH
MUTED = _mf.MUTED
RULE = _mf.RULE
AXIS = _mf.AXIS

# --- type sizes, in points; see TRICKS §11.6 for what they become on screen -
BODY = _mf.BODY
SMALL = _mf.SMALL
TICK = _mf.TICK

# --- machinery --------------------------------------------------------------
setup = _mf.setup                 # rcParams + fonts; call once, first
save = _mf.save                   # save(fig, "l05-slug", raster=False)
cached = _mf.cached               # cached("key", lambda: expensive())
load_cache = _mf.load_cache
usd = _mf.usd                     # a dollar-formatted axis tick formatter
OUT = _mf.OUT                     # assets/figures/
SEED = _mf.SEED

CANVAS_W = _mf.CANVAS_W
CAP_WIDE = _mf.CAP_WIDE
CAP_TALL = _mf.CAP_TALL
FLOOR_PX = _mf.FLOOR_PX
check_text_floor = _mf.check_text_floor


def export(**values) -> None:
    """Merge measured numbers into assets/figures/figures.json.

    Every quantity a slide quotes must land here or tools/check_provenance.py
    fails the build. Merging rather than overwriting is what lets each
    application own its own script.
    """
    path = OUT / "figures.json"
    data = json.loads(path.read_text()) if path.is_file() else {}
    data.update(values)
    path.write_text(json.dumps(data, indent=2, sort_keys=False))
    for k in values:
        print(f"    exported {k}")
