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
import re
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


def plain_log(ax, which="y", ticks=None, fmt="{:,.0f}"):
    """Label a log axis without mathtext. **Call this on every log axis.**

    `setup()` sets ``text.parse_math: False`` course-wide, because two dollar
    signs in one string are mathtext to matplotlib exactly as they are maths to
    KaTeX (TRICKS §9.1). Matplotlib's own log formatter then loses: it emits
    ``$\\mathdefault{10^{3}}$`` and, with parsing off, that string is drawn
    **literally** on the axis.

    It is silent, it survives every check we have, and it has already shipped
    into three finished decks — eight figures across Lectures 6, 12 and 16 are
    currently displaying raw LaTeX where their tick labels should be.

        ax.set_yscale("log")
        plain_log(ax, "y", [1, 10, 100, 1_000])
        plain_log(ax, "y", [0.005, 0.05, 0.5], fmt="{:.3f}")   # small values

    **Call it AFTER you plot, not before.** `semilogx`/`semilogy`/`loglog`
    re-apply the log scale on every call, and applying a scale installs that
    scale's default formatter — so a `plain_log` written above the plotting line
    is silently undone. Two of the thirteen figures were "fixed" that way first
    and came back still broken.

    Passing `ticks` is strongly advised: without explicit ticks matplotlib may
    place minor ticks that the major formatter never reaches, and the minor
    labels come back in mathtext. Where the majors are hand-labelled, kill the
    minors outright with `NullLocator` — a blanking formatter still leaves
    matplotlib free to re-derive them.
    """
    from matplotlib.ticker import FuncFormatter, NullFormatter
    axis = ax.yaxis if which == "y" else ax.xaxis
    if ticks is not None:
        (ax.set_yticks if which == "y" else ax.set_xticks)(ticks)
    axis.set_major_formatter(FuncFormatter(lambda v, _: fmt.format(v)))
    axis.set_minor_formatter(NullFormatter())


def export(**values) -> None:
    """Merge measured numbers into assets/figures/figures.json.

    Every quantity a slide quotes must land here or tools/check_provenance.py
    fails the build. Merging rather than overwriting is what lets each
    application own its own script.

    **Keys are a shared namespace and a collision is silent data loss.** Three
    scripts exported a bare `n_test`; the last to run set it to MNIST's 12,000
    and Lectures 1-2 quietly lost housing's 4,128, so seven slides in two decks
    became unprovenanced with nothing to say why. Prefix your keys with the
    lecture (`l09_n_test`), and this refuses to overwrite an existing key with a
    different value rather than letting it happen again.
    """
    path = OUT / "figures.json"
    data = json.loads(path.read_text()) if path.is_file() else {}
    for k, v in values.items():
        # A namespaced key (l07_*, l19_*) belongs to one lecture, so its owner
        # may update it freely — timings and anything else measured on the
        # machine legitimately change between runs. An UNPREFIXED key is the
        # dangerous case: it is the one two scripts can both claim, and the
        # collision is silent.
        # `app02` / `app10` are the same thing one level up: a whole
        # application's block, written by exactly one script that is named
        # after it. Those owners were not exempt, so the first time
        # figures_app02.py legitimately CHANGED one of its own numbers the
        # guard refused the export and told it to rename a key 438 other
        # entries already depend on. Ownership is what earns the exemption,
        # and a key named after its only writer has it.
        if re.match(r"l\d\d_|app\d\d", k):
            continue
        # Compare like with like. JSON has no integer keys, so a dict keyed by
        # ints comes back keyed by strings, and a straight != then reports a
        # collision on every re-run of a script that exports one. app10 exports
        # a weight-per-lag map keyed by int and could not be re-run at all.
        if k in data and data[k] != json.loads(json.dumps(v)):
            raise RuntimeError(
                f"export('{k}') would overwrite a value another script wrote:\n"
                f"    was {data[k]!r}\n    now {v!r}\n"
                f"figures.json is a shared namespace. Prefix the key with the "
                f"lecture it belongs to — e.g. 'l09_{k}' — which also marks you "
                f"as its owner and makes future updates to it free.")
    data.update(values)
    path.write_text(json.dumps(data, indent=2, sort_keys=False))
    for k in values:
        print(f"    exported {k}")
