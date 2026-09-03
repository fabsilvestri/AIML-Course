#!/usr/bin/env python3
"""
The machine-checkable half of AUTHORING.md, on the generated notebooks.

    python3 tools/check_notebooks.py            # all 24
    python3 tools/check_notebooks.py 19 20      # named lectures
    python3 tools/check_notebooks.py --advisory # include the noisy checks

Seven rules, from AUTHORING.md §4 and §5. Five are deterministic and FAIL the
run; two are advisory and only print, because they have irreducible false
positives and a check that cries wolf is a check nobody reads.

    HARD
      §5.1      no markdown line indented >= 4 outside a fence; no fence
                marker indented >= 4 -- an indented closing fence does not
                close, and the argument ships as grey monospace
      §4.1      every code cell is preceded by a prompt box (see below)
      §4.1a     every prompt box carries a `try` (see check_every_box_has_a_try)
      §4.5      any code cell whose stored execution took more than 20 s must
                have a clock marker in the markdown above it

    ADVISORY (--advisory)
      §4.3      a ```python block in markdown that appears in no code cell.
                Quoting code you are deliberately NOT running is a legitimate
                pattern, so this lists rather than fails.
      §4.3      prose figures that appear in no stored output. Derived numbers
                ("10% of 55,399 is about 5,500") are legitimate and cannot be
                distinguished mechanically, so this lists rather than fails.
      §4.4      a name assigned from two different constructors across cells.
                Legitimate in places -- rebinding `model` from LinearRegression
                to an RNN is the case it is for.

    NOTE. The old §6.1 "annotation budget" is gone with the device it policed:
    the three-bullet block under each prompt box no longer exists, so counting
    them is meaningless. What replaces it is check_every_cell_has_a_box below.

Why the split matters. The first version of this file failed the run on §1.2
and every notebook went red on legitimate arithmetic, which trains you to pass
`--no-verify`. A check that is ignored is worse than no check, because it looks
like coverage.
"""

from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "notebooks"

BOLD, RED, YEL, GRN, OFF = "\033[1m", "\033[31m", "\033[33m", "\033[32m", "\033[0m"

# A number worth checking: four digits or more, or anything with a decimal
# point. Below that the false-positive rate is hopeless -- "3 folds", "56
# lags", "2019" are all prose.
NUM = re.compile(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+\.\d+|\d{4,}")
FENCE = re.compile(r"^\s*```")
LIST_MARKER = re.compile(r"^([*+-]\s|\d+\.\s|>)")
CLOCK = "⏱"                                    # the stopwatch used in the decks


def cells(nb):
    return nb["cells"]


def src(c):
    return "".join(c["source"])


def outputs_text(c):
    t = []
    for o in c.get("outputs", []):
        t.append("".join(o.get("text", "")))
        t.append("".join(o.get("data", {}).get("text/plain", "")))
    return "".join(t)


def exec_seconds(c):
    """Seconds this cell took, if the notebook recorded it. Colab stores
    `execution` timing in cell metadata; without it we cannot judge."""
    ex = c.get("metadata", {}).get("execution", {})
    a, b = ex.get("shell.execute_reply.started"), ex.get("shell.execute_reply")
    if not (a and b):
        return None
    from datetime import datetime
    f = "%Y-%m-%dT%H:%M:%S.%fZ"
    try:
        return (datetime.strptime(b, f) - datetime.strptime(a, f)).total_seconds()
    except ValueError:
        return None


# ------------------------------------------------------------------ hard

def check_indentation(nb, fails):
    """§5.1/5.2 — prose that will silently render as a code block."""
    for i, c in enumerate(cells(nb)):
        if c["cell_type"] != "markdown":
            continue
        infence = False
        for k, line in enumerate(src(c).split("\n")):
            if FENCE.match(line):
                infence = not infence
                if len(line) - len(line.lstrip()) >= 4:
                    fails.append(f"cell {i} line {k}: fence marker indented "
                                 f"{len(line) - len(line.lstrip())} spaces — it does "
                                 f"not close the block (§5.2)")
                continue
            if infence or not line.strip():
                continue
            lead = len(line) - len(line.lstrip())
            if lead < 4 or LIST_MARKER.match(line.lstrip()):
                continue
            # A 4-space indent is often DELIBERATE: an author putting a formula
            # in monospace, or a continuation line inside a $$...$$ span. What
            # is never deliberate is a deep indent, or an indent that swallows
            # markdown syntax the author plainly meant to render. Lecture 19's
            # defect was 40 spaces over bold text, a blockquote and a heading;
            # lecture 14's legitimate case is 4 spaces over a bare formula.
            has_markup = any(t in line for t in ("**", "> ", "`", "](" ))
            if lead >= 8 or has_markup:
                fails.append(f"cell {i} line {k}: prose indented {lead} spaces, "
                             f"will render as a code block (§5.1) — "
                             f"{line.strip()[:52]!r}")


def check_quoted_code(nb, fails):
    """§3.1 — a ```python block in prose must exist in a cell of this notebook."""
    code_bodies = "\n".join(src(c) for c in cells(nb) if c["cell_type"] == "code")
    for i, c in enumerate(cells(nb)):
        if c["cell_type"] != "markdown":
            continue
        for m in re.finditer(r"```python\n(.*?)```", src(c), re.S):
            block = m.group(1)
            # A quoted block may legitimately elide with `...`, and may carry a
            # trailing comment the cell does not have. Match on the longest
            # line that is real code.
            lines = [l.strip() for l in block.split("\n")
                     if l.strip() and not l.strip().startswith("#")
                     and "..." not in l]
            if not lines:
                continue
            probe = max(lines, key=len)
            probe = probe.split("#")[0].strip()      # drop trailing comment
            if probe and probe not in code_bodies:
                fails.append(f"cell {i}: quoted code appears in no code cell "
                             f"(§3.1) — {probe[:58]!r}")


def check_annotation_budget(nb, fails, limit=0):
    """The old §6.1 budget, inverted: the annotation must be GONE.

    `_prompt.py` stopped rendering left_open/student/catch in the 2026-09
    redesign, so any surviving "Watch this prompt" block means a notebook was
    committed from a stale build rather than regenerated.
    """
    left = sum(1 for c in cells(nb)
               if c["cell_type"] == "markdown" and "Watch this prompt" in src(c))
    if left:
        fails.append(f"{left} cell(s) still carry the retired three-bullet "
                     f"annotation — regenerate with tools/make_notebooks.py")


def check_every_cell_has_a_box(nb, fails):
    """§4.1 — every code cell is preceded by its specification.

    make_notebooks.py enforces this at build time for the notebooks it writes;
    this catches an .ipynb edited by hand afterwards, which is the only way the
    two can disagree.
    """
    cs = cells(nb)
    for i, c in enumerate(cs):
        if c["cell_type"] != "code":
            continue
        prior = [x for x in cs[:i] if x["cell_type"] == "markdown"]
        if not prior or not src(prior[-1]).lstrip().startswith("> **Prompt"):
            fails.append(f"cell {i}: no prompt box precedes this code cell "
                         f"(§4.1)")


def check_every_box_has_a_try(nb, fails):
    """§4.1a — every prompt box names one modification and its consequence.

    Hard rather than advisory, and that was a decision rather than a default.
    The field was missing from 289 of the 538 boxes until 2026-09-03, because
    nothing enforced it and ten lectures were converted in bulk. It is now
    present on all 538, so a hard rule cannot fire on anything that exists —
    which is exactly the moment to make it hard, since from here the only way
    to violate it is to write a new box without one.

    The counter-argument was that `_prompt.py` says to omit `try` where there
    is genuinely nothing to vary. In practice no box turned out to be like
    that: even the setup cell has a seed to change, and "nothing came to mind"
    is the failure the docstring already warns against. If a genuine exception
    ever appears, it belongs here as a named exemption rather than as silence.
    """
    for i, c in enumerate(cells(nb)):
        if c["cell_type"] != "markdown":
            continue
        text = src(c).lstrip()
        if not text.startswith("> **Prompt"):
            continue
        if "**try**" not in text:
            head = text.split("\n", 1)[0][:60]
            fails.append(f"cell {i}: prompt box with no try — {head} (§4.1a)")


def check_clock_markers(nb, fails, threshold=20.0):
    """§7.1 — a slow cell must say so in the markdown above it."""
    cs = cells(nb)
    for i, c in enumerate(cs):
        if c["cell_type"] != "code":
            continue
        secs = exec_seconds(c)
        if secs is None or secs <= threshold:
            continue
        above = "".join(src(x) for x in cs[max(0, i - 3):i]
                        if x["cell_type"] == "markdown")
        if CLOCK not in above:
            fails.append(f"cell {i}: took {secs:.0f} s and no {CLOCK} marker "
                         f"above it (§7.1)")


# -------------------------------------------------------------- advisory

def advise_numbers(nb, notes):
    """§1.2 — prose figures matching no stored output."""
    out = " ".join(outputs_text(c) for c in cells(nb))
    seen = {m.group().replace(",", "") for m in NUM.finditer(out)}
    rounded = set()
    for v in seen:
        try:
            f = float(v)
        except ValueError:
            continue
        rounded |= {f"{f:.0f}", f"{f:.1f}", f"{f:.2f}", str(int(f)),
                    str(int(f) + 1)}                  # round-half-up either way
    ok = seen | rounded
    for i, c in enumerate(cells(nb)):
        if c["cell_type"] != "markdown":
            continue
        for m in NUM.finditer(src(c)):
            v = m.group().replace(",", "")
            if v in ok:
                continue
            ctx = src(c)[max(0, m.start() - 42):m.end() + 22].replace("\n", " ")
            notes.append(f"cell {i}: {m.group()} not in any output — ...{ctx.strip()}...")


def advise_rebinding(nb, notes):
    """§4.1 — one name, two constructors."""
    bound: dict[str, set[str]] = {}
    where: dict[str, set[int]] = {}
    for i, c in enumerate(cells(nb)):
        if c["cell_type"] != "code":
            continue
        try:
            tree = ast.parse(src(c))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Call):
                continue
            f = node.value.func
            ctor = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", None)
            if not ctor:
                continue
            for t in node.targets:
                if isinstance(t, ast.Name):
                    bound.setdefault(t.id, set()).add(ctor)
                    where.setdefault(t.id, set()).add(i)
    for name, ctors in sorted(bound.items()):
        if len(ctors) > 1 and len(where[name]) > 1:
            notes.append(f"`{name}` built from {sorted(ctors)} "
                         f"in cells {sorted(where[name])}")


# ------------------------------------------------------------------ main

def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    advisory = "--advisory" in sys.argv
    want = {int(a) for a in args} if args else None

    paths = sorted(OUT.glob("lecture-*.ipynb"))
    if want:
        paths = [p for p in paths if int(re.search(r"(\d+)", p.name).group(1)) in want]

    total_fail = 0
    for p in paths:
        nb = json.loads(p.read_text())
        fails: list[str] = []
        check_indentation(nb, fails)
        check_annotation_budget(nb, fails)
        check_every_cell_has_a_box(nb, fails)
        check_every_box_has_a_try(nb, fails)
        check_clock_markers(nb, fails)

        notes: list[str] = []
        if advisory:
            check_quoted_code(nb, notes)
            advise_numbers(nb, notes)
            advise_rebinding(nb, notes)

        if fails:
            total_fail += len(fails)
            print(f"{RED}FAIL{OFF}  {p.name}")
            for f in fails:
                print(f"        {f}")
        elif not advisory:
            print(f"{GRN}ok{OFF}    {p.name}")
        if notes:
            print(f"{YEL}note{OFF}  {p.name}  ({len(notes)} advisory)")
            for n in notes[:8]:
                print(f"        {n}")
            if len(notes) > 8:
                print(f"        ... and {len(notes) - 8} more")

    print()
    if total_fail:
        print(f"{BOLD}{RED}{total_fail} violation(s) of AUTHORING.md{OFF}")
        return 1
    print(f"{BOLD}{GRN}notebooks conform to the machine-checkable rules{OFF}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
