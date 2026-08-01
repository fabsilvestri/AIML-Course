#!/usr/bin/env python3
"""
Insert a specification box before each code cell of a notebook module.

    python3 tools/add_prompts.py 20 --list        # what needs a spec, in order
    python3 tools/add_prompts.py 20 --apply specs.py

`--list` prints each code cell with a short excerpt and a stable KEY (the first
meaningful line of the cell). `--apply` takes a file defining `SPECS`, a dict
from KEY to the four-part specification, and rewrites the module so a `prompt()`
call precedes every matching `code(...)`.

Keying on cell content rather than line number is deliberate: the first attempt
used line numbers computed before an earlier edit had shifted them, and inserted
nothing at all while reporting success. Content keys survive their own edits.

Idempotent: a code cell already preceded by `prompt(` is skipped, so a partly
specified module can be finished in a second pass.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# Order matters: this is the order they are written into the call, and
# _prompt.py renders the four specification fields as the quoted prompt
# and the last three as the "Watch this prompt" lines beneath it.
FIELDS = ("label", "input", "output", "constraint", "check",
          "left_open", "student", "catch")


def module_path(n: int) -> Path:
    return ROOT / "tools" / "notebooks" / f"lecture_{n:02d}.py"


def cells_of(src: str) -> list[tuple[int, str, str]]:
    """(start offset, key, excerpt) for every code cell in the module."""
    out = []
    # `(?<!def )` and `(?<![\w.])` matter: lecture 9 defines its own
    # `def code(text: str)` helper, and without the first guard that
    # definition is matched as a code cell — the tool would then splice a
    # prompt box into the middle of the function that BUILDS cells. The second
    # keeps `nbf.new_code(` and similar from matching too.
    for m in re.finditer(r"[ \t]*(?<!def )(?<![\w.])code\(", src):
        start = m.start()
        # the cell body: either a ''' literal or a NAME reference
        # 2000, not 400: lecture 6 has a cell whose opening comment block runs
        # to 700 characters, and with a short window every line in the window
        # was a comment, so the cell yielded no key and was skipped in silence
        # — no box, no warning, no way to notice except by counting.
        after = src[m.end():m.end() + 2000]
        if after.startswith("'''"):
            body = after[3:]
        elif after.lstrip().startswith(("SETUP", "LOADER")) or re.match(r"\w+\)", after):
            body = after
        else:
            body = after
        lines = [l for l in body.split("\n")
                 if l.strip() and not l.strip().startswith("#")]
        if not lines:
            # Every code cell must be keyable. Silently skipping one is how a
            # cell ends up with no specification and nobody finds out.
            raise SystemExit(
                f"cannot key the code cell at offset {start}: no line in the "
                f"first 2000 characters is anything but a comment. Widen the "
                f"window in cells_of() or shorten the comment.")
        excerpt = " / ".join(l.strip()[:60] for l in lines[:2])
        out.append((start, [l.strip() for l in lines[:4]], excerpt))
    return _disambiguate(out)


def _disambiguate(raw: list) -> list[tuple[int, str, str]]:
    """Give every cell a key that identifies it UNIQUELY within the module.

    The key is the first meaningful line, which is enough almost always. It is
    not enough for `with warnings.catch_warnings():`, which opens three
    different cells of lecture 5 — and a dict cannot hold the same key three
    times, so two of the three specifications would be silently discarded by
    Python and the surviving one applied to all three cells. Wrong boxes on
    the wrong cells is worse than no boxes.

    So: where the first line collides, extend the key with the next line,
    joined by ` // `, until the colliding cells are told apart.
    """
    out = [(start, lines[0], excerpt) for start, lines, excerpt in raw]
    for depth in range(1, 4):
        counts: dict[str, int] = {}
        for _, key, _ in out:
            counts[key] = counts.get(key, 0) + 1
        if all(n == 1 for n in counts.values()):
            break
        out = [(start,
                " // ".join(lines[:depth + 1]) if counts[key] > 1 else key,
                excerpt)
               for (start, key, excerpt), (_, lines, _) in zip(out, raw)]
    return out


def already_specified(src: str, start: int) -> bool:
    """Is this code cell already preceded by a prompt() call?"""
    head = src[max(0, start - 900):start]
    return head.rstrip().endswith("),") and "prompt(" in head.rsplit("code(", 1)[-1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("lecture", type=int)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--apply", metavar="SPECFILE")
    args = ap.parse_args()

    path = module_path(args.lecture)
    if not path.is_file():
        print(f"no module for lecture {args.lecture}")
        return 1
    src = path.read_text()

    if args.list:
        for i, (start, key, excerpt) in enumerate(cells_of(src), 1):
            done = "已" if "prompt(" in src[max(0, start - 400):start] else "  "
            print(f"{i:>3} {done} {key[:64]}")
            print(f"       {excerpt[:96]}")
        return 0

    if not args.apply:
        ap.error("give --list or --apply")

    spec_src = Path(args.apply).read_text()
    ns: dict = {}
    exec(spec_src, ns)
    specs: dict = ns["SPECS"]

    # A duplicated key in the SPECS literal is discarded by Python before this
    # code ever runs: `{"a": 1, "a": 2}` is `{"a": 2}`, silently. The specs
    # file is written by hand against `--list`, so a repeated key means one
    # cell's specification was lost. Count the top-level keys in the SOURCE
    # and compare with the dict that survived.
    literal_keys = re.findall(r'^(["\']).*?\1(?=\s*:)', spec_src, re.M | re.S)
    n_literal = len(re.findall(r"^[\"'].*?[\"']\s*:\s*dict\(", spec_src, re.M | re.S))
    if n_literal and n_literal != len(specs):
        print(f"refusing: {args.apply} defines {n_literal} entries but only "
              f"{len(specs)} survived — a key is repeated, and Python kept "
              f"the last one. Give the colliding cells their longer ` // ` "
              f"keys from --list.")
        return 1

    # insert from the END so earlier offsets stay valid
    inserted, missed, assigned = 0, [], []
    for start, key, _ in reversed(cells_of(src)):
        spec = specs.get(key)
        if spec is None:
            missed.append(key)
            continue
        line_start = src.rfind("\n", 0, start) + 1
        prefix = src[line_start:start]
        if "prompt(" in src[max(0, line_start - 400):line_start]:
            continue                              # already done
        # `code(` is not always at the start of its line — the setup cell often
        # sits inline in a list literal, `[md(HEADER), md("..."), code(SETUP)]`.
        # Inserting at the line start there splices the box into the middle of
        # an expression. Insert immediately before the `code(` token instead,
        # and indent the box to match whatever column it lands in.
        # `NAME = code('''...''')` is a module-level constant, not a cell in a
        # list, and a prompt spliced between the `=` and the `code(` produces
        # `MNIST_LOADER =prompt(...), code(...)` — which is not even valid
        # Python. The box belongs where that constant is USED, in the cells
        # list, so refuse here and say so rather than corrupting the module.
        m_const = re.search(r"\b(\w+)\s*=\s*$", prefix)
        if m_const:
            # A module-level cell constant. The box belongs where the constant
            # is USED, in the cells list — so find that use and insert there.
            # If the constant is used more than once, or not at all in a cells
            # list, refuse and say so: guessing which use to annotate would
            # put the box on the wrong cell.
            name = m_const.group(1)
            uses = [u.start() for u in re.finditer(rf"(?<![\w.]){name}\s*,", src)
                    if u.start() > start]
            if len(uses) != 1:
                assigned.append((key, name, len(uses)))
                continue
            at = uses[0]
            line_start = src.rfind("\n", 0, at) + 1
            prefix = src[line_start:at]
            if "prompt(" in src[max(0, line_start - 900):at]:
                continue
            pad = " " * (len(prefix) + 7)
            body = ",\n".join(f'{pad}{k}="{spec[k]}"' for k in FIELDS if k in spec)
            box = f"prompt(\n{body}),\n" + " " * len(prefix)
            src = src[:at] + box + src[at:]
            inserted += 1
            continue
        inline = bool(prefix.strip())
        at = start if inline else line_start
        indent = " " * len(prefix) if not inline else ""
        pad = " " * (len(prefix) + 7) if inline else indent + "       "
        body = ",\n".join(f'{pad}{k}="{spec[k]}"' for k in FIELDS if k in spec)
        box = f"prompt(\n{body}),\n" + (" " * len(prefix) if inline else indent)
        src = src[:at] + (indent + box if not inline else box) + src[at:]
        inserted += 1

    if "from _prompt import prompt" not in src:
        # The lecture modules import from make_notebooks, not from nbformat —
        # the old anchor here matched nothing, so the import was never added
        # and every freshly specified module failed to build with a bare
        # NameError. Anchor on the import that these files actually have.
        # Most lecture modules import their cell constructors from
        # make_notebooks; a few (lecture 9 among them) define md/code
        # themselves and import nbformat directly. Anchor on whichever is
        # present rather than assuming the common case.
        m = (re.search(r"^from make_notebooks import .*$", src, re.M)
             or re.search(r"^import nbformat as nbf$", src, re.M))
        if not m:
            print("refusing: cannot find the make_notebooks import to anchor "
                  "`from _prompt import prompt` after. Add it by hand.")
            return 1
        src = (src[:m.end()]
               + "\nfrom _prompt import prompt                                "
                 "# noqa: E402"
               + src[m.end():])
    path.write_text(src)
    print(f"lecture {args.lecture}: {inserted} specification(s) inserted")
    for k in missed:
        print(f"  no spec for: {k[:70]}")
    for k, name, n_uses in assigned:
        where = ("is never used in a cells list" if n_uses == 0
                 else f"is used {n_uses} times")
        print(f"  SKIPPED {k[:46]!r}: it is the body of `{name} = code(...)`, "
              f"a module-level constant, and it {where} — so there is no "
              f"single place the box belongs. Put its prompt() beside the "
              f"right use by hand.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
