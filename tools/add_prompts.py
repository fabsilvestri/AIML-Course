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
    for m in re.finditer(r"[ \t]*code\(", src):
        start = m.start()
        # the cell body: either a ''' literal or a NAME reference
        after = src[m.end():m.end() + 400]
        if after.startswith("'''"):
            body = after[3:]
        elif after.lstrip().startswith(("SETUP", "LOADER")) or re.match(r"\w+\)", after):
            body = after
        else:
            body = after
        lines = [l for l in body.split("\n")
                 if l.strip() and not l.strip().startswith("#")]
        if not lines:
            continue
        key = lines[0].strip()
        excerpt = " / ".join(l.strip()[:60] for l in lines[:2])
        out.append((start, key, excerpt))
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

    ns: dict = {}
    exec(Path(args.apply).read_text(), ns)
    specs: dict = ns["SPECS"]

    # insert from the END so earlier offsets stay valid
    inserted, missed = 0, []
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
        inline = bool(prefix.strip())
        at = start if inline else line_start
        indent = " " * len(prefix) if not inline else ""
        pad = " " * (len(prefix) + 7) if inline else indent + "       "
        body = ",\n".join(f'{pad}{k}="{spec[k]}"' for k in FIELDS if k in spec)
        box = f"prompt(\n{body}),\n" + (" " * len(prefix) if inline else indent)
        src = src[:at] + (indent + box if not inline else box) + src[at:]
        inserted += 1

    if "from _prompt import prompt" not in src:
        src = src.replace("import nbformat as nbf\n",
                          "import nbformat as nbf\n\nfrom _prompt import prompt\n", 1)
    path.write_text(src)
    print(f"lecture {args.lecture}: {inserted} specification(s) inserted")
    for k in missed:
        print(f"  no spec for: {k[:70]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
