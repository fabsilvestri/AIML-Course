#!/usr/bin/env python3
"""
Re-indent `prompt(...)` calls that add_prompts.py inserted flush-left.

    python3 tools/reindent_prompts.py 3 4 5

add_prompts.py inserts at the start of the line holding `code(`, which lands the
call at column zero inside a list literal. That is valid Python and unreadable,
and these modules are read by people as often as they are executed. This puts
each call at the indentation of the `code(` it precedes and its arguments one
level further in.

Separate from add_prompts.py on purpose: it is idempotent and safe to run over a
module whether or not anything needs moving, so it can be run after every batch
without thinking about it.
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def reindent(text: str, body_indent: str = "        ") -> str:
    lines = text.split("\n")
    out, i, moved = [], 0, 0
    arg_indent = body_indent + "    "
    while i < len(lines):
        if lines[i].startswith("prompt("):
            moved += 1
            out.append(f"{body_indent}prompt(")
            i += 1
            depth = 1
            while i < len(lines) and depth > 0:
                stripped = lines[i].strip()
                # count only parens outside string literals; the arguments are
                # prose and contain plenty of both
                bare = re.sub(r'"(?:[^"\\]|\\.)*"', "", stripped)
                depth += bare.count("(") - bare.count(")")
                out.append(arg_indent + stripped)
                i += 1
            continue
        out.append(lines[i])
        i += 1
    return "\n".join(out)


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    for arg in sys.argv[1:]:
        path = ROOT / "tools" / "notebooks" / f"lecture_{int(arg):02d}.py"
        before = path.read_text()
        after = reindent(before)
        if after == before:
            print(f"lecture {arg}: already indented")
            continue
        ast.parse(after)          # refuse to write something that will not parse
        path.write_text(after)
        print(f"lecture {arg}: re-indented")
    return 0


if __name__ == "__main__":
    sys.exit(main())
