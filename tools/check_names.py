#!/usr/bin/env python3
"""
Every name a notebook uses is defined before it is used.

    python3 tools/check_names.py [N ...]

Concatenates a notebook's code cells in order and asks pyflakes for undefined
names. That is the whole check, and it exists because the alternative found the
same class of defect one execution at a time, at up to half an hour each.

WHY THIS CLASS KEEPS HAPPENING. Lectures 15, 16, 13 and 17 were built by
splitting older two-lecture modules in half. A split can put the cells that USE
a variable in one lecture and the cell that DEFINES it in the other, and
reordering a section within a lecture can move a use above its definition.
Neither shows up in check_notebooks, which only compiles each cell: a cell that
reads an undefined global compiles perfectly.

Six real defects found on the first run: F, BATCH, lossf, cut, lin, and pool
twice.

ONE FALSE POSITIVE IT HAD TO LEARN. pyflakes reports a use inside a function as
undefined when a later cell does `del name` to free memory -- Lecture 7 drops a
250 MB array that way. The name is bound at top level before that use, so the
notebook is fine. A report is therefore only kept when the name has no top-level
binding at or before the line that uses it.

WHAT IT DOES NOT CATCH. A name defined in an earlier cell but whose VALUE is
wrong; a name that exists only because a previous notebook is still in memory
(nothing here shares a kernel); or an assignment inside an `if` that never runs.
For those, executing the notebook is still the only answer -- which is
check_consistency's job.
"""
from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RED, GREEN, DIM, BOLD, OFF = "\033[31m", "\033[32m", "\033[2m", "\033[1m", "\033[0m"

# Names supplied by the runtime rather than by a cell.
NOTEBOOK_BUILTINS = {"get_ipython", "display", "In", "Out", "exit", "quit"}


def source(nb_path: Path) -> tuple[str, list[tuple[int, int]]]:
    """The code cells joined, plus (first_line, cell_index) for each cell."""
    doc = json.loads(nb_path.read_text(encoding="utf-8"))
    lines: list[str] = []
    starts: list[tuple[int, int]] = []
    idx = 0
    for c in doc["cells"]:
        if c["cell_type"] != "code":
            continue
        src = "".join(c["source"])
        # A cell-magic line is not Python; blank it rather than dropping the
        # line, so reported line numbers still point at the right cell.
        body = ["" if ln.lstrip().startswith(("%", "!")) else ln
                for ln in src.split("\n")]
        starts.append((len(lines) + 1, idx))
        lines.extend(body)
        lines.append("")
        idx += 1
    return "\n".join(lines), starts


def cell_of(line: int, starts: list[tuple[int, int]]) -> int:
    out = -1
    for first, idx in starts:
        if first <= line:
            out = idx
        else:
            break
    return out


def module_bindings(tree: ast.Module) -> dict[str, int]:
    """name -> the earliest line at which the notebook binds it at top level.

    Only top level: a name bound inside a function is not available to a later
    cell, so it should not excuse anything.
    """
    first: dict[str, int] = {}

    def note(name: str, line: int) -> None:
        if name not in first or line < first[name]:
            first[name] = line

    def targets(node: ast.AST, line: int) -> None:
        for t in ast.walk(node):
            if isinstance(t, ast.Name) and isinstance(t.ctx, ast.Store):
                note(t.id, line)

    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
            targets(node, node.lineno)
        elif isinstance(node, (ast.For, ast.AsyncFor)):
            targets(node.target, node.lineno)
        elif isinstance(node, (ast.With, ast.AsyncWith)):
            for item in node.items:
                if item.optional_vars is not None:
                    targets(item.optional_vars, node.lineno)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            note(node.name, node.lineno)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for al in node.names:
                note((al.asname or al.name).split(".")[0], node.lineno)
        elif isinstance(node, (ast.If, ast.Try)):
            # a conditional definition still defines the name, and the notebook
            # is responsible for the branch it takes
            targets(node, node.lineno)
    return first


def undefined(text: str) -> list[tuple[int, str]]:
    from pyflakes.api import check
    from pyflakes.reporter import Reporter
    import io

    err = io.StringIO()
    out = io.StringIO()
    check(text, "notebook", Reporter(out, err))
    found = []
    for ln in out.getvalue().splitlines():
        if "undefined name" not in ln:
            continue
        parts = ln.split(":")
        try:
            line = int(parts[1])
        except (IndexError, ValueError):
            continue
        name = ln.rsplit("'", 2)[-2] if "'" in ln else ln
        if name in NOTEBOOK_BUILTINS:
            continue
        found.append((line, name))
    return found


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("lectures", nargs="*", type=int)
    a = ap.parse_args()

    paths = [ROOT / f"notebooks/lecture-{n:02d}.ipynb" for n in a.lectures] \
        if a.lectures else sorted((ROOT / "notebooks").glob("lecture-*.ipynb"))

    total = 0
    for p in paths:
        if not p.exists():
            continue
        text, starts = source(p)
        try:
            ast.parse(text)
        except SyntaxError as e:
            print(f"{RED}FAIL{OFF}  {p.name} — joined cells do not parse: {e}")
            total += 1
            continue
        bound = module_bindings(ast.parse(text))
        bad = []
        for line, name in undefined(text):
            # pyflakes reports a use inside a function as undefined when a later
            # cell does `del name` to free memory. The name is bound at top
            # level before that use, so the notebook is fine and the report is
            # not. Only a name with no earlier top-level binding is a defect.
            at = bound.get(name)
            if at is not None and at <= line:
                continue
            bad.append((line, name))
        if bad:
            print(f"{RED}FAIL{OFF}  {p.name} — {len(bad)} name(s) used before "
                  f"they are defined")
            for line, name in bad[:10]:
                print(f"        cell {cell_of(line, starts)}: {name}")
            if len(bad) > 10:
                print(f"        … and {len(bad) - 10} more")
            total += len(bad)
        else:
            print(f"{GREEN}ok{OFF}    {p.name}")

    print()
    if total:
        print(f"{BOLD}{RED}{total} undefined name(s){OFF}")
        return 1
    print(f"{BOLD}{GREEN}every name is defined before it is used{OFF}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
