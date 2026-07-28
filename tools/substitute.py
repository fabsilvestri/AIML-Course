#!/usr/bin/env python3
"""
Replace @@TOKEN@@ placeholders in the decks with measured values.

    python3 tools/substitute.py --dry-run    # show what would change
    python3 tools/substitute.py              # write it

Two authors drafted decks with placeholders and were interrupted before writing
the substitution step, leaving 68 tokens that would have shown literal
`@@l09_best_k@@` on a projector.

**This runs once per deck, in place.** After it, the deck quotes literal
numbers, exactly as Lectures 1-2 always did, and `tools/check_provenance.py` is
what keeps them honest from then on — it fails the build if a quoted number
stops matching `figures.json`. So the token is scaffolding for drafting, not a
live binding, and re-running after the data changes will not update anything.
If a measurement changes, provenance will tell you which slide to edit.

Token grammar, as the drafting authors designed it:

    @@key@@                     a top-level value in figures.json
    @@key.sub.leaf@@            a nested one
    @@key:,@@                   with a Python format spec after the colon
    @@l09_ari_audit_at_best:+.4f@@

Putting the format at the point of use is the right design — `{:,.0f}` for a
count, `{:.3f}` for a silhouette score and `{:.1%}` for a rate are all
different, and guessing from the value alone gives "0.74 accuracy" where the
slide wants "74.0%". A token with no spec falls back to `_auto`, which is
reported so it can be checked by eye.

ALIASES exists only for tokens whose name does not match the key the figure
script exported.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FIGURES = ROOT / "assets" / "figures" / "figures.json"

# token -> (dotted path into figures.json, format spec)
# Where a deck used a short name for something the figure script exported under
# a longer one, the mapping is recorded here rather than either name being
# changed — the script's name says what was measured, the deck's says what the
# sentence needs.
ALIASES: dict[str, tuple[str, str]] = {
    # Two different quantities, and the draft used one token for both. The
    # twenty-split table reports a cut-off re-chosen on every split (mean 0.86,
    # sd 0.04); the defence paragraph and the summary quote the single operating
    # point the lecture adopts, which is what `flagged` and `fn` are counted at.
    "T_STAR":        ("l05_threshold.best.t",                      "{:.2f}"),
    "T_STAR_CV":     ("l05_accuracy_trap.chosen_threshold",        "{:.2f}"),
    "ACC_HALF":      ("l05_accuracy_trap.accuracy_at_half",        "{:.1%}"),
    "ACC_BEST":      ("l05_accuracy_trap.accuracy_at_best_cost",   "{:.1%}"),
    "COST_HALF":     ("l05_accuracy_trap.cost_at_half",            "{:,.0f}"),
    "COST_BEST":     ("l05_accuracy_trap.cost_at_best",            "{:,.0f}"),
    "COST_PENALTY":  ("l05_accuracy_trap.cost_penalty",            "{:,.0f}"),
    "SEEDS_WORSE":   ("l05_accuracy_trap.seeds_where_half_is_worse", "{:.0f}"),
    # "the worst distortion at 200 components": ds[2] == 200 in the measured
    # sweep, so the value is index 2 of the parallel `max` array.
    "l10_jl_measured_max_at_200": ("l10_jl_measured.max.2",      "{:.2f}"),
    "N_FLAGGED":     ("l05_threshold.best.flagged",                "{:,.0f}"),
    "N_FN":          ("l05_threshold.best.fn",                     "{:,.0f}"),
}

# explicit formats for tokens that resolve directly to a top-level key
FORMATS: dict[str, str] = {
    "l09_best_silhouette": "{:.3f}",
    "l09_random_baseline": "{:.3f}",
    "l09_sil_at_40":       "{:.3f}",
    "l09_labelled_pct":    "{:.1f}",
    "l10_jl":              "{:,.0f}",
    "l10_speedup":         "{:.1f}",
}


def _lookup(node, dotted: str):
    """Walk a dotted path.

    Two wrinkles the data forces. Numeric segments index into a list, so
    `l10_jl_measured.max.2` is the third element. And a dict key may itself
    contain a dot — the Johnson-Lindenstrauss table is keyed by epsilon, so
    `l10_jl.n400.0.1` means `["l10_jl"]["n400"]["0.1"]`, not four segments.
    So at each step try the longest joinable key first.
    """
    parts = dotted.split(".")
    while parts:
        if isinstance(node, dict):
            for take in range(len(parts), 0, -1):     # longest key first
                key = ".".join(parts[:take])
                if key in node:
                    node, parts = node[key], parts[take:]
                    break
            else:
                return None
        elif isinstance(node, (list, tuple)) and parts[0].lstrip("-").isdigit():
            i = int(parts[0])
            if not -len(node) <= i < len(node):
                return None
            node, parts = node[i], parts[1:]
        else:
            return None
    return node


def _auto(value) -> str:
    """Best guess for a token with no declared format. Check these by eye."""
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        if value.is_integer():
            return f"{int(value):,}"
        return f"{value:,.2f}" if abs(value) >= 1 else f"{value:.3f}"
    return str(value)


def resolve(token: str, data: dict) -> tuple[str | None, bool]:
    """Return (rendered, was_the_format_explicit)."""
    path, _, spec = token.partition(":")          # "key.sub:,.1f"
    if path in ALIASES:
        aliased, alias_spec = ALIASES[path]
        value = _lookup(data, aliased)
        spec = spec or alias_spec.strip("{}:")
    else:
        value = _lookup(data, path)
    if value is None:
        return None, bool(spec)
    if spec:
        try:
            return format(value, spec), True
        except (ValueError, TypeError):
            return None, True
    if path in FORMATS:
        return FORMATS[path].format(value), True
    return _auto(value), False


def main() -> int:
    dry = "--dry-run" in sys.argv
    data = json.loads(FIGURES.read_text())

    total, guessed, missing = 0, [], []
    for deck in sorted(ROOT.glob("slides/*.html")):
        html = deck.read_text()
        # keys may contain spaces and percent signs ("40 at random",
        # "propagated to closest 75%"), so the path is "anything but @ or :"
        tokens = set(re.findall(r"@@([^@:]+(?::[^@]*)?)@@", html))
        if not tokens:
            continue
        out, changed = html, 0
        for tok in sorted(tokens):
            rendered, explicit = resolve(tok, data)
            if rendered is None:
                missing.append(f"{deck.name}: @@{tok}@@")
                continue
            n = out.count(f"@@{tok}@@")
            out = out.replace(f"@@{tok}@@", rendered)
            changed += n
            if not explicit:
                guessed.append(f"{deck.name}: @@{tok}@@ -> {rendered}")
            if dry:
                print(f"  {deck.name}: @@{tok}@@ -> {rendered}  ({n}x)")
        total += changed
        if not dry and changed:
            deck.write_text(out)
            print(f"  {deck.name}: {changed} substitution(s)")

    if guessed:
        print(f"\n{len(guessed)} token(s) formatted by guess — check these by eye:")
        for g in guessed:
            print("  " + g)
    if missing:
        print(f"\n{len(missing)} token(s) NOT in figures.json:")
        for m in missing:
            print("  " + m)
        print("\nEither export them from the application's figure script, or add "
              "an entry to ALIASES.")
        return 1
    print(f"\n{total} substitution(s){' (dry run)' if dry else ''}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
