#!/usr/bin/env python3
"""
Every exercise must be solvable from its own deck, plus the ones before it.

    python3 tools/check_exercises.py           # all 24
    python3 tools/check_exercises.py -v        # list the terms it accepted

THE RULE, set by the lecturer: an exercise at the end of lecture N may use
anything taught in lectures 1..N and nothing else. No forward references, and
nothing from outside the course.

The forward reference is the half a machine can settle. For every distinctive
term an exercise uses, find the earliest deck that term appears in. If the
earliest is a deck AFTER N, the exercise is asking about something the student
has not been shown, and it fails.

What counts as a distinctive term: a `<code>` span, or a token that is not
ordinary English -- an acronym (NDCG, ALS, IoU), an identifier
(pack_padded_sequence), a name (Glorot, Eckart-Young). Ordinary words are
skipped because every deck contains them, so they can never fail and would
only add noise.

The other half -- "is this actually derivable from the deck's argument?" --
needs a reader, and the audit of that lives in REBUILD.md.
"""
from __future__ import annotations

import argparse
import html
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

BEGIN, END = "<!-- BEGIN EXERCISES -->", "<!-- END EXERCISES -->"

# A token worth checking is one that is NOT ordinary English: an acronym
# (NDCG, IoU, ALS), an identifier (pack_padded_sequence, model.eval), a proper
# noun (Glorot, Eckart). Ordinary words are skipped, because every deck is full
# of them and flagging "monotonically" or "detects" as a forward reference is
# noise that buries the real thing -- which is exactly what the first version
# of this check did.
TERM = re.compile(r"[A-Za-z][A-Za-z0-9_]*(?:[._][A-Za-z0-9_]+)*")

WORDS = set()
_dict = Path("/usr/share/dict/words")
if _dict.is_file():
    WORDS = {w.strip().lower() for w in _dict.read_text().splitlines() if w.strip()}


def english(low: str) -> bool:
    """Ordinary English, allowing for inflection.

    /usr/share/dict/words holds base forms, so "detects", "implies" and
    "leaked" are all absent from it and were all reported as forward
    references by the first version of this check.
    """
    if low in WORDS:
        return True
    for suf, repl in (("s", ""), ("es", ""), ("ed", ""), ("ed", "e"),
                      ("d", ""), ("ing", ""), ("ing", "e"), ("ly", ""),
                      ("ies", "y"), ("ied", "y"), ("er", ""), ("est", "")):
        if low.endswith(suf) and (low[: -len(suf)] + repl) in WORDS:
            return True
    return False


def technical(t: str) -> bool:
    """Is this token course vocabulary rather than English?"""
    if "_" in t or "." in t:
        return True                       # an identifier
    if len(t) > 1 and t.isupper():
        return True                       # an acronym: NDCG, ALS, TLU, XOR
    if re.match(r"^[A-Z][a-z]*[A-Z]", t):
        return True                       # CamelCase: IoU, GIoU, RandomForest
    low = t.lower()
    if english(low):
        return False                      # ordinary English
    # not in the dictionary and not obviously an identifier: a name, a coinage,
    # or a domain word. Worth checking.
    return len(low) >= 3


def deck_text(n: int) -> str:
    """A deck's prose, with its own exercise block removed.

    The block has to go: an exercise mentioning NDCG would otherwise be
    evidence that NDCG was taught here, and every exercise would pass itself.
    """
    s = (ROOT / f"slides/lecture-{n:02d}.html").read_text(encoding="utf-8")
    if BEGIN in s:
        s = s[:s.index(BEGIN)] + s[s.index(END) + len(END):]
    s = re.sub(r"<[^>]+>", " ", s)
    return html.unescape(s).lower()


def exercises(n: int) -> list[tuple[int, str]]:
    """(number, all of its text) for each exercise on deck n."""
    s = (ROOT / f"slides/lecture-{n:02d}.html").read_text(encoding="utf-8")
    if BEGIN not in s:
        return []
    block = s[s.index(BEGIN):s.index(END)]
    # the questions carry the marks badge; the solutions carry the answers
    out = []
    for m in re.finditer(r'<li>(.*?)<span class="ex-marks">', block, re.S):
        out.append(html.unescape(re.sub(r"<[^>]+>", " ", m.group(1))))
    return list(enumerate(out, start=1))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()

    corpus = {n: deck_text(n) for n in range(1, 25)}
    bad = 0
    for n in range(1, 25):
        seen_by = " ".join(corpus[m] for m in range(1, n + 1))
        later = {m: corpus[m] for m in range(n + 1, 25)}
        for k, text in exercises(n):
            forward, outside = [], []
            for t in TERM.findall(text):
                low = t.lower()
                if low.isdigit() or not technical(t):
                    continue
                if low in seen_by:
                    continue
                first = next((m for m, txt in later.items() if low in txt), None)
                if first:
                    forward.append((t, first))
                else:
                    # in no deck at all: either outside knowledge, or a word
                    # invented for the question's scenario. A human decides,
                    # so this is advisory rather than a failure.
                    outside.append(t)
            if outside and a.verbose:
                print(f"note  lecture {n:02d}, exercise {k}: "
                      f"in no deck — {', '.join(sorted(set(outside)))}")
            if forward:
                bad += 1
                print(f"FAIL  lecture {n:02d}, exercise {k}")
                for t, m in sorted(set(forward), key=lambda x: x[1]):
                    print(f"        '{t}' is not in decks 1-{n}; first appears "
                          f"in lecture {m}")
                print(f"        {' '.join(text.split())[:110]}…")
    print()
    if bad:
        print(f"{bad} exercise(s) reach forward to material the student has "
              f"not been shown")
        return 1
    print("every exercise uses only its own deck and the ones before it")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
