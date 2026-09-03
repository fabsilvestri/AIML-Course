#!/usr/bin/env python3
"""
Every number a slide states must be one its own notebook prints.

    python3 tools/check_consistency.py            # every built lecture
    python3 tools/check_consistency.py 3 8        # named lectures
    python3 tools/check_consistency.py --execute  # re-run the notebooks first
    python3 tools/check_consistency.py -v         # show what matched, too

This is the check that has found a real defect in every lecture converted so
far, and the only one that does. The others verify each artefact against
itself: `check_provenance` checks a slide against figures.json, and
`check_notebooks` checks a notebook against its own stored output. Both pass
while the deck and the notebook quietly disagree with each other, because each
is internally consistent. What that costs, historically:

    L1  the price stripes counted on the full frame in the notebook and on the
        training split on the slide -- 79 districts against 62, both right
    L2  the notebook searched cv=5 where every slide figure used KFold(10), so
        it chose different hyperparameters and reported a different RMSE
    L3  the deck quoted a 90.39% recall that the notebook never printed at all
    L8  the notebook measured Johnson-Lindenstrauss on UNSQUARED distances,
        understating the distortion by about a factor of two, beside a slide
        stating the theorem it was supposed to illustrate

None of those is visible by reading. All are obvious in a diff.

HOW IT WORKS. Executed notebooks are cached under the scratchpad, keyed by a
hash of the .ipynb, so a clean run is cheap and a changed notebook re-executes
automatically. Every number in the notebook's stdout is collected; every number
in the deck's prose and tables is collected; the second set is checked against
the first.

WHAT IT DELIBERATELY IGNORES, and why each would otherwise drown the signal:

  * durations -- AUTHORING section 3.2a: a wall-clock second is a property of a
    machine, so the deck and the notebook cannot agree on one and should not
    pretend to. Anything adjacent to s/sec/ms/min/hour is skipped.
  * years, chapter and lecture numbers, section numbers, percentages under 10
    that read as ordinary prose ("one in ten"), and small integers below 24,
    which are almost always counts of slides, folds or lectures rather than
    results.
  * numbers inside <pre><code> -- source shown on a slide is not a claim about
    a measurement.
  * a figure that differs from a printed one only by rounding or by a factor of
    100 (0.9039 on one side, 90.39% on the other) -- that is a presentation
    choice, not a disagreement.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CACHE = Path(os.environ.get("TMPDIR", "/tmp")) / "aiml-consistency"

BOLD, RED, GREEN, YELLOW, DIM, OFF = (
    "\033[1m", "\033[31m", "\033[32m", "\033[33m", "\033[2m", "\033[0m")

# A number, with optional thousands separators and decimals.
NUM = re.compile(r"\d[\d,]*(?:\.\d+)?")
# Figures a notebook deliberately does not reproduce, because the deck reports a
# run the notebook is not sized to repeat. Every notebook that appears here says
# so in its own header, in a "Scale." paragraph naming both numbers.
#
# This list is the same contract as ALLOWED in check_provenance.py: adding to it
# is fine, adding to it WITHOUT A REASON is how the check stops meaning anything.
# It is never the right home for "the notebook computes a different number" —
# that is the defect this tool exists to find. It is only for "the notebook
# computes the same quantity on less data, on purpose, and says so."
#
# Exempted figures are listed in the output rather than skipped silently.
SCALE_ONLY: dict[str, str] = {
    "l22_errors_scratch":
        "L18 deck scores all 25,000 test reviews; the notebook scores 3,000 so "
        "it finishes on a CPU. It prints its own error counts at its own scale.",
    "l22_errors_finetuned":
        "same slide, same reason.",
    "l22_clusters":
        "L18 deck clusters all 12,500 negative reviews; the notebook clusters "
        "2,000. The cluster sizes therefore differ; the themes do not.",
}

# Figures a deck quotes from ANOTHER lecture's experiment. The number is real and
# it is reproduced -- in the other lecture's notebook, named here so the claim
# can be checked rather than taken on trust. Same contract as SCALE_ONLY: a
# reason, or it does not belong in the list.
CROSS_LECTURE: dict[str, str] = {
    "l15:app10/torch/rnn_random_cv":
        "L15 quotes the RNN under a random split to contrast two protocols on "
        "two models. The RNN is Lecture 16's; L15's notebook fits no network.",
    "l15:app10/torch/ladder":
        "same slide: the RNN's forward-split MAE. Reproduced by Lecture 16's "
        "ladder, row 'Simple RNN, 32 units'.",
    "l15:app10/margins/rnn/gap":
        "the difference between those two, so it inherits their provenance.",
    "l16:app10/linear/random_cv":
        "L16 contrasts the recurrent model's two protocols against the linear "
        "model's. The linear numbers are Lecture 15's, reproduced there over "
        "the same twenty seeds.",
    "l16:app10/linear/honest_split":
        "same slide, same lecture: the linear model's forward-split MAE.",
}

# Units that make a figure a duration rather than a result.
DURATION = re.compile(
    r"\b(s|sec|secs|second|seconds|ms|min|mins|minute|minutes|h|hour|hours)\b"
    r"|\bper\s+(second|minute|hour)\b", re.I)


def numbers(text: str) -> set[float]:
    out = set()
    for m in NUM.finditer(text):
        raw = m.group(0).rstrip(".").replace(",", "")
        if not raw or raw.endswith("."):
            continue
        try:
            out.add(float(raw))
        except ValueError:
            pass
    return out


# Which figures.json namespaces belong to which NEW lecture.
#
# The keys in figures.json are prefixed by the lecture that produced them under
# the OLD numbering, and the rebuild renumbers. Without this map a deck matches
# any coincidentally-equal value anywhere in the file -- "48 academic hours"
# matched a random-forest standard deviation, and "O'Reilly, 2025" matched a
# gradient. Keep it in step with REBUILD.md's Source column; a lecture absent
# from it is skipped with a warning rather than checked against everything.
# Taken from REBUILD.md's Source column, which is authoritative. Do NOT infer
# this by looking for which prefix produces the most matches: Lecture 7 scores
# 15 hits against l06_*, the Titanic keys, purely because accuracies and shares
# both live in [0, 1] and collide at four significant figures. An inferred map
# would quietly bless the wrong namespace and then pass.
NAMESPACES: dict[int, tuple[str, ...]] = {
    1:  ("",),                       # housing, unprefixed
    2:  ("",),                       # housing, unprefixed
    3:  ("l03", "l04", "app02"),     # old 3 + 4, MNIST
    4:  ("l05",),                    # old 5, Titanic
    5:  ("l06",),                    # old 6, Titanic
    6:  ("l07", "app04"),            # old 7, CoverType
    7:  ("app04",),                  # old 8, CoverType (no l08_* keys exist)
    8:  ("l09", "l10"),              # old 9 + 10, Olivetti
    9:  ("l11",),                   # old 11, Fashion-MNIST
    10: ("l12",),                   # old 12, Fashion-MNIST / PyTorch
    11: ("l13", "l14"),            # old 13 + 14, CIFAR-10
    12: ("l15",),                   # old 15, Flowers102
    13: ("l16",),                   # old 16, Flowers102 / transfer
    14: ("l17", "l18"),            # old 17 + 18, COCO detection
    15: ("app10",),                  # old 19 + 20, Chicago transit
    16: ("app10",),                  # old 20, recurrent networks
    17: ("l21",),                   # old 21, IMDb
    18: ("l22",),                   # old 22, IMDb / transformers
    19: ("l19",),                   # SciFact, lexical retrieval — figures_ir.py
    20: ("l20",),                   # SciFact, dense retrieval — figures_dense.py
    # rec21/rec22, not l21/l22: those names belong to Lectures 17 and 18, which
    # carried those numbers under the old plan. The facts did not move when the
    # lectures were renumbered, and renaming several hundred keys to repair it
    # is risk with no benefit to a student.
    21: ("rec21",),                 # MovieLens, factorisation — figures_recsys.py
    22: ("rec22",),                 # MovieLens, ranking and protocols
    23: ("l23",),                   # old 23, COCO multimodal
    24: ("l24",),                   # old 24, RAG and closing
}


def facts(lecture: int) -> list[tuple[str, float]]:
    """(key path, value) for every figures.json number this lecture owns."""
    doc = json.loads((ROOT / "assets/figures/figures.json").read_text())
    allowed = NAMESPACES.get(lecture)
    if allowed is None:
        return []
    out: list[tuple[str, float]] = []

    def walk(o, path=""):
        if isinstance(o, dict):
            for k, v in o.items():
                walk(v, f"{path}/{k}")
        elif isinstance(o, list):
            for i, v in enumerate(o):
                walk(v, f"{path}[{i}]")
        elif isinstance(o, (int, float)) and not isinstance(o, bool):
            top = path.lstrip("/").split("/")[0].split("[")[0]
            # `rec\d\d` was missing here until 2026-09-03, and the omission was
            # silent in the worst way: facts(21) and facts(22) returned nothing,
            # so stated_facts found nothing to anchor on, so `uniq` was empty,
            # so the run printed a GREEN OK for two lectures it had not checked
            # at all. Part V's two recommender lectures were the only ones whose
            # figures.json keys are prefixed rec21_/rec22_ rather than lNN_,
            # which is why they were the two that slipped.
            root = re.match(r"(l\d\d|app\d\d|rec\d\d)", top)
            root = root.group(1) if root else ""
            if root in allowed:
                out.append((path, float(o)))

    walk(doc)
    return out


def significant(raw: str) -> int:
    """Significant digits in a number AS WRITTEN on the slide.

    A short round number matches something somewhere by chance; 48,180 or
    0.170746 does not. Four is the threshold at which a match stops being a
    coincidence and starts being a quotation.
    """
    d = raw.replace(",", "").lstrip("-")
    if "." in d:
        whole, frac = d.split(".", 1)
        whole = whole.lstrip("0")
        return len(whole) + len(frac) if whole else len(frac.lstrip("0")) or 1
    return len(d.rstrip("0")) or 1


def stated_facts(deck: Path, own) -> list[tuple[int, float, str, str]]:
    """Which of this lecture's figures.json values the deck states, and where.

    Anchoring on figures.json is what keeps this quiet enough to read. Every
    value in that file is by construction the result of a computation, so a
    slide quoting one is making a claim about a measurement -- exactly the claim
    the notebook has to reproduce. Course metadata, exam weights, chapter
    numbers and dataset constants are not in figures.json and are never
    considered, with no list of exceptions to maintain.
    """
    src = deck.read_text(encoding="utf-8")
    # Code shown on a slide is not a claim about a measurement -- but replace
    # it with its own newlines rather than deleting it, or every line number
    # after the first <pre> is wrong by the height of that block. The first
    # version deleted, and reported figures against unrelated slides.
    src = re.sub(r"<pre.*?</pre>", lambda m: "\n" * m.group(0).count("\n"),
                 src, flags=re.S)
    hits: list[tuple[int, float, str, str]] = []
    for m in re.finditer(r">([^<]+)<", src):
        run = m.group(1)
        if not run.strip():
            continue
        line = src[: m.start()].count("\n") + 1
        for nm in NUM.finditer(run):
            raw = nm.group(0).rstrip(".")
            try:
                v = float(raw.replace(",", ""))
            except ValueError:
                continue
            if DURATION.match(run[nm.end(): nm.end() + 12].strip()):
                continue                          # AUTHORING 3.2a
            if significant(raw) < 4:
                continue                          # too round to be a quotation
            if 1900 <= v <= 2100 and float(v).is_integer():
                continue                          # a year
            key = fact_for(v, own)
            if key:
                hits.append((line, v, key, " ".join(run.split())[:88]))
    return hits


def fact_for(v: float, own) -> str | None:
    """The figures.json key this slide figure is quoting, if any."""
    for key, f in own:
        if DURATION.search(key.replace("_", " ")):
            continue                              # a duration key: 3.2a
        for scale in (1.0, 100.0):
            w = f * scale
            if w == 0:
                continue
            if abs(v - w) <= abs(w) * 1e-9:
                return key
            for dp in (0, 1, 2, 3, 4):
                if round(w, dp) == v and round(w, dp) != 0:
                    return key
    return None


def matches(v: float, printed: set[float]) -> bool:
    """Did the notebook print this figure?

    Allows the two presentation differences that are choices rather than
    disagreements: a fraction stated as a percentage (0.9039 printed, "90.39%"
    on the slide), and rounding (0.170746 printed, "0.171" stated).
    """
    for scale in (1.0, 100.0, 0.01):
        w = v * scale
        for p in printed:
            if p == 0:
                continue
            if abs(w - p) <= max(abs(p), abs(w)) * 5e-4:
                return True
            for dp in (0, 1, 2, 3, 4):
                if round(p, dp) == round(w, dp) and round(p, dp) != 0:
                    return True
    return False


def executed(nb_path: Path, force: bool) -> Path | None:
    """Execute the notebook if its cached run is missing or stale."""
    CACHE.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(nb_path.read_bytes()).hexdigest()[:16]
    out = CACHE / f"{nb_path.stem}-{digest}.ipynb"
    if out.exists() and not force:
        return out
    print(f"    executing {nb_path.name} (not cached) …", flush=True)
    r = subprocess.run(
        [sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook",
         "--execute", "--output", str(out), str(nb_path)],
        capture_output=True, text=True, cwd=ROOT / "notebooks")
    if r.returncode != 0 or not out.exists():
        tail = (r.stderr or "").strip().splitlines()[-3:]
        print(f"    {RED}execution failed{OFF}: " + " / ".join(tail))
        return None
    return out


def printed_numbers(nb: Path) -> set[float]:
    doc = json.loads(nb.read_text(encoding="utf-8"))
    text = []
    for c in doc["cells"]:
        if c["cell_type"] != "code":
            continue
        for o in c.get("outputs", []):
            if o.get("output_type") == "stream":
                text.append("".join(o.get("text", "")))
            elif o.get("output_type") == "execute_result":
                text.append("".join(o.get("data", {}).get("text/plain", "")))
    return numbers("\n".join(text))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("lectures", nargs="*", type=int)
    ap.add_argument("--execute", action="store_true",
                    help="re-execute even if a cached run exists")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()

    pairs = []
    for n in (a.lectures or range(1, 25)):
        deck, nb = ROOT / f"slides/lecture-{n:02d}.html", ROOT / f"notebooks/lecture-{n:02d}.ipynb"
        if deck.exists() and nb.exists():
            pairs.append((n, deck, nb))
    if not pairs:
        print("no built lecture pairs found")
        return 0

    total = 0
    for n, deck, nb in pairs:
        if n not in NAMESPACES:
            print(f"{YELLOW}skip{OFF}  lecture {n:02d} — no figures.json "
                  f"namespace mapped; add it to NAMESPACES")
            continue
        run = executed(nb, a.execute)
        if run is None:
            print(f"{RED}FAIL{OFF}  lecture {n:02d} — notebook does not execute")
            total += 1
            continue
        printed = printed_numbers(run)
        own = facts(n)
        if not own:
            # A namespace that matches no key checks nothing, and until this
            # guard existed it reported "ok" for doing so. A check that cannot
            # fail is worse than no check, because it occupies the line where
            # the real one would have been.
            print(f"{RED}FAIL{OFF}  lecture {n:02d} — namespace "
                  f"{NAMESPACES[n]} matches no key in figures.json, so nothing "
                  f"was checked. Fix NAMESPACES or facts(), do not skip.")
            total += 1
            continue
        stated = stated_facts(deck, own)
        seen, uniq, excused, excused_cross = set(), [], set(), set()
        for line, v, key, ctx in stated:
            if matches(v, printed) or v in seen:
                continue
            path = key.lstrip("/")
            root = path.split("/")[0].split("[")[0]
            if root in SCALE_ONLY:
                excused.add(root)
                continue
            cross = next((c for c in CROSS_LECTURE
                          if c.startswith(f"l{n:02d}:")
                          and path.startswith(c.split(":", 1)[1])), None)
            if cross:
                excused_cross.add(cross)
                continue
            seen.add(v)
            uniq.append((line, v, key, ctx))
        if uniq:
            print(f"{RED}FAIL{OFF}  lecture {n:02d} — {len(uniq)} figure(s) the "
                  f"notebook never prints")
            for line, v, key, ctx in uniq[:12]:
                print(f"        {deck.name}:{line}  {v:g}   {DIM}{key}{OFF}")
                print(f"        {DIM}{ctx}{OFF}")
            if len(uniq) > 12:
                print(f"        … and {len(uniq) - 12} more")
            total += len(uniq)
        else:
            print(f"{GREEN}ok{OFF}    lecture {n:02d} — every stated figure is "
                  f"printed by its notebook")
        for root in sorted(excused):
            print(f"        {DIM}scale-exempt: {root} — {SCALE_ONLY[root]}{OFF}")
        for c in sorted(excused_cross):
            print(f"        {DIM}quoted from another lecture: "
                  f"{c.split(':', 1)[1]} — {CROSS_LECTURE[c]}{OFF}")
        if a.verbose:
            print(f"        {DIM}{len(stated)} figures.json values stated on "
                  f"the deck; {len(printed)} numbers printed by the notebook{OFF}")

    print()
    if total:
        print(f"{BOLD}{RED}{total} figure(s) stated on a slide and not "
              f"reproduced by its notebook{OFF}")
        return 1
    print(f"{BOLD}{GREEN}slides and notebooks agree{OFF}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
