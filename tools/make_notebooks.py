#!/usr/bin/env python3
"""
Build the Colab notebooks the slides link to.

    python3 tools/make_notebooks.py            # write notebooks/lecture-NN.ipynb
    python3 tools/make_notebooks.py --check    # ...then execute each one and fail
                                               #    on the first error

Why generated rather than hand-edited JSON: the same reason the figures are.
Twenty-four notebooks have to stay consistent with each other and with the
decks, and a diff of .ipynb JSON is unreadable, so drift would be invisible.

The notebooks are **ours**. No cell is taken from the textbook's own notebooks.
Each one follows AUTHORING.md §4: it mirrors the lecture, precedes every code
cell with the specification that would produce it, asserts after every
structural step, and states an expected wall-clock next to anything slow — for
an audience of mathematicians rather than engineers, "no output for four
minutes" otherwise reads as "it hung" and gets interrupted.

Nothing in a notebook is wrong on purpose.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import nbformat as nbf
# _prompt lives in tools/notebooks/, beside the per-lecture modules. That
# directory goes on sys.path in _discover() so those modules can import their
# siblings — but this import runs at module load, long before _discover() is
# called, so it needs the path itself.
sys.path.insert(0, str(Path(__file__).resolve().parent / "notebooks"))
from _prompt import prompt                                # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "notebooks"

SEED = 42


# ------------------------------------------------------------------ helpers

def md(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(text.strip("\n"))


def code(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(text.strip("\n"))


def header(n: int, title: str, kind: str, chapters: str, thread: str = "") -> list:
    """The opening cell. `kind` and `thread` are vestiges of the Build/Fix
    design and are ignored; they stay in the signature only until the last
    lecture module stops passing them."""
    source = f"Géron, {chapters}" if chapters else "Lecture notes"
    return [md(f"""
# {title}

**Lecture {n}** · {source}

Applications of Machine Learning — BSc Mathematics of Artificial Intelligence

---

**How to use this notebook.** You are not expected to type the code. You are
expected to *read* it before you run it, and to be able to say what every line
does and what would break if it changed.

Every code cell is preceded by the **specification that would produce it** —
input, output, constraint, check. Read the box, work out what the check should
say, *then* run the cell. That order is the whole point of the box.

Run the cells in order. Anything that takes more than a few seconds says so,
and anything that needs a GPU says that too. Nothing here is wrong on purpose.
""")]


# The honesty note that every notebook's opening cell must carry. It is applied
# at BUILD time rather than written into each header, because the headers are
# produced three different ways — `header()` above, a `HEADER` constant in
# lectures 11-18 and 20-24, and an inline `md(...)` first cell in lectures 9 and
# 10. Patching three mechanisms means the fourth one somebody adds next month
# quietly ships without the note, and the note is the thing that keeps the
# prompt boxes honest.
PROMPT_NOTE = """
**About the prompt boxes.** Every code cell in this notebook is preceded by a
quoted prompt naming four things: the input, the output, the constraint the
method must respect, and a check whose answer you can work out before running
anything. Read the box, answer the check in your head, then run the cell.

The prompts are **specifications, not transcripts** — this is what you would
have to ask for in order to get this cell, not a recording of somebody asking
for it. If your own prompt is vaguer than the box, expect worse code than the
cell below it.
"""

NOTE_EXEMPT: set[int] = set()


def _ensure_prompt_note(nb: nbf.NotebookNode, n: int) -> None:
    """Append the honesty note to the opening markdown cell, once.

    Only where there is something to be honest ABOUT. The first version of this
    added the note unconditionally, so twenty-two notebooks with no prompt boxes
    at all shipped a header announcing prompt boxes — a claim in the one place a
    student reads before anything else, and false. That is the failure this
    course spends twenty-four lectures on, committed by its own build script, so
    the note is now gated on the boxes actually existing and worded to stay true
    at partial coverage.
    """
    if n in NOTE_EXEMPT or not nb.cells:
        return
    # The note now claims EVERY code cell has a box, so verify that before
    # writing it. The first version of this claimed boxes existed when none
    # did; the fix was to gate on their existing at all, and this is the same
    # fix again one step stronger, because the claim itself got stronger.
    for i, c in enumerate(nb.cells):
        if c.cell_type != "code":
            continue
        prev = nb.cells[i - 1] if i else None
        if not (prev is not None and prev.cell_type == "markdown"
                and prev.source.lstrip().startswith("> **Prompt")):
            raise RuntimeError(
                f"lecture {n}: the code cell at index {i} has no prompt box, "
                f"but the header note about to be added says every cell does. "
                f"Add the box or weaken the note — do not ship the claim.")
    first = nb.cells[0]
    if first.cell_type != "markdown":
        raise RuntimeError(f"lecture {n}: first cell is {first.cell_type}, "
                           f"expected the markdown header")
    if "specifications, not transcripts" in first.source:
        return
    first.source = first.source.rstrip() + "\n" + PROMPT_NOTE.rstrip() + "\n"


SETUP = code('''
# --- setup -------------------------------------------------------------------
# Not examinable: this is engineering hygiene, not machine learning. It is here
# because a version mismatch produces a confusing error twenty cells later.
import sys, sklearn, numpy as np, pandas as pd, matplotlib

print(f"python       {sys.version.split()[0]}")
print(f"scikit-learn {sklearn.__version__}")
print(f"numpy        {np.__version__}")
print(f"pandas       {pd.__version__}")

# root_mean_squared_error arrived in scikit-learn 1.4
assert tuple(int(p) for p in sklearn.__version__.split(".")[:2]) >= (1, 4), \\
    "This notebook needs scikit-learn >= 1.4.  In Colab: %pip install -U scikit-learn"

RANDOM_STATE = 42          # every split, every model, every shuffle
pd.set_option("display.width", 100)
''')


LOADER = code('''
# --- the data ----------------------------------------------------------------
# A function, not a manual download: the data will change, and you will need
# this on another machine.  ~5 s the first time, instant afterwards.
from pathlib import Path
import tarfile, urllib.request

def load_housing():
    tarball = Path("datasets/housing.tgz")
    if not tarball.is_file():
        Path("datasets").mkdir(parents=True, exist_ok=True)
        url = "https://github.com/ageron/data/raw/main/housing.tgz"
        urllib.request.urlretrieve(url, tarball)
        with tarfile.open(tarball) as t:
            t.extractall(path="datasets", filter="data")
    return pd.read_csv("datasets/housing/housing.csv")

housing_full = load_housing()

assert housing_full.shape == (20640, 10), f"unexpected shape {housing_full.shape}"
print(f"{len(housing_full):,} districts, {housing_full.shape[1]} columns")
housing_full.head()
''')


SETUP_PROMPT = prompt(
    label="setup",
    input="nothing",
    output="the version of every library this notebook depends on, and one "
           "seed",
    constraint="ASSERT the scikit-learn version rather than printing it — "
               "`root_mean_squared_error` arrived in 1.4, and on an older "
               "Colab image the failure is an ImportError twenty cells from "
               "here",
    check="RANDOM_STATE is defined here ONCE and used for every split, every "
          "model and every shuffle in the notebook. A notebook carrying three "
          "different seeds cannot be reproduced by reading it")

LOADER_PROMPT = prompt(
    label="the data",
    input="the California housing tarball",
    output="20,640 districts and 10 columns",
    constraint="a FUNCTION that downloads if absent and reads if present — the "
               "data will change, and you will need this on another machine",
    check="assert the shape, rather than trusting the download",
    **{"try": "delete the `datasets/` directory and re-run the cell. If it "
              "cannot rebuild its own input from nothing, it is not "
              "reproducible — it is cached."})


# ------------------------------------------------------------------ lecture 1

def lecture_01() -> nbf.NotebookNode:
    cells = header(
        1, "What machine learning is, and how we will work", "", "Chapters 1–2")

    cells += [
        md("""
This notebook is the whole of Lecture 1's second half, in runnable form: the
brief, the data, the split, and the exploration. Nothing is fitted here — the
first model arrives in Lecture 2, on purpose. Looking properly at data before
modelling it is not a preliminary, it is the part that decides whether the
model can work at all.

Runs on free CPU in about two minutes.
"""),

        md("## 1 · Setup"), SETUP_PROMPT, SETUP,
        md("## 2 · The data"), LOADER_PROMPT, LOADER,

        md("""
### What is in it

Ten attributes per district. One of them is not numeric, and one column has
holes in it. Find both before reading on.
"""),
        prompt(
            label="what is in it",
            input="the loaded frame",
            output="every column, its type and its non-null count",
            constraint="`.info()`, not `.head()` — the two things worth finding "
                       "here are a non-numeric column and a column with holes in "
                       "it, and neither is visible in five rows",
            check="the non-null count of nine columns equals the row count, and "
                  "of one column it does not",
            **{"try": "`housing_full.head()` instead. Which of the two "
                      "findings above is still visible in five rows, and which "
                      "is not?"}),
        code('''
housing_full.info()
'''),
        prompt(
            label="count the holes, and the categories",
            input="the frame",
            output="how many districts are missing total_bedrooms, and the "
                   "counts of every category level",
            constraint="print the missing count as a PERCENTAGE as well as a "
                       "count — 207 sounds like a lot and 1% does not",
            check="`value_counts()` on the categorical sums to 20,640, and one "
                  "of its levels has n < 10",
            **{"try": "`normalize=True` on the value_counts. ISLAND becomes "
                      "0.0002 — at what point does a rare level stop being a "
                      "curiosity and start being a problem?"}),
        code('''
n_missing = housing_full["total_bedrooms"].isna().sum()
print(f"total_bedrooms is missing in {n_missing} districts "
      f"({100 * n_missing / len(housing_full):.1f}%)")
print()
print(housing_full["ocean_proximity"].value_counts())
'''),
        md("""
`ISLAND` has five districts in the whole of California. Remember that: a level
with n=5 is a level that will be absent from some cross-validation folds, which
matters from the next lecture onwards.
"""),

        md("""
## 3 · A quick look at the whole set

One look at everything, to find what is *structurally* wrong with the data —
the kind of fact you need before you can decide anything at all. Then we split,
and from that point on every number is computed on the training half.

Two things should jump out of the histograms. Take thirty seconds before you
scroll past them.
"""),
        prompt(
            label="histograms",
            input="the full frame",
            output="a histogram of every numeric column",
            constraint="50 bins, not the default 10 — a cap at the top of a "
                       "distribution is one bar, and at 10 bins it is inside a "
                       "bar with everything else",
            check="nine panels, one per numeric column, and two of them have a "
                  "conspicuous spike at their right-hand edge",
            **{"try": "`bins=10`, the default. Both spikes vanish into a "
                      "neighbouring bar. That is why the constraint is there."}),
        code('''
import matplotlib.pyplot as plt

housing_full.hist(bins=50, figsize=(12, 8))
plt.tight_layout(); plt.show()
'''),
        prompt(
            label="the range of every column",
            input="the full frame",
            output="the minimum and maximum of every numeric column",
            constraint="print the EXTREMES, not `describe()`'s quartiles — the "
                       "two ends are where the capping and the scaling show, "
                       "and the middle of a distribution hides both",
            check="two columns end on a suspiciously exact value. Find them "
                  "before reading on: one is the income, one is the target",
            **{"try": "add `.median()` to the frame. The medians are "
                      "unremarkable, which is why the histogram and this table "
                      "are worth more than a summary of the middle."}),
        code('''
ranges = pd.DataFrame({
    "min": housing_full.min(numeric_only=True),
    "max": housing_full.max(numeric_only=True),
})
print(ranges.to_string(float_format=lambda v: f"{v:,.4f}"))
'''),
        md("""
`median_income` runs from **0.4999** to **15.0001**, and `median_house_value`
stops dead at **500,001**. Neither is a number nature produces: both are the
signature of a cap applied when the data was recorded. `total_rooms` runs from
2 to 39,320, which is not a cap — it is a district-size effect, and it is why
the totals need turning into ratios later in this notebook.

**The income is not in dollars** — it is scaled, and capped at 15.0001.

**The target is capped too**, and the target is our label — so those districts
carry a label that is not the answer, and no model can be right about them.

That is enough to know before splitting. We *count* it, and look at the stripes
under it, after the split — on the training half, like every other number in
this notebook.
"""),
        md("""
## 4 · Split before you explore

Everything you learn from the data *before* the split leaks into the choices you
make afterwards — through you, not through the code. There is no library that
prevents this, which is why it is a rule about the order of your own actions.

We stratify on income because the domain experts said income predicts price, and
because `median_income` is continuous, we band it first. Five bands, chosen so
that no band is tiny: stratification needs enough districts per stratum to be
worth doing.
"""),
        prompt(
            label="the stratified split",
            input="the whole frame",
            output="an 80/20 split stratified on the income band",
            constraint="stratify on the income BAND, not on the raw income — "
                       "`train_test_split` stratifies on a categorical, and "
                       "20,640 distinct incomes are 20,640 strata",
            check="the two halves sum to 20,640 and their indices are disjoint",
            **{"try": "`stratify=housing_full[\"median_income\"]`, the raw "
                      "income, instead of the band. Read the error — it says "
                      "exactly why the banding step exists."}),
        code('''
from sklearn.model_selection import train_test_split

# five bands, on the scaled income. The last is open-ended because the top of
# the distribution is thin and a fixed upper edge would leave a near-empty band.
income_cat = pd.cut(housing_full["median_income"],
                    bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                    labels=[1, 2, 3, 4, 5])

train_set, test_set = train_test_split(
    housing_full, test_size=0.2, random_state=RANDOM_STATE, stratify=income_cat)

# assert, do not hope
assert len(train_set) + len(test_set) == len(housing_full)
assert set(train_set.index).isdisjoint(test_set.index), "the split overlaps"
print(f"train {len(train_set):,}   test {len(test_set):,}")

# From here to the last cell of the NEXT lecture, `test_set` is not touched.
housing = train_set.copy()
'''),
        md("""
### Did stratifying actually buy anything?

The claim is that a random split gets the income mix wrong and a stratified one
does not. That is a measurable claim, so measure it: take the proportion of
districts in each income band in the full dataset, and compare it with the
proportion each kind of split produces.
"""),
        prompt(
            label="sampling bias, measured",
            input="the income bands, the full frame, and one test set of each kind",
            output="the band proportions under each, and the percentage error "
                   "of each against the full-data proportions",
            constraint="the same seed for both splits, so the only difference "
                       "between them is the stratification",
            check="the stratified error is smaller in every band; the interesting "
                  "question is by how much",
            **{"try": "change `random_state` to 0, then 1, then 2. The random "
                      "error moves every time; does the stratified one?"}),
        code('''
random_test = train_test_split(housing_full, test_size=0.2,
                               random_state=RANDOM_STATE)[1]

def band_share(frame):
    return income_cat.loc[frame.index].value_counts(normalize=True).sort_index()

overall = income_cat.value_counts(normalize=True).sort_index()
comparison = pd.DataFrame({
    "overall %":    100 * overall,
    "stratified %": 100 * band_share(test_set),
    "random %":     100 * band_share(random_test),
})
comparison["stratified error %"] = (
    100 * (comparison["stratified %"] / comparison["overall %"] - 1))
comparison["random error %"] = (
    100 * (comparison["random %"] / comparison["overall %"] - 1))

print(comparison.round(2).to_string())
print(f"\\nworst error — stratified {comparison['stratified error %'].abs().max():.2f}%"
      f"   random {comparison['random error %'].abs().max():.2f}%")
'''),

        md("""
## 5 · Now explore — the training set, and only that

Everything below is computed on `housing`, the training copy. Every correlation,
every scatter, every ratio. The 4,128 test districts play no part in any decision
made from here on.
"""),
        md("""
### First, the cap — counted, not squinted at
"""),
        prompt(
            label="count the cap",
            input="the training half's target column",
            output="how many training districts sit at the cap, and the "
                   "commonest values below it",
            constraint="count it — a histogram shows you a spike, and a count "
                       "tells you whether it is 5% of your labels or 0.5%",
            check="the commonest values below the cap are all multiples of the "
                  "same number; work out which before running it",
            **{"try": "raise the threshold from 500,000 to 500,001. The count "
                      "does not change — what does that tell you about how the "
                      "cap was applied?"}),
        code('''
capped = (housing["median_house_value"] >= 500_000).sum()
print(f"{capped} districts sit at the cap "
      f"({100 * capped / len(housing):.1f}% of the training set)")

# a continuous target should have an almost flat value_counts. Where it is not,
# the recording process is visible — a fact about the survey, not California.
counts = housing["median_house_value"].value_counts()
print(f"\\na typical price is shared by {counts.median():.0f} districts")
print("\\nthe five commonest values below the cap:")
print(counts.drop(counts.index.max()).head(5))
'''),
        md("""
Every one of those is a multiple of **$12,500** — artefacts of how the survey
recorded prices, not facts about California.

The cap is the one that matters. The target is the *label*, so a capped district
has a label that is not the answer and no model can be right about it. Two
responses are legitimate, and the choice belongs to the stakeholder rather than
to us: collect proper labels for those districts, or drop them from both halves
and state that the system does not predict above $500,000.

A well-known description of this dataset also names fainter lines at \\$450,000,
\\$350,000 and \\$280,000. When a source names specific numbers about your data,
those numbers are checkable.
"""),
        prompt(
            label="check the famous claim",
            input="three values named in a well-known description of this dataset",
            output="how many training districts sit at each",
            constraint="check the claim against the counts you just computed "
                       "rather than repeating it",
            check="compare each against the median count printed above — one of "
                  "the three is real, one is marginal, and one is "
                  "indistinguishable from the background",
            **{"try": "three values nobody claimed — 460,000, 340,000 and "
                      "270,000. If those come back comparable, the original "
                      "claim was about the background, not about the data."}),
        code('''
for value in (450_000, 350_000, 280_000):
    print(f"${value:>9,}  {counts.get(value, 0):>4d} districts")
'''),
        md("""
### Then the geography
"""),
        prompt(
            label="geography",
            input="the training districts' longitude and latitude",
            output="a scatter of California, with population as the marker size "
                   "and price as the colour",
            constraint="alpha well below 1 — at alpha=1 the dense areas saturate "
                       "into a solid blob and the density information, which is "
                       "the point of the plot, is destroyed",
            check="the coastline is legible, and the expensive districts are "
                  "visibly not uniformly distributed",
            **{"try": "`alpha=1`, then separately `cmap=\"jet\"`. Two "
                      "different lessons, and the second is easier to see than "
                      "to explain."}),
        code('''
housing.plot(kind="scatter", x="longitude", y="latitude",
             alpha=0.2,                       # density, not just position
             s=housing["population"] / 100, label="population",
             c="median_house_value", cmap="viridis",   # not jet: see below
             colorbar=True, figsize=(9, 6), sharex=False)
plt.title("training districts: size = population, colour = median price")
plt.tight_layout(); plt.show()
'''),
        md("""
The colormap is `viridis` rather than the `jet` you will see in older code.
`jet` is not perceptually uniform: it has a bright band in the middle and dark
ends, so equal steps in the data are not equal steps in apparent brightness, and
it invents boundaries in smooth data that a reader then interprets as structure.
It also collapses to an unreadable grey ramp when printed or seen by a
colour-blind reader. `viridis` is monotone in lightness and survives both.

Price is high near the ocean and near the two big cities. That is a fact you can
use — and one we will make explicit as a feature in the next lecture.
"""),
        prompt(
            label="correlations",
            input="the numeric training columns",
            output="the linear correlation of every attribute with the target, "
                   "ranked",
            constraint="Pearson only, and say so — it measures LINEAR "
                       "association and nothing else",
            check="median_income is far the strongest; every other column is "
                  "below 0.15 in absolute value",
            **{"try": "`method=\"spearman\"`, which ranks rather than "
                      "measures. Which column moves most, and what does its "
                      "histogram above look like?"}),
        code('''
corr = housing.select_dtypes(include=[np.number]).corr(numeric_only=True)
print("linear (Pearson) correlation with the target:\\n")
print(corr["median_house_value"].sort_values(ascending=False).round(3).to_string())
'''),
        md("""
`median_income` at about 0.69 is far and away the strongest single predictor,
which is why we stratified on it. But read the weak entries carefully rather
than dismissing them: `total_rooms` correlates with the target at about 0.14,
and that is not because the number of rooms is irrelevant to price. It is
because `total_rooms` is a *district* total, so it mostly measures how many
people live in the district.

The quantity that should matter is rooms **per household**. Correlation cannot
tell you that; only knowing what the column means can.
"""),
        prompt(
            label="attribute combinations",
            input="the training frame",
            output="three per-household and per-room ratios, and their "
                   "correlation with the target",
            constraint="ratios, not totals — a district total is a proxy for "
                       "district size, and district size is not what we are "
                       "predicting",
            check="at least one ratio correlates more strongly than either "
                  "column it was built from",
            **{"try": "add `bedrooms_per_person`. It is a ratio too — is it "
                      "any use? Not every combination is worth having, and the "
                      "correlation is how you find out."}),
        code('''
housing["rooms_per_house"]   = housing["total_rooms"] / housing["households"]
housing["bedrooms_ratio"]    = housing["total_bedrooms"] / housing["total_rooms"]
housing["people_per_house"]  = housing["population"] / housing["households"]

new_corr = housing.select_dtypes(include=[np.number]).corr(
    numeric_only=True)["median_house_value"]

for name in ("rooms_per_house", "bedrooms_ratio", "people_per_house",
             "total_rooms", "population", "households"):
    print(f"{name:20s} {new_corr[name]:+.3f}")
'''),
        md("""
`bedrooms_ratio` is more strongly correlated with price than any of the three
raw columns it was derived from — a district where a small share of the rooms
are bedrooms is a district of larger, more expensive houses. Nothing in the data
told us to compute that. Knowing what the columns *mean* did.

One warning to carry into the next lecture: when you build combined features,
avoid simple weighted sums of columns you already have. A feature that is a
linear combination of existing ones adds no information and makes the linear
algebra worse — the next lecture derives exactly why, when it derives the normal
equation.
"""),

        md("""
## 6 · Where we are

- The data has one categorical column, 207 missing values in `total_bedrooms`,
  a target capped at $500,000, and per-district totals that mostly measure
  district size.
- The split is done, stratified on income, and the test set is sealed.
- The strongest single predictor is `median_income`; the most useful engineered
  feature is `bedrooms_ratio`.

Nothing has been fitted. **Next lecture:** the normal equation, then a
preprocessing pipeline that handles all of the above, cross-validation, and the
first honest number.

**Before then:** run this notebook top to bottom once. Then change the income
bands in `pd.cut` — try three bands, or eight — and re-run from that cell to the
sampling-bias table. What happens to the stratified error, and why?
"""),
    ]
    return build(cells)


# ------------------------------------------------------------------ lecture 2

def lecture_02() -> nbf.NotebookNode:
    cells = header(
        2, "The end-to-end project", "", "Chapter 2")

    cells += [
        md("""
The whole pipeline, end to end: the preprocessing that has to be *learned* from
data, the cross-validation that gives an honest estimate of error, the search
that tunes it, and the single use of the test set at the end.

The first two cells repeat the Lecture 1 load and split, so this notebook stands
on its own. Runs on free CPU; the search cell takes two to four minutes.
"""),
        md("## 1 · Setup, and the same split as last time"),
        SETUP_PROMPT, SETUP, LOADER_PROMPT, LOADER,
prompt(
       label="every import, and the same split",
       input="the same data and the same seed",
       output="the identical 16,512 / 4,128 split as the previous lecture",
       constraint="every import this notebook needs in ONE place, and the split rebuilt from the seed rather than inherited",
       check="assert the two sizes exactly — if they differ, every comparison against the previous lecture is void",
       **{"try": "restart the runtime and run this cell first, before anything else. A notebook that only runs because another one is still in memory is not reproducible, and restart-and-run-all is the only test of that."}),
        code('''
# Every import this notebook needs, in one place — a notebook that only runs
# because a previous one is still in memory is not reproducible.
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import (GridSearchCV, KFold, cross_val_predict,
                                     cross_val_score, train_test_split)
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, add_dummy_feature
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt

income_cat = pd.cut(housing_full["median_income"],
                    bins=[0., 1.5, 3.0, 4.5, 6., np.inf], labels=[1, 2, 3, 4, 5])
train_set, test_set = train_test_split(
    housing_full, test_size=0.2, random_state=RANDOM_STATE, stratify=income_cat)

housing  = train_set.copy()
X_train  = housing.drop(columns=["median_house_value"])
y_train  = housing["median_house_value"]
X_test   = test_set.drop(columns=["median_house_value"])
y_test   = test_set["median_house_value"]

assert len(X_train) == 16512 and len(X_test) == 4128
print("same split as the previous lecture — the seed guarantees it")
'''),

        md("""
## 2 · What `LinearRegression().fit()` actually computed

Minimising $\\lVert X\\theta - y\\rVert^2$ gives the normal equation

$$X^{\\mathsf T}(X\\hat\\theta - y) = 0$$

Read row by row, that says the residual is orthogonal to **every column of X**.
Least squares is not an algebraic trick — it is a projection onto the column
space of $X$. Verify it rather than believing it:
"""),
prompt(
       label="what LinearRegression().fit() actually computed",
       input="the numeric features with an intercept column added",
       output="the normal-equation solution, and the residual's inner product with every column of X",
       constraint="add the intercept column with `add_dummy_feature` BEFORE solving — without it the residual is not orthogonal to the constant and the assert fails for the wrong reason",
       check="assert the largest |Xᵀ(Xθ̂ − y)| is negligible RELATIVE to the scale of y — an absolute tolerance on dollars is meaningless. Read the assert row by row: it says the residual is orthogonal to every column of X",
       **{"try": "append a column that is the sum of two existing ones, and solve again. `np.linalg.inv` raises or returns nonsense; `np.linalg.pinv` does not. That is the invertibility condition, met in practice."}),
        code('''
num = X_train.select_dtypes(include=[np.number])
X = make_pipeline(SimpleImputer(strategy="median"), StandardScaler()).fit_transform(num)
X_b = add_dummy_feature(X)                     # the x0 = 1 column, for the intercept
y = y_train.values

theta = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y
residual = X_b @ theta - y

# every column of X is orthogonal to the residual, to numerical precision
orth = X_b.T @ residual
print(f"largest |Xᵀ(Xθ̂ − y)| = {np.abs(orth).max():.3e}")
print(f"relative to the scale of y ({np.abs(y).mean():,.0f}): "
      f"{np.abs(orth).max() / np.abs(y).mean():.2e}")
assert np.abs(orth).max() / np.abs(y).mean() < 1e-6, "not orthogonal — check X"
'''),
        md("""
`np.linalg.inv` is not what scikit-learn uses. It computes the pseudoinverse via
SVD, which still returns an answer when $X^{\\mathsf T}X$ is singular — when you
have more features than instances, or when two columns are collinear. That is
the whole failure condition, and it is why you were warned against engineering a
feature as a weighted sum of existing ones.
"""),

        md("""
## 3 · A number that means nothing

Fit an unconstrained decision tree, then score it on the very rows it was
fitted to. The result is not evidence that the tree is good — it is what any
model flexible enough to memorise will produce, and it is the reason the next
section exists.
"""),
        prompt(
            label="a training score on a model that can memorise",
            input="a ColumnTransformer, and an unconstrained decision tree",
            output="the training RMSE of a constant, a linear model, an "
                   "unconstrained tree and a forest",
            constraint="score every one on the TRAINING rows, deliberately — "
                       "this is a demonstration, not a result, and the tree's "
                       "number only means something beside the other three",
            check="work out the tree's before running it: an unconstrained tree "
                  "splits until every leaf is pure, so what can it score on the "
                  "rows it split on? And the constant's, which cannot overfit, "
                  "should be about the spread of the prices",
            **{"try": "set `max_depth=4` and run it again. The training RMSE "
                      "stops being zero — what has changed about the model, and "
                      "has anything changed about the measurement?"}),
        code('''
num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
preprocessing = ColumnTransformer([
    ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), ["ocean_proximity"]),
])

# All four, scored on the rows they were fitted to. Only the tree's number is
# surprising, and it is only surprising beside the other three.
candidates = {
    "Predict a constant": DummyRegressor(strategy="mean"),
    "Linear regression":  LinearRegression(),
    "Decision tree":      DecisionTreeRegressor(random_state=RANDOM_STATE),
    "Random forest":      RandomForestRegressor(n_estimators=100,
                                                random_state=RANDOM_STATE,
                                                n_jobs=-1),
}

train_rmse = {}
for name, model in candidates.items():
    pipe = Pipeline([("prep", preprocessing), ("model", model)])
    pipe.fit(X_train, y_train)
    train_rmse[name] = root_mean_squared_error(y_train, pipe.predict(X_train))
    print(f"{name:20s} ${train_rmse[name]:>10,.0f}")

print("\\nAn unconstrained tree can put every training row in its own leaf.")
print("The constant cannot overfit anything, so its number is already honest.")
'''),

        md("""
## 4 · Measure it honestly

`shuffle=True` is not decoration. The default `KFold` does **not** shuffle, so
two students whose dataframes are in different row orders get different folds
and cannot work out why their numbers disagree.

⏱ **about 90 seconds** — thirty fits in total, ten of them forests.
"""),
prompt(
       label="⏱ 90 s — measure it honestly",
       input="the three models, ten folds each",
       output="mean, standard deviation and range of the fold RMSEs",
       constraint="`shuffle=True` is NOT decoration — the default KFold does not shuffle, so two people whose dataframes are in different row orders get different folds and cannot work out why their numbers disagree",
       check="print the fold minimum and maximum beside the mean. The spread decides which differences you are allowed to talk about: a mean of $50,000 from folds spanning $8,000 supports very different claims from one built from folds spanning $500",
       **{"try": "drop `shuffle=True` and re-run. The mean moves. Which of the two numbers is right, and what does the question even mean?"}),
        code('''
cv = KFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)

models = {
    "Linear regression": LinearRegression(),
    "Decision tree":     DecisionTreeRegressor(random_state=RANDOM_STATE),
    "Random forest":     RandomForestRegressor(n_estimators=100,
                                               random_state=RANDOM_STATE, n_jobs=-1),
}

results = {}
for name, model in models.items():
    pipe = Pipeline([("prep", preprocessing), ("model", model)])
    folds = -cross_val_score(pipe, X_train, y_train, cv=cv,
                             scoring="neg_root_mean_squared_error")
    results[name] = folds
    print(f"{name:20s} ${folds.mean():>9,.0f}  ± ${folds.std():>6,.0f}"
          f"   (folds ${folds.min():,.0f} – ${folds.max():,.0f})")
'''),
        md("""
Report the spread, not just the mean. The folds span several thousand dollars,
so **any comparison that turns on less than a couple of thousand is not a
comparison.**

Compare the models on the *same* folds rather than comparing two averages —
paired differences remove the fold-to-fold variation that both models share:
"""),
prompt(
       label="compare on the SAME folds",
       input="the two arrays of per-fold scores",
       output="the paired difference, and how many folds each model wins",
       constraint="subtract PER FOLD rather than comparing two averages — the folds differ in difficulty and both models feel that identically, so pairing cancels it",
       check="the paired standard deviation should be much smaller than either model's own; and the win count, '10 of 10 folds', is an argument that a mean difference alone is not",
       **{"try": "compare the two models the unpaired way instead — mean ± std against mean ± std. The intervals overlap and the comparison looks inconclusive. The paired version is not."}),
        code('''
# ddof=1 throughout: ten folds are a sample, not the population.
for name in ("Decision tree", "Random forest"):
    d = results[name] - results["Linear regression"]
    wins = (d < 0).sum()
    verdict = "same sign in all 10" if wins in (0, 10) else "sign changes"
    print(f"{name:15s} − linear:  mean ${d.mean():+,.0f}   "
          f"sd ${d.std(ddof=1):,.0f}   wins {wins}/10   {verdict}")

print("\\nOnly a difference whose sign survives every fold is a difference.")
'''),

        prompt(
            label="the preprocessing is a hyperparameter too",
            input="the same pipeline, with three imputation strategies",
            output="the cross-validated RMSE of each",
            constraint="vary ONLY the imputer, on the same folds, so the "
                       "difference is attributable to it and nothing else",
            check="the three land within a few hundred dollars of each other. "
                  "Before running it, decide what you would conclude if they "
                  "did — and what the fold spread from section 4 says about "
                  "whether such a gap is a gap at all",
            **{"try": "add `strategy='constant'` with `fill_value=0`. It is "
                      "worse, and by enough to see — which tells you the other "
                      "three were not close by accident."}),
        code('''
for strategy in ("median", "mean", "most_frequent"):
    prep = ColumnTransformer([
        ("num", make_pipeline(SimpleImputer(strategy=strategy), StandardScaler()),
         num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), ["ocean_proximity"]),
    ])
    pipe = Pipeline([("prep", prep),
                     ("model", RandomForestRegressor(n_estimators=200,
                                                     max_features=8,
                                                     random_state=RANDOM_STATE,
                                                     n_jobs=-1))])
    rmse = -cross_val_score(pipe, X_train, y_train, cv=cv,
                            scoring="neg_root_mean_squared_error").mean()
    print(f"{strategy:15s} ${rmse:,.0f}")
'''),
        md("""
Three strategies, a few hundred dollars apart, on folds that themselves span
thousands. The honest reading is that the choice does not matter here — which
is worth knowing, because it is the kind of knob people spend a week on.

## 5 · Tune — on validation folds, never on the test set

⏱ **2–4 minutes.** Fifteen combinations × five folds = 75 forest fits. The
lecture's figure uses `cv=10`, which takes twice as long; five is enough here.
"""),
prompt(
       label="⏱ 3-6 min — tune on validation folds, never on the test set",
       input="fifteen combinations, ten folds each — 150 fits",
       output="the best parameters and the best cross-validated RMSE",
       constraint="the grid searches the WHOLE PIPELINE, so the preprocessing is refitted inside every fold — `model__` prefixes because the parameters belong to a step — and it reuses the SAME KFold object as the section above, so its number is comparable with the ones already printed",
       check="detect whether the winner sits on the EDGE of the grid and say so — an optimum at the boundary means the optimum may lie outside it, and the search was too small",
       **{"try": "read `best_score_` and ask what it is the score OF. It is the winner of a fifteen-way selection, measured on the folds that did the selecting, so it is optimistic by construction. The honest number is still three sections away."}),
        code('''
grid = {"model__max_features": [4, 6, 8, 10, 12],
        "model__n_estimators": [30, 100, 200]}

search = GridSearchCV(
    Pipeline([("prep", preprocessing),
              ("model", RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1))]),
    grid, cv=cv,                     # the SAME ten folds as section 4, so the
                                     # numbers here are comparable with those
    scoring="neg_root_mean_squared_error", n_jobs=-1)
search.fit(X_train, y_train)

print(f"best {search.best_params_}")
print(f"best cross-validated RMSE ${-search.best_score_:,.0f}")

best_n = search.best_params_["model__n_estimators"]
if best_n == max(grid["model__n_estimators"]):
    print("\\n⚠ the winner sits on the EDGE of the grid — the optimum may lie "
          "outside it. Search again with larger values.")
'''),

        md("""
## 6 · What the encoder does with a category it has never seen
"""),
        md("""
`ISLAND` — five districts in the whole state — is the category the first lecture
warned you about. With ten folds, some folds contain no ISLAND training row at
all, and `handle_unknown="ignore"` then encodes it as an all-zero column and
says nothing.
"""),
prompt(
       label="what an unseen category actually encodes to",
       input="an encoder fitted without ISLAND, asked to transform ISLAND",
       output="the resulting row, and its sum",
       constraint="DEMONSTRATE it rather than describing it — fit without the category and transform with it",
       check="every other district's encoded row sums to one. Work out what this one sums to, and what the model therefore sees",
       **{"try": "set `handle_unknown='error'` instead and re-run. It raises — which is the same information, delivered loudly. Decide which you would rather have inside a cross-validation loop, and why the quiet option is still the right default here."}),
        code('''
enc = OneHotEncoder(handle_unknown="ignore").fit(
    housing[["ocean_proximity"]].query("ocean_proximity != 'ISLAND'"))
# a DataFrame, not a bare list: transform() warns about missing feature names
# otherwise, and a warning here would obscure the point of the cell
island = pd.DataFrame({"ocean_proximity": ["ISLAND"]})
row = enc.transform(island).toarray()[0]
print("an unseen category encodes to:", row)
print("sum:", row.sum(), "— and no warning, no error")
'''),

        md("""
## 7 · The test set. Once.

Everything so far used only training data. This is the first and last time the
test set is touched.
"""),
prompt(
       label="the test set, once",
       input="the 4,128 held-out districts",
       output="the test RMSE with a 95% bootstrap interval, beside the cross-validated estimate",
       constraint="bootstrap the SQUARED ERRORS and take the square root of the interval, rather than bootstrapping the RMSE directly — the mean is what the bootstrap is good at, and the square root comes afterwards. `method='percentile'` is not optional either: scipy defaults to BCa, and we are claiming the percentile bootstrap",
       check="compare the test RMSE with the cross-validated estimate. Is the gap between them bigger or smaller than the interval — and what follows if it is smaller?",
       **{"try": "nothing. This is the one cell in the course with no `try` line: the test set has now been used, and any change you make in response to its number is you fitting it."}),
        code('''
from scipy import stats

best = search.best_estimator_          # refitted on the whole training set
final_pred = best.predict(X_test)
final_rmse = root_mean_squared_error(y_test, final_pred)

squared = (final_pred - y_test.values) ** 2
# method= is not optional: scipy defaults to BCa, and we are claiming the
# PERCENTILE bootstrap. Name the estimator you mean.
lo, hi = np.sqrt(stats.bootstrap([squared], np.mean,
                                 confidence_level=0.95, method="percentile",
                                 random_state=RANDOM_STATE).confidence_interval)

print(f"test RMSE  ${final_rmse:,.0f}")
print(f"95% interval  ${lo:,.0f} – ${hi:,.0f}   (±${(hi - lo) / 2:,.0f})")
print(f"the cap the test set stops at: ${y_test.max():,.0f}")
print(f"\\ncross-validated estimate was ${-search.best_score_:,.0f}")
print("The two agree within the interval. A gap smaller than the interval is "
      "not evidence of anything.")
'''),
        md("""
**Do not tune now.** If you adjust hyperparameters to improve that number you
are fitting the test set, and the improvement will not generalise. The number
you have is the number you report.
"""),

        prompt(
            label="the ten worst predictions",
            input="the final model's test predictions",
            output="the ten districts with the largest absolute error, with "
                   "actual, predicted and error",
            constraint="sort by ABSOLUTE error, not by error — the ten worst "
                       "misses matter whichever direction they run in",
            check="before running it, predict what the actual column will hold. "
                  "The cap was 4.8% of the data and the model cannot exceed it",
            **{"try": "sort by signed error instead and take the ten smallest. "
                      "Those are the over-predictions, and they are a different "
                      "story from the ten under-predictions."}),
        code('''
worst = pd.DataFrame({
    "actual":     y_test.values,
    "predicted":  final_pred,
    "income":     test_set["median_income"].values,
    "ocean":      test_set["ocean_proximity"].values,
}).assign(error=lambda d: d.predicted - d.actual)
worst["abs_error"] = worst["error"].abs()

print(worst.nlargest(10, "abs_error").to_string(
    index=False, float_format=lambda v: f"{v:,.4f}"))
'''),
        md("""
Every one of the ten is a district at or near the **$500,001** cap, and every
one is *under*-predicted. The model cannot exceed the cap because nothing in
its training data ever did — so these are not mistakes it could learn its way
out of. They are the labelling decision from Lecture 1, arriving as error.

## 8 · One number for 4,128 districts is a summary, not a finding

Analysing the final model's errors is not tuning, and it is the part of the
report the client actually acts on. Break the error out by the categories they
care about.
"""),
        prompt(
            label="slice the error",
            input="the final model's test predictions",
            output="RMSE and district count, broken down by ocean proximity and "
                   "by income band",
            constraint="report the COUNT beside every RMSE — a group of three "
                       "districts and a group of 1,862 do not deserve the same "
                       "weight in your conclusion — and pass `observed=True`, "
                       "or pandas emits a row for every unobserved combination "
                       "of categories",
            check="one group is far worse than the rest, and its count is tiny. "
                  "Before running it, predict which: the first lecture named a "
                  "category with five districts in the whole state",
            **{"try": "drop `observed=True`. Count the rows you get, and how "
                      "many of them are NaN."}),
        code('''
err = pd.DataFrame({
    "error": final_pred - y_test.values,
    "ocean": test_set["ocean_proximity"].values,
    "income_cat": pd.cut(test_set["median_income"],
                         bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                         labels=[1, 2, 3, 4, 5]).values,
})

def slice_by(col):
    g = err.groupby(col, observed=True)["error"]
    return pd.DataFrame({"n": g.size(),
                         "RMSE": g.apply(lambda e: np.sqrt((e ** 2).mean()))})

for col in ("income_cat", "ocean"):
    out = slice_by(col).sort_values("RMSE")
    out["RMSE"] = out["RMSE"].map(lambda v: f"${v:,.0f}")
    print(f"by {col}:"); print(out.to_string()); print()
'''),
        md("""
Read the counts, not only the errors. The poorest band is predicted worst in
dollars — and its districts are the cheapest, so in *relative* terms it is worse
still. `ISLAND` has three districts in the test set, so its RMSE is an average
over three numbers and carries almost no information: it is a reminder that the
model was asked about a category it saw twice, not a measurement you could act
on.

**What you would tell the client:** the system is usable, and it should not be
deployed unqualified for the poorest districts, for the capped ones, or for
`ISLAND` at all.
"""),

        md("""
## 9 · Two comparisons that look like improvements and are not
"""),
        prompt(
            label="dropping the capped districts",
            input="the test predictions, with and without the capped rows",
            output="n, model RMSE and constant-baseline RMSE for each",
            constraint="recompute the BASELINE on each subset too — a model "
                       "score that moves while its baseline moves with it has "
                       "not improved, and only the pair shows that",
            check="the model's RMSE falls. Work out before running it whether "
                  "the baseline falls by more or less, in percentage terms",
            **{"try": "drop the cheapest 5% instead of the capped districts. "
                      "The RMSE falls again, for a third unrelated reason."}),
        code('''
def scored(mask, label):
    y, p = y_test.values[mask], final_pred[mask]
    base = np.full(mask.sum(), y_train.mean())
    print(f"{label:26s} n={mask.sum():>5,}   "
          f"model ${root_mean_squared_error(y, p):>8,.0f}   "
          f"baseline ${root_mean_squared_error(y, base):>8,.0f}")

capped = y_test.values >= 500_000
scored(np.ones(len(y_test), bool), "all test districts")
scored(~capped,                    "capped districts removed")
'''),
        md("""
The model looks **$4,840 better** with the capped districts gone. It is not: the
baseline fell by about 15% at the same time, because the question changed. We
did not improve the model, we asked it an easier question — *how well does it do
on districts below the cap?* Two numbers measured on different rows are not
comparable, whatever the column headings say.
"""),
        prompt(
            label="dollars or ratios — measured, not argued",
            input="the same pipeline, trained on the price and on its log",
            output="each one's cross-validated RMSE in dollars, its median "
                   "error as a percentage of price, and the share within 30%",
            constraint="compare on the TRAINING folds — choosing between two "
                       "differently-trained models is a choice, and choices are "
                       "not made on the test set",
            check="one wins in dollars and the other in percentage terms. Decide "
                  "which the stakeholder's brief actually asked for before you "
                  "look at the numbers",
            **{"try": "report the mean percentage error instead of the median. "
                      "It is much worse for both, because the capped districts "
                      "have percentage errors that no average survives."}),
        code('''
# Five folds, and the MEAN OF THE PER-FOLD RMSEs -- not the RMSE of the pooled
# out-of-fold predictions, which is a different (smaller) number. The slides
# quote the first, so this must too.
kf5 = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

def target_arm(on_log):
    rmses, apes = [], []
    for tr, va in kf5.split(X_train):
        Xa, Xb = X_train.iloc[tr], X_train.iloc[va]
        ya, yb = y_train.iloc[tr], y_train.iloc[va]
        # A fresh preprocessor per fit: a Pipeline fits the object it holds, so
        # reusing one would carry the previous fold's statistics across.
        prep = ColumnTransformer([
            ("num", make_pipeline(SimpleImputer(strategy="median"),
                                  StandardScaler()), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), ["ocean_proximity"]),
        ])
        m = Pipeline([("prep", prep),
                      ("model", RandomForestRegressor(
                          n_estimators=200, max_features=8,
                          random_state=RANDOM_STATE, n_jobs=-1))])
        m.fit(Xa, np.log(ya) if on_log else ya)
        p = m.predict(Xb)
        if on_log:
            p = np.exp(p)
        rmses.append(root_mean_squared_error(yb, p))
        apes.append(np.abs(p - yb.values) / yb.values)
    ape = np.concatenate(apes)
    return np.mean(rmses), 100 * np.median(ape), 100 * np.mean(ape <= 0.30)

for on_log, label in ((False, "the price (what we did)"), (True, "log of the price")):
    rmse, med, within = target_arm(on_log)
    print(f"{label:26s} ${rmse:>8,.0f}   median {med:5.1f}%   within 30% {within:5.1f}%")
'''),
        md("""
The brief asked for a **relative** criterion — within 30% of the price — and we
optimised an **absolute** one. Regressing the log wins on the criterion the
client stated and loses on the one we chose. Note that we compared these on the
training folds: picking between two differently-trained models is a choice, and
the test set is not where choices are made.
"""),
        prompt(
            label="where the error lives",
            input="the test predictions and the actual prices",
            output="RMSE for the cheapest tenth of districts and the dearest",
            constraint="split by the ACTUAL price, not the predicted one — "
                       "bucketing by the prediction hides exactly the districts "
                       "the model got most wrong",
            check="the dearest tenth has a far larger RMSE. Now divide each by "
                  "its bucket's median price and see which way the ordering goes",
            **{"try": "use the predicted price to form the deciles instead. The "
                      "dearest bucket's RMSE falls sharply — the capped "
                      "districts are no longer in it, because the model never "
                      "predicts that high."}),
        code('''
# Threshold on the quantile rather than bucketing with qcut: the cap puts many
# districts on exactly the same price, and qcut has to break that tie somewhere,
# which moves the boundary and the RMSE with it.
for q, op, name in ((0.1, "le", "cheapest tenth"), (0.9, "ge", "dearest tenth")):
    thr = y_test.quantile(q)
    m = (y_test <= thr) if op == "le" else (y_test >= thr)
    print(f"{name:16s} n={m.sum():>4,}   "
          f"RMSE ${root_mean_squared_error(y_test[m], final_pred[m.values]):>8,.0f}"
          f"   median price ${y_test[m].median():>8,.0f}")

# And the pair the deck shows in order to say it decides NOTHING: the same two
# targets compared on the test set. The comparison above, on the training
# folds, is the one that chose. This is a report, not a decision -- which is
# exactly what the specimen exam answer marks wrong when it is used as one.
log_model = Pipeline([("prep", preprocessing),
                      ("model", RandomForestRegressor(
                          n_estimators=200, max_features=8,
                          random_state=RANDOM_STATE, n_jobs=-1))])
log_model.fit(X_train, np.log(y_train))
log_test = root_mean_squared_error(y_test, np.exp(log_model.predict(X_test)))
print(f"\\non the test set:  price ${final_rmse:,.0f}   log ${log_test:,.0f}")
print("Two numbers on the sealed test set, and they chose nothing.")
'''),
        md("""
In dollars the model is worst where the money is. Divide each by its bucket's
median price and the ordering reverses: in *percentage* terms it is worst where
the money is not. Which of those two sentences you put in the report is the
whole of the absolute-versus-relative question, and the brief already answered
it.

## 9 · Where we are

- Fitting a linear model is orthogonal projection, and you verified the
  orthogonality numerically rather than taking it on trust.
- A score computed on the training rows cannot see overfitting. The tree proves
  it in one cell.
- Cross-validation gives an honest estimate **and** its spread, and two models
  differ only if the paired per-fold difference says so.
- All preprocessing lives inside the `Pipeline` handed to cross-validation, so
  leakage is structurally impossible rather than merely avoided.
- The test set was touched once, and the number came with an interval.

**Six questions to ask of any reported number** — yours or anyone else's:

1. Was every transformer fitted *after* the split, on training data only?
2. Is the preprocessing inside the pipeline passed to cross-validation?
3. Was the reported metric computed on held-out data?
4. Was the test set touched more than once?
5. Were hyperparameters selected using validation data, not test data?
6. Does any feature encode information unavailable at prediction time?

**Before the next lecture:** run this notebook top to bottom. Then change the
`KFold` from 10 splits to 3 and re-run from that cell. What happens to the mean
RMSE, and what happens to the spread — and which of the two changes should worry
you more?
"""),
    ]
    return build(cells)


# ------------------------------------------------------------------ driver

def build(cells: list) -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook(cells=cells)
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python",
                       "name": "python3"},
        "language_info": {"name": "python"},
        "colab": {"provenance": [], "toc_visible": True},
    }
    return nb


LECTURES = {1: lecture_01, 2: lecture_02}

# Lectures 3+ live in tools/notebooks/lecture_NN.py, each exporting build() ->
# list[cell]. One file per lecture so that authors working on different
# applications never touch the same file.
def _discover() -> None:
    import importlib.util
    # The lecture modules are loaded by FILE PATH, which leaves them with no
    # package — so `from ._prompt import prompt` cannot resolve. Putting their
    # own directory on sys.path lets them import siblings plainly instead.
    mod_dir = Path(__file__).parent / "notebooks"
    if str(mod_dir) not in sys.path:
        sys.path.insert(0, str(mod_dir))
    for path in sorted(mod_dir.glob("lecture_*.py")):
        n = int(path.stem.split("_")[1])
        if n in LECTURES:
            continue
        spec = importlib.util.spec_from_file_location(path.stem, path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        LECTURES[n] = (lambda m: lambda: build(m.build()))(mod)


_discover()


# Notebooks that are NOT generated from a module, and must not be overwritten.
#
# Empty since the 2026-09 redesign. Lecture 19 used to be here: it had been
# built cell by cell in Colab against Gemini, verbatim prompts and a planted
# `shuffle=True`, as the course's one worked assistant failure. The redesign
# dropped that device — nothing in a notebook is wrong on purpose any more —
# so lecture 19 is generated like every other. The guard stays because the
# failure it prevents (silently overwriting hand-authored work) is real.
COLAB_AUTHORED: set[int] = set()


def _keep_cell_ids(nb: nbf.NotebookNode, path: Path) -> None:
    """Reuse the ids already on disk wherever the cell sequence is unchanged.

    nbformat mints a fresh random id for every cell on every build, so
    regenerating a notebook whose content did not change still produced a diff
    touching every cell — which buries the one line that did change and makes
    review of a generated artefact useless. Ids carry no meaning here beyond
    identity, so holding them steady costs nothing and keeps the diff honest.

    Positional, and only where the cell types still line up: once a cell is
    inserted or removed the ids below it are allowed to shift, which is the
    correct signal that the structure moved.
    """
    if not path.exists():
        return
    try:
        old = nbf.read(path, as_version=4).cells
    except Exception:
        return
    for a, b in zip(old, nb.cells):
        if a.cell_type == b.cell_type and "id" in a:
            b["id"] = a["id"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="execute each notebook and fail on the first error")
    ap.add_argument("--only", default="",
                    help="comma-separated lecture numbers, e.g. --only 19,20. "
                         "Executing all 24 takes hours, so a notebook under "
                         "development needs a way to be checked on its own.")
    args = ap.parse_args()
    wanted = {int(x) for x in args.only.split(",") if x.strip()}

    OUT.mkdir(parents=True, exist_ok=True)
    written = []
    for n, fn in sorted(LECTURES.items()):
        if wanted and n not in wanted:
            continue
        if n in COLAB_AUTHORED:
            print(f"  lecture-{n:02d}  SKIPPED — authored in Colab, not generated. "
                  f"See COLAB_AUTHORED in this file.")
            continue
        path = OUT / f"lecture-{n:02d}.ipynb"
        nb = fn()
        _ensure_prompt_note(nb, n)
        _keep_cell_ids(nb, path)
        nbf.write(nb, path)
        cells = nbf.read(path, as_version=4).cells
        n_code = sum(c.cell_type == "code" for c in cells)
        # Compile every code cell before claiming the notebook was written. The
        # modules hold cell bodies as ''' literals, so a single
        # backslash-n in one becomes a REAL newline in the generated cell and
        # cuts an f-string in half — valid Python in the module, a SyntaxError
        # in the notebook, and invisible until someone runs it. Executing all 24
        # to find that out takes hours; compiling them takes milliseconds.
        for i, c in enumerate(cells):
            if c.cell_type != "code":
                continue
            src = c.source
            if any(line.lstrip().startswith(("!", "%")) for line in src.splitlines()):
                continue                      # shell and magic lines are not Python
            try:
                compile(src, f"{path.name}:cell{i}", "exec")
            except SyntaxError as exc:
                print(f"  {path.name}: cell {i} does not compile — "
                      f"{exc.msg} at line {exc.lineno}")
                print(f"    {(exc.text or '').strip()[:90]}")
                return 1
        print(f"  {path.relative_to(ROOT)}  {n_code} code cells")
        written.append(path)

    missing = [] if wanted else [n for n in range(1, 25) if n not in LECTURES]
    if missing:
        print(f"\nnot yet written: lectures {missing[0]}–{missing[-1]} "
              f"({len(missing)} of 24)")

    if args.check:
        from nbclient import NotebookClient
        print("\nexecuting:")
        for path in written:
            nb = nbf.read(path, as_version=4)
            client = NotebookClient(nb, timeout=1800, kernel_name="python3",
                                    resources={"metadata": {"path": str(OUT)}})
            try:
                client.execute()
            except Exception as exc:                      # noqa: BLE001
                print(f"  {path.name}: FAILED — {type(exc).__name__}: "
                      f"{str(exc)[:300]}")
                return 1
            print(f"  {path.name}: ran clean")
    return 0


if __name__ == "__main__":
    sys.exit(main())
