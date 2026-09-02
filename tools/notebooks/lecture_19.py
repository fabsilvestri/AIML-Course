"""
Lecture 19 — Information retrieval: the lexical foundation.

Derivation 19: evaluating a ranking — P@k, R@k, RR, AP, DCG, NDCG. Then the
inverted index, BM25 assembled from the failures each of its parts corrects,
and the pooling assumption underneath every number in the lecture.

Exports build() -> list[nbformat cell].

SciFact is 5,183 abstracts. That is small enough to score EXACTLY: every
document for every query, no approximate index. So each number here is a
property of the method rather than of an index nobody inspected — which is the
only reason the comparison with Lecture 20 will mean anything.

Nothing is trained. Runs on CPU in about two minutes, most of it the download.
"""

from __future__ import annotations

import nbformat as nbf
from _prompt import prompt                                # noqa: E402


def md(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(text.strip("\n"))


def code(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(text.strip("\n"))


HEADER = """
# Information retrieval: the lexical foundation

**Lecture 19** · *Lecture notes — outside the textbook, and examinable* ·
*Derivation: evaluating a ranking*

Applications of Machine Learning — BSc Mathematics of Artificial Intelligence

---

**How to use this notebook.** Read before you run. Every code cell is preceded
by the specification that would produce it — input, output, constraint, check.

Cells marked **⚠** deliberately run code that is wrong, and say so in the
heading before you reach them. They are the failures this lecture is about;
each runs the broken version beside the correct one and prices the difference.

Runs on CPU, in about two minutes. Nothing here is trained — there is nothing
in this lecture to train, which is itself the point of the baseline.

**Scale.** Identical to the deck. SciFact is small enough that this notebook
scores every abstract for every claim exactly, so every number printed below is
the number on the slide.
"""


def build() -> list:
    cells: list = [md(HEADER)]

    # ------------------------------------------------------------------ 1
    cells += [
        md("## 1 · Setup and the corpus"),
        prompt(
            label="setup",
            input="nothing",
            output="versions and the seed",
            constraint="cap the BLAS thread count before numpy is imported, so the timings below are a property of the method rather than of this machine's core count",
            check="print the versions. A retrieval result you cannot pin to a corpus version is not reproducible, and BEIR has been re-released."),
        code('''
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import sys, math, re, urllib.request
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

print(f"python  {sys.version.split()[0]}")
print(f"numpy   {np.__version__}")
print(f"pandas  {pd.__version__}")

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
'''),
        prompt(
            label="SciFact, from the BEIR mirror",
            input="three files: the corpus, the claims, the test judgements",
            output="a DataFrame each, cached on disk",
            constraint="cache to disk so a re-run costs nothing, and read the qrels ids as STRINGS — pandas will helpfully turn document ids into integers and then two ids that differ only by a leading zero become the same document",
            check="assert the three shapes, and that every judged corpus-id exists in the corpus. A judgement pointing at a document you do not have is silently scored as a miss, and it will look like your method failed."),
        code('''
HF    = "https://huggingface.co/datasets/BeIR/"
CACHE = Path("datasets/scifact")
CACHE.mkdir(parents=True, exist_ok=True)

FILES = {
    "corpus":  (CACHE / "corpus.parquet",
                HF + "scifact/resolve/main/corpus/corpus-00000-of-00001.parquet"),
    "queries": (CACHE / "queries.parquet",
                HF + "scifact/resolve/main/queries/queries-00000-of-00001.parquet"),
    "qrels":   (CACHE / "qrels_test.tsv",
                HF + "scifact-qrels/resolve/main/test.tsv"),
}
for name, (path, url) in FILES.items():
    if not path.is_file():
        print(f"downloading {name} ...", flush=True)
        req = urllib.request.Request(url, headers={"User-Agent": "curl/8"})
        path.write_bytes(urllib.request.urlopen(req, timeout=60).read())

corpus  = pd.read_parquet(FILES["corpus"][0])
queries = pd.read_parquet(FILES["queries"][0])
# dtype=str, not the default: ids are opaque labels, and "0012" != 12.
qrels   = pd.read_csv(FILES["qrels"][0], sep="\\t",
                      dtype={"query-id": str, "corpus-id": str})

docs_id  = corpus["_id"].astype(str).tolist()
assert set(qrels["corpus-id"]) <= set(docs_id), "a judgement points at a document we do not have"
print(f"corpus   {len(corpus):,} abstracts")
print(f"claims   {len(queries):,} in total")
print(f"qrels    {len(qrels):,} judgements in the test split")
'''),
        prompt(
            label="what a document looks like",
            input="the first abstract",
            output="its title, the first 45 words, and its length in tokens",
            constraint="print it rather than describing it — the way these abstracts are written is the reason for the whole second half of this lecture",
            check="nothing to assert. Read it, and notice how little of a claim's wording is likely to appear here verbatim."),
        code('''
title0 = str(corpus["title"].iloc[0])
text0  = str(corpus["text"].iloc[0])
print(title0)
print()
print(" ".join(text0.split()[:45]), "...")
'''),
    ]

    # ------------------------------------------------------------------ 2
    cells += [
        md("""
## 2 · What a term is

Before any weighting there is a decision nobody writes up: how the text becomes
a list of things to count. Lower-case, split on runs of letters and digits.
That is three decisions already, and the third one silently splits
`0-dimensional` into two terms that mean nothing apart.
"""),
        prompt(
            label="tokenise",
            input="title and abstract, joined",
            output="the token lists, and the vocabulary size",
            constraint="one regex, applied identically to documents and to queries — a query tokenised differently from the corpus cannot match it, and the failure is silent",
            check="print the tokens of one real claim. Look at them before trusting anything downstream."),
        code('''
TOKEN = re.compile(r"[a-z0-9]+")

def tok(s: str) -> list[str]:
    return TOKEN.findall(s.lower())

docs_txt = [f"{t} {x}" for t, x in zip(corpus["title"], corpus["text"])]
toks     = [tok(d) for d in docs_txt]

doc_len_mean   = float(np.mean([len(t) for t in toks]))
doc_len_median = float(np.median([len(t) for t in toks]))
vocab          = len({w for t in toks for w in t})

print(f"abstracts        {len(toks):,}")
print(f"mean length      {doc_len_mean:.1f} tokens")
print(f"median length    {doc_len_median:.0f} tokens")
print(f"vocabulary       {vocab:,} distinct terms")
print(f"first abstract   {len(toks[0])} tokens")
'''),
        prompt(
            label="the judgements, as a lookup",
            input="the qrels table",
            output="claim id -> set of relevant document ids, and the claim texts",
            constraint="keep only judgements with a score above zero — BEIR files also carry explicit zeros, and counting them as relevant inflates every metric in this notebook",
            check="assert every claim kept has at least one relevant document, and report how many have exactly one. That fraction decides which metrics can distinguish anything here."),
        code('''
rel = defaultdict(set)
for _, r in qrels.iterrows():
    if int(r["score"]) > 0:                 # explicit zeros are judged-irrelevant
        rel[str(r["query-id"])].add(str(r["corpus-id"]))

qtext = dict(zip(queries["_id"].astype(str), queries["text"]))
qids  = [q for q in rel if q in qtext]
assert all(rel[q] for q in qids)

rel_per_query_mean = float(np.mean([len(rel[q]) for q in qids]))
frac_single_rel    = float(np.mean([len(rel[q]) == 1 for q in qids]))

print(f"test claims                 {len(qids)}")
print(f"relevant per claim, mean    {rel_per_query_mean:.2f}")
print(f"relevant per claim, max     {max(len(rel[q]) for q in qids)}")
print(f"claims with exactly one     {100*frac_single_rel:.1f}%")
print()
print("example claim:", qtext[qids[0]])
'''),
        prompt(
            label="what the tokeniser did to one claim",
            input="a claim from the test set",
            output="its tokens, with how many abstracts contain each",
            constraint="report the document frequency beside each token, because that is what decides whether the token can find anything",
            check="notice the spread. One term here occurs in a single abstract and another in a fifth of the corpus, and the weighting scheme has not been chosen yet."),
        code('''
claim_tok = qtext[qids[0]]
print(claim_tok)
print(tok(claim_tok))
'''),
    ]

    cells += [
        md("""
### What we are not doing: stemming

Our tokeniser treats `infect`, `infects`, `infection` and `infected` as four
unrelated terms. A stemmer would merge them, pooling their evidence — and would
also merge things that should not be merged, with a net effect that is
corpus-dependent and usually small.

We leave it out because every unexplained step between the text and the number
is somewhere a result can hide, and because the mismatch it half-solves has a
better answer in Lecture 20.
"""),
        prompt(
            label="what a stemmer would merge",
            input="two families of word forms",
            output="the document frequency of each form",
            constraint="use the index built above rather than re-scanning, and report the forms separately — the point is how differently the evidence is spread",
            check="compare the rarest form against the commonest in each family. That ratio is what a query for the rare form is giving up."),
        code('''
for family in (("cell", "cells", "cellular"),
               ("infect", "infects", "infection", "infected")):
    print("  " + "   ".join(f"{t}: {len(post.get(t, [])):,}" for t in family))
print()
print("A query for 'infect' matches only the abstracts using that exact form.")
'''),
    ]

    # ------------------------------------------------------------------ 3
    cells += [
        md("""
## 3 · The obvious first idea, and why it fails

Return the documents containing **all** the query terms. Exact, fast, and how
search worked for thirty years. Before running the next cell: on a corpus of
scientific abstracts, how many abstracts do you expect to contain every word of
a scientist's claim?
"""),
        prompt(
            label="the inverted index",
            input="the tokenised corpus",
            output="term -> [(document, count)], plus the postings total",
            constraint="one pass, a Counter per document — this is the entire data structure the field is built on, and it is four lines",
            check="assert the postings total equals the sum of distinct terms per document. Compare it against the dense term-document matrix it replaces.",
            **{"try": "count the postings a second time by summing len(set(d)) over documents. If the two disagree, the index has lost or duplicated an entry."}),
        code('''
post = defaultdict(list)
for i, d in enumerate(toks):
    for term, tf in Counter(d).items():
        post[term].append((i, tf))

postings = sum(len(p) for p in post.values())
assert postings == sum(len(set(d)) for d in toks)

dense = vocab * len(toks)
print(f"vocabulary   {vocab:,} terms")
print(f"postings     {postings:,} term-document pairs")
print(f"dense matrix {dense/1e6:.0f} million cells")
print(f"postings are {100*postings/dense:.1f}% of the dense matrix")
'''),
        prompt(
            label="document frequency of one claim's terms",
            input="the claim tokenised above, and the index",
            output="each term with the number of abstracts containing it",
            constraint="use the index, not a scan of the corpus — this is what the index is for",
            check="print them all. The rare term decides the answer; the common ones decide nothing."),
        code('''
for t in dict.fromkeys(tok(claim_tok)):
    print(f"  {t:<14} {len(post.get(t, [])):>6,} abstracts")
'''),
        prompt(
            label="Boolean retrieval, measured",
            input="every test claim",
            output="how many abstracts contain every term of the claim",
            constraint="intersect the postings sets — and report the fraction of claims matching NOTHING, because that is the failure mode",
            check="an empty result set is indistinguishable from 'no such document exists'. Count them.",
            **{"try": "relax the conjunction to a disjunction and count again. You will go from returning nothing to returning most of the corpus, unordered — which is why scoring, not filtering, is the answer."}),
        code('''
def boolean_and(q: str) -> int:
    sets = [set(i for i, _ in post.get(t, [])) for t in dict.fromkeys(tok(q))]
    if not sets:
        return 0
    out = sets[0]
    for s in sets[1:]:
        out &= s
    return len(out)

ands = [boolean_and(qtext[q]) for q in qids]
frac_zero     = float(np.mean([a == 0 for a in ands]))
mean_matching = float(np.mean(ands))

print(f"claims matching no document at all   {100*frac_zero:.1f}%")
print(f"mean documents matching every term   {mean_matching:.3f}")
print(f"most any claim matched               {max(ands)}")
'''),
        md("""
So the conjunction returns an **empty page** for almost every claim. One word of
the claim absent from an otherwise perfect abstract and the abstract is
excluded — and the failure is silent, because nothing looks like nothing.

Stop filtering. Start **scoring**: every document gets a number, and a missing
term costs something rather than everything.
"""),
    ]

    # ------------------------------------------------------------------ 4
    cells += [
        md("""
## 4 · BM25, one correction at a time

Raw term frequency has two failures, and BM25 is one correction for each, plus
a saturating curve. We build it in pieces so that each piece can be removed and
priced in section 6.
"""),
        prompt(
            label="idf",
            input="the index",
            output="a weight per term, falling as more documents contain it",
            constraint="use the BM25 form log(1 + (N - n + 0.5)/(n + 0.5)), which stays positive for a term in every document — the older log(N/n) goes negative there and silently rewards documents for NOT containing a query term",
            check="print the five commonest terms with their weights. If a stop-word list would have been needed, this has failed."),
        code('''
N = len(toks)
idf = {t: math.log(1 + (N - len(p) + 0.5) / (len(p) + 0.5)) for t, p in post.items()}

df = {t: len(p) for t, p in post.items()}
common = sorted(df.items(), key=lambda kv: -kv[1])[:5]
print("the five commonest terms in the corpus:")
for t, n in common:
    print(f"  {t:<8} {n:>6,} abstracts   idf {idf[t]:.4f}")
'''),
        md("""
No stop-word list was written. The weighting **derived** the stop words from the
corpus it was given, and a weight near zero is near-deletion — with everything
in between still available, which a list cannot offer.
"""),
        prompt(
            label="saturation",
            input="a term frequency and k1",
            output="the tf factor, for a document of exactly average length",
            constraint="show that it is bounded — the tenth occurrence of a word must not count ten times the first, or a page repeating a word wins every query",
            check="tabulate the factor at tf = 1, 2, 3, 5, 10, 20 and confirm the increments shrink."),
        code('''
K1, B = 0.9, 0.4          # the two BM25 parameters, tuned by measurement below

def tf_factor(tf, k1=K1):
    """The tf term for a document of exactly average length (so the b term is 1)."""
    return tf * (k1 + 1) / (tf + k1)

print("tf   factor")
for t in (1, 2, 3, 5, 10, 20):
    print(f"{t:>3}   {tf_factor(t):.3f}")
print()
print(f"1 -> 2 occurrences adds {tf_factor(2)-tf_factor(1):.3f}")
print(f"10 -> 20 adds           {tf_factor(20)-tf_factor(10):.3f}")
'''),
        prompt(
            label="BM25 itself",
            input="a tokenised query",
            output="a score for every document in the corpus",
            constraint="walk the postings, not the corpus — a document sharing no term with the query scores zero and must never be touched",
            check="assert that a query of one very rare term touches only the handful of documents containing it. Score every document exactly: no approximate index, so every number below is a property of the method.",
            **{"try": "set b to 0 in the denominator and re-score. Section 6 prices exactly this change."}),
        code('''
doc_len = np.array([len(t) for t in toks], dtype=np.float64)
avgdl   = float(doc_len.mean())

def bm25_scores(q_toks, k1=K1, b=B, idf=idf, post=post):
    s = np.zeros(N)
    for t in q_toks:
        p = post.get(t)
        if not p:
            continue                        # a term nobody uses contributes nothing
        w = idf[t]
        for i, tf in p:
            denom = tf + k1 * (1 - b + b * doc_len[i] / avgdl)
            s[i] += w * tf * (k1 + 1) / denom
    return s

rare = min(df, key=df.get)
assert int((bm25_scores([rare]) > 0).sum()) == df[rare]

print(f"average document length {avgdl:.1f} tokens")
print(f"a query of the rarest term touches {df[rare]} of {N:,} documents")
'''),
    ]

    # ------------------------------------------------------------------ 5
    cells += [
        md("""
## 5 · Derivation 19 — evaluating a ranking

Given a ranked list and the set $R$ of relevant documents, and nothing else.
Everything below is a function of those two objects, which is why the same
metrics will apply unchanged to the neural retrievers of Lecture 20.
"""),
        prompt(
            label="the metrics, from the derivation",
            input="a ranked list of document ids and the relevant set",
            output="P@k, R@k, RR, AP and NDCG@10",
            constraint="log2(i+1), not log2(i) — the latter divides by zero at rank 1. And divide AP by |R|, not by the number of hits found: dividing by the hits gives mean precision, which is a different and more flattering quantity",
            check="check it against a hand-computed case in the next cell before trusting it on 300 claims.",
            **{"try": "change the discount to log2(i) and watch rank 1 become infinite; change the AP denominator to the hit count and watch every score rise."}),
        code('''
def rank_metrics(ranked, R, ks=(1, 5, 10, 100)):
    out = {}
    for k in ks:
        hits = sum(1 for d in ranked[:k] if d in R)
        out[f"p@{k}"] = hits / k
        out[f"r@{k}"] = hits / len(R) if R else 0.0

    out["rr"] = next((1.0 / i for i, d in enumerate(ranked, 1) if d in R), 0.0)

    dcg  = sum(1.0 / math.log2(i + 1) for i, d in enumerate(ranked[:10], 1) if d in R)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(R), 10) + 1))
    out["ndcg@10"] = dcg / idcg if idcg else 0.0

    ap, hits = 0.0, 0
    for i, d in enumerate(ranked, 1):
        if d in R:
            hits += 1
            ap += hits / i
    out["ap"] = ap / len(R) if R else 0.0    # |R|, not hits
    return out
'''),
        prompt(
            label="check it by hand first",
            input="a made-up ranking with relevant documents at ranks 1, 2 and 8",
            output="RR, AP and NDCG@10, each also computed longhand",
            constraint="write the longhand sums out as literal arithmetic, so the assert compares two independent computations rather than one function against itself",
            check="assert the two agree to 1e-12. A metric implementation that has never been checked against a hand computation is the commonest silent error in this literature."),
        code('''
toy_rank = [f"d{i}" for i in range(1, 11)]
toy_rel  = {"d1", "d2", "d8"}
m = rank_metrics(toy_rank, toy_rel)

hand_rr   = 1 / 1
hand_ap   = (1/1 + 2/2 + 3/8) / 3
hand_dcg  = 1/math.log2(2) + 1/math.log2(3) + 1/math.log2(9)
hand_idcg = 1/math.log2(2) + 1/math.log2(3) + 1/math.log2(4)
hand_ndcg = hand_dcg / hand_idcg

assert abs(m["rr"]      - hand_rr)   < 1e-12
assert abs(m["ap"]      - hand_ap)   < 1e-12
assert abs(m["ndcg@10"] - hand_ndcg) < 1e-12

print(f"RR       {m['rr']:.4f}")
print(f"AP       {m['ap']:.4f}")
print(f"DCG@10   {hand_dcg:.4f}")
print(f"IDCG@10  {hand_idcg:.4f}")
print(f"NDCG@10  {m['ndcg@10']:.4f}")
print(f"R@10     {m['r@10']:.4f}   <- calls this ranking perfect")
'''),
        md("""
Recall@10 is 1: all three relevant documents are on the page. NDCG is not,
because one of them is at rank 8. **The metric you report decides what you are
allowed to notice.**
"""),
        prompt(
            label="one relevant document, moved down the list",
            input="ranks 1, 2, 5 and 10",
            output="RR, NDCG@10 and recall@10 at each",
            constraint="hold everything else fixed — one relevant document, always inside the top ten, only its position changing",
            check="recall@10 should be constant at 1 across the whole row. That is the point of the cell."),
        code('''
print("rank   RR      NDCG@10   R@10")
for r in (1, 2, 5, 10):
    ranked = [f"x{i}" for i in range(1, 11)]
    ranked[r-1] = "gold"
    mm = rank_metrics(ranked, {"gold"})
    print(f"{r:>4}   {mm['rr']:.4f}  {mm['ndcg@10']:.4f}    {mm['r@10']:.4f}")
'''),
    ]

    # ------------------------------------------------------------------ 6
    cells += [
        md("""
## 6 · BM25 measured, and each part priced

Rule 2 first: what does *no* method score? Then BM25, then each of its
corrections removed in turn.
"""),
        prompt(
            label="the baseline",
            input="the corpus in its stored order, the same order for every claim",
            output="the same metrics",
            constraint="score it before scoring anything else, so the comparison exists before the result does",
            check="it should be near zero. Unlike accuracy under imbalance, a ranking metric has an honest floor."),
        code('''
fixed = [rank_metrics(docs_id, rel[q]) for q in qids]
fixed_order = {k: float(np.mean([r[k] for r in fixed])) for k in fixed[0]}
print("returning the corpus unranked:")
print(f"  recall@10   {fixed_order['r@10']:.4f}")
print(f"  MRR         {fixed_order['rr']:.4f}")
print(f"  NDCG@10     {fixed_order['ndcg@10']:.4f}")
'''),
        prompt(
            label="BM25 over every test claim",
            input="all 300 claims, scored against all 5,183 abstracts",
            output="the mean of each metric",
            constraint="average the per-claim metric (macro), not pooled counts — and say which, because for precision and recall the two differ",
            check="assert every claim was scored. A silently skipped query lowers nothing and raises the mean."),
        code('''
rows = []
for q in qids:
    order = np.argsort(-bm25_scores(tok(qtext[q])))
    rows.append(rank_metrics([docs_id[i] for i in order], rel[q]))
assert len(rows) == len(qids)

bm25 = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
print("BM25, macro-averaged over every test claim:")
for k in ("p@1", "r@10", "r@100", "rr", "ap", "ndcg@10"):
    print(f"  {k:<8} {bm25[k]:.4f}")
'''),
        md("""
No training, no labels, no gradient — a dictionary and two corrections.

Hold on to the gap between recall@10 and recall@100: almost everything relevant
is retrieved *somewhere* in the first hundred. Closing that gap is what
re-ranking is for, and it is Lecture 20.
"""),
        prompt(
            label="one document, scored term by term",
            input="a claim and the abstract BM25 ranked first for it",
            output="each query term with its tf, its idf and its contribution",
            constraint="reproduce the sum — the contributions must add to the score the scorer returned, or one of the two is wrong",
            check="assert the term contributions sum to the total within 1e-9."),
        code('''
# The same selection rule as the slide: the first test claim for which BM25
# puts at least three relevant abstracts in the top ten, chosen so the
# arithmetic is visible. It is therefore better than typical; the honest
# summary of the method is the mean over all 300 claims, computed above.
def hits_at_10(q):
    top10 = [docs_id[i] for i in np.argsort(-bm25_scores(tok(qtext[q])))[:10]]
    return sum(1 for d in top10 if d in rel[q])

wq       = next(q for q in qids if len(rel[q]) >= 3 and hits_at_10(q) >= 3)
wq_score = bm25_scores(tok(qtext[wq]))
top      = int(np.argmax(wq_score))

print(qtext[wq])
print(f"top abstract: {doc_len[top]:.0f} tokens, judged relevant: {docs_id[top] in rel[wq]}")
print()
print("term          tf     idf    contribution")
total = 0.0
# Counter, not a set: bm25_scores walks the query token list, so a term the
# query repeats contributes once per occurrence. Deduplicating here would
# print a table that does not add up to the score beside it.
for t, q_count in Counter(tok(qtext[wq])).items():
    tf = dict(post.get(t, [])).get(top, 0)
    if not tf:
        continue
    denom = tf + K1 * (1 - B + B * doc_len[top] / avgdl)
    c = q_count * idf[t] * tf * (K1 + 1) / denom
    total += c
    print(f"{t:<12} {tf:>3}   {idf[t]:>6.3f}    {c:>7.3f}")
assert abs(total - wq_score[top]) < 1e-9, "the table does not sum to the score"
print(f"{'total':<12} {'':>3}   {'':>6}    {total:>7.3f}")
'''),
        prompt(
            label="each correction, removed and priced",
            input="four scoring functions: raw tf, no idf, no saturation, no length normalisation",
            output="NDCG@10 for each, beside full BM25",
            constraint="change ONE thing per row — an ablation that changes two tells you nothing about either",
            check="raw term frequency should be barely above the unranked baseline. If it is not, the idf implementation is wrong.",
            **{"try": "add a row that removes idf AND saturation together. It will not equal the sum of the two individual drops, which is what interaction means."}),
        code('''
def evaluate(scorer):
    return float(np.mean([
        rank_metrics([docs_id[i] for i in np.argsort(-scorer(tok(qtext[q])))],
                     rel[q])["ndcg@10"]
        for q in qids]))

def raw_tf(q):
    s = np.zeros(N)
    for t in q:
        for i, tf in post.get(t, []):
            s[i] += tf
    return s

def no_idf(q):
    s = np.zeros(N)
    for t in q:
        for i, tf in post.get(t, []):
            denom = tf + K1 * (1 - B + B * doc_len[i] / avgdl)
            s[i] += tf * (K1 + 1) / denom
    return s

def no_saturation(q):          # idf-weighted counts, no saturation, no length
    s = np.zeros(N)
    for t in q:
        if t in post:
            for i, tf in post[t]:
                s[i] += idf[t] * tf
    return s

def no_length(q):
    return bm25_scores(q, b=0.0)

ablation = [
    ("raw term frequency",       evaluate(raw_tf)),
    ("no idf",                   evaluate(no_idf)),
    ("no saturation, no length", evaluate(no_saturation)),
    ("no length normalisation",  evaluate(no_length)),
    ("BM25",                     bm25["ndcg@10"]),
]
for name, v in ablation:
    print(f"  {name:<28} ndcg@10 {v:.4f}")
'''),
        prompt(
            label="the two parameters, swept",
            input="b in {0, 0.4, 0.75, 1} and k1 in {0.5, 0.9, 1.2, 2}",
            output="NDCG@10 for each, over the first 150 claims",
            constraint="sweep one parameter with the other held fixed, and use the same 150 claims throughout so the rows are comparable",
            check="report the spread, not just the best value. A parameter whose whole range moves the metric by less than the noise is a parameter you should stop tuning."),
        code('''
def sweep_ndcg(k1=K1, b=B, n=150):
    return float(np.mean([
        rank_metrics([docs_id[i] for i in np.argsort(-bm25_scores(tok(qtext[q]), k1=k1, b=b))],
                     rel[q])["ndcg@10"]
        for q in qids[:n]]))

print("b      ndcg@10")
b_sweep = [(b, sweep_ndcg(b=b)) for b in (0.0, 0.4, 0.75, 1.0)]
for b, v in b_sweep:
    print(f"{b:<5}  {v:.4f}")

print()
print("k1     ndcg@10")
k1_sweep = [(k1, sweep_ndcg(k1=k1)) for k1 in (0.5, 0.9, 1.2, 2.0)]
for k1, v in k1_sweep:
    print(f"{k1:<5}  {v:.4f}")
print()
print(f"spread across the whole k1 range: {max(v for _,v in k1_sweep)-min(v for _,v in k1_sweep):.4f}")
'''),
    ]

    # ------------------------------------------------------------------ 7
    cells += [
        md("""
## 7 · The failure BM25 cannot fix

Everything above matches **strings**. Relevance is about **meaning**. Here is
what that costs, on the worst case in this test set and then on average.
"""),
        prompt(
            label="the worst single-evidence claim",
            input="every claim with exactly one relevant abstract",
            output="the claim BM25 ranks its evidence lowest, with the term overlap",
            constraint="report the overlap as well as the rank, because the overlap is the explanation and the rank alone is just a bad number",
            check="print which terms are shared. If they are function words, no parameter setting rescues this."),
        code('''
worst = None
for q in qids:
    if len(rel[q]) != 1:
        continue
    order = [docs_id[i] for i in np.argsort(-bm25_scores(tok(qtext[q])))]
    r = order.index(next(iter(rel[q]))) + 1
    if worst is None or r > worst[1]:
        worst = (q, r)

wq2, wrank = worst
gold      = next(iter(rel[wq2]))
gold_toks = set(toks[docs_id.index(gold)])
q_terms   = [t for t in dict.fromkeys(tok(qtext[wq2])) if t in post]
shared    = [t for t in q_terms if t in gold_toks]
missing   = [t for t in q_terms if t not in gold_toks]

print(qtext[wq2])
print()
print(f"BM25 ranks its one relevant abstract    {wrank:,} of {N:,}")
print(f"query terms in the vocabulary           {len(q_terms)}")
print(f"terms the relevant abstract contains    {len(shared)}")
print(f"which terms                             {shared}")
print(f"mean idf of the shared terms            {float(np.mean([idf[t] for t in shared])):.4f}")
print(f"mean idf of the missing terms           {float(np.mean([idf[t] for t in missing])):.4f}")
'''),
        prompt(
            label="vocabulary mismatch in general",
            input="every judged claim-abstract pair",
            output="the mean fraction of query terms the relevant abstract does not contain",
            constraint="over judged pairs, not over retrieved ones — otherwise you measure what BM25 found rather than what is there",
            check="the fraction should be large. If it is near zero, the tokeniser is merging things it should not."),
        code('''
frac = []
for q in qids:
    qt = [t for t in dict.fromkeys(tok(qtext[q])) if t in post]
    if not qt:
        continue
    for d in rel[q]:
        dt = set(toks[docs_id.index(d)])
        frac.append(sum(1 for t in qt if t not in dt) / len(qt))

mean_fraction = float(np.mean(frac))
over_half     = float(np.mean([x > 0.5 for x in frac]))
print(f"mean fraction of query terms absent from the relevant abstract  {100*mean_fraction:.1f}%")
print(f"pairs sharing fewer than half the query terms                   {100*over_half:.1f}%")
'''),
        md("""
A relevant document is missing two-fifths of the query's words on average.
People restate, abbreviate, use the Latin name, or describe the mechanism
instead of naming it — and a method whose only evidence is shared strings cannot
follow them.

That gap is not a tuning problem. It is what an embedding is for, and it is
Lecture 20.
"""),
    ]

    # ------------------------------------------------------------------ 8
    cells += [
        md("""
## 8 · The assumption under every number above

Each metric took $R$ as given. Somebody decided, for 300 claims against 5,183
abstracts, which pairs are relevant — and they did not make all 1.55 million
decisions.
"""),
        prompt(
            label="how sparse the judgements are",
            input="the qrels",
            output="the judged fraction of all claim-document pairs",
            constraint="state it as a fraction of the whole grid, because that is the quantity that makes the pooling problem visible",
            check="everything unjudged is scored as irrelevant. Print how much of the grid that is."),
        code('''
grid   = len(qids) * N
judged = sum(len(rel[q]) for q in qids)
print(f"claim-abstract pairs in the grid   {grid:,}")
print(f"pairs judged relevant              {judged:,}")
print(f"judged relevant, as a fraction     {100*judged/grid:.4f}%")
print()
print("Everything else is treated as irrelevant. Most of it was never read.")
'''),
        md("""
For SciFact the judgements come from the authors of the claims, who cited the
abstracts: unusually reliable, and unusually **sparse**. An abstract that
supports a claim but was not cited is scored as a miss.

So the NDCG above is a **lower bound** on what a reader would call the quality
of this ranking, not an estimate of it. Reporting it without that sentence is
the kind of claim this course exists to stop.
"""),
    ]

    # ------------------------------------------------------------------ 9
    cells += [
        md("""
## 9 · Five questions to ask of any retrieval result

1. **Which metric, and at what cutoff?** "Better retrieval" with no metric named
   is not a claim.
2. **Against which baseline?** If BM25 is absent from the table, ask why — it is
   free, and it is often close.
3. **Whose judgements, produced how?** Pooled from which systems, how long ago?
4. **Same corpus, same queries, same preprocessing?** A different tokeniser is a
   different experiment.
5. **How large is the difference against the spread across queries?** A gain of
   0.005 on 300 queries is not a finding.

---

**Next.** Lecture 20 closes the two gaps this notebook opened: the documents
BM25 found but ranked too low, and the ones it could never have found at all.
"""),
    ]

    return cells
