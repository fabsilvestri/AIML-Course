"""
Lecture 20 — Information retrieval: dense retrieval.

Bi-encoders, in-batch negatives, cross-encoder re-ranking, hybrid fusion, and
an IVF index built by hand. No derivation: the mathematics is a normalised
inner product.

Exports build() -> list[nbformat cell].

The BM25 half is the code of Lecture 19, reproduced rather than imported, so
the baseline in every comparison here is the baseline the student built. A
comparison against a re-implemented baseline is not a comparison.

Runs on CPU in about ten minutes, most of it encoding the corpus once. Nothing
is trained -- which is the honest form of the question this lecture asks.
"""

from __future__ import annotations

import nbformat as nbf
from _prompt import prompt                                # noqa: E402


def md(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(text.strip("\n"))


def code(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(text.strip("\n"))


HEADER = """
# Information retrieval: dense retrieval

**Lecture 20** · *Lecture notes — outside the textbook, and examinable*

Applications of Machine Learning — BSc Mathematics of Artificial Intelligence

---

**These notes are the primary source for this lecture.** Lectures 19–22 sit outside the textbook, so the extended notes — [lecture-20.pdf](https://fabsilvestri.github.io/AIML-Course/notes/lecture-20.pdf) — are what you are examined from, and this notebook is where their figures come from.

**How to use this notebook.** Read before you run. Every code cell is preceded
by the specification that would produce it — input, output, constraint, check.

Cells marked **⚠** deliberately run code that is wrong, and say so in the
heading before you reach them. They are the failures this lecture is about;
each runs the broken version beside the correct one and prices the difference.

Runs on CPU, in about ten minutes. Encoding the 5,183 abstracts once is most of
it. Nothing is trained.

**Scale.** Identical to the deck. Both retrievers score every abstract for every
claim exactly — no approximate index — so the numbers measure the methods. The
approximate index is then built separately and measured *against* that exact
answer, which is the only way its recall means anything.
"""


def build() -> list:
    cells: list = [md(HEADER)]

    # ------------------------------------------------------------------ 1
    cells += [
        md("## 1 · Setup, the corpus, and last week's baseline"),
        prompt(
            label="setup",
            input="nothing",
            output="versions, seed, device",
            constraint="cap the thread counts before torch is imported — they are read at import time and after that they do nothing",
            check="print the model versions. A retrieval number without the encoder version is not reproducible, and these checkpoints are updated.",
            **{"try": "the check asks for the model versions and this cell does "
                      "not print them. Add the bi-encoder and cross-encoder "
                      "checkpoint names and their revision hashes once they are "
                      "loaded, and decide whether a specification whose check is "
                      "not implemented is a specification at all."}),
        code('''
import os
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import sys, math, re, time, urllib.request
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

print(f"python  {sys.version.split()[0]}")
print(f"numpy   {np.__version__}")
print(f"torch   {torch.__version__}")

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)
'''),
        prompt(
            label="the same SciFact, the same judgements",
            input="the three cached files from Lecture 19",
            output="corpus, claims and test qrels",
            constraint="read the ids as strings, exactly as last week — an id read as an integer silently merges documents whose ids differ by a leading zero",
            check="assert the corpus size and the number of test claims match Lecture 19's. If either differs, every comparison below is against a different experiment.",
            **{"try": "delete the cached parquet files and re-download. If the "
                      "corpus-size assert fires, BEIR has been re-released and "
                      "Lecture 19's numbers are no longer the baseline this "
                      "notebook subtracts from. What would you do next, and what "
                      "would you publish?"}),
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
qrels   = pd.read_csv(FILES["qrels"][0], sep="\\t",
                      dtype={"query-id": str, "corpus-id": str})

docs_id  = corpus["_id"].astype(str).tolist()
docs_txt = [f"{t} {x}" for t, x in zip(corpus["title"], corpus["text"])]
N = len(docs_id)

rel = defaultdict(set)
for _, r in qrels.iterrows():
    if int(r["score"]) > 0:
        rel[str(r["query-id"])].add(str(r["corpus-id"]))
qtext = dict(zip(queries["_id"].astype(str), queries["text"]))
qids  = [q for q in rel if q in qtext]

assert N == 5183 and len(qids) == 300, "not the corpus Lecture 19 used"
print(f"{N:,} abstracts, {len(qids)} test claims — same as Lecture 19")
'''),
        prompt(
            label="BM25 and the metrics, as built last week",
            input="the tokenised corpus",
            output="the index, the scorer, and rank_metrics",
            constraint="reproduce Lecture 19's code exactly rather than importing a library — the whole point of this notebook is a comparison, and a comparison needs the baseline to be the one you understand",
            check="assert the resulting NDCG@10 reproduces Lecture 19's number. If it does not, stop: something in the pipeline changed and nothing below can be trusted.",
            **{"try": "change K1 to 1.2 in this cell alone. The assert "
                      "reproducing Lecture 19's NDCG@10 fires, which is "
                      "exactly what it is for: every comparison below "
                      "subtracts a baseline, and the baseline has to be the "
                      "same one."}),
        code('''
TOKEN = re.compile(r"[a-z0-9]+")
def tok(s): return TOKEN.findall(s.lower())

toks = [tok(d) for d in docs_txt]
post = defaultdict(list)
for i, d in enumerate(toks):
    for term, tf in Counter(d).items():
        post[term].append((i, tf))

doc_len = np.array([len(t) for t in toks], dtype=np.float64)
avgdl   = float(doc_len.mean())
idf = {t: math.log(1 + (N - len(p) + 0.5) / (len(p) + 0.5)) for t, p in post.items()}
K1, B = 0.9, 0.4

def bm25_scores(q_toks):
    s = np.zeros(N)
    for t in q_toks:
        p = post.get(t)
        if not p:
            continue
        w = idf[t]
        for i, tf in p:
            s[i] += w * tf * (K1 + 1) / (tf + K1 * (1 - B + B * doc_len[i] / avgdl))
    return s

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
    out["ap"] = ap / len(R) if R else 0.0
    return out

bm_order, rows = {}, []
for q in qids:
    o = np.argsort(-bm25_scores(tok(qtext[q])))
    bm_order[q] = o
    rows.append(rank_metrics([docs_id[i] for i in o], rel[q]))
bm25 = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
print(f"BM25 ndcg@10 {bm25['ndcg@10']:.4f}   r@10 {bm25['r@10']:.4f}   "
      f"r@100 {bm25['r@100']:.4f}")

# The reproduction assert the specification above asks for, and which was
# missing until 2026-09-04. Every comparison in this notebook subtracts this
# baseline, so if the rebuilt BM25 is not Lecture 19's BM25 to the fourth
# decimal, nothing below is a comparison -- it is two different experiments.
# Pinning a number is right here for the same reason it is right in Lecture 13,
# where the architecture is re-typed and its parameter count is asserted
# against the previous lecture's: it is an invariant ACROSS notebooks.
assert abs(bm25["ndcg@10"] - 0.6611) < 1e-4, (
    f"BM25 scores {bm25['ndcg@10']:.4f} here and 0.6611 in Lecture 19 — "
    f"the baseline moved, so stop and find out why before reading on")
'''),
    ]

    # ------------------------------------------------------------------ 2
    cells += [
        md("""
## 2 · The failure we are here to fix

Lecture 19 ended on the claim BM25 buries. Find it again, and note what it is
about the pair that defeats a lexical index.
"""),
        prompt(
            label="the worst single-evidence claim",
            input="every claim with exactly one relevant abstract",
            output="the claim, its evidence's rank, and the shared terms",
            constraint="report the overlap alongside the rank — the rank is the symptom and the overlap is the cause",
            check="the shared terms should be function words. If they are not, this is a different failure.",
            **{"try": "print this claim's terms with their idf. The shared "
                      "ones are the low-idf terms and the missing ones carry "
                      "all the weight. BM25 did not rank the evidence low by "
                      "accident — it ranked it low correctly, on the evidence "
                      "available to it."}),
        code('''
worst = None
for q in qids:
    if len(rel[q]) != 1:
        continue
    r = [docs_id[i] for i in bm_order[q]].index(next(iter(rel[q]))) + 1
    if worst is None or r > worst[1]:
        worst = (q, r)
wq, bm_rank = worst
gold      = next(iter(rel[wq]))
gold_toks = set(toks[docs_id.index(gold)])
q_terms   = [t for t in dict.fromkeys(tok(qtext[wq])) if t in post]
shared    = [t for t in q_terms if t in gold_toks]

print(qtext[wq])
print()
print(f"BM25 ranks its one relevant abstract  {bm_rank:,} of {N:,}")
print(f"of {len(q_terms)} query terms, the abstract contains {len(shared)}: {shared}")
'''),
    ]

    # ------------------------------------------------------------------ 3
    cells += [
        md("""
## 3 · A bi-encoder, zero shot

One encoder applied separately to the query and to the document; the score is
the inner product of the two normalised vectors. Because $E(d)$ does not depend
on the query, the corpus can be encoded once and stored — which is the entire
reason this architecture is usable, and also why it is the weaker of the two
neural models in this lecture.

The model is `all-MiniLM-L6-v2`, trained on about a billion general web sentence
pairs and never shown a scientific claim. **Write down now** whether you expect
it to beat BM25's 0.6611.
"""),
        prompt(
            label="encode the corpus once",
            input="the 5,183 abstracts",
            output="a matrix of unit-norm vectors",
            constraint="normalise, so the inner product is a cosine — unnormalised, the magnitude tracks length and token frequency, and long documents win by default, which is the failure BM25's b was invented for",
            check="assert every row has unit norm and that the matrix is the shape you expect. Report the storage cost beside the postings count from section 1.",
            **{"try": "pass normalize_embeddings=False and re-run Section 4. "
                      "The ranking changes, because the dot product now "
                      "rewards magnitude and magnitude tracks length. That is "
                      "the failure BM25's b was invented for, turning up in a "
                      "model that has no b."}),
        code('''
from sentence_transformers import SentenceTransformer

BI = "sentence-transformers/all-MiniLM-L6-v2"
bi = SentenceTransformer(BI)

t0 = time.perf_counter()
D = bi.encode(docs_txt, batch_size=64, convert_to_numpy=True,
              normalize_embeddings=True, show_progress_bar=False)
print(f"encoded {N:,} abstracts in about {time.perf_counter()-t0:.0g} s")

assert D.shape == (N, 384)
assert np.allclose(np.linalg.norm(D, axis=1), 1.0, atol=1e-5)

postings = sum(len(p) for p in post.values())
print(f"dense index    {D.nbytes/1e6:.1f} MB  ({D.shape[1]*4:,} bytes per document)")
print(f"postings       {4*postings/1e6:.1f} MB  ({4*postings//N:,} bytes per document)")
print("every byte of the dense index is read by every exact query;")
print("the postings list skips documents sharing no term.")
'''),
        prompt(
            label="search, exactly",
            input="the claim texts",
            output="a full ranking of the corpus per claim, and the same metrics",
            constraint="score every document — no approximate index yet, so these numbers are a property of the model and not of an index setting",
            check="assert one ranking is a permutation of the corpus. A partially-scored ranking silently truncates recall@100.",
            **{"try": "check whether this checkpoint was trained with "
                      "distinct query and passage roles. all-MiniLM-L6-v2 is "
                      "symmetric, so encoding both sides the same way is "
                      "right — for e5 or bge it is wrong, silently, by "
                      "several points. Where would you have had to look to "
                      "find that out?"}),
        code('''
Q = bi.encode([qtext[q] for q in qids], batch_size=64, convert_to_numpy=True,
              normalize_embeddings=True, show_progress_bar=False)

dense_order, rows = {}, []
for j, q in enumerate(qids):
    o = np.argsort(-(D @ Q[j]))
    dense_order[q] = o
    rows.append(rank_metrics([docs_id[i] for i in o], rel[q]))
assert len(set(dense_order[qids[0]].tolist())) == N

dense = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
print("               BM25      dense")
for k in ("ndcg@10", "p@1", "rr", "r@10", "r@100"):
    print(f"  {k:<8} {bm25[k]:>8.4f}  {dense[k]:>8.4f}")
'''),
        md("""
**The neural model loses on the headline metric.** No training, no GPU and no
vector store, and BM25 is ahead on NDCG@10, P@1 and MRR.

It is not a bug in the experiment: same corpus, same claims, same judgements,
same metric code. It is the result BEIR was built to report — out-of-domain
dense retrieval loses to BM25 on a good fraction of its tasks, and scientific
text is exactly where it should be worst.

Now look at recall@100, which is what a *first stage* is judged on.
"""),
        prompt(
            label="the mismatch case, retried",
            input="the claim from section 2",
            output="the rank its evidence gets from each retriever",
            constraint="report both ranks side by side and nothing else — this is one comparison, not a table",
            check="nothing about the text changed. If the dense rank is also poor, the vocabulary-mismatch story does not hold on this corpus and you should say so.",
            **{"try": "run this comparison for the ten worst single-evidence "
                      "claims instead of one. If the dense retriever wins on "
                      "all ten the story holds; if it wins on six you have an "
                      "anecdote, and the table in Section 4 is the "
                      "measurement."}),
        code('''
d_rank = [docs_id[i] for i in dense_order[wq]].index(gold) + 1
print(qtext[wq])
print()
print(f"BM25   rank {bm_rank:>6,}")
print(f"dense  rank {d_rank:>6,}")
'''),
        prompt(
            label="do they find the same documents?",
            input="every relevant abstract in the test set",
            output="how many are in the top ten of both, one, or neither",
            constraint="count relevant DOCUMENTS, not queries — two systems can score the same and fail on entirely different documents, and only a document-level count shows it",
            check="if the 'only' counts are near zero the two systems are redundant and fusion will not help. If they are large, it will.",
            **{"try": "repeat the count at the top 100 rather than the top "
                      "10. The 'neither' column shrinks and the two 'only' "
                      "columns change relative size. At what depth do the two "
                      "systems stop being complementary, and what does that "
                      "say about where fusion belongs?"}),
        code('''
both = only_bm = only_de = neither = 0
for q in qids:
    b = {docs_id[i] for i in bm_order[q][:10]} & rel[q]
    d = {docs_id[i] for i in dense_order[q][:10]} & rel[q]
    for g in rel[q]:
        if g in b and g in d: both += 1
        elif g in b:          only_bm += 1
        elif g in d:          only_de += 1
        else:                 neither += 1
total = both + only_bm + only_de + neither

print(f"relevant abstracts in the test set   {total}")
print(f"  found in the top 10 by both        {both}")
print(f"  by BM25 only                       {only_bm}")
print(f"  by the dense model only            {only_de}")
print(f"  by neither                         {neither}")
print()
print(f"{only_bm + only_de} of {total} were found by exactly one of the two.")
'''),
    ]

    # ------------------------------------------------------------------ 3b
    cells += [
        md("""
### What a batch buys, if you were training one

We are not training a bi-encoder here, but the arithmetic of how one is trained
is examinable and it is one line. In a batch of $B$ query-document pairs, every
*other* document in the batch is a negative for every query: encoding grows
linearly with $B$, and the scores available grow quadratically.
"""),
        prompt(
            label="in-batch negatives, counted",
            input="a range of batch sizes",
            output="encoder passes, scores available, and negatives per query",
            constraint="separate the two columns that grow differently — the encodings are the cost and the scores are the benefit, and conflating them is why people call in-batch negatives free",
            check="the scores column should be the square of the batch. That quadratic is why dense-retrieval papers report batch sizes in the thousands.",
            **{"try": "add a column for the memory the score matrix needs in "
                      "float32. At batch 512 it is about a megabyte; at "
                      "16,384 it is a gigabyte. The dot products are nearly "
                      "free in arithmetic and are not free in memory, and "
                      "that is what actually caps the batch."}),
        code('''
print(f"{'batch':>7} {'encoded':>9} {'scores':>10} {'negatives/query':>16}")
for b in (8, 32, 128, 512):
    print(f"{b:>7,} {2*b:>9,} {b*b:>10,} {b-1:>16,}")
print()
print("Encoding is the cost and grows linearly; the dot products are nearly")
print("free and grow quadratically. The batch size is a modelling decision.")
'''),
    ]

    # ------------------------------------------------------------------ 4
    cells += [
        md("""
## 4 · Hybrid: keep both

The scores are not on a comparable scale — BM25 returns unbounded sums, the
dense model returns cosines in $[-1,1]$ — so fuse the **ranks**. Reciprocal rank
fusion needs nothing from either system but its ordering, and has nothing to
tune, which is why it is what people deploy.
"""),
        prompt(
            label="reciprocal rank fusion",
            input="the two rankings",
            output="a fused ranking, and the metrics",
            constraint="fuse ranks, not scores — normalising two incomparable score scales is another modelling choice, and it needs held-out labels to tune",
            check="assert the fused ranking is still a permutation. The constant k=60 is convention; try 10 and 200 and see how little it matters.",
            **{"try": "replace RRF with a normalised score sum, alpha * bm25 + (1-alpha) * cosine, and tune alpha. You will need held-out queries to do it, which is the cost RRF avoids."}),
        code('''
def rrf(q, k=60):
    s = np.zeros(N)
    for order in (bm_order[q], dense_order[q]):
        s[order] += 1.0 / (k + 1 + np.arange(N))
    return s

hy_order, rows = {}, []
for q in qids:
    o = np.argsort(-rrf(q))
    hy_order[q] = o
    rows.append(rank_metrics([docs_id[i] for i in o], rel[q]))
hybrid = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}

print("           BM25     dense    hybrid")
for k in ("ndcg@10", "p@1", "r@10", "r@100"):
    print(f"  {k:<8} {bm25[k]:.4f}   {dense[k]:.4f}   {hybrid[k]:.4f}")
'''),
        md("""
Better than either, on every metric, from adding two reciprocals. Which is not
mysterious: the previous cell counted the relevant abstracts each system found
alone, and fusion is how you keep both.
"""),
    ]

    # ------------------------------------------------------------------ 5
    cells += [
        md("""
## 5 · Cross-encoders, and why they go second

A cross-encoder puts the query and the document through the model **together**,
so every query token can attend to every document token. That is what a
bi-encoder forbids by construction — and it is why nothing can be precomputed:
the document's representation does not exist until the query arrives.
"""),
        prompt(
            label="the cost, before the accuracy",
            input="the corpus size and the number of claims",
            output="the number of transformer passes a full cross-encoder scan needs",
            constraint="state it before running anything — the argument for the pipeline is arithmetic, and it should be made before the measurement rather than after",
            check="compare it against the one pass a bi-encoder needs. The ratio is the reason production systems have two stages.",
            **{"try": "redo the three lines for a corpus of 10 million abstracts "
                      "and a thousand queries a second. The bi-encoder line is "
                      "still one pass per query and the cross-encoder line has "
                      "become impossible. The two-stage pipeline is not an "
                      "optimisation; it is the only shape that fits."}),
        code('''
print(f"full cross-encoder scan   {N * len(qids):,} transformer passes")
print(f"re-rank BM25's top 100    {100 * len(qids):,}")
print(f"bi-encoder                {len(qids):,} (one per claim)")
print()
print(f"and this corpus is {N:,} documents. A web index is 10 billion.")
'''),
        prompt(
            label="⏱ 3 min — re-rank the top k",
            input="BM25's top 10, 50 and 100 candidates per claim",
            output="NDCG@10 and P@1 at each depth",
            constraint="keep everything below the re-ranked prefix in its original order, or recall@100 changes and you are measuring two things at once",
            check="assert recall@100 is unchanged by re-ranking at depth 100. A re-ranker reorders a candidate set; it cannot retrieve, and if that number moved, the splice is wrong.",
            **{"try": "re-rank the HYBRID top 100 instead. The ceiling argument in the next section predicts what happens to recall; check whether NDCG follows."}),
        code('''
from sentence_transformers import CrossEncoder

ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=384)

rerank = {}
for depth in (10, 50, 100):
    rows = []
    for q in qids:
        cand = [int(i) for i in bm_order[q][:depth]]
        sc = ce.predict([(qtext[q], docs_txt[i]) for i in cand],
                        batch_size=64, show_progress_bar=False)
        head = [cand[j] for j in np.argsort(-np.asarray(sc))]
        tail = [int(i) for i in bm_order[q][depth:]]
        rows.append(rank_metrics([docs_id[i] for i in head + tail], rel[q]))
    rerank[depth] = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
    print(f"  depth {depth:>3}   ndcg@10 {rerank[depth]['ndcg@10']:.4f}   "
          f"p@1 {rerank[depth]['p@1']:.4f}")

assert abs(rerank[100]["r@100"] - bm25["r@100"]) < 1e-9, \\
    "re-ranking changed recall@100 — the candidate splice is wrong"
'''),
        prompt(
            label="the ceiling the re-ranker cannot pass",
            input="BM25's recall at several depths",
            output="the bound on any re-ranker of that depth",
            constraint="state it as a bound, not a score — this is the number that decides whether a better second stage is worth building at all",
            check="compare against the hybrid first stage's recall@100. Improving the first stage raises this ceiling; improving the second stage never does.",
            **{"try": "compute the same recall curve for the hybrid ranking "
                      "and put the two columns side by side. The gap at depth "
                      "100 is the extra ceiling fusion buys the re-ranker, "
                      "and it is the argument for spending your next week on "
                      "the first stage rather than the second."}),
        code('''
print("depth   BM25 recall@depth")
for d in (10, 50, 100, 1000):
    v = float(np.mean([len({docs_id[i] for i in bm_order[q][:d]} & rel[q]) / len(rel[q])
                       for q in qids]))
    print(f"{d:>5}   {v:.4f}")
print()
print(f"hybrid recall@100        {hybrid['r@100']:.4f}")
print("A perfect re-ranker over BM25's top 100 cannot exceed BM25's recall@100.")
'''),
    ]

    # ------------------------------------------------------------------ 6
    cells += [
        md("""
## 6 · The approximate index, and what it costs

The dense scan touches every vector for every query and, unlike the inverted
index, cannot skip anything. The standard answer is to cluster the vectors once
and probe only the nearest few clusters at query time.

Its recall is measured **against the exact scan**, never against the relevance
judgements — otherwise "my index missed it" and "my model ranked it low" become
one number with two different fixes.
"""),
        prompt(
            label="k-means, written out",
            input="the document vectors",
            output="64 unit-norm centres and an assignment",
            constraint="write it rather than importing it — sklearn and torch each ship an OpenMP runtime and with both loaded this deadlocks on some machines, and the clustering is the simple part of an IVF index anyway",
            check="assert every document is assigned and no cluster is empty. An empty cluster silently reduces the number of probes.",
            **{"try": "initialise the centres at the first 64 documents "
                      "rather than a random sample. The corpus is stored in "
                      "some order, and if that order is topical the clusters "
                      "start out correlated with it. Do the printed cluster "
                      "sizes become more even or less?"}),
        code('''
def kmeans(X, k=64, iters=25, seed=RANDOM_STATE):
    g = np.random.default_rng(seed)
    C = X[g.choice(len(X), k, replace=False)].copy()
    for _ in range(iters):
        a = np.argmax(X @ C.T, axis=1)          # vectors are unit-norm
        for c in range(k):
            m = a == c
            if m.any():
                v = X[m].sum(0)
                C[c] = v / (np.linalg.norm(v) + 1e-12)
    return C, np.argmax(X @ C.T, axis=1)

centres, assign = kmeans(D, k=64)
buckets = [np.where(assign == c)[0] for c in range(64)]
sizes = np.array([len(b) for b in buckets])
assert sizes.sum() == N and sizes.min() > 0
print(f"64 clusters, sizes {sizes.min()} to {sizes.max()}, mean {sizes.mean():.0f}")
'''),
        prompt(
            label="the recall/latency curve",
            input="the number of clusters probed",
            output="recall of the exact top ten, and the fraction of the corpus scanned",
            constraint="measure recall against the EXACT dense ranking computed above, not against the judgements",
            check="at nprobe = 64 the recall must be exactly 1.0 — probing every cluster is the exact scan. If it is not, the index is losing documents and the bug is in the bucketing.",
            **{"try": "raise the cluster count to 256 and re-run. More, smaller clusters means a finer dial and a worse recall at the same nprobe."}),
        code('''
print("nprobe   scanned   recall@10 vs exact")
for nprobe in (1, 2, 4, 8, 16, 64):
    recalls, scanned = [], []
    for j, q in enumerate(qids):
        cs = np.argsort(-(centres @ Q[j]))[:nprobe]
        cand = np.concatenate([buckets[c] for c in cs])
        scanned.append(len(cand) / N)
        top = cand[np.argsort(-(D[cand] @ Q[j]))[:10]]
        exact = set(dense_order[q][:10].tolist())
        recalls.append(len(set(top.tolist()) & exact) / 10)
    print(f"{nprobe:>6}   {100*np.mean(scanned):>6.1f}%   {np.mean(recalls):.4f}")
'''),
    ]

    # ------------------------------------------------------------------ 7
    cells += [
        md("""
## 7 · Everything, in one table

Six systems, one corpus, one metric named before any of them was run.
"""),
        prompt(
            label="the summary table",
            input="every result computed above",
            output="NDCG@10 and recall@100 per system",
            constraint="one row per system and no row omitted, including the ones that lost",
            check="the best NDCG@10 should belong to a row with no training in it. If a table like this ever omits BM25, that is the first question to ask of it.",
            **{"try": "add Lecture 19's unranked-corpus floor as the first "
                      "row. Every number in this table is a distance from it, "
                      "and a summary table whose worst row is still a working "
                      "system says nothing about how hard the task actually "
                      "is."}),
        code('''
table = [
    ("BM25",                     bm25["ndcg@10"],       bm25["r@100"]),
    ("dense, zero shot",         dense["ndcg@10"],      dense["r@100"]),
    ("BM25 + cross-encoder@10",  rerank[10]["ndcg@10"], rerank[10]["r@100"]),
    ("BM25 + cross-encoder@100", rerank[100]["ndcg@10"],rerank[100]["r@100"]),
    ("hybrid (RRF)",             hybrid["ndcg@10"],     hybrid["r@100"]),
]
print(f"{'system':<26} {'ndcg@10':>8} {'r@100':>8}")
for name, nd, rc in table:
    print(f"{name:<26} {nd:>8.4f} {rc:>8.4f}")
print()
best = max(table, key=lambda t: t[1])
print(f"best ndcg@10: {best[0]}")
'''),
        md("""
## 8 · Five questions for a neural retrieval claim

1. **Is BM25 in the table**, on the same corpus, queries and judgements?
2. **Zero shot or fine-tuned** — and if fine-tuned, on what, and how close is it
   to the test corpus?
3. **Exact scan or approximate index**, and at what recall against the exact
   scan?
4. **Where did the negatives come from**, and what fraction are unjudged rather
   than judged irrelevant?
5. **First stage or whole pipeline?** A re-ranker's NDCG is a property of the
   candidates it was given.

---

**Next.** Lecture 21 removes the query. There is no query in recommendation —
only a user, and what they did.
"""),
    ]

    return cells
