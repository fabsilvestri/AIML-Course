#!/usr/bin/env python3
"""
Figures and facts for Lectures 19 and 20 — information retrieval, SciFact.

    python3 tools/figures_ir.py

Writes l19_* and l20_* keys into assets/figures/figures.json, MERGING rather
than overwriting: that file is shared with every other figure script and
clobbering it silently destroys several hundred values belonging to other
lectures. Collisions raise.

Everything here runs on CPU in a couple of minutes. SciFact is 5,183 abstracts
and 1,109 claims, small enough that an exact BM25 scan and an exact dense scan
are both affordable, which is what lets the lectures compare them honestly
rather than through an approximate index.
"""
from __future__ import annotations

import io, json, math, re, ssl, sys, urllib.request
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "assets" / "figures"
CACHE = ROOT / "datasets" / "scifact"
SEED = 42
HF = "https://huggingface.co/datasets/BeIR/"


def _get(url: str) -> bytes:
    # This machine's certificate store is incomplete; Colab's is not. The
    # notebooks use plain requests, and only this build script needs the escape.
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    req = urllib.request.Request(url, headers={"User-Agent": "curl/8"})
    return urllib.request.urlopen(req, context=ctx, timeout=60).read()


def load():
    """Corpus, queries and test qrels, cached under datasets/scifact/."""
    CACHE.mkdir(parents=True, exist_ok=True)
    paths = {
        "corpus":  (CACHE / "corpus.parquet",  HF + "scifact/resolve/main/corpus/corpus-00000-of-00001.parquet"),
        "queries": (CACHE / "queries.parquet", HF + "scifact/resolve/main/queries/queries-00000-of-00001.parquet"),
        "qrels":   (CACHE / "qrels_test.tsv",  HF + "scifact-qrels/resolve/main/test.tsv"),
    }
    for name, (p, url) in paths.items():
        if not p.is_file():
            print(f"    downloading {name} …", flush=True)
            p.write_bytes(_get(url))
    corpus = pd.read_parquet(paths["corpus"][0])
    queries = pd.read_parquet(paths["queries"][0])
    qrels = pd.read_csv(paths["qrels"][0], sep="\t", dtype={"query-id": str, "corpus-id": str})
    return corpus, queries, qrels


TOKEN = re.compile(r"[a-z0-9]+")


def tok(s: str) -> list[str]:
    return TOKEN.findall(s.lower())


class BM25:
    """Exact BM25 over the whole corpus. No index library, so the lecture can
    show what an inverted index is by building one."""

    def __init__(self, docs: list[list[str]], k1: float = 0.9, b: float = 0.4):
        self.k1, self.b = k1, b
        self.N = len(docs)
        self.len = np.array([len(d) for d in docs], dtype=np.float64)
        self.avgdl = float(self.len.mean())
        self.post: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for i, d in enumerate(docs):
            for term, tf in Counter(d).items():
                self.post[term].append((i, tf))
        self.idf = {t: math.log(1 + (self.N - len(p) + 0.5) / (len(p) + 0.5))
                    for t, p in self.post.items()}

    def scores(self, q: list[str]) -> np.ndarray:
        s = np.zeros(self.N)
        for t in q:
            p = self.post.get(t)
            if not p:
                continue
            idf = self.idf[t]
            for i, tf in p:
                denom = tf + self.k1 * (1 - self.b + self.b * self.len[i] / self.avgdl)
                s[i] += idf * tf * (self.k1 + 1) / denom
        return s


def rank_metrics(ranked: list[str], rel: set[str], ks=(1, 5, 10, 100)):
    out = {}
    for k in ks:
        top = ranked[:k]
        hits = sum(1 for d in top if d in rel)
        out[f"p@{k}"] = hits / k
        out[f"r@{k}"] = hits / len(rel) if rel else 0.0
    rr = 0.0
    for i, d in enumerate(ranked, 1):
        if d in rel:
            rr = 1.0 / i
            break
    out["rr"] = rr
    dcg = sum(1.0 / math.log2(i + 1) for i, d in enumerate(ranked[:10], 1) if d in rel)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(rel), 10) + 1))
    out["ndcg@10"] = dcg / idcg if idcg else 0.0
    ap, hits = 0.0, 0
    for i, d in enumerate(ranked, 1):
        if d in rel:
            hits += 1
            ap += hits / i
    out["ap"] = ap / len(rel) if rel else 0.0
    return out


def main() -> int:
    print("Lectures 19-20: information retrieval, SciFact")
    corpus, queries, qrels = load()
    facts: dict = {}

    docs_id = corpus["_id"].tolist()
    docs_txt = [f"{t} {x}" for t, x in zip(corpus["title"], corpus["text"])]
    toks = [tok(d) for d in docs_txt]

    facts["l19_n_docs"] = int(len(docs_id))
    facts["l19_n_queries_all"] = int(len(queries))
    facts["l19_doc_len_mean"] = float(np.mean([len(t) for t in toks]))
    facts["l19_doc_len_median"] = float(np.median([len(t) for t in toks]))
    facts["l19_vocab"] = int(len({w for t in toks for w in t}))

    rel = defaultdict(set)
    for _, r in qrels.iterrows():
        if int(r["score"]) > 0:
            rel[str(r["query-id"])].add(str(r["corpus-id"]))
    qtext = dict(zip(queries["_id"].astype(str), queries["text"]))
    qids = [q for q in rel if q in qtext]
    facts["l19_n_queries_test"] = int(len(qids))
    facts["l19_rel_per_query_mean"] = float(np.mean([len(rel[q]) for q in qids]))
    facts["l19_rel_per_query_max"] = int(max(len(rel[q]) for q in qids))
    facts["l19_frac_single_rel"] = float(np.mean([len(rel[q]) == 1 for q in qids]))

    print(f"    {facts['l19_n_docs']:,} abstracts, {facts['l19_n_queries_test']} test claims")

    bm = BM25(toks)
    facts["l19_postings"] = int(sum(len(p) for p in bm.post.values()))
    facts["l19_avgdl"] = float(bm.avgdl)

    print("    scoring BM25 over every test claim …", flush=True)
    rows = []
    for q in qids:
        s = bm.scores(tok(qtext[q]))
        order = np.argsort(-s)
        rows.append(rank_metrics([docs_id[i] for i in order], rel[q]))
    bm25 = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
    facts["l19_bm25"] = bm25
    print("      ndcg@10 {ndcg@10:.4f}  r@10 {r@10:.4f}  mrr {rr:.4f}".format(**bm25))

    # the baseline any ranking has to beat: return documents in corpus order
    fixed = [rank_metrics(docs_id, rel[q]) for q in qids]
    facts["l19_fixed_order"] = {k: float(np.mean([r[k] for r in fixed])) for k in fixed[0]}

    # what each BM25 parameter is for, measured
    sweep = []
    for b in (0.0, 0.4, 0.75, 1.0):
        m = BM25(toks, k1=0.9, b=b)
        sc = [rank_metrics([docs_id[i] for i in np.argsort(-m.scores(tok(qtext[q])))], rel[q])
              for q in qids[:150]]
        sweep.append({"b": b, "ndcg@10": float(np.mean([r["ndcg@10"] for r in sc]))})
        print(f"      b={b}  ndcg@10 {sweep[-1]['ndcg@10']:.4f}")
    facts["l19_b_sweep"] = sweep

    # idf, concretely: the commonest and rarest terms actually in the corpus
    df = {t: len(p) for t, p in bm.post.items()}
    common = sorted(df.items(), key=lambda kv: -kv[1])[:5]
    facts["l19_common_terms"] = [[t, int(n), float(bm.idf[t])] for t, n in common]

    out = OUT / "figures.json"
    existing = json.loads(out.read_text()) if out.is_file() else {}
    clash = {k for k in facts if k in existing and existing[k] != facts[k]}
    if clash:
        raise SystemExit(f"figures.json collision on {sorted(clash)}")
    existing.update(facts)
    out.write_text(json.dumps(existing, indent=2))
    print(f"\n    merged {len(facts)} keys into {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
