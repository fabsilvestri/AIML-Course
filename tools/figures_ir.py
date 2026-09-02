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

    # A worked example for the derivation: one real claim, BM25's real ranking.
    # Selection rule, stated on the slide: the first test claim for which BM25
    # places at least three relevant abstracts in the top ten. It is chosen to
    # make the arithmetic visible -- a single hit gives a one-term sum and shows
    # nothing about the discount -- and it is therefore better than typical. The
    # mean over all 300 claims is the table, not this.
    def _hits10(q):
        r = [docs_id[i] for i in np.argsort(-bm.scores(tok(qtext[q])))][:10]
        return sum(1 for d in r if d in rel[q])
    ex_q = next(q for q in qids if len(rel[q]) >= 3 and _hits10(q) >= 3)
    ex_ranked = [docs_id[i] for i in np.argsort(-bm.scores(tok(qtext[ex_q])))]
    ex_pat = [1 if d in rel[ex_q] else 0 for d in ex_ranked[:10]]
    ex_ranks = [i for i, h in enumerate(ex_pat, 1) if h]
    dcg_terms = [{"rank": i, "gain": 1, "discount": math.log2(i + 1),
                  "term": 1.0 / math.log2(i + 1)} for i in ex_ranks]
    idcg_terms = [{"rank": i, "term": 1.0 / math.log2(i + 1)}
                  for i in range(1, min(len(rel[ex_q]), 10) + 1)]
    facts["l19_worked"] = {
        "claim": qtext[ex_q],
        "n_rel": len(rel[ex_q]),
        "pattern": ex_pat,
        "hit_ranks": ex_ranks,
        "rr": 1.0 / ex_ranks[0],
        "ap_terms": [{"rank": r, "prec": (j + 1) / r} for j, r in enumerate(ex_ranks)],
        "ap": sum((j + 1) / r for j, r in enumerate(ex_ranks)) / len(rel[ex_q]),
        "dcg_terms": dcg_terms,
        "dcg": sum(t["term"] for t in dcg_terms),
        "idcg_terms": idcg_terms,
        "idcg": sum(t["term"] for t in idcg_terms),
        "ndcg": sum(t["term"] for t in dcg_terms) / sum(t["term"] for t in idcg_terms),
    }
    print(f"      worked example: pattern {ex_pat}, ndcg@10 "
          f"{facts['l19_worked']['ndcg']:.4f}")

    # why order matters: same documents retrieved, different positions.
    # Both rankings hold one relevant document in the top ten.
    facts["l19_order_matters"] = [
        {"rank": r, "rr": 1.0 / r, "ndcg@10": (1.0 / math.log2(r + 1))}
        for r in (1, 2, 5, 10)
    ]

    # --- the method, taken apart -------------------------------------------
    # What a "term" is, on a real claim from the corpus.
    ex_tok_q = qids[0]
    ex_toks = tok(qtext[ex_tok_q])
    facts["l19_tok_example"] = {
        "claim": qtext[ex_tok_q],
        "tokens": ex_toks,
        "n_tokens": len(ex_toks),
        "df": {t: len(bm.post.get(t, [])) for t in dict.fromkeys(ex_toks)},
    }

    # Boolean retrieval: how many documents contain EVERY query term?
    # Run over the test claims, because the failure mode is the lesson.
    def _and_count(q):
        sets = [set(i for i, _ in bm.post.get(t, [])) for t in dict.fromkeys(tok(qtext[q]))]
        if not sets:
            return 0
        out = sets[0]
        for x in sets[1:]:
            out &= x
        return len(out)
    ands = [_and_count(q) for q in qids]
    facts["l19_boolean"] = {
        "mean_matching": float(np.mean(ands)),
        "frac_zero": float(np.mean([a == 0 for a in ands])),
        "frac_over_100": float(np.mean([a > 100 for a in ands])),
        "max": int(max(ands)),
    }

    # The k1 saturation parameter, measured the same way as b.
    k1s = []
    for k1 in (0.5, 0.9, 1.2, 2.0):
        m = BM25(toks, k1=k1, b=0.4)
        sc = [rank_metrics([docs_id[i] for i in np.argsort(-m.scores(tok(qtext[q])))], rel[q])
              for q in qids[:150]]
        k1s.append({"k1": k1, "ndcg@10": float(np.mean([r["ndcg@10"] for r in sc]))})
        print(f"      k1={k1}  ndcg@10 {k1s[-1]['ndcg@10']:.4f}")
    facts["l19_k1_sweep"] = k1s

    # How tf saturates: what the tf factor is worth at each occurrence count,
    # for a document of exactly average length.
    facts["l19_saturation"] = [
        {"tf": t, "factor": t * (0.9 + 1) / (t + 0.9)} for t in (1, 2, 3, 5, 10, 20)
    ]

    # Ablation: each correction removed in turn, on the full test set.
    def _eval(scorer):
        return float(np.mean([rank_metrics(
            [docs_id[i] for i in np.argsort(-scorer(tok(qtext[q])))], rel[q])["ndcg@10"]
            for q in qids]))

    def _raw_tf(q):
        s = np.zeros(bm.N)
        for t in q:
            for i, tf in bm.post.get(t, []):
                s[i] += tf
        return s

    def _no_idf(q):
        s = np.zeros(bm.N)
        for t in q:
            for i, tf in bm.post.get(t, []):
                d = tf + bm.k1 * (1 - bm.b + bm.b * bm.len[i] / bm.avgdl)
                s[i] += tf * (bm.k1 + 1) / d
        return s

    def _no_sat(q):  # idf-weighted tf, no saturation, no length term
        s = np.zeros(bm.N)
        for t in q:
            if t in bm.post:
                for i, tf in bm.post[t]:
                    s[i] += bm.idf[t] * tf
        return s

    bm_b0 = BM25(toks, k1=0.9, b=0.0)
    facts["l19_ablation"] = [
        {"name": "raw term frequency", "ndcg@10": _eval(_raw_tf)},
        {"name": "no idf", "ndcg@10": _eval(_no_idf)},
        {"name": "no saturation, no length", "ndcg@10": _eval(_no_sat)},
        {"name": "no length normalisation", "ndcg@10": _eval(bm_b0.scores)},
        {"name": "BM25", "ndcg@10": facts["l19_bm25"]["ndcg@10"]},
    ]
    for a in facts["l19_ablation"]:
        print(f"      {a['name']:<28} ndcg@10 {a['ndcg@10']:.4f}")

    # One query scored term by term, so the sum on the slide is a real sum.
    wq = ex_q
    wq_toks = [t for t in dict.fromkeys(tok(qtext[wq])) if t in bm.post]
    wq_scores = bm.scores(tok(qtext[wq]))
    wq_top = int(np.argmax(wq_scores))
    # Counter, not a set: scores() walks the query token list, so a repeated
    # query term contributes once per occurrence. A deduplicated table would
    # not sum to the total printed beside it.
    q_counts = Counter(tok(qtext[wq]))
    contrib = []
    for t in wq_toks:
        tf = dict(bm.post[t]).get(wq_top, 0)
        if tf:
            d = tf + bm.k1 * (1 - bm.b + bm.b * bm.len[wq_top] / bm.avgdl)
            c = q_counts[t] * bm.idf[t] * tf * (bm.k1 + 1) / d
            contrib.append({"term": t, "tf": int(tf), "idf": float(bm.idf[t]),
                            "contribution": float(c)})
    contrib.sort(key=lambda c: -c["contribution"])
    assert abs(sum(c["contribution"] for c in contrib) - wq_scores[wq_top]) < 1e-9
    facts["l19_scoring"] = {
        "claim": qtext[wq],
        "doc_len": int(bm.len[wq_top]),
        "terms": contrib,
        "total": float(wq_scores[wq_top]),
        "is_relevant": docs_id[wq_top] in rel[wq],
    }

    # The failure that motivates Lecture 20: a claim whose relevant abstract
    # BM25 buries. Reported with the term overlap, because the overlap is the
    # explanation -- the abstract says the same thing in different words.
    worst = None
    for q in qids:
        if len(rel[q]) != 1:
            continue
        order = [docs_id[i] for i in np.argsort(-bm.scores(tok(qtext[q])))]
        r = order.index(next(iter(rel[q]))) + 1
        if worst is None or r > worst[1]:
            worst = (q, r)
    wq, wrank = worst
    gold = next(iter(rel[wq]))
    gold_toks = set(toks[docs_id.index(gold)])
    q_terms = [t for t in dict.fromkeys(tok(qtext[wq])) if t in bm.post]
    shared = [t for t in q_terms if t in gold_toks]
    facts["l19_mismatch"] = {
        "claim": qtext[wq],
        "rank_of_relevant": int(wrank),
        "n_query_terms": len(q_terms),
        "n_shared": len(shared),
        "shared": shared,
        "missing": [t for t in q_terms if t not in gold_toks],
        "shared_idf_mean": float(np.mean([bm.idf[t] for t in shared])) if shared else 0.0,
        "missing_idf_mean": float(np.mean([bm.idf[t] for t in q_terms if t not in gold_toks])),
    }
    print(f"      worst single-relevant claim: rank {wrank}, "
          f"{len(shared)}/{len(q_terms)} query terms shared")

    # How often the lexical assumption is strained, over the whole test set:
    # the fraction of query terms the relevant document does not contain.
    fr = []
    for q in qids:
        qt = [t for t in dict.fromkeys(tok(qtext[q])) if t in bm.post]
        if not qt:
            continue
        for d in rel[q]:
            dt = set(toks[docs_id.index(d)])
            fr.append(sum(1 for t in qt if t not in dt) / len(qt))
    facts["l19_missing_terms"] = {
        "mean_fraction": float(np.mean(fr)),
        "frac_pairs_over_half": float(np.mean([x > 0.5 for x in fr])),
    }

    # What a document in this corpus actually looks like, and what stemming
    # would and would not merge -- both quoted rather than described.
    facts["l19_doc_example"] = {
        "title": str(corpus["title"].iloc[0]),
        "text_head": " ".join(str(corpus["text"].iloc[0]).split()[:45]),
        "n_tokens": len(toks[0]),
    }
    fam = [("cell", "cells", "cellular"), ("infect", "infects", "infection", "infected")]
    facts["l19_stemming"] = [
        {"forms": [{"term": t, "df": len(bm.post.get(t, []))} for t in group]}
        for group in fam
    ]

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
