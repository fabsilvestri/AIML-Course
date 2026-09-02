#!/usr/bin/env python3
"""
Figures and facts for Lecture 20 — dense retrieval, SciFact.

    python3 tools/figures_dense.py

Writes l20_* keys into assets/figures/figures.json, MERGING rather than
overwriting. Collisions raise.

Reuses the loader, tokeniser, BM25 and metrics of figures_ir.py, so that the
lexical baseline in this file is bit-for-bit the one Lecture 19 reports. A
comparison against a re-implemented baseline is not a comparison.

Everything is scored EXACTLY: 5,183 abstracts is small enough for a full dense
scan, so the bi-encoder numbers are properties of the model rather than of an
approximate index. The approximate index is then measured separately and
against that exact answer, which is the only way the recall/latency trade-off
means anything.

CPU, a few minutes, most of it encoding the corpus once.
"""
from __future__ import annotations

import json, math, os, sys, time
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "4")

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from figures_ir import BM25, load, rank_metrics, tok            # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "assets" / "figures"
SEED = 42
BI = "sentence-transformers/all-MiniLM-L6-v2"
CROSS = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def encode(model, texts, batch=64):
    return model.encode(texts, batch_size=batch, convert_to_numpy=True,
                        normalize_embeddings=True, show_progress_bar=False)


def main() -> int:
    from collections import defaultdict
    from sentence_transformers import SentenceTransformer, CrossEncoder

    print("Lecture 20: dense retrieval, SciFact")
    corpus, queries, qrels = load()
    facts: dict = {}

    docs_id = corpus["_id"].astype(str).tolist()
    docs_txt = [f"{t} {x}" for t, x in zip(corpus["title"], corpus["text"])]
    toks = [tok(d) for d in docs_txt]

    rel = defaultdict(set)
    for _, r in qrels.iterrows():
        if int(r["score"]) > 0:
            rel[str(r["query-id"])].add(str(r["corpus-id"]))
    qtext = dict(zip(queries["_id"].astype(str), queries["text"]))
    qids = [q for q in rel if q in qtext]
    N = len(docs_id)

    # --- the lexical baseline, from Lecture 19's own code -------------------
    bm = BM25(toks)
    bm_order = {}
    rows = []
    for q in qids:
        order = np.argsort(-bm.scores(tok(qtext[q])))
        bm_order[q] = order
        rows.append(rank_metrics([docs_id[i] for i in order], rel[q]))
    facts["l20_bm25"] = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
    print("    BM25 ndcg@10 {ndcg@10:.4f}".format(**facts["l20_bm25"]))

    # --- the bi-encoder, zero-shot -----------------------------------------
    print(f"    encoding {N:,} abstracts with {BI} …", flush=True)
    bi = SentenceTransformer(BI)
    t0 = time.perf_counter()
    D = encode(bi, docs_txt)
    corpus_encode_s = time.perf_counter() - t0
    Q = encode(bi, [qtext[q] for q in qids])
    facts["l20_dim"] = int(D.shape[1])
    facts["l20_model"] = BI.split("/")[-1]
    facts["l20_index_bytes"] = int(D.size * 4)
    facts["l20_postings"] = int(sum(len(p) for p in bm.post.values()))

    dense_order, rows = {}, []
    for j, q in enumerate(qids):
        order = np.argsort(-(D @ Q[j]))
        dense_order[q] = order
        rows.append(rank_metrics([docs_id[i] for i in order], rel[q]))
    facts["l20_dense"] = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
    print("    dense ndcg@10 {ndcg@10:.4f}".format(**facts["l20_dense"]))

    # --- what each method finds that the other does not --------------------
    only_bm, only_de, both, neither = 0, 0, 0, 0
    for q in qids:
        b = {docs_id[i] for i in bm_order[q][:10]} & rel[q]
        d = {docs_id[i] for i in dense_order[q][:10]} & rel[q]
        for g in rel[q]:
            if g in b and g in d:
                both += 1
            elif g in b:
                only_bm += 1
            elif g in d:
                only_de += 1
            else:
                neither += 1
    tot = both + only_bm + only_de + neither
    facts["l20_overlap"] = {
        "both": both, "only_bm25": only_bm, "only_dense": only_de,
        "neither": neither, "total": tot,
        "frac_only_dense": only_de / tot, "frac_only_bm25": only_bm / tot,
        "frac_neither": neither / tot,
    }
    print(f"    of {tot} relevant docs: both {both}, BM25 only {only_bm}, "
          f"dense only {only_de}, neither {neither}")

    # --- the Lecture 19 failure case, re-tried -----------------------------
    worst = None
    for q in qids:
        if len(rel[q]) != 1:
            continue
        r = [docs_id[i] for i in bm_order[q]].index(next(iter(rel[q]))) + 1
        if worst is None or r > worst[1]:
            worst = (q, r)
    wq, wrank = worst
    d_rank = [docs_id[i] for i in dense_order[wq]].index(next(iter(rel[wq]))) + 1
    facts["l20_mismatch_fixed"] = {
        "claim": qtext[wq], "bm25_rank": int(wrank), "dense_rank": int(d_rank),
    }
    print(f"    the mismatch case: BM25 rank {wrank:,} -> dense rank {d_rank:,}")

    # --- hybrid: reciprocal rank fusion ------------------------------------
    def rrf(q, k=60):
        s = np.zeros(N)
        for order in (bm_order[q], dense_order[q]):
            s[order] += 1.0 / (k + 1 + np.arange(N))
        return s
    rows = [rank_metrics([docs_id[i] for i in np.argsort(-rrf(q))], rel[q]) for q in qids]
    facts["l20_hybrid"] = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
    print("    hybrid (RRF) ndcg@10 {ndcg@10:.4f}".format(**facts["l20_hybrid"]))

    # --- cross-encoder re-ranking ------------------------------------------
    # Re-ranks BM25's top 100. The depth matters and is reported: a re-ranker
    # cannot recover a document the first stage never returned, which is the
    # whole argument for why the accurate model goes second.
    print(f"    re-ranking with {CROSS} …", flush=True)
    ce = CrossEncoder(CROSS, max_length=384)
    for depth in (10, 50, 100):
        rows = []
        t0 = time.perf_counter()
        for q in qids:
            cand = [int(i) for i in bm_order[q][:depth]]
            sc = ce.predict([(qtext[q], docs_txt[i]) for i in cand],
                            batch_size=64, show_progress_bar=False)
            order = [cand[j] for j in np.argsort(-np.asarray(sc))]
            rows.append(rank_metrics([docs_id[i] for i in order] +
                                     [docs_id[i] for i in bm_order[q][depth:]], rel[q]))
        key = f"l20_rerank_{depth}"
        facts[key] = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
        facts[key]["pairs_scored"] = int(depth * len(qids))
        print(f"      depth {depth:>3}  ndcg@10 {facts[key]['ndcg@10']:.4f}  "
              f"({time.perf_counter()-t0:.0f} s)")

    # the ceiling a re-ranker of a given depth cannot pass
    facts["l20_first_stage_ceiling"] = [
        {"depth": d,
         "bm25_recall": float(np.mean([
             len({docs_id[i] for i in bm_order[q][:d]} & rel[q]) / len(rel[q])
             for q in qids]))}
        for d in (10, 50, 100, 1000)
    ]

    # --- the approximate index, measured against the exact answer ----------
    print("    building an IVF index by hand …", flush=True)
    # k-means in twenty lines rather than sklearn's: PyTorch and sklearn each
    # ship their own OpenMP runtime and with both loaded this deadlocks on
    # macOS. Writing it out also keeps the notebook free of the conflict, and
    # the whole point of an IVF index is that the clustering is the simple part.
    def kmeans(X, k=64, iters=25, seed=SEED):
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
    ann = []
    for nprobe in (1, 2, 4, 8, 16, 64):
        recalls, scanned = [], []
        for j, q in enumerate(qids):
            cs = np.argsort(-(centres @ Q[j]))[:nprobe]
            cand = np.concatenate([buckets[c] for c in cs]) if len(cs) else np.array([], int)
            scanned.append(len(cand) / N)
            if len(cand) == 0:
                recalls.append(0.0)
                continue
            top = cand[np.argsort(-(D[cand] @ Q[j]))[:10]]
            exact = set(dense_order[q][:10].tolist())
            recalls.append(len(set(top.tolist()) & exact) / 10)
        ann.append({"nprobe": nprobe,
                    "recall_at_10_vs_exact": float(np.mean(recalls)),
                    "frac_corpus_scanned": float(np.mean(scanned))})
        print(f"      nprobe {nprobe:>2}  recall {ann[-1]['recall_at_10_vs_exact']:.4f}  "
              f"scanned {100*ann[-1]['frac_corpus_scanned']:.1f}%")
    facts["l20_ann"] = ann

    # --- in-batch negatives: what a batch actually supplies ----------------
    facts["l20_in_batch"] = [
        {"batch": b, "negatives_per_query": b - 1, "pairs_encoded": 2 * b,
         "pairs_scored": b * b} for b in (8, 32, 128, 512)
    ]

    # --- cost, as ratios rather than seconds (AUTHORING 3.2a) --------------
    facts["l20_cost"] = {
        "corpus_encode_one_sig_fig_s": float(f"{corpus_encode_s:.0g}"),
        "bytes_per_doc_dense": int(D.shape[1] * 4),
        "bytes_per_doc_postings": int(4 * facts["l20_postings"] / N),
        "cross_encoder_pairs_full_scan": int(N * len(qids)),
    }

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
