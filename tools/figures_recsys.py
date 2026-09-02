#!/usr/bin/env python3
"""
Figures and facts for Lectures 21 and 22 — recommender systems, MovieLens 1M.

    python3 tools/figures_recsys.py

Writes rec21_* and rec22_* keys into assets/figures/figures.json, MERGING
rather than overwriting. Collisions raise.

The prefix is "rec21", not "l21", because the l21_/l22_ names are already taken
by Lectures 17 and 18 -- they were written when those lectures carried those
numbers, and the facts did not move when the lectures were renumbered. Renaming
several hundred keys to repair that is a change with real risk and no benefit to
a student, so the new lectures take a prefix that cannot collide with any
numbering, past or future.

MovieLens 1M: 1,000,209 ratings, 6,040 users, 3,706 films, with timestamps —
which is why this and not the smaller release. Half of Lecture 22 is about the
difference between splitting a rating history at random and splitting it in
time, and without timestamps that comparison cannot be made at all.

Everything is exact and everything is on CPU: the factorisations are full-batch
numpy, the ranking evaluations score every film in the catalogue for every test
user rather than a sample. That last choice is not for accuracy, it is the
subject of Lecture 22 — the sampled protocol is measured here too, against the
exact answer, so the bias can be reported rather than asserted.
"""
from __future__ import annotations

import json, ssl, sys, urllib.request, zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "assets" / "figures"
CACHE = ROOT / "datasets" / "movielens"
URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
SEED = 42
RANKS = (1, 2, 4, 8, 16, 32, 64)


def load() -> pd.DataFrame:
    CACHE.mkdir(parents=True, exist_ok=True)
    z = CACHE / "ml-1m.zip"
    if not z.is_file():
        print("    downloading MovieLens 1M …", flush=True)
        # This machine's certificate store is incomplete; Colab's is not. The
        # notebooks use plain urlopen, and only this build script needs the escape.
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        req = urllib.request.Request(URL, headers={"User-Agent": "curl/8"})
        z.write_bytes(urllib.request.urlopen(req, context=ctx, timeout=180).read())
    if not (CACHE / "ml-1m" / "ratings.dat").is_file():
        with zipfile.ZipFile(z) as f:
            f.extractall(CACHE)
    r = pd.read_csv(CACHE / "ml-1m" / "ratings.dat", sep="::", engine="python",
                    names=["user", "item", "rating", "ts"], encoding="latin-1")
    r["u"] = pd.factorize(r["user"])[0]
    r["i"] = pd.factorize(r["item"])[0]
    return r


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def group(idx, n):
    """Row-grouped view: order[start[k]:start[k+1]] are the entries with idx==k.

    One argsort rather than n boolean masks over a million ratings, which is the
    difference between a few seconds and a few minutes.
    """
    order = np.argsort(idx, kind="stable")
    counts = np.bincount(idx, minlength=n)
    start = np.concatenate([[0], np.cumsum(counts)])
    return order, start


def als(u, i, r, n_u, n_i, d=32, iters=12, reg=10.0, seed=SEED, biases=True):
    """Alternating least squares on the OBSERVED entries only.

    Holding one factor matrix fixed makes the objective quadratic in the other,
    so each half-step is a ridge regression per user (or per item) with a closed
    form. That is the whole appeal, and it is why the missing entries cost
    nothing here: a user's system is built only from the films that user rated.

    Biases are folded in the standard way -- fitted first, then the factors are
    fitted to what the biases left behind.
    """
    rng = np.random.default_rng(seed)
    mu = float(r.mean())
    bu = np.zeros(n_u)
    bi = np.zeros(n_i)
    cu = np.bincount(u, minlength=n_u)
    ci = np.bincount(i, minlength=n_i)
    if biases:
        for _ in range(15):
            bu = np.bincount(u, weights=r - mu - bi[i], minlength=n_u) / (cu + reg)
            bi = np.bincount(i, weights=r - mu - bu[u], minlength=n_i) / (ci + reg)
    resid = r - mu - (bu[u] + bi[i] if biases else 0.0)

    P = rng.normal(0, 0.1, (n_u, d))
    Q = rng.normal(0, 0.1, (n_i, d))
    ou, su = group(u, n_u)
    oi, si = group(i, n_i)
    I = np.eye(d)

    def half(F, G, other_idx, order, start, n):
        """Solve the ridge system for every row of F, holding G fixed."""
        for k in range(n):
            sl = order[start[k]:start[k + 1]]
            if sl.size == 0:
                F[k] = 0.0
                continue
            Gk = G[other_idx[sl]]
            F[k] = np.linalg.solve(Gk.T @ Gk + reg * I, Gk.T @ resid[sl])

    for _ in range(iters):
        half(P, Q, i, ou, su, n_u)
        half(Q, P, u, oi, si, n_i)
    return P, Q, mu, bu, bi


def sgd(u, i, r, n_u, n_i, d=32, epochs=15, lr=0.01, reg=0.05, seed=SEED):
    """Per-rating stochastic gradient descent, for the comparison with ALS.

    Shuffled once per epoch and applied one rating at a time, which is the form
    the lecture writes down. It is slower per epoch than ALS and reaches a
    comparable place; the point of having both is that they optimise the same
    objective by different routes.
    """
    rng = np.random.default_rng(seed)
    mu = float(r.mean())
    bu = np.zeros(n_u)
    bi = np.zeros(n_i)
    P = rng.normal(0, 0.1, (n_u, d))
    Q = rng.normal(0, 0.1, (n_i, d))
    hist = []
    for ep in range(epochs):
        for n in rng.permutation(len(r)):
            uu, ii = u[n], i[n]
            e = r[n] - (mu + bu[uu] + bi[ii] + P[uu] @ Q[ii])
            bu[uu] += lr * (e - reg * bu[uu])
            bi[ii] += lr * (e - reg * bi[ii])
            pu = P[uu].copy()
            P[uu] += lr * (e * Q[ii] - reg * pu)
            Q[ii] += lr * (e * pu - reg * Q[ii])
        hist.append(float(np.sqrt(np.mean(
            (r - np.clip(mu + bu[u] + bi[i] + (P[u] * Q[i]).sum(1), 1, 5)) ** 2))))
    return P, Q, mu, bu, bi, hist


def predict(P, Q, mu, bu, bi, u, i, biases=True):
    p = (P[u] * Q[i]).sum(1) + mu
    if biases:
        p = p + bu[u] + bi[i]
    return np.clip(p, 1.0, 5.0)


def main() -> int:
    print("Lectures 21-22: recommender systems, MovieLens 1M")
    r = load()
    facts: dict = {}
    n_u, n_i = r["u"].nunique(), r["i"].nunique()

    facts["rec21_n_ratings"] = int(len(r))
    facts["rec21_n_users"] = int(n_u)
    facts["rec21_n_items"] = int(n_i)
    facts["rec21_density"] = float(len(r) / (n_u * n_i))
    facts["rec21_rating_mean"] = float(r["rating"].mean())
    facts["rec21_rating_dist"] = [
        {"rating": int(k), "count": int(v), "frac": float(v / len(r))}
        for k, v in sorted(r["rating"].value_counts().items())]
    facts["rec21_ratings_per_user_median"] = float(r.groupby("u").size().median())
    facts["rec21_ratings_per_item_median"] = float(r.groupby("i").size().median())
    print(f"    {len(r):,} ratings, {n_u:,} users, {n_i:,} films, "
          f"density {100*facts['rec21_density']:.2f}%")

    # the long tail, which decides most of what Lecture 22 is about
    pop = r.groupby("i").size().sort_values(ascending=False).values
    facts["rec21_long_tail"] = [
        {"top_frac": f, "share_of_ratings":
         float(pop[:max(1, int(f * n_i))].sum() / len(r))}
        for f in (0.01, 0.05, 0.10, 0.20)]
    facts["rec21_items_under_20"] = float(np.mean(pop < 20))

    # --- a split, held fixed for every model in Lecture 21 -----------------
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(r))
    n_test = int(0.1 * len(r))
    te, tr = perm[:n_test], perm[n_test:]
    u_tr, i_tr, y_tr = r["u"].values[tr], r["i"].values[tr], r["rating"].values[tr].astype(float)
    u_te, i_te, y_te = r["u"].values[te], r["i"].values[te], r["rating"].values[te].astype(float)
    facts["rec21_n_train"] = int(len(tr))
    facts["rec21_n_test"] = int(len(te))

    mu = float(y_tr.mean())
    # user and item means, computed on the training half only
    su = np.bincount(u_tr, weights=y_tr, minlength=n_u)
    cu = np.bincount(u_tr, minlength=n_u)
    si = np.bincount(i_tr, weights=y_tr, minlength=n_i)
    ci = np.bincount(i_tr, minlength=n_i)
    user_mean = np.where(cu > 0, su / np.maximum(cu, 1), mu)
    item_mean = np.where(ci > 0, si / np.maximum(ci, 1), mu)

    baselines = {
        "global mean": rmse(y_te, np.full(len(y_te), mu)),
        "user mean": rmse(y_te, user_mean[u_te]),
        "item mean": rmse(y_te, item_mean[i_te]),
    }
    # the bias model: mu + b_u + b_i, fitted by alternating closed form
    bu = np.zeros(n_u)
    bi = np.zeros(n_i)
    LAM = 10.0
    for _ in range(15):
        resid = y_tr - mu - bi[i_tr]
        bu = np.bincount(u_tr, weights=resid, minlength=n_u) / (cu + LAM)
        resid = y_tr - mu - bu[u_tr]
        bi = np.bincount(i_tr, weights=resid, minlength=n_i) / (ci + LAM)
    baselines["bias model"] = rmse(y_te, np.clip(mu + bu[u_te] + bi[i_te], 1, 5))
    facts["rec21_baselines"] = baselines
    for k, v in baselines.items():
        print(f"      {k:<14} rmse {v:.4f}")

    facts["rec21_bias_extremes"] = {
        "user_bias_min": float(bu.min()), "user_bias_max": float(bu.max()),
        "item_bias_min": float(bi.min()), "item_bias_max": float(bi.max()),
    }

    # --- why the SVD cannot simply be taken --------------------------------
    # Fill the missing entries, take a truncated SVD, and price the fill.
    from numpy.linalg import svd
    M_zero = np.zeros((n_u, n_i))
    M_zero[u_tr, i_tr] = y_tr
    M_mean = np.full((n_u, n_i), mu)
    M_mean[u_tr, i_tr] = y_tr
    fills = {}
    for name, M in (("filled with zeros", M_zero), ("filled with the mean", M_mean)):
        U, S, Vt = svd(M, full_matrices=False)
        k = 32
        R = (U[:, :k] * S[:k]) @ Vt[:k]
        fills[name] = rmse(y_te, np.clip(R[u_te, i_te], 1, 5))
        print(f"      SVD rank 32, {name:<22} rmse {fills[name]:.4f}")
    facts["rec21_svd_fill"] = fills

    # --- matrix factorisation on the observed entries only -----------------
    sweep = []
    for d in RANKS:
        P, Q, m2, b2u, b2i = als(u_tr, i_tr, y_tr, n_u, n_i, d=d)
        sweep.append({"rank": d,
                      "rmse": rmse(y_te, predict(P, Q, m2, b2u, b2i, u_te, i_te))})
        print(f"      MF rank {d:>2}  rmse {sweep[-1]['rmse']:.4f}")
    facts["rec21_mf_rank_sweep"] = sweep
    # ALS and SGD on the same objective, to show they arrive at the same place
    Ps, Qs, ms, bus, bis, hist = sgd(u_tr, i_tr, y_tr, n_u, n_i, d=32)
    facts["rec21_sgd_curve"] = [{"epoch": e + 1, "train_rmse": v}
                                for e, v in enumerate(hist)]
    facts["rec21_sgd_test_rmse"] = rmse(
        y_te, predict(Ps, Qs, ms, bus, bis, u_te, i_te))
    print(f"      SGD rank 32  test rmse {facts['rec21_sgd_test_rmse']:.4f}")
    best = min(sweep, key=lambda s: s["rmse"])
    facts["rec21_mf_best"] = best

    # how much of that is the biases rather than the factors
    P, Q, m2, b2u, b2i = als(u_tr, i_tr, y_tr, n_u, n_i, d=best["rank"],
                             biases=False)
    facts["rec21_mf_no_bias"] = rmse(y_te, predict(P, Q, m2, b2u, b2i, u_te, i_te,
                                                 biases=False))
    print(f"      MF rank {best['rank']} without biases  rmse {facts['rec21_mf_no_bias']:.4f}")

    # --- item-item neighbourhood -------------------------------------------
    print("    item-item cosine neighbourhood …", flush=True)
    R_tr = np.zeros((n_u, n_i), dtype=np.float32)
    R_tr[u_tr, i_tr] = (y_tr - mu - bu[u_tr] - bi[i_tr]).astype(np.float32)
    norms = np.linalg.norm(R_tr, axis=0) + 1e-9
    Rn = R_tr / norms
    S_ii = Rn.T @ Rn
    np.fill_diagonal(S_ii, 0.0)
    knn = []
    for k in (10, 20, 50):
        idx = np.argsort(-S_ii, axis=1)[:, :k]
        preds = np.empty(len(y_te))
        for n, (uu, ii) in enumerate(zip(u_te, i_te)):
            nb = idx[ii]
            w = S_ii[ii, nb]
            v = R_tr[uu, nb]
            m = (v != 0) & (w > 0)
            preds[n] = (mu + bu[uu] + bi[ii] +
                        (float(w[m] @ v[m]) / float(w[m].sum()) if m.any() else 0.0))
        knn.append({"k": k, "rmse": rmse(y_te, np.clip(preds, 1, 5))})
        print(f"      item-item k={k:<3} rmse {knn[-1]['rmse']:.4f}")
    facts["rec21_item_knn"] = knn

    # ================= Lecture 22: ranking, and how it is measured =========
    print("    Lecture 22: implicit feedback and the evaluation protocols")
    imp = r[r["rating"] >= 4].copy()
    facts["rec22_n_positive"] = int(len(imp))
    facts["rec22_frac_positive"] = float(len(imp) / len(r))

    # two protocols on the same data: leave-one-out at random, and in time
    imp = imp.sort_values("ts")
    last_time = imp.groupby("u").tail(1)
    rng2 = np.random.default_rng(SEED)
    rand_pick = (imp.groupby("u")
                 .apply(lambda g: g.iloc[rng2.integers(len(g))], include_groups=False)
                 .reset_index())

    def build(holdout_idx):
        mask = np.ones(len(imp), dtype=bool)
        pos = {(a, b) for a, b in zip(holdout_idx["u"], holdout_idx["i"])}
        arr_u, arr_i = imp["u"].values, imp["i"].values
        for n in range(len(imp)):
            if (arr_u[n], arr_i[n]) in pos:
                mask[n] = False
                pos.discard((arr_u[n], arr_i[n]))
        return imp[mask], imp[~mask]

    protocols = {}
    for name, hold in (("temporal (leave last)", last_time),
                       ("random (leave one)", rand_pick)):
        train, test = build(hold[["u", "i"]])
        Rb = np.zeros((n_u, n_i), dtype=np.float32)
        Rb[train["u"].values, train["i"].values] = 1.0
        # a two-tower factorisation of the implicit matrix, by truncated SVD of
        # the binary matrix -- the simplest thing that gives user and item vectors
        U, S, Vt = svd(Rb, full_matrices=False)
        k = 32
        Pu = U[:, :k] * S[:k]
        Qi = Vt[:k].T
        popularity = Rb.sum(0)

        def eval_full(scorer):
            hr, nd = [], []
            for uu, ii in zip(test["u"].values, test["i"].values):
                s = scorer(uu).copy()
                s[Rb[uu] > 0] = -np.inf           # never re-recommend a seen film
                top = np.argpartition(-s, 10)[:10]
                top = top[np.argsort(-s[top])]
                hit = int(ii in top)
                hr.append(hit)
                nd.append(1.0 / np.log2(list(top).index(ii) + 2) if hit else 0.0)
            return float(np.mean(hr)), float(np.mean(nd))

        def eval_sampled(scorer, n_neg=100, seed=SEED):
            g = np.random.default_rng(seed)
            hr, nd = [], []
            for uu, ii in zip(test["u"].values, test["i"].values):
                s = scorer(uu)
                seen = Rb[uu] > 0
                cand = g.choice(n_i, size=n_neg * 3, replace=False)
                cand = [c for c in cand if not seen[c] and c != ii][:n_neg]
                pool = np.array([ii] + cand)
                rank = int(np.argsort(-s[pool]).tolist().index(0)) + 1
                hr.append(int(rank <= 10))
                nd.append(1.0 / np.log2(rank + 1) if rank <= 10 else 0.0)
            return float(np.mean(hr)), float(np.mean(nd))

        mf_score = lambda uu: Pu[uu] @ Qi.T
        pop_score = lambda uu: popularity
        rec = {}
        for mname, sc in (("popularity", pop_score), ("factorisation", mf_score)):
            hr_f, nd_f = eval_full(sc)
            hr_s, nd_s = eval_sampled(sc)
            rec[mname] = {"hr@10_full": hr_f, "ndcg@10_full": nd_f,
                          "hr@10_sampled100": hr_s, "ndcg@10_sampled100": nd_s}
            print(f"      {name:<22} {mname:<14} "
                  f"HR@10 full {hr_f:.4f}  sampled {hr_s:.4f}")
        protocols[name] = rec
    facts["rec22_protocols"] = protocols

    # what a batch of B supplies as in-batch negatives
    facts["rec22_in_batch"] = [
        {"batch": b, "negatives_per_user": b - 1, "encodes": 2 * b, "dots": b * b}
        for b in (8, 32, 128, 512, 2048)]

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
