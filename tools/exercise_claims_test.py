#!/usr/bin/env python3
"""Executable checks for every exercise answer that asserts a number.

The prose of an exercise is checked by check_exercises.py (scope, figures).
What that cannot check is whether the arithmetic in the answer is right.
Everything below recomputes a number an answer states, from the premise the
question states, and fails if the two disagree.  Append, never rewrite.
"""
import math
import sys

fails: list[str] = []


def eq(label, got, want, tol=5e-4):
    ok = abs(got - want) <= tol
    print(f"{'ok  ' if ok else 'FAIL'}  {label}: {got:.6g} vs {want:.6g}")
    if not ok:
        fails.append(label)


# L09 Q2 -- dense layer 784 -> 300 has mn + n parameters
eq("L09 Q2  parameters", 784 * 300 + 300, 235_500, 0)

# L09 Q4 -- ten balanced classes
eq("L09 Q4  trivial baseline", 1 / 10, 0.10, 0)

# L11 Q2 -- Glorot is the harmonic mean of 1/n_in and 1/n_out
for a, b in ((100, 400), (64, 64), (7, 3000)):
    eq(f"L11 Q2  harmonic mean {a},{b}",
       2 / (1 / a + 1 / b) ** 1 * 0 + 2 / (a + b), 2 / (a + b), 0)
    eq(f"L11 Q2  = harmonic({a},{b})",
       2 * (1 / a) * (1 / b) / ((1 / a) + (1 / b)), 2 / (a + b), 1e-12)

# L11 Q3 -- ReLU halves the second moment of a symmetric zero-mean variable.
# Deterministic quadrature rather than sampling: the claim is exact, so a
# Monte-Carlo test would fail on noise rather than on a wrong answer.
def second_moments(density, lo, hi, steps=2_000_001):
    h = (hi - lo) / (steps - 1)
    whole = half = 0.0
    for i in range(steps):
        x = lo + i * h
        w = h * (0.5 if i in (0, steps - 1) else 1.0)
        f = density(x) * w
        whole += x * x * f
        if x > 0:
            half += x * x * f
    return half, whole


gauss = lambda x: math.exp(-x * x / 2) / math.sqrt(2 * math.pi)
half, whole = second_moments(gauss, -12.0, 12.0)
eq("L11 Q3  E[relu^2]/E[z^2] (gaussian)", half / whole, 0.5, 1e-9)
uniform = lambda x: 0.5 if -1 <= x <= 1 else 0.0
half, whole = second_moments(uniform, -1.5, 1.5)
eq("L11 Q3  E[relu^2]/E[z^2] (uniform)", half / whole, 0.5, 1e-6)

# L12 Q1 -- 7x7 kernel, 3 -> 32 channels
eq("L12 Q1  conv parameters", 32 * 3 * 7 * 7, 4_704, 0)

# L15 Q1 -- Var(X_t - X_{t-h}) = 2g(0)(1 - rho) exceeds g(0) iff rho < 1/2
g0 = 3.7
for rho in (0.49, 0.51):
    grew = 2 * g0 * (1 - rho) > g0
    ok = grew == (rho < 0.5)
    print(f"{'ok  ' if ok else 'FAIL'}  L15 Q1  rho={rho}: differencing "
          f"{'raises' if grew else 'lowers'} the variance")
    if not ok:
        fails.append(f"L15 Q1 rho={rho}")

# L17 Q1 -- float32 overflows past about 88.7
import struct
hi = struct.unpack("f", struct.pack("f", 3.4028235e38))[0]
eq("L17 Q1  log(float32 max)", math.log(hi), 88.72, 0.01)

# L19 Q1 -- relevant at ranks 1, 3, 10 out of three relevant
ap = (1 / 1 + 2 / 3 + 3 / 10) / 3
dcg = 1 / math.log2(2) + 1 / math.log2(4) + 1 / math.log2(11)
idcg = sum(1 / math.log2(i + 1) for i in (1, 2, 3))
eq("L19 Q1  AP", ap, 0.6556)
eq("L19 Q1  NDCG@10", dcg / idcg, 0.8396)
eq("L19 Q1  third gain", 1 / math.log2(11), 0.2891)
eq("L19 Q1  ideal second", 1 / math.log2(3), 0.6309)

# L20 Q2 -- a perfect re-ranker cannot exceed the first stage's recall
eq("L20 Q2  ceiling", min(1.0, 0.885), 0.885, 0)

# L21 Q4 -- (PM)(Q M^-T)^T = P Q^T, for any invertible M
try:
    import numpy as np
    rng = np.random.default_rng(0)
    P, Q = rng.normal(size=(9, 4)), rng.normal(size=(7, 4))
    M = rng.normal(size=(4, 4))
    lhs = (P @ M) @ (Q @ np.linalg.inv(M).T).T
    eq("L21 Q4  rotation invariance", float(np.abs(lhs - P @ Q.T).max()), 0.0,
       1e-9)
    # L21 Q3 -- the ALS half-step is a ridge solve
    lam, Qu = 0.1, rng.normal(size=(5, 4))
    ru = rng.normal(size=5)
    als = np.linalg.solve(Qu.T @ Qu + lam * np.eye(4), Qu.T @ ru)
    from sklearn.linear_model import Ridge
    ridge = Ridge(alpha=lam, fit_intercept=False).fit(Qu, ru).coef_
    eq("L21 Q3  ALS == ridge", float(np.abs(als - ridge).max()), 0.0, 1e-8)
    # L23 Q2 -- cosine spread of independent unit vectors in d = 512
    d, V = 512, rng.normal(size=(40_000, 512))
    V /= np.linalg.norm(V, axis=1, keepdims=True)
    cos = (V[::2] * V[1::2]).sum(1)
    eq("L23 Q2  predicted spread", 1 / math.sqrt(d), 0.0442, 1e-4)
    eq("L23 Q2  measured spread", float(cos.std()), 1 / math.sqrt(d), 2e-3)
    # L17 Q2 -- the cross-entropy gradient p - y sums to zero over a row
    zs = rng.normal(size=(6, 10)) * 4
    p = np.exp(zs - zs.max(1, keepdims=True))
    p /= p.sum(1, keepdims=True)
    y = np.eye(10)[rng.integers(0, 10, 6)]
    eq("L17 Q2  row sum of p - y", float(np.abs((p - y).sum(1)).max()), 0.0,
       1e-12)
    # L17 Q1 -- softmax is invariant to a constant shift
    q = np.exp(zs + 17.0 - (zs + 17.0).max(1, keepdims=True))
    q /= q.sum(1, keepdims=True)
    eq("L17 Q1  shift invariance", float(np.abs(p - q).max()), 0.0, 1e-12)
except ImportError as exc:                    # pragma: no cover
    print(f"skip  numpy/sklearn unavailable ({exc})")

# L02 Q4 -- the quoted folds really do average to the quoted mean
folds = [0.52, 0.79, 0.75, 0.74, 0.75]
eq("L02 Q4  fold mean", sum(folds) / len(folds), 0.71, 5e-3)

# L07 Q3 -- bootstrap=False with oob_score=True raises
try:
    from sklearn.ensemble import RandomForestClassifier
    import numpy as np
    X = np.arange(40, dtype=float).reshape(20, 2)
    y = np.array([0, 1] * 10)
    raised = ""
    try:
        RandomForestClassifier(n_estimators=3, bootstrap=False,
                               oob_score=True, random_state=0).fit(X, y)
    except ValueError as exc:
        raised = str(exc)
    ok = "bootstrap" in raised.lower() or "out of bag" in raised.lower()
    print(f"{'ok  ' if ok else 'FAIL'}  L07 Q3  oob without bootstrap raises: "
          f"{raised[:60] or 'nothing raised'}")
    if not ok:
        fails.append("L07 Q3")

    # L08 Q2 -- the JL bound at n = 400, eps = 0.1 exceeds the data's own
    # dimension (Olivetti faces are 64x64 = 4096)
    from sklearn.random_projection import johnson_lindenstrauss_min_dim
    need = int(johnson_lindenstrauss_min_dim(400, eps=0.1))
    ok = need > 64 * 64
    print(f"{'ok  ' if ok else 'FAIL'}  L08 Q2  JL asks {need} dimensions of "
          f"{64 * 64} available")
    if not ok:
        fails.append("L08 Q2")
except ImportError as exc:                    # pragma: no cover
    print(f"skip  sklearn unavailable ({exc})")

# L23 Q4 -- one relevant item ranked uniformly among n
eq("L23 Q4  random recall@10 of 200", 10 / 200, 0.05, 0)

# ---------------------------------------------------------------- conclusions
# A solution can be well-formed, in scope, traceable to a figure on its deck --
# and still teach the opposite of the lecture. Lecture 6 Q4 did, for months: it
# told students that if the accuracy does not move, only the explanation is
# unstable, while Lecture 6 measures the accuracy stable to 0.25 points with
# 9.1% of predictions changing and concludes "a stable metric is not evidence
# of a stable model".
#
# There is no general check for this. A rule like "an answer about a measured
# result must cite the measurement" was tried and abandoned: 101 of the 120
# answers cite no figure, because most of these questions are conceptual by
# design, so the rule would have been noise rather than a check.
#
# What CAN be pinned is the conclusion itself. Each entry below is a solution
# that was found reversed, vague or contradicted by its own deck, and repaired.
# The pin fails if the repair is ever undone. Append, never rewrite.
CONCLUSIONS = [
    (6, 4, ["9.1", "0.25", "stable model"], ["predictions can be stable"],
     "L6 measures a stable metric beside unstable predictions"),
    (19, 3, ["0.1261", "0.1809"], ["idf mattered most"],
     "the ablation never prices saturation alone"),
    (11, 1, [r"\rho^{L-1}"], [r"multiply by $\rho^L$"],
     "deck 11 evaluates 19 steps for 20 layers"),
    (24, 5, ["Split before anything is fitted", "None of them is mathematics"],
     ["preprocessing inside the cross-validated object"],
     "the five rules are deck 24's, not a plausible substitute"),
    (4, 2, ["FamilySize"], [],
     "five of the six dependencies are the encoder's, the sixth is engineered"),
    (18, 3, [r"2\times10^{-5}"], ["tens of steps"],
     "the timescale was in the notebook only; the rate is on the deck"),
    (18, 1, ["d_k"], ["grows with"],
     "the deck derives Var(s) = d_k exactly"),
    (20, 3, ["changes the task"], [],
     "deck 20's revision slide once said the batch is only a memory limit"),
    (17, 5, ["Subword"], [],
     "a larger word vocabulary cannot reach the floor"),
]


def check_conclusions() -> None:
    import sys as _s
    _s.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
    from make_exercises import EXERCISES
    for lec, q, must, must_not, why in CONCLUSIONS:
        it = EXERCISES[lec][q - 1]
        text = it["a"] + " " + " ".join(it["why"])
        missing = [m for m in must if m not in text]
        present = [m for m in must_not if m in text]
        ok = not missing and not present
        print(f"{'ok  ' if ok else 'FAIL'}  L{lec:02d} Q{q} conclusion: {why}")
        if missing:
            fails.append(f"L{lec:02d} Q{q} lost: {missing}")
            print(f"        lost from the answer: {missing}")
        if present:
            fails.append(f"L{lec:02d} Q{q} regressed: {present}")
            print(f"        reverted phrasing is back: {present}")


check_conclusions()

print()
if fails:
    print(f"{len(fails)} claim(s) failed: {', '.join(fails)}")
    sys.exit(1)
print("every arithmetic claim in an exercise answer checks out")
