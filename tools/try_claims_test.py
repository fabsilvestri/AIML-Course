#!/usr/bin/env python3
"""
The try-field claims that can be reproduced standalone, executed.

    python3 tools/try_claims_test.py

Each check below is one `try` field's prediction, rebuilt small enough to run
in seconds without executing a notebook. A PASS means a student who follows
that `try` will see what it told them they would see.

This file is the executable half of the audit described in REBUILD.md
§ "The try-field audit". It is append-only in spirit: when you verify another
claim, add it here rather than verifying it once in a shell and forgetting.

Two claims failed when this file was first run, both in Lecture 8, and both
were corrected in the notebook rather than deleted from here -- the fixed
versions are the ones now asserted.
"""
import numpy as np, warnings
warnings.filterwarnings("ignore")
R = []
def check(lec, claim, fn):
    try:
        ok, detail = fn()
    except Exception as e:
        ok, detail = False, f"raised {type(e).__name__}: {e}"
    R.append((lec, claim, ok, detail))

# ---- L02: singular design matrix, inv vs pinv -------------------------------
def t():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 3)); X = np.c_[np.ones(50), X, X[:,0]+X[:,1]]
    y = rng.normal(size=50)
    A = X.T @ X
    try:
        w = np.linalg.inv(A) @ X.T @ y
        inv_bad = not np.isfinite(w).all() or np.linalg.cond(A) > 1e12
    except np.linalg.LinAlgError:
        inv_bad = True
    wp = np.linalg.pinv(A) @ X.T @ y
    return (inv_bad and np.isfinite(wp).all()), f"cond={np.linalg.cond(A):.2e} inv_bad={inv_bad}"
check("L02", "inv raises or returns nonsense on a collinear column; pinv does not", t)

# ---- L02: handle_unknown='error' raises -------------------------------------
def t():
    from sklearn.preprocessing import OneHotEncoder
    e = OneHotEncoder(handle_unknown="error").fit([["a"],["b"]])
    try:
        e.transform([["z"]]); return False, "did not raise"
    except ValueError as ex:
        return True, "ValueError as predicted"
check("L02", "handle_unknown='error' raises on an unseen category", t)

# ---- L03: recall of an all-negative predictor is 0.0 ------------------------
def t():
    from sklearn.metrics import recall_score
    y = np.array([1]*9 + [0]*91)
    r = recall_score(y, np.zeros(100, dtype=int), zero_division=0)
    return r == 0.0, f"recall={r}"
check("L03", "never-fires scored with recall returns 0.0", t)

# ---- L07: bootstrap=False with oob_score=True raises ------------------------
def t():
    from sklearn.ensemble import RandomForestClassifier
    X = np.random.default_rng(0).normal(size=(60, 4)); y = (X[:,0] > 0).astype(int)
    try:
        RandomForestClassifier(n_estimators=5, bootstrap=False,
                               oob_score=True).fit(X, y)
        return False, "did not raise"
    except (ValueError, TypeError) as ex:
        return True, f"{type(ex).__name__}: {str(ex)[:70]}"
check("L07", "bootstrap=False with oob_score=True raises", t)

# ---- L08: IncrementalPCA batch_size < n_components raises -------------------
def t():
    from sklearn.decomposition import IncrementalPCA
    X = np.random.default_rng(0).normal(size=(400, 200))
    try:
        IncrementalPCA(n_components=123, batch_size=100).fit(X)
        return False, "did not raise"
    except ValueError as ex:
        # The original try said the message "names the constraint you just
        # violated". It does not -- it talks about the number of input features
        # changing. The corrected try says so, and this asserts that.
        misleading = "input features has changed" in str(ex)
        return misleading, f"misleading message as documented: {str(ex)[:80]}"
check("L08", "IncrementalPCA batch_size=100 raises, with a message naming "
             "the WRONG constraint", t)

# ---- L08: PCA without centring — ALL FIVE components move -------------------
# The original try said "the assert fails on component 1 and the rest still
# agree". Measured on the real Olivetti faces, every one of the five disagrees,
# so the try was corrected to say so. This asserts the corrected claim.
def t():
    from sklearn.datasets import fetch_olivetti_faces
    from sklearn.decomposition import PCA
    X = fetch_olivetti_faces(shuffle=False).data
    pca = PCA(random_state=42).fit(X)
    _, _, Vt = np.linalg.svd(X, full_matrices=False)          # NOT centred
    per = [float(np.abs(np.abs(Vt[i]) - np.abs(pca.components_[i])).max())
           for i in range(5)]
    return (max(per) >= 1e-4 and min(per) >= 1e-4), \
           f"per-component max abs diff {[round(x, 4) for x in per]} — all five fail 1e-4"
check("L08", "uncentred PCA: the assert fires and ALL FIVE components move", t)

# ---- L14: IoU without the clamp still passes all three asserts --------------
def t():
    def iou_noclip(a, b):
        a, b = np.asarray(a, float), np.asarray(b, float)
        lt, rb = np.maximum(a[:2], b[:2]), np.minimum(a[2:], b[2:])
        wh = rb - lt                       # NO clip
        inter = wh[0] * wh[1]
        aa = (a[2]-a[0])*(a[3]-a[1]); ab = (b[2]-b[0])*(b[3]-b[1])
        return float(inter / (aa + ab - inter))
    box = np.array([0., 0., 100., 100.])
    a1 = iou_noclip(box, box) == 1.0
    a2 = iou_noclip(box, box + np.array([100,0,100,0])) == 0.0
    a3 = abs(iou_noclip(box, box + np.array([50,0,50,0])) - 1/3) < 1e-12
    return (a1 and a2 and a3), f"identical={a1} edge={a2} half={a3}"
check("L14", "removing np.clip: all three existing asserts still pass", t)

# ---- L14: a range check catches only the one-axis case ----------------------
def t():
    def iou_broken(a, b):
        x1, y1 = max(a[0], b[0]), max(a[1], b[1])
        x2, y2 = min(a[2], b[2]), min(a[3], b[3])
        inter = (x2-x1) * (y2-y1)
        aa = (a[2]-a[0])*(a[3]-a[1]); ab = (b[2]-b[0])*(b[3]-b[1])
        return inter / (aa + ab - inter)
    one  = iou_broken([0,0,100,100], [300,0,400,100])
    d150 = iou_broken([0,0,100,100], [150,150,250,250])
    d200 = iou_broken([0,0,100,100], [200,200,300,300])
    fires = lambda v: not (0.0 <= v <= 1.0)
    return (fires(one) and not fires(d150) and not fires(d200)), \
           f"one-axis={one:.3f}(fires) both150={d150:.4f} both200={d200:.3f}"
check("L14", "range check fires only on the one-axis pair", t)

# ---- L14: the diagonal sweep diverges near 241 px ---------------------------
def t():
    def iou_broken_d(x):
        a = np.array([0.,0.,100.,100.]); b = a + np.array([x,x,x,x])
        lt, rb = np.maximum(a[:2],b[:2]), np.minimum(a[2:],b[2:])
        wh = rb - lt; inter = wh[0]*wh[1]
        return inter / (10000 + 10000 - inter)
    vals = [(x, iou_broken_d(x)) for x in (200, 235, 241, 245, 300)]
    diverges = max(abs(v) for _, v in vals) > 50
    neg_after = iou_broken_d(300) < 0
    return (diverges and neg_after), f"{[(x, round(v,2)) for x,v in vals]}"
check("L14", "broken IoU diverges near 241 px then returns negative", t)

# ---- L13: kernel 7 -> 3 costs 3,840 weights --------------------------------
def t():
    d = 32*3*7*7 - 32*3*3*3
    return d == 3840, f"4704 - 864 = {d}"
check("L13", "k=7 to k=3 costs exactly 3,840 weights", t)

# ---- L13: frozen-head trainable count is 52,326 -----------------------------
def t():
    return 512*102 + 102 == 52326, f"512*102+102 = {512*102+102}"
check("L13", "trainable head parameter count is 52,326", t)

# ---- L18: ten-class double-softmax floor -----------------------------------
def t():
    import math
    floor = -math.log(math.e / (math.e + 9))
    return abs(floor - 1.46) < 0.01 and floor < math.log(10), \
           f"floor={floor:.4f}  log10={math.log(10):.4f}"
check("L18", "ten-class double-softmax floor is about 1.46, below log 10", t)

# ---- L23: at d=1 the cosine sd is 1 = 1/sqrt(d) ----------------------------
def t():
    rng = np.random.default_rng(0)
    A = np.sign(rng.normal(size=(20000,1))); B = np.sign(rng.normal(size=(20000,1)))
    c = (A*B).sum(1)
    return abs(c.std(ddof=1) - 1.0) < 0.01, f"sd={c.std(ddof=1):.4f}, 1/sqrt(1)=1"
check("L23", "at d=1 the cosine sd is 1, which is still 1/sqrt(d)", t)

# ---- L21: raw film ids leave 246 empty columns ------------------------------
def t():
    return 3952 - 3706 == 246, f"3952 - 3706 = {3952-3706}"
check("L21", "un-factorised film ids add 246 all-zero columns", t)

# ---- L19: k1 = 0 makes every tf factor exactly 1 ----------------------------
def t():
    f = lambda tf, k1: tf*(k1+1)/(tf+k1)
    vals = [f(tf, 0.0) for tf in (1,2,3,5,10,20)]
    return all(v == 1.0 for v in vals), f"{vals}"
check("L19", "k1 = 0 collapses every tf factor to exactly 1", t)

# ---- L17: float64 moves the softmax overflow to ~709 ------------------------
def t():
    a = np.log(np.finfo(np.float32).max); b = np.log(np.finfo(np.float64).max)
    return abs(a-88.7) < 0.1 and abs(b-709.8) < 0.5, f"f32={a:.2f} f64={b:.2f}"
check("L17", "exp overflows float32 at 88.7 and float64 at ~709", t)

# ---- L17: reduction='mean' disagrees by exactly the batch size --------------
def t():
    import torch, torch.nn as nn
    torch.manual_seed(0)
    z = torch.randn(7, 5, dtype=torch.float64, requires_grad=True)
    y = torch.randint(0, 5, (7,))
    nn.CrossEntropyLoss(reduction="mean")(z, y).backward()
    g_mean = z.grad.clone(); z.grad = None
    nn.CrossEntropyLoss(reduction="sum")(z, y).backward()
    ratio = (z.grad / g_mean).abs()
    return bool(np.allclose(ratio.numpy(), 7.0)), f"sum/mean ratio = {ratio.mean():.4f}"
check("L17", "mean vs sum reduction differs by exactly the batch size 7", t)

for lec, claim, ok, detail in R:
    print(f"{'PASS' if ok else 'FAIL':4}  {lec}  {claim}")
    print(f"        {detail}")
print(f"\n{sum(1 for *_ , ok, d in [(r[0],r[1],r[2],r[3]) for r in R] if ok)}/{len(R)} verified")
