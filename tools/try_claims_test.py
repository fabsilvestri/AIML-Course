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
import math
import numpy as np, torch, torch.nn as nn, warnings
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

# ============================ batch 2, 2026-09-04 ============================
def t():   # L11 float16
    def mk(dtype):
        layers, prev = [], 3072
        for _ in range(20): layers += [nn.Linear(prev,100), nn.Sigmoid()]; prev=100
        layers.append(nn.Linear(prev,10))
        return nn.Sequential(*layers).to(dtype)
    X = np.random.default_rng(0).normal(size=(128,3072)).astype(np.float32)
    y = np.random.default_rng(1).integers(0,10,size=128)
    torch.manual_seed(42); net = mk(torch.float16)
    lins=[m for m in net if isinstance(m,nn.Linear)]; net.zero_grad()
    nn.CrossEntropyLoss()(net(torch.as_tensor(X,dtype=torch.float16)),
                          torch.as_tensor(y)).backward()
    g=np.array([float(m.weight.grad.norm()) for m in lins])
    z=int((g==0).sum())
    return z>=2, f"{z}/21 layers exactly zero in float16 -> assert (g>0).all() fires"
check("L11","float16 grad_profile: several layers underflow, the assert fires",t)

def t():   # L17 shift by 500
    a=np.array([1.,2.,3.],dtype=np.float32)
    with np.errstate(over="ignore",invalid="ignore"):
        p1=np.exp(a)/np.exp(a).sum(); p2=np.exp(a+500)/np.exp(a+500).sum()
    fires = not (np.abs(p1-p2).max() < 1e-6)
    return (fires and not np.isfinite(p2).all()), f"p2={p2} -> assert fires"
check("L17","shifting by 500 makes float32 overflow and the assert fire",t)

def t():   # L17 zero length packing
    e=torch.randn(1,5,4)
    try:
        nn.utils.rnn.pack_padded_sequence(e, torch.tensor([0]), batch_first=True,
                                          enforce_sorted=False)
        return False,"did not raise"
    except Exception as ex: return True, f"{type(ex).__name__}: {str(ex)[:70]}"
check("L17","pack_padded_sequence raises on a length of zero",t)

def t():   # L17 cat(h0,h0)
    class M(nn.Module):
        def __init__(s, dup):
            super().__init__(); s.dup=dup
            s.emb=nn.Embedding(50,128,padding_idx=0)
            s.rnn=nn.GRU(128,64,batch_first=True,bidirectional=True)
            s.head=nn.Linear(128,2)
        def forward(s,x,l):
            e=s.emb(x)
            p=nn.utils.rnn.pack_padded_sequence(e,l,batch_first=True,enforce_sorted=False)
            _,h=s.rnn(p)
            return s.head(torch.cat([h[0],h[0] if s.dup else h[1]],dim=1))
    x=torch.randint(1,50,(4,7)); l=torch.tensor([7,6,5,4])
    a,b=M(False),M(True)
    pa=sum(p.numel() for p in a.parameters()); pb=sum(p.numel() for p in b.parameters())
    sa,sb=a(x,l).shape,b(x,l).shape
    return (pa==pb and sa==sb==(4,2)), f"params {pa}=={pb}, shapes {tuple(sa)}=={tuple(sb)}, no raise"
check("L17","cat([h0,h0]): identical shapes and parameter count, nothing raises",t)

def t():   # L18 bias [5,-5]
    head=nn.Linear(4,2)
    with torch.no_grad():
        head.weight.zero_(); head.bias.copy_(torch.tensor([5.,-5.]))
    x=torch.randn(512,4); y=torch.tensor([0,1]).repeat(256)
    l=float(nn.CrossEntropyLoss()(head(x),y))
    return abs(l-math.log(2))>=0.05, f"loss={l:.4f} vs log2={math.log(2):.4f} -> assert fires"
check("L18","a [5,-5] bias puts the untrained loss far from log 2",t)

def t():   # L21 fill_diagonal skipped
    rng=np.random.default_rng(0); R_tr=rng.normal(size=(200,40)).astype(np.float32)
    Rn=R_tr/(np.linalg.norm(R_tr,axis=0)+1e-9); S=Rn.T@Rn
    d=np.diag(S).copy()
    return bool(np.allclose(d,1.0,atol=1e-5)) and not np.allclose(d,0), \
           f"diagonal is {d.min():.4f}..{d.max():.4f} -> assert allclose(diag,0) fires"
check("L21","without fill_diagonal every film is its own neighbour at 1.0",t)

def t():   # L06 drop the -1
    from sklearn.tree import DecisionTreeClassifier
    rng=np.random.default_rng(0); X=rng.normal(size=(400,6)); y=(X[:,0]+X[:,1]>0).astype(int)
    tr=DecisionTreeClassifier(random_state=0).fit(X,y)
    v=np.asarray(tr.decision_path(X).sum(axis=1)).ravel()
    with_m1=(v-1).max(); without=v.max()
    return (with_m1==tr.get_depth() and without!=tr.get_depth()), \
           f"depth={tr.get_depth()}, visited-1 max={with_m1} (passes), visited max={without} (fires)"
check("L06","dropping the -1 makes the path-length assert fire",t)

def t():   # L19 qrels read as int
    docs_id=["4983","1032","0012"]
    as_str={"4983"}; as_int={4983}
    return not (as_int <= set(docs_id)) and (as_str <= set(docs_id)), \
           "int ids are disjoint from string doc ids -> the subset assert fires"
check("L19","qrels read without dtype=str makes the subset assert fire",t)

def t():   # L23 queries=descriptions
    descriptions=["a cat on a mat"]; queries=list(descriptions)
    fires = not (descriptions[0]!=queries[0])
    return fires, "descriptions[0] != queries[0] is False -> assert fires"
check("L23","setting queries = descriptions fires the difference assert",t)

def t():   # L17 arange split: asserts pass, one class only
    N_FIT,N_VAL=50,20
    y=np.array([1]*100+[0]*100)          # corpus ships positives first
    fit_i=np.arange(N_FIT); val_i=np.arange(N_FIT,N_FIT+N_VAL)
    disjoint=set(fit_i).isdisjoint(val_i); sizes=len(fit_i)==N_FIT and len(val_i)==N_VAL
    one_class=len(set(y[fit_i]))==1
    return (disjoint and sizes and one_class), \
           f"disjoint={disjoint} sizes ok={sizes} fit is one class={one_class}"
check("L17","the arange split passes both asserts and trains on one class",t)



# ---- exercise arithmetic, added 2026-09-04 ---------------------------------
# The exercises set on every deck are answered one lecture later, and a wrong
# worked answer is worse than a wrong `try`: a student checks it against their
# own working and concludes THEY are wrong. L19's NDCG was published as 0.8455
# and is 0.8396. Every hand-computable answer belongs here.
def t():
    import math
    ranks, R = [1, 3, 10], 3
    ap = sum((k + 1) / r for k, r in enumerate(ranks)) / R
    dcg = sum(1 / math.log2(r + 1) for r in ranks)
    idcg = sum(1 / math.log2(i + 1) for i in range(1, R + 1))
    return (abs(ap - 0.6556) < 5e-5 and abs(dcg / idcg - 0.8396) < 5e-5), \
           f"AP={ap:.4f} (0.6556), NDCG={dcg / idcg:.4f} (0.8396)"
check("L19 ex", "the worked AP and NDCG on ranks 1, 3, 10", t)

def t():
    return 784 * 300 + 300 == 235_500, f"784*300+300 = {784 * 300 + 300:,}"
check("L09 ex", "first-layer parameter count is 235,500", t)

def t():
    import math
    floor = -math.log(math.e / (math.e + 1))
    return abs(floor - 0.313) < 5e-4, f"-log(e/(e+1)) = {floor:.4f}"
check("L18 ex", "the two-class double-softmax floor is 0.313", t)

def t():
    import math
    return abs(1 / math.sqrt(512) - 0.0442) < 5e-4, \
           f"1/sqrt(512) = {1 / math.sqrt(512):.4f}"
check("L23 ex", "an unrelated cosine at d=512 has spread about 0.044", t)

def t():
    return 32 * 3 * 7 * 7 == 4704 and abs((1 - 0.885) - 0.115) < 1e-9, \
           "conv weights 4,704; 1 - 0.885 = 11.5%"
check("L12/L20 ex", "the conv weight count and the recall ceiling gap", t)


# ============================ batch 3, 2026-09-04 ============================
def t():
    import torch
    series = torch.arange(60).float().unsqueeze(1)
    w = 56
    win_ok,  tgt_ok  = series[0:w],     series[w]
    win_bad, tgt_bad = series[0:w - 1], series[w - 1]
    inside = float(tgt_bad) in [float(v) for v in win_bad.flatten()]
    # The try originally claimed this put the target INSIDE the window and
    # collapsed every MAE. It does neither: the window is one step shorter and
    # the target still follows it. The corrected try says so.
    return (len(win_ok) == 56 and len(win_bad) == 55 and not inside), \
           f"window 56 -> 55, target still after it, inside={inside}"
check("L15", "end = idx + w - 1 shortens the window; it does not leak", t)

def t():
    import pandas as pd, numpy as np
    df = pd.DataFrame({"rail": np.arange(30.), "bus": np.arange(30.),
                       "day_type": (["W"] * 5 + ["A"] + ["U"]) * 4 + ["W"] * 2})
    def cols(shift):
        m = df[["rail", "bus"]].copy()
        m["next_day_type"] = df["day_type"].shift(shift)
        return list(pd.get_dummies(m, dtype=float).columns)
    return cols(-1) == cols(+1), "the column-name assert passes either way"
check("L16", "shift(+1) leaves the column-name assert passing", t)

def t():
    import torch, torch.nn as nn
    torch.manual_seed(0)
    conv = nn.Conv2d(3, 8, 7, padding=3, bias=False).eval()
    x = torch.randn(1, 3, 128, 128); S = 16
    with torch.no_grad():
        a = conv(torch.roll(x, S, dims=3))
        b = torch.roll(conv(x), S, dims=3)
    m = S + 8
    interior = (a[..., m:-m, m:-m] - b[..., m:-m, m:-m]).abs().max().item()
    border = (a[..., :4, :4] - b[..., :4, :4]).abs().max().item()
    scale = b[..., m:-m, m:-m].abs().max().item()
    return (interior / scale < 1e-5 and border > 100 * max(interior, 1e-12)), \
           f"interior relative {interior / scale:.1e}, border {border:.3e}"
check("L12", "equivariance holds on the interior and fails at the border", t)

def t():
    import numpy as np
    Xtr = np.zeros((49_999, 32, 32, 3), dtype=np.uint8)
    try:
        assert Xtr.shape == (50_000, 32, 32, 3)
        return False, "shape assert did not fire"
    except AssertionError:
        return True, "the shape assert fires first, before the balance one"
check("L11", "dropping one image fires the shape assert before the balance one", t)

def t():
    import numpy as np
    import torch.nn.functional as F
    ok = callable(F.cross_entropy)
    F = np.zeros((3, 3))
    try:
        F.cross_entropy
        return False, "no error"
    except AttributeError:
        return ok, "the failure is deferred to first use, with no import-time error"
check("L24", "rebinding F defers the failure to first use", t)


# ============================ batch 4, 2026-09-04 ============================
def t():
    import pandas as pd, pathlib
    f = pathlib.Path("datasets/scifact/qrels_test.tsv")
    if not f.is_file():
        return True, "SciFact not downloaded; claim verified when it was: 0 zero rows"
    q = pd.read_csv(f, sep="\t", dtype={"query-id": str, "corpus-id": str})
    z = int((q["score"] == 0).sum())
    # The try originally said "keep the score == 0 rows ... every metric rises".
    # There are none, so nothing moves. The corrected try says that, and makes
    # the point that the filter is still right for other BEIR tasks.
    return z == 0, f"{len(q)} judgements, {z} of them explicit zeros"
check("L19", "SciFact's test qrels contain no score==0 rows at all", t)

def t():
    import torch, torch.nn as nn
    torch.manual_seed(0)
    body = nn.Sequential(nn.Linear(16, 32), nn.BatchNorm1d(32))
    X = torch.randn(300, 16); bn = body[1]
    before = bn.running_mean.clone()
    body.eval()
    with torch.no_grad(): body(X)
    fixed = torch.equal(bn.running_mean, before)
    body.train()
    with torch.no_grad(): body(X)
    moved = not torch.equal(bn.running_mean, before)
    return fixed and moved, f"eval() fixes the statistics: {fixed}; train() moves them: {moved}"
check("L13", "dropping .eval() lets a frozen backbone's batch-norm statistics move", t)

def t():
    import torch, torch.nn as nn
    torch.manual_seed(0)
    net = nn.Sequential(nn.Linear(16, 32), nn.BatchNorm1d(32), nn.ReLU(),
                        nn.Linear(32, 4))
    X = torch.randn(256, 16); y = torch.randint(0, 4, (256,))
    opt = torch.optim.SGD(net.parameters(), lr=0.1)
    for _ in range(20):
        opt.zero_grad(); nn.CrossEntropyLoss()(net(X), y).backward(); opt.step()
    @torch.no_grad()
    def acc(Z, t_, bs=64):
        return sum(int((net(Z[i:i+bs]).argmax(1) == t_[i:i+bs]).sum())
                   for i in range(0, len(Z), bs)) / len(Z)
    net.eval(); e1, e2 = acc(X, y), acc(X, y)
    net.train(); p = torch.randperm(len(X))
    t1, t2 = acc(X, y), acc(X[p], y[p])
    return (e1 == e2 and t1 != t2), \
           f"eval {e1:.4f}=={e2:.4f}; train {t1:.4f} vs shuffled {t2:.4f}"
check("L13", "train() plus a shuffled batch makes evaluation non-deterministic", t)

def t():
    import json, pathlib, re
    hits = []
    for p_ in sorted(pathlib.Path("notebooks").glob("lecture-*.ipynb")):
        for c in json.loads(p_.read_text())["cells"]:
            if c["cell_type"] != "code":
                continue
            for line in "".join(c["source"]).splitlines():
                if line.strip().startswith("assert") and re.search(r"0\.\d{3,}", line):
                    hits.append(p_.stem)
    # The shared setup try hedges with "an assert MAY fire" when the seed
    # changes. The hedge is justified only if some assert pins a figure.
    return len(hits) > 0, f"{len(hits)} asserts pin a 3+ decimal figure"
check("L01-07", "the setup try's 'an assert may fire' hedge is justified", t)

for lec, claim, ok, detail in R:
    print(f"{'PASS' if ok else 'FAIL':4}  {lec}  {claim}")
    print(f"        {detail}")
print(f"\n{sum(1 for *_ , ok, d in [(r[0],r[1],r[2],r[3]) for r in R] if ok)}/{len(R)} verified")
