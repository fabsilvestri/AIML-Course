#!/usr/bin/env python3
"""
Application 8 — Lectures 15 and 16. Flowers102, a convolutional network built
from scratch, and the transfer-learning repair.

    python3 tools/figures_app08.py

Everything printed on slides/lecture-15.html and slides/lecture-16.html comes
from here, via figkit.export() into assets/figures/figures.json. Expensive fits
are cached (figkit.cached) so a cosmetic re-run takes seconds; delete
/private/tmp/claude-501/aiml-data/fits-v2.pkl to refit from scratch.

Timings are wall-clock on the machine that generated the figures — an Apple
Silicon laptop with an MPS backend and no CUDA. They are measurements of this
machine, not guarantees about a Colab T4, and the decks say so on the slide
that quotes them.

The decoded images are cached as uint8 tensors under
/private/tmp/claude-501/aiml-data/flowers_<split>_<size>.pt. Decoding 8,189
JPEGs takes about twenty seconds and would otherwise dominate every run.
"""

from __future__ import annotations

import sys
import time
import xml.dom.minidom
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))

from figkit import (setup, save, cached, load_cache, export, OUT, SEED,   # noqa: E402
                    PRIMARY, ACCENT, SUCCESS, MATH, MUTED, RULE, AXIS,
                    BODY, SMALL, TICK, check_text_floor)

import torch                                                    # noqa: E402
import torch.nn as nn                                           # noqa: E402
import torch.nn.functional as F                                 # noqa: E402
import torchvision                                              # noqa: E402
from torchvision import transforms as T                         # noqa: E402
from torchvision.models import ResNet18_Weights                 # noqa: E402

CACHE = Path("/private/tmp/claude-501/aiml-data")
DATA = CACHE / "flowers"

DEVICE = ("cuda" if torch.cuda.is_available()
          else "mps" if torch.backends.mps.is_available() else "cpu")

# --- the application's constants, quoted on the slides ----------------------
N_CLASSES = 102
IMG = 128                 # the from-scratch network's input, H = W
BATCH = 32
EPOCHS = 80
LR = 3e-4                 # Adam; 1e-3 does not train this net on 1,020 images
DROPOUT = 0.5

TRANSFER_IMG = 224        # what the pretrained weights were trained at
PROBE_EPOCHS = 60         # the linear head on frozen features
FT_EPOCHS = 15            # fine-tuning layer4 + fc
FT_LR_BACKBONE = 1e-4
FT_LR_HEAD = 1e-3

# The eight species whose names appear on the sample grid. Flowers102 ships
# integer labels only; these are the standard class names for the dataset and
# are quoted, not derived, so they are stated here rather than measured.
CLASS_NAMES = [
    "pink primrose", "hard-leaved pocket orchid", "canterbury bells",
    "sweet pea", "english marigold", "tiger lily", "moon orchid",
    "bird of paradise", "monkshood", "globe thistle", "snapdragon",
    "colt's foot", "king protea", "spear thistle", "yellow iris",
    "globe-flower", "purple coneflower", "peruvian lily", "balloon flower",
    "giant white arum lily", "fire lily", "pincushion flower", "fritillary",
    "red ginger", "grape hyacinth", "corn poppy", "prince of wales feathers",
    "stemless gentian", "artichoke", "sweet william", "carnation",
    "garden phlox", "love in the mist", "mexican aster", "alpine sea holly",
    "ruby-lipped cattleya", "cape flower", "great masterwort", "siam tulip",
    "lenten rose", "barbeton daisy", "daffodil", "sword lily", "poinsettia",
    "bolero deep blue", "wallflower", "marigold", "buttercup", "oxeye daisy",
    "common dandelion", "petunia", "wild pansy", "primula", "sunflower",
    "pelargonium", "bishop of llandaff", "gaura", "geranium", "orange dahlia",
    "pink-yellow dahlia", "cautleya spicata", "japanese anemone",
    "black-eyed susan", "silverbush", "californian poppy", "osteospermum",
    "spring crocus", "bearded iris", "windflower", "tree poppy", "gazania",
    "azalea", "water lily", "rose", "thorn apple", "morning glory",
    "passion flower", "lotus", "toad lily", "anthurium", "frangipani",
    "clematis", "hibiscus", "columbine", "desert-rose", "tree mallow",
    "magnolia", "cyclamen", "watercress", "canna lily", "hippeastrum",
    "bee balm", "ball moss", "foxglove", "bougainvillea", "camellia",
    "mallow", "mexican petunia", "bromelia", "blanket flower",
    "trumpet creeper", "blackberry lily",
]

# ImageNet statistics, quoted from the pretrained weights' own transform.
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])[None, :, None, None]
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225])[None, :, None, None]


# ------------------------------------------------------------------ the data

def preload(split: str, size: int):
    """Decode one split once, at one resolution, and keep it as uint8.

    Returns (N, 3, size, size) uint8 and (N,) int64. Kept outside the pickle
    cache: the test split at 224 is 925 MB and pickling it with everything else
    would make every cosmetic re-run rewrite a gigabyte.
    """
    f = CACHE / f"flowers_{split}_{size}.pt"
    if f.is_file():
        d = torch.load(f)
        return d["x"], d["y"]
    ds = torchvision.datasets.Flowers102(DATA, split=split, download=True)
    tf = T.Compose([T.Resize((size, size)), T.PILToTensor()])
    t0 = time.perf_counter()
    x = torch.stack([tf(ds[i][0]) for i in range(len(ds))])
    y = torch.as_tensor(ds._labels)
    print(f"    decoded {len(ds):,} {split} images at {size}px in "
          f"{time.perf_counter() - t0:.0f} s")
    CACHE.mkdir(parents=True, exist_ok=True)
    torch.save({"x": x, "y": y}, f)
    return x, y


def load_flowers() -> dict:
    d = {}
    for split in ("train", "val", "test"):
        d[f"X_{split}"], d[f"y_{split}"] = preload(split, IMG)
    # The normalisation statistics come from the TRAINING split only. That is
    # the whole of the Lecture 15 assistant failure, so it is written once here
    # and never recomputed on a wider set by accident.
    xf = d["X_train"].float() / 255.0
    d["mean"] = xf.mean((0, 2, 3))
    d["std"] = xf.std((0, 2, 3))
    return d


def normalise(x_u8: torch.Tensor, mean: torch.Tensor, std: torch.Tensor):
    return ((x_u8.float() / 255.0) - mean[None, :, None, None]) \
        / std[None, :, None, None]


# --------------------------------------------------------------- the network

def conv_block(c_in: int, c_out: int, k: int = 3):
    """Conv, batch-norm, ReLU — the Lecture 14 toolkit, applied.

    `bias=False` because the batch-norm that follows has its own shift, so a
    convolution bias would be a parameter with no effect on the function.
    """
    return [nn.Conv2d(c_in, c_out, k, padding=k // 2, bias=False),
            nn.BatchNorm2d(c_out), nn.ReLU()]


def make_net() -> nn.Sequential:
    """The architecture the whole of Lecture 15 is about.

    128x128x3 -> four pooling stages -> 8x8x256 -> 16,384 -> 256 -> 102.
    The first layer is 7x7 rather than 3x3 so that the learned filters are
    large enough to read on a projector; every other layer is 3x3.
    """
    return nn.Sequential(
        *conv_block(3, 32, k=7), *conv_block(32, 32), nn.MaxPool2d(2),
        *conv_block(32, 64), *conv_block(64, 64), nn.MaxPool2d(2),
        *conv_block(64, 128), *conv_block(128, 128), nn.MaxPool2d(2),
        *conv_block(128, 256), nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(256 * (IMG // 16) ** 2, 256), nn.ReLU(),
        nn.Dropout(DROPOUT),
        nn.Linear(256, N_CLASSES),
    )


def layer_table() -> list[dict]:
    """Per-layer output shape, parameter count and activation count.

    Every line of this table appears on a slide, so every number in it has to
    be traceable — a total alone would let a single wrong row through.
    """
    net = make_net()
    rows: list[dict] = []
    x = torch.zeros(1, 3, IMG, IMG)
    for mod in net:
        x = mod(x)
        p = sum(q.numel() for q in mod.parameters())
        rows.append({"layer": type(mod).__name__,
                     "shape": list(x.shape[1:]),
                     "params": int(p),
                     "activations": int(x.numel())})
    return rows


def n_params(net: nn.Module) -> int:
    return sum(p.numel() for p in net.parameters())


# -------------------------------------------------------------- training loop

@torch.no_grad()
def accuracy(net, X, y, bs=256) -> float:
    net.eval()
    right = 0
    for k in range(0, len(X), bs):
        xb = X[k:k + bs]
        xb = xb if xb.device.type == DEVICE else xb.to(DEVICE)
        right += (net(xb).argmax(1).cpu() == y[k:k + bs].cpu()).sum().item()
    return right / len(X)


def train_scratch(d, *, epochs=EPOCHS, lr=LR, seed=SEED, mean=None, std=None,
                  track=True):
    """Train the from-scratch network and record the wall clock as it goes.

    The x-axis of the learning curve is seconds, not epochs, because the cost
    is the point of the lecture.
    """
    mean = d["mean"] if mean is None else mean
    std = d["std"] if std is None else std

    torch.manual_seed(seed)
    np.random.seed(seed)
    net = make_net().to(DEVICE)
    init_filters = net[0].weight.detach().cpu().clone()

    Xtr = normalise(d["X_train"], mean, std).to(DEVICE)
    ytr = d["y_train"].to(DEVICE)
    Xva = normalise(d["X_val"], mean, std).to(DEVICE)

    opt = torch.optim.Adam(net.parameters(), lr=lr)
    lossf = nn.CrossEntropyLoss()
    gen = torch.Generator(device=DEVICE).manual_seed(seed)

    hist = {"epoch": [], "seconds": [], "loss": [], "train_acc": [],
            "val_acc": []}
    t0 = time.perf_counter()
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(len(Xtr), device=DEVICE, generator=gen)
        run = 0.0
        for k in range(0, len(Xtr), BATCH):
            idx = perm[k:k + BATCH]
            opt.zero_grad()
            loss = lossf(net(Xtr[idx]), ytr[idx])
            loss.backward()
            opt.step()
            run += loss.item() * len(idx)
        if track:
            hist["epoch"].append(ep + 1)
            hist["seconds"].append(time.perf_counter() - t0)
            hist["loss"].append(run / len(Xtr))
            hist["train_acc"].append(accuracy(net, Xtr, d["y_train"]))
            hist["val_acc"].append(accuracy(net, Xva, d["y_val"]))
    seconds = time.perf_counter() - t0

    Xte = normalise(d["X_test"], mean, std)
    test_acc = accuracy(net, Xte, d["y_test"])
    return {"hist": hist, "seconds": seconds, "test_acc": test_acc,
            "val_acc": accuracy(net, Xva, d["y_val"]),
            "train_acc": accuracy(net, Xtr, d["y_train"]),
            "n_params": n_params(net), "epochs": epochs, "lr": lr, "seed": seed,
            "init_filters": init_filters,
            "state": {k: v.detach().cpu() for k, v in net.state_dict().items()}}


# ------------------------------------------------------------- thread 8 · (1)

def dense_vs_conv() -> dict:
    """Weights in a dense layer against weights in the real first conv layer.

    Same input (3 x 128 x 128) and the same output shape (32 x 128 x 128) in
    both cases, so the comparison is between two ways of computing the same
    sized thing rather than between two different layers.
    """
    h = w = IMG
    c_in, c_out, k = 3, 32, 7
    n_in = c_in * h * w
    n_out = c_out * h * w
    dense = n_in * n_out
    conv = c_out * c_in * k * k
    return {"h": h, "w": w, "c_in": c_in, "c_out": c_out, "k": k,
            "n_in": n_in, "n_out": n_out,
            "dense_weights": int(dense), "conv_weights": int(conv),
            "ratio": dense / conv,
            "dense_bytes": int(dense) * 4, "conv_bytes": int(conv) * 4,
            "dense_gb": dense * 4 / 1024 ** 3,
            # the same comparison for the 3x3 layers, which is what the rest of
            # the network uses
            "conv3x3_weights": c_out * c_in * 3 * 3,
            "ratio_3x3": dense / (c_out * c_in * 3 * 3)}


# ------------------------------------------------------------- thread 8 · (2)

def equivariance(state, d, shift=16) -> dict:
    """Measure equivariance and invariance on the trained first layer.

    Equivariance: conv(shift(x)) equals shift(conv(x)) away from the border.
    Invariance:   pool(conv(shift(x))) equals pool(conv(x)), everywhere.

    Both are measured rather than asserted, because the first is exact only up
    to float32 rounding and the second only up to what leaves the frame.
    """
    torch.manual_seed(SEED)
    conv = nn.Conv2d(3, 32, 7, padding=3, bias=False)
    conv.weight.data = state["0.weight"].clone()
    conv.eval()

    x = normalise(d["X_test"][:1], d["mean"], d["std"])
    xs = torch.roll(x, shifts=shift, dims=3)

    with torch.no_grad():
        y = conv(x)
        ys = conv(xs)
    y_shift = torch.roll(y, shifts=shift, dims=3)

    # the seam where the roll wraps, and the border where zero padding bites,
    # are excluded: a convolution is equivariant on the interior, and the claim
    # is about the interior
    m = shift + 8
    a = ys[..., m:-m, m:-m]
    b = y_shift[..., m:-m, m:-m]
    max_abs = (a - b).abs().max().item()
    scale = b.abs().max().item()

    with torch.no_grad():
        pooled = F.adaptive_max_pool2d(F.relu(y), 1).flatten()
        pooled_s = F.adaptive_max_pool2d(F.relu(ys), 1).flatten()
    cos_pooled = F.cosine_similarity(pooled, pooled_s, dim=0).item()
    cos_maps = F.cosine_similarity(F.relu(y).flatten(),
                                   F.relu(ys).flatten(), dim=0).item()

    return {"shift_px": shift,
            "equivariance_max_abs": max_abs,
            "activation_scale": scale,
            "equivariance_relative": max_abs / scale,
            "cos_pooled": cos_pooled,
            "cos_maps": cos_maps,
            "pooled_change_pct": 100 * (1 - cos_pooled),
            "maps_change_pct": 100 * (1 - cos_maps),
            # what pooling costs a per-pixel task: the last feature map is
            # IMG/16 on a side, so one cell answers for 16 x 16 input pixels
            "final_map": IMG // 16,
            "cell_covers_px": 16 * 16,
            "resolution_loss": (IMG * IMG) / ((IMG // 16) ** 2)}


# ------------------------------------------------------------- thread 8 · (3)

def memory_budget(batches=(1, 8, 16, 32, 64, 128)) -> dict:
    """Where the memory actually goes: parameters against activations.

    The activation figure is the sum of every module output, which is what
    autograd keeps alive between the forward and the backward pass. It is per
    image, so it scales with the batch; the parameter figure does not.
    """
    net = make_net()
    sizes: list[dict] = []
    x = torch.zeros(1, 3, IMG, IMG)
    for i, mod in enumerate(net):
        x = mod(x)
        sizes.append({"i": i, "layer": type(mod).__name__,
                      "elements": int(x.numel()),
                      "shape": list(x.shape[1:])})
    act_per_image = sum(s["elements"] for s in sizes)
    n_par = n_params(net)

    rows = []
    for b in batches:
        act_mb = act_per_image * b * 4 / 1024 ** 2
        rows.append({"batch": b, "activations_mb": act_mb})

    par_mb = n_par * 4 / 1024 ** 2
    return {
        "n_params": int(n_par),
        "params_mb": par_mb,
        # weights + gradients + Adam's two moments, all float32
        "optimizer_mb": 4 * par_mb,
        "act_per_image": int(act_per_image),
        "act_per_image_mb": act_per_image * 4 / 1024 ** 2,
        "layers": sizes,
        "rows": rows,
        "batch": BATCH,
        "act_at_batch_mb": act_per_image * BATCH * 4 / 1024 ** 2,
        "ratio_at_batch": (act_per_image * BATCH) / (4 * n_par),
    }


def measured_memory() -> dict:
    """Corroborate the arithmetic against what the allocator actually reserves.

    A prediction nobody checks is a claim. This runs one training step at the
    lecture's batch size and reads the backend's own counter, so the slide can
    say the arithmetic was verified rather than merely stated.
    """
    if DEVICE != "mps":
        return {"available": False, "device": DEVICE}
    torch.mps.empty_cache()
    net = make_net().to(DEVICE)
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    x = torch.randn(BATCH, 3, IMG, IMG, device=DEVICE)
    y = torch.randint(0, N_CLASSES, (BATCH,), device=DEVICE)
    lossf = nn.CrossEntropyLoss()
    # one step first, so Adam's state exists and is counted in the baseline
    opt.zero_grad(); lossf(net(x), y).backward(); opt.step()
    torch.mps.synchronize()
    torch.mps.empty_cache()
    base = torch.mps.current_allocated_memory()
    net.train()
    out = net(x)
    torch.mps.synchronize()
    peak = torch.mps.current_allocated_memory()
    del out
    return {"available": True,
            "baseline_mb": base / 1024 ** 2,
            "forward_mb": (peak - base) / 1024 ** 2,
            "batch": BATCH}


# ------------------------------------------------------------ transfer learning

def imagenet_norm(x_u8: torch.Tensor) -> torch.Tensor:
    return ((x_u8.float() / 255.0) - IMAGENET_MEAN) / IMAGENET_STD


def backbone_features(X_u8, bs=64) -> torch.Tensor:
    """Everything in resnet18 except the classifier, run once, in eval mode."""
    rn = torchvision.models.resnet18(weights=ResNet18_Weights.DEFAULT)
    body = nn.Sequential(*list(rn.children())[:-1]).to(DEVICE).eval()
    out = []
    with torch.no_grad():
        for k in range(0, len(X_u8), bs):
            out.append(body(imagenet_norm(X_u8[k:k + bs]).to(DEVICE))
                       .flatten(1).cpu())
    return torch.cat(out)


def linear_probe() -> dict:
    """Freeze the whole backbone; train 52,326 parameters on its output."""
    Xtr, ytr = preload("train", TRANSFER_IMG)
    Xva, yva = preload("val", TRANSFER_IMG)
    Xte, yte = preload("test", TRANSFER_IMG)

    t0 = time.perf_counter()
    Ftr = backbone_features(Xtr)
    Fva = backbone_features(Xva)
    t_feat = time.perf_counter() - t0

    torch.manual_seed(SEED)
    head = nn.Linear(512, N_CLASSES).to(DEVICE)
    opt = torch.optim.Adam(head.parameters(), lr=1e-3, weight_decay=1e-4)
    lossf = nn.CrossEntropyLoss()
    Ftr_d, ytr_d = Ftr.to(DEVICE), ytr.to(DEVICE)
    gen = torch.Generator(device=DEVICE).manual_seed(SEED)

    hist = {"epoch": [], "seconds": [], "val_acc": []}
    t1 = time.perf_counter()
    for ep in range(PROBE_EPOCHS):
        head.train()
        perm = torch.randperm(len(Ftr_d), device=DEVICE, generator=gen)
        for k in range(0, len(Ftr_d), BATCH):
            idx = perm[k:k + BATCH]
            opt.zero_grad()
            lossf(head(Ftr_d[idx]), ytr_d[idx]).backward()
            opt.step()
        head.eval()
        with torch.no_grad():
            va = (head(Fva.to(DEVICE)).argmax(1).cpu() == yva).float().mean()
        hist["epoch"].append(ep + 1)
        hist["seconds"].append(t_feat + time.perf_counter() - t1)
        hist["val_acc"].append(va.item())
    t_head = time.perf_counter() - t1

    Fte = backbone_features(Xte)
    head.eval()
    with torch.no_grad():
        te = (head(Fte.to(DEVICE)).argmax(1).cpu() == yte).float().mean().item()
    return {"hist": hist, "feature_seconds": t_feat, "head_seconds": t_head,
            "seconds": t_feat + t_head, "test_acc": te,
            "val_acc": hist["val_acc"][-1],
            "n_trainable": 512 * N_CLASSES + N_CLASSES,
            "n_frozen": n_params(torchvision.models.resnet18()) - (512 * 1000 + 1000),
            "epochs": PROBE_EPOCHS}


def augment(x_u8: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    """Random resized crop and a horizontal flip, on a uint8 batch.

    Written by hand on tensors rather than as a torchvision transform pipeline
    so that the notebook and this script apply exactly the same operation, and
    so the augmentation is visible as arithmetic rather than as a class name.
    """
    n = len(x_u8)
    out = torch.empty_like(x_u8)
    size = x_u8.shape[-1]
    scales = 0.55 + 0.45 * torch.rand(n, generator=gen)
    for i in range(n):
        s = int(size * scales[i].item())
        top = int(torch.randint(0, size - s + 1, (1,), generator=gen).item())
        left = int(torch.randint(0, size - s + 1, (1,), generator=gen).item())
        crop = x_u8[i:i + 1, :, top:top + s, left:left + s].float()
        crop = F.interpolate(crop, size=size, mode="bilinear",
                             align_corners=False)
        if torch.rand(1, generator=gen).item() < 0.5:
            crop = crop.flip(-1)
        out[i] = crop[0].clamp(0, 255).to(torch.uint8)
    return out


def finetune(*, augmented=True, epochs=FT_EPOCHS, seed=SEED) -> dict:
    """Unfreeze layer4 and the head; two learning rates, one optimiser.

    The backbone's early layers stay frozen — they hold edge and colour
    detectors that a flower dataset of 1,020 images cannot improve on.
    """
    Xtr, ytr = preload("train", TRANSFER_IMG)
    Xva, yva = preload("val", TRANSFER_IMG)
    Xte, yte = preload("test", TRANSFER_IMG)

    torch.manual_seed(seed)
    rn = torchvision.models.resnet18(weights=ResNet18_Weights.DEFAULT)
    rn.fc = nn.Linear(512, N_CLASSES)
    rn = rn.to(DEVICE)

    for p in rn.parameters():
        p.requires_grad = False
    for p in rn.layer4.parameters():
        p.requires_grad = True
    for p in rn.fc.parameters():
        p.requires_grad = True

    opt = torch.optim.Adam([
        {"params": rn.layer4.parameters(), "lr": FT_LR_BACKBONE},
        {"params": rn.fc.parameters(), "lr": FT_LR_HEAD},
    ])
    lossf = nn.CrossEntropyLoss()
    gen = torch.Generator().manual_seed(seed)

    n_train = sum(p.numel() for p in rn.parameters() if p.requires_grad)
    n_froz = sum(p.numel() for p in rn.parameters() if not p.requires_grad)

    ytr_d = ytr.to(DEVICE)
    hist = {"epoch": [], "seconds": [], "val_acc": [], "val_acc_aug": []}
    t0 = time.perf_counter()
    for ep in range(epochs):
        rn.train()
        # batch-norm running statistics in the frozen layers must not drift on
        # 1,020 images: freeze the layers that are not being trained
        for m in [rn.bn1, rn.layer1, rn.layer2, rn.layer3]:
            m.eval()
        perm = torch.randperm(len(Xtr), generator=gen)
        for k in range(0, len(Xtr), BATCH):
            idx = perm[k:k + BATCH]
            xb = augment(Xtr[idx], gen) if augmented else Xtr[idx]
            xb = imagenet_norm(xb).to(DEVICE)
            opt.zero_grad()
            lossf(rn(xb), ytr_d[idx]).backward()
            opt.step()
        hist["epoch"].append(ep + 1)
        hist["seconds"].append(time.perf_counter() - t0)
        hist["val_acc"].append(accuracy(rn, imagenet_norm(Xva), yva, bs=64))
        # the same weights, scored on an AUGMENTED validation set — this is the
        # Lecture 16 assistant failure, measured while the run is happening
        hist["val_acc_aug"].append(
            accuracy(rn, imagenet_norm(augment(Xva, gen)), yva, bs=64))
    seconds = time.perf_counter() - t0

    test_acc = accuracy(rn, imagenet_norm(Xte), yte, bs=64)

    # The Lecture 16 assistant failure, measured on the FINISHED model: score
    # one fixed set of weights on an augmented validation set, ten times. A
    # deterministic function of fixed weights and fixed data does not wobble.
    clean = accuracy(rn, imagenet_norm(Xva), yva, bs=64)
    scores = [accuracy(rn, imagenet_norm(augment(Xva, gen)), yva, bs=64)
              for _ in range(10)]
    wobble = {"clean": clean, "scores": scores, "repeats": len(scores),
              "mean": float(np.mean(scores)), "sd": float(np.std(scores)),
              "spread_pts": float(100 * (max(scores) - min(scores))),
              "penalty_pts": float(100 * (clean - np.mean(scores)))}

    return {"hist": hist, "seconds": seconds, "test_acc": test_acc,
            "val_acc": hist["val_acc"][-1], "epochs": epochs,
            "n_trainable": int(n_train), "n_frozen": int(n_froz),
            "augmented": augmented, "wobble": wobble,
            "state_fc": {k: v.detach().cpu() for k, v in rn.fc.state_dict().items()}}


# ------------------------------------------- Lecture 15 · the assistant failure

def normalisation_leak(d, *, seeds=5, epochs=40) -> dict:
    """Statistics over the whole dataset against statistics over the train split.

    Both the size of the difference in the statistics themselves and the size
    of the difference it makes to the score. Reported honestly even when the
    second is smaller than the run-to-run spread — especially then, because
    that is the decision rule the lecture is selling.
    """
    all_u8 = torch.cat([d["X_train"], d["X_val"], d["X_test"]])
    xf = all_u8.float() / 255.0
    mean_all, std_all = xf.mean((0, 2, 3)), xf.std((0, 2, 3))

    honest, leaky = [], []
    for s in range(seeds):
        honest.append(train_scratch(d, epochs=epochs, seed=SEED + s,
                                    track=False)["test_acc"])
        leaky.append(train_scratch(d, epochs=epochs, seed=SEED + s,
                                   mean=mean_all, std=std_all,
                                   track=False)["test_acc"])
    honest, leaky = np.array(honest), np.array(leaky)
    return {
        "n_all": int(len(all_u8)),
        "mean_train": d["mean"].tolist(), "std_train": d["std"].tolist(),
        "mean_all": mean_all.tolist(), "std_all": std_all.tolist(),
        "mean_abs_diff": float((mean_all - d["mean"]).abs().max()),
        "std_abs_diff": float((std_all - d["std"]).abs().max()),
        "mean_rel_pct": float(100 * ((mean_all - d["mean"]).abs()
                                     / d["mean"]).max()),
        "honest_mean": float(honest.mean()), "honest_sd": float(honest.std()),
        "leaky_mean": float(leaky.mean()), "leaky_sd": float(leaky.std()),
        "gap_pts": float(100 * (leaky.mean() - honest.mean())),
        "seed_spread_pts": float(100 * honest.std()),
        "seeds": seeds, "epochs": epochs,
        "honest": honest.tolist(), "leaky": leaky.tolist(),
    }


# ------------------------------------------- Lecture 16 · the assistant failure

# ------------------------------------------------------- Lecture 15 · figures

def fig_grid(d):
    """Sixteen flowers with their species names, four per row."""
    rng = np.random.default_rng(SEED)
    labels = d["y_train"].numpy()
    chosen = []
    for c in rng.choice(N_CLASSES, 16, replace=False):
        chosen.append(int(np.where(labels == c)[0][0]))

    fig, axes = plt.subplots(2, 8, figsize=(11.0, 3.4))
    for ax, i in zip(axes.ravel(), chosen):
        ax.imshow(d["X_train"][i].permute(1, 2, 0).numpy())
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
        name = CLASS_NAMES[labels[i]]
        ax.set_title(name if len(name) <= 17 else name[:16] + "…",
                     fontsize=SMALL, color=MUTED, pad=4)
    fig.suptitle(f"16 of the {N_CLASSES} species, one image each, "
                 f"resized to {IMG} × {IMG}",
                 fontsize=SMALL, color=MUTED, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save(fig, "l15-grid", raster=True)


def fig_splits(d):
    """The three splits, and the class-count distribution inside each."""
    counts = {s: np.bincount(d[f"y_{s}"].numpy(), minlength=N_CLASSES)
              for s in ("train", "val", "test")}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 3.3),
                                   gridspec_kw={"width_ratios": [1, 1.8]})

    names = ["train", "val", "test"]
    totals = [int(counts[s].sum()) for s in names]
    bars = ax1.bar(names, totals, color=[PRIMARY, PRIMARY, ACCENT], width=0.62)
    for b, t in zip(bars, totals):
        ax1.annotate(f"{t:,}", (b.get_x() + b.get_width() / 2, t),
                     ha="center", va="bottom", fontsize=SMALL, color=MUTED)
    ax1.set_ylabel("images")
    ax1.set_ylim(0, max(totals) * 1.18)
    ax1.set_title("The split we are given")

    order = np.argsort(counts["test"])[::-1]
    ax2.bar(range(N_CLASSES), counts["test"][order], color=ACCENT, width=1.0)
    ax2.axhline(counts["train"][0], color=PRIMARY, lw=2)
    ax2.annotate(f"training: exactly {counts['train'][0]} per class",
                 xy=(60, counts["train"][0]), xytext=(38, 120),
                 fontsize=SMALL, color=PRIMARY,
                 bbox=dict(fc="white", ec="none", alpha=0.9),
                 arrowprops=dict(arrowstyle="->", color=PRIMARY, lw=1.8))
    ax2.annotate(f"largest test class: {counts['test'].max()}",
                 xy=(0, counts["test"].max()), xytext=(14, 205),
                 fontsize=SMALL, color=MUTED,
                 bbox=dict(fc="white", ec="none", alpha=0.9),
                 arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.6))
    ax2.set_xlabel("species, sorted by test count")
    ax2.set_ylabel("images")
    ax2.set_title("The test split is not balanced")
    fig.tight_layout()
    save(fig, "l15-splits")
    return {s: {"total": int(counts[s].sum()),
                "min": int(counts[s].min()), "max": int(counts[s].max())}
            for s in names}


def fig_baseline(base):
    fig, ax = plt.subplots(figsize=(11.0, 2.9))
    names = ["always the\ncommonest class", "uniform random\nguess",
             "what the operator\nwould accept"]
    vals = [100 * base["majority"], 100 * base["uniform"], 90.0]
    cols = [ACCENT, ACCENT, SUCCESS]
    bars = ax.barh(names[::-1], vals[::-1], color=cols[::-1], height=0.58)
    for b, v in zip(bars, vals[::-1]):
        ax.annotate(f"{v:.2f}%" if v < 10 else f"{v:.0f}%",
                    (v, b.get_y() + b.get_height() / 2),
                    xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=SMALL, color=MUTED)
    ax.set_xlim(0, 100)
    ax.set_xlabel("test accuracy, %")
    ax.set_title("Anchor the commitment before you write it down")
    fig.tight_layout()
    save(fig, "l15-baseline")


def _filter_image(w: torch.Tensor) -> np.ndarray:
    """One (3, k, k) filter, rescaled to [0, 1] so its structure is visible."""
    a = w.numpy().transpose(1, 2, 0)
    lo, hi = a.min(), a.max()
    return (a - lo) / (hi - lo + 1e-12)


def fig_filters(res):
    """The 32 first-layer filters, at initialisation and after training."""
    after = res["state"]["0.weight"]
    before = res["init_filters"]
    fig, axes = plt.subplots(4, 16, figsize=(11.0, 3.2))
    for col, (block, title) in enumerate(((before, "at initialisation"),
                                          (after, "after training"))):
        for j in range(32):
            r, c = divmod(j, 8)
            ax = axes[r, c + 8 * col]
            ax.imshow(_filter_image(block[j]), interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
            for s in ax.spines.values():
                s.set_visible(False)
        axes[0, 8 * col + 3].set_title(title, fontsize=BODY, color=MUTED,
                                       loc="left", pad=8)
    fig.suptitle(f"All 32 filters of the first layer, 7 × 7 × 3, "
                 f"each rescaled to its own range",
                 fontsize=SMALL, color=MUTED, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    save(fig, "l15-filters", raster=True)


def fig_featmaps(res, d):
    """One image, and what eight of the 32 first-layer filters make of it."""
    net = make_net()
    net.load_state_dict(res["state"])
    net.eval()
    x = normalise(d["X_test"][:1], d["mean"], d["std"])
    with torch.no_grad():
        a1 = net[2](net[1](net[0](x)))            # conv, bn, relu
        deep = x
        for mod in net[:14]:
            deep = mod(deep)

    fig, axes = plt.subplots(2, 9, figsize=(11.0, 2.9))
    axes[0, 0].imshow(d["X_test"][0].permute(1, 2, 0).numpy())
    axes[0, 0].set_title("input", fontsize=SMALL, color=MUTED, pad=4)
    axes[1, 0].imshow(d["X_test"][0].permute(1, 2, 0).numpy())
    axes[1, 0].set_title("input", fontsize=SMALL, color=MUTED, pad=4)
    for j in range(8):
        axes[0, j + 1].imshow(a1[0, j].numpy(), cmap="magma")
        axes[0, j + 1].set_title(f"f{j}", fontsize=SMALL, color=MUTED, pad=4)
        axes[1, j + 1].imshow(deep[0, j].numpy(), cmap="magma")
        axes[1, j + 1].set_title(f"f{j}", fontsize=SMALL, color=MUTED, pad=4)
    for ax in axes.ravel():
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
    axes[0, 0].set_ylabel(f"layer 1\n{a1.shape[2]}×{a1.shape[3]}",
                          fontsize=SMALL, color=MUTED)
    axes[1, 0].set_ylabel(f"4 convs on\n{deep.shape[2]}×{deep.shape[3]}",
                          fontsize=SMALL, color=MUTED)
    fig.suptitle("Feature maps: the first layer keeps the picture, "
                 "the fourth keeps a summary",
                 fontsize=SMALL, color=MUTED, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    save(fig, "l15-featmaps", raster=True)


def fig_curve(res, base):
    """Accuracy against WALL CLOCK, because the cost is the point."""
    h = res["hist"]
    fig, ax = plt.subplots(figsize=(11.0, 3.3))
    ax.plot(h["seconds"], [100 * v for v in h["train_acc"]],
            color=PRIMARY, lw=2.4, label="training set")
    ax.plot(h["seconds"], [100 * v for v in h["val_acc"]],
            color=ACCENT, lw=2.4, label="validation set")
    ax.axhline(100 * base["majority"], color=MUTED, lw=1.6, ls=":")
    ax.annotate("always the commonest class", xy=(5, 100 * base["majority"]),
                xytext=(5, 12), fontsize=SMALL, color=MUTED,
                bbox=dict(fc="white", ec="none", alpha=0.9))
    ax.annotate(f"{100 * res['val_acc']:.1f}% on validation\n"
                f"after {res['seconds']:.0f} s",
                xy=(h["seconds"][-1], 100 * res["val_acc"]),
                xytext=(h["seconds"][-1] * 0.42, 100 * res["val_acc"] + 26),
                fontsize=SMALL, color=ACCENT,
                bbox=dict(fc="white", ec="none", alpha=0.9),
                arrowprops=dict(arrowstyle="->", color=ACCENT, lw=1.8))
    ax.set_xlabel("wall clock, seconds")
    ax.set_ylabel("accuracy, %")
    ax.set_ylim(0, 100)
    ax.legend(loc="upper left")
    ax.set_title(f"{res['epochs']} epochs, {res['n_params']:,} parameters, "
                 f"1,020 training images")
    fig.tight_layout()
    save(fig, "l15-curve")


def fig_gap(res):
    """The gap the next lecture has to close."""
    h = res["hist"]
    gap = [100 * (a - b) for a, b in zip(h["train_acc"], h["val_acc"])]
    fig, ax = plt.subplots(figsize=(11.0, 2.9))
    ax.plot(h["epoch"], gap, color=MATH, lw=2.4)
    ax.fill_between(h["epoch"], 0, gap, color=MATH, alpha=0.12)
    ax.annotate(f"{gap[-1]:.0f} points of pure memorisation",
                xy=(h["epoch"][-1], gap[-1]),
                xytext=(h["epoch"][-1] * 0.30, gap[-1] * 0.55),
                fontsize=SMALL, color=MATH,
                bbox=dict(fc="white", ec="none", alpha=0.9),
                arrowprops=dict(arrowstyle="->", color=MATH, lw=1.8))
    ax.set_xlabel("epoch")
    ax.set_ylabel("train − validation, points")
    ax.set_title("1,020 images against 4.8 million parameters")
    fig.tight_layout()
    save(fig, "l15-gap")
    return {"final_gap_pts": gap[-1], "max_gap_pts": max(gap)}


def fig_pooling():
    """Max pooling, worked on numbers, so nobody has to take it on trust."""
    rng = np.random.default_rng(SEED)
    a = rng.integers(0, 10, (4, 4))
    pooled = a.reshape(2, 2, 2, 2).max(axis=(1, 3))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.6, 2.8),
                                   gridspec_kw={"width_ratios": [1, 1]})
    for ax, m, title in ((ax1, a, "4 × 4 feature map"),
                         (ax2, pooled, "after 2 × 2 max pool")):
        ax.imshow(np.zeros_like(m), cmap="Greys", vmin=0, vmax=1)
        for (i, j), v in np.ndenumerate(m):
            ax.text(j, i, str(v), ha="center", va="center",
                    fontsize=BODY, color=PRIMARY)
        ax.set_xticks(np.arange(-.5, m.shape[1], 1), minor=True)
        ax.set_yticks(np.arange(-.5, m.shape[0], 1), minor=True)
        ax.grid(which="minor", color=RULE, lw=1.4)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(title, fontsize=SMALL, color=MUTED)
    for i in range(2):
        for j in range(2):
            ax1.add_patch(plt.Rectangle((2 * j - .5, 2 * i - .5), 2, 2,
                                        fill=False, ec=ACCENT, lw=2.4))
    fig.tight_layout()
    save(fig, "l15-pooling")
    return {"before": a.tolist(), "after": pooled.tolist()}


# ------------------------------------------------------- Lecture 16 · figures

def fig_params(dc):
    fig, ax = plt.subplots(figsize=(11.0, 2.6))
    names = ["dense layer\nsame input, same output",
             f"convolutional layer\n32 filters of {dc['k']}×{dc['k']}×3"]
    vals = [dc["dense_weights"], dc["conv_weights"]]
    bars = ax.barh(names[::-1], vals[::-1], color=[SUCCESS, ACCENT], height=0.55)
    ax.set_xscale("log")
    ax.set_xlim(1e2, 1e12)
    for b, v in zip(bars, vals[::-1]):
        ax.annotate(f"{v:,} weights", (v, b.get_y() + b.get_height() / 2),
                    xytext=(8, 0), textcoords="offset points",
                    va="center", fontsize=SMALL, color=MUTED)
    ax.set_xlabel("weights, log scale")
    ax.set_title(f"A factor of {dc['ratio']:,.0f}")
    fig.tight_layout()
    save(fig, "l16-params")


def fig_equivariance(res, d, eq):
    """Shift the input; the feature map shifts with it."""
    net = make_net()
    net.load_state_dict(res["state"])
    net.eval()
    x = normalise(d["X_test"][:1], d["mean"], d["std"])
    xs = torch.roll(x, shifts=eq["shift_px"], dims=3)
    with torch.no_grad():
        y = F.relu(net[1](net[0](x)))
        ys = F.relu(net[1](net[0](xs)))
    ch = int(y[0].flatten(1).max(1).values.argmax())

    raw = d["X_test"][0].permute(1, 2, 0).numpy()
    raw_s = np.roll(raw, eq["shift_px"], axis=1)

    fig, axes = plt.subplots(1, 4, figsize=(11.0, 2.9))
    panels = [(raw, f"input", None),
              (y[0, ch].numpy(), f"feature map, filter {ch}", "magma"),
              (raw_s, f"input shifted {eq['shift_px']} px", None),
              (ys[0, ch].numpy(), "its feature map", "magma")]
    for ax, (img, title, cmap) in zip(axes, panels):
        ax.imshow(img, cmap=cmap)
        ax.set_title(title, fontsize=SMALL, color=MUTED, pad=5)
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
    fig.suptitle("The map did not change — it moved, by exactly the same "
                 f"{eq['shift_px']} pixels",
                 fontsize=SMALL, color=MUTED, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    save(fig, "l16-equivariance", raster=True)


def fig_memory(mem):
    """Parameters are a flat line; activations are a slope."""
    fig, ax = plt.subplots(figsize=(11.0, 3.2))
    b = [r["batch"] for r in mem["rows"]]
    act = [r["activations_mb"] for r in mem["rows"]]
    ax.plot(b, act, "o-", color=ACCENT, lw=2.6, ms=9, label="activations")
    ax.axhline(mem["optimizer_mb"], color=PRIMARY, lw=2.4,
               label="weights + gradients + Adam state")
    ax.annotate(f"{mem['optimizer_mb']:.0f} MB, whatever the batch",
                xy=(8, mem["optimizer_mb"]), xytext=(1.6, 260),
                fontsize=SMALL, color=PRIMARY,
                bbox=dict(fc="white", ec="none", alpha=0.9),
                arrowprops=dict(arrowstyle="->", color=PRIMARY, lw=1.8))
    ax.annotate(f"batch {mem['batch']}: {mem['act_at_batch_mb']:,.0f} MB",
                xy=(mem["batch"], mem["act_at_batch_mb"]),
                xytext=(2.2, mem["act_at_batch_mb"] * 1.05),
                fontsize=SMALL, color=ACCENT,
                bbox=dict(fc="white", ec="none", alpha=0.9),
                arrowprops=dict(arrowstyle="->", color=ACCENT, lw=1.8))
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xticks(b); ax.set_xticklabels([str(v) for v in b])
    ax.set_xlabel("batch size")
    ax.set_ylabel("memory, MB (log)")
    ax.legend(loc="upper left")
    ax.set_title("Where the memory goes")
    fig.tight_layout()
    save(fig, "l16-memory")


def fig_memory_layers(mem):
    """And it goes into the layers nearest the image."""
    keep = [s for s in mem["layers"] if s["layer"] in ("Conv2d", "MaxPool2d")]
    fig, ax = plt.subplots(figsize=(11.0, 3.0))
    mb = [s["elements"] * BATCH * 4 / 1024 ** 2 for s in keep]
    labels = [f"{s['layer'].replace('2d', '')}\n{s['shape'][0]}×"
              f"{s['shape'][1]}×{s['shape'][2]}" for s in keep]
    cols = [ACCENT if s["layer"] == "Conv2d" else PRIMARY for s in keep]
    ax.bar(range(len(keep)), mb, color=cols, width=0.68)
    ax.set_xticks(range(len(keep)))
    ax.set_xticklabels(labels, fontsize=SMALL - 3)
    ax.set_ylabel(f"MB at batch {BATCH}")
    ax.annotate("the first two layers alone cost\n"
                f"{mb[0] + mb[1]:.0f} MB — "
                f"{100 * (mb[0] + mb[1]) / sum(mb):.0f}% of the convolutional total",
                xy=(0.5, mb[0]), xytext=(2.4, mb[0] * 0.72),
                fontsize=SMALL, color=MUTED,
                bbox=dict(fc="white", ec="none", alpha=0.92),
                arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.8))
    ax.set_title("Activation memory, layer by layer")
    fig.tight_layout()
    save(fig, "l16-memory-layers")
    convs = [(s, m) for s, m in zip(keep, mb) if s["layer"] == "Conv2d"]
    return {"first_two_mb": mb[0] + mb[1],
            "first_two_pct": 100 * (mb[0] + mb[1]) / sum(mb),
            "conv_total_mb": sum(m for _, m in convs),
            "first_conv_mb": convs[0][1],
            "third_conv_mb": convs[2][1],
            "last_conv_mb": convs[-1][1]}


def fig_transfer(scratch, probe, ft):
    """Accuracy against wall clock, for all three runs, on one pair of axes."""
    fig, ax = plt.subplots(figsize=(11.0, 3.4))
    ax.plot(scratch["hist"]["seconds"],
            [100 * v for v in scratch["hist"]["val_acc"]],
            color=ACCENT, lw=2.6, label="from scratch")
    ax.plot(probe["hist"]["seconds"],
            [100 * v for v in probe["hist"]["val_acc"]],
            color=MATH, lw=2.6, label="frozen backbone, linear head")
    ax.plot(ft["hist"]["seconds"],
            [100 * v for v in ft["hist"]["val_acc"]],
            color=SUCCESS, lw=2.6, label="fine-tuned + augmented")
    for res, col in ((scratch, ACCENT), (probe, MATH), (ft, SUCCESS)):
        ax.annotate(f"{100 * res['val_acc']:.1f}%",
                    xy=(res["hist"]["seconds"][-1], 100 * res["val_acc"]),
                    xytext=(6, -4), textcoords="offset points",
                    fontsize=SMALL, color=col)
    ax.set_xlabel("wall clock, seconds")
    ax.set_ylabel("validation accuracy, %")
    ax.set_ylim(0, 100)
    ax.legend(loc="center right")
    ax.set_title("Same data, same laptop, same 1,020 labelled images")
    fig.tight_layout()
    save(fig, "l16-transfer")


def fig_augment():
    """One image, eight of the views the network will be shown."""
    X, _ = preload("train", IMG)
    gen = torch.Generator().manual_seed(SEED)
    base = X[7:8]
    views = [augment(base, gen)[0] for _ in range(7)]
    fig, axes = plt.subplots(1, 8, figsize=(11.0, 1.9))
    axes[0].imshow(base[0].permute(1, 2, 0).numpy())
    axes[0].set_title("original", fontsize=SMALL, color=PRIMARY, pad=4)
    for ax, v in zip(axes[1:], views):
        ax.imshow(v.permute(1, 2, 0).numpy())
        ax.set_title("augmented", fontsize=SMALL, color=MUTED, pad=4)
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
    fig.tight_layout()
    save(fig, "l16-augment", raster=True)


def fig_aug_val(ft):
    """The clean validation curve, and the one an augmented set would give."""
    h = ft["hist"]
    fig, ax = plt.subplots(figsize=(11.0, 3.1))
    ax.plot(h["epoch"], [100 * v for v in h["val_acc"]],
            color=SUCCESS, lw=2.6, label="validation, as it should be")
    ax.plot(h["epoch"], [100 * v for v in h["val_acc_aug"]],
            color=ACCENT, lw=2.6, ls="--",
            label="validation, augmented too")
    best_clean = int(np.argmax(h["val_acc"])) + 1
    best_aug = int(np.argmax(h["val_acc_aug"])) + 1
    ax.axvline(best_clean, color=SUCCESS, lw=1.4, ls=":")
    ax.axvline(best_aug, color=ACCENT, lw=1.4, ls=":")
    ax.annotate(f"epoch {best_clean} chosen", xy=(best_clean, 20),
                xytext=(4, 4), textcoords="offset points",
                fontsize=SMALL, color=SUCCESS)
    ax.annotate(f"epoch {best_aug} chosen", xy=(best_aug, 8),
                xytext=(4, 4), textcoords="offset points",
                fontsize=SMALL, color=ACCENT)
    ax.set_xlabel("epoch")
    ax.set_ylabel("accuracy, %")
    ax.set_ylim(0, 100)
    ax.legend(loc="lower right")
    ax.set_title("The same weights, scored two ways")
    fig.tight_layout()
    save(fig, "l16-aug-val")
    return {"best_epoch_clean": best_clean, "best_epoch_aug": best_aug,
            "clean_at_best": h["val_acc"][best_clean - 1],
            "aug_at_best": h["val_acc_aug"][best_aug - 1],
            "penalty_pts": 100 * (h["val_acc"][-1] - h["val_acc_aug"][-1])}


def fig_leak(leak):
    """The normalisation leak, against the run-to-run spread it hides in."""
    fig, ax = plt.subplots(figsize=(11.0, 2.9))
    h = 100 * np.array(leak["honest"])
    l = 100 * np.array(leak["leaky"])
    ax.plot(h, np.zeros_like(h) + 1, "o", color=SUCCESS, ms=11,
            label="statistics from the training split")
    ax.plot(l, np.zeros_like(l) + 0, "o", color=ACCENT, ms=11,
            label="statistics from all 8,189 images")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["leaky", "honest"])
    ax.set_ylim(-0.6, 1.6)
    ax.set_xlabel("test accuracy, %")
    ax.legend(loc="upper center", ncols=2)
    ax.set_title(f"{leak['seeds']} seeds each, {leak['epochs']} epochs — "
                 f"the two clouds overlap")
    fig.tight_layout()
    save(fig, "l16-leak")


# ------------------------------------------------------- hand-drawn diagrams

SVG_HEAD = """<svg xmlns="http://www.w3.org/2000/svg"
     width="{w}" height="{h}" viewBox="{vb}"
     font-family="'Source Sans 3','Source Sans Pro',Helvetica,sans-serif">
  <style>
    .box   {{ fill:#fff; stroke:#0b3d62; stroke-width:2; }}
    .box-r {{ fill:#fff; stroke:#c0392b; stroke-width:2; }}
    .box-g {{ fill:#fff; stroke:#14663a; stroke-width:2; }}
    .box-m {{ fill:#fff; stroke:#6c3483; stroke-width:2; }}
    .t     {{ font-size:19px; fill:#16212b; }}
    .t-sub {{ font-size:16px; fill:#4b5563; }}
    .t-hd  {{ font-size:23px; font-weight:700; }}
    .flow  {{ stroke-width:2.5; fill:none; }}
  </style>
  <defs>
    <marker id="a" markerWidth="9" markerHeight="9" refX="7.5" refY="3"
            orient="auto"><path d="M0,0 L0,6 L8,3 z" fill="#0b3d62"/></marker>
    <marker id="ar" markerWidth="9" markerHeight="9" refX="7.5" refY="3"
            orient="auto"><path d="M0,0 L0,6 L8,3 z" fill="#c0392b"/></marker>
    <marker id="ag" markerWidth="9" markerHeight="9" refX="7.5" refY="3"
            orient="auto"><path d="M0,0 L0,6 L8,3 z" fill="#14663a"/></marker>
    <marker id="am" markerWidth="9" markerHeight="9" refX="7.5" refY="3"
            orient="auto"><path d="M0,0 L0,6 L8,3 z" fill="#6c3483"/></marker>
  </defs>
"""


def write_svg(name: str, viewbox: str, body: str) -> None:
    """Write a hand-authored diagram.

    width and height are written explicitly, not left to the viewBox: an SVG
    loaded through <img> takes no intrinsic size from a viewBox alone, and
    renders at 0x0. Measured in Chrome, and checked by make_figures.py.
    """
    _, _, w, h = viewbox.split()
    (OUT / f"{name}.svg").write_text(
        SVG_HEAD.format(vb=viewbox, w=w, h=h) + body + "\n</svg>\n")
    print(f"  assets/figures/{name}.svg")


def diagram_weight_sharing():
    """One filter, three positions, the same nine numbers."""
    b = ['<text class="t-hd" x="20" y="34" fill="#0b3d62">'
         'One filter, every position, the same weights</text>']
    # the input grid
    x0, y0, cell = 40, 70, 34
    for i in range(8):
        for j in range(8):
            b.append(f'<rect x="{x0 + j * cell}" y="{y0 + i * cell}" '
                     f'width="{cell}" height="{cell}" fill="#eef4f8" '
                     f'stroke="#b0bcc7" stroke-width="1"/>')
    b.append(f'<text class="t-sub" x="{x0}" y="{y0 + 8 * cell + 24}">'
             f'input feature map</text>')
    # three windows
    for k, (i, j, col) in enumerate(((0, 0, "#c0392b"), (2, 3, "#c0392b"),
                                     (5, 5, "#c0392b"))):
        b.append(f'<rect x="{x0 + j * cell}" y="{y0 + i * cell}" '
                 f'width="{3 * cell}" height="{3 * cell}" rx="7" fill="none" '
                 f'stroke="{col}" stroke-width="3"/>')
        b.append(f'<line x1="{x0 + (j + 1.5) * cell}" '
                 f'y1="{y0 + (i + 1.5) * cell}" x2="600" y2="200" '
                 f'class="flow" stroke="#c0392b" stroke-dasharray="5 4" '
                 f'opacity="0.55"/>')
    # the kernel
    b.append('<rect x="600" y="140" width="150" height="120" rx="7" '
             'class="box-r"/>')
    b.append('<text class="t" x="675" y="180" text-anchor="middle" '
             'fill="#c0392b">3 &#215; 3 &#215; C</text>')
    b.append('<text class="t" x="675" y="208" text-anchor="middle" '
             'fill="#c0392b">kernel</text>')
    b.append('<text class="t-sub" x="675" y="238" text-anchor="middle">'
             'one set of weights</text>')
    # the output
    b.append('<line x1="760" y1="200" x2="828" y2="200" class="flow" '
             'stroke="#0b3d62" marker-end="url(#a)"/>')
    ox = 850
    for i in range(6):
        for j in range(6):
            b.append(f'<rect x="{ox + j * 28}" y="{110 + i * 28}" '
                     f'width="28" height="28" fill="#eef4f8" '
                     f'stroke="#b0bcc7" stroke-width="1"/>')
    b.append(f'<text class="t-sub" x="{ox}" y="{110 + 6 * 28 + 24}">'
             f'output feature map</text>')
    b.append('<text class="t" x="40" y="386">'
             'A dense layer would learn a different weight for every one of '
             'those windows.</text>')
    write_svg("d-weightsharing", "0 0 1080 410", "\n".join(b))


def diagram_equivariance():
    """Equivariance moves the answer; invariance discards where it was."""
    b = ['<text class="t-hd" x="20" y="34" fill="#0b3d62">'
         'Equivariance moves the answer &#183; invariance forgets where it '
         'was</text>']
    rows = [(80, "#6c3483", "am", "equivariant", "conv(shift(x)) = shift(conv(x))"),
            (250, "#14663a", "ag", "invariant",
             "pool(conv(shift(x))) = pool(conv(x))")]
    for y, col, mk, name, law in rows:
        b.append(f'<text class="t" x="20" y="{y - 14}" fill="{col}">'
                 f'{name}</text>')
        for k, (bx, label) in enumerate(((60, "x"), (300, "shift(x)"))):
            b.append(f'<rect x="{bx}" y="{y}" width="120" height="90" rx="7" '
                     f'class="box"/>')
            b.append(f'<text class="t-sub" x="{bx + 60}" y="{y + 52}" '
                     f'text-anchor="middle">{label}</text>')
        b.append(f'<line x1="180" y1="{y + 45}" x2="292" y2="{y + 45}" '
                 f'class="flow" stroke="#0b3d62" marker-end="url(#a)" '
                 f'stroke-dasharray="6 4"/>')
        b.append(f'<text class="t-sub" x="236" y="{y + 34}" '
                 f'text-anchor="middle">shift</text>')
        for k, bx in enumerate((60, 300)):
            b.append(f'<line x1="{bx + 60}" y1="{y + 90}" x2="{bx + 60}" '
                     f'y2="{y + 118}" class="flow" stroke="{col}" '
                     f'marker-end="url(#{mk})"/>')
        b.append(f'<rect x="560" y="{y}" width="440" height="90" rx="7" '
                 f'class="box-{"m" if col == "#6c3483" else "g"}"/>')
        b.append(f'<text class="t" x="780" y="{y + 52}" text-anchor="middle" '
                 f'fill="{col}">{law}</text>')
    b.append('<text class="t" x="20" y="404">'
             'Classification wants the second line. Per-pixel prediction '
             'cannot afford it.</text>')
    write_svg("d-equivariance", "0 0 1080 430", "\n".join(b))


def diagram_transfer(probe, ft):
    """What is frozen, what moves, and at what rate."""
    b = ['<text class="t-hd" x="20" y="34" fill="#0b3d62">'
         'Freeze the general, train the specific</text>']
    stages = [("conv1 + layer1", "#4b5563", "frozen"),
              ("layer2", "#4b5563", "frozen"),
              ("layer3", "#4b5563", "frozen"),
              ("layer4", "#6c3483", "lr 1e-4"),
              ("new fc &#8594; 102", "#14663a", "lr 1e-3")]
    x = 40
    for name, col, note in stages:
        w = 190 if "fc" in name else 180
        cls = ("box" if col == "#4b5563"
               else "box-m" if col == "#6c3483" else "box-g")
        b.append(f'<rect x="{x}" y="110" width="{w}" height="96" rx="7" '
                 f'class="{cls}"/>')
        b.append(f'<text class="t" x="{x + w / 2}" y="{152}" '
                 f'text-anchor="middle" fill="{col}">{name}</text>')
        b.append(f'<text class="t-sub" x="{x + w / 2}" y="{180}" '
                 f'text-anchor="middle" fill="{col}">{note}</text>')
        if x > 40:
            b.append(f'<line x1="{x - 22}" y1="158" x2="{x - 6}" y2="158" '
                     f'class="flow" stroke="#0b3d62" marker-end="url(#a)"/>')
        x += w + 22
    b.append('<text class="t-sub" x="40" y="88">ImageNet weights, '
             '11,689,512 parameters, downloaded not trained</text>')
    b.append(f'<text class="t" x="40" y="264">'
             f'{ft["n_frozen"]:,} parameters never move. '
             f'{ft["n_trainable"]:,} do.</text>')
    b.append('<text class="t-sub" x="40" y="298">'
             'Two learning rates, one optimiser: the early layers already hold '
             'edges and colours,</text>')
    b.append('<text class="t-sub" x="40" y="322">'
             'and 1,020 flower photographs cannot improve on them.</text>')
    write_svg("d-transfer", "0 0 1080 350", "\n".join(b))


def diagram_memory():
    """The picture behind the RAM arithmetic."""
    b = ['<text class="t-hd" x="20" y="34" fill="#0b3d62">'
         'What is alive between the forward and the backward pass</text>']
    b.append('<rect x="40" y="80" width="300" height="110" rx="7" '
             'class="box"/>')
    b.append('<text class="t" x="190" y="120" text-anchor="middle">'
             'parameters</text>')
    b.append('<text class="t-sub" x="190" y="148" text-anchor="middle">'
             'weights + gradients + Adam</text>')
    b.append('<text class="t-sub" x="190" y="172" text-anchor="middle">'
             'fixed &#8212; independent of the batch</text>')
    b.append('<rect x="40" y="220" width="640" height="150" rx="7" '
             'class="box-r"/>')
    b.append('<text class="t" x="360" y="262" text-anchor="middle" '
             'fill="#c0392b">every layer&#8217;s output</text>')
    b.append('<text class="t-sub" x="360" y="292" text-anchor="middle">'
             'kept because the backward pass evaluates each Jacobian at the '
             'value</text>')
    b.append('<text class="t-sub" x="360" y="316" text-anchor="middle">'
             'the forward pass reached &#8212; thread 6, in bytes</text>')
    b.append('<text class="t" x="360" y="350" text-anchor="middle" '
             'fill="#c0392b">&#215; batch size</text>')
    b.append('<line x1="700" y1="295" x2="756" y2="295" class="flow" '
             'stroke="#c0392b" marker-end="url(#ar)"/>')
    b.append('<rect x="770" y="220" width="280" height="150" rx="7" '
             'class="box-r"/>')
    b.append('<text class="t" x="910" y="280" text-anchor="middle" '
             'fill="#c0392b">out of memory</text>')
    b.append('<text class="t-sub" x="910" y="312" text-anchor="middle">'
             'halve the batch, not the model</text>')
    write_svg("d-memory", "0 0 1080 400", "\n".join(b))


def validate_diagrams() -> list[str]:
    """A malformed hand-authored SVG fails silently to an empty box."""
    bad = []
    for p in sorted(OUT.glob("d-*.svg")):
        try:
            xml.dom.minidom.parse(str(p))
        except Exception as exc:                              # noqa: BLE001
            bad.append(f"{p.name}: {exc}")
    return bad


# ---------------------------------------------------------------------- main

def _strip(r: dict) -> dict:
    """Drop the tensors before pickling — the cache is for numbers."""
    return {k: v for k, v in r.items()
            if k not in ("state", "init_filters", "state_fc")}


def main() -> int:
    setup()
    load_cache()

    print("Flowers102…")
    d = load_flowers()
    counts = {s: np.bincount(d[f"y_{s}"].numpy(), minlength=N_CLASSES)
              for s in ("train", "val", "test")}

    base = {"majority": float(counts["test"].max() / counts["test"].sum()),
            "uniform": 1.0 / N_CLASSES}

    facts: dict = {
        "l15_n_classes": N_CLASSES,
        "l15_n_train": int(len(d["y_train"])),
        "l15_n_val": int(len(d["y_val"])),
        "l15_n_test": int(len(d["y_test"])),
        "l15_n_total": int(len(d["y_train"]) + len(d["y_val"])
                           + len(d["y_test"])),
        "l15_per_class_train": int(counts["train"][0]),
        "l15_test_min": int(counts["test"].min()),
        "l15_test_max": int(counts["test"].max()),
        "l15_img": IMG,
        "l15_n_pixels": IMG * IMG * 3,
        "l15_batch": BATCH,
        "l15_epochs": EPOCHS,
        "l15_lr": LR,
        "l15_dropout": DROPOUT,
        "l15_baseline_majority": base["majority"],
        "l15_baseline_uniform": base["uniform"],
        "l15_train_mean": d["mean"].tolist(),
        "l15_train_std": d["std"].tolist(),
        "l15_device": DEVICE,
        "l15_torch": torch.__version__,
        "l15_torchvision": torchvision.__version__,
    }

    print("Lecture 15 — the data:")
    fig_grid(d)
    facts["l15_splits"] = fig_splits(d)
    fig_baseline(base)
    facts["l15_pooling_demo"] = fig_pooling()

    print("Lecture 15 — the architecture:")
    rows = layer_table()
    facts["l15_layers"] = rows
    facts["l15_n_params"] = sum(r["params"] for r in rows)
    facts["l15_dense_head_params"] = [r for r in rows
                                      if r["layer"] == "Linear"][0]["params"]
    facts["l15_conv_params"] = sum(r["params"] for r in rows
                                   if r["layer"] in ("Conv2d", "BatchNorm2d"))
    facts["l15_steps_per_epoch"] = -(-int(len(d["y_train"])) // BATCH)
    facts["l15_flatten_dim"] = [r for r in rows
                                if r["layer"] == "Flatten"][0]["activations"]
    facts["l15_total_steps"] = EPOCHS * facts["l15_steps_per_epoch"]
    facts["l16_resnet_params"] = int(n_params(torchvision.models.resnet18()))
    facts["l15_params_per_image"] = (facts["l15_n_params"]
                                     / len(d["y_train"]))
    facts["l15_conv_share_pct"] = (100 * facts["l15_conv_params"]
                                   / facts["l15_n_params"])
    facts["l15_head_share_pct"] = (100 * facts["l15_dense_head_params"]
                                   / facts["l15_n_params"])
    print(f"    {facts['l15_n_params']:,} parameters, "
          f"{facts['l15_dense_head_params']:,} of them in the first dense "
          f"layer")

    print(f"Lecture 15 — training from scratch ({EPOCHS} epochs, {DEVICE}):")
    scratch = cached("app08_scratch", lambda: train_scratch(d))
    fig_filters(scratch)
    fig_featmaps(scratch, d)
    fig_curve(scratch, base)
    facts["l15_gap"] = fig_gap(scratch)
    facts["l15_scratch"] = {
        "seconds": scratch["seconds"], "test_acc": scratch["test_acc"],
        "val_acc": scratch["val_acc"], "train_acc": scratch["train_acc"],
        "epochs": scratch["epochs"], "n_params": scratch["n_params"],
        "hist": scratch["hist"]}
    facts["l15_test_acc"] = scratch["test_acc"]
    facts["l15_seconds"] = scratch["seconds"]
    facts["l15_minutes"] = scratch["seconds"] / 60
    facts["l15_vs_majority"] = scratch["test_acc"] / base["majority"]
    print(f"    test {100 * scratch['test_acc']:.2f}%  in "
          f"{scratch['seconds']:.0f} s")

    print("Lecture 15 — the assistant's normalisation statistics:")
    leak = cached("app08_norm_leak", lambda: normalisation_leak(d))
    fig_leak(leak)
    facts["l15_leak"] = leak
    print(f"    honest {100 * leak['honest_mean']:.2f}%  "
          f"leaky {100 * leak['leaky_mean']:.2f}%  "
          f"gap {leak['gap_pts']:+.2f} pts against a seed spread of "
          f"{leak['seed_spread_pts']:.2f} pts")

    print("Thread 8 — a dense layer against a convolutional one:")
    dc = dense_vs_conv()
    fig_params(dc)
    facts["l16_dense_vs_conv"] = dc
    print(f"    {dc['dense_weights']:,} against {dc['conv_weights']:,} "
          f"— a factor of {dc['ratio']:,.0f}")

    print("Thread 8 — equivariance and invariance, measured:")
    eq = equivariance(scratch["state"], d)
    fig_equivariance(scratch, d, eq)
    facts["l16_equivariance"] = eq
    print(f"    max |conv(shift(x)) − shift(conv(x))| = "
          f"{eq['equivariance_max_abs']:.2e} on activations of scale "
          f"{eq['activation_scale']:.2f}")
    print(f"    cosine similarity after a {eq['shift_px']}px shift: "
          f"pooled {eq['cos_pooled']:.4f}, maps {eq['cos_maps']:.4f}")

    print("Thread 8 — where the memory goes:")
    mem = memory_budget()
    fig_memory(mem)
    facts["l16_memory_layers"] = fig_memory_layers(mem)
    facts["l16_memory"] = mem
    facts["l16_measured_memory"] = cached("app08_measured_mem", measured_memory)
    print(f"    parameters {mem['params_mb']:.1f} MB, "
          f"optimiser total {mem['optimizer_mb']:.1f} MB, "
          f"activations at batch {BATCH} {mem['act_at_batch_mb']:,.0f} MB "
          f"— {mem['ratio_at_batch']:.1f}x")
    mm = facts["l16_measured_memory"]
    if mm.get("available"):
        print(f"    allocator says the forward pass costs "
              f"{mm['forward_mb']:,.0f} MB at batch {BATCH}")

    print("Lecture 16 — the frozen backbone:")
    probe = cached("app08_probe", linear_probe)
    probe["params_per_image"] = probe["n_trainable"] / len(d["y_train"])
    facts["l16_probe"] = probe
    print(f"    test {100 * probe['test_acc']:.2f}%  in "
          f"{probe['seconds']:.0f} s "
          f"({probe['feature_seconds']:.0f} s of it extracting features)")

    print("Lecture 16 — fine-tuning layer4 with augmentation:")
    ft = cached("app08_finetune_v2", lambda: _strip(finetune()))
    facts["l16_finetune"] = ft
    print(f"    test {100 * ft['test_acc']:.2f}%  in {ft['seconds']:.0f} s")

    print("Lecture 16 — fine-tuning without augmentation:")
    ft_noaug = cached("app08_finetune_noaug_v2",
                      lambda: _strip(finetune(augmented=False)))
    facts["l16_finetune_noaug"] = ft_noaug
    print(f"    test {100 * ft_noaug['test_acc']:.2f}%  in "
          f"{ft_noaug['seconds']:.0f} s")

    fig_transfer(scratch, probe, ft)
    facts["l16_aug_val"] = fig_aug_val(ft)
    fig_augment()

    print("Lecture 16 — the assistant's augmented validation set:")
    wob = ft["wobble"]
    facts["l16_aug_wobble"] = wob
    print(f"    the same weights score {100 * min(wob['scores']):.2f}% to "
          f"{100 * max(wob['scores']):.2f}% — a spread of "
          f"{wob['spread_pts']:.2f} points")

    # the comparison the whole application exists to make
    facts["l16_comparison"] = {
        "scratch_acc": scratch["test_acc"], "scratch_seconds": scratch["seconds"],
        "probe_acc": probe["test_acc"], "probe_seconds": probe["seconds"],
        "ft_acc": ft["test_acc"], "ft_seconds": ft["seconds"],
        "acc_gain_pts": 100 * (ft["test_acc"] - scratch["test_acc"]),
        "acc_ratio": ft["test_acc"] / scratch["test_acc"],
        "probe_speedup": scratch["seconds"] / probe["seconds"],
        "ft_speedup": scratch["seconds"] / ft["seconds"],
        "aug_gain_pts": 100 * (ft["test_acc"] - ft_noaug["test_acc"]),
    }

    print("Diagrams:")
    diagram_weight_sharing()
    diagram_equivariance()
    diagram_transfer(probe, ft)
    diagram_memory()

    export(**facts)

    bad = validate_diagrams()
    if bad:
        print("\nmalformed diagrams:")
        for b in bad:
            print("  " + b)
        return 1

    problems = check_text_floor()
    if problems:
        print("\ntext floor:")
        for p in problems:
            print("  " + p)
        return 1
    print("\nall figures clear the 15px text floor; all d-*.svg parse as XML")
    return 0


if __name__ == "__main__":
    sys.exit(main())
