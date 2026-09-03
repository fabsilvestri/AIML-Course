"""
Lecture 13 — Transfer learning. Thread 8 measured, then the transfer
learning repair.

Exports build() -> list[nbformat cell]. Self-contained: it reloads and
re-decodes the data rather than assuming Lecture 15's kernel is still alive.

The deck quotes an 80-epoch from-scratch run and a 15-epoch fine-tune. This
notebook runs 20 and 8, so that a free Colab session finishes it, and says so
next to each. Every ratio the lecture turns on survives the shortening; the
absolute numbers are a little lower.
"""

from __future__ import annotations

import nbformat as nbf
from _prompt import prompt                                # noqa: E402


def md(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(text.strip("\n"))


def code(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(text.strip("\n"))


HEADER = """
# Transfer learning

**Lecture 13** · Géron, Chapter 12 · *Mathematical thread: weight
sharing, equivariance, and where the memory goes*

Applications of Machine Learning — BSc Mathematics of Artificial Intelligence

---

**How to use this notebook.** Read before you run. Every code cell is preceded
by the specification that would produce it — input, output, constraint, check.

Section 5 deliberately runs a broken evaluation, and says so before you reach
it: an augmented validation loader makes the score a random variable rather
than a function of the weights, and the only way to see that is to run it twice.
Nothing else here is wrong.

Runs on CPU. The backbone is small and mostly frozen, which is exactly why this
lecture is cheap and the previous one was not. The from-scratch comparison is
made between runs *in this notebook*, so it is internally consistent even though
the epoch counts are shorter than the deck's.
"""


def build() -> list:
    return [
        md(HEADER),

        md("## 1 · Setup and the same data"),
        prompt(
            label="setup",
            input="nothing",
            output="versions, seeds, device",
            constraint="the same seeds as the previous lecture",
            check="a notebook that exists to compare two approaches must fix everything except the approach.",
            **{"try": "set OMP_NUM_THREADS to 1, restart the kernel, and run "
                      "the notebook again. Every accuracy below is unchanged "
                      "and every wall clock grows. The lecture's headline is "
                      "that transfer learning is both faster and more "
                      "accurate — which half of it did you just make fragile?"}),
        code('''
# Not examinable, and only needed on some machines: PyTorch, numpy and
# torchvision can each end up loading their own OpenMP runtime, and with more
# than one loaded a training cell can deadlock -- no error, no output, and no
# CPU use. These have to be set BEFORE torch is imported, because they are read
# at import time and after that they do nothing.
import os
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# --- setup -------------------------------------------------------------------
import sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.datasets import Flowers102
from torchvision.models import resnet18, ResNet18_Weights

print(f"python       {sys.version.split()[0]}")
print(f"torch        {torch.__version__}")
print(f"torchvision  {torchvision.__version__}")

RANDOM_STATE = 42
torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
print(f"\\ndevice       {device}")
'''),

        md("""
Reloaded here rather than inherited. If your split differs from the previous
lecture's by one image, none of the comparisons below mean anything.

⏱ **about 40 seconds** — the dataset is already downloaded if you ran the
previous notebook; this decodes it at two resolutions, 128 for the network you
built and 224 for the pretrained one.
"""),
        prompt(
            label="⏱ 40 s — the same data, at TWO resolutions",
            input="Flowers102",
            output="every split decoded at 128 for your network and at 224 for the pretrained one",
            constraint="decode both sizes up front — the pretrained weights fix their input size, and resizing inside the loop would dominate the wall clock this lecture is measuring",
            check="assert the pair of tensors per split have equal length, and that the transfer tensors really are 224. Two resolutions means two tensors and two normalisations. Name them differently, because passing the wrong one raises nothing.",
            **{"try": "decode the transfer tensors at 128 instead of 224 and "
                      "re-run the frozen probe. It loses several points: the "
                      "pretrained filters were fitted at 224, and a receptive "
                      "field is measured in pixels, not in fractions of the "
                      "image."}),
        code('''
IMG, TRANSFER_IMG, N_CLASSES = 128, 224, 102

def load_split(split, size):
    tf = transforms.Compose([transforms.Resize((size, size)),
                             transforms.PILToTensor()])
    ds = Flowers102("datasets", split=split, download=True)
    x = torch.stack([tf(img) for img, _ in ds])
    y = torch.tensor([label for _, label in ds], dtype=torch.long)
    return x, y

t0 = time.perf_counter()
X_train, y_train = load_split("train", IMG)
X_val,   y_val   = load_split("val",   IMG)
X_test,  y_test  = load_split("test",  IMG)
T_train, _ = load_split("train", TRANSFER_IMG)
T_val,   _ = load_split("val",   TRANSFER_IMG)
T_test,  _ = load_split("test",  TRANSFER_IMG)
print(f"decoded in {time.perf_counter() - t0:.0f} s")

assert len(X_train) == len(T_train) == 1020
assert len(X_val)   == len(T_val)   == 1020
assert len(X_test)  == len(T_test)  == 6149
assert T_train.shape[-1] == TRANSFER_IMG

xf = X_train.float() / 255.0
MEAN, STD = xf.mean(dim=(0, 2, 3)), xf.std(dim=(0, 2, 3))
del xf

def normalise(x_u8, mean=MEAN, std=STD):
    return (x_u8.float() / 255.0 - mean[:, None, None]) / std[:, None, None]

counts_test = torch.bincount(y_test, minlength=N_CLASSES)
majority = float(counts_test.max()) / float(counts_test.sum())
print(f"majority baseline {majority:.2%}")
'''),

        prompt(
            label="the previous lecture's architecture, so this stands alone",
            input="nothing",
            output="the same network, and a batched accuracy function",
            constraint="assert the parameter count EXACTLY matches the previous lecture's 4,807,494 — a re-typed architecture that differs anywhere invalidates the comparison",
            check="the parameter-count assert is the check. When you duplicate code across notebooks, assert an invariant that pins it. A number is a better copy-check than reading.",
            **{"try": "change the first block's kernel from k=7 to k=3 and "
                      "re-run. The assert on 4,807,494 fires — that is what "
                      "it is for. It costs 3,840 weights, all of them in one "
                      "layer. Which layer holds most of this network's "
                      "parameters, and is it that one?"}),
        code('''
# the architecture from the previous lecture, so this notebook stands alone
def conv_block(c_in, c_out, k=3):
    return [nn.Conv2d(c_in, c_out, k, padding=k // 2, bias=False),
            nn.BatchNorm2d(c_out), nn.ReLU()]

def make_net():
    return nn.Sequential(
        *conv_block(3, 32, k=7), *conv_block(32, 32), nn.MaxPool2d(2),
        *conv_block(32, 64),     *conv_block(64, 64), nn.MaxPool2d(2),
        *conv_block(64, 128),  *conv_block(128, 128), nn.MaxPool2d(2),
        *conv_block(128, 256),                        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(256 * (IMG // 16) ** 2, 256), nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, N_CLASSES)
    )

@torch.no_grad()
def accuracy(model, X_u8, y, norm, bs=64):
    """Normalise one batch at a time.

    inorm(T_test) as a single tensor would be 6,149 x 3 x 224 x 224 float32 =
    3.5 GB, and that is how a Colab session dies. The uint8 tensor stays put
    and each batch is converted as it is needed.
    """
    model.eval()
    right = 0
    for k in range(0, len(X_u8), bs):
        xb = norm(X_u8[k:k + bs]).to(device)
        right += (model(xb).argmax(1).cpu() == y[k:k + bs]).sum().item()
    return right / len(X_u8)

n_scratch = sum(p.numel() for p in make_net().parameters())
assert n_scratch == 4_807_494
print(f"architecture matches the previous lecture: {n_scratch:,} parameters")
'''),

        md("""
## 2 · Why 15.3% is not an architecture problem

Start with what is *not* wrong: the loop, the metric, the split, the
architecture. Nothing here is a bug.

Retrain the from-scratch network briefly so this notebook has its own baseline
to compare against.

⏱ **a minute or so on CPU.** 20 epochs.
"""),
        prompt(
            label="⏱ ~1 min — the from-scratch baseline, rebuilt here",
            input="the previous lecture's network, 20 epochs",
            output="train, validation and test accuracy, the wall clock, and the gap",
            constraint="retrain it IN THIS NOTEBOOK, so the comparison below is between two runs on the same machine",
            check="the observation that motivates everything below: the first layer learned colour blobs and oriented edges, and none of them is about flowers. We spent 1,020 precious labels rediscovering something not specific to this problem.",
            **{"try": "raise EPOCHS_SCRATCH from 20 to 40. Training accuracy "
                      "climbs and test accuracy barely moves: on 1,020 images "
                      "more epochs buy memorisation, not generalisation, and "
                      "the gap you print is the receipt."}),
        code('''
EPOCHS_SCRATCH, LR, BATCH = 20, 3e-4, 32

torch.manual_seed(RANDOM_STATE)
scratch = make_net().to(device)
Xtr, ytr = normalise(X_train).to(device), y_train.to(device)
opt = torch.optim.Adam(scratch.parameters(), lr=LR)
lossf = nn.CrossEntropyLoss()
gen = torch.Generator(device=device).manual_seed(RANDOM_STATE)

t0 = time.perf_counter()
for ep in range(EPOCHS_SCRATCH):
    scratch.train()
    perm = torch.randperm(len(Xtr), device=device, generator=gen)
    for k in range(0, len(Xtr), BATCH):
        idx = perm[k:k + BATCH]
        opt.zero_grad()
        lossf(scratch(Xtr[idx]), ytr[idx]).backward()
        opt.step()
SCRATCH_SECONDS = time.perf_counter() - t0

SCRATCH_TEST = accuracy(scratch, X_test, y_test, normalise)
TRAIN_ACC = accuracy(scratch, X_train, y_train, normalise)
VAL_ACC   = accuracy(scratch, X_val, y_val, normalise)
print(f"train {TRAIN_ACC:.2%}")
print(f"val   {VAL_ACC:.2%}")
print(f"test  {SCRATCH_TEST:.2%}   in {SCRATCH_SECONDS:.0f} s")
print(f"\\ngap: {100*(TRAIN_ACC - VAL_ACC):.0f} points")
'''),
        prompt(
            label="what weight sharing bought, in weights",
            input="the first convolution's shape, and the dense layer it replaced",
            output="both weight counts and the ratio",
            constraint="count the DENSE layer that would produce the same output map — 128x128x3 inputs to 128x128x32 outputs — rather than an arbitrary dense layer, so the comparison is between two ways of computing the same thing",
            check="the ratio is the whole argument for convolution and it is one division. Compute it rather than quoting it.",
            **{"try": "set K = 3 and re-run. The ratio grows by (7/3)^2, "
                      "about 5.4x, because the dense count does not depend on "
                      "the kernel at all. Then set C_OUT = 3: the saving "
                      "survives, so it is a property of sharing rather than "
                      "of the layer being small."}),
        code('''
H, W, C_IN, C_OUT, K = 128, 128, 3, 32, 7

n_in  = H * W * C_IN
n_out = H * W * C_OUT
dense_weights = n_in * n_out                 # every input to every output
conv_weights  = C_OUT * C_IN * K * K         # one kernel, reused everywhere

print(f"a dense layer doing the same job: {dense_weights:,} weights")
print(f"the convolution:                  {conv_weights:,} weights")
print(f"ratio:                            {dense_weights / conv_weights:,.0f}x")
'''),

        md("""
A persistent gap between two plateaus is **variance** — Lecture 6. Its two
cures are more data and more constraint. We cannot buy labels, so the
constraint has to come from somewhere.

**The observation from the previous lecture:** the first layer learned colour
blobs and oriented edges. Which of those is about *flowers*? None of them. We
spent our 1,020 precious labels rediscovering something that is not specific to
this problem at all.
"""),

        md("""
## 3 · A backbone somebody else trained

ImageNet: 1.28 million labelled photographs, 1,000 classes, none of them our
species. Somebody already paid for edges, colours, textures and parts. Those
weights are a download.
"""),
        prompt(
            label="a backbone somebody else trained",
            input="nothing",
            output="the weights' own declared preprocessing, and the parameter count",
            constraint="print `weights.transforms()` — the weights come WITH their preprocessing, and it is not the one you computed from the flowers",
            check="every pretrained checkpoint states its preprocessing. Read it and use it — it is metadata, not a suggestion.",
            **{"try": "swap resnet18 for resnet34 in both calls. The "
                      "parameter count roughly doubles and "
                      "`weights.transforms()` prints the same 224, the same "
                      "crop and the same normalisation. Which of those two "
                      "numbers actually constrains how the rest of this "
                      "notebook is written?"}),
        code('''
weights = ResNet18_Weights.DEFAULT
print(weights.transforms())
print(f"\\nparameters: {sum(p.numel() for p in resnet18().parameters()):,}")

first = resnet18(weights=weights).conv1.weight
print(f"first layer: {first.shape[0]} filters of {first.shape[1]}x"
      f"{first.shape[2]}x{first.shape[3]} = {first.numel():,} numbers")
print("every one of them a fact about photographs, not about flowers")
'''),
        md("""
**The weights come with their own preprocessing.** Use those statistics, not
the ones you computed from the flowers. The first layer expects inputs on the
scale it was trained with, and substituting your own is a silent, uncrashing
degradation.
"""),
        prompt(
            label="ImageNet normalisation, kept separate",
            input="the transfer-resolution tensors",
            output="a second normalise function",
            constraint="a SEPARATE function with a different name — the notebook now has two normalisations and passing the wrong one raises nothing",
            check="assert the output shape. When two incompatible preprocessings coexist, give them different names and pass them explicitly. `norm=` as a parameter of `accuracy` exists for exactly this.",
            **{"try": "pass `normalise` — the flowers' own statistics — "
                      "instead of `inorm` when the features are extracted "
                      "below. No shape is wrong, nothing raises, and the "
                      "probe loses accuracy. That silence is the reason the "
                      "two functions have different names."}),
        code('''
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225])[:, None, None]

def inorm(x_u8):
    return (x_u8.float() / 255.0 - IMAGENET_MEAN) / IMAGENET_STD

assert inorm(T_train[:2]).shape == (2, 3, TRANSFER_IMG, TRANSFER_IMG)
print("ImageNet normalisation ready")
'''),

        md("""
### Step 1 · freeze everything, replace the head

Note the order: `fc` is replaced **after** the freezing loop, because a newly
constructed module has `requires_grad=True`. That is not an accident of
style.
"""),
        prompt(
            label="freeze everything, then replace the head",
            input="the pretrained network",
            output="the trainable and frozen parameter counts",
            constraint="replace `fc` AFTER the freezing loop — a newly constructed module has requires_grad=True, and freezing then replacing is the only order that leaves the head trainable",
            check="assert the trainable count is exactly 512·102 + 102. Assert the trainable parameter count against an arithmetic expression. It is the only thing that distinguishes the two orders.",
            **{"try": "move the `net_t.fc = nn.Linear(...)` line ABOVE the "
                      "freezing loop and re-run. The assert on 52,326 fires "
                      "because the new head was frozen along with everything "
                      "else. Without it, the only symptom is a loss that "
                      "never moves."}),
        code('''
torch.manual_seed(RANDOM_STATE)
net_t = resnet18(weights=weights)

for p in net_t.parameters():
    p.requires_grad = False

net_t.fc = nn.Linear(512, N_CLASSES)
net_t = net_t.to(device)

n_train_p = sum(p.numel() for p in net_t.parameters() if p.requires_grad)
n_froz_p  = sum(p.numel() for p in net_t.parameters() if not p.requires_grad)
print(f"trainable {n_train_p:,}   frozen {n_froz_p:,}")
assert n_train_p == 512 * N_CLASSES + N_CLASSES
'''),

        md("""
### If nothing before the head is training, run it once

The 1,020 images pass through the backbone once rather than once per epoch.

⏱ **about 25 seconds** — the test split is 6,149 images at 224 × 224.
"""),
        prompt(
            label="⏱ 25 s — if nothing before the head trains, run it once",
            input="all three splits through the frozen backbone",
            output="512-dimensional features per image",
            constraint="`.eval()` on the body and `no_grad()` on the extraction — the backbone has batch norms whose running statistics would otherwise be updated by the extraction pass",
            check="assert the training features are (1020, 512). A frozen prefix is a fixed function. Compute it once and cache the output — that is not an optimisation, it is the definition of frozen.",
            **{"try": "drop the `.eval()` from `body` and re-run the "
                      "extraction and the probe. The backbone's batch norms "
                      "now update their running statistics on your 1,020 "
                      "flowers, so the three splits are encoded by three "
                      "slightly different functions and the test accuracy "
                      "falls. Nothing raises."}),
        code('''
body = nn.Sequential(*list(net_t.children())[:-1]).eval()

@torch.no_grad()
def features(X_u8, bs=64):
    out = []
    for k in range(0, len(X_u8), bs):
        out.append(body(inorm(X_u8[k:k + bs]).to(device)).flatten(1).cpu())
    return torch.cat(out)

t0 = time.perf_counter()
F_train, F_val, F_test = features(T_train), features(T_val), features(T_test)
FEATURE_SECONDS = time.perf_counter() - t0

assert F_train.shape == (1020, 512), F_train.shape
print(f"{FEATURE_SECONDS:.0f} s   features {tuple(F_train.shape)}")
'''),
        prompt(
            label="the head, on cached features",
            input="the 512-dimensional features",
            output="the probe's test accuracy and its total wall clock",
            constraint="count the FEATURE EXTRACTION time in the total — the probe is not free just because the head is",
            check="the variance problem was not solved. It was MOVED to a dataset large enough to absorb it.",
            **{"try": "raise the head's weight_decay from 1e-4 to 1e-1 and "
                      "re-run. Training accuracy falls a long way and test "
                      "accuracy far less. With 512 fixed features and 1,020 "
                      "labels, how much of the training fit was worth having?"}),
        code('''
torch.manual_seed(RANDOM_STATE)
head = nn.Linear(512, N_CLASSES).to(device)
opt = torch.optim.Adam(head.parameters(), lr=1e-3, weight_decay=1e-4)
F_train_d = F_train.to(device)
g = torch.Generator(device=device).manual_seed(RANDOM_STATE)

t0 = time.perf_counter()
for ep in range(60):
    head.train()
    perm = torch.randperm(len(F_train_d), device=device, generator=g)
    for k in range(0, len(F_train_d), BATCH):
        idx = perm[k:k + BATCH]
        opt.zero_grad()
        lossf(head(F_train_d[idx]), ytr[idx]).backward()
        opt.step()
HEAD_SECONDS = time.perf_counter() - t0

head.eval()
with torch.no_grad():
    PROBE_TEST = (head(F_test.to(device)).argmax(1).cpu()
                  == y_test).float().mean().item()
PROBE_SECONDS = FEATURE_SECONDS + HEAD_SECONDS

print(f"frozen backbone + linear head:  {PROBE_TEST:.2%}  "
      f"in {PROBE_SECONDS:.0f} s")
print(f"from scratch:                   {SCRATCH_TEST:.2%}  "
      f"in {SCRATCH_SECONDS:.0f} s")
'''),
        md("""
The 52,326 trainable parameters are fitted with 1,020 examples — about 51
parameters per image, against 4,713 before. The 11,176,512 frozen ones were
fitted with 1.28 million examples, by somebody else.

The variance problem was not solved. It was moved to a dataset large enough to
absorb it.
"""),

        md("""
### Step 2 · let the last block move, at a different learning rate

The early layers hold edges and colours, which are not about flowers. The late
layers hold parts and textures, which partly are.
"""),
        prompt(
            label="augmentation, as arithmetic",
            input="one image repeated seven times",
            output="seven random crops and flips, beside the original",
            constraint="written on tensors rather than as a transform pipeline, so the operation is visible as arithmetic rather than as a class name",
            check="look at the augmented images before training on them. A crop scale that occasionally excludes the subject is a labelled photograph of a leaf.",
            **{"try": "widen the crop scale from `0.55 + 0.45 * rand` to `0.2 "
                      "+ 0.8 * rand` and look at the grid again before you "
                      "train on it. Some crops no longer contain the flower, "
                      "and you have just added mislabelled images to the "
                      "training set on purpose."}),
        code('''
def augment(x_u8, gen):
    """Random resized crop and a horizontal flip, on a uint8 batch.

    Written on tensors rather than as a transform pipeline so that the
    operation is visible as arithmetic rather than as a class name.
    """
    n, size = len(x_u8), x_u8.shape[-1]
    out = torch.empty_like(x_u8)
    scales = 0.55 + 0.45 * torch.rand(n, generator=gen)
    for i in range(n):
        s = int(size * scales[i].item())
        top  = int(torch.randint(0, size - s + 1, (1,), generator=gen))
        left = int(torch.randint(0, size - s + 1, (1,), generator=gen))
        crop = x_u8[i:i+1, :, top:top+s, left:left+s].float()
        crop = F.interpolate(crop, size=size, mode="bilinear",
                             align_corners=False)
        if torch.rand(1, generator=gen).item() < 0.5:
            crop = crop.flip(-1)
        out[i] = crop[0].clamp(0, 255).to(torch.uint8)
    return out

gen_cpu = torch.Generator().manual_seed(RANDOM_STATE)
views = augment(T_train[7:8].repeat(7, 1, 1, 1), gen_cpu)
fig, axes = plt.subplots(1, 8, figsize=(14, 2.0))
axes[0].imshow(T_train[7].permute(1, 2, 0).numpy()); axes[0].set_title("original",
                                                                       fontsize=9)
for j in range(7):
    axes[j+1].imshow(views[j].permute(1, 2, 0).numpy())
for ax in axes:
    ax.axis("off")
plt.tight_layout(); plt.show()
print("the label is the same in all eight — that is the only thing that makes "
      "this legitimate")
'''),

        md("""
⏱ **a few minutes on CPU.** 8 epochs of fine-tuning at 224 × 224, plus a
validation pass each epoch. Most of the backbone is frozen, which is what keeps
this affordable without an accelerator.

Three things happen in this cell that did not happen before, and each one is a
line you should be able to defend:

1. `layer4` is unfrozen; everything before it is not
2. two parameter groups, two learning rates, one optimiser
3. the frozen batch-norm layers are put in `eval()` — freezing weights does not
   freeze running statistics, because those are buffers updated in the
   **forward** pass
"""),
        prompt(
            label="⏱ 90 s — fine-tune the last block",
            input="the pretrained network with layer4 and fc unfrozen",
            output="the validation curve, clean and augmented, and the test accuracy",
            constraint="THREE things that did not happen before, each defensible: layer4 unfrozen and nothing earlier; two parameter groups at two learning rates in one optimiser; and the frozen batch-norms put in eval() — freezing weights does NOT freeze running statistics, because those are buffers updated in the forward pass",
            check="1e-4 on the pretrained block and 1e-3 on the random head. The head starts from noise and the block starts from something good; one learning rate for both damages whichever it is wrong for.",
            **{"try": "give both parameter groups lr=1e-3 and re-run. The "
                      "random head does not mind; layer4 does. A block that "
                      "already knows something, taking steps sized for noise, "
                      "unlearns it — watch the test accuracy fall back toward "
                      "the from-scratch run."}),
        code('''
FT_EPOCHS = 8

torch.manual_seed(RANDOM_STATE)
ft = resnet18(weights=weights)
ft.fc = nn.Linear(512, N_CLASSES)
ft = ft.to(device)

for p in ft.parameters():
    p.requires_grad = False
for p in ft.layer4.parameters():
    p.requires_grad = True
for p in ft.fc.parameters():
    p.requires_grad = True

n_ft_train = sum(p.numel() for p in ft.parameters() if p.requires_grad)
n_ft_froz  = sum(p.numel() for p in ft.parameters() if not p.requires_grad)
print(f"fine-tuning: {n_ft_froz:,} parameters never move, {n_ft_train:,} do")

opt = torch.optim.Adam([
    {"params": ft.layer4.parameters(), "lr": 1e-4},   # already good
    {"params": ft.fc.parameters(),     "lr": 1e-3},   # random
])
gen = torch.Generator().manual_seed(RANDOM_STATE)
# A SECOND generator, for the diagnostic only. Sharing one would mean the
# augmented-validation line below consumed draws from the same stream that
# shuffles the training batches — so the training data order would depend on
# whether the diagnostic ran at all. Harmless with a fixed seed and one
# configuration; it is also exactly the coupling this course asks you to find.
gen_val = torch.Generator().manual_seed(RANDOM_STATE + 1)

clean_curve, aug_curve = [], []
t0 = time.perf_counter()
for ep in range(FT_EPOCHS):
    ft.train()
    for m_ in [ft.bn1, ft.layer1, ft.layer2, ft.layer3]:
        m_.eval()                          # freeze the buffers too
    perm = torch.randperm(len(T_train), generator=gen)
    for k in range(0, len(T_train), BATCH):
        idx = perm[k:k + BATCH]
        xb = inorm(augment(T_train[idx], gen)).to(device)
        opt.zero_grad()
        lossf(ft(xb), ytr[idx]).backward()
        opt.step()
    clean_curve.append(accuracy(ft, T_val, y_val, inorm))
    aug_curve.append(accuracy(ft, augment(T_val, gen_val), y_val, inorm))
    print(f"epoch {ep+1}  val {clean_curve[-1]:.3f}  "
          f"({time.perf_counter()-t0:.0f} s)")

FT_SECONDS = time.perf_counter() - t0
FT_TEST = accuracy(ft, T_test, y_test, inorm)
print(f"\\nfine-tuned: {FT_TEST:.2%} in {FT_SECONDS:.0f} s")
'''),

        md("""
## 4 · The comparison this lecture exists to make
"""),
        prompt(
            label="the comparison this application exists to make",
            input="all five results",
            output="accuracy and wall clock for each",
            constraint="wall clock in the SAME column — the frozen probe's headline is that it is both faster and more accurate, and one of those is invisible without the time",
            check="assert the fine-tune beat the from-scratch run, with a message pointing at the normalisation if it did not. The assert names the likely cause of failure. `check inorm` is worth more than `assert FT_TEST > SCRATCH_TEST` alone.",
            **{"try": "cover the wall-clock column with your hand and read "
                      "the table again. The frozen probe's whole argument — "
                      "nearly the accuracy, a fraction of the cost — is "
                      "invisible, and the fine-tune looks like the only "
                      "sensible choice. A results table without a cost column "
                      "recommends the wrong model."}),
        code('''
rows = [("uniform guess",            1 / N_CLASSES,  None),
        ("commonest species",        majority,       None),
        ("convolutional net, scratch", SCRATCH_TEST, SCRATCH_SECONDS),
        ("frozen backbone + head",   PROBE_TEST,     PROBE_SECONDS),
        ("fine-tuned + augmented",   FT_TEST,        FT_SECONDS)]

for name, acc, secs in rows:
    t = "—" if secs is None else f"{secs:6.0f} s"
    print(f"{name:30s} {acc:8.2%}  {t}")

print(f"\\naccuracy gain over from-scratch: "
      f"{100*(FT_TEST - SCRATCH_TEST):+.1f} points")
print(f"frozen probe was {SCRATCH_SECONDS / PROBE_SECONDS:.1f}x faster than "
      f"training from scratch, and far more accurate")

assert FT_TEST > SCRATCH_TEST, "transfer learning did not help — check inorm"
'''),

        md("""
## 5 · One transform, two loaders

Augmentation is a **training-time** transform. It is easy to build one pipeline
and hand it to both loaders — the code below does, and it runs, trains, and its
validation accuracy climbs over epochs exactly as it should.

The failure is that the evaluation is no longer a function of the weights: score
the same fixed model twice and you get two different numbers. Measure it.

```python
tf = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.55, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

train_ds = Flowers102("datasets", split="train", transform=tf)
val_ds   = Flowers102("datasets", split="val",   transform=tf)   # <--
```

### Reviewer question 5: what is the default I did not ask for?

One `tf` for both splits. Nothing is *leaking* — the model never trains on the
validation set. The damage is entirely in what the validation number now
**means**.
"""),
        prompt(
            label="one transform for two splits, and what it costs",
            input="'add random resized crops and horizontal flips with ImageNet normalisation, and build the training and validation dataloaders'",
            output="one fixed set of weights scored ten times on the AUGMENTED validation set",
            constraint="score the same weights repeatedly — the wobble is the whole diagnostic",
            check="assert that the CLEAN evaluation is deterministic, so the wobble is attributable to the augmentation and not to eval mode. Lecture 12's reviewer question arriving again: a deterministic function of fixed weights and fixed data does not wobble.",
            **{"try": "change the seed in `gen_w` and re-run. The clean "
                      "number is identical and all ten augmented numbers "
                      "move. A score that depends on the evaluator's seed is "
                      "a property of the evaluation, not of the model."}),
        code('''
# score ONE fixed set of weights on the augmented validation set, ten times
gen_w = torch.Generator().manual_seed(RANDOM_STATE)
clean = accuracy(ft, T_val, y_val, inorm)
scores = [accuracy(ft, augment(T_val, gen_w), y_val, inorm)
          for _ in range(10)]

print(f"clean validation set        {clean:.2%}")
print(f"augmented, mean of 10       {np.mean(scores):.2%}")
print(f"augmented, min to max       {min(scores):.2%} to {max(scores):.2%}")
print(f"spread                      {100*(max(scores)-min(scores)):.2f} points")

assert accuracy(ft, T_val, y_val, inorm) == clean, \\
    "even the clean evaluation is not deterministic"
'''),
        md("""
The same weights, the same set, several points apart. Lecture 12's reviewer
question arriving again: **a deterministic function of fixed weights and fixed
data does not wobble.**

And it does not only make the number noisy — it selects a different model.
"""),
        prompt(
            label="and it selects a different model",
            input="both validation curves over the fine-tuning run",
            output="the two curves, and the epoch each would early-stop on",
            constraint="report the chosen EPOCH under each rule, not just the curves — the point is that the bug changes which weights you keep",
            check="the corrected specification's last sentence is the part that generalises: assert that evaluating the same model twice gives identical numbers. One line, one second, and it catches this bug, a missing eval(), and any accidental shuffling of the labels.",
            **{"try": "smooth the augmented curve with a three-epoch moving "
                      "average before taking its argmax, and compare the "
                      "chosen epoch with `best_clean` again. Which of the two "
                      "curves does the smoothing move, and what does that "
                      "tell you about whether the bug is bias or variance?"}),
        code('''
fig, ax = plt.subplots(figsize=(9, 3.2))
ep = range(1, FT_EPOCHS + 1)
ax.plot(ep, [100*v for v in clean_curve], label="validation, as it should be")
ax.plot(ep, [100*v for v in aug_curve], "--", label="validation, augmented too")
ax.set_xlabel("epoch"); ax.set_ylabel("accuracy, %"); ax.set_ylim(0, 100)
ax.legend(); plt.tight_layout(); plt.show()

best_clean = int(np.argmax(clean_curve)) + 1
best_aug   = int(np.argmax(aug_curve)) + 1
print(f"early stopping on the clean curve keeps epoch     {best_clean}")
print(f"early stopping on the augmented curve keeps epoch {best_aug}")
print(f"reported number is {100*(clean_curve[-1]-aug_curve[-1]):.1f} points "
      f"pessimistic")
'''),
        md("""
### The corrected specification

> *"Build **two** transform pipelines: a training one with random resized crop
> and horizontal flip, and a deterministic evaluation one with resize and
> centre crop. Use the second for validation and test. Assert that evaluating
> the same model on the validation set twice gives identical numbers."*

The assertion is the part that generalises. It costs one line and a second of
wall clock, and it catches this bug, the missing `eval()`, and any accidental
shuffling of the labels.
"""),
        prompt(
            label="the one-line check that generalises",
            input="the same model, the same data, twice",
            output="the assertion, and the number printed once",
            constraint="assert EXACT equality — floating-point evaluation of a fixed function on fixed data is bit-identical, and any tolerance here would hide the thing being tested",
            check="put this line at the bottom of every notebook that evaluates a model. It costs a second and it is the highest-yield assertion in the course.",
            **{"try": "add `ft.train()` above the two calls, and score a "
                      "shuffled copy of the validation set as well. The batch "
                      "norms now normalise by each batch's own statistics, so "
                      "the answer depends on which images happen to share a "
                      "batch. The assert fires, and not one weight changed."}),
        code('''
a = accuracy(ft, T_val, y_val, inorm)
b = accuracy(ft, T_val, y_val, inorm)
assert a == b, f"evaluation is not deterministic: {a} vs {b}"
print(f"evaluation is deterministic: {a:.4f} twice")
'''),

        md("""
## 6 · Where we are

- 1,020 images could not pay for 4.8 million parameters. A backbone trained on
  a far larger corpus already encodes what the bottom of that network was trying
  to learn.
- A **frozen probe** is the measurement that decides everything else: if the
  frozen features barely beat the trivial baseline, the backbone does not
  understand your images and fine-tuning will not hide it.
- Freezing weights does not freeze a network — batch normalisation keeps
  updating its running statistics in the forward pass until you call `eval()`.
- Every augmentation asserts an invariance of the task. A flipped flower is a
  flower; a flipped digit is not.

**Five questions to ask of any transfer pipeline**, with this application's
specific answers:

| | |
|---|---|
| 1 | normalisation statistics computed over all 8,189 images |
| 2 | the backbone was fitted on ImageNet, not on your data — say so when you report |
| 3 | 224, not 128: the pretrained weights fix the input size |
| 4 | nothing is dropped here — count anyway, and say zero |
| 5 | `padding=0`, `train()` mode on frozen batch-norms, one transform for two splits |

Report what you **found**, not what you would have done differently.

### One line to keep

Before you train a vision model from scratch, spend the seconds it takes to
measure what a frozen pretrained backbone and a linear head already give you.
You are entitled to train from scratch. You are not entitled to do it without
knowing what you turned down.
"""),
    ]
