"""
Lecture 16 — Don't train from scratch. Thread 8 measured, then the transfer
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
# Don't train from scratch

**Lecture 16 · Fix** · Géron, Chapter 12 · *Mathematical thread: weight
sharing, equivariance, and where the memory goes*

Applications of Machine Learning — BSc Mathematics of Artificial Intelligence

---

**How to use this notebook.** Read before you run. Cells marked
**⚠ read before running** contain a defect on purpose.

Sections 2 to 5 are the mathematical thread, and every claim in them is
*measured* here rather than asserted. Sections 6 onwards are the repair.

The deck's from-scratch number comes from 80 epochs; this notebook runs 20, and
the fine-tune runs 8 rather than 15. The comparison is made between runs in
this notebook, so it is internally consistent.
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
            left_open="that `torch.nn.functional` is imported here and was not before. The thread measures cosine similarities and pooling directly, outside any module.",
            student="changing the seed between the from-scratch run and the transfer run. Every headline in this notebook is a difference of two accuracies.",
            catch="a notebook that exists to compare two approaches must fix everything except the approach."),
        code('''
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
            check="assert the pair of tensors per split have equal length, and that the transfer tensors really are 224",
            left_open="that 224 is not a preference. It is what the ResNet weights were trained at, and it is red-team answer 3 for this application.",
            student="feeding 128-pixel images to the pretrained network. It runs — ResNet is fully convolutional up to the pool — and the features are computed at a scale the weights never saw.",
            catch="two resolutions means two tensors and two normalisations. Name them differently, because passing the wrong one raises nothing."),
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
            check="the parameter-count assert is the check",
            left_open="why `accuracy` normalises one batch at a time. `inorm(T_test)` as a single tensor is 6,149 × 3 × 224 × 224 float32 = 3.5 GB, and that is how a Colab session dies.",
            student="re-typing the architecture from memory and getting one channel count wrong. The parameter assert catches it in one second; the accuracy comparison would not catch it at all.",
            catch="when you duplicate code across notebooks, assert an invariant that pins it. A number is a better copy-check than reading."),
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
        nn.Linear(256, N_CLASSES),
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

assert sum(p.numel() for p in make_net().parameters()) == 4_807_494
print("architecture matches the previous lecture")
'''),

        md("""
## 2 · Thread 8, part one — the parameter count

Fix the input and the output shape so the comparison is between two ways of
computing *the same sized thing*: a `3 × 128 × 128` input and a
`32 × 128 × 128` output.
"""),
        prompt(
            label="thread 8 — the parameter count",
            input="a 3×128×128 input and a 32×128×128 output",
            output="the weight count for a dense layer and for a convolution computing the same sized thing",
            constraint="fix the input AND output shapes so the comparison is between two ways of computing the SAME object",
            check="assert the convolutional count is exactly 32·3·7·7, and that the ratio exceeds five million",
            left_open="that H and W do not appear in the convolutional count. Not approximately absent — genuinely absent.",
            student="concluding convolutions save arithmetic. They do not: the convolution still computes C_out·H·W outputs, each a sum over C_in·k² terms. What shrank is the number of DISTINCT numbers that have to be stored and learned.",
            catch="print the dense layer's size in gigabytes. 25 GB of float32 for one layer is the kind of number that ends an argument."),
        code('''
H = W = IMG
n_in, n_out = 3 * H * W, 32 * H * W

dense_weights = n_in * n_out                 # every output to every input
conv_weights  = 32 * 3 * 7 * 7               # one 7x7x3 kernel per filter

print(f"inputs                {n_in:>18,}")
print(f"outputs               {n_out:>18,}")
print(f"dense layer weights   {dense_weights:>18,}")
print(f"conv layer weights    {conv_weights:>18,}")
print(f"\\nratio                 {dense_weights / conv_weights:>18,.0f}")
print(f"dense layer, float32  {dense_weights * 4 / 2**30:>18,.0f} GB")

# the conv count does not contain H or W. That is the whole point.
assert conv_weights == 32 * 3 * 7 * 7
assert dense_weights // conv_weights > 5_000_000
'''),
        md("""
`H` and `W` do not appear in the convolutional count. That is not an
approximation — the image size is genuinely absent.

**What the saving is not:** it is not a saving in arithmetic. The convolution
still computes `C_out · H · W` outputs, each a sum over `C_in · k²` terms.
What shrank is the number of *distinct numbers that have to be stored and
learned*.
"""),

        md("""
## 3 · Thread 8, part two — equivariance, measured

Let `T_v` shift an image by `v`. A map `f` is **equivariant** when
`f(T_v x) = T_v f(x)`: shift then compute, or compute then shift, gives the
same answer.

Convolution satisfies this because the kernel does not depend on position —
which is exactly what weight sharing is. Untie the weights and the proof's
middle step fails.

We check it on a randomly initialised convolution, because the property is of
the *operation* and not of the training.
"""),
        prompt(
            label="equivariance, measured",
            input="one image and the same image shifted by 16 pixels",
            output="the largest difference between shift-then-convolve and convolve-then-shift, on the interior",
            constraint="drop the border before comparing — zero padding invents input that was not there, and `roll` wraps, so both edges violate the identity for reasons that are not about equivariance",
            check="assert the relative difference is below 1e-5",
            left_open="that this is checked on a RANDOMLY INITIALISED convolution, deliberately. The property belongs to the operation and not to the training.",
            student="comparing the full tensors, seeing a large maximum difference at the edge, and concluding convolutions are not equivariant.",
            catch="the residue that survives is float32 rounding, not mathematics. Say which one you are looking at, and give the relative figure rather than the absolute."),
        code('''
torch.manual_seed(RANDOM_STATE)
conv = nn.Conv2d(3, 32, 7, padding=3, bias=False).eval()

x  = normalise(X_test[:1])
SHIFT = 16
xs = torch.roll(x, SHIFT, dims=3)

with torch.no_grad():
    y_of_shift = conv(xs)                      # f(T x)
    shift_of_y = torch.roll(conv(x), SHIFT, dims=3)   # T f(x)

# drop the border, where zero padding invents input, and the wrap seam
m = SHIFT + 8
a = y_of_shift[..., m:-m, m:-m]
b = shift_of_y[..., m:-m, m:-m]

print(f"largest |f(Tx) - Tf(x)| on the interior  {(a - b).abs().max():.3e}")
print(f"largest activation there                {b.abs().max():.3f}")
print(f"relative                                "
      f"{(a - b).abs().max() / b.abs().max():.3e}")

assert (a - b).abs().max() / b.abs().max() < 1e-5, "not equivariant"
print("\\nthat residue is float32 rounding, not mathematics")
'''),
        md("""
The equality is exact on the interior and **not** at the border: zero padding
invents input that was not there, so a shift moves real content into invented
content. Check that too, rather than taking the caveat on trust.
"""),
        prompt(
            label="and the border, where it fails",
            input="the same two tensors, at the corner",
            output="the difference there",
            constraint="check the caveat rather than stating it — the previous cell's assert is only meaningful if the excluded region really is different",
            left_open="why padding breaks it. A shift moves real content into invented content, so the two orders of operations no longer agree.",
            student="taking the interior-only caveat on trust. It is one cell to demonstrate, and it turns a hedge into a measurement.",
            catch="when an identity holds only on part of the domain, measure it on the other part too. Orders of magnitude larger is the evidence that your exclusion was necessary rather than convenient."),
        code('''
edge_a = y_of_shift[..., :4, :4]
edge_b = shift_of_y[..., :4, :4]
print(f"largest difference at the border         "
      f"{(edge_a - edge_b).abs().max():.3e}")
print("orders of magnitude larger — the identity is an interior statement")
'''),

        md("""
## 4 · Thread 8, part three — invariance is not equivariance

A map `g` is **invariant** when `g(T_v x) = g(x)`: the output does not change
at all. Invariance is strictly stronger and strictly lossier — it is what you
get by *discarding* the equivariant structure.

Pooling is what performs the discarding. Measure both representations of the
same shifted image.
"""),
        prompt(
            label="invariance is not equivariance",
            input="the feature maps of an image and its shift, before and after global pooling",
            output="the cosine similarity of each pair",
            constraint="compare the SAME quantity before and after pooling — cosine similarity on flattened maps and on the pooled vectors",
            check="assert pooling increased the similarity, which is what buying invariance means",
            left_open="why classification wants this and per-pixel prediction cannot afford it. 'This is a sunflower' is true wherever the sunflower is; segmentation asks for every pixel which class it is, and the answer IS the position.",
            student="conflating the two words. Invariance is strictly stronger and strictly lossier — it is what you get by DISCARDING the equivariant structure.",
            catch="a cosine similarity of 0.98 and one of 0.999 sound alike. Print 1 − cos as a percentage and they do not."),
        code('''
with torch.no_grad():
    maps      = F.relu(conv(x))
    maps_s    = F.relu(conv(xs))
    pooled    = F.adaptive_max_pool2d(maps,   1).flatten()
    pooled_s  = F.adaptive_max_pool2d(maps_s, 1).flatten()

cos_maps   = F.cosine_similarity(maps.flatten(), maps_s.flatten(), dim=0).item()
cos_pooled = F.cosine_similarity(pooled, pooled_s, dim=0).item()

print(f"cosine similarity, spatial maps   {cos_maps:.4f}")
print(f"cosine similarity, global pooled  {cos_pooled:.4f}")
print(f"\\nthe map changed by    {100*(1-cos_maps):.1f}%")
print(f"the pooled vector by  {100*(1-cos_pooled):.2f}%")

assert cos_pooled > cos_maps, "pooling did not buy invariance"
'''),
        md("""
**Why classification wants this.** "This is a sunflower" is true wherever the
sunflower is, so a representation that still carries the position is carrying a
nuisance variable.

**Why per-pixel prediction cannot afford it.** Segmentation asks *for every
pixel, which class is it?* — the answer **is** the position.
"""),
        prompt(
            label="what pooling costs in resolution",
            input="the network and a dummy input",
            output="the last feature map size, and how many input pixels one cell answers for",
            constraint="find the last pooling output by walking the network, not by arithmetic — four MaxPool2d layers is easy to miscount",
            check="assert the final grid is IMG // 16",
            left_open="that two flower boundaries fifteen pixels apart are the same cell. Lecture 18 has to put that resolution back, and its whole architecture is about how.",
            student="assuming the resolution loss is harmless because classification works. It is harmless FOR CLASSIFICATION, which is the point being made.",
            catch="express it as input pixels per output cell. '256x resolution lost' is abstract; 'one cell answers for 256 pixels' is not."),
        code('''
net = make_net()
z = torch.zeros(1, 3, IMG, IMG)
for m_ in net:
    z = m_(z)
    if isinstance(m_, nn.MaxPool2d):
        last = z.shape[-1]

print(f"input grid          {IMG} x {IMG}")
print(f"last feature map    {last} x {last}")
print(f"one cell answers for {(IMG // last) ** 2} input pixels")
print(f"spatial resolution lost: {(IMG * IMG) / (last * last):.0f}x")
assert last == IMG // 16
'''),
        md("""
Two flower boundaries fifteen pixels apart are the same cell. Lecture 18 has to
put that resolution back, and the whole of its architecture is about how.
"""),

        md("""
## 5 · Thread 8, part four — where the memory goes

Your network has 4,807,494 parameters and your session died with `out of
memory`. **Which of those two facts caused the other?**

Commit to an answer before running the next cell.
"""),
        prompt(
            label="where the memory goes",
            input="the network and a batch of 32",
            output="parameter memory, optimiser memory, and activation memory",
            constraint="count FOUR float32 arrays per parameter — weights, gradients, and Adam's two moments — and every module output, which stays alive from the moment it is computed until the backward pass reaches it",
            check="assert activations exceed everything parameter-shaped, since the whole section depends on that being true",
            left_open="the question posed above it: your network has 4.8 million parameters and your session died with out-of-memory. WHICH of those two facts caused the other? Commit before running.",
            student="reducing the parameter count. Activation memory is linear in the batch size and parameter memory does not move at all.",
            catch="when a run runs out of memory, halve the batch, not the model. In order: smaller batch, then smaller resolution (quadratic), then gradient checkpointing, then mixed precision."),
        code('''
BATCH = 32
net = make_net()

n_par = sum(p.numel() for p in net.parameters())
par_mb = n_par * 4 / 2**20

# weights + gradients + Adam's two moments: four float32 arrays per parameter
opt_mb = 4 * par_mb

# every module output stays alive from the moment it is computed until the
# backward pass reaches it — that is what reverse-mode autodiff is
z, act_per_image = torch.zeros(1, 3, IMG, IMG), 0
for m_ in net:
    z = m_(z)
    act_per_image += z.numel()

act_mb = act_per_image * BATCH * 4 / 2**20

print(f"parameters                       {n_par:>12,}   {par_mb:8.1f} MB")
print(f"+ gradients + Adam state                        {opt_mb:8.1f} MB")
print(f"activations, per image           {act_per_image:>12,}   "
      f"{act_per_image * 4 / 2**20:8.1f} MB")
print(f"activations, batch of {BATCH}                        {act_mb:8.1f} MB")
print(f"\\nactivations / parameters              {act_mb / par_mb:8.1f}x")
print(f"activations / everything parameter-shaped {act_mb / opt_mb:8.1f}x")

assert act_mb > opt_mb, "the arithmetic says parameters dominate — check it"
'''),
        md("""
### It is not spread evenly either

The activation cost of a layer is `C × H × W`. Channels double as the map
quarters, so the total halves at every pooling stage — and the expensive layers
are the ones nearest the image, which are the ones with almost no parameters.
"""),
        prompt(
            label="and it is not spread evenly",
            input="every convolution's output shape",
            output="the activation memory of each",
            constraint="report the share held by the first two convolutions",
            left_open="the shape of it: activation cost is C×H×W, channels double as the map quarters, so the total halves at every pooling stage — and the expensive layers are the ones nearest the image, which are the ones with almost no parameters.",
            student="looking for the memory in the biggest layer by parameter count. It is in the smallest ones.",
            catch="parameters and activations are anti-correlated across a convolutional stack. Any intuition transferred from dense networks points the wrong way."),
        code('''
z, rows = torch.zeros(1, 3, IMG, IMG), []
for i, m_ in enumerate(net):
    z = m_(z)
    if isinstance(m_, nn.Conv2d):
        rows.append((i, tuple(z.shape[1:]), z.numel() * BATCH * 4 / 2**20))

for i, shape, mb in rows:
    print(f"conv at index {i:2d}  {str(shape):>18s}  {mb:7.1f} MB")

conv_total = sum(mb for _, _, mb in rows)
print(f"\\nfirst two convolutions: {100*(rows[0][2]+rows[1][2])/conv_total:.0f}%"
      f" of the convolutional activation memory")
'''),
        md("""
### Check the arithmetic against the allocator

A prediction nobody checks is a claim.
"""),
        prompt(
            label="check the arithmetic against the allocator",
            input="one real forward pass on the accelerator",
            output="the predicted activation memory beside the measured one",
            constraint="take ONE optimiser step first, so Adam's state exists and is in the baseline rather than appearing as activation memory",
            left_open="that the CPU branch prints nan and says so. There is no allocator counter there, and the arithmetic above still holds.",
            student="measuring without synchronising, on cuda or mps, and reading a number from before the work finished.",
            catch="a prediction nobody checks is a claim. This is four lines and it converts the whole section from arithmetic into a measurement."),
        code('''
net_d = make_net().to(device)
opt_d = torch.optim.Adam(net_d.parameters(), lr=3e-4)
xb = torch.randn(BATCH, 3, IMG, IMG, device=device)
yb = torch.randint(0, N_CLASSES, (BATCH,), device=device)
lossf = nn.CrossEntropyLoss()

# one step first, so Adam's state exists and is in the baseline
opt_d.zero_grad(); lossf(net_d(xb), yb).backward(); opt_d.step()

if device == "cuda":
    torch.cuda.synchronize(); torch.cuda.empty_cache()
    base = torch.cuda.memory_allocated()
    out = net_d(xb); torch.cuda.synchronize()
    measured = (torch.cuda.memory_allocated() - base) / 2**20
elif device == "mps":
    torch.mps.synchronize(); torch.mps.empty_cache()
    base = torch.mps.current_allocated_memory()
    out = net_d(xb); torch.mps.synchronize()
    measured = (torch.mps.current_allocated_memory() - base) / 2**20
else:
    measured = float("nan")
    print("no accelerator counter on CPU; the arithmetic above still holds")

print(f"predicted, by counting outputs  {act_mb:8.1f} MB")
print(f"measured, by the backend        {measured:8.1f} MB")
'''),
        md("""
### The rule that follows

**When a run runs out of memory, halve the batch, not the model.** Activation
memory is linear in the batch size; parameter memory does not move at all.

In order: smaller batch, then smaller input resolution (quadratic), then
gradient checkpointing, then mixed precision. Reducing the parameter count is
near the bottom of the list.
"""),

        md("""
## 6 · Diagnose

Start with what is *not* wrong: the loop, the metric, the split, the
architecture. Nothing here is a bug.

Retrain the from-scratch network briefly so this notebook has its own baseline
to compare against.

⏱ **about 30 seconds on a GPU or MPS.** 20 epochs.
"""),
        prompt(
            label="⏱ 30 s — the from-scratch baseline, rebuilt here",
            input="the previous lecture's network, 20 epochs",
            output="train, validation and test accuracy, the wall clock, and the gap",
            constraint="retrain it IN THIS NOTEBOOK, so the comparison below is between two runs on the same machine",
            left_open="that a persistent gap between two plateaus is variance, and its two cures are more data and more constraint. We cannot buy labels, so the constraint has to come from somewhere.",
            student="quoting the previous lecture's number against a transfer run measured today. Different session, possibly different hardware, certainly different epoch count.",
            catch="the observation that motivates everything below: the first layer learned colour blobs and oriented edges, and none of them is about flowers. We spent 1,020 precious labels rediscovering something not specific to this problem."),
        code('''
EPOCHS_SCRATCH, LR = 20, 3e-4

torch.manual_seed(RANDOM_STATE)
scratch = make_net().to(device)
Xtr, ytr = normalise(X_train).to(device), y_train.to(device)
opt = torch.optim.Adam(scratch.parameters(), lr=LR)
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
## 7 · Repair — a backbone somebody else trained

ImageNet: 1.28 million labelled photographs, 1,000 classes, none of them our
species. Somebody already paid for edges, colours, textures and parts. Those
weights are a download.
"""),
        prompt(
            label="a backbone somebody else trained",
            input="nothing",
            output="the weights' own declared preprocessing, and the parameter count",
            constraint="print `weights.transforms()` — the weights come WITH their preprocessing, and it is not the one you computed from the flowers",
            left_open="what was paid for: 1.28 million labelled photographs, 1,000 classes, none of them our species. Edges, colours, textures and parts are a download.",
            student="ignoring the declared transform and reusing the flower statistics. The first layer expects inputs on the scale it was trained with, and substituting your own is a silent, uncrashing degradation.",
            catch="every pretrained checkpoint states its preprocessing. Read it and use it — it is metadata, not a suggestion."),
        code('''
weights = ResNet18_Weights.DEFAULT
print(weights.transforms())
print(f"\\nparameters: {sum(p.numel() for p in resnet18().parameters()):,}")
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
            check="assert the output shape",
            left_open="that these constants are famous and are not universal. They are ImageNet's, and using them on a dataset whose statistics differ is a choice with a reason, not a default.",
            student="one `normalise` function mutated to take the ImageNet constants as defaults. Then every earlier cell in the notebook silently changes meaning.",
            catch="when two incompatible preprocessings coexist, give them different names and pass them explicitly. `norm=` as a parameter of `accuracy` exists for exactly this."),
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
            check="assert the trainable count is exactly 512·102 + 102",
            left_open="that this is not an accident of style. Reverse the two and every parameter in the network is frozen, including the head, and training runs happily and changes nothing.",
            student="replacing the head first and then freezing everything, which produces a model whose loss never moves and no error at all.",
            catch="assert the trainable parameter count against an arithmetic expression. It is the only thing that distinguishes the two orders."),
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
            check="assert the training features are (1020, 512)",
            left_open="the saving. The 1,020 images pass through the backbone ONCE rather than once per epoch, which is why the head can then be trained for 60 epochs in seconds.",
            student="running the full network every epoch with the body frozen. It gives the same answer and costs sixty times the compute.",
            catch="a frozen prefix is a fixed function. Compute it once and cache the output — that is not an optimisation, it is the definition of frozen."),
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
            left_open="the arithmetic that explains why it works: 52,326 trainable parameters fitted with 1,020 examples is about 51 parameters per image, against 4,713 before. The 11,176,512 frozen ones were fitted with 1.28 million examples, by somebody else.",
            student="reporting only the head's training time. The comparison in section 8 is against a from-scratch run, and it has to include everything the probe cost.",
            catch="the variance problem was not solved. It was MOVED to a dataset large enough to absorb it."),
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
            left_open="what makes it legitimate at all: the label is the same in all eight. An augmentation that could change the label is not an augmentation, it is a mislabelling.",
            student="adding vertical flips or large rotations because more augmentation sounds better. For some datasets that changes the class; for flowers it mostly does not, and 'mostly' is worth checking.",
            catch="look at the augmented images before training on them. A crop scale that occasionally excludes the subject is a labelled photograph of a leaf."),
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
⏱ **about 90 seconds on a GPU or MPS.** 8 epochs of fine-tuning at 224 × 224,
plus a validation pass each epoch.

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
            left_open="the second generator. Sharing one would mean the augmented-validation diagnostic consumed draws from the stream that shuffles training batches, so the training order would depend on whether the diagnostic ran at all.",
            student="`for p in ft.parameters(): p.requires_grad = False` and then training, expecting the batch norms to be frozen too. The running means keep moving, on augmented flower data, and the frozen backbone quietly stops being the one that was downloaded.",
            catch="1e-4 on the pretrained block and 1e-3 on the random head. The head starts from noise and the block starts from something good; one learning rate for both damages whichever it is wrong for."),
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
## 8 · The comparison this application exists to make
"""),
        prompt(
            label="the comparison this application exists to make",
            input="all five results",
            output="accuracy and wall clock for each",
            constraint="wall clock in the SAME column — the frozen probe's headline is that it is both faster and more accurate, and one of those is invisible without the time",
            check="assert the fine-tune beat the from-scratch run, with a message pointing at the normalisation if it did not",
            left_open="that both anchors stay in the table. A transfer result quoted without the majority baseline is missing its denominator.",
            student="dropping the baselines once the numbers get good. They are what makes 'far more accurate' a measurement.",
            catch="the assert names the likely cause of failure. `check inorm` is worth more than `assert FT_TEST > SCRATCH_TEST` alone."),
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
## 9 · An assistant writes the augmentation

**⚠ Read before running.** It runs, it trains, and the validation accuracy goes
up over epochs exactly as it should.

> *"Add data augmentation to the flower classifier: random resized crops and
> horizontal flips, with the ImageNet normalisation. Build the training and
> validation dataloaders."*

Perfectly reasonable, and it names both loaders. That is the hole.

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
            label="⚠ what the assistant returns — one transform for two splits",
            input="'add random resized crops and horizontal flips with ImageNet normalisation, and build the training and validation dataloaders'",
            output="one fixed set of weights scored ten times on the AUGMENTED validation set",
            constraint="score the same weights repeatedly — the wobble is the whole diagnostic",
            check="assert that the CLEAN evaluation is deterministic, so the wobble is attributable to the augmentation and not to eval mode",
            left_open="that nothing is leaking. The model never trains on the validation set; the damage is entirely in what the validation number now MEANS.",
            student="reusing `tf` for both splits, which is one word shorter and is exactly what the prompt's phrasing invites.",
            catch="Lecture 12's reviewer question arriving again: a deterministic function of fixed weights and fixed data does not wobble."),
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
            left_open="that a noisy metric is the smaller half of the problem. The larger half is that it is also pessimistic, so the reported number understates the model you shipped.",
            student="treating this as a cosmetic issue about a wobbly plot. It changes the model selection, which changes what you deploy.",
            catch="the corrected specification's last sentence is the part that generalises: assert that evaluating the same model twice gives identical numbers. One line, one second, and it catches this bug, a missing eval(), and any accidental shuffling of the labels."),
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
            left_open="how many distinct bugs this single line catches: augmented evaluation, a missing model.eval(), a shuffling DataLoader on the validation split, and label misalignment introduced by a re-shuffle.",
            student="asserting `abs(a - b) < 1e-3`, which passes under all four of those.",
            catch="put this line at the bottom of every notebook that evaluates a model. It costs a second and it is the highest-yield assertion in the course."),
        code('''
a = accuracy(ft, T_val, y_val, inorm)
b = accuracy(ft, T_val, y_val, inorm)
assert a == b, f"evaluation is not deterministic: {a} vs {b}"
print(f"evaluation is deterministic: {a:.4f} twice")
'''),

        md("""
## 10 · Red-team

Swap notebooks with the team beside you. Ten minutes, five questions:

1. What touched the test set?
2. What was fitted, and on what? (`fit` and `transform` are different verbs)
3. What is the shape here?
4. What was dropped — rows, columns, images? Count them.
5. What is the default I did not ask for?

In *this* application the specific answers are:

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
