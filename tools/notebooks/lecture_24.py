"""
Lecture 24 — Closing the loop, and closing the course.

Fix. Chapters 15–16. Thread 12: the contrastive objective and its temperature.

Exports build() -> list[nbformat cell]. Self-contained: it rebuilds the 200-entry
catalogue and re-encodes it rather than assuming Lecture 23's kernel is alive. A
notebook that only runs because a previous one left variables in memory is not
reproducible.

Two things are deliberately smaller here than on the slides, and both are stated
in the notebook where they are used: the retrieval-augmented section uses six of
the twelve ambiguous queries, and the captioner runs on the same sixty entries
but with a shorter beam. The deck quotes the full run.
"""

from __future__ import annotations

import nbformat as nbf
from _prompt import prompt                                # noqa: E402


def md(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(text.strip("\n"))


def code(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(text.strip("\n"))


HEADER = """
# Generation, retrieval-augmented systems, and where this leaves you

**Lecture 24** · Géron, Chapters 15–16 · *Mathematical thread: the
contrastive objective and its temperature*

Applications of Machine Learning — BSc Mathematics of Artificial Intelligence

---

**How to use this notebook.** Read before you run. Every code cell is preceded
by the specification that would produce it — input, output, constraint, check.

Cells marked **⚠** deliberately run code that is wrong, and say so in the
heading before you reach them. They are the failures this lecture is about;
each runs the broken version beside the correct one and prices the difference.

Runs on CPU. Nothing here needs an accelerator.

**What it downloads.** The same 200-entry catalogue as the previous lecture
(cached), plus two more checkpoints: a captioner (about 1 GB) and a small
instruction-tuned language model (about 1 GB). It does **not** download COCO.

**Expected wall clock on Colab's free CPU tier:** five to eight minutes end to
end, most of it downloads. That is the tier every notebook in this course is
sized for.
"""


def build() -> list:
    return [
        md(HEADER),

        md("## 1 · Setup, and the catalogue again"),
        prompt(
            label="setup",
            input="nothing",
            output="versions, seeds, device, N_CATALOGUE",
            constraint="the same constants as the previous lecture, so the numbers are comparable",
            check="`torch.nn.functional as Fn` rather than `F` — `F` is already the feature matrix in the previous notebook's namespace, and a collision there is the kind of bug that produces a confident wrong number.",
            **{"try": "import torch.nn.functional as F instead, then bind F "
                      "to a feature matrix later, as the previous lecture "
                      "does. Nothing raises until cross_entropy is looked up "
                      "on a numpy array twenty cells away. A namespace "
                      "collision is a bug whose stack trace points nowhere "
                      "near its cause."}),
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
import ast, re, sys, time, urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as Fn
import transformers
from PIL import Image

print(f"python        {sys.version.split()[0]}")
print(f"torch         {torch.__version__}")
print(f"transformers  {transformers.__version__}")

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
print(f"device        {device}")

N_CATALOGUE = 200
'''),

        md("""
The same deterministic catalogue: the split index sorted by COCO id, first 200
entries. Cached from the previous lecture if you ran it in this runtime.
**Expected wall clock: under a minute** if the cache is cold, instant otherwise.
"""),
        prompt(
            label="the same catalogue, rebuilt",
            input="the split index and the first 200 images by id",
            output="the identical 200 entries, with descriptions and queries",
            constraint="the same deterministic rule — sorted by cocoid, first 200 — so this notebook and the last are talking about the same corpus",
            check="assert the count and that the SKUs are unique. Whitespace-normalise the captions on the way in, exactly as before. Two notebooks that normalise differently have different queries.",
            **{"try": "delete the whitespace normalisation here but leave it "
                      "in the previous lecture's notebook, then compare the "
                      "two R@1 figures. A query that differs only by a "
                      "newline is a different query, and the two notebooks "
                      "are no longer talking about the same corpus."}),
        code('''
CACHE = Path("datasets/app12")
CACHE.mkdir(parents=True, exist_ok=True)
CSV_URL = ("https://huggingface.co/datasets/nlphuji/"
           "mscoco_2014_5k_test_image_text_retrieval/resolve/main/"
           "test_5k_mscoco_2014.csv")

csv_path = CACHE / "coco_karpathy_test.csv"
if not csv_path.is_file():
    urllib.request.urlretrieve(CSV_URL, csv_path)
split = pd.read_csv(csv_path).sort_values("cocoid").reset_index(drop=True)

imgdir = CACHE / "images"
imgdir.mkdir(exist_ok=True)
catalogue = []
for _, row in split.head(N_CATALOGUE).iterrows():
    dest = imgdir / row["filename"]
    if not dest.is_file():
        urllib.request.urlretrieve(
            f"http://images.cocodataset.org/val2014/{row['filename']}", dest)
    captions = [" ".join(c.split()) for c in ast.literal_eval(row["raw"])]
    catalogue.append({"sku": f"CAT-{int(row['cocoid']):06d}",
                      "file": dest, "captions": captions})

assert len(catalogue) == N_CATALOGUE, len(catalogue)
assert len({e["sku"] for e in catalogue}) == N_CATALOGUE, "duplicate SKU"

images       = [Image.open(e["file"]).convert("RGB") for e in catalogue]
descriptions = [e["captions"][0] for e in catalogue]
queries      = [e["captions"][1] for e in catalogue]
valid_skus   = {e["sku"] for e in catalogue}
print(f"catalogue: {len(catalogue)} entries")
'''),

        prompt(
            label="the embeddings, and the raw ones kept",
            input="the images and queries",
            output="both the RAW features and the unit-normalised ones",
            constraint="keep `I_raw` and `Q_raw` deliberately — the assistant failure in section 6 is about what happens when you use them, and it cannot be demonstrated if they were normalised in place",
            check="assert both normalised matrices really have unit rows. `np.allclose(norm, 1.0)` as an assert, not a print. It is two lines, and section 6 is entirely about the run where it would have fired.",
            **{"try": "normalise in place — set I_raw = unit(I_raw) on this "
                      "line — and then run Section 2. The assistant failure "
                      "disappears and the section has nothing left to show. "
                      "Keeping the broken input alive is what makes the bug "
                      "measurable rather than merely describable."}),
        code('''
from transformers import CLIPModel, CLIPProcessor

CLIP_ID = "openai/clip-vit-base-patch32"
clip_proc = CLIPProcessor.from_pretrained(CLIP_ID)
clip = CLIPModel.from_pretrained(CLIP_ID).to(device).eval()


def unit(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def features(out):
    """The projected embedding, whichever transformers version is installed.

    In transformers 4.x `get_image_features` returned the tensor itself; in 5.x
    it returns an output object whose `pooler_output` holds the same vector.
    Both are the PROJECTED and UNNORMALISED embedding, which is the whole
    subject of the section below -- so this helper must not normalise.
    """
    return out if torch.is_tensor(out) else out.pooler_output


def clip_images(imgs, batch=32):
    out = []
    with torch.no_grad():
        for i in range(0, len(imgs), batch):
            b = clip_proc(images=imgs[i:i + batch], return_tensors="pt").to(device)
            out.append(features(clip.get_image_features(**b)).cpu().numpy())
    return np.concatenate(out).astype(np.float64)


def clip_text(sentences, batch=64):
    out = []
    with torch.no_grad():
        for i in range(0, len(sentences), batch):
            b = clip_proc(text=sentences[i:i + batch], return_tensors="pt",
                          padding=True, truncation=True, max_length=77).to(device)
            out.append(features(clip.get_text_features(**b)).cpu().numpy())
    return np.concatenate(out).astype(np.float64)


I_raw = clip_images(images)          # NOT normalised — deliberately
Q_raw = clip_text(queries)
I, Q = unit(I_raw), unit(Q_raw)
d = I.shape[1]

assert np.allclose(np.linalg.norm(I, axis=1), 1.0), "images are not unit vectors"
assert np.allclose(np.linalg.norm(Q, axis=1), 1.0), "texts are not unit vectors"
print(f"embeddings {I.shape}, shared dimension d = {d}")
'''),

        md("""
Accuracy falls with *B* and chance falls faster, so the gap — the learning signal
— widens. That is the whole argument for a large batch, and it has two
consequences worth carrying away:

1. **A contrastive loss value is not comparable across papers.** It is measured
   against a batch-dependent ceiling of `log B`.
2. **Doubling the batch changes the task**, not just the gradient noise. Which is
   why these models are trained with batches in the tens of thousands, across
   many devices. That engineering is outside Chapters 1–16.

## 2 · ⚠ Read before running — the assistant failure

**The prompt:** *"Write the symmetric contrastive loss for a batch of image and
text embeddings, with a temperature of 0.01."*

One clause missing — the clause section 2 spent five minutes on.
"""),
        prompt(
            label="⚠ what the assistant returns",
            input="'write the symmetric contrastive loss for a batch of image and text embeddings, with a temperature of 0.01'",
            output="the loss, computed on the RAW features",
            constraint="run it exactly as returned. It runs, the docstring is accurate, and the shapes, the target and the factor of one half are all right",
            check="reviewer question 5 again. The default nobody asked for is that a function called `get_*_features` returns something unnormalised.",
            **{"try": "call the same contrastive_loss on the normalised I and "
                      "Q, at the same tau. Same function, same shapes, same "
                      "target, different number. Which line of that docstring "
                      "would have had to be wrong for a reader to catch this "
                      "without running it?"}),
        code('''
# The temperature CLIP itself was trained with, rather than a number chosen to
# make the demonstration work: logit_scale is stored as a log and exponentiated.
tau = float(1.0 / clip.logit_scale.exp().item())
norms = np.linalg.norm(I_raw, axis=1)
print(f"CLIP's own temperature: tau = {tau:.4f}")


def infonce(sim, tau):
    """Symmetric InfoNCE over a similarity matrix, plus its top-1 accuracy.

    Takes the matrix rather than the two embedding sets, so that the SAME loss
    can be applied to a cosine matrix and to a raw dot-product matrix and the
    only difference between the two rows below is the thing being compared.
    """
    logits = torch.tensor(sim / tau, dtype=torch.float32)
    target = torch.arange(len(logits))
    loss = 0.5 * (Fn.cross_entropy(logits, target) +
                  Fn.cross_entropy(logits.T, target))
    return {"loss": float(loss),
            "accuracy": float((logits.argmax(1) == target).float().mean())}


# --- what the weak prompt returns --------------------------------------------
def contrastive_loss(img, txt, tau=0.01):
    """Symmetric InfoNCE over a batch of paired embeddings."""
    logits = img @ txt.T / tau                 # (B, B)
    target = torch.arange(len(img), device=img.device)
    return 0.5 * (Fn.cross_entropy(logits, target) +
                  Fn.cross_entropy(logits.T, target))


img_t = torch.tensor(I_raw, dtype=torch.float32)     # as get_*_features returns
txt_t = torch.tensor(Q_raw, dtype=torch.float32)
print(f"loss on raw features: {contrastive_loss(img_t, txt_t, tau).item():.3f}")
'''),

        md("""
It runs. The docstring is accurate. The shapes, the target and the factor of one
half are all right.

**The review question:** *is `img @ txt.T` a cosine?* Only if both sides are unit
vectors — and `get_image_features` does not return unit vectors. So the entries
are `||a|| ||b|| cos(theta)`, the temperature is dividing a quantity with no
fixed scale, and the row-wise softmax compares lengths as much as directions.
Reviewer question 5 again: the default nobody asked for.
"""),
        prompt(
            label="what it costs, and how",
            input="the loss and accuracy both ways",
            output="both, plus the rank correlation between embedding length and queries won",
            constraint="show the MECHANISM, not just the loss difference — count how many queries each image wins and correlate that with its length",
            check="the corrected specification's assertion is the part that would have caught it in silence: assert every row of both matrices has unit norm to within 1e-5. Two lines, and the bug becomes a crash.",
            **{"try": "recompute the rank correlation against the TEXT "
                      "embedding lengths instead of the image ones. It "
                      "collapses, because a text length is constant down a "
                      "column and cancels out of that argmax. Which side of "
                      "an unnormalised dot product does the softmax actually "
                      "punish, and why only that one?"}),
        code('''
good = infonce(I @ Q.T, tau)
bad  = infonce(I_raw @ Q_raw.T, tau)
print(f"{'':22s} {'loss':>10} {'top-1':>8}")
print(f"{'on the unit sphere':22s} {good['loss']:10.3f} {good['accuracy']:8.1%}")
print(f"{'raw dot products':22s} {bad['loss']:10.3f} {bad['accuracy']:8.1%}")

wins = np.bincount((I_raw @ Q_raw.T).argmax(0), minlength=N_CATALOGUE)
order = np.argsort(np.argsort(norms))
corr = np.corrcoef(order, np.argsort(np.argsort(wins)))[0, 1]
print(f"\\nrank correlation between embedding length and queries won: {corr:+.2f}")
print(f"one image took {wins.max()} of {N_CATALOGUE} queries; "
      f"{(wins == 0).sum()} images were never ranked first")
'''),

        md("""
**The corrected specification:**

> Symmetric InfoNCE. **L2-normalise both embedding sets along the feature axis
> before the matrix product**, so the logits are cosines divided by `tau`. Assert
> that every row of both matrices has unit norm to within 1e-5. Report the loss
> *and* the in-batch top-1 accuracy, and print `log B` beside the loss.

The assertion is the part that would have caught it in silence. Two lines, and
the bug becomes a crash.

## 3 · Repair 1 — entries with no description

Sixty of the two hundred entries have no description, so they score exactly zero
on the text route. Use a model built for *generating* text from an image and
write the missing ones.

**Expected wall clock: 1–3 min**, most of it the 1 GB download.
"""),
        prompt(
            label="rebuild the text route and its hole",
            input="the descriptions and queries through MiniLM",
            output="R@1 on the 60 blanked entries, with and without their descriptions",
            constraint="the same fixed blanking rule as the previous lecture, and the same −inf convention for 'not in the index at all'",
            check="assert exactly 60 entries were blanked. Reproduce the fault before repairing it, from the same rule and the same seed. A repair measured against a differently-broken baseline is not measured.",
            **{"try": "flip the rule to i % 10 >= 3, so the other 140 entries "
                      "lose their descriptions instead. R@1 on the blanked "
                      "set is still exactly zero. A structural zero does not "
                      "depend on which entries you choose, and that is what "
                      "makes it structural rather than bad luck."}),
        code('''
from transformers import AutoTokenizer, BlipForConditionalGeneration, BlipProcessor
from transformers import AutoModel

TEXT_ID = "sentence-transformers/all-MiniLM-L6-v2"
mtok = AutoTokenizer.from_pretrained(TEXT_ID)
menc = AutoModel.from_pretrained(TEXT_ID).to(device).eval()


def minilm(sentences, batch=64):
    out = []
    with torch.no_grad():
        for i in range(0, len(sentences), batch):
            b = mtok(sentences[i:i + batch], return_tensors="pt", padding=True,
                     truncation=True, max_length=128).to(device)
            h = menc(**b).last_hidden_state
            m = b["attention_mask"].unsqueeze(-1).float()
            out.append(((h * m).sum(1) / m.sum(1)).cpu().numpy())
    return np.concatenate(out).astype(np.float64)


def ranks_of_truth(sim):
    return (sim >= np.diag(sim)[:, None]).sum(axis=1)


D = unit(minilm(descriptions))
Qt = unit(minilm(queries))
blanked = np.array([i for i in range(N_CATALOGUE) if i % 10 < 3])
assert len(blanked) == 60, len(blanked)

sim_full = Qt @ D.T
sim_missing = sim_full.copy()
sim_missing[:, blanked] = -np.inf

r_full, r_missing = ranks_of_truth(sim_full), ranks_of_truth(sim_missing)
print(f"R@1 on the 60 blanked entries — described: "
      f"{(r_full[blanked] <= 1).mean():.1%},  deleted: "
      f"{(r_missing[blanked] <= 1).mean():.1%}")
'''),

        prompt(
            label="⏱ 1-3 min — write the missing descriptions",
            input="the 60 images with no description",
            output="a generated caption for each, with four shown beside their human captions",
            constraint="generate for the BLANKED entries only — captioning all 200 would replace descriptions that already exist and confound the measurement",
            check="assert one caption came back per blanked entry. Print human and generated side by side for a few. It is the only way to see that the generated ones are shorter, blander and occasionally wrong.",
            **{"try": "set num_beams=1 and re-run. The captions get shorter "
                      "and blander and one or two change meaning. Then repeat "
                      "the measurement in the next cell: how much of the "
                      "recovered recall was the beam width rather than the "
                      "captioner?"}),
        code('''
proc = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
cap = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base").to(device).eval()

t0 = time.perf_counter()
generated = []
with torch.no_grad():
    for i in range(0, len(blanked), 8):
        chunk = blanked[i:i + 8]
        b = proc(images=[images[j] for j in chunk], return_tensors="pt").to(device)
        ids = cap.generate(**b, max_new_tokens=30, num_beams=3)
        generated += [" ".join(t.split())
                      for t in proc.batch_decode(ids, skip_special_tokens=True)]

assert len(generated) == len(blanked), (len(generated), len(blanked))
print(f"{len(generated)} captions written in {time.perf_counter() - t0:.0f}s\\n")
for j, g in list(zip(blanked, generated))[:4]:
    print(f"human    : {descriptions[j]}")
    print(f"generated: {g}\\n")
'''),

        prompt(
            label="what the repair recovers",
            input="the description matrix with the generated captions filled in",
            output="R@1 on the 60, four ways, and the overall text-route figure",
            constraint="measure on the SAME 60 entries throughout — the overall number is diluted by the 140 that never changed",
            check="three things not to claim: a generated caption is not evidence about the product; 60 entries is a small sample with a wide interval; and we are scoring generated captions against human captions of the SAME image, which is a friendly test. The failure mode to watch for is an auto-caption that is WRONG, making an entry findable under the wrong query — worse than unfindable, and nothing measured here detects it.",
            **{"try": "caption all 200 entries and replace every description, "
                      "the human ones included. The overall R@1 falls. The "
                      "repair is worth a great deal where there was nothing "
                      "and is a downgrade everywhere else, which is exactly "
                      "what the constraint above is protecting."}),
        code('''
D_filled = D.copy()
D_filled[blanked] = unit(minilm(generated))
r_filled = ranks_of_truth(Qt @ D_filled.T)

sim_clip = Q @ I.T
r_clip = ranks_of_truth(sim_clip)

print(f"R@1, measured on the same 60 entries:")
print(f"  human description present   {(r_full[blanked] <= 1).mean():6.1%}")
print(f"  description deleted         {(r_missing[blanked] <= 1).mean():6.1%}")
print(f"  auto-caption                {(r_filled[blanked] <= 1).mean():6.1%}")
print(f"  joint image route (unused)  {(r_clip[blanked] <= 1).mean():6.1%}")
print(f"\\nover all {N_CATALOGUE} queries, text route R@1: "
      f"{(r_full <= 1).mean():.1%} -> {(r_missing <= 1).mean():.1%} "
      f"-> {(r_filled <= 1).mean():.1%}")
'''),

        md("""
A generated caption recovers **part** of the loss, not all of it. Report the
part, not the direction.

Three things not to claim: a generated caption is not evidence about the product;
sixty entries is a small sample with a wide interval around it; and we are
scoring generated captions against human captions of the same image, which is a
friendly test. The failure mode to watch for is an auto-caption that is *wrong*,
making an entry findable under the wrong query — worse than unfindable, and
nothing measured here detects it.

## 4 · Repair 2 — queries with no single right answer

Customers do not type captions. They type "something to sit on". A ranked list of
pictures is a poor answer, because the customer wants a shortlist **with
reasons**, and a ranking has no place to put a reason.

First decide how to check it. "The answer is good" is not measurable in a
lecture, so require the assistant to cite stock numbers: a SKU either exists in
the catalogue or it does not. That measures **grounding**, not helpfulness — say
which one you measured.

**Expected wall clock: 2–4 min**, most of it the 1 GB download.
"""),
        prompt(
            label="⏱ 2-4 min — queries with no single right answer",
            input="six deliberately ambiguous queries",
            output="the model, the prompt helper, and the top-5 shortlist per query",
            constraint="decide how to CHECK it before generating anything — 'the answer is good' is not measurable in a lecture, so require cited SKUs, which either exist in the catalogue or do not",
            check="assert the shortlist has the shape you expect. `do_sample=False`. A sampled generation gives a different answer every run and the grounding rate becomes a random variable you have not characterised.",
            **{"try": "set do_sample=True and run the grounding measurement "
                      "three times. The percentage moves every run. A "
                      "grounding rate computed from a sampled generation is a "
                      "random variable, and nothing in this notebook has "
                      "characterised its spread."}),
        code('''
from transformers import AutoModelForCausalLM

LLM_ID = "Qwen/Qwen2.5-0.5B-Instruct"
ltok = AutoTokenizer.from_pretrained(LLM_ID)
llm = AutoModelForCausalLM.from_pretrained(LLM_ID).to(device).eval()


def ask(prompt, max_new_tokens=140):
    msgs = [{"role": "system",
             "content": "You are a product-catalogue assistant. Cite catalogue "
                        "SKUs, which always look like CAT-123456."},
            {"role": "user", "content": prompt}]
    text = ltok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    b = ltok([text], return_tensors="pt").to(device)
    with torch.no_grad():
        out = llm.generate(**b, max_new_tokens=max_new_tokens, do_sample=False,
                           pad_token_id=ltok.eos_token_id)
    return ltok.decode(out[0][b["input_ids"].shape[1]:], skip_special_tokens=True)


# six of the twelve queries on the slides, so this cell finishes in the lecture
AMBIGUOUS = ["something to sit on",
             "somewhere to eat outdoors",
             "a way to get across town without a car",
             "gear for bad weather",
             "something to put flowers in",
             "a machine that heats food"]

SKU_RE = re.compile(r"CAT-\\d{6}")
q_emb = unit(clip_text(AMBIGUOUS))
top5 = np.argsort(-(q_emb @ I.T), axis=1)[:, :5]
assert top5.shape == (len(AMBIGUOUS), 5)
'''),

        prompt(
            label="closed book against retrieval-augmented",
            input="the same six queries, asked both ways",
            output="how many cited SKUs actually exist, under each condition",
            constraint="identical prompts except for the shortlist — the only difference must be whether the model was given the catalogue",
            check="the retriever is now the ceiling. If the right entry is not in the top five, no amount of generation recovers it — which is why the R@5 from the previous lecture is the number that matters here.",
            **{"try": "count each SKU once per answer rather than once per "
                      "mention — question 4 of the checklist below. A model "
                      "that cites one real stock number five times currently "
                      "earns five successes. Does the gap between the two "
                      "conditions survive the fix?"}),
        code('''
t0 = time.perf_counter()
closed_cited, grounded_cited = [], []
for qi, q in enumerate(AMBIGUOUS):
    closed = ask(f"Our catalogue has {N_CATALOGUE} entries. A customer asks: "
                 f'"{q}". Recommend three entries and cite their SKUs.')
    shortlist = "\\n".join(f"- {catalogue[j]['sku']}: {catalogue[j]['captions'][0]}"
                          for j in top5[qi])
    grounded = ask(f'Here are the catalogue entries our search returned for the '
                   f'query "{q}":\\n{shortlist}\\n\\nRecommend the best ones for '
                   f'the customer, citing only SKUs from the list above. '
                   f'If none fit, say so.')
    closed_cited += SKU_RE.findall(closed)
    grounded_cited += SKU_RE.findall(grounded)

print(f"{len(AMBIGUOUS)} queries, both ways, in {time.perf_counter() - t0:.0f}s\\n")
for name, cited in [("closed book", closed_cited), ("retrieval-augmented", grounded_cited)]:
    ok = sum(s in valid_skus for s in cited)
    pct = 100 * ok / len(cited) if cited else 0.0
    print(f"{name:22s} {ok:3d} of {len(cited):3d} cited SKUs exist  ({pct:5.1f}%)")
'''),

        md("""
The closed-book failure is not that the model refuses. It is that it does *not*
refuse: it produces a fluent recommendation, in exactly the right SKU format,
and nothing in the output distinguishes an invented stock number from a real one.

What this does **not** fix:

* **Grounding is not correctness.** A cited entry can exist and still be a bad
  recommendation. We measured the cheap half.
* **The retriever is now the ceiling.** If the right entry is not in the top
  five, no amount of generation recovers it — which is why the R@5 from the
  previous lecture is the number that matters here.
* **Fluency is unchanged.** A wrong answer built from real SKUs reads better than
  one built from invented ones.

Evaluating generated answers — faithfulness, attribution, judging with another
model — is a live research area, outside Chapters 1–16 and not examinable.

## 5 · Where we are

Four questions to ask of any retrieval-augmented system:

1. Are the embeddings on both sides unit vectors? Assert it, do not read it.
2. Is every recall reported with its candidate-set size?
3. Were the auto-captions generated for entries that are also in the evaluation
   queries — and does that matter here?
4. Does the grounding metric count a SKU cited twice as two citations? What would
   that do to the percentage?

---

### The course, in five rules

1. Split before anything is fitted.
2. All preprocessing inside the object that is cross-validated.
3. Nothing derived from the test set in the training path.
4. Fixed seeds; report per-fold scores, not only the mean.
5. Every number gets a baseline, and every baseline gets stated.

Twelve applications and twenty-four lectures, and those five never needed
extending — not for images, not for sequences, and not for two modalities at
once.

You will forget most of the syntax. Keep the five reviewer questions, the four
rules, and the habit of writing the number down first.
""")]
