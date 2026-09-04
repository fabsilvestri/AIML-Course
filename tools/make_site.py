#!/usr/bin/env python3
"""Regenerate the parts of index.html that repeat, from the plan.

    python3 tools/make_site.py

The lecture list is twenty-four near-identical blocks and the derivation list
is eighteen; hand-editing either is how a lecture ends up on the site with the
wrong chapter, or published before its deck exists. Both are generated from the
tables below, between the BEGIN/END markers in index.html. Everything else on
the page is hand-written and untouched.

Keep LECTURES here in step with LECTURES.md. If they disagree, LECTURES.md is
the plan and this is the bug.
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# (n, title, source, dataset, derivation, published)
#   source     — the chapter, or "" for the lecture-notes lectures
#   derivation — the object derived in this lecture, "" if none
#   published  — whether the deck and notebook are on the new design yet
LECTURES = [
    (1, "What machine learning is, and how we will work", "Ch 1–2",
     "California housing", "", True),
    (2, "The end-to-end project", "Ch 2", "California housing",
     "Least squares and the normal equation", True),
    (3, "Classification and its metrics", "Ch 3", "MNIST",
     "Imbalance, and the non-monotonicity of precision", True),
    (4, "Training models", "Ch 4", "Titanic", "Gradient descent", True),
    (5, "Regularisation and the bias–variance trade-off", "Ch 4", "Titanic",
     "The bias–variance decomposition", True),
    (6, "Decision trees", "Ch 5", "CoverType", "Impurity: Gini and entropy",
     True),
    (7, "Ensembles and random forests", "Ch 6", "CoverType",
     "The variance of an average of correlated predictors", True),
    (8, "Dimensionality reduction and unsupervised learning", "Ch 7–8",
     "Olivetti faces", "PCA via the SVD; Johnson–Lindenstrauss", True),
    (9, "Neural networks, from the perceptron up", "Ch 9", "Fashion-MNIST",
     "What a layer computes", True),
    (10, "PyTorch", "Ch 10", "Fashion-MNIST",
     "Backpropagation as reverse-mode automatic differentiation", True),
    (11, "Training deep networks", "Ch 11", "CIFAR-10",
     "Variance propagation and weight initialisation", True),
    (12, "Convolutional networks", "Ch 12", "Flowers102",
     "Weight sharing, equivariance and memory", True),
    (13, "Transfer learning", "Ch 12", "Flowers102", "", True),
    (14, "Detection and segmentation", "Ch 12", "COCO",
     "IoU’s vanishing gradient; mAP", True),
    (15, "Time series", "Ch 13", "Chicago transit ridership",
     "Stationarity, differencing and autocorrelation", True),
    (16, "Recurrent networks", "Ch 13", "Chicago transit ridership", "", True),
    (17, "Text", "Ch 14", "IMDb", "Softmax, cross-entropy and logits", True),
    (18, "Attention and transformers", "Ch 14–15", "IMDb",
     "Scaled dot-product attention", True),
    (19, "Information retrieval: the lexical foundation", "", "SciFact (BEIR)",
     "Evaluating a ranking: MRR, AP, NDCG", True),
    (20, "Information retrieval: dense retrieval", "", "SciFact (BEIR)", "",
     True),
    (21, "Recommender systems: from ratings to factors", "", "MovieLens",
     "Matrix factorisation, and its relation to the SVD", True),
    (22, "Recommender systems: neural, and evaluated honestly", "",
     "MovieLens", "", True),
    (23, "Vision transformers and multimodal retrieval", "Ch 15–16", "COCO",
     "The contrastive objective and its temperature", True),
    (24, "Generation, retrieval-augmented systems, and where this leaves you",
     "Ch 15–16", "COCO and the Part V corpora", "", True),
]

PARTS = [
    (1, 8, "Part I — Tabular data and classical models",
     "Lectures 1–8 · Chapters 1–8 · runs on CPU"),
    (9, 11, "Part II — Neural networks",
     "Lectures 9–11 · Chapters 9–11 · runs on CPU"),
    (12, 14, "Part III — Computer vision",
     "Lectures 12–14 · Chapter 12 · runs on CPU"),
    (15, 18, "Part IV — Sequences and language",
     "Lectures 15–18 · Chapters 13–15 · runs on CPU"),
    (19, 22, "Part V — Search and recommendation",
     "Lectures 19–22 · Lecture notes · examinable"),
    (23, 24, "Part VI — Multimodal models, and closing the course",
     "Lectures 23–24 · Chapters 15–16 · runs on CPU"),
]

COLAB = ("https://colab.research.google.com/github/fabsilvestri/AIML-Course"
         "/blob/main/notebooks/lecture-{:02d}.ipynb")


def lecture_list() -> str:
    out = []
    for lo, hi, name, meta in PARTS:
        title, _, sub = name.partition(" — ")
        out += ['    <div class="part-head">',
                f'      <h3>{title} &mdash; {sub}</h3>',
                f'      <span class="part-meta">{meta.replace(" · ", " &middot; ")}</span>',
                '    </div>',
                '    <ol class="lectures">']
        for n, t, src, data, deriv, pub in LECTURES:
            if not lo <= n <= hi:
                continue
            chip = (f'<span class="badge badge-ch">{src}</span>' if src
                    else '<span class="badge badge-math">Lecture notes</span>')
            out += ['      <li class="lecture">',
                    f'        <span class="n">{n:02d}</span>',
                    '        <div class="body">',
                    f'          <p class="t">{t}</p>',
                    f'          <p class="meta">{data} {chip}</p>']
            if deriv:
                out.append(f'          <p class="thread">Derivation &middot; {deriv}</p>')
            out.append('        </div>')
            if pub:
                out += ['        <div class="links">',
                        f'          <a class="btn" href="slides/lecture-{n:02d}.html">Slides</a>',
                        # Between Slides and Notebook: the same deck, printed.
                        # Built by tools/make_deck_pdfs.py, one page per slide.
                        f'          <a class="btn btn-pdf" href="slides/pdf/lecture-{n:02d}.pdf">PDF</a>',
                        f'          <a class="btn btn-colab" href="{COLAB.format(n)}">Notebook</a>']
                # The lecture-notes lectures -- those with no chapter -- carry a
                # third link. For them the PDF is the primary source, not a
                # supplement, so it sits beside the other two rather than below.
                if not src:
                    out.append(
                        f'          <a class="btn btn-notes" href="notes/lecture-{n:02d}.pdf">Notes (PDF)</a>')
                out.append('        </div>')
            else:
                out += ['        <div class="links">',
                        '          <span class="btn btn-soon">In preparation</span>',
                        '        </div>']
            out.append('      </li>')
        out.append('    </ol>')
    return "\n".join(out)


def derivation_list() -> str:
    return "\n".join(
        f'      <li>{d} <span class="where">&middot; Lecture {n}</span></li>'
        for n, _, _, _, d, _ in LECTURES if d)


def main() -> int:
    path = ROOT / "index.html"
    html = path.read_text(encoding="utf-8")
    for marker, body in (("LECTURES", lecture_list()),
                         ("DERIVATIONS", derivation_list())):
        pattern = re.compile(
            rf"(<!-- BEGIN {marker} -->\n).*?(\s*<!-- END {marker} -->)",
            re.S)
        if not pattern.search(html):
            print(f"index.html has no {marker} markers — nothing written")
            return 1
        html = pattern.sub(lambda m: m.group(1) + body + m.group(2), html)
    path.write_text(html, encoding="utf-8")
    published = sum(1 for *_, pub in LECTURES if pub)
    print(f"index.html — {len(LECTURES)} lectures, {published} published, "
          f"{sum(1 for *_, d, _ in LECTURES if d)} derivations")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
