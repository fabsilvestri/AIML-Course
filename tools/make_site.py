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
    (4, "Training models", "Ch 4", "Titanic", "Gradient descent", False),
    (5, "Regularisation and the bias–variance trade-off", "Ch 4", "Titanic",
     "The bias–variance decomposition", False),
    (6, "Decision trees", "Ch 5", "CoverType", "Impurity: Gini and entropy",
     False),
    (7, "Ensembles and random forests", "Ch 6", "CoverType",
     "The variance of an average of correlated predictors", False),
    (8, "Dimensionality reduction and unsupervised learning", "Ch 7–8",
     "Olivetti faces", "PCA via the SVD; Johnson–Lindenstrauss", False),
    (9, "Neural networks, from the perceptron up", "Ch 9", "Fashion-MNIST",
     "What a layer computes", False),
    (10, "PyTorch", "Ch 10", "Fashion-MNIST",
     "Backpropagation as reverse-mode automatic differentiation", False),
    (11, "Training deep networks", "Ch 11", "CIFAR-10",
     "Variance propagation and weight initialisation", False),
    (12, "Convolutional networks", "Ch 12", "Flowers102",
     "Weight sharing, equivariance and memory", False),
    (13, "Transfer learning", "Ch 12", "Flowers102", "", False),
    (14, "Detection and segmentation", "Ch 12", "COCO",
     "IoU’s vanishing gradient; mAP", False),
    (15, "Time series", "Ch 13", "Chicago transit ridership",
     "Stationarity, differencing and autocorrelation", False),
    (16, "Recurrent networks", "Ch 13", "Chicago transit ridership", "", False),
    (17, "Text", "Ch 14", "IMDb", "Softmax, cross-entropy and logits", False),
    (18, "Attention and transformers", "Ch 14–15", "IMDb",
     "Scaled dot-product attention", False),
    (19, "Information retrieval: the lexical foundation", "", "SciFact (BEIR)",
     "Evaluating a ranking: MRR, AP, NDCG", False),
    (20, "Information retrieval: dense retrieval", "", "SciFact (BEIR)", "",
     False),
    (21, "Recommender systems: from ratings to factors", "", "MovieLens",
     "Matrix factorisation, and its relation to the SVD", False),
    (22, "Recommender systems: neural, and evaluated honestly", "",
     "MovieLens", "", False),
    (23, "Vision transformers and multimodal retrieval", "Ch 15–16", "COCO",
     "The contrastive objective and its temperature", False),
    (24, "Generation, retrieval-augmented systems, and where this leaves you",
     "Ch 15–16", "COCO and the Part V corpora", "", False),
]

PARTS = [
    (1, 8, "Part I — Tabular data and classical models",
     "Lectures 1–8 · Chapters 1–8 · CPU only"),
    (9, 11, "Part II — Neural networks",
     "Lectures 9–11 · Chapters 9–11 · GPU from Lecture 10"),
    (12, 14, "Part III — Computer vision",
     "Lectures 12–14 · Chapter 12 · GPU"),
    (15, 18, "Part IV — Sequences and language",
     "Lectures 15–18 · Chapters 13–15 · GPU"),
    (19, 22, "Part V — Search and recommendation",
     "Lectures 19–22 · Lecture notes · examinable"),
    (23, 24, "Part VI — Multimodal models, and closing the course",
     "Lectures 23–24 · Chapters 15–16 · GPU"),
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
                        f'          <a class="btn btn-colab" href="{COLAB.format(n)}">Notebook</a>',
                        '        </div>']
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
