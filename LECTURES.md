# Machine Learning Applications — Lecture Plan

**Programme:** BSc Mathematics of Artificial Intelligence, third year
**Load:** 48 academic hours = 24 lectures × 2 academic hours (90 minutes each)
**Primary text:** A. Géron, *Hands-On Machine Learning with Scikit-Learn and PyTorch*, O'Reilly, 2025 — Chapters 1–16
**Beyond the text:** Lectures 19–22 (information retrieval and recommender systems) are taught from the lecture notes and are examinable.

---

## Organising principle

One topic per lecture, in the order of the primary text, each lecture
self-contained: the mathematics that the method rests on, the method itself, a
worked example, and a complete notebook that implements it.

A student who misses a lecture can read that lecture's deck and notebook and
catch up without reconstructing anything from the lecture before it.

### Shape of a lecture (90 minutes)

| min | block |
|---|---|
| 0–10 | where we are, what today's method is for, and the problem it solves |
| 10–30 | **the mathematics** — the one object the method rests on, derived |
| 30–70 | **the method** — how it works, its hyperparameters, its failure conditions, and a worked example with real numbers |
| 70–85 | further ground: variants, when to prefer each, what practitioners actually use |
| 85–90 | the notebook — what is in it, what to run, what to change |

The mathematics is not a separate device with its own numbering. It is the first
half of the topic, because the topic does not make sense without it.

### The notebooks

One per lecture, `notebooks/lecture-NN.ipynb`, complete and correct. Every
notebook:

- runs top to bottom from a cold Colab kernel with no manual steps;
- is heavily commented — the comments explain *why*, not what the line does;
- precedes every code cell with the **prompt that would generate it**: input,
  output, constraint, check. The prompts are specifications, so a reader who
  works through the notebook has also seen how to ask an assistant for each
  piece of it;
- states the wall-clock cost of any cell that takes more than about 20 seconds,
  and whether it needs a GPU;
- contains nothing that is wrong on purpose.

Notebooks are for self-study. The last five minutes of the lecture tour the
notebook's structure and say what to do with it; the running happens at home.

### AI-assisted development

Students will use an assistant to write code — in this course and after it. The
course treats that as a working method to be taught explicitly rather than
assumed: Lecture 1 devotes a block to it (specify, read, test, verify), and the
prompt preceding every notebook cell is a worked specification. No lecture is
built around an assistant failing.

### Standing practice

Introduced in Lectures 1–2 and observed by every notebook thereafter: split
before anything is fitted; all preprocessing inside a `Pipeline` passed to
cross-validation; nothing derived from the test set in the training path; fixed
seeds; per-fold scores reported, not only the mean.

**Presentation convention:** slides and notebooks never name a weekday. Lectures
refer to one another relatively — *in the next lecture*, *in Lecture 8* — so the
material is independent of the timetable.

---

# Part I — Tabular data and classical models
*Lectures 1–8 · Chapters 1–8 · CPU only*

## Lecture 1 — What machine learning is, and how we will work
**Ch 1–2 · California housing**

What a learning system is and when one is the right tool; supervised,
unsupervised and self-supervised; instance-based against model-based learning;
the standard failure modes stated up front — insufficient data, unrepresentative
data, overfitting, underfitting. Training, validation and test sets, and why the
test set is spent once.

Then the course's working method: how to specify a piece of code, how to read
what an assistant returns, how to test it against a case whose answer is known.
Assessment and scope.

The application starts: the valuation brief, the shape of the data, the
histograms, the capped target, and the stratified split.

## Lecture 2 — The end-to-end project
**Ch 2 · California housing**

*Mathematics: least squares and the normal equation.* θ̂ = (XᵀX)⁻¹Xᵀy as the
solution of a projection problem, the condition for XᵀX to be invertible, and
the SVD-based pseudoinverse when it is not.

The full pipeline: imputation, scaling, categorical encoding, `ColumnTransformer`
and `Pipeline`; cross-validation; grid and randomised search; the final
evaluation on the test set with a confidence interval. RMSE and MAE, and which
one the brief actually asks for.

## Lecture 3 — Classification and its metrics
**Ch 3 · MNIST**

Binary classification, then multiclass. The confusion matrix. Precision, recall
and F1, and what each is blind to.

*Mathematics: why accuracy fails under imbalance, and why precision is not
monotone in the threshold.* The always-negative classifier attains the base
rate. As the threshold rises recall is monotone — the denominator is fixed —
while precision has both numerator and denominator moving.

The precision/recall trade-off, PR and ROC curves, when to prefer each, choosing
an operating point and defending it. Multiclass, multilabel and multioutput.

## Lecture 4 — Training models
**Ch 4 · Titanic**

*Mathematics: gradient descent.* The gradient of the mean squared error; batch,
mini-batch and stochastic descent; the learning rate and what too large and too
small each look like; convergence on a convex surface.

Linear regression by descent rather than by the normal equation, and when each
is preferable. Polynomial features. Logistic regression: the logistic function,
the log-loss, and reading coefficients as log-odds. Softmax regression for more
than two classes. Decision boundaries, and calibrated probabilities against bare
labels.

## Lecture 5 — Regularisation and the bias–variance trade-off
**Ch 4 · Titanic**

*Mathematics: the bias–variance decomposition.* The three terms — squared bias,
variance, irreducible error — and how each maps onto a learning curve: both
curves high and close together is bias; a persistent gap is variance.

Learning curves and validation curves, read properly. Ridge, lasso and elastic
net; what the ℓ₁ and ℓ₂ penalties each do to the coefficients and why lasso
selects; early stopping. Ridge closes Lecture 2: adding αI makes the matrix
invertible for every α > 0.

## Lecture 6 — Decision trees
**Ch 5 · CoverType**

*Mathematics: impurity.* Gini and entropy, what each measures, why they usually
select the same split, and the greedy CART objective.

Growing a tree, reading one, tracing a single prediction to a human-readable
justification. The regularisation hyperparameters and what each controls.
Sensitivity to rotation and to small changes in the data — the property that
motivates the next lecture. Trees for regression.

## Lecture 7 — Ensembles and random forests
**Ch 6 · CoverType**

*Mathematics: the variance of an average of correlated predictors.* Averaging
destroys the independent component of the variance and leaves the correlated
part untouched — one result that explains bagging, the feature subsampling in
random forests, and the random thresholds in extra-trees.

Voting classifiers, bagging and pasting, out-of-bag evaluation, random forests,
feature importance. Boosting: AdaBoost, gradient boosting, and the
histogram-based implementations used in practice. Stacking.

## Lecture 8 — Dimensionality reduction and unsupervised learning
**Ch 7–8 · Olivetti faces**

*Mathematics: PCA via the SVD.* The sense in which the principal subspace
minimises reconstruction error, the explained variance ratio, and the
Johnson–Lindenstrauss bound — whose required dimensionality depends on the
number of points and the tolerance, not on the original dimensionality.

PCA, randomised and incremental PCA, random projection, and the speed/quality
trade-off measured rather than asserted. Then clustering: k-means, choosing k
by inertia and properly by silhouette, DBSCAN and Gaussian mixtures for
non-spherical structure. Anomaly detection by density and by reconstruction
error. Semi-supervised learning by label propagation.

---

# Part II — Neural networks
*Lectures 9–11 · Chapters 9–11 · CPU*

## Lecture 9 — Neural networks, from the perceptron up
**Ch 9 · Fashion-MNIST**

The biological analogy and where it stops. The perceptron, what it can and
cannot separate, and the multilayer perceptron. Activation functions and why a
network of linear layers is a linear model.

*Mathematics: what a layer computes.* The affine map plus non-linearity, in
matrix form, with the shapes tracked explicitly.

The architecture tables for regression and for classification — output units,
output activation, loss — as a reference to return to. A first network with
Scikit-Learn's MLP, tuned by hand, and an honest account of why that is not how
this is done in practice.

## Lecture 10 — PyTorch
**Ch 10 · Fashion-MNIST**

*Mathematics: backpropagation as reverse-mode automatic differentiation.* Why
the chain rule can be applied in two directions; why forward mode costs one pass
per input and reverse mode one pass per output; and why a scalar loss over
millions of parameters makes reverse mode the only viable choice.

Tensors, devices and hardware acceleration; autograd and what `requires_grad`
builds; the training loop written out in full, then `DataLoader`, `Dataset` and
custom `Module`s. Evaluation, checkpointing and saving. Hyperparameter search
with Optuna. The two silent bugs everyone writes once — a missing `zero_grad()`
and a missing `model.eval()` — named, so they are recognised when met.

## Lecture 11 — Training deep networks
**Ch 11 · CIFAR-10**

*Mathematics: variance propagation through layers.* How the variance of a
layer's output depends on fan-in and weight variance; why preserving the forward
signal and the backward gradient impose conflicting requirements; Glorot's
compromise and He's adjustment for ReLU; and why a scale error compounds
geometrically with depth.

Vanishing and exploding gradients, instrumented and measured rather than
asserted. Initialisation, activation functions, batch and layer normalisation,
gradient clipping. Faster optimisers — momentum, RMSProp, Adam, AdamW — and
learning-rate schedules. Regularisation: ℓ₂, dropout, and max-norm.

---

# Part III — Computer vision
*Lectures 12–14 · Chapter 12 · CPU*

## Lecture 12 — Convolutional networks
**Ch 12 · Flowers102**

*Mathematics: weight sharing, equivariance and memory.* The parameter count of
a dense layer against a convolutional layer on the same image; translation
equivariance as a property of convolution; equivariance against invariance, and
why pooling's invariance helps classification and hurts segmentation. Then the
RAM calculation showing that activations, not parameters, exhaust the GPU.

Convolutional layers, filters, feature maps, stride and padding; pooling; the
classic architectures and what each contributed. A network built and trained
from scratch, with its cost stated.

## Lecture 13 — Transfer learning
**Ch 12 · Flowers102**

Why features learned on one corpus transfer to another, and how far down the
stack that holds. Pretrained backbones, layer freezing and progressive
unfreezing, differential learning rates. Data augmentation: what to apply, what
it implicitly asserts about the task, and what it costs. The same accuracy as
Lecture 12 in a fraction of the training time, measured side by side.

## Lecture 14 — Detection and segmentation
**Ch 12 · COCO**

*Mathematics: IoU and mAP.* Intersection over union, and why it provides no
gradient when boxes are disjoint regardless of how far apart they are; GIoU and
CIoU as repairs. Then average precision, defined by the maximum precision at or
above each recall level — a definition that exists precisely to repair the
non-monotonicity established in Lecture 3 — and mAP averaged over classes and
IoU thresholds.

Localisation, detection, and the pretrained detectors worth knowing.
Non-maximum suppression. Semantic against instance segmentation. Object
tracking, briefly.

---

# Part IV — Sequences and language
*Lectures 15–18 · Chapters 13–15 · CPU*

## Lecture 15 — Time series
**Ch 13 · Chicago transit ridership**

*Mathematics: stationarity, differencing and autocorrelation.* What
stationarity requires and which models depend on it; how differencing removes a
polynomial trend and seasonal differencing a periodic one; the autocorrelation
function.

Naive and seasonal-naive forecasts as the baselines to beat, and why they are
strong. Linear models on lagged features. **Backtesting** — why a random split
is invalid on autocorrelated data, and how to evaluate on a rolling origin
instead. ARMA and its relatives, for context.

## Lecture 16 — Recurrent networks
**Ch 13 · Chicago transit ridership**

Recurrent cells and unrolling through time; training by backpropagation through
time and why it is unstable. Deep RNNs. Multivariate inputs. Forecasting several
steps ahead: recursive against direct, and sequence-to-sequence. LSTM and GRU
cells, gate by gate, and what each gate is for. Dilated convolutions as a
non-recurrent alternative.

## Lecture 17 — Text
**Ch 14 · IMDb**

*Mathematics: softmax, cross-entropy and logits.* Softmax and its invariance
under a constant shift; cross-entropy against a one-hot target; the gradient
with respect to the logits, which reduces to prediction minus target. Why the
loss consumes logits rather than probabilities — the cancellation, and the
numerical stability of the combined form. Cross-entropy and KL divergence.

Subword tokenisation; trainable embeddings and what the embedding space
encodes; a recurrent text classifier; then pretrained embeddings, and the same
architecture with a better starting point.

## Lecture 18 — Attention and transformers
**Ch 14–15 · IMDb**

*Mathematics: scaled dot-product attention.* Queries, keys and values; why the
scores are divided by √d_k; multi-head attention as several projections of the
same sequence; positional encoding, and why a permutation-invariant model needs
it.

The encoder–decoder architecture; encoder-only, decoder-only and
encoder–decoder families and what each is for. Fine-tuning a pretrained
transformer for the task, against the from-scratch model of Lecture 17. Hugging
Face as the working interface.

---

# Part V — Information retrieval and recommender systems
*Lectures 19–22 · Lecture notes · Examinable*

These four lectures sit outside the primary text. They are here because they are
where the embedding machinery of Part IV earns its living, and because search
and recommendation are the two applications of machine learning most students
will actually meet.

## Lecture 19 — Information retrieval: the lexical foundation
**Lecture notes · SciFact (BEIR)**

The retrieval problem: a query, a corpus, a ranking. Why it is not
classification. The inverted index and why retrieval is cheap. Term weighting:
term frequency, inverse document frequency, length normalisation, and BM25
derived from what each of its parameters is for.

*Mathematics: evaluating a ranking.* Precision@k and recall@k and why neither
suffices; reciprocal rank; average precision again, now over a ranked list;
discounted cumulative gain and its normalisation. Why graded relevance changes
the answer, and how relevance judgements are actually produced — pooling, and
what pooling misses.

## Lecture 20 — Information retrieval: dense retrieval
**Lecture notes · SciFact (BEIR)**

The vocabulary mismatch problem, and why a lexical index cannot solve it.
Bi-encoders: encode the corpus once, encode the query at request time, retrieve
by inner product. Training a bi-encoder with in-batch negatives, and why the
choice of negatives is most of the work. Approximate nearest-neighbour search
and the recall/latency trade-off.

Cross-encoders and re-ranking: why the accurate model cannot be the first stage.
Hybrid lexical–dense retrieval. All three evaluated against the BM25 baseline of
Lecture 19 on the same judgements.

## Lecture 21 — Recommender systems: from ratings to factors
**Lecture notes · MovieLens**

The recommendation problem and how it differs from retrieval: no query, and the
feedback is a by-product of use. Explicit ratings against implicit feedback, and
why implicit data has no negatives. Neighbourhood methods, user-based and
item-based.

*Mathematics: matrix factorisation.* The low-rank model, its objective, and its
relationship to the SVD of Lecture 8 — including why the missing entries mean
the SVD cannot simply be taken. Alternating least squares and SGD. Bias terms,
and why they carry more of the signal than anyone expects. BPR and pairwise
ranking losses for implicit feedback.

## Lecture 22 — Recommender systems: neural, and evaluated honestly
**Lecture notes · MovieLens**

Two-tower models — the same architecture as Lecture 20's bi-encoder, with users
in place of queries — and why retrieval and recommendation converge here.
Negative sampling and the sampled-softmax correction. Sequential
recommendation, briefly.

Then evaluation, which is where recommender systems are usually wrong: leave-one-out
against temporal splits, sampled metrics and their bias, popularity bias, the
cold-start problem, and why an offline gain need not survive contact with users.

---

# Part VI — Multimodal models, and closing the course
*Lectures 23–24 · Chapters 15–16 · CPU*

## Lecture 23 — Vision transformers and multimodal retrieval
**Ch 15–16 · COCO**

The vision transformer: an image as a sequence of patches, and what it gives up
relative to a convolutional network. Encoding images and text separately, and
the discovery that the two spaces are unrelated.

*Mathematics: the contrastive objective and its temperature.* Why embeddings are
normalised onto the unit sphere; why unrelated pairs should target zero rather
than −1 — the concentration result of Lecture 8 in a new guise; the learned
temperature as a rescaling of similarities into usable logits; and why the number
of negatives, and therefore the batch size, sets the difficulty of the task.

Jointly trained dual encoders. Zero-shot classification and text-to-image
retrieval, evaluated with the ranking metrics of Lecture 19.

## Lecture 24 — Generation, retrieval-augmented systems, and where this leaves you
**Ch 15–16 · COCO + the Part V corpora**

Captioning with a multimodal model, for catalogue entries with no usable
description. Then retrieval-augmented generation: the retriever of Lecture 20,
the generator of Lecture 18, and the failure modes that belong to the join
rather than to either half.

The closing 25 minutes: the course as one argument, from the normal equation to
the contrastive objective; an explicit statement of what was not covered —
reinforcement learning, generative image models, causality, fairness and model
governance — and where each sits in the field; how to keep learning; and what
the examination will ask.

---

## Coverage map

| Chapter | Lectures |
|---|---|
| 1 · The ML landscape | 1 |
| 2 · End-to-end project | 1, 2 |
| 3 · Classification | 3 |
| 4 · Training models | 4, 5 |
| 5 · Decision trees | 6 |
| 6 · Ensembles and random forests | 7 |
| 7 · Dimensionality reduction | 8 |
| 8 · Unsupervised learning | 8 |
| 9 · Introduction to ANNs | 9 |
| 10 · Building networks with PyTorch | 10 |
| 11 · Training deep networks | 11 |
| 12 · Deep computer vision | 12, 13, 14 |
| 13 · Sequences | 15, 16 |
| 14 · NLP with RNNs and attention | 17, 18 |
| 15 · Transformers | 18, 23, 24 |
| 16 · Vision and multimodal transformers | 23, 24 |
| *(lecture notes)* · Information retrieval | 19, 20 |
| *(lecture notes)* · Recommender systems | 21, 22 |

## Datasets

| Dataset | Lectures | Task |
|---|---|---|
| California housing | 1, 2 | regression |
| MNIST | 3 | classification, metrics |
| Titanic | 4, 5 | classification, regularisation |
| CoverType | 6, 7 | trees, ensembles |
| Olivetti faces | 8 | dimensionality reduction, clustering |
| Fashion-MNIST | 9, 10 | first networks, PyTorch |
| CIFAR-10 | 11 | deep training |
| Flowers102 | 12, 13 | convolution, transfer learning |
| COCO | 14, 23, 24 | detection, multimodal |
| Chicago transit ridership | 15, 16 | forecasting |
| IMDb | 17, 18 | text |
| SciFact (BEIR) | 19, 20 | retrieval |
| MovieLens | 21, 22 | recommendation |

## Where the mathematics is

Eighteen derivations, each in the lecture whose method rests on it. They are
cross-referential and the order matters.

| Lecture | Object |
|---|---|
| 2 | least squares and the normal equation |
| 3 | imbalance, and the non-monotonicity of precision |
| 4 | gradient descent |
| 5 | the bias–variance decomposition |
| 6 | impurity: Gini and entropy |
| 7 | the variance of an average of correlated predictors |
| 8 | PCA via the SVD; Johnson–Lindenstrauss |
| 9 | what a layer computes |
| 10 | backpropagation as reverse-mode autodiff |
| 11 | variance propagation and weight initialisation |
| 12 | weight sharing, equivariance and memory |
| 14 | IoU's vanishing gradient; mAP |
| 15 | stationarity, differencing and autocorrelation |
| 17 | softmax, cross-entropy and logits |
| 18 | scaled dot-product attention |
| 19 | evaluating a ranking: MRR, AP, NDCG |
| 21 | matrix factorisation and its relation to the SVD |
| 23 | the contrastive objective and its temperature |

Dependencies worth preserving: Lecture 5 completes Lecture 2; Lecture 14 uses
Lecture 3; Lecture 21 and Lecture 23 both use Lecture 8; Lecture 20 and Lecture
22 are the same architecture; Lecture 23 uses Lecture 19's metrics.
