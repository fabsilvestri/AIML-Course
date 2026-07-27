# Machine Learning Applications — Lecture Plan

**Programme:** BSc Mathematics of Artificial Intelligence, third year
**Load:** 48 academic hours = 24 lectures × 2 academic hours (90 minutes each)
**Textbook:** A. Géron, *Hands-On Machine Learning with Scikit-Learn and PyTorch*, O'Reilly, 2025 — Chapters 1–16
**Mode:** AI-assisted development. Students specify, review, debug and verify; they do not type from scratch.

---

## Organising principle

Chapters never introduce themselves. **Every method enters the course at the moment an application breaks without it.**

Applications are covered in pairs of consecutive lectures:

| | Lecture type | Shape (90 min) |
|---|---|---|
| **A** | **Build** | 15 min problem, data and stakeholder · 15 min choose a metric and commit to a target number *in writing* · 60 min build the simplest thing that runs |
| **B** | **Break → Fix** | 20 min mathematical thread · 20 min diagnose why the number was wrong · 35 min the method that fixes it · 15 min re-measure and red-team a peer's notebook |

The number committed at the end of every A lecture is what makes the following B lecture land: students predicted, they were wrong, and now they want to know why.

The **mathematical thread** at the top of each B lecture develops exactly one object, chosen because the application just built depends on it. The twelve threads reference one another and are intended to be taken in order.

All datasets are ones used or named in the textbook.

---

## Working method — AI-assisted development

The book supplies the syllabus. **It does not supply the notebooks.** No notebook
from the author, or from any other third party, is used in this course. Every
notebook is written during the lecture, from a specification, with AI assistance
— and every lecture shows that process rather than describing it.

The loop, run explicitly in each lecture:

| # | Step | Who |
|---|---|---|
| 1 | **Specify** — input, output, constraint, check | student |
| 2 | **Generate** — and then stop, before running | assistant |
| 3 | **Read** — the five reviewer questions | student |
| 4 | **Test** — against a case whose answer is known | student |
| 5 | **Verify** — that the number means what it appears to | student |

Each lecture carries at least one **worked assistant failure**: a prompt that is
under-specified in one place, the plausible code it returns, the review question
that catches it, and the corrected specification. The failures are chosen to be
the ones that lecture is about, so the demonstration is never decorative.

Standing constraint, extended application by application, and repeated in every
deck from Lecture 2 onward: split before anything is fitted; all preprocessing
inside a `Pipeline` passed to cross-validation; nothing derived from the test set
in the training path; fixed seeds; report per-fold scores, not only the mean.

**Presentation convention:** slides and notebooks never name a weekday. Lectures
refer to one another relatively — *in the next lecture*, *in the previous
lecture*, *in two lectures* — so the material is independent of the timetable.

---

# Part I — Tabular data and classical models
*Lectures 1–10 · Chapters 1–8 · CPU only*

## Lecture 1 — Welcome, and a price you can't trust
**Build · California housing · Ch 1–2**

Opens with 20 minutes of course welcome: what the course is and is not, the AI-assisted working method and its rules, how assessment works, and an explicit statement of scope — this is a course on applied machine learning as covered in Chapters 1–16, not a survey of the field.

Then straight into the first application. Students are handed a real estate valuation brief, look at the data structure, plot histograms, and notice the capped target and the skewed features. They pick RMSE, commit to a target, and build an end-to-end regression by the end of the session. Everyone leaves with a number they are pleased with.

## Lecture 2 — Your RMSE was a lie
**Fix · Ch 2**

*Mathematical thread: least squares and the normal equation.* The closed-form solution θ̂ = (XᵀX)⁻¹Xᵀy, the condition for XᵀX to be invertible, and what to do when it is not — the SVD-based pseudoinverse. This explains the textbook's warning against engineering features as weighted sums of existing ones.

The diagnosis: scaling was fitted before the split, the test set was consulted more than once, and the split was not stratified on the strongest predictor. Repair with pipelines, `ColumnTransformer`, stratified sampling, cross-validation and grid search. The honest number is worse than the one they committed to, and that is the lesson.

## Lecture 3 — Finding the rare event
**Build · MNIST binary detector · Ch 3**

A rare-event detection brief. Students build a binary classifier and evaluate it with accuracy using cross-validation. It reaches well above 90%. They commit to that number as their headline result and go home satisfied.

## Lecture 4 — It never fires
**Fix · Ch 3**

*Mathematical thread: why accuracy fails under imbalance, and why precision is not monotone in the threshold.* A classifier that always predicts the negative class attains the base rate. Then: as the decision threshold rises, recall is monotone because the denominator is fixed, while precision has both numerator and denominator moving — with the textbook's own counterexample, 4/5 falling to 3/4.

Repair with the confusion matrix, precision, recall, F1, the precision/recall trade-off, PR and ROC curves, and when to prefer each. Students tune a threshold to a stated operating point and defend the choice.

## Lecture 5 — Who survives, and how sure are we?
**Build · Titanic · Ch 4**

The brief demands calibrated probabilities and a defensible cut-off, not bare labels — so logistic regression. Students engineer features, fit the model, read off coefficients, and plot decision boundaries. They then push model complexity up with polynomial features until validation performance starts to fall, and record both curves.

## Lecture 6 — Reading a learning curve
**Fix · Ch 4**

*Mathematical thread: the bias–variance decomposition.* Derive the three terms — squared bias, variance, irreducible error — and map them onto the two curves the students plotted in the build session: both plateauing high and close together means bias; a persistent gap means variance.

Repair with ridge, lasso and elastic net, plus early stopping. Ridge is also the callback to Lecture 2: adding αA makes the matrix invertible for every α > 0.

## Lecture 7 — A model the regulator will accept
**Build · CoverType · Ch 5**

Land-classification brief with an interpretability requirement: every prediction must come with a human-readable justification. Students train a decision tree, export and read it, trace individual predictions down the tree, and tune the regularisation hyperparameters. Accuracy is modest but the rules are legible.

## Lecture 8 — Retrain it and watch it change
**Fix · Ch 5–6**

*Mathematical thread: impurity, and why averaging reduces variance.* Gini and entropy as impurity measures and why they usually agree. Then the variance of an average of correlated predictors — generalising the textbook's two-regressor calculation — showing that averaging destroys the independent component of the variance and leaves the correlated part untouched.

That single result explains bagging, random forests' feature subsampling, and extra-trees' random thresholds: all three attack the correlation term. Students rebuild with ensembles, recover the accuracy, lose the interpretability, and get part of it back through feature importance.

## Lecture 9 — Forty labels for four hundred faces
**Build · Olivetti faces · Ch 8**

An identity-grouping brief where labelling is expensive: the corpus is unlabelled apart from a handful of examples. Students cluster with k-means, choose k using inertia and the elbow, then properly with silhouette scores and silhouette diagrams, and inspect the clusters visually. The pipeline is correct and unbearably slow.

## Lecture 10 — Four thousand dimensions is too many
**Fix · Ch 7–8**

*Mathematical thread: SVD, PCA, and Johnson–Lindenstrauss.* PCA via SVD, the sense in which it minimises reconstruction error, and the explained variance ratio. Then the Johnson–Lindenstrauss bound as the textbook states it — with the point students must notice: the required dimensionality depends on the number of points and the tolerance, **not** on the original dimensionality.

Repair by compressing first: PCA, randomised and incremental PCA, and random projection, with the speed/quality trade-off measured rather than asserted. Then the pipeline extends — DBSCAN and Gaussian mixtures for non-spherical structure, anomaly detection by density and by reconstruction error, and label propagation from the forty labels.

---

# Part II — Neural networks
*Lectures 11–14 · Chapters 9–11 · GPU from Lecture 11 onward*

## Lecture 11 — The first neural network
**Build · Fashion MNIST · Ch 9**

A document-sorting brief. Students build a multilayer perceptron with Scikit-Learn, tune the hidden layers and learning rate by hand, and get it working. Along the way they meet the perceptron, the multilayer perceptron, and the architecture tables for regression and classification networks. It works — and it is slow, CPU-bound, and impossible to modify.

## Lecture 12 — Rebuilding it in PyTorch
**Fix · Ch 9–10**

*Mathematical thread: backpropagation as reverse-mode automatic differentiation.* Why the chain rule can be applied in two directions; why the cost of forward mode scales with the number of inputs and reverse mode with the number of outputs; and why training — with millions of parameters and a single scalar loss — makes reverse mode the only viable choice.

Rebuild in PyTorch: tensors, hardware acceleration, autograd, the training loop, `DataLoader`, custom modules, evaluation. Students see why `requires_grad` builds a graph and why forgetting `zero_grad()` produces a silent, uncrashing bug. Close with Optuna for hyperparameter search and saving the model.

## Lecture 13 — Twenty layers, no learning
**Build · CIFAR-10 · Ch 11**

A harder recognition brief with colour images and more classes. Students deliberately build a deep stack — twenty hidden layers — and train it. Loss barely moves. They instrument the network, log per-layer activation and gradient statistics, and observe the signal disappearing as it descends. They leave with a diagnosis they cannot yet explain.

## Lecture 14 — Making it train
**Fix · Ch 11**

*Mathematical thread: variance propagation through layers.* How the variance of a layer's output relates to fan-in and the weight variance; why preserving the forward signal and the backward gradient impose conflicting requirements; Glorot's compromise and He's adjustment for ReLU. Then the consequence: a scale error compounds geometrically with depth, which is precisely the attenuation they measured in the build session.

Repair with initialisation, better activation functions, batch normalisation and layer normalisation, gradient clipping, faster optimisers, learning-rate schedules, and regularisation by dropout. The network trains.

---

# Part III — Computer vision
*Lectures 15–18 · Chapter 12*

## Lecture 15 — Visual inspection
**Build · Flowers102 · Ch 12**

A visual quality-control brief. Students build a convolutional network from scratch — convolutional layers, filters, feature maps, pooling — and train it on a small labelled set. It reaches middling accuracy after a long wait. They commit to that number.

## Lecture 16 — Don't train from scratch
**Fix · Ch 12**

*Mathematical thread: weight sharing, equivariance, and where the memory goes.* The parameter count of a dense layer against a convolutional layer on the same image; translation equivariance as a property of convolution; the distinction between equivariance and invariance, and why pooling's invariance is desirable for classification and harmful for segmentation. Then the RAM calculation showing that activations, not parameters, exhaust the GPU.

Repair with transfer learning from a pretrained backbone, layer freezing, differential learning rates, and data augmentation. Accuracy jumps in a fraction of the training time.

## Lecture 17 — Where is it, exactly?
**Build · COCO · Ch 12**

The brief now requires locating objects, not just naming them. Students run a pretrained detector over a corpus, visualise the predicted boxes, and count objects per image. Some boxes are visibly wrong, and students have no principled way to say *how* wrong. They propose a metric and commit to it.

## Lecture 18 — Scoring a box, scoring a detector
**Fix · Ch 12**

*Mathematical thread: IoU's vanishing gradient, and mAP as a mean of a mean.* Intersection over union, and why it provides no gradient when boxes are disjoint regardless of separation; how GIoU and CIoU repair this. Then average precision — defined using the maximum precision at or above each recall level, which exists precisely to repair the non-monotonicity proved in Lecture 4 — and mAP averaged over classes and over IoU thresholds.

Repair the evaluation, then extend to non-maximum suppression, object tracking, and per-pixel prediction: semantic versus instance segmentation.

---

# Part IV — Sequences, language and multimodality
*Lectures 19–24 · Chapters 13–16*

## Lecture 19 — Forecasting demand
**Build · Chicago transit ridership · Ch 13**

A capacity-planning brief. Students plot the series, spot weekly and yearly seasonality, and build a naive forecast that turns out to be hard to beat. They then fit a linear model and a first recurrent network, evaluate with a random cross-validation split, and record an excellent score.

## Lecture 20 — You forecast the past
**Fix · Ch 13**

*Mathematical thread: stationarity, differencing and autocorrelation.* What stationarity requires and why models depend on it; how differencing removes polynomial trends and seasonal differencing removes periodic structure; autocorrelation as the reason the naive forecast was strong. Then the leakage argument: with positive autocorrelation, a random split places a point's own near-future in the training set, so the reported score is optimistically biased.

Repair the evaluation with proper time-based backtesting, then improve the model: deep RNNs, multivariate inputs, forecasting several steps ahead, sequence-to-sequence, LSTM and GRU cells, and a convolutional alternative.

## Lecture 21 — Reading the customers
**Build · IMDb · Ch 14**

A feedback-analytics brief on free text. Students meet subword tokenisation and trainable embeddings, build a recurrent classifier from scratch, and get a workable but unremarkable result. They then swap in a pretrained tokenizer and pretrained embeddings and watch the number move without touching the architecture.

## Lecture 22 — Reusing what someone else learned
**Fix · Ch 14–15**

*Mathematical thread: cross-entropy, softmax and logits.* Softmax and its invariance under a constant shift; cross-entropy against a one-hot target; the gradient with respect to the logits, which reduces to prediction minus target. Then why the loss consumes logits rather than probabilities — the cancellation and the numerical stability of the combined form — and the relationship between cross-entropy and KL divergence.

Fine-tune a pretrained transformer for the task and beat the from-scratch model decisively. Then extend beyond classification: sentence embeddings for semantic search over the feedback corpus, and clustering to group recurring complaints.

## Lecture 23 — One catalogue, two modalities
**Build · COCO · Ch 15–16**

A product-catalogue brief where the entries are images and the queries are text. Students encode images with a vision transformer, encode text with a sentence encoder, discover the two spaces are unrelated, and then bring in a jointly trained dual encoder. Zero-shot classification and text-to-image retrieval work immediately. They commit to a retrieval metric and measure it.

## Lecture 24 — Closing the loop, and closing the course
**Fix + course summary · Ch 15–16**

*Mathematical thread: the contrastive objective and its temperature.* Why embeddings are normalised onto the unit sphere; why unrelated pairs should target zero rather than −1, which is the concentration result from Lecture 10 returning in a new guise; the role of the learned temperature in rescaling similarities into usable logits; and why the number of negatives — and therefore the batch size — governs the difficulty of the task.

Repair the weak spots: entries with no usable description get automatic captions from a multimodal model, and ambiguous queries are handled by retrieval-augmented generation over the catalogue.

The final 25 minutes sum up the course as a single argument: every method met over twelve applications was introduced because something measurable broke without it. Review of the thread of failures — leakage, imbalance, variance, vanishing signal, wrong metric, temporal leakage — and where each recurs outside the applications used here. Explicit statement of what the course did not cover and where in the field those gaps sit.

---

## Coverage map

| Chapter | Lectures |
|---|---|
| 1 · The ML landscape | 1 |
| 2 · End-to-end project | 1, 2 |
| 3 · Classification | 3, 4 |
| 4 · Training models | 5, 6 |
| 5 · Decision trees | 7, 8 |
| 6 · Ensembles and random forests | 8 |
| 7 · Dimensionality reduction | 10 |
| 8 · Unsupervised learning | 9, 10 |
| 9 · Introduction to ANNs | 11, 12 |
| 10 · Building networks with PyTorch | 12 |
| 11 · Training deep networks | 13, 14 |
| 12 · Deep computer vision | 15, 16, 17, 18 |
| 13 · Sequences | 19, 20 |
| 14 · NLP with RNNs and attention | 21, 22 |
| 15 · Transformers | 22, 23, 24 |
| 16 · Vision and multimodal transformers | 23, 24 |

## The twelve mathematical threads, in order

1. Least squares and the normal equation *(L2)*
2. Imbalance, and the non-monotonicity of precision *(L4)*
3. The bias–variance decomposition *(L6)*
4. Impurity, and variance reduction by averaging *(L8)*
5. SVD, PCA and Johnson–Lindenstrauss *(L10)*
6. Backpropagation as reverse-mode autodiff *(L12)*
7. Variance propagation and weight initialisation *(L14)*
8. Weight sharing, equivariance and memory *(L16)*
9. IoU's vanishing gradient; mAP *(L18)*
10. Stationarity, differencing and temporal leakage *(L20)*
11. Cross-entropy, softmax and logits *(L22)*
12. The contrastive objective and its temperature *(L24)*

The threads are cross-referential and should not be reordered. Thread 2 is used by thread 9; thread 1 is completed by thread 3; thread 5 returns in thread 12; thread 4 explains the ensemble variants used from Lecture 8 onward.
