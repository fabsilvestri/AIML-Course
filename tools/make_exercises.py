#!/usr/bin/env python3
"""
Five exam-style exercises at the end of every deck, answered one lecture later.

    python3 tools/make_exercises.py          # rewrite every deck
    python3 tools/make_exercises.py 7 8      # just these

THE DESIGN, set by the lecturer 2026-09-03.

  * Every deck, 1 to 24, ends with five exercises in the style of the written
    paper. They are NOT presented -- they are there for a student reading the
    deck afterwards.
  * Lecture N's solutions appear on lecture N+1's deck, so a student has to
    attempt them before the answer is available.
  * Lecture 24 is the exception: no lecture 25 exists, so it carries lecture
    23's solutions, its own five, AND its own solutions.

This extends a pattern the decks already had. Lecture 1 sets a specimen Part B
question whose last two parts are deliberately unanswerable until
cross-validation exists, and Lecture 2 answers them -- "◇ Answer — 3" and
"◇ Answer — 4". The exercises below use the same slide vocabulary, so a
student cannot tell the new ones from the ones that were always there.

WHY GENERATED. The same reason index.html is: 24 decks x (5 questions + 5
solutions) is 240 near-identical blocks, hand-editing them is how a solution
ends up on the wrong deck, and a solution on the wrong deck is worse than no
solution at all. Everything lives in EXERCISES below, and the injection is
idempotent between BEGIN/END markers.
"""
from __future__ import annotations

import html
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BEGIN, END = "<!-- BEGIN EXERCISES -->", "<!-- END EXERCISES -->"

# ----------------------------------------------------------------------
# One entry per exercise:
#   q     the question, as HTML. Keep it to a sentence or two -- the paper does.
#   marks what it is worth, so the length of the expected answer is obvious.
#   a     the answer, in as few words as the question allows.
#   why   the reasoning, one bullet per step. Two or three, never five.
# ----------------------------------------------------------------------
EXERCISES: dict[int, list[dict]] = {}


def ex(n, q, marks, a, why):
    EXERCISES.setdefault(n, []).append(
        dict(q=q, marks=marks, a=a, why=why))


# ------------------------------------------------------------------ L01
ex(1, "A colleague reports that their model reaches an RMSE of $48{,}000$ on "
      "the California housing data, and that they chose the model by trying "
      "six of them and keeping the best. State, in one sentence, what that "
      "number is an estimate of &mdash; and what it is <em>not</em> an "
      "estimate of.", 4,
   "It estimates the best-of-six score on that particular split; it does not "
   "estimate the error on new data.",
   ["Choosing the minimum of six numbers measured on the same data makes that "
    "minimum a <em>selection</em>, not a measurement",
    "The quantity a client cares about &mdash; error on districts nobody has "
    "seen &mdash; needs data that took part in no choice"])
ex(1, "The median income column is capped at 15. Give one consequence for the "
      "<em>model</em> and one for the <em>evaluation</em>, and say which of "
      "the two you would raise with the client first.", 5,
   "Model: it can never learn what happens above the cap. Evaluation: the test "
   "set is capped too, so the error is silent about exactly those districts.",
   ["The cap is in both halves of the split, so no held-out number reveals it",
    "Raise the evaluation one first: the model can be retrained, but a metric "
    "that cannot see the failure will not tell you to"])
ex(1, "You are given a stratified split on income category and a naive random "
      "split of the same data. Both report similar RMSE. Does that mean the "
      "stratification was unnecessary? Answer yes or no, with a reason.", 4,
   "No.",
   ["Similar RMSE on <em>one</em> draw says nothing about the spread over "
    "draws, and the stratification is there to reduce that spread",
    "The failure a naive split causes is occasional and large, which is "
    "precisely what a single comparison cannot show"])
ex(1, "The brief says the output feeds a downstream system that expects a "
      "price. Name the one thing this fact settles about the problem framing, "
      "and one thing it leaves open.", 4,
   "It settles that this is supervised regression, not classification. It "
   "leaves the metric open.",
   ["'A price' fixes the type of the output and nothing about how error is "
    "weighed",
    "Whether being wrong by $50{,}000$ on a cheap district matters as much as "
    "on an expensive one is a question for the client, not the data"])
ex(1, "In the working loop &mdash; specify, generate, read, test, verify "
      "&mdash; exactly one step is done by the assistant. Name it, and say "
      "what goes wrong if a student treats a second step as the "
      "assistant&rsquo;s.", 3,
   "Generate. If reading is also delegated, nothing checks that the code does "
   "what the specification asked for.",
   ["The specification is the student's claim about what the code must do",
    "Only a reader who holds that claim can notice the code meeting a "
    "different one"])



# ------------------------------------------------------------------ L02
ex(2, "The normal equation is $\\hat\\theta = (X^{\\mathsf T}X)^{-1}X^{\\mathsf T}y$. "
      "Your design matrix has a column equal to the sum of two others. State "
      "what happens to $X^{\\mathsf T}X$, and what <code>np.linalg.pinv</code> "
      "returns instead.", 5,
   "$X^{\\mathsf T}X$ is singular; the pseudoinverse returns the minimum-norm "
   "least-squares solution.",
   ["A linear dependence among columns makes $X$ rank-deficient, so "
    "$X^{\\mathsf T}X$ has a zero eigenvalue and no inverse",
    "The fitted <em>values</em> are still unique &mdash; the projection onto the "
    "column space is &mdash; only the coefficients are not"])
ex(2, "A pipeline scales the features and then fits a ridge model, and is "
      "passed whole to <code>cross_val_score</code>. Explain what would go "
      "wrong if the scaling were done once, before the call.", 5,
   "The scaler would be fitted on every fold&rsquo;s validation rows, so each "
   "fold&rsquo;s score would be optimistic.",
   ["A scaler is a fitted object: its mean and variance are parameters learned "
    "from data",
    "Fitting it before the split lets validation rows influence the "
    "transformation applied to themselves"])
ex(2, "You report a 95% confidence interval for the test RMSE. A colleague "
      "says the interval proves the model is better than the baseline. Give "
      "the one question you would ask before agreeing.", 4,
   "Was the baseline scored on the same test rows?",
   ["An interval describes the uncertainty of <em>one</em> estimate, not the "
    "difference between two",
    "Comparing two systems needs their difference&rsquo;s spread, which needs "
    "them measured on the same rows"])
ex(2, "k-fold cross-validation reports a mean of 0.71 with folds "
      "$[0.52, 0.79, 0.75, 0.74, 0.75]$. What do you conclude, and what would "
      "you do next?", 5,
   "One fold is an outlier; investigate it rather than reporting the mean.",
   ["Four folds agree closely and one does not, which is a property of the "
    "data in that fold, not noise around a common value",
    "The mean of a bimodal set of scores describes none of them"])
ex(2, "Explain, in two sentences, why the test set may be looked at exactly "
      "once, and what you may legitimately do if the number disappoints.", 4,
   "Looking at it more than once makes it a validation set. If it "
   "disappoints, you may report it &mdash; not tune against it.",
   ["Every decision taken after seeing the test score is a decision the score "
    "no longer measures honestly",
    "The permitted response is a new experiment on new held-out data, or an "
    "honest report of what was found"])

# ------------------------------------------------------------------ L03
ex(3, "On the MNIST 5-detector, a classifier that never fires scores "
      "$90.96\\%$ accuracy on the training set. State what that number is a "
      "measurement of.", 3,
   "The share of the data that is not a 5.",
   ["With no positive prediction, accuracy is exactly the negative class&rsquo;s "
    "base rate",
    "It measures the data, not the model &mdash; which is why it is the anchor "
    "every other number is read against"])
ex(3, "Prove or disprove: as the decision threshold is lowered, precision "
      "increases monotonically.", 5,
   "Disprove &mdash; precision is not monotone in the threshold.",
   ["Lowering the threshold admits one instance at a time; recall cannot fall, "
    "but precision rises or falls depending on whether that instance is a "
    "true or a false positive",
    "A single false positive admitted at a high-precision point lowers it, and "
    "a later true positive can raise it again"])
ex(3, "A client asks for &ldquo;90% precision&rdquo;. Explain why that is not "
      "yet a specification, and give the one thing you would add.", 4,
   "It names no recall, and precision alone is met by predicting almost "
   "nothing. Add the recall it must hold at.",
   ["A classifier firing once, correctly, has precision 1.0 and is useless",
    "An operating point is a pair, and the threshold that achieves it is a "
    "decision someone has to own"])
ex(3, "Two models have the same ROC AUC. One is far better on a "
      "precision&ndash;recall curve. What does that tell you about the "
      "problem?", 4,
   "The positive class is rare.",
   ["ROC uses the false-positive <em>rate</em>, whose denominator is the large "
    "negative class, so many false positives barely move it",
    "PR uses precision, whose denominator is what the model predicted, so the "
    "same false positives dominate it"])
ex(3, "The confusion matrix of the never-fires classifier has two zero cells. "
      "Name them, and say which single metric would have exposed the model "
      "immediately.", 4,
   "True positives and false positives are zero; recall exposes it, at 0.0.",
   ["Nothing was predicted positive, so both positive-prediction cells are "
    "empty",
    "Recall divides by the actual positives, so it is 0 however rare they are"])

# ------------------------------------------------------------------ L04
ex(4, "Gradient descent on the same data, with the features unscaled, takes "
      "many more steps to converge than on scaled features. Explain why, in "
      "terms of the shape of the cost surface.", 5,
   "Unscaled features make the contours elongated, so the gradient points "
   "across the valley rather than along it.",
   ["The curvature in each direction is proportional to that feature&rsquo;s "
    "scale, so a large-scale feature dominates the step",
    "The learning rate must then suit the steepest direction, which makes it "
    "far too small for the shallow one"])
ex(4, "Your logistic regression on one-hot encoded columns produces "
      "coefficients that look wrong, and the design matrix has rank six less "
      "than its column count. State the cause and the standard repair.", 5,
   "Each fully encoded categorical adds a redundant column; drop one level per "
   "categorical.",
   ["The indicator columns of one categorical sum to the intercept column, "
    "which is an exact linear dependence",
    "With a reference level dropped the coefficients become differences from "
    "that level, and are interpretable again"])
ex(4, "A model reaches 0.80 accuracy and is <em>badly calibrated</em>. Explain "
      "what that means, and name a decision it would make wrongly that "
      "accuracy cannot see.", 4,
   "Its predicted probabilities do not match observed frequencies; any "
   "decision using a cost-weighted threshold is then wrong.",
   ["Accuracy reads only the arg-max, so it is blind to the probability that "
    "produced it",
    "Choosing a threshold from expected cost requires the probability to mean "
    "what it says"])
ex(4, "Why does the log loss have no closed-form minimiser, when squared error "
      "does?", 4,
   "Its gradient is not linear in the parameters, so setting it to zero gives "
   "no solvable system.",
   ["Squared error is quadratic in $\\theta$, so its gradient is affine and "
    "the stationary condition is a linear system",
    "The sigmoid makes the log-loss gradient transcendental in $\\theta$"])
ex(4, "State the update rule for batch gradient descent, and say what changes "
      "in it for stochastic gradient descent &mdash; and what does not.", 4,
   "$\\theta \\leftarrow \\theta - \\eta\\nabla J$. Only the set the "
   "gradient is computed over changes; the rule does not.",
   ["SGD estimates the same gradient from one instance rather than all of them",
    "The estimate is unbiased and noisy, which is why the step size usually has "
    "to decay"])

# ------------------------------------------------------------------ L05
ex(5, "Write the bias&ndash;variance decomposition of the expected squared "
      "error, and name the one term no model can reduce.", 4,
   "$\\mathbb{E}[(y-\\hat f)^2] = \\text{bias}^2 + \\text{variance} + "
   "\\sigma^2$; the noise $\\sigma^2$ cannot be reduced.",
   ["Two passengers with identical features and different outcomes put a floor "
    "under any predictor",
    "A model reporting error below that floor has been measured wrongly"])
ex(5, "A learning curve shows training and validation error both high and "
      "close together, and flat as data is added. Diagnose it, and say whether "
      "more data would help.", 5,
   "High bias. More data would not help.",
   ["The two curves have already met, so the gap &mdash; the variance &mdash; "
    "is small",
    "The remaining error is the model&rsquo;s inability to represent the "
    "function, which more rows do not change"])
ex(5, "Ridge adds $\\alpha I$ to $X^{\\mathsf T}X$ before inverting. Give "
      "the numerical consequence, in terms of the condition number.", 4,
   "It raises every eigenvalue by $\\alpha$, so the condition number falls "
   "and the inverse is stable.",
   ["Near-collinear columns give eigenvalues near zero, which the inverse "
    "amplifies",
    "The bound on the solution&rsquo;s sensitivity improves as $\\alpha$ grows "
    "&mdash; at the cost of bias"])
ex(5, "Why must features be scaled before ridge or lasso, when they need not be "
      "for ordinary least squares?", 4,
   "The penalty is on the coefficients, and a coefficient&rsquo;s size depends "
   "on its feature&rsquo;s units.",
   ["Least squares is equivariant to rescaling a column: the coefficient "
    "absorbs it",
    "A penalised fit is not, so an unscaled feature is penalised in proportion "
    "to the units someone chose"])
ex(5, "You tune $\\alpha$ over a grid using cross-validation, then report "
      "the best cross-validated score as your estimate of future performance. "
      "Name the error, and quantify its direction.", 5,
   "The best-of-grid score is optimistic: it is a minimum over the same folds "
   "used to choose.",
   ["Selection over $k$ candidates on the same data biases the winner downward "
    "in error",
    "The honest number comes from data that took part in no choice &mdash; a "
    "held-out test set, scored once"])

# ------------------------------------------------------------------ L06
ex(6, "Gini and entropy give nearly the same trees on CoverType. State the "
      "property they share that explains it, and name a case where they can "
      "differ.", 4,
   "Both are strictly concave with a maximum at the uniform distribution and "
   "zero at a pure node; they differ only where two splits are nearly tied.",
   ["Any impurity with those properties ranks most candidate splits identically",
    "Entropy weighs rare classes slightly more, so it can prefer a split that "
    "isolates a small class"])
ex(6, "A decision tree is described as &lsquo;nonparametric&rsquo;. Explain "
      "what that means here, and what it implies about the need for "
      "hyperparameters.", 4,
   "Its structure is not fixed before fitting, so it will grow to fit the data "
   "exactly unless constrained.",
   ["The parameter count is decided by the data, not by the model class",
    "Constraints such as depth or minimum leaf size are what stop it "
    "memorising"])
ex(6, "The brief allows at most eight conditions per decision. You measure the "
      "number of conditions as <code>decision_path().sum() - 1</code>. Explain "
      "the $-1$.", 3,
   "The decision path counts the nodes visited, including the leaf, and the "
   "leaf applies no condition.",
   ["A path of depth $d$ visits $d+1$ nodes",
    "Without the subtraction the count exceeds the tree&rsquo;s own depth, "
    "which is what the assert catches"])
ex(6, "Refitting a tree on two halves of the same data gives visibly different "
      "trees. Does this make the model unreliable? Answer, with a reason.", 5,
   "Not necessarily &mdash; the predictions can be stable while the structure "
   "is not.",
   ["Greedy splitting makes the choice of root sensitive to near-ties among "
    "candidate thresholds",
    "What matters is whether the accuracy moves; if it does not, the "
    "<em>explanation</em> is unstable rather than the model"])
ex(6, "Per-class recall shows one class far below the rest. Give two responses "
      "that do not involve changing the model, and say which you would try "
      "first.", 4,
   "Re-weight the classes, or change what is reported. Report first.",
   ["A single headline accuracy hides the class, so reporting per-class recall "
    "is free and immediate",
    "Class weighting trades other classes&rsquo; recall for it, which is a "
    "decision the client should make"])



# ------------------------------------------------------------------ L07
ex(7, r"For $n$ predictors each of variance $\sigma^2$ and pairwise correlation "
      r"$\rho$, the variance of their average is "
      r"$\rho\sigma^2 + \frac{1-\rho}{n}\sigma^2$. State the limit as "
      r"$n \to \infty$, and what it means for ensembles.", 5,
   r"It tends to $\rho\sigma^2$ &mdash; correlation, not ensemble size, sets the "
   "floor.",
   ["The second term vanishes; the first does not depend on $n$ at all",
    "Adding members past a point buys nothing, which is why the design effort "
    "goes into <em>decorrelating</em> them"])
ex(7, "Extra-trees are usually faster to fit than a random forest and often no "
      "less accurate. Give the mechanism, in terms of the formula above.", 4,
   r"Random split thresholds lower $\rho$, which lowers the floor.",
   ["Choosing the threshold at random rather than optimally makes members "
    "disagree more",
    r"Each member is individually worse &mdash; higher $\sigma^2$ &mdash; and the "
    "product can still improve"])
ex(7, "You set <code>bootstrap=False</code> and <code>oob_score=True</code>. "
      "State what happens and why.", 3,
   "It raises: with no bootstrap there are no out-of-bag rows.",
   ["Out-of-bag scoring uses the rows a member did not sample",
    "Without sampling, every member sees every row and the estimate is undefined"])
ex(7, "Impurity-based feature importance ranks a random decoy column above "
      "several real ones. Explain how that can happen, and name the measure "
      "you would use instead.", 5,
   "A high-cardinality column offers more split points, so it accumulates "
   "impurity decrease by chance. Use permutation importance.",
   ["Impurity importance rewards a column for being <em>splittable</em>, not "
    "for being informative",
    "Permutation importance measures the score lost when the column is "
    "shuffled, which a decoy cannot fake"])
ex(7, "A forest improves accuracy over one tree by far less than the variance "
      "reduction would suggest. Is this a contradiction? Answer with a reason.", 4,
   "No &mdash; variance is only one term of the error.",
   ["Averaging reduces variance and leaves bias untouched",
    "If the remaining error is mostly bias or noise, removing variance moves "
    "the accuracy very little"])

# ------------------------------------------------------------------ L08
ex(8, r"PCA is derived by maximising $\|XW\|_F^2$ subject to $W^{\mathsf T}W=I$. "
      "State why the data must be centred first, and what the first component "
      "becomes if it is not.", 5,
   "Without centring the leading direction points at the mean, not at the "
   "direction of greatest variance.",
   ["The objective measures distance from the origin, and only after centring "
    "is that the same as spread",
    "On the Olivetti faces the uncentred first component is the mean face"])
ex(8, "The Johnson&ndash;Lindenstrauss bound, evaluated for 400 images at "
      r"$\varepsilon = 0.1$, asks for more dimensions than the data has. What "
      "do you conclude about the bound, and about the method?", 5,
   "The bound is vacuous here; the method still works.",
   ["JL is a worst-case guarantee over all point sets, and this one is not "
    "worst case",
    "A bound that does not bind is not evidence against the technique &mdash; "
    "measure the distortion instead"])
ex(8, "A pipeline fits PCA on the whole dataset and then splits. Name what "
      "leaked, and say whether the leak grows or shrinks as the corpus grows.", 4,
   "The components leaked; the damage shrinks as the corpus grows.",
   ["The principal directions are fitted parameters, so fitting them on test "
    "rows lets those rows shape their own representation",
    "With more rows each individual test row moves the components less"])
ex(8, "Inertia always falls as $k$ rises, so it cannot choose $k$. State what "
      "silhouette adds, and one situation where it also fails.", 5,
   "Silhouette compares within-cluster to nearest-other-cluster distance, so it "
   "can fall. It fails when the clusters are not compact and separated.",
   ["Inertia is monotone by construction: more centres can only shorten "
    "distances",
    "Silhouette assumes the geometry k-means assumes, so elongated or nested "
    "clusters defeat both"])
ex(8, "You have 4,096-dimensional face vectors and are told two faces are "
      "&ldquo;close&rdquo;. Explain why that alone does not mean they are the "
      "same person.", 4,
   "Distance is not identity &mdash; lighting and pose move a face further "
   "than identity does.",
   ["Pixel distance is dominated by whatever varies most, and that is "
    "illumination",
    "The metric has to be one under which the thing you care about is the "
    "thing that varies"])

# ------------------------------------------------------------------ L09
ex(9, "A multilayer perceptron with the activation removed is written as "
      "$W_3W_2W_1x$. State what function class it can represent, and why depth "
      "then buys nothing.", 4,
   "Only linear maps &mdash; the product of matrices is a matrix.",
   ["Composition of linear maps is linear, whatever the depth",
    "The nonlinearity is what makes an extra layer a new function rather than a "
    "re-parameterisation"])
ex(9, "Give the parameter count of a dense layer from $m$ inputs to $n$ units, "
      "and evaluate it for the first layer of a network on 28&times;28 images "
      "with 300 units.", 4,
   r"$mn + n$; here $784 \times 300 + 300 = 235{,}500$.",
   ["One weight per input-output pair, plus one bias per unit",
    "The first layer of an image network usually dominates the count, because "
    "$m$ is the pixel count"])
ex(9, "A single TLU cannot represent XOR. State the property of XOR that "
      "prevents it, and the smallest change to the network that fixes it.", 4,
   "XOR is not linearly separable; one hidden layer fixes it.",
   ["A TLU's decision boundary is a hyperplane, and no line separates the two "
    "XOR classes",
    "A hidden layer can re-represent the inputs so that the classes become "
    "separable in the new coordinates"])
ex(9, "Fashion-MNIST is exactly balanced across ten classes. State the trivial "
      "baseline&rsquo;s accuracy, and why balance is what makes accuracy "
      "defensible here.", 3,
   "10%. With equal class sizes and equal costs, accuracy is not dominated by "
   "one class.",
   ["The majority-class predictor scores the largest class share, which is "
    "one tenth",
    "Lecture 3's objection to accuracy was imbalance, and it does not apply"])
ex(9, "An architecture sweep finds the best hidden-layer size on this dataset. "
      "State what that result does and does not transfer to.", 4,
   "It transfers to this dataset and task; it does not transfer to another "
   "dataset.",
   ["The best capacity depends on the amount and difficulty of the data",
    "Reporting it as a general recommendation is the error &mdash; it is a "
    "measurement, not a rule"])

# ------------------------------------------------------------------ L10
ex(10, "Reverse-mode automatic differentiation costs one sweep for all "
       "partial derivatives; forward mode costs one per input. State the "
       "condition on the function&rsquo;s shape that makes reverse mode the "
       "right choice.", 4,
   "Many inputs, one output &mdash; which is exactly a loss.",
   ["Forward mode's cost scales with the number of inputs, reverse mode's with "
    "the number of outputs",
    "A neural network has millions of parameters and one scalar loss"])
ex(10, "Explain why the loss must be a scalar for <code>.backward()</code> to "
       "be called without arguments.", 3,
   r"The reverse sweep is seeded with $\partial L/\partial L = 1$, which only "
   "exists for a scalar.",
   ["For a vector output there is no single derivative to seed with",
    "PyTorch then requires an explicit vector to weight the outputs by"])
ex(10, "The five lines of the training loop are reordered so that "
       "<code>opt.step()</code> comes before <code>loss.backward()</code>. "
       "State what the student observes.", 4,
   "Nothing raises, and the model stays at chance.",
   ["The step is taken with the gradients zeroed at the top of the iteration",
    "The gradients computed afterwards are cleared before they are ever used"])
ex(10, "A network with dropout is evaluated without calling "
       "<code>model.eval()</code>. Describe the symptom precisely, and the "
       "one-line diagnostic that catches it.", 5,
   "The score wobbles between identical calls. Evaluate twice and assert the "
   "two agree.",
   ["Dropout keeps sampling masks in training mode, so the metric is a random "
    "variable",
    "A deterministic function of fixed weights and fixed data does not change "
    "between calls"])
ex(10, "GPUs did not help at small model sizes in the measured comparison. "
       "Give the reason, and the quantity that decides where the crossover "
       "sits.", 4,
   "Transfer and launch overhead dominates; the crossover is set by arithmetic "
   "per byte moved.",
   ["A small matrix product finishes faster than the cost of getting it there",
    "The device pays once the work per transferred byte is large enough"])

# ------------------------------------------------------------------ L11
ex(11, "One layer multiplies the standard deviation of the backward signal by "
       r"$\rho = \sqrt{n_{\text{out}}\operatorname{Var}(w)\,"
       r"\mathbb{E}[\varphi'^2]}$. State what $L$ layers do to it, and why only "
       r"$\rho = 1$ survives depth.", 5,
   r"They multiply by $\rho^L$; anything else vanishes or explodes "
   "geometrically.",
   ["A constant applied $L$ times is exponential in $L$",
    "There is no regime where a repeated constant other than 1 is safe"])
ex(11, "Preserving the forward signal asks for "
       r"$\operatorname{Var}(w) = 1/n_{\text{in}}$ and preserving the gradient "
       r"asks for $1/n_{\text{out}}$. State when the two agree, and what Glorot "
       "does when they do not.", 4,
   r"They agree only when $n_{\text{in}} = n_{\text{out}}$; Glorot takes the "
   r"harmonic mean, $2/(n_{\text{in}}+n_{\text{out}})$.",
   ["Any layer that changes width cannot satisfy both",
    "The compromise accepts a small error in each direction rather than a large "
    "one in either"])
ex(11, "He initialisation doubles the weight variance for ReLU. Derive the "
       "factor of two in one line.", 4,
   "ReLU zeroes half of a symmetric zero-mean input, so "
   r"$\mathbb{E}[\text{ReLU}(z)^2] = \tfrac12\mathbb{E}[z^2]$.",
   ["Exactly half the variance is discarded at each layer",
    r"Doubling $\operatorname{Var}(w)$ puts it back"])
ex(11, "A gradient profile that is a straight line on a log axis tells you "
       "something a curved one does not. State what, and why it identifies the "
       "cause.", 4,
   "The attenuation is the same factor at every layer, so it is the "
   "initialisation rather than one bad layer.",
   ["A constant ratio per layer is a straight line in log space",
    "A single broken layer would show a step, not a slope"])
ex(11, "Gradient clipping is applied to a network whose gradients are fifteen "
       "orders of magnitude too small. Predict the effect, and say what the "
       "attempt reveals about the diagnosis.", 4,
   "No effect &mdash; clipping bounds gradients that are too large.",
   ["The threshold is never reached, so nothing is rescaled",
    "Reaching for it means the failure mode was never measured"])

# ------------------------------------------------------------------ L12
ex(12, r"A dense layer mapping a $3\times128\times128$ input to a "
       r"$32\times128\times128$ output has $n_{\text{in}}n_{\text{out}}$ "
       r"weights. Give the convolutional count for a $7\times7$ kernel, and say "
       "which symbols are absent from it.", 5,
   r"$32\times3\times7\times7 = 4{,}704$. Neither $H$ nor $W$ appears.",
   ["One kernel per output channel, reused at every position",
    "The image size is genuinely absent &mdash; that is what weight sharing "
    "means"])
ex(12, "State the difference between equivariance and invariance, and say "
       "which of the two segmentation cannot afford.", 4,
   "Equivariance: $f(T x) = T f(x)$. Invariance: $f(T x) = f(x)$. Segmentation "
   "cannot afford invariance.",
   ["Invariance discards the position, and the answer to a segmentation "
    "question <em>is</em> the position",
    "Classification wants invariance for the same reason"])
ex(12, "Convolution is equivariant to translation on the interior of an image "
       "but not at the border. Give the reason.", 3,
   "Zero padding invents input that was not there, so a shift moves real "
   "content into invented content.",
   ["The identity assumes the operator sees the same neighbourhood before and "
    "after the shift",
    "At the edge it does not, so the equality is an interior statement"])
ex(12, "A training run fails with out-of-memory. State which of parameters and "
       "activations usually dominates, and give the first thing to change.", 4,
   "Activations dominate; halve the batch.",
   ["Activation memory is linear in the batch size and parameter memory does "
    "not depend on it at all",
    "Reducing the parameter count is near the bottom of the list"])
ex(12, "Parameters and activations are anti-correlated across a convolutional "
       "stack. State where each is concentrated, and what that means for any "
       "intuition carried over from dense networks.", 4,
   "Activations are largest near the input, parameters near the output. Dense "
   "intuition points the wrong way.",
   ["Early layers have large maps and tiny kernels; late layers the reverse",
    "In a dense network both grow together, so the habit transfers badly"])

# ------------------------------------------------------------------ L13
ex(13, "State the order in which a pretrained network&rsquo;s head must be "
       "replaced and its body frozen, and what goes wrong in the other order.", 4,
   "Freeze first, then replace. In the other order the new head is frozen too.",
   ["A newly constructed module has <code>requires_grad=True</code>, so the "
    "freezing loop would turn it off",
    "The symptom is a loss that never moves, with nothing raising"])
ex(13, "Fine-tuning uses $10^{-4}$ on the pretrained block and $10^{-3}$ on the "
       "new head. Justify the difference in one sentence.", 4,
   "The block already encodes something and the head starts from noise; one "
   "rate cannot suit both.",
   ["A large step destroys what the block knows",
    "A small step leaves the random head untrained"])
ex(13, "Freezing a batch-norm layer&rsquo;s weights does not freeze the layer. "
       "State what still changes, and the call that stops it.", 4,
   "The running statistics are buffers, updated in the forward pass; "
   "<code>.eval()</code> stops them.",
   ["<code>requires_grad=False</code> governs gradients, not buffers",
    "A frozen feature extractor left in train mode still drifts"])
ex(13, "An augmented validation loader is used to select the epoch to keep. "
       "State the two things this costs, and the one-line test that detects "
       "it.", 5,
   "The score becomes a random variable and the chosen epoch changes. Evaluate "
   "twice and require identical answers.",
   ["Augmentation makes the metric depend on the evaluator's seed",
    "A deterministic function of fixed weights and fixed data does not wobble"])
ex(13, "A frozen-backbone probe is both more accurate and faster than training "
       "from scratch on 1,020 images. Explain why both, and name the cost that "
       "must still be counted.", 5,
   "The backbone is a fitted function reused for free; the feature extraction "
   "pass must still be counted in the total.",
   ["Only the head is fitted, so the variance problem moves to a dataset large "
    "enough to absorb it",
    "The probe is not free just because the head is"])

# ------------------------------------------------------------------ L14
ex(14, "Two boxes are disjoint and moved further apart. State what IoU and its "
       "derivative do, and why no learning rate repairs it.", 5,
   "Both are identically zero; a constant has no descent direction.",
   ["Past contact the intersection is exactly zero for every separation",
    "The gradient is absent rather than small, so no optimiser can act on it"])
ex(14, "An IoU implementation without a clamp is tested on overlapping boxes "
       "only and passes. Give the input that breaks it, and the property test "
       "that would have caught it.", 5,
   "Boxes separated on both axes. Assert the result lies in $[0,1]$ and is zero "
   "exactly when the boxes are disjoint.",
   ["Two negative differences multiply to a positive 'intersection'",
    "A range check alone is necessary and not sufficient &mdash; the diagonal "
    "case can land inside $[0,1]$"])
ex(14, "Average precision is defined with a maximum inside it. State what that "
       "maximum repairs, and which earlier lecture proved the problem.", 4,
   "It replaces the non-monotone precision by its running maximum &mdash; the "
   "non-monotonicity proved in Lecture 3.",
   ["Precision falls at every false positive and can rise again",
    "Integrating the raw curve would reward the sawtooth rather than the "
    "ranking"])
ex(14, "mAP is a mean over classes of a mean over IoU thresholds. Give one "
       "thing each averaging hides.", 4,
   "The class mean hides that a one-instance class weighs as much as a "
   "350-instance one; the threshold mean hides the spread between loose and "
   "tight matching.",
   ["Unweighted means ignore support",
    "A single number in the middle stands for two very different regimes"])
ex(14, "A team computes AP per image and averages those. State the direction of "
       "the error and its cause.", 4,
   "It is optimistic: single images contain few objects and often score 1.0.",
   ["Averaging many easy sub-problems is not the hard problem",
    "It is the metric averaged per batch rather than over the set"])

# ------------------------------------------------------------------ L15
ex(15, "For a stationary series, "
       r"$\operatorname{Var}(X_t - X_{t-h}) = 2\gamma(0)(1-\rho(h))$. State the "
       r"condition on $\rho(h)$ under which differencing <em>increases</em> the "
       "variance.", 4,
   r"$\rho(h) &lt; 1/2$.",
   [r"The factor $2(1-\rho)$ exceeds 1 exactly then",
    "So the reflex 'it is not stationary, difference it' can make the problem "
    "harder"])
ex(15, "Explain why the seasonal-naive forecast is hard to beat on daily "
       "ridership, in terms of the autocorrelation function.", 4,
   r"$\rho(7)$ is high, so the lag-7 difference is close to white noise &mdash; "
   "and copying last week is the model that assumes it is.",
   ["The weekly cycle carries most of the structure",
    "What remains after removing it is close to unpredictable"])
ex(15, "State why MAPE is the wrong metric for transit ridership, using a "
       "specific day as the example.", 4,
   "It divides by the truth, so an error on Christmas &mdash; a tenth of "
   "normal ridership &mdash; counts ten times an equal error on an ordinary weekday.",
   ["The metric would spend capacity on the days nobody staffs for",
    "MAE in boardings is in the units the operations team already uses"])
ex(15, "A cross-validated forecast uses <code>KFold(shuffle=True)</code> and "
       "reports a good score. Name the two conditions the split violates.", 5,
   "No training row may come after a test row, and none may be adjacent to one.",
   ["Shuffling puts a point's own future in the training set",
    "Neighbouring days are nearly the same number, so an adjacent row nearly "
    "gives away the answer"])
ex(15, "In March 2020 the series level falls by three quarters and the model "
       "stops working. State whether this is a leak, a bug, or neither, and "
       "what it implies for deployment.", 4,
   "Neither &mdash; the protocol was correct and the world changed. It implies "
   "monitoring, with a written trigger.",
   ["No split protects against a regime change",
    "A model never re-measured after deployment is an assumption wearing a "
    "number's clothes"])


def slide(title, body, menu=None, cls=""):
    menu = menu or title
    c = f' class="{cls}"' if cls else ""
    return (f'<section{c} data-menu-title="{menu}">\n{body}\n</section>\n')


def question_slides(n, items):
    """Two slides: three questions then two, so neither runs over the canvas."""
    out = []
    out.append(slide(
        "",
        '  <p class="kicker">Not presented &mdash; for afterwards</p>\n'
        f'  <h1>Exercises<br>Lecture {n}</h1>\n'
        '  <p class="clock">five, in the style of the written paper</p>',
        menu=f"◆ Exercises · Lecture {n}", cls="divider"))
    for part, (lo, hi) in enumerate(((0, 3), (3, 5))):
        lis = "\n".join(
            f'    <li>{it["q"]} <span class="ex-marks">[{it["marks"]}]</span></li>'
            for it in items[lo:hi])
        cnt = f' style="counter-reset: ex {lo}"' if lo else ""
        out.append(slide(
            "", f'  <h2>Exercises &mdash; Lecture {n}'
                f'{" (continued)" if lo else ""}</h2>\n'
                f'  <ul class="ex-list"{cnt}>\n{lis}\n  </ul>\n'
                f'  <p class="ex-note">Solutions on Lecture {n + 1}&rsquo;s deck.</p>'
            if n < 24 else
                f'  <h2>Exercises &mdash; Lecture {n}'
                f'{" (continued)" if lo else ""}</h2>\n'
                f'  <ul class="ex-list"{cnt}>\n{lis}\n  </ul>\n'
                f'  <p class="ex-note">Solutions overleaf &mdash; there is no '
                f'Lecture 25.</p>',
            menu=f"Exercises · {n}{' (2)' if lo else ''}"))
    return out


def solution_slides(n, items):
    """Two answers a slide, side by side.

    Three to a slide ran 706px on Lecture 2 against a footer at 674 -- the
    check_overflow warning that made this layout two-up. It is also the layout
    the deck's own specimen answers already use, so the pages match.
    """
    out = []
    out.append(slide(
        "",
        '  <p class="kicker">Set last time</p>\n'
        f'  <h1>Solutions<br>Lecture {n}</h1>\n'
        '  <p class="clock">not presented</p>',
        menu=f"◇ Solutions · Lecture {n}", cls="divider"))
    for part, (lo, hi) in enumerate(((0, 2), (2, 4), (4, 5))):
        chunk = items[lo:hi]
        if not chunk:
            continue
        blocks = []
        for k, it in enumerate(chunk, start=lo + 1):
            why = "\n".join(f'        <li>{w}</li>' for w in it["why"])
            blocks.append(
                f'    <div>\n'
                f'      <p class="smaller"><strong>{k}.</strong> '
                f'{short(it["q"])}</p>\n'
                f'      <p class="lead"><span class="fix">{it["a"]}</span></p>\n'
                f'      <ul class="tight smaller">\n{why}\n      </ul>\n'
                f'    </div>')
        out.append(slide(
            "", f'  <h2>Solutions &mdash; Lecture {n}'
                f'{f" ({part + 1})" if part else ""}</h2>\n'
                f'  <div class="cols">\n' + "\n".join(blocks) + '\n  </div>',
            menu=f"Solutions · {n}{f' ({part + 1})' if part else ''}"))
    return out


def short(q, limit=110):
    """The question, trimmed to a reminder rather than repeated in full.

    Truncation must not cut through inline mathematics. An odd number of `$`
    leaves KaTeX rendering the rest of the slide as literal text, which is
    exactly what check_decks caught on Lectures 12 and 16 the first time this
    ran -- so back up to before the unclosed delimiter rather than shipping it.
    """
    plain = " ".join(re.sub(r"<[^>]+>", "", q).split())
    if len(plain) <= limit:
        return plain
    cut = plain[:limit].rsplit(" ", 1)[0]
    if cut.count("$") % 2:
        cut = cut[:cut.rfind("$")].rstrip()
    return cut + "&hellip;"


def build(n):
    """Every exercise/solution slide lecture n's deck should carry."""
    out = []
    if n - 1 in EXERCISES:
        out += solution_slides(n - 1, EXERCISES[n - 1])
    if n in EXERCISES:
        out += question_slides(n, EXERCISES[n])
        if n == 24:
            out += solution_slides(24, EXERCISES[24])
    return out


def inject(n) -> str | None:
    path = ROOT / f"slides/lecture-{n:02d}.html"
    src = path.read_text(encoding="utf-8")
    body = "\n".join(build(n))
    if not body:
        return None
    block = f"{BEGIN}\n{body}{END}\n"

    if BEGIN in src:
        # lambda, not the string: the block is full of LaTeX, and re.sub reads
        # a backslash in a replacement as a group escape -- "\\hat" is a
        # PatternError, and "\\1" would silently splice in a capture group.
        src = re.sub(re.escape(BEGIN) + r".*?" + re.escape(END) + r"\n",
                     lambda _m: block, src, flags=re.S)
    else:
        # before the closing "Next" divider, which is always the last slide
        i = src.rfind('<section class="divider"')
        if i < 0:
            i = src.rfind("</section>")
            i = src.rfind("<section", 0, i)
        src = src[:i] + block + src[i:]
    path.write_text(src, encoding="utf-8")
    return path.name


def main() -> int:
    want = [int(a) for a in sys.argv[1:]] or list(range(1, 25))
    done = [r for n in want if (r := inject(n))]
    have = sorted(EXERCISES)
    print(f"{len(done)} deck(s) rewritten; exercises defined for "
          f"{len(have)} lecture(s): {have}")
    missing = [n for n in range(1, 25) if n not in EXERCISES]
    if missing:
        print(f"still to write: {missing}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
