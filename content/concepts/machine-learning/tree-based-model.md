---
title: Tree-Based Model
tags:
  - machine-learning
---

# Tree-Based Model

A tree-based model splits the feature space into regions and assigns predictions inside each region. Decision trees, random forests, and gradient-boosted trees belong to this family.

At a split, the model chooses a feature and threshold that reduce impurity or loss:

$$
(j^\*, t^\*)
= \arg\max_{j,t}
\Delta I(j,t)
$$

Here $j$ is a feature index, $t$ is a threshold, and $\Delta I$ is the impurity reduction or loss improvement from the split.

For classification, one common impurity measure is Gini impurity:

$$
I_{\mathrm{Gini}}(S)
=
1-\sum_{k=1}^{K}p_k^2
$$

where $p_k$ is the class fraction in node sample set $S$.

For regression, a split often minimizes squared error inside child nodes:

$$
\sum_{c\in\{\mathrm{left},\mathrm{right}\}}
\sum_{i\in S_c}
(y_i-\bar{y}_c)^2
$$

where $\bar{y}_c$ is the mean target in child node $c$.

## Strengths

- Handles nonlinear feature interactions.
- Works well on tabular data.
- Requires less feature scaling than many linear or kernel methods.
- Captures threshold effects and feature interactions without explicit feature crosses.

## Prediction Form

A regression tree partitions feature space into leaves $R_\ell$ and predicts a constant in each leaf:

$$
f(x)
=
\sum_{\ell=1}^{L}
c_\ell \mathbf{1}[x\in R_\ell]
$$

For classification, each leaf stores class probabilities:

$$
\hat{p}(y=k\mid x)
=
\frac{1}{|S_\ell|}
\sum_{i\in S_\ell}\mathbf{1}[y_i=k]
$$

where $S_\ell$ is the training set that lands in the same leaf as $x$.

## Tree Ensembles

Most strong tree-based systems are ensembles.

| Model | Main mechanism | Good for |
| --- | --- | --- |
| decision tree | one recursive partition | interpretability and simple baselines |
| random forest | bagged trees with feature randomness | robust tabular baseline |
| gradient boosting | additive trees fit to residual/gradient signal | high-performance tabular prediction |
| calibrated tree ensemble | post-hoc probability calibration | reliable probabilities |

Gradient boosting builds:

$$
F_m(x)
=
F_{m-1}(x)+\eta h_m(x)
$$

where $h_m$ is the next tree and $\eta$ is the learning rate.

## Tabular Boundary

Tree models are often strong on structured tabular data, but less natural for raw images, sequences, graphs, or coordinates unless features are engineered first.

| Input | Tree model requirement |
| --- | --- |
| tabular descriptors | usually direct |
| molecules | needs fingerprints/descriptors |
| proteins | needs sequence/structure features |
| images | needs extracted features or embeddings |
| text | needs sparse features or embeddings |

## Limits

- Single trees can overfit.
- Extrapolation outside the observed feature range is weak.
- Large ensembles can be harder to interpret than a single tree.
- Split decisions can be unstable when correlated features provide similar signals.
- Probabilities from leaves or ensembles may need calibration.
- Missing values and categorical variables need explicit handling.

## Regularization Knobs

- Maximum depth.
- Minimum samples per leaf.
- Minimum impurity decrease.
- Number of leaves.
- Pruning.
- Learning rate and number of trees for boosting.
- Feature subsampling and row subsampling for ensembles.

## Checks

- Are features available at deployment exactly as during training?
- Is the evaluation split grouped to prevent leakage through near-duplicate rows?
- Are categorical and missing values handled consistently?
- Does the tree ensemble extrapolate poorly outside the observed feature range?
- Are probability outputs calibrated if used for decisions?

## Related

- [[concepts/machine-learning/ensemble-method|Ensemble method]]
- [[concepts/machine-learning/regularization|Regularization]]
- [[concepts/machine-learning/feature-engineering|Feature engineering]]
- [[concepts/machine-learning/classification|Classification]]
- [[concepts/machine-learning/regression|Regression]]
- [[concepts/evaluation/index|Evaluation]]
