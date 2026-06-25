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

## Limits

- Single trees can overfit.
- Extrapolation outside the observed feature range is weak.
- Large ensembles can be harder to interpret than a single tree.
- Split decisions can be unstable when correlated features provide similar signals.

## Regularization Knobs

- Maximum depth.
- Minimum samples per leaf.
- Minimum impurity decrease.
- Number of leaves.
- Pruning.

## Related

- [[concepts/machine-learning/ensemble-method|Ensemble method]]
- [[concepts/machine-learning/regularization|Regularization]]
- [[concepts/machine-learning/classification|Classification]]
- [[concepts/machine-learning/regression|Regression]]
- [[concepts/evaluation/index|Evaluation]]
