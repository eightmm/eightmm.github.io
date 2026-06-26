---
title: Mean Squared Error
tags:
  - machine-learning
  - loss
  - regression
---

# Mean Squared Error

Mean squared error penalizes the squared distance between predictions and targets. It is a standard regression loss and a common reconstruction loss.

For predictions $\hat{y}_i=f_\theta(x_i)$ and targets $y_i$:

$$
\operatorname{MSE}
=
\frac{1}{n}
\sum_{i=1}^{n}
\lVert \hat{y}_i-y_i\rVert_2^2
$$

For scalar targets:

$$
\operatorname{MSE}
=
\frac{1}{n}
\sum_{i=1}^{n}
(\hat{y}_i-y_i)^2
$$

The gradient with respect to a scalar prediction is:

$$
\frac{\partial}{\partial \hat{y}_i}
(\hat{y}_i-y_i)^2
=
2(\hat{y}_i-y_i)
$$

Large residuals therefore produce larger gradients than small residuals.

## Likelihood View

If observations follow a Gaussian distribution with fixed variance:

$$
y\mid x \sim \mathcal{N}(f_\theta(x), \sigma^2)
$$

then minimizing MSE is equivalent to minimizing Gaussian [[concepts/machine-learning/negative-log-likelihood|negative log-likelihood]] up to constants and scale.

More explicitly, for $y\in\mathbb{R}^d$:

$$
y\mid x
\sim
\mathcal{N}(\mu_\theta(x), \sigma^2 I)
$$

gives:

$$
-
\log p_\theta(y\mid x)
=
\frac{1}{2\sigma^2}
\lVert y-\mu_\theta(x)\rVert_2^2
+
\frac{d}{2}\log(2\pi\sigma^2)
$$

The fixed isotropic variance assumption is part of the claim. If each target dimension has different noise, a weighted MSE or learned-variance Gaussian NLL may be more appropriate.

## Scale and Weighting

MSE is sensitive to target scale:

$$
\operatorname{MSE}(a\hat{y}, ay)
=
a^2\operatorname{MSE}(\hat{y},y)
$$

This matters for affinity units, log-transformed labels, standardized targets, coordinate losses, and multi-task objectives. If targets are standardized:

$$
\tilde{y}
=
\frac{y-\mu_{\mathrm{train}}}{\sigma_{\mathrm{train}}}
$$

then $\mu_{\mathrm{train}}$ and $\sigma_{\mathrm{train}}$ must be computed only on the training split.

For weighted regression:

$$
\operatorname{WMSE}
=
\frac{\sum_i w_i \lVert \hat{y}_i-y_i\rVert_2^2}
{\sum_i w_i}
$$

Weights can represent confidence, assay quality, task balance, or sampling correction, but they change the population being optimized.

## Coordinate and Structure Losses

For coordinate targets:

$$
\mathcal{L}_{X}
=
\frac{1}{N}
\sum_{a=1}^{N}
\lVert \hat{x}_a - x_{\pi(a)}\rVert_2^2
$$

where $\pi$ maps predicted atoms or residues to reference atoms or residues. This mapping is not optional for symmetric atoms, alternate conformers, missing residues, or ligand poses.

Coordinate MSE is not automatically rotation- or translation-invariant. A note should state whether coordinates are:

- aligned before scoring;
- represented through distances;
- predicted as equivariant coordinate updates;
- evaluated with RMSD, distance error, contact accuracy, or physical validity.

## Metric Boundary

Lower MSE supports a squared-error claim. It does not automatically imply better rank ordering, threshold classification, calibration, enrichment, or structural validity. For molecular and protein tasks, pair it with the metric that matches the downstream use.

## Checks

- Is the target scale raw, log-transformed, standardized, clipped, or censored?
- Are large residuals meaningful rare cases or measurement artifacts?
- Does the downstream task care about squared error, absolute error, rank order, or threshold crossing?
- Are target normalization statistics computed only from the training split?
- Is MSE used as a training loss, an evaluation metric, or both?

## Related

- [[concepts/machine-learning/loss-function|Loss function]]
- [[concepts/machine-learning/regression|Regression]]
- [[concepts/machine-learning/negative-log-likelihood|Negative log-likelihood]]
- [[concepts/evaluation/regression-metrics|Regression metrics]]
- [[concepts/evaluation/uncertainty-estimation|Uncertainty estimation]]
- [[concepts/math/statistical-estimator|Statistical estimator]]
