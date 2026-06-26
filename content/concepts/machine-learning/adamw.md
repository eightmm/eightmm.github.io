---
title: AdamW
tags:
  - machine-learning
  - optimization
  - regularization
---

# AdamW

AdamW is Adam with decoupled weight decay. The key distinction is that weight decay is applied directly to parameters, rather than being mixed into the gradient that Adam rescales.

Adam's adaptive step is:

$$
u_t =
\frac{\hat{m}_t}{\sqrt{\hat{v}_t}+\epsilon}
$$

AdamW applies:

$$
\theta_{t+1}
=
\theta_t
-
\eta u_t
-
\eta \lambda \theta_t
$$

where $\lambda$ is the weight decay coefficient. Some implementations write the last two terms as:

$$
\theta_{t+1}
=
(1-\eta\lambda)\theta_t - \eta u_t
$$

## Why Decoupling Matters

In classical L2 regularization, the penalty contributes $\lambda\theta$ to the gradient. Adam rescales gradients per parameter, so coupling L2 penalty to the adaptive gradient can change the intended regularization. AdamW keeps the decay term separate.

## Parameter Groups

Deep learning code often applies weight decay only to selected parameters:

- Usually decayed: large weight matrices in linear, convolution, and attention projections.
- Often not decayed: biases, normalization scale/shift parameters, and sometimes embeddings.

This is an implementation contract, not a universal law. The paper, code, or run record should state the grouping.

## Checks

- Is the optimizer actually AdamW, not Adam with L2 penalty?
- Which parameter groups receive weight decay?
- Is the weight decay coefficient reported separately from the learning rate?
- Are normalization and bias parameters excluded intentionally?
- Are optimizer state and scheduler state saved for resume?

## Related

- [[concepts/machine-learning/adam|Adam]]
- [[concepts/machine-learning/optimizer|Optimizer]]
- [[concepts/machine-learning/weight-decay|Weight decay]]
- [[concepts/machine-learning/regularization|Regularization]]
- [[concepts/machine-learning/learning-rate-schedule|Learning rate schedule]]
- [[infra/reproducibility/run-record|Reproducible run record]]
