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

More explicitly, a common step is:

$$
\theta_t'
=
(1-\eta_t\lambda)\theta_t
$$

$$
\theta_{t+1}
=
\theta_t' - \eta_t
\frac{\hat{m}_t}{\sqrt{\hat{v}_t}+\epsilon}
$$

The decay magnitude depends on both $\eta_t$ and $\lambda$. A learning-rate schedule also changes the effective amount of shrinkage over training.

## Why Decoupling Matters

In classical L2 regularization, the penalty contributes $\lambda\theta$ to the gradient. Adam rescales gradients per parameter, so coupling L2 penalty to the adaptive gradient can change the intended regularization. AdamW keeps the decay term separate.

The distinction can be summarized as:

| Method | Gradient Seen by Adam | Parameter Shrinkage |
| --- | --- | --- |
| Adam + $L_2$ | $g_t+\lambda\theta_t$ | adaptive, coordinate-scaled |
| AdamW | $g_t$ | direct $(1-\eta_t\lambda)\theta_t$ |

## Parameter Groups

Deep learning code often applies weight decay only to selected parameters:

- Usually decayed: large weight matrices in linear, convolution, and attention projections.
- Often not decayed: biases, normalization scale/shift parameters, and sometimes embeddings.

This is an implementation contract, not a universal law. The paper, code, or run record should state the grouping.

| Parameter Type | Common AdamW Choice | Reason |
| --- | --- | --- |
| linear/attention/convolution weights | decay | regularize large learned maps |
| bias | no decay | bias magnitude is not usually a capacity proxy |
| normalization gain/bias | no decay | can harm scale calibration |
| embedding tables | task-dependent | large tables may need separate tuning |
| adapter or LoRA weights | task-dependent | often use separate learning rate and decay |

## Training Recipe Boundary

AdamW should be recorded together with:

$$
(\eta_t,\ \lambda,\ \beta_1,\ \beta_2,\ \epsilon,\ \mathcal{G})
$$

where $\mathcal{G}$ is the parameter-group assignment. Reporting "AdamW" without $\mathcal{G}$ is incomplete because two runs can use the same optimizer name but decay different parameters.

## Failure Modes

| Failure | Symptom |
| --- | --- |
| decay applied to all parameters blindly | normalization or bias parameters become poorly calibrated |
| coefficient copied across batch/schedule changes | validation changes without clear model reason |
| Adam + L2 mistaken for AdamW | reproduction mismatch |
| parameter groups not checkpointed or logged | run cannot be reconstructed |

## Checks

- Is the optimizer actually AdamW, not Adam with L2 penalty?
- Which parameter groups receive weight decay?
- Is the weight decay coefficient reported separately from the learning rate?
- Are normalization and bias parameters excluded intentionally?
- Are optimizer state and scheduler state saved for resume?
- Does the learning-rate schedule change the effective decay strength?
- Are embedding, adapter, or task-specific heads using separate groups?

## Related

- [[concepts/machine-learning/adam|Adam]]
- [[concepts/machine-learning/optimizer|Optimizer]]
- [[concepts/machine-learning/weight-decay|Weight decay]]
- [[concepts/machine-learning/regularization|Regularization]]
- [[concepts/machine-learning/learning-rate-schedule|Learning rate schedule]]
- [[concepts/machine-learning/model-state-contract|Model state contract]]
- [[infra/reproducibility/run-record|Reproducible run record]]
