---
title: Uncertainty Estimation
tags:
  - evaluation
  - uncertainty
  - methodology
---

# Uncertainty Estimation

Uncertainty estimation asks how much trust to place in a prediction. It is different from raw confidence: a model can output high confidence while being wrong or out of distribution.

For a predictive distribution:

$$
p_\theta(y\mid x)
$$

a point prediction hides uncertainty, while the full distribution can expose ambiguity:

$$
\hat{y} = \arg\max_y p_\theta(y\mid x)
$$

## Types

- Aleatoric uncertainty: noise or ambiguity in the data itself.
- Epistemic uncertainty: uncertainty from limited data or model knowledge.
- Distributional uncertainty: uncertainty caused by shift from training data.

These are different failure stories:

| Type | Question | Typical response |
| --- | --- | --- |
| aleatoric | is the label inherently noisy or ambiguous? | predict distribution, interval, or abstain |
| epistemic | does the model lack evidence in this region? | collect data, ensemble, Bayesian approximation |
| distributional | is the input outside the training support? | OOD check, applicability domain, human review |

Do not collapse all three into one confidence score without stating the decision use.

For regression with Gaussian likelihood:

$$
y \sim \mathcal{N}(\mu_\theta(x), \sigma_\theta^2(x))
$$

the model predicts both mean and variance.

For classification, predictive entropy is a simple uncertainty score:

$$
H[p_\theta(y\mid x)]
=
-\sum_{k=1}^{K}
p_\theta(y=k\mid x)
\log p_\theta(y=k\mid x)
$$

High entropy means the predictive distribution is diffuse, but it does not by itself prove the model knows when it is wrong.

## Decomposition

With model uncertainty represented by parameter samples or ensemble members $\theta_m$, predictive uncertainty can be decomposed conceptually:

$$
p(y\mid x,\mathcal{D})
=
\int p(y\mid x,\theta)p(\theta\mid\mathcal{D})\,d\theta
$$

For classification ensembles, predictive entropy measures total uncertainty:

$$
H\left[
\frac{1}{M}\sum_{m=1}^{M}p_{\theta_m}(y\mid x)
\right]
$$

Disagreement across members is often used as an epistemic signal, but it is only meaningful if ensemble members are sufficiently diverse and evaluated on held-out data.

## Decision Boundary

Uncertainty only matters through a decision rule:

$$
d(x)
=
\begin{cases}
\text{accept}, & u(x)\le \tau \\
\text{abstain}, & u(x)>\tau
\end{cases}
$$

The threshold $\tau$ should be chosen on validation or calibration data, not on the final test set.

## Evaluation

Uncertainty should be evaluated against a decision:

- Abstention: are uncertain examples more likely to be wrong?
- Selective prediction: does risk decrease as the system accepts fewer examples?
- Conformal prediction: do prediction sets or intervals achieve the target coverage?
- Active learning: do queried examples improve the model efficiently?
- OOD detection: does uncertainty increase under shift?
- Risk ranking: do high-risk predictions receive larger uncertainty?
- Calibration: do confidence levels match empirical correctness?

## Checks

- Is uncertainty used for ranking, abstention, active learning, or decision-making?
- Are coverage and selective risk reported if the system may abstain?
- Is uncertainty calibrated on held-out data?
- Does uncertainty increase under out-of-distribution inputs?
- Are epistemic and aleatoric uncertainty conflated?
- Is the uncertainty estimate evaluated against downstream risk?
- Is the abstention or escalation threshold chosen without final-test tuning?
- Are uncertainty failures inspected by subgroup, scaffold, family, or source?

## Related

- [[concepts/evaluation/evaluation-protocol|Evaluation protocol]]
- [[concepts/evaluation/calibration|Calibration]]
- [[concepts/evaluation/conformal-prediction|Conformal prediction]]
- [[concepts/evaluation/selective-prediction|Selective prediction]]
- [[concepts/evaluation/ood-generalization|OOD generalization]]
- [[concepts/evaluation/applicability-domain|Applicability domain]]
- [[concepts/evaluation/threshold-selection|Threshold selection]]
- [[concepts/evaluation/reliability-diagram|Reliability diagram]]
- [[concepts/math/bayes-rule|Bayes rule]]
- [[concepts/machine-learning/regression|Regression]]
