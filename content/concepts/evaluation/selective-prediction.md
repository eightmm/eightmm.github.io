---
title: Selective Prediction
tags:
  - evaluation
  - uncertainty
  - decision
---

# Selective Prediction

Selective prediction evaluates a model that can abstain instead of always returning an answer. It connects [[concepts/evaluation/uncertainty-estimation|Uncertainty estimation]], [[concepts/evaluation/calibration|Calibration]], [[concepts/evaluation/threshold-selection|Threshold selection]], and human review workflows.

A selective predictor has two parts:

$$
f(x) = \text{prediction}
$$

$$
g(x) \in \{0,1\}
$$

where $g(x)=1$ means accept the model prediction and $g(x)=0$ means abstain, defer, or request another process.

For a confidence or uncertainty score $s(x)$ and threshold $\tau$:

$$
g_\tau(x)
=
\mathbf{1}[s(x) \ge \tau]
$$

if higher $s(x)$ means more confidence. If higher score means more uncertainty, the inequality is reversed.

## Coverage and Selective Risk

Coverage is the fraction of examples the model accepts:

$$
\operatorname{coverage}(\tau)
=
\frac{1}{n}
\sum_{i=1}^{n}
g_\tau(x_i)
$$

Selective risk is the average loss only on accepted examples:

$$
\operatorname{risk}(\tau)
=
\frac{
\sum_{i=1}^{n}
g_\tau(x_i)
\mathcal{L}(f(x_i),y_i)
}{
\sum_{i=1}^{n}
g_\tau(x_i)
}
$$

The core trade-off is:

$$
\text{lower risk}
\quad \leftrightarrow \quad
\text{lower coverage}
$$

A useful uncertainty score should reduce risk as coverage decreases.

## Decision View

Abstention is not free. If abstained examples go to human review, slower inference, wet-lab validation, or another model, evaluation should include that cost:

$$
U(\tau)
=
\operatorname{Benefit}(\mathrm{accepted\ correct})
-
\operatorname{Cost}(\mathrm{accepted\ wrong})
-
\operatorname{Cost}(\mathrm{abstained})
$$

This is why selective prediction should be evaluated with the deployment decision, not only with accuracy.

## Common Uses

- Classification systems that route low-confidence cases to a human.
- Retrieval or QA systems that answer only when evidence is strong.
- Molecular or protein screens that escalate uncertain predictions to expensive validation.
- Agent systems that ask for confirmation before high-impact actions.
- OOD-sensitive systems that reject examples outside the [[concepts/evaluation/applicability-domain|applicability domain]].
- Prediction-set systems where [[concepts/evaluation/conformal-prediction|Conformal prediction]] returns multiple candidates instead of one.

## Checks

- Is the abstention score calibrated or at least monotonic with error risk?
- Is the threshold chosen on validation data, not the test set?
- Are coverage, selective risk, and full-set metric reported together?
- What happens to abstained examples in the actual workflow?
- Is abstention evaluated under distribution shift and hard slices?
- Does the system abstain on the examples where mistakes are expensive?

## Related

- [[concepts/evaluation/uncertainty-estimation|Uncertainty estimation]]
- [[concepts/evaluation/conformal-prediction|Conformal prediction]]
- [[concepts/evaluation/calibration|Calibration]]
- [[concepts/evaluation/threshold-selection|Threshold selection]]
- [[concepts/evaluation/probability-metrics|Probability metrics]]
- [[concepts/evaluation/error-analysis|Error analysis]]
- [[concepts/evaluation/applicability-domain|Applicability domain]]
- [[concepts/evaluation/ood-generalization|OOD generalization]]
- [[agents/verification/human-in-the-loop|Human in the loop]]
