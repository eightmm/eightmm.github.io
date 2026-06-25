---
title: Interpretability
tags:
  - evaluation
  - interpretability
  - methodology
---

# Interpretability

Interpretability asks whether a model's behavior can be understood in terms useful for debugging, scientific reasoning, or decision support. It is not the same as correctness.

A local explanation often tries to approximate model behavior around one input:

$$
f(x+\delta) \approx f(x) + \nabla_x f(x)^\top \delta
$$

but gradients, attention maps, feature importances, and counterfactuals can all be misleading if they are not validated.

## Common Methods

- Feature importance for tabular or engineered features.
- Saliency or gradient-based attribution.
- Attention inspection.
- Counterfactual examples.
- Example-based explanations through nearest neighbors.
- Mechanistic or structural analysis for scientific models.

## Checks

- Is the explanation faithful to the model or only plausible to humans?
- Does removing the highlighted feature change the prediction?
- Is the explanation stable under small input perturbations?
- Does the explanation reflect data leakage or shortcut features?
- Is interpretability being used for debugging, scientific insight, or user trust?

## Related

- [[concepts/evaluation/error-analysis|Error analysis]]
- [[concepts/evaluation/robustness|Robustness]]
- [[concepts/architectures/attention|Attention]]
- [[concepts/machine-learning/feature-engineering|Feature engineering]]
- [[concepts/evaluation/leakage|Leakage]]
