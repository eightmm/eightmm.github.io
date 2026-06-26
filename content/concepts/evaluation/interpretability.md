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

## Explanation Contract

| Field | Question |
| --- | --- |
| Object explained | prediction, representation, attention pattern, failure case, or dataset artifact |
| Scope | one example, a slice, one class, one target, or the whole model |
| Method | gradient, perturbation, attention, example-based, concept-based, mechanistic |
| Faithfulness check | whether changing the highlighted evidence changes the model output |
| Stability check | whether the explanation survives small label-preserving perturbations |
| Use | debugging, scientific hypothesis, trust, or communication |

Interpretability evidence should say what it explains and what it does not explain.

## Faithfulness Tests

One simple perturbation test removes or masks the alleged evidence:

$$
\Delta_{\mathrm{attr}}
=
f(x) - f(x_{\setminus S})
$$

where $S$ is the highlighted feature set. A strong explanation should cause a meaningful output change when $S$ is removed and a smaller change when irrelevant features are removed.

For counterfactuals:

$$
x' = x + \delta,
\qquad
f(x') \neq f(x)
$$

The counterfactual is useful only if $\delta$ is plausible under the domain constraints.

## Method Risks

| Method | Useful For | Risk |
| --- | --- | --- |
| Attention map | inspecting token or residue interactions | attention is not automatically causal |
| Gradient attribution | local sensitivity | saturated or noisy gradients |
| Feature importance | tabular/debugging settings | correlated features and leakage |
| Counterfactual | decision boundary probing | invalid or unrealistic changes |
| Nearest neighbors | representation inspection | dataset duplicates and spurious similarity |
| Mechanistic analysis | model internals | overclaiming from small circuits or examples |

## Computational Biology Examples

| Explanation | Check Before Trusting |
| --- | --- |
| important atom or substructure | chemical state, scaffold leakage, activity cliff |
| important residue or pocket contact | receptor state, residue indexing, ligand-defined pocket |
| attention between sequence positions | homolog/template leakage and contact-map agreement |
| docking interaction explanation | pose quality, interaction fingerprint, receptor preparation |
| protein representation cluster | family split, sequence identity, annotation source |

## Checks

- Is the explanation faithful to the model or only plausible to humans?
- Does removing the highlighted feature change the prediction?
- Is the explanation stable under small input perturbations?
- Does the explanation reflect data leakage or shortcut features?
- Is interpretability being used for debugging, scientific insight, or user trust?
- Are explanations evaluated on failures as well as successes?
- Does the explanation respect domain constraints such as chemistry, geometry, or sequence validity?

## Related

- [[concepts/evaluation/error-analysis|Error analysis]]
- [[concepts/evaluation/robustness|Robustness]]
- [[concepts/architectures/attention|Attention]]
- [[concepts/machine-learning/feature-engineering|Feature engineering]]
- [[concepts/evaluation/leakage|Leakage]]
- [[concepts/sbdd/interaction-fingerprint|Interaction fingerprint]]
