---
title: Guidance
tags:
  - generative-models
  - conditioning
  - sampling
---

# Guidance

Guidance changes the sampling process so generated outputs better satisfy a condition, classifier, score, property, or constraint.

A generic guided sampler modifies an update direction:

$$
d_{\mathrm{guided}}
=
d_\theta(x_t,t,c)
+ \lambda g(x_t,c)
$$

$d_\theta$ is the base model direction, $g$ is a guidance signal, and $\lambda$ controls guidance strength.

## Common Forms

- Classifier guidance: use gradients from a classifier or property predictor.
- Classifier-free guidance: combine conditional and unconditional model predictions.
- Energy or score guidance: bias samples toward low-energy or high-score regions.
- Constraint guidance: enforce validity, syntax, geometry, or domain constraints.
- Human or tool guidance: filter or rerank samples using external evaluation.

## Tradeoff

Guidance can improve condition satisfaction but reduce diversity or introduce predictor shortcuts. A generated sample can satisfy the guide while failing the true task.

## Checks

- What signal guides sampling?
- Is the guide trained on data that leaks test labels or targets?
- Does increasing guidance improve validity but reduce diversity?
- Is the guide aligned with the final evaluation metric?
- Are rejected or filtered samples included in efficiency and validity reports?

## Related

- [[concepts/generative-models/conditional-generation|Conditional generation]]
- [[concepts/generative-models/sampling|Sampling]]
- [[concepts/evaluation/generation-evaluation|Generation evaluation]]
- [[concepts/evaluation/robustness|Robustness]]
- [[concepts/sbdd/scoring-function|Scoring function]]
