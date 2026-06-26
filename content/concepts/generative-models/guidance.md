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

| Form | Signal | Main Risk |
| --- | --- | --- |
| classifier guidance | gradient or score from classifier/property predictor | guide exploits classifier shortcuts |
| classifier-free guidance | conditional and unconditional model predictions | high scale reduces diversity or creates artifacts |
| energy or score guidance | external energy, reward, docking score, or property score | proxy hacking and distribution shift |
| constraint guidance | syntax, graph, geometry, validity, or hard rule | narrows sample space and changes novelty claim |
| human or tool guidance | verifier, reranker, search, or external evaluator | hidden best-of-n and selection bias |

Classifier-free guidance is often written as:

$$
\hat{\epsilon}_{\mathrm{cfg}}
=
(1+w)\hat{\epsilon}_\theta(x_t,t,c)
-
w\hat{\epsilon}_\theta(x_t,t,\varnothing)
$$

where $w$ is the guidance scale. Equivalent forms are used for denoised sample, score, or velocity prediction depending on the model.

## Tradeoff

Guidance can improve condition satisfaction but reduce diversity or introduce predictor shortcuts. A generated sample can satisfy the guide while failing the true task.

## Claim Boundary

| Claim | Required Evidence |
| --- | --- |
| better controllability | condition satisfaction vs diversity curve over guidance scale |
| better utility | downstream utility measured after counting all rejected candidates |
| better sampler | same base model and matched NFE/candidate/filtering budget |
| better molecular generation | validity, novelty, diversity, property/interaction utility, and proxy bias checks |
| better structure generation | geometry validity, clash/constraint checks, coordinate-frame and symmetry handling |

## Leakage and Proxy Risks

Guidance is only as trustworthy as the guide. For a guide $r_\psi(x,c)$:

$$
x^\star
=
\arg\max_{x\in \mathcal{C}}
r_\psi(x,c)
$$

can select artifacts if $r_\psi$ was trained on leaked labels, near-duplicates, template-derived structures, or a benchmark-specific proxy. A paper should state the guide training data, split boundary, and whether the final metric is independent of the guide.

## Checks

- What signal guides sampling?
- Is the guide trained on data that leaks test labels or targets?
- Does increasing guidance improve validity but reduce diversity?
- Is the guide aligned with the final evaluation metric?
- Are rejected or filtered samples included in efficiency and validity reports?
- Is the same guide, scale, and filtering rule used for all baselines?
- Is the guide differentiable, black-box, learned, physics-based, or rule-based?
- Does the paper report an unguided baseline and a guidance-scale sweep?

## Related

- [[concepts/generative-models/conditional-generation|Conditional generation]]
- [[concepts/generative-models/sampling|Sampling]]
- [[concepts/generative-models/diffusion-model|Diffusion model]]
- [[concepts/generative-models/flow-matching|Flow matching]]
- [[concepts/evaluation/generation-evaluation|Generation evaluation]]
- [[concepts/evaluation/metric-selection|Metric selection]]
- [[concepts/evaluation/robustness|Robustness]]
- [[concepts/sbdd/scoring-function|Scoring function]]
