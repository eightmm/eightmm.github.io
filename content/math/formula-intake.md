---
title: Formula Intake
tags:
  - math
  - papers
---

# Formula Intake

논문 수식은 notation이 아니라 claim의 계약입니다. 수식을 읽을 때는 모양을 외우기보다 어떤 object, distribution, parameter, objective, metric을 고정하는지 확인합니다.

$$
\text{formula}
=
(\text{objects}, \text{indices}, \text{distribution}, \text{operation}, \text{claim})
$$

## Intake Fields

| Field | Question | Route |
| --- | --- | --- |
| Explanation depth | Should the note use name-only, canonical formula, operational form, claim contract, or derivation depth? | [Formula explanation ladder](/math/formula-explanation-ladder) |
| Pattern | Does the equation match a known loss, likelihood, contrastive, generative, graph, coordinate, or metric pattern? | [Formula pattern catalog](/math/formula-patterns) |
| Object type | Is the quantity a scalar, vector, matrix, tensor, graph, coordinate, distribution, or trajectory? | [Linear algebra](/math/linear-algebra), [Tensor shape notation](/concepts/math/tensor-shape-notation), [Geometry](/math/geometry-symmetry) |
| Index set | Does the index run over batch, token, node, edge, residue, atom, time, sample, or candidate? | [Discrete math and graphs](/math/discrete-graphs) |
| Distribution | Is the expectation over data, model samples, noise, policy rollouts, or test population? | [Probability and statistics](/math/probability-statistics) |
| Objective | Is the formula a loss, likelihood, bound, score, reward, constraint, or metric? | [Information and likelihood](/math/information-likelihood) |
| Bayesian quantity | Is the formula a prior, likelihood, posterior, MAP estimate, or posterior predictive distribution? | [Bayesian inference](/concepts/math/bayesian-inference) |
| Objective-metric link | Does the optimized loss support the reported metric and claimed utility? | [Objective-metric alignment](/concepts/machine-learning/objective-metric-alignment) |
| Derivative | Is the derivative with respect to parameters, inputs, coordinates, time, or latent variables? | [Calculus and gradients](/math/calculus-gradients) |
| Curvature | Does the method use Hessian, Fisher, Newton step, curvature, or local quadratic approximation? | [Second-order optimization](/concepts/machine-learning/second-order-optimization) |
| Symmetry | Should the output be invariant or equivariant under permutation, translation, rotation, or reflection? | [Geometry and symmetry](/math/geometry-symmetry) |
| Numerics | Does implementation require stable softmax, log-sum-exp, precision control, or conditioning checks? | [Numerical computing](/math/numerical-computing) |
| Evidence | Is the reported number a point estimate, confidence interval, paired comparison, or selected checkpoint? | [Evaluation math](/math/evaluation-math) |
| Benchmark score | What aggregation, split, selection rule, and denominator define the reported number? | [Benchmark intake](/concepts/data/benchmark-intake) |

## Rewrite Pattern

For any objective, rewrite the paper equation into this shape when possible:

$$
\mathcal{J}(\theta)
=
\mathbb{E}_{u\sim q(u)}
\left[
w(u)\,\ell_\theta(u)
\right],
\qquad
\hat{\theta}=\arg\min_\theta \mathcal{J}(\theta)
$$

Then define:

- $u$: sampled unit, such as example, token, node, pair, time, noise, or trajectory.
- $q(u)$: sampling distribution used by the objective.
- $w(u)$: weighting, mask, importance weight, or class balance term.
- $\ell_\theta(u)$: per-unit loss, score, residual, or reward term.
- $\theta$: optimized parameter set.

This makes hidden assumptions visible: sampling policy, masking policy, class imbalance, time weighting, and whether the objective matches the evaluation metric.

## Common Formula Families

| Family | Canonical Question |
| --- | --- |
| Linear map | What are the input/output shapes, axes, rank, and projection matrices? |
| Attention | How are $Q$, $K$, $V$, mask, softmax, head concatenation, and output projection defined? |
| Message passing | What node, edge, neighbor, message, aggregation, and update functions are used? |
| Likelihood | What distribution is modeled, and what observation is conditioned on? |
| Bayesian inference | What is prior, likelihood, posterior, evidence, MAP estimate, or predictive distribution? |
| Contrastive loss | What is the anchor, positive, negative set, similarity, and temperature? |
| Diffusion or flow | What is the state, time variable, noise/score/velocity target, and sampling path? |
| Equivariance | What group acts on the input, and how should the output transform? |
| Metric | What is averaged, over which examples, after which model-selection rule? |

For worked canonical patterns, use [[math/formula-patterns|Formula pattern catalog]].

## Explanation Depth

Use [[math/formula-explanation-ladder|Formula explanation ladder]] before deciding how much math to place in the current note.

| Target | Minimum |
| --- | --- |
| Korean post | canonical formula with symbols, or operational form for the main method |
| Concept note | operational form with shapes, indices, distribution, and checks |
| Paper note | claim contract connecting objective, estimator, benchmark, and evidence |
| Math note | derivation or assumptions when the formula itself is the topic |

## Red Flags

- A metric is treated as a loss without explaining the surrogate.
- A loss is assumed to optimize the reported metric without an alignment argument.
- An expectation omits the sampling distribution.
- A graph or set equation does not state permutation behavior.
- A coordinate model does not state translation or rotation handling.
- A benchmark average hides per-target, per-scaffold, or per-source variation.
- A selected checkpoint is reported as if it were an untouched test estimate.

## Related

- [[math/index|Math]]
- [[math/formula-patterns|Formula pattern catalog]]
- [[math/formula-explanation-ladder|Formula explanation ladder]]
- [[ai/paper-intake|AI paper intake]]
- [[molecular-modeling/paper-intake|Molecular modeling paper intake]]
- [[papers/workflows/claim-routing|Claim routing]]
- [[concepts/machine-learning/objective-metric-alignment|Objective-metric alignment]]
- [[concepts/math/bayesian-inference|Bayesian inference]]
- [[concepts/machine-learning/second-order-optimization|Second-order optimization]]
- [[concepts/data/benchmark-intake|Benchmark intake]]
- [[papers/workflows/paper-review-workflow|Paper review workflow]]
