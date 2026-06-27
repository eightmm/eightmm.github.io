---
title: Formula Explanation Ladder
tags:
  - math
  - writing
  - papers
unlisted: true
---

# Formula Explanation Ladder

Formula explanation ladder defines how much math to include when turning a paper equation into a wiki note or Korean synthesis post. The goal is not to prove everything; it is to make the claim readable and checkable.

$$
\text{formula explanation}
=
\text{symbols}
+ \text{shapes}
+ \text{distribution}
+ \text{operation}
+ \text{claim}
$$

## Levels

| Level | Include | Use When |
| --- | --- | --- |
| 0. Name | Concept name and link only | the equation is not central to the claim |
| 1. Canonical formula | one displayed equation plus symbol definitions | a reader needs the formula to follow the post |
| 2. Operational form | intermediate variables, shapes, masks, aggregation, update rule | the method is defined by the computation |
| 3. Claim contract | objective, estimator, metric, split, uncertainty, and failure mode | the formula supports a paper or benchmark claim |
| 4. Derivation note | assumptions, approximation, proof sketch, edge cases | the math itself is the topic |

Most Korean posts should use level 1 or 2 and link to wiki notes for level 3 or 4. Paper notes should reach level 3 when a reported result depends on the equation.

## Required Pieces

| Piece | Question |
| --- | --- |
| Symbols | What does every variable mean? |
| Shape | Is each quantity scalar, vector, matrix, tensor, graph, coordinate set, or distribution? |
| Index | What does the index range over: token, node, edge, residue, atom, time, sample, or candidate? |
| Distribution | Where does each expectation or sample come from? |
| Operation | Is the formula projecting, normalizing, aggregating, differentiating, sampling, or scoring? |
| Parameter | What is optimized, estimated, conditioned on, or held fixed? |
| Evidence | How does this formula connect to the metric, split, benchmark, or claim? |

## Example Pattern

Use this shape for objectives when possible:

$$
\mathcal{J}(\theta)
=
\mathbb{E}_{u\sim q(u)}
\left[
w(u)\ell_\theta(u)
\right]
$$

- $u$: sampled unit.
- $q(u)$: sampling distribution.
- $w(u)$: optional weight, mask, or importance term.
- $\ell_\theta(u)$: per-unit loss, score, reward, residual, or surrogate.
- $\theta$: optimized parameters.

Then state whether $\mathcal{J}$ is aligned with the reported metric or only a training surrogate.

## Formula Families

| Family | Minimum Level | Extra Detail |
| --- | --- | --- |
| Attention | 2 | $Q$, $K$, $V$, scaling, mask, softmax, head concatenation, output projection |
| GNN message passing | 2 | node state, edge state, neighbor set, message, aggregation, update |
| Diffusion or flow | 2 | state, time, noise/score/velocity target, conditioning, sampler |
| Contrastive learning | 2 | anchor, positive, negative set, similarity, temperature, denominator |
| Coordinate model | 3 | frame, symmetry, alignment, atom/residue mapping, coordinate metric |
| Benchmark metric | 3 | denominator, aggregation, selection rule, uncertainty, invalid-output policy |
| Likelihood or ELBO | 3 | modeled distribution, latent variables, approximation, bound direction |

## Post vs Wiki Boundary

| Surface | Good Depth |
| --- | --- |
| Korean post | level 1 or 2, enough to explain the reader-facing claim |
| Concept wiki note | level 2 or 3, with reusable symbol definitions and checks |
| Paper note | level 3, because the equation is evidence for a specific claim |
| Math note | level 3 or 4 when assumptions or derivation are the topic |

## Stop Conditions

- A formula is copied without local notation.
- Symbols are defined by memory instead of the source.
- A shorthand hides important intermediate variables.
- A loss is presented as if it were the final metric.
- A finite-sample metric is presented as a population guarantee.
- A coordinate equation omits frame, symmetry, or mapping assumptions.

## Related

- [[math/formula-intake|Formula intake]]
- [[math/index|Math]]
- [[concepts/math/index|Math foundations]]
- [[concepts/machine-learning/objective-metric-alignment|Objective-metric alignment]]
- [[concepts/evaluation/benchmark-claim-contract|Benchmark claim contract]]
- [[concepts/evaluation/claim-evidence-boundary|Claim-evidence boundary]]
