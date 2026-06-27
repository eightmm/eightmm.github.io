---
title: Formula Pattern Catalog
tags:
  - math
  - ai
  - papers
unlisted: true
---

# Formula Pattern Catalog

Use this catalog when a paper equation is important but the notation is unfamiliar. The goal is to rewrite the equation into a known pattern, define the sampled unit, and connect it to the claim.

Most paper formulas can be reduced to:

$$
\text{quantity}
=
\operatorname{aggregate}_{u \sim q(u)}
\left[
\text{operation}_\theta(u)
\right]
$$

where $u$ is the sampled unit, $q(u)$ is the sampling distribution, and $\text{operation}_\theta$ is a model, loss, score, update, estimator, or metric.

## Pattern Map

| Pattern | Canonical Form | Main Question |
| --- | --- | --- |
| Empirical risk | $\frac{1}{n}\sum_i \mathcal{L}(f_\theta(x_i), y_i)$ | what unit, label, and loss are optimized? |
| Maximum likelihood | $\sum_i \log p_\theta(x_i)$ | what distribution is modeled? |
| Conditional likelihood | $\sum_i \log p_\theta(y_i\mid x_i)$ | what is conditioned on and what is predicted? |
| Cross entropy | $-\sum_k y_k \log p_{\theta,k}$ | are targets hard labels, soft labels, or distributions? |
| Contrastive loss | $-\log \frac{\exp(s(a,p)/\tau)}{\sum_j \exp(s(a,j)/\tau)}$ | what are anchor, positive, negative set, and temperature? |
| Reconstruction | $\lVert x-\hat{x}_\theta\rVert$ | what information is allowed through the bottleneck? |
| ELBO | $\mathbb{E}_{q_\phi(z\mid x)}[\log p_\theta(x\mid z)]-\mathrm{KL}(q_\phi(z\mid x)\Vert p(z))$ | what is latent, decoder, prior, and approximation? |
| Denoising | $\lVert \epsilon-\epsilon_\theta(x_t,t,c)\rVert^2$ | what noise level and target are used? |
| Flow matching | $\lVert v_\theta(x_t,t,c)-u_t\rVert^2$ | what path and velocity target define training? |
| Score matching | $\lVert s_\theta(x_t,t)-\nabla_x\log p_t(x_t)\rVert^2$ | what score is estimated and how is it sampled? |
| Message passing | $h_i'=\phi(h_i,\square_{j\in\mathcal{N}(i)}\psi(h_i,h_j,e_{ij}))$ | what graph, neighbors, messages, and aggregation are used? |
| Attention | $\operatorname{softmax}(QK^\top/\sqrt{d_k}+M)V$ | what tokens attend to what, under which mask? |
| Equivariance | $\phi(g\cdot x)=\rho(g)\phi(x)$ | what group acts and how should output transform? |
| Metric estimate | $\frac{1}{m}\sum_j M(\hat{y}_j,y_j)$ | what examples, selection rule, and denominator define the score? |

## Supervised Objective

The default supervised pattern is:

$$
\hat{\theta}
=
\arg\min_\theta
\frac{1}{n}
\sum_{i=1}^{n}
\mathcal{L}
\left(
f_\theta(x_i), y_i
\right)
$$

- $x_i$: input object or representation.
- $y_i$: label, target, class, scalar, ranking, coordinates, or structured output.
- $f_\theta$: model.
- $\mathcal{L}$: surrogate loss.
- $n$: training examples after sampling and filtering.

Checks:

- Does $x_i$ mean raw input or processed representation?
- Does $y_i$ include context such as target, assay, time, or condition?
- Is $\mathcal{L}$ the same object as the final metric?
- Is the final test set untouched by model selection?

## Likelihood and Conditional Likelihood

Unconditional likelihood:

$$
\hat{\theta}
=
\arg\max_\theta
\sum_{i=1}^{n}
\log p_\theta(x_i)
$$

Conditional likelihood:

$$
\hat{\theta}
=
\arg\max_\theta
\sum_{i=1}^{n}
\log p_\theta(y_i \mid x_i, c_i)
$$

- $x_i$: observed object or condition.
- $y_i$: predicted object.
- $c_i$: optional context such as prompt, target, pocket, assay, or class.

Checks:

- Is the paper modeling data likelihood, label likelihood, or sequence likelihood?
- Is likelihood tractable or replaced by a surrogate bound?
- Does high likelihood imply the reported utility claim?

## Contrastive Objective

A common contrastive form is:

$$
\mathcal{L}_i
=
-
\log
\frac{
\exp(\operatorname{sim}(z_i, z_i^+)/\tau)
}{
\exp(\operatorname{sim}(z_i, z_i^+)/\tau)
+
\sum_{j\in\mathcal{N}_i}
\exp(\operatorname{sim}(z_i, z_j^-)/\tau)
}
$$

- $z_i$: anchor representation.
- $z_i^+$: positive representation.
- $z_j^-$: negative representation.
- $\mathcal{N}_i$: negative set.
- $\tau$: temperature.

Checks:

- What makes a positive pair valid?
- Are negatives true negatives, in-batch negatives, hard negatives, or false negatives?
- Does the downstream evaluator test representation quality or only the contrastive setup?

## Generative Path Objective

Diffusion, flow matching, and score models usually define a path:

$$
x_0 \sim p_{\mathrm{data}},
\qquad
x_t \sim p_t(x_t\mid x_0),
\qquad
\mathcal{L}(\theta)
=
\mathbb{E}_{x_0,t,x_t}
\left[
\lVert g_\theta(x_t,t,c)-g^\star(x_t,t,x_0,c)\rVert^2
\right]
$$

- $x_0$: clean data sample.
- $x_t$: noised or interpolated state.
- $t$: time or noise level.
- $c$: condition.
- $g_\theta$: predicted noise, score, velocity, or clean sample.
- $g^\star$: training target.

Checks:

- Is the target noise, score, velocity, $x_0$, or another parameterization?
- What sampling solver turns the learned field into a sample?
- Are validity, novelty, diversity, and utility evaluated separately from the training loss?

## Graph and Set Aggregation

Graph equations often hide permutation assumptions:

$$
m_i
=
\square_{j\in\mathcal{N}(i)}
\psi_\theta(h_i,h_j,e_{ij}),
\qquad
h_i'
=
\phi_\theta(h_i,m_i)
$$

- $\mathcal{N}(i)$: neighbors of node $i$.
- $\psi_\theta$: message function.
- $\square$: permutation-invariant aggregation such as sum, mean, max, or attention.
- $\phi_\theta$: update function.

Checks:

- What is a node: atom, residue, token, region, candidate, or object?
- What is an edge: bond, contact, distance threshold, sequence adjacency, or learned relation?
- Does the aggregation preserve the required permutation behavior?

## Coordinate and Symmetry Rule

Coordinate-heavy models should state the transformation rule:

$$
x_i'
=
R x_i + t,
\qquad
d_{ij}
=
\lVert x_i - x_j\rVert_2
$$

For equivariant outputs:

$$
\phi(RX + t)
=
R\phi(X)
$$

For invariant scalar outputs:

$$
f(RX+t)
=
f(X)
$$

Checks:

- Is the target scalar, vector, coordinate, force, velocity, or field?
- Is the group translation, rotation, reflection, permutation, $SO(3)$, $SE(3)$, or $E(3)$?
- Does preprocessing leak a coordinate frame unavailable at inference?

## Metric and Estimator

Reported scores are estimators:

$$
\hat{M}
=
\frac{1}{m}
\sum_{j=1}^{m}
M(\hat{y}_j,y_j)
$$

- $m$: evaluated examples after filtering.
- $M$: metric function.
- $\hat{y}_j$: prediction after thresholding, ranking, sampling, or selection.
- $y_j$: reference label or object.

Checks:

- Is the metric averaged per example, per target, per scaffold, per dataset, or per task?
- Was the threshold, checkpoint, or sampler selected on validation data?
- Are confidence intervals, paired comparisons, or seed variation needed?

## Claim Routing

| Formula Looks Like | Route First |
| --- | --- |
| supervised loss | [Machine learning](/ai/machine-learning), [Loss function](/concepts/machine-learning/loss-function) |
| likelihood or KL | [Information and likelihood](/math/information-likelihood) |
| contrastive or masked objective | [Learning methods](/ai/learning-methods) |
| diffusion, score, or flow objective | [Generative models](/ai/generative-models) |
| graph message passing | [Architectures](/ai/architectures), [Discrete math and graphs](/math/discrete-graphs) |
| coordinate transform | [Geometry and symmetry](/math/geometry-symmetry) |
| reported metric | [Evaluation math](/math/evaluation-math) |
| mixed AI and domain claim | [AI Computational Biology Math contract](/concepts/ai-computational-biology-math-contract) |

## Related

- [[math/formula-intake|Formula intake]]
- [[math/formula-explanation-ladder|Formula explanation ladder]]
- [[math/information-likelihood|Information and likelihood]]
- [[math/dynamical-systems|Dynamical systems]]
- [[math/evaluation-math|Evaluation math]]
- [[ai/paper-intake|AI paper intake]]
- [[concepts/ai-computational-biology-math-contract|AI Computational Biology Math contract]]
