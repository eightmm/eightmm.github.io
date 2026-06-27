---
title: Constrained Optimization
tags:
  - math
  - optimization
  - constraints
---

# Constrained Optimization

Constrained optimization optimizes an objective while requiring the solution to satisfy equality or inequality constraints. It appears in regularized training, normalization, simplex probabilities, KL-constrained post-training, molecular geometry, and constrained generation.

A general problem is:

$$
\begin{aligned}
\min_x \quad & f(x) \\
\text{subject to}\quad
& h_i(x)=0,\quad i=1,\ldots,m \\
& g_j(x)\le 0,\quad j=1,\ldots,k
\end{aligned}
$$

The objective says what should be improved. The constraints say what must remain valid.

## Equality Constraints and Lagrangian

For equality constraints $h(x)=0$, the Lagrangian is:

$$
\mathcal{L}(x,\lambda)
=
f(x)
+
\lambda^\top h(x)
$$

At a local optimum under regularity conditions:

$$
\nabla_x \mathcal{L}(x^\*,\lambda^\*)
=
0,
\qquad
h(x^\*)=0
$$

The multiplier $\lambda$ measures how strongly the constraint affects the optimum.

## Inequality Constraints and KKT Conditions

For inequality constraints $g_j(x)\le 0$, the Lagrangian is:

$$
\mathcal{L}(x,\lambda,\mu)
=
f(x)
+
\lambda^\top h(x)
+
\mu^\top g(x)
$$

The Karush-Kuhn-Tucker conditions include:

$$
\nabla_x \mathcal{L}(x^\*,\lambda^\*,\mu^\*)=0
$$

$$
h(x^\*)=0,
\qquad
g(x^\*)\le 0
$$

$$
\mu^\*\ge 0,
\qquad
\mu_j^\* g_j(x^\*)=0
$$

The last condition is complementary slackness: an inactive inequality has zero multiplier.

## Penalty and Barrier Forms

Many ML and molecular workflows do not solve constrained problems exactly. They turn constraints into penalties:

$$
\min_x
f(x)
+
\rho
\sum_i h_i(x)^2
+
\rho
\sum_j
\max(0,g_j(x))^2
$$

or use barriers that become large near invalid regions:

$$
\min_x
f(x)
-
\tau
\sum_j \log(-g_j(x))
$$

Penalties are easier to optimize, but they do not guarantee exact feasibility unless the protocol enforces it.

## Projection View

Projected gradient descent takes an unconstrained step and projects back to the feasible set $\mathcal{C}$:

$$
x_{t+1}
=
\Pi_{\mathcal{C}}
\left(
x_t-\eta\nabla f(x_t)
\right)
$$

where:

$$
\Pi_{\mathcal{C}}(z)
=
\arg\min_{x\in\mathcal{C}}
\lVert x-z\rVert_2^2
$$

This is useful for simplex constraints, norm balls, valid coordinate sets, and constrained decoding abstractions.

## Common AI and Computational Biology Examples

| Setting | Constraint | Typical handling |
|---|---|---|
| probability vector | $\sum_i p_i=1,\ p_i\ge 0$ | softmax or simplex projection |
| KL-regularized policy | $D_{\mathrm{KL}}(\pi_\theta\|\pi_{\mathrm{ref}})\le \epsilon$ | penalty, clipping, or trust region |
| molecular coordinates | bond lengths, clashes, chirality, pocket geometry | force field, projection, filtering |
| graph generation | valence, connectivity, atom types | constrained decoder or validity filter |
| normalization | mean/variance or norm constraint | reparameterization or layer normalization |
| resource-aware training | memory, latency, token budget | constrained model selection |

## Constraint vs Regularization

Hard constraint:

$$
x\in\mathcal{C}
$$

Soft regularization:

$$
\min_x f(x)+\lambda R(x)
$$

A regularizer encourages behavior; a constraint requires behavior. Many papers blur this distinction, so check whether invalid outputs are impossible, penalized, filtered, or silently removed.

## Paper Reading Checks

- Is the constraint hard, soft, architectural, decoding-time, or post-hoc?
- Is feasibility guaranteed or only encouraged by a penalty?
- Are invalid samples counted in the denominator?
- Are constraint coefficients tuned on validation data?
- Does the constraint exist during deployment, or only during evaluation?
- Is the reported improvement from the model, projection, repair, or filtering?

## Related

- [[concepts/machine-learning/optimization|Optimization]]
- [[concepts/machine-learning/regularization|Regularization]]
- [[concepts/learning/policy-gradient|Policy gradient]]
- [[concepts/generative-models/guidance|Guidance]]
- [[concepts/generative-models/molecular-generation|Molecular generation]]
- [[concepts/molecular-modeling/energy-minimization|Energy minimization]]
- [[concepts/geometric-deep-learning/coordinate-update|Coordinate update]]
- [[concepts/tasks/structured-prediction|Structured prediction]]
