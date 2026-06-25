---
title: Policy Gradient
tags:
  - reinforcement-learning
  - policy-gradient
  - optimization
---

# Policy Gradient

Policy gradient methods optimize a parameterized policy directly by estimating how changing $\theta$ changes expected return.

For a stochastic policy $\pi_\theta(a\mid s)$, the policy gradient theorem gives:

$$
\nabla_\theta J(\theta)
=
\mathbb{E}_{\tau\sim\pi_\theta}
\left[
\sum_t
\nabla_\theta \log \pi_\theta(a_t\mid s_t)
G_t
\right]
$$

where $G_t$ is a return estimate from timestep $t$.

A baseline can reduce variance:

$$
\nabla_\theta J(\theta)
=
\mathbb{E}
\left[
\nabla_\theta \log \pi_\theta(a_t\mid s_t)
(G_t - b(s_t))
\right]
$$

Actor-critic methods use a learned value function as the baseline:

$$
A_t = G_t - V_\psi(s_t)
$$

where $A_t$ is an advantage estimate.

## Key Ideas

- Policy gradients can optimize non-differentiable rewards because the reward does not need to be backpropagated through the environment.
- Variance control is central; baselines, advantage estimation, and clipping are practical stabilizers.
- The policy can drift away from a useful reference unless constrained.
- In LLM post-training, actions are generated tokens or trajectories, and rewards often come from a [[concepts/learning/reward-modeling|reward model]] or verifier.

## Practical Checks

- What trajectory distribution produced the samples: current policy, old policy, or logged data?
- Is there a KL constraint, clipping rule, or reference policy?
- Are rewards normalized or shaped in a way that changes the intended behavior?
- Does higher reward correspond to better external task success?

## Related

- [[concepts/learning/reinforcement-learning|Reinforcement learning]]
- [[concepts/learning/reward-modeling|Reward modeling]]
- [[concepts/learning/preference-optimization|Preference optimization]]
- [[concepts/machine-learning/optimization|Optimization]]
- [[agents/verification-loop|Verification loop]]
