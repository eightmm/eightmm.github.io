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

The trajectory objective is:

$$
J(\theta)
=
\mathbb{E}_{\tau\sim\pi_\theta}
\left[
R(\tau)
\right]
$$

where $\tau=(s_0,a_0,\ldots)$ and $R(\tau)$ is the total return. The log-derivative trick gives:

$$
\nabla_\theta J(\theta)
=
\mathbb{E}_{\tau\sim\pi_\theta}
\left[
R(\tau)\nabla_\theta \log p_\theta(\tau)
\right]
$$

and for a policy-controlled trajectory:

$$
\log p_\theta(\tau)
=
\sum_t \log \pi_\theta(a_t\mid s_t)
+ \text{terms independent of }\theta
$$

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

Generalized advantage estimation uses temporal-difference residuals:

$$
\delta_t
=
r_t+\gamma V(s_{t+1})-V(s_t)
$$

and:

$$
A_t^{\mathrm{GAE}(\gamma,\lambda)}
=
\sum_{l=0}^{\infty}
(\gamma\lambda)^l\delta_{t+l}
$$

This trades bias and variance through $\lambda$.

## KL-Regularized Objective

Many post-training setups constrain the learned policy near a reference policy:

$$
\max_\theta
\mathbb{E}_{x,y\sim\pi_\theta}
\left[
r(x,y)
-
\beta
D_{\mathrm{KL}}
(
\pi_\theta(\cdot\mid x)
\|
\pi_{\mathrm{ref}}(\cdot\mid x)
)
\right]
$$

For language models, $a_t$ is a token and the trajectory is the generated response. The KL term prevents the policy from exploiting reward-model artifacts too aggressively.

## PPO-Style Clipped Surrogate

For samples from an old policy, define:

$$
\rho_t(\theta)
=
\frac{\pi_\theta(a_t\mid s_t)}
{\pi_{\theta_{\mathrm{old}}}(a_t\mid s_t)}
$$

The clipped objective is:

$$
L^{\mathrm{CLIP}}(\theta)
=
\mathbb{E}_t
\left[
\min
\left(
\rho_t(\theta)A_t,
\operatorname{clip}(\rho_t(\theta),1-\epsilon,1+\epsilon)A_t
\right)
\right]
$$

This limits large policy updates when the new policy diverges from the sampled distribution.

## Key Ideas

- Policy gradients can optimize non-differentiable rewards because the reward does not need to be backpropagated through the environment.
- Variance control is central; baselines, advantage estimation, and clipping are practical stabilizers.
- The policy can drift away from a useful reference unless constrained.
- In LLM post-training, actions are generated tokens or trajectories, and rewards often come from a [[concepts/learning/reward-modeling|reward model]] or verifier.

## Where It Fits

| Setting | State | Action | Reward |
|---|---|---|---|
| classical RL | environment observation | environment action | task return |
| LLM post-training | prompt and partial response | next token or response | reward model/verifier score |
| tool-using agent | task state and history | tool call or message | success, cost, safety, verification |
| molecule generation | partial molecule/trajectory | edit, token, atom, bond | property or constraint score |

Policy gradient is attractive when the reward is non-differentiable, but sample efficiency and reward misspecification become central risks.

## Practical Checks

- What trajectory distribution produced the samples: current policy, old policy, or logged data?
- Is there a KL constraint, clipping rule, or reference policy?
- Are rewards normalized or shaped in a way that changes the intended behavior?
- Does higher reward correspond to better external task success?
- Is the reward model evaluated separately from the optimized policy?
- Are KL, clipping, and entropy terms reported with coefficients?
- Is the metric computed on policy samples or a filtered subset?
- Does the agent exploit the evaluator, tool environment, or formatting rule?

## Related

- [[concepts/learning/reinforcement-learning|Reinforcement learning]]
- [[concepts/learning/reward-modeling|Reward modeling]]
- [[concepts/learning/preference-optimization|Preference optimization]]
- [[concepts/machine-learning/objective-metric-alignment|Objective metric alignment]]
- [[concepts/machine-learning/optimization|Optimization]]
- [[agents/verification/verification-loop|Verification loop]]
