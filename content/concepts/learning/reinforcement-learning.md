---
title: Reinforcement Learning
tags:
  - reinforcement-learning
  - learning
  - agents
---

# Reinforcement Learning

Reinforcement learning trains a policy to choose actions that maximize reward over time. It is the learning framework behind many control, game, robotics, alignment, and agent optimization settings.

An agent interacts with an environment through states $s_t$, actions $a_t$, rewards $r_t$, and transitions:

$$
s_t \xrightarrow{a_t\sim \pi_\theta(\cdot\mid s_t)} s_{t+1},
\qquad
r_t = r(s_t,a_t,s_{t+1})
$$

The objective is expected discounted return:

$$
J(\theta)
=
\mathbb{E}_{\tau\sim\pi_\theta}
\left[
\sum_{t=0}^{T}\gamma^t r(s_t,a_t)
\right]
$$

where $\tau$ is a trajectory and $\gamma$ discounts future reward.

## Key Ideas

- RL optimizes sequential decisions, not just independent input-output labels.
- The reward function defines what behavior is reinforced, so reward misspecification is a central failure mode.
- Exploration matters because the policy only learns from states and actions it visits.
- In language and agent systems, the environment may be a user, task suite, verifier, simulator, or reward model.
- Many post-training pipelines combine [[concepts/learning/supervised-learning|supervised learning]], [[concepts/learning/reward-modeling|reward modeling]], and [[concepts/learning/policy-gradient|policy gradient]] methods.

## MDP Contract

| Field | Question |
| --- | --- |
| State $s_t$ | What information is available to the policy? |
| Action $a_t$ | What can the policy choose? |
| Transition | simulator, user, tool, environment, or static dataset? |
| Reward $r_t$ | task metric, learned reward, verifier, human feedback, or sparse success? |
| Episode boundary | when does the trajectory end? |
| Policy class | language model, controller, search policy, agent planner, or molecular generator? |
| Evaluation | return, success rate, win rate, safety, cost, or external metric? |

If a paper does not specify these fields, the RL claim is underspecified.

## Policy Gradient Sketch

For trajectories $\tau=(s_0,a_0,\ldots)$:

$$
\nabla_\theta J(\theta)
=
\mathbb{E}_{\tau\sim\pi_\theta}
\left[
\sum_t
\nabla_\theta \log \pi_\theta(a_t\mid s_t)
\hat{A}_t
\right]
$$

where $\hat{A}_t$ is an advantage estimate. The estimator is stochastic, so sample count, rollout policy, and variance reduction matter.

## Offline vs Online RL

| Setting | Data Source | Main Risk |
| --- | --- | --- |
| online RL | current policy interacts with environment | reward hacking and unsafe exploration |
| offline RL | fixed logged dataset | out-of-distribution actions |
| RLHF/RLAIF | reward or preference model | proxy exploitation |
| verifier RL | automated checks | overfitting to verifier loopholes |
| agent RL | task environment with tools | hidden cost, invalid actions, brittle reward |

## Paper Evidence Boundary

| Claim | Required Evidence |
| --- | --- |
| higher reward | independent task metric or reward-model audit |
| better agent performance | held-out tasks, tool-call validity, cost accounting |
| better safety | adversarial and slice-specific safety eval |
| better scientific design | external property, assay, simulation, or expert review |
| sample efficiency | fixed environment interactions and compute budget |

## Practical Checks

- What is the state, action, reward, and episode boundary?
- Is the reward sparse, dense, delayed, learned, or human-provided?
- Does the policy overfit to the evaluator or exploit loopholes?
- Is evaluation done on held-out tasks with external checks rather than only reward score?
- Are rollouts, seeds, environment versions, and action constraints reported?
- Is cost measured in environment steps, tokens, tool calls, wall time, or compute?

## Related

- [[concepts/learning/policy-gradient|Policy gradient]]
- [[concepts/learning/reward-modeling|Reward modeling]]
- [[concepts/learning/preference-optimization|Preference optimization]]
- [[concepts/learning/imitation-learning|Imitation learning]]
- [[concepts/math/monte-carlo-estimation|Monte Carlo estimation]]
- [[agents/index|Agents]]
- [[agents/verification/agent-evaluation|Agent evaluation]]
