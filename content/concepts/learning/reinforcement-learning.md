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

## Practical Checks

- What is the state, action, reward, and episode boundary?
- Is the reward sparse, dense, delayed, learned, or human-provided?
- Does the policy overfit to the evaluator or exploit loopholes?
- Is evaluation done on held-out tasks with external checks rather than only reward score?

## Related

- [[concepts/learning/policy-gradient|Policy gradient]]
- [[concepts/learning/reward-modeling|Reward modeling]]
- [[concepts/learning/preference-optimization|Preference optimization]]
- [[concepts/learning/imitation-learning|Imitation learning]]
- [[agents/index|Agents]]
- [[agents/agent-evaluation|Agent evaluation]]
