---
title: Imitation Learning
tags:
  - imitation-learning
  - supervised-learning
  - agents
---

# Imitation Learning

Imitation learning trains a policy from demonstrations. Instead of discovering behavior only through reward, the model learns from expert trajectories or high-quality examples.

For demonstration data $\mathcal{D}=\{(s_i,a_i)\}_{i=1}^{n}$, behavioral cloning uses supervised learning:

$$
\hat{\theta}
=
\arg\min_\theta
-\frac{1}{n}\sum_{i=1}^{n}
\log \pi_\theta(a_i\mid s_i)
$$

For language models and coding agents, supervised fine-tuning on high-quality traces is a form of imitation learning.

## Key Ideas

- Demonstrations provide a dense learning signal when rewards are sparse or expensive.
- The model can copy expert behavior but may fail when it reaches states not covered by demonstrations.
- Imitation learning is often a first stage before [[concepts/learning/preference-optimization|preference optimization]] or [[concepts/learning/reinforcement-learning|reinforcement learning]].
- Trace quality matters more than volume when demonstrations encode tool use, verification, or reasoning style.

## Practical Checks

- Are demonstrations representative of the target deployment setting?
- Does the policy recover from mistakes, or only imitate clean trajectories?
- Are bad examples filtered or labeled separately?
- Does imitation preserve private or sensitive details that should be removed before public training or documentation?

## Related

- [[concepts/learning/supervised-learning|Supervised learning]]
- [[concepts/learning/fine-tuning|Fine-tuning]]
- [[concepts/learning/reinforcement-learning|Reinforcement learning]]
- [[agents/tool-use|Tool use]]
- [[agents/verification-loop|Verification loop]]
