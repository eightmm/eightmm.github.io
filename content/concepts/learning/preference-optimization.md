---
title: Preference Optimization
tags:
  - preference-optimization
  - alignment
  - machine-learning
---

# Preference Optimization

Preference optimization trains a model to prefer outputs that humans, users, tests, or another evaluator rank higher. It includes reward-model-based [[concepts/learning/reinforcement-learning|reinforcement learning]] and direct objectives over preference pairs.

For pairwise preference data $(y_w, y_l)$, a common loss is:

$$
\mathcal{L}
= -\log \sigma(r_\theta(x,y_w) - r_\theta(x,y_l))
$$

Here $y_w$ is the preferred output, $y_l$ is the less preferred output, and $r_\theta$ is a reward or preference score.

## Why It Matters

- Aligns model behavior with goals that are hard to specify as a loss.
- Common in the post-training stage of large generative models.
- Quality depends heavily on the preference data and reward signal.

## Checks

- Are preference labels consistent and free of annotator bias?
- Does optimization reward-hack the proxy instead of the true objective?
- How much does the policy drift from the reference model?
- Is the preference signal collected from realistic tasks or artificial comparisons?

## Related

- [[concepts/learning/fine-tuning|Fine-tuning]]
- [[concepts/learning/reward-modeling|Reward modeling]]
- [[concepts/learning/policy-gradient|Policy gradient]]
- [[concepts/learning/reinforcement-learning|Reinforcement learning]]
- [[concepts/learning/supervised-learning|Supervised learning]]
- [[concepts/generative-models/autoregressive-model|Autoregressive model]]
- [[agents/agent-evaluation|Agent evaluation]]
