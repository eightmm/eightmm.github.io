---
title: Preference Optimization
tags:
  - preference-optimization
  - alignment
  - machine-learning
---

# Preference Optimization

Preference optimization trains a model to prefer outputs that humans (or a proxy) rank higher. It includes reward-model-based reinforcement learning and direct objectives over preference pairs.

## Why It Matters

- Aligns model behavior with goals that are hard to specify as a loss.
- Common in the post-training stage of large generative models.
- Quality depends heavily on the preference data and reward signal.

## Checks

- Are preference labels consistent and free of annotator bias?
- Does optimization reward-hack the proxy instead of the true objective?
- How much does the policy drift from the reference model?

## Related

- [[concepts/learning/fine-tuning|Fine-tuning]]
- [[concepts/learning/supervised-learning|Supervised learning]]
- [[concepts/generative-models/autoregressive-model|Autoregressive model]]
