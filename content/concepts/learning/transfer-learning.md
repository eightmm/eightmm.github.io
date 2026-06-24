---
title: Transfer Learning
tags:
  - transfer-learning
  - representation-learning
  - machine-learning
---

# Transfer Learning

Transfer learning reuses knowledge learned on a source task or domain to improve learning on a related target task, typically by reusing pretrained representations.

## Why It Matters

- Reduces the labeled data needed on the target task.
- Lets large-scale pretraining amortize across many downstream uses.
- Common in language, vision, and molecular/protein modeling.

## Checks

- How close are the source and target distributions?
- Does the pretrained feature space cover the target's relevant variation?
- Are gains real, or an artifact of overlapping train/test data?

## Related

- [[concepts/learning/fine-tuning|Fine-tuning]]
- [[concepts/learning/self-supervised-learning|Self-supervised learning]]
- [[concepts/learning/supervised-learning|Supervised learning]]
