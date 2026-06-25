---
title: Transfer Learning
tags:
  - transfer-learning
  - representation-learning
  - machine-learning
---

# Transfer Learning

Transfer learning reuses knowledge learned on a source task or domain to improve learning on a related target task, typically by reusing pretrained representations.

One way to view it is source pretraining followed by target adaptation:

$$
\theta_s = \arg\min_\theta \mathcal{L}_{\mathrm{source}}(\theta),
\qquad
\theta_t = \arg\min_\theta \mathcal{L}_{\mathrm{target}}(\theta;\theta_s)
$$

The benefit depends on how much useful structure transfers from source to target.

## Why It Matters

- Reduces the labeled data needed on the target task.
- Lets large-scale pretraining amortize across many downstream uses.
- Common in language, vision, and molecular/protein modeling.

## Checks

- How close are the source and target distributions?
- Does the pretrained feature space cover the target's relevant variation?
- Are gains real, or an artifact of overlapping train/test data?
- Is transfer measured with a frozen probe, full fine-tuning, retrieval, or a task-specific evaluator?

## Related

- [[concepts/learning/pretraining|Pretraining]]
- [[concepts/learning/domain-adaptation|Domain adaptation]]
- [[concepts/learning/fine-tuning|Fine-tuning]]
- [[concepts/learning/linear-probing|Linear probing]]
- [[concepts/learning/representation-evaluation|Representation evaluation]]
- [[concepts/learning/self-supervised-learning|Self-supervised learning]]
- [[concepts/learning/supervised-learning|Supervised learning]]
