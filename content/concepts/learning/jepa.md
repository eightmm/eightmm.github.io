---
title: JEPA
tags:
  - learning
  - self-supervised-learning
  - jepa
---

# JEPA

Joint-embedding predictive architectures (JEPA) train models to predict the representation of a missing or future part of the data in embedding space, rather than reconstructing it in input space. A context encoder, a target encoder, and a predictor are trained together; the target encoder is usually held stable (e.g. an EMA copy) to avoid trivial collapse.

## Why It Matters

- Avoids reconstructing every low-level detail, unlike pixel/token-level [[concepts/learning/masked-modeling|masked modeling]].
- Focuses learning on useful abstractions instead of unpredictable noise.
- Can be considered for sequences, structures, graphs, and multimodal settings.

## Checks

- Is representation collapse prevented (stop-gradient, EMA target, or regularization on the target encoder)?
- Are the predicted targets at the right level of abstraction for the downstream task?
- Does the masking/context strategy leave a genuinely predictive but non-trivial gap?
- Does the learned representation transfer beyond the pretraining distribution?

## Related

- [[concepts/learning/self-supervised-learning|Self-supervised learning]]
- [[concepts/learning/contrastive-learning|Contrastive learning]]
- [[concepts/architectures/transformer|Transformer]]
