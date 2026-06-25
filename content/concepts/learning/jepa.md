---
title: JEPA
tags:
  - learning
  - self-supervised-learning
  - jepa
---

# JEPA

Joint-embedding predictive architectures (JEPA) train models to predict the representation of a missing or future part of the data in embedding space, rather than reconstructing it in input space. A context encoder, a target encoder, and a predictor are trained together; the target encoder is usually held stable (e.g. an EMA copy) to avoid trivial collapse.

A simplified JEPA objective is:

$$
\mathcal{L}
= \left\|
q_\theta(f_\theta(x_{\mathrm{context}}))
- \operatorname{sg}(g_\xi(x_{\mathrm{target}}))
\right\|_2^2
$$

Here $f_\theta$ encodes context, $g_\xi$ encodes the target view, $q_\theta$ predicts in embedding space, and $\operatorname{sg}$ stops gradients through the target.

The target encoder is often updated by exponential moving average:

$$
\xi \leftarrow m\xi + (1-m)\theta
$$

where $m\in[0,1)$ controls how slowly the target network follows the online network.

## Compared With Other SSL Objectives

- Masked modeling predicts tokens, pixels, nodes, residues, or other input-space targets.
- Contrastive learning aligns positives and separates negatives.
- JEPA predicts latent target representations and may avoid explicit negatives.

The main design question is whether the target representation contains the abstraction needed for downstream tasks.

## Why It Matters

- Avoids reconstructing every low-level detail, unlike pixel/token-level [[concepts/learning/masked-modeling|masked modeling]].
- Focuses learning on useful abstractions instead of unpredictable noise.
- Can be considered for sequences, structures, graphs, and multimodal settings.
- Separates "what is predictable" from "what must be reconstructed."

## Checks

- Is representation collapse prevented (stop-gradient, EMA target, or regularization on the target encoder)?
- Are the predicted targets at the right level of abstraction for the downstream task?
- Does the masking/context strategy leave a genuinely predictive but non-trivial gap?
- Does the learned representation transfer beyond the pretraining distribution?
- Are context and target views sampled without leaking future or deployment-unavailable information?

## Related

- [[concepts/learning/self-supervised-learning|Self-supervised learning]]
- [[concepts/learning/contrastive-learning|Contrastive learning]]
- [[concepts/learning/masked-modeling|Masked modeling]]
- [[concepts/learning/representation-collapse|Representation collapse]]
- [[concepts/learning/augmentation-policy|Augmentation policy]]
- [[concepts/architectures/transformer|Transformer]]
