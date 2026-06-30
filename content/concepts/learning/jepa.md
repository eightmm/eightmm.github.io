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

## View Contract

JEPA needs a context view and a target view:

$$
x_c = V_c(x),
\qquad
x_t = V_t(x)
$$

The objective is useful only if $x_c$ contains enough information to predict a meaningful abstraction of $x_t$, but not enough for a trivial copy. The view contract should state:

| Part | Question |
| --- | --- |
| context view | what information remains visible? |
| target view | what is hidden, future, cropped, masked, or held out? |
| target encoder | online, EMA, frozen, or teacher model? |
| predictor | what maps context embedding to target embedding? |
| collapse prevention | stop-gradient, EMA, variance regularization, normalization, or architectural asymmetry? |
| downstream claim | what task should the latent target help solve? |

## Collapse Boundary

A trivial solution is:

$$
q_\theta(f_\theta(x_c))
=
g_\xi(x_t)
=
\text{constant}
$$

for all examples. This can make the squared error small while destroying representation quality. JEPA-style systems therefore need an explicit non-collapse mechanism and downstream evaluation.

Common mechanisms include:

| Mechanism | Role |
| --- | --- |
| stop-gradient | prevents both sides from chasing each other symmetrically |
| EMA target encoder | gives a slowly moving target representation |
| predictor asymmetry | makes online and target pathways different |
| variance/covariance regularization | discourages constant embeddings |
| batch or feature normalization | controls embedding scale |

## Target Abstraction

JEPA differs from input reconstruction because the target is a latent representation:

$$
z_t = g_\xi(x_t)
$$

The target encoder decides which details matter. For images, it may avoid pixel-level noise. For protein, molecule, graph, or structure inputs, the target representation must preserve the biological or geometric information needed downstream.

## Compared With Other SSL Objectives

- Masked modeling predicts tokens, pixels, nodes, residues, or other input-space targets.
- Contrastive learning aligns positives and separates negatives.
- JEPA predicts latent target representations and may avoid explicit negatives.

The main design question is whether the target representation contains the abstraction needed for downstream tasks.

| Objective | Predicts | Main Failure |
| --- | --- | --- |
| masked modeling | input-space target | learns local reconstruction shortcuts |
| contrastive learning | positive identity among negatives | false negatives and augmentation errors |
| JEPA | target embedding | collapse or wrong target abstraction |

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
- Is the target encoder defining a useful abstraction or merely a moving shortcut?
- Is collapse checked with variance, covariance, nearest-neighbor, or downstream metrics?

## Related

- [[concepts/learning/self-supervised-learning|Self-supervised learning]]
- [[concepts/learning/contrastive-learning|Contrastive learning]]
- [[concepts/learning/masked-modeling|Masked modeling]]
- [[concepts/learning/representation-collapse|Representation collapse]]
- [[concepts/learning/representation-evaluation|Representation evaluation]]
- [[concepts/learning/augmentation-policy|Augmentation policy]]
- [[concepts/architectures/transformer|Transformer]]
