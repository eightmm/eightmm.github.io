---
title: Contrastive Learning
tags:
  - learning
  - self-supervised-learning
  - contrastive-learning
---

# Contrastive Learning

Contrastive learning trains representations by pulling related views (positives) together in embedding space and pushing unrelated views (negatives) apart. Positives are usually augmented views of the same instance; negatives are other instances.

The common InfoNCE form is:

$$
\mathcal{L}_i
= -\log
\frac{\exp(\operatorname{sim}(z_i,z_i^+)/\tau)}
\sum_{j} \exp(\operatorname{sim}(z_i,z_j)/\tau)}
$$

Here $z_i^+$ is a positive view, $z_j$ are candidate positives/negatives, $\operatorname{sim}$ is a similarity function, and $\tau$ is temperature.

With normalized embeddings, cosine similarity is:

$$
\operatorname{sim}(z_i,z_j)
=
\frac{z_i^\top z_j}
{\lVert z_i\rVert_2\lVert z_j\rVert_2}
$$

The temperature $\tau$ controls how sharply the model distinguishes close and far examples.

## Batch Objective View

For a batch of anchors and candidates, InfoNCE is a classification problem over the positive index:

$$
p_\theta(j\mid i)
=
\frac{\exp(s_{ij}/\tau)}
{\sum_{k\in \mathcal{C}_i}\exp(s_{ik}/\tau)}
$$

$$
\mathcal{L}_i
=
-\log p_\theta(j=i^+\mid i)
$$

where $s_{ij}=\operatorname{sim}(z_i,z_j)$ and $\mathcal{C}_i$ is the candidate set. This makes the negative set part of the objective, not an implementation detail.

## Why It Matters

- Learns useful representations without labels, from the structure of views alone.
- The augmentation policy encodes the invariances you want the representation to have.
- Applies to sequences, graphs, molecules, and proteins where defining a meaningful "same entity, different view" is natural.

## Design Questions

- What defines a positive pair?
- What defines a negative pair?
- Which augmentations preserve the meaning needed for downstream tasks?
- Is the objective instance-discrimination, supervised contrastive learning, cross-modal alignment, or retrieval training?

## Positive and Negative Semantics

For an anchor $x_i$, a positive should preserve the relevant identity:

$$
y(x_i) = y(x_i^+)
$$

for the intended downstream meaning $y$. A false negative occurs when $x_j$ is treated as negative even though it shares the relevant class, scaffold, family, target, or semantic state.

## Design Matrix

| Choice | Question | Failure Mode |
| --- | --- | --- |
| view function | what transformations produce $v_1,v_2$? | augmentation removes task-relevant signal |
| positive rule | what counts as same identity? | positives too easy or semantically wrong |
| negative set | batch, memory bank, queue, corpus, supervised labels? | false negatives or train/test contamination |
| similarity | dot product, cosine, learned score? | scale and normalization change temperature meaning |
| projection head | train-only or downstream representation? | evaluation uses a different representation than training |
| temperature | fixed, learned, scheduled? | overly sharp distribution and unstable gradients |

## Chem-Bio Examples

| Object | Possible Positive | Risk |
| --- | --- | --- |
| molecule | two SMILES views of same standardized molecule | tautomer/protonation changes may alter label |
| molecular graph | graph augmentation preserving scaffold or pharmacophore | atom/bond deletion may change activity |
| protein sequence | masked/cropped homologous views | homolog positives can leak family-level labels |
| protein structure | coordinate/noise views of same fold | alignment or template source may leak |
| complex | ligand-pocket views of same interaction | known pose can leak deployment-unavailable geometry |

## Checks

- Do augmentations preserve label-relevant content, or do they destroy it (e.g. a molecular edit that changes activity)?
- Are negatives actually negative, or do they include near-duplicates and same-family members (false negatives)?
- Does the split avoid leakage through near-duplicates, shared scaffolds, or protein families?
- Is representation collapse occurring (all embeddings converging to one point)?
- Does the batch or memory bank contain enough hard negatives without contaminating the split?
- Is the downstream evaluation linear probe, kNN, retrieval, or full fine-tuning?
- Are false negatives likely under scaffold, family, or target-based semantics?

## Related

- [[concepts/learning/self-supervised-learning|Self-supervised learning]]
- [[concepts/learning/augmentation-policy|Augmentation policy]]
- [[concepts/learning/representation-collapse|Representation collapse]]
- [[concepts/learning/representation-evaluation|Representation evaluation]]
- [[concepts/machine-learning/objective-metric-alignment|Objective-metric alignment]]
- [[concepts/learning/jepa|JEPA]]
- [[concepts/evaluation/leakage|Leakage]]
- [[entities/protein|Protein]]
