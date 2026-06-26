---
title: Self-Supervised Learning
tags:
  - self-supervised-learning
  - representation-learning
  - machine-learning
---

# Self-Supervised Learning

Self-supervised learning trains representations from structure in the data instead of direct human labels. The target can be masked tokens, corrupted views, contrastive pairs, future states, or reconstruction objectives.

The common pattern is to create a training signal from the input:

$$
t = T(x),
\qquad
\min_\theta \mathcal{L}(f_\theta(V(x)), t)
$$

Here $V(x)$ is the visible or augmented view and $T(x)$ constructs a target from the same raw example.

More explicitly, many SSL methods sample one or more views:

$$
v_1, v_2 \sim \mathcal{A}(x)
$$

where $\mathcal{A}$ is an [[concepts/learning/augmentation-policy|augmentation policy]]. The method then defines which information should match between views and which information can be ignored.

## Objective Families

| Family | Target | Main Risk |
| --- | --- | --- |
| Masked prediction | hidden token, pixel, node, residue, atom, or feature | target may be too local or too easy |
| Contrastive learning | positive identity among negatives | false negatives and augmentation semantics |
| Joint-embedding prediction | target representation | collapse prevention and target abstraction |
| Autoregressive prediction | future or next token/state | teacher-forcing mismatch and context leakage |
| Reconstruction or denoising | clean input from corrupted input | reconstructing nuisance detail instead of useful abstraction |

## Pretraining Data Boundary

SSL is often limited by data construction rather than architecture. Record:

$$
\mathcal{D}_{\mathrm{ssl}}
=
\{x_i,\ a_i,\ s_i,\ t_i\}_{i=1}^{n}
$$

where $a_i$ is augmentation or corruption metadata, $s_i$ is source, and $t_i$ is time or split group when relevant. If test-like examples or near-duplicates appear in pretraining, downstream evaluation can overstate generalization.

## Why It Matters

- Useful when labels are sparse or expensive.
- Can pretrain sequence, graph, molecular, and protein representations.
- Often shifts the main question from architecture to data construction and evaluation.
- Makes data curation and leakage control part of the learning method, not only preprocessing.

## Evaluation

SSL is usually evaluated indirectly through downstream performance:

$$
z=f_\theta(x),
\qquad
\hat{y}=g_\phi(z)
$$

Common evaluation modes are [[concepts/learning/linear-probing|linear probing]], [[concepts/learning/fine-tuning-protocol|full fine-tuning]], retrieval, clustering quality, and task-specific metrics. These belong to [[concepts/learning/representation-evaluation|representation evaluation]], not to the pretraining objective itself.

## Evaluation Budget

Different evaluation protocols answer different questions:

| Protocol | Tests | Caveat |
| --- | --- | --- |
| frozen linear probe | linear separability of frozen representation | weak evaluator may understate useful representation |
| kNN or retrieval | neighborhood structure | corpus and metric define the claim |
| full fine-tuning | adaptation performance | mixes pretraining quality with optimization budget |
| low-data fine-tuning | sample efficiency | needs matched search budget and seeds |
| OOD or grouped split | transfer beyond pretraining distribution | split must match the biological or deployment claim |

## Checks

- Is the pretext task aligned with the downstream task?
- Can train and test data leak through near-duplicates, scaffolds, protein families, or temporal splits?
- Does the representation transfer beyond the pretraining distribution?
- Does the objective avoid [[concepts/learning/representation-collapse|representation collapse]]?
- Are augmentation and masking choices valid for the domain?
- Are linear probe, fine-tuning, and retrieval protocols kept separate?
- Are pretraining and evaluation splits deduplicated at the right entity level?
- Is the adaptation budget fixed before comparing representations?

## Related

- [[ai/learning-methods|Learning methods]]
- [[concepts/learning/pretraining|Pretraining]]
- [[concepts/learning/masked-modeling|Masked modeling]]
- [[concepts/learning/contrastive-learning|Contrastive learning]]
- [[concepts/learning/jepa|JEPA]]
- [[concepts/learning/augmentation-policy|Augmentation policy]]
- [[concepts/learning/representation-collapse|Representation collapse]]
- [[concepts/learning/representation-evaluation|Representation evaluation]]
- [[concepts/learning/linear-probing|Linear probing]]
- [[concepts/learning/fine-tuning-protocol|Fine-tuning protocol]]
- [[concepts/machine-learning/objective-metric-alignment|Objective-metric alignment]]
- [[concepts/evaluation/test-set-contamination|Test set contamination]]
- [[molecular-modeling/protein-modeling|Protein modeling]]
- [[concepts/protein-modeling/protein-representation|Protein representation]]
- [[concepts/geometric-deep-learning/equivariant-gnn|Equivariant GNN]]
