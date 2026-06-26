---
title: Representation Learning
tags:
  - machine-learning
  - representation-learning
---

# Representation Learning

Representation learning trains a model to transform raw inputs into features that make downstream tasks easier.

A representation model maps inputs to embeddings:

$$
z = f_\theta(x),
\qquad
z\in\mathbb{R}^{d}
$$

A downstream model then predicts from $z$:

$$
\hat{y}=g_\phi(z)
$$

The representation can be learned with supervised labels, self-supervised signals, contrastive objectives, masked modeling, or generative pretraining.

The useful representation is not necessarily the one that preserves everything. It should preserve information needed for downstream tasks and discard nuisance variation:

$$
z=f_\theta(x),
\qquad
I(z;y)\ \text{large},
\qquad
I(z;\epsilon)\ \text{small}
$$

where $y$ is task-relevant information and $\epsilon$ is nuisance variation.

## Representation Contract

| Field | Question |
| --- | --- |
| input object | Is $x$ a token, sequence, image, graph, molecule, protein, pocket, complex, or structure? |
| representation unit | Does one $z$ represent a token, residue, node, graph, molecule, protein, pair, or full example? |
| pooling/readout | Is $z$ obtained by CLS token, mean pooling, graph pooling, attention pooling, or task-specific readout? |
| invariance | Which transformations should leave $z$ unchanged: permutation, rotation, translation, augmentation, or assay source? |
| equivariance | Does the representation need to transform with coordinates, vectors, forces, or fields? |
| selection rule | Which checkpoint, layer, pooling rule, or embedding dimension is selected on validation data? |
| evaluation budget | Is the downstream evaluator linear, kNN, retrieval, shallow model, or full fine-tuning? |

## Objective Families

| Family | Typical Objective | Main Claim |
| --- | --- | --- |
| supervised representation | $\mathcal{L}(g_\phi(f_\theta(x)), y)$ | representation is useful for the supervised label |
| contrastive learning | pull positive views together, push negatives apart | representation preserves view-invariant semantics |
| masked modeling | reconstruct masked token, patch, node, residue, or coordinate | context encodes predictable structure |
| predictive coding / JEPA | predict latent target from context | representation captures abstract state rather than raw pixels/tokens |
| autoencoding | reconstruct $x$ through a bottleneck | compression preserves enough information |
| generative pretraining | maximize or approximate $p_\theta(x)$ | learned features support likelihood or sampling |

For contrastive representation learning, a common normalized similarity form is:

$$
s_{ij}
= \frac{z_i^\top z_j}{\lVert z_i\rVert_2\lVert z_j\rVert_2},
\qquad
z_i=f_\theta(x_i)
$$

The meaning of a "positive" pair must match the downstream claim. Two augmented molecules, two protein homologs, or two views of the same image encode very different assumptions.

## Failure Modes

| Failure | Symptom | Check |
| --- | --- | --- |
| collapse | embeddings become nearly identical | variance, rank, covariance spectrum |
| shortcut feature | representation encodes source, split, scaffold, or artifact | source-stratified metric and leakage audit |
| wrong invariance | augmentation removes task-relevant signal | view policy vs downstream label semantics |
| overpowered evaluator | downstream head learns the task from labels, not the representation | compare linear probe, kNN, and fine-tuning budgets |
| cache leakage | embeddings are fitted, normalized, or selected using test data | cache provenance and split-aware preprocessing |

## Checks

- What information should the representation preserve or discard?
- Is the representation instance-level, token-level, graph-level, or structure-level?
- Does it transfer to downstream tasks under a realistic split?
- Are embeddings evaluated with [[concepts/learning/linear-probing|linear probes]], [[concepts/learning/fine-tuning-protocol|fine-tuning]], retrieval, or task metrics?
- Does the representation avoid [[concepts/learning/representation-collapse|collapse]]?
- Is the selected layer, pooling rule, and embedding dimension fixed before final test evaluation?
- Does the representation claim survive a simple baseline such as fingerprints, sequence identity, or bag-of-tokens?

## Related

- [[concepts/modalities/representation-contract|Representation contract]]
- [[concepts/learning/self-supervised-learning|Self-supervised learning]]
- [[concepts/learning/contrastive-learning|Contrastive learning]]
- [[concepts/learning/masked-modeling|Masked modeling]]
- [[concepts/learning/jepa|JEPA]]
- [[concepts/learning/augmentation-policy|Augmentation policy]]
- [[concepts/learning/representation-evaluation|Representation evaluation]]
- [[concepts/learning/linear-probing|Linear probing]]
- [[concepts/learning/fine-tuning-protocol|Fine-tuning protocol]]
- [[concepts/learning/representation-collapse|Representation collapse]]
- [[concepts/architectures/pooling-readout|Pooling and readout]]
- [[concepts/math/singular-value-decomposition|Singular value decomposition]]
- [[concepts/machine-learning/feature-engineering|Feature engineering]]
- [[concepts/protein-modeling/protein-representation|Protein representation]]
- [[concepts/architectures/embedding|Embedding]]
