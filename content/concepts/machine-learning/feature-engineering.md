---
title: Feature Engineering
tags:
  - machine-learning
  - representation-learning
---

# Feature Engineering

Feature engineering chooses the variables given to a model. Classical machine learning depends heavily on explicit features, while deep learning often learns features through an architecture.

A feature map converts an input object into a vector:

$$
\phi: \mathcal{X}\rightarrow\mathbb{R}^{d}
$$

The model then predicts from $\phi(x)$:

$$
\hat{y}=f_\theta(\phi(x))
$$

For linear models, feature quality often determines whether a simple model is competitive:

$$
\hat{y}=w^\top \phi(x)+b
$$

## Common Choices

- Tabular descriptors, counts, fingerprints, or metadata.
- Protein sequence features, structure features, or family annotations.
- Molecular descriptors, fingerprints, conformer features, or graph features.
- Learned embeddings from pretrained models.

## Checks

- Does any feature encode the target, future information, or split identity?
- Are features fitted or selected using train data only?
- Are engineered features robust under the deployment distribution?
- Is the feature representation aligned with the architecture and metric?

## Related

- [[concepts/machine-learning/data-preprocessing|Data preprocessing]]
- [[concepts/molecular-modeling/molecular-fingerprint|Molecular fingerprint]]
- [[concepts/machine-learning/linear-model|Linear model]]
- [[concepts/learning/self-supervised-learning|Self-supervised learning]]
- [[concepts/evaluation/leakage|Leakage]]
