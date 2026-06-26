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

## Feature Contract

| Field | Required Detail |
| --- | --- |
| object | what raw object is transformed: table row, sequence, graph, molecule, protein, structure, assay record |
| feature source | raw measurement, derived descriptor, annotation, pretrained embedding, or external metadata |
| fit scope | whether normalization, vocabulary, PCA, imputation, or selection is fit on training data only |
| availability | whether the feature is available at inference time |
| leakage risk | whether the feature encodes target, split identity, future information, source, or post hoc filtering |
| stability | whether the feature changes with software version, database version, conformer generator, or annotation source |

## Common Choices

| Domain | Feature Examples | Risk |
| --- | --- | --- |
| tabular data | descriptors, counts, metadata, missingness indicators | target leakage through post-outcome fields |
| text or sequence | token counts, k-mers, pretrained embeddings, position features | vocabulary or duplicate leakage |
| molecules | descriptors, fingerprints, graph features, conformer features | standardization, stereochemistry, tautomer, protonation, conformer policy |
| proteins | sequence identity, domains, MSA features, structure features, family annotations | homolog and template leakage |
| structures | distances, contacts, angles, pocket descriptors, interaction fingerprints | coordinate-frame and known-pose leakage |
| learned embeddings | frozen encoder features or cached representations | cache construction and checkpoint selection leakage |

## Preprocessing Pipeline

A feature pipeline should be viewed as part of the model:

$$
x
\xrightarrow{\mathrm{clean}}
\tilde{x}
\xrightarrow{\phi}
u
\xrightarrow{\mathrm{normalize}}
\bar{u}
\xrightarrow{f_\theta}
\hat{y}
$$

If any stage is fit using validation or test examples, the reported metric can become optimistic. For example, fitting a scaler on all data changes:

$$
\bar{u}
=
\frac{u-\mu_{\mathrm{train}}}{\sigma_{\mathrm{train}}}
$$

into a leakage-prone estimate if $\mu$ and $\sigma$ are computed from train plus test.

## Baseline Role

Feature-engineered models are useful baselines even when the paper proposes a deep architecture:

| Baseline | What It Tests |
| --- | --- |
| linear model on descriptors | whether the task is almost linearly separable |
| tree model on descriptors | whether tabular nonlinear interactions are enough |
| fingerprint similarity | whether analog retrieval solves molecule tasks |
| sequence identity or family baseline | whether protein tasks are mostly homolog interpolation |
| docking or physics heuristic | whether learned scoring beats classical structure-based signal |

## Checks

- Does any feature encode the target, future information, or split identity?
- Are features fitted or selected using train data only?
- Are engineered features robust under the deployment distribution?
- Is the feature representation aligned with the architecture and metric?
- Are feature versions, software versions, and external databases recorded?
- Is a simple feature baseline reported before claiming an architectural gain?

## Related

- [[concepts/machine-learning/data-preprocessing|Data preprocessing]]
- [[concepts/data/preprocessing-contract|Preprocessing contract]]
- [[concepts/data/metadata-provenance|Metadata and provenance]]
- [[concepts/modalities/representation-contract|Representation contract]]
- [[concepts/machine-learning/representation-learning|Representation learning]]
- [[concepts/molecular-modeling/molecular-fingerprint|Molecular fingerprint]]
- [[concepts/molecular-modeling/molecular-featurization-contract|Molecular featurization contract]]
- [[concepts/sbdd/interaction-fingerprint|Interaction fingerprint]]
- [[concepts/machine-learning/linear-model|Linear model]]
- [[concepts/machine-learning/tree-based-model|Tree-based model]]
- [[concepts/learning/self-supervised-learning|Self-supervised learning]]
- [[concepts/evaluation/leakage|Leakage]]
