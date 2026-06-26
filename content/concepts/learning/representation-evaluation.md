---
title: Representation Evaluation
tags:
  - representation-learning
  - evaluation
  - self-supervised-learning
---

# Representation Evaluation

Representation evaluation measures whether a learned embedding is useful for downstream tasks, retrieval, clustering, robustness, and transfer. It should separate the representation from the downstream evaluator as much as the claim requires.

For an encoder:

$$
z = f_\theta(x)
$$

the evaluator defines a task-specific readout:

$$
\hat{y} = g_\phi(z)
$$

Different choices of $g_\phi$ answer different questions:

- Linear probe: is task information linearly available in $z$?
- k-nearest-neighbor classifier: does local neighborhood structure match labels?
- Retrieval score: are relevant candidates close under the chosen similarity?
- Full fine-tuning: can the pretrained model adapt to the task with target labels?

## Evaluation Modes

| Mode | Trainable part | Main question |
| --- | --- | --- |
| [Linear probing](/concepts/learning/linear-probing) | Simple head only | Is the feature linearly separable? |
| kNN / retrieval | Often none | Are neighborhoods meaningful? |
| Clustering | None or clustering model | Does unsupervised structure align with labels? |
| [Fine-tuning protocol](/concepts/learning/fine-tuning-protocol) | Some or all model weights | Does the representation adapt well? |
| Task model | Readout or decoder | Does it solve the real downstream task? |

For retrieval-style evaluation:

$$
s(q, x_i)
=
\frac{z_q^\top z_i}{\lVert z_q\rVert_2\lVert z_i\rVert_2},
\qquad
\mathrm{rank}(q)
=
\mathrm{argsort}_{i}(-s(q,x_i))
$$

where $z_q=f_\theta(q)$, $z_i=f_\theta(x_i)$, and $s$ is cosine similarity. The metric can be Recall@$k$, mean reciprocal rank, enrichment, or another [[concepts/evaluation/ranking-metrics|ranking metric]].

## Protocol Requirements

- Define the representation unit: token, sequence, image, graph, molecule, protein, pocket, complex, or structure.
- Define the pooling or readout used to obtain $z$.
- Keep train, validation, and test split units aligned with the claim.
- Fit normalization, dimensionality reduction, and cache construction without test leakage.
- Separate representation selection, downstream model selection, and final test.
- Compare against simple baselines such as raw features, fingerprints, bag-of-tokens, or random encoders.

## Bio-AI Notes

For protein, molecule, and structure-based tasks, representation evaluation must state the biological or chemical split unit:

- Molecule tasks: scaffold, source, temporal, or assay split.
- Protein tasks: sequence identity cluster, family, or structure class split.
- Protein-ligand tasks: ligand group, protein family, complex pair, and assay/source boundaries.

Otherwise, the evaluation may only show interpolation over near-duplicates.

## Checks

- What object does one embedding represent?
- What information should the embedding preserve and discard?
- Which evaluator is used, and what extra capacity does it add?
- Does the split prevent near-duplicate, scaffold, family, or source leakage?
- Does the result support representation quality, downstream performance, or deployment generalization?

## Related

- [[concepts/machine-learning/representation-learning|Representation learning]]
- [[concepts/learning/self-supervised-learning|Self-supervised learning]]
- [[concepts/learning/linear-probing|Linear probing]]
- [[concepts/learning/fine-tuning-protocol|Fine-tuning protocol]]
- [[concepts/tasks/similarity-search|Similarity search]]
- [[concepts/tasks/retrieval|Retrieval]]
- [[concepts/evaluation/evaluation-protocol|Evaluation protocol]]
- [[concepts/evaluation/ranking-metrics|Ranking metrics]]
- [[concepts/evaluation/scaffold-split|Scaffold split]]
- [[concepts/evaluation/protein-family-split|Protein family split]]
- [[concepts/protein-modeling/protein-representation|Protein representation]]
