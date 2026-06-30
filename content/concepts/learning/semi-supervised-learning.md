---
title: Semi-Supervised Learning
tags:
  - learning
  - semi-supervised-learning
  - weak-labels
---

# Semi-Supervised Learning

Semi-supervised learning uses a small labeled set together with a larger unlabeled or weakly labeled set. It is useful when labels are expensive, incomplete, noisy, or selectively observed.

Let:

$$
\mathcal{D}_L
=
\{(x_i,y_i)\}_{i=1}^{n}
$$

be labeled data and:

$$
\mathcal{D}_U
=
\{x_j\}_{j=1}^{m}
$$

be unlabeled data. A common objective combines supervised and unsupervised terms:

$$
\min_\theta
\mathcal{L}_{\mathrm{sup}}(\theta;\mathcal{D}_L)
+
\lambda
\mathcal{L}_{\mathrm{unsup}}(\theta;\mathcal{D}_U)
$$

where $\lambda$ controls the strength of the unlabeled-data objective.

## Common Signals

- Pseudo-labeling: use a model prediction as a temporary label.
- Consistency regularization: predictions should remain stable under augmentation.
- Teacher-student training: use a teacher model or moving average model to guide the student.
- Graph or nearest-neighbor propagation: spread labels through similarity structure.
- Weak-label bootstrapping: combine heuristic labels with cleaner held-out evaluation labels.

## Pseudo-Labeling

For a model distribution $p_\theta(y\mid x)$, a pseudo-label can be:

$$
\tilde{y}
=
\arg\max_y p_\theta(y\mid x)
$$

Often only high-confidence predictions are used:

$$
\max_y p_\theta(y\mid x) \ge \tau
$$

where $\tau$ is a confidence threshold selected without using the final test set.

## Consistency Regularization

Consistency methods ask predictions to be stable under perturbations:

$$
\mathcal{L}_{\mathrm{cons}}
=
\mathbb{E}_{x\sim\mathcal{D}_U}
\left[
d\left(
p_\theta(\cdot\mid a_1(x)),
p_\theta(\cdot\mid a_2(x))
\right)
\right]
$$

where $a_1,a_2$ are augmentations and $d$ is a divergence or distance.

This only helps if augmentations preserve the label. In molecular, protein, or scientific data, augmentation validity must be domain-aware.

## Data Contract

Semi-supervised learning should keep label sources separate.

| Set | Contains | Use |
| --- | --- | --- |
| clean labeled | human/assay/verified labels | supervised loss and validation/test |
| unlabeled | inputs without labels | consistency, pseudo-labeling, pretraining |
| weak labeled | heuristic/noisy labels | auxiliary signal |
| pseudo-labeled | model-generated labels | training only, not final truth |

Do not merge these into one label column without a label-source field.

## Failure Modes

| Failure | Symptom |
| --- | --- |
| confirmation bias | pseudo-label errors become stronger over training |
| threshold bias | only easy examples enter pseudo-label set |
| distribution mismatch | unlabeled pool changes representation in wrong direction |
| false negative pool | unknown positives treated as unlabeled negatives |
| evaluation contamination | unlabeled pool contains test near-duplicates |

## Risks

- Confirmation bias: the model reinforces its own wrong predictions.
- Hidden label noise: weak labels are treated as clean labels.
- Sampling bias: unlabeled data comes from a different distribution.
- Leakage: unlabeled data includes near-duplicates or future test examples.
- Evaluation mismatch: final claims use weak labels instead of clean labels.

## Checks

- Which data are labeled, unlabeled, weakly labeled, or pseudo-labeled?
- Is the unlabeled pool from the same target distribution as deployment?
- Are pseudo-label thresholds selected on validation data?
- Is final evaluation done on clean labels?
- Are weak negatives separated from true negatives?
- Can the method leak information through duplicate unlabeled examples?
- Is label source preserved in the schema?
- Are pseudo-labels evaluated against clean labels before being trusted?

## Related

- [[concepts/learning/supervised-learning|Supervised learning]]
- [[concepts/learning/self-supervised-learning|Self-supervised learning]]
- [[concepts/learning/knowledge-distillation|Knowledge distillation]]
- [[concepts/learning/active-learning|Active learning]]
- [[concepts/data/weak-label|Weak label]]
- [[concepts/data/data-schema|Data schema]]
- [[concepts/data/missing-data|Missing data]]
- [[concepts/data/label-noise|Label noise]]
- [[concepts/data/sampling-bias|Sampling bias]]
- [[concepts/evaluation/evaluation-protocol|Evaluation protocol]]
