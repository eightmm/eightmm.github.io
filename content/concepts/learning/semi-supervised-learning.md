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

## Related

- [[concepts/learning/supervised-learning|Supervised learning]]
- [[concepts/learning/self-supervised-learning|Self-supervised learning]]
- [[concepts/learning/knowledge-distillation|Knowledge distillation]]
- [[concepts/learning/active-learning|Active learning]]
- [[concepts/data/weak-label|Weak label]]
- [[concepts/data/missing-data|Missing data]]
- [[concepts/data/label-noise|Label noise]]
- [[concepts/data/sampling-bias|Sampling bias]]
- [[concepts/evaluation/evaluation-protocol|Evaluation protocol]]
