---
title: Weak Label
tags:
  - data
  - labels
  - supervision
---

# Weak Label

A weak label is a noisy, indirect, incomplete, or heuristic supervision signal. It can be useful, but it should not be described as ground truth without qualification.

A weak-label dataset can be written as:

$$
\tilde{y}_i
=
h(x_i, m_i, r_i)
$$

where $\tilde{y}_i$ is the observed weak label, $m_i$ is metadata or source context, and $r_i$ is a rule, annotator, retrieval process, teacher model, assay proxy, or distant supervision method.

## Sources

- Heuristic rules or keyword matches.
- Distant supervision from metadata or linked databases.
- Pseudo-labels from a teacher model.
- User interactions such as clicks, views, or selections.
- Thresholded or aggregated noisy measurements.
- Weak positives and assumed negatives in partially observed datasets.

## Risk

Weak labels can encode shortcuts:

$$
\tilde{y}
\approx
\operatorname{source}(x)
$$

instead of the intended semantic target. A model can then learn source artifacts, annotation policy, or collection bias.

## Uses

- Pretraining or bootstrapping when clean labels are scarce.
- Candidate filtering before manual review.
- Semi-supervised learning or teacher-student training.
- Noisy ranking or retrieval supervision.

Weak labels are often best treated as training signals, not final evaluation labels.

## Checks

- What process generated the weak label?
- What is the expected noise or bias pattern?
- Is the weak label used for training, validation, or final test evaluation?
- Is there a cleaner held-out set for final claims?
- Could metadata or source fields leak the weak-label rule?
- Are weak negatives actually unknown examples?

## Related

- [[concepts/data/annotation-labeling|Annotation and labeling]]
- [[concepts/data/label-semantics|Label semantics]]
- [[concepts/data/label-noise|Label noise]]
- [[concepts/data/missing-data|Missing data]]
- [[concepts/data/sampling-bias|Sampling bias]]
- [[concepts/learning/semi-supervised-learning|Semi-supervised learning]]
- [[concepts/learning/knowledge-distillation|Knowledge distillation]]
- [[concepts/evaluation/evaluation-protocol|Evaluation protocol]]
