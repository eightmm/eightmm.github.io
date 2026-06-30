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

## Label Source Contract

Weak labels should preserve their source:

$$
y_i^{\mathrm{obs}}
=
( \tilde{y}_i,\ s_i,\ c_i )
$$

where $s_i$ is the label source and $c_i$ is confidence, rule version, teacher model, assay proxy, or annotation context.

| Source | Main risk | Safer use |
| --- | --- | --- |
| heuristic rule | rule shortcut | training signal only |
| teacher model | teacher bias and hidden data | distillation with clean evaluation |
| metadata link | source leakage | provenance-aware split |
| user interaction | exposure bias | counterfactual or logged-policy checks |
| weak negative | unlabeled positive hidden inside negative pool | positive-unlabeled framing |

Do not overwrite clean labels and weak labels into one undifferentiated column.

## Noise Model

A weak label can be viewed as a noisy channel:

$$
p(\tilde{y}\mid y,x,s)
$$

The important question is not only noise rate, but whether noise depends on features, source, class, or time. Feature-dependent noise can create misleading validation performance if the same weak-label process appears in every split.

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
- Is label source stored separately from label value?
- Is the final evaluation based on clean labels or the same weak-label generator?

## Related

- [[concepts/data/annotation-labeling|Annotation and labeling]]
- [[concepts/data/label-semantics|Label semantics]]
- [[concepts/data/label-noise|Label noise]]
- [[concepts/data/missing-data|Missing data]]
- [[concepts/data/sampling-bias|Sampling bias]]
- [[concepts/data/metadata-provenance|Metadata and provenance]]
- [[concepts/learning/semi-supervised-learning|Semi-supervised learning]]
- [[concepts/learning/knowledge-distillation|Knowledge distillation]]
- [[concepts/evaluation/evaluation-protocol|Evaluation protocol]]
