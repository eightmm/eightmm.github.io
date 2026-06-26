---
title: Annotation and Labeling
tags:
  - data
  - labels
  - annotation
---

# Annotation and Labeling

Annotation converts observations into target labels. A label is not automatically ground truth; it is a measurement, human judgment, derived proxy, or heuristic under a protocol.

A supervised dataset can be written as:

$$
\mathcal{D}=\{(x_i,y_i,m_i)\}_{i=1}^{n}
$$

where $m_i$ stores metadata about how the label was produced.

## Label Types

- Human annotation: category, span, mask, preference, ranking, or free-text explanation.
- Experimental measurement: assay value, binding proxy, phenotype, expression value, or structure-derived label.
- Derived label: computed property, thresholded activity, similarity label, or benchmark heuristic.
- [[concepts/data/weak-label|Weak label]]: noisy signal from metadata, rules, retrieval, or distant supervision.

## Label Protocol

A label should be tied to a protocol:

$$
y_i
=
\operatorname{label}
(o_i;\ a,\ r,\ \tau)
$$

where $o_i$ is an observation, $a$ is the annotator or measurement apparatus, $r$ is the rule or rubric, and $\tau$ is an optional threshold.

For thresholded labels:

$$
y_i
=
\mathbf{1}[z_i \ge \tau]
$$

where $z_i$ is a score or measurement. The direction and unit of $z_i$ must be explicit.

## Agreement and Aggregation

With multiple annotations $y_{i1},\ldots,y_{ik}$, aggregation is part of the label definition:

$$
\tilde{y}_i
=
\operatorname{aggregate}
(y_{i1},\ldots,y_{ik})
$$

Aggregation may use majority vote, median, mean, expert adjudication, or keep-all measurements. The choice affects uncertainty and label noise.

## Failure Modes

- Human labels encode annotator bias or inconsistent rubric interpretation.
- Experimental labels mix units, thresholds, or assay conditions.
- Derived labels use information not available at deployment.
- Censored or missing labels are converted into point labels without policy.
- Disagreement is hidden by a single aggregate value.

## Checks

- Who or what produced the label?
- Is the label categorical, ordinal, scalar, structured, censored, or missing?
- Is the label exact, [[concepts/data/censored-label|censored]], [[concepts/data/missing-data|missing]], or weak?
- Are repeated annotations consistent?
- Are label thresholds documented?
- Does the input contain metadata that directly reveals the label?
- Is label aggregation defined before model evaluation?
- Are units, direction, and censoring rules explicit?
- Are uncertain labels modeled, filtered, or marked separately?

## Related

- [[concepts/learning/supervised-learning|Supervised learning]]
- [[concepts/data/label-semantics|Label semantics]]
- [[concepts/data/label-noise|Label noise]]
- [[concepts/data/weak-label|Weak label]]
- [[concepts/data/censored-label|Censored label]]
- [[concepts/data/missing-data|Missing data]]
- [[concepts/data/metadata-provenance|Metadata and provenance]]
- [[concepts/data/dataset-card|Dataset card]]
- [[concepts/evaluation/assay-harmonization|Assay harmonization]]
- [[entities/assay|Assay]]
- [[concepts/evaluation/leakage|Leakage]]
