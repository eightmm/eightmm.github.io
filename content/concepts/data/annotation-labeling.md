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
- Weak label: noisy signal from metadata, rules, retrieval, or distant supervision.

## Checks

- Who or what produced the label?
- Is the label categorical, ordinal, scalar, structured, censored, or missing?
- Are repeated annotations consistent?
- Are label thresholds documented?
- Does the input contain metadata that directly reveals the label?

## Related

- [[concepts/learning/supervised-learning|Supervised learning]]
- [[concepts/evaluation/assay-harmonization|Assay harmonization]]
- [[entities/assay|Assay]]
- [[concepts/data/metadata-provenance|Metadata and provenance]]
- [[concepts/evaluation/leakage|Leakage]]
