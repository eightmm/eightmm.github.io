---
title: Label Semantics
tags:
  - data
  - labels
  - evaluation
---

# Label Semantics

Label semantics define what a target value actually means. Two datasets can use the same label name while measuring different biological, behavioral, visual, or operational concepts.

A label is produced by a measurement or annotation process:

$$
y_i
=
g(x_i, a_i, q_i)
$$

where $a_i$ represents annotators, assay protocol, instrument, or source, and $q_i$ represents quality control or preprocessing decisions.

## Common Label Questions

- Is the label observed directly, measured indirectly, inferred, or curated?
- Is it binary, multiclass, multilabel, ordinal, continuous, censored, or weak?
- Does it represent ground truth, proxy signal, preference, rank, or a noisy observation?
- Is the threshold arbitrary or domain-defined?
- Are negative labels true negatives or simply unobserved positives?

## Examples of Semantic Drift

- A binary active/inactive label may depend on assay threshold.
- A toxicity label may depend on cell line, dose, duration, and endpoint.
- A relevance label may mean clicked, judged useful, cited, or retrieved.
- A segmentation mask may represent visible object boundary, annotator convention, or task-specific region.

## Label Function

A label should be described as a function of example, context, and measurement:

$$
y_i = h(u_i, c_i, m_i)
$$

where $u_i$ is the example unit, $c_i$ is the task context, and $m_i$ is the measurement or annotation process. If two records have the same $u_i$ but different $c_i$ or $m_i$, their labels may not be interchangeable.

For chem-bio data, the context is often the [[entities/target-assay-label|Target-assay-label contract]]: molecule, target, assay, endpoint, unit, threshold, censoring, and source.

## Missing and Negative Labels

Missing labels are not automatically negative labels:

$$
y_i\ \text{missing}
\not\Rightarrow
y_i = 0
$$

This matters in retrieval, virtual screening, activity prediction, preference data, and any dataset where observations are selectively recorded.

## Checks

- What process produced $y$?
- Are labels comparable across sources?
- Are thresholds, units, and censoring rules documented?
- Are missing labels treated as negatives?
- Does the metric reward the intended semantic target or an easier proxy?
- Does the label require context such as assay, target, dose, pocket, prompt, or source?
- Is the target-assay-label contract preserved through preprocessing and splitting?

## Related

- [[concepts/data/dataset-construction-checklist|Dataset construction checklist]]
- [[concepts/data/example-unit|Example unit]]
- [[concepts/data/annotation-labeling|Annotation and labeling]]
- [[concepts/data/label-noise|Label noise]]
- [[concepts/evaluation/negative-set|Negative set]]
- [[concepts/evaluation/assay-harmonization|Assay harmonization]]
- [[entities/assay|Assay]]
- [[entities/target-assay-label|Target-assay-label contract]]
- [[entities/bioactivity-label|Bioactivity label]]
