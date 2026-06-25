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

## Checks

- What process produced $y$?
- Are labels comparable across sources?
- Are thresholds, units, and censoring rules documented?
- Are missing labels treated as negatives?
- Does the metric reward the intended semantic target or an easier proxy?

## Related

- [[concepts/data/annotation-labeling|Annotation and labeling]]
- [[concepts/data/label-noise|Label noise]]
- [[concepts/evaluation/negative-set|Negative set]]
- [[concepts/evaluation/assay-harmonization|Assay harmonization]]
- [[entities/assay|Assay]]
