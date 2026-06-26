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
- Is it binary, multiclass, multilabel, ordinal, continuous, [[concepts/data/censored-label|censored]], [[concepts/data/weak-label|weak]], or missing?
- Does it represent ground truth, proxy signal, preference, rank, or a noisy observation?
- Is the threshold arbitrary or domain-defined?
- Are negative labels true negatives or simply unobserved positives?

## Label Tuple

For supervised AI and computational biology notes, record a label as:

$$
y
=
(\text{endpoint}, \text{unit}, \text{direction}, \text{threshold}, \text{censoring}, \text{source})
$$

The same numeric value can mean different things if any field changes. For example, an activity value from one assay should not be pooled with another endpoint unless the note explains the harmonization rule.

| Field | Question |
| --- | --- |
| Endpoint | What is measured: affinity, inhibition, toxicity, expression, relevance, class, preference, or reward? |
| Unit | Is the value in molar units, transformed units, probability, count, rank, score, or category? |
| Direction | Does larger mean stronger, weaker, safer, more likely, or worse? |
| Threshold | How is a continuous measurement converted to a class or decision? |
| Censoring | Is the value exact, upper-bounded, lower-bounded, interval-censored, or only a qualifier? |
| Source | Which assay, annotator, benchmark, simulator, or curation process produced it? |

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

[[concepts/data/missing-data|Missing labels]] are not automatically negative labels:

$$
y_i\ \text{missing}
\not\Rightarrow
y_i = 0
$$

This matters in retrieval, virtual screening, activity prediction, preference data, and any dataset where observations are selectively recorded.

## Objective and Metric Consequence

Label semantics decide which loss and metric are defensible:

$$
\ell(f_\theta(x), y)
\quad\text{is meaningful only if}\quad
y \sim p_{\mathrm{label}}(y \mid u, c, m)
$$

where $u$ is the example unit, $c$ is context, and $m$ is the measurement process. If $m$ changes across sources, the model may learn source effects instead of the intended target.

| Label Type | Common Loss | Evaluation Risk |
| --- | --- | --- |
| binary threshold | cross-entropy, focal loss | threshold and class prevalence define the claim |
| continuous measurement | MSE, MAE, Gaussian NLL | unit, censoring, and outlier policy dominate interpretation |
| rank or preference | pairwise/listwise loss | pair construction may not match deployment ranking |
| censored assay value | censored likelihood or bound-aware loss | treating bounds as exact values biases the model |
| weak or noisy label | robust loss, positive-unlabeled setup | missing labels can become false negatives |

## Checks

- What process produced $y$?
- Are labels comparable across sources?
- Are thresholds, units, and censoring rules documented?
- Are missing labels treated as negatives?
- Does the metric reward the intended semantic target or an easier proxy?
- Does the label require context such as assay, target, dose, pocket, prompt, or source?
- Is the target-assay-label contract preserved through preprocessing and splitting?
- Is the optimized loss compatible with the label type and censoring rule?
- Is the reported metric measuring the same semantic target as the label?

## Related

- [[concepts/data/dataset-construction-checklist|Dataset construction checklist]]
- [[concepts/data/example-unit|Example unit]]
- [[concepts/data/split-unit|Split unit]]
- [[concepts/data/annotation-labeling|Annotation and labeling]]
- [[concepts/data/label-noise|Label noise]]
- [[concepts/data/missing-data|Missing data]]
- [[concepts/data/censored-label|Censored label]]
- [[concepts/data/weak-label|Weak label]]
- [[concepts/evaluation/negative-set|Negative set]]
- [[concepts/evaluation/assay-harmonization|Assay harmonization]]
- [[entities/assay|Assay]]
- [[entities/target-assay-label|Target-assay-label contract]]
- [[entities/bioactivity-label|Bioactivity label]]
- [[concepts/machine-learning/objective-metric-alignment|Objective-metric alignment]]
