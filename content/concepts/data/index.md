---
title: Data
tags:
  - data
  - machine-learning
---

# Data

Data notes describe how examples, labels, metadata, splits, and benchmarks are constructed before any model is trained. In applied AI, the dataset often defines the real problem more strongly than the architecture.

A dataset induces an empirical distribution:

$$
\hat{p}_{\mathcal{D}}(x,y)
= \frac{1}{|\mathcal{D}|}\sum_{(x_i,y_i)\in\mathcal{D}}
\delta(x=x_i,y=y_i)
$$

Training and evaluation are only meaningful when this empirical distribution is aligned with the intended deployment distribution.

## Topics

- [[entities/dataset|Dataset]]
- [[concepts/data/dataset-construction-checklist|Dataset construction checklist]]
- [[concepts/data/dataset-card|Dataset card]]
- [[concepts/data/example-unit|Example unit]]
- [[concepts/data/split-unit|Split unit]]
- [[concepts/data/dataset-split-contract|Dataset split contract]]
- [[concepts/data/data-distribution|Data distribution]]
- [[concepts/data/data-schema|Data schema]]
- [[concepts/data/preprocessing-contract|Preprocessing contract]]
- [[concepts/data/data-curation|Data curation]]
- [[concepts/data/data-versioning|Data versioning]]
- [[concepts/data/annotation-labeling|Annotation and labeling]]
- [[concepts/data/label-semantics|Label semantics]]
- [[concepts/data/label-noise|Label noise]]
- [[concepts/data/missing-data|Missing data]]
- [[concepts/data/censored-label|Censored label]]
- [[concepts/data/weak-label|Weak label]]
- [[concepts/data/class-imbalance|Class imbalance]]
- [[concepts/data/dataset-shift|Dataset shift]]
- [[concepts/data/sampling-strategy|Sampling strategy]]
- [[concepts/data/sampling-bias|Sampling bias]]
- [[concepts/data/metadata-provenance|Metadata and provenance]]
- [[concepts/data/data-lineage|Data lineage]]
- [[concepts/data/benchmark-intake|Benchmark intake]]
- [[concepts/data/benchmark|Benchmark]]
- [[concepts/evaluation/benchmark-claim-contract|Benchmark claim contract]]
- [[concepts/evaluation/evaluation-set-design|Evaluation set design]]
- [[concepts/machine-learning/data-preprocessing|Data preprocessing]]

## Checks

- What is one example?
- What is the split unit?
- What distribution should the dataset represent?
- What schema defines the fields, units, identifiers, and relationships?
- What produces the target label?
- What exactly does the label mean?
- Are missing, censored, and weak labels represented explicitly?
- What metadata explains batch, source, protocol, time, species, target, or structure context?
- What examples are missing because of collection bias?
- Is the class or label distribution imbalanced?
- Does sampling bias make validation easier than deployment?
- What shift exists between train, validation, test, and deployment?
- Does the split match the intended generalization claim?
- Can another person reconstruct the dataset version and filtering policy?
- Are labels noisy, censored, inconsistent, or protocol-dependent?

## Construction Path

1. Define [[concepts/data/example-unit|example unit]] and [[concepts/data/split-unit|split unit]].
2. Write the [[concepts/data/data-schema|data schema]] and [[concepts/data/label-semantics|label semantics]].
3. Specify the [[concepts/data/preprocessing-contract|preprocessing contract]].
4. Record [[concepts/data/metadata-provenance|metadata and provenance]] and [[concepts/data/data-lineage|data lineage]].
5. Choose the [[concepts/data/sampling-strategy|sampling strategy]] and [[concepts/data/dataset-split-contract|dataset split contract]].
6. Summarize the result in a [[concepts/data/dataset-card|dataset card]].
7. Design the [[concepts/evaluation/evaluation-set-design|evaluation set]].
8. Attach an [[concepts/evaluation/evaluation-protocol|evaluation protocol]].
9. Use [[concepts/data/benchmark-intake|benchmark intake]] and [[concepts/evaluation/benchmark-claim-contract|benchmark claim contract]] before treating a score as evidence for a paper claim.

## Related

- [[ai/index|AI]]
- [[concepts/modalities/index|Modalities]]
- [[concepts/tasks/index|Tasks]]
- [[concepts/evaluation/index|Evaluation]]
- [[concepts/evaluation/leakage|Leakage]]
- [[concepts/evaluation/test-set-contamination|Test-set contamination]]
- [[concepts/systems/reproducibility|Reproducibility]]
