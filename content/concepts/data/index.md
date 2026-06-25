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
- [[concepts/data/dataset-card|Dataset card]]
- [[concepts/data/example-unit|Example unit]]
- [[concepts/data/split-unit|Split unit]]
- [[concepts/data/data-distribution|Data distribution]]
- [[concepts/data/data-schema|Data schema]]
- [[concepts/data/preprocessing-contract|Preprocessing contract]]
- [[concepts/data/data-curation|Data curation]]
- [[concepts/data/data-versioning|Data versioning]]
- [[concepts/data/annotation-labeling|Annotation and labeling]]
- [[concepts/data/label-semantics|Label semantics]]
- [[concepts/data/label-noise|Label noise]]
- [[concepts/data/dataset-shift|Dataset shift]]
- [[concepts/data/sampling-strategy|Sampling strategy]]
- [[concepts/data/metadata-provenance|Metadata and provenance]]
- [[concepts/data/benchmark|Benchmark]]
- [[concepts/machine-learning/data-preprocessing|Data preprocessing]]

## Checks

- What is one example?
- What is the split unit?
- What distribution should the dataset represent?
- What schema defines the fields, units, identifiers, and relationships?
- What produces the target label?
- What exactly does the label mean?
- What metadata explains batch, source, protocol, time, species, target, or structure context?
- What examples are missing because of collection bias?
- What shift exists between train, validation, test, and deployment?
- Does the split match the intended generalization claim?
- Can another person reconstruct the dataset version and filtering policy?
- Are labels noisy, censored, inconsistent, or protocol-dependent?

## Related

- [[ai/index|AI]]
- [[concepts/modalities/index|Modalities]]
- [[concepts/tasks/index|Tasks]]
- [[concepts/evaluation/index|Evaluation]]
- [[concepts/evaluation/leakage|Leakage]]
- [[concepts/systems/reproducibility|Reproducibility]]
