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

## Route Map

| Need | Start | Then Check |
| --- | --- | --- |
| define what one row means | [Example unit](/concepts/data/example-unit) | [Data schema](/concepts/data/data-schema), [Label semantics](/concepts/data/label-semantics) |
| define what must be held out | [Split unit](/concepts/data/split-unit) | [Dataset split contract](/concepts/data/dataset-split-contract), [Leakage](/concepts/evaluation/leakage) |
| prepare raw records | [Preprocessing contract](/concepts/data/preprocessing-contract) | [Data preprocessing](/concepts/machine-learning/data-preprocessing), [Data lineage](/concepts/data/data-lineage) |
| describe a reusable dataset | [Dataset card](/concepts/data/dataset-card) | [Metadata and provenance](/concepts/data/metadata-provenance), [Data versioning](/concepts/data/data-versioning) |
| build a benchmark | [Benchmark intake](/concepts/data/benchmark-intake) | [Benchmark claim contract](/concepts/evaluation/benchmark-claim-contract), [Evaluation set design](/concepts/evaluation/evaluation-set-design) |
| handle imperfect labels | [Label noise](/concepts/data/label-noise) | [Censored label](/concepts/data/censored-label), [Weak label](/concepts/data/weak-label) |
| handle distribution problems | [Dataset shift](/concepts/data/dataset-shift) | [Sampling bias](/concepts/data/sampling-bias), [Class imbalance](/concepts/data/class-imbalance) |

## Topic Groups

| Group | Notes |
| --- | --- |
| Dataset definition | [Dataset](/entities/dataset), [Dataset construction checklist](/concepts/data/dataset-construction-checklist), [Data distribution](/concepts/data/data-distribution) |
| Schema and metadata | [Data schema](/concepts/data/data-schema), [Metadata and provenance](/concepts/data/metadata-provenance), [Data lineage](/concepts/data/data-lineage) |
| Curation and versions | [Data curation](/concepts/data/data-curation), [Data versioning](/concepts/data/data-versioning), [Annotation and labeling](/concepts/data/annotation-labeling) |
| Sampling | [Sampling strategy](/concepts/data/sampling-strategy), [Sampling bias](/concepts/data/sampling-bias), [Missing data](/concepts/data/missing-data) |
| Benchmarking | [Benchmark](/concepts/data/benchmark), [Benchmark intake](/concepts/data/benchmark-intake), [Evaluation protocol](/concepts/evaluation/evaluation-protocol) |

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
