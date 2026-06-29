---
title: Data
tags:
  - data
  - machine-learning
---

# Data

Data note는 모델을 학습하기 전에 example, label, metadata, split, benchmark가 어떻게 만들어지는지 설명합니다. Applied AI에서는 architecture보다 dataset이 실제 문제를 더 강하게 정의하는 경우가 많습니다.

A dataset induces an empirical distribution:

$$
\hat{p}_{\mathcal{D}}(x,y)
= \frac{1}{|\mathcal{D}|}\sum_{(x_i,y_i)\in\mathcal{D}}
\delta(x=x_i,y=y_i)
$$

Training과 evaluation은 이 empirical distribution이 의도한 deployment distribution과 맞을 때만 의미가 있습니다.

## 이동 지도

| 필요 | 시작점 | 이어서 확인할 것 |
| --- | --- | --- |
| define what one row means | [Example unit](/concepts/data/example-unit) | [Data schema](/concepts/data/data-schema), [Label semantics](/concepts/data/label-semantics) |
| define what must be held out | [Split unit](/concepts/data/split-unit) | [Dataset split contract](/concepts/data/dataset-split-contract), [Leakage](/concepts/evaluation/leakage) |
| prepare raw records | [Preprocessing contract](/concepts/data/preprocessing-contract) | [Data preprocessing](/concepts/machine-learning/data-preprocessing), [Data lineage](/concepts/data/data-lineage) |
| describe a reusable dataset | [Dataset card](/concepts/data/dataset-card) | [Metadata and provenance](/concepts/data/metadata-provenance), [Data versioning](/concepts/data/data-versioning) |
| build a benchmark | [Benchmark intake](/concepts/data/benchmark-intake) | [Benchmark claim contract](/concepts/evaluation/benchmark-claim-contract), [Evaluation set design](/concepts/evaluation/evaluation-set-design) |
| handle imperfect labels | [Label noise](/concepts/data/label-noise) | [Censored label](/concepts/data/censored-label), [Weak label](/concepts/data/weak-label) |
| handle distribution problems | [Dataset shift](/concepts/data/dataset-shift) | [Sampling bias](/concepts/data/sampling-bias), [Class imbalance](/concepts/data/class-imbalance) |

## 주제 묶음

| 그룹 | 노트 |
| --- | --- |
| Dataset definition | [Dataset](/entities/dataset), [Dataset construction checklist](/concepts/data/dataset-construction-checklist), [Data distribution](/concepts/data/data-distribution) |
| Schema and metadata | [Data schema](/concepts/data/data-schema), [Metadata and provenance](/concepts/data/metadata-provenance), [Data lineage](/concepts/data/data-lineage) |
| Curation and versions | [Data curation](/concepts/data/data-curation), [Data versioning](/concepts/data/data-versioning), [Annotation and labeling](/concepts/data/annotation-labeling) |
| Sampling | [Sampling strategy](/concepts/data/sampling-strategy), [Sampling bias](/concepts/data/sampling-bias), [Missing data](/concepts/data/missing-data) |
| Benchmarking | [Benchmark](/concepts/data/benchmark), [Benchmark intake](/concepts/data/benchmark-intake), [Evaluation protocol](/concepts/evaluation/evaluation-protocol) |

## 확인할 것

- 하나의 example은 무엇인가?
- split unit은 무엇인가?
- dataset은 어떤 distribution을 대표해야 하는가?
- field, unit, identifier, relationship을 정의하는 schema가 있는가?
- target label은 무엇에서 만들어지는가?
- label이 정확히 무엇을 의미하는가?
- missing, censored, weak label이 명시적으로 표현되는가?
- batch, source, protocol, time, species, target, structure context를 설명하는 metadata가 있는가?
- collection bias 때문에 빠진 example은 무엇인가?
- class 또는 label distribution이 imbalanced한가?
- sampling bias가 validation을 deployment보다 쉽게 만들지 않는가?
- train, validation, test, deployment 사이에 어떤 shift가 있는가?
- split이 의도한 generalization claim과 맞는가?
- 다른 사람이 dataset version과 filtering policy를 재구성할 수 있는가?
- label이 noisy, censored, inconsistent, protocol-dependent하지 않은가?

## 구성 경로

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
