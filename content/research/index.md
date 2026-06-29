---
title: Research
tags:
  - research
---

# Research

실제 연구 질문으로 이어질 수 있는 아이디어, 가설, 방향성, synthesis note를 모아두는 입구입니다. 방법론 자체는 [[ai/index|AI]], [[molecular-modeling/index|Computational Biology]], [[math/index|Math]], [[concepts/index|Concepts]]에 두고, 이곳은 "무엇을 연구해볼 것인가"가 명확할 때 채웁니다.

미공개 실험, 내부 과제 정보, 협업 세부 내용은 공개하지 않고, 공개 가능한 질문과 방법론만 남깁니다.

## Projects와의 차이

| Area | Use When |
| --- | --- |
| Research | 질문, 가설, 아이디어, 비교 관점, 실험 설계가 중심일 때 |
| Projects | 구현물, pipeline, tool, public artifact, 운영 workflow가 중심일 때 |

예를 들어 "structure-based generative model에서 pose validity를 어떻게 평가할까?"는 Research입니다. "pose evaluation report generator를 만든다"는 Projects입니다.

## 현재 위치

| Need | Route |
| --- | --- |
| 구조기반 모델링 기본기 | [Structure-based modeling](/molecular-modeling/structure-based) |
| 단백질 모델링 기본기 | [Protein modeling](/molecular-modeling/protein-modeling) |
| 분자/리간드/컨포머 | [Molecules](/molecular-modeling/molecules) |
| 논문 읽기 | [Papers](/papers), [Paper review workflow](/papers/workflows/paper-review-workflow) |
| 공개 가능한 실험 기록 형식 | [Experiment ledger](/concepts/research-methodology/experiment-ledger), [Run record](/infra/reproducibility/run-record) |

## Research Areas

| Area | 질문 |
| --- | --- |
| [Computational Biology Research](/research/computational-biology) | molecule, protein, pocket, structure-based modeling에서 무엇을 검증할 것인가 |
| [Architecture Research](/research/architectures) | architecture와 inductive bias가 어떤 task에서 실제로 유리한가 |

## Seed Questions

| Question | Area |
| --- | --- |
| [Pose-aware molecular generation](/research/computational-biology/pose-aware-molecular-generation) | [Computational Biology Research](/research/computational-biology) |
| [Pocket-ligand representation alignment](/research/computational-biology/pocket-ligand-representation-alignment) | [Computational Biology Research](/research/computational-biology) |
| [Geometric inductive bias](/research/architectures/geometric-inductive-bias) | [Architecture Research](/research/architectures) |

## Research Note 조건

| Field | Requirement |
| --- | --- |
| Question | 공개 가능한 연구 질문이 명확해야 합니다. |
| Scope | 어떤 object, task, data, metric을 다루는지 제한해야 합니다. |
| Evidence | paper, public benchmark, 공개 artifact, 또는 공개 가능한 실험 설계로 뒷받침해야 합니다. |
| Boundary | 내부 과제명, 협업 세부 정보, 미공개 결과, private dataset을 포함하지 않아야 합니다. |

## 좋은 Research Note 형태

| Section | Write |
| --- | --- |
| Motivation | 왜 이 질문이 중요한가 |
| Question | 한 문장으로 된 연구 질문 |
| Hypothesis | 틀릴 수 있는 가설 |
| Method axis | 어떤 AI/CompBio/Math 방법과 연결되는가 |
| Evidence plan | 어떤 공개 데이터, benchmark, paper로 확인할 수 있는가 |
| Project handoff | 구현으로 넘어가면 어떤 project artifact가 필요한가 |

## 연결되는 방법론

| Need | Links |
| --- | --- |
| Question and hypothesis | [Research question](/concepts/research-methodology/research-question), [Hypothesis](/concepts/research-methodology/hypothesis) |
| Experiment design | [Experiment design](/concepts/research-methodology/experiment-design), [Minimum viable experiment](/concepts/research-methodology/minimum-viable-experiment) |
| Interpretation | [Result interpretation](/concepts/research-methodology/result-interpretation), [Threat to validity](/concepts/research-methodology/threat-to-validity) |
| Research records | [Experiment ledger](/concepts/research-methodology/experiment-ledger), [Decision record](/concepts/research-methodology/decision-record) |
| Modeling methods | [Learning methods](/concepts/learning), [Architectures](/concepts/architectures), [Generative models](/ai/generative-models) |
| Geometry | [Geometric deep learning](/concepts/geometric-deep-learning) |

## 관련 입구

- [[molecular-modeling/index|Computational Biology]]
- [[entities/index|Entities]]
- [[papers/index|Papers]]
- [[concepts/index|Concepts]]
- [[projects/index|Projects]]
