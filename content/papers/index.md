---
title: Papers
tags:
  - papers
---

# Papers

Paper note는 공개 논문을 요약하고, 논문 안의 claim을 재사용 가능한 concept로 다시 연결하는 공간입니다. 사이드바에서 보이는 트리는 workflow, checklist, template을 모두 쌓아둔 덤프가 아니라 선별된 reading shelf처럼 보여야 합니다.

매일 모이는 raw paper candidate는 먼저 [[inbox/index|Inbox]]에 두고, 실제로 읽고 남길 가치가 있는 논문만 여기로 승격합니다.

## 선별 영역

| 영역 | 용도 |
| --- | --- |
| [Architecture papers](/papers/architectures) | Transformer, GNN, SSM/Mamba, MoE, geometric architecture design |
| [LLM papers](/papers/llm) | language models, context, retrieval, instruction tuning, alignment, tool use |
| [Generative model papers](/papers/generative-models) | diffusion, flow, molecule/protein generation |
| [Computational Biology papers](/papers/computational-biology) | structure-based modeling, protein modeling, molecular generation, domain evaluation |

## Essential AI Reading Queue

논문 노트로 바로 만들기 전, 먼저 [[papers/essential-ai-reading|Essential AI reading queue]]에서 큰 축별 후보만 관리합니다. 이 목록은 세부 리뷰가 아니라 앞으로 채울 paper note의 후보 선반이며, 사이드바에는 직접 노출하지 않습니다.

## 현재 논문 노트

| 논문 노트 | 영역 |
| --- | --- |
| [PoseBusters](/papers/sbdd/posebusters) | structure-based evaluation |
| [Multi-scale ML for Antibody-Antigen Binding](/papers/protein-modeling/multi-scale-antibody-binding) | antibody-antigen binding |
| [MEET](/papers/protein-modeling/meet-equivariant-peptide) | equivariant peptide modeling |
| [Molexar](/papers/generative-models/molexar) | molecular generation |

## 보조 묶음

아래 묶음은 좁은 논문군의 stable URL을 유지하기 위한 곳입니다. 사이드바에서는 더 큰 선반을 먼저 보이게 둡니다.

| 묶음 | 상위 선반 |
| --- | --- |
| [Structure-based modeling papers](/papers/sbdd) | [Computational Biology papers](/papers/computational-biology) |
| [Protein modeling papers](/papers/protein-modeling) | [Computational Biology papers](/papers/computational-biology) |
| [Learning method papers](/papers/learning-methods) | [AI learning methods](/ai/learning-methods), [LLM papers](/papers/llm), or [Architecture papers](/papers/architectures) by strongest claim |
| [Systems papers](/papers/systems) | [AI systems](/ai/systems), [Infra](/infra), or [Agents](/agents) by strongest claim |

## 읽기 도구

아래 페이지들은 논문 묶음이 아니라 논문 읽기를 돕는 support material입니다. 이 페이지에서 직접 연결하되 `unlisted`로 두어 Explorer sidebar는 선별된 논문 영역과 실제 논문 노트에 집중하게 합니다.

| 도구 | 용도 |
| --- | --- |
| [Paper workflows](/papers/workflows) | triage, reading state, note format |
| [Paper analysis](/papers/analysis) | claims, benchmarks, ablations, limitations |
| [Paper reproducibility](/papers/reproducibility) | artifacts, readiness, plans, results |

## 함께 봐야 할 개념

| 맥락 | 링크 |
| --- | --- |
| Data and benchmark | [Benchmark](/concepts/data/benchmark) |
| Inputs and tasks | [Modalities](/concepts/modalities), [Tasks](/concepts/tasks), [Multimodal learning](/concepts/modalities/multimodal-learning) |
| Models and evidence | [Architectures](/concepts/architectures), [Evaluation](/concepts/evaluation), [Coverage matrix](/concepts/coverage-matrix) |
| Paper creation | [Longform paper review guide](/papers/workflows/longform-paper-review-guide), [AI-Molecular-Math paper template](/papers/workflows/ai-molecular-math-paper-template), [Paper review workflow](/papers/workflows/paper-review-workflow) |

## Related

| 영역 | 링크 |
| --- | --- |
| Research | [Research](/research) |
| SBDD | [Scoring function](/concepts/sbdd/scoring-function) |
