---
title: Papers
tags:
  - papers
---

# Papers

Paper note는 공개 논문을 요약하고, 논문 안의 claim을 재사용 가능한 concept로 다시 연결하는 공간입니다. 이 영역은 workflow, checklist, template을 모두 쌓아둔 덤프가 아니라 선별된 reading shelf처럼 읽히는 것을 목표로 합니다.

매일 모이는 raw paper candidate는 먼저 [[inbox/index|Inbox]]에 두고, 실제로 읽고 남길 가치가 있는 논문만 여기로 승격합니다.

## 선별 영역

| 영역 | 용도 |
| --- | --- |
| [Architecture papers](/papers/architectures) | Transformer, GNN, SSM/Mamba, MoE, geometric architecture design |
| [LLM papers](/papers/llm) | language models, context, retrieval, instruction tuning, alignment, tool use |
| [Generative model papers](/papers/generative-models) | diffusion, flow, molecule/protein generation |
| [Computational Biology papers](/papers/computational-biology) | structure-based modeling, protein modeling, molecular generation, domain evaluation |

## Essential AI Reading Queue

논문 노트로 바로 만들기 전, 먼저 [[papers/essential-ai-reading|Essential AI reading queue]]에서 큰 축별 후보만 관리합니다. 이 목록은 세부 리뷰가 아니라 앞으로 채울 paper note의 후보 선반입니다.

## Paper Note Boundary

Paper note는 논문 하나의 주장, 방법, 실험, 한계를 읽기 위한 페이지입니다. 논문을 핑계로 전체 개념을 설명하려면 해당 설명은 [[concepts/index|Concepts]]로 분리합니다.

$$
\text{paper note}
=
\text{citation}
+ \text{claim}
+ \text{method}
+ \text{evidence}
+ \text{limits}
+ \text{links}
$$

| Field | Write |
| --- | --- |
| Citation | paper title, year, venue/arXiv if public |
| Claim | 논문이 실제로 주장하는 것 |
| Method | architecture, data, objective, evaluation setup |
| Evidence | table, benchmark, ablation, theorem, qualitative result |
| Limits | dataset, metric, baseline, leakage, scaling, reproducibility issue |
| Links | reusable concepts, related papers, possible projects |

Unsupported personal interpretation은 `open question`이나 `to verify`로 표시합니다.

## Promotion and Routing

Raw paper candidates should not all become paper notes. The promotion path is:

$$
\text{candidate}
\rightarrow
\text{triage}
\rightarrow
\text{paper note}
\rightarrow
\text{concept update or project}
$$

| If the paper mainly contributes | Route |
| --- | --- |
| architecture or inductive bias | [Architecture papers](/papers/architectures), then [[concepts/architectures/index|Architectures]] |
| learning objective or supervision setup | [Learning method papers](/papers/learning-methods), then [[ai/learning-methods|Learning methods]] |
| molecular/protein/structure modeling | [Computational Biology papers](/papers/computational-biology) |
| benchmark or evaluation protocol | [Paper analysis](/papers/analysis), [[ai/evaluation|Evaluation]] |
| reproducible implementation target | [Paper reproducibility](/papers/reproducibility), then [[projects/index|Projects]] |
| agent/tool/workflow behavior | [[agents/index|Agents]] and paper workflow notes |

## Reading Depth

Not every paper needs a long review.

| Depth | Use when | Output |
| --- | --- | --- |
| candidate | maybe relevant | inbox item or queue row |
| skim | useful but not central | short note with claim and route |
| full note | central to wiki topic | paper note with evidence and links |
| reproduction | implementation matters | readiness, plan, result pages |
| synthesis | multiple papers form a pattern | post or concept update |

## 현재 논문 노트

| 논문 노트 | 영역 |
| --- | --- |
| [Attention Is All You Need](/papers/architectures/attention-is-all-you-need) | Transformer architecture |
| [PoseBusters](/papers/sbdd/posebusters) | structure-based evaluation |
| [Multi-scale ML for Antibody-Antigen Binding](/papers/protein-modeling/multi-scale-antibody-binding) | antibody-antigen binding |
| [MEET](/papers/protein-modeling/meet-equivariant-peptide) | equivariant peptide modeling |
| [Molexar](/papers/generative-models/molexar) | molecular generation |

## Longform Posts

| Post | Anchor paper |
| --- | --- |
| [Attention Is All You Need를 지금 다시 읽는 법](/posts/attention-is-all-you-need-transformer-review) | [Attention Is All You Need](/papers/architectures/attention-is-all-you-need) |

## 보조 묶음

아래 묶음은 좁은 논문군을 잃지 않기 위한 보조 선반입니다. 처음 읽을 때는 더 큰 paper shelf에서 시작하고, 특정 분야로 좁힐 때 들어갑니다.

| 묶음 | 상위 선반 |
| --- | --- |
| [Structure-based modeling papers](/papers/sbdd) | [Computational Biology papers](/papers/computational-biology) |
| [Protein modeling papers](/papers/protein-modeling) | [Computational Biology papers](/papers/computational-biology) |
| [Learning method papers](/papers/learning-methods) | [AI learning methods](/ai/learning-methods), [LLM papers](/papers/llm), or [Architecture papers](/papers/architectures) by strongest claim |
| [Systems papers](/papers/systems) | [AI systems](/ai/systems), [Infra](/infra), or [Agents](/agents) by strongest claim |

## 읽기 도구

아래 페이지들은 논문 묶음이 아니라 논문 읽기를 돕는 support material입니다. 실제 논문 리뷰를 쓰거나 후보를 승격할 때 필요한 기준을 확인하는 데 사용합니다.

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
