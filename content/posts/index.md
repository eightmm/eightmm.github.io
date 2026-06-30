---
title: Posts
tags:
  - posts
---

# Posts

연구, AI, computational biology, infra, agent workflow를 한글 글로 풀어 쓰는 공간입니다. Wiki note가 개념을 짧게 고정한다면, post는 여러 개념을 묶어 "왜 이 순서로 읽어야 하는가"를 설명합니다.

Post는 public entry point입니다. 개념 정의를 모두 post 안에 가두지 않고, 깊은 정의는 wiki note로 보내고 post는 읽는 순서와 판단 흐름을 제공합니다.

$$
\text{post}
=
\text{reader problem}
+ \text{map}
+ \text{examples}
+ \text{links}
+ \text{next route}
$$

## Post vs Wiki Note

| If the page should | Put it in |
| --- | --- |
| define a reusable concept | [[concepts/index|Concepts]] |
| explain one paper's claim | [[papers/index|Papers]] |
| document a maintained artifact | [[projects/index|Projects]] |
| explore a public research question | [[research/index|Research]] |
| guide a reader through several notes | Posts |

Post는 한글로 읽기 쉽게 쓰되, 핵심 용어와 하위 wiki note는 영어 제목을 유지해도 됩니다. 중요한 것은 언어 통일보다 reader route가 명확한가입니다.

## Post Promotion Rule

Wiki note 묶음이 충분히 쌓였을 때 post로 승격합니다.

| Gate | Check |
| --- | --- |
| Anchor notes | 핵심 개념 3개 이상이 wiki note로 존재하는가? |
| Reader problem | 왜 지금 읽어야 하는지 한 문장으로 말할 수 있는가? |
| Evidence | paper, concept, project, infra note로 claim을 연결할 수 있는가? |
| Public safety | private path, account, server, unpublished result가 없는가? |
| Next route | 독자가 다음에 볼 wiki note나 project가 있는가? |

## Published Essays

| Post | Topic |
| --- | --- |
| [Attention Is All You Need를 지금 다시 읽는 법](/posts/attention-is-all-you-need-transformer-review) | Transformer paper longform review |
| [AI Wiki를 어떤 축으로 나눠 볼 것인가](/posts/2026-06-26-ai-wiki-map) | AI taxonomy and reading map |
| [블로그와 위키를 같이 쓰는 방식](/posts/2026-06-25-blog-and-wiki-workflow) | blog/wiki workflow |
| [구조 기반 모델링을 어떻게 정리할 것인가](/posts/2026-06-25-structure-based-ai-map) | structure-based modeling map |
| [Wiki에서 Post로 승격하는 방식](/posts/wiki-to-post-workflow) | note-to-post promotion |

## Writing System

| Page | Use |
| --- | --- |
| [AI Computational Biology Math 포스트 인테이크](/posts/ai-molecular-math-post-intake) | AI, computational biology, and math synthesis writing |
| [Wiki bundle checklist](/posts/wiki-bundle-checklist) | minimum support notes before writing a post |
| [Post promotion gate](/posts/post-promotion-gate) | post readiness checklist |
| [블로그 글 작성 가이드](/posts/blog-writing-guide) | style and structure |
| [Synthesis post template](/posts/synthesis-post-template) | reusable post skeleton |
| [글감 로드맵](/posts/topic-roadmap) | future essay map |
| [Content promotion workflow](/agents/workflows/content-promotion-workflow) | note-to-public workflow |

## Candidate Essays

| Topic | Anchor |
| --- | --- |
| Flow matching 직관 | [Flow matching](/concepts/generative-models/flow-matching) |
| Protein representation 읽는 법 | [Protein representation](/concepts/protein-modeling/protein-representation) |
| Pose quality와 binding affinity 분리 | [Pose quality](/concepts/sbdd/pose-quality), [Binding affinity](/concepts/sbdd/binding-affinity) |
| 공개 가능한 infra 운영 노하우 | [Infra](/infra) |
| Coding agents in research workflow | [Coding agents](/agents/workflows/coding-agents) |
| 논문 읽기 흐름 | [Papers](/papers) |

## 연결되는 영역

- [[ai/index|AI]]
- [[molecular-modeling/index|Computational Biology]]
- [[research/index|Research]]
- [[papers/index|Papers]]
- [[infra/index|Infra]]
- [[agents/index|Agents]]
