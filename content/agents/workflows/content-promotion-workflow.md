---
title: Content Promotion Workflow
tags:
  - agents
  - workflows
  - wiki
  - publishing
---

# Content Promotion Workflow

Content promotion은 raw note를 durable wiki page, 한국어 post, paper note, project note, public log로 바꾸는 workflow입니다. 이 과정이 있어야 사이트가 읽기 어려운 note dump나 재사용 가능한 지식이 없는 얕은 blog로 흐르지 않습니다.

유용한 promotion path는 아래와 같습니다.

$$
\text{raw input}
\rightarrow
\text{inbox}
\rightarrow
\text{canonical wiki note}
\rightarrow
\text{synthesis}
\rightarrow
\text{post or project note}
$$

## Destinations

- Inbox: unverified candidate, daily brief, follow-up queue.
- Concept: reusable definition, equation, checklist, evaluation rule.
- Paper: paper-specific claim, evidence, artifact, limitation, reproduction plan.
- Project: public artifact, design, verification, status, next work.
- Infra: public operational lesson 또는 systems pattern.
- Log: cleaned work record, incident note, experiment summary.
- Log promotion: cleaned record를 standalone log로 남기기 전에 [[logs/public-log-taxonomy|Public log taxonomy]]와 [[logs/log-promotion-rule|Log promotion rule]]을 사용합니다.
- Post: context, reading order, interpretation을 제공하는 한국어 narrative.

## Promotion Criteria

Page의 가장 강한 역할을 destination으로 삼습니다.

$$
D(n)
=
\arg\max_{d \in \mathcal{D}}
\mathrm{fit}(n,d)
$$

여기서 $n$은 candidate note, $\mathcal{D}$는 destination set, $\mathrm{fit}$은 note가 reusable, paper-specific, artifact-specific, operational, narrative, still unverified 중 어디에 맞는지 점수화합니다.

## Post를 쓸 때

한국어 post는 아래 조건을 만족할 때 씁니다.

- 이미 관련 wiki note가 여러 개 있습니다.
- 독자에게 또 하나의 isolated definition보다 map이나 point of view가 필요합니다.
- post가 답할 clear question이 있습니다.
- 개념을 복사하지 않고 concept, paper, project, infra note로 link할 수 있습니다.
- AI, Computational Biology, Math-heavy topic은 [[posts/ai-molecular-math-post-intake|AI Computational Biology Math post intake]]와 [[posts/wiki-bundle-checklist|Wiki bundle checklist]]를 통과합니다.
- Post candidate는 drafting 전에 [[posts/post-promotion-gate|Post promotion gate]]를 통과합니다.
- 여러 축을 가로지르는 draft는 [[posts/synthesis-post-template|Synthesis post template]]를 따릅니다.
- content는 [[inbox/publishing-gate|Publishing gate]]와 [[logs/sanitization-checklist|Sanitization checklist]]를 통과합니다.

## Agent 역할

Agent는 아래를 해야 합니다.

- raw input을 classify합니다.
- 약한 duplicate를 만들기보다 existing page update를 우선합니다.
- 작고 유용하며 link된 경우에만 stub을 만듭니다.
- concept note를 reusable하게 취급하기 전에 [[concepts/wiki-note-quality-gate|Wiki note quality gate]]를 실행합니다.
- paper candidate가 reusable definition, formula, contract, evidence boundary를 포함하면 [[papers/workflows/concept-update-contract|Concept update contract]]를 사용합니다.
- objective, metric, update rule, decision criteria를 명확하게 만드는 곳에는 formula를 추가합니다.
- 빠진 fact는 `to verify`로 표시합니다.
- commit 전에 link, privacy, build check를 실행합니다.

## 확인할 것

- destination이 note의 가장 강한 역할과 맞는가?
- page에 inbound link와 outbound link가 있는가?
- canonical explanation이 post에 중복되지 않고 wiki에 있는가?
- 한국어 post가 context와 reading order를 제공하는가?
- public promotion 전에 private detail이 제거되었는가?
- inbox에 남은 item에 explicit next action이 있는가?
- candidate가 log라면 taxonomy와 promotion destination이 명확한가?

## Related

- [[agents/workflows/llm-wiki|LLM Wiki]]
- [[concepts/wiki-note-quality-gate|Wiki note quality gate]]
- [[agents/workflows/paper-brief-workflow|Paper brief workflow]]
- [[papers/workflows/concept-update-contract|Concept update contract]]
- [[agents/workflows/agent-runbook|Agent runbook]]
- [[inbox/inbox-triage|Inbox triage]]
- [[inbox/curation-queue|Curation queue]]
- [[inbox/publishing-gate|Publishing gate]]
- [[posts/blog-writing-guide|Blog writing guide]]
- [[posts/ai-molecular-math-post-intake|AI Computational Biology Math post intake]]
- [[posts/wiki-bundle-checklist|Wiki bundle checklist]]
- [[posts/post-promotion-gate|Post promotion gate]]
- [[posts/synthesis-post-template|Synthesis post template]]
- [[posts/wiki-to-post-workflow|Wiki에서 post로 승격하는 방식]]
- [[projects/llm-wiki-blog|LLM Wiki blog]]
- [[logs/public-log-taxonomy|Public log taxonomy]]
- [[logs/log-promotion-rule|Log promotion rule]]
