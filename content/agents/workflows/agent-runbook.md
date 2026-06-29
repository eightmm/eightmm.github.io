---
title: Agent Runbook
tags:
  - agents
  - workflows
  - runbook
---

# Agent Runbook

Agent runbook은 반복되는 agent workflow를 위한 재사용 가능한 operating procedure입니다. 모호하게 반복되던 작업을 input, step, check, public-output rule이 있는 checklist로 바꿉니다.

Runbook은 아래처럼 볼 수 있습니다.

$$
R
=
(\mathcal{I}, \mathcal{S}, \mathcal{V}, \mathcal{O})
$$

여기서 $\mathcal{I}$는 input contract, $\mathcal{S}$는 ordered step, $\mathcal{V}$는 verification, $\mathcal{O}$는 output contract입니다.

Runbook은 agent를 자동화하기 위한 prompt 모음이 아닙니다. 반복 작업에서 매번 빠지는 decision, verification, sanitization을 procedure로 고정하는 문서입니다.

## Runbook section

- Purpose: 이 runbook이 다루는 workflow.
- Inputs: 필요한 file, prompt, ticket, paper, data.
- Preconditions: branch, environment, privacy boundary, approval need.
- Steps: stop condition이 있는 ordered action.
- Verification: command, review gate, evidence.
- Output: summary, changed file, open question, next action.
- Sanitization: 공개하면 안 되는 것.

## Runbook Quality

| 기준 | 설명 |
| --- | --- |
| Reproducible | 다른 agent가 같은 input으로 따라 할 수 있음 |
| Bounded | scope와 stop condition이 있음 |
| Evidence-led | 각 step이 check나 artifact를 남김 |
| Public-safe | private detail 제거 기준이 있음 |
| Maintainable | 제품 기능 변화와 개인 환경 의존을 분리함 |

## 유용한 runbook

- daily paper brief ingestion.
- blog/wiki note curation.
- raw input에서 wiki note, post, project note, public log로 가는 content promotion.
- coding agent patch admission.
- multi-agent review.
- HPC failure summary.
- public incident 또는 experiment note cleanup.

## 확인할 것

- runbook이 반복되는 ambiguity를 줄이는가?
- edit 이후 verification step이 포함되어 있는가?
- 공개하면 안 되는 것을 명시하는가?
- 다른 agent가 private context 없이 따라갈 수 있는가?
- one-off chat output이 아니라 durable wiki improvement를 만드는가?
- 실패했을 때 recovery 또는 handoff path가 있는가?

## Related

- [[agents/workflows/paper-brief-workflow|Paper brief workflow]]
- [[agents/workflows/llm-wiki|LLM Wiki]]
- [[agents/workflows/content-promotion-workflow|Content promotion workflow]]
- [[agents/workflows/agent-handoff|Agent handoff]]
- [[agents/verification/verification-loop|Verification loop]]
- [[posts/blog-writing-guide|블로그 글 작성 가이드]]
