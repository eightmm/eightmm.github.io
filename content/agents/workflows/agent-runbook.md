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

## Runbook section

- Purpose: 이 runbook이 다루는 workflow.
- Inputs: 필요한 file, prompt, ticket, paper, data.
- Preconditions: branch, environment, privacy boundary, approval need.
- Steps: stop condition이 있는 ordered action.
- Verification: command, review gate, evidence.
- Output: summary, changed file, open question, next action.
- Sanitization: 공개하면 안 되는 것.

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

## Related

- [[agents/workflows/paper-brief-workflow|Paper brief workflow]]
- [[agents/workflows/llm-wiki|LLM Wiki]]
- [[agents/workflows/content-promotion-workflow|Content promotion workflow]]
- [[agents/workflows/agent-handoff|Agent handoff]]
- [[agents/verification/verification-loop|Verification loop]]
- [[posts/blog-writing-guide|블로그 글 작성 가이드]]
