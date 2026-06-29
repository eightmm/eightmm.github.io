---
title: Agent Operating Contract
tags:
  - agents
  - workflows
  - verification
---

# Agent Operating Contract

Agent operating contract는 agent가 무엇을 할 수 있는지, 어떤 evidence를 inspect해야 하는지, 언제 멈춰야 하는지, completion을 어떻게 verify하는지 정의합니다. Open-ended instruction을 bounded workflow로 바꾸는 역할을 합니다.

간단한 contract는 아래처럼 쓸 수 있습니다.

$$
C =
(\mathcal{G}, \mathcal{S}, \mathcal{A}, \mathcal{V}, \mathcal{B})
$$

여기서 $\mathcal{G}$는 goal, $\mathcal{S}$는 observable state, $\mathcal{A}$는 allowed action space, $\mathcal{V}$는 verification set, $\mathcal{B}$는 boundary 또는 stop condition입니다.

## Contract 구성

- Goal: agent가 만들 artifact 또는 decision.
- State: agent가 inspect해야 하는 file, log, rendered page, issue, run output, external evidence.
- Action space: agent가 수행할 수 있는 tool과 edit.
- Verification: success를 보고하기 전에 필요한 command, review, check, screenshot, acceptance criteria.
- Boundaries: private information, destructive action, high-risk change, human approval이 필요한 경우.

## Completion rule

Completion은 evidence-based여야 합니다.

$$
\operatorname{done}(g)
=
\bigwedge_{v\in\mathcal{V}(g)}
\operatorname{pass}(v)
$$

필수 check가 skipped, unavailable, 또는 goal에 비해 너무 좁다면 agent는 completion을 주장하지 말고 gap을 보고해야 합니다.

넓은 goal에서는 마지막 단계가 [[agents/verification/evidence-ledger|Evidence ledger]]로 뒷받침되는 [[agents/verification/completion-audit|Completion audit]]이어야 합니다.

## Public Wiki에서의 사용

Public research blog 또는 LLM Wiki에서는 operating contract에 아래가 포함되어야 합니다.

- private server detail, credential, private path, collaborator detail, unpublished result 금지.
- source-grounded paper metadata.
- wikilink integrity.
- build verification.
- draft, inbox, curated note, published post의 명확한 구분.

## 확인할 것

- goal이 verify할 만큼 구체적인가?
- side effect가 허용되고 reversible한가?
- private boundary가 명시적인가?
- verification set이 blast radius와 맞는가?
- final report가 model confidence가 아니라 evidence에 묶여 있는가?
- completion이 좁힌 subtask가 아니라 original objective 기준으로 audit되는가?

## Related

- [[agents/core/agent-loop|Agent loop]]
- [[agents/core/action-space|Action space]]
- [[agents/core/planning|Planning]]
- [[agents/verification/acceptance-criteria|Acceptance criteria]]
- [[agents/verification/verification-loop|Verification loop]]
- [[agents/verification/evidence-ledger|Evidence ledger]]
- [[agents/verification/completion-audit|Completion audit]]
- [[agents/workflows/agent-runbook|Agent runbook]]
- [[agents/workflows/llm-wiki|LLM Wiki]]
