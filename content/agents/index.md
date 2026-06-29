---
title: Agents
tags:
  - agents
  - llm
  - workflows
---

# Agents

Agents는 LLM이 답변만 생성하는 것이 아니라, context를 모으고, tool을 쓰고, state를 추적하고, 결과를 검증해서 작업을 끝내는 방식입니다.

이 섹션은 제품 기능 목록보다 오래 남는 agent의 원리를 먼저 봅니다. 핵심은 model, context, tools, state, verification이 어떻게 묶여 하나의 작업 흐름을 만드는지입니다.

$$
\text{agent}
=
\text{model}
+ \text{context}
+ \text{tools}
+ \text{state}
+ \text{verification}
$$

## Agent Contract

Agent 글은 “LLM이 똑똑하다”가 아니라, 어떤 관찰을 보고 어떤 행동을 하며 어떤 증거로 완료를 판단하는지를 설명해야 합니다.

$$
A
=
(\pi_\theta,\ \mathcal{O},\ \mathcal{A},\ S,\ M,\ T,\ V)
$$

| Part | Meaning | Note location |
| --- | --- | --- |
| $\pi_\theta$ | model policy or reasoning surface | [[agents/models/index|Models]], [[ai/architectures|Architectures]] |
| $\mathcal{O}$ | observations and context | [[agents/core/context-engineering|Context engineering]] |
| $\mathcal{A}$ | allowed actions | [[agents/core/action-space|Action space]], [[agents/tools/index|Tools]] |
| $S$ | current task state | [[agents/core/agent-state|Agent state]] |
| $M$ | reusable memory or knowledge | [[agents/core/agent-memory|Agent memory]], [[agents/features/memory-projects|Memory and projects]] |
| $T$ | tool interface | [[agents/tools/tool-contract|Tool contract]] |
| $V$ | verifier or completion rule | [[agents/verification/index|Verification]] |

The practical question is:

$$
\text{given context and tools, what evidence lets the agent stop?}
$$

## 먼저 볼 지도

| 질문 | 볼 곳 |
| --- | --- |
| agent가 내부적으로 어떻게 움직이는가? | [Core](/agents/core) |
| 사용자가 실제로 누르는 기능은 무엇인가? | [Features](/agents/features) |
| tool call, connector, side effect는 어떻게 다루는가? | [Tools](/agents/tools) |
| coding, paper brief, LLM Wiki 같은 반복 작업은 어떻게 묶는가? | [Workflows](/agents/workflows) |
| ChatGPT, Claude, Gemini, Copilot은 어떤 agent surface를 갖는가? | [Models](/agents/models) |
| agent 결과를 어떻게 검증하고 완료 판단하는가? | [Verification](/agents/verification) |

## 섹션

| 섹션 | 담는 것 | 대표 노트 |
| --- | --- | --- |
| [Core](/agents/core) | architecture, loop, state, memory, planning | [Agent loop](/agents/core/agent-loop), [Context engineering](/agents/core/context-engineering) |
| [Features](/agents/features) | chat, files, research, coding, memory, canvas, connectors | [Files and knowledge](/agents/features/files-and-knowledge), [Coding workspace](/agents/features/coding-workspace) |
| [Tools](/agents/tools) | tool contract, permission, result handling | [Tool use](/agents/tools/tool-use), [Tool contract](/agents/tools/tool-contract) |
| [Workflows](/agents/workflows) | 반복 가능한 작업 절차 | [Coding agents](/agents/workflows/coding-agents), [LLM Wiki](/agents/workflows/llm-wiki) |
| [Models](/agents/models) | 공개 제품군의 agent interface | [ChatGPT](/agents/models/chatgpt), [Claude](/agents/models/claude) |
| [Verification](/agents/verification) | acceptance criteria, evidence, audit, safety | [Verification loop](/agents/verification/verification-loop), [Completion audit](/agents/verification/completion-audit) |

## 분류 기준

| 내용 | 둘 곳 |
| --- | --- |
| prompt, context, memory, planning처럼 agent를 이루는 부품 | [Core](/agents/core) |
| 사용자가 보는 기능 이름과 사용 패턴 | [Features](/agents/features) |
| 외부 상태를 읽거나 바꾸는 인터페이스 | [Tools](/agents/tools) |
| 여러 단계로 반복되는 작업 방식 | [Workflows](/agents/workflows) |
| 제품별 차이와 interface 비교 | [Models](/agents/models) |
| 결과가 맞는지 확인하는 기준과 절차 | [Verification](/agents/verification) |
| model architecture, loss, training method 자체 | [AI](/ai), [Architectures](/ai/architectures), [Learning methods](/ai/learning-methods) |
| 서버, repo, browser, job 같은 실행 환경 | [Infra](/infra), [AI Systems](/ai/systems) |

## Workflow View

Agent workflow는 다음 항목을 분리해서 써야 비교가 됩니다.

| Field | Write |
| --- | --- |
| Task | coding, paper curation, wiki maintenance, review, browsing, deployment |
| Inputs | user request, files, paper metadata, repo state, browser state, prior notes |
| Tools | read-only tools, write tools, browser, shell, GitHub, search, local agents |
| State | todo list, current branch, open runs, unresolved decisions |
| Side effects | file edit, commit, push, PR, issue, message, public page update |
| Verification | tests, build, rendered page, source citation, human review, completion audit |
| Public boundary | secrets, private paths, account names, unpublished work, collaborator details |

This keeps product usage notes from turning into vague impressions. A workflow note should say what changed, what evidence was checked, and what remains uncertain.

## Agent vs Automation vs Workflow

| Term | Meaning | Example note |
| --- | --- | --- |
| Agent core | generic mechanism of context, state, tools, and verification | [[agents/core/agent-loop|Agent loop]] |
| Product feature | user-facing capability exposed by a product | [[agents/features/files-and-knowledge|Files and knowledge]] |
| Tool | typed interface that reads or mutates external state | [[agents/tools/tool-use|Tool use]] |
| Workflow | repeatable procedure that uses an agent to finish a task | [[agents/workflows/coding-agents|Coding agents]] |
| Automation | scheduled or event-driven workflow with less human prompting | [[agents/workflows/agent-runbook|Agent runbook]] |
| Evaluation | evidence that an agent behavior meets a claim | [[agents/verification/agent-evaluation|Agent evaluation]] |

## Good Agent Notes

| Weak note | Better note |
| --- | --- |
| “This agent is good at coding.” | Define the coding task, allowed edits, tests, and review evidence. |
| “Tool use makes it powerful.” | State the tool contract, side effects, retry behavior, and verifier. |
| “It remembered context.” | Separate current context, durable memory, retrieved knowledge, and user-provided state. |
| “It finished the task.” | Link completion to acceptance criteria and evidence. |
| “Multi-agent review helped.” | Say what independent evidence or failure mode each reviewer checked. |

## 작성 규칙

- 개인 자동화보다 공개 제품에서 공통으로 보이는 pattern을 우선합니다.
- 세부 개념은 Core, Tools, Workflows, Verification 같은 안정적인 묶음 안에서 연결합니다.
- vendor-specific claim은 변동 가능성이 크므로 날짜, 범위, 공식 문서 확인 필요성을 남깁니다.
- private prompt, private repo detail, credential, internal task name, unpublished work는 공개하지 않습니다.
- “agent가 잘했다”보다 어떤 evidence로 확인했는지를 남깁니다.
- 제품 이름은 바뀔 수 있으므로 durable concept와 product surface를 분리합니다.
- side effect가 있는 workflow는 항상 verification과 rollback 또는 handoff 기준을 함께 씁니다.

## Related

- [[ai/index|AI]]
- [[agents/workflows/llm-wiki|LLM Wiki]]
- [[projects/paper-brief-agent-pipeline|Paper brief agent pipeline]]
- [[agents/verification/completion-audit|Completion audit]]
