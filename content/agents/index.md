---
title: Agents
tags:
  - agents
  - llm
  - workflows
---

# Agents

Agents는 LLM이 단순히 답변을 생성하는 것을 넘어, context를 모으고, 도구를 쓰고, 작업 상태를 추적하고, 결과를 검증하는 workflow입니다.

이 섹션은 일반 사용자가 실제로 접하는 agent 기능을 기준으로 정리합니다. 개인적으로 만든 skill이나 내부 자동화는 나중에 별도 문서로 분리하고, 여기서는 ChatGPT, Claude, Gemini, GitHub Copilot 같은 공개 제품에서 공통적으로 보이는 기능과 개념을 먼저 다룹니다.

기본 분류는 다음과 같습니다.

$$
\text{agent note}
\rightarrow
\{\text{core},\ \text{features},\ \text{tools},\ \text{workflows},\ \text{models},\ \text{verification}\}
$$

제품명보다 오래 남는 개념을 먼저 둡니다. 다만 유명 제품은 사용자가 기능을 이해하는 진입점이므로 [[agents/models/index|Agent model families]]에서 따로 비교합니다.

## 섹션

| 섹션 | 용도 |
| --- | --- |
| [Agent Core](/agents/core) | architecture, loop, state, memory, planning |
| [Agent Features](/agents/features) | chat, file, research, coding, memory, canvas, connector 기능 |
| [Agent Tools](/agents/tools) | tool contract, side effect, result handling |
| [Agent Workflows](/agents/workflows) | coding, paper brief, orchestration, handoff |
| [Agent Models](/agents/models) | ChatGPT, Claude, Gemini, GitHub Copilot |
| [Agent Verification](/agents/verification) | acceptance criteria, evidence, audit, evaluation |

## 분류 기준

| 노트가 다루는 것 | 둘 곳 |
| --- | --- |
| prompt, context, loop, state, memory, planning, harness | [Agent Core](/agents/core) |
| 사용자에게 보이는 product function | [Agent Features](/agents/features) |
| schema, side effect, permission, output | [Agent Tools](/agents/tools) |
| 반복 가능한 use case와 handoff | [Agent Workflows](/agents/workflows) |
| 유명 model/product family | [Agent Models](/agents/models) |
| acceptance criteria, evidence, audit, safety check | [Agent Verification](/agents/verification) |
| reader-facing story | [Posts](/posts) |

## 핵심 개념

| 그룹 | 노트 |
| --- | --- |
| Core model | [Agent architecture](/agents/core/agent-architecture), [Agent operating contract](/agents/core/agent-operating-contract), [Agent loop](/agents/core/agent-loop) |
| Environment and state | [Agent environment](/agents/core/agent-environment), [Action space](/agents/core/action-space), [Agent state](/agents/core/agent-state) |
| Prompt, context, memory | [Prompt engineering](/agents/core/prompt-engineering), [Context engineering](/agents/core/context-engineering), [Agent memory](/agents/core/agent-memory), [Memory boundary](/agents/core/memory-boundary) |
| Planning | [Planning](/agents/core/planning), [Task decomposition](/agents/core/task-decomposition) |
| Harness | [Harness engineering](/agents/core/harness-engineering), [Agent loop](/agents/core/agent-loop), [Tool contract](/agents/tools/tool-contract) |
| LLM interface | [Prompting](/concepts/llm/prompting), [Structured output](/concepts/llm/structured-output) |
| Tools | [Tool use](/agents/tools/tool-use), [Tool contract](/agents/tools/tool-contract), [Tool result handling](/agents/tools/tool-result-handling) |
| Verification | [Verification loop](/agents/verification/verification-loop), [Acceptance criteria](/agents/verification/acceptance-criteria), [Evidence ledger](/agents/verification/evidence-ledger) |
| Review and safety | [Completion audit](/agents/verification/completion-audit), [Reflection and critique](/agents/verification/reflection-and-critique), [Agent evaluation](/agents/verification/agent-evaluation), [Human in the loop](/agents/verification/human-in-the-loop), [Prompt injection](/agents/verification/prompt-injection) |

## 사용자-facing 기능

| 기능 | 용도 |
| --- | --- |
| [Chat and prompting](/agents/features/chat-and-prompting) | request design, explanation, drafting |
| [Files and knowledge](/agents/features/files-and-knowledge) | uploaded document, project knowledge, retrieval |
| [Research browsing](/agents/features/research-browsing) | search, deep research, citation-based report |
| [Coding workspace](/agents/features/coding-workspace) | codebase edit, test, PR assistance |
| [Memory and projects](/agents/features/memory-projects) | long-term context와 scoped project context |
| [Artifacts and canvas](/agents/features/artifacts-canvas) | editable document, app, code, visual output |
| [Connectors and actions](/agents/features/connectors-actions) | external app, API, browser/computer action |

## Workflows

| Workflow | 용도 |
| --- | --- |
| [Coding agents](/agents/workflows/coding-agents) | implementation과 repository work |
| [Paper brief workflow](/agents/workflows/paper-brief-workflow) | paper intake와 synthesis |
| [Agent orchestration](/agents/workflows/agent-orchestration) | 여러 agent 또는 tool 조율 |
| [Agent handoff](/agents/workflows/agent-handoff) | worker 사이의 state 전달 |
| [Agent runbook](/agents/workflows/agent-runbook) | 반복 가능한 operating procedure |
| [Multi-agent review](/agents/workflows/multi-agent-review) | independent verification |
| [LLM Wiki](/agents/workflows/llm-wiki) | knowledge-base maintenance |
| [Content promotion workflow](/agents/workflows/content-promotion-workflow) | inbox에서 note/post/project로 promotion |
| [Paper brief agent pipeline](/projects/paper-brief-agent-pipeline) | project-level paper workflow |

## Model Families

| 제품 | 주요 agent surface |
| --- | --- |
| [ChatGPT](/agents/models/chatgpt) | chat, search, deep research, projects, canvas, apps, agent tasks |
| [Claude](/agents/models/claude) | artifacts, projects, memory, tool use, computer use, Claude Code |
| [Gemini](/agents/models/gemini) | Google ecosystem, Deep Research, Canvas, Gems, multimodal work |
| [GitHub Copilot](/agents/models/github-copilot) | IDE completion, chat, code review, coding agent, PR workflow |

## Learning and Feedback

| 방법 | 링크 |
| --- | --- |
| Imitation | [Imitation learning](/concepts/learning/imitation-learning) |
| Reinforcement | [Reinforcement learning](/concepts/learning/reinforcement-learning) |
| Reward modeling | [Reward modeling](/concepts/learning/reward-modeling) |
| Preference objective | [Preference optimization](/concepts/learning/preference-optimization) |

## 작성 규칙

- workflow는 generic example로 설명합니다.
- private custom skill보다 user-facing feature를 우선합니다.
- vendor-specific claim은 date나 scope를 붙입니다.
- private prompt, private repo detail, credential, internal task name, unreleased work를 공개하지 않습니다.
- broad productivity claim보다 concrete verification habit를 우선합니다.

## Related

| 영역 | 링크 |
| --- | --- |
| Knowledge workflow | [LLM Wiki](/agents/workflows/llm-wiki), [Inbox](/inbox) |
| LLM behavior | [Hallucination and grounding](/concepts/llm/hallucination-grounding), [Transformer](/concepts/architectures/transformer) |
| Public work records | [Projects](/projects), [Public logs](/logs) |
