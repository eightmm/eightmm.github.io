---
title: Agent Features
tags:
  - agents
  - features
---

# Agent Features

Agent 기능은 제품명이 아니라 사용자가 실제로 누르는 기능과 작업 단위로 정리합니다. 같은 기능이 ChatGPT, Claude, Gemini, Copilot 같은 여러 제품에서 서로 다른 이름으로 제공될 수 있기 때문입니다.

이 섹션의 기준은 단순합니다.

$$
\text{agent feature}
=
\text{model call}
+ \text{context}
+ \text{tool or workspace}
+ \text{verification boundary}
$$

모델이 한 번 답하는 기능은 [[concepts/llm/index|LLM concepts]]에 가깝고, 사용자가 파일, 웹, 코드, 브라우저, 프로젝트 지식, 외부 앱을 붙여 작업을 끝내는 기능은 [[agents/index|Agents]]에 둡니다.

## Feature Axes

제품 기능을 비교할 때는 이름보다 아래 축을 먼저 봅니다.

| Axis | Question |
| --- | --- |
| Context source | chat history, uploaded file, web, repo, connector, memory 중 무엇을 읽는가? |
| Action space | 답변만 하는가, 파일/앱/브라우저/코드를 바꾸는가? |
| Artifact | 대화 답변, 문서, 코드 diff, report, issue, PR, scheduled task 중 무엇이 남는가? |
| Persistence | 결과나 memory가 다음 세션에 남는가? |
| Verification | 무엇이 성공을 증명하는가? |
| Public boundary | private context가 public artifact로 넘어갈 위험이 있는가? |

이 축을 고정하면 제품명이 바뀌어도 기능을 안정적으로 비교할 수 있습니다.

## 기능 분류

| 기능 | 사용자가 기대하는 일 | 먼저 볼 노트 |
| --- | --- | --- |
| Chat and prompting | 질문, 초안, 설명, 변환 | [Chat and prompting](/agents/features/chat-and-prompting) |
| Files and knowledge | PDF, 문서, 노트, 프로젝트 지식 활용 | [Files and knowledge](/agents/features/files-and-knowledge) |
| Research browsing | 웹 검색, Deep Research, 출처 기반 보고서 | [Research browsing](/agents/features/research-browsing) |
| Coding workspace | 코드 읽기, 수정, 테스트, PR 보조 | [Coding workspace](/agents/features/coding-workspace) |
| Memory and projects | 장기 맥락, 프로젝트별 맥락, 사용자 선호 | [Memory and projects](/agents/features/memory-projects) |
| Artifacts and canvas | 문서, 앱, 코드, 시각화 결과물을 별도 작업면에서 편집 | [Artifacts and canvas](/agents/features/artifacts-canvas) |
| Connectors and actions | Drive, Slack, GitHub, 이메일 같은 외부 앱 연결 | [Connectors and actions](/agents/features/connectors-actions) |

## Read, Write, Persist

Agent feature risk는 대체로 아래 순서로 커집니다.

$$
\text{read}
<
\text{transform}
<
\text{write}
<
\text{persist}
<
\text{schedule}
$$

| Level | Examples | Main Check |
| --- | --- | --- |
| Read | file Q&A, web search, repo read | source relevance and freshness |
| Transform | summarize, rewrite, extract table | output matches source |
| Write | edit file, create issue, send message | preview, permission, rollback |
| Persist | memory, project knowledge, saved artifact | scope and stale context |
| Schedule | recurring task, reminder, monitor | duplicate prevention and stop condition |

## Choosing A Feature

| Need | Prefer |
| --- | --- |
| answer a conceptual question | chat and prompting |
| ground a claim in documents | files and knowledge or research browsing |
| produce a long editable artifact | artifacts and canvas |
| modify a repository | coding workspace |
| use private project context across sessions | memory and projects |
| act on external apps | connectors and actions |

## 작성 기준

- 기능의 이름보다 입력, 출력, 검증 방법을 먼저 설명합니다.
- 제품별 지원 여부는 변동이 크므로 구체 버전명보다 공식 문서 링크와 기능 패턴을 남깁니다.
- 개인 자동화, 개인 프롬프트, 개인 스킬은 여기 넣지 않습니다.
- 외부 계정, 비공개 문서, 사내 저장소, 연구실 서버 정보는 예시로 쓰지 않습니다.
- 외부 앱 action은 read/write/persist/schedule 중 어느 수준인지 표시합니다.
- 기능 소개 글은 제품 홍보보다 workflow, failure mode, verification boundary를 중심으로 씁니다.

## Related

- [[agents/index|Agents]]
- [[agents/core/agent-loop|Agent loop]]
- [[agents/core/context-engineering|Context engineering]]
- [[agents/tools/tool-use|Tool use]]
- [[agents/tools/tool-contract|Tool contract]]
- [[agents/verification/verification-loop|Verification loop]]
