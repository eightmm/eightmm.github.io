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

## 작성 기준

- 기능의 이름보다 입력, 출력, 검증 방법을 먼저 설명합니다.
- 제품별 지원 여부는 변동이 크므로 구체 버전명보다 공식 문서 링크와 기능 패턴을 남깁니다.
- 개인 자동화, 개인 프롬프트, 개인 스킬은 여기 넣지 않습니다.
- 외부 계정, 비공개 문서, 사내 저장소, 연구실 서버 정보는 예시로 쓰지 않습니다.

## Related

- [[agents/index|Agents]]
- [[agents/core/agent-loop|Agent loop]]
- [[agents/core/context-engineering|Context engineering]]
- [[agents/tools/tool-use|Tool use]]
- [[agents/verification/verification-loop|Verification loop]]
