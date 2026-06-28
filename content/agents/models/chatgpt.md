---
title: ChatGPT
tags:
  - agents
  - models
  - chatgpt
---

# ChatGPT

ChatGPT는 general-purpose assistant에서 agent workspace로 확장된 제품군입니다. 단순 질의응답뿐 아니라 search, deep research, projects, memory, canvas, apps/connectors, agent task 같은 기능을 통해 여러 단계 작업을 수행합니다.

## 기능 축

| 기능 | Agent 관점 |
| --- | --- |
| Chat | prompt와 context를 받아 응답 생성 |
| Search | 최신 정보와 출처를 context로 추가 |
| Deep Research | 계획, 웹 조사, synthesis, report 생성 |
| Projects | 대화와 파일을 project context로 묶음 |
| Memory | 사용자 선호와 반복 정보를 재사용 |
| Canvas | writing/coding 산출물을 별도 작업면에서 수정 |
| Apps/connectors | 외부 데이터와 앱을 context 또는 action으로 연결 |
| ChatGPT agent | browser, files, apps를 사용해 복합 작업 수행 |

## 잘 맞는 작업

- 공개 자료 기반 조사와 요약.
- 문서 초안과 반복 수정.
- 파일 기반 Q&A.
- 일반 코딩 보조와 작은 artifact 작성.
- 여러 tool을 쓰는 온라인 작업.

## 확인할 것

- 출처가 필요한 답변이면 citation이 claim을 실제로 지지하는지 확인합니다.
- project memory와 일반 memory의 경계를 확인합니다.
- connector나 agent action은 읽기/쓰기 권한을 구분합니다.

## Official References

- [ChatGPT release notes](https://help.openai.com/en/articles/6825453-chatgpt-release-notes)
- [ChatGPT Projects](https://help.openai.com/en/articles/10169521-projects-in-chatgpt)
- [ChatGPT Canvas](https://help.openai.com/en/articles/9930697-what-is-the-canvas-feature-in-chatgpt)
- [ChatGPT Deep Research](https://help.openai.com/en/articles/10500283-deep-research-in-chatgpt)
- [ChatGPT agent](https://help.openai.com/en/articles/11752874-chatgpt-agent)

## Related

- [[agents/features/research-browsing|Research browsing]]
- [[agents/features/memory-projects|Memory and projects]]
- [[agents/features/connectors-actions|Connectors and actions]]
- [[agents/models/index|Agent model families]]
