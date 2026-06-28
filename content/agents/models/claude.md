---
title: Claude
tags:
  - agents
  - models
  - claude
---

# Claude

Claude는 long-form reasoning, writing, artifacts, coding, tool use가 강한 agent product family입니다. 사용자-facing 기능과 개발자 tool-use API가 모두 agent workflow와 연결됩니다.

## 기능 축

| 기능 | Agent 관점 |
| --- | --- |
| Chat | 긴 지시와 문서 기반 응답 |
| Artifacts | 문서, 코드, 앱, 시각화 결과물을 별도 pane에서 유지 |
| Projects | project knowledge와 대화를 묶어 context 제공 |
| Memory | 이전 대화 기반 맥락 재사용 |
| Web search | 최신 웹 정보를 tool result로 추가 |
| Tool use | 개발자가 정의한 function/API 호출 |
| Computer use | screenshot, mouse, keyboard 기반 desktop/browser action |
| Claude Code | repo 읽기, 파일 수정, 명령 실행, 개발 workflow 보조 |

## 잘 맞는 작업

- 긴 글, proposal, report, spec 작성.
- artifact를 보면서 반복 수정하는 작업.
- 코드베이스 이해와 수정.
- tool contract가 명확한 API/workflow 자동화.

## 확인할 것

- artifact가 conversation과 분리되어도 최종 산출물 검증은 따로 해야 합니다.
- computer/browser use는 실제 외부 상태를 바꿀 수 있으므로 human gate가 필요합니다.
- project knowledge retrieval은 근거 문서를 확인해야 합니다.

## Official References

- [Claude Artifacts](https://support.anthropic.com/en/articles/9487310-what-are-artifacts-and-how-do-i-use-them)
- [Claude memory](https://support.anthropic.com/en/articles/11817273-using-claude-s-chat-search-and-memory-to-build-on-previous-context)
- [Claude Projects RAG](https://support.anthropic.com/en/articles/11473015-retrieval-augmented-generation-rag-for-projects)
- [Claude tool use](https://docs.anthropic.com/en/docs/build-with-claude/tool-use/overview)
- [Claude Code](https://docs.anthropic.com/en/docs/claude-code/overview)

## Related

- [[agents/features/artifacts-canvas|Artifacts and canvas]]
- [[agents/features/coding-workspace|Coding workspace]]
- [[agents/tools/tool-contract|Tool contract]]
- [[agents/models/index|Agent model families]]
