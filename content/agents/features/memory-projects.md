---
title: Memory and Projects
tags:
  - agents
  - memory
  - projects
---

# Memory and Projects

Memory와 project 기능은 agent가 한 번의 대화 밖에서도 사용자의 선호, 작업 맥락, 업로드 지식, 프로젝트별 대화를 재사용하게 합니다. 편리하지만 stale context와 privacy 문제가 생기기 쉬운 기능입니다.

$$
m^\ast = \operatorname{retrieve}(M, g, C_t)
$$

여기서 $M$은 stored memory, $g$는 current goal, $C_t$는 current context입니다.

Memory는 context를 절약하지만, 잘못 쓰면 오래된 추정을 계속 강화합니다. Project 기능은 memory보다 boundary가 더 중요합니다. 같은 사용자의 다른 project라도 private context가 섞이면 안 됩니다.

## 구분

| 기능 | 저장되는 것 | 주의점 |
| --- | --- | --- |
| Chat history | 이전 대화 | 오래된 결정이 최신처럼 쓰일 수 있음 |
| User memory | 선호, 반복 정보 | 과잉 일반화 가능 |
| Project memory | 프로젝트 안의 대화와 파일 | 프로젝트 밖으로 새면 안 됨 |
| Knowledge base | 업로드 문서, Drive, repo, wiki | retrieval 근거가 필요함 |

## Memory Policy

| 항목 | 저장해도 좋은가? | 이유 |
| --- | --- | --- |
| 반복되는 writing preference | 대체로 좋음 | 낮은 위험, 반복 효율 |
| 공개 프로젝트 구조 | 조건부 | 최신성 확인 필요 |
| server IP, account, SSH port | 안 됨 | public/private boundary 위반 |
| unpublished experiment result | 안 됨 | 연구 claim과 협업 경계 위험 |
| temporary decision | 조심 | 시간이 지나면 stale context가 됨 |

## 좋은 사용처

- 반복되는 writing preference.
- 장기 프로젝트의 용어와 구조.
- 같은 코드베이스나 문서 묶음에 대한 반복 작업.
- 공개 지식베이스를 지속적으로 정리하는 workflow.

## 확인할 것

- memory가 current task와 직접 관련 있는가?
- project boundary 밖의 정보가 섞이지 않았는가?
- 오래된 memory가 current file이나 source보다 우선되고 있지 않은가?
- public artifact에 들어가기 전 private detail을 제거했는가?

## Official References

- [ChatGPT Projects](https://help.openai.com/en/articles/10169521-projects-in-chatgpt)
- [ChatGPT project-only memory](https://help.openai.com/en/articles/6825453-chatgpt-release-notes)
- [Claude memory](https://support.anthropic.com/en/articles/11817273-using-claude-s-chat-search-and-memory-to-build-on-previous-context)
- [Claude Projects RAG](https://support.anthropic.com/en/articles/11473015-retrieval-augmented-generation-rag-for-projects)

## Related

- [[agents/core/agent-memory|Agent memory]]
- [[agents/core/memory-boundary|Memory boundary]]
- [[agents/core/context-engineering|Context engineering]]
- [[agents/workflows/llm-wiki|LLM Wiki]]
