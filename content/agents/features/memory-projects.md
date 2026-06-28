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

where $M$ is stored memory, $g$ is the current goal, and $C_t$ is the current context.

## 구분

| 기능 | 저장되는 것 | 주의점 |
| --- | --- | --- |
| Chat history | 이전 대화 | 오래된 결정이 최신처럼 쓰일 수 있음 |
| User memory | 선호, 반복 정보 | 과잉 일반화 가능 |
| Project memory | 프로젝트 안의 대화와 파일 | 프로젝트 밖으로 새면 안 됨 |
| Knowledge base | 업로드 문서, Drive, repo, wiki | retrieval 근거가 필요함 |

## 좋은 사용처

- 반복되는 writing preference.
- 장기 프로젝트의 용어와 구조.
- 같은 코드베이스나 문서 묶음에 대한 반복 작업.
- 공개 지식베이스를 지속적으로 정리하는 workflow.

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
