---
title: Chat and Prompting
tags:
  - agents
  - prompting
---

# Chat and Prompting

Chat은 가장 기본적인 LLM 인터페이스입니다. agent 관점에서는 사용자의 요청, 시스템 규칙, 과거 대화, 첨부 자료가 하나의 context로 합쳐지고, 모델은 다음 응답 또는 다음 행동을 선택합니다.

$$
y_t \sim p_\theta(y \mid I, C_t, x_t)
$$

where $I$ is instruction, $C_t$ is current context, and $x_t$ is the user request.

## 좋은 프롬프트의 형태

| 구성 요소 | 역할 |
| --- | --- |
| Goal | 무엇을 끝내야 하는지 |
| Context | 판단에 필요한 배경 |
| Constraints | 하지 말아야 할 것, 형식, 공개 범위 |
| Output | 원하는 산출물의 형태 |
| Verification | 답을 어떻게 확인할지 |

프롬프트는 길수록 좋은 것이 아니라, 다음 결정을 내리기에 충분해야 합니다. 같은 요청이라도 연구 요약, 코드 수정, 논문 비교, 공개 블로그 작성은 필요한 context가 다릅니다.

## Agent와의 차이

단순 chat은 응답을 생성하고 끝납니다. Agent workflow는 응답 이후에도 도구를 호출하거나, 파일을 수정하거나, 결과를 검증할 수 있습니다.

$$
\text{chat}:\ x \rightarrow y
$$

$$
\text{agent}:\ x \rightarrow a_1 \rightarrow o_1 \rightarrow a_2 \rightarrow o_2 \rightarrow y
$$

## Related

- [[concepts/llm/prompting|Prompting]]
- [[agents/core/prompt-engineering|Prompt engineering]]
- [[agents/core/context-engineering|Context engineering]]
- [[agents/core/agent-loop|Agent loop]]
