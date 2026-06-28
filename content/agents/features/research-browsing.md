---
title: Research Browsing
tags:
  - agents
  - research
  - web
---

# Research Browsing

Research browsing 기능은 웹 검색, 문서 탐색, 출처 수집, 보고서 생성을 한 workflow로 묶습니다. ChatGPT Deep Research, Gemini Deep Research, Claude web search 같은 기능은 제품명은 달라도 같은 구조를 가집니다.

$$
q
\rightarrow
\text{plan}
\rightarrow
\{s_1,\dots,s_n\}
\rightarrow
\text{synthesis}
\rightarrow
\text{report}
$$

## 필요한 단계

| 단계 | 확인할 것 |
| --- | --- |
| Planning | 질문을 하위 질문으로 나눴는가 |
| Search | 최신성, 공식성, 1차 출처 여부 |
| Reading | source가 실제 claim을 지지하는가 |
| Synthesis | 서로 충돌하는 근거를 구분했는가 |
| Citation | 핵심 주장에 출처가 붙었는가 |

## 좋은 사용처

- 빠르게 변하는 제품 기능 조사.
- 여러 공식 문서의 차이 비교.
- 논문 후보, benchmark, tool landscape 조사.
- 공개 블로그 글을 쓰기 전 배경 조사.

## 주의점

Deep Research 결과도 최종 사실이 아니라 조사 초안입니다. 특히 가격, 모델 이름, plan 제한, benchmark 수치는 자주 바뀌므로 날짜와 출처를 같이 남겨야 합니다.

## Official References

- [ChatGPT Deep Research](https://help.openai.com/en/articles/10500283-deep-research-in-chatgpt)
- [Gemini Deep Research](https://support.google.com/gemini/answer/15719111)
- [Claude web search tool](https://docs.anthropic.com/en/docs/build-with-claude/tool-use/web-search-tool)

## Related

- [[agents/verification/evidence-ledger|Evidence ledger]]
- [[agents/verification/verification-loop|Verification loop]]
- [[concepts/llm/citation-grounding|Citation grounding]]
- [[papers/workflows/paper-triage|Paper triage]]
