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

Research browsing은 검색 결과를 많이 모으는 기능이 아니라, 질문을 evidence-backed report로 바꾸는 workflow입니다. 좋은 browsing agent는 source discovery와 claim synthesis를 분리합니다.

## 필요한 단계

| 단계 | 확인할 것 |
| --- | --- |
| Planning | 질문을 하위 질문으로 나눴는가 |
| Search | 최신성, 공식성, 1차 출처 여부 |
| Reading | source가 실제 claim을 지지하는가 |
| Synthesis | 서로 충돌하는 근거를 구분했는가 |
| Citation | 핵심 주장에 출처가 붙었는가 |

## Source Priority

| 질문 | 우선 출처 |
| --- | --- |
| 제품 기능, 가격, 제한 | 공식 문서, release note |
| 논문 주장 | paper, arXiv, project page, official code |
| benchmark 수치 | 원 논문, benchmark leaderboard, evaluation script |
| 법/정책/규정 | 공식 기관 문서 |
| community signal | forum, issue, discussion, social source |

## 좋은 사용처

- 빠르게 변하는 제품 기능 조사.
- 여러 공식 문서의 차이 비교.
- 논문 후보, benchmark, tool landscape 조사.
- 공개 블로그 글을 쓰기 전 배경 조사.

## 주의점

Deep Research 결과도 최종 사실이 아니라 조사 초안입니다. 특히 가격, 모델 이름, plan 제한, benchmark 수치는 자주 바뀌므로 날짜와 출처를 같이 남겨야 합니다.

## Failure Mode

- search snippet을 source 내용처럼 취급합니다.
- 같은 보도자료를 여러 출처로 세어 consensus처럼 보입니다.
- publication date와 event date를 구분하지 않습니다.
- official limitation과 user workaround를 섞습니다.
- citation이 문장 전체가 아니라 일부 claim만 지지합니다.

## Output Contract

좋은 research browsing 결과는 claim, source, date, confidence, open question을 분리합니다. 공개 블로그로 승격할 때는 출처를 남기되 긴 인용문보다 요약과 해석을 중심으로 둡니다.

## Official References

- [ChatGPT Deep Research](https://help.openai.com/en/articles/10500283-deep-research-in-chatgpt)
- [Gemini Deep Research](https://support.google.com/gemini/answer/15719111)
- [Claude web search tool](https://docs.anthropic.com/en/docs/build-with-claude/tool-use/web-search-tool)

## Related

- [[agents/verification/evidence-ledger|Evidence ledger]]
- [[agents/verification/verification-loop|Verification loop]]
- [[concepts/llm/citation-grounding|Citation grounding]]
- [[papers/workflows/paper-triage|Paper triage]]
