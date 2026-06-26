---
title: 블로그 글 작성 가이드
tags:
  - posts
  - writing
  - workflow
---

# 블로그 글 작성 가이드

이 사이트의 글은 두 층으로 나눠서 씁니다.

- 바깥 글: 사람이 읽는 한글 post
- 안쪽 노트: 검색과 연결에 좋은 영어 wiki note

한글 글은 완결된 설명을 목표로 하지만, 모든 세부 개념을 글 안에 반복해서 넣지는 않습니다. 세부 개념은 [[concepts/index|Concepts]], 논문은 [[papers/index|Papers]], 연구 축은 [[research/index|Research]], 구현과 운영 경험은 [[projects/index|Projects]]와 [[infra/index|Infra]]로 넘깁니다.

AI, computational biology, Math가 함께 나오는 글은 [[posts/ai-molecular-math-post-intake|AI Computational Biology Math 포스트 인테이크]]로 먼저 중심축을 정합니다. 실제 post로 승격할지는 [[posts/post-promotion-gate|Post promotion gate]]로 확인합니다.

## 기본 구조

좋은 post는 아래 흐름을 따릅니다.

1. 문제의식: 왜 이 주제를 정리하는가
2. 큰 지도: 어떤 축으로 나눠서 볼 것인가
3. 핵심 개념: 독자가 알아야 할 최소 개념
4. 연결: 더 깊게 볼 wiki note
5. 판단 기준: 평가, 실패 모드, 주의점
6. 다음 글감: 이어서 정리할 질문

## 추천 템플릿

```markdown
---
title: 글 제목
date: YYYY-MM-DD
tags:
  - posts
---

# 글 제목

## 왜 이 주제인가

문제의식.

## 큰 그림

주제를 2-4개 축으로 나눠 설명.

## 핵심 개념

- `concepts/...`
- `papers/...`
- `research/...`

## 평가와 주의점

무엇을 조심해야 하는지.

## 다음 질문

이어질 글감.
```

## 글과 Wiki의 역할 분리

- Post: 관점, 맥락, 읽는 순서, 개인적인 해석
- Concept: 재사용 가능한 정의, 수식, 체크리스트
- Paper: 특정 논문의 주장과 한계
- Research: 장기 연구 질문과 도메인 지도
- Project: 내가 만든 것과 설계 결정
- Infra: 공개 가능한 운영 지식
- Agent: LLM/agent workflow와 검증 방식

## 승격 기준

Wiki note가 충분히 쌓였을 때만 post로 승격합니다. 자세한 흐름은 [[posts/wiki-to-post-workflow|Wiki에서 Post로 승격하는 방식]], [[posts/post-promotion-gate|Post promotion gate]], [[agents/workflows/content-promotion-workflow|Content promotion workflow]]를 따릅니다.

$$
\text{post-ready}
=
\text{reader question}
\land
\text{linked notes}
\land
\text{public-safe}
\land
\text{next path}
$$

## Checks

- 한글 글이 독립적으로 읽히는가?
- 세부 개념은 영어 wiki note로 빠져 있는가?
- 내부 정보, 미공개 결과, 서버 정보가 들어가지 않았는가?
- 읽는 사람이 다음 링크로 자연스럽게 이동할 수 있는가?
- 글 하나가 하나의 질문에 답하는가?

## Related

- [[posts/index|Posts]]
- [[posts/topic-roadmap|Topic roadmap]]
- [[posts/ai-molecular-math-post-intake|AI Computational Biology Math 포스트 인테이크]]
- [[posts/post-promotion-gate|Post promotion gate]]
- [[posts/synthesis-post-template|Synthesis post template]]
- [[posts/wiki-to-post-workflow|Wiki에서 Post로 승격하는 방식]]
- [[posts/2026-06-25-blog-and-wiki-workflow|블로그와 위키를 같이 쓰는 방식]]
- [[agents/workflows/content-promotion-workflow|Content promotion workflow]]
- [[agents/workflows/llm-wiki|LLM Wiki]]
- [[logs/sanitization-checklist|Sanitization checklist]]
