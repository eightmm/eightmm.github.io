---
title: Wiki에서 Post로 승격하는 방식
tags:
  - posts
  - workflow
  - wiki
---

# Wiki에서 Post로 승격하는 방식

이 블로그는 글을 먼저 많이 쓰는 방식보다, 작은 wiki note를 쌓고 충분히 연결된 주제를 한글 post로 승격하는 방식이 맞습니다.

영어 wiki note는 오래 남는 정의, 수식, 체크리스트를 담습니다. 한글 post는 독자가 왜 이 주제를 읽어야 하는지, 어떤 순서로 보면 되는지, 어디서 조심해야 하는지를 설명합니다.

## 기본 흐름

$$
\text{idea}
\rightarrow
\text{wiki notes}
\rightarrow
\text{synthesis}
\rightarrow
\text{post}
$$

아이디어가 바로 post가 되지는 않습니다. 먼저 [[concepts/index|Concepts]], [[papers/index|Papers]], [[projects/index|Projects]], [[infra/index|Infra]], [[agents/index|Agents]] 중 어디에 들어갈지 정합니다.

AI, Bio-AI, Math가 섞인 주제는 [[posts/ai-bio-math-post-intake|AI-Bio-Math 포스트 인테이크]]로 중심축, 최소 수식, benchmark boundary를 먼저 정합니다.

## Post로 올릴 때

Post는 아래 조건을 만족할 때 씁니다.

- 연결된 wiki note가 여러 개 있다.
- 독자에게 큰 지도나 읽는 순서가 필요하다.
- 하나의 질문에 답할 수 있다.
- 세부 정의를 post 안에서 반복하지 않고 링크로 넘길 수 있다.
- 공개해도 되는 내용만 남아 있다.

예를 들어 [[concepts/architectures/transformer|Transformer]], [[concepts/architectures/state-space-model|State-space model]], [[concepts/architectures/gnn|GNN]] 노트가 각각 있을 때, post는 “sequence, graph, structure 입력에서 architecture를 어떻게 고를까?” 같은 질문에 답하는 글이 됩니다.

## Wiki Note로 남길 때

아래 경우는 post보다 wiki note가 낫습니다.

- 용어 하나의 정의가 중심이다.
- 수식, metric, update rule, checklist가 핵심이다.
- 여러 글에서 반복해서 링크할 내용이다.
- 아직 개인적인 관점보다 객관적인 정리가 먼저 필요하다.
- 논문 하나의 claim과 evidence를 다룬다.

이때는 [[agents/workflows/content-promotion-workflow|Content promotion workflow]]에 따라 destination을 정하고, 필요하면 [[inbox/curation-queue|Curation queue]]에 남깁니다.

## 좋은 Post의 역할

좋은 post는 wiki를 대체하지 않습니다. 대신 wiki를 읽을 이유와 순서를 줍니다.

- 문제의식: 왜 지금 이 주제를 봐야 하는가
- 지도: 어떤 축으로 나눠야 하는가
- 최소 개념: 독자가 막히지 않을 정도의 설명
- 링크: 더 정확한 정의와 수식은 wiki note로 이동
- 판단 기준: 평가, 실패 모드, leakage, reproducibility
- 다음 질문: 이어서 쓸 글감

## Checks

- 이 글이 하나의 질문에 답하는가?
- post 안의 긴 정의를 wiki note로 빼낼 수 있는가?
- 관련 wiki note가 충분히 연결되어 있는가?
- 민감한 내부 정보, 미공개 결과, 서버 정보가 빠졌는가?
- 읽은 사람이 다음 링크로 이동할 수 있는가?

## Related

- [[posts/blog-writing-guide|블로그 글 작성 가이드]]
- [[posts/topic-roadmap|글감 로드맵]]
- [[posts/ai-bio-math-post-intake|AI-Bio-Math 포스트 인테이크]]
- [[posts/2026-06-25-blog-and-wiki-workflow|블로그와 위키를 같이 쓰는 방식]]
- [[agents/workflows/content-promotion-workflow|Content promotion workflow]]
- [[agents/workflows/llm-wiki|LLM Wiki]]
- [[inbox/publishing-gate|Publishing gate]]
