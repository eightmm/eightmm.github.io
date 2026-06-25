---
title: 블로그와 위키를 같이 쓰는 방식
date: 2026-06-25
tags:
  - posts
  - llm-wiki
  - workflow
---

# 블로그와 위키를 같이 쓰는 방식

이 사이트는 두 가지 역할을 동시에 가지도록 정리하고 있습니다.

첫 번째는 사람이 읽는 공개 블로그입니다. 여기에는 생각의 흐름, 분야를 바라보는 관점, 연구 주제의 큰 지도를 한글로 씁니다.

두 번째는 LLM과 사람이 함께 쓰는 wiki입니다. 여기에는 개념, 논문, 연구 주제, 인프라 운영 지식을 짧고 연결 가능한 영어 노트로 남깁니다.

## 왜 나누는가

블로그 글은 읽기 좋지만, 시간이 지나면 세부 개념을 다시 찾기 어렵습니다. 반대로 wiki note는 검색과 재사용에 좋지만 처음 읽는 사람에게는 맥락이 부족합니다.

그래서 역할을 나눕니다.

- [[posts/index|Posts]]: 한글 글, 생각의 흐름, 입구 역할
- [[ai/index|AI]]: ML, architecture, learning, generation, evaluation의 큰 지도
- [[bio-ai/index|Bio-AI]]: structure-based AI, protein, molecule, ligand 중심의 지도
- [[research/index|Research]]: 실제 연구 질문으로 이어지는 영역
- [[papers/index|Papers]]: 검증된 paper note와 reading workflow
- [[infra/index|Infra]]: 공개 가능한 HPC, GPU, server operation 지식
- [[agents/index|Agents]]: agent workflow, verification, LLM Wiki 운영 방식

## 읽는 경로

처음 보는 사람에게는 아래 경로가 가장 자연스럽습니다.

1. [[ai/index|AI]]에서 전체 방법론을 봅니다.
2. [[bio-ai/index|Bio-AI]]에서 적용 대상을 봅니다.
3. [[research/index|Research]]에서 연구 축을 좁힙니다.
4. [[papers/index|Papers]]에서 관련 논문을 읽습니다.
5. [[concepts/index|Concepts]]에서 반복되는 개념을 확인합니다.

예를 들어 구조 기반 AI를 읽는다면:

1. [[bio-ai/index|Bio-AI]]
2. [[research/structure-based-ai/index|Structure-based AI]]
3. [[research/structure-based-ai/protein-ligand-docking|Protein-ligand docking]]
4. [[concepts/sbdd/index|Structure-based drug discovery]]
5. [[papers/sbdd/posebusters|PoseBusters]]

## 작성 원칙

공개 블로그라서 내부 시스템 정보나 미공개 결과는 쓰지 않습니다. 대신 공개 가능한 수준으로 일반화된 문제, 개념, 검증 방법, 실패 패턴을 남깁니다.

좋은 노트는 하나의 질문에 답합니다.

- 이 개념은 무엇인가?
- 어떤 수식이나 구조로 이해할 수 있는가?
- 어디에 쓰이는가?
- 어떤 평가나 실패 모드가 있는가?
- 어떤 다른 노트와 연결되는가?

## Agent와 함께 쓰는 방식

Agent는 초안을 만들고, 링크를 채우고, 빌드와 깨진 링크를 검사하는 데 유용합니다. 하지만 공개되는 지식의 최종 책임은 사람에게 있습니다.

그래서 agent 작업은 [[agents/verification-loop|Verification loop]], [[agents/human-in-the-loop|Human in the loop]], [[agents/context-engineering|Context engineering]] 같은 원칙을 따릅니다.

이 구조가 잘 유지되면 블로그는 단순 기록장이 아니라 연구 지도, 논문 읽기 시스템, HPC 운영 노트, agent workflow 실험장이 됩니다.

## Related

- [[agents/llm-wiki|LLM Wiki]]
- [[projects/llm-wiki-blog|LLM Wiki blog]]
- [[papers/paper-review-workflow|Paper review workflow]]
- [[infra/reproducible-run-record|Reproducible run record]]
