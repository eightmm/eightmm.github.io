# BLOG_POST_PROTOCOL.md

이 문서는 `eightmm.github.io` 블로그의 **논문 리뷰 포스트 작성 프로토콜**이다.
앞으로 이 블로그의 논문 리뷰는 기본적으로 **깊은 해설형(long-form technical review)** 을 표준으로 삼는다.

즉 목표는 단순 요약이 아니라:

- 논문의 문제의식과 배경을 설명하고
- 핵심 수식과 모델링 선택을 풀어주고
- 아키텍처와 학습/추론을 구현 감각이 들 정도로 해설하고
- 실험 결과와 한계를 비판적으로 평가하는 것

한 줄로 말하면:

> **논문을 안 읽은 사람도 이 글 하나로 핵심, 수식, 설계 이유, 한계까지 따라올 수 있는 수준의 깊은 리뷰**

---

## 0. 기본 원칙

이 블로그의 paper review는 앞으로 다음 5가지를 반드시 지향한다.

1. **맥락**: 왜 이 문제가 중요한가
2. **핵심 주장**: 저자들이 무엇을 바꿨는가
3. **수학적 구조**: 어떤 공간/목적함수/확률모형 위에서 문제를 푸는가
4. **구현 감각**: 실제로 어떤 모듈이 어떤 역할을 하는가
5. **평가와 한계**: 결과가 얼마나 설득력 있는가, 어디까지 믿어야 하는가

금지:
- 초록 번역체
- 홍보문
- 수식만 많은 설명 부재 글
- “좋다/강하다”만 반복하는 얕은 감상문

권장:
- 직관 + 수식 + 구조 + 비판이 함께 있는 글

---

## 1. 이제부터의 기본 스타일: “더 깊은 버전”

이 블로그의 논문 리뷰는 앞으로 **읽기 쉬운 요약형**보다 **깊은 해설형**을 기본값으로 한다.

### 의미하는 바

- `How It Works`는 충분히 길어야 한다
- 핵심 수식은 가능하면 원 논문의 notation과 연결해서 소개한다
- theorem / proposition / lemma가 중요한 논문이면 반드시 해설한다
- 모델의 forward / backward process, objective, inference를 구분해서 설명한다
- 아키텍처는 "모듈 이름 나열"이 아니라 **입력 → 표현 → 업데이트 → 출력** 흐름으로 설명한다
- 결과는 숫자만 적지 말고 **왜 그 수치가 중요한지** 해석한다

### 목표 글감

독자가 글을 읽고 다음 중 최소 3개는 할 수 있어야 한다.

- 논문 아이디어를 동료에게 설명
- 핵심 수식을 화이트보드에 다시 적기
- 모델 구현 흐름을 대략 재구성
- baseline 대비 차이를 명확히 말하기
- 논문의 한계를 구체적으로 지적

---

## 2. 기본 길이/밀도 기준

### 길이
- 기본 목표: **긴 글**
- 논문 리뷰는 가능하면 short note가 아니라 **세미나 노트 수준**으로 작성

### 밀도 기준
- `How It Works` 비중: **최소 40% 이상**, 가능하면 **45~55%**
- 수식: 논문 특성상 중요하면 **최소 3개 이상**, 보통 **4~8개** 권장
- 코드 블록: **최소 1개**, 가능하면 **2개 이상**
- 표: **최소 1개 이상**
- figure: **대표 figure 1개 이상**, 가능하면 results/ablation figure도 추가

### 가독성 기준
- inline math는 **짧고 안전한 표기만** 사용한다
- 아래 경우는 inline math 대신 **별도 블록 수식**으로 뺀다:
  - 첨자/위첨자가 많은 식
  - `\hat`, `\tilde`, `\mathcal`, `\nabla`, `\otimes`, 조건부 확률 표기 등이 섞인 식
  - 문장 중간에 넣으면 흐름이 끊기는 식
- 리스트 bullet 안에서는 복잡한 수식을 피한다
- 변수 정의가 여러 개 나오면 문장으로 줄줄 쓰지 말고, 짧은 bullet 또는 블록으로 정리한다
- 한 문단 안에 수식이 2개 이상 섞이면 문단을 쪼갠다
- heading / bullet / 본문 사이에 공백을 충분히 둬서 들여쓰기 레벨이 시각적으로 안정되게 보이도록 한다

### 우선순위
1. 설명의 정확성
2. 설계 이유의 해설
3. 구현 감각
4. 읽기 흐름
5. 장식적 표현

---

## 3. 작성 전 수집해야 할 정보

### 필수
1. 논문 제목
2. 논문 링크
3. 저자 / 소속
4. 출판 시점 / venue
5. 핵심 figure
6. 핵심 result table / figure

### 강력 권장
- appendix / supplementary
- project page
- code repo
- ablation table
- failure case figure
- theorem / proposition / lemma
- train-test split 설명
- runtime / parameter count / data size

### 반드시 확인
- 주요 수치 정확한가
- notation을 잘못 옮기지 않았는가
- baseline 비교 조건이 fair한가
- metric 정의가 섞이지 않았는가
- paper claim과 empirical evidence를 구분했는가

---

## 4. 파일/이미지 규칙

### 위치
- 포스트: `_posts/YYYY-MM-DD-slug.md`
- 이미지: `assets/img/posts/<slug>/`

### slug
- kebab-case
- 너무 길면 핵심 표현만 남김

### 이미지 파일명 권장
- `fig1_overview.png`
- `fig2_method.png`
- `fig3_results.png`
- `fig4_ablation.png`
- `fig5_failure_cases.png`

PDF에서 뽑은 임시 page image를 사용할 수는 있지만, 가능하면 나중에 의미 있는 figure 단위로 정리한다.

---

## 5. Front Matter 규칙

기본 템플릿:

```yaml
---
title: "논문 제목"
date: YYYY-MM-DD HH:MM:SS +0900
description: "핵심 기여 + 핵심 기술 요소 + 대표 결과를 압축한 설명"
categories: [AI, 세부카테고리]
tags: [tag1, tag2, tag3, tag4]
math: true
mermaid: true
image:
  path: /assets/img/posts/<slug>/fig1_overview.png
  alt: "대표 그림 설명"
---
```

### 카테고리
현재 저장소의 validator 기준을 따른다.
- 메인 카테고리: `AI`, `Bio`, `Dev`, `General`
- 논문 리뷰는 보통 `AI` 또는 `Bio`
- 예: `[AI, Drug Discovery]`, `[AI, Protein Structure]`

### description
- 초록 복붙 금지
- 120~180자 정도 권장
- 누가 / 무엇을 / 어떻게 / 얼마나 잘했는지 포함

### math / mermaid
- 깊은 리뷰는 기본적으로 둘 다 `true`

---

## 6. 기본 문서 구조

앞으로는 아래 구조를 기본 템플릿으로 사용한다.

```md
## Hook
## Problem
## Key Idea
## How It Works
## Results
## Discussion
## Limitations
## Conclusion
## TL;DR
## Paper Info
```

### 원칙
- `Hook`은 짧고 강하게
- `Problem`은 구조적으로
- `Key Idea`는 압축적으로
- `How It Works`는 길고 깊게
- `Results`는 해석 중심
- `Discussion`은 논문의 의미와 비교 중심
- `Limitations`는 독립 섹션으로 분리

---

## 7. 섹션별 상세 규칙

## 7.1 Hook

역할:
- 이 논문이 왜 중요한지 2~4문단 안에 납득시킨다.
- 문제 배경과 최근 흐름 속 위치를 잡아준다.

포함 요소:
- 분야의 현재 상태
- 기존 방법의 핵심 한계
- 논문의 핵심 주장
- 이 글에서 무엇을 설명할지

---

## 7.2 Problem

역할:
- 이 논문이 푸는 문제를 정확히 정리한다.
- 기존 방법이 실패하는 이유를 **구조적으로** 설명한다.

작성법:
- 2~4개의 병목으로 나눠라
- 단순 성능 부족이 아니라 원인을 써라
  - representation 문제
  - complexity 문제
  - inductive bias 문제
  - objective mismatch
  - train-test leakage
  - physical validity

---

## 7.3 Key Idea

역할:
- 논문의 핵심 기여를 가장 압축적으로 정리한다.

포함 권장:
- 한 문장 요약
- 핵심 기여 bullet 3~4개
- baseline 대비 차이 표 1개

주의:
- 여기서는 큰 그림 위주
- 너무 많은 세부 구현은 `How It Works`로 넘긴다

---

## 7.4 How It Works

가장 중요하다.

이 섹션은 앞으로 **깊은 리뷰의 중심**이다.

### 반드시 포함할 것
- 전체 파이프라인
- 문제의 수학적 공간/표현
- 핵심 모듈
- training objective
- inference / sampling / decoding / ranking
- 최소 1개 이상의 코드 블록
- 가능하면 mermaid 1개 이상
- 중요한 수식 3개 이상

### 권장 소구성

```md
### Overview
### Representation / Problem Formulation
### Core Mathematical Setup
### Core Architecture
### Training Objective
### Inference / Sampling
### Why this works
```

### 수식 규칙
수식은 넣는 것 자체가 목적이 아니다. 다음 4단계를 지켜라.

1. **가능하면 짧은 문장으로 먼저 의미를 설명**
2. **수식을 별도 블록으로 제시**
3. **각 항이 무엇인지 설명**
4. **왜 이 수식이 논문의 핵심인지 해설**

즉, 수식 뒤에는 반드시 자연어 해설이 따라야 한다.

추가 원칙:
- 복잡한 식을 억지로 inline math로 넣지 말 것
- 렌더링이 불안정하면 식을 더 단순한 형태로 다시 쓰고, 의미는 본문에서 보충할 것
- 엄밀한 원문 notation보다 **블로그에서 안정적으로 읽히는 표기**를 우선할 수 있다

### theorem / lemma / proposition 처리 규칙
논문에서 theorem/lemma가 핵심이면 반드시:
- 원문을 짧게 요약하고
- 직관을 설명하고
- 글 전체 논리에서 왜 중요한지 말한다

특히 다음 질문에 답해야 한다.
- 이 정리가 무엇을 보장하는가?
- 실제 모델 설계와 어떤 연결이 있는가?
- 단순 기술적 사실인가, 핵심 정당화인가?

### 코드 블록 규칙
- conceptual pseudocode 또는 Python/PyTorch 스타일 모두 가능
- 독자가 구현 흐름을 느낄 수 있을 만큼 구체적이어야 한다
- 논문에 없는 세부는 추정임을 드러내야 한다

### 깊이 기준
좋은 `How It Works`는 아래를 만족한다.
- 표면 설명이 아니라 설계 이유가 나온다
- notation과 구현 감각이 연결된다
- baseline과의 차이가 자연스럽게 드러난다

---

## 7.5 Results

역할:
- 이 모델이 실제로 얼마나 잘 되는지 보여준다.

포함 권장:
- main benchmark
- ablation
- generalization / OOD 결과
- runtime / efficiency
- failure case 또는 subset analysis

### 작성 원칙
- 숫자 나열 금지
- 반드시 해석을 붙인다
- 조건 차이가 있으면 명시한다
  - best@k
  - PB-valid 포함/미포함
  - leakage 차이
  - train split 차이

---

## 7.6 Discussion

역할:
- 논문의 의미를 해석하고 관련 방법들과 연결한다.

포함 권장:
- 왜 이 접근이 먹혔는가
- 다른 최근 논문과 어떤 축에서 다른가
- 실제 연구/실무 의미
- representation 선택의 장단점
- 저자 주장 중 어디까지 강하게 받아들여야 하는가

이 섹션에서는 **내 해석**이 들어가도 된다. 다만 사실과 의견을 섞지 말고 구분한다.

---

## 7.7 Limitations

역할:
- 한계를 분명히 적는다.

가능하면 아래를 체크:
- evaluation setting이 제한적인가
- data leakage 가능성이 있는가
- runtime/compute 부담이 큰가
- cofactor / multimodal / induced fit 등 어려운 설정을 빠뜨렸는가
- baseline 비교가 완전 공정한가
- open-source / reproducibility 상태는 어떤가

---

## 7.8 Conclusion

역할:
- 논문을 한 문단으로 기억하게 만든다.

구성:
- 가장 중요한 기여
- 가장 중요한 기술 포인트
- 가장 중요한 caveat

짧고 선명하게 쓴다.

---

## 7.9 TL;DR

- bullet 3~6개
- 가장 중요한 수치 1개 이상 포함
- 이 글 전체의 압축판이어야 함

---

## 7.10 Paper Info

최소 필드:

```md
## Paper Info

| 항목 | 내용 |
|---|---|
| **Title** | ... |
| **Authors** | ... |
| **Affiliations** | ... |
| **Venue** | ... |
| **Published** | ... |
| **Link** | ... |
| **Paper** | ... |
| **Code** | ... |
```

---

## 8. 문체 규칙

### 해야 할 것
- 한국어 중심, 용어는 필요시 영어 유지
- 직관과 해석을 같이 제공
- 강조는 정량과 구조에 사용
- 비교는 구체적으로
- 단정이 어려우면 caveat 명시

### 피해야 할 것
- 추상적 칭찬 반복
- 초록 재서술
- 수식 던지기만 하고 해설 없음
- 구현과 무관한 장식적 비유 과다

좋은 문장 예:
- “핵심은 torsion을 안 쓴 것이 아니라, torsion의 기하학적 복잡도를 fragment pose로 다시 썼다는 데 있다.”
- “이 theorem은 단순 형식적 보장이 아니라, 왜 fragment space가 학습하기 쉬운지를 정당화하는 축이다.”

---

## 9. 작성 절차

### Step 1. 논문 읽기
- 초록만 말고 method / results / appendix까지 본다
- 핵심 claim 1문장, 핵심 기여 3개, 핵심 식 3개를 먼저 뽑는다

### Step 2. 구조 잡기
- Hook
- Problem
- Key Idea
- How It Works 세부 소제목
- Results
- Discussion
- Limitations
- Conclusion
- TL;DR
- Paper Info

### Step 3. 수학적 중심축 정리
다음 중 무엇이 핵심인지 먼저 정한다.
- state space
- objective
- theorem
- inductive bias
- inference algorithm
- ranking / confidence

### Step 4. 본문 작성
우선순위:
1. Hook / Problem / Key Idea
2. How It Works 확장
3. Results 해설
4. Discussion / Limitations

### Step 5. polish
- 첫 3문단 sharpen
- 수식 뒤 해설 보강
- TL;DR 강화
- 내부 링크 추가

---

## 10. 품질 체크리스트

### 최소 기준
- [ ] front matter 완비
- [ ] `How It Works`가 글의 중심
- [ ] 수식 3개 이상 또는 그에 준하는 수학적 설명
- [ ] 코드 블록 1개 이상
- [ ] 결과 표/수치 포함
- [ ] `Limitations` 존재
- [ ] `Paper Info` 존재
- [ ] 마지막 고정 블록 존재

### 깊은 리뷰 기준
- [ ] 핵심 theorem/lemma/설계 이유가 설명됨
- [ ] notation과 직관이 연결됨
- [ ] 결과 해석이 비판적임
- [ ] baseline 대비 차이가 명확함
- [ ] 구현 감각이 생김

---

## 11. 마지막 고정 블록

```md
---

> 이 글은 LLM(Large Language Model)의 도움을 받아 작성되었습니다. 
> 논문의 내용을 기반으로 작성되었으나, 부정확한 내용이 있을 수 있습니다.
> 오류 지적이나 피드백은 언제든 환영합니다.
{: .prompt-info }
```

---

## 12. 새 포스트 템플릿

```md
---
title: "<PAPER TITLE>"
date: YYYY-MM-DD HH:MM:SS +0900
description: "<핵심 기여 + 핵심 기술 + 대표 결과>"
categories: [AI, <SUBCATEGORY>]
tags: [<tag1>, <tag2>, <tag3>, <tag4>]
math: true
mermaid: true
image:
  path: /assets/img/posts/<slug>/fig1_overview.png
  alt: "<대표 그림 설명>"
---

## Hook

## Problem

## Key Idea

## How It Works

### Overview

### Representation / Problem Formulation

### Core Mathematical Setup

### Core Architecture

### Training Objective

### Inference / Sampling

## Results

## Discussion

## Limitations

## Conclusion

## TL;DR

## Paper Info

| 항목 | 내용 |
|---|---|
| **Title** | ... |
| **Authors** | ... |
| **Affiliations** | ... |
| **Venue** | ... |
| **Published** | ... |
| **Link** | ... |
| **Paper** | ... |
| **Code** | ... |

---

> 이 글은 LLM(Large Language Model)의 도움을 받아 작성되었습니다. 
> 논문의 내용을 기반으로 작성되었으나, 부정확한 내용이 있을 수 있습니다.
> 오류 지적이나 피드백은 언제든 환영합니다.
{: .prompt-info }
```

---

## 13. 최종 요약

앞으로 이 블로그의 논문 리뷰는 **짧은 요약형이 아니라, 수식/구조/구현 감각/한계 해설이 포함된 깊은 long-form technical review**를 기본으로 한다.
