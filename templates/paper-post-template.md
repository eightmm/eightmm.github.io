---
title: "{{TITLE}}"
date: {{DATE}}
description: "{{DESCRIPTION}}"
categories: [{{MAIN_CATEGORY}}, {{SUBCATEGORY}}]
tags: [{{TAGS}}]
math: true
mermaid: false
image:
  path: /assets/img/posts/{{SLUG}}/fig1_overview.png
  alt: "{{IMAGE_ALT}}"
---

## Hook

이 논문이 왜 지금 중요한지, 그리고 기존 접근의 어디를 건드리는지부터 짧고 선명하게 설명한다.

## Problem

이 논문이 푸는 문제를 2~4개의 구조적 병목으로 정리한다.

### 병목 1

### 병목 2

## Key Idea

핵심 기여를 가장 압축적으로 설명한다.

- 무엇이 바뀌었는가
- 왜 그 변화가 중요한가
- 기존 패러다임과 무엇이 다른가

## How It Works

### Overview

대표 figure를 넣고 전체 흐름을 설명한다.

### Representation / Formulation

핵심 표현, 목적함수, 확률적 해석을 정리한다.

$$
\text{Put one important equation here.}
$$

### Architecture or Algorithm

모듈 흐름을 입력 → 표현 → 업데이트 → 출력 순서로 설명한다.

```python
# minimal implementation sketch
import torch
import torch.nn as nn
```

### Why this works

설계 직관을 설명한다.

## Results

- 핵심 수치
- 비교 조건
- 왜 이 결과가 의미 있는지

## Discussion

이 논문이 분야 안에서 어떤 의미를 갖는지 해석한다.

## Limitations

- empirical scope
- compute cost
- generalization limits
- theory gaps

## Conclusion

논문의 실제 기여를 과장 없이 요약한다.

## TL;DR

- 핵심 bullet 1
- 핵심 bullet 2
- 핵심 bullet 3

## Paper Info

- **Title:** {{PAPER_TITLE}}
- **Authors:** {{AUTHORS}}
- **Affiliations:** {{AFFILIATIONS}}
- **Venue:** {{VENUE}}
- **Published:** {{PUBLISHED}}
- **Paper:** {{PAPER_URL}}
- **Project:** {{PROJECT_URL}}
- **Code:** {{CODE_URL}}
