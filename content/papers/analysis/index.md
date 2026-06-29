---
title: Paper Analysis
unlisted: true
tags:
  - papers
  - methodology
  - evaluation
---

# Paper Analysis

Paper analysis note는 논문에서 claim, evidence, benchmark, ablation, limitation, comparison axis를 추출합니다.

Analysis layer는 논문 문장을 확인 가능한 구조로 바꿉니다.

$$
\text{claim}
\rightarrow
\text{evidence}
\rightarrow
\text{scope}
\rightarrow
\text{limitation}
$$

이렇게 해야 paper note가 abstract 요약만 모아둔 페이지가 되지 않습니다.

## Scope

- claim extraction과 evidence table.
- benchmark card, split interpretation, metric selection.
- component claim을 위한 ablation map.
- limitation taxonomy와 paper comparison matrix.

## 노트

- [[papers/analysis/claim-extraction|Claim extraction]]
- [[papers/analysis/evidence-table|Evidence table]]
- [[papers/analysis/result-table-reading|Result table reading]]
- [[papers/analysis/benchmark-card|Benchmark card]]
- [[papers/analysis/ablation-map|Ablation map]]
- [[papers/analysis/limitation-taxonomy|Limitation taxonomy]]
- [[papers/analysis/paper-comparison-matrix|Paper comparison matrix]]

## 확인할 것

- 중요한 claim마다 method, task, benchmark, metric, protocol, baseline이 명시되는가?
- 각 evidence item이 실제로 support하는 scope를 적는가?
- ablation을 단순 performance number가 아니라 component와 연결하는가?
- limitation을 data, split, metric, baseline, ablation, generalization, reproducibility, efficiency, domain limit로 분류하는가?
- paper 간 비교가 같은 task와 evaluation boundary 위에서 이뤄지는가?

## 새 노트 위치

- analysis template과 claim structure는 여기에 둡니다.
- reproduction planning은 [[papers/reproducibility/index|Paper reproducibility]]에 둡니다.
- 일반 evaluation concept는 [[concepts/evaluation/index|Evaluation]]에 둡니다.
- 내 연구 방법론 claim은 [[concepts/research-methodology/index|Research methodology]]에 둡니다.

## Related

- [[concepts/evaluation/index|Evaluation]]
- [[concepts/data/benchmark|Benchmark]]
- [[papers/workflows/index|Paper workflows]]
- [[papers/reproducibility/index|Paper reproducibility]]
