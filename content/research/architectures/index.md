---
title: Architecture Research
tags:
  - research
  - architectures
---

# Architecture Research

Architecture research는 model structure와 inductive bias가 어떤 task에서 실제로 도움이 되는지 묻는 공간입니다. 여기서는 특정 benchmark 점수보다 입력 구조, computational cost, evaluation boundary를 함께 봅니다.

현재 문서는 architecture topic을 확정 연구 주제로 선언하지 않고, 검증 가능한 질문 후보로만 유지합니다. 실제 비교 실험이나 paper cluster가 생기면 evidence plan을 먼저 업데이트합니다.

## 주제 후보

| 주제 | 질문 | 연결 |
| --- | --- | --- |
| [Geometric inductive bias](/research/architectures/geometric-inductive-bias) | geometry-aware architecture는 언제 Transformer보다 나은가? | [Architectures](/ai/architectures), [Geometry and symmetry](/math/geometry-symmetry) |

## 경계

- Architecture 이름만 모으지 않고, 어떤 data structure와 claim에 필요한지 적습니다.
- 구현 실험이 중심이 되면 [[projects/index|Projects]]로 넘깁니다.
- 논문 하나의 claim은 [[papers/index|Papers]]에 두고, 반복되는 질문만 이곳에 둡니다.
- 결과가 없는 상태에서는 performance claim을 쓰지 않고 hypothesis와 evaluation boundary만 둡니다.

## Related

- [[ai/architectures|Architectures]]
- [[math/geometry-symmetry|Geometry and symmetry]]
- [[concepts/architectures/inductive-bias|Inductive bias]]
