---
title: Adaptive Computation
tags:
  - concepts
  - architectures
  - efficiency
---

# Adaptive Computation

Adaptive computation은 모든 입력에 같은 depth, iteration, sampling step을 적용하지 않고, 입력의 난이도나 상태에 따라 계산량을 조절하는 설계입니다.

고정 계산 모델은 다음과 같습니다.

$$
y=f_\theta^{(T)}(x)
$$

여기서 모든 입력이 같은 $T$개의 transition을 거칩니다. Adaptive computation에서는 입력별 계산량을 $T(x)$로 둡니다.

$$
y=f_\theta^{(T(x))}(x)
$$

## 주요 구현 방식

| 방식 | 계산량을 정하는 신호 | 대표 trade-off |
| --- | --- | --- |
| learned halting | halt probability 또는 stop head | accuracy와 ponder cost |
| early exit | 중간 classifier confidence | calibration과 latency |
| dynamic depth | 입력별 layer/step 선택 | routing overhead와 hardware 효율 |
| adaptive sampling | uncertainty 또는 solver tolerance | sample quality와 compute |
| conditional routing | token별 expert 선택 | load balance와 capacity |

Universal Transformer에서는 각 위치에 대해 halting step을 예측합니다. 일반적인 목적함수는 task loss에 계산 penalty를 더한 형태입니다.

$$
\mathcal{L}
=
\mathcal{L}_{\mathrm{task}}
+
\lambda_{\mathrm{compute}}C(x)
$$

여기서 $C(x)$는 transition 수, executed FLOPs, latency proxy 중 하나입니다. 어떤 proxy를 쓰는지 명시하지 않으면 “adaptive”라는 말만으로 실제 속도 향상을 보장할 수 없습니다.

## 평가 계약

Adaptive computation claim은 최소한 다음을 분리해야 합니다.

1. 동일한 품질에서 평균 계산량이 줄었는가?
2. 동일한 계산량에서 품질이 좋아졌는가?
3. 평균 계산량 감소가 실제 wall-clock latency로 나타나는가?
4. 쉬운 입력만 빨리 종료하고 어려운 입력에는 충분한 계산을 사용하는가?

평균 step 수만 보고하는 것은 부족합니다. step 분포, tail latency, batch padding, kernel utilization, 최대 계산량도 함께 봐야 합니다.

## Related

- [[papers/architectures/universal-transformers|Universal Transformers]]
- [[concepts/architectures/computational-complexity|Computational complexity]]
- [[concepts/systems/latency-throughput|Throughput and latency]]
- [[concepts/architectures/mixture-of-experts|Mixture of experts]]
