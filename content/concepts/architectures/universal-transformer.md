---
title: Universal Transformer
tags:
  - concepts
  - architectures
  - transformer
---

# Universal Transformer

Universal Transformer는 Transformer transition block을 computation depth 방향으로 반복 적용하는 architecture입니다. 일반 Transformer가 서로 다른 $L$개 layer를 쌓는다면, Universal Transformer는 하나의 transition $\mathcal{T}_\theta$를 여러 step에 공유합니다.

$$
H^{(t+1)}=\mathcal{T}_\theta(H^{(t)},t)
$$

여기서 recurrence는 sequence position을 순서대로 처리하는 RNN recurrence가 아니라, 같은 token representation을 여러 번 정제하는 **depth recurrence**입니다.

## 핵심 구분

| 축 | 질문 | Universal Transformer의 답 |
| --- | --- | --- |
| Token interaction | 어떤 위치가 통신하는가? | self-attention으로 전체 위치가 통신 |
| Computation depth | 몇 번 정제하는가? | shared transition을 여러 번 적용 |
| Parameterization | step마다 다른가? | transition parameter를 공유 |
| Compute allocation | 모든 위치가 같은가? | adaptive halting으로 다르게 가능 |

## Adaptive halting

각 위치 $i$에 대해 halting probability를 계산할 수 있습니다.

$$
p_i^{(t)}=\sigma(W_hH_i^{(t)}+b_h)
$$

누적 확률이 threshold를 넘으면 해당 위치의 추가 계산을 멈춥니다.

$$
T_i\in\{1,\ldots,T_{\max}\}
$$

평균 계산량을 줄이려면 알고리즘상의 halt가 실제 kernel 실행에서도 계산량 감소로 이어지는지 확인해야 합니다. 단순히 halting step을 기록하는 것만으로 latency 개선을 주장할 수 없습니다.

## Complexity

Dense self-attention을 사용하면 recurrent step 수를 $T$, sequence length를 $n$, hidden width를 $d$라 할 때 대략적인 attention 비용은 다음과 같습니다.

$$
O(Tn^2d)
$$

Parameter count는 transition을 공유하므로 고정 depth를 늘리는 것처럼 선형 증가하지 않지만, inference latency와 activation memory는 step 수에 따라 증가합니다.

## 언제 유용한가

- 여러 단계의 iterative refinement가 필요한 문제
- 입력 길이 또는 난이도에 따라 계산량을 배분하고 싶은 경우
- Transformer의 global receptive field와 recurrent inductive bias를 함께 사용하고 싶은 경우

## 주의점

- shared weights가 layer별 specialization을 제한할 수 있습니다.
- adaptive halting은 task loss와 compute penalty의 trade-off를 추가합니다.
- 고정 depth baseline과 parameter, training budget, inference compute를 맞춰 비교해야 합니다.
- position별 halting이 hardware에서 실제 sparse execution이 되는지 별도로 측정해야 합니다.

## Related

- [[papers/architectures/universal-transformers|Universal Transformers]]
- [[concepts/architectures/transformer|Transformer]]
- [[concepts/architectures/parameter-sharing|Parameter sharing]]
- [[concepts/architectures/computational-complexity|Computational complexity]]
