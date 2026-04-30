---
title: "AlphaFold 2: protein folding problem을 푼 건 distance가 아니라 representation이었다"
date: 2026-02-20 12:00:00 +0900
description: "AlphaFold 2는 distogram 기반 파이프라인을 넘어 MSA와 pair representation을 함께 진화시키는 Evoformer, 그리고 Invariant Point Attention 기반 structure module로 end-to-end 단백질 구조 예측을 실현했다. CASP14에서 실험 구조에 근접한 정확도를 보인 핵심은 더 많은 규칙이 아니라 더 나은 표현과 좌표 생성 방식이었다."
categories: [AI, Protein Structure]
tags: [protein-structure, alphafold2, evoformer, invariant-point-attention, casp14, structure-prediction]
math: true
mermaid: false
image:
  path: /assets/img/posts/alphafold2-highly-accurate-protein-structure-prediction/fig1.png
  alt: "AlphaFold 2 architecture and prediction examples"
---

## Hook

AlphaFold 2 이전에도 단백질 구조 예측은 이미 많이 좋아지고 있었다. contact prediction은 점점 정확해졌고, distogram 같은 더 풍부한 geometric supervision도 등장했다. 그래서 겉으로 보면 AF2는 “그 흐름의 정점”처럼 보일 수 있다. 하지만 실제로 논문을 뜯어보면 느낌이 좀 다르다. AF2는 단순히 더 깊은 네트워크나 더 많은 데이터로 정확도를 밀어 올린 모델이 아니라, **구조 예측 문제를 neural representation 안에서 다시 조직한 모델**에 더 가깝다.

기존 접근의 전형적인 흐름은 대체로 이랬다. 먼저 pairwise distance나 contact를 예측하고, 그다음 별도의 구조 최적화 단계에서 3D 좌표를 복원한다. 이 두 단계는 서로 연결되어 있지만 완전히 같은 문제는 아니다. 중간 representation이 좋아도 최종 structure realization이 흔들릴 수 있고, 반대로 후처리를 아무리 잘해도 upstream signal이 약하면 한계가 있다.

**AlphaFold 2**는 이 분리를 크게 줄인다. MSA에서 얻은 진화정보와 residue pair 관계를 Evoformer 안에서 반복적으로 같이 업데이트하고, 마지막엔 Invariant Point Attention을 통해 residue frame 위에서 직접 3D 구조를 만들어간다. 이 구조는 “distance를 예측한 뒤 구조를 끼워 맞춘다”기보다, **서열과 진화정보가 구조로 접히는 계산 자체를 모델 안에 넣었다**는 쪽에 더 가깝다.

내가 보기엔 AF2의 진짜 혁신은 CASP14 숫자만이 아니다. 더 중요한 건, 단백질 구조 예측의 핵심 병목이 물리 시뮬레이션 부족이나 contact accuracy 부족만이 아니라 **representation과 coordinate generation 사이의 단절**이었다는 걸 보여준 점이다. 이 글에서는 AF2가 정확히 무엇을 바꿨는지, 왜 Evoformer와 IPA가 그렇게 중요했는지, 그리고 AF3로 가는 길이 사실 어디서부터 시작됐는지 정리해보겠다.

## Problem

AlphaFold 2가 푼 문제는 단순히 “contact prediction을 더 잘하자”가 아니다. 더 정확히는 다음 질문이다.

> **서열과 진화정보로부터 단백질의 3D 구조를 end-to-end로 예측하려면, 어떤 representation과 update rule이 필요한가?**

이 질문은 크게 네 가지 병목으로 나뉜다.

### 병목 1: contact나 distance만으로는 구조 문제가 끝나지 않는다

AlphaFold 1 세대의 큰 진전은 binary contact 대신 distogram 같은 richer geometric target을 예측했다는 점이었다. 하지만 여전히 구조 파이프라인은 두 단계였다.

- pairwise geometric signal을 예측하고
- 그걸 바탕으로 별도 최적화나 relaxation으로 3D 구조를 만든다

이 과정에는 자연스러운 정보 손실이 있다. pairwise constraints가 충분히 좋아 보여도, 실제 3D embedding은 여전히 어렵다. 특히 장거리 상호작용, 삼체 관계, global consistency는 단순 거리 행렬만으로는 깔끔하게 해결되지 않는다.

### 병목 2: MSA 정보와 pair geometry가 따로 놀기 쉽다

진화정보는 구조 예측에 강력하지만, 많은 기존 접근은 MSA에서 feature를 뽑은 뒤 그걸 정적인 입력처럼 사용했다. 문제는 구조 예측에서 중요한 신호가 MSA 내부에만 있지 않다는 점이다.

- 어떤 residue pair가 함께 변하는가
- 그 pair가 다른 residue들과 어떤 삼각 관계를 이루는가
- sequence-level signal이 pair geometry와 어떻게 연결되는가

즉, 좋은 구조 예측기는 **MSA representation과 pair representation이 서로 영향을 주고받으며 함께 성숙**해야 한다.

### 병목 3: 3D reasoning이 network 안에서 invariant해야 한다

단백질 구조는 좌표로 표현되지만, 실제 중요한 것은 절대 좌표계가 아니다. 분자를 통째로 회전하거나 평행이동해도 같은 구조여야 한다. 따라서 structure module은 SE(3) transformation에 대해 자연스럽게 작동해야 한다.

기존 구조 예측 파이프라인은 이런 부분을 후처리나 별도 energy minimization에 많이 의존했다. AF2는 이걸 모델 내부 attention 메커니즘으로 끌어들인다.

### 병목 4: local stereochemistry와 global fold를 함께 맞춰야 한다

좋은 구조 예측은 두 가지를 동시에 해야 한다.

- 전역적으로 맞는 fold topology
- 국소적으로 말이 되는 backbone / side-chain geometry

이 둘을 따로 풀면 엇갈리기 쉽다. global fold는 맞는데 local orientation이 깨지거나, local geometry는 괜찮은데 전체 topology가 틀릴 수 있다. AF2의 핵심은 이 두 수준을 **하나의 반복적 structure refinement 과정**으로 묶는 데 있다.

## Key Idea

AlphaFold 2의 핵심은 세 가지로 압축된다.

1. **MSA와 pair representation을 Evoformer 안에서 함께 진화시킨다.**
2. **Invariant Point Attention을 이용해 residue frame 위에서 직접 3D 구조를 갱신한다.**
3. **intermediate geometry prediction과 final structure realization의 단절을 크게 줄인다.**

AF1과 비교하면 차이는 이렇게 요약할 수 있다.

- AF1
  - distogram prediction이 중심
  - 구조 복원은 별도 최적화 문제에 가깝다
  - pairwise geometry는 강하지만 end-to-end성은 제한적이다
- AF2
  - MSA/pair 공동 업데이트
  - 3D structure module이 모델 안에 들어온다
  - structure refinement가 end-to-end로 연결된다

내가 보기에 AF2를 특별하게 만든 건 “정확한 distance를 예측했다”보다, **구조를 직접 만드는 representation learning stack**을 설계했다는 점이다.

## How It Works

### Overview

![AlphaFold 2 overview](/assets/img/posts/alphafold2-highly-accurate-protein-structure-prediction/fig1.png)
_Figure 1: AlphaFold 2 architecture and prediction examples. Source: original paper._

전체 흐름은 크게 두 단계다.

- **Evoformer trunk**가 MSA representation과 pair representation을 반복적으로 갱신한다.
- **Structure module**이 query sequence에 대한 single representation과 pair geometry를 바탕으로 3D 좌표를 만든다.

아주 단순화하면 이런 형태다.

```python
# conceptual pseudocode
msa_repr = embed_msa(sequence, msa)
pair_repr = embed_pair(sequence, templates)
for _ in range(num_evoformer_blocks):
    msa_repr, pair_repr = evoformer_block(msa_repr, pair_repr)
single_repr = msa_repr[0]
frames = structure_module(single_repr, pair_repr)
coords = build_atom_coordinates(frames)
confidence = predict_plddt(single_repr)
return coords, confidence
```

여기서 중요한 건 마지막 `coords`가 아니라, 그 앞에서 **MSA와 pair가 어떻게 서로를 계속 수정하느냐**다. AF2는 단백질 구조를 “한 번 계산한 feature를 좌표로 decode”하는 문제가 아니라, **representation 자체가 구조적 일관성을 띠도록 만드는 문제**로 본다.

### Representation: MSA와 pair를 동시에 키운다

AF2의 기본 표현은 세 가지다.

- **MSA representation**: homologous sequence들의 정렬 정보
- **pair representation**: residue i와 j 사이 관계
- **single representation**: query sequence residue별 상태

여기서 single은 사실상 MSA에서 query row를 뽑아 쓰는 쪽에 가깝고, 핵심 주연은 MSA와 pair다.

왜 이 이중 표현이 중요하냐면,

- MSA는 coevolution signal을 준다.
- pair는 구조적 관계를 누적한다.
- 둘이 서로 독립이면 정보가 금방 막힌다.

AF2는 이 둘을 매 layer에서 이어준다. 구조적으로 보면 “sequence-derived evidence”와 “geometry hypothesis”가 네트워크 내부에서 계속 대화하는 셈이다.

### Evoformer: AF2의 진짜 엔진

![Evoformer and structure module](/assets/img/posts/alphafold2-highly-accurate-protein-structure-prediction/fig3.png)
_Figure 2: Evoformer block and Structure Module details. Source: original paper._

Evoformer는 AF2의 가장 중요한 발명 중 하나다. 이름만 보면 Transformer 계열처럼 보이지만, 실제 역할은 더 특수하다. 이건 sequence attention 블록이 아니라 **MSA와 pair representation 사이 정보 흐름을 조직하는 구조 추론 엔진**이다.

대표적인 구성 요소는 아래와 같다.

- MSA row attention with pair bias
- MSA column attention
- outer product mean
- triangle multiplicative update
- triangle attention
- pair transition / MSA transition

특히 중요한 건 pair 쪽 업데이트다.

#### Triangle update의 의미

단백질 구조는 residue 간 관계가 독립적이지 않다. 예를 들어 residue i가 k와 가깝고, k가 j와 가깝다면, i와 j의 관계도 아무렇게나 정해질 수 없다. AF2의 triangle multiplicative update와 triangle attention은 이런 **삼체 일관성**을 pair representation 안에 주입한다.

개념적 스케치는 이 정도다.

```python
import torch
import torch.nn as nn

class EvoformerPairUpdate(nn.Module):
    def __init__(self, pair_dim: int):
        super().__init__()
        self.linear_a = nn.Linear(pair_dim, pair_dim)
        self.linear_b = nn.Linear(pair_dim, pair_dim)

    def forward(self, pair_repr):
        a = self.linear_a(pair_repr)
        b = self.linear_b(pair_repr)
        triangle_signal = torch.einsum('ikc,kjc->ijc', a, b)
        return pair_repr + triangle_signal
```

실제 구현은 gating, normalization, orientation 분리 등 훨씬 더 정교하지만, 핵심 아이디어는 단순하다. **edge (i, j)를 직접만 보지 말고, 중간 node k를 경유한 relational consistency를 학습하자**는 것이다.

#### MSA와 pair의 왕복

AF2는 단순히 MSA에서 pair를 한 번 만들고 끝내지 않는다.

- MSA는 pair bias를 받아 attention한다.
- MSA 정보는 outer product를 통해 pair를 갱신한다.
- pair는 triangle updates로 구조적 일관성을 얻는다.
- 다시 그 pair가 MSA 처리에 영향을 준다.

이 왕복이 48개 블록 수준으로 반복되면서, 초기엔 noisy하던 진화/기하 정보가 점점 구조적으로 응축된다. 이게 AF2가 단순 distogram predictor와 다른 지점이다.

### Structure Module: 3D reasoning을 모델 내부로 넣는다

Evoformer가 구조적 hypothesis를 만들었다면, Structure Module은 그걸 실제 3D 좌표로 바꾸는 단계다. 여기서 핵심은 **Invariant Point Attention (IPA)** 이다.

AF2는 각 residue를 rigid frame으로 다룬다. residue마다 local frame을 갖고, attention이 scalar feature뿐 아니라 **3D point** 수준에서도 작동한다. 중요한 건 이 attention이 global rotation / translation에 대해 invariant하게 설계된다는 점이다.

직관적으로 보면 IPA는 이렇게 작동한다.

- 각 residue가 local frame에서 query/key/value point를 만든다.
- 이를 global frame으로 보낸다.
- point 간 거리와 orientation 정보를 이용해 attention을 계산한다.
- 다시 local frame 기준으로 residue state와 frame을 업데이트한다.

이 과정을 통해 structure module은 단순 feature refinement가 아니라, **geometry-aware message passing**을 수행한다.

구조적으로는 residue frame이 반복적으로 갱신되며 backbone과 side-chain을 더 일관된 방향으로 접어 간다.

### FAPE: 왜 이 loss가 중요했나

AF2의 loss에서 중요한 축 중 하나는 **Frame Aligned Point Error (FAPE)** 다. 이 loss는 예측 좌표를 각 residue의 local frame 기준으로 비교한다.

개념적으로는 이런 감각이다.

$$
\mathrm{FAPE} = \frac{1}{NK} \sum_{k,i} \left\| T_k^{-1}(x_i) - {T_k^{\ast}}^{-1}(x_i^{\ast}) \right\|
$$

여기서 $T_k$는 residue k의 frame이다. 중요한 건 global alignment가 아니라 **local frame 기준 오차**를 본다는 점이다. 이 덕분에 residue orientation, side-chain 배치, chirality 같은 구조적 요소를 더 직접적으로 학습할 수 있다.

즉 FAPE는 단순 오차 함수가 아니라, AF2가 structure module을 제대로 학습하게 만드는 표현 선택의 일부다.

### Recycling: 한 번 계산하고 끝내지 않는다

AF2는 한 번 예측한 구조와 representation을 다시 입력처럼 써서 refine하는 **recycling**도 핵심이다. 이건 아주 직관적인 아이디어지만 효과가 크다.

- 첫 패스에서 거친 구조 가설을 만든다.
- 그 구조와 pair state를 다시 참고한다.
- 다음 패스에서 더 정교한 consistency를 맞춘다.

이건 iterative refinement를 모델 외부 optimizer가 아니라 **모델 자체의 recurrent use**로 흡수한 셈이다. AF2가 구조 예측을 “한 번 forward하고 끝”이 아니라, learned iterative inference에 가깝게 만든 포인트다.

### 왜 이 설계가 먹혔나

AF2의 설계를 한 문장으로 요약하면 이렇다.

> **진화정보, pair geometry, 3D reasoning을 분리된 단계로 두지 않고 하나의 representation stack 안에서 통합했다.**

더 풀면,

- Evoformer는 sequence-level evidence와 pair geometry를 공동 업데이트한다.
- Triangle updates는 pair relation에 3D consistency를 주입한다.
- IPA는 structure reasoning을 좌표계 불변한 방식으로 수행한다.
- FAPE와 recycling은 refinement를 학습 가능한 형태로 만든다.

그래서 AF2는 단순히 더 좋은 contact predictor가 아니라, **단백질 구조 예측을 end-to-end geometric learning 문제로 재정의한 모델**이 된다.

## Results

![AlphaFold 2 results](/assets/img/posts/alphafold2-highly-accurate-protein-structure-prediction/fig4.png)
_Figure 3: AlphaFold 2 benchmark performance and CASP14-level accuracy. Source: original paper._

AF2의 결과는 이제 너무 유명해서 오히려 쉽게 뭉개어 말하게 된다. 하지만 핵심은 “좋았다”가 아니라 **어디서 얼마나 질적으로 선을 넘었는가**다.

### 1. CASP14에서 사실상 게임의 규칙을 바꿨다

AF2는 CASP14에서 기존 참가자들을 크게 앞서는 정확도를 보였다. 이건 incremental improvement가 아니라, community가 오랫동안 benchmark ceiling처럼 보던 구간을 한 번에 넘어선 사건에 가까웠다.

특히 중요한 건 많은 target에서 예측 구조가 실험 구조에 매우 가깝고, 실무적으로도 바로 쓸 수 있을 정도의 품질을 보였다는 점이다.

### 2. 단순 fold classification을 넘어 atomic accuracy에 근접했다

과거엔 “대략 맞는 topology”만 나와도 성공으로 여겨지는 경우가 많았다. AF2는 여기서 한 단계 더 나간다. backbone 수준만이 아니라, 많은 경우 residue placement와 overall packing이 실제로 usable한 수준까지 올라온다.

이게 중요했던 이유는 구조 예측이 더 이상 speculative modeling이 아니라, **생물학적 가설 생성과 구조 기반 해석의 실질적 입력**이 되기 시작했기 때문이다.

### 3. end-to-end 구조 생성이 실제로 먹혔다

AF2의 성과는 single component가 아니라 시스템 설계 전체가 맞물렸을 때 나온 결과다. 단순히 더 나은 MSA 활용이나 더 나은 pair update 하나만으로는 이 정도 점프가 어렵다. 결과적으로 이 논문은 representation, geometry, refinement를 통합하는 방식이 구조 예측에서 얼마나 강력한지 보여줬다.

## Discussion

내가 보기엔 AF2의 진짜 의의는 “protein folding problem solved” 같은 상징적 문장보다도, **어떤 종류의 inductive bias가 구조 예측에 가장 잘 먹히는가**를 보여줬다는 데 있다.

과거 접근은 대개 두 축 중 하나에 더 기대는 편이었다.

- 물리 기반 sampling을 많이 하거나
- coevolution signal을 중간 예측 타깃으로 바꾸어 쓰거나

AF2는 그 둘을 representation inside에 녹였다. 명시적 molecular dynamics를 돌리진 않지만, geometry-aware attention과 local frame refinement를 통해 구조적 제약을 학습한다. hand-crafted contact pipeline을 유지하지 않지만, pair representation 안에서 관계를 반복적으로 정제한다.

이건 이후 구조 생물학 모델들에도 큰 영향을 줬다. 실제로 AF3까지 이어지는 흐름도 AF2가 만든 이 교훈 위에 서 있다.

- rich pair representation이 중요하다
- 3D reasoning은 model 내부에 있어야 한다
- coordinate generation은 별도 후처리가 아니라 학습 대상이어야 한다

AF3가 diffusion으로 갔더라도, **구조 예측을 geometric generative problem으로 보는 관점 자체는 AF2에서 이미 강하게 시작**된 셈이다.

## Limitations

AF2가 아무리 강해도 한계는 분명하다.

### 1. 본질적으로 단백질 중심 모델이다

AF2는 단백질 folding에 최적화된 설계다. residue frame, torsion angle, MSA 활용 방식 모두 protein-centric하다. 그래서 리간드, 핵산, 이온까지 자연스럽게 통합하는 데는 구조적 한계가 있었다. AF3가 등장한 이유도 여기 있다.

### 2. static structure accuracy와 full biophysics는 다르다

AF2가 정적 구조를 잘 맞힌다고 해서 folding kinetics, conformational ensemble, allostery, disorder, binding thermodynamics를 다 해결하는 건 아니다. 구조 예측은 매우 큰 진전이지만, 단백질 동역학 전체와 동일시하면 안 된다.

### 3. MSA 의존성은 여전히 중요하다

AF2는 shallow MSA나 difficult target에서도 인상적이었지만, 전반적으로 진화정보를 강하게 활용하는 모델이다. 따라서 homolog scarcity가 심한 경우나 sequence space coverage가 낮은 영역에서는 여전히 어려움이 남는다.

### 4. confidence와 correctness는 다를 수 있다

pLDDT와 관련 confidence score는 매우 유용하지만, 특히 flexible region이나 multi-state system에서는 confidence가 biological correctness를 완전히 대변하진 않는다.

## Conclusion

AlphaFold 2는 단백질 구조 예측을 더 잘한 모델이라기보다, **구조 예측 문제의 계산 그래프 자체를 다시 설계한 모델**이다. MSA와 pair representation을 Evoformer 안에서 함께 진화시키고, Invariant Point Attention으로 residue frame 위에서 직접 구조를 만들며, FAPE와 recycling으로 learned refinement를 수행한다. 이 조합 덕분에 AF2는 “distance를 예측한 뒤 구조를 맞춘다”는 패턴을 넘어서, **representation이 스스로 구조를 접어 나가게 만드는 모델**이 되었다.

내 기준에서 AF2의 가장 큰 의미는 CASP14 점수보다도, 구조 생물학에서 neural network가 어디까지 직접 geometric reasoning을 해도 되는지 보여준 데 있다. 그리고 그 교훈은 AF3에서도 이어진다. AF3가 더 범용적이고 diffusion 기반이긴 해도, **구조를 end-to-end geometric generation으로 다루는 길**은 사실 AF2가 결정적으로 연 셈이다.

## TL;DR

- AlphaFold 2의 핵심은 더 좋은 distogram이 아니라 **MSA와 pair representation을 함께 진화시키는 Evoformer**다.
- Structure Module은 **Invariant Point Attention**으로 residue frame 위에서 직접 3D 구조를 갱신한다.
- 이 설계는 intermediate geometry prediction과 final coordinate generation 사이의 단절을 크게 줄였다.
- Triangle updates는 pair relation에 삼체 일관성을 주입하고, FAPE는 local frame 기준으로 구조 오차를 학습하게 만든다.
- Recycling까지 포함해 AF2는 구조 예측을 learned iterative refinement 문제로 바꿨다.
- 결과적으로 AF2는 CASP14에서 실험 구조에 근접한 정확도를 보이며, 단백질 구조 예측의 기준점을 바꿨다.
- 다만 AF2는 여전히 protein-centric 구조를 갖고 있어, heterogeneous biomolecular interaction 문제로 가려면 AF3 같은 재설계가 필요했다.

## Paper Info

- **Title:** Highly Accurate Protein Structure Prediction with AlphaFold
- **Authors:** Jumper et al.
- **Affiliations:** DeepMind and collaborators
- **Venue:** Nature
- **Published:** 2021-07-15
- **Paper:** https://www.nature.com/articles/s41586-021-03819-2

---

> 이 글은 LLM(Large Language Model)의 도움을 받아 작성되었습니다.
> 논문의 내용을 기반으로 작성되었으나, 부정확한 내용이 있을 수 있습니다.
> 오류 지적이나 피드백은 언제든 환영합니다.
{: .prompt-info }
